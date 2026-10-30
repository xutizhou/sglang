# DeepSeek-V4 Prefill CP 的 KV Cache 分层切片（Layer-Split）设计与实测

日期：2026-10-28 · 作者：Xuting ZHOU（实现与实验由 Claude 代理执行）
代码：fork `xutizhou/sglang` 分支 `feat/dsv4-prefill-cp-layer-split`（基于 `deepseek-base-optimization` @ 2c638cb7），commits 4f96a89 / 8deed7a，12 个文件 +1085/−14 行，核心文件 `python/sglang/srt/mem_cache/deepseek_v4_layer_split.py`
测试环境：MaaS DeepSeek-V4-Flash PD 分离，Prefill 4×RTX 6000D（10.6.142.24 GPU0–3，PCIe，无 NVLink），Decode 4×H20（10.6.131.28 GPU4–7，V13 recipe 的 decode 原样），AgentX 256K 轨迹回放。

---

## 1. 背景：为什么要做

AgentX 这类 agentic 负载的 prefill 大多数步只有一个请求在飞。V02–V13 的 prefill 都是 **TP4 / DP4 + DP attention + EP4**：一个请求只能落在一条 DP lane 上，其余 3 条 lane 在 EP 通信里空等，profile 显示 lane 间 attention 不均衡占每层关键路径的 33–44%（10/26 EPLB 报告第 6 节）。

去掉 DP 的候选并行方式里，fork 自带 DeepSeek-V4 的 **prefill CP**（`--enable-prefill-cp --cp-strategy interleave --attn-cp-size 4`：一个请求的 token 按 `idx % 4` 轮转到 4 卡，MoE 仍走 EP4 megamoe）。它在 cache 命中时 TTFT 很好（p50 0.8 s vs DP 0.97），但 AgentX C24 实测命中率 95% → 32.6%、TTFT p50 89 s，不可用。

根因是**存储没切**：上游实现在每一层算完本 rank 的 1/4 K/V 后 `cp_materialize_global_token_order()` all-gather 成全序列，按整个 chunk 的 `out_cache_loc` 写进本 rank 自己的 pool。上游代码注释原话：

> *Prefill CP materializes global token order before writing state pools, so every CP rank holds the same state.* —— `disaggregation/common/conn.py`

4 张卡存 4 份一样的 KV，一个 CP worker 的前缀缓存容量只有单卡 1.7M token；24 个 ~100K 的会话（工作集 ≈ 2.4M）互相驱逐。上游为 GLM-5.2 做过 `LayerSplitDSATokenToKVPool`（PR #29421，"reducing per-rank KV memory"），但只覆盖 `DSATokenToKVPool`，且只支持 mooncake；DeepSeek-V4 的 pool 结构（五类池）没有对应实现。

## 2. 设计

### 2.1 按层切，而不是按 token 切

CP 把**计算**按 token 切，但 DSA 稀疏注意力里每个 rank 要拿自己那 1/4 的 query 对**整段上下文**的 K 做 indexer top-k 和 attention。所以无论 KV 怎么存，每层算之前每个 rank 都必须能读到全部 token 的 K：

| 切法 | 每层通信 | 实现代价 |
|---|---|---|
| 按 token（每 rank 存自己 1/4 token 的 KV） | 每层 all-gather 整段上下文的 K，4 个 rank 各自拼全量 | kernel 寻址要改，PD 传输要新协议 |
| **按层（每 rank 存 ~1/4 的层）** | 每层一次 owner → 其他 3 rank 的广播；owner 自己那层零通信 | kernel 不改；PD 传输复用"按层区间切片"（与 PP stage 协议相同） |

通信总量相同，按层切实现简单得多，选它。

### 2.2 切分布局

V4 的 KV 由五类 pool 组成：SWA（滑窗 128，每层一个）、c4（4× 压缩 KV，21 层）、c4-indexer（21 层）、c128（128× 压缩，20 层）、compress-state（压缩器流式状态，每层每请求一个 ring）。

```
43 层连续切 4 段（get_layer_shard_range）：
  rank0: [0,11)   rank1: [11,22)   rank2: [22,33)   rank3: [33,43)

每类分页池（SWA / c4 / c4-indexer / c128）在每个 rank 上：
  owned 层    -> 真·全尺寸 buffer
  非 owned 层 -> 0 页占位（显存不分配）
  + 1 块全尺寸 scratch（页数与真池相同）           -> 每 rank 共 1.25 GB

compress-state 池：不切，全部层复制
```

- **连续区间**而非轮转：PD 传输用 `prefill_start_layer / end_layer` 一个区间就能描述切片，decode 把 4 个 CP rank 当 4 个 PP stage 收。代价是 c4 层数在各 rank 为 4/6/5/6 不均，容量按最差 rank 算。
- **compress-state 不切**：compressor 本来就在每个 rank 对 all-gather 后的全量 token 算一遍，状态写入是确定性的、各 rank 相同，复制零成本、不需要广播。

### 2.3 一步（一个 chunk）内非 owned 层的读写协议

```
begin_step(forward_batch)               # dsv4 backend.init_forward_metadata 调用；各 rank 输入相同 -> 结果相同
    tokens    = req_to_token[req_pool_indices, :seq_len]      # 本步所有请求的全部 token（前缀 + 新）
    kv_pages  = unique(tokens // 256)                         # c4 / c128 / indexer 共用 full-KV 页号
    swa_pages = unique(full_to_swa_mapping[tokens] // 128)    # SWA 池有自己的页空间
    scratch_layer[*] = None                                   # 上一步的 scratch 作废

第 l 层在本步第一次被任何 accessor（读或写）触碰时 (_materialize)：
    owner : payload = pool_l[pages]              # index_select，只取本步会碰的页
    all   : attn_cp_group.broadcast(payload, src=owner)
    others: scratch[pages] = payload             # index_copy_，页号原位放回
    scratch_layer[kind] = l
    返回 owner 的真 buffer / 其他 rank 的 scratch
    （prefetch 版：此时在 side stream 上发起同类池下一层的 gather+broadcast，轮到它时 wait_event + index_copy_）

写入新 token 的 K（all-gather 后每个 rank 都有完整一份）：
    owner 写自己的池；其他 rank 写当前持有该层的 scratch  -> 本 chunk 的 token 对本 rank 的 attention 可见
    下一个 chunk 到来时这些页已在 owner 池里，自然进入下一步的广播集合
```

关键是**页号不变**：c4 的 slot 是 `token_loc // 4`、页大小 256/4 = 64，所以 c4 页号 == full-KV 页号，c128、indexer 同理；scratch 与真池页数一样，按同一页号 scatter 回去，attention / indexer / compressor 的 kernel 拿到的仍是"一块按页索引的 buffer"，**一行 kernel 代码都没改**。广播量只有本步触及的页（一个 100K 请求一层 ≈ 15 MB），不是整个池。

### 2.4 为什么不会死锁

所有 rank 跑完全相同的程序顺序（CP-v1 下 `forward_batch` 是未切分的整批，page 集合一致；各层首次触碰点一致），每个 rank 发起的广播序列一模一样。prefetch 版把下一层的 gather+broadcast 放到 side stream，但仍用同一个 process group，NCCL 用自己的流串行，顺序不变。

### 2.5 容量计算

`pool_configurator` 的每 token 字节数原来按全部层算，改为按"owned 层 + 1 个 scratch 层"算（取最差 rank，保证 4 个 scheduler 对 `max_total_num_tokens` 一致）；compress-state 项仍按全部层。

分层后复制的 SWA 池和 state 池占了近一半显存，且随 full-token 容量等比放大，而它们只需装"在飞请求的窗口 + 每条缓存路径一个 128-token 窗口"。把 `swa-full-tokens-ratio` 从 0.1 降到 0.03（SWA 池停在单卡绝对量级 ~200K token）：

| 配置 | pool 容量（token / rank） | 相对单卡 |
|---|---:|---:|
| 原 CP（复制） | 1,715,456 | 1× |
| 分层，swa ratio 0.1 | 3,970,048 | 2.3× |
| **分层，swa ratio 0.03** | **6,963,712** | **4.1×**（= DP4 四池总和 6.9M，但是单池、无 lane 碎片） |

### 2.6 PD 传输

每个 CP rank 只发自己 owned 层的切片。decode 镜像的 fork 版本已支持 `enable_dsa_cache_layer_split` 时从全部 CP rank 拉取、期望 `attn_cp_size` 份；缺的是 prefill 侧：

- V4 的传输条目按压缩桶排 `[21 c4 kv][21 c4 indexer][20 c128 kv] = 62` 项；非 prepped 路径已有 `_mla_slice_ptrs_for_pp` 按区间切，但 prepped 路径在切片时要求双方提供 layer id，而 V4 pool 不提供 → 新增 `synthesize_compressed_mla_layer_ids()` 按桶布局合成两侧 id 再配对；
- `prefill_end_layer` 改从 pool 的 `layer_shard_end` 取（原公式 `start + len(kv_data_ptrs)` 对按桶排布的 V4 不成立）；
- `get_state_buf_infos / get_c128_state_buf_infos` 只报 owned 层（SWA、c4 state、c4 indexer state、c128 state 的 item 长度与基类一致：state 的 item = 一个请求的 ring，`t[0].nbytes * ring_size`）。

### 2.7 改动清单

| 文件 | 改动 |
|---|---|
| `mem_cache/deepseek_v4_layer_split.py`（新） | `LayerSplitDeepSeekV4TokenToKVPool`：切片子池、scratch、page 集合、广播/prefetch、全部 accessor 覆写、PD buf infos |
| `mem_cache/kv_cache_configurator.py` | V4 + prefill-CP + `--enable-dsa-cache-layer-split` 时选用分层 pool |
| `model_executor/pool_configurator.py` | 按 owned 层数(+scratch)算容量 |
| `layers/cp/utils.py` | `is_dsv4_cache_layer_split_enabled / get_dsv4_cp_layer_shard_info` |
| `layers/attention/deepseek_v4_backend.py` | `init_forward_metadata` → `pool.begin_step(forward_batch, req_to_token)`；`SGLANG_DSV4_KV_DUMP` 逐层校验和 |
| `disaggregation/prefill.py`、`nixl/conn.py`、`utils.py` | 切片区间、合成 layer id 配对 |
| `server_args.py`、`arg_groups/deepseek_v4_hook.py` | 放行 `DeepseekV4ForCausalLM` + nixl；校验 |
| `layers/moe/mega_moe.py` | CP 下 MegaMoE 的 token 数取 `ceil(global / cp_size)`（原来拿整句 16384 撞 8192 上限崩溃） |
| `mem_cache/deepseek_v4_memory_pool.py` | `debug_layer_checksums()`（调试） |

开关：prefill yaml 加 `enable-dsa-cache-layer-split: true`（需 `enable-prefill-cp` + `cp-strategy interleave` + `disaggregation-mode prefill`）、`swa-full-tokens-ratio: 0.03`；环境变量 `SGLANG_DSV4_LAYER_SPLIT_PREFETCH=0` 关 prefetch，`SGLANG_DSV4_LAYER_SPLIT_DEBUG=1` 打每步广播统计，`SGLANG_DSV4_KV_DUMP=1` 打逐层校验和（慢，只用于验证）。

## 3. 正确性验证

不靠肉眼看文本，用 `SGLANG_DSV4_KV_DUMP` 在每步结束对本步触及的页逐层算字节校验和，分层模式（4 个 rank 的 owned 层拼起来）对比复制模式：

| 用例 | 结果 |
|---|---|
| 18 token prompt | 105 个 layer-buffer（43 SWA + 21 c4 + 21 indexer + 20 c128）**比特一致** |
| 4,470 / 24,087 token prompt（多 chunk） | 相差 100/105；但复制模式自己冷跑两遍也相差 100/105（第 2 层 MoE 起 kernel 本身 run-to-run 不确定）→ 分层实现与原实现数值等价 |
| 6 条 prompt 贪心前 8 token，cached 路径 | 与复制模式逐条一致 |
| 24K prompt 端到端 | 推理内容正确（"count occurrences of 'fox'"） |

## 4. 实测

同一套 1P:1D 硬件，同一 AgentX 256K 协议（20 min、per-trace cap 300 s），decode 全部是 V13 的配置。**成本一律全时口径**（所有卡 × 整个窗口含空闲 = 21.3 元/20 min，各配置相同）。

### 4.1 AgentX C24 四方对比

| | DP4（V13 原样） | PP4 + chunk 4K | CP4 复制 KV（上游原样） | **CP4 + 分层 KV** |
|---|---:|---:|---:|---:|
| 完成请求 | 822 | 871 | 216 | **870** |
| 命中率 | 95.0% | 95.9% | 32.6% | 94.3% |
| TTFT p50 / p90 / p99 (s) | 1.53 / 5.8 / 17.0 | 1.15 / 3.5 / 7.7 | 89 / 129 / 145 | **0.47** / 5.1 / 13.2 |
| TPS/用户 | 196 | 189 | 268 | 179 |
| 输出 tok/s | 644 | 719 | 209 | 719 |
| 每请求成本 | 0.0260 元 | 0.0244 | 0.0928 | 0.0245 |
| 输出成本 | 15.52 元/M | 13.90 | 47.8 | 13.91 |
| 未命中输入（10% 分摊） | 0.731 元/M | 0.718 | 0.580 | 0.648 |

### 4.2 clean 单请求 TTFT（无 cache，秒）

| ISL | DP4 | PP4 chunk4K | CP4 分层 | CP4 分层 + prefetch |
|---|---:|---:|---:|---:|
| 8K | 1.64 | 1.10 | **0.54** | 0.53 |
| 32K | 4.94 | 2.60 | 2.31 | 2.26 |
| 64K | 9.59 | 4.72 | 4.57 | 4.53 |
| 128K | 19.2 | 9.18 | 9.47 | 9.31 |
| 32K × C4 中位 | 9.45 | 7.39 | 8.19 | 8.12 |

prefetch 命中 101/105 层广播，但只快 1–2%：广播不是瓶颈，长请求受算力限制。

### 4.3 并发阶梯（CP4 分层 + prefetch，pool 6.96M；SLA = TTFT p50 ≤ 1.05 s 且 TPS ≥ 70）

| 并发 | 请求 | 命中 | TTFT p50/p90/p99 (s) | TPS | 输出 tok/s | 每请求（全时） | 输出 元/M | 未命中 元/M |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| C24 | 870 | 94.3% | 0.47 / 5.1 / 13.2 | 179 | 719 | 0.0245 | 13.91 | 0.648 |
| **C32（SLA 点）** | **1,389** | **95.6%** | **0.62 / 4.1 / 11.8** | **142** | **945** | **0.0154** | **10.59** | **0.478** |
| C40 | 1,640 | 94.5% | 1.40 / 10.0 / 22.0 | 122 | 1,256 | 0.0133 | 7.96 | 0.348 |

DP4 的 SLA 点是 C24（C28 时 p50 1.18 已破线），CP 分层把 SLA 并发推到 C32：**同一套硬件、SLA 内，每请求成本 0.0260 → 0.0154（−41%），输出 15.52 → 10.59 元/M**。

## 5. 踩过的坑（都已修）

1. 上游 MegaMoE 在 CP 下读到 DP 同步的整句 token 数（16384）撞 8192 上限直接崩溃；改为 `ceil(global / cp_size)`。
2. 这个 fork 的 `ForwardBatch` 没有 `req_to_token_pool`，`begin_step` 静默早退、一次广播都没发，scratch 全是垃圾，输出"看着通顺但错"——改成 backend 传 `req_to_token`。
3. c4 compress-state 的传输 item 长度要乘 `ring_size`，少乘导致 decode 收到的状态不完整、续写从第 3–5 个 token 起偏离。
4. 共享主机上 ephemeral 端口被别的 decode 的重试风暴打穿，frontend/decode 随机 bind 失败——所有端口移到 <32768 并固定 `nccl-port`。
5. V13 decode（fused MoE + DSpark）随机 TP-desync 挂死（20 min 窗口约 1/3 概率），与 prefill 模式无关，阶梯脚本遇到即重启 decode 重跑。

## 6. 未决与下一步

- **一个待钉死的差异**：首 token 比对里 18-token 的一条 prompt 在 prefetch 版（且 decode 重启后）稳定输出 "asks for the"，无 prefetch 版 / 复制版为 "asks to list"。prefill 侧校验和已证明一致，差异发生在 decode 续写的第 6 个 token，怀疑是 decode 新进程对近平手 token 的固定偏置；待做 prefetch on/off + KV_DUMP 的 A/B。在此之前 C32 的数字请视为"待确认"。
- 容量还能涨：把 compress-state 池也按层切（需为 state 行做选择性广播），或进一步压 SWA 比例。
- owner 分布按字节均衡（c4 层 4/6/5/6 → 容量按最差 rank 算）。
- 进 recipe（V14 = V13 + 这套 prefill）需重建 prefill 镜像（当前用 bind-mount 源码跑）并补精度 eval；再在 4P:1D@C72 生产拓扑上复测。
- 代码对上游 CP-v2 / unified_kv 路径未接；仅支持 prefill 侧（decode 不切）。
