# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FlashInfer ``moe_ep`` MegaMoE prefill overlay for FP4-expert DeepSeek-V4 on SM120.

``--moe-a2a-backend deepep_v2`` builds the model exactly as the baseline.  With ``SGLANG_FI_MOE_EP=<kernel>``
every forward step that contains an extend batch (the DP-synchronised ``is_extend_in_batch``, rank-uniform) runs
the routed experts through ``flashinfer.moe_ep.MoEEpLayer`` (one layer object per MoE layer, all sharing the
backend's pooled workspace); pure-decode steps keep the baseline DeepEP masked DeepGEMM path.  The shared expert
stays on the model's replicated MLP for both paths.

Kernel (FlashInfer public config, used as shipped): ``Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig`` (one persistent
kernel, per-pair SELF/LSA/GIN transport).

Weights: FP4 experts are stored once.  The canonical E2M1 + UE8M0 pack is handed to the kernel's own
preprocessing; the result's gate/up 8-row interleaved FC1 is copied over ``w13_weight`` in place and the kernel's
transformed weights alias ``w13_weight`` / ``w2_weight`` (only the kernel-specific scale layouts are extra).  The
DeepGEMM decode path reads the interleaved ``w13`` through its swizzle-aware SwiGLU (same layout as DeepGEMM
MegaMoE), with its packed UE8M0 scales permuted the same way; when no step can reach that path (FlashInfer for every
step, no CUDA graphs, no runtime EPLB) those DeepGEMM scales are not kept (see SGLANG_FI_MOE_EP_KEEP_DG_SCALES).

Environment:
  SGLANG_FI_MOE_EP              cake | cake_nvfp4 | cake_mxfp4 ("" = off, the baseline).  cake_nvfp4 =
                                Sm120_Nvfp4_Nvfp4_Bf16_Cake (W4A4): the MXFP4 checkpoint scales convert losslessly to
                                E4M3 block-16 (per-expert power-of-two shift folded into fc1_alpha / fc2_alpha);
                                activations NVFP4.  cake_mxfp4 = Sm120_Mxfp4_Mxfp4_Bf16_Cake (W4A4, MOE_ACT_CODEC=2):
                                the cake (W4A8) weight pack unchanged, activations MXFP4 (E2M1 + UE8M0 per 32, no
                                global scales); KERNEL_SCHED / RETURN_CODEC as for cake
  SGLANG_FI_MOE_EP_NVFP4_INPUT_NORM  cake_nvfp4: input_norm_const (default 1.0; fc1_alpha carries 1/it)
  SGLANG_FI_MOE_EP_NVFP4_NORM_FILE   cake_nvfp4: torch file {layer_id: fc1_norm_const [num_experts] fp32} (calibrated
                                intermediate norm; fc2_alpha carries 1/norm); default: 1.0
  SGLANG_FI_MOE_EP_TABLE        cake transport table: all_gin | all_lsa | lsa_where_reachable | tuned | auto |
                                lsa_pairs (explicit table: LSA inside rank pairs (2i, 2i+1), GIN otherwise)
  SGLANG_FI_MOE_EP_MAX_ROWS     per-rank row capacity of a layer call (default 4096)
  SGLANG_FI_MOE_EP_CHECK_EVERY  read protocol_error every N layer calls (host sync; 0 = never, default 430)
  SGLANG_FI_MOE_EP_LOG_EVERY    log call statistics every N layer calls (default 4300)
  SGLANG_FI_MOE_EP_CALLER_ROWS  1 (default): pass the step's per-DP-rank token counts (already all-gathered by DP
                                attention), so the kernel call has no host sync: as MoEEpTensors.num_tokens_per_rank
                                where the FlashInfer tree has that field, else as the rank-uniform bound
                                stage_inputs(compile_tokens_per_rank=max(counts)) (merged moe_ep API);
                                0: the backend all-gathers [rows, expert histogram] per call
  SGLANG_FI_MOE_EP_ROWS_HINT    1 (default): with CALLER_ROWS=1 on the bound API, also pass the counts as
                                stage_inputs(rows_by_rank=...) where the FlashInfer tree takes it: the cake kernel picks
                                its row schedule (m64 / m128) from the mean rows per rank instead of the largest count
                                (a 1-source-rank P/D prefill step at 1024 rows runs m64; identical results); 0: off
  SGLANG_FI_MOE_EP_KERNEL_SCHED 1: build the cake layers with kernel_sched=True (closure cake_mxfp4_megamoe_ks);
                                0 (default): the config default
  SGLANG_FI_MOE_EP_DECODE       1: steps without an extend batch (decode / idle, not captured into a CUDA graph) also
                                run through the FlashInfer layer; 0 (default): they keep the DeepEP masked path
  SGLANG_FI_MOE_EP_LOAD_POOL    1 (default): each layer's weight conversion allocates its temporaries in a private CUDA
                                memory pool and copies the persistent outputs out (no allocator fragmentation left
                                behind: 1.4-2.1 GiB per GPU NVFP4 EP4); 0: convert in the default pool
  SGLANG_FI_MOE_EP_KEEP_DG_SCALES  1: always keep the DeepGEMM-layout copy of the expert scales.  Default (0): the
                                copy is dropped (~2 GiB per GPU at EP4) when no forward can reach the DeepGEMM path:
                                SGLANG_FI_MOE_EP_DECODE=1, CUDA graphs disabled (captured graphs keep the DeepEP
                                path) and no runtime EPLB rebalancing (--enable-eplb moves the scale parameters)
  SGLANG_FI_MOE_EP_RETURN_CODEC cake expert-result return codec: bf16 (lossless codec) | int8 (per-64 int8 + BF16
                                scale; needs a FlashInfer tree with the return_codec config); unset = config default
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import os
import time
from contextlib import nullcontext
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.runner import get_is_capture_mode

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

logger = logging.getLogger(__name__)

ENV_KERNEL = "SGLANG_FI_MOE_EP"
ENV_TABLE = "SGLANG_FI_MOE_EP_TABLE"
ENV_MAX_ROWS = "SGLANG_FI_MOE_EP_MAX_ROWS"
ENV_CHECK_EVERY = "SGLANG_FI_MOE_EP_CHECK_EVERY"
ENV_LOG_EVERY = "SGLANG_FI_MOE_EP_LOG_EVERY"
ENV_CALLER_ROWS = "SGLANG_FI_MOE_EP_CALLER_ROWS"
ENV_KERNEL_SCHED = "SGLANG_FI_MOE_EP_KERNEL_SCHED"
ENV_ROWS_HINT = "SGLANG_FI_MOE_EP_ROWS_HINT"
ENV_DECODE = "SGLANG_FI_MOE_EP_DECODE"
ENV_RETURN_CODEC = "SGLANG_FI_MOE_EP_RETURN_CODEC"
ENV_KEEP_DG_SCALES = "SGLANG_FI_MOE_EP_KEEP_DG_SCALES"
ENV_LOAD_POOL = "SGLANG_FI_MOE_EP_LOAD_POOL"
KERNELS = ("cake", "cake_nvfp4", "cake_mxfp4")
ENV_NVFP4_INPUT_NORM = "SGLANG_FI_MOE_EP_NVFP4_INPUT_NORM"
ENV_NVFP4_NORM_FILE = "SGLANG_FI_MOE_EP_NVFP4_NORM_FILE"
_SCALE_GROUP = 32


class _Stats:
    calls: int = 0
    rows: int = 0
    max_rows: int = 0
    empty_calls: int = 0
    checks: int = 0
    seconds_build: float = 0.0
    layers: list = []


def fi_moe_ep_kernel() -> str:
    kernel = os.environ.get(ENV_KERNEL, "").strip()
    if kernel and kernel not in KERNELS:
        raise ValueError(f"{ENV_KERNEL}={kernel!r}: expected one of {KERNELS} or empty")
    return kernel


def fi_moe_ep_overlay_enabled() -> bool:
    return bool(fi_moe_ep_kernel()) and get_moe_a2a_backend().is_deepep_v2()


def _max_rows() -> int:
    return int(os.environ.get(ENV_MAX_ROWS, "4096"))


def _caller_rows() -> bool:
    return os.environ.get(ENV_CALLER_ROWS, "1") == "1"


def _kernel_sched() -> bool:
    return os.environ.get(ENV_KERNEL_SCHED, "0") == "1"


@functools.lru_cache(maxsize=None)
def _stage_takes_rows_hint() -> bool:
    """FlashInfer trees whose MoEEpMegaLayer.stage_inputs accepts rows_by_rank (row-schedule hint)."""
    if os.environ.get(ENV_ROWS_HINT, "1") != "1":
        return False
    import inspect

    from flashinfer.moe_ep.modes.mega_layer import MoEEpMegaLayer

    return "rows_by_rank" in inspect.signature(MoEEpMegaLayer.stage_inputs).parameters


@functools.lru_cache(maxsize=None)
def _tensors_take_rows() -> bool:
    """FlashInfer trees before the merged moe_ep API carry the per-rank counts on MoEEpTensors."""
    from flashinfer.moe_ep import MoEEpTensors

    return "num_tokens_per_rank" in {f.name for f in dataclasses.fields(MoEEpTensors)}


def _check_caller_rows_mapping(*, world: int, rank: int) -> None:
    """global_num_tokens_cpu is indexed by attention-DP rank; the moe_ep list is indexed by EP rank."""
    from sglang.srt.layers.dp_attention import (
        get_attention_dp_rank,
        get_attention_dp_size,
        is_dp_attention_enabled,
    )

    if (
        not is_dp_attention_enabled()
        or get_attention_dp_size() != world
        or get_attention_dp_rank() != rank
    ):
        raise RuntimeError(
            f"{ENV_CALLER_ROWS}=1 needs DP attention with DP rank == EP rank (EP{world} rank {rank}); "
            f"set {ENV_CALLER_ROWS}=0"
        )


# ---------------------------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------------------------


def _ue8m0_bytes(scale: torch.Tensor) -> torch.Tensor:
    """FP32 power-of-two block scales -> UE8M0 exponent bytes (exact; refuses anything else)."""
    if scale.dtype != torch.float32:
        raise RuntimeError(
            f"{ENV_KERNEL}: expected fp32 FP4 block scales, got {scale.dtype}"
        )
    bits = scale.contiguous().view(torch.int32)
    if bool(((bits & 0x807FFFFF) != 0).any()):
        raise RuntimeError(
            f"{ENV_KERNEL}: FP4 block scales are not positive powers of two"
        )
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def _transport_table(name: str, world: int):
    if name != "lsa_pairs":
        return name
    return tuple(
        0 if a == b else (1 if a // 2 == b // 2 else 2)
        for a in range(world)
        for b in range(world)
    )


def _megakernel_config_nvfp4(
    *,
    intermediate: int,
    top_k: int,
    swiglu_limit: Optional[float],
    world: int,
    alpha1,
    alpha2,
    norm,
):
    from flashinfer.moe_ep import Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig

    limit = 0 if swiglu_limit is None else int(swiglu_limit)
    if swiglu_limit is not None and float(limit) != float(swiglu_limit):
        raise RuntimeError(
            f"{ENV_KERNEL}=cake_nvfp4: swiglu_limit {swiglu_limit} is not an integer"
        )
    extra = {}
    fields = {
        f.name for f in dataclasses.fields(Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig)
    }
    if _kernel_sched():
        if "kernel_sched" not in fields:
            raise RuntimeError(
                f"{ENV_KERNEL_SCHED}=1: this FlashInfer tree has no kernel_sched config"
            )
        extra["kernel_sched"] = True
    codec = os.environ.get(ENV_RETURN_CODEC, "").strip()
    if codec:
        extra["combine_dtype"] = codec
    input_norm = float(os.environ.get(ENV_NVFP4_INPUT_NORM, "1.0"))
    return Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig(
        intermediate_size=intermediate,
        top_k=top_k,
        gate_up_clamp=(limit or None),
        shared_expert=None,
        transport_table=_transport_table(os.environ.get(ENV_TABLE, "all_gin"), world),
        input_norm_const=input_norm,
        fc1_alpha=alpha1 / input_norm,
        fc2_alpha=alpha2 if norm is None else alpha2 / norm,
        fc1_norm_const=norm,
        **extra,
    )


def _megakernel_config(
    *,
    intermediate: int,
    top_k: int,
    swiglu_limit: Optional[float],
    world: int,
    kernel: str = "cake",
):
    """cake (W4A8) or cake_mxfp4 (W4A4 MXFP4 activations): same fields, same weight pack."""
    if kernel == "cake_mxfp4":
        try:
            from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp4_bf16_cake import (
                Sm120_Mxfp4_Mxfp4_Bf16_Cake_MegaMoeConfig as Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                f"{ENV_KERNEL}=cake_mxfp4: this FlashInfer tree has no MXFP4 W4A4 backend"
            ) from exc
    else:
        from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
            Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig,
        )

    limit = 0 if swiglu_limit is None else int(swiglu_limit)
    if swiglu_limit is not None and float(limit) != float(swiglu_limit):
        raise RuntimeError(
            f"{ENV_KERNEL}={kernel}: swiglu_limit {swiglu_limit} is not an integer"
        )
    extra = {}
    fields = {
        f.name for f in dataclasses.fields(Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig)
    }
    if _kernel_sched():
        if "kernel_sched" not in fields:
            raise RuntimeError(
                f"{ENV_KERNEL_SCHED}=1: this FlashInfer tree has no kernel_sched config"
            )
        extra["kernel_sched"] = True
    codec = os.environ.get(ENV_RETURN_CODEC, "").strip()
    if codec:
        if "return_codec" not in fields:
            raise RuntimeError(
                f"{ENV_RETURN_CODEC}={codec}: this FlashInfer tree has no return_codec config"
            )
        extra["return_codec"] = codec
    return Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(
        intermediate_size=intermediate,
        top_k=top_k,
        swiglu_limit=limit,
        shared_expert=None,
        transport_table=_transport_table(os.environ.get(ENV_TABLE, "all_gin"), world),
        **extra,
    )


def _shared_filler(tensor: torch.Tensor, cache: dict) -> torch.Tensor:
    """One copy of the noshared variant's constant FP8 filler operand per (shape, device), shared by all layers."""
    key = (tuple(tensor.shape), tensor.dtype, tensor.device)
    if key not in cache:
        cache[key] = tensor
    return cache[key]


_FILLERS: dict = {}


def _transform_aliasing(
    pack, w13_u8: torch.Tensor, w2_u8: torch.Tensor, *, hidden: int, intermediate: int
):
    """Kernel preprocessing of the canonical pack; FC1/FC2 then alias the layer's (interleaved) w13 / w2 storage."""
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
        preprocess_mega_weights,
    )

    tw = preprocess_mega_weights(
        pack, hidden_size=hidden, intermediate_size=intermediate
    )
    w13_u8.copy_(tw.w1_fp4.view(torch.uint8))
    if tw.w2_fp4.data_ptr() != w2_u8.data_ptr() and not torch.equal(
        tw.w2_fp4.view(torch.uint8), w2_u8
    ):
        raise RuntimeError(f"{ENV_KERNEL}=cake: FC2 is not the canonical w2 layout")
    return dataclasses.replace(
        tw,
        w1_fp4=w13_u8.view(tw.w1_fp4.dtype),
        w2_fp4=w2_u8.view(tw.w2_fp4.dtype),
        w1_b8=_shared_filler(tw.w1_b8, _FILLERS),
        w2_b8=_shared_filler(tw.w2_b8, _FILLERS),
    )


_NORMS: dict = {}


def _nvfp4_norm(layer_id: int, num_experts: int, rank: int, local: int, device):
    """This rank's slice of the calibrated fc1_norm_const of ``layer_id`` (None = uncalibrated 1.0)."""
    path = os.environ.get(ENV_NVFP4_NORM_FILE, "")
    if not path:
        return None
    if path not in _NORMS:
        _NORMS[path] = torch.load(path, map_location="cpu")
    full = _NORMS[path][layer_id].to(torch.float32)
    if full.numel() != num_experts:
        raise RuntimeError(
            f"{ENV_NVFP4_NORM_FILE}: layer {layer_id} has {full.numel()} norms, expected {num_experts}"
        )
    return full[rank * local : (rank + 1) * local].contiguous().to(device)


def _transform_nvfp4_aliasing(
    layer, w13_u8: torch.Tensor, w2_u8: torch.Tensor, *, hidden: int, intermediate: int
):
    """MXFP4 pack -> exact NVFP4 (per-expert shifted E4M3 scales + alpha) -> kernel layout; FC1/FC2 alias w13 / w2."""
    from flashinfer.moe_ep import MoEWeightPack, mxfp4_scales_to_nvfp4_shifted
    from flashinfer.moe_ep.backends.mega.kernel.sm120.nvfp4_nvfp4_bf16_cake import (
        preprocess_nvfp4_mega_weights,
    )

    sc13, alpha1 = mxfp4_scales_to_nvfp4_shifted(
        _ue8m0_bytes(layer.w13_weight_scale_inv.data)
    )
    sc2, alpha2 = mxfp4_scales_to_nvfp4_shifted(
        _ue8m0_bytes(layer.w2_weight_scale_inv.data)
    )
    pack = MoEWeightPack(w13_u8, w2_u8, sc13, sc2)
    tw = preprocess_nvfp4_mega_weights(
        pack, hidden_size=hidden, intermediate_size=intermediate
    )
    w13_u8.copy_(tw.w1_fp4.view(torch.uint8))
    if tw.w2_fp4.data_ptr() != w2_u8.data_ptr() and not torch.equal(
        tw.w2_fp4.view(torch.uint8), w2_u8
    ):
        raise RuntimeError(
            f"{ENV_KERNEL}=cake_nvfp4: FC2 is not the canonical w2 layout"
        )
    tw = dataclasses.replace(
        tw,
        w1_fp4=w13_u8.view(tw.w1_fp4.dtype),
        w2_fp4=w2_u8.view(tw.w2_fp4.dtype),
        w1_b8=_shared_filler(tw.w1_b8, _FILLERS),
        w2_b8=_shared_filler(tw.w2_b8, _FILLERS),
    )
    return tw, alpha1, alpha2


def _interleave_index(intermediate: int, device) -> torch.Tensor:
    try:
        from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.weights import (
            gate_up_interleave_index,
        )
    except ImportError:  # merged moe_ep layout (cake_* module names)
        from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_weights import (
            gate_up_interleave_index,
        )

    return gate_up_interleave_index(intermediate).to(device)


def _deepgemm_scales_interleaved(layer, *, index: torch.Tensor) -> None:
    """Baseline DeepGEMM UE8M0 packing of both scale tensors, then the w13 scales permuted like w13 (decode path)."""
    from deep_gemm import transform_sf_into_required_layout

    for scale_param, weight_param in (
        (layer.w13_weight_scale_inv, layer.w13_weight),
        (layer.w2_weight_scale_inv, layer.w2_weight),
    ):
        num_experts, n, _ = scale_param.data.shape
        scale_param.data = transform_sf_into_required_layout(
            scale_param.data.contiguous(),
            mn=n,
            k=weight_param.shape[2] * 2,
            recipe=(1, _SCALE_GROUP),
            num_groups=num_experts,
            disable_ue8m0_cast=False,
        )
        scale_param.format_ue8m0 = True
    sf13 = layer.w13_weight_scale_inv.data
    e, n1, kb1 = sf13.shape
    if tuple(sf13.stride()) != (kb1 * n1, 1, n1):
        raise RuntimeError(
            f"{ENV_KERNEL}: DeepGEMM w13 scales are not mn-major (strides {tuple(sf13.stride())})"
        )
    phys = sf13.as_strided((e, kb1, n1), (kb1 * n1, n1, 1))
    for i in range(e):
        phys[i].copy_(phys[i].index_select(1, index))


class _ConversionPool:
    """Private CUDA allocator pool for one layer's weight conversion (SGLANG_FI_MOE_EP_LOAD_POOL=1, default).

    The conversion allocates large temporaries (the interleaved FC1 copy, fp32 -> UE8M0 / E4M3 scale planes, int32
    code planes) next to a few persistent outputs (kernel-layout scales, alphas).  In the default pool the persistent
    tensors are carved out of freed but still cached segments (the temporaries', and the 128 MiB segments that other
    weight post-processing just freed) and pin the rest of them: measured 1.4-2.1 GiB (NVFP4) / ~0.3 GiB (W4A8) per
    GPU at EP4 that the empty_cache() before SGLang's KV sizing cannot return.  Here the temporaries live in a private
    pool; ``out()`` releases the default pool's free segments once, then copies each persistent tensor into fresh
    default-pool memory, and ``close()`` releases the private segments."""

    def __init__(self) -> None:
        self.pool = (
            torch.cuda.MemPool() if os.environ.get(ENV_LOAD_POOL, "1") == "1" else None
        )

    def __enter__(self):
        self._released = False
        if self.pool is not None:
            self._ctx = torch.cuda.use_mem_pool(self.pool)
            self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        if self.pool is not None:
            self._ctx.__exit__(*exc)
        return False

    def out(self, tensor):
        """A default-pool copy of a persistent conversion output (same strides)."""
        if self.pool is None or tensor is None:
            return tensor
        if not self._released:
            # free segments SGLang's own weight post-processing left cached (e.g. 128 MiB fp8 / fp32 scale
            # segments): a copy carved out of one would pin the rest of it
            torch.cuda.empty_cache()
            self._released = True
        copy = torch.empty_strided(
            tensor.shape, tensor.stride(), dtype=tensor.dtype, device=tensor.device
        )
        copy.copy_(tensor)
        return copy

    def close(self) -> None:
        if self.pool is not None:
            self.pool = None
            torch.cuda.empty_cache()


def _transformed_out(tw, conv: _ConversionPool, keep_storage: tuple):
    """Copy every tensor field of the transformed weights out of the conversion pool, except the aliased w13 / w2
    (``keep_storage``) and the shared constant fillers (FlashInfer caches them per device; created in the default
    pool by ``_warm_fillers``)."""
    keep = {t.untyped_storage().data_ptr() for t in keep_storage}
    keep |= {v.untyped_storage().data_ptr() for v in _FILLERS.values()}
    changes = {}
    for f in dataclasses.fields(tw):
        t = getattr(tw, f.name)
        if isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() not in keep:
            changes[f.name] = conv.out(t)
    return dataclasses.replace(tw, **changes)


def _warm_fillers(device, hidden: int, intermediate: int) -> None:
    """Create FlashInfer's per-device constant filler operands (noshared variant) in the default pool, before the first
    pooled conversion would put them (permanently, pinning two private segments) into the private pool.
    """
    try:
        from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_weights import (
            _noshared_operands,
        )
    except ImportError:
        try:
            from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.weights import (
                _noshared_operands,
            )
        except ImportError:
            return
    for t in _noshared_operands(device, hidden, intermediate):
        _shared_filler(t, _FILLERS)


def _deepgemm_path_reachable() -> bool:
    """Whether any forward can still run this layer through the baseline DeepEP + DeepGEMM path (which reads the
    DeepGEMM-layout scales).  With SGLANG_FI_MOE_EP_DECODE=1 every non-captured step takes the FlashInfer layer, so the
    DeepGEMM path is only reached from captured CUDA graphs; runtime EPLB rebalancing also moves the scale
    parameters between ranks."""
    if (
        os.environ.get(ENV_KEEP_DG_SCALES, "0") == "1"
        or os.environ.get(ENV_DECODE, "0") != "1"
    ):
        return True
    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.server_args import get_global_server_args

    args = get_global_server_args()
    graphs = getattr(args, "cuda_graph_config", None)
    any_graphs = not getattr(args, "disable_cuda_graph", False) and (
        graphs is None
        or graphs.decode.backend != Backend.DISABLED
        or graphs.prefill.backend != Backend.DISABLED
    )
    return any_graphs or bool(getattr(args, "enable_eplb", False))


def _drop_deepgemm_scales(layer) -> None:
    """Free the checkpoint-layout scale parameters (the FlashInfer layer holds its own kernel-layout copy)."""
    for scale_param in (layer.w13_weight_scale_inv, layer.w2_weight_scale_inv):
        scale_param.data = scale_param.data.new_empty((scale_param.shape[0], 0, 0))
        scale_param.format_ue8m0 = True
    layer._fi_moe_ep_dg_dropped = True


def build_fi_moe_ep_experts_weights(layer) -> None:
    """Fp8MoEMethod.process_weights_after_loading hook for FP4 experts under the overlay (replaces the baseline's
    DeepGEMM scale packing, which it performs itself after the FlashInfer layer is built).
    """
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEWeightPack,
    )

    from sglang.srt.distributed.parallel_state import get_moe_ep_group

    t0 = time.time()
    kernel = fi_moe_ep_kernel()
    num_experts, ep_size = int(layer.num_experts), int(layer.moe_ep_size)
    if num_experts % ep_size or int(layer.num_local_experts) != num_experts // ep_size:
        raise RuntimeError(
            f"{ENV_KERNEL}: expects plain EP sharding (no redundant / fused shared slots); got num_experts="
            f"{num_experts} ep_size={ep_size} num_local_experts={layer.num_local_experts}"
        )
    w13, w2 = layer.w13_weight.data, layer.w2_weight.data
    if w13.dtype != torch.int8 or w2.dtype != torch.int8:
        raise RuntimeError(
            f"{ENV_KERNEL}: expected packed E2M1 int8 experts, got {w13.dtype}/{w2.dtype}"
        )
    e, n1, half_h = w13.shape
    hidden, intermediate = 2 * half_h, n1 // 2
    w13_u8, w2_u8 = w13.view(torch.uint8), w2.view(torch.uint8)
    conv = _ConversionPool()
    if conv.pool is not None:
        _warm_fillers(w13.device, hidden, intermediate)
    conv.__enter__()
    pack = MoEWeightPack(
        w13_u8,
        w2_u8,
        _ue8m0_bytes(layer.w13_weight_scale_inv.data),
        _ue8m0_bytes(layer.w2_weight_scale_inv.data),
    )
    index = _interleave_index(intermediate, w13.device)
    check_layout = (
        not _Stats.layers
    )  # one-time cross-check of the shared FC1 layout on the first layer
    reference = w13_u8.index_select(1, index) if check_layout else None
    nv = kernel == "cake_nvfp4"
    if nv:
        transformed, alpha1, alpha2 = _transform_nvfp4_aliasing(
            layer, w13_u8, w2_u8, hidden=hidden, intermediate=intermediate
        )
    else:
        transformed = _transform_aliasing(
            pack, w13_u8, w2_u8, hidden=hidden, intermediate=intermediate
        )
    if reference is not None:
        if not torch.equal(reference, w13_u8):
            raise RuntimeError(
                f"{ENV_KERNEL}={kernel}: FC1 layout differs from the DeepGEMM gate/up 8-row interleave"
            )
        del reference
    conv.__exit__(None, None, None)
    transformed = _transformed_out(transformed, conv, (w13_u8, w2_u8))
    if nv:
        alpha1, alpha2 = conv.out(alpha1), conv.out(alpha2)

    group = get_moe_ep_group().device_group
    world = torch.distributed.get_world_size(group)
    rank = torch.distributed.get_rank(group)
    if world != ep_size:
        raise RuntimeError(
            f"{ENV_KERNEL}: MoE EP group size {world} != moe_ep_size {ep_size}"
        )
    if _caller_rows():
        _check_caller_rows_mapping(world=world, rank=rank)
    layer.fi_moe_ep_layer = MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=world,
            rank=rank,
            device=torch.cuda.current_device(),
            process_group=group,
        ),
        fleet_params=FleetParams(
            num_experts=num_experts,
            max_tokens_per_rank=_max_rows(),
            token_hidden_size=hidden,
        ),
        weights=pack,  # not stored when transformed_weights is supplied
        backend=MegaConfig(
            megakernel=(
                _megakernel_config_nvfp4(
                    intermediate=intermediate,
                    top_k=int(layer.top_k),
                    swiglu_limit=layer.moe_runner_config.swiglu_limit,
                    world=world,
                    alpha1=alpha1,
                    alpha2=alpha2,
                    norm=_nvfp4_norm(layer.layer_id, num_experts, rank, e, w13.device),
                )
                if nv
                else _megakernel_config(
                    intermediate=intermediate,
                    top_k=int(layer.top_k),
                    swiglu_limit=layer.moe_runner_config.swiglu_limit,
                    world=world,
                    kernel=kernel,
                )
            ),
            quantize_input=True,
            preprocess_weights=False,
            transformed_weights=transformed,
        ),
    )
    del pack
    if _deepgemm_path_reachable():
        with conv:
            _deepgemm_scales_interleaved(layer, index=index)
        for scale_param in (layer.w13_weight_scale_inv, layer.w2_weight_scale_inv):
            scale_param.data = conv.out(scale_param.data)
    else:
        _drop_deepgemm_scales(layer)
    del transformed
    conv.close()
    layer._fi_moe_ep_built = True
    _Stats.layers.append(layer.layer_id)
    _Stats.seconds_build += time.time() - t0
    if len(_Stats.layers) == 1 or len(_Stats.layers) % 10 == 0:
        logger.info(
            "%s=%s: built %d MoE layers (EP%d rank %d, %d local experts, max_rows %d, kernel_sched %d, return_codec %s, "
            "rows API %s, %.1fs)",
            ENV_KERNEL,
            kernel,
            len(_Stats.layers),
            world,
            rank,
            e,
            _max_rows(),
            int(_kernel_sched()),
            os.environ.get(ENV_RETURN_CODEC, "") or "default",
            (
                (
                    "tensors"
                    if _tensors_take_rows()
                    else ("bound+hint" if _stage_takes_rows_hint() else "bound")
                )
                if _caller_rows()
                else "gather"
            ),
            _Stats.seconds_build,
        )


# ---------------------------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------------------------


def should_use_fi_moe_ep(
    moe: DeepseekV2MoE, forward_batch: Optional[ForwardBatch]
) -> bool:
    if not fi_moe_ep_overlay_enabled():
        return False
    if not getattr(moe.experts, "_fi_moe_ep_built", False):
        raise RuntimeError(
            f"{ENV_KERNEL}: layer {moe.layer_id} has no FlashInfer moe_ep layer (FP4 experts required)"
        )
    if os.environ.get(ENV_DECODE, "0") == "1" and not get_is_capture_mode():
        return True
    if getattr(moe.experts, "_fi_moe_ep_dg_dropped", False):
        raise RuntimeError(
            f"{ENV_KERNEL}: layer {moe.layer_id} would take the DeepEP/DeepGEMM path but its DeepGEMM scales were "
            f"dropped at load (set {ENV_KEEP_DG_SCALES}=1)"
        )
    if get_is_capture_mode():
        return False  # captured decode graphs stay on the DeepEP masked path
    from sglang.srt.layers.dp_attention import get_is_extend_in_batch

    return bool(get_is_extend_in_batch())


def forward_fi_moe_ep(
    moe: DeepseekV2MoE,
    hidden_states: torch.Tensor,
    forward_batch: Optional[ForwardBatch] = None,
    input_ids_global: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from flashinfer.moe_ep import MoEEpTensors

    experts = moe.experts
    num_tokens = int(hidden_states.shape[0])
    top_k = int(experts.top_k)
    shared_output = None
    rows_by_rank = (
        forward_batch.global_num_tokens_cpu
        if (forward_batch is not None and _caller_rows())
        else None
    )
    if num_tokens > 0:
        router_logits = moe.gate(hidden_states, forward_batch=forward_batch)
        topk_kwargs = (
            {"input_ids": input_ids_global} if getattr(moe, "is_hash", False) else {}
        )
        # the DeepSeek-V4 layer loop already holds the recorder's current layer (nested with_value asserts)
        _rec = get_global_expert_distribution_recorder()
        _held = (
            getattr(getattr(_rec, "_current_layer_idx", None), "value", None)
            is not None
        )
        with nullcontext() if _held else _rec.with_current_layer(moe.layer_id):
            topk_output = moe.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=(
                    forward_batch.num_token_non_padded
                    if forward_batch is not None
                    else None
                ),
                expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                    layer_id=moe.layer_id
                ),
                **topk_kwargs,
            )
        topk_ids = topk_output.topk_ids.to(torch.int64).contiguous()
        topk_weights = topk_output.topk_weights.to(torch.float32)
        if not experts.should_fuse_routed_scaling_factor_in_topk:
            topk_weights = topk_weights * float(moe.routed_scaling_factor)
        topk_weights = topk_weights.contiguous()
        shared_output = moe._forward_shared_experts(hidden_states)
    else:
        topk_ids = torch.empty(
            (0, top_k), dtype=torch.int64, device=hidden_states.device
        )
        topk_weights = torch.empty(
            (0, top_k), dtype=torch.float32, device=hidden_states.device
        )

    # Collective on the EP group: every rank calls it for every MoE layer of the step, also with 0 tokens.
    if _caller_rows():
        if rows_by_rank is None:
            raise RuntimeError(
                f"{ENV_CALLER_ROWS}=1: the forward batch carries no global_num_tokens_cpu"
            )
    fi_layer = experts.fi_moe_ep_layer
    if rows_by_rank is not None and not _tensors_take_rows():
        tensors = MoEEpTensors(
            hidden_states=hidden_states.contiguous(),
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        # Owned output allocated before staging, as MoEEpMegaLayer.forward does.
        out = torch.empty_like(tensors.hidden_states, dtype=torch.bfloat16)
        rows_kw = (
            {"rows_by_rank": [int(r) for r in rows_by_rank]}
            if _stage_takes_rows_hint()
            else {}
        )
        fi_layer.stage_inputs(
            tensors,
            compile_tokens_per_rank=max(int(r) for r in rows_by_rank),
            **rows_kw,
        )
        routed = fi_layer.compute_staged(output=out)
    else:
        rows_kw = {} if rows_by_rank is None else {"num_tokens_per_rank": rows_by_rank}
        routed = fi_layer.forward(
            MoEEpTensors(
                hidden_states=hidden_states.contiguous(),
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                **rows_kw,
            )
        )
    _after_call(experts, num_tokens)
    if num_tokens == 0:
        return routed
    if shared_output is not None:
        return shared_output.add_(routed)
    return routed


def _after_call(experts, num_tokens: int) -> None:
    _Stats.calls += 1
    _Stats.rows += num_tokens
    _Stats.max_rows = max(_Stats.max_rows, num_tokens)
    _Stats.empty_calls += num_tokens == 0
    kernel = fi_moe_ep_kernel()
    check_every = int(os.environ.get(ENV_CHECK_EVERY, "430"))
    if check_every > 0 and _Stats.calls % check_every == 0:
        fi_layer = experts.fi_moe_ep_layer
        perr = fi_layer._kernel.protocol_error(fi_layer._workspace)
        _Stats.checks += 1
        if perr != 0:
            raise RuntimeError(
                f"{ENV_KERNEL}={kernel}: protocol_error={perr} after {_Stats.calls} layer calls "
                f"(layer {experts.layer_id}); the call's output is invalid"
            )
    log_every = int(os.environ.get(ENV_LOG_EVERY, "4300"))
    if _Stats.calls == 1 or (log_every > 0 and _Stats.calls % log_every == 0):
        logger.info(
            "%s=%s: %d layer calls, %d rows (max %d per call, %d empty calls), protocol_error checks %d (all 0)",
            ENV_KERNEL,
            kernel,
            _Stats.calls,
            _Stats.rows,
            _Stats.max_rows,
            _Stats.empty_calls,
            _Stats.checks,
        )
