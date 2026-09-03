from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from sglang.srt.utils import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not is_sm120_supported(),
    reason="DeepEP v2 W4A4 adapter requires SM120",
)


def _quantize_expert_weights(weight: torch.Tensor):
    from deep_gemm.utils import per_token_cast_to_fp4

    groups, n, k = weight.shape
    values = torch.empty((groups, n, k // 2), device="cuda", dtype=torch.int8)
    scales = torch.empty((groups, n, k // 32), device="cuda", dtype=torch.float32)
    for expert in range(groups):
        values[expert], scales[expert] = per_token_cast_to_fp4(
            weight[expert], use_ue8m0=True, gran_k=32
        )
    return values, scales


def test_deepep_v2_prefill_uses_g1_and_g2_w4a4(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_DEEPGEMM_G1_W4A4", "1")
    monkeypatch.setenv("SGLANG_DEEPGEMM_G2_W4A4", "1")

    from deep_gemm.testing import calc_diff
    from deep_gemm.utils import cast_back_from_fp4
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )
    from sglang.srt.layers.moe.moe_runner import deep_gemm as runner_module
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.deep_gemm import (
        DeepGemmMoeQuantInfo,
        DeepGemmRunnerCore,
        pre_permute_deepep_v2_to_deep_gemm,
    )
    from sglang.srt.layers.moe.token_dispatcher.deepep_v2 import (
        DeepEPv2DispatchOutput,
    )

    monkeypatch.setattr(runner_module, "get_tp_group", lambda: None)
    monkeypatch.setattr(runner_module, "is_allocation_symmetric", lambda: False)

    torch.manual_seed(20260902)
    torch.cuda.manual_seed_all(20260902)

    tokens, local_experts, topk = 256, 8, 2
    hidden, intermediate = 1024, 512
    source = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    token_ids = torch.arange(tokens, device="cuda")
    topk_ids = torch.stack(
        (token_ids % local_experts, (token_ids * 3 + 1) % local_experts), dim=1
    ).to(torch.int32)
    topk_weights = torch.full(
        (tokens, topk), 1.0 / topk, device="cuda", dtype=torch.float32
    )

    counts = torch.bincount(topk_ids.flatten().to(torch.int64), minlength=local_experts)
    aligned_counts = (counts.to(torch.int32) + 127) // 128 * 128
    psum = torch.cumsum(aligned_counts, dim=0)
    fp8_values, fp8_scales = sglang_per_token_group_quant_fp8(
        source,
        128,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    w13 = (
        torch.randn(
            (local_experts, 2 * intermediate, hidden),
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    w2 = (
        torch.randn(
            (local_experts, hidden, intermediate),
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    w13_values, w13_scales = _quantize_expert_weights(w13)
    w2_values, w2_scales = _quantize_expert_weights(w2)
    quant_info = DeepGemmMoeQuantInfo(
        w13_weight=w13_values,
        w2_weight=w2_values,
        use_fp8=True,
        w13_scale=w13_scales,
        w2_scale=w2_scales,
        block_shape=[1, 32],
        is_fp4_experts=True,
    )
    config = MoeRunnerConfig(
        num_experts=local_experts * 4,
        num_local_experts=local_experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        top_k=topk,
        activation="silu",
        is_gated=True,
        swiglu_limit=10.0,
    )
    dispatch_output = DeepEPv2DispatchOutput(
        fp8_values,
        fp8_scales,
        topk_ids,
        topk_weights,
        counts.tolist(),
        psum,
        False,
        True,
        False,
        0,
        0,
        0,
        128,
    )

    state = {}
    runner_input = pre_permute_deepep_v2_to_deep_gemm(
        dispatch_output, quant_info, config, state
    )
    assert runner_input.g1_act_fp4
    assert runner_input.hidden_states.dtype == torch.int8
    assert runner_input.hidden_states.shape[-1] == hidden // 2
    assert runner_input.m_indices.shape == (int(psum[-1]),)

    scattered = runner_input.hidden_states[state["output_index"].long()]
    scattered_scale = runner_input.hidden_states_scale[state["output_index"].long()]
    restored = cast_back_from_fp4(
        scattered.reshape(-1, hidden // 2),
        scattered_scale.reshape(-1, hidden // 128),
        gran_k=32,
        use_packed_ue8m0=True,
    ).reshape(tokens, topk, hidden)
    source_per_slot = source[:, None, :].expand_as(restored)
    assert float(calc_diff(restored, source_per_slot)) < 0.04

    output = (
        DeepGemmRunnerCore(config)
        .run(runner_input, quant_info, state)
        .hidden_states[state["output_index"].long()]
    )
    reference = torch.empty_like(output)
    for token in range(tokens):
        for slot in range(topk):
            expert = int(topk_ids[token, slot])
            gateup = source[token].float() @ w13[expert].float().T
            gate, up = gateup.chunk(2)
            activated = F.silu(gate.clamp(max=10.0)) * up.clamp(min=-10.0, max=10.0)
            reference[token, slot] = (activated @ w2[expert].float().T).to(
                torch.bfloat16
            )

    torch.cuda.synchronize()
    # Covers DeepEP MXFP8 dispatch, direct G1 requantization, both W4A4
    # grouped GEMMs, and the fused G2 clamp-SwiGLU quantizer.
    assert float(calc_diff(output, reference)) < 0.06


def test_deepep_v2_expanded_decode_keeps_w4a8_activations(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_DEEPGEMM_G1_W4A4", "1")
    monkeypatch.setenv("SGLANG_DEEPGEMM_G2_W4A4", "1")

    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.deep_gemm import (
        DeepGemmMoeQuantInfo,
        pre_permute_deepep_v2_to_deep_gemm,
    )
    from sglang.srt.layers.moe.token_dispatcher.deepep_v2 import (
        DeepEPv2DispatchOutput,
    )

    local_experts, hidden, alignment, max_m = 4, 1024, 128, 16
    total_expanded = 384
    expanded = torch.zeros(
        (total_expanded, hidden), device="cuda", dtype=torch.float8_e4m3fn
    )
    scales = torch.ones(
        (total_expanded, hidden // 128), device="cuda", dtype=torch.float32
    )
    psum = torch.tensor([3, 128, 133, 257], device="cuda", dtype=torch.int32)
    dispatch_output = DeepEPv2DispatchOutput(
        expanded,
        scales,
        None,
        torch.ones(total_expanded, device="cuda", dtype=torch.float32),
        [],
        psum,
        True,
        False,
        True,
        4,
        max_m,
        total_expanded,
        alignment,
    )
    empty_weight = torch.empty((local_experts, 0, 0), device="cuda", dtype=torch.int8)
    quant_info = DeepGemmMoeQuantInfo(
        w13_weight=empty_weight,
        w2_weight=empty_weight,
        use_fp8=True,
        block_shape=[1, 32],
        is_fp4_experts=True,
    )
    config = MoeRunnerConfig(
        num_experts=local_experts * 4,
        num_local_experts=local_experts,
        hidden_size=hidden,
        intermediate_size_per_partition=512,
        top_k=6,
        activation="silu",
        is_gated=True,
        swiglu_limit=10.0,
    )

    runner_input = pre_permute_deepep_v2_to_deep_gemm(
        dispatch_output, quant_info, config, {}
    )

    assert runner_input.use_masked_gemm
    assert not runner_input.g1_act_fp4
    assert runner_input.hidden_states.shape == (local_experts, max_m, hidden)
    assert runner_input.hidden_states_scale.shape == (
        local_experts,
        max_m,
        hidden // 128,
    )
