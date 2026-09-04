from __future__ import annotations

import pytest
import torch

from sglang.srt.utils import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=90,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
)

pytestmark = pytest.mark.skipif(
    not is_sm120_supported(),
    reason="DSV4 MXFP4 activation kernels require SM120",
)


@pytest.mark.parametrize("num_tokens", [0, 1, 17, 513])
def test_deepep_mxfp8_to_mxfp4_requantization(num_tokens: int) -> None:
    from deep_gemm.testing import calc_diff
    from deep_gemm.utils import cast_back_from_fp4

    from sglang.kernels.ops.attention.dsv4 import mxfp8_to_mxfp4
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    torch.manual_seed(42)
    logical_k = 7168
    source = torch.randn((num_tokens, logical_k), device="cuda", dtype=torch.bfloat16)
    if num_tokens:
        fp8_values, fp8_scale = sglang_per_token_group_quant_fp8(
            source,
            128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
    else:
        # Preserve the TMA-major (M, K/512) packed-scale stride contract for
        # the valid DeepEP case where a rank receives no routed tokens.
        fp8_values = torch.empty_like(source, dtype=torch.float8_e4m3fn)
        fp8_scale = torch.empty(
            (logical_k // 512, 1), device="cuda", dtype=torch.int32
        )[:, :0].T

    fp4_values = torch.empty(
        (num_tokens, logical_k // 2), device="cuda", dtype=torch.int8
    )
    fp4_scale = torch.empty(
        (num_tokens, logical_k // 128), device="cuda", dtype=torch.int32
    )

    mxfp8_to_mxfp4(fp8_values, fp8_scale, fp4_values, fp4_scale)
    torch.cuda.synchronize()

    assert fp4_values.shape == (num_tokens, logical_k // 2)
    assert fp4_values.dtype == torch.int8
    if num_tokens:
        restored = cast_back_from_fp4(
            fp4_values,
            fp4_scale,
            gran_k=32,
            use_packed_ue8m0=True,
        )
        assert float(calc_diff(restored, source)) < 0.04


@pytest.mark.parametrize("num_tokens", [1, 17, 513])
@pytest.mark.parametrize("hidden_dim", [256, 512, 7168])
@pytest.mark.parametrize("swiglu_limit", [None, 10.0])
def test_silu_and_mul_contig_fp4_matches_reference(
    num_tokens: int, hidden_dim: int, swiglu_limit: float | None
) -> None:
    from deep_gemm.testing import calc_diff
    from deep_gemm.utils import cast_back_from_fp4
    from deep_gemm.utils.layout import get_tma_aligned_size

    from sglang.kernels.ops.attention.dsv4 import silu_and_mul_contig_fp4_post_quant

    torch.manual_seed(20260902)
    source = (
        torch.randn(
            (num_tokens, hidden_dim * 2),
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 3
    )
    values = torch.empty((num_tokens, hidden_dim // 2), device="cuda", dtype=torch.int8)
    m_padded = get_tma_aligned_size(num_tokens, 4)
    scale_storage = torch.empty(
        (hidden_dim // 128, m_padded), device="cuda", dtype=torch.int32
    )
    scales = scale_storage[:, :num_tokens].T

    silu_and_mul_contig_fp4_post_quant(
        input=source,
        output=values,
        output_scale=scales,
        swiglu_limit=swiglu_limit,
    )
    actual = cast_back_from_fp4(
        values,
        # DeepGEMM consumes the TMA-major view directly; its reference helper
        # expects row-major packed int32 scales.
        scales.contiguous(),
        gran_k=32,
        use_packed_ue8m0=True,
    )

    gate, up = source.float().chunk(2, dim=1)
    if swiglu_limit is not None:
        gate.clamp_(max=swiglu_limit)
        up.clamp_(min=-swiglu_limit, max=swiglu_limit)
    reference = torch.nn.functional.silu(gate) * up
    torch.cuda.synchronize()

    assert float(calc_diff(actual, reference)) < 0.04
