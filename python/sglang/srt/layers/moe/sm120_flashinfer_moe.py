"""FlashInfer SM120 Cake MegaMoE transport path with FP8 input staging."""

import os

import torch


def adapt_weights(l1, l2):
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_weights import (
        NOSHARED_SCALE_FILL_WORD,
        TransformedCakeMegaWeights,
        _noshared_operands,
        scale_vector_layout,
    )

    w1, s1 = l1
    w2, s2 = l2
    local, twice_intermediate, packed_hidden = w1.shape
    hidden, intermediate = packed_hidden * 2, twice_intermediate // 2
    if tuple(w2.shape) != (local, hidden, intermediate // 2):
        raise ValueError("Inconsistent SM120 routed weight shapes")
    if any(
        w.dtype not in (torch.uint8, torch.int8) or not w.is_contiguous()
        for w in (w1, w2)
    ):
        raise ValueError("SM120 weights must be contiguous packed E2M1 bytes")

    def scales(value, n, k):
        if value.dtype not in (torch.int32, torch.uint32) or tuple(value.shape) != (
            local,
            k // 128,
            n,
        ):
            raise ValueError("Expected packed K-block-major UE8M0 weight scales")
        filler = torch.full(
            (k // 128, n),
            NOSHARED_SCALE_FILL_WORD,
            dtype=torch.int32,
            device=value.device,
        )
        packed = torch.cat((value.reshape(-1, n).view(torch.int32), filler))
        return scale_vector_layout(packed.view(torch.uint32))

    b1, b2 = _noshared_operands(w1.device, hidden, intermediate)
    return TransformedCakeMegaWeights(
        w1_fp4=w1.view(torch.uint8),
        w1_sfb=scales(s1, twice_intermediate, hidden),
        w2_fp4=w2.view(torch.uint8),
        w2_sfb=scales(s2, hidden, intermediate),
        w1_b8=b1,
        w2_b8=b2,
        hidden_size=hidden,
        intermediate_size=intermediate,
        local_experts=local,
        has_shared_expert=False,
    )


def run(moe, group, x, scales, topk_ids, topk_weights, bucket):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig,
    )

    if not 0 < bucket <= 8192 or x.shape[0] > bucket:
        raise ValueError("SM120 token bucket must bound local rows and be at most 8192")
    layer = getattr(moe.experts, "_sm120_flashinfer_layer", None)
    if layer is None:
        activation_dtype = os.environ.get(
            "SGLANG_FLASHINFER_MEGAMOE_SM120_ACTIVATION_DTYPE", "mxfp8"
        )
        if activation_dtype not in ("mxfp8", "mxfp4"):
            raise ValueError("SM120 routed activation dtype must be mxfp8 or mxfp4")
        activation_config = (
            {"routed_activation_dtype": "mxfp4"}
            if activation_dtype == "mxfp4"
            else {}
        )
        limit = moe.config.swiglu_limit
        limit = 0 if limit is None else limit
        if isinstance(limit, bool) or limit != int(limit):
            raise ValueError("FlashInfer SM120 requires an integral SwiGLU limit")
        weights = adapt_weights(
            moe.experts.mega_l1_weights, moe.experts.mega_l2_weights
        )
        layer = MoEEpLayer(
            bootstrap=BootstrapConfig(
                world_size=group.size(),
                rank=group.rank(),
                device=x.device.index,
                process_group=group,
            ),
            fleet_params=FleetParams(
                num_experts=weights.local_experts * group.size(),
                max_tokens_per_rank=8192,
                token_hidden_size=x.shape[1],
            ),
            weights=None,
            backend=MegaConfig(
                megakernel=Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(
                    **activation_config,
                    intermediate_size=weights.intermediate_size,
                    top_k=topk_ids.shape[1],
                    swiglu_limit=int(limit),
                    transport_table=os.environ.get(
                        "SGLANG_FLASHINFER_MEGAMOE_SM120_TRANSPORT_TABLE",
                        "lsa_where_reachable",
                    ),
                ),
                quantize_input=False,
                preprocess_weights=False,
                transformed_weights=weights,
            ),
        )
        moe.experts._sm120_flashinfer_layer = layer
    inputs = MoEEpTensors(
        hidden_states=x,
        scales=scales.contiguous().view(torch.uint8),
        topk_ids=topk_ids,
        topk_weights=topk_weights,
    )
    layer.stage_inputs(inputs, compile_tokens_per_rank=bucket)
    return layer.compute_staged(output=None)
