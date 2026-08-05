from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.eplb.lplb_solver import (
    LPLBSolver,
    assert_lplb_supported_model,
    clear_global_lplb_solvers,
    set_global_lplb_solver,
)
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.utils import get_bool_env_var, is_hip, log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()


def prepare_moe_topk(
    *,
    model,
    model_config: ModelConfig,
    server_args: ServerArgs,
    moe_ep_size: int,
    moe_ep_rank: int,
) -> None:
    balancer_cls = None
    num_prepared = 0
    num_routed_experts = None
    for module in model.modules():
        if not isinstance(module, (TopK, HashTopK)):
            continue
        if not module.enable_waterfill or module.waterfill_balancer is not None:
            continue
        if num_routed_experts is None:
            num_routed_experts = getattr(
                model_config.hf_config, "n_routed_experts", None
            )
            if num_routed_experts is None:
                raise ValueError("Waterfill requires model config n_routed_experts.")
        if balancer_cls is None:
            from sglang.srt.layers.moe.waterfill import WaterfillBalancer

            balancer_cls = WaterfillBalancer
        # Static EPLB remaps TopK ids to physical expert ids before Waterfill.
        # Redundant experts therefore need to be included in the per-rank
        # expert count used for Waterfill's shared-expert slot remapping.
        num_physical_routed_experts = (
            num_routed_experts + server_args.ep_num_redundant_experts
        )
        if isinstance(module, TopK):
            routed_scaling_factor = module.topk_config.routed_scaling_factor
        else:
            routed_scaling_factor = module.routed_scaling_factor
        module.waterfill_balancer = balancer_cls(
            num_routed_experts=num_physical_routed_experts,
            world_size=moe_ep_size,
            rank=moe_ep_rank,
            layer_id=module.layer_id,
            routed_scaling_factor=(
                routed_scaling_factor if routed_scaling_factor is not None else 1.0
            ),
        )
        num_prepared += 1
    if num_prepared:
        log_info_on_rank0(logger, f"Prepared {num_prepared} Waterfill TopK modules.")


def prepare_moe_load_balancer(
    *, model, model_config: ModelConfig, server_args: ServerArgs
):
    """Create one MLB orchestrator with UltraEP configured as its L3 policy."""

    if not server_args.enable_ultraep:
        return None

    try:
        from moe_load_balancer import (
            ExpertLayerStorage,
            ExpertProjectionStorage,
            MoELoadBalancer,
        )
        from moe_load_balancer.policies.l3 import UltraEPL3Policy
    except ImportError as exc:
        raise ImportError(
            "--enable-ultraep requires moe-load-balancer installed with its "
            "UltraEP backend extension (MLB_BUILD_ULTRAEP=1)."
        ) from exc

    num_logical_experts = getattr(model_config.hf_config, "n_routed_experts", None)
    if num_logical_experts is None:
        raise ValueError(
            "UltraEP POC currently supports DeepSeek-style n_routed_experts."
        )

    layer_specs = []
    moe_modules = []
    for module in model.modules():
        if not (
            hasattr(module, "forward_deepep")
            and hasattr(module, "experts")
            and hasattr(module, "layer_id")
        ):
            continue
        if getattr(module, "num_fused_shared_experts", 0) != 0:
            raise ValueError(
                "UltraEP POC requires shared-expert fusion to be disabled."
            )
        experts = module.experts
        assert experts.quant_method is not None
        try:
            replica_storage = experts.quant_method.get_expert_replica_storage(experts)
        except NotImplementedError as exc:
            raise ValueError(
                "UltraEP does not support expert replication for quantization method "
                f"{type(experts.quant_method).__name__}."
            ) from exc
        if replica_storage.auxiliary_tensors:
            unsupported = [name for name, _ in replica_storage.auxiliary_tensors]
            raise ValueError(
                "UltraEP does not yet synchronize these per-expert quantization "
                f"tensors: {unsupported}."
            )
        for projection, weight, scale in (
            ("fc1", replica_storage.w13_weight, replica_storage.w13_weight_scale),
            ("fc2", replica_storage.w2_weight, replica_storage.w2_weight_scale),
        ):
            if weight.element_size() == 1 and scale is None:
                raise ValueError(
                    f"Low-precision UltraEP {projection} weights require per-expert scales."
                )
        layer_specs.append(
            ExpertLayerStorage(
                layer_id=module.layer_id,
                fc1=ExpertProjectionStorage(
                    weight=replica_storage.w13_weight.detach(),
                    weight_scale=(
                        replica_storage.w13_weight_scale.detach()
                        if replica_storage.w13_weight_scale is not None
                        else None
                    ),
                ),
                fc2=ExpertProjectionStorage(
                    weight=replica_storage.w2_weight.detach(),
                    weight_scale=(
                        replica_storage.w2_weight_scale.detach()
                        if replica_storage.w2_weight_scale is not None
                        else None
                    ),
                ),
            )
        )
        moe_modules.append(module)

    if not layer_specs:
        raise ValueError("UltraEP did not find any compatible DeepEP MoE layers.")

    from sglang.srt.distributed import get_moe_ep_group

    ep_group = get_moe_ep_group()
    ultraep_policy = UltraEPL3Policy(
        group=ep_group.device_group,
        layers=layer_specs,
        num_logical_experts=num_logical_experts,
        ep_size=ep_group.world_size,
        num_redundant_experts_per_rank=(
            server_args.ultraep_num_redundant_experts_per_rank
        ),
    )
    load_balancer = MoELoadBalancer(l3_policy=ultraep_policy)
    for module in moe_modules:
        module.moe_load_balancer = load_balancer
    log_info_on_rank0(
        logger,
        f"Prepared one MoELoadBalancer with UltraEP L3 for {len(moe_modules)} MoE layers.",
    )
    return load_balancer


def init_lplb_solvers(*, model_config: ModelConfig) -> None:
    """Initialize per-layer LPLB solvers from current expert location metadata."""
    from sglang.srt.distributed import get_moe_ep_group

    # Gate: refuse LP for non-DeepSeek MoE families whose empty-token paths
    # don't participate in the EP all-reduce (would deadlock under DP-
    # attention). Failure here happens before any forward pass.
    architectures = getattr(model_config.hf_config, "architectures", None)
    if architectures:
        assert_lplb_supported_model(architectures[0])

    metadata = get_global_expert_location_metadata()
    if metadata is None:
        return
    clear_global_lplb_solvers()
    ep_group = get_moe_ep_group()
    for lid in range(metadata.num_layers):
        solver = LPLBSolver(
            phy2log=metadata.physical_to_logical_map[lid],
            log2phy=metadata.logical_to_all_physical_map[lid],
            num_gpus=metadata.ep_size,
            ep_group=ep_group,
            logical_to_all_physical_map_num_valid=(
                metadata.logical_to_all_physical_map_num_valid[lid]
            ),
        )
        set_global_lplb_solver(lid, solver)
    logger.info(f"Initialized LPLB solvers for {metadata.num_layers} layers")


def check_quantized_moe_compatibility(
    *,
    model_config: ModelConfig,
    tp_size: int,
    moe_ep_size: int,
    moe_dp_size: int,
) -> None:
    if (
        quantization_config := getattr(
            model_config.hf_config, "quantization_config", None
        )
    ) is not None and (
        weight_block_size := quantization_config.get("weight_block_size", None)
    ) is not None:
        weight_block_size_n = weight_block_size[0]

        if tp_size % moe_ep_size != 0:
            raise ValueError(
                f"tp_size {tp_size} must be divisible by ep_size {moe_ep_size}"
            )
        moe_tp_size = tp_size // moe_ep_size // moe_dp_size

        moe_intermediate_size = getattr(
            model_config.hf_text_config, "moe_intermediate_size", None
        )
        if moe_intermediate_size is None:
            return

        if moe_intermediate_size % moe_tp_size != 0:
            raise ValueError(
                f"moe_intermediate_size {moe_intermediate_size} must be divisible by moe_tp_size ({moe_tp_size}) which is tp_size ({tp_size}) divided by moe_ep_size ({moe_ep_size})."
            )

        if (
            not envs.SGLANG_SHARED_EXPERT_TP1.get()
            and (moe_intermediate_size // moe_tp_size) % weight_block_size_n != 0
            and not _use_aiter
        ):
            raise ValueError(
                f"For quantized MoE models, please make sure ({moe_intermediate_size=} / {moe_tp_size=}) % {weight_block_size_n=} == 0 "
                f"where moe_tp_size is equal to tp_size ({tp_size}) divided by ep_size ({moe_ep_size}). "
                f"You can fix this by setting arguments `--tp` and `--ep` correctly."
            )
