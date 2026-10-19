"""Unit tests for fused shared-expert weight scaling.

These tests pin the fused shared expert's topk weight contract on three paths:

  * aiter (HIP) per-rank-slot path: routed_scaling_factor is folded into the
    routed weights and the post-MoE multiply is skipped, so the shared weight
    must be 1.0 for a net 1.0x contribution.
  * post-MoE scaling per-rank-slot path (default): the whole MoE output is
    multiplied by routed_scaling_factor afterward, so the shared weight must
    be 1/rsf.
  * standard EP path (no per-rank slots): every rank computes the fused shared
    expert and the outputs are all-reduced, so the model-supplied 1/ep_size
    factor must be applied to the shared weight.
  * Waterfill: the shared slot appended by WaterfillBalancer follows the same
    rule as the per-rank slot -- 1/rsf when the MoE epilogue scales the output
    afterwards, 1.0 when the router already folded routed_scaling_factor into
    the routed weights (should_fuse_routed_scaling_factor_in_topk quant
    methods such as ModelOpt NVFP4 on flashinfer_cutedsl, aiter routes).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.topk import TopK, TopKConfig
from sglang.srt.model_executor.model_runner_components import moe_ep_setup
from sglang.test.test_utils import CustomTestCase


class TestFusedSharedExpertScaling(CustomTestCase):
    # 256 physical routed experts over ep_size=8 -> 32 local routed per rank.
    NUM_PHYSICAL_ROUTED = 256
    EP_SIZE = 8
    EP_RANK = 0
    ROUTED_SCALING_FACTOR = 2.5

    def _run_remap(self, *, use_aiter):
        # Layout: [routed, routed, routed, shared]; the trailing column is the
        # fused shared expert. The shared weight starts at a sentinel to prove
        # the function overwrites it.
        topk_ids = torch.tensor([[5, 40, 100, 999]], dtype=torch.int32)
        routed_weights = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float32)
        topk_weights = torch.tensor([[1.0, 0.5, 0.25, -123.0]], dtype=torch.float32)

        topk_config = TopKConfig(
            top_k=4,
            num_fused_shared_experts=1,
            routed_scaling_factor=self.ROUTED_SCALING_FACTOR,
        )

        with (
            patch.object(topk_module, "_use_aiter", use_aiter),
            patch.object(
                topk_module,
                "get_parallel",
                return_value=SimpleNamespace(
                    moe_ep_size=self.EP_SIZE, moe_ep_rank=self.EP_RANK
                ),
            ),
        ):
            _out_ids, out_weights = topk_module.remap_topk_for_per_rank_shared_slots(
                topk_ids.clone(),
                topk_weights.clone(),
                num_fused_shared_experts=1,
                num_physical_routed_experts=self.NUM_PHYSICAL_ROUTED,
                topk_config=topk_config,
            )

        # Routed weights must never be touched by the shared-weight override.
        self.assertTrue(torch.equal(out_weights[0, :-1], routed_weights))
        return out_weights[0, -1].item()

    def test_aiter_path_uses_unit_shared_weight(self):
        # routed_scaling_factor is folded into the routed weights and the
        # post-MoE multiply is skipped -> shared weight must be 1.0, NOT 1/rsf.
        shared_weight = self._run_remap(use_aiter=True)
        self.assertAlmostEqual(shared_weight, 1.0)

    def test_post_moe_scaling_path_compensates_with_inverse_rsf(self):
        # Default path: the whole MoE output is multiplied by rsf afterward, so
        # the shared weight must be 1/rsf to net out to 1.0.
        shared_weight = self._run_remap(use_aiter=False)
        self.assertAlmostEqual(shared_weight, 1.0 / self.ROUTED_SCALING_FACTOR)

    def _run_post_process_standard_path(self, *, scaling_factor):
        topk_ids = torch.tensor([[5, 40, 100, 256]], dtype=torch.int32)
        topk_weights = torch.tensor([[1.0, 0.5, 0.25, 1.0]], dtype=torch.float32)
        topk_config = TopKConfig(
            top_k=4,
            num_fused_shared_experts=1,
            fused_shared_experts_scaling_factor=scaling_factor,
            allow_routed_experts_capture=False,
        )
        router_logits = torch.zeros((1, 256), dtype=torch.float32)
        with (
            patch.object(topk_module, "_is_cuda", False),
            patch.object(topk_module, "_is_hip", False),
            patch.object(topk_module, "_use_aiter", False),
            patch.object(
                topk_module, "has_per_rank_fused_shared_slots", return_value=False
            ),
        ):
            _out_ids, out_weights, _recorder_ids = topk_module._post_process_topk_ids(
                topk_ids.clone(),
                topk_weights.clone(),
                topk_config,
                router_logits,
                layer_id=0,
            )
        self.assertTrue(torch.equal(out_weights[0, :-1], topk_weights[0, :-1]))
        return out_weights[0, -1].item()

    def test_standard_ep_path_applies_shared_scaling_factor(self):
        # Regression: models pass 1/ep_size under standard EP (every rank
        # computes the fused shared expert and outputs are all-reduced), but
        # the standard CUDA post-process dropped the factor, so the shared
        # contribution was summed ep_size times (corrupt EP4 output on
        # BailingMoeV3, BF16 and FP8 alike).
        shared_weight = self._run_post_process_standard_path(scaling_factor=0.25)
        self.assertAlmostEqual(shared_weight, 0.25)

    def test_standard_path_without_factor_keeps_shared_weight(self):
        # TP mode passes no factor; the shared weight must pass through
        # unscaled (guards the predicate against degrading to always-scale).
        shared_weight = self._run_post_process_standard_path(scaling_factor=None)
        self.assertAlmostEqual(shared_weight, 1.0)

    def test_shared_expert_ids_route_to_home_rank(self):
        # Sanity: the shared slot id is placed at this rank's interleaved
        # position (ep_rank * num_local_experts + num_local_routed).
        topk_ids = torch.tensor([[5, 40, 100, 999]], dtype=torch.int32)
        topk_weights = torch.tensor([[1.0, 0.5, 0.25, 0.0]], dtype=torch.float32)
        topk_config = TopKConfig(
            top_k=4,
            num_fused_shared_experts=1,
            routed_scaling_factor=self.ROUTED_SCALING_FACTOR,
        )
        with (
            patch.object(topk_module, "_use_aiter", True),
            patch.object(
                topk_module,
                "get_parallel",
                return_value=SimpleNamespace(
                    moe_ep_size=self.EP_SIZE, moe_ep_rank=self.EP_RANK
                ),
            ),
        ):
            out_ids, _ = topk_module.remap_topk_for_per_rank_shared_slots(
                topk_ids.clone(),
                topk_weights.clone(),
                num_fused_shared_experts=1,
                num_physical_routed_experts=self.NUM_PHYSICAL_ROUTED,
                topk_config=topk_config,
            )
        num_local_routed = self.NUM_PHYSICAL_ROUTED // self.EP_SIZE  # 32
        num_local_experts = num_local_routed + 1  # 33
        expected_shared_id = self.EP_RANK * num_local_experts + num_local_routed
        self.assertEqual(out_ids[0, -1].item(), expected_shared_id)

    def test_glm53_idle_rank_preserves_routed_topk_contract(self):
        """An idle rank must return the pre-append top-8 shape; returning top-9
        here would make the AITER postprocessor append shared ID 288 twice."""
        topk = TopK.__new__(TopK)
        torch.nn.Module.__init__(topk)
        topk.topk_config = TopKConfig(top_k=9, num_fused_shared_experts=1)
        topk.enable_waterfill = False
        topk.waterfill_balancer = None

        with (
            patch.object(
                topk_module,
                "get_parallel",
                return_value=SimpleNamespace(tp_group=None),
            ),
            patch.object(topk_module, "is_allocation_symmetric", return_value=False),
            patch.object(
                topk_module,
                "use_symmetric_memory",
                side_effect=lambda *_args, **_kwargs: nullcontext(),
            ),
            patch.object(
                topk_module, "has_per_rank_fused_shared_slots", return_value=False
            ),
        ):
            output = topk.empty_topk_output(torch.device("cpu"))

        self.assertEqual(output.topk_ids.shape, (0, 8))
        self.assertEqual(output.topk_weights.shape, (0, 8))


class TestWaterfillSharedWeight(CustomTestCase):
    """WaterfillBalancer's shared slot must count the shared expert exactly
    once, whichever side of the experts routed_scaling_factor is applied on."""

    ROUTED_SCALING_FACTOR = 2.5
    NUM_ROUTED_EXPERTS = 256
    EP_SIZE = 8

    def _make_topk(self, **config_kwargs) -> TopK:
        topk = TopK.__new__(TopK)
        torch.nn.Module.__init__(topk)
        topk.layer_id = 3
        topk.enable_waterfill = True
        topk.waterfill_balancer = None
        topk.topk_config = TopKConfig(
            top_k=8,
            num_fused_shared_experts=0,
            routed_scaling_factor=self.ROUTED_SCALING_FACTOR,
            **config_kwargs,
        )
        return topk

    def _make_hash_topk(self, *, apply_routed_scaling_factor_on_output) -> HashTopK:
        topk = HashTopK.__new__(HashTopK)
        torch.nn.Module.__init__(topk)
        topk.layer_id = 3
        topk.enable_waterfill = True
        topk.waterfill_balancer = None
        topk.routed_scaling_factor = self.ROUTED_SCALING_FACTOR
        topk.apply_routed_scaling_factor_on_output = (
            apply_routed_scaling_factor_on_output
        )
        return topk

    def _prepare(self, module, *, use_aiter=False):
        model = torch.nn.Module()
        model.topk = module
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(n_routed_experts=self.NUM_ROUTED_EXPERTS)
        )
        with (
            patch.object(topk_module, "_use_aiter", use_aiter),
            patch.object(
                moe_ep_setup,
                "get_parallel",
                return_value=SimpleNamespace(moe_ep_size=self.EP_SIZE, moe_ep_rank=1),
            ),
            patch.object(
                moe_ep_setup,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(ep_num_redundant_experts=0)
                ),
            ),
        ):
            moe_ep_setup.prepare_moe_topk(model=model, model_config=model_config)
        self.assertIsNotNone(module.waterfill_balancer)
        self.assertEqual(
            module.waterfill_balancer.num_routed_experts, self.NUM_ROUTED_EXPERTS
        )
        return module.waterfill_balancer.shared_weight

    def test_post_moe_scaling_uses_inverse_rsf(self):
        # Default CUDA path (e.g. FP8 + DeepEP + deep_gemm): forward_deepep
        # multiplies the combined MoE output by rsf afterwards, so the shared
        # slot must carry 1/rsf.
        topk = self._make_topk(use_grouped_topk=True, correction_bias=torch.zeros(1))
        self.assertAlmostEqual(self._prepare(topk), 1.0 / self.ROUTED_SCALING_FACTOR)

    def test_folded_rsf_uses_unit_weight(self):
        # should_fuse_routed_scaling_factor_in_topk quant methods (ModelOpt
        # NVFP4 on flashinfer_cutedsl, cutlass / trtllm-routed FP8): the routed
        # weights already carry rsf and the epilogue multiply is skipped, so a
        # 1/rsf shared slot would shrink the shared expert to 0.4x.
        topk = self._make_topk(
            use_grouped_topk=True,
            correction_bias=torch.zeros(1),
            apply_routed_scaling_factor_on_output=True,
        )
        self.assertAlmostEqual(self._prepare(topk), 1.0)

    def test_aiter_biased_grouped_routing_uses_unit_weight(self):
        # aiter_biased_grouped_topk folds rsf regardless of the flag and
        # forward_deepep skips the multiply on aiter.
        topk = self._make_topk(use_grouped_topk=True, correction_bias=torch.zeros(1))
        self.assertAlmostEqual(self._prepare(topk, use_aiter=True), 1.0)

    def test_aiter_unbiased_grouped_routing_keeps_inverse_rsf(self):
        # Grouped routing without a correction bias does not fold on aiter.
        topk = self._make_topk(use_grouped_topk=True, correction_bias=None)
        self.assertAlmostEqual(
            self._prepare(topk, use_aiter=True), 1.0 / self.ROUTED_SCALING_FACTOR
        )

    def test_hash_topk_follows_its_own_fold_flag(self):
        self.assertAlmostEqual(
            self._prepare(
                self._make_hash_topk(apply_routed_scaling_factor_on_output=False)
            ),
            1.0 / self.ROUTED_SCALING_FACTOR,
        )
        self.assertAlmostEqual(
            self._prepare(
                self._make_hash_topk(apply_routed_scaling_factor_on_output=True)
            ),
            1.0,
        )

    def test_redundant_experts_extend_physical_count(self):
        topk = self._make_topk(use_grouped_topk=True, correction_bias=torch.zeros(1))
        model = torch.nn.Module()
        model.topk = topk
        model_config = SimpleNamespace(
            hf_config=SimpleNamespace(n_routed_experts=self.NUM_ROUTED_EXPERTS)
        )
        with (
            patch.object(topk_module, "_use_aiter", False),
            patch.object(
                moe_ep_setup,
                "get_parallel",
                return_value=SimpleNamespace(moe_ep_size=self.EP_SIZE, moe_ep_rank=1),
            ),
            patch.object(
                moe_ep_setup,
                "get_exec",
                return_value=SimpleNamespace(
                    moe=SimpleNamespace(ep_num_redundant_experts=32)
                ),
            ),
        ):
            moe_ep_setup.prepare_moe_topk(model=model, model_config=model_config)
        self.assertEqual(
            topk.waterfill_balancer.num_routed_experts, self.NUM_ROUTED_EXPERTS + 32
        )
        self.assertEqual(topk.waterfill_balancer.old_experts_per_rank, 36)

    def test_fused_shared_expert_weight_helper(self):
        fw = topk_module.fused_shared_expert_weight
        self.assertAlmostEqual(fw(routed_scaling_factor=2.5, fused_in_topk=False), 0.4)
        self.assertAlmostEqual(fw(routed_scaling_factor=2.5, fused_in_topk=True), 1.0)
        # No factor at all: nothing to compensate for on either path.
        self.assertAlmostEqual(fw(routed_scaling_factor=None, fused_in_topk=False), 1.0)
        self.assertAlmostEqual(fw(routed_scaling_factor=0.0, fused_in_topk=False), 1.0)


if __name__ == "__main__":
    unittest.main()
