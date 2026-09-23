import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.arg_groups.mega_moe_hook import _check_mega_moe_arch
from sglang.srt.layers.moe import mega_moe
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMegaMoeArchValidation(CustomTestCase):

    def test_supported_arches_allowed(self):
        for device_sm in (90, 100, 120):
            _check_mega_moe_arch("megamoe", device_sm)

    def test_unknown_arch_rejected(self):
        for device_sm in (None, 80, 121):
            with self.assertRaises(ValueError):
                _check_mega_moe_arch("megamoe", device_sm)

    def test_other_backends_unaffected(self):
        for backend in ("none", "deepep", "deepep_v2", "flashinfer"):
            _check_mega_moe_arch(backend, 120)

    def test_sm120_selects_routed_api_only_for_fp4_weights(self):
        fp4 = torch.empty(1, dtype=torch.int8)
        fp8 = torch.empty(1, dtype=torch.float8_e4m3fn)
        with mock.patch.object(mega_moe, "_device_sm", 120):
            config = mega_moe._select_mega_moe_arch_config(fp4, fp4)
            self.assertTrue(config.uses_sm120_routed_api)
            self.assertTrue(config.use_dp_max_tokens)
            self.assertIsNone(mega_moe._select_mega_moe_arch_config(fp8, fp8))

    def test_sm90_fp4_selection_is_preserved(self):
        fp4 = torch.empty(1, dtype=torch.int8)
        with mock.patch.object(mega_moe, "_device_sm", 90):
            config = mega_moe._select_mega_moe_arch_config(fp4, fp4)
        self.assertEqual(config.name, "sm90_mxfp4")
        self.assertFalse(config.uses_sm120_routed_api)

    def test_sm120_scale_layout_interleaves_gate_up(self):
        scales = (
            torch.arange(2 * 32, dtype=torch.float32)
            .reshape(2, 32, 1)
            .expand(-1, -1, 4)
            .contiguous()
        )
        captured = {}

        def transform(value, **kwargs):
            captured["value"] = value.clone()
            self.assertEqual(kwargs["recipe"], (1, 32))
            return value[..., :1].to(torch.int32)

        fake_deep_gemm = SimpleNamespace(
            transform_sf_into_required_layout=transform,
        )
        with mock.patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}):
            result = mega_moe._transform_sm120_weight_scales(
                scales,
                mn=32,
                k=128,
                num_groups=2,
                interleave_gate_up=True,
            )

        expected_rows = list(range(8)) + list(range(16, 24))
        expected_rows += list(range(8, 16)) + list(range(24, 32))
        self.assertEqual(captured["value"][0, :, 0].tolist(), expected_rows)
        self.assertEqual(result.shape, (2, 1, 32))
        self.assertTrue(result.is_contiguous())

    def test_sm120_session_workspace_are_reused(self):
        calls = []

        class Session:
            def __init__(self, group, device):
                calls.append(("session", group, device))

        class Workspace:
            def __init__(self, **kwargs):
                calls.append(("workspace", kwargs))

        fake_deep_gemm = SimpleNamespace(
            SM120RoutedMoESession=Session,
            SM120RoutedMoEWorkspace=Workspace,
        )
        group = mock.Mock()
        group.size.return_value = 4
        mega_moe._SM120_ROUTED_MOE_STATE.clear()
        with (
            mock.patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}),
            mock.patch.object(torch.cuda, "current_device", return_value=2),
            mock.patch.object(torch.distributed, "barrier") as barrier,
        ):
            first = mega_moe._get_sm120_routed_moe_state(group)
            second = mega_moe._get_sm120_routed_moe_state(group)

        self.assertIs(first, second)
        self.assertEqual(len(calls), 2)
        barrier.assert_called_once_with(group=group, device_ids=[2])

    def test_sm120_run_aligns_ep_launch(self):
        hidden_states = torch.ones((1, 4), dtype=torch.bfloat16)
        topk_ids = torch.zeros((1, 2), dtype=torch.int64)
        topk_weights = torch.ones((1, 2), dtype=torch.float32)
        experts = SimpleNamespace(
            mega_l1_weights=object(),
            mega_l2_weights=object(),
            should_fuse_routed_scaling_factor_in_topk=True,
        )
        moe = SimpleNamespace(
            config=SimpleNamespace(num_experts_per_tok=2),
            experts=experts,
        )
        output = torch.full((32, 4), 2, dtype=torch.bfloat16)
        events = []

        def launch(*args):
            events.append("launch")
            return output

        fake_deep_gemm = SimpleNamespace(
            fp8_fp4_routed_moe_sm120=mock.Mock(side_effect=launch),
        )
        group = SimpleNamespace(device_group=object())

        with (
            mock.patch.dict(sys.modules, {"deep_gemm": fake_deep_gemm}),
            mock.patch(
                "sglang.kernels.ops.quantization.fp8_kernel."
                "sglang_per_token_group_quant_fp8_ue8m0",
                return_value=(
                    torch.ones((32, 4), dtype=torch.uint8),
                    torch.ones((32, 1), dtype=torch.int32),
                ),
            ) as quantize,
            mock.patch(
                "sglang.srt.distributed.parallel_state.get_moe_ep_group",
                return_value=group,
            ),
            mock.patch.object(
                mega_moe,
                "_get_sm120_routed_moe_state",
                return_value=(object(), object()),
            ),
            mock.patch.object(
                torch.cuda,
                "synchronize",
                side_effect=lambda *_: events.append("synchronize"),
            ) as synchronize,
            mock.patch.object(
                torch.distributed,
                "barrier",
                side_effect=lambda **_: events.append("barrier"),
            ) as barrier,
        ):
            result = mega_moe._run_sm120_routed(
                moe, hidden_states, topk_ids, topk_weights, 32
            )

        self.assertTrue(torch.equal(result, output[:1]))
        self.assertEqual(quantize.call_args.args[0].shape, (32, 4))
        launch = fake_deep_gemm.fp8_fp4_routed_moe_sm120.call_args.args
        self.assertEqual(launch[4].shape, (32, 2))
        self.assertTrue(torch.equal(launch[4][0], topk_ids[0]))
        self.assertTrue(torch.all(launch[4][1:] == -1))
        self.assertTrue(torch.all(launch[5][1:] == 0))
        self.assertEqual(
            synchronize.call_args_list,
            [mock.call(hidden_states.device)],
        )
        barrier.assert_called_once_with(
            group=group.device_group,
            device_ids=[hidden_states.device.index],
        )
        self.assertEqual(events, ["barrier", "synchronize", "launch"])

if __name__ == "__main__":
    unittest.main()
