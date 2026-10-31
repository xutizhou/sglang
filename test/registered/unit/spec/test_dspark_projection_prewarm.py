"""Unit tests for prewarm_target_hidden_projection: the DSpark Decode compiles
the eager-only target-hidden projection at startup instead of on the first
unfolded verify commit while serving (2026-10-31: the DeepGEMM pre-compile
blocked every DP-attention scheduler for ~80 s)."""

import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.speculative.dspark_components.dspark_kv_inject import (
    prewarm_target_hidden_projection,
)


class _Draft:
    def __init__(self, input_size, with_proj=True):
        stage0 = SimpleNamespace()
        if with_proj:
            stage0.main_proj = SimpleNamespace(input_size=input_size)
        self.stages = [stage0, SimpleNamespace()]
        self.calls = []

    def project_target_hidden(self, main_hidden):
        self.calls.append(
            (
                tuple(main_hidden.shape),
                main_hidden.dtype,
                main_hidden.device,
                torch.is_inference_mode_enabled(),
                bool(main_hidden.eq(0).all()),
            )
        )
        return main_hidden


class TestPrewarmTargetHiddenProjection(CustomTestCase):
    def test_runs_projection_once_with_one_row(self):
        draft = _Draft(input_size=3 * 64)
        ran = prewarm_target_hidden_projection(
            draft, device=torch.device("cpu"), dtype=torch.bfloat16
        )
        self.assertTrue(ran)
        self.assertEqual(
            draft.calls,
            [((1, 192), torch.bfloat16, torch.device("cpu"), True, True)],
        )

    def test_skips_drafts_without_projection(self):
        no_proj = _Draft(input_size=192, with_proj=False)
        self.assertFalse(
            prewarm_target_hidden_projection(
                no_proj, device=torch.device("cpu"), dtype=torch.bfloat16
            )
        )
        self.assertEqual(no_proj.calls, [])
        for draft in (SimpleNamespace(), SimpleNamespace(stages=[])):
            self.assertFalse(
                prewarm_target_hidden_projection(
                    draft, device=torch.device("cpu"), dtype=torch.bfloat16
                )
            )


if __name__ == "__main__":
    unittest.main()
