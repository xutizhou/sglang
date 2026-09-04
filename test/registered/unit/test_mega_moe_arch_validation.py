"""MegaMoE must be rejected at startup on architectures with no kernel."""

import unittest

from sglang.srt.arg_groups.mega_moe_hook import _check_mega_moe_arch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMegaMoeArchValidation(CustomTestCase):

    def test_sm120_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            _check_mega_moe_arch("megamoe", 120)
        self.assertIn("SM120", str(ctx.exception))

    def test_unknown_arch_rejected(self):
        # get_device_sm() returns None off CUDA.
        with self.assertRaises(ValueError):
            _check_mega_moe_arch("megamoe", None)

    def test_supported_arches_allowed(self):
        _check_mega_moe_arch("megamoe", 90)
        _check_mega_moe_arch("megamoe", 100)

    def test_other_backends_unaffected(self):
        # Must not fire for backends other than megamoe.
        for backend in ("none", "deepep", "deepep_v2", "flashinfer"):
            _check_mega_moe_arch(backend, 120)


if __name__ == "__main__":
    unittest.main()
