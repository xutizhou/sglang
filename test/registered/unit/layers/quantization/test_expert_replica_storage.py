from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _SemanticQuantMethod(FusedMoEMethodBase):
    def __init__(self, quant_info):
        self.quant_info = quant_info

    def apply(self, layer, dispatch_output):
        raise NotImplementedError

    def get_triton_quant_info(self, layer):
        return self.quant_info


def test_replica_storage_uses_quant_method_semantics_not_layer_attribute_names():
    w13 = torch.zeros((2, 4))
    w2 = torch.zeros((2, 4))
    s13 = torch.ones((2, 1))
    s2 = torch.ones((2, 1))
    method = _SemanticQuantMethod(
        TritonMoeQuantInfo(
            w13_weight=w13,
            w2_weight=w2,
            w13_scale=s13,
            w2_scale=s2,
        )
    )

    layer = SimpleNamespace(
        num_local_experts=2,
        named_per_expert_tensors=lambda _: [
            ("backend_specific_fc1_scale_name", s13),
            ("backend_specific_fc2_scale_name", s2),
        ],
    )
    storage = method.get_expert_replica_storage(layer)

    assert storage.w13_weight is w13
    assert storage.w2_weight is w2
    assert storage.w13_weight_scale is s13
    assert storage.w2_weight_scale is s2
    assert storage.auxiliary_tensors == ()


def test_replica_storage_reports_per_expert_state_the_backend_cannot_sync():
    zero_points = torch.zeros((2, 1))
    method = _SemanticQuantMethod(
        TritonMoeQuantInfo(
            w13_weight=torch.zeros((2, 4)),
            w2_weight=torch.zeros((2, 4)),
            w13_zp=zero_points,
        )
    )

    storage = method.get_expert_replica_storage(SimpleNamespace(num_local_experts=2))

    assert len(storage.auxiliary_tensors) == 1
    name, tensor = storage.auxiliary_tensors[0]
    assert name == "w13_zp"
    assert tensor is zero_points


def test_replica_storage_does_not_silently_drop_unmodeled_side_tensors():
    extra = torch.zeros((2, 1))
    method = _SemanticQuantMethod(
        TritonMoeQuantInfo(
            w13_weight=torch.zeros((2, 4)),
            w2_weight=torch.zeros((2, 4)),
        )
    )
    layer = SimpleNamespace(
        num_local_experts=2,
        named_per_expert_tensors=lambda _: [("backend_alpha", extra)],
    )

    storage = method.get_expert_replica_storage(layer)

    assert storage.auxiliary_tensors[0][0] == "backend_alpha"
    assert storage.auxiliary_tensors[0][1] is extra
