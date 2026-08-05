# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/v0.5.5/vllm/model_executor/layers/quantization/base_config.py
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import torch
from torch import nn

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
    from sglang.srt.models.utils import WeightsMapper


@dataclass(frozen=True)
class ExpertReplicaStorage:
    """Semantic view of the tensors required to execute replicated experts."""

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_weight_scale: Optional[torch.Tensor] = None
    w2_weight_scale: Optional[torch.Tensor] = None
    auxiliary_tensors: tuple[tuple[str, torch.Tensor], ...] = ()


class QuantizeMethodBase(ABC):
    """Base class for different quantized methods."""

    def create_weights(
        self, layer: torch.nn.Module, *weight_args, **extra_weight_attrs
    ):
        """Create weights for a layer.

        The weights will be set as attributes of the layer."""
        raise NotImplementedError()

    @abstractmethod
    def apply(self, layer: torch.nn.Module, *args, **kwargs) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.

        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError()

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        """Process the weight after loading.

        This can be used for example, to transpose weights for computation.
        """
        return


class LinearMethodBase(QuantizeMethodBase):
    """Base class for different (maybe quantized) linear methods."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for a linear layer.
           The weights will be set as attributes of the layer.

        Args:
            layer: The layer that is using the LinearMethodBase factory.
            input_size_per_partition: Size of the weight input dim on rank X.
            output_partition_sizes: Sizes of the output dim of each logical
                weight on rank X. E.g., output_partition_sizes for QKVLinear
                is a list contains the width of Wq, Wk, Wv on rank X.
            input_size: Size of the input dim of the weight across all ranks.
            output_size: Size of the output dim of the weight across all ranks.
            params_dtype: Datatype of the parameters.
        """
        raise NotImplementedError()

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the weights in layer to the input tensor.
        Expects create_weights to have been called before on the layer."""
        raise NotImplementedError()


class FusedMoEMethodBase(QuantizeMethodBase):

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        raise NotImplementedError

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        raise NotImplementedError

    def get_triton_quant_info(self, layer: torch.nn.Module) -> TritonMoeQuantInfo:
        """Return a ``TritonMoeQuantInfo`` describing the quantisation state
        stored on *layer*.

        The LoRA MoE runner calls this so that ``invoke_fused_moe_kernel``
        receives the correct flags / scales / block-shape for the base
        weights.  Each quantisation method must override this with the
        same construction it already uses inside ``apply()``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_triton_quant_info()"
        )

    def get_expert_replica_storage(
        self, layer: torch.nn.Module
    ) -> ExpertReplicaStorage:
        """Return the expert state that a dynamic replication policy must move.

        Quantization methods remain the source of truth for their weight and
        scale tensors. The default implementation reuses their existing
        semantic ``TritonMoeQuantInfo`` instead of inspecting layer attribute
        names. Methods with a different representation can override this
        method directly.
        """

        quant_info = self.get_triton_quant_info(layer)
        num_local_experts = layer.num_local_experts
        w13_weight = quant_info.w13_weight
        w2_weight = quant_info.w2_weight
        w13_weight_scale = getattr(quant_info, "w13_scale", None)
        w2_weight_scale = getattr(quant_info, "w2_scale", None)
        declared = tuple(
            tensor
            for tensor in (w13_weight, w2_weight, w13_weight_scale, w2_weight_scale)
            if isinstance(tensor, torch.Tensor)
        )

        auxiliary_tensors = {}
        for name in ("b13", "b2", "w13_zp", "w2_zp"):
            tensor = getattr(quant_info, name, None)
            if (
                isinstance(tensor, torch.Tensor)
                and tensor.ndim > 0
                and tensor.shape[0] == num_local_experts
            ):
                auxiliary_tensors[name] = tensor

        named_tensors = getattr(layer, "named_per_expert_tensors", None)
        if named_tensors is not None:
            for name, tensor in named_tensors(num_local_experts):
                if not any(
                    tensor.shape == known.shape
                    and tensor.dtype == known.dtype
                    and tensor.device == known.device
                    and tensor.data_ptr() == known.data_ptr()
                    for known in declared
                ):
                    auxiliary_tensors[name] = tensor

        return ExpertReplicaStorage(
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            w13_weight_scale=w13_weight_scale,
            w2_weight_scale=w2_weight_scale,
            auxiliary_tensors=tuple(sorted(auxiliary_tensors.items())),
        )


class QuantizationConfig(ABC):
    """Base class for quantization configs."""

    def __init__(self):
        super().__init__()
        # mapping is updated by models as they initialize
        self.packed_modules_mapping: Dict[str, List[str]] = dict()

    def update_packed_modules_mapping(self, mapping: Dict[str, List[str]]) -> None:
        self.packed_modules_mapping = mapping

    @abstractmethod
    def get_name(self) -> str:
        """Name of the quantization method."""
        raise NotImplementedError()

    @abstractmethod
    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        """List of supported activation dtypes."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        """Minimum GPU capability to support the quantization method.

        E.g., 70 for Volta, 75 for Turing, 80 for Ampere.
        This requirement is due to the custom CUDA kernels used by the
        quantization method.
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def get_config_filenames() -> List[str]:
        """List of filenames to search for in the model directory."""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> QuantizationConfig:
        """Create a config class from the model's quantization config."""
        raise NotImplementedError()

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        """
        Detects if this quantization method can support a given checkpoint
        format by overriding the user specified quantization method --
        this method should only be overwritten by subclasses in exceptional
        circumstances
        """
        return None

    @classmethod
    def _modelopt_override_quantization_method(
        cls, hf_quant_config, user_quant
    ) -> Optional[str]:
        """Shared ModelOpt quantization method override logic."""
        if hf_quant_config is None:
            return None

        # Check if this is a ModelOpt config
        quant_algo = hf_quant_config.get("quant_algo", "").upper()

        # If user specified generic "modelopt", auto-detect the specific method
        if user_quant == "modelopt":
            if "FP8" in quant_algo:
                return "modelopt_fp8"
            elif "NVFP4" in quant_algo or "FP4" in quant_algo:
                return "modelopt_fp4"

        # The hf_quant_config may be a parsed quant config, so we need to check the
        # quant_method.
        if hf_quant_config.get("quant_method", "") == "modelopt_fp8":
            return "modelopt_fp8"
        elif hf_quant_config.get("quant_method", "") == "modelopt_fp4":
            return "modelopt_fp4"

        return None

    @staticmethod
    def get_from_keys(config: Dict[str, Any], keys: List[str]) -> Any:
        """Get a value from the model's quantization config."""
        for key in keys:
            if key in config:
                return config[key]
        raise ValueError(
            f"Cannot find any of {keys} in the model's " "quantization config."
        )

    @staticmethod
    def get_from_keys_or(config: Dict[str, Any], keys: List[str], default: Any) -> Any:
        """Get a optional value from the model's quantization config."""
        try:
            return QuantizationConfig.get_from_keys(config, keys)
        except ValueError:
            return default

    @abstractmethod
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        """Get the quantize method to use for the quantized layer.

        Args:
            layer: The layer for the quant method.
            prefix: The full name of the layer in the state dict
        Returns:
            The quantize method. None if the given layer doesn't support quant
            method.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_scaled_act_names(self) -> List[str]:
        """Returns the activation function names that should be post-scaled.

        For now, this is only used by AWQ.
        """
        raise NotImplementedError()

    def apply_weight_name_mapper(
        self, hf_to_sglang_mapper: WeightsMapper
    ):  # noqa: B027
        """
        Interface for models to update module names referenced in
        quantization configs in order to reflect the sglang model structure
        :param hf_to_sglang_mapper: maps from hf model structure (the assumed
            structure of the qconfig) to sglang model structure
        """
        pass


def method_has_implemented_embedding(method_class: Type[QuantizeMethodBase]) -> bool:
    """
    Not all quant methods have embedding implemented, so we need to check that
    it exists for our given method. We check this by making sure the function
    has been changed from the base implementation.
    """
    base_embedding = inspect.getattr_static(QuantizeMethodBase, "embedding", None)
    class_embedding = inspect.getattr_static(method_class, "embedding", None)

    return class_embedding is not None and class_embedding is not base_embedding
