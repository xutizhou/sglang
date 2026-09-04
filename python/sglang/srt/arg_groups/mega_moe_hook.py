from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

from sglang.srt.arg_groups.overrides import declare_resolution

logger = logging.getLogger(__name__)


def handle_mega_moe(server_args: ServerArgs) -> None:
    handle_moe_runner_backend_alias(server_args)

    from sglang.srt.models.deepseek_common.utils import _device_sm

    _check_mega_moe_arch(server_args.moe_a2a_backend, _device_sm)
    handle_w4a4_mxfp4_megamoe_env(server_args)


def _check_mega_moe_arch(moe_a2a_backend: str, device_sm: Optional[int]) -> None:
    """Reject MegaMoE on architectures DeepGEMM has no kernel for."""
    if moe_a2a_backend != "megamoe":
        return

    from sglang.srt.layers.moe.mega_moe import MEGA_MOE_SUPPORTED_SM

    if device_sm in MEGA_MOE_SUPPORTED_SM:
        return

    supported = ", ".join(f"SM{sm}" for sm in MEGA_MOE_SUPPORTED_SM)
    raise ValueError(
        f"--moe-a2a-backend megamoe requires {supported}; this device reports "
        f"SM{device_sm}. DeepGEMM has no MegaMoE kernel for it."
    )


def handle_moe_runner_backend_alias(server_args: ServerArgs) -> None:
    if server_args.moe_runner_backend != "megamoe":
        return

    if server_args.moe_a2a_backend not in ("none", "megamoe"):
        logger.warning(
            "--moe-runner-backend megamoe is an alias for "
            "--moe-a2a-backend megamoe; overriding "
            "--moe-a2a-backend %s.",
            server_args.moe_a2a_backend,
        )
    declare_resolution(
        server_args,
        "handle_moe_runner_backend_alias",
        moe_runner_backend="auto",
        moe_a2a_backend="megamoe",
    )


def handle_w4a4_mxfp4_megamoe_env(server_args: ServerArgs) -> None:
    if not server_args.enable_w4a4_mxfp4_megamoe:
        return

    os.environ["DG_USE_FP4_ACTS"] = "1"
    os.environ["DG_USE_MXF4_KIND"] = "1"
