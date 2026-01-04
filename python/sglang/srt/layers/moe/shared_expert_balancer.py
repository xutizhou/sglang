# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Shared Expert Balancer for MoE models.

This module implements load balancing for shared expert computation in MoE models
(e.g., DeepSeek V3/R1). The algorithm distributes shared expert computation across
ranks based on routed expert load.

Two kernel implementations:
- Triton (default): Fused kernels for waterfill algorithm
  Kernel 1: histogram (count routed tokens per rank)
  Kernel 2: assign + filter (waterfill assignment + filter indices)
  Uses .item() sync to get count.
- PyTorch: Same waterfill algorithm using PyTorch ops (bincount + searchsorted + nonzero)

Environment variables:
- SGLANG_USE_TRITON_WATERFILL=0/1: Use Triton kernel (default: 1)
- SGLANG_FAKE_SYNC_EXPERIMENT=0/1: Skip .item() sync for benchmarking (default: 0)
"""

import os
from typing import Optional, Tuple

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ============== Environment Variables ==============
USE_TRITON_WATERFILL = os.environ.get("SGLANG_USE_TRITON_WATERFILL", "1") == "1"
FAKE_SYNC_EXPERIMENT = os.environ.get("SGLANG_FAKE_SYNC_EXPERIMENT", "0") == "1"


# ============== Triton Kernels for Waterfill ==============
if HAS_TRITON:

    @triton.jit
    def _histogram_kernel(
        topk_ids_ptr,
        histogram_ptr,
        num_tokens,
        topk: tl.constexpr,
        experts_per_rank: tl.constexpr,
        world_size: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Compute histogram of routed tokens per rank.
        Each thread block processes BLOCK_SIZE tokens and accumulates locally,
        then atomically adds to global histogram.
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE

        # Local histogram for this block
        local_hist = tl.zeros([8], dtype=tl.int32)  # Max 8 ranks

        for i in range(BLOCK_SIZE):
            token_idx = block_start + i
            if token_idx < num_tokens:
                for k in range(topk):
                    expert_id = tl.load(topk_ids_ptr + token_idx * topk + k)
                    rank_id = expert_id // experts_per_rank
                    rank_id = tl.minimum(tl.maximum(rank_id, 0), world_size - 1)
                    # Manual increment for each rank
                    local_hist = tl.where(
                        tl.arange(0, 8) == rank_id,
                        local_hist + 1,
                        local_hist,
                    )

        # Atomically add local histogram to global
        for r in range(world_size):
            if tl.sum(tl.where(tl.arange(0, 8) == r, local_hist, 0)) > 0:
                count = tl.sum(tl.where(tl.arange(0, 8) == r, local_hist, 0))
                tl.atomic_add(histogram_ptr + r, count)

    @triton.jit
    def _waterfill_assign_filter_kernel(
        cum_weights_ptr,  # [world_size] cumulative weights
        output_indices_ptr,
        output_count_ptr,
        num_tokens,
        world_size: tl.constexpr,
        rank: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Assign tokens to ranks using waterfill (via cumulative weights) and filter.

        For each token:
        1. Compute position = (token_idx + 0.5) / num_tokens
        2. Find rank where cum_weights[rank-1] < position <= cum_weights[rank]
        3. If assigned to current rank, add to output
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE

        # Load cumulative weights (small, fits in registers)
        cum_w0 = tl.load(cum_weights_ptr + 0) if world_size > 0 else 1.0
        cum_w1 = tl.load(cum_weights_ptr + 1) if world_size > 1 else 1.0
        cum_w2 = tl.load(cum_weights_ptr + 2) if world_size > 2 else 1.0
        cum_w3 = tl.load(cum_weights_ptr + 3) if world_size > 3 else 1.0
        cum_w4 = tl.load(cum_weights_ptr + 4) if world_size > 4 else 1.0
        cum_w5 = tl.load(cum_weights_ptr + 5) if world_size > 5 else 1.0
        cum_w6 = tl.load(cum_weights_ptr + 6) if world_size > 6 else 1.0
        cum_w7 = tl.load(cum_weights_ptr + 7) if world_size > 7 else 1.0

        for i in range(BLOCK_SIZE):
            token_idx = block_start + i
            if token_idx < num_tokens:
                # Compute normalized position
                pos = (token_idx.to(tl.float32) + 0.5) / num_tokens.to(tl.float32)

                # Searchsorted: find first cum_weight >= pos
                assigned_rank = 0
                if pos > cum_w0:
                    assigned_rank = 1
                if pos > cum_w1:
                    assigned_rank = 2
                if pos > cum_w2:
                    assigned_rank = 3
                if pos > cum_w3:
                    assigned_rank = 4
                if pos > cum_w4:
                    assigned_rank = 5
                if pos > cum_w5:
                    assigned_rank = 6
                if pos > cum_w6:
                    assigned_rank = 7

                # Clamp to valid range
                assigned_rank = tl.minimum(assigned_rank, world_size - 1)

                if assigned_rank == rank:
                    out_idx = tl.atomic_add(output_count_ptr, 1)
                    tl.store(output_indices_ptr + out_idx, token_idx)


class TritonWaterfillBuffers:
    """Pre-allocated buffers for Triton waterfill kernels."""

    def __init__(self, max_tokens: int, world_size: int, device):
        self.device = device
        self.world_size = world_size
        self.max_tokens = max_tokens
        self.indices_buffer = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        self.count = torch.zeros(1, dtype=torch.int32, device=device)
        self.histogram = torch.zeros(world_size, dtype=torch.int32, device=device)
        self.cum_weights = torch.zeros(world_size, dtype=torch.float32, device=device)

    def resize_if_needed(self, num_tokens: int):
        if num_tokens > self.max_tokens:
            self.indices_buffer = torch.zeros(
                num_tokens, dtype=torch.int64, device=self.device
            )
            self.max_tokens = num_tokens


def get_my_indices_triton(
    topk_ids: Tensor,
    num_experts: int,
    world_size: int,
    rank: int,
    buffers: TritonWaterfillBuffers,
) -> Tensor:
    """
    Triton implementation of waterfill algorithm.

    Two kernels:
    1. _histogram_kernel: Count routed tokens per rank
    2. _waterfill_assign_filter_kernel: Assign + filter based on inverse weights

    Args:
        topk_ids: [num_tokens, topk] tensor of expert IDs
        num_experts: Total number of experts
        world_size: Number of ranks
        rank: Current rank ID
        buffers: Pre-allocated buffers

    Returns:
        indices: 1D tensor of token indices assigned to this rank
    """
    if not HAS_TRITON:
        raise RuntimeError("Triton not available")

    num_tokens = topk_ids.shape[0]
    topk = topk_ids.shape[1]
    experts_per_rank = num_experts // world_size
    device = topk_ids.device

    buffers.resize_if_needed(num_tokens)
    buffers.count.zero_()
    buffers.histogram.zero_()

    # Pre-initialize buffer with uniform indices (for FAKE_SYNC_EXPERIMENT safety)
    # This ensures valid indices even if we use wrong count
    uniform_indices = torch.arange(rank, num_tokens, world_size, device=device)
    uniform_count = uniform_indices.shape[0]
    buffers.indices_buffer[:uniform_count] = uniform_indices

    BLOCK_SIZE = 256
    num_blocks = (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Kernel 1: Compute histogram of routed tokens per rank
    _histogram_kernel[(num_blocks,)](
        topk_ids,
        buffers.histogram,
        num_tokens,
        topk,
        experts_per_rank,
        world_size,
        BLOCK_SIZE,
    )

    # Compute inverse weights on GPU (small tensor, fast)
    routed_counts = buffers.histogram.float()
    max_routed = routed_counts.max()
    inverse_load = max_routed - routed_counts + 1.0
    weights = inverse_load / inverse_load.sum()
    buffers.cum_weights.copy_(torch.cumsum(weights, dim=0))

    # Kernel 2: Assign tokens based on cumulative weights + filter
    _waterfill_assign_filter_kernel[(num_blocks,)](
        buffers.cum_weights,
        buffers.indices_buffer,
        buffers.count,
        num_tokens,
        world_size,
        rank,
        BLOCK_SIZE,
    )

    # Get count - this is the CPU-GPU sync point
    if FAKE_SYNC_EXPERIMENT:
        # Skip .item() sync, use uniform count (for benchmarking)
        count = uniform_count
    else:
        count = buffers.count[0].item()

    return buffers.indices_buffer[:count].clone()


# ============== PyTorch Implementation ==============


def is_cuda_graph_capturing() -> bool:
    """Check if we're currently capturing a CUDA graph."""
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def waterfill(routed_counts: Tensor, total_shared: int) -> Tensor:
    """
    Waterfill algorithm to distribute shared tokens across ranks.
    """
    world_size = routed_counts.shape[0]
    device = routed_counts.device

    vals, idx = torch.sort(routed_counts.clone().to(torch.int64))
    cum_vals = torch.cumsum(vals, dim=0)
    k_range = torch.arange(1, world_size + 1, device=device)
    costs = k_range * vals - cum_vals

    can_fill = costs <= total_shared
    k_idx = torch.sum(can_fill.to(torch.int64)) - 1
    k_idx = torch.clamp(k_idx, 0, world_size - 1)

    cost_at_k = torch.gather(costs, 0, k_idx.unsqueeze(0)).squeeze(0)
    vals_at_k = torch.gather(vals, 0, k_idx.unsqueeze(0)).squeeze(0)

    remaining_after_fill = total_shared - cost_at_k
    base_level = vals_at_k

    divisor = k_idx + 1
    inc_all = remaining_after_fill // divisor
    rem = remaining_after_fill % divisor

    mask = torch.arange(world_size, device=device) <= k_idx
    vals = torch.where(mask, base_level + inc_all, vals)

    rem_mask = torch.arange(world_size, device=device) < rem
    vals = torch.where(rem_mask, vals + 1, vals)

    target = torch.empty_like(vals)
    target[idx] = vals

    return target


def assign_shared_expert(
    topk_ids: Tensor,
    num_experts: int,
    world_size: int,
) -> Tensor:
    """
    Assign shared expert computation to ranks using inverse-proportional algorithm.
    """
    num_tokens = topk_ids.shape[0]
    experts_per_rank = num_experts // world_size
    device = topk_ids.device

    if num_tokens <= world_size:
        return torch.arange(num_tokens, device=device) % world_size

    # Count routed tokens per rank
    rank_ids = topk_ids // experts_per_rank
    flat_ranks = rank_ids.flatten().to(torch.int64)
    flat_ranks = torch.clamp(flat_ranks, 0, world_size - 1)
    routed_counts = torch.bincount(flat_ranks, minlength=world_size)[
        :world_size
    ].float()

    # Compute inverse weights
    max_routed = routed_counts.max()
    inverse_load = max_routed - routed_counts + 1.0
    weights = inverse_load / inverse_load.sum()

    # Assign tokens based on cumulative weights
    cum_weights = torch.cumsum(weights, dim=0)
    token_positions = (
        torch.arange(num_tokens, device=device, dtype=torch.float32) + 0.5
    ) / num_tokens
    assignment = torch.searchsorted(cum_weights, token_positions)
    assignment = torch.clamp(assignment, 0, world_size - 1)

    return assignment


def get_my_indices_pytorch(
    topk_ids: Tensor,
    num_experts: int,
    world_size: int,
    rank: int,
) -> Tensor:
    """
    PyTorch implementation: waterfill algorithm + nonzero().
    """
    assignment = assign_shared_expert(topk_ids, num_experts, world_size)
    mask = assignment == rank
    return mask.nonzero(as_tuple=True)[0]


# ============== Main Class ==============


class SharedExpertBalancer:
    """
    Shared Expert load balancer for MoE models.

    Two kernel implementations (both do waterfill load balancing):
    - Triton (default): Fused kernels (histogram + assign/filter)
    - PyTorch: bincount + searchsorted + nonzero()

    Usage:
        balancer = SharedExpertBalancer(num_experts=256, world_size=8, rank=0)
        my_indices = balancer.get_my_indices(topk_ids, num_tokens, device)
        my_hidden = hidden_states[my_indices]
    """

    MIN_BATCH_FOR_BALANCE = 64

    def __init__(
        self,
        num_experts: int,
        world_size: int,
        rank: int,
        shared_scaling: float = 1.0,
        use_waterfill: bool = True,
        use_triton: bool = None,
    ):
        self.num_experts = num_experts
        self.world_size = world_size
        self.rank = rank
        self.shared_scaling = shared_scaling
        self.use_waterfill = use_waterfill

        if use_triton is None:
            use_triton = USE_TRITON_WATERFILL
        self.use_triton = use_triton and HAS_TRITON

        self._triton_buffers: Optional[TritonWaterfillBuffers] = None
        self._static_indices_cache: dict = {}

    def get_static_indices(self, num_tokens: int, device: torch.device) -> Tensor:
        """Get static round-robin indices (CUDA Graph compatible)."""
        cache_key = (num_tokens, device)
        if cache_key not in self._static_indices_cache:
            if self.rank >= num_tokens:
                indices = torch.empty(0, dtype=torch.int64, device=device)
            else:
                indices = torch.arange(
                    self.rank, num_tokens, self.world_size, device=device
                )
            self._static_indices_cache[cache_key] = indices
        return self._static_indices_cache[cache_key]

    def get_my_indices(
        self,
        topk_ids: Optional[Tensor],
        num_tokens: int,
        device: torch.device,
        force_static: bool = False,
    ) -> Tensor:
        """
        Get indices of tokens assigned to this rank.
        """
        use_static = (
            force_static
            or topk_ids is None
            or is_cuda_graph_capturing()
            or not self.use_waterfill
            or num_tokens < self.MIN_BATCH_FOR_BALANCE
        )

        if use_static:
            return self.get_static_indices(num_tokens, device)

        if self.use_triton:
            # Triton: fused waterfill kernels
            if self._triton_buffers is None:
                self._triton_buffers = TritonWaterfillBuffers(
                    num_tokens, self.world_size, device
                )
            return get_my_indices_triton(
                topk_ids,
                self.num_experts,
                self.world_size,
                self.rank,
                self._triton_buffers,
            )
        else:
            # PyTorch: waterfill + nonzero()
            return get_my_indices_pytorch(
                topk_ids, self.num_experts, self.world_size, self.rank
            )


# ============== Analysis Utilities ==============


def compute_imbalance_score(
    topk_ids: Tensor,
    num_experts: int,
    world_size: int,
    use_balance: bool = True,
) -> Tuple[float, Tensor, Tensor]:
    """Compute imbalance score for analysis."""
    num_tokens = topk_ids.shape[0]
    experts_per_rank = num_experts // world_size

    rank_ids = topk_ids // experts_per_rank
    flat_ranks = rank_ids.flatten()
    routed_counts = torch.bincount(flat_ranks, minlength=world_size)[:world_size].to(
        torch.int64
    )

    if use_balance:
        assignment = assign_shared_expert(topk_ids, num_experts, world_size)
        shared_counts = torch.bincount(assignment, minlength=world_size)[
            :world_size
        ].to(torch.int64)
        total_counts = routed_counts + shared_counts
    else:
        total_counts = routed_counts + num_tokens

    total_float = total_counts.float()
    imbalance_score = (total_float.max() / total_float.mean()).item()

    return imbalance_score, routed_counts, total_counts
