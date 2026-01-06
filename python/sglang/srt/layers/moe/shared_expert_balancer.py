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
- SGLANG_FAKE_DISPATCH=0/1: Use uniform dispatch while running waterfill algorithm (default: 0)
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
FAKE_DISPATCH = os.environ.get("SGLANG_FAKE_DISPATCH", "0") == "1"
LOG_LOAD_DISTRIBUTION = os.environ.get("SGLANG_LOG_LOAD_DISTRIBUTION", "0") == "1"
LOG_LOAD_FILE = os.environ.get(
    "SGLANG_LOG_LOAD_FILE",
    "/lustre/raplab/client/xutingz/workspace/bench/waterfill_analysis.jsonl",
)
# 详细日志：每次调用都打印 token 分布（用于调试）
LOG_WATERFILL_VERBOSE = os.environ.get("SGLANG_LOG_WATERFILL_VERBOSE", "0") == "1"

# Global counter for logging frequency
_log_counter = 0
_LOG_INTERVAL = 10  # Log every N calls
_LOG_SKIP_FIRST = 500  # Skip first N calls (warmup)
_LOG_MIN_TOKENS = 32  # Only log batches with >= this many tokens
_log_file_handle = None


# ============== Triton Kernels for Waterfill ==============
if HAS_TRITON:

    @triton.jit
    def _waterfill_compute_kernel(
        routed_ptr,  # [world_size] routed counts per rank (int32)
        cum_weights_ptr,  # [world_size] output cumulative weights (float32)
        shared_ptr,  # [world_size] output shared tokens per rank (int32)
        total_shared,  # Total shared tokens to distribute
        min_threshold: tl.constexpr,  # Minimum shared tokens threshold
        WORLD_SIZE: tl.constexpr,  # world_size (typically 8)
    ):
        """Wrapper that calls optimized v2 kernel."""
        # Vectorized load
        offs = tl.arange(0, 8)
        mask = offs < WORLD_SIZE
        routed = tl.load(routed_ptr + offs, mask=mask, other=0).to(tl.int32)

        # Compute inverse weights
        max_routed = tl.max(routed)
        inv_weights = (max_routed - routed + 1).to(tl.float32)
        sum_weights = tl.sum(inv_weights)
        weights = inv_weights / sum_weights

        # Cumulative sum
        cum_w = tl.cumsum(weights, axis=0)
        tl.store(cum_weights_ptr + offs, cum_w, mask=mask)

        # Compute shared tokens
        total_f = total_shared.to(tl.float32)
        shared = tl.math.floor(weights * total_f).to(tl.int32)

        # Handle remainder
        total_assigned = tl.sum(shared)
        remainder = total_shared - total_assigned

        # Add to first min routed rank
        min_routed = tl.min(routed)
        is_min = (routed == min_routed).to(tl.int32)
        cumsum_min = tl.cumsum(is_min, axis=0)
        first_min_mask = (is_min == 1) & (cumsum_min == 1)
        shared = shared + tl.where(first_min_mask, remainder, 0)

        # Apply threshold
        shared = tl.where(shared < min_threshold, 0, shared)
        tl.store(shared_ptr + offs, shared, mask=mask)

        # Store shared tokens per rank
        for i in range(WORLD_SIZE):
            s = tl.sum(tl.where(tl.arange(0, 8) == i, shared, 0))
            tl.store(shared_ptr + i, s)

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
        Optimized histogram kernel v3.
        Uses vectorized operations with tl.where.
        """
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE

        # Local histogram
        local_hist = tl.zeros([8], dtype=tl.int32)
        offs = tl.arange(0, 8)

        # Process tokens
        for i in range(BLOCK_SIZE):
            token_idx = block_start + i
            if token_idx < num_tokens:
                base_ptr = topk_ids_ptr + token_idx * topk
                for k in range(topk):
                    expert_id = tl.load(base_ptr + k)
                    rank_id = expert_id // experts_per_rank
                    local_hist = tl.where(offs == rank_id, local_hist + 1, local_hist)

        # Atomic adds
        for r in range(8):
            count = tl.sum(tl.where(offs == r, local_hist, 0))
            if count > 0:
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
        Optimized assign + filter kernel v3.
        Computes token range and writes indices using vectorized stores.
        """
        pid = tl.program_id(0)

        # Load cumulative weights once (vectorized)
        offs = tl.arange(0, 8)
        cum_w = tl.load(cum_weights_ptr + offs, mask=offs < world_size, other=1.0)

        # Get bounds for this rank efficiently
        num_tokens_f = num_tokens.to(tl.float32)
        lower_w = 0.0 if rank == 0 else tl.sum(tl.where(offs == rank - 1, cum_w, 0.0))
        upper_w = tl.sum(tl.where(offs == rank, cum_w, 0.0))

        # Compute token range
        start_token = tl.maximum(tl.math.floor(lower_w * num_tokens_f + 0.5), 0.0).to(
            tl.int32
        )
        end_token = tl.minimum(
            tl.math.floor(upper_w * num_tokens_f + 0.5), num_tokens_f
        ).to(tl.int32)
        if rank == 0:
            start_token = 0

        count = end_token - start_token

        # Vectorized write - process BLOCK_SIZE indices at a time
        block_offs = tl.arange(0, BLOCK_SIZE)
        num_full_blocks = count // BLOCK_SIZE
        remainder = count % BLOCK_SIZE

        # Write full blocks
        for b in range(num_full_blocks):
            write_offs = b * BLOCK_SIZE + block_offs
            indices = (start_token + write_offs).to(tl.int64)
            tl.store(output_indices_ptr + write_offs, indices)

        # Write remainder
        if remainder > 0:
            write_offs = num_full_blocks * BLOCK_SIZE + block_offs
            indices = (start_token + write_offs).to(tl.int64)
            mask = block_offs < remainder
            tl.store(output_indices_ptr + write_offs, indices, mask=mask)

        # Store count (only from block 0)
        if pid == 0:
            tl.store(output_count_ptr, count)

    # Keep old kernel signature for compatibility but redirect to optimized version
    @triton.jit
    def _waterfill_assign_filter_kernel_old(
        cum_weights_ptr,
        output_indices_ptr,
        output_count_ptr,
        num_tokens,
        world_size: tl.constexpr,
        rank: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Old kernel kept for reference."""
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE

        cum_w0 = tl.load(cum_weights_ptr + 0) if world_size > 0 else 1.0
        cum_w1 = tl.load(cum_weights_ptr + 1) if world_size > 1 else 1.0
        cum_w2 = tl.load(cum_weights_ptr + 2) if world_size > 2 else 1.0
        cum_w3 = tl.load(cum_weights_ptr + 3) if world_size > 3 else 1.0
        cum_w4 = tl.load(cum_weights_ptr + 4) if world_size > 4 else 1.0
        cum_w5 = tl.load(cum_weights_ptr + 5) if world_size > 5 else 1.0
        cum_w6 = tl.load(cum_weights_ptr + 6) if world_size > 6 else 1.0
        cum_w7 = tl.load(cum_weights_ptr + 7) if world_size > 7 else 1.0

        num_tokens_f = num_tokens.to(tl.float32)
        lower_w = (
            0.0
            if rank == 0
            else (
                cum_w0
                if rank == 1
                else (
                    cum_w1
                    if rank == 2
                    else (
                        cum_w2
                        if rank == 3
                        else (
                            cum_w3
                            if rank == 4
                            else (
                                cum_w4
                                if rank == 5
                                else (cum_w5 if rank == 6 else cum_w6)
                            )
                        )
                    )
                )
            )
        )
        upper_w = (
            cum_w0
            if rank == 0
            else (
                cum_w1
                if rank == 1
                else (
                    cum_w2
                    if rank == 2
                    else (
                        cum_w3
                        if rank == 3
                        else (
                            cum_w4
                            if rank == 4
                            else (
                                cum_w5
                                if rank == 5
                                else (cum_w6 if rank == 6 else cum_w7)
                            )
                        )
                    )
                )
            )
        )

        for i in range(BLOCK_SIZE):
            token_idx = block_start + i
            if token_idx < num_tokens:
                # Compute normalized position
                pos = (token_idx.to(tl.float32) + 0.5) / num_tokens_f

                # Check if token belongs to current rank
                is_mine = 0
                if rank == 0:
                    is_mine = 1 if pos <= cum_w0 else 0
                else:
                    is_mine = 1 if (pos > lower_w) & (pos <= upper_w) else 0

                if is_mine == 1:
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
        self.shared_per_rank = torch.zeros(world_size, dtype=torch.int32, device=device)

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

    # BLOCK_SIZE=64 gives best performance (2.5x faster than 256)
    BLOCK_SIZE = 64
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

    # Kernel 2: Fused waterfill computation
    # Computes inverse weights, shared per rank, cumulative weights - all in one kernel
    MIN_THRESHOLD = 8  # Minimum shared tokens to assign
    _waterfill_compute_kernel[(1,)](
        buffers.histogram,
        buffers.cum_weights,
        buffers.shared_per_rank,
        num_tokens,
        MIN_THRESHOLD,
        world_size,
    )

    # 详细日志：打印每次调用的 token 分布
    if LOG_WATERFILL_VERBOSE:
        routed_list = buffers.histogram.tolist()
        shared_list = buffers.shared_per_rank.tolist()
        total_list = [r + s for r, s in zip(routed_list, shared_list)]
        min_shared = min(shared_list)
        max_total = max(total_list) if total_list else 0
        zeros = sum(1 for s in shared_list if s == 0)
        small = sum(1 for s in shared_list if 0 < s < 10)
        print(
            f"[WF] n={num_tokens} rank={rank} | "
            f"routed={routed_list} | "
            f"shared={shared_list} | "
            f"zeros={zeros} small(<10)={small} min_shared={min_shared} max_total={max_total}"
        )

    # Kernel 3: Assign tokens based on cumulative weights + filter
    # New optimized kernel only needs 1 block - computes range directly
    _waterfill_assign_filter_kernel[(1,)](
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

    # Log load distribution for analysis
    if LOG_LOAD_DISTRIBUTION:
        global _log_counter, _log_file_handle
        _log_counter += 1

        # Skip warmup and small batches
        should_log = (
            _log_counter > _LOG_SKIP_FIRST
            and _log_counter % _LOG_INTERVAL == 0
            and num_tokens >= _LOG_MIN_TOKENS
        )

        if should_log:
            import json
            import time

            # 1. Routed tokens per rank (from histogram)
            routed = buffers.histogram.tolist()
            routed_max = max(routed) if routed else 0
            routed_avg = sum(routed) / len(routed) if routed else 1
            routed_ratio = routed_max / routed_avg if routed_avg > 0 else 1.0

            # 2. Compute waterfill shared tokens for ALL ranks
            # Use the actual shared_per_rank from fused kernel
            waterfill_shared = buffers.shared_per_rank.tolist()

            # 3. Uniform shared tokens for each rank
            uniform_shared = []
            base = num_tokens // world_size
            remainder = num_tokens % world_size
            for i in range(world_size):
                # Uniform: rank i gets tokens[i::world_size]
                uniform_shared.append(base + (1 if i < remainder else 0))

            # 4. Total load per rank
            waterfill_total = [
                routed[i] + waterfill_shared[i] for i in range(world_size)
            ]
            uniform_total = [routed[i] + uniform_shared[i] for i in range(world_size)]

            # 5. Max load reduction
            wf_max_total = max(waterfill_total)
            uni_max_total = max(uniform_total)
            reduction = uni_max_total - wf_max_total
            reduction_pct = reduction / uni_max_total * 100 if uni_max_total > 0 else 0

            log_entry = {
                "timestamp": time.time(),
                "call_count": _log_counter,
                "rank": rank,
                "num_tokens": num_tokens,
                # Routed tokens per rank
                "routed_per_rank": routed,
                "routed_max": routed_max,
                "routed_ratio": round(routed_ratio, 3),
                # Shared tokens per rank (waterfill)
                "waterfill_shared_per_rank": waterfill_shared,
                # Shared tokens per rank (uniform)
                "uniform_shared_per_rank": uniform_shared,
                # Total load per rank
                "waterfill_total_per_rank": waterfill_total,
                "uniform_total_per_rank": uniform_total,
                # Reduction metrics
                "waterfill_max_total": wf_max_total,
                "uniform_max_total": uni_max_total,
                "max_reduction": reduction,
                "max_reduction_pct": round(reduction_pct, 2),
                # Current rank's actual count
                "my_shared_count": count,
            }

            # Write to file
            if _log_file_handle is None:
                _log_file_handle = open(LOG_LOAD_FILE, "a")
            _log_file_handle.write(json.dumps(log_entry) + "\n")
            _log_file_handle.flush()

            # Also print summary
            print(
                f"[LoadDist] #{_log_counter} n={num_tokens} "
                f"routed_max/avg={routed_ratio:.2f}x "
                f"wf_max={wf_max_total} uni_max={uni_max_total} "
                f"reduction={reduction} ({reduction_pct:.1f}%)"
            )

    # Return indices based on dispatch mode
    if FAKE_DISPATCH:
        # Use uniform dispatch (ignore waterfill assignment) while still running algorithm
        return uniform_indices
    else:
        # Use actual waterfill assignment
        return buffers.indices_buffer[:count].clone()


# ============== PyTorch Implementation ==============


def is_cuda_graph_capturing() -> bool:
    """Check if we're currently capturing a CUDA graph."""
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False


def waterfill(
    routed_counts: Tensor, total_shared: int, min_shared_threshold: int = 8
) -> Tensor:
    """
    Optimized waterfill algorithm to distribute shared tokens across ranks.

    Strategy: Use inverse-proportional weights (more tokens to lower-loaded ranks),
    then set small values to 0 to avoid kernel launch overhead for tiny batches.

    This is optimized for GPU execution with no CPU-GPU synchronization.

    Args:
        routed_counts: Routed tokens per rank
        total_shared: Total shared tokens to distribute
        min_shared_threshold: Minimum shared tokens to assign (smaller values set to 0)

    Returns: Target TOTAL load per rank (routed + shared)
    """
    routed = routed_counts.float()
    max_routed = routed.max()

    # Compute inverse-proportional weights
    # Lower routed load -> higher weight -> more shared tokens
    w = max_routed - routed + 1.0
    w = w / w.sum()

    # Direct shared calculation using floor + adjustment
    # This avoids rounding issues and is faster
    shared_float = w * total_shared
    shared = shared_float.floor().to(torch.int64)

    # Add remainder to rank with lowest routed (highest weight)
    # Use scatter_add for fully GPU execution (no sync)
    remainder = total_shared - shared.sum()
    min_idx = routed.argmin().view(1)
    shared = shared.scatter_add(0, min_idx, remainder.view(1))

    # Threshold processing: set small values to 0
    # This avoids kernel launch overhead for tiny batches
    shared = torch.where(
        shared < min_shared_threshold, torch.zeros_like(shared), shared
    )

    return routed_counts + shared


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
