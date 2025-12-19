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
Benchmark script for SharedExpertBalancer performance.

Compares three modes:
1. Baseline 1 (TP8): Shared expert uses TP, each rank holds 1/8 weights, computes all tokens
2. Baseline 2 (Replicated + Uniform): Each rank holds full weights, computes tokens[rank::world_size]
3. Optimized (Replicated + Waterfill): Each rank holds full weights, uses waterfill load balancing
"""

import time
from dataclasses import dataclass
from typing import Tuple

import torch

from sglang.srt.layers.moe.shared_expert_balancer import assign_shared_expert, waterfill


def create_power_law_routing(
    num_tokens: int, num_experts: int, topk: int, alpha: float = 1.0
) -> torch.Tensor:
    """Create power-law distributed routing (simulates real-world imbalanced routing)."""
    probs = torch.arange(1, num_experts + 1, dtype=torch.float32) ** (-alpha)
    probs = probs / probs.sum()
    expert_ids = torch.multinomial(probs, num_tokens * topk, replacement=True)
    return expert_ids.reshape(num_tokens, topk)


def compute_routed_counts(
    topk_ids: torch.Tensor, num_experts: int, world_size: int
) -> torch.Tensor:
    """Compute routed token counts per rank."""
    experts_per_rank = num_experts // world_size
    rank_ids = topk_ids // experts_per_rank
    flat_ranks = rank_ids.flatten()
    routed_counts = torch.bincount(flat_ranks, minlength=world_size)[:world_size].to(
        torch.int64
    )
    return routed_counts


def compute_loads_tp8(
    topk_ids: torch.Tensor, num_tokens: int, num_experts: int, world_size: int
) -> Tuple[torch.Tensor, int]:
    """
    Baseline 1 (TP8): Each rank computes all tokens with 1/8 weights.
    Shared load is uniform across all ranks (all tokens).
    Total load = routed_load + num_tokens (shared is same for all)
    """
    routed_counts = compute_routed_counts(topk_ids, num_experts, world_size)
    # In TP8 mode, shared expert work is distributed via TP, so each rank
    # does 1/8 of the FLOPS for ALL tokens. The bottleneck is still
    # determined by routed_load, since shared_load is uniform.
    # For comparison, we treat shared_load as num_tokens for each rank
    # (representing the "token dimension" work, even though weight dimension is split)
    shared_counts = torch.full((world_size,), num_tokens, dtype=torch.int64)
    total_counts = routed_counts + shared_counts
    return total_counts, total_counts.max().item()


def compute_loads_uniform(
    topk_ids: torch.Tensor, num_tokens: int, num_experts: int, world_size: int
) -> Tuple[torch.Tensor, int]:
    """
    Baseline 2 (Replicated + Uniform): Each rank holds full weights,
    computes tokens[rank::world_size] - simple round-robin distribution.
    """
    routed_counts = compute_routed_counts(topk_ids, num_experts, world_size)
    # Uniform distribution: each rank gets approximately num_tokens // world_size
    base_shared = num_tokens // world_size
    remainder = num_tokens % world_size
    shared_counts = torch.full((world_size,), base_shared, dtype=torch.int64)
    # First 'remainder' ranks get one extra token
    shared_counts[:remainder] += 1
    total_counts = routed_counts + shared_counts
    return total_counts, total_counts.max().item()


def compute_loads_waterfill(
    topk_ids: torch.Tensor, num_tokens: int, num_experts: int, world_size: int
) -> Tuple[torch.Tensor, int]:
    """
    Optimized (Replicated + Waterfill): Each rank holds full weights,
    uses waterfill algorithm to balance total load across ranks.
    """
    routed_counts = compute_routed_counts(topk_ids, num_experts, world_size)
    # Use waterfill to compute target totals
    target_totals = waterfill(routed_counts, num_tokens)
    # shared_counts = target_totals - routed_counts
    shared_counts = target_totals - routed_counts
    total_counts = routed_counts + shared_counts
    return total_counts, total_counts.max().item()


@dataclass
class BenchmarkResult:
    num_tokens: int
    topk: int
    world_size: int
    num_experts: int
    # Routed load stats
    routed_max: int
    routed_imbalance: float
    # Three modes
    max_tp8: int
    max_uniform: int
    max_waterfill: int
    # Improvements
    uniform_vs_tp8_pct: float
    waterfill_vs_tp8_pct: float
    waterfill_vs_uniform_pct: float
    # Assignment time
    assign_time_us: float


def benchmark_single_config(
    num_tokens: int, topk: int, world_size: int, num_experts: int, num_runs: int = 10
) -> BenchmarkResult:
    topk_ids = create_power_law_routing(num_tokens, num_experts, topk)

    # Compute routed counts for reference
    routed_counts = compute_routed_counts(topk_ids, num_experts, world_size)
    routed_max = routed_counts.max().item()
    routed_avg = routed_counts.float().mean().item()
    routed_imbalance = routed_max / routed_avg if routed_avg > 0 else 1.0

    # Compute loads for three modes
    _, max_tp8 = compute_loads_tp8(topk_ids, num_tokens, num_experts, world_size)
    _, max_uniform = compute_loads_uniform(
        topk_ids, num_tokens, num_experts, world_size
    )
    _, max_waterfill = compute_loads_waterfill(
        topk_ids, num_tokens, num_experts, world_size
    )

    # Benchmark assignment time
    # Warmup
    for _ in range(3):
        _ = assign_shared_expert(topk_ids, num_experts, world_size)

    start = time.perf_counter()
    for _ in range(num_runs):
        _ = assign_shared_expert(topk_ids, num_experts, world_size)
    assign_time_us = (time.perf_counter() - start) * 1e6 / num_runs

    # Compute improvements (lower max load = better)
    uniform_vs_tp8 = (1 - max_uniform / max_tp8) * 100 if max_tp8 > 0 else 0
    waterfill_vs_tp8 = (1 - max_waterfill / max_tp8) * 100 if max_tp8 > 0 else 0
    waterfill_vs_uniform = (
        (1 - max_waterfill / max_uniform) * 100 if max_uniform > 0 else 0
    )

    return BenchmarkResult(
        num_tokens=num_tokens,
        topk=topk,
        world_size=world_size,
        num_experts=num_experts,
        routed_max=routed_max,
        routed_imbalance=routed_imbalance,
        max_tp8=max_tp8,
        max_uniform=max_uniform,
        max_waterfill=max_waterfill,
        uniform_vs_tp8_pct=uniform_vs_tp8,
        waterfill_vs_tp8_pct=waterfill_vs_tp8,
        waterfill_vs_uniform_pct=waterfill_vs_uniform,
        assign_time_us=assign_time_us,
    )


def main():
    print("=" * 120)
    print("SharedExpertBalancer Benchmark: Comparing Three Modes")
    print("=" * 120)
    print()
    print("Modes:")
    print(
        "  1. TP8 (Baseline 1): Shared expert uses TP8, each rank computes all tokens with 1/8 weights"
    )
    print(
        "  2. Uniform (Baseline 2): Each rank holds full weights, computes tokens[rank::world_size]"
    )
    print(
        "  3. Waterfill (Optimized): Each rank holds full weights, uses waterfill load balancing"
    )
    print()
    print("Note: 'Max Load' = max(routed_load + shared_load) across all ranks")
    print("      Lower max load = better performance (less bottleneck)")
    print()
    print("-" * 120)
    print(
        f"{'Config':^30} | {'Routed':^15} | {'Max Load by Mode':^35} | {'Improvement':^30}"
    )
    print(
        f"{'Tokens/TopK/World/Experts':^30} | {'Max (Imbal)':^15} | {'TP8':^10} {'Uniform':^10} {'Waterfill':^10} | {'Uni/TP8':^9} {'WF/TP8':^9} {'WF/Uni':^9}"
    )
    print("-" * 120)

    configs = [
        # Standard DeepSeek V3 config variations
        (1024, 8, 8, 256),
        (4096, 8, 8, 256),
        (16384, 8, 8, 256),
        (65536, 8, 8, 256),
        # Different world sizes
        (4096, 8, 4, 256),
        (4096, 8, 16, 256),
        (4096, 8, 32, 256),
        # Different topk
        (4096, 6, 8, 256),
        (4096, 4, 8, 256),
    ]

    for cfg in configs:
        r = benchmark_single_config(*cfg)
        config_str = f"{r.num_tokens}/{r.topk}/{r.world_size}/{r.num_experts}"
        routed_str = f"{r.routed_max} ({r.routed_imbalance:.2f}x)"
        print(
            f"{config_str:^30} | {routed_str:^15} | {r.max_tp8:^10} {r.max_uniform:^10} {r.max_waterfill:^10} | "
            f"{r.uniform_vs_tp8_pct:>+8.1f}% {r.waterfill_vs_tp8_pct:>+8.1f}% {r.waterfill_vs_uniform_pct:>+8.1f}%"
        )

    print("-" * 120)
    print()
    print("Key Insights:")
    print(
        "  - TP8 vs Uniform: Uniform reduces max load because each rank only computes 1/8 tokens"
    )
    print(
        "  - Waterfill vs Uniform: Waterfill further reduces max load by assigning fewer shared"
    )
    print("                          tokens to ranks with high routed load")
    print("  - The benefit of Waterfill depends on the routed load imbalance")
    print()


if __name__ == "__main__":
    main()
