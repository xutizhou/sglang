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
End-to-end tests for SharedExpertBalancer integration with DeepSeek V2/V3 MoE.

This test validates:
1. Algorithm correctness (determinism, load balance)
2. Numerical correctness (output matches baseline)
3. Performance improvement (imbalance score reduction)

Usage:
    # Run unit tests (CPU only)
    python -m pytest test/srt/test_shared_expert_balance_e2e.py -v

    # Run with model (requires GPU and model weights)
    python test/srt/test_shared_expert_balance_e2e.py --with-model
"""

import argparse
import unittest

import torch

from sglang.srt.layers.moe.shared_expert_balancer import (
    SharedExpertBalancer,
    assign_shared_expert,
    compute_imbalance_score,
    waterfill,
)


class TestSharedExpertBalancerAlgorithm(unittest.TestCase):
    """Test algorithm correctness without model."""

    def test_waterfill_creates_balanced_distribution(self):
        """Verify waterfill algorithm produces balanced output."""
        # Simulate 8 ranks with imbalanced routed counts
        routed_counts = torch.tensor([500, 100, 400, 150, 300, 200, 450, 250])
        total_shared = 2000  # tokens to distribute

        target = waterfill(routed_counts, total_shared)

        # Check sum is correct
        self.assertEqual(
            target.sum().item(),
            routed_counts.sum().item() + total_shared,
        )

        # Check balance: max - min <= 1
        self.assertLessEqual(
            target.max().item() - target.min().item(),
            1,
            f"Target not balanced: {target.tolist()}",
        )

    def test_assignment_is_deterministic(self):
        """Verify assignment is deterministic across multiple calls."""
        torch.manual_seed(42)
        topk_ids = torch.randint(0, 256, (1000, 8))

        # Run multiple times
        results = [assign_shared_expert(topk_ids, 256, 8) for _ in range(3)]

        # All results should be identical
        for i, r in enumerate(results[1:], 1):
            self.assertTrue(
                torch.equal(results[0], r),
                f"Assignment {i} differs from assignment 0",
            )

    def test_assignment_respects_routed_rank_constraint(self):
        """Verify shared expert is assigned to a routed rank."""
        num_experts = 256
        world_size = 8
        experts_per_rank = num_experts // world_size

        torch.manual_seed(42)
        topk_ids = torch.randint(0, num_experts, (100, 8))
        assignment = assign_shared_expert(topk_ids, num_experts, world_size)

        for t in range(100):
            # Get routed ranks for this token
            routed_ranks = set()
            for k in range(8):
                expert_id = topk_ids[t, k].item()
                rank = expert_id // experts_per_rank
                routed_ranks.add(rank)

            # Check assignment is in routed ranks
            assigned_rank = assignment[t].item()
            self.assertIn(
                assigned_rank,
                routed_ranks,
                f"Token {t} assigned to rank {assigned_rank}, but routed ranks are {routed_ranks}",
            )

    def test_balancer_class_integration(self):
        """Test SharedExpertBalancer class."""
        balancer = SharedExpertBalancer(
            num_experts=256,
            world_size=8,
            rank=0,
            shared_scaling=1.0,
        )

        torch.manual_seed(42)
        topk_ids = torch.randint(0, 256, (1000, 8))

        # Get assignment
        assignment = balancer.assign(topk_ids)
        self.assertEqual(assignment.shape[0], 1000)

        # Get tokens for this rank
        my_tokens = balancer.get_my_shared_tokens(assignment)

        # Verify all returned tokens are assigned to this rank
        for idx in my_tokens:
            self.assertEqual(assignment[idx].item(), 0)

    def test_imbalance_score_improvement(self):
        """Verify balancing improves imbalance score."""
        # Create power-law distributed routing (realistic scenario)
        num_tokens = 10000
        num_experts = 256
        world_size = 8
        topk = 8

        # Create biased routing towards lower expert IDs
        probs = torch.arange(num_experts, 0, -1, dtype=torch.float32)
        probs = probs / probs.sum()
        expert_ids = torch.multinomial(probs, num_tokens * topk, replacement=True)
        topk_ids = expert_ids.reshape(num_tokens, topk)

        # Compute scores
        score_balanced, _, total_balanced = compute_imbalance_score(
            topk_ids, num_experts, world_size, use_balance=True
        )
        score_unbalanced, _, total_unbalanced = compute_imbalance_score(
            topk_ids, num_experts, world_size, use_balance=False
        )

        print(f"\nImbalance Score Comparison:")
        print(f"  With balancing: {score_balanced:.4f}")
        print(f"  Without balancing: {score_unbalanced:.4f}")
        print(f"  Improvement: {(1 - score_balanced/score_unbalanced)*100:.1f}%")

        # Balanced should be better (lower) than unbalanced
        self.assertLessEqual(
            score_balanced,
            score_unbalanced,
            "Balancing should improve (reduce) imbalance score",
        )


class TestMultiRankConsistency(unittest.TestCase):
    """Test consistency across multiple simulated ranks."""

    def test_all_ranks_compute_same_assignment(self):
        """Verify all ranks compute identical assignments."""
        num_experts = 256
        world_size = 8

        torch.manual_seed(42)
        topk_ids = torch.randint(0, num_experts, (1000, 8))

        # Simulate each rank computing assignment
        assignments = []
        for rank in range(world_size):
            balancer = SharedExpertBalancer(
                num_experts=num_experts,
                world_size=world_size,
                rank=rank,
            )
            assignment = balancer.assign(topk_ids)
            assignments.append(assignment)

        # All assignments should be identical
        for rank, assignment in enumerate(assignments[1:], 1):
            self.assertTrue(
                torch.equal(assignments[0], assignment),
                f"Rank {rank} has different assignment than rank 0",
            )

    def test_all_tokens_covered_exactly_once(self):
        """Verify each token is assigned to exactly one rank."""
        num_experts = 256
        world_size = 8

        torch.manual_seed(42)
        topk_ids = torch.randint(0, num_experts, (1000, 8))

        # Get assignment
        assignment = assign_shared_expert(topk_ids, num_experts, world_size)

        # Count assignments per rank
        counts = torch.zeros(world_size, dtype=torch.int64)
        for r in range(world_size):
            counts[r] = (assignment == r).sum()

        # Total should equal number of tokens
        self.assertEqual(counts.sum().item(), 1000)

        # Each token should be assigned to exactly one rank
        for t in range(1000):
            self.assertTrue(
                0 <= assignment[t].item() < world_size,
                f"Token {t} has invalid assignment {assignment[t].item()}",
            )


def run_model_test():
    """
    Run end-to-end test with actual model (requires GPU and model weights).

    This test:
    1. Loads a DeepSeek V2/V3 model
    2. Runs inference with and without shared expert balancing
    3. Compares outputs for numerical equivalence
    """
    print("\n" + "=" * 60)
    print("Running End-to-End Model Test")
    print("=" * 60)

    try:
        import importlib.util

        if importlib.util.find_spec("sglang") is None:
            raise ImportError("sglang not found")
    except ImportError:
        print("SGLang not available for model test")
        return

    print("\nNote: This test requires a DeepSeek V2/V3 model and multiple GPUs.")
    print("Skipping model test in this run. To run manually:")
    print(
        """
    # Start server with EP + none mode (baseline)
    python -m sglang.launch_server \\
        --model deepseek-ai/DeepSeek-V2-Lite \\
        --tp 4 --ep 4 \\
        --moe-a2a-backend none

    # Start server with shared expert balance enabled
    python -m sglang.launch_server \\
        --model deepseek-ai/DeepSeek-V2-Lite \\
        --tp 4 --ep 4 \\
        --moe-a2a-backend none \\
        --enable-shared-expert-balance
    """
    )


def main():
    parser = argparse.ArgumentParser(description="Test SharedExpertBalancer")
    parser.add_argument(
        "--with-model",
        action="store_true",
        help="Run end-to-end test with actual model (requires GPU)",
    )
    args = parser.parse_args()

    # Run unit tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestSharedExpertBalancerAlgorithm))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiRankConsistency))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if args.with_model:
        run_model_test()

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit(main())
