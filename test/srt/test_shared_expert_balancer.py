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
Unit tests for SharedExpertBalancer.
"""

import unittest

import torch

from sglang.srt.layers.moe.shared_expert_balancer import (
    assign_shared_expert,
    compute_imbalance_score,
    waterfill,
)


class TestWaterfill(unittest.TestCase):
    def test_waterfill_balance_simple(self):
        routed = torch.tensor([200, 50, 180, 70])
        total_shared = 400
        target = waterfill(routed, total_shared)
        self.assertEqual(target.sum().item(), routed.sum().item() + total_shared)
        self.assertLessEqual(target.max().item() - target.min().item(), 1)

    def test_waterfill_extreme_imbalance(self):
        routed = torch.tensor([1000, 0, 0, 0])
        total_shared = 1000
        target = waterfill(routed, total_shared)
        self.assertEqual(target[0].item(), 1000)
        self.assertLessEqual(target[1:].max().item() - target[1:].min().item(), 1)


class TestAssignSharedExpert(unittest.TestCase):
    def test_assign_deterministic(self):
        topk_ids = torch.randint(0, 256, (1024, 8))
        result1 = assign_shared_expert(topk_ids, 256, 8)
        result2 = assign_shared_expert(topk_ids, 256, 8)
        self.assertTrue(torch.equal(result1, result2))

    def test_assign_load_balance(self):
        num_tokens = 10000
        num_experts = 256
        world_size = 8
        topk_ids = torch.randint(0, num_experts, (10000, 8))
        shared_assignment = assign_shared_expert(topk_ids, num_experts, world_size)
        shared_counts = torch.bincount(shared_assignment, minlength=world_size)
        self.assertEqual(shared_counts.sum().item(), num_tokens)


class TestImbalanceScore(unittest.TestCase):
    def test_balance_reduces_max_load(self):
        num_tokens = 10000
        num_experts = 256
        world_size = 8
        topk = 8
        probs = torch.arange(num_experts, 0, -1, dtype=torch.float32)
        probs = probs / probs.sum()
        expert_ids = torch.multinomial(probs, num_tokens * topk, replacement=True)
        topk_ids = expert_ids.reshape(num_tokens, topk)

        _, _, total_balanced = compute_imbalance_score(
            topk_ids, num_experts, world_size, use_balance=True
        )
        _, _, total_unbalanced = compute_imbalance_score(
            topk_ids, num_experts, world_size, use_balance=False
        )

        max_balanced = total_balanced.max().item()
        max_unbalanced = total_unbalanced.max().item()

        print(f"Max load - Balanced: {max_balanced}, Unbalanced: {max_unbalanced}")
        self.assertLess(max_balanced, max_unbalanced)


if __name__ == "__main__":
    unittest.main()
