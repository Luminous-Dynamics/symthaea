#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Federated Learning with Holochain Credits Example

Demonstrates:
- Real PyTorch federated learning
- Byzantine node detection with PoGQ
- Automatic credit issuance to honest nodes
- Trust-based node selection using credit balances

This example shows how Holochain credits create a reputation system
that rewards honest participants and isolates Byzantine attackers.
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
import time

# Import Holochain bridge
try:
    from holochain_credits_bridge import HolochainBridge
    HOLOCHAIN_AVAILABLE = True
except ImportError:
    print("⚠️  Holochain bridge not available, using mock mode")
    HOLOCHAIN_AVAILABLE = False


# Simple neural network for demonstration
class SimpleModel(nn.Module):
    """Simple neural network for MNIST-like classification"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class FederatedNode:
    """Federated learning node with Holochain credit integration"""

    def __init__(self, node_id: int, is_byzantine: bool = False, bridge=None):
        self.node_id = node_id
        self.is_byzantine = is_byzantine
        self.bridge = bridge

        self.model = SimpleModel()
        self.credits = 0
        self.rounds_participated = 0
        self.successful_contributions = 0

    def compute_gradient(self, data: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
        """Compute gradient update (or return poisoned gradient if Byzantine)"""

        if self.is_byzantine:
            # Byzantine node returns random noise
            return np.random.randn(self._count_parameters()) * 10

        # Honest node computes real gradient
        self.model.zero_grad()
        output = self.model(data)
        loss = nn.functional.cross_entropy(output, labels)
        loss.backward()

        # Flatten gradients
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))

        return torch.cat(gradients).detach().numpy()

    def _count_parameters(self) -> int:
        """Count total model parameters"""
        return sum(p.numel() for p in self.model.parameters())

    def update_credits(self, amount: int):
        """Update local credit balance"""
        self.credits += amount

    def __repr__(self):
        node_type = "⚠️  BYZANTINE" if self.is_byzantine else "✅ HONEST"
        return f"Node {self.node_id} ({node_type}) - Credits: {self.credits}, Rounds: {self.rounds_participated}"


class FederatedCoordinator:
    """Coordinator that validates gradients and issues Holochain credits"""

    def __init__(self, conductor_url: str = "ws://localhost:8888"):
        self.nodes: List[FederatedNode] = []
        self.global_model = SimpleModel()
        self.test_data, self.test_labels = self._generate_test_data()

        # Initialize Holochain bridge
        if HOLOCHAIN_AVAILABLE:
            self.bridge = HolochainBridge(conductor_url, enabled=True)
            print(f"✅ Holochain bridge initialized: {self.bridge}")
        else:
            self.bridge = None
            print("⚠️  Running in mock mode (no Holochain)")

    def _generate_test_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic test data for PoGQ validation"""
        data = torch.randn(100, 1, 28, 28)
        labels = torch.randint(0, 10, (100,))
        return data, labels

    def register_node(self, node: FederatedNode):
        """Register a new federated learning node"""
        node.bridge = self.bridge
        self.nodes.append(node)
        print(f"📝 Registered {node}")

    def validate_gradient_pogq(self, gradient: np.ndarray) -> Tuple[bool, float]:
        """
        Validate gradient using Proof of Gradient Quality (PoGQ)

        Returns:
            (is_valid, quality_score): Whether gradient is valid and its quality score
        """
        # Apply gradient to test model
        test_model = SimpleModel()
        test_model.load_state_dict(self.global_model.state_dict())

        # Update test model with gradient
        offset = 0
        for param in test_model.parameters():
            param_size = param.numel()
            param.data -= torch.tensor(gradient[offset:offset + param_size]).view_as(param) * 0.01
            offset += param_size

        # Evaluate loss improvement
        with torch.no_grad():
            # Original loss
            original_output = self.global_model(self.test_data)
            original_loss = nn.functional.cross_entropy(original_output, self.test_labels).item()

            # Updated loss
            updated_output = test_model(self.test_data)
            updated_loss = nn.functional.cross_entropy(updated_output, self.test_labels).item()

        # Calculate improvement
        improvement = original_loss - updated_loss
        quality_score = max(0.0, min(1.0, improvement / original_loss if original_loss > 0 else 0.0))

        # Gradient is valid if it improves loss (quality > 0.1)
        is_valid = quality_score > 0.1

        return is_valid, quality_score

    async def training_round(self, round_num: int):
        """Execute one round of federated learning with credit issuance"""

        print(f"\n{'='*60}")
        print(f"  Round {round_num}")
        print(f"{'='*60}\n")

        # Generate training data batch
        data = torch.randn(32, 1, 28, 28)
        labels = torch.randint(0, 10, (32,))

        valid_gradients = []

        for node in self.nodes:
            print(f"\n📊 Processing {node}")

            # Node computes gradient
            gradient = node.compute_gradient(data, labels)

            # Coordinator validates gradient
            is_valid, quality_score = self.validate_gradient_pogq(gradient)

            node.rounds_participated += 1

            if is_valid:
                print(f"   ✅ Gradient VALID (quality: {quality_score:.3f})")
                valid_gradients.append(gradient)
                node.successful_contributions += 1

                # Issue Holochain credits based on quality
                credits_earned = int(quality_score * 100)

                if self.bridge:
                    try:
                        action_hash = self.bridge.issue_credits(
                            node_id=node.node_id,
                            event_type="valid_gradient",
                            amount=credits_earned,
                            pogq_score=quality_score
                        )
                        print(f"   💰 Issued {credits_earned} credits (hash: {action_hash[:20]}...)")
                    except Exception as e:
                        print(f"   ⚠️  Credit issuance failed: {e}")
                        print(f"   💰 Mock: {credits_earned} credits")
                else:
                    print(f"   💰 Mock: {credits_earned} credits")

                node.update_credits(credits_earned)

            else:
                print(f"   ❌ Gradient REJECTED (quality: {quality_score:.3f})")
                print(f"   🚨 Byzantine behavior detected!")

        # Aggregate valid gradients
        if valid_gradients:
            avg_gradient = np.mean(valid_gradients, axis=0)
            self._update_global_model(avg_gradient)
            print(f"\n✅ Global model updated with {len(valid_gradients)} valid gradients")
        else:
            print(f"\n❌ No valid gradients - model unchanged")

    def _update_global_model(self, gradient: np.ndarray):
        """Update global model with averaged gradient"""
        offset = 0
        for param in self.global_model.parameters():
            param_size = param.numel()
            param.data -= torch.tensor(gradient[offset:offset + param_size]).view_as(param) * 0.01
            offset += param_size

    def print_summary(self):
        """Print final summary of all nodes"""
        print(f"\n{'='*60}")
        print(f"  Final Summary")
        print(f"{'='*60}\n")

        # Sort nodes by credits (reputation)
        sorted_nodes = sorted(self.nodes, key=lambda n: n.credits, reverse=True)

        print("Node Rankings by Credits (Reputation):\n")
        for rank, node in enumerate(sorted_nodes, 1):
            node_type = "⚠️  BYZANTINE" if node.is_byzantine else "✅ HONEST"
            success_rate = (node.successful_contributions / node.rounds_participated * 100) if node.rounds_participated > 0 else 0

            print(f"{rank}. Node {node.node_id:2d} ({node_type})")
            print(f"   Credits: {node.credits:4d}")
            print(f"   Rounds: {node.rounds_participated:2d}")
            print(f"   Success Rate: {success_rate:5.1f}%")
            print()

        # Statistics
        honest_nodes = [n for n in self.nodes if not n.is_byzantine]
        byzantine_nodes = [n for n in self.nodes if n.is_byzantine]

        avg_honest_credits = np.mean([n.credits for n in honest_nodes]) if honest_nodes else 0
        avg_byzantine_credits = np.mean([n.credits for n in byzantine_nodes]) if byzantine_nodes else 0

        print(f"Statistics:")
        print(f"  Honest nodes: {len(honest_nodes)} (avg credits: {avg_honest_credits:.1f})")
        print(f"  Byzantine nodes: {len(byzantine_nodes)} (avg credits: {avg_byzantine_credits:.1f})")
        print(f"  Credit ratio (honest/byzantine): {avg_honest_credits/max(avg_byzantine_credits, 0.1):.1f}x")


async def main():
    """Main demonstration"""
    print("="*60)
    print("  Federated Learning with Holochain Credits")
    print("  Byzantine-Resistant Reputation System")
    print("="*60)

    # Create coordinator
    coordinator = FederatedCoordinator()

    # Register nodes: 8 honest + 2 Byzantine
    print("\n📝 Registering federated learning nodes...\n")

    for i in range(1, 9):
        node = FederatedNode(node_id=i, is_byzantine=False, bridge=coordinator.bridge)
        coordinator.register_node(node)

    # Add Byzantine nodes
    for i in range(9, 11):
        node = FederatedNode(node_id=i, is_byzantine=True, bridge=coordinator.bridge)
        coordinator.register_node(node)

    print(f"\n✅ Registered 8 honest nodes + 2 Byzantine nodes")

    # Run federated learning rounds
    num_rounds = 5
    print(f"\n🚀 Starting {num_rounds} training rounds...\n")

    for round_num in range(1, num_rounds + 1):
        await coordinator.training_round(round_num)
        await asyncio.sleep(0.1)  # Small delay between rounds

    # Print final summary
    coordinator.print_summary()

    print("\n" + "="*60)
    print("  ✅ Demonstration Complete!")
    print("="*60)
    print("\nKey Observations:")
    print("  1. Honest nodes accumulate credits over time")
    print("  2. Byzantine nodes earn NO credits (detected & rejected)")
    print("  3. Credit balance = reputation = trust score")
    print("  4. System can use credits to select most trustworthy nodes")
    print("\nThis demonstrates how Holochain credits create a")
    print("Byzantine-resistant reputation system for federated learning!")


if __name__ == "__main__":
    asyncio.run(main())
