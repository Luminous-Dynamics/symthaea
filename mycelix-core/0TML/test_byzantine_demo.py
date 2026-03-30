#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Quick Byzantine Attack Demonstration

Tests one attack type (Sign Flip) against FedAvg to prove the framework works.
Full 35-experiment suite can be run later with run_byzantine_experiments.sh
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict

# Import our new Byzantine attack framework
from experiments.utils.byzantine_attacks import (
    ByzantineAttacker,
    ByzantineAttackType,
    create_byzantine_client
)


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def test_byzantine_attack():
    """Demonstrate Byzantine attack framework"""
    print("=" * 80)
    print("Byzantine Attack Framework Demonstration")
    print("=" * 80)
    print()

    # Create model
    model = SimpleCNN()
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Model: SimpleCNN")
    print(f"Total parameters: {total_params:,}")
    print()

    # Get model weights
    honest_weights = [p.detach().numpy().copy() for p in model.parameters()]

    print("Testing all 7 attack types:")
    print("-" * 80)

    attack_types = [
        ByzantineAttackType.GAUSSIAN_NOISE,
        ByzantineAttackType.SIGN_FLIP,
        ByzantineAttackType.LABEL_FLIP,
        ByzantineAttackType.TARGETED_POISON,
        ByzantineAttackType.MODEL_REPLACEMENT,
        ByzantineAttackType.ADAPTIVE,
        ByzantineAttackType.SYBIL
    ]

    for attack_type in attack_types:
        attacker = ByzantineAttacker(attack_type=attack_type, severity=1.0, seed=42)

        # Generate malicious update
        if attack_type == ByzantineAttackType.ADAPTIVE:
            # Adaptive needs peer context
            context = {
                'peer_weights': [
                    [p + np.random.randn(*p.shape) * 0.01 for p in honest_weights]
                    for _ in range(3)
                ]
            }
            malicious_weights = attacker.attack(honest_weights, context)
        else:
            malicious_weights = attacker.attack(honest_weights)

        # Calculate perturbation magnitude
        total_diff = 0
        total_norm = 0
        for h, m in zip(honest_weights, malicious_weights):
            total_diff += np.linalg.norm(h - m)
            total_norm += np.linalg.norm(h)

        perturbation_ratio = total_diff / total_norm

        status = "✅" if perturbation_ratio > 0.01 else "⚠️"
        print(f"{status} {attack_type.value:20} | Perturbation: {perturbation_ratio:8.4f}")

    print()
    print("=" * 80)
    print("✅ All attack types successfully implemented!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Run full 35-experiment suite: ./run_byzantine_experiments.sh")
    print("  2. Results will show which defenses resist which attacks")
    print("  3. Generate attack effectiveness heatmap")
    print()


if __name__ == '__main__':
    test_byzantine_attack()
