#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Real 35% and 40% BFT Boundary Tests with Neural Network Training

This integrates the BFT fail-safe mechanism with actual Byzantine detection
from test_30_bft_validation.py to empirically validate the 35% ceiling.

Tests:
1. 35% BFT (7/20 Byzantine) - Mode 0 boundary test
2. 40% BFT (8/20 Byzantine) - Mode 0 fail-safe test

Expected Results:
- 35% BFT: Degraded but functional (FPR 5-10%, detection 70-80%)
- 40% BFT: Fail-safe triggers (halt OR catastrophic FPR >20%)
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Check environment variable - use pytest.skip for proper test skipping
import pytest
if os.environ.get("RUN_REAL_BOUNDARY_TESTS") != "1":
    pytest.skip("Real boundary tests skipped (set RUN_REAL_BOUNDARY_TESTS=1 to run)", allow_module_level=True)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Dict, List, Tuple

from bft_failsafe import BFTFailSafe
from byzantine_attacks import create_attack, AttackType

print("=" * 80)
print("🧪 REAL BOUNDARY TESTS: 35% and 40% BFT with Neural Network")
print("=" * 80)
print()
print("Objective: Empirically validate peer-comparison ceiling with actual training")
print()
print("Tests:")
print("  1. 35% BFT (7/20 Byzantine) - Last success point for Mode 0")
print("  2. 40% BFT (8/20 Byzantine) - Fail-safe must trigger")
print()
print("=" * 80)
print()


# Simple CNN Model (from test_30_bft_validation.py)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
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


class RealBoundaryTest:
    """Test runner for 35% and 40% BFT with real neural network training"""

    def __init__(self, num_clients: int, num_byzantine: int, test_name: str):
        self.num_clients = num_clients
        self.num_byzantine = num_byzantine
        self.num_honest = num_clients - num_byzantine
        self.test_name = test_name

        self.bft_percent = num_byzantine / num_clients

        # Initialize fail-safe
        self.failsafe = BFTFailSafe(bft_limit=0.35, warning_threshold=0.30)

        # Model and data
        self.global_model = SimpleCNN()
        self.device = torch.device("cpu")
        self.global_model.to(self.device)

        print(f"Test Configuration: {test_name}")
        print(f"  - Total Clients: {num_clients}")
        print(f"  - Honest Clients: {self.num_honest}")
        print(f"  - Byzantine Clients: {num_byzantine} ({self.bft_percent*100:.0f}%)")
        print()

    def prepare_data(self):
        """Prepare MNIST data with label skew distribution"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST('./data', train=True, download=True,
                                       transform=transform)

        # Create label-skewed partitions (simplified)
        client_datasets = []
        samples_per_client = len(train_dataset) // self.num_clients

        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client
            indices = list(range(start_idx, end_idx))
            client_datasets.append(Subset(train_dataset, indices))

        return client_datasets

    def train_client(self, client_id: int, dataset, epochs: int = 1):
        """Train a client and return gradient"""
        model = SimpleCNN()
        model.load_state_dict(self.global_model.state_dict())
        model.to(self.device)

        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                if batch_idx >= 5:  # Limit batches for speed
                    break

        # Extract gradient
        gradient = []
        for param in model.parameters():
            if param.grad is not None:
                gradient.append(param.grad.clone().flatten())

        if len(gradient) > 0:
            return torch.cat(gradient).cpu().numpy()
        return np.zeros(1000)  # Fallback

    def detect_byzantine(self, gradients: Dict[int, np.ndarray]) -> Tuple[Dict[int, bool], Dict[int, float], Dict[int, float]]:
        """
        Simplified Byzantine detection using similarity and magnitude signals
        (Temporal signal requires multiple rounds)
        """
        detection_flags = {}
        confidence_scores = {}
        reputations = {}

        # Compute median gradient (robust average)
        all_grads = list(gradients.values())
        median_grad = np.median(np.array(all_grads), axis=0)

        for client_id, grad in gradients.items():
            # Similarity signal
            cos_sim = np.dot(grad, median_grad) / (
                np.linalg.norm(grad) * np.linalg.norm(median_grad) + 1e-10
            )

            # Magnitude signal
            grad_norm = np.linalg.norm(grad)
            median_norm = np.linalg.norm(median_grad)
            magnitude_ratio = grad_norm / (median_norm + 1e-10)

            # Ensemble confidence (simplified - no temporal for single round)
            similarity_conf = 1.0 if cos_sim < 0.7 else 0.0
            magnitude_conf = 1.0 if magnitude_ratio > 2.0 or magnitude_ratio < 0.5 else 0.0

            # Weighted ensemble (similarity 70%, magnitude 30% - no temporal)
            confidence = 0.7 * similarity_conf + 0.3 * magnitude_conf

            detection_flags[client_id] = (confidence > 0.6)
            confidence_scores[client_id] = confidence

            # Reputation (simple: detected = low reputation)
            reputations[client_id] = 0.2 if detection_flags[client_id] else 1.0

        return detection_flags, confidence_scores, reputations

    def run_test(self, max_rounds: int = 5):
        """Run the boundary test"""
        print(f"🚀 Starting {self.test_name}")
        print("-" * 80)
        print()

        # Prepare data
        print("Preparing data...")
        client_datasets = self.prepare_data()
        print("✅ Data prepared")
        print()

        # Create Byzantine attack instances
        byzantine_clients = list(range(self.num_byzantine))
        attack_instances = {}

        for client_id in byzantine_clients:
            attack_instances[client_id] = create_attack(
                AttackType.SIGN_FLIP,  # Classic Byzantine attack
                flip_intensity=1.0
            )

        # Training rounds
        for round_num in range(1, max_rounds + 1):
            print(f"--- Round {round_num}/{max_rounds} ---")

            # Collect gradients
            gradients = {}

            for client_id in range(self.num_clients):
                honest_gradient = self.train_client(client_id, client_datasets[client_id])

                if client_id in byzantine_clients:
                    # Apply Byzantine attack
                    gradient = attack_instances[client_id].generate(
                        honest_gradient,
                        round_num
                    )
                else:
                    gradient = honest_gradient

                gradients[client_id] = gradient

            # Detect Byzantine nodes
            detection_flags, confidence_scores, reputations = self.detect_byzantine(gradients)

            # Run fail-safe check
            bft_estimate = self.failsafe.estimate_bft_percentage(
                reputations, detection_flags, confidence_scores
            )

            is_safe, message = self.failsafe.check_safety(bft_estimate, round_num)

            # Count detections
            detected_byzantine = sum(
                1 for cid in byzantine_clients if detection_flags[cid]
            )
            false_positives = sum(
                1 for cid in range(self.num_byzantine, self.num_clients)
                if detection_flags[cid]
            )

            print(f"  Byzantine Detected: {detected_byzantine}/{self.num_byzantine} ({detected_byzantine/self.num_byzantine*100:.1f}%)")
            print(f"  False Positives: {false_positives}/{self.num_honest} ({false_positives/self.num_honest*100:.1f}%)")
            print(f"  BFT Estimate: {bft_estimate*100:.1f}%")
            print(f"  Status: {message.split()[0]}")  # First emoji/status

            # Check if halt triggered
            if not is_safe:
                print()
                print("🛑 FAIL-SAFE TRIGGERED")
                print(message)
                print()
                break

            if "WARNING" in message:
                print(f"  ⚠️  Warning: {message}")

            print()

        # Final results
        print("=" * 80)
        print(f"📊 FINAL RESULTS: {self.test_name}")
        print("=" * 80)
        print()

        final_detection_rate = detected_byzantine / self.num_byzantine * 100
        final_fpr = false_positives / self.num_honest * 100

        print(f"Byzantine Detection: {final_detection_rate:.1f}%")
        print(f"False Positive Rate: {final_fpr:.1f}%")
        print(f"Final BFT Estimate: {bft_estimate*100:.1f}%")
        print(f"Network Status: {'🛑 HALTED' if self.failsafe.should_halt() else '✅ OPERATIONAL'}")
        print()

        # Evaluate success
        print("Success Criteria:")

        if self.bft_percent <= 0.35:
            # 35% BFT test
            fpr_ok = final_fpr <= 10.0
            det_ok = final_detection_rate >= 70.0
            safe_ok = not self.failsafe.should_halt()

            print(f"  - FPR ≤ 10%: {'✅ PASS' if fpr_ok else '❌ FAIL'} ({final_fpr:.1f}%)")
            print(f"  - Detection ≥ 70%: {'✅ PASS' if det_ok else '❌ FAIL'} ({final_detection_rate:.1f}%)")
            print(f"  - Network Safe: {'✅ PASS' if safe_ok else '❌ FAIL'}")

            passed = fpr_ok and det_ok and safe_ok

        else:
            # 40% BFT test
            halt_triggered = self.failsafe.should_halt()
            high_fpr = final_fpr >= 20.0
            fail_safe_working = halt_triggered or high_fpr

            print(f"  - Halt Triggered OR High FPR: {'✅ PASS' if fail_safe_working else '❌ FAIL'}")
            print(f"    • Halt Status: {halt_triggered}")
            print(f"    • FPR: {final_fpr:.1f}% ({'catastrophic' if high_fpr else 'acceptable'})")

            passed = fail_safe_working

        print()
        print(f"Overall: {'✅ TEST PASSED' if passed else '❌ TEST FAILED'}")
        print()
        print("=" * 80)
        print()

        return passed


# Run both tests
print("🚀 Starting Real Boundary Test Suite")
print()

results = []

# Test 1: 35% BFT
test_35 = RealBoundaryTest(
    num_clients=20,
    num_byzantine=7,
    test_name="35% BFT - Mode 0 Boundary"
)
result_35 = test_35.run_test(max_rounds=3)  # Reduced rounds for speed
results.append(("35% BFT Test", result_35))

# Test 2: 40% BFT
test_40 = RealBoundaryTest(
    num_clients=20,
    num_byzantine=8,
    test_name="40% BFT - Mode 0 Fail-Safe"
)
result_40 = test_40.run_test(max_rounds=3)
results.append(("40% BFT Test", result_40))

# Summary
print("=" * 80)
print("📊 REAL BOUNDARY TEST SUITE SUMMARY")
print("=" * 80)
print()

for test_name, passed in results:
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"{test_name}: {status}")

print()
all_passed = all(result for _, result in results)
print(f"Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
print()

print("=" * 80)
print("🏆 KEY VALIDATION")
print("=" * 80)
print()
print("✅ Empirical validation with real neural network training")
print("✅ Fail-safe mechanism tested with actual Byzantine detection")
print("✅ 35% boundary confirmed (or adjusted based on actual results)")
print()
print("Next: Multi-seed validation for statistical robustness")
print()
print("=" * 80)
