# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Mode 1 Boundary Tests: Empirical Validation of PoGQ Beyond 33% BFT

This test suite validates the core claim: Mode 1 (Ground Truth) exceeds
the 33-35% BFT ceiling of peer-comparison methods.

Tests:
1. Mode 1 at 35% BFT (peer-comparison boundary) - Should succeed
2. Mode 1 at 40% BFT - Should succeed
3. Mode 1 at 45% BFT (PoGQ whitepaper claim) - Should succeed
4. Mode 1 at 50% BFT (Mode 1 boundary) - Expected to fail/degrade

Author: Zero-TrustML Research Team
Date: November 4, 2025
Status: Critical validation tests
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List
import time

from src.ground_truth_detector import GroundTruthDetector, HybridGroundTruthDetector
from src.byzantine_attacks import create_attack, AttackType


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST testing."""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # After conv1: 28->26, pool: 26->13
        # After conv2: 13->11, pool: 11->5
        # So: 64 * 5 * 5 = 1600
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


def create_synthetic_mnist_data(num_samples: int = 1000, seed: int = 42, train: bool = True):
    """
    Create synthetic MNIST-like data with learnable BUT imperfect patterns.

    Training data: Noisy patterns with class overlap (targets ~75% accuracy, not 100%)
    Validation data: Same distribution

    CRITICAL: We intentionally make this challenging so models achieve 70-80% accuracy,
    simulating realistic federated learning scenarios where:
    - Honest gradients meaningfully improve the model
    - Byzantine gradients significantly degrade it
    - Quality score separation is clear (not microscopic)
    """
    np.random.seed(seed + (0 if train else 1000))
    torch.manual_seed(seed + (0 if train else 1000))

    images = torch.randn(num_samples, 1, 28, 28) * 0.3  # Start with noise
    labels = torch.randint(0, 10, (num_samples,))

    # Create OVERLAPPING patterns for each class (harder to learn)
    for i in range(num_samples):
        label = labels[i].item()

        # Each class has a pattern, but with significant overlap
        row = label // 5
        col = label % 5

        # Create pattern with 50% noise (makes learning imperfect)
        pattern_strength = 0.7 if np.random.rand() > 0.3 else 0.3  # Sometimes weak signal
        images[i, 0, row*4:row*4+8, col*4:col*4+8] += pattern_strength

        # Add confusing patterns from other classes (creates ambiguity)
        if np.random.rand() > 0.5:
            other_label = (label + np.random.randint(1, 10)) % 10
            other_row = other_label // 5
            other_col = other_label % 5
            images[i, 0, other_row*4:other_row*4+6, other_col*4:other_col*4+6] += 0.3

        # Significant random noise (prevents overfitting)
        images[i] += torch.randn(1, 28, 28) * 0.4

    # Normalize to [0, 1] range
    images = torch.clamp(images, 0, 1)

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True if train else False)

    return loader


def create_synthetic_mnist_validation(num_samples: int = 1000, seed: int = 42):
    """Create validation dataset with same patterns as training."""
    return create_synthetic_mnist_data(num_samples, seed, train=False)


def pretrain_model(model: nn.Module, num_epochs: int = 3, seed: int = 42, device: str = "cpu", target_accuracy: float = 75.0):
    """
    Pre-train the global model to establish baseline performance.

    This is critical: In real FL, the global model has been training for rounds.
    We need baseline performance so gradient quality can be measured.

    IMPORTANT: We target ~75% accuracy (not 100%) to simulate realistic FL scenarios.
    This creates meaningful quality score separation between honest and Byzantine.
    """
    print(f"Pre-training global model (target ~{target_accuracy:.0f}% accuracy)...")

    # Set seeds for reproducible pre-training
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create training data with learnable patterns
    train_loader = create_synthetic_mnist_data(num_samples=2000, seed=seed, train=True)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        accuracy = 100. * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.1f}%")

        # Stop if we reach target accuracy (don't overtrain to 100%)
        if accuracy >= target_accuracy and accuracy < 95.0:
            print(f"✓ Reached target accuracy ({accuracy:.1f}%). Stopping to maintain realistic FL scenario.\n")
            return model

    print(f"✓ Pre-training complete. Model has baseline performance.\n")
    return model


def generate_honest_gradient(
    model: nn.Module,
    device: str = "cpu",
    train_loader = None
) -> Dict[str, np.ndarray]:
    """
    Generate a realistic honest gradient from actual training.

    Honest clients:
    - Run forward/backward pass on their local data
    - Compute gradients that would improve model performance
    - Send these gradients to the aggregator
    """
    # Create fresh training data if not provided
    if train_loader is None:
        train_loader = create_synthetic_mnist_data(num_samples=100, seed=int(time.time()), train=True)

    # Create a copy of model for gradient computation
    import copy
    temp_model = copy.deepcopy(model)
    temp_model.train()

    criterion = nn.CrossEntropyLoss()

    # Compute gradient on one batch
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        temp_model.zero_grad()
        output = temp_model(data)
        loss = criterion(output, target)
        loss.backward()

        # Extract gradients
        gradient = {}
        for name, param in temp_model.named_parameters():
            if param.grad is not None:
                gradient[name] = param.grad.cpu().numpy().copy()

        # Only use first batch
        break

    return gradient


def generate_byzantine_gradient(
    model: nn.Module,
    attack_type: AttackType,
    intensity: float = 1.0,
    device: str = "cpu"
) -> Dict[str, np.ndarray]:
    """Generate Byzantine gradient using attack."""
    honest_grad = generate_honest_gradient(model, device)

    # Apply attack
    attack = create_attack(attack_type, flip_intensity=intensity)

    # Convert to flat array for attack
    flat_honest = np.concatenate([g.flatten() for g in honest_grad.values()])
    flat_byzantine = attack.generate(flat_honest, round_num=0)  # Use generate() method

    # Reshape back
    byzantine_grad = {}
    idx = 0
    for name, param in model.named_parameters():
        size = param.numel()
        byzantine_grad[name] = flat_byzantine[idx:idx+size].reshape(param.shape)
        idx += size

    return byzantine_grad


class Mode1BoundaryTest:
    """Test harness for Mode 1 boundary validation."""

    def __init__(
        self,
        num_clients: int,
        num_byzantine: int,
        test_name: str,
        seed: int = 42,
        pretrain_epochs: int = 5
    ):
        self.num_clients = num_clients
        self.num_byzantine = num_byzantine
        self.num_honest = num_clients - num_byzantine
        self.bft_percent = num_byzantine / num_clients
        self.test_name = test_name
        self.seed = seed

        # Set seeds for reproducible model initialization
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Initialize model
        self.device = "cpu"
        self.global_model = SimpleCNN().to(self.device)

        # CRITICAL: Pre-train the model to establish baseline performance
        # This simulates real FL where global model has been training for rounds
        print(f"\n{'='*80}")
        print(f"🔧 Initializing {test_name}")
        print(f"{'='*80}")
        self.global_model = pretrain_model(
            self.global_model,
            num_epochs=pretrain_epochs,
            seed=seed,
            device=self.device
        )

        # Create validation dataset (same distribution as training)
        self.validation_loader = create_synthetic_mnist_validation(
            num_samples=1000,
            seed=seed
        )

        # Initialize Mode 1 detector with ADAPTIVE THRESHOLD
        self.detector = GroundTruthDetector(
            global_model=self.global_model,
            validation_loader=self.validation_loader,
            quality_threshold=0.5,  # Fallback only (adaptive overrides this)
            learning_rate=0.01,
            device=self.device,
            adaptive_threshold=True,  # ENABLED: Robust to heterogeneous data
            mad_multiplier=3.0
        )

        # Results storage
        self.results = {
            'num_clients': num_clients,
            'num_byzantine': num_byzantine,
            'bft_percent': self.bft_percent,
            'honest_flagged': 0,
            'byzantine_detected': 0,
            'total_honest': self.num_honest,
            'total_byzantine': num_byzantine,
            'detection_rate': 0.0,
            'fpr': 0.0,
            'quality_scores': []
        }

    def run_test(self, num_rounds: int = 3) -> Dict:
        """Run Mode 1 boundary test."""
        print(f"\n{'='*80}")
        print(f"🧪 {self.test_name}")
        print(f"{'='*80}")
        print(f"Configuration:")
        print(f"  - Clients: {self.num_clients} ({self.num_honest} honest, {self.num_byzantine} Byzantine)")
        print(f"  - BFT Ratio: {self.bft_percent*100:.1f}%")
        print(f"  - Detector: Mode 1 (Ground Truth - PoGQ)")
        print(f"  - Adaptive Threshold: ENABLED (outlier-robust gap-based)")
        print(f"  - Data Distribution: HETEROGENEOUS (unique per client)")
        print(f"  - Rounds: {num_rounds}")
        print(f"  - Seed: {self.seed}")
        print()

        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1}/{num_rounds} ---")

            # Generate gradients with HETEROGENEOUS DATA
            # Each client gets unique local data for realistic FL
            gradients = {}
            ground_truth = {}  # Track which are actually Byzantine

            # Honest gradients - each with UNIQUE local data
            for i in range(self.num_honest):
                # Unique seed per client for heterogeneous data
                client_seed = self.seed + i * 1000 + round_num * 100000
                client_data = create_synthetic_mnist_data(
                    num_samples=100,
                    seed=client_seed,
                    train=True
                )
                gradients[i] = generate_honest_gradient(
                    self.global_model,
                    self.device,
                    train_loader=client_data
                )
                ground_truth[i] = False

            # Byzantine gradients - each with UNIQUE local data + sign flip
            for i in range(self.num_honest, self.num_clients):
                # Unique seed per client
                client_seed = self.seed + i * 1000 + round_num * 100000
                client_data = create_synthetic_mnist_data(
                    num_samples=100,
                    seed=client_seed,
                    train=True
                )

                # Generate honest gradient from their local data
                honest_grad = generate_honest_gradient(
                    self.global_model,
                    self.device,
                    train_loader=client_data
                )

                # Then sign flip it (Byzantine attack)
                attack = create_attack(AttackType.SIGN_FLIP, flip_intensity=1.0)
                flat_honest = np.concatenate([g.flatten() for g in honest_grad.values()])
                flat_byz = attack.generate(flat_honest, round_num=round_num)

                # Reshape back to gradient dict
                byz_grad = {}
                idx = 0
                for name, param in self.global_model.named_parameters():
                    size = param.numel()
                    byz_grad[name] = flat_byz[idx:idx+size].reshape(param.shape)
                    idx += size

                gradients[i] = byz_grad
                ground_truth[i] = True

            # Run detection
            detections = self.detector.detect_byzantine(gradients)

            # Compute metrics
            honest_flagged = sum(
                1 for node_id in range(self.num_honest)
                if detections[node_id]
            )
            byzantine_detected = sum(
                1 for node_id in range(self.num_honest, self.num_clients)
                if detections[node_id]
            )

            fpr = honest_flagged / self.num_honest if self.num_honest > 0 else 0.0
            detection_rate = byzantine_detected / self.num_byzantine if self.num_byzantine > 0 else 0.0

            print(f"  Byzantine Detection: {byzantine_detected}/{self.num_byzantine} ({detection_rate*100:.1f}%)")
            print(f"  Honest Flagged (FPR): {honest_flagged}/{self.num_honest} ({fpr*100:.1f}%)")
            print(f"  Adaptive Threshold: {self.detector.computed_threshold:.6f}")

            # Update cumulative results
            self.results['honest_flagged'] += honest_flagged
            self.results['byzantine_detected'] += byzantine_detected

        # Compute final metrics
        total_honest_opportunities = self.num_honest * num_rounds
        total_byzantine_opportunities = self.num_byzantine * num_rounds

        self.results['detection_rate'] = (
            self.results['byzantine_detected'] / total_byzantine_opportunities
            if total_byzantine_opportunities > 0 else 0.0
        )
        self.results['fpr'] = (
            self.results['honest_flagged'] / total_honest_opportunities
            if total_honest_opportunities > 0 else 0.0
        )

        # Get quality statistics
        stats = self.detector.get_quality_statistics()
        self.results['quality_stats'] = stats

        # Print summary
        self.print_summary()

        return self.results

    def print_summary(self):
        """Print test summary."""
        print(f"\n{'='*80}")
        print(f"📊 RESULTS: {self.test_name}")
        print(f"{'='*80}")
        print(f"Detection Performance:")
        print(f"  - Byzantine Detection Rate: {self.results['detection_rate']*100:.1f}% ({self.results['byzantine_detected']}/{self.results['total_byzantine']*3})")
        print(f"  - False Positive Rate: {self.results['fpr']*100:.1f}% ({self.results['honest_flagged']}/{self.results['total_honest']*3})")
        print(f"\nAdaptive Threshold (last round): {self.detector.computed_threshold:.6f}")
        print(f"\nQuality Score Statistics:")
        stats = self.results['quality_stats']
        print(f"  - Mean Quality: {stats['mean_quality']:.3f}")
        print(f"  - Std Quality: {stats['std_quality']:.3f}")
        print(f"  - Min Quality: {stats['min_quality']:.3f}")
        print(f"  - Max Quality: {stats['max_quality']:.3f}")
        print(f"  - Range: {stats['max_quality'] - stats['min_quality']:.3f} ({(stats['max_quality'] - stats['min_quality'])*100:.1f}% spread)")

        # Success criteria
        print(f"\n{'='*80}")
        print(f"✅ SUCCESS CRITERIA:")
        print(f"{'='*80}")

        if self.bft_percent <= 0.35:
            # At 35% BFT: Should have excellent performance
            detection_pass = self.results['detection_rate'] >= 0.95
            fpr_pass = self.results['fpr'] <= 0.10
            print(f"  Expected at {self.bft_percent*100:.0f}% BFT (Heterogeneous Data):")
            print(f"    - Detection ≥95%: {'✅ PASS' if detection_pass else '❌ FAIL'} ({self.results['detection_rate']*100:.1f}%)")
            print(f"    - FPR ≤10%: {'✅ PASS' if fpr_pass else '❌ FAIL'} ({self.results['fpr']*100:.1f}%)")
            self.results['test_passed'] = detection_pass and fpr_pass
        elif self.bft_percent <= 0.45:
            # At 40-45% BFT: Should still work well (PoGQ claim)
            detection_pass = self.results['detection_rate'] >= 0.85
            fpr_pass = self.results['fpr'] <= 0.15
            print(f"  Expected at {self.bft_percent*100:.0f}% BFT (Heterogeneous Data):")
            print(f"    - Detection ≥85%: {'✅ PASS' if detection_pass else '❌ FAIL'} ({self.results['detection_rate']*100:.1f}%)")
            print(f"    - FPR ≤15%: {'✅ PASS' if fpr_pass else '❌ FAIL'} ({self.results['fpr']*100:.1f}%)")
            self.results['test_passed'] = detection_pass and fpr_pass
        else:
            # At >45% BFT: Expected to degrade
            print(f"  Expected at {self.bft_percent*100:.0f}% BFT:")
            print(f"    - System should degrade (>45% is beyond Mode 1 capacity)")
            print(f"    - Actual Detection: {self.results['detection_rate']*100:.1f}%")
            print(f"    - Actual FPR: {self.results['fpr']*100:.1f}%")
            self.results['test_passed'] = True  # Degradation is expected

        print(f"\n{'='*80}")
        overall = "✅ PASSED" if self.results['test_passed'] else "❌ FAILED"
        print(f"Overall: {overall}")
        print(f"{'='*80}\n")


# Test functions
def test_mode1_35_bft():
    """Test Mode 1 at 35% BFT (peer-comparison boundary)."""
    test = Mode1BoundaryTest(
        num_clients=20,
        num_byzantine=7,  # 35%
        test_name="Mode 1 at 35% BFT (Peer-Comparison Boundary)",
        seed=42
    )
    # Use single round for realistic FL testing
    # (Multi-round requires model updates between rounds)
    results = test.run_test(num_rounds=1)

    # Assertions
    assert results['detection_rate'] >= 0.95, f"Detection rate {results['detection_rate']*100:.1f}% < 95%"
    assert results['fpr'] <= 0.10, f"FPR {results['fpr']*100:.1f}% > 10%"
    assert results['test_passed'], "Test did not pass success criteria"


def test_mode1_40_bft():
    """Test Mode 1 at 40% BFT."""
    test = Mode1BoundaryTest(
        num_clients=20,
        num_byzantine=8,  # 40%
        test_name="Mode 1 at 40% BFT",
        seed=42
    )
    results = test.run_test(num_rounds=1)

    # Assertions (more lenient at higher BFT)
    assert results['detection_rate'] >= 0.85, f"Detection rate {results['detection_rate']*100:.1f}% < 85%"
    assert results['fpr'] <= 0.15, f"FPR {results['fpr']*100:.1f}% > 15%"
    assert results['test_passed'], "Test did not pass success criteria"


def test_mode1_45_bft():
    """Test Mode 1 at 45% BFT (PoGQ whitepaper claim)."""
    test = Mode1BoundaryTest(
        num_clients=20,
        num_byzantine=9,  # 45%
        test_name="Mode 1 at 45% BFT (PoGQ Whitepaper Validation)",
        seed=42
    )
    results = test.run_test(num_rounds=1)

    # Assertions (PoGQ claim: should work at 45%)
    assert results['detection_rate'] >= 0.85, f"Detection rate {results['detection_rate']*100:.1f}% < 85%"
    assert results['fpr'] <= 0.15, f"FPR {results['fpr']*100:.1f}% > 15%"
    assert results['test_passed'], "Test did not pass success criteria"


def test_mode1_50_bft():
    """Test Mode 1 at 50% BFT (Mode 1 boundary)."""
    test = Mode1BoundaryTest(
        num_clients=20,
        num_byzantine=10,  # 50%
        test_name="Mode 1 at 50% BFT (Mode 1 Boundary)",
        seed=42
    )
    results = test.run_test(num_rounds=1)

    # At 50%, we expect degradation but not complete failure
    # This documents the Mode 1 boundary
    print(f"\n📊 Mode 1 Boundary Behavior at 50% BFT:")
    print(f"  - Detection: {results['detection_rate']*100:.1f}%")
    print(f"  - FPR: {results['fpr']*100:.1f}%")
    print(f"  - (Degradation expected at this level)")


def test_multiseed_mode1_45_bft():
    """Multi-seed validation at 45% BFT for statistical robustness."""
    seeds = [42, 123, 456]
    results = []

    print(f"\n{'='*80}")
    print(f"🎲 MULTI-SEED VALIDATION: Mode 1 at 45% BFT")
    print(f"{'='*80}\n")

    for seed in seeds:
        test = Mode1BoundaryTest(
            num_clients=20,
            num_byzantine=9,
            test_name=f"Mode 1 at 45% BFT (Seed {seed})",
            seed=seed
        )
        result = test.run_test(num_rounds=1)
        results.append(result)

    # Compute statistics across seeds
    detection_rates = [r['detection_rate'] for r in results]
    fprs = [r['fpr'] for r in results]

    print(f"\n{'='*80}")
    print(f"📊 MULTI-SEED SUMMARY")
    print(f"{'='*80}")
    print(f"Detection Rate: {np.mean(detection_rates)*100:.1f}% ± {np.std(detection_rates)*100:.1f}%")
    print(f"FPR: {np.mean(fprs)*100:.1f}% ± {np.std(fprs)*100:.1f}%")
    print(f"Success Rate: {sum(r['test_passed'] for r in results)}/{len(seeds)}")
    print(f"{'='*80}\n")

    # All seeds should pass
    assert all(r['test_passed'] for r in results), "Not all seeds passed"


if __name__ == "__main__":
    # Run all tests
    print("\n" + "="*80)
    print("🚀 MODE 1 BOUNDARY VALIDATION SUITE")
    print("="*80 + "\n")

    # Critical tests
    print("\n### Test 1: Mode 1 at 35% BFT ###")
    test_mode1_35_bft()

    print("\n### Test 2: Mode 1 at 40% BFT ###")
    test_mode1_40_bft()

    print("\n### Test 3: Mode 1 at 45% BFT (PoGQ Validation) ###")
    test_mode1_45_bft()

    print("\n### Test 4: Mode 1 at 50% BFT (Boundary) ###")
    test_mode1_50_bft()

    print("\n### Test 5: Multi-Seed Validation (45% BFT) ###")
    test_multiseed_mode1_45_bft()

    print("\n" + "="*80)
    print("✅ ALL MODE 1 BOUNDARY TESTS COMPLETE")
    print("="*80 + "\n")
