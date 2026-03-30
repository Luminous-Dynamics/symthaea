# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test: Full 0TML Hybrid Detector at 35% BFT

This test validates our ACTUAL 0TML Hybrid Byzantine Detector (with all signals enabled)
at the critical 35% BFT boundary where simplified peer-comparison methods fail.

CRITICAL VALIDATION:
- We already proved a simplified Mode 0 (peer-comparison) fails at 35% BFT (100% FPR)
- We already proved Mode 1 (ground truth) succeeds at 35% BFT (0% FPR)
- NOW we must prove our FULL 0TML Hybrid Detector succeeds at 35% BFT (<10% FPR)

The Full 0TML Hybrid Detector includes:
1. Similarity Signal (cosine similarity analysis) - 50% weight
2. Temporal Consistency Signal (behavioral tracking) - 30% weight
3. Magnitude Distribution Signal (gradient norm analysis) - 20% weight
4. Ensemble Voting System (weighted combination with 0.6 threshold)

Expected Outcome:
- Detection Rate: ≥95% (catch Byzantine nodes)
- False Positive Rate: <10% (don't flag honest nodes)
- This proves our multi-signal architecture prevents detector inversion

Author: Zero-TrustML Research Team
Date: November 5, 2025
Status: CRITICAL VALIDATION GAP - Paper completion depends on this
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List
import time

from src.byzantine_detection.hybrid_detector import HybridByzantineDetector
from src.byzantine_detection.gradient_analyzer import GradientDimensionalityAnalyzer
from test_mode1_boundaries import (
    SimpleCNN,
    create_synthetic_mnist_data,
    generate_honest_gradient,
    pretrain_model
)
from src.byzantine_attacks import create_attack, AttackType


def print_header(title: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"📊 {title}")
    print("=" * 80)


class FullHybridDetectorTest:
    """Test harness for Full 0TML Hybrid Detector validation."""

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

        # Pre-train model to establish baseline
        print_header("Initializing Test Environment")
        print(f"Test: {test_name}")
        print(f"Configuration: {num_clients} clients ({self.num_honest} honest, {num_byzantine} Byzantine = {self.bft_percent*100:.0f}% BFT)")
        print()

        self.global_model = pretrain_model(
            self.global_model,
            num_epochs=pretrain_epochs,
            seed=seed,
            device=self.device
        )

        # Initialize Full 0TML Hybrid Detector with ALL SIGNALS ENABLED
        print_header("Initializing Full 0TML Hybrid Detector")
        print("Configuration:")
        print("  - Similarity Signal: ENABLED (50% weight)")
        print("  - Temporal Signal: ENABLED (30% weight)")
        print("  - Magnitude Signal: ENABLED (20% weight)")
        print("  - Ensemble Voting: ENABLED (0.6 threshold)")
        print()

        self.detector = HybridByzantineDetector(
            # Temporal detector (behavioral tracking over 5 rounds)
            temporal_window_size=5,
            temporal_cosine_var_threshold=0.1,
            temporal_magnitude_var_threshold=0.5,
            temporal_min_observations=1,  # Start detecting after 1 observation

            # Magnitude detector (outlier detection)
            magnitude_z_score_threshold=3.0,
            magnitude_min_samples=3,

            # Ensemble weights (multi-signal fusion)
            similarity_weight=0.5,  # 50% - Cosine similarity to peers
            temporal_weight=0.3,    # 30% - Behavioral consistency
            magnitude_weight=0.2,   # 20% - Gradient norm distribution

            # Ensemble threshold (confidence needed to flag Byzantine)
            ensemble_threshold=0.6  # 60% confidence required
        )

        print(f"✓ Full 0TML Hybrid Detector initialized\n")
        print(self.detector)
        print()

        # Initialize gradient analyzer (for profile computation)
        self.gradient_analyzer = GradientDimensionalityAnalyzer()

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
            'decisions': []  # Store per-node decisions
        }

    def run_test(self, num_rounds: int = 3) -> Dict:
        """
        Run Full 0TML Hybrid Detector test.

        Args:
            num_rounds: Number of rounds to test (temporal detector needs multiple rounds)

        Returns:
            Dictionary with test results
        """
        print_header(f"Running {self.test_name}")
        print(f"Configuration:")
        print(f"  - Clients: {self.num_clients} ({self.num_honest} honest, {self.num_byzantine} Byzantine)")
        print(f"  - BFT Ratio: {self.bft_percent*100:.1f}%")
        print(f"  - Detector: Full 0TML Hybrid (ALL signals enabled)")
        print(f"  - Data Distribution: HETEROGENEOUS (unique per client)")
        print(f"  - Rounds: {num_rounds}")
        print(f"  - Seed: {self.seed}")
        print()

        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1}/{num_rounds} ---")

            # Generate gradients with HETEROGENEOUS DATA
            gradients_dict = {}
            ground_truth = {}

            # Honest gradients - each with UNIQUE local data
            for i in range(self.num_honest):
                client_seed = self.seed + i * 1000 + round_num * 100000
                client_data = create_synthetic_mnist_data(
                    num_samples=100,
                    seed=client_seed,
                    train=True
                )
                gradients_dict[i] = generate_honest_gradient(
                    self.global_model,
                    self.device,
                    train_loader=client_data
                )
                ground_truth[i] = False

            # Byzantine gradients - sign flip attack with unique local data
            for i in range(self.num_honest, self.num_clients):
                client_seed = self.seed + i * 1000 + round_num * 100000
                client_data = create_synthetic_mnist_data(
                    num_samples=100,
                    seed=client_seed,
                    train=True
                )

                # Generate honest gradient first
                honest_grad = generate_honest_gradient(
                    self.global_model,
                    self.device,
                    train_loader=client_data
                )

                # Apply sign flip attack
                attack = create_attack(AttackType.SIGN_FLIP, flip_intensity=1.0)
                flat_honest = np.concatenate([g.flatten() for g in honest_grad.values()])
                flat_byz = attack.generate(flat_honest, round_num=round_num)

                # Reshape back
                byz_grad = {}
                idx = 0
                for name, param in self.global_model.named_parameters():
                    size = param.numel()
                    byz_grad[name] = flat_byz[idx:idx+size].reshape(param.shape)
                    idx += size

                gradients_dict[i] = byz_grad
                ground_truth[i] = True

            # Convert gradients to tensor list for detector
            gradient_tensors = []
            node_ids = []

            for node_id in range(self.num_clients):
                # Flatten gradient dict to single tensor
                grad_dict = gradients_dict[node_id]
                flat_grad = np.concatenate([g.flatten() for g in grad_dict.values()])
                gradient_tensors.append(torch.from_numpy(flat_grad).float())
                node_ids.append(node_id)

            # Analyze gradient characteristics (for profile)
            print("  Analyzing gradient characteristics...")
            profile = self.gradient_analyzer.analyze_gradients(gradient_tensors)
            print(f"  Gradient Profile: {profile.detection_strategy} dimensionality")
            print(f"  Mean cosine similarity: {profile.mean_cosine_similarity:.3f}")

            # Run hybrid detector (batch analysis)
            print("  Running Full 0TML Hybrid Detector...")
            decisions = self.detector.batch_analyze(
                gradient_tensors,
                node_ids,
                profile
            )

            # Compute metrics for this round
            honest_flagged = sum(
                1 for i in range(self.num_honest)
                if decisions[i].is_byzantine
            )
            byzantine_detected = sum(
                1 for i in range(self.num_honest, self.num_clients)
                if decisions[i].is_byzantine
            )

            detection_rate = byzantine_detected / self.num_byzantine if self.num_byzantine > 0 else 0.0
            fpr = honest_flagged / self.num_honest if self.num_honest > 0 else 0.0

            print(f"  Byzantine Detection: {byzantine_detected}/{self.num_byzantine} ({detection_rate*100:.1f}%)")
            print(f"  Honest Flagged (FPR): {honest_flagged}/{self.num_honest} ({fpr*100:.1f}%)")

            # Show signal breakdown for a few nodes
            print(f"  Signal Breakdown (sample nodes):")
            for i in [0, self.num_honest-1, self.num_honest, self.num_clients-1]:  # First/last honest, first/last Byzantine
                decision = decisions[i]
                node_type = "Byzantine" if ground_truth[i] else "Honest"
                print(f"    Node {i} ({node_type}): "
                      f"Sim={decision.similarity_confidence:.2f}, "
                      f"Temp={decision.temporal_confidence:.2f}, "
                      f"Mag={decision.magnitude_confidence:.2f} → "
                      f"Ensemble={decision.ensemble_confidence:.2f} "
                      f"({'FLAGGED' if decision.is_byzantine else 'ACCEPTED'})")

            # Update cumulative results
            self.results['honest_flagged'] += honest_flagged
            self.results['byzantine_detected'] += byzantine_detected
            self.results['decisions'].append({
                'round': round_num + 1,
                'detection_rate': detection_rate,
                'fpr': fpr,
                'decisions': decisions
            })

        # Compute final metrics across all rounds
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

        # Print summary
        self.print_summary()

        return self.results

    def print_summary(self):
        """Print test summary."""
        print_header(f"RESULTS: {self.test_name}")

        print(f"Detection Performance:")
        print(f"  - Byzantine Detection Rate: {self.results['detection_rate']*100:.1f}% "
              f"({self.results['byzantine_detected']}/{self.results['total_byzantine'] * len(self.results['decisions'])})")
        print(f"  - False Positive Rate: {self.results['fpr']*100:.1f}% "
              f"({self.results['honest_flagged']}/{self.results['total_honest'] * len(self.results['decisions'])})")

        print(f"\nPer-Round Breakdown:")
        for round_data in self.results['decisions']:
            print(f"  Round {round_data['round']}: "
                  f"Detection={round_data['detection_rate']*100:.1f}%, "
                  f"FPR={round_data['fpr']*100:.1f}%")

        # Success criteria
        print_header("SUCCESS CRITERIA")

        detection_pass = self.results['detection_rate'] >= 0.95
        fpr_pass = self.results['fpr'] <= 0.10

        print(f"Expected at {self.bft_percent*100:.0f}% BFT (Heterogeneous Data):")
        print(f"  - Detection ≥95%: {'✅ PASS' if detection_pass else '❌ FAIL'} ({self.results['detection_rate']*100:.1f}%)")
        print(f"  - FPR ≤10%: {'✅ PASS' if fpr_pass else '❌ FAIL'} ({self.results['fpr']*100:.1f}%)")

        self.results['test_passed'] = detection_pass and fpr_pass

        print(f"\n{'='*80}")
        overall = "✅ PASSED" if self.results['test_passed'] else "❌ FAILED"
        print(f"Overall: {overall}")

        if self.results['test_passed']:
            print("\n🎉 SUCCESS! The Full 0TML Hybrid Detector successfully handles 35% BFT")
            print("   with heterogeneous data, proving our multi-signal architecture prevents")
            print("   detector inversion where simplified peer-comparison methods fail.")
        else:
            print("\n⚠️  FAILURE: The Full 0TML Hybrid Detector did not meet success criteria.")
            print("   This is a CRITICAL GAP in our paper's empirical validation.")

        print(f"{'='*80}\n")


def test_full_hybrid_35_bft():
    """
    Test Full 0TML Hybrid Detector at 35% BFT (peer-comparison boundary).

    This is the CRITICAL test that validates our actual solution works at the
    boundary where simplified peer-comparison methods fail catastrophically.
    """
    test = FullHybridDetectorTest(
        num_clients=20,
        num_byzantine=7,  # 35% BFT
        test_name="Full 0TML Hybrid Detector at 35% BFT",
        seed=42
    )

    # Run for 3 rounds (temporal detector needs history)
    results = test.run_test(num_rounds=3)

    # Assertions
    assert results['detection_rate'] >= 0.95, \
        f"Detection rate {results['detection_rate']*100:.1f}% < 95%"
    assert results['fpr'] <= 0.10, \
        f"FPR {results['fpr']*100:.1f}% > 10%"
    assert results['test_passed'], \
        "Test did not pass success criteria"

    return results


def test_full_hybrid_vs_simplified_comparison():
    """
    Comparison test: Show the difference between simplified Mode 0 vs Full 0TML Hybrid.

    This demonstrates WHY we needed the multi-signal architecture.
    """
    print_header("COMPARISON: Simplified Mode 0 vs Full 0TML Hybrid at 35% BFT")

    print("\nSimplified Mode 0 (Peer-Comparison Only) Results:")
    print("  - Detection Rate: 100.0% (7/7) ✅")
    print("  - False Positive Rate: 100.0% (13/13) ❌ CATASTROPHIC FAILURE")
    print("  - Problem: ALL honest nodes flagged as Byzantine!")
    print("  - Cause: Heterogeneous data makes honest clients look 'different'")

    print("\nFull 0TML Hybrid Detector Results:")
    results = test_full_hybrid_35_bft()
    print(f"  - Detection Rate: {results['detection_rate']*100:.1f}% ({results['byzantine_detected']}/{results['total_byzantine']*3}) ✅")
    print(f"  - False Positive Rate: {results['fpr']*100:.1f}% ({results['honest_flagged']}/{results['total_honest']*3}) ✅")
    print("  - Success: Multi-signal architecture prevents detector inversion!")
    print("  - Cause: Temporal + magnitude signals compensate for similarity variance")

    print_header("CONCLUSION")
    print("✅ The Full 0TML Hybrid Detector solves the detector inversion problem")
    print("   at 35% BFT where simplified peer-comparison methods fail.")
    print("\n📝 This empirical validation completes our paper's core claim:")
    print("   'Multi-signal ensemble detection enables Byzantine-robust FL beyond 33%'")
    print("="*80 + "\n")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 FULL 0TML HYBRID DETECTOR VALIDATION - CRITICAL PAPER GAP")
    print("="*80 + "\n")

    print("This test validates that our ACTUAL 0TML Hybrid Detector (with all signals)")
    print("successfully handles 35% BFT, proving our solution works where simplified")
    print("peer-comparison methods catastrophically fail (100% FPR).\n")

    # Run the critical validation
    print("\n### Test 1: Full 0TML Hybrid at 35% BFT ###")
    test_full_hybrid_35_bft()

    # Run comparison to highlight the difference
    print("\n### Test 2: Comparison Analysis ###")
    test_full_hybrid_vs_simplified_comparison()

    print("\n" + "="*80)
    print("✅ FULL 0TML HYBRID DETECTOR VALIDATION COMPLETE")
    print("="*80 + "\n")

    print("📄 NEXT STEP: Add these results to Section 5 of the paper")
    print("   (ZEROTRUSTML_PAPER_V1_SUBMISSION_READY.md)")
    print("\n   New section: '5.3 Full 0TML Hybrid Detector Validation'")
    print("   Key finding: Multi-signal architecture achieves <10% FPR at 35% BFT")
    print("   where simplified peer-comparison has 100% FPR.\n")
