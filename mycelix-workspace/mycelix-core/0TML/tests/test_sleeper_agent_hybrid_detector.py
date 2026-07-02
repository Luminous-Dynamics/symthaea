# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Test: Sleeper Agent Attack with Full 0TML Hybrid Detector

This test validates the Temporal Consistency Signal's ability to detect stateful
Byzantine attacks where nodes change behavior mid-training.

CRITICAL VALIDATION:
- Sleeper Agents behave HONESTLY during rounds 1-5 (build reputation)
- Sleeper Agents ACTIVATE and attack during rounds 6-10
- Temporal signal should detect behavioral change within 1-2 rounds

Attack Scenario:
- 20 clients (14 honest, 6 Sleeper Agents = 30% BFT when activated)
- Sleeper Agents: Honest rounds 1-5, Byzantine (sign flip) rounds 6+
- Challenge: Stateless detectors (similarity, magnitude) cannot catch this
- Solution: Temporal consistency detector tracks behavior over rolling window

Expected Outcome:
- Rounds 1-5: Low detection (<20%) - Sleepers building reputation
- Round 6: Detection increases (activation detected)
- Rounds 7-10: High detection (>80%) - Temporal signal catches behavioral change

This validates our most novel contribution: temporal consistency detection
for stateful Byzantine attacks that evade traditional detectors.

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
from src.byzantine_attacks import SleeperAgentAttack
from test_mode1_boundaries import (
    SimpleCNN,
    create_synthetic_mnist_data,
    generate_honest_gradient,
    pretrain_model
)


def print_header(title: str):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"🕵️  {title}")
    print("=" * 80)


class SleeperAgentTest:
    """Test harness for Sleeper Agent validation with Full 0TML Hybrid Detector."""

    def __init__(
        self,
        num_clients: int,
        num_sleeper_agents: int,
        activation_round: int,
        test_name: str,
        seed: int = 42,
        pretrain_epochs: int = 5
    ):
        self.num_clients = num_clients
        self.num_sleeper_agents = num_sleeper_agents
        self.num_honest = num_clients - num_sleeper_agents
        self.activation_round = activation_round
        self.test_name = test_name
        self.seed = seed

        # Set seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Initialize model
        self.device = "cpu"
        self.global_model = SimpleCNN().to(self.device)

        # Pre-train model
        print_header("Initializing Sleeper Agent Test Environment")
        print(f"Test: {test_name}")
        print(f"Configuration:")
        print(f"  - Total Clients: {num_clients}")
        print(f"  - Honest: {self.num_honest}")
        print(f"  - Sleeper Agents: {num_sleeper_agents}")
        print(f"  - Activation Round: {activation_round}")
        print(f"  - Attack Type: Sign Flip (after activation)")
        print()

        self.global_model = pretrain_model(
            self.global_model,
            num_epochs=pretrain_epochs,
            seed=seed,
            device=self.device
        )

        # Initialize Full 0TML Hybrid Detector
        print_header("Initializing Full 0TML Hybrid Detector (Temporal Signal Critical)")
        print("Configuration:")
        print("  - Similarity Signal: ENABLED (50% weight)")
        print("  - Temporal Signal: ENABLED (30% weight) ⭐ KEY FOR SLEEPER DETECTION")
        print("  - Magnitude Signal: ENABLED (20% weight)")
        print("  - Ensemble Voting: ENABLED (0.6 threshold)")
        print("  - Temporal Window: 5 rounds (tracks behavioral consistency)")
        print()

        self.detector = HybridByzantineDetector(
            # Temporal detector - CRITICAL for Sleeper Agents
            temporal_window_size=5,  # Rolling window of 5 rounds
            temporal_cosine_var_threshold=0.1,  # Variance threshold for cosine
            temporal_magnitude_var_threshold=0.5,  # Variance threshold for magnitude
            temporal_min_observations=2,  # Need at least 2 observations

            # Magnitude detector
            magnitude_z_score_threshold=3.0,
            magnitude_min_samples=3,

            # Ensemble weights
            similarity_weight=0.5,
            temporal_weight=0.3,  # Temporal signal will catch activation
            magnitude_weight=0.2,

            # Ensemble threshold
            ensemble_threshold=0.6
        )

        print(f"✓ Full 0TML Hybrid Detector initialized\n")

        # Initialize gradient analyzer
        self.gradient_analyzer = GradientDimensionalityAnalyzer()

        # Create Sleeper Agent attack instances
        self.sleeper_attacks = []
        for _ in range(num_sleeper_agents):
            attack = SleeperAgentAttack(
                activation_round=activation_round,
                byzantine_attack_type="sign_flip",  # Clear signal after activation
                honest_period_noise=0.01  # Small noise during honest phase
            )
            self.sleeper_attacks.append(attack)

        # Results storage
        self.results = {
            'num_clients': num_clients,
            'num_sleeper_agents': num_sleeper_agents,
            'num_honest': self.num_honest,
            'activation_round': activation_round,
            'per_round_results': [],
            'detection_before_activation': 0,
            'detection_after_activation': 0,
            'detection_at_activation': 0,
            'rounds_to_detect': None,
            'temporal_signal_effectiveness': 0.0
        }

    def run_test(self, num_rounds: int = 10) -> Dict:
        """
        Run Sleeper Agent test over multiple rounds.

        Args:
            num_rounds: Number of training rounds (must be > activation_round)

        Returns:
            Dictionary with test results
        """
        if num_rounds <= self.activation_round:
            raise ValueError(
                f"num_rounds ({num_rounds}) must be > activation_round ({self.activation_round})"
            )

        print_header(f"Running {self.test_name}")
        print(f"Configuration:")
        print(f"  - Total Clients: {self.num_clients}")
        print(f"  - Honest: {self.num_honest}")
        print(f"  - Sleeper Agents: {self.num_sleeper_agents}")
        print(f"  - Activation Round: {self.activation_round}")
        print(f"  - Total Rounds: {num_rounds}")
        print(f"  - Seed: {self.seed}")
        print()

        print("Expectations:")
        print(f"  Rounds 1-{self.activation_round}: Sleepers behave honestly (low detection)")
        print(f"  Round {self.activation_round+1}: Activation detected by temporal signal")
        print(f"  Rounds {self.activation_round+1}-{num_rounds}: High detection (>80%)")
        print()

        for round_num in range(1, num_rounds + 1):
            print(f"\n{'='*80}")
            print(f"Round {round_num}/{num_rounds}")
            print(f"{'='*80}")

            if round_num == self.activation_round:
                print("⚠️  WARNING: This is the activation round!")
                print("   Sleeper Agents will BEGIN attacking in the NEXT round.")
                print()
            elif round_num == self.activation_round + 1:
                print("🚨 CRITICAL: First round after activation!")
                print("   Temporal signal should detect sudden behavior change.")
                print()

            # Generate gradients
            gradients_dict = {}
            ground_truth = {}  # Track actual status (honest vs Byzantine this round)

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
                ground_truth[i] = False  # Always honest

            # Sleeper Agent gradients
            for i in range(self.num_honest, self.num_clients):
                sleeper_idx = i - self.num_honest
                attack = self.sleeper_attacks[sleeper_idx]

                client_seed = self.seed + i * 1000 + round_num * 100000
                client_data = create_synthetic_mnist_data(
                    num_samples=100,
                    seed=client_seed,
                    train=True
                )

                # Generate base honest gradient
                honest_grad = generate_honest_gradient(
                    self.global_model,
                    self.device,
                    train_loader=client_data
                )

                # Flatten
                flat_honest = np.concatenate([g.flatten() for g in honest_grad.values()])

                # Apply Sleeper Agent attack (may be honest or Byzantine depending on round)
                flat_sleeper = attack.generate(flat_honest, round_num)

                # Reshape back
                sleeper_grad = {}
                idx = 0
                for name, param in self.global_model.named_parameters():
                    size = param.numel()
                    sleeper_grad[name] = flat_sleeper[idx:idx+size].reshape(param.shape)
                    idx += size

                gradients_dict[i] = sleeper_grad

                # Track whether this Sleeper Agent is currently Byzantine
                metrics = attack.get_detection_metrics()
                ground_truth[i] = metrics['is_currently_byzantine']

            # Convert to tensors
            gradient_tensors = []
            node_ids = []

            for node_id in range(self.num_clients):
                grad_dict = gradients_dict[node_id]
                flat_grad = np.concatenate([g.flatten() for g in grad_dict.values()])
                gradient_tensors.append(torch.from_numpy(flat_grad).float())
                node_ids.append(node_id)

            # Analyze gradients
            profile = self.gradient_analyzer.analyze_gradients(gradient_tensors)

            # Run hybrid detector
            decisions = self.detector.batch_analyze(
                gradient_tensors,
                node_ids,
                profile
            )

            # Compute metrics
            # Current Byzantine nodes (Sleepers that are active)
            current_byzantine_nodes = [i for i in range(self.num_clients) if ground_truth[i]]
            num_current_byzantine = len(current_byzantine_nodes)

            # Detection metrics
            byzantine_detected = sum(
                1 for i in current_byzantine_nodes
                if decisions[i].is_byzantine
            )
            honest_flagged = sum(
                1 for i in range(self.num_honest)
                if decisions[i].is_byzantine
            )
            sleeper_detected = sum(
                1 for i in range(self.num_honest, self.num_clients)
                if decisions[i].is_byzantine
            )

            detection_rate = byzantine_detected / num_current_byzantine if num_current_byzantine > 0 else 0.0
            fpr = honest_flagged / self.num_honest if self.num_honest > 0 else 0.0

            print(f"\nStatus:")
            print(f"  - Active Sleeper Agents: {num_current_byzantine}/{self.num_sleeper_agents}")
            print(f"  - Detected: {sleeper_detected}/{self.num_sleeper_agents}")
            if num_current_byzantine > 0:
                print(f"  - Detection Rate: {detection_rate*100:.1f}% ({byzantine_detected}/{num_current_byzantine})")
            print(f"  - Honest Flagged (FPR): {honest_flagged}/{self.num_honest} ({fpr*100:.1f}%)")

            # Show temporal signal status for Sleeper Agents
            print(f"\nTemporal Signal Analysis (Sleeper Agents):")
            for i in range(self.num_honest, min(self.num_honest + 3, self.num_clients)):  # Show first 3
                decision = decisions[i]
                status = "ACTIVATED" if ground_truth[i] else "DORMANT"
                flagged = "FLAGGED" if decision.is_byzantine else "ACCEPTED"

                print(f"  Node {i} ({status}): "
                      f"Sim={decision.similarity_confidence:.2f}, "
                      f"Temp={decision.temporal_confidence:.2f} ⭐, "
                      f"Mag={decision.magnitude_confidence:.2f} → "
                      f"Ensemble={decision.ensemble_confidence:.2f} "
                      f"({flagged})")

                # Get temporal statistics if available
                temp_stats = self.detector.get_temporal_statistics(i)
                if temp_stats and temp_stats.num_observations >= 2:
                    print(f"         Temporal: "
                          f"Cosine Var={temp_stats.cosine_variance:.4f}, "
                          f"Mag Var={temp_stats.magnitude_variance:.4f}, "
                          f"Obs={temp_stats.num_observations}")

            # Store round results
            round_result = {
                'round': round_num,
                'detection_rate': detection_rate,
                'fpr': fpr,
                'byzantine_detected': byzantine_detected,
                'sleeper_detected': sleeper_detected,
                'num_current_byzantine': num_current_byzantine,
                'honest_flagged': honest_flagged,
                'decisions': decisions
            }
            self.results['per_round_results'].append(round_result)

            # Track detection phases
            if round_num < self.activation_round:
                # Pre-activation: Should have low detection
                pass
            elif round_num == self.activation_round:
                # Activation round: Sleepers still honest THIS round (activate NEXT round)
                pass
            else:
                # Post-activation: Should detect
                self.results['detection_after_activation'] += sleeper_detected

                # Track when we first detected majority
                if self.results['rounds_to_detect'] is None:
                    if detection_rate >= 0.5:  # Detected at least 50%
                        self.results['rounds_to_detect'] = round_num - self.activation_round

        # Compute final statistics
        self.compute_final_statistics()

        # Print summary
        self.print_summary()

        return self.results

    def compute_final_statistics(self):
        """Compute final test statistics."""
        # Average detection rate after activation
        post_activation_rounds = [
            r for r in self.results['per_round_results']
            if r['round'] > self.activation_round and r['num_current_byzantine'] > 0
        ]

        if post_activation_rounds:
            avg_detection_after = np.mean([r['detection_rate'] for r in post_activation_rounds])
            avg_fpr_after = np.mean([r['fpr'] for r in post_activation_rounds])

            self.results['avg_detection_after_activation'] = avg_detection_after
            self.results['avg_fpr_after_activation'] = avg_fpr_after
        else:
            self.results['avg_detection_after_activation'] = 0.0
            self.results['avg_fpr_after_activation'] = 0.0

        # Temporal signal effectiveness
        # (How much did detection improve after activation?)
        pre_activation_rounds = [
            r for r in self.results['per_round_results']
            if r['round'] < self.activation_round
        ]

        if pre_activation_rounds:
            avg_detection_before = np.mean([r['sleeper_detected'] / self.num_sleeper_agents for r in pre_activation_rounds])
            self.results['avg_detection_before_activation'] = avg_detection_before
        else:
            self.results['avg_detection_before_activation'] = 0.0

    def print_summary(self):
        """Print test summary."""
        print_header(f"RESULTS: {self.test_name}")

        print("Phase 1: Pre-Activation (Building Reputation)")
        print(f"  - Rounds 1-{self.activation_round}")
        print(f"  - Average Sleeper Detection: {self.results['avg_detection_before_activation']*100:.1f}%")
        print("  - Expected: Low detection (<20%)")
        if self.results['avg_detection_before_activation'] < 0.2:
            print("  - ✅ PASS: Sleepers successfully built reputation")
        else:
            print("  - ⚠️  WARNING: High pre-activation detection")

        print(f"\nPhase 2: Post-Activation (Temporal Signal Detection)")
        print(f"  - Rounds {self.activation_round+1}-{len(self.results['per_round_results'])}")
        print(f"  - Average Detection Rate: {self.results['avg_detection_after_activation']*100:.1f}%")
        print(f"  - Average FPR: {self.results['avg_fpr_after_activation']*100:.1f}%")
        print(f"  - Rounds to Detect (>50%): {self.results['rounds_to_detect']} rounds after activation")

        print_header("SUCCESS CRITERIA")

        # Check success criteria
        criteria_1 = self.results['avg_detection_before_activation'] < 0.2  # Low pre-activation
        criteria_2 = self.results['rounds_to_detect'] is not None and self.results['rounds_to_detect'] <= 2  # Fast detection
        criteria_3 = self.results['avg_detection_after_activation'] >= 0.8  # High final detection
        criteria_4 = self.results['avg_fpr_after_activation'] <= 0.15  # Low false positives

        print(f"1. Pre-activation detection <20%: {'✅ PASS' if criteria_1 else '❌ FAIL'} "
              f"({self.results['avg_detection_before_activation']*100:.1f}%)")
        print(f"2. Detection within 1-2 rounds: {'✅ PASS' if criteria_2 else '❌ FAIL'} "
              f"({self.results['rounds_to_detect']} rounds)")
        print(f"3. Final detection >80%: {'✅ PASS' if criteria_3 else '❌ FAIL'} "
              f"({self.results['avg_detection_after_activation']*100:.1f}%)")
        print(f"4. FPR <15%: {'✅ PASS' if criteria_4 else '❌ FAIL'} "
              f"({self.results['avg_fpr_after_activation']*100:.1f}%)")

        self.results['test_passed'] = criteria_1 and criteria_2 and criteria_3 and criteria_4

        print(f"\n{'='*80}")
        overall = "✅ PASSED" if self.results['test_passed'] else "❌ FAILED"
        print(f"Overall: {overall}")

        if self.results['test_passed']:
            print("\n🎉 SUCCESS! Temporal Consistency Signal successfully detected Sleeper Agent activation")
            print("   within 1-2 rounds, validating our stateful Byzantine attack defense.")
        else:
            print("\n⚠️  FAILURE: Temporal signal did not meet detection criteria.")
            print("   This is a CRITICAL GAP in validating our temporal consistency detector.")

        print(f"{'='*80}\n")


def test_sleeper_agent_hybrid_detector():
    """
    Test Sleeper Agent attack with Full 0TML Hybrid Detector.

    This validates the temporal consistency signal's ability to detect
    stateful Byzantine attacks that change behavior over time.
    """
    test = SleeperAgentTest(
        num_clients=20,
        num_sleeper_agents=6,  # 30% BFT when activated
        activation_round=5,     # Activate at round 5 (attack in round 6+)
        test_name="Sleeper Agent Detection with Full 0TML Hybrid Detector",
        seed=42
    )

    # Run for 10 rounds (5 honest + 5 Byzantine)
    results = test.run_test(num_rounds=10)

    # Assertions
    assert results['avg_detection_before_activation'] < 0.2, \
        f"Pre-activation detection too high: {results['avg_detection_before_activation']*100:.1f}%"
    assert results['rounds_to_detect'] is not None and results['rounds_to_detect'] <= 2, \
        f"Detection took too long: {results['rounds_to_detect']} rounds"
    assert results['avg_detection_after_activation'] >= 0.8, \
        f"Final detection too low: {results['avg_detection_after_activation']*100:.1f}%"
    assert results['test_passed'], \
        "Test did not pass success criteria"

    return results


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 SLEEPER AGENT VALIDATION - CRITICAL TEMPORAL SIGNAL TEST")
    print("="*80 + "\n")

    print("This test validates the Temporal Consistency Signal's ability to detect")
    print("stateful Byzantine attacks where nodes change behavior mid-training.\n")

    print("This is the ONLY test that empirically validates our temporal signal,")
    print("which is the most novel part of our Mode 0 detector.\n")

    # Run the critical validation
    print("\n### Sleeper Agent Test ###")
    test_sleeper_agent_hybrid_detector()

    print("\n" + "="*80)
    print("✅ SLEEPER AGENT VALIDATION COMPLETE")
    print("="*80 + "\n")

    print("📄 NEXT STEP: Add these results to Section 5 of the paper")
    print("   (ZEROTRUSTML_PAPER_V1_SUBMISSION_READY.md)")
    print("\n   New section: '5.4 Temporal Signal Validation (Sleeper Agent Attack)'")
    print("   Key finding: Temporal consistency detector catches stateful attacks")
    print("   within 1-2 rounds of activation.\n")
