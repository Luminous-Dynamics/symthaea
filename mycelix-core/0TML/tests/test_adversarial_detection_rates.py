#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Adversarial Detection Rate Testing - Finding the REAL Numbers

This test runs sophisticated attacks against PoGQ and reports
HONEST detection rates without tuning the detector.

Goal: Find out if we really have "100% detection" or if that's
only true for the attacks we designed ourselves.
"""

import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Tuple

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from baselines.pogq_real import analyze_gradient_quality
    from tests.adversarial_attacks.stealthy_attacks import StealthyAttacks
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Make sure you're running from the project root")
    sys.exit(1)


class AdversarialDetectionTester:
    """Test PoGQ against sophisticated attacks"""

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.results = []
        self.attacks = StealthyAttacks()

    def generate_honest_gradients(self, num_nodes: int = 5, gradient_size: int = 1000) -> List[np.ndarray]:
        """Generate simulated honest gradients"""
        # In real system, these would come from actual training
        # For testing, we simulate with reasonable statistical properties
        honest_gradients = []

        # Mean gradient direction (what helps the model)
        mean_direction = np.random.randn(gradient_size)
        mean_direction /= np.linalg.norm(mean_direction)

        for _ in range(num_nodes):
            # Each node's gradient is similar but with variance
            magnitude = np.random.uniform(0.8, 1.2)
            noise = np.random.normal(0, 0.1, gradient_size)
            gradient = magnitude * mean_direction + noise
            honest_gradients.append(gradient)

        return honest_gradients

    def test_attack(self,
                    attack_name: str,
                    attack_fn,
                    num_trials: int = 100,
                    **attack_kwargs) -> Dict:
        """
        Test single attack type and measure detection rate

        NO TUNING - We accept whatever detection rate we get
        """
        print(f"\n🎯 Testing: {attack_name}")
        print(f"   Trials: {num_trials}")

        detected_count = 0
        pogq_scores_detected = []
        pogq_scores_missed = []

        for trial in range(num_trials):
            # Generate fresh honest gradients
            honest_grads = self.generate_honest_gradients()

            # Generate attack gradient
            if 'honest_gradient' in attack_fn.__code__.co_varnames:
                attack_grad = attack_fn(honest_grads[0], **attack_kwargs)
            elif 'honest_gradients' in attack_fn.__code__.co_varnames:
                attack_grad = attack_fn(honest_grads, **attack_kwargs)
            else:
                attack_grad = attack_fn(**attack_kwargs)

            # Calculate PoGQ score for attack
            all_grads = honest_grads + [attack_grad]
            pogq_scores = [analyze_gradient_quality(g, all_grads) for g in all_grads]
            attack_pogq_score = pogq_scores[-1]

            # Check if detected
            is_detected = attack_pogq_score < self.threshold

            if is_detected:
                detected_count += 1
                pogq_scores_detected.append(attack_pogq_score)
            else:
                pogq_scores_missed.append(attack_pogq_score)

        # Calculate statistics
        detection_rate = detected_count / num_trials
        avg_score_detected = np.mean(pogq_scores_detected) if pogq_scores_detected else 0
        avg_score_missed = np.mean(pogq_scores_missed) if pogq_scores_missed else 0

        result = {
            "attack_name": attack_name,
            "num_trials": num_trials,
            "detected": detected_count,
            "missed": num_trials - detected_count,
            "detection_rate": detection_rate,
            "avg_pogq_when_detected": float(avg_score_detected),
            "avg_pogq_when_missed": float(avg_score_missed),
            "threshold": self.threshold
        }

        # Print results
        print(f"   ✅ Detected: {detected_count}/{num_trials} ({detection_rate:.1%})")
        print(f"   ❌ Missed: {num_trials - detected_count}/{num_trials} ({1-detection_rate:.1%})")
        if pogq_scores_detected:
            print(f"   📊 Avg PoGQ when detected: {avg_score_detected:.3f}")
        if pogq_scores_missed:
            print(f"   📊 Avg PoGQ when missed: {avg_score_missed:.3f}")

        self.results.append(result)
        return result

    def run_full_adversarial_test_suite(self) -> Dict:
        """Run all adversarial tests and report honest results"""

        print("=" * 70)
        print("🔬 ADVERSARIAL DETECTION RATE TESTING")
        print("=" * 70)
        print("\nTesting PoGQ against sophisticated attacks we didn't design")
        print("NO TUNING - Accepting whatever detection rate we get\n")
        print(f"PoGQ Threshold: {self.threshold}")
        print(f"Goal: Find REAL detection rate, not inflated claims")
        print("=" * 70)

        # Test 1: Noise-Masked Poisoning
        self.test_attack(
            "Noise-Masked Poisoning (Stealthy)",
            self.attacks.noise_masked_poisoning,
            num_trials=100,
            poison_intensity=0.3,
            noise_level=0.4
        )

        # Test 2: Slow Degradation (Early Rounds)
        self.test_attack(
            "Slow Degradation - Early Rounds (Building Rep)",
            self.attacks.slow_degradation,
            num_trials=100,
            round_num=20,  # Still building reputation
            reputation_building_rounds=30
        )

        # Test 3: Slow Degradation (Late Rounds)
        self.test_attack(
            "Slow Degradation - Late Rounds (Attack Phase)",
            self.attacks.slow_degradation,
            num_trials=100,
            round_num=40,  # Attack has escalated
            reputation_building_rounds=30
        )

        # Test 4: Targeted Neuron Attack
        self.test_attack(
            "Targeted Neuron Attack (Backdoor-like)",
            self.attacks.targeted_neuron_attack,
            num_trials=100,
            target_percentage=0.05  # Only 5% of neurons poisoned
        )

        # Test 5: Statistical Mimicry
        self.test_attack(
            "Statistical Mimicry (Match Honest Stats)",
            self.attacks.statistical_mimicry,
            num_trials=100,
            target_mean_shift=0.3
        )

        # Test 6: Adaptive Noise (Recently Detected)
        self.test_attack(
            "Adaptive Noise - High Detection Pressure",
            self.attacks.adaptive_noise_injection,
            num_trials=100,
            previous_detections=[True, True, False, True],  # 75% detection
            base_noise=0.2
        )

        # Test 7: Adaptive Noise (Not Recently Detected)
        self.test_attack(
            "Adaptive Noise - Low Detection Pressure",
            self.attacks.adaptive_noise_injection,
            num_trials=100,
            previous_detections=[False, False, False, True],  # 25% detection
            base_noise=0.2
        )

        # Calculate overall statistics
        total_trials = sum(r['num_trials'] for r in self.results)
        total_detected = sum(r['detected'] for r in self.results)
        overall_detection_rate = total_detected / total_trials

        # Calculate by sophistication level
        stealthy_attacks = [r for r in self.results if "Stealthy" in r['attack_name'] or "Mimicry" in r['attack_name'] or "Targeted" in r['attack_name']]
        adaptive_attacks = [r for r in self.results if "Adaptive" in r['attack_name']]
        reputation_attacks = [r for r in self.results if "Degradation" in r['attack_name']]

        summary = {
            "test_date": datetime.now().isoformat(),
            "pogq_threshold": self.threshold,
            "total_trials": total_trials,
            "total_detected": total_detected,
            "total_missed": total_trials - total_detected,
            "overall_detection_rate": overall_detection_rate,
            "detection_by_category": {
                "stealthy_attacks": np.mean([r['detection_rate'] for r in stealthy_attacks]) if stealthy_attacks else 0,
                "adaptive_attacks": np.mean([r['detection_rate'] for r in adaptive_attacks]) if adaptive_attacks else 0,
                "reputation_attacks": np.mean([r['detection_rate'] for r in reputation_attacks]) if reputation_attacks else 0,
            },
            "detailed_results": self.results
        }

        return summary

    def print_summary(self, summary: Dict):
        """Print comprehensive summary of adversarial testing"""

        print("\n" + "=" * 70)
        print("📊 ADVERSARIAL TESTING SUMMARY")
        print("=" * 70)

        print(f"\n🎯 Overall Detection Rate: {summary['overall_detection_rate']:.1%}")
        print(f"   Total Trials: {summary['total_trials']}")
        print(f"   Detected: {summary['total_detected']}")
        print(f"   Missed: {summary['total_missed']}")

        print("\n📈 Detection Rate by Attack Sophistication:")
        for category, rate in summary['detection_by_category'].items():
            category_name = category.replace('_', ' ').title()
            print(f"   {category_name}: {rate:.1%}")

        print("\n🔍 Attack-by-Attack Breakdown:")
        for result in summary['detailed_results']:
            status = "✅ STRONG" if result['detection_rate'] > 0.85 else "⚠️  WEAK" if result['detection_rate'] < 0.70 else "🟡 MODERATE"
            print(f"   {status} {result['attack_name']}: {result['detection_rate']:.1%}")

        print("\n💡 Honest Assessment:")
        overall_rate = summary['overall_detection_rate']

        if overall_rate > 0.9:
            assessment = "EXCELLENT - PoGQ is robust against sophisticated attacks"
        elif overall_rate > 0.75:
            assessment = "STRONG - PoGQ handles most attacks but has weaknesses"
        elif overall_rate > 0.60:
            assessment = "MODERATE - PoGQ provides some protection but needs improvement"
        else:
            assessment = "WEAK - PoGQ struggles against sophisticated attacks"

        print(f"   {assessment}")
        print(f"\n   Realistic claim: 'PoGQ achieves {overall_rate:.0%} detection rate'")
        print(f"   against sophisticated adversarial attacks.'")

        print("\n⚠️  Weaknesses Identified:")
        weak_attacks = [r for r in summary['detailed_results'] if r['detection_rate'] < 0.70]
        if weak_attacks:
            for attack in weak_attacks:
                print(f"   - {attack['attack_name']}: Only {attack['detection_rate']:.1%} detection")
        else:
            print("   - No major weaknesses found!")

        print("\n✅ Strengths Identified:")
        strong_attacks = [r for r in summary['detailed_results'] if r['detection_rate'] > 0.85]
        if strong_attacks:
            for attack in strong_attacks:
                print(f"   - {attack['attack_name']}: {attack['detection_rate']:.1%} detection")
        else:
            print("   - No attacks had >85% detection rate")

        print("\n" + "=" * 70)

    def save_results(self, summary: Dict, output_file: str = "adversarial_testing_results.json"):
        """Save detailed results to JSON"""
        output_path = Path(__file__).parent.parent / "results" / output_file

        # Ensure results directory exists
        output_path.parent.mkdir(exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Results saved to: {output_path}")


def main():
    """Run full adversarial testing suite"""

    print("\n🎯 ZERO-TRUSTML ADVERSARIAL DETECTION TESTING")
    print("Finding the REAL detection rate against sophisticated attacks\n")

    # Create tester
    tester = AdversarialDetectionTester(threshold=0.7)

    # Run full test suite
    summary = tester.run_full_adversarial_test_suite()

    # Print comprehensive summary
    tester.print_summary(summary)

    # Save results
    tester.save_results(summary)

    print("\n✅ Adversarial testing complete!")
    print(f"📊 Overall Detection Rate: {summary['overall_detection_rate']:.1%}")
    print("\n🎯 Use this number for honest grant claims, not '100%'")


if __name__ == "__main__":
    main()
