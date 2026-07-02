#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Integrated Byzantine Attack Testing with Sybil Detection
Week 3 Priority 1: Complete validation with all 8 attack types detected

Combines:
- Byzantine Attack Simulator (basic attacks)
- Sybil Detector (coordinated attacks)
- PoGQ scoring
- RB-BFT reputation
"""

import torch
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from byzantine_attack_simulator import ByzantineAttackSimulator
from sybil_detector import SybilDetector


class IntegratedByzantineDetector:
    """Integrated detector combining PoGQ + RB-BFT + Sybil detection"""

    def __init__(self, config_path: str = "byzantine-configs/attack-strategies.json"):
        self.attack_simulator = ByzantineAttackSimulator(config_path)
        self.sybil_detector = SybilDetector(
            similarity_threshold=0.98,
            cluster_size_threshold=2,
            entropy_threshold=0.5,
            temporal_window=3
        )

    def run_integrated_test(self, num_rounds: int = 5):
        """Run complete integrated test with all detection layers"""
        print("=" * 70)
        print("INTEGRATED BYZANTINE DETECTION TEST")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Total nodes: 20")
        print(f"  Honest nodes: 12 (60%)")
        print(f"  Byzantine nodes: 8 (40%)")
        print(f"  Test rounds: {num_rounds}")
        print(f"  Detection layers: PoGQ + RB-BFT + Sybil")

        # Track overall detection across all rounds
        total_detections = {
            "pogq_only": 0,
            "sybil_only": 0,
            "combined": 0,
            "false_positives": 0
        }

        for round_num in range(1, num_rounds + 1):
            print(f"\n{'=' * 70}")
            print(f"ROUND {round_num}/{num_rounds}")
            print(f"{'=' * 70}")

            # Generate honest gradient baseline
            honest_gradient = torch.randn(100) * 0.01

            # Generate gradients for all nodes (including Byzantine attacks)
            gradients = {}
            sybil_gradients_cache = {}  # For Sybil coordination

            for node_id in range(20):
                # Get Sybil gradients if this is a Sybil node
                sybil_grads = []
                if self.attack_simulator.get_attack_strategy(node_id) == "sybil_coordination":
                    # Get gradients from other Sybil nodes in same pair
                    for other_id in range(20):
                        if (self.attack_simulator.get_attack_strategy(other_id) == "sybil_coordination" and
                            other_id != node_id and other_id in sybil_gradients_cache):
                            sybil_grads.append(sybil_gradients_cache[other_id])

                # Generate gradient (honest or Byzantine)
                gradient = self.attack_simulator.generate_gradient(
                    node_id, honest_gradient, sybil_grads
                )
                gradients[node_id] = gradient

                # Cache for Sybil coordination
                if self.attack_simulator.get_attack_strategy(node_id) == "sybil_coordination":
                    sybil_gradients_cache[node_id] = gradient

            # LAYER 1: PoGQ + Basic Validation
            print(f"\n📊 Layer 1: PoGQ + Basic Validation")
            honest_nodes = self.attack_simulator.honest_nodes
            honest_grads = [gradients[i] for i in honest_nodes]

            pogq_detections = set()
            for node_id, gradient in gradients.items():
                # PoGQ scoring
                is_valid, pogq_score = self.attack_simulator.validate_statistical_outlier(
                    gradient, honest_grads
                )

                if not is_valid:
                    pogq_detections.add(node_id)
                    is_byz = "✓" if self.attack_simulator.is_byzantine(node_id) else "✗ FP"
                    strategy = self.attack_simulator.get_attack_strategy(node_id)
                    print(f"  Node {node_id:2d}: PoGQ={pogq_score:.3f} → DETECTED {is_byz} ({strategy})")

            # LAYER 2: Sybil Coordination Detection
            print(f"\n🔍 Layer 2: Sybil Coordination Detection")
            sybil_clusters, sybil_details = self.sybil_detector.detect_sybil_coordination(gradients)

            sybil_detections = set()
            for cluster in sybil_clusters:
                sybil_detections.update(cluster)
                print(f"  Sybil Cluster: {sorted(cluster)}")

            # LAYER 3: Combined Detection + Reputation Update
            print(f"\n🛡️  Layer 3: Combined Detection + RB-BFT Reputation")

            detected_nodes = pogq_detections | sybil_detections

            for node_id in range(20):
                was_detected = node_id in detected_nodes
                is_byzantine = self.attack_simulator.is_byzantine(node_id)
                strategy = self.attack_simulator.get_attack_strategy(node_id) or "honest"

                # Update reputation
                self.attack_simulator.update_reputation(node_id, was_detected)
                self.attack_simulator.record_detection(node_id, was_detected)

                # Get Sybil score
                sybil_score = self.sybil_detector.get_sybil_score(node_id, gradients[node_id], gradients)

                # Classification
                if was_detected:
                    if is_byzantine:
                        status = "✅ TRUE POSITIVE"
                        total_detections["combined"] += 1
                        if node_id in pogq_detections and node_id not in sybil_detections:
                            total_detections["pogq_only"] += 1
                        elif node_id in sybil_detections and node_id not in pogq_detections:
                            total_detections["sybil_only"] += 1
                    else:
                        status = "❌ FALSE POSITIVE"
                        total_detections["false_positives"] += 1
                else:
                    if is_byzantine:
                        status = "⚠️  FALSE NEGATIVE"
                    else:
                        status = "✅ TRUE NEGATIVE"

                rep = self.attack_simulator.reputations[node_id]
                detection_method = ""
                if node_id in pogq_detections:
                    detection_method += "PoGQ"
                if node_id in sybil_detections:
                    detection_method += "+Sybil" if detection_method else "Sybil"
                if not detection_method:
                    detection_method = "None"

                print(f"  Node {node_id:2d} ({strategy:18s}): {status:20s} | Rep={rep:.2f} | Sybil={sybil_score:.3f} | {detection_method}")

            # Round summary
            print(f"\n📈 Round {round_num} Summary:")
            print(f"  PoGQ Detections: {len(pogq_detections)}")
            print(f"  Sybil Detections: {len(sybil_detections)}")
            print(f"  Combined Detections: {len(detected_nodes)}")

        # Final metrics
        print(f"\n{'=' * 70}")
        print(f"FINAL METRICS (All Rounds)")
        print(f"{'=' * 70}")

        metrics = self.attack_simulator.calculate_metrics()
        print(f"\nDetection Performance:")
        print(f"  Detection Rate:       {metrics['detection_rate']:.1%}")
        print(f"  False Positive Rate:  {metrics['false_positive_rate']:.1%}")
        print(f"  Accuracy:            {metrics['accuracy']:.1%}")
        print(f"  Precision:           {metrics['precision']:.1%}")
        print(f"  Recall:              {metrics['recall']:.1%}")

        print(f"\nDetection Breakdown:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")

        print(f"\nDetection Method Attribution:")
        print(f"  PoGQ Only:    {total_detections['pogq_only']}")
        print(f"  Sybil Only:   {total_detections['sybil_only']}")
        print(f"  Combined:     {total_detections['combined']}")

        # Success check
        print(f"\n{'=' * 70}")
        if metrics['detection_rate'] >= 0.95 and metrics['false_positive_rate'] < 0.05:
            print(f"🎉 SUCCESS: Achieved ≥95% detection with <5% false positives!")
        else:
            print(f"⚠️  Goal: Achieve ≥95% detection with <5% false positives")
            print(f"   Current: {metrics['detection_rate']:.1%} detection, {metrics['false_positive_rate']:.1%} FP")
        print(f"{'=' * 70}")

        return metrics


def main():
    """Run integrated Byzantine detection test"""
    parser = argparse.ArgumentParser(description='Integrated Byzantine Detection Test')
    parser.add_argument('--config', type=str,
                       default='byzantine-configs/attack-strategies.json',
                       help='Path to attack configuration JSON file')
    parser.add_argument('--output', type=str,
                       help='Path to save JSON results')
    parser.add_argument('--rounds', type=int, default=5,
                       help='Number of test rounds to run')

    args = parser.parse_args()

    # Create detector with specified config
    detector = IntegratedByzantineDetector(args.config)
    metrics = detector.run_integrated_test(num_rounds=args.rounds)

    # Save results to JSON if output path specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✅ Results saved to {args.output}")

    # Return exit code based on success
    if metrics['detection_rate'] >= 0.95 and metrics['false_positive_rate'] < 0.05:
        return 0  # Success
    else:
        return 1  # Not yet meeting goals


if __name__ == "__main__":
    exit(main())
