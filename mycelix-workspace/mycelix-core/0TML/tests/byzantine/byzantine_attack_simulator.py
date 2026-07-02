#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Byzantine Attack Simulator for Holochain DHT Testing
Week 3 - Testing 8 validation rules under 40% Byzantine attack

Implements 4 attack strategies:
1. Label Flipping
2. Gradient Reversal
3. Random Noise Injection
4. Sybil Coordination
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path


class ByzantineAttackSimulator:
    """Simulates various Byzantine attacks on federated learning gradients"""

    def __init__(self, config_path: str = "byzantine-configs/attack-strategies.json"):
        """Initialize Byzantine attack simulator"""
        self.config_path = Path(config_path)
        self.load_config()

        # Initialize node reputations (all start at 1.0)
        self.reputations = {i: 1.0 for i in range(20)}

        # Track detection history
        self.detection_history = {
            "true_positives": [],
            "false_positives": [],
            "true_negatives": [],
            "false_negatives": []
        }

    def load_config(self):
        """Load Byzantine configuration from JSON"""
        with open(self.config_path) as f:
            self.config = json.load(f)

        self.honest_nodes = self.config["node_assignments"]["honest"]
        self.byzantine_nodes = self.config["node_assignments"]["byzantine"]
        self.strategies = self.config["attack_strategies"]
        threshold_cfg = (
            self.config
            .get("validation_rules", {})
            .get("3_statistical_outlier", {})
            .get("threshold", 0.5)
        )
        try:
            self.pogq_threshold = float(threshold_cfg)
        except (TypeError, ValueError):
            self.pogq_threshold = 0.5

    def is_byzantine(self, node_id: int) -> bool:
        """Check if node is Byzantine"""
        return node_id in self.byzantine_nodes

    def get_attack_strategy(self, node_id: int) -> str:
        """Get attack strategy for Byzantine node"""
        for strategy_name, strategy_config in self.strategies.items():
            if node_id in strategy_config["nodes"]:
                return strategy_name
        return None

    def apply_label_flipping(self, labels: torch.Tensor) -> torch.Tensor:
        """Attack Strategy 1: Flip CIFAR-10 labels"""
        # CIFAR-10 has 10 classes (0-9)
        # Flip: 0↔9, 1↔8, 2↔7, 3↔6, 4↔5
        return 9 - labels

    def apply_gradient_reversal(self, gradient: torch.Tensor) -> torch.Tensor:
        """Attack Strategy 2: Reverse gradient direction"""
        return -1.0 * gradient

    def apply_random_noise(self, gradient: torch.Tensor) -> torch.Tensor:
        """Attack Strategy 3: Inject high-variance Gaussian noise"""
        noise = torch.randn_like(gradient) * 100
        return noise

    def apply_sybil_coordination(self, gradient: torch.Tensor,
                                 sybil_gradients: List[torch.Tensor]) -> torch.Tensor:
        """Attack Strategy 4: Coordinate with other Sybil nodes

        Creates gradients that are:
        1. Highly similar to each other (coordinated)
        2. Subtly different from honest gradients (to avoid detection)
        3. Still plausible enough to pass basic checks
        """
        if len(sybil_gradients) > 0:
            # Average with other Sybil nodes for coordination
            coordinated = torch.stack([gradient] + sybil_gradients).mean(dim=0)
            # Add consistent bias to differentiate from honest
            # (makes them coordinated with each other but distinct from honest)
            coordinated *= 1.05  # Slight magnitude increase
            # Add tiny noise to avoid exact duplication
            coordinated += torch.randn_like(coordinated) * 0.0001
            return coordinated
        else:
            # First Sybil node in pair: apply slight modification
            modified = gradient * 1.05
            modified += torch.randn_like(modified) * 0.0001
            return modified

    def generate_gradient(self, node_id: int, honest_gradient: torch.Tensor,
                         sybil_gradients: List[torch.Tensor] = None) -> torch.Tensor:
        """Generate gradient for node (honest or Byzantine)"""
        if not self.is_byzantine(node_id):
            # Honest node: return clean gradient
            return honest_gradient.clone()

        # Byzantine node: apply attack strategy
        strategy = self.get_attack_strategy(node_id)

        if strategy == "label_flipping":
            # For label flipping, we would need to retrain with flipped labels
            # For simulation, reverse gradient as proxy
            return self.apply_gradient_reversal(honest_gradient)

        elif strategy == "gradient_reversal":
            return self.apply_gradient_reversal(honest_gradient)

        elif strategy == "random_noise":
            return self.apply_random_noise(honest_gradient)

        elif strategy == "sybil_coordination":
            if sybil_gradients is None:
                sybil_gradients = []
            return self.apply_sybil_coordination(honest_gradient, sybil_gradients)

        else:
            # Default: no attack
            return honest_gradient.clone()

    def validate_dimension(self, gradient: torch.Tensor,
                          expected_shape: torch.Size) -> bool:
        """Validation Rule 1: Vector dimension validation"""
        return gradient.shape == expected_shape

    def validate_magnitude(self, gradient: torch.Tensor,
                          honest_gradients: List[torch.Tensor],
                          threshold: float = 3.0) -> bool:
        """Validation Rule 2: Magnitude bounds checking"""
        # Calculate mean and std of honest gradients
        stacked = torch.stack(honest_gradients)
        mean = stacked.mean()
        std = stacked.std()

        # Check if gradient is within threshold * std from mean
        grad_mean = gradient.mean()
        return abs(grad_mean - mean) < threshold * std

    def validate_statistical_outlier(self, gradient: torch.Tensor,
                                    honest_gradients: List[torch.Tensor]) -> Tuple[bool, float]:
        """Validation Rule 3: Statistical outlier detection (PoGQ)"""
        # Calculate PoGQ score: cosine similarity with median honest gradient
        if len(honest_gradients) == 0:
            return True, 1.0

        median_gradient = torch.stack(honest_gradients).median(dim=0)[0]

        # Flatten for cosine similarity
        grad_flat = gradient.flatten()
        median_flat = median_gradient.flatten()

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            grad_flat.unsqueeze(0),
            median_flat.unsqueeze(0)
        ).item()

        threshold = getattr(self, "pogq_threshold", 0.5)
        # Clamp to sensible range to avoid misconfiguration
        threshold = max(min(threshold, 1.0), -1.0)
        is_valid = cos_sim > threshold
        return is_valid, cos_sim

    def update_reputation(self, node_id: int, was_detected: bool):
        """Validation Rule 5: Reputation scoring (RB-BFT)"""
        decay_rate = self.config["validation_rules"]["5_reputation_scoring"]["decay_rate"]

        if was_detected:
            # Byzantine detected: decrease reputation
            self.reputations[node_id] = max(0.0, self.reputations[node_id] - decay_rate)
        else:
            # Not detected: increase reputation slightly
            self.reputations[node_id] = min(1.0, self.reputations[node_id] + 0.05)

    def check_reputation(self, node_id: int) -> bool:
        """Check if node reputation is above threshold"""
        threshold = self.config["validation_rules"]["5_reputation_scoring"]["threshold"]
        return self.reputations[node_id] >= threshold

    def validate_cross_validation(self, gradient: torch.Tensor,
                                  all_gradients: List[torch.Tensor]) -> bool:
        """Validation Rule 6: Cross-validation consensus"""
        if len(all_gradients) < 3:
            return True

        # Calculate median of all gradients
        stacked = torch.stack(all_gradients)
        median = stacked.median(dim=0)[0]

        # Check if gradient is within 2σ of median
        std = stacked.std(dim=0).mean()
        distance = (gradient - median).abs().mean()

        return distance < 2 * std

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate detection metrics"""
        tp = len(self.detection_history["true_positives"])
        fp = len(self.detection_history["false_positives"])
        tn = len(self.detection_history["true_negatives"])
        fn = len(self.detection_history["false_negatives"])

        total = tp + fp + tn + fn
        if total == 0:
            return {
                "detection_rate": 0.0,
                "false_positive_rate": 0.0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }

        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = detection_rate

        return {
            "detection_rate": detection_rate,
            "false_positive_rate": false_positive_rate,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        }

    def record_detection(self, node_id: int, was_detected: bool):
        """Record detection result for metrics"""
        is_byzantine = self.is_byzantine(node_id)

        if is_byzantine and was_detected:
            self.detection_history["true_positives"].append(node_id)
        elif is_byzantine and not was_detected:
            self.detection_history["false_negatives"].append(node_id)
        elif not is_byzantine and was_detected:
            self.detection_history["false_positives"].append(node_id)
        else:  # not byzantine and not detected
            self.detection_history["true_negatives"].append(node_id)


def main():
    """Demo of Byzantine attack simulator"""
    print("Byzantine Attack Simulator - Demo")
    print("=" * 70)

    # Initialize simulator
    simulator = ByzantineAttackSimulator()

    print(f"\nConfiguration:")
    print(f"  Total nodes: 20")
    print(f"  Honest nodes: {len(simulator.honest_nodes)} (60%)")
    print(f"  Byzantine nodes: {len(simulator.byzantine_nodes)} (40%)")
    print(f"  Exceeds classical BFT limit: {simulator.config['byzantine_configuration']['exceeds_classical_bft']}")

    print(f"\nAttack Strategies:")
    for strategy_name, strategy_config in simulator.strategies.items():
        print(f"  {strategy_name}: Nodes {strategy_config['nodes']}")
        print(f"    Severity: {strategy_config['severity']}, Detectability: {strategy_config['detectability']}")

    print(f"\nSimulating gradient generation...")

    # Create dummy honest gradient
    honest_gradient = torch.randn(100) * 0.01

    # Generate gradients for all nodes
    gradients = {}
    for node_id in range(20):
        gradient = simulator.generate_gradient(node_id, honest_gradient)
        gradients[node_id] = gradient

        is_byz = "Byzantine" if simulator.is_byzantine(node_id) else "Honest"
        strategy = simulator.get_attack_strategy(node_id) if simulator.is_byzantine(node_id) else "N/A"
        print(f"  Node {node_id:2d} ({is_byz:9s}): {gradient.mean().item():+.6f} ± {gradient.std().item():.6f} | Strategy: {strategy}")

    print(f"\nValidation results:")
    honest_grads = [gradients[i] for i in simulator.honest_nodes]

    for node_id in range(20):
        gradient = gradients[node_id]

        # Apply validation rules
        dim_ok = simulator.validate_dimension(gradient, honest_gradient.shape)
        mag_ok = simulator.validate_magnitude(gradient, honest_grads)
        outlier_ok, pogq_score = simulator.validate_statistical_outlier(gradient, honest_grads)

        was_detected = not (dim_ok and mag_ok and outlier_ok)
        simulator.record_detection(node_id, was_detected)
        simulator.update_reputation(node_id, was_detected)

        status = "❌ BYZANTINE" if was_detected else "✅ HONEST"
        print(f"  Node {node_id:2d}: {status} | PoGQ={pogq_score:.3f} | Rep={simulator.reputations[node_id]:.2f}")

    print(f"\nMetrics:")
    metrics = simulator.calculate_metrics()
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.1%}")
        else:
            print(f"  {metric_name}: {value}")


if __name__ == "__main__":
    main()
