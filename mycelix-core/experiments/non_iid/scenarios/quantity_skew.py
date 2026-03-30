#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Quantity Skew Non-IID Scenario

Tests Byzantine detection under extreme data quantity imbalance across nodes.
This simulates real-world scenarios where:

- Large institutions have millions of samples
- Medium organizations have thousands
- Small participants have hundreds or fewer

Configuration creates 100x imbalance:
- Large nodes: 10,000 samples (2 nodes)
- Medium nodes: 1,000 samples (8 nodes)
- Small nodes: 100 samples (10 nodes)

Key questions:
1. Does Byzantine detection bias toward large nodes?
2. Are small nodes unfairly penalized?
3. How does gradient aggregation handle imbalance?

Author: Luminous Dynamics
Date: January 2026
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "0TML" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

logger = logging.getLogger(__name__)


@dataclass
class QuantitySkewConfig:
    """Configuration for quantity skew experiments."""
    # Network settings
    num_nodes: int = 20
    byzantine_ratio: float = 0.30

    # Training settings
    num_rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    warmup_rounds: int = 5

    # Dataset
    dataset: str = "cifar10"
    data_dir: str = "./data"

    # Quantity distribution
    large_nodes: int = 2
    large_samples: int = 10000
    medium_nodes: int = 8
    medium_samples: int = 1000
    small_nodes: int = 10
    small_samples: int = 100

    # Byzantine attack configuration
    attack_types: List[str] = field(
        default_factory=lambda: ["gradient_scaling", "sign_flip", "gaussian_noise"]
    )

    # Where to place Byzantine nodes: large, medium, small, or random
    byzantine_placement: str = "random"

    # Reproducibility
    seed: int = 42

    @property
    def max_imbalance_ratio(self) -> float:
        """Calculate maximum data imbalance ratio."""
        return self.large_samples / self.small_samples

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "byzantine_ratio": self.byzantine_ratio,
            "num_rounds": self.num_rounds,
            "large_nodes": self.large_nodes,
            "large_samples": self.large_samples,
            "medium_nodes": self.medium_nodes,
            "medium_samples": self.medium_samples,
            "small_nodes": self.small_nodes,
            "small_samples": self.small_samples,
            "max_imbalance_ratio": self.max_imbalance_ratio,
            "byzantine_placement": self.byzantine_placement,
            "dataset": self.dataset,
            "attack_types": self.attack_types,
            "seed": self.seed,
        }


class QuantitySkewScenario:
    """
    Quantity skew experiment scenario.

    Creates a federated learning setup with extreme data quantity
    imbalance (100x between largest and smallest nodes), then
    measures Byzantine detection accuracy and fairness.

    Example:
        >>> config = QuantitySkewConfig(num_nodes=20)
        >>> scenario = QuantitySkewScenario(config)
        >>> results = scenario.run()
        >>> print(f"Detection: {results['detection_accuracy']:.2%}")
        >>> print(f"Small node FP rate: {results['fp_by_size']['small']:.2%}")
    """

    def __init__(self, config: Optional[QuantitySkewConfig] = None):
        """Initialize scenario with configuration."""
        self.config = config or QuantitySkewConfig()
        self.rng = np.random.RandomState(self.config.seed)

        # Validate config
        total = self.config.large_nodes + self.config.medium_nodes + self.config.small_nodes
        if total != self.config.num_nodes:
            logger.warning(
                f"Node count mismatch: {total} vs {self.config.num_nodes}, adjusting"
            )
            self.config.num_nodes = total

        # Import dependencies
        self._setup_imports()

        # Node size assignments
        self.node_sizes: Dict[str, str] = {}  # node_id -> size category
        self.node_samples: Dict[str, int] = {}  # node_id -> sample count
        self.byzantine_ids: Set[str] = set()

        logger.info(
            f"QuantitySkewScenario initialized "
            f"({self.config.max_imbalance_ratio:.0f}x imbalance)"
        )

    def _setup_imports(self):
        """Setup and validate imports."""
        try:
            from mycelix_fl import MycelixFL, FLConfig
            self.MycelixFL = MycelixFL
            self.FLConfig = FLConfig
            self.fl_available = True
        except ImportError as e:
            logger.warning(f"mycelix_fl not available: {e}")
            self.fl_available = False

    def assign_node_sizes(self) -> None:
        """Assign data quantity to each node."""
        node_ids = [f"node_{i}" for i in range(self.config.num_nodes)]
        self.rng.shuffle(node_ids)

        idx = 0
        # Large nodes
        for _ in range(self.config.large_nodes):
            self.node_sizes[node_ids[idx]] = "large"
            self.node_samples[node_ids[idx]] = self.config.large_samples
            idx += 1

        # Medium nodes
        for _ in range(self.config.medium_nodes):
            self.node_sizes[node_ids[idx]] = "medium"
            self.node_samples[node_ids[idx]] = self.config.medium_samples
            idx += 1

        # Small nodes
        for _ in range(self.config.small_nodes):
            self.node_sizes[node_ids[idx]] = "small"
            self.node_samples[node_ids[idx]] = self.config.small_samples
            idx += 1

        logger.info(
            f"Assigned sizes: large={self.config.large_nodes}, "
            f"medium={self.config.medium_nodes}, small={self.config.small_nodes}"
        )

    def setup_byzantine(self) -> Set[str]:
        """Setup Byzantine nodes based on placement strategy."""
        num_byzantine = int(self.config.num_nodes * self.config.byzantine_ratio)
        node_ids = list(self.node_sizes.keys())

        if self.config.byzantine_placement == "random":
            self.byzantine_ids = set(
                self.rng.choice(node_ids, num_byzantine, replace=False)
            )
        elif self.config.byzantine_placement == "large":
            large_nodes = [n for n, s in self.node_sizes.items() if s == "large"]
            self.byzantine_ids = set(large_nodes[:num_byzantine])
        elif self.config.byzantine_placement == "small":
            small_nodes = [n for n, s in self.node_sizes.items() if s == "small"]
            self.byzantine_ids = set(small_nodes[:num_byzantine])
        else:
            # Default to random
            self.byzantine_ids = set(
                self.rng.choice(node_ids, num_byzantine, replace=False)
            )

        # Log Byzantine distribution by size
        byz_by_size = {"large": 0, "medium": 0, "small": 0}
        for node_id in self.byzantine_ids:
            size = self.node_sizes.get(node_id, "unknown")
            if size in byz_by_size:
                byz_by_size[size] += 1

        logger.info(f"Byzantine nodes: {sorted(self.byzantine_ids)}")
        logger.info(f"Byzantine by size: {byz_by_size}")

        return self.byzantine_ids

    def generate_gradients(self, round_num: int) -> Dict[str, np.ndarray]:
        """
        Generate gradients reflecting quantity skew.

        Nodes with more data produce more confident (lower variance) gradients.
        Nodes with less data produce noisier gradients.
        """
        gradient_size = 10000
        gradients = {}

        # Base gradient (population gradient)
        base_gradient = self.rng.randn(gradient_size).astype(np.float32) * 0.1

        for node_id in self.node_sizes.keys():
            num_samples = self.node_samples[node_id]

            # Gradient variance inversely proportional to sqrt(samples)
            # This follows statistical theory: more samples = lower variance
            variance_scale = 1.0 / np.sqrt(num_samples / 100)  # Normalize to small nodes

            # Generate gradient with appropriate noise
            noise = self.rng.randn(gradient_size).astype(np.float32) * 0.1 * variance_scale
            gradient = base_gradient + noise

            gradients[node_id] = gradient

        return gradients

    def inject_byzantine_attacks(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Inject Byzantine attacks into gradients."""
        modified = dict(gradients)

        for node_id in self.byzantine_ids:
            if node_id not in gradients:
                continue

            grad = gradients[node_id]
            attack_type = self.rng.choice(self.config.attack_types)

            if attack_type == "gradient_scaling":
                scale = self.rng.uniform(5, 100)
                modified[node_id] = grad * scale
            elif attack_type == "sign_flip":
                modified[node_id] = -grad
            elif attack_type == "gaussian_noise":
                noise = self.rng.randn(*grad.shape).astype(np.float32) * 10
                modified[node_id] = grad + noise
            else:
                modified[node_id] = self.rng.randn(*grad.shape).astype(np.float32)

        return modified

    def calculate_detection_metrics(
        self,
        detected: Set[str],
        ground_truth: Set[str],
        num_nodes: int,
    ) -> Dict[str, float]:
        """Calculate detection metrics."""
        tp = len(detected & ground_truth)
        fp = len(detected - ground_truth)
        fn = len(ground_truth - detected)
        tn = num_nodes - tp - fp - fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / num_nodes if num_nodes > 0 else 0.0

        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "detection_rate": recall,
            "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        }

    def _analyze_by_size(
        self,
        detected: Set[str],
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze detection results by node size."""
        size_analysis = {
            "large": {"total": 0, "byzantine": 0, "detected": 0, "fp": 0},
            "medium": {"total": 0, "byzantine": 0, "detected": 0, "fp": 0},
            "small": {"total": 0, "byzantine": 0, "detected": 0, "fp": 0},
        }

        for node_id, size in self.node_sizes.items():
            if size not in size_analysis:
                continue

            size_analysis[size]["total"] += 1

            is_byzantine = node_id in self.byzantine_ids
            is_detected = node_id in detected

            if is_byzantine:
                size_analysis[size]["byzantine"] += 1
                if is_detected:
                    size_analysis[size]["detected"] += 1
            else:
                if is_detected:
                    size_analysis[size]["fp"] += 1

        # Calculate rates
        for size, stats in size_analysis.items():
            honest_count = stats["total"] - stats["byzantine"]
            stats["fp_rate"] = stats["fp"] / honest_count if honest_count > 0 else 0.0
            stats["detection_rate"] = (
                stats["detected"] / stats["byzantine"]
                if stats["byzantine"] > 0 else 1.0
            )

        return size_analysis

    def _calculate_fairness_metrics(
        self,
        size_analysis: Dict[str, Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate fairness metrics across node sizes."""
        # Gini coefficient of false positive rates
        fp_rates = [
            size_analysis["large"]["fp_rate"],
            size_analysis["medium"]["fp_rate"],
            size_analysis["small"]["fp_rate"],
        ]

        # Simple Gini calculation
        fp_rates_sorted = sorted(fp_rates)
        n = len(fp_rates_sorted)
        if sum(fp_rates_sorted) == 0:
            gini = 0.0
        else:
            numerator = sum((i + 1) * fp for i, fp in enumerate(fp_rates_sorted))
            gini = (2 * numerator) / (n * sum(fp_rates_sorted)) - (n + 1) / n

        # Max difference in FP rate
        max_fp_diff = max(fp_rates) - min(fp_rates)

        # Detection rate parity
        det_rates = [
            size_analysis["large"]["detection_rate"],
            size_analysis["medium"]["detection_rate"],
            size_analysis["small"]["detection_rate"],
        ]
        det_rate_parity = max(det_rates) - min(det_rates)

        return {
            "fp_rate_gini": abs(gini),
            "max_fp_rate_difference": max_fp_diff,
            "detection_rate_parity": det_rate_parity,
            "small_node_fp_rate": size_analysis["small"]["fp_rate"],
            "large_node_fp_rate": size_analysis["large"]["fp_rate"],
        }

    def run(self) -> Dict[str, Any]:
        """
        Run the complete quantity skew experiment.

        Returns:
            Dictionary with all experiment results.
        """
        logger.info("=" * 60)
        logger.info(f"Quantity Skew Experiment ({self.config.max_imbalance_ratio:.0f}x imbalance)")
        logger.info("=" * 60)

        start_time = time.time()

        # Setup
        self.assign_node_sizes()
        self.setup_byzantine()

        # Initialize FL system
        if self.fl_available:
            fl_config = self.FLConfig(
                byzantine_threshold=0.49,
                use_detection=True,
                use_healing=True,
            )
            fl_system = self.MycelixFL(config=fl_config)
        else:
            fl_system = None
            logger.warning("Running in fallback mode without mycelix_fl")

        # Results collection
        all_detected: Set[str] = set()
        latencies = []
        round_results = []

        # Run FL rounds
        for round_num in range(1, self.config.num_rounds + 1):
            # Generate gradients with quantity skew
            gradients = self.generate_gradients(round_num)

            # Inject Byzantine attacks
            modified_gradients = self.inject_byzantine_attacks(gradients)

            # Run FL round
            start = time.perf_counter()
            if fl_system:
                result = fl_system.execute_round(modified_gradients, round_num=round_num)
                detected = result.byzantine_nodes
            else:
                detected = self._fallback_detection(modified_gradients)
            latency = (time.perf_counter() - start) * 1000

            latencies.append(latency)
            all_detected.update(detected)

            # Skip warmup rounds
            if round_num <= self.config.warmup_rounds:
                continue

            # Per-round metrics
            metrics = self.calculate_detection_metrics(
                detected, self.byzantine_ids, self.config.num_nodes
            )
            metrics["round"] = round_num
            round_results.append(metrics)

            if round_num % 10 == 0:
                logger.info(
                    f"Round {round_num}: "
                    f"Det={metrics['detection_rate']:.2%}, "
                    f"FP={metrics['false_positive_rate']:.2%}"
                )

        # Final metrics
        final_metrics = self.calculate_detection_metrics(
            all_detected, self.byzantine_ids, self.config.num_nodes
        )

        # Analyze by size
        size_analysis = self._analyze_by_size(all_detected)
        fairness_metrics = self._calculate_fairness_metrics(size_analysis)

        elapsed = time.time() - start_time

        results = {
            "scenario": "quantity_skew",
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "detection_accuracy": final_metrics["detection_rate"],
            "false_positive_rate": final_metrics["false_positive_rate"],
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "f1_score": final_metrics["f1_score"],
            "model_accuracy": 0.0,
            "timing": {
                "total_seconds": elapsed,
                "mean_latency_ms": float(np.mean(latencies)) if latencies else 0,
                "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0,
            },
            "size_analysis": size_analysis,
            "fairness_metrics": fairness_metrics,
            "node_sizes": self.node_sizes,
            "node_samples": self.node_samples,
            "per_round_metrics": round_results,
            "byzantine_nodes": list(self.byzantine_ids),
            "detected_nodes": list(all_detected),
        }

        logger.info(f"\nFinal Results:")
        logger.info(f"  Detection Rate: {final_metrics['detection_rate']:.2%}")
        logger.info(f"  False Positive Rate: {final_metrics['false_positive_rate']:.2%}")
        logger.info(f"  Small Node FP Rate: {size_analysis['small']['fp_rate']:.2%}")
        logger.info(f"  Large Node FP Rate: {size_analysis['large']['fp_rate']:.2%}")
        logger.info(f"  FP Rate Gini: {fairness_metrics['fp_rate_gini']:.3f}")
        logger.info(f"  Total Time: {elapsed:.1f}s")

        return results

    def _fallback_detection(self, gradients: Dict[str, np.ndarray]) -> Set[str]:
        """Fallback Byzantine detection."""
        norms = {k: np.linalg.norm(v) for k, v in gradients.items()}
        values = list(norms.values())

        if len(values) < 3:
            return set()

        mean_norm = np.mean(values)
        std_norm = np.std(values)

        if std_norm < 1e-6:
            return set()

        detected = set()
        for node_id, norm in norms.items():
            z_score = abs(norm - mean_norm) / std_norm
            if z_score > 2.0:
                detected.add(node_id)

        return detected


def run_quantity_skew_experiment(
    output_dir: Optional[Path] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run quantity skew experiment.

    Args:
        output_dir: Directory for output files
        **kwargs: Additional config parameters

    Returns:
        Experiment results dictionary
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config = QuantitySkewConfig(**kwargs)
    scenario = QuantitySkewScenario(config)
    results = scenario.run()

    if output_dir:
        output_path = Path(output_dir) / "quantity_skew.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results to {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantity Skew Non-IID Experiment")
    parser.add_argument("--num-nodes", type=int, default=20, help="Total nodes")
    parser.add_argument("--byzantine-ratio", type=float, default=0.3, help="Byzantine ratio")
    parser.add_argument("--num-rounds", type=int, default=100, help="Number of rounds")
    parser.add_argument("--large-nodes", type=int, default=2, help="Large node count")
    parser.add_argument("--medium-nodes", type=int, default=8, help="Medium node count")
    parser.add_argument("--small-nodes", type=int, default=10, help="Small node count")
    parser.add_argument("--byzantine-placement", type=str, default="random",
                        choices=["random", "large", "small"],
                        help="Where to place Byzantine nodes")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    results = run_quantity_skew_experiment(
        output_dir=args.output_dir,
        num_nodes=args.large_nodes + args.medium_nodes + args.small_nodes,
        byzantine_ratio=args.byzantine_ratio,
        num_rounds=args.num_rounds,
        large_nodes=args.large_nodes,
        medium_nodes=args.medium_nodes,
        small_nodes=args.small_nodes,
        byzantine_placement=args.byzantine_placement,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("Quantity Skew Results:")
    print(f"  Detection Rate: {results['detection_accuracy']:.2%}")
    print(f"  False Positive Rate: {results['false_positive_rate']:.2%}")
    print(f"  Small Node FP: {results['size_analysis']['small']['fp_rate']:.2%}")
    print(f"  Fairness (Gini): {results['fairness_metrics']['fp_rate_gini']:.3f}")
