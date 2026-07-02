#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Combined Skew Non-IID Scenario

Maximum stress test combining all Non-IID scenarios:
- Label skew (Dirichlet distribution)
- Feature skew (domain shift)
- Quantity skew (data imbalance)

This represents the most challenging real-world federated learning
scenario where all types of data heterogeneity exist simultaneously.

Example: A healthcare FL network where:
- Different hospitals specialize in different conditions (label skew)
- Different equipment produces different image characteristics (feature skew)
- Hospital sizes vary from 50 to 5000 beds (quantity skew)

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
class CombinedSkewConfig:
    """Configuration for combined skew experiments."""
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

    # Label skew (Dirichlet)
    label_alpha: float = 0.5

    # Feature skew
    feature_ratio: float = 0.3  # Fraction of nodes with feature shift
    feature_types: List[str] = field(
        default_factory=lambda: ["brightness", "noise", "blur"]
    )

    # Quantity skew
    quantity_imbalance: bool = True
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

    # Reproducibility
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.num_nodes,
            "byzantine_ratio": self.byzantine_ratio,
            "num_rounds": self.num_rounds,
            "label_alpha": self.label_alpha,
            "feature_ratio": self.feature_ratio,
            "feature_types": self.feature_types,
            "quantity_imbalance": self.quantity_imbalance,
            "large_nodes": self.large_nodes,
            "medium_nodes": self.medium_nodes,
            "small_nodes": self.small_nodes,
            "dataset": self.dataset,
            "attack_types": self.attack_types,
            "seed": self.seed,
        }


class CombinedSkewScenario:
    """
    Combined skew experiment scenario.

    Applies all three types of Non-IID skew simultaneously:
    1. Label skew via Dirichlet distribution
    2. Feature skew via transformations
    3. Quantity skew via data imbalance

    This is the most challenging scenario for Byzantine detection
    as legitimate gradient variance is maximized.

    Example:
        >>> config = CombinedSkewConfig(num_nodes=20)
        >>> scenario = CombinedSkewScenario(config)
        >>> results = scenario.run()
        >>> print(f"Detection: {results['detection_accuracy']:.2%}")
    """

    def __init__(self, config: Optional[CombinedSkewConfig] = None):
        """Initialize scenario with configuration."""
        self.config = config or CombinedSkewConfig()
        self.rng = np.random.RandomState(self.config.seed)

        # Import dependencies
        self._setup_imports()

        # Node characteristics
        self.label_distributions: Dict[str, np.ndarray] = {}
        self.feature_transforms: Dict[str, Dict[str, Any]] = {}
        self.node_sizes: Dict[str, str] = {}
        self.node_samples: Dict[str, int] = {}
        self.byzantine_ids: Set[str] = set()

        # Combined skew scores for analysis
        self.skew_scores: Dict[str, float] = {}

        logger.info("CombinedSkewScenario initialized")

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

    def setup_label_skew(self) -> None:
        """Setup label distribution using Dirichlet."""
        for i in range(self.config.num_nodes):
            node_id = f"node_{i}"
            class_probs = self.rng.dirichlet([self.config.label_alpha] * 10)
            self.label_distributions[node_id] = class_probs

        logger.info(f"Label skew: alpha={self.config.label_alpha}")

    def setup_feature_skew(self) -> None:
        """Setup feature transformations."""
        node_ids = [f"node_{i}" for i in range(self.config.num_nodes)]
        num_transformed = int(len(node_ids) * self.config.feature_ratio)

        transformed_nodes = set(self.rng.choice(node_ids, num_transformed, replace=False))

        for node_id in node_ids:
            if node_id in transformed_nodes:
                transform_type = self.rng.choice(self.config.feature_types)

                if transform_type == "brightness":
                    param = self.rng.uniform(0.5, 1.5)
                elif transform_type == "noise":
                    param = self.rng.uniform(0.0, 0.3)
                elif transform_type == "blur":
                    param = self.rng.randint(0, 4)
                else:
                    param = 1.0

                self.feature_transforms[node_id] = {
                    "type": transform_type,
                    "param": param,
                    "transformed": True,
                }
            else:
                self.feature_transforms[node_id] = {
                    "type": "none",
                    "param": 0,
                    "transformed": False,
                }

        logger.info(f"Feature skew: {num_transformed} nodes transformed")

    def setup_quantity_skew(self) -> None:
        """Setup data quantity distribution."""
        node_ids = [f"node_{i}" for i in range(self.config.num_nodes)]
        self.rng.shuffle(node_ids)

        if self.config.quantity_imbalance:
            idx = 0
            for _ in range(self.config.large_nodes):
                if idx < len(node_ids):
                    self.node_sizes[node_ids[idx]] = "large"
                    self.node_samples[node_ids[idx]] = self.config.large_samples
                    idx += 1

            for _ in range(self.config.medium_nodes):
                if idx < len(node_ids):
                    self.node_sizes[node_ids[idx]] = "medium"
                    self.node_samples[node_ids[idx]] = self.config.medium_samples
                    idx += 1

            for _ in range(self.config.small_nodes):
                if idx < len(node_ids):
                    self.node_sizes[node_ids[idx]] = "small"
                    self.node_samples[node_ids[idx]] = self.config.small_samples
                    idx += 1

            # Fill remaining with medium
            while idx < len(node_ids):
                self.node_sizes[node_ids[idx]] = "medium"
                self.node_samples[node_ids[idx]] = self.config.medium_samples
                idx += 1
        else:
            # Equal distribution
            for node_id in node_ids:
                self.node_sizes[node_id] = "medium"
                self.node_samples[node_id] = self.config.medium_samples

        logger.info(f"Quantity skew: imbalance={self.config.quantity_imbalance}")

    def setup_byzantine(self) -> Set[str]:
        """Setup Byzantine nodes."""
        num_byzantine = int(self.config.num_nodes * self.config.byzantine_ratio)
        node_ids = [f"node_{i}" for i in range(self.config.num_nodes)]

        self.byzantine_ids = set(
            self.rng.choice(node_ids, num_byzantine, replace=False)
        )

        logger.info(f"Byzantine nodes: {len(self.byzantine_ids)}")
        return self.byzantine_ids

    def calculate_skew_score(self, node_id: str) -> float:
        """
        Calculate combined skew score for a node.

        Higher score = more heterogeneous from population.
        """
        score = 0.0

        # Label skew contribution
        label_dist = self.label_distributions.get(node_id, np.ones(10) / 10)
        # Measure concentration (low entropy = high skew)
        entropy = -np.sum(label_dist * np.log(label_dist + 1e-10))
        max_entropy = np.log(10)
        label_skew = 1 - (entropy / max_entropy)
        score += label_skew * 0.4

        # Feature skew contribution
        if self.feature_transforms.get(node_id, {}).get("transformed", False):
            score += 0.3

        # Quantity skew contribution
        size = self.node_sizes.get(node_id, "medium")
        if size == "small":
            score += 0.3  # Small nodes have high variance
        elif size == "large":
            score += 0.0  # Large nodes are stable

        self.skew_scores[node_id] = score
        return score

    def generate_gradients(self, round_num: int) -> Dict[str, np.ndarray]:
        """Generate gradients with all types of skew applied."""
        gradient_size = 10000
        gradients = {}

        # Population gradient
        base_gradient = self.rng.randn(gradient_size).astype(np.float32) * 0.1

        for node_id in self.label_distributions.keys():
            gradient = np.zeros(gradient_size, dtype=np.float32)

            # 1. Apply label skew
            label_dist = self.label_distributions[node_id]
            class_vectors = [
                self.rng.randn(gradient_size).astype(np.float32)
                for _ in range(10)
            ]
            for c in range(10):
                gradient += label_dist[c] * class_vectors[c]
            gradient = gradient / (np.linalg.norm(gradient) + 1e-8) * 0.1

            # 2. Apply quantity skew (variance based on sample count)
            num_samples = self.node_samples.get(node_id, 1000)
            variance_scale = 1.0 / np.sqrt(num_samples / 100)
            noise = self.rng.randn(gradient_size).astype(np.float32) * 0.05 * variance_scale
            gradient = base_gradient + gradient + noise

            # 3. Apply feature skew
            transform = self.feature_transforms.get(node_id, {})
            if transform.get("transformed", False):
                t_type = transform["type"]
                t_param = transform["param"]

                if t_type == "brightness":
                    gradient = gradient * t_param
                elif t_type == "noise":
                    extra_noise = self.rng.randn(gradient_size).astype(np.float32) * t_param
                    gradient = gradient + extra_noise
                elif t_type == "blur":
                    if t_param > 0:
                        kernel = np.ones(int(t_param)) / int(t_param)
                        gradient = np.convolve(gradient, kernel, mode='same').astype(np.float32)

            gradients[node_id] = gradient

            # Calculate combined skew score
            self.calculate_skew_score(node_id)

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

    def _analyze_fp_by_skew_level(
        self,
        detected: Set[str],
    ) -> Dict[str, Any]:
        """Analyze false positives by combined skew level."""
        # Categorize nodes by skew score
        low_skew = []
        medium_skew = []
        high_skew = []

        for node_id, score in self.skew_scores.items():
            if node_id in self.byzantine_ids:
                continue  # Skip Byzantine nodes

            if score < 0.3:
                low_skew.append(node_id)
            elif score < 0.6:
                medium_skew.append(node_id)
            else:
                high_skew.append(node_id)

        # Calculate FP rates
        def fp_rate(nodes):
            if not nodes:
                return 0.0
            fps = sum(1 for n in nodes if n in detected)
            return fps / len(nodes)

        return {
            "low_skew": {
                "count": len(low_skew),
                "fp_rate": fp_rate(low_skew),
            },
            "medium_skew": {
                "count": len(medium_skew),
                "fp_rate": fp_rate(medium_skew),
            },
            "high_skew": {
                "count": len(high_skew),
                "fp_rate": fp_rate(high_skew),
            },
        }

    def run(self) -> Dict[str, Any]:
        """
        Run the complete combined skew experiment.

        Returns:
            Dictionary with all experiment results.
        """
        logger.info("=" * 60)
        logger.info("Combined Skew Experiment (Maximum Stress Test)")
        logger.info("=" * 60)

        start_time = time.time()

        # Setup all skew types
        self.setup_label_skew()
        self.setup_feature_skew()
        self.setup_quantity_skew()
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
            # Generate gradients with combined skew
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

        # Analyze by skew level
        skew_analysis = self._analyze_fp_by_skew_level(all_detected)

        elapsed = time.time() - start_time

        results = {
            "scenario": "combined_skew",
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
            "skew_analysis": skew_analysis,
            "skew_scores": self.skew_scores,
            "label_distributions": {
                k: v.tolist() for k, v in self.label_distributions.items()
            },
            "feature_transforms": self.feature_transforms,
            "node_sizes": self.node_sizes,
            "per_round_metrics": round_results,
            "byzantine_nodes": list(self.byzantine_ids),
            "detected_nodes": list(all_detected),
        }

        logger.info(f"\nFinal Results:")
        logger.info(f"  Detection Rate: {final_metrics['detection_rate']:.2%}")
        logger.info(f"  False Positive Rate: {final_metrics['false_positive_rate']:.2%}")
        logger.info(f"  High-skew node FP rate: {skew_analysis['high_skew']['fp_rate']:.2%}")
        logger.info(f"  Low-skew node FP rate: {skew_analysis['low_skew']['fp_rate']:.2%}")
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


def run_combined_skew_experiment(
    output_dir: Optional[Path] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run combined skew experiment.

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

    config = CombinedSkewConfig(**kwargs)
    scenario = CombinedSkewScenario(config)
    results = scenario.run()

    if output_dir:
        output_path = Path(output_dir) / "combined_skew.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results to {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combined Skew Non-IID Experiment")
    parser.add_argument("--num-nodes", type=int, default=20, help="Number of nodes")
    parser.add_argument("--byzantine-ratio", type=float, default=0.3, help="Byzantine ratio")
    parser.add_argument("--num-rounds", type=int, default=100, help="Number of rounds")
    parser.add_argument("--label-alpha", type=float, default=0.5, help="Dirichlet alpha")
    parser.add_argument("--feature-ratio", type=float, default=0.3,
                        help="Fraction with feature shift")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    results = run_combined_skew_experiment(
        output_dir=args.output_dir,
        num_nodes=args.num_nodes,
        byzantine_ratio=args.byzantine_ratio,
        num_rounds=args.num_rounds,
        label_alpha=args.label_alpha,
        feature_ratio=args.feature_ratio,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("Combined Skew Results:")
    print(f"  Detection Rate: {results['detection_accuracy']:.2%}")
    print(f"  False Positive Rate: {results['false_positive_rate']:.2%}")
    print(f"  F1 Score: {results['f1_score']:.3f}")
    print(f"  High-skew FP: {results['skew_analysis']['high_skew']['fp_rate']:.2%}")
