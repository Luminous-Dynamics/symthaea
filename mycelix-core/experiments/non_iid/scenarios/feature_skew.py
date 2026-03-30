#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Feature Skew Non-IID Scenario

Tests Byzantine detection when different nodes receive data with different
feature characteristics, even for the same labels. This simulates:

- Domain shift (e.g., different hospitals with different scanner settings)
- Device heterogeneity (e.g., different camera qualities)
- Environmental factors (e.g., lighting conditions)

Feature transformations applied:
- Brightness shift: Images darker/lighter
- Rotation: Images rotated by varying degrees
- Noise injection: Different noise levels
- Blur: Different blur amounts

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
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "0TML" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))
sys.path.insert(0, str(PROJECT_ROOT / "0TML" / "benchmarks" / "datasets"))

logger = logging.getLogger(__name__)


@dataclass
class FeatureTransformation:
    """Definition of a feature transformation."""
    type: str  # brightness, rotation, noise, blur
    range: Tuple[float, float]  # Parameter range

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "range": list(self.range)}


@dataclass
class FeatureSkewConfig:
    """Configuration for feature skew experiments."""
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

    # Feature transformation settings
    transformations: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {"type": "brightness", "range": [0.5, 1.5]},
            {"type": "rotation", "range": [-30, 30]},
            {"type": "noise", "range": [0.0, 0.3]},
            {"type": "blur", "range": [0, 3]},
        ]
    )

    # Fraction of nodes with transformed features
    transformation_ratio: float = 0.5

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
            "transformations": self.transformations,
            "transformation_ratio": self.transformation_ratio,
            "dataset": self.dataset,
            "attack_types": self.attack_types,
            "seed": self.seed,
        }


class FeatureTransformer:
    """Applies feature transformations to data."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def apply_brightness(
        self, data: np.ndarray, factor: float
    ) -> np.ndarray:
        """Apply brightness transformation."""
        return np.clip(data * factor, 0, 1)

    def apply_rotation(
        self, data: np.ndarray, angle: float
    ) -> np.ndarray:
        """Apply rotation transformation (simulated as gradient shift)."""
        # For gradient-level simulation, rotation affects gradient direction
        rotation_matrix = np.array([
            [np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
            [np.sin(np.radians(angle)), np.cos(np.radians(angle))]
        ])
        # Apply rotation to pairs of gradient elements
        if len(data) >= 2:
            data_2d = data[:len(data)//2*2].reshape(-1, 2)
            rotated = data_2d @ rotation_matrix.T
            result = np.concatenate([rotated.flatten(), data[len(data)//2*2:]])
            return result.astype(np.float32)
        return data

    def apply_noise(
        self, data: np.ndarray, std: float
    ) -> np.ndarray:
        """Apply Gaussian noise."""
        noise = self.rng.randn(*data.shape).astype(np.float32) * std
        return data + noise

    def apply_blur(
        self, data: np.ndarray, kernel_size: int
    ) -> np.ndarray:
        """Apply blur (simulated as smoothing)."""
        if kernel_size <= 0:
            return data
        # Simple moving average for gradient smoothing
        kernel = np.ones(kernel_size) / kernel_size
        return np.convolve(data, kernel, mode='same').astype(np.float32)

    def transform(
        self,
        data: np.ndarray,
        transform_type: str,
        param: float,
    ) -> np.ndarray:
        """Apply specified transformation."""
        if transform_type == "brightness":
            return self.apply_brightness(data, param)
        elif transform_type == "rotation":
            return self.apply_rotation(data, param)
        elif transform_type == "noise":
            return self.apply_noise(data, param)
        elif transform_type == "blur":
            return self.apply_blur(data, int(param))
        else:
            logger.warning(f"Unknown transformation: {transform_type}")
            return data


class FeatureSkewScenario:
    """
    Feature skew experiment scenario.

    Creates a federated learning setup where different nodes have
    different feature distributions (domain shift), then measures
    Byzantine detection accuracy.

    Example:
        >>> config = FeatureSkewConfig(num_nodes=20)
        >>> scenario = FeatureSkewScenario(config)
        >>> results = scenario.run()
        >>> print(f"Detection: {results['detection_accuracy']:.2%}")
    """

    def __init__(self, config: Optional[FeatureSkewConfig] = None):
        """Initialize scenario with configuration."""
        self.config = config or FeatureSkewConfig()
        self.rng = np.random.RandomState(self.config.seed)
        self.transformer = FeatureTransformer(seed=self.config.seed)

        # Import dependencies
        self._setup_imports()

        # Node feature assignments
        self.node_transformations: Dict[str, Dict[str, Any]] = {}
        self.byzantine_ids: Set[str] = set()

        logger.info(f"FeatureSkewScenario initialized")

    def _setup_imports(self):
        """Setup and validate imports."""
        # Try importing mycelix_fl
        try:
            from mycelix_fl import (
                MycelixFL, FLConfig,
                MultiLayerByzantineDetector,
                AttackOrchestrator, create_attack, AttackType,
            )
            self.MycelixFL = MycelixFL
            self.FLConfig = FLConfig
            self.fl_available = True
        except ImportError as e:
            logger.warning(f"mycelix_fl not available: {e}")
            self.fl_available = False

    def assign_feature_transformations(self) -> None:
        """Assign feature transformations to nodes."""
        node_ids = [f"node_{i}" for i in range(self.config.num_nodes)]
        num_transformed = int(len(node_ids) * self.config.transformation_ratio)

        # Select nodes to transform
        transformed_nodes = set(self.rng.choice(node_ids, num_transformed, replace=False))

        for node_id in node_ids:
            if node_id in transformed_nodes:
                # Assign random transformation
                transform = self.rng.choice(self.config.transformations)
                param_range = transform["range"]
                param = self.rng.uniform(param_range[0], param_range[1])

                self.node_transformations[node_id] = {
                    "type": transform["type"],
                    "param": param,
                    "transformed": True,
                }
            else:
                self.node_transformations[node_id] = {
                    "type": "none",
                    "param": 0,
                    "transformed": False,
                }

        logger.info(f"Assigned transformations to {num_transformed} nodes")

    def setup_byzantine(self) -> Set[str]:
        """Setup Byzantine nodes."""
        num_byzantine = int(self.config.num_nodes * self.config.byzantine_ratio)
        node_ids = [f"node_{i}" for i in range(self.config.num_nodes)]

        self.byzantine_ids = set(
            self.rng.choice(node_ids, num_byzantine, replace=False)
        )

        logger.info(f"Byzantine nodes: {sorted(self.byzantine_ids)}")
        return self.byzantine_ids

    def generate_gradients(self, round_num: int) -> Dict[str, np.ndarray]:
        """Generate gradients with feature skew applied."""
        gradient_size = 10000
        gradients = {}

        # Base gradient
        base_gradient = self.rng.randn(gradient_size).astype(np.float32) * 0.1

        for i in range(self.config.num_nodes):
            node_id = f"node_{i}"

            # Start with base gradient + small noise
            gradient = base_gradient.copy()
            noise = self.rng.randn(gradient_size).astype(np.float32) * 0.01
            gradient = gradient + noise

            # Apply feature transformation if assigned
            transform = self.node_transformations.get(node_id, {})
            if transform.get("transformed", False):
                gradient = self.transformer.transform(
                    gradient,
                    transform["type"],
                    transform["param"],
                )

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

    def _analyze_fp_by_transformation(
        self,
        detected: Set[str],
    ) -> Dict[str, Any]:
        """Analyze false positives by transformation type."""
        fp_by_type = {}

        for node_id in detected:
            if node_id in self.byzantine_ids:
                continue  # True positive

            transform = self.node_transformations.get(node_id, {})
            t_type = transform.get("type", "none")

            if t_type not in fp_by_type:
                fp_by_type[t_type] = 0
            fp_by_type[t_type] += 1

        return fp_by_type

    def run(self) -> Dict[str, Any]:
        """
        Run the complete feature skew experiment.

        Returns:
            Dictionary with all experiment results.
        """
        logger.info("=" * 60)
        logger.info("Feature Skew Experiment")
        logger.info("=" * 60)

        start_time = time.time()

        # Setup
        self.assign_feature_transformations()
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
            # Generate gradients with feature skew
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
            metrics["latency_ms"] = latency
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

        # Analyze FP by transformation
        fp_analysis = self._analyze_fp_by_transformation(all_detected)

        elapsed = time.time() - start_time

        results = {
            "scenario": "feature_skew",
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "detection_accuracy": final_metrics["detection_rate"],
            "false_positive_rate": final_metrics["false_positive_rate"],
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "f1_score": final_metrics["f1_score"],
            "model_accuracy": 0.0,  # Would be set by actual training
            "timing": {
                "total_seconds": elapsed,
                "mean_latency_ms": float(np.mean(latencies)) if latencies else 0,
                "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0,
            },
            "fp_by_transformation": fp_analysis,
            "node_transformations": {
                k: v for k, v in self.node_transformations.items()
            },
            "per_round_metrics": round_results,
            "byzantine_nodes": list(self.byzantine_ids),
            "detected_nodes": list(all_detected),
        }

        logger.info(f"\nFinal Results:")
        logger.info(f"  Detection Rate: {final_metrics['detection_rate']:.2%}")
        logger.info(f"  False Positive Rate: {final_metrics['false_positive_rate']:.2%}")
        logger.info(f"  FP by transformation: {fp_analysis}")
        logger.info(f"  Total Time: {elapsed:.1f}s")

        return results

    def _fallback_detection(self, gradients: Dict[str, np.ndarray]) -> Set[str]:
        """Fallback Byzantine detection using z-score."""
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


def run_feature_skew_experiment(
    output_dir: Optional[Path] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run feature skew experiment.

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

    config = FeatureSkewConfig(**kwargs)
    scenario = FeatureSkewScenario(config)
    results = scenario.run()

    if output_dir:
        output_path = Path(output_dir) / "feature_skew.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results to {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature Skew Non-IID Experiment")
    parser.add_argument("--num-nodes", type=int, default=20, help="Number of nodes")
    parser.add_argument("--byzantine-ratio", type=float, default=0.3, help="Byzantine ratio")
    parser.add_argument("--num-rounds", type=int, default=100, help="Number of rounds")
    parser.add_argument("--transformation-ratio", type=float, default=0.5,
                        help="Fraction of nodes with feature shift")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    results = run_feature_skew_experiment(
        output_dir=args.output_dir,
        num_nodes=args.num_nodes,
        byzantine_ratio=args.byzantine_ratio,
        num_rounds=args.num_rounds,
        transformation_ratio=args.transformation_ratio,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("Feature Skew Results:")
    print(f"  Detection Rate: {results['detection_accuracy']:.2%}")
    print(f"  False Positive Rate: {results['false_positive_rate']:.2%}")
    print(f"  F1 Score: {results['f1_score']:.3f}")
