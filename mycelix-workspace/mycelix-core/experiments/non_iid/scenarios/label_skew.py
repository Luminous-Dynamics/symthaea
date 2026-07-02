#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Label Skew Non-IID Scenario

Tests Byzantine detection under label distribution heterogeneity using
Dirichlet allocation. Each node receives samples with different class
distributions controlled by the concentration parameter alpha:

- alpha = 0.1: Extreme skew (1-2 classes per node)
- alpha = 0.5: Moderate skew (3-5 classes per node)
- alpha = 1.0: Mild skew (roughly uniform)

This scenario validates that the Byzantine detection system can
distinguish between legitimate gradient differences (caused by
non-IID data) and actual malicious behavior.

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
sys.path.insert(0, str(PROJECT_ROOT / "0TML" / "benchmarks" / "datasets"))

logger = logging.getLogger(__name__)


@dataclass
class LabelSkewConfig:
    """Configuration for label skew experiments."""
    # Dirichlet concentration parameter
    alpha: float = 0.5

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

    # Byzantine attack configuration
    attack_types: List[str] = field(
        default_factory=lambda: ["gradient_scaling", "sign_flip", "gaussian_noise"]
    )

    # Detection settings
    cosine_threshold_min: float = -0.5
    cosine_threshold_max: float = 0.95

    # Reproducibility
    seed: int = 42

    # Minimum samples per client
    min_samples_per_client: int = 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "num_nodes": self.num_nodes,
            "byzantine_ratio": self.byzantine_ratio,
            "num_rounds": self.num_rounds,
            "local_epochs": self.local_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_rounds": self.warmup_rounds,
            "dataset": self.dataset,
            "attack_types": self.attack_types,
            "seed": self.seed,
        }


class LabelSkewScenario:
    """
    Label skew experiment scenario.

    Creates a federated learning setup where each client has a different
    distribution of labels, then measures Byzantine detection accuracy.

    Example:
        >>> config = LabelSkewConfig(alpha=0.1, num_nodes=20)
        >>> scenario = LabelSkewScenario(config)
        >>> results = scenario.run()
        >>> print(f"Detection: {results['detection_accuracy']:.2%}")
    """

    def __init__(self, config: Optional[LabelSkewConfig] = None):
        """Initialize scenario with configuration."""
        self.config = config or LabelSkewConfig()
        self.rng = np.random.RandomState(self.config.seed)

        # Import dependencies
        self._setup_imports()

        # Data storage
        self.client_loaders: List[Any] = []
        self.test_loader: Any = None
        self.model: Any = None
        self.label_distributions: Dict[int, np.ndarray] = {}

        # Results
        self.round_metrics: List[Dict[str, Any]] = []
        self.detection_history: List[Set[str]] = []
        self.byzantine_ids: Set[str] = set()

        logger.info(f"LabelSkewScenario initialized (alpha={self.config.alpha})")

    def _setup_imports(self):
        """Setup and validate imports."""
        # Try importing PyTorch
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            self.torch = torch
            self.nn = nn
            self.optim = optim
            self.torch_available = True
        except ImportError:
            logger.warning("PyTorch not available")
            self.torch_available = False

        # Try importing dataset loaders
        try:
            from mnist_loader import MNISTFederated
            from cifar10_loader import CIFAR10Federated
            self.MNISTFederated = MNISTFederated
            self.CIFAR10Federated = CIFAR10Federated
            self.datasets_available = True
        except ImportError as e:
            logger.warning(f"Dataset loaders not available: {e}")
            self.datasets_available = False

        # Try importing mycelix_fl
        try:
            from mycelix_fl import (
                MycelixFL, FLConfig, RoundResult,
                MultiLayerByzantineDetector,
                AttackOrchestrator, create_attack, AttackType,
            )
            self.MycelixFL = MycelixFL
            self.FLConfig = FLConfig
            self.MultiLayerByzantineDetector = MultiLayerByzantineDetector
            self.AttackOrchestrator = AttackOrchestrator
            self.create_attack = create_attack
            self.AttackType = AttackType
            self.fl_available = True
        except ImportError as e:
            logger.warning(f"mycelix_fl not available: {e}")
            self.fl_available = False

    def _create_simple_model(self) -> Any:
        """Create a simple CNN model for benchmarking."""
        if not self.torch_available:
            return None

        class SimpleCNN(self.nn.Module):
            def __init__(inner_self, num_classes: int = 10, input_channels: int = 3):
                super().__init__()
                inner_self.features = self.nn.Sequential(
                    self.nn.Conv2d(input_channels, 32, 3, padding=1),
                    self.nn.ReLU(),
                    self.nn.MaxPool2d(2),
                    self.nn.Conv2d(32, 64, 3, padding=1),
                    self.nn.ReLU(),
                    self.nn.MaxPool2d(2),
                    self.nn.Conv2d(64, 64, 3, padding=1),
                    self.nn.ReLU(),
                )
                inner_self.classifier = self.nn.Sequential(
                    self.nn.Flatten(),
                    self.nn.Linear(64 * 8 * 8, 256),
                    self.nn.ReLU(),
                    self.nn.Dropout(0.5),
                    self.nn.Linear(256, num_classes),
                )

            def forward(inner_self, x):
                x = inner_self.features(x)
                x = inner_self.classifier(x)
                return x

        input_channels = 1 if self.config.dataset == "mnist" else 3
        return SimpleCNN(num_classes=10, input_channels=input_channels)

    def setup_data(self) -> None:
        """Setup Non-IID data distribution using Dirichlet allocation."""
        logger.info(f"Setting up label-skewed data (alpha={self.config.alpha})")

        if not self.datasets_available:
            logger.warning("Using synthetic data (dataset loaders not available)")
            self._setup_synthetic_data()
            return

        # Load dataset
        if self.config.dataset == "mnist":
            dataset = self.MNISTFederated(
                data_dir=self.config.data_dir,
                download=True,
            )
        else:
            dataset = self.CIFAR10Federated(
                data_dir=self.config.data_dir,
                download=True,
            )

        # Create Non-IID Dirichlet split
        self.client_loaders = dataset.create_non_iid_dirichlet(
            num_clients=self.config.num_nodes,
            alpha=self.config.alpha,
            min_samples=self.config.min_samples_per_client,
        )
        self.test_loader = dataset.get_test_loader()

        # Record label distributions
        for i, loader in enumerate(self.client_loaders):
            labels = []
            for _, batch_labels in loader:
                labels.extend(batch_labels.numpy().tolist())
            dist = np.bincount(labels, minlength=10)
            self.label_distributions[i] = dist / dist.sum()

        logger.info(f"Created {len(self.client_loaders)} client loaders")

    def _setup_synthetic_data(self) -> None:
        """Create synthetic Non-IID data for testing without real datasets."""
        # Generate synthetic gradients that simulate label skew
        gradient_size = 10000

        for i in range(self.config.num_nodes):
            # Assign dominant classes based on Dirichlet
            class_probs = self.rng.dirichlet([self.config.alpha] * 10)
            self.label_distributions[i] = class_probs

        logger.info("Using synthetic data mode")

    def setup_byzantine(self) -> Set[str]:
        """Setup Byzantine nodes."""
        num_byzantine = int(self.config.num_nodes * self.config.byzantine_ratio)
        node_ids = [f"node_{i}" for i in range(self.config.num_nodes)]

        byzantine_ids = set(
            self.rng.choice(node_ids, num_byzantine, replace=False)
        )

        self.byzantine_ids = byzantine_ids
        logger.info(f"Byzantine nodes: {sorted(byzantine_ids)}")

        return byzantine_ids

    def generate_gradients(self, round_num: int) -> Dict[str, np.ndarray]:
        """
        Generate gradients reflecting label skew.

        Honest nodes produce gradients that vary based on their label distribution.
        Byzantine nodes produce malicious gradients.
        """
        gradient_size = 10000
        gradients = {}

        # Base gradient (what a fully balanced node would produce)
        base_gradient = self.rng.randn(gradient_size).astype(np.float32) * 0.1

        for i in range(self.config.num_nodes):
            node_id = f"node_{i}"

            # Create gradient based on label distribution
            # Nodes with different label distributions will have different gradient directions
            label_dist = self.label_distributions.get(i, np.ones(10) / 10)

            # Dominant classes influence gradient direction
            class_vectors = [
                self.rng.randn(gradient_size).astype(np.float32)
                for _ in range(10)
            ]

            # Weighted combination based on label distribution
            gradient = np.zeros(gradient_size, dtype=np.float32)
            for c in range(10):
                gradient += label_dist[c] * class_vectors[c]

            gradient = gradient / (np.linalg.norm(gradient) + 1e-8) * 0.1

            # Add noise to simulate training variance
            noise = self.rng.randn(gradient_size).astype(np.float32) * 0.01
            gradient = base_gradient + gradient + noise

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
            elif attack_type == "label_flip":
                # Reverse gradient direction for label flip attack
                modified[node_id] = -grad * self.rng.uniform(1, 3)
            else:
                # Default: random garbage
                modified[node_id] = self.rng.randn(*grad.shape).astype(np.float32)

        return modified

    def run_fl_round(
        self,
        round_num: int,
        fl_system: Any,
        gradients: Dict[str, np.ndarray],
    ) -> Tuple[Any, float]:
        """Run a single FL round and return result with timing."""
        start = time.perf_counter()
        result = fl_system.execute_round(gradients, round_num=round_num)
        elapsed = (time.perf_counter() - start) * 1000
        return result, elapsed

    def evaluate_model(self) -> float:
        """Evaluate model accuracy (placeholder for real training)."""
        # In a full implementation, this would evaluate the trained model
        # For now, return a simulated accuracy based on round number
        return 0.0

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

    def run(self) -> Dict[str, Any]:
        """
        Run the complete label skew experiment.

        Returns:
            Dictionary with all experiment results.
        """
        logger.info("=" * 60)
        logger.info(f"Label Skew Experiment (alpha={self.config.alpha})")
        logger.info("=" * 60)

        start_time = time.time()

        # Setup
        self.setup_data()
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
            # Generate gradients with label skew
            gradients = self.generate_gradients(round_num)

            # Inject Byzantine attacks
            modified_gradients = self.inject_byzantine_attacks(gradients)

            # Run FL round
            if fl_system:
                result, latency = self.run_fl_round(round_num, fl_system, modified_gradients)
                detected = result.byzantine_nodes
            else:
                # Fallback: simple z-score detection
                detected = self._fallback_detection(modified_gradients)
                latency = 0.0

            latencies.append(latency)
            all_detected.update(detected)

            # Skip warmup rounds for metrics
            if round_num <= self.config.warmup_rounds:
                continue

            # Calculate per-round metrics
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
                    f"FP={metrics['false_positive_rate']:.2%}, "
                    f"Latency={latency:.1f}ms"
                )

        # Final metrics
        final_metrics = self.calculate_detection_metrics(
            all_detected, self.byzantine_ids, self.config.num_nodes
        )

        # Calculate convergence (round when detection stabilizes)
        convergence_round = self._calculate_convergence_round(round_results)

        # Compile results
        elapsed = time.time() - start_time

        results = {
            "scenario": "label_skew",
            "alpha": self.config.alpha,
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "detection_accuracy": final_metrics["detection_rate"],
            "false_positive_rate": final_metrics["false_positive_rate"],
            "precision": final_metrics["precision"],
            "recall": final_metrics["recall"],
            "f1_score": final_metrics["f1_score"],
            "model_accuracy": self.evaluate_model(),
            "convergence_round": convergence_round,
            "timing": {
                "total_seconds": elapsed,
                "mean_latency_ms": float(np.mean(latencies)) if latencies else 0,
                "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0,
                "p99_latency_ms": float(np.percentile(latencies, 99)) if latencies else 0,
            },
            "per_round_metrics": round_results,
            "label_distributions": {
                str(k): v.tolist() for k, v in self.label_distributions.items()
            },
            "byzantine_nodes": list(self.byzantine_ids),
            "detected_nodes": list(all_detected),
        }

        logger.info(f"\nFinal Results:")
        logger.info(f"  Detection Rate: {final_metrics['detection_rate']:.2%}")
        logger.info(f"  False Positive Rate: {final_metrics['false_positive_rate']:.2%}")
        logger.info(f"  F1 Score: {final_metrics['f1_score']:.3f}")
        logger.info(f"  Convergence Round: {convergence_round}")
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

    def _calculate_convergence_round(
        self, round_results: List[Dict[str, Any]]
    ) -> Optional[int]:
        """Calculate the round at which detection stabilizes."""
        if not round_results:
            return None

        # Look for first round with detection_rate >= 0.9 that stays stable
        target_rate = 0.9
        window_size = 5

        for i in range(len(round_results) - window_size):
            window = round_results[i:i + window_size]
            if all(r["detection_rate"] >= target_rate for r in window):
                return window[0]["round"]

        return None


def run_label_skew_experiment(
    alpha: float = 0.5,
    output_dir: Optional[Path] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run label skew experiment with specified alpha.

    Args:
        alpha: Dirichlet concentration parameter
        output_dir: Directory for output files
        **kwargs: Additional config parameters

    Returns:
        Experiment results dictionary
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config = LabelSkewConfig(alpha=alpha, **kwargs)
    scenario = LabelSkewScenario(config)
    results = scenario.run()

    if output_dir:
        output_path = Path(output_dir) / f"label_skew_alpha_{alpha}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results to {output_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Label Skew Non-IID Experiment")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha")
    parser.add_argument("--num-nodes", type=int, default=20, help="Number of nodes")
    parser.add_argument("--byzantine-ratio", type=float, default=0.3, help="Byzantine ratio")
    parser.add_argument("--num-rounds", type=int, default=100, help="Number of rounds")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    results = run_label_skew_experiment(
        alpha=args.alpha,
        output_dir=args.output_dir,
        num_nodes=args.num_nodes,
        byzantine_ratio=args.byzantine_ratio,
        num_rounds=args.num_rounds,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print(f"Label Skew (alpha={args.alpha}) Results:")
    print(f"  Detection Rate: {results['detection_accuracy']:.2%}")
    print(f"  False Positive Rate: {results['false_positive_rate']:.2%}")
    print(f"  F1 Score: {results['f1_score']:.3f}")
