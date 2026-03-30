#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Byzantine Detection Accuracy Benchmark
=======================================

Tests PoGQ against baseline defenses across multiple attack types
and Byzantine ratios for MLSys 2026 paper.

Attack Types:
- Label-flip attack
- Gradient scaling attack
- Sign-flip attack
- Backdoor attack

Byzantine Ratios: 10%, 20%, 33%, 45%

Author: Luminous Dynamics
Date: January 8, 2026
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
import sys

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import defenses
try:
    from src.defenses.fltrust import FLTrust
    from src.defenses.krum import Krum, MultiKrum
    from src.defenses.fedavg import FedAvg
    from src.pogq_system import ProofOfGoodQuality
except ImportError:
    # Fallback implementations
    FLTrust = None
    Krum = None
    MultiKrum = None
    FedAvg = None
    ProofOfGoodQuality = None


@dataclass
class AttackConfig:
    """Configuration for a Byzantine attack."""
    name: str
    attack_type: str
    params: Dict = field(default_factory=dict)


@dataclass
class DetectionMetrics:
    """Metrics for Byzantine detection evaluation."""
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 1.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 1.0

    @property
    def f1_score(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def detection_rate(self) -> float:
        """Same as recall - fraction of Byzantines detected."""
        return self.recall

    @property
    def false_positive_rate(self) -> float:
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_negatives": self.true_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "detection_rate": self.detection_rate,
            "false_positive_rate": self.false_positive_rate,
        }


class ByzantineAttacks:
    """Byzantine attack implementations."""

    @staticmethod
    def label_flip(honest_gradient: np.ndarray, params: Dict = None) -> np.ndarray:
        """
        Label-flip attack: Submit gradient that would train wrong labels.
        Simulated by negating gradient with some noise.
        """
        scale = params.get("scale", 1.0) if params else 1.0
        noise_std = params.get("noise_std", 0.1) if params else 0.1

        # Negate and add noise
        attack_grad = -honest_gradient * scale
        attack_grad += np.random.normal(0, noise_std, honest_gradient.shape)

        return attack_grad

    @staticmethod
    def gradient_scaling(honest_gradient: np.ndarray, params: Dict = None) -> np.ndarray:
        """
        Gradient scaling attack: Multiply gradient by large factor.
        """
        scale = params.get("scale", 100.0) if params else 100.0

        return honest_gradient * scale

    @staticmethod
    def sign_flip(honest_gradient: np.ndarray, params: Dict = None) -> np.ndarray:
        """
        Sign-flip attack: Negate the gradient direction.
        """
        intensity = params.get("intensity", 1.0) if params else 1.0

        return -honest_gradient * intensity

    @staticmethod
    def backdoor(honest_gradient: np.ndarray, params: Dict = None) -> np.ndarray:
        """
        Backdoor attack: Inject trigger pattern while maintaining
        reasonable gradient magnitude.
        """
        trigger_strength = params.get("trigger_strength", 2.0) if params else 2.0
        blend_ratio = params.get("blend_ratio", 0.5) if params else 0.5

        # Create trigger pattern
        trigger = np.random.randn(*honest_gradient.shape) * trigger_strength

        # Blend with honest gradient to maintain magnitude
        attack_grad = blend_ratio * honest_gradient + (1 - blend_ratio) * trigger

        # Normalize to similar magnitude as honest
        honest_norm = np.linalg.norm(honest_gradient)
        attack_norm = np.linalg.norm(attack_grad)
        if attack_norm > 0:
            attack_grad = attack_grad * (honest_norm / attack_norm) * 1.5

        return attack_grad


class PoGQDefense:
    """PoGQ Byzantine defense wrapper for benchmarking."""

    def __init__(self, quality_threshold: float = 0.3):
        self.quality_threshold = quality_threshold
        self.name = "PoGQ"

    def aggregate_and_detect(
        self,
        gradients: Dict[str, np.ndarray],
        server_gradient: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Set[str]]:
        """
        Aggregate gradients and detect Byzantine nodes.

        Returns:
            Tuple of (aggregated_gradient, detected_byzantine_set)
        """
        if len(gradients) == 0:
            raise ValueError("No gradients provided")

        # Compute reference gradient (mean of all)
        all_grads = list(gradients.values())
        reference = np.mean(all_grads, axis=0)
        reference_norm = np.linalg.norm(reference)

        detected = set()
        valid_gradients = []
        valid_weights = []

        for node_id, grad in gradients.items():
            # Compute quality score based on alignment and magnitude
            grad_norm = np.linalg.norm(grad)

            # Cosine similarity with reference
            if grad_norm > 1e-8 and reference_norm > 1e-8:
                cosine_sim = np.dot(grad, reference) / (grad_norm * reference_norm)
            else:
                cosine_sim = 0.0

            # Magnitude ratio
            if reference_norm > 1e-8:
                mag_ratio = grad_norm / reference_norm
            else:
                mag_ratio = 1.0

            # Quality score: combine direction and magnitude
            direction_score = (cosine_sim + 1) / 2  # Map [-1, 1] to [0, 1]

            # Penalize extreme magnitudes
            if mag_ratio > 3 or mag_ratio < 0.1:
                magnitude_score = 0.2
            else:
                magnitude_score = 1.0 - min(abs(mag_ratio - 1.0) / 2.0, 0.5)

            quality = 0.7 * direction_score + 0.3 * magnitude_score

            if quality < self.quality_threshold:
                detected.add(node_id)
            else:
                valid_gradients.append(grad)
                valid_weights.append(quality)

        # Weighted aggregation of valid gradients
        if len(valid_gradients) > 0:
            weights = np.array(valid_weights)
            weights = weights / weights.sum()
            aggregated = np.zeros_like(valid_gradients[0])
            for w, g in zip(weights, valid_gradients):
                aggregated += w * g
        else:
            # Fallback to median if all rejected
            aggregated = np.median(all_grads, axis=0)

        return aggregated, detected


class FLTrustDefense:
    """FLTrust defense wrapper for benchmarking."""

    def __init__(self, clip_threshold: float = 1.0):
        self.clip_threshold = clip_threshold
        self.name = "FLTrust"

    def aggregate_and_detect(
        self,
        gradients: Dict[str, np.ndarray],
        server_gradient: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Set[str]]:
        """Aggregate using FLTrust and detect based on trust scores."""
        if len(gradients) == 0:
            raise ValueError("No gradients provided")

        all_grads = list(gradients.values())
        node_ids = list(gradients.keys())

        # Use server gradient or compute reference
        if server_gradient is not None:
            reference = server_gradient
        else:
            # Use trimmed mean as reference
            reference = np.mean(all_grads, axis=0)

        ref_norm = np.linalg.norm(reference)

        detected = set()
        trust_scores = []

        for node_id, grad in gradients.items():
            # Compute trust score (ReLU of cosine similarity)
            grad_norm = np.linalg.norm(grad)

            if grad_norm > 1e-8 and ref_norm > 1e-8:
                cosine = np.dot(grad, reference) / (grad_norm * ref_norm)
            else:
                cosine = 0.0

            trust = max(0.0, cosine)
            trust_scores.append(trust)

            # Detect as Byzantine if no positive alignment
            if trust < 0.01:
                detected.add(node_id)

        # Normalize trust scores
        trust_scores = np.array(trust_scores)
        trust_scores = np.clip(trust_scores, 0, self.clip_threshold)

        if trust_scores.sum() > 1e-8:
            trust_scores = trust_scores / trust_scores.sum()
        else:
            trust_scores = np.ones(len(all_grads)) / len(all_grads)

        # Weighted aggregation
        aggregated = np.zeros_like(all_grads[0])
        for w, g in zip(trust_scores, all_grads):
            aggregated += w * g

        return aggregated, detected


class KrumDefense:
    """Krum defense wrapper for benchmarking."""

    def __init__(self, f: int = 2):
        self.f = f
        self.name = "Krum"

    def aggregate_and_detect(
        self,
        gradients: Dict[str, np.ndarray],
        server_gradient: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Set[str]]:
        """Aggregate using Krum and detect outliers."""
        if len(gradients) == 0:
            raise ValueError("No gradients provided")

        all_grads = list(gradients.values())
        node_ids = list(gradients.keys())
        n = len(all_grads)

        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sum((all_grads[i] - all_grads[j]) ** 2)
                distances[i, j] = dist
                distances[j, i] = dist

        # Compute Krum scores
        m = max(1, n - self.f - 2)
        scores = []
        for i in range(n):
            sorted_dists = np.sort(distances[i])
            score = np.sum(sorted_dists[1:m+1])
            scores.append(score)

        scores = np.array(scores)

        # Select best gradient
        selected_idx = int(np.argmin(scores))
        aggregated = all_grads[selected_idx]

        # Detect: nodes with high scores are Byzantine
        median_score = np.median(scores)
        mad = np.median(np.abs(scores - median_score))
        threshold = median_score + 3 * max(mad, 1e-8)

        detected = set()
        for i, score in enumerate(scores):
            if score > threshold:
                detected.add(node_ids[i])

        return aggregated, detected


class FedAvgDefense:
    """FedAvg (baseline) defense wrapper for benchmarking."""

    def __init__(self):
        self.name = "FedAvg"

    def aggregate_and_detect(
        self,
        gradients: Dict[str, np.ndarray],
        server_gradient: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Set[str]]:
        """Simple averaging - no Byzantine detection."""
        if len(gradients) == 0:
            raise ValueError("No gradients provided")

        all_grads = list(gradients.values())
        aggregated = np.mean(all_grads, axis=0)

        # FedAvg does not detect Byzantines
        detected = set()

        return aggregated, detected


class ByzantineAccuracyBenchmark:
    """
    Byzantine detection accuracy benchmark suite.

    Compares PoGQ against FLTrust, Krum, and FedAvg across
    multiple attack types and Byzantine ratios.
    """

    def __init__(
        self,
        seed: int = 42,
        quick_mode: bool = False,
        gradient_dim: int = 10000,
    ):
        """
        Initialize benchmark.

        Args:
            seed: Random seed
            quick_mode: Run fewer trials
            gradient_dim: Dimension of gradient vectors
        """
        self.seed = seed
        self.quick_mode = quick_mode
        self.gradient_dim = gradient_dim

        np.random.seed(seed)

        # Benchmark configuration
        if quick_mode:
            self.num_nodes = 20
            self.num_trials = 3
            self.byzantine_ratios = [0.10, 0.20, 0.33, 0.45]
        else:
            self.num_nodes = 50
            self.num_trials = 10
            self.byzantine_ratios = [0.10, 0.20, 0.33, 0.45]

        # Attack configurations
        self.attacks = [
            AttackConfig("label_flip", "label_flip", {"scale": 1.0}),
            AttackConfig("gradient_scaling", "gradient_scaling", {"scale": 100.0}),
            AttackConfig("sign_flip", "sign_flip", {"intensity": 1.0}),
            AttackConfig("backdoor", "backdoor", {"trigger_strength": 2.0}),
        ]

        # Defenses to compare
        self.defenses = [
            PoGQDefense(quality_threshold=0.35),
            FLTrustDefense(clip_threshold=1.0),
            KrumDefense(f=2),
            FedAvgDefense(),
        ]

    def generate_gradients(
        self,
        byzantine_ratio: float,
        attack_config: AttackConfig,
    ) -> Tuple[Dict[str, np.ndarray], Set[str], np.ndarray]:
        """
        Generate honest and Byzantine gradients.

        Returns:
            gradients: Dict of node_id -> gradient
            actual_byzantine: Set of Byzantine node IDs
            true_gradient: The true (honest) gradient direction
        """
        num_byzantine = int(self.num_nodes * byzantine_ratio)
        num_honest = self.num_nodes - num_byzantine

        # Generate true gradient direction
        true_gradient = np.random.randn(self.gradient_dim).astype(np.float32)
        true_gradient /= (np.linalg.norm(true_gradient) + 1e-8)

        gradients = {}
        actual_byzantine = set()

        # Generate honest gradients (with small noise)
        for i in range(num_honest):
            node_id = f"honest_{i}"
            noise = np.random.randn(self.gradient_dim).astype(np.float32) * 0.1
            gradients[node_id] = true_gradient + noise

        # Generate Byzantine gradients
        attack_func = getattr(ByzantineAttacks, attack_config.attack_type)

        for i in range(num_byzantine):
            node_id = f"byzantine_{i}"
            actual_byzantine.add(node_id)

            # Use average honest gradient as base
            honest_grads = [g for nid, g in gradients.items() if "honest" in nid]
            base_gradient = np.mean(honest_grads, axis=0) if honest_grads else true_gradient

            gradients[node_id] = attack_func(base_gradient, attack_config.params)

        return gradients, actual_byzantine, true_gradient

    def evaluate_defense(
        self,
        defense,
        gradients: Dict[str, np.ndarray],
        actual_byzantine: Set[str],
        true_gradient: np.ndarray,
    ) -> Tuple[DetectionMetrics, Dict]:
        """
        Evaluate a defense mechanism.

        Returns:
            metrics: Detection metrics
            extra: Additional metrics (gradient error, etc.)
        """
        node_ids = set(gradients.keys())
        honest_nodes = node_ids - actual_byzantine

        # Run defense
        start_time = time.perf_counter()
        aggregated, detected = defense.aggregate_and_detect(gradients)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # Compute detection metrics
        metrics = DetectionMetrics(
            true_positives=len(detected & actual_byzantine),
            false_positives=len(detected & honest_nodes),
            false_negatives=len(actual_byzantine - detected),
            true_negatives=len(honest_nodes - detected),
        )

        # Compute aggregation quality
        agg_norm = np.linalg.norm(aggregated)
        true_norm = np.linalg.norm(true_gradient)

        if agg_norm > 1e-8 and true_norm > 1e-8:
            cosine_sim = np.dot(aggregated, true_gradient) / (agg_norm * true_norm)
        else:
            cosine_sim = 0.0

        gradient_error = np.linalg.norm(aggregated - true_gradient)

        extra = {
            "cosine_similarity": float(cosine_sim),
            "gradient_error": float(gradient_error),
            "latency_ms": float(latency_ms),
        }

        return metrics, extra

    def run_single_attack(
        self,
        attack_config: AttackConfig,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run benchmark for a single attack type.

        Returns:
            Dict mapping byzantine_ratio -> defense_name -> metrics
        """
        results = {}

        for byz_ratio in self.byzantine_ratios:
            ratio_key = f"{byz_ratio:.2f}"
            results[ratio_key] = {}

            # Aggregate metrics across trials
            trial_metrics = {d.name.lower(): [] for d in self.defenses}
            trial_extras = {d.name.lower(): [] for d in self.defenses}

            for trial in range(self.num_trials):
                # Generate gradients
                gradients, actual_byz, true_grad = self.generate_gradients(
                    byz_ratio, attack_config
                )

                # Evaluate each defense
                for defense in self.defenses:
                    metrics, extra = self.evaluate_defense(
                        defense, gradients, actual_byz, true_grad
                    )
                    trial_metrics[defense.name.lower()].append(metrics)
                    trial_extras[defense.name.lower()].append(extra)

            # Aggregate results
            for defense in self.defenses:
                name = defense.name.lower()
                metrics_list = trial_metrics[name]
                extras_list = trial_extras[name]

                # Average detection metrics
                avg_metrics = DetectionMetrics(
                    true_positives=int(np.mean([m.true_positives for m in metrics_list])),
                    false_positives=int(np.mean([m.false_positives for m in metrics_list])),
                    false_negatives=int(np.mean([m.false_negatives for m in metrics_list])),
                    true_negatives=int(np.mean([m.true_negatives for m in metrics_list])),
                )

                result = avg_metrics.to_dict()
                result["cosine_similarity"] = float(np.mean([e["cosine_similarity"] for e in extras_list]))
                result["gradient_error"] = float(np.mean([e["gradient_error"] for e in extras_list]))
                result["latency_ms"] = float(np.mean([e["latency_ms"] for e in extras_list]))

                results[ratio_key][name] = result

        return results

    def run_all(self) -> Dict[str, Any]:
        """
        Run all benchmarks.

        Returns:
            Complete benchmark results
        """
        results = {}

        for attack in self.attacks:
            print(f"  Running {attack.name} attack benchmark...")
            attack_results = self.run_single_attack(attack)
            results[attack.name] = attack_results

        return results


def main():
    """Run Byzantine accuracy benchmark standalone."""
    print("Byzantine Detection Accuracy Benchmark")
    print("=" * 50)

    benchmark = ByzantineAccuracyBenchmark(
        seed=42,
        quick_mode=True,
    )

    results = benchmark.run_all()

    # Print summary
    print("\nResults Summary:")
    print("-" * 50)

    for attack_name, attack_results in results.items():
        print(f"\n{attack_name}:")
        for ratio, ratio_results in attack_results.items():
            print(f"  Byzantine ratio {ratio}:")
            for method, metrics in ratio_results.items():
                print(f"    {method}: detection_rate={metrics['detection_rate']:.1%}, "
                      f"f1={metrics['f1_score']:.1%}")


if __name__ == "__main__":
    main()
