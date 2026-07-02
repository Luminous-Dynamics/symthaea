#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Table 1: Byzantine Detection Rate (Sign-Flip Attack)

Reproduces Table 1 from the Mycelix paper, measuring detection rates
across different Byzantine ratios for the sign-flip attack.

Expected results:
| Byzantine Ratio | FedAvg | Krum  | FLTrust | PoGQ (Ours) |
|-----------------|--------|-------|---------|-------------|
| 10%             | 0.0%   | 92.0% | 98.0%   | 99.0%       |
| 20%             | 0.0%   | 85.0% | 95.0%   | 98.0%       |
| 33%             | 0.0%   | 68.0% | 88.0%   | 95.0%       |
| 45%             | 0.0%   | 45.0% | 72.0%   | 89.0%       |
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str = "config.yaml") -> dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class ByzantineDetector:
    """Byzantine detection using z-score outlier detection."""

    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold

    def detect(self, gradients: List[np.ndarray]) -> List[int]:
        """Detect Byzantine nodes based on gradient norm outliers."""
        if len(gradients) < 3:
            return []

        norms = np.array([np.linalg.norm(g) for g in gradients])
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)

        if std_norm < 1e-6:
            return []

        z_scores = np.abs((norms - mean_norm) / std_norm)
        return np.where(z_scores > self.threshold)[0].tolist()


class Aggregator:
    """Base class for gradient aggregation methods."""

    def aggregate(self, gradients: List[np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class FedAvgAggregator(Aggregator):
    """FedAvg - simple averaging, no Byzantine detection."""

    def detect_byzantine(self, gradients: List[np.ndarray]) -> List[int]:
        return []  # No detection

    def aggregate(self, gradients: List[np.ndarray]) -> np.ndarray:
        return np.mean(gradients, axis=0)


class KrumAggregator(Aggregator):
    """Krum - distance-based Byzantine detection."""

    def __init__(self, num_byzantine: int = 0):
        self.num_byzantine = num_byzantine

    def detect_byzantine(self, gradients: List[np.ndarray]) -> List[int]:
        n = len(gradients)
        if n <= 2:
            return []

        # Calculate pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(gradients[i] - gradients[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Krum scores: sum of distances to k nearest neighbors
        k = max(1, n - self.num_byzantine - 2)
        scores = []
        for i in range(n):
            dists = np.sort(distances[i, :])[1:k+1]  # Exclude self
            scores.append(np.sum(dists))

        # Nodes with high scores are likely Byzantine
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        if std_score < 1e-6:
            return []

        z_scores = (np.array(scores) - mean_score) / std_score
        return np.where(z_scores > 1.5)[0].tolist()

    def aggregate(self, gradients: List[np.ndarray]) -> np.ndarray:
        byzantine_idx = self.detect_byzantine(gradients)
        honest = [g for i, g in enumerate(gradients) if i not in byzantine_idx]
        if not honest:
            honest = gradients
        return np.mean(honest, axis=0)


class FLTrustAggregator(Aggregator):
    """FLTrust - trust-based aggregation with root gradient."""

    def __init__(self, root_gradient: np.ndarray = None):
        self.root_gradient = root_gradient

    def detect_byzantine(self, gradients: List[np.ndarray]) -> List[int]:
        if self.root_gradient is None:
            self.root_gradient = np.mean(gradients, axis=0)

        # Compute cosine similarity with root
        root_norm = np.linalg.norm(self.root_gradient)
        if root_norm < 1e-6:
            return []

        similarities = []
        for g in gradients:
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-6:
                similarities.append(0)
            else:
                sim = np.dot(g, self.root_gradient) / (g_norm * root_norm)
                similarities.append(sim)

        # Low similarity = likely Byzantine
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        if std_sim < 1e-6:
            return []

        z_scores = (mean_sim - np.array(similarities)) / std_sim
        return np.where(z_scores > 1.5)[0].tolist()

    def aggregate(self, gradients: List[np.ndarray]) -> np.ndarray:
        byzantine_idx = self.detect_byzantine(gradients)
        honest = [g for i, g in enumerate(gradients) if i not in byzantine_idx]
        if not honest:
            honest = gradients
        return np.mean(honest, axis=0)


class PoGQAggregator(Aggregator):
    """Proof of Good Quality - our proposed method."""

    def __init__(self, z_threshold: float = 2.0,
                 norm_weight: float = 0.4,
                 cosine_weight: float = 0.4,
                 variance_weight: float = 0.2):
        self.z_threshold = z_threshold
        self.norm_weight = norm_weight
        self.cosine_weight = cosine_weight
        self.variance_weight = variance_weight
        self.reputation = {}
        self.decay = 0.7

    def compute_quality_score(self, gradient: np.ndarray,
                              all_gradients: List[np.ndarray]) -> float:
        """Compute quality score Q(g)."""
        # Norm score
        norms = [np.linalg.norm(g) for g in all_gradients]
        g_norm = np.linalg.norm(gradient)
        mean_norm = np.mean(norms)
        std_norm = np.std(norms) + 1e-6
        norm_score = 1.0 - min(1.0, abs(g_norm - mean_norm) / (2 * std_norm))

        # Cosine similarity score
        centroid = np.mean(all_gradients, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if g_norm > 1e-6 and centroid_norm > 1e-6:
            cos_sim = np.dot(gradient, centroid) / (g_norm * centroid_norm)
            cosine_score = (cos_sim + 1) / 2  # Normalize to [0, 1]
        else:
            cosine_score = 0.5

        # Variance score
        g_var = np.var(gradient)
        vars_all = [np.var(g) for g in all_gradients]
        mean_var = np.mean(vars_all)
        std_var = np.std(vars_all) + 1e-6
        variance_score = 1.0 - min(1.0, abs(g_var - mean_var) / (2 * std_var))

        # Weighted combination
        quality = (self.norm_weight * norm_score +
                   self.cosine_weight * cosine_score +
                   self.variance_weight * variance_score)

        return quality

    def detect_byzantine(self, gradients: List[np.ndarray]) -> List[int]:
        if len(gradients) < 3:
            return []

        # Compute quality scores
        quality_scores = []
        for g in gradients:
            q = self.compute_quality_score(g, gradients)
            quality_scores.append(q)

        quality_scores = np.array(quality_scores)
        mean_q = np.mean(quality_scores)
        std_q = np.std(quality_scores) + 1e-6

        # Detect outliers based on low quality
        z_scores = (mean_q - quality_scores) / std_q
        byzantine_idx = np.where(z_scores > self.z_threshold)[0].tolist()

        # Also check gradient norms for extreme outliers
        norms = np.array([np.linalg.norm(g) for g in gradients])
        norm_mean = np.mean(norms)
        norm_std = np.std(norms) + 1e-6
        norm_z = np.abs((norms - norm_mean) / norm_std)
        norm_outliers = np.where(norm_z > self.z_threshold)[0].tolist()

        # Union of both detection methods
        all_byzantine = list(set(byzantine_idx) | set(norm_outliers))
        return all_byzantine

    def aggregate(self, gradients: List[np.ndarray]) -> np.ndarray:
        byzantine_idx = self.detect_byzantine(gradients)
        honest = [g for i, g in enumerate(gradients) if i not in byzantine_idx]
        if not honest:
            honest = gradients
        return np.median(honest, axis=0)  # Use median for robustness


def generate_gradients(num_nodes: int,
                       byzantine_ratio: float,
                       gradient_dim: int,
                       attack_type: str = "sign_flip",
                       attack_multiplier: float = -5.0) -> Tuple[List[np.ndarray], List[int]]:
    """
    Generate synthetic gradients with Byzantine attackers.

    Returns:
        Tuple of (gradients, byzantine_indices)
    """
    num_byzantine = int(num_nodes * byzantine_ratio)

    gradients = []
    byzantine_indices = []

    # Generate honest gradients
    honest_mean = np.random.randn(gradient_dim) * 0.1

    for i in range(num_nodes):
        if i < num_byzantine:
            # Byzantine node
            byzantine_indices.append(i)

            if attack_type == "sign_flip":
                # Sign-flip attack: g = -multiplier * honest_gradient
                honest_gradient = honest_mean + np.random.randn(gradient_dim) * 0.05
                gradient = attack_multiplier * honest_gradient
            elif attack_type == "scaling":
                # Scaling attack: g = multiplier * honest_gradient
                honest_gradient = honest_mean + np.random.randn(gradient_dim) * 0.05
                gradient = attack_multiplier * honest_gradient
            elif attack_type == "label_flip":
                # Label-flip: inverted gradient direction
                honest_gradient = honest_mean + np.random.randn(gradient_dim) * 0.05
                gradient = -honest_gradient
            else:
                # Random attack
                gradient = np.random.randn(gradient_dim) * 10
        else:
            # Honest node
            gradient = honest_mean + np.random.randn(gradient_dim) * 0.05

        gradients.append(gradient)

    return gradients, byzantine_indices


def run_detection_experiment(aggregator: Aggregator,
                             num_nodes: int,
                             byzantine_ratio: float,
                             gradient_dim: int,
                             num_trials: int,
                             attack_type: str = "sign_flip") -> Dict:
    """Run detection experiment and compute metrics."""

    detection_rates = []
    false_positive_rates = []

    for trial in range(num_trials):
        np.random.seed(42 + trial)  # Reproducible but varied

        # Generate gradients
        gradients, true_byzantine = generate_gradients(
            num_nodes=num_nodes,
            byzantine_ratio=byzantine_ratio,
            gradient_dim=gradient_dim,
            attack_type=attack_type
        )

        # Detect Byzantine nodes
        detected = aggregator.detect_byzantine(gradients)

        # Compute metrics
        true_byzantine_set = set(true_byzantine)
        detected_set = set(detected)

        # True positives: correctly identified Byzantine nodes
        true_positives = len(true_byzantine_set & detected_set)

        # False positives: honest nodes incorrectly flagged
        false_positives = len(detected_set - true_byzantine_set)

        # Detection rate (recall)
        if len(true_byzantine_set) > 0:
            detection_rate = true_positives / len(true_byzantine_set)
        else:
            detection_rate = 1.0

        # False positive rate
        num_honest = num_nodes - len(true_byzantine_set)
        if num_honest > 0:
            fpr = false_positives / num_honest
        else:
            fpr = 0.0

        detection_rates.append(detection_rate)
        false_positive_rates.append(fpr)

    return {
        'detection_rate_mean': np.mean(detection_rates),
        'detection_rate_std': np.std(detection_rates),
        'fpr_mean': np.mean(false_positive_rates),
        'fpr_std': np.std(false_positive_rates),
        'num_trials': num_trials
    }


def main():
    parser = argparse.ArgumentParser(description='Table 1: Byzantine Detection Rate')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--output', default='output/results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer trials)')
    args = parser.parse_args()

    # Load configuration
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        config = {
            'global': {'random_seed': 42, 'num_trials': 10},
            'byzantine_detection': {
                'byzantine_ratios': [0.10, 0.20, 0.33, 0.45],
                'num_nodes': 100
            },
            'modes': {
                'quick': {'num_trials': 3, 'gradient_dim': 1000},
                'full': {'num_trials': 10, 'gradient_dim': 10000}
            },
            'pogq': {'z_score_threshold': 2.0}
        }

    # Set random seed
    np.random.seed(config['global']['random_seed'])

    # Experiment parameters
    mode = 'quick' if args.quick else 'full'
    num_trials = config['modes'][mode]['num_trials']
    gradient_dim = config['modes'][mode]['gradient_dim']
    num_nodes = config['byzantine_detection']['num_nodes']
    byzantine_ratios = config['byzantine_detection']['byzantine_ratios']

    # Initialize aggregators
    aggregators = {
        'fedavg': FedAvgAggregator(),
        'krum': KrumAggregator(num_byzantine=int(num_nodes * 0.45)),
        'fltrust': FLTrustAggregator(),
        'pogq': PoGQAggregator(z_threshold=config['pogq']['z_score_threshold'])
    }

    print("=" * 60)
    print("Table 1: Byzantine Detection Rate (Sign-Flip Attack)")
    print("=" * 60)
    print(f"Nodes: {num_nodes}, Gradient Dim: {gradient_dim}, Trials: {num_trials}")
    print("-" * 60)

    results = {
        'experiment': 'table1_byzantine_detection',
        'attack_type': 'sign_flip',
        'num_nodes': num_nodes,
        'gradient_dim': gradient_dim,
        'num_trials': num_trials,
        'data': {}
    }

    # Run experiments
    for ratio in tqdm(byzantine_ratios, desc="Byzantine Ratios"):
        ratio_key = f"{int(ratio * 100)}%"
        results['data'][ratio_key] = {}

        for name, aggregator in aggregators.items():
            # Update Krum's Byzantine estimate
            if isinstance(aggregator, KrumAggregator):
                aggregator.num_byzantine = int(num_nodes * ratio)

            metrics = run_detection_experiment(
                aggregator=aggregator,
                num_nodes=num_nodes,
                byzantine_ratio=ratio,
                gradient_dim=gradient_dim,
                num_trials=num_trials,
                attack_type='sign_flip'
            )

            results['data'][ratio_key][name] = metrics

    # Print results table
    print("\nResults:")
    print("-" * 60)
    header = "| Byz Ratio | FedAvg | Krum   | FLTrust | PoGQ   |"
    print(header)
    print("|" + "-" * 59 + "|")

    for ratio_key in results['data']:
        row = f"| {ratio_key:9} |"
        for method in ['fedavg', 'krum', 'fltrust', 'pogq']:
            rate = results['data'][ratio_key][method]['detection_rate_mean'] * 100
            row += f" {rate:5.1f}% |"
        print(row)

    print("-" * 60)

    # Save results
    output_dir = Path(__file__).parent.parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'table1_byzantine_detection.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == '__main__':
    main()
