#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Table 2: Aggregation Latency Comparison

Reproduces Table 3 from the Mycelix paper, measuring aggregation latency
across different methods (PoGQ, Krum, FLTrust, FedAvg).

Expected results:
| Method  | Median (ms) | P95 (ms) | P99 (ms) | Mean (ms) |
|---------|-------------|----------|----------|-----------|
| PoGQ    | 0.452       | 0.812    | 1.124    | 0.498     |
| FLTrust | 0.385       | 0.702    | 0.958    | 0.421     |
| Krum    | 2.845       | 4.521    | 5.823    | 3.012     |
| FedAvg  | 0.124       | 0.215    | 0.298    | 0.138     |
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str = "config.yaml") -> dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class FedAvgAggregator:
    """FedAvg - simple averaging."""

    def aggregate(self, gradients: List[np.ndarray]) -> np.ndarray:
        return np.mean(gradients, axis=0)


class KrumAggregator:
    """Krum - distance-based selection (O(n^2) complexity)."""

    def __init__(self, num_byzantine: int = 0):
        self.num_byzantine = num_byzantine

    def aggregate(self, gradients: List[np.ndarray]) -> np.ndarray:
        n = len(gradients)
        if n <= 2:
            return np.mean(gradients, axis=0)

        # Calculate pairwise distances - O(n^2)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(gradients[i] - gradients[j])
                distances[i, j] = dist
                distances[j, i] = dist

        # Krum scores
        k = max(1, n - self.num_byzantine - 2)
        scores = []
        for i in range(n):
            sorted_dists = np.sort(distances[i, :])[1:k+1]
            scores.append(np.sum(sorted_dists))

        # Select gradient with minimum score
        best_idx = np.argmin(scores)
        return gradients[best_idx]


class FLTrustAggregator:
    """FLTrust - trust-weighted aggregation."""

    def __init__(self):
        self.root_gradient = None

    def aggregate(self, gradients: List[np.ndarray]) -> np.ndarray:
        if not gradients:
            raise ValueError("No gradients to aggregate")

        # Use mean as root gradient (simulated trusted data)
        self.root_gradient = np.mean(gradients, axis=0)
        root_norm = np.linalg.norm(self.root_gradient) + 1e-10

        # Compute trust scores based on cosine similarity
        trust_scores = []
        for g in gradients:
            g_norm = np.linalg.norm(g) + 1e-10
            cos_sim = np.dot(g, self.root_gradient) / (g_norm * root_norm)
            # ReLU trust: only positive similarity contributes
            trust = max(0, cos_sim)
            trust_scores.append(trust)

        # Normalize trust scores
        total_trust = sum(trust_scores) + 1e-10
        weights = [t / total_trust for t in trust_scores]

        # Weighted aggregation
        result = np.zeros_like(gradients[0])
        for w, g in zip(weights, gradients):
            # Normalize gradient to root gradient's direction/magnitude
            g_norm = np.linalg.norm(g) + 1e-10
            normalized_g = g * (root_norm / g_norm)
            result += w * normalized_g

        return result


class PoGQAggregator:
    """Proof of Good Quality - our method."""

    def __init__(self, z_threshold: float = 2.0):
        self.z_threshold = z_threshold

    def aggregate(self, gradients: List[np.ndarray]) -> np.ndarray:
        if not gradients:
            raise ValueError("No gradients to aggregate")

        if len(gradients) < 3:
            return np.median(gradients, axis=0)

        # Compute gradient norms - O(n)
        norms = np.array([np.linalg.norm(g) for g in gradients])
        mean_norm = np.mean(norms)
        std_norm = np.std(norms) + 1e-6

        # Detect outliers using z-score
        z_scores = np.abs((norms - mean_norm) / std_norm)
        honest_mask = z_scores <= self.z_threshold

        # Filter honest gradients
        honest_gradients = [g for g, honest in zip(gradients, honest_mask) if honest]

        if not honest_gradients:
            honest_gradients = gradients

        # Use median for Byzantine robustness
        return np.median(honest_gradients, axis=0)


def benchmark_latency(aggregator,
                      gradients: List[np.ndarray],
                      warmup: int = 100,
                      iterations: int = 1000) -> Dict:
    """Benchmark aggregation latency."""

    # Warmup
    for _ in range(warmup):
        _ = aggregator.aggregate(gradients)

    # Measure
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        _ = aggregator.aggregate(gradients)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)

    return {
        'median_ms': float(np.median(latencies)),
        'mean_ms': float(np.mean(latencies)),
        'std_ms': float(np.std(latencies)),
        'p50_ms': float(np.percentile(latencies, 50)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'min_ms': float(np.min(latencies)),
        'max_ms': float(np.max(latencies)),
        'iterations': iterations
    }


def main():
    parser = argparse.ArgumentParser(description='Table 2: Aggregation Latency')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--output', default='output/results', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer iterations)')
    parser.add_argument('--trials', type=int, default=None, help='Number of benchmark trials')
    args = parser.parse_args()

    # Load configuration
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        config = {
            'global': {'random_seed': 42},
            'latency': {
                'warmup_iterations': 100,
                'measurement_iterations': 1000,
                'num_nodes': 100
            },
            'modes': {
                'quick': {'gradient_dim': 1000},
                'full': {'gradient_dim': 10000}
            }
        }

    # Set random seed
    np.random.seed(config['global']['random_seed'])

    # Parameters
    mode = 'quick' if args.quick else 'full'
    gradient_dim = config['modes'][mode].get('gradient_dim', 10000)
    num_nodes = config['latency']['num_nodes']
    warmup = 50 if args.quick else config['latency']['warmup_iterations']
    iterations = args.trials or (200 if args.quick else config['latency']['measurement_iterations'])

    print("=" * 60)
    print("Table 2: Aggregation Latency Comparison")
    print("=" * 60)
    print(f"Nodes: {num_nodes}, Gradient Dim: {gradient_dim}")
    print(f"Warmup: {warmup}, Iterations: {iterations}")
    print("-" * 60)

    # Generate test gradients
    gradients = [np.random.randn(gradient_dim) * 0.1 for _ in range(num_nodes)]

    # Initialize aggregators
    aggregators = {
        'fedavg': FedAvgAggregator(),
        'krum': KrumAggregator(num_byzantine=int(num_nodes * 0.2)),
        'fltrust': FLTrustAggregator(),
        'pogq': PoGQAggregator(z_threshold=2.0)
    }

    results = {
        'experiment': 'table2_latency',
        'num_nodes': num_nodes,
        'gradient_dim': gradient_dim,
        'warmup_iterations': warmup,
        'measurement_iterations': iterations,
        'data': {}
    }

    # Run benchmarks
    for name, aggregator in tqdm(aggregators.items(), desc="Methods"):
        metrics = benchmark_latency(
            aggregator=aggregator,
            gradients=gradients,
            warmup=warmup,
            iterations=iterations
        )
        results['data'][name] = metrics

    # Print results table
    print("\nResults:")
    print("-" * 65)
    print("| Method  | Median (ms) | P95 (ms) | P99 (ms) | Mean (ms) |")
    print("|" + "-" * 64 + "|")

    for method in ['pogq', 'fltrust', 'krum', 'fedavg']:
        data = results['data'][method]
        print(f"| {method:7} | {data['median_ms']:11.3f} | {data['p95_ms']:8.3f} | "
              f"{data['p99_ms']:8.3f} | {data['mean_ms']:9.3f} |")

    print("-" * 65)

    # Compute speedup
    krum_median = results['data']['krum']['median_ms']
    pogq_median = results['data']['pogq']['median_ms']
    speedup = krum_median / pogq_median
    print(f"\nPoGQ is {speedup:.1f}x faster than Krum")

    # Save results
    output_dir = Path(__file__).parent.parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'table2_latency.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == '__main__':
    main()
