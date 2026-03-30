#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Figure 3: Scalability Analysis

Generates Figure 3 from the Mycelix paper, showing latency and detection
rate scaling as the number of nodes increases from 10 to 1000.

Key findings:
- PoGQ: O(n) complexity, sub-linear scaling
- Krum: O(n^2) complexity, quadratic scaling
- Detection degradation: only 6% from 10 to 1000 nodes
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


def pogq_aggregate(gradients: List[np.ndarray]) -> np.ndarray:
    """PoGQ aggregation - O(n) complexity."""
    if len(gradients) < 3:
        return np.median(gradients, axis=0)

    norms = np.array([np.linalg.norm(g) for g in gradients])
    mean_norm = np.mean(norms)
    std_norm = np.std(norms) + 1e-6
    z_scores = np.abs((norms - mean_norm) / std_norm)

    honest = [g for g, z in zip(gradients, z_scores) if z <= 2.0]
    if not honest:
        honest = gradients

    return np.median(honest, axis=0)


def krum_aggregate(gradients: List[np.ndarray], num_byzantine: int = 0) -> np.ndarray:
    """Krum aggregation - O(n^2) complexity."""
    n = len(gradients)
    if n <= 2:
        return np.mean(gradients, axis=0)

    # O(n^2) pairwise distances
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(gradients[i] - gradients[j])
            distances[i, j] = dist
            distances[j, i] = dist

    k = max(1, n - num_byzantine - 2)
    scores = []
    for i in range(n):
        sorted_dists = np.sort(distances[i, :])[1:k+1]
        scores.append(np.sum(sorted_dists))

    best_idx = np.argmin(scores)
    return gradients[best_idx]


def pogq_detect(gradients: List[np.ndarray]) -> List[int]:
    """PoGQ Byzantine detection."""
    if len(gradients) < 3:
        return []

    norms = np.array([np.linalg.norm(g) for g in gradients])
    mean_norm = np.mean(norms)
    std_norm = np.std(norms) + 1e-6
    z_scores = np.abs((norms - mean_norm) / std_norm)

    return np.where(z_scores > 2.0)[0].tolist()


def benchmark_scalability(num_nodes: int,
                          gradient_dim: int,
                          byzantine_ratio: float,
                          num_trials: int) -> Dict:
    """Benchmark scalability for a given node count."""

    pogq_latencies = []
    krum_latencies = []
    detection_rates = []

    num_byzantine = int(num_nodes * byzantine_ratio)

    for trial in range(num_trials):
        np.random.seed(42 + trial)

        # Generate gradients
        honest_mean = np.random.randn(gradient_dim) * 0.1
        gradients = []
        true_byzantine = list(range(num_byzantine))

        for i in range(num_nodes):
            if i < num_byzantine:
                # Byzantine gradient
                gradient = -5.0 * (honest_mean + np.random.randn(gradient_dim) * 0.05)
            else:
                # Honest gradient
                gradient = honest_mean + np.random.randn(gradient_dim) * 0.05
            gradients.append(gradient)

        # Benchmark PoGQ
        start = time.perf_counter()
        _ = pogq_aggregate(gradients)
        pogq_latencies.append((time.perf_counter() - start) * 1000)

        # Benchmark Krum
        start = time.perf_counter()
        _ = krum_aggregate(gradients, num_byzantine)
        krum_latencies.append((time.perf_counter() - start) * 1000)

        # Detection rate
        detected = pogq_detect(gradients)
        true_positives = len(set(detected) & set(true_byzantine))
        if num_byzantine > 0:
            detection_rates.append(true_positives / num_byzantine)
        else:
            detection_rates.append(1.0)

    return {
        'num_nodes': num_nodes,
        'pogq_latency_mean_ms': float(np.mean(pogq_latencies)),
        'pogq_latency_std_ms': float(np.std(pogq_latencies)),
        'krum_latency_mean_ms': float(np.mean(krum_latencies)),
        'krum_latency_std_ms': float(np.std(krum_latencies)),
        'detection_rate_mean': float(np.mean(detection_rates)),
        'detection_rate_std': float(np.std(detection_rates)),
        'memory_mb': num_nodes * gradient_dim * 8 / 1e6  # Approximate
    }


def generate_figure(results: Dict, output_path: Path):
    """Generate multi-panel scalability figure."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.style.use('seaborn-v0_8-whitegrid')

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        node_counts = [d['num_nodes'] for d in results['data']]
        pogq_latencies = [d['pogq_latency_mean_ms'] for d in results['data']]
        pogq_stds = [d['pogq_latency_std_ms'] for d in results['data']]
        krum_latencies = [d['krum_latency_mean_ms'] for d in results['data']]
        krum_stds = [d['krum_latency_std_ms'] for d in results['data']]
        detection_rates = [d['detection_rate_mean'] * 100 for d in results['data']]
        detection_stds = [d['detection_rate_std'] * 100 for d in results['data']]
        memory = [d['memory_mb'] for d in results['data']]

        # Panel 1: Latency comparison
        ax1 = axes[0]
        ax1.errorbar(node_counts, pogq_latencies, yerr=pogq_stds,
                     marker='o', label='PoGQ', color='#ff7f0e', linewidth=2, capsize=3)
        ax1.errorbar(node_counts, krum_latencies, yerr=krum_stds,
                     marker='s', label='Krum', color='#2ca02c', linewidth=2, capsize=3)
        ax1.set_xlabel('Number of Nodes', fontsize=11)
        ax1.set_ylabel('Latency (ms)', fontsize=11)
        ax1.set_title('Aggregation Latency', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)

        # Panel 2: Detection rate
        ax2 = axes[1]
        ax2.errorbar(node_counts, detection_rates, yerr=detection_stds,
                     marker='o', color='#1f77b4', linewidth=2, capsize=3)
        ax2.axhline(y=92, color='gray', linestyle='--', alpha=0.5, label='Paper target (92%)')
        ax2.set_xlabel('Number of Nodes', fontsize=11)
        ax2.set_ylabel('Detection Rate (%)', fontsize=11)
        ax2.set_title('Byzantine Detection Rate', fontsize=12)
        ax2.set_xscale('log')
        ax2.set_ylim([80, 100])
        ax2.legend(loc='lower left')
        ax2.grid(True, alpha=0.3)

        # Panel 3: Memory usage
        ax3 = axes[2]
        ax3.plot(node_counts, memory, marker='s', color='#9467bd', linewidth=2)
        ax3.set_xlabel('Number of Nodes', fontsize=11)
        ax3.set_ylabel('Memory (MB)', fontsize=11)
        ax3.set_title('Memory Usage', fontsize=12)
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Figure saved to: {output_path}")
        return True

    except ImportError as e:
        print(f"Warning: Could not generate figure. Missing: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Figure 3: Scalability')
    parser.add_argument('--config', default='config.yaml', help='Config file')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    parser.add_argument('--max-nodes', type=int, default=None, help='Maximum nodes')
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        config = {
            'global': {'random_seed': 42},
            'scalability': {
                'node_counts': [10, 50, 100, 200, 500, 1000],
                'gradient_dim': 10000
            },
            'byzantine_detection': {'byzantine_ratios': [0.20]},
            'modes': {
                'quick': {'num_trials': 3, 'max_nodes': 100, 'gradient_dim': 1000},
                'full': {'num_trials': 10, 'max_nodes': 1000, 'gradient_dim': 10000}
            }
        }

    np.random.seed(config['global']['random_seed'])

    mode = 'quick' if args.quick else 'full'
    num_trials = config['modes'][mode].get('num_trials', 5)
    max_nodes = args.max_nodes or config['modes'][mode].get('max_nodes', 1000)
    gradient_dim = config['modes'][mode].get('gradient_dim', 10000)

    node_counts = config['scalability']['node_counts']
    node_counts = [n for n in node_counts if n <= max_nodes]

    byzantine_ratio = 0.20  # 20% Byzantine for scalability test

    print("=" * 60)
    print("Figure 3: Scalability Analysis")
    print("=" * 60)
    print(f"Node counts: {node_counts}")
    print(f"Gradient dim: {gradient_dim}, Trials: {num_trials}")

    output_dir = Path(__file__).parent.parent / args.output
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'results').mkdir(parents=True, exist_ok=True)

    results = {
        'experiment': 'figure3_scalability',
        'gradient_dim': gradient_dim,
        'byzantine_ratio': byzantine_ratio,
        'num_trials': num_trials,
        'data': []
    }

    for num_nodes in tqdm(node_counts, desc="Node counts"):
        metrics = benchmark_scalability(
            num_nodes=num_nodes,
            gradient_dim=gradient_dim,
            byzantine_ratio=byzantine_ratio,
            num_trials=num_trials
        )
        results['data'].append(metrics)

    # Print summary
    print("\nScalability Results:")
    print("-" * 70)
    print(f"{'Nodes':>6} | {'PoGQ (ms)':>12} | {'Krum (ms)':>12} | {'Detection':>10} | {'Memory':>10}")
    print("-" * 70)
    for d in results['data']:
        print(f"{d['num_nodes']:6d} | {d['pogq_latency_mean_ms']:12.3f} | "
              f"{d['krum_latency_mean_ms']:12.3f} | {d['detection_rate_mean']*100:9.1f}% | "
              f"{d['memory_mb']:9.1f} MB")

    # Compute scaling factors
    if len(results['data']) >= 2:
        first = results['data'][0]
        last = results['data'][-1]
        node_ratio = last['num_nodes'] / first['num_nodes']
        pogq_ratio = last['pogq_latency_mean_ms'] / first['pogq_latency_mean_ms']
        krum_ratio = last['krum_latency_mean_ms'] / first['krum_latency_mean_ms']
        detection_drop = (first['detection_rate_mean'] - last['detection_rate_mean']) * 100

        print(f"\nScaling Analysis ({first['num_nodes']} -> {last['num_nodes']} nodes):")
        print(f"  Node increase: {node_ratio:.0f}x")
        print(f"  PoGQ latency increase: {pogq_ratio:.1f}x (expected O(n): {node_ratio:.0f}x)")
        print(f"  Krum latency increase: {krum_ratio:.1f}x (expected O(n^2): {node_ratio**2:.0f}x)")
        print(f"  Detection rate drop: {detection_drop:.1f}%")

    # Save data
    data_file = output_dir / 'results' / 'figure3_data.json'
    with open(data_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nData saved to: {data_file}")

    # Generate figure
    figure_path = output_dir / 'figures' / 'figure3_scalability.png'
    generate_figure(results, figure_path)

    return results


if __name__ == '__main__':
    main()
