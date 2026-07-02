#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Figure 2: Model Accuracy Convergence Over Rounds

Generates Figure 2 from the Mycelix paper, showing model accuracy
convergence with and without Byzantine attackers.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config(config_path: str = "config.yaml") -> dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class SimulatedFLNetwork:
    """Simulated federated learning network with Byzantine nodes."""

    def __init__(self, num_nodes: int, byzantine_ratio: float,
                 aggregation_method: str = 'pogq',
                 model_dim: int = 1000):
        self.num_nodes = num_nodes
        self.byzantine_ratio = byzantine_ratio
        self.num_byzantine = int(num_nodes * byzantine_ratio)
        self.aggregation_method = aggregation_method
        self.model_dim = model_dim

        # Initialize model
        self.global_model = np.zeros(model_dim)

        # True model (for accuracy computation)
        self.true_model = np.random.randn(model_dim) * 0.5

        # Learning rate
        self.lr = 0.01

    def compute_accuracy(self) -> float:
        """Compute current model accuracy (simulated)."""
        # Use cosine similarity as proxy for accuracy
        g_norm = np.linalg.norm(self.global_model)
        t_norm = np.linalg.norm(self.true_model)

        if g_norm < 1e-10 or t_norm < 1e-10:
            return 0.1  # Random baseline

        cos_sim = np.dot(self.global_model, self.true_model) / (g_norm * t_norm)

        # Map cosine similarity [-1, 1] to accuracy [0.1, 0.95]
        accuracy = 0.1 + (cos_sim + 1) / 2 * 0.85
        return min(0.95, max(0.1, accuracy))

    def generate_gradients(self) -> Tuple[List[np.ndarray], List[int]]:
        """Generate gradients from nodes."""
        gradients = []
        byzantine_indices = []

        # True gradient (towards true model)
        true_gradient = self.true_model - self.global_model

        for i in range(self.num_nodes):
            if i < self.num_byzantine:
                # Byzantine: sign-flip attack
                byzantine_indices.append(i)
                gradient = -5.0 * (true_gradient + np.random.randn(self.model_dim) * 0.05)
            else:
                # Honest: noisy gradient towards true model
                gradient = true_gradient + np.random.randn(self.model_dim) * 0.1

            gradients.append(gradient)

        return gradients, byzantine_indices

    def aggregate_pogq(self, gradients: List[np.ndarray]) -> np.ndarray:
        """PoGQ aggregation."""
        norms = np.array([np.linalg.norm(g) for g in gradients])
        mean_norm = np.mean(norms)
        std_norm = np.std(norms) + 1e-6
        z_scores = np.abs((norms - mean_norm) / std_norm)

        honest = [g for g, z in zip(gradients, z_scores) if z <= 2.0]
        if not honest:
            honest = gradients

        return np.median(honest, axis=0)

    def aggregate_krum(self, gradients: List[np.ndarray]) -> np.ndarray:
        """Krum aggregation."""
        n = len(gradients)
        if n <= 2:
            return np.mean(gradients, axis=0)

        # Pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(gradients[i] - gradients[j])
                distances[i, j] = dist
                distances[j, i] = dist

        k = max(1, n - self.num_byzantine - 2)
        scores = []
        for i in range(n):
            sorted_dists = np.sort(distances[i, :])[1:k+1]
            scores.append(np.sum(sorted_dists))

        best_idx = np.argmin(scores)
        return gradients[best_idx]

    def aggregate_fedavg(self, gradients: List[np.ndarray]) -> np.ndarray:
        """FedAvg aggregation."""
        return np.mean(gradients, axis=0)

    def aggregate_fltrust(self, gradients: List[np.ndarray]) -> np.ndarray:
        """FLTrust aggregation."""
        root = np.mean(gradients, axis=0)
        root_norm = np.linalg.norm(root) + 1e-10

        trust_scores = []
        for g in gradients:
            g_norm = np.linalg.norm(g) + 1e-10
            cos_sim = np.dot(g, root) / (g_norm * root_norm)
            trust_scores.append(max(0, cos_sim))

        total_trust = sum(trust_scores) + 1e-10
        weights = [t / total_trust for t in trust_scores]

        result = np.zeros_like(gradients[0])
        for w, g in zip(weights, gradients):
            g_norm = np.linalg.norm(g) + 1e-10
            normalized_g = g * (root_norm / g_norm)
            result += w * normalized_g

        return result

    def training_round(self) -> float:
        """Execute one training round."""
        gradients, _ = self.generate_gradients()

        if self.aggregation_method == 'pogq':
            aggregated = self.aggregate_pogq(gradients)
        elif self.aggregation_method == 'krum':
            aggregated = self.aggregate_krum(gradients)
        elif self.aggregation_method == 'fltrust':
            aggregated = self.aggregate_fltrust(gradients)
        else:  # fedavg
            aggregated = self.aggregate_fedavg(gradients)

        # Update model
        self.global_model += self.lr * aggregated

        return self.compute_accuracy()


def run_convergence_experiment(num_nodes: int,
                               byzantine_ratio: float,
                               num_rounds: int,
                               method: str,
                               num_trials: int) -> Dict:
    """Run convergence experiment."""
    all_accuracies = []

    for trial in range(num_trials):
        np.random.seed(42 + trial)

        network = SimulatedFLNetwork(
            num_nodes=num_nodes,
            byzantine_ratio=byzantine_ratio,
            aggregation_method=method
        )

        accuracies = [network.compute_accuracy()]  # Initial
        for _ in range(num_rounds):
            acc = network.training_round()
            accuracies.append(acc)

        all_accuracies.append(accuracies)

    all_accuracies = np.array(all_accuracies)

    return {
        'mean': all_accuracies.mean(axis=0).tolist(),
        'std': all_accuracies.std(axis=0).tolist(),
        'min': all_accuracies.min(axis=0).tolist(),
        'max': all_accuracies.max(axis=0).tolist(),
        'final_accuracy': float(all_accuracies[:, -1].mean())
    }


def generate_figure(results: Dict, output_path: Path):
    """Generate convergence figure."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.style.use('seaborn-v0_8-whitegrid')

        fig, ax = plt.subplots(figsize=(10, 6))

        rounds = list(range(len(results['data']['pogq']['mean'])))

        methods = ['pogq', 'fltrust', 'krum', 'fedavg']
        labels = {'pogq': 'PoGQ (Ours)', 'fltrust': 'FLTrust', 'krum': 'Krum', 'fedavg': 'FedAvg'}
        colors = {'pogq': '#ff7f0e', 'fltrust': '#1f77b4', 'krum': '#2ca02c', 'fedavg': '#d62728'}

        for method in methods:
            mean = np.array(results['data'][method]['mean']) * 100
            std = np.array(results['data'][method]['std']) * 100

            ax.plot(rounds, mean, label=labels[method], color=colors[method], linewidth=2)
            ax.fill_between(rounds, mean - std, mean + std, alpha=0.2, color=colors[method])

        ax.set_xlabel('Training Round', fontsize=12)
        ax.set_ylabel('Model Accuracy (%)', fontsize=12)
        ax.set_title(f'Convergence Under {int(results["byzantine_ratio"]*100)}% Byzantine Attack', fontsize=14)
        ax.legend(loc='lower right', fontsize=11)
        ax.set_xlim([0, len(rounds) - 1])
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Figure saved to: {output_path}")
        return True

    except ImportError as e:
        print(f"Warning: Could not generate figure. Missing: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Figure 2: Convergence')
    parser.add_argument('--config', default='config.yaml', help='Config file')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    args = parser.parse_args()

    # Load config
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        config = {
            'global': {'random_seed': 42},
            'convergence': {
                'num_rounds': 50,
                'num_nodes': 100,
                'byzantine_ratio': 0.30
            },
            'modes': {
                'quick': {'num_trials': 3, 'max_rounds': 20},
                'full': {'num_trials': 10, 'max_rounds': 50}
            }
        }

    np.random.seed(config['global']['random_seed'])

    mode = 'quick' if args.quick else 'full'
    num_trials = config['modes'][mode].get('num_trials', 5)
    num_rounds = config['modes'][mode].get('max_rounds', 50)
    num_nodes = config['convergence']['num_nodes']
    byzantine_ratio = config['convergence']['byzantine_ratio']

    print("=" * 60)
    print("Figure 2: Model Accuracy Convergence")
    print("=" * 60)
    print(f"Nodes: {num_nodes}, Byzantine: {int(byzantine_ratio*100)}%, Rounds: {num_rounds}")

    output_dir = Path(__file__).parent.parent / args.output
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'results').mkdir(parents=True, exist_ok=True)

    results = {
        'experiment': 'figure2_convergence',
        'num_nodes': num_nodes,
        'byzantine_ratio': byzantine_ratio,
        'num_rounds': num_rounds,
        'num_trials': num_trials,
        'data': {}
    }

    methods = ['pogq', 'krum', 'fltrust', 'fedavg']
    for method in tqdm(methods, desc="Methods"):
        results['data'][method] = run_convergence_experiment(
            num_nodes=num_nodes,
            byzantine_ratio=byzantine_ratio,
            num_rounds=num_rounds,
            method=method,
            num_trials=num_trials
        )

    # Print final accuracies
    print("\nFinal Accuracies:")
    for method in methods:
        acc = results['data'][method]['final_accuracy'] * 100
        print(f"  {method:8}: {acc:.1f}%")

    # Save data
    data_file = output_dir / 'results' / 'figure2_data.json'
    with open(data_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nData saved to: {data_file}")

    # Generate figure
    figure_path = output_dir / 'figures' / 'figure2_convergence.png'
    generate_figure(results, figure_path)

    return results


if __name__ == '__main__':
    main()
