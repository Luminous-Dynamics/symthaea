#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Figure 1: Detection Rate vs Byzantine Ratio

Generates Figure 1 from the Mycelix paper, showing detection rate
across different Byzantine ratios for all methods.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import experiment functions from table1
from table1_byzantine_detection import (
    FedAvgAggregator, KrumAggregator, FLTrustAggregator, PoGQAggregator,
    run_detection_experiment, load_config
)


def generate_figure(results: Dict, output_path: Path):
    """Generate the detection rate vs Byzantine ratio figure."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")

        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract data
        ratios = []
        method_data = {'FedAvg': [], 'Krum': [], 'FLTrust': [], 'PoGQ': []}
        method_err = {'FedAvg': [], 'Krum': [], 'FLTrust': [], 'PoGQ': []}

        for ratio_key in sorted(results['data'].keys(), key=lambda x: int(x.replace('%', ''))):
            ratio_val = int(ratio_key.replace('%', ''))
            ratios.append(ratio_val)

            name_map = {'fedavg': 'FedAvg', 'krum': 'Krum', 'fltrust': 'FLTrust', 'pogq': 'PoGQ'}
            for method_key, method_name in name_map.items():
                data = results['data'][ratio_key][method_key]
                method_data[method_name].append(data['detection_rate_mean'] * 100)
                method_err[method_name].append(data['detection_rate_std'] * 100)

        # Plot lines with error bars
        markers = {'FedAvg': 'x', 'Krum': 's', 'FLTrust': '^', 'PoGQ': 'o'}
        colors = {'FedAvg': '#d62728', 'Krum': '#2ca02c', 'FLTrust': '#1f77b4', 'PoGQ': '#ff7f0e'}

        for method in ['PoGQ', 'FLTrust', 'Krum', 'FedAvg']:
            ax.errorbar(ratios, method_data[method],
                       yerr=method_err[method],
                       marker=markers[method],
                       color=colors[method],
                       linewidth=2,
                       markersize=8,
                       capsize=4,
                       label=method)

        ax.set_xlabel('Byzantine Ratio (%)', fontsize=12)
        ax.set_ylabel('Detection Rate (%)', fontsize=12)
        ax.set_title('Byzantine Detection Rate vs Adversarial Ratio', fontsize=14)
        ax.legend(loc='lower left', fontsize=11)
        ax.set_xlim([5, 50])
        ax.set_ylim([0, 105])
        ax.set_xticks([10, 20, 33, 45])
        ax.grid(True, alpha=0.3)

        # Add classical BFT limit line
        ax.axvline(x=33, color='gray', linestyle='--', alpha=0.5)
        ax.text(34, 50, 'Classical BFT\nLimit (33%)', fontsize=9, color='gray')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Figure saved to: {output_path}")
        return True

    except ImportError as e:
        print(f"Warning: Could not generate figure. Missing dependency: {e}")
        print("Install matplotlib and seaborn: pip install matplotlib seaborn")
        return False


def main():
    parser = argparse.ArgumentParser(description='Figure 1: Detection Rate vs Byzantine Ratio')
    parser.add_argument('--config', default='config.yaml', help='Config file path')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick mode')
    parser.add_argument('--from-json', type=str, help='Generate from existing JSON results')
    args = parser.parse_args()

    output_dir = Path(__file__).parent.parent / args.output
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'results').mkdir(parents=True, exist_ok=True)

    if args.from_json:
        # Load existing results
        with open(args.from_json, 'r') as f:
            results = json.load(f)
    else:
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

        np.random.seed(config['global']['random_seed'])

        mode = 'quick' if args.quick else 'full'
        num_trials = config['modes'][mode]['num_trials']
        gradient_dim = config['modes'][mode]['gradient_dim']
        num_nodes = config['byzantine_detection']['num_nodes']
        byzantine_ratios = config['byzantine_detection']['byzantine_ratios']

        print("=" * 60)
        print("Figure 1: Detection Rate vs Byzantine Ratio")
        print("=" * 60)
        print(f"Running experiments (nodes={num_nodes}, dim={gradient_dim}, trials={num_trials})")

        aggregators = {
            'fedavg': FedAvgAggregator(),
            'krum': KrumAggregator(num_byzantine=int(num_nodes * 0.45)),
            'fltrust': FLTrustAggregator(),
            'pogq': PoGQAggregator(z_threshold=config['pogq']['z_score_threshold'])
        }

        results = {
            'experiment': 'figure1_detection_vs_ratio',
            'num_nodes': num_nodes,
            'gradient_dim': gradient_dim,
            'num_trials': num_trials,
            'data': {}
        }

        from tqdm import tqdm
        for ratio in tqdm(byzantine_ratios, desc="Byzantine Ratios"):
            ratio_key = f"{int(ratio * 100)}%"
            results['data'][ratio_key] = {}

            for name, aggregator in aggregators.items():
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

        # Save data
        data_file = output_dir / 'results' / 'figure1_data.json'
        with open(data_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nData saved to: {data_file}")

    # Generate figure
    figure_path = output_dir / 'figures' / 'figure1_detection_vs_ratio.png'
    generate_figure(results, figure_path)

    return results


if __name__ == '__main__':
    main()
