#!/usr/bin/env python3
"""
Coordinate-Median Safe Guard Activation Heatmaps
=================================================

Visualizes when and why CM-Safe guards trigger across experiments.

Usage:
    python scripts/plot_guard_activations.py
    python scripts/plot_guard_activations.py --attack sign_flip
    python scripts/plot_guard_activations.py --format pdf

Output:
    figures/guard_heatmap_*.png - Heatmap of guard activations by round
    figures/guard_summary_*.png - Bar chart of guard activation frequency

Author: Luminous Dynamics
Date: November 8, 2025
"""

import json
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Publication-quality defaults
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'

# Guard names
GUARD_NAMES = [
    'min_clients',
    'norm_clamp',
    'direction_check',
    'reject_all'
]

GUARD_COLORS = {
    'min_clients': '#e74c3c',      # Red - critical
    'norm_clamp': '#f39c12',       # Orange - moderate
    'direction_check': '#3498db',  # Blue - informational
    'reject_all': '#95a5a6'        # Gray - fallback
}


def load_cm_safe_diagnostics(artifact_dir: str) -> Dict:
    """Load coord_median_diagnostics.json from artifact directory."""
    diag_file = os.path.join(artifact_dir, 'coord_median_diagnostics.json')

    if not os.path.exists(diag_file):
        return {}

    with open(diag_file, 'r') as f:
        return json.load(f)


def extract_experiment_metadata(artifact_dir: str) -> Dict:
    """Extract dataset, attack, defense, seed from directory name."""
    basename = os.path.basename(artifact_dir.rstrip('/'))
    parts = basename.split('_')

    timestamp_end = 3  # artifacts_YYYYMMDD_HHMMSS
    remaining = '_'.join(parts[timestamp_end:])

    if '_seed' in remaining:
        config_part, seed_part = remaining.rsplit('_seed', 1)
        seed = int(seed_part)
    else:
        config_part = remaining
        seed = 0

    config_parts = config_part.split('_')
    dataset = config_parts[0]
    defense = config_parts[-1]
    attack = '_'.join(config_parts[1:-1])

    return {
        'dataset': dataset,
        'attack': attack,
        'defense': defense,
        'seed': seed
    }


def collect_guard_data(
    results_dir: str = "results",
    attack_filter: Optional[str] = None
) -> Dict[str, List[Dict]]:
    """
    Collect guard activation data from all CM-Safe experiments.

    Returns:
        Dictionary mapping attack -> [experiment_data]
    """
    artifact_dirs = sorted(glob.glob(f"{results_dir}/artifacts_*/"))

    guard_data = {}

    for artifact_dir in artifact_dirs:
        meta = extract_experiment_metadata(artifact_dir)

        # Only process coord_median_safe
        if meta['defense'] != 'coord_median_safe':
            continue

        # Apply filter
        if attack_filter and meta['attack'] != attack_filter:
            continue

        diagnostics = load_cm_safe_diagnostics(artifact_dir)

        if not diagnostics:
            continue

        attack = meta['attack']

        if attack not in guard_data:
            guard_data[attack] = []

        # Extract guard activations by round
        guard_data[attack].append({
            'dataset': meta['dataset'],
            'seed': meta['seed'],
            'diagnostics': diagnostics
        })

    return guard_data


def plot_guard_heatmap(
    experiments: List[Dict],
    attack: str,
    output_path: str
):
    """
    Plot heatmap of guard activations across rounds.

    Args:
        experiments: List of experiment data for this attack
        attack: Attack name
        output_path: Output file path
    """
    if not experiments:
        print(f"⚠️  No data for attack: {attack}")
        return

    # Determine max rounds across all experiments
    max_rounds = 0
    for exp in experiments:
        diag = exp['diagnostics']
        if 'experiments' in diag and len(diag['experiments']) > 0:
            for exp_record in diag['experiments']:
                if 'guard_activations' in exp_record:
                    max_rounds = max(max_rounds, len(exp_record.get('per_round_guards', [])))

    if max_rounds == 0:
        print(f"⚠️  No round data for attack: {attack}")
        return

    # Create matrix: (num_experiments, num_rounds, num_guards)
    num_experiments = len(experiments)
    num_guards = len(GUARD_NAMES)

    activation_matrix = np.zeros((num_experiments, max_rounds, num_guards))

    for i, exp in enumerate(experiments):
        diag = exp['diagnostics']

        if 'experiments' not in diag or len(diag['experiments']) == 0:
            continue

        for exp_record in diag['experiments']:
            per_round_guards = exp_record.get('per_round_guards', [])

            for round_idx, round_guards in enumerate(per_round_guards):
                if round_idx >= max_rounds:
                    break

                for guard_idx, guard_name in enumerate(GUARD_NAMES):
                    if round_guards.get(guard_name, 0) > 0:
                        activation_matrix[i, round_idx, guard_idx] = 1

    # Plot each guard type as a separate heatmap
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for guard_idx, guard_name in enumerate(GUARD_NAMES):
        ax = axes[guard_idx]

        # Extract this guard's data
        guard_matrix = activation_matrix[:, :, guard_idx]

        # Plot heatmap
        sns.heatmap(
            guard_matrix,
            ax=ax,
            cmap='YlOrRd',
            cbar_kws={'label': 'Activated'},
            vmin=0,
            vmax=1,
            linewidths=0.5,
            linecolor='white'
        )

        ax.set_title(f'{guard_name.replace("_", " ").title()} Guard')
        ax.set_xlabel('Round')
        ax.set_ylabel('Experiment')

    plt.suptitle(f'CM-Safe Guard Activations: {attack.replace("_", " ").title()}', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Guard heatmap saved: {output_path}")


def plot_guard_summary(
    experiments: List[Dict],
    attack: str,
    output_path: str
):
    """
    Plot summary bar chart of total guard activations.

    Args:
        experiments: List of experiment data for this attack
        attack: Attack name
        output_path: Output file path
    """
    if not experiments:
        return

    # Count total activations per guard
    guard_counts = {guard: 0 for guard in GUARD_NAMES}

    for exp in experiments:
        diag = exp['diagnostics']

        if 'experiments' not in diag or len(diag['experiments']) == 0:
            continue

        for exp_record in diag['experiments']:
            activations = exp_record.get('guard_activations', {})

            for guard in GUARD_NAMES:
                guard_counts[guard] += activations.get(guard, 0)

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(8, 5))

    guards = list(guard_counts.keys())
    counts = [guard_counts[g] for g in guards]
    colors = [GUARD_COLORS[g] for g in guards]

    bars = ax.bar(
        range(len(guards)),
        counts,
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5
    )

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.02,
                str(count),
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )

    ax.set_xticks(range(len(guards)))
    ax.set_xticklabels([g.replace('_', ' ').title() for g in guards], rotation=15, ha='right')
    ax.set_ylabel('Total Activations')
    ax.set_title(f'CM-Safe Guard Activation Summary: {attack.replace("_", " ").title()}')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Guard summary saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot CM-Safe guard activations')
    parser.add_argument('--attack', type=str, help='Filter by attack type')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'pdf', 'svg'])
    parser.add_argument('--output-dir', type=str, default='figures')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CM-Safe Guard Activation Visualizer")
    print("=" * 70)

    # Collect data
    print("\n📊 Collecting guard activation data...")
    guard_data = collect_guard_data(attack_filter=args.attack)

    if not guard_data:
        print("❌ No guard data found. Check that coord_median_safe experiments have completed.")
        return

    print(f"✅ Found data for {len(guard_data)} attack types")

    # Generate plots for each attack
    for attack, experiments in guard_data.items():
        print(f"\n📈 Plotting: {attack} ({len(experiments)} experiments)")

        # Heatmap
        heatmap_path = os.path.join(
            args.output_dir,
            f"guard_heatmap_{attack}.{args.format}"
        )
        plot_guard_heatmap(experiments, attack, heatmap_path)

        # Summary bar chart
        summary_path = os.path.join(
            args.output_dir,
            f"guard_summary_{attack}.{args.format}"
        )
        plot_guard_summary(experiments, attack, summary_path)

    print("\n" + "=" * 70)
    print(f"✅ All plots saved to {args.output_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
