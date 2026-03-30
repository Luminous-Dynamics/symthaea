#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Time-To-Detection (TTD) Histogram Plotter
==========================================

Visualizes how quickly each defense detects Byzantine clients.

Usage:
    python scripts/plot_ttd_histograms.py
    python scripts/plot_ttd_histograms.py --defense pogq_v4_1
    python scripts/plot_ttd_histograms.py --format pdf

Output:
    figures/ttd_histogram_*.png - TTD distribution for each defense
    figures/ttd_comparison_*.png - Side-by-side comparison across defenses

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

# Defense colors (same as ROC plots)
DEFENSE_COLORS = {
    'fedavg': '#1f77b4',
    'coord_median': '#ff7f0e',
    'coord_median_safe': '#2ca02c',
    'rfa': '#d62728',
    'fltrust': '#9467bd',
    'boba': '#8c564b',
    'cbf': '#e377c2',
    'pogq_v4_1': '#7f7f7f',
}


def load_detection_metrics(artifact_dir: str) -> Dict:
    """Load detection_metrics.json from artifact directory."""
    metrics_file = os.path.join(artifact_dir, 'detection_metrics.json')

    if not os.path.exists(metrics_file):
        return {}

    with open(metrics_file, 'r') as f:
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


def collect_ttd_data(
    results_dir: str = "results",
    defense_filter: Optional[str] = None,
    attack_filter: Optional[str] = None
) -> Dict[str, Dict[str, List[float]]]:
    """
    Collect time-to-detection data from all artifacts.

    Returns:
        Dictionary mapping defense -> {attack: [ttd_values]}
    """
    artifact_dirs = sorted(glob.glob(f"{results_dir}/artifacts_*/"))

    ttd_data = {}

    for artifact_dir in artifact_dirs:
        meta = extract_experiment_metadata(artifact_dir)

        # Apply filters
        if defense_filter and meta['defense'] != defense_filter:
            continue
        if attack_filter and meta['attack'] != attack_filter:
            continue

        metrics = load_detection_metrics(artifact_dir)

        if not metrics or 'time_to_detection' not in metrics:
            continue

        defense = meta['defense']
        attack = meta['attack']

        if defense not in ttd_data:
            ttd_data[defense] = {}

        if attack not in ttd_data[defense]:
            ttd_data[defense][attack] = []

        # Add TTD values (list of detection rounds for each Byzantine client)
        ttd_values = metrics['time_to_detection']

        # Filter out -1 (not detected)
        ttd_values = [v for v in ttd_values if v > 0]

        if ttd_values:
            ttd_data[defense][attack].extend(ttd_values)

    return ttd_data


def plot_ttd_histogram(
    ttd_values: List[float],
    defense: str,
    attack: str,
    output_path: str,
    max_rounds: int = 20
):
    """
    Plot TTD histogram for a single defense+attack combination.

    Args:
        ttd_values: List of detection round numbers
        defense: Defense name
        attack: Attack name
        output_path: Output file path
        max_rounds: Maximum round number for x-axis
    """
    if not ttd_values:
        print(f"⚠️  No TTD data for {defense} / {attack}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    bins = np.arange(0.5, max_rounds + 1.5, 1)  # Bin edges for rounds 1, 2, 3, ...

    counts, edges, patches = ax.hist(
        ttd_values,
        bins=bins,
        color=DEFENSE_COLORS.get(defense, '#333333'),
        alpha=0.7,
        edgecolor='black',
        linewidth=1.2
    )

    # Color bins by urgency (earlier = better)
    for i, patch in enumerate(patches):
        if i < 3:  # Rounds 1-3: excellent (green)
            patch.set_facecolor('#2ecc71')
        elif i < 6:  # Rounds 4-6: good (yellow)
            patch.set_facecolor('#f39c12')
        elif i < 10:  # Rounds 7-10: moderate (orange)
            patch.set_facecolor('#e67e22')
        else:  # Rounds 11+: slow (red)
            patch.set_facecolor('#e74c3c')

    # Statistics overlay
    mean_ttd = np.mean(ttd_values)
    median_ttd = np.median(ttd_values)

    ax.axvline(mean_ttd, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ttd:.1f}')
    ax.axvline(median_ttd, color='blue', linestyle='--', linewidth=2, label=f'Median: {median_ttd:.1f}')

    # Annotations
    textstr = '\n'.join([
        f'Total Detected: {len(ttd_values)}',
        f'Mean TTD: {mean_ttd:.2f} rounds',
        f'Median TTD: {median_ttd:.2f} rounds',
        f'Std Dev: {np.std(ttd_values):.2f}',
        f'Min: {min(ttd_values):.0f}, Max: {max(ttd_values):.0f}'
    ])

    ax.text(
        0.98, 0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    ax.set_xlabel('Round Number (Time-to-Detection)')
    ax.set_ylabel('Number of Byzantine Clients Detected')
    ax.set_title(f'TTD Distribution: {defense.upper()} / {attack.replace("_", " ").title()}')
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ TTD histogram saved: {output_path}")


def plot_ttd_comparison(
    ttd_data: Dict[str, List[float]],
    attack: str,
    output_path: str
):
    """
    Plot side-by-side comparison of TTD across defenses.

    Args:
        ttd_data: {defense: [ttd_values]}
        attack: Attack name
        output_path: Output file path
    """
    if not ttd_data:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Violin plot
    defenses = sorted(ttd_data.keys())
    data_to_plot = [ttd_data[d] for d in defenses]

    positions = range(len(defenses))
    colors = [DEFENSE_COLORS.get(d, '#333333') for d in defenses]

    parts = ax.violinplot(
        data_to_plot,
        positions=positions,
        widths=0.7,
        showmeans=True,
        showmedians=True
    )

    # Color violins
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    # Overlay box plot for quartiles
    bp = ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=0.3,
        patch_artist=False,
        showfliers=False,
        showcaps=True,
        boxprops=dict(linewidth=1.5, color='black'),
        whiskerprops=dict(linewidth=1.5, color='black'),
        medianprops=dict(linewidth=2, color='red')
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(defenses, rotation=20, ha='right')
    ax.set_ylabel('Time-to-Detection (Rounds)')
    ax.set_title(f'TTD Comparison Across Defenses: {attack.replace("_", " ").title()}')
    ax.grid(True, axis='y', alpha=0.3)

    # Add mean values as text
    for i, defense in enumerate(defenses):
        mean_val = np.mean(ttd_data[defense])
        ax.text(
            i, mean_val,
            f'{mean_val:.1f}',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ TTD comparison saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot time-to-detection histograms')
    parser.add_argument('--defense', type=str, help='Filter by defense')
    parser.add_argument('--attack', type=str, help='Filter by attack')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'pdf', 'svg'])
    parser.add_argument('--output-dir', type=str, default='figures')
    parser.add_argument('--max-rounds', type=int, default=20, help='Max round for histogram x-axis')

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Time-to-Detection Histogram Generator")
    print("=" * 70)

    # Collect data
    print("\n📊 Collecting TTD data from artifacts...")
    ttd_data = collect_ttd_data(
        defense_filter=args.defense,
        attack_filter=args.attack
    )

    if not ttd_data:
        print("❌ No TTD data found. Check that artifacts contain time_to_detection in detection_metrics.json")
        return

    print(f"✅ Found data for {len(ttd_data)} defenses")

    # Generate individual histograms
    for defense, attacks in ttd_data.items():
        for attack, ttd_values in attacks.items():
            print(f"\n📈 Plotting: {defense} / {attack} ({len(ttd_values)} detections)")

            hist_path = os.path.join(
                args.output_dir,
                f"ttd_histogram_{defense}_{attack}.{args.format}"
            )
            plot_ttd_histogram(ttd_values, defense, attack, hist_path, args.max_rounds)

    # Generate comparison plots (one per attack, all defenses)
    print("\n📊 Generating comparison plots...")

    # Reorganize data: attack -> {defense: ttd_values}
    by_attack = {}
    for defense, attacks in ttd_data.items():
        for attack, ttd_values in attacks.items():
            if attack not in by_attack:
                by_attack[attack] = {}
            by_attack[attack][defense] = ttd_values

    for attack, defense_data in by_attack.items():
        if len(defense_data) > 1:  # Only if multiple defenses
            print(f"   Comparing defenses for: {attack}")
            comp_path = os.path.join(
                args.output_dir,
                f"ttd_comparison_{attack}.{args.format}"
            )
            plot_ttd_comparison(defense_data, attack, comp_path)

    print("\n" + "=" * 70)
    print(f"✅ All plots saved to {args.output_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
