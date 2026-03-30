#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
ROC and DET Curve Plotting for Table II Results
================================================

Generates publication-quality ROC and DET curves comparing
detection performance across defenses.

Usage:
    python scripts/plot_roc_det_curves.py
    python scripts/plot_roc_det_curves.py --dataset emnist --attack sign_flip
    python scripts/plot_roc_det_curves.py --format pdf --dpi 300

Output:
    figures/roc_curves_*.png - ROC curves for each dataset+attack combo
    figures/det_curves_*.png - DET curves (log-scale FPR/FNR)

Requirements:
    pip install matplotlib seaborn

Author: Luminous Dynamics
Date: November 8, 2025
"""

import json
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-quality defaults
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Color palette for defenses (colorblind-safe)
DEFENSE_COLORS = {
    'fedavg': '#1f77b4',  # Blue (baseline)
    'coord_median': '#ff7f0e',  # Orange
    'coord_median_safe': '#2ca02c',  # Green
    'rfa': '#d62728',  # Red
    'fltrust': '#9467bd',  # Purple
    'boba': '#8c564b',  # Brown
    'cbf': '#e377c2',  # Pink
    'pogq_v4_1': '#7f7f7f',  # Gray
}

# Line styles
DEFENSE_LINESTYLE = {
    'fedavg': '--',  # Baseline dashed
    'coord_median': '-',
    'coord_median_safe': '-',
    'rfa': '-',
    'fltrust': '-',
    'boba': '-',
    'cbf': '-',
    'pogq_v4_1': '-',
}


def load_detection_metrics(artifact_dir: str) -> Dict:
    """Load detection_metrics.json from artifact directory."""
    metrics_file = os.path.join(artifact_dir, 'detection_metrics.json')

    if not os.path.exists(metrics_file):
        return {}

    with open(metrics_file, 'r') as f:
        return json.load(f)


def extract_experiment_metadata(artifact_dir: str) -> Dict:
    """Extract dataset, attack, defense from artifact directory name."""
    # Example: results/artifacts_20251108_160000_emnist_sign_flip_pogq_v4_1_seed42
    basename = os.path.basename(artifact_dir.rstrip('/'))
    parts = basename.split('_')

    # Find where timestamp ends (8 digits + 6 digits)
    timestamp_end = 3  # artifacts_YYYYMMDD_HHMMSS

    # Remaining parts: dataset_attack_defense_seedN
    remaining = '_'.join(parts[timestamp_end:])

    # Split by known seeds pattern
    if '_seed' in remaining:
        config_part, seed_part = remaining.rsplit('_seed', 1)
        seed = int(seed_part)
    else:
        config_part = remaining
        seed = 0

    # Now split config_part into dataset, attack, defense
    # This is tricky because names can have underscores
    # Heuristic: dataset is first part, defense is last part, attack is middle
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


def collect_roc_data(
    results_dir: str = "results",
    dataset_filter: Optional[str] = None,
    attack_filter: Optional[str] = None
) -> Dict[Tuple[str, str], Dict[str, List[Dict]]]:
    """
    Collect ROC curve data from all artifacts.

    Returns:
        Dictionary mapping (dataset, attack) -> {defense: [points]}
        where each point has {'fpr': float, 'tpr': float, 'threshold': float}
    """
    artifact_dirs = sorted(glob.glob(f"{results_dir}/artifacts_*/"))

    roc_data = {}

    for artifact_dir in artifact_dirs:
        meta = extract_experiment_metadata(artifact_dir)

        # Apply filters
        if dataset_filter and meta['dataset'] != dataset_filter:
            continue
        if attack_filter and meta['attack'] != attack_filter:
            continue

        metrics = load_detection_metrics(artifact_dir)

        if not metrics or 'roc_curve' not in metrics:
            continue

        key = (meta['dataset'], meta['attack'])
        defense = meta['defense']

        if key not in roc_data:
            roc_data[key] = {}

        if defense not in roc_data[key]:
            roc_data[key][defense] = []

        # Add this experiment's ROC curve
        roc_data[key][defense].append({
            'fpr': np.array(metrics['roc_curve']['fpr']),
            'tpr': np.array(metrics['roc_curve']['tpr']),
            'thresholds': np.array(metrics['roc_curve']['thresholds']),
            'auroc': metrics.get('auroc', 0.0),
            'seed': meta['seed']
        })

    return roc_data


def plot_roc_curve(
    roc_data: Dict[str, List[Dict]],
    dataset: str,
    attack: str,
    output_path: str,
    show_legend: bool = True
):
    """
    Plot ROC curves for a single dataset+attack combination.

    Args:
        roc_data: {defense: [curve_data]}
        dataset: Dataset name
        attack: Attack name
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')

    # Plot each defense
    for defense in sorted(roc_data.keys()):
        curves = roc_data[defense]

        if not curves:
            continue

        # Average across seeds
        if len(curves) > 1:
            # Interpolate to common FPR points
            common_fpr = np.linspace(0, 1, 100)
            tprs = []

            for curve in curves:
                tpr_interp = np.interp(common_fpr, curve['fpr'], curve['tpr'])
                tprs.append(tpr_interp)

            mean_tpr = np.mean(tprs, axis=0)
            std_tpr = np.std(tprs, axis=0)

            # Mean AUROC
            mean_auroc = np.mean([c['auroc'] for c in curves])

            # Plot mean curve
            ax.plot(
                common_fpr,
                mean_tpr,
                color=DEFENSE_COLORS.get(defense, '#000000'),
                linestyle=DEFENSE_LINESTYLE.get(defense, '-'),
                linewidth=2,
                label=f'{defense} (AUROC={mean_auroc:.3f})'
            )

            # Confidence interval
            ax.fill_between(
                common_fpr,
                np.maximum(mean_tpr - std_tpr, 0),
                np.minimum(mean_tpr + std_tpr, 1),
                color=DEFENSE_COLORS.get(defense, '#000000'),
                alpha=0.1
            )
        else:
            # Single seed
            curve = curves[0]
            ax.plot(
                curve['fpr'],
                curve['tpr'],
                color=DEFENSE_COLORS.get(defense, '#000000'),
                linestyle=DEFENSE_LINESTYLE.get(defense, '-'),
                linewidth=2,
                label=f'{defense} (AUROC={curve["auroc"]:.3f})'
            )

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve: {dataset.upper()} / {attack.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    if show_legend:
        ax.legend(loc='lower right', frameon=True, fancybox=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ ROC curve saved: {output_path}")


def plot_det_curve(
    roc_data: Dict[str, List[Dict]],
    dataset: str,
    attack: str,
    output_path: str,
    show_legend: bool = True
):
    """
    Plot DET (Detection Error Tradeoff) curve.

    DET uses log-scale for both FPR and FNR (False Negative Rate = 1 - TPR).
    This emphasizes low-error regions which are critical for security.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot each defense
    for defense in sorted(roc_data.keys()):
        curves = roc_data[defense]

        if not curves:
            continue

        # Average across seeds (same as ROC)
        if len(curves) > 1:
            common_fpr = np.logspace(-4, 0, 100)  # Log-spaced points
            fnrs = []

            for curve in curves:
                fnr = 1 - curve['tpr']
                fnr_interp = np.interp(common_fpr, curve['fpr'], fnr)
                fnrs.append(fnr_interp)

            mean_fnr = np.mean(fnrs, axis=0)

            ax.plot(
                common_fpr,
                mean_fnr,
                color=DEFENSE_COLORS.get(defense, '#000000'),
                linestyle=DEFENSE_LINESTYLE.get(defense, '-'),
                linewidth=2,
                label=defense
            )
        else:
            curve = curves[0]
            fnr = 1 - curve['tpr']

            ax.plot(
                curve['fpr'],
                fnr,
                color=DEFENSE_COLORS.get(defense, '#000000'),
                linestyle=DEFENSE_LINESTYLE.get(defense, '-'),
                linewidth=2,
                label=defense
            )

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('False Negative Rate')
    ax.set_title(f'DET Curve: {dataset.upper()} / {attack.replace("_", " ").title()}')
    ax.grid(True, alpha=0.3, which='both')

    if show_legend:
        ax.legend(loc='upper right', frameon=True, fancybox=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ DET curve saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot ROC and DET curves')
    parser.add_argument('--dataset', type=str, help='Filter by dataset')
    parser.add_argument('--attack', type=str, help='Filter by attack')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'pdf', 'svg'])
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--output-dir', type=str, default='figures')

    args = parser.parse_args()

    # Update DPI
    plt.rcParams['savefig.dpi'] = args.dpi

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ROC/DET Curve Generator")
    print("=" * 70)

    # Collect data
    print("\n📊 Collecting ROC data from artifacts...")
    roc_data = collect_roc_data(
        dataset_filter=args.dataset,
        attack_filter=args.attack
    )

    if not roc_data:
        print("❌ No ROC data found. Check that artifacts contain detection_metrics.json")
        return

    print(f"✅ Found data for {len(roc_data)} dataset+attack combinations")

    # Generate plots for each combination
    for (dataset, attack), defenses in roc_data.items():
        print(f"\n📈 Plotting: {dataset} / {attack}")
        print(f"   Defenses: {list(defenses.keys())}")

        # ROC curve
        roc_path = os.path.join(
            args.output_dir,
            f"roc_{dataset}_{attack}.{args.format}"
        )
        plot_roc_curve(defenses, dataset, attack, roc_path)

        # DET curve
        det_path = os.path.join(
            args.output_dir,
            f"det_{dataset}_{attack}.{args.format}"
        )
        plot_det_curve(defenses, dataset, attack, det_path)

    print("\n" + "=" * 70)
    print(f"✅ All plots saved to {args.output_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
