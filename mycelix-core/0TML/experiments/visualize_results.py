#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Generate publication-quality visualizations from experiment results.

Creates:
- Convergence plots (accuracy over rounds)
- Comparison bar charts (final accuracies)
- Heterogeneity impact analysis
- Byzantine attack effectiveness heatmaps
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Use publication-quality settings
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


def extract_from_log(log_file: Path) -> Dict:
    """Extract experiment results from log file."""
    with open(log_file) as f:
        content = f.read()

    results = {'baselines': []}

    # Extract experiment metadata - be more flexible with filename matching
    filename_lower = str(log_file).lower()
    if '0.1' in filename_lower and 'dirichlet' in filename_lower or '0.1.log' in filename_lower:
        results['name'] = 'Dirichlet α=0.1'
        results['heterogeneity'] = 'high'
    elif '0.5' in filename_lower and 'dirichlet' in filename_lower or '0.5.log' in filename_lower:
        results['name'] = 'Dirichlet α=0.5'
        results['heterogeneity'] = 'moderate'
    elif 'pathological' in filename_lower:
        results['name'] = 'Pathological'
        results['heterogeneity'] = 'extreme'
    else:
        # Fallback - use filename as name
        results['name'] = log_file.stem
        results['heterogeneity'] = 'unknown'

    # Extract per-baseline results
    for baseline in ['FEDAVG', 'FEDPROX', 'SCAFFOLD', 'KRUM', 'MULTIKRUM', 'MEDIAN', 'BULYAN']:
        # Find all rounds for this baseline
        pattern = rf"Round\s+(\d+):\s+Train Loss=([\d.]+),\s+Train Acc=([\d.]+),\s+Test Loss=([\d.]+),\s+Test Acc=([\d.]+)"

        # Find baseline section
        baseline_start = content.find(f"Baseline: {baseline}")
        if baseline_start == -1:
            continue

        baseline_end = content.find("======", baseline_start + 100)
        baseline_section = content[baseline_start:baseline_end]

        rounds = []
        train_losses = []
        train_accs = []
        test_losses = []
        test_accs = []

        for match in re.finditer(pattern, baseline_section):
            rounds.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))
            train_accs.append(float(match.group(3)))
            test_losses.append(float(match.group(4)))
            test_accs.append(float(match.group(5)))

        if rounds:
            results['baselines'].append({
                'name': baseline.lower(),
                'rounds': rounds,
                'train_loss': train_losses,
                'train_acc': train_accs,
                'test_loss': test_losses,
                'test_acc': test_accs
            })

    return results


def plot_convergence(results: List[Dict], output_dir: Path):
    """Plot test accuracy convergence over rounds."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = {
        'fedavg': '#1f77b4',
        'fedprox': '#ff7f0e',
        'scaffold': '#2ca02c',
        'krum': '#d62728',
        'multikrum': '#9467bd',
        'median': '#8c564b',
        'bulyan': '#e377c2'
    }

    for idx, result in enumerate(results):
        ax = axes[idx]
        ax.set_title(result['name'], fontweight='bold')
        ax.set_xlabel('Round')
        ax.set_ylabel('Test Accuracy')
        ax.grid(True, alpha=0.3)

        for baseline in result['baselines']:
            name = baseline['name']
            if name in colors:
                ax.plot(baseline['rounds'], baseline['test_acc'],
                       label=name.upper(), color=colors[name], linewidth=2)

        ax.legend(loc='lower right')
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / 'convergence_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'convergence_comparison.pdf', bbox_inches='tight')
    print(f"✅ Saved: convergence_comparison.png/.pdf")
    plt.close()


def plot_final_accuracy_bars(results: List[Dict], output_dir: Path):
    """Bar chart of final test accuracies."""
    fig, ax = plt.subplots(figsize=(12, 6))

    baselines = ['fedavg', 'fedprox', 'scaffold', 'krum', 'multikrum', 'median']
    exp_names = [r['name'] for r in results]

    x = np.arange(len(baselines))
    width = 0.25

    colors_exp = ['#3498db', '#e74c3c', '#2ecc71']

    for i, result in enumerate(results):
        accs = []
        for baseline_name in baselines:
            found = False
            for b in result['baselines']:
                if b['name'] == baseline_name:
                    accs.append(b['test_acc'][-1])
                    found = True
                    break
            if not found:
                accs.append(0)

        ax.bar(x + i * width, accs, width, label=result['name'], color=colors_exp[i])

    ax.set_xlabel('Aggregation Method', fontweight='bold')
    ax.set_ylabel('Final Test Accuracy', fontweight='bold')
    ax.set_title('Impact of Data Heterogeneity on Federated Learning Algorithms', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels([b.upper() for b in baselines], rotation=15, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_dir / 'final_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'final_accuracy_comparison.pdf', bbox_inches='tight')
    print(f"✅ Saved: final_accuracy_comparison.png/.pdf")
    plt.close()


def plot_heterogeneity_impact(results: List[Dict], output_dir: Path):
    """Line plot showing how each algorithm degrades with heterogeneity."""
    fig, ax = plt.subplots(figsize=(10, 6))

    baselines = ['fedavg', 'fedprox', 'scaffold', 'multikrum']
    colors = {'fedavg': '#1f77b4', 'fedprox': '#ff7f0e', 'scaffold': '#2ca02c', 'multikrum': '#9467bd'}
    markers = {'fedavg': 'o', 'fedprox': 's', 'scaffold': '^', 'multikrum': 'D'}

    # Order: moderate -> high -> extreme
    ordered_results = sorted(results, key=lambda r:
        {'moderate': 0, 'high': 1, 'extreme': 2}[r['heterogeneity']])

    heterogeneity_labels = [r['name'] for r in ordered_results]

    for baseline_name in baselines:
        accs = []
        for result in ordered_results:
            for b in result['baselines']:
                if b['name'] == baseline_name:
                    accs.append(b['test_acc'][-1])
                    break

        if len(accs) == 3:
            ax.plot(heterogeneity_labels, accs,
                   label=baseline_name.upper(),
                   color=colors[baseline_name],
                   marker=markers[baseline_name],
                   linewidth=2, markersize=8)

    ax.set_xlabel('Data Heterogeneity Level', fontweight='bold')
    ax.set_ylabel('Final Test Accuracy', fontweight='bold')
    ax.set_title('Algorithm Robustness to Data Heterogeneity', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.2, 1.05])

    # Add annotation for key insight
    ax.annotate('SCAFFOLD maintains\nhigh accuracy',
                xy=(2, 0.96), xytext=(1.5, 0.85),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=10, color='green', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'heterogeneity_impact.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'heterogeneity_impact.pdf', bbox_inches='tight')
    print(f"✅ Saved: heterogeneity_impact.png/.pdf")
    plt.close()


def generate_latex_table(results: List[Dict], output_dir: Path):
    """Generate LaTeX table for publication."""
    baselines = ['fedavg', 'fedprox', 'scaffold', 'krum', 'multikrum', 'median']

    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{Final test accuracy under different data heterogeneity levels}")
    latex.append("\\label{tab:non_iid_results}")
    latex.append("\\begin{tabular}{l|ccc}")
    latex.append("\\hline")
    latex.append("Algorithm & $\\alpha=0.5$ & $\\alpha=0.1$ & Pathological \\\\")
    latex.append("\\hline")

    for baseline_name in baselines:
        row = [baseline_name.upper()]
        for result in results:
            for b in result['baselines']:
                if b['name'] == baseline_name:
                    acc = b['test_acc'][-1]
                    row.append(f"{acc:.4f}")
                    break
        latex.append(" & ".join(row) + " \\\\")

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    latex_content = "\n".join(latex)

    with open(output_dir / 'results_table.tex', 'w') as f:
        f.write(latex_content)

    print(f"✅ Saved: results_table.tex")
    print("\nLaTeX Table Preview:")
    print(latex_content)


def main():
    parser = argparse.ArgumentParser(description='Visualize ZeroTrustML experiment results')
    parser.add_argument('--log-dir', type=str, default='/tmp',
                       help='Directory containing log files')
    parser.add_argument('--output-dir', type=str, default='results/figures',
                       help='Output directory for figures')
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ZeroTrustML Experiment Results Visualization")
    print("=" * 80)
    print()

    # Find log files
    log_files = {
        'dirichlet_0.5': log_dir / 'mnist_non_iid_0.5.log',
        'dirichlet_0.1': log_dir / 'mnist_non_iid_0.1.log',
        'pathological': log_dir / 'mnist_non_iid_pathological.log'
    }

    results = []
    for name, log_file in log_files.items():
        if log_file.exists():
            print(f"📊 Processing: {log_file.name}")
            result = extract_from_log(log_file)
            results.append(result)
        else:
            print(f"⚠️  Missing: {log_file.name}")

    if not results:
        print("❌ No log files found!")
        return

    print(f"\n✅ Loaded {len(results)} experiments\n")

    # Generate visualizations
    print("Generating visualizations...")
    print()

    plot_convergence(results, output_dir)
    plot_final_accuracy_bars(results, output_dir)
    plot_heterogeneity_impact(results, output_dir)
    generate_latex_table(results, output_dir)

    print()
    print("=" * 80)
    print("All visualizations complete!")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 80)


if __name__ == '__main__':
    main()
