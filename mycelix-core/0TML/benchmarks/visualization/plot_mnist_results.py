# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Generate Publication-Ready Figures from MNIST Experiment Results

Creates IEEE-format figures for academic paper:
- Figure 1: Accuracy vs Communication Rounds
- Figure 2: Convergence Speed Comparison
- Table 1: Performance Metrics Summary
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Set publication-quality defaults
matplotlib.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def load_latest_results(results_dir: Path, pattern: str) -> Dict:
    """Load the most recent results matching pattern"""
    files = sorted(results_dir.glob(pattern), key=lambda x: x.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No results found matching: {pattern}")

    latest = files[-1]
    print(f"Loading: {latest.name}")

    with open(latest, 'r') as f:
        return json.load(f)


def plot_accuracy_comparison(
    fedavg_results: Dict,
    zerotrustml_results: Dict,
    output_path: Path
):
    """
    Figure 1: Accuracy vs Communication Rounds

    IEEE two-column format: 3.5 inches wide
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # Plot FedAvg
    ax.plot(
        fedavg_results['rounds'],
        fedavg_results['accuracies'],
        'o-',
        label='FedAvg (Baseline)',
        linewidth=1.5,
        markersize=4,
        color='#1f77b4',
        markevery=5
    )

    # Plot ZeroTrustML (Krum)
    ax.plot(
        zerotrustml_results['rounds'],
        zerotrustml_results['accuracies'],
        's-',
        label='ZeroTrustML (Krum)',
        linewidth=1.5,
        markersize=4,
        color='#ff7f0e',
        markevery=5
    )

    # Formatting
    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim([0.85, 1.0])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', framealpha=0.9)

    # Add 90% accuracy reference line
    ax.axhline(y=0.90, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(45, 0.905, '90%', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(output_path / 'mnist_accuracy_comparison.pdf')
    plt.savefig(output_path / 'mnist_accuracy_comparison.png')
    print(f"✅ Saved: {output_path / 'mnist_accuracy_comparison.pdf'}")


def plot_convergence_analysis(
    fedavg_results: Dict,
    zerotrustml_results: Dict,
    output_path: Path
):
    """
    Figure 2: Convergence Speed Analysis

    Bar chart showing rounds to reach accuracy thresholds
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    thresholds = [0.90, 0.95, 0.97, 0.98]
    fedavg_rounds = []
    zerotrustml_rounds = []

    for threshold in thresholds:
        # FedAvg rounds to threshold
        fedavg_round = next(
            (i+1 for i, acc in enumerate(fedavg_results['accuracies']) if acc >= threshold),
            len(fedavg_results['rounds'])
        )
        fedavg_rounds.append(fedavg_round)

        # ZeroTrustML rounds to threshold
        zerotrustml_round = next(
            (i+1 for i, acc in enumerate(zerotrustml_results['accuracies']) if acc >= threshold),
            len(zerotrustml_results['rounds'])
        )
        zerotrustml_rounds.append(zerotrustml_round)

    x = np.arange(len(thresholds))
    width = 0.35

    ax.bar(x - width/2, fedavg_rounds, width, label='FedAvg', color='#1f77b4')
    ax.bar(x + width/2, zerotrustml_rounds, width, label='ZeroTrustML', color='#ff7f0e')

    ax.set_xlabel('Target Accuracy')
    ax.set_ylabel('Rounds to Converge')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{t*100:.0f}%' for t in thresholds])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path / 'mnist_convergence_comparison.pdf')
    plt.savefig(output_path / 'mnist_convergence_comparison.png')
    print(f"✅ Saved: {output_path / 'mnist_convergence_comparison.pdf'}")


def generate_metrics_table(
    fedavg_results: Dict,
    zerotrustml_results: Dict,
    output_path: Path
):
    """
    Table 1: Performance Metrics Summary

    LaTeX table for paper
    """
    # Calculate metrics
    fedavg_final = fedavg_results['accuracies'][-1]
    zerotrustml_final = zerotrustml_results['accuracies'][-1]

    fedavg_avg_time = np.mean(fedavg_results['round_times'])
    zerotrustml_avg_time = np.mean(zerotrustml_results['round_times'])

    # Rounds to 90% accuracy
    fedavg_conv = next(
        (i+1 for i, acc in enumerate(fedavg_results['accuracies']) if acc >= 0.90),
        len(fedavg_results['rounds'])
    )
    zerotrustml_conv = next(
        (i+1 for i, acc in enumerate(zerotrustml_results['accuracies']) if acc >= 0.90),
        len(zerotrustml_results['rounds'])
    )

    # Generate LaTeX table
    latex_table = f"""
\\begin{{table}}[t]
\\centering
\\caption{{MNIST Classification Results: FedAvg vs ZeroTrustML (Krum)}}
\\label{{tab:mnist_results}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Metric}} & \\textbf{{FedAvg}} & \\textbf{{ZeroTrustML (Krum)}} \\\\
\\midrule
Final Accuracy (\\%) & {fedavg_final*100:.2f} & {zerotrustml_final*100:.2f} \\\\
Rounds to 90\\% & {fedavg_conv} & {zerotrustml_conv} \\\\
Avg. Round Time (s) & {fedavg_avg_time:.2f} & {zerotrustml_avg_time:.2f} \\\\
Communication Rounds & 50 & 50 \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""

    # Save LaTeX table
    with open(output_path / 'mnist_results_table.tex', 'w') as f:
        f.write(latex_table)

    # Also create markdown version for README
    md_table = f"""
| Metric | FedAvg | ZeroTrustML (Krum) |
|--------|--------|----------------|
| Final Accuracy | {fedavg_final*100:.2f}% | {zerotrustml_final*100:.2f}% |
| Rounds to 90% | {fedavg_conv} | {zerotrustml_conv} |
| Avg Round Time | {fedavg_avg_time:.2f}s | {zerotrustml_avg_time:.2f}s |
| Total Rounds | 50 | 50 |
"""

    with open(output_path / 'mnist_results_table.md', 'w') as f:
        f.write(md_table)

    print(f"✅ Saved: {output_path / 'mnist_results_table.tex'}")
    print(f"✅ Saved: {output_path / 'mnist_results_table.md'}")


def plot_loss_curves(
    fedavg_results: Dict,
    zerotrustml_results: Dict,
    output_path: Path
):
    """
    Supplementary Figure: Training Loss Curves
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    ax.plot(
        fedavg_results['rounds'],
        fedavg_results['losses'],
        'o-',
        label='FedAvg',
        linewidth=1.5,
        markersize=4,
        color='#1f77b4',
        markevery=5
    )

    ax.plot(
        zerotrustml_results['rounds'],
        zerotrustml_results['losses'],
        's-',
        label='ZeroTrustML (Krum)',
        linewidth=1.5,
        markersize=4,
        color='#ff7f0e',
        markevery=5
    )

    ax.set_xlabel('Communication Rounds')
    ax.set_ylabel('Test Loss')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_path / 'mnist_loss_curves.pdf')
    plt.savefig(output_path / 'mnist_loss_curves.png')
    print(f"✅ Saved: {output_path / 'mnist_loss_curves.pdf'}")


def main():
    """Generate all figures for MNIST experiment"""
    print("=" * 70)
    print("Generating Publication-Ready Figures for MNIST Experiment")
    print("=" * 70)

    # Paths
    results_dir = Path("../results")
    output_dir = Path("../figures")
    output_dir.mkdir(exist_ok=True)

    # Load results
    print("\n📥 Loading experiment results...")
    try:
        fedavg_results = load_latest_results(
            results_dir,
            "mnist_accuracy_FedAvg_*.json"
        )
        zerotrustml_results = load_latest_results(
            results_dir,
            "mnist_accuracy_ZeroTrustML_Krum_*.json"
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nMake sure MNIST experiment has completed!")
        return

    # Generate all figures
    print("\n🎨 Generating figures...")

    plot_accuracy_comparison(fedavg_results, zerotrustml_results, output_dir)
    plot_convergence_analysis(fedavg_results, zerotrustml_results, output_dir)
    plot_loss_curves(fedavg_results, zerotrustml_results, output_dir)
    generate_metrics_table(fedavg_results, zerotrustml_results, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("✅ ALL FIGURES GENERATED")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  📊 mnist_accuracy_comparison.pdf/png")
    print("  📊 mnist_convergence_comparison.pdf/png")
    print("  📊 mnist_loss_curves.pdf/png")
    print("  📄 mnist_results_table.tex")
    print("  📄 mnist_results_table.md")
    print("\nReady for paper submission! 🎉")


if __name__ == "__main__":
    main()
