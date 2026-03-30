# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Results Analysis and Visualization Utilities

Tools for analyzing and plotting FL experiment results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


class ResultsAnalyzer:
    """Analyze and visualize FL experiment results."""

    def __init__(self, results_path: str):
        """
        Initialize analyzer with results file.

        Args:
            results_path: Path to JSON results file
        """
        self.results_path = Path(results_path)
        self.results = self.load_results()

    def load_results(self) -> Dict:
        """Load results from JSON file."""
        with open(self.results_path, 'r') as f:
            results = json.load(f)
        return results

    def get_baseline_names(self) -> List[str]:
        """Get list of baselines in results."""
        return list(self.results['baselines'].keys())

    def get_final_accuracy(self, baseline: str) -> float:
        """
        Get final test accuracy for a baseline.

        Args:
            baseline: Baseline name

        Returns:
            Final test accuracy (last non-None value)
        """
        history = self.results['baselines'][baseline]
        test_acc = [acc for acc in history['test_accuracy'] if acc is not None]
        return test_acc[-1] if test_acc else 0.0

    def get_convergence_round(self, baseline: str, target_acc: float = 0.95) -> int:
        """
        Get round number where baseline first reaches target accuracy.

        Args:
            baseline: Baseline name
            target_acc: Target accuracy threshold

        Returns:
            Round number, or -1 if never reached
        """
        history = self.results['baselines'][baseline]
        test_acc = history['test_accuracy']

        for round_num, acc in enumerate(test_acc):
            if acc is not None and acc >= target_acc:
                return round_num

        return -1  # Never reached

    def create_comparison_table(self) -> pd.DataFrame:
        """
        Create comparison table of baseline performance.

        Returns:
            DataFrame with metrics for each baseline
        """
        baselines = self.get_baseline_names()

        data = []
        for baseline in baselines:
            final_acc = self.get_final_accuracy(baseline)
            conv_round = self.get_convergence_round(baseline, 0.95)

            history = self.results['baselines'][baseline]
            final_loss = [loss for loss in history['test_loss'] if loss is not None][-1]

            data.append({
                'Baseline': baseline,
                'Final Accuracy': f"{final_acc:.4f}",
                'Final Loss': f"{final_loss:.4f}",
                'Rounds to 95%': conv_round if conv_round != -1 else 'N/A'
            })

        df = pd.DataFrame(data)
        return df

    def plot_convergence(
        self,
        save_path: Optional[str] = None,
        metric: str = 'accuracy'
    ):
        """
        Plot convergence curves for all baselines.

        Args:
            save_path: Path to save figure (optional)
            metric: 'accuracy' or 'loss'
        """
        plt.figure(figsize=(10, 6))

        baselines = self.get_baseline_names()
        colors = plt.cm.tab10(np.linspace(0, 1, len(baselines)))

        for baseline, color in zip(baselines, colors):
            history = self.results['baselines'][baseline]

            rounds = history['rounds']

            if metric == 'accuracy':
                values = history['test_accuracy']
                ylabel = 'Test Accuracy'
                title = 'Test Accuracy vs Training Rounds'
            else:
                values = history['test_loss']
                ylabel = 'Test Loss'
                title = 'Test Loss vs Training Rounds'

            # Filter out None values
            plot_rounds = [r for r, v in zip(rounds, values) if v is not None]
            plot_values = [v for v in values if v is not None]

            plt.plot(plot_rounds, plot_values, label=baseline, color=color, linewidth=2)

        plt.xlabel('Round', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved plot to: {save_path}")
        else:
            plt.show()

    def plot_comparison_bar(
        self,
        metric: str = 'accuracy',
        save_path: Optional[str] = None
    ):
        """
        Plot bar chart comparing final performance.

        Args:
            metric: 'accuracy' or 'loss'
            save_path: Path to save figure (optional)
        """
        baselines = self.get_baseline_names()

        values = []
        for baseline in baselines:
            if metric == 'accuracy':
                val = self.get_final_accuracy(baseline)
            else:
                history = self.results['baselines'][baseline]
                test_loss = [loss for loss in history['test_loss'] if loss is not None]
                val = test_loss[-1] if test_loss else 0.0
            values.append(val)

        plt.figure(figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(baselines)))

        bars = plt.bar(baselines, values, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{val:.3f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        if metric == 'accuracy':
            ylabel = 'Final Test Accuracy'
            title = 'Final Test Accuracy Comparison'
        else:
            ylabel = 'Final Test Loss'
            title = 'Final Test Loss Comparison'

        plt.xlabel('Baseline', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved plot to: {save_path}")
        else:
            plt.show()

    def generate_report(self, output_dir: Optional[str] = None):
        """
        Generate complete analysis report with tables and plots.

        Args:
            output_dir: Directory to save report files
        """
        if output_dir is None:
            output_dir = self.results_path.parent / f"{self.results_path.stem}_analysis"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("Generating Analysis Report")
        print("=" * 70)

        # Create comparison table
        print("\n📊 Baseline Comparison Table:")
        df = self.create_comparison_table()
        print(df.to_string(index=False))

        # Save table to CSV
        csv_path = output_dir / "comparison_table.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Saved table to: {csv_path}")

        # Generate plots
        print("\n📈 Generating plots...")

        # Convergence plots
        self.plot_convergence(
            save_path=output_dir / "convergence_accuracy.png",
            metric='accuracy'
        )
        self.plot_convergence(
            save_path=output_dir / "convergence_loss.png",
            metric='loss'
        )

        # Comparison bar charts
        self.plot_comparison_bar(
            metric='accuracy',
            save_path=output_dir / "comparison_accuracy.png"
        )
        self.plot_comparison_bar(
            metric='loss',
            save_path=output_dir / "comparison_loss.png"
        )

        print("\n" + "=" * 70)
        print(f"✅ Report generated in: {output_dir}")
        print("=" * 70)


def compare_experiments(results_paths: List[str], save_path: Optional[str] = None):
    """
    Compare multiple experiments (e.g., IID vs non-IID).

    Args:
        results_paths: List of paths to results JSON files
        save_path: Path to save comparison plot
    """
    plt.figure(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_paths)))

    for results_path, color in zip(results_paths, colors):
        analyzer = ResultsAnalyzer(results_path)

        # Use experiment name from config
        exp_name = analyzer.results['config']['experiment_name']

        # Plot FedAvg from each experiment for comparison
        if 'fedavg' in analyzer.results['baselines']:
            history = analyzer.results['baselines']['fedavg']
            rounds = history['rounds']
            test_acc = history['test_accuracy']

            plot_rounds = [r for r, v in zip(rounds, test_acc) if v is not None]
            plot_values = [v for v in test_acc if v is not None]

            plt.plot(plot_rounds, plot_values, label=exp_name, color=color, linewidth=2)

    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('FedAvg Across Different Experiments', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved comparison plot to: {save_path}")
    else:
        plt.show()


# ============================================================================
# Command-line interface
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Analyze FL experiment results')
    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for analysis'
    )
    args = parser.parse_args()

    # Create analyzer and generate report
    analyzer = ResultsAnalyzer(args.results)
    analyzer.generate_report(args.output)
