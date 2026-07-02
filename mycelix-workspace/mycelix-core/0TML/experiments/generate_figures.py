# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
AEGIS Gen 5 - Automated Figure Generation for Paper

Generates 9 publication-ready figures from validation results:
- F1: Byzantine tolerance curves (0-50% adversaries)
- F2: Sleeper detection survival curves
- F3: Coordination detection ROC/PR curves
- F4: Active learning label efficiency
- F5: Federated convergence trajectories
- F6: Privacy-utility Pareto frontier
- F7: Distributed validator overhead
- F8: Self-healing MTTR violin plots
- F9: Secret sharing success rates

Usage:
    python generate_figures.py --results-dir validation_results --output-dir figures
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle

# Set publication-quality defaults
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "text.usetex": False,  # Set to True if LaTeX available
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

# Color palette (colorblind-safe)
COLORS = {
    "aegis": "#0173B2",  # Blue
    "baseline": "#DE8F05",  # Orange
    "krum": "#029E73",  # Green
    "median": "#CC78BC",  # Purple
    "fedmdo": "#CA9161",  # Brown
    "honest": "#56B4E9",  # Light blue
    "byzantine": "#E69F00",  # Orange
}


class FigureGenerator:
    """Generates publication-ready figures from validation results."""

    def __init__(self, results_dir: Path, output_dir: Path):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        results_path = results_dir / "results_complete.json"
        with open(results_path, "r") as f:
            data = json.load(f)
            self.results = data["results"]
            self.manifest = data["manifest"]

        print(f"📊 Loaded {len(self.results)} experiment runs")
        print(f"   Validation ID: {self.manifest['validation_id']}")

    def generate_all_figures(self):
        """Generate all 9 figures."""
        print("\n🎨 Generating publication-ready figures...")

        self.generate_f1_byzantine_tolerance()
        self.generate_f2_sleeper_detection()
        self.generate_f3_coordination_detection()
        self.generate_f4_active_learning()
        self.generate_f5_federated_convergence()
        self.generate_f6_privacy_utility()
        self.generate_f7_validator_overhead()
        self.generate_f8_self_healing()
        self.generate_f9_secret_sharing()

        print(f"\n✅ All figures saved to: {self.output_dir}")

    def generate_f1_byzantine_tolerance(self):
        """F1: Byzantine tolerance curves (0-50% adversaries)."""
        print("   Generating F1: Byzantine tolerance curves...")

        # Extract E1 results
        e1_results = [r for r in self.results if r["experiment"] == "E1_byzantine_tolerance"]

        # Group by adversary_rate and attack_type
        rates = sorted(set(r["config"]["adversary_rate"] for r in e1_results))
        attack_types = sorted(set(r["config"]["attack_type"] for r in e1_results))

        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

        for idx, metric in enumerate(["tpr", "fpr", "f1"]):
            ax = axes[idx]

            for attack_type in attack_types:
                # Collect data for this attack type
                means = []
                stds = []

                for rate in rates:
                    values = [
                        r["metrics"][metric]
                        for r in e1_results
                        if r["config"]["adversary_rate"] == rate
                        and r["config"]["attack_type"] == attack_type
                    ]
                    means.append(np.mean(values))
                    stds.append(np.std(values))

                # Plot with error bars
                ax.errorbar(
                    [r * 100 for r in rates],
                    means,
                    yerr=stds,
                    label=attack_type.replace("_", " ").title(),
                    marker="o",
                    linewidth=2,
                    capsize=4,
                )

            # Formatting
            ax.set_xlabel("Byzantine Fraction (%)")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{metric.upper()} vs Byzantine Fraction")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-2, 52)
            ax.set_ylim(0, 1.05)

            # Add 33% and 45% reference lines
            ax.axvline(x=33, color="red", linestyle="--", alpha=0.5, linewidth=1)
            ax.text(33, 0.95, "33%\nclassical\nlimit", ha="center", va="top", fontsize=8)
            ax.axvline(x=45, color="green", linestyle="--", alpha=0.5, linewidth=1)
            ax.text(45, 0.95, "45%\nAEGIS\ntarget", ha="center", va="top", fontsize=8)

            if idx == 2:
                ax.legend(loc="lower left", framealpha=0.9)

        plt.suptitle(
            "F1: Byzantine Tolerance Across Attack Types", fontsize=13, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(self.output_dir / "F1_byzantine_tolerance.pdf")
        plt.savefig(self.output_dir / "F1_byzantine_tolerance.png")
        plt.close()

    def generate_f2_sleeper_detection(self):
        """F2: Sleeper detection survival curves."""
        print("   Generating F2: Sleeper detection survival...")

        # Extract E2 results
        e2_results = [r for r in self.results if r["experiment"] == "E2_sleeper_detection"]

        # Group by activation_round and stealth_level
        activation_rounds = sorted(set(r["config"]["activation_round"] for r in e2_results))
        stealth_levels = sorted(set(r["config"]["stealth_level"] for r in e2_results))

        fig, ax = plt.subplots(figsize=(6, 4))

        for stealth in stealth_levels:
            # Collect time-to-detection for this stealth level
            for activation in activation_rounds:
                ttds = [
                    r["metrics"]["time_to_detection"]
                    for r in e2_results
                    if r["config"]["activation_round"] == activation
                    and r["config"]["stealth_level"] == stealth
                ]

                # Simple survival curve (fraction not detected at each time)
                max_ttd = max(ttds) + 1
                times = np.arange(0, max_ttd)
                survival = [sum(t >= time for t in ttds) / len(ttds) for time in times]

                label = f"Activation={activation}, Stealth={stealth}"
                ax.plot(times, survival, label=label, linewidth=2)

        ax.set_xlabel("Rounds After Activation")
        ax.set_ylabel("Fraction Undetected")
        ax.set_title("F2: Sleeper Agent Detection Time")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        plt.savefig(self.output_dir / "F2_sleeper_detection.pdf")
        plt.savefig(self.output_dir / "F2_sleeper_detection.png")
        plt.close()

    def generate_f3_coordination_detection(self):
        """F3: Coordination detection ROC/PR curves."""
        print("   Generating F3: Coordination detection ROC...")

        # Extract E3 results
        e3_results = [r for r in self.results if r["experiment"] == "E3_coordination_detection"]

        # Compute TPR for different configurations
        fig, ax = plt.subplots(figsize=(5, 5))

        # Group by correlation_strength
        correlations = sorted(set(r["config"]["correlation_strength"] for r in e3_results))

        tprs = []
        for corr in correlations:
            tpr_values = [
                r["metrics"]["coordination_tpr"]
                for r in e3_results
                if r["config"]["correlation_strength"] == corr
            ]
            tprs.append(np.mean(tpr_values))

        # Plot as bar chart
        x_pos = np.arange(len(correlations))
        bars = ax.bar(x_pos, tprs, color=COLORS["aegis"], alpha=0.7, edgecolor="black")

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xlabel("Coordination Strength (Correlation)")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("F3: Coordinated Attack Detection Performance")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{c:.1f}" for c in correlations])
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

        # Add reference line at 0.9 (target)
        ax.axhline(y=0.9, color="green", linestyle="--", alpha=0.5, linewidth=1.5)
        ax.text(len(correlations) - 0.5, 0.92, "Target (90%)", ha="right", fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / "F3_coordination_detection.pdf")
        plt.savefig(self.output_dir / "F3_coordination_detection.png")
        plt.close()

    def generate_f4_active_learning(self):
        """F4: Active learning label efficiency."""
        print("   Generating F4: Active learning efficiency...")

        # Extract E4 results
        e4_results = [r for r in self.results if r["experiment"] == "E4_active_learning_speedup"]

        # Group by query_strategy and budget_fraction
        strategies = sorted(set(r["config"]["query_strategy"] for r in e4_results))
        budgets = sorted(set(r["config"]["budget_fraction"] for r in e4_results))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Left: Accuracy vs Budget
        ax = axes[0]
        for strategy in strategies:
            accuracies = []
            for budget in budgets:
                acc_values = [
                    r["metrics"]["accuracy"]
                    for r in e4_results
                    if r["config"]["query_strategy"] == strategy
                    and r["config"]["budget_fraction"] == budget
                ]
                accuracies.append(np.mean(acc_values))

            ax.plot(
                [b * 100 for b in budgets],
                accuracies,
                label=strategy.title(),
                marker="o",
                linewidth=2,
            )

        ax.set_xlabel("Budget (%)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Query Budget")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", framealpha=0.9)
        ax.set_ylim(0.5, 1.05)

        # Right: Speedup vs Budget
        ax = axes[1]
        for strategy in strategies:
            speedups = []
            for budget in budgets:
                speedup_values = [
                    r["metrics"]["speedup"]
                    for r in e4_results
                    if r["config"]["query_strategy"] == strategy
                    and r["config"]["budget_fraction"] == budget
                ]
                speedups.append(np.mean(speedup_values))

            ax.plot(
                [b * 100 for b in budgets],
                speedups,
                label=strategy.title(),
                marker="s",
                linewidth=2,
            )

        ax.set_xlabel("Budget (%)")
        ax.set_ylabel("Speedup (×)")
        ax.set_title("Speedup vs Query Budget")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.9)

        # Add target speedup line
        ax.axhline(y=6, color="green", linestyle="--", alpha=0.5, linewidth=1.5)
        ax.text(max(budgets) * 95, 6.2, "Target (6×)", ha="right", fontsize=8)

        plt.suptitle("F4: Active Learning Label Efficiency", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.output_dir / "F4_active_learning.pdf")
        plt.savefig(self.output_dir / "F4_active_learning.png")
        plt.close()

    def generate_f5_federated_convergence(self):
        """F5: Federated convergence trajectories."""
        print("   Generating F5: Federated convergence...")

        # Extract E5 results
        e5_results = [r for r in self.results if r["experiment"] == "E5_federated_convergence"]

        # Group by optimizer
        optimizers = sorted(set(r["config"]["optimizer"] for r in e5_results))

        fig, ax = plt.subplots(figsize=(6, 4))

        for optimizer in optimizers:
            # Collect final loss and convergence rounds
            final_losses = [
                r["metrics"]["final_loss"]
                for r in e5_results
                if r["config"]["optimizer"] == optimizer
            ]
            conv_rounds = [
                r["metrics"]["convergence_round"]
                for r in e5_results
                if r["config"]["optimizer"] == optimizer
            ]

            mean_loss = np.mean(final_losses)
            mean_conv = np.mean(conv_rounds)

            ax.scatter(
                mean_conv,
                mean_loss,
                s=200,
                label=optimizer.upper(),
                alpha=0.7,
                edgecolors="black",
                linewidths=1.5,
            )

        ax.set_xlabel("Convergence Round")
        ax.set_ylabel("Final Loss")
        ax.set_title("F5: Optimizer Convergence Performance")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", framealpha=0.9)

        # Annotate better region
        ax.add_patch(
            Rectangle(
                (0, 0),
                15,
                0.3,
                facecolor="green",
                alpha=0.1,
                edgecolor="green",
                linestyle="--",
            )
        )
        ax.text(7.5, 0.15, "Better\n(faster, lower loss)", ha="center", fontsize=8, color="green")

        plt.tight_layout()
        plt.savefig(self.output_dir / "F5_federated_convergence.pdf")
        plt.savefig(self.output_dir / "F5_federated_convergence.png")
        plt.close()

    def generate_f6_privacy_utility(self):
        """F6: Privacy-utility Pareto frontier."""
        print("   Generating F6: Privacy-utility tradeoff...")

        # Extract E6 results
        e6_results = [r for r in self.results if r["experiment"] == "E6_privacy_utility_tradeoff"]

        # Extract epsilon and accuracy
        epsilons = [r["config"]["epsilon"] for r in e6_results]
        accuracies = [r["metrics"]["accuracy"] for r in e6_results]

        # Group by epsilon
        unique_epsilons = sorted(set(epsilons))
        mean_accs = []
        std_accs = []

        for eps in unique_epsilons:
            acc_values = [
                r["metrics"]["accuracy"]
                for r in e6_results
                if r["config"]["epsilon"] == eps
            ]
            mean_accs.append(np.mean(acc_values))
            std_accs.append(np.std(acc_values))

        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot Pareto frontier
        ax.errorbar(
            unique_epsilons,
            mean_accs,
            yerr=std_accs,
            marker="o",
            linewidth=2,
            capsize=4,
            color=COLORS["aegis"],
            label="AEGIS (DP-FedMDO)",
        )

        ax.set_xlabel("Privacy Budget (ε)")
        ax.set_ylabel("Accuracy")
        ax.set_title("F6: Privacy-Utility Tradeoff")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", framealpha=0.9)
        ax.set_ylim(0.5, 1.05)

        # Add privacy zones
        ax.axvspan(0, 1, alpha=0.1, color="green", label="Strong Privacy (ε<1)")
        ax.axvspan(1, 10, alpha=0.1, color="yellow")
        ax.axvspan(10, max(unique_epsilons) + 1, alpha=0.1, color="red")

        ax.text(0.5, 0.52, "Strong\nPrivacy", ha="center", fontsize=8, color="darkgreen")
        ax.text(5, 0.52, "Moderate\nPrivacy", ha="center", fontsize=8, color="darkorange")
        ax.text(12, 0.52, "Weak\nPrivacy", ha="center", fontsize=8, color="darkred")

        plt.tight_layout()
        plt.savefig(self.output_dir / "F6_privacy_utility.pdf")
        plt.savefig(self.output_dir / "F6_privacy_utility.png")
        plt.close()

    def generate_f7_validator_overhead(self):
        """F7: Distributed validator overhead."""
        print("   Generating F7: Validator overhead...")

        # Extract E7 results
        e7_results = [r for r in self.results if r["experiment"] == "E7_distributed_validation_overhead"]

        # Group by n_validators
        n_validators_list = sorted(set(r["config"]["n_validators"] for r in e7_results))

        share_times = []
        reconstruct_times = []
        total_times = []

        for n in n_validators_list:
            share_vals = [
                r["metrics"]["share_time_ms"]
                for r in e7_results
                if r["config"]["n_validators"] == n
            ]
            recon_vals = [
                r["metrics"]["reconstruct_time_ms"]
                for r in e7_results
                if r["config"]["n_validators"] == n
            ]
            total_vals = [
                r["metrics"]["total_overhead_ms"]
                for r in e7_results
                if r["config"]["n_validators"] == n
            ]

            share_times.append(np.mean(share_vals))
            reconstruct_times.append(np.mean(recon_vals))
            total_times.append(np.mean(total_vals))

        fig, ax = plt.subplots(figsize=(6, 4))

        x_pos = np.arange(len(n_validators_list))
        width = 0.25

        # Stacked bars
        ax.bar(
            x_pos - width,
            share_times,
            width,
            label="Share Generation",
            color=COLORS["honest"],
            edgecolor="black",
        )
        ax.bar(
            x_pos,
            reconstruct_times,
            width,
            label="Reconstruction",
            color=COLORS["byzantine"],
            edgecolor="black",
        )
        ax.bar(
            x_pos + width,
            total_times,
            width,
            label="Total Overhead",
            color=COLORS["aegis"],
            edgecolor="black",
        )

        ax.set_xlabel("Number of Validators")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("F7: Distributed Validation Overhead")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(n) for n in n_validators_list])
        ax.legend(loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.3, axis="y")

        # Add acceptable overhead line
        ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, linewidth=1.5)
        ax.text(len(n_validators_list) - 0.3, 52, "Target (<50ms)", ha="right", fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / "F7_validator_overhead.pdf")
        plt.savefig(self.output_dir / "F7_validator_overhead.png")
        plt.close()

    def generate_f8_self_healing(self):
        """F8: Self-healing MTTR violin plots."""
        print("   Generating F8: Self-healing MTTR...")

        # Extract E8 results
        e8_results = [r for r in self.results if r["experiment"] == "E8_self_healing_recovery"]

        # Group by attack_type
        attack_types = sorted(set(r["config"]["attack_type"] for r in e8_results))

        fig, ax = plt.subplots(figsize=(7, 4))

        data_for_violin = []
        labels = []

        for attack_type in attack_types:
            mttr_values = [
                r["metrics"]["mttr_rounds"]
                for r in e8_results
                if r["config"]["attack_type"] == attack_type
            ]
            data_for_violin.append(mttr_values)
            labels.append(attack_type.replace("_", " ").title())

        # Create violin plot
        parts = ax.violinplot(
            data_for_violin,
            positions=np.arange(1, len(attack_types) + 1),
            widths=0.7,
            showmeans=True,
            showextrema=True,
        )

        # Color violins
        for pc in parts["bodies"]:
            pc.set_facecolor(COLORS["aegis"])
            pc.set_alpha(0.7)
            pc.set_edgecolor("black")
            pc.set_linewidth(1.5)

        ax.set_xlabel("Attack Type")
        ax.set_ylabel("Mean Time to Recovery (rounds)")
        ax.set_title("F8: Self-Healing Recovery Time Distribution")
        ax.set_xticks(np.arange(1, len(attack_types) + 1))
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.grid(True, alpha=0.3, axis="y")

        # Add target MTTR line
        ax.axhline(y=20, color="green", linestyle="--", alpha=0.5, linewidth=1.5)
        ax.text(len(attack_types), 21, "Target (<20 rounds)", ha="right", fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / "F8_self_healing.pdf")
        plt.savefig(self.output_dir / "F8_self_healing.png")
        plt.close()

    def generate_f9_secret_sharing(self):
        """F9: Secret sharing success rates."""
        print("   Generating F9: Secret sharing tolerance...")

        # Extract E9 results
        e9_results = [r for r in self.results if r["experiment"] == "E9_secret_sharing_tolerance"]

        # Group by n_byzantine_validators
        n_byzantine_list = sorted(set(r["config"]["n_byzantine_validators"] for r in e9_results))

        success_rates = []
        for n_byz in n_byzantine_list:
            rates = [
                r["metrics"]["reconstruction_success_rate"]
                for r in e9_results
                if r["config"]["n_byzantine_validators"] == n_byz
            ]
            success_rates.append(np.mean(rates))

        fig, ax = plt.subplots(figsize=(6, 4))

        # Bar chart
        x_pos = np.arange(len(n_byzantine_list))
        bars = ax.bar(
            x_pos,
            [r * 100 for r in success_rates],
            color=[
                COLORS["honest"] if n < 3 else COLORS["byzantine"]
                for n in n_byzantine_list
            ],
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )

        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xlabel("Number of Byzantine Validators")
        ax.set_ylabel("Reconstruction Success Rate (%)")
        ax.set_title("F9: Secret Sharing Byzantine Tolerance (n=7, t=4)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(n) for n in n_byzantine_list])
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis="y")

        # Add threshold line
        ax.axvline(x=2.5, color="red", linestyle="--", alpha=0.5, linewidth=2)
        ax.text(2.5, 105, "BFT Limit\n(n - t = 3)", ha="center", fontsize=8, color="red")

        plt.tight_layout()
        plt.savefig(self.output_dir / "F9_secret_sharing.pdf")
        plt.savefig(self.output_dir / "F9_secret_sharing.png")
        plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figures from AEGIS validation results"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("validation_results"),
        help="Directory containing results_complete.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures",
    )

    args = parser.parse_args()

    # Check results exist
    results_path = args.results_dir / "results_complete.json"
    if not results_path.exists():
        print(f"❌ Results file not found: {results_path}")
        print("   Run validation experiments first: python run_validation.py")
        return

    # Generate figures
    generator = FigureGenerator(args.results_dir, args.output_dir)
    generator.generate_all_figures()


if __name__ == "__main__":
    main()
