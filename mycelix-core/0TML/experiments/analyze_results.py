#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Aggregate and Analyze 96 Experiment Results

This script:
1. Loads all experiment artifacts
2. Aggregates metrics across datasets/attacks/defenses
3. Computes statistical comparisons
4. Generates summary tables and data for figures
5. Identifies key findings

Usage:
    python experiments/analyze_results.py
    python experiments/analyze_results.py --output-dir analysis/
    python experiments/analyze_results.py --datasets mnist emnist
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict


@dataclass
class ExperimentResult:
    """Single experiment result"""
    name: str
    dataset: str
    attack: str
    defense: str
    seed: int

    # Metrics
    final_accuracy: float
    asr: float  # Attack Success Rate
    detection_rate: float
    false_positive_rate: float

    # Convergence
    rounds_to_converge: int
    converged: bool

    # Raw data
    accuracy_history: List[float]
    loss_history: List[float]


@dataclass
class AggregatedResults:
    """Aggregated results across seeds"""
    dataset: str
    attack: str
    defense: str

    # Mean ± std
    accuracy_mean: float
    accuracy_std: float

    asr_mean: float
    asr_std: float

    detection_mean: float
    detection_std: float

    fpr_mean: float
    fpr_std: float

    # Sample size
    n_experiments: int

    # Raw results
    raw_results: List[ExperimentResult]


class ResultsAnalyzer:
    """Analyze experiment results"""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.experiments: List[ExperimentResult] = []
        self.aggregated: Dict[Tuple[str, str, str], AggregatedResults] = {}

    def load_all_results(self) -> int:
        """Load all experiment artifacts"""
        print("=" * 70)
        print("Loading Experiment Results")
        print("=" * 70)

        artifact_dirs = sorted(self.results_dir.glob("artifacts_*"))

        loaded = 0
        failed = 0

        for artifact_dir in artifact_dirs:
            try:
                result = self._load_experiment(artifact_dir)
                if result:
                    self.experiments.append(result)
                    loaded += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"⚠️  Failed to load {artifact_dir.name}: {e}")
                failed += 1

        print(f"\n✅ Loaded: {loaded}/{loaded + failed} experiments")
        print(f"❌ Failed: {failed}/{loaded + failed} experiments\n")

        return loaded

    def _load_experiment(self, artifact_dir: Path) -> ExperimentResult | None:
        """Load single experiment result"""

        # Parse experiment name from directory
        # Format: artifacts_final_3datasets_mnist_sign_flip_fedavg_seed42_20251111_123456
        parts = artifact_dir.name.split("_")

        # Find dataset, attack, defense, seed
        try:
            # Extract from parts
            dataset_idx = parts.index("mnist") if "mnist" in parts else \
                         parts.index("emnist") if "emnist" in parts else \
                         parts.index("cifar10") if "cifar10" in parts else None

            if dataset_idx is None:
                return None

            dataset = parts[dataset_idx]
            attack = parts[dataset_idx + 1]
            defense = parts[dataset_idx + 2]

            # Extract seed
            seed_str = [p for p in parts if p.startswith("seed")][0]
            seed = int(seed_str.replace("seed", ""))

        except (ValueError, IndexError):
            return None

        # Load metrics
        metrics_file = artifact_dir / "detection_metrics.json"
        if not metrics_file.exists():
            return None

        with open(metrics_file) as f:
            metrics = json.load(f)

        # Extract key metrics
        try:
            # Get final accuracy from defense results
            defense_results = metrics.get(defense, {})
            accuracy_history = defense_results.get("test_accuracy", [])
            loss_history = defense_results.get("test_loss", [])

            if not accuracy_history:
                return None

            final_accuracy = accuracy_history[-1]

            # Detection metrics
            detection = metrics.get("detection", {})
            detection_rate = detection.get("detection_rate", 0.0)
            fpr = detection.get("false_positive_rate", 0.0)

            # ASR (attack success rate) - accuracy degradation
            baseline_acc = metrics.get("fedavg", {}).get("test_accuracy", [])
            if baseline_acc:
                asr = max(0, baseline_acc[-1] - final_accuracy)
            else:
                asr = 0.0

            # Convergence check (stable for last 10 rounds)
            converged = False
            rounds_to_converge = len(accuracy_history)

            if len(accuracy_history) >= 10:
                last_10 = accuracy_history[-10:]
                if max(last_10) - min(last_10) < 0.01:  # Within 1%
                    converged = True
                    # Find first convergence point
                    for i in range(len(accuracy_history) - 10):
                        window = accuracy_history[i:i+10]
                        if max(window) - min(window) < 0.01:
                            rounds_to_converge = i + 10
                            break

            return ExperimentResult(
                name=artifact_dir.name,
                dataset=dataset,
                attack=attack,
                defense=defense,
                seed=seed,
                final_accuracy=final_accuracy,
                asr=asr,
                detection_rate=detection_rate,
                false_positive_rate=fpr,
                rounds_to_converge=rounds_to_converge,
                converged=converged,
                accuracy_history=accuracy_history,
                loss_history=loss_history
            )

        except (KeyError, IndexError, ValueError) as e:
            print(f"⚠️  Error parsing metrics from {artifact_dir.name}: {e}")
            return None

    def aggregate_by_configuration(self):
        """Aggregate results across seeds"""
        print("=" * 70)
        print("Aggregating Results by Configuration")
        print("=" * 70)

        # Group by (dataset, attack, defense)
        grouped: Dict[Tuple[str, str, str], List[ExperimentResult]] = defaultdict(list)

        for exp in self.experiments:
            key = (exp.dataset, exp.attack, exp.defense)
            grouped[key].append(exp)

        # Compute aggregates
        for key, experiments in grouped.items():
            dataset, attack, defense = key

            accuracies = [e.final_accuracy for e in experiments]
            asrs = [e.asr for e in experiments]
            detections = [e.detection_rate for e in experiments]
            fprs = [e.false_positive_rate for e in experiments]

            self.aggregated[key] = AggregatedResults(
                dataset=dataset,
                attack=attack,
                defense=defense,
                accuracy_mean=np.mean(accuracies),
                accuracy_std=np.std(accuracies),
                asr_mean=np.mean(asrs),
                asr_std=np.std(asrs),
                detection_mean=np.mean(detections),
                detection_std=np.std(detections),
                fpr_mean=np.mean(fprs),
                fpr_std=np.std(fprs),
                n_experiments=len(experiments),
                raw_results=experiments
            )

        print(f"\n✅ Created {len(self.aggregated)} aggregated configurations\n")

    def compare_defenses(self) -> Dict[str, Dict]:
        """Compare defense mechanisms across attacks and datasets"""
        print("=" * 70)
        print("Defense Comparison")
        print("=" * 70)

        comparison = {}

        for dataset in ["mnist", "emnist", "cifar10"]:
            comparison[dataset] = {}

            for attack in ["sign_flip", "scaling_x100", "collusion", "sleeper_agent"]:
                comparison[dataset][attack] = {}

                for defense in ["fedavg", "fltrust", "boba", "pogq_v4_1"]:
                    key = (dataset, attack, defense)

                    if key in self.aggregated:
                        agg = self.aggregated[key]
                        comparison[dataset][attack][defense] = {
                            "accuracy": f"{agg.accuracy_mean:.4f} ± {agg.accuracy_std:.4f}",
                            "asr": f"{agg.asr_mean:.4f} ± {agg.asr_std:.4f}",
                            "detection": f"{agg.detection_mean:.4f} ± {agg.detection_std:.4f}",
                            "fpr": f"{agg.fpr_mean:.4f} ± {agg.fpr_std:.4f}",
                        }
                    else:
                        comparison[dataset][attack][defense] = None

        return comparison

    def find_best_defense(self) -> Dict[str, Dict[str, str]]:
        """Find best defense for each dataset/attack combination"""
        print("=" * 70)
        print("Best Defense per Configuration")
        print("=" * 70)

        best = {}

        for dataset in ["mnist", "emnist", "cifar10"]:
            best[dataset] = {}

            for attack in ["sign_flip", "scaling_x100", "collusion", "sleeper_agent"]:
                best_defense = None
                best_accuracy = -1

                for defense in ["fedavg", "fltrust", "boba", "pogq_v4_1"]:
                    key = (dataset, attack, defense)

                    if key in self.aggregated:
                        agg = self.aggregated[key]
                        if agg.accuracy_mean > best_accuracy:
                            best_accuracy = agg.accuracy_mean
                            best_defense = defense

                best[dataset][attack] = best_defense
                print(f"{dataset:8} / {attack:15} → {best_defense or 'NONE':10} ({best_accuracy:.4f})")

        print()
        return best

    def generate_summary_table(self, output_file: str = "analysis/summary_table.txt"):
        """Generate formatted summary table"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("=" * 120 + "\n")
            f.write("EXPERIMENT RESULTS SUMMARY\n")
            f.write("=" * 120 + "\n\n")

            for dataset in ["mnist", "emnist", "cifar10"]:
                f.write(f"\n{'=' * 120}\n")
                f.write(f"Dataset: {dataset.upper()}\n")
                f.write(f"{'=' * 120}\n\n")

                for attack in ["sign_flip", "scaling_x100", "collusion", "sleeper_agent"]:
                    f.write(f"\nAttack: {attack}\n")
                    f.write("-" * 120 + "\n")
                    f.write(f"{'Defense':<15} {'Accuracy':<20} {'ASR':<20} {'Detection':<20} {'FPR':<20}\n")
                    f.write("-" * 120 + "\n")

                    for defense in ["fedavg", "fltrust", "boba", "pogq_v4_1"]:
                        key = (dataset, attack, defense)

                        if key in self.aggregated:
                            agg = self.aggregated[key]
                            f.write(
                                f"{defense:<15} "
                                f"{agg.accuracy_mean:.4f}±{agg.accuracy_std:.4f}    "
                                f"{agg.asr_mean:.4f}±{agg.asr_std:.4f}    "
                                f"{agg.detection_mean:.4f}±{agg.detection_std:.4f}    "
                                f"{agg.fpr_mean:.4f}±{agg.fpr_std:.4f}\n"
                            )
                        else:
                            f.write(f"{defense:<15} NO DATA\n")

                    f.write("\n")

        print(f"✅ Summary table written to: {output_path}")

    def export_for_figures(self, output_dir: str = "analysis/figure_data"):
        """Export data in format ready for figure generation"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Figure 1: Accuracy comparison across defenses
        fig1_data = {}
        for dataset in ["mnist", "emnist", "cifar10"]:
            fig1_data[dataset] = {}
            for attack in ["sign_flip", "scaling_x100", "collusion", "sleeper_agent"]:
                fig1_data[dataset][attack] = []
                for defense in ["fedavg", "fltrust", "boba", "pogq_v4_1"]:
                    key = (dataset, attack, defense)
                    if key in self.aggregated:
                        agg = self.aggregated[key]
                        fig1_data[dataset][attack].append({
                            "defense": defense,
                            "accuracy_mean": agg.accuracy_mean,
                            "accuracy_std": agg.accuracy_std
                        })

        with open(output_path / "fig1_accuracy_comparison.json", "w") as f:
            json.dump(fig1_data, f, indent=2)

        print(f"✅ Figure data exported to: {output_path}")

    def generate_report(self, output_file: str = "analysis/ANALYSIS_REPORT.md"):
        """Generate comprehensive markdown report"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("# Experiment Results Analysis\n\n")
            f.write(f"**Generated**: {Path.cwd()}\n")
            f.write(f"**Total Experiments**: {len(self.experiments)}\n")
            f.write(f"**Configurations**: {len(self.aggregated)}\n\n")

            f.write("## Overview\n\n")
            f.write("This report summarizes results from 96 experiments testing Byzantine-resistant ")
            f.write("federated learning across 3 datasets, 4 attack types, and 4 defense mechanisms.\n\n")

            f.write("## Key Findings\n\n")

            # TODO: Add automated key findings detection
            f.write("1. **Defense Performance**: [To be filled after analysis]\n")
            f.write("2. **Attack Effectiveness**: [To be filled after analysis]\n")
            f.write("3. **Dataset Differences**: [To be filled after analysis]\n\n")

            f.write("## Best Defense per Configuration\n\n")
            best = self.find_best_defense()

            f.write("| Dataset | Attack | Best Defense |\n")
            f.write("|---------|--------|-------------|\n")
            for dataset in ["mnist", "emnist", "cifar10"]:
                for attack in ["sign_flip", "scaling_x100", "collusion", "sleeper_agent"]:
                    defense = best.get(dataset, {}).get(attack, "N/A")
                    f.write(f"| {dataset} | {attack} | {defense} |\n")

        print(f"✅ Analysis report written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--results-dir", default="results", help="Results directory")
    parser.add_argument("--output-dir", default="analysis", help="Output directory")
    parser.add_argument("--datasets", nargs="+", help="Filter by datasets")

    args = parser.parse_args()

    # Create analyzer
    analyzer = ResultsAnalyzer(args.results_dir)

    # Load all results
    n_loaded = analyzer.load_all_results()

    if n_loaded == 0:
        print("❌ No results found!")
        return 1

    # Filter by datasets if specified
    if args.datasets:
        analyzer.experiments = [
            e for e in analyzer.experiments
            if e.dataset in args.datasets
        ]
        print(f"Filtered to {len(analyzer.experiments)} experiments for datasets: {args.datasets}")

    # Aggregate results
    analyzer.aggregate_by_configuration()

    # Generate outputs
    analyzer.generate_summary_table(f"{args.output_dir}/summary_table.txt")
    analyzer.export_for_figures(f"{args.output_dir}/figure_data")
    analyzer.generate_report(f"{args.output_dir}/ANALYSIS_REPORT.md")

    # Compare defenses
    comparison = analyzer.compare_defenses()
    with open(f"{args.output_dir}/defense_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs in: {args.output_dir}/")
    print("  - summary_table.txt")
    print("  - ANALYSIS_REPORT.md")
    print("  - defense_comparison.json")
    print("  - figure_data/*.json")

    return 0


if __name__ == "__main__":
    exit(main())
