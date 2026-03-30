#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Aggregate v4.1 Experiment Results

Matrix: 2 datasets × 4 attacks × 4 defenses × 2 seeds = 64 experiments

Datasets: mnist (IID), mnist (non-IID α=0.3)
Attacks: sign_flip, scaling_x100, collusion, sleeper_agent
Defenses: fedavg, fltrust, boba, pogq_v4_1
Seeds: 42, 1337

Outputs:
- Attack-defense matrix (4×4 table)
- Statistical analysis across seeds
- LaTeX table generation
- Figure data preparation

Usage:
    python experiments/aggregate_v4_1_results.py
    python experiments/aggregate_v4_1_results.py --output results/v4.1/aggregated/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict
import sys


@dataclass
class ExperimentMetrics:
    """Metrics from single experiment"""
    dataset: str
    attack: str
    defense: str
    seed: int

    # Detection metrics
    detection_rate: float  # TPR
    false_positive_rate: float  # FPR
    precision: float
    f1_score: float

    # Model performance
    final_accuracy: float
    final_loss: float

    # Additional
    converged: bool
    rounds: int


@dataclass
class AggregatedMetrics:
    """Aggregated across seeds"""
    dataset: str
    attack: str
    defense: str
    n_seeds: int

    # Detection (mean ± std)
    detection_mean: float
    detection_std: float
    fpr_mean: float
    fpr_std: float
    precision_mean: float
    precision_std: float

    # Model performance
    accuracy_mean: float
    accuracy_std: float

    # Individual results
    seeds_tested: List[int]
    raw_metrics: List[ExperimentMetrics]


class V4_1_Analyzer:
    """Analyze v4.1 experiment results"""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.experiments: List[ExperimentMetrics] = []
        self.aggregated: Dict[Tuple[str, str, str], AggregatedMetrics] = {}

    def load_experiments(self) -> int:
        """Load all v4.1 experiment artifacts"""
        print("=" * 80)
        print("Loading v4.1 Experiment Results")
        print("=" * 80)

        # Look for artifacts_* directories
        artifact_dirs = sorted(self.results_dir.glob("artifacts_*"))

        if not artifact_dirs:
            print(f"⚠️  No artifact directories found in {self.results_dir}")
            return 0

        loaded = 0
        failed = 0

        for artifact_dir in artifact_dirs:
            try:
                metrics_file = artifact_dir / "detection_metrics.json"

                if not metrics_file.exists():
                    failed += 1
                    continue

                with open(metrics_file) as f:
                    data = json.load(f)

                # Extract experiment parameters from filename
                exp_name = artifact_dir.name.replace("artifacts_", "")
                parts = exp_name.split("_")

                # Parse: {dataset}_{attack}_{defense}_seed{seed}
                dataset = parts[0]
                # Handle non-IID naming
                if len(parts) > 1 and parts[1] in ["noniid", "nonIID"]:
                    dataset = f"{dataset}_noniid"
                    attack_idx = 2
                else:
                    attack_idx = 1

                attack = parts[attack_idx]
                defense = parts[attack_idx + 1]
                seed_part = parts[attack_idx + 2] if len(parts) > attack_idx + 2 else "seed42"
                seed = int(seed_part.replace("seed", ""))

                metrics = ExperimentMetrics(
                    dataset=dataset,
                    attack=attack,
                    defense=defense,
                    seed=seed,
                    detection_rate=data.get("detection_rate", 0.0),
                    false_positive_rate=data.get("false_positive_rate", 0.0),
                    precision=data.get("precision", 0.0),
                    f1_score=data.get("f1_score", 0.0),
                    final_accuracy=data.get("final_accuracy", 0.0),
                    final_loss=data.get("final_loss", 0.0),
                    converged=data.get("converged", False),
                    rounds=data.get("rounds", 0)
                )

                self.experiments.append(metrics)
                loaded += 1

            except Exception as e:
                print(f"❌ Failed to load {artifact_dir.name}: {e}")
                failed += 1

        print(f"✅ Loaded: {loaded} experiments")
        if failed > 0:
            print(f"❌ Failed: {failed} experiments")
        print()

        return loaded

    def aggregate_by_configuration(self):
        """Aggregate results by (dataset, attack, defense)"""
        print("Aggregating by configuration...")

        grouped = defaultdict(list)

        for exp in self.experiments:
            key = (exp.dataset, exp.attack, exp.defense)
            grouped[key].append(exp)

        for key, exps in grouped.items():
            dataset, attack, defense = key

            detection_rates = [e.detection_rate for e in exps]
            fprs = [e.false_positive_rate for e in exps]
            precisions = [e.precision for e in exps]
            accuracies = [e.final_accuracy for e in exps]
            seeds = [e.seed for e in exps]

            agg = AggregatedMetrics(
                dataset=dataset,
                attack=attack,
                defense=defense,
                n_seeds=len(exps),
                detection_mean=np.mean(detection_rates),
                detection_std=np.std(detection_rates),
                fpr_mean=np.mean(fprs),
                fpr_std=np.std(fprs),
                precision_mean=np.mean(precisions),
                precision_std=np.std(precisions),
                accuracy_mean=np.mean(accuracies),
                accuracy_std=np.std(accuracies),
                seeds_tested=seeds,
                raw_metrics=exps
            )

            self.aggregated[key] = agg

        print(f"✅ Created {len(self.aggregated)} aggregated configurations\n")

    def generate_attack_defense_matrix(self, dataset: str = "mnist") -> Dict:
        """Generate attack-defense matrix for paper"""
        print(f"Generating attack-defense matrix for {dataset}...")

        attacks = ["sign_flip", "scaling_x100", "collusion", "sleeper_agent"]
        defenses = ["fedavg", "fltrust", "boba", "pogq_v4_1"]

        matrix = {
            "dataset": dataset,
            "attacks": attacks,
            "defenses": defenses,
            "detection_rates": {},
            "fprs": {},
            "accuracies": {}
        }

        for attack in attacks:
            for defense in defenses:
                key = (dataset, attack, defense)

                if key in self.aggregated:
                    agg = self.aggregated[key]
                    matrix["detection_rates"][(attack, defense)] = {
                        "mean": agg.detection_mean,
                        "std": agg.detection_std
                    }
                    matrix["fprs"][(attack, defense)] = {
                        "mean": agg.fpr_mean,
                        "std": agg.fpr_std
                    }
                    matrix["accuracies"][(attack, defense)] = {
                        "mean": agg.accuracy_mean,
                        "std": agg.accuracy_std
                    }
                else:
                    print(f"  ⚠️  Missing: {attack} × {defense}")

        return matrix

    def generate_latex_table(self, matrix: Dict, output_file: Path):
        """Generate LaTeX attack-defense matrix table"""
        print(f"Generating LaTeX table: {output_file}")

        attacks = matrix["attacks"]
        defenses = matrix["defenses"]

        # Map defense names to paper abbreviations
        defense_labels = {
            "fedavg": "FedAvg",
            "fltrust": "FLTrust",
            "boba": "BOBA",
            "pogq_v4_1": "PoGQ"
        }

        attack_labels = {
            "sign_flip": "Sign Flip",
            "scaling_x100": "Scaling",
            "collusion": "Collusion",
            "sleeper_agent": "Sleeper"
        }

        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Attack-Defense Matrix: Detection Rate (\\%) on MNIST}")
        lines.append("\\label{tab:attack_defense_matrix}")
        lines.append("\\begin{tabular}{l" + "c" * len(defenses) + "}")
        lines.append("\\toprule")

        # Header
        header = "Attack"
        for defense in defenses:
            header += f" & {defense_labels.get(defense, defense)}"
        header += " \\\\"
        lines.append(header)
        lines.append("\\midrule")

        # Rows
        for attack in attacks:
            row = attack_labels.get(attack, attack)
            for defense in defenses:
                key = (attack, defense)
                if key in matrix["detection_rates"]:
                    dr = matrix["detection_rates"][key]
                    mean_pct = dr["mean"] * 100
                    std_pct = dr["std"] * 100

                    # Format: mean ± std
                    if dr["std"] < 0.01:
                        # Low variance
                        row += f" & {mean_pct:.1f}"
                    else:
                        row += f" & {mean_pct:.1f} $\\pm$ {std_pct:.1f}"
                else:
                    row += " & ---"
            row += " \\\\"
            lines.append(row)

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        latex = "\n".join(lines)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(latex)

        print(f"✅ LaTeX table written to {output_file}\n")
        return latex

    def generate_summary_report(self, output_file: Path):
        """Generate human-readable summary"""
        print(f"Generating summary report: {output_file}")

        lines = []
        lines.append("=" * 80)
        lines.append("v4.1 Experiment Results Summary")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Total Experiments: {len(self.experiments)}")
        lines.append(f"Aggregated Configurations: {len(self.aggregated)}")
        lines.append("")

        # Count by dataset
        datasets = set(e.dataset for e in self.experiments)
        lines.append("Datasets:")
        for dataset in sorted(datasets):
            count = len([e for e in self.experiments if e.dataset == dataset])
            lines.append(f"  - {dataset}: {count} experiments")
        lines.append("")

        # Best performers
        lines.append("Top Performers (by detection rate):")
        sorted_agg = sorted(
            self.aggregated.values(),
            key=lambda x: x.detection_mean,
            reverse=True
        )[:10]

        for agg in sorted_agg:
            lines.append(
                f"  {agg.defense} vs {agg.attack} ({agg.dataset}): "
                f"{agg.detection_mean*100:.1f}% ± {agg.detection_std*100:.1f}%"
            )
        lines.append("")

        # Key findings
        lines.append("Key Findings:")

        # PoGQ performance
        pogq_results = [a for a in self.aggregated.values() if a.defense == "pogq_v4_1"]
        if pogq_results:
            pogq_mean_dr = np.mean([a.detection_mean for a in pogq_results])
            pogq_mean_fpr = np.mean([a.fpr_mean for a in pogq_results])
            lines.append(
                f"  - PoGQ v4.1: {pogq_mean_dr*100:.1f}% detection, "
                f"{pogq_mean_fpr*100:.1f}% FPR (avg across all attacks)"
            )

        # FedAvg baseline
        fedavg_results = [a for a in self.aggregated.values() if a.defense == "fedavg"]
        if fedavg_results:
            fedavg_mean_dr = np.mean([a.detection_mean for a in fedavg_results])
            lines.append(f"  - FedAvg (baseline): {fedavg_mean_dr*100:.1f}% detection")

        lines.append("")
        lines.append("=" * 80)

        report = "\n".join(lines)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(report)

        print(report)
        print(f"✅ Summary written to {output_file}\n")

    def save_aggregated_json(self, output_file: Path):
        """Save aggregated results as JSON"""
        print(f"Saving aggregated JSON: {output_file}")

        data = {}
        for key, agg in self.aggregated.items():
            key_str = f"{key[0]}_{key[1]}_{key[2]}"
            data[key_str] = {
                "dataset": agg.dataset,
                "attack": agg.attack,
                "defense": agg.defense,
                "n_seeds": agg.n_seeds,
                "detection": {"mean": agg.detection_mean, "std": agg.detection_std},
                "fpr": {"mean": agg.fpr_mean, "std": agg.fpr_std},
                "precision": {"mean": agg.precision_mean, "std": agg.precision_std},
                "accuracy": {"mean": agg.accuracy_mean, "std": agg.accuracy_std},
                "seeds_tested": agg.seeds_tested
            }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ JSON written to {output_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate v4.1 experiment results")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing experiment artifacts"
    )
    parser.add_argument(
        "--output-dir",
        default="results/v4.1/aggregated",
        help="Output directory for aggregated results"
    )
    args = parser.parse_args()

    analyzer = V4_1_Analyzer(results_dir=args.results_dir)

    # Load experiments
    n_loaded = analyzer.load_experiments()

    if n_loaded == 0:
        print("❌ No experiments found. Exiting.")
        sys.exit(1)

    # Aggregate
    analyzer.aggregate_by_configuration()

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate attack-defense matrices
    for dataset in ["mnist", "mnist_noniid"]:
        matrix = analyzer.generate_attack_defense_matrix(dataset)

        # LaTeX table
        latex_file = output_dir / f"attack_defense_matrix_{dataset}.tex"
        analyzer.generate_latex_table(matrix, latex_file)

        # JSON
        json_file = output_dir / f"attack_defense_matrix_{dataset}.json"
        with open(json_file, 'w') as f:
            # Convert tuple keys to strings
            matrix_serializable = {
                "dataset": matrix["dataset"],
                "attacks": matrix["attacks"],
                "defenses": matrix["defenses"],
                "detection_rates": {
                    f"{k[0]}_{k[1]}": v for k, v in matrix["detection_rates"].items()
                },
                "fprs": {
                    f"{k[0]}_{k[1]}": v for k, v in matrix["fprs"].items()
                },
                "accuracies": {
                    f"{k[0]}_{k[1]}": v for k, v in matrix["accuracies"].items()
                }
            }
            json.dump(matrix_serializable, f, indent=2)
        print(f"✅ Matrix JSON: {json_file}\n")

    # Summary report
    summary_file = output_dir / "summary_report.txt"
    analyzer.generate_summary_report(summary_file)

    # Full aggregated JSON
    json_file = output_dir / "aggregated_results.json"
    analyzer.save_aggregated_json(json_file)

    print("=" * 80)
    print("✅ Analysis Complete!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Review: cat results/v4.1/aggregated/summary_report.txt")
    print("  2. LaTeX: Copy tables to paper sections/05-results.tex")
    print("  3. Figures: Use JSON data for plotting")
    print()


if __name__ == "__main__":
    main()
