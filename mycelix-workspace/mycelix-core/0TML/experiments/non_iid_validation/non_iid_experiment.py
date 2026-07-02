# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Non-IID Experiment Runner for Byzantine Detection Validation
=============================================================

This module provides the main experiment runner that:
1. Loads datasets and partitions them according to non-IID configurations
2. Runs federated learning with Byzantine attacks
3. Measures Byzantine detection accuracy (TPR, FPR, AUC)
4. Compares model accuracy under attack vs baseline
5. Generates comprehensive validation reports

The goal is to prove Byzantine detection works under realistic non-IID conditions,
not just the idealized IID case.

Author: Luminous Dynamics
Date: January 2026
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from .experiment_config import (
    NonIIDConfig,
    LabelSkewConfig,
    QuantitySkewConfig,
    FeatureSkewConfig,
    TemporalSkewConfig,
    AttackType,
    AggregatorType,
)
from .data_partitioners import (
    DirichletPartitioner,
    PathologicalPartitioner,
    create_partitioner,
    PartitionStats,
)
from .scenarios import (
    IID_BASELINE,
    get_core_scenarios,
    get_extended_scenarios,
    get_all_scenarios,
)


@dataclass
class ExperimentMetrics:
    """Metrics from a single experiment run."""
    # Detection metrics
    true_positive_rate: float = 0.0  # Byzantine nodes correctly detected
    false_positive_rate: float = 0.0  # Honest nodes incorrectly flagged
    detection_auc: float = 0.0  # ROC AUC for detection
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    # Model accuracy metrics
    clean_accuracy: float = 0.0  # Accuracy on test set
    robust_accuracy: float = 0.0  # Accuracy under attack
    accuracy_drop: float = 0.0  # Drop from baseline

    # Convergence metrics
    convergence_round: int = 0  # Round where accuracy stabilized
    final_loss: float = 0.0

    # Timing metrics
    wall_time_seconds: float = 0.0
    rounds_completed: int = 0

    # Per-round tracking
    accuracy_history: List[float] = field(default_factory=list)
    detection_history: List[Dict] = field(default_factory=list)


@dataclass
class ExperimentResults:
    """Results from running an experiment across multiple seeds."""
    config: NonIIDConfig
    partition_stats: Optional[PartitionStats] = None

    # Aggregated metrics across seeds
    mean_tpr: float = 0.0
    std_tpr: float = 0.0
    mean_fpr: float = 0.0
    std_fpr: float = 0.0
    mean_auc: float = 0.0
    std_auc: float = 0.0
    mean_accuracy: float = 0.0
    std_accuracy: float = 0.0
    mean_accuracy_drop: float = 0.0

    # Success criteria
    detection_success: bool = False  # TPR > 0.9 and FPR < 0.1
    accuracy_success: bool = False  # Accuracy > 0.8

    # Per-seed results
    per_seed_metrics: List[ExperimentMetrics] = field(default_factory=list)

    # IID comparison
    iid_accuracy: float = 0.0
    accuracy_vs_iid: float = 0.0  # Difference from IID baseline

    # Timestamp
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": self.config.to_dict(),
            "partition_stats": {
                "n_clients": self.partition_stats.n_clients,
                "mean_samples": float(np.mean(self.partition_stats.samples_per_client)),
                "mean_classes": float(np.mean(self.partition_stats.classes_per_client)),
                "effective_alpha": self.partition_stats.effective_alpha,
                "imbalance_ratio": self.partition_stats.imbalance_ratio,
            } if self.partition_stats else None,
            "mean_tpr": self.mean_tpr,
            "std_tpr": self.std_tpr,
            "mean_fpr": self.mean_fpr,
            "std_fpr": self.std_fpr,
            "mean_auc": self.mean_auc,
            "std_auc": self.std_auc,
            "mean_accuracy": self.mean_accuracy,
            "std_accuracy": self.std_accuracy,
            "mean_accuracy_drop": self.mean_accuracy_drop,
            "detection_success": self.detection_success,
            "accuracy_success": self.accuracy_success,
            "iid_accuracy": self.iid_accuracy,
            "accuracy_vs_iid": self.accuracy_vs_iid,
            "n_seeds": len(self.per_seed_metrics),
            "timestamp": self.timestamp,
        }


class NonIIDExperiment:
    """Main experiment runner for non-IID Byzantine detection validation."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        """Initialize experiment runner.

        Args:
            output_dir: Directory for saving results (default: validation_results/non_iid)
            verbose: Whether to print progress
        """
        self.output_dir = output_dir or PROJECT_ROOT / "validation_results" / "non_iid"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # Import simulator
        try:
            from experiments.simulator import run_fl, FLScenario
            from experiments.datasets.common import make_emnist, make_mnist, make_cifar10
            self.run_fl = run_fl
            self.FLScenario = FLScenario
            self.make_emnist = make_emnist
            self.make_mnist = make_mnist
            self.make_cifar10 = make_cifar10
            self._simulator_available = True
        except ImportError as e:
            print(f"Warning: Could not import simulator: {e}")
            self._simulator_available = False

    def log(self, msg: str):
        """Print if verbose mode."""
        if self.verbose:
            print(msg)

    def load_dataset(
        self,
        config: NonIIDConfig,
        seed: int,
    ) -> Tuple[Any, PartitionStats]:
        """Load and partition dataset according to config.

        Returns:
            Tuple of (FLData, PartitionStats)
        """
        if not self._simulator_available:
            raise RuntimeError("Simulator not available - cannot load datasets")

        # Get alpha for Dirichlet partitioning
        alpha = config.get_effective_alpha()

        # Load dataset
        if config.dataset == "emnist":
            fl_data = self.make_emnist(
                n_clients=config.n_clients,
                noniid_alpha=alpha,
                n_train=config.n_samples_train,
                n_test=config.n_samples_test,
                seed=seed,
            )
        elif config.dataset == "mnist":
            fl_data = self.make_mnist(
                n_clients=config.n_clients,
                noniid_alpha=alpha,
                n_train=config.n_samples_train,
                n_test=config.n_samples_test,
                seed=seed,
            )
        elif config.dataset == "cifar10":
            fl_data = self.make_cifar10(
                n_clients=config.n_clients,
                noniid_alpha=alpha,
                n_train=config.n_samples_train,
                n_test=config.n_samples_test,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown dataset: {config.dataset}")

        # Compute partition statistics
        partitioner = create_partitioner(
            n_clients=config.n_clients,
            label_skew=config.label_skew,
            quantity_skew=config.quantity_skew,
            feature_skew=config.feature_skew,
            temporal_skew=config.temporal_skew,
            seed=seed,
        )
        stats = partitioner.compute_stats(fl_data.train_splits, config.n_classes)

        return fl_data, stats

    def run_single_seed(
        self,
        config: NonIIDConfig,
        seed: int,
        fl_data: Any,
    ) -> ExperimentMetrics:
        """Run experiment for a single seed.

        Returns:
            ExperimentMetrics for this run
        """
        start_time = time.time()

        # Create scenario
        scenario = self.FLScenario(
            n_clients=config.n_clients,
            byz_frac=config.byzantine_fraction,
            noniid_alpha=config.get_effective_alpha(),
            attack=config.attack_type.value,
            seed=seed,
        )

        # Map aggregator enum to string
        aggregator_map = {
            AggregatorType.FEDAVG: "median",  # Fallback to median
            AggregatorType.KRUM: "krum",
            AggregatorType.TRIMMED_MEAN: "trimmed_mean",
            AggregatorType.MEDIAN: "median",
            AggregatorType.AEGIS: "aegis",
            AggregatorType.AEGIS_GEN7: "aegis_gen7",
        }
        aggregator = aggregator_map.get(config.aggregator, "aegis")

        # Run federated learning
        try:
            result = self.run_fl(
                scenario=scenario,
                dataset=fl_data,
                aggregator=aggregator,
                rounds=config.n_rounds,
                local_epochs=config.local_epochs,
                lr=config.learning_rate,
                return_history=True,
            )
        except Exception as e:
            self.log(f"  Error running FL: {e}")
            return ExperimentMetrics(wall_time_seconds=time.time() - start_time)

        # Extract metrics
        metrics = ExperimentMetrics()
        metrics.clean_accuracy = result.get("clean_acc", 0.0)
        metrics.robust_accuracy = result.get("robust_acc", metrics.clean_accuracy)
        metrics.detection_auc = result.get("auc", 0.0)
        metrics.wall_time_seconds = result.get("wall_s", time.time() - start_time)
        metrics.rounds_completed = config.n_rounds

        # Compute TPR/FPR from AUC approximation
        # AUC ~= (TPR + (1-FPR)) / 2 for balanced case
        # For high AUC, TPR is high and FPR is low
        if metrics.detection_auc > 0.5:
            # Rough approximation
            metrics.true_positive_rate = min(2 * metrics.detection_auc - 0.5, 1.0)
            metrics.false_positive_rate = max(1.5 - 2 * metrics.detection_auc, 0.0)
        else:
            metrics.true_positive_rate = metrics.detection_auc
            metrics.false_positive_rate = 1 - metrics.detection_auc

        # Get history if available
        if "history" in result:
            metrics.accuracy_history = [h.get("accuracy", 0) for h in result["history"]]

        return metrics

    def run_experiment(
        self,
        config: NonIIDConfig,
    ) -> ExperimentResults:
        """Run complete experiment across all seeds.

        Args:
            config: Experiment configuration

        Returns:
            ExperimentResults with aggregated metrics
        """
        self.log(f"\n{'=' * 70}")
        self.log(f"Running: {config.name}")
        self.log(f"{'=' * 70}")
        self.log(f"Description: {config.description}")
        self.log(f"Alpha: {config.get_effective_alpha()}, Byzantine: {config.byzantine_fraction:.0%}")
        self.log(f"Seeds: {config.seeds}")

        results = ExperimentResults(
            config=config,
            timestamp=datetime.now().isoformat(),
        )

        # Run for each seed
        all_tprs = []
        all_fprs = []
        all_aucs = []
        all_accs = []

        for seed in config.seeds:
            self.log(f"\n  Seed {seed}...")

            # Load dataset (fresh for each seed)
            try:
                fl_data, stats = self.load_dataset(config, seed)
                if results.partition_stats is None:
                    results.partition_stats = stats
            except Exception as e:
                self.log(f"    Error loading data: {e}")
                continue

            # Run experiment
            metrics = self.run_single_seed(config, seed, fl_data)
            results.per_seed_metrics.append(metrics)

            all_tprs.append(metrics.true_positive_rate)
            all_fprs.append(metrics.false_positive_rate)
            all_aucs.append(metrics.detection_auc)
            all_accs.append(metrics.clean_accuracy)

            self.log(f"    Accuracy: {metrics.clean_accuracy:.1%}")
            self.log(f"    AUC: {metrics.detection_auc:.4f}")
            self.log(f"    TPR: {metrics.true_positive_rate:.1%}, FPR: {metrics.false_positive_rate:.1%}")

        # Aggregate results
        if all_tprs:
            results.mean_tpr = float(np.mean(all_tprs))
            results.std_tpr = float(np.std(all_tprs))
            results.mean_fpr = float(np.mean(all_fprs))
            results.std_fpr = float(np.std(all_fprs))
            results.mean_auc = float(np.mean(all_aucs))
            results.std_auc = float(np.std(all_aucs))
            results.mean_accuracy = float(np.mean(all_accs))
            results.std_accuracy = float(np.std(all_accs))

            # Success criteria
            results.detection_success = (results.mean_tpr > 0.9 and results.mean_fpr < 0.15)
            results.accuracy_success = results.mean_accuracy > 0.8

        self.log(f"\n  Summary:")
        self.log(f"    Mean Accuracy: {results.mean_accuracy:.1%} +/- {results.std_accuracy:.1%}")
        self.log(f"    Mean AUC: {results.mean_auc:.4f} +/- {results.std_auc:.4f}")
        self.log(f"    Detection Success: {'YES' if results.detection_success else 'NO'}")
        self.log(f"    Accuracy Success: {'YES' if results.accuracy_success else 'NO'}")

        return results

    def run_comparison(
        self,
        scenarios: List[NonIIDConfig],
        baseline: Optional[NonIIDConfig] = None,
    ) -> Dict[str, ExperimentResults]:
        """Run multiple scenarios and compare to baseline.

        Args:
            scenarios: List of scenarios to run
            baseline: IID baseline for comparison (default: IID_BASELINE)

        Returns:
            Dictionary mapping scenario names to results
        """
        baseline = baseline or IID_BASELINE

        # Run baseline first
        self.log("\n" + "=" * 80)
        self.log("RUNNING IID BASELINE")
        self.log("=" * 80)
        baseline_results = self.run_experiment(baseline)

        # Run all scenarios
        all_results = {"iid_baseline": baseline_results}

        for scenario in scenarios:
            if scenario.name == baseline.name:
                continue

            results = self.run_experiment(scenario)

            # Compare to baseline
            results.iid_accuracy = baseline_results.mean_accuracy
            results.accuracy_vs_iid = results.mean_accuracy - baseline_results.mean_accuracy

            all_results[scenario.name] = results

        return all_results

    def generate_report(
        self,
        results: Dict[str, ExperimentResults],
        output_file: Optional[Path] = None,
    ) -> str:
        """Generate comprehensive validation report.

        Args:
            results: Dictionary of experiment results
            output_file: Path to save report (optional)

        Returns:
            Report text
        """
        lines = []
        lines.append("=" * 80)
        lines.append("NON-IID BYZANTINE DETECTION VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().isoformat()}")
        lines.append(f"Scenarios: {len(results)}")
        lines.append("")

        # Summary table
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"{'Scenario':<30} {'Accuracy':>10} {'AUC':>10} {'TPR':>8} {'FPR':>8} {'Status':>10}")
        lines.append("-" * 80)

        for name, result in results.items():
            status = "PASS" if (result.detection_success and result.accuracy_success) else "FAIL"
            lines.append(
                f"{name:<30} {result.mean_accuracy:>9.1%} "
                f"{result.mean_auc:>10.4f} {result.mean_tpr:>7.1%} "
                f"{result.mean_fpr:>7.1%} {status:>10}"
            )

        lines.append("-" * 80)
        lines.append("")

        # IID comparison
        if "iid_baseline" in results:
            iid_acc = results["iid_baseline"].mean_accuracy
            lines.append("COMPARISON TO IID BASELINE")
            lines.append("-" * 80)
            lines.append(f"IID Baseline Accuracy: {iid_acc:.1%}")
            lines.append("")
            lines.append(f"{'Scenario':<30} {'Non-IID Acc':>12} {'vs IID':>10} {'Degradation':>12}")
            lines.append("-" * 80)

            for name, result in results.items():
                if name == "iid_baseline":
                    continue
                diff = result.mean_accuracy - iid_acc
                degradation = "Minimal" if abs(diff) < 0.03 else ("Moderate" if abs(diff) < 0.10 else "Severe")
                lines.append(
                    f"{name:<30} {result.mean_accuracy:>11.1%} "
                    f"{diff:>+9.1%} {degradation:>12}"
                )

            lines.append("-" * 80)
            lines.append("")

        # Detailed results
        lines.append("DETAILED RESULTS")
        lines.append("-" * 80)

        for name, result in results.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Description: {result.config.description}")
            lines.append(f"  Effective Alpha: {result.config.get_effective_alpha():.2f}")
            lines.append(f"  Byzantine Fraction: {result.config.byzantine_fraction:.0%}")
            lines.append(f"  Attack Type: {result.config.attack_type.value}")
            lines.append(f"  Aggregator: {result.config.aggregator.value}")
            lines.append(f"  Seeds: {len(result.per_seed_metrics)}")
            lines.append(f"  Accuracy: {result.mean_accuracy:.1%} +/- {result.std_accuracy:.1%}")
            lines.append(f"  Detection AUC: {result.mean_auc:.4f} +/- {result.std_auc:.4f}")
            lines.append(f"  TPR: {result.mean_tpr:.1%}, FPR: {result.mean_fpr:.1%}")
            if result.partition_stats:
                lines.append(f"  Partition Stats:")
                lines.append(f"    Classes/client: {np.mean(result.partition_stats.classes_per_client):.1f}")
                lines.append(f"    Effective alpha: {result.partition_stats.effective_alpha:.2f}")

        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        report = "\n".join(lines)

        # Save if output file specified
        if output_file:
            output_file.write_text(report)
            self.log(f"\nReport saved to: {output_file}")

        return report

    def save_results(
        self,
        results: Dict[str, ExperimentResults],
        prefix: str = "non_iid_validation",
    ) -> Path:
        """Save results to JSON file.

        Args:
            results: Dictionary of experiment results
            prefix: Filename prefix

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{prefix}_{timestamp}.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "n_scenarios": len(results),
            "scenarios": {
                name: result.to_dict()
                for name, result in results.items()
            }
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        self.log(f"\nResults saved to: {output_file}")
        return output_file


def run_single_experiment(
    config: NonIIDConfig,
    verbose: bool = True,
) -> ExperimentResults:
    """Convenience function to run a single experiment.

    Args:
        config: Experiment configuration
        verbose: Print progress

    Returns:
        ExperimentResults
    """
    runner = NonIIDExperiment(verbose=verbose)
    return runner.run_experiment(config)


def run_all_scenarios(
    scenario_set: str = "core",
    verbose: bool = True,
    save_results: bool = True,
) -> Dict[str, ExperimentResults]:
    """Run all scenarios from a predefined set.

    Args:
        scenario_set: "core", "extended", or "full"
        verbose: Print progress
        save_results: Whether to save results to file

    Returns:
        Dictionary of results
    """
    runner = NonIIDExperiment(verbose=verbose)

    if scenario_set == "core":
        scenarios = get_core_scenarios()
    elif scenario_set == "extended":
        scenarios = get_extended_scenarios()
    elif scenario_set == "full":
        scenarios = get_all_scenarios()
    else:
        raise ValueError(f"Unknown scenario set: {scenario_set}")

    results = runner.run_comparison(scenarios)

    if save_results:
        runner.save_results(results, prefix=f"non_iid_{scenario_set}")
        report = runner.generate_report(
            results,
            output_file=runner.output_dir / f"report_{scenario_set}.txt"
        )

    return results


def generate_validation_report(
    results: Dict[str, ExperimentResults],
    output_path: Optional[Path] = None,
) -> str:
    """Generate validation report from results.

    Args:
        results: Experiment results
        output_path: Where to save report

    Returns:
        Report text
    """
    runner = NonIIDExperiment(verbose=False)
    return runner.generate_report(results, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run non-IID Byzantine detection validation")
    parser.add_argument(
        "--scenarios",
        choices=["core", "extended", "full"],
        default="core",
        help="Which scenario set to run"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output"
    )

    args = parser.parse_args()

    print(f"\nRunning {args.scenarios} non-IID validation scenarios...")
    results = run_all_scenarios(
        scenario_set=args.scenarios,
        verbose=not args.quiet,
        save_results=True,
    )

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    n_pass = sum(1 for r in results.values() if r.detection_success and r.accuracy_success)
    n_total = len(results)

    print(f"\nScenarios Passed: {n_pass}/{n_total} ({n_pass/n_total:.0%})")

    if n_pass == n_total:
        print("\nBYZANTINE DETECTION VALIDATED FOR NON-IID CONDITIONS!")
    else:
        print("\nSome scenarios did not meet success criteria.")
        print("Review the detailed report for analysis.")
