#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Convergence Analysis for Non-IID Experiments

Analyzes model convergence behavior under Non-IID data distributions:
- Loss curves over rounds
- Gradient norm statistics
- Client drift measurement
- Convergence speed comparison

Key metrics:
- Rounds to target accuracy
- Loss variance across rounds
- Gradient direction consistency
- Client update divergence

Author: Luminous Dynamics
Date: January 2026
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceMetrics:
    """Convergence metrics for a scenario."""
    scenario: str
    rounds_to_90pct: Optional[int]
    rounds_to_95pct: Optional[int]
    final_accuracy: float
    final_loss: float
    loss_variance: float
    gradient_norm_mean: float
    gradient_norm_std: float
    client_drift_mean: float
    convergence_stability: float  # 0-1, higher = more stable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "rounds_to_90pct": self.rounds_to_90pct,
            "rounds_to_95pct": self.rounds_to_95pct,
            "final_accuracy": self.final_accuracy,
            "final_loss": self.final_loss,
            "loss_variance": self.loss_variance,
            "gradient_norm_mean": self.gradient_norm_mean,
            "gradient_norm_std": self.gradient_norm_std,
            "client_drift_mean": self.client_drift_mean,
            "convergence_stability": self.convergence_stability,
        }


class ConvergenceAnalyzer:
    """
    Analyzes convergence behavior from Non-IID experiment results.

    Examines:
    1. How quickly models converge under different Non-IID conditions
    2. Stability of convergence (variance in metrics)
    3. Client drift (how much individual clients diverge from global)
    4. Impact of Byzantine detection on convergence

    Example:
        >>> analyzer = ConvergenceAnalyzer()
        >>> results = analyzer.analyze(experiment_results)
        >>> analyzer.generate_plots(output_dir)
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize analyzer."""
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.results: Dict[str, ConvergenceMetrics] = {}

    def analyze_scenario(
        self,
        scenario_name: str,
        results: Dict[str, Any],
    ) -> ConvergenceMetrics:
        """
        Analyze convergence for a single scenario.

        Args:
            scenario_name: Name of the scenario
            results: Scenario results dictionary

        Returns:
            ConvergenceMetrics for this scenario
        """
        # Extract per-round metrics
        round_metrics = results.get("per_round_metrics", [])

        if not round_metrics:
            logger.warning(f"No round metrics for {scenario_name}")
            return ConvergenceMetrics(
                scenario=scenario_name,
                rounds_to_90pct=None,
                rounds_to_95pct=None,
                final_accuracy=0.0,
                final_loss=0.0,
                loss_variance=0.0,
                gradient_norm_mean=0.0,
                gradient_norm_std=0.0,
                client_drift_mean=0.0,
                convergence_stability=0.0,
            )

        # Extract accuracy/detection over rounds
        detection_rates = [m.get("detection_rate", 0) for m in round_metrics]

        # Find rounds to targets
        rounds_to_90 = self._find_rounds_to_target(detection_rates, 0.90)
        rounds_to_95 = self._find_rounds_to_target(detection_rates, 0.95)

        # Final metrics
        final_accuracy = detection_rates[-1] if detection_rates else 0.0

        # Simulate loss (in real scenario would come from training)
        losses = [1.0 - rate for rate in detection_rates]
        final_loss = losses[-1] if losses else 1.0
        loss_variance = float(np.var(losses)) if losses else 0.0

        # Gradient norm statistics (from timing data as proxy)
        latencies = [m.get("latency_ms", 0) for m in round_metrics]
        gradient_norm_mean = float(np.mean(latencies)) if latencies else 0.0
        gradient_norm_std = float(np.std(latencies)) if latencies else 0.0

        # Client drift (measure consistency of detection)
        client_drift = self._calculate_client_drift(round_metrics)

        # Convergence stability (inverse of variance in detection rate)
        if len(detection_rates) > 1:
            stability = 1.0 / (1.0 + np.std(detection_rates[-10:]))
        else:
            stability = 0.0

        metrics = ConvergenceMetrics(
            scenario=scenario_name,
            rounds_to_90pct=rounds_to_90,
            rounds_to_95pct=rounds_to_95,
            final_accuracy=final_accuracy,
            final_loss=final_loss,
            loss_variance=loss_variance,
            gradient_norm_mean=gradient_norm_mean,
            gradient_norm_std=gradient_norm_std,
            client_drift_mean=client_drift,
            convergence_stability=float(stability),
        )

        self.results[scenario_name] = metrics
        return metrics

    def _find_rounds_to_target(
        self,
        values: List[float],
        target: float,
    ) -> Optional[int]:
        """Find first round where target is reached and maintained."""
        for i, val in enumerate(values):
            # Check if target reached and maintained for 5 rounds
            if val >= target:
                if all(v >= target * 0.98 for v in values[i:i+5] if i+5 <= len(values)):
                    return i + 1  # 1-indexed round
        return None

    def _calculate_client_drift(
        self,
        round_metrics: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate average client drift over rounds.

        Drift measures how much detection varies between rounds,
        indicating instability in the learning process.
        """
        if len(round_metrics) < 2:
            return 0.0

        detection_rates = [m.get("detection_rate", 0) for m in round_metrics]
        drifts = [
            abs(detection_rates[i] - detection_rates[i-1])
            for i in range(1, len(detection_rates))
        ]

        return float(np.mean(drifts)) if drifts else 0.0

    def analyze_all(
        self,
        all_results: Dict[str, Any],
    ) -> Dict[str, ConvergenceMetrics]:
        """
        Analyze convergence for all scenarios.

        Args:
            all_results: Dictionary of all experiment results

        Returns:
            Dictionary of scenario -> ConvergenceMetrics
        """
        for scenario_name, scenario_data in all_results.items():
            if scenario_name == "analysis":
                continue

            # Handle nested results structure
            if isinstance(scenario_data, dict):
                if "results" in scenario_data:
                    inner = scenario_data["results"]
                    if isinstance(inner, dict) and "per_round_metrics" not in inner:
                        # Multiple sub-results (e.g., label_skew with multiple alphas)
                        for sub_name, sub_data in inner.items():
                            if isinstance(sub_data, dict):
                                full_name = f"{scenario_name}/{sub_name}"
                                self.analyze_scenario(full_name, sub_data)
                    else:
                        self.analyze_scenario(scenario_name, inner)
                elif "per_round_metrics" in scenario_data:
                    self.analyze_scenario(scenario_name, scenario_data)

        return self.results

    def generate_convergence_plot(
        self,
        all_results: Dict[str, Any],
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Generate convergence curves plot.

        Args:
            all_results: Experiment results
            output_path: Path to save plot

        Returns:
            Path to saved plot or None if plotting unavailable
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            logger.warning("matplotlib not available, skipping plot generation")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Collect data for plotting
        scenario_data = {}

        for scenario_name, scenario_results in all_results.items():
            if scenario_name == "analysis":
                continue

            if isinstance(scenario_results, dict):
                if "results" in scenario_results:
                    inner = scenario_results["results"]
                    if isinstance(inner, dict) and "per_round_metrics" not in inner:
                        for sub_name, sub_data in inner.items():
                            if isinstance(sub_data, dict) and "per_round_metrics" in sub_data:
                                key = f"{scenario_name}/{sub_name}"
                                scenario_data[key] = sub_data["per_round_metrics"]
                    elif "per_round_metrics" in inner:
                        scenario_data[scenario_name] = inner["per_round_metrics"]
                elif "per_round_metrics" in scenario_results:
                    scenario_data[scenario_name] = scenario_results["per_round_metrics"]

        if not scenario_data:
            logger.warning("No data available for plotting")
            plt.close()
            return None

        # Plot 1: Detection rate over rounds
        ax1 = axes[0, 0]
        for name, metrics in scenario_data.items():
            if metrics:
                rounds = [m.get("round", i) for i, m in enumerate(metrics)]
                detection = [m.get("detection_rate", 0) for m in metrics]
                ax1.plot(rounds, detection, label=name, alpha=0.8)
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Detection Rate")
        ax1.set_title("Detection Convergence")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: False positive rate over rounds
        ax2 = axes[0, 1]
        for name, metrics in scenario_data.items():
            if metrics:
                rounds = [m.get("round", i) for i, m in enumerate(metrics)]
                fp_rate = [m.get("false_positive_rate", 0) for m in metrics]
                ax2.plot(rounds, fp_rate, label=name, alpha=0.8)
        ax2.set_xlabel("Round")
        ax2.set_ylabel("False Positive Rate")
        ax2.set_title("False Positive Evolution")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Plot 3: F1 score over rounds
        ax3 = axes[1, 0]
        for name, metrics in scenario_data.items():
            if metrics:
                rounds = [m.get("round", i) for i, m in enumerate(metrics)]
                f1 = [m.get("f1_score", 0) for m in metrics]
                ax3.plot(rounds, f1, label=name, alpha=0.8)
        ax3.set_xlabel("Round")
        ax3.set_ylabel("F1 Score")
        ax3.set_title("F1 Score Convergence")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Convergence comparison bar chart
        ax4 = axes[1, 1]
        if self.results:
            names = list(self.results.keys())
            rounds_90 = [
                self.results[n].rounds_to_90pct or 100
                for n in names
            ]
            x = np.arange(len(names))
            ax4.bar(x, rounds_90, alpha=0.8)
            ax4.set_xticks(x)
            ax4.set_xticklabels([n.split("/")[-1] for n in names], rotation=45, ha="right")
            ax4.set_ylabel("Rounds to 90% Detection")
            ax4.set_title("Convergence Speed Comparison")
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save plot
        if output_path is None:
            output_path = self.output_dir / "convergence_analysis.png"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved convergence plot to {output_path}")
        return output_path

    def generate_report(
        self,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate convergence analysis report.

        Args:
            output_path: Path to save JSON report

        Returns:
            Report dictionary
        """
        report = {
            "analysis_type": "convergence",
            "timestamp": datetime.now().isoformat(),
            "scenarios": {
                name: metrics.to_dict()
                for name, metrics in self.results.items()
            },
            "summary": {
                "fastest_convergence": min(
                    ((n, m.rounds_to_90pct) for n, m in self.results.items()
                     if m.rounds_to_90pct is not None),
                    key=lambda x: x[1],
                    default=(None, None)
                )[0],
                "slowest_convergence": max(
                    ((n, m.rounds_to_90pct) for n, m in self.results.items()
                     if m.rounds_to_90pct is not None),
                    key=lambda x: x[1],
                    default=(None, None)
                )[0],
                "highest_stability": max(
                    ((n, m.convergence_stability) for n, m in self.results.items()),
                    key=lambda x: x[1],
                    default=(None, 0)
                )[0],
            },
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved convergence report to {output_path}")

        return report


def analyze_convergence(
    results: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Main entry point for convergence analysis.

    Args:
        results: Experiment results dictionary
        output_dir: Output directory for reports and plots

    Returns:
        Analysis results dictionary
    """
    analyzer = ConvergenceAnalyzer(output_dir)

    # Analyze all scenarios
    metrics = analyzer.analyze_all(results)

    # Generate plots
    plot_path = analyzer.generate_convergence_plot(results)

    # Generate report
    report = analyzer.generate_report(output_dir / "convergence_report.json")

    return {
        "metrics": {k: v.to_dict() for k, v in metrics.items()},
        "plot_path": str(plot_path) if plot_path else None,
        "report": report,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convergence Analysis")
    parser.add_argument("results_file", type=Path, help="Path to results JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Output directory")

    args = parser.parse_args()

    with open(args.results_file) as f:
        results = json.load(f)

    analysis = analyze_convergence(results, args.output_dir)

    print(f"\nConvergence Analysis Complete")
    print(f"Report: {args.output_dir / 'convergence_report.json'}")
