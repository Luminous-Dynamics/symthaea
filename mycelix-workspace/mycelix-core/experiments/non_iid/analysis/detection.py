#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Byzantine Detection Analysis for Non-IID Experiments

Analyzes Byzantine detection performance under Non-IID conditions:
- Detection accuracy by data distribution
- False positive analysis
- Attack-specific detection rates
- ROC curves and confusion matrices

Key questions answered:
1. Does Non-IID data increase false positives?
2. Which attack types are harder to detect under Non-IID?
3. How does label skew vs quantity skew affect detection?

Author: Luminous Dynamics
Date: January 2026
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionMetrics:
    """Detection metrics for a scenario."""
    scenario: str
    true_positive_rate: float
    false_positive_rate: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    auc_roc: Optional[float]
    detection_by_attack: Dict[str, float]
    fp_by_distribution: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "true_positive_rate": self.true_positive_rate,
            "false_positive_rate": self.false_positive_rate,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "accuracy": self.accuracy,
            "auc_roc": self.auc_roc,
            "detection_by_attack": self.detection_by_attack,
            "fp_by_distribution": self.fp_by_distribution,
        }


class DetectionAnalyzer:
    """
    Analyzes Byzantine detection accuracy under Non-IID conditions.

    Examines:
    1. Overall detection rates (TP, FP, TN, FN)
    2. Detection rates by attack type
    3. False positive rates by data distribution
    4. Comparison across Non-IID scenarios

    Example:
        >>> analyzer = DetectionAnalyzer()
        >>> results = analyzer.analyze(experiment_results)
        >>> analyzer.generate_confusion_matrix(output_dir)
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize analyzer."""
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.results: Dict[str, DetectionMetrics] = {}

    def analyze_scenario(
        self,
        scenario_name: str,
        results: Dict[str, Any],
    ) -> DetectionMetrics:
        """
        Analyze detection for a single scenario.

        Args:
            scenario_name: Name of the scenario
            results: Scenario results dictionary

        Returns:
            DetectionMetrics for this scenario
        """
        # Extract detection metrics
        tp_rate = results.get("detection_accuracy", results.get("recall", 0))
        fp_rate = results.get("false_positive_rate", 0)
        precision = results.get("precision", 0)
        recall = results.get("recall", tp_rate)
        f1 = results.get("f1_score", 0)
        accuracy = results.get("accuracy", 0)

        # Detection by attack type (if available)
        detection_by_attack = {}
        attack_data = results.get("detection_by_attack", {})
        if attack_data:
            detection_by_attack = attack_data

        # FP by distribution (size, transformation type, etc.)
        fp_by_distribution = {}

        # Check for size-based analysis
        size_analysis = results.get("size_analysis", {})
        for size, data in size_analysis.items():
            if isinstance(data, dict):
                fp_by_distribution[f"size_{size}"] = data.get("fp_rate", 0)

        # Check for skew-based analysis
        skew_analysis = results.get("skew_analysis", {})
        for skew_level, data in skew_analysis.items():
            if isinstance(data, dict):
                fp_by_distribution[f"skew_{skew_level}"] = data.get("fp_rate", 0)

        # Check for transformation-based analysis
        fp_by_transform = results.get("fp_by_transformation", {})
        for t_type, count in fp_by_transform.items():
            fp_by_distribution[f"transform_{t_type}"] = count

        metrics = DetectionMetrics(
            scenario=scenario_name,
            true_positive_rate=tp_rate,
            false_positive_rate=fp_rate,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            auc_roc=None,  # Calculated separately if probability scores available
            detection_by_attack=detection_by_attack,
            fp_by_distribution=fp_by_distribution,
        )

        self.results[scenario_name] = metrics
        return metrics

    def analyze_all(
        self,
        all_results: Dict[str, Any],
    ) -> Dict[str, DetectionMetrics]:
        """
        Analyze detection for all scenarios.

        Args:
            all_results: Dictionary of all experiment results

        Returns:
            Dictionary of scenario -> DetectionMetrics
        """
        for scenario_name, scenario_data in all_results.items():
            if scenario_name == "analysis":
                continue

            if isinstance(scenario_data, dict):
                if "results" in scenario_data:
                    inner = scenario_data["results"]
                    if isinstance(inner, dict) and "detection_accuracy" not in inner:
                        # Multiple sub-results
                        for sub_name, sub_data in inner.items():
                            if isinstance(sub_data, dict):
                                full_name = f"{scenario_name}/{sub_name}"
                                self.analyze_scenario(full_name, sub_data)
                    else:
                        self.analyze_scenario(scenario_name, inner)
                elif "detection_accuracy" in scenario_data:
                    self.analyze_scenario(scenario_name, scenario_data)

        return self.results

    def compare_iid_vs_non_iid(
        self,
        iid_baseline: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Compare Non-IID results against IID baseline.

        Args:
            iid_baseline: IID baseline metrics (default: 100% detection, 0% FP)

        Returns:
            Comparison dictionary
        """
        if iid_baseline is None:
            # Default: perfect IID performance
            iid_baseline = {
                "detection_rate": 1.0,
                "false_positive_rate": 0.0,
                "f1_score": 1.0,
            }

        comparison = {}

        for name, metrics in self.results.items():
            comparison[name] = {
                "detection_degradation": iid_baseline["detection_rate"] - metrics.true_positive_rate,
                "fp_increase": metrics.false_positive_rate - iid_baseline["false_positive_rate"],
                "f1_degradation": iid_baseline["f1_score"] - metrics.f1_score,
                "relative_detection": metrics.true_positive_rate / iid_baseline["detection_rate"]
                if iid_baseline["detection_rate"] > 0 else 0,
            }

        return comparison

    def generate_confusion_matrix_plot(
        self,
        results: Dict[str, Any],
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Generate confusion matrix visualization.

        Args:
            results: Experiment results
            output_path: Path to save plot

        Returns:
            Path to saved plot or None
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            logger.warning("matplotlib not available")
            return None

        if not self.results:
            return None

        n_scenarios = len(self.results)
        cols = min(3, n_scenarios)
        rows = (n_scenarios + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if n_scenarios == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]

        for idx, (name, metrics) in enumerate(self.results.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col] if rows > 1 else axes[col]

            # Create confusion matrix
            # Using rates to approximate counts
            tp = metrics.true_positive_rate * 10
            fp = metrics.false_positive_rate * 10
            fn = (1 - metrics.recall) * 10 if metrics.recall > 0 else 0
            tn = 10 - fp

            cm = np.array([[tn, fp], [fn, tp]])

            im = ax.imshow(cm, cmap='Blues')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Honest', 'Byzantine'])
            ax.set_yticklabels(['Honest', 'Byzantine'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

            # Add text annotations
            for i in range(2):
                for j in range(2):
                    text = ax.text(j, i, f'{cm[i, j]:.1f}',
                                   ha='center', va='center',
                                   color='white' if cm[i, j] > cm.max()/2 else 'black')

            ax.set_title(name.split("/")[-1][:20])

        # Hide empty subplots
        for idx in range(len(self.results), rows * cols):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col] if rows > 1 else axes[col]
            ax.axis('off')

        plt.suptitle('Confusion Matrices by Scenario')
        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / "detection_confusion_matrices.png"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved confusion matrix plot to {output_path}")
        return output_path

    def generate_fp_analysis_plot(
        self,
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Generate false positive analysis plot.

        Shows FP rates by data distribution characteristics.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            return None

        # Collect all FP distributions
        all_fp_data = {}
        for name, metrics in self.results.items():
            for dist_type, fp_rate in metrics.fp_by_distribution.items():
                if dist_type not in all_fp_data:
                    all_fp_data[dist_type] = []
                all_fp_data[dist_type].append((name, fp_rate))

        if not all_fp_data:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        x_labels = []
        values = []
        colors = []

        color_map = {
            "size": "#3498db",
            "skew": "#e74c3c",
            "transform": "#2ecc71",
        }

        for dist_type, data in sorted(all_fp_data.items()):
            for scenario, fp_rate in data:
                x_labels.append(f"{scenario.split('/')[-1][:10]}\n{dist_type}")
                values.append(fp_rate if isinstance(fp_rate, (int, float)) else 0)
                prefix = dist_type.split("_")[0]
                colors.append(color_map.get(prefix, "#95a5a6"))

        x = np.arange(len(x_labels))
        ax.bar(x, values, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel("False Positive Rate")
        ax.set_title("False Positive Rates by Distribution Type")
        ax.axhline(y=0.05, color='r', linestyle='--', label='5% threshold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / "detection_fp_analysis.png"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved FP analysis plot to {output_path}")
        return output_path

    def generate_detection_comparison_plot(
        self,
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Generate detection rate comparison across scenarios.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            return None

        if not self.results:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Detection rates
        ax1 = axes[0]
        names = list(self.results.keys())
        short_names = [n.split("/")[-1][:15] for n in names]
        detection_rates = [self.results[n].true_positive_rate for n in names]
        fp_rates = [self.results[n].false_positive_rate for n in names]

        x = np.arange(len(names))
        width = 0.35

        bars1 = ax1.bar(x - width/2, detection_rates, width, label='Detection Rate', color='#27ae60')
        bars2 = ax1.bar(x + width/2, fp_rates, width, label='FP Rate', color='#e74c3c')

        ax1.set_ylabel('Rate')
        ax1.set_title('Detection vs False Positive Rates')
        ax1.set_xticks(x)
        ax1.set_xticklabels(short_names, rotation=45, ha='right')
        ax1.legend()
        ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
        ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: F1 scores
        ax2 = axes[1]
        f1_scores = [self.results[n].f1_score for n in names]

        bars = ax2.bar(x, f1_scores, color='#3498db', alpha=0.8)
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(short_names, rotation=45, ha='right')
        ax2.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='90% target')
        ax2.legend()
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / "detection_comparison.png"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved detection comparison plot to {output_path}")
        return output_path

    def generate_report(
        self,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate detection analysis report.

        Args:
            output_path: Path to save JSON report

        Returns:
            Report dictionary
        """
        # Find best and worst scenarios
        if self.results:
            best_detection = max(self.results.items(), key=lambda x: x[1].true_positive_rate)
            worst_detection = min(self.results.items(), key=lambda x: x[1].true_positive_rate)
            lowest_fp = min(self.results.items(), key=lambda x: x[1].false_positive_rate)
            highest_fp = max(self.results.items(), key=lambda x: x[1].false_positive_rate)
        else:
            best_detection = worst_detection = lowest_fp = highest_fp = (None, None)

        report = {
            "analysis_type": "detection",
            "timestamp": datetime.now().isoformat(),
            "scenarios": {
                name: metrics.to_dict()
                for name, metrics in self.results.items()
            },
            "summary": {
                "best_detection": {
                    "scenario": best_detection[0],
                    "rate": best_detection[1].true_positive_rate if best_detection[1] else 0,
                },
                "worst_detection": {
                    "scenario": worst_detection[0],
                    "rate": worst_detection[1].true_positive_rate if worst_detection[1] else 0,
                },
                "lowest_fp": {
                    "scenario": lowest_fp[0],
                    "rate": lowest_fp[1].false_positive_rate if lowest_fp[1] else 0,
                },
                "highest_fp": {
                    "scenario": highest_fp[0],
                    "rate": highest_fp[1].false_positive_rate if highest_fp[1] else 0,
                },
                "average_detection": np.mean([m.true_positive_rate for m in self.results.values()])
                if self.results else 0,
                "average_fp": np.mean([m.false_positive_rate for m in self.results.values()])
                if self.results else 0,
            },
            "iid_comparison": self.compare_iid_vs_non_iid(),
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved detection report to {output_path}")

        return report


def analyze_detection(
    results: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Main entry point for detection analysis.

    Args:
        results: Experiment results dictionary
        output_dir: Output directory for reports and plots

    Returns:
        Analysis results dictionary
    """
    analyzer = DetectionAnalyzer(output_dir)

    # Analyze all scenarios
    metrics = analyzer.analyze_all(results)

    # Generate plots
    cm_path = analyzer.generate_confusion_matrix_plot(results)
    fp_path = analyzer.generate_fp_analysis_plot()
    comp_path = analyzer.generate_detection_comparison_plot()

    # Generate report
    report = analyzer.generate_report(output_dir / "detection_report.json")

    return {
        "metrics": {k: v.to_dict() for k, v in metrics.items()},
        "plots": {
            "confusion_matrix": str(cm_path) if cm_path else None,
            "fp_analysis": str(fp_path) if fp_path else None,
            "comparison": str(comp_path) if comp_path else None,
        },
        "report": report,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detection Analysis")
    parser.add_argument("results_file", type=Path, help="Path to results JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Output directory")

    args = parser.parse_args()

    with open(args.results_file) as f:
        results = json.load(f)

    analysis = analyze_detection(results, args.output_dir)

    print(f"\nDetection Analysis Complete")
    print(f"Report: {args.output_dir / 'detection_report.json'}")
