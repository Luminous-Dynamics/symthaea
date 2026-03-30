#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Fairness Analysis for Non-IID Experiments

Analyzes fairness of Byzantine detection and aggregation under data imbalance:
- Reputation score distribution
- Contribution equity
- Treatment of minority data distributions
- Bias toward large vs small nodes

Key metrics:
- Gini coefficient of influence
- Disparity in false positive rates
- Representation in aggregation
- Reputation fairness

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
class FairnessMetrics:
    """Fairness metrics for a scenario."""
    scenario: str

    # False positive parity
    fp_rate_disparity: float  # Max - min FP rate across groups
    fp_rate_gini: float  # Gini coefficient of FP rates

    # Detection parity
    detection_rate_disparity: float
    detection_rate_gini: float

    # Influence/contribution fairness
    contribution_gini: float
    small_node_influence: float  # 0-1, how much small nodes matter
    large_node_dominance: float  # 0-1, how much large nodes dominate

    # Reputation fairness
    reputation_variance: float
    min_reputation: float
    max_reputation: float

    # Overall fairness score (0-1, higher = more fair)
    fairness_score: float

    group_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "fp_rate_disparity": self.fp_rate_disparity,
            "fp_rate_gini": self.fp_rate_gini,
            "detection_rate_disparity": self.detection_rate_disparity,
            "detection_rate_gini": self.detection_rate_gini,
            "contribution_gini": self.contribution_gini,
            "small_node_influence": self.small_node_influence,
            "large_node_dominance": self.large_node_dominance,
            "reputation_variance": self.reputation_variance,
            "min_reputation": self.min_reputation,
            "max_reputation": self.max_reputation,
            "fairness_score": self.fairness_score,
            "group_metrics": self.group_metrics,
        }


class FairnessAnalyzer:
    """
    Analyzes fairness of FL system under Non-IID conditions.

    Examines:
    1. Whether small nodes are unfairly penalized
    2. Whether large nodes dominate aggregation
    3. Disparity in false positive rates
    4. Equity in reputation scores

    Example:
        >>> analyzer = FairnessAnalyzer()
        >>> results = analyzer.analyze(experiment_results)
        >>> analyzer.generate_fairness_report(output_dir)
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize analyzer."""
        self.output_dir = Path(output_dir) if output_dir else Path(".")
        self.results: Dict[str, FairnessMetrics] = {}

    @staticmethod
    def calculate_gini(values: List[float]) -> float:
        """
        Calculate Gini coefficient.

        0 = perfect equality
        1 = perfect inequality
        """
        if not values or all(v == 0 for v in values):
            return 0.0

        values = sorted(values)
        n = len(values)
        cumsum = np.cumsum(values)
        total = cumsum[-1]

        if total == 0:
            return 0.0

        # Gini = (2 * sum(i * x_i) - (n + 1) * sum(x_i)) / (n * sum(x_i))
        weighted_sum = sum((i + 1) * v for i, v in enumerate(values))
        gini = (2 * weighted_sum) / (n * total) - (n + 1) / n

        return max(0.0, min(1.0, gini))

    def analyze_scenario(
        self,
        scenario_name: str,
        results: Dict[str, Any],
    ) -> FairnessMetrics:
        """
        Analyze fairness for a single scenario.

        Args:
            scenario_name: Name of the scenario
            results: Scenario results dictionary

        Returns:
            FairnessMetrics for this scenario
        """
        # Extract size-based analysis
        size_analysis = results.get("size_analysis", {})
        skew_analysis = results.get("skew_analysis", {})
        fairness_data = results.get("fairness_metrics", {})

        # Calculate FP rate disparity
        fp_rates = []
        detection_rates = []
        group_metrics = {}

        for size, data in size_analysis.items():
            if isinstance(data, dict):
                fp_rate = data.get("fp_rate", 0)
                det_rate = data.get("detection_rate", 1.0)
                fp_rates.append(fp_rate)
                detection_rates.append(det_rate)
                group_metrics[f"size_{size}"] = {
                    "fp_rate": fp_rate,
                    "detection_rate": det_rate,
                    "count": data.get("total", 0),
                }

        for skew_level, data in skew_analysis.items():
            if isinstance(data, dict):
                fp_rate = data.get("fp_rate", 0)
                fp_rates.append(fp_rate)
                group_metrics[f"skew_{skew_level}"] = {
                    "fp_rate": fp_rate,
                    "count": data.get("count", 0),
                }

        # FP rate metrics
        fp_disparity = max(fp_rates) - min(fp_rates) if fp_rates else 0.0
        fp_gini = self.calculate_gini(fp_rates) if fp_rates else 0.0

        # Detection rate metrics
        det_disparity = max(detection_rates) - min(detection_rates) if detection_rates else 0.0
        det_gini = self.calculate_gini(detection_rates) if detection_rates else 0.0

        # Contribution fairness from node samples
        node_samples = results.get("node_samples", {})
        if node_samples:
            sample_counts = list(node_samples.values())
            contribution_gini = self.calculate_gini(sample_counts)

            # Small node influence
            total_samples = sum(sample_counts)
            if total_samples > 0:
                sorted_samples = sorted(sample_counts)
                smallest_half = sorted_samples[:len(sorted_samples)//2]
                small_influence = sum(smallest_half) / total_samples
                large_dominance = 1 - small_influence
            else:
                small_influence = 0.5
                large_dominance = 0.5
        else:
            contribution_gini = 0.0
            small_influence = 0.5
            large_dominance = 0.5

        # Use pre-calculated fairness metrics if available
        if fairness_data:
            fp_gini = fairness_data.get("fp_rate_gini", fp_gini)
            fp_disparity = fairness_data.get("max_fp_rate_difference", fp_disparity)

        # Reputation metrics (simulated if not available)
        skew_scores = results.get("skew_scores", {})
        if skew_scores:
            scores = list(skew_scores.values())
            rep_variance = float(np.var(scores))
            min_rep = min(scores)
            max_rep = max(scores)
        else:
            rep_variance = 0.0
            min_rep = 0.0
            max_rep = 1.0

        # Calculate overall fairness score
        # Lower disparity and Gini = higher fairness
        fairness_score = 1.0 - (
            0.3 * fp_disparity +
            0.2 * fp_gini +
            0.2 * det_disparity +
            0.2 * contribution_gini +
            0.1 * (large_dominance - 0.5)  # Penalize if large nodes dominate
        )
        fairness_score = max(0.0, min(1.0, fairness_score))

        metrics = FairnessMetrics(
            scenario=scenario_name,
            fp_rate_disparity=fp_disparity,
            fp_rate_gini=fp_gini,
            detection_rate_disparity=det_disparity,
            detection_rate_gini=det_gini,
            contribution_gini=contribution_gini,
            small_node_influence=small_influence,
            large_node_dominance=large_dominance,
            reputation_variance=rep_variance,
            min_reputation=min_rep,
            max_reputation=max_rep,
            fairness_score=fairness_score,
            group_metrics=group_metrics,
        )

        self.results[scenario_name] = metrics
        return metrics

    def analyze_all(
        self,
        all_results: Dict[str, Any],
    ) -> Dict[str, FairnessMetrics]:
        """
        Analyze fairness for all scenarios.

        Args:
            all_results: Dictionary of all experiment results

        Returns:
            Dictionary of scenario -> FairnessMetrics
        """
        for scenario_name, scenario_data in all_results.items():
            if scenario_name == "analysis":
                continue

            if isinstance(scenario_data, dict):
                if "results" in scenario_data:
                    inner = scenario_data["results"]
                    if isinstance(inner, dict) and "size_analysis" not in inner and "skew_analysis" not in inner:
                        # Multiple sub-results
                        for sub_name, sub_data in inner.items():
                            if isinstance(sub_data, dict):
                                full_name = f"{scenario_name}/{sub_name}"
                                self.analyze_scenario(full_name, sub_data)
                    else:
                        self.analyze_scenario(scenario_name, inner)
                elif "size_analysis" in scenario_data or "skew_analysis" in scenario_data:
                    self.analyze_scenario(scenario_name, scenario_data)

        return self.results

    def generate_fairness_distribution_plot(
        self,
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Generate fairness distribution visualization.
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

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: FP Rate Disparity
        ax1 = axes[0, 0]
        names = list(self.results.keys())
        short_names = [n.split("/")[-1][:15] for n in names]
        fp_disparity = [self.results[n].fp_rate_disparity for n in names]

        colors = ['#e74c3c' if d > 0.1 else '#27ae60' for d in fp_disparity]
        ax1.bar(range(len(names)), fp_disparity, color=colors, alpha=0.8)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(short_names, rotation=45, ha='right')
        ax1.set_ylabel("FP Rate Disparity")
        ax1.set_title("False Positive Rate Disparity\n(Lower = More Fair)")
        ax1.axhline(y=0.05, color='orange', linestyle='--', label='5% threshold')
        ax1.axhline(y=0.10, color='red', linestyle='--', label='10% threshold')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3, axis='y')

        # Plot 2: Gini Coefficients
        ax2 = axes[0, 1]
        fp_gini = [self.results[n].fp_rate_gini for n in names]
        contrib_gini = [self.results[n].contribution_gini for n in names]

        x = np.arange(len(names))
        width = 0.35
        ax2.bar(x - width/2, fp_gini, width, label='FP Rate Gini', color='#e74c3c', alpha=0.8)
        ax2.bar(x + width/2, contrib_gini, width, label='Contribution Gini', color='#3498db', alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(short_names, rotation=45, ha='right')
        ax2.set_ylabel("Gini Coefficient")
        ax2.set_title("Inequality Metrics\n(Lower = More Equal)")
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        # Plot 3: Node Influence Distribution
        ax3 = axes[1, 0]
        small_influence = [self.results[n].small_node_influence for n in names]
        large_dominance = [self.results[n].large_node_dominance for n in names]

        ax3.bar(x - width/2, small_influence, width, label='Small Node Influence', color='#2ecc71', alpha=0.8)
        ax3.bar(x + width/2, large_dominance, width, label='Large Node Dominance', color='#9b59b6', alpha=0.8)
        ax3.set_xticks(x)
        ax3.set_xticklabels(short_names, rotation=45, ha='right')
        ax3.set_ylabel("Influence Share")
        ax3.set_title("Node Size Influence Distribution")
        ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Equal share')
        ax3.legend()
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Overall Fairness Score
        ax4 = axes[1, 1]
        fairness_scores = [self.results[n].fairness_score for n in names]

        colors = ['#27ae60' if s > 0.8 else '#f39c12' if s > 0.6 else '#e74c3c' for s in fairness_scores]
        ax4.bar(range(len(names)), fairness_scores, color=colors, alpha=0.8)
        ax4.set_xticks(range(len(names)))
        ax4.set_xticklabels(short_names, rotation=45, ha='right')
        ax4.set_ylabel("Fairness Score")
        ax4.set_title("Overall Fairness Score\n(Higher = More Fair)")
        ax4.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
        ax4.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5)
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / "fairness_distribution.png"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved fairness plot to {output_path}")
        return output_path

    def generate_group_comparison_plot(
        self,
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Generate group-level fairness comparison.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            return None

        # Collect group data across scenarios
        all_groups = {}
        for name, metrics in self.results.items():
            for group, data in metrics.group_metrics.items():
                if group not in all_groups:
                    all_groups[group] = {"scenarios": [], "fp_rates": []}
                all_groups[group]["scenarios"].append(name)
                all_groups[group]["fp_rates"].append(data.get("fp_rate", 0))

        if not all_groups:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        groups = list(all_groups.keys())
        avg_fp_rates = [np.mean(all_groups[g]["fp_rates"]) for g in groups]

        colors = ['#3498db' if 'size' in g else '#e74c3c' if 'skew' in g else '#2ecc71' for g in groups]

        bars = ax.bar(range(len(groups)), avg_fp_rates, color=colors, alpha=0.8)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.set_ylabel("Average False Positive Rate")
        ax.set_title("FP Rates by Group (Across All Scenarios)")
        ax.axhline(y=0.05, color='orange', linestyle='--', label='5% target')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if output_path is None:
            output_path = self.output_dir / "fairness_group_comparison.png"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved group comparison plot to {output_path}")
        return output_path

    def generate_report(
        self,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate fairness analysis report.

        Args:
            output_path: Path to save JSON report

        Returns:
            Report dictionary
        """
        if self.results:
            most_fair = max(self.results.items(), key=lambda x: x[1].fairness_score)
            least_fair = min(self.results.items(), key=lambda x: x[1].fairness_score)
            avg_fairness = np.mean([m.fairness_score for m in self.results.values()])
        else:
            most_fair = least_fair = (None, None)
            avg_fairness = 0

        report = {
            "analysis_type": "fairness",
            "timestamp": datetime.now().isoformat(),
            "scenarios": {
                name: metrics.to_dict()
                for name, metrics in self.results.items()
            },
            "summary": {
                "most_fair": {
                    "scenario": most_fair[0],
                    "score": most_fair[1].fairness_score if most_fair[1] else 0,
                },
                "least_fair": {
                    "scenario": least_fair[0],
                    "score": least_fair[1].fairness_score if least_fair[1] else 0,
                },
                "average_fairness": avg_fairness,
                "average_fp_disparity": np.mean([m.fp_rate_disparity for m in self.results.values()])
                if self.results else 0,
                "average_gini": np.mean([m.contribution_gini for m in self.results.values()])
                if self.results else 0,
            },
            "recommendations": self._generate_recommendations(),
        }

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved fairness report to {output_path}")

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate fairness improvement recommendations."""
        recommendations = []

        for name, metrics in self.results.items():
            if metrics.fp_rate_disparity > 0.1:
                recommendations.append(
                    f"{name}: High FP disparity ({metrics.fp_rate_disparity:.2%}). "
                    "Consider adjusting detection thresholds for minority groups."
                )

            if metrics.contribution_gini > 0.6:
                recommendations.append(
                    f"{name}: High contribution inequality (Gini={metrics.contribution_gini:.2f}). "
                    "Consider weighted aggregation to balance node influence."
                )

            if metrics.small_node_influence < 0.3:
                recommendations.append(
                    f"{name}: Small nodes have low influence ({metrics.small_node_influence:.2%}). "
                    "Consider reputation boosting for small but consistent nodes."
                )

        if not recommendations:
            recommendations.append("Overall fairness is acceptable. Continue monitoring.")

        return recommendations


def analyze_fairness(
    results: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Main entry point for fairness analysis.

    Args:
        results: Experiment results dictionary
        output_dir: Output directory for reports and plots

    Returns:
        Analysis results dictionary
    """
    analyzer = FairnessAnalyzer(output_dir)

    # Analyze all scenarios
    metrics = analyzer.analyze_all(results)

    # Generate plots
    dist_path = analyzer.generate_fairness_distribution_plot()
    group_path = analyzer.generate_group_comparison_plot()

    # Generate report
    report = analyzer.generate_report(output_dir / "fairness_report.json")

    return {
        "metrics": {k: v.to_dict() for k, v in metrics.items()},
        "plots": {
            "distribution": str(dist_path) if dist_path else None,
            "group_comparison": str(group_path) if group_path else None,
        },
        "report": report,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fairness Analysis")
    parser.add_argument("results_file", type=Path, help="Path to results JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Output directory")

    args = parser.parse_args()

    with open(args.results_file) as f:
        results = json.load(f)

    analysis = analyze_fairness(results, args.output_dir)

    print(f"\nFairness Analysis Complete")
    print(f"Report: {args.output_dir / 'fairness_report.json'}")
