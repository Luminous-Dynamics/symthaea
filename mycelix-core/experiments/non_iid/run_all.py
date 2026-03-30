#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Non-IID Validation Experiments Runner

Runs all Non-IID validation scenarios for the Mycelix FL system:
- Label Skew (Dirichlet alpha = 0.1, 0.5, 1.0)
- Feature Skew (brightness, rotation, noise)
- Quantity Skew (100x imbalance)
- Combined Skew

Usage:
    python experiments/non_iid/run_all.py
    python experiments/non_iid/run_all.py --config custom_config.yaml
    python experiments/non_iid/run_all.py --scenarios label_skew,feature_skew
    python experiments/non_iid/run_all.py --quick  # Quick test mode

Author: Luminous Dynamics
Date: January 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "0TML" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class NonIIDExperimentRunner:
    """
    Orchestrates all Non-IID validation experiments.

    Loads configuration, runs scenarios in sequence, collects metrics,
    and generates comprehensive reports.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        scenarios: Optional[List[str]] = None,
        quick_mode: bool = False,
    ):
        """
        Initialize experiment runner.

        Args:
            config_path: Path to YAML configuration file
            output_dir: Override output directory
            scenarios: List of scenarios to run (None = all)
            quick_mode: Run with reduced iterations for testing
        """
        self.config_path = config_path or Path(__file__).parent / "config.yaml"
        self.config = self._load_config()

        # Override settings
        if output_dir:
            self.config["experiment"]["output_dir"] = str(output_dir)
        if quick_mode:
            self._apply_quick_mode()

        self.scenarios = scenarios or self._get_enabled_scenarios()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(self.config["experiment"]["output_dir"])

        # Results storage
        self.results: Dict[str, Any] = {}

        logger.info(f"Initialized Non-IID experiment runner")
        logger.info(f"Config: {self.config_path}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Scenarios: {self.scenarios}")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            logger.warning(f"Config not found: {self.config_path}, using defaults")
            return self._default_config()

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded config from {self.config_path}")
        return config

    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "experiment": {
                "name": "non_iid_validation",
                "seed": 42,
                "output_dir": "results/non_iid",
            },
            "network": {
                "num_nodes": 20,
                "byzantine_ratio": 0.30,
                "attack_types": ["gradient_scaling", "sign_flip", "gaussian_noise"],
            },
            "training": {
                "num_rounds": 100,
                "local_epochs": 5,
                "batch_size": 32,
                "warmup_rounds": 5,
            },
            "dataset": {
                "name": "cifar10",
                "data_dir": "./data",
            },
            "label_skew": {
                "enabled": True,
                "alphas": [0.1, 0.5, 1.0],
            },
            "feature_skew": {
                "enabled": True,
            },
            "quantity_skew": {
                "enabled": True,
            },
            "combined_skew": {
                "enabled": True,
            },
            "plotting": {
                "enabled": True,
                "dpi": 300,
            },
        }

    def _apply_quick_mode(self):
        """Apply quick mode settings for fast testing."""
        logger.info("Quick mode enabled - using reduced settings")
        self.config["training"]["num_rounds"] = 10
        self.config["training"]["warmup_rounds"] = 2
        self.config["network"]["num_nodes"] = 10
        self.config["label_skew"]["alphas"] = [0.5]

    def _get_enabled_scenarios(self) -> List[str]:
        """Get list of enabled scenarios from config."""
        scenarios = []
        for name in ["label_skew", "feature_skew", "quantity_skew", "combined_skew"]:
            if self.config.get(name, {}).get("enabled", True):
                scenarios.append(name)
        return scenarios

    def _setup_output_dirs(self, scenario: str) -> Path:
        """Create output directory structure for a scenario."""
        scenario_dir = self.output_dir / scenario / self.timestamp
        (scenario_dir / "plots").mkdir(parents=True, exist_ok=True)
        return scenario_dir

    def run_label_skew(self) -> Dict[str, Any]:
        """Run label skew experiments with Dirichlet distribution."""
        logger.info("=" * 60)
        logger.info("Running Label Skew Experiments")
        logger.info("=" * 60)

        from scenarios.label_skew import LabelSkewScenario, LabelSkewConfig

        output_dir = self._setup_output_dirs("label_skew")
        alphas = self.config["label_skew"]["alphas"]

        results = {}
        for alpha in alphas:
            logger.info(f"\n--- Alpha = {alpha} ---")

            config = LabelSkewConfig(
                alpha=alpha,
                num_nodes=self.config["network"]["num_nodes"],
                byzantine_ratio=self.config["network"]["byzantine_ratio"],
                num_rounds=self.config["training"]["num_rounds"],
                dataset=self.config["dataset"]["name"],
                seed=self.config["experiment"]["seed"],
            )

            scenario = LabelSkewScenario(config)
            result = scenario.run()
            results[f"alpha_{alpha}"] = result

            # Save intermediate results
            result_file = output_dir / f"alpha_{alpha}_results.json"
            with open(result_file, "w") as f:
                json.dump(result, f, indent=2, default=str)

            logger.info(
                f"Alpha {alpha}: Detection={result.get('detection_accuracy', 0):.2%}, "
                f"Model Acc={result.get('model_accuracy', 0):.2%}"
            )

        # Generate summary
        summary = self._generate_label_skew_summary(results, output_dir)
        return {"results": results, "summary": summary, "output_dir": str(output_dir)}

    def run_feature_skew(self) -> Dict[str, Any]:
        """Run feature skew experiments."""
        logger.info("=" * 60)
        logger.info("Running Feature Skew Experiments")
        logger.info("=" * 60)

        from scenarios.feature_skew import FeatureSkewScenario, FeatureSkewConfig

        output_dir = self._setup_output_dirs("feature_skew")

        config = FeatureSkewConfig(
            num_nodes=self.config["network"]["num_nodes"],
            byzantine_ratio=self.config["network"]["byzantine_ratio"],
            num_rounds=self.config["training"]["num_rounds"],
            dataset=self.config["dataset"]["name"],
            transformations=self.config.get("feature_skew", {}).get("transformations", []),
            seed=self.config["experiment"]["seed"],
        )

        scenario = FeatureSkewScenario(config)
        result = scenario.run()

        # Save results
        result_file = output_dir / "metrics.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(
            f"Feature Skew: Detection={result.get('detection_accuracy', 0):.2%}, "
            f"Model Acc={result.get('model_accuracy', 0):.2%}"
        )

        return {"results": result, "output_dir": str(output_dir)}

    def run_quantity_skew(self) -> Dict[str, Any]:
        """Run quantity skew experiments with 100x imbalance."""
        logger.info("=" * 60)
        logger.info("Running Quantity Skew Experiments")
        logger.info("=" * 60)

        from scenarios.quantity_skew import QuantitySkewScenario, QuantitySkewConfig

        output_dir = self._setup_output_dirs("quantity_skew")

        qs_config = self.config.get("quantity_skew", {}).get("distribution", {})

        config = QuantitySkewConfig(
            num_nodes=self.config["network"]["num_nodes"],
            byzantine_ratio=self.config["network"]["byzantine_ratio"],
            num_rounds=self.config["training"]["num_rounds"],
            dataset=self.config["dataset"]["name"],
            large_nodes=qs_config.get("large_nodes", 2),
            large_samples=qs_config.get("large_samples", 10000),
            medium_nodes=qs_config.get("medium_nodes", 8),
            medium_samples=qs_config.get("medium_samples", 1000),
            small_nodes=qs_config.get("small_nodes", 10),
            small_samples=qs_config.get("small_samples", 100),
            seed=self.config["experiment"]["seed"],
        )

        scenario = QuantitySkewScenario(config)
        result = scenario.run()

        # Save results
        result_file = output_dir / "metrics.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(
            f"Quantity Skew: Detection={result.get('detection_accuracy', 0):.2%}, "
            f"Model Acc={result.get('model_accuracy', 0):.2%}"
        )

        return {"results": result, "output_dir": str(output_dir)}

    def run_combined_skew(self) -> Dict[str, Any]:
        """Run combined skew experiments."""
        logger.info("=" * 60)
        logger.info("Running Combined Skew Experiments")
        logger.info("=" * 60)

        from scenarios.combined_skew import CombinedSkewScenario, CombinedSkewConfig

        output_dir = self._setup_output_dirs("combined_skew")

        cs_config = self.config.get("combined_skew", {})

        config = CombinedSkewConfig(
            num_nodes=self.config["network"]["num_nodes"],
            byzantine_ratio=self.config["network"]["byzantine_ratio"],
            num_rounds=self.config["training"]["num_rounds"],
            dataset=self.config["dataset"]["name"],
            label_alpha=cs_config.get("label_alpha", 0.5),
            feature_ratio=cs_config.get("feature_ratio", 0.3),
            quantity_imbalance=cs_config.get("quantity_imbalance", True),
            seed=self.config["experiment"]["seed"],
        )

        scenario = CombinedSkewScenario(config)
        result = scenario.run()

        # Save results
        result_file = output_dir / "metrics.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(
            f"Combined Skew: Detection={result.get('detection_accuracy', 0):.2%}, "
            f"Model Acc={result.get('model_accuracy', 0):.2%}"
        )

        return {"results": result, "output_dir": str(output_dir)}

    def _generate_label_skew_summary(
        self, results: Dict[str, Any], output_dir: Path
    ) -> Dict[str, Any]:
        """Generate summary for label skew experiments."""
        summary = {
            "alphas_tested": list(results.keys()),
            "best_detection": max(
                (r.get("detection_accuracy", 0) for r in results.values()),
                default=0,
            ),
            "best_model_accuracy": max(
                (r.get("model_accuracy", 0) for r in results.values()),
                default=0,
            ),
        }

        # Write summary markdown
        summary_md = output_dir / "summary.md"
        with open(summary_md, "w") as f:
            f.write("# Label Skew Experiment Summary\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            f.write("## Results by Alpha\n\n")
            f.write("| Alpha | Detection | Model Acc | False Positives | Convergence Rounds |\n")
            f.write("|-------|-----------|-----------|-----------------|-------------------|\n")
            for name, result in results.items():
                alpha = name.replace("alpha_", "")
                det = result.get("detection_accuracy", 0)
                acc = result.get("model_accuracy", 0)
                fp = result.get("false_positive_rate", 0)
                conv = result.get("convergence_round", "N/A")
                f.write(f"| {alpha} | {det:.2%} | {acc:.2%} | {fp:.2%} | {conv} |\n")

        return summary

    def run_analysis(self) -> Dict[str, Any]:
        """Run analysis on collected results."""
        logger.info("=" * 60)
        logger.info("Running Analysis")
        logger.info("=" * 60)

        analysis_results = {}

        # Convergence analysis
        try:
            from analysis.convergence import analyze_convergence
            convergence = analyze_convergence(self.results, self.output_dir)
            analysis_results["convergence"] = convergence
            logger.info("Convergence analysis complete")
        except ImportError as e:
            logger.warning(f"Convergence analysis skipped: {e}")

        # Detection analysis
        try:
            from analysis.detection import analyze_detection
            detection = analyze_detection(self.results, self.output_dir)
            analysis_results["detection"] = detection
            logger.info("Detection analysis complete")
        except ImportError as e:
            logger.warning(f"Detection analysis skipped: {e}")

        # Fairness analysis
        try:
            from analysis.fairness import analyze_fairness
            fairness = analyze_fairness(self.results, self.output_dir)
            analysis_results["fairness"] = fairness
            logger.info("Fairness analysis complete")
        except ImportError as e:
            logger.warning(f"Fairness analysis skipped: {e}")

        return analysis_results

    def generate_report(self) -> Path:
        """Generate comprehensive experiment report."""
        logger.info("Generating final report...")

        report_dir = self.output_dir / "reports" / self.timestamp
        report_dir.mkdir(parents=True, exist_ok=True)

        # Write JSON report
        report_file = report_dir / "full_report.json"
        report_data = {
            "experiment": {
                "name": self.config["experiment"]["name"],
                "timestamp": self.timestamp,
                "config": self.config,
            },
            "results": self.results,
            "environment": self._get_environment_info(),
        }

        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        # Write markdown summary
        summary_file = report_dir / "summary.md"
        self._write_markdown_summary(summary_file)

        logger.info(f"Report saved to {report_dir}")
        return report_dir

    def _get_environment_info(self) -> Dict[str, Any]:
        """Collect environment information."""
        import platform
        import socket

        try:
            import torch
            torch_version = torch.__version__
            cuda_available = torch.cuda.is_available()
        except ImportError:
            torch_version = "N/A"
            cuda_available = False

        return {
            "hostname": socket.gethostname(),
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "torch_version": torch_version,
            "cuda_available": cuda_available,
        }

    def _write_markdown_summary(self, filepath: Path):
        """Write markdown summary of all experiments."""
        with open(filepath, "w") as f:
            f.write("# Non-IID Validation Experiment Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            f.write("## Configuration\n\n")
            f.write(f"- Nodes: {self.config['network']['num_nodes']}\n")
            f.write(f"- Byzantine Ratio: {self.config['network']['byzantine_ratio']:.0%}\n")
            f.write(f"- Rounds: {self.config['training']['num_rounds']}\n")
            f.write(f"- Dataset: {self.config['dataset']['name']}\n\n")

            f.write("## Results Summary\n\n")
            f.write("| Scenario | Detection | Model Acc | FP Rate | Notes |\n")
            f.write("|----------|-----------|-----------|---------|-------|\n")

            for scenario, data in self.results.items():
                if isinstance(data, dict) and "results" in data:
                    if isinstance(data["results"], dict):
                        # Multiple sub-results (e.g., label skew with multiple alphas)
                        for name, result in data["results"].items():
                            if isinstance(result, dict):
                                det = result.get("detection_accuracy", 0)
                                acc = result.get("model_accuracy", 0)
                                fp = result.get("false_positive_rate", 0)
                                f.write(f"| {scenario}/{name} | {det:.2%} | {acc:.2%} | {fp:.2%} | |\n")
                    else:
                        result = data["results"]
                        det = result.get("detection_accuracy", 0)
                        acc = result.get("model_accuracy", 0)
                        fp = result.get("false_positive_rate", 0)
                        f.write(f"| {scenario} | {det:.2%} | {acc:.2%} | {fp:.2%} | |\n")

            f.write("\n## Conclusions\n\n")
            f.write("*Analysis pending*\n")

    def run(self) -> Dict[str, Any]:
        """Run all enabled experiments."""
        start_time = time.time()

        logger.info("=" * 60)
        logger.info("Non-IID Validation Experiments")
        logger.info("Mycelix Federated Learning System")
        logger.info("=" * 60)
        logger.info(f"Timestamp: {self.timestamp}")
        logger.info(f"Scenarios: {self.scenarios}")

        # Run each scenario
        scenario_runners = {
            "label_skew": self.run_label_skew,
            "feature_skew": self.run_feature_skew,
            "quantity_skew": self.run_quantity_skew,
            "combined_skew": self.run_combined_skew,
        }

        for scenario in self.scenarios:
            if scenario in scenario_runners:
                try:
                    result = scenario_runners[scenario]()
                    self.results[scenario] = result
                except Exception as e:
                    logger.error(f"Scenario {scenario} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    self.results[scenario] = {"error": str(e)}

        # Run analysis
        try:
            analysis = self.run_analysis()
            self.results["analysis"] = analysis
        except Exception as e:
            logger.error(f"Analysis failed: {e}")

        # Generate report
        report_dir = self.generate_report()

        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"All experiments completed in {elapsed:.1f}s")
        logger.info(f"Results saved to {report_dir}")
        logger.info("=" * 60)

        return self.results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Non-IID validation experiments for Mycelix FL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_all.py
    python run_all.py --config custom_config.yaml
    python run_all.py --scenarios label_skew,feature_skew
    python run_all.py --quick
    python run_all.py --output-dir /path/to/results
        """,
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        help="Comma-separated list of scenarios to run",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode with reduced iterations",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    scenarios = None
    if args.scenarios:
        scenarios = [s.strip() for s in args.scenarios.split(",")]

    runner = NonIIDExperimentRunner(
        config_path=args.config,
        output_dir=args.output_dir,
        scenarios=scenarios,
        quick_mode=args.quick,
    )

    results = runner.run()

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    for scenario, data in results.items():
        if scenario == "analysis":
            continue
        if isinstance(data, dict) and "error" in data:
            print(f"{scenario}: ERROR - {data['error']}")
        else:
            print(f"{scenario}: Complete")

    return 0


if __name__ == "__main__":
    sys.exit(main())
