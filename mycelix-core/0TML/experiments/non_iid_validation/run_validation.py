#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Non-IID Byzantine Detection Validation Runner
==============================================

This script executes comprehensive non-IID validation experiments and generates
a validation report proving Byzantine detection works under realistic conditions.

Usage:
    # Quick validation (core scenarios only)
    python run_validation.py --quick

    # Full validation (all scenarios)
    python run_validation.py --full

    # Specific scenario
    python run_validation.py --scenario moderate_non_iid

    # Alpha sweep
    python run_validation.py --alpha-sweep

    # Aggregator comparison
    python run_validation.py --compare-aggregators

Output:
    - JSON results file with all metrics
    - Text report with summary and analysis
    - Comparison to IID baseline

Author: Luminous Dynamics
Date: January 2026
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.non_iid_validation.experiment_config import (
    NonIIDConfig,
    LabelSkewConfig,
    AttackType,
    AggregatorType,
)
from experiments.non_iid_validation.scenarios import (
    IID_BASELINE,
    MILD_NON_IID,
    MODERATE_NON_IID,
    SEVERE_NON_IID,
    PATHOLOGICAL_NON_IID,
    QUANTITY_SKEW_SCENARIO,
    FEATURE_SKEW_SCENARIO,
    TEMPORAL_DRIFT_SCENARIO,
    get_core_scenarios,
    get_extended_scenarios,
    get_all_scenarios,
    get_scenario_by_name,
    create_alpha_sweep_scenarios,
    create_aggregator_comparison_scenarios,
)
from experiments.non_iid_validation.non_iid_experiment import (
    NonIIDExperiment,
    ExperimentResults,
)


def print_banner():
    """Print validation banner."""
    print("=" * 80)
    print(" NON-IID BYZANTINE DETECTION VALIDATION")
    print(" Proving detection works under realistic federated learning conditions")
    print("=" * 80)
    print()


def print_summary(results: Dict[str, ExperimentResults]):
    """Print summary of validation results."""
    print("\n" + "=" * 80)
    print(" VALIDATION SUMMARY")
    print("=" * 80)

    # Count successes
    n_total = len(results)
    n_detection_pass = sum(1 for r in results.values() if r.detection_success)
    n_accuracy_pass = sum(1 for r in results.values() if r.accuracy_success)
    n_full_pass = sum(1 for r in results.values() if r.detection_success and r.accuracy_success)

    print(f"\nTotal Scenarios: {n_total}")
    print(f"Detection Success (TPR>90%, FPR<15%): {n_detection_pass}/{n_total} ({n_detection_pass/n_total:.0%})")
    print(f"Accuracy Success (>80%): {n_accuracy_pass}/{n_total} ({n_accuracy_pass/n_total:.0%})")
    print(f"Full Pass (Both): {n_full_pass}/{n_total} ({n_full_pass/n_total:.0%})")

    # Find IID baseline for comparison
    iid_acc = results.get("iid_baseline", results.get(list(results.keys())[0])).mean_accuracy

    print(f"\nIID Baseline Accuracy: {iid_acc:.1%}")
    print()

    # Table of results
    print(f"{'Scenario':<35} {'Accuracy':>10} {'vs IID':>8} {'AUC':>8} {'Status':>8}")
    print("-" * 75)

    for name, result in sorted(results.items(), key=lambda x: x[0]):
        diff = result.mean_accuracy - iid_acc
        status = "PASS" if (result.detection_success and result.accuracy_success) else "FAIL"
        print(
            f"{name:<35} {result.mean_accuracy:>9.1%} {diff:>+7.1%} "
            f"{result.mean_auc:>8.4f} {status:>8}"
        )

    print("-" * 75)

    # Overall verdict
    print()
    if n_full_pass == n_total:
        print("VALIDATION PASSED: Byzantine detection works under all tested non-IID conditions!")
    elif n_full_pass >= n_total * 0.9:
        print("VALIDATION MOSTLY PASSED: Byzantine detection works for 90%+ of scenarios.")
    else:
        print("VALIDATION INCOMPLETE: Some scenarios need investigation.")
        print("  Failed scenarios:")
        for name, result in results.items():
            if not (result.detection_success and result.accuracy_success):
                print(f"    - {name}: Acc={result.mean_accuracy:.1%}, AUC={result.mean_auc:.4f}")


def run_quick_validation(runner: NonIIDExperiment) -> Dict[str, ExperimentResults]:
    """Run quick validation with core scenarios."""
    print("\nRunning QUICK validation (core scenarios)...")
    print("Scenarios: IID baseline, Mild, Moderate, Severe non-IID\n")

    scenarios = get_core_scenarios()
    return runner.run_comparison(scenarios)


def run_extended_validation(runner: NonIIDExperiment) -> Dict[str, ExperimentResults]:
    """Run extended validation with more scenarios."""
    print("\nRunning EXTENDED validation...")
    print("Includes: Core + Pathological + Quantity/Feature/Temporal skew\n")

    scenarios = get_extended_scenarios()
    return runner.run_comparison(scenarios)


def run_full_validation(runner: NonIIDExperiment) -> Dict[str, ExperimentResults]:
    """Run full validation suite."""
    print("\nRunning FULL validation suite...")
    print("WARNING: This may take 30-60 minutes depending on hardware.\n")

    scenarios = get_all_scenarios()
    return runner.run_comparison(scenarios)


def run_alpha_sweep(runner: NonIIDExperiment) -> Dict[str, ExperimentResults]:
    """Run alpha sweep across non-IID severity levels."""
    print("\nRunning ALPHA SWEEP validation...")
    print("Testing detection across non-IID severity spectrum\n")

    alphas = [100.0, 1.0, 0.5, 0.3, 0.1, 0.05]
    scenarios = create_alpha_sweep_scenarios(alphas=alphas)

    # Add IID baseline
    results = {"iid_baseline": runner.run_experiment(IID_BASELINE)}

    for scenario in scenarios:
        result = runner.run_experiment(scenario)
        result.iid_accuracy = results["iid_baseline"].mean_accuracy
        result.accuracy_vs_iid = result.mean_accuracy - result.iid_accuracy
        results[scenario.name] = result

    return results


def run_aggregator_comparison(runner: NonIIDExperiment) -> Dict[str, ExperimentResults]:
    """Compare different aggregators on non-IID data."""
    print("\nRunning AGGREGATOR COMPARISON...")
    print("Comparing Krum, TrimmedMean, Median, AEGIS, AEGIS+Gen7\n")

    scenarios = create_aggregator_comparison_scenarios(base_alpha=0.5)

    results = {}
    for scenario in scenarios:
        result = runner.run_experiment(scenario)
        results[scenario.name] = result

    return results


def run_single_scenario(runner: NonIIDExperiment, name: str) -> Dict[str, ExperimentResults]:
    """Run a single named scenario."""
    print(f"\nRunning single scenario: {name}")

    try:
        scenario = get_scenario_by_name(name)
    except ValueError:
        print(f"ERROR: Unknown scenario '{name}'")
        print("Available scenarios:")
        for s in get_all_scenarios():
            print(f"  - {s.name}")
        sys.exit(1)

    baseline_result = runner.run_experiment(IID_BASELINE)
    scenario_result = runner.run_experiment(scenario)
    scenario_result.iid_accuracy = baseline_result.mean_accuracy
    scenario_result.accuracy_vs_iid = scenario_result.mean_accuracy - baseline_result.mean_accuracy

    return {
        "iid_baseline": baseline_result,
        name: scenario_result,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run non-IID Byzantine detection validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_validation.py --quick              # Quick core scenarios
  python run_validation.py --extended           # Extended scenarios
  python run_validation.py --full               # Full test suite
  python run_validation.py --alpha-sweep        # Alpha parameter sweep
  python run_validation.py --compare-aggregators # Aggregator comparison
  python run_validation.py --scenario severe_non_iid  # Single scenario
        """
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true", help="Quick validation (core scenarios)")
    mode.add_argument("--extended", action="store_true", help="Extended validation")
    mode.add_argument("--full", action="store_true", help="Full validation suite")
    mode.add_argument("--alpha-sweep", action="store_true", help="Alpha parameter sweep")
    mode.add_argument("--compare-aggregators", action="store_true", help="Aggregator comparison")
    mode.add_argument("--scenario", type=str, help="Run single named scenario")

    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to file")
    parser.add_argument("--output-dir", type=str, help="Custom output directory")

    args = parser.parse_args()

    print_banner()

    # Initialize runner
    output_dir = Path(args.output_dir) if args.output_dir else None
    runner = NonIIDExperiment(output_dir=output_dir, verbose=not args.quiet)

    # Run selected validation
    if args.quick:
        results = run_quick_validation(runner)
        prefix = "quick"
    elif args.extended:
        results = run_extended_validation(runner)
        prefix = "extended"
    elif args.full:
        results = run_full_validation(runner)
        prefix = "full"
    elif args.alpha_sweep:
        results = run_alpha_sweep(runner)
        prefix = "alpha_sweep"
    elif args.compare_aggregators:
        results = run_aggregator_comparison(runner)
        prefix = "aggregator_comparison"
    elif args.scenario:
        results = run_single_scenario(runner, args.scenario)
        prefix = f"scenario_{args.scenario}"
    else:
        parser.print_help()
        return

    # Print summary
    print_summary(results)

    # Save results
    if not args.no_save:
        results_path = runner.save_results(results, prefix=f"non_iid_{prefix}")
        report_path = runner.output_dir / f"report_{prefix}.txt"
        runner.generate_report(results, output_file=report_path)

        print(f"\nResults saved to: {results_path}")
        print(f"Report saved to: {report_path}")

    # Exit code based on validation success
    n_pass = sum(1 for r in results.values() if r.detection_success and r.accuracy_success)
    if n_pass == len(results):
        print("\nExit code 0: All scenarios passed")
        sys.exit(0)
    else:
        print(f"\nExit code 1: {len(results) - n_pass} scenario(s) did not pass")
        sys.exit(1)


if __name__ == "__main__":
    main()
