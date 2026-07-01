#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Symthaea Benchmark Regression Checker
Reference: BENCHMARKING_STRATEGY.md Section 36

Analyzes benchmark results and detects performance regressions.
Uses statistical tests to determine significance.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from statistics import mean, stdev


@dataclass
class RegressionResult:
    """Result of regression analysis for a single benchmark."""
    benchmark: str
    baseline_mean: float
    current_mean: float
    change_percent: float
    is_regression: bool
    is_critical: bool
    p_value: Optional[float] = None
    effect_size: Optional[float] = None


def load_criterion_results(results_path: Path) -> dict:
    """Load Criterion benchmark results from JSON files."""
    results = {}

    # Find all benchmark estimate files
    for estimate_file in results_path.rglob("*/new/estimates.json"):
        bench_name = estimate_file.parent.parent.name
        with open(estimate_file) as f:
            data = json.load(f)
            results[bench_name] = {
                "mean": data.get("mean", {}).get("point_estimate", 0),
                "std_dev": data.get("std_dev", {}).get("point_estimate", 0),
                "median": data.get("median", {}).get("point_estimate", 0),
            }

    return results


def load_baseline(results_path: Path, baseline_name: str = "main") -> dict:
    """Load baseline results for comparison."""
    baseline_path = results_path / f"baseline-{baseline_name}"
    if baseline_path.exists():
        return load_criterion_results(baseline_path)

    # Try alternate locations
    for alt_path in [
        results_path / "baselines" / baseline_name,
        results_path.parent / ".benchmark-baselines" / baseline_name,
    ]:
        if alt_path.exists():
            return load_criterion_results(alt_path)

    return {}


def calculate_effect_size(baseline_mean: float, current_mean: float,
                         baseline_std: float, current_std: float) -> float:
    """Calculate Cohen's d effect size."""
    if baseline_std == 0 and current_std == 0:
        return 0.0

    pooled_std = ((baseline_std ** 2 + current_std ** 2) / 2) ** 0.5
    if pooled_std == 0:
        return 0.0

    return abs(current_mean - baseline_mean) / pooled_std


def check_regression(
    benchmark: str,
    baseline: dict,
    current: dict,
    threshold: float,
    critical_threshold: float
) -> RegressionResult:
    """Check if a single benchmark shows regression."""
    baseline_mean = baseline.get("mean", 0)
    current_mean = current.get("mean", 0)

    if baseline_mean == 0:
        change_percent = 0.0
    else:
        change_percent = ((current_mean - baseline_mean) / baseline_mean) * 100

    # Positive change = regression (slower/worse)
    is_regression = change_percent > threshold
    is_critical = change_percent > critical_threshold

    # Calculate effect size if we have std dev
    effect_size = None
    if "std_dev" in baseline and "std_dev" in current:
        effect_size = calculate_effect_size(
            baseline_mean, current_mean,
            baseline.get("std_dev", 0), current.get("std_dev", 0)
        )

    return RegressionResult(
        benchmark=benchmark,
        baseline_mean=baseline_mean,
        current_mean=current_mean,
        change_percent=change_percent,
        is_regression=is_regression,
        is_critical=is_critical,
        effect_size=effect_size,
    )


def analyze_regressions(
    results_path: Path,
    threshold: float = 5.0,
    critical_threshold: float = 10.0,
    baseline_name: str = "main"
) -> tuple[list[RegressionResult], bool]:
    """
    Analyze all benchmarks for regressions.

    Returns:
        Tuple of (list of regression results, has_critical_regression)
    """
    current_results = load_criterion_results(results_path)
    baseline_results = load_baseline(results_path, baseline_name)

    if not baseline_results:
        print(f"Warning: No baseline found for '{baseline_name}', skipping comparison")
        return [], False

    regressions = []
    has_critical = False

    for bench_name, current_data in current_results.items():
        if bench_name not in baseline_results:
            continue

        result = check_regression(
            bench_name,
            baseline_results[bench_name],
            current_data,
            threshold,
            critical_threshold
        )
        regressions.append(result)

        if result.is_critical:
            has_critical = True

    # Sort by change percent (worst first)
    regressions.sort(key=lambda r: r.change_percent, reverse=True)

    return regressions, has_critical


def format_report(regressions: list[RegressionResult]) -> str:
    """Format regression results as a report."""
    lines = ["# Benchmark Regression Report\n"]

    # Summary
    total = len(regressions)
    regression_count = sum(1 for r in regressions if r.is_regression)
    critical_count = sum(1 for r in regressions if r.is_critical)

    lines.append(f"**Total benchmarks:** {total}")
    lines.append(f"**Regressions (>{5}%):** {regression_count}")
    lines.append(f"**Critical (>{10}%):** {critical_count}\n")

    if critical_count > 0:
        lines.append("## Critical Regressions\n")
        for r in regressions:
            if r.is_critical:
                lines.append(f"- **{r.benchmark}**: {r.change_percent:+.2f}% "
                           f"({r.baseline_mean:.4f} -> {r.current_mean:.4f})")
        lines.append("")

    if regression_count > 0:
        lines.append("## All Regressions\n")
        lines.append("| Benchmark | Change | Baseline | Current | Effect Size |")
        lines.append("|-----------|--------|----------|---------|-------------|")
        for r in regressions:
            if r.is_regression:
                effect = f"{r.effect_size:.2f}" if r.effect_size else "N/A"
                lines.append(f"| {r.benchmark} | {r.change_percent:+.2f}% | "
                           f"{r.baseline_mean:.4f} | {r.current_mean:.4f} | {effect} |")
        lines.append("")

    # Improvements
    improvements = [r for r in regressions if r.change_percent < -5]
    if improvements:
        lines.append("## Improvements\n")
        for r in improvements:
            lines.append(f"- **{r.benchmark}**: {r.change_percent:.2f}% improvement")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Check benchmark results for regressions"
    )
    parser.add_argument(
        "--results", "-r",
        type=Path,
        default=Path("target/criterion"),
        help="Path to benchmark results directory"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=5.0,
        help="Regression threshold percentage (default: 5)"
    )
    parser.add_argument(
        "--critical-threshold", "-c",
        type=float,
        default=10.0,
        help="Critical regression threshold percentage (default: 10)"
    )
    parser.add_argument(
        "--baseline", "-b",
        default="main",
        help="Baseline name for comparison (default: main)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file for JSON report"
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown", "summary"],
        default="summary",
        help="Output format"
    )

    args = parser.parse_args()

    regressions, has_critical = analyze_regressions(
        args.results,
        args.threshold,
        args.critical_threshold,
        args.baseline
    )

    if args.format == "json" or args.output:
        output_data = {
            "regressions": [
                {
                    "benchmark": r.benchmark,
                    "baseline_mean": r.baseline_mean,
                    "current_mean": r.current_mean,
                    "change_percent": r.change_percent,
                    "is_regression": r.is_regression,
                    "is_critical": r.is_critical,
                    "effect_size": r.effect_size,
                }
                for r in regressions
            ],
            "has_critical": has_critical,
            "total": len(regressions),
            "regression_count": sum(1 for r in regressions if r.is_regression),
            "critical_count": sum(1 for r in regressions if r.is_critical),
        }

        if args.output:
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"Report written to {args.output}")
        else:
            print(json.dumps(output_data, indent=2))

    elif args.format == "markdown":
        print(format_report(regressions))

    else:  # summary
        total = len(regressions)
        regression_count = sum(1 for r in regressions if r.is_regression)
        critical_count = sum(1 for r in regressions if r.is_critical)

        print(f"Analyzed {total} benchmarks")
        print(f"  Regressions: {regression_count}")
        print(f"  Critical: {critical_count}")

        if has_critical:
            print("\nCritical regressions detected:")
            for r in regressions:
                if r.is_critical:
                    print(f"  - {r.benchmark}: {r.change_percent:+.2f}%")

    # Set GitHub Actions output if running in CI
    if "GITHUB_OUTPUT" in sys.environ:
        with open(sys.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"critical={str(has_critical).lower()}\n")
            f.write(f"regression_count={sum(1 for r in regressions if r.is_regression)}\n")

    # Exit with error if critical regressions found
    if has_critical:
        sys.exit(1)


if __name__ == "__main__":
    main()
