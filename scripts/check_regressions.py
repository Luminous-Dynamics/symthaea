#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Symthaea Benchmark Regression Checker
Reference: BENCHMARKING_STRATEGY.md Section 36

Analyzes Criterion baseline comparisons and reports bootstrap confidence intervals.
The checker fails closed when required comparison evidence is incomplete.
"""

import argparse
import json
import os
import sys
from math import isfinite
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


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
    change_ci_lower_percent: Optional[float] = None
    change_ci_upper_percent: Optional[float] = None
    confidence_level: Optional[float] = None
    is_inconclusive: bool = False


def _read_estimate(estimate_file: Path) -> dict:
    """Read Criterion point estimates and reject malformed or non-finite data."""
    try:
        with open(estimate_file, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read Criterion estimates from {estimate_file}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Criterion estimates must be a JSON object: {estimate_file}")

    def point_estimate(metric: str) -> float:
        section = data.get(metric)
        if not isinstance(section, dict):
            raise ValueError(f"Missing or invalid '{metric}' section in {estimate_file}")
        value = section.get("point_estimate")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError(f"Missing, non-numeric, or non-finite '{metric}' estimate in {estimate_file}")
        return float(value)

    values = {
        "mean": point_estimate("mean"),
        "std_dev": point_estimate("std_dev"),
        "median": point_estimate("median"),
    }
    if values["mean"] <= 0 or values["median"] <= 0 or values["std_dev"] < 0:
        raise ValueError(f"Non-positive/invalid Criterion estimate in {estimate_file}")
    return values


def _benchmark_key(estimate_file: Path, root: Path) -> str:
    """Use the relative benchmark path to avoid collisions between equal leaf names."""
    return estimate_file.parent.parent.relative_to(root).as_posix()


def load_criterion_results(results_path: Path) -> dict:
    """Load current Criterion results from the standard */new/estimates.json layout."""
    results = {}
    if not results_path.exists():
        return results

    for estimate_file in results_path.rglob("*/new/estimates.json"):
        bench_name = _benchmark_key(estimate_file, results_path)
        results[bench_name] = _read_estimate(estimate_file)

    return results


def load_baseline(results_path: Path, baseline_name: str = "main") -> dict:
    """Load a named Criterion baseline from each benchmark's standard layout."""
    results = {}
    if not results_path.exists():
        return results

    # Criterion stores a named baseline beside "new", e.g.
    # target/criterion/my_benchmark/main/estimates.json.
    for estimate_file in results_path.rglob(f"*/{baseline_name}/estimates.json"):
        bench_name = _benchmark_key(estimate_file, results_path)
        results[bench_name] = _read_estimate(estimate_file)

    # Keep support for the legacy exported baseline-directory layout.
    if results:
        return results
    for alt_path in (
        results_path / f"baseline-{baseline_name}",
        results_path / "baselines" / baseline_name,
        results_path.parent / ".benchmark-baselines" / baseline_name,
    ):
        if alt_path.exists():
            results = load_criterion_results(alt_path)
            if results:
                return results

    return {}


def _read_change_estimate(estimate_file: Path) -> dict:
    """Read Criterion's bootstrap interval for current-versus-baseline relative change."""
    try:
        with open(estimate_file, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read Criterion change estimates from {estimate_file}: {exc}") from exc

    if not isinstance(data, dict) or not isinstance(data.get("mean"), dict):
        raise ValueError(f"Missing Criterion relative-mean estimate in {estimate_file}")
    mean = data["mean"]
    ci = mean.get("confidence_interval")
    point = mean.get("point_estimate")
    if not isinstance(ci, dict):
        raise ValueError(f"Missing Criterion change confidence interval in {estimate_file}")

    lower = ci.get("lower_bound")
    upper = ci.get("upper_bound")
    confidence_level = ci.get("confidence_level")
    numbers = (point, lower, upper, confidence_level)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value)
           for value in numbers):
        raise ValueError(f"Non-numeric or non-finite Criterion change estimate in {estimate_file}")
    if not (0.0 < confidence_level < 1.0) or lower > upper:
        raise ValueError(f"Invalid Criterion change interval in {estimate_file}")
    if lower <= -1.0:
        raise ValueError(f"Invalid relative timing change lower bound in {estimate_file}")

    return {
        "mean_change": float(point),
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "confidence_level": float(confidence_level),
    }


def load_criterion_changes(results_path: Path) -> dict:
    """Load Criterion's own bootstrap relative-change estimates."""
    results = {}
    if not results_path.exists():
        return results
    for estimate_file in results_path.rglob("*/change/estimates.json"):
        benchmark = _benchmark_key(estimate_file, results_path)
        results[benchmark] = _read_change_estimate(estimate_file)
    return results


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

    # Prefer Criterion's bootstrapped relative-change interval when available.
    # With an interval, call a regression only when its lower bound exceeds the
    # threshold; if the interval crosses either decision boundary, report uncertainty.
    ci_lower = ci_upper = confidence_level = None
    if "change_estimate" in current:
        change = current["change_estimate"]
        change_percent = change["mean_change"] * 100.0
        ci_lower = change["lower_bound"] * 100.0
        ci_upper = change["upper_bound"] * 100.0
        confidence_level = change["confidence_level"]
        is_regression = ci_lower > threshold
        is_critical = ci_lower > critical_threshold
        is_inconclusive = (
            ci_lower <= threshold < ci_upper
            or ci_lower <= critical_threshold < ci_upper
        )
    else:
        # Backwards-compatible exploratory mode for saved data without Criterion
        # comparison samples. The required-baseline CI workflow rejects this mode.
        is_regression = change_percent > threshold
        is_critical = change_percent > critical_threshold
        is_inconclusive = False

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
        change_ci_lower_percent=ci_lower,
        change_ci_upper_percent=ci_upper,
        confidence_level=confidence_level,
        is_inconclusive=is_inconclusive,
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
    change_results = load_criterion_changes(results_path)

    if not baseline_results:
        print(f"Warning: No baseline found for '{baseline_name}', skipping comparison")
        return [], False

    regressions = []
    has_critical = False

    for bench_name, current_data in current_results.items():
        if bench_name not in baseline_results:
            continue

        if bench_name in change_results:
            current_data = {**current_data, "change_estimate": change_results[bench_name]}
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


def format_report(
    regressions: list[RegressionResult],
    threshold: float = 5.0,
    critical_threshold: float = 10.0,
) -> str:
    """Format regression results, including confidence intervals and uncertainty."""
    lines = ["# Benchmark Regression Report\n"]

    total = len(regressions)
    regression_count = sum(1 for r in regressions if r.is_regression)
    critical_count = sum(1 for r in regressions if r.is_critical)
    inconclusive_count = sum(1 for r in regressions if r.is_inconclusive)

    lines.append(f"**Total benchmarks:** {total}")
    lines.append(f"**Regressions (>{threshold:g}% supported by interval):** {regression_count}")
    lines.append(f"**Critical (>{critical_threshold:g}% supported by interval):** {critical_count}")
    lines.append(f"**Inconclusive (interval crosses a decision threshold):** {inconclusive_count}\n")

    if critical_count > 0 or regression_count > 0 or inconclusive_count > 0:
        lines.append("## Comparison details\n")
        lines.append("| Benchmark | Change | Criterion CI | Assessment |")
        lines.append("|-----------|--------|--------------|------------|")
        for r in regressions:
            if r.change_ci_lower_percent is not None:
                ci_text = (
                    f"[{r.change_ci_lower_percent:+.2f}%, {r.change_ci_upper_percent:+.2f}%]"
                    f" ({r.confidence_level:.0%})"
                )
            else:
                ci_text = "Not available; point-estimate only"
            if r.is_critical:
                assessment = "Critical regression"
            elif r.is_regression:
                assessment = "Regression"
            elif r.is_inconclusive:
                assessment = "Inconclusive"
            else:
                assessment = "No regression supported at configured threshold"
            lines.append(f"| {r.benchmark} | {r.change_percent:+.2f}% | {ci_text} | {assessment} |")
        lines.append("")

    improvements = [r for r in regressions if r.change_percent < -5]
    if improvements:
        lines.append("## Improvements (point estimate)\n")
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

    parser.add_argument(
        "--require-baseline",
        action="store_true",
        help="Require a complete, exactly matched set of current and baseline benchmarks",
    )

    args = parser.parse_args()

    if (
        not isfinite(args.threshold)
        or not isfinite(args.critical_threshold)
        or args.threshold < 0
        or args.critical_threshold < args.threshold
    ):
        parser.error(
            "Thresholds must be finite and non-negative, and critical-threshold must be >= threshold"
        )

    if args.require_baseline:
        current_results = load_criterion_results(args.results)
        baseline_results = load_baseline(args.results, args.baseline)
        if not current_results:
            parser.error(f"No current Criterion estimates found under {args.results}")
        if not baseline_results:
            parser.error(f"Required Criterion baseline '{args.baseline}' was not found under {args.results}")

        current_names = set(current_results)
        baseline_names = set(baseline_results)
        change_names = set(load_criterion_changes(args.results))
        current_only = sorted(current_names - baseline_names)
        baseline_only = sorted(baseline_names - current_names)
        change_missing = sorted(current_names - change_names)
        unexpected_changes = sorted(change_names - current_names)
        if current_only or baseline_only or change_missing or unexpected_changes:
            details = []
            if current_only:
                details.append("current benchmarks missing a baseline: " + ", ".join(current_only))
            if baseline_only:
                details.append("baseline benchmarks missing current results: " + ", ".join(baseline_only))
            if change_missing:
                details.append("benchmarks missing Criterion change estimates: " + ", ".join(change_missing))
            if unexpected_changes:
                details.append("change estimates without current results: " + ", ".join(unexpected_changes))
            parser.error(
                f"Incomplete Criterion comparison against '{args.baseline}'; "
                + "; ".join(details)
                + ". Refusing to report a clean comparison"
            )

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
                    "is_inconclusive": r.is_inconclusive,
                    "change_ci_lower_percent": r.change_ci_lower_percent,
                    "change_ci_upper_percent": r.change_ci_upper_percent,
                    "confidence_level": r.confidence_level,
                    "effect_size": r.effect_size,
                }
                for r in regressions
            ],
            "has_critical": has_critical,
            "total": len(regressions),
            "regression_count": sum(1 for r in regressions if r.is_regression),
            "critical_count": sum(1 for r in regressions if r.is_critical),
            "inconclusive_count": sum(1 for r in regressions if r.is_inconclusive),
        }

        if args.output:
            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"Report written to {args.output}")
        else:
            print(json.dumps(output_data, indent=2))

    elif args.format == "markdown":
        print(format_report(regressions, args.threshold, args.critical_threshold))

    else:  # summary
        total = len(regressions)
        regression_count = sum(1 for r in regressions if r.is_regression)
        critical_count = sum(1 for r in regressions if r.is_critical)

        print(f"Analyzed {total} benchmarks")
        print(f"  Regressions: {regression_count}")
        print(f"  Critical: {critical_count}")
        print(f"  Inconclusive: {sum(1 for r in regressions if r.is_inconclusive)}")

        if has_critical:
            print("\nCritical regressions detected:")
            for r in regressions:
                if r.is_critical:
                    print(f"  - {r.benchmark}: {r.change_percent:+.2f}%")

    # Set GitHub Actions output if running in CI
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"critical={str(has_critical).lower()}\n")
            f.write(f"regression_count={sum(1 for r in regressions if r.is_regression)}\n")
            f.write(f"inconclusive_count={sum(1 for r in regressions if r.is_inconclusive)}\n")

    # Exit with error if critical regressions found
    if has_critical:
        sys.exit(1)


if __name__ == "__main__":
    main()
