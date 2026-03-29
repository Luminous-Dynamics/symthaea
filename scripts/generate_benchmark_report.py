#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Symthaea Benchmark Report Generator
Reference: BENCHMARKING_STRATEGY.md Section 36

Generates comprehensive markdown reports from benchmark results.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def load_results(results_path: Path) -> dict:
    """Load all benchmark results from Criterion output."""
    benchmarks = {}

    for estimate_file in results_path.rglob("*/new/estimates.json"):
        bench_name = estimate_file.parent.parent.name
        category = bench_name.split("_")[0] if "_" in bench_name else "general"

        with open(estimate_file) as f:
            data = json.load(f)

        if category not in benchmarks:
            benchmarks[category] = {}

        benchmarks[category][bench_name] = {
            "mean": data.get("mean", {}).get("point_estimate", 0),
            "std_dev": data.get("std_dev", {}).get("point_estimate", 0),
            "median": data.get("median", {}).get("point_estimate", 0),
            "unit": "ns",  # Default Criterion unit
        }

    return benchmarks


def format_duration(ns: float) -> str:
    """Format nanoseconds to human-readable duration."""
    if ns < 1000:
        return f"{ns:.2f} ns"
    elif ns < 1_000_000:
        return f"{ns/1000:.2f} μs"
    elif ns < 1_000_000_000:
        return f"{ns/1_000_000:.2f} ms"
    else:
        return f"{ns/1_000_000_000:.2f} s"


def generate_summary_table(benchmarks: dict) -> str:
    """Generate summary table for all categories."""
    lines = ["## Summary by Category\n"]
    lines.append("| Category | Benchmarks | Avg Time | Min | Max |")
    lines.append("|----------|------------|----------|-----|-----|")

    for category, benches in sorted(benchmarks.items()):
        count = len(benches)
        times = [b["mean"] for b in benches.values()]
        avg = sum(times) / count if count > 0 else 0

        lines.append(
            f"| {category} | {count} | {format_duration(avg)} | "
            f"{format_duration(min(times))} | {format_duration(max(times))} |"
        )

    return "\n".join(lines)


def generate_detailed_results(benchmarks: dict) -> str:
    """Generate detailed results for each benchmark."""
    lines = ["## Detailed Results\n"]

    for category, benches in sorted(benchmarks.items()):
        lines.append(f"### {category.title()}\n")
        lines.append("| Benchmark | Mean | Std Dev | Median |")
        lines.append("|-----------|------|---------|--------|")

        for name, data in sorted(benches.items()):
            lines.append(
                f"| {name} | {format_duration(data['mean'])} | "
                f"±{format_duration(data['std_dev'])} | "
                f"{format_duration(data['median'])} |"
            )

        lines.append("")

    return "\n".join(lines)


def generate_phi_section(results_path: Path) -> Optional[str]:
    """Generate Φ correlation section if consciousness benchmarks exist."""
    phi_results = results_path / "consciousness"
    if not phi_results.exists():
        return None

    lines = ["## Consciousness Metrics (Φ)\n"]
    lines.append("### Topology Rankings\n")

    # Look for phi results
    for result_file in phi_results.glob("*/new/estimates.json"):
        topology_name = result_file.parent.parent.name
        with open(result_file) as f:
            data = json.load(f)
        phi_value = data.get("mean", {}).get("point_estimate", 0)
        lines.append(f"- **{topology_name}**: Φ = {phi_value:.4f}")

    return "\n".join(lines) if len(lines) > 2 else None


def generate_ethics_section(results_path: Path) -> Optional[str]:
    """Generate ethics benchmark section if results exist."""
    ethics_results = results_path / "ethics"
    if not ethics_results.exists():
        return None

    lines = ["## Ethics & Safety Metrics\n"]

    for result_file in ethics_results.glob("*/new/estimates.json"):
        bench_name = result_file.parent.parent.name
        with open(result_file) as f:
            data = json.load(f)
        accuracy = data.get("mean", {}).get("point_estimate", 0)
        lines.append(f"- **{bench_name}**: {accuracy:.2%}")

    return "\n".join(lines) if len(lines) > 1 else None


def generate_report(
    results_path: Path,
    commit: str,
    suite: str,
    output_path: Optional[Path] = None
) -> str:
    """Generate complete benchmark report."""
    benchmarks = load_results(results_path)

    lines = [
        "# Symthaea Benchmark Report\n",
        f"**Commit:** `{commit[:8]}`",
        f"**Suite:** {suite}",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"**Total Benchmarks:** {sum(len(b) for b in benchmarks.values())}\n",
        "---\n",
    ]

    # Summary table
    lines.append(generate_summary_table(benchmarks))
    lines.append("\n---\n")

    # Detailed results
    lines.append(generate_detailed_results(benchmarks))

    # Special sections
    phi_section = generate_phi_section(results_path)
    if phi_section:
        lines.append("---\n")
        lines.append(phi_section)

    ethics_section = generate_ethics_section(results_path)
    if ethics_section:
        lines.append("---\n")
        lines.append(ethics_section)

    # Footer
    lines.append("\n---\n")
    lines.append(
        "*Report generated by Symthaea Benchmark Suite. "
        "See [BENCHMARKING_STRATEGY.md](../BENCHMARKING_STRATEGY.md) for methodology.*"
    )

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        print(f"Report written to {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark report from Criterion results"
    )
    parser.add_argument(
        "--results", "-r",
        type=Path,
        default=Path("target/criterion"),
        help="Path to benchmark results directory"
    )
    parser.add_argument(
        "--commit", "-c",
        default="unknown",
        help="Commit SHA"
    )
    parser.add_argument(
        "--suite", "-s",
        default="standard",
        help="Benchmark suite name"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path"
    )

    args = parser.parse_args()

    report = generate_report(
        args.results,
        args.commit,
        args.suite,
        args.output
    )

    if not args.output:
        print(report)


if __name__ == "__main__":
    main()
