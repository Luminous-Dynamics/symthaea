#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Symthaea Summary Formatter
Reference: BENCHMARKING_STRATEGY.md Section 36

Formats aggregated results for GitHub Actions summary.
"""

import argparse
import json
from pathlib import Path


def format_duration(ns: float) -> str:
    """Format nanoseconds to human-readable."""
    if ns is None:
        return "N/A"
    if ns < 1000:
        return f"{ns:.2f}ns"
    elif ns < 1_000_000:
        return f"{ns/1000:.2f}μs"
    elif ns < 1_000_000_000:
        return f"{ns/1_000_000:.2f}ms"
    else:
        return f"{ns/1_000_000_000:.2f}s"


def format_summary(data: dict) -> str:
    """Format aggregated data as markdown summary."""
    lines = []

    # Header
    metadata = data.get("metadata", {})
    lines.append(f"### Benchmark Run Summary")
    lines.append(f"")
    lines.append(f"- **Commit**: `{metadata.get('commit', 'unknown')[:8]}`")
    lines.append(f"- **Total Benchmarks**: {metadata.get('total_benchmarks', 0)}")
    lines.append(f"- **Timestamp**: {metadata.get('timestamp', 'unknown')}")
    lines.append("")

    # Category summary
    by_category = data.get("summary", {}).get("by_category", {})
    if by_category:
        lines.append("### Results by Category")
        lines.append("")
        lines.append("| Category | Count | Avg Time |")
        lines.append("|----------|-------|----------|")

        for cat, cat_data in sorted(by_category.items()):
            count = cat_data.get("count", 0)
            avg = cat_data.get("aggregate_mean")
            avg_str = format_duration(avg) if avg else "N/A"
            lines.append(f"| {cat.title()} | {count} | {avg_str} |")

        lines.append("")

    # Consciousness highlights
    consciousness = data.get("consciousness", {})
    topologies = consciousness.get("topologies", {})
    if topologies:
        lines.append("### Φ Topology Highlights")
        lines.append("")

        # Sort by phi value
        sorted_topos = sorted(
            topologies.items(),
            key=lambda x: x[1].get("phi", 0) or 0,
            reverse=True
        )

        for i, (name, values) in enumerate(sorted_topos[:5], 1):
            phi = values.get("phi")
            if phi:
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                lines.append(f"{medal} **{name}**: Φ = {phi:.4f}")

        lines.append("")

    # Ethics highlights
    ethics = data.get("ethics", {})
    moral = ethics.get("moral_reasoning", {})
    if moral:
        lines.append("### Ethics Scores")
        lines.append("")

        for name, values in list(moral.items())[:5]:
            score = values.get("score")
            if score:
                lines.append(f"- **{name}**: {score:.2%}")

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Format benchmark summary for GitHub Actions"
    )
    parser.add_argument(
        "--results", "-r",
        type=Path,
        required=True,
        help="Path to aggregated results JSON"
    )

    args = parser.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    print(format_summary(data))


if __name__ == "__main__":
    main()
