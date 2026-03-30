#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Symthaea Metrics Aggregator
Reference: BENCHMARKING_STRATEGY.md Section 36

Aggregates benchmark results from multiple sources into unified format.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Optional


def find_result_files(artifacts_path: Path) -> list[Path]:
    """Find all benchmark result files in artifacts directory."""
    result_files = []

    for pattern in ["**/estimates.json", "**/results.json", "**/*.json"]:
        result_files.extend(artifacts_path.glob(pattern))

    return [f for f in result_files if f.is_file()]


def parse_criterion_estimate(file_path: Path) -> Optional[dict]:
    """Parse a Criterion estimates.json file."""
    try:
        with open(file_path) as f:
            data = json.load(f)

        return {
            "mean": data.get("mean", {}).get("point_estimate"),
            "std_dev": data.get("std_dev", {}).get("point_estimate"),
            "median": data.get("median", {}).get("point_estimate"),
            "confidence_interval": {
                "lower": data.get("mean", {}).get("confidence_interval", {}).get("lower_bound"),
                "upper": data.get("mean", {}).get("confidence_interval", {}).get("upper_bound"),
            }
        }
    except (json.JSONDecodeError, KeyError):
        return None


def aggregate_by_category(results: list[dict]) -> dict:
    """Aggregate results by benchmark category."""
    categories = {}

    for result in results:
        category = result.get("category", "unknown")
        if category not in categories:
            categories[category] = {
                "benchmarks": [],
                "means": [],
                "count": 0,
            }

        categories[category]["benchmarks"].append(result)
        if result.get("mean") is not None:
            categories[category]["means"].append(result["mean"])
        categories[category]["count"] += 1

    # Calculate category statistics
    for category, data in categories.items():
        means = data["means"]
        if means:
            data["aggregate_mean"] = mean(means)
            data["aggregate_std"] = stdev(means) if len(means) > 1 else 0
        else:
            data["aggregate_mean"] = None
            data["aggregate_std"] = None

    return categories


def aggregate_phi_results(results: list[dict]) -> dict:
    """Aggregate consciousness/Φ specific results."""
    phi_results = {
        "topologies": {},
        "dimensional_sweep": {},
        "correlations": {},
    }

    for result in results:
        if "phi" in result.get("name", "").lower() or result.get("category") == "consciousness":
            name = result.get("name", "")

            # Topology results
            if "topology" in name:
                topo_name = name.replace("phi_", "").replace("_topology", "")
                phi_results["topologies"][topo_name] = {
                    "phi": result.get("mean"),
                    "std_dev": result.get("std_dev"),
                }

            # Dimensional sweep
            elif "dimensional" in name or "hypercube" in name:
                dim = result.get("dimension", name)
                phi_results["dimensional_sweep"][str(dim)] = {
                    "phi": result.get("mean"),
                    "std_dev": result.get("std_dev"),
                }

    return phi_results


def aggregate_ethics_results(results: list[dict]) -> dict:
    """Aggregate ethics/safety specific results."""
    ethics_results = {
        "moral_reasoning": {},
        "fairness": {},
        "safety": {},
    }

    for result in results:
        if result.get("category") == "ethics":
            name = result.get("name", "")

            if "bias" in name.lower() or "bbq" in name.lower() or "wino" in name.lower():
                subcategory = "fairness"
            elif "sycophancy" in name.lower() or "power" in name.lower() or "safety" in name.lower():
                subcategory = "safety"
            else:
                subcategory = "moral_reasoning"

            ethics_results[subcategory][name] = {
                "score": result.get("mean"),
                "std_dev": result.get("std_dev"),
            }

    return ethics_results


def create_aggregated_output(
    artifacts_path: Path,
    commit: Optional[str] = None
) -> dict:
    """Create complete aggregated output."""
    result_files = find_result_files(artifacts_path)

    all_results = []
    for file_path in result_files:
        parsed = parse_criterion_estimate(file_path)
        if parsed:
            # Extract benchmark name from path
            bench_name = file_path.parent.parent.name
            category = bench_name.split("_")[0] if "_" in bench_name else "general"

            all_results.append({
                "name": bench_name,
                "category": category,
                "source_file": str(file_path),
                **parsed,
            })

    aggregated = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "commit": commit,
            "total_benchmarks": len(all_results),
            "total_files_processed": len(result_files),
        },
        "summary": {
            "by_category": aggregate_by_category(all_results),
        },
        "consciousness": aggregate_phi_results(all_results),
        "ethics": aggregate_ethics_results(all_results),
        "all_results": all_results,
    }

    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate benchmark metrics from multiple sources"
    )
    parser.add_argument(
        "--artifacts", "-a",
        type=Path,
        required=True,
        help="Path to artifacts directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--commit", "-c",
        help="Commit SHA for metadata"
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output"
    )

    args = parser.parse_args()

    aggregated = create_aggregated_output(args.artifacts, args.commit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(aggregated, f, indent=2 if args.pretty else None)

    print(f"Aggregated {aggregated['metadata']['total_benchmarks']} benchmarks")
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
