#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""
Symthaea Dashboard Poster
Reference: BENCHMARKING_STRATEGY.md Section 36.3

Posts benchmark results to the live dashboard.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import urllib.request
import urllib.error


def load_results(results_path: Path) -> dict:
    """Load benchmark results."""
    metrics = {
        "categories": {},
        "overall": {
            "total_benchmarks": 0,
            "total_time_ns": 0,
        }
    }

    for estimate_file in results_path.rglob("*/new/estimates.json"):
        bench_name = estimate_file.parent.parent.name
        category = bench_name.split("_")[0] if "_" in bench_name else "general"

        with open(estimate_file) as f:
            data = json.load(f)

        mean = data.get("mean", {}).get("point_estimate", 0)

        if category not in metrics["categories"]:
            metrics["categories"][category] = {
                "benchmarks": [],
                "total_time_ns": 0,
            }

        metrics["categories"][category]["benchmarks"].append({
            "name": bench_name,
            "mean_ns": mean,
            "std_dev_ns": data.get("std_dev", {}).get("point_estimate", 0),
        })
        metrics["categories"][category]["total_time_ns"] += mean
        metrics["overall"]["total_benchmarks"] += 1
        metrics["overall"]["total_time_ns"] += mean

    return metrics


def post_to_dashboard(
    results: dict,
    commit: str,
    suite: str,
    branch: str,
    dashboard_url: Optional[str] = None,
    token: Optional[str] = None
) -> bool:
    """Post results to the dashboard API."""
    dashboard_url = dashboard_url or os.environ.get(
        "DASHBOARD_URL",
        "https://dashboard.symthaea.dev/api/benchmarks"
    )
    token = token or os.environ.get("DASHBOARD_TOKEN")

    if not token:
        print("Warning: No dashboard token provided, skipping upload")
        return False

    payload = {
        "commit": commit,
        "suite": suite,
        "branch": branch,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": results,
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            dashboard_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            if response.status == 200 or response.status == 201:
                print(f"Successfully posted results to dashboard")
                return True
            else:
                print(f"Dashboard returned status {response.status}")
                return False

    except urllib.error.URLError as e:
        print(f"Failed to post to dashboard: {e}")
        return False


def save_local_copy(results: dict, commit: str, suite: str, output_path: Path):
    """Save a local copy of the results for offline dashboard."""
    data = {
        "commit": commit,
        "suite": suite,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "metrics": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Local copy saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Post benchmark results to dashboard"
    )
    parser.add_argument(
        "--results", "-r",
        type=Path,
        default=Path("target/criterion"),
        help="Path to benchmark results"
    )
    parser.add_argument(
        "--commit", "-c",
        required=True,
        help="Commit SHA"
    )
    parser.add_argument(
        "--suite", "-s",
        default="standard",
        help="Benchmark suite name"
    )
    parser.add_argument(
        "--branch", "-b",
        default="main",
        help="Branch name"
    )
    parser.add_argument(
        "--dashboard-url",
        help="Dashboard API URL (or set DASHBOARD_URL env)"
    )
    parser.add_argument(
        "--token",
        help="Dashboard API token (or set DASHBOARD_TOKEN env)"
    )
    parser.add_argument(
        "--local-output",
        type=Path,
        help="Save local copy for offline dashboard"
    )

    args = parser.parse_args()

    results = load_results(args.results)

    # Post to remote dashboard
    post_to_dashboard(
        results,
        args.commit,
        args.suite,
        args.branch,
        args.dashboard_url,
        args.token
    )

    # Also save local copy
    if args.local_output:
        save_local_copy(results, args.commit, args.suite, args.local_output)
    else:
        local_path = Path(f".benchmark-history/{args.commit[:8]}_{args.suite}.json")
        save_local_copy(results, args.commit, args.suite, local_path)


if __name__ == "__main__":
    main()
