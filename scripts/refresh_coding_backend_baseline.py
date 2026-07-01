#!/usr/bin/env python3
"""Refresh the coding backend regression baseline from a benchmark report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--baseline",
        default=Path("tests/fixtures/coding_backends_baseline.json"),
        type=Path,
    )
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    report = load_json(args.report.read_text())
    baseline = {
        "benchmark": report["benchmark"],
        "require_feature_geodesic": report.get("feature_geodesic", True),
        "min_task_count": report["task_count"],
        "min_pass_rate": round(report["pass_rate"] * 0.95, 3),
        "min_quality_pass_rate": round(
            report.get("quality_pass_rate", report["pass_rate"]) * 0.95,
            3,
        ),
        "max_mean_attempts_per_task": round(report["mean_attempts_per_task"] * 1.2, 2),
        "max_certificates_sheaf_incoherent": report["certificates_sheaf_incoherent"],
        "min_certificates_with_sheaf": report["certificates_with_sheaf"],
        "require_broca_eval_gate_passed": report.get("broca_eval_gate_passed", False),
        "min_broca_selection_score": round(
            report.get("broca_selection_score", 0.0) * 0.95,
            3,
        ),
        "min_repair_success_rate": round(
            report.get("repair_success_rate", 0.0) * 0.9,
            3,
        ),
        "min_repair_attempts": report.get("repair_attempts", 0),
        "min_repair_successes": report.get("repair_successes", 0),
        "max_repair_attempts": max(
            report.get("repair_attempts", 0),
            int(report.get("repair_attempts", 0) * 1.2),
        ),
        "min_repair_prior_uses": report.get("repair_prior_uses", 0),
        "min_repair_prior_label_count": report.get("repair_prior_label_count", 0),
        "min_repair_memory_hits": report.get("repair_memory_hits", 0),
        "min_repair_memory_successes": report.get("repair_memory_successes", 0),
        "min_repair_memory_success_rate": round(
            report.get("repair_memory_success_rate", 0.0) * 0.9,
            3,
        ),
        "min_geodesic_rejection_shadow_hits": report.get(
            "geodesic_rejection_shadow_hits", 0
        ),
        "min_geodesic_rejection_shadow_true_positives": report.get(
            "geodesic_rejection_shadow_true_positives", 0
        ),
        "max_geodesic_rejection_shadow_false_positives": report.get(
            "geodesic_rejection_shadow_false_positives", 0
        ),
        "max_hard_geodesic_rejections": report.get("hard_geodesic_rejections", 0),
        "min_category_pass_rates": {
            category: round(data["pass_rate"] * 0.9, 3)
            for category, data in sorted(report.get("category_pass_rates", {}).items())
        },
        "notes": args.notes
        or f"Refreshed from benchmark_coding_backends at {dt.datetime.now(dt.UTC).isoformat()}",
    }

    args.baseline.parent.mkdir(parents=True, exist_ok=True)
    args.baseline.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"Baseline refreshed: {args.baseline}")
    return 0


def load_json(text: str) -> dict[str, Any]:
    trimmed = text.strip()
    start = trimmed.find("{")
    if start < 0:
        raise SystemExit("JSON object not found in report")
    return json.loads(trimmed[start:])


if __name__ == "__main__":
    raise SystemExit(main())
