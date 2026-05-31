#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Refresh the ForecastBench-style calibration baseline from a fresh report."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    text = path.read_text().strip()
    start = text.find("{")
    if start < 0:
        raise SystemExit(f"JSON object not found in {path}")
    return json.loads(text[start:])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--baseline",
        default=Path("tests/fixtures/forecastbench_baseline.json"),
        type=Path,
    )
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    report = load_json(args.report)
    brier = float(report["brier_score"])
    ece = float(report["expected_calibration_error"])
    accuracy = float(report["accuracy_at_50"])
    baseline = {
        "benchmark": report["benchmark"],
        "source": report["source"],
        "min_task_count": int(report["task_count"]),
        "min_resolved_count": int(report["resolved_count"]),
        "max_brier_score": round(min(0.25, brier * 1.20), 4),
        "max_expected_calibration_error": round(min(0.25, ece * 1.20), 4),
        "min_accuracy_at_50": round(max(0.0, accuracy * 0.95), 4),
        "require_quality_gate_passed": bool(report["quality_gate_passed"]),
        "min_router_trust_multiplier": round(
            float(report.get("router_trust_multiplier", 0.0)) * 0.90,
            4,
        ),
        "notes": args.notes
        or f"Refreshed from forecastbench_eval at {dt.datetime.now(dt.UTC).isoformat()}",
    }

    args.baseline.parent.mkdir(parents=True, exist_ok=True)
    args.baseline.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"Baseline refreshed: {args.baseline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
