#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check a ForecastBench-style JSON report against a conservative baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 - CLI should surface concise failures
        raise SystemExit(f"failed to read {path}: {exc}") from exc


def fail(message: str) -> None:
    print(f"[forecastbench-baseline] FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_json(args.baseline)
    report = load_json(args.report)

    if baseline.get("benchmark") and report.get("benchmark") != baseline["benchmark"]:
        fail(f"benchmark mismatch: {report.get('benchmark')} != {baseline['benchmark']}")
    if baseline.get("source") and report.get("source") != baseline["source"]:
        fail(f"source mismatch: {report.get('source')} != {baseline['source']}")

    task_count = int(report.get("task_count", 0))
    resolved_count = int(report.get("resolved_count", 0))
    if task_count < int(baseline.get("min_task_count", 0)):
        fail(f"task_count {task_count} < {baseline['min_task_count']}")
    if resolved_count < int(baseline.get("min_resolved_count", 0)):
        fail(f"resolved_count {resolved_count} < {baseline['min_resolved_count']}")

    if baseline.get("require_quality_gate_passed", False) and not report.get(
        "quality_gate_passed", False
    ):
        fail("quality_gate_passed is false")

    checks = [
        ("brier_score", "max_brier_score", "<="),
        ("expected_calibration_error", "max_expected_calibration_error", "<="),
        ("accuracy_at_50", "min_accuracy_at_50", ">="),
        ("router_trust_multiplier", "min_router_trust_multiplier", ">="),
    ]
    for report_key, baseline_key, op in checks:
        if baseline_key not in baseline:
            continue
        value = report.get(report_key)
        if value is None:
            fail(f"{report_key} missing from report")
        threshold = float(baseline[baseline_key])
        if op == "<=" and float(value) > threshold:
            fail(f"{report_key} {value:.6f} > {threshold:.6f}")
        if op == ">=" and float(value) < threshold:
            fail(f"{report_key} {value:.6f} < {threshold:.6f}")

    print("[forecastbench-baseline] PASS")
    print(f"  report: {args.report}")
    print(
        "  brier={:.4f} ece={:.4f} accuracy={:.4f}".format(
            float(report["brier_score"]),
            float(report["expected_calibration_error"]),
            float(report["accuracy_at_50"]),
        )
    )


if __name__ == "__main__":
    main()
