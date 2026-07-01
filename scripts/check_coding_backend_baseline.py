#!/usr/bin/env python3
"""Check coding backend benchmark JSON against a baseline.

Typical use:
  cargo run --example benchmark_coding_backends \
    --features code_generation,geodesic_synthesis \
    -- --json --simulated-llm \
    | python scripts/check_coding_backend_baseline.py \
        --baseline tests/fixtures/coding_backends_baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        help="Benchmark report JSON. Reads stdin when omitted.",
    )
    args = parser.parse_args()

    baseline = load_json(args.baseline.read_text())
    report_text = args.report.read_text() if args.report else sys.stdin.read()
    report = load_json(report_text)

    failures: list[str] = []
    check_equal(report, baseline, "benchmark", failures)
    check_equal(report, baseline, "feature_geodesic", failures, baseline_key="require_feature_geodesic")
    check_min(report, baseline, "task_count", failures, baseline_key="min_task_count")
    check_min(report, baseline, "pass_rate", failures, baseline_key="min_pass_rate")
    check_min(
        report,
        baseline,
        "quality_pass_rate",
        failures,
        baseline_key="min_quality_pass_rate",
    )
    check_max(
        report,
        baseline,
        "mean_attempts_per_task",
        failures,
        baseline_key="max_mean_attempts_per_task",
    )
    check_max(
        report,
        baseline,
        "certificates_sheaf_incoherent",
        failures,
        baseline_key="max_certificates_sheaf_incoherent",
    )
    check_min(
        report,
        baseline,
        "certificates_with_sheaf",
        failures,
        baseline_key="min_certificates_with_sheaf",
    )
    check_equal(
        report,
        baseline,
        "broca_eval_gate_passed",
        failures,
        baseline_key="require_broca_eval_gate_passed",
    )
    check_min(
        report,
        baseline,
        "broca_selection_score",
        failures,
        baseline_key="min_broca_selection_score",
    )
    check_min(
        report,
        baseline,
        "repair_success_rate",
        failures,
        baseline_key="min_repair_success_rate",
    )
    check_min(
        report,
        baseline,
        "repair_attempts",
        failures,
        baseline_key="min_repair_attempts",
    )
    check_min(
        report,
        baseline,
        "repair_successes",
        failures,
        baseline_key="min_repair_successes",
    )
    check_max(
        report,
        baseline,
        "repair_attempts",
        failures,
        baseline_key="max_repair_attempts",
    )
    check_min(
        report,
        baseline,
        "repair_prior_uses",
        failures,
        baseline_key="min_repair_prior_uses",
    )
    check_min(
        report,
        baseline,
        "repair_prior_label_count",
        failures,
        baseline_key="min_repair_prior_label_count",
    )
    check_min(
        report,
        baseline,
        "repair_memory_hits",
        failures,
        baseline_key="min_repair_memory_hits",
    )
    check_min(
        report,
        baseline,
        "repair_memory_successes",
        failures,
        baseline_key="min_repair_memory_successes",
    )
    check_min(
        report,
        baseline,
        "repair_memory_success_rate",
        failures,
        baseline_key="min_repair_memory_success_rate",
    )
    check_min(
        report,
        baseline,
        "geodesic_rejection_shadow_hits",
        failures,
        baseline_key="min_geodesic_rejection_shadow_hits",
    )
    check_min(
        report,
        baseline,
        "geodesic_rejection_shadow_true_positives",
        failures,
        baseline_key="min_geodesic_rejection_shadow_true_positives",
    )
    check_max(
        report,
        baseline,
        "geodesic_rejection_shadow_false_positives",
        failures,
        baseline_key="max_geodesic_rejection_shadow_false_positives",
    )
    check_max(
        report,
        baseline,
        "hard_geodesic_rejections",
        failures,
        baseline_key="max_hard_geodesic_rejections",
    )
    check_category_pass_rates(report, baseline, failures)

    if failures:
        print("coding backend benchmark regression detected:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        print(
            "\nAfter an intentional improvement, refresh with:\n"
            "  python scripts/refresh_coding_backend_baseline.py "
            "--report /path/to/benchmark.json "
            "--baseline tests/fixtures/coding_backends_baseline.json",
            file=sys.stderr,
        )
        return 1

    print(
        "coding backend benchmark OK: "
        f"pass_rate={report.get('pass_rate')} "
        f"quality_pass_rate={report.get('quality_pass_rate')} "
        f"mean_attempts={report.get('mean_attempts_per_task')} "
        f"repair_success_rate={report.get('repair_success_rate')} "
        f"repair_attempts={report.get('repair_attempts')} "
        f"repair_prior_uses={report.get('repair_prior_uses')} "
        f"repair_memory_hits={report.get('repair_memory_hits')} "
        f"repair_memory_success_rate={report.get('repair_memory_success_rate')} "
        f"geodesic_shadow_hits={report.get('geodesic_rejection_shadow_hits')} "
        f"geodesic_shadow_fp={report.get('geodesic_rejection_shadow_false_positives')} "
        f"hard_geodesic_rejections={report.get('hard_geodesic_rejections')} "
        f"sheaf_incoherent={report.get('certificates_sheaf_incoherent')} "
        f"broca_selection_score={report.get('broca_selection_score')}"
    )
    return 0


def load_json(text: str) -> dict[str, Any]:
    trimmed = text.strip()
    if not trimmed:
        raise SystemExit("empty JSON input")
    start = trimmed.find("{")
    if start < 0:
        raise SystemExit("JSON object not found in input")
    return json.loads(trimmed[start:])


def check_equal(
    report: dict[str, Any],
    baseline: dict[str, Any],
    report_key: str,
    failures: list[str],
    *,
    baseline_key: str | None = None,
) -> None:
    baseline_key = baseline_key or report_key
    if baseline_key not in baseline:
        return
    actual = report.get(report_key)
    expected = baseline[baseline_key]
    if actual != expected:
        failures.append(f"{report_key}={actual!r}, expected {expected!r}")


def check_min(
    report: dict[str, Any],
    baseline: dict[str, Any],
    report_key: str,
    failures: list[str],
    *,
    baseline_key: str,
) -> None:
    if baseline_key not in baseline:
        return
    actual = report.get(report_key)
    minimum = baseline[baseline_key]
    if actual is None or actual < minimum:
        failures.append(f"{report_key}={actual!r}, below minimum {minimum!r}")


def check_max(
    report: dict[str, Any],
    baseline: dict[str, Any],
    report_key: str,
    failures: list[str],
    *,
    baseline_key: str,
) -> None:
    if baseline_key not in baseline:
        return
    actual = report.get(report_key)
    maximum = baseline[baseline_key]
    if actual is None or actual > maximum:
        failures.append(f"{report_key}={actual!r}, above maximum {maximum!r}")


def check_category_pass_rates(
    report: dict[str, Any], baseline: dict[str, Any], failures: list[str]
) -> None:
    expected = baseline.get("min_category_pass_rates", {})
    categories = report.get("category_pass_rates", {})
    for category, minimum in expected.items():
        actual = categories.get(category, {}).get("pass_rate")
        if actual is None:
            failures.append(f"category {category!r} missing from report")
        elif actual < minimum:
            failures.append(
                f"category {category!r} pass_rate={actual!r}, below minimum {minimum!r}"
            )


if __name__ == "__main__":
    raise SystemExit(main())
