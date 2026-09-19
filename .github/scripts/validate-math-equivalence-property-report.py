#!/usr/bin/env python3
"""Semantic validator for MATH-REP-001D1 JSON reports.

This validates evidence structure/coherence against the frozen D0 manifest.
It can validate both passing and failing reports. `--require-pass` is a separate
promotion gate and must not be used to erase or reject legitimate failure data.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data/benchmarks/math_equivalence_property_q0_v2.json"

EXPECTED_TOP_KEYS = {
    "version",
    "evaluator_id",
    "generator_id",
    "normalizer_id",
    "authority",
    "split",
    "seeds_hex",
    "total_cases",
    "passed_cases",
    "all_passed",
    "report",
}
EXPECTED_REPORT_KEYS = {
    "same_normal_form",
    "different_normal_form",
    "refusal_contract",
    "pair_families",
    "refusal_families",
}
EXPECTED_STATS_KEYS = {"total", "passed", "unexpected_normalization_errors"}


def fail(message: str) -> None:
    raise ValueError(message)


def require_exact_keys(value: dict, expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        fail(f"{label} key mismatch; missing={missing}, extra={extra}")


def validate_stats(stats: dict, label: str) -> None:
    if not isinstance(stats, dict):
        fail(f"{label} must be an object")
    require_exact_keys(stats, EXPECTED_STATS_KEYS, label)
    for key in EXPECTED_STATS_KEYS:
        value = stats[key]
        if type(value) is not int or value < 0:
            fail(f"{label}.{key} must be a non-negative integer")
    if stats["passed"] > stats["total"]:
        fail(f"{label}.passed exceeds total")
    if stats["unexpected_normalization_errors"] > stats["total"]:
        fail(f"{label}.unexpected_normalization_errors exceeds total")


def expected_family_counts(cycle: list[str], cases_per_seed: int, seed_count: int) -> Counter[str]:
    one_seed: Counter[str] = Counter()
    for index in range(cases_per_seed):
        one_seed[cycle[index % len(cycle)]] += 1
    return Counter({family: count * seed_count for family, count in one_seed.items()})


def validate(report_path: Path, manifest_path: Path, require_pass: bool) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not isinstance(report, dict):
        fail("report root must be an object")
    require_exact_keys(report, EXPECTED_TOP_KEYS, "report")

    expected_scalars = {
        "version": "math-equivalence-property-report-v1",
        "evaluator_id": "math-equivalence-property-evaluator-v1",
        "generator_id": manifest["generator_id"],
        "normalizer_id": "symthaea-exact-polynomial-term-v2",
        "authority": "MeasurementOnly",
    }
    for key, expected in expected_scalars.items():
        if report[key] != expected:
            fail(f"{key}={report[key]!r}; expected {expected!r}")

    split = report["split"]
    if split not in {"development", "evaluation"}:
        fail(f"invalid split: {split!r}")
    manifest_seed_key = "dev_seeds_hex" if split == "development" else "evaluation_seeds_hex"
    expected_seeds = manifest[manifest_seed_key]
    if report["seeds_hex"] != expected_seeds:
        fail(f"{split} seeds do not exactly match frozen manifest order")

    expected_total = manifest["cases_per_split"]
    if report["total_cases"] != expected_total:
        fail(f"total_cases={report['total_cases']}; expected {expected_total}")
    if type(report["passed_cases"]) is not int or not 0 <= report["passed_cases"] <= expected_total:
        fail("passed_cases outside valid range")
    if type(report["all_passed"]) is not bool:
        fail("all_passed must be boolean")

    nested = report["report"]
    if not isinstance(nested, dict):
        fail("nested report must be an object")
    require_exact_keys(nested, EXPECTED_REPORT_KEYS, "report.report")

    same = nested["same_normal_form"]
    different = nested["different_normal_form"]
    refusal = nested["refusal_contract"]
    validate_stats(same, "same_normal_form")
    validate_stats(different, "different_normal_form")
    validate_stats(refusal, "refusal_contract")

    pair_cycle = manifest["pair_family_cycle"]
    same_families = set(manifest["same_normal_form_families"])
    different_families = set(manifest["different_normal_form_families"])
    refusal_cycle = manifest["refusal_family_cycle"]
    seed_count = len(expected_seeds)

    pair_counts = expected_family_counts(pair_cycle, manifest["pairs_per_seed"], seed_count)
    refusal_counts = expected_family_counts(
        refusal_cycle, manifest["refusals_per_seed"], seed_count
    )

    expected_same_total = sum(pair_counts[name] for name in same_families)
    expected_different_total = sum(pair_counts[name] for name in different_families)
    expected_refusal_total = sum(refusal_counts.values())
    if same["total"] != expected_same_total:
        fail(f"same_normal_form.total={same['total']}; expected {expected_same_total}")
    if different["total"] != expected_different_total:
        fail(
            f"different_normal_form.total={different['total']}; expected {expected_different_total}"
        )
    if refusal["total"] != expected_refusal_total:
        fail(f"refusal_contract.total={refusal['total']}; expected {expected_refusal_total}")
    if same["total"] + different["total"] + refusal["total"] != expected_total:
        fail("class totals do not sum to total_cases")

    pair_families = nested["pair_families"]
    refusal_families = nested["refusal_families"]
    if not isinstance(pair_families, dict) or set(pair_families) != set(pair_cycle):
        fail("pair_families keys must exactly match frozen pair family cycle")
    if not isinstance(refusal_families, dict) or set(refusal_families) != set(refusal_cycle):
        fail("refusal_families keys must exactly match frozen refusal family cycle")

    for family, stats in pair_families.items():
        validate_stats(stats, f"pair_families.{family}")
        if stats["total"] != pair_counts[family]:
            fail(
                f"pair_families.{family}.total={stats['total']}; expected {pair_counts[family]}"
            )
    for family, stats in refusal_families.items():
        validate_stats(stats, f"refusal_families.{family}")
        if stats["total"] != refusal_counts[family]:
            fail(
                f"refusal_families.{family}.total={stats['total']}; expected {refusal_counts[family]}"
            )
        if stats["unexpected_normalization_errors"] != 0:
            fail(f"refusal_families.{family} cannot carry pair-normalization error counts")

    same_family_passed = sum(pair_families[name]["passed"] for name in same_families)
    different_family_passed = sum(
        pair_families[name]["passed"] for name in different_families
    )
    refusal_family_passed = sum(stats["passed"] for stats in refusal_families.values())
    if same_family_passed != same["passed"]:
        fail("same-normal-form family passed counts do not reconcile with class summary")
    if different_family_passed != different["passed"]:
        fail("different-normal-form family passed counts do not reconcile with class summary")
    if refusal_family_passed != refusal["passed"]:
        fail("refusal family passed counts do not reconcile with class summary")

    pair_family_errors = sum(
        stats["unexpected_normalization_errors"] for stats in pair_families.values()
    )
    if pair_family_errors != (
        same["unexpected_normalization_errors"]
        + different["unexpected_normalization_errors"]
    ):
        fail("pair family normalization-error counts do not reconcile with class summaries")
    if refusal["unexpected_normalization_errors"] != 0:
        fail("refusal_contract cannot carry pair-normalization error counts")

    nested_passed = same["passed"] + different["passed"] + refusal["passed"]
    if report["passed_cases"] != nested_passed:
        fail("top-level passed_cases does not reconcile with class summaries")

    computed_all_passed = nested_passed == expected_total
    if report["all_passed"] != computed_all_passed:
        fail("all_passed is inconsistent with passed_cases / total_cases")

    if require_pass and not report["all_passed"]:
        fail("report is structurally valid but does not satisfy --require-pass")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="add a promotion gate after validating the report; failure reports remain valid evidence without this flag",
    )
    args = parser.parse_args()

    try:
        validate(args.report, args.manifest, args.require_pass)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    mode = "PASS-REQUIRED" if args.require_pass else "STRUCTURALLY-VALID"
    print(f"PASS: MATH-REP-001D1 report is {mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
