#!/usr/bin/env python3
"""Fail-closed verifier for SPINE-000B-P1R qualification coverage.

This verifier is measurement-only. It does not execute cognition and it does not
establish causal load. Its job is to prevent a partial golden-equivalence suite
from being mislabeled as satisfying Issue #3036.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path("tests/fixtures/spine_000b_golden_fixtures.json")
RUST_SUBJECT = Path("src/cognitive_loop/subsystem_trait.rs")

REQUIRED_COVERAGE = {
    "zero.empty",
    "one.confidence_only",
    "one.lr_only",
    "one.exploration_only",
    "one.arousal_only",
    "one.valence_only",
    "one.flag_only",
    "one.mixed_scalar_flag",
    "two.identical_scalar",
    "two.unequal_scalar",
    "two.shared_flag",
    "two.unique_flag",
    "two.mixed_shared_unique_flags",
    "two.lr_geometric_mean",
    "two.mixed_f64_f32",
    "n.mixed_all_channels",
}

BIT_FIELDS = (
    "confidence_delta_bits",
    "lr_modulation_bits",
    "exploration_delta_bits",
    "arousal_delta_bits",
    "valence_delta_bits",
)


def fail(message: str) -> None:
    raise SystemExit(f"SPINE-000B-P1R CONTRACT FAIL: {message}")


def main() -> int:
    if not FIXTURES.is_file():
        fail(f"missing fixture file: {FIXTURES}")
    if not RUST_SUBJECT.is_file():
        fail(f"missing Rust subject: {RUST_SUBJECT}")

    fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))
    if not isinstance(fixtures, list) or not fixtures:
        fail("fixtures must be a non-empty JSON array")

    seen_names: set[str] = set()
    seen_tags: set[str] = set()
    saw_n = False

    for case in fixtures:
        if not isinstance(case, dict):
            fail("every fixture must be an object")
        name = case.get("name")
        if not isinstance(name, str) or not name:
            fail("every fixture needs a non-empty name")
        if name in seen_names:
            fail(f"duplicate fixture name: {name}")
        seen_names.add(name)

        tags = case.get("coverage_tags")
        if not isinstance(tags, list) or not tags or not all(isinstance(t, str) and t for t in tags):
            fail(f"{name}: missing explicit coverage_tags")
        seen_tags.update(tags)

        proposals = case.get("input_proposals")
        if not isinstance(proposals, list):
            fail(f"{name}: input_proposals must be an array")
        if len(proposals) >= 4:
            saw_n = True

        report = case.get("expected_report")
        if not isinstance(report, dict):
            fail(f"{name}: missing expected_report")
        if report.get("authority") != "measurement-only":
            fail(f"{name}: authority must be measurement-only")
        if report.get("runtime_evidence_claimed") is not False:
            fail(f"{name}: runtime_evidence_claimed must be false")
        if report.get("causal_load_claimed") is not False:
            fail(f"{name}: causal_load_claimed must be false")

        integrated = report.get("integrated_all")
        if not isinstance(integrated, dict):
            fail(f"{name}: missing integrated_all")
        for field in (*BIT_FIELDS, "flags", "n_contributors"):
            if field not in integrated:
                fail(f"{name}: integrated_all missing {field}")

        receipts = report.get("receipts")
        if not isinstance(receipts, list):
            fail(f"{name}: receipts must be an array")
        if len(receipts) != len(proposals):
            fail(f"{name}: receipt count does not equal admitted proposal count")
        for receipt in receipts:
            if not isinstance(receipt, dict):
                fail(f"{name}: receipt must be an object")
            without = receipt.get("integrated_without_subject")
            if not isinstance(without, dict):
                fail(f"{name}: receipt missing integrated_without_subject")
            for field in (*BIT_FIELDS, "flags", "n_contributors"):
                if field not in without:
                    fail(f"{name}: integrated_without_subject missing {field}")
            for field in ("changed_channels", "uniquely_contributed_flags", "integration_changed"):
                if field not in receipt:
                    fail(f"{name}: receipt missing {field}")

    missing = sorted(REQUIRED_COVERAGE - seen_tags)
    if missing:
        fail("missing preregistered fixture coverage: " + ", ".join(missing))
    if not saw_n:
        fail("no deterministic N>=4 contributor fixture found")

    rust = RUST_SUBJECT.read_text(encoding="utf-8")

    # P1R must compare the full leave-one-out canonical object, not only the
    # derived changed-channel classification.
    if 'exp_receipt["integrated_without_subject"]' not in rust:
        fail("Rust P1R test does not read expected integrated_without_subject")
    for field, expr in (
        ("confidence_delta_bits", "integrated_without.confidence_delta.to_bits()"),
        ("lr_modulation_bits", "integrated_without.lr_modulation.to_bits()"),
        ("exploration_delta_bits", "integrated_without.exploration_delta.to_bits()"),
        ("arousal_delta_bits", "integrated_without.arousal_delta.to_bits()"),
        ("valence_delta_bits", "integrated_without.valence_delta.to_bits()"),
        ("flags", "integrated_without.flags"),
        ("n_contributors", "integrated_without.n_contributors"),
    ):
        if expr not in rust:
            fail(f"Rust P1R test lacks exact I_withoutS check for {field}")

    # Python is an independent fault domain and therefore must be required,
    # not silently skipped when unavailable.
    if 'if let Ok(status) = std::process::Command::new("python3")' in rust:
        fail("Python oracle execution is still best-effort")
    if 'Command::new("python3")' not in rust:
        fail("Rust P1R test does not invoke the independent Python oracle")

    print("SPINE-000B-P1R contract verifier: PASS")
    print(f"fixtures={len(fixtures)} coverage_tags={len(seen_tags)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
