#!/usr/bin/env python3
"""Fail-closed verifier for SPINE-000B-P1R qualification coverage.

Measurement-only. This verifier does not execute cognition and cannot establish
runtime influence or causal load. It proves that the exact-head qualification
surface contains the preregistered fixture classes and that the strict Rust
harness exercises the real crate-private OutputCollector without widening the
production API.
"""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES = Path("tests/fixtures/spine_000b_golden_fixtures.json")
PRODUCTION = Path("src/cognitive_loop/subsystem_trait.rs")
HARNESS = Path("scripts/run_spine_000b_p1r_rust_harness.py")
WORKFLOW = Path(".github/workflows/spine-000b-p1r.yml")
GENERATOR = Path("scripts/generate_spine_golden_fixtures.py")
ORACLE = Path("scripts/spine_000b_influence_oracle.py")

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
    for path in (FIXTURES, PRODUCTION, HARNESS, WORKFLOW, GENERATOR, ORACLE):
        if not path.is_file():
            fail(f"missing subject file: {path}")

    fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))
    if not isinstance(fixtures, list) or not fixtures:
        fail("fixtures must be a non-empty JSON array")
    if len(fixtures) != 16:
        fail(f"expected exact 16-case preregistered matrix, got {len(fixtures)}")

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
        proposal_names = [p.get("subsystem_name") for p in proposals if isinstance(p, dict)]
        if len(proposal_names) != len(proposals) or any(not isinstance(n, str) or not n for n in proposal_names):
            fail(f"{name}: every proposal needs a non-empty subsystem_name")
        if len(set(proposal_names)) != len(proposal_names):
            fail(f"{name}: duplicate subsystem identity")
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
        if report.get("admitted_proposal_count") != len(proposals):
            fail(f"{name}: admitted_proposal_count mismatch")

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
        receipt_names = [r.get("subsystem_name") for r in receipts if isinstance(r, dict)]
        if receipt_names != sorted(proposal_names):
            fail(f"{name}: receipts must be sorted by subsystem identity")
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
    extra = sorted(seen_tags - REQUIRED_COVERAGE)
    if missing:
        fail("missing preregistered fixture coverage: " + ", ".join(missing))
    if extra:
        fail("unexpected unpreregistered coverage tags: " + ", ".join(extra))
    if not saw_n:
        fail("no deterministic N>=4 contributor fixture found")

    production = PRODUCTION.read_text(encoding="utf-8")
    harness = HARNESS.read_text(encoding="utf-8")
    workflow = WORKFLOW.read_text(encoding="utf-8")

    # Qualification must exercise the real production collector. The harness
    # injects a test into the crate-private module; no public API widening is allowed.
    if "pub(crate) mod subsystem_trait;" not in Path("src/cognitive_loop/mod.rs").read_text(encoding="utf-8"):
        fail("subsystem_trait visibility changed; P1R must not widen production API")
    if "pub struct OutputCollector" not in production or "pub fn integrate(&self) -> IntegratedOutput" not in production:
        fail("production OutputCollector integration surface not found")
    for phrase in (
        "test_spine_000b_p1r_strict_generated_harness",
        "build_collector(proposals, None)",
        'expected_receipt["integrated_without_subject"]',
        "integrated_without.confidence_delta.to_bits()",
        "integrated_without.lr_modulation.to_bits()",
        "integrated_without.exploration_delta.to_bits()",
        "integrated_without.arousal_delta.to_bits()",
        "integrated_without.valence_delta.to_bits()",
        "integrated_without.flags",
        "integrated_without.n_contributors",
        "SUBJECT.write_bytes(original)",
        "restored != original",
    ):
        if phrase not in harness:
            fail(f"strict Rust harness missing required surface: {phrase}")

    # The independent Python fault domain is mandatory at the workflow level,
    # rather than being spawned best-effort from the Rust unit test.
    if "python3 scripts/spine_000b_influence_oracle.py --self-test" not in workflow:
        fail("exact-head workflow does not require independent Python oracle")
    if "python3 scripts/run_spine_000b_p1r_rust_harness.py" not in workflow:
        fail("exact-head workflow does not run strict production Rust harness")
    if "git diff --exit-code -- src/cognitive_loop/subsystem_trait.rs" not in workflow:
        fail("workflow does not prove production source restoration after harness")
    if "python3 scripts/generate_spine_golden_fixtures.py" not in workflow:
        fail("workflow does not regenerate independent golden fixtures")

    print("SPINE-000B-P1R contract verifier: PASS")
    print(f"fixtures={len(fixtures)} coverage_tags={len(seen_tags)}")
    print("rust_surface=production_OutputCollector_via_test_only_injection")
    print("authority=measurement-only")
    print("runtime_evidence_claimed=false")
    print("causal_load_claimed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
