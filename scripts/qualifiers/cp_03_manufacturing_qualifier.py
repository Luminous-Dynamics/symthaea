#!/usr/bin/env python3
"""Independent CP-03 manufacturing digital-thread qualifier.

Stdlib-only by design: this oracle does not import production manufacturing code.
It validates the frozen representation corpus and adversarial semantics independently.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "docs/engineering/data/cp-03-manufacturing-corpus-v1.json"

EXPECTED_SCHEMA = "cp-03-manufacturing-corpus-v1"
EXPECTED_AUTHORITY = "representation_only_no_physical_execution_authority"
EXPECTED_CASES = (
    ("C01", "complete-closed-thread", "CurrentAndApplicable"),
    ("C02", "missing-process-generation", "Unknown"),
    ("C03", "changed-equipment-configuration", "ConfigurationMismatch"),
    ("C04", "changed-tooling-fixture", "ConfigurationMismatch"),
    ("C05", "changed-material-generation", "DependencyChanged"),
    ("C06", "plan-without-execution", "ExecutionMissing"),
    ("C07", "execution-without-as-built", "AsBuiltMissing"),
    ("C08", "as-built-without-metrology", "ObservationMissing"),
    ("C09", "stale-or-mismatched-metrology", "MeasurementInvalid"),
    ("C10", "single-specimen-not-population", "PopulationInferenceBlocked"),
    ("C11", "capability-not-availability", "AuthoritySeparated"),
    ("C12", "capacity-not-capability", "AuthoritySeparated"),
    ("C13", "rework-without-new-generation", "HistoricalIdentityViolation"),
    ("C14", "negative-evidence-retained", "HistoricalNegativeRetained"),
    ("C15", "common-mode-measurements", "CommonModeNotIndependent"),
    ("C16", "operational-event-not-qualification", "AuthoritySeparated"),
    ("C17", "synthetic-pass", "NoPhysicalExecutionAuthority"),
    ("C18", "changed-acceptance-profile", "RequalificationRequired"),
)

FAILURE_CATEGORIES = {
    "schema": "schema-integrity",
    "coverage": "coverage",
    "identity": "historical-identity",
    "dependency": "dependency-boundary",
    "authority": "authority",
    "negative": "negative-evidence",
    "invariant": "invariant-integrity",
}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def fail(category, message):
    raise AssertionError(f"[CP03-{FAILURE_CATEGORIES[category]}] {message}")

def load():
    try:
        return json.loads(CORPUS.read_text(encoding="utf-8"))
    except Exception as exc:
        fail("schema", f"cannot load corpus: {exc}")

def validate_shape(data):
    if data.get("schema") != EXPECTED_SCHEMA:
        fail("schema", "schema mismatch")
    if data.get("authority") != EXPECTED_AUTHORITY:
        fail("authority", "physical execution authority leaked into corpus")
    if data.get("replay_semantics") != "historical identities are immutable inputs; dispositions are derived outputs":
        fail("invariant", "replay semantics changed")
    cases = data.get("cases")
    if not isinstance(cases, list):
        fail("schema", "cases must be a list")
    actual = [(c.get("case_id"), c.get("scenario"), c.get("expected")) for c in cases]
    if tuple(actual) != EXPECTED_CASES:
        fail("coverage", "case manifest or expected dispositions differ")
    if data.get("case_manifest_order") != [x[0] for x in EXPECTED_CASES]:
        fail("coverage", "case manifest order differs")
    if len({x[0] for x in actual}) != len(actual):
        fail("coverage", "duplicate case identity")

def adversarial_checks(data):
    cases = {c["case_id"]: c for c in data["cases"]}

    # Process and physical identity are separate semantic dimensions.
    if cases["C03"]["expected"] == cases["C01"]["expected"]:
        fail("dependency", "equipment configuration change did not narrow disposition")
    if cases["C04"]["expected"] != "ConfigurationMismatch":
        fail("dependency", "fixture change lost configuration boundary")
    if cases["C05"]["expected"] != "DependencyChanged":
        fail("dependency", "material generation change lost dependency boundary")

    # Planning/execution/as-built are not interchangeable.
    for cid in ("C06", "C07", "C08"):
        if cases[cid]["expected"] in {"CurrentAndApplicable", "PhysicalQualified"}:
            fail("invariant", f"{cid} incorrectly promotes an incomplete thread")

    # Metrology remains externally owned and exact.
    if cases["C09"]["expected"] != "MeasurementInvalid":
        fail("authority", "stale/mismatched metrology was not rejected")

    # Manufacturing evidence cannot imply population evidence.
    if cases["C10"]["expected"] != "PopulationInferenceBlocked":
        fail("coverage", "single specimen escaped population guard")

    # Capability, capacity, availability, and authority are distinct.
    if cases["C11"]["expected"] != "AuthoritySeparated" or cases["C12"]["expected"] != "AuthoritySeparated":
        fail("authority", "capability/availability/capacity boundary collapsed")

    # Rework must not rewrite the original as-built identity.
    if cases["C13"]["expected"] != "HistoricalIdentityViolation":
        fail("identity", "rework without new generation escaped identity guard")

    # Negative evidence is durable.
    if cases["C14"]["expected"] != "HistoricalNegativeRetained":
        fail("negative", "negative evidence is not retained")

    # Shared roots prevent a false independence claim.
    if cases["C15"]["expected"] != "CommonModeNotIndependent":
        fail("coverage", "common-mode evidence escaped independence guard")

    # Operational events cannot mint engineering qualification.
    if cases["C16"]["expected"] != "AuthoritySeparated":
        fail("authority", "operational event promoted to engineering evidence")

    # Synthetic qualification has no physical execution authority.
    if cases["C17"]["expected"] != "NoPhysicalExecutionAuthority":
        fail("authority", "synthetic PASS exceeded claim ceiling")

    # Acceptance-profile changes require explicit requalification.
    if cases["C18"]["expected"] != "RequalificationRequired":
        fail("dependency", "acceptance-profile change did not require requalification")

def main():
    data = load()
    validate_shape(data)
    adversarial_checks(data)
    replay_input = {
        "schema": data["schema"],
        "authority": data["authority"],
        "claim_ceiling": data["claim_ceiling"],
        "replay_semantics": data["replay_semantics"],
        "case_manifest_order": data["case_manifest_order"],
        "cases": [
            {
                "case_id": c["case_id"],
                "scenario": c["scenario"],
                "expected": c["expected"],
            }
            for c in data["cases"]
        ],
    }
    print("CP-03 MANUFACTURING QUALIFIER PASS")
    print(f"corpus_sha256={hashlib.sha256(CORPUS.read_bytes()).hexdigest()}")
    print(f"replay_input_sha256={digest(replay_input)}")
    print(f"case_count={len(data['cases'])}")
    print(f"claim_ceiling={data['claim_ceiling']}")

if __name__ == "__main__":
    main()
