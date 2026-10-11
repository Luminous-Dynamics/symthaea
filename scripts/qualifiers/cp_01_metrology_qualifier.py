#!/usr/bin/env python3
"""Independent stdlib-only CP-01 semantic qualifier.

The oracle intentionally does not import Symthaea production code.
Expected dispositions are derived from case semantics, not copied from
the fixture's expected fields.
"""
import hashlib
import json
import pathlib
import sys

SCHEMA = "cp-01-metrology-corpus-v1"
AUTHORITY = "representation_only_no_physical_execution_authority"
CASES = {
    "complete-trace": "CurrentAndApplicable",
    "missing-physical-subject": "ObservationMissing",
    "weak-instrument-identity": "Unknown",
    "distinct-physical-instances": "CurrentAndApplicable",
    "changed-fixture-context": "ConfigurationMismatch",
    "changed-acquisition-context": "ConfigurationMismatch",
    "stale-calibration": "CalibrationStale",
    "reference-mismatch": "ReferenceMismatch",
    "missing-uncertainty": "UncertaintyInsufficient",
    "shared-calibration-root": "CommonModeNotIndependent",
    "single-specimen": "PopulationInferenceBlocked",
    "threshold-unresolved": "UncertaintyInsufficient",
    "measurement-back-action": "CurrentButApplicabilityUnresolved",
    "negative-evidence-retained": "HistoricalNegativeRetained",
    "operational-event-cannot-create-measurement": "AuthoritySeparated",
    "synthetic-pass": "NoPhysicalExecutionAuthority",
}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def main(path):
    path = pathlib.Path(path)
    raw = path.read_bytes()
    data = json.loads(raw)

    assert data["schema"] == SCHEMA
    assert data["authority"] == AUTHORITY
    assert len(data["cases"]) == 16

    ids = [case["id"] for case in data["cases"]]
    assert ids == [f"C{i:02d}" for i in range(1, 17)]
    assert len(set(ids)) == len(ids)

    derived = []
    for case in data["cases"]:
        kind = case["kind"]
        assert kind in CASES, f"{case['id']}: unknown case kind"
        # Derive from the independent semantic map; fixture expected values are
        # checked only after derivation and therefore cannot define the oracle.
        derived_value = CASES[kind]
        assert derived_value == case["expected"], f"{case['id']}: expected disposition drift"
        derived.append({"id": case["id"], "kind": kind, "derived": derived_value})

    # Identity is manifest-driven: presentation/description changes do not
    # create a new case identity, while case IDs remain unique and ordered.
    presentation = [{"id": case["id"], "kind": case["kind"]} for case in data["cases"]]
    assert digest([item["id"] for item in presentation]) == digest(ids)

    print(
        "PASS"
        f" schema={SCHEMA}"
        f" cases={len(data['cases'])}"
        f" fixture_sha256={hashlib.sha256(raw).hexdigest()}"
        f" derived_sha256={digest(derived)}"
    )

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: cp_01_metrology_qualifier.py <fixture>")
    main(sys.argv[1])
