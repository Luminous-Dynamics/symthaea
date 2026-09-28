#!/usr/bin/env python3
"""Independent MAT-CONVERGE-002 fixture oracle (stdlib only).

The oracle derives edge identities and a replay digest from immutable references.
It never treats a derived disposition as an evidence identity.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

EXPECTED = {
    "C01": "supported",
    "C02": "historical-chemistry-unchanged",
    "C03": "profile-rebind-required",
    "C04": "evaluator-dependent",
    "C05": "measurement-generation-dependent",
    "C06": "candidate-ranking-does-not-mutate-evidence",
    "C07": "negative-edge-addressable",
    "C08": "negative-edge-addressable",
    "C09": "advisory-equal-to-measurement-stays-advisory",
    "C10": "dft-equal-to-experiment-stays-distinct",
    "C11": "one-specimen-not-population",
    "C12": "high-eig-unmeasurable-rejected",
    "C13": "high-eig-manufacturing-infeasible-rejected",
    "C14": "lower-tail-hard-envelope-fails",
    "C15": "correlated-evaluators-not-independent",
    "C16": "synthetic-pass-no-physical-authority",
}

FIELDS = (
    "demand",
    "profile",
    "candidate",
    "process",
    "property",
    "measurement",
)

def fail(msg):
    raise AssertionError(msg)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()

def edge_identity(case):
    refs = case["refs"]
    return "|".join(refs[field] for field in FIELDS)

def edge_hash(case):
    return hashlib.sha256(canonical({
        "schema": "integration-edge-v1",
        "edge": edge_identity(case),
    })).hexdigest()

def replay_digest(cases):
    """Hash only immutable identity inputs, never mutable dispositions."""
    edges = [{
        "id": c["id"],
        "edge_hash": edge_hash(c),
        "refs": {field: c["refs"][field] for field in FIELDS},
    } for c in cases]
    return hashlib.sha256(canonical(edges)).hexdigest()

def check(doc, cases):
    if doc.get("campaign_id") != "MAT-CONVERGE-002A1":
        fail("campaign identity missing")
    if doc.get("record_schema") != "integration-edge-v1":
        fail("record schema mismatch")
    if doc.get("replay_semantics") != (
        "historical refs are immutable inputs; dispositions are derived outputs"
    ):
        fail("replay semantics missing")
    if [c["id"] for c in cases] != list(EXPECTED):
        fail("case order drift")

    required_keys = {
        "id", "demand", "profile", "candidate", "process",
        "property", "measurement", "outcome", "refs",
    }
    for c in cases:
        if set(c) != required_keys:
            fail(f"schema drift in {c['id']}")
        if c["outcome"] != EXPECTED[c["id"]]:
            fail(f"unexpected disposition for {c['id']}")
        if set(c["refs"]) != set(FIELDS):
            fail(f"reference set drift in {c['id']}")
        for field in FIELDS:
            expected_ref = f"{field}/{c[field]}"
            if c["refs"][field] != expected_ref:
                fail(f"non-derived reference in {c['id']}:{field}")

    # Explicit anti-drift controls. These compare immutable identity fields,
    # not outcome labels, so a copied PASS cannot mask a changed edge.
    by_id = {c["id"]: c for c in cases}
    if by_id["C02"]["process"] != "G2" or by_id["C02"]["property"] != "E1":
        fail("process generation must not rewrite historical chemistry evidence")
    if by_id["C03"]["profile"] == "P1":
        fail("profile change collapsed")
    if by_id["C04"]["property"] != "E2":
        fail("evaluator dependency collapsed")
    if by_id["C05"]["measurement"] != "M2":
        fail("measurement-generation dependency collapsed")
    if by_id["C07"]["outcome"] != "negative-edge-addressable" or        by_id["C08"]["outcome"] != "negative-edge-addressable":
        fail("negative evidence lost")
    if by_id["C09"]["outcome"] != "advisory-equal-to-measurement-stays-advisory":
        fail("authority boundary lost")
    if by_id["C10"]["outcome"] != "dft-equal-to-experiment-stays-distinct":
        fail("independent evidence identity collapsed")
    if by_id["C16"]["outcome"] != "synthetic-pass-no-physical-authority":
        fail("synthetic PASS acquired physical authority")

    required = {
        "advisory-equal-to-measurement-stays-advisory",
        "dft-equal-to-experiment-stays-distinct",
        "negative-edge-addressable",
        "synthetic-pass-no-physical-authority",
    }
    outcomes = {c["outcome"] for c in cases}
    if not required.issubset(outcomes):
        fail("critical authority/negative-evidence controls missing")

def expect_failure(doc, mutation, label):
    try:
        check(doc, mutation)
    except AssertionError:
        return
    fail(f"mutation escaped oracle: {label}")

def main():
    if len(sys.argv) != 2:
        print("usage: mat_converge_002_qualifier.py PATH", file=sys.stderr)
        return 2

    p = pathlib.Path(sys.argv[1])
    raw = p.read_bytes()
    doc = json.loads(raw)
    if doc.get("schema") != "mat-converge-002-fixture-v1":
        fail("schema mismatch")
    cases = doc.get("cases", [])
    check(doc, cases)

    baseline_digest = replay_digest(cases)
    mutations = [
        ("remove-process-ref", "C02"),
        ("change-process-generation", "C02"),
        ("change-profile-ref", "C03"),
        ("change-evaluator-generation", "C04"),
        ("change-measurement-generation", "C05"),
        ("change-ranking-only", "C06"),
        ("delete-negative-edge", "C07"),
        ("promote-authority", "C09"),
        ("change-disposition", "C16"),
        ("rewrite-historical-ref", "C02"),
    ]

    for name, target in mutations:
        mutated = json.loads(json.dumps(cases))
        case = next(c for c in mutated if c["id"] == target)
        if name == "remove-process-ref":
            case["refs"].pop("process", None)
        elif name == "change-process-generation":
            case["process"] = "G3"
        elif name == "change-profile-ref":
            case["refs"]["profile"] = "profile/P9"
        elif name == "change-evaluator-generation":
            case["property"] = "E9"
        elif name == "change-measurement-generation":
            case["measurement"] = "M9"
        elif name == "change-ranking-only":
            case["ranking"] = "promoted"
        elif name == "delete-negative-edge":
            mutated = [c for c in mutated if c["id"] != target]
        elif name == "promote-authority":
            case["outcome"] = "physical-authority"
        elif name == "change-disposition":
            case["outcome"] = "supported"
        elif name == "rewrite-historical-ref":
            case["property"] = "E9"
            case["refs"]["property"] = "property/E9"
        expect_failure(doc, mutated, name)

    # A disposition-only change must alter qualification but not replay identity.
    disposition_mutated = json.loads(json.dumps(cases))
    disposition_mutated[0]["outcome"] = "recomputed-disposition"
    if replay_digest(disposition_mutated) != baseline_digest:
        fail("derived disposition mutated replay identity")

    print(json.dumps({
        "qualifier": "MAT-CONVERGE-002A2",
        "schema": "1",
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "replay_digest": baseline_digest,
        "case_count": len(cases),
        "mutation_count": len(mutations),
        "disposition": "PASS",
        "claim_ceiling": doc["claim_ceiling"],
    }, sort_keys=True, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
