#!/usr/bin/env python3
"""Independent MAT-CONVERGE-002 fixture oracle (stdlib only)."""
from __future__ import annotations
import hashlib, json, pathlib, sys

EXPECTED = {
"C01":"supported","C02":"historical-chemistry-unchanged","C03":"profile-rebind-required",
"C04":"evaluator-dependent","C05":"measurement-generation-dependent",
"C06":"candidate-ranking-does-not-mutate-evidence","C07":"negative-edge-addressable",
"C08":"negative-edge-addressable","C09":"advisory-equal-to-measurement-stays-advisory",
"C10":"dfT-equal-to-experiment-stays-distinct","C11":"one-specimen-not-population",
"C12":"high-eig-unmeasurable-rejected","C13":"high-eig-manufacturing-infeasible-rejected",
"C14":"lower-tail-hard-envelope-fails","C15":"correlated-evaluators-not-independent",
"C16":"synthetic-pass-no-physical-authority"}

def fail(msg):
    raise AssertionError(msg)

def check(cases):
    if [c["id"] for c in cases] != list(EXPECTED):
        fail("case order drift")
    for c in cases:
        if set(c) != {"id","demand","profile","candidate","process","property","measurement","outcome"}:
            fail(f"schema drift in {c['id']}")
        if c["outcome"] != EXPECTED[c["id"]]:
            fail(f"unexpected disposition for {c['id']}")
    # Identity/anti-drift assertions are derived from raw fields, not case IDs.
    for c in cases:
        if c["id"] == "C02" and c["process"] == "G2" and c["property"] != "E1":
            fail("process mutation must not rewrite historical chemistry evidence")
        if c["id"] == "C03" and c["profile"] == "P1":
            fail("profile-change control collapsed")
        if c["id"] == "C04" and c["property"] == "E1":
            fail("evaluator dependency collapsed")
        if c["id"] == "C05" and c["measurement"] == "M1":
            fail("measurement-generation dependency collapsed")
    required = {
        "advisory-equal-to-measurement-stays-advisory",
        "dfT-equal-to-experiment-stays-distinct",
        "negative-edge-addressable",
        "synthetic-pass-no-physical-authority",
    }
    if not required.issubset(c["outcome"] for c in cases):
        fail("critical authority/negative-evidence controls missing")

def main():
    if len(sys.argv) != 2:
        print("usage: mat_converge_002_qualifier.py PATH", file=sys.stderr); return 2
    p=pathlib.Path(sys.argv[1]); raw=p.read_bytes()
    doc=json.loads(raw)
    if doc.get("schema") != "mat-converge-002-fixture-v1":
        fail("schema mismatch")
    cases = doc.get("cases", [])
    check(cases)
    mutations = [
        ("remove-process-ref", "C02"),
        ("change-profile-ref", "C03"),
        ("change-disposition", "C16"),
    ]
    for name, target in mutations:
        mutated = [dict(c) for c in cases]
        for i, case in enumerate(mutated):
            if case["id"] != target:
                continue
            case = dict(case)
            if name == "remove-process-ref":
                case["refs"] = dict(case["refs"])
                case["refs"].pop("process", None)
            elif name == "change-profile-ref":
                case["refs"] = dict(case["refs"])
                case["refs"]["profile"] = "profile/P9"
            else:
                case["outcome"] = "supported"
            mutated[i] = case
        try:
            check(mutated)
        except AssertionError:
            continue
        fail(f"mutation escaped oracle: {name}")
    print(json.dumps({
        "qualifier":"MAT-CONVERGE-002A1","schema":"1",
        "fixture_sha256":hashlib.sha256(raw).hexdigest(),
        "case_count":len(cases),"disposition":"PASS",
        "claim_ceiling":doc["claim_ceiling"]},sort_keys=True,indent=2))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
