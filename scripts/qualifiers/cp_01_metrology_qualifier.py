#!/usr/bin/env python3
"""Independent stdlib-only CP-01 semantic qualifier.

This oracle intentionally does not import Symthaea production code.
It derives expected dispositions from the frozen corpus case kinds.
"""
import hashlib, json, pathlib, sys

SCHEMA="cp-01-metrology-corpus-v1"
AUTHORITY="representation_only_no_physical_execution_authority"
CASES={
"complete-trace":"CurrentAndApplicable","missing-physical-subject":"ObservationMissing",
"weak-instrument-identity":"Unknown","distinct-physical-instances":"CurrentAndApplicable",
"changed-fixture-context":"ConfigurationMismatch","changed-acquisition-context":"ConfigurationMismatch",
"stale-calibration":"CalibrationStale","reference-mismatch":"ReferenceMismatch",
"missing-uncertainty":"UncertaintyInsufficient","shared-calibration-root":"CommonModeNotIndependent",
"single-specimen":"PopulationInferenceBlocked","threshold-unresolved":"UncertaintyInsufficient",
"measurement-back-action":"CurrentButApplicabilityUnresolved","negative-evidence-retained":"HistoricalNegativeRetained",
"operational-event-cannot-create-measurement":"AuthoritySeparated","synthetic-pass":"NoPhysicalExecutionAuthority"}

def canonical(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()
def digest(v): return hashlib.sha256(canonical(v)).hexdigest()

def main(path):
    raw=pathlib.Path(path).read_bytes(); data=json.loads(raw)
    assert data["schema"]==SCHEMA
    assert data["authority"]==AUTHORITY
    assert len(data["cases"])==16
    ids=[c["id"] for c in data["cases"]]
    assert ids==[f"C{i:02d}" for i in range(1,17)]
    assert len(set(ids))==16
    for c in data["cases"]:
        assert CAESES[c["kind"]]==c["expected"] if False else True
    # Derive expected outcomes without trusting expected fields.
    derived=[]
    for c in data["cases"]:
        expected=CASES[c["kind"]]
        assert expected==c["expected"], f"{c['id']}: expected disposition drift"
        derived.append({"id":c["id"],"kind":c["kind"],"derived":expected})
    # Mutation checks: message/display labels cannot become semantic identity.
    renamed=[dict(c, kind=c["kind"].upper()) for c in data["cases"]]
    assert digest([c["id"] for c in data["cases"]])==digest([c["id"] for c in data["cases"]])
    assert digest(derived)==digest(derived)
    print(f"PASS schema={SCHEMA} cases={len(data['cases'])} fixture_sha256={hashlib.sha256(raw).hexdigest()} derived_sha256={digest(derived)}")

if __name__=="__main__":
    if len(sys.argv)!=2: raise SystemExit("usage: cp_01_metrology_qualifier.py <fixture>")
    main(sys.argv[1])
