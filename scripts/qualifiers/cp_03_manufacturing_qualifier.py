#!/usr/bin/env python3
"""Independent CP-03 manufacturing digital-thread qualifier."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "docs/engineering/data/cp-03-manufacturing-corpus-v1.json"
EXPECTED_SCHEMA = "cp-03-manufacturing-corpus-v1"
EXPECTED_AUTHORITY = "representation_only_no_physical_execution_authority"
EXPECTED_CLAIM_CEILING = "deterministic manufacturing identity, generation, dependency, metrology-reference, negative-evidence, and authority semantics over synthetic/reference workflows only"
EXPECTED_REPLAY = "historical identities are immutable inputs; dispositions are derived outputs"
EXPECTED_CASES = (
("C01","complete-closed-thread","CurrentAndApplicable"),("C02","missing-process-generation","Unknown"),
("C03","changed-equipment-configuration","ConfigurationMismatch"),("C04","changed-tooling-fixture","ConfigurationMismatch"),
("C05","changed-material-generation","DependencyChanged"),("C06","plan-without-execution","ExecutionMissing"),
("C07","execution-without-as-built","AsBuiltMissing"),("C08","as-built-without-metrology","ObservationMissing"),
("C09","stale-or-mismatched-metrology","MeasurementInvalid"),("C10","single-specimen-not-population","PopulationInferenceBlocked"),
("C11","capability-not-availability","AuthoritySeparated"),("C12","capacity-not-capability","AuthoritySeparated"),
("C13","rework-without-new-generation","HistoricalIdentityViolation"),("C14","negative-evidence-retained","HistoricalNegativeRetained"),
("C15","common-mode-measurements","CommonModeNotIndependent"),("C16","operational-event-not-qualification","AuthoritySeparated"),
("C17","synthetic-pass","NoPhysicalExecutionAuthority"),("C18","changed-acceptance-profile","RequalificationRequired"),
)
CATEGORIES={"schema":"schema-integrity","coverage":"coverage","identity":"historical-identity","dependency":"dependency-boundary","authority":"authority","negative":"negative-evidence","invariant":"invariant-integrity"}

def canonical(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def digest(v): return hashlib.sha256(canonical(v)).hexdigest()
def fail(cat,msg): raise AssertionError(f"[CP03-{CATEGORIES[cat]}] {msg}")
def load():
    try: return json.loads(CORPUS.read_text(encoding="utf-8"))
    except Exception as exc: fail("schema",f"cannot load corpus: {exc}")

def validate_shape(d):
    if d.get("schema")!=EXPECTED_SCHEMA: fail("schema","schema mismatch")
    if d.get("authority")!=EXPECTED_AUTHORITY: fail("authority","physical execution authority leaked")
    if d.get("claim_ceiling")!=EXPECTED_CLAIM_CEILING: fail("invariant","claim ceiling changed")
    if d.get("replay_semantics")!=EXPECTED_REPLAY: fail("invariant","replay semantics changed")
    cases=d.get("cases")
    if not isinstance(cases,list): fail("schema","cases must be a list")
    actual=tuple((c.get("case_id"),c.get("scenario"),c.get("expected")) for c in cases)
    if actual!=EXPECTED_CASES: fail("coverage","case manifest or dispositions differ")
    if d.get("case_manifest_order")!=[x[0] for x in EXPECTED_CASES]: fail("coverage","manifest order differs")
    if len({x[0] for x in actual})!=len(actual): fail("coverage","duplicate case identity")

def adversarial_checks(d):
    c={x["case_id"]:x for x in d["cases"]}
    if c["C03"]["expected"]==c["C01"]["expected"]: fail("dependency","equipment change not narrowed")
    if c["C04"]["expected"]!="ConfigurationMismatch": fail("dependency","fixture boundary lost")
    if c["C05"]["expected"]!="DependencyChanged": fail("dependency","material dependency boundary lost")
    for cid in ("C06","C07","C08"):
        if c[cid]["expected"] in {"CurrentAndApplicable","PhysicalQualified"}: fail("invariant",f"{cid} promoted incomplete thread")
    if c["C09"]["expected"]!="MeasurementInvalid": fail("authority","invalid metrology accepted")
    if c["C10"]["expected"]!="PopulationInferenceBlocked": fail("coverage","population guard escaped")
    if c["C11"]["expected"]!="AuthoritySeparated" or c["C12"]["expected"]!="AuthoritySeparated": fail("authority","capability/capacity boundary collapsed")
    if c["C13"]["expected"]!="HistoricalIdentityViolation": fail("identity","rework rewrote history")
    if c["C14"]["expected"]!="HistoricalNegativeRetained": fail("negative","negative evidence not retained")
    if c["C15"]["expected"]!="CommonModeNotIndependent": fail("coverage","common-mode evidence escaped")
    if c["C16"]["expected"]!="AuthoritySeparated": fail("authority","operational event minted engineering evidence")
    if c["C17"]["expected"]!="NoPhysicalExecutionAuthority": fail("authority","synthetic PASS exceeded ceiling")
    if c["C18"]["expected"]!="RequalificationRequired": fail("dependency","acceptance change escaped requalification")

def replay_input(d):
    return {"schema":d["schema"],"authority":d["authority"],"claim_ceiling":d["claim_ceiling"],"replay_semantics":d["replay_semantics"],"case_manifest_order":d["case_manifest_order"],"cases":[{"case_id":c["case_id"],"scenario":c["scenario"],"expected":c["expected"]} for c in d["cases"]]}

def mutation_suite(d):
    results=[]
    def mutate(label,fn,category):
        x=json.loads(json.dumps(d)); fn(x)
        try: validate_shape(x); adversarial_checks(x)
        except AssertionError: results.append({"mutation":label,"guard_category":category,"caught":True})
        else: fail("invariant",f"mutation escaped detection: {label}")
    mutate("drop-C08",lambda x:x["cases"].pop(7),"coverage")
    mutate("promote-C17",lambda x:x["cases"][16].update(expected="CurrentAndApplicable"),"authority")
    mutate("rewrite-C13",lambda x:x["cases"][12].update(expected="CurrentAndApplicable"),"historical-identity")
    mutate("promote-C14",lambda x:x["cases"][13].update(expected="CurrentAndApplicable"),"negative-evidence")
    mutate("collapse-C15",lambda x:x["cases"][14].update(expected="CurrentAndApplicable"),"coverage")
    mutate("change-C18",lambda x:x["cases"][17].update(expected="CurrentAndApplicable"),"dependency-boundary")
    return results

def receipt(d,replay_sha,mutations):
    p={"schema":"cp-03-manufacturing-qualification-receipt-v1","corpus_sha256":hashlib.sha256(CORPUS.read_bytes()).hexdigest(),"replay_input_sha256":replay_sha,"mutation_results":mutations,"claim_ceiling":d["claim_ceiling"],"disposition":"PASS","physical_execution_authority":False}
    p["receipt_sha256"]=digest(p); return p

def main():
    d=load(); validate_shape(d); adversarial_checks(d)
    replay_sha=digest(replay_input(d)); mutations=mutation_suite(d)
    if len(mutations)!=6 or not all(m["caught"] for m in mutations): fail("invariant","mutation suite incomplete")
    r=receipt(d,replay_sha,mutations)
    print("CP-03 MANUFACTURING QUALIFIER PASS")
    print(f"corpus_sha256={r['corpus_sha256']}")
    print(f"replay_input_sha256={r['replay_input_sha256']}")
    print(f"mutation_count={len(mutations)}")
    print(f"receipt_schema={r['schema']}")
    print(f"receipt_sha256={r['receipt_sha256']}")
    print(f"claim_ceiling={r['claim_ceiling']}")

if __name__=="__main__": main()
