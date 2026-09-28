#!/usr/bin/env python3
"""Independent CP-04 synthetic compute-evidence qualifier (stdlib-only)."""
from __future__ import annotations
import hashlib, json
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CORPUS=ROOT/"docs/engineering/data/cp-04-compute-corpus-v1.json"
SCHEMA="cp-04-compute-corpus-v1"
AUTHORITY="representation_only_no_physical_execution_authority"
CEILING="deterministic compute identity, generation, dependency, evidence-separation, replay, negative-result, and authority semantics over synthetic/reference workflows only"
EXPECTED={"C01":"CurrentAndApplicable","C02":"DependencyChanged","C03":"ConfigurationMismatch","C04":"ConfigurationMismatch","C05":"DeploymentMismatch","C06":"ObservationMissing","C07":"UncertaintyInsufficient","C08":"PopulationInferenceBlocked","C09":"HardEnvelopeFailure","C10":"CommonModeNotIndependent","C11":"AuthoritySeparated","C12":"Stale","C13":"StateRestorationUnresolved","C14":"AuthoritySeparated","C15":"AuthoritySeparated","C16":"HistoricalNegativeRetained","C17":"NoPhysicalExecutionAuthority","C18":"RequalificationRequired"}
FAILURES=[]
def canonical(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()
def digest(v): return hashlib.sha256(canonical(v)).hexdigest()
def fail(m): FAILURES.append(m)
def eq(a,b,m):
    if a!=b: fail(f"{m}: expected {b!r}, got {a!r}")
def replay_input(c):
    return {"schema":c["schema"],"authority":c["authority"],"claim_ceiling":c["claim_ceiling"],"replay_semantics":c["replay_semantics"],"case_manifest":[{"case_id":x["case_id"],"scenario":x["scenario"]} for x in c["cases"]]}
def validate(c):
    eq(c.get("schema"),SCHEMA,"schema"); eq(c.get("authority"),AUTHORITY,"authority"); eq(c.get("claim_ceiling"),CEILING,"claim ceiling")
    eq(c.get("replay_semantics"),"historical identities are immutable inputs; dispositions are derived outputs","replay semantics")
    cases=c.get("cases")
    if not isinstance(cases,list): fail("cases must be list"); return
    ids=[x.get("case_id") for x in cases]; eq(ids,list(EXPECTED),"ordered case manifest")
    if len(set(ids))!=len(ids): fail("duplicate case IDs")
    for x in cases:
        cid=x.get("case_id")
        if not isinstance(x.get("scenario"),str): fail(f"{cid}: scenario missing")
        eq(x.get("expected_disposition"),EXPECTED.get(cid,"Unknown"),f"{cid} disposition")
    guards={"model-output-equals-measurement":"AuthoritySeparated","attestation-not-performance":"AuthoritySeparated","synthetic-pass":"NoPhysicalExecutionAuthority","negative-result-retained":"HistoricalNegativeRetained","lossy-cfc-snapshot":"StateRestorationUnresolved"}
    for x in cases:
        if x["scenario"] in guards: eq(x["expected_disposition"],guards[x["scenario"]],f"{x['case_id']} semantic guard")
def mutations(c):
    specs=[
      ("drop-C06",lambda x:x["cases"].pop(5),"coverage"),
      ("promote-C17",lambda x:x["cases"][16].update(expected_disposition="CurrentAndApplicable"),"authority"),
      ("rewrite-C16",lambda x:x["cases"][15].update(scenario="no-result"),"negative-evidence"),
      ("collapse-C10",lambda x:x["cases"][9].update(expected_disposition="CurrentAndApplicable"),"dependency-boundary"),
      ("change-C18",lambda x:x["cases"][17].update(expected_disposition="CurrentAndApplicable"),"currentness"),
      ("erase-C05",lambda x:x["cases"][4].update(scenario="complete-thread"),"historical-identity")]
    out=[]
    for label,mut,cat in specs:
        y=deepcopy(c); old=FAILURES[:]; FAILURES.clear(); mut(y); validate(y); caught=bool(FAILURES); detail=FAILURES[:]; FAILURES.clear(); FAILURES.extend(old)
        if not caught: fail(f"mutation {label} was not rejected")
        out.append({"mutation_id":label,"expected_failure_category":cat,"rejected":caught,"diagnostics":detail})
    return out
def receipt(c,m,r):
    p={"schema":"cp-04-compute-qualification-receipt-v1","corpus_sha256":digest(c),"replay_input_sha256":r,"mutation_results":m,"claim_ceiling":c["claim_ceiling"],"disposition":"PASS","physical_execution_authority":False}
    p["receipt_sha256"]=digest(p); return p
def main():
    c=json.loads(CORPUS.read_text(encoding="utf-8")); validate(c); r=digest(replay_input(c)); m=mutations(c)
    if FAILURES or not all(x["rejected"] for x in m):
        print("CP-04 COMPUTE QUALIFIER FAIL"); [print(" -",x) for x in FAILURES]; raise SystemExit(1)
    q=receipt(c,m,r); print("CP-04 COMPUTE QUALIFIER PASS"); print("corpus_sha256="+q["corpus_sha256"]); print("replay_input_sha256="+q["replay_input_sha256"]); print("receipt_sha256="+q["receipt_sha256"]); print("claim_ceiling="+q["claim_ceiling"]); print("physical_execution_authority=False")
if __name__=="__main__": main()
