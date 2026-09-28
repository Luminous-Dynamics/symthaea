#!/usr/bin/env python3
"""Independent CP-04 synthetic compute-evidence qualifier (stdlib-only)."""
from __future__ import annotations
import ast, hashlib, json
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
CORPUS=ROOT/"docs/engineering/data/cp-04-compute-corpus-v1.json"
MANIFEST=ROOT/"docs/engineering/data/cp-04-compute-mutation-manifest-v1.json"
SOURCE=Path(__file__).resolve()
SCHEMA="cp-04-compute-corpus-v1"
AUTHORITY="representation_only_no_physical_execution_authority"
CEILING="deterministic compute identity, generation, dependency, evidence-separation, replay, negative-result, and authority semantics over synthetic/reference workflows only"
EXPECTED={"C01":"CurrentAndApplicable","C02":"DependencyChanged","C03":"ConfigurationMismatch","C04":"ConfigurationMismatch","C05":"DeploymentMismatch","C06":"ObservationMissing","C07":"UncertaintyInsufficient","C08":"PopulationInferenceBlocked","C09":"HardEnvelopeFailure","C10":"CommonModeNotIndependent","C11":"AuthoritySeparated","C12":"Stale","C13":"StateRestorationUnresolved","C14":"AuthoritySeparated","C15":"AuthoritySeparated","C16":"HistoricalNegativeRetained","C17":"NoPhysicalExecutionAuthority","C18":"RequalificationRequired"}
GUARDS={"CP-COMP-COVERAGE":"coverage","CP-COMP-AUTHORITY":"authority","CP-COMP-NEGATIVE":"negative-evidence","CP-COMP-DEPENDENCY":"dependency-boundary","CP-COMP-CURRENTNESS":"currentness","CP-COMP-HISTORICAL":"historical-identity"}
FAILURES=[]
def canonical(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()
def digest(v): return hashlib.sha256(canonical(v)).hexdigest()
def fail(m): FAILURES.append(m)
def eq(a,b,m):
    if a!=b: fail(f"{m}: expected {b!r}, got {a!r}")
def replay_input(c):
    return {"schema":c["schema"],"authority":c["authority"],"claim_ceiling":c["claim_ceiling"],"replay_semantics":c["replay_semantics"],"case_manifest":[{"case_id":x["case_id"],"scenario":x["scenario"]} for x in c["cases"]]}
def source_audit():
    source=SOURCE.read_text(encoding="utf-8")
    tree=ast.parse(source)
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="fail" and len(n.args)!=1: fail("source audit: fail() signature changed")
    forbidden={"symthaea","torch","numpy","pandas","onnx","tensorflow"}
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)) and any(a.name.split(".")[0] in forbidden for a in n.names): fail("source audit: forbidden production/runtime import")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()
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
def validate_manifest(m):
    eq(m.get("schema"),"cp-04-compute-mutation-manifest-v1","manifest schema")
    eq(m.get("guard_registry"),GUARDS,"guard registry")
    expected=["drop-C06","promote-C17","rewrite-C16","collapse-C10","change-C18","erase-C05"]
    eq([x.get("mutation_id") for x in m.get("mutations",[])],expected,"mutation manifest")
    for x in m.get("mutations",[]):
        if x.get("guard_id") not in GUARDS: fail(f"unknown guard: {x.get('guard_id')}")
def mutations(c):
    specs=[("drop-C06",lambda x:x["cases"].pop(5),"CP-COMP-COVERAGE"),("promote-C17",lambda x:x["cases"][16].update(expected_disposition="CurrentAndApplicable"),"CP-COMP-AUTHORITY"),("rewrite-C16",lambda x:x["cases"][15].update(scenario="no-result"),"CP-COMP-NEGATIVE"),("collapse-C10",lambda x:x["cases"][9].update(expected_disposition="CurrentAndApplicable"),"CP-COMP-DEPENDENCY"),("change-C18",lambda x:x["cases"][17].update(expected_disposition="CurrentAndApplicable"),"CP-COMP-CURRENTNESS"),("erase-C05",lambda x:x["cases"][4].update(scenario="complete-thread"),"CP-COMP-HISTORICAL")]
    out=[]
    for label,mut,guard in specs:
        y=deepcopy(c); old=FAILURES[:]; FAILURES.clear(); mut(y); validate(y); caught=bool(FAILURES); detail=FAILURES[:]; FAILURES.clear(); FAILURES.extend(old)
        if not caught: fail(f"mutation {label} was not rejected by {guard}")
        out.append({"mutation_id":label,"guard_id":guard,"rejected":caught,"diagnostics":detail})
    return out
def receipt(c,m,r,manifest_sha,source_sha):
    p={"schema":"cp-04-compute-qualification-receipt-v1","corpus_sha256":digest(c),"replay_input_sha256":r,"mutation_manifest_sha256":manifest_sha,"oracle_source_sha256":source_sha,"guard_registry_sha256":digest(GUARDS),"mutation_results":m,"claim_ceiling":c["claim_ceiling"],"disposition":"PASS","physical_execution_authority":False}
    p["receipt_sha256"]=digest(p); return p
def main():
    c=json.loads(CORPUS.read_text(encoding="utf-8")); mfest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_sha=source_audit(); validate(c); validate_manifest(mfest); r=digest(replay_input(c)); muts=mutations(c)
    if FAILURES or not all(x["rejected"] for x in muts):
        print("CP-04 COMPUTE QUALIFIER FAIL"); [print(" -",x) for x in FAILURES]; raise SystemExit(1)
    q=receipt(c,muts,r,digest(mfest),source_sha); print("CP-04 COMPUTE QUALIFIER PASS"); print("corpus_sha256="+q["corpus_sha256"]); print("replay_input_sha256="+q["replay_input_sha256"]); print("mutation_manifest_sha256="+q["mutation_manifest_sha256"]); print("oracle_source_sha256="+q["oracle_source_sha256"]); print("guard_registry_sha256="+q["guard_registry_sha256"]); print("receipt_sha256="+q["receipt_sha256"]); print("claim_ceiling="+q["claim_ceiling"]); print("physical_execution_authority=False")
if __name__=="__main__": main()
