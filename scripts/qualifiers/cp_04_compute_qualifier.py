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
SCENARIOS={"C01":"complete-thread","C02":"model-parameter-generation-changed","C03":"runtime-toolchain-changed","C04":"accelerator-configuration-changed","C05":"deployment-artifact-changed","C06":"execution-observation-missing","C07":"uncertainty-missing","C08":"single-benchmark-not-population","C09":"tail-envelope-failure","C10":"correlated-channels","C11":"model-output-equals-measurement","C12":"stale-currentness","C13":"lossy-cfc-snapshot","C14":"operational-event-not-qualification","C15":"attestation-not-performance","C16":"negative-result-retained","C17":"synthetic-pass","C18":"workload-profile-changed"}
CASE_GUARDS={"C05":"CP-COMP-HISTORICAL","C06":"CP-COMP-COVERAGE","C10":"CP-COMP-DEPENDENCY","C16":"CP-COMP-NEGATIVE","C17":"CP-COMP-AUTHORITY","C18":"CP-COMP-CURRENTNESS"}
GUARDS={"CP-COMP-COVERAGE":"coverage","CP-COMP-AUTHORITY":"authority","CP-COMP-NEGATIVE":"negative-evidence","CP-COMP-DEPENDENCY":"dependency-boundary","CP-COMP-CURRENTNESS":"currentness","CP-COMP-HISTORICAL":"historical-identity","CP-COMP-INVARIANT":"invariant-integrity"}
FAILURES=[]
GRAPH_NODES=["requirement","representation","model","runtime","accelerator","deployment","execution","observation","statistics","disposition"]
GRAPH_EDGES=[["requirement","representation"],["representation","model"],["model","runtime"],["runtime","accelerator"],["accelerator","deployment"],["deployment","execution"],["execution","observation"],["observation","statistics"],["statistics","disposition"]]
COMPUTE_GRAPH={
 "schema":"cp-04-compute-dependency-graph-v1",
 "nodes":GRAPH_NODES,
 "edges":GRAPH_EDGES
}
def validate_graph(g):
    if not isinstance(g,dict): fail("dependency graph must be object","CP-COMP-DEPENDENCY"); return
    eq(g.get("schema"),"cp-04-compute-dependency-graph-v1","graph schema","CP-COMP-DEPENDENCY")
    eq(g.get("nodes"),GRAPH_NODES,"graph node identity","CP-COMP-DEPENDENCY")
    eq(g.get("edges"),GRAPH_EDGES,"graph edge identity","CP-COMP-DEPENDENCY")
    if len(g.get("nodes",[])) != len(set(g.get("nodes",[]))): fail("dependency graph contains duplicate node","CP-COMP-DEPENDENCY")
    nodes=set(g.get("nodes",[]))
    if any(len(e)!=2 or e[0] not in nodes or e[1] not in nodes for e in g.get("edges",[])): fail("dependency graph contains dangling edge","CP-COMP-DEPENDENCY")
    adjacency={n:[] for n in nodes}
    for a,b in g.get("edges",[]): adjacency[a].append(b)
    def visit(n,path):
        if n in path: return True
        return any(visit(x,path|{n}) for x in adjacency[n])
    if any(visit(n,set()) for n in nodes): fail("dependency graph contains cycle","CP-COMP-DEPENDENCY")
def dependency_closure(changed):
    # Directed descendants: a changed dependency invalidates its consumers,
    # never its prerequisites.
    adjacency={n:set() for n in COMPUTE_GRAPH["nodes"]}
    for a,b in COMPUTE_GRAPH["edges"]: adjacency[a].add(b)
    seen=set(changed); todo=list(changed)
    while todo:
        n=todo.pop()
        for nxt in adjacency.get(n,set()):
            if nxt not in seen: seen.add(nxt); todo.append(nxt)
    return sorted(seen)
def canonical(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()
def digest(v): return hashlib.sha256(canonical(v)).hexdigest()
def fail(m, guard_id):
    if guard_id not in GUARDS: raise ValueError(f"unregistered guard: {guard_id}")
    FAILURES.append({"guard_id":guard_id,"category":GUARDS[guard_id],"message":m})
def eq(a,b,m,guard_id="CP-COMP-INVARIANT"):
    if a!=b: fail(f"{m}: expected {b!r}, got {a!r}",guard_id)
def replay_input(c):
    return {"schema":c["schema"],"authority":c["authority"],"claim_ceiling":c["claim_ceiling"],"replay_semantics":c["replay_semantics"],"graph_identity":graph_identity(),"case_manifest":[{"case_id":x["case_id"],"scenario":x["scenario"]} for x in c["cases"]]}
def source_audit():
    source=SOURCE.read_text(encoding="utf-8")
    tree=ast.parse(source)
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="fail":
            if len(n.args)!=2 or not (isinstance(n.args[1],ast.Name) and n.args[1].id=="guard_id" or isinstance(n.args[1],ast.Constant) and n.args[1].value in GUARDS):
                fail("source audit: fail() must carry explicit registered guard_id","CP-COMP-INVARIANT")
    forbidden={"symthaea","torch","numpy","pandas","onnx","tensorflow"}
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)) and any(a.name.split(".")[0] in forbidden for a in n.names): fail("source audit: forbidden production/runtime import","CP-COMP-INVARIANT")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()
def validate(c):
    eq(c.get("schema"),SCHEMA,"schema"); eq(c.get("authority"),AUTHORITY,"authority"); eq(c.get("claim_ceiling"),CEILING,"claim ceiling")
    eq(c.get("replay_semantics"),"historical identities are immutable inputs; dispositions are derived outputs","replay semantics")
    cases=c.get("cases")
    if not isinstance(cases,list): fail("cases must be list","CP-COMP-COVERAGE"); return
    ids=[x.get("case_id") for x in cases]; eq(ids,list(EXPECTED),"ordered case manifest")
    if len(set(ids))!=len(ids): fail("duplicate case IDs","CP-COMP-COVERAGE")
    for x in cases:
        cid=x.get("case_id")
        if not isinstance(x.get("scenario"),str): fail(f"{cid}: scenario missing","CP-COMP-COVERAGE")
        eq(x.get("scenario"),SCENARIOS.get(cid),"case scenario",CASE_GUARDS.get(cid,"CP-COMP-COVERAGE"))
        eq(x.get("expected_disposition"),EXPECTED.get(cid,"Unknown"),f"{cid} disposition",CASE_GUARDS.get(cid,"CP-COMP-INVARIANT"))
    guards={"model-output-equals-measurement":"AuthoritySeparated","attestation-not-performance":"AuthoritySeparated","synthetic-pass":"NoPhysicalExecutionAuthority","negative-result-retained":"HistoricalNegativeRetained","lossy-cfc-snapshot":"StateRestorationUnresolved"}
    for x in cases:
        if x["scenario"] in guards: eq(x["expected_disposition"],guards[x["scenario"]],f"{x['case_id']} semantic guard")
def validate_manifest(m):
    eq(m.get("schema"),"cp-04-compute-mutation-manifest-v1","manifest schema")
    eq(m.get("guard_registry"),GUARDS,"guard registry")
    expected=["remove-model-runtime-edge","add-graph-cycle","rename-graph-node","drop-C06","promote-C17","rewrite-C16","collapse-C10","change-C18","erase-C05"]
    eq([x.get("mutation_id") for x in m.get("mutations",[])],expected,"mutation manifest")
    for x in m.get("mutations",[]):
        if x.get("guard_id") not in GUARDS: fail(f"unknown guard: {x.get('guard_id')}","CP-COMP-INVARIANT")
def mutations(c):
    specs=[
        ("remove-model-runtime-edge",lambda x:x[0]["edges"].remove(["model","runtime"]),"CP-COMP-DEPENDENCY"),
        ("add-graph-cycle",lambda x:x[0]["edges"].append(["disposition","model"]),"CP-COMP-DEPENDENCY"),
        ("rename-graph-node",lambda x:x[0]["nodes"].__setitem__(2,"model-v2"),"CP-COMP-DEPENDENCY"),
("drop-C06",lambda x:x["cases"].pop(5),"CP-COMP-COVERAGE"),("promote-C17",lambda x:x["cases"][16].update(expected_disposition="CurrentAndApplicable"),"CP-COMP-AUTHORITY"),("rewrite-C16",lambda x:x["cases"][15].update(scenario="no-result"),"CP-COMP-NEGATIVE"),("collapse-C10",lambda x:x["cases"][9].update(expected_disposition="CurrentAndApplicable"),"CP-COMP-DEPENDENCY"),("change-C18",lambda x:x["cases"][17].update(expected_disposition="CurrentAndApplicable"),"CP-COMP-CURRENTNESS"),("erase-C05",lambda x:x["cases"][4].update(scenario="complete-thread"),"CP-COMP-HISTORICAL")]
    out=[]
    for label,mut,guard in specs:
        y=deepcopy(c); old=FAILURES[:]; FAILURES.clear(); graph=deepcopy(COMPUTE_GRAPH); mut([graph]); validate_graph(graph); validate(y); caught=bool(FAILURES); detail=FAILURES[:]; FAILURES.clear(); FAILURES.extend(old)
        if not caught: fail(f"mutation {label} was not rejected by {guard}","CP-COMP-INVARIANT")
        out.append({"mutation_id":label,"guard_id":guard,"rejected":caught,"diagnostics":detail})
    return out
def receipt(c,m,r,manifest_sha,source_sha):
    p={"schema":"cp-04-compute-qualification-receipt-v1","corpus_sha256":digest(c),"replay_input_sha256":r,"graph_identity_sha256":graph_identity(),"mutation_manifest_sha256":manifest_sha,"oracle_source_sha256":source_sha,"guard_registry_sha256":digest(GUARDS),"mutation_results":m,"claim_ceiling":c["claim_ceiling"],"disposition":"PASS","physical_execution_authority":False}
    p["receipt_sha256"]=digest(p); return p
def main():
    c=json.loads(CORPUS.read_text(encoding="utf-8")); mfest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    source_sha=source_audit(); validate(c); validate_manifest(mfest); validate_graph(COMPUTE_GRAPH); r=digest(replay_input(c)); muts=mutations(c)
    closure=dependency_closure(["model"])
    if not {"model","runtime","accelerator","deployment","execution","observation","statistics","disposition"}.issubset(closure): fail("model dependency closure incomplete","CP-COMP-DEPENDENCY")
    deployment_closure=dependency_closure(["deployment"])
    if not {"deployment","execution","observation","statistics","disposition"}.issubset(deployment_closure) or "model" in deployment_closure or "representation" in deployment_closure: fail("deployment invalidation boundary incorrect","CP-COMP-DEPENDENCY")
    if dependency_closure(["runtime"]) != sorted(["runtime","accelerator","deployment","execution","observation","statistics","disposition"]): fail("runtime invalidation closure incorrect","CP-COMP-DEPENDENCY")
    if FAILURES or not all(x["rejected"] for x in muts):
        print("CP-04 COMPUTE QUALIFIER FAIL"); [print(" -",x) for x in FAILURES]; raise SystemExit(1)
    q=receipt(c,muts,r,digest(mfest),source_sha); print("CP-04 COMPUTE QUALIFIER PASS"); print("corpus_sha256="+q["corpus_sha256"]); print("replay_input_sha256="+q["replay_input_sha256"]); print("graph_identity_sha256="+q["graph_identity_sha256"]); print("mutation_manifest_sha256="+q["mutation_manifest_sha256"]); print("oracle_source_sha256="+q["oracle_source_sha256"]); print("guard_registry_sha256="+q["guard_registry_sha256"]); print("receipt_sha256="+q["receipt_sha256"]); print("claim_ceiling="+q["claim_ceiling"]); print("physical_execution_authority=False")
if __name__=="__main__": main()
