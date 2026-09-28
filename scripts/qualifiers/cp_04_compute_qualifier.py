#!/usr/bin/env python3
"""Independent CP-04 synthetic compute-evidence qualifier (stdlib-only)."""
from __future__ import annotations
import ast, hashlib, json
from copy import deepcopy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CORPUS=ROOT/"docs/engineering/data/cp-04-compute-corpus-v1.json"
MANIFEST=ROOT/"docs/engineering/data/cp-04-compute-mutation-manifest-v1.json"
GRAPH=ROOT/"docs/engineering/data/cp-04-compute-dependency-graph-v1.json"
SOURCE=Path(__file__).resolve()
SCHEMA="cp-04-compute-corpus-v1"
AUTHORITY="representation_only_no_physical_execution_authority"
CEILING="deterministic compute identity, generation, dependency, evidence-separation, replay, negative-result, and authority semantics over synthetic/reference workflows only"
EXPECTED={"C01":"CurrentAndApplicable","C02":"DependencyChanged","C03":"ConfigurationMismatch","C04":"ConfigurationMismatch","C05":"DeploymentMismatch","C06":"ObservationMissing","C07":"UncertaintyInsufficient","C08":"PopulationInferenceBlocked","C09":"HardEnvelopeFailure","C10":"CommonModeNotIndependent","C11":"AuthoritySeparated","C12":"Stale","C13":"StateRestorationUnresolved","C14":"AuthoritySeparated","C15":"AuthoritySeparated","C16":"HistoricalNegativeRetained","C17":"NoPhysicalExecutionAuthority","C18":"RequalificationRequired"}
SCENARIOS={"C01":"complete-thread","C02":"model-parameter-generation-changed","C03":"runtime-toolchain-changed","C04":"accelerator-configuration-changed","C05":"deployment-artifact-changed","C06":"execution-observation-missing","C07":"uncertainty-missing","C08":"single-benchmark-not-population","C09":"tail-envelope-failure","C10":"correlated-channels","C11":"model-output-equals-measurement","C12":"stale-currentness","C13":"lossy-cfc-snapshot","C14":"operational-event-not-qualification","C15":"attestation-not-performance","C16":"negative-result-retained","C17":"synthetic-pass","C18":"workload-profile-changed"}
CASE_GUARDS={"C05":"CP-COMP-HISTORICAL","C06":"CP-COMP-COVERAGE","C10":"CP-COMP-DEPENDENCY","C16":"CP-COMP-NEGATIVE","C17":"CP-COMP-AUTHORITY","C18":"CP-COMP-CURRENTNESS"}
GUARDS={"CP-COMP-COVERAGE":"coverage","CP-COMP-AUTHORITY":"authority","CP-COMP-NEGATIVE":"negative-evidence","CP-COMP-DEPENDENCY":"dependency-boundary","CP-COMP-CURRENTNESS":"currentness","CP-COMP-HISTORICAL":"historical-identity","CP-COMP-INVARIANT":"invariant-integrity"}
FAILURES=[]
EXPECTED_GRAPH_NODES=["requirement","representation","model","runtime","accelerator","deployment","execution","observation","statistics","disposition"]
EXPECTED_GRAPH_EDGES=[["requirement","representation"],["representation","model"],["model","runtime"],["runtime","accelerator"],["accelerator","deployment"],["deployment","execution"],["execution","observation"],["observation","statistics"],["statistics","disposition"]]
EXPECTED_CLOSURES={
    "requirement":["requirement","representation","model","runtime","accelerator","deployment","execution","observation","statistics","disposition"],
    "representation":["representation","model","runtime","accelerator","deployment","execution","observation","statistics","disposition"],
    "model":["model","runtime","accelerator","deployment","execution","observation","statistics","disposition"],
    "runtime":["runtime","accelerator","deployment","execution","observation","statistics","disposition"],
    "accelerator":["accelerator","deployment","execution","observation","statistics","disposition"],
    "deployment":["deployment","execution","observation","statistics","disposition"],
    "execution":["execution","observation","statistics","disposition"],
    "observation":["observation","statistics","disposition"],
    "statistics":["statistics","disposition"],
    "disposition":["disposition"],
}

def canonical(v):
    return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()
def digest(v):
    return hashlib.sha256(canonical(v)).hexdigest()
def load_graph():
    return json.loads(GRAPH.read_text(encoding="utf-8"))
def graph_identity(g):
    return digest(g)
def fail(m,guard_id):
    if guard_id not in GUARDS:
        raise ValueError(f"unregistered guard: {guard_id}")
    FAILURES.append({"guard_id":guard_id,"category":GUARDS[guard_id],"message":m})
def eq(a,b,m,guard_id="CP-COMP-INVARIANT"):
    if a!=b:
        fail(f"{m}: expected {b!r}, got {a!r}",guard_id)
def validate_graph(g):
    if not isinstance(g,dict):
        fail("dependency graph must be object","CP-COMP-DEPENDENCY"); return
    eq(set(g.keys()),{"schema","nodes","edges"},"graph top-level schema","CP-COMP-DEPENDENCY")
    eq(g.get("schema"),"cp-04-compute-dependency-graph-v1","graph schema","CP-COMP-DEPENDENCY")
    nodes=g.get("nodes")
    edges=g.get("edges")
    if not isinstance(nodes,list) or any(not isinstance(n,str) for n in nodes):
        fail("dependency graph nodes must be an ordered list of strings","CP-COMP-DEPENDENCY"); return
    if not isinstance(edges,list) or any(not isinstance(e,list) or len(e)!=2 or any(not isinstance(x,str) for x in e) for e in edges):
        fail("dependency graph edges must be ordered 2-tuples of strings","CP-COMP-DEPENDENCY"); return
    eq(nodes,EXPECTED_GRAPH_NODES,"graph node identity","CP-COMP-DEPENDENCY")
    eq(edges,EXPECTED_GRAPH_EDGES,"graph edge identity","CP-COMP-DEPENDENCY")
    if len(nodes)!=len(set(nodes)):
        fail("dependency graph contains duplicate node","CP-COMP-DEPENDENCY")
    node_set=set(nodes)
    if any(e[0] not in node_set or e[1] not in node_set for e in edges):
        fail("dependency graph contains dangling edge","CP-COMP-DEPENDENCY"); return
    adjacency={n:[] for n in node_set}
    for a,b in edges:
        adjacency[a].append(b)
    def visit(n,path):
        if n in path: return True
        return any(visit(x,path|{n}) for x in adjacency[n])
    if any(visit(n,set()) for n in node_set):
        fail("dependency graph contains cycle","CP-COMP-DEPENDENCY")
def dependency_closure(changed,graph):
    adjacency={n:set() for n in graph["nodes"]}
    for a,b in graph["edges"]:
        adjacency[a].add(b)
    seen=set(changed); todo=list(changed)
    while todo:
        n=todo.pop()
        for nxt in adjacency.get(n,set()):
            if nxt not in seen:
                seen.add(nxt); todo.append(nxt)
    return sorted(seen)
def replay_input(c,graph):
    return {"schema":c["schema"],"authority":c["authority"],"claim_ceiling":c["claim_ceiling"],"replay_semantics":c["replay_semantics"],"graph_identity":graph_identity(graph),"case_manifest":[{"case_id":x["case_id"],"scenario":x["scenario"]} for x in c["cases"]]}
def validate_replay_boundary(c,graph):
    baseline=digest(replay_input(c,graph))
    for field,value in (("schema","cp-04-compute-corpus-v1-mutated"),("authority","broader_authority"),("claim_ceiling","broader_claims"),("replay_semantics","mutable history")):
        mutated=deepcopy(c); mutated[field]=value
        if digest(replay_input(mutated,graph))==baseline:
            fail(f"replay identity ignores historical corpus field {field}","CP-COMP-INVARIANT")
    for case in c["cases"]:
        mutated=deepcopy(c)
        for candidate in mutated["cases"]:
            if candidate["case_id"]==case["case_id"]:
                candidate["scenario"]=candidate["scenario"]+"-mutated"
                break
        if digest(replay_input(mutated,graph))==baseline:
            fail(f"replay identity ignores scenario for {case['case_id']}","CP-COMP-INVARIANT")
    mutated=deepcopy(c)
    mutated["cases"][0]["expected_disposition"]="DerivedMutation"
    if digest(replay_input(mutated,graph))!=baseline:
        fail("replay identity incorrectly depends on derived disposition","CP-COMP-INVARIANT")
def source_audit():
    source=SOURCE.read_text(encoding="utf-8")
    tree=ast.parse(source)
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="fail":
            if len(n.args)!=2 or not ((isinstance(n.args[1],ast.Name) and n.args[1].id=="guard_id") or (isinstance(n.args[1],ast.Constant) and n.args[1].value in GUARDS)):
                fail("source audit: fail() must carry explicit registered guard_id","CP-COMP-INVARIANT")
    forbidden={"symthaea","torch","numpy","pandas","onnx","tensorflow"}
    for n in ast.walk(tree):
        if isinstance(n,(ast.Import,ast.ImportFrom)) and any(a.name.split(".")[0] in forbidden for a in n.names):
            fail("source audit: forbidden production/runtime import","CP-COMP-INVARIANT")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()
def validate_replay_completeness(c,graph):
    baseline=replay_input(c,graph)
    required={"schema","authority","claim_ceiling","replay_semantics","graph_identity","case_manifest"}
    eq(set(baseline),required,"replay input field set","CP-COMP-INVARIANT")
    eq(len(baseline["case_manifest"]),len(c["cases"]),"replay case coverage","CP-COMP-INVARIANT")
    eq([x["case_id"] for x in baseline["case_manifest"]],[x["case_id"] for x in c["cases"]],"replay case order","CP-COMP-INVARIANT")
    eq([x["scenario"] for x in baseline["case_manifest"]],[x["scenario"] for x in c["cases"]],"replay scenario binding","CP-COMP-INVARIANT")

def validate(c):
    eq(c.get("schema"),SCHEMA,"schema")
    eq(c.get("authority"),AUTHORITY,"authority")
    eq(c.get("claim_ceiling"),CEILING,"claim ceiling")
    eq(c.get("replay_semantics"),"historical identities are immutable inputs; dispositions are derived outputs","replay semantics")
    cases=c.get("cases")
    if not isinstance(cases,list):
        fail("cases must be list","CP-COMP-COVERAGE"); return
    ids=[x.get("case_id") for x in cases]
    eq(ids,list(EXPECTED),"ordered case manifest")
    if len(set(ids))!=len(ids):
        fail("duplicate case IDs","CP-COMP-COVERAGE")
    for x in cases:
        cid=x.get("case_id")
        if not isinstance(x.get("scenario"),str):
            fail(f"{cid}: scenario missing","CP-COMP-COVERAGE")
        eq(x.get("scenario"),SCENARIOS.get(cid),"case scenario",CASE_GUARDS.get(cid,"CP-COMP-COVERAGE"))
        eq(x.get("expected_disposition"),EXPECTED.get(cid,"Unknown"),f"{cid} disposition",CASE_GUARDS.get(cid,"CP-COMP-INVARIANT"))
    semantic={"model-output-equals-measurement":"AuthoritySeparated","attestation-not-performance":"AuthoritySeparated","synthetic-pass":"NoPhysicalExecutionAuthority","negative-result-retained":"HistoricalNegativeRetained","lossy-cfc-snapshot":"StateRestorationUnresolved"}
    for x in cases:
        if x["scenario"] in semantic:
            eq(x["expected_disposition"],semantic[x["scenario"]],f"{x['case_id']} semantic guard")
def validate_manifest(m):
    eq(m.get("schema"),"cp-04-compute-mutation-manifest-v1","manifest schema")
    eq(m.get("guard_registry"),GUARDS,"guard registry")
    expected=["remove-model-runtime-edge","reverse-model-runtime-edge","duplicate-model-runtime-edge","add-graph-cycle","rename-graph-node","add-graph-field","malformed-graph-edge","duplicate-model-node","dangling-graph-edge","drop-C06","promote-C17","rewrite-C16","collapse-C10","change-C18","erase-C05"]
    eq([x.get("mutation_id") for x in m.get("mutations",[])],expected,"mutation manifest")
    for x in m.get("mutations",[]):
        if x.get("guard_id") not in GUARDS:
            fail(f"unknown guard: {x.get('guard_id')}","CP-COMP-INVARIANT")
def mutations(c,graph):
    graph_specs=[
        ("remove-model-runtime-edge",lambda g:g["edges"].remove(["model","runtime"]),"CP-COMP-DEPENDENCY"),
        ("reverse-model-runtime-edge",lambda g:g["edges"].__setitem__(2,["runtime","model"]),"CP-COMP-DEPENDENCY"),
        ("duplicate-model-runtime-edge",lambda g:g["edges"].append(["model","runtime"]),"CP-COMP-DEPENDENCY"),
        ("add-graph-cycle",lambda g:g["edges"].append(["disposition","model"]),"CP-COMP-DEPENDENCY"),
        ("rename-graph-node",lambda g:g["nodes"].__setitem__(2,"model-v2"),"CP-COMP-DEPENDENCY"),
        ("add-graph-field",lambda g:g.__setitem__("unbound","unexpected"),"CP-COMP-DEPENDENCY"),
        ("malformed-graph-edge",lambda g:g["edges"].__setitem__(2,["model"]),"CP-COMP-DEPENDENCY"),
        ("duplicate-model-node",lambda g:g["nodes"].append("model"),"CP-COMP-DEPENDENCY"),
        ("dangling-graph-edge",lambda g:g["edges"].append(["model","missing-node"]),"CP-COMP-DEPENDENCY"),
    ]
    case_specs=[
        ("drop-C06",lambda x:x["cases"].pop(5),"CP-COMP-COVERAGE"),
        ("promote-C17",lambda x:x["cases"][16].update(expected_disposition="CurrentAndApplicable"),"CP-COMP-AUTHORITY"),
        ("rewrite-C16",lambda x:x["cases"][15].update(scenario="no-result"),"CP-COMP-NEGATIVE"),
        ("collapse-C10",lambda x:x["cases"][9].update(expected_disposition="CurrentAndApplicable"),"CP-COMP-DEPENDENCY"),
        ("change-C18",lambda x:x["cases"][17].update(expected_disposition="CurrentAndApplicable"),"CP-COMP-CURRENTNESS"),
        ("erase-C05",lambda x:x["cases"][4].update(scenario="complete-thread"),"CP-COMP-HISTORICAL"),
    ]
    out=[]
    for label,mut,guard in graph_specs:
        old=FAILURES[:]; FAILURES.clear()
        mutated=deepcopy(graph); mut(mutated); validate_graph(mutated)
        caught=bool(FAILURES); detail=FAILURES[:]
        FAILURES.clear(); FAILURES.extend(old)
        if not caught:
            fail(f"mutation {label} was not rejected by {guard}","CP-COMP-INVARIANT")
        elif not any(d["guard_id"] == guard for d in detail):
            fail(f"mutation {label} was rejected by the wrong guard","CP-COMP-INVARIANT")
        out.append({"mutation_id":label,"guard_id":guard,"rejected":caught,"diagnostics":detail})
    for label,mut,guard in case_specs:
        old=FAILURES[:]; FAILURES.clear()
        mutated=deepcopy(c); mut(mutated); validate(mutated)
        caught=bool(FAILURES); detail=FAILURES[:]
        FAILURES.clear(); FAILURES.extend(old)
        if not caught:
            fail(f"mutation {label} was not rejected by {guard}","CP-COMP-INVARIANT")
        elif not any(d["guard_id"] == guard for d in detail):
            fail(f"mutation {label} was rejected by the wrong guard","CP-COMP-INVARIANT")
        out.append({"mutation_id":label,"guard_id":guard,"rejected":caught,"diagnostics":detail})
    return out
def receipt(c,m,r,graph_sha,manifest_sha,source_sha):
    p={"schema":"cp-04-compute-qualification-receipt-v1","corpus_sha256":digest(c),"replay_input_sha256":r,"graph_identity_sha256":graph_sha,"mutation_manifest_sha256":manifest_sha,"oracle_source_sha256":source_sha,"guard_registry_sha256":digest(GUARDS),"mutation_results":m,"claim_ceiling":c["claim_ceiling"],"disposition":"PASS","physical_execution_authority":False}
    p["receipt_sha256"]=digest(p); return p
def main():
    c=json.loads(CORPUS.read_text(encoding="utf-8"))
    manifest=json.loads(MANIFEST.read_text(encoding="utf-8"))
    graph=load_graph()
    source_sha=source_audit()
    validate(c); validate_manifest(manifest); validate_graph(graph)
    graph_sha=graph_identity(graph)
    replay_sha=digest(replay_input(c,graph))
    validate_replay_boundary(c,graph)
    validate_replay_completeness(c,graph)
    muts=mutations(c,graph)
    manifest_pairs=[(x.get("mutation_id"),x.get("guard_id")) for x in manifest.get("mutations",[])]
    actual_pairs=[(x["mutation_id"],x["guard_id"]) for x in muts]
    eq(actual_pairs,manifest_pairs,"mutation implementation/manifest binding","CP-COMP-INVARIANT")
    for node,expected_closure in EXPECTED_CLOSURES.items():
        if dependency_closure([node],graph)!=expected_closure:
            fail(f"{node} invalidation closure incorrect","CP-COMP-DEPENDENCY")
    derived_only=deepcopy(c)
    for case in derived_only["cases"]:
        case["expected_disposition"]="SyntheticDerivedDisposition"
    if digest(replay_input(c,graph))!=digest(replay_input(derived_only,graph)):
        fail("replay input incorrectly depends on derived disposition","CP-COMP-INVARIANT")
    altered_graph=deepcopy(graph)
    altered_graph["edges"]=altered_graph["edges"][:-1]
    if graph_identity(altered_graph)==graph_sha:
        fail("graph identity is not sensitive to graph changes","CP-COMP-DEPENDENCY")
    deployment=dependency_closure(["deployment"],graph)
    if not {"deployment","execution","observation","statistics","disposition"}.issubset(deployment) or "model" in deployment or "representation" in deployment:
        fail("deployment invalidation boundary incorrect","CP-COMP-DEPENDENCY")
    if dependency_closure(["runtime"],graph)!=sorted(["runtime","accelerator","deployment","execution","observation","statistics","disposition"]):
        fail("runtime invalidation closure incorrect","CP-COMP-DEPENDENCY")
    if FAILURES or not all(x["rejected"] for x in muts):
        print("CP-04 COMPUTE QUALIFIER FAIL")
        for x in FAILURES: print(" -",x)
        raise SystemExit(1)
    q=receipt(c,muts,replay_sha,graph_sha,digest(manifest),source_sha)
    receipt_body=deepcopy(q)
    receipt_body.pop("receipt_sha256")
    eq(digest(receipt_body),q["receipt_sha256"],"receipt self-verification","CP-COMP-INVARIANT")
    if FAILURES:
        print("CP-04 COMPUTE QUALIFIER FAIL")
        for x in FAILURES: print(" -",x)
        raise SystemExit(1)
    print("CP-04 COMPUTE QUALIFIER PASS")
    for k in ("corpus_sha256","replay_input_sha256","graph_identity_sha256","mutation_manifest_sha256","oracle_source_sha256","guard_registry_sha256","receipt_sha256"):
        print(k+"="+q[k])
    print("claim_ceiling="+q["claim_ceiling"])
    print("physical_execution_authority=False")
if __name__=="__main__":
    main()
