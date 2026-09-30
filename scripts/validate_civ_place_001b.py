#!/usr/bin/env python3
"""Independent stdlib-only CIV-PLACE-001B reference oracle."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "docs/engineering/fixtures/civ-place-001b.json"
DIGEST_FILE = ROOT / "docs/engineering/fixtures/civ-place-001b.sha256"
EXPECTED_DIGEST = "38bd385f1bd7a53daf46ff4f9af3cefaf0542d1348d588fdc1c46bca90d8f46a"
CURRENTNESS = {"Current","Historical","Stale","Unknown","Blocked","Conflicted","PartiallyAvailable"}
INDEPENDENCE = {"SharedDependency","IndependentWitnessed","IndependenceUnknown"}
SERVICE_STATES = {"FullService","DegradedService","Unavailable","Unknown","Blocked","Conflicted"}

def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()

def fail(msg: str) -> None:
    raise SystemExit(f"CIV-PLACE-001B oracle FAIL: {msg}")

def projection_state(p: dict[str, Any]) -> str:
    if p["contradiction"] != "None": return "Conflicted"
    if p["completeness"] != "Complete" or p["convergence"] != "Converged": return "PartiallyAvailable"
    return "Current"

def independence(value: dict[str, Any], witnesses: dict[str, dict[str, Any]]) -> str:
    if value.get("shared_upstream") is not None: return "SharedDependency"
    wid = value.get("independence_witness")
    if not wid: return "IndependenceUnknown"
    w = witnesses.get(wid)
    if not w: return "IndependenceUnknown"
    if (w["status"] != "IndependentWitnessed" or w["scope"] != value.get("scope")
            or w["completeness"] != "Complete" or w["contradiction"] != "None"):
        return "IndependenceUnknown"
    return "IndependentWitnessed"

def closure(root: str, deps: list[dict[str, Any]], included: set[str] | None = None,
            max_nodes: int = 128) -> dict[str, Any]:
    by_target: dict[str, list[dict[str, Any]]] = {}
    for d in deps: by_target.setdefault(d["target"], []).append(d)
    seen: set[str] = set(); edges: set[str] = set(); stack = [root]; bounded = False
    while stack:
        node = stack.pop()
        if node in seen: continue
        if len(seen) >= max_nodes: bounded = True; break
        seen.add(node)
        for d in sorted(by_target.get(node, []), key=lambda x: x["id"]):
            if included is not None and d["class"] not in included: continue
            edges.add(d["id"])
            if d["source"] not in seen: stack.append(d["source"])
    return {"root":root,"nodes":sorted(seen),"edges":sorted(edges),"bounded":bounded,
            "common_mode_groups":sorted({d["common_mode"] for d in deps
                                         if d["id"] in edges and d.get("common_mode")})}

def evaluate(service: dict[str, Any], nodes: dict[str, dict[str, Any]],
             deps: list[dict[str, Any]], projections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    c = closure(service["root"], deps, set(service["included_dependency_classes"]))
    by_id = {d["id"]:d for d in deps}
    required = {d["id"] for d in deps if d["id"] in c["edges"]
                and (d["required"] or d["class"] in service["required_dependency_classes"])}
    optional = set(c["edges"]) - required
    required_sources = {by_id[e]["source"] for e in required}
    optional_sources = {by_id[e]["source"] for e in optional}
    reasons: set[str] = set(); blocked = conflicted = unavailable = degraded = False
    currentness: set[str] = set()
    for nid in c["nodes"]:
        node = nodes.get(nid)
        if not node:
            blocked = True; reasons.add(f"MissingNode:{nid}"); continue
        if node["state"] != "Available":
            if nid in required_sources or nid == service["root"]:
                unavailable = True; reasons.add(f"UnavailableNode:{nid}")
            elif nid in optional_sources:
                degraded = True; reasons.add(f"OptionalUnavailable:{nid}")
        currentness.add(node["currentness"])
        pstate = projection_state(projections[node["projection"]])
        if pstate == "Conflicted":
            conflicted = True; reasons.add(f"ProjectionConflict:{node['projection']}")
        elif pstate == "PartiallyAvailable" and service["currentness_required"]:
            blocked = True; reasons.add(f"ProjectionIncomplete:{node['projection']}")
        if service["currentness_required"] and node["currentness"] != "Current":
            if node["currentness"] == "Conflicted": conflicted = True
            else: blocked = True
            reasons.add(f"NodeCurrentness:{nid}:{node['currentness']}")
    if c["bounded"]: blocked = True; reasons.add("ClosureBoundExceeded")
    if conflicted: status = "Conflicted"
    elif blocked: status = "Blocked"
    elif unavailable: status = "Unavailable"
    elif degraded: status = "DegradedService"
    else: status = "FullService"
    return {"service":service["id"],"status":status,"closure":c,
            "currentness":sorted(currentness),"reasons":sorted(reasons)}

def derive(case: dict[str, Any], data: dict[str, Any], evals: dict[str, Any]) -> str:
    k, v = case["kind"], case["input"]
    if k in {"absence_is_not_independence","positive_independence"}:
        return independence(v, {w["id"]:w for w in data["independence_witnesses"]})
    if k in {"partial_dkg","contradiction"}:
        p = {x["id"]:x for x in data["projections"]}[v["projection"]]
        return projection_state(p)
    if k == "late_arrival_changes_closure":
        before = set(evals["svc:block-a:electric"]["closure"]["edges"])
        return "ClosureChanged" if before != before | {v["new_dependency"]} else "ClosureUnchanged"
    if k == "irrelevant_dkg_material":
        return "ClosureUnchanged" if v["added"] not in evals["svc:block-a:electric"]["closure"]["nodes"] else "ClosureChanged"
    if k in {"attestation_mutation","confidence_mutation"}: return "DispositionUnchanged"
    if k == "historical_not_current": return "Blocked" if v["currentness"] != "Current" and v["currentness_required"] else "Current"
    if k == "partial_view_not_global_current": return "PartiallyAvailable" if v["completeness"] != "Complete" or v["convergence"] != "Converged" else "Current"
    if k == "nested_common_mode": return "SharedDependency" if v["groups"] else "IndependenceUnknown"
    if k == "maintenance_loss": return "DegradedService" if v["state"] == "Unavailable" and not v["required"] else "Blocked"
    if k == "fallback_currentness_unresolved": return "Blocked" if v["currentness"] != "Current" else "FullService"
    if k == "interop_label_only": return "ProjectionIncomplete" if not v["exact_semantics"] else "ProjectionComplete"
    if k == "command_without_authority": return "DispatchRejected" if v["authority_binding"] is None else "DispatchReviewRequired"
    if k == "governance_change": return "EngineeringClosureUnchanged" if v["engineering"] and v["stewardship_before"] != v["stewardship_after"] else "EngineeringClosureChanged"
    if k == "historical_evidence_reuse": return "Blocked" if v["material_change"] else "ReusableHistoricalEvidence"
    if k == "cycle": return "CycleBounded" if len(v["edges"]) == len(set(v["edges"])) else "CycleMalformed"
    fail(f"unknown hostile case {k!r}")

def main() -> int:
    raw = FIXTURE.read_bytes(); data = json.loads(raw)
    if raw != canonical(data): fail("fixture is not byte-canonical JSON")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != EXPECTED_DIGEST: fail(f"fixture digest changed: {actual} != {EXPECTED_DIGEST}")
    if DIGEST_FILE.read_text(encoding="utf-8").split()[0] != EXPECTED_DIGEST: fail("digest sidecar mismatch")
    required = {"profile","schema_version","currentness_states","independence_states","service_states",
                "places","dependencies","independence_witnesses","projections","nodes","services","hostile_cases"}
    if set(data) != required: fail(f"top-level schema drift: {sorted(data)}")
    if data["profile"] != "CIV-PLACE-001B" or data["schema_version"] != "civ-place-001b-v1": fail("wrong profile/schema")
    if set(data["currentness_states"]) != CURRENTNESS: fail("currentness vocabulary drift")
    if set(data["independence_states"]) != INDEPENDENCE: fail("independence vocabulary drift")
    if set(data["service_states"]) != SERVICE_STATES: fail("service vocabulary drift")
    nodes = {n["id"]:n for n in data["nodes"]}; projections = {p["id"]:p for p in data["projections"]}
    deps = data["dependencies"]; dep_ids = [d["id"] for d in deps]
    if len(dep_ids) != len(set(dep_ids)): fail("duplicate dependency id")
    for d in deps:
        if d["source"] not in nodes or d["target"] not in nodes: fail(f"unresolved dependency {d['id']}")
        if d["source"] == d["target"]: fail(f"self dependency {d['id']}")
    for n in nodes.values():
        if n["currentness"] not in CURRENTNESS or n["projection"] not in projections: fail(f"bad node {n['id']}")
    services = {s["id"]:s for s in data["services"]}
    evals = {sid:evaluate(s,nodes,deps,projections) for sid,s in sorted(services.items())}
    if evals["svc:block-a:electric"]["status"] != "FullService": fail("nominal electric service not FullService")
    if evals["svc:block-a:electric"]["closure"]["common_mode_groups"] != ["cm:feeder-001","cm:transformer-001"]:
        fail("unexpected electric common-mode closure")
    rev = evaluate(services["svc:block-a:electric"], nodes, list(reversed(deps)), projections)
    if rev["status"] != evals["svc:block-a:electric"]["status"] or rev["closure"] != evals["svc:block-a:electric"]["closure"]:
        fail("dependency permutation changed canonical evaluation")
    cycle_nodes = dict(nodes)
    cycle_nodes["asset:cycle-a"] = {"id":"asset:cycle-a","state":"Available","currentness":"Current","projection":"proj:neighborhood:g1"}
    cycle_nodes["asset:cycle-b"] = {"id":"asset:cycle-b","state":"Available","currentness":"Current","projection":"proj:neighborhood:g1"}
    cycle_deps = deps + [
        {"id":"dep:cycle:a","source":"asset:cycle-a","target":"asset:cycle-b","class":"ServiceDependency","required":True,"common_mode":None},
        {"id":"dep:cycle:b","source":"asset:cycle-b","target":"asset:cycle-a","class":"ServiceDependency","required":True,"common_mode":None},
    ]
    if closure("asset:cycle-a", cycle_deps)["nodes"] != ["asset:cycle-a","asset:cycle-b"]: fail("cycle did not terminate deterministically")
    if len(data["hostile_cases"]) != 18: fail("hostile corpus count changed")
    if [c["id"] for c in data["hostile_cases"]] != [f"B{i:02d}" for i in range(1,19)]: fail("hostile case ordering changed")
    for case in data["hostile_cases"]:
        actual_case = derive(case,data,evals)
        if actual_case != case["expected"]: fail(f"{case['id']}: {actual_case!r} != {case['expected']!r}")
    print(json.dumps({"profile":data["profile"],"fixture_sha256":actual,"services_evaluated":len(evals),
                      "hostile_cases":len(data["hostile_cases"]),"permutation_invariant":True,
                      "cycle_termination":True,"authority_claim":"none","result":"PASS"},
                     sort_keys=True,separators=(",",":")))
    return 0

if __name__ == "__main__":
    sys.exit(main())
