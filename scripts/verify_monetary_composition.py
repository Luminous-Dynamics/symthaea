#!/usr/bin/env python3
"""Independent structural checker for monetary composition graphs v1."""
from __future__ import annotations
import json, sys
from pathlib import Path

def fail(msg: str) -> None: raise ValueError(msg)

def check_graph(g: dict) -> None:
    if g.get("schema_version") != "monetary-composition-v1": fail("schema version")
    for key in ("composition_id","semantic_version","nodes","edges","topology","routing","shared_dependencies","generation","claim_ceiling"):
        if key not in g: fail(f"missing {key}")
    node_ids = {n["profile_id"] for n in g["nodes"]}
    if not node_ids: fail("empty nodes")
    if len(node_ids) != len(g["nodes"]): fail("duplicate node")
    for n in g["nodes"]:
        if len(n["profile_digest"]) != 64: fail(f"bad node digest {n['profile_id']}")
    seen_edges=set()
    for e in g["edges"]:
        if e["source"] not in node_ids or e["target"] not in node_ids: fail(f"edge references unknown node {e['edge_id']}")
        if e["edge_id"] in seen_edges: fail(f"duplicate edge {e['edge_id']}")
        seen_edges.add(e["edge_id"])
    for key in ("topology_id","topology_digest"):
        if key not in g["topology"]: fail(f"missing topology {key}")
    for key in ("policy_id","policy_digest"):
        if key not in g["routing"]: fail(f"missing routing {key}")
    for dep in g["shared_dependencies"]:
        if dep["kind"] not in {"settlement_asset","liquidity_pool","backstop","governance","identity","oracle","other"}:
            fail(f"bad dependency kind {dep['kind']}")
        if len(dep["digest"]) != 64: fail(f"bad dependency digest {dep['dependency_id']}")
    if len(g["generation"]["composition_digest"]) != 64: fail("bad composition digest")

def check_negative(f: dict) -> None:
    expected={"COMP-X01","COMP-X02","COMP-X03","COMP-X04","COMP-X05","COMP-X06","COMP-X07","COMP-X08"}
    cases=f.get("cases")
    if {c.get("id") for c in cases} != expected: fail("negative fixture set mismatch")
    for c in cases:
        if not c.get("expected"): fail(f"{c['id']}: missing expected disposition")

if __name__=="__main__":
    if len(sys.argv)!=3:
        print("usage: verify_monetary_composition.py GRAPH.json NEGATIVE.json", file=sys.stderr); raise SystemExit(2)
    try:
        check_graph(json.loads(Path(sys.argv[1]).read_text()))
        check_negative(json.loads(Path(sys.argv[2]).read_text()))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr); raise SystemExit(1)
    print("independent monetary-composition check: graph structure valid; 8 negative fixtures")
