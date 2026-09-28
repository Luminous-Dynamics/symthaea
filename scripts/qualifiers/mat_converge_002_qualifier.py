#!/usr/bin/env python3
"""Independent MAT-CONVERGE-002 fixture oracle (stdlib only)."""
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
FIELDS = ("demand", "profile", "candidate", "process", "property", "measurement")
GRAPH_NODE_KEYS = {"id", "kind", "ref", "generation"}
GRAPH_EDGE_KEYS = {"id", "from", "to", "kind", "historical"}

def fail(msg):
    raise AssertionError(msg)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()

def edge_hash(case):
    return hashlib.sha256(canonical({
        "schema": "integration-edge-v1",
        "edge": "|".join(case["refs"][field] for field in FIELDS),
    })).hexdigest()

def graph_identity(graph):
    return {
        "schema": graph["schema"],
        "nodes": [
            {k: n[k] for k in ("id", "kind", "ref", "generation")}
            for n in graph["nodes"]
        ],
        "edges": [
            {k: e[k] for k in ("id", "from", "to", "kind", "historical")}
            for e in graph["edges"]
        ],
        "replay_rule": graph["replay_rule"],
    }

def replay_digest(cases, graph):
    """Hash immutable case references and graph identity, never dispositions."""
    payload = {
        "cases": [{
            "id": c["id"],
            "edge_hash": edge_hash(c),
            "refs": {field: c["refs"][field] for field in FIELDS},
        } for c in cases],
        "state_graph": graph_identity(graph),
    }
    return hashlib.sha256(canonical(payload)).hexdigest()

def check_graph(graph):
    if set(graph) != {"schema", "historical_records_immutable",
                      "derived_dispositions_recomputable", "nodes", "edges",
                      "replay_rule"}:
        fail("state graph schema drift")
    if graph["schema"] != "mat-converge-002-state-graph-v1":
        fail("state graph schema mismatch")
    if graph["historical_records_immutable"] is not True:
        fail("historical graph records are not immutable")
    if graph["derived_dispositions_recomputable"] is not True:
        fail("derived dispositions are not declared recomputable")
    if graph["replay_rule"] != (
        "replay digest is over immutable node/edge identities; derived dispositions are excluded"
    ):
        fail("state graph replay rule mismatch")
    nodes, edges = graph["nodes"], graph["edges"]
    if not nodes or not edges:
        fail("state graph is empty")
    node_ids = [n["id"] for n in nodes]
    edge_ids = [e["id"] for e in edges]
    if len(node_ids) != len(set(node_ids)):
        fail("duplicate state graph node identity")
    if len(edge_ids) != len(set(edge_ids)):
        fail("duplicate state graph edge identity")
    node_set = set(node_ids)
    for node in nodes:
        if set(node) != GRAPH_NODE_KEYS:
            fail(f"state graph node schema drift: {node.get('id')}")
    for edge in edges:
        if set(edge) != GRAPH_EDGE_KEYS:
            fail(f"state graph edge schema drift: {edge.get('id')}")
        if edge["historical"] is not True:
            fail(f"non-historical edge: {edge['id']}")
        if edge["from"] not in node_set or edge["to"] not in node_set:
            fail(f"dangling state graph edge: {edge['id']}")
    if not any(n["kind"] == "negative-edge" for n in nodes):
        fail("negative evidence is absent from state graph")
    required_edges = {"edge/C07", "edge/C09", "edge/C10", "edge/C16"}
    if not required_edges.issubset(edge_ids):
        fail("critical state graph edges missing")

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
    required_keys = {"id", "demand", "profile", "candidate", "process",
                     "property", "measurement", "outcome", "refs"}
    for c in cases:
        if set(c) != required_keys:
            fail(f"schema drift in {c['id']}")
        if c["outcome"] != EXPECTED[c["id"]]:
            fail(f"unexpected disposition for {c['id']}")
        if set(c["refs"]) != set(FIELDS):
            fail(f"reference set drift in {c['id']}")
        for field in FIELDS:
            if c["refs"][field] != f"{field}/{c[field]}":
                fail(f"non-derived reference in {c['id']}:{field}")
    by_id = {c["id"]: c for c in cases}
    if by_id["C02"]["process"] != "G2" or by_id["C02"]["property"] != "E1":
        fail("process generation rewrote historical chemistry evidence")
    if by_id["C03"]["profile"] == "P1":
        fail("profile change collapsed")
    if by_id["C04"]["property"] != "E2":
        fail("evaluator dependency collapsed")
    if by_id["C05"]["measurement"] != "M2":
        fail("measurement-generation dependency collapsed")
    if by_id["C07"]["outcome"] != "negative-edge-addressable" or by_id["C08"]["outcome"] != "negative-edge-addressable":
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
    if not required.issubset({c["outcome"] for c in cases}):
        fail("critical authority/negative-evidence controls missing")
    check_graph(doc["state_graph"])

def expect_failure(doc, cases, label):
    try:
        check(doc, cases)
    except AssertionError:
        return
    fail(f"mutation escaped oracle: {label}")

def main():
    if len(sys.argv) != 2:
        print("usage: mat_converge_002_qualifier.py PATH", file=sys.stderr)
        return 2
    path = pathlib.Path(sys.argv[1])
    raw = path.read_bytes()
    doc = json.loads(raw)
    if doc.get("schema") != "mat-converge-002-fixture-v1":
        fail("schema mismatch")
    cases = doc.get("cases", [])
    check(doc, cases)
    graph = doc["state_graph"]
    baseline_digest = replay_digest(cases, graph)
    mutations = [
        ("remove-process-ref", "C02"),
        ("change-process-generation", "C02"),
        ("change-profile-ref", "C03"),
        ("change-evaluator-generation", "C04"),
        ("change-measurement-generation", "C05"),
        ("change-ranking-only", "C06"),
        ("delete-negative-case", "C07"),
        ("promote-authority", "C09"),
        ("change-disposition", "C16"),
        ("rewrite-historical-ref", "C02"),
    ]
    for name, target in mutations:
        mutated = json.loads(json.dumps(cases))
        case = next(c for c in mutated if c["id"] == target)
        if name == "remove-process-ref":
            case["refs"].pop("process")
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
        elif name == "delete-negative-case":
            mutated = [c for c in mutated if c["id"] != target]
        elif name == "promote-authority":
            case["outcome"] = "physical-authority"
        elif name == "change-disposition":
            case["outcome"] = "supported"
        elif name == "rewrite-historical-ref":
            case["property"] = "E9"
            case["refs"]["property"] = "property/E9"
        expect_failure(doc, mutated, name)

    graph_mutations = [
        ("rewrite-graph-edge-kind", lambda g: g["edges"][0].__setitem__("kind", "mutated-kind")),
        ("delete-graph-negative-edge", lambda g: g["edges"].remove(next(e for e in g["edges"] if e["id"] == "edge/C07"))),
    ]
    for name, mutate in graph_mutations:
        mutated_graph = json.loads(json.dumps(graph))
        mutate(mutated_graph)
        mutated_doc = json.loads(json.dumps(doc))
        mutated_doc["state_graph"] = mutated_graph
        expect_failure(mutated_doc, cases, name)

    disposition_mutated = json.loads(json.dumps(cases))
    disposition_mutated[0]["outcome"] = "recomputed-disposition"
    if replay_digest(disposition_mutated, graph) != baseline_digest:
        fail("derived disposition mutated replay identity")

    print(json.dumps({
        "qualifier": "MAT-CONVERGE-002A2",
        "schema": "1",
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "replay_digest": baseline_digest,
        "case_count": len(cases),
        "mutation_count": len(mutations) + len(graph_mutations),
        "disposition": "PASS",
        "claim_ceiling": doc["claim_ceiling"],
    }, sort_keys=True, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
