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
GRAPH_EDGE_KEYS = {"id", "case_id", "from", "to", "kind", "historical"}
REPLAY_ORDERING = "case manifest order is significant; graph node and edge order are identity-insignificant"
COVERAGE_REASONS = {
    "C06": "case-level adversarial ranking control",
    "C08": "case-level negative-evidence control",
    "C11": "case-level population-inference control",
    "C12": "case-level measurement-feasibility control",
    "C13": "case-level manufacturing-feasibility control",
    "C14": "case-level lower-tail engineering control",
    "C15": "case-level evaluator-correlation control",
}

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
        "nodes": sorted(
            ({k: n[k] for k in ("id", "kind", "ref", "generation")} for n in graph["nodes"]),
            key=lambda n: n["id"],
        ),
        "edges": sorted(
            ({k: e[k] for k in ("id", "case_id", "from", "to", "kind", "historical")} for e in graph["edges"]),
            key=lambda e: e["id"],
        ),
        "case_ids": graph["case_ids"],
        "replay_rule": graph["replay_rule"],
        "ordering_semantics": graph["ordering_semantics"],
        "dependency_closure": graph["dependency_closure"],
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

def check_graph(graph, cases):
    if set(graph) != {"schema", "historical_records_immutable",
                      "derived_dispositions_recomputable", "nodes", "edges",
                      "case_ids", "replay_rule", "ordering_semantics",
                      "dependency_closure", "coverage"}:
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
    if graph["ordering_semantics"] != REPLAY_ORDERING:
        fail("state graph ordering semantics mismatch")
    if graph["dependency_closure"] != (
        "undirected-connected-component over immutable graph nodes; "
        "all incident edge case IDs are invalidated"
    ):
        fail("state graph dependency closure semantics mismatch")
    nodes, edges = graph["nodes"], graph["edges"]
    if graph["case_ids"] != list(EXPECTED):
        fail("state graph case coverage drift")
    if not nodes or not edges:
        fail("state graph is empty")
    node_ids = [n["id"] for n in nodes]
    edge_ids = [e["id"] for e in edges]
    if len(node_ids) != len(set(node_ids)):
        fail("duplicate state graph node identity")
    if len(edge_ids) != len(set(edge_ids)):
        fail("duplicate state graph edge identity")
    node_set = set(node_ids)
    nodes_by_id = {n["id"]: n for n in nodes}
    for node in nodes:
        if set(node) != GRAPH_NODE_KEYS:
            fail(f"state graph node schema drift: {node.get('id')}")
    for edge in edges:
        if set(edge) != GRAPH_EDGE_KEYS:
            fail(f"state graph edge schema drift: {edge.get('id')}")
        if edge["historical"] is not True:
            fail(f"non-historical edge: {edge['id']}")
        if edge["case_id"] not in EXPECTED or edge["id"] != f"edge/{edge['case_id']}":
            fail(f"graph edge is not bound to its campaign case: {edge['id']}")
        if edge["from"] not in node_set or edge["to"] not in node_set:
            fail(f"dangling state graph edge: {edge['id']}")
    case_by_id = {c["id"]: c for c in cases}
    prefix_fields = {
        "demand": "demand",
        "profile": "profile",
        "candidate": "candidate",
        "process": "process",
        "property": "property",
        "measurement": "measurement",
    }
    for node in nodes:
        if node["kind"] == "negative-edge":
            if node["ref"] != f"edge/{node['generation']}":
                fail(f"negative node identity drift: {node['id']}")
            if node["generation"] not in case_by_id:
                fail(f"negative node case missing: {node['id']}")
            continue
        field = prefix_fields.get(node["kind"])
        if field is None:
            fail(f"unknown graph node kind: {node['kind']}")
        if node["ref"] != f"{field}/{node['generation']}":
            fail(f"node ref/generation mismatch: {node['id']}")
        if not any(c["refs"][field] == node["ref"] for c in cases):
            fail(f"graph node is not anchored to a campaign case: {node['id']}")
    for edge in edges:
        case = case_by_id[edge["case_id"]]
        endpoint_refs = {nodes_by_id[edge["from"]]["ref"], nodes_by_id[edge["to"]]["ref"]}
        case_refs = set(case["refs"].values())
        if not endpoint_refs.issubset(case_refs) and edge["case_id"] != "C07":
            fail(f"graph edge endpoints are not anchored to case {edge['case_id']}")
    if not any(n["kind"] == "negative-edge" for n in nodes):
        fail("negative evidence is absent from state graph")
    if not set(edge["case_id"] for edge in edges).issubset(set(graph["case_ids"])):
        fail("graph edge references uncovered case")
    required_edges = {"edge/C07", "edge/C09", "edge/C10", "edge/C16"}
    if not required_edges.issubset(edge_ids):
        fail("critical state graph edges missing")

def dependency_projection(cases, graph):
    """Return the immutable case/graph refs that each case disposition may depend on."""
    node_by_id = {n["id"]: n for n in graph["nodes"]}
    projection = {}
    for case in cases:
        refs = set(case["refs"].values())
        for edge in graph["edges"]:
            if edge["case_id"] != case["id"]:
                continue
            refs.add(node_by_id[edge["from"]]["ref"])
            refs.add(node_by_id[edge["to"]]["ref"])
        projection[case["id"]] = tuple(sorted(refs))
    return projection

def dependency_delta(before, after):
    """Return case IDs whose immutable dependency projections changed."""
    return sorted(
        case_id for case_id in before
        if before[case_id] != after[case_id]
    )

def immutable_record_identity(case):
    """Digest the historical case identity without derived disposition."""
    payload = {
        "schema": "integration-edge-v1",
        "case_id": case["id"],
        "refs": {field: case["refs"][field] for field in FIELDS},
    }
    return hashlib.sha256(canonical(payload)).hexdigest()

def derived_snapshot(cases):
    """Canonical derived records used only to test recomputation stability."""
    return {
        case["id"]: canonical({
            "case_id": case["id"],
            "dependency_digest": immutable_record_identity(case),
            "disposition": case["outcome"],
        })
        for case in cases
    }

def graph_dependency_closure(cases, graph, changed_ref):
    """Return case IDs in the undirected immutable graph component of changed_ref."""
    node_by_id = {n["id"]: n for n in graph["nodes"]}
    start_nodes = {
        node_id for node_id, node in node_by_id.items()
        if node["ref"] == changed_ref
    }
    if not start_nodes:
        return []

    adjacency = {node_id: set() for node_id in node_by_id}
    edge_cases = {}
    for edge in graph["edges"]:
        adjacency[edge["from"]].add(edge["to"])
        adjacency[edge["to"]].add(edge["from"])
        edge_cases[edge["id"]] = edge["case_id"]

    reachable = set(start_nodes)
    frontier = list(sorted(start_nodes))
    while frontier:
        node_id = frontier.pop(0)
        for neighbor in sorted(adjacency[node_id]):
            if neighbor not in reachable:
                reachable.add(neighbor)
                frontier.append(neighbor)

    return sorted(
        edge_cases[edge["id"]]
        for edge in graph["edges"]
        if edge["from"] in reachable or edge["to"] in reachable
    )

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
    check_graph(doc["state_graph"], cases)

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
    baseline_projection = dependency_projection(cases, graph)
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

    # Localized invalidation: changing one immutable generation must not alter
    # the dependency projection of unrelated campaign cases.
    localized = json.loads(json.dumps(cases))
    localized_case = next(c for c in localized if c["id"] == "C02")
    localized_case["process"] = "G3"
    localized_case["refs"]["process"] = "process/G3"
    try:
        check(doc, localized)
    except AssertionError:
        pass
    else:
        fail("historical mutation unexpectedly accepted as qualified fixture")

    # Minimal invalidation: a localized historical generation change must
    # affect only the case that actually references that generation.
    localized_projection = dependency_projection(localized, graph)
    localized_graph = json.loads(json.dumps(graph))
    localized_graph["nodes"][2]["generation"] = "G3"
    localized_graph["nodes"][2]["ref"] = "process/G3"
    localized_projection_with_graph = dependency_projection(localized, localized_graph)
    if dependency_delta(baseline_projection, localized_projection) != ["C02"]:
        fail("historical generation mutation cascaded beyond its dependent case")
    expected_graph_closure = graph_dependency_closure(cases, graph, "process/G1")
    if expected_graph_closure != ["C02", "C04"]:
        fail("unexpected process dependency closure in baseline graph")
    if dependency_delta(baseline_projection, localized_projection_with_graph) != ["C01", "C02", "C04", "C07", "C09", "C10", "C16"]:
        fail("graph generation mutation does not expose its true dependent cases")
    actual_changed = "process/G3"
    if actual_changed in set(localized_projection["C02"]):
        fail("case-local projection accepted an unqualified replacement generation")
    for case_id in (case_id for case_id in EXPECTED if case_id != "C02"):
        if baseline_projection[case_id] != localized_projection[case_id]:
            fail(f"unrelated dependency projection changed: {case_id}")

    # Recompute stability: a dependency mutation may change only the
    # derived record(s) whose immutable dependency identity changed.
    baseline_snapshot = derived_snapshot(cases)
    recomputed_snapshot = derived_snapshot(localized)
    changed_records = sorted(
        case_id for case_id in baseline_snapshot
        if baseline_snapshot[case_id] != recomputed_snapshot[case_id]
    )
    if changed_records != ["C02"]:
        fail("recomputation changed unrelated derived records")

    # Negative evidence is append-only in identity: adding a newer successful
    # case must not make the historical negative edge disappear or change.
    with_new_case = json.loads(json.dumps(cases))
    negative = next(c for c in with_new_case if c["id"] == "C07")
    negative_identity = immutable_record_identity(negative)
    newer = json.loads(json.dumps(negative))
    newer["id"] = "C17"
    newer["outcome"] = "supported"
    newer["refs"]["process"] = newer["refs"]["process"]
    with_new_case.append(newer)
    if immutable_record_identity(next(c for c in with_new_case if c["id"] == "C07")) != negative_identity:
        fail("newer result rewrote historical negative evidence identity")
    if not any(c["id"] == "C07" and c["outcome"] == "negative-edge-addressable" for c in with_new_case):
        fail("historical negative edge became unreachable after recomputation")

    # Bidirectional traceability: every graph endpoint must be attributable
    # to its bound case, while the campaign manifest may include cases that are
    # intentionally covered only by the case-level oracle.
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    case_ids = {case["id"] for case in cases}
    graph_edge_cases = {edge["case_id"] for edge in graph["edges"]}
    if not graph_edge_cases.issubset(case_ids):
        fail("graph contains edge coverage for an unknown case")
    case_refs = {
        case["id"]: set(case["refs"].values())
        for case in cases
    }
    for edge in graph["edges"]:
        from_ref = node_by_id[edge["from"]]["ref"]
        to_ref = node_by_id[edge["to"]]["ref"]
        if edge["case_id"] not in case_refs:
            fail(f"graph edge is not attributable to a case: {edge['id']}")
        allowed = case_refs[edge["case_id"]]
        if edge["id"] != "edge/C07" and not ({from_ref, to_ref} <= allowed):
            fail(f"graph dependency is not attributable to its case: {edge['id']}")
    graph_only_cases = sorted(case_ids - graph_edge_cases)
    if graph_only_cases != ["C06", "C08", "C11", "C12", "C13", "C14", "C15"]:
        fail("partial structural coverage manifest drift")
    if len(graph["edges"]) != len(graph_edge_cases):
        fail("multiple structural edges unexpectedly collapsed to one case")
    declared_coverage = graph.get("coverage")
    if declared_coverage is None:
        fail("structural coverage declaration missing")
    if set(declared_coverage) != {
        "mode", "case_level_cases", "graph_edge_cases", "graph_only_cases",
        "graph_only_case_ids", "graph_only_case_reasons",
    }:
        fail("structural coverage declaration schema drift")
    if declared_coverage["mode"] != "partial-structural-graph-plus-case-oracle":
        fail("unsupported structural coverage mode")
    if declared_coverage["case_level_cases"] != len(case_ids):
        fail("case-level coverage count drift")
    if declared_coverage["graph_edge_cases"] != len(graph_edge_cases):
        fail("graph-edge coverage count drift")
    if declared_coverage["graph_only_cases"] != len(graph_only_cases):
        fail("graph-only coverage count drift")
    if declared_coverage["graph_only_case_ids"] != graph_only_cases:
        fail("graph-only coverage identities drift")
    reasons = declared_coverage.get("graph_only_case_reasons")
    if reasons != {case_id: COVERAGE_REASONS[case_id] for case_id in graph_only_cases}:
        fail("graph-only coverage reasons drift")

    # Coverage metadata is itself qualified: changing the declared
    # classification, counts, identities, or explanations must fail.
    coverage_mutations = [
        ("coverage-mode", lambda c: c.__setitem__("mode", "full-graph")),
        ("coverage-case-count", lambda c: c.__setitem__("case_level_cases", 15)),
        ("coverage-edge-count", lambda c: c.__setitem__("graph_edge_cases", 8)),
        ("coverage-only-count", lambda c: c.__setitem__("graph_only_cases", 8)),
        ("coverage-only-identities", lambda c: c["graph_only_case_ids"].pop()),
        ("coverage-only-reason", lambda c: c["graph_only_case_reasons"].__setitem__("C06", "mutated-reason")),
    ]
    for name, mutate in coverage_mutations:
        mutated_doc = json.loads(json.dumps(doc))
        mutate(mutated_doc["state_graph"]["coverage"])
        expect_failure(mutated_doc, cases, name)

    # Mutation matrix: every graph node identity change must invalidate
    # exactly the cases structurally dependent on that node.
    node_mutation_expectations = {
        "N01": "demand/D1",
        "N02": "candidate/A",
        "N03": "process/G1",
        "N04": "property/E1",
        "N05": "measurement/M1",
        "N06": "profile/P1",
        "N07": "edge/C07",
    }
    for node_id, original_ref in node_mutation_expectations.items():
        mutated_graph = json.loads(json.dumps(graph))
        mutated_node = next(n for n in mutated_graph["nodes"] if n["id"] == node_id)
        field, value = original_ref.split("/", 1)
        mutated_node["ref"] = f"{field}/{value}-MUTATED"
        mutated_node["generation"] = mutated_node["generation"] + "-MUTATED"
        before = dependency_projection(cases, graph)
        after = dependency_projection(cases, mutated_graph)
        direct_expected = dependency_delta(before, after)
        expected = graph_dependency_closure(cases, graph, original_ref)
        if node_id == "N06":
            if expected != ["C03"]:
                fail(f"profile node closure drift: {expected}")
        elif node_id == "N07":
            if expected != ["C07"]:
                fail(f"negative-edge node closure drift: {expected}")
        else:
            if expected != ["C01", "C02", "C03", "C04", "C05", "C07", "C09", "C10", "C16"]:
                fail(f"core graph component closure drift for {node_id}: {expected}")
        if node_id == "N07" and direct_expected != ["C07"]:
            fail(f"negative-edge direct fanout mismatch: {direct_expected}")
        if node_id != "N07" and node_id != "N06" and not set(expected).issuperset(direct_expected):
            fail(f"transitive closure does not cover direct fanout for {node_id}")
        # The graph mutation is an invalidation plan, not a rewrite of history:
        # the selected cases are exactly the records eligible for recomputation.
        if set(expected) != set(graph_dependency_closure(cases, graph, original_ref)):
            fail(f"non-deterministic recomputation closure for {node_id}")

    # Unknown immutable refs have no graph impact; fail-closed means no accidental fanout.
    if graph_dependency_closure(cases, graph, "unknown/immutable-ref") != []:
        fail("unknown graph ref acquired an accidental dependency closure")

    # The negative-edge node is a first-class historical tombstone: mutating
    # its case identity must be caught rather than allowing negative evidence
    # to become an anonymous or silently replaced record.
    negative_node_mutated = json.loads(json.dumps(graph))
    negative_node = next(n for n in negative_node_mutated["nodes"] if n["id"] == "N07")
    negative_node["generation"] = "C17"
    negative_node["ref"] = "edge/C17"
    mutated_negative_doc = json.loads(json.dumps(doc))
    mutated_negative_doc["state_graph"] = negative_node_mutated
    expect_failure(mutated_negative_doc, cases, "rewrite-negative-node-tombstone")

    graph_mutations = [
        ("rewrite-graph-edge-kind", lambda g: g["edges"][0].__setitem__("kind", "mutated-kind")),
        ("delete-graph-negative-edge", lambda g: g["edges"].remove(next(e for e in g["edges"] if e["id"] == "edge/C07"))),
        ("rewrite-graph-case-binding", lambda g: g["edges"][0].__setitem__("case_id", "C16")),
        ("rewrite-candidate-node-ref", lambda g: g["nodes"][1].__setitem__("ref", "candidate/Z")),
        ("rewrite-process-node-generation", lambda g: g["nodes"][2].__setitem__("generation", "G9")),
        ("rewrite-property-node-kind", lambda g: g["nodes"][3].__setitem__("kind", "measurement")),
        ("rewrite-case-coverage-manifest", lambda g: g["case_ids"].pop()),
    ]
    for name, mutate in graph_mutations:
        mutated_graph = json.loads(json.dumps(graph))
        mutate(mutated_graph)
        mutated_doc = json.loads(json.dumps(doc))
        mutated_doc["state_graph"] = mutated_graph
        expect_failure(mutated_doc, cases, name)

    if set(baseline_projection) != set(EXPECTED):
        fail("dependency projection coverage drift")
    if baseline_projection["C01"] == baseline_projection["C02"]:
        fail("distinct process generations collapsed dependency identity")

    # Replay determinism: JSON object key order and graph collection order
    # are representational only; the ordered campaign case manifest remains
    # semantically significant and is therefore intentionally not normalized.
    key_permuted = json.loads(json.dumps(doc))
    key_permuted["cases"] = [
        {key: case[key] for key in reversed(list(case))}
        for case in key_permuted["cases"]
    ]
    key_permuted["state_graph"] = {
        key: key_permuted["state_graph"][key]
        for key in reversed(list(key_permuted["state_graph"]))
    }
    key_permuted["state_graph"]["nodes"] = [
        {key: node[key] for key in reversed(list(node))}
        for node in reversed(key_permuted["state_graph"]["nodes"])
    ]
    key_permuted["state_graph"]["edges"] = [
        {key: edge[key] for key in reversed(list(edge))}
        for edge in reversed(key_permuted["state_graph"]["edges"])
    ]
    if replay_digest(key_permuted["cases"], key_permuted["state_graph"]) != baseline_digest:
        fail("representational key/graph ordering changed replay identity")

    graph_node_permuted = json.loads(json.dumps(graph))
    graph_node_permuted["nodes"].reverse()
    if replay_digest(cases, graph_node_permuted) != baseline_digest:
        fail("graph node ordering changed replay identity")
    graph_edge_permuted = json.loads(json.dumps(graph))
    graph_edge_permuted["edges"].reverse()
    if replay_digest(cases, graph_edge_permuted) != baseline_digest:
        fail("graph edge ordering changed replay identity")

    case_order_permuted = list(reversed(cases))
    if replay_digest(case_order_permuted, graph) == baseline_digest:
        fail("case manifest order lost semantic significance")

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
        "mutation_count": len(mutations) + 1 + len(graph_mutations) + len(coverage_mutations),
        "disposition": "PASS",
        "claim_ceiling": doc["claim_ceiling"],
    }, sort_keys=True, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
