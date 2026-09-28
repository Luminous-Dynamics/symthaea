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
FAILURE_DIAGNOSTICS = []
REQUIRED_FAILURE_CATEGORIES = {
    "authority", "negative-evidence", "coverage",
    "historical-identity", "dependency-boundary", "schema-integrity",
}

def record_failure(doc, cases, label, expected_category):
    try:
        check(doc, cases)
    except AssertionError as exc:
        actual_category = failure_category(str(exc))
        if actual_category != expected_category:
            fail(
                f"mutation category drift: {label}: "
                f"expected {expected_category}, got {actual_category}"
            )
        FAILURE_DIAGNOSTICS.append({
            "mutation": label,
            "category": actual_category,
        })
        return actual_category
    fail(f"mutation escaped oracle: {label}")

def failure_category(message):
    """Normalize oracle failures into stable diagnostic classes."""
    text = message.lower()
    if any(token in text for token in ("authority", "physical-authority", "advisory")):
        return "authority"
    if any(token in text for token in ("negative", "tombstone")):
        return "negative-evidence"
    if any(token in text for token in ("coverage", "case coverage", "graph coverage")):
        return "coverage"
    if any(token in text for token in ("historical", "immutable", "generation", "ref")):
        return "historical-identity"
    if any(token in text for token in ("dependency", "closure", "fanout")):
        return "dependency-boundary"
    if any(token in text for token in ("schema", "ordering")):
        return "schema-integrity"
    return "invariant-integrity"

EXPECTED_CATEGORIES = {\n    "remove-process-ref": "historical-identity",\n    "change-process-generation": "historical-identity",\n    "change-profile-ref": "historical-identity",\n    "change-evaluator-generation": "historical-identity",\n    "change-measurement-generation": "historical-identity",\n    "change-ranking-only": "schema-integrity",\n    "delete-negative-case": "negative-evidence",\n    "promote-authority": "authority",\n    "change-disposition": "invariant-integrity",\n    "rewrite-historical-ref": "historical-identity",\n    "rewrite-negative-node-tombstone": "negative-evidence",\n}\n
    try:
        check(doc, cases)
    except AssertionError as exc:
        category = failure_category(str(exc))
        FAILURE_DIAGNOSTICS.append({"mutation": label, "category": category})
        return category
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
        record_failure(doc, mutated, name, EXPECTED_CATEGORIES[name])

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
    if expected_graph_closure != ["C01", "C02", "C03", "C04", "C05", "C07", "C09", "C10", "C16"]:
        fail("unexpected transitive process dependency closure in baseline graph")
    direct_graph_cases = graph_direct_case_ids(cases, graph, "process/G1")
    if direct_graph_cases != ["C02", "C04"]:
        fail("unexpected direct process dependency fanout")
    if not set(direct_graph_cases).issubset(expected_graph_closure):
        fail("transitive closure dropped a directly dependent graph case")
    plan = graph_invalidation_plan(cases, graph, "process/G1")
    if plan["direct_case_ids"] != direct_graph_cases:
        fail("invalidation plan direct fanout drift")
    if plan["transitive_case_ids"] != expected_graph_closure:
        fail("invalidation plan transitive fanout drift")
    if plan["recompute_case_ids"] != expected_graph_closure:
        fail("invalidation plan recomputation set drift")
    canonical_plan = canonical_invalidation_plan(cases, graph, "process/G1")
    if canonical_plan["schema"] != "mat-converge-002-invalidation-plan-v1":
        fail("invalidation plan schema drift")
    if canonical_plan["direct_case_ids"] != direct_graph_cases:
        fail("canonical invalidation plan direct fanout drift")
    if canonical_plan["transitive_case_ids"] != expected_graph_closure:
        fail("canonical invalidation plan closure drift")
    if canonical_plan["recompute_case_ids"] != expected_graph_closure:
        fail("canonical invalidation plan recomputation drift")
    plan_payload = {
        "schema": canonical_plan["schema"],
        "graph_schema": graph["schema"],
        "changed_ref": canonical_plan["changed_ref"],
        "direct_case_ids": canonical_plan["direct_case_ids"],
        "transitive_case_ids": canonical_plan["transitive_case_ids"],
        "recompute_case_ids": canonical_plan["recompute_case_ids"],
    }
    if canonical_plan["digest"] != hashlib.sha256(canonical(plan_payload)).hexdigest():
        fail("invalidation plan digest is not self-consistent")
    key_permuted_plan = json.loads(json.dumps(canonical_plan))
    key_permuted_plan["recompute_case_ids"] = list(reversed(key_permuted_plan["recompute_case_ids"]))
    permuted_payload = {
        "schema": key_permuted_plan["schema"],
        "graph_schema": graph["schema"],
        "changed_ref": key_permuted_plan["changed_ref"],
        "direct_case_ids": key_permuted_plan["direct_case_ids"],
        "transitive_case_ids": key_permuted_plan["transitive_case_ids"],
        "recompute_case_ids": key_permuted_plan["recompute_case_ids"],
    }
    permuted_digest = hashlib.sha256(canonical(permuted_payload)).hexdigest()
    if permuted_digest == canonical_plan["digest"]:
        fail("recomputation ordering was incorrectly treated as representational")

    # Two-phase replay artifact: the plan fixes the recomputation boundary,
    # while the snapshot binds only the immutable historical identities selected
    # by that plan.
    replay_snapshot = recomputation_snapshot(cases, graph, canonical_plan)
    if replay_snapshot["schema"] != "mat-converge-002-recomputation-snapshot-v1":
        fail("recomputation snapshot schema drift")
    if replay_snapshot["invalidation_plan_digest"] != canonical_plan["digest"]:
        fail("recomputation snapshot lost plan binding")
    if [record["case_id"] for record in replay_snapshot["records"]] != expected_graph_closure:
        fail("recomputation snapshot selected the wrong cases")
    snapshot_payload = {
        "schema": replay_snapshot["schema"],
        "invalidation_plan_digest": replay_snapshot["invalidation_plan_digest"],
        "records": replay_snapshot["records"],
    }
    if replay_snapshot["digest"] != hashlib.sha256(canonical(snapshot_payload)).hexdigest():
        fail("recomputation snapshot digest is not self-consistent")
    validate_recomputation_snapshot(cases, graph, canonical_plan, replay_snapshot)

    # Replay session binds the immutable base, the exact invalidation boundary,
    # the recomputation snapshot, and the derived result produced by recomputation.
    session_baseline = replay_session(cases, graph, canonical_plan, cases)
    if session_baseline["schema"] != "mat-converge-002-replay-session-v1":
        fail("replay session schema drift")
    if session_baseline["base_replay_digest"] != baseline_digest:
        fail("replay session lost base replay binding")
    if session_baseline["invalidation_plan_digest"] != canonical_plan["digest"]:
        fail("replay session lost invalidation-plan binding")
    if session_baseline["recomputation_snapshot_digest"] != replay_snapshot["digest"]:
        fail("replay session lost recomputation binding")
    session_payload = {
        "schema": session_baseline["schema"],
        "base_replay_digest": session_baseline["base_replay_digest"],
        "invalidation_plan_digest": session_baseline["invalidation_plan_digest"],
        "recomputation_snapshot_digest": session_baseline["recomputation_snapshot_digest"],
        "recomputed_result_digest": session_baseline["recomputed_result_digest"],
    }
    if session_baseline["digest"] != hashlib.sha256(canonical(session_payload)).hexdigest():
        fail("replay session digest is not self-consistent")
    validate_replay_session(cases, graph, canonical_plan, cases, session_baseline)

    # Cross-epoch composition is forbidden: a session must not accept a result
    # produced under a different invalidation boundary.
    alternate_plan = canonical_invalidation_plan(cases, graph, "profile/P1")
    alternate_cases = json.loads(json.dumps(cases))
    alternate_cases[2]["outcome"] = "alternate-recomputed-disposition"
    alternate_session = replay_session(
        cases, graph, alternate_plan, alternate_cases
    )
    mixed_session = json.loads(json.dumps(session_baseline))
    mixed_session["invalidation_plan_digest"] = alternate_session["invalidation_plan_digest"]
    mixed_session["recomputed_result_digest"] = alternate_session["recomputed_result_digest"]
    mixed_session["digest"] = hashlib.sha256(canonical({
        "schema": mixed_session["schema"],
        "base_replay_digest": mixed_session["base_replay_digest"],
        "invalidation_plan_digest": mixed_session["invalidation_plan_digest"],
        "recomputation_snapshot_digest": mixed_session["recomputation_snapshot_digest"],
        "recomputed_result_digest": mixed_session["recomputed_result_digest"],
    })).hexdigest()
    try:
        validate_replay_session(
            cases, graph, canonical_plan, alternate_cases, mixed_session
        )
    except AssertionError:
        pass
    else:
        fail("cross-epoch replay artifacts were composable")

    # Replay-session tamper matrix: every independently addressable artifact
    # binding is mandatory. Recomputing the outer digest must not make a splice valid.
    session_fields = (
        "base_replay_digest",
        "invalidation_plan_digest",
        "recomputation_snapshot_digest",
        "recomputed_result_digest",
    )
    for field in session_fields:
        tampered = json.loads(json.dumps(session_baseline))
        tampered[field] = "f" * 64
        tampered["digest"] = hashlib.sha256(canonical({
            "schema": tampered["schema"],
            "base_replay_digest": tampered["base_replay_digest"],
            "invalidation_plan_digest": tampered["invalidation_plan_digest"],
            "recomputation_snapshot_digest": tampered["recomputation_snapshot_digest"],
            "recomputed_result_digest": tampered["recomputed_result_digest"],
        })).hexdigest()
        try:
            validate_replay_session(
                cases, graph, canonical_plan, cases, tampered
            )
        except AssertionError:
            pass
        else:
            fail(f"tampered replay-session binding escaped oracle: {field}")

    # Session identity must also move when the immutable graph epoch changes.
    graph_epoch = json.loads(json.dumps(graph))
    graph_epoch["schema"] = "mat-converge-002-state-graph-v2"
    graph_epoch_plan = canonical_invalidation_plan(cases, graph_epoch, "process/G1")
    if graph_epoch_plan["digest"] == canonical_plan["digest"]:
        fail("graph schema epoch change did not alter invalidation identity")
    graph_epoch_session = replay_session(
        cases, graph_epoch, graph_epoch_plan, cases
    )
    if graph_epoch_session["base_replay_digest"] == session_baseline["base_replay_digest"]:
        fail("graph schema epoch change did not alter replay identity")
    if graph_epoch_session["digest"] == session_baseline["digest"]:
        fail("graph schema epoch change did not alter session identity")

    # Replay generation is observational: producing a session must not mutate
    # the historical case manifest or immutable graph.
    cases_before_session = json.loads(json.dumps(cases))
    graph_before_session = json.loads(json.dumps(graph))
    replay_session(cases, graph, canonical_plan, cases)
    if cases != cases_before_session or graph != graph_before_session:
        fail("replay-session generation mutated historical inputs")

    # A derived-only recomputation changes the result identity, while the
    # immutable replay inputs and recomputation boundary remain unchanged.
    session_recomputed_cases = json.loads(json.dumps(cases))
    for case in session_recomputed_cases:
        if case["id"] in set(expected_graph_closure):
            case["outcome"] = "recomputed-disposition"
    session_recomputed = replay_session(
        cases, graph, canonical_plan, session_recomputed_cases
    )
    if session_recomputed["base_replay_digest"] != session_baseline["base_replay_digest"]:
        fail("derived recomputation changed base replay identity")
    if session_recomputed["invalidation_plan_digest"] != session_baseline["invalidation_plan_digest"]:
        fail("derived recomputation changed invalidation identity")
    if session_recomputed["recomputation_snapshot_digest"] != session_baseline["recomputation_snapshot_digest"]:
        fail("derived recomputation changed historical recomputation identity")
    if session_recomputed["recomputed_result_digest"] == session_baseline["recomputed_result_digest"]:
        fail("derived recomputation failed to change result identity")
    if session_recomputed["digest"] == session_baseline["digest"]:
        fail("replay session failed to distinguish recomputed result")

    # Representation-equivalent replay must survive an independently rebuilt
    # graph object, not merely a reordered view of the original object.
    rebuilt_graph = {
        "schema": graph["schema"],
        "historical_records_immutable": graph["historical_records_immutable"],
        "derived_dispositions_recomputable": graph["derived_dispositions_recomputable"],
        "nodes": [
            {"id": node["id"], "kind": node["kind"], "ref": node["ref"],
             "generation": node["generation"]}
            for node in sorted(graph["nodes"], key=lambda node: node["id"], reverse=True)
        ],
        "edges": [
            {"id": edge["id"], "case_id": edge["case_id"], "from": edge["from"],
             "to": edge["to"], "kind": edge["kind"], "historical": edge["historical"]}
            for edge in sorted(graph["edges"], key=lambda edge: edge["id"], reverse=True)
        ],
        "case_ids": list(graph["case_ids"]),
        "replay_rule": graph["replay_rule"],
        "ordering_semantics": graph["ordering_semantics"],
        "dependency_closure": graph["dependency_closure"],
        "coverage": json.loads(json.dumps(graph["coverage"])),
    }
    rebuilt_plan = canonical_invalidation_plan(cases, rebuilt_graph, "process/G1")
    rebuilt_snapshot = recomputation_snapshot(cases, rebuilt_graph, rebuilt_plan)
    if graph_identity(rebuilt_graph) != graph_identity(graph):
        fail("independently rebuilt graph changed graph identity")
    if replay_digest(cases, rebuilt_graph) != baseline_digest:
        fail("independently rebuilt graph changed replay identity")
    if rebuilt_plan["digest"] != canonical_plan["digest"]:
        fail("independently rebuilt graph changed invalidation-plan identity")
    if rebuilt_snapshot["digest"] != replay_snapshot["digest"]:
        fail("independently rebuilt graph changed recomputation identity")

    # Tampering with any plan boundary or its binding must be detected before
    # a recomputation snapshot can be treated as replayable.
    for field in ("direct_case_ids", "transitive_case_ids", "recompute_case_ids"):
        tampered_plan = json.loads(json.dumps(canonical_plan))
        tampered_plan[field] = []
        try:
            validate_recomputation_snapshot(cases, graph, tampered_plan, replay_snapshot)
        except AssertionError:
            pass
        else:
            fail(f"tampered invalidation plan escaped binding oracle: {field}")

    tampered_digest_plan = json.loads(json.dumps(canonical_plan))
    tampered_digest_plan["digest"] = "0" * 64
    try:
        validate_recomputation_snapshot(cases, graph, tampered_digest_plan, replay_snapshot)
    except AssertionError:
        pass
    else:
        fail("tampered invalidation plan digest escaped binding oracle")

    # Derived dispositions are intentionally excluded from the snapshot identity.
    # The case manifest order is semantic: changing it changes the canonical
    # recomputation record order and therefore its identity.
    manifest_reordered_snapshot = recomputation_snapshot(
        case_order_permuted, graph, canonical_invalidation_plan(case_order_permuted, graph, "process/G1")
    )
    if manifest_reordered_snapshot["digest"] == replay_snapshot["digest"]:
        fail("case manifest reordering lost recomputation identity")

    disposition_mutated = json.loads(json.dumps(cases))
    for case in disposition_mutated:
        if case["id"] in set(expected_graph_closure):
            case["outcome"] = "recomputed-disposition"
    mutated_snapshot = recomputation_snapshot(disposition_mutated, graph, canonical_plan)
    if mutated_snapshot["digest"] != replay_snapshot["digest"]:
        fail("derived disposition mutated replay snapshot identity")

    # Historical identity mutation must alter the replay artifact.
    historical_mutated = json.loads(json.dumps(cases))
    historical_case = next(c for c in historical_mutated if c["id"] == "C02")
    historical_case["process"] = "G99"
    historical_case["refs"]["process"] = "process/G99"
    historical_snapshot = recomputation_snapshot(historical_mutated, graph, canonical_plan)
    if historical_snapshot["digest"] == replay_snapshot["digest"]:
        fail("historical dependency mutation did not alter replay snapshot identity")
    if graph_invalidation_plan(cases, graph, "unknown/immutable-ref") != {
        "changed_ref": "unknown/immutable-ref",
        "direct_case_ids": [],
        "transitive_case_ids": [],
        "recompute_case_ids": [],
    }:
        fail("unknown graph ref did not fail closed in invalidation plan")
    unknown_plan = canonical_invalidation_plan(cases, graph, "unknown/immutable-ref")
    if unknown_plan["direct_case_ids"] != [] or unknown_plan["recompute_case_ids"] != []:
        fail("unknown canonical invalidation plan did not fail closed")
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
        record_failure(mutated_doc, cases, name, EXPECTED_CATEGORIES[name])

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
        direct_graph_expected = graph_direct_case_ids(cases, graph, original_ref)
        if direct_graph_expected and not set(direct_graph_expected).issubset(expected):
            fail(f"transitive closure dropped direct graph fanout for {node_id}")
        if node_id == "N07" and direct_expected != ["C07"]:
            fail(f"negative-edge direct fanout mismatch: {direct_expected}")
        # The graph mutation is an invalidation plan, not a rewrite of history:
        # the selected cases are exactly the records eligible for recomputation.
        if set(expected) != set(graph_dependency_closure(cases, graph, original_ref)):
            fail(f"non-deterministic recomputation closure for {node_id}")

    # Unknown immutable refs have no graph impact; fail-closed means no accidental fanout.
    if graph_dependency_closure(cases, graph, "unknown/immutable-ref") != []:
        fail("unknown graph ref acquired an accidental dependency closure")
    negative_plan = graph_invalidation_plan(cases, graph, "edge/C07")
    if negative_plan["direct_case_ids"] != ["C07"] or negative_plan["recompute_case_ids"] != ["C07"]:
        fail("negative-edge tombstone acquired an unrelated invalidation fanout")

    # The negative-edge node is a first-class historical tombstone: mutating
    # its case identity must be caught rather than allowing negative evidence
    # to become an anonymous or silently replaced record.
    negative_node_mutated = json.loads(json.dumps(graph))
    negative_node = next(n for n in negative_node_mutated["nodes"] if n["id"] == "N07")
    negative_node["generation"] = "C17"
    negative_node["ref"] = "edge/C17"
    mutated_negative_doc = json.loads(json.dumps(doc))
    mutated_negative_doc["state_graph"] = negative_node_mutated
    record_failure(mutated_negative_doc, cases, "rewrite-negative-node-tombstone", EXPECTED_CATEGORIES["rewrite-negative-node-tombstone"])

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
        record_failure(mutated_doc, cases, name, EXPECTED_CATEGORIES[name])

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

    observed_categories = {item["category"] for item in FAILURE_DIAGNOSTICS}
    missing_categories = sorted(REQUIRED_FAILURE_CATEGORIES - observed_categories)
    if missing_categories:
        fail("adversarial coverage contract missing categories: " + ",".join(missing_categories))

    print(json.dumps({
        "qualifier": "MAT-CONVERGE-002A2",
        "schema": "1",
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "replay_digest": baseline_digest,
        "case_count": len(cases),
        "mutation_count": len(FAILURE_DIAGNOSTICS),
        "mutation_manifest_schema": "mat-converge-002-mutation-manifest-v1",
        "mutation_manifest_digest": mutation_manifest_digest(),
        "failure_categories": {
            category: sum(1 for item in FAILURE_DIAGNOSTICS if item["category"] == category)
            for category in sorted({item["category"] for item in FAILURE_DIAGNOSTICS})
        },
        "mutation_diagnostics": FAILURE_DIAGNOSTICS,
        "disposition": "PASS",
        "claim_ceiling": doc["claim_ceiling"],
    }, sort_keys=True, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
