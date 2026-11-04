#!/usr/bin/env python3
"""Independent standard-library oracle for connectome-query-ensemble-audit.

This is test/evaluation code only. It deliberately does not import production
Rust code or share its graph data structures/algorithms. The reference path
calculation uses a Boolean reachability matrix and Floyd-Warshall, whereas the
production implementation uses adjacency maps and breadth-first search.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    raise AssertionError(message)


def close(actual: Any, expected: float, context: str) -> None:
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        fail(f"{context}: expected numeric value, got {actual!r}")
    if not math.isfinite(float(actual)) or not math.isclose(
        float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        fail(f"{context}: expected {expected!r}, got {actual!r}")


def graph_from_scenario(scenario: dict[str, Any]) -> dict[tuple[int, int], float]:
    graph: dict[tuple[int, int], float] = {}
    for edge in scenario["edges"]:
        source = edge["source"]
        target = edge["target"]
        weight = float(edge["weight"])
        if not isinstance(source, int) or not isinstance(target, int):
            fail(f"{scenario['scenario_id']}: edge endpoints must be integers")
        if source < 0 or target < 0 or source == target:
            fail(f"{scenario['scenario_id']}: unsupported edge endpoint {source}->{target}")
        if not math.isfinite(weight) or weight <= 0.0:
            fail(f"{scenario['scenario_id']}: edge weight must be finite and positive")
        key = (source, target)
        if key in graph:
            fail(f"{scenario['scenario_id']}: duplicate edge {source}->{target}")
        graph[key] = weight
    return graph


def graph_digest(graph: dict[tuple[int, int], float]) -> str:
    """Independent encoding of the V1 canonical graph identity."""
    digest = hashlib.sha256()
    digest.update(b"connectome-query-scenario-graph-v1")
    for (source, target), weight in sorted(graph.items()):
        digest.update(struct.pack(">Q", source))
        digest.update(struct.pack(">Q", target))
        digest.update(struct.pack(">d", weight))
    return digest.hexdigest()


def length_prefixed(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack(">Q", len(encoded)) + encoded


def scenario_set_digest(scenarios: list[dict[str, Any]]) -> tuple[str, int]:
    """Recompute the V1 scenario-set digest from canonical semantic records."""
    rows = []
    all_groups: set[str] = set()
    for scenario in scenarios:
        groups = sorted(scenario["lineage_groups"])
        rows.append((scenario["scenario_id"], scenario["graph_sha256"], groups))
        all_groups.update(groups)
    rows.sort(key=lambda row: row[0])

    digest = hashlib.sha256()
    digest.update(b"connectome-query-finite-scenario-set-v1")
    digest.update(struct.pack(">Q", len(rows)))
    for scenario_id, graph_sha, groups in rows:
        digest.update(b"SCENARIO")
        digest.update(length_prefixed(scenario_id))
        digest.update(graph_sha.encode("ascii"))
        digest.update(struct.pack(">Q", len(groups)))
        for group in groups:
            digest.update(b"LINEAGE")
            digest.update(length_prefixed(group))
    return digest.hexdigest(), len(all_groups)


def reachability_matrix(
    graph: dict[tuple[int, int], float], nodes: list[int]
) -> dict[tuple[int, int], bool]:
    """Compute transitive closure with a matrix algorithm, independent of BFS."""
    reachable = {(source, target): source == target for source in nodes for target in nodes}
    for (source, target), weight in graph.items():
        if weight > 0.0:
            reachable[(source, target)] = True
    for pivot in nodes:
        for source in nodes:
            if reachable[(source, pivot)]:
                for target in nodes:
                    if reachable[(pivot, target)]:
                        reachable[(source, target)] = True
    return reachable


def query_value(
    graph: dict[tuple[int, int], float], metric: dict[str, Any]
) -> float:
    kind = metric["kind"]
    if kind == "edge_weight":
        return graph.get((metric["source"], metric["target"]), 0.0)
    if kind == "in_degree":
        node = metric["node"]
        return math.fsum(
            weight for (source, target), weight in graph.items() if target == node
        )
    if kind == "out_degree":
        node = metric["node"]
        return math.fsum(
            weight for (source, target), weight in graph.items() if source == node
        )
    if kind == "path_exists":
        source = metric["source"]
        target = metric["target"]
        if source == target:
            return 1.0
        nodes = sorted({source, target, *(n for edge in graph for n in edge)})
        return 1.0 if reachability_matrix(graph, nodes)[(source, target)] else 0.0
    fail(f"unsupported query metric {kind!r}")


def decision_met(value: float, decision: dict[str, Any]) -> bool:
    kind = decision["kind"]
    if kind == "boolean_is":
        return (value == 1.0) is decision["value"]
    if kind == "greater_than":
        return value > decision["threshold"]
    if kind == "greater_or_equal":
        return value >= decision["threshold"]
    if kind == "less_than":
        return value < decision["threshold"]
    if kind == "less_or_equal":
        return value <= decision["threshold"]
    if kind == "between_inclusive":
        return decision["lower"] <= value <= decision["upper"]
    fail(f"unsupported decision rule {kind!r}")


def metric_name(metric: dict[str, Any]) -> str:
    return metric["kind"]


def verify_report(input_bytes: bytes, source: dict[str, Any], report: dict[str, Any]) -> None:
    if report.get("schema_version") != 1:
        fail("report schema_version must equal 1")
    if report.get("tool") != "connectome-query-ensemble-audit":
        fail("report tool identity mismatch")
    if report.get("status") != "EVALUATED_FINITE_SCENARIO_SET":
        fail("report did not declare finite-scenario evaluation")
    if report.get("probability_claim") != "NONE_FINITE_SCENARIO_SET_ONLY":
        fail("finite unweighted scenario results must not make probability claims")
    if report.get("input_sha256") != hashlib.sha256(input_bytes).hexdigest():
        fail("report input digest does not match exact input bytes")
    if report.get("dataset_revision_commit_sha") != source["dataset_revision_commit_sha"]:
        fail("report dataset revision does not match input")
    if report.get("scope_coverage") != source["scope_coverage"]:
        fail("report scope_coverage does not match input")

    scenarios = source["scenarios"]
    scenario_by_id: dict[str, dict[str, Any]] = {}
    graphs: dict[str, dict[tuple[int, int], float]] = {}
    graph_hashes: dict[str, str] = {}
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        if scenario_id in scenario_by_id:
            fail(f"duplicate scenario ID {scenario_id!r}")
        graph = graph_from_scenario(scenario)
        actual_graph_hash = graph_digest(graph)
        if actual_graph_hash != scenario["graph_sha256"]:
            fail(f"{scenario_id}: declared graph digest is not canonical")
        scenario_by_id[scenario_id] = scenario
        graphs[scenario_id] = graph
        graph_hashes[scenario_id] = actual_graph_hash

    if report.get("scenario_count") != len(scenarios):
        fail("report scenario_count does not match input")
    expected_set_hash, expected_lineage_count = scenario_set_digest(scenarios)
    if report.get("scenario_set_sha256") != expected_set_hash:
        fail("report scenario-set digest mismatch")
    if report.get("unique_lineage_group_count") != expected_lineage_count:
        fail("report lineage-group count mismatch")

    reported_queries = report.get("queries")
    if not isinstance(reported_queries, list) or len(reported_queries) != len(source["queries"]):
        fail("report query list does not match input query count")
    report_by_query = {item.get("query_id"): item for item in reported_queries}
    if len(report_by_query) != len(reported_queries):
        fail("report contains duplicate query IDs")

    for query in source["queries"]:
        query_id = query["query_id"]
        actual = report_by_query.get(query_id)
        if actual is None:
            fail(f"report omits query {query_id!r}")
        if actual.get("metric") != metric_name(query["metric"]):
            fail(f"{query_id}: metric identity mismatch")

        expected_rows = []
        for scenario_id in sorted(scenario_by_id):
            value = query_value(graphs[scenario_id], query["metric"])
            expected_rows.append(
                {
                    "scenario_id": scenario_id,
                    "graph_sha256": graph_hashes[scenario_id],
                    "value": value,
                    "decision_met": decision_met(value, query["decision"]),
                }
            )

        actual_rows = actual.get("scenario_results")
        if not isinstance(actual_rows, list) or len(actual_rows) != len(expected_rows):
            fail(f"{query_id}: report scenario-result cardinality mismatch")
        actual_by_id = {row.get("scenario_id"): row for row in actual_rows}
        if len(actual_by_id) != len(actual_rows):
            fail(f"{query_id}: duplicate scenario result")
        if set(actual_by_id) != set(scenario_by_id):
            fail(f"{query_id}: report scenario membership differs from input")

        for expected in expected_rows:
            observed = actual_by_id[expected["scenario_id"]]
            context = f"{query_id}/{expected['scenario_id']}"
            if observed.get("graph_sha256") != expected["graph_sha256"]:
                fail(f"{context}: graph digest mismatch")
            close(observed.get("value"), expected["value"], f"{context}/value")
            if observed.get("decision_met") is not expected["decision_met"]:
                fail(f"{context}: decision result mismatch")

        values = [row["value"] for row in expected_rows]
        met = sum(1 for row in expected_rows if row["decision_met"])
        not_met = len(expected_rows) - met
        expected_status = (
            "ROBUST_DECISION_MET"
            if met == len(expected_rows)
            else "ROBUST_DECISION_NOT_MET"
            if not_met == len(expected_rows)
            else "SENSITIVE_TO_RECONSTRUCTION"
        )
        if actual.get("scenarios_meeting_decision") != met:
            fail(f"{query_id}: met-count mismatch")
        if actual.get("scenarios_not_meeting_decision") != not_met:
            fail(f"{query_id}: not-met-count mismatch")
        if actual.get("decision_status") != expected_status:
            fail(f"{query_id}: decision status mismatch")
        if actual.get("scenario_count") != len(expected_rows):
            fail(f"{query_id}: scenario_count mismatch")
        close(actual.get("minimum_value"), min(values), f"{query_id}/minimum_value")
        close(actual.get("maximum_value"), max(values), f"{query_id}/maximum_value")
        close(actual.get("range_width"), max(values) - min(values), f"{query_id}/range_width")


def exhaustive_small_graph_checks() -> dict[str, int]:
    """Exhaust all 4-node loop-free directed graph topologies (2^12 cases)."""
    nodes = [0, 1, 2, 3]
    possible_edges = [(source, target) for source in nodes for target in nodes if source != target]
    weights = [0.5 + (index + 1) / 8.0 for index in range(len(possible_edges))]
    query_checks = 0

    for mask in range(1 << len(possible_edges)):
        graph = {
            edge: weights[index]
            for index, edge in enumerate(possible_edges)
            if mask & (1 << index)
        }
        closure = reachability_matrix(graph, nodes)

        for source, target in possible_edges:
            expected = graph.get((source, target), 0.0)
            observed = query_value(graph, {"kind": "edge_weight", "source": source, "target": target})
            close(observed, expected, f"exhaustive edge_weight/{mask}/{source}/{target}")
            query_checks += 1

        for node in nodes:
            expected_in = math.fsum(weight for (_, target), weight in graph.items() if target == node)
            expected_out = math.fsum(weight for (source, _), weight in graph.items() if source == node)
            close(
                query_value(graph, {"kind": "in_degree", "node": node}),
                expected_in,
                f"exhaustive in_degree/{mask}/{node}",
            )
            close(
                query_value(graph, {"kind": "out_degree", "node": node}),
                expected_out,
                f"exhaustive out_degree/{mask}/{node}",
            )
            query_checks += 2

        for source in nodes:
            for target in nodes:
                observed = query_value(graph, {"kind": "path_exists", "source": source, "target": target})
                expected = 1.0 if closure[(source, target)] else 0.0
                close(observed, expected, f"exhaustive path_exists/{mask}/{source}/{target}")
                query_checks += 1

    # Metamorphic controls: relabeling, input ordering and edge reversal.
    probe = {
        (0, 1): 1.25,
        (1, 2): 0.75,
        (2, 0): 1.5,
        (2, 3): 2.0,
    }
    permutation = {0: 2, 1: 0, 2: 3, 3: 1}
    relabeled = {
        (permutation[source], permutation[target]): weight
        for (source, target), weight in probe.items()
    }
    for source, target in itertools.product(nodes, repeat=2):
        original = query_value(probe, {"kind": "path_exists", "source": source, "target": target})
        renamed = query_value(
            relabeled,
            {
                "kind": "path_exists",
                "source": permutation[source],
                "target": permutation[target],
            },
        )
        close(renamed, original, f"node-permutation path {source}->{target}")
        query_checks += 1

    for node in nodes:
        original_in = query_value(probe, {"kind": "in_degree", "node": node})
        original_out = query_value(probe, {"kind": "out_degree", "node": node})
        reversed_graph = {(target, source): weight for (source, target), weight in probe.items()}
        close(
            query_value(reversed_graph, {"kind": "out_degree", "node": node}),
            original_in,
            f"edge-reversal out-degree/{node}",
        )
        close(
            query_value(reversed_graph, {"kind": "in_degree", "node": node}),
            original_out,
            f"edge-reversal in-degree/{node}",
        )
        query_checks += 2

    # Exact threshold boundaries protect inclusive/exclusive decision semantics.
    boundary = 1.25
    controls = [
        ({"kind": "greater_than", "threshold": boundary}, False),
        ({"kind": "greater_or_equal", "threshold": boundary}, True),
        ({"kind": "less_than", "threshold": boundary}, False),
        ({"kind": "less_or_equal", "threshold": boundary}, True),
        ({"kind": "between_inclusive", "lower": boundary, "upper": boundary}, True),
    ]
    for decision, expected in controls:
        if decision_met(boundary, decision) is not expected:
            fail(f"decision boundary semantics failed for {decision!r}")
        query_checks += 1

    return {
        "directed_graph_topologies": 1 << len(possible_edges),
        "possible_directed_edges_per_graph": len(possible_edges),
        "differential_query_checks": query_checks,
        "metamorphic_and_boundary_checks": 4 * 4 + 4 * 2 + len(controls),
    }


def oracle_negative_controls(
    input_bytes: bytes, source: dict[str, Any], report: dict[str, Any]
) -> int:
    mutations = 0

    altered_value = json.loads(json.dumps(report))
    first_result = altered_value["queries"][0]["scenario_results"][0]
    first_result["value"] = float(first_result["value"]) + 0.125
    try:
        verify_report(input_bytes, source, altered_value)
    except AssertionError:
        mutations += 1
    else:
        fail("oracle did not detect a deliberately altered query value")

    altered_digest = json.loads(json.dumps(report))
    altered_digest["scenario_set_sha256"] = "0" * 64
    try:
        verify_report(input_bytes, source, altered_digest)
    except AssertionError:
        mutations += 1
    else:
        fail("oracle did not detect a deliberately altered scenario-set digest")

    altered_membership = json.loads(json.dumps(report))
    altered_membership["queries"][0]["scenario_results"].pop()
    try:
        verify_report(input_bytes, source, altered_membership)
    except AssertionError:
        mutations += 1
    else:
        fail("oracle did not detect a missing scenario result")

    return mutations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="exact JSON input given to the Rust evaluator")
    parser.add_argument("--report", required=True, type=Path, help="JSON report emitted by the Rust evaluator")
    parser.add_argument("--exhaustive", action="store_true", help="run exhaustive 4-node graph and metamorphic checks")
    args = parser.parse_args()

    input_bytes = args.input.read_bytes()
    source = json.loads(input_bytes)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    verify_report(input_bytes, source, report)
    negative_controls = oracle_negative_controls(input_bytes, source, report)

    summary: dict[str, Any] = {
        "status": "INDEPENDENT_ORACLE_AGREEMENT",
        "oracle": "python-standard-library-floyd-warshall-v1",
        "production_report": str(args.report),
        "scenario_count": len(source["scenarios"]),
        "query_count": len(source["queries"]),
        "negative_control_mutations_detected": negative_controls,
        "probability_claim": "NONE",
    }
    if args.exhaustive:
        summary["exhaustive"] = exhaustive_small_graph_checks()
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"independent query oracle failed: {error}", file=sys.stderr)
        raise SystemExit(1)
