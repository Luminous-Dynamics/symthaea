#!/usr/bin/env python3
"""Independent standard-library oracle for the query-flipping witness audit.

The oracle intentionally uses Floyd-Warshall reachability and Python combinations,
not the Rust implementation's queue-based reachability or bit-mask subset search.
It recomputes every admissible subset in the declared finite edit universe.
"""

from __future__ import annotations

import argparse
import copy
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


def json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def close(actual: Any, expected: float, context: str) -> None:
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        fail(f"{context}: expected numeric value, got {actual!r}")
    if not math.isfinite(float(actual)) or not math.isclose(
        float(actual), expected, rel_tol=1e-12, abs_tol=1e-12
    ):
        fail(f"{context}: expected {expected!r}, got {actual!r}")


def graph_from_edges(edges: list[dict[str, Any]]) -> dict[tuple[int, int], float]:
    graph: dict[tuple[int, int], float] = {}
    for edge in edges:
        source, target, weight = edge["source"], edge["target"], float(edge["weight"])
        if source == target or not math.isfinite(weight) or weight <= 0.0:
            fail(f"invalid baseline edge {source}->{target}")
        key = (source, target)
        if key in graph:
            fail(f"duplicate baseline edge {source}->{target}")
        graph[key] = weight
    return graph


def graph_digest(graph: dict[tuple[int, int], float]) -> str:
    digest = hashlib.sha256()
    digest.update(b"connectome-query-scenario-graph-v1")
    for (source, target), weight in sorted(graph.items()):
        digest.update(struct.pack(">Q", source))
        digest.update(struct.pack(">Q", target))
        digest.update(struct.pack(">d", weight))
    return digest.hexdigest()


def canonical_query(query: dict[str, Any]) -> dict[str, Any]:
    metric = query["metric"]
    kind = metric["kind"]
    if kind in ("edge_weight", "path_exists"):
        canonical_metric = {
            "kind": kind,
            "source": metric["source"],
            "target": metric["target"],
        }
    elif kind in ("in_degree", "out_degree"):
        canonical_metric = {"kind": kind, "node": metric["node"]}
    else:
        fail(f"unsupported metric {kind!r}")
    decision = query["decision"]
    decision_kind = decision["kind"]
    if decision_kind == "boolean_is":
        canonical_decision = {"kind": decision_kind, "value": decision["value"]}
    elif decision_kind in ("greater_than", "greater_or_equal", "less_than", "less_or_equal"):
        canonical_decision = {"kind": decision_kind, "threshold": decision["threshold"]}
    elif decision_kind == "between_inclusive":
        canonical_decision = {
            "kind": decision_kind,
            "lower": decision["lower"],
            "upper": decision["upper"],
        }
    else:
        fail(f"unsupported decision rule {decision_kind!r}")
    return {
        "query_id": query["query_id"],
        "metric": canonical_metric,
        "decision": canonical_decision,
    }


def canonical_edit_universe(edits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for edit in edits:
        changes = sorted(
            edit["changes"], key=lambda change: (change["source"], change["target"])
        )
        rows.append(
            {
                "edit_id": edit["edit_id"],
                "kind": edit["kind"],
                "effort_units": edit["effort_units"],
                "evidence_sha256": edit["evidence_sha256"],
                "requires": sorted(edit["requires"]),
                "conflicts_with": sorted(edit["conflicts_with"]),
                "changes": [
                    {
                        "source": change["source"],
                        "target": change["target"],
                        "expected_weight": change["expected_weight"],
                        "replacement_weight": change["replacement_weight"],
                    }
                    for change in changes
                ],
            }
        )
    return sorted(rows, key=lambda row: row["edit_id"])


def reachability_matrix(
    graph: dict[tuple[int, int], float], nodes: list[int]
) -> dict[tuple[int, int], bool]:
    reachable = {
        (source, target): source == target
        for source in nodes
        for target in nodes
    }
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


def query_value(graph: dict[tuple[int, int], float], metric: dict[str, Any]) -> float:
    kind = metric["kind"]
    if kind == "edge_weight":
        return graph.get((metric["source"], metric["target"]), 0.0)
    if kind == "in_degree":
        return math.fsum(
            weight for (_, target), weight in graph.items() if target == metric["node"]
        )
    if kind == "out_degree":
        return math.fsum(
            weight for (source, _), weight in graph.items() if source == metric["node"]
        )
    if kind == "path_exists":
        source, target = metric["source"], metric["target"]
        nodes = sorted({source, target, *(node for edge in graph for node in edge)})
        closure = reachability_matrix(graph, nodes)
        return 1.0 if closure[(source, target)] else 0.0
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


def apply_subset(
    baseline: dict[tuple[int, int], float], selected: tuple[dict[str, Any], ...]
) -> dict[tuple[int, int], float]:
    graph = dict(baseline)
    touched: set[tuple[int, int]] = set()
    for edit in selected:
        if edit["kind"] == "evidence_gate":
            continue
        for change in edit["changes"]:
            key = (change["source"], change["target"])
            if key in touched:
                fail(f"admissible subset double-mutated edge {key}")
            touched.add(key)
            if change["replacement_weight"] == 0.0:
                graph.pop(key, None)
            else:
                graph[key] = float(change["replacement_weight"])
    return graph


def expected_enumeration(source: dict[str, Any]) -> dict[str, Any]:
    edits = source["edits"]
    if not (1 <= len(edits) <= 20):
        fail("oracle supports between 1 and 20 edit atoms")
    indices = {edit["edit_id"]: index for index, edit in enumerate(edits)}
    if len(indices) != len(edits):
        fail("duplicate edit IDs")
    by_id = {edit["edit_id"]: edit for edit in edits}
    for edit in edits:
        for required in edit["requires"]:
            if required not in by_id:
                fail(f"unknown requirement {required!r}")
        for conflict in edit["conflicts_with"]:
            if conflict not in by_id:
                fail(f"unknown conflict {conflict!r}")
            if edit["edit_id"] not in by_id[conflict]["conflicts_with"]:
                fail("conflict declarations are not symmetric")

    baseline = graph_from_edges(source["baseline_graph"])
    if graph_digest(baseline) != source["baseline_graph_sha256"]:
        fail("baseline graph digest mismatch")
    query = source["query"]
    base_value = query_value(baseline, query["metric"])
    base_decision = decision_met(base_value, query["decision"])
    budget = source["max_total_effort_units"]

    budget_rejected = 0
    constraint_rejected = 0
    admissible = 0
    flipping = 0
    best_cost: int | None = None
    minimal: list[dict[str, Any]] = []
    subset_count = (1 << len(edits)) - 1

    # itertools.combinations deliberately differs from the production bit-mask search.
    for size in range(1, len(edits) + 1):
        for selected in itertools.combinations(edits, size):
            effort = sum(edit["effort_units"] for edit in selected)
            selected_ids = {edit["edit_id"] for edit in selected}
            if effort > budget:
                budget_rejected += 1
                continue
            if any(not set(edit["requires"]).issubset(selected_ids) for edit in selected):
                constraint_rejected += 1
                continue
            if any(set(edit["conflicts_with"]) & selected_ids for edit in selected):
                constraint_rejected += 1
                continue
            admissible += 1
            graph = apply_subset(baseline, selected)
            value = query_value(graph, query["metric"])
            met = decision_met(value, query["decision"])
            if met == base_decision:
                continue
            flipping += 1
            if best_cost is None or effort < best_cost:
                best_cost = effort
                minimal = []
            if effort == best_cost:
                minimal.append(
                    {
                        "edit_ids": sorted(selected_ids),
                        "effort_units": effort,
                        "resulting_graph_sha256": graph_digest(graph),
                        "resulting_query_value": value,
                        "resulting_decision_met": met,
                    }
                )

    minimal.sort(key=lambda row: row["edit_ids"])
    return {
        "baseline_query_value": base_value,
        "baseline_decision_met": base_decision,
        "subset_count_evaluated": subset_count,
        "budget_rejected_subset_count": budget_rejected,
        "constraint_rejected_subset_count": constraint_rejected,
        "admissible_subset_count_within_budget": admissible,
        "decision_flipping_subset_count_within_budget": flipping,
        "minimal_witness_effort_units": best_cost,
        "minimal_witness_count": len(minimal),
        "witnesses": minimal[:128],
    }


def verify_report(input_bytes: bytes, source: dict[str, Any], report: dict[str, Any]) -> None:
    if report.get("schema_version") != 1:
        fail("report schema_version mismatch")
    if report.get("tool") != "connectome-query-flipping-witness-audit":
        fail("report tool identity mismatch")
    if report.get("input_sha256") != hashlib.sha256(input_bytes).hexdigest():
        fail("report does not bind exact input bytes")
    if report.get("dataset_revision_commit_sha") != source["dataset_revision_commit_sha"]:
        fail("dataset revision mismatch")
    if report.get("edit_universe_scope") != source["edit_universe_scope"]:
        fail("edit universe scope mismatch")
    if report.get("baseline_graph_sha256") != source["baseline_graph_sha256"]:
        fail("baseline graph binding mismatch")
    if report.get("query_id") != source["query"]["query_id"]:
        fail("query ID mismatch")
    expected_query_sha = hashlib.sha256(json_bytes(canonical_query(source["query"]))).hexdigest()
    if report.get("query_spec_sha256") != expected_query_sha:
        fail("query specification digest mismatch")
    expected_universe_sha = hashlib.sha256(json_bytes(canonical_edit_universe(source["edits"]))).hexdigest()
    if report.get("edit_universe_sha256") != expected_universe_sha:
        fail("canonical edit-universe digest mismatch")
    if report.get("edit_atom_count") != len(source["edits"]):
        fail("edit atom count mismatch")
    if report.get("effort_unit") != source["effort_unit"]:
        fail("effort unit mismatch")
    if report.get("max_total_effort_units") != source["max_total_effort_units"]:
        fail("effort budget mismatch")

    expected = expected_enumeration(source)
    for key in (
        "subset_count_evaluated",
        "budget_rejected_subset_count",
        "constraint_rejected_subset_count",
        "admissible_subset_count_within_budget",
        "decision_flipping_subset_count_within_budget",
        "minimal_witness_effort_units",
        "minimal_witness_count",
    ):
        if report.get(key) != expected[key]:
            fail(f"{key} mismatch: expected {expected[key]!r}, got {report.get(key)!r}")
    close(report.get("baseline_query_value"), expected["baseline_query_value"], "baseline query value")
    if report.get("baseline_decision_met") is not expected["baseline_decision_met"]:
        fail("baseline decision mismatch")

    expected_status = (
        "COUNTEREXAMPLE_FOUND"
        if expected["minimal_witness_effort_units"] is not None
        else "NO_WITNESS_WITHIN_DECLARED_EDIT_UNIVERSE_AND_BUDGET"
    )
    if report.get("status") != expected_status:
        fail("witness status mismatch")
    if report.get("search_complete") is not True:
        fail("search must report complete exact enumeration")
    if report.get("probability_claim") != "NONE_DETERMINISTIC_FINITE_EDIT_UNIVERSE_ONLY":
        fail("deterministic finite edit search must not make a probability claim")
    if report.get("witnesses_returned") != len(expected["witnesses"]):
        fail("witness return count mismatch")
    if report.get("minimal_witnesses_truncated") is not (expected["minimal_witness_count"] > 128):
        fail("witness truncation flag mismatch")

    actual_witnesses = report.get("witnesses")
    if not isinstance(actual_witnesses, list) or len(actual_witnesses) != len(expected["witnesses"]):
        fail("witness list cardinality mismatch")
    for index, expected_witness in enumerate(expected["witnesses"]):
        actual = actual_witnesses[index]
        if actual.get("edit_ids") != expected_witness["edit_ids"]:
            fail(f"witness {index} edit membership mismatch")
        if actual.get("effort_units") != expected_witness["effort_units"]:
            fail(f"witness {index} effort mismatch")
        if actual.get("resulting_graph_sha256") != expected_witness["resulting_graph_sha256"]:
            fail(f"witness {index} graph digest mismatch")
        if actual.get("resulting_decision_met") is not expected_witness["resulting_decision_met"]:
            fail(f"witness {index} decision mismatch")
        close(actual.get("resulting_query_value"), expected_witness["resulting_query_value"], f"witness {index} query value")


def negative_controls(input_bytes: bytes, source: dict[str, Any], report: dict[str, Any]) -> int:
    detected = 0
    mutations = []

    changed_cost = copy.deepcopy(report)
    changed_cost["minimal_witness_effort_units"] = (
        changed_cost["minimal_witness_effort_units"] + 1
    )
    mutations.append(changed_cost)

    changed_membership = copy.deepcopy(report)
    changed_membership["witnesses"][0]["edit_ids"].pop()
    mutations.append(changed_membership)

    changed_graph = copy.deepcopy(report)
    changed_graph["witnesses"][0]["resulting_graph_sha256"] = "0" * 64
    mutations.append(changed_graph)

    for mutated in mutations:
        try:
            verify_report(input_bytes, source, mutated)
        except AssertionError:
            detected += 1
        else:
            fail("independent oracle accepted a deliberately mutated report")
    return detected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    input_bytes = args.input.read_bytes()
    source = json.loads(input_bytes)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    verify_report(input_bytes, source, report)
    detected = negative_controls(input_bytes, source, report)
    print(
        json.dumps(
            {
                "status": "INDEPENDENT_WITNESS_ORACLE_AGREEMENT",
                "oracle": "python-standard-library-combinations-floyd-warshall-v1",
                "edit_atom_count": len(source["edits"]),
                "subset_count_exhaustively_checked": (1 << len(source["edits"])) - 1,
                "minimal_witness_count": report["minimal_witness_count"],
                "negative_control_mutations_detected": detected,
                "probability_claim": "NONE",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"independent witness oracle failed: {error}", file=sys.stderr)
        raise SystemExit(1)
