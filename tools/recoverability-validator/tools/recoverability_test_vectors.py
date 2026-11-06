#!/usr/bin/env python3
"""Independent synthetic vectors for the Rust recoverability manifest auditor.

Only opaque IDs, synthetic graph edges, and fake SHA-256 values are used.
This is an evaluator-side reference implementation, not a retrieval agent.
"""
from __future__ import annotations

import copy
import hashlib
import heapq
import json
import sys
from pathlib import Path
from typing import Any

PREFIX = b"symthaea.science_bench_refresh.recoverability.v1\0"
SCHEMA = "symthaea.science_bench_refresh_recoverability_manifest.v1"
ROOT = Path(__file__).resolve().parents[1] / "fixtures"


def compact(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def framed_commitment(domain: bytes, payload: bytes) -> str:
    framed = (
        PREFIX
        + len(domain).to_bytes(8, "big")
        + domain
        + len(payload).to_bytes(8, "big")
        + payload
    )
    return hashlib.sha256(framed).hexdigest()


def fields_of(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_commitment_sha256"}


def commit_fields(fields: dict[str, Any]) -> str:
    return framed_commitment(b"manifest-fields", compact(fields))


def save_manifest(fields: dict[str, Any], path: Path) -> dict[str, Any]:
    manifest = copy.deepcopy(fields)
    manifest["manifest_commitment_sha256"] = commit_fields(fields)
    path.write_bytes(compact(manifest) + b"\n")
    return manifest


def shortest_route(
    manifest: dict[str, Any],
    target: str,
    mode: str,
) -> tuple[int, tuple[str, ...]] | None:
    allowed = {
        "direct": {"initial_retrieval"},
        "reformulation": {
            "initial_retrieval",
            "query_reformulation",
            "post_reformulation_retrieval",
        },
        "navigation": None,
    }[mode]
    start = manifest["start_node_ref"]
    initial = reformulated = post_retrieval = pending_reformulation = navigated = False
    start_state = (
        start,
        initial,
        reformulated,
        post_retrieval,
        pending_reformulation,
        navigated,
    )
    queue: list[tuple[int, tuple[str, ...], tuple[Any, ...]]] = [(0, (), start_state)]
    best: dict[tuple[Any, ...], tuple[int, tuple[str, ...]]] = {start_state: (0, ())}
    adjacency: dict[str, list[dict[str, Any]]] = {}
    for edge in manifest["edges"]:
        adjacency.setdefault(edge["from_node_ref"], []).append(edge)
    for edges in adjacency.values():
        edges.sort(key=lambda edge: edge["edge_ref"])

    while queue:
        cost, route, state = heapq.heappop(queue)
        if best.get(state) != (cost, route):
            continue
        (
            node,
            used_initial,
            used_reform,
            used_post,
            pending_reformulation,
            used_navigation,
        ) = state
        qualifies = (
            (mode == "direct" and used_initial and not used_reform and not used_navigation)
            or (
                mode == "reformulation"
                and used_initial
                and used_reform
                and used_post
                and not pending_reformulation
                and not used_navigation
            )
            or (mode == "navigation" and used_navigation)
        )
        if node == target and qualifies:
            return cost, route

        for edge in adjacency.get(node, []):
            operation = edge["operation"]
            if allowed is not None and operation not in allowed:
                continue
            navigation_operation = operation in {
                "open_document",
                "follow_link",
                "open_table_of_contents",
                "pagination",
                "in_document_find",
            }
            if operation == "initial_retrieval":
                order_ok = (
                    not used_initial
                    and not used_reform
                    and not used_navigation
                )
            elif operation == "query_reformulation":
                order_ok = (
                    used_initial
                    and not used_reform
                    and not pending_reformulation
                    and not used_post
                    and not used_navigation
                )
            elif operation == "post_reformulation_retrieval":
                order_ok = (
                    pending_reformulation
                    and not used_post
                    and not used_navigation
                )
            else:
                order_ok = used_initial and navigation_operation
            if not order_ok:
                continue

            next_cost = cost + edge["cost"]
            if next_cost > manifest["max_route_cost"]:
                continue
            next_state = (
                edge["to_node_ref"],
                used_initial or operation == "initial_retrieval",
                used_reform or operation == "query_reformulation",
                used_post or operation == "post_reformulation_retrieval",
                (
                    True
                    if operation == "query_reformulation"
                    else False
                    if operation == "post_reformulation_retrieval"
                    else pending_reformulation
                ),
                used_navigation or navigation_operation,
            )
            next_route = route + (edge["edge_ref"],)
            candidate = (next_cost, next_route)
            if next_state not in best or candidate < best[next_state]:
                best[next_state] = candidate
                heapq.heappush(queue, (next_cost, next_route, next_state))
    return None


def classify(manifest: dict[str, Any], unit: dict[str, Any]) -> dict[str, Any]:
    artifact_available = unit["artifact_availability"] == "present_pinned"
    surface_unknown = manifest["surface_state"] == "unknown"
    if not artifact_available or surface_unknown:
        label, route = "NotEvaluable", None
    else:
        direct = shortest_route(manifest, unit["target_node_ref"], "direct")
        reformulation = shortest_route(manifest, unit["target_node_ref"], "reformulation")
        navigation = shortest_route(manifest, unit["target_node_ref"], "navigation")
        if direct is not None:
            label, route = "DirectlyReachable", direct
        elif reformulation is not None:
            label, route = "ReformulationRecoverable", reformulation
        elif navigation is not None:
            label, route = "NavigationRecoverable", navigation
        elif manifest["surface_state"] == "complete_frozen_inventory":
            label, route = "UnreachableWithinBudget", None
        else:
            label, route = "NotEvaluable", None

    return {
        "evidence_unit_id": unit["evidence_unit_id"],
        "recoverability": label,
        "route_cost": route[0] if route else None,
        "route_edge_refs": list(route[1]) if route else [],
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: recoverability_test_vectors.py OUTPUT_DIR", file=sys.stderr)
        return 2
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    original_bytes = (ROOT / "synthetic-manifest.json").read_bytes()
    base = json.loads(original_bytes)
    if base["schema"] != SCHEMA:
        raise SystemExit("fixture schema mismatch")
    base_fields = fields_of(base)
    expected_commitment = commit_fields(base_fields)
    if expected_commitment != base["manifest_commitment_sha256"]:
        raise SystemExit("Python typed commitment does not match frozen fixture")

    fields_path = out_dir / "fields.json"
    fields_path.write_bytes(compact(base_fields) + b"\n")
    valid_path = out_dir / "valid-manifest.json"
    valid_path.write_bytes(original_bytes)

    expected_records = [classify(base, unit) for unit in base["evidence_units"]]
    expected_labels = {
        "unit:direct": "DirectlyReachable",
        "unit:rewrite": "ReformulationRecoverable",
        "unit:navigation": "NavigationRecoverable",
        "unit:unreachable": "UnreachableWithinBudget",
        "unit:unknown": "NotEvaluable",
    }
    actual_labels = {row["evidence_unit_id"]: row["recoverability"] for row in expected_records}
    if actual_labels != expected_labels:
        raise SystemExit(f"reference classification mismatch: {actual_labels!r}")

    # A changed manifest with an unchanged commitment must fail closed.
    tampered = copy.deepcopy(base)
    tampered["max_route_cost"] += 1
    tampered_path = out_dir / "tampered-manifest.json"
    tampered_path.write_bytes(compact(tampered) + b"\n")

    # An agent/model annotation is not part of the model-free manifest contract.
    model_field = copy.deepcopy(base)
    model_field["model_judgment"] = "directly_reachable"
    (out_dir / "model-field-manifest.json").write_bytes(compact(model_field) + b"\n")

    # Positive route witnesses survive on a partial observed surface; absent routes do not.
    partial_fields = copy.deepcopy(base_fields)
    partial_fields["surface_state"] = "partial_observed"
    partial_path = out_dir / "partial-surface-manifest.json"
    partial = save_manifest(partial_fields, partial_path)
    partial_rows = [classify(partial, unit) for unit in partial["evidence_units"]]
    partial_by_id = {row["evidence_unit_id"]: row["recoverability"] for row in partial_rows}
    if partial_by_id["unit:direct"] != "DirectlyReachable":
        raise SystemExit("partial surface discarded a concrete direct route")
    if partial_by_id["unit:rewrite"] != "ReformulationRecoverable":
        raise SystemExit("partial surface discarded a concrete reformulation route")
    if partial_by_id["unit:navigation"] != "NavigationRecoverable":
        raise SystemExit("partial surface discarded a concrete navigation route")
    if partial_by_id["unit:unreachable"] != "NotEvaluable":
        raise SystemExit("partial surface incorrectly asserted unreachability")

    unknown_fields = copy.deepcopy(base_fields)
    unknown_fields["surface_state"] = "unknown"
    unknown_path = out_dir / "unknown-surface-manifest.json"
    unknown_manifest = save_manifest(unknown_fields, unknown_path)
    if any(classify(unknown_manifest, unit)["recoverability"] != "NotEvaluable" for unit in unknown_manifest["evidence_units"]):
        raise SystemExit("unknown surface did not force NotEvaluable")

    # Reducing route budget cannot improve any recoverability class.
    lower_budget = copy.deepcopy(base_fields)
    lower_budget["max_route_cost"] = 2
    lower = save_manifest(lower_budget, out_dir / "lower-budget-manifest.json")
    rank = {
        "DirectlyReachable": 0,
        "ReformulationRecoverable": 1,
        "NavigationRecoverable": 2,
        "UnreachableWithinBudget": 3,
        "NotEvaluable": 4,
    }
    base_by_id = {row["evidence_unit_id"]: row for row in expected_records}
    lower_by_id = {unit["evidence_unit_id"]: classify(lower, unit) for unit in lower["evidence_units"]}
    if any(
        rank[lower_by_id[key]["recoverability"]] < rank[row["recoverability"]]
        for key, row in base_by_id.items()
    ):
        raise SystemExit("lowering the budget improved recoverability")

    # A concrete new first-pass edge can improve one unit to DirectlyReachable.
    added = copy.deepcopy(base_fields)
    added["edges"].append(
        {
            "edge_ref": "route:added-direct",
            "from_node_ref": "state:start",
            "to_node_ref": "evidence:unreachable",
            "operation": "initial_retrieval",
            "cost": 1,
        }
    )
    added_manifest = save_manifest(added, out_dir / "added-route-manifest.json")
    added_row = next(
        classify(added_manifest, unit)
        for unit in added_manifest["evidence_units"]
        if unit["evidence_unit_id"] == "unit:unreachable"
    )
    if added_row["recoverability"] != "DirectlyReachable":
        raise SystemExit("adding a valid direct route did not improve reachability")

    # A post-reformulation retrieval cannot occur before the rewrite it depends on.
    out_of_order = copy.deepcopy(base_fields)
    out_of_order["nodes"].append({"node_ref": "evidence:orphan-post-retrieval"})
    out_of_order["edges"].append(
        {
            "edge_ref": "route:orphan-post-retrieval",
            "from_node_ref": "state:start",
            "to_node_ref": "evidence:orphan-post-retrieval",
            "operation": "post_reformulation_retrieval",
            "cost": 1,
        }
    )
    out_of_order["evidence_units"].append(
        {
            "evidence_unit_id": "unit:orphan-post-retrieval",
            "target_node_ref": "evidence:orphan-post-retrieval",
            "artifact_sha256": "6" * 64,
            "source_family_ref": "family:five",
            "artifact_availability": "present_pinned",
        }
    )
    out_of_order_manifest = save_manifest(
        out_of_order, out_dir / "out-of-order-reformulation-manifest.json"
    )
    orphan = next(
        classify(out_of_order_manifest, unit)
        for unit in out_of_order_manifest["evidence_units"]
        if unit["evidence_unit_id"] == "unit:orphan-post-retrieval"
    )
    if orphan["recoverability"] != "UnreachableWithinBudget":
        raise SystemExit("post-reformulation retrieval succeeded without a prior rewrite")

    # Query reformulation cannot occur without a first-pass retrieval.
    no_initial = copy.deepcopy(base_fields)
    no_initial["edges"] = [
        edge for edge in no_initial["edges"] if edge["edge_ref"] != "route:rewrite-initial"
    ]
    no_initial_manifest = save_manifest(no_initial, out_dir / "no-initial-before-rewrite-manifest.json")
    no_initial_row = next(
        classify(no_initial_manifest, unit)
        for unit in no_initial_manifest["evidence_units"]
        if unit["evidence_unit_id"] == "unit:rewrite"
    )
    if no_initial_row["recoverability"] != "UnreachableWithinBudget":
        raise SystemExit("query reformulation succeeded without first-pass retrieval")

    # Structural navigation cannot begin from an unobserved document state.
    no_navigation_initial = copy.deepcopy(base_fields)
    no_navigation_initial["edges"] = [
        edge for edge in no_navigation_initial["edges"]
        if edge["edge_ref"] != "route:navigation-initial"
    ]
    no_navigation_manifest = save_manifest(
        no_navigation_initial, out_dir / "no-initial-before-navigation-manifest.json"
    )
    no_navigation_row = next(
        classify(no_navigation_manifest, unit)
        for unit in no_navigation_manifest["evidence_units"]
        if unit["evidence_unit_id"] == "unit:navigation"
    )
    if no_navigation_row["recoverability"] != "UnreachableWithinBudget":
        raise SystemExit("navigation succeeded without first-pass retrieval")

    summary = {
        "schema": "symthaea.science_bench_refresh_recoverability_python_vectors.v1",
        "expected_manifest_commitment_sha256": expected_commitment,
        "expected_records": expected_records,
        "expected_evidence_unit_count": len(expected_records),
        "expected_distinct_source_family_count": len(
            {unit["source_family_ref"] for unit in base["evidence_units"]}
        ),
        "metamorphic_controls": {
            "partial_surface_preserves_positive_routes_only": True,
            "unknown_surface_all_not_evaluable": True,
            "lower_budget_never_improves": True,
            "added_direct_route_improves_unreachable_unit": True,
            "post_retrieval_requires_prior_reformulation": True,
            "reformulation_requires_prior_initial_retrieval": True,
            "navigation_requires_prior_initial_retrieval": True,
            "model_free_reference_classifier": True,
        },
    }
    (out_dir / "summary.json").write_bytes(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
