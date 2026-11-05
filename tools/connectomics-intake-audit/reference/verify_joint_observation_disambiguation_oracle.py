#!/usr/bin/env python3
"""Independent tuple-projection oracle for correlated connectome observations."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_field(hasher, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)


def scenario_digest(scenarios: list[dict]) -> str:
    rows = sorted(scenarios, key=lambda row: row["scenario_id"])
    h = hashlib.sha256()
    h.update(b"connectome-query-disambiguation-scenario-set-v1")
    h.update(len(rows).to_bytes(8, "big"))
    for world in rows:
        h.update(b"WORLD")
        hash_field(h, world["scenario_id"])
        h.update(world["graph_sha256"].encode("ascii"))
        h.update(bytes([int(world["decision_met"])]))
    return h.hexdigest()


def action_digest(actions: list[dict]) -> str:
    rows = sorted(actions, key=lambda row: row["action_id"])
    h = hashlib.sha256()
    h.update(b"connectome-joint-observation-action-set-v1")
    h.update(len(rows).to_bytes(8, "big"))
    for action in rows:
        h.update(b"ACTION")
        hash_field(h, action["action_id"])
        hash_field(h, action["action_kind"])
        h.update(action["effort_units"].to_bytes(8, "big"))
        h.update(action["source_evidence_sha256"].encode("ascii"))
        h.update(action["observation_model_sha256"].encode("ascii"))
        groups = sorted(action["dependence_groups"])
        h.update(len(groups).to_bytes(8, "big"))
        for group in groups:
            h.update(b"DEPENDENCE")
            hash_field(h, group)
    return h.hexdigest()


def normalized_rows(support: dict, action_ids: list[str]) -> list[tuple[str, ...]]:
    rows = []
    for item in support["possible_joint_outcomes"]:
        mapping = {outcome["action_id"]: outcome["outcome"] for outcome in item["outcomes"]}
        if len(mapping) != len(action_ids) or set(mapping) != set(action_ids):
            raise AssertionError("joint tuple does not bind exactly one value to each action")
        rows.append(tuple(mapping[action_id] for action_id in action_ids))
    if len(rows) != len(set(rows)):
        raise AssertionError("duplicate joint tuple")
    return sorted(rows)


def joint_digest(scenarios: list[dict], actions: list[dict], supports: list[dict]) -> str:
    worlds = sorted(scenarios, key=lambda row: row["scenario_id"])
    action_ids = sorted(action["action_id"] for action in actions)
    by_world = {support["scenario_id"]: support for support in supports}
    h = hashlib.sha256()
    h.update(b"connectome-joint-observation-support-set-v1")
    h.update(len(worlds).to_bytes(8, "big"))
    for world in worlds:
        support = by_world[world["scenario_id"]]
        h.update(b"WORLD-SUPPORT")
        hash_field(h, world["scenario_id"])
        h.update(support["source_evidence_sha256"].encode("ascii"))
        h.update(support["joint_observation_model_sha256"].encode("ascii"))
        groups = sorted(support["dependence_groups"])
        h.update(len(groups).to_bytes(8, "big"))
        for group in groups:
            h.update(b"DEPENDENCE")
            hash_field(h, group)
        rows = normalized_rows(support, action_ids)
        h.update(len(rows).to_bytes(8, "big"))
        for row in rows:
            h.update(b"JOINT-TUPLE")
            for action_id, outcome in zip(action_ids, row):
                hash_field(h, action_id)
                hash_field(h, outcome)
    return h.hexdigest()


def conflict_pairs(scenarios: list[dict]) -> list[dict]:
    worlds = sorted(scenarios, key=lambda row: row["scenario_id"])
    return [
        {"world_a": left["scenario_id"], "world_b": right["scenario_id"]}
        for i, left in enumerate(worlds)
        for right in worlds[i + 1 :]
        if left["decision_met"] != right["decision_met"]
    ]


def projection(support: list[tuple[str, ...]], selected: tuple[int, ...]) -> set[tuple[str, ...]]:
    return {tuple(row[index] for index in selected) for row in support}


def separated(
    left: list[tuple[str, ...]],
    right: list[tuple[str, ...]],
    selected: tuple[int, ...],
) -> bool:
    return projection(left, selected).isdisjoint(projection(right, selected))


def run(data: dict, input_bytes: bytes) -> dict:
    assert scenario_digest(data["scenarios"]) == data["scenario_set_sha256"]
    assert action_digest(data["actions"]) == data["action_set_sha256"]
    assert joint_digest(data["scenarios"], data["actions"], data["joint_world_supports"]) == data["joint_support_sha256"]

    scenarios = sorted(data["scenarios"], key=lambda row: row["scenario_id"])
    actions = sorted(data["actions"], key=lambda row: row["action_id"])
    action_ids = [action["action_id"] for action in actions]
    action_effort = [action["effort_units"] for action in actions]
    support_by_world = {support["scenario_id"]: support for support in data["joint_world_supports"]}
    tuple_rows = [
        normalized_rows(support_by_world[world["scenario_id"]], action_ids)
        for world in scenarios
    ]
    world_index = {world["scenario_id"]: index for index, world in enumerate(scenarios)}
    pairs = conflict_pairs(scenarios)

    diagnostics = []
    marginally_separable_pair_count = 0
    jointly_undiscriminable_pairs = []
    for pair in pairs:
        left = tuple_rows[world_index[pair["world_a"]]]
        right = tuple_rows[world_index[pair["world_b"]]]
        marginal_ids = [
            action_ids[i]
            for i in range(len(action_ids))
            if separated(left, right, (i,))
        ]
        if marginal_ids:
            marginally_separable_pair_count += 1
        jointly_separable = separated(left, right, tuple(range(len(action_ids))))
        if not jointly_separable:
            jointly_undiscriminable_pairs.append(pair)
        diagnostics.append({
            "pair": pair,
            "marginally_separated_by_action_ids": marginal_ids,
            "jointly_separable_with_full_action_universe": jointly_separable,
        })

    action_coverage = []
    for action_index, action_id in enumerate(action_ids):
        count = sum(
            separated(
                tuple_rows[world_index[pair["world_a"]]],
                tuple_rows[world_index[pair["world_b"]]],
                (action_index,),
            )
            for pair in pairs
        )
        action_coverage.append({
            "action_id": action_id,
            "effort_units": action_effort[action_index],
            "marginally_separated_decision_conflict_pair_count": count,
        })

    jointly_separable_pair_count = len(pairs) - len(jointly_undiscriminable_pairs)
    subset_count = (1 << len(action_ids)) - 1
    best_cost = None
    best_count = 0
    plans = []
    if not pairs:
        best_cost = 0
        best_count = 1
        plans = [{"action_ids": [], "effort_units": 0,
                  "jointly_separated_decision_conflict_pair_count": 0}]
        evaluated_subsets = 1
    elif jointly_undiscriminable_pairs:
        evaluated_subsets = 0
    else:
        evaluated_subsets = subset_count
        for size in range(1, len(action_ids) + 1):
            for selected in itertools.combinations(range(len(action_ids)), size):
                cost = sum(action_effort[index] for index in selected)
                if best_cost is not None and cost > best_cost:
                    continue
                if not all(
                    separated(
                        tuple_rows[world_index[pair["world_a"]]],
                        tuple_rows[world_index[pair["world_b"]]],
                        selected,
                    )
                    for pair in pairs
                ):
                    continue
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_count = 0
                    plans = []
                if cost == best_cost:
                    best_count += 1
                    if len(plans) < 128:
                        plans.append({
                            "action_ids": sorted(action_ids[index] for index in selected),
                            "effort_units": cost,
                            "jointly_separated_decision_conflict_pair_count": len(pairs),
                        })
        plans.sort(key=lambda row: row["action_ids"])

    within_budget = None if best_cost is None else best_cost <= data["max_total_effort_units"]
    if not pairs:
        status = "NO_DECISION_DISAGREEMENT_WITHIN_DECLARED_SCENARIOS"
    elif jointly_undiscriminable_pairs:
        status = "NO_JOINT_DISAMBIGUATION_WITHIN_DECLARED_SUPPORT"
    elif within_budget:
        status = "MINIMUM_COST_JOINT_DISAMBIGUATION_PLAN"
    else:
        status = "MINIMUM_COST_JOINT_DISAMBIGUATION_EXCEEDS_BUDGET"

    return {
        "schema_version": 1,
        "tool": "connectome-joint-observation-disambiguation-audit",
        "analysis_id": data["analysis_id"],
        "input_sha256": sha256(input_bytes),
        "dataset_revision_commit_sha": data["dataset_revision_commit_sha"],
        "query_id": data["query_id"],
        "query_spec_sha256": data["query_spec_sha256"],
        "scenario_set_scope": data["scenario_set_scope"],
        "scenario_set_sha256": data["scenario_set_sha256"],
        "action_set_sha256": data["action_set_sha256"],
        "joint_support_scope": data["joint_support_scope"],
        "joint_support_sha256": data["joint_support_sha256"],
        "effort_unit": data["effort_unit"],
        "max_total_effort_units": data["max_total_effort_units"],
        "scenario_count": len(data["scenarios"]),
        "observation_action_count": len(data["actions"]),
        "joint_support_tuple_count": sum(len(rows) for rows in tuple_rows),
        "decision_conflict_pair_count": len(pairs),
        "marginally_separable_pair_count": marginally_separable_pair_count,
        "jointly_separable_pair_count": jointly_separable_pair_count,
        "jointly_undiscriminable_pairs": jointly_undiscriminable_pairs,
        "pair_diagnostics": diagnostics,
        "action_marginal_coverage": action_coverage,
        "subset_count_evaluated": evaluated_subsets,
        "minimum_joint_disambiguation_effort_units": best_cost,
        "minimum_joint_plan_count": best_count,
        "joint_plans_returned": len(plans),
        "minimum_joint_plans_truncated": best_count > 128,
        "minimum_joint_plan_within_budget": within_budget,
        "joint_plans": plans,
        "marginal_separation_rule": "SINGLE_ACTION_MARGINAL_SUPPORTS_DISJOINT",
        "joint_separation_rule": "PROJECTED_JOINT_SUPPORT_RELATIONS_DISJOINT",
        "probability_claim": "NONE_FINITE_SET_VALUED_CORRELATED_SUPPORT_ONLY",
        "status": status,
        "search_complete": True,
        "scope_note": "The planner uses caller-supplied finite joint-support relations and exact projection onto selected action subsets. It does not factorize marginal supports, infer independence, assign probabilities, prove that joint-support rows are complete, validate measurement models, or establish biological truth. Results are scoped to the declared worlds, actions, and joint support relations.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        input_bytes = args.input.read_bytes()
        data = json.loads(input_bytes)
        expected = run(data, input_bytes)
        actual = json.loads(args.report.read_text(encoding="utf-8"))
        assert actual == expected, "Rust report differs from independent tuple-projection oracle"
        print(
            f"verified {expected['status']}: "
            f"marginally separable={expected['marginally_separable_pair_count']}; "
            f"jointly separable={expected['jointly_separable_pair_count']}; "
            f"minimum={expected['minimum_joint_disambiguation_effort_units']}"
        )
        return 0
    except Exception as exc:
        print(f"joint observation oracle failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
