#!/usr/bin/env python3
"""Independent exact adaptive policy oracle using tuple filtering and memoized recursion."""
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
    worlds = sorted(scenarios, key=lambda row: row["scenario_id"])
    h = hashlib.sha256()
    h.update(b"connectome-query-disambiguation-scenario-set-v1")
    h.update(len(worlds).to_bytes(8, "big"))
    for world in worlds:
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
    for record in support["possible_joint_outcomes"]:
        mapping = {item["action_id"]: item["outcome"] for item in record["outcomes"]}
        if len(mapping) != len(action_ids) or set(mapping) != set(action_ids):
            raise AssertionError("joint tuple must bind exactly one value to every action")
        rows.append(tuple(mapping[action_id] for action_id in action_ids))
    if len(rows) != len(set(rows)):
        raise AssertionError("duplicate joint tuple")
    return sorted(rows)


def joint_digest(scenarios: list[dict], actions: list[dict], supports: list[dict]) -> str:
    worlds = sorted(scenarios, key=lambda row: row["scenario_id"])
    ordered_actions = sorted(actions, key=lambda row: row["action_id"])
    action_ids = [action["action_id"] for action in ordered_actions]
    by_world = {support["scenario_id"]: support for support in supports}
    if len(by_world) != len(supports) or set(by_world) != {w["scenario_id"] for w in worlds}:
        raise AssertionError("joint support must include each world exactly once")
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


def build_outcome_masks(
    action_ids: list[str],
    tuple_rows: list[list[tuple[str, ...]]],
) -> list[list[dict[str, int]]]:
    masks: list[list[dict[str, int]]] = []
    for action_index, _action_id in enumerate(action_ids):
        per_world = []
        for rows in tuple_rows:
            by_outcome: dict[str, int] = {}
            for row_index, row in enumerate(rows):
                by_outcome[row[action_index]] = by_outcome.get(row[action_index], 0) | (1 << row_index)
            per_world.append(by_outcome)
        masks.append(per_world)
    return masks


def fixed_plan_summary(
    actions: list[dict],
    scenarios: list[dict],
    tuple_rows: list[list[tuple[str, ...]]],
    budget: int,
) -> dict:
    conflicts = [
        (left, right)
        for i, left in enumerate(scenarios)
        for right in scenarios[i + 1 :]
        if left["decision_met"] != right["decision_met"]
    ]
    if not conflicts:
        return {
            "minimum_effort_units": 0,
            "minimum_action_ids": [],
            "minimum_plan_count": 1,
            "within_declared_budget": True,
            "adaptive_worst_case_savings_units": None,
        }
    best_cost = None
    best_ids: list[str] = []
    best_count = 0
    for size in range(1, len(actions) + 1):
        for selected in itertools.combinations(range(len(actions)), size):
            valid = True
            for left, right in conflicts:
                left_index = scenarios.index(left)
                right_index = scenarios.index(right)
                left_projection = {
                    tuple(row[index] for index in selected)
                    for row in tuple_rows[left_index]
                }
                right_projection = {
                    tuple(row[index] for index in selected)
                    for row in tuple_rows[right_index]
                }
                if not left_projection.isdisjoint(right_projection):
                    valid = False
                    break
            if not valid:
                continue
            cost = sum(actions[index]["effort_units"] for index in selected)
            candidate_ids = sorted(actions[index]["action_id"] for index in selected)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_ids = candidate_ids
                best_count = 1
            elif cost == best_cost:
                best_count += 1
                if candidate_ids < best_ids:
                    best_ids = candidate_ids
    return {
        "minimum_effort_units": best_cost,
        "minimum_action_ids": best_ids,
        "minimum_plan_count": best_count,
        "within_declared_budget": None if best_cost is None else best_cost <= budget,
        "adaptive_worst_case_savings_units": None,
    }


def run(data: dict, input_bytes: bytes) -> dict:
    assert scenario_digest(data["scenarios"]) == data["scenario_set_sha256"]
    assert action_digest(data["actions"]) == data["action_set_sha256"]
    assert joint_digest(data["scenarios"], data["actions"], data["joint_world_supports"]) == data["joint_support_sha256"]

    scenarios = sorted(data["scenarios"], key=lambda row: row["scenario_id"])
    actions = sorted(data["actions"], key=lambda row: row["action_id"])
    action_ids = [row["action_id"] for row in actions]
    action_costs = [row["effort_units"] for row in actions]
    support_by_world = {row["scenario_id"]: row for row in data["joint_world_supports"]}
    tuple_rows = [
        normalized_rows(support_by_world[world["scenario_id"]], action_ids)
        for world in scenarios
    ]
    outcome_masks = build_outcome_masks(action_ids, tuple_rows)

    # State key: (remaining-action bitmask, per-world compatible joint-tuple bitmasks).
    memo: dict[tuple[int, tuple[int, ...]], dict | None] = {}
    state_counter = 0

    def compatible_world_indices(row_masks: tuple[int, ...]) -> list[int]:
        return [index for index, mask in enumerate(row_masks) if mask != 0]

    def terminal_decision(row_masks: tuple[int, ...]) -> bool | None:
        active = compatible_world_indices(row_masks)
        if not active:
            return None
        decision = scenarios[active[0]]["decision_met"]
        return decision if all(scenarios[index]["decision_met"] == decision for index in active) else None

    def solve(remaining_mask: int, row_masks: tuple[int, ...]) -> dict | None:
        nonlocal state_counter
        key = (remaining_mask, row_masks)
        if key in memo:
            return memo[key]
        state_counter += 1

        decision = terminal_decision(row_masks)
        active = compatible_world_indices(row_masks)
        world_ids = [scenarios[index]["scenario_id"] for index in active]
        if decision is not None:
            leaf = {
                "worst_case_cost": 0,
                "minimum_path_cost": 0,
                "maximum_path_cost": 0,
                "node": {
                    "action_id": None,
                    "action_cost_units": 0,
                    "terminal_query_decision": decision,
                    "compatible_world_ids": world_ids,
                    "branches": [],
                },
            }
            memo[key] = leaf
            return leaf
        if remaining_mask == 0:
            memo[key] = None
            return None

        best = None
        for action_index, action_id in enumerate(action_ids):
            bit = 1 << action_index
            if remaining_mask & bit == 0:
                continue

            observed = set()
            for world_index in active:
                for outcome, mask in outcome_masks[action_index][world_index].items():
                    if row_masks[world_index] & mask:
                        observed.add(outcome)
            if not observed:
                continue

            branches = []
            child_solutions = []
            valid_action = True
            for outcome in sorted(observed):
                child_masks = [0] * len(row_masks)
                for world_index in active:
                    mask = outcome_masks[action_index][world_index].get(outcome, 0)
                    child_masks[world_index] = row_masks[world_index] & mask
                child = solve(remaining_mask & ~bit, tuple(child_masks))
                if child is None:
                    valid_action = False
                    break
                branches.append({"observed_outcome": outcome, "child": child["node"]})
                child_solutions.append(child)
            if not valid_action or not branches:
                continue

            child_worst = max(child["worst_case_cost"] for child in child_solutions)
            child_min = min(child["minimum_path_cost"] for child in child_solutions)
            child_max = max(child["maximum_path_cost"] for child in child_solutions)
            cost = action_costs[action_index]
            candidate = {
                "worst_case_cost": cost + child_worst,
                "minimum_path_cost": cost + child_min,
                "maximum_path_cost": cost + child_max,
                "node": {
                    "action_id": action_id,
                    "action_cost_units": cost,
                    "terminal_query_decision": None,
                    "compatible_world_ids": world_ids,
                    "branches": branches,
                },
            }
            # Actions are iterated in lexical order. Keep the first exact tie,
            # matching the Rust deterministic policy selection contract.
            if best is None or candidate["worst_case_cost"] < best["worst_case_cost"]:
                best = candidate

        memo[key] = best
        return best

    initial_masks = tuple((1 << len(rows)) - 1 for rows in tuple_rows)
    initial_remaining = (1 << len(action_ids)) - 1
    solution = solve(initial_remaining, initial_masks)
    tuple_count = sum(len(rows) for rows in tuple_rows)
    fixed_summary = fixed_plan_summary(actions, scenarios, tuple_rows, data["max_total_effort_units"])
    if fixed_summary["minimum_effort_units"] is not None and solution is not None:
        fixed_summary["adaptive_worst_case_savings_units"] = (
            fixed_summary["minimum_effort_units"] - solution["worst_case_cost"]
        )

    def count_tree(node: dict) -> tuple[int, int]:
        nodes = 1
        branches = len(node["branches"])
        for branch in node["branches"]:
            child_nodes, child_branches = count_tree(branch["child"])
            nodes += child_nodes
            branches += child_branches
        return nodes, branches

    if solution is None:
        worst = minimum_path = maximum_path = within = policy = None
        policy_nodes = policy_branches = 0
        status = "NO_ADAPTIVE_POLICY_WITHIN_DECLARED_ACTIONS"
    else:
        worst = solution["worst_case_cost"]
        minimum_path = solution["minimum_path_cost"]
        maximum_path = solution["maximum_path_cost"]
        within = worst <= data["max_total_effort_units"]
        policy = solution["node"]
        policy_nodes, policy_branches = count_tree(policy)
        status = (
            "MINIMUM_WORST_CASE_ADAPTIVE_POLICY"
            if within
            else "MINIMUM_WORST_CASE_ADAPTIVE_POLICY_EXCEEDS_BUDGET"
        )

    return {
        "schema_version": 1,
        "tool": "connectome-adaptive-query-policy-audit",
        "analysis_id": data["analysis_id"],
        "input_sha256": sha256(input_bytes),
        "dataset_revision_commit_sha": data["dataset_revision_commit_sha"],
        "query_id": data["query_id"],
        "query_spec_sha256": data["query_spec_sha256"],
        "scenario_set_sha256": data["scenario_set_sha256"],
        "action_set_sha256": data["action_set_sha256"],
        "joint_support_sha256": data["joint_support_sha256"],
        "effort_unit": data["effort_unit"],
        "max_total_effort_units": data["max_total_effort_units"],
        "scenario_count": len(data["scenarios"]),
        "observation_action_count": len(data["actions"]),
        "joint_support_tuple_count": tuple_count,
        "exact_policy_state_count": state_counter,
        "policy_node_count": policy_nodes,
        "policy_branch_count": policy_branches,
        "worst_case_effort_units": worst,
        "minimum_terminal_path_effort_units": minimum_path,
        "maximum_terminal_path_effort_units": maximum_path,
        "worst_case_policy_within_budget": within,
        "fixed_non_adaptive_optimum": fixed_summary,
        "policy": policy,
        "out_of_model_observation_disposition": "FAIL_CLOSED_NO_QUERY_DECISION",
        "expected_cost_claim": "NONE_NO_PRIOR_OR_EXPECTED_COST_COMPUTED",
        "probability_claim": "NONE_FINITE_SET_VALUED_CORRELATED_SUPPORT_ONLY",
        "status": status,
        "search_complete": True,
        "scope_note": "The policy conditions the original joint tuple relations after each observed action outcome. It stops only when all compatible worlds agree on the frozen query decision, not when the exact world is identified. It does not factorize marginals, infer independence, assign probabilities, prove joint-support completeness, validate measurement models, or establish biological truth. An observation category outside every declared compatible tuple is out-of-model and must fail closed.",
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
        assert actual == expected, "Rust report differs from independent adaptive set-filter oracle"
        print(
            f"verified {expected['status']}: "
            f"states={expected['exact_policy_state_count']}; "
            f"worst_case={expected['worst_case_effort_units']}; "
            f"path_range={expected['minimum_terminal_path_effort_units']}..{expected['maximum_terminal_path_effort_units']}"
        )
        return 0
    except Exception as exc:
        print(f"adaptive query policy oracle failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
