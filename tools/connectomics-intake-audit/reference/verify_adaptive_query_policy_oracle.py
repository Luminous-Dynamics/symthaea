#!/usr/bin/env python3
"""Independent exact adaptive policy oracle using tuple filtering and memoized recursion."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import math
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


def adaptive_solution(
    actions: list[dict],
    scenarios: list[dict],
    tuple_rows: list[list[tuple[str, ...]]],
) -> tuple[dict | None, int]:
    action_ids = [row["action_id"] for row in actions]
    action_costs = [row["effort_units"] for row in actions]
    outcome_masks = build_outcome_masks(action_ids, tuple_rows)
    memo: dict[tuple[int, tuple[int, ...]], dict | None] = {}
    state_counter = 0

    def active_worlds(row_masks: tuple[int, ...]) -> list[int]:
        return [index for index, mask in enumerate(row_masks) if mask != 0]

    def terminal_decision(row_masks: tuple[int, ...]) -> bool | None:
        active = active_worlds(row_masks)
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
        active = active_worlds(row_masks)
        world_ids = [scenarios[index]["scenario_id"] for index in active]
        if decision is not None:
            leaf = {
                "worst_case_cost": 0,
                "minimum_path_cost": 0,
                "maximum_path_cost": 0,
                "optimal_policy_count": 1,
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
            cost = action_costs[action_index]
            candidate = {
                "worst_case_cost": cost + max(child["worst_case_cost"] for child in child_solutions),
                "minimum_path_cost": cost + min(child["minimum_path_cost"] for child in child_solutions),
                "maximum_path_cost": cost + max(child["maximum_path_cost"] for child in child_solutions),
                "optimal_policy_count": math.prod(child["optimal_policy_count"] for child in child_solutions),
                "node": {
                    "action_id": action_id,
                    "action_cost_units": cost,
                    "terminal_query_decision": None,
                    "compatible_world_ids": world_ids,
                    "branches": branches,
                },
            }
            if best is None or candidate["worst_case_cost"] < best["worst_case_cost"]:
                best = candidate
            elif candidate["worst_case_cost"] == best["worst_case_cost"]:
                best["optimal_policy_count"] += candidate["optimal_policy_count"]
        memo[key] = best
        return best

    initial_masks = tuple((1 << len(rows)) - 1 for rows in tuple_rows)
    solution = solve((1 << len(action_ids)) - 1, initial_masks)
    return solution, state_counter


def effort_non_decreasing(before: int | None, after: int | None) -> bool:
    if before is None:
        return after is None
    return after is None or after >= before


def support_challenge_digest(base_support_sha: str, challenge: dict, action_ids: list[str]) -> tuple[str, list[tuple[str, list[tuple[str, ...]]]]]:
    additions = sorted(challenge["additions"], key=lambda row: row["scenario_id"])
    h = hashlib.sha256()
    h.update(b"connectome-adaptive-support-expansion-challenge-v1")
    h.update(base_support_sha.encode("ascii"))
    hash_field(h, challenge["challenge_id"])
    h.update(challenge["challenger_evidence_sha256"].encode("ascii"))
    h.update(challenge["challenger_model_sha256"].encode("ascii"))
    h.update(len(additions).to_bytes(8, "big"))
    normalized_additions = []
    for addition in additions:
        h.update(b"WORLD-ADDITION")
        hash_field(h, addition["scenario_id"])
        h.update(addition["source_evidence_sha256"].encode("ascii"))
        h.update(addition["joint_observation_model_sha256"].encode("ascii"))
        groups = sorted(addition["dependence_groups"])
        h.update(len(groups).to_bytes(8, "big"))
        for group in groups:
            h.update(b"DEPENDENCE")
            hash_field(h, group)
        rows = normalized_rows(addition, action_ids)
        h.update(len(rows).to_bytes(8, "big"))
        for row in rows:
            h.update(b"ADDED-JOINT-TUPLE")
            for action_id, outcome in zip(action_ids, row):
                hash_field(h, action_id)
                hash_field(h, outcome)
        normalized_additions.append((addition["scenario_id"], rows))
    return h.hexdigest(), normalized_additions


def valid_sha(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


def scenario_expansion_challenge_digest(
    data: dict,
    challenge: dict,
    action_ids: list[str],
) -> str:
    additions = sorted(
        challenge["additions"],
        key=lambda item: item["scenario"]["scenario_id"],
    )
    h = hashlib.sha256()
    h.update(b"connectome-adaptive-hypothesis-space-expansion-v1")
    h.update(data["scenario_set_sha256"].encode("ascii"))
    h.update(data["joint_support_sha256"].encode("ascii"))
    hash_field(h, challenge["challenge_id"])
    h.update(challenge["challenger_evidence_sha256"].encode("ascii"))
    h.update(challenge["challenger_model_sha256"].encode("ascii"))
    h.update(len(additions).to_bytes(8, "big"))
    for addition in additions:
        scenario = addition["scenario"]
        support = addition["support"]
        if scenario["scenario_id"] != support["scenario_id"]:
            raise AssertionError("scenario/support identity mismatch")
        h.update(b"NEW-WORLD")
        hash_field(h, scenario["scenario_id"])
        h.update(scenario["graph_sha256"].encode("ascii"))
        h.update(bytes([int(scenario["decision_met"])]))
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


def build_scenario_challenges(
    data: dict,
    scenarios: list[dict],
    actions: list[dict],
    baseline_solution: dict | None,
    baseline_fixed: dict,
) -> list[dict]:
    action_ids = [action["action_id"] for action in actions]
    seen_challenges = set()
    results = []
    for challenge in data.get("scenario_challenges", []):
        challenge_id = challenge["challenge_id"]
        assert challenge_id and challenge_id not in seen_challenges, "duplicate/empty scenario challenge ID"
        seen_challenges.add(challenge_id)
        assert valid_sha(challenge["challenger_evidence_sha256"], 64)
        assert valid_sha(challenge["challenger_model_sha256"], 64)
        assert challenge["additions"], "scenario challenge must append at least one world"
        assert len(challenge["additions"]) <= 12 - len(scenarios), "scenario world cap exceeded"
        challenge_sha = scenario_expansion_challenge_digest(data, challenge, action_ids)
        expanded = copy.deepcopy(data)
        base_ids = {world["scenario_id"] for world in scenarios}
        added_ids = set()
        for addition in challenge["additions"]:
            scenario = addition["scenario"]
            support = addition["support"]
            scenario_id = scenario["scenario_id"]
            assert scenario_id not in base_ids and scenario_id not in added_ids, "duplicate/invalid new world"
            assert scenario_id == support["scenario_id"], "scenario/support identity mismatch"
            assert scenario["scenario_id"].strip()
            assert valid_sha(scenario["graph_sha256"], 64)
            assert valid_sha(support["source_evidence_sha256"], 64)
            assert valid_sha(support["joint_observation_model_sha256"], 64)
            assert len(support["dependence_groups"]) == len(set(support["dependence_groups"]))
            assert all(group.strip() for group in support["dependence_groups"])
            normalized = normalized_rows(support, action_ids)
            assert 0 < len(normalized) <= 64
            added_ids.add(scenario_id)
            expanded["scenarios"].append(scenario)
            expanded["joint_world_supports"].append(support)
        assert len(expanded["scenarios"]) <= 12, "world cap exceeded"
        expanded["scenario_set_scope"] = "sampled_sensitivity_set"
        expanded["joint_support_scope"] = "sampled_sensitivity_support"
        expanded["scenario_set_sha256"] = scenario_digest(expanded["scenarios"])
        expanded["joint_support_sha256"] = joint_digest(
            expanded["scenarios"], expanded["actions"], expanded["joint_world_supports"]
        )
        expanded_scenarios = sorted(expanded["scenarios"], key=lambda row: row["scenario_id"])
        expanded_actions = sorted(expanded["actions"], key=lambda row: row["action_id"])
        expanded_action_ids = [row["action_id"] for row in expanded_actions]
        support_by_world = {row["scenario_id"]: row for row in expanded["joint_world_supports"]}
        expanded_rows = [
            normalized_rows(support_by_world[world["scenario_id"]], expanded_action_ids)
            for world in expanded_scenarios
        ]
        expanded_solution, _states = adaptive_solution(
            expanded_actions, expanded_scenarios, expanded_rows
        )
        expanded_fixed = fixed_plan_summary(
            expanded_actions, expanded_scenarios, expanded_rows, data["max_total_effort_units"]
        )
        adaptive_before = None if baseline_solution is None else baseline_solution["worst_case_cost"]
        adaptive_after = None if expanded_solution is None else expanded_solution["worst_case_cost"]
        fixed_before = baseline_fixed["minimum_effort_units"]
        fixed_after = expanded_fixed["minimum_effort_units"]
        adaptive_monotone = effort_non_decreasing(adaptive_before, adaptive_after)
        fixed_monotone = effort_non_decreasing(fixed_before, fixed_after)
        results.append({
            "challenge_id": challenge_id,
            "challenge_sha256": challenge_sha,
            "expanded_scenario_set_sha256": expanded["scenario_set_sha256"],
            "expanded_joint_support_sha256": expanded["joint_support_sha256"],
            "added_scenario_count": len(added_ids),
            "adaptive_worst_case_effort_before": adaptive_before,
            "adaptive_worst_case_effort_after": adaptive_after,
            "adaptive_status_before": "POLICY_EXISTS" if baseline_solution is not None else "NO_POLICY",
            "adaptive_status_after": "POLICY_EXISTS" if expanded_solution is not None else "NO_POLICY",
            "adaptive_effort_non_decreasing": adaptive_monotone,
            "fixed_minimum_effort_before": fixed_before,
            "fixed_minimum_effort_after": fixed_after,
            "fixed_status_before": "FIXED_PLAN_EXISTS" if fixed_before is not None else "NO_FIXED_PLAN",
            "fixed_status_after": "FIXED_PLAN_EXISTS" if fixed_after is not None else "NO_FIXED_PLAN",
            "fixed_effort_non_decreasing": fixed_monotone,
            "monotonicity_passed": adaptive_monotone and fixed_monotone,
        })
    return results


def build_support_challenges(
    data: dict,
    scenarios: list[dict],
    actions: list[dict],
    tuple_rows: list[list[tuple[str, ...]]],
    baseline_solution: dict | None,
    baseline_fixed: dict,
) -> list[dict]:
    action_ids = [row["action_id"] for row in actions]
    world_index = {row["scenario_id"]: index for index, row in enumerate(scenarios)}
    seen_challenges = set()
    results = []
    for challenge in data.get("support_challenges", []):
        challenge_id = challenge["challenge_id"]
        assert challenge_id not in seen_challenges, "duplicate support challenge ID"
        seen_challenges.add(challenge_id)
        assert challenge_id.strip(), "empty support challenge ID"
        assert len(challenge["challenger_evidence_sha256"]) == 64
        assert len(challenge["challenger_model_sha256"]) == 64
        challenge_sha, additions = support_challenge_digest(
            data["joint_support_sha256"], challenge, action_ids
        )
        expanded_rows = [list(rows) for rows in tuple_rows]
        seen_worlds = set()
        added_count = 0
        for scenario_id, rows in additions:
            assert scenario_id in world_index, "challenge references unknown world"
            assert scenario_id not in seen_worlds, "challenge repeats world"
            seen_worlds.add(scenario_id)
            index = world_index[scenario_id]
            existing = set(expanded_rows[index])
            assert all(row not in existing for row in rows), "challenge adds tuple already in base support"
            assert len(expanded_rows[index]) + len(rows) <= 64, "expanded world support exceeds tuple cap"
            expanded_rows[index].extend(rows)
            expanded_rows[index].sort()
            added_count += len(rows)

        expanded_hash = hashlib.sha256(
            b"connectome-adaptive-expanded-joint-support-v1"
            + data["joint_support_sha256"].encode("ascii")
            + challenge_sha.encode("ascii")
        ).hexdigest()
        expanded_solution, _states = adaptive_solution(actions, scenarios, expanded_rows)
        expanded_fixed = fixed_plan_summary(
            actions, scenarios, expanded_rows, data["max_total_effort_units"]
        )
        adaptive_before = None if baseline_solution is None else baseline_solution["worst_case_cost"]
        adaptive_after = None if expanded_solution is None else expanded_solution["worst_case_cost"]
        fixed_before = baseline_fixed["minimum_effort_units"]
        fixed_after = expanded_fixed["minimum_effort_units"]
        adaptive_monotone = effort_non_decreasing(adaptive_before, adaptive_after)
        fixed_monotone = effort_non_decreasing(fixed_before, fixed_after)
        results.append({
            "challenge_id": challenge_id,
            "challenge_sha256": challenge_sha,
            "expanded_joint_support_sha256": expanded_hash,
            "added_joint_tuple_count": added_count,
            "adaptive_worst_case_effort_before": adaptive_before,
            "adaptive_worst_case_effort_after": adaptive_after,
            "adaptive_status_before": "POLICY_EXISTS" if baseline_solution is not None else "NO_POLICY",
            "adaptive_status_after": "POLICY_EXISTS" if expanded_solution is not None else "NO_POLICY",
            "adaptive_effort_non_decreasing": adaptive_monotone,
            "fixed_minimum_effort_before": fixed_before,
            "fixed_minimum_effort_after": fixed_after,
            "fixed_status_before": "FIXED_PLAN_EXISTS" if fixed_before is not None else "NO_FIXED_PLAN",
            "fixed_status_after": "FIXED_PLAN_EXISTS" if fixed_after is not None else "NO_FIXED_PLAN",
            "fixed_effort_non_decreasing": fixed_monotone,
            "monotonicity_passed": adaptive_monotone and fixed_monotone,
        })
    return results


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
    solution, state_counter = adaptive_solution(actions, scenarios, tuple_rows)
    tuple_count = sum(len(rows) for rows in tuple_rows)
    fixed_summary = fixed_plan_summary(actions, scenarios, tuple_rows, data["max_total_effort_units"])
    if fixed_summary["minimum_effort_units"] is not None and solution is not None:
        fixed_summary["adaptive_worst_case_savings_units"] = (
            fixed_summary["minimum_effort_units"] - solution["worst_case_cost"]
        )
    support_challenges = build_support_challenges(
        data, scenarios, actions, tuple_rows, solution, fixed_summary
    )
    scenario_challenges = build_scenario_challenges(
        data, scenarios, actions, solution, fixed_summary
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
        optimal_policy_count = 0
        policy_nodes = policy_branches = 0
        status = "NO_ADAPTIVE_POLICY_WITHIN_DECLARED_ACTIONS"
    else:
        worst = solution["worst_case_cost"]
        optimal_policy_count = solution["optimal_policy_count"]
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

    support_expansion_monotonicity_passed = all(
        challenge["monotonicity_passed"] for challenge in support_challenges
    )
    hypothesis_expansion_monotonicity_passed = all(
        challenge["monotonicity_passed"] for challenge in scenario_challenges
    )
    if not support_expansion_monotonicity_passed:
        reported_status = "SUPPORT_EXPANSION_MONOTONICITY_VIOLATION"
    elif not hypothesis_expansion_monotonicity_passed:
        reported_status = "HYPOTHESIS_EXPANSION_MONOTONICITY_VIOLATION"
    else:
        reported_status = status

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
        "optimal_policy_count": optimal_policy_count,
        "minimum_terminal_path_effort_units": minimum_path,
        "maximum_terminal_path_effort_units": maximum_path,
        "worst_case_policy_within_budget": within,
        "fixed_non_adaptive_optimum": fixed_summary,
        "support_challenge_count": len(support_challenges),
        "support_challenges": support_challenges,
        "support_expansion_monotonicity_passed": support_expansion_monotonicity_passed,
        "scenario_challenge_count": len(scenario_challenges),
        "scenario_challenges": scenario_challenges,
        "hypothesis_expansion_monotonicity_passed": hypothesis_expansion_monotonicity_passed,
        "policy": policy,
        "out_of_model_observation_disposition": "FAIL_CLOSED_NO_QUERY_DECISION",
        "expected_cost_claim": "NONE_NO_PRIOR_OR_EXPECTED_COST_COMPUTED",
        "probability_claim": "NONE_FINITE_SET_VALUED_CORRELATED_SUPPORT_ONLY",
        "status": reported_status,
        "search_complete": True,
        "scope_note": "The policy conditions the original joint tuple relations after each observed action outcome. It stops only when all compatible worlds agree on the frozen query decision, not when the exact world is identified. It does not factorize marginals, infer independence, assign probabilities, prove joint-support completeness, validate measurement models, or establish biological truth. Optional support challenges are additive-only sensitivity checks; scenario challenges append new worlds without altering the base universe. A monotonicity pass does not prove scenario-space completeness. An observation category outside every declared compatible tuple is out-of-model and must fail closed.",
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
