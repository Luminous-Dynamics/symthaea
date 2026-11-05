#!/usr/bin/env python3
"""Independent joint-support oracle using projections and combinations."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path


MAX_RETURNED_PLANS = 128
MAX_SUBSET_MARK_STEPS = 50_000_000


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
    for row in rows:
        h.update(b"WORLD")
        hash_field(h, row["scenario_id"])
        h.update(row["graph_sha256"].encode("ascii"))
        h.update(bytes([int(row["decision_met"])]))
    return h.hexdigest()


def action_digest(actions: list[dict]) -> str:
    rows = sorted(actions, key=lambda row: row["action_id"])
    h = hashlib.sha256()
    h.update(b"connectome-joint-observation-action-set-v1")
    h.update(len(rows).to_bytes(8, "big"))
    for row in rows:
        h.update(b"ACTION")
        hash_field(h, row["action_id"])
        hash_field(h, row["action_kind"])
        h.update(row["effort_units"].to_bytes(8, "big"))
        h.update(row["source_evidence_sha256"].encode("ascii"))
        groups = sorted(row["dependence_groups"])
        h.update(len(groups).to_bytes(8, "big"))
        for group in groups:
            h.update(b"DEPENDENCE")
            hash_field(h, group)
    return h.hexdigest()


def joint_support_digest(supports: list[dict]) -> str:
    worlds = sorted(supports, key=lambda row: row["scenario_id"])
    h = hashlib.sha256()
    h.update(b"connectome-joint-observation-support-set-v1")
    h.update(len(worlds).to_bytes(8, "big"))
    for world in worlds:
        h.update(b"SUPPORT")
        hash_field(h, world["scenario_id"])
        h.update(world["joint_observation_model_sha256"].encode("ascii"))
        h.update(world["source_evidence_sha256"].encode("ascii"))
        groups = sorted(world["dependence_groups"])
        h.update(len(groups).to_bytes(8, "big"))
        for group in groups:
            h.update(b"DEPENDENCE")
            hash_field(h, group)
        rows = [
            sorted(row["outcomes"], key=lambda outcome: outcome["action_id"])
            for row in world["rows"]
        ]
        rows.sort(
            key=lambda row: tuple(
                (outcome["action_id"], outcome["value"]) for outcome in row
            )
        )
        h.update(len(rows).to_bytes(8, "big"))
        for row in rows:
            h.update(b"JOINT-ROW")
            h.update(len(row).to_bytes(8, "big"))
            for outcome in row:
                h.update(b"OUTCOME")
                hash_field(h, outcome["action_id"])
                hash_field(h, outcome["value"])
    return h.hexdigest()


def conflict_pairs(scenarios: list[dict]) -> list[dict]:
    worlds = sorted(scenarios, key=lambda row: row["scenario_id"])
    return [
        {"world_a": left["scenario_id"], "world_b": right["scenario_id"]}
        for index, left in enumerate(worlds)
        for right in worlds[index + 1 :]
        if left["decision_met"] != right["decision_met"]
    ]


def row_map(world_support: dict) -> list[dict[str, str]]:
    return [
        {outcome["action_id"]: outcome["value"] for outcome in row["outcomes"]}
        for row in world_support["rows"]
    ]


def projected_support(world_support: dict, selected_ids: tuple[str, ...]) -> set[tuple[str, ...]]:
    return {
        tuple(row[action_id] for action_id in selected_ids)
        for row in row_map(world_support)
    }


def pair_separated(left_support: dict, right_support: dict, selected_ids: tuple[str, ...]) -> bool:
    return projected_support(left_support, selected_ids).isdisjoint(
        projected_support(right_support, selected_ids)
    )


def calculate(base: dict, raw: bytes) -> dict:
    assert base["schema_version"] == 1
    assert 1 <= len(base["actions"]) <= 20
    assert 2 <= len(base["scenarios"]) <= 24
    assert scenario_digest(base["scenarios"]) == base["scenario_set_sha256"]
    assert action_digest(base["actions"]) == base["action_set_sha256"]
    assert joint_support_digest(base["joint_supports"]) == base["joint_support_set_sha256"]

    actions = sorted(base["actions"], key=lambda row: row["action_id"])
    scenarios = sorted(base["scenarios"], key=lambda row: row["scenario_id"])
    supports = sorted(base["joint_supports"], key=lambda row: row["scenario_id"])
    action_ids = [row["action_id"] for row in actions]
    support_by_world = {row["scenario_id"]: row for row in supports}
    pairs = conflict_pairs(scenarios)

    marginal_action_coverage = []
    for action in actions:
        action_id = action["action_id"]
        separated = []
        for pair in pairs:
            left = support_by_world[pair["world_a"]]
            right = support_by_world[pair["world_b"]]
            if pair_separated(left, right, (action_id,)):
                separated.append(pair)
        marginal_action_coverage.append(
            {
                "action_id": action_id,
                "effort_units": action["effort_units"],
                "separated_conflict_pairs": separated,
            }
        )

    marginally_separable_pair_count = sum(
        any(pair in row["separated_conflict_pairs"] for row in marginal_action_coverage)
        for pair in pairs
    )
    marginally_undiscriminable_pairs = [
        pair
        for pair in pairs
        if not any(pair in row["separated_conflict_pairs"] for row in marginal_action_coverage)
    ]

    distinct_constraints: set[int] = set()
    unavoidable_shared = []
    for pair in pairs:
        left_rows = row_map(support_by_world[pair["world_a"]])
        right_rows = row_map(support_by_world[pair["world_b"]])
        pair_constraints = set()
        shared_full = False
        for left in left_rows:
            for right in right_rows:
                difference = 0
                for index, action_id in enumerate(action_ids):
                    if left[action_id] != right[action_id]:
                        difference |= 1 << index
                if difference == 0:
                    shared_full = True
                else:
                    pair_constraints.add(difference)
        if shared_full:
            unavoidable_shared.append(pair)
        else:
            distinct_constraints.update(pair_constraints)

    candidate_subsets = (1 << len(actions)) - 1
    subset_mark_steps = 0
    minimal_constraints = []
    min_cost = None
    min_count = 0
    plans = []
    search_complete = True

    if not pairs:
        min_cost = 0
        min_count = 1
        plans = [{"action_ids": [], "effort_units": 0}]
        jointly_undiscriminable_pairs = []
        jointly_separable_pair_count = 0
        status = "NO_DECISION_DISAGREEMENT_WITHIN_DECLARED_SCENARIOS"
    elif unavoidable_shared:
        jointly_undiscriminable_pairs = unavoidable_shared
        jointly_separable_pair_count = len(pairs) - len(unavoidable_shared)
        status = "NO_JOINT_SUPPORT_PLAN_WITHIN_DECLARED_RELATIONS"
    else:
        for constraint in sorted(distinct_constraints, key=lambda value: (value.bit_count(), value)):
            if not any((kept & constraint) == kept for kept in minimal_constraints):
                minimal_constraints.append(constraint)
        subset_mark_steps = sum(
            1 << (len(actions) - constraint.bit_count())
            for constraint in minimal_constraints
        )
        if subset_mark_steps > MAX_SUBSET_MARK_STEPS:
            search_complete = False
            jointly_undiscriminable_pairs = []
            jointly_separable_pair_count = len(pairs)
            status = "JOINT_SUPPORT_SEARCH_LIMIT_REACHED"
        else:
            valid_subsets = []
            for size in range(1, len(actions) + 1):
                for chosen in itertools.combinations(actions, size):
                    selected_ids = tuple(row["action_id"] for row in chosen)
                    if all(
                        pair_separated(
                            support_by_world[pair["world_a"]],
                            support_by_world[pair["world_b"]],
                            selected_ids,
                        )
                        for pair in pairs
                    ):
                        effort = sum(row["effort_units"] for row in chosen)
                        valid_subsets.append((effort, selected_ids))
            if valid_subsets:
                min_cost = min(cost for cost, _ in valid_subsets)
                best = sorted(ids for cost, ids in valid_subsets if cost == min_cost)
                min_count = len(best)
                plans = [
                    {"action_ids": list(ids), "effort_units": min_cost}
                    for ids in best[:MAX_RETURNED_PLANS]
                ]
                jointly_undiscriminable_pairs = []
                jointly_separable_pair_count = len(pairs)
                status = (
                    "MINIMUM_COST_JOINT_ROBUST_PLAN"
                    if min_cost <= base["max_total_effort_units"]
                    else "MINIMUM_COST_JOINT_ROBUST_PLAN_EXCEEDS_BUDGET"
                )
            else:
                jointly_undiscriminable_pairs = pairs
                jointly_separable_pair_count = 0
                status = "NO_JOINT_SUPPORT_PLAN_WITHIN_DECLARED_RELATIONS"

    return {
        "schema_version": 1,
        "tool": "connectome-joint-support-audit",
        "analysis_id": base["analysis_id"],
        "dataset_revision_commit_sha": base["dataset_revision_commit_sha"],
        "query_id": base["query_id"],
        "query_spec_sha256": base["query_spec_sha256"],
        "scenario_set_sha256": base["scenario_set_sha256"],
        "action_set_sha256": base["action_set_sha256"],
        "joint_support_set_sha256": base["joint_support_set_sha256"],
        "input_sha256": sha256(raw),
        "scenario_count": len(scenarios),
        "action_count": len(actions),
        "total_joint_support_rows": sum(len(row["rows"]) for row in supports),
        "joint_support_scope": base["joint_support_scope"],
        "effort_unit": base["effort_unit"],
        "max_total_effort_units": base["max_total_effort_units"],
        "decision_conflict_pair_count": len(pairs),
        "marginal_action_coverage": marginal_action_coverage,
        "marginally_separable_pair_count": marginally_separable_pair_count,
        "marginally_undiscriminable_pairs": marginally_undiscriminable_pairs,
        "jointly_separable_pair_count": jointly_separable_pair_count,
        "jointly_undiscriminable_pairs": jointly_undiscriminable_pairs,
        "unavoidable_shared_full_joint_tuple_pairs": unavoidable_shared,
        "distinct_difference_constraint_count": len(distinct_constraints),
        "minimal_difference_constraint_count": (
            0 if unavoidable_shared or not pairs else len(minimal_constraints)
        ),
        "candidate_action_subsets": candidate_subsets,
        "subset_mark_steps": subset_mark_steps,
        "minimum_joint_disambiguation_effort_units": min_cost,
        "minimum_joint_plan_count": min_count,
        "minimum_joint_plans_returned": len(plans),
        "minimum_joint_plans_truncated": min_count > MAX_RETURNED_PLANS,
        "minimum_joint_plans": plans,
        "minimum_joint_plan_within_budget": (
            None if min_cost is None else min_cost <= base["max_total_effort_units"]
        ),
        "search_complete": search_complete,
        "status": status,
        "probability_claim": "NONE_JOINT_SET_VALUED_DISCRIMINATION_ONLY",
        "scope_note": (
            "The planner evaluates the declared finite joint-support relation directly. "
            "It does not factorize marginal supports, infer independence, infer likelihoods, "
            "or prove support completeness. Results are conditional on the supplied scenarios, "
            "action set, joint support rows, provenance digests and scope declaration. "
            "A digest binds a declaration; it does not prove biological truth or real-world completeness."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        raw = args.input.read_bytes()
        base = json.loads(raw)
        expected = calculate(base, raw)
        actual = json.loads(args.report.read_text(encoding="utf-8"))
        assert actual == expected, "report differs from independent projections/combinations oracle"
        print(
            f"verified {expected['status']}: "
            f"marginally separable={expected['marginally_separable_pair_count']}; "
            f"joint minimum={expected['minimum_joint_disambiguation_effort_units']}; "
            f"ties={expected['minimum_joint_plan_count']}; "
            f"subset-mark-steps={expected['subset_mark_steps']}; "
            f"shared-full-tuples={len(expected['unavoidable_shared_full_joint_tuple_pairs'])}"
        )
        return 0
    except Exception as error:
        print(f"joint-support oracle failure: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
