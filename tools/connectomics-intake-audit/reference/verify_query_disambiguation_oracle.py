#!/usr/bin/env python3
"""Independent exact set-cover oracle for query-disambiguation observation plans.

The oracle uses itertools.combinations rather than the Rust bit-mask set-cover
search and recomputes scenario/action identities and all decision-conflict pairs.
It treats per-world outcomes as declared deterministic categories; it assigns no
probabilities, entropies or expected values.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import struct
import sys
from pathlib import Path
from typing import Any


def fail(message: str) -> None:
    raise AssertionError(message)


def length_prefixed(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack(">Q", len(encoded)) + encoded


def canonical_scenario_set_sha256(scenarios: list[dict[str, Any]]) -> str:
    rows = sorted(scenarios, key=lambda row: row["scenario_id"])
    digest = hashlib.sha256()
    digest.update(b"connectome-query-disambiguation-scenario-set-v1")
    digest.update(struct.pack(">Q", len(rows)))
    for scenario in rows:
        digest.update(b"WORLD")
        digest.update(length_prefixed(scenario["scenario_id"]))
        digest.update(scenario["graph_sha256"].encode("ascii"))
        digest.update(bytes([1 if scenario["decision_met"] else 0]))
    return digest.hexdigest()


def canonical_action_set_sha256(actions: list[dict[str, Any]]) -> str:
    rows = sorted(actions, key=lambda row: row["action_id"])
    digest = hashlib.sha256()
    digest.update(b"connectome-query-disambiguation-action-set-v1")
    digest.update(struct.pack(">Q", len(rows)))
    for action in rows:
        digest.update(b"ACTION")
        digest.update(length_prefixed(action["action_id"]))
        digest.update(length_prefixed(action["action_kind"]))
        digest.update(struct.pack(">Q", action["effort_units"]))
        digest.update(action["source_evidence_sha256"].encode("ascii"))
        digest.update(action["observation_model_sha256"].encode("ascii"))
        groups = sorted(action["dependence_groups"])
        digest.update(struct.pack(">Q", len(groups)))
        for group in groups:
            digest.update(b"DEPENDENCE")
            digest.update(length_prefixed(group))
        outcomes = sorted(
            action["outcomes"],
            key=lambda row: (row["scenario_id"], row["outcome_id"]),
        )
        digest.update(struct.pack(">Q", len(outcomes)))
        for outcome in outcomes:
            digest.update(b"OUTCOME")
            digest.update(length_prefixed(outcome["scenario_id"]))
            digest.update(length_prefixed(outcome["outcome_id"]))
    return digest.hexdigest()


def validate_input(source: dict[str, Any]) -> None:
    scenarios = source["scenarios"]
    actions = source["actions"]
    if not (2 <= len(scenarios) <= 24):
        fail("scenario count outside the V1 exact pair-analysis bound")
    if not (1 <= len(actions) <= 20):
        fail("action count outside the V1 exact set-cover bound")
    if len({row["scenario_id"] for row in scenarios}) != len(scenarios):
        fail("duplicate scenario IDs")
    if len({row["action_id"] for row in actions}) != len(actions):
        fail("duplicate action IDs")
    scenario_ids = {row["scenario_id"] for row in scenarios}
    for scenario in scenarios:
        if len(scenario["graph_sha256"]) != 64:
            fail("malformed graph digest")
    if source["scenario_set_sha256"] != canonical_scenario_set_sha256(scenarios):
        fail("scenario-set digest mismatch")
    for action in actions:
        if action["effort_units"] <= 0:
            fail(f"action {action['action_id']} has non-positive effort")
        if len(action["source_evidence_sha256"]) != 64 or len(action["observation_model_sha256"]) != 64:
            fail(f"action {action['action_id']} has malformed provenance digest")
        if len(set(action["dependence_groups"])) != len(action["dependence_groups"]):
            fail(f"action {action['action_id']} has duplicate dependence groups")
        outcomes = action["outcomes"]
        ids = [row["scenario_id"] for row in outcomes]
        if set(ids) != scenario_ids or len(ids) != len(scenario_ids):
            fail(f"action {action['action_id']} does not provide exactly one outcome per world")
        if any(not row["outcome_id"] for row in outcomes):
            fail(f"action {action['action_id']} has an empty outcome label")


def greedy_plan(
    actions: list[dict[str, Any]],
    action_coverage: dict[str, set[tuple[str, str]]],
    conflict_pairs: set[tuple[str, str]],
    exact_minimum_cost: int | None,
) -> dict[str, Any]:
    uncovered = set(conflict_pairs)
    selected: list[str] = []
    effort = 0
    while uncovered:
        candidates = []
        for action in actions:
            action_id = action["action_id"]
            if action_id in selected:
                continue
            marginal = len(uncovered & action_coverage[action_id])
            if marginal == 0:
                continue
            candidates.append((action, marginal))
        if not candidates:
            break
        best_action, best_marginal = candidates[0]
        for action, marginal in candidates[1:]:
            left = marginal * best_action["effort_units"]
            right = best_marginal * action["effort_units"]
            if (
                left > right
                or (
                    left == right
                    and (
                        marginal > best_marginal
                        or (
                            marginal == best_marginal
                            and action["action_id"] < best_action["action_id"]
                        )
                    )
                )
            ):
                best_action, best_marginal = action, marginal
        selected.append(best_action["action_id"])
        effort += best_action["effort_units"]
        uncovered -= action_coverage[best_action["action_id"]]

    selected.sort()
    complete = not uncovered
    return {
        "action_ids": selected,
        "effort_units": effort,
        "separated_decision_conflict_pair_count": len(conflict_pairs) - len(uncovered),
        "remaining_pairs": [
            {"world_a": left, "world_b": right}
            for left, right in sorted(uncovered)
        ],
        "complete": complete,
        "regret_to_exact_minimum_effort_units": (
            None
            if not complete or exact_minimum_cost is None
            else max(0, effort - exact_minimum_cost)
        ),
    }


def expected_result(source: dict[str, Any]) -> dict[str, Any]:
    validate_input(source)
    scenarios = sorted(source["scenarios"], key=lambda row: row["scenario_id"])
    actions = sorted(source["actions"], key=lambda row: row["action_id"])
    outcomes_by_action = {
        action["action_id"]: {
            row["scenario_id"]: row["outcome_id"] for row in action["outcomes"]
        }
        for action in actions
    }

    conflict_pairs = [
        (left["scenario_id"], right["scenario_id"])
        for left, right in itertools.combinations(scenarios, 2)
        if left["decision_met"] != right["decision_met"]
    ]
    action_coverage: dict[str, set[tuple[str, str]]] = {}
    for action in actions:
        outcomes = outcomes_by_action[action["action_id"]]
        action_coverage[action["action_id"]] = {
            pair for pair in conflict_pairs if outcomes[pair[0]] != outcomes[pair[1]]
        }
    covered_by_any = set().union(*action_coverage.values()) if action_coverage else set()
    undiscriminable = sorted(set(conflict_pairs) - covered_by_any)

    coverage_rows = [
        {
            "action_id": action["action_id"],
            "effort_units": action["effort_units"],
            "separated_decision_conflict_pair_count": len(action_coverage[action["action_id"]]),
        }
        for action in actions
    ]

    if not conflict_pairs:
        status = "NO_DECISION_DISAGREEMENT_WITHIN_DECLARED_SCENARIOS"
        subset_count = 0
        minimum_cost = None
        plans: list[dict[str, Any]] = []
        minimum_count = 0
    elif undiscriminable:
        status = "NO_COMPLETE_DISAMBIGUATION_WITHIN_DECLARED_ACTIONS"
        subset_count = 0
        minimum_cost = None
        plans = []
        minimum_count = 0
    else:
        best_cost: int | None = None
        best_plans: list[dict[str, Any]] = []
        subset_count = (1 << len(actions)) - 1
        for size in range(1, len(actions) + 1):
            for selected in itertools.combinations(actions, size):
                selected_ids = [action["action_id"] for action in selected]
                coverage = set().union(*(action_coverage[action_id] for action_id in selected_ids))
                if not set(conflict_pairs).issubset(coverage):
                    continue
                cost = sum(action["effort_units"] for action in selected)
                if best_cost is None or cost < best_cost:
                    best_cost = cost
                    best_plans = []
                if cost == best_cost:
                    best_plans.append(
                        {
                            "action_ids": sorted(selected_ids),
                            "effort_units": cost,
                            "separated_decision_conflict_pair_count": len(conflict_pairs),
                        }
                    )
        best_plans.sort(key=lambda row: row["action_ids"])
        minimum_cost = best_cost
        minimum_count = len(best_plans)
        plans = best_plans[:128]
        status = (
            "MINIMUM_COST_DISAMBIGUATION_PLAN"
            if best_cost is not None and best_cost <= source["max_total_effort_units"]
            else "MINIMUM_COST_DISAMBIGUATION_EXCEEDS_BUDGET"
        )

    pair_rows = [
        {
            "world_a": left,
            "world_b": right,
        }
        for left, right in sorted(conflict_pairs)
    ]
    undiscriminable_rows = [
        {"world_a": left, "world_b": right} for left, right in undiscriminable
    ]
    greedy = greedy_plan(actions, action_coverage, set(conflict_pairs), minimum_cost)
    return {
        "scenario_count": len(scenarios),
        "observation_action_count": len(actions),
        "greedy_comparator": greedy,
        "decision_conflict_pair_count": len(conflict_pairs),
        "individually_discriminable_pair_count": len(conflict_pairs) - len(undiscriminable),
        "undiscriminable_pairs": undiscriminable_rows,
        "action_pair_coverage": coverage_rows,
        "subset_count_evaluated": subset_count,
        "minimum_disambiguation_effort_units": minimum_cost,
        "minimum_plan_count": minimum_count,
        "plans_returned": len(plans),
        "minimum_plans_truncated": minimum_count > 128,
        "minimum_plan_within_budget": (
            None if minimum_cost is None else minimum_cost <= source["max_total_effort_units"]
        ),
        "plans": plans,
        "status": status,
        "search_complete": True,
        "probability_claim": "NONE_DETERMINISTIC_WORLD_DISCRIMINATION_ONLY",
    }


def verify_report(input_bytes: bytes, source: dict[str, Any], report: dict[str, Any]) -> None:
    expected = expected_result(source)
    if report.get("schema_version") != 1:
        fail("report schema_version mismatch")
    if report.get("tool") != "connectome-query-disambiguation-audit":
        fail("report tool identity mismatch")
    if report.get("input_sha256") != hashlib.sha256(input_bytes).hexdigest():
        fail("report does not bind exact input bytes")
    for key in (
        "dataset_revision_commit_sha",
        "query_id",
        "query_spec_sha256",
        "scenario_set_scope",
        "scenario_set_sha256",
        "effort_unit",
        "max_total_effort_units",
    ):
        input_key = key
        if report.get(key) != source.get(input_key):
            fail(f"{key} input binding mismatch")
    if report.get("action_set_sha256") != canonical_action_set_sha256(source["actions"]):
        fail("canonical observation-action set digest mismatch")
    for key, value in expected.items():
        if report.get(key) != value:
            fail(f"{key} mismatch: expected {value!r}, got {report.get(key)!r}")


def negative_controls(input_bytes: bytes, source: dict[str, Any], report: dict[str, Any]) -> int:
    mutations = []
    if report.get("plans"):
        changed_cost = copy.deepcopy(report)
        changed_cost["minimum_disambiguation_effort_units"] += 1
        mutations.append(changed_cost)

        changed_plan = copy.deepcopy(report)
        changed_plan["plans"][0]["action_ids"].pop()
        mutations.append(changed_plan)

        changed_coverage = copy.deepcopy(report)
        changed_coverage["action_pair_coverage"][0]["separated_decision_conflict_pair_count"] += 1
        mutations.append(changed_coverage)
    else:
        changed_status = copy.deepcopy(report)
        changed_status["status"] = "MINIMUM_COST_DISAMBIGUATION_PLAN"
        mutations.append(changed_status)

    detected = 0
    for mutated in mutations:
        try:
            verify_report(input_bytes, source, mutated)
        except AssertionError:
            detected += 1
        else:
            fail("oracle accepted deliberately corrupted disambiguation report")
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
                "status": "INDEPENDENT_DISAMBIGUATION_ORACLE_AGREEMENT",
                "oracle": "python-standard-library-combinations-set-cover-v1",
                "decision_conflict_pair_count": report["decision_conflict_pair_count"],
                "minimum_plan_count": report["minimum_plan_count"],
                "minimum_disambiguation_effort_units": report["minimum_disambiguation_effort_units"],
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
        print(f"independent disambiguation oracle failed: {error}", file=sys.stderr)
        raise SystemExit(1)
