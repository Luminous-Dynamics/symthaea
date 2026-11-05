#!/usr/bin/env python3
"""Independent exact oracle for robust set-valued observation disambiguation.

Possible outcomes are support sets, not probabilities. An action separates two
worlds robustly only if their possible-outcome sets are disjoint. The oracle
uses itertools.combinations rather than the Rust bit-mask subset search.
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


def lp(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack(">Q", len(encoded)) + encoded


def scenario_digest(scenarios: list[dict[str, Any]]) -> str:
    rows = sorted(scenarios, key=lambda row: row["scenario_id"])
    digest = hashlib.sha256()
    digest.update(b"connectome-query-disambiguation-scenario-set-v1")
    digest.update(struct.pack(">Q", len(rows)))
    for row in rows:
        digest.update(b"WORLD")
        digest.update(lp(row["scenario_id"]))
        digest.update(row["graph_sha256"].encode("ascii"))
        digest.update(bytes([1 if row["decision_met"] else 0]))
    return digest.hexdigest()


def action_digest(actions: list[dict[str, Any]]) -> str:
    rows = sorted(actions, key=lambda row: row["action_id"])
    digest = hashlib.sha256()
    digest.update(b"connectome-robust-query-disambiguation-action-set-v1")
    digest.update(struct.pack(">Q", len(rows)))
    for action in rows:
        digest.update(b"ACTION")
        digest.update(lp(action["action_id"]))
        digest.update(lp(action["action_kind"]))
        digest.update(struct.pack(">Q", action["effort_units"]))
        digest.update(action["source_evidence_sha256"].encode("ascii"))
        digest.update(action["observation_model_sha256"].encode("ascii"))
        groups = sorted(action["dependence_groups"])
        digest.update(struct.pack(">Q", len(groups)))
        for group in groups:
            digest.update(b"DEPENDENCE")
            digest.update(lp(group))
        outcomes = sorted(action["outcomes"], key=lambda row: row["scenario_id"])
        digest.update(struct.pack(">Q", len(outcomes)))
        for outcome in outcomes:
            digest.update(b"WORLD-OUTCOMES")
            digest.update(lp(outcome["scenario_id"]))
            possible = sorted(outcome["possible_outcomes"])
            digest.update(struct.pack(">Q", len(possible)))
            for item in possible:
                digest.update(b"POSSIBLE")
                digest.update(lp(item))
    return digest.hexdigest()


def validate(source: dict[str, Any]) -> None:
    scenarios = source["scenarios"]
    actions = source["actions"]
    if not (2 <= len(scenarios) <= 24):
        fail("scenario count outside exact-analysis bound")
    if not (1 <= len(actions) <= 20):
        fail("action count outside exact-analysis bound")
    scenario_ids = [row["scenario_id"] for row in scenarios]
    if len(set(scenario_ids)) != len(scenario_ids):
        fail("duplicate scenario IDs")
    if len({row["action_id"] for row in actions}) != len(actions):
        fail("duplicate action IDs")
    if scenario_digest(scenarios) != source["scenario_set_sha256"]:
        fail("scenario-set digest mismatch")
    scenario_id_set = set(scenario_ids)
    for scenario in scenarios:
        if len(scenario["graph_sha256"]) != 64:
            fail("malformed graph SHA-256")
    for action in actions:
        if action["effort_units"] <= 0:
            fail("action effort must be positive")
        if len(action["source_evidence_sha256"]) != 64 or len(action["observation_model_sha256"]) != 64:
            fail("malformed action provenance digest")
        if len(set(action["dependence_groups"])) != len(action["dependence_groups"]):
            fail("duplicate dependence group")
        outcome_rows = action["outcomes"]
        outcome_ids = [row["scenario_id"] for row in outcome_rows]
        if set(outcome_ids) != scenario_id_set or len(outcome_ids) != len(scenario_ids):
            fail("each action requires exactly one possible-outcome set per world")
        for row in outcome_rows:
            possible = row["possible_outcomes"]
            if not possible or any(not item for item in possible):
                fail("possible-outcome support set must be non-empty")
            if len(set(possible)) != len(possible):
                fail("duplicate possible-outcome category")


def expected(source: dict[str, Any]) -> dict[str, Any]:
    validate(source)
    scenarios = sorted(source["scenarios"], key=lambda row: row["scenario_id"])
    actions = sorted(source["actions"], key=lambda row: row["action_id"])
    outcome_maps = {
        action["action_id"]: {
            row["scenario_id"]: set(row["possible_outcomes"])
            for row in action["outcomes"]
        }
        for action in actions
    }
    conflict_pairs = [
        (left["scenario_id"], right["scenario_id"])
        for left, right in itertools.combinations(scenarios, 2)
        if left["decision_met"] != right["decision_met"]
    ]
    action_coverage = {
        action["action_id"]: {
            pair
            for pair in conflict_pairs
            if outcome_maps[action["action_id"]][pair[0]].isdisjoint(
                outcome_maps[action["action_id"]][pair[1]]
            )
        }
        for action in actions
    }
    all_covered = set().union(*action_coverage.values()) if action_coverage else set()
    undiscriminable = sorted(set(conflict_pairs) - all_covered)

    coverage_rows = [
        {
            "action_id": action["action_id"],
            "effort_units": action["effort_units"],
            "robustly_separated_decision_conflict_pair_count": len(
                action_coverage[action["action_id"]]
            ),
        }
        for action in actions
    ]
    if not conflict_pairs:
        subset_count = 0
        min_cost = None
        min_count = 0
        plans = []
        status = "NO_DECISION_DISAGREEMENT_WITHIN_DECLARED_SCENARIOS"
    elif undiscriminable:
        subset_count = 0
        min_cost = None
        min_count = 0
        plans = []
        status = "NO_ROBUST_DISAMBIGUATION_WITHIN_DECLARED_ACTIONS"
    else:
        subset_count = (1 << len(actions)) - 1
        min_cost = None
        plans = []
        for size in range(1, len(actions) + 1):
            for selected in itertools.combinations(actions, size):
                ids = [row["action_id"] for row in selected]
                covered = set().union(*(action_coverage[action_id] for action_id in ids))
                if not set(conflict_pairs).issubset(covered):
                    continue
                cost = sum(row["effort_units"] for row in selected)
                if min_cost is None or cost < min_cost:
                    min_cost = cost
                    plans = []
                if cost == min_cost:
                    plans.append(
                        {
                            "action_ids": sorted(ids),
                            "effort_units": cost,
                            "robustly_separated_decision_conflict_pair_count": len(conflict_pairs),
                        }
                    )
        plans.sort(key=lambda row: row["action_ids"])
        min_count = len(plans)
        plans = plans[:128]
        status = (
            "MINIMUM_COST_ROBUST_DISAMBIGUATION_PLAN"
            if min_cost is not None and min_cost <= source["max_total_effort_units"]
            else "MINIMUM_COST_ROBUST_DISAMBIGUATION_EXCEEDS_BUDGET"
        )
    return {
        "scenario_count": len(scenarios),
        "observation_action_count": len(actions),
        "decision_conflict_pair_count": len(conflict_pairs),
        "robustly_separable_pair_count": len(conflict_pairs) - len(undiscriminable),
        "robustly_undiscriminable_pairs": [
            {"world_a": a, "world_b": b} for a, b in undiscriminable
        ],
        "action_pair_coverage": coverage_rows,
        "subset_count_evaluated": subset_count,
        "minimum_disambiguation_effort_units": min_cost,
        "minimum_plan_count": min_count,
        "plans_returned": len(plans),
        "minimum_plans_truncated": min_count > 128,
        "minimum_plan_within_budget": None if min_cost is None else min_cost <= source["max_total_effort_units"],
        "plans": plans,
        "status": status,
        "search_complete": True,
        "probability_claim": "NONE_SET_VALUED_ROBUST_DISCRIMINATION_ONLY",
    }


def verify(input_bytes: bytes, source: dict[str, Any], report: dict[str, Any]) -> None:
    result = expected(source)
    if report.get("schema_version") != 1:
        fail("schema version mismatch")
    if report.get("tool") != "connectome-robust-query-disambiguation-audit":
        fail("tool identity mismatch")
    if report.get("input_sha256") != hashlib.sha256(input_bytes).hexdigest():
        fail("exact input digest mismatch")
    for key in (
        "dataset_revision_commit_sha",
        "query_id",
        "query_spec_sha256",
        "scenario_set_scope",
        "scenario_set_sha256",
        "effort_unit",
        "max_total_effort_units",
    ):
        if report.get(key) != source[key]:
            fail(f"{key} binding mismatch")
    if report.get("action_set_sha256") != action_digest(source["actions"]):
        fail("action-set digest mismatch")
    if report.get("robust_separation_rule") != "POSSIBLE_OUTCOME_SETS_DISJOINT":
        fail("robust separation rule mismatch")
    for key, value in result.items():
        if report.get(key) != value:
            fail(f"{key} mismatch: expected {value!r}, got {report.get(key)!r}")


def negative_controls(input_bytes: bytes, source: dict[str, Any], report: dict[str, Any]) -> int:
    mutations = []
    if report.get("plans"):
        cost = copy.deepcopy(report)
        cost["minimum_disambiguation_effort_units"] += 1
        mutations.append(cost)

        coverage = copy.deepcopy(report)
        coverage["action_pair_coverage"][0]["robustly_separated_decision_conflict_pair_count"] += 1
        mutations.append(coverage)

        member = copy.deepcopy(report)
        member["plans"][0]["action_ids"].pop()
        mutations.append(member)
    else:
        status = copy.deepcopy(report)
        status["status"] = "MINIMUM_COST_ROBUST_DISAMBIGUATION_PLAN"
        mutations.append(status)

    found = 0
    for mutated in mutations:
        try:
            verify(input_bytes, source, mutated)
        except AssertionError:
            found += 1
        else:
            fail("oracle accepted a deliberately corrupted robust-disambiguation report")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    input_bytes = args.input.read_bytes()
    source = json.loads(input_bytes)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    verify(input_bytes, source, report)
    detected = negative_controls(input_bytes, source, report)
    print(
        json.dumps(
            {
                "status": "INDEPENDENT_ROBUST_DISAMBIGUATION_ORACLE_AGREEMENT",
                "oracle": "python-standard-library-combinations-set-valued-cover-v1",
                "conflict_pairs": report["decision_conflict_pair_count"],
                "minimum_plan_count": report["minimum_plan_count"],
                "minimum_cost": report["minimum_disambiguation_effort_units"],
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
        print(f"independent robust disambiguation oracle failed: {error}", file=sys.stderr)
        raise SystemExit(1)
