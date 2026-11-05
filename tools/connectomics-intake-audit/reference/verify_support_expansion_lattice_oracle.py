#!/usr/bin/env python3
"""Independent set-based oracle for the exhaustive connectome support lattice."""
from __future__ import annotations

import argparse
import functools
import itertools
import json
import sys
from pathlib import Path

ACTIONS = ("inspect-a", "inspect-b")
ACTION_COSTS = (1, 2)
TUPLES = tuple(itertools.product((0, 1), repeat=2))
SUPPORTS = tuple(
    frozenset(rows)
    for size in range(1, len(TUPLES) + 1)
    for rows in itertools.combinations(TUPLES, size)
)
SUPERSETS = {
    base: tuple(expanded for expanded in SUPPORTS if base.issubset(expanded))
    for base in SUPPORTS
}


@functools.lru_cache(maxsize=None)
def adaptive_cost(
    remaining_actions: tuple[int, ...],
    world_supports: tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]],
) -> int | None:
    # The two worlds have opposite frozen query decisions. A leaf is valid
    # exactly when one world has no compatible joint tuple.
    active = tuple(bool(support) for support in world_supports)
    if active[0] != active[1]:
        return 0
    if not remaining_actions:
        return None

    best = None
    for action_index in remaining_actions:
        observed = sorted({
            row[action_index]
            for support in world_supports
            for row in support
        })
        children = []
        valid = True
        next_actions = tuple(index for index in remaining_actions if index != action_index)
        for outcome in observed:
            child_supports = tuple(
                frozenset(row for row in support if row[action_index] == outcome)
                for support in world_supports
            )
            child_cost = adaptive_cost(next_actions, child_supports)
            if child_cost is None:
                valid = False
                break
            children.append(child_cost)
        if valid and children:
            candidate = ACTION_COSTS[action_index] + max(children)
            if best is None or candidate < best:
                best = candidate
    return best


def fixed_cost(
    world_supports: tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]
) -> int | None:
    best = None
    for size in range(1, len(ACTIONS) + 1):
        for selected in itertools.combinations(range(len(ACTIONS)), size):
            left = {
                tuple(row[index] for index in selected)
                for row in world_supports[0]
            }
            right = {
                tuple(row[index] for index in selected)
                for row in world_supports[1]
            }
            if left.isdisjoint(right):
                cost = sum(ACTION_COSTS[index] for index in selected)
                if best is None or cost < best:
                    best = cost
    return best


def non_decreasing(before: int | None, after: int | None) -> bool:
    if before is None:
        return after is None
    return after is None or after >= before


def evaluate() -> dict:
    counters = {
        "adaptive_feasible_base_cases": 0,
        "adaptive_feasible_expanded_cases": 0,
        "adaptive_cost_unchanged_cases": 0,
        "adaptive_cost_increased_cases": 0,
        "adaptive_policy_lost_cases": 0,
        "adaptive_monotonicity_violations": 0,
        "adaptive_policy_resurrections": 0,
        "fixed_feasible_base_cases": 0,
        "fixed_feasible_expanded_cases": 0,
        "fixed_cost_unchanged_cases": 0,
        "fixed_cost_increased_cases": 0,
        "fixed_plan_lost_cases": 0,
        "fixed_monotonicity_violations": 0,
        "fixed_plan_resurrections": 0,
    }
    ordered_pairs = 0
    counterexamples = []
    for base_0 in SUPPORTS:
        for expanded_0 in SUPERSETS[base_0]:
            for base_1 in SUPPORTS:
                for expanded_1 in SUPERSETS[base_1]:
                    base = (base_0, base_1)
                    expanded = (expanded_0, expanded_1)
                    adaptive_before = adaptive_cost((0, 1), base)
                    adaptive_after = adaptive_cost((0, 1), expanded)
                    fixed_before = fixed_cost(base)
                    fixed_after = fixed_cost(expanded)
                    ordered_pairs += 1

                    if adaptive_before is not None:
                        counters["adaptive_feasible_base_cases"] += 1
                    if adaptive_after is not None:
                        counters["adaptive_feasible_expanded_cases"] += 1
                    if adaptive_before is not None and adaptive_after is not None:
                        if adaptive_after == adaptive_before:
                            counters["adaptive_cost_unchanged_cases"] += 1
                        elif adaptive_after > adaptive_before:
                            counters["adaptive_cost_increased_cases"] += 1
                    elif adaptive_before is not None and adaptive_after is None:
                        counters["adaptive_policy_lost_cases"] += 1
                    adaptive_bad = not non_decreasing(adaptive_before, adaptive_after)
                    if adaptive_bad:
                        counters["adaptive_monotonicity_violations"] += 1
                    if adaptive_before is None and adaptive_after is not None:
                        counters["adaptive_policy_resurrections"] += 1

                    if fixed_before is not None:
                        counters["fixed_feasible_base_cases"] += 1
                    if fixed_after is not None:
                        counters["fixed_feasible_expanded_cases"] += 1
                    if fixed_before is not None and fixed_after is not None:
                        if fixed_after == fixed_before:
                            counters["fixed_cost_unchanged_cases"] += 1
                        elif fixed_after > fixed_before:
                            counters["fixed_cost_increased_cases"] += 1
                    elif fixed_before is not None and fixed_after is None:
                        counters["fixed_plan_lost_cases"] += 1
                    fixed_bad = not non_decreasing(fixed_before, fixed_after)
                    if fixed_bad:
                        counters["fixed_monotonicity_violations"] += 1
                    if fixed_before is None and fixed_after is not None:
                        counters["fixed_plan_resurrections"] += 1

                    if (adaptive_bad or fixed_bad) and len(counterexamples) < 16:
                        counterexamples.append({
                            "base_world_0_support_mask": sum(1 << TUPLES.index(row) for row in base_0),
                            "expanded_world_0_support_mask": sum(1 << TUPLES.index(row) for row in expanded_0),
                            "base_world_1_support_mask": sum(1 << TUPLES.index(row) for row in base_1),
                            "expanded_world_1_support_mask": sum(1 << TUPLES.index(row) for row in expanded_1),
                            "adaptive_cost_before": adaptive_before,
                            "adaptive_cost_after": adaptive_after,
                            "fixed_cost_before": fixed_before,
                            "fixed_cost_after": fixed_after,
                            "adaptive_monotonicity_violated": adaptive_bad,
                            "fixed_monotonicity_violated": fixed_bad,
                        })

    passed = (
        ordered_pairs == 4_225
        and counters["adaptive_monotonicity_violations"] == 0
        and counters["fixed_monotonicity_violations"] == 0
        and counters["adaptive_policy_resurrections"] == 0
        and counters["fixed_plan_resurrections"] == 0
    )
    return {
        "schema_version": 1,
        "tool": "connectome-support-expansion-lattice-audit",
        "world_count": 2,
        "action_count": 2,
        "joint_tuple_universe_per_world": 4,
        "action_costs": [1, 2],
        "nonempty_base_support_configurations_per_world": 15,
        "ordered_support_inclusion_pairs_checked": ordered_pairs,
        **counters,
        "counterexamples_retained": len(counterexamples),
        "counterexamples": counterexamples,
        "monotonicity_passed": passed,
        "exhaustive": True,
        "status": (
            "PASS_EXHAUSTIVE_SUPPORT_INCLUSION_LATTICE"
            if passed
            else "FAIL_SUPPORT_EXPANSION_MONOTONICITY"
        ),
        "scope_note": "Exhaustive over all non-empty support masks for two opposing-decision worlds, two binary observation actions and four possible joint tuples per world. It verifies only finite-set monotonicity; it does not validate biological support completeness, evidence provenance or measurement-model validity.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        expected = evaluate()
        actual = json.loads(args.report.read_text(encoding="utf-8"))
        assert actual == expected, "Rust support lattice report differs from independent set-based oracle"
        assert expected["ordered_support_inclusion_pairs_checked"] == 4_225
        assert expected["monotonicity_passed"] is True
        print(
            "verified exhaustive support lattice: "
            f"pairs={expected['ordered_support_inclusion_pairs_checked']}; "
            f"adaptive violations={expected['adaptive_monotonicity_violations']}; "
            f"fixed violations={expected['fixed_monotonicity_violations']}"
        )
        return 0
    except Exception as exc:
        print(f"support expansion lattice oracle failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
