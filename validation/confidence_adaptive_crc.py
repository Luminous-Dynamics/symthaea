#!/usr/bin/env python3
"""Exact confidence-adaptive CRC recursion with optional breakpoint capping/grouping.

Research implementation of Bai & Ramdas (2026), Appendix B.7.
A calibration point is (u_i,v_i,c_i), u_i=1-p_g, v_i=1-p_f, c_i in {0,1,2}.
The query is a feature with breakpoints (u_x,v_x). Type 0 has zero loss and
is excluded from the active recursion, but remains in the calibration size.

This module provides:
- exact ungrouped recursion as a small-family oracle;
- exact breakpoint capping;
- exact grouping by identical capped (u,v) pairs;
- differential expansion of grouped states back to every ungrouped assignment.

It does not include the CIFAR cached ResNet-18 prediction artifact.
"""
from __future__ import annotations
from itertools import combinations, product
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Iterable


def loss(error_type: int, depth: float) -> float:
    if error_type == 0:
        return 0.0
    if error_type == 1:
        return 0.5 * max(depth - 1.0, 0.0)
    if error_type == 2:
        return 0.5 * depth
    raise ValueError("error type must be 0, 1 or 2")


def validate_pair(pair: tuple[float, float]) -> tuple[float, float]:
    if len(pair) != 2:
        raise ValueError("a confidence pair must be (u,v)")
    u, v = float(pair[0]), float(pair[1])
    if not (math.isfinite(u) and math.isfinite(v) and 0 <= u <= v <= 1):
        raise ValueError("breakpoints must satisfy 0 <= u <= v <= 1")
    return u, v


class ConfidenceAdaptiveCRC:
    """Exact finite recursion for one calibration sample and one query feature."""

    def __init__(
        self,
        calibration: Iterable[tuple[float, float, int]],
        query_pair: tuple[float, float],
        alpha: float,
        *,
        cap_breakpoints: bool = True,
        group_duplicates: bool = True,
    ):
        rows = [(validate_pair((u, v))[0], validate_pair((u, v))[1], int(c))
                for u, v, c in calibration]
        if any(c not in (0, 1, 2) for _, _, c in rows):
            raise ValueError("calibration error types must be 0, 1 or 2")
        if not math.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        self.calibration = rows
        self.query_pair = validate_pair(query_pair)
        self.n = len(rows)
        self.alpha = float(alpha)
        self.budget = (self.n + 1) * self.alpha
        self.b0 = 2.0 * (self.budget - 1.0)
        self.b = math.floor(self.b0 + 1e-12)

        # Active points are nonzero-type calibration examples plus the query.
        # Type-0 calibration observations are fixed, zero-loss and only affect n.
        active_raw = [(i, (u, v)) for i, (u, v, c) in enumerate(rows) if c != 0]
        self.query_active_index = len(active_raw)
        active_raw.append((None, self.query_pair))
        self.active_calibration_indices = [i for i, _ in active_raw[:-1]]
        raw_pairs = [pair for _, pair in active_raw]

        self.cap_threshold = None
        if cap_breakpoints and self.b >= 0 and len(raw_pairs) >= self.b + 2:
            leaf_breakpoints = sorted(v for _, v in raw_pairs)
            self.cap_threshold = float(leaf_breakpoints[self.b + 1])
            raw_pairs = [
                (min(u, self.cap_threshold), min(v, self.cap_threshold))
                for u, v in raw_pairs
            ]

        self.active_raw = active_raw
        self.raw_pairs = raw_pairs
        self.cap_breakpoints = bool(cap_breakpoints)
        self.group_duplicates = bool(group_duplicates)

        group_index: dict[tuple[float, float], int] = {}
        group_pairs: list[tuple[float, float]] = []
        group_members: list[list[int]] = []
        raw_to_group: list[int] = []
        for raw_index, pair in enumerate(raw_pairs):
            key = pair if group_duplicates else (pair[0], pair[1], float(raw_index))
            if key not in group_index:
                group_index[key] = len(group_pairs)
                group_pairs.append(pair)
                group_members.append([])
            g = group_index[key]
            group_members[g].append(raw_index)
            raw_to_group.append(g)

        self.group_pairs = group_pairs
        self.group_members = group_members
        self.group_sizes = [len(members) for members in group_members]
        self.raw_to_group = raw_to_group
        self.query_group = raw_to_group[self.query_active_index]
        self.num_groups = len(group_pairs)
        self.num_active_points = len(raw_pairs)

        # A state (j,m) holds out one point from group j. m_h counts type-2
        # among the remaining active calibration points in group h.
        self.states: list[tuple[int, tuple[int, ...]]] = []
        for j in range(self.num_groups):
            ranges = [
                range(self.group_sizes[h]) if h == j
                else range(self.group_sizes[h] + 1)
                for h in range(self.num_groups)
            ]
            for counts in product(*ranges):
                self.states.append((j, tuple(int(x) for x in counts)))
        self.state_count = len(self.states)
        self.expected_state_count = sum(
            self.group_sizes[j] * math.prod(
                self.group_sizes[h] + 1 for h in range(self.num_groups) if h != j
            )
            for j in range(self.num_groups)
        )
        if self.state_count != self.expected_state_count:
            raise ArithmeticError("grouped state count does not match the closed-form count")

        self.baseline_table = {state: self._baseline_depth(*state) for state in self.states}

    def _calibration_group_size(self, heldout_group: int, group: int) -> int:
        return self.group_sizes[group] - (1 if group == heldout_group else 0)

    def _baseline_depth(self, heldout_group: int, counts: tuple[int, ...]) -> float:
        jumps = []
        for h, (u, v) in enumerate(self.group_pairs):
            size = self._calibration_group_size(heldout_group, h)
            type2 = counts[h]
            if not 0 <= type2 <= size:
                raise ValueError("state type-2 count exceeds group calibration size")
            # Every nonzero type contributes one leaf breakpoint; type 2 also
            # contributes the superclass breakpoint.
            jumps.extend([u] * type2)
            jumps.extend([v] * size)
        if self.b < 0:
            threshold = 0.0
        else:
            jumps.sort()
            threshold = 1.0 if self.b >= len(jumps) else jumps[self.b]
        u_query, v_query = self.group_pairs[heldout_group]
        return float(threshold > u_query) + float(threshold > v_query)

    def _substate(
        self,
        heldout_group: int,
        counts: tuple[int, ...],
        test_type: int,
        calibration_group: int,
        heldout_type: int,
    ) -> tuple[int, tuple[int, ...]]:
        updated = list(counts)
        test_is_type2 = 1 if test_type == 2 else 0
        heldout_is_type2 = 1 if heldout_type == 2 else 0
        updated[heldout_group] += test_is_type2
        updated[calibration_group] -= heldout_is_type2
        return calibration_group, tuple(updated)

    def total_for_state(
        self,
        table: dict[tuple[int, tuple[int, ...]], float],
        heldout_group: int,
        counts: tuple[int, ...],
        test_type: int,
    ) -> float:
        """Exact leave-one-out total for this augmented active multiset."""
        if test_type not in (1, 2):
            raise ValueError("only nonzero hypothetical test types need certification")
        total = loss(test_type, table[(heldout_group, counts)])
        for h in range(self.num_groups):
            size = self._calibration_group_size(heldout_group, h)
            type2_count = counts[h]
            type1_count = size - type2_count
            if type1_count:
                substate = self._substate(
                    heldout_group, counts, test_type, h, 1
                )
                total += type1_count * loss(1, table[substate])
            if type2_count:
                substate = self._substate(
                    heldout_group, counts, test_type, h, 2
                )
                total += type2_count * loss(2, table[substate])
        return total

    def validity_report(
        self, table: dict[tuple[int, tuple[int, ...]], float],
        tolerance: float = 1e-9,
    ) -> dict:
        if set(table) != set(self.states):
            raise ValueError("depth table must contain every grouped state exactly once")
        if any(not math.isfinite(a) or not 0 <= a <= 2 for a in table.values()):
            raise ValueError("depth values must be finite and in [0,2]")
        worst = (-math.inf, None, None)
        for j, counts in self.states:
            for c in (1, 2):
                total = self.total_for_state(table, j, counts, c)
                if total > worst[0]:
                    worst = (total, (j, counts), c)
        excess = float(worst[0] - self.budget)
        return {
            "valid": bool(excess <= tolerance),
            "max_budget_excess": excess,
            "worst_total": float(worst[0]),
            "worst_state": worst[1],
            "worst_hypothetical_type": worst[2],
            "budget": self.budget,
            "state_count": self.state_count,
            "num_groups": self.num_groups,
            "num_active_points": self.num_active_points,
            "cap_threshold": self.cap_threshold,
        }

    def recertify_round(
        self, table: dict[tuple[int, tuple[int, ...]], float]
    ) -> dict[tuple[int, tuple[int, ...]], float]:
        if not self.validity_report(table)["valid"]:
            raise ValueError("starting depth table is invalid")
        updated = {}
        for j, counts in self.states:
            a = table[(j, counts)]
            t1 = self.total_for_state(table, j, counts, 1)
            t2 = self.total_for_state(table, j, counts, 2)
            s1 = self.alpha - t1 / (self.n + 1)
            s2 = self.alpha - t2 / (self.n + 1)
            if s1 < -1e-8 or s2 < -1e-8:
                raise ArithmeticError("negative unused budget in a hypothetical-label state")
            updated[(j, counts)] = min(
                2.0,
                a + 2.0 * max(0.0, s2),
                1.0 + max(a - 1.0, 0.0) + 2.0 * max(0.0, s1),
            )
        if not self.validity_report(updated)["valid"]:
            raise ArithmeticError("recertification update broke a hypothetical-label constraint")
        if any(updated[state] + 1e-10 < table[state] for state in self.states):
            raise ArithmeticError("recertification became more conservative")
        return updated

    def recertify(
        self, rounds: int = 1, tolerance: float = 1e-10
    ) -> tuple[dict, list[dict]]:
        if rounds < 0 or tolerance <= 0:
            raise ValueError("invalid number of rounds or tolerance")
        current = dict(self.baseline_table)
        history = [dict(current)]
        for _ in range(rounds):
            updated = self.recertify_round(current)
            history.append(updated)
            change = max(abs(updated[s] - current[s]) for s in self.states)
            current = updated
            if change <= tolerance:
                break
        return current, history

    def query_depth(self, table: dict[tuple[int, tuple[int, ...]], float]) -> float:
        counts = [0] * self.num_groups
        for raw_index, calibration_index in enumerate(self.active_calibration_indices):
            _, _, error_type = self.calibration[calibration_index]
            group = self.raw_to_group[raw_index]
            if error_type == 2:
                counts[group] += 1
        state = (self.query_group, tuple(counts))
        return float(table[state])

    def expand_group_state_against_ungrouped(
        self,
        ungrouped: "ConfidenceAdaptiveCRC",
        heldout_group: int,
        counts: tuple[int, ...],
        tolerance: float = 1e-10,
        grouped_table: dict[tuple[int, tuple[int, ...]], float] | None = None,
        ungrouped_table: dict[tuple[int, tuple[int, ...]], float] | None = None,
    ) -> dict:
        """Compare one grouped cell with every point-level assignment.

        Optional table arguments let tests compare *recertified* tables, not
        only the initial baseline. This is intended for small exact fixtures.
        """
        if ungrouped.group_duplicates:
            raise ValueError("comparison engine must use group_duplicates=False")
        if self.num_active_points != ungrouped.num_active_points:
            raise ValueError("grouped and ungrouped active point sets differ")
        grouped_values = self.baseline_table if grouped_table is None else grouped_table
        ungrouped_values = ungrouped.baseline_table if ungrouped_table is None else ungrouped_table
        heldout_raw_indices = [
            i for i, g in enumerate(self.raw_to_group) if g == heldout_group
        ]
        values = []
        for heldout_raw in heldout_raw_indices:
            group_choices = []
            for h, members in enumerate(self.group_members):
                eligible = [i for i in members if i != heldout_raw]
                group_choices.append(list(combinations(eligible, counts[h])))
            for chosen_per_group in product(*group_choices):
                type2_set = {i for subset in chosen_per_group for i in subset}
                ungrouped_counts = tuple(
                    0 if raw == heldout_raw else (1 if raw in type2_set else 0)
                    for raw in range(self.num_active_points)
                )
                value = ungrouped_values.get((heldout_raw, ungrouped_counts))
                if value is None:
                    raise KeyError("ungrouped state was not found")
                values.append(float(value))
        reference = float(grouped_values[(heldout_group, counts)])
        if not values:
            raise ArithmeticError("no point-level assignments expanded this group state")
        return {
            "match": bool(max(abs(v - reference) for v in values) <= tolerance),
            "reference": reference,
            "min_expanded": min(values),
            "max_expanded": max(values),
            "expanded_assignments": len(values),
        }


def build_feature_adaptive(
    calibration: Iterable[tuple[float, float, int]],
    query_pair: tuple[float, float],
    alpha: float,
    rounds: int = 1,
    *,
    cap_breakpoints: bool = True,
    group_duplicates: bool = True,
) -> dict:
    engine = ConfidenceAdaptiveCRC(
        calibration, query_pair, alpha,
        cap_breakpoints=cap_breakpoints,
        group_duplicates=group_duplicates,
    )
    table, history = engine.recertify(rounds)
    return {
        "engine": engine,
        "table": table,
        "history": history,
        "query_depth": engine.query_depth(table),
        "validity": engine.validity_report(table),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    # Small deterministic smoke example only; real CIFAR probabilities are not present.
    rng_seed = args.seed
    calibration = [
        (0.02, 0.30, 1), (0.08, 0.42, 2),
        (0.02, 0.30, 1), (0.11, 0.55, 1),
        (0.0, 0.0, 0), (0.08, 0.42, 2),
    ]
    query = (0.02, 0.30)
    started = time.perf_counter()
    result = build_feature_adaptive(calibration, query, alpha=.25, rounds=10)
    engine = result["engine"]
    payload = {
        "schema": "symthaea-confidence-adaptive-crc-v1",
        "status": "SYNTHETIC_SMALL_FAMILY_ORACLE_NOT_CIFAR_REPRODUCTION",
        "seed": rng_seed,
        "alpha": .25,
        "calibration_n": len(calibration),
        "active_points": engine.num_active_points,
        "groups": engine.num_groups,
        "group_sizes": engine.group_sizes,
        "table_states": engine.state_count,
        "query_depth": result["query_depth"],
        "validity": result["validity"],
        "rounds": len(result["history"])-1,
        "runtime_seconds": time.perf_counter()-started,
        "limitations": [
            "small synthetic confidence breakpoints only",
            "not the CIFAR-100 cached ResNet-18 experiment",
            "finite exchangeability only; no shift guarantee",
        ],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
