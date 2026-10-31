#!/usr/bin/env python3
"""Exact two-budget Pareto optimizer for raw projective stopped-FDR closure.

For E_t^S = 1 + sum_{i in S} w_i (M_i,t - 1), all-intersection
membership is separable. At fixed rejection count k, the closure budget is
an additive nonnegative per-selected-hypothesis charge. Capacity and closure
therefore form a two-resource, exact-cardinality selection problem.

This module solves that problem by exact Fraction Pareto dynamic programming.
The fast frontier pruner uses a Fenwick tree; an independent quadratic
frontier replay is included for verification. Raw e-processes are not
necessarily time-monotone: this provides stopped-FDR scope only, not SupFDR.
"""
from __future__ import annotations

from bisect import bisect_left
from fractions import Fraction
from hashlib import sha256
import json
import math
from typing import Any, Callable, Sequence


def as_fraction(value: int | float | str | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("bool is not a rational input")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("rational inputs must be finite")
        return Fraction(str(value))
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError(f"unsupported rational input: {type(value).__name__}")


def fraction_pair(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def _normalise(
    evalue_paths: Sequence[Sequence[int | float | str | Fraction]],
    weights: Sequence[int | float | str | Fraction],
    active_masks: Sequence[int],
    alpha: int | float | str | Fraction,
    values: Sequence[int | float | str | Fraction],
    costs: Sequence[int | float | str | Fraction],
    capacities: Sequence[int | float | str | Fraction],
):
    n = len(evalue_paths)
    t_count = len(active_masks)
    if n < 1 or t_count < 1:
        raise ValueError("at least one hypothesis and one update are required")
    if not (len(weights) == len(values) == len(costs) == n):
        raise ValueError("weights, values and costs must match hypothesis count")
    if len(capacities) != t_count:
        raise ValueError("capacity path must match update count")
    paths = tuple(tuple(as_fraction(x) for x in row) for row in evalue_paths)
    if any(len(row) != t_count for row in paths):
        raise ValueError("each coordinate e-process needs one value per update")
    if any(x < 0 for row in paths for x in row):
        raise ValueError("coordinate e-process values must be nonnegative")
    w = tuple(as_fraction(x) for x in weights)
    if any(x < 0 for x in w) or sum(w) > 1:
        raise ValueError("global weights must be nonnegative and sum to at most one")
    a = as_fraction(alpha)
    if not 0 < a < 1:
        raise ValueError("alpha must lie in (0,1)")
    v = tuple(as_fraction(x) for x in values)
    c = tuple(as_fraction(x) for x in costs)
    b = tuple(as_fraction(x) for x in capacities)
    if any(x < 0 for x in v) or any(x < 0 for x in c) or any(x < 0 for x in b):
        raise ValueError("utilities, costs and capacities must be nonnegative")
    if any(b[t] < b[t - 1] for t in range(1, t_count)):
        raise ValueError("cumulative capacity must be nondecreasing")
    full_mask = (1 << n) - 1
    previous = 0
    active = tuple(active_masks)
    for t, mask in enumerate(active):
        if isinstance(mask, bool) or not isinstance(mask, int):
            raise ValueError("active masks must be integer bit masks")
        if mask < 0 or mask & ~full_mask:
            raise ValueError("active mask contains unknown hypotheses")
        if mask & previous != previous:
            raise ValueError("active hypotheses must arrive in nested order")
        previous = mask
        for i in range(n):
            if not ((mask >> i) & 1) and paths[i][t] != 1:
                raise ValueError("dormant coordinate e-values must equal one")
    return paths, w, active, a, v, c, b


def _positive_part(x: Fraction) -> Fraction:
    return x if x > 0 else Fraction(0)


def _charges_for_cardinality(
    evalues: tuple[Fraction, ...],
    weights: tuple[Fraction, ...],
    active_mask: int,
    alpha: Fraction,
    k: int,
) -> tuple[Fraction, tuple[Fraction, ...]]:
    """Return baseline load and per-item additive closure charges.

    Set h_i = alpha*w_i*(M_i-1). The all-subsets maximum has baseline
    sum_i [-h_i]_+ for nonselected hypotheses. Selecting i changes this
    contribution by
      [1/k-h_i]_+ - [-h_i]_+
      = min(1/k, [1/k-h_i]_+).
    Each charge is nonnegative, so two-budget Pareto DP applies.
    """
    if k < 1:
        raise ValueError("cardinality must be positive")
    baseline = Fraction(0)
    charges = []
    inv_k = Fraction(1, k)
    for i, (m_i, w_i) in enumerate(zip(evalues, weights)):
        if not ((active_mask >> i) & 1):
            charges.append(Fraction(0))
            continue
        h_i = alpha * w_i * (m_i - 1)
        baseline += _positive_part(-h_i)
        charges.append(min(inv_k, _positive_part(inv_k - h_i)))
    return baseline, tuple(charges)


# A state is (closure_charge, operational_cost, utility, selected_optional_indices).
_State = tuple[Fraction, Fraction, Fraction, tuple[int, ...]]


def _prune_frontier_fast(states: Sequence[_State]) -> list[_State]:
    """Pareto prune in O(F log F) with a Fenwick prefix-maximum tree.

    States are ordered by increasing cost, then risk, decreasing utility,
    then lexicographic selection. A state is dominated if an earlier state
    has no greater cost/risk and no smaller utility.
    """
    if not states:
        return []
    ordered = sorted(states, key=lambda s: (s[1], s[0], -s[2], s[3]))
    risks = sorted({s[0] for s in ordered})
    rank = {risk: i + 1 for i, risk in enumerate(risks)}
    tree: list[Fraction | None] = [None] * (len(risks) + 1)

    def query(index: int) -> Fraction | None:
        best = None
        while index > 0:
            current = tree[index]
            if current is not None and (best is None or current > best):
                best = current
            index -= index & -index
        return best

    def update(index: int, value: Fraction) -> None:
        while index < len(tree):
            if tree[index] is None or value > tree[index]:
                tree[index] = value
            index += index & -index

    kept: list[_State] = []
    for state in ordered:
        best_utility = query(rank[state[0]])
        if best_utility is not None and best_utility >= state[2]:
            continue
        kept.append(state)
        update(rank[state[0]], state[2])
    return kept


def _prune_frontier_independent(states: Sequence[_State]) -> list[_State]:
    """Quadratic, structurally independent Pareto-pruning implementation."""
    ordered = sorted(set(states), key=lambda s: (s[1], s[0], -s[2], s[3]))
    kept: list[_State] = []
    for candidate in ordered:
        dominated = False
        for other in ordered:
            if other is candidate:
                continue
            weakly_better = (
                other[0] <= candidate[0]
                and other[1] <= candidate[1]
                and other[2] >= candidate[2]
            )
            if not weakly_better:
                continue
            strictly_better = (
                other[0] < candidate[0]
                or other[1] < candidate[1]
                or other[2] > candidate[2]
            )
            if strictly_better or other[3] < candidate[3]:
                dominated = True
                break
        if not dominated:
            kept.append(candidate)
    return kept


def _solve_cardinality(
    charges: tuple[Fraction, ...],
    baseline: Fraction,
    values: tuple[Fraction, ...],
    costs: tuple[Fraction, ...],
    active_mask: int,
    previous_mask: int,
    capacity: Fraction,
    alpha: Fraction,
    k: int,
    *,
    prune: Callable[[Sequence[_State]], list[_State]],
    frontier_limit: int,
) -> tuple[tuple[tuple[int, ...], Fraction, Fraction, Fraction] | None, dict]:
    n = len(charges)
    mandatory = [i for i in range(n) if (previous_mask >> i) & 1]
    optional = [
        i for i in range(n)
        if ((active_mask >> i) & 1) and not ((previous_mask >> i) & 1)
    ]
    choose = k - len(mandatory)
    if choose < 0 or choose > len(optional):
        return None, {"skipped": "cardinality_infeasible", "max_frontier": 0}

    mandatory_risk = sum((charges[i] for i in mandatory), Fraction(0))
    mandatory_cost = sum((costs[i] for i in mandatory), Fraction(0))
    mandatory_value = sum((values[i] for i in mandatory), Fraction(0))
    risk_budget = alpha - baseline - mandatory_risk
    cost_budget = capacity - mandatory_cost
    if risk_budget < 0 or cost_budget < 0:
        return None, {"skipped": "mandatory_budget_exceeded", "max_frontier": 0}

    # Cheap necessary feasibility bounds before dynamic programming.
    optimistic_risk = sum(
        sorted((charges[i] for i in optional))[:choose], Fraction(0)
    )
    optimistic_cost = sum(
        sorted((costs[i] for i in optional))[:choose], Fraction(0)
    )
    if optimistic_risk > risk_budget or optimistic_cost > cost_budget:
        return None, {
            "skipped": "optimistic_resource_bound",
            "max_frontier": 0,
        }

    frontiers: list[list[_State]] = [[] for _ in range(choose + 1)]
    frontiers[0] = [(Fraction(0), Fraction(0), Fraction(0), ())]
    max_frontier = 1
    total_frontier_states = 1
    for processed, i in enumerate(optional, start=1):
        upper = min(choose, processed)
        for count in range(upper, 0, -1):
            prior = frontiers[count - 1]
            if not prior:
                continue
            added: list[_State] = []
            item_risk = charges[i]
            item_cost = costs[i]
            item_value = values[i]
            for risk, cost, utility, selected in prior:
                next_risk = risk + item_risk
                next_cost = cost + item_cost
                if next_risk <= risk_budget and next_cost <= cost_budget:
                    added.append((
                        next_risk,
                        next_cost,
                        utility + item_value,
                        selected + (i,),
                    ))
            if added:
                merged = frontiers[count] + added
                frontiers[count] = prune(merged)
                max_frontier = max(max_frontier, len(frontiers[count]))
                total_frontier_states = sum(len(frontier) for frontier in frontiers)
                if max_frontier > frontier_limit or total_frontier_states > frontier_limit * (choose + 1):
                    raise RuntimeError(
                        f"exact Pareto frontier limit exceeded at item={i}, count={count}; "
                        f"max_frontier={max_frontier}, total_states={total_frontier_states}"
                    )

    final_frontier = frontiers[choose]
    if not final_frontier:
        return None, {
            "skipped": "no_feasible_state",
            "max_frontier": max_frontier,
            "total_frontier_states": total_frontier_states,
        }
    # Primary objective: maximize utility. Deterministic ties prefer lower
    # operational cost, then smaller certified risk consumption, then indices.
    best = min(final_frontier, key=lambda s: (-s[2], s[1], s[0], s[3]))
    selected = tuple(sorted(tuple(mandatory) + best[3]))
    total_risk = baseline + mandatory_risk + best[0]
    total_cost = mandatory_cost + best[1]
    total_value = mandatory_value + best[2]
    return (selected, total_value, total_cost, total_risk), {
        "max_frontier": max_frontier,
        "total_frontier_states": total_frontier_states,
        "frontier_at_target_cardinality": len(final_frontier),
        "optimistic_risk": optimistic_risk,
        "optimistic_cost": optimistic_cost,
    }


def _prefix_digest(
    paths, weights, active_masks, alpha, values, costs, capacities, t
) -> str:
    active_universe = 0
    for mask in active_masks[: t + 1]:
        active_universe |= mask
    indices = [i for i in range(len(weights)) if (active_universe >> i) & 1]
    payload = {
        "time": t,
        "active_masks": list(active_masks[: t + 1]),
        "coordinate_evalues": {
            str(i): [fraction_pair(paths[i][j]) for j in range(t + 1)]
            for i in indices
        },
        "weights": {str(i): fraction_pair(weights[i]) for i in indices},
        "alpha": fraction_pair(alpha),
        "values": {str(i): fraction_pair(values[i]) for i in indices},
        "costs": {str(i): fraction_pair(costs[i]) for i in indices},
        "capacity_prefix": [fraction_pair(x) for x in capacities[: t + 1]],
    }
    return _digest(payload)


def _solve_one_time(
    current_e: tuple[Fraction, ...],
    weights: tuple[Fraction, ...],
    active_mask: int,
    previous_mask: int,
    alpha: Fraction,
    values: tuple[Fraction, ...],
    costs: tuple[Fraction, ...],
    capacity: Fraction,
    *,
    prune: Callable[[Sequence[_State]], list[_State]],
    frontier_limit: int,
) -> tuple[int, tuple[Fraction, Fraction, Fraction], dict]:
    best = None
    best_key = None
    summaries = {}
    max_frontier = 0
    if previous_mask == 0:
        best = ((), Fraction(0), Fraction(0), Fraction(0))
        best_key = (Fraction(0), Fraction(0), Fraction(0), ())
    for k in range(max(1, previous_mask.bit_count()), active_mask.bit_count() + 1):
        baseline, charges = _charges_for_cardinality(
            current_e, weights, active_mask, alpha, k
        )
        candidate, summary = _solve_cardinality(
            charges, baseline, values, costs, active_mask, previous_mask,
            capacity, alpha, k, prune=prune, frontier_limit=frontier_limit,
        )
        summaries[str(k)] = summary
        max_frontier = max(max_frontier, summary.get("max_frontier", 0))
        if candidate is None:
            continue
        selected, utility, total_cost, total_risk = candidate
        key = (-utility, total_cost, total_risk, selected)
        if best is None or key < best_key:
            best = candidate
            best_key = key
    if best is None:
        raise RuntimeError(
            "no nested certified action exists at this update. Raw projective "
            "e-closure is not setwise persistent; the previous irrevocable "
            "set may no longer be certifiable. Use a time-monotone adjusted "
            "SupFDR construction or change the reporting/commitment policy."
        )
    selected, utility, total_cost, total_risk = best
    mask = sum(1 << i for i in selected)
    check = _verify_separator(current_e, weights, active_mask, alpha, mask)
    if not check["certified"]:
        raise ArithmeticError("Pareto DP selected a set outside the certified menu")
    expected_slack = alpha - total_risk if selected else None
    if selected and check["minimum_slack"] != expected_slack:
        raise ArithmeticError("additive risk charge disagrees with O(N) separator")
    return mask, (utility, total_cost, total_risk), {
        "cardinality_summaries": summaries,
        "max_frontier_states": max_frontier,
        "minimum_slack": check["minimum_slack"],
        "worst_intersection_mask": check["worst_intersection_mask"],
        "feasible": check["certified"],
    }


def _verify_separator(
    evalues: tuple[Fraction, ...],
    weights: tuple[Fraction, ...],
    active_mask: int,
    alpha: Fraction,
    rejection_mask: int,
) -> dict:
    n = len(evalues)
    if rejection_mask < 0 or rejection_mask & ~active_mask:
        raise ValueError("rejection set must be a subset of active hypotheses")
    k = rejection_mask.bit_count()
    if k == 0:
        return {
            "certified": True,
            "minimum_slack": None,
            "worst_intersection_mask": 0,
            "positive_part_sum": Fraction(0),
        }
    positive_sum = Fraction(0)
    worst = 0
    for i in range(n):
        if not ((active_mask >> i) & 1):
            continue
        coefficient = (
            (Fraction(1, k) if ((rejection_mask >> i) & 1) else Fraction(0))
            - alpha * weights[i] * (evalues[i] - 1)
        )
        if coefficient > 0:
            positive_sum += coefficient
            worst |= 1 << i
    slack = alpha - positive_sum
    return {
        "certified": slack >= 0,
        "minimum_slack": slack,
        "worst_intersection_mask": worst,
        "positive_part_sum": positive_sum,
    }


def select_capacity_aware_pareto(
    evalue_paths: Sequence[Sequence[int | float | str | Fraction]],
    weights: Sequence[int | float | str | Fraction],
    active_masks: Sequence[int],
    alpha: int | float | str | Fraction,
    values: Sequence[int | float | str | Fraction],
    costs: Sequence[int | float | str | Fraction],
    cumulative_capacity: Sequence[int | float | str | Fraction],
    *,
    frontier_limit: int = 100_000,
    verify_independent: bool = True,
    independent_frontier_limit: int = 100_000,
) -> dict:
    """Select nested actions by exact two-budget Pareto DP.

    Membership is O(N). For fixed rejection cardinality, the all-intersection
    condition is one additive risk budget, so the exact optimizer maintains a
    Pareto frontier over risk, cost and utility. Worst-case frontier size can
    still be exponential; the routine fails closed at the configured limit.
    """
    paths, w, active, level, v, c, capacity = _normalise(
        evalue_paths, weights, active_masks, alpha, values, costs,
        cumulative_capacity,
    )
    n = len(paths)
    if frontier_limit < 1 or independent_frontier_limit < 1:
        raise ValueError("frontier limits must be positive")
    decisions = []
    previous = 0
    max_seen_frontier = 0
    for t, active_mask in enumerate(active):
        current_e = tuple(paths[i][t] for i in range(n))
        mask, (utility, total_cost, total_risk), detail = _solve_one_time(
            current_e, w, active_mask, previous, level, v, c, capacity[t],
            prune=_prune_frontier_fast, frontier_limit=frontier_limit,
        )
        max_seen_frontier = max(max_seen_frontier, detail["max_frontier_states"])
        decisions.append({
            "time": t,
            "active_mask": active_mask,
            "rejection_mask": mask,
            "rejections": [i + 1 for i in range(n) if (mask >> i) & 1],
            "utility": fraction_pair(utility),
            "cost": fraction_pair(total_cost),
            "capacity": fraction_pair(capacity[t]),
            "closure_load": fraction_pair(total_risk) if mask else None,
            "minimum_stopped_fdr_slack": (
                fraction_pair(detail["minimum_slack"])
                if detail["minimum_slack"] is not None else None
            ),
            "worst_intersection_mask": detail["worst_intersection_mask"],
            "max_frontier_states": detail["max_frontier_states"],
            "cardinality_summaries": detail["cardinality_summaries"],
            "prefix_sha256": _prefix_digest(
                paths, w, active, level, v, c, capacity, t
            ),
        })
        previous = mask

    validity = verify_capacity_aware_pareto(
        decisions, paths, w, active, level, v, c, capacity,
        independent=verify_independent,
        independent_frontier_limit=independent_frontier_limit,
    )
    if not validity["valid"]:
        raise ArithmeticError(f"independent Pareto replay failed: {validity}")
    return {
        "schema": "symthaea-stopped-fdr-capacity-pareto-v1",
        "status": "EXACT_FRACTION_PARETO_DP_REFERENCE_NOT_WORST_CASE_POLYNOMIAL",
        "guarantee_scope": (
            "simultaneous stopped-FDR under future-extension-coherent raw "
            "projective e-processes; NOT SupFDR because raw certificates need "
            "not be time-monotone"
        ),
        "weights": [fraction_pair(x) for x in w],
        "alpha": fraction_pair(level),
        "decisions": decisions,
        "verification": validity,
        "complexity": {
            "membership_per_candidate": "O(N)",
            "optimizer": "cardinality-indexed two-resource Pareto DP",
            "worst_case_frontier": "exponential; fail-closed frontier cap",
            "observed_max_frontier_states": max_seen_frontier,
        },
        "limitations": [
            "coordinate e-process validity in the common filtration is assumed",
            "raw closure is not setwise persistent; irrevocable nested extension may become infeasible",
            "cumulative capacity only",
            "exact Pareto frontier can be exponential",
            "no SupFDR claim",
        ],
    }


def verify_capacity_aware_pareto(
    decisions,
    paths,
    weights,
    active,
    alpha,
    values,
    costs,
    capacities,
    *,
    independent: bool = True,
    independent_frontier_limit: int = 100_000,
) -> dict:
    """Replay input, separator, receipt, and DP optimum with another pruner."""
    paths, weights, active, alpha, values, costs, capacities = _normalise(
        paths, weights, active, alpha, values, costs, capacities
    )
    failures = []
    previous = 0
    independent_verified = True
    frontier_peak = 0
    if len(decisions) != len(active):
        failures.append("decision_count_mismatch")
    for t, record in enumerate(decisions):
        if t >= len(active) or not isinstance(record, dict):
            failures.append(("invalid_or_extra_decision", t))
            continue
        mask = record.get("rejection_mask")
        if isinstance(mask, bool) or not isinstance(mask, int) or mask < 0 or mask >= (1 << len(paths)):
            failures.append(("invalid_rejection_mask", t))
            continue
        if mask & active[t] != mask:
            failures.append(("inactive_hypothesis_rejected", t))
        if mask & previous != previous:
            failures.append(("non_nested_decision", t))
        current_e = tuple(paths[i][t] for i in range(len(paths)))
        cert = _verify_separator(current_e, weights, active[t], alpha, mask)
        if not cert["certified"]:
            failures.append(("uncertified_rejection_set", t))
        utility = sum(
            (values[i] for i in range(len(paths)) if (mask >> i) & 1),
            Fraction(0),
        )
        cost = sum(
            (costs[i] for i in range(len(paths)) if (mask >> i) & 1),
            Fraction(0),
        )
        if cost > capacities[t]:
            failures.append(("capacity_exceeded", t))
        expected = {
            "time": t,
            "active_mask": active[t],
            "rejections": [i + 1 for i in range(len(paths)) if (mask >> i) & 1],
            "utility": fraction_pair(utility),
            "cost": fraction_pair(cost),
            "capacity": fraction_pair(capacities[t]),
            "minimum_stopped_fdr_slack": (
                fraction_pair(cert["minimum_slack"])
                if cert["minimum_slack"] is not None else None
            ),
            "worst_intersection_mask": cert["worst_intersection_mask"],
            "prefix_sha256": _prefix_digest(
                paths, weights, active, alpha, values, costs, capacities, t
            ),
        }
        for field, expected_value in expected.items():
            if record.get(field) != expected_value:
                failures.append(("receipt_mismatch", t, field))

        if independent:
            try:
                expected_mask, _, details = _solve_one_time(
                    current_e, weights, active[t], previous, alpha, values,
                    costs, capacities[t], prune=_prune_frontier_independent,
                    frontier_limit=independent_frontier_limit,
                )
                frontier_peak = max(frontier_peak, details["max_frontier_states"])
                if expected_mask != mask:
                    failures.append(("utility_optimality_failure", t, expected_mask, mask))
            except RuntimeError as exc:
                independent_verified = False
                failures.append(("independent_frontier_limit", t, str(exc)))
        previous = mask

    if len(decisions) != len(active):
        independent_verified = False
    return {
        "valid": not failures,
        "failures": failures,
        "decisions_checked": min(len(decisions), len(active)),
        "independent_optimality_verified": independent and independent_verified and not failures,
        "independent_verifier_frontier_peak": frontier_peak,
        "scope": "raw projective stopped-FDR feasibility, nested capacity, exact Pareto optimum",
    }


if __name__ == "__main__":
    raise SystemExit("Run test_stopped_eclosure_capacity_pareto.py for exact fixtures.")
