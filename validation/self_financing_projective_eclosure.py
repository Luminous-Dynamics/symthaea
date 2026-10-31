#!/usr/bin/env python3
"""Self-financing, projective dynamic e-closure research oracle.

Each coordinate supplies one-step conditional e-value increments X[i,t].
Predictable portfolio shares are normalized over the active hypotheses at
each global update. For each finite intersection S, the process multiplies
by the convex combination of cash and the increments for S's active members.

The code verifies pathwise algebra, projectivity, adjusted dynamic closure,
capacity feasibility, and finite-horizon utility optimality. It does NOT prove
the upstream conditional-expectation contract E[X[i,t]|F_{t-1}] <= 1; that
must be justified in the common global filtration by the evidence producer.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json
import math
from typing import Any, Sequence


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
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return sha256(raw).hexdigest()


def _normalise(
    increment_paths: Sequence[Sequence[int | float | str | Fraction]],
    active_masks: Sequence[int],
    priority_scores: Sequence[int | float | str | Fraction],
    alpha: int | float | str | Fraction,
    values: Sequence[int | float | str | Fraction],
    costs: Sequence[int | float | str | Fraction],
    capacities: Sequence[int | float | str | Fraction],
):
    n = len(increment_paths)
    t_count = len(active_masks)
    if n < 1 or t_count < 1:
        raise ValueError("at least one hypothesis and one global update are required")
    if not (len(priority_scores) == len(values) == len(costs) == n):
        raise ValueError("scores, values and costs must match hypothesis count")
    if len(capacities) != t_count:
        raise ValueError("capacity path must match the global update count")

    x = tuple(tuple(as_fraction(v) for v in row) for row in increment_paths)
    if any(len(row) != t_count for row in x):
        raise ValueError("each coordinate increment path must match update count")
    if any(v < 0 for row in x for v in row):
        raise ValueError("one-step e-value increments must be nonnegative")

    active = tuple(active_masks)
    full_mask = (1 << n) - 1
    previous = 0
    for t, mask in enumerate(active):
        if isinstance(mask, bool) or not isinstance(mask, int):
            raise ValueError("active masks must be integer bit masks")
        if mask < 0 or mask & ~full_mask:
            raise ValueError("active mask contains an unknown coordinate")
        if mask & previous != previous:
            raise ValueError("active hypothesis sets must be nested")
        previous = mask
        for i in range(n):
            if not (mask >> i) & 1 and x[i][t] != 1:
                raise ValueError("inactive coordinate increments must be exactly one")

    scores = tuple(as_fraction(v) for v in priority_scores)
    if any(v < 0 for v in scores):
        raise ValueError("portfolio priority scores must be nonnegative")
    a = as_fraction(alpha)
    if not 0 < a < 1:
        raise ValueError("alpha must lie in (0,1)")
    vals = tuple(as_fraction(v) for v in values)
    c = tuple(as_fraction(v) for v in costs)
    b = tuple(as_fraction(v) for v in capacities)
    if any(v < 0 for v in vals) or any(v < 0 for v in c) or any(v < 0 for v in b):
        raise ValueError("values, costs and capacities must be nonnegative")
    if any(b[t] < b[t - 1] for t in range(1, t_count)):
        raise ValueError("cumulative capacity must be nondecreasing")
    return x, active, scores, a, vals, c, b


def predictable_shares(
    active_mask: int,
    priority_scores: Sequence[Fraction],
) -> tuple[Fraction, ...]:
    """Normalize frozen scores over the pre-decision active set.

    These shares are determined before the current coordinate increments are
    observed. If all active scores are zero, the portfolio holds cash.
    """
    active_scores = [
        priority_scores[i]
        for i in range(len(priority_scores))
        if (active_mask >> i) & 1
    ]
    total = sum(active_scores, Fraction(0))
    if total == 0:
        return tuple(Fraction(0) for _ in priority_scores)
    return tuple(
        priority_scores[i] / total if (active_mask >> i) & 1 else Fraction(0)
        for i in range(len(priority_scores))
    )


def _policy_scores(
    policy: str,
    fixed_scores: Sequence[Fraction],
    past_wealth: Sequence[Fraction],
    active_mask: int,
) -> tuple[Fraction, ...]:
    if policy == "static_priority":
        return tuple(fixed_scores)
    if policy == "equal_active":
        return tuple(
            Fraction(1) if (active_mask >> i) & 1 else Fraction(0)
            for i in range(len(fixed_scores))
        )
    if policy == "past_wealth":
        return tuple(past_wealth)
    if policy == "past_wealth_squared":
        return tuple(value * value for value in past_wealth)
    if policy == "largest_past_wealth":
        active_values = [
            past_wealth[i] for i in range(len(past_wealth))
            if (active_mask >> i) & 1
        ]
        if not active_values:
            return tuple(Fraction(0) for _ in past_wealth)
        maximum = max(active_values)
        return tuple(
            Fraction(1)
            if (active_mask >> i) & 1 and past_wealth[i] == maximum
            else Fraction(0)
            for i in range(len(past_wealth))
        )
    raise ValueError(
        "share_policy must be static_priority, equal_active, past_wealth, "
        "past_wealth_squared, or largest_past_wealth"
    )


def build_self_financing_intersection_paths(
    increment_paths: Sequence[Sequence[int | float | str | Fraction]],
    active_masks: Sequence[int],
    priority_scores: Sequence[int | float | str | Fraction],
    *,
    share_policy: str = "static_priority",
) -> tuple[dict[int, tuple[Fraction, ...]], tuple[tuple[Fraction, ...], ...]]:
    """Build every intersection product process and the predictable shares.

    For S and update t:
      G_t^S = 1 - sum_{i in S cap I_t} p[i,t]
                + sum_{i in S cap I_t} p[i,t] * X[i,t]
      E_t^S = E_{t-1}^S * G_t^S.

    Under H_S, if X[i,t] are nonnegative one-step conditional e-values in
    the same global filtration, E[G_t^S | F_{t-1}] <= 1. The factor is
    nonnegative because the total active portfolio share is at most one.
    """
    n = len(increment_paths)
    t_count = len(active_masks)
    x = tuple(tuple(as_fraction(v) for v in row) for row in increment_paths)
    scores = tuple(as_fraction(v) for v in priority_scores)
    if n < 1 or t_count < 1 or len(scores) != n:
        raise ValueError("invalid stream dimensions")
    if any(len(row) != t_count for row in x):
        raise ValueError("increment path length mismatch")
    if any(v < 0 for row in x for v in row) or any(v < 0 for v in scores):
        raise ValueError("increments and scores must be nonnegative")

    if share_policy not in (
        "static_priority", "equal_active", "past_wealth",
        "past_wealth_squared", "largest_past_wealth",
    ):
        raise ValueError("unsupported predictable share policy")
    past_wealth = [Fraction(1) for _ in range(n)]
    shares_list = []
    for t, active in enumerate(active_masks):
        current_scores = _policy_scores(
            share_policy, scores, past_wealth, active
        )
        shares_list.append(predictable_shares(active, current_scores))
        # Update the base coordinate wealth only after shares for this time
        # have been chosen: the portfolio policy is predictable.
        for i in range(n):
            past_wealth[i] *= x[i][t]
    shares = tuple(shares_list)
    result: dict[int, tuple[Fraction, ...]] = {}
    for subset in range(1, 1 << n):
        wealth = Fraction(1)
        row = []
        for t, active in enumerate(active_masks):
            mass = Fraction(0)
            bet = Fraction(0)
            for i in range(n):
                if ((subset & active) >> i) & 1:
                    p = shares[t][i]
                    mass += p
                    bet += p * x[i][t]
            factor = 1 - mass + bet
            if factor < 0:
                raise ArithmeticError("self-financing portfolio factor became negative")
            wealth *= factor
            row.append(wealth)
        result[subset] = tuple(row)
    return result, shares


def _adjust_paths(
    raw_paths: dict[int, Sequence[Fraction]],
    thresholds: Sequence[int | float | str | Fraction],
    increments: Sequence[int | float | str | Fraction],
) -> tuple[dict[int, tuple[Fraction, ...]], Fraction]:
    c = tuple(as_fraction(v) for v in thresholds)
    a = tuple(as_fraction(v) for v in increments)
    if len(c) != len(a):
        raise ValueError("adjuster thresholds and increments differ in length")
    if any(v <= 1 for v in c) or any(c[i] >= c[i + 1] for i in range(len(c) - 1)):
        raise ValueError("adjuster thresholds must be strictly increasing and exceed one")
    if any(v < 0 for v in a):
        raise ValueError("adjuster increments must be nonnegative")
    budget = sum((a[i] / c[i] for i in range(len(c))), Fraction(0))
    if budget > 1:
        raise ValueError("adjuster integral budget exceeds one")

    adjusted = {}
    for subset, values in raw_paths.items():
        running_max = Fraction(1)
        row = []
        for value in values:
            running_max = max(running_max, as_fraction(value))
            row.append(sum(
                (a[j] for j in range(len(c)) if running_max >= c[j]),
                Fraction(0),
            ))
        adjusted[subset] = tuple(row)
    return adjusted, budget


def _verify_dynamic_collection(
    adjusted: dict[int, Sequence[Fraction]],
    active_masks: Sequence[int],
) -> dict:
    subsets = sorted(adjusted)
    if not subsets or subsets != list(range(1, max(subsets) + 1)):
        raise ValueError("all nonempty intersection paths are required")
    n = max(subsets).bit_length()
    if max(subsets) != (1 << n) - 1:
        raise ValueError("intersection paths must cover a full coordinate universe")
    t_count = len(active_masks)
    paths = {s: tuple(as_fraction(v) for v in adjusted[s]) for s in subsets}
    if any(len(row) != t_count for row in paths.values()):
        raise ValueError("adjusted path length mismatch")
    if any(row[0] > 1 or any(v < 0 for v in row) for row in paths.values()):
        raise ValueError("adjusted intersection e-processes must start at most one and be nonnegative")

    failures = []
    monotone_checks = 0
    coherence_checks = 0
    equality_checks = 0
    for t, active in enumerate(active_masks):
        for subset, row in paths.items():
            if t:
                monotone_checks += 1
                if row[t] < row[t - 1]:
                    failures.append(("time_monotonicity", subset, t))
            restricted = subset & active
            if restricted:
                coherence_checks += 1
                if paths[restricted][t] != row[t]:
                    failures.append(("projective_future_extension", subset, restricted, t))
                else:
                    equality_checks += 1
    return {
        "valid": not failures,
        "failures": failures,
        "time_monotonicity_checks": monotone_checks,
        "projective_checks": coherence_checks,
        "projective_equalities": equality_checks,
    }


def _closure_check(
    rejection_mask: int,
    active_mask: int,
    time_index: int,
    adjusted: dict[int, Sequence[Fraction]],
    alpha: Fraction,
):
    count = rejection_mask.bit_count()
    worst = None
    worst_subset = 0
    checked = 0
    subset = active_mask
    while subset:
        fdp = Fraction((subset & rejection_mask).bit_count(), max(count, 1))
        slack = alpha * adjusted[subset][time_index] - fdp
        if worst is None or slack < worst:
            worst, worst_subset = slack, subset
        checked += 1
        subset = (subset - 1) & active_mask
    return worst is None or worst >= 0, worst, worst_subset, checked


def _prefix_digest(
    x, active, scores, alpha, values, costs, capacities, thresholds,
    increments, t,
):
    active_universe = set()
    for mask in active[: t + 1]:
        active_universe.update(i for i in range(len(scores)) if (mask >> i) & 1)
    payload = {
        "time": t,
        "active_masks": list(active[: t + 1]),
        "active_coordinate_increments": {
            str(i): [fraction_pair(x[i][u]) for u in range(t + 1)
                     if (active[u] >> i) & 1]
            for i in sorted(active_universe)
        },
        "active_scores": {
            str(i): fraction_pair(scores[i]) for i in sorted(active_universe)
        },
        "alpha": fraction_pair(alpha),
        "values": {str(i): fraction_pair(values[i]) for i in sorted(active_universe)},
        "costs": {str(i): fraction_pair(costs[i]) for i in sorted(active_universe)},
        "capacity_prefix": [fraction_pair(v) for v in capacities[: t + 1]],
        "adjuster_thresholds": [fraction_pair(v) for v in thresholds],
        "adjuster_increments": [fraction_pair(v) for v in increments],
    }
    return _digest(payload)


def select_self_financing_capacity_actions(
    increment_paths,
    active_masks,
    priority_scores,
    alpha,
    values,
    costs,
    cumulative_capacity,
    thresholds,
    adjuster_increments,
    *,
    exhaustive_limit: int = 8,
) -> dict:
    x, active, scores, level, vals, c, capacity = _normalise(
        increment_paths, active_masks, priority_scores, alpha, values, costs,
        cumulative_capacity,
    )
    n = len(x)
    if n > exhaustive_limit:
        raise ValueError(f"exact self-financing oracle limited to {exhaustive_limit} hypotheses")
    raw, shares = build_self_financing_intersection_paths(x, active, scores)
    adjusted, adjuster_budget = _adjust_paths(raw, thresholds, adjuster_increments)
    collection = _verify_dynamic_collection(adjusted, active)
    if not collection["valid"]:
        raise ValueError(f"dynamic collection failed coherence/persistence: {collection}")

    c_thresholds = tuple(as_fraction(v) for v in thresholds)
    a_increments = tuple(as_fraction(v) for v in adjuster_increments)
    decisions = []
    previous = 0
    for t, active_mask in enumerate(active):
        best, best_value, best_cost = None, None, None
        feasible_count = certified_count = 0
        for candidate in range(1 << n):
            if candidate & active_mask != candidate or candidate & previous != previous:
                continue
            candidate_cost = sum(
                (c[i] for i in range(n) if (candidate >> i) & 1), Fraction(0)
            )
            if candidate_cost > capacity[t]:
                continue
            feasible_count += 1
            certified, _, _, _ = _closure_check(
                candidate, active_mask, t, adjusted, level
            )
            if not certified:
                continue
            certified_count += 1
            candidate_value = sum(
                (vals[i] for i in range(n) if (candidate >> i) & 1), Fraction(0)
            )
            better = best is None or candidate_value > best_value
            if not better and candidate_value == best_value:
                better = candidate_cost < best_cost
                if not better and candidate_cost == best_cost:
                    best_tuple = tuple(i for i in range(n) if (best >> i) & 1)
                    candidate_tuple = tuple(i for i in range(n) if (candidate >> i) & 1)
                    better = candidate_tuple < best_tuple
            if better:
                best, best_value, best_cost = candidate, candidate_value, candidate_cost
        if best is None:
            raise ArithmeticError("no feasible action; previous action must remain feasible")
        certified, slack, worst, checked = _closure_check(
            best, active_mask, t, adjusted, level
        )
        if not certified:
            raise ArithmeticError("selected action lacks dynamic closure certificate")
        decisions.append({
            "time": t,
            "active_mask": active_mask,
            "rejection_mask": best,
            "rejections": [i + 1 for i in range(n) if (best >> i) & 1],
            "utility": fraction_pair(best_value),
            "cost": fraction_pair(best_cost),
            "capacity": fraction_pair(capacity[t]),
            "minimum_closure_slack": fraction_pair(slack) if slack is not None else None,
            "worst_intersection": worst,
            "intersections_checked": checked,
            "feasible_candidates": feasible_count,
            "certified_candidates": certified_count,
            "shares": [fraction_pair(v) for v in shares[t]],
            "prefix_sha256": _prefix_digest(
                x, active, scores, level, vals, c, capacity, c_thresholds,
                a_increments, t,
            ),
        })
        previous = best

    verification = verify_self_financing_sequence(
        decisions, x, active, scores, level, vals, c, capacity,
        c_thresholds, a_increments,
    )
    if not verification["valid"]:
        raise ArithmeticError(f"independent replay failed: {verification}")
    return {
        "schema": "symthaea-self-financing-projective-eclosure-v1",
        "status": "EXACT_FINITE_REFERENCE_ORACLE_NOT_PRODUCTION",
        "theorem_contract": (
            "Conditional on each X_i,t being a nonnegative one-step conditional "
            "e-value in a common global filtration under H_i, and deterministic "
            "nested activation, each intersection product is a test supermartingale; "
            "common running-max adjuster with integral budget <= 1 gives time-monotone "
            "projective e-processes; dynamic e-closure then supplies SupFDR."
        ),
        "priority_scores": [fraction_pair(v) for v in scores],
        "alpha": fraction_pair(level),
        "adjuster_integral_budget": fraction_pair(adjuster_budget),
        "dynamic_collection": collection,
        "decisions": decisions,
        "verification": verification,
        "limitations": [
            "one-step conditional e-value expectation is assumed, not proved from the observed paths",
            "requires one common global filtration",
            "active sets are deterministic and nested",
            "scores are fixed; future adaptive score policies need a predictable-measurability proof",
            "cumulative capacity only; no per-period queue-service guarantee",
            "finite exhaustive oracle with exponential complexity",
        ],
    }


def verify_self_financing_sequence(
    decisions, x, active, scores, alpha, values, costs, capacities,
    thresholds, adjuster_increments,
) -> dict:
    """Independent deterministic replay of all certificates and optimum choices."""
    raw, shares = build_self_financing_intersection_paths(x, scores, active)
    adjusted, budget = _adjust_paths(raw, thresholds, adjuster_increments)
    collection = _verify_dynamic_collection(adjusted, active)
    failures = []
    if not collection["valid"]:
        failures.append(("invalid_dynamic_collection", collection["failures"]))
    previous = 0
    closure_checks = candidate_checks = optimality_prefixes = 0
    for t, record in enumerate(decisions):
        if t >= len(active):
            failures.append(("extra_decision", t))
            continue
        if not isinstance(record, dict):
            failures.append(("invalid_record", t))
            continue
        mask = record.get("rejection_mask")
        if isinstance(mask, bool) or not isinstance(mask, int) or not 0 <= mask < (1 << len(x)):
            failures.append(("invalid_mask", t))
            continue
        if mask & active[t] != mask:
            failures.append(("inactive_rejection", t))
        if mask & previous != previous:
            failures.append(("non_nested_rejection", t))
        cost = sum((costs[i] for i in range(len(x)) if (mask >> i) & 1), Fraction(0))
        value = sum((values[i] for i in range(len(x)) if (mask >> i) & 1), Fraction(0))
        if cost > capacities[t]:
            failures.append(("capacity_exceeded", t))
        closure = _closure_check(mask, active[t], t, adjusted, alpha)
        closure_checks += closure[3]
        if not closure[0]:
            failures.append(("uncertified_rejection", t))
        expected = {
            "time": t,
            "active_mask": active[t],
            "rejections": [i + 1 for i in range(len(x)) if (mask >> i) & 1],
            "utility": fraction_pair(value),
            "cost": fraction_pair(cost),
            "capacity": fraction_pair(capacities[t]),
            "minimum_closure_slack": fraction_pair(closure[1]) if closure[1] is not None else None,
            "worst_intersection": closure[2],
            "intersections_checked": closure[3],
            "shares": [fraction_pair(v) for v in shares[t]],
            "prefix_sha256": _prefix_digest(
                x, active, scores, alpha, values, costs, capacities,
                tuple(as_fraction(v) for v in thresholds),
                tuple(as_fraction(v) for v in adjuster_increments), t,
            ),
        }
        for field, expected_value in expected.items():
            if record.get(field) != expected_value:
                failures.append(("receipt_mismatch", t, field))

        best, best_value, best_cost = None, None, None
        for candidate in range(1 << len(x)):
            if candidate & active[t] != candidate or candidate & previous != previous:
                continue
            candidate_cost = sum(
                (costs[i] for i in range(len(x)) if (candidate >> i) & 1), Fraction(0)
            )
            if candidate_cost > capacities[t]:
                continue
            candidate_checks += 1
            check = _closure_check(candidate, active[t], t, adjusted, alpha)
            if not check[0]:
                continue
            candidate_value = sum(
                (values[i] for i in range(len(x)) if (candidate >> i) & 1), Fraction(0)
            )
            better = best is None or candidate_value > best_value
            if not better and candidate_value == best_value:
                better = candidate_cost < best_cost
                if not better and candidate_cost == best_cost:
                    better = tuple(i for i in range(len(x)) if (candidate >> i) & 1) < tuple(
                        i for i in range(len(x)) if (best >> i) & 1
                    )
            if better:
                best, best_value, best_cost = candidate, candidate_value, candidate_cost
        optimality_prefixes += 1
        if best != mask:
            failures.append(("utility_optimality_failure", t, best, mask))
        previous = mask

    if len(decisions) != len(active):
        failures.append("decision_count_mismatch")
    return {
        "valid": not failures,
        "failures": failures,
        "decisions_checked": min(len(decisions), len(active)),
        "closure_intersections_checked": closure_checks,
        "feasible_candidate_sets_checked": candidate_checks,
        "optimality_prefixes_checked": optimality_prefixes,
        "optimality_verified": optimality_prefixes == len(active) and not any(
            isinstance(f, tuple) and f and f[0] == "utility_optimality_failure"
            for f in failures
        ),
        "adjuster_integral_budget": fraction_pair(budget),
    }


if __name__ == "__main__":
    raise SystemExit("Run test_self_financing_projective_eclosure.py for exact fixtures.")
