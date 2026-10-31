#!/usr/bin/env python3
"""Exact dynamic e-closure oracle for projective any-time-valid evidence.

The module constructs intersection e-process paths from globally weighted
coordinate e-processes, applies an exact dyadic running-maximum adjuster to
obtain time-monotone projective certificates, and selects capacity-aware
nested actions from the resulting dynamic e-closure menus.

It checks pathwise algebraic conditions and decision optimality. It cannot
prove that supplied coordinate paths are valid e-processes; that is an
upstream theorem/filtration contract.
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
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def dyadic_adjuster(x: Fraction) -> int:
    """A(x)=sum_{k>=1} 1{x >= 2^k}; its integral A(x)/x^2 is exactly 1."""
    x = as_fraction(x)
    if x < 1:
        raise ValueError("adjuster input must be at least one")
    threshold = Fraction(2)
    increments = 0
    while threshold <= x:
        increments += 1
        threshold *= 2
    return increments


def step_adjuster(
    x: Fraction,
    thresholds: Sequence[Fraction],
    increments: Sequence[Fraction],
) -> Fraction:
    """General exact step adjuster with integral sum_j increment_j/threshold_j."""
    x = as_fraction(x)
    if x < 1:
        raise ValueError("adjuster input must be at least one")
    if len(thresholds) != len(increments):
        raise ValueError("threshold and increment lengths differ")
    result = Fraction(0)
    for threshold, increment in zip(thresholds, increments):
        if x >= threshold:
            result += increment
    return result


def validate_step_adjuster(
    thresholds: Sequence[int | float | str | Fraction],
    increments: Sequence[int | float | str | Fraction],
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...], Fraction]:
    c = tuple(as_fraction(x) for x in thresholds)
    a = tuple(as_fraction(x) for x in increments)
    if len(c) != len(a):
        raise ValueError("threshold and increment lengths differ")
    if any(x <= 1 for x in c):
        raise ValueError("adjuster thresholds must exceed one")
    if any(c[i] >= c[i + 1] for i in range(len(c) - 1)):
        raise ValueError("adjuster thresholds must be strictly increasing")
    if any(x < 0 for x in a):
        raise ValueError("adjuster increments must be nonnegative")
    budget = sum((a[i] / c[i] for i in range(len(c))), Fraction(0))
    if budget > 1:
        raise ValueError("adjuster violates integral budget sum(increment/threshold) <= 1")
    return c, a, budget


def dyadic_adjuster_spec(max_threshold_power: int = 16):
    if not isinstance(max_threshold_power, int) or max_threshold_power < 1:
        raise ValueError("max_threshold_power must be positive")
    thresholds = [2**k for k in range(1, max_threshold_power + 1)]
    increments = [1] * max_threshold_power
    return thresholds, increments


def _normalise_stream(
    coordinate_paths: Sequence[Sequence[int | float | str | Fraction]],
    weights: Sequence[int | float | str | Fraction],
    active_masks: Sequence[int],
    alpha: int | float | str | Fraction,
    values: Sequence[int | float | str | Fraction],
    costs: Sequence[int | float | str | Fraction],
    cumulative_capacity: Sequence[int | float | str | Fraction],
) -> tuple[
    tuple[tuple[Fraction, ...], ...],
    tuple[Fraction, ...],
    tuple[int, ...],
    Fraction,
    tuple[Fraction, ...],
    tuple[Fraction, ...],
    tuple[Fraction, ...],
]:
    n = len(coordinate_paths)
    t_count = len(active_masks)
    if n < 1 or t_count < 1:
        raise ValueError("at least one coordinate and one global time are required")
    if not (len(weights) == len(values) == len(costs) == n):
        raise ValueError("weights, values and costs must match coordinate count")
    if len(cumulative_capacity) != t_count:
        raise ValueError("capacity path must match the number of global updates")

    paths = tuple(tuple(as_fraction(x) for x in row) for row in coordinate_paths)
    if any(len(row) != t_count for row in paths):
        raise ValueError("every coordinate e-process path must have one value per update")
    if any(x < 0 for row in paths for x in row):
        raise ValueError("coordinate e-process values must be nonnegative")

    w = tuple(as_fraction(x) for x in weights)
    if any(x < 0 for x in w) or sum(w) > 1:
        raise ValueError("global weights must be nonnegative and sum to at most one")

    active = tuple(active_masks)
    full_mask = (1 << n) - 1
    previous = 0
    for t, mask in enumerate(active):
        if isinstance(mask, bool) or not isinstance(mask, int):
            raise ValueError("active masks must be integer bit masks")
        if mask < 0 or mask & ~full_mask:
            raise ValueError("active mask contains an unknown coordinate")
        if (mask & previous) != previous:
            raise ValueError("active hypothesis sets must be nested")
        previous = mask
        for i in range(n):
            if not ((mask >> i) & 1) and paths[i][t] != 1:
                raise ValueError(
                    f"dormant coordinate {i + 1} must have neutral value one at time {t}"
                )

    a = as_fraction(alpha)
    if not 0 < a < 1:
        raise ValueError("alpha must lie in (0,1)")
    v = tuple(as_fraction(x) for x in values)
    c = tuple(as_fraction(x) for x in costs)
    b = tuple(as_fraction(x) for x in cumulative_capacity)
    if any(x < 0 for x in v):
        raise ValueError("utilities must be nonnegative")
    if any(x < 0 for x in c):
        raise ValueError("costs must be nonnegative")
    if any(x < 0 for x in b):
        raise ValueError("capacities must be nonnegative")
    if any(b[t] < b[t - 1] for t in range(1, t_count)):
        raise ValueError("cumulative capacity must be nondecreasing")

    return paths, w, active, a, v, c, b


def build_projective_intersection_paths(
    coordinate_paths: Sequence[Sequence[int | float | str | Fraction]],
    weights: Sequence[int | float | str | Fraction],
    active_masks: Sequence[int],
) -> dict[int, tuple[Fraction, ...]]:
    """Build E_t^S = 1 + sum_{i in S} w_i (M_{i,t} - 1) for all nonempty S.

    Under the external assumption that every M_i is a global e-process for H_i,
    dormant at one before activation, and sum_i w_i <= 1, each intersection
    process is an e-process under arbitrary cross-coordinate dependence.
    """
    n = len(coordinate_paths)
    t_count = len(active_masks)
    if n < 1 or t_count < 1 or len(weights) != n:
        raise ValueError("invalid coordinate/weight/time dimensions")
    paths = tuple(tuple(as_fraction(x) for x in row) for row in coordinate_paths)
    w = tuple(as_fraction(x) for x in weights)
    if any(len(row) != t_count for row in paths):
        raise ValueError("coordinate path length mismatch")
    if any(x < 0 for row in paths for x in row):
        raise ValueError("coordinate e-values must be nonnegative")
    if any(x < 0 for x in w) or sum(w) > 1:
        raise ValueError("weights must be nonnegative and globally summable to <= 1")
    for t, mask in enumerate(active_masks):
        for i in range(n):
            if not ((mask >> i) & 1) and paths[i][t] != 1:
                raise ValueError("inactive coordinates must remain exactly one")

    result: dict[int, tuple[Fraction, ...]] = {}
    for subset in range(1, 1 << n):
        values = []
        for t in range(t_count):
            value = Fraction(1)
            for i in range(n):
                if (subset >> i) & 1:
                    value += w[i] * (paths[i][t] - 1)
            if value < 0:
                raise ArithmeticError("global weighted merger became negative")
            values.append(value)
        result[subset] = tuple(values)
    return result


def adjust_projective_paths(
    raw_paths: dict[int, Sequence[Fraction]],
    *,
    thresholds: Sequence[int | float | str | Fraction] | None = None,
    increments: Sequence[int | float | str | Fraction] | None = None,
    dyadic_power_limit: int = 16,
) -> tuple[dict[int, tuple[Fraction, ...]], dict[str, Any]]:
    """Apply one common adjuster to running maxima, preserving projectivity."""
    if (thresholds is None) != (increments is None):
        raise ValueError("supply both thresholds and increments, or neither")
    if thresholds is None:
        thresholds, increments = dyadic_adjuster_spec(dyadic_power_limit)
    c, a, budget = validate_step_adjuster(thresholds, increments)

    adjusted: dict[int, tuple[Fraction, ...]] = {}
    time_count = None
    for subset, row in raw_paths.items():
        values = tuple(as_fraction(x) for x in row)
        if time_count is None:
            time_count = len(values)
        if len(values) != time_count:
            raise ValueError("intersection paths have different lengths")
        running_max = Fraction(1)
        result = []
        for value in values:
            if value < 0:
                raise ValueError("intersection e-process values must be nonnegative")
            running_max = max(running_max, value)
            result.append(step_adjuster(running_max, c, a))
        adjusted[subset] = tuple(result)

    return adjusted, {
        "kind": "step_adjuster",
        "thresholds": c,
        "increments": a,
        "integral_budget": budget,
    }


def verify_dynamic_collection(
    adjusted_paths: dict[int, Sequence[int | float | str | Fraction]],
    active_masks: Sequence[int],
    *,
    tolerance: Fraction = Fraction(0),
) -> dict[str, Any]:
    """Check pathwise time-monotonicity and future-extension coherence.

    This is a pathwise verifier only. It does not establish e-process validity
    under a probability law or global filtration.
    """
    t_count = len(active_masks)
    if t_count < 1:
        raise ValueError("active_masks must be nonempty")
    subsets = sorted(adjusted_paths)
    if not subsets:
        raise ValueError("intersection collection is empty")
    full_mask = max(subsets)
    if full_mask <= 0 or subsets != list(range(1, full_mask + 1)):
        raise ValueError("intersection collection must contain every nonempty subset mask")
    n = full_mask.bit_length()
    for mask in active_masks:
        if isinstance(mask, bool) or not isinstance(mask, int) or mask < 0 or mask >= (1 << n):
            raise ValueError("invalid active mask")
    paths = {
        mask: tuple(as_fraction(x) for x in adjusted_paths[mask])
        for mask in subsets
    }
    if any(len(row) != t_count for row in paths.values()):
        raise ValueError("intersection path length does not match active timeline")
    if any(x < 0 for row in paths.values() for x in row):
        raise ValueError("adjusted intersection values must be nonnegative")

    failures = []
    monotonicity_checks = 0
    coherence_checks = 0
    projective_equalities = 0
    max_coherence_violation = Fraction(0)

    previous_active = 0
    for t, active in enumerate(active_masks):
        if (active & previous_active) != previous_active:
            failures.append(("non_nested_active_set", t))
        previous_active = active
        for subset in subsets:
            row = paths[subset]
            if t > 0:
                monotonicity_checks += 1
                if row[t] < row[t - 1]:
                    failures.append(("time_monotonicity_failure", subset, t))
            current = subset & active
            if current:
                coherence_checks += 1
                lhs = paths[current][t]
                rhs = row[t]
                violation = lhs - rhs
                if violation > max_coherence_violation:
                    max_coherence_violation = violation
                if violation > tolerance:
                    failures.append(("future_extension_coherence_failure", subset, current, t))
                if lhs == rhs:
                    projective_equalities += 1

    return {
        "valid": not failures,
        "failures": failures,
        "time_monotonicity_checks": monotonicity_checks,
        "future_extension_checks": coherence_checks,
        "projective_equalities": projective_equalities,
        "maximum_coherence_violation": max_coherence_violation,
        "scope": "pathwise monotonicity/coherence only; e-process probability contract external",
    }


def _closure_check(
    rejection_mask: int,
    active_mask: int,
    time_index: int,
    adjusted_paths: dict[int, Sequence[Fraction]],
    alpha: Fraction,
) -> dict[str, Any]:
    count = rejection_mask.bit_count()
    worst_slack = None
    worst_intersection = 0
    checked = 0
    subset = active_mask
    while subset:
        fdp = Fraction((subset & rejection_mask).bit_count(), max(count, 1))
        slack = alpha * adjusted_paths[subset][time_index] - fdp
        checked += 1
        if worst_slack is None or slack < worst_slack:
            worst_slack = slack
            worst_intersection = subset
        subset = (subset - 1) & active_mask
    return {
        "certified": worst_slack is None or worst_slack >= 0,
        "minimum_slack": worst_slack,
        "worst_intersection": worst_intersection,
        "intersections_checked": checked,
    }


def _prefix_digest(
    coordinate_paths: tuple[tuple[Fraction, ...], ...],
    weights: tuple[Fraction, ...],
    active_masks: tuple[int, ...],
    alpha: Fraction,
    values: tuple[Fraction, ...],
    costs: tuple[Fraction, ...],
    capacities: tuple[Fraction, ...],
    time_index: int,
    adjuster_spec: dict[str, Any],
) -> str:
    payload = {
        "time": time_index,
        "coordinate_paths": [
            [fraction_pair(x) for x in row[: time_index + 1]]
            for row in coordinate_paths
        ],
        "weights": [fraction_pair(x) for x in weights],
        "active_masks": list(active_masks[: time_index + 1]),
        "alpha": fraction_pair(alpha),
        "values": [fraction_pair(x) for x in values],
        "costs": [fraction_pair(x) for x in costs],
        "capacities": [
            fraction_pair(x) for x in capacities[: time_index + 1]
        ],
        "adjuster": {
            "thresholds": [fraction_pair(x) for x in adjuster_spec["thresholds"]],
            "increments": [fraction_pair(x) for x in adjuster_spec["increments"]],
            "integral_budget": fraction_pair(adjuster_spec["integral_budget"]),
        },
    }
    return _digest(payload)


def select_dynamic_capacity_actions(
    coordinate_paths: Sequence[Sequence[int | float | str | Fraction]],
    weights: Sequence[int | float | str | Fraction],
    active_masks: Sequence[int],
    alpha: int | float | str | Fraction,
    values: Sequence[int | float | str | Fraction],
    costs: Sequence[int | float | str | Fraction],
    cumulative_capacity: Sequence[int | float | str | Fraction],
    *,
    thresholds: Sequence[int | float | str | Fraction] | None = None,
    increments: Sequence[int | float | str | Fraction] | None = None,
    dyadic_power_limit: int = 16,
    exhaustive_limit: int = 8,
) -> dict[str, Any]:
    """Select nested actions from a persistent dynamic e-closure menu.

    A SupFDR claim is conditional on each input coordinate path being a valid
    global e-process for its null in the common filtration. Fixed global
    weights make the intersection collection projective under arbitrary
    cross-stream dependence. The common step adjuster converts running maxima
    into time-monotone e-processes. This function checks the pathwise algebra,
    not the upstream stochastic assumptions.
    """
    paths, w, active, level, v, c, capacity = _normalise_stream(
        coordinate_paths, weights, active_masks, alpha, values, costs,
        cumulative_capacity,
    )
    n = len(paths)
    if n > exhaustive_limit:
        raise ValueError(
            f"exact dynamic reference oracle limited to {exhaustive_limit} coordinates"
        )

    raw = build_projective_intersection_paths(paths, w, active)
    adjusted, adjuster = adjust_projective_paths(
        raw, thresholds=thresholds, increments=increments,
        dyadic_power_limit=dyadic_power_limit,
    )
    collection_check = verify_dynamic_collection(adjusted, active)
    if not collection_check["valid"]:
        raise ValueError(f"dynamic intersection collection is invalid: {collection_check}")

    decisions = []
    previous = 0
    for t, active_mask in enumerate(active):
        best_mask = None
        best_value = None
        best_cost = None
        candidate_count = 0
        certified_count = 0
        for candidate in range(1 << n):
            if candidate & active_mask != candidate:
                continue
            if candidate & previous != previous:
                continue
            candidate_cost = sum(
                (c[i] for i in range(n) if (candidate >> i) & 1),
                Fraction(0),
            )
            if candidate_cost > capacity[t]:
                continue
            candidate_count += 1
            check = _closure_check(candidate, active_mask, t, adjusted, level)
            if not check["certified"]:
                continue
            certified_count += 1
            candidate_value = sum(
                (v[i] for i in range(n) if (candidate >> i) & 1),
                Fraction(0),
            )
            better = best_mask is None or candidate_value > best_value
            if not better and candidate_value == best_value:
                better = candidate_cost < best_cost
                if not better and candidate_cost == best_cost:
                    previous_tuple = tuple(
                        i for i in range(n) if (best_mask >> i) & 1
                    )
                    candidate_tuple = tuple(
                        i for i in range(n) if (candidate >> i) & 1
                    )
                    better = candidate_tuple < previous_tuple
            if better:
                best_mask = candidate
                best_value = candidate_value
                best_cost = candidate_cost
        if best_mask is None:
            raise ArithmeticError("no feasible candidate; previous decision should remain feasible")

        closure = _closure_check(best_mask, active_mask, t, adjusted, level)
        decisions.append({
            "time": t,
            "active_mask": active_mask,
            "rejection_mask": best_mask,
            "rejections": [i + 1 for i in range(n) if (best_mask >> i) & 1],
            "utility": fraction_pair(best_value),
            "cumulative_cost": fraction_pair(best_cost),
            "cumulative_capacity": fraction_pair(capacity[t]),
            "minimum_closure_slack": (
                fraction_pair(closure["minimum_slack"])
                if closure["minimum_slack"] is not None else None
            ),
            "worst_intersection_mask": closure["worst_intersection"],
            "intersections_checked": closure["intersections_checked"],
            "candidate_sets_checked": candidate_count,
            "certified_candidates": certified_count,
            "prefix_sha256": _prefix_digest(
                paths, w, active, level, v, c, capacity, t, adjuster
            ),
        })
        previous = best_mask

    verification = verify_dynamic_decision_sequence(
        decisions, paths, w, active, level, v, c, capacity, adjuster
    )
    if not verification["valid"]:
        raise ArithmeticError(f"independent dynamic selection check failed: {verification}")

    return {
        "schema": "symthaea-dynamic-eclosure-capacity-v1",
        "status": "EXACT_FINITE_REFERENCE_ORACLE_NOT_PRODUCTION",
        "validity_contract": (
            "SupFDR <= alpha conditional on globally valid coordinate e-processes "
            "in one common filtration; fixed globally summable weights; dormant "
            "value one; future-extension coherence; and the step-adjuster integral budget"
        ),
        "weights": [fraction_pair(x) for x in w],
        "alpha": fraction_pair(level),
        "adjuster": {
            "kind": adjuster["kind"],
            "thresholds": [fraction_pair(x) for x in adjuster["thresholds"]],
            "increments": [fraction_pair(x) for x in adjuster["increments"]],
            "integral_budget": fraction_pair(adjuster["integral_budget"]),
        },
        "collection_check": {
            key: value for key, value in collection_check.items()
            if key not in ("maximum_coherence_violation", "failures")
        } | {
            "maximum_coherence_violation": fraction_pair(
                collection_check["maximum_coherence_violation"]
            )
        },
        "decisions": decisions,
        "verification": verification,
        "limitations": [
            "coordinate e-process validity in the common filtration is assumed, not proved by a finite path",
            "fixed global weights; no active-set renormalization",
            "cumulative capacity only; no per-period queue stability",
            "finite exhaustive reference oracle, not a scalable optimizer",
            "no claim for data-adaptive weights without a separate stochastic proof",
        ],
    }


def verify_dynamic_decision_sequence(
    decisions: Sequence[dict],
    coordinate_paths: Sequence[Sequence[Fraction]],
    weights: Sequence[Fraction],
    active_masks: Sequence[int],
    alpha: Fraction,
    values: Sequence[Fraction],
    costs: Sequence[Fraction],
    capacities: Sequence[Fraction],
    adjuster_spec: dict[str, Any],
) -> dict[str, Any]:
    """Independent dynamic replay including optimality over all feasible sets."""
    paths = tuple(tuple(as_fraction(x) for x in row) for row in coordinate_paths)
    w = tuple(as_fraction(x) for x in weights)
    active = tuple(active_masks)
    level = as_fraction(alpha)
    v = tuple(as_fraction(x) for x in values)
    c = tuple(as_fraction(x) for x in costs)
    capacity = tuple(as_fraction(x) for x in capacities)
    n = len(paths)
    t_count = len(active)
    failures = []
    if len(decisions) != t_count:
        failures.append("decision count mismatch")

    raw = build_projective_intersection_paths(paths, w, active)
    adjusted, _ = adjust_projective_paths(
        raw,
        thresholds=adjuster_spec["thresholds"],
        increments=adjuster_spec["increments"],
    )
    collection_check = verify_dynamic_collection(adjusted, active)
    if not collection_check["valid"]:
        failures.append(("invalid_dynamic_collection", collection_check["failures"]))

    previous = 0
    intersections_checked = 0
    feasible_candidates = 0
    certified_candidates = 0
    optimality_prefixes = 0
    for t, record in enumerate(decisions):
        if t >= t_count:
            failures.append(("extra_decision", t))
            continue
        if not isinstance(record, dict):
            failures.append(("invalid_record", t))
            continue
        mask = record.get("rejection_mask")
        if isinstance(mask, bool) or not isinstance(mask, int) or mask < 0 or mask >= (1 << n):
            failures.append(("invalid_rejection_mask", t))
            continue
        if mask & active[t] != mask:
            failures.append(("inactive_hypothesis_rejected", t))
        if mask & previous != previous:
            failures.append(("non_nested_selection", t))

        actual_cost = sum(
            (c[i] for i in range(n) if (mask >> i) & 1), Fraction(0)
        )
        actual_value = sum(
            (v[i] for i in range(n) if (mask >> i) & 1), Fraction(0)
        )
        if actual_cost > capacity[t]:
            failures.append(("capacity_exceeded", t))
        for field, expected in (
            ("time", t),
            ("active_mask", active[t]),
            ("rejections", [i + 1 for i in range(n) if (mask >> i) & 1]),
            ("utility", fraction_pair(actual_value)),
            ("cumulative_cost", fraction_pair(actual_cost)),
            ("cumulative_capacity", fraction_pair(capacity[t])),
        ):
            if record.get(field) != expected:
                failures.append(("receipt_field_mismatch", t, field))

        closure = _closure_check(mask, active[t], t, adjusted, level)
        intersections_checked += closure["intersections_checked"]
        if not closure["certified"]:
            failures.append(("uncertified_selection", t))
        expected_slack = (
            fraction_pair(closure["minimum_slack"])
            if closure["minimum_slack"] is not None else None
        )
        if record.get("minimum_closure_slack") != expected_slack:
            failures.append(("closure_slack_mismatch", t))
        if record.get("worst_intersection_mask") != closure["worst_intersection"]:
            failures.append(("worst_intersection_mismatch", t))
        if record.get("intersections_checked") != closure["intersections_checked"]:
            failures.append(("intersection_count_mismatch", t))

        expected_digest = _prefix_digest(
            paths, w, active, level, v, c, capacity, t, {
                "thresholds": tuple(as_fraction(x) for x in adjuster_spec["thresholds"]),
                "increments": tuple(as_fraction(x) for x in adjuster_spec["increments"]),
                "integral_budget": sum(
                    (as_fraction(a) / as_fraction(th) for a, th in zip(
                        adjuster_spec["increments"], adjuster_spec["thresholds"]
                    )), Fraction(0)
                ),
            }
        )
        if record.get("prefix_sha256") != expected_digest:
            failures.append(("prefix_digest_mismatch", t))

        # Independent exhaustive optimality over every nested, certified,
        # capacity-feasible set in this dynamic menu.
        best = None
        best_val = None
        best_cost = None
        for candidate in range(1 << n):
            if candidate & active[t] != candidate or candidate & previous != previous:
                continue
            candidate_cost = sum(
                (c[i] for i in range(n) if (candidate >> i) & 1), Fraction(0)
            )
            if candidate_cost > capacity[t]:
                continue
            feasible_candidates += 1
            candidate_closure = _closure_check(
                candidate, active[t], t, adjusted, level
            )
            if not candidate_closure["certified"]:
                continue
            certified_candidates += 1
            candidate_value = sum(
                (v[i] for i in range(n) if (candidate >> i) & 1), Fraction(0)
            )
            better = best is None or candidate_value > best_val
            if not better and candidate_value == best_val:
                better = candidate_cost < best_cost
                if not better and candidate_cost == best_cost:
                    best_tuple = tuple(i for i in range(n) if (best >> i) & 1)
                    candidate_tuple = tuple(i for i in range(n) if (candidate >> i) & 1)
                    better = candidate_tuple < best_tuple
            if better:
                best, best_val, best_cost = candidate, candidate_value, candidate_cost
        optimality_prefixes += 1
        if best != mask:
            failures.append(("utility_optimality_failure", t, best, mask))
        previous = mask

    if len(decisions) != t_count:
        # Do not count a truncated decision stream as optimality-verified.
        optimality_prefixes = min(optimality_prefixes, len(decisions))

    return {
        "valid": not failures,
        "failures": failures,
        "decisions_checked": min(len(decisions), t_count),
        "intersection_constraints_checked": intersections_checked,
        "feasible_candidates_checked": feasible_candidates,
        "certified_candidates_checked": certified_candidates,
        "optimality_prefixes_checked": optimality_prefixes,
        "optimality_verified": optimality_prefixes == t_count and not any(
            isinstance(x, tuple) and x and x[0] == "utility_optimality_failure"
            for x in failures
        ),
        "dynamic_collection_verified": collection_check["valid"],
        "scope": "pathwise dynamic e-closure membership, coherence, persistence and utility optimality",
    }


if __name__ == "__main__":
    raise SystemExit("Run test_dynamic_eclosure_capacity_oracle.py for exact fixtures.")
