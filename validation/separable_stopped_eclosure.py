#!/usr/bin/env python3
"""Linear-time separation oracle for raw projective e-closure.

For globally weighted intersection e-values
    E_t^S = 1 + sum_{i in S} w_i (M_{i,t} - 1),
the condition that every intersection certifies a nonempty rejection set R
is equivalent to one sum of positive parts. This module supplies an exact
Fraction implementation, a nested capacity-aware small-stream optimizer,
and an independent exhaustive verifier used only as a test oracle.

Guarantee boundary: raw projective intersection e-processes plus
future-extension coherence support simultaneous stopped-FDR under the
dynamic e-closure theorem. These raw processes need not be time-monotone.
This module therefore makes NO SupFDR claim.
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


def _normalise(
    evalues: Sequence[int | float | str | Fraction],
    weights: Sequence[int | float | str | Fraction],
    active_mask: int,
    alpha: int | float | str | Fraction,
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...], int, Fraction]:
    if not evalues or len(evalues) != len(weights):
        raise ValueError("e-values and weights must have the same nonzero length")
    n = len(evalues)
    if isinstance(active_mask, bool) or not isinstance(active_mask, int):
        raise ValueError("active_mask must be an integer bit mask")
    if active_mask < 0 or active_mask >= (1 << n):
        raise ValueError("active_mask contains unknown hypotheses")
    e = tuple(as_fraction(x) for x in evalues)
    w = tuple(as_fraction(x) for x in weights)
    a = as_fraction(alpha)
    if any(x < 0 for x in e):
        raise ValueError("coordinate e-process values must be nonnegative")
    if any(x < 0 for x in w) or sum(w) > 1:
        raise ValueError("global weights must be nonnegative and sum to at most one")
    if not 0 < a < 1:
        raise ValueError("alpha must lie in (0,1)")
    for i in range(n):
        if not ((active_mask >> i) & 1) and e[i] != 1:
            raise ValueError("dormant coordinate e-values must equal one")
    return e, w, active_mask, a


def _submasks(mask: int):
    subset = mask
    while True:
        yield subset
        if subset == 0:
            break
        subset = (subset - 1) & mask


def exhaustive_intersection_certificate(
    rejection_mask: int,
    evalues: Sequence[int | float | str | Fraction],
    weights: Sequence[int | float | str | Fraction],
    active_mask: int,
    alpha: int | float | str | Fraction,
) -> dict:
    """Slow all-intersections reference implementation, for differential tests."""
    e, w, active, a = _normalise(evalues, weights, active_mask, alpha)
    n = len(e)
    if isinstance(rejection_mask, bool) or not isinstance(rejection_mask, int):
        raise ValueError("rejection_mask must be an integer bit mask")
    if rejection_mask < 0 or rejection_mask & ~active:
        raise ValueError("rejection set must be a subset of the active hypotheses")
    r = rejection_mask.bit_count()
    minimum_slack: Fraction | None = None
    worst_subset = 0
    checked = 0
    for subset in _submasks(active):
        intersection_count = (subset & rejection_mask).bit_count()
        fdp = Fraction(intersection_count, r) if r else Fraction(0)
        intersection_evalue = Fraction(1) + sum(
            (w[i] * (e[i] - 1) for i in range(n) if (subset >> i) & 1),
            Fraction(0),
        )
        slack = a * intersection_evalue - fdp
        checked += 1
        if minimum_slack is None or slack < minimum_slack:
            minimum_slack = slack
            worst_subset = subset
    assert minimum_slack is not None
    return {
        "certified": minimum_slack >= 0,
        "minimum_slack": minimum_slack,
        "worst_intersection_mask": worst_subset,
        "intersections_checked": checked,
    }


def verify_raw_projective_rejection_set(
    rejection_mask: int,
    evalues: Sequence[int | float | str | Fraction],
    weights: Sequence[int | float | str | Fraction],
    active_mask: int,
    alpha: int | float | str | Fraction,
) -> dict:
    """Check all closed-eBH intersections in O(N) using positive-part separation.

    For nonempty R, the constraint for S is equivalent to
      sum_{i in S} [1{i in R}/|R| - alpha*w_i*(M_i-1)] <= alpha.
    The maximum over S subset active is the sum of positive coefficients.
    Therefore the minimum intersection slack is alpha - sum_i coefficient_i^+.
    """
    e, w, active, a = _normalise(evalues, weights, active_mask, alpha)
    n = len(e)
    if isinstance(rejection_mask, bool) or not isinstance(rejection_mask, int):
        raise ValueError("rejection_mask must be an integer bit mask")
    if rejection_mask < 0 or rejection_mask & ~active:
        raise ValueError("rejection set must be a subset of active hypotheses")
    r = rejection_mask.bit_count()
    if r == 0:
        # FDP is zero for every intersection; the empty decision is always certified.
        return {
            "certified": True,
            "minimum_slack": None,
            "worst_intersection_mask": 0,
            "positive_part_sum": Fraction(0),
            "coefficients": tuple(Fraction(0) for _ in range(n)),
            "intersections_checked": 0,
            "separation_complexity": "O(N)",
        }

    coefficients = []
    positive_sum = Fraction(0)
    worst_subset = 0
    for i in range(n):
        if not ((active >> i) & 1):
            coefficient = Fraction(0)
        else:
            fdp_contribution = Fraction(1, r) if ((rejection_mask >> i) & 1) else Fraction(0)
            coefficient = fdp_contribution - a * w[i] * (e[i] - 1)
        coefficients.append(coefficient)
        if coefficient > 0:
            positive_sum += coefficient
            worst_subset |= 1 << i

    slack = a - positive_sum
    return {
        "certified": slack >= 0,
        "minimum_slack": slack,
        "worst_intersection_mask": worst_subset,
        "positive_part_sum": positive_sum,
        "coefficients": tuple(coefficients),
        "intersections_checked": 0,
        "separation_complexity": "O(N)",
    }


def _prefix_digest(
    epaths: Sequence[Sequence[Fraction]],
    weights: Sequence[Fraction],
    active_masks: Sequence[int],
    alpha: Fraction,
    values: Sequence[Fraction],
    costs: Sequence[Fraction],
    capacities: Sequence[Fraction],
    t: int,
) -> str:
    active_universe = 0
    for mask in active_masks[: t + 1]:
        active_universe |= mask
    indices = [i for i in range(len(weights)) if (active_universe >> i) & 1]
    payload = {
        "time": t,
        "active_masks": list(active_masks[: t + 1]),
        "coordinate_evalues": {
            str(i): [fraction_pair(epaths[i][j]) for j in range(t + 1)]
            for i in indices
        },
        "weights": {str(i): fraction_pair(weights[i]) for i in indices},
        "alpha": fraction_pair(alpha),
        "values": {str(i): fraction_pair(values[i]) for i in indices},
        "costs": {str(i): fraction_pair(costs[i]) for i in indices},
        "capacity": [fraction_pair(x) for x in capacities[: t + 1]],
    }
    return _digest(payload)


def _normalise_stream(
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
        raise ValueError("at least one hypothesis and update are required")
    if not (len(weights) == len(values) == len(costs) == n):
        raise ValueError("weights, values and costs must match the number of hypotheses")
    if len(capacities) != t_count:
        raise ValueError("capacity path must match the number of updates")
    paths = tuple(tuple(as_fraction(x) for x in row) for row in evalue_paths)
    if any(len(row) != t_count for row in paths):
        raise ValueError("each coordinate e-process must have one value per update")
    w = tuple(as_fraction(x) for x in weights)
    if any(x < 0 for x in w) or sum(w) > 1:
        raise ValueError("global weights must be nonnegative and sum to at most one")
    a = as_fraction(alpha)
    if not 0 < a < 1:
        raise ValueError("alpha must lie in (0,1)")
    v = tuple(as_fraction(x) for x in values)
    c = tuple(as_fraction(x) for x in costs)
    b = tuple(as_fraction(x) for x in capacities)
    if any(x < 0 for row in paths for x in row):
        raise ValueError("coordinate e-processes must be nonnegative")
    if any(x < 0 for x in v) or any(x < 0 for x in c) or any(x < 0 for x in b):
        raise ValueError("values, costs and capacities must be nonnegative")
    if any(b[t] < b[t - 1] for t in range(1, t_count)):
        raise ValueError("cumulative capacity must be nondecreasing")

    full_mask = (1 << n) - 1
    previous = 0
    for t, mask in enumerate(active_masks):
        if isinstance(mask, bool) or not isinstance(mask, int) or mask < 0 or mask & ~full_mask:
            raise ValueError("invalid active mask")
        if mask & previous != previous:
            raise ValueError("active hypotheses must arrive in nested order")
        previous = mask
        for i in range(n):
            if not ((mask >> i) & 1) and paths[i][t] != 1:
                raise ValueError("dormant e-process values must equal one")
    return paths, w, tuple(active_masks), a, v, c, b


def _mask_cost(mask: int, costs: tuple[Fraction, ...], n: int) -> Fraction:
    return sum((costs[i] for i in range(n) if (mask >> i) & 1), Fraction(0))


def _mask_value(mask: int, values: tuple[Fraction, ...], n: int) -> Fraction:
    return sum((values[i] for i in range(n) if (mask >> i) & 1), Fraction(0))


def select_raw_projective_capacity_actions(
    evalue_paths: Sequence[Sequence[int | float | str | Fraction]],
    weights: Sequence[int | float | str | Fraction],
    active_masks: Sequence[int],
    alpha: int | float | str | Fraction,
    values: Sequence[int | float | str | Fraction],
    costs: Sequence[int | float | str | Fraction],
    cumulative_capacity: Sequence[int | float | str | Fraction],
    *,
    exhaustive_limit: int = 16,
) -> dict:
    """Exhaustive small-stream optimizer with O(N) intersection membership checks.

    Previous decisions are irrevocable and cumulative capacity is nondecreasing.
    The selected decision is optimal for total predeclared utility within the
    nested, capacity-feasible raw e-closure menu.
    """
    paths, w, active, a, v, c, capacity = _normalise_stream(
        evalue_paths, weights, active_masks, alpha, values, costs,
        cumulative_capacity,
    )
    n = len(paths)
    if not isinstance(exhaustive_limit, int) or exhaustive_limit < 1:
        raise ValueError("exhaustive_limit must be positive")
    if n > exhaustive_limit:
        raise ValueError(
            f"reference optimizer limited to {exhaustive_limit} hypotheses; "
            "the exact membership oracle itself is linear-time"
        )

    decisions = []
    previous = 0
    total_candidate_checks = 0
    for t, active_mask in enumerate(active):
        best = None
        best_value = None
        best_cost = None
        feasible_count = certified_count = 0
        current_e = tuple(paths[i][t] for i in range(n))
        for candidate in range(1 << n):
            if candidate & active_mask != candidate or candidate & previous != previous:
                continue
            candidate_cost = _mask_cost(candidate, c, n)
            if candidate_cost > capacity[t]:
                continue
            feasible_count += 1
            total_candidate_checks += 1
            check = verify_raw_projective_rejection_set(
                candidate, current_e, w, active_mask, a
            )
            if not check["certified"]:
                continue
            certified_count += 1
            candidate_value = _mask_value(candidate, v, n)
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
            raise ArithmeticError("no capacity-feasible certified action")
        certificate = verify_raw_projective_rejection_set(
            best, current_e, w, active_mask, a
        )
        decisions.append({
            "time": t,
            "active_mask": active_mask,
            "rejection_mask": best,
            "rejections": [i + 1 for i in range(n) if (best >> i) & 1],
            "utility": fraction_pair(best_value),
            "cost": fraction_pair(best_cost),
            "capacity": fraction_pair(capacity[t]),
            "minimum_stopped_fdr_slack": (
                fraction_pair(certificate["minimum_slack"])
                if certificate["minimum_slack"] is not None else None
            ),
            "worst_intersection_mask": certificate["worst_intersection_mask"],
            "feasible_candidate_sets": feasible_count,
            "certified_candidate_sets": certified_count,
            "prefix_sha256": _prefix_digest(
                paths, w, active, a, v, c, capacity, t
            ),
        })
        previous = best

    verification = verify_raw_projective_sequence(
        decisions, paths, w, active, a, v, c, capacity
    )
    if not verification["valid"]:
        raise ArithmeticError(f"independent replay failed: {verification}")
    return {
        "schema": "symthaea-raw-projective-stopped-fdr-v1",
        "status": "EXACT_SMALL_REFERENCE_ORACLE_NOT_PRODUCTION",
        "guarantee_scope": (
            "simultaneous stopped-FDR via future-extension-coherent intersection "
            "e-processes; NOT SupFDR because raw intersection values are not "
            "required to be time-monotone"
        ),
        "weights": [fraction_pair(x) for x in w],
        "alpha": fraction_pair(a),
        "decisions": decisions,
        "verification": verification,
        "complexity": {
            "membership_per_candidate": "O(N)",
            "candidate_optimizer": "O(2^N * N) per update (reference enumeration)",
            "intersection_enumeration": "not used by production membership path",
        },
        "limitations": [
            "coordinate e-process validity in a common filtration is assumed, not proved from paths",
            "raw intersection e-processes need not be time-monotone",
            "therefore no SupFDR claim",
            "cumulative capacity and irrevocable nested decisions only",
            "small exact optimizer remains exponential in N",
        ],
    }


def verify_raw_projective_sequence(
    decisions: Sequence[dict],
    paths: tuple[tuple[Fraction, ...], ...],
    weights: tuple[Fraction, ...],
    active_masks: tuple[int, ...],
    alpha: Fraction,
    values: tuple[Fraction, ...],
    costs: tuple[Fraction, ...],
    capacities: tuple[Fraction, ...],
) -> dict:
    """Independent replay with O(N) certificate checks and exhaustive optimality."""
    paths, weights, active_masks, alpha, values, costs, capacities = _normalise_stream(
        paths, weights, active_masks, alpha, values, costs, capacities
    )
    n = len(paths)
    t_count = len(active_masks)
    failures = []
    intersections_not_enumerated = 0
    candidates_checked = 0
    certified_candidates = 0
    optimality_prefixes = 0
    previous = 0
    if len(decisions) != t_count:
        failures.append("decision count mismatch")

    for t, record in enumerate(decisions):
        if t >= t_count or not isinstance(record, dict):
            failures.append(("invalid_or_extra_decision", t))
            continue
        mask = record.get("rejection_mask")
        if isinstance(mask, bool) or not isinstance(mask, int) or mask < 0 or mask >= (1 << n):
            failures.append(("invalid_rejection_mask", t))
            continue
        if mask & active_masks[t] != mask:
            failures.append(("inactive_hypothesis_rejected", t))
        if mask & previous != previous:
            failures.append(("non_nested_decision", t))

        current_e = tuple(paths[i][t] for i in range(n))
        certificate = verify_raw_projective_rejection_set(
            mask, current_e, weights, active_masks[t], alpha
        )
        if not certificate["certified"]:
            failures.append(("uncertified_rejection_set", t))
        expected_value = _mask_value(mask, values, n)
        expected_cost = _mask_cost(mask, costs, n)
        if expected_cost > capacities[t]:
            failures.append(("capacity_exceeded", t))
        expected = {
            "time": t,
            "active_mask": active_masks[t],
            "rejections": [i + 1 for i in range(n) if (mask >> i) & 1],
            "utility": fraction_pair(expected_value),
            "cost": fraction_pair(expected_cost),
            "capacity": fraction_pair(capacities[t]),
            "minimum_stopped_fdr_slack": (
                fraction_pair(certificate["minimum_slack"])
                if certificate["minimum_slack"] is not None else None
            ),
            "worst_intersection_mask": certificate["worst_intersection_mask"],
            "prefix_sha256": _prefix_digest(
                paths, weights, active_masks, alpha, values, costs, capacities, t
            ),
        }
        for field, expected_value in expected.items():
            if record.get(field) != expected_value:
                failures.append(("receipt_mismatch", t, field))

        best = None
        best_value = None
        best_cost = None
        for candidate in range(1 << n):
            if candidate & active_masks[t] != candidate or candidate & previous != previous:
                continue
            candidate_cost = _mask_cost(candidate, costs, n)
            if candidate_cost > capacities[t]:
                continue
            candidates_checked += 1
            check = verify_raw_projective_rejection_set(
                candidate, current_e, weights, active_masks[t], alpha
            )
            if not check["certified"]:
                continue
            certified_candidates += 1
            candidate_value = _mask_value(candidate, values, n)
            better = best is None or candidate_value > best_value
            if not better and candidate_value == best_value:
                better = candidate_cost < best_cost
                if not better and candidate_cost == best_cost:
                    best_tuple = tuple(i for i in range(n) if (best >> i) & 1)
                    candidate_tuple = tuple(i for i in range(n) if (candidate >> i) & 1)
                    better = candidate_tuple < best_tuple
            if better:
                best, best_value, best_cost = candidate, candidate_value, candidate_cost
        optimality_prefixes += 1
        if best != mask:
            failures.append(("utility_optimality_failure", t, best, mask))
        previous = mask

    if len(decisions) != t_count:
        optimality_prefixes = min(optimality_prefixes, len(decisions))
    return {
        "valid": not failures,
        "failures": failures,
        "decisions_checked": min(len(decisions), t_count),
        "intersection_subsets_enumerated": intersections_not_enumerated,
        "candidate_sets_checked": candidates_checked,
        "certified_candidates_checked": certified_candidates,
        "optimality_prefixes_checked": optimality_prefixes,
        "optimality_verified": optimality_prefixes == t_count and not any(
            isinstance(x, tuple) and x and x[0] == "utility_optimality_failure"
            for x in failures
        ),
        "scope": "raw projective closure membership, stopped-FDR scope, capacity and utility optimality",
    }


if __name__ == "__main__":
    raise SystemExit("Run test_separable_stopped_eclosure.py for exact fixtures.")
