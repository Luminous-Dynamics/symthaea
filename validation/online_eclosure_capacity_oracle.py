#!/usr/bin/env python3
"""Exact small-stream oracle for capacity-aware selection inside online e-closure.

The online e-closure theorem certifies every R_t in C_t under SupFDR. This
module selects a nested sequence from those menus while enforcing cumulative
resource constraints. It is an exponential reference oracle, not a scalable
online testing algorithm and not an e-process validity checker.
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


def _fraction_pair(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def _canonical_digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def _normalise_inputs(
    evalues: Sequence[int | float | str | Fraction],
    gamma: Sequence[int | float | str | Fraction],
    delta: int | float | str | Fraction,
    values: Sequence[int | float | str | Fraction],
    costs: Sequence[int | float | str | Fraction],
    cumulative_capacity: Sequence[int | float | str | Fraction],
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...], Fraction,
           tuple[Fraction, ...], tuple[Fraction, ...], tuple[Fraction, ...]]:
    length = len(evalues)
    if length < 1:
        raise ValueError("the stream must contain at least one hypothesis")
    if not (len(gamma) == len(values) == len(costs) == len(cumulative_capacity) == length):
        raise ValueError("all stream inputs must have the same length")

    e = tuple(as_fraction(x) for x in evalues)
    g = tuple(as_fraction(x) for x in gamma)
    v = tuple(as_fraction(x) for x in values)
    c = tuple(as_fraction(x) for x in costs)
    b = tuple(as_fraction(x) for x in cumulative_capacity)
    d = as_fraction(delta)

    if any(x < 0 for x in e):
        raise ValueError("e-values must be nonnegative")
    if any(x < 0 for x in g) or sum(g) > 1:
        raise ValueError("gamma must be nonnegative and sum to at most one")
    if not 0 < d < 1:
        raise ValueError("delta must lie in (0,1)")
    if any(x < 0 for x in v):
        raise ValueError("utilities must be nonnegative")
    if any(x < 0 for x in c):
        raise ValueError("resource costs must be nonnegative")
    if any(x < 0 for x in b):
        raise ValueError("cumulative capacities must be nonnegative")
    if any(b[i] < b[i - 1] for i in range(1, length)):
        raise ValueError("cumulative capacity must be nondecreasing")
    return e, g, d, v, c, b


def _prefix_digest(
    e: tuple[Fraction, ...],
    gamma: tuple[Fraction, ...],
    delta: Fraction,
    values: tuple[Fraction, ...],
    costs: tuple[Fraction, ...],
    capacity: tuple[Fraction, ...],
    t: int,
) -> str:
    payload = {
        "t": t,
        "evalues": [_fraction_pair(x) for x in e[:t]],
        "gamma": [_fraction_pair(x) for x in gamma[:t]],
        "delta": _fraction_pair(delta),
        "values": [_fraction_pair(x) for x in values[:t]],
        "costs": [_fraction_pair(x) for x in costs[:t]],
        "capacity": [_fraction_pair(x) for x in capacity[:t]],
    }
    return _canonical_digest(payload)


def _intersection_evalue(
    subset_mask: int,
    evalues: tuple[Fraction, ...],
    gamma: tuple[Fraction, ...],
) -> Fraction:
    """E_S = sum_i gamma_rank_S(i) * E_i for the selected intersection S."""
    rank = 0
    result = Fraction(0)
    i = 0
    mask = subset_mask
    while mask:
        if mask & 1:
            result += gamma[rank] * evalues[i]
            rank += 1
        mask >>= 1
        i += 1
    return result


def verify_certified_set(
    rejection_mask: int,
    t: int,
    evalues: Sequence[int | float | str | Fraction],
    gamma: Sequence[int | float | str | Fraction],
    delta: int | float | str | Fraction,
) -> dict:
    """Check all 2^t e-closure intersections for one candidate rejection set."""
    if not isinstance(t, int) or not 1 <= t <= len(evalues) or len(gamma) < t:
        raise ValueError("invalid stream prefix length")
    if isinstance(rejection_mask, bool) or not isinstance(rejection_mask, int):
        raise ValueError("rejection_mask must be an integer bit mask")
    if rejection_mask < 0 or rejection_mask >= (1 << t):
        raise ValueError("rejection mask contains indices outside the prefix")

    e = tuple(as_fraction(x) for x in evalues[:t])
    g = tuple(as_fraction(x) for x in gamma[:t])
    level = as_fraction(delta)
    if any(x < 0 for x in e) or any(x < 0 for x in g) or sum(g) > 1:
        raise ValueError("invalid e-values or gamma weights")
    if not 0 < level < 1:
        raise ValueError("delta must lie in (0,1)")

    rejection_count = rejection_mask.bit_count()
    worst_slack: Fraction | None = None
    worst_subset = 0
    checked = 0
    for subset_mask in range(1 << t):
        intersection_count = (subset_mask & rejection_mask).bit_count()
        fdp = (
            Fraction(intersection_count, rejection_count)
            if rejection_count
            else Fraction(0)
        )
        e_subset = _intersection_evalue(subset_mask, e, g)
        slack = level * e_subset - fdp
        checked += 1
        if worst_slack is None or slack < worst_slack:
            worst_slack = slack
            worst_subset = subset_mask

    assert worst_slack is not None
    return {
        "certified": worst_slack >= 0,
        "minimum_closure_slack": worst_slack,
        "worst_intersection_mask": worst_subset,
        "candidate_mask": rejection_mask,
        "hypotheses_in_prefix": t,
        "intersections_checked": checked,
    }


def _mask_cost(mask: int, costs: tuple[Fraction, ...], t: int) -> Fraction:
    return sum((costs[i] for i in range(t) if (mask >> i) & 1), Fraction(0))


def _mask_value(mask: int, values: tuple[Fraction, ...], t: int) -> Fraction:
    return sum((values[i] for i in range(t) if (mask >> i) & 1), Fraction(0))


def select_capacity_aware_sequence(
    evalues: Sequence[int | float | str | Fraction],
    gamma: Sequence[int | float | str | Fraction],
    delta: int | float | str | Fraction,
    values: Sequence[int | float | str | Fraction],
    costs: Sequence[int | float | str | Fraction],
    cumulative_capacity: Sequence[int | float | str | Fraction],
    *,
    exhaustive_limit: int = 16,
) -> dict:
    """Select a nested, cumulative-capacity-feasible sequence from e-closure.

    The policy uses only prefix inputs at time t. It requires decisions to be
    irrevocable (R_{t-1} subset R_t) and capacity to be cumulative/nondecreasing.
    For a valid increasing e-collection, every selected R_t is in the online
    e-closure menu C_t, so the online SupFDR theorem applies. This function
    cannot verify the statistical validity of the incoming e-values.
    """
    e, g, level, v, c, capacity = _normalise_inputs(
        evalues, gamma, delta, values, costs, cumulative_capacity
    )
    total = len(e)
    if not isinstance(exhaustive_limit, int) or exhaustive_limit < 1:
        raise ValueError("exhaustive_limit must be positive")
    if total > exhaustive_limit:
        raise ValueError(
            f"exact reference oracle limited to {exhaustive_limit} hypotheses; "
            "use a scalable solver only after its independent verifier is defined"
        )

    previous_mask = 0
    decisions = []
    for t in range(1, total + 1):
        best_mask: int | None = None
        best_value: Fraction | None = None
        best_cost: Fraction | None = None
        candidate_count = 0
        closure_checks = 0
        selected_check = None

        for candidate_mask in range(1 << t):
            if (candidate_mask & previous_mask) != previous_mask:
                continue
            candidate_cost = _mask_cost(candidate_mask, c, t)
            if candidate_cost > capacity[t - 1]:
                continue
            candidate_count += 1
            check = verify_certified_set(candidate_mask, t, e, g, level)
            closure_checks += 1
            if not check["certified"]:
                continue
            candidate_value = _mask_value(candidate_mask, v, t)

            better = best_mask is None or candidate_value > best_value
            if not better and candidate_value == best_value:
                better = candidate_cost < best_cost
                if not better and candidate_cost == best_cost:
                    # Stable final tie break: lexicographically smaller 1-based
                    # selected-index tuple.
                    chosen_tuple = tuple(i for i in range(t) if (best_mask >> i) & 1)
                    candidate_tuple = tuple(i for i in range(t) if (candidate_mask >> i) & 1)
                    better = candidate_tuple < chosen_tuple
            if better:
                best_mask = candidate_mask
                best_value = candidate_value
                best_cost = candidate_cost
                selected_check = check

        if best_mask is None:
            raise ArithmeticError(
                "no capacity-feasible certified action; previous set should remain feasible"
            )
        if selected_check is None or not selected_check["certified"]:
            raise ArithmeticError("selected action lacks a valid e-closure certificate")

        decisions.append({
            "time": t,
            "rejection_mask": best_mask,
            "rejections": [i + 1 for i in range(t) if (best_mask >> i) & 1],
            "utility": _fraction_pair(best_value),
            "cumulative_cost": _fraction_pair(best_cost),
            "cumulative_capacity": _fraction_pair(capacity[t - 1]),
            "minimum_closure_slack": _fraction_pair(selected_check["minimum_closure_slack"]),
            "worst_intersection_mask": selected_check["worst_intersection_mask"],
            "intersections_checked_for_selected_set": selected_check["intersections_checked"],
            "candidate_sets_considered": candidate_count,
            "candidate_closure_checks": closure_checks,
            "prefix_sha256": _prefix_digest(e, g, level, v, c, capacity, t),
        })
        previous_mask = best_mask

    verification = verify_decision_sequence(
        decisions, e, g, level, v, c, capacity
    )
    if not verification["valid"]:
        raise ArithmeticError(f"independent sequence verification failed: {verification}")
    return {
        "schema": "symthaea-online-eclosure-capacity-v1",
        "status": "EXACT_SMALL_STREAM_REFERENCE_ORACLE_NOT_PRODUCTION",
        "guarantee_contract": (
            "SupFDR <= delta conditional on valid marginal e-values, the "
            "increasing e-collection and the online e-closure theorem assumptions"
        ),
        "delta": _fraction_pair(level),
        "gamma": [_fraction_pair(x) for x in g],
        "decisions": decisions,
        "verification": verification,
        "limitations": [
            "incoming e-value validity is assumed, not verified here",
            "finite stream and exponential exhaustive optimization",
            "capacity is a cumulative nondecreasing budget",
            "values and costs must be fixed or available before each e-value",
            "not a proof for asynchronous e-process evidence or per-period capacity",
        ],
    }


def verify_decision_sequence(
    decisions: Sequence[dict],
    evalues: Sequence[int | float | str | Fraction],
    gamma: Sequence[int | float | str | Fraction],
    delta: int | float | str | Fraction,
    values: Sequence[int | float | str | Fraction],
    costs: Sequence[int | float | str | Fraction],
    cumulative_capacity: Sequence[int | float | str | Fraction],
) -> dict:
    """Independent replay of nestedness, resource feasibility and all closures."""
    e, g, level, v, c, capacity = _normalise_inputs(
        evalues, gamma, delta, values, costs, cumulative_capacity
    )
    failures = []
    previous_mask = 0
    previous_cost = Fraction(0)
    checked = 0
    feasible_candidates_checked = 0
    certified_candidates_checked = 0
    optimality_checked = 0

    if len(decisions) != len(e):
        failures.append("decision count does not match stream length")

    for index, decision in enumerate(decisions):
        t = index + 1
        if t > len(e):
            failures.append(("extra_decision", t))
            continue
        if not isinstance(decision, dict):
            failures.append(("invalid_decision_record", t))
            continue
        mask = decision.get("rejection_mask")
        if isinstance(mask, bool) or not isinstance(mask, int) or not 0 <= mask < (1 << t):
            failures.append(("invalid_rejection_mask", t))
            continue
        if (mask & previous_mask) != previous_mask:
            failures.append(("non_nested_decision", t))
        actual_cost = _mask_cost(mask, c, t)
        if actual_cost > capacity[index]:
            failures.append(("capacity_exceeded", t))
        if actual_cost < previous_cost:
            # Costs are nonnegative and sets nested; this is a consistency guard.
            failures.append(("cumulative_cost_decreased", t))
        actual_value = _mask_value(mask, v, t)
        if decision.get("utility") != _fraction_pair(actual_value):
            failures.append(("utility_receipt_mismatch", t))
        if decision.get("cumulative_cost") != _fraction_pair(actual_cost):
            failures.append(("cost_receipt_mismatch", t))
        if decision.get("cumulative_capacity") != _fraction_pair(capacity[index]):
            failures.append(("capacity_receipt_mismatch", t))
        closure = verify_certified_set(mask, t, e, g, level)
        checked += closure["intersections_checked"]
        if not closure["certified"]:
            failures.append(("uncertified_rejection_set", t, closure["worst_intersection_mask"]))
        expected_indices = [i + 1 for i in range(t) if (mask >> i) & 1]
        if decision.get("rejections") != expected_indices:
            failures.append(("rejection_list_mismatch", t))
        expected_digest = _prefix_digest(e, g, level, v, c, capacity, t)
        if decision.get("prefix_sha256") != expected_digest:
            failures.append(("prefix_digest_mismatch", t))
        if decision.get("minimum_closure_slack") != _fraction_pair(closure["minimum_closure_slack"]):
            failures.append(("closure_slack_receipt_mismatch", t))
        if decision.get("worst_intersection_mask") != closure["worst_intersection_mask"]:
            failures.append(("worst_intersection_receipt_mismatch", t))
        if decision.get("intersections_checked_for_selected_set") != closure["intersections_checked"]:
            failures.append(("intersection_count_receipt_mismatch", t))

        # Independent exhaustive optimality check over every nested,
        # cumulative-capacity-feasible action in the certified menu.
        best_mask = None
        best_value = None
        best_cost = None
        for candidate_mask in range(1 << t):
            if (candidate_mask & previous_mask) != previous_mask:
                continue
            candidate_cost = sum(
                (c[j] for j in range(t) if (candidate_mask >> j) & 1),
                Fraction(0),
            )
            if candidate_cost > capacity[index]:
                continue
            feasible_candidates_checked += 1
            candidate_check = verify_certified_set(candidate_mask, t, e, g, level)
            if not candidate_check["certified"]:
                continue
            certified_candidates_checked += 1
            candidate_value = sum(
                (v[j] for j in range(t) if (candidate_mask >> j) & 1),
                Fraction(0),
            )
            better = best_mask is None or candidate_value > best_value
            if not better and candidate_value == best_value:
                better = candidate_cost < best_cost
                if not better and candidate_cost == best_cost:
                    chosen_tuple = tuple(
                        j for j in range(t) if (best_mask >> j) & 1
                    )
                    candidate_tuple = tuple(
                        j for j in range(t) if (candidate_mask >> j) & 1
                    )
                    better = candidate_tuple < chosen_tuple
            if better:
                best_mask = candidate_mask
                best_value = candidate_value
                best_cost = candidate_cost
        optimality_checked += 1
        if best_mask != mask:
            failures.append(("utility_optimality_failure", t, best_mask, mask))
        previous_mask = mask
        previous_cost = actual_cost

    return {
        "valid": not failures,
        "failures": failures,
        "decisions_checked": min(len(decisions), len(e)),
        "intersection_constraints_checked": checked,
        "feasible_candidates_checked": feasible_candidates_checked,
        "certified_candidates_checked": certified_candidates_checked,
        "optimality_prefixes_checked": optimality_checked,
        "optimality_verified": optimality_checked == len(e) and not any(
            isinstance(failure, tuple) and failure and failure[0] == "utility_optimality_failure"
            for failure in failures
        ),
        "scope": "finite-prefix e-closure membership, cumulative capacity and utility optimality",
    }


if __name__ == "__main__":
    raise SystemExit(
        "Run test_online_eclosure_capacity_oracle.py for exact reference fixtures."
    )
