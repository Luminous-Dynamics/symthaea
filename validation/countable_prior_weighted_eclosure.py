#!/usr/bin/env python3
"""Exact finite-family oracle for countable prior-weighted dynamic e-closure.

The statistical theorem is countable. This module is deliberately only a
finite-family algebra/decision oracle: it checks every nonempty subset of a
finite active family using exact rational arithmetic. It does not establish
that supplied component paths are e-processes under their intersection nulls.

Hypothesis identifiers are positive integers from one fixed, pre-data
enumeration of the countable universe. A prefix-code prior assigns positive
mass to every finite nonempty intersection and has total mass exactly one.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import isfinite
from typing import Callable, Iterable, Mapping, Sequence

Intersection = tuple[int, ...]
Prior = Callable[[Iterable[int]], Fraction]


def as_fraction(value: int | float | str | Fraction) -> Fraction:
    """Convert supported scalar inputs without silently accepting booleans."""
    if isinstance(value, bool):
        raise TypeError("boolean values are not valid numeric inputs")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        return Fraction(value)
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("numeric inputs must be finite")
        return Fraction(str(value))
    raise TypeError(f"unsupported numeric input: {type(value).__name__}")


def canonical_intersection(indices: Iterable[int]) -> Intersection:
    """Return a sorted, duplicate-free tuple of positive global indices."""
    try:
        values = tuple(indices)
    except TypeError as error:
        raise TypeError("an intersection must be an iterable of integer ids") from error
    if not values:
        raise ValueError("the empty intersection has no prior mass")
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in values):
        raise ValueError("hypothesis ids must be positive integers")
    if len(set(values)) != len(values):
        raise ValueError("an intersection cannot contain duplicate hypothesis ids")
    return tuple(sorted(values))


def elias_gamma_code(n: int) -> str:
    """Canonical Elias gamma code for a positive integer."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("Elias gamma coding requires a positive integer")
    binary = format(n, "b")
    return "0" * (len(binary) - 1) + binary


def elias_gamma_length(n: int) -> int:
    """Length of the Elias gamma code, computed without constructing it."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("Elias gamma coding requires a positive integer")
    return 2 * (n.bit_length() - 1) + 1


def intersection_code(indices: Iterable[int]) -> str:
    """Prefix-free code: gamma(size), then gamma-coded positive index gaps."""
    item = canonical_intersection(indices)
    gaps = (item[0],) + tuple(item[i] - item[i - 1] for i in range(1, len(item)))
    return elias_gamma_code(len(item)) + "".join(elias_gamma_code(gap) for gap in gaps)


def intersection_code_length(indices: Iterable[int]) -> int:
    item = canonical_intersection(indices)
    gaps = (item[0],) + tuple(item[i] - item[i - 1] for i in range(1, len(item)))
    return elias_gamma_length(len(item)) + sum(elias_gamma_length(gap) for gap in gaps)


def intersection_prior_weight(indices: Iterable[int]) -> Fraction:
    """A proper countable prior: pi_S = 2**(-L(S)); sum_S pi_S = 1.

    The code first gamma-encodes |S|, then the positive gaps in the sorted
    index sequence. This is prefix-free. Since Elias gamma codes have Kraft
    sum one, the concatenated family has Kraft sum
      sum_k 2^-L_gamma(k) * (sum_g 2^-L_gamma(g))**k = 1.
    """
    length = intersection_code_length(indices)
    return Fraction(1, 1 << length)


def nonempty_subsets(indices: Iterable[int]) -> tuple[Intersection, ...]:
    """Enumerate all nonempty subsets in stable size-then-lexical order."""
    values = tuple(sorted(indices))
    if len(set(values)) != len(values):
        raise ValueError("the active family contains duplicate ids")
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in values):
        raise ValueError("hypothesis ids must be positive integers")
    return tuple(
        subset
        for size in range(1, len(values) + 1)
        for subset in combinations(values, size)
    )


def finite_prior_mass(indices: Iterable[int], prior: Prior = intersection_prior_weight) -> Fraction:
    """Mass assigned to all nonempty subsets of a finite part of the universe."""
    return sum((prior(s) for s in nonempty_subsets(indices)), Fraction(0))


def verify_finite_prior_budget(
    indices: Iterable[int], prior: Prior = intersection_prior_weight
) -> dict:
    """Check positivity and total mass for a finite portion of the universe."""
    subsets = nonempty_subsets(indices)
    weights = []
    failures = []
    for subset in subsets:
        try:
            weight = as_fraction(prior(subset))
        except (TypeError, ValueError, ZeroDivisionError) as error:
            failures.append(("invalid_prior_weight", subset, str(error)))
            continue
        if weight <= 0:
            failures.append(("nonpositive_prior_weight", subset, weight))
        weights.append((subset, weight))
    total = sum((weight for _, weight in weights), Fraction(0))
    if total > 1:
        failures.append(("prior_mass_budget_exceeded", total))
    return {
        "valid": not failures and len(weights) == len(subsets),
        "failures": failures,
        "intersections_checked": len(weights),
        "finite_prefix_mass": total,
        "scope": "finite prefix check only; the countable prior normalization needs an analytic proof",
    }


def fdp_for_intersection(rejections: Iterable[int], null_intersection: Iterable[int]) -> Fraction:
    rejected = set(rejections)
    if not rejected:
        return Fraction(0)
    nulls = set(null_intersection)
    return Fraction(len(rejected & nulls), len(rejected))


def verify_local_closure(
    active: Iterable[int],
    rejections: Iterable[int],
    time_index: int,
    intersection_processes: Mapping[Intersection, Sequence[int | float | str | Fraction]],
    alpha: int | float | str | Fraction,
    *,
    prior: Prior = intersection_prior_weight,
) -> dict:
    """Verify every local FDP bound for one finite active family and time.

    For each nonempty S subset of active, require
       FDP_S(R) <= alpha * pi_S * E_t^S.
    Missing, malformed, negative, or truncated component paths fail closed.
    """
    active_ids = tuple(sorted(active))
    if len(set(active_ids)) != len(active_ids):
        raise ValueError("the active family contains duplicate ids")
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in active_ids):
        raise ValueError("hypothesis ids must be positive integers")
    rejected_ids = tuple(sorted(rejections))
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in rejected_ids):
        raise ValueError("rejection ids must be positive integers")
    if len(set(rejected_ids)) != len(rejected_ids):
        raise ValueError("the rejection set contains duplicate ids")
    if not set(rejected_ids).issubset(active_ids):
        raise ValueError("rejections must be a subset of the active family")
    if isinstance(time_index, bool) or not isinstance(time_index, int) or time_index < 0:
        raise ValueError("time_index must be a nonnegative integer")
    level = as_fraction(alpha)
    if not 0 < level < 1:
        raise ValueError("alpha must lie strictly between zero and one")

    failures = []
    rows = []
    active_intersections = nonempty_subsets(active_ids)
    prior_budget = verify_finite_prior_budget(active_ids, prior)
    if not prior_budget["valid"]:
        failures.append(("active_intersection_prior_budget_invalid", prior_budget["failures"]))
    for intersection in active_intersections:
        path = intersection_processes.get(intersection)
        if path is None:
            failures.append(("missing_intersection_process", intersection))
            continue
        if time_index >= len(path):
            failures.append(("intersection_path_too_short", intersection))
            continue
        try:
            e_value = as_fraction(path[time_index])
            weight = as_fraction(prior(intersection))
        except (TypeError, ValueError, ZeroDivisionError) as error:
            failures.append(("invalid_intersection_value_or_weight", intersection, str(error)))
            continue
        if e_value < 0:
            failures.append(("negative_intersection_e_value", intersection))
            continue
        if weight <= 0:
            failures.append(("nonpositive_intersection_prior_mass", intersection))
            continue
        local_fdp = fdp_for_intersection(rejected_ids, intersection)
        bound = level * weight * e_value
        slack = bound - local_fdp
        rows.append({
            "intersection": intersection,
            "prior_weight": weight,
            "e_value": e_value,
            "fdp": local_fdp,
            "bound": bound,
            "slack": slack,
            "certified": slack >= 0,
        })
        if slack < 0:
            failures.append(("local_closure_violation", intersection, local_fdp, bound))

    return {
        "certified": not failures and len(rows) == (1 << len(active_ids)) - 1,
        "failures": failures,
        "constraints_checked": len(rows),
        "expected_constraints": (1 << len(active_ids)) - 1,
        "active_intersection_prior_mass": prior_budget["finite_prefix_mass"],
        "minimum_slack": min((row["slack"] for row in rows), default=None),
        "constraints": rows,
        "scope": "exact pathwise finite-family closure check; component e-process validity is external",
    }


def enumerate_certified_rejection_sets(
    active: Iterable[int],
    time_index: int,
    intersection_processes: Mapping[Intersection, Sequence[int | float | str | Fraction]],
    alpha: int | float | str | Fraction,
    *,
    prior: Prior = intersection_prior_weight,
) -> tuple[Intersection, ...]:
    """Exhaustive finite-family oracle for all locally certified action sets."""
    active_ids = tuple(sorted(active))
    certified = []
    for candidate in ((),) + nonempty_subsets(active_ids):
        check = verify_local_closure(
            active_ids, candidate, time_index, intersection_processes, alpha, prior=prior
        )
        if check["certified"]:
            certified.append(candidate)
    return tuple(certified)


def countable_mixture_value(
    true_null_universe: Iterable[int],
    time_index: int,
    intersection_processes: Mapping[Intersection, Sequence[int | float | str | Fraction]],
    *,
    prior: Prior = intersection_prior_weight,
) -> Fraction:
    """Compute the exact mixture for a finite truth fixture (not a countable proof)."""
    true_ids = tuple(sorted(true_null_universe))
    total = Fraction(0)
    for intersection in nonempty_subsets(true_ids):
        path = intersection_processes.get(intersection)
        if path is None or time_index >= len(path):
            raise ValueError(f"missing or truncated component process for {intersection}")
        e_value = as_fraction(path[time_index])
        if e_value < 0:
            raise ValueError(f"negative component e-value for {intersection}")
        total += as_fraction(prior(intersection)) * e_value
    return total


def verify_finite_truth_path(
    active_by_time: Sequence[Iterable[int]],
    rejections_by_time: Sequence[Iterable[int]],
    true_null_universe: Iterable[int],
    intersection_processes: Mapping[Intersection, Sequence[int | float | str | Fraction]],
    alpha: int | float | str | Fraction,
    *,
    prior: Prior = intersection_prior_weight,
) -> dict:
    """Check local closure and the pathwise stopped/SupFDR mixture domination.

    Truth is supplied only for this finite deterministic fixture. A production
    procedure never observes it. The countable theorem uses the full global
    true-null universe; finite fixtures check exact truncations and algebra.
    """
    if len(active_by_time) != len(rejections_by_time):
        raise ValueError("active and rejection paths must have equal length")
    true_ids = set(true_null_universe)
    alpha_value = as_fraction(alpha)
    rows = []
    failures = []
    for t, (active, rejections) in enumerate(zip(active_by_time, rejections_by_time)):
        active_ids = tuple(sorted(active))
        check = verify_local_closure(
            active_ids, rejections, t, intersection_processes, alpha_value, prior=prior
        )
        if not check["certified"]:
            failures.append(("local_closure_failed", t, check["failures"]))
        active_true = tuple(i for i in active_ids if i in true_ids)
        loss = fdp_for_intersection(rejections, active_true)
        mixture = countable_mixture_value(true_ids, t, intersection_processes, prior=prior)
        bound = alpha_value * mixture
        if loss > bound:
            failures.append(("mixture_domination_failed", t, loss, bound))
        rows.append({
            "time": t,
            "active": active_ids,
            "rejections": tuple(sorted(rejections)),
            "active_true_nulls": active_true,
            "fdp": loss,
            "mixture": mixture,
            "alpha_mixture_bound": bound,
            "dominated": loss <= bound,
        })
    return {
        "valid": not failures,
        "failures": failures,
        "times_checked": len(rows),
        "sup_fdp": max((row["fdp"] for row in rows), default=Fraction(0)),
        "sup_mixture_bound": max((row["alpha_mixture_bound"] for row in rows), default=Fraction(0)),
        "path": rows,
        "scope": "finite exact fixture; stochastic e-process contracts and infinite sums are not inferred from paths",
    }


if __name__ == "__main__":
    raise SystemExit("Run test_countable_prior_weighted_eclosure.py for exact fixtures.")
