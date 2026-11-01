"""Finite-family exact checker for pathwise supremum-anchored e-closure.

This module verifies deterministic pathwise inequalities only. It cannot prove
the stochastic coordinate budgets q_i or auxiliary running-supremum validity.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from typing import Iterable, Mapping, Sequence


def _q(value: Fraction | int | str) -> Fraction:
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError("use exact Fraction, int or rational string")
    return value if isinstance(value, Fraction) else Fraction(value)


def _ids(universe: Iterable[str]) -> tuple[str, ...]:
    raw = tuple(universe)
    if any(not isinstance(x, str) or not x for x in raw):
        raise ValueError("hypothesis IDs must be non-empty strings")
    if len(set(raw)) != len(raw):
        raise ValueError("hypothesis IDs must be unique")
    return tuple(sorted(raw))


def nonempty_subsets(universe: Iterable[str]) -> tuple[frozenset[str], ...]:
    ids = _ids(universe)
    return tuple(
        frozenset(group)
        for size in range(1, len(ids) + 1)
        for group in combinations(ids, size)
    )


def fdp_for_intersection(
    rejections: Iterable[str], intersection: Iterable[str]
) -> Fraction:
    rejected = frozenset(rejections)
    if not rejected:
        return Fraction(0)
    nulls = frozenset(intersection)
    return Fraction(len(rejected & nulls), len(rejected))


def _normalize_path(
    path: Sequence[Iterable[str]], universe: frozenset[str], name: str
) -> tuple[frozenset[str], ...]:
    normalized = tuple(frozenset(row) for row in path)
    for t, row in enumerate(normalized):
        outside = row - universe
        if outside:
            raise ValueError(
                f"{name}[{t}] contains unknown hypotheses: {sorted(outside)}"
            )
    return normalized


def coordinate_ever_rejected(
    baseline_path: Sequence[frozenset[str]], universe: Iterable[str]
) -> dict[str, tuple[Fraction, ...]]:
    """Coordinate-local ever-rejected indicators, stable under universe extension."""
    ids = _ids(universe)
    seen = {i: False for i in ids}
    result: dict[str, list[Fraction]] = {i: [] for i in ids}
    for rejected in baseline_path:
        for i in ids:
            seen[i] = seen[i] or i in rejected
            result[i].append(Fraction(1) if seen[i] else Fraction(0))
    return {i: tuple(values) for i, values in result.items()}


def build_supremum_anchored_certificates(
    *,
    universe: Iterable[str],
    baseline_path: Sequence[Iterable[str]],
    coordinate_budgets: Mapping[str, Fraction | int | str],
    alpha: Fraction | int | str,
    adjusted_auxiliary: Mapping[
        frozenset[str], Sequence[Fraction | int | str]
    ],
) -> dict[frozenset[str], tuple[Fraction, ...]]:
    """Build E_{S,t}=A_{S,t}+(1-c_S)F_{S,t} with exact rational arithmetic.

    External statistical contract:
      P_Hi(i is ever rejected by the coordinate-local baseline) <= alpha*q_i,
      sum_i q_i <= 1; and E[sup_t F_S,t | H_S] <= 1.
    """
    ids = _ids(universe)
    if not ids:
        raise ValueError("universe must be nonempty")
    universe_set = frozenset(ids)
    a = _q(alpha)
    if not 0 < a <= 1:
        raise ValueError("alpha must be in (0,1]")
    if set(coordinate_budgets) != set(ids):
        raise ValueError("coordinate_budgets must contain exactly the universe IDs")
    q = {i: _q(value) for i, value in coordinate_budgets.items()}
    if any(value < 0 for value in q.values()):
        raise ValueError("coordinate budgets must be nonnegative")
    if sum(q.values(), Fraction(0)) > 1:
        raise ValueError("coordinate budgets must sum to at most one")

    baseline = _normalize_path(baseline_path, universe_set, "baseline_path")
    if not baseline:
        raise ValueError("baseline_path must contain at least one round")
    horizon = len(baseline)
    subsets = nonempty_subsets(ids)
    if set(adjusted_auxiliary) != set(subsets):
        raise ValueError("adjusted_auxiliary must contain every nonempty intersection")

    ever = coordinate_ever_rejected(baseline, ids)
    output: dict[frozenset[str], tuple[Fraction, ...]] = {}
    for intersection in subsets:
        aux = tuple(_q(x) for x in adjusted_auxiliary[intersection])
        if len(aux) != horizon:
            raise ValueError(f"auxiliary path length mismatch for {sorted(intersection)}")
        if any(x < 0 for x in aux):
            raise ValueError("auxiliary path values must be nonnegative")
        c_s = sum((q[i] for i in intersection), Fraction(0))
        if c_s > 1:
            raise ValueError("intersection coordinate budget exceeds one")
        residual = 1 - c_s
        output[intersection] = tuple(
            sum((ever[i][t] / a for i in intersection), Fraction(0))
            + residual * aux[t]
            for t in range(horizon)
        )
    return output


def verify_closure_path(
    *,
    universe: Iterable[str],
    rejection_path: Sequence[Iterable[str]],
    local_e_paths: Mapping[frozenset[str], Sequence[Fraction]],
    alpha: Fraction | int | str,
) -> tuple[bool, list[dict[str, object]]]:
    """Check every closure inequality for every intersection and time."""
    ids = _ids(universe)
    universe_set = frozenset(ids)
    path = _normalize_path(rejection_path, universe_set, "rejection_path")
    a = _q(alpha)
    subsets = nonempty_subsets(ids)
    if set(local_e_paths) != set(subsets):
        raise ValueError("local_e_paths must contain every nonempty intersection")
    if any(len(local_e_paths[s]) != len(path) for s in subsets):
        raise ValueError("local e-path horizon mismatch")
    valid = True
    records: list[dict[str, object]] = []
    for t, rejected in enumerate(path):
        for intersection in subsets:
            e_value = _q(local_e_paths[intersection][t])
            if e_value < 0:
                raise ValueError("local e-values must be nonnegative")
            fdp = fdp_for_intersection(rejected, intersection)
            bound = a * e_value
            ok = fdp <= bound
            valid = valid and ok
            records.append({
                "time": t,
                "intersection": tuple(sorted(intersection)),
                "fdp": fdp,
                "bound": bound,
                "certified": ok,
            })
    return valid, records


def verify_baseline_retention(
    *,
    universe: Iterable[str],
    baseline_path: Sequence[Iterable[str]],
    local_e_paths: Mapping[frozenset[str], Sequence[Fraction]],
    alpha: Fraction | int | str,
) -> bool:
    valid, _ = verify_closure_path(
        universe=universe,
        rejection_path=baseline_path,
        local_e_paths=local_e_paths,
        alpha=alpha,
    )
    return valid


def select_largest_certified_superset(
    *,
    universe: Iterable[str],
    anchor: Iterable[str],
    local_e_values: Mapping[frozenset[str], Fraction | int | str],
    alpha: Fraction | int | str,
) -> frozenset[str]:
    """Exponential-time exact toy selector; lexicographic deterministic tie-break."""
    ids = _ids(universe)
    universe_set = frozenset(ids)
    base = frozenset(anchor)
    if not base <= universe_set:
        raise ValueError("anchor contains unknown hypotheses")
    a = _q(alpha)
    subsets = nonempty_subsets(ids)
    if set(local_e_values) != set(subsets):
        raise ValueError("local_e_values must contain every nonempty intersection")
    certified: list[frozenset[str]] = []
    for size in range(len(base), len(ids) + 1):
        for candidate_tuple in combinations(ids, size):
            candidate = frozenset(candidate_tuple)
            if not base <= candidate:
                continue
            if all(
                fdp_for_intersection(candidate, intersection)
                <= a * _q(local_e_values[intersection])
                for intersection in subsets
            ):
                certified.append(candidate)
    if not certified:
        raise ValueError("baseline anchor is not itself certified")
    certified.sort(key=lambda row: tuple(sorted(row)))
    largest_size = max(map(len, certified))
    return next(row for row in certified if len(row) == largest_size)


def pathwise_supremum_bound(
    *,
    truth: Iterable[str],
    rejection_path: Sequence[Iterable[str]],
    local_e_paths: Mapping[frozenset[str], Sequence[Fraction]],
    alpha: Fraction | int | str,
) -> tuple[Fraction, Fraction]:
    """Return observed sup-FDP and its finite-path certificate bound."""
    true_set = frozenset(truth)
    if not true_set:
        return Fraction(0), Fraction(0)
    if true_set not in local_e_paths:
        raise ValueError("truth intersection must have a local e-value path")
    a = _q(alpha)
    path = tuple(frozenset(x) for x in rejection_path)
    if len(path) != len(local_e_paths[true_set]):
        raise ValueError("path horizon mismatch")
    observed = max(
        (fdp_for_intersection(rejected, true_set) for rejected in path),
        default=Fraction(0),
    )
    certificate = a * max(
        (_q(x) for x in local_e_paths[true_set]),
        default=Fraction(0),
    )
    if observed > certificate:
        raise AssertionError("pathwise SupFDR certificate inequality failed")
    return observed, certificate
