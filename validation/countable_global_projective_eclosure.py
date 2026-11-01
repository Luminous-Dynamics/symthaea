#!/usr/bin/env python3
"""Finite reference oracle for countable global-weight projective e-closure.

The theorem uses one fixed summable weight per coordinate, not a separate
prior over every intersection. This module checks finite paths exactly using
Fraction arithmetic. It does not establish stochastic e-process validity,
predictability of real-world activation, or completeness of the filtration.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import isfinite
from typing import Iterable, Mapping, Sequence

CoordinatePaths = Mapping[int, Sequence[int | float | str | Fraction]]


def as_fraction(value: int | float | str | Fraction) -> Fraction:
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


def validate_ids(ids: Iterable[int]) -> tuple[int, ...]:
    try:
        values = tuple(ids)
    except TypeError as error:
        raise TypeError("hypothesis ids must be an iterable of positive integers") from error
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 1 for i in values):
        raise ValueError("hypothesis ids must be positive integers")
    if len(set(values)) != len(values):
        raise ValueError("hypothesis ids must be unique")
    return tuple(sorted(values))


def coordinate_weight(index: int, scheme: str = "geometric") -> Fraction:
    """Exactly summable ex-ante global coordinate weights."""
    if isinstance(index, bool) or not isinstance(index, int) or index < 1:
        raise ValueError("coordinate index must be a positive integer")
    if scheme == "geometric":
        return Fraction(1, 1 << index)
    if scheme == "telescoping":
        return Fraction(1, index * (index + 1))
    raise ValueError("weight scheme must be 'geometric' or 'telescoping'")


def finite_weight_mass(ids: Iterable[int], scheme: str = "geometric") -> Fraction:
    return sum((coordinate_weight(i, scheme) for i in validate_ids(ids)), Fraction(0))


def nonempty_subsets(ids: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    values = validate_ids(ids)
    return tuple(
        subset
        for size in range(1, len(values) + 1)
        for subset in combinations(values, size)
    )


def projective_certificate(
    intersection: Iterable[int],
    coordinate_values: Mapping[int, int | float | str | Fraction],
    *,
    scheme: str = "geometric",
) -> Fraction:
    """E_t^S = 1 - sum_{i in S}w_i + sum_{i in S}w_i M_{i,t}."""
    total = Fraction(1)
    for i in validate_ids(intersection):
        if i not in coordinate_values:
            raise ValueError(f"missing coordinate value for hypothesis {i}")
        value = as_fraction(coordinate_values[i])
        if value < 0:
            raise ValueError(f"negative coordinate e-value for hypothesis {i}")
        total += coordinate_weight(i, scheme) * (value - 1)
    if total < 0:
        raise AssertionError("nonnegative affine merger became negative")
    return total


def fdp(rejections: Iterable[int], null_intersection: Iterable[int]) -> Fraction:
    rejected = set(validate_ids(rejections))
    nulls = set(validate_ids(null_intersection))
    return Fraction(len(rejected & nulls), len(rejected)) if rejected else Fraction(0)


def separate_local_closure(
    active: Iterable[int],
    rejections: Iterable[int],
    coordinate_values: Mapping[int, int | float | str | Fraction],
    alpha: int | float | str | Fraction,
    *,
    scheme: str = "geometric",
) -> dict:
    """Check all nonempty local constraints with an exact O(|I|) separator.

    For a nonempty rejection set R, each intersection S must satisfy
      sum_{i in S} [1{i in R}/|R| - alpha*w_i*(M_i-1)] <= alpha.
    The maximizing subset contains exactly the coordinates with positive
    bracket values, so all 2^|I|-1 constraints reduce to one linear scan.
    """
    ids = validate_ids(active)
    rejected = validate_ids(rejections)
    if not set(rejected).issubset(ids):
        raise ValueError("rejections must be a subset of the active family")
    level = as_fraction(alpha)
    if not 0 < level < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if not ids:
        return {
            "certified": True, "active_count": 0,
            "subsets_implicitly_checked": 0, "coordinate_terms_evaluated": 0,
            "separating_subset": (), "maximum_nonempty_lhs": None,
            "positive_part_sum": Fraction(0), "slack": None,
            "scope": "empty active family has no nonempty local constraints",
        }

    r = len(rejected)
    rejected_set = set(rejected)
    coefficients = {}
    for i in ids:
        if i not in coordinate_values:
            raise ValueError(f"missing coordinate value for hypothesis {i}")
        m = as_fraction(coordinate_values[i])
        if m < 0:
            raise ValueError(f"negative coordinate e-value for hypothesis {i}")
        d_i = coordinate_weight(i, scheme) * (m - 1)
        coefficients[i] = (Fraction(1, r) if i in rejected_set else Fraction(0)) - level * d_i

    positive_ids = tuple(i for i in ids if coefficients[i] > 0)
    positive_part_sum = sum((coefficients[i] for i in positive_ids), Fraction(0))
    if positive_ids:
        witness = positive_ids
        maximum_nonempty_lhs = positive_part_sum
    else:
        witness = (max(ids, key=lambda i: coefficients[i]),)
        maximum_nonempty_lhs = coefficients[witness[0]]
    return {
        "certified": positive_part_sum <= level,
        "active_count": len(ids),
        "subsets_implicitly_checked": (1 << len(ids)) - 1,
        "coordinate_terms_evaluated": len(ids),
        "separating_subset": witness,
        "maximum_nonempty_lhs": maximum_nonempty_lhs,
        "positive_part_sum": positive_part_sum,
        "slack": level - maximum_nonempty_lhs,
        "coefficients": coefficients,
        "scope": "exact O(|I|) separator equivalent to every local affine-closure constraint",
    }


def _normalize_path_inputs(
    active_by_time: Sequence[Iterable[int]],
    rejections_by_time: Sequence[Iterable[int]],
    coordinate_paths: CoordinatePaths,
    true_null_ids: Iterable[int],
    *,
    require_nested: bool,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], dict[int, tuple[Fraction, ...]], tuple[int, ...]]:
    if not active_by_time:
        raise ValueError("at least one reporting time is required")
    if len(active_by_time) != len(rejections_by_time):
        raise ValueError("active and rejection paths must have equal lengths")
    active = [validate_ids(row) for row in active_by_time]
    rejected = [validate_ids(row) for row in rejections_by_time]
    for t, (i_t, r_t) in enumerate(zip(active, rejected)):
        if not set(r_t).issubset(i_t):
            raise ValueError(f"rejections at time {t} must be a subset of the active family")
    if require_nested:
        for t in range(1, len(active)):
            if not set(active[t - 1]).issubset(active[t]):
                raise ValueError("active families must be nested for the arrival model")

    truth = validate_ids(true_null_ids)
    all_ids = set(truth)
    for row in active:
        all_ids.update(row)
    normalized = {}
    for i in sorted(all_ids):
        if i not in coordinate_paths:
            raise ValueError(f"missing coordinate path for hypothesis {i}")
        path = coordinate_paths[i]
        if len(path) != len(active):
            raise ValueError(f"coordinate path for hypothesis {i} has wrong horizon")
        values = tuple(as_fraction(v) for v in path)
        if any(v < 0 for v in values):
            raise ValueError(f"coordinate path for hypothesis {i} contains a negative value")
        first_active = next((t for t, row in enumerate(active) if i in row), None)
        if first_active is None:
            if any(v != 1 for v in values):
                raise ValueError(f"never-active coordinate {i} must remain neutral at one")
        elif any(values[t] != 1 for t in range(first_active)):
            raise ValueError(f"coordinate {i} must be dormant at one before activation")
        normalized[i] = values
    return active, rejected, normalized, truth


def verify_countable_global_projective_path(
    active_by_time: Sequence[Iterable[int]],
    rejections_by_time: Sequence[Iterable[int]],
    true_null_ids: Iterable[int],
    coordinate_paths: CoordinatePaths,
    alpha: int | float | str | Fraction,
    *,
    scheme: str = "geometric",
    require_nested: bool = True,
) -> dict:
    """Verify finite fixtures for local closure and full-true-set projectivity.

    The true-null set is finite in this oracle. The mathematical construction
    extends to countable true-null sets by nonnegative countable mixing.
    """
    level = as_fraction(alpha)
    if not 0 < level < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    active, rejected, paths, truth = _normalize_path_inputs(
        active_by_time, rejections_by_time, coordinate_paths, true_null_ids,
        require_nested=require_nested,
    )
    rows, failures = [], []
    monotone = all(
        all(path[t] <= path[t + 1] for t in range(len(path) - 1))
        for path in paths.values()
    )

    for t, (i_t, r_t) in enumerate(zip(active, rejected)):
        values_t = {i: path[t] for i, path in paths.items()}
        separation = separate_local_closure(
            i_t, r_t, values_t, level, scheme=scheme
        )
        if not separation["certified"]:
            witness = separation["separating_subset"]
            witness_cert = projective_certificate(witness, values_t, scheme=scheme)
            witness_loss = fdp(r_t, witness)
            failures.append(("local_closure_violation", t, witness, witness_loss, level * witness_cert))

        active_true = tuple(i for i in truth if i in set(i_t))
        active_true_cert = projective_certificate(active_true, values_t, scheme=scheme)
        full_true_cert = projective_certificate(truth, values_t, scheme=scheme)
        if active_true_cert != full_true_cert:
            failures.append(("projective_dormant_padding_identity_failed", t))
        actual_loss = fdp(r_t, active_true)
        domination_slack = level * full_true_cert - actual_loss
        if domination_slack < 0:
            failures.append(("true_null_domination_failed", t, actual_loss, level * full_true_cert))

        rows.append({
            "time": t, "active": i_t, "rejections": r_t,
            "active_true_nulls": active_true,
            "local_constraints_covered": separation["subsets_implicitly_checked"],
            "local_constraint_terms_evaluated": separation["coordinate_terms_evaluated"],
            "local_constraints_certified": separation["certified"],
            "local_closure_separator": separation,
            "full_true_certificate": full_true_cert,
            "active_true_certificate": active_true_cert,
            "projective_identity": full_true_cert == active_true_cert,
            "true_null_fdp": actual_loss,
            "alpha_full_true_certificate": level * full_true_cert,
            "true_null_domination_slack": domination_slack,
            "dominated": domination_slack >= 0,
        })

    return {
        "valid": not failures,
        "failures": failures,
        "schema": "countable-global-projective-eclosure-finite-oracle-v1",
        "weight_scheme": scheme,
        "finite_true_null_ids": truth,
        "finite_true_null_weight_mass": finite_weight_mass(truth, scheme),
        "active_sizes": [len(row) for row in active],
        "times_checked": len(rows),
        "local_constraints_covered": sum(row["local_constraints_covered"] for row in rows),
        "local_constraint_terms_evaluated": sum(row["local_constraint_terms_evaluated"] for row in rows),
        "all_projective_identities_verified": all(row["projective_identity"] for row in rows),
        "all_true_null_losses_dominated": all(row["dominated"] for row in rows),
        "coordinate_paths_monotone": monotone,
        "supfdr_pathwise_condition_satisfied": monotone,
        "path": rows,
        "scope": (
            "finite exact pathwise oracle; global countable-mixing theorem and "
            "coordinate e-process validity are mathematical assumptions, not inferred from paths"
        ),
    }


def verify_weight_scheme_prefix(scheme: str = "geometric", n: int = 100) -> dict:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    mass = finite_weight_mass(range(1, n + 1), scheme)
    if scheme == "geometric":
        expected = Fraction(1) - Fraction(1, 1 << n)
        proof = "geometric series sum_i 2^-i = 1"
    elif scheme == "telescoping":
        expected = Fraction(1) - Fraction(1, n + 1)
        proof = "telescoping series sum_i (1/i - 1/(i+1)) = 1"
    else:
        raise ValueError("weight scheme must be 'geometric' or 'telescoping'")
    return {
        "scheme": scheme, "prefix_size": n, "prefix_mass": mass,
        "expected_prefix_mass": expected, "matches": mass == expected,
        "infinite_mass": Fraction(1), "normalization_proof": proof,
    }


if __name__ == "__main__":
    raise SystemExit("Run test_countable_global_projective_eclosure.py.")
