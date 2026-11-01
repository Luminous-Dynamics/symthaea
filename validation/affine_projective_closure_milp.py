#!/usr/bin/env python3
"""Exact MILP formulation for global-affine projective e-closure.

For a candidate rejection vector x in {0,1}^n, r=sum_i x_i and
d_i=w_i(M_i-1), all local intersection constraints are equivalent to

    y_i >= x_i - alpha*r*d_i
    y_i >= 0
    sum_i y_i <= alpha*r.

The model builder emits exact Fraction coefficients. It does not call a
solver; independent exact verification is provided for returned solutions.
"""
from __future__ import annotations

from fractions import Fraction
from math import isfinite
from typing import Iterable, Mapping, Sequence

from countable_global_projective_eclosure import (
    as_fraction,
    coordinate_weight,
    validate_ids,
)


def build_affine_closure_milp(
    active: Iterable[int],
    coordinate_values: Mapping[int, int | float | str | Fraction],
    alpha: int | float | str | Fraction,
    *,
    scheme: str = "geometric",
    values: Mapping[int, int | float | str | Fraction] | None = None,
    costs: Mapping[int, int | float | str | Fraction] | None = None,
    capacity: int | float | str | Fraction | None = None,
    previous_rejections: Iterable[int] = (),
) -> dict:
    """Build the O(n)-size exact MILP for local affine closure and capacity."""
    ids = validate_ids(active)
    level = as_fraction(alpha)
    if not 0 < level < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    previous = validate_ids(previous_rejections)
    if not set(previous).issubset(ids):
        raise ValueError("previous rejections must remain active in the current family")

    d = {}
    for i in ids:
        if i not in coordinate_values:
            raise ValueError(f"missing coordinate value for hypothesis {i}")
        m = as_fraction(coordinate_values[i])
        if m < 0:
            raise ValueError("coordinate e-values must be nonnegative")
        d[i] = coordinate_weight(i, scheme) * (m - 1)

    utility = {i: Fraction(0) for i in ids}
    if values is not None:
        for i in ids:
            if i not in values:
                raise ValueError(f"missing utility value for hypothesis {i}")
            utility[i] = as_fraction(values[i])

    cost = {i: Fraction(0) for i in ids}
    if costs is not None:
        for i in ids:
            if i not in costs:
                raise ValueError(f"missing cost for hypothesis {i}")
            cost[i] = as_fraction(costs[i])
            if cost[i] < 0:
                raise ValueError("costs must be nonnegative")

    cap = None if capacity is None else as_fraction(capacity)
    if cap is not None and cap < 0:
        raise ValueError("capacity must be nonnegative")
    if cap is None and costs is not None:
        raise ValueError("capacity is required when costs are supplied")

    return {
        "schema": "affine-projective-closure-milp-v1",
        "sense": "maximize",
        "objective": {f"x_{i}": utility[i] for i in ids},
        "variables": {
            **{f"x_{i}": {"type": "binary"} for i in ids},
            "r": {"type": "integer", "lower": 0, "upper": len(ids)},
            **{f"y_{i}": {"type": "continuous", "lower": 0} for i in ids},
        },
        "constraints": {
            "rejection_count": {
                "sense": "equal",
                "lhs": {"r": Fraction(1), **{f"x_{i}": Fraction(-1) for i in ids}},
                "rhs": Fraction(0),
            },
            "closure_auxiliary": [
                {
                    "name": f"closure_aux_{i}",
                    "sense": "greater_equal",
                    "lhs": {
                        f"y_{i}": Fraction(1),
                        f"x_{i}": Fraction(-1),
                        "r": level * d[i],
                    },
                    "rhs": Fraction(0),
                }
                for i in ids
            ],
            "closure_sum": {
                "sense": "less_equal",
                "lhs": {
                    **{f"y_{i}": Fraction(1) for i in ids},
                    "r": -level,
                },
                "rhs": Fraction(0),
            },
            "capacity": None if cap is None else {
                "sense": "less_equal",
                "lhs": {f"x_{i}": cost[i] for i in ids},
                "rhs": cap,
            },
            "persistence": [
                {"sense": "equal", "lhs": {f"x_{i}": Fraction(1)}, "rhs": Fraction(1)}
                for i in previous
            ],
        },
        "metadata": {
            "active_ids": ids,
            "alpha": level,
            "weight_scheme": scheme,
            "coordinate_evidence": {i: as_fraction(coordinate_values[i]) for i in ids},
            "coordinate_offsets": d,
            "global_weights": {i: coordinate_weight(i, scheme) for i in ids},
            "costs": cost,
            "capacity_value": cap,
            "previous_rejections": previous,
            "closure_constraints_covered": (1 << len(ids)) - 1,
            "model_variables": 2 * len(ids) + 1,
            "model_constraint_rows": len(ids) + 2 + len(previous) + (1 if cap is not None else 0),
            "complexity": "O(n) variables and O(n) constraints; 0-1 optimization remains combinatorial",
        },
    }


def verify_affine_closure_milp_solution(
    model: dict,
    rejection_values: Mapping[int, int],
    auxiliary_values: Mapping[int, int | float | str | Fraction],
) -> dict:
    """Check a solver result exactly, including all original intersection tests."""
    meta = model["metadata"]
    ids = meta["active_ids"]
    level = meta["alpha"]
    d = meta["coordinate_offsets"]
    previous = set(meta["previous_rejections"])
    failures = []
    x = {}
    y = {}
    for i in ids:
        bit = rejection_values.get(i)
        if isinstance(bit, bool) or not isinstance(bit, int) or bit not in (0, 1):
            failures.append(("invalid_binary_rejection", i, bit))
            continue
        x[i] = int(bit)
        try:
            y[i] = as_fraction(auxiliary_values[i])
        except (KeyError, TypeError, ValueError) as error:
            failures.append(("invalid_auxiliary_value", i, str(error)))
            continue
        if y[i] < 0:
            failures.append(("negative_auxiliary_value", i))
    if failures:
        return {"valid": False, "failures": failures}

    r = sum(x.values())
    if any(x[i] != 1 for i in previous):
        failures.append(("persistence_violation", tuple(sorted(previous))))

    for i in ids:
        required = Fraction(x[i]) - level * r * d[i]
        if y[i] < required:
            failures.append(("closure_auxiliary_understated", i, y[i], required))
    if sum(y.values(), Fraction(0)) > level * r:
        failures.append(("closure_sum_exceeded", sum(y.values(), Fraction(0)), level * r))

    costs = meta["costs"]
    capacity = meta["capacity_value"]
    total_cost = sum((costs[i] * x[i] for i in ids), Fraction(0))
    if capacity is not None and total_cost > capacity:
        failures.append(("capacity_exceeded", total_cost, capacity))

    # Differential check against the original 2^n local intersection constraints.
    rejected = tuple(i for i in ids if x[i] == 1)
    count = len(rejected)
    exact_constraints = 0
    for mask in range(1, 1 << len(ids)):
        subset = tuple(ids[j] for j in range(len(ids)) if (mask >> j) & 1)
        loss = Fraction(sum(x[i] for i in subset), count) if count else Fraction(0)
        certificate = Fraction(1) + sum(
            (meta["global_weights"][i] * (meta["coordinate_evidence"][i] - 1) for i in subset),
            Fraction(0),
        )
        exact_constraints += 1
        if loss > level * certificate:
            failures.append(("original_intersection_constraint_violation", subset, loss, level * certificate))

    return {
        "valid": not failures,
        "failures": failures,
        "rejections": rejected,
        "rejection_count": r,
        "auxiliary_sum": sum(y.values(), Fraction(0)),
        "closure_budget": level * r,
        "cost": total_cost,
        "original_intersections_checked": exact_constraints,
        "model_claimed_intersections": meta["closure_constraints_covered"],
        "scope": "exact finite-solution replay; stochastic validity of coordinate e-processes remains external",
    }


def minimal_auxiliary_values(
    model: dict,
    rejection_values: Mapping[int, int],
) -> dict[int, Fraction]:
    """Construct the componentwise-minimal y_i for a proposed binary action."""
    meta = model["metadata"]
    ids = meta["active_ids"]
    level = meta["alpha"]
    d = meta["coordinate_offsets"]
    x = {}
    for i in ids:
        bit = rejection_values.get(i)
        if isinstance(bit, bool) or not isinstance(bit, int) or bit not in (0, 1):
            raise ValueError(f"rejection indicator for {i} must be binary")
        x[i] = int(bit)
    r = sum(x.values())
    return {
        i: max(Fraction(0), Fraction(x[i]) - level * r * d[i])
        for i in ids
    }


if __name__ == "__main__":
    raise SystemExit("Import this module from the exact MILP and differential tests.")
