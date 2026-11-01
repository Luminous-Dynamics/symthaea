#!/usr/bin/env python3
"""Benchmark the compact affine-projective closure MILP with SciPy/HiGHS.

SciPy is an optional research dependency. Every returned binary decision is
rechecked by the exact O(n) Fraction verifier; for small instances an exact
exhaustive action oracle can additionally compare the objective optimum.

This is an engineering benchmark, not a statistical-validity simulation.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import platform
import random
import time
from typing import Any, Iterable

from affine_projective_closure_milp import (
    build_affine_closure_milp,
    verify_affine_binary_decision,
)


def make_instance(
    n: int,
    seed: int,
    *,
    alpha: Fraction = Fraction(1, 10),
    scheme: str = "geometric",
    capacity_fraction: Fraction = Fraction(95, 100),
) -> dict[str, Any]:
    """Build a reproducible, nontrivial knapsack/closure instance.

    Evidence is constructed so d_i=w_i(M_i-1)=12/n for every coordinate.
    For alpha=0.1 this makes the closure condition require at least 75% of
    the active family to be selected, while capacity and heterogeneous
    utility/cost determine which candidates are selected.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if not 0 < capacity_fraction <= 1:
        raise ValueError("capacity_fraction must lie in (0, 1]")
    if scheme not in ("geometric", "telescoping"):
        raise ValueError("unknown global weight scheme")

    rng = random.Random(seed)
    ids = tuple(range(1, n + 1))
    d = Fraction(12, n)
    from countable_global_projective_eclosure import coordinate_weight

    weights = {i: coordinate_weight(i, scheme) for i in ids}
    evidence = {i: Fraction(1) + d / weights[i] for i in ids}
    utility = {i: Fraction(rng.randint(10, 100)) for i in ids}
    costs = {i: Fraction(rng.randint(5, 20)) for i in ids}
    total_cost = sum(costs.values(), Fraction(0))
    capacity = capacity_fraction * total_cost

    payload = {
        "schema": "affine-projective-milp-benchmark-instance-v1",
        "n": n,
        "seed": seed,
        "alpha": [alpha.numerator, alpha.denominator],
        "scheme": scheme,
        "capacity_fraction": [capacity_fraction.numerator, capacity_fraction.denominator],
        "d_i": [d.numerator, d.denominator],
        "utility": [[i, utility[i].numerator, utility[i].denominator] for i in ids],
        "costs": [[i, costs[i].numerator, costs[i].denominator] for i in ids],
    }
    instance_digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "ids": ids,
        "alpha": alpha,
        "scheme": scheme,
        "evidence": evidence,
        "utility": utility,
        "costs": costs,
        "capacity": capacity,
        "capacity_fraction": capacity_fraction,
        "seed": seed,
        "instance_sha256": instance_digest,
    }


def _scipy_milp_inputs(model: dict[str, Any]):
    """Convert exact model rows to SciPy's sparse LinearConstraint inputs."""
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint
        from scipy.sparse import coo_matrix
    except ImportError as error:
        raise RuntimeError(
            "SciPy is required for the HiGHS adapter; install scipy>=1.9"
        ) from error

    ids = model["metadata"]["active_ids"]
    variable_names = (
        [f"x_{i}" for i in ids]
        + ["r"]
        + [f"y_{i}" for i in ids]
    )
    index = {name: j for j, name in enumerate(variable_names)}
    nvar = len(variable_names)

    objective = np.zeros(nvar, dtype=float)
    for name, coefficient in model["objective"].items():
        objective[index[name]] = -float(coefficient)  # SciPy minimizes.

    integrality = np.zeros(nvar, dtype=np.uint8)
    lower = np.zeros(nvar, dtype=float)
    upper = np.full(nvar, np.inf, dtype=float)
    for i in ids:
        integrality[index[f"x_{i}"]] = 1
        upper[index[f"x_{i}"]] = 1.0
    upper[index["r"]] = float(len(ids))

    rows: list[dict[str, Any]] = []
    constraints = model["constraints"]
    rows.append(constraints["rejection_count"])
    rows.extend(constraints["closure_auxiliary"])
    rows.append(constraints["closure_sum"])
    if constraints["capacity"] is not None:
        rows.append(constraints["capacity"])
    rows.extend(constraints["persistence"])

    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []
    row_lower = np.full(len(rows), -np.inf, dtype=float)
    row_upper = np.full(len(rows), np.inf, dtype=float)

    for row_index, row in enumerate(rows):
        for variable, coefficient in row["lhs"].items():
            value = float(coefficient)
            if value != 0.0:
                row_indices.append(row_index)
                col_indices.append(index[variable])
                data.append(value)
        rhs = float(row["rhs"])
        if row["sense"] == "equal":
            row_lower[row_index] = rhs
            row_upper[row_index] = rhs
        elif row["sense"] == "less_equal":
            row_upper[row_index] = rhs
        elif row["sense"] == "greater_equal":
            row_lower[row_index] = rhs
        else:
            raise ValueError(f"unknown row sense {row['sense']!r}")

    matrix = coo_matrix(
        (data, (row_indices, col_indices)),
        shape=(len(rows), nvar),
        dtype=float,
    ).tocsr()
    linear = LinearConstraint(matrix, row_lower, row_upper)
    bounds = Bounds(lower, upper)
    return {
        "objective": objective,
        "integrality": integrality,
        "bounds": bounds,
        "linear_constraint": linear,
        "variable_names": variable_names,
        "rows": len(rows),
        "nonzeros": len(data),
        "matrix_shape": matrix.shape,
        "matrix": matrix,
    }


def solve_with_scipy_highs(
    model: dict[str, Any],
    *,
    time_limit: float = 30.0,
    relative_gap: float = 1e-6,
) -> dict[str, Any]:
    """Solve using scipy.optimize.milp (HiGHS); exactly verify returned x."""
    try:
        import numpy as np
        import scipy
        from scipy.optimize import milp
    except ImportError as error:
        raise RuntimeError(
            "SciPy is required for the HiGHS adapter; install scipy>=1.9"
        ) from error

    if time_limit <= 0:
        raise ValueError("time_limit must be positive")
    if relative_gap < 0:
        raise ValueError("relative_gap must be nonnegative")

    built_at = time.perf_counter()
    inputs = _scipy_milp_inputs(model)
    build_seconds = time.perf_counter() - built_at

    started = time.perf_counter()
    result = milp(
        c=inputs["objective"],
        integrality=inputs["integrality"],
        bounds=inputs["bounds"],
        constraints=inputs["linear_constraint"],
        options={
            "time_limit": float(time_limit),
            "mip_rel_gap": float(relative_gap),
            "presolve": True,
        },
    )
    solve_seconds = time.perf_counter() - started

    output: dict[str, Any] = {
        "solver": "scipy.optimize.milp / HiGHS",
        "scipy_version": scipy.__version__,
        "status_code": int(result.status),
        "status": {
            0: "optimal",
            1: "limit_reached",
            2: "infeasible",
            3: "unbounded",
            4: "solver_error",
        }.get(int(result.status), "unknown"),
        "message": str(result.message),
        "optimality_proven": int(result.status) == 0,
        "build_seconds": build_seconds,
        "solve_seconds": solve_seconds,
        "total_seconds": build_seconds + solve_seconds,
        "model_variables": len(inputs["variable_names"]),
        "model_rows": inputs["rows"],
        "matrix_nonzeros": inputs["nonzeros"],
        "matrix_shape": list(inputs["matrix_shape"]),
        "mip_node_count": getattr(result, "mip_node_count", None),
        "mip_gap": getattr(result, "mip_gap", None),
        "minimization_dual_bound": getattr(result, "mip_dual_bound", None),
    }
    if result.x is None:
        output.update({
            "has_incumbent": False,
            "exact_verification": {"valid": False, "failures": ["no solver incumbent"]},
        })
        return output

    ids = model["metadata"]["active_ids"]
    n = len(ids)
    raw_x = result.x[:n]
    rounded = np.rint(raw_x).astype(int)
    integrality_error = float(np.max(np.abs(raw_x - rounded))) if n else 0.0
    rejection_map = {i: int(rounded[j]) for j, i in enumerate(ids)}
    verifier_started = time.perf_counter()
    exact = verify_affine_binary_decision(model, rejection_map)
    verifier_seconds = time.perf_counter() - verifier_started

    utility = sum(
        (model["objective"][f"x_{i}"] * rejection_map[i] for i in ids),
        Fraction(0),
    )
    cost = sum(
        (model["metadata"]["costs"][i] * rejection_map[i] for i in ids),
        Fraction(0),
    )
    dual = getattr(result, "mip_dual_bound", None)
    utility_upper = None if dual is None else -float(dual)
    utility_value = float(utility)
    gap_abs = (
        None if utility_upper is None
        else max(0.0, utility_upper - utility_value)
    )
    output.update({
        "has_incumbent": True,
        "rejection_ids": [i for i in ids if rejection_map[i]],
        "rejection_count": sum(rounded),
        "utility_exact": [utility.numerator, utility.denominator],
        "utility": utility_value,
        "utility_upper_bound": utility_upper,
        "absolute_objective_gap": gap_abs,
        "relative_objective_gap": (
            None if gap_abs is None
            else gap_abs / max(1.0, abs(utility_value))
        ),
        "cost_exact": [cost.numerator, cost.denominator],
        "cost": float(cost),
        "integrality_error": integrality_error,
        "verifier_seconds": verifier_seconds,
        "exact_verification": exact,
        "accepted": exact["valid"] and integrality_error <= 1e-7,
    })
    if integrality_error > 1e-7:
        output["accepted"] = False
        output["integrality_failure"] = (
            "solver binary variables were not within the declared rounding tolerance"
        )
    return output


def exhaustive_optimum(model: dict[str, Any], *, max_n: int = 22) -> dict[str, Any]:
    """Exact action oracle for small families; used to audit solver optimality."""
    ids = model["metadata"]["active_ids"]
    n = len(ids)
    if n > max_n:
        raise ValueError(f"exhaustive oracle limited to n <= {max_n}")
    best_utility = None
    best_cost = None
    best_mask = None
    checked = 0
    for mask in range(1 << n):
        action = {i: (mask >> j) & 1 for j, i in enumerate(ids)}
        check = verify_affine_binary_decision(model, action)
        checked += 1
        if not check["valid"]:
            continue
        utility = sum(
            (model["objective"][f"x_{i}"] * action[i] for i in ids),
            Fraction(0),
        )
        cost = check["cost"]
        selected_tuple = tuple(i for i in ids if action[i])
        best_tuple = (
            tuple(i for j, i in enumerate(ids) if best_mask & (1 << j))
            if best_mask is not None else None
        )
        better = best_mask is None or utility > best_utility
        if not better and utility == best_utility:
            better = cost < best_cost
        if not better and utility == best_utility and cost == best_cost:
            better = selected_tuple < best_tuple
        if better:
            best_mask, best_utility, best_cost = mask, utility, cost
    return {
        "checked_actions": checked,
        "feasible": best_mask is not None,
        "rejection_ids": (
            [] if best_mask is None
            else [i for j, i in enumerate(ids) if best_mask & (1 << j)]
        ),
        "utility_exact": (
            None if best_utility is None
            else [best_utility.numerator, best_utility.denominator]
        ),
        "cost_exact": (
            None if best_cost is None
            else [best_cost.numerator, best_cost.denominator]
        ),
    }


def run_benchmark(
    sizes: Iterable[int],
    *,
    seed: int = 20261101,
    time_limit: float = 30.0,
    relative_gap: float = 1e-6,
    include_exhaustive_up_to: int = 16,
) -> dict[str, Any]:
    """Run a deterministic solver scaling sweep and preserve all raw outcomes."""
    rows = []
    for n in sizes:
        instance = make_instance(n, seed + n)
        model = build_affine_closure_milp(
            instance["ids"],
            instance["evidence"],
            instance["alpha"],
            scheme=instance["scheme"],
            values=instance["utility"],
            costs=instance["costs"],
            capacity=instance["capacity"],
        )
        solved = solve_with_scipy_highs(
            model,
            time_limit=time_limit,
            relative_gap=relative_gap,
        )
        row = {
            "n": n,
            "seed": instance["seed"],
            "instance_sha256": instance["instance_sha256"],
            "weight_scheme": instance["scheme"],
            "capacity": [
                instance["capacity"].numerator,
                instance["capacity"].denominator,
            ],
            **solved,
        }
        if n <= include_exhaustive_up_to:
            exact = exhaustive_optimum(model)
            row["exhaustive_optimum"] = exact
            if solved.get("accepted") and exact["feasible"]:
                row["exhaustive_objective_match"] = (
                    solved["utility_exact"] == exact["utility_exact"]
                    and solved["cost_exact"] == exact["cost_exact"]
                    and solved["rejection_ids"] == exact["rejection_ids"]
                )
            else:
                row["exhaustive_objective_match"] = None
        rows.append(row)
    try:
        package_version = importlib.metadata.version("scipy")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    return {
        "schema": "affine-projective-closure-highs-benchmark-v1",
        "purpose": "solver engineering only; not a statistical FDR validity or power benchmark",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scipy": package_version,
            "solver": "HiGHS through scipy.optimize.milp",
        },
        "configuration": {
            "base_seed": seed,
            "time_limit_seconds": time_limit,
            "relative_mip_gap_target": relative_gap,
            "exhaustive_check_n_max": include_exhaustive_up_to,
        },
        "results": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="10,25,50,100,250,500,1000")
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument("--relative-gap", type=float, default=1e-6)
    parser.add_argument("--exhaustive-up-to", type=int, default=16)
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    sizes = tuple(int(item) for item in args.sizes.split(",") if item.strip())
    result = run_benchmark(
        sizes,
        seed=args.seed,
        time_limit=args.time_limit,
        relative_gap=args.relative_gap,
        include_exhaustive_up_to=args.exhaustive_up_to,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output == "-":
        print(rendered)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
