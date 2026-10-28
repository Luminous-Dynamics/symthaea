#!/usr/bin/env python3
"""Robust policy optimization over explicit constrained prior polytopes.

Research-only extension to prior_set_coverage_double_oracle_benchmark.py. It
uses the same synthetic likelihood/risk tensor but replaces a list of sampled
priors with interpretable linear constraints on prior weights and marginals.
This is not yet integrated with Symthaea's repository SCM semantics.
"""
from __future__ import annotations
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from prior_set_coverage_double_oracle_benchmark import (
    OBJECTIVES, build_risk_tensor, candidate_priors, prior_risk_matrix,
    solve_zero_sum, point_mass, N_GRAPH, N_KERNEL, N_HYP, TOL,
)

SCHEMA = "symthaea-constrained-prior-polytope-benchmark-v1"
MARGIN_BITS = 1e-9


def ambiguity_polytope(floor: float = 0.005, marginal_low: float = 0.10, marginal_high: float = 0.40):
    """Q: simplex plus graph/kernel marginal and association-mass bounds."""
    if not math.isfinite(floor) or floor < 0 or floor * N_HYP >= 1:
        raise ValueError("prior floor must be finite, nonnegative, and < 1/n_hypotheses")
    if not (math.isfinite(marginal_low) and math.isfinite(marginal_high) and 0 <= marginal_low <= marginal_high <= 1):
        raise ValueError("invalid marginal bounds")
    rows, lows, highs, names = [], [], [], []
    def add_bound(name, indices, low, high):
        row = np.zeros(N_HYP, dtype=float)
        row[list(indices)] = 1.0
        rows.append(row); lows.append(low); highs.append(high); names.append(name)
    for g in range(N_GRAPH):
        add_bound(f"graph_marginal_{g}", [g * N_KERNEL + k for k in range(N_KERNEL)], marginal_low, marginal_high)
    for k in range(N_KERNEL):
        add_bound(f"kernel_marginal_{k}", [g * N_KERNEL + k for g in range(N_GRAPH)], marginal_low, marginal_high)
    add_bound("graph_kernel_diagonal_mass", [g * N_KERNEL + g for g in range(N_GRAPH)], 0.10, 0.60)
    add_bound("graph_kernel_antidiagonal_mass", [g * N_KERNEL + (N_KERNEL - 1 - g) for g in range(N_KERNEL)], 0.10, 0.60)
    # A q <= b representation of both upper and lower bounds.
    A_ub, b_ub = [], []
    for row, low, high in zip(rows, lows, highs):
        A_ub.append(row); b_ub.append(high)
        A_ub.append(-row); b_ub.append(-low)
    A_ub = np.asarray(A_ub, dtype=float)
    b_ub = np.asarray(b_ub, dtype=float)
    # Confirm feasibility before optimizing a risk objective.
    feasibility = linprog(np.zeros(N_HYP), A_ub=A_ub, b_ub=b_ub,
                          A_eq=np.ones((1, N_HYP)), b_eq=[1.0],
                          bounds=[(floor, 1.0)] * N_HYP, method="highs")
    if not feasibility.success:
        raise ValueError(f"ambiguity polytope is infeasible: {feasibility.message}")
    return {"floor": floor, "A_ub": A_ub, "b_ub": b_ub, "bound_names": names,
            "declared_bounds": {n: [lo, hi] for n, lo, hi in zip(names, lows, highs)},
            "feasible_witness": feasibility.x.tolist()}


def maximize_linear_over_polytope(coefficients, polytope):
    c = np.asarray(coefficients, dtype=float)
    if c.shape != (N_HYP,) or np.any(~np.isfinite(c)):
        raise ValueError("risk coefficients must be a finite vector over all hypotheses")
    result = linprog(-c, A_ub=polytope["A_ub"], b_ub=polytope["b_ub"],
                     A_eq=np.ones((1, N_HYP)), b_eq=[1.0],
                     bounds=[(polytope["floor"], 1.0)] * N_HYP, method="highs")
    if not result.success:
        raise RuntimeError(f"world LP failed: {result.message}")
    q = np.asarray(result.x, dtype=float)
    value = float(c @ q)
    # Independent feasibility and objective re-score.
    if abs(float(q.sum()) - 1.0) > 1e-8:
        raise AssertionError("world oracle returned a non-normalized prior")
    if np.any(polytope["A_ub"] @ q - polytope["b_ub"] > 1e-8):
        raise AssertionError("world oracle returned a prior outside ambiguity set")
    if np.any(q < polytope["floor"] - 1e-8):
        raise AssertionError("world oracle violated prior floor")
    if abs(value - float(-result.fun)) > 1e-8:
        raise AssertionError("world oracle objective failed independent re-score")
    return {"value_bits": value, "prior": q.tolist(), "minimum_prior_atom": float(q.min()),
            "lp_status": result.message}


def solve_constrained_minimax(A, polytope, tolerance=TOL):
    """Solve max_{q in Q} min_policy R(policy,q), then recover minimax policy by dual.

    Minimax equality holds for this finite policy simplex and compact convex prior
    polytope. The prior LP maximizes t subject to A q >= t for every deterministic
    policy and q in Q. Dual multipliers on the policy constraints yield a policy mix.
    An independent worst-case-prior LP for that mix gives the upper endpoint.
    """
    A = np.asarray(A, dtype=float)
    if A.shape != (8, N_HYP) or np.any(~np.isfinite(A)):
        raise ValueError("risk matrix must be finite with shape (8, 16)")
    # maximize t -> minimize -t; -Aq + t <= 0 enforces every policy risk >= t.
    n_policy = A.shape[0]
    risk_ub = np.column_stack([-A, np.ones(n_policy)])
    poly_ub = np.column_stack([polytope["A_ub"], np.zeros(len(polytope["b_ub"]))])
    A_ub = np.vstack([risk_ub, poly_ub])
    b_ub = np.r_[np.zeros(n_policy), polytope["b_ub"]]
    A_eq = np.r_[np.ones(N_HYP), 0.0][None, :]
    result = linprog(np.r_[np.zeros(N_HYP), -1.0], A_ub=A_ub, b_ub=b_ub,
                     A_eq=A_eq, b_eq=[1.0],
                     bounds=[(polytope["floor"], 1.0)] * N_HYP + [(None, None)], method="highs")
    if not result.success:
        raise RuntimeError(f"ambiguity-set value LP failed: {result.message}")
    q_star = np.asarray(result.x[:N_HYP], dtype=float)
    value_lp = float(result.x[-1])
    policy_duals = -np.asarray(result.ineqlin.marginals[:n_policy], dtype=float)
    policy_duals = np.maximum(policy_duals, 0.0)
    if policy_duals.sum() <= 0:
        raise AssertionError("dual policy mixture has no positive mass")
    mu = policy_duals / policy_duals.sum()
    # Lower endpoint: adversarial prior q*, followed by best policy response.
    lower_raw = float(np.min(A @ q_star))
    # Upper endpoint: independently maximize risk of recovered policy mix over Q.
    upper_oracle = maximize_linear_over_polytope(A.T @ mu, polytope)
    upper_raw = upper_oracle["value_bits"]
    if lower_raw > upper_raw + 2e-7:
        raise AssertionError(f"inverted robust interval lower={lower_raw}, upper={upper_raw}")
    lower, upper = lower_raw - MARGIN_BITS, upper_raw + MARGIN_BITS
    gap = upper - lower
    if abs(value_lp - lower_raw) > 2e-7:
        raise AssertionError("value LP disagrees with best-policy response at adversarial prior")
    if abs(value_lp - upper_raw) > 2e-7:
        raise AssertionError("LP dual policy mix does not close independent upper bound")
    if abs(float(mu.sum()) - 1.0) > 1e-8 or np.any(mu < -1e-12):
        raise AssertionError("dual weights do not form a policy mixture")
    return {
        "status": "CERTIFIED_CONSTRAINED_POLYTOPE_GAME" if gap <= tolerance + 1e-7 else "NOT_CERTIFIED",
        "lp_value_bits": value_lp,
        "lower_uninflated_bits": lower_raw,
        "upper_uninflated_bits": upper_raw,
        "lower_bits": lower,
        "upper_bits": upper,
        "gap_bits": gap,
        "numerical_margin_each_side_bits": MARGIN_BITS,
        "margin_is_formal_interval_arithmetic_proof": False,
        "policy_weights": mu.tolist(),
        "policy_support_size": int(np.count_nonzero(mu > 1e-8)),
        "adversarial_prior": q_star.tolist(),
        "adversarial_prior_min_atom": float(q_star.min()),
        "upper_bound_world_oracle": upper_oracle,
        "lp_iterations": int(getattr(result, "nit", 0)),
    }


def run_benchmark():
    start = time.perf_counter()
    tensor, priors, trees = build_risk_tensor()
    polytopes = {
        "bounded_marginals": ambiguity_polytope(marginal_low=0.10, marginal_high=0.40),
        "contains_all_legacy_priors": ambiguity_polytope(marginal_low=0.05, marginal_high=0.60),
    }
    rows = []
    named = list(priors.values())
    for poly_name, poly in polytopes.items():
        for objective in OBJECTIVES:
            A = tensor[objective]["risk_by_policy_and_world"]
            t0 = time.perf_counter()
            six = solve_zero_sum(prior_risk_matrix(A, named))
            six_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            full = solve_zero_sum(A)
            full_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            constrained = solve_constrained_minimax(A, poly)
            constrained_seconds = time.perf_counter() - t0
            six_mu = np.asarray(six["policy_weights"], dtype=float)
            t0 = time.perf_counter()
            legacy_upper = maximize_linear_over_polytope(A.T @ six_mu, poly)
            legacy_rescore_seconds = time.perf_counter() - t0
            prior_membership = []
            for name, q in priors.items():
                q_arr = np.asarray(q, dtype=float)
                violations = poly["A_ub"] @ q_arr - poly["b_ub"]
                prior_membership.append({"name": name,
                    "inside": bool(np.max(violations) <= 1e-9 and q_arr.min() >= poly["floor"] - 1e-9),
                    "max_linear_constraint_violation": float(max(0.0, np.max(violations))),
                    "minimum_atom": float(q_arr.min())})
            if poly_name == "contains_all_legacy_priors" and not all(x["inside"] for x in prior_membership):
                raise AssertionError("the declared coverage-preserving polytope excludes a legacy prior")
            rows.append({
                "ambiguity_set": poly_name,
                "objective": objective,
                "six_prior_value_bits": six["value_bits"],
                "six_prior_policy_worst_case_over_Q_bits": legacy_upper["value_bits"],
                "full_simplex_value_bits": full["value_bits"],
                "constrained_polytope": constrained,
                "legacy_six_prior_policy_regret_over_Q_bits": max(0.0, legacy_upper["value_bits"] - constrained["lp_value_bits"]),
                "legacy_prior_membership": prior_membership,
                "all_six_legacy_priors_inside_Q": all(x["inside"] for x in prior_membership),
                "timing_seconds_local": {
                    "six_prior_master_lp": six_seconds,
                    "full_simplex_master_lp": full_seconds,
                    "constrained_polytope_master_and_certificate": constrained_seconds,
                    "legacy_policy_rescore_over_polytope": legacy_rescore_seconds
                }
            })
    return {
        "schema": SCHEMA,
        "status": "CONSTRAINED_PRIOR_POLYTOPE_SYNTHETIC_PILOT_NOT_SCM_INTEGRATION",
        "model": {"latent_hypotheses": N_HYP, "policy_trees": len(trees), "horizon": 2,
                  "objectives": list(OBJECTIVES), "prior_floor": 0.005,
                  "ambiguity_sets": {
                    "bounded_marginals": {"graph_marginal_each": [0.10, 0.40], "kernel_marginal_each": [0.10, 0.40], "diagonal_mass": [0.10, 0.60], "antidiagonal_mass": [0.10, 0.60]},
                    "contains_all_legacy_priors": {"graph_marginal_each": [0.05, 0.60], "kernel_marginal_each": [0.05, 0.60], "diagonal_mass": [0.10, 0.60], "antidiagonal_mass": [0.10, 0.60]}},
                  "interpretation": "declared partial prior knowledge over fixed support; not calibrated from external data"},
        "execution": {"python": platform.python_version(), "numpy": np.__version__, "scipy": __import__("scipy").__version__,
                      "total_benchmark_seconds_local": time.perf_counter() - start},
        "limitations": ["synthetic likelihood, not Symthaea SCM implementation", "horizon two and eight policy trees",
                        "LP tolerances and outward margin are not formal interval arithmetic", "no external benchmark", "no speedup claim"],
        "rows": rows,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = run_benchmark()
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({"output": str(args.output), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                      "rows": len(result["rows"]), "all_certified": all(r["constrained_polytope"]["status"] == "CERTIFIED_CONSTRAINED_POLYTOPE_GAME" for r in result["rows"])}, indent=2))

if __name__ == "__main__":
    main()
