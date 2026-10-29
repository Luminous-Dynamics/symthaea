#!/usr/bin/env python3
"""Split-calibrated prior ambiguity sets for finite adaptive causal probing.

This is a synthetic research experiment, not Symthaea repository SCM integration.
A fixed synthetic meta-population generates true priors over 16 hypotheses. A
multinomial calibration sample yields a smoothed prior estimate. The ambiguity
set constrains a compact feature map (graph marginals, kernel marginals,
diagonal mass, anti-diagonal mass) to lie within a radius calibrated on a held-out
calibration fold. Policy quality and set coverage are evaluated on an independent
test fold. Claims are conditional on this declared synthetic meta-distribution.
"""
from __future__ import annotations
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np

from constrained_prior_polytope_benchmark import (
    ambiguity_polytope as fixed_polytope,
    maximize_linear_over_polytope,
    solve_constrained_minimax,
)
from prior_set_coverage_double_oracle_benchmark import (
    OBJECTIVES, N_GRAPH, N_KERNEL, N_HYP, build_risk_tensor,
    candidate_priors, prior_risk_matrix, solve_zero_sum,
)

SCHEMA = "symthaea-split-calibrated-prior-ambiguity-v1"
TARGET_COVERAGE = 0.90
OBSERVATION_COUNT = 120
ATOM_PSEUDOCOUNT = 0.5


def feature_matrix():
    """Linear feature matrix for graph/kernel marginals and association masses."""
    C = []
    for g in range(N_GRAPH):
        C.append([1.0 if i // N_KERNEL == g else 0.0 for i in range(N_HYP)])
    for k in range(N_KERNEL):
        C.append([1.0 if i % N_KERNEL == k else 0.0 for i in range(N_HYP)])
    C.append([1.0 if i // N_KERNEL == i % N_KERNEL else 0.0 for i in range(N_HYP)])
    C.append([1.0 if i % N_KERNEL == N_KERNEL - 1 - i // N_KERNEL else 0.0 for i in range(N_HYP)])
    return np.asarray(C, dtype=float)


def meta_regime_alphas():
    """Illustrative prior-generating regimes, including sparse and dependent priors."""
    alpha = []
    alpha.append(np.full(N_HYP, 2.0))  # diffuse
    alpha.append(np.full(N_HYP, 0.15))  # sparse/extreme
    graph_w = np.asarray([0.55, 0.25, 0.13, 0.07])
    alpha.append(np.asarray([0.12 + 2.5 * graph_w[g] for g in range(N_GRAPH) for _k in range(N_KERNEL)]))
    kernel_w = np.asarray([0.52, 0.26, 0.14, 0.08])
    alpha.append(np.asarray([0.12 + 2.5 * kernel_w[k] for _g in range(N_GRAPH) for k in range(N_KERNEL)]))
    alpha.append(np.asarray([2.8 if g == k else 0.13 for g in range(N_GRAPH) for k in range(N_KERNEL)]))
    alpha.append(np.asarray([2.8 if k == N_KERNEL - 1 - g else 0.13 for g in range(N_GRAPH) for k in range(N_KERNEL)]))
    return tuple(alpha)


def draw_true_prior(rng, alphas):
    regime = int(rng.integers(0, len(alphas)))
    return rng.dirichlet(alphas[regime]), regime


def estimate_prior(rng, q_true, n_observations=OBSERVATION_COUNT):
    if n_observations < 1:
        raise ValueError("n_observations must be positive")
    counts = rng.multinomial(n_observations, q_true)
    # Dirichlet(0.5) smoothing ensures positive support.
    q_hat = (counts + ATOM_PSEUDOCOUNT) / (n_observations + ATOM_PSEUDOCOUNT * N_HYP)
    return q_hat


def feature_radius(q_true, q_hat, C=None):
    C = feature_matrix() if C is None else C
    return float(np.max(np.abs(C @ np.asarray(q_true) - C @ np.asarray(q_hat))))


def conformal_radius(scores, target_coverage=TARGET_COVERAGE):
    """Finite-sample split-conformal order statistic for exchangeable scores."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError("calibration scores must be a nonempty finite vector")
    if not 0.0 < target_coverage < 1.0:
        raise ValueError("target coverage must lie strictly between zero and one")
    rank = int(math.ceil((len(values) + 1) * target_coverage))
    # If the requested level exceeds finite-sample resolution, the conservative
    # answer is an unbounded feature interval; here the feature range is [0,1].
    if rank > len(values):
        return 1.0
    return float(np.sort(values)[rank - 1])


def calibrated_polytope(q_hat, radius, floor=0.0, C=None):
    """Create Q(q_hat,radius)={q: |Cq-Cq_hat|<=radius, sum(q)=1, q>=floor}."""
    C = feature_matrix() if C is None else np.asarray(C, dtype=float)
    q_hat = np.asarray(q_hat, dtype=float)
    if q_hat.shape != (N_HYP,) or np.any(~np.isfinite(q_hat)) or np.any(q_hat < 0):
        raise ValueError("q_hat must be a finite nonnegative prior vector")
    if abs(float(q_hat.sum()) - 1.0) > 1e-9:
        raise ValueError("q_hat must sum to one")
    if not math.isfinite(radius) or not 0 <= radius <= 1:
        raise ValueError("radius must lie in [0,1]")
    if not math.isfinite(floor) or floor < 0 or floor * N_HYP >= 1:
        raise ValueError("invalid atom floor")
    center = C @ q_hat
    lo = np.maximum(0.0, center - radius)
    hi = np.minimum(1.0, center + radius)
    A_ub, b_ub, names = [], [], []
    for j, row in enumerate(C):
        A_ub.append(row); b_ub.append(hi[j]); names.append(f"feature_{j}_upper")
        A_ub.append(-row); b_ub.append(-lo[j]); names.append(f"feature_{j}_lower")
    poly = {"floor": floor, "A_ub": np.asarray(A_ub), "b_ub": np.asarray(b_ub),
            "bound_names": names, "feature_center": center.tolist(), "radius": radius}
    # q_hat must itself be feasible, modulo floating-point tolerance.
    if np.any(poly["A_ub"] @ q_hat - poly["b_ub"] > 1e-9):
        raise AssertionError("calibrated ambiguity set excludes its center")
    return poly


def inside_polytope(q, poly, tol=1e-9):
    q = np.asarray(q, dtype=float)
    return bool(abs(float(q.sum()) - 1.0) <= tol and np.min(q) >= poly["floor"] - tol
                and np.max(poly["A_ub"] @ q - poly["b_ub"]) <= tol)


def wilson_interval(successes, n, z=1.959963984540054):
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt((p*(1-p)/n) + z*z/(4*n*n)) / den
    return [max(0.0, center-half), min(1.0, center+half)]


def run_benchmark(n_calibration=400, n_test=120, seed=20261028,
                  target_coverage=TARGET_COVERAGE, n_observations=OBSERVATION_COUNT):
    if n_calibration < 10 or n_test < 1:
        raise ValueError("need at least 10 calibration and 1 test draws")
    start = time.perf_counter()
    rng = np.random.default_rng(seed)
    alphas = meta_regime_alphas()
    C = feature_matrix()
    calibration = []
    for _ in range(n_calibration):
        q, regime = draw_true_prior(rng, alphas)
        q_hat = estimate_prior(rng, q, n_observations)
        calibration.append({"q": q, "q_hat": q_hat, "regime": regime,
                            "score": feature_radius(q, q_hat, C)})
    radius = conformal_radius([x["score"] for x in calibration], target_coverage)

    # Fix the risk matrix once; only ambiguity set and test prior change.
    tensor, legacy_priors, trees = build_risk_tensor()
    fixed_Q = fixed_polytope(marginal_low=0.10, marginal_high=0.40)
    fixed_results = {}
    six_results = {}
    full_results = {}
    for objective in OBJECTIVES:
        A = tensor[objective]["risk_by_policy_and_world"]
        fixed_results[objective] = solve_constrained_minimax(A, fixed_Q)
        six_results[objective] = solve_zero_sum(prior_risk_matrix(A, list(legacy_priors.values())))
        full_results[objective] = solve_zero_sum(A)

    eval_rows = []
    for test_id in range(n_test):
        q, regime = draw_true_prior(rng, alphas)
        q_hat = estimate_prior(rng, q, n_observations)
        score = feature_radius(q, q_hat, C)
        Q_cal = calibrated_polytope(q_hat, radius, floor=0.0, C=C)
        in_cal = inside_polytope(q, Q_cal)
        in_fixed = inside_polytope(q, fixed_Q)
        row = {"test_id": test_id, "meta_regime": regime, "feature_score": score,
               "calibrated_set_contains_true_prior": in_cal,
               "fixed_marginal_set_contains_true_prior": in_fixed}
        for objective in OBJECTIVES:
            A = tensor[objective]["risk_by_policy_and_world"]
            calibrated = solve_constrained_minimax(A, Q_cal)
            q_risks = A @ q
            oracle_known_q = float(np.min(q_risks))
            cal_mu = np.asarray(calibrated["policy_weights"], dtype=float)
            fixed_mu = np.asarray(fixed_results[objective]["policy_weights"], dtype=float)
            six_mu = np.asarray(six_results[objective]["policy_weights"], dtype=float)
            full_mu = np.asarray(full_results[objective]["policy_weights"], dtype=float)
            row[objective] = {
                "oracle_known_prior_risk_bits": oracle_known_q,
                "calibrated_policy_risk_bits": float(cal_mu @ q_risks),
                "calibrated_policy_regret_bits": max(0.0, float(cal_mu @ q_risks) - oracle_known_q),
                "calibrated_robust_value_bits": calibrated["lp_value_bits"],
                "calibrated_certificate_gap_bits": calibrated["gap_bits"],
                "fixed_010_040_policy_regret_bits": max(0.0, float(fixed_mu @ q_risks) - oracle_known_q),
                "six_prior_policy_regret_bits": max(0.0, float(six_mu @ q_risks) - oracle_known_q),
                "full_simplex_policy_regret_bits": max(0.0, float(full_mu @ q_risks) - oracle_known_q),
            }
        eval_rows.append(row)

    metrics = {}
    for objective in OBJECTIVES:
        key = objective
        cal_regrets = np.asarray([r[key]["calibrated_policy_regret_bits"] for r in eval_rows])
        fixed_regrets = np.asarray([r[key]["fixed_010_040_policy_regret_bits"] for r in eval_rows])
        six_regrets = np.asarray([r[key]["six_prior_policy_regret_bits"] for r in eval_rows])
        full_regrets = np.asarray([r[key]["full_simplex_policy_regret_bits"] for r in eval_rows])
        metrics[key] = {
            "calibrated_set_empirical_coverage": sum(r["calibrated_set_contains_true_prior"] for r in eval_rows) / n_test,
            "calibrated_set_coverage_wilson_95": wilson_interval(sum(r["calibrated_set_contains_true_prior"] for r in eval_rows), n_test),
            "fixed_010_040_empirical_coverage": sum(r["fixed_marginal_set_contains_true_prior"] for r in eval_rows) / n_test,
            "target_coverage": target_coverage,
            "calibrated_radius": radius,
            "calibrated_policy_regret_mean_bits": float(cal_regrets.mean()),
            "calibrated_policy_regret_p95_bits": float(np.quantile(cal_regrets, 0.95)),
            "fixed_set_policy_regret_mean_bits": float(fixed_regrets.mean()),
            "six_prior_policy_regret_mean_bits": float(six_regrets.mean()),
            "full_simplex_policy_regret_mean_bits": float(full_regrets.mean()),
            "all_calibrated_cases_closed_certificate": bool(all(r[key]["calibrated_certificate_gap_bits"] <= 2.1e-7 for r in eval_rows)),
        }
    return {
        "schema": SCHEMA,
        "status": "SYNTHETIC_META_DISTRIBUTION_SPLIT_CALIBRATION_NOT_REAL_WORLD_COVERAGE",
        "protocol": {
            "n_calibration": n_calibration, "n_test": n_test, "seed": seed,
            "target_coverage": target_coverage, "n_observations_for_q_estimate": n_observations,
            "estimate": "multinomial counts plus symmetric Dirichlet(0.5) smoothing",
            "set": "linear constraints on 4 graph marginals, 4 kernel marginals, diagonal and anti-diagonal masses, radius calibrated on held-out calibration fold",
            "exchangeability_assumption": "calibration and test pairs independently sampled from the same declared synthetic meta-population",
            "coverage_claim": "split-conformal marginal coverage conditional on this synthetic exchangeable protocol only; finite test coverage reported with Wilson interval",
            "meta_regimes": ["diffuse", "sparse", "graph-skewed", "kernel-skewed", "diagonal-correlated", "anti-correlated"],
        },
        "model": {"latent_hypotheses": N_HYP, "policy_trees": len(trees), "horizon": 2,
                  "objectives": list(OBJECTIVES), "fixed_baseline_marginal_bounds": [0.10, 0.40],
                  "atom_floor_calibrated_set": 0.0, "likelihood": "same synthetic model as prior-set coverage pilot"},
        "metrics": metrics,
        "timing_seconds_local": time.perf_counter() - start,
        "limitations": ["synthetic 16-hypothesis support and horizon-two policies", "meta-population is illustrative, not fitted to real causal environments",
                        "only selected low-dimensional prior features are calibrated; the full joint prior is not directly covered by feature radius alone beyond implied LP constraints",
                        "no Symthaea SCM integration", "no external causal benchmark", "no SOTA or speedup claim"],
        "rows": eval_rows,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--n-calibration", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=120)
    ap.add_argument("--observations", type=int, default=OBSERVATION_COUNT)
    ap.add_argument("--seed", type=int, default=20261028)
    args = ap.parse_args()
    result = run_benchmark(args.n_calibration, args.n_test, args.seed, n_observations=args.observations)
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({"output": str(args.output), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
                      "calibrated_radius": next(iter(result["metrics"].values()))["calibrated_radius"],
                      "coverage_by_objective": {k:v["calibrated_set_empirical_coverage"] for k,v in result["metrics"].items()},
                      "rows": len(result["rows"]), "all_certificates_closed": all(m["all_calibrated_cases_closed_certificate"] for m in result["metrics"].values())}, indent=2))


if __name__ == "__main__":
    main()
