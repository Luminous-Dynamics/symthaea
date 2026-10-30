#!/usr/bin/env python3
"""Independent, corrected reproduction of the 200-hypothesis e-process benchmark.

The benchmark follows the dependent-normal model described by Tavyrikov,
Goeman & de Heide (EJS 2026), but pins all randomness and corrects the
reference-script adjuster-domain issue by defining M_t=max(1,sup_{s<=t} E_s).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import random_correlation

SCHEMA = "symthaea-multistream-dependent-normal-v1"


def log_adjuster_a1_from_log_m(log_m: np.ndarray) -> np.ndarray:
    x = np.asarray(log_m, dtype=np.float64)
    if np.any(~np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("log M must be finite and nonnegative")
    out = np.empty_like(x)
    small = x < 1e-2
    z = x[small]
    # log(A1) where A1 = (exp(x)-1-x)/x^2. Taylor expansion avoids
    # catastrophic cancellation around x=0; A1(1)=1/2.
    ratio = (0.5 + z/6.0 + z*z/24.0 + z**3/120.0
             + z**4/720.0 + z**5/5040.0 + z**6/40320.0)
    out[small] = np.log(ratio)
    z = x[~small]
    out[~small] = z + np.log1p(-(1.0 + z) * np.exp(-z)) - 2.0 * np.log(z)
    return out


def log_adjuster_a2_from_log_m(log_m: np.ndarray) -> np.ndarray:
    x = np.asarray(log_m, dtype=np.float64)
    if np.any(~np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("log M must be finite and nonnegative")
    out = np.full_like(x, -np.inf)
    positive = x > 0.0
    z = x[positive] / 2.0
    small = z < 50.0
    tmp = np.empty_like(z)
    tmp[small] = np.log(np.expm1(z[small]))
    tmp[~small] = z[~small] + np.log1p(-np.exp(-z[~small]))
    out[positive] = tmp
    return out


def generate_correlation_matrix(n: int = 200, seed: int = 20261030) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError("n must be at least two")
    eig_rng = np.random.default_rng(seed)
    eigenvalues = eig_rng.exponential(size=n)
    eigenvalues = n * eigenvalues / eigenvalues.sum()
    corr_rng = np.random.default_rng(seed + 1)
    sigma = random_correlation.rvs(eigenvalues, random_state=corr_rng)
    sigma = np.asarray(sigma, dtype=np.float64)
    sigma = (sigma + sigma.T) / 2.0
    np.fill_diagonal(sigma, 1.0)
    if not np.allclose(sigma, sigma.T, atol=1e-10):
        raise RuntimeError("generated correlation matrix is not symmetric")
    if not np.allclose(np.diag(sigma), 1.0, atol=1e-10):
        raise RuntimeError("generated correlation matrix has non-unit diagonal")
    if np.linalg.eigvalsh(sigma).min() < -1e-8:
        raise RuntimeError("generated correlation matrix is not positive semidefinite")
    return sigma, eigenvalues


def correlation_digest(sigma: np.ndarray) -> str:
    canonical = np.asarray(sigma, dtype="<f8", order="C")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def e_bh_fdp_power(log_e: np.ndarray, null_mask: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized e-BH on the last axis; inputs may be (replicate,time,hypothesis)."""
    values = np.asarray(log_e, dtype=np.float64)
    nulls = np.asarray(null_mask, dtype=bool)
    if values.ndim < 1 or values.shape[-1] != nulls.size or nulls.size == 0:
        raise ValueError("last log_e dimension must equal nonempty null_mask")
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError("log e-values may be finite or -inf, but not NaN/+inf")
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    k = nulls.size
    order = np.argsort(-values, axis=-1, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=-1)
    ranks = np.arange(1, k + 1, dtype=np.float64)
    thresholds = np.log(k / (alpha * ranks))
    qualifies = sorted_values >= thresholds
    k_star = np.max(np.where(qualifies, ranks, 0.0), axis=-1).astype(np.int64)
    sorted_nulls = nulls[order]
    rejected = ranks <= k_star[..., None]
    false_discoveries = np.sum(rejected & sorted_nulls, axis=-1)
    total_rejections = k_star
    fdp = false_discoveries / np.maximum(total_rejections, 1)
    n_alternatives = int((~nulls).sum())
    if n_alternatives:
        true_discoveries = np.sum(rejected & (~sorted_nulls), axis=-1)
        power = true_discoveries / n_alternatives
    else:
        power = np.zeros_like(fdp)
    return fdp, power, total_rejections


def run_dependent_normal_benchmark(
    *, repetitions: int = 250, timepoints: int = 2000, n_hypotheses: int = 200,
    n_true: int = 100, mu_null: float = 0.0, mu_alt: float = 0.1,
    alpha: float = 0.05, correlation_seed: int = 20261030,
    data_seed: int = 20261031, batch_size: int = 8
) -> dict:
    if repetitions < 1 or timepoints < 1 or n_hypotheses < 2:
        raise ValueError("repetitions/timepoints must be positive and hypotheses >=2")
    if not 0 < n_true < n_hypotheses:
        raise ValueError("n_true must be between zero and n_hypotheses")
    if not (math.isfinite(alpha) and 0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    n_alt = n_hypotheses - n_true
    sigma, eigenvalues = generate_correlation_matrix(n_hypotheses, correlation_seed)
    chol = np.linalg.cholesky(sigma)
    null_mask = np.zeros(n_hypotheses, dtype=bool)
    null_mask[:n_true] = True
    means = np.full(n_hypotheses, mu_alt, dtype=np.float64)
    means[:n_true] = mu_null
    # For N(mu_null,1) vs N(mu_alt,1), log likelihood-ratio increment is
    # (mu_alt-mu_null)*X - (mu_alt^2-mu_null^2)/2.
    coef = mu_alt - mu_null
    offset = -0.5 * (mu_alt * mu_alt - mu_null * mu_null)
    rng = np.random.default_rng(data_seed)
    methods = ("reference_script_single_step_ebh", "current_cumulative_ebh",
               "naive_running_max_ebh", "adjusted_running_max_a1", "adjusted_running_max_a2")
    sum_fdp = {m: np.zeros(timepoints, dtype=np.float64) for m in methods}
    sum_sup_fdp = {m: np.zeros(timepoints, dtype=np.float64) for m in methods}
    sum_power = {m: np.zeros(timepoints, dtype=np.float64) for m in methods}
    sum_rejections = {m: np.zeros(timepoints, dtype=np.float64) for m in methods}
    moments = {m: {metric: [0.0, 0.0] for metric in ("final_fdp", "sup_fdp", "final_power")}
               for m in methods}
    processed = 0
    started = time.perf_counter()
    while processed < repetitions:
        b = min(batch_size, repetitions - processed)
        z = rng.standard_normal((b, timepoints, n_hypotheses))
        # Correlate hypotheses while retaining independent time increments.
        x = z @ chol.T + means
        log_increment = coef * x + offset
        log_e = np.cumsum(log_increment, axis=1)
        log_m = np.maximum.accumulate(log_e, axis=1)
        # Paper adjusters are defined for M>=1. Include baseline 1 at all times.
        log_m = np.maximum(log_m, 0.0)
        adjusted_a1 = log_adjuster_a1_from_log_m(log_m)
        adjusted_a2 = log_adjuster_a2_from_log_m(log_m)
        method_values = {
            # The linked simulation script feeds one-step likelihood ratios
            # E[:, t] here. It is not the current cumulative e-process.
            "reference_script_single_step_ebh": log_increment,
            "current_cumulative_ebh": log_e,
            "naive_running_max_ebh": log_m,
            "adjusted_running_max_a1": adjusted_a1,
            "adjusted_running_max_a2": adjusted_a2,
        }
        for method, values in method_values.items():
            fdp, power, n_rej = e_bh_fdp_power(values, null_mask, alpha)
            sum_fdp[method] += fdp.sum(axis=0)
            sum_sup_fdp[method] += np.maximum.accumulate(fdp, axis=1).sum(axis=0)
            sum_power[method] += power.sum(axis=0)
            sum_rejections[method] += n_rej.sum(axis=0)
            replicate_sup_fdp = np.max(fdp, axis=1)
            for metric, vals in (
                ("final_fdp", fdp[:, -1]),
                ("sup_fdp", replicate_sup_fdp),
                ("final_power", power[:, -1]),
            ):
                moments[method][metric][0] += float(vals.sum())
                moments[method][metric][1] += float(np.square(vals).sum())
        processed += b
    elapsed = time.perf_counter() - started
    summary = {}
    for method in methods:
        fdr_curve = sum_fdp[method] / repetitions
        sup_fdr_curve = sum_sup_fdp[method] / repetitions
        power_curve = sum_power[method] / repetitions
        rejection_curve = sum_rejections[method] / repetitions

        def mc_stats(pair):
            total, total_sq = pair
            if repetitions < 2:
                return {"se": None, "ci95": None}
            mean = total / repetitions
            variance = max(0.0, (total_sq - repetitions * mean * mean) / (repetitions - 1))
            se = math.sqrt(variance / repetitions)
            return {"se": se, "ci95": [max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se)]}

        summary[method] = {
            "final_fdr": float(fdr_curve[-1]),
            "final_fdr_mc": mc_stats(moments[method]["final_fdp"]),
            "max_fdr_over_time": float(fdr_curve.max()),
            "final_sup_fdr": float(sup_fdr_curve[-1]),
            "final_sup_fdr_mc": mc_stats(moments[method]["sup_fdp"]),
            "max_sup_fdr_over_time": float(sup_fdr_curve.max()),
            "final_power": float(power_curve[-1]),
            "final_power_mc": mc_stats(moments[method]["final_power"]),
            "mean_power_over_time": float(power_curve.mean()),
            "final_mean_rejections": float(rejection_curve[-1]),
            "final_fdr_curve": fdr_curve.tolist(),
            "final_sup_fdr_curve": sup_fdr_curve.tolist(),
            "final_power_curve": power_curve.tolist(),
        }
    sigma_sha = correlation_digest(sigma)
    return {
        "schema": SCHEMA,
        "status": "INDEPENDENT_CORRECTED_DEPENDENT_NORMAL_PILOT_NOT_FULL_PAPER_REPRODUCTION",
        "protocol": {
            "repetitions": repetitions, "timepoints": timepoints,
            "n_hypotheses": n_hypotheses, "n_true_nulls": n_true,
            "n_alternatives": n_alt, "mu_null": mu_null, "mu_alternative": mu_alt,
            "marginal_variance": 1.0, "alpha": alpha,
            "correlation_seed": correlation_seed, "data_seed": data_seed,
            "batch_size": batch_size,
            "correlation_generator": "scipy.stats.random_correlation.rvs",
            "eigenvalue_distribution": "iid Exp(1), normalized to sum n_hypotheses",
            "eigenvalues": eigenvalues.tolist(),
            "correlation_matrix_sha256_little_endian_f64": sigma_sha,
            "adjuster_domain": "M_t=max(1,sup_{s<=t} E_s)",
            "log_bayes_factor_increment": "(mu_alt-mu_null)*X - (mu_alt^2-mu_null^2)/2",
            "fdr_definition": "mean across replicates of FDP at time t",
            "sup_fdr_definition": "mean across replicates of max_{s<=t} FDP_s",
            "power_definition": "mean proportion of alternatives rejected at time t",
        },
        "methods": summary,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "runtime_seconds": elapsed,
        "limitations": [
            "reference_script_single_step_ebh reproduces the linked script E[:,t] call; current_cumulative_ebh is the mathematically intended current e-process comparator",
            "lower Monte Carlo repetition count than the paper's described 10000; this is a local pilot",
            "one deterministic random-correlation matrix; dependence uncertainty is not explored",
            "normal likelihood-ratio e-process assumptions are exact only for the specified Gaussian marginal model",
            "no mixed-null proportions, alternative effect-size sweep or multi-matrix assessment",
            "no confidence intervals for FDR/power curves in this pilot",
            "no comparison against an independently executed official implementation",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--timepoints", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    result = run_dependent_normal_benchmark(repetitions=args.repetitions,
        timepoints=args.timepoints, data_seed=args.seed, batch_size=args.batch_size)
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({
        "output": str(args.output), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "runtime_seconds": result["runtime_seconds"],
        "correlation_matrix_sha256": result["protocol"]["correlation_matrix_sha256_little_endian_f64"],
        "methods": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_curve")} for k,v in result["methods"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
