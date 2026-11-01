#!/usr/bin/env python3
"""Monte Carlo comparison for shared-control joint vs affine e-processes.

Optional dependency: NumPy. The null generator uses a Gaussian vector with
unit marginal variance and equicorrelation rho. Random arrivals are nested
and depend only on the preceding update's observations.

Methods:
- joint_nominal: history-dependent intersection betting using a declared
  covariance upper bound rho_bound_nominal;
- joint_robust: same betting policy with a conservative covariance upper
  bound rho_bound_robust;
- affine_global: fixed globally summable coordinate weights and valid
  one-coordinate Gaussian e-processes, requiring no cross-coordinate model.

This is a component e-process crossing benchmark, not a closed-FDR/SupFDR
simulation. It reports first-crossing probability under the all-null and
power/time-to-crossing under a common positive-mean alternative.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import time
from typing import Any, Iterable


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _logsumexp_rows(log_terms):
    import numpy as np

    maximum = np.max(log_terms, axis=1)
    return maximum + np.log(np.exp(log_terms - maximum[:, None]).sum(axis=1))


def _simulate_one(
    *,
    replicates: int,
    horizon: int,
    n_hypotheses: int,
    rho_actual: float,
    mean: float,
    alpha: float,
    seed: int,
    rho_bound_nominal: float,
    rho_bound_robust: float,
    arrival_threshold: float,
    coordinate_lambda: float,
    joint_base_lambda: float,
    joint_gain: float,
    joint_max_lambda: float,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError("NumPy is required for this simulation benchmark") from error

    for name, value in (
        ("rho_actual", rho_actual),
        ("rho_bound_nominal", rho_bound_nominal),
        ("rho_bound_robust", rho_bound_robust),
    ):
        if not math.isfinite(value) or not (-1 / max(1, n_hypotheses - 1) < value < 1):
            raise ValueError(f"{name} is outside the positive-definite equicorrelation range")
    if rho_bound_nominal < rho_actual:
        nominal_model_valid = False
    else:
        nominal_model_valid = True
    if rho_bound_robust < rho_actual:
        robust_model_valid = False
    else:
        robust_model_valid = True

    rng = np.random.default_rng(seed)
    active_count = np.ones(replicates, dtype=np.int32)
    observations = np.full((replicates, horizon, n_hypotheses), np.nan, dtype=np.float64)
    log_joint_nominal = np.zeros(replicates, dtype=np.float64)
    log_joint_robust = np.zeros(replicates, dtype=np.float64)
    log_coordinate = np.zeros((replicates, n_hypotheses), dtype=np.float64)
    crossing_times = {
        "joint_nominal": np.full(replicates, horizon + 1, dtype=np.int32),
        "joint_robust": np.full(replicates, horizon + 1, dtype=np.int32),
        "affine_global": np.full(replicates, horizon + 1, dtype=np.int32),
    }
    weights = np.asarray([2.0 ** (-(i + 1)) for i in range(n_hypotheses)])
    log_weights = np.log(weights)
    log_threshold = math.log(1.0 / alpha)
    active_counts = np.zeros((replicates, horizon), dtype=np.int32)

    for t in range(horizon):
        if t > 0:
            previous = observations[:, t - 1, :]
            previous_sum = np.nansum(previous, axis=1)
            previous_count = np.sum(np.isfinite(previous), axis=1)
            previous_mean = previous_sum / np.maximum(previous_count, 1)
            add_next = (active_count < n_hypotheses) & (previous_mean > arrival_threshold)
            active_count += add_next.astype(np.int32)

        active_counts[:, t] = active_count
        z_current = np.zeros((replicates, n_hypotheses), dtype=np.float64)

        for m in np.unique(active_count):
            idx = np.flatnonzero(active_count == m)
            covariance = np.full((m, m), rho_actual, dtype=np.float64)
            np.fill_diagonal(covariance, 1.0)
            draw = rng.multivariate_normal(
                mean=np.full(m, mean, dtype=np.float64),
                cov=covariance,
                size=len(idx),
            )
            z_current[np.ix_(idx, np.arange(m))] = draw

        # Predictable intersection-specific lambda: only past values for
        # coordinates active now. Current z_current is not read until after.
        lambdas = np.full(replicates, joint_base_lambda, dtype=np.float64)
        for r in range(replicates):
            m = int(active_count[r])
            past = observations[r, :t, :m]
            finite = past[np.isfinite(past)]
            if finite.size:
                historical_mean = float(finite.mean())
                lambdas[r] = min(
                    joint_max_lambda,
                    joint_base_lambda + joint_gain * max(0.0, historical_mean),
                )

        for m in np.unique(active_count):
            idx = np.flatnonzero(active_count == m)
            zsum = z_current[idx, :m].sum(axis=1)
            variance_nominal = m + m * (m - 1) * rho_bound_nominal
            variance_robust = m + m * (m - 1) * rho_bound_robust
            lam = lambdas[idx]
            log_joint_nominal[idx] += lam * zsum - 0.5 * lam * lam * variance_nominal
            log_joint_robust[idx] += lam * zsum - 0.5 * lam * lam * variance_robust

        # Arbitrary-dependence coordinate e-process baseline. Each coordinate
        # has unit marginal variance; its fixed positive lambda is valid under
        # its one-sided Gaussian null, regardless of cross-coordinate rho.
        for m in np.unique(active_count):
            idx = np.flatnonzero(active_count == m)
            log_coordinate[np.ix_(idx, np.arange(m))] += (
                coordinate_lambda * z_current[np.ix_(idx, np.arange(m))]
                - 0.5 * coordinate_lambda * coordinate_lambda
            )
        observations[:, t, :] = np.where(
            np.arange(n_hypotheses)[None, :] < active_count[:, None],
            z_current,
            np.nan,
        )

        # All methods are evaluated after the current increment.
        newly_crossed = (crossing_times["joint_nominal"] == horizon + 1) & (
            log_joint_nominal >= log_threshold
        )
        crossing_times["joint_nominal"][newly_crossed] = t + 1
        newly_crossed = (crossing_times["joint_robust"] == horizon + 1) & (
            log_joint_robust >= log_threshold
        )
        crossing_times["joint_robust"][newly_crossed] = t + 1

        log_affine_values = np.empty(replicates, dtype=np.float64)
        for m in np.unique(active_count):
            idx = np.flatnonzero(active_count == m)
            active_weights = weights[:m]
            cash = 1.0 - float(active_weights.sum())
            terms = log_weights[:m][None, :] + log_coordinate[np.ix_(idx, np.arange(m))]
            if cash > 0:
                cash_column = np.full((len(idx), 1), math.log(cash))
                terms = np.concatenate([cash_column, terms], axis=1)
            log_affine_values[idx] = _logsumexp_rows(terms)
        newly_crossed = (crossing_times["affine_global"] == horizon + 1) & (
            log_affine_values >= log_threshold
        )
        crossing_times["affine_global"][newly_crossed] = t + 1

    rows = {}
    for method, times in crossing_times.items():
        crossed = times <= horizon
        count = int(crossed.sum())
        lo, hi = _wilson(count, replicates)
        rows[method] = {
            "crossings": count,
            "crossing_rate": count / replicates,
            "crossing_rate_wilson_95": [lo, hi],
            "median_crossing_time_given_crossing": (
                float(np.median(times[crossed])) if count else None
            ),
            "mean_crossing_time_censored_at_horizon": float(
                np.minimum(times, horizon + 1).mean()
            ),
            "mean_final_log_e_value": float(
                {
                    "joint_nominal": log_joint_nominal,
                    "joint_robust": log_joint_robust,
                    "affine_global": log_affine_values,
                }[method].mean()
            ),
        }

    return {
        "replicates": replicates,
        "horizon": horizon,
        "n_hypotheses": n_hypotheses,
        "actual_rho": rho_actual,
        "common_mean": mean,
        "alpha": alpha,
        "seed": seed,
        "nominal_joint_covariance_bound_valid": nominal_model_valid,
        "robust_joint_covariance_bound_valid": robust_model_valid,
        "mean_final_active_count": float(active_counts[:, -1].mean()),
        "methods": rows,
        "warning": (
            "component e-process crossing benchmark only; not a closed-FDR or "
            "SupFDR comparison and not a proof of conditional model correctness"
        ),
    }


def run_benchmark(
    *,
    replicates: int = 5000,
    horizon: int = 60,
    n_hypotheses: int = 5,
    correlations: Iterable[float] = (0.2, 0.5, 0.8),
    alternative_mean: float = 0.2,
    alpha: float = 0.05,
    seed: int = 20261101,
) -> dict[str, Any]:
    correlations = tuple(float(rho) for rho in correlations)
    scenarios = []
    started = time.perf_counter()
    scenario_id = 0
    for rho in correlations:
        for mean in (0.0, alternative_mean):
            scenarios.append(_simulate_one(
                replicates=replicates,
                horizon=horizon,
                n_hypotheses=n_hypotheses,
                rho_actual=float(rho),
                mean=float(mean),
                alpha=alpha,
                seed=seed + scenario_id,
                rho_bound_nominal=0.5,
                rho_bound_robust=0.8,
                arrival_threshold=0.0,
                coordinate_lambda=0.2,
                joint_base_lambda=0.05,
                joint_gain=0.10,
                joint_max_lambda=0.5,
            ))
            scenario_id += 1
    return {
        "schema": "history-dependent-joint-vs-affine-component-benchmark-v1",
        "purpose": (
            "compare model-specific joint intersection betting with arbitrary-"
            "dependence global affine coordinate mixing; not an FDR/SupFDR claim"
        ),
        "environment": {
            "python": platform.python_version(),
        },
        "configuration": {
            "replicates": replicates,
            "horizon": horizon,
            "n_hypotheses": n_hypotheses,
            "correlations": list(correlations),
            "alternative_mean": alternative_mean,
            "alpha": alpha,
            "base_seed": seed,
            "arrival_rule": "activate next coordinate when previous-update active-coordinate mean > 0",
            "joint_nominal_rho_bound": 0.5,
            "joint_robust_rho_bound": 0.8,
            "coordinate_lambda": 0.2,
            "joint_lambda": "clip(0.05 + 0.10 * positive historical mean, 0, 0.5)",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--n-hypotheses", type=int, default=5)
    parser.add_argument("--alternative-mean", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--output", default="-")
    args = parser.parse_args()
    result = run_benchmark(
        replicates=args.replicates,
        horizon=args.horizon,
        n_hypotheses=args.n_hypotheses,
        alternative_mean=args.alternative_mean,
        alpha=args.alpha,
        seed=args.seed,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output == "-":
        print(rendered)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
