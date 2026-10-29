#!/usr/bin/env python3
"""Uniform versus adaptive audit allocation under anytime-valid cellwise confidence sequences.

Synthetic Bernoulli risk-cell benchmark. Each cell has an independent infinite stream;
cell selection is predictable from past observations. Beta-mixture confidence sequences
are time-uniform per cell, so union-bounding over the fixed finite cell family supports
adaptive allocation. This does not cover biased audits, missing support, or model changes.
"""
from __future__ import annotations
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import numpy as np

SCHEMA = "symthaea-decision-risk-adaptive-audit-allocation-v1"
SEED = 20261029
DELTA = 0.10
RISK_BUDGET = 0.20
DECISION_RADIUS = 0.05
EPSILON_ALLOWANCE = RISK_BUDGET / 2.0 - DECISION_RADIUS
BATCH = 128
PILOT = 128
MAX_PER_CELL = 8192
N_REPEATS = 200
BISECTIONS = 36
TOL = 1e-12
BETA_A = 0.5
BETA_B = 0.5


def beta_mixture_log_evalue(mu: float, successes: int, n: int, a: float = BETA_A, b: float = BETA_B) -> float:
    if not 0.0 <= mu <= 1.0 or not 0 <= successes <= n or n < 1 or a <= 0 or b <= 0:
        raise ValueError("invalid beta-mixture inputs")
    failures = n - successes
    log_marginal = (math.lgamma(successes + a) + math.lgamma(failures + b)
                    - math.lgamma(n + a + b) - math.lgamma(a) - math.lgamma(b) + math.lgamma(a + b))
    if mu == 0.0:
        return log_marginal if successes == 0 else math.inf
    if mu == 1.0:
        return log_marginal if failures == 0 else math.inf
    return log_marginal - successes * math.log(mu) - failures * math.log1p(-mu)


def beta_mixture_cs(successes: int, n: int, alpha_cell: float) -> tuple[float, float]:
    """Two-sided time-uniform CS by beta-mixture e-process inversion + Ville's inequality."""
    if n < 1 or not 0 <= successes <= n or not (0 < alpha_cell < 1):
        raise ValueError("invalid CS inputs")
    threshold = math.log(1.0 / alpha_cell)
    failures = n - successes
    # Precompute the beta-binomial mixture constant once: the original form that
    # recomputed lgamma at every bisection iteration was needlessly expensive.
    log_marginal = (math.lgamma(successes + BETA_A) + math.lgamma(failures + BETA_B)
                    - math.lgamma(n + BETA_A + BETA_B) - math.lgamma(BETA_A)
                    - math.lgamma(BETA_B) + math.lgamma(BETA_A + BETA_B))
    phat = successes / n
    def log_evalue(mu: float) -> float:
        if mu <= 0.0:
            return log_marginal if successes == 0 else math.inf
        if mu >= 1.0:
            return log_marginal if failures == 0 else math.inf
        return log_marginal - successes * math.log(mu) - failures * math.log1p(-mu)
    if successes == 0:
        lower = 0.0
    else:
        outside, inside = 0.0, phat
        for _ in range(BISECTIONS):
            mid = (outside + inside) / 2.0
            if log_evalue(mid) >= threshold:
                outside = mid
            else:
                inside = mid
        lower = inside
    if successes == n:
        upper = 1.0
    else:
        inside, outside = phat, 1.0
        for _ in range(BISECTIONS):
            mid = (inside + outside) / 2.0
            if log_evalue(mid) < threshold:
                inside = mid
            else:
                outside = mid
        upper = inside
    return max(0.0, lower), min(1.0, upper)


def cell_bounds(model: np.ndarray, counts: np.ndarray, ns: np.ndarray, alpha_cell: float):
    lowers = np.zeros(len(model), dtype=float)
    uppers = np.ones(len(model), dtype=float)
    err_upper = np.ones(len(model), dtype=float) * np.inf
    err_lower = np.zeros(len(model), dtype=float)
    for i, n in enumerate(ns):
        if n == 0:
            continue
        lo, hi = beta_mixture_cs(int(counts[i]), int(n), alpha_cell)
        lowers[i], uppers[i] = lo, hi
        err_upper[i] = max(abs(float(model[i]) - lo), abs(float(model[i]) - hi))
        err_lower[i] = max(0.0, lo - float(model[i]), float(model[i]) - hi)
    return lowers, uppers, err_lower, err_upper


def classify(err_lower: np.ndarray, err_upper: np.ndarray) -> str | None:
    if float(np.max(err_upper)) <= EPSILON_ALLOWANCE + TOL:
        return "admit"
    if float(np.max(err_lower)) > EPSILON_ALLOWANCE + TOL:
        return "reject"
    return None


def run_strategy(streams: np.ndarray, model: np.ndarray, *, strategy: str, batch: int = BATCH) -> dict:
    """Run uniform or adaptive sampling over the same per-cell streams."""
    k = len(model)
    if strategy not in ("uniform", "adaptive"):
        raise ValueError("strategy must be uniform or adaptive")
    ns = np.zeros(k, dtype=np.int64)
    successes = np.zeros(k, dtype=np.int64)
    # Pilot every cell to make initial cell-wise confidence intervals available.
    for i in range(k):
        ns[i] = PILOT
        successes[i] = int(np.sum(streams[i, :PILOT]))
    alpha_cell = DELTA / k
    lo, hi, err_lo, err_hi = cell_bounds(model, successes, ns, alpha_cell)
    status = classify(err_lo, err_hi)
    total = int(np.sum(ns))
    if status:
        return {"status": status, "total_observations": total, "per_cell_observations": ns.tolist(), "final_error_upper": err_hi.tolist(), "final_error_lower": err_lo.tolist(), "looks": 1}
    looks = 1
    while int(np.max(ns)) + batch <= MAX_PER_CELL:
        if strategy == "uniform":
            chosen = np.arange(k)
        else:
            # Allocate to the cell with the largest current certified error upper bound.
            # Stable tie-break by index makes the policy deterministic.
            eligible = np.where(ns + batch <= MAX_PER_CELL)[0]
            if len(eligible) == 0:
                break
            chosen = np.array([int(eligible[np.argmax(err_hi[eligible])])])
        for i in chosen:
            start, stop = int(ns[i]), int(ns[i] + batch)
            successes[i] += int(np.sum(streams[i, start:stop]))
            ns[i] += batch
        total += int(len(chosen) * batch)
        looks += 1
        lo, hi, err_lo, err_hi = cell_bounds(model, successes, ns, alpha_cell)
        status = classify(err_lo, err_hi)
        if status:
            break
    if status is None:
        status = "refuse_unresolved"
    return {"status": status, "total_observations": total, "per_cell_observations": ns.tolist(), "final_error_upper": [float(x) for x in err_hi], "final_error_lower": [float(x) for x in err_lo], "looks": looks}


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / trials
    den = 1 + z*z/trials
    center = (p + z*z/(2*trials)) / den
    half = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / den
    return [max(0.0, center-half), min(1.0, center+half)]


def run(*, repeats: int = N_REPEATS) -> dict:
    if repeats < 50:
        raise ValueError("use at least 50 repeats")
    # Heterogeneous variance is the reason adaptive allocation might help.
    truth = np.array([0.015, 0.035, 0.075, 0.14, 0.27, 0.50, 0.73, 0.94], dtype=float)
    cell_ids = [f"policy_{i // 2}::world_{i % 2}" for i in range(len(truth))]
    models = {
        "correct_model": truth.copy(),
        "safe_offset_high_variance_cell": truth.copy(),
        "near_over_budget_high_variance_cell": truth.copy(),
        "clearly_over_budget_high_variance_cell": truth.copy(),
    }
    models["safe_offset_high_variance_cell"][5] += 0.02
    models["near_over_budget_high_variance_cell"][5] += EPSILON_ALLOWANCE + 0.001
    models["clearly_over_budget_high_variance_cell"][5] += EPSILON_ALLOWANCE + 0.02
    true_errors = {name: float(np.max(np.abs(m - truth))) for name, m in models.items()}
    rng = np.random.default_rng(SEED)
    collected = {name: {method: [] for method in ("uniform", "adaptive")} for name in models}
    for _ in range(repeats):
        streams = np.vstack([rng.binomial(1, p, size=MAX_PER_CELL) for p in truth]).astype(np.int8)
        for name, model in models.items():
            for method in ("uniform", "adaptive"):
                collected[name][method].append(run_strategy(streams, model, strategy=method))
    results = {}
    for name in models:
        results[name] = {}
        for method, rows in collected[name].items():
            adm = sum(row["status"] == "admit" for row in rows)
            rej = sum(row["status"] == "reject" for row in rows)
            unsafe = sum(row["status"] == "admit" and true_errors[name] > EPSILON_ALLOWANCE + TOL for row in rows)
            status_counts = {status: sum(row["status"] == status for row in rows) for status in ("admit", "reject", "refuse_unresolved")}
            total_obs = np.array([row["total_observations"] for row in rows], dtype=float)
            per_cell = np.array([row["per_cell_observations"] for row in rows], dtype=float)
            results[name][method] = {
                "true_max_error_scoring_only": true_errors[name],
                "status_counts": status_counts,
                "admission_rate": adm / repeats,
                "false_admission_count": unsafe,
                "false_admission_rate": unsafe / repeats,
                "false_admission_rate_wilson95": wilson_interval(unsafe, repeats),
                "mean_total_observations": float(total_obs.mean()),
                "median_total_observations": float(np.median(total_obs)),
                "p95_total_observations": float(np.quantile(total_obs, 0.95)),
                "mean_observations_by_cell": [float(x) for x in per_cell.mean(axis=0)],
                "mean_number_of_looks": float(np.mean([row["looks"] for row in rows])),
                "mean_final_error_upper_by_cell": [float(np.mean([row["final_error_upper"][i] for row in rows])) for i in range(len(truth))],
            }
    clean_uniform = results["correct_model"]["uniform"]["mean_total_observations"]
    clean_adaptive = results["correct_model"]["adaptive"]["mean_total_observations"]
    return {
        "schema": SCHEMA,
        "status": "SYNTHETIC_ADAPTIVE_ALLOCATION_BENCHMARK_NOT_PRODUCTION_CERTIFICATION",
        "protocol": {
            "seed": SEED, "repeats": repeats, "cells": len(truth), "cell_ids": cell_ids,
            "true_bernoulli_means": [float(x) for x in truth],
            "loss_range": [0.0, 1.0], "familywise_delta": DELTA, "per_cell_delta": DELTA / len(truth),
            "decision_risk_budget": RISK_BUDGET, "decision_radius": DECISION_RADIUS,
            "maximum_admissible_error": EPSILON_ALLOWANCE, "pilot_per_cell": PILOT,
            "batch_size": BATCH, "max_observations_per_cell": MAX_PER_CELL,
            "allocation": "uniform samples every cell per look; adaptive allocates next batch to cell with largest current confidence-sequence error upper bound; equal independent streams and same stopping classifications",
            "methods_are_anytime_valid": "Both use per-cell beta-mixture confidence sequences at alpha=delta/K; each cell stream is iid Bernoulli and selection is based only on past data. This assumption does not cover shared audit bias or model/support changes.",
        },
        "results": results,
        "clean_model_mean_observation_reduction_percent_adaptive_vs_uniform": 100.0 * (1.0 - clean_adaptive / clean_uniform),
        "limitations": [
            "synthetic 8-cell Bernoulli model with heterogeneous means; not a real-world benchmark",
            "true means used only to generate audit streams and score the unsafe-admission event",
            "per-cell iid and fixed-support assumptions are required; the separate audit-integrity preflight remains mandatory",
            "adaptive policy is a simple maximum-upper-bound heuristic, not a proven optimal allocation algorithm",
            "no published adaptive allocation or confidence-sequence package has been executed as a baseline",
            "Python prototype only, not integrated into the Rust planner or dispatch gate",
        ],
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    args = parser.parse_args()
    start = time.perf_counter()
    payload = run(repeats=args.repeats)
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({"path": str(args.output), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "runtime_seconds": round(time.perf_counter() - start, 3)}, indent=2))

if __name__ == "__main__":
    main()
