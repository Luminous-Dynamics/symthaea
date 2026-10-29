#!/usr/bin/env python3
"""Decision-frontier deficit allocation under anytime-valid risk-cell confidence sequences.

The scheduler estimates the number of observations needed to resolve each active
constraint against its nearest admissible boundary using Bernoulli KL divergence.
If observed means suggest at least one violation, it spends on the cheapest estimated
rejection candidate; otherwise it works on the hardest unresolved admission blocker.
The scheduler changes sampling allocation only; certification still uses the same
per-cell beta-mixture confidence sequences and familywise delta as the baselines.
Synthetic iid Bernoulli evidence only; not a production certificate.
"""
from __future__ import annotations
import hashlib, json, math, platform, time
from pathlib import Path
import numpy as np
import decision_risk_adaptive_audit_allocation as base
import decision_risk_constraint_aware_allocation as prior

SCHEMA = "symthaea-decision-risk-frontier-deficit-allocation-v1"
SEED = 20261029
DELTA = base.DELTA
EPS = base.EPSILON_ALLOWANCE
PILOT = base.PILOT
BATCH = base.BATCH
MAX_PER_CELL = base.MAX_PER_CELL
TOL = 1e-12


def bernoulli_kl(p: float, q: float) -> float:
    """KL(Bernoulli(p) || Bernoulli(q)); guarded at support boundaries."""
    p = min(1.0, max(0.0, float(p)))
    q = min(1.0, max(0.0, float(q)))
    if p == q:
        return 0.0
    if (q == 0.0 and p > 0.0) or (q == 1.0 and p < 1.0):
        return math.inf
    terms = 0.0
    if p > 0.0:
        terms += p * math.log(p / q)
    if p < 1.0:
        terms += (1.0 - p) * math.log((1.0 - p) / (1.0 - q))
    return max(0.0, terms)


def estimated_decision_samples(phat: float, model_value: float, n: int, alpha_cell: float) -> float:
    """Approximate total n needed to separate phat from the nearest error boundary.

    This is a scheduling proxy, not a confidence bound. It must never control the
    admission/rejection decision; the anytime-valid CS remains the sole gate.
    """
    lower = max(0.0, model_value - EPS)
    upper = min(1.0, model_value + EPS)
    if phat < lower:
        boundary = lower
    elif phat > upper:
        boundary = upper
    else:
        boundary = lower if (phat - lower) <= (upper - phat) else upper
    divergence = bernoulli_kl(phat, boundary)
    if not math.isfinite(divergence):
        return float(max(1, n))
    # log(1/alpha) is a transparent first-order information target. The cap only
    # stabilizes scheduling near the boundary; it cannot influence certification.
    return min(float(MAX_PER_CELL * 8), math.log(1.0 / alpha_cell) / max(divergence, 1e-12))


def run_strategy(streams: np.ndarray, model: np.ndarray, strategy: str) -> dict:
    if strategy not in ("uniform", "max_error_upper", "widest_active_ci", "marginal_ci_reduction", "frontier_deficit"):
        raise ValueError("unknown allocation strategy")
    k = len(model)
    ns = np.full(k, PILOT, dtype=np.int64)
    successes = np.array([int(np.sum(streams[i, :PILOT])) for i in range(k)], dtype=np.int64)
    alpha_cell = DELTA / k

    def bounds():
        lows = np.zeros(k); highs = np.ones(k)
        err_lo = np.zeros(k); err_hi = np.full(k, np.inf); widths = np.ones(k)
        phat = np.divide(successes, ns, dtype=float)
        for i, n in enumerate(ns):
            lo, hi = base.beta_mixture_cs(int(successes[i]), int(n), alpha_cell)
            lows[i], highs[i] = lo, hi
            err_hi[i] = max(abs(float(model[i]) - lo), abs(float(model[i]) - hi))
            err_lo[i] = max(0.0, lo - float(model[i]), float(model[i]) - hi)
            widths[i] = hi - lo
        return lows, highs, err_lo, err_hi, widths, phat

    lows, highs, err_lo, err_hi, widths, phat = bounds()
    status = base.classify(err_lo, err_hi)
    total = int(ns.sum()); looks = 1
    if status:
        return {"status": status, "total_observations": total, "per_cell_observations": ns.tolist(), "looks": looks}
    while True:
        eligible = np.where(ns + BATCH <= MAX_PER_CELL)[0]
        if not len(eligible):
            break
        capped_blocker = (ns >= MAX_PER_CELL) & (err_hi > EPS + TOL) & (err_lo <= EPS + TOL)
        if np.any(capped_blocker):
            break
        blockers = eligible[err_hi[eligible] > EPS + TOL]
        if strategy == "uniform":
            if np.any(ns[eligible] != ns.max()) or len(eligible) != k:
                break
            chosen = np.arange(k)
        elif not len(blockers):
            break
        elif strategy == "max_error_upper":
            chosen = np.array([int(blockers[np.argmax(err_hi[blockers])])])
        elif strategy == "widest_active_ci":
            chosen = np.array([int(blockers[np.argmax(widths[blockers])])])
        elif strategy == "marginal_ci_reduction":
            chosen = np.array([int(blockers[np.argmax(widths[blockers] / ns[blockers])])])
        else:
            # A point estimate outside the admissible band is a candidate for a
            # global rejection; choose the one with lowest estimated remaining
            # sample cost. When all point estimates are inside, admission requires
            # every blocker resolved, so attack the largest estimated deficit.
            outward = blockers[np.abs(phat[blockers] - model[blockers]) > EPS + TOL]
            target_n = np.array([estimated_decision_samples(float(phat[i]), float(model[i]), int(ns[i]), alpha_cell) for i in blockers])
            deficits = np.maximum(1.0, target_n - ns[blockers])
            if len(outward):
                # Identify corresponding blocker-local indices, then choose the
                # estimated cheapest-to-reject candidate; ties break by index.
                local = np.array([j for j, idx in enumerate(blockers) if idx in set(outward.tolist())], dtype=int)
                pick_local = local[int(np.argmin(deficits[local]))]
            else:
                pick_local = int(np.argmax(deficits))
            chosen = np.array([int(blockers[pick_local])])
        for i in chosen:
            start, stop = int(ns[i]), int(ns[i] + BATCH)
            successes[i] += int(np.sum(streams[i, start:stop]))
            ns[i] += BATCH
        total += int(len(chosen) * BATCH); looks += 1
        lows, highs, err_lo, err_hi, widths, phat = bounds()
        status = base.classify(err_lo, err_hi)
        if status:
            break
    if status is None:
        status = "refuse_unresolved"
    return {"status": status, "total_observations": total, "per_cell_observations": ns.tolist(), "looks": looks}


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / trials
    den = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / den
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def run(repeats: int = 100) -> dict:
    if repeats < 50:
        raise ValueError("use at least 50 repeats")
    truth = np.array([0.015, 0.035, 0.075, 0.14, 0.27, 0.50, 0.73, 0.94], dtype=float)
    models = {"correct_model": truth.copy(), "safe_offset_high_variance_cell": truth.copy(),
              "near_over_budget_high_variance_cell": truth.copy(), "clearly_over_budget_high_variance_cell": truth.copy()}
    models["safe_offset_high_variance_cell"][5] += 0.02
    models["near_over_budget_high_variance_cell"][5] += EPS + 0.001
    models["clearly_over_budget_high_variance_cell"][5] += EPS + 0.02
    true_err = {name: float(np.max(np.abs(m - truth))) for name, m in models.items()}
    methods = ("uniform", "max_error_upper", "widest_active_ci", "marginal_ci_reduction", "frontier_deficit")
    rows = {name: {method: [] for method in methods} for name in models}
    rng = np.random.default_rng(SEED)
    for _ in range(repeats):
        streams = np.vstack([rng.binomial(1, p, size=MAX_PER_CELL) for p in truth]).astype(np.int8)
        for name, model in models.items():
            for method in methods:
                rows[name][method].append(run_strategy(streams, model, method))
    results = {}
    for name in models:
        results[name] = {}
        for method, runs in rows[name].items():
            unsafe = sum(r["status"] == "admit" and true_err[name] > EPS + TOL for r in runs)
            false_rej = sum(r["status"] == "reject" and true_err[name] <= EPS + TOL for r in runs)
            obs = np.array([r["total_observations"] for r in runs], dtype=float)
            results[name][method] = {
                "true_max_error_for_scoring_only": true_err[name],
                "status_counts": {s: sum(r["status"] == s for r in runs) for s in ("admit", "reject", "refuse_unresolved")},
                "admission_rate": sum(r["status"] == "admit" for r in runs) / repeats,
                "false_admissions": unsafe, "false_admission_rate": unsafe / repeats,
                "false_admission_wilson95": wilson(unsafe, repeats),
                "false_rejections": false_rej, "false_rejection_rate": false_rej / repeats,
                "mean_total_observations": float(obs.mean()), "median_total_observations": float(np.median(obs)),
                "p95_total_observations": float(np.quantile(obs, .95)),
                "mean_per_cell_observations": [float(x) for x in np.mean([r["per_cell_observations"] for r in runs], axis=0)],
            }
    ref = results["correct_model"]["uniform"]["mean_total_observations"]
    for method in methods:
        results["correct_model"][method]["observations_saved_vs_uniform_percent"] = 100 * (1 - results["correct_model"][method]["mean_total_observations"] / ref)
    return {
        "schema": SCHEMA,
        "status": "SYNTHETIC_FRONTIER_DEFICIT_EXPERIMENT_NOT_PRODUCTION_CERTIFICATION",
        "protocol": {"seed": SEED, "repeats": repeats, "cells": len(truth), "truth_means_for_generating_streams": truth.tolist(),
            "loss_range": [0, 1], "familywise_delta": DELTA, "per_cell_delta": DELTA / len(truth), "epsilon_allowance": EPS,
            "pilot_per_cell": PILOT, "batch_size": BATCH, "max_per_cell": MAX_PER_CELL,
            "stream_reuse": "all five methods receive identical per-cell Bernoulli streams within each repeat and scenario",
            "frontier_deficit_rule": "KL sample-complexity proxy to nearest admissible error boundary; choose cheapest projected rejection if any blocker point-estimate is outside, else largest projected admission deficit; confidence-sequence classifier alone decides admit/reject",
            "methods": {"uniform": "sample all cells evenly", "max_error_upper": "largest active upper error bound", "widest_active_ci": "widest active confidence interval", "marginal_ci_reduction": "active CI width divided by samples", "frontier_deficit": "decision-aware KL target sample deficit"}},
        "results": results,
        "limitations": ["8-cell iid Bernoulli synthetic workload only", "KL target is an allocation heuristic, not a confidence bound or optimality proof", "same anytime-valid beta-mixture CS and fixed-support assumptions are used by all methods", "100 repeats are not enough to establish a low false-admission tail; 0/100 Wilson upper bound is about 3.7%", "does not cover shared audit bias, missing support, model shift, or identity mismatch", "not a direct published SOTA run; no Rust production integration"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=100)
    a = p.parse_args()
    start = time.perf_counter(); out = run(a.repeats)
    raw = (json.dumps(out, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_bytes(raw)
    print(json.dumps({"path": str(a.output), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "runtime_seconds": round(time.perf_counter()-start, 3)}, indent=2))
