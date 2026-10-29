#!/usr/bin/env python3
"""Sensitivity-adjusted anytime risk certificates under known Huber contamination caps.

This experiment uses iid Bernoulli observations from Q=(1-rho_actual)P+rho_actual R.
A beta-mixture confidence sequence covers the observed mixture mean q; a transparent
Huber set inversion maps that interval to a set for the clean target mean p. The
nominal comparator incorrectly treats q as p. In a multi-cell certificate, any
missing support, identity mismatch, or unknown contamination bound fails closed.

This is a Bernoulli sensitivity transformation, not an implementation of the general
Huber-robust confidence-sequence algorithm of Wang & Ramdas (AISTATS 2023).
"""
from __future__ import annotations
import hashlib, json, math, platform, time
from pathlib import Path
import numpy as np
import decision_risk_adaptive_audit_allocation as base

SCHEMA = "symthaea-decision-risk-huber-sensitivity-v1"
SEED = 20261029
DELTA = 0.10
EPS = 0.05
BATCH = 256
PILOT = 256
MAX_PER_CELL = 8192
TOL = 1e-12
RHOS = (0.0, 0.05, 0.10, 0.12, 0.18)


def clean_mean_interval(q_lower: float, q_upper: float, rho_upper: float) -> tuple[float, float]:
    """Invert Huber Bernoulli-mixture bounds for clean p from observed q CI.

    For q=(1-r)p+r*s, s in [0,1], and r <= rho_upper:
      p in [max(0,(q-rho)/(1-rho)), min(1,q/(1-rho))].
    Monotonicity gives the same map applied to a confidence set for q.
    """
    if not (0.0 <= q_lower <= q_upper <= 1.0):
        raise ValueError("invalid observed-mean confidence interval")
    if not (0.0 <= rho_upper < 1.0):
        raise ValueError("contamination upper bound must be in [0,1)")
    lo = max(0.0, (q_lower - rho_upper) / (1.0 - rho_upper))
    hi = min(1.0, q_upper / (1.0 - rho_upper))
    return lo, hi


def decision(model: np.ndarray, lowers: np.ndarray, uppers: np.ndarray) -> tuple[str | None, float, float]:
    err_lower = np.maximum(0.0, np.maximum(lowers - model, model - uppers))
    err_upper = np.maximum(np.abs(model - lowers), np.abs(model - uppers))
    max_lower = float(np.max(err_lower)); max_upper = float(np.max(err_upper))
    if max_upper <= EPS + TOL:
        return "admit", max_lower, max_upper
    if max_lower > EPS + TOL:
        return "reject", max_lower, max_upper
    return None, max_lower, max_upper


def preflight(*, support_covered: bool, identity_matches: bool, contamination_bound_known: bool) -> bool:
    """Certification requires all semantic/integrity assumptions to be explicit."""
    return support_covered and identity_matches and contamination_bound_known


def run_certifier(streams: np.ndarray, model: np.ndarray, *, mode: str, rho_upper: float | None) -> dict:
    if mode not in ("nominal", "huber_robust"):
        raise ValueError("mode must be nominal or huber_robust")
    if not preflight(support_covered=True, identity_matches=True,
                     contamination_bound_known=(mode == "nominal" or rho_upper is not None)):
        return {"status": "refuse_unresolved", "reason": "integrity_or_sensitivity_preflight_failed", "total_observations": 0,
                "per_cell_observations": [0] * len(model)}
    k = len(model)
    ns = np.full(k, PILOT, dtype=np.int64)
    successes = np.array([int(np.sum(streams[i, :PILOT])) for i in range(k)], dtype=np.int64)
    alpha_cell = DELTA / k

    def bounds():
        lo_arr = np.zeros(k); hi_arr = np.ones(k)
        for i, n in enumerate(ns):
            lo_q, hi_q = base.beta_mixture_cs(int(successes[i]), int(n), alpha_cell)
            if mode == "huber_robust":
                lo_arr[i], hi_arr[i] = clean_mean_interval(lo_q, hi_q, float(rho_upper))
            else:
                lo_arr[i], hi_arr[i] = lo_q, hi_q
        return lo_arr, hi_arr

    lo, hi = bounds()
    status, err_lo, err_hi = decision(model, lo, hi)
    while status is None and bool(np.all(ns + BATCH <= MAX_PER_CELL)):
        for i in range(k):
            start, stop = int(ns[i]), int(ns[i] + BATCH)
            successes[i] += int(np.sum(streams[i, start:stop]))
            ns[i] += BATCH
        lo, hi = bounds()
        status, err_lo, err_hi = decision(model, lo, hi)
    if status is None:
        status = "refuse_unresolved"
    return {"status": status, "total_observations": int(ns.sum()), "per_cell_observations": ns.tolist(),
            "final_error_lower": err_lo, "final_error_upper": err_hi}


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    p = successes / trials; den = 1 + z*z/trials
    center = (p + z*z/(2*trials))/den
    half = z*math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials))/den
    return [max(0.0, center-half), min(1.0, center+half)]


def run(repeats: int = 100) -> dict:
    if repeats < 50:
        raise ValueError("use at least 50 repeats")
    model_safe = np.array([0.15, 0.30, 0.60, 0.80], dtype=float)
    clean_safe = np.array([0.15, 0.32, 0.60, 0.80], dtype=float)
    model_unsafe = model_safe.copy()
    clean_unsafe = np.array([0.15, 0.37, 0.60, 0.80], dtype=float)
    scenarios = {"safe_0_02_error": (model_safe, clean_safe), "unsafe_0_07_error_masked": (model_unsafe, clean_unsafe)}
    rng = np.random.default_rng(SEED)
    samples: dict[str, dict[str, dict[str, list]]] = {}
    for scenario, (model, clean) in scenarios.items():
        samples[scenario] = {}
        for rho in RHOS:
            samples[scenario][f"rho_{rho:.2f}"] = {"nominal": [], "huber_robust": []}
            observed_means = clean.copy()
            observed_means[1] = (1.0-rho) * clean[1]
            for _ in range(repeats):
                streams = np.vstack([rng.binomial(1, q, size=MAX_PER_CELL) for q in observed_means]).astype(np.int8)
                for mode in ("nominal", "huber_robust"):
                    result = run_certifier(streams, model, mode=mode, rho_upper=(rho if mode == "huber_robust" else 0.0))
                    samples[scenario][f"rho_{rho:.2f}"][mode].append(result)

    results = {}
    for scenario, (model, clean) in scenarios.items():
        true_error = float(np.max(np.abs(model-clean)))
        results[scenario] = {"true_clean_model_max_error_for_scoring_only": true_error, "by_contamination": {}}
        unsafe_truth = true_error > EPS + TOL
        for rho in RHOS:
            label = f"rho_{rho:.2f}"
            results[scenario]["by_contamination"][label] = {"actual_contamination_fraction": rho, "observed_target_mean": float((1-rho)*clean[1]), "methods": {}}
            for mode in ("nominal", "huber_robust"):
                runs = samples[scenario][label][mode]
                admits = sum(x["status"] == "admit" for x in runs)
                rejects = sum(x["status"] == "reject" for x in runs)
                unresolved = sum(x["status"] == "refuse_unresolved" for x in runs)
                unsafe_admits = admits if unsafe_truth else 0
                false_rejects = rejects if not unsafe_truth else 0
                totals = np.array([x["total_observations"] for x in runs], dtype=float)
                results[scenario]["by_contamination"][label]["methods"][mode] = {
                    "admit": admits, "reject": rejects, "refuse_unresolved": unresolved,
                    "unsafe_admissions": unsafe_admits, "unsafe_admission_rate": unsafe_admits/repeats,
                    "unsafe_admission_wilson95": wilson(unsafe_admits, repeats),
                    "false_rejections": false_rejects, "mean_observations": float(totals.mean()),
                    "median_observations": float(np.median(totals)), "p95_observations": float(np.quantile(totals,.95)),
                }
    preflight_controls = {
        "missing_support_fails_closed": not preflight(support_covered=False, identity_matches=True, contamination_bound_known=True),
        "identity_mismatch_fails_closed": not preflight(support_covered=True, identity_matches=False, contamination_bound_known=True),
        "unknown_contamination_bound_fails_closed": not preflight(support_covered=True, identity_matches=True, contamination_bound_known=False),
        "all_integrity_assumptions_present_pass_preflight": preflight(support_covered=True, identity_matches=True, contamination_bound_known=True),
    }
    return {
        "schema": SCHEMA, "status": "SYNTHETIC_IID_HUBER_BERNOULLI_SENSITIVITY_EXPERIMENT_NOT_PRODUCTION_CERTIFICATION",
        "protocol": {"seed": SEED, "repeats_per_scenario_rho_method": repeats, "cells": 4, "familywise_delta": DELTA,
            "per_cell_delta": DELTA/4, "error_allowance": EPS, "pilot_per_cell": PILOT, "batch_size": BATCH,
            "max_observations_per_cell": MAX_PER_CELL, "rho_upper_grid": list(RHOS), "contamination_target": "cell 1 only; R=0",
            "clean_safe_model": [0.15,0.30,0.60,0.80], "clean_safe_target": [0.15,0.32,0.60,0.80],
            "unsafe_model": [0.15,0.30,0.60,0.80], "unsafe_clean_target": [0.15,0.37,0.60,0.80],
            "methods": {"nominal":"ignores contamination and treats observed mixture mean as clean target mean", "huber_robust":"maps observed-mean anytime CS through known contamination upper bound"}},
        "results": results, "preflight_negative_controls": preflight_controls,
        "limitations": ["IID Bernoulli Huber-mixture streams, contamination only in one cell", "the robust method uses a transparent Bernoulli sensitivity inversion, not the general Wang-Ramdas robust CS algorithm", "requires a defensible known upper bound on contamination fraction", "does not cover arbitrary time-varying corruption, dependent samples, unknown target support, or semantic identity mismatch except fail-closed preflight", "100 repeats are exploratory; zero unsafe admissions has nonzero population upper bound", "no direct SOTA implementation comparison or production Rust integration"],
        "literature": ["https://proceedings.mlr.press/v206/wang23p.html", "https://ojs.aaai.org/index.php/AAAI/article/view/39063", "https://www.jmlr.org/papers/v27/24-0680.html"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    start = time.perf_counter(); result = run(a.repeats)
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_bytes(raw)
    print(json.dumps({"path":str(a.output),"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"runtime_seconds":round(time.perf_counter()-start,3)}, indent=2))
