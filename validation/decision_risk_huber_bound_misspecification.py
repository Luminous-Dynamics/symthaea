#!/usr/bin/env python3
"""Paired stress benchmark: actual Huber contamination versus declared upper bound.

A robust certificate is only justified when its contamination upper bound is defensible.
This experiment deliberately sweeps actual rho and the bound supplied to the same
Bernoulli sensitivity-adjusted anytime-CS classifier. Per actual-rho repetition, all
claimed-bound conditions reuse identical observed streams. The claimed bound may be
correct/conservative or understated; only the former supports the stated robust claim.
"""
from __future__ import annotations
import hashlib, json, math, platform, time
from pathlib import Path
import numpy as np
import decision_risk_huber_sensitivity as huber

SCHEMA = "symthaea-decision-risk-huber-bound-misspecification-v1"
SEED = 20261030
REPEATS = 500
ACTUAL_RHOS = (0.0, 0.05, 0.10, 0.12, 0.18)
CLAIMED_RHOS = (0.0, 0.025, 0.05, 0.075, 0.10, 0.12, 0.15, 0.18, 0.20)


def run(repeats: int = REPEATS, actual_rhos=ACTUAL_RHOS, claimed_rhos=CLAIMED_RHOS) -> dict:
    actual_rhos = tuple(float(x) for x in actual_rhos)
    claimed_rhos = tuple(float(x) for x in claimed_rhos)
    if any(not (0.0 <= x < 1.0) for x in (*actual_rhos, *claimed_rhos)):
        raise ValueError("rho values must be in [0,1)")
    if repeats < 100:
        raise ValueError("use at least 100 repeats")
    model = np.array([0.15, 0.30, 0.60, 0.80], dtype=float)
    clean = np.array([0.15, 0.37, 0.60, 0.80], dtype=float)
    true_error = float(np.max(np.abs(model - clean)))
    rng = np.random.default_rng(SEED)
    results: dict[str, dict] = {}
    for actual_rho in actual_rhos:
        observed = clean.copy()
        observed[1] = (1 - actual_rho) * clean[1]
        accum = {f"{claim:.3f}": [] for claim in claimed_rhos}
        for _ in range(repeats):
            streams = np.vstack([rng.binomial(1, q, size=huber.MAX_PER_CELL) for q in observed]).astype(np.int8)
            for claimed_rho in claimed_rhos:
                out = huber.run_certifier(streams, model, mode="huber_robust", rho_upper=claimed_rho)
                accum[f"{claimed_rho:.3f}"].append(out)
        by_claim = {}
        for claimed_rho in claimed_rhos:
            runs = accum[f"{claimed_rho:.3f}"]
            admits = sum(r["status"] == "admit" for r in runs)
            rejects = sum(r["status"] == "reject" for r in runs)
            unresolved = sum(r["status"] == "refuse_unresolved" for r in runs)
            counts = np.array([r["total_observations"] for r in runs], dtype=float)
            by_claim[f"{claimed_rho:.3f}"] = {
                "claimed_upper_bound": claimed_rho,
                "bound_covers_actual": claimed_rho + 1e-12 >= actual_rho,
                "bound_understatement": max(0.0, actual_rho - claimed_rho),
                "admit": admits, "reject": rejects, "refuse_unresolved": unresolved,
                "unsafe_admissions": admits, "unsafe_admission_rate": admits / repeats,
                "unsafe_admission_wilson95": huber.wilson(admits, repeats),
                "mean_observations": float(counts.mean()),
                "median_observations": float(np.median(counts)),
                "p95_observations": float(np.quantile(counts, 0.95)),
            }
        results[f"actual_rho_{actual_rho:.3f}"] = {
            "actual_contamination_fraction": actual_rho,
            "observed_attacked_cell_mean": float(observed[1]),
            "clean_target_mean": float(clean[1]),
            "true_clean_max_model_error_for_scoring_only": true_error,
            "claimed_bound_grid": by_claim,
        }
    unknown = huber.run_certifier(np.zeros((4, huber.MAX_PER_CELL), dtype=np.int8), model,
                                  mode="huber_robust", rho_upper=None)
    return {
        "schema": SCHEMA,
        "status": "SYNTHETIC_BOUND_MISSPECIFICATION_STRESS_NOT_PRODUCTION_CERTIFICATION",
        "protocol": {
            "seed": SEED, "repeats_per_actual_claimed_pair": repeats,
            "actual_rho_grid": list(actual_rhos), "claimed_rho_grid": list(claimed_rhos),
            "cells": 4, "familywise_delta": huber.DELTA, "per_cell_delta": huber.DELTA / 4,
            "error_tolerance": huber.EPS, "pilot": huber.PILOT, "batch": huber.BATCH,
            "max_per_cell": huber.MAX_PER_CELL, "contaminant": "Bernoulli contaminant R=0 in attacked cell only",
            "same_streams_across_claims": True,
            "truth_model": model.tolist(), "clean_target": clean.tolist(),
            "question": "Does the sensitivity certificate remain safe when the declared contamination bound underestimates actual contamination?",
        },
        "results": results,
        "unknown_bound_negative_control": {"status": unknown["status"], "reason": unknown.get("reason"), "total_observations": unknown["total_observations"]},
        "limitations": [
            "IID Bernoulli mixture with contamination concentrated in one cell and R=0",
            "the true rho is known only to the simulator; production systems need an independent, defensible upper bound",
            "validity claim applies only where claimed_rho >= actual_rho and other assumptions hold",
            "0 unsafe admissions in finite repetitions is not zero population risk",
            "this uses Bernoulli sensitivity inversion rather than a reproduction of Wang-Ramdas robust CS",
            "no direct published SOTA code run; no production Rust integration",
        ],
        "literature": ["https://proceedings.mlr.press/v206/wang23p.html", "https://github.com/HJ-Wg/catoni-confseq"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repeats", type=int, default=REPEATS)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--actual-rhos", default=",".join(str(x) for x in ACTUAL_RHOS))
    p.add_argument("--claimed-rhos", default=",".join(str(x) for x in CLAIMED_RHOS))
    a = p.parse_args()
    actual_rhos = tuple(float(x) for x in a.actual_rhos.split(","))
    claimed_rhos = tuple(float(x) for x in a.claimed_rhos.split(","))
    start = time.perf_counter(); result = run(a.repeats, actual_rhos, claimed_rhos)
    data = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_bytes(data)
    print(json.dumps({"path": str(a.output), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "runtime_seconds": round(time.perf_counter()-start, 3)}, indent=2))
