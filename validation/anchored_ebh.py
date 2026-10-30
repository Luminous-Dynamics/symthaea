#!/usr/bin/env python3
"""Anchored e-BH under arbitrary dependence: independent research implementation.

Implements Proposition 7.1 / equations (39)-(40) in Ramdas (2026),
"Anchored multiple testing". This is an offline fixed-family procedure.
It is not a sequential SupFDR method and must not be applied repeatedly at
arbitrary times without a separate time-uniform proof.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any
import numpy as np

SCHEMA = "symthaea-anchored-ebh-v1"


def _validate_evalues(evalues: np.ndarray) -> np.ndarray:
    e = np.asarray(evalues, dtype=np.float64)
    if e.ndim != 1 or e.size == 0:
        raise ValueError("evalues must be a nonempty one-dimensional vector")
    if np.any(~np.isfinite(e)) or np.any(e < 0.0):
        raise ValueError("evalues must be finite and nonnegative")
    return e


def ebh(evalues: np.ndarray, alpha: float) -> np.ndarray:
    """Return standard e-BH rejection indices, sorted by descending e-value."""
    e = _validate_evalues(evalues)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0,1)")
    m = e.size
    order = np.argsort(-e, kind="stable")
    sorted_e = e[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    valid = sorted_e >= m / (alpha * ranks)
    k_star = int(ranks[valid].max()) if np.any(valid) else 0
    return order[:k_star]


def _anchored_size_from_sorted(sorted_e: np.ndarray, alpha: float) -> int:
    """O(m log m) largest-cardinality anchored closure superset.

    h_i=(e_i-1)_+, C=m-sum_i(1-e_i)_+, and candidate R of size r
    (the top-r e-values) is certified iff
        sum_{i in R} (m/(alpha*r)-h_i)_+ <= C.
    This is the equivalent form of equation (40) in Ramdas (2026).
    """
    m = sorted_e.size
    h = np.maximum(sorted_e - 1.0, 0.0)
    c_budget = float(m - np.maximum(1.0 - sorted_e, 0.0).sum())
    prefix_h = np.concatenate(([0.0], np.cumsum(h, dtype=np.float64)))
    best = 0
    for r in range(1, m + 1):
        threshold = m / (alpha * r)
        first = int(np.searchsorted(-h[:r], -threshold, side="left"))
        count = r - first
        suffix_sum = float(prefix_h[r] - prefix_h[first])
        cost = count * threshold - suffix_sum
        if cost <= c_budget + 1e-10 * max(1.0, c_budget):
            best = r
    return best


def anchored_ebh(evalues: np.ndarray, alpha: float) -> dict[str, Any]:
    """Return baseline e-BH and largest certified anchored superset.

    FDR guarantee: offline fixed family, arbitrary dependence among marginal
    e-values. This does not imply SupFDR when recomputed over time.
    """
    e = _validate_evalues(evalues)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0,1)")
    order = np.argsort(-e, kind="stable")
    sorted_e = e[order]
    baseline = ebh(e, alpha)
    size = _anchored_size_from_sorted(sorted_e, alpha)
    if size < baseline.size:
        raise RuntimeError("anchoring failed to preserve the baseline e-BH set")
    return {
        "baseline_indices": baseline,
        "anchored_indices": order[:size],
        "baseline_size": int(baseline.size),
        "anchored_size": int(size),
        "added_indices": order[baseline.size:size],
    }


def anchored_closure_feasible(evalues: np.ndarray, alpha: float, rejected: np.ndarray) -> bool:
    """Direct all-intersections check using E_S=(sum_{i in S}e_i+m-|S|)/m.

    For nonempty R, the collection of closure constraints is equivalent to
      sum_i (1 + m/(alpha*|R|) 1{i in R} - e_i)_+ <= m.
    """
    e = _validate_evalues(evalues)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0,1)")
    idx = np.asarray(rejected, dtype=np.int64)
    if idx.ndim != 1 or np.any(idx < 0) or np.any(idx >= e.size):
        raise ValueError("rejected indices are invalid")
    if np.unique(idx).size != idx.size:
        raise ValueError("rejected indices must be unique")
    r = idx.size
    if r == 0:
        return True
    selected = np.zeros(e.size, dtype=bool)
    selected[idx] = True
    terms = 1.0 - e
    terms[selected] += e.size / (alpha * r)
    return bool(np.maximum(terms, 0.0).sum() <= e.size + 1e-10)


def exact_eclosure_feasible(evalues: np.ndarray, alpha: float, rejected: np.ndarray) -> bool:
    """Exhaustive equation (1) verifier for small m only; O(2^m)."""
    e = _validate_evalues(evalues)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0,1)")
    idx = np.asarray(rejected, dtype=np.int64)
    if idx.ndim != 1 or np.any(idx < 0) or np.any(idx >= e.size):
        raise ValueError("rejected indices are invalid")
    if np.unique(idx).size != idx.size:
        raise ValueError("rejected indices must be unique")
    r = idx.size
    if r == 0:
        return True
    selected = set(idx.tolist())
    m = e.size
    for mask in range(1, 1 << m):
        intersection = [i for i in range(m) if mask & (1 << i)]
        fdp = sum(i in selected for i in intersection) / r
        local_e = (sum(e[i] for i in intersection) + m - len(intersection)) / m
        if fdp > alpha * local_e + 1e-10:
            return False
    return True


def anchored_ebh_batch(evalues: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    """Vectorized multi-replicate evaluator; each replicate is O(m log m)."""
    e = np.asarray(evalues, dtype=np.float64)
    if e.ndim != 2 or e.shape[1] == 0:
        raise ValueError("evalues must be a nonempty (replicates,m) matrix")
    if np.any(~np.isfinite(e)) or np.any(e < 0.0):
        raise ValueError("evalues must be finite and nonnegative")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0,1)")
    reps, m = e.shape
    order = np.argsort(-e, axis=1, kind="stable")
    sorted_e = np.take_along_axis(e, order, axis=1)
    ranks = np.arange(1, m + 1, dtype=np.float64)[None, :]
    baseline_ok = sorted_e >= m / (alpha * ranks)
    baseline_size = np.max(np.where(baseline_ok, ranks, 0.0), axis=1).astype(np.int64)
    h = np.maximum(sorted_e - 1.0, 0.0)
    c_budget = m - np.maximum(1.0 - sorted_e, 0.0).sum(axis=1)
    prefix_h = np.concatenate(
        [np.zeros((reps, 1), dtype=np.float64), np.cumsum(h, axis=1)], axis=1
    )
    best = np.zeros(reps, dtype=np.int64)
    row = np.arange(reps)
    for r in range(1, m + 1):
        threshold = m / (alpha * r)
        first = np.minimum(r, np.sum(h >= threshold, axis=1))
        suffix = prefix_h[:, r] - prefix_h[row, first]
        cost = (r - first) * threshold - suffix
        feasible = cost <= c_budget + 1e-10 * np.maximum(1.0, c_budget)
        best = np.where(feasible, r, best)
    if np.any(best < baseline_size):
        raise RuntimeError("anchoring failed to preserve baseline e-BH")
    return {"order": order, "baseline_size": baseline_size, "anchored_size": best}


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    p = successes / trials
    den = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / den
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def paired_power_metrics(evalues: np.ndarray, null_mask: np.ndarray, alpha: float) -> dict:
    e = np.asarray(evalues, dtype=np.float64)
    nulls = np.asarray(null_mask, dtype=bool)
    if e.ndim != 2 or e.shape[1] != nulls.size:
        raise ValueError("evalues must be (replicates,m) and null_mask must have length m")
    reps, m = e.shape
    if not np.any(~nulls):
        raise ValueError("power metrics require at least one nonnull hypothesis")
    start = time.perf_counter()
    result = anchored_ebh_batch(e, alpha)
    order, base_size, anchored_size = (
        result["order"], result["baseline_size"], result["anchored_size"]
    )
    null_sorted = np.take_along_axis(np.broadcast_to(nulls, (reps, m)), order, axis=1)
    positions = np.arange(m)[None, :]
    base_mask = positions < base_size[:, None]
    anchored_mask = positions < anchored_size[:, None]

    def fdp_power(rej_mask, sizes):
        false = np.sum(rej_mask & null_sorted, axis=1)
        true = np.sum(rej_mask & ~null_sorted, axis=1)
        fdp = false / np.maximum(sizes, 1)
        power = true / int((~nulls).sum())
        return fdp.astype(np.float64), power.astype(np.float64)

    baseline_fdp, baseline_power = fdp_power(base_mask, base_size)
    anchored_fdp, anchored_power = fdp_power(anchored_mask, anchored_size)
    elapsed = time.perf_counter() - start

    def stats(x):
        mean = float(x.mean())
        se = float(x.std(ddof=1) / math.sqrt(reps)) if reps > 1 else None
        return {
            "mean": mean, "se": se,
            "ci95": [max(0.0, mean - 1.96 * se), min(1.0, mean + 1.96 * se)]
            if se is not None else None,
        }

    delta = anchored_power - baseline_power
    return {
        "eBH": {
            "fdr": stats(baseline_fdp), "power": stats(baseline_power),
            "mean_rejections": float(base_size.mean()),
            "zero_rejection_rate": float(np.mean(base_size == 0)),
        },
        "anchored_eBH": {
            "fdr": stats(anchored_fdp), "power": stats(anchored_power),
            "mean_rejections": float(anchored_size.mean()),
            "zero_rejection_rate": float(np.mean(anchored_size == 0)),
        },
        "paired_power_gain": stats(delta),
        "power_win_rate": float(np.mean(delta > 0)),
        "baseline_containment_rate": 1.0,
        "decision_runtime_seconds": elapsed,
    }


def generate_likelihood_ratio_evalues(
    *, repetitions: int, m: int, nonnull_count: int, effect: float,
    dependence: str, rho: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Marginally valid LR e-values for N(0,1) versus N(effect,1)."""
    if repetitions < 1 or m < 1 or not 0 <= nonnull_count <= m:
        raise ValueError("invalid dimensions")
    if not math.isfinite(effect) or effect <= 0.0:
        raise ValueError("effect must be positive")
    if dependence not in {"independent", "equicorrelated"}:
        raise ValueError("unsupported dependence mode")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0,1)")
    rng = np.random.default_rng(seed)
    if dependence == "independent":
        z = rng.standard_normal((repetitions, m))
    else:
        common = rng.standard_normal((repetitions, 1))
        idiosyncratic = rng.standard_normal((repetitions, m))
        z = math.sqrt(rho) * common + math.sqrt(1.0 - rho) * idiosyncratic
    null_mask = np.ones(m, dtype=bool)
    if nonnull_count:
        null_mask[m - nonnull_count:] = False
        z[:, m - nonnull_count:] += effect
    log_e = effect * z - 0.5 * effect * effect
    e = np.exp(log_e)
    if np.any(~np.isfinite(e)):
        raise FloatingPointError("likelihood-ratio e-value overflow; use a lower effect")
    return e, null_mask


def simulate_sharp_all_null(
    repetitions: int = 20000, m: int = 100, alpha: float = 0.05, seed: int = 20261030
) -> dict:
    """Boundary-focused null: independent e_i in {0,m/alpha}, E[e_i]=1."""
    if repetitions < 1 or m < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("invalid simulation arguments")
    rng = np.random.default_rng(seed)
    spike = m / alpha
    values = (rng.random((repetitions, m)) < (1.0 / spike)).astype(np.float64) * spike
    result = anchored_ebh_batch(values, alpha)
    base_size = result["baseline_size"]
    anchored_size = result["anchored_size"]
    base_false = int(np.sum(base_size > 0))
    anchored_false = int(np.sum(anchored_size > 0))
    return {
        "protocol": {
            "repetitions": repetitions, "m": m, "alpha": alpha,
            "null_e_value_distribution": f"{spike} with probability {1/spike}; 0 otherwise",
            "marginal_expectation": 1.0,
        },
        "eBH_false_discoveries": base_false,
        "eBH_fdr": base_false / repetitions,
        "eBH_fdr_wilson95": wilson_interval(base_false, repetitions),
        "anchored_false_discoveries": anchored_false,
        "anchored_fdr": anchored_false / repetitions,
        "anchored_fdr_wilson95": wilson_interval(anchored_false, repetitions),
        "mean_anchored_additions": float((anchored_size - base_size).mean()),
    }


def run_pilot() -> dict:
    cases = []
    idx = 0
    for dependence, rho in (("independent", 0.0), ("equicorrelated", 0.5)):
        for fraction in (0.1, 0.5, 0.9):
            q = int(100 * fraction)
            for effect in (2.5, 3.0, 4.0):
                e, nulls = generate_likelihood_ratio_evalues(
                    repetitions=5000, m=100, nonnull_count=q, effect=effect,
                    dependence=dependence, rho=rho, seed=20261030 + idx
                )
                metrics = paired_power_metrics(e, nulls, 0.05)
                cases.append({
                    "dependence": dependence, "rho": rho, "m": 100,
                    "nonnull_fraction": fraction, "nonnull_count": q,
                    "effect": effect, "alpha": 0.05, "repetitions": 5000,
                    "seed": 20261030 + idx, **metrics,
                })
                idx += 1
    return {
        "schema": SCHEMA,
        "status": "LOCAL_SYNTHETIC_PILOT_NOT_A_SOTA_CLAIM",
        "reference": {
            "title": "Anchored multiple testing: a transparent use of e-closure to improve FDR procedures",
            "author": "Aaditya Ramdas", "date": "2026-10-15 version",
            "url": "https://arxiv.org/abs/2610.11279",
            "section": "7 Anchored e-BH under arbitrary dependence",
        },
        "method": {
            "baseline": "eBH_alpha(e)",
            "local_intersection_evalue": "E_S=(sum_{i in S} e_i + m - |S|)/m",
            "certificate": "FDP_S(R) <= alpha*E_S for every nonempty intersection S",
            "optimization": "For candidate size r, top-r e-values minimize closure burden; check sum_{i in R}(m/(alpha*r)-(e_i-1)_+)_+ <= m-sum_i(1-e_i)_+",
            "complexity": "O(m log m) time and O(m) memory",
            "guarantee_scope": "offline fixed family; arbitrary dependence among marginal e-values",
        },
        "sharp_null": simulate_sharp_all_null(),
        "cases": cases,
        "limitations": [
            "offline fixed family only; no sequential SupFDR claim",
            "one normal likelihood-ratio alternative family; not an exhaustive power benchmark",
            "independent and positive equicorrelation only in this primary run",
            "Anchored-eBH is incomparable with mean closed e-BH, not a universal replacement",
            "local tests independently validate closure constraints for small m; they do not independently prove the theorem",
        ],
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_pilot()
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({
        "output": str(args.output), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "sharp_null": result["sharp_null"], "cases": len(result["cases"]),
    }, indent=2))


if __name__ == "__main__":
    main()
