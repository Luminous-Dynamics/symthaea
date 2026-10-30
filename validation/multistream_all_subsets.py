#!/usr/bin/env python3
"""All-subsets certificate for running maxima of e-processes.

Implements the SAVI-2026 poster's simultaneous group tail inequality. For
V_k=max(1,sup_t E_t^k), let V_(1)>=...>=V_(K). The largest subset sum among
sets of size m is the top-m prefix, so all 2^K subsets are handled in O(K log K)
per replicate rather than enumerated.

This is a simultaneous group/intersection-null certificate, not individual
e-values, not an FDR procedure, and not a closed-testing implementation.
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

SCHEMA = "symthaea-running-max-all-subsets-v1"


def _validate_log_v(log_v: np.ndarray) -> np.ndarray:
    x = np.asarray(log_v, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError("log_v must have a nonempty final axis")
    if np.any(~np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("log_v must be finite and nonnegative (V_k >= 1)")
    return x


def all_subset_log_score(log_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return log calibrated score and maximizing subset size.

    Poster inequality:
      P(exists nonempty S subset [K]: sum_{k in S} V_k/sqrt(H_|S|) > lam)
        <= 2*K*sqrt(H_K)/lam.
    For a fixed size m, the largest subset sum is the sum of the m largest V's.
    Sorting once and scanning all K prefixes therefore computes the maximum.
    Returns log(score) and the selected top-m cardinality.
    """
    x = _validate_log_v(log_v)
    k = x.shape[-1]
    ordered = np.sort(x, axis=-1)[..., ::-1]
    log_prefix_sum = np.logaddexp.accumulate(ordered, axis=-1)
    harmonic = np.cumsum(1.0 / np.arange(1, k + 1, dtype=np.float64))
    log_candidates = log_prefix_sum - 0.5 * np.log(harmonic)
    best_index = np.argmax(log_candidates, axis=-1)
    log_top_statistic = np.max(log_candidates, axis=-1)
    log_normalizer = math.log(2.0 * k) + 0.5 * math.log(float(harmonic[-1]))
    return log_top_statistic - log_normalizer, best_index + 1


def bonferroni_max_log_score(log_v: np.ndarray) -> np.ndarray:
    x = _validate_log_v(log_v)
    return np.max(x, axis=-1) - math.log(x.shape[-1])


def direct_power_log_score(log_v: np.ndarray, power: float) -> np.ndarray:
    x = _validate_log_v(log_v)
    s = float(power)
    if not math.isfinite(s) or s <= 0.0:
        raise ValueError("power must be finite and positive")
    k = x.shape[-1]
    ranks = np.arange(1, k + 1, dtype=np.float64)
    from scipy.special import logsumexp
    log_norm = math.log((s + 1.0) / s) + math.log(k) + math.log(float(np.sum(ranks ** (-s)))) / s
    return logsumexp(s * x, axis=-1) / s - log_norm


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    p = successes / trials
    den = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / den
    half = z * math.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / den
    lower = 0.0 if successes == 0 else max(0.0, center - half)
    upper = 1.0 if successes == trials else min(1.0, center + half)
    return [lower, upper]


def simulate_case(*, k: int, timepoints: int, repetitions: int,
                  signal_count: int, rho: float, seed: int,
                  alternative_mean: float = 0.1, batch_size: int = 25,
                  alpha: float = 0.05) -> dict:
    if k < 1 or timepoints < 1 or repetitions < 1 or batch_size < 1:
        raise ValueError("k, timepoints, repetitions and batch_size must be positive")
    if not 0 <= signal_count <= k:
        raise ValueError("signal_count must lie in [0,k]")
    if not 0 <= rho < 1:
        raise ValueError("rho must lie in [0,1)")
    if not 0 < alpha < 1 or alternative_mean <= 0:
        raise ValueError("alpha must be in (0,1) and alternative_mean positive")

    rng = np.random.default_rng(seed)
    means = np.zeros(k, dtype=np.float64)
    if signal_count:
        means[-signal_count:] = alternative_mean
    names = ("all_subsets", "bonferroni_max", "direct_power_s1",
             "direct_power_s2", "direct_power_s4")
    hits = {name: 0 for name in names}
    selected_sizes = []
    processed = 0
    started = time.perf_counter()
    while processed < repetitions:
        b = min(batch_size, repetitions - processed)
        idiosyncratic = rng.standard_normal((b, timepoints, k))
        if rho:
            common = rng.standard_normal((b, timepoints, 1))
            innovations = math.sqrt(rho) * common + math.sqrt(1.0 - rho) * idiosyncratic
        else:
            innovations = idiosyncratic
        observations = innovations + means
        increments = alternative_mean * observations - 0.5 * alternative_mean**2
        cumulative_log_e = np.cumsum(increments, axis=1)
        log_v = np.maximum(0.0, np.max(cumulative_log_e, axis=1))
        all_score, best_m = all_subset_log_score(log_v)
        score_map = {
            "all_subsets": all_score,
            "bonferroni_max": bonferroni_max_log_score(log_v),
            "direct_power_s1": direct_power_log_score(log_v, 1.0),
            "direct_power_s2": direct_power_log_score(log_v, 2.0),
            "direct_power_s4": direct_power_log_score(log_v, 4.0),
        }
        cutoff = math.log(1.0 / alpha)
        for name, scores in score_map.items():
            hits[name] += int(np.sum(scores >= cutoff))
        selected_sizes.extend(best_m.tolist())
        processed += b

    results = {
        name: {"rejections": count, "rate": count / repetitions,
               "wilson95": wilson_interval(count, repetitions)}
        for name, count in hits.items()
    }
    selected = np.asarray(selected_sizes, dtype=np.int64)
    return {
        "protocol": {
            "k": k, "timepoints": timepoints, "repetitions": repetitions,
            "signal_count": signal_count, "alternative_mean": alternative_mean,
            "rho": rho, "seed": seed, "batch_size": batch_size,
            "alpha": alpha,
            "dependence": "independent-over-time equicorrelated Gaussian innovations",
            "null_process": "E_t=exp(sum_{u<=t}(mu*X_u-mu^2/2)), X_u marginal N(0,1)",
            "tested_claim": "simultaneous nonempty subset/group intersection-null certificate",
        },
        "results": results,
        "selected_subset_size": {
            "mean": float(selected.mean()), "median": float(np.median(selected)),
            "p10": float(np.quantile(selected, .10)), "p90": float(np.quantile(selected, .90)),
        },
        "runtime_seconds": time.perf_counter() - started,
    }


def run_pilot() -> dict:
    cases = []
    for rho, seed in ((0.0, 20261030), (0.3, 20261031)):
        cases.append({"kind": "all_null_type_i", **simulate_case(
            k=200, timepoints=500, repetitions=3000, signal_count=0,
            rho=rho, seed=seed, batch_size=100,
        )})
    for k, q, seed in (
        (20, 1, 123), (20, 20, 144), (100, 20, 154),
        (100, 100, 204), (200, 20, 244), (200, 100, 304),
    ):
        cases.append({"kind": "adaptive_group_detection_power", **simulate_case(
            k=k, timepoints=1000, repetitions=500, signal_count=q,
            rho=0.3, seed=seed, batch_size=25,
        )})
    return {
        "schema": SCHEMA,
        "status": "LOCAL_SYNTHETIC_PILOT_NOT_A_SOTA_CLAIM",
        "theorem_source": "SAVI 2026 poster; associated paper listed as in preparation",
        "theorem": "P(exists nonempty S: sum_{k in S} V_k/sqrt(H_|S|)>lambda) <= 2*K*sqrt(H_K)/lambda",
        "algorithm": "sort V descending; maximize top-m prefix/sqrt(H_m), m=1..K; O(K log K) time and O(K) space",
        "complexity_note": "avoids enumerating 2^K subsets",
        "cases": cases,
        "limitations": [
            "one global simultaneous subset certificate; not per-hypothesis e-values, not e-BH, not FDR-sup",
            "SAVI 2026 poster; associated paper listed as in preparation; theorem not independently reproved here",
            "one dependence family (equicorrelated Gaussian innovations)",
            "pilot rejection rates far below alpha in all-null runs; no calibration stress near boundary yet",
            "no arbitrary-dependence adversarial construction or closed-testing implementation",
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
    print(json.dumps({"output": str(args.output), "bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest(),
                      "cases": len(result["cases"])}, indent=2))


if __name__ == "__main__":
    main()
