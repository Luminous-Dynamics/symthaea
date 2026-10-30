#!/usr/bin/env python3
"""Dependence-robust group tests for running maxima of e-processes.

Research implementation of the general-aggregator result in Tavyrikov,
Goeman & de Heide, SAVI 2026 poster (associated paper listed as in preparation).
This is a group-level intersection-null certificate, not an e-value and not
an individual-hypothesis FDR procedure.
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
from scipy.special import logsumexp

SCHEMA = "symthaea-running-max-group-aggregators-v1"
DEFAULT_POWERS = (1.0, 2.0, 4.0)


def _validate_log_v(log_v: np.ndarray) -> np.ndarray:
    x = np.asarray(log_v, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError("log_v must have a nonempty final axis")
    if np.any(~np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("log_v must be finite and nonnegative (V_k >= 1)")
    return x


def log_adjuster_a1(log_v: np.ndarray) -> np.ndarray:
    """log A1(V), A1(V)=(V-1-log(V))/log(V)^2, with A1(1)=1/2."""
    x = _validate_log_v(log_v)
    out = np.empty_like(x)
    small = x < 1e-3
    z = x[small]
    ratio = (0.5 + z / 6.0 + z**2 / 24.0 + z**3 / 120.0
             + z**4 / 720.0 + z**5 / 5040.0 + z**6 / 40320.0)
    out[small] = np.log(ratio)
    z = x[~small]
    out[~small] = z + np.log1p(-(1.0 + z) * np.exp(-z)) - 2.0 * np.log(z)
    return out


def log_adjuster_a2(log_v: np.ndarray) -> np.ndarray:
    """log A2(V), A2(V)=sqrt(V)-1, with A2(1)=0."""
    x = _validate_log_v(log_v)
    out = np.full_like(x, -np.inf)
    positive = x > 0.0
    z = x[positive] / 2.0
    large = z > 50.0
    vals = np.empty_like(z)
    vals[~large] = np.log(np.expm1(z[~large]))
    vals[large] = z[large] + np.log1p(-np.exp(-z[large]))
    out[positive] = vals
    return out


def power_group_log_scores(log_v: np.ndarray, powers: tuple[float, ...] = DEFAULT_POWERS) -> dict[float, np.ndarray]:
    """Return log Z_s where P(Z_s >= t) <= 1/t under the intersection null.

    Poster theorem:
      P(sum_k V_k^s > lambda)
        <= ((s+1)/s) * K * (H_K^(s))^(1/s) / lambda^(1/s).
    Therefore Z_s = ||V||_s / [((s+1)/s) K (H_K^(s))^(1/s)]
    has a Ville-type 1/t tail.
    """
    x = _validate_log_v(log_v)
    k = x.shape[-1]
    ranks = np.arange(1, k + 1, dtype=np.float64)
    out: dict[float, np.ndarray] = {}
    for raw_s in powers:
        s = float(raw_s)
        if not math.isfinite(s) or s <= 0.0:
            raise ValueError("each power must be finite and positive")
        h_s = float(np.sum(ranks ** (-s)))
        log_l_s = logsumexp(s * x, axis=-1) / s
        log_c = math.log((s + 1.0) / s) + math.log(k) + math.log(h_s) / s
        out[s] = log_l_s - log_c
    return out


def mixed_power_group_log_score(
    log_v: np.ndarray,
    powers: tuple[float, ...] = DEFAULT_POWERS,
    weights: tuple[float, ...] | None = None,
    iterations: int = 60,
) -> np.ndarray:
    """Single calibrated f-mixture score from the general-aggregator theorem.

    Let f(x)=sum_j w_j*x**s_j, F=sum_k f(V_k), and choose mu solving
      F = sum_j w_j H_K^(s_j) mu**s_j.
    The theorem defines
      Lambda_K(F)=mu*F-sum_j w_j H_K^(s_j)*mu**(s_j+1)/(s_j+1).
    Then Z_f=Lambda_K(F)/(K*F) has a 1/t tail.
    """
    x = _validate_log_v(log_v)
    ss = np.asarray(powers, dtype=np.float64)
    if ss.ndim != 1 or ss.size == 0:
        raise ValueError("powers must be a nonempty vector")
    if weights is None:
        raw = np.power(10.0, -ss)
        ww = raw / raw.sum()
    else:
        ww = np.asarray(weights, dtype=np.float64)
    if ww.shape != ss.shape:
        raise ValueError("powers and weights must have equal size")
    if np.any(~np.isfinite(ss)) or np.any(ss <= 0.0):
        raise ValueError("powers must be positive and finite")
    if np.any(~np.isfinite(ww)) or np.any(ww <= 0.0):
        raise ValueError("weights must be positive and finite")
    if abs(float(ww.sum()) - 1.0) > 1e-10:
        raise ValueError("weights must sum to one")
    if iterations < 20:
        raise ValueError("at least 20 root iterations are required")

    k = x.shape[-1]
    ranks = np.arange(1, k + 1, dtype=np.float64)
    log_w = np.log(ww)
    log_h = np.log(np.array([np.sum(ranks ** (-s)) for s in ss]))
    log_f_terms = np.stack(
        [log_w[j] + logsumexp(ss[j] * x, axis=-1) for j in range(ss.size)], axis=-1
    )
    log_f = logsumexp(log_f_terms, axis=-1)
    single_power_roots = np.stack(
        [(log_f - log_w[j] - log_h[j]) / ss[j] for j in range(ss.size)], axis=-1
    )
    lo = np.min(single_power_roots, axis=-1) - 5.0
    hi = np.max(single_power_roots, axis=-1) + 5.0
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        log_g = logsumexp(
            np.stack([log_w[j] + log_h[j] + ss[j] * mid for j in range(ss.size)], axis=-1),
            axis=-1,
        )
        move_right = log_g < log_f
        lo = np.where(move_right, mid, lo)
        hi = np.where(move_right, hi, mid)
    log_mu = 0.5 * (lo + hi)
    log_q = np.stack(
        [log_w[j] + log_h[j] + ss[j] * log_mu - log_f for j in range(ss.size)], axis=-1
    )
    log_factor = logsumexp(log_q + np.log(ss / (ss + 1.0)), axis=-1)
    return log_mu - math.log(k) + log_factor


def group_log_scores(log_v: np.ndarray) -> dict[str, np.ndarray]:
    """Return log-calibrated group statistics; larger means stronger evidence."""
    x = _validate_log_v(log_v)
    k = x.shape[-1]
    power = power_group_log_scores(x)
    portfolio_weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    portfolio = np.max(
        np.stack(
            [math.log(portfolio_weights[i]) + power[s]
             for i, s in enumerate(DEFAULT_POWERS)], axis=-1
        ), axis=-1
    )
    return {
        "bonferroni_max": np.max(x, axis=-1) - math.log(k),
        "adjuster_a1_mean": logsumexp(log_adjuster_a1(x), axis=-1) - math.log(k),
        "adjuster_a2_mean": logsumexp(log_adjuster_a2(x), axis=-1) - math.log(k),
        "direct_power_s1": power[1.0],
        "direct_power_s2": power[2.0],
        "direct_power_s4": power[4.0],
        "power_portfolio_s1_s2_s4": portfolio,
        "mixed_power_f_s1_s2_s4": mixed_power_group_log_score(x),
    }


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


def simulate_group_case(
    *, k: int = 200, timepoints: int = 1000, repetitions: int = 500,
    signal_count: int = 20, alternative_mean: float = 0.1, rho: float = 0.3,
    seed: int = 20261030, batch_size: int = 25, alpha: float = 0.05,
) -> dict:
    """Simulate a pre-registered group intersection-null test.

    Under the all-null intersection, every stream is an e-process with respect
    to the common filtration generated by the joint innovation vectors: time
    increments are independent, and each coordinate is conditionally N(0,1).
    Cross-stream dependence is equicorrelation rho.
    """
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
    names = tuple(group_log_scores(np.zeros((1, k))).keys())
    hits = {name: 0 for name in names}
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
        log_increment = alternative_mean * observations - 0.5 * alternative_mean**2
        cumulative_log_e = np.cumsum(log_increment, axis=1)
        log_v = np.maximum(0.0, np.max(cumulative_log_e, axis=1))
        scores = group_log_scores(log_v)
        for name, values in scores.items():
            hits[name] += int(np.sum(values >= math.log(1.0 / alpha)))
        processed += b
    results = {
        name: {"rejections": count, "rate": count / repetitions,
               "wilson95": wilson_interval(count, repetitions)}
        for name, count in hits.items()
    }
    return {
        "protocol": {
            "k": k, "timepoints": timepoints, "repetitions": repetitions,
            "signal_count": signal_count, "alternative_mean": alternative_mean,
            "rho": rho, "seed": seed, "batch_size": batch_size, "alpha": alpha,
            "dependence": "independent-over-time equicorrelated Gaussian innovations",
            "null_process": "E_t=exp(sum_{u<=t}(mu*X_u-mu^2/2)) with X_u marginal N(0,1)",
            "tested_claim": "global intersection null for one pre-registered group",
        },
        "results": results, "runtime_seconds": time.perf_counter() - started,
    }


def run_pilot() -> dict:
    cases = []
    for rho, seed in ((0.0, 20261030), (0.3, 20261031)):
        cases.append({
            "kind": "all_null_type_i",
            **simulate_group_case(k=200, timepoints=500, repetitions=3000,
                signal_count=0, rho=rho, seed=seed, batch_size=20),
        })
    for k, q, seed in (
        (20, 1, 20261400), (20, 20, 20261401), (100, 20, 20261402),
        (100, 100, 20261403), (200, 20, 20261404), (200, 100, 20261405),
    ):
        cases.append({
            "kind": "pre_registered_group_detection_power",
            **simulate_group_case(k=k, timepoints=1000, repetitions=500,
                signal_count=q, rho=0.3, seed=seed, batch_size=25),
        })
    return {
        "schema": SCHEMA,
        "status": "LOCAL_SYNTHETIC_PILOT_NOT_A_SOTA_CLAIM",
        "source_reference": {
            "title": "Multiple Testing with Anytime-Valid Evidence: Tail bounds for sums of running maximums of dependent E-processes",
            "authors": "Yury Tavyrikov, Jelle Goeman, Rianne de Heide",
            "venue": "SAVI 2026 poster; associated paper listed as in preparation",
            "url": "https://riannedeheide.github.io/PosterSAVI2026Yury.pdf",
        },
        "methods": {
            "direct_power_s": "general-aggregator theorem; log ||V||_s divided by log(((s+1)/s)*K*(H_K^(s))^(1/s))",
            "mixed_power": "single f(x)=sum_j w_j*x^s_j; solve Lambda_K calibration without alpha spending over s",
            "portfolio": "max of separate calibrated s-scores with fixed one-third weights",
            "a1_a2": "mean adjusted maxima, a group e-value under the global intersection null",
            "bonferroni": "max V_k / K",
        },
        "mixture": {
            "powers": list(DEFAULT_POWERS),
            "weights": [0.9082652134, 0.0908265213, 0.0009082652],
            "interpretation": "fixed scale-tuned mixture; not data-adaptive and not claimed optimal",
        },
        "limitations": [
            "poster theorem and general-f construction have not yet been independently formalized or peer-reviewed in this project",
            "equicorrelation is one limited dependence family, not an arbitrary-dependence stress proof",
            "the experiment tests one pre-registered group intersection null; it does not implement closed testing over all subsets",
            "no per-hypothesis FDR, FDR-sup, or FWER comparison is claimed",
            "single Gaussian alternative effect size; no misspecification, heavy-tail, or nonstationarity sweep",
            "the fixed mixture weights were selected as a scale-tuned pilot, not optimized on a holdout benchmark",
        ],
        "cases": cases,
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
        "sha256": hashlib.sha256(raw).hexdigest(), "cases": len(result["cases"]),
    }, indent=2))


if __name__ == "__main__":
    main()
