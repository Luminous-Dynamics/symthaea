#!/usr/bin/env python3
"""Red-team multi-stream e-process aggregation under asynchronous evidence.

Independent research implementation of the counterexample and adjusters in
Tavyrikov, Goeman & de Heide (EJS 2026). It is not copied from their code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np

SCHEMA = "symthaea-multistream-eprocess-fdr-sup-v1"
LOG2 = math.log(2.0)


def log_adjuster_a1(log_m: float) -> float:
    """log[(M - 1 - log M)/(log M)^2], M=exp(log_m)>=1; A1(1)=1/2."""
    if not math.isfinite(log_m) or log_m < 0.0:
        raise ValueError("log_m must be finite and nonnegative")
    x = log_m
    if x < 1e-3:
        ratio = 0.5 + x/6.0 + x*x/24.0 + x**3/120.0 + x**4/720.0 + x**5/5040.0
        return math.log(ratio)
    return x + math.log1p(-(1.0+x)*math.exp(-x)) - 2.0*math.log(x)


def log_adjuster_a2(log_m: float) -> float:
    """log[sqrt(M)-1], M=exp(log_m)>=1; A2(1)=0."""
    if not math.isfinite(log_m) or log_m < 0.0:
        raise ValueError("log_m must be finite and nonnegative")
    if log_m == 0.0:
        return -math.inf
    x = log_m / 2.0
    if x < 50.0:
        return math.log(math.expm1(x))
    return x + math.log1p(-math.exp(-x))


def adjuster_a1(m: float) -> float:
    if not math.isfinite(m) or m < 1.0:
        raise ValueError("running maximum must be finite and >= 1")
    v = log_adjuster_a1(math.log(m))
    return math.exp(v)


def adjuster_a2(m: float) -> float:
    if not math.isfinite(m) or m < 1.0:
        raise ValueError("running maximum must be finite and >= 1")
    v = log_adjuster_a2(math.log(m))
    return 0.0 if v == -math.inf else math.exp(v)


def ehb_rejections(log_e_values: list[float] | np.ndarray, alpha: float) -> set[int]:
    """e-BH rejection set from log e-values, with stable index tie-breaking."""
    values = [float(x) for x in log_e_values]
    k_total = len(values)
    if k_total == 0 or not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError("need at least one e-value and alpha in (0,1)")
    order = sorted(range(k_total), key=lambda i: (-values[i], i))
    k_star = 0
    for rank in range(1, k_total + 1):
        threshold = math.log(k_total / (alpha * rank))
        if values[order[rank-1]] >= threshold:
            k_star = rank
    return set(order[:k_star])


def adjusted_running_max_ehb(log_running_maxima: list[float] | np.ndarray,
                             alpha: float, adjuster: str = "a1") -> set[int]:
    """e-BH on an admissibly adjusted running maximum (EJS 2026 construction)."""
    fn = log_adjuster_a1 if adjuster == "a1" else log_adjuster_a2 if adjuster == "a2" else None
    if fn is None:
        raise ValueError("adjuster must be 'a1' or 'a2'")
    return ehb_rejections([fn(float(x)) for x in log_running_maxima], alpha)


def e_bonferroni_adjusted(log_running_maxima: list[float] | np.ndarray,
                          alpha_total: float, weights: list[float] | np.ndarray,
                          adjuster: str = "a1") -> set[int]:
    """FWER control by per-stream Ville bounds plus Bonferroni over adjusted maxima."""
    logs = [float(x) for x in log_running_maxima]
    ws = [float(w) for w in weights]
    if len(logs) == 0 or len(logs) != len(ws):
        raise ValueError("need one positive weight per stream")
    if not (math.isfinite(alpha_total) and 0.0 < alpha_total < 1.0):
        raise ValueError("alpha_total must be in (0,1)")
    if any((not math.isfinite(w) or w <= 0.0) for w in ws):
        raise ValueError("all weights must be finite and positive")
    if not math.isclose(math.fsum(ws), 1.0, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("weights must sum to one")
    fn = log_adjuster_a1 if adjuster == "a1" else log_adjuster_a2 if adjuster == "a2" else None
    if fn is None:
        raise ValueError("adjuster must be 'a1' or 'a2'")
    out = set()
    for i, (m, w) in enumerate(zip(logs, ws)):
        adjusted_log = fn(m)
        if adjusted_log >= -math.log(alpha_total*w):
            out.add(i)
    return out


def log_adjuster_vector(log_m: np.ndarray, adjuster: str) -> np.ndarray:
    """Stable vectorized log adjusters for log(M)>=0."""
    x = np.asarray(log_m, dtype=np.float64)
    if np.any(~np.isfinite(x)) or np.any(x < 0.0):
        raise ValueError("log running maxima must be finite and nonnegative")
    if adjuster == "a1":
        out = np.empty_like(x)
        small = x < 1e-3
        z = x[small]
        ratio = 0.5 + z/6.0 + z*z/24.0 + z**3/120.0 + z**4/720.0 + z**5/5040.0
        out[small] = np.log(ratio)
        z = x[~small]
        out[~small] = z + np.log1p(-(1.0+z)*np.exp(-z)) - 2.0*np.log(z)
        return out
    if adjuster == "a2":
        out = np.full_like(x, -np.inf)
        positive = x > 0.0
        z = x[positive]/2.0
        small = z < 50.0
        tmp = np.empty_like(z)
        tmp[small] = np.log(np.expm1(z[small]))
        tmp[~small] = z[~small] + np.log1p(-np.exp(-z[~small]))
        out[positive] = tmp
        return out
    raise ValueError("adjuster must be a1 or a2")


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    p = successes / trials
    den = 1.0 + z*z/trials
    center = (p + z*z/(2*trials))/den
    half = z*math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials))/den
    return [max(0.0, center-half), min(1.0, center+half)]


def simulate_paper_counterexample(*, repetitions: int = 1_000_000, alpha: float = 0.05,
                                  horizon: int = 100, seed: int = 20261030) -> dict:
    """Reproduce the two-dependent-true-null counterexample described in the paper."""
    if repetitions < 1000 or horizon < 1:
        raise ValueError("use at least 1,000 repetitions and a positive horizon")
    if not (0.0 < alpha < 0.5):
        raise ValueError("alpha must be in (0,0.5)")
    rng = np.random.default_rng(seed)
    active = rng.random(repetitions) < 2.0*alpha
    initial = math.log(1.0/(2.0*alpha))
    e1 = np.where(active, initial, -np.inf)
    e2 = e1.copy()
    # Running maxima are clipped at one, the domain of the admissible adjusters.
    m1 = np.maximum(0.0, e1)
    m2 = np.maximum(0.0, e2)
    rejected = {name: np.zeros(repetitions, dtype=np.bool_)
                for name in ("current_ebh", "naive_running_max_ebh",
                             "adjusted_maximum_a1_ebh", "adjusted_maximum_a2_ebh")}
    for _ in range(horizon):
        draw = rng.integers(0, 3, size=repetitions)
        # Joint increments: (1/2,1/2), (2,1/2), (1/2,2), each with probability 1/3.
        e1 += np.where(draw == 1, LOG2, -LOG2)
        e2 += np.where(draw == 2, LOG2, -LOG2)
        m1 = np.maximum(m1, e1)
        m2 = np.maximum(m2, e2)

        current_top = np.maximum(e1, e2)
        current_second = np.minimum(e1, e2)
        rejected["current_ebh"] |= ((current_top >= math.log(2.0/alpha))
                                    | (current_second >= math.log(1.0/alpha)))

        max_top = np.maximum(m1, m2)
        max_second = np.minimum(m1, m2)
        rejected["naive_running_max_ebh"] |= ((max_top >= math.log(2.0/alpha))
                                               | (max_second >= math.log(1.0/alpha)))

        for name, adjuster in (("adjusted_maximum_a1_ebh", "a1"),
                               ("adjusted_maximum_a2_ebh", "a2")):
            a1 = log_adjuster_vector(m1, adjuster)
            a2 = log_adjuster_vector(m2, adjuster)
            top = np.maximum(a1, a2)
            second = np.minimum(a1, a2)
            rejected[name] |= ((top >= math.log(2.0/alpha))
                               | (second >= math.log(1.0/alpha)))

    methods = {}
    for name, mask in rejected.items():
        count = int(mask.sum())
        rate = count/repetitions
        methods[name] = {
            "false_discovery_rate_all_null": rate,
            "false_discoveries": count,
            "repetitions": repetitions,
            "wilson95": wilson_interval(count, repetitions),
            "ratio_to_nominal_alpha": rate/alpha,
        }
    return {
        "schema": SCHEMA,
        "status": "INDEPENDENT_REPRODUCTION_OF_PUBLISHED_COUNTEREXAMPLE_NOT_GENERAL_SOTA_COMPARISON",
        "protocol": {"repetitions": repetitions, "alpha": alpha, "horizon": horizon,
                     "seed": seed, "hypotheses": 2, "truth": "both hypotheses are null",
                     "initial_joint_value": "both 1/(2*alpha) with probability 2*alpha, else both zero",
                     "increment_pairs": [[0.5, 0.5], [2.0, 0.5], [0.5, 2.0]],
                     "increment_pair_probabilities": [1/3, 1/3, 1/3]},
        "methods": methods,
        "interpretation": "In the all-null setting FDR equals the probability of at least one false rejection. Naive e-BH on running maxima is expected to exceed nominal FDR under arbitrary dependence; admissible adjusters recover the published FDR-sup guarantee under its assumptions, often with substantial power cost.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "limitations": [
            "two true-null hypotheses and one paper-specified dependence construction only",
            "Monte Carlo reproduction does not prove the adjuster theorem",
            "no mixed-null power or multi-issuer deployment experiment",
            "the FDR guarantee is not an FWER guarantee in general mixed-null settings",
            "adjusted running maxima need an admissible adjuster and the paper's assumptions",
        ],
    }


def async_order_witness(alpha: float = 0.05) -> dict:
    """A deterministic two-stream witness for non-carefree current e-BH."""
    k = 2
    first = [1.5/alpha, 0.75/alpha]
    both_updated = [0.75/alpha, 1.5/alpha]  # stream 1 halves; stream 2 doubles
    stream1_frozen = [first[0], both_updated[1]]
    def reject(values):
        return sorted(ehb_rejections([math.log(x) for x in values], alpha))
    return {
        "alpha": alpha,
        "time_t": {"e_values": first, "rejections": reject(first)},
        "time_t_plus_1_both_streams_updated": {
            "e_values": both_updated, "rejections": reject(both_updated)},
        "time_t_plus_1_stream_1_frozen": {
            "e_values": stream1_frozen, "rejections": reject(stream1_frozen)},
        "interpretation": "The rejection set depends on whether previously collected evidence is frozen or the process is advanced; the current-value procedure is not carefree.",
        "note": "This is an algebraic decision-order witness; the Monte Carlo counterexample separately establishes FDR inflation for running maxima under the paper's dependent-null construction.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=1_000_000)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    result = simulate_paper_counterexample(repetitions=args.repetitions,
        horizon=args.horizon, seed=args.seed)
    result["async_order_witness"] = async_order_witness()
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({"output": str(args.output), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(), "methods": result["methods"]}, indent=2))


if __name__ == "__main__":
    main()
