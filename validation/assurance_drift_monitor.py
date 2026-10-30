#!/usr/bin/env python3
"""Anytime-valid monitoring of assumption-issuer failure rates under shift.

Research benchmark only. It compares a static historical assumption, a
predeclared Bonferroni rolling-binomial monitor, and a mixture e-process.
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform
from functools import lru_cache
from pathlib import Path
import numpy as np

SCHEMA = "symthaea-anytime-assurance-drift-monitor-v1"


def validate_probability(name: str, p: float, *, strict: bool = False) -> float:
    if not math.isfinite(p) or p < 0.0 or p > 1.0 or (strict and p in (0.0, 1.0)):
        raise ValueError(f"{name} must be a valid probability")
    return p


@lru_cache(maxsize=4096)
def binomial_upper_tail(k: int, n: int, p: float) -> float:
    """Exact P[X >= k] for X~Binomial(n,p), computed with log-sum-exp."""
    if n < 0 or k < 0 or k > n:
        raise ValueError("require n >= 0 and 0 <= k <= n")
    validate_probability("p", p)
    if k == 0:
        return 1.0
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    terms = []
    lp, lq = math.log(p), math.log1p(-p)
    for i in range(k, n + 1):
        terms.append(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n-i) * lq)
    peak = max(terms)
    return min(1.0, math.exp(peak) * math.fsum(math.exp(t - peak) for t in terms))


def log_mixture_e_value(observations: list[int] | np.ndarray, null_rate: float,
                        alternatives: tuple[float, ...]) -> float:
    """Log e-value for H0: conditional failure probability <= null_rate.

    Each fixed alternative q > p0 contributes LR q^S(1-q)^(n-S) /
    [p0^S(1-p0)^(n-S)]. Its product is a supermartingale under the composite
    conditional null P(X_t=1 | past) <= p0. A positive mixture preserves this.
    """
    p0 = validate_probability("null_rate", null_rate, strict=True)
    if not alternatives or any(not math.isfinite(q) or not p0 < q < 1.0 for q in alternatives):
        raise ValueError("all alternatives must lie strictly above null_rate and below 1")
    arr = np.asarray(observations, dtype=np.int8)
    if arr.ndim != 1 or np.any((arr != 0) & (arr != 1)):
        raise ValueError("observations must be a 1D binary vector")
    s, n = int(arr.sum()), int(arr.size)
    log_e = [s * math.log(q / p0) + (n-s) * math.log((1.0-q) / (1.0-p0)) for q in alternatives]
    max_log = max(log_e)
    return max_log + math.log(math.fsum(math.exp(x-max_log) for x in log_e) / len(log_e))


def monitor_e_process(observations: list[int] | np.ndarray, *, null_rate: float,
                      alternatives: tuple[float, ...], alpha: float) -> dict:
    if not (math.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    for n in range(1, len(observations)+1):
        le = log_mixture_e_value(observations[:n], null_rate, alternatives)
        if le >= math.log(1.0 / alpha):
            return {"alarm": True, "time": n, "log_e": le, "threshold_log_e": math.log(1.0/alpha)}
    last = log_mixture_e_value(observations, null_rate, alternatives) if len(observations) else 0.0
    return {"alarm": False, "time": None, "log_e": last, "threshold_log_e": math.log(1.0/alpha)}


def monitor_bonferroni_rolling(observations: list[int] | np.ndarray, *, null_rate: float,
                               alpha: float, window: int = 50, step: int = 25) -> dict:
    """Fixed-look exact tests with Bonferroni over a horizon known in advance."""
    if window < 1 or step < 1 or not (0.0 < alpha < 1.0):
        raise ValueError("invalid window, step, or alpha")
    arr = np.asarray(observations, dtype=np.int8)
    if arr.ndim != 1 or np.any((arr != 0) & (arr != 1)):
        raise ValueError("observations must be a 1D binary vector")
    looks = list(range(window, len(arr)+1, step))
    if not looks or looks[-1] != len(arr):
        if len(arr) >= window:
            looks.append(len(arr))
    look_count = len(looks)
    threshold = alpha / max(1, look_count)
    for t in looks:
        sample = arr[t-window:t]
        p_value = binomial_upper_tail(int(sample.sum()), window, null_rate)
        if p_value <= threshold:
            return {"alarm": True, "time": t, "p_value": p_value,
                    "per_look_alpha": threshold, "look_count": look_count}
    return {"alarm": False, "time": None, "p_value": None,
            "per_look_alpha": threshold, "look_count": look_count}


def preflight_scope(*, evidence_scope: str, requested_scope: str,
                    evidence_epoch: str, requested_epoch: str, expired: bool) -> dict:
    if expired:
        return {"status": "refuse_unresolved", "reason": "evidence_expired"}
    if not evidence_scope or evidence_scope != requested_scope:
        return {"status": "refuse_unresolved", "reason": "scope_mismatch_or_missing"}
    if not evidence_epoch or evidence_epoch != requested_epoch:
        return {"status": "refuse_unresolved", "reason": "epoch_mismatch_or_missing"}
    return {"status": "continue_monitoring", "reason": "scope_epoch_valid"}


def generate_stream(rng, *, horizon: int, change_after: int, baseline: float,
                    future: float, scenario: str) -> np.ndarray:
    out = np.empty(horizon, dtype=np.int8)
    for i in range(horizon):
        if scenario == "stable":
            p = baseline
        elif scenario == "abrupt":
            p = baseline if i < change_after else future
        elif scenario == "gradual":
            if i < change_after:
                p = baseline
            else:
                frac = min(1.0, (i-change_after+1)/100.0)
                p = baseline + frac * (future-baseline)
        else:
            raise ValueError(f"unknown scenario: {scenario}")
        out[i] = rng.binomial(1, p)
    return out


def _first_alarm_e(arr, p0, alts, alpha):
    """Incremental implementation of the same mixture e-process."""
    logs = np.zeros(len(alts), dtype=float)
    threshold = math.log(1.0/alpha)
    for i, x in enumerate(arr, start=1):
        for j, q in enumerate(alts):
            logs[j] += math.log(q/p0) if x else math.log((1-q)/(1-p0))
        m = float(np.max(logs))
        log_mix = m + math.log(float(np.exp(logs-m).sum())/len(alts))
        if log_mix >= threshold:
            return i, log_mix
    m = float(np.max(logs))
    return None, m + math.log(float(np.exp(logs-m).sum())/len(alts))


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    p = successes / trials
    den = 1.0 + z*z/trials
    center = (p + z*z/(2*trials)) / den
    half = z*math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / den
    return [max(0.0, center-half), min(1.0, center+half)]


def run_benchmark(*, repeats: int = 2000, horizon: int = 500, change_after: int = 100,
                  baseline: float = 0.01, future_rates=(0.05, 0.10), alpha: float = 0.01,
                  window: int = 50, step: int = 25, seed: int = 20261030) -> dict:
    if repeats <= 0 or horizon <= change_after or change_after < window:
        raise ValueError("need repeats>0 and horizon>change_after>=window")
    validate_probability("baseline", baseline, strict=True)
    if any(not baseline < p < 1.0 for p in future_rates):
        raise ValueError("future rates must exceed baseline and be less than 1")
    alts = tuple(sorted(set([min(0.99, baseline*1.5), 0.03, 0.05, 0.10, 0.20])))
    alts = tuple(q for q in alts if q > baseline and q < 1.0)
    rng = np.random.default_rng(seed)
    scenarios = [("stable", baseline), ("abrupt", future_rates[0]), ("abrupt", future_rates[1]), ("gradual", future_rates[1])]
    results = {}
    for scenario, future in scenarios:
        rows = {name: {"false_alarm": 0, "detected_after_change": 0, "missed": 0, "delays": [], "alarm_times": []}
                for name in ("static_reuse", "mixture_e_process", "bonferroni_rolling")}
        for _ in range(repeats):
            arr = generate_stream(rng, horizon=horizon, change_after=change_after,
                                  baseline=baseline, future=future, scenario=scenario)
            e_time, _ = _first_alarm_e(arr, baseline, alts, alpha)
            b = monitor_bonferroni_rolling(arr, null_rate=baseline, alpha=alpha, window=window, step=step)
            alarms = {"static_reuse": None, "mixture_e_process": e_time, "bonferroni_rolling": b["time"]}
            for name, alarm_time in alarms.items():
                row = rows[name]
                if alarm_time is not None:
                    row["alarm_times"].append(alarm_time)
                if scenario == "stable":
                    row["false_alarm"] += int(alarm_time is not None)
                else:
                    if alarm_time is not None and alarm_time <= change_after:
                        row["false_alarm"] += 1
                    elif alarm_time is not None:
                        row["detected_after_change"] += 1
                        row["delays"].append(alarm_time-change_after)
                    else:
                        row["missed"] += 1
        summary = {}
        for name, row in rows.items():
            d = row["delays"]
            at = row["alarm_times"]
            false_rate = row["false_alarm"]/repeats
            if scenario == "stable":
                detection_count = detection_rate = detection_interval = None
                miss_count = miss_rate = miss_interval = None
            else:
                detection_count = row["detected_after_change"]
                miss_count = row["missed"]
                detection_rate = detection_count/repeats
                miss_rate = miss_count/repeats
                detection_interval = wilson_interval(detection_count, repeats)
                miss_interval = wilson_interval(miss_count, repeats)
            summary[name] = {
                "false_alarm_count": row["false_alarm"],
                "false_alarm_rate": false_rate,
                "false_alarm_wilson95": wilson_interval(row["false_alarm"], repeats),
                "post_change_detection_count": detection_count,
                "post_change_detection_rate": detection_rate,
                "post_change_detection_wilson95": detection_interval,
                "miss_count": miss_count, "miss_rate": miss_rate, "miss_wilson95": miss_interval,
                "mean_delay_given_detection": float(np.mean(d)) if d else None,
                "median_delay_given_detection": float(np.median(d)) if d else None,
                "p95_delay_given_detection": float(np.quantile(d, .95)) if d else None,
                "total_alarm_rate": len(at)/repeats,
            }
        results[f"{scenario}_future_{future:.3f}"] = summary
    return {
        "schema": SCHEMA,
        "status": "SYNTHETIC_ANYTIME_MONITOR_BENCHMARK_NOT_PRODUCTION_VALIDATION",
        "protocol": {"repeats": repeats, "horizon": horizon, "change_after": change_after,
                     "baseline_rate": baseline, "future_rates": list(future_rates),
                     "alpha_anytime": alpha, "rolling_window": window, "rolling_step": step,
                     "rolling_look_correction": "Bonferroni over all predeclared looks", "seed": seed,
                     "mixture_alternatives": list(alts)},
        "results": results,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            "synthetic Bernoulli audit labels; no real issuer/audit stream",
            "e-process false-alarm guarantee requires each null conditional failure probability to be <= baseline rate and outcomes to be valid observations",
            "rolling binomial tests assume a Bernoulli null for every fixed window; Bonferroni handles the planned repeated looks, not invalid dependence assumptions",
            "the static baseline demonstrates cost of no monitoring, not a comparable detector",
            "no monitor detects every arbitrary shift and detection cannot authenticate labels",
        ],
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=2000)
    p.add_argument("--horizon", type=int, default=500)
    a = p.parse_args()
    result = run_benchmark(repeats=a.repeats, horizon=a.horizon)
    raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_bytes(raw)
    print(json.dumps({"output": str(a.output), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(), "results": result["results"]}, indent=2))


if __name__ == "__main__":
    main()
