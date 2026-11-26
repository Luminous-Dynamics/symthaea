#!/usr/bin/env python3
"""HIST-CAUSAL-001 v0.7 finite-sample OOD detection ceiling.

An oracle likelihood-ratio comparator knows the OOD mixture and nearest
in-family model. This is an optimistic pairwise reference, not an agent score.
Exact Fraction probabilities are retained; logarithms are used for LLR sums.
"""
from __future__ import annotations
from fractions import Fraction as F
from math import log, sqrt
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if not (ROOT / "v0.5").is_dir():
    ROOT = ROOT / "HIST_CAUSAL_001_v04_bundle"
sys.path.insert(0, str(ROOT / "v0.6"))
sys.path.insert(0, str(ROOT / "v0.5"))
import ood_falsification as ood
import generator as base

LR_THRESHOLD = 19.0
DEFAULT_N = (1, 2, 5, 10, 20, 50)
DRAW_CONST = 0xD6E8FEB86659FD93
TRUTH_SALT = {"ood": 0x1234ABCD1234ABCD, "candidate": 0xC3C3C3C3C3C3C3C3}


def _kl(p: dict, q: dict) -> float:
    value = 0.0
    for outcome, pp in p.items():
        if pp == 0:
            continue
        qq = q.get(outcome, F(0))
        if qq == 0:
            return float("inf")
        value += float(pp) * log(float(pp / qq))
    return value


def jensen_shannon(p: dict, q: dict) -> float:
    keys = set(p) | set(q)
    midpoint = {k: (p.get(k, F(0)) + q.get(k, F(0))) / 2 for k in keys}
    return 0.5 * _kl(p, midpoint) + 0.5 * _kl(q, midpoint)


def nearest_candidate(seed: int) -> tuple[str, float]:
    params = base.generate_recipe(seed)["_params"]
    scores = {}
    for model in base.MODELS:
        scores[model] = sum(
            jensen_shannon(ood.ood_distribution(probe, params), base._dist(model, probe, params))
            for probe in base.PROBES
        ) / len(base.PROBES)
    best = min(scores, key=lambda m: (scores[m], m))
    return best, scores[best]


def _prepared_pair(seed: int, candidate: str):
    params = base.generate_recipe(seed)["_params"]
    prepared = {}
    for probe in base.PROBES:
        p_ood = ood.ood_distribution(probe, params)
        p_candidate = base._dist(candidate, probe, params)
        outcomes = sorted(set(p_ood) | set(p_candidate))
        ood_cdf = []
        cand_cdf = []
        cumulative = F(0)
        for outcome in outcomes:
            cumulative += p_ood.get(outcome, F(0))
            ood_cdf.append((cumulative, outcome))
        cumulative = F(0)
        for outcome in outcomes:
            cumulative += p_candidate.get(outcome, F(0))
            cand_cdf.append((cumulative, outcome))
        llr = {}
        for outcome in outcomes:
            po = p_ood.get(outcome, F(0))
            pc = p_candidate.get(outcome, F(0))
            if po == 0 and pc > 0:
                llr[outcome] = float("-inf")
            elif pc == 0 and po > 0:
                llr[outcome] = float("inf")
            elif po == 0 and pc == 0:
                continue
            else:
                llr[outcome] = log(float(po / pc))
        salt = sum((i + 1) * b for i, b in enumerate(probe.encode()))
        prepared[probe] = (ood_cdf, cand_cdf, llr, salt)
    return prepared


def _draw(cdf, seed: int, salt: int, draw_id: int, truth: str):
    _, raw = base._splitmix64((seed ^ salt ^ ((draw_id + 1) * DRAW_CONST) ^ TRUTH_SALT[truth]) & base.MASK)
    u = F(raw, 1 << 64)
    for cumulative, outcome in cdf:
        if u < cumulative:
            return outcome
    return cdf[-1][1]


def classify_llr(log_bf: float) -> str:
    if log_bf >= log(LR_THRESHOLD):
        return "ood"
    if log_bf <= -log(LR_THRESHOLD):
        return "candidate"
    return "abstain"


def _simulate_prepared(seed: int, n: int, replicate: int, truth: str, prepared: dict) -> float:
    total = 0.0
    for probe, (ood_cdf, cand_cdf, llr, salt) in prepared.items():
        cdf = ood_cdf if truth == "ood" else cand_cdf
        for j in range(n):
            draw_id = replicate * n + j
            outcome = _draw(cdf, seed, salt, draw_id, truth)
            total += llr[outcome]
    return total


def trial_log_bayes_factor(seed: int, sample_count_per_probe: int, replicate: int,
                           truth: str, candidate: str) -> float:
    if not isinstance(sample_count_per_probe, int) or sample_count_per_probe <= 0:
        raise ValueError("sample_count_per_probe must be a positive integer")
    if not isinstance(replicate, int) or replicate < 0:
        raise ValueError("replicate must be a non-negative integer")
    if truth not in TRUTH_SALT:
        raise ValueError("truth must be 'ood' or 'candidate'")
    prepared = _prepared_pair(seed, candidate)
    return _simulate_prepared(seed, sample_count_per_probe, replicate, truth, prepared)


def _wilson_interval(successes: int, n: int, z: float = 1.959963984540054):
    """Marginal 95% Wilson interval for a Monte Carlo proportion."""
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n))) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def evaluate(seed_count: int = 20, n_values=DEFAULT_N, replicates: int = 250) -> dict:
    if not isinstance(seed_count, int) or seed_count <= 0:
        raise ValueError("seed_count must be a positive integer")
    if not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("replicates must be a positive integer")
    if not n_values or any(not isinstance(n, int) or n <= 0 for n in n_values):
        raise ValueError("n_values must contain positive integers")
    if len(set(n_values)) != len(n_values):
        raise ValueError("n_values must be unique")
    aggregate = {n: {truth: {"correct_confident": 0, "wrong_confident": 0, "abstain": 0}
                     for truth in ("ood", "candidate")} for n in n_values}
    nearest_counts = {}
    js_values = []
    trials = 0
    for seed in range(seed_count):
        candidate, js_value = nearest_candidate(seed)
        nearest_counts[candidate] = nearest_counts.get(candidate, 0) + 1
        js_values.append(js_value)
        prepared = _prepared_pair(seed, candidate)
        for n in n_values:
            for truth in ("ood", "candidate"):
                for rep in range(replicates):
                    decision = classify_llr(_simulate_prepared(seed, n, rep, truth, prepared))
                    if decision == "abstain":
                        aggregate[n][truth]["abstain"] += 1
                    elif decision == truth:
                        aggregate[n][truth]["correct_confident"] += 1
                    else:
                        aggregate[n][truth]["wrong_confident"] += 1
                    trials += 1
    rows = {}
    for n in n_values:
        rows[str(n)] = {}
        for truth, counts in aggregate[n].items():
            total = sum(counts.values())
            rows[str(n)][truth] = {
                "n_trials": total,
                "rates": {k: v / total for k, v in counts.items()},
                "marginal_95pct_wilson_intervals": {k: _wilson_interval(v, total) for k, v in counts.items()},
            }
    return {
        "benchmark": "HIST-CAUSAL-001",
        "version": "0.7",
        "method": "oracle pairwise likelihood-ratio reference; known OOD and nearest-candidate likelihoods",
        "seed_count": seed_count,
        "replicates_per_seed_per_truth_per_n": replicates,
        "independent_samples_per_probe": list(n_values),
        "probe_count": len(base.PROBES),
        "equal_prior_confident_posterior_threshold": "0.95",
        "likelihood_ratio_threshold": 19,
        "nearest_candidate_counts": nearest_counts,
        "mean_nearest_pair_js_nats": sum(js_values) / len(js_values),
        "outcomes": rows,
        "trials_scored": trials,
        "claim_ceiling": "optimistic pairwise detection sensitivity with known likelihoods; not a solver result or general OOD guarantee",
    }


if __name__ == "__main__":
    import json
    print(json.dumps(evaluate(), indent=2))
