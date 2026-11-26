#!/usr/bin/env python3
"""HIST-CAUSAL-001 v0.8 candidate-catalogue adequacy test.

This detector knows the six declared candidate likelihoods and recipe
parameters, but not the hidden mixture likelihood. It flags a dataset only
when the best candidate's likelihood is unusually poor against a
least-favourable parametric-bootstrap threshold calibrated across all models.
Research reference, not a general OOD detector.
"""
from __future__ import annotations
from collections import Counter
from fractions import Fraction as F
from math import ceil, log
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v0.6"))
sys.path.insert(0, str(ROOT / "v0.5"))
import ood_falsification as ood
import generator as base

DEFAULT_N = (1, 2, 5, 10, 20)
DEFAULT_ALPHA = 0.05
DRAW_CONST = 0xD6E8FEB86659FD93
PROBE_CONST = 0x9E3779B97F4A7C15
TRUTH_TAG = {"ood": 0x1234ABCD1234ABCD, "candidate": 0xC3C3C3C3C3C3C3C3}


def _cdf(dist: dict):
    cumulative = F(0)
    out = []
    for outcome, probability in sorted(dist.items()):
        cumulative += probability
        out.append((cumulative, outcome))
    if cumulative != 1:
        raise ArithmeticError("distribution does not sum to one")
    return tuple(out)


def _prepare(params: dict):
    candidate_dist = {m: {p: base._dist(m, p, params) for p in base.PROBES}
                      for m in base.MODELS}
    ood_dist = {p: ood.ood_distribution(p, params) for p in base.PROBES}
    cdfs = {"candidate": {m: {p: _cdf(candidate_dist[m][p]) for p in base.PROBES}
                          for m in base.MODELS},
            "ood": {p: _cdf(ood_dist[p]) for p in base.PROBES}}
    logp = {m: {p: {k: (log(float(v)) if v else float("-inf"))
                        for k, v in candidate_dist[m][p].items()}
                    for p in base.PROBES} for m in base.MODELS}
    return {"candidate_dist": candidate_dist, "ood_dist": ood_dist,
            "cdfs": cdfs, "logp": logp}


def _draw(cdf, seed: int, probe_index: int, draw_id: int, tag: str):
    x = (seed ^ ((probe_index + 1) * PROBE_CONST) ^
         ((draw_id + 1) * DRAW_CONST) ^ TRUTH_TAG[tag]) & base.MASK
    _, raw = base._splitmix64(x)
    u = F(raw, 1 << 64)
    for cumulative, outcome in cdf:
        if u < cumulative:
            return outcome
    return cdf[-1][1]


def sample_dataset(seed: int, replicate: int, n: int, truth: str,
                   model: str | None, prepared: dict) -> dict:
    if truth not in ("candidate", "ood"):
        raise ValueError("truth must be candidate or ood")
    if not isinstance(n, int) or n <= 0 or not isinstance(replicate, int) or replicate < 0:
        raise ValueError("n must be positive and replicate non-negative")
    if truth == "candidate" and model not in base.MODELS:
        raise ValueError("candidate model must be declared")
    data = {}
    for probe_index, probe in enumerate(base.PROBES):
        cdf = (prepared["cdfs"]["ood"][probe] if truth == "ood"
               else prepared["cdfs"]["candidate"][model][probe])
        data[probe] = [_draw(cdf, seed, probe_index, replicate * n + j, truth)
                       for j in range(n)]
    return data


def log_likelihood(data: dict, model: str, prepared: dict) -> float:
    total = 0.0
    table = prepared["logp"][model]
    for probe, observations in data.items():
        counts = Counter(observations)
        for outcome, count in counts.items():
            probability_log = table[probe].get(outcome, float("-inf"))
            if probability_log == float("-inf"):
                return float("-inf")
            total += count * probability_log
    return total


def best_catalogue_loglik(data: dict, prepared: dict) -> tuple[float, str]:
    scores = {m: log_likelihood(data, m, prepared) for m in base.MODELS}
    model = max(base.MODELS, key=lambda m: (scores[m], m))
    return scores[model], model


def _lower_order_statistic(values: list[float], alpha: float) -> float:
    if not values:
        raise ValueError("cannot calibrate from an empty sample")
    ordered = sorted(values)
    rank = max(1, min(len(ordered), ceil(alpha * (len(ordered) + 1))))
    return ordered[rank - 1]


def calibrate_threshold(seed: int, n: int, calibration_reps: int,
                        prepared: dict, alpha: float = DEFAULT_ALPHA) -> dict:
    if calibration_reps < 20:
        raise ValueError("at least 20 calibration replicates required")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be between zero and one half")
    thresholds = {}
    for model_index, model in enumerate(base.MODELS):
        scores = []
        for rep in range(calibration_reps):
            data = sample_dataset(seed + model_index * 100003, rep, n,
                                  "candidate", model, prepared)
            score, _ = best_catalogue_loglik(data, prepared)
            scores.append(score)
        thresholds[model] = _lower_order_statistic(scores, alpha)
    return {"threshold": min(thresholds.values()),
            "per_model_thresholds": thresholds,
            "alpha_target": alpha,
            "calibration_reps_per_model": calibration_reps}


def evaluate_seed(seed: int, n_values=DEFAULT_N, calibration_reps: int = 200,
                  evaluation_reps: int = 200, alpha: float = DEFAULT_ALPHA) -> dict:
    params = base.generate_recipe(seed)["_params"]
    prepared = _prepare(params)
    results = {}
    for n in n_values:
        cal = calibrate_threshold(seed + 5000003, n, calibration_reps, prepared, alpha)
        counts = {"ood": {"flagged_misspecified": 0, "accepted_catalogue": 0},
                  "candidate": {m: {"flagged_misspecified": 0, "accepted_catalogue": 0}
                                for m in base.MODELS}}
        for rep in range(evaluation_reps):
            data = sample_dataset(seed + 7000001, rep, n, "ood", None, prepared)
            score, _ = best_catalogue_loglik(data, prepared)
            label = "flagged_misspecified" if score < cal["threshold"] else "accepted_catalogue"
            counts["ood"][label] += 1
            for model_index, model in enumerate(base.MODELS):
                data = sample_dataset(seed + 9000011 + model_index * 65537,
                                      rep, n, "candidate", model, prepared)
                score, _ = best_catalogue_loglik(data, prepared)
                label = "flagged_misspecified" if score < cal["threshold"] else "accepted_catalogue"
                counts["candidate"][model][label] += 1
        candidate_n = evaluation_reps * len(base.MODELS)
        results[str(n)] = {
            "threshold": cal["threshold"],
            "calibration_thresholds_by_candidate": cal["per_model_thresholds"],
            "ood": {k: v / evaluation_reps for k, v in counts["ood"].items()},
            "candidate_pooled": {
                "flagged_misspecified": sum(v["flagged_misspecified"] for v in counts["candidate"].values()) / candidate_n,
                "accepted_catalogue": sum(v["accepted_catalogue"] for v in counts["candidate"].values()) / candidate_n,
            },
            "candidate_by_model": {
                m: {k: v / evaluation_reps for k, v in vals.items()}
                for m, vals in counts["candidate"].items()
            }
        }
    return {"seed": seed, "alpha": alpha, "calibration_reps": calibration_reps,
            "evaluation_reps": evaluation_reps, "results": results}


def evaluate(seed_count: int = 12, n_values=DEFAULT_N,
             calibration_reps: int = 200, evaluation_reps: int = 200,
             alpha: float = DEFAULT_ALPHA) -> dict:
    if seed_count <= 0 or evaluation_reps <= 0:
        raise ValueError("seed_count and evaluation_reps must be positive")
    if not n_values or any(not isinstance(n, int) or n <= 0 for n in n_values):
        raise ValueError("n_values must contain positive integers")
    if len(set(n_values)) != len(n_values):
        raise ValueError("n_values must be unique")
    rows = [evaluate_seed(seed, n_values, calibration_reps, evaluation_reps, alpha)
            for seed in range(seed_count)]
    aggregate = {}
    for n in n_values:
        ood_rate = sum(r["results"][str(n)]["ood"]["flagged_misspecified"] for r in rows) / seed_count
        cand_rate = sum(r["results"][str(n)]["candidate_pooled"]["flagged_misspecified"] for r in rows) / seed_count
        by_model = {m: sum(r["results"][str(n)]["candidate_by_model"][m]["flagged_misspecified"] for r in rows) / seed_count
                    for m in base.MODELS}
        aggregate[str(n)] = {"ood_detection_rate": ood_rate,
                             "in_family_false_ood_rate_pooled": cand_rate,
                             "in_family_false_ood_rate_by_model": by_model}
    return {"benchmark": "HIST-CAUSAL-001", "version": "0.8",
            "method": "least-favourable parametric-bootstrap catalogue adequacy; known candidate likelihoods, unknown OOD likelihood",
            "seed_count": seed_count, "calibration_reps_per_model": calibration_reps,
            "evaluation_reps_per_truth_per_model": evaluation_reps,
            "alpha_target": alpha, "samples_per_probe": list(n_values),
            "probe_count": len(base.PROBES), "aggregate": aggregate,
            "claim_ceiling": "finite known-parameter catalogue adequacy baseline; not general open-world detection"}


if __name__ == "__main__":
    import json
    print(json.dumps(evaluate(), indent=2))
