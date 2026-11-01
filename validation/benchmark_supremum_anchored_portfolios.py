"""Anytime benchmark for fixed-prior portfolios of anchored e-process experts.

Synthetic Bernoulli likelihood-ratio experts are valid e-processes under the
registered null p0. Each expert's running maximum is transformed by an
admissible power adjuster A_beta. A coordinate-local weighted e-Bonferroni
baseline supplies q_i; auxiliary experts receive v_i. The fixed prior is
chosen ex ante. e-BH is run at every time and FDR-sup is measured.

This is a simulation of the additive special case, not a proof. The guarantee
requires the coordinate nulls, ex-ante q_i/v_i budgets, valid likelihood-ratio
e-processes, and admissible adjusters exactly as documented.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from typing import Any

import numpy as np


def _log_adjuster_from_log_max(log_max: np.ndarray, beta: float) -> np.ndarray:
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must lie in [0,1)")
    out = np.full_like(log_max, -np.inf, dtype=np.float64)
    positive = log_max > 0.0
    if beta == 0.0:
        out[positive] = np.log(log_max[positive])
        return out
    z = beta * log_max[positive]
    vals = np.empty_like(z)
    small = z < 50.0
    vals[small] = np.log(np.expm1(z[small]))
    vals[~small] = z[~small] + np.log1p(-np.exp(-z[~small]))
    out[positive] = math.log((1.0 - beta) / beta) + vals
    return out


def _log_lr(x: np.ndarray, p1: float, p0: float) -> np.ndarray:
    if not 0.0 < p0 < 1.0 or not 0.0 < p1 < 1.0:
        raise ValueError("Bernoulli probabilities must lie strictly inside (0,1)")
    return np.where(
        x,
        math.log(p1 / p0),
        math.log((1.0 - p1) / (1.0 - p0)),
    )


def _ebh(log_e: np.ndarray, alpha: float) -> np.ndarray:
    """Vectorized e-BH on rows of log e-values; returns selected mask."""
    streams, n = log_e.shape
    order = np.argsort(-log_e, axis=1)
    sorted_log = np.take_along_axis(log_e, order, axis=1)
    cutoff = -np.log(alpha * np.arange(1, n + 1, dtype=np.float64))
    k_star = (sorted_log >= cutoff[None, :]).sum(axis=1)
    rank = np.empty_like(order)
    rank[np.arange(streams)[:, None], order] = np.arange(n)[None, :]
    return rank < k_star[:, None]


def _summarize_path(
    log_e: np.ndarray,
    baseline_rejected: np.ndarray,
    truth_null: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    streams, n, horizon = log_e.shape
    sup_fdp = np.zeros(streams, dtype=np.float64)
    baseline_retained = np.ones(streams, dtype=bool)
    final_rejected = np.zeros((streams, n), dtype=bool)
    fdp_by_time = []
    for t in range(horizon):
        rejected = _ebh(log_e[:, :, t], alpha)
        total = rejected.sum(axis=1)
        false = (rejected & truth_null[None, :]).sum(axis=1)
        fdp = false / np.maximum(total, 1)
        sup_fdp = np.maximum(sup_fdp, fdp)
        baseline_retained &= np.all(
            (~baseline_rejected[:, :, t]) | rejected, axis=1
        )
        fdp_by_time.append(float(fdp.mean()))
        if t == horizon - 1:
            final_rejected = rejected
    alternatives = ~truth_null
    n_alt = int(alternatives.sum())
    true_rejections = (final_rejected & alternatives[None, :]).sum(axis=1)
    power = float(true_rejections.mean() / n_alt) if n_alt else 0.0
    se = float(sup_fdp.std(ddof=1) / math.sqrt(streams)) if streams > 1 else 0.0
    return {
        "estimated_supfdr": float(sup_fdp.mean()),
        "supfdr_monte_carlo_se": se,
        "final_power": power,
        "mean_final_rejections": float(final_rejected.sum(axis=1).mean()),
        "mean_final_true_discoveries": float(true_rejections.mean()),
        "baseline_retention_rate": float(baseline_retained.mean()),
        "max_timewise_fdr_estimate": max(fdp_by_time, default=0.0),
    }


def simulate_scenario(
    *,
    seed: int,
    n: int,
    streams: int,
    horizon: int,
    alpha: float,
    p0: float,
    p_alt: float,
    nonnull_fraction: float,
    q_total: float,
    v_total: float,
    baseline_p1: float,
    expert_p1: tuple[float, ...] = (0.01, 0.05, 0.20, 0.80),
    betas: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75),
) -> dict[str, Any]:
    if n < 2 or streams < 1 or horizon < 1:
        raise ValueError("n, streams and horizon must be positive")
    if not 0.0 < alpha < 1.0 or not 0.0 < q_total < 1.0:
        raise ValueError("invalid alpha or q_total")
    if not 0.0 < v_total or q_total + v_total > 1.0:
        raise ValueError("require 0 < v_total and q_total+v_total <= 1")
    if len(expert_p1) != len(betas) or not expert_p1:
        raise ValueError("one beta is required per auxiliary expert")
    if not 0.0 <= nonnull_fraction <= 1.0:
        raise ValueError("nonnull_fraction must lie in [0,1]")

    rng = np.random.default_rng(seed)
    n_alt = int(round(n * nonnull_fraction))
    alt_mask = np.zeros(n, dtype=bool)
    if n_alt:
        alt_mask[:n_alt] = True
    null_mask = ~alt_mask
    probabilities = np.where(alt_mask, p_alt, p0)
    observations = rng.random((streams, n, horizon)) < probabilities[None, :, None]

    q_i = q_total / n
    v_i = v_total / n
    log_baseline = np.cumsum(
        _log_lr(observations, baseline_p1, p0), axis=2
    )
    baseline_threshold = math.log(1.0 / (alpha * q_i))
    baseline_rejected = np.maximum.accumulate(
        log_baseline >= baseline_threshold, axis=2
    )

    log_adjusted_experts = []
    for p1, beta in zip(expert_p1, betas):
        raw_log = np.cumsum(_log_lr(observations, p1, p0), axis=2)
        running_log_max = np.maximum.accumulate(
            np.maximum(raw_log, 0.0), axis=2
        )
        log_adjusted_experts.append(
            _log_adjuster_from_log_max(running_log_max, beta)
        )
    log_adjusted = np.stack(log_adjusted_experts, axis=0)
    prior = np.full(len(expert_p1), 1.0 / len(expert_p1), dtype=np.float64)
    log_auxiliary_weight = math.log(v_i)

    log_baseline_term = np.where(
        baseline_rejected,
        -math.log(alpha),
        -np.inf,
    )
    methods: dict[str, np.ndarray] = {}
    for index, (p1, beta) in enumerate(zip(expert_p1, betas)):
        methods[f"single_expert_p1_{p1}_beta_{beta}"] = np.logaddexp(
            log_baseline_term,
            log_auxiliary_weight + log_adjusted[index],
        )

    mixture_log_aux = np.logaddexp.reduce(
        log_adjusted
        + np.log(prior)[:, None, None, None]
        + log_auxiliary_weight,
        axis=0,
    )
    methods["fixed_prior_expert_portfolio"] = np.logaddexp(
        log_baseline_term, mixture_log_aux
    )

    output = {
        name: _summarize_path(path, baseline_rejected, null_mask, alpha)
        for name, path in methods.items()
    }
    oracle_name = max(
        (name for name in output if name.startswith("single_expert_")),
        key=lambda name: output[name]["final_power"],
    )
    return {
        "scenario": {
            "seed": seed,
            "n": n,
            "streams": streams,
            "horizon": horizon,
            "alpha": alpha,
            "p0": p0,
            "p_alt": p_alt,
            "nonnull_fraction": nonnull_fraction,
            "nonnull_count": n_alt,
            "q_total": q_total,
            "v_total": v_total,
            "baseline_p1": baseline_p1,
            "auxiliary_p1": list(expert_p1),
            "adjuster_betas": list(betas),
            "fixed_prior": [1.0 / len(betas)] * len(betas),
        },
        "methods": output,
        "posthoc_best_single_expert": oracle_name,
        "interpretation": (
            "The fixed-prior expert portfolio is the valid comparator. "
            "The post-hoc best single expert is reported only as an unattainable "
            "diagnostic ceiling; it is not a valid data-dependent procedure."
        ),
    }


def run_benchmark(
    *,
    seed: int = 97802,
    n: int = 100,
    streams: int = 5000,
    horizon: int = 30,
    alpha: float = 0.05,
) -> dict[str, Any]:
    scenarios = [
        ("all_null", 0.0, 0.0),
        ("sparse_10pct_mild", 0.10, 0.05),
        ("sparse_10pct_strong", 0.10, 0.40),
        ("dense_70pct", 0.70, 0.20),
    ]
    rows = [
        simulate_scenario(
            seed=seed + 101 * index,
            n=n,
            streams=streams,
            horizon=horizon,
            alpha=alpha,
            p0=0.001,
            p_alt=p_alt,
            nonnull_fraction=fraction,
            q_total=0.10,
            v_total=0.90,
            baseline_p1=0.20,
        )
        for index, (_, fraction, p_alt) in enumerate(scenarios)
    ]
    return {
        "benchmark": "projective_anchored_fixed_prior_expert_portfolios",
        "version": 1,
        "master_seed": seed,
        "n": n,
        "streams_per_scenario": streams,
        "horizon": horizon,
        "alpha": alpha,
        "coordinate_null_risk_budget": "q_i=q_total/n",
        "coordinate_auxiliary_budget": "v_i=v_total/n per expert",
        "budget_contract": "sum_i(q_i+v_i)=1 for each expert",
        "scenarios": rows,
        "limitations": [
            "Synthetic Bernoulli likelihood-ratio model; not a comparison with all industry procedures.",
            "The guarantee relies on valid coordinate e-processes and admissible running-maximum adjusters.",
            "This is the additive anchored special case; no intersection-specific joint e-value is used here.",
            "Monte Carlo FDR-sup estimates do not prove the theorem.",
            "The post-hoc best expert is invalid and reported only as a diagnostic ceiling.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=97802)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--streams", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--output", type=str, default="-")
    args = parser.parse_args()
    result = run_benchmark(
        seed=args.seed, n=args.n, streams=args.streams, horizon=args.horizon
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output == "-":
        print(rendered)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
