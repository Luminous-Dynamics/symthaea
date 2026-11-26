#!/usr/bin/env python3
"""Exploratory rank-geometric staircase calibration benchmark.

Research-only simulation. This is not production code and does not replace
proofs, exact Rust tests, or frozen confirmatory experiments.

Dependencies: NumPy.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from typing import Any

import numpy as np


def balanced_bernoulli(
    rng: np.random.Generator, n_streams: int, n_coords: int, probability: float
) -> np.ndarray:
    """Fresh within-round dependent vector with exact per-coordinate marginal p."""
    if n_coords == 0:
        return np.zeros((n_streams, 0), dtype=np.int8)
    expected = n_coords * probability
    lower = math.floor(expected)
    fraction = expected - lower
    extra = (
        (rng.random(n_streams) < fraction).astype(int)
        if fraction
        else np.zeros(n_streams, dtype=int)
    )
    counts = lower + extra
    order = np.argsort(rng.random((n_streams, n_coords)), axis=1)
    output = np.zeros((n_streams, n_coords), dtype=np.int8)
    rows = np.arange(n_streams)[:, None]
    selected = np.arange(n_coords)[None, :] < counts[:, None]
    output[rows, order] = selected.astype(np.int8)
    return output


def e_bh_rejections(values: np.ndarray, alpha: float) -> np.ndarray:
    """Ordinary e-BH step-up rejection masks; rows are independent streams."""
    n_streams, family_size = values.shape
    ordered = np.sort(values, axis=1)[:, ::-1]
    ranks = np.arange(1, family_size + 1, dtype=float)
    qualifies = ordered >= (family_size / (alpha * ranks))[None, :]
    selected_rank = np.where(qualifies, ranks[None, :], 0).max(axis=1)
    threshold = np.where(
        selected_rank > 0,
        family_size / (alpha * np.maximum(selected_rank, 1)),
        np.inf,
    )
    return values >= threshold[:, None]


def rank_geometric_points(family_size: int, alpha: float) -> list[tuple[float, float]]:
    """Mirror the Rust builder's exact fixed-point ceiling arithmetic."""
    ranks: list[int] = []
    rank = 1
    while rank <= family_size:
        ranks.append(rank)
        rank *= 2
    if ranks[-1] != family_size:
        ranks.append(family_size)

    point_count = len(ranks)
    prior_mass_scale = 10**9
    evidence_scale = 10**6
    mass_ticks = prior_mass_scale // point_count
    alpha_fraction = Fraction(str(alpha))
    a, b = alpha_fraction.numerator, alpha_fraction.denominator
    merged: dict[int, int] = {}
    for rank in ranks:
        h_num = family_size * b * evidence_scale
        h_den = a * rank
        h_ticks = (h_num + h_den - 1) // h_den
        raw_num = h_ticks * prior_mass_scale
        raw_ticks = (raw_num + mass_ticks - 1) // mass_ticks
        merged[raw_ticks] = merged.get(raw_ticks, 0) + mass_ticks
    return [
        (threshold_ticks / evidence_scale, mass / prior_mass_scale)
        for threshold_ticks, mass in sorted(merged.items())
    ]


def projective_rank_points(family_size: int, alpha: float) -> list[tuple[float, float]]:
    """Horizon-independent dyadic-rank prior for gamma_i = 1/family_size."""
    prior_scale = 10**9
    evidence_scale = 10**6
    alpha_fraction = Fraction(str(alpha))
    alpha_num, alpha_den = alpha_fraction.numerator, alpha_fraction.denominator
    points: dict[int, int] = {}
    for exponent in range(64):
        rank = 1 << exponent
        denominator = (exponent + 1) * (exponent + 2)
        mass_ticks = prior_scale // denominator
        if mass_ticks == 0:
            raise ValueError("rank prior mass quantized to zero")
        target_num = family_size * alpha_den * evidence_scale
        target_den = alpha_num * rank
        target_ticks = (target_num + target_den - 1) // target_den
        raw_num = target_ticks * prior_scale
        raw_ticks = max(evidence_scale, (raw_num + mass_ticks - 1) // mass_ticks)
        points[raw_ticks] = points.get(raw_ticks, 0) + mass_ticks
    return [
        (threshold / evidence_scale, mass / prior_scale)
        for threshold, mass in sorted(points.items())
    ]


def projective_rank_hybrid_points(family_size: int, alpha: float) -> list[tuple[float, float]]:
    """Half telescoping rank prior, half uniform over all 64 u64 dyadic ranks."""
    prior_scale = 10**9
    evidence_scale = 10**6
    alpha_fraction = Fraction(str(alpha))
    alpha_num, alpha_den = alpha_fraction.numerator, alpha_fraction.denominator
    points: dict[int, int] = {}
    for exponent in range(64):
        rank = 1 << exponent
        denominator = (exponent + 1) * (exponent + 2)
        mass_ticks = prior_scale // (2 * denominator) + prior_scale // 128
        target_num = family_size * alpha_den * evidence_scale
        target_den = alpha_num * rank
        target_ticks = (target_num + target_den - 1) // target_den
        raw_num = target_ticks * prior_scale
        raw_ticks = max(evidence_scale, (raw_num + mass_ticks - 1) // mass_ticks)
        points[raw_ticks] = points.get(raw_ticks, 0) + mass_ticks
    return [
        (threshold / evidence_scale, mass / prior_scale)
        for threshold, mass in sorted(points.items())
    ]


def calibrate(running_max: np.ndarray, family_size: int, alpha: float) -> dict[str, np.ndarray]:
    sqrt = np.sqrt(running_max) - 1.0

    dyadic = np.zeros_like(running_max)
    for exponent in range(1, 26):
        threshold = float(2**exponent)
        prior_mass = 1.0 / (exponent * (exponent + 1))
        dyadic += prior_mass * threshold * (running_max >= threshold)

    rank_geometric = np.zeros_like(running_max)
    for threshold, prior_mass in rank_geometric_points(family_size, alpha):
        rank_geometric += prior_mass * threshold * (running_max >= threshold)

    projective_rank = np.zeros_like(running_max)
    for threshold, prior_mass in projective_rank_points(family_size, alpha):
        projective_rank += prior_mass * threshold * (running_max >= threshold)

    projective_hybrid = np.zeros_like(running_max)
    for threshold, prior_mass in projective_rank_hybrid_points(family_size, alpha):
        projective_hybrid += prior_mass * threshold * (running_max >= threshold)

    return {
        "square_root": sqrt,
        "universal_dyadic_25": dyadic,
        "rank_geometric_fixed_family": rank_geometric,
        "rank_geometric_projective": projective_rank,
        "rank_geometric_projective_hybrid": projective_hybrid,
    }


def run_cell(
    *,
    family_size: int,
    rounds: int,
    streams: int,
    alpha: float,
    null_probability: float,
    alternative_probability: float,
    alternative_fraction: float,
    seed: int,
    dependence: str,
) -> dict[str, Any]:
    if family_size < 1 or rounds < 1 or streams < 1:
        raise ValueError("family_size, rounds and streams must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 <= alternative_fraction <= 1.0:
        raise ValueError("alternative_fraction must be in [0, 1]")
    if not 0.0 < null_probability < 1.0:
        raise ValueError("null_probability must be in (0, 1)")
    if not 0.0 < alternative_probability < 1.0:
        raise ValueError("alternative_probability must be in (0, 1)")

    rng = np.random.default_rng(seed)
    number_alternatives = int(round(family_size * alternative_fraction))
    number_nulls = family_size - number_alternatives
    truth = np.zeros(family_size, dtype=bool)
    truth[:number_alternatives] = True

    log_success = math.log(alternative_probability / null_probability)
    log_failure = math.log(
        (1.0 - alternative_probability) / (1.0 - null_probability)
    )
    log_wealth = np.zeros((streams, family_size), dtype=float)
    running_max = np.ones((streams, family_size), dtype=float)

    names = (
        "square_root",
        "universal_dyadic_25",
        "rank_geometric_fixed_family",
        "rank_geometric_projective",
        "rank_geometric_projective_hybrid",
    )
    max_fdp = {name: np.zeros(streams, dtype=float) for name in names}
    terminal_fdp = {name: np.zeros(streams, dtype=float) for name in names}
    terminal_power = {name: np.zeros(streams, dtype=float) for name in names}
    any_false = {name: np.zeros(streams, dtype=bool) for name in names}

    for _round in range(rounds):
        if dependence == "independent":
            observations = rng.binomial(
                1,
                np.where(truth, alternative_probability, null_probability),
                size=(streams, family_size),
            ).astype(np.int8)
        elif dependence == "balanced_negative_null":
            null_observations = balanced_bernoulli(
                rng, streams, number_nulls, null_probability
            )
            alternative_observations = rng.binomial(
                1, alternative_probability, size=(streams, number_alternatives)
            ).astype(np.int8)
            observations = np.concatenate(
                (alternative_observations, null_observations), axis=1
            )
        else:
            raise ValueError(f"unsupported dependence mode: {dependence}")

        log_wealth += np.where(observations == 1, log_success, log_failure)
        current_wealth = np.exp(np.clip(log_wealth, -700.0, 700.0))
        np.maximum(running_max, current_wealth, out=running_max)

        calibrated = calibrate(running_max, family_size, alpha)
        for name in names:
            rejected = e_bh_rejections(calibrated[name], alpha)
            count_rejected = rejected.sum(axis=1)
            count_false = (rejected & (~truth)[None, :]).sum(axis=1)
            fdp = np.divide(
                count_false,
                count_rejected,
                out=np.zeros(streams, dtype=float),
                where=count_rejected > 0,
            )
            max_fdp[name] = np.maximum(max_fdp[name], fdp)
            any_false[name] |= count_false > 0

            if _round == rounds - 1:
                terminal_fdp[name] = fdp
                if number_alternatives:
                    terminal_power[name] = (
                        (rejected & truth[None, :]).sum(axis=1)
                        / number_alternatives
                    )

    hoeffding_half_width = math.sqrt(math.log(1.0 / 0.05) / (2.0 * streams))
    results: dict[str, Any] = {}
    for name in names:
        sup_fdr = float(max_fdp[name].mean())
        results[name] = {
            "fixed_time_fdr": float(terminal_fdp[name].mean()),
            "sup_fdr": sup_fdr,
            "sup_fdr_one_sided_95_hoeffding_upper": float(
                min(1.0, sup_fdr + hoeffding_half_width)
            ),
            "terminal_power": float(terminal_power[name].mean()),
            "probability_any_false": float(any_false[name].mean()),
        }

    raw_thresholds = {
        "fixed_family": [threshold for threshold, _mass in rank_geometric_points(family_size, alpha)],
        "projective": [threshold for threshold, _mass in projective_rank_points(family_size, alpha)],
        "projective_hybrid": [threshold for threshold, _mass in projective_rank_hybrid_points(family_size, alpha)],
    }
    return {
        "metadata": {
            "family_size": family_size,
            "rounds": rounds,
            "streams": streams,
            "alpha": alpha,
            "null_probability": null_probability,
            "alternative_probability": alternative_probability,
            "alternative_fraction": alternative_fraction,
            "seed": seed,
            "dependence": dependence,
            "rank_grid_raw_thresholds": raw_thresholds,
            "numpy_version": np.__version__,
            "claim_class": "exploratory_simulation_only",
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", type=int, default=3000)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--family-size", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--p0", type=float, default=0.5)
    parser.add_argument("--p1", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--dependence",
        choices=("independent", "balanced_negative_null"),
        default="independent",
    )
    parser.add_argument(
        "--alternative-fractions",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.2, 0.5],
    )
    args = parser.parse_args()

    output = []
    for fraction in args.alternative_fractions:
        base_seed = args.seed if args.seed is not None else (
            992301 if args.dependence == "balanced_negative_null" else 20261102
        )
        seed = base_seed + int(args.p1 * 1000) + int(fraction * 10000)
        output.append(
            run_cell(
                family_size=args.family_size,
                rounds=args.rounds,
                streams=args.streams,
                alpha=args.alpha,
                null_probability=args.p0,
                alternative_probability=args.p1,
                alternative_fraction=fraction,
                seed=seed,
                dependence=args.dependence,
            )
        )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
