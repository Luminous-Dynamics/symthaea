#!/usr/bin/env python3
"""Exploratory family-size transfer benchmark for projective weighted e-BH.

All candidates use a fixed global weight gamma=1/max_family_size. The same
projective calibration function is therefore reused when the active family
is extended from a short prefix to a longer prefix. This is a research-only
simulation; it is not a proof, production implementation, or SOTA claim.

Dependencies: NumPy.
"""

from __future__ import annotations

import argparse
import json
import math
from fractions import Fraction
from typing import Any

import numpy as np


PRIOR_SCALE = 10**9
EVIDENCE_SCALE = 10**6


def weighted_e_bh(values: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """Weighted e-BH with equal fixed candidate weight gamma, sum gamma_i <= 1."""
    streams, family_size = values.shape
    ordered = np.sort(values, axis=1)[:, ::-1]
    ranks = np.arange(1, family_size + 1, dtype=float)
    thresholds = 1.0 / (alpha * gamma * ranks)
    qualifies = ordered >= thresholds[None, :]
    selected = np.where(qualifies, ranks[None, :], 0).max(axis=1)
    cutoff = np.where(
        selected > 0,
        1.0 / (alpha * gamma * np.maximum(selected, 1)),
        np.inf,
    )
    return values >= cutoff[:, None]


def prior_points(
    family_size: int,
    alpha: float,
    weight: Fraction,
    design: str,
) -> list[tuple[float, float]]:
    """Exact integer construction, returned as float pairs for simulation."""
    alpha_fraction = Fraction(str(alpha))
    a, b = alpha_fraction.numerator, alpha_fraction.denominator
    wn, wd = weight.numerator, weight.denominator

    if design == "fixed_family":
        ranks: list[int] = []
        rank = 1
        while rank <= family_size:
            ranks.append(rank)
            rank *= 2
        if ranks[-1] != family_size:
            ranks.append(family_size)
        mass = PRIOR_SCALE // len(ranks)
        rank_masses = [(rank, mass) for rank in ranks]
    elif design in ("projective", "projective_hybrid"):
        rank_masses = []
        for exponent in range(64):
            rank = 1 << exponent
            denominator = (exponent + 1) * (exponent + 2)
            if design == "projective":
                mass = PRIOR_SCALE // denominator
            else:
                mass = PRIOR_SCALE // (2 * denominator) + PRIOR_SCALE // 128
            rank_masses.append((rank, mass))
    else:
        raise ValueError(f"unsupported design: {design}")

    merged: dict[int, int] = {}
    for rank, mass in rank_masses:
        target_num = b * wd * EVIDENCE_SCALE
        target_den = a * rank * wn
        target = (target_num + target_den - 1) // target_den
        raw_num = target * PRIOR_SCALE
        raw = max(EVIDENCE_SCALE, (raw_num + mass - 1) // mass)
        merged[raw] = merged.get(raw, 0) + mass

    return [
        (threshold / EVIDENCE_SCALE, mass / PRIOR_SCALE)
        for threshold, mass in sorted(merged.items())
    ]


def calibrators(
    running_max: np.ndarray,
    family_size: int,
    alpha: float,
    weight: Fraction,
) -> dict[str, np.ndarray]:
    out = {"square_root": np.sqrt(running_max) - 1.0}

    dyadic = np.zeros_like(running_max)
    for exponent in range(1, 26):
        threshold = float(2**exponent)
        mass = 1.0 / (exponent * (exponent + 1))
        dyadic += mass * threshold * (running_max >= threshold)
    out["universal_dyadic_25"] = dyadic

    for design, name in [
        ("fixed_family", "fixed_family_rank"),
        ("projective", "projective_rank"),
        ("projective_hybrid", "projective_hybrid_rank"),
    ]:
        value = np.zeros_like(running_max)
        for threshold, mass in prior_points(family_size, alpha, weight, design):
            value += mass * threshold * (running_max >= threshold)
        out[name] = value
    return out


def run_cell(
    *,
    family_size: int,
    max_family_size: int,
    rounds: int,
    streams: int,
    alpha: float,
    alternative_fraction: float,
    p0: float,
    p1: float,
    seed: int,
) -> dict[str, Any]:
    if not 1 <= family_size <= max_family_size:
        raise ValueError("family_size must be within [1, max_family_size]")
    if max_family_size < 1 or rounds < 1 or streams < 1:
        raise ValueError("max_family_size, rounds and streams must be positive")
    if not 0.0 < alpha < 1.0 or not 0.0 < p0 < 1.0 or not 0.0 < p1 < 1.0:
        raise ValueError("alpha, p0 and p1 must be in (0, 1)")
    if not 0.0 <= alternative_fraction <= 1.0:
        raise ValueError("alternative_fraction must be in [0, 1]")

    # The first max_family_size candidate weights are fixed globally; no
    # renormalization occurs when a shorter prefix is tested.
    gamma = Fraction(1, max_family_size)
    rng = np.random.default_rng(seed)
    n_alt = int(round(family_size * alternative_fraction))
    truth = np.zeros(family_size, dtype=bool)
    truth[:n_alt] = True

    log_success = math.log(p1 / p0)
    log_failure = math.log((1.0 - p1) / (1.0 - p0))
    log_wealth = np.zeros((streams, family_size), dtype=float)
    running_max = np.ones((streams, family_size), dtype=float)

    names = (
        "square_root",
        "universal_dyadic_25",
        "fixed_family_rank",
        "projective_rank",
        "projective_hybrid_rank",
    )
    max_fdp = {name: np.zeros(streams) for name in names}
    terminal_fdp = {name: np.zeros(streams) for name in names}
    terminal_power = {name: np.zeros(streams) for name in names}
    any_false = {name: np.zeros(streams, dtype=bool) for name in names}

    for look in range(rounds):
        observations = rng.binomial(
            1,
            np.where(truth, p1, p0),
            size=(streams, family_size),
        ).astype(np.int8)
        log_wealth += np.where(observations == 1, log_success, log_failure)
        current = np.exp(np.clip(log_wealth, -700.0, 700.0))
        np.maximum(running_max, current, out=running_max)

        calibrated = calibrators(running_max, family_size, alpha, gamma)
        for name in names:
            rejected = weighted_e_bh(calibrated[name], alpha, float(gamma))
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
            if look == rounds - 1:
                terminal_fdp[name] = fdp
                if n_alt:
                    terminal_power[name] = (
                        (rejected & truth[None, :]).sum(axis=1) / n_alt
                    )

    half_width = math.sqrt(math.log(20.0) / (2.0 * streams))
    results = {}
    for name in names:
        sup_fdr = float(max_fdp[name].mean())
        results[name] = {
            "fixed_time_fdr": float(terminal_fdp[name].mean()),
            "sup_fdr": sup_fdr,
            "sup_fdr_hoeffding_upper_95": min(1.0, sup_fdr + half_width),
            "terminal_power": float(terminal_power[name].mean()),
            "probability_any_false": float(any_false[name].mean()),
        }

    return {
        "metadata": {
            "family_size": family_size,
            "max_family_size": max_family_size,
            "fixed_candidate_weight": f"{gamma.numerator}/{gamma.denominator}",
            "active_weight_mass": str(gamma * family_size),
            "rounds": rounds,
            "streams": streams,
            "alpha": alpha,
            "alternative_fraction": alternative_fraction,
            "p0": p0,
            "p1": p1,
            "seed": seed,
            "claim_class": "exploratory_simulation_only",
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-sizes", type=int, nargs="+", default=[10, 100, 1000])
    parser.add_argument("--max-family-size", type=int, default=1000)
    parser.add_argument("--alternative-fractions", type=float, nargs="+", default=[0.1, 0.5])
    parser.add_argument("--streams", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--p0", type=float, default=0.5)
    parser.add_argument("--p1", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()

    output = []
    for family_size in args.family_sizes:
        for alt_fraction in args.alternative_fractions:
            seed = args.seed + family_size + int(alt_fraction * 1000)
            output.append(
                run_cell(
                    family_size=family_size,
                    max_family_size=args.max_family_size,
                    rounds=args.rounds,
                    streams=args.streams,
                    alpha=args.alpha,
                    alternative_fraction=alt_fraction,
                    p0=args.p0,
                    p1=args.p1,
                    seed=seed,
                )
            )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
