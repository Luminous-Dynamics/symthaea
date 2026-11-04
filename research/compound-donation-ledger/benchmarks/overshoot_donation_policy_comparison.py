#!/usr/bin/env python3
"""Matched exploratory benchmark for compound-donation reserve policies.

Compares:
  * no donation,
  * fixed-rate donation from one precommitted donor to one recipient,
  * capped weighted-overshoot donation to that same recipient.

All methods use identical likelihood-ratio streams, a fixed global weight
sequence, identical arrival/observation schedules, and weighted e-BH
self-consistency. Null LR streams are conditionally valid under all three
dependence modes. This is research-only Monte Carlo, not a theorem or SOTA
benchmark.

Dependencies: NumPy.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np


METHODS = ("none", "fixed_rate", "capped_overshoot")


def make_weights(m: int) -> np.ndarray:
    """Fixed summable weights: 1/2 to donor 0, 1/100 to recipient 1."""
    if m < 3:
        raise ValueError("family size must be at least three")
    weights = np.full(m, 0.49 / (m - 2), dtype=np.float64)
    weights[0] = 0.50
    weights[1] = 0.01
    assert abs(float(weights.sum()) - 1.0) < 1e-12
    return weights


def weighted_e_bh(weighted_scores: np.ndarray, alpha: float) -> np.ndarray:
    """Weighted e-BH using fixed weights, expressed in weighted-score units."""
    n_streams, m = weighted_scores.shape
    ordered = np.sort(weighted_scores, axis=1)[:, ::-1]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    qualifies = ordered >= (1.0 / (alpha * ranks))[None, :]
    selected = np.where(qualifies, ranks[None, :], 0.0).max(axis=1)
    cutoff = np.where(
        selected > 0,
        1.0 / (alpha * np.maximum(selected, 1.0)),
        np.inf,
    )
    return weighted_scores >= cutoff[:, None]


def draw_observations(
    rng: np.random.Generator,
    probabilities: np.ndarray,
    truth: np.ndarray,
    n_streams: int,
    dependence: str,
) -> np.ndarray:
    """Generate one look; each null coordinate has conditional success p=1/2."""
    m = len(probabilities)
    if dependence == "independent":
        return rng.random((n_streams, m)) < probabilities[None, :]
    if dependence == "shared_uniform_null":
        y = np.empty((n_streams, m), dtype=bool)
        null_mask = ~truth
        if null_mask.any():
            # One independent U per stream and look, shared across that
            # stream's null coordinates. Marginal conditional P(Y_i=1)=1/2.
            shared = rng.random((n_streams, 1)) < 0.5
            y[:, null_mask] = shared
        if truth.any():
            y[:, truth] = (
                rng.random((n_streams, int(truth.sum())))
                < probabilities[truth][None, :]
            )
        return y
    if dependence == "balanced_negative_null":
        y = np.empty((n_streams, m), dtype=bool)
        n_alt = int(truth.sum())
        if n_alt:
            y[:, truth] = (
                rng.random((n_streams, n_alt))
                < probabilities[truth][None, :]
            )
        n_null = m - n_alt
        if n_null:
            expected = n_null * 0.5
            lower = math.floor(expected)
            fraction = expected - lower
            counts = lower + (rng.random(n_streams) < fraction).astype(np.int64)
            order = np.argsort(rng.random((n_streams, n_null)), axis=1)
            null_y = np.zeros((n_streams, n_null), dtype=bool)
            rows = np.arange(n_streams)[:, None]
            selected = np.arange(n_null)[None, :] < counts[:, None]
            null_y[rows, order] = selected
            y[:, ~truth] = null_y
        return y
    raise ValueError(f"unsupported dependence mode {dependence!r}")


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    p = successes / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denom
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))
        / denom
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def run_cell(
    *,
    streams: int,
    m: int,
    rounds: int,
    alpha: float,
    alt_pattern: str,
    dependence: str,
    fixed_rate_numerator: int,
    fixed_rate_denominator: int,
    seed: int,
) -> dict[str, Any]:
    if streams < 1 or rounds < 1 or m < 3:
        raise ValueError("streams and rounds must be positive; m must be at least 3")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0 <= fixed_rate_numerator <= fixed_rate_denominator or fixed_rate_denominator == 0:
        raise ValueError("fixed rate must be a rational in [0, 1]")

    probabilities = np.full(m, 0.5, dtype=np.float64)
    if alt_pattern == "all_null":
        pass
    elif alt_pattern == "strong_to_many_weak":
        probabilities[0] = 0.90
        probabilities[1:41] = 0.71
    elif alt_pattern == "weak_to_many_weak":
        probabilities[:40] = 0.71
    elif alt_pattern == "strong_donor_only":
        probabilities[0] = 0.90
    else:
        raise ValueError(f"unsupported alternative pattern {alt_pattern!r}")

    truth = probabilities > 0.5
    weights = make_weights(m)
    gamma_donor = float(weights[0])
    fixed_rate = fixed_rate_numerator / fixed_rate_denominator

    rng = np.random.default_rng(seed)
    log_e = np.zeros((streams, m), dtype=np.float64)
    running_log_max = np.zeros_like(log_e)
    previous_adjusted = np.zeros_like(log_e)

    # Cumulative outgoing weighted capital. Only candidate 0 has a committed
    # destination (candidate 1). Others are configured with destination=None.
    outgoing = {name: np.zeros(streams, dtype=np.float64) for name in METHODS}
    previous_scores = {name: np.zeros((streams, m), dtype=np.float64) for name in METHODS}
    max_fdp = {name: np.zeros(streams, dtype=np.float64) for name in METHODS}
    terminal_fdp = {name: np.zeros(streams, dtype=np.float64) for name in METHODS}
    terminal_power = {name: np.zeros(streams, dtype=np.float64) for name in METHODS}
    any_false = {name: np.zeros(streams, dtype=bool) for name in METHODS}

    log_lr_success = math.log(1.5)
    log_lr_failure = math.log(0.5)

    for look in range(rounds):
        y = draw_observations(rng, probabilities, truth, streams, dependence)
        log_e += np.where(y, log_lr_success, log_lr_failure)
        np.maximum(running_log_max, log_e, out=running_log_max)

        # Valid time-monotone adjuster X_t = sqrt(max_{s<=t} E_s) - 1.
        adjusted = np.expm1(0.5 * running_log_max)
        delta_adjusted = adjusted - previous_adjusted

        for method in METHODS:
            previous_outgoing = outgoing[method]
            if method == "none":
                next_outgoing = np.zeros(streams, dtype=np.float64)
            elif method == "fixed_rate":
                requested = (
                    previous_outgoing
                    + fixed_rate * gamma_donor * delta_adjusted[:, 0]
                )
                donor_cap = gamma_donor * np.minimum(adjusted[:, 0], 1.0)
                next_outgoing = np.minimum(requested, donor_cap)
            else:
                raw_overshoot = np.maximum(
                    gamma_donor * adjusted[:, 0] - 1.0, 0.0
                )
                next_outgoing = np.minimum(raw_overshoot, gamma_donor)

            # One precommitted route 0 -> 1; no candidate can be re-donated
            # from incoming capital. The last 98 candidates have no destination.
            incoming = np.zeros((streams, m), dtype=np.float64)
            incoming[:, 1] = next_outgoing
            score = weights[None, :] * adjusted
            score[:, 0] -= next_outgoing
            score += incoming

            # Pathwise accounting checks for every simulation look.
            if np.any(next_outgoing < previous_outgoing - 1e-12):
                raise AssertionError("outgoing reserve decreased")
            donor_cap = gamma_donor * np.minimum(adjusted[:, 0], 1.0)
            if np.any(next_outgoing > donor_cap + 1e-10):
                raise AssertionError("donor cap exceeded")
            donor_increment = next_outgoing - previous_outgoing
            evidence_increment = adjusted[:, 0] - previous_adjusted[:, 0]
            if np.any(donor_increment > gamma_donor * evidence_increment + 1e-10):
                raise AssertionError("reserve exceeded donor's own evidence increment")
            if method == "capped_overshoot":
                expected = np.minimum(
                    np.maximum(gamma_donor * adjusted[:, 0] - 1.0, 0.0),
                    gamma_donor,
                )
                if not np.allclose(next_outgoing, expected, atol=1e-10, rtol=0):
                    raise AssertionError("capped overshoot target mismatch")

            # Every finite prefix has nonpositive incoming-minus-outgoing
            # transfer balance. The source is candidate 0 and destination 1.
            outgoing_by_candidate = np.zeros_like(incoming)
            outgoing_by_candidate[:, 0] = next_outgoing
            prefix_balance = np.cumsum(incoming - outgoing_by_candidate, axis=1)
            if np.any(prefix_balance > 1e-8):
                raise AssertionError("prefix budget violated")
            if np.any(np.abs(incoming.sum(axis=1) - outgoing_by_candidate.sum(axis=1)) > 1e-8):
                raise AssertionError("escrow transfer conservation failed")

            if np.any(score + 1e-10 < previous_scores[method]):
                raise AssertionError("compound score decreased")

            rejected = weighted_e_bh(score, alpha)
            n_rejected = rejected.sum(axis=1)
            n_false = (rejected & (~truth)[None, :]).sum(axis=1)
            fdp = np.divide(
                n_false,
                n_rejected,
                out=np.zeros(streams, dtype=np.float64),
                where=n_rejected > 0,
            )
            max_fdp[method] = np.maximum(max_fdp[method], fdp)
            any_false[method] |= n_false > 0

            if look == rounds - 1:
                terminal_fdp[method] = fdp
                n_true = int(truth.sum())
                if n_true:
                    terminal_power[method] = (
                        (rejected & truth[None, :]).sum(axis=1) / n_true
                    )
            outgoing[method] = next_outgoing
            previous_scores[method] = score

        previous_adjusted = adjusted.copy()

    # FDR and SupFDR are bounded means of per-stream FDP variables. Use a
    # one-sided Hoeffding bound; P(any false) is Bernoulli, use Wilson.
    hoeffding_half_width = math.sqrt(math.log(1.0 / 0.05) / (2.0 * streams))
    results: dict[str, Any] = {}
    for method in METHODS:
        sup_fdr = float(max_fdp[method].mean())
        fixed_fdr = float(terminal_fdp[method].mean())
        any_count = int(any_false[method].sum())
        results[method] = {
            "fixed_time_fdr": fixed_fdr,
            "fixed_time_fdr_hoeffding_upper_95": min(1.0, fixed_fdr + hoeffding_half_width),
            "sup_fdr": sup_fdr,
            "sup_fdr_hoeffding_upper_95": min(1.0, sup_fdr + hoeffding_half_width),
            "terminal_power": float(terminal_power[method].mean()),
            "p_any_false": any_count / streams,
            "p_any_false_wilson_95": wilson_interval(any_count, streams),
            "final_donor_outgoing_weighted": float(outgoing[method].mean()),
        }

    return {
        "metadata": {
            "streams": streams,
            "family_size": m,
            "rounds": rounds,
            "alpha": alpha,
            "alternative_pattern": alt_pattern,
            "dependence": dependence,
            "seed": seed,
            "weights_sum": float(weights.sum()),
            "donor_weight": gamma_donor,
            "recipient_weight": float(weights[1]),
            "fixed_rate": f"{fixed_rate_numerator}/{fixed_rate_denominator}",
            "methods": list(METHODS),
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--streams", type=int, default=20000)
    parser.add_argument("--family-size", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--fixed-rate-numerator", type=int, default=1)
    parser.add_argument("--fixed-rate-denominator", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()

    cells = []
    patterns = ("all_null", "strong_to_many_weak", "weak_to_many_weak", "strong_donor_only")
    dependences = ("independent", "shared_uniform_null", "balanced_negative_null")
    for dep_index, dependence in enumerate(dependences):
        for pattern_index, pattern in enumerate(patterns):
            seed = args.seed + dep_index * 100000 + pattern_index * 1000
            cells.append(
                run_cell(
                    streams=args.streams,
                    m=args.family_size,
                    rounds=args.rounds,
                    alpha=args.alpha,
                    alt_pattern=pattern,
                    dependence=dependence,
                    fixed_rate_numerator=args.fixed_rate_numerator,
                    fixed_rate_denominator=args.fixed_rate_denominator,
                    seed=seed,
                )
            )
    print(json.dumps(cells, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
