#!/usr/bin/env python3
"""AWAIRE-style intersection-local portfolios with dynamic projective closure.

For each finite intersection S, this constructs a separate predictable convex
mixture of its currently active coordinate e-value increments. The shares are
normalized within S intersect I_t, as in adaptive intersection
test-supermartingales. Because future inactive coordinates do not participate
and the weights depend only on the currently active intersection histories,
the resulting family is future-extension coherent.

The module verifies pathwise projectivity and time monotonicity after a common
adjuster. It does not prove the upstream conditional e-value contract.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Sequence

from self_financing_projective_eclosure import (
    _adjust_paths,
    _policy_scores,
    _verify_dynamic_collection,
    as_fraction,
    predictable_shares,
)


def build_intersection_local_paths(
    increment_paths: Sequence[Sequence[int | float | str | Fraction]],
    active_masks: Sequence[int],
    priority_scores: Sequence[int | float | str | Fraction],
    *,
    share_policy: str = "past_wealth",
) -> tuple[dict[int, tuple[Fraction, ...]], dict[int, tuple[tuple[Fraction, ...], ...]]]:
    """Build one AWAIRE-style adaptive mixture for every finite intersection.

    At update t, the portfolio for S is normalized over S intersect I_t:
      G_t^S = 1 - sum_i p_i^S,t + sum_i p_i^S,t X_i,t.
    For nonempty S intersect I_t and nonzero score mass the weights sum to one,
    recovering the AWAIRE convex mixture. If all scores are zero, the factor is
    one (cash) and remains a valid supermartingale factor.

    Weights are functions only of fixed priorities or of coordinate wealths
    accumulated strictly before t. This is the key projectivity condition.
    """
    n = len(increment_paths)
    t_count = len(active_masks)
    if n < 1 or t_count < 1 or len(priority_scores) != n:
        raise ValueError("invalid coordinate, time or score dimensions")
    x = tuple(tuple(as_fraction(v) for v in row) for row in increment_paths)
    scores = tuple(as_fraction(v) for v in priority_scores)
    if any(len(row) != t_count for row in x):
        raise ValueError("coordinate increment path length mismatch")
    if any(v < 0 for row in x for v in row):
        raise ValueError("coordinate increments must be nonnegative")
    if any(v < 0 for v in scores):
        raise ValueError("priority scores must be nonnegative")

    full_mask = (1 << n) - 1
    active = tuple(active_masks)
    previous = 0
    for t, mask in enumerate(active):
        if isinstance(mask, bool) or not isinstance(mask, int):
            raise ValueError("active masks must be integer bit masks")
        if mask < 0 or mask & ~full_mask:
            raise ValueError("active mask contains unknown coordinates")
        if mask & previous != previous:
            raise ValueError("active hypothesis sets must be nested")
        previous = mask
        for i in range(n):
            if not (mask >> i) & 1 and x[i][t] != 1:
                raise ValueError("dormant coordinate increments must equal one")

    result: dict[int, tuple[Fraction, ...]] = {}
    shares_by_subset: dict[int, tuple[tuple[Fraction, ...], ...]] = {}
    for subset in range(1, 1 << n):
        past_wealth = [Fraction(1) for _ in range(n)]
        wealth = Fraction(1)
        row = []
        subset_shares = []
        for t, active_mask in enumerate(active):
            local_mask = subset & active_mask
            current_scores = _policy_scores(
                share_policy, scores, past_wealth, local_mask
            )
            shares = predictable_shares(local_mask, current_scores)
            subset_shares.append(shares)
            mass = sum(shares, Fraction(0))
            bet = sum(
                (shares[i] * x[i][t] for i in range(n)),
                Fraction(0),
            )
            factor = 1 - mass + bet
            if factor < 0:
                raise ArithmeticError("intersection-local portfolio factor became negative")
            wealth *= factor
            row.append(wealth)
            for i in range(n):
                past_wealth[i] *= x[i][t]
        result[subset] = tuple(row)
        shares_by_subset[subset] = tuple(subset_shares)
    return result, shares_by_subset


def verify_intersection_local_collection(
    raw_paths: dict[int, Sequence[Fraction]],
    active_masks: Sequence[int],
    thresholds: Sequence[int | float | str | Fraction],
    increments: Sequence[int | float | str | Fraction],
) -> dict:
    adjusted, budget = _adjust_paths(raw_paths, thresholds, increments)
    coherence = _verify_dynamic_collection(adjusted, active_masks)
    return {
        "valid": coherence["valid"],
        "failures": coherence["failures"],
        "time_monotonicity_checks": coherence["time_monotonicity_checks"],
        "projective_checks": coherence["projective_checks"],
        "projective_equalities": coherence["projective_equalities"],
        "adjuster_integral_budget": budget,
        "adjusted_paths": adjusted,
        "scope": "pathwise projectivity/time-monotonicity; conditional e-value contract external",
    }


if __name__ == "__main__":
    raise SystemExit("Run test_awaire_dynamic_projective_baseline.py for exact fixtures.")
