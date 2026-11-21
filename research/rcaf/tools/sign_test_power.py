#!/usr/bin/env python3
"""Exact power calculations for the RCAF participant-level sign test."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def exact_two_sided_sign_p(n: int, positive_count: int) -> float:
    if n < 1 or not 0 <= positive_count <= n:
        raise ValueError("n must be positive and positive_count must be in [0, n]")
    k = min(positive_count, n - positive_count)
    lower_tail = sum(math.comb(n, j) for j in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * lower_tail)


def critical_rejection_count(n: int, alpha: float = 0.05) -> int:
    if n < 1:
        raise ValueError("n must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    eligible = [k for k in range(n // 2 + 1) if exact_two_sided_sign_p(n, k) <= alpha]
    return max(eligible) if eligible else -1


def exact_sign_test_power(n: int, q: float, alpha: float = 0.05) -> float:
    """Exact two-sided sign-test power when P(positive gap | non-tied) = q."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    k_critical = critical_rejection_count(n, alpha)
    if k_critical < 0:
        return 0.0
    return sum(
        math.comb(n, k) * (q**k) * ((1.0 - q) ** (n - k))
        for k in range(0, k_critical + 1)
    ) + sum(
        math.comb(n, k) * (q**k) * ((1.0 - q) ** (n - k))
        for k in range(n - k_critical, n + 1)
    )


def minimum_n_for_power(q: float, target_power: float = 0.80, alpha: float = 0.05, max_n: int = 1000) -> int | None:
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must be in (0, 1)")
    if max_n < 2:
        raise ValueError("max_n must be at least 2")
    for n in range(2, max_n + 1):
        if exact_sign_test_power(n, q, alpha) >= target_power:
            return n
    return None


def build_power_grid(
    sample_sizes: list[int],
    q_values: list[float],
    alpha: float = 0.05,
) -> dict:
    return {
        "schema_version": "rcaf.sign-test-power/0.1.0",
        "status": "analytical_design_calculation_not_empirical_result",
        "test": {
            "name": "two_sided_exact_sign_test",
            "alpha": alpha,
            "unit": "independent_non_tied_participant",
            "null_positive_gap_probability": 0.5,
            "alternative_positive_gap_probabilities": q_values,
        },
        "power_grid": [
            {
                "n_eff": n,
                "power": [round(exact_sign_test_power(n, q, alpha), 6) for q in q_values],
            }
            for n in sample_sizes
        ],
        "approximate_minimum_n_for_power_0_8": [
            {"q": q, "n_eff": minimum_n_for_power(q, 0.80, alpha)}
            for q in q_values
        ],
        "assumptions": [
            "independent participant-level pairs",
            "no ties among n_eff",
            "fixed q under alternative",
            "two-sided exact binomial p value",
            "no effect-size or hierarchical variance modeling",
        ],
        "interpretation_limits": [
            "does not power median-gap CI",
            "does not account for missingness unless modeled",
            "not a clinical or causal inference",
            "not an empirical result",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[23, 30, 50, 80, 100, 150, 200])
    parser.add_argument("--q-values", type=float, nargs="+", default=[0.60, 0.65, 0.70, 0.75, 0.80])
    args = parser.parse_args()
    result = build_power_grid(args.sample_sizes, args.q_values, args.alpha)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
