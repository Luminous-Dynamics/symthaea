#!/usr/bin/env python3
"""Independent stdlib reference checks for the Rust certification kernel.

This file is test-only. It intentionally re-implements the small statistical
formulas rather than importing or parsing the Rust implementation.
"""
from __future__ import annotations

import math


def delta(alpha: float, streams: int, n: int) -> float:
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if streams <= 0 or n <= 0:
        raise ValueError("streams and n must be positive")
    return alpha / (streams * n * (n + 1))


def radius(n: int, allocated_alpha: float) -> float:
    if n <= 0 or not 0.0 < allocated_alpha < 1.0:
        raise ValueError("invalid one-sided confidence input")
    return math.sqrt(math.log(1.0 / allocated_alpha) / (2.0 * n))


def lower(successes: int, n: int, allocated_alpha: float) -> float:
    if not 0 <= successes <= n:
        raise ValueError("invalid binomial counts")
    return max(0.0, min(1.0, successes / n - radius(n, allocated_alpha)))


def upper(successes: int, n: int, allocated_alpha: float) -> float:
    if not 0 <= successes <= n:
        raise ValueError("invalid binomial counts")
    return max(0.0, min(1.0, successes / n + radius(n, allocated_alpha)))


def channel(
    tp: int, fn: int, tn: int, fp: int, test_pos: int, test_n: int,
    *, alpha: float = 0.05, streams: int = 4,
) -> dict[str, float | bool | None]:
    n_pos = tp + fn
    n_neg = tn + fp
    if min(n_pos, n_neg, test_n) <= 0:
        raise ValueError("each calibration/test stream needs observations")
    if not 0 <= test_pos <= test_n:
        raise ValueError("invalid test prevalence counts")

    tpr_l = lower(tp, n_pos, delta(alpha, streams, n_pos))
    fpr_l = lower(fp, n_neg, delta(alpha, streams, n_neg))
    fpr_u = upper(fp, n_neg, delta(alpha, streams, n_neg))
    q_u = upper(test_pos, test_n, delta(alpha, streams, test_n))

    separated = tpr_l > fpr_u
    consistent = q_u >= fpr_l
    prevalence_upper = None
    if separated and consistent:
        prevalence_upper = min(
            1.0, max(0.0, (q_u - fpr_l) / (tpr_l - fpr_l))
        )
    return {
        "tpr_lower": tpr_l,
        "fpr_lower": fpr_l,
        "fpr_upper": fpr_u,
        "test_positive_upper": q_u,
        "separated": separated,
        "consistent": consistent,
        "prevalence_upper": prevalence_upper,
    }


def main() -> None:
    alpha = 0.05
    streams = 4

    # Summable alpha budget: S * sum_n alpha/(S*n*(n+1)) <= alpha.
    allocated = sum(delta(alpha, streams, n) for n in range(1, 100_001))
    assert allocated * streams <= alpha
    assert allocated * streams > alpha * (1.0 - 2.0 / 100_000.0)

    clean = channel(9500, 500, 9800, 200, 500, 10000)
    assert clean["separated"] is True
    assert clean["consistent"] is True
    assert isinstance(clean["prevalence_upper"], float)
    assert clean["prevalence_upper"] < 0.12

    # TPR lower > FPR lower is not enough: the upper FPR bound still overlaps.
    overlap = channel(8000, 2000, 2200, 7800, 7000, 10000)
    assert overlap["tpr_lower"] > overlap["fpr_lower"]
    assert overlap["tpr_lower"] <= overlap["fpr_upper"]
    assert overlap["separated"] is False
    assert overlap["prevalence_upper"] is None

    # Even a discriminative judge is inconclusive when test q_U < calibration f_L.
    incompatible = channel(9900, 100, 8500, 1500, 0, 10000)
    assert incompatible["separated"] is True
    assert incompatible["consistent"] is False
    assert incompatible["prevalence_upper"] is None

    # Inverted judge: separation must fail.
    inverted = channel(1000, 9000, 2000, 8000, 5000, 10000)
    assert inverted["separated"] is False
    assert inverted["prevalence_upper"] is None

    print("independent anytime certification reference checks: PASS")


if __name__ == "__main__":
    main()
