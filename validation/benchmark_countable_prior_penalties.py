#!/usr/bin/env python3
"""Deterministic prior-mass penalty comparison for countable intersections.

This is an allocation-sensitivity benchmark, not a power or FDR experiment.
It compares the gamma-gap code prior with two proper product priors on a fixed
set of interpretable intersection motifs.
"""
from __future__ import annotations

import json
from math import exp, log2, log1p, sinh, pi
from typing import Iterable

from countable_prior_weighted_eclosure import intersection_code_length


SETS: tuple[tuple[int, ...], ...] = (
    (1,), (2,), (10,), (100,), (1000,),
    (1, 2, 3, 4, 5),
    (10, 11, 12, 13, 14),
    (100, 101, 102, 103, 104),
    (1000, 1001, 1002, 1003, 1004),
    (1, 10, 100, 1000),
)


def geometric_product_normalizer(limit: int = 1024) -> float:
    """Approximate Z for lambda_i=2^-i; omitted tail is below 2^-limit."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 16:
        raise ValueError("limit must be an integer >= 16")
    log_product = sum(log1p(2.0 ** (-i)) for i in range(1, limit + 1))
    return exp(log_product) - 1.0


def polynomial_product_normalizer() -> float:
    """Exact identity in floating arithmetic for lambda_i=i^-2.

    Product_{i>=1}(1+i^-2) = sinh(pi)/pi.
    """
    return sinh(pi) / pi - 1.0


def prior_penalty_bits(indices: Iterable[int]) -> dict[str, float | int]:
    """Return -log2(pi_S) for three proper countable priors."""
    values = tuple(indices)
    if not values or any(isinstance(i, bool) or not isinstance(i, int) or i < 1 for i in values):
        raise ValueError("indices must be a nonempty collection of positive integers")
    if len(set(values)) != len(values):
        raise ValueError("indices must be unique")
    ordered = tuple(sorted(values))
    code_bits = intersection_code_length(ordered)
    z_poly = polynomial_product_normalizer()
    z_geo = geometric_product_normalizer()
    polynomial_bits = log2(z_poly) + sum(2.0 * log2(i) for i in ordered)
    geometric_bits = log2(z_geo) + sum(float(i) for i in ordered)
    return {
        "code_bits": code_bits,
        "polynomial_product_bits": polynomial_bits,
        "geometric_product_bits": geometric_bits,
    }


def benchmark_rows(sets: Iterable[Iterable[int]] = SETS) -> list[dict]:
    rows = []
    for values in sets:
        indices = tuple(sorted(values))
        penalties = prior_penalty_bits(indices)
        rows.append({"intersection": list(indices), **penalties})
    return rows


def main() -> None:
    output = {
        "schema": "countable-prior-penalty-benchmark-v1",
        "purpose": "prior allocation sensitivity only; not empirical power or validity",
        "normalizers": {
            "polynomial_lambda_i_i_minus_2": polynomial_product_normalizer(),
            "geometric_lambda_i_2_minus_i": geometric_product_normalizer(),
            "geometric_tail_bound": "omitted log-product tail is < 2^-1024",
        },
        "rows": benchmark_rows(),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
