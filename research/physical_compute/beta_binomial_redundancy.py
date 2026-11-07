#!/usr/bin/env python3
"""Dependency-free beta-binomial redundancy sizing for correlated channel survival.

Research utility only. This does not model spatial defect geometry or any
specific device. rho=0 is the independent binomial case. For rho>0, an
array-level latent survival probability is beta-distributed, followed by
conditionally independent Bernoulli channel outcomes.
"""
from __future__ import annotations

import argparse
import json
from typing import Any


def probability_at_least_m(
    installed: int,
    required: int,
    mean_survival: float,
    intraclass_correlation: float,
) -> float:
    """Return P(X >= required) for binomial (rho=0) or beta-binomial (rho>0)."""
    if installed < required or required < 0:
        return 0.0
    if not (0.0 < mean_survival < 1.0):
        raise ValueError("mean_survival must be in (0, 1)")
    if not (0.0 <= intraclass_correlation < 1.0):
        raise ValueError("intraclass_correlation must be in [0, 1)")

    n = installed
    p = mean_survival

    if intraclass_correlation == 0.0:
        # Binomial recurrence, initialized at P(X=0).
        pmf = (1.0 - p) ** n
        total = 0.0
        for k in range(n + 1):
            if k >= required:
                total += pmf
            if k < n:
                pmf *= ((n - k) / (k + 1)) * (p / (1.0 - p))
        return min(1.0, max(0.0, total))

    concentration = (1.0 / intraclass_correlation) - 1.0
    alpha = p * concentration
    beta = (1.0 - p) * concentration

    # Beta-binomial P(X=0) = (beta)_n / (alpha+beta)_n.
    pmf = 1.0
    for i in range(n):
        pmf *= (beta + i) / (alpha + beta + i)

    total = 0.0
    for k in range(n + 1):
        if k >= required:
            total += pmf
        if k < n:
            pmf *= ((n - k) / (k + 1)) * (
                (alpha + k) / (beta + n - k - 1)
            )
    return min(1.0, max(0.0, total))


def minimum_installed(
    required: int,
    mean_survival: float,
    rho: float,
    availability: float,
    maximum: int = 10000,
) -> dict[str, Any]:
    if not (0.0 < availability < 1.0):
        raise ValueError("availability must be in (0, 1)")
    for n in range(required, maximum + 1):
        probability = probability_at_least_m(n, required, mean_survival, rho)
        if probability >= availability:
            previous = probability_at_least_m(
                n - 1, required, mean_survival, rho
            )
            return {
                "installed_channels": n,
                "probability_at_n": probability,
                "probability_at_n_minus_1": previous,
                "raw_two_factor_parameter_ratio": (2.0 * n) / required,
            }
    raise RuntimeError(f"no solution found through n={maximum}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required", type=int, default=32)
    parser.add_argument("--mean-survival", type=float, default=0.61)
    parser.add_argument(
        "--rhos",
        type=float,
        nargs="+",
        default=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    )
    parser.add_argument(
        "--availability",
        type=float,
        nargs="+",
        default=[0.95, 0.99, 0.999, 0.9999],
    )
    args = parser.parse_args()

    rows = []
    for rho in args.rhos:
        targets = {}
        for target in args.availability:
            targets[str(target)] = minimum_installed(
                args.required, args.mean_survival, rho, target
            )
        rows.append({
            "intraclass_correlation": rho,
            "probability_at_53_installed": probability_at_least_m(
                53, args.required, args.mean_survival, rho
            ),
            "availability_targets": targets,
        })

    print(json.dumps({
        "model": "beta_binomial_exchangeable_channel_survival",
        "required_active_channels": args.required,
        "mean_channel_survival": args.mean_survival,
        "rows": rows,
        "limitations": [
            "rho=0 is independent binomial survival",
            "rho>0 uses one shared beta-distributed survival probability per array/group",
            "exchangeable correlation is not a spatial defect model",
            "no device-specific parameters are inferred",
            "raw two-factor parameter ratio is not physical area or cell count",
        ],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
