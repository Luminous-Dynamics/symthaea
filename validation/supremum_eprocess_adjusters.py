"""Admissible adjusters for running maxima of e-processes.

The power-family adjuster A_beta(x) = ((1-beta)/beta)*(x**beta-1),
0<beta<1, and its beta->0 limit log(x), satisfy
int_1^infinity A_beta(x)/x^2 dx = 1. Per Tavyrikov, Goeman and de Heide
(2026), applying an admissible adjuster to the running maximum of an
e-process yields another e-process. Because the adjusted running maximum is
nondecreasing, its expected path supremum is at most one.

This module performs the deterministic path transformation only. The caller
must supply a genuinely valid e-process starting at one.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


def _finite_nonnegative(values: Iterable[float], name: str) -> list[float]:
    result = [float(value) for value in values]
    if any(not math.isfinite(value) or value < 0.0 for value in result):
        raise ValueError(f"{name} must contain finite nonnegative values")
    return result


def _normalize_prior(
    betas: Sequence[float], weights: Sequence[float]
) -> tuple[list[float], list[float]]:
    if len(betas) == 0 or len(betas) != len(weights):
        raise ValueError("betas and weights must have equal nonzero lengths")
    beta_values = [float(x) for x in betas]
    weight_values = _finite_nonnegative(weights, "weights")
    if any(not math.isfinite(beta) or beta < 0.0 or beta >= 1.0 for beta in beta_values):
        raise ValueError("beta values must lie in [0,1)")
    if len(set(beta_values)) != len(beta_values):
        raise ValueError("beta values must be unique")
    total = math.fsum(weight_values)
    if total <= 0.0 or not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("fixed prior weights must sum to one")
    return beta_values, weight_values


def admissible_power_adjuster(value: float, beta: float) -> float:
    """Evaluate A_beta(x); beta=0 is the continuous log(x) limit."""
    x = float(value)
    b = float(beta)
    if not math.isfinite(x) or x < 1.0:
        raise ValueError("running maximum must be finite and at least one")
    if not math.isfinite(b) or b < 0.0 or b >= 1.0:
        raise ValueError("beta must lie in [0,1)")
    if b == 0.0:
        return math.log(x)
    return ((1.0 - b) / b) * math.expm1(b * math.log(x))


def admissible_power_adjuster_area(beta: float) -> float:
    """Return the analytic integral of A_beta(x)/x^2 over [1,infinity)."""
    b = float(beta)
    if not math.isfinite(b) or b < 0.0 or b >= 1.0:
        raise ValueError("beta must lie in [0,1)")
    if b == 0.0:
        return 1.0
    return ((1.0 - b) / b) * (1.0 / (1.0 - b) - 1.0)


def running_maximum(values: Sequence[float]) -> tuple[float, ...]:
    """Return prefix maxima for an e-process path beginning at exactly one."""
    path = _finite_nonnegative(values, "e_process")
    if not path or path[0] != 1.0:
        raise ValueError("e-process path must be nonempty and start at exactly one")
    running: list[float] = []
    maximum = 1.0
    for value in path:
        maximum = max(maximum, value)
        running.append(maximum)
    return tuple(running)


def adjusted_running_maximum_path(
    e_process: Sequence[float], beta: float
) -> tuple[float, ...]:
    """Apply A_beta to the running maximum of one valid e-process path."""
    maxima = running_maximum(e_process)
    return tuple(admissible_power_adjuster(value, beta) for value in maxima)


def fixed_prior_adjuster_portfolio(
    e_process: Sequence[float],
    *,
    betas: Sequence[float] = (0.0, 0.25, 0.5, 0.75),
    weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
) -> tuple[float, ...]:
    """Fixed ex-ante mixture over adjusted running maxima."""
    beta_values, prior = _normalize_prior(betas, weights)
    maxima = running_maximum(e_process)
    components = [
        tuple(admissible_power_adjuster(value, beta) for value in maxima)
        for beta in beta_values
    ]
    return tuple(
        math.fsum(prior[j] * components[j][t] for j in range(len(prior)))
        for t in range(len(maxima))
    )


def adjusted_intersection_paths(
    raw_intersection_e_paths: Mapping[frozenset[str], Sequence[float]],
    *,
    betas: Sequence[float] = (0.0, 0.25, 0.5, 0.75),
    weights: Sequence[float] = (0.25, 0.25, 0.25, 0.25),
) -> dict[frozenset[str], tuple[float, ...]]:
    """Convert each raw intersection e-process into a path-adjusted auxiliary."""
    beta_values, _ = _normalize_prior(betas, weights)
    return {
        intersection: fixed_prior_adjuster_portfolio(
            path, betas=beta_values, weights=weights
        )
        for intersection, path in raw_intersection_e_paths.items()
    }
