#!/usr/bin/env python3
"""Independent implementation of Wang-Ramdas Huber-Catoni CS equations.

Derived from the mathematical construction described in Wang & Ramdas, AISTATS 2023,
not copied from their GPL-3.0 reference implementation. Benchmark research only.
"""
from __future__ import annotations
import math
import numpy as np


def _phi_wide(x: np.ndarray, p: float) -> np.ndarray:
    return np.where(x > 0, np.log1p(x + np.abs(x) ** p / p), -np.log1p(-x + np.abs(x) ** p / p))


def _phi_narrow(x: np.ndarray, p: float) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos_flat = x > 1
    pos = (x > 0) & ~pos_flat
    neg_flat = x <= -1
    neg = (x <= 0) & ~neg_flat
    out[pos_flat] = math.log(p)
    out[pos] = -np.log1p(-x[pos] + np.abs(x[pos]) ** p / p)
    out[neg_flat] = -math.log(p)
    out[neg] = np.log1p(x[neg] + np.abs(x[neg]) ** p / p)
    return out


def catoni_huber_cs(samples, *, alpha: float, eps: float, moment_bound: float = 0.25,
                    p: float = 2.0, bisections: int = 28,
                    support: tuple[float, float] = (0.0, 1.0)) -> tuple[float, float]:
    """Return two-sided Huber-robust anytime CS interval at current time.

    Assumptions follow the paper: iid observations from a distribution within the
    stated Huber contamination radius of a clean distribution whose p-th central
    moment is bounded by moment_bound. alpha is total two-sided error level.
    The interval is intersected with support, known here to be [0,1].
    """
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("samples must be a nonempty finite vector")
    if not (0 < alpha < 0.5) or not (0 <= eps < 1) or moment_bound <= 0 or p <= 1:
        raise ValueError("invalid alpha, contamination, moment, or order")
    if p < 1 + (p - 1 / p) * eps:
        raise ValueError("contamination level exceeds the method's breakdown condition")
    n = x.size
    alpha_side = alpha / 2.0
    t = np.arange(1, n + 1, dtype=float)
    if eps == 0:
        lambdas = np.minimum(1.0, 0.5 * (2 * p * np.log(2 / alpha) / (t * moment_bound)) ** (1 / p))
        phi = _phi_wide
    else:
        lambdas = np.full(n, 0.5 * (eps / moment_bound) ** (1 / p), dtype=float)
        phi = _phi_narrow
    log_denom = np.log(1 + (lambdas ** p) * moment_bound / p + (p - 1 / p) * eps)
    log_threshold = math.log(1 / alpha_side)
    lower_support, upper_support = support
    if lower_support >= upper_support:
        raise ValueError("invalid support")

    def root_value(mu: float, side: str) -> float:
        z = lambdas * (x - mu)
        total_phi = float(np.sum(phi(z, p)))
        total_denom = float(np.sum(log_denom))
        if side == "lower":
            return total_phi - total_denom - log_threshold
        return total_phi + total_denom + log_threshold

    def invert(side: str) -> float:
        f_lo = root_value(lower_support, side)
        f_hi = root_value(upper_support, side)
        if f_lo <= 0:
            return lower_support
        if f_hi >= 0:
            return upper_support
        lo, hi = lower_support, upper_support
        for _ in range(bisections):
            mid = (lo + hi) / 2.0
            if root_value(mid, side) > 0:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2.0

    return invert("lower"), invert("upper")
