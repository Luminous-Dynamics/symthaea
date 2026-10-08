#!/usr/bin/env python3
"""RH-006 generic identification-set -> surface-propagation diagnostic.

Research-only. This script does not estimate from empirical RH-006 data and
cannot authorize inference. It evaluates a serialized-style identification
receipt concept against deterministic scalar surface functions.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float

    def validate(self) -> None:
        if not (math.isfinite(self.lower) and math.isfinite(self.upper)):
            raise ValueError("interval endpoints must be finite")
        if self.lower > self.upper:
            raise ValueError("interval lower endpoint exceeds upper endpoint")

    @property
    def singleton(self) -> bool:
        return self.lower == self.upper


def classify_interval(points: Iterable[float], tol: float = 1e-12) -> str:
    values = list(points)
    if not values:
        raise ValueError("empty identified set")
    if not all(math.isfinite(x) for x in values):
        raise ValueError("non-finite identified-set witness")
    lo, hi = min(values), max(values)
    if abs(hi - lo) <= tol:
        return "MI"
    return "PI"


def surface_range(interval: Interval, fn: Callable[[float], float],
                  samples: int = 200001) -> Interval:
    interval.validate()
    if samples < 2:
        raise ValueError("samples must be >= 2")
    values = []
    for i in range(samples):
        c = interval.lower + (interval.upper - interval.lower) * i / (samples - 1)
        value = fn(c)
        if not math.isfinite(value):
            raise ValueError("surface returned non-finite value")
        values.append(value)
    return Interval(min(values), max(values))


def branch_regime(c: float) -> str:
    d = 0.25 - c * c
    if d > 1e-12:
        return "two-real-branches"
    if abs(d) <= 1e-12:
        return "double-root"
    return "no-real-branch"


def classify_surface(interval: Interval) -> dict:
    d = surface_range(interval, lambda c: 0.25 - c * c, samples=10001)
    regimes = {branch_regime(interval.lower), branch_regime(interval.upper), branch_regime(0.0)}
    if interval.lower <= 0.5 <= interval.upper:
        regimes.add(branch_regime(0.5))
    if interval.lower <= 0.99 <= interval.upper:
        regimes.add(branch_regime(0.99))

    # A downstream decision is stable only if every admissible target value
    # induces the same qualitative branch regime.
    dense_regimes = {
        branch_regime(
            interval.lower
            + (interval.upper - interval.lower) * i / 10000.0
        )
        for i in range(10001)
    }
    decision_status = (
        "decision-stable-under-identified-set"
        if len(dense_regimes) == 1
        else "decision-unstable-under-identification-set"
    )
    return {
        "surface_id": "quadratic-branch-discriminant-v1",
        "input_interval": {
            "lower": interval.lower,
            "upper": interval.upper,
        },
        "surface_image": {
            "lower": d.lower,
            "upper": d.upper,
        },
        "admissible_branch_regimes": sorted(dense_regimes),
        "decision_status": decision_status,
    }


def contraction_surface(interval: Interval) -> dict:
    # sqrt(1-c^2) is real on [-1,1]. Its derivative becomes unbounded at
    # the contraction boundary, so report both image and maximal sampled
    # derivative amplification.
    if interval.lower < -1.0 or interval.upper > 1.0:
        raise ValueError("contraction surface domain exceeded")
    image = surface_range(interval, lambda c: math.sqrt(max(0.0, 1.0 - c*c)))
    max_slope = 0.0
    for i in range(10001):
        c = interval.lower + (interval.upper - interval.lower) * i / 10000.0
        r = math.sqrt(max(0.0, 1.0 - c*c))
        slope = float("inf") if r == 0.0 else abs(c) / r
        max_slope = max(max_slope, slope)
    slope_status = "nonregular-boundary" if not math.isfinite(max_slope) else "finite-on-grid"
    return {
        "surface_id": "contraction-radius-v1",
        "input_interval": {"lower": interval.lower, "upper": interval.upper},
        "surface_image": {"lower": image.lower, "upper": image.upper},
        "max_sampled_abs_derivative": max_slope if math.isfinite(max_slope) else "infinity",
        "conditioning_status": slope_status,
    }


def main() -> None:
    interval = Interval(-1.0, 1.0)
    interval.validate()

    branches = classify_surface(interval)
    contraction = contraction_surface(interval)

    receipt = {
        "schema": "rh006-identification-surface-gate/v1",
        "status": "research-diagnostic-only",
        "classification": "NI",
        "identified_set": {"lower": -1.0, "upper": 1.0},
        "surface_results": [branches, contraction],
        "gate": {
            "point_estimation_permitted": False,
            "formal_inference_permitted": False,
            "downstream_decision_permitted": False,
            "reason": "target is NI and the identified set crosses multiple downstream branch regimes",
        },
        "determinism": {
            "surface_grid": 10001,
            "range_grid": 200001,
        },
        "nonclaims": [
            "This is not a proof of the RH-006 empirical latent model.",
            "Surface sampling approximates the image for numerical diagnostics; analytic bounds remain preferred.",
            "No p-value, confidence interval, or estimator validity is established.",
        ],
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
