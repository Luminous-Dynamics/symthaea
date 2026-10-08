#!/usr/bin/env python3
"""RH-006 exact 2-D nuisance-box propagation diagnostic.

Research-only. A quadratic surface over a compact 2-D rectangle is bounded
exactly by checking corners, all four edge stationary points, and an interior
stationary point when it exists. Dense grids are used only to falsify the
analytic implementation in this diagnostic; they do not define the result.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass

TOL = 1e-12

@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float
    def validate(self) -> None:
        if not all(math.isfinite(v) for v in (self.lower, self.upper)):
            raise ValueError("non-finite interval")
        if self.lower > self.upper:
            raise ValueError("invalid interval")

@dataclass(frozen=True)
class Quadratic:
    a: float
    b: float
    c: float
    def value(self, x: float) -> float:
        return self.a * x * x + self.b * x + self.c
    def range_on(self, domain: Interval) -> Interval:
        domain.validate()
        points = [domain.lower, domain.upper]
        if abs(self.a) > TOL:
            vertex = -self.b / (2.0 * self.a)
            if domain.lower - TOL <= vertex <= domain.upper + TOL:
                points.append(min(domain.upper, max(domain.lower, vertex)))
        vals = [self.value(x) for x in points]
        return Interval(min(vals), max(vals))

@dataclass(frozen=True)
class Quadratic2D:
    a_xx: float
    b_yy: float
    c_xy: float
    d_x: float
    e_y: float
    f0: float
    def value(self, x: float, y: float) -> float:
        return (self.a_xx*x*x + self.b_yy*y*y + self.c_xy*x*y +
                self.d_x*x + self.e_y*y + self.f0)
    def range_on(self, x_domain: Interval, y_domain: Interval) -> Interval:
        x_domain.validate(); y_domain.validate()
        values = [self.value(x, y) for x in (x_domain.lower, x_domain.upper)
                  for y in (y_domain.lower, y_domain.upper)]
        for x in (x_domain.lower, x_domain.upper):
            edge = Quadratic(self.b_yy, self.c_xy*x + self.e_y,
                             self.a_xx*x*x + self.d_x*x + self.f0).range_on(y_domain)
            values.extend([edge.lower, edge.upper])
        for y in (y_domain.lower, y_domain.upper):
            edge = Quadratic(self.a_xx, self.c_xy*y + self.d_x,
                             self.b_yy*y*y + self.e_y*y + self.f0).range_on(x_domain)
            values.extend([edge.lower, edge.upper])
        det = 4.0*self.a_xx*self.b_yy - self.c_xy*self.c_xy
        if abs(det) > TOL:
            x = (self.c_xy*self.e_y - 2.0*self.b_yy*self.d_x) / det
            y = (self.c_xy*self.d_x - 2.0*self.a_xx*self.e_y) / det
            if (x_domain.lower-TOL <= x <= x_domain.upper+TOL and
                y_domain.lower-TOL <= y <= y_domain.upper+TOL):
                values.append(self.value(x, y))
        if not all(math.isfinite(v) for v in values):
            raise ValueError("non-finite quadratic image")
        return Interval(min(values), max(values))

def canonical_fixtures() -> dict:
    box = Interval(-1.0, 1.0)
    discriminant = Quadratic2D(-1.0, -1.0, 0.0, 0.0, 0.0, 0.25)
    d_image = discriminant.range_on(box, box)
    stable = Quadratic2D(0.0, 0.0, 0.0, 0.1, 0.05, -0.5)
    stable_image = stable.range_on(box, box)
    return {
        "canonical_discriminant": {
            "image": [d_image.lower, d_image.upper],
            "expected": [-1.75, 0.25],
            "all_three_regimes_reachable": True,
        },
        "stable_linear_box_surface": {
            "image": [stable_image.lower, stable_image.upper],
            "expected": [-0.65, -0.35],
        },
    }

def box_decision_count(d: Interval, b: Interval, c: Interval) -> str:
    if d.upper < -TOL:
        return "zero"
    if d.lower > TOL and c.upper < -TOL:
        return "one"
    if d.lower > TOL and c.lower > TOL and b.upper < -TOL:
        return "two"
    if d.lower > TOL and c.lower > TOL and b.lower > TOL:
        return "zero"
    return "unknown"


def randomized_grid_falsification(seed: int = 20261008, trials: int = 250) -> dict:
    rng = random.Random(seed)
    max_lower_gap = 0.0
    max_upper_gap = 0.0
    violations = 0
    xs = [(-1.0 + 2.0 * i / 128.0) for i in range(129)]
    ys = xs
    box = Interval(-1.0, 1.0)
    for _ in range(trials):
        q = Quadratic2D(
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(-2.0, 2.0),
        )
        exact = q.range_on(box, box)
        sampled = [q.value(x, y) for x in xs for y in ys]
        sample_lo, sample_hi = min(sampled), max(sampled)
        if sample_lo < exact.lower - 1e-10 or sample_hi > exact.upper + 1e-10:
            violations += 1
        max_lower_gap = max(max_lower_gap, exact.lower - sample_lo)
        max_upper_gap = max(max_upper_gap, sample_hi - exact.upper)
    return {
        "seed": seed,
        "trials": trials,
        "grid": "129x129",
        "containment_violations": violations,
        "max_exact_minus_grid_lower": max_lower_gap,
        "max_grid_minus_exact_upper": max_upper_gap,
        "interpretation": "grid is only a falsification cross-check; exact image comes from analytic boundary/stationary evaluation",
    }

def main() -> None:
    fixtures = canonical_fixtures()
    fuzz = randomized_grid_falsification()
    assert fixtures["canonical_discriminant"]["image"] == [-1.75, 0.25]
    assert fixtures["stable_linear_box_surface"]["image"] == [-0.65, -0.35]
    assert fuzz["containment_violations"] == 0
    assert box_decision_count(Interval(1.0, 2.0), Interval(1.0, 2.0), Interval(1.0, 2.0)) == "zero"
    assert box_decision_count(Interval(1.0, 2.0), Interval(0.0, 0.0), Interval(1.0, 2.0)) == "unknown"
    assert box_decision_count(Interval(1.0, 2.0), Interval(-2.0, -1.0), Interval(1.0, 2.0)) == "two"
    assert box_decision_count(Interval(1.0, 2.0), Interval(-2.0, -1.0), Interval(-2.0, -1.0)) == "one"
    result = {
        "schema": "rh006-multivariate-set-propagation/v1",
        "status": "research-diagnostic-only",
        "object": "2-D compact axis-aligned nuisance box",
        "analytic_rule": "quadratic extrema occur at corners, edge stationary points, or interior stationary points",
        "grid_role": "falsification-only",
        "fixtures": fixtures,
        "randomized_falsification": fuzz,
        "fail_closed": [
            "non-finite coefficients or domain fail",
            "grid-only classification is not accepted as proof",
            "undecidable sign/boundary state remains unknown",
            "no formal inference is enabled",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
