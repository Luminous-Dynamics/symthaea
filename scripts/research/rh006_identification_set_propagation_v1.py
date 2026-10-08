#!/usr/bin/env python3
"""RH-006 exact identified-set -> downstream-surface propagation diagnostic.

Research-only. This harness deliberately avoids dense sampling as the definition
of the identified-set image. For the RH-006 ray slice
    Delta(tu;s) = A t^2 + B(s) t + C(s)
with B(s) affine and C(s) quadratic, the discriminant is quadratic in s.
Surface ranges are obtained from endpoints and analytic critical points;
positive-root support is partitioned at polynomial roots.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass


TOL = 1e-12


class SetRepresentation:
    WITNESS_ONLY = "witness-only"
    INNER = "inner-approximation"
    OUTER = "outer-approximation"
    SHARP = "sharp"

    @classmethod
    def permits_stable_decision_claim(cls, representation: str) -> bool:
        return representation in (cls.OUTER, cls.SHARP)

    @classmethod
    def permits_unstable_decision_claim(cls, representation: str) -> bool:
        return representation in (cls.INNER, cls.SHARP)


@dataclass(frozen=True)
class ClosedInterval:
    lower: float
    upper: float

    def validate(self) -> None:
        if not (math.isfinite(self.lower) and math.isfinite(self.upper)):
            raise ValueError("interval endpoints must be finite")
        if self.lower > self.upper:
            raise ValueError("interval lower endpoint exceeds upper endpoint")


@dataclass(frozen=True)
class Quadratic:
    a: float
    b: float
    c: float

    def evaluate(self, x: float) -> float:
        return self.a * x * x + self.b * x + self.c

    def range_on(self, domain: ClosedInterval) -> ClosedInterval:
        domain.validate()
        candidates = [domain.lower, domain.upper]
        if abs(self.a) > TOL:
            vertex = -self.b / (2.0 * self.a)
            if domain.lower - TOL <= vertex <= domain.upper + TOL:
                candidates.append(min(domain.upper, max(domain.lower, vertex)))
        values = [self.evaluate(x) for x in candidates]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("non-finite polynomial image")
        return ClosedInterval(min(values), max(values))


def linear_or_quadratic_roots(poly: Quadratic, domain: ClosedInterval) -> list[float]:
    domain.validate()
    if abs(poly.a) <= TOL:
        if abs(poly.b) <= TOL:
            return []
        roots = [-poly.c / poly.b]
    else:
        discriminant = poly.b * poly.b - 4.0 * poly.a * poly.c
        if discriminant < -TOL:
            return []
        root_disc = math.sqrt(max(0.0, discriminant))
        # Stable quadratic formula: q avoids cancellation near double/weak roots.
        q = -0.5 * (poly.b + math.copysign(root_disc, poly.b))
        if abs(q) <= TOL:
            roots = [-poly.b / (2.0 * poly.a)]
        else:
            roots = [q / poly.a, poly.c / q]
    clipped = []
    for root in roots:
        if math.isfinite(root) and domain.lower - TOL <= root <= domain.upper + TOL:
            clipped.append(min(domain.upper, max(domain.lower, root)))
    return sorted(set(clipped))


@dataclass(frozen=True)
class RaySurface:
    A: float
    B0: float
    B1: float
    C0: float
    C1: float
    C2: float

    def validate(self) -> None:
        if not math.isfinite(self.A) or self.A <= TOL:
            raise ValueError("A must be strictly positive for the quadratic-ray calculus")
        if not all(math.isfinite(x) for x in (self.B0, self.B1, self.C0, self.C1, self.C2)):
            raise ValueError("ray coefficients must be finite")

    def B(self, s: float) -> float:
        return self.B0 + self.B1 * s

    def C(self, s: float) -> float:
        return self.C0 + self.C1 * s + self.C2 * s * s

    def discriminant_poly(self) -> Quadratic:
        return Quadratic(
            self.B1 * self.B1 - 4.0 * self.A * self.C2,
            2.0 * self.B0 * self.B1 - 4.0 * self.A * self.C1,
            self.B0 * self.B0 - 4.0 * self.A * self.C0,
        )

    def effect_radius_boundary_points(
        self,
        domain: ClosedInterval,
        effect_radius: float,
        directional_effect_scale: float,
        sigma_ref: float,
    ) -> list[float]:
        self.validate()
        domain.validate()
        if (
            not math.isfinite(effect_radius)
            or effect_radius < 0.0
            or not math.isfinite(directional_effect_scale)
            or directional_effect_scale <= TOL
            or not math.isfinite(sigma_ref)
            or sigma_ref <= TOL
        ):
            raise ValueError("effect-radius scale and threshold must be finite and positive")
        t = effect_radius * sigma_ref / directional_effect_scale
        if not math.isfinite(t):
            raise ValueError("effect-radius coefficient threshold is non-finite")
        threshold_surface = Quadratic(
            self.C2,
            self.C1 + self.B1 * t,
            self.C0 + self.B0 * t + self.A * t * t,
        )
        return linear_or_quadratic_roots(threshold_surface, domain)


    def point_structure(self, s: float) -> tuple[str, int]:
        B = self.B(s)
        D = self.discriminant_poly().evaluate(s)
        if D < -TOL:
            return ("no-real-branch", 0)
        if abs(D) <= TOL:
            root = -B / (2.0 * self.A)
            return ("double-root", int(root > TOL))
        # With A > 0, positive-root count follows from the signs of
        # B and C, avoiding cancellation from direct root evaluation.
        if C < -TOL:
            positive_count = 1
        elif C > TOL:
            positive_count = 2 if B < -TOL else 0
        else:
            positive_count = 1 if B < -TOL else 0
        return ("two-real-branches", positive_count)

    def decision_partition(self, domain: ClosedInterval):
        self.validate()
        domain.validate()
        d = self.discriminant_poly()
        critical = [domain.lower, domain.upper]
        critical.extend(linear_or_quadratic_roots(d, domain))
        critical.extend(linear_or_quadratic_roots(Quadratic(self.C2, self.C1, self.C0), domain))
        critical.extend(linear_or_quadratic_roots(Quadratic(0.0, self.B1, self.B0), domain))
        critical = sorted(set(critical))

        segments = []
        for left, right in zip(critical, critical[1:]):
            if right - left <= TOL:
                continue
            mid = 0.5 * (left + right)
            regime, positive_count = self.point_structure(mid)
            segments.append({
                "lower": left,
                "upper": right,
                "regime": regime,
                "positive_root_count": positive_count,
            })

        endpoints = [
            {
                "s": x,
                "regime": self.point_structure(x)[0],
                "positive_root_count": self.point_structure(x)[1],
                "discriminant": d.evaluate(x),
                "B": self.B(x),
                "C": self.C(x),
            }
            for x in critical
        ]
        return segments, endpoints

    def classify(self, domain: ClosedInterval) -> dict:
        segments, endpoints = self.decision_partition(domain)
        signatures = {
            item["positive_root_count"] for item in segments
            + [{"positive_root_count": e["positive_root_count"]} for e in endpoints]
        }
        regimes = {e["regime"] for e in endpoints} | {s["regime"] for s in segments}
        d_image = self.discriminant_poly().range_on(domain)
        boundary_reachable = any(abs(e["discriminant"]) <= TOL for e in endpoints)
        return {
            "discriminant_image": {
                "lower": d_image.lower,
                "upper": d_image.upper,
            },
            "admissible_regimes": sorted(regimes),
            "positive_root_counts": sorted(signatures),
            "decision_status": (
                "decision-stable-under-identification-set"
                if len(signatures) == 1 and 0 not in signatures
                else "decision-unstable-under-identification-set"
            ),
            "discriminant_boundary_reachable": boundary_reachable,
            "segments": segments,
            "boundary_evaluations": endpoints,
        }


def classify_identification_set(domain: ClosedInterval) -> str:
    domain.validate()
    return "MI" if domain.lower == domain.upper else "PI"


def contraction_surface(domain: ClosedInterval) -> dict:
    domain.validate()
    if domain.lower < -1.0 - TOL or domain.upper > 1.0 + TOL:
        raise ValueError("sqrt(1-s^2) surface domain exceeded")
    max_abs = max(abs(domain.lower), abs(domain.upper))
    min_abs = 0.0 if domain.lower <= 0.0 <= domain.upper else min(abs(domain.lower), abs(domain.upper))
    image = ClosedInterval(
        math.sqrt(max(0.0, 1.0 - max_abs * max_abs)),
        math.sqrt(max(0.0, 1.0 - min_abs * min_abs)),
    )
    boundary = abs(domain.lower) >= 1.0 - TOL or abs(domain.upper) >= 1.0 - TOL
    return {
        "surface_id": "contraction-radius-v1",
        "surface_image": {"lower": image.lower, "upper": image.upper},
        "boundary_reachable": boundary,
        "derivative_status": "nonregular-boundary" if boundary else "finite-interior",
    }


def stable_pi_fixture() -> dict:
    surface = RaySurface(A=1.0, B0=-2.0, B1=0.0, C0=-0.5, C1=0.1, C2=0.0)
    domain = ClosedInterval(0.0, 1.0)
    return {"identification": classify_identification_set(domain), "surface": surface.classify(domain)}


def unstable_fixture() -> dict:
    surface = RaySurface(A=1.0, B0=0.0, B1=0.0, C0=0.25, C1=-1.0, C2=0.0)
    domain = ClosedInterval(0.0, 1.0)
    return {"identification": classify_identification_set(domain), "surface": surface.classify(domain)}


def main() -> None:
    c_set = ClosedInterval(-1.0, 1.0)
    quadratic_image = Quadratic(-1.0, 0.0, 0.25).range_on(c_set)
    c_fixture = {
        "target": "C",
        "identification_class": "PI",
        "identified_set": {"lower": -1.0, "upper": 1.0},
        "surface": {
            "surface_id": "quadratic-branch-discriminant-v1",
            "formula": "D(C)=0.25-C^2",
            "image": {"lower": quadratic_image.lower, "upper": quadratic_image.upper},
            "regimes": sorted({"two-real-branches", "double-root", "no-real-branch"}),
        },
    }

    stable = stable_pi_fixture()
    unstable = unstable_fixture()
    contraction = contraction_surface(c_set)

    assert c_fixture["surface"]["image"] == {"lower": -0.75, "upper": 0.25}
    assert c_fixture["identification_class"] == "PI"
    assert stable["surface"]["decision_status"] == "decision-stable-under-identification-set"
    assert stable["surface"]["positive_root_counts"] == [1]
    assert unstable["surface"]["decision_status"] == "decision-unstable-under-identification-set"
    assert unstable["surface"]["discriminant_boundary_reachable"] is True
    assert unstable["surface"]["positive_root_counts"] == [0, 1]
    assert unstable["surface"]["decision_status"] == "decision-unstable-under-identification-set"
    assert unstable["surface"]["discriminant_boundary_reachable"] is True
    cancellation_surface = RaySurface(A=1.0, B0=1.0e12, B1=0.0, C0=-1.0e3, C1=0.0, C2=0.0)
    cancellation_regime, cancellation_count = cancellation_surface.point_structure(0.0)
    assert cancellation_regime == "two-real-branches"
    assert cancellation_count == 1
    unstable_surface = RaySurface(A=1.0, B0=0.0, B1=0.0, C0=0.25, C1=-1.0, C2=0.0)
    boundary_points = unstable_surface.effect_radius_boundary_points(
        ClosedInterval(0.0, 1.0), 0.5, 1.0, 1.0
    )
    assert len(boundary_points) == 1
    assert abs(boundary_points[0] - 0.5) <= TOL
    assert contraction["surface_image"] == {"lower": 0.0, "upper": 1.0}
    assert SetRepresentation.permits_stable_decision_claim(SetRepresentation.OUTER)
    assert not SetRepresentation.permits_unstable_decision_claim(SetRepresentation.OUTER)
    assert SetRepresentation.permits_unstable_decision_claim(SetRepresentation.INNER)
    assert not SetRepresentation.permits_stable_decision_claim(SetRepresentation.INNER)
    assert not SetRepresentation.permits_stable_decision_claim(SetRepresentation.WITNESS_ONLY)
    assert not SetRepresentation.permits_unstable_decision_claim(SetRepresentation.WITNESS_ONLY)
    try:
        contraction_surface(ClosedInterval(-1.1, 1.0))
    except ValueError:
        pass
    else:
        raise AssertionError("contraction-domain violation did not fail closed")

    result = {
        "schema": "rh006-identified-set-propagation/v1",
        "status": "research-diagnostic-only",
        "semantic_repair": {
            "PI": "explicitly characterized non-singleton identified set",
            "NI": "target is not currently characterized by a qualified identified set; witness disagreement alone is insufficient to label the target PI",
            "decision_claim": "may be permitted for PI when every admissible value yields the same decision and regularity is separately acceptable",
        },
        "rh006_ray_calculus": {
            "surface": "Delta(tu;s)=A t^2+(B0+B1*s)t+(C0+C1*s+C2*s^2)",
            "derived_discriminant": "D(s)=d0+d1*s+d2*s^2",
            "coefficients": "d0=B0^2-4*A*C0; d1=2*B0*B1-4*A*C1; d2=B1^2-4*A*C2",
            "root_support_partition": "partition at roots of D(s), C(s), and B(s), then classify open cells and boundaries",
        },
        "fixtures": {
            "c_identified_set": c_fixture,
            "stable_pi": stable,
            "unstable": unstable,
            "contraction": contraction,
        },
        "fail_closed_rules": [
            "uncharacterized NI target cannot be propagated as a point or finite set",
            "domain violation is fatal",
            "discriminant boundary is not regular even when decision status is otherwise stable",
            "numerical sampling may be used only as a convergence diagnostic, never as the definition of the set image",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
