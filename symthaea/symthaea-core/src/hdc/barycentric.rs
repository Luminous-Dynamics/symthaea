// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Barycentric coordinates (Phase 2C of IMO roadmap)
//!
//! Algebraic fallback for the synthetic-geometry engine. When forward
//! saturation stalls, the tactic system can drop into barycentric
//! coordinates, compute quantities directly, and lift the result back to
//! a GeomPredicate.
//!
//! This module provides:
//! - `Barycentric { u, v, w }` — coordinates normalized so u + v + w = 1
//! - Conversion between Cartesian and barycentric in a given triangle
//! - Signed area of a triangle
//! - Classical triangle centers: centroid, incenter, circumcenter,
//!   orthocenter — each as both Cartesian points and barycentric triples
//! - Tests verifying each center against independent Euclidean constructions

use crate::hdc::computational_geometry::Point2D;

/// Tolerance for numerical geometry in this module. Tighter than GEOM_EPS
/// because barycentric computations are intentionally algebraic.
const BARY_EPS: f64 = 1e-9;

/// Barycentric coordinates of a point with respect to an implied reference
/// triangle. The invariant `u + v + w = 1` is maintained by all constructors
/// (construct via `Barycentric::new`, which normalizes).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Barycentric {
    pub u: f64,
    pub v: f64,
    pub w: f64,
}

impl Barycentric {
    /// Construct a normalized triple (u + v + w = 1). Panics if the sum is
    /// zero (degenerate).
    pub fn new(u: f64, v: f64, w: f64) -> Self {
        let sum = u + v + w;
        assert!(sum.abs() > BARY_EPS, "barycentric degenerate (u+v+w = 0)");
        Self {
            u: u / sum,
            v: v / sum,
            w: w / sum,
        }
    }

    /// Raw constructor without normalization — used internally for center
    /// formulas where the normalization is done in `to_cartesian`.
    fn raw(u: f64, v: f64, w: f64) -> Self {
        Self { u, v, w }
    }

    /// Convert to Cartesian given the reference triangle ABC.
    pub fn to_cartesian(&self, a: &Point2D, b: &Point2D, c: &Point2D) -> Point2D {
        let sum = self.u + self.v + self.w;
        let inv = if sum.abs() > BARY_EPS { 1.0 / sum } else { 0.0 };
        Point2D::new(
            (self.u * a.x + self.v * b.x + self.w * c.x) * inv,
            (self.u * a.y + self.v * b.y + self.w * c.y) * inv,
        )
    }

    /// Compute the barycentric coordinates of a point P with respect to the
    /// triangle ABC. Returns None if the triangle is degenerate.
    pub fn from_cartesian(p: &Point2D, a: &Point2D, b: &Point2D, c: &Point2D) -> Option<Self> {
        let denom = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
        if denom.abs() < BARY_EPS {
            return None;
        }
        let u = ((b.y - c.y) * (p.x - c.x) + (c.x - b.x) * (p.y - c.y)) / denom;
        let v = ((c.y - a.y) * (p.x - c.x) + (a.x - c.x) * (p.y - c.y)) / denom;
        let w = 1.0 - u - v;
        Some(Self { u, v, w })
    }
}

// ─── Signed area (twice) ────────────────────────────────────────────────────

/// Twice the signed area of triangle ABC. Positive if (A, B, C) is
/// counter-clockwise, negative if clockwise. Zero iff A, B, C are collinear.
pub fn signed_area_2x(a: &Point2D, b: &Point2D, c: &Point2D) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

/// Absolute area of triangle ABC.
pub fn triangle_area(a: &Point2D, b: &Point2D, c: &Point2D) -> f64 {
    signed_area_2x(a, b, c).abs() / 2.0
}

/// Side length opposite vertex (convention: side a is BC, side b is CA,
/// side c is AB — classical triangle notation).
pub fn side_lengths(a: &Point2D, b: &Point2D, c: &Point2D) -> (f64, f64, f64) {
    let la = b.distance(c); // side a is opposite A, which is BC
    let lb = c.distance(a);
    let lc = a.distance(b);
    (la, lb, lc)
}

// ─── Classical triangle centers ─────────────────────────────────────────────

/// Centroid (intersection of medians). Barycentric: (1, 1, 1) / 3.
pub fn centroid(a: &Point2D, b: &Point2D, c: &Point2D) -> Point2D {
    Barycentric::raw(1.0, 1.0, 1.0).to_cartesian(a, b, c)
}

/// Barycentric coordinates of the centroid: (1 : 1 : 1).
pub fn centroid_bary() -> Barycentric {
    Barycentric::new(1.0, 1.0, 1.0)
}

/// Incenter (intersection of angle bisectors). Barycentric: (a : b : c)
/// where a, b, c are the side lengths opposite the corresponding vertices.
pub fn incenter(a: &Point2D, b: &Point2D, c: &Point2D) -> Point2D {
    let (la, lb, lc) = side_lengths(a, b, c);
    Barycentric::raw(la, lb, lc).to_cartesian(a, b, c)
}

/// Barycentric coordinates of the incenter for a triangle with given side
/// lengths (a opposite A, b opposite B, c opposite C).
pub fn incenter_bary(a_side: f64, b_side: f64, c_side: f64) -> Barycentric {
    Barycentric::new(a_side, b_side, c_side)
}

/// Circumcenter (equidistant from all three vertices). Barycentric:
///   (a²(b² + c² − a²) : b²(c² + a² − b²) : c²(a² + b² − c²))
pub fn circumcenter(a: &Point2D, b: &Point2D, c: &Point2D) -> Point2D {
    let (la, lb, lc) = side_lengths(a, b, c);
    let (a2, b2, c2) = (la * la, lb * lb, lc * lc);
    let u = a2 * (b2 + c2 - a2);
    let v = b2 * (c2 + a2 - b2);
    let w = c2 * (a2 + b2 - c2);
    Barycentric::raw(u, v, w).to_cartesian(a, b, c)
}

/// Orthocenter (intersection of altitudes). Barycentric:
///   (tan(A) : tan(B) : tan(C))
/// or equivalently (1/(b²+c²−a²) : 1/(c²+a²−b²) : 1/(a²+b²−c²)) up to scale.
pub fn orthocenter(a: &Point2D, b: &Point2D, c: &Point2D) -> Point2D {
    let (la, lb, lc) = side_lengths(a, b, c);
    let (a2, b2, c2) = (la * la, lb * lb, lc * lc);
    // Use the secant-based formula to avoid division by zero in right triangles.
    // Instead we use the direct algebraic form:
    //   H = A + B + C − 2·O, where O is circumcenter — valid only for
    // specific cases. Simpler: compute via vector formula
    //   H = A + tan(B)tan(C) contributions …
    // To keep this robust for all triangle types, we compute the orthocenter
    // as the intersection of two altitudes algebraically.
    let foot_from_a = foot_of_perpendicular(a, b, c);
    let foot_from_b = foot_of_perpendicular(b, a, c);
    // Altitude from A: line from A through foot_from_a
    // Altitude from B: line from B through foot_from_b
    // Intersect.
    intersect_lines(a, &foot_from_a, b, &foot_from_b)
        .unwrap_or_else(|| {
            // Should not happen for non-degenerate triangles.
            let (_, _, _) = (a2, b2, c2);
            *a
        })
}

/// Foot of the perpendicular from `p` onto the line through `q` and `r`.
fn foot_of_perpendicular(p: &Point2D, q: &Point2D, r: &Point2D) -> Point2D {
    let (dx, dy) = (r.x - q.x, r.y - q.y);
    let d2 = dx * dx + dy * dy;
    if d2 < BARY_EPS {
        return *q;
    }
    let t = ((p.x - q.x) * dx + (p.y - q.y) * dy) / d2;
    Point2D::new(q.x + t * dx, q.y + t * dy)
}

/// Intersection of the line through (p1, p2) with the line through (p3, p4).
fn intersect_lines(p1: &Point2D, p2: &Point2D, p3: &Point2D, p4: &Point2D) -> Option<Point2D> {
    let denom = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x);
    if denom.abs() < BARY_EPS {
        return None;
    }
    let t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / denom;
    Some(Point2D::new(
        p1.x + t * (p2.x - p1.x),
        p1.y + t * (p2.y - p1.y),
    ))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, y: f64) -> Point2D {
        Point2D::new(x, y)
    }

    fn close(a: &Point2D, b: &Point2D) -> bool {
        (a.x - b.x).abs() < 1e-7 && (a.y - b.y).abs() < 1e-7
    }

    #[test]
    fn test_barycentric_round_trip() {
        let a = p(0.0, 0.0);
        let b = p(4.0, 0.0);
        let c = p(1.0, 3.0);
        let q = p(2.0, 1.0);
        let bary = Barycentric::from_cartesian(&q, &a, &b, &c).unwrap();
        assert!((bary.u + bary.v + bary.w - 1.0).abs() < 1e-9);
        let back = bary.to_cartesian(&a, &b, &c);
        assert!(close(&q, &back));
    }

    #[test]
    fn test_centroid_of_equilateral() {
        let a = p(0.0, 0.0);
        let b = p(2.0, 0.0);
        let c = p(1.0, 3.0_f64.sqrt());
        let g = centroid(&a, &b, &c);
        assert!(close(&g, &p(1.0, 3.0_f64.sqrt() / 3.0)));
    }

    #[test]
    fn test_incenter_of_right_triangle() {
        // 3-4-5 right triangle at origin.
        let a = p(0.0, 0.0);
        let b = p(4.0, 0.0);
        let c = p(0.0, 3.0);
        let i = incenter(&a, &b, &c);
        // Incircle radius r = (a + b − c)/2 for a right triangle where c is
        // the hypotenuse. Wait: r = (leg1 + leg2 − hyp)/2 = (3+4−5)/2 = 1.
        // The incenter is at (r, r) for a triangle right-angled at the origin.
        assert!(close(&i, &p(1.0, 1.0)));
    }

    #[test]
    fn test_circumcenter_equidistant() {
        let a = p(0.0, 0.0);
        let b = p(4.0, 0.0);
        let c = p(0.0, 3.0);
        let o = circumcenter(&a, &b, &c);
        let ra = o.distance(&a);
        let rb = o.distance(&b);
        let rc = o.distance(&c);
        assert!((ra - rb).abs() < 1e-7);
        assert!((rb - rc).abs() < 1e-7);
        // For a 3-4-5 right triangle, the circumradius is 2.5 and the
        // circumcenter is the midpoint of the hypotenuse.
        assert!((ra - 2.5).abs() < 1e-7);
        assert!(close(&o, &p(2.0, 1.5)));
    }

    #[test]
    fn test_orthocenter_right_triangle() {
        // In a right triangle, the orthocenter is the right-angle vertex.
        let a = p(0.0, 0.0);
        let b = p(4.0, 0.0);
        let c = p(0.0, 3.0);
        let h = orthocenter(&a, &b, &c);
        assert!(close(&h, &a));
    }

    #[test]
    fn test_orthocenter_acute_triangle() {
        // Equilateral triangle — orthocenter coincides with centroid.
        let a = p(0.0, 0.0);
        let b = p(2.0, 0.0);
        let c = p(1.0, 3.0_f64.sqrt());
        let h = orthocenter(&a, &b, &c);
        let g = centroid(&a, &b, &c);
        assert!(close(&h, &g));
    }

    #[test]
    fn test_signed_area_and_triangle_area() {
        let a = p(0.0, 0.0);
        let b = p(4.0, 0.0);
        let c = p(0.0, 3.0);
        assert!((triangle_area(&a, &b, &c) - 6.0).abs() < 1e-9);
        assert!(signed_area_2x(&a, &b, &c) > 0.0); // CCW
        assert!(signed_area_2x(&a, &c, &b) < 0.0); // CW
    }

    #[test]
    fn test_incenter_barycentric_sum_1() {
        let bary = incenter_bary(5.0, 4.0, 3.0);
        assert!((bary.u + bary.v + bary.w - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_centroid_barycentric_is_thirds() {
        let bary = centroid_bary();
        assert!((bary.u - 1.0 / 3.0).abs() < 1e-9);
        assert!((bary.v - 1.0 / 3.0).abs() < 1e-9);
        assert!((bary.w - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_from_cartesian_degenerate_returns_none() {
        let a = p(0.0, 0.0);
        let b = p(1.0, 0.0);
        let c = p(2.0, 0.0); // collinear with a, b
        assert!(Barycentric::from_cartesian(&p(0.5, 0.0), &a, &b, &c).is_none());
    }
}
