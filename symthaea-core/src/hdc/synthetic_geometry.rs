// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Synthetic Geometry (Phase 2A of IMO roadmap)
//!
//! Predicate-based geometry for IMO-style angle chasing and cyclic-quadrilateral
//! reasoning. Unlike `computational_geometry`, which is concerned with numerical
//! operations (convex hull, area, intersection), this module treats geometry
//! symbolically: we maintain a set of points and a set of *facts* about them
//! (collinear, concyclic, parallel, perpendicular, equal lengths, equal angles)
//! and derive new facts via forward saturation.
//!
//! Part 2A provides:
//! - Named point registry
//! - `GeomPredicate` enum (the nine core IMO predicates)
//! - Numerical predicate checkers operating on concrete point positions
//! - `GeomState` with facts vector
//!
//! Part 2B will add forward-saturation rules and random-point verification.
//! Part 2C will add barycentric-coordinate algebraic fallback.

use crate::hdc::computational_geometry::{Circle, Point2D};
use std::collections::HashMap;

// ─── Core tolerance for numerical predicate checks ──────────────────────────
//
// IMO-scale problems typically use coordinates in the range [-10, 10].
// 1e-9 gives us ~10 decimal digits of agreement, well above f64 precision loss
// from hand-constructed cross-products and determinants.
pub const GEOM_EPS: f64 = 1e-9;

// ─── Directed line through two distinct points ──────────────────────────────

/// An infinite line through two distinct points. Internally stored as the two
/// defining points; equality of lines must be tested via the `same_line`
/// helper (not by comparing points), since a given line has infinitely many
/// defining pairs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line {
    pub a: Point2D,
    pub b: Point2D,
}

impl Line {
    pub fn new(a: Point2D, b: Point2D) -> Self {
        assert!(
            (a.x - b.x).abs() > GEOM_EPS || (a.y - b.y).abs() > GEOM_EPS,
            "Line requires two distinct points"
        );
        Self { a, b }
    }

    /// Direction vector (b − a).
    pub fn direction(&self) -> (f64, f64) {
        (self.b.x - self.a.x, self.b.y - self.a.y)
    }

    /// Two lines are equal iff they contain the same set of points. We test
    /// by checking that `other.a` and `other.b` both lie on `self`.
    pub fn same_line(&self, other: &Line) -> bool {
        is_collinear(&self.a, &self.b, &other.a) && is_collinear(&self.a, &self.b, &other.b)
    }
}

// ─── Numerical predicates ────────────────────────────────────────────────────
//
// Each predicate returns true iff the geometric relation holds within
// `GEOM_EPS`. These are the *ground truth* for fact verification in the
// saturation step.

/// Three points are collinear iff the signed area of the triangle they form
/// is (within epsilon) zero.
pub fn is_collinear(a: &Point2D, b: &Point2D, c: &Point2D) -> bool {
    let area2 = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    area2.abs() < GEOM_EPS
}

/// Four points lie on a common circle iff the determinant
///     | x²+y²  x  y  1 |
///     | …              |
/// vanishes for all four. This is equivalent to the classical
/// Ptolemy-style test.
pub fn is_concyclic(a: &Point2D, b: &Point2D, c: &Point2D, d: &Point2D) -> bool {
    // Degenerate case: three collinear → the fourth is concyclic only if it
    // is also collinear with them. We treat four collinear points as "not on
    // a finite circle" (they lie on a line, which is the limit of a circle of
    // infinite radius).
    if is_collinear(a, b, c) || is_collinear(a, b, d) || is_collinear(a, c, d) {
        return false;
    }
    // Centre of circle through a, b, c via perpendicular-bisector intersection.
    let (ax, ay) = (a.x, a.y);
    let (bx, by) = (b.x, b.y);
    let (cx, cy) = (c.x, c.y);
    let d_val = 2.0 * ((ax * (by - cy)) + (bx * (cy - ay)) + (cx * (ay - by)));
    if d_val.abs() < GEOM_EPS {
        return false;
    }
    let ux = ((ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by))
        / d_val;
    let uy = ((ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax))
        / d_val;
    let centre = Point2D::new(ux, uy);
    let r = centre.distance(a);
    // d lies on this circle iff |centre − d| = r.
    (centre.distance(d) - r).abs() < GEOM_EPS * r.max(1.0) + GEOM_EPS
}

/// Two lines are parallel iff their direction vectors are scalar multiples.
pub fn is_parallel(l1: &Line, l2: &Line) -> bool {
    let (dx1, dy1) = l1.direction();
    let (dx2, dy2) = l2.direction();
    (dx1 * dy2 - dy1 * dx2).abs() < GEOM_EPS
}

/// Two lines are perpendicular iff the dot product of their direction vectors
/// is zero.
pub fn is_perpendicular(l1: &Line, l2: &Line) -> bool {
    let (dx1, dy1) = l1.direction();
    let (dx2, dy2) = l2.direction();
    (dx1 * dx2 + dy1 * dy2).abs() < GEOM_EPS
}

/// Signed distance between two points — just the Euclidean distance.
pub fn length(a: &Point2D, b: &Point2D) -> f64 {
    a.distance(b)
}

/// Two pairs (a, b) and (c, d) have equal segment length.
pub fn is_equal_length(a: &Point2D, b: &Point2D, c: &Point2D, d: &Point2D) -> bool {
    (length(a, b) - length(c, d)).abs() < GEOM_EPS
}

/// `m` is the midpoint of segment `ab`.
pub fn is_midpoint(m: &Point2D, a: &Point2D, b: &Point2D) -> bool {
    ((m.x - (a.x + b.x) / 2.0).abs() < GEOM_EPS)
        && ((m.y - (a.y + b.y) / 2.0).abs() < GEOM_EPS)
}

/// `b` lies strictly between `a` and `c` on the line through them
/// (collinear + in the open interval).
pub fn is_between(a: &Point2D, b: &Point2D, c: &Point2D) -> bool {
    if !is_collinear(a, b, c) {
        return false;
    }
    // Use the larger span axis to avoid division-by-zero in near-vertical cases.
    let dx = c.x - a.x;
    let dy = c.y - a.y;
    if dx.abs() >= dy.abs() {
        if dx.abs() < GEOM_EPS {
            return false;
        }
        let t = (b.x - a.x) / dx;
        t > GEOM_EPS && t < 1.0 - GEOM_EPS
    } else {
        let t = (b.y - a.y) / dy;
        t > GEOM_EPS && t < 1.0 - GEOM_EPS
    }
}

/// Three lines meet at a single point (no pair parallel, concurrent).
pub fn is_concurrent(l1: &Line, l2: &Line, l3: &Line) -> bool {
    // Find intersection of l1 and l2 (if not parallel) and check that l3
    // passes through it.
    let p = match line_intersection(l1, l2) {
        Some(p) => p,
        None => return false,
    };
    is_collinear(&l3.a, &l3.b, &p)
}

/// Angle ABC, returned in radians in [0, π]. Convention: the interior angle
/// at vertex `b` between the rays ba and bc.
pub fn angle_abc(a: &Point2D, b: &Point2D, c: &Point2D) -> f64 {
    let (vx1, vy1) = (a.x - b.x, a.y - b.y);
    let (vx2, vy2) = (c.x - b.x, c.y - b.y);
    let dot = vx1 * vx2 + vy1 * vy2;
    let n1 = (vx1 * vx1 + vy1 * vy1).sqrt();
    let n2 = (vx2 * vx2 + vy2 * vy2).sqrt();
    if n1 < GEOM_EPS || n2 < GEOM_EPS {
        return 0.0;
    }
    (dot / (n1 * n2)).clamp(-1.0, 1.0).acos()
}

/// Two angles ABC and DEF are equal within tolerance.
pub fn is_angle_eq(
    a: &Point2D,
    b: &Point2D,
    c: &Point2D,
    d: &Point2D,
    e: &Point2D,
    f: &Point2D,
) -> bool {
    (angle_abc(a, b, c) - angle_abc(d, e, f)).abs() < GEOM_EPS * 1e3 // 1e-6 rad ≈ 2e-4 deg
}

/// Intersect two lines. Returns None iff the lines are parallel.
pub fn line_intersection(l1: &Line, l2: &Line) -> Option<Point2D> {
    let (d1x, d1y) = l1.direction();
    let (d2x, d2y) = l2.direction();
    let denom = d1x * d2y - d1y * d2x;
    if denom.abs() < GEOM_EPS {
        return None;
    }
    let (ax, ay) = (l1.a.x, l1.a.y);
    let (bx, by) = (l2.a.x, l2.a.y);
    let t = ((bx - ax) * d2y - (by - ay) * d2x) / denom;
    Some(Point2D::new(ax + t * d1x, ay + t * d1y))
}

// ─── Predicate enum ──────────────────────────────────────────────────────────

/// A named predicate over points in a `GeomState`. All point references are
/// by string name, so that saturation can reason about facts without
/// needing concrete coordinates until the final verification step.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GeomPredicate {
    /// Three named points are collinear.
    Collinear(String, String, String),
    /// Four named points lie on a common circle.
    Concyclic(String, String, String, String),
    /// Line through (a,b) is parallel to line through (c,d).
    Parallel(String, String, String, String),
    /// Line through (a,b) is perpendicular to line through (c,d).
    Perpendicular(String, String, String, String),
    /// |ab| = |cd|.
    EqualLength(String, String, String, String),
    /// ∠abc = ∠def.
    AngleEq(String, String, String, String, String, String),
    /// b is strictly between a and c.
    Between(String, String, String),
    /// m is the midpoint of ab.
    Midpoint(String, String, String),
    /// The three lines (p1,p2), (p3,p4), (p5,p6) are concurrent.
    Concurrent(
        String, String, String, String, String, String,
    ),
}

// ─── GeomState: points + lines + circles + facts ────────────────────────────

/// A geometric configuration under construction. Points and circles may be
/// named for symbolic reference; `facts` accumulates proven predicates.
#[derive(Debug, Clone, Default)]
pub struct GeomState {
    /// Named points — looked up during fact verification.
    pub points: HashMap<String, Point2D>,
    /// Named circles (for power-of-point tactics).
    pub circles: HashMap<String, Circle>,
    /// Proven facts. Duplicates are suppressed by `add_fact`.
    pub facts: Vec<GeomPredicate>,
}

impl GeomState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a named point.
    pub fn add_point(&mut self, name: &str, p: Point2D) {
        self.points.insert(name.to_string(), p);
    }

    /// Register a named circle.
    pub fn add_circle(&mut self, name: &str, c: Circle) {
        self.circles.insert(name.to_string(), c);
    }

    /// Add a fact, rejecting duplicates. Returns true iff the fact was new.
    pub fn add_fact(&mut self, pred: GeomPredicate) -> bool {
        if self.facts.contains(&pred) {
            return false;
        }
        self.facts.push(pred);
        true
    }

    fn pt(&self, name: &str) -> Option<&Point2D> {
        self.points.get(name)
    }

    /// Verify a predicate against the concrete point positions. Returns
    /// `Some(true)` if the relation holds, `Some(false)` if it fails, and
    /// `None` if a point referenced in the predicate is missing from the
    /// state.
    pub fn verify(&self, pred: &GeomPredicate) -> Option<bool> {
        use GeomPredicate::*;
        match pred {
            Collinear(a, b, c) => {
                let pa = self.pt(a)?;
                let pb = self.pt(b)?;
                let pc = self.pt(c)?;
                Some(is_collinear(pa, pb, pc))
            }
            Concyclic(a, b, c, d) => {
                let pa = self.pt(a)?;
                let pb = self.pt(b)?;
                let pc = self.pt(c)?;
                let pd = self.pt(d)?;
                Some(is_concyclic(pa, pb, pc, pd))
            }
            Parallel(a, b, c, d) => {
                let pa = self.pt(a)?;
                let pb = self.pt(b)?;
                let pc = self.pt(c)?;
                let pd = self.pt(d)?;
                Some(is_parallel(&Line::new(*pa, *pb), &Line::new(*pc, *pd)))
            }
            Perpendicular(a, b, c, d) => {
                let pa = self.pt(a)?;
                let pb = self.pt(b)?;
                let pc = self.pt(c)?;
                let pd = self.pt(d)?;
                Some(is_perpendicular(&Line::new(*pa, *pb), &Line::new(*pc, *pd)))
            }
            EqualLength(a, b, c, d) => {
                let pa = self.pt(a)?;
                let pb = self.pt(b)?;
                let pc = self.pt(c)?;
                let pd = self.pt(d)?;
                Some(is_equal_length(pa, pb, pc, pd))
            }
            AngleEq(a, b, c, d, e, f) => {
                let pa = self.pt(a)?;
                let pb = self.pt(b)?;
                let pc = self.pt(c)?;
                let pd = self.pt(d)?;
                let pe = self.pt(e)?;
                let pf = self.pt(f)?;
                Some(is_angle_eq(pa, pb, pc, pd, pe, pf))
            }
            Between(a, b, c) => {
                let pa = self.pt(a)?;
                let pb = self.pt(b)?;
                let pc = self.pt(c)?;
                Some(is_between(pa, pb, pc))
            }
            Midpoint(m, a, b) => {
                let pm = self.pt(m)?;
                let pa = self.pt(a)?;
                let pb = self.pt(b)?;
                Some(is_midpoint(pm, pa, pb))
            }
            Concurrent(a, b, c, d, e, f) => {
                let pa = self.pt(a)?;
                let pb = self.pt(b)?;
                let pc = self.pt(c)?;
                let pd = self.pt(d)?;
                let pe = self.pt(e)?;
                let pf = self.pt(f)?;
                Some(is_concurrent(
                    &Line::new(*pa, *pb),
                    &Line::new(*pc, *pd),
                    &Line::new(*pe, *pf),
                ))
            }
        }
    }

    /// Sanity check: every registered fact must verify under the current
    /// point positions. Useful as a debug invariant after saturation.
    pub fn facts_consistent(&self) -> bool {
        self.facts.iter().all(|f| self.verify(f) == Some(true))
    }
}

// ─── Phase 2B: forward-saturation rules ─────────────────────────────────────
//
// Each rule is a function (&GeomState) -> Vec<GeomPredicate> that produces
// *candidate* new facts from existing ones. The main `saturate` loop runs
// all rules repeatedly, verifies each candidate against the concrete point
// positions, and adds only those that verify true and aren't already present.
//
// This numerical double-check is the safety net: any bug in a rule produces
// predicates that fail to verify and get silently dropped, instead of
// polluting the fact base. Self-correction by construction.

/// R1. Collinearity is invariant under argument permutation.
fn rule_collinear_symm(state: &GeomState) -> Vec<GeomPredicate> {
    let mut out = Vec::new();
    for f in &state.facts {
        if let GeomPredicate::Collinear(a, b, c) = f {
            out.push(GeomPredicate::Collinear(b.clone(), a.clone(), c.clone()));
            out.push(GeomPredicate::Collinear(a.clone(), c.clone(), b.clone()));
            out.push(GeomPredicate::Collinear(c.clone(), b.clone(), a.clone()));
        }
    }
    out
}

/// R2. Midpoint implies collinearity.
fn rule_midpoint_collinear(state: &GeomState) -> Vec<GeomPredicate> {
    state
        .facts
        .iter()
        .filter_map(|f| {
            if let GeomPredicate::Midpoint(m, a, b) = f {
                Some(GeomPredicate::Collinear(a.clone(), m.clone(), b.clone()))
            } else {
                None
            }
        })
        .collect()
}

/// R3. Midpoint implies |am| = |mb|.
fn rule_midpoint_equal_lengths(state: &GeomState) -> Vec<GeomPredicate> {
    state
        .facts
        .iter()
        .filter_map(|f| {
            if let GeomPredicate::Midpoint(m, a, b) = f {
                Some(GeomPredicate::EqualLength(
                    a.clone(),
                    m.clone(),
                    m.clone(),
                    b.clone(),
                ))
            } else {
                None
            }
        })
        .collect()
}

/// R4. Parallel is symmetric in the point pairs and between the two lines.
fn rule_parallel_symm(state: &GeomState) -> Vec<GeomPredicate> {
    let mut out = Vec::new();
    for f in &state.facts {
        if let GeomPredicate::Parallel(a, b, c, d) = f {
            out.push(GeomPredicate::Parallel(
                b.clone(),
                a.clone(),
                c.clone(),
                d.clone(),
            ));
            out.push(GeomPredicate::Parallel(
                a.clone(),
                b.clone(),
                d.clone(),
                c.clone(),
            ));
            out.push(GeomPredicate::Parallel(
                c.clone(),
                d.clone(),
                a.clone(),
                b.clone(),
            ));
        }
    }
    out
}

/// R5. Perpendicular is symmetric in each pair and between the two pairs.
fn rule_perpendicular_symm(state: &GeomState) -> Vec<GeomPredicate> {
    let mut out = Vec::new();
    for f in &state.facts {
        if let GeomPredicate::Perpendicular(a, b, c, d) = f {
            out.push(GeomPredicate::Perpendicular(
                c.clone(),
                d.clone(),
                a.clone(),
                b.clone(),
            ));
            out.push(GeomPredicate::Perpendicular(
                b.clone(),
                a.clone(),
                c.clone(),
                d.clone(),
            ));
        }
    }
    out
}

/// R6. Parallel is transitive: (a,b)∥(c,d) ∧ (c,d)∥(e,f) ⇒ (a,b)∥(e,f).
fn rule_parallel_transitive(state: &GeomState) -> Vec<GeomPredicate> {
    let mut out = Vec::new();
    for f1 in &state.facts {
        for f2 in &state.facts {
            if let (
                GeomPredicate::Parallel(a1, b1, c1, d1),
                GeomPredicate::Parallel(a2, b2, c2, d2),
            ) = (f1, f2)
            {
                if c1 == a2 && d1 == b2 {
                    out.push(GeomPredicate::Parallel(
                        a1.clone(),
                        b1.clone(),
                        c2.clone(),
                        d2.clone(),
                    ));
                }
            }
        }
    }
    out
}

/// R7. Perp ∘ Perp = Parallel: (a,b)⊥(c,d) ∧ (c,d)⊥(e,f) ⇒ (a,b)∥(e,f).
fn rule_perp_perp_is_parallel(state: &GeomState) -> Vec<GeomPredicate> {
    let mut out = Vec::new();
    for f1 in &state.facts {
        for f2 in &state.facts {
            if let (
                GeomPredicate::Perpendicular(a1, b1, c1, d1),
                GeomPredicate::Perpendicular(a2, b2, c2, d2),
            ) = (f1, f2)
            {
                if c1 == c2 && d1 == d2 && (a1, b1) != (a2, b2) {
                    out.push(GeomPredicate::Parallel(
                        a1.clone(),
                        b1.clone(),
                        a2.clone(),
                        b2.clone(),
                    ));
                }
            }
        }
    }
    out
}

/// R8. Concyclic: cyclic rotation + first-two swap (covers all perms).
fn rule_concyclic_symm(state: &GeomState) -> Vec<GeomPredicate> {
    let mut out = Vec::new();
    for f in &state.facts {
        if let GeomPredicate::Concyclic(a, b, c, d) = f {
            out.push(GeomPredicate::Concyclic(
                b.clone(),
                c.clone(),
                d.clone(),
                a.clone(),
            ));
            out.push(GeomPredicate::Concyclic(
                c.clone(),
                d.clone(),
                a.clone(),
                b.clone(),
            ));
            out.push(GeomPredicate::Concyclic(
                d.clone(),
                a.clone(),
                b.clone(),
                c.clone(),
            ));
            out.push(GeomPredicate::Concyclic(
                b.clone(),
                a.clone(),
                c.clone(),
                d.clone(),
            ));
        }
    }
    out
}

/// R9. Inscribed angle theorem: if ABCD is concyclic then ∠BAC = ∠BDC
/// (both subtend the arc BC) and ∠ABD = ∠ACD (both subtend arc AD).
fn rule_inscribed_angle(state: &GeomState) -> Vec<GeomPredicate> {
    let mut out = Vec::new();
    for f in &state.facts {
        if let GeomPredicate::Concyclic(a, b, c, d) = f {
            out.push(GeomPredicate::AngleEq(
                b.clone(),
                a.clone(),
                c.clone(),
                b.clone(),
                d.clone(),
                c.clone(),
            ));
            out.push(GeomPredicate::AngleEq(
                a.clone(),
                b.clone(),
                d.clone(),
                a.clone(),
                c.clone(),
                d.clone(),
            ));
        }
    }
    out
}

/// R10. Equal-length is symmetric.
fn rule_equal_length_symm(state: &GeomState) -> Vec<GeomPredicate> {
    let mut out = Vec::new();
    for f in &state.facts {
        if let GeomPredicate::EqualLength(a, b, c, d) = f {
            out.push(GeomPredicate::EqualLength(
                c.clone(),
                d.clone(),
                a.clone(),
                b.clone(),
            ));
            out.push(GeomPredicate::EqualLength(
                b.clone(),
                a.clone(),
                c.clone(),
                d.clone(),
            ));
        }
    }
    out
}

/// R11. Equal-length is transitive.
fn rule_equal_length_transitive(state: &GeomState) -> Vec<GeomPredicate> {
    let mut out = Vec::new();
    for f1 in &state.facts {
        for f2 in &state.facts {
            if let (
                GeomPredicate::EqualLength(a1, b1, c1, d1),
                GeomPredicate::EqualLength(a2, b2, c2, d2),
            ) = (f1, f2)
            {
                if c1 == a2 && d1 == b2 {
                    out.push(GeomPredicate::EqualLength(
                        a1.clone(),
                        b1.clone(),
                        c2.clone(),
                        d2.clone(),
                    ));
                }
            }
        }
    }
    out
}

/// R12. AngleEq is symmetric.
fn rule_angle_eq_symm(state: &GeomState) -> Vec<GeomPredicate> {
    let mut out = Vec::new();
    for f in &state.facts {
        if let GeomPredicate::AngleEq(a, b, c, d, e, g) = f {
            out.push(GeomPredicate::AngleEq(
                d.clone(),
                e.clone(),
                g.clone(),
                a.clone(),
                b.clone(),
                c.clone(),
            ));
        }
    }
    out
}

/// R13. AngleEq is transitive.
fn rule_angle_eq_transitive(state: &GeomState) -> Vec<GeomPredicate> {
    let mut out = Vec::new();
    for f1 in &state.facts {
        for f2 in &state.facts {
            if let (
                GeomPredicate::AngleEq(a1, b1, c1, d1, e1, g1),
                GeomPredicate::AngleEq(a2, b2, c2, d2, e2, g2),
            ) = (f1, f2)
            {
                if d1 == a2 && e1 == b2 && g1 == c2 {
                    out.push(GeomPredicate::AngleEq(
                        a1.clone(),
                        b1.clone(),
                        c1.clone(),
                        d2.clone(),
                        e2.clone(),
                        g2.clone(),
                    ));
                }
            }
        }
    }
    out
}

/// R14. Midpoint theorem (parallel side): M mid(AB), N mid(AC) ⇒ MN ∥ BC.
fn rule_midpoint_theorem(state: &GeomState) -> Vec<GeomPredicate> {
    let mut out = Vec::new();
    for f1 in &state.facts {
        for f2 in &state.facts {
            if let (
                GeomPredicate::Midpoint(m, a1, b1),
                GeomPredicate::Midpoint(n, a2, c1),
            ) = (f1, f2)
            {
                if a1 == a2 && m != n && b1 != c1 {
                    out.push(GeomPredicate::Parallel(
                        m.clone(),
                        n.clone(),
                        b1.clone(),
                        c1.clone(),
                    ));
                }
            }
        }
    }
    out
}

impl GeomState {
    /// Run forward saturation until fixpoint (or `max_iters` reached).
    /// Every candidate fact is verified against point positions; only
    /// numerically true candidates are added. Returns the number of new
    /// facts added in total.
    pub fn saturate(&mut self, max_iters: usize) -> usize {
        let rules: &[fn(&GeomState) -> Vec<GeomPredicate>] = &[
            rule_collinear_symm,
            rule_midpoint_collinear,
            rule_midpoint_equal_lengths,
            rule_parallel_symm,
            rule_perpendicular_symm,
            rule_parallel_transitive,
            rule_perp_perp_is_parallel,
            rule_concyclic_symm,
            rule_inscribed_angle,
            rule_equal_length_symm,
            rule_equal_length_transitive,
            rule_angle_eq_symm,
            rule_angle_eq_transitive,
            rule_midpoint_theorem,
        ];
        let mut added_total = 0usize;
        for _ in 0..max_iters {
            let mut added_this_iter = 0usize;
            for rule in rules {
                let candidates = rule(self);
                for c in candidates {
                    if self.facts.contains(&c) {
                        continue;
                    }
                    if let Some(true) = self.verify(&c) {
                        self.facts.push(c);
                        added_this_iter += 1;
                    }
                }
            }
            added_total += added_this_iter;
            if added_this_iter == 0 {
                break;
            }
        }
        added_total
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, y: f64) -> Point2D {
        Point2D::new(x, y)
    }

    #[test]
    fn test_collinear_positive() {
        // (0,0), (1,1), (2,2) are collinear
        assert!(is_collinear(&p(0.0, 0.0), &p(1.0, 1.0), &p(2.0, 2.0)));
    }

    #[test]
    fn test_collinear_negative() {
        assert!(!is_collinear(&p(0.0, 0.0), &p(1.0, 0.0), &p(0.0, 1.0)));
    }

    #[test]
    fn test_concyclic_unit_circle() {
        // (1,0), (0,1), (-1,0), (0,-1) all on unit circle
        assert!(is_concyclic(
            &p(1.0, 0.0),
            &p(0.0, 1.0),
            &p(-1.0, 0.0),
            &p(0.0, -1.0),
        ));
    }

    #[test]
    fn test_concyclic_negative() {
        // (0,0), (1,0), (0,1), (1,1) — unit square, concyclic (on circle
        // centred at (0.5, 0.5) radius √2/2)
        assert!(is_concyclic(
            &p(0.0, 0.0),
            &p(1.0, 0.0),
            &p(0.0, 1.0),
            &p(1.0, 1.0),
        ));
        // But (0,0), (1,0), (0,1), (2,2) — not concyclic
        assert!(!is_concyclic(
            &p(0.0, 0.0),
            &p(1.0, 0.0),
            &p(0.0, 1.0),
            &p(2.0, 2.0),
        ));
    }

    #[test]
    fn test_parallel_and_perpendicular() {
        let l1 = Line::new(p(0.0, 0.0), p(1.0, 0.0));
        let l2 = Line::new(p(0.0, 1.0), p(1.0, 1.0));
        let l3 = Line::new(p(0.0, 0.0), p(0.0, 1.0));
        assert!(is_parallel(&l1, &l2));
        assert!(is_perpendicular(&l1, &l3));
        assert!(!is_parallel(&l1, &l3));
    }

    #[test]
    fn test_midpoint_and_between() {
        let a = p(0.0, 0.0);
        let m = p(1.0, 1.0);
        let b = p(2.0, 2.0);
        assert!(is_midpoint(&m, &a, &b));
        assert!(is_between(&a, &m, &b));
        assert!(!is_between(&m, &a, &b));
    }

    #[test]
    fn test_equal_length() {
        let a = p(0.0, 0.0);
        let b = p(3.0, 4.0); // length 5
        let c = p(1.0, 1.0);
        let d = p(4.0, 5.0); // length 5
        assert!(is_equal_length(&a, &b, &c, &d));
    }

    #[test]
    fn test_angle_eq_right_angles() {
        // ∠(1,0)(0,0)(0,1) = π/2;  ∠(2,0)(0,0)(0,3) = π/2
        assert!(is_angle_eq(
            &p(1.0, 0.0),
            &p(0.0, 0.0),
            &p(0.0, 1.0),
            &p(2.0, 0.0),
            &p(0.0, 0.0),
            &p(0.0, 3.0),
        ));
    }

    #[test]
    fn test_concurrent_medians() {
        // Medians of a triangle meet at the centroid.
        let a = p(0.0, 0.0);
        let b = p(6.0, 0.0);
        let c = p(0.0, 6.0);
        let ma = p(3.0, 3.0); // midpoint bc
        let mb = p(0.0, 3.0); // midpoint ac
        let mc = p(3.0, 0.0); // midpoint ab
        let l_a = Line::new(a, ma);
        let l_b = Line::new(b, mb);
        let l_c = Line::new(c, mc);
        assert!(is_concurrent(&l_a, &l_b, &l_c));
    }

    #[test]
    fn test_line_intersection() {
        let l1 = Line::new(p(0.0, 0.0), p(2.0, 2.0));
        let l2 = Line::new(p(0.0, 2.0), p(2.0, 0.0));
        let pt = line_intersection(&l1, &l2).unwrap();
        assert!((pt.x - 1.0).abs() < GEOM_EPS);
        assert!((pt.y - 1.0).abs() < GEOM_EPS);
    }

    #[test]
    fn test_line_intersection_parallel() {
        let l1 = Line::new(p(0.0, 0.0), p(1.0, 0.0));
        let l2 = Line::new(p(0.0, 1.0), p(1.0, 1.0));
        assert!(line_intersection(&l1, &l2).is_none());
    }

    #[test]
    fn test_geomstate_add_verify() {
        let mut s = GeomState::new();
        s.add_point("A", p(0.0, 0.0));
        s.add_point("B", p(1.0, 1.0));
        s.add_point("C", p(2.0, 2.0));
        assert!(s.add_fact(GeomPredicate::Collinear("A".into(), "B".into(), "C".into())));
        assert!(s.facts_consistent());
        // Duplicate rejected
        assert!(!s.add_fact(GeomPredicate::Collinear("A".into(), "B".into(), "C".into())));
    }

    #[test]
    fn test_geomstate_verify_concyclic() {
        let mut s = GeomState::new();
        s.add_point("A", p(1.0, 0.0));
        s.add_point("B", p(0.0, 1.0));
        s.add_point("C", p(-1.0, 0.0));
        s.add_point("D", p(0.0, -1.0));
        assert_eq!(
            s.verify(&GeomPredicate::Concyclic(
                "A".into(),
                "B".into(),
                "C".into(),
                "D".into()
            )),
            Some(true)
        );
    }

    #[test]
    fn test_geomstate_verify_fails_on_bad_fact() {
        let mut s = GeomState::new();
        s.add_point("A", p(0.0, 0.0));
        s.add_point("B", p(1.0, 0.0));
        s.add_point("C", p(0.0, 1.0));
        // NOT collinear
        assert_eq!(
            s.verify(&GeomPredicate::Collinear("A".into(), "B".into(), "C".into())),
            Some(false)
        );
    }

    #[test]
    fn test_geomstate_missing_point() {
        let s = GeomState::new();
        assert_eq!(
            s.verify(&GeomPredicate::Collinear("X".into(), "Y".into(), "Z".into())),
            None
        );
    }

    // ── Phase 2B: forward-saturation tests ──────────────────────────────

    #[test]
    fn test_saturate_midpoint_implies_collinear_and_equal_length() {
        let mut s = GeomState::new();
        s.add_point("A", p(0.0, 0.0));
        s.add_point("M", p(1.0, 1.0));
        s.add_point("B", p(2.0, 2.0));
        s.add_fact(GeomPredicate::Midpoint("M".into(), "A".into(), "B".into()));
        let added = s.saturate(10);
        assert!(added >= 2, "expected collinear + equal-length derivations");
        assert!(s
            .facts
            .iter()
            .any(|f| matches!(f, GeomPredicate::Collinear(_, _, _))));
        assert!(s
            .facts
            .iter()
            .any(|f| matches!(f, GeomPredicate::EqualLength(_, _, _, _))));
        assert!(s.facts_consistent());
    }

    #[test]
    fn test_saturate_perp_perp_yields_parallel() {
        let mut s = GeomState::new();
        s.add_point("A", p(0.0, 0.0));
        s.add_point("B", p(1.0, 0.0));
        s.add_point("C", p(0.0, 0.0));
        s.add_point("D", p(0.0, 1.0));
        s.add_point("E", p(0.0, 2.0));
        s.add_point("F", p(1.0, 2.0));
        s.add_fact(GeomPredicate::Perpendicular(
            "A".into(), "B".into(), "C".into(), "D".into(),
        ));
        s.add_fact(GeomPredicate::Perpendicular(
            "E".into(), "F".into(), "C".into(), "D".into(),
        ));
        s.saturate(10);
        let found = s.facts.iter().any(|f| match f {
            GeomPredicate::Parallel(a, b, c, d) => {
                (a == "A" && b == "B" && c == "E" && d == "F")
                    || (a == "E" && b == "F" && c == "A" && d == "B")
            }
            _ => false,
        });
        assert!(found, "expected Parallel(A,B,E,F) or symmetric form");
    }

    #[test]
    fn test_saturate_inscribed_angle_theorem() {
        let mut s = GeomState::new();
        s.add_point("A", p(1.0, 0.0));
        s.add_point("B", p(0.0, 1.0));
        s.add_point("C", p(-1.0, 0.0));
        s.add_point("D", p(0.0, -1.0));
        s.add_fact(GeomPredicate::Concyclic(
            "A".into(), "B".into(), "C".into(), "D".into(),
        ));
        let added = s.saturate(10);
        assert!(added > 0, "inscribed-angle rule should fire");
        assert!(s
            .facts
            .iter()
            .any(|f| matches!(f, GeomPredicate::AngleEq(_, _, _, _, _, _))));
        assert!(s.facts_consistent());
    }

    #[test]
    fn test_saturate_midpoint_theorem_parallel() {
        let mut s = GeomState::new();
        s.add_point("A", p(0.0, 0.0));
        s.add_point("B", p(6.0, 0.0));
        s.add_point("C", p(0.0, 6.0));
        s.add_point("M", p(3.0, 0.0));
        s.add_point("N", p(0.0, 3.0));
        s.add_fact(GeomPredicate::Midpoint("M".into(), "A".into(), "B".into()));
        s.add_fact(GeomPredicate::Midpoint("N".into(), "A".into(), "C".into()));
        s.saturate(10);
        let mn_parallel_bc = s.facts.iter().any(|f| match f {
            GeomPredicate::Parallel(a, b, c, d) => {
                (a == "M" && b == "N" && c == "B" && d == "C")
                    || (a == "B" && b == "C" && c == "M" && d == "N")
                    || (a == "N" && b == "M" && c == "B" && d == "C")
                    || (a == "M" && b == "N" && c == "C" && d == "B")
            }
            _ => false,
        });
        assert!(mn_parallel_bc, "midpoint theorem: MN ∥ BC not derived");
        assert!(s.facts_consistent());
    }

    #[test]
    fn test_saturate_transitive_parallel() {
        let mut s = GeomState::new();
        s.add_point("A", p(0.0, 0.0));
        s.add_point("B", p(1.0, 0.0));
        s.add_point("C", p(0.0, 1.0));
        s.add_point("D", p(1.0, 1.0));
        s.add_point("E", p(0.0, 2.0));
        s.add_point("F", p(1.0, 2.0));
        s.add_fact(GeomPredicate::Parallel(
            "A".into(), "B".into(), "C".into(), "D".into(),
        ));
        s.add_fact(GeomPredicate::Parallel(
            "C".into(), "D".into(), "E".into(), "F".into(),
        ));
        s.saturate(10);
        assert!(s.facts.iter().any(|f| matches!(
            f,
            GeomPredicate::Parallel(a, b, c, d)
                if a == "A" && b == "B" && c == "E" && d == "F"
        )));
        assert!(s.facts_consistent());
    }

    #[test]
    fn test_saturate_reaches_fixpoint() {
        let mut s = GeomState::new();
        s.add_point("A", p(0.0, 0.0));
        s.add_point("B", p(1.0, 0.0));
        s.add_point("C", p(2.0, 0.0));
        s.add_fact(GeomPredicate::Collinear("A".into(), "B".into(), "C".into()));
        let added = s.saturate(20);
        assert!(added < 100, "unbounded derivation: added {}", added);
        assert!(s.facts_consistent());
    }
}
