// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Research-only identified-set propagation for RH-006.
//!
//! An explicitly characterized non-singleton identified set is PI. A target
//! for which the current analysis has not characterized a qualified set is NI.
//! These are deliberately not synonyms.
//!
//! Downstream decision stability and mathematical regularity are separate
//! dimensions. A PI target may support a valid decision claim when every
//! admissible value yields the same decision, even though point estimation of
//! the target remains forbidden.
//!
//! For the RH-006 ray slice
//! Delta(tu;s) = A t^2 + (B0 + B1 s)t + (C0 + C1 s + C2 s^2)
//!
//! the discriminant is itself quadratic in s. This module propagates a scalar
//! identified interval through that polynomial structure without using a
//! dense numerical grid as the definition of the resulting set image.

const TOL: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClosedInterval {
    pub lower: f64,
    pub upper: f64,
}

impl ClosedInterval {
    pub fn new(lower: f64, upper: f64) -> Result<Self, &'static str> {
        if !lower.is_finite() || !upper.is_finite() {
            return Err("interval endpoints must be finite");
        }
        if lower > upper {
            return Err("interval lower endpoint exceeds upper endpoint");
        }
        Ok(Self { lower, upper })
    }

    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower - TOL && value <= self.upper + TOL
    }
}

/// Identification status with mutually exclusive operational semantics.
///
/// PI means the analysis has an explicit non-singleton identified set.
/// NI means the analysis has not produced a qualified set characterization.
/// Merely exhibiting two observationally equivalent witnesses with different
/// target values establishes failure of point identification, but does not by
/// itself establish a sharp or qualified set characterization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdentificationLevel {
    Observable,
    ModelIdentified,
    PartiallyIdentified,
    NotIdentified,
}

impl IdentificationLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Observable => "O",
            Self::ModelIdentified => "MI",
            Self::PartiallyIdentified => "PI",
            Self::NotIdentified => "NI",
        }
    }

    pub fn permits_point_target_claim(self) -> bool {
        matches!(self, Self::Observable | Self::ModelIdentified)
    }
}

/// Qualification of an identified-set representation.
///
/// The distinction is semantic, not numerical:
/// - WitnessOnly: demonstrates admissible examples, but is not a set bound.
/// - InnerApproximation: guaranteed subset of the true identified set.
/// - OuterApproximation: guaranteed superset of the true identified set.
/// - Sharp: exact identified set under the declared assumptions.
/// Origin of a propagated parameter region.
///
/// A geometrically exact region can still be only a maintained restriction or
/// sensitivity-analysis domain. The origin must be explicit before the region
/// can be described as an identified set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOrigin {
    ObservationalEquivalence,
    MaintainedRestriction,
    SensitivityAnalysis,
    Unknown,
}

impl SetOrigin {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ObservationalEquivalence => "observational-equivalence-derived",
            Self::MaintainedRestriction => "maintained-restriction-region",
            Self::SensitivityAnalysis => "sensitivity-analysis-region",
            Self::Unknown => "unknown",
        }
    }

    pub fn supports_identified_set_claim(self) -> bool {
        matches!(self, Self::ObservationalEquivalence)
    }
}

/// Typed provenance for a propagated parameter region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetCertificate {
    pub representation: SetRepresentation,
    pub origin: SetOrigin,
}

impl SetCertificate {
    pub fn permits_identified_set_claim(self) -> bool {
        self.origin.supports_identified_set_claim()
            && !matches!(self.representation, SetRepresentation::WitnessOnly)
    }

    pub fn permits_stable_identified_decision(self) -> bool {
        self.permits_identified_set_claim()
            && self.representation.permits_stable_decision_claim()
    }

    pub fn permits_unstable_identified_decision(self) -> bool {
        self.permits_identified_set_claim()
            && self.representation.permits_unstable_decision_claim()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetRepresentation {
    WitnessOnly,
    InnerApproximation,
    OuterApproximation,
    Sharp,
}

impl SetRepresentation {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WitnessOnly => "witness-only",
            Self::InnerApproximation => "inner-approximation",
            Self::OuterApproximation => "outer-approximation",
            Self::Sharp => "sharp",
        }
    }

    /// A universal stable-decision claim is safe from an outer bound only:
    /// if every point in a superset has the same decision, the unknown true
    /// set must have that decision too.
    pub fn permits_stable_decision_claim(self) -> bool {
        matches!(self, Self::OuterApproximation | Self::Sharp)
    }

    /// A claim that the decision is unstable is safe from an inner bound only:
    /// if the subset already contains conflicting admissible decisions, the
    /// true identified set must contain them as well.
    pub fn permits_unstable_decision_claim(self) -> bool {
        matches!(self, Self::InnerApproximation | Self::Sharp)
    }

    pub fn permits_point_target_claim(self) -> bool {
        matches!(self, Self::Sharp)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionStability {
    Stable,
    Unstable,
    Unknown,
}

impl DecisionStability {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stable => "decision-stable-under-identification-set",
            Self::Unstable => "decision-unstable-under-identification-set",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegularityStatus {
    Regular,
    Boundary,
    InvalidDomain,
    Unknown,
}

impl RegularityStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Regular => "regular",
            Self::Boundary => "nonregular-boundary",
            Self::InvalidDomain => "invalid-domain",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RayRegime {
    NoRealBranch,
    DoubleRoot,
    TwoRealBranches,
}

impl RayRegime {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoRealBranch => "no-real-branch",
            Self::DoubleRoot => "double-root",
            Self::TwoRealBranches => "two-real-branches",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadraticPolynomial {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl QuadraticPolynomial {
    pub fn evaluate(&self, x: f64) -> f64 {
        self.a * x * x + self.b * x + self.c
    }

    /// Exact scalar image for a quadratic on a closed interval:
    /// endpoint values plus the vertex when it lies in the interval.
    pub fn range_on(&self, domain: ClosedInterval) -> Result<ClosedInterval, &'static str> {
        let mut candidates = vec![domain.lower, domain.upper];

        if self.a.abs() > TOL {
            let vertex = -self.b / (2.0 * self.a);
            if domain.contains(vertex) {
                candidates.push(vertex.clamp(domain.lower, domain.upper));
            }
        }

        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for x in candidates {
            let value = self.evaluate(x);
            if !value.is_finite() {
                return Err("polynomial image is non-finite");
            }
            min = min.min(value);
            max = max.max(value);
        }

        ClosedInterval::new(min, max)
    }

    /// Analytic real roots restricted to a closed interval.
    pub fn roots_on(&self, domain: ClosedInterval) -> Vec<f64> {
        let mut roots = Vec::new();

        if self.a.abs() <= TOL {
            if self.b.abs() > TOL {
                let root = -self.c / self.b;
                if domain.contains(root) {
                    roots.push(root.clamp(domain.lower, domain.upper));
                }
            }
        } else {
            let discriminant = self.b * self.b - 4.0 * self.a * self.c;
            if discriminant >= -TOL {
                let root_disc = discriminant.max(0.0).sqrt();
                // Stable quadratic formula: avoid cancellation when b and
                // sqrt(discriminant) have the same sign. This matters near
                // narrowly supported RH-006 branch boundaries.
                let q = -0.5 * (self.b + root_disc.copysign(self.b));
                let candidate_roots = if q.abs() <= TOL {
                    vec![-self.b / (2.0 * self.a)]
                } else {
                    vec![q / self.a, self.c / q]
                };
                for root in candidate_roots {
                    if root.is_finite() && domain.contains(root) {
                        roots.push(root.clamp(domain.lower, domain.upper));
                    }
                }
            }
        }

        roots.sort_by(f64::total_cmp);
        roots.dedup_by(|left, right| (*left - *right).abs() <= TOL);
        roots
    }
}

/// Fixed-ray RH-006 equal-risk surface under an affine scalar nuisance
/// strength coordinate s.
///
/// Delta(tu;s) = A t^2 + B(s)t + C(s),
/// B(s) = B0 + B1 s,
/// C(s) = C0 + C1 s + C2 s^2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaySurface {
    pub a: f64,
    pub b0: f64,
    pub b1: f64,
    pub c0: f64,
    pub c1: f64,
    pub c2: f64,
}

impl RaySurface {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.a.is_finite() || self.a <= TOL {
            return Err("quadratic ray curvature A must be strictly positive");
        }
        if [self.b0, self.b1, self.c0, self.c1, self.c2]
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err("ray coefficients must be finite");
        }
        Ok(())
    }

    pub fn b(&self, s: f64) -> f64 {
        self.b0 + self.b1 * s
    }

    pub fn c(&self, s: f64) -> f64 {
        self.c0 + self.c1 * s + self.c2 * s * s
    }

    /// D(s) = B(s)^2 - 4 A C(s), which is quadratic in s.
    pub fn discriminant_polynomial(&self) -> QuadraticPolynomial {
        QuadraticPolynomial {
            a: self.b1 * self.b1 - 4.0 * self.a * self.c2,
            b: 2.0 * self.b0 * self.b1 - 4.0 * self.a * self.c1,
            c: self.b0 * self.b0 - 4.0 * self.a * self.c0,
        }
    }

    pub fn discriminant(&self, s: f64) -> f64 {
        self.discriminant_polynomial().evaluate(s)
    }

    /// Locate nuisance values at which a positive ray root reaches a frozen
    /// effect-radius threshold.
    ///
    /// For gamma = t*u and rho_effect = t*directional_effect_scale/sigma_ref,
    /// a positive target radius rho corresponds to the fixed t threshold
    /// rho*sigma_ref/directional_effect_scale. Substituting that t into
    /// Delta(tu;s)=0 leaves a quadratic in s, so these boundaries are found
    /// analytically rather than by scanning the nuisance domain.
    pub fn effect_radius_boundary_points(
        &self,
        domain: ClosedInterval,
        effect_radius: f64,
        directional_effect_scale: f64,
        sigma_ref: f64,
    ) -> Result<Vec<f64>, &'static str> {
        self.validate()?;
        domain.validate().map_err(|_| "invalid nuisance domain")?;
        if !effect_radius.is_finite()
            || effect_radius < 0.0
            || !directional_effect_scale.is_finite()
            || directional_effect_scale <= TOL
            || !sigma_ref.is_finite()
            || sigma_ref <= TOL
        {
            return Err("effect-radius scale and threshold must be finite and positive");
        }

        let t = effect_radius * sigma_ref / directional_effect_scale;
        if !t.is_finite() {
            return Err("effect-radius coefficient threshold is non-finite");
        }

        let threshold_surface = QuadraticPolynomial {
            a: self.c2,
            b: self.c1 + self.b1 * t,
            c: self.c0 + self.b0 * t + self.a * t * t,
        };
        Ok(threshold_surface.roots_on(domain))
    }

    fn point_structure(&self, s: f64) -> (RayRegime, u8) {
        let b = self.b(s);
        let discriminant = self.discriminant(s);

        if discriminant < -TOL {
            return (RayRegime::NoRealBranch, 0);
        }

        if discriminant.abs() <= TOL {
            let root = -b / (2.0 * self.a);
            return (
                RayRegime::DoubleRoot,
                if root > TOL { 1 } else { 0 },
            );
        }

        // With A > 0, positive-root count can be determined without
        // evaluating the two roots, avoiding cancellation when one root is
        // extremely close to zero.
        let positive_roots = if self.c(s) < -TOL {
            1
        } else if self.c(s) > TOL {
            if b < -TOL { 2 } else { 0 }
        } else if b < -TOL {
            1
        } else {
            0
        };

        (RayRegime::TwoRealBranches, positive_roots)
    }

    /// Critical nuisance values are roots of D, C, and B plus endpoints.
    /// Between them the relevant polynomial signs are invariant.
    pub fn critical_points(&self, domain: ClosedInterval) -> Vec<f64> {
        let mut points = vec![domain.lower, domain.upper];
        points.extend(self.discriminant_polynomial().roots_on(domain));
        points.extend(
            QuadraticPolynomial {
                a: self.c2,
                b: self.c1,
                c: self.c0,
            }
            .roots_on(domain),
        );
        points.extend(
            QuadraticPolynomial {
                a: 0.0,
                b: self.b1,
                c: self.b0,
            }
            .roots_on(domain),
        );

        points.sort_by(f64::total_cmp);
        points.dedup_by(|left, right| (*left - *right).abs() <= TOL);
        points
    }

    pub fn classify(
        &self,
        domain: ClosedInterval,
    ) -> Result<SurfacePropagationSummary, &'static str> {
        self.validate()?;
        domain.validate().map_err(|_| "invalid nuisance domain")?;

        let critical = self.critical_points(domain);
        let mut positive_root_counts = Vec::new();
        let mut regimes = Vec::new();

        for point in critical.iter().copied() {
            let (regime, positive_root_count) = self.point_structure(point);
            positive_root_counts.push(positive_root_count);
            regimes.push(regime);
        }

        for pair in critical.windows(2) {
            let left = pair[0];
            let right = pair[1];
            if right - left <= TOL {
                continue;
            }
            let midpoint = 0.5 * (left + right);
            let (regime, positive_root_count) = self.point_structure(midpoint);
            positive_root_counts.push(positive_root_count);
            regimes.push(regime);
        }

        positive_root_counts.sort_unstable();
        positive_root_counts.dedup();
        regimes.sort_unstable();
        regimes.dedup();

        let discriminant = self.discriminant_polynomial();
        let discriminant_image = discriminant.range_on(domain)?;
        let discriminant_boundary_reachable = critical
            .iter()
            .copied()
            .any(|point| self.discriminant(point).abs() <= TOL);
        let positive_root_boundary_reachable = critical
            .iter()
            .copied()
            .any(|point| self.c(point).abs() <= TOL);

        let decision_status = if positive_root_counts.len() == 1
            && positive_root_counts.first().copied().unwrap_or(0) > 0
        {
            DecisionStability::Stable
        } else {
            DecisionStability::Unstable
        };

        let regularity_status = if discriminant_boundary_reachable {
            RegularityStatus::Boundary
        } else {
            RegularityStatus::Regular
        };

        Ok(SurfacePropagationSummary {
            discriminant_image,
            positive_root_counts,
            regimes,
            decision_status,
            regularity_status,
            discriminant_boundary_reachable,
            positive_root_boundary_reachable,
            critical_points: critical,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SurfacePropagationSummary {
    pub discriminant_image: ClosedInterval,
    pub positive_root_counts: Vec<u8>,
    pub regimes: Vec<RayRegime>,
    pub decision_status: DecisionStability,
    pub regularity_status: RegularityStatus,
    pub discriminant_boundary_reachable: bool,
    pub positive_root_boundary_reachable: bool,
    pub critical_points: Vec<f64>,
}

impl SurfacePropagationSummary {
    /// A downstream decision claim must be checked against the
    /// approximation polarity. Calling code must not infer claimability from
    /// decision stability alone.
    pub fn permits_identified_decision_claim(
        &self,
        certificate: SetCertificate,
    ) -> bool {
        let decision_ok = match self.decision_status {
            DecisionStability::Stable => certificate.permits_stable_identified_decision(),
            DecisionStability::Unstable => certificate.permits_unstable_identified_decision(),
            DecisionStability::Unknown => false,
        };
        decision_ok && matches!(self.regularity_status, RegularityStatus::Regular)
    }

    /// Apply approximation polarity before allowing a downstream decision claim.
    pub fn permits_decision_claim_with_representation(
        &self,
        representation: SetRepresentation,
    ) -> bool {
        let decision_ok = match self.decision_status {
            DecisionStability::Stable => representation.permits_stable_decision_claim(),
            DecisionStability::Unstable => representation.permits_unstable_decision_claim(),
            DecisionStability::Unknown => false,
        };
        decision_ok && matches!(self.regularity_status, RegularityStatus::Regular)
    }

    pub fn permits_point_target_claim(&self, identification: IdentificationLevel) -> bool {
        identification.permits_point_target_claim()
    }
}


/// Exact two-dimensional quadratic surface over a closed axis-aligned box.
///
/// f(x,y) = ax² + by² + cxy + dx + ey + f.
///
/// The global extrema of a quadratic on a rectangle occur at a corner, at an
/// edge stationary point, or at an interior stationary point. This makes the
/// 2-D box image exactly computable without gridding. Degenerate interior
/// stationary sets do not require a separate value because any non-isolated
/// stationary extremum has a boundary extremum with the same value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadraticBox2d {
    pub a: f64,
    pub b: f64,
    pub c_xy: f64,
    pub d: f64,
    pub e: f64,
    pub f0: f64,
}

impl QuadraticBox2d {
    pub fn validate(&self) -> Result<(), &'static str> {
        if [self.a, self.b, self.c_xy, self.d, self.e, self.f0]
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err("quadratic coefficients must be finite");
        }
        Ok(())
    }

    pub fn evaluate(&self, x: f64, y: f64) -> f64 {
        self.a * x * x
            + self.b * y * y
            + self.c_xy * x * y
            + self.d * x
            + self.e * y
            + self.f0
    }

    pub fn range_on(
        &self,
        x_domain: ClosedInterval,
        y_domain: ClosedInterval,
    ) -> Result<ClosedInterval, &'static str> {
        self.validate()?;
        x_domain.validate().map_err(|_| "invalid x domain")?;
        y_domain.validate().map_err(|_| "invalid y domain")?;

        let mut values = Vec::with_capacity(10);

        for (x, y) in [
            (x_domain.lower, y_domain.lower),
            (x_domain.lower, y_domain.upper),
            (x_domain.upper, y_domain.lower),
            (x_domain.upper, y_domain.upper),
        ] {
            values.push(self.evaluate(x, y));
        }

        for x in [x_domain.lower, x_domain.upper] {
            let edge = QuadraticPolynomial {
                a: self.b,
                b: self.c_xy * x + self.e,
                c: self.a * x * x + self.d * x + self.f0,
            }
            .range_on(y_domain)?;
            values.extend([edge.lower, edge.upper]);
        }

        for y in [y_domain.lower, y_domain.upper] {
            let edge = QuadraticPolynomial {
                a: self.a,
                b: self.c_xy * y + self.d,
                c: self.b * y * y + self.e * y + self.f0,
            }
            .range_on(x_domain)?;
            values.extend([edge.lower, edge.upper]);
        }

        let determinant = 4.0 * self.a * self.b - self.c_xy * self.c_xy;
        if determinant.abs() > TOL {
            let stationary_x =
                (self.c_xy * self.e - 2.0 * self.b * self.d) / determinant;
            let stationary_y =
                (self.c_xy * self.d - 2.0 * self.a * self.e) / determinant;
            if x_domain.contains(stationary_x) && y_domain.contains(stationary_y) {
                values.push(self.evaluate(stationary_x, stationary_y));
            }
        }

        if values.iter().any(|value| !value.is_finite()) {
            return Err("quadratic box image is non-finite");
        }

        ClosedInterval::new(
            values.iter().copied().fold(f64::INFINITY, f64::min),
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    }
}

/// Exact quadratic image over a closed convex polygon.
///
/// The global extrema of a quadratic on a compact convex polygon occur at a
/// vertex, an edge stationary point, or an admissible interior stationary
/// point. This preserves joint parameter constraints that a coordinatewise
/// rectangle would discard.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvexPolygon2d {
    pub vertices: Vec<(f64, f64)>,
}

impl ConvexPolygon2d {
    pub fn new(vertices: Vec<(f64, f64)>) -> Result<Self, &'static str> {
        if vertices.len() < 3 {
            return Err("convex polygon needs at least three vertices");
        }
        if vertices
            .iter()
            .any(|(x, y)| !x.is_finite() || !y.is_finite())
        {
            return Err("polygon vertices must be finite");
        }

        let mut turn_sign = 0.0_f64;
        for i in 0..vertices.len() {
            let a = vertices[i];
            let b = vertices[(i + 1) % vertices.len()];
            let d = vertices[(i + 2) % vertices.len()];
            let cross = (b.0 - a.0) * (d.1 - b.1) - (b.1 - a.1) * (d.0 - b.0);
            if cross.abs() > TOL {
                if turn_sign == 0.0 {
                    turn_sign = cross.signum();
                } else if cross.signum() != turn_sign {
                    return Err("polygon vertices are not convex and consistently ordered");
                }
            }
        }
        if turn_sign == 0.0 {
            return Err("polygon vertices are collinear");
        }

        Ok(Self { vertices })
    }

    fn contains(&self, point: (f64, f64)) -> bool {
        let mut sign = 0.0_f64;
        for i in 0..self.vertices.len() {
            let a = self.vertices[i];
            let b = self.vertices[(i + 1) % self.vertices.len()];
            let cross =
                (b.0 - a.0) * (point.1 - a.1) - (b.1 - a.1) * (point.0 - a.0);
            if cross.abs() <= TOL {
                continue;
            }
            if sign == 0.0 {
                sign = cross.signum();
            } else if cross.signum() != sign {
                return false;
            }
        }
        true
    }

    pub fn range_of(
        &self,
        surface: &QuadraticBox2d,
    ) -> Result<ClosedInterval, &'static str> {
        surface.validate()?;

        let mut values = self
            .vertices
            .iter()
            .map(|(x, y)| surface.evaluate(*x, *y))
            .collect::<Vec<_>>();

        for i in 0..self.vertices.len() {
            let (x0, y0) = self.vertices[i];
            let (x1, y1) = self.vertices[(i + 1) % self.vertices.len()];
            let dx = x1 - x0;
            let dy = y1 - y0;

            let edge = QuadraticPolynomial {
                a: surface.a_xx * dx * dx
                    + surface.b_yy * dy * dy
                    + surface.c_xy * dx * dy,
                b: 2.0 * surface.a_xx * x0 * dx
                    + 2.0 * surface.b_yy * y0 * dy
                    + surface.c_xy * (x0 * dy + y0 * dx)
                    + surface.d_x * dx
                    + surface.e_y * dy,
                c: surface.value(x0, y0),
            };
            let roots = edge.roots_on(ClosedInterval::new(0.0, 1.0)?);
            for t in roots {
                values.push(surface.value(x0 + t * dx, y0 + t * dy));
            }
        }

        let determinant =
            4.0 * surface.a_xx * surface.b_yy - surface.c_xy * surface.c_xy;
        if determinant.abs() > TOL {
            let stationary = (
                (surface.c_xy * surface.e_y - 2.0 * surface.b_yy * surface.d_x)
                    / determinant,
                (surface.c_xy * surface.d_x - 2.0 * surface.a_xx * surface.e_y)
                    / determinant,
            );
            if self.contains(stationary) {
                values.push(surface.evaluate(stationary.0, stationary.1));
            }
        }

        if values.iter().any(|value| !value.is_finite()) {
            return Err("polygon quadratic image is non-finite");
        }

        ClosedInterval::new(
            values.iter().copied().fold(f64::INFINITY, f64::min),
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    }
}

impl RaySurface {
    /// Propagate a two-parameter nuisance surface through a convex polygon.
    ///
    /// The parameter set itself may encode joint restrictions such as
    /// x >= 0, y >= 0, x + y <= 1; it is not replaced by a coordinatewise box.
    pub fn discriminant_image_over_polygon(
        &self,
        polygon: &ConvexPolygon2d,
        b_x: f64,
        b_y: f64,
        c_x: f64,
        c_y: f64,
        c_xx: f64,
        c_xy: f64,
        c_yy: f64,
    ) -> Result<ClosedInterval, &'static str> {
        self.validate()?;
        for value in [b_x, b_y, c_x, c_y, c_xx, c_xy, c_yy] {
            if !value.is_finite() {
                return Err("two-parameter coefficients must be finite");
            }
        }

        let discriminant = QuadraticBox2d {
            a_xx: b_x * b_x - 4.0 * self.a * c_xx,
            b_yy: b_y * b_y - 4.0 * self.a * c_yy,
            c_xy: 2.0 * b_x * b_y - 4.0 * self.a * c_xy,
            d_x: 2.0 * self.b0 * b_x - 4.0 * self.a * c_x,
            e_y: 2.0 * self.b0 * b_y - 4.0 * self.a * c_y,
            f0: self.b0 * self.b0 - 4.0 * self.a * self.c0,
        };
        polygon.range_of(&discriminant)
    }
}

/// Sound positive-root decision bounds over a two-dimensional nuisance box.
///
/// The returned count is only exact when the signs of D, B, and C establish
/// the same positive-root count everywhere. Otherwise the result is unknown
/// and the caller must fail closed rather than infer stability from a grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxDecisionCount {
    Zero,
    One,
    Two,
    Unknown,
}

impl BoxDecisionCount {
    pub fn as_u8(self) -> Option<u8> {
        match self {
            Self::Zero => Some(0),
            Self::One => Some(1),
            Self::Two => Some(2),
            Self::Unknown => None,
        }
    }

    pub fn is_resolved(self) -> bool {
        !matches!(self, Self::Unknown)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TwoParameterRaySummary {
    pub discriminant_image: ClosedInterval,
    pub b_image: ClosedInterval,
    pub c_image: ClosedInterval,
    pub decision_count: BoxDecisionCount,
    pub boundary_reachable: bool,
    pub domain: [ClosedInterval; 2],
}

impl RaySurface {
    /// Generalized two-parameter nuisance slice:
    ///
    /// Delta(tu;x,y) = A t² + B(x,y)t + C(x,y)
    ///
    /// where B is affine and C is quadratic. The discriminant is therefore a
    /// quadratic surface in (x,y). Exact surface range is obtained over the
    /// full box; no grid defines the answer.
    pub fn classify_box_2d(
        &self,
        x_domain: ClosedInterval,
        y_domain: ClosedInterval,
        b_x: f64,
        b_y: f64,
        c_x: f64,
        c_y: f64,
        c_xx: f64,
        c_xy: f64,
        c_yy: f64,
    ) -> Result<TwoParameterRaySummary, &'static str> {
        self.validate()?;
        x_domain.validate().map_err(|_| "invalid x domain")?;
        y_domain.validate().map_err(|_| "invalid y domain")?;

        for value in [b_x, b_y, c_x, c_y, c_xx, c_xy, c_yy] {
            if !value.is_finite() {
                return Err("two-parameter coefficients must be finite");
            }
        }

        let b_surface = QuadraticBox2d {
            a: 0.0,
            b: 0.0,
            c_xy: 0.0,
            d: b_x,
            e: b_y,
            f0: self.b0,
        };
        let c_surface = QuadraticBox2d {
            a: c_xx,
            b: c_yy,
            c_xy,
            d: c_x,
            e: c_y,
            f0: self.c0,
        };

        let b_image = b_surface.range_on(x_domain, y_domain)?;
        let c_image = c_surface.range_on(x_domain, y_domain)?;

        // B² is quadratic because B is affine.
        // D = B² - 4A C remains quadratic in (x,y).
        let d_surface = QuadraticBox2d {
            a: b_x * b_x - 4.0 * self.a * c_xx,
            b: b_y * b_y - 4.0 * self.a * c_yy,
            c_xy: 2.0 * b_x * b_y - 4.0 * self.a * c_xy,
            d: 2.0 * self.b0 * b_x - 4.0 * self.a * c_x,
            e: 2.0 * self.b0 * b_y - 4.0 * self.a * c_y,
            f0: self.b0 * self.b0 - 4.0 * self.a * self.c0,
        };
        let d_image = d_surface.range_on(x_domain, y_domain)?;

        let decision_count = if d_image.upper < -TOL {
            BoxDecisionCount::Zero
        } else if d_image.lower > TOL && c_image.upper < -TOL {
            BoxDecisionCount::One
        } else if d_image.lower > TOL
            && c_image.lower > TOL
            && b_image.upper < -TOL
        {
            BoxDecisionCount::Two
        } else if d_image.lower > TOL
            && c_image.lower > TOL
            && b_image.lower > TOL
        {
            BoxDecisionCount::Zero
        } else {
            BoxDecisionCount::Unknown
        };

        let boundary_reachable =
            d_image.contains(0.0) || c_image.contains(0.0) || b_image.contains(0.0);

        Ok(TwoParameterRaySummary {
            discriminant_image: d_image,
            b_image,
            c_image,
            decision_count,
            boundary_reachable,
            domain: [x_domain, y_domain],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_origin_prevents_sensitivity_region_from_becoming_identified_set() {
        let sensitivity = SetCertificate {
            representation: SetRepresentation::Sharp,
            origin: SetOrigin::SensitivityAnalysis,
        };
        assert!(!sensitivity.permits_identified_set_claim());
        assert!(!sensitivity.permits_stable_identified_decision());
        assert!(!sensitivity.permits_unstable_identified_decision());

        let observed_outer = SetCertificate {
            representation: SetRepresentation::OuterApproximation,
            origin: SetOrigin::ObservationalEquivalence,
        };
        assert!(observed_outer.permits_identified_set_claim());
        assert!(observed_outer.permits_stable_identified_decision());
        assert!(!observed_outer.permits_unstable_identified_decision());
    }

    #[test]
    fn approximation_polarity_is_fail_closed() {
        assert!(SetRepresentation::Sharp.permits_stable_decision_claim());
        assert!(SetRepresentation::Sharp.permits_unstable_decision_claim());
        assert!(SetRepresentation::OuterApproximation.permits_stable_decision_claim());
        assert!(!SetRepresentation::OuterApproximation.permits_unstable_decision_claim());
        assert!(SetRepresentation::InnerApproximation.permits_unstable_decision_claim());
        assert!(!SetRepresentation::InnerApproximation.permits_stable_decision_claim());
        assert!(!SetRepresentation::WitnessOnly.permits_stable_decision_claim());
        assert!(!SetRepresentation::WitnessOnly.permits_unstable_decision_claim());
    }

    #[test]
    fn explicit_pi_is_not_ni_and_point_estimation_remains_closed() {
        let set = ClosedInterval::new(-1.0, 1.0).unwrap();
        assert_ne!(set.lower, set.upper);
        assert_eq!(IdentificationLevel::PartiallyIdentified.as_str(), "PI");
        assert!(!IdentificationLevel::PartiallyIdentified.permits_point_target_claim());
        assert!(!IdentificationLevel::NotIdentified.permits_point_target_claim());
    }

    #[test]
    fn canonical_c_surface_has_exact_quadratic_image() {
        let domain = ClosedInterval::new(-1.0, 1.0).unwrap();
        let surface = QuadraticPolynomial {
            a: -1.0,
            b: 0.0,
            c: 0.25,
        };
        let image = surface.range_on(domain).unwrap();

        assert!((image.lower + 0.75).abs() <= TOL);
        assert!((image.upper - 0.25).abs() <= TOL);
    }

    #[test]
    fn pi_stable_decision_is_allowed_without_point_target_claim() {
        let domain = ClosedInterval::new(0.0, 1.0).unwrap();
        let surface = RaySurface {
            a: 1.0,
            b0: -2.0,
            b1: 0.0,
            c0: -0.5,
            c1: 0.1,
            c2: 0.0,
        };

        let summary = surface.classify(domain).unwrap();
        assert_eq!(summary.decision_status, DecisionStability::Stable);
        assert_eq!(summary.regularity_status, RegularityStatus::Regular);
        assert!(summary.permits_identified_decision_claim(SetCertificate {
            representation: SetRepresentation::Sharp,
            origin: SetOrigin::ObservationalEquivalence,
        }));
        assert!(
            !summary.permits_point_target_claim(IdentificationLevel::PartiallyIdentified)
        );
    }

    #[test]
    fn positive_root_count_is_stable_near_cancellation() {
        let surface = RaySurface {
            a: 1.0,
            b0: 1.0e12,
            b1: 0.0,
            c0: -1.0e3,
            c1: 0.0,
            c2: 0.0,
        };
        let (regime, count) = surface.point_structure(0.0);
        assert_eq!(regime, RayRegime::TwoRealBranches);
        assert_eq!(count, 1);
    }

    #[test]
    fn effect_radius_boundary_is_analytic() {
        let domain = ClosedInterval::new(0.0, 1.0).unwrap();
        let surface = RaySurface {
            a: 1.0,
            b0: 0.0,
            b1: 0.0,
            c0: 0.25,
            c1: -1.0,
            c2: 0.0,
        };

        let points = surface
            .effect_radius_boundary_points(domain, 0.5, 1.0, 1.0)
            .unwrap();
        assert_eq!(points.len(), 1);
        assert!((points[0] - 0.5).abs() <= TOL);
    }

    #[test]
    fn convex_polygon_preserves_joint_geometry() {
        let polygon = ConvexPolygon2d::new(vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
        ])
        .unwrap();
        let surface = QuadraticBox2d {
            a_xx: -1.0,
            b_yy: -1.0,
            c_xy: 0.0,
            d_x: 0.0,
            e_y: 0.0,
            f0: 0.25,
        };

        let image = polygon.range_of(&surface).unwrap();
        assert!((image.lower + 0.75).abs() <= TOL);
        assert!((image.upper - 0.25).abs() <= TOL);
    }

    #[test]
    fn polygon_discriminant_adapter_preserves_constraints() {
        let polygon = ConvexPolygon2d::new(vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
        ])
        .unwrap();
        let surface = RaySurface {
            a: 1.0,
            b0: 0.0,
            b1: 0.0,
            c0: -0.0625,
            c1: 0.0,
            c2: 0.0,
        };

        let image = surface
            .discriminant_image_over_polygon(&polygon, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.25)
            .unwrap();
        assert!((image.lower + 0.75).abs() <= TOL);
        assert!((image.upper - 0.25).abs() <= TOL);
    }

    #[test]
    fn two_parameter_quadratic_box_has_exact_canonical_image() {
        let x = ClosedInterval::new(-1.0, 1.0).unwrap();
        let y = ClosedInterval::new(-1.0, 1.0).unwrap();
        let surface = RaySurface {
            a: 1.0,
            b0: 0.0,
            b1: 0.0,
            c0: 0.0,
            c1: 0.0,
            c2: 0.0,
        };

        let summary = surface
            .classify_box_2d(x, y, 0.0, 0.0, -1.0, 0.0, -1.0)
            .unwrap();

        assert!((summary.discriminant_image.lower + 3.0).abs() <= TOL);
        assert!((summary.discriminant_image.upper - 1.0).abs() <= TOL);
        assert!(summary.boundary_reachable);
        assert_eq!(summary.decision_count, BoxDecisionCount::Unknown);
    }

    #[test]
    fn two_parameter_zero_root_requires_strictly_positive_b() {
        let domain = ClosedInterval::new(0.0, 1.0).unwrap();
        let surface = RaySurface {
            a: 1.0,
            b0: 0.0,
            b1: 0.0,
            c0: 0.5,
            c1: 0.0,
            c2: 0.0,
        };

        let summary = surface
            .classify_box_2d(domain, domain, 0.0, 0.0, 0.0, 0.0, 0.0)
            .unwrap();

        assert_eq!(summary.decision_count, BoxDecisionCount::Unknown);
    }

    #[test]
    fn two_parameter_stable_zero_root_box_is_certified() {
        let domain = ClosedInterval::new(0.0, 1.0).unwrap();
        let surface = RaySurface {
            a: 1.0,
            b0: 1.0,
            b1: 0.0,
            c0: 0.5,
            c1: 0.0,
            c2: 0.0,
        };

        let summary = surface
            .classify_box_2d(domain, domain, 0.0, 0.0, 0.0, 0.0, 0.0)
            .unwrap();

        assert!(summary.discriminant_image.lower > 0.0);
        assert_eq!(summary.decision_count, BoxDecisionCount::Zero);
        assert!(summary.decision_count.is_resolved());
    }

    #[test]
    fn two_parameter_stable_one_root_box_is_certified() {
        let domain = ClosedInterval::new(0.0, 1.0).unwrap();
        let surface = RaySurface {
            a: 1.0,
            b0: -2.0,
            b1: 0.0,
            c0: -0.5,
            c1: 0.0,
            c2: 0.0,
        };

        let summary = surface
            .classify_box_2d(domain, domain, 0.0, 0.0, 0.0, 0.0, 0.0)
            .unwrap();

        assert!(summary.discriminant_image.lower > 0.0);
        assert_eq!(summary.decision_count, BoxDecisionCount::One);
        assert!(!summary.boundary_reachable);
    }

    #[test]
    fn canonical_discriminant_crossing_is_unstable_and_nonregular() {
        let domain = ClosedInterval::new(0.0, 1.0).unwrap();
        let surface = RaySurface {
            a: 1.0,
            b0: 0.0,
            b1: 0.0,
            c0: 0.25,
            c1: -1.0,
            c2: 0.0,
        };

        let summary = surface.classify(domain).unwrap();
        assert_eq!(summary.decision_status, DecisionStability::Unstable);
        assert_eq!(summary.regularity_status, RegularityStatus::Boundary);
        assert!(!summary.permits_decision_claim());
        assert!(summary.discriminant_boundary_reachable);
        assert_eq!(summary.positive_root_counts, vec![0, 1]);
    }

    #[test]
    fn invalid_interval_is_rejected() {
        assert!(ClosedInterval::new(2.0, 1.0).is_err());
        assert!(ClosedInterval::new(f64::NAN, 1.0).is_err());
    }
}
