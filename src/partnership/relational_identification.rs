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
    /// A PI target may support a downstream decision, but only when the
    /// decision is invariant and the surface remains regular.
    pub fn permits_decision_claim(&self) -> bool {
        matches!(self.decision_status, DecisionStability::Stable)
            && matches!(self.regularity_status, RegularityStatus::Regular)
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(summary.permits_decision_claim());
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
