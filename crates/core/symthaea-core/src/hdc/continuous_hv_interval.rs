// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Conservative score intervals for canonical full-resolution ContinuousHV cosine.
//!
//! The module provides three complementary uncertainty bounds: a coordinate-box
//! relaxation of an L2 residual, a joint L2-ball bound for group centroids, and
//! caller-supplied component bounds for immutable group/page synopses. The
//! joint-ball method uses Cauchy–Schwarz and reverse-triangle geometry plus an
//! explicit binary32 reduction-error envelope. The box method propagates
//! outward intervals through the canonical group-of-eight and ordered-tail
//! reduction. HME can intersect independently validated enclosures; uncertainty
//! or malformed metadata must widen the result or disable pruning.
//!
//! Residual admission is separate: verify_l2_residual_certificate checks the
//! original payload against its reconstruction. A digest binding certificate
//! bytes is not proof of the residual inequality, group membership or eligibility.
//!
//! Numeric assumptions are documented on each public method. They include
//! round-to-nearest floating-point operations and gradual underflow; compiler,
//! target and runtime qualification is still required. Interval endpoints are
//! expanded outward, and the canonical score's reduction order remains part
//! of the versioned HME score contract.

use super::unified_hv::{ContinuousHV, ContinuousHVProfileError};

/// Closed interval enclosing a canonical full-resolution ContinuousHV score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContinuousHVScoreInterval {
    /// Conservative lower endpoint, in [-1, 1].
    pub lower: f32,
    /// Conservative upper endpoint, in [-1, 1].
    pub upper: f32,
}

impl ContinuousHVScoreInterval {
    /// The interval carrying no pruning information.
    pub const fn uninformative() -> Self {
        Self { lower: -1.0, upper: 1.0 }
    }

    /// Whether a canonical score is enclosed by this interval.
    pub fn contains(&self, score: f32) -> bool {
        score.is_finite() && self.lower <= score && score <= self.upper
    }

    /// Shape check only; this does not validate residual provenance or coverage.
    pub fn is_well_formed(&self) -> bool {
        self.lower.is_finite()
            && self.upper.is_finite()
            && -1.0 <= self.lower
            && self.lower <= self.upper
            && self.upper <= 1.0
    }
}


/// Conservative interval for the real-arithmetic cosine over an L2 ball.
///
/// This is a geometry research primitive, not a canonical f32 score interval.
/// It must not be passed to the exact HME selector or used to prune payloads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContinuousHVRealCosineInterval {
    /// Lower real-cosine endpoint in [-1, 1].
    pub lower: f64,
    /// Upper real-cosine endpoint in [-1, 1].
    pub upper: f64,
}

impl ContinuousHVRealCosineInterval {
    /// An interval with no geometric pruning information.
    pub const fn uninformative() -> Self {
        Self { lower: -1.0, upper: 1.0 }
    }

    /// Whether a finite real cosine is enclosed.
    pub fn contains(&self, cosine: f64) -> bool {
        cosine.is_finite() && self.lower <= cosine && cosine <= self.upper
    }

    /// Structural range check; this does not validate group membership.
    pub fn is_well_formed(&self) -> bool {
        self.lower.is_finite()
            && self.upper.is_finite()
            && -1.0 <= self.lower
            && self.lower <= self.upper
            && self.upper <= 1.0
    }

    /// Width of the interval in real cosine units.
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }
}


/// Candidate canonical-f32 score enclosure derived from spherical-cap geometry.
///
/// This is kept distinct from ContinuousHVScoreInterval until independent
/// differential and target qualification is complete. It is not accepted by
/// the HME global selector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContinuousHVSphericalCapF32ScoreInterval {
    /// Conservative lower canonical f32 score.
    pub lower: f32,
    /// Conservative upper canonical f32 score.
    pub upper: f32,
}

impl ContinuousHVSphericalCapF32ScoreInterval {
    /// An interval with no pruning information.
    pub const fn uninformative() -> Self {
        Self { lower: -1.0, upper: 1.0 }
    }

    /// Exact zero score for the canonical denominator-below-threshold branch.
    pub const fn zero_score() -> Self {
        Self { lower: 0.0, upper: 0.0 }
    }

    /// Whether a canonical finite score is enclosed.
    pub fn contains(&self, score: f32) -> bool {
        score.is_finite() && self.lower <= score && score <= self.upper
    }

    /// Structural score-range check.
    pub fn is_well_formed(&self) -> bool {
        self.lower.is_finite()
            && self.upper.is_finite()
            && -1.0 <= self.lower
            && self.lower <= self.upper
            && self.upper <= 1.0
    }
}

/// Admission errors for a bounded ContinuousHV score request.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ContinuousHVIntervalError {
    /// The query or reconstruction is outside the canonical score profile.
    #[error(transparent)]
    InvalidProfile(#[from] ContinuousHVProfileError),
    /// The supplied L2 residual upper bound must be finite and non-negative.
    #[error("L2 residual upper bound must be finite and non-negative")]
    InvalidResidualBound,
    /// Component-wise candidate lower/upper bounds are malformed.
    #[error("candidate component bounds are invalid at coordinate {index}")]
    InvalidCandidateComponentBounds {
        /// Coordinate whose lower/upper bounds failed validation.
        index: usize,
    },
}

/// Result of independently checking an original/reconstruction L2 residual.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualCertificateVerification {
    /// Outward upper bound on the squared L2 residual, accumulated in f64.
    pub residual_norm_squared_upper: f64,
    /// Exact f64 square of the supplied binary32 epsilon.
    pub certified_norm_squared: f64,
    /// Number of coordinates whose original and reconstruction differ.
    pub changed_coordinates: usize,
}

/// Errors from validating a candidate's residual certificate against its payload.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ResidualCertificateError {
    /// Original and reconstruction dimensions differ.
    #[error("residual certificate dimension mismatch: original={original_dim}, reconstruction={reconstruction_dim}")]
    DimensionMismatch {
        /// Original payload dimension.
        original_dim: usize,
        /// Reconstruction dimension.
        reconstruction_dim: usize,
    },
    /// Original or reconstruction components are not in the admitted finite [-1, 1] domain.
    #[error(transparent)]
    InvalidProfile(#[from] ContinuousHVProfileError),
    /// The supplied epsilon must be finite and non-negative.
    #[error("residual certificate epsilon must be finite and non-negative")]
    InvalidEpsilon,
    /// The measured residual cannot be enclosed by the supplied epsilon.
    #[error("residual certificate under-reports the payload error")]
    Underreported {
        /// Conservative squared residual upper bound.
        residual_norm_squared_upper: f64,
        /// Exact squared radius represented by epsilon.
        certified_norm_squared: f64,
    },
}

/// Advance a finite non-negative f64 to its next representable larger value.
#[inline]
fn f64_next_up_nonnegative(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    f64::from_bits(value.to_bits() + 1)
}

/// Independently verify `||original - reconstruction||_2 <= epsilon`.
///
/// The inputs are the actual stored binary32 payload and the reconstruction.
/// Each binary32 value is exactly representable in f64. The absolute
/// subtraction, square and running sum are rounded outward before comparison
/// with the exact f64 square of the binary32 epsilon (24-bit significand
/// product). The returned upper bound is conservative; equality at the
/// certificate radius is accepted only when the outward sum still fits.
///
/// This is an admission/build-time check, not a cryptographic proof receipt.
/// The caller must bind its result to the exact payload bytes, reconstruction,
/// transform/codebook and generation. A digest alone is not certificate proof.
pub fn verify_l2_residual_certificate(
    original: &ContinuousHV,
    reconstruction: &ContinuousHV,
    epsilon: f32,
) -> Result<ResidualCertificateVerification, ResidualCertificateError> {
    if original.values.len() != reconstruction.values.len() {
        return Err(ResidualCertificateError::DimensionMismatch {
            original_dim: original.values.len(),
            reconstruction_dim: reconstruction.values.len(),
        });
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(ResidualCertificateError::InvalidEpsilon);
    }
    original.validate_full_resolution_profile("original")?;
    reconstruction.validate_full_resolution_profile("reconstruction")?;

    let epsilon_f64 = epsilon as f64;
    // The product of two binary32 significands is exactly representable in
    // binary64 (at most 48 significant bits), including binary32 subnormals.
    let certified_norm_squared = epsilon_f64 * epsilon_f64;
    let mut residual_norm_squared_upper = 0.0f64;
    let mut changed_coordinates = 0usize;

    for (&actual, &approximate) in original.values.iter().zip(&reconstruction.values) {
        if actual == approximate {
            continue;
        }
        changed_coordinates += 1;
        // Expand every non-zero operation outward. The subtraction of widely
        // separated binary32 values need not be exact in binary64.
        let delta_rounded = ((actual as f64) - (approximate as f64)).abs();
        let delta_upper = f64_next_up_nonnegative(delta_rounded);
        let square_rounded = delta_upper * delta_upper;
        let square_upper = f64_next_up_nonnegative(square_rounded);
        residual_norm_squared_upper = f64_next_up_nonnegative(
            residual_norm_squared_upper + square_upper,
        );
    }

    if residual_norm_squared_upper > certified_norm_squared {
        return Err(ResidualCertificateError::Underreported {
            residual_norm_squared_upper,
            certified_norm_squared,
        });
    }

    Ok(ResidualCertificateVerification {
        residual_norm_squared_upper,
        certified_norm_squared,
        changed_coordinates,
    })
}

/// Outward-enclosed interval over binary32 values.
#[derive(Debug, Clone, Copy)]
struct F32Interval {
    lo: f32,
    hi: f32,
}

impl F32Interval {
    const fn point(value: f32) -> Self {
        Self { lo: value, hi: value }
    }

    fn add(self, rhs: Self) -> Option<Self> {
        let lo = self.lo + rhs.lo;
        let hi = self.hi + rhs.hi;
        if !lo.is_finite() || !hi.is_finite() {
            return None;
        }
        Some(Self { lo: next_down(lo), hi: next_up(hi) })
    }

    fn add_nonnegative(self, rhs: Self) -> Option<Self> {
        let mut sum = self.add(rhs)?;
        // Both operands represent sums of squares.
        sum.lo = sum.lo.max(0.0);
        Some(sum)
    }

    fn mul(self, rhs: Self) -> Option<Self> {
        let products = [
            self.lo * rhs.lo,
            self.lo * rhs.hi,
            self.hi * rhs.lo,
            self.hi * rhs.hi,
        ];
        if products.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let lo = products.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = products.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        Some(Self { lo: next_down(lo), hi: next_up(hi) })
    }

    fn square(self) -> Option<Self> {
        let min_abs = if self.lo <= 0.0 && self.hi >= 0.0 {
            0.0
        } else {
            self.lo.abs().min(self.hi.abs())
        };
        let max_abs = self.lo.abs().max(self.hi.abs());
        let lo = min_abs * min_abs;
        let hi = max_abs * max_abs;
        if !lo.is_finite() || !hi.is_finite() {
            return None;
        }
        Some(Self {
            lo: next_down(lo).max(0.0),
            hi: next_up(hi),
        })
    }

    fn sqrt_nonnegative(self) -> Option<Self> {
        if self.lo < 0.0 || !self.hi.is_finite() {
            return None;
        }
        let lo = self.lo.sqrt();
        let hi = self.hi.sqrt();
        if !lo.is_finite() || !hi.is_finite() {
            return None;
        }
        Some(Self {
            lo: next_down(lo).max(0.0),
            hi: next_up(hi),
        })
    }

    fn div_positive(self, denominator: Self) -> Option<Self> {
        if denominator.lo <= 0.0 {
            return None;
        }
        let quotients = [
            self.lo / denominator.lo,
            self.lo / denominator.hi,
            self.hi / denominator.lo,
            self.hi / denominator.hi,
        ];
        if quotients.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let lo = quotients.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = quotients.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        Some(Self { lo: next_down(lo), hi: next_up(hi) })
    }

    fn clamp_cosine(self) -> ContinuousHVScoreInterval {
        ContinuousHVScoreInterval {
            lower: self.lo.clamp(-1.0, 1.0),
            upper: self.hi.clamp(-1.0, 1.0),
        }
    }
}

// IEEE-754 adjacent-value operations, implemented without an MSRV dependency
// on f32::next_up/next_down (stabilized after the workspace's edition floor).
#[inline]
fn next_up(value: f32) -> f32 {
    if value.is_nan() || value == f32::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f32::from_bits(1);
    }
    let bits = value.to_bits();
    f32::from_bits(if value > 0.0 { bits + 1 } else { bits - 1 })
}

#[inline]
fn next_down(value: f32) -> f32 {
    if value.is_nan() || value == f32::NEG_INFINITY {
        return value;
    }
    if value == f32::INFINITY {
        return f32::MAX;
    }
    if value == 0.0 {
        return f32::from_bits(0x8000_0001);
    }
    let bits = value.to_bits();
    f32::from_bits(if value > 0.0 { bits - 1 } else { bits + 1 })
}

#[inline]
fn component_interval(center: f32, epsilon: f32) -> F32Interval {
    // For epsilon >= 2, the admitted component domain [-1, 1] is the tightest
    // profile-wide box and avoids arithmetic overflow.
    if epsilon >= 2.0 {
        return F32Interval { lo: -1.0, hi: 1.0 };
    }
    F32Interval {
        lo: next_down(center - epsilon).max(-1.0),
        hi: next_up(center + epsilon).min(1.0),
    }
}

/// Reproduce the canonical query norm-squared reduction in the same f32 order.
fn canonical_norm_squared(values: &[f32]) -> f32 {
    let mut norm_sq = 0.0f32;
    let mut chunks = values.chunks_exact(8);
    for chunk in chunks.by_ref() {
        norm_sq += chunk[0] * chunk[0]
            + chunk[1] * chunk[1]
            + chunk[2] * chunk[2]
            + chunk[3] * chunk[3]
            + chunk[4] * chunk[4]
            + chunk[5] * chunk[5]
            + chunk[6] * chunk[6]
            + chunk[7] * chunk[7];
    }
    for &value in chunks.remainder() {
        norm_sq += value * value;
    }
    norm_sq
}

/// Sum eight products in the canonical left-associated order.
fn product_group(query: &[f32], candidate: &[F32Interval; 8]) -> Option<F32Interval> {
    let mut sum = F32Interval::point(query[0]).mul(candidate[0])?;
    for index in 1..8 {
        sum = sum.add(F32Interval::point(query[index]).mul(candidate[index])?)?;
    }
    Some(sum)
}

/// Sum eight candidate squares in the canonical left-associated order.
fn square_group(candidate: &[F32Interval; 8]) -> Option<F32Interval> {
    let mut sum = candidate[0].square()?;
    for value in candidate.iter().skip(1) {
        sum = sum.add_nonnegative(value.square()?)?;
    }
    Some(sum)
}

/// Advance a finite f64 to the next representable value toward +infinity.
#[inline]
fn f64_next_up(value: f64) -> f64 {
    if value.is_nan() || value == f64::INFINITY {
        return value;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits + 1 } else { bits - 1 })
}

/// Advance a finite f64 to the next representable value toward -infinity.
#[inline]
fn f64_next_down(value: f64) -> f64 {
    if value.is_nan() || value == f64::NEG_INFINITY {
        return value;
    }
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = value.to_bits();
    f64::from_bits(if value > 0.0 { bits - 1 } else { bits + 1 })
}

fn f64_sum_squares_bounds(values: &[f32]) -> (f64, f64) {
    let mut lower = 0.0f64;
    let mut upper = 0.0f64;
    for &value in values {
        let value = value as f64;
        // Every f32 value is exact in f64, and its square has at most 48
        // significant bits. Outward sum steps cover accumulated f64 rounding.
        let square = value * value;
        lower = f64_next_down(lower + square).max(0.0);
        upper = f64_next_up(upper + square);
    }
    (lower, upper)
}

fn f64_dot_bounds(left: &[f32], right: &[f32]) -> (f64, f64) {
    let mut lower = 0.0f64;
    let mut upper = 0.0f64;
    for (&a, &b) in left.iter().zip(right) {
        // A product of two binary32 significands is exactly representable in
        // binary64. The outward sum bounds the real dot product.
        let product = (a as f64) * (b as f64);
        lower = f64_next_down(lower + product);
        upper = f64_next_up(upper + product);
    }
    (lower, upper)
}

#[inline]
fn f32_lower_from_f64(value: f64) -> Option<f32> {
    if !value.is_finite() || value < -(f32::MAX as f64) || value > f32::MAX as f64 {
        return None;
    }
    let rounded = value as f32;
    if !rounded.is_finite() {
        return None;
    }
    Some(if (rounded as f64) > value {
        next_down(rounded)
    } else {
        rounded
    })
}

#[inline]
fn f32_upper_from_f64(value: f64) -> Option<f32> {
    if !value.is_finite() || value < -(f32::MAX as f64) || value > f32::MAX as f64 {
        return None;
    }
    let rounded = value as f32;
    if !rounded.is_finite() {
        return None;
    }
    Some(if (rounded as f64) < value {
        next_up(rounded)
    } else {
        rounded
    })
}


/// Outward interval for f64 geometric calculations.
#[derive(Debug, Clone, Copy)]
struct F64Interval {
    lo: f64,
    hi: f64,
}

impl F64Interval {
    fn mul(self, rhs: Self) -> Option<Self> {
        let products = [
            self.lo * rhs.lo,
            self.lo * rhs.hi,
            self.hi * rhs.lo,
            self.hi * rhs.hi,
        ];
        if products.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let lo = products.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some(Self { lo: f64_next_down(lo), hi: f64_next_up(hi) })
    }

    fn div_positive(self, rhs: Self) -> Option<Self> {
        if rhs.lo <= 0.0 || !rhs.hi.is_finite() {
            return None;
        }
        let quotients = [
            self.lo / rhs.lo,
            self.lo / rhs.hi,
            self.hi / rhs.lo,
            self.hi / rhs.hi,
        ];
        if quotients.iter().any(|value| !value.is_finite()) {
            return None;
        }
        let lo = quotients.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = quotients.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some(Self { lo: f64_next_down(lo), hi: f64_next_up(hi) })
    }

    fn square(self) -> Option<Self> {
        let min_abs = if self.lo <= 0.0 && self.hi >= 0.0 {
            0.0
        } else {
            self.lo.abs().min(self.hi.abs())
        };
        let max_abs = self.lo.abs().max(self.hi.abs());
        let lo = min_abs * min_abs;
        let hi = max_abs * max_abs;
        if !lo.is_finite() || !hi.is_finite() {
            return None;
        }
        Some(Self {
            lo: f64_next_down(lo).max(0.0),
            hi: f64_next_up(hi),
        })
    }

    fn sqrt_nonnegative(self) -> Option<Self> {
        if self.lo < 0.0 || !self.hi.is_finite() {
            return None;
        }
        let lo = self.lo.sqrt();
        let hi = self.hi.sqrt();
        if !lo.is_finite() || !hi.is_finite() {
            return None;
        }
        Some(Self {
            lo: f64_next_down(lo).max(0.0),
            hi: f64_next_up(hi),
        })
    }

    fn one_minus_square(self) -> Option<Self> {
        let squared = self.square()?;
        let lo = f64_next_down(1.0 - squared.hi).max(0.0);
        let hi = f64_next_up(1.0 - squared.lo).min(1.0);
        (lo <= hi).then_some(Self { lo, hi })
    }
}

impl ContinuousHV {

    /// Bound the real-arithmetic cosine of every nonzero vector in the
    /// L2 ball around center using its induced spherical cone.
    ///
    /// If r < ||center||, every candidate direction lies within a cone
    /// around the center direction with half-angle asin(r / ||center||).
    /// This evaluates the equivalent algebraic spherical-cap formula with
    /// outward f64 intervals; it avoids asin/acos and falls back to [-1, 1]
    /// whenever the center ball may contain the origin or arithmetic cannot
    /// be bounded. f64 square root is correctly rounded by the Rust primitive
    /// contract; the implementation expands its endpoints by one adjacent value.
    ///
    /// Arithmetic assumes IEEE round-to-nearest for f64 addition, multiplication
    /// and division, and gradual underflow. Do not compile this module with
    /// fast-math or reassociation transformations.
    ///
    /// IMPORTANT: this bounds real cosine only. It does not enclose the
    /// canonical f32 score from similarity_full_resolution and is not
    /// prune-authorizing. A future canonical-score method must add and qualify
    /// a separate f32 reduction/denominator error envelope.
    pub fn real_cosine_interval_for_l2_ball_center(
        &self,
        center: &Self,
        residual_l2_upper: f32,
    ) -> Result<ContinuousHVRealCosineInterval, ContinuousHVIntervalError> {
        self.validate_full_resolution_pair(center)?;
        if !residual_l2_upper.is_finite() || residual_l2_upper < 0.0 {
            return Err(ContinuousHVIntervalError::InvalidResidualBound);
        }

        let (query_sq_lower, query_sq_upper) = f64_sum_squares_bounds(&self.values);
        let (center_sq_lower, center_sq_upper) = f64_sum_squares_bounds(&center.values);
        if query_sq_lower <= 0.0 || center_sq_lower <= 0.0 {
            return Ok(ContinuousHVRealCosineInterval::uninformative());
        }

        let query_sq = F64Interval {
            lo: query_sq_lower.max(0.0),
            hi: query_sq_upper.max(0.0),
        };
        let center_sq = F64Interval {
            lo: center_sq_lower.max(0.0),
            hi: center_sq_upper.max(0.0),
        };
        let query_norm = match query_sq.sqrt_nonnegative() {
            Some(value) if value.lo > 0.0 => value,
            _ => return Ok(ContinuousHVRealCosineInterval::uninformative()),
        };
        let center_norm = match center_sq.sqrt_nonnegative() {
            Some(value) if value.lo > 0.0 => value,
            _ => return Ok(ContinuousHVRealCosineInterval::uninformative()),
        };

        let radius = residual_l2_upper as f64;
        // Require strict separation from the origin under the lower norm
        // endpoint; otherwise every angular direction remains possible.
        if radius >= center_norm.lo {
            return Ok(ContinuousHVRealCosineInterval::uninformative());
        }

        let dot_bounds = f64_dot_bounds(&self.values, &center.values);
        let dot = F64Interval { lo: dot_bounds.0, hi: dot_bounds.1 };
        let norm_product = match query_norm.mul(center_norm) {
            Some(value) if value.lo > 0.0 => value,
            _ => return Ok(ContinuousHVRealCosineInterval::uninformative()),
        };
        let mut d = match dot.div_positive(norm_product) {
            Some(value) => value,
            None => return Ok(ContinuousHVRealCosineInterval::uninformative()),
        };
        // The exact normalized dot lies in [-1, 1]; intersect numeric
        // overhang with this analytic invariant.
        d.lo = d.lo.max(-1.0);
        d.hi = d.hi.min(1.0);
        if d.lo > d.hi {
            return Ok(ContinuousHVRealCosineInterval::uninformative());
        }

        // a = sin(alpha) = r / ||center||. Its interval is non-negative.
        let a = F64Interval {
            // The lower quotient must be rounded downward as well: ordinary
            // f64 division is nearest-rounded and can otherwise overstate the
            // smallest admissible cone half-angle.
            lo: f64_next_down(radius / center_norm.hi).max(0.0),
            hi: f64_next_up(radius / center_norm.lo),
        };
        if !a.lo.is_finite() || !a.hi.is_finite() || a.hi >= 1.0 {
            return Ok(ContinuousHVRealCosineInterval::uninformative());
        }

        // cos(alpha) = sqrt(1 - a^2), and sin(theta) = sqrt(1 - d^2).
        let cos_alpha = match a.one_minus_square().and_then(F64Interval::sqrt_nonnegative) {
            Some(value) => value,
            None => return Ok(ContinuousHVRealCosineInterval::uninformative()),
        };
        let sin_theta = match d.one_minus_square().and_then(F64Interval::sqrt_nonnegative) {
            Some(value) => value,
            None => return Ok(ContinuousHVRealCosineInterval::uninformative()),
        };
        let center_term = match d.mul(cos_alpha) {
            Some(value) => value,
            None => return Ok(ContinuousHVRealCosineInterval::uninformative()),
        };
        let spread = match sin_theta.mul(a) {
            Some(value) => value,
            None => return Ok(ContinuousHVRealCosineInterval::uninformative()),
        };

        // The upper endpoint saturates at +1 when the query direction may
        // lie inside the cone. The lower endpoint similarly saturates at -1
        // for the antipodal cone. Ambiguity widens rather than guesses.
        let upper_may_saturate = d.hi >= cos_alpha.lo;
        let lower_may_saturate = d.lo <= -cos_alpha.lo;
        let lower = if lower_may_saturate {
            -1.0
        } else {
            f64_next_down(center_term.lo - spread.hi).max(-1.0)
        };
        let upper = if upper_may_saturate {
            1.0
        } else {
            f64_next_up(center_term.hi + spread.hi).min(1.0)
        };

        let interval = ContinuousHVRealCosineInterval { lower, upper };
        if interval.is_well_formed() {
            Ok(interval)
        } else {
            Ok(ContinuousHVRealCosineInterval::uninformative())
        }
    }


    /// Candidate enclosure of the canonical f32 cosine using a spherical-cap
    /// geometric interval plus a separate reduction/denominator error budget.
    ///
    /// This is a research-stage result type and is not yet accepted by the
    /// global HME exact selector. The derivation bounds the canonical dot and
    /// norm reductions with gamma/eta envelopes, then includes the final f32
    /// norm-product, square-root and quotient roundings. It assumes the
    /// original members are admitted finite [-1,1] vectors and are all within
    /// the supplied L2 radius of center. The method does not prove membership.
    ///
    /// Any ambiguous denominator, nonpositive norm lower bound, overflow or
    /// insufficient numeric separation returns [-1,1]. A zero radius returns
    /// the canonical point score. This must pass the full adversarial
    /// differential corpus and target/compiler inspection before selector use.
    pub fn similarity_interval_full_resolution_for_l2_ball_center_spherical_cap(
        &self,
        center: &Self,
        residual_l2_upper: f32,
    ) -> Result<ContinuousHVSphericalCapF32ScoreInterval, ContinuousHVIntervalError> {
        self.validate_full_resolution_pair(center)?;
        if !residual_l2_upper.is_finite() || residual_l2_upper < 0.0 {
            return Err(ContinuousHVIntervalError::InvalidResidualBound);
        }
        if self.values.iter().all(|value| *value == 0.0) {
            return Ok(ContinuousHVSphericalCapF32ScoreInterval::zero_score());
        }
        if residual_l2_upper == 0.0 {
            let score = self.similarity_full_resolution(center);
            return Ok(ContinuousHVSphericalCapF32ScoreInterval {
                lower: score,
                upper: score,
            });
        }

        let (query_sq_lower, query_sq_upper) = f64_sum_squares_bounds(&self.values);
        let (center_sq_lower, center_sq_upper) = f64_sum_squares_bounds(&center.values);
        if query_sq_lower <= 0.0 || center_sq_lower <= 0.0 {
            return Ok(ContinuousHVSphericalCapF32ScoreInterval::uninformative());
        }

        let query_sq_interval = F64Interval {
            lo: query_sq_lower,
            hi: query_sq_upper,
        };
        let center_sq_interval = F64Interval {
            lo: center_sq_lower,
            hi: center_sq_upper,
        };
        let query_norm = match query_sq_interval.sqrt_nonnegative() {
            Some(value) if value.lo > 0.0 => value,
            _ => return Ok(ContinuousHVSphericalCapF32ScoreInterval::uninformative()),
        };
        let center_norm = match center_sq_interval.sqrt_nonnegative() {
            Some(value) if value.lo > 0.0 => value,
            _ => return Ok(ContinuousHVSphericalCapF32ScoreInterval::uninformative()),
        };

        let radius = residual_l2_upper as f64;
        if radius >= center_norm.lo {
            return Ok(ContinuousHVSphericalCapF32ScoreInterval::uninformative());
        }
        let candidate_norm_lower =
            f64_next_down(center_norm.lo - radius).max(0.0);
        let candidate_norm_upper =
            f64_next_up(center_norm.hi + radius);
        if candidate_norm_lower <= 0.0 || !candidate_norm_upper.is_finite() {
            return Ok(ContinuousHVSphericalCapF32ScoreInterval::uninformative());
        }

        let real_cosine = self
            .real_cosine_interval_for_l2_ball_center(center, residual_l2_upper)?;
        if !real_cosine.is_well_formed()
            || (real_cosine.lower == -1.0 && real_cosine.upper == 1.0)
        {
            return Ok(ContinuousHVSphericalCapF32ScoreInterval::uninformative());
        }

        // Standard dot/norm reduction model: gamma bounds accumulated relative
        // error; eta bounds the sum of worst-case subnormal absolute errors.
        let dimension = self.values.len();
        let operation_count = (dimension as f64) * 2.0;
        let unit_roundoff = 2.0f64.powi(-24);
        let nu = operation_count * unit_roundoff;
        if !nu.is_finite() || nu >= 1.0 {
            return Ok(ContinuousHVSphericalCapF32ScoreInterval::uninformative());
        }
        let gamma = f64_next_up(nu / (1.0 - nu));
        let min_subnormal = f32::from_bits(1) as f64;
        let eta = f64_next_up(
            (operation_count * min_subnormal) / (1.0 - nu),
        );

        let candidate_norm_sq_lower = f64_next_down(
            candidate_norm_lower * candidate_norm_lower,
        )
        .max(0.0);
        if candidate_norm_sq_lower <= 0.0 {
            return Ok(ContinuousHVSphericalCapF32ScoreInterval::uninformative());
        }
        let query_eta_ratio = f64_next_up(eta / query_sq_lower);
        let candidate_eta_ratio = f64_next_up(eta / candidate_norm_sq_lower);
        let query_relative_error = f64_next_up(gamma + query_eta_ratio);
        let candidate_relative_error = f64_next_up(gamma + candidate_eta_ratio);
        if !query_relative_error.is_finite()
            || !candidate_relative_error.is_finite()
            || query_relative_error >= 1.0
            || candidate_relative_error >= 1.0
        {
            return Ok(ContinuousHVSphericalCapF32ScoreInterval::uninformative());
        }

        let one_minus_u = f64_next_down(1.0 - unit_roundoff).max(0.0);
        let one_plus_u = f64_next_up(1.0 + unit_roundoff);
        let query_norm_factor = F64Interval {
            lo: f64_next_down(1.0 - query_relative_error).max(0.0),
            hi: f64_next_up(1.0 + query_relative_error),
        };
        let candidate_norm_factor = F64Interval {
            lo: f64_next_down(1.0 - candidate_relative_error).max(0.0),
            hi: f64_next_up(1.0 + candidate_relative_error),
        };
        let product_rounding_factor = F64Interval {
            lo: one_minus_u,
            hi: one_plus_u,
        };
        let sqrt_rounding_factor = F64Interval {
            lo: one_minus_u,
            hi: one_plus_u,
        };
        let denominator_factor = match query_norm_factor
            .mul(candidate_norm_factor)
            .and_then(|value| value.mul(product_rounding_factor))
            .and_then(F64Interval::sqrt_nonnegative)
            .and_then(|value| value.mul(sqrt_rounding_factor))
        {
            Some(value) if value.lo > 0.0 => value,
            _ => return Ok(ContinuousHVSphericalCapF32ScoreInterval::uninformative()),
        };

        // The canonical near-zero branch is evaluated on f32 sqrt(normA*normB).
        // Bound its possible denominator before using a quotient interval.
        let norm_product_lower = f64_next_down(query_norm.lo * candidate_norm_lower);
        let norm_product_upper = f64_next_up(query_norm.hi * candidate_norm_upper);
        let denominator_lower =
            f64_next_down(norm_product_lower * denominator_factor.lo);
        let denominator_upper =
            f64_next_up(norm_product_upper * denominator_factor.hi);
        if !denominator_lower.is_finite() || !denominator_upper.is_finite() {
            return Ok(ContinuousHVSphericalCapF32ScoreInterval::uninformative());
        }
        let canonical_threshold = 1.0e-10f32 as f64;
        if denominator_upper < canonical_threshold {
            return Ok(ContinuousHVSphericalCapF32ScoreInterval::zero_score());
        }
        if denominator_lower < canonical_threshold {
            return Ok(ContinuousHVSphericalCapF32ScoreInterval::uninformative());
        }

        // |dot_rounding_error| / (||q|| ||x||) <= gamma + eta / (q_norm*x_norm).
        let normalized_dot_eta_ratio = f64_next_up(eta / norm_product_lower);
        let normalized_dot_error = f64_next_up(gamma + normalized_dot_eta_ratio);
        if !normalized_dot_error.is_finite() {
            return Ok(ContinuousHVSphericalCapF32ScoreInterval::uninformative());
        }
        let normalized_dot = F64Interval {
            lo: f64_next_down(real_cosine.lower - normalized_dot_error),
            hi: f64_next_up(real_cosine.upper + normalized_dot_error),
        };
        let quotient = match normalized_dot.div_positive(denominator_factor) {
            Some(value) => value,
            None => return Ok(ContinuousHVSphericalCapF32ScoreInterval::uninformative()),
        };
        let quotient_rounding_factor = F64Interval {
            lo: one_minus_u,
            hi: one_plus_u,
        };
        let rounded_quotient = match quotient.mul(quotient_rounding_factor) {
            Some(value) => value,
            None => return Ok(ContinuousHVSphericalCapF32ScoreInterval::uninformative()),
        };

        // The final division may underflow to a subnormal; a full minimum
        // subnormal on either side safely covers its absolute rounding error.
        let lower_f64 = f64_next_down(rounded_quotient.lo - min_subnormal)
            .max(-1.0);
        let upper_f64 = f64_next_up(rounded_quotient.hi + min_subnormal)
            .min(1.0);
        let lower = match f32_lower_from_f64(lower_f64) {
            Some(value) => value.clamp(-1.0, 1.0),
            None => return Ok(ContinuousHVSphericalCapF32ScoreInterval::uninformative()),
        };
        let upper = match f32_upper_from_f64(upper_f64) {
            Some(value) => value.clamp(-1.0, 1.0),
            None => return Ok(ContinuousHVSphericalCapF32ScoreInterval::uninformative()),
        };
        let interval = ContinuousHVSphericalCapF32ScoreInterval { lower, upper };
        if interval.is_well_formed() {
            Ok(interval)
        } else {
            Ok(ContinuousHVSphericalCapF32ScoreInterval::uninformative())
        }
    }

    /// Enclose canonical cosine for every candidate x in the L2 ball
    /// ||x - center||_2 <= residual_l2_upper.
    ///
    /// This bound retains the joint L2 constraint and is intended for group
    /// centroids/radii. It uses Cauchy-Schwarz for the dot product and reverse
    /// triangle inequality for candidate norms, then adds a conservative
    /// binary32 reduction-error envelope for the #10918 scorer.
    ///
    /// Preconditions: query and center are admitted finite [-1,1] ContinuousHV
    /// values; the caller has independently verified that every original group
    /// member is within residual_l2_upper of center. This method does not prove
    /// group membership or radius validity. The error envelope assumes IEEE-754
    /// binary32 round-to-nearest, gradual underflow, and no fast-math,
    /// reassociation or fused contraction of the canonical multiply/add chain.
    ///
    /// If the operation-count error bound is too large, the denominator can
    /// cross the canonical near-zero threshold, or any interval becomes
    /// unbounded, this returns [-1,1] and authorizes no pruning.
    pub fn similarity_interval_full_resolution_for_l2_ball_center(
        &self,
        center: &Self,
        residual_l2_upper: f32,
    ) -> Result<ContinuousHVScoreInterval, ContinuousHVIntervalError> {
        self.validate_full_resolution_pair(center)?;
        if !residual_l2_upper.is_finite() || residual_l2_upper < 0.0 {
            return Err(ContinuousHVIntervalError::InvalidResidualBound);
        }
        if residual_l2_upper == 0.0 {
            let score = self.similarity_full_resolution(center);
            return Ok(ContinuousHVScoreInterval { lower: score, upper: score });
        }

        let query_norm_sq_f32 = canonical_norm_squared(&self.values);
        if !query_norm_sq_f32.is_finite() {
            return Ok(ContinuousHVScoreInterval::uninformative());
        }
        if query_norm_sq_f32 == 0.0 {
            return Ok(ContinuousHVScoreInterval { lower: 0.0, upper: 0.0 });
        }

        let dimension = self.values.len();
        let operation_count = (dimension as f64) * 2.0;
        let unit_roundoff = 2.0f64.powi(-24);
        let nu = operation_count * unit_roundoff;
        if !nu.is_finite() || nu >= 1.0 {
            return Ok(ContinuousHVScoreInterval::uninformative());
        }
        let gamma = f64_next_up(nu / (1.0 - nu));
        let min_subnormal = f32::from_bits(1) as f64;
        // The additive term covers the worst absolute subnormal rounding per
        // multiply/add; division by (1-nu) bounds later amplification.
        let eta = f64_next_up(
            (operation_count * min_subnormal) / (1.0 - nu),
        );

        let (_query_norm_sq_lower, query_norm_sq_upper) =
            f64_sum_squares_bounds(&self.values);
        let (center_norm_sq_lower, center_norm_sq_upper) =
            f64_sum_squares_bounds(&center.values);
        let (center_dot_lower, center_dot_upper) =
            f64_dot_bounds(&self.values, &center.values);

        let query_norm_upper =
            f64_next_up(query_norm_sq_upper.max(0.0).sqrt());
        let center_norm_lower =
            f64_next_down(query_norm_sq_lower.max(0.0).sqrt()).max(0.0);
        let center_norm_upper =
            f64_next_up(center_norm_sq_upper.max(0.0).sqrt());
        let radius = residual_l2_upper as f64;

        let candidate_norm_lower =
            f64_next_down(center_norm_lower - radius).max(0.0);
        let candidate_norm_upper =
            f64_next_up(center_norm_upper + radius);

        // Real dot-product interval from the centroid plus a Cauchy-Schwarz
        // radius, expanded again to enclose the canonical f32 reduction.
        let dot_slack = f64_next_up(query_norm_upper * radius);
        let geometric_dot_lower = f64_next_down(center_dot_lower - dot_slack);
        let geometric_dot_upper = f64_next_up(center_dot_upper + dot_slack);
        let absolute_dot_sum_upper =
            f64_next_up(query_norm_upper * candidate_norm_upper);
        let dot_rounding_error =
            f64_next_up(f64_next_up(gamma * absolute_dot_sum_upper) + eta);
        let dot_lower = f64_next_down(geometric_dot_lower - dot_rounding_error);
        let dot_upper = f64_next_up(geometric_dot_upper + dot_rounding_error);

        // The canonical positive norm-squared reduction is enclosed by the
        // standard gamma_n relative-error model plus an absolute subnormal
        // allowance. Reverse triangle bounds the real candidate norm.
        let norm_lower_real_sq =
            f64_next_down(candidate_norm_lower * candidate_norm_lower).max(0.0);
        let norm_upper_real_sq =
            f64_next_up(candidate_norm_upper * candidate_norm_upper);
        let norm_lower_scaled =
            f64_next_down((1.0 - gamma).max(0.0) * norm_lower_real_sq);
        let norm_upper_scaled =
            f64_next_up((1.0 + gamma) * norm_upper_real_sq);
        let candidate_norm_sq_lower =
            f64_next_down(norm_lower_scaled - eta).max(0.0);
        let candidate_norm_sq_upper =
            f64_next_up(norm_upper_scaled + eta);

        let dot_interval = match (
            f32_lower_from_f64(dot_lower),
            f32_upper_from_f64(dot_upper),
        ) {
            (Some(lower), Some(upper)) if lower <= upper => F32Interval {
                lo: lower,
                hi: upper,
            },
            _ => return Ok(ContinuousHVScoreInterval::uninformative()),
        };
        let candidate_norm_sq_interval = match (
            f32_lower_from_f64(candidate_norm_sq_lower),
            f32_upper_from_f64(candidate_norm_sq_upper),
        ) {
            (Some(lower), Some(upper)) if lower <= upper => F32Interval {
                lo: lower.max(0.0),
                hi: upper.max(0.0),
            },
            _ => return Ok(ContinuousHVScoreInterval::uninformative()),
        };

        let norm_product = match F32Interval::point(query_norm_sq_f32)
            .mul(candidate_norm_sq_interval)
        {
            Some(value) => F32Interval {
                lo: value.lo.max(0.0),
                hi: value.hi.max(0.0),
            },
            None => return Ok(ContinuousHVScoreInterval::uninformative()),
        };
        let denominator = match norm_product.sqrt_nonnegative() {
            Some(value) => value,
            None => return Ok(ContinuousHVScoreInterval::uninformative()),
        };

        const CANONICAL_NEAR_ZERO_DENOMINATOR: f32 = 1e-10;
        if denominator.hi < CANONICAL_NEAR_ZERO_DENOMINATOR {
            return Ok(ContinuousHVScoreInterval { lower: 0.0, upper: 0.0 });
        }
        if denominator.lo < CANONICAL_NEAR_ZERO_DENOMINATOR {
            return Ok(ContinuousHVScoreInterval::uninformative());
        }

        let quotient = match dot_interval.div_positive(denominator) {
            Some(value) => value,
            None => return Ok(ContinuousHVScoreInterval::uninformative()),
        };
        let interval = quotient.clamp_cosine();
        if interval.is_well_formed() {
            Ok(interval)
        } else {
            Ok(ContinuousHVScoreInterval::uninformative())
        }
    }

    /// Enclose the canonical full-resolution score for every candidate inside
    /// a component-wise box. This is intended for immutable group/page
    /// synopses: lower[j] and upper[j] must conservatively enclose coordinate j
    /// for every original payload in the group.
    ///
    /// The extrema are assumed to have been computed/verified against every
    /// member at segment-build time and bound to the exact membership and
    /// generation. This method validates only the numeric shape of the box; it
    /// cannot prove the extrema or corpus membership. All arithmetic after
    /// admission follows the canonical group-eight/tail f32 score order and
    /// widens to [-1,1] when numeric uncertainty prevents safe tightening.
    pub fn similarity_interval_full_resolution_for_component_bounds(
        &self,
        candidate_lower: &[f32],
        candidate_upper: &[f32],
    ) -> Result<ContinuousHVScoreInterval, ContinuousHVIntervalError> {
        if candidate_lower.len() != candidate_upper.len() {
            return Err(ContinuousHVIntervalError::InvalidProfile(
                ContinuousHVProfileError::DimensionMismatch {
                    query_dim: candidate_lower.len(),
                    candidate_dim: candidate_upper.len(),
                },
            ));
        }
        if self.values.len() != candidate_lower.len() {
            return Err(ContinuousHVIntervalError::InvalidProfile(
                ContinuousHVProfileError::DimensionMismatch {
                    query_dim: self.values.len(),
                    candidate_dim: candidate_lower.len(),
                },
            ));
        }
        self.validate_full_resolution_profile("query")?;
        for (index, (&lower, &upper)) in candidate_lower
            .iter()
            .zip(candidate_upper)
            .enumerate()
        {
            if !lower.is_finite()
                || !upper.is_finite()
                || !(-1.0..=1.0).contains(&lower)
                || !(-1.0..=1.0).contains(&upper)
                || lower > upper
            {
                return Err(
                    ContinuousHVIntervalError::InvalidCandidateComponentBounds { index },
                );
            }
        }

        let query_norm_sq = canonical_norm_squared(&self.values);
        if !query_norm_sq.is_finite() {
            return Ok(ContinuousHVScoreInterval::uninformative());
        }
        if query_norm_sq == 0.0 {
            return Ok(ContinuousHVScoreInterval { lower: 0.0, upper: 0.0 });
        }

        let mut dot = F32Interval::point(0.0);
        let mut candidate_norm_sq = F32Interval::point(0.0);
        let mut query_chunks = self.values.chunks_exact(8);
        let mut lower_chunks = candidate_lower.chunks_exact(8);
        let mut upper_chunks = candidate_upper.chunks_exact(8);

        for ((query_chunk, lower_chunk), upper_chunk) in query_chunks
            .by_ref()
            .zip(lower_chunks.by_ref())
            .zip(upper_chunks.by_ref())
        {
            let candidate_chunk: [F32Interval; 8] = std::array::from_fn(|index| {
                F32Interval {
                    lo: lower_chunk[index],
                    hi: upper_chunk[index],
                }
            });
            let dot_group = match product_group(query_chunk, &candidate_chunk) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            let norm_group = match square_group(&candidate_chunk) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            dot = match dot.add(dot_group) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            candidate_norm_sq = match candidate_norm_sq.add_nonnegative(norm_group) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
        }

        for ((&query_value, &lower), &upper) in query_chunks
            .remainder()
            .iter()
            .zip(lower_chunks.remainder())
            .zip(upper_chunks.remainder())
        {
            let candidate_value = F32Interval { lo: lower, hi: upper };
            let product = match F32Interval::point(query_value).mul(candidate_value) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            let square = match candidate_value.square() {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            dot = match dot.add(product) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            candidate_norm_sq = match candidate_norm_sq.add_nonnegative(square) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
        }

        let norm_product = match F32Interval::point(query_norm_sq).mul(candidate_norm_sq) {
            Some(value) => F32Interval {
                lo: value.lo.max(0.0),
                hi: value.hi.max(0.0),
            },
            None => return Ok(ContinuousHVScoreInterval::uninformative()),
        };
        let denominator = match norm_product.sqrt_nonnegative() {
            Some(value) => value,
            None => return Ok(ContinuousHVScoreInterval::uninformative()),
        };

        const CANONICAL_NEAR_ZERO_DENOMINATOR: f32 = 1e-10;
        if denominator.hi < CANONICAL_NEAR_ZERO_DENOMINATOR {
            return Ok(ContinuousHVScoreInterval { lower: 0.0, upper: 0.0 });
        }
        if denominator.lo < CANONICAL_NEAR_ZERO_DENOMINATOR {
            return Ok(ContinuousHVScoreInterval::uninformative());
        }

        let quotient = match dot.div_positive(denominator) {
            Some(value) => value,
            None => return Ok(ContinuousHVScoreInterval::uninformative()),
        };
        let interval = quotient.clamp_cosine();
        if interval.is_well_formed() {
            Ok(interval)
        } else {
            Ok(ContinuousHVScoreInterval::uninformative())
        }
    }

    /// Enclose canonical full-resolution cosine over component bounds
    /// supplied by a deterministic, allocation-free indexed callback.
    ///
    /// This supports compact outward-quantized synopsis formats. The callback
    /// is invoked during validation and again during accumulation, so it must
    /// return the same pair for each index on every invocation. Returned lower
    /// and upper values must enclose the true component range.
    ///
    /// The implementation uses the same group-of-eight and ordered-tail
    /// reduction contract as the slice-based component-bound kernel. It does
    /// not allocate a temporary decoded bounds vector.
    pub fn similarity_interval_full_resolution_for_component_bounds_with<F>(
        &self,
        mut component_bounds: F,
    ) -> Result<ContinuousHVScoreInterval, ContinuousHVIntervalError>
    where
        F: FnMut(usize) -> (f32, f32),
    {
        self.validate_full_resolution_profile("query")?;
        for index in 0..self.values.len() {
            let (lower, upper) = component_bounds(index);
            if !lower.is_finite()
                || !upper.is_finite()
                || !(-1.0..=1.0).contains(&lower)
                || !(-1.0..=1.0).contains(&upper)
                || lower > upper
            {
                return Err(
                    ContinuousHVIntervalError::InvalidCandidateComponentBounds { index },
                );
            }
        }

        let query_norm_sq = canonical_norm_squared(&self.values);
        if !query_norm_sq.is_finite() {
            return Ok(ContinuousHVScoreInterval::uninformative());
        }
        if query_norm_sq == 0.0 {
            return Ok(ContinuousHVScoreInterval { lower: 0.0, upper: 0.0 });
        }

        let mut dot = F32Interval::point(0.0);
        let mut candidate_norm_sq = F32Interval::point(0.0);
        let mut query_chunks = self.values.chunks_exact(8);
        let mut offset = 0usize;
        for query_chunk in query_chunks.by_ref() {
            let candidate_chunk: [F32Interval; 8] = std::array::from_fn(|index| {
                let (lower, upper) = component_bounds(offset + index);
                F32Interval { lo: lower, hi: upper }
            });
            let dot_group = match product_group(query_chunk, &candidate_chunk) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            let norm_group = match square_group(&candidate_chunk) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            dot = match dot.add(dot_group) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            candidate_norm_sq = match candidate_norm_sq.add_nonnegative(norm_group) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            offset += 8;
        }

        for &query_value in query_chunks.remainder() {
            let (lower, upper) = component_bounds(offset);
            let candidate_value = F32Interval { lo: lower, hi: upper };
            let product = match F32Interval::point(query_value).mul(candidate_value) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            let square = match candidate_value.square() {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            dot = match dot.add(product) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            candidate_norm_sq = match candidate_norm_sq.add_nonnegative(square) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            offset += 1;
        }

        let norm_product = match F32Interval::point(query_norm_sq).mul(candidate_norm_sq) {
            Some(value) => F32Interval {
                lo: value.lo.max(0.0),
                hi: value.hi.max(0.0),
            },
            None => return Ok(ContinuousHVScoreInterval::uninformative()),
        };
        let denominator = match norm_product.sqrt_nonnegative() {
            Some(value) => value,
            None => return Ok(ContinuousHVScoreInterval::uninformative()),
        };
        const CANONICAL_NEAR_ZERO_DENOMINATOR: f32 = 1e-10;
        if denominator.hi < CANONICAL_NEAR_ZERO_DENOMINATOR {
            return Ok(ContinuousHVScoreInterval { lower: 0.0, upper: 0.0 });
        }
        if denominator.lo < CANONICAL_NEAR_ZERO_DENOMINATOR {
            return Ok(ContinuousHVScoreInterval::uninformative());
        }
        let quotient = match dot.div_positive(denominator) {
            Some(value) => value,
            None => return Ok(ContinuousHVScoreInterval::uninformative()),
        };
        let interval = quotient.clamp_cosine();
        if interval.is_well_formed() {
            Ok(interval)
        } else {
            Ok(ContinuousHVScoreInterval::uninformative())
        }
    }

    /// Enclose the canonical full-resolution score for any admitted candidate
    /// x satisfying ||x - reconstruction||_2 <= residual_l2_upper.
    ///
    /// The query and reconstruction must have equal dimensions and finite
    /// components in [-1, 1]. The unknown original candidate must also be
    /// admitted to [-1, 1]; this is a segment-admission precondition.
    ///
    /// The L2 certificate must be independently validated by the caller. This
    /// method does not prove it. A zero radius returns a point interval at the
    /// canonical score. Near-zero denominator, overflow, or ambiguous interval
    /// arithmetic returns [-1, 1], safe but unusable for pruning.
    ///
    /// This reference relaxes the L2 ball to coordinate boxes. It is
    /// conservative under the canonical IEEE-754 f32 score contract, but may
    /// be wider than a future blockwise interval kernel.
    pub fn similarity_interval_full_resolution_for_l2_residual(
        &self,
        reconstruction: &Self,
        residual_l2_upper: f32,
    ) -> Result<ContinuousHVScoreInterval, ContinuousHVIntervalError> {
        self.validate_full_resolution_pair(reconstruction)?;
        if !residual_l2_upper.is_finite() || residual_l2_upper < 0.0 {
            return Err(ContinuousHVIntervalError::InvalidResidualBound);
        }

        if residual_l2_upper == 0.0 {
            let score = self.similarity_full_resolution(reconstruction);
            return Ok(ContinuousHVScoreInterval { lower: score, upper: score });
        }

        let query_norm_sq = canonical_norm_squared(&self.values);
        if !query_norm_sq.is_finite() {
            return Ok(ContinuousHVScoreInterval::uninformative());
        }
        if query_norm_sq == 0.0 {
            return Ok(ContinuousHVScoreInterval { lower: 0.0, upper: 0.0 });
        }

        let mut dot = F32Interval::point(0.0);
        let mut candidate_norm_sq = F32Interval::point(0.0);
        let mut query_chunks = self.values.chunks_exact(8);
        let mut reconstruction_chunks = reconstruction.values.chunks_exact(8);

        for (query_chunk, reconstruction_chunk) in query_chunks
            .by_ref()
            .zip(reconstruction_chunks.by_ref())
        {
            let candidate_chunk: [F32Interval; 8] = std::array::from_fn(|index| {
                component_interval(reconstruction_chunk[index], residual_l2_upper)
            });
            let dot_group = match product_group(query_chunk, &candidate_chunk) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            let norm_group = match square_group(&candidate_chunk) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            dot = match dot.add(dot_group) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            candidate_norm_sq = match candidate_norm_sq.add_nonnegative(norm_group) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
        }

        for (&query_value, &reconstruction_value) in query_chunks
            .remainder()
            .iter()
            .zip(reconstruction_chunks.remainder().iter())
        {
            let candidate_value = component_interval(reconstruction_value, residual_l2_upper);
            let product = match F32Interval::point(query_value).mul(candidate_value) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            let square = match candidate_value.square() {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            dot = match dot.add(product) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
            candidate_norm_sq = match candidate_norm_sq.add_nonnegative(square) {
                Some(value) => value,
                None => return Ok(ContinuousHVScoreInterval::uninformative()),
            };
        }

        let norm_product = match F32Interval::point(query_norm_sq).mul(candidate_norm_sq) {
            Some(value) => F32Interval {
                lo: value.lo.max(0.0),
                hi: value.hi.max(0.0),
            },
            None => return Ok(ContinuousHVScoreInterval::uninformative()),
        };
        let denominator = match norm_product.sqrt_nonnegative() {
            Some(value) => value,
            None => return Ok(ContinuousHVScoreInterval::uninformative()),
        };

        const CANONICAL_NEAR_ZERO_DENOMINATOR: f32 = 1e-10;
        if denominator.hi < CANONICAL_NEAR_ZERO_DENOMINATOR {
            return Ok(ContinuousHVScoreInterval { lower: 0.0, upper: 0.0 });
        }
        if denominator.lo < CANONICAL_NEAR_ZERO_DENOMINATOR {
            return Ok(ContinuousHVScoreInterval::uninformative());
        }

        let quotient = match dot.div_positive(denominator) {
            Some(value) => value,
            None => return Ok(ContinuousHVScoreInterval::uninformative()),
        };
        let interval = quotient.clamp_cosine();
        if interval.is_well_formed() {
            Ok(interval)
        } else {
            Ok(ContinuousHVScoreInterval::uninformative())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random_unit(state: &mut u32) -> f32 {
        *state ^= *state << 13;
        *state ^= *state >> 17;
        *state ^= *state << 5;
        let unit = (*state >> 8) as f32 / (u32::MAX >> 8) as f32;
        unit * 2.0 - 1.0
    }

    #[test]
    fn residual_certificate_verifier_accepts_exact_and_conservative_radii() {
        let original = ContinuousHV::from_vec(vec![0.0, 0.5, -0.25, 1.0]);
        let same = original.clone();
        let exact = verify_l2_residual_certificate(&original, &same, 0.0).unwrap();
        assert_eq!(exact.residual_norm_squared_upper, 0.0);
        assert_eq!(exact.changed_coordinates, 0);

        let reconstruction = ContinuousHV::from_vec(vec![0.01, 0.49, -0.25, 1.0]);
        let verified =
            verify_l2_residual_certificate(&original, &reconstruction, 0.015).unwrap();
        assert_eq!(verified.changed_coordinates, 2);
        assert!(verified.residual_norm_squared_upper <= verified.certified_norm_squared);
    }

    #[test]
    fn residual_certificate_verifier_rejects_underreported_and_invalid_claims() {
        let original = ContinuousHV::from_vec(vec![0.5, -0.5, 0.25]);
        let reconstruction = ContinuousHV::from_vec(vec![0.6, -0.5, 0.25]);
        assert!(matches!(
            verify_l2_residual_certificate(&original, &reconstruction, 0.05),
            Err(ResidualCertificateError::Underreported { .. })
        ));
        assert_eq!(
            verify_l2_residual_certificate(&original, &reconstruction, f32::NAN),
            Err(ResidualCertificateError::InvalidEpsilon)
        );
        let invalid = ContinuousHV::from_vec(vec![f32::INFINITY, 0.0, 0.0]);
        assert!(matches!(
            verify_l2_residual_certificate(&invalid, &invalid, 0.1),
            Err(ResidualCertificateError::InvalidProfile(
                ContinuousHVProfileError::NonFiniteComponent { .. }
            ))
        ));
        let wrong_dim = ContinuousHV::ones(2);
        assert!(matches!(
            verify_l2_residual_certificate(&original, &wrong_dim, 1.0),
            Err(ResidualCertificateError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn adjacent_value_helpers_cover_signed_zero_subnormals_and_infinities() {
        let tiny = f32::from_bits(1);
        assert_eq!(next_up(0.0).to_bits(), 1);
        assert_eq!(next_up(-0.0).to_bits(), 1);
        assert_eq!(next_down(0.0).to_bits(), 0x8000_0001);
        assert_eq!(next_up(-tiny).to_bits(), (-0.0f32).to_bits());
        assert_eq!(next_down(tiny).to_bits(), 0.0f32.to_bits());
        assert_eq!(next_up(f32::NEG_INFINITY), -f32::MAX);
        assert_eq!(next_down(f32::INFINITY), f32::MAX);
        assert_eq!(next_up(f32::MAX), f32::INFINITY);
        assert_eq!(next_down(f32::INFINITY), f32::MAX);
        assert_eq!(next_down(-f32::MAX), f32::NEG_INFINITY);
    }

    #[test]
    fn l2_ball_interval_contains_candidates_inside_verified_radius() {
        let query = ContinuousHV::from_vec(
            (0..257).map(|i| ((i as f32) * 0.013).sin()).collect(),
        );
        let center = ContinuousHV::from_vec(
            (0..257).map(|i| ((i as f32) * 0.017).cos() * 0.8).collect(),
        );
        let mut actual_values = center.values.clone();
        for (index, value) in actual_values.iter_mut().enumerate() {
            let delta = (((index as f32) * 0.19).sin()) * 0.0005;
            *value = (*value + delta).clamp(-1.0, 1.0);
        }
        let actual = ContinuousHV::from_vec(actual_values);
        let radius = 0.01;
        verify_l2_residual_certificate(&actual, &center, radius).unwrap();

        let interval = query
            .similarity_interval_full_resolution_for_l2_ball_center(&center, radius)
            .unwrap();
        assert!(interval.contains(query.similarity_full_resolution(&actual)));
        assert!(interval.is_well_formed());
    }

    #[test]
    fn l2_ball_interval_handles_zero_radius_and_rejects_invalid_radius() {
        let query = ContinuousHV::from_vec(vec![0.5, -0.25, 0.75, 0.1, -0.9, 0.2, 0.3, -0.4]);
        let center = ContinuousHV::from_vec(vec![0.2, -0.5, 0.8, -0.1, 0.7, 0.4, -0.3, 0.9]);
        let score = query.similarity_full_resolution(&center);
        let point = query
            .similarity_interval_full_resolution_for_l2_ball_center(&center, 0.0)
            .unwrap();
        assert_eq!(point.lower, score);
        assert_eq!(point.upper, score);
        assert_eq!(
            query.similarity_interval_full_resolution_for_l2_ball_center(&center, f32::NAN),
            Err(ContinuousHVIntervalError::InvalidResidualBound)
        );
    }

    #[test]
    fn l2_ball_interval_contains_multiple_members_and_zero_radius_group() {
        let query = ContinuousHV::from_vec(
            (0..129).map(|i| ((i as f32) * 0.031).sin()).collect(),
        );
        let center = ContinuousHV::from_vec(
            (0..129).map(|i| ((i as f32) * 0.021).cos() * 0.75).collect(),
        );
        let mut members = Vec::new();
        for offset in [0.0001f32, -0.0002, 0.0003, -0.0004] {
            let member = ContinuousHV::from_vec(
                center
                    .values
                    .iter()
                    .enumerate()
                    .map(|(index, value)| {
                        (*value + offset * (((index as f32) * 0.17).sin())).clamp(-1.0, 1.0)
                    })
                    .collect(),
            );
            verify_l2_residual_certificate(&member, &center, 0.01).unwrap();
            members.push(member);
        }
        let interval = query
            .similarity_interval_full_resolution_for_l2_ball_center(&center, 0.01)
            .unwrap();
        for member in &members {
            assert!(interval.contains(query.similarity_full_resolution(member)));
        }

        let point = query
            .similarity_interval_full_resolution_for_l2_ball_center(&center, 0.0)
            .unwrap();
        let exact = query.similarity_full_resolution(&center);
        assert_eq!(point.lower, exact);
        assert_eq!(point.upper, exact);
    }

    #[test]
    fn component_bounds_callback_matches_slice_kernel_without_decoded_storage() {
        let query = ContinuousHV::from_vec(vec![
            0.7, -0.2, 0.4, -0.9, 0.1, 0.8, -0.3, 0.5, 0.6,
        ]);
        let lower = vec![-0.2, -0.4, 0.1, -0.8, -0.1, 0.2, -0.6, 0.1, 0.0];
        let upper = vec![0.3, 0.1, 0.8, -0.4, 0.5, 0.9, -0.1, 0.7, 0.5];
        let slice_interval = query
            .similarity_interval_full_resolution_for_component_bounds(&lower, &upper)
            .unwrap();
        let callback_interval = query
            .similarity_interval_full_resolution_for_component_bounds_with(|index| {
                (lower[index], upper[index])
            })
            .unwrap();
        assert_eq!(slice_interval, callback_interval);
    }

    #[test]
    fn component_bounds_callback_rejects_invalid_bound_and_checks_tail() {
        let query = ContinuousHV::from_vec(vec![0.1; 9]);
        let interval = query
            .similarity_interval_full_resolution_for_component_bounds_with(|index| {
                if index == 8 { (-0.2, 0.3) } else { (-0.1, 0.1) }
            })
            .unwrap();
        let member = ContinuousHV::from_vec(
            vec![0.0; 8].into_iter().chain([0.25]).collect(),
        );
        assert!(interval.contains(query.similarity_full_resolution(&member)));

        let invalid = query.similarity_interval_full_resolution_for_component_bounds_with(
            |index| if index == 8 { (0.5, -0.5) } else { (-0.1, 0.1) },
        );
        assert!(matches!(
            invalid,
            Err(ContinuousHVIntervalError::InvalidCandidateComponentBounds { index: 8 })
        ));
    }

    #[test]
    fn component_box_interval_contains_every_group_member_score() {
        let query = ContinuousHV::from_vec(vec![
            0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.1,
        ]);
        let members = vec![
            ContinuousHV::from_vec(vec![
                0.91, -0.79, 0.69, -0.61, 0.51, -0.39, 0.29, -0.21, 0.11,
            ]),
            ContinuousHV::from_vec(vec![
                -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0,
            ]),
            ContinuousHV::from_vec(vec![
                0.0, 0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4,
            ]),
        ];
        let dimension = query.values.len();
        let mut lower = vec![f32::INFINITY; dimension];
        let mut upper = vec![f32::NEG_INFINITY; dimension];
        for member in &members {
            for index in 0..dimension {
                lower[index] = lower[index].min(member.values[index]);
                upper[index] = upper[index].max(member.values[index]);
            }
        }
        let interval = query
            .similarity_interval_full_resolution_for_component_bounds(&lower, &upper)
            .unwrap();
        for member in &members {
            let score = query.similarity_full_resolution(member);
            assert!(
                interval.contains(score),
                "member score {score} escaped [{}, {}]",
                interval.lower,
                interval.upper
            );
        }
    }

    #[test]
    fn component_box_interval_rejects_malformed_extrema() {
        let query = ContinuousHV::ones(9);
        let reversed = vec![0.0; 9];
        let upper = vec![0.0; 9];
        let mut invalid_lower = reversed;
        invalid_lower[3] = 0.5;
        assert!(matches!(
            query.similarity_interval_full_resolution_for_component_bounds(
                &invalid_lower,
                &upper
            ),
            Err(ContinuousHVIntervalError::InvalidCandidateComponentBounds { index: 3 })
        ));
        assert!(matches!(
            query.similarity_interval_full_resolution_for_component_bounds(
                &[0.0; 8],
                &[0.0; 9]
            ),
            Err(ContinuousHVIntervalError::InvalidProfile(
                ContinuousHVProfileError::DimensionMismatch { .. }
            ))
        ));
    }

    #[test]
    fn zero_residual_is_a_point_at_the_canonical_score() {
        let query = ContinuousHV::from_vec(vec![
            0.25, -0.75, 0.5, 0.125, -0.5, 0.9, 0.3, -0.2, 0.7,
        ]);
        let candidate = ContinuousHV::from_vec(vec![
            -0.1, 0.6, 0.4, -0.8, 0.2, 0.1, -0.9, 0.3, 0.2,
        ]);
        let score = query.similarity_full_resolution(&candidate);
        let interval = query
            .similarity_interval_full_resolution_for_l2_residual(&candidate, 0.0)
            .unwrap();
        assert_eq!(interval.lower, score);
        assert_eq!(interval.upper, score);
    }

    #[test]
    fn interval_contains_scores_for_candidates_inside_the_l2_certificate() {
        let mut state = 0xA11C_E55Du32;
        for dimension in [1usize, 7, 8, 9, 17, 64, 257] {
            for _ in 0..64 {
                let query = ContinuousHV::from_vec(
                    (0..dimension).map(|_| pseudo_random_unit(&mut state)).collect(),
                );
                let actual_values: Vec<f32> = (0..dimension)
                    .map(|_| pseudo_random_unit(&mut state))
                    .collect();
                let mut reconstruction_values = actual_values.clone();
                let delta = 0.01f32;
                if reconstruction_values[0] <= 0.98 {
                    reconstruction_values[0] += delta;
                } else {
                    reconstruction_values[0] -= delta;
                }
                let actual = ContinuousHV::from_vec(actual_values);
                let reconstruction = ContinuousHV::from_vec(reconstruction_values);
                let interval = query
                    .similarity_interval_full_resolution_for_l2_residual(
                        &reconstruction,
                        0.010_001,
                    )
                    .unwrap();
                let score = query.similarity_full_resolution(&actual);
                assert!(
                    interval.contains(score),
                    "score {score} escaped [{}, {}] at dimension {dimension}",
                    interval.lower,
                    interval.upper
                );
                assert!(interval.is_well_formed());
            }
        }
    }

    #[test]
    fn zero_query_and_near_zero_denominator_are_safe() {
        let zero_query = ContinuousHV::zero(9);
        let reconstruction = ContinuousHV::ones(9);
        let interval = zero_query
            .similarity_interval_full_resolution_for_l2_residual(&reconstruction, 0.1)
            .unwrap();
        assert_eq!(interval, ContinuousHVScoreInterval { lower: 0.0, upper: 0.0 });

        let small_query = ContinuousHV::from_vec(vec![1.0e-5; 2]);
        let small_reconstruction = ContinuousHV::from_vec(vec![1.0e-5; 2]);
        let interval = small_query
            .similarity_interval_full_resolution_for_l2_residual(
                &small_reconstruction,
                1.0e-5,
            )
            .unwrap();
        assert_eq!(interval, ContinuousHVScoreInterval::uninformative());
        assert!(interval.contains(small_query.similarity_full_resolution(&small_reconstruction)));
    }

    #[test]
    fn radius_two_covers_the_entire_admitted_candidate_domain() {
        let query = ContinuousHV::from_vec(vec![1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 0.75, -0.75]);
        let reconstruction = ContinuousHV::zero(8);
        let interval = query
            .similarity_interval_full_resolution_for_l2_residual(&reconstruction, 2.0)
            .unwrap();
        assert_eq!(interval, ContinuousHVScoreInterval::uninformative());
    }

    #[test]
    fn coordinate_box_contains_multiple_coordinate_residuals_within_l2_radius() {
        let query = ContinuousHV::from_vec(vec![0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.1]);
        let actual = ContinuousHV::from_vec(vec![0.91, -0.79, 0.69, -0.61, 0.51, -0.39, 0.29, -0.21, 0.11]);
        let reconstruction = ContinuousHV::from_vec(vec![0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2, 0.1]);
        let radius = (9.0f32 * 0.0001).sqrt() + 1.0e-6;
        let residual_sq = actual
            .values
            .iter()
            .zip(&reconstruction.values)
            .map(|(left, right)| (left - right) * (left - right))
            .sum::<f32>();
        assert!(residual_sq.sqrt() <= radius);
        let interval = query
            .similarity_interval_full_resolution_for_l2_residual(&reconstruction, radius)
            .unwrap();
        let score = query.similarity_full_resolution(&actual);
        assert!(interval.contains(score));
    }

    #[test]
    fn invalid_residual_bounds_are_rejected() {
        let query = ContinuousHV::ones(8);
        let reconstruction = ContinuousHV::ones(8);
        for invalid in [-0.1, f32::NAN, f32::INFINITY] {
            assert_eq!(
                query.similarity_interval_full_resolution_for_l2_residual(
                    &reconstruction,
                    invalid,
                ),
                Err(ContinuousHVIntervalError::InvalidResidualBound)
            );
        }
    }

    #[test]
    fn invalid_query_and_reconstruction_profiles_fail_closed() {
        let query = ContinuousHV::ones(8);
        let invalid = ContinuousHV::from_vec(vec![f32::NAN; 8]);
        assert!(matches!(
            query.similarity_interval_full_resolution_for_l2_residual(&invalid, 0.1),
            Err(ContinuousHVIntervalError::InvalidProfile(
                ContinuousHVProfileError::NonFiniteComponent { .. }
            ))
        ));
        let mismatch = ContinuousHV::ones(7);
        assert!(matches!(
            query.similarity_interval_full_resolution_for_l2_residual(&mismatch, 0.1),
            Err(ContinuousHVIntervalError::InvalidProfile(
                ContinuousHVProfileError::DimensionMismatch { .. }
            ))
        ));
    }


    fn exact_real_cosine(left: &[f32], right: &[f32]) -> f64 {
        let mut dot = 0.0f64;
        let mut left_norm_sq = 0.0f64;
        let mut right_norm_sq = 0.0f64;
        for (&a, &b) in left.iter().zip(right) {
            let a = a as f64;
            let b = b as f64;
            dot += a * b;
            left_norm_sq += a * a;
            right_norm_sq += b * b;
        }
        let denominator = (left_norm_sq * right_norm_sq).sqrt();
        if denominator == 0.0 { 0.0 } else { (dot / denominator).clamp(-1.0, 1.0) }
    }

    #[test]
    fn spherical_cap_real_cosine_interval_contains_high_dimensional_members() {
        let mut state = 0x51A7_C0DEu32;
        for dimension in [8usize, 9, 257, 16_384] {
            for scenario in 0..5 {
                let center_values: Vec<f32> = (0..dimension)
                    .map(|index| match scenario {
                        0 => ((index as f32) * 0.013).sin() * 0.7,
                        1 => if index % 2 == 0 { 0.8 } else { -0.8 },
                        2 => pseudo_random_unit(&mut state) * 0.7,
                        3 => if index == dimension / 2 { 0.7 } else { 0.0 },
                        _ => ((index as f32) * 0.017).cos() * 0.02,
                    })
                    .collect();
                let query_values: Vec<f32> = (0..dimension)
                    .map(|index| match scenario {
                        0 => center_values[index] * 0.9,
                        1 => -center_values[index],
                        2 => pseudo_random_unit(&mut state),
                        3 => if index == dimension / 2 { -1.0 } else { 0.0 },
                        _ => ((index as f32) * 0.009).sin(),
                    })
                    .collect();
                let center = ContinuousHV::from_vec(center_values);
                let query = ContinuousHV::from_vec(query_values);
                let radius = 0.01f32;
                let actual = ContinuousHV::from_vec(
                    center.values.iter().enumerate().map(|(index, value)| {
                        (*value + 0.0001 * ((index as f32) * 0.037).sin()).clamp(-1.0, 1.0)
                    }).collect(),
                );
                verify_l2_residual_certificate(&actual, &center, radius).unwrap();

                let interval = query
                    .real_cosine_interval_for_l2_ball_center(&center, radius)
                    .unwrap();
                let exact_real = exact_real_cosine(&query.values, &actual.values);
                assert!(
                    interval.contains(exact_real),
                    "scenario={scenario}, dimension={dimension}: real cosine={exact_real}, interval=[{}, {}]",
                    interval.lower,
                    interval.upper
                );
                assert!(interval.is_well_formed());
            }
        }
    }


    #[test]
    fn spherical_cap_f32_interval_contains_canonical_high_dimensional_members() {
        let mut state = 0xF32C_A9E5u32;
        for dimension in [8usize, 9, 17, 257, 2_048, 16_384] {
            for scenario in 0..4 {
                let center = ContinuousHV::from_vec(
                    (0..dimension)
                        .map(|index| match scenario {
                            0 => ((index as f32) * 0.013).sin() * 0.55,
                            1 => if index % 2 == 0 { 0.7 } else { -0.7 },
                            2 => pseudo_random_unit(&mut state) * 0.6,
                            _ => if index == dimension / 2 { 0.7 } else { 0.0 },
                        })
                        .collect(),
                );
                let query = ContinuousHV::from_vec(
                    (0..dimension)
                        .map(|index| match scenario {
                            0 => center.values[index] * 0.9,
                            1 => -center.values[index],
                            2 => pseudo_random_unit(&mut state),
                            _ => if index == dimension / 2 { 1.0 } else { 0.0 },
                        })
                        .collect(),
                );
                let actual = ContinuousHV::from_vec(
                    center
                        .values
                        .iter()
                        .enumerate()
                        .map(|(index, value)| {
                            (*value + 0.0001 * ((index as f32) * 0.037).sin())
                                .clamp(-1.0, 1.0)
                        })
                        .collect(),
                );
                let radius = 0.02f32;
                verify_l2_residual_certificate(&actual, &center, radius).unwrap();
                let interval = query
                    .similarity_interval_full_resolution_for_l2_ball_center_spherical_cap(
                        &center,
                        radius,
                    )
                    .unwrap();
                let exact = query.similarity_full_resolution(&actual);
                assert!(
                    interval.contains(exact),
                    "scenario={scenario}, dimension={dimension}: exact={exact}, interval=[{}, {}]",
                    interval.lower,
                    interval.upper
                );
                assert!(interval.is_well_formed());
            }
        }
    }


    #[test]
    fn spherical_cap_f32_interval_property_random_residual_directions() {
        let mut state = 0xF32B_A11Du32;
        for dimension in [1usize, 7, 8, 9, 17, 257, 2_048, 16_384] {
            for _ in 0..16 {
                let center = ContinuousHV::from_vec(
                    (0..dimension)
                        .map(|_| pseudo_random_unit(&mut state) * 0.5)
                        .collect(),
                );
                let query = ContinuousHV::from_vec(
                    (0..dimension)
                        .map(|_| pseudo_random_unit(&mut state))
                        .collect(),
                );
                let center_norm = center
                    .values
                    .iter()
                    .map(|value| (*value as f64) * (*value as f64))
                    .sum::<f64>()
                    .sqrt();
                if center_norm <= 1.0e-6 {
                    continue;
                }
                let radius = (center_norm * 0.04).min(0.1) as f32;
                let direction: Vec<f32> = (0..dimension)
                    .map(|_| pseudo_random_unit(&mut state))
                    .collect();
                let direction_norm = direction
                    .iter()
                    .map(|value| (*value as f64) * (*value as f64))
                    .sum::<f64>()
                    .sqrt();
                if direction_norm == 0.0 {
                    continue;
                }
                let scale = (radius as f64 * 0.9) / direction_norm;
                let actual = ContinuousHV::from_vec(
                    center
                        .values
                        .iter()
                        .zip(direction.iter())
                        .map(|(c, d)| ((*c as f64 + *d as f64 * scale) as f32).clamp(-1.0, 1.0))
                        .collect(),
                );
                verify_l2_residual_certificate(&actual, &center, radius).unwrap();

                let interval = query
                    .similarity_interval_full_resolution_for_l2_ball_center_spherical_cap(
                        &center,
                        radius,
                    )
                    .unwrap();
                let exact = query.similarity_full_resolution(&actual);
                assert!(
                    interval.contains(exact),
                    "dimension={dimension}: canonical score={exact}, interval=[{}, {}]",
                    interval.lower,
                    interval.upper
                );
            }
        }
    }

    #[test]
    fn spherical_cap_f32_interval_uses_canonical_near_zero_branch() {
        let query = ContinuousHV::from_vec(vec![1.0e-6; 8]);
        let center = ContinuousHV::from_vec(vec![1.0e-6; 8]);
        let interval = query
            .similarity_interval_full_resolution_for_l2_ball_center_spherical_cap(
                &center,
                1.0e-6,
            )
            .unwrap();
        assert_eq!(interval, ContinuousHVSphericalCapF32ScoreInterval::zero_score());
        assert_eq!(query.similarity_full_resolution(&center), 0.0);
    }

    #[test]
    fn spherical_cap_f32_interval_contains_cancellation_sensitive_canonical_score() {
        let one = 1.0f32;
        let a0 = one - f32::EPSILON / 2.0;
        let b1 = one - f32::EPSILON;
        let query = ContinuousHV::from_vec(vec![a0, one, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let center =
            ContinuousHV::from_vec(vec![a0, -b1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // This exact cancellation is sensitive to accidental FMA contraction
        // in the canonical group-of-eight dot reduction.
        assert_eq!(a0 * a0, b1);
        let exact = query.similarity_full_resolution(&center);
        assert_eq!(exact, 0.0);

        let interval = query
            .similarity_interval_full_resolution_for_l2_ball_center_spherical_cap(
                &center,
                0.01,
            )
            .unwrap();
        assert!(
            interval.contains(exact),
            "canonical score {exact} escaped [{}, {}]",
            interval.lower,
            interval.upper
        );
    }

    #[test]
    fn spherical_cap_f32_interval_handles_zero_radius_and_zero_query() {
        let query = ContinuousHV::from_vec(vec![0.5, -0.2, 0.7, 0.1, -0.8, 0.3, 0.9, -0.4]);
        let center = ContinuousHV::from_vec(vec![0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, -0.8]);
        let exact = query.similarity_full_resolution(&center);
        let point = query
            .similarity_interval_full_resolution_for_l2_ball_center_spherical_cap(&center, 0.0)
            .unwrap();
        assert_eq!(point.lower, exact);
        assert_eq!(point.upper, exact);

        let zero_query = ContinuousHV::zero(8);
        let zero_score = zero_query
            .similarity_interval_full_resolution_for_l2_ball_center_spherical_cap(&center, 0.1)
            .unwrap();
        assert_eq!(zero_score, ContinuousHVSphericalCapF32ScoreInterval::zero_score());
    }

    #[test]
    fn spherical_cap_real_cosine_interval_property_residual_directions() {
        let mut state = 0xB0A1_5EEDu32;
        for dimension in [1usize, 7, 8, 9, 17, 257, 2_048, 16_384] {
            for _ in 0..32 {
                let center = ContinuousHV::from_vec(
                    (0..dimension)
                        .map(|_| pseudo_random_unit(&mut state) * 0.5)
                        .collect(),
                );
                let query = ContinuousHV::from_vec(
                    (0..dimension)
                        .map(|_| pseudo_random_unit(&mut state))
                        .collect(),
                );
                let center_norm = center
                    .values
                    .iter()
                    .map(|value| (*value as f64) * (*value as f64))
                    .sum::<f64>()
                    .sqrt();
                if center_norm <= 1.0e-6 {
                    continue;
                }
                // A range of small-to-large balls, all strictly separated
                // from the origin, while preserving headroom in [-1, 1].
                let radius = (center_norm * 0.05).min(0.2) as f32;
                let mut direction: Vec<f32> = (0..dimension)
                    .map(|_| pseudo_random_unit(&mut state))
                    .collect();
                let direction_norm = direction
                    .iter()
                    .map(|value| (*value as f64) * (*value as f64))
                    .sum::<f64>()
                    .sqrt();
                if direction_norm == 0.0 {
                    direction[0] = 1.0;
                }
                let direction_norm = direction
                    .iter()
                    .map(|value| (*value as f64) * (*value as f64))
                    .sum::<f64>()
                    .sqrt();
                let scale = (radius as f64 * 0.9) / direction_norm;
                let actual = ContinuousHV::from_vec(
                    center
                        .values
                        .iter()
                        .zip(direction.iter())
                        .map(|(c, d)| ((*c as f64 + *d as f64 * scale) as f32).clamp(-1.0, 1.0))
                        .collect(),
                );
                verify_l2_residual_certificate(&actual, &center, radius).unwrap();

                let interval = query
                    .real_cosine_interval_for_l2_ball_center(&center, radius)
                    .unwrap();
                let exact_real = exact_real_cosine(&query.values, &actual.values);
                assert!(
                    interval.contains(exact_real),
                    "dimension={dimension}: real cosine={exact_real}, interval=[{}, {}]",
                    interval.lower,
                    interval.upper
                );
            }
        }
    }

    #[test]
    fn spherical_cap_saturates_aligned_and_antipodal_directions() {
        let center = ContinuousHV::from_vec(vec![0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let aligned = ContinuousHV::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let antipodal = ContinuousHV::from_vec(vec![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let upper = aligned
            .real_cosine_interval_for_l2_ball_center(&center, 0.2)
            .unwrap();
        let lower = antipodal
            .real_cosine_interval_for_l2_ball_center(&center, 0.2)
            .unwrap();
        assert_eq!(upper.upper, 1.0);
        assert_eq!(lower.lower, -1.0);
        assert!(upper.contains(1.0));
        assert!(lower.contains(-1.0));
    }

    #[test]
    fn spherical_cap_falls_back_when_ball_may_contain_origin() {
        let center = ContinuousHV::from_vec(vec![0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let query = ContinuousHV::ones(8);
        let interval = query
            .real_cosine_interval_for_l2_ball_center(&center, 0.2)
            .unwrap();
        assert_eq!(interval, ContinuousHVRealCosineInterval::uninformative());
    }

    #[test]
    fn l2_ball_interval_contains_high_dimensional_adversarial_payloads() {
        // Differential containment against the canonical original-payload
        // scorer. This is not a performance result or a platform proof.
        let mut state = 0xC0DE_F00Du32;
        for dimension in [8usize, 9, 257, 16_384] {
            for scenario in 0..4 {
                let query_values: Vec<f32> = (0..dimension)
                    .map(|index| match scenario {
                        0 => ((index as f32) * 0.013).sin() * 0.8,
                        1 => if index % 2 == 0 { 0.999 } else { -0.999 },
                        2 => if index == dimension / 2 { 1.0 } else { 0.0 },
                        _ => ((index as f32) * 0.019).cos() * 1.0e-5,
                    })
                    .collect();
                let center_values: Vec<f32> = (0..dimension)
                    .map(|index| match scenario {
                        0 => query_values[index] * 0.8,
                        1 => -query_values[index] * 0.8,
                        2 => pseudo_random_unit(&mut state) * 0.8,
                        _ => ((index as f32) * 0.017).sin() * 0.6,
                    })
                    .collect();
                let query = ContinuousHV::from_vec(query_values);
                let center = ContinuousHV::from_vec(center_values);
                let actual = ContinuousHV::from_vec(
                    center
                        .values
                        .iter()
                        .enumerate()
                        .map(|(index, value)| {
                            let direction = ((index as f32) * 0.071).sin();
                            (*value + 0.0005 * direction).clamp(-1.0, 1.0)
                        })
                        .collect(),
                );
                let epsilon = 0.07f32;
                verify_l2_residual_certificate(&actual, &center, epsilon)
                    .unwrap_or_else(|error| panic!("valid generated residual rejected: {error}"));
                let interval = query
                    .similarity_interval_full_resolution_for_l2_ball_center(&center, epsilon)
                    .unwrap();
                let exact = query.similarity_full_resolution(&actual);
                assert!(
                    interval.contains(exact),
                    "scenario={scenario}, dimension={dimension}: exact={exact}, interval=[{}, {}]",
                    interval.lower,
                    interval.upper
                );
                assert!(interval.is_well_formed());
            }
        }
    }

}
