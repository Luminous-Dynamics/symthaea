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

impl ContinuousHV {
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
