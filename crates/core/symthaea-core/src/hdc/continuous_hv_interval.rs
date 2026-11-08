// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Conservative f32 score intervals for compressed ContinuousHV candidates.
//!
//! A verified L2 residual bound implies a per-coordinate error bound. This
//! reference relaxes the L2 ball to a coordinate box, then propagates outward
//! f32 intervals through the same group-of-eight and ordered-tail reduction as
//! ContinuousHV::similarity_full_resolution. The box relaxation may be loose,
//! but uncertainty can only widen the result, never authorize an unsafe prune.
//!
//! The caller must independently validate the residual certificate and ensure
//! that the original payload belongs to the admitted [-1, 1] profile. A digest
//! binding certificate bytes is not proof of the residual inequality.

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

impl ContinuousHV {
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
}
