// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Analytic null model for finite-dimensional validity-memory crosstalk.
//!
//! This module is deliberately separate from the empirical capacity sweep. It
//! freezes a pre-result prediction under a simplified random-association model so
//! the sweep can test a theory rather than merely fit a curve after observation.
//!
//! ## Null-model assumptions
//!
//! 1. Every causal checkpoint contains exactly one active fact for each relation
//!    key, so the archive contains `K * H` key/checkpoint facts.
//! 2. Relative to a queried association, every non-target association behaves as
//!    an independent Rademacher sign vector across dimensions.
//! 3. Temporal frequencies are iid uniform on `[-pi, pi)`, so for a non-zero
//!    integer checkpoint displacement `delta`,
//!    `E[cos(omega * delta)] = 0` and `E[cos^2(omega * delta)] = 1/2`.
//! 4. Interference terms are treated as uncorrelated. The real archive reuses key
//!    and candidate codewords, so repeated associations can violate this idealized
//!    independence. The empirical sweep records codebook coherence and realized
//!    semantic changes specifically to expose such departures.
//!
//! Under those assumptions, for a correct candidate queried at one checkpoint:
//!
//! - the target contributes exactly `1`;
//! - `K - 1` same-checkpoint facts contribute unit-variance Rademacher noise;
//! - `K * (H - 1)` cross-checkpoint facts contribute half-variance temporal noise.
//!
//! Therefore the target-score noise variance is
//!
//! `((K - 1) + 0.5 * K * (H - 1)) / D`
//!
//! `= (K * (H + 1) - 2) / (2D)`.
//!
//! For an unrelated distractor candidate there is no unit target term, so all `K`
//! same-checkpoint facts are interference and the corresponding null variance is
//!
//! `(K + 0.5 * K * (H - 1)) / D`
//!
//! `= K * (H + 1) / (2D)`.
//!
//! Writing `rho = K * H / D`, both variances approach `rho / 2` for large
//! horizons. This is the analytic reason `research_v0` treats facts-per-dimension
//! as its primary load coordinate.

use crate::validity_capacity::ValidityCapacityCase;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValidityCapacityNullModel {
    pub key_count: usize,
    pub candidate_count: usize,
    pub horizon: u64,
    pub dim: usize,
    pub represented_facts: u64,
    pub facts_per_dimension: f64,
    pub same_checkpoint_target_interferers: u64,
    pub cross_checkpoint_interferers: u64,
    /// Idealized variance of the correct-candidate score around mean 1.
    pub target_noise_variance: f64,
    pub target_noise_std: f64,
    /// Idealized variance of one unrelated distractor score around mean 0.
    pub distractor_noise_variance: f64,
    pub distractor_noise_std: f64,
    /// `1 / target_noise_std`; infinity when the null variance is exactly zero.
    pub target_signal_to_noise: f64,
    /// Large-horizon approximation `rho / 2` shared by target/distractor variance.
    pub large_horizon_variance_proxy: f64,
    /// Gaussian extreme-value scale `sigma_d * sqrt(2 ln(C-1))`.
    ///
    /// This is a descriptive characteristic scale, not an exact expectation or a
    /// formal probability bound. Candidate scores in the real archive are
    /// correlated because they share one memory state.
    pub gaussian_distractor_extreme_scale: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidityCapacityTheoryError {
    ZeroDimension,
    ZeroKeyCount,
    TooFewCandidates,
    ZeroHorizon,
    SizeOverflow,
}

impl fmt::Display for ValidityCapacityTheoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "validity null model dimension must be non-zero"),
            Self::ZeroKeyCount => write!(f, "validity null model key count must be non-zero"),
            Self::TooFewCandidates => write!(f, "validity null model requires at least two candidates"),
            Self::ZeroHorizon => write!(f, "validity null model horizon must be non-zero"),
            Self::SizeOverflow => write!(f, "validity null model derived size overflowed u64"),
        }
    }
}

impl std::error::Error for ValidityCapacityTheoryError {}

impl ValidityCapacityNullModel {
    pub fn from_case(case: ValidityCapacityCase) -> Result<Self, ValidityCapacityTheoryError> {
        Self::new(case.dim, case.key_count, case.candidate_count, case.horizon)
    }

    pub fn new(
        dim: usize,
        key_count: usize,
        candidate_count: usize,
        horizon: u64,
    ) -> Result<Self, ValidityCapacityTheoryError> {
        if dim == 0 {
            return Err(ValidityCapacityTheoryError::ZeroDimension);
        }
        if key_count == 0 {
            return Err(ValidityCapacityTheoryError::ZeroKeyCount);
        }
        if candidate_count < 2 {
            return Err(ValidityCapacityTheoryError::TooFewCandidates);
        }
        if horizon == 0 {
            return Err(ValidityCapacityTheoryError::ZeroHorizon);
        }

        let key_count_u64 = u64::try_from(key_count)
            .map_err(|_| ValidityCapacityTheoryError::SizeOverflow)?;
        let represented_facts = key_count_u64
            .checked_mul(horizon)
            .ok_or(ValidityCapacityTheoryError::SizeOverflow)?;
        let same_checkpoint_target_interferers = key_count_u64 - 1;
        let cross_checkpoint_interferers = key_count_u64
            .checked_mul(horizon - 1)
            .ok_or(ValidityCapacityTheoryError::SizeOverflow)?;

        let dim_f = dim as f64;
        let target_noise_variance = (
            same_checkpoint_target_interferers as f64
                + 0.5 * cross_checkpoint_interferers as f64
        ) / dim_f;
        let distractor_noise_variance = (
            key_count_u64 as f64 + 0.5 * cross_checkpoint_interferers as f64
        ) / dim_f;
        let target_noise_std = target_noise_variance.sqrt();
        let distractor_noise_std = distractor_noise_variance.sqrt();
        let facts_per_dimension = represented_facts as f64 / dim_f;
        let large_horizon_variance_proxy = 0.5 * facts_per_dimension;
        let distractor_count = (candidate_count - 1) as f64;
        let gaussian_distractor_extreme_scale = if distractor_count <= 1.0 {
            0.0
        } else {
            distractor_noise_std * (2.0 * distractor_count.ln()).sqrt()
        };

        Ok(Self {
            key_count,
            candidate_count,
            horizon,
            dim,
            represented_facts,
            facts_per_dimension,
            same_checkpoint_target_interferers,
            cross_checkpoint_interferers,
            target_noise_variance,
            target_noise_std,
            distractor_noise_variance,
            distractor_noise_std,
            target_signal_to_noise: if target_noise_std == 0.0 {
                f64::INFINITY
            } else {
                1.0 / target_noise_std
            },
            large_horizon_variance_proxy,
            gaussian_distractor_extreme_scale,
        })
    }

    /// Exact finite-horizon correction to the `rho / 2` target-variance proxy.
    ///
    /// `target_variance = rho/2 + (K - 2)/(2D)`.
    pub fn target_variance_correction(&self) -> f64 {
        self.target_noise_variance - self.large_horizon_variance_proxy
    }

    /// Exact finite-horizon correction to the `rho / 2` distractor-variance proxy.
    ///
    /// `distractor_variance = rho/2 + K/(2D)`.
    pub fn distractor_variance_correction(&self) -> f64 {
        self.distractor_noise_variance - self.large_horizon_variance_proxy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validity_capacity::ValidityCapacityAxis;

    #[test]
    fn null_variance_matches_interference_count_derivation() {
        let model = ValidityCapacityNullModel::new(4096, 8, 8, 128).unwrap();
        assert_eq!(model.represented_facts, 1024);
        assert_eq!(model.same_checkpoint_target_interferers, 7);
        assert_eq!(model.cross_checkpoint_interferers, 8 * 127);

        let target = (7.0 + 0.5 * (8.0 * 127.0)) / 4096.0;
        let distractor = (8.0 + 0.5 * (8.0 * 127.0)) / 4096.0;
        assert!((model.target_noise_variance - target).abs() < 1e-15);
        assert!((model.distractor_noise_variance - distractor).abs() < 1e-15);
    }

    #[test]
    fn rho_half_relation_has_exact_finite_horizon_correction() {
        let model = ValidityCapacityNullModel::new(2048, 16, 8, 256).unwrap();
        let expected_target_correction = (16.0 - 2.0) / (2.0 * 2048.0);
        let expected_distractor_correction = 16.0 / (2.0 * 2048.0);
        assert!((model.target_variance_correction() - expected_target_correction).abs() < 1e-15);
        assert!((model.distractor_variance_correction() - expected_distractor_correction).abs() < 1e-15);
    }

    #[test]
    fn longer_horizon_makes_rho_half_a_tighter_relative_target_proxy() {
        let short = ValidityCapacityNullModel::new(4096, 8, 8, 16).unwrap();
        let long = ValidityCapacityNullModel::new(4096, 8, 8, 512).unwrap();
        let short_relative = short.target_variance_correction().abs() / short.target_noise_variance;
        let long_relative = long.target_variance_correction().abs() / long.target_noise_variance;
        assert!(long_relative < short_relative);
    }

    #[test]
    fn candidate_count_changes_extreme_scale_not_single_score_variance() {
        let small = ValidityCapacityNullModel::new(4096, 8, 4, 128).unwrap();
        let large = ValidityCapacityNullModel::new(4096, 8, 32, 128).unwrap();
        assert_eq!(small.target_noise_variance, large.target_noise_variance);
        assert_eq!(small.distractor_noise_variance, large.distractor_noise_variance);
        assert!(large.gaussian_distractor_extreme_scale > small.gaussian_distractor_extreme_scale);
    }

    #[test]
    fn capacity_case_adapter_ignores_span_length_by_design() {
        let case = ValidityCapacityCase {
            axis: ValidityCapacityAxis::SpanLength,
            dim: 4096,
            key_count: 8,
            candidate_count: 8,
            horizon: 256,
            span_length: 1,
        };
        let first = ValidityCapacityNullModel::from_case(case).unwrap();
        let second = ValidityCapacityNullModel::from_case(ValidityCapacityCase {
            span_length: 64,
            ..case
        })
        .unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn malformed_null_model_inputs_fail_closed() {
        assert!(matches!(
            ValidityCapacityNullModel::new(0, 8, 8, 128),
            Err(ValidityCapacityTheoryError::ZeroDimension)
        ));
        assert!(matches!(
            ValidityCapacityNullModel::new(4096, 0, 8, 128),
            Err(ValidityCapacityTheoryError::ZeroKeyCount)
        ));
        assert!(matches!(
            ValidityCapacityNullModel::new(4096, 8, 1, 128),
            Err(ValidityCapacityTheoryError::TooFewCandidates)
        ));
        assert!(matches!(
            ValidityCapacityNullModel::new(4096, 8, 8, 0),
            Err(ValidityCapacityTheoryError::ZeroHorizon)
        ));
    }
}
