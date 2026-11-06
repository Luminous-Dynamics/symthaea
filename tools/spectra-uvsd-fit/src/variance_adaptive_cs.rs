// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Variance-adaptive time-uniform confidence sequences for bounded score streams.
//!
//! Experimental implementation of Theorem 3.1, equation (3.9), in Ramdas and
//! Martínez-Taboada (2026), arXiv:2610.42058. It assumes a fixed conditional
//! mean and a known conditional sub-Gaussian proxy. For scores in a declared
//! interval [lower, upper], Hoeffding's lemma permits proxy
//! s = (upper - lower) / 2.
//!
//! This module is deliberately standalone. It is not yet wired into
//! ScoreBoundsReceipt, allocation, stopping, or replay composition. Callers
//! must allocate alpha across claims and freeze the estimand, stream,
//! conditional-mean assumption, score range, and source revision.
//!
//! The interval sequence is valid at every time under the stated model.
//! The sharp LIL-width conclusion additionally requires positive limiting
//! average conditional variance; finite-sample coverage does not.

use std::fmt;

/// Stable method identifier for the experimental Ramdas–Martínez-Taboada CS.
pub const VARIANCE_ADAPTIVE_SUBGAUSSIAN_CS_METHOD_ID: &str =
    "spectra-subgaussian-variance-adaptive-lil-cs-ramdas-martinez-taboada-2026-v1";

/// Errors returned by the online confidence-sequence state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarianceAdaptiveCsError {
    InvalidScoreRange,
    InvalidAlpha,
    NonFiniteSample,
    SampleOutsideDeclaredRange,
    SampleCountOverflow,
    NumericFailure,
    NoObservations,
}

impl fmt::Display for VarianceAdaptiveCsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidScoreRange => "score range must be finite, ordered, and have finite width",
            Self::InvalidAlpha => "two-sided alpha must be finite and in (0, 1)",
            Self::NonFiniteSample => "score sample must be finite",
            Self::SampleOutsideDeclaredRange => "score sample lies outside the declared range",
            Self::SampleCountOverflow => "score sample count overflowed u64",
            Self::NumericFailure => "variance-adaptive confidence sequence was not representable",
            Self::NoObservations => "a non-degenerate confidence sequence needs at least one observation",
        };
        f.write_str(message)
    }
}

impl std::error::Error for VarianceAdaptiveCsError {}

/// A score confidence interval at one look.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VarianceAdaptiveScoreInterval {
    pub sample_count: u64,
    pub point_estimate: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub half_width: f64,
}

impl VarianceAdaptiveScoreInterval {
    pub fn width(self) -> f64 {
        self.upper_bound - self.lower_bound
    }
}

/// Online state for equation (3.9) of Ramdas and Martínez-Taboada (2026).
///
/// two_sided_alpha is the total error budget for both tails of this one
/// stream. When the surrounding SPECTRA familywise contract allocates delta
/// to each directional claim, construct with new_for_one_sided_claim(...,
/// delta), which uses a two-sided budget of 2 * delta.
///
/// Observations must satisfy the fixed-mean conditional sub-Gaussian model:
/// E[X_i | F_(i-1)] = mu and
/// E[exp(theta (X_i - mu)) | F_(i-1)] <= exp(theta^2 s^2 / 2).
/// For bounded observations the declared range supplies s by Hoeffding's
/// lemma. This state cannot verify stationarity or conditional sub-Gaussianity.
#[derive(Debug, Clone, PartialEq)]
pub struct VarianceAdaptiveSubGaussianCs {
    range_lower: f64,
    range_upper: f64,
    subgaussian_proxy: f64,
    two_sided_alpha: f64,
    sample_count: u64,
    mean: f64,
    centered_sum_squares: f64,
    harmonic_sum: f64,
}

impl VarianceAdaptiveSubGaussianCs {
    /// Construct a sequence with a total two-sided error budget alpha.
    pub fn new(
        range_lower: f64,
        range_upper: f64,
        two_sided_alpha: f64,
    ) -> Result<Self, VarianceAdaptiveCsError> {
        let width = range_upper - range_lower;
        if !range_lower.is_finite()
            || !range_upper.is_finite()
            || range_lower > range_upper
            || !width.is_finite()
        {
            return Err(VarianceAdaptiveCsError::InvalidScoreRange);
        }
        if !two_sided_alpha.is_finite()
            || two_sided_alpha <= 0.0
            || two_sided_alpha >= 1.0
        {
            return Err(VarianceAdaptiveCsError::InvalidAlpha);
        }

        let subgaussian_proxy = width * 0.5;
        if !subgaussian_proxy.is_finite()
            || !(subgaussian_proxy * subgaussian_proxy).is_finite()
            || !(subgaussian_proxy * subgaussian_proxy * subgaussian_proxy).is_finite()
        {
            return Err(VarianceAdaptiveCsError::InvalidScoreRange);
        }

        Ok(Self {
            range_lower,
            range_upper,
            subgaussian_proxy,
            two_sided_alpha,
            sample_count: 0,
            mean: 0.0,
            centered_sum_squares: 0.0,
            harmonic_sum: 0.0,
        })
    }

    /// Construct a two-sided CS that spends at most one_sided_alpha on
    /// either directional endpoint. This preserves a familywise allocation
    /// expressed as six one-sided claims per action × scenario cell.
    pub fn new_for_one_sided_claim(
        range_lower: f64,
        range_upper: f64,
        one_sided_alpha: f64,
    ) -> Result<Self, VarianceAdaptiveCsError> {
        if !one_sided_alpha.is_finite()
            || one_sided_alpha <= 0.0
            || one_sided_alpha >= 0.5
        {
            return Err(VarianceAdaptiveCsError::InvalidAlpha);
        }
        Self::new(range_lower, range_upper, 2.0 * one_sided_alpha)
    }

    /// Add one observation in its original, pre-registered stream order.
    /// Returns the current time-uniform interval.
    pub fn update(
        &mut self,
        value: f64,
    ) -> Result<VarianceAdaptiveScoreInterval, VarianceAdaptiveCsError> {
        if !value.is_finite() {
            return Err(VarianceAdaptiveCsError::NonFiniteSample);
        }
        if value < self.range_lower || value > self.range_upper {
            return Err(VarianceAdaptiveCsError::SampleOutsideDeclaredRange);
        }

        let next_count = self
            .sample_count
            .checked_add(1)
            .ok_or(VarianceAdaptiveCsError::SampleCountOverflow)?;

        if self.subgaussian_proxy == 0.0 {
            if value != self.range_lower {
                return Err(VarianceAdaptiveCsError::SampleOutsideDeclaredRange);
            }
            self.sample_count = next_count;
            self.mean = self.range_lower;
            self.harmonic_sum += 1.0 / next_count as f64;
            return self.current_interval();
        }

        let count = next_count as f64;
        let delta = value - self.mean;
        let next_mean = self.mean + delta / count;
        let delta_after_update = value - next_mean;
        let next_scatter = self.centered_sum_squares + delta * delta_after_update;
        let next_harmonic = self.harmonic_sum + 1.0 / count;

        if !next_mean.is_finite()
            || !next_scatter.is_finite()
            || next_scatter < 0.0
            || !next_harmonic.is_finite()
        {
            return Err(VarianceAdaptiveCsError::NumericFailure);
        }

        self.sample_count = next_count;
        self.mean = next_mean;
        self.centered_sum_squares = next_scatter;
        self.harmonic_sum = next_harmonic;
        self.current_interval()
    }

    /// Return the current interval. A non-degenerate stream needs one sample.
    pub fn current_interval(
        &self,
    ) -> Result<VarianceAdaptiveScoreInterval, VarianceAdaptiveCsError> {
        if self.subgaussian_proxy == 0.0 {
            return Ok(VarianceAdaptiveScoreInterval {
                sample_count: self.sample_count,
                point_estimate: self.range_lower,
                lower_bound: self.range_lower,
                upper_bound: self.range_upper,
                half_width: 0.0,
            });
        }
        if self.sample_count == 0 {
            return Err(VarianceAdaptiveCsError::NoObservations);
        }

        let n = self.sample_count as f64;
        let s = self.subgaussian_proxy;
        let alpha = self.two_sided_alpha;
        let scatter = self.centered_sum_squares.max(0.0);
        let harmonic = self.harmonic_sum;
        let variance_charge = scatter + s * s * harmonic;
        if !variance_charge.is_finite() || variance_charge <= 0.0 {
            return Err(VarianceAdaptiveCsError::NumericFailure);
        }

        // Equations (3.5)–(3.8): select one member from a calibrated,
        // countable grid of bets whose total allocation is one.
        let g = ((std::f64::consts::E.powf(std::f64::consts::E) + n).ln()).ln()
            + (20.0 / alpha).ln();
        let b = 1.0 / (5.0 * s);
        let lambda_star = b
            .min((2.0 * g / variance_charge).sqrt())
            .min((g / (2.0 * s * s * s * n)).cbrt());
        if !g.is_finite()
            || g <= 0.0
            || !b.is_finite()
            || b <= 0.0
            || !lambda_star.is_finite()
            || lambda_star <= 0.0
        {
            return Err(VarianceAdaptiveCsError::NumericFailure);
        }

        let q = (b / lambda_star).log2();
        if !q.is_finite() || q < 0.0 || q > (u64::MAX as f64) / 2.0 {
            return Err(VarianceAdaptiveCsError::NumericFailure);
        }
        let j = q.floor() as u64;
        let fractional = (q - j as f64).clamp(0.0, 1.0);
        let m_float = ((j as f64 + std::f64::consts::E).ln()).ceil();
        if !m_float.is_finite() || m_float < 1.0 || m_float > (u64::MAX as f64) {
            return Err(VarianceAdaptiveCsError::NumericFailure);
        }
        let m = (m_float as u64).max(1);
        let k = ((m as f64 * fractional).floor() as u64).min(m - 1);
        let lambda = b * 2.0_f64.powf(-(j as f64) - (k as f64 / m as f64));
        let weight = std::f64::consts::LN_2
            * (1.0 / ((j + 2) as f64).ln() - 1.0 / ((j + 3) as f64).ln());
        let grid_mass = weight / m as f64;
        if !lambda.is_finite()
            || lambda <= 0.0
            || !grid_mass.is_finite()
            || grid_mass <= 0.0
        {
            return Err(VarianceAdaptiveCsError::NumericFailure);
        }

        let grid_log_threshold = (10.0 / (alpha * grid_mass)).ln();
        let coarse_log_threshold = (10.0 / alpha).ln();
        let radius_for = |bet: f64, log_threshold: f64| {
            let first = log_threshold / (n * bet);
            let second = bet * variance_charge / (2.0 * n);
            let third = ((n - harmonic).max(0.0) / (n * bet))
                * (s * s * s * bet * bet * bet).ln_1p();
            first + second + third
        };

        let mut radius = radius_for(lambda, grid_log_threshold);
        for coarse_index in 0..4 {
            let coarse_bet = b * 2.0_f64.powi(-(coarse_index as i32));
            radius = radius.min(radius_for(coarse_bet, coarse_log_threshold));
        }
        if !radius.is_finite() || radius < 0.0 {
            return Err(VarianceAdaptiveCsError::NumericFailure);
        }

        let lower_bound = (self.mean - radius).max(self.range_lower);
        let upper_bound = (self.mean + radius).min(self.range_upper);
        if !lower_bound.is_finite()
            || !upper_bound.is_finite()
            || lower_bound > upper_bound
        {
            return Err(VarianceAdaptiveCsError::NumericFailure);
        }

        Ok(VarianceAdaptiveScoreInterval {
            sample_count: self.sample_count,
            point_estimate: self.mean,
            lower_bound,
            upper_bound,
            half_width: radius,
        })
    }

    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    pub fn point_estimate(&self) -> Option<f64> {
        (self.sample_count > 0 || self.subgaussian_proxy == 0.0).then_some(self.mean)
    }

    pub fn two_sided_alpha(&self) -> f64 {
        self.two_sided_alpha
    }

    pub fn subgaussian_proxy(&self) -> f64 {
        self.subgaussian_proxy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(
        values: &[f64],
        lower: f64,
        upper: f64,
        delta: f64,
    ) -> VarianceAdaptiveScoreInterval {
        let mut cs =
            VarianceAdaptiveSubGaussianCs::new_for_one_sided_claim(lower, upper, delta).unwrap();
        let mut interval = None;
        for value in values {
            interval = Some(cs.update(*value).unwrap());
        }
        interval.unwrap()
    }

    #[test]
    fn one_sided_claim_constructor_doubles_alpha_and_preserves_domain() {
        let cs = VarianceAdaptiveSubGaussianCs::new_for_one_sided_claim(0.0, 1.0, 0.01).unwrap();
        assert_eq!(cs.two_sided_alpha(), 0.02);
        assert_eq!(cs.subgaussian_proxy(), 0.5);
        assert_eq!(
            VarianceAdaptiveSubGaussianCs::new_for_one_sided_claim(0.0, 1.0, 0.5),
            Err(VarianceAdaptiveCsError::InvalidAlpha)
        );
    }

    #[test]
    fn zero_width_range_is_a_deterministic_singleton() {
        let mut cs =
            VarianceAdaptiveSubGaussianCs::new_for_one_sided_claim(3.0, 3.0, 0.01).unwrap();
        let before = cs.current_interval().unwrap();
        assert_eq!(before.lower_bound, 3.0);
        assert_eq!(before.upper_bound, 3.0);
        let after = cs.update(3.0).unwrap();
        assert_eq!(after.lower_bound, 3.0);
        assert_eq!(after.upper_bound, 3.0);
        assert_eq!(after.sample_count, 1);
        assert_eq!(
            cs.update(3.1),
            Err(VarianceAdaptiveCsError::SampleOutsideDeclaredRange)
        );
    }

    #[test]
    fn rejects_invalid_ranges_alpha_and_observations() {
        assert_eq!(
            VarianceAdaptiveSubGaussianCs::new(f64::NAN, 1.0, 0.05),
            Err(VarianceAdaptiveCsError::InvalidScoreRange)
        );
        assert_eq!(
            VarianceAdaptiveSubGaussianCs::new(2.0, 1.0, 0.05),
            Err(VarianceAdaptiveCsError::InvalidScoreRange)
        );
        assert_eq!(
            VarianceAdaptiveSubGaussianCs::new(0.0, f64::MAX, 0.05),
            Err(VarianceAdaptiveCsError::InvalidScoreRange)
        );
        assert_eq!(
            VarianceAdaptiveSubGaussianCs::new(0.0, 1.0, 1.0),
            Err(VarianceAdaptiveCsError::InvalidAlpha)
        );
        let mut cs =
            VarianceAdaptiveSubGaussianCs::new_for_one_sided_claim(0.0, 1.0, 0.01).unwrap();
        assert_eq!(
            cs.update(f64::NAN),
            Err(VarianceAdaptiveCsError::NonFiniteSample)
        );
        assert_eq!(
            cs.update(1.1),
            Err(VarianceAdaptiveCsError::SampleOutsideDeclaredRange)
        );
        assert_eq!(
            cs.current_interval(),
            Err(VarianceAdaptiveCsError::NoObservations)
        );
    }

    #[test]
    fn interval_matches_direct_two_pass_equation_3_9_calculation() {
        let samples = [0.1, 0.2, 0.15, 0.12, 0.18, 0.11, 0.13, 0.17, 0.16, 0.14];
        let interval = stream(&samples, 0.0, 1.0, 0.01);
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let scatter = samples.iter().map(|value| (value - mean).powi(2)).sum::<f64>();
        let harmonic = (1..=samples.len()).map(|index| 1.0 / index as f64).sum::<f64>();
        let s: f64 = 0.5;
        let alpha = 0.02;
        let charge = scatter + s * s * harmonic;
        let g = ((std::f64::consts::E.powf(std::f64::consts::E) + n).ln()).ln()
            + (20.0 / alpha).ln();
        let b = 1.0 / (5.0 * s);
        let lambda_star = b
            .min((2.0 * g / charge).sqrt())
            .min((g / (2.0 * s.powi(3) * n)).cbrt());
        let q = (b / lambda_star).log2();
        let j = q.floor() as u64;
        let m = ((j as f64 + std::f64::consts::E).ln()).ceil() as u64;
        let k = ((m as f64 * (q - j as f64)).floor() as u64).min(m - 1);
        let lambda = b * 2.0_f64.powf(-(j as f64) - k as f64 / m as f64);
        let weight = std::f64::consts::LN_2
            * (1.0 / ((j + 2) as f64).ln() - 1.0 / ((j + 3) as f64).ln());
        let grid_mass = weight / m as f64;
        let radius_for = |bet: f64, log_threshold: f64| {
            log_threshold / (n * bet)
                + bet * charge / (2.0 * n)
                + (n - harmonic) / (n * bet) * (s.powi(3) * bet.powi(3)).ln_1p()
        };
        let mut expected = radius_for(lambda, (10.0 / (alpha * grid_mass)).ln());
        for ell in 0..4 {
            expected = expected.min(radius_for(
                b * 2.0_f64.powi(-ell),
                (10.0 / alpha).ln(),
            ));
        }
        assert!((interval.point_estimate - mean).abs() < 1.0e-14);
        assert!((interval.half_width - expected).abs() < 1.0e-12);
    }

    #[test]
    fn stream_is_deterministic_and_welford_scatter_matches_two_pass() {
        let samples = [0.2, 0.7, 0.1, 0.6, 0.3, 0.8, 0.4, 0.9, 0.5];
        let mut left =
            VarianceAdaptiveSubGaussianCs::new_for_one_sided_claim(0.0, 1.0, 0.005).unwrap();
        let mut right = left.clone();
        for value in samples {
            assert_eq!(left.update(value), right.update(value));
        }
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let scatter = samples
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>();
        assert!((left.centered_sum_squares - scatter).abs() < 1.0e-14);
        assert_eq!(left.sample_count(), samples.len() as u64);
    }

    #[test]
    fn interval_is_affine_equivariant_under_positive_score_rescaling() {
        let samples = [0.1, 0.2, 0.15, 0.12, 0.18, 0.11, 0.13, 0.17, 0.16, 0.14];
        let original = stream(&samples, 0.0, 1.0, 0.01);
        let scaled_samples: Vec<f64> = samples.iter().map(|value| 5.0 + 20.0 * value).collect();
        let scaled = stream(&scaled_samples, 5.0, 25.0, 0.01);
        assert!((scaled.point_estimate - (5.0 + 20.0 * original.point_estimate)).abs() < 1.0e-11);
        assert!((scaled.lower_bound - (5.0 + 20.0 * original.lower_bound)).abs() < 1.0e-10);
        assert!((scaled.upper_bound - (5.0 + 20.0 * original.upper_bound)).abs() < 1.0e-10);
    }

    #[test]
    fn smaller_alpha_never_narrows_the_reported_interval() {
        let mut state = 0x5eed_c5a1_u64;
        let mut samples = Vec::with_capacity(512);
        for _ in 0..512 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            samples.push((state >> 11) as f64 / ((1_u64 << 53) as f64));
        }
        let permissive = stream(&samples, 0.0, 1.0, 0.02);
        let conservative = stream(&samples, 0.0, 1.0, 0.001);
        assert!(conservative.lower_bound <= permissive.lower_bound);
        assert!(conservative.upper_bound >= permissive.upper_bound);
    }

    #[test]
    fn deterministic_campaign_prefixes_always_produce_valid_domain_intervals() {
        let mut cs =
            VarianceAdaptiveSubGaussianCs::new_for_one_sided_claim(-2.0, 3.0, 0.0001).unwrap();
        let mut state = 0x5eed_c5a1_u64;
        for index in 0..4096_u64 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = (state >> 11) as f64 / ((1_u64 << 53) as f64);
            let value = -2.0 + 5.0 * unit;
            let interval = cs.update(value).unwrap();
            assert!(interval.lower_bound.is_finite());
            assert!(interval.upper_bound.is_finite());
            assert!(interval.lower_bound <= interval.point_estimate);
            assert!(interval.point_estimate <= interval.upper_bound);
            assert!(interval.lower_bound >= -2.0);
            assert!(interval.upper_bound <= 3.0);
            assert_eq!(interval.sample_count, index + 1);
        }
    }
}
