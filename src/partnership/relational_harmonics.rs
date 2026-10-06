// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Research-only relational dynamics measurement.
//!
//! This module does NOT control cognition, partnership state, relational_psi,
//! or response generation. It provides a multidimensional profile of observed
//! paired time-series so relational hypotheses can be tested without collapsing
//! them into a single relationship score.
//!
//! Semantic boundaries:
//! - similarity is not mutual information;
//! - correlation is not causality;
//! - synchrony is not relationship quality;
//! - a derived profile is not evidence of consciousness;
//! - harmonic structure here means temporal spectral structure, not a physical
//!   resonance field.

use std::f64::consts::PI;

/// Evidence status attached to a relational measurement channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceStatus {
    Measured,
    Proxy,
    InsufficientData,
    NotApplicable,
}

/// One normalized observation of two interacting agents.
///
/// Directional fields are predictive-coupling proxies. They must not be
/// interpreted as causal influence without a separate estimator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelationalSample {
    pub time: f64,
    pub alignment: f64,
    pub a_to_b: f64,
    pub b_to_a: f64,
    pub turn_taking: f64,
}

impl RelationalSample {
    pub fn new(
        time: f64,
        alignment: f64,
        a_to_b: f64,
        b_to_a: f64,
        turn_taking: f64,
    ) -> Result<Self, RelationalHarmonicError> {
        if !time.is_finite() {
            return Err(RelationalHarmonicError::NonFiniteTime);
        }

        for (name, value) in [
            ("alignment", alignment),
            ("a_to_b", a_to_b),
            ("b_to_a", b_to_a),
            ("turn_taking", turn_taking),
        ] {
            if !value.is_finite() {
                return Err(RelationalHarmonicError::NonFiniteValue(name));
            }
            if !(0.0..=1.0).contains(&value) {
                return Err(RelationalHarmonicError::OutOfRange(name, value));
            }
        }

        Ok(Self {
            time,
            alignment,
            a_to_b,
            b_to_a,
            turn_taking,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RelationalHarmonicError {
    NonFiniteTime,
    NonFiniteValue(&'static str),
    OutOfRange(&'static str, f64),
    InsufficientSamples(usize),
    InvalidSurrogateCount,
    NonMonotonicTime,
    NonUniformSampling,
}

impl std::fmt::Display for RelationalHarmonicError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteTime => write!(f, "relational sample time must be finite"),
            Self::NonFiniteValue(name) => write!(f, "{name} must be finite"),
            Self::OutOfRange(name, value) => {
                write!(f, "{name}={value} is outside [0, 1]")
            }
            Self::InsufficientSamples(n) => {
                write!(f, "at least 2 samples are required; got {n}")
            }
            Self::InvalidSurrogateCount => {
                write!(f, "surrogate_count must be greater than zero")
            }
            Self::NonMonotonicTime => {
                write!(f, "relational sample times must increase strictly")
            }
            Self::NonUniformSampling => {
                write!(f, "temporal information-flow estimation requires uniform sampling")
            }
        }
    }
}

impl std::error::Error for RelationalHarmonicError {}

/// Literal spectral summary of a uniformly sampled relational observable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarmonicSummary {
    pub status: HarmonicStatus,
    pub sample_count: usize,
    pub sample_interval: Option<f64>,
    pub dominant_bin: Option<usize>,
    pub spectral_concentration: f64,
    pub spectral_entropy: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarmonicStatus {
    Computed,
    InsufficientSamples,
    NonUniformSampling,
    NoVariation,
}

impl HarmonicSummary {
    fn unavailable(status: HarmonicStatus, sample_count: usize) -> Self {
        Self {
            status,
            sample_count,
            sample_interval: None,
            dominant_bin: None,
            spectral_concentration: 0.0,
            spectral_entropy: 0.0,
        }
    }
}

/// Multidimensional relational dynamics profile.
///
/// There is intentionally no aggregate consciousness or relationship score.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationalHarmonicProfile {
    pub sample_count: usize,
    pub alignment: f64,
    pub persistence: f64,
    pub directional_coupling: f64,
    pub directional_asymmetry: f64,
    pub reciprocity: f64,
    pub turn_taking: f64,
    pub harmonic: HarmonicSummary,

    pub alignment_status: EvidenceStatus,
    pub persistence_status: EvidenceStatus,
    pub turn_taking_status: EvidenceStatus,
    pub directionality_status: EvidenceStatus,
}

impl RelationalHarmonicProfile {
    pub fn compute(samples: &[RelationalSample]) -> Result<Self, RelationalHarmonicError> {
        if samples.len() < 2 {
            return Err(RelationalHarmonicError::InsufficientSamples(samples.len()));
        }

        for pair in samples.windows(2) {
            if pair[1].time <= pair[0].time {
                return Err(RelationalHarmonicError::NonMonotonicTime);
            }
        }

        let n = samples.len() as f64;

        let alignment = samples.iter().map(|s| s.alignment).sum::<f64>() / n;
        let turn_taking = samples.iter().map(|s| s.turn_taking).sum::<f64>() / n;

        let directional_coupling = samples
            .iter()
            .map(|s| 0.5 * (s.a_to_b + s.b_to_a))
            .sum::<f64>()
            / n;

        let directional_asymmetry = samples
            .iter()
            .map(|s| (s.a_to_b - s.b_to_a).abs())
            .sum::<f64>()
            / n;

        // Harmonic mean preserves the meaning of reciprocity:
        // both directions must be present and strong.
        let reciprocity = samples
            .iter()
            .map(|s| {
                let sum = s.a_to_b + s.b_to_a;
                if sum <= f64::EPSILON {
                    0.0
                } else {
                    (2.0 * s.a_to_b * s.b_to_a / sum).clamp(0.0, 1.0)
                }
            })
            .sum::<f64>()
            / n;

        let persistence = (1.0
            - samples
                .windows(2)
                .map(|w| (w[1].alignment - w[0].alignment).abs())
                .sum::<f64>()
                / (n - 1.0))
        .clamp(0.0, 1.0);

        Ok(Self {
            sample_count: samples.len(),
            alignment,
            persistence,
            directional_coupling: directional_coupling.clamp(0.0, 1.0),
            directional_asymmetry: directional_asymmetry.clamp(0.0, 1.0),
            reciprocity,
            turn_taking,
            harmonic: harmonic_summary(samples),
            alignment_status: EvidenceStatus::Measured,
            persistence_status: EvidenceStatus::Measured,
            turn_taking_status: EvidenceStatus::Measured,
            directionality_status: EvidenceStatus::Proxy,
        })
    }
}

fn harmonic_summary(samples: &[RelationalSample]) -> HarmonicSummary {
    let n = samples.len();
    if n < 4 {
        return HarmonicSummary::unavailable(HarmonicStatus::InsufficientSamples, n);
    }

    let dt0 = samples[1].time - samples[0].time;
    let tolerance = 1e-6 * dt0.abs().max(1.0);

    let uniform = samples
        .windows(2)
        .all(|w| ((w[1].time - w[0].time) - dt0).abs() <= tolerance);

    if !uniform {
        return HarmonicSummary::unavailable(HarmonicStatus::NonUniformSampling, n);
    }

    let mean = samples.iter().map(|s| s.alignment).sum::<f64>() / n as f64;
    let centered: Vec<f64> = samples.iter().map(|s| s.alignment - mean).collect();

    let positive_bins = n / 2;
    let mut powers = Vec::with_capacity(positive_bins);

    for k in 1..=positive_bins {
        let mut re = 0.0;
        let mut im = 0.0;

        for (t, value) in centered.iter().enumerate() {
            let theta = 2.0 * PI * (k * t) as f64 / n as f64;
            re += value * theta.cos();
            im -= value * theta.sin();
        }

        powers.push(re * re + im * im);
    }

    let total_power = powers.iter().sum::<f64>();
    if total_power <= 1e-20 {
        return HarmonicSummary {
            status: HarmonicStatus::NoVariation,
            sample_count: n,
            sample_interval: Some(dt0),
            dominant_bin: None,
            spectral_concentration: 0.0,
            spectral_entropy: 0.0,
        };
    }

    let dominant_offset = powers
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index);

    let normalized: Vec<f64> = powers.iter().map(|p| p / total_power).collect();

    let support = normalized.len() as f64;
    let entropy_raw = normalized
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| -p * p.ln())
        .sum::<f64>();

    let spectral_entropy = if support > 1.0 {
        (entropy_raw / support.ln()).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let spectral_concentration = normalized
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .clamp(0.0, 1.0);

    HarmonicSummary {
        status: HarmonicStatus::Computed,
        sample_count: n,
        sample_interval: Some(dt0),
        dominant_bin: dominant_offset.map(|index| index + 1),
        spectral_concentration,
        spectral_entropy,
    }
}

/// Scalar agent observations used by the existing Symthaea transfer-entropy
/// estimator.
///
/// These observations are deliberately separate from RelationalSample:
/// RH-001's alignment/coupling channels must not be fed back into themselves
/// as the purported source and target signals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelationalSignalSample {
    /// Monotone observation time in caller-defined units.
    pub time: f64,
    /// Scalar state summary for agent A in caller-defined units.
    pub agent_a: f64,
    /// Scalar state summary for agent B in caller-defined units.
    pub agent_b: f64,
}

impl RelationalSignalSample {
    pub fn new(
        time: f64,
        agent_a: f64,
        agent_b: f64,
    ) -> Result<Self, RelationalHarmonicError> {
        if !time.is_finite() {
            return Err(RelationalHarmonicError::NonFiniteTime);
        }

        for (name, value) in [("agent_a", agent_a), ("agent_b", agent_b)] {
            if !value.is_finite() {
                return Err(RelationalHarmonicError::NonFiniteValue(name));
            }
        }

        Ok(Self {
            time,
            agent_a,
            agent_b,
        })
    }
}

/// Directional information-flow estimate using the repository's existing
/// transfer-entropy estimator.
///
/// These values are intentionally labeled Proxy: transfer entropy is an
/// information-theoretic directional measure, but the current estimator is a
/// finite-sample histogram estimator without an attached significance test.
/// It must not be called mechanistic causality.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalInformationFlow {
    pub samples: usize,
    pub te_a_to_b: f64,
    pub te_b_to_a: f64,
    pub net_a_to_b: f64,
    pub status: EvidenceStatus,
}

impl DirectionalInformationFlow {
    /// Estimate directional information flow from independent agent signals.
    ///
    /// min_samples is an explicit caller-side stability floor. The current
    /// estimator itself accepts much smaller windows, which is useful for
    /// experimentation but not sufficient for a stable qualification claim.
    pub fn compute(
        samples: &[RelationalSignalSample],
        min_samples: usize,
    ) -> Result<Self, RelationalHarmonicError> {
        if samples.len() < min_samples.max(2) {
            return Err(RelationalHarmonicError::InsufficientSamples(samples.len()));
        }

        validate_uniform_sampling(samples)?;

        let (te_a_to_b, te_b_to_a) = transfer_entropy_pair(samples);

        Ok(Self {
            samples: samples.len(),
            te_a_to_b,
            te_b_to_a,
            net_a_to_b: te_a_to_b - te_b_to_a,
            status: EvidenceStatus::Proxy,
        })
    }
}



/// Deterministic pseudo-partner calibration for the existing transfer-entropy
/// wrapper. The output remains a proxy because the surrogate family is fixed
/// and finite rather than a complete inferential protocol.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalInformationFlowSurrogateSummary {
    pub samples: usize,
    pub surrogate_count: usize,
    pub observed_te_a_to_b: f64,
    pub observed_te_b_to_a: f64,
    pub max_surrogate_te_a_to_b: f64,
    pub max_surrogate_te_b_to_a: f64,
    pub exceedance_a_to_b: usize,
    pub exceedance_b_to_a: usize,
    pub exceedance_fraction_a_to_b: f64,
    pub exceedance_fraction_b_to_a: f64,
    pub status: EvidenceStatus,
}

impl DirectionalInformationFlowSurrogateSummary {
    pub fn compute(
        samples: &[RelationalSignalSample],
        min_samples: usize,
        surrogate_count: usize,
    ) -> Result<Self, RelationalHarmonicError> {
        if samples.len() < min_samples.max(2) {
            return Err(RelationalHarmonicError::InsufficientSamples(samples.len()));
        }

        if surrogate_count == 0 {
            return Err(RelationalHarmonicError::InvalidSurrogateCount);
        }

        validate_uniform_sampling(samples)?;

        let (observed_a_to_b, observed_b_to_a) = transfer_entropy_pair(samples);
        let count = surrogate_count.min(samples.len() - 1);

        let mut max_a_to_b = 0.0_f64;
        let mut max_b_to_a = 0.0_f64;
        let mut exceedance_a_to_b = 0usize;
        let mut exceedance_b_to_a = 0usize;

        for index in 0..count {
            let shift = 1 + (index * (samples.len() - 1) / count);
            let shifted = samples
                .iter()
                .enumerate()
                .map(|(i, sample)| RelationalSignalSample {
                    time: sample.time,
                    agent_a: sample.agent_a,
                    agent_b: samples[(i + shift) % samples.len()].agent_b,
                })
                .collect::<Vec<_>>();

            let (te_a_to_b, te_b_to_a) = transfer_entropy_pair(&shifted);
            max_a_to_b = max_a_to_b.max(te_a_to_b);
            max_b_to_a = max_b_to_a.max(te_b_to_a);

            if te_a_to_b >= observed_a_to_b - 1e-12 {
                exceedance_a_to_b += 1;
            }
            if te_b_to_a >= observed_b_to_a - 1e-12 {
                exceedance_b_to_a += 1;
            }
        }

        Ok(Self {
            samples: samples.len(),
            surrogate_count: count,
            observed_te_a_to_b: observed_a_to_b,
            observed_te_b_to_a: observed_b_to_a,
            max_surrogate_te_a_to_b: max_a_to_b,
            max_surrogate_te_b_to_a: max_b_to_a,
            exceedance_a_to_b,
            exceedance_b_to_a,
            exceedance_fraction_a_to_b: exceedance_a_to_b as f64 / count as f64,
            exceedance_fraction_b_to_a: exceedance_b_to_a as f64 / count as f64,
            status: EvidenceStatus::Proxy,
        })
    }
}

fn validate_uniform_sampling(
    samples: &[RelationalSignalSample],
) -> Result<(), RelationalHarmonicError> {
    for pair in samples.windows(2) {
        if pair[1].time <= pair[0].time {
            return Err(RelationalHarmonicError::NonMonotonicTime);
        }
    }

    let dt0 = samples[1].time - samples[0].time;
    let tolerance = 1e-6 * dt0.abs().max(1.0);
    if !samples
        .windows(2)
        .all(|w| ((w[1].time - w[0].time) - dt0).abs() <= tolerance)
    {
        return Err(RelationalHarmonicError::NonUniformSampling);
    }

    Ok(())
}

fn transfer_entropy_pair(samples: &[RelationalSignalSample]) -> (f64, f64) {
    let config = crate::hdc::information_theory::InformationTheoryConfig::default();
    let mut estimator =
        crate::hdc::information_theory::TransferEntropyEstimator::new(config, samples.len());

    for sample in samples {
        estimator.observe_scalars(sample.agent_a, sample.agent_b);
    }

    (
        estimator.transfer_entropy_x_to_y().unwrap_or(0.0),
        estimator.transfer_entropy_y_to_x().unwrap_or(0.0),
    )
}

/// A paired signal plus an explicitly observed common driver.
///
/// The controlled statistic removes only the supplied driver and is intended
/// to detect synchrony that can be explained by a shared external signal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommonDriverSignalSample {
    pub time: f64,
    pub common_driver: f64,
    pub agent_a: f64,
    pub agent_b: f64,
}

impl CommonDriverSignalSample {
    pub fn new(
        time: f64,
        common_driver: f64,
        agent_a: f64,
        agent_b: f64,
    ) -> Result<Self, RelationalHarmonicError> {
        if !time.is_finite() {
            return Err(RelationalHarmonicError::NonFiniteTime);
        }

        for (name, value) in [
            ("common_driver", common_driver),
            ("agent_a", agent_a),
            ("agent_b", agent_b),
        ] {
            if !value.is_finite() {
                return Err(RelationalHarmonicError::NonFiniteValue(name));
            }
        }

        Ok(Self {
            time,
            common_driver,
            agent_a,
            agent_b,
        })
    }
}

/// Zero-lag association before and after a linear control for one common driver.
///
/// This is a narrow linear control rather than a general confounding solution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommonDriverControlSummary {
    pub samples: usize,
    pub raw_zero_lag_correlation: f64,
    pub controlled_zero_lag_correlation: f64,
    pub absolute_correlation_reduction: f64,
    pub status: EvidenceStatus,
}

impl CommonDriverControlSummary {
    pub fn compute(
        samples: &[CommonDriverSignalSample],
    ) -> Result<Self, RelationalHarmonicError> {
        if samples.len() < 16 {
            return Err(RelationalHarmonicError::InsufficientSamples(samples.len()));
        }

        for pair in samples.windows(2) {
            if pair[1].time <= pair[0].time {
                return Err(RelationalHarmonicError::NonMonotonicTime);
            }
        }

        let dt0 = samples[1].time - samples[0].time;
        let tolerance = 1e-6 * dt0.abs().max(1.0);
        if !samples
            .windows(2)
            .all(|w| ((w[1].time - w[0].time) - dt0).abs() <= tolerance)
        {
            return Err(RelationalHarmonicError::NonUniformSampling);
        }

        let a = samples.iter().map(|s| s.agent_a).collect::<Vec<_>>();
        let b = samples.iter().map(|s| s.agent_b).collect::<Vec<_>>();
        let z = samples.iter().map(|s| s.common_driver).collect::<Vec<_>>();

        let raw = pearson_correlation(&a, &b);
        let controlled = partial_correlation_against_shared_driver(&a, &b, &z);

        Ok(Self {
            samples: samples.len(),
            raw_zero_lag_correlation: raw,
            controlled_zero_lag_correlation: controlled,
            absolute_correlation_reduction: raw.abs() - controlled.abs(),
            status: EvidenceStatus::Measured,
        })
    }
}

fn partial_correlation_against_shared_driver(
    a: &[f64],
    b: &[f64],
    driver: &[f64],
) -> f64 {
    let n = a.len().min(b.len()).min(driver.len());
    if n < 3 {
        return 0.0;
    }

    let residual_a = residualize_on_single_driver(&a[..n], &driver[..n]);
    let residual_b = residualize_on_single_driver(&b[..n], &driver[..n]);

    pearson_correlation(&residual_a, &residual_b)
}

fn residualize_on_single_driver(values: &[f64], driver: &[f64]) -> Vec<f64> {
    let n = values.len().min(driver.len());
    if n < 2 {
        return vec![0.0; n];
    }

    let mean_value = values[..n].iter().sum::<f64>() / n as f64;
    let mean_driver = driver[..n].iter().sum::<f64>() / n as f64;

    let mut covariance = 0.0;
    let mut variance_driver = 0.0;

    for i in 0..n {
        let dv = driver[i] - mean_driver;
        covariance += dv * (values[i] - mean_value);
        variance_driver += dv * dv;
    }

    let slope = if variance_driver <= 1e-20 {
        0.0
    } else {
        covariance / variance_driver
    };

    (0..n)
        .map(|i| values[i] - (mean_value + slope * (driver[i] - mean_driver)))
        .collect()
}


/// Deterministic pseudo-partner calibration for lag correlation.
///
/// Each surrogate circularly shifts B relative to A. The result is an
/// empirical null comparison, not a formal p-value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PartnerShuffleSurrogateSummary {
    pub samples: usize,
    pub surrogate_count: usize,
    pub observed_best_absolute_correlation: f64,
    pub observed_best_lag: Option<usize>,
    pub max_surrogate_absolute_correlation: f64,
    pub exceedance_count: usize,
    pub exceedance_fraction: f64,
    pub status: EvidenceStatus,
}

impl PartnerShuffleSurrogateSummary {
    pub fn compute(
        samples: &[RelationalSignalSample],
        max_lag: usize,
        surrogate_count: usize,
    ) -> Result<Self, RelationalHarmonicError> {
        if samples.len() < 16 {
            return Err(RelationalHarmonicError::InsufficientSamples(samples.len()));
        }

        if surrogate_count == 0 {
            return Err(RelationalHarmonicError::InsufficientSamples(0));
        }

        let observed = LagCorrelationSummary::compute(samples, max_lag)?;
        let count = surrogate_count.min(samples.len() - 1);

        let mut exceedance_count = 0usize;
        let mut max_surrogate = 0.0_f64;

        for index in 0..count {
            let shift = 1 + (index * (samples.len() - 1) / count);
            let shifted = samples
                .iter()
                .enumerate()
                .map(|(i, sample)| RelationalSignalSample {
                    time: sample.time,
                    agent_a: sample.agent_a,
                    agent_b: samples[(i + shift) % samples.len()].agent_b,
                })
                .collect::<Vec<_>>();

            let surrogate = LagCorrelationSummary::compute(&shifted, max_lag)?;
            max_surrogate = max_surrogate.max(surrogate.best_absolute_correlation);

            if surrogate.best_absolute_correlation
                >= observed.best_absolute_correlation - 1e-12
            {
                exceedance_count += 1;
            }
        }

        Ok(Self {
            samples: samples.len(),
            surrogate_count: count,
            observed_best_absolute_correlation: observed.best_absolute_correlation,
            observed_best_lag: observed.best_lag,
            max_surrogate_absolute_correlation: max_surrogate,
            exceedance_count,
            exceedance_fraction: exceedance_count as f64 / count as f64,
            status: EvidenceStatus::Proxy,
        })
    }
}

/// Lagged correlation structure computed directly from the independent
/// agent signals.
///
/// A positive lag means B follows A by that many samples:
/// corr(A[t-lag], B[t]). This is temporal association, not causality.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LagCorrelationSummary {
    pub status: EvidenceStatus,
    pub best_lag: Option<usize>,
    pub best_signed_correlation: f64,
    pub best_absolute_correlation: f64,
    pub zero_lag_correlation: Option<f64>,
}

impl LagCorrelationSummary {
    pub fn compute(
        samples: &[RelationalSignalSample],
        max_lag: usize,
    ) -> Result<Self, RelationalHarmonicError> {
        if samples.len() < 16 {
            return Err(RelationalHarmonicError::InsufficientSamples(samples.len()));
        }

        for pair in samples.windows(2) {
            if pair[1].time <= pair[0].time {
                return Err(RelationalHarmonicError::NonMonotonicTime);
            }
        }

        let dt0 = samples[1].time - samples[0].time;
        let tolerance = 1e-6 * dt0.abs().max(1.0);
        if !samples
            .windows(2)
            .all(|w| ((w[1].time - w[0].time) - dt0).abs() <= tolerance)
        {
            return Err(RelationalHarmonicError::NonUniformSampling);
        }

        let max_lag = max_lag.min(samples.len() - 2);
        let mut best_lag = None;
        let mut best_signed = 0.0;
        let mut best_abs = -1.0;

        for lag in 0..=max_lag {
            let start = lag;
            let n = samples.len() - lag;
            if n < 16 {
                continue;
            }

            let a = samples[..n].iter().map(|s| s.agent_a).collect::<Vec<_>>();
            let b = samples[start..].iter().map(|s| s.agent_b).collect::<Vec<_>>();
            let corr = pearson_correlation(&a, &b);
            let magnitude = corr.abs();
            if magnitude > best_abs {
                best_abs = magnitude;
                best_signed = corr;
                best_lag = Some(lag);
            }
        }

        let zero_lag = pearson_correlation(
            &samples.iter().map(|s| s.agent_a).collect::<Vec<_>>(),
            &samples.iter().map(|s| s.agent_b).collect::<Vec<_>>(),
        );

        Ok(Self {
            status: EvidenceStatus::Measured,
            best_lag,
            best_signed_correlation: best_signed.clamp(-1.0, 1.0),
            best_absolute_correlation: best_abs.clamp(0.0, 1.0),
            zero_lag_correlation: Some(zero_lag),
        })
    }
}

fn pearson_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }

    let mean_a = a[..n].iter().sum::<f64>() / n as f64;
    let mean_b = b[..n].iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for i in 0..n {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        num += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom <= 1e-20 {
        0.0
    } else {
        (num / denom).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(i: usize, alignment: f64, a_to_b: f64, b_to_a: f64) -> RelationalSample {
        RelationalSample::new(i as f64, alignment, a_to_b, b_to_a, 0.5).unwrap()
    }

    #[test]
    fn rejects_invalid_observables() {
        assert!(matches!(
            RelationalSample::new(0.0, 1.1, 0.0, 0.0, 0.0),
            Err(RelationalHarmonicError::OutOfRange("alignment", _))
        ));

        assert!(matches!(
            RelationalSample::new(f64::NAN, 0.5, 0.5, 0.5, 0.5),
            Err(RelationalHarmonicError::NonFiniteTime)
        ));
    }

    #[test]
    fn rejects_non_monotonic_time() {
        let samples = [
            sample(0, 0.5, 0.5, 0.5),
            RelationalSample::new(0.0, 0.5, 0.5, 0.5, 0.5).unwrap(),
        ];

        assert_eq!(
            RelationalHarmonicProfile::compute(&samples),
            Err(RelationalHarmonicError::NonMonotonicTime)
        );
    }

    #[test]
    fn distinguishes_one_way_from_reciprocal_coupling() {
        let one_way = (0..4)
            .map(|i| sample(i, 0.8, 0.9, 0.0))
            .collect::<Vec<_>>();

        let reciprocal = (0..4)
            .map(|i| sample(i, 0.8, 0.9, 0.9))
            .collect::<Vec<_>>();

        let a = RelationalHarmonicProfile::compute(&one_way).unwrap();
        let b = RelationalHarmonicProfile::compute(&reciprocal).unwrap();

        assert_eq!(a.reciprocity, 0.0);
        assert!(b.reciprocity > a.reciprocity);
        assert_eq!(b.directional_asymmetry, 0.0);
        assert_eq!(a.directionality_status, EvidenceStatus::Proxy);
    }

    #[test]
    fn zero_bidirectional_coupling_is_not_reciprocal() {
        let samples = (0..8)
            .map(|i| sample(i, 0.5, 0.0, 0.0))
            .collect::<Vec<_>>();

        let profile = RelationalHarmonicProfile::compute(&samples).unwrap();

        assert_eq!(profile.directional_coupling, 0.0);
        assert_eq!(profile.reciprocity, 0.0);
        assert_eq!(profile.directional_asymmetry, 0.0);
    }

    #[test]
    fn weak_bidirectional_coupling_remains_low_reciprocity() {
        let samples = (0..8)
            .map(|i| sample(i, 0.5, 0.1, 0.1))
            .collect::<Vec<_>>();

        let profile = RelationalHarmonicProfile::compute(&samples).unwrap();

        assert!((profile.reciprocity - 0.1).abs() < 1e-12);
        assert!(profile.reciprocity < 0.2);
    }

    #[test]
    fn constant_signal_has_no_false_harmonic() {
        let samples = (0..16)
            .map(|i| sample(i, 0.5, 0.5, 0.5))
            .collect::<Vec<_>>();

        let profile = RelationalHarmonicProfile::compute(&samples).unwrap();

        assert_eq!(profile.harmonic.status, HarmonicStatus::NoVariation);
        assert_eq!(profile.harmonic.dominant_bin, None);
        assert_eq!(profile.harmonic.spectral_concentration, 0.0);
    }

    #[test]
    fn dft_recovers_known_harmonic() {
        let samples = (0..32)
            .map(|i| {
                let theta = 2.0 * PI * 3.0 * i as f64 / 32.0;
                sample(i, 0.5 + 0.4 * theta.sin(), 0.5, 0.5)
            })
            .collect::<Vec<_>>();

        let profile = RelationalHarmonicProfile::compute(&samples).unwrap();

        assert_eq!(profile.harmonic.status, HarmonicStatus::Computed);
        assert_eq!(profile.harmonic.dominant_bin, Some(3));
        assert!(profile.harmonic.spectral_concentration > 0.95);
    }

    #[test]
    fn non_uniform_sampling_refuses_literal_dft() {
        let samples = vec![
            sample(0, 0.2, 0.5, 0.5),
            RelationalSample::new(1.0, 0.4, 0.5, 0.5, 0.5).unwrap(),
            RelationalSample::new(2.1, 0.6, 0.5, 0.5, 0.5).unwrap(),
            RelationalSample::new(3.1, 0.8, 0.5, 0.5, 0.5).unwrap(),
            RelationalSample::new(4.25, 0.6, 0.5, 0.5, 0.5).unwrap(),
        ];

        let profile = RelationalHarmonicProfile::compute(&samples).unwrap();

        assert_eq!(
            profile.harmonic.status,
            HarmonicStatus::NonUniformSampling
        );
        assert_eq!(profile.harmonic.dominant_bin, None);
    }

    #[test]
    fn lag_correlation_recovers_known_delay() {
        let source = (0..96)
            .map(|i| {
                let x = ((i as f64) * 0.173).sin() + 0.2 * ((i as f64) * 0.071).cos();
                x
            })
            .collect::<Vec<_>>();

        let samples = (0..96)
            .map(|i| {
                let b = if i >= 3 { source[i - 3] } else { 0.0 };
                RelationalSignalSample::new(i as f64, source[i], b)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let lag = LagCorrelationSummary::compute(&samples, 8).unwrap();

        assert_eq!(lag.status, EvidenceStatus::Measured);
        assert_eq!(lag.best_lag, Some(3));
        assert!(lag.best_absolute_correlation > 0.95);
    }

    #[test]
    fn lag_correlation_marks_temporal_association_not_causality() {
        let samples = (0..64)
            .map(|i| {
                RelationalSignalSample::new(
                    i as f64,
                    (i as f64 * 0.11).sin(),
                    (i as f64 * 0.11).sin(),
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let lag = LagCorrelationSummary::compute(&samples, 4).unwrap();

        assert_eq!(lag.status, EvidenceStatus::Measured);
        assert!(lag.best_absolute_correlation > 0.9);
        assert_eq!(lag.best_lag, Some(0));
    }

    #[test]
    fn directional_information_flow_has_explicit_sample_floor() {
        let samples = (0..8)
            .map(|i| RelationalSignalSample::new(i as f64, i as f64, (i + 1) as f64))
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert!(matches!(
            DirectionalInformationFlow::compute(&samples, 16),
            Err(RelationalHarmonicError::InsufficientSamples(8))
        ));
    }

    #[test]
    fn directional_information_flow_refuses_irregular_sampling() {
        let samples = vec![
            RelationalSignalSample::new(0.0, 0.1, 0.2).unwrap(),
            RelationalSignalSample::new(1.0, 0.2, 0.3).unwrap(),
            RelationalSignalSample::new(2.2, 0.3, 0.4).unwrap(),
            RelationalSignalSample::new(3.2, 0.4, 0.5).unwrap(),
            RelationalSignalSample::new(4.2, 0.5, 0.6).unwrap(),
            RelationalSignalSample::new(5.2, 0.6, 0.7).unwrap(),
        ];

        assert_eq!(
            DirectionalInformationFlow::compute(&samples, 6),
            Err(RelationalHarmonicError::NonUniformSampling)
        );
    }

    #[test]
    fn directional_information_flow_reuses_existing_estimator() {
        let samples = (0..128)
            .map(|i| {
                let a = ((i as f64) * 0.17).sin();
                let b = ((i as f64 - 1.0) * 0.17).sin();
                RelationalSignalSample::new(i as f64, a, b)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let flow = DirectionalInformationFlow::compute(&samples, 100).unwrap();

        assert_eq!(flow.samples, 128);
        assert_eq!(flow.status, EvidenceStatus::Proxy);
        assert!(flow.te_a_to_b.is_finite());
        assert!(flow.te_b_to_a.is_finite());
        assert!(flow.net_a_to_b.is_finite());
    }



    #[test]
    fn transfer_entropy_surrogate_calibration_is_deterministic() {
        let samples = (0..128)
            .map(|i| {
                let a = (i as f64 * 0.173).sin() + 0.05 * (i as f64 * 0.041).cos();
                let b = ((i as f64 - 2.0) * 0.173).sin();
                RelationalSignalSample::new(i as f64, a, b)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let first =
            DirectionalInformationFlowSurrogateSummary::compute(&samples, 100, 10).unwrap();
        let second =
            DirectionalInformationFlowSurrogateSummary::compute(&samples, 100, 10).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.status, EvidenceStatus::Proxy);
        assert_eq!(first.surrogate_count, 10);
        assert!(first.observed_te_a_to_b.is_finite());
        assert!(first.observed_te_b_to_a.is_finite());
        assert!(first.max_surrogate_te_a_to_b.is_finite());
        assert!(first.max_surrogate_te_b_to_a.is_finite());
        assert!((0.0..=1.0).contains(&first.exceedance_fraction_a_to_b));
        assert!((0.0..=1.0).contains(&first.exceedance_fraction_b_to_a));
    }

    #[test]
    fn transfer_entropy_surrogate_calibration_rejects_empty_request() {
        let samples = (0..32)
            .map(|i| {
                RelationalSignalSample::new(
                    i as f64,
                    (i as f64 * 0.1).sin(),
                    (i as f64 * 0.2).cos(),
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(
            DirectionalInformationFlowSurrogateSummary::compute(&samples, 16, 0),
            Err(RelationalHarmonicError::InvalidSurrogateCount)
        );
    }


    #[test]
    fn common_driver_control_collapses_shared_stimulus_correlation() {
        let samples = (0..96)
            .map(|i| {
                let z = (i as f64 * 0.13).sin();
                let a = z + 0.02 * (i as f64 * 0.71).sin();
                let b = z + 0.02 * (i as f64 * 0.37).cos();
                CommonDriverSignalSample::new(i as f64, z, a, b)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let control = CommonDriverControlSummary::compute(&samples).unwrap();

        assert!(control.raw_zero_lag_correlation.abs() > 0.9);
        assert!(control.controlled_zero_lag_correlation.abs() < 0.35);
        assert!(control.absolute_correlation_reduction > 0.55);
        assert_eq!(control.status, EvidenceStatus::Measured);
    }

    #[test]
    fn common_driver_control_requires_uniform_sampling() {
        let samples = (0..16)
            .map(|i| {
                let time = if i == 8 { i as f64 + 0.25 } else { i as f64 };
                CommonDriverSignalSample::new(
                    time,
                    (i as f64 * 0.11).sin(),
                    (i as f64 * 0.17).sin(),
                    (i as f64 * 0.19).sin(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        assert_eq!(
            CommonDriverControlSummary::compute(&samples),
            Err(RelationalHarmonicError::NonUniformSampling)
        );
    }

    #[test]
    fn shuffled_partner_null_is_deterministic_and_reported_as_proxy() {
        let samples = (0..96)
            .map(|i| {
                let a = (i as f64 * 0.173).sin();
                let b = ((i as f64 - 3.0) * 0.173).sin();
                RelationalSignalSample::new(i as f64, a, b)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        let first = PartnerShuffleSurrogateSummary::compute(&samples, 8, 12).unwrap();
        let second = PartnerShuffleSurrogateSummary::compute(&samples, 8, 12).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.status, EvidenceStatus::Proxy);
        assert_eq!(first.surrogate_count, 12);
        assert!(first.observed_best_absolute_correlation > 0.9);
        assert!(first.exceedance_fraction >= 0.0);
        assert!(first.exceedance_fraction <= 1.0);
        assert!(first.max_surrogate_absolute_correlation >= 0.0);
    }

    #[test]
    fn shuffled_partner_calibration_rejects_empty_surrogate_request() {
        let samples = (0..16)
            .map(|i| {
                RelationalSignalSample::new(
                    i as f64,
                    (i as f64 * 0.2).sin(),
                    (i as f64 * 0.2).cos(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        assert_eq!(
            PartnerShuffleSurrogateSummary::compute(&samples, 4, 0),
            Err(RelationalHarmonicError::InsufficientSamples(0))
        );
    }

    #[test]
    fn no_aggregate_score_is_exposed() {
        let samples = (0..8)
            .map(|i| sample(i, 0.4 + 0.01 * i as f64, 0.7, 0.6))
            .collect::<Vec<_>>();

        let profile = RelationalHarmonicProfile::compute(&samples).unwrap();

        assert_eq!(profile.sample_count, 8);
        assert!(profile.alignment > 0.4);
        assert!(profile.persistence > 0.0);
        assert!(profile.turn_taking >= 0.0);
    }
}
