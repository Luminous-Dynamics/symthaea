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
    NonMonotonicTime,
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
            Self::NonMonotonicTime => {
                write!(f, "relational sample times must increase strictly")
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
