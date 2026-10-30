// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Exhaustive deterministic factorial reference engine for synthetic audio interactions.
//!
//! This is a benchmark oracle, not a production renderer. It enumerates every
//! event coalition for small fixtures and computes Möbius/Harsanyi dividends
//! independently in PCM and fixed-window spectral-magnitude representations.
//! Spectral-magnitude non-additivity is intentionally not interpreted as PCM
//! renderer nonlinearity.

use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const FACTORIAL_GROUND_TRUTH_VERSION: &str =
    "mel-mr-eval-020-factorial-ground-truth-v0.1";
const REPORT_DOMAIN: &[u8] = b"luminous-dynamics-mel-mr-eval-020-factorial-reference-v1\0";
const MAX_EVENT_COUNT: usize = 8;
const MAX_INPUT_FRAMES: usize = 65_536;
const MAX_FIR_TAPS: usize = 256;
const MAX_SAMPLE_ABS: f64 = 16.0;
const MAX_OUTPUT_ABS: f64 = 1.0e9;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntheticEventSignal {
    /// Canonical source event index; the event vector must be ordered 0..N-1.
    pub source_event_index: usize,
    /// Aligned mono source-isolated signal in f64 benchmark units.
    pub samples: Vec<f64>,
}

/// Deterministic reference effects. All parameters are explicit and serialized.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SyntheticEffect {
    /// Identity / linear summation: y(t) = sum_i x_i(t).
    LinearMix,
    /// Fixed causal FIR convolution after linear summation.
    LinearFir {
        impulse_response: Vec<f64>,
    },
    /// Pointwise polynomial: y = x + q*x^2 + c*x^3.
    Polynomial {
        quadratic_coefficient: f64,
        cubic_coefficient: f64,
    },
    /// Static hard limiter applied to the shared sum.
    HardClip {
        threshold: f64,
    },
    /// Exact third-order term added only when all three named events are present.
    /// y = sum_i x_i + g*x_a*x_b*x_c.
    ThreeWayProduct {
        event_indices: [usize; 3],
        coefficient: f64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactorialReferenceConfig {
    /// Number of initial frames used for a single Hann-windowed magnitude spectrum.
    pub spectral_window_frames: usize,
}

impl Default for FactorialReferenceConfig {
    fn default() -> Self {
        Self {
            spectral_window_frames: 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntheticCoalitionRender {
    pub coalition_mask: u16,
    pub included_event_indices: Vec<usize>,
    pub pcm_samples: Vec<f64>,
    /// Single-window rFFT magnitudes; stored separately from the PCM output.
    pub spectral_magnitudes: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HarsanyiDividend {
    pub coalition_mask: u16,
    pub event_indices: Vec<usize>,
    pub order: u8,
    /// Samplewise Möbius dividend in waveform space.
    pub pcm_samples: Vec<f64>,
    pub pcm_l2_norm: f64,
    pub pcm_peak_abs: f64,
    /// Möbius dividend after the nonlinear magnitude-spectrum feature transform.
    pub spectral_magnitude_values: Vec<f64>,
    pub spectral_l2_norm: f64,
    pub spectral_peak_abs: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactorialGroundTruthReport {
    pub version: String,
    pub event_count: usize,
    pub coalition_count: usize,
    pub input_frame_count: usize,
    pub output_frame_count: usize,
    pub input_sha256: String,
    pub effect_sha256: String,
    pub config: FactorialReferenceConfig,
    /// Includes the empty coalition (mask zero) and every non-empty coalition.
    pub coalitions: Vec<SyntheticCoalitionRender>,
    /// Includes all non-empty Harsanyi dividends, including singleton main effects.
    pub dividends: Vec<HarsanyiDividend>,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FactorialReferenceError {
    InvalidEventCount,
    NonCanonicalEventIndex,
    InvalidFrameCount,
    UnequalEventFrameCounts,
    NonFiniteInput,
    InputOutOfRange,
    InvalidEffect(&'static str),
    InvalidConfig(&'static str),
    NonFiniteOutput,
    OutputOutOfRange,
    SerializationFailure,
    InvalidReceipt,
    ReplayMismatch,
}

impl std::fmt::Display for FactorialReferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "factorial ground-truth validation failed: {self:?}")
    }
}

impl std::error::Error for FactorialReferenceError {}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_digest<T: Serialize>(value: &T) -> Result<String, FactorialReferenceError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|_| FactorialReferenceError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(REPORT_DOMAIN);
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_effect(
    effect: &SyntheticEffect,
    event_count: usize,
) -> Result<(), FactorialReferenceError> {
    match effect {
        SyntheticEffect::LinearMix => Ok(()),
        SyntheticEffect::LinearFir { impulse_response } => {
            if impulse_response.is_empty() || impulse_response.len() > MAX_FIR_TAPS {
                return Err(FactorialReferenceError::InvalidEffect(
                    "FIR must contain 1..=MAX_FIR_TAPS coefficients",
                ));
            }
            if impulse_response.iter().any(|value| !value.is_finite()) {
                return Err(FactorialReferenceError::InvalidEffect(
                    "FIR coefficients must be finite",
                ));
            }
            if impulse_response.iter().any(|value| value.abs() > MAX_SAMPLE_ABS) {
                return Err(FactorialReferenceError::InvalidEffect(
                    "FIR coefficient magnitude exceeds bound",
                ));
            }
            Ok(())
        }
        SyntheticEffect::Polynomial {
            quadratic_coefficient,
            cubic_coefficient,
        } => {
            if !quadratic_coefficient.is_finite()
                || !cubic_coefficient.is_finite()
                || quadratic_coefficient.abs() > MAX_SAMPLE_ABS
                || cubic_coefficient.abs() > MAX_SAMPLE_ABS
            {
                return Err(FactorialReferenceError::InvalidEffect(
                    "polynomial coefficients must be finite and bounded",
                ));
            }
            Ok(())
        }
        SyntheticEffect::HardClip { threshold } => {
            if !threshold.is_finite() || *threshold <= 0.0 || *threshold > MAX_SAMPLE_ABS {
                return Err(FactorialReferenceError::InvalidEffect(
                    "hard-clip threshold must be finite, positive and bounded",
                ));
            }
            Ok(())
        }
        SyntheticEffect::ThreeWayProduct {
            event_indices,
            coefficient,
        } => {
            let mut unique = *event_indices;
            unique.sort_unstable();
            if unique[0] == unique[1]
                || unique[1] == unique[2]
                || unique[2] >= event_count
            {
                return Err(FactorialReferenceError::InvalidEffect(
                    "three-way product requires three distinct in-range event indices",
                ));
            }
            if !coefficient.is_finite() || coefficient.abs() > MAX_SAMPLE_ABS {
                return Err(FactorialReferenceError::InvalidEffect(
                    "three-way coefficient must be finite and bounded",
                ));
            }
            Ok(())
        }
    }
}

fn validate_inputs(
    events: &[SyntheticEventSignal],
    effect: &SyntheticEffect,
    config: &FactorialReferenceConfig,
) -> Result<(), FactorialReferenceError> {
    if events.len() < 2 || events.len() > MAX_EVENT_COUNT {
        return Err(FactorialReferenceError::InvalidEventCount);
    }
    let frames = events[0].samples.len();
    if frames < 16 || frames > MAX_INPUT_FRAMES {
        return Err(FactorialReferenceError::InvalidFrameCount);
    }
    if config.spectral_window_frames < 16
        || !config.spectral_window_frames.is_power_of_two()
        || config.spectral_window_frames > frames
    {
        return Err(FactorialReferenceError::InvalidConfig(
            "spectral window must be a power of two in 16..=input frames",
        ));
    }

    for (expected_index, event) in events.iter().enumerate() {
        if event.source_event_index != expected_index {
            return Err(FactorialReferenceError::NonCanonicalEventIndex);
        }
        if event.samples.len() != frames {
            return Err(FactorialReferenceError::UnequalEventFrameCounts);
        }
        if event.samples.iter().any(|sample| !sample.is_finite()) {
            return Err(FactorialReferenceError::NonFiniteInput);
        }
        if event.samples.iter().any(|sample| sample.abs() > MAX_SAMPLE_ABS) {
            return Err(FactorialReferenceError::InputOutOfRange);
        }
    }
    validate_effect(effect, events.len())
}

fn included_indices(mask: u16, event_count: usize) -> Vec<usize> {
    (0..event_count)
        .filter(|index| mask & (1u16 << index) != 0)
        .collect()
}

fn render_coalition(
    events: &[SyntheticEventSignal],
    effect: &SyntheticEffect,
    mask: u16,
) -> Result<Vec<f64>, FactorialReferenceError> {
    let input_frames = events[0].samples.len();
    let mut mixed = vec![0.0f64; input_frames];
    for event in events {
        if mask & (1u16 << event.source_event_index) != 0 {
            for (output, sample) in mixed.iter_mut().zip(&event.samples) {
                *output += *sample;
            }
        }
    }

    let output = match effect {
        SyntheticEffect::LinearMix => mixed,
        SyntheticEffect::LinearFir { impulse_response } => {
            let output_len = input_frames + impulse_response.len() - 1;
            let mut convolved = vec![0.0f64; output_len];
            for (sample_index, sample) in mixed.iter().enumerate() {
                for (tap_index, tap) in impulse_response.iter().enumerate() {
                    convolved[sample_index + tap_index] += sample * tap;
                }
            }
            convolved
        }
        SyntheticEffect::Polynomial {
            quadratic_coefficient,
            cubic_coefficient,
        } => mixed
            .into_iter()
            .map(|sample| {
                sample
                    + quadratic_coefficient * sample * sample
                    + cubic_coefficient * sample * sample * sample
            })
            .collect(),
        SyntheticEffect::HardClip { threshold } => mixed
            .into_iter()
            .map(|sample| sample.clamp(-threshold, *threshold))
            .collect(),
        SyntheticEffect::ThreeWayProduct {
            event_indices,
            coefficient,
        } => {
            let all_present = event_indices
                .iter()
                .all(|index| mask & (1u16 << index) != 0);
            let mut output = mixed;
            if all_present {
                let a = &events[event_indices[0]].samples;
                let b = &events[event_indices[1]].samples;
                let c = &events[event_indices[2]].samples;
                for index in 0..input_frames {
                    output[index] += coefficient * a[index] * b[index] * c[index];
                }
            }
            output
        }
    };

    if output.iter().any(|sample| !sample.is_finite()) {
        return Err(FactorialReferenceError::NonFiniteOutput);
    }
    if output.iter().any(|sample| sample.abs() > MAX_OUTPUT_ABS) {
        return Err(FactorialReferenceError::OutputOutOfRange);
    }
    Ok(output)
}

fn spectral_magnitudes(
    samples: &[f64],
    window_frames: usize,
) -> Vec<f64> {
    let hann: Vec<f64> = (0..window_frames)
        .map(|index| {
            0.5 - 0.5 * (std::f64::consts::TAU * index as f64 / window_frames as f64).cos()
        })
        .collect();
    let mut buffer: Vec<Complex<f64>> = samples[..window_frames]
        .iter()
        .zip(hann.iter())
        .map(|(sample, window)| Complex::new(sample * window, 0.0))
        .collect();
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(window_frames);
    fft.process(&mut buffer);
    buffer[..=(window_frames / 2)]
        .iter()
        .map(|value| value.norm())
        .collect()
}

fn feature_dividends(
    coalition_features: &[Vec<f64>],
    event_count: usize,
) -> Vec<Vec<f64>> {
    let coalition_count = coalition_features.len();
    let feature_count = coalition_features[0].len();
    let mut dividends = vec![vec![0.0f64; feature_count]; coalition_count];

    // Möbius transform on the Boolean lattice:
    // H(S) = F(S) - sum_{T proper subset S} H(T).
    // Numeric masks are topologically ordered: every proper submask is smaller.
    for mask in 1..coalition_count {
        let mut value = coalition_features[mask].clone();
        let mut submask = (mask - 1) & mask;
        while submask > 0 {
            for index in 0..feature_count {
                value[index] -= dividends[submask][index];
            }
            submask = (submask - 1) & mask;
        }
        dividends[mask] = value;
    }
    let _ = event_count;
    dividends
}

fn l2_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn peak_abs(values: &[f64]) -> f64 {
    values.iter().map(|value| value.abs()).fold(0.0, f64::max)
}

fn report_digest_valid(
    report: &FactorialGroundTruthReport,
) -> Result<bool, FactorialReferenceError> {
    let mut unsigned = report.clone();
    let receipt = std::mem::take(&mut unsigned.receipt_sha256);
    if receipt.len() != 64 {
        return Ok(false);
    }
    Ok(canonical_digest(&unsigned)? == receipt)
}

/// Enumerate all 2^N coalitions (N <= 8) and compute PCM and magnitude-space
/// Harsanyi dividends. This is the deterministic reference oracle for EVAL-020.
pub fn build_factorial_ground_truth(
    events: &[SyntheticEventSignal],
    effect: &SyntheticEffect,
    config: &FactorialReferenceConfig,
) -> Result<FactorialGroundTruthReport, FactorialReferenceError> {
    validate_inputs(events, effect, config)?;

    let event_count = events.len();
    let coalition_count = 1usize << event_count;
    let input_frames = events[0].samples.len();

    let input_bytes = serde_json::to_vec(&(events, effect, config))
        .map_err(|_| FactorialReferenceError::SerializationFailure)?;
    let effect_bytes = serde_json::to_vec(effect)
        .map_err(|_| FactorialReferenceError::SerializationFailure)?;

    let mut coalitions = Vec::with_capacity(coalition_count);
    for mask in 0..coalition_count {
        let pcm_samples = render_coalition(events, effect, mask as u16)?;
        let spectrum = spectral_magnitudes(&pcm_samples, config.spectral_window_frames);
        coalitions.push(SyntheticCoalitionRender {
            coalition_mask: mask as u16,
            included_event_indices: included_indices(mask as u16, event_count),
            pcm_samples,
            spectral_magnitudes: spectrum,
        });
    }

    let pcm_features: Vec<Vec<f64>> = coalitions
        .iter()
        .map(|coalition| coalition.pcm_samples.clone())
        .collect();
    let spectral_features: Vec<Vec<f64>> = coalitions
        .iter()
        .map(|coalition| coalition.spectral_magnitudes.clone())
        .collect();
    let pcm_dividends = feature_dividends(&pcm_features, event_count);
    let spectral_dividends = feature_dividends(&spectral_features, event_count);

    let mut dividends = Vec::with_capacity(coalition_count - 1);
    for mask in 1..coalition_count {
        let pcm = pcm_dividends[mask].clone();
        let spectral = spectral_dividends[mask].clone();
        dividends.push(HarsanyiDividend {
            coalition_mask: mask as u16,
            event_indices: included_indices(mask as u16, event_count),
            order: (mask as u16).count_ones() as u8,
            pcm_l2_norm: l2_norm(&pcm),
            pcm_peak_abs: peak_abs(&pcm),
            spectral_l2_norm: l2_norm(&spectral),
            spectral_peak_abs: peak_abs(&spectral),
            pcm_samples: pcm,
            spectral_magnitude_values: spectral,
        });
    }

    let output_frames = coalitions[0].pcm_samples.len();
    let mut report = FactorialGroundTruthReport {
        version: FACTORIAL_GROUND_TRUTH_VERSION.to_owned(),
        event_count,
        coalition_count,
        input_frame_count: input_frames,
        output_frame_count: output_frames,
        input_sha256: sha256_hex(&input_bytes),
        effect_sha256: sha256_hex(&effect_bytes),
        config: config.clone(),
        coalitions,
        dividends,
        receipt_sha256: String::new(),
    };
    report.receipt_sha256 = canonical_digest(&report)?;
    Ok(report)
}

/// Recompute every coalition and dividend from the exact source fixture.
pub fn verify_factorial_ground_truth(
    report: &FactorialGroundTruthReport,
    events: &[SyntheticEventSignal],
    effect: &SyntheticEffect,
    config: &FactorialReferenceConfig,
) -> Result<bool, FactorialReferenceError> {
    if !report_digest_valid(report)? {
        return Err(FactorialReferenceError::InvalidReceipt);
    }
    let expected = build_factorial_ground_truth(events, effect, config)?;
    if &expected != report {
        return Err(FactorialReferenceError::ReplayMismatch);
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> FactorialReferenceConfig {
        FactorialReferenceConfig {
            spectral_window_frames: 64,
        }
    }

    fn constant_event(index: usize, value: f64) -> SyntheticEventSignal {
        SyntheticEventSignal {
            source_event_index: index,
            samples: vec![value; 64],
        }
    }

    fn dividend(report: &FactorialGroundTruthReport, mask: u16) -> &HarsanyiDividend {
        report.dividends
            .iter()
            .find(|value| value.coalition_mask == mask)
            .expect("non-empty coalition dividend")
    }

    #[test]
    fn linear_mix_has_no_second_or_higher_order_pcm_dividends() {
        let events = vec![
            constant_event(0, 0.2),
            constant_event(1, 0.3),
            constant_event(2, -0.1),
        ];
        let report = build_factorial_ground_truth(&events, &SyntheticEffect::LinearMix, &config()).unwrap();
        for value in report.dividends.iter().filter(|value| value.order >= 2) {
            assert!(value.pcm_peak_abs < 1.0e-12);
        }
        assert_eq!(report.coalitions.len(), 8);
        assert_eq!(report.dividends.len(), 7);
    }

    #[test]
    fn fixed_linear_fir_preserves_pcm_additivity_and_exposes_tail_frames() {
        let events = vec![
            constant_event(0, 0.2),
            constant_event(1, 0.3),
            constant_event(2, 0.1),
        ];
        let effect = SyntheticEffect::LinearFir {
            impulse_response: vec![1.0, 0.5, -0.25, 0.125],
        };
        let report = build_factorial_ground_truth(&events, &effect, &config()).unwrap();
        assert_eq!(report.output_frame_count, 67);
        assert!(report.dividends.iter().filter(|value| value.order >= 2)
            .all(|value| value.pcm_peak_abs < 1.0e-12));
    }

    #[test]
    fn quadratic_shared_processor_creates_known_pairwise_dividend() {
        let events = vec![constant_event(0, 0.2), constant_event(1, 0.3)];
        let effect = SyntheticEffect::Polynomial {
            quadratic_coefficient: 0.5,
            cubic_coefficient: 0.0,
        };
        let report = build_factorial_ground_truth(&events, &effect, &config()).unwrap();
        let pair = dividend(&report, 0b11);
        // 2*q*a*b = 2*0.5*0.2*0.3 = 0.06 per frame.
        assert!((pair.pcm_samples[0] - 0.06).abs() < 1.0e-12);
        assert!(report.dividends.iter().filter(|value| value.order > 2)
            .all(|value| value.pcm_peak_abs < 1.0e-12));
    }

    #[test]
    fn pure_three_way_product_has_third_order_but_no_pairwise_dividends() {
        let events = vec![
            constant_event(0, 0.2),
            constant_event(1, 0.3),
            constant_event(2, 0.4),
        ];
        let effect = SyntheticEffect::ThreeWayProduct {
            event_indices: [0, 1, 2],
            coefficient: 0.5,
        };
        let report = build_factorial_ground_truth(&events, &effect, &config()).unwrap();
        for mask in [0b011, 0b101, 0b110] {
            assert!(dividend(&report, mask).pcm_peak_abs < 1.0e-12);
        }
        let triple = dividend(&report, 0b111);
        // 0.5 * 0.2 * 0.3 * 0.4 = 0.012 per frame.
        assert!((triple.pcm_samples[0] - 0.012).abs() < 1.0e-12);
        assert_eq!(triple.order, 3);
    }

    #[test]
    fn phase_quadrature_has_spectral_dividend_without_pcm_dividend() {
        let frames = 64usize;
        let sine = SyntheticEventSignal {
            source_event_index: 0,
            samples: (0..frames)
                .map(|index| (std::f64::consts::TAU * 4.0 * index as f64 / frames as f64).sin())
                .collect(),
        };
        let cosine = SyntheticEventSignal {
            source_event_index: 1,
            samples: (0..frames)
                .map(|index| (std::f64::consts::TAU * 4.0 * index as f64 / frames as f64).cos())
                .collect(),
        };
        let report = build_factorial_ground_truth(&[sine, cosine], &SyntheticEffect::LinearMix, &config()).unwrap();
        let pair = dividend(&report, 0b11);
        assert!(pair.pcm_peak_abs < 1.0e-12);
        assert!(pair.spectral_peak_abs > 1.0e-3);
    }

    #[test]
    fn hard_clip_fixture_creates_shared_processing_interaction() {
        let events = vec![constant_event(0, 0.8), constant_event(1, 0.8)];
        let report = build_factorial_ground_truth(
            &events,
            &SyntheticEffect::HardClip { threshold: 1.0 },
            &config(),
        ).unwrap();
        assert!((dividend(&report, 0b11).pcm_samples[0] + 0.6).abs() < 1.0e-12);
    }

    #[test]
    fn report_replays_and_rejects_tampering() {
        let events = vec![constant_event(0, 0.2), constant_event(1, 0.3)];
        let effect = SyntheticEffect::Polynomial {
            quadratic_coefficient: 0.25,
            cubic_coefficient: 0.0,
        };
        let report = build_factorial_ground_truth(&events, &effect, &config()).unwrap();
        assert!(verify_factorial_ground_truth(&report, &events, &effect, &config()).unwrap());
        let mut tampered = report.clone();
        tampered.event_count += 1;
        assert_eq!(
            verify_factorial_ground_truth(&tampered, &events, &effect, &config()).unwrap_err(),
            FactorialReferenceError::InvalidReceipt
        );
    }

    #[test]
    fn malformed_inputs_and_effect_parameters_fail_closed() {
        let events = vec![constant_event(0, 0.2), constant_event(1, 0.3)];
        let bad_events = vec![
            SyntheticEventSignal { source_event_index: 1, samples: vec![0.0; 64] },
            SyntheticEventSignal { source_event_index: 0, samples: vec![0.0; 64] },
        ];
        assert_eq!(
            build_factorial_ground_truth(&bad_events, &SyntheticEffect::LinearMix, &config()).unwrap_err(),
            FactorialReferenceError::NonCanonicalEventIndex
        );
        assert_eq!(
            build_factorial_ground_truth(
                &events,
                &SyntheticEffect::HardClip { threshold: f64::NAN },
                &config(),
            ).unwrap_err(),
            FactorialReferenceError::InvalidEffect("hard-clip threshold must be finite, positive and bounded")
        );
        let mut nan_event = events.clone();
        nan_event[0].samples[0] = f64::NAN;
        assert_eq!(
            build_factorial_ground_truth(&nan_event, &SyntheticEffect::LinearMix, &config()).unwrap_err(),
            FactorialReferenceError::NonFiniteInput
        );
    }
}
