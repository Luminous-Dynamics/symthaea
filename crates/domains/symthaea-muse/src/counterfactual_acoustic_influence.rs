// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Paired-render, event-local acoustic influence measurement.
//!
//! This module estimates the acoustic consequence of a declared symbolic
//! intervention by comparing a baseline render with a counterfactual render.
//! Repeated baseline renders establish a per-window noise floor. The result is
//! an influence measurement, not exclusive sample ownership, renderer
//! attestation, or a preservation verdict.
//!
//! A nonzero effect can include shared effects, phase interference, and
//! nonlinear mastering interactions. It is not a source-separation mask.
//! Renderer receipt digests are bound but must be independently verified by
//! the caller. Thresholds require external calibration; an arbitrary
//! calibration ID is recorded, not authenticated or validated here.

use crate::audio_preservation::AudioView;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const COUNTERFACTUAL_ACOUSTIC_INFLUENCE_VERSION: &str =
    "mel-mr-eval-017-counterfactual-acoustic-influence-v0.1";
const MAX_FRAME_SIZE: usize = 1_048_576;

/// A typed symbolic intervention. Schedule digests bind the exact input
/// schedules; these fields describe the declared operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "operation")]
pub enum CounterfactualIntervention {
    Remove,
    Transpose { semitones: i16 },
    ShiftOnset { delta_ticks: i64 },
    ChangeDuration { delta_ticks: i64 },
    ChangeVelocity { delta: i16 },
}

impl CounterfactualIntervention {
    fn validate(self) -> Result<(), InfluenceInputError> {
        match self {
            Self::Transpose { semitones: 0 }
            | Self::ShiftOnset { delta_ticks: 0 }
            | Self::ChangeDuration { delta_ticks: 0 }
            | Self::ChangeVelocity { delta: 0 } => Err(InfluenceInputError::NoOpIntervention),
            Self::Transpose { semitones } if !(-127..=127).contains(&semitones) => {
                Err(InfluenceInputError::InvalidIntervention)
            }
            Self::ChangeVelocity { delta } if !(-127..=127).contains(&delta) => {
                Err(InfluenceInputError::InvalidIntervention)
            }
            _ => Ok(()),
        }
    }
}

/// Thresholds for classifying a window as measurably affected.
///
/// noise_multiplier is applied to the maximum same-schedule repeat-render
/// difference observed in that window. Absolute floors prevent a perfectly
/// deterministic fixture from treating floating-point dust as influence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CounterfactualInfluenceConfig {
    /// Analysis window size in audio frames; must be at least 16.
    pub frame_size: usize,
    /// Distance between window starts; must be in 1..=frame_size.
    pub hop_size: usize,
    /// Multiplier applied to per-window repeat-render noise.
    pub noise_multiplier: f32,
    /// Minimum RMS of the sample-wise counterfactual difference.
    pub minimum_rms_delta: f32,
    /// Minimum normalized spectral-magnitude distance in [0, 1].
    pub minimum_spectral_distance: f32,
    /// Minimum unique time windows needed for a classified result.
    pub minimum_covered_windows: usize,
    /// External frozen calibration report identifier. None forces abstention.
    pub calibration_id: Option<String>,
}

impl Default for CounterfactualInfluenceConfig {
    fn default() -> Self {
        Self {
            frame_size: 1024,
            hop_size: 256,
            noise_multiplier: 3.0,
            minimum_rms_delta: 0.0001,
            minimum_spectral_distance: 0.01,
            minimum_covered_windows: 1,
            calibration_id: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CounterfactualInfluenceIdentity {
    pub source_score_sha256: String,
    /// Stable identity of the source event being intervened on.
    pub source_event_id: String,
    pub intervention: CounterfactualIntervention,
    pub baseline_schedule_sha256: String,
    pub counterfactual_schedule_sha256: String,
    /// Commits to renderer binary/assets/configuration/topology as defined by
    /// the caller's renderer profile.
    pub renderer_profile_sha256: String,
    pub baseline_render_receipt_sha256: String,
    /// One same-schedule receipt per repeat audio buffer, in matching order.
    pub baseline_repeat_receipt_sha256: Vec<String>,
    pub counterfactual_render_receipt_sha256: String,
}

pub struct CounterfactualInfluenceContext<'a> {
    pub source_score_sha256: &'a str,
    pub source_event_id: &'a str,
    pub intervention: CounterfactualIntervention,
    pub baseline_schedule_sha256: &'a str,
    pub counterfactual_schedule_sha256: &'a str,
    pub renderer_profile_sha256: &'a str,
    pub baseline_render_receipt_sha256: &'a str,
    pub baseline_repeat_receipt_sha256: &'a [&'a str],
    pub counterfactual_render_receipt_sha256: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InfluenceVerdict {
    /// At least one time/channel window exceeded its repeat-noise threshold
    /// and configured absolute floor under a supplied calibration.
    Detected,
    /// No window exceeded its repeat-noise threshold under a supplied
    /// calibration. This does not prove the source event is inaudible.
    NoDetectableInfluence,
    /// Calibration or coverage was insufficient for a classified result.
    Abstain,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InfluenceWindowMeasurement {
    pub start_frame: usize,
    pub end_frame: usize,
    pub channel: usize,
    pub effect_rms_delta: f32,
    pub repeat_noise_rms_floor: f32,
    pub rms_detection_threshold: f32,
    pub effect_spectral_distance: f32,
    pub repeat_noise_spectral_floor: f32,
    pub spectral_detection_threshold: f32,
    pub detected: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InfluenceSupportInterval {
    pub start_frame: usize,
    pub end_frame: usize,
    pub channel: usize,
}

/// Content-bound result for one exact baseline/intervention pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CounterfactualInfluenceReport {
    pub verifier_version: String,
    pub identity: CounterfactualInfluenceIdentity,
    pub config: CounterfactualInfluenceConfig,
    pub sample_rate_hz: u32,
    pub channels: usize,
    pub total_frames: usize,
    pub baseline_audio_sha256: String,
    pub baseline_repeat_audio_sha256: Vec<String>,
    pub counterfactual_audio_sha256: String,
    pub unique_windows: usize,
    pub audited_channel_windows: usize,
    pub detected_channel_windows: usize,
    pub support_intervals: Vec<InfluenceSupportInterval>,
    pub worst_effect_rms_delta: f32,
    pub worst_effect_spectral_distance: f32,
    pub verdict: InfluenceVerdict,
    pub reasons: Vec<String>,
    pub measurements: Vec<InfluenceWindowMeasurement>,
    /// SHA-256 over the canonical report with this field blank.
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InfluenceInputError {
    ZeroSampleRate,
    ZeroChannels,
    EmptyAudio,
    InterleavedLengthNotDivisibleByChannels,
    NonFiniteSample,
    FormatMismatch,
    LengthMismatch,
    MissingRepeatRender,
    RepeatReceiptCountMismatch,
    InvalidSha256(&'static str),
    InvalidEventId,
    InvalidIntervention,
    NoOpIntervention,
    InvalidConfig(&'static str),
    SerializationFailure,
}

impl std::fmt::Display for InfluenceInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSampleRate => write!(f, "sample rate must be non-zero"),
            Self::ZeroChannels => write!(f, "channel count must be non-zero"),
            Self::EmptyAudio => write!(f, "audio must contain at least one frame"),
            Self::InterleavedLengthNotDivisibleByChannels => {
                write!(f, "interleaved sample count is not divisible by channel count")
            }
            Self::NonFiniteSample => write!(f, "audio contains NaN or infinite samples"),
            Self::FormatMismatch => write!(f, "paired render formats differ"),
            Self::LengthMismatch => write!(f, "paired render frame counts differ"),
            Self::MissingRepeatRender => write!(f, "at least one same-schedule repeat render is required"),
            Self::RepeatReceiptCountMismatch => {
                write!(f, "repeat audio and repeat renderer receipt counts differ")
            }
            Self::InvalidSha256(field) => write!(f, "invalid SHA-256 field: {field}"),
            Self::InvalidEventId => write!(f, "source event identity is empty, too long or contains controls"),
            Self::InvalidIntervention => write!(f, "intervention parameter is outside its valid range"),
            Self::NoOpIntervention => write!(f, "intervention must declare a non-zero symbolic change"),
            Self::InvalidConfig(reason) => write!(f, "invalid influence config: {reason}"),
            Self::SerializationFailure => write!(f, "could not serialize influence report"),
        }
    }
}

impl std::error::Error for InfluenceInputError {}

/// Estimate one event's conditional acoustic effect from an exact paired render.
///
/// Every buffer must have identical sample rate, channel count and frame count.
/// At least one same-schedule repeat is required. A classified verdict requires
/// a calibration ID and the minimum number of measured windows.
pub fn estimate_counterfactual_influence(
    baseline: AudioView<'_>,
    baseline_repeats: &[AudioView<'_>],
    counterfactual: AudioView<'_>,
    context: CounterfactualInfluenceContext<'_>,
    config: &CounterfactualInfluenceConfig,
) -> Result<CounterfactualInfluenceReport, InfluenceInputError> {
    validate_view(baseline)?;
    validate_view(counterfactual)?;
    if baseline_repeats.is_empty() {
        return Err(InfluenceInputError::MissingRepeatRender);
    }
    if baseline.sample_rate_hz != counterfactual.sample_rate_hz
        || baseline.channels != counterfactual.channels
    {
        return Err(InfluenceInputError::FormatMismatch);
    }
    if baseline.interleaved_samples.len() != counterfactual.interleaved_samples.len() {
        return Err(InfluenceInputError::LengthMismatch);
    }
    for repeat in baseline_repeats {
        validate_view(*repeat)?;
        if repeat.sample_rate_hz != baseline.sample_rate_hz
            || repeat.channels != baseline.channels
        {
            return Err(InfluenceInputError::FormatMismatch);
        }
        if repeat.interleaved_samples.len() != baseline.interleaved_samples.len() {
            return Err(InfluenceInputError::LengthMismatch);
        }
    }
    validate_config(config)?;
    context.intervention.validate()?;
    let identity = validate_context(context, baseline_repeats.len())?;

    let total_frames = baseline.interleaved_samples.len() / baseline.channels;
    if config.frame_size > total_frames {
        return Err(InfluenceInputError::InvalidConfig(
            "frame_size must not exceed the rendered audio length",
        ));
    }
    let starts = analysis_starts(total_frames, config.frame_size, config.hop_size);
    let fft = FftPlanner::<f32>::new().plan_fft_forward(config.frame_size);
    let window = hann_window(config.frame_size);
    let mut measurements = Vec::with_capacity(starts.len().saturating_mul(baseline.channels));
    let mut support_intervals = Vec::new();
    let mut detected_channel_windows = 0usize;
    let mut worst_effect_rms_delta = 0.0f32;
    let mut worst_effect_spectral_distance = 0.0f32;

    for start_frame in starts.iter().copied() {
        let end_frame = start_frame + config.frame_size;
        for channel in 0..baseline.channels {
            let base = channel_window(baseline, start_frame, config.frame_size, channel);
            let counter = channel_window(counterfactual, start_frame, config.frame_size, channel);
            let (effect_rms_delta, effect_spectral_distance) =
                compare_windows(&base, &counter, &window, fft.as_ref());

            // Use the maximum observed same-schedule deviation so one noisy
            // repeat cannot be diluted by several unusually quiet repeats.
            let mut repeat_noise_rms_floor = 0.0f32;
            let mut repeat_noise_spectral_floor = 0.0f32;
            for repeat in baseline_repeats {
                let repeat_samples =
                    channel_window(*repeat, start_frame, config.frame_size, channel);
                let (rms_noise, spectral_noise) =
                    compare_windows(&base, &repeat_samples, &window, fft.as_ref());
                repeat_noise_rms_floor = repeat_noise_rms_floor.max(rms_noise);
                repeat_noise_spectral_floor = repeat_noise_spectral_floor.max(spectral_noise);
            }

            let rms_detection_threshold = (repeat_noise_rms_floor * config.noise_multiplier)
                .max(config.minimum_rms_delta);
            let spectral_detection_threshold =
                (repeat_noise_spectral_floor * config.noise_multiplier)
                    .max(config.minimum_spectral_distance);
            let detected = effect_rms_delta > rms_detection_threshold
                || effect_spectral_distance > spectral_detection_threshold;

            worst_effect_rms_delta = worst_effect_rms_delta.max(effect_rms_delta);
            worst_effect_spectral_distance =
                worst_effect_spectral_distance.max(effect_spectral_distance);
            if detected {
                detected_channel_windows += 1;
                support_intervals.push(InfluenceSupportInterval {
                    start_frame,
                    end_frame,
                    channel,
                });
            }
            measurements.push(InfluenceWindowMeasurement {
                start_frame,
                end_frame,
                channel,
                effect_rms_delta,
                repeat_noise_rms_floor,
                rms_detection_threshold,
                effect_spectral_distance,
                repeat_noise_spectral_floor,
                spectral_detection_threshold,
                detected,
            });
        }
    }

    support_intervals = merge_support_intervals(support_intervals);
    let mut reasons = Vec::new();
    let verdict = if config.calibration_id.is_none() {
        reasons.push(
            "no calibration ID supplied; exploratory measurements cannot be classified".to_owned(),
        );
        InfluenceVerdict::Abstain
    } else if starts.len() < config.minimum_covered_windows {
        reasons.push(format!(
            "only {} unique windows were measured; {} required",
            starts.len(),
            config.minimum_covered_windows
        ));
        InfluenceVerdict::Abstain
    } else if detected_channel_windows > 0 {
        InfluenceVerdict::Detected
    } else {
        InfluenceVerdict::NoDetectableInfluence
    };

    let mut report = CounterfactualInfluenceReport {
        verifier_version: COUNTERFACTUAL_ACOUSTIC_INFLUENCE_VERSION.to_owned(),
        identity,
        config: config.clone(),
        sample_rate_hz: baseline.sample_rate_hz,
        channels: baseline.channels,
        total_frames,
        baseline_audio_sha256: audio_digest(baseline),
        baseline_repeat_audio_sha256: baseline_repeats
            .iter()
            .map(|view| audio_digest(*view))
            .collect(),
        counterfactual_audio_sha256: audio_digest(counterfactual),
        unique_windows: starts.len(),
        audited_channel_windows: measurements.len(),
        detected_channel_windows,
        support_intervals,
        worst_effect_rms_delta,
        worst_effect_spectral_distance,
        verdict,
        reasons,
        measurements,
        receipt_sha256: String::new(),
    };
    report.receipt_sha256 = compute_receipt_sha256(&report)?;
    Ok(report)
}

/// Replay the measurements and support map against the exact supplied buffers.
/// The report digest alone is insufficient; this function rebuilds the result.
pub fn verify_counterfactual_influence_report(
    report: &CounterfactualInfluenceReport,
    baseline: AudioView<'_>,
    baseline_repeats: &[AudioView<'_>],
    counterfactual: AudioView<'_>,
) -> Result<bool, InfluenceInputError> {
    if report.receipt_sha256.is_empty()
        || compute_receipt_sha256(report)? != report.receipt_sha256
    {
        return Ok(false);
    }
    let repeat_receipts: Vec<&str> = report
        .identity
        .baseline_repeat_receipt_sha256
        .iter()
        .map(String::as_str)
        .collect();
    let context = CounterfactualInfluenceContext {
        source_score_sha256: &report.identity.source_score_sha256,
        source_event_id: &report.identity.source_event_id,
        intervention: report.identity.intervention,
        baseline_schedule_sha256: &report.identity.baseline_schedule_sha256,
        counterfactual_schedule_sha256: &report.identity.counterfactual_schedule_sha256,
        renderer_profile_sha256: &report.identity.renderer_profile_sha256,
        baseline_render_receipt_sha256: &report.identity.baseline_render_receipt_sha256,
        baseline_repeat_receipt_sha256: &repeat_receipts,
        counterfactual_render_receipt_sha256: &report.identity.counterfactual_render_receipt_sha256,
    };
    let expected = estimate_counterfactual_influence(
        baseline,
        baseline_repeats,
        counterfactual,
        context,
        &report.config,
    )?;
    Ok(&expected == report)
}

fn validate_context(
    context: CounterfactualInfluenceContext<'_>,
    repeat_count: usize,
) -> Result<CounterfactualInfluenceIdentity, InfluenceInputError> {
    if context.source_event_id.trim().is_empty()
        || context.source_event_id.len() > 256
        || context.source_event_id.chars().any(char::is_control)
    {
        return Err(InfluenceInputError::InvalidEventId);
    }
    for (field, value) in [
        ("source_score_sha256", context.source_score_sha256),
        ("baseline_schedule_sha256", context.baseline_schedule_sha256),
        ("counterfactual_schedule_sha256", context.counterfactual_schedule_sha256),
        ("renderer_profile_sha256", context.renderer_profile_sha256),
        ("baseline_render_receipt_sha256", context.baseline_render_receipt_sha256),
        ("counterfactual_render_receipt_sha256", context.counterfactual_render_receipt_sha256),
    ] {
        validate_sha256(field, value)?;
    }
    if context.baseline_repeat_receipt_sha256.len() != repeat_count {
        return Err(InfluenceInputError::RepeatReceiptCountMismatch);
    }
    for value in context.baseline_repeat_receipt_sha256 {
        validate_sha256("baseline_repeat_render_receipt_sha256", value)?;
    }
    context.intervention.validate()?;
    Ok(CounterfactualInfluenceIdentity {
        source_score_sha256: context.source_score_sha256.to_owned(),
        source_event_id: context.source_event_id.to_owned(),
        intervention: context.intervention,
        baseline_schedule_sha256: context.baseline_schedule_sha256.to_owned(),
        counterfactual_schedule_sha256: context.counterfactual_schedule_sha256.to_owned(),
        renderer_profile_sha256: context.renderer_profile_sha256.to_owned(),
        baseline_render_receipt_sha256: context.baseline_render_receipt_sha256.to_owned(),
        baseline_repeat_receipt_sha256: context
            .baseline_repeat_receipt_sha256
            .iter()
            .map(|digest| (*digest).to_owned())
            .collect(),
        counterfactual_render_receipt_sha256: context
            .counterfactual_render_receipt_sha256
            .to_owned(),
    })
}

fn validate_sha256(field: &'static str, value: &str) -> Result<(), InfluenceInputError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(InfluenceInputError::InvalidSha256(field));
    }
    Ok(())
}

fn validate_view(view: AudioView<'_>) -> Result<(), InfluenceInputError> {
    if view.sample_rate_hz == 0 {
        return Err(InfluenceInputError::ZeroSampleRate);
    }
    if view.channels == 0 {
        return Err(InfluenceInputError::ZeroChannels);
    }
    if view.interleaved_samples.is_empty() {
        return Err(InfluenceInputError::EmptyAudio);
    }
    if view.interleaved_samples.len() % view.channels != 0 {
        return Err(InfluenceInputError::InterleavedLengthNotDivisibleByChannels);
    }
    if view.interleaved_samples.iter().any(|sample| !sample.is_finite()) {
        return Err(InfluenceInputError::NonFiniteSample);
    }
    Ok(())
}

fn validate_config(config: &CounterfactualInfluenceConfig) -> Result<(), InfluenceInputError> {
    if config.frame_size < 16 {
        return Err(InfluenceInputError::InvalidConfig(
            "frame_size must be at least 16",
        ));
    }
    if config.frame_size > MAX_FRAME_SIZE {
        return Err(InfluenceInputError::InvalidConfig(
            "frame_size exceeds the maximum supported allocation",
        ));
    }
    if config.hop_size == 0 || config.hop_size > config.frame_size {
        return Err(InfluenceInputError::InvalidConfig(
            "hop_size must be in 1..=frame_size",
        ));
    }
    if !config.noise_multiplier.is_finite()
        || config.noise_multiplier < 1.0
        || config.noise_multiplier > 100.0
    {
        return Err(InfluenceInputError::InvalidConfig(
            "noise_multiplier must be finite and in 1..=100",
        ));
    }
    if !config.minimum_rms_delta.is_finite()
        || config.minimum_rms_delta < 0.0
        || config.minimum_rms_delta > 2.0
    {
        return Err(InfluenceInputError::InvalidConfig(
            "minimum_rms_delta must be finite and in 0..=2",
        ));
    }
    if !config.minimum_spectral_distance.is_finite()
        || !(0.0..=1.0).contains(&config.minimum_spectral_distance)
    {
        return Err(InfluenceInputError::InvalidConfig(
            "minimum_spectral_distance must be finite and in 0..=1",
        ));
    }
    if config.minimum_covered_windows == 0 {
        return Err(InfluenceInputError::InvalidConfig(
            "minimum_covered_windows must be non-zero",
        ));
    }
    if config
        .calibration_id
        .as_ref()
        .is_some_and(|id| id.trim().is_empty())
    {
        return Err(InfluenceInputError::InvalidConfig(
            "calibration_id must be non-empty when present",
        ));
    }
    Ok(())
}

fn analysis_starts(total_frames: usize, frame_size: usize, hop_size: usize) -> Vec<usize> {
    let final_start = total_frames - frame_size;
    let mut starts = Vec::new();
    let mut start = 0usize;
    loop {
        starts.push(start);
        if start >= final_start {
            break;
        }
        let next = start.saturating_add(hop_size);
        if next > final_start {
            if starts.last().copied() != Some(final_start) {
                starts.push(final_start);
            }
            break;
        }
        start = next;
    }
    starts
}

fn channel_window(view: AudioView<'_>, start: usize, frames: usize, channel: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(frames);
    for frame in start..start + frames {
        output.push(view.interleaved_samples[frame * view.channels + channel]);
    }
    output
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|index| {
            0.5 - 0.5
                * (std::f32::consts::TAU * index as f32 / (size - 1) as f32).cos()
        })
        .collect()
}

fn compare_windows(
    left: &[f32],
    right: &[f32],
    window: &[f32],
    fft: &dyn Fft<f32>,
) -> (f32, f32) {
    let mean_square_delta = left
        .iter()
        .zip(right)
        .map(|(x, y)| {
            let delta = x - y;
            delta * delta
        })
        .sum::<f32>()
        / left.len() as f32;
    let rms_delta = mean_square_delta.max(0.0).sqrt();

    let mut left_spectrum: Vec<Complex<f32>> = left
        .iter()
        .zip(window)
        .map(|(sample, taper)| Complex::new(sample * taper, 0.0))
        .collect();
    let mut right_spectrum: Vec<Complex<f32>> = right
        .iter()
        .zip(window)
        .map(|(sample, taper)| Complex::new(sample * taper, 0.0))
        .collect();
    fft.process(&mut left_spectrum);
    fft.process(&mut right_spectrum);

    // One-sided magnitude spectra. The denominator is the per-bin maximum,
    // so the normalized L1 difference is bounded in [0, 1].
    let bins = left_spectrum.len() / 2 + 1;
    let mut numerator = 0.0f32;
    let mut denominator = 0.0f32;
    for index in 0..bins {
        let left_magnitude = left_spectrum[index].norm();
        let right_magnitude = right_spectrum[index].norm();
        numerator += (left_magnitude - right_magnitude).abs();
        denominator += left_magnitude.max(right_magnitude);
    }
    let spectral_distance = if denominator <= f32::EPSILON {
        0.0
    } else {
        (numerator / denominator).clamp(0.0, 1.0)
    };
    (rms_delta, spectral_distance)
}

fn merge_support_intervals(
    mut intervals: Vec<InfluenceSupportInterval>,
) -> Vec<InfluenceSupportInterval> {
    intervals.sort_by_key(|interval| (interval.channel, interval.start_frame, interval.end_frame));
    let mut merged: Vec<InfluenceSupportInterval> = Vec::with_capacity(intervals.len());
    for interval in intervals {
        if let Some(previous) = merged.last_mut()
            && previous.channel == interval.channel
            && interval.start_frame <= previous.end_frame
        {
            previous.end_frame = previous.end_frame.max(interval.end_frame);
        } else {
            merged.push(interval);
        }
    }
    merged
}

fn audio_digest(view: AudioView<'_>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous-dynamics-counterfactual-influence-audio-v1\0");
    hasher.update(view.sample_rate_hz.to_le_bytes());
    hasher.update((view.channels as u64).to_le_bytes());
    hasher.update(((view.interleaved_samples.len() / view.channels) as u64).to_le_bytes());
    for sample in view.interleaved_samples {
        hasher.update(sample.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn compute_receipt_sha256(
    report: &CounterfactualInfluenceReport,
) -> Result<String, InfluenceInputError> {
    let mut unsealed = report.clone();
    unsealed.receipt_sha256.clear();
    let bytes =
        serde_json::to_vec(&unsealed).map_err(|_| InfluenceInputError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous-dynamics-mel-mr-eval-017-influence-report-v1\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const DIGEST_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const DIGEST_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const DIGEST_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";

    fn sine(frames: usize, sample_rate_hz: u32, frequency_hz: f32) -> Vec<f32> {
        (0..frames)
            .map(|index| {
                (std::f32::consts::TAU * frequency_hz * index as f32
                    / sample_rate_hz as f32)
                    .sin()
            })
            .collect()
    }

    fn view(samples: &[f32]) -> AudioView<'_> {
        AudioView {
            interleaved_samples: samples,
            sample_rate_hz: 16_000,
            channels: 1,
        }
    }

    fn context<'a>(repeat_receipts: &'a [&'a str]) -> CounterfactualInfluenceContext<'a> {
        CounterfactualInfluenceContext {
            source_score_sha256: DIGEST_A,
            source_event_id: "score_event_index:3",
            intervention: CounterfactualIntervention::Remove,
            baseline_schedule_sha256: DIGEST_B,
            counterfactual_schedule_sha256: DIGEST_C,
            renderer_profile_sha256: DIGEST_D,
            baseline_render_receipt_sha256: DIGEST_E,
            baseline_repeat_receipt_sha256: repeat_receipts,
            counterfactual_render_receipt_sha256: DIGEST_A,
        }
    }

    fn calibrated_config() -> CounterfactualInfluenceConfig {
        CounterfactualInfluenceConfig {
            frame_size: 64,
            hop_size: 32,
            noise_multiplier: 3.0,
            minimum_rms_delta: 0.001,
            minimum_spectral_distance: 0.02,
            minimum_covered_windows: 4,
            calibration_id: Some("synthetic-influence-calibration-v0".to_owned()),
        }
    }

    #[test]
    fn source_event_effect_is_detected_against_repeat_render_noise() {
        let baseline = sine(1024, 16_000, 440.0);
        let repeat = baseline.clone();
        let mut counterfactual = baseline.clone();
        // The removed note has a 128-frame body followed by a synthetic release.
        for frame in 256..512 {
            let envelope = if frame < 384 {
                1.0
            } else {
                (-((frame - 384) as f32) / 48.0).exp()
            };
            counterfactual[frame] *= 1.0 - envelope;
        }
        let repeat_receipts = [DIGEST_A];
        let report = estimate_counterfactual_influence(
            view(&baseline),
            &[view(&repeat)],
            view(&counterfactual),
            context(&repeat_receipts),
            &calibrated_config(),
        )
        .expect("valid paired counterfactual");

        assert_eq!(report.verdict, InfluenceVerdict::Detected);
        assert!(report.detected_channel_windows > 0);
        assert!(report.support_intervals.iter().any(|interval| {
            interval.start_frame <= 256 && interval.end_frame >= 512
        }));
        assert!(verify_counterfactual_influence_report(
            &report,
            view(&baseline),
            &[view(&repeat)],
            view(&counterfactual)
        )
        .expect("report replay"));
    }

    #[test]
    fn repeat_render_variance_is_not_misreported_as_intervention_effect() {
        let baseline = sine(1024, 16_000, 440.0);
        let mut repeat = baseline.clone();
        let mut counterfactual = baseline.clone();
        for (index, sample) in repeat.iter_mut().enumerate() {
            *sample += (index as f32 * 0.013).sin() * 0.0005;
        }
        for (index, sample) in counterfactual.iter_mut().enumerate() {
            *sample += (index as f32 * 0.013).sin() * 0.0005;
        }
        let repeat_receipts = [DIGEST_A];
        let report = estimate_counterfactual_influence(
            view(&baseline),
            &[view(&repeat)],
            view(&counterfactual),
            context(&repeat_receipts),
            &calibrated_config(),
        )
        .expect("valid noisy repeat control");

        assert_eq!(report.verdict, InfluenceVerdict::NoDetectableInfluence);
        assert_eq!(report.detected_channel_windows, 0);
    }

    #[test]
    fn insufficient_calibration_abstains_but_retains_measurements() {
        let baseline = sine(1024, 16_000, 440.0);
        let repeat = baseline.clone();
        let mut counterfactual = baseline.clone();
        counterfactual[256..384].fill(0.0);
        let repeat_receipts = [DIGEST_A];
        let mut config = calibrated_config();
        config.calibration_id = None;
        let report = estimate_counterfactual_influence(
            view(&baseline),
            &[view(&repeat)],
            view(&counterfactual),
            context(&repeat_receipts),
            &config,
        )
        .expect("valid exploratory run");

        assert_eq!(report.verdict, InfluenceVerdict::Abstain);
        assert!(report.detected_channel_windows > 0);
        assert!(!report.measurements.is_empty());
    }

    #[test]
    fn stereo_channels_receive_independent_influence_intervals() {
        let mono = sine(1024, 16_000, 440.0);
        let mut baseline = Vec::with_capacity(mono.len() * 2);
        let mut repeat = Vec::with_capacity(mono.len() * 2);
        let mut counterfactual = Vec::with_capacity(mono.len() * 2);
        for (index, sample) in mono.iter().copied().enumerate() {
            baseline.extend([sample, sample * 0.5]);
            repeat.extend([sample, sample * 0.5]);
            counterfactual.extend([
                sample,
                if (256..384).contains(&index) {
                    0.0
                } else {
                    sample * 0.5
                },
            ]);
        }
        let stereo = |samples: &[f32]| AudioView {
            interleaved_samples: samples,
            sample_rate_hz: 16_000,
            channels: 2,
        };
        let repeat_receipts = [DIGEST_A];
        let report = estimate_counterfactual_influence(
            stereo(&baseline),
            &[stereo(&repeat)],
            stereo(&counterfactual),
            context(&repeat_receipts),
            &calibrated_config(),
        )
        .expect("valid stereo counterfactual");

        assert!(
            !report.support_intervals.iter().any(|interval| interval.channel == 0)
        );
        assert!(report.support_intervals.iter().any(|interval| interval.channel == 1));
    }

    #[test]
    fn report_replay_rejects_changed_audio_or_context_commitment() {
        let baseline = sine(1024, 16_000, 440.0);
        let repeat = baseline.clone();
        let mut counterfactual = baseline.clone();
        counterfactual[256..384].fill(0.0);
        let repeat_receipts = [DIGEST_A];
        let report = estimate_counterfactual_influence(
            view(&baseline),
            &[view(&repeat)],
            view(&counterfactual),
            context(&repeat_receipts),
            &calibrated_config(),
        )
        .unwrap();
        let mut changed = counterfactual.clone();
        changed[10] = 0.5;
        assert!(!verify_counterfactual_influence_report(
            &report,
            view(&baseline),
            &[view(&repeat)],
            view(&changed)
        )
        .unwrap());
        let mut tampered = report.clone();
        tampered.identity.renderer_profile_sha256 = DIGEST_E.to_owned();
        assert!(!verify_counterfactual_influence_report(
            &tampered,
            view(&baseline),
            &[view(&repeat)],
            view(&counterfactual)
        )
        .unwrap());
    }

    #[test]
    fn malformed_context_and_no_op_interventions_reject() {
        let baseline = sine(1024, 16_000, 440.0);
        let repeat = baseline.clone();
        let repeat_receipts = [DIGEST_A];
        let mut invalid = context(&repeat_receipts);
        invalid.source_event_id = "";
        assert_eq!(
            estimate_counterfactual_influence(
                view(&baseline),
                &[view(&repeat)],
                view(&baseline),
                invalid,
                &calibrated_config()
            )
            .unwrap_err(),
            InfluenceInputError::InvalidEventId
        );

        let mut no_op = context(&repeat_receipts);
        no_op.intervention = CounterfactualIntervention::Transpose { semitones: 0 };
        assert_eq!(
            estimate_counterfactual_influence(
                view(&baseline),
                &[view(&repeat)],
                view(&baseline),
                no_op,
                &calibrated_config()
            )
            .unwrap_err(),
            InfluenceInputError::NoOpIntervention
        );
    }
}
