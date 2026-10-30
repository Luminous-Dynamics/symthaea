// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Counterfactual, paired-render acoustic influence measurements.
//!
//! This module measures the empirical difference between two exact PCM
//! artifacts associated with a declared single-event intervention. It does
//! NOT infer exclusive sample ownership, prove that the declared symbolic
//! intervention was isolated correctly, or itself execute a renderer.
//!
//! Repeat-render controls establish an empirical measurement floor for a
//! particular anchor output, format, analysis configuration and renderer
//! artifact set. A calibration becomes usable only after replaying its exact
//! no-op controls. An influence report likewise replays from the exact input
//! artifacts and PCM buffers. Digests provide content integrity, not signer
//! identity or remote execution attestation.
//!
//! The first implementation deliberately uses phase-sensitive waveform RMS
//! alongside phase-insensitive spectral-magnitude distance. A later
//! perceptual representation may be added as a separate reported channel;
//! it must not replace these raw measurements or make sample ownership claims.

use crate::audio_preservation::{AudioView, FrameInterval};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;

pub const COUNTERFACTUAL_INFLUENCE_VERSION: &str =
    "mel-mr-eval-017-counterfactual-influence-v0.1";
pub const REPEATABILITY_CALIBRATION_VERSION: &str =
    "mel-mr-eval-017-repeatability-calibration-v0.1";
const CALIBRATION_DOMAIN: &[u8] = b"luminous-dynamics-mel-mr-eval-017-repeatability-v1\0";
const INFLUENCE_DOMAIN: &[u8] = b"luminous-dynamics-mel-mr-eval-017-influence-v1\0";
const MINIMUM_NOOP_RENDERS: usize = 3;
const MAX_EVENT_ID_BYTES: usize = 256;
const MAX_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InfluenceAnalysisConfig {
    /// STFT window length, measured in audio frames.
    pub window_size_frames: usize,
    /// Distance between adjacent window starts.
    pub hop_size_frames: usize,
    /// Absolute lower bound for a measurable waveform RMS difference.
    pub minimum_rms_delta: f32,
    /// Absolute lower bound for a measurable normalized spectral difference.
    pub minimum_spectral_distance: f32,
    /// A counterfactual change must exceed this multiple of the largest
    /// measured no-op delta for the matching window/channel.
    pub noise_floor_multiplier: f32,
}

impl Default for InfluenceAnalysisConfig {
    fn default() -> Self {
        Self {
            window_size_frames: 1024,
            hop_size_frames: 256,
            // Deliberately zero until a renderer-specific calibration is supplied.
            // The empirical no-op floor is still required for every report.
            minimum_rms_delta: 0.0,
            minimum_spectral_distance: 0.0,
            noise_floor_multiplier: 3.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventIntervention {
    Remove,
    Transpose { semitones: i16 },
    ShiftOnset { delta_ticks: i64 },
    ChangeDuration { delta_ticks: i64 },
    ChangeVelocity { delta: i16 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InfluenceDisposition {
    /// At least one channel/window exceeded both its absolute and empirical
    /// repeatability thresholds in one of the independent measured channels.
    EffectDetected,
    /// The declared paired outputs did not exceed the calibrated thresholds.
    /// This does not prove that the intervention had no audible effect.
    NoEffectAboveNoiseFloor,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InfluenceNoiseCell {
    pub start_frame: usize,
    pub end_frame: usize,
    pub channel: usize,
    pub maximum_noop_rms_delta: f32,
    pub maximum_noop_spectral_distance: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepeatabilityCalibration {
    pub version: String,
    pub sample_rate_hz: u32,
    pub channels: usize,
    pub total_frames: usize,
    pub analysis: InfluenceAnalysisConfig,
    pub anchor_audio_sha256: String,
    /// Exact PCM digests for every no-op rerender, in supplied order.
    pub noop_audio_sha256: Vec<String>,
    /// A conservative per-window/channel maximum over the supplied no-op runs.
    pub cells: Vec<InfluenceNoiseCell>,
    pub receipt_sha256: String,
}

/// An opaque token that can only be constructed by replay-verifying a
/// RepeatabilityCalibration against its exact anchor and no-op render buffers.
#[derive(Debug, Clone)]
pub struct VerifiedRepeatabilityCalibration {
    calibration: RepeatabilityCalibration,
}

impl VerifiedRepeatabilityCalibration {
    pub fn calibration(&self) -> &RepeatabilityCalibration {
        &self.calibration
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InfluenceWindowMeasurement {
    pub start_frame: usize,
    pub end_frame: usize,
    pub channel: usize,
    pub rms_delta: f32,
    pub peak_sample_delta: f32,
    pub spectral_magnitude_distance: f32,
    pub calibrated_rms_threshold: f32,
    pub calibrated_spectral_threshold: f32,
    pub effect_detected: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CounterfactualInfluenceReport {
    pub version: String,
    pub source_event_id: String,
    pub intervention: EventIntervention,
    pub source_score_sha256: String,
    pub performed_event_map_sha256: String,
    pub baseline_schedule_sha256: String,
    pub counterfactual_schedule_sha256: String,
    pub renderer_identity_sha256: String,
    pub renderer_config_sha256: String,
    pub baseline_audio_sha256: String,
    pub counterfactual_audio_sha256: String,
    pub repeatability_calibration_sha256: String,
    pub sample_rate_hz: u32,
    pub channels: usize,
    pub total_frames: usize,
    pub analysis: InfluenceAnalysisConfig,
    pub total_channel_windows: usize,
    pub affected_channel_windows: usize,
    /// Merged frame intervals where at least one channel/window exceeded its
    /// calibrated threshold. These are measured support regions, not ownership.
    pub measured_influence_intervals: Vec<FrameInterval>,
    pub disposition: InfluenceDisposition,
    pub receipt_sha256: String,
}

/// Exact artifacts and audio supplied to the paired measurement. The caller
/// must create the counterfactual schedule by changing only the declared event;
/// this module binds the bytes but does not independently parse the schedule.
pub struct CounterfactualInfluenceInput<'a> {
    pub source_event_id: &'a str,
    pub intervention: EventIntervention,
    pub source_score_bytes: &'a [u8],
    pub performed_event_map_bytes: &'a [u8],
    pub baseline_schedule_bytes: &'a [u8],
    pub counterfactual_schedule_bytes: &'a [u8],
    pub renderer_identity_bytes: &'a [u8],
    pub renderer_config_bytes: &'a [u8],
    pub baseline_audio: &'a AudioView<'a>,
    pub counterfactual_audio: &'a AudioView<'a>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CounterfactualInfluenceError {
    InvalidConfig(&'static str),
    ZeroSampleRate,
    ZeroChannels,
    EmptyAudio,
    InterleavedLengthNotDivisibleByChannels,
    NonFiniteSample,
    AudioLayoutMismatch,
    WindowLongerThanAudio,
    TooFewNoopRenders,
    InvalidCalibrationReceipt,
    CalibrationReplayMismatch,
    CalibrationAnchorMismatch,
    CalibrationConfigMismatch,
    InvalidEventId,
    InvalidIntervention,
    EmptyArtifact(&'static str),
    ArtifactTooLarge(&'static str),
    IdenticalSchedules,
    SerializationFailure,
    InvalidInfluenceReceipt,
    InfluenceReplayMismatch,
    WindowIndexMismatch,
}

impl std::fmt::Display for CounterfactualInfluenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(reason) => write!(f, "invalid influence analysis config: {reason}"),
            Self::ZeroSampleRate => write!(f, "sample rate must be non-zero"),
            Self::ZeroChannels => write!(f, "channel count must be non-zero"),
            Self::EmptyAudio => write!(f, "audio must contain at least one frame"),
            Self::InterleavedLengthNotDivisibleByChannels => {
                write!(f, "interleaved sample count is not divisible by channel count")
            }
            Self::NonFiniteSample => write!(f, "audio contains a non-finite sample"),
            Self::AudioLayoutMismatch => {
                write!(f, "paired audio must have identical rate, channels and frame count")
            }
            Self::WindowLongerThanAudio => write!(f, "analysis window is longer than audio"),
            Self::TooFewNoopRenders => write!(
                f,
                "at least {MINIMUM_NOOP_RENDERS} no-op rerenders are required"
            ),
            Self::InvalidCalibrationReceipt => {
                write!(f, "repeatability calibration receipt is invalid")
            }
            Self::CalibrationReplayMismatch => {
                write!(f, "repeatability calibration does not replay from supplied controls")
            }
            Self::CalibrationAnchorMismatch => {
                write!(f, "baseline output differs from the calibrated anchor output")
            }
            Self::CalibrationConfigMismatch => {
                write!(f, "measurement config differs from the frozen calibration config")
            }
            Self::InvalidEventId => write!(f, "invalid source event ID"),
            Self::InvalidIntervention => write!(f, "intervention is a no-op or has invalid parameters"),
            Self::EmptyArtifact(name) => write!(f, "artifact is empty: {name}"),
            Self::ArtifactTooLarge(name) => write!(f, "artifact exceeds size limit: {name}"),
            Self::IdenticalSchedules => {
                write!(f, "baseline and counterfactual schedules are byte-identical")
            }
            Self::SerializationFailure => write!(f, "could not serialize influence evidence"),
            Self::InvalidInfluenceReceipt => write!(f, "influence report receipt is invalid"),
            Self::InfluenceReplayMismatch => {
                write!(f, "influence report does not replay from supplied artifacts")
            }
            Self::WindowIndexMismatch => {
                write!(f, "calibration cells do not match the expected window/channel layout")
            }
        }
    }
}

impl std::error::Error for CounterfactualInfluenceError {}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn audio_sha256(audio: &AudioView<'_>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous-dynamics-mel-mr-eval-017-audio-v1\0");
    hasher.update(audio.sample_rate_hz.to_le_bytes());
    hasher.update((audio.channels as u64).to_le_bytes());
    hasher.update((audio.interleaved_samples.len() as u64).to_le_bytes());
    for sample in audio.interleaved_samples {
        hasher.update(sample.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn seal_digest<T: Serialize>(
    value: &T,
    receipt_field: &mut String,
    domain: &[u8],
) -> Result<(), CounterfactualInfluenceError> {
    receipt_field.clear();
    let bytes = serde_json::to_vec(value)
        .map_err(|_| CounterfactualInfluenceError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    *receipt_field = format!("{:x}", hasher.finalize());
    Ok(())
}

fn verify_digest<T: Serialize>(
    value: &T,
    receipt: &str,
    domain: &[u8],
) -> Result<bool, CounterfactualInfluenceError> {
    if receipt.len() != 64
        || !receipt
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Ok(false);
    }
    let bytes = serde_json::to_vec(value)
        .map_err(|_| CounterfactualInfluenceError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()) == receipt)
}

fn validate_config(
    config: &InfluenceAnalysisConfig,
) -> Result<(), CounterfactualInfluenceError> {
    if config.window_size_frames < 16 {
        return Err(CounterfactualInfluenceError::InvalidConfig(
            "window_size_frames must be at least 16",
        ));
    }
    if config.hop_size_frames == 0 || config.hop_size_frames > config.window_size_frames {
        return Err(CounterfactualInfluenceError::InvalidConfig(
            "hop_size_frames must be in 1..=window_size_frames",
        ));
    }
    for value in [
        config.minimum_rms_delta,
        config.minimum_spectral_distance,
        config.noise_floor_multiplier,
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(CounterfactualInfluenceError::InvalidConfig(
                "thresholds must be finite and non-negative",
            ));
        }
    }
    if config.noise_floor_multiplier < 1.0 {
        return Err(CounterfactualInfluenceError::InvalidConfig(
            "noise_floor_multiplier must be at least 1",
        ));
    }
    Ok(())
}

fn validate_audio(audio: &AudioView<'_>) -> Result<usize, CounterfactualInfluenceError> {
    if audio.sample_rate_hz == 0 {
        return Err(CounterfactualInfluenceError::ZeroSampleRate);
    }
    if audio.channels == 0 {
        return Err(CounterfactualInfluenceError::ZeroChannels);
    }
    if audio.interleaved_samples.is_empty() {
        return Err(CounterfactualInfluenceError::EmptyAudio);
    }
    if audio.interleaved_samples.len() % audio.channels != 0 {
        return Err(CounterfactualInfluenceError::InterleavedLengthNotDivisibleByChannels);
    }
    if audio.interleaved_samples.iter().any(|sample| !sample.is_finite()) {
        return Err(CounterfactualInfluenceError::NonFiniteSample);
    }
    Ok(audio.interleaved_samples.len() / audio.channels)
}

fn validate_pair(
    left: &AudioView<'_>,
    right: &AudioView<'_>,
) -> Result<usize, CounterfactualInfluenceError> {
    let left_frames = validate_audio(left)?;
    let right_frames = validate_audio(right)?;
    if left.sample_rate_hz != right.sample_rate_hz
        || left.channels != right.channels
        || left_frames != right_frames
    {
        return Err(CounterfactualInfluenceError::AudioLayoutMismatch);
    }
    Ok(left_frames)
}

fn frame_windows(
    total_frames: usize,
    config: &InfluenceAnalysisConfig,
) -> Result<Vec<FrameInterval>, CounterfactualInfluenceError> {
    if config.window_size_frames > total_frames {
        return Err(CounterfactualInfluenceError::WindowLongerThanAudio);
    }
    let mut windows = Vec::new();
    let mut start = 0usize;
    loop {
        let end = start
            .checked_add(config.window_size_frames)
            .ok_or(CounterfactualInfluenceError::InvalidConfig("window overflow"))?;
        if end > total_frames {
            break;
        }
        windows.push(FrameInterval {
            start_frame: start,
            end_frame: end,
        });
        match start.checked_add(config.hop_size_frames) {
            Some(next) if next > start => start = next,
            _ => break,
        }
    }
    let final_start = total_frames - config.window_size_frames;
    if windows
        .last()
        .is_none_or(|window| window.start_frame != final_start)
    {
        windows.push(FrameInterval {
            start_frame: final_start,
            end_frame: total_frames,
        });
    }
    Ok(windows)
}

#[derive(Debug, Clone, Copy, Default)]
struct PairMetrics {
    rms_delta: f32,
    peak_sample_delta: f32,
    spectral_magnitude_distance: f32,
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|index| {
            0.5 - 0.5 * (std::f32::consts::TAU * index as f32 / size as f32).cos()
        })
        .collect()
}

fn compare_channel_window(
    baseline: &AudioView<'_>,
    counterfactual: &AudioView<'_>,
    window: FrameInterval,
    channel: usize,
    hann: &[f32],
    fft: &dyn Fft<f32>,
) -> PairMetrics {
    let size = window.end_frame - window.start_frame;
    let mut left = Vec::with_capacity(size);
    let mut right = Vec::with_capacity(size);
    let mut squared_delta = 0.0f64;
    let mut peak_delta = 0.0f32;

    for offset in 0..size {
        let frame = window.start_frame + offset;
        let index = frame * baseline.channels + channel;
        let a = baseline.interleaved_samples[index];
        let b = counterfactual.interleaved_samples[index];
        let delta = (a - b).abs();
        squared_delta += (delta as f64) * (delta as f64);
        peak_delta = peak_delta.max(delta);
        left.push(Complex::new(a * hann[offset], 0.0));
        right.push(Complex::new(b * hann[offset], 0.0));
    }

    fft.process(&mut left);
    fft.process(&mut right);

    let mut spectral_difference = 0.0f64;
    let mut total_magnitude = 0.0f64;
    for index in 0..=(size / 2) {
        let left_magnitude = left[index].norm() as f64;
        let right_magnitude = right[index].norm() as f64;
        spectral_difference += (left_magnitude - right_magnitude).abs();
        total_magnitude += left_magnitude + right_magnitude;
    }

    PairMetrics {
        rms_delta: (squared_delta / size as f64).sqrt() as f32,
        peak_sample_delta: peak_delta,
        spectral_magnitude_distance: if total_magnitude <= f64::EPSILON {
            0.0
        } else {
            (spectral_difference / total_magnitude).clamp(0.0, 1.0) as f32
        },
    }
}

fn analyze_pair(
    baseline: &AudioView<'_>,
    counterfactual: &AudioView<'_>,
    windows: &[FrameInterval],
) -> Vec<PairMetrics> {
    let frame_size = windows[0].end_frame - windows[0].start_frame;
    let hann = hann_window(frame_size);
    let fft = FftPlanner::<f32>::new().plan_fft_forward(frame_size);
    let mut measurements = Vec::with_capacity(windows.len() * baseline.channels);
    for window in windows {
        for channel in 0..baseline.channels {
            measurements.push(compare_channel_window(
                baseline,
                counterfactual,
                *window,
                channel,
                &hann,
                fft.as_ref(),
            ));
        }
    }
    measurements
}

fn merge_intervals(mut intervals: Vec<FrameInterval>) -> Vec<FrameInterval> {
    intervals.sort_by_key(|interval| (interval.start_frame, interval.end_frame));
    let mut merged: Vec<FrameInterval> = Vec::new();
    for interval in intervals {
        if let Some(last) = merged.last_mut()
            && interval.start_frame <= last.end_frame
        {
            last.end_frame = last.end_frame.max(interval.end_frame);
        } else {
            merged.push(interval);
        }
    }
    merged
}

fn seal_calibration(
    mut calibration: RepeatabilityCalibration,
) -> Result<RepeatabilityCalibration, CounterfactualInfluenceError> {
    calibration.receipt_sha256.clear();
    seal_digest(
        &calibration,
        &mut calibration.receipt_sha256,
        CALIBRATION_DOMAIN,
    )?;
    Ok(calibration)
}

/// Build an empirical no-op measurement floor from the exact same-render
/// anchor and at least three independent repeats with identical format and
/// duration. Each cell uses the largest observed no-op delta; this is a
/// conservative fixture-level floor, not a population quantile or guarantee.
pub fn calibrate_repeatability(
    anchor: &AudioView<'_>,
    noop_renders: &[AudioView<'_>],
    analysis: &InfluenceAnalysisConfig,
) -> Result<RepeatabilityCalibration, CounterfactualInfluenceError> {
    validate_config(analysis)?;
    let total_frames = validate_audio(anchor)?;
    if noop_renders.len() < MINIMUM_NOOP_RENDERS {
        return Err(CounterfactualInfluenceError::TooFewNoopRenders);
    }
    let windows = frame_windows(total_frames, analysis)?;
    let mut cells = vec![
        InfluenceNoiseCell {
            start_frame: 0,
            end_frame: 0,
            channel: 0,
            maximum_noop_rms_delta: 0.0,
            maximum_noop_spectral_distance: 0.0,
        };
        windows.len() * anchor.channels
    ];

    for (window_index, window) in windows.iter().enumerate() {
        for channel in 0..anchor.channels {
            let cell = &mut cells[window_index * anchor.channels + channel];
            cell.start_frame = window.start_frame;
            cell.end_frame = window.end_frame;
            cell.channel = channel;
        }
    }

    let mut noop_audio_sha256 = Vec::with_capacity(noop_renders.len());
    for noop in noop_renders {
        validate_pair(anchor, noop)?;
        let digest = audio_sha256(noop);
        let metrics = analyze_pair(anchor, noop, &windows);
        for (cell, metric) in cells.iter_mut().zip(metrics) {
            cell.maximum_noop_rms_delta = cell.maximum_noop_rms_delta.max(metric.rms_delta);
            cell.maximum_noop_spectral_distance = cell
                .maximum_noop_spectral_distance
                .max(metric.spectral_magnitude_distance);
        }
        noop_audio_sha256.push(digest);
    }

    seal_calibration(RepeatabilityCalibration {
        version: REPEATABILITY_CALIBRATION_VERSION.to_owned(),
        sample_rate_hz: anchor.sample_rate_hz,
        channels: anchor.channels,
        total_frames,
        analysis: analysis.clone(),
        anchor_audio_sha256: audio_sha256(anchor),
        noop_audio_sha256,
        cells,
        receipt_sha256: String::new(),
    })
}

/// Replay a calibration from the exact anchor and no-op PCM outputs. This
/// verifier creates an opaque token required by the measurement API so a
/// caller cannot accidentally treat a merely deserialized calibration as
/// already replay-verified.
pub fn verify_repeatability_calibration(
    calibration: &RepeatabilityCalibration,
    anchor: &AudioView<'_>,
    noop_renders: &[AudioView<'_>],
) -> Result<VerifiedRepeatabilityCalibration, CounterfactualInfluenceError> {
    if !verify_digest(calibration, &calibration.receipt_sha256, CALIBRATION_DOMAIN)? {
        return Err(CounterfactualInfluenceError::InvalidCalibrationReceipt);
    }
    let expected = calibrate_repeatability(anchor, noop_renders, &calibration.analysis)?;
    if &expected != calibration {
        return Err(CounterfactualInfluenceError::CalibrationReplayMismatch);
    }
    Ok(VerifiedRepeatabilityCalibration {
        calibration: calibration.clone(),
    })
}

fn validate_intervention(
    intervention: &EventIntervention,
) -> Result<(), CounterfactualInfluenceError> {
    match intervention {
        EventIntervention::Remove => Ok(()),
        EventIntervention::Transpose { semitones } if *semitones != 0 => Ok(()),
        EventIntervention::ShiftOnset { delta_ticks } if *delta_ticks != 0 => Ok(()),
        EventIntervention::ChangeDuration { delta_ticks } if *delta_ticks != 0 => Ok(()),
        EventIntervention::ChangeVelocity { delta } if *delta != 0 => Ok(()),
        _ => Err(CounterfactualInfluenceError::InvalidIntervention),
    }
}

fn validate_input(
    input: &CounterfactualInfluenceInput<'_>,
    calibration: &RepeatabilityCalibration,
) -> Result<usize, CounterfactualInfluenceError> {
    validate_intervention(&input.intervention)?;
    if input.source_event_id.trim().is_empty()
        || input.source_event_id.len() > MAX_EVENT_ID_BYTES
        || input.source_event_id.chars().any(char::is_control)
    {
        return Err(CounterfactualInfluenceError::InvalidEventId);
    }
    for (name, bytes) in [
        ("source_score", input.source_score_bytes),
        ("performed_event_map", input.performed_event_map_bytes),
        ("baseline_schedule", input.baseline_schedule_bytes),
        ("counterfactual_schedule", input.counterfactual_schedule_bytes),
        ("renderer_identity", input.renderer_identity_bytes),
        ("renderer_config", input.renderer_config_bytes),
    ] {
        if bytes.is_empty() {
            return Err(CounterfactualInfluenceError::EmptyArtifact(name));
        }
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(CounterfactualInfluenceError::ArtifactTooLarge(name));
        }
    }
    if input.baseline_schedule_bytes == input.counterfactual_schedule_bytes {
        return Err(CounterfactualInfluenceError::IdenticalSchedules);
    }
    let frames = validate_pair(input.baseline_audio, input.counterfactual_audio)?;
    if input.baseline_audio.sample_rate_hz != calibration.sample_rate_hz
        || input.baseline_audio.channels != calibration.channels
        || frames != calibration.total_frames
    {
        return Err(CounterfactualInfluenceError::AudioLayoutMismatch);
    }
    if audio_sha256(input.baseline_audio) != calibration.anchor_audio_sha256 {
        return Err(CounterfactualInfluenceError::CalibrationAnchorMismatch);
    }
    Ok(frames)
}

fn seal_report(
    mut report: CounterfactualInfluenceReport,
) -> Result<CounterfactualInfluenceReport, CounterfactualInfluenceError> {
    report.receipt_sha256.clear();
    seal_digest(&report, &mut report.receipt_sha256, INFLUENCE_DOMAIN)?;
    Ok(report)
}

/// Measure the paired output difference using an already replay-verified
/// no-op calibration. The result is an empirical influence estimate for the
/// declared intervention and renderer configuration, not exclusive ownership
/// of affected audio samples.
pub fn measure_counterfactual_influence(
    input: &CounterfactualInfluenceInput<'_>,
    verified_calibration: &VerifiedRepeatabilityCalibration,
) -> Result<CounterfactualInfluenceReport, CounterfactualInfluenceError> {
    let calibration = &verified_calibration.calibration;
    validate_config(&calibration.analysis)?;
    validate_input(input, calibration)?;
    let windows = frame_windows(calibration.total_frames, &calibration.analysis)?;
    if calibration.cells.len() != windows.len() * calibration.channels {
        return Err(CounterfactualInfluenceError::WindowIndexMismatch);
    }
    for (index, (cell, window)) in calibration.cells.iter().zip(&windows).enumerate() {
        let expected_channel = index % calibration.channels;
        if cell.start_frame != window.start_frame
            || cell.end_frame != window.end_frame
            || cell.channel != expected_channel
        {
            return Err(CounterfactualInfluenceError::WindowIndexMismatch);
        }
    }

    let metrics = analyze_pair(input.baseline_audio, input.counterfactual_audio, &windows);
    let mut measurements = Vec::with_capacity(metrics.len());
    let mut affected = Vec::new();
    for (index, metric) in metrics.into_iter().enumerate() {
        let cell = &calibration.cells[index];
        let rms_threshold = calibration
            .analysis
            .minimum_rms_delta
            .max(cell.maximum_noop_rms_delta * calibration.analysis.noise_floor_multiplier);
        let spectral_threshold = calibration
            .analysis
            .minimum_spectral_distance
            .max(
                cell.maximum_noop_spectral_distance
                    * calibration.analysis.noise_floor_multiplier,
            );
        let effect_detected = metric.rms_delta > rms_threshold
            || metric.spectral_magnitude_distance > spectral_threshold;
        if effect_detected {
            affected.push(FrameInterval {
                start_frame: cell.start_frame,
                end_frame: cell.end_frame,
            });
        }
        measurements.push(InfluenceWindowMeasurement {
            start_frame: cell.start_frame,
            end_frame: cell.end_frame,
            channel: cell.channel,
            rms_delta: metric.rms_delta,
            peak_sample_delta: metric.peak_sample_delta,
            spectral_magnitude_distance: metric.spectral_magnitude_distance,
            calibrated_rms_threshold: rms_threshold,
            calibrated_spectral_threshold: spectral_threshold,
            effect_detected,
        });
    }

    let affected_channel_windows = measurements
        .iter()
        .filter(|measurement| measurement.effect_detected)
        .count();
    let disposition = if affected_channel_windows > 0 {
        InfluenceDisposition::EffectDetected
    } else {
        InfluenceDisposition::NoEffectAboveNoiseFloor
    };
    seal_report(CounterfactualInfluenceReport {
        version: COUNTERFACTUAL_INFLUENCE_VERSION.to_owned(),
        source_event_id: input.source_event_id.to_owned(),
        intervention: input.intervention.clone(),
        source_score_sha256: sha256_hex(input.source_score_bytes),
        performed_event_map_sha256: sha256_hex(input.performed_event_map_bytes),
        baseline_schedule_sha256: sha256_hex(input.baseline_schedule_bytes),
        counterfactual_schedule_sha256: sha256_hex(input.counterfactual_schedule_bytes),
        renderer_identity_sha256: sha256_hex(input.renderer_identity_bytes),
        renderer_config_sha256: sha256_hex(input.renderer_config_bytes),
        baseline_audio_sha256: audio_sha256(input.baseline_audio),
        counterfactual_audio_sha256: audio_sha256(input.counterfactual_audio),
        repeatability_calibration_sha256: calibration.receipt_sha256.clone(),
        sample_rate_hz: calibration.sample_rate_hz,
        channels: calibration.channels,
        total_frames: calibration.total_frames,
        analysis: calibration.analysis.clone(),
        total_channel_windows: measurements.len(),
        affected_channel_windows,
        measured_influence_intervals: merge_intervals(affected),
        disposition,
        receipt_sha256: String::new(),
    })
}

/// Replay the influence measurements and receipt from the exact score,
/// event-map, schedule, renderer, configuration, calibration and PCM inputs.
pub fn verify_counterfactual_influence(
    report: &CounterfactualInfluenceReport,
    input: &CounterfactualInfluenceInput<'_>,
    verified_calibration: &VerifiedRepeatabilityCalibration,
) -> Result<bool, CounterfactualInfluenceError> {
    if !verify_digest(report, &report.receipt_sha256, INFLUENCE_DOMAIN)? {
        return Err(CounterfactualInfluenceError::InvalidInfluenceReceipt);
    }
    let expected = measure_counterfactual_influence(input, verified_calibration)?;
    if &expected != report {
        return Err(CounterfactualInfluenceError::InfluenceReplayMismatch);
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> InfluenceAnalysisConfig {
        InfluenceAnalysisConfig {
            window_size_frames: 32,
            hop_size_frames: 16,
            minimum_rms_delta: 1.0e-6,
            minimum_spectral_distance: 1.0e-4,
            noise_floor_multiplier: 2.0,
        }
    }

    fn audio(samples: &[f32]) -> AudioView<'_> {
        AudioView {
            interleaved_samples: samples,
            sample_rate_hz: 48_000,
            channels: 1,
        }
    }

    fn calibration_for<'a>(
        anchor: &AudioView<'a>,
        repeats: &[AudioView<'a>],
    ) -> VerifiedRepeatabilityCalibration {
        let calibration = calibrate_repeatability(anchor, repeats, &config()).unwrap();
        verify_repeatability_calibration(&calibration, anchor, repeats).unwrap()
    }

    fn input<'a>(
        baseline: &'a AudioView<'a>,
        candidate: &'a AudioView<'a>,
    ) -> CounterfactualInfluenceInput<'a> {
        CounterfactualInfluenceInput {
            source_event_id: "score-event-17",
            intervention: EventIntervention::Remove,
            source_score_bytes: br#"{"score":"frozen"}"#,
            performed_event_map_bytes: br#"{"map":"frozen"}"#,
            baseline_schedule_bytes: b"schedule-with-event-17",
            counterfactual_schedule_bytes: b"schedule-without-event-17",
            renderer_identity_bytes: b"renderer-fixture-v1",
            renderer_config_bytes: b"config-seed-4-samplerate-48000",
            baseline_audio: baseline,
            counterfactual_audio: candidate,
        }
    }

    #[test]
    fn calibration_requires_three_noop_rerenders_and_replays() {
        let base = vec![0.0f32; 256];
        let anchor = audio(&base);
        let one = [audio(&base)];
        assert_eq!(
            calibrate_repeatability(&anchor, &one, &config()).unwrap_err(),
            CounterfactualInfluenceError::TooFewNoopRenders
        );

        let repeats = [audio(&base), audio(&base), audio(&base)];
        let calibration = calibrate_repeatability(&anchor, &repeats, &config()).unwrap();
        assert_eq!(calibration.noop_audio_sha256.len(), 3);
        assert_eq!(
            calibration.cells.iter().map(|cell| cell.maximum_noop_rms_delta).fold(0.0f32, f32::max),
            0.0
        );
        assert!(verify_repeatability_calibration(&calibration, &anchor, &repeats).is_ok());
    }

    #[test]
    fn localized_intervention_produces_a_replayable_measured_support() {
        let base = vec![0.0f32; 256];
        let mut edited = base.clone();
        for sample in &mut edited[80..112] {
            *sample = 0.35;
        }
        let anchor = audio(&base);
        let candidate = audio(&edited);
        let repeats = [audio(&base), audio(&base), audio(&base)];
        let calibration = calibration_for(&anchor, &repeats);
        let experiment = input(&anchor, &candidate);

        let report = measure_counterfactual_influence(&experiment, &calibration).unwrap();
        assert_eq!(report.source_event_id, "score-event-17");
        assert_eq!(report.disposition, InfluenceDisposition::EffectDetected);
        assert!(report.affected_channel_windows > 0);
        assert!(report.measured_influence_intervals.iter().any(|interval| {
            interval.start_frame <= 80 && interval.end_frame >= 112
        }));
        assert!(verify_counterfactual_influence(&report, &experiment, &calibration).unwrap());
    }

    #[test]
    fn no_op_noise_floor_prevents_a_small_repeatability_level_change_from_counting_as_effect() {
        let base = vec![0.0f32; 256];
        let mut noop_shift = base.clone();
        for sample in &mut noop_shift[80..112] {
            *sample = 0.02;
        }
        let mut smaller_candidate = base.clone();
        for sample in &mut smaller_candidate[80..112] {
            *sample = 0.005;
        }
        let anchor = audio(&base);
        let repeats = [
            audio(&noop_shift),
            audio(&noop_shift),
            audio(&noop_shift),
        ];
        let calibration = calibration_for(&anchor, &repeats);
        let candidate = audio(&smaller_candidate);
        let experiment = input(&anchor, &candidate);

        let report = measure_counterfactual_influence(&experiment, &calibration).unwrap();
        assert_eq!(
            report.disposition,
            InfluenceDisposition::NoEffectAboveNoiseFloor
        );
        assert_eq!(report.affected_channel_windows, 0);
    }

    #[test]
    fn stale_calibration_anchor_is_rejected() {
        let base = vec![0.0f32; 256];
        let changed = vec![0.1f32; 256];
        let anchor = audio(&base);
        let repeats = [audio(&base), audio(&base), audio(&base)];
        let calibration = calibration_for(&anchor, &repeats);
        let wrong_baseline = audio(&changed);
        let counterfactual = audio(&base);
        let experiment = input(&wrong_baseline, &counterfactual);

        assert_eq!(
            measure_counterfactual_influence(&experiment, &calibration).unwrap_err(),
            CounterfactualInfluenceError::CalibrationAnchorMismatch
        );
    }

    #[test]
    fn byte_identical_schedule_is_rejected_as_a_noop_intervention() {
        let base = vec![0.0f32; 256];
        let anchor = audio(&base);
        let repeats = [audio(&base), audio(&base), audio(&base)];
        let calibration = calibration_for(&anchor, &repeats);
        let candidate = audio(&base);
        let mut experiment = input(&anchor, &candidate);
        experiment.counterfactual_schedule_bytes = experiment.baseline_schedule_bytes;

        assert_eq!(
            measure_counterfactual_influence(&experiment, &calibration).unwrap_err(),
            CounterfactualInfluenceError::IdenticalSchedules
        );
    }

    #[test]
    fn report_mutation_is_detected_and_exact_artifacts_are_bound() {
        let base = vec![0.0f32; 256];
        let mut changed = base.clone();
        changed[96] = 0.7;
        let anchor = audio(&base);
        let candidate = audio(&changed);
        let repeats = [audio(&base), audio(&base), audio(&base)];
        let calibration = calibration_for(&anchor, &repeats);
        let experiment = input(&anchor, &candidate);
        let mut report = measure_counterfactual_influence(&experiment, &calibration).unwrap();

        report.renderer_config_sha256 = sha256_hex(b"substituted-config");
        assert_eq!(
            verify_counterfactual_influence(&report, &experiment, &calibration).unwrap_err(),
            CounterfactualInfluenceError::InvalidInfluenceReceipt
        );
    }

    #[test]
    fn malformed_audio_and_analysis_config_fail_closed() {
        let empty = AudioView {
            interleaved_samples: &[],
            sample_rate_hz: 48_000,
            channels: 1,
        };
        assert_eq!(
            validate_audio(&empty).unwrap_err(),
            CounterfactualInfluenceError::EmptyAudio
        );
        let short_window = InfluenceAnalysisConfig {
            window_size_frames: 8,
            ..config()
        };
        assert!(validate_config(&short_window).is_err());
        let nan_samples = [0.0f32, f32::NAN, 0.0];
        let nan_audio = audio(&nan_samples);
        assert_eq!(
            validate_audio(&nan_audio).unwrap_err(),
            CounterfactualInfluenceError::NonFiniteSample
        );
    }
}
