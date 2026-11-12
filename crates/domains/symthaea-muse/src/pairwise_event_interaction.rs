// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Four-coalition, paired-render measurement of second-order event interaction.
//!
//! The PCM residual and spectral-magnitude residual are deliberately separate.
//! A non-zero magnitude residual can arise from phase interference even when
//! the underlying PCM renderer is additive. Neither residual alone proves a
//! nonlinear renderer, exclusive source ownership, or perceptual importance.
//!
//! Renderer receipt digests are bound, not verified here. Callers must replay
//! verify each receipt and establish that all eight renders were independent
//! executions under the same renderer profile and common-random-number policy.

use crate::audio_preservation::AudioView;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const PAIRWISE_EVENT_INTERACTION_VERSION: &str =
    "mel-mr-eval-018-pairwise-event-interaction-v0.1";
const MAX_FRAME_SIZE: usize = 1_048_576;
const EPSILON_RMS: f64 = 1.0e-12;

/// One renderer result in a 2×2 factorial intervention.
#[derive(Debug, Clone, Copy)]
pub struct CoalitionRender<'a> {
    /// Sorted, unique source-event identities retained in this coalition.
    pub active_event_ids: &'a [&'a str],
    /// Digest of the exact performed schedule delivered to the renderer.
    pub schedule_sha256: &'a str,
    /// Independently replay-verifiable performed-event-map receipt digest.
    pub performed_event_map_sha256: &'a str,
    /// Independently replay-verifiable renderer receipt digest.
    pub renderer_receipt_sha256: &'a str,
    pub audio: AudioView<'a>,
}

/// The four exact coalitions: Y00, Y10, Y01, Y11.
#[derive(Debug, Clone, Copy)]
pub struct PairwiseRenderQuartet<'a> {
    /// Y00 = render with all source events present.
    pub y00: CoalitionRender<'a>,
    /// Y10 = render with event A removed and all other events retained.
    pub y10: CoalitionRender<'a>,
    /// Y01 = render with event B removed and all other events retained.
    pub y01: CoalitionRender<'a>,
    /// Y11 = render with exactly A and B removed.
    pub y11: CoalitionRender<'a>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairwiseInteractionConfig {
    /// Analysis window in PCM frames.
    pub frame_size: usize,
    /// Window-start distance in PCM frames.
    pub hop_size: usize,
    /// Multiply the largest repeat-quartet deviation by this factor.
    pub noise_multiplier: f32,
    /// Absolute floor for normalized PCM second-difference RMS.
    pub minimum_pcm_interaction: f32,
    /// Absolute floor for normalized spectral-magnitude second difference.
    pub minimum_magnitude_interaction: f32,
    /// Minimum number of unique time windows for a classified result.
    pub minimum_covered_windows: usize,
    /// External frozen calibration artifact; absent means abstain.
    pub calibration_id: Option<String>,
}

impl Default for PairwiseInteractionConfig {
    fn default() -> Self {
        Self {
            frame_size: 1024,
            hop_size: 256,
            noise_multiplier: 3.0,
            minimum_pcm_interaction: 0.0001,
            minimum_magnitude_interaction: 0.01,
            minimum_covered_windows: 1,
            calibration_id: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoalitionCommitment {
    pub schedule_sha256: String,
    pub performed_event_map_sha256: String,
    pub renderer_receipt_sha256: String,
    pub audio_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairwiseInteractionIdentity {
    pub source_score_sha256: String,
    pub all_source_event_ids: Vec<String>,
    pub event_a_id: String,
    pub event_b_id: String,
    pub renderer_profile_sha256: String,
    pub y00: CoalitionCommitment,
    pub y10: CoalitionCommitment,
    pub y01: CoalitionCommitment,
    pub y11: CoalitionCommitment,
    pub repeat_quartets: Vec<[CoalitionCommitment; 4]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairwiseInteractionVerdict {
    /// PCM and magnitude-feature second differences both exceed their
    /// repeat-render noise floors and frozen absolute thresholds.
    DetectedBoth,
    /// PCM second difference exceeds its calibrated threshold.
    DetectedPcm,
    /// Magnitude-feature second difference exceeds its calibrated threshold.
    /// This can arise from phase interference in an additive PCM renderer.
    DetectedMagnitudeOnly,
    /// Neither representation exceeded its calibrated threshold.
    NoDetectableInteraction,
    /// Calibration or measurement coverage was insufficient.
    Abstain,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairwiseInteractionWindow {
    pub start_frame: usize,
    pub end_frame: usize,
    pub channel: usize,
    /// RMS of Y00 - Y10 - Y01 + Y11, normalized by Y00 window RMS.
    pub pcm_interaction_rms: f32,
    /// Peak absolute second difference, normalized by Y00 window peak.
    pub pcm_interaction_peak: f32,
    /// Largest repeat-quartet residual deviation, normalized by Y00 RMS.
    pub pcm_repeat_noise_floor: f32,
    pub pcm_detection_threshold: f32,
    /// L1 norm of the magnitude-spectrum second difference divided by the
    /// sum of the four coalition magnitude spectra. Bounded in [0, 1].
    pub magnitude_feature_interaction: f32,
    /// Largest per-bin normalized magnitude-residual deviation across repeats.
    pub magnitude_repeat_noise_floor: f32,
    pub magnitude_detection_threshold: f32,
    pub pcm_detected: bool,
    pub magnitude_feature_detected: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairwiseInteractionSupport {
    pub start_frame: usize,
    pub end_frame: usize,
    pub channel: usize,
    pub pcm_detected: bool,
    pub magnitude_feature_detected: bool,
}

/// Content-bound measurement of one pair of source events in one render context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairwiseEventInteractionReport {
    pub verifier_version: String,
    pub identity: PairwiseInteractionIdentity,
    pub config: PairwiseInteractionConfig,
    pub sample_rate_hz: u32,
    pub channels: usize,
    pub total_frames: usize,
    pub unique_windows: usize,
    pub audited_channel_windows: usize,
    pub pcm_detected_channel_windows: usize,
    pub magnitude_detected_channel_windows: usize,
    pub support: Vec<PairwiseInteractionSupport>,
    pub worst_pcm_interaction_rms: f32,
    pub worst_pcm_interaction_peak: f32,
    pub worst_magnitude_feature_interaction: f32,
    pub verdict: PairwiseInteractionVerdict,
    pub reasons: Vec<String>,
    pub measurements: Vec<PairwiseInteractionWindow>,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairwiseInteractionError {
    ZeroSampleRate,
    ZeroChannels,
    EmptyAudio,
    InterleavedLengthNotDivisibleByChannels,
    NonFiniteSample,
    FormatMismatch,
    LengthMismatch,
    MissingRepeatQuartet,
    InvalidEventSet,
    InvalidEventId,
    SameEventTwice,
    EventNotInSource,
    InvalidSha256(&'static str),
    DuplicateRendererReceipt,
    RepeatScheduleMismatch,
    InvalidConfig(&'static str),
    SerializationFailure,
}

impl std::fmt::Display for PairwiseInteractionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroSampleRate => write!(f, "sample rate must be non-zero"),
            Self::ZeroChannels => write!(f, "channel count must be non-zero"),
            Self::EmptyAudio => write!(f, "audio must contain at least one frame"),
            Self::InterleavedLengthNotDivisibleByChannels => {
                write!(
                    f,
                    "interleaved sample count is not divisible by channel count"
                )
            }
            Self::NonFiniteSample => write!(f, "audio contains NaN or infinite samples"),
            Self::FormatMismatch => write!(f, "coalition audio formats differ"),
            Self::LengthMismatch => write!(f, "coalition audio frame counts differ"),
            Self::MissingRepeatQuartet => {
                write!(f, "at least one independent repeat quartet is required")
            }
            Self::InvalidEventSet => write!(
                f,
                "coalition event set does not match the exact factorial intervention"
            ),
            Self::InvalidEventId => {
                write!(f, "source event ID is empty, too long or contains controls")
            }
            Self::SameEventTwice => write!(f, "pairwise event IDs must be distinct"),
            Self::EventNotInSource => {
                write!(f, "intervened event is absent from the source event set")
            }
            Self::InvalidSha256(field) => write!(f, "invalid SHA-256 field: {field}"),
            Self::DuplicateRendererReceipt => {
                write!(f, "renderer receipt digest is reused across executions")
            }
            Self::RepeatScheduleMismatch => write!(
                f,
                "repeat quartet does not reuse the exact four coalition schedules"
            ),
            Self::InvalidConfig(reason) => write!(f, "invalid interaction config: {reason}"),
            Self::SerializationFailure => write!(f, "could not serialize interaction report"),
        }
    }
}

impl std::error::Error for PairwiseInteractionError {}

/// Compute a factorial interaction report from the four primary coalitions
/// and one or more independent same-schedule repeat quartets.
///
/// All coalitions must have identical output layout. The repeat quartet must
/// reuse the four exact schedule digests while using distinct renderer receipts.
/// A calibration ID is required before a verdict other than Abstain is emitted.
pub fn measure_pairwise_event_interaction(
    source_score_sha256: &str,
    all_source_event_ids: &[&str],
    event_a_id: &str,
    event_b_id: &str,
    renderer_profile_sha256: &str,
    primary: &PairwiseRenderQuartet<'_>,
    repeats: &[PairwiseRenderQuartet<'_>],
    config: &PairwiseInteractionConfig,
) -> Result<PairwiseEventInteractionReport, PairwiseInteractionError> {
    if repeats.is_empty() {
        return Err(PairwiseInteractionError::MissingRepeatQuartet);
    }
    let all_ids = validate_source_event_ids(all_source_event_ids)?;
    validate_event_id(event_a_id)?;
    validate_event_id(event_b_id)?;
    if event_a_id == event_b_id {
        return Err(PairwiseInteractionError::SameEventTwice);
    }
    if !all_ids.iter().any(|id| id == event_a_id) || !all_ids.iter().any(|id| id == event_b_id) {
        return Err(PairwiseInteractionError::EventNotInSource);
    }
    validate_sha256("source_score_sha256", source_score_sha256)?;
    validate_sha256("renderer_profile_sha256", renderer_profile_sha256)?;
    validate_config(config)?;

    validate_quartet(primary, &all_ids, event_a_id, event_b_id, None)?;
    let reference_audio = primary.y00.audio;
    let total_frames = reference_audio.interleaved_samples.len() / reference_audio.channels;
    if config.frame_size > total_frames {
        return Err(PairwiseInteractionError::InvalidConfig(
            "frame_size must not exceed the rendered audio length",
        ));
    }

    let mut all_receipts = Vec::with_capacity((repeats.len() + 1) * 4);
    for coalition in quartet_coalitions(primary) {
        all_receipts.push(coalition.renderer_receipt_sha256);
    }
    let mut repeat_commitments = Vec::with_capacity(repeats.len());
    for repeat in repeats {
        validate_quartet(repeat, &all_ids, event_a_id, event_b_id, Some(primary))?;
        for coalition in quartet_coalitions(repeat) {
            all_receipts.push(coalition.renderer_receipt_sha256);
        }
        repeat_commitments.push(quartet_commitments(repeat));
    }
    let mut receipt_set = std::collections::BTreeSet::new();
    for receipt in all_receipts {
        if !receipt_set.insert(receipt.to_owned()) {
            return Err(PairwiseInteractionError::DuplicateRendererReceipt);
        }
    }

    let starts = analysis_starts(total_frames, config.frame_size, config.hop_size);
    let fft = FftPlanner::<f32>::new().plan_fft_forward(config.frame_size);
    let window = hann_window(config.frame_size);
    let mut measurements =
        Vec::with_capacity(starts.len().saturating_mul(reference_audio.channels));
    let mut support = Vec::new();
    let mut pcm_detected_channel_windows = 0usize;
    let mut magnitude_detected_channel_windows = 0usize;
    let mut worst_pcm_interaction_rms = 0.0f32;
    let mut worst_pcm_interaction_peak = 0.0f32;
    let mut worst_magnitude_feature_interaction = 0.0f32;

    for start_frame in starts.iter().copied() {
        let end_frame = start_frame + config.frame_size;
        for channel in 0..reference_audio.channels {
            let base_window = coalition_windows(primary, start_frame, config.frame_size, channel);
            let primary_pcm = pcm_interaction(&base_window);
            let base_pcm: Vec<f64> = base_window[0]
                .iter()
                .map(|sample| f64::from(*sample))
                .collect();
            let base_rms = stable_rms(&base_pcm).max(EPSILON_RMS);
            let pcm_interaction_rms = stable_rms(&primary_pcm) / base_rms;
            let pcm_interaction_peak = max_abs(&primary_pcm) / max_abs(&base_pcm).max(EPSILON_RMS);

            let mut pcm_repeat_noise_floor = 0.0f64;
            let mut magnitude_repeat_noise_floor = 0.0f64;
            for repeat in repeats {
                let repeat_window =
                    coalition_windows(repeat, start_frame, config.frame_size, channel);
                let repeat_pcm = pcm_interaction(&repeat_window);
                let difference: Vec<f64> = repeat_pcm
                    .iter()
                    .zip(&primary_pcm)
                    .map(|(x, y)| x - y)
                    .collect();
                pcm_repeat_noise_floor =
                    pcm_repeat_noise_floor.max(stable_rms(&difference) / base_rms);
            }

            let (magnitude_feature_interaction, primary_magnitude_residual) =
                magnitude_interaction(&base_window, &window, fft.as_ref());
            for repeat in repeats {
                let repeat_window =
                    coalition_windows(repeat, start_frame, config.frame_size, channel);
                let (_, repeat_magnitude_residual) =
                    magnitude_interaction(&repeat_window, &window, fft.as_ref());
                let difference: f64 = repeat_magnitude_residual
                    .iter()
                    .zip(&primary_magnitude_residual)
                    .map(|(x, y)| (x - y).abs())
                    .sum();
                magnitude_repeat_noise_floor = magnitude_repeat_noise_floor.max(difference);
            }

            let pcm_detection_threshold = (pcm_repeat_noise_floor
                * f64::from(config.noise_multiplier))
            .max(f64::from(config.minimum_pcm_interaction));
            let magnitude_detection_threshold = (magnitude_repeat_noise_floor
                * f64::from(config.noise_multiplier))
            .max(f64::from(config.minimum_magnitude_interaction));
            let pcm_detected = pcm_interaction_rms > pcm_detection_threshold;
            let magnitude_feature_detected =
                magnitude_feature_interaction > magnitude_detection_threshold;

            worst_pcm_interaction_rms = worst_pcm_interaction_rms.max(pcm_interaction_rms as f32);
            worst_pcm_interaction_peak =
                worst_pcm_interaction_peak.max(pcm_interaction_peak as f32);
            worst_magnitude_feature_interaction =
                worst_magnitude_feature_interaction.max(magnitude_feature_interaction as f32);
            if pcm_detected {
                pcm_detected_channel_windows += 1;
            }
            if magnitude_feature_detected {
                magnitude_detected_channel_windows += 1;
            }
            if pcm_detected || magnitude_feature_detected {
                support.push(PairwiseInteractionSupport {
                    start_frame,
                    end_frame,
                    channel,
                    pcm_detected,
                    magnitude_feature_detected,
                });
            }
            measurements.push(PairwiseInteractionWindow {
                start_frame,
                end_frame,
                channel,
                pcm_interaction_rms: pcm_interaction_rms as f32,
                pcm_interaction_peak: pcm_interaction_peak as f32,
                pcm_repeat_noise_floor: pcm_repeat_noise_floor as f32,
                pcm_detection_threshold: pcm_detection_threshold as f32,
                magnitude_feature_interaction: magnitude_feature_interaction as f32,
                magnitude_repeat_noise_floor: magnitude_repeat_noise_floor as f32,
                magnitude_detection_threshold: magnitude_detection_threshold as f32,
                pcm_detected,
                magnitude_feature_detected,
            });
        }
    }

    let mut reasons = Vec::new();
    let verdict = if config.calibration_id.is_none() {
        reasons.push(
            "no calibration ID supplied; the interaction classification is exploratory".to_owned(),
        );
        PairwiseInteractionVerdict::Abstain
    } else if starts.len() < config.minimum_covered_windows {
        reasons.push(format!(
            "only {} unique windows were measured; {} required",
            starts.len(),
            config.minimum_covered_windows
        ));
        PairwiseInteractionVerdict::Abstain
    } else {
        match (
            pcm_detected_channel_windows > 0,
            magnitude_detected_channel_windows > 0,
        ) {
            (true, true) => PairwiseInteractionVerdict::DetectedBoth,
            (true, false) => PairwiseInteractionVerdict::DetectedPcm,
            (false, true) => PairwiseInteractionVerdict::DetectedMagnitudeOnly,
            (false, false) => PairwiseInteractionVerdict::NoDetectableInteraction,
        }
    };

    let identity = PairwiseInteractionIdentity {
        source_score_sha256: source_score_sha256.to_owned(),
        all_source_event_ids: all_ids,
        event_a_id: event_a_id.to_owned(),
        event_b_id: event_b_id.to_owned(),
        renderer_profile_sha256: renderer_profile_sha256.to_owned(),
        y00: coalition_commitment(primary.y00),
        y10: coalition_commitment(primary.y10),
        y01: coalition_commitment(primary.y01),
        y11: coalition_commitment(primary.y11),
        repeat_quartets: repeat_commitments,
    };
    let mut report = PairwiseEventInteractionReport {
        verifier_version: PAIRWISE_EVENT_INTERACTION_VERSION.to_owned(),
        identity,
        config: config.clone(),
        sample_rate_hz: reference_audio.sample_rate_hz,
        channels: reference_audio.channels,
        total_frames,
        unique_windows: starts.len(),
        audited_channel_windows: measurements.len(),
        pcm_detected_channel_windows,
        magnitude_detected_channel_windows,
        support,
        worst_pcm_interaction_rms,
        worst_pcm_interaction_peak,
        worst_magnitude_feature_interaction,
        verdict,
        reasons,
        measurements,
        receipt_sha256: String::new(),
    };
    report.receipt_sha256 = compute_receipt_sha256(&report)?;
    Ok(report)
}

/// Replay the full measurement from the exact coalitions/repeats and compare
/// every report field. A valid report hash alone is not sufficient.
pub fn verify_pairwise_event_interaction_report(
    report: &PairwiseEventInteractionReport,
    primary: &PairwiseRenderQuartet<'_>,
    repeats: &[PairwiseRenderQuartet<'_>],
) -> Result<bool, PairwiseInteractionError> {
    if report.receipt_sha256.is_empty() || compute_receipt_sha256(report)? != report.receipt_sha256
    {
        return Ok(false);
    }
    let all_ids: Vec<&str> = report
        .identity
        .all_source_event_ids
        .iter()
        .map(String::as_str)
        .collect();
    let expected = measure_pairwise_event_interaction(
        &report.identity.source_score_sha256,
        &all_ids,
        &report.identity.event_a_id,
        &report.identity.event_b_id,
        &report.identity.renderer_profile_sha256,
        primary,
        repeats,
        &report.config,
    )?;
    Ok(&expected == report)
}

fn validate_quartet(
    quartet: &PairwiseRenderQuartet<'_>,
    all_ids: &[String],
    event_a: &str,
    event_b: &str,
    primary: Option<&PairwiseRenderQuartet<'_>>,
) -> Result<(), PairwiseInteractionError> {
    let expected_y00 = all_ids.to_vec();
    let expected_y10: Vec<String> = all_ids
        .iter()
        .filter(|id| id.as_str() != event_a)
        .cloned()
        .collect();
    let expected_y01: Vec<String> = all_ids
        .iter()
        .filter(|id| id.as_str() != event_b)
        .cloned()
        .collect();
    let expected_y11: Vec<String> = all_ids
        .iter()
        .filter(|id| id.as_str() != event_a && id.as_str() != event_b)
        .cloned()
        .collect();
    let coalitions = quartet_coalitions(quartet);
    let expected = [&expected_y00, &expected_y10, &expected_y01, &expected_y11];
    for (coalition, expected_ids) in coalitions.into_iter().zip(expected) {
        validate_view(coalition.audio)?;
        validate_sha256("schedule_sha256", coalition.schedule_sha256)?;
        validate_sha256(
            "performed_event_map_sha256",
            coalition.performed_event_map_sha256,
        )?;
        validate_sha256("renderer_receipt_sha256", coalition.renderer_receipt_sha256)?;
        let actual: Vec<String> = coalition
            .active_event_ids
            .iter()
            .map(|id| (*id).to_owned())
            .collect();
        if actual.as_slice() != expected_ids.as_slice() {
            return Err(PairwiseInteractionError::InvalidEventSet);
        }
    }
    let reference = quartet.y00.audio;
    for coalition in quartet_coalitions(quartet).into_iter().skip(1) {
        if coalition.audio.sample_rate_hz != reference.sample_rate_hz
            || coalition.audio.channels != reference.channels
        {
            return Err(PairwiseInteractionError::FormatMismatch);
        }
        if coalition.audio.interleaved_samples.len() != reference.interleaved_samples.len() {
            return Err(PairwiseInteractionError::LengthMismatch);
        }
    }
    let [schedule_00, schedule_10, schedule_01, schedule_11] = quartet_coalitions(quartet);
    let schedules = [
        schedule_00.schedule_sha256,
        schedule_10.schedule_sha256,
        schedule_01.schedule_sha256,
        schedule_11.schedule_sha256,
    ];
    if schedules
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != 4
    {
        return Err(PairwiseInteractionError::InvalidEventSet);
    }
    if let Some(primary) = primary {
        for (repeat, first) in quartet_coalitions(quartet)
            .into_iter()
            .zip(quartet_coalitions(primary))
        {
            if repeat.schedule_sha256 != first.schedule_sha256
                || repeat.performed_event_map_sha256 != first.performed_event_map_sha256
            {
                return Err(PairwiseInteractionError::RepeatScheduleMismatch);
            }
            if repeat.audio.sample_rate_hz != first.audio.sample_rate_hz
                || repeat.audio.channels != first.audio.channels
            {
                return Err(PairwiseInteractionError::FormatMismatch);
            }
            if repeat.audio.interleaved_samples.len() != first.audio.interleaved_samples.len() {
                return Err(PairwiseInteractionError::LengthMismatch);
            }
        }
    }
    Ok(())
}

fn validate_source_event_ids(ids: &[&str]) -> Result<Vec<String>, PairwiseInteractionError> {
    if ids.is_empty() {
        return Err(PairwiseInteractionError::InvalidEventSet);
    }
    let mut owned = Vec::with_capacity(ids.len());
    for id in ids {
        validate_event_id(id)?;
        owned.push((*id).to_owned());
    }
    let mut sorted = owned.clone();
    sorted.sort();
    sorted.dedup();
    if sorted.len() != owned.len() || sorted != owned {
        return Err(PairwiseInteractionError::InvalidEventSet);
    }
    Ok(owned)
}

fn validate_event_id(id: &str) -> Result<(), PairwiseInteractionError> {
    if id.trim().is_empty() || id.len() > 256 || id.chars().any(char::is_control) {
        return Err(PairwiseInteractionError::InvalidEventId);
    }
    Ok(())
}

fn validate_sha256(field: &'static str, value: &str) -> Result<(), PairwiseInteractionError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(PairwiseInteractionError::InvalidSha256(field));
    }
    Ok(())
}

fn validate_view(view: AudioView<'_>) -> Result<(), PairwiseInteractionError> {
    if view.sample_rate_hz == 0 {
        return Err(PairwiseInteractionError::ZeroSampleRate);
    }
    if view.channels == 0 {
        return Err(PairwiseInteractionError::ZeroChannels);
    }
    if view.interleaved_samples.is_empty() {
        return Err(PairwiseInteractionError::EmptyAudio);
    }
    if view.interleaved_samples.len() % view.channels != 0 {
        return Err(PairwiseInteractionError::InterleavedLengthNotDivisibleByChannels);
    }
    if view
        .interleaved_samples
        .iter()
        .any(|sample| !sample.is_finite())
    {
        return Err(PairwiseInteractionError::NonFiniteSample);
    }
    Ok(())
}

fn validate_config(config: &PairwiseInteractionConfig) -> Result<(), PairwiseInteractionError> {
    if config.frame_size < 16 {
        return Err(PairwiseInteractionError::InvalidConfig(
            "frame_size must be at least 16",
        ));
    }
    if config.frame_size > MAX_FRAME_SIZE {
        return Err(PairwiseInteractionError::InvalidConfig(
            "frame_size exceeds maximum allocation",
        ));
    }
    if config.hop_size == 0 || config.hop_size > config.frame_size {
        return Err(PairwiseInteractionError::InvalidConfig(
            "hop_size must be in 1..=frame_size",
        ));
    }
    if !config.noise_multiplier.is_finite()
        || config.noise_multiplier < 1.0
        || config.noise_multiplier > 100.0
    {
        return Err(PairwiseInteractionError::InvalidConfig(
            "noise multiplier must be finite and in 1..=100",
        ));
    }
    if !config.minimum_pcm_interaction.is_finite()
        || config.minimum_pcm_interaction < 0.0
        || config.minimum_pcm_interaction > 10.0
    {
        return Err(PairwiseInteractionError::InvalidConfig(
            "PCM interaction floor must be finite and in 0..=10",
        ));
    }
    if !config.minimum_magnitude_interaction.is_finite()
        || !(0.0..=1.0).contains(&config.minimum_magnitude_interaction)
    {
        return Err(PairwiseInteractionError::InvalidConfig(
            "magnitude interaction floor must be finite and in 0..=1",
        ));
    }
    if config.minimum_covered_windows == 0 {
        return Err(PairwiseInteractionError::InvalidConfig(
            "minimum_covered_windows must be non-zero",
        ));
    }
    if config
        .calibration_id
        .as_ref()
        .is_some_and(|id| id.trim().is_empty())
    {
        return Err(PairwiseInteractionError::InvalidConfig(
            "calibration ID must be non-empty when present",
        ));
    }
    Ok(())
}

fn quartet_coalitions(quartet: &PairwiseRenderQuartet<'_>) -> [CoalitionRender<'_>; 4] {
    [quartet.y00, quartet.y10, quartet.y01, quartet.y11]
}

fn coalition_commitment(coalition: CoalitionRender<'_>) -> CoalitionCommitment {
    CoalitionCommitment {
        schedule_sha256: coalition.schedule_sha256.to_owned(),
        performed_event_map_sha256: coalition.performed_event_map_sha256.to_owned(),
        renderer_receipt_sha256: coalition.renderer_receipt_sha256.to_owned(),
        audio_sha256: audio_digest(coalition.audio),
    }
}

fn quartet_commitments(quartet: &PairwiseRenderQuartet<'_>) -> [CoalitionCommitment; 4] {
    let [y00, y10, y01, y11] = quartet_coalitions(quartet);
    [
        coalition_commitment(y00),
        coalition_commitment(y10),
        coalition_commitment(y01),
        coalition_commitment(y11),
    ]
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

fn coalition_windows(
    quartet: &PairwiseRenderQuartet<'_>,
    start: usize,
    frames: usize,
    channel: usize,
) -> [Vec<f32>; 4] {
    quartet_coalitions(quartet)
        .map(|coalition| channel_window(coalition.audio, start, frames, channel))
}

fn channel_window(view: AudioView<'_>, start: usize, frames: usize, channel: usize) -> Vec<f32> {
    (start..start + frames)
        .map(|frame| view.interleaved_samples[frame * view.channels + channel])
        .collect()
}

fn pcm_interaction(windows: &[Vec<f32>; 4]) -> Vec<f64> {
    (0..windows[0].len())
        .map(|index| {
            f64::from(windows[0][index])
                - f64::from(windows[1][index])
                - f64::from(windows[2][index])
                + f64::from(windows[3][index])
        })
        .collect()
}

fn stable_rms(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let scale = samples.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    if scale <= f64::MIN_POSITIVE {
        return 0.0;
    }
    let mean_square =
        samples.iter().map(|x| (x / scale).powi(2)).sum::<f64>() / samples.len() as f64;
    scale * mean_square.sqrt()
}

fn max_abs(samples: &[f64]) -> f64 {
    samples.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|index| 0.5 - 0.5 * (std::f32::consts::TAU * index as f32 / (size - 1) as f32).cos())
        .collect()
}

/// Return the normalized magnitude-feature interaction and a per-bin normalized
/// residual vector used to compare repeat-render noise with the primary quartet.
fn magnitude_interaction(
    windows: &[Vec<f32>; 4],
    taper: &[f32],
    fft: &dyn Fft<f32>,
) -> (f64, Vec<f64>) {
    let max_sample = windows
        .iter()
        .flat_map(|window| window.iter())
        .fold(0.0f32, |m, x| m.max(x.abs()))
        .max(f32::MIN_POSITIVE);
    let mut magnitudes = Vec::with_capacity(4);
    for samples in windows {
        let mut spectrum: Vec<Complex<f32>> = samples
            .iter()
            .zip(taper)
            .map(|(sample, window)| Complex::new((sample / max_sample) * window, 0.0))
            .collect();
        fft.process(&mut spectrum);
        magnitudes.push(
            spectrum[..=spectrum.len() / 2]
                .iter()
                .map(|bin| f64::from(bin.norm()))
                .collect::<Vec<_>>(),
        );
    }
    let mut denominator = 0.0f64;
    let mut residual = Vec::with_capacity(magnitudes[0].len());
    for index in 0..magnitudes[0].len() {
        let scale = magnitudes.iter().map(|row| row[index]).sum::<f64>();
        denominator += scale;
        residual.push(
            magnitudes[0][index] - magnitudes[1][index] - magnitudes[2][index]
                + magnitudes[3][index],
        );
    }
    let score = if denominator <= f64::EPSILON {
        0.0
    } else {
        residual.iter().map(|x| x.abs()).sum::<f64>() / denominator
    };
    let normalized_residual = if denominator <= f64::EPSILON {
        vec![0.0; residual.len()]
    } else {
        residual.into_iter().map(|x| x / denominator).collect()
    };
    (score, normalized_residual)
}

fn audio_digest(view: AudioView<'_>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous-dynamics-pairwise-interaction-audio-v1\0");
    hasher.update(view.sample_rate_hz.to_le_bytes());
    hasher.update((view.channels as u64).to_le_bytes());
    hasher.update(((view.interleaved_samples.len() / view.channels) as u64).to_le_bytes());
    for sample in view.interleaved_samples {
        hasher.update(sample.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn compute_receipt_sha256(
    report: &PairwiseEventInteractionReport,
) -> Result<String, PairwiseInteractionError> {
    let mut unsealed = report.clone();
    unsealed.receipt_sha256.clear();
    let bytes = serde_json::to_vec(&unsealed)
        .map_err(|_| PairwiseInteractionError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous-dynamics-mel-mr-eval-018-pairwise-interaction-report-v1\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const H_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const H_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const H_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const H_D: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const H_E: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const H_F: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    const H_1: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const H_2: &str = "2222222222222222222222222222222222222222222222222222222222222222";
    const H_3: &str = "3333333333333333333333333333333333333333333333333333333333333333";
    const H_4: &str = "4444444444444444444444444444444444444444444444444444444444444444";
    const H_5: &str = "5555555555555555555555555555555555555555555555555555555555555555";
    const H_6: &str = "6666666666666666666666666666666666666666666666666666666666666666";
    const H_7: &str = "7777777777777777777777777777777777777777777777777777777777777777";
    const H_8: &str = "8888888888888888888888888888888888888888888888888888888888888888";

    fn sine(frames: usize, frequency: f32) -> Vec<f32> {
        (0..frames)
            .map(|index| (std::f32::consts::TAU * frequency * index as f32 / 16_000.0).sin() * 0.25)
            .collect()
    }

    fn view(samples: &[f32]) -> AudioView<'_> {
        AudioView {
            interleaved_samples: samples,
            sample_rate_hz: 16_000,
            channels: 1,
        }
    }

    fn calibrated_config() -> PairwiseInteractionConfig {
        PairwiseInteractionConfig {
            frame_size: 64,
            hop_size: 32,
            noise_multiplier: 3.0,
            minimum_pcm_interaction: 0.001,
            minimum_magnitude_interaction: 0.02,
            minimum_covered_windows: 4,
            calibration_id: Some("synthetic-pairwise-calibration-v0".to_owned()),
        }
    }

    static ALL_EVENT_IDS: [&str; 3] = ["event-a", "event-b", "event-c"];
    static NO_A_EVENT_IDS: [&str; 2] = ["event-b", "event-c"];
    static NO_B_EVENT_IDS: [&str; 2] = ["event-a", "event-c"];
    static NO_AB_EVENT_IDS: [&str; 1] = ["event-c"];

    fn event_ids() -> [&'static str; 3] {
        ALL_EVENT_IDS
    }

    fn quartet<'a>(
        y00: &'a [f32],
        y10: &'a [f32],
        y01: &'a [f32],
        y11: &'a [f32],
        receipt_start: usize,
    ) -> PairwiseRenderQuartet<'a> {
        let hashes = [
            H_A, H_B, H_C, H_D, H_E, H_F, H_1, H_2, H_3, H_4, H_5, H_6, H_7, H_8,
        ];
        let receipt = |index: usize| hashes[(receipt_start + index) % hashes.len()];
        let schedule = |index: usize| hashes[4 + index];
        let make = |ids: &'a [&'a str], samples: &'a [f32], index: usize| CoalitionRender {
            active_event_ids: ids,
            schedule_sha256: schedule(index),
            performed_event_map_sha256: schedule(index),
            renderer_receipt_sha256: receipt(index),
            audio: view(samples),
        };
        PairwiseRenderQuartet {
            y00: make(&ALL_EVENT_IDS, y00, 0),
            y10: make(&NO_A_EVENT_IDS, y10, 1),
            y01: make(&NO_B_EVENT_IDS, y01, 2),
            y11: make(&NO_AB_EVENT_IDS, y11, 3),
        }
    }

    fn measure(
        primary: &PairwiseRenderQuartet<'_>,
        repeat: &PairwiseRenderQuartet<'_>,
        config: &PairwiseInteractionConfig,
    ) -> Result<PairwiseEventInteractionReport, PairwiseInteractionError> {
        measure_pairwise_event_interaction(
            H_A,
            &event_ids(),
            "event-a",
            "event-b",
            H_B,
            primary,
            &[*repeat],
            config,
        )
    }

    #[test]
    fn additive_pcm_control_has_no_pcm_interaction() {
        let a = sine(1024, 440.0);
        let b = sine(1024, 660.0);
        let sum: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        let zero = vec![0.0; a.len()];
        let primary = quartet(&sum, &b, &a, &zero, 0);
        let repeat = quartet(&sum, &b, &a, &zero, 8);
        let report = measure(&primary, &repeat, &calibrated_config()).unwrap();

        assert_eq!(report.pcm_detected_channel_windows, 0);
        assert!(matches!(
            report.verdict,
            PairwiseInteractionVerdict::NoDetectableInteraction
                | PairwiseInteractionVerdict::DetectedMagnitudeOnly
        ));
        assert!(verify_pairwise_event_interaction_report(&report, &primary, &[repeat]).unwrap());
    }

    #[test]
    fn nonlinear_pcm_cross_term_is_detected() {
        let a = sine(1024, 440.0);
        let b = sine(1024, 660.0);
        let mut y00 = Vec::with_capacity(a.len());
        for (x, y) in a.iter().zip(&b) {
            y00.push(x + y + 0.8 * x * y);
        }
        let zero = vec![0.0; a.len()];
        let primary = quartet(&y00, &b, &a, &zero, 0);
        let repeat = quartet(&y00, &b, &a, &zero, 8);
        let report = measure(&primary, &repeat, &calibrated_config()).unwrap();

        assert!(report.pcm_detected_channel_windows > 0);
        assert!(matches!(
            report.verdict,
            PairwiseInteractionVerdict::DetectedPcm | PairwiseInteractionVerdict::DetectedBoth
        ));
    }

    #[test]
    fn magnitude_only_phase_interaction_is_not_mislabelled_as_pcm_nonadditivity() {
        let a = sine(1024, 440.0);
        let b = sine(1024, 660.0);
        let sum: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        let zero = vec![0.0; a.len()];
        let primary = quartet(&sum, &b, &a, &zero, 0);
        let repeat = quartet(&sum, &b, &a, &zero, 8);
        let mut config = calibrated_config();
        config.minimum_magnitude_interaction = 0.000001;
        let report = measure(&primary, &repeat, &config).unwrap();

        assert_eq!(report.pcm_detected_channel_windows, 0);
        // A magnitude transform is nonlinear even when PCM summation is additive.
        assert!(report.magnitude_detected_channel_windows > 0);
        assert_eq!(
            report.verdict,
            PairwiseInteractionVerdict::DetectedMagnitudeOnly
        );
    }

    #[test]
    fn repeat_quartet_variation_is_used_as_a_noise_floor() {
        let a = sine(1024, 440.0);
        let b = sine(1024, 660.0);
        let zero = vec![0.0; a.len()];
        let sum: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        let primary = quartet(&sum, &b, &a, &zero, 0);
        let mut noisy = sum.clone();
        for (index, sample) in noisy.iter_mut().enumerate() {
            *sample += (index as f32 * 0.017).sin() * 0.01;
        }
        let repeat = quartet(&noisy, &b, &a, &zero, 8);
        let mut config = calibrated_config();
        config.minimum_pcm_interaction = 0.0;
        config.minimum_magnitude_interaction = 0.0;
        config.noise_multiplier = 10.0;
        let report = measure(&primary, &repeat, &config).unwrap();

        assert!(
            report
                .measurements
                .iter()
                .any(|m| m.pcm_repeat_noise_floor > 0.0)
        );
        assert_eq!(report.pcm_detected_channel_windows, 0);
    }

    #[test]
    fn missing_calibration_abstains_and_preserves_diagnostics() {
        let a = sine(1024, 440.0);
        let b = sine(1024, 660.0);
        let zero = vec![0.0; a.len()];
        let mut y00 = Vec::with_capacity(a.len());
        for (x, y) in a.iter().zip(&b) {
            y00.push(x + y + 0.8 * x * y);
        }
        let primary = quartet(&y00, &b, &a, &zero, 0);
        let repeat = quartet(&y00, &b, &a, &zero, 8);
        let mut config = calibrated_config();
        config.calibration_id = None;
        let report = measure(&primary, &repeat, &config).unwrap();

        assert_eq!(report.verdict, PairwiseInteractionVerdict::Abstain);
        assert!(report.pcm_detected_channel_windows > 0);
    }

    #[test]
    fn invalid_coalition_identity_and_receipt_reuse_reject() {
        let a = sine(1024, 440.0);
        let b = sine(1024, 660.0);
        let sum: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        let zero = vec![0.0; a.len()];
        let mut primary = quartet(&sum, &b, &a, &zero, 0);
        let repeat = quartet(&sum, &b, &a, &zero, 8);

        primary.y10.active_event_ids = &["event-a", "event-c"];
        assert_eq!(
            measure(&primary, &repeat, &calibrated_config()).unwrap_err(),
            PairwiseInteractionError::InvalidEventSet
        );

        let mut reused = quartet(&sum, &b, &a, &zero, 0);
        reused.y00.renderer_receipt_sha256 = reused.y10.renderer_receipt_sha256;
        assert_eq!(
            measure(&reused, &repeat, &calibrated_config()).unwrap_err(),
            PairwiseInteractionError::DuplicateRendererReceipt
        );
    }
    #[test]
    fn repeat_quartet_must_reuse_the_exact_event_map_commitments() {
        let a = sine(1024, 440.0);
        let b = sine(1024, 660.0);
        let sum: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        let zero = vec![0.0; a.len()];
        let primary = quartet(&sum, &b, &a, &zero, 0);
        let mut repeat = quartet(&sum, &b, &a, &zero, 8);
        repeat.y00.performed_event_map_sha256 = H_F;
        assert_eq!(
            measure(&primary, &repeat, &calibrated_config()).unwrap_err(),
            PairwiseInteractionError::RepeatScheduleMismatch
        );
    }

    #[test]
    fn repeat_quartet_must_preserve_sample_layout() {
        let a = sine(1024, 440.0);
        let b = sine(1024, 660.0);
        let sum: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        let zero = vec![0.0; a.len()];
        let short_a = sine(960, 440.0);
        let short_b = sine(960, 660.0);
        let short_sum: Vec<f32> = short_a.iter().zip(&short_b).map(|(x, y)| x + y).collect();
        let short_zero = vec![0.0; short_a.len()];
        let primary = quartet(&sum, &b, &a, &zero, 0);
        let repeat = quartet(&short_sum, &short_b, &short_a, &short_zero, 8);
        assert_eq!(
            measure(&primary, &repeat, &calibrated_config()).unwrap_err(),
            PairwiseInteractionError::LengthMismatch
        );
    }
}
