// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pairwise factorial interaction diagnostics for verified musical renders.
//!
//! A four-coalition experiment estimates the second-order finite difference
//! between two source-event interventions. It reports signed PCM and
//! time-frequency residuals separately. A non-zero spectral-magnitude
//! residual is not by itself evidence of a nonlinear renderer: magnitude
//! features are nonlinear even when PCM synthesis is additive.
//!
//! This module does not assign exclusive sample ownership, and it does not
//! issue a calibrated additive/nonlinear decision. Such a decision requires
//! a renderer-specific, seed-matched null distribution in a later increment.

use crate::audio_preservation::{AudioView, FrameInterval};
use crate::counterfactual_influence::{
    VerifiedRendererAudioBinding, audio_sha256,
};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const PAIRWISE_INTERACTION_VERSION: &str =
    "mel-mr-eval-018-pairwise-interaction-v0.1";
const PAIRWISE_DOMAIN: &[u8] =
    b"luminous-dynamics-mel-mr-eval-018-pairwise-interaction-v1\0";
const MAX_EVENT_COUNT: usize = 100_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairwiseInteractionConfig {
    /// Analysis window length in PCM frames.
    pub window_size_frames: usize,
    /// Distance between adjacent window starts.
    pub hop_size_frames: usize,
    /// Denominator guard for effect-normalized ratios.
    pub ratio_epsilon: f64,
}

impl Default for PairwiseInteractionConfig {
    fn default() -> Self {
        Self {
            window_size_frames: 1024,
            hop_size_frames: 256,
            ratio_epsilon: 1.0e-12,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairwiseInteractionWindow {
    pub start_frame: usize,
    pub end_frame: usize,
    pub channel: usize,
    /// RMS of the signed PCM second-order finite difference.
    pub pcm_interaction_rms: f64,
    pub pcm_interaction_peak: f64,
    /// Mean of the two first-order PCM effect RMS values.
    pub mean_main_effect_rms: f64,
    /// PCM interaction RMS divided by mean main-effect RMS, with epsilon guard.
    pub pcm_interaction_to_main_effect_ratio: f64,
    /// RMS of the signed second-order finite difference over STFT magnitudes.
    /// This is a distinct nonlinear feature-space diagnostic, not a PCM
    /// nonlinearity claim.
    pub spectral_magnitude_interaction_rms: f64,
    pub spectral_magnitude_interaction_to_main_effect_ratio: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairwiseInteractionReport {
    pub version: String,
    pub source_event_a: String,
    pub source_event_b: String,
    pub expected_score_event_count: usize,
    pub source_score_sha256: String,
    pub renderer_identity_sha256: String,
    pub renderer_config_sha256: String,
    pub sample_rate_hz: u32,
    pub channels: usize,
    pub total_frames: usize,

    pub baseline_renderer_receipt_sha256: String,
    pub remove_a_renderer_receipt_sha256: String,
    pub remove_b_renderer_receipt_sha256: String,
    pub remove_ab_renderer_receipt_sha256: String,

    pub baseline_schedule_sha256: String,
    pub remove_a_schedule_sha256: String,
    pub remove_b_schedule_sha256: String,
    pub remove_ab_schedule_sha256: String,

    pub baseline_map_artifact_sha256: String,
    pub remove_a_map_artifact_sha256: String,
    pub remove_b_map_artifact_sha256: String,
    pub remove_ab_map_artifact_sha256: String,

    pub baseline_map_receipt_sha256: String,
    pub remove_a_map_receipt_sha256: String,
    pub remove_b_map_receipt_sha256: String,
    pub remove_ab_map_receipt_sha256: String,

    pub baseline_audio_sha256: String,
    pub remove_a_audio_sha256: String,
    pub remove_b_audio_sha256: String,
    pub remove_ab_audio_sha256: String,

    pub analysis: PairwiseInteractionConfig,
    pub total_channel_windows: usize,
    pub windows: Vec<PairwiseInteractionWindow>,
    pub receipt_sha256: String,
}

/// A render coalition is usable only when it carries the opaque token produced
/// by replay-verifying RendererExecutionReceipt and matching exact decoded PCM.
pub struct VerifiedInteractionRender<'view, 'audio> {
    pub evidence: &'view VerifiedRendererAudioBinding,
    pub audio: &'view AudioView<'audio>,
}

/// Four coalitions for a 2x2 event-removal factorial experiment:
/// baseline, remove A, remove B, remove both A and B.
pub struct PairwiseInteractionInput<'view, 'audio> {
    pub source_event_a: &'view str,
    pub source_event_b: &'view str,
    pub expected_score_event_count: usize,
    pub baseline: VerifiedInteractionRender<'view, 'audio>,
    pub remove_a: VerifiedInteractionRender<'view, 'audio>,
    pub remove_b: VerifiedInteractionRender<'view, 'audio>,
    pub remove_ab: VerifiedInteractionRender<'view, 'audio>,
    pub analysis: PairwiseInteractionConfig,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairwiseInteractionError {
    InvalidConfig(&'static str),
    InvalidEventId,
    DuplicateEventIds,
    InvalidEventCount,
    InvalidCoalition(&'static str),
    SourceScoreMismatch,
    RendererIdentityMismatch,
    RendererConfigMismatch,
    AudioLayoutMismatch,
    RendererAudioMismatch,
    DuplicateScheduleDigest,
    DuplicateMapDigest,
    NonFiniteSample,
    EmptyAudio,
    NonDivisibleAudio,
    SerializationFailure,
    InvalidReceipt,
    ReplayMismatch,
}

impl std::fmt::Display for PairwiseInteractionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConfig(reason) => write!(f, "invalid pairwise interaction config: {reason}"),
            Self::InvalidEventId => write!(f, "event ID must be canonical score_event_index:<n>"),
            Self::DuplicateEventIds => write!(f, "pairwise intervention requires two distinct source events"),
            Self::InvalidEventCount => write!(f, "invalid expected Score event count"),
            Self::InvalidCoalition(name) => write!(f, "render coalition is invalid: {name}"),
            Self::SourceScoreMismatch => write!(f, "render coalitions do not share the same source Score"),
            Self::RendererIdentityMismatch => write!(f, "render coalitions use different renderer identities"),
            Self::RendererConfigMismatch => write!(f, "render coalitions use different renderer execution profiles"),
            Self::AudioLayoutMismatch => write!(f, "render coalitions have different PCM layouts"),
            Self::RendererAudioMismatch => write!(f, "verified renderer PCM digest differs from supplied AudioView"),
            Self::DuplicateScheduleDigest => write!(f, "coalition schedules are not four distinct interventions"),
            Self::DuplicateMapDigest => write!(f, "coalition performed-event maps are not distinct"),
            Self::NonFiniteSample => write!(f, "audio contains a non-finite sample"),
            Self::EmptyAudio => write!(f, "audio must contain at least one frame"),
            Self::NonDivisibleAudio => write!(f, "interleaved sample count is not divisible by channel count"),
            Self::SerializationFailure => write!(f, "could not serialize pairwise interaction evidence"),
            Self::InvalidReceipt => write!(f, "pairwise interaction report receipt is invalid"),
            Self::ReplayMismatch => write!(f, "pairwise interaction report does not replay"),
        }
    }
}

impl std::error::Error for PairwiseInteractionError {}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_digest<T: Serialize>(
    value: &T,
) -> Result<String, PairwiseInteractionError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|_| PairwiseInteractionError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(PAIRWISE_DOMAIN);
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn report_digest_valid(
    report: &PairwiseInteractionReport,
) -> Result<bool, PairwiseInteractionError> {
    let mut unsigned = report.clone();
    let receipt = std::mem::take(&mut unsigned.receipt_sha256);
    if receipt.len() != 64 {
        return Ok(false);
    }
    Ok(canonical_digest(&unsigned)? == receipt)
}

fn parse_source_index(value: &str) -> Option<usize> {
    let suffix = value.strip_prefix("score_event_index:")?;
    if suffix.is_empty()
        || (suffix.len() > 1 && suffix.starts_with('0'))
        || !suffix.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    suffix.parse().ok()
}

fn validate_config(
    config: &PairwiseInteractionConfig,
) -> Result<(), PairwiseInteractionError> {
    if config.window_size_frames < 16 {
        return Err(PairwiseInteractionError::InvalidConfig(
            "window_size_frames must be at least 16",
        ));
    }
    if config.hop_size_frames == 0 || config.hop_size_frames > config.window_size_frames {
        return Err(PairwiseInteractionError::InvalidConfig(
            "hop_size_frames must be in 1..=window_size_frames",
        ));
    }
    if !config.ratio_epsilon.is_finite() || config.ratio_epsilon <= 0.0 {
        return Err(PairwiseInteractionError::InvalidConfig(
            "ratio_epsilon must be finite and positive",
        ));
    }
    Ok(())
}

fn validate_audio(audio: &AudioView<'_>) -> Result<usize, PairwiseInteractionError> {
    if audio.interleaved_samples.is_empty() {
        return Err(PairwiseInteractionError::EmptyAudio);
    }
    if audio.channels == 0 || audio.interleaved_samples.len() % audio.channels != 0 {
        return Err(PairwiseInteractionError::NonDivisibleAudio);
    }
    if audio.sample_rate_hz == 0 {
        return Err(PairwiseInteractionError::AudioLayoutMismatch);
    }
    if audio.interleaved_samples.iter().any(|sample| !sample.is_finite()) {
        return Err(PairwiseInteractionError::NonFiniteSample);
    }
    Ok(audio.interleaved_samples.len() / audio.channels)
}

fn windows_for(
    total_frames: usize,
    config: &PairwiseInteractionConfig,
) -> Result<Vec<FrameInterval>, PairwiseInteractionError> {
    if config.window_size_frames > total_frames {
        return Err(PairwiseInteractionError::InvalidConfig(
            "analysis window is longer than the render",
        ));
    }
    let mut windows = Vec::new();
    let mut start = 0usize;
    loop {
        let end = start.checked_add(config.window_size_frames)
            .ok_or(PairwiseInteractionError::InvalidConfig("window index overflow"))?;
        if end > total_frames {
            break;
        }
        windows.push(FrameInterval { start_frame:start, end_frame:end });
        match start.checked_add(config.hop_size_frames) {
            Some(next) if next > start => start = next,
            _ => break,
        }
    }
    let final_start = total_frames - config.window_size_frames;
    if windows.last().is_none_or(|window| window.start_frame != final_start) {
        windows.push(FrameInterval { start_frame:final_start, end_frame:total_frames });
    }
    Ok(windows)
}

fn lineage_matches(left: &crate::performed_event_map::PerformedEventLineage,
                   right: &crate::performed_event_map::PerformedEventLineage) -> bool {
    left.source_event_id == right.source_event_id
        && left.descendant_ordinal == right.descendant_ordinal
        && left.midi_track_index == right.midi_track_index
        && left.midi_channel == right.midi_channel
        && left.midi_note == right.midi_note
        && left.onset_micros == right.onset_micros
        && left.end_micros == right.end_micros
        && left.onset_frame == right.onset_frame
        && left.end_frame == right.end_frame
}

fn validate_coalition<'view, 'audio>(
    name: &'static str,
    render: &VerifiedInteractionRender<'view, 'audio>,
    expected_ids: &BTreeSet<usize>,
    expected_score_sha256: &str,
    renderer_identity_sha256: &str,
    renderer_config_sha256: &str,
    sample_rate_hz: u32,
    channels: usize,
    total_frames: usize,
) -> Result<(), PairwiseInteractionError> {
    let frames = validate_audio(render.audio)?;
    if frames != total_frames
        || render.audio.sample_rate_hz != sample_rate_hz
        || render.audio.channels != channels
    {
        return Err(PairwiseInteractionError::AudioLayoutMismatch);
    }
    if audio_sha256(render.audio) != render.evidence.audio_sha256() {
        return Err(PairwiseInteractionError::RendererAudioMismatch);
    }
    if render.evidence.source_score_sha256() != expected_score_sha256 {
        return Err(PairwiseInteractionError::SourceScoreMismatch);
    }
    if render.evidence.renderer_identity_sha256() != renderer_identity_sha256 {
        return Err(PairwiseInteractionError::RendererIdentityMismatch);
    }
    if render.evidence.renderer_config_sha256() != renderer_config_sha256 {
        return Err(PairwiseInteractionError::RendererConfigMismatch);
    }

    let actual_ids: BTreeSet<usize> = render.evidence.performed_event_lineage()
        .iter()
        .filter_map(|row| parse_source_index(&row.source_event_id))
        .collect();
    if &actual_ids != expected_ids {
        return Err(PairwiseInteractionError::InvalidCoalition(name));
    }

    // Reject duplicated or unknown source IDs instead of allowing set
    // conversion to hide malformed lineage rows.
    if render.evidence.performed_event_lineage().iter().any(|row| {
        parse_source_index(&row.source_event_id).is_none()
    }) {
        return Err(PairwiseInteractionError::InvalidCoalition(name));
    }
    let actual_count = render.evidence.performed_event_lineage()
        .iter()
        .map(|row| row.source_event_id.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    if actual_count != expected_ids.len() {
        return Err(PairwiseInteractionError::InvalidCoalition(name));
    }
    let _ = samples;
    Ok(())
}

fn validate_protected_lineage(
    baseline: &VerifiedInteractionRender<'_, '_>,
    candidate: &VerifiedInteractionRender<'_, '_>,
    removed_ids: &[usize],
) -> Result<(), PairwiseInteractionError> {
    let removed: BTreeSet<String> = removed_ids
        .iter()
        .map(|index| format!("score_event_index:{index}"))
        .collect();
    let baseline_protected: Vec<_> = baseline.evidence.performed_event_lineage()
        .iter()
        .filter(|row| !removed.contains(&row.source_event_id))
        .collect();
    let candidate_rows = candidate.evidence.performed_event_lineage();
    if baseline_protected.len() != candidate_rows.len()
        || baseline_protected.iter().any(|baseline_row| {
            !candidate_rows.iter().any(|candidate_row| {
                lineage_matches(baseline_row, candidate_row)
            })
        })
    {
        return Err(PairwiseInteractionError::InvalidCoalition("protected event lineage changed"));
    }
    Ok(())
}

fn validate_input(
    input: &PairwiseInteractionInput<'_, '_>,
) -> Result<usize, PairwiseInteractionError> {
    validate_config(&input.analysis)?;
    let event_a = parse_source_index(input.source_event_a)
        .ok_or(PairwiseInteractionError::InvalidEventId)?;
    let event_b = parse_source_index(input.source_event_b)
        .ok_or(PairwiseInteractionError::InvalidEventId)?;
    if event_a == event_b {
        return Err(PairwiseInteractionError::DuplicateEventIds);
    }
    if input.expected_score_event_count == 0
        || input.expected_score_event_count > MAX_EVENT_COUNT
        || event_a >= input.expected_score_event_count
        || event_b >= input.expected_score_event_count
    {
        return Err(PairwiseInteractionError::InvalidEventCount);
    }

    let baseline_frames = validate_audio(input.baseline.audio)?;
    if input.baseline.audio.sample_rate_hz != input.remove_a.audio.sample_rate_hz
        || input.baseline.audio.sample_rate_hz != input.remove_b.audio.sample_rate_hz
        || input.baseline.audio.sample_rate_hz != input.remove_ab.audio.sample_rate_hz
        || input.baseline.audio.channels != input.remove_a.audio.channels
        || input.baseline.audio.channels != input.remove_b.audio.channels
        || input.baseline.audio.channels != input.remove_ab.audio.channels
        || baseline_frames != validate_audio(input.remove_a.audio)?
        || baseline_frames != validate_audio(input.remove_b.audio)?
        || baseline_frames != validate_audio(input.remove_ab.audio)?
    {
        return Err(PairwiseInteractionError::AudioLayoutMismatch);
    }

    let baseline_evidence = input.baseline.evidence;
    let source_score = baseline_evidence.source_score_sha256();
    let renderer_identity = baseline_evidence.renderer_identity_sha256();
    let renderer_config = baseline_evidence.renderer_config_sha256();
    let expected_pitch_inventory = baseline_evidence.expected_primary_pitches_sha256();
    let rate = input.baseline.audio.sample_rate_hz;
    let channels = input.baseline.audio.channels;

    let all: BTreeSet<usize> = (0..input.expected_score_event_count).collect();
    let without_a: BTreeSet<usize> = all.iter().copied().filter(|id| *id != event_a).collect();
    let without_b: BTreeSet<usize> = all.iter().copied().filter(|id| *id != event_b).collect();
    let without_ab: BTreeSet<usize> = all.iter()
        .copied()
        .filter(|id| *id != event_a && *id != event_b)
        .collect();

    validate_coalition("baseline", &input.baseline, &all, source_score, renderer_identity, renderer_config, rate, channels, baseline_frames)?;
    validate_coalition("remove_a", &input.remove_a, &without_a, source_score, renderer_identity, renderer_config, rate, channels, baseline_frames)?;
    validate_coalition("remove_b", &input.remove_b, &without_b, source_score, renderer_identity, renderer_config, rate, channels, baseline_frames)?;
    validate_coalition("remove_ab", &input.remove_ab, &without_ab, source_score, renderer_identity, renderer_config, rate, channels, baseline_frames)?;

    for render in [
        &input.remove_a,
        &input.remove_b,
        &input.remove_ab,
    ] {
        if render.evidence.expected_primary_pitches_sha256() != expected_pitch_inventory {
            return Err(PairwiseInteractionError::SourceScoreMismatch);
        }
    }

    validate_protected_lineage(&input.baseline, &input.remove_a, &[event_a])?;
    validate_protected_lineage(&input.baseline, &input.remove_b, &[event_b])?;
    validate_protected_lineage(&input.baseline, &input.remove_ab, &[event_a, event_b])?;

    let schedule_digests = [
        baseline_evidence.input_artifact_sha256(),
        input.remove_a.evidence.input_artifact_sha256(),
        input.remove_b.evidence.input_artifact_sha256(),
        input.remove_ab.evidence.input_artifact_sha256(),
    ].into_iter().collect::<BTreeSet<_>>();
    if schedule_digests.len() != 4 {
        return Err(PairwiseInteractionError::DuplicateScheduleDigest);
    }
    let map_digests = [
        baseline_evidence.performed_event_map_sha256(),
        input.remove_a.evidence.performed_event_map_sha256(),
        input.remove_b.evidence.performed_event_map_sha256(),
        input.remove_ab.evidence.performed_event_map_sha256(),
    ].into_iter().collect::<BTreeSet<_>>();
    if map_digests.len() != 4 {
        return Err(PairwiseInteractionError::DuplicateMapDigest);
    }
    Ok(baseline_frames)
}

#[derive(Debug, Clone, Copy, Default)]
struct WindowMetrics {
    pcm_rms: f64,
    pcm_peak: f64,
    main_effect_rms: f64,
    spectral_rms: f64,
    spectral_main_effect_rms: f64,
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|index| 0.5 - 0.5 * (std::f32::consts::TAU * index as f32 / size as f32).cos())
        .collect()
}

fn channel_window_metrics(
    baseline: &AudioView<'_>,
    remove_a: &AudioView<'_>,
    remove_b: &AudioView<'_>,
    remove_ab: &AudioView<'_>,
    window: FrameInterval,
    channel: usize,
    hann: &[f32],
    fft: &dyn Fft<f32>,
) -> WindowMetrics {
    let size = window.end_frame - window.start_frame;
    let mut b = Vec::with_capacity(size);
    let mut a = Vec::with_capacity(size);
    let mut c = Vec::with_capacity(size);
    let mut d = Vec::with_capacity(size);
    let mut pcm_sq = 0.0f64;
    let mut pcm_peak = 0.0f64;
    let mut main_sq = 0.0f64;

    for offset in 0..size {
        let frame = window.start_frame + offset;
        let index = frame * baseline.channels + channel;
        let y00 = baseline.interleaved_samples[index];
        let y10 = remove_a.interleaved_samples[index];
        let y01 = remove_b.interleaved_samples[index];
        let y11 = remove_ab.interleaved_samples[index];
        let residual = y00 as f64 - y10 as f64 - y01 as f64 + y11 as f64;
        let effect_a = y00 as f64 - y10 as f64;
        let effect_b = y00 as f64 - y01 as f64;
        pcm_sq += residual * residual;
        pcm_peak = pcm_peak.max(residual.abs());
        main_sq += 0.5 * (effect_a * effect_a + effect_b * effect_b);
        b.push(Complex::new(y00 * hann[offset], 0.0));
        a.push(Complex::new(y10 * hann[offset], 0.0));
        c.push(Complex::new(y01 * hann[offset], 0.0));
        d.push(Complex::new(y11 * hann[offset], 0.0));
    }
    fft.process(&mut b);
    fft.process(&mut a);
    fft.process(&mut c);
    fft.process(&mut d);

    let mut spectral_sq = 0.0f64;
    let mut spectral_main_sq = 0.0f64;
    let mut bins = 0usize;
    for index in 0..=(size / 2) {
        let y00 = b[index].norm() as f64;
        let y10 = a[index].norm() as f64;
        let y01 = c[index].norm() as f64;
        let y11 = d[index].norm() as f64;
        let residual = y00 - y10 - y01 + y11;
        let main_a = y00 - y10;
        let main_b = y00 - y01;
        spectral_sq += residual * residual;
        spectral_main_sq += 0.5 * (main_a * main_a + main_b * main_b);
        bins += 1;
    }

    WindowMetrics {
        pcm_rms: (pcm_sq / size as f64).sqrt(),
        pcm_peak,
        main_effect_rms: (main_sq / size as f64).sqrt(),
        spectral_rms: (spectral_sq / bins as f64).sqrt(),
        spectral_main_effect_rms: (spectral_main_sq / bins as f64).sqrt(),
    }
}

fn analyze(
    input: &PairwiseInteractionInput<'_, '_>,
) -> Result<Vec<PairwiseInteractionWindow>, PairwiseInteractionError> {
    let frames = validate_input(input)?;
    let windows = windows_for(frames, &input.analysis)?;
    let size = input.analysis.window_size_frames;
    let hann = hann_window(size);
    let fft = FftPlanner::<f32>::new().plan_fft_forward(size);
    let mut results = Vec::with_capacity(windows.len() * input.baseline.audio.channels);

    for window in windows {
        for channel in 0..input.baseline.audio.channels {
            let metric = channel_window_metrics(
                input.baseline.audio,
                input.remove_a.audio,
                input.remove_b.audio,
                input.remove_ab.audio,
                window,
                channel,
                &hann,
                fft.as_ref(),
            );
            results.push(PairwiseInteractionWindow {
                start_frame: window.start_frame,
                end_frame: window.end_frame,
                channel,
                pcm_interaction_rms: metric.pcm_rms,
                pcm_interaction_peak: metric.pcm_peak,
                mean_main_effect_rms: metric.main_effect_rms,
                pcm_interaction_to_main_effect_ratio: metric.pcm_rms
                    / metric.main_effect_rms.max(input.analysis.ratio_epsilon),
                spectral_magnitude_interaction_rms: metric.spectral_rms,
                spectral_magnitude_interaction_to_main_effect_ratio: metric.spectral_rms
                    / metric.spectral_main_effect_rms.max(input.analysis.ratio_epsilon),
            });
        }
    }
    Ok(results)
}

fn seal_report(
    mut report: PairwiseInteractionReport,
) -> Result<PairwiseInteractionReport, PairwiseInteractionError> {
    report.receipt_sha256.clear();
    let digest = canonical_digest(&report)?;
    report.receipt_sha256 = digest;
    Ok(report)
}

/// Produce the four-coalition pairwise interaction report.
///
/// This is a measurement diagnostic only. It deliberately does not classify
/// the renderer as additive/nonlinear; calibration of the four-way null is a
/// separate requirement.
pub fn measure_pairwise_interaction(
    input: &PairwiseInteractionInput<'_, '_>,
) -> Result<PairwiseInteractionReport, PairwiseInteractionError> {
    let windows = analyze(input)?;
    let base = input.baseline.evidence;
    seal_report(PairwiseInteractionReport {
        version: PAIRWISE_INTERACTION_VERSION.to_owned(),
        source_event_a: input.source_event_a.to_owned(),
        source_event_b: input.source_event_b.to_owned(),
        expected_score_event_count: input.expected_score_event_count,
        source_score_sha256: base.source_score_sha256().to_owned(),
        renderer_identity_sha256: base.renderer_identity_sha256().to_owned(),
        renderer_config_sha256: base.renderer_config_sha256().to_owned(),
        sample_rate_hz: input.baseline.audio.sample_rate_hz,
        channels: input.baseline.audio.channels,
        total_frames: input.baseline.audio.interleaved_samples.len() / input.baseline.audio.channels,
        baseline_renderer_receipt_sha256: base.receipt_sha256().to_owned(),
        remove_a_renderer_receipt_sha256: input.remove_a.evidence.receipt_sha256().to_owned(),
        remove_b_renderer_receipt_sha256: input.remove_b.evidence.receipt_sha256().to_owned(),
        remove_ab_renderer_receipt_sha256: input.remove_ab.evidence.receipt_sha256().to_owned(),
        baseline_schedule_sha256: base.input_artifact_sha256().to_owned(),
        remove_a_schedule_sha256: input.remove_a.evidence.input_artifact_sha256().to_owned(),
        remove_b_schedule_sha256: input.remove_b.evidence.input_artifact_sha256().to_owned(),
        remove_ab_schedule_sha256: input.remove_ab.evidence.input_artifact_sha256().to_owned(),
        baseline_map_artifact_sha256: base.performed_event_map_artifact_sha256().to_owned(),
        remove_a_map_artifact_sha256: input.remove_a.evidence.performed_event_map_artifact_sha256().to_owned(),
        remove_b_map_artifact_sha256: input.remove_b.evidence.performed_event_map_artifact_sha256().to_owned(),
        remove_ab_map_artifact_sha256: input.remove_ab.evidence.performed_event_map_artifact_sha256().to_owned(),
        baseline_map_receipt_sha256: base.performed_event_map_sha256().to_owned(),
        remove_a_map_receipt_sha256: input.remove_a.evidence.performed_event_map_sha256().to_owned(),
        remove_b_map_receipt_sha256: input.remove_b.evidence.performed_event_map_sha256().to_owned(),
        remove_ab_map_receipt_sha256: input.remove_ab.evidence.performed_event_map_sha256().to_owned(),
        baseline_audio_sha256: base.audio_sha256().to_owned(),
        remove_a_audio_sha256: input.remove_a.evidence.audio_sha256().to_owned(),
        remove_b_audio_sha256: input.remove_b.evidence.audio_sha256().to_owned(),
        remove_ab_audio_sha256: input.remove_ab.evidence.audio_sha256().to_owned(),
        analysis: input.analysis.clone(),
        total_channel_windows: windows.len(),
        windows,
        receipt_sha256: String::new(),
    })
}

/// Replay the complete four-coalition validation and interaction measurements.
pub fn verify_pairwise_interaction(
    report: &PairwiseInteractionReport,
    input: &PairwiseInteractionInput<'_, '_>,
) -> Result<bool, PairwiseInteractionError> {
    if !report_digest_valid(report)? {
        return Err(PairwiseInteractionError::InvalidReceipt);
    }
    let expected = measure_pairwise_interaction(input)?;
    if &expected != report {
        return Err(PairwiseInteractionError::ReplayMismatch);
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> PairwiseInteractionConfig {
        PairwiseInteractionConfig {
            window_size_frames: 32,
            hop_size_frames: 16,
            ratio_epsilon: 1.0e-12,
        }
    }

    fn audio(samples: &[f32]) -> AudioView<'_> {
        AudioView {
            interleaved_samples: samples,
            sample_rate_hz: 48_000,
            channels: 1,
        }
    }

    fn window_metrics(
        baseline: &[f32],
        remove_a: &[f32],
        remove_b: &[f32],
        remove_ab: &[f32],
    ) -> WindowMetrics {
        let b = audio(baseline);
        let a = audio(remove_a);
        let c = audio(remove_b);
        let d = audio(remove_ab);
        let hann = hann_window(baseline.len());
        let fft = FftPlanner::<f32>::new().plan_fft_forward(baseline.len());
        channel_window_metrics(
            &b,
            &a,
            &c,
            &d,
            FrameInterval {
                start_frame: 0,
                end_frame: baseline.len(),
            },
            0,
            &hann,
            fft.as_ref(),
        )
    }

    #[test]
    fn pcm_interaction_is_zero_for_a_linear_additive_fixture() {
        // A+B = 0.5+0.3. Removing A leaves B; removing B leaves A.
        let baseline = vec![0.8f32; 32];
        let remove_a = vec![0.3f32; 32];
        let remove_b = vec![0.5f32; 32];
        let remove_ab = vec![0.0f32; 32];
        let metrics = window_metrics(&baseline, &remove_a, &remove_b, &remove_ab);
        assert!(metrics.pcm_rms < 1.0e-7);
        assert!(metrics.pcm_peak < 1.0e-7);
    }

    #[test]
    fn pcm_interaction_detects_a_non_additive_fixture() {
        let baseline = vec![0.8f32; 32];
        let remove_a = vec![0.3f32; 32];
        let remove_b = vec![0.4f32; 32];
        let remove_ab = vec![0.0f32; 32];
        let metrics = window_metrics(&baseline, &remove_a, &remove_b, &remove_ab);
        assert!((metrics.pcm_rms - 0.1).abs() < 1.0e-6);
        assert!((metrics.pcm_peak - 0.1).abs() < 1.0e-6);
    }

    #[test]
    fn spectral_magnitude_interaction_can_exist_without_pcm_nonadditivity() {
        // Two equal-frequency sinusoids in quadrature add linearly in PCM.
        // Their magnitude spectra do not add linearly, so this must not be
        // mislabeled as a nonlinear renderer interaction.
        let size = 32usize;
        let signal_a: Vec<f32> = (0..size)
            .map(|index| (std::f32::consts::TAU * 2.0 * index as f32 / size as f32).sin())
            .collect();
        let signal_b: Vec<f32> = (0..size)
            .map(|index| (std::f32::consts::TAU * 2.0 * index as f32 / size as f32).cos())
            .collect();
        let baseline: Vec<f32> = signal_a.iter().zip(&signal_b).map(|(a, b)| a + b).collect();
        let remove_a = signal_b.clone();
        let remove_b = signal_a.clone();
        let remove_ab = vec![0.0f32; size];

        let metrics = window_metrics(&baseline, &remove_a, &remove_b, &remove_ab);
        assert!(metrics.pcm_rms < 1.0e-7);
        assert!(metrics.spectral_rms > 1.0e-3);
    }

    #[test]
    fn analysis_rejects_non_finite_and_malformed_audio() {
        let nan_audio = audio(&[0.0, f32::NAN, 0.0]);
        assert_eq!(
            validate_audio(&nan_audio).unwrap_err(),
            PairwiseInteractionError::NonFiniteSample
        );
        let empty = AudioView {
            interleaved_samples: &[],
            sample_rate_hz: 48_000,
            channels: 1,
        };
        assert_eq!(
            validate_audio(&empty).unwrap_err(),
            PairwiseInteractionError::EmptyAudio
        );
        let invalid = AudioView {
            interleaved_samples: &[0.0, 0.0, 0.0],
            sample_rate_hz: 48_000,
            channels: 2,
        };
        assert_eq!(
            validate_audio(&invalid).unwrap_err(),
            PairwiseInteractionError::NonDivisibleAudio
        );
    }

    #[test]
    fn windows_cover_final_tail_without_overrun() {
        let windows = windows_for(67, &config()).unwrap();
        assert_eq!(windows.first().unwrap().start_frame, 0);
        assert_eq!(windows.last().unwrap().end_frame, 67);
        assert!(windows.iter().all(|window| window.end_frame <= 67));
    }

    #[test]
    fn invalid_event_pair_and_analysis_configuration_are_rejected() {
        assert_eq!(
            parse_source_index("score_event_index:01"),
            None
        );
        assert_eq!(parse_source_index("score_event_index:2"), Some(2));
        assert_eq!(
            validate_config(&PairwiseInteractionConfig {
                ratio_epsilon: 0.0,
                ..config()
            })
            .unwrap_err(),
            PairwiseInteractionError::InvalidConfig(
                "ratio_epsilon must be finite and positive"
            )
        );
    }

    fn lineage_for(indices: &[usize]) -> Vec<crate::performed_event_map::PerformedEventLineage> {
        indices
            .iter()
            .enumerate()
            .map(|(ordinal, source_index)| {
                let onset_micros = (*source_index as u64) * 100_000;
                crate::performed_event_map::PerformedEventLineage {
                    source_event_id: format!("score_event_index:{source_index}"),
                    descendant_ordinal: 0,
                    midi_track_index: 0,
                    note_on_ordinal: ordinal as u32,
                    midi_channel: 0,
                    midi_note: [60, 64, 67][*source_index],
                    onset_micros,
                    end_micros: onset_micros + 50_000,
                    onset_frame: (onset_micros * 48_000 + 500_000) / 1_000_000,
                    end_frame: ((onset_micros + 50_000) * 48_000 + 500_000) / 1_000_000,
                }
            })
            .collect()
    }

    fn binding(
        audio: &AudioView<'_>,
        label: &str,
        schedule: &str,
        map: &str,
        indices: &[usize],
    ) -> VerifiedRendererAudioBinding {
        crate::counterfactual_influence::test_verified_renderer_audio_binding(
            audio,
            sha256_hex(format!("receipt:{label}").as_bytes()),
            sha256_hex(b"fixed-renderer-identity"),
            sha256_hex(b"fixed-renderer-profile"),
            sha256_hex(b"fixed-source-score"),
            sha256_hex(schedule.as_bytes()),
            sha256_hex(format!("map-receipt:{map}").as_bytes()),
            sha256_hex(format!("map-artifact:{map}").as_bytes()),
            sha256_hex(&[60, 64, 67]),
            lineage_for(indices),
        )
    }

    #[test]
    fn four_coalition_report_replays_for_an_additive_pcm_fixture() {
        let base_samples = vec![0.8f32; 64];
        let remove_a_samples = vec![0.3f32; 64];
        let remove_b_samples = vec![0.5f32; 64];
        let remove_ab_samples = vec![0.0f32; 64];
        let base_audio = audio(&base_samples);
        let remove_a_audio = audio(&remove_a_samples);
        let remove_b_audio = audio(&remove_b_samples);
        let remove_ab_audio = audio(&remove_ab_samples);

        let baseline = binding(&base_audio, "baseline", "midi-00", "map-00", &[0, 1, 2]);
        let remove_a = binding(&remove_a_audio, "remove-a", "midi-10", "map-10", &[1, 2]);
        let remove_b = binding(&remove_b_audio, "remove-b", "midi-01", "map-01", &[0, 2]);
        let remove_ab = binding(&remove_ab_audio, "remove-ab", "midi-11", "map-11", &[2]);

        let input = PairwiseInteractionInput {
            source_event_a: "score_event_index:0",
            source_event_b: "score_event_index:1",
            expected_score_event_count: 3,
            baseline: VerifiedInteractionRender { evidence: &baseline, audio: &base_audio },
            remove_a: VerifiedInteractionRender { evidence: &remove_a, audio: &remove_a_audio },
            remove_b: VerifiedInteractionRender { evidence: &remove_b, audio: &remove_b_audio },
            remove_ab: VerifiedInteractionRender { evidence: &remove_ab, audio: &remove_ab_audio },
            analysis: config(),
        };

        let report = measure_pairwise_interaction(&input).unwrap();
        assert_eq!(report.windows.len(), report.total_channel_windows);
        assert!(report.windows.iter().all(|window| window.pcm_interaction_rms < 1.0e-7));
        assert!(verify_pairwise_interaction(&report, &input).unwrap());
    }

    #[test]
    fn coalition_validator_rejects_a_double_removal_that_keeps_one_target_event() {
        let base_samples = vec![0.8f32; 64];
        let remove_a_samples = vec![0.3f32; 64];
        let remove_b_samples = vec![0.5f32; 64];
        let remove_ab_samples = vec![0.0f32; 64];
        let base_audio = audio(&base_samples);
        let remove_a_audio = audio(&remove_a_samples);
        let remove_b_audio = audio(&remove_b_samples);
        let remove_ab_audio = audio(&remove_ab_samples);

        let baseline = binding(&base_audio, "baseline", "midi-00", "map-00", &[0, 1, 2]);
        let remove_a = binding(&remove_a_audio, "remove-a", "midi-10", "map-10", &[1, 2]);
        let remove_b = binding(&remove_b_audio, "remove-b", "midi-01", "map-01", &[0, 2]);
        // Event 0 remains, despite the declared coalition requiring both 0 and 1 removed.
        let remove_ab = binding(&remove_ab_audio, "wrong-remove-ab", "midi-11", "map-11", &[0, 2]);

        let input = PairwiseInteractionInput {
            source_event_a: "score_event_index:0",
            source_event_b: "score_event_index:1",
            expected_score_event_count: 3,
            baseline: VerifiedInteractionRender { evidence: &baseline, audio: &base_audio },
            remove_a: VerifiedInteractionRender { evidence: &remove_a, audio: &remove_a_audio },
            remove_b: VerifiedInteractionRender { evidence: &remove_b, audio: &remove_b_audio },
            remove_ab: VerifiedInteractionRender { evidence: &remove_ab, audio: &remove_ab_audio },
            analysis: config(),
        };
        assert_eq!(
            measure_pairwise_interaction(&input).unwrap_err(),
            PairwiseInteractionError::InvalidCoalition("remove_ab")
        );
    }

    #[test]
    fn coalition_validator_rejects_renderer_profile_substitution() {
        let base_samples = vec![0.8f32; 64];
        let remove_a_samples = vec![0.3f32; 64];
        let remove_b_samples = vec![0.5f32; 64];
        let remove_ab_samples = vec![0.0f32; 64];
        let base_audio = audio(&base_samples);
        let remove_a_audio = audio(&remove_a_samples);
        let remove_b_audio = audio(&remove_b_samples);
        let remove_ab_audio = audio(&remove_ab_samples);

        let baseline = binding(&base_audio, "baseline", "midi-00", "map-00", &[0, 1, 2]);
        let remove_a = binding(&remove_a_audio, "remove-a", "midi-10", "map-10", &[1, 2]);
        let remove_b = binding(&remove_b_audio, "remove-b", "midi-01", "map-01", &[0, 2]);
        let mut remove_ab = binding(&remove_ab_audio, "remove-ab", "midi-11", "map-11", &[2]);
        // The test-only constructor is deliberately used to exercise the
        // fail-closed consumer boundary with a substituted execution profile.
        let remove_ab = crate::counterfactual_influence::test_verified_renderer_audio_binding(
            &remove_ab_audio,
            remove_ab.receipt_sha256().to_owned(),
            remove_ab.renderer_identity_sha256().to_owned(),
            sha256_hex(b"substituted-renderer-profile"),
            remove_ab.source_score_sha256().to_owned(),
            remove_ab.input_artifact_sha256().to_owned(),
            remove_ab.performed_event_map_sha256().to_owned(),
            remove_ab.performed_event_map_artifact_sha256().to_owned(),
            remove_ab.expected_primary_pitches_sha256().to_owned(),
            remove_ab.performed_event_lineage().to_vec(),
        );

        let input = PairwiseInteractionInput {
            source_event_a: "score_event_index:0",
            source_event_b: "score_event_index:1",
            expected_score_event_count: 3,
            baseline: VerifiedInteractionRender { evidence: &baseline, audio: &base_audio },
            remove_a: VerifiedInteractionRender { evidence: &remove_a, audio: &remove_a_audio },
            remove_b: VerifiedInteractionRender { evidence: &remove_b, audio: &remove_b_audio },
            remove_ab: VerifiedInteractionRender { evidence: &remove_ab, audio: &remove_ab_audio },
            analysis: config(),
        };
        assert_eq!(
            measure_pairwise_interaction(&input).unwrap_err(),
            PairwiseInteractionError::RendererIdentityMismatch
        );
    }
}
