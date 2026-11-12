// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Four-coalition paired-render measurements of musical-event interaction.
//!
//! The four exact renders are:
//!   Y00 = R(S),       Y10 = R(S \ A),
//!   Y01 = R(S \ B),   Y11 = R(S \ {A, B}).
//!
//! This module computes second-order finite differences in sample space and
//! in windowed spectral-magnitude space. The sample-space residual measures
//! departure from additive waveform composition. The spectral residual is a
//! different diagnostic: it can be nonzero under perfectly additive PCM
//! synthesis because magnitude is nonlinear with respect to phase.
//!
//! The module never labels an interaction as nonlinear, harmful, or pass/fail.
//! Such decisions require calibrated null controls, common-random-number
//! evidence, independent coalition validation, and human/perceptual review.
//! This is measurement evidence, not source ownership or renderer attestation.

use crate::audio_preservation::AudioView;
use crate::counterfactual_influence::{
    CounterfactualInfluenceError, VerifiedRendererAudioBinding,
    verify_renderer_audio_binding_for_score,
    verify_renderer_audio_binding_for_score_excluding_set,
};
use crate::performed_event_map::PerformedEventLineage;
use crate::renderer_execution_receipt::{
    RendererExecutionReceipt, RendererReceiptInputs, sha256_hex,
};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use symthaea_music_theory::Score;

pub const PAIRWISE_ACOUSTIC_INTERACTION_VERSION: &str =
    "mel-mr-eval-018-pairwise-acoustic-interaction-v0.1";
const MAX_WINDOW_FRAMES: usize = 1_048_576;
const MAX_SCORE_EVENTS: usize = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairedRandomnessMode {
    /// The backend promises bit-identical output for identical frozen inputs.
    DeterministicReplay,
    /// The four renders share an identical, explicitly committed random stream.
    CommonRandomNumbers,
    /// Seeds/streams are not paired; observations are exploratory only.
    UncontrolledExploratory,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairwiseInteractionRequest {
    pub source_event_a_id: String,
    pub source_event_b_id: String,
    pub source_event_a_index: usize,
    pub source_event_b_index: usize,
    pub paired_randomness_mode: PairedRandomnessMode,
    /// Digest of a frozen randomness/seed policy record. This module binds it;
    /// the caller must independently establish the policy it describes.
    pub randomness_evidence_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairwiseInteractionConfig {
    pub window_size_frames: usize,
    pub hop_size_frames: usize,
    /// Small denominator guard for normalized spectral interaction.
    pub spectral_energy_epsilon: f32,
}

impl Default for PairwiseInteractionConfig {
    fn default() -> Self {
        Self {
            window_size_frames: 2048,
            hop_size_frames: 512,
            spectral_energy_epsilon: 1.0e-12,
        }
    }
}

/// One exact coalition render and its source-bound evidence.
pub struct PairwiseRenderEvidence<'a> {
    pub receipt: &'a RendererExecutionReceipt,
    pub receipt_inputs: RendererReceiptInputs<'a>,
    pub performed_event_map_bytes: &'a [u8],
    pub audio: AudioView<'a>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairwiseInteractionWindow {
    pub start_frame: usize,
    pub end_frame: usize,
    pub channel: usize,
    /// RMS of y00 - y10 - y01 + y11, in normalized PCM units.
    pub pcm_second_order_rms: f32,
    /// Peak absolute value of the PCM second-order residual.
    pub pcm_second_order_peak: f32,
    /// RMS of |FFT(y00)| - |FFT(y10)| - |FFT(y01)| + |FFT(y11)|.
    /// This is a representation-specific diagnostic, not a nonlinearity test.
    pub spectral_magnitude_second_order_rms: f32,
    /// Spectral residual RMS divided by the sum of coalition magnitudes.
    pub normalized_spectral_second_order_residual: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairwiseInteractionDisposition {
    /// Exact coalition-bound measurements only. No calibrated null decision.
    MeasurementOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairwiseInteractionReport {
    pub version: String,
    pub source_score_sha256: String,
    pub source_event_a_id: String,
    pub source_event_b_id: String,
    pub source_event_a_index: usize,
    pub source_event_b_index: usize,
    pub coalition_schedule_sha256: [String; 4],
    pub coalition_map_artifact_sha256: [String; 4],
    pub coalition_map_receipt_sha256: [String; 4],
    pub coalition_renderer_receipt_sha256: [String; 4],
    pub coalition_audio_sha256: [String; 4],
    pub renderer_identity_sha256: String,
    pub renderer_config_sha256: String,
    pub paired_randomness_mode: PairedRandomnessMode,
    pub randomness_evidence_sha256: String,
    pub sample_rate_hz: u32,
    pub channels: usize,
    pub total_frames: usize,
    pub config: PairwiseInteractionConfig,
    pub windows: Vec<PairwiseInteractionWindow>,
    pub maximum_pcm_second_order_rms: f32,
    pub maximum_pcm_second_order_peak: f32,
    pub maximum_spectral_magnitude_second_order_rms: f32,
    pub disposition: PairwiseInteractionDisposition,
    pub reasons: Vec<String>,
    pub receipt_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairwiseInteractionError {
    SameSourceEvent,
    InvalidSourceEvent,
    InvalidSha256(&'static str),
    InvalidConfig(&'static str),
    InvalidRendererReceipt,
    RendererProfileMismatch,
    AudioLayoutMismatch,
    WindowLongerThanAudio,
    CoalitionScheduleMismatch,
    DuplicateCoalitionArtifact,
    RetainedEventLineageChanged,
    SerializationFailure,
    InvalidReportReceipt,
    ReplayMismatch,
}

impl std::fmt::Display for PairwiseInteractionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SameSourceEvent => write!(f, "pairwise interaction requires two distinct source events"),
            Self::InvalidSourceEvent => write!(f, "source event IDs or indices are invalid"),
            Self::InvalidSha256(field) => write!(f, "invalid SHA-256 field: {field}"),
            Self::InvalidConfig(reason) => write!(f, "invalid pairwise interaction config: {reason}"),
            Self::InvalidRendererReceipt => write!(f, "one or more renderer receipts failed replay"),
            Self::RendererProfileMismatch => write!(f, "four coalition renders do not share one renderer profile"),
            Self::AudioLayoutMismatch => write!(f, "four coalition audio layouts differ"),
            Self::WindowLongerThanAudio => write!(f, "analysis window is longer than the coalition audio"),
            Self::CoalitionScheduleMismatch => write!(f, "one or more exact coalition schedules are invalid"),
            Self::DuplicateCoalitionArtifact => write!(f, "coalition schedules, maps, or renderer receipts are reused"),
            Self::RetainedEventLineageChanged => write!(f, "a retained source event changed identity, pitch, track, channel, or time"),
            Self::SerializationFailure => write!(f, "could not serialize pairwise interaction report"),
            Self::InvalidReportReceipt => write!(f, "pairwise interaction report digest is invalid"),
            Self::ReplayMismatch => write!(f, "pairwise interaction report does not replay"),
        }
    }
}

impl std::error::Error for PairwiseInteractionError {}

struct VerifiedCoalition<'a> {
    binding: VerifiedRendererAudioBinding,
    audio: AudioView<'a>,
    schedule_sha256: String,
    map_artifact_sha256: String,
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

fn validate_config(config: &PairwiseInteractionConfig) -> Result<(), PairwiseInteractionError> {
    if config.window_size_frames < 16 || config.window_size_frames > MAX_WINDOW_FRAMES {
        return Err(PairwiseInteractionError::InvalidConfig(
            "window_size_frames must be in 16..=1048576",
        ));
    }
    if config.hop_size_frames == 0 || config.hop_size_frames > config.window_size_frames {
        return Err(PairwiseInteractionError::InvalidConfig(
            "hop_size_frames must be in 1..=window_size_frames",
        ));
    }
    if !config.spectral_energy_epsilon.is_finite()
        || config.spectral_energy_epsilon <= 0.0
        || config.spectral_energy_epsilon > 1.0
    {
        return Err(PairwiseInteractionError::InvalidConfig(
            "spectral_energy_epsilon must be finite and in (0, 1]",
        ));
    }
    Ok(())
}

fn validate_audio(audio: &AudioView<'_>) -> Result<usize, PairwiseInteractionError> {
    if audio.sample_rate_hz == 0
        || audio.channels == 0
        || audio.interleaved_samples.is_empty()
        || audio.interleaved_samples.len() % audio.channels != 0
        || audio.interleaved_samples.iter().any(|sample| !sample.is_finite())
    {
        return Err(PairwiseInteractionError::AudioLayoutMismatch);
    }
    Ok(audio.interleaved_samples.len() / audio.channels)
}

fn verify_coalition<'a>(
    evidence: PairwiseRenderEvidence<'a>,
    score: &Score,
    excluded: &[usize],
) -> Result<VerifiedCoalition<'a>, PairwiseInteractionError> {
    let midi_bytes = evidence
        .receipt_inputs
        .input_midi_bytes
        .ok_or(PairwiseInteractionError::InvalidRendererReceipt)?;
    let schedule_sha256 = sha256_hex(midi_bytes);
    let binding = if excluded.is_empty() {
        verify_renderer_audio_binding_for_score(
            evidence.receipt,
            evidence.receipt_inputs,
            evidence.performed_event_map_bytes,
            score,
            None,
            &evidence.audio,
        )
    } else {
        verify_renderer_audio_binding_for_score_excluding_set(
            evidence.receipt,
            evidence.receipt_inputs,
            evidence.performed_event_map_bytes,
            score,
            excluded,
            &evidence.audio,
        )
    }
    .map_err(|_| PairwiseInteractionError::InvalidRendererReceipt)?;

    Ok(VerifiedCoalition {
        binding,
        audio: evidence.audio,
        schedule_sha256,
        map_artifact_sha256: sha256_hex(evidence.performed_event_map_bytes),
    })
}

fn lineage_key(event: &PerformedEventLineage) -> (String, u32) {
    (event.source_event_id.clone(), event.descendant_ordinal)
}

/// Compare every retained event field except note_on_ordinal. Removing an
/// earlier note legitimately shifts subsequent note-on ordinals within a MIDI
/// track, but must not change the retained event's stable source identity,
/// descendant, track/channel, pitch or scheduled interval.
fn retained_lineage_equal(
    baseline: &[PerformedEventLineage],
    candidate: &[PerformedEventLineage],
    excluded: &[usize],
) -> bool {
    let excluded_ids: std::collections::BTreeSet<String> =
        excluded.iter().map(|index| format!("score_event_index:{index}")).collect();
    let baseline_by_key: BTreeMap<(String, u32), &PerformedEventLineage> = baseline
        .iter()
        .filter(|event| !excluded_ids.contains(&event.source_event_id))
        .map(|event| (lineage_key(event), event))
        .collect();
    let candidate_by_key: BTreeMap<(String, u32), &PerformedEventLineage> = candidate
        .iter()
        .map(|event| (lineage_key(event), event))
        .collect();
    if baseline_by_key.len() != candidate_by_key.len() {
        return false;
    }
    baseline_by_key.iter().all(|(key, left)| {
        candidate_by_key.get(key).is_some_and(|right| {
            left.source_event_id == right.source_event_id
                && left.descendant_ordinal == right.descendant_ordinal
                && left.midi_track_index == right.midi_track_index
                && left.midi_channel == right.midi_channel
                && left.midi_note == right.midi_note
                && left.onset_micros == right.onset_micros
                && left.end_micros == right.end_micros
                && left.onset_frame == right.onset_frame
                && left.end_frame == right.end_frame
        })
    })
}

fn window_starts(
    total_frames: usize,
    config: &PairwiseInteractionConfig,
) -> Vec<(usize, usize)> {
    let mut windows = Vec::new();
    let final_start = total_frames - config.window_size_frames;
    let mut start = 0usize;
    loop {
        let end = start + config.window_size_frames;
        windows.push((start, end));
        if start >= final_start {
            break;
        }
        let next = start.saturating_add(config.hop_size_frames);
        if next > final_start {
            if windows.last().is_none_or(|window| window.0 != final_start) {
                windows.push((final_start, total_frames));
            }
            break;
        }
        start = next;
    }
    windows
}

fn mono_window(
    audio: &AudioView<'_>,
    start: usize,
    end: usize,
    channel: usize,
) -> Vec<f32> {
    (start..end)
        .map(|frame| audio.interleaved_samples[frame * audio.channels + channel])
        .collect()
}

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|index| {
            if size <= 1 {
                1.0
            } else {
                0.5 - 0.5
                    * (std::f32::consts::TAU * index as f32 / (size - 1) as f32).cos()
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy, Default)]
struct WindowInteraction {
    pcm_rms: f32,
    pcm_peak: f32,
    spectral_rms: f32,
    normalized_spectral: f32,
}

fn compute_window_interaction(
    y00: &[f32],
    y10: &[f32],
    y01: &[f32],
    y11: &[f32],
    taper: &[f32],
    fft: &dyn Fft<f32>,
    epsilon: f32,
) -> WindowInteraction {
    let mut pcm_sum_sq = 0.0f64;
    let mut pcm_peak = 0.0f32;
    let mut s00 = Vec::with_capacity(y00.len());
    let mut s10 = Vec::with_capacity(y10.len());
    let mut s01 = Vec::with_capacity(y01.len());
    let mut s11 = Vec::with_capacity(y11.len());
    for index in 0..y00.len() {
        let residual = y00[index] - y10[index] - y01[index] + y11[index];
        pcm_sum_sq += f64::from(residual) * f64::from(residual);
        pcm_peak = pcm_peak.max(residual.abs());
        s00.push(Complex::new(y00[index] * taper[index], 0.0));
        s10.push(Complex::new(y10[index] * taper[index], 0.0));
        s01.push(Complex::new(y01[index] * taper[index], 0.0));
        s11.push(Complex::new(y11[index] * taper[index], 0.0));
    }
    fft.process(&mut s00);
    fft.process(&mut s10);
    fft.process(&mut s01);
    fft.process(&mut s11);

    let mut spectral_sum_sq = 0.0f64;
    let mut spectral_energy = 0.0f64;
    for index in 0..=(y00.len() / 2) {
        let m00 = s00[index].norm();
        let m10 = s10[index].norm();
        let m01 = s01[index].norm();
        let m11 = s11[index].norm();
        let residual = m00 - m10 - m01 + m11;
        spectral_sum_sq += f64::from(residual) * f64::from(residual);
        spectral_energy += f64::from(m00 + m10 + m01 + m11);
    }

    let pcm_rms = (pcm_sum_sq / y00.len() as f64).sqrt() as f32;
    let spectral_rms = (spectral_sum_sq / (y00.len() / 2 + 1) as f64).sqrt() as f32;
    WindowInteraction {
        pcm_rms,
        pcm_peak,
        spectral_rms,
        normalized_spectral: (f64::from(spectral_rms)
            / spectral_energy.max(f64::from(epsilon))) as f32,
    }
}

fn audio_sha256(audio: &AudioView<'_>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous-dynamics-mel-mr-eval-018-audio-v1\0");
    hasher.update(audio.sample_rate_hz.to_le_bytes());
    hasher.update((audio.channels as u64).to_le_bytes());
    hasher.update(((audio.interleaved_samples.len() / audio.channels) as u64).to_le_bytes());
    for sample in audio.interleaved_samples {
        hasher.update(sample.to_bits().to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn report_digest(report: &PairwiseInteractionReport) -> Result<String, PairwiseInteractionError> {
    let mut unsealed = report.clone();
    unsealed.receipt_sha256.clear();
    let bytes = serde_json::to_vec(&unsealed)
        .map_err(|_| PairwiseInteractionError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous-dynamics-mel-mr-eval-018-pairwise-interaction-v1\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Measure the 2×2 event interaction for four exact, source-bound FluidSynth
/// renders. Coalitions are verified from MIDI bytes and parsed lineage maps:
/// baseline contains all Score events, and each other render excludes exactly
/// the declared event set. The final report is measurement-only.
pub fn measure_pairwise_acoustic_interaction(
    score: &Score,
    event_a_index: usize,
    event_b_index: usize,
    y00: PairwiseRenderEvidence<'_>,
    y10_remove_a: PairwiseRenderEvidence<'_>,
    y01_remove_b: PairwiseRenderEvidence<'_>,
    y11_remove_both: PairwiseRenderEvidence<'_>,
    request: &PairwiseInteractionRequest,
    config: &PairwiseInteractionConfig,
) -> Result<PairwiseInteractionReport, PairwiseInteractionError> {
    validate_config(config)?;
    let events = score.events();
    if event_a_index == event_b_index
        || event_a_index >= events.len()
        || event_b_index >= events.len()
        || events.len() > MAX_SCORE_EVENTS
    {
        return Err(PairwiseInteractionError::SameSourceEvent);
    }
    let expected_a = format!("score_event_index:{event_a_index}");
    let expected_b = format!("score_event_index:{event_b_index}");
    if request.source_event_a_index != event_a_index
        || request.source_event_b_index != event_b_index
        || request.source_event_a_id != expected_a
        || request.source_event_b_id != expected_b
    {
        return Err(PairwiseInteractionError::InvalidSourceEvent);
    }
    validate_sha256("randomness_evidence_sha256", &request.randomness_evidence_sha256)?;

    let baseline = verify_coalition(y00, score, &[])?;
    let remove_a = verify_coalition(y10_remove_a, score, &[event_a_index])?;
    let remove_b = verify_coalition(y01_remove_b, score, &[event_b_index])?;
    let remove_both = verify_coalition(y11_remove_both, score, &[event_a_index, event_b_index])?;
    let coalitions = [&baseline, &remove_a, &remove_b, &remove_both];

    let baseline_binding = &baseline.binding;
    let total_frames = validate_audio(&baseline.audio)?;
    for coalition in &coalitions[1..] {
        let frames = validate_audio(&coalition.audio)?;
        if coalition.audio.sample_rate_hz != baseline.audio.sample_rate_hz
            || coalition.audio.channels != baseline.audio.channels
            || frames != total_frames
        {
            return Err(PairwiseInteractionError::AudioLayoutMismatch);
        }
        if coalition.binding.source_score_sha256() != baseline_binding.source_score_sha256()
            || coalition.binding.renderer_identity_sha256()
                != baseline_binding.renderer_identity_sha256()
            || coalition.binding.renderer_config_sha256()
                != baseline_binding.renderer_config_sha256()
        {
            return Err(PairwiseInteractionError::RendererProfileMismatch);
        }
    }

    if !retained_lineage_equal(
        baseline_binding.performed_event_lineage(),
        remove_a.binding.performed_event_lineage(),
        &[event_a_index],
    ) || !retained_lineage_equal(
        baseline_binding.performed_event_lineage(),
        remove_b.binding.performed_event_lineage(),
        &[event_b_index],
    ) || !retained_lineage_equal(
        baseline_binding.performed_event_lineage(),
        remove_both.binding.performed_event_lineage(),
        &[event_a_index, event_b_index],
    ) {
        return Err(PairwiseInteractionError::RetainedEventLineageChanged);
    }

    if config.window_size_frames > total_frames {
        return Err(PairwiseInteractionError::WindowLongerThanAudio);
    }

    let schedules = [
        baseline.schedule_sha256.clone(),
        remove_a.schedule_sha256.clone(),
        remove_b.schedule_sha256.clone(),
        remove_both.schedule_sha256.clone(),
    ];
    let map_artifacts = [
        baseline.map_artifact_sha256.clone(),
        remove_a.map_artifact_sha256.clone(),
        remove_b.map_artifact_sha256.clone(),
        remove_both.map_artifact_sha256.clone(),
    ];
    let map_receipts = [
        baseline_binding.performed_event_map_sha256().to_owned(),
        remove_a.binding.performed_event_map_sha256().to_owned(),
        remove_b.binding.performed_event_map_sha256().to_owned(),
        remove_both.binding.performed_event_map_sha256().to_owned(),
    ];
    let renderer_receipts = [
        baseline_binding.receipt_sha256().to_owned(),
        remove_a.binding.receipt_sha256().to_owned(),
        remove_b.binding.receipt_sha256().to_owned(),
        remove_both.binding.receipt_sha256().to_owned(),
    ];
    let audio_digests = [
        audio_sha256(&baseline.audio),
        audio_sha256(&remove_a.audio),
        audio_sha256(&remove_b.audio),
        audio_sha256(&remove_both.audio),
    ];

    for group in [&schedules[..], &map_artifacts[..], &map_receipts[..]] {
        let unique: std::collections::BTreeSet<&str> = group.iter().map(String::as_str).collect();
        if unique.len() != 4 {
            return Err(PairwiseInteractionError::DuplicateCoalitionArtifact);
        }
    }

    let windows = window_starts(total_frames, config);
    let mut taper = hann_window(config.window_size_frames);
    if taper.len() != config.window_size_frames {
        return Err(PairwiseInteractionError::InvalidConfig("window taper length mismatch"));
    }
    let fft = FftPlanner::<f32>::new().plan_fft_forward(config.window_size_frames);
    let mut measurements = Vec::with_capacity(windows.len() * baseline.audio.channels);
    let mut maximum_pcm_rms = 0.0f32;
    let mut maximum_pcm_peak = 0.0f32;
    let mut maximum_spectral_rms = 0.0f32;

    for (start_frame, end_frame) in windows {
        for channel in 0..baseline.audio.channels {
            let y00 = mono_window(&baseline.audio, start_frame, end_frame, channel);
            let y10 = mono_window(&remove_a.audio, start_frame, end_frame, channel);
            let y01 = mono_window(&remove_b.audio, start_frame, end_frame, channel);
            let y11 = mono_window(&remove_both.audio, start_frame, end_frame, channel);
            let metric = compute_window_interaction(
                &y00,
                &y10,
                &y01,
                &y11,
                &taper,
                fft.as_ref(),
                config.spectral_energy_epsilon,
            );
            maximum_pcm_rms = maximum_pcm_rms.max(metric.pcm_rms);
            maximum_pcm_peak = maximum_pcm_peak.max(metric.pcm_peak);
            maximum_spectral_rms = maximum_spectral_rms.max(metric.spectral_rms);
            measurements.push(PairwiseInteractionWindow {
                start_frame,
                end_frame,
                channel,
                pcm_second_order_rms: metric.pcm_rms,
                pcm_second_order_peak: metric.pcm_peak,
                spectral_magnitude_second_order_rms: metric.spectral_rms,
                normalized_spectral_second_order_residual: metric.normalized_spectral,
            });
        }
    }

    let mut reasons = vec![
        "second-order finite differences are representation-specific measurements, not exclusive source ownership".to_owned(),
        "no calibrated additive null distribution is applied; the report does not classify linearity or nonlinearity".to_owned(),
    ];
    if request.paired_randomness_mode == PairedRandomnessMode::UncontrolledExploratory {
        reasons.push("render randomness was not paired; measurements are exploratory only".to_owned());
    }
    let mut report = PairwiseInteractionReport {
        version: PAIRWISE_ACOUSTIC_INTERACTION_VERSION.to_owned(),
        source_score_sha256: baseline_binding.source_score_sha256().to_owned(),
        source_event_a_id: expected_a,
        source_event_b_id: expected_b,
        source_event_a_index: event_a_index,
        source_event_b_index: event_b_index,
        coalition_schedule_sha256: schedules,
        coalition_map_artifact_sha256: map_artifacts,
        coalition_map_receipt_sha256: map_receipts,
        coalition_renderer_receipt_sha256: renderer_receipts,
        coalition_audio_sha256: audio_digests,
        renderer_identity_sha256: baseline_binding.renderer_identity_sha256().to_owned(),
        renderer_config_sha256: baseline_binding.renderer_config_sha256().to_owned(),
        paired_randomness_mode: request.paired_randomness_mode,
        randomness_evidence_sha256: request.randomness_evidence_sha256.clone(),
        sample_rate_hz: baseline.audio.sample_rate_hz,
        channels: baseline.audio.channels,
        total_frames,
        config: config.clone(),
        windows: measurements,
        maximum_pcm_second_order_rms: maximum_pcm_rms,
        maximum_pcm_second_order_peak: maximum_pcm_peak,
        maximum_spectral_magnitude_second_order_rms: maximum_spectral_rms,
        disposition: PairwiseInteractionDisposition::MeasurementOnly,
        reasons: std::mem::take(&mut reasons),
        receipt_sha256: String::new(),
    };
    report.receipt_sha256 = report_digest(&report)?;
    Ok(report)
}

/// Recompute a report from the same four exact coalition artifacts.
pub fn verify_pairwise_acoustic_interaction_report(
    report: &PairwiseInteractionReport,
    score: &Score,
    y00: PairwiseRenderEvidence<'_>,
    y10_remove_a: PairwiseRenderEvidence<'_>,
    y01_remove_b: PairwiseRenderEvidence<'_>,
    y11_remove_both: PairwiseRenderEvidence<'_>,
) -> Result<bool, PairwiseInteractionError> {
    if report.receipt_sha256.is_empty() || report_digest(report)? != report.receipt_sha256 {
        return Err(PairwiseInteractionError::InvalidReportReceipt);
    }
    let request = PairwiseInteractionRequest {
        source_event_a_id: report.source_event_a_id.clone(),
        source_event_b_id: report.source_event_b_id.clone(),
        source_event_a_index: report.source_event_a_index,
        source_event_b_index: report.source_event_b_index,
        paired_randomness_mode: report.paired_randomness_mode,
        randomness_evidence_sha256: report.randomness_evidence_sha256.clone(),
    };
    let expected = measure_pairwise_acoustic_interaction(
        score,
        report.source_event_a_index,
        report.source_event_b_index,
        y00,
        y10_remove_a,
        y01_remove_b,
        y11_remove_both,
        &request,
        &report.config,
    )?;
    Ok(&expected == report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(frames: usize, frequency: f32, phase: f32) -> Vec<f32> {
        (0..frames)
            .map(|index| {
                (std::f32::consts::TAU * frequency * index as f32 / 48_000.0 + phase).sin() * 0.2
            })
            .collect()
    }

    fn interaction(
        y00: &[f32],
        y10: &[f32],
        y01: &[f32],
        y11: &[f32],
    ) -> WindowInteraction {
        let fft = FftPlanner::<f32>::new().plan_fft_forward(y00.len());
        compute_window_interaction(
            y00,
            y10,
            y01,
            y11,
            &hann_window(y00.len()),
            fft.as_ref(),
            1.0e-12,
        )
    }

    #[test]
    fn additive_pcm_coalitions_cancel_the_second_order_residual() {
        let a = sine(1024, 440.0, 0.0);
        let b = sine(1024, 660.0, 0.4);
        let background = sine(1024, 110.0, 0.7);
        let y00: Vec<f32> = (0..1024).map(|i| background[i] + a[i] + b[i]).collect();
        let y10: Vec<f32> = (0..1024).map(|i| background[i] + b[i]).collect();
        let y01: Vec<f32> = (0..1024).map(|i| background[i] + a[i]).collect();
        let y11 = background.clone();
        let measured = interaction(&y00, &y10, &y01, &y11);

        assert!(measured.pcm_second_order_rms < 1.0e-6);
        assert!(measured.pcm_second_order_peak < 1.0e-6);
    }

    #[test]
    fn phase_sensitive_magnitude_interaction_can_exist_without_pcm_interaction() {
        let a = sine(1024, 440.0, 0.0);
        let b = sine(1024, 440.0, std::f32::consts::PI / 2.0);
        let background = sine(1024, 110.0, 0.7);
        let y00: Vec<f32> = (0..1024).map(|i| background[i] + a[i] + b[i]).collect();
        let y10: Vec<f32> = (0..1024).map(|i| background[i] + b[i]).collect();
        let y01: Vec<f32> = (0..1024).map(|i| background[i] + a[i]).collect();
        let y11 = background.clone();
        let measured = interaction(&y00, &y10, &y01, &y11);

        assert!(measured.pcm_second_order_rms < 1.0e-6);
        assert!(measured.spectral_magnitude_second_order_rms > 1.0e-3);
    }

    #[test]
    fn nonlinear_shared_saturation_creates_a_pcm_second_order_residual() {
        let a = sine(1024, 440.0, 0.0);
        let b = sine(1024, 660.0, 0.3);
        let background = sine(1024, 110.0, 0.7);
        let y00: Vec<f32> = (0..1024).map(|i| (background[i] + a[i] + b[i]).tanh()).collect();
        let y10: Vec<f32> = (0..1024).map(|i| (background[i] + b[i]).tanh()).collect();
        let y01: Vec<f32> = (0..1024).map(|i| (background[i] + a[i]).tanh()).collect();
        let y11: Vec<f32> = background.iter().map(|value| value.tanh()).collect();
        let measured = interaction(&y00, &y10, &y01, &y11);

        assert!(measured.pcm_second_order_rms > 1.0e-3);
        assert!(measured.pcm_second_order_peak > 1.0e-3);
    }

    #[test]
    fn event_lineage_comparison_ignores_only_note_on_ordinal_shifts() {
        let baseline = PerformedEventLineage {
            source_event_id: "score_event_index:2".into(),
            descendant_ordinal: 0,
            midi_track_index: 1,
            note_on_ordinal: 8,
            midi_channel: 0,
            midi_note: 67,
            onset_micros: 1_000_000,
            end_micros: 1_500_000,
            onset_frame: 48_000,
            end_frame: 72_000,
        };
        let mut retained = baseline.clone();
        retained.note_on_ordinal = 7;
        assert!(retained_lineage_equal(&[baseline.clone()], &[retained.clone()], &[0]));
        retained.midi_note = 66;
        assert!(!retained_lineage_equal(&[baseline], &[retained], &[0]));
    }

    #[test]
    fn identical_event_indices_reject() {
        assert_eq!(
            validate_source_pair(3, 3, 4),
            Err(PairwiseInteractionError::SameSourceEvent)
        );
    }

    fn validate_source_pair(a: usize, b: usize, count: usize) -> Result<(), PairwiseInteractionError> {
        if a == b {
            return Err(PairwiseInteractionError::SameSourceEvent);
        }
        if a >= count || b >= count {
            return Err(PairwiseInteractionError::InvalidSourceEvent);
        }
        Ok(())
    }
}
