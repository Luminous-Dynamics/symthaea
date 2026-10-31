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
use crate::performed_event_map::{PerformedEventLineage, PerformedEventMap};
use crate::renderer_execution_receipt::{RendererExecutionReceipt, RendererReceiptInputs};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const COUNTERFACTUAL_INFLUENCE_VERSION: &str =
    "mel-mr-eval-017-counterfactual-influence-v0.2";
pub const INFLUENCE_SUPPORT_POLICY_VERSION: &str =
    "mel-mr-eval-017-support-policy-v1";
pub const REPEATABILITY_CALIBRATION_VERSION: &str =
    "mel-mr-eval-017-repeatability-calibration-v0.1";
const CALIBRATION_DOMAIN: &[u8] = b"luminous-dynamics-mel-mr-eval-017-repeatability-v1\0";
const INFLUENCE_DOMAIN: &[u8] = b"luminous-dynamics-mel-mr-eval-017-influence-v2\0";
const MINIMUM_NOOP_RENDERS: usize = 3;
const MAX_EVENT_ID_BYTES: usize = 256;
const MAX_ARTIFACT_BYTES: usize = 64 * 1024 * 1024;
const MAX_RECORDED_EFFECTS_GROUPS: u8 = 128;
const MAX_SUPPORT_GAP_WINDOWS: usize = 1024;
const MICROS_PER_RATIO_UNIT: u32 = 1_000_000;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InfluenceSupportPolicy {
    pub version: String,
    /// Exit threshold as a fraction of the enter threshold, in millionths.
    /// Must be strictly between zero and 1_000_000.
    pub release_threshold_ratio_microunits: u32,
    /// Minimum number of release-positive windows required to accept an episode.
    pub minimum_active_windows: usize,
    /// Maximum consecutive sub-release windows bridged inside an episode.
    pub maximum_gap_windows: usize,
    /// Conservative frame extension after the final sustained window.
    /// It is never applied to a measured-empty or rejected episode.
    pub tail_extension_frames: u64,
}

impl Default for InfluenceSupportPolicy {
    fn default() -> Self {
        Self {
            version: INFLUENCE_SUPPORT_POLICY_VERSION.to_owned(),
            release_threshold_ratio_microunits: 750_000,
            minimum_active_windows: 1,
            maximum_gap_windows: 1,
            tail_extension_frames: 0,
        }
    }
}

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
    /// At least one temporally qualified support episode passed the frozen
    /// hysteresis, minimum-window and bounded-gap policy.
    EffectDetected,
    /// No raw effect episode passed the temporal support policy. This does not
    /// prove zero physical influence or absence of an audible effect.
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
    pub source_score_sha256: String,
    /// SHA-256 of the exact serialized baseline map artifact.
    pub performed_event_map_sha256: String,
    /// Parser-reconstructed receipt digest from the baseline MIDI bytes.
    pub performed_event_map_receipt_sha256: String,
    pub expected_primary_pitches_sha256: String,
    pub baseline_schedule_sha256: String,
    pub renderer_identity_sha256: String,
    pub renderer_config_sha256: String,
    pub anchor_renderer_receipt_sha256: String,
    /// One replay-verified receipt digest for every no-op render.
    pub noop_renderer_receipt_sha256: Vec<String>,
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

/// Opaque token proving that a local RendererExecutionReceipt replayed against
/// its exact RendererReceiptInputs and that the supplied AudioView exactly
/// matches the receipt's decoded PCM digest and format metadata.
///
/// Fields are private: production callers must use verify_renderer_audio_binding.
/// This remains local artifact evidence, not cryptographic attestation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRendererAudioBinding {
    receipt_sha256: String,
    audio_sha256: String,
    renderer_identity_sha256: String,
    renderer_config_sha256: String,
    source_score_sha256: String,
    input_artifact_sha256: String,
    performed_event_map_sha256: String,
    performed_event_map_artifact_sha256: String,
    expected_primary_pitches_sha256: String,
    performed_event_lineage: Vec<PerformedEventLineage>,
    /// Explicit FluidSynth effects-group count parsed from the exact, receipt-
    /// bound child argv. None means the historical/unknown topology did not
    /// explicitly record the setting; downstream routing adapters must abstain.
    fluid_synth_effects_groups: Option<u8>,
}

/// Test-only constructor for a verified FluidSynth token carrying an
/// explicitly recorded effects-group count. Production must obtain this value
/// only through the receipt-replaying verifier above.
#[cfg(test)]
pub(crate) fn test_verified_renderer_audio_binding_with_effects_groups(
    audio: &AudioView<'_>,
    receipt_sha256: String,
    renderer_identity_sha256: String,
    renderer_config_sha256: String,
    source_score_sha256: String,
    input_artifact_sha256: String,
    performed_event_map_sha256: String,
    performed_event_map_artifact_sha256: String,
    expected_primary_pitches_sha256: String,
    performed_event_lineage: Vec<PerformedEventLineage>,
    effects_groups: u8,
) -> VerifiedRendererAudioBinding {
    let mut binding = test_verified_renderer_audio_binding(
        audio,
        receipt_sha256,
        renderer_identity_sha256,
        renderer_config_sha256,
        source_score_sha256,
        input_artifact_sha256,
        performed_event_map_sha256,
        performed_event_map_artifact_sha256,
        expected_primary_pitches_sha256,
        performed_event_lineage,
    );
    binding.fluid_synth_effects_groups = Some(effects_groups);
    binding
}

/// Test-only constructor for exercising downstream coalition validators.
/// Production code must use verify_renderer_audio_binding or its typed-Score
/// wrapper; this helper is excluded from non-test builds.
#[cfg(test)]
pub(crate) fn test_verified_renderer_audio_binding(
    audio: &AudioView<'_>,
    receipt_sha256: String,
    renderer_identity_sha256: String,
    renderer_config_sha256: String,
    source_score_sha256: String,
    input_artifact_sha256: String,
    performed_event_map_sha256: String,
    performed_event_map_artifact_sha256: String,
    expected_primary_pitches_sha256: String,
    performed_event_lineage: Vec<PerformedEventLineage>,
) -> VerifiedRendererAudioBinding {
    VerifiedRendererAudioBinding {
        receipt_sha256,
        audio_sha256: audio_sha256(audio),
        renderer_identity_sha256,
        renderer_config_sha256,
        source_score_sha256,
        input_artifact_sha256,
        performed_event_map_sha256,
        performed_event_map_artifact_sha256,
        expected_primary_pitches_sha256,
        performed_event_lineage,
        fluid_synth_effects_groups: None,
    }
}

impl VerifiedRendererAudioBinding {
    pub fn receipt_sha256(&self) -> &str {
        &self.receipt_sha256
    }

    pub fn audio_sha256(&self) -> &str {
        &self.audio_sha256
    }

    pub fn renderer_identity_sha256(&self) -> &str {
        &self.renderer_identity_sha256
    }

    pub fn renderer_config_sha256(&self) -> &str {
        &self.renderer_config_sha256
    }

    pub fn source_score_sha256(&self) -> &str {
        &self.source_score_sha256
    }

    pub fn input_artifact_sha256(&self) -> &str {
        &self.input_artifact_sha256
    }

    pub fn performed_event_map_sha256(&self) -> &str {
        &self.performed_event_map_sha256
    }

    pub fn performed_event_map_artifact_sha256(&self) -> &str {
        &self.performed_event_map_artifact_sha256
    }

    pub fn expected_primary_pitches_sha256(&self) -> &str {
        &self.expected_primary_pitches_sha256
    }

    pub fn performed_event_lineage(&self) -> &[PerformedEventLineage] {
        &self.performed_event_lineage
    }

    /// Explicit FluidSynth FX group count from the verified receipt argv.
    /// None must be treated as incomplete routing, not as the default group.
    pub fn fluid_synth_effects_groups(&self) -> Option<u8> {
        self.fluid_synth_effects_groups
    }
}

/// Hash the interleaved decoded f32 samples exactly as the renderer receipt
/// module does: little-endian f32 values, without format metadata or rescaling.
fn decoded_pcm_f32_le_sha256(audio: &AudioView<'_>) -> String {
    let mut hasher = Sha256::new();
    for sample in audio.interleaved_samples {
        hasher.update(sample.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

/// Build the process-independent execution profile for FluidSynth. The exact
/// output and input paths are intentionally normalized because they are
/// temporary artifact locations, not renderer settings; the input MIDI bytes
/// and output PCM are committed independently by the receipt. All actual
/// options, SoundFont path, and the exact child environment digest remain bound.
fn renderer_execution_profile_sha256(
    receipt: &RendererExecutionReceipt,
) -> Result<(String, String), CounterfactualInfluenceError> {
    let process = receipt
        .process_observation
        .as_ref()
        .ok_or(CounterfactualInfluenceError::RendererReceiptInvalid)?;

    let identity_bytes = serde_json::to_vec(&(
        receipt.backend,
        process.executable_path.as_str(),
        process.executable_sha256_after.as_str(),
        process.soundfont_path.as_str(),
        process.soundfont_sha256_after.as_str(),
    ))
    .map_err(|_| CounterfactualInfluenceError::SerializationFailure)?;

    let mut normalized_arguments = process.arguments.clone();
    let output_flag = normalized_arguments
        .iter()
        .position(|argument| argument == "-F")
        .ok_or(CounterfactualInfluenceError::RendererReceiptInvalid)?;
    if output_flag + 1 >= normalized_arguments.len() {
        return Err(CounterfactualInfluenceError::RendererReceiptInvalid);
    }
    normalized_arguments[output_flag + 1] = "<output-wav>".to_owned();

    // In this adapter's fixed argv contract, the final argument is the exact
    // input MIDI path. Its bytes are separately receipt-bound as input_midi.
    let input_path = normalized_arguments
        .last_mut()
        .ok_or(CounterfactualInfluenceError::RendererReceiptInvalid)?;
    if input_path.starts_with('-') {
        return Err(CounterfactualInfluenceError::RendererReceiptInvalid);
    }
    *input_path = "<input-midi>".to_owned();

    let config_bytes = serde_json::to_vec(&(
        receipt.render_config_sha256.as_str(),
        normalized_arguments,
        process.environment_sha256.as_str(),
    ))
    .map_err(|_| CounterfactualInfluenceError::SerializationFailure)?;

    Ok((sha256_hex(&identity_bytes), sha256_hex(&config_bytes)))
}


fn receipt_fluidsynth_effects_groups(
    receipt: &RendererExecutionReceipt,
) -> Option<u8> {
    let process = receipt.process_observation.as_ref()?;
    let mut matches = Vec::new();
    for pair in process.arguments.windows(2) {
        if pair[0] == "-o" {
            if let Some(value) = pair[1].strip_prefix("synth.effects-groups=") {
                let groups = value.parse::<u8>().ok()?;
                if !(1..=MAX_RECORDED_EFFECTS_GROUPS).contains(&groups) {
                    return None;
                }
                matches.push(groups);
            }
        }
    }
    if matches.len() == 1 {
        matches.first().copied()
    } else {
        None
    }
}

/// Verify the receipt against its exact serialized Score/config/MIDI-or-schedule,
/// executable/assets, output WAV and process observation, then bind the decoded
/// PCM view to the receipt. The caller must not mint this token from self-asserted
/// digests or an unverified receipt.
pub(crate) fn verify_renderer_audio_binding(
    receipt: &RendererExecutionReceipt,
    inputs: RendererReceiptInputs<'_>,
    performed_event_map_bytes: &[u8],
    expected_score_event_count: usize,
    expected_primary_midi_by_score_index: &[u8],
    excluded_source_event_index: Option<usize>,
    audio: &AudioView<'_>,
) -> Result<VerifiedRendererAudioBinding, CounterfactualInfluenceError> {
    let excluded = excluded_source_event_index
        .map(|index| vec![index])
        .unwrap_or_default();
    verify_renderer_audio_binding_excluding_set(
        receipt,
        inputs,
        performed_event_map_bytes,
        expected_score_event_count,
        expected_primary_midi_by_score_index,
        &excluded,
        audio,
    )
}

/// Multi-event variant used by factorial interaction experiments. Exclusions
/// are explicit source-score indices and do not weaken coverage for any other
/// event. The same receipt/PCM/MIDI/map replay checks apply.
pub(crate) fn verify_renderer_audio_binding_excluding_set(
    receipt: &RendererExecutionReceipt,
    inputs: RendererReceiptInputs<'_>,
    performed_event_map_bytes: &[u8],
    expected_score_event_count: usize,
    expected_primary_midi_by_score_index: &[u8],
    excluded_source_event_indices: &[usize],
    audio: &AudioView<'_>,
) -> Result<VerifiedRendererAudioBinding, CounterfactualInfluenceError> {
    let frames = validate_audio(audio)?;
    if receipt.backend != crate::renderer_execution_receipt::RendererBackend::FluidSynth {
        // NativeMuse needs its own exact performed-schedule parser before it
        // can mint event-level provenance tokens.
        return Err(CounterfactualInfluenceError::RendererReceiptInvalid);
    }
    let midi_bytes = inputs
        .input_midi_bytes
        .ok_or(CounterfactualInfluenceError::RendererReceiptInvalid)?;
    let parsed_map = PerformedEventMap::from_midi_bytes(
        midi_bytes,
        &receipt.score_sha256,
        audio.sample_rate_hz,
        frames as u64,
    )
    .map_err(|_| CounterfactualInfluenceError::RendererReceiptInvalid)?;

    if expected_primary_midi_by_score_index.len() != expected_score_event_count {
        return Err(CounterfactualInfluenceError::RendererReceiptInvalid);
    }
    if excluded_source_event_indices.is_empty() {
        parsed_map.verify_for_score_and_pitches(
            &receipt.score_sha256,
            &parsed_map.performed_midi_sha256,
            audio.sample_rate_hz,
            frames as u64,
            expected_score_event_count,
            expected_primary_midi_by_score_index,
        )
    } else {
        parsed_map.verify_for_score_excluding_many_and_pitches(
            &receipt.score_sha256,
            &parsed_map.performed_midi_sha256,
            audio.sample_rate_hz,
            frames as u64,
            expected_score_event_count,
            excluded_source_event_indices,
            expected_primary_midi_by_score_index,
        )
    }
    .map_err(|_| CounterfactualInfluenceError::RendererReceiptInvalid)?;

    let supplied_map: PerformedEventMap = serde_json::from_slice(performed_event_map_bytes)
        .map_err(|_| CounterfactualInfluenceError::RendererReceiptInvalid)?;
    if supplied_map != parsed_map {
        return Err(CounterfactualInfluenceError::RendererReceiptInvalid);
    }
    if let Some(receipt_map_sha256) = receipt.performed_event_map_sha256.as_deref()
        && receipt_map_sha256 != parsed_map.receipt_sha256
    {
        return Err(CounterfactualInfluenceError::RendererReceiptInvalid);
    }
    if excluded_source_event_indices.is_empty()
        && receipt.performed_event_map_sha256.as_deref()
            != Some(parsed_map.receipt_sha256.as_str())
    {
        return Err(CounterfactualInfluenceError::RendererReceiptInvalid);
    }

    receipt
        .verify_artifacts(inputs)
        .map_err(|_| CounterfactualInfluenceError::RendererReceiptInvalid)?;
    if receipt.output.sample_rate_hz != audio.sample_rate_hz
        || receipt.output.channels as usize != audio.channels
        || receipt.output.frame_count != frames as u64
        || receipt.output.decoded_pcm_f32_le_sha256 != decoded_pcm_f32_le_sha256(audio)
    {
        return Err(CounterfactualInfluenceError::RendererAudioMismatch);
    }

    let (renderer_identity_sha256, renderer_config_sha256) =
        renderer_execution_profile_sha256(receipt)?;
    let input_artifact_sha256 = receipt
        .input_midi_sha256
        .clone()
        .ok_or(CounterfactualInfluenceError::RendererReceiptInvalid)?;

    Ok(VerifiedRendererAudioBinding {
        receipt_sha256: receipt.receipt_sha256.clone(),
        audio_sha256: audio_sha256(audio),
        renderer_identity_sha256,
        renderer_config_sha256,
        source_score_sha256: receipt.score_sha256.clone(),
        input_artifact_sha256,
        performed_event_map_sha256: parsed_map.receipt_sha256,
        performed_event_map_artifact_sha256: sha256_hex(performed_event_map_bytes),
        expected_primary_pitches_sha256: sha256_hex(expected_primary_midi_by_score_index),
        performed_event_lineage: parsed_map.events,
        fluid_synth_effects_groups: receipt_fluidsynth_effects_groups(receipt),
    })
}

/// Verify renderer output against the exact typed source Score. This is
/// the production-facing entry point in theory-enabled builds: serialized
/// Score bytes must match the receipt inputs byte-for-byte, and expected MIDI
/// pitches are derived directly from Score::events() rather than caller input.
#[cfg(feature = "theory")]
pub fn verify_renderer_audio_binding_for_score(
    receipt: &RendererExecutionReceipt,
    inputs: RendererReceiptInputs<'_>,
    performed_event_map_bytes: &[u8],
    score: &symthaea_music_theory::Score,
    excluded_source_event_index: Option<usize>,
    audio: &AudioView<'_>,
) -> Result<VerifiedRendererAudioBinding, CounterfactualInfluenceError> {
    let serialized_score = serde_json::to_vec(score)
        .map_err(|_| CounterfactualInfluenceError::SerializationFailure)?;
    if inputs.score_bytes != serialized_score.as_slice()
        || receipt.score_sha256 != sha256_hex(&serialized_score)
    {
        return Err(CounterfactualInfluenceError::RendererReceiptInvalid);
    }
    let expected_primary_midi_by_score_index: Vec<u8> = score
        .events()
        .iter()
        .map(|event| event.pitch.midi())
        .collect();
    verify_renderer_audio_binding(
        receipt,
        inputs,
        performed_event_map_bytes,
        expected_primary_midi_by_score_index.len(),
        &expected_primary_midi_by_score_index,
        excluded_source_event_index,
        audio,
    )
}

/// Typed-Score multi-event variant for factorial interaction experiments.
/// The caller cannot supply a different pitch inventory: it is derived from
/// the exact serialized Score and checked against the renderer receipt.
#[cfg(feature = "theory")]
pub fn verify_renderer_audio_binding_for_score_excluding_set(
    receipt: &RendererExecutionReceipt,
    inputs: RendererReceiptInputs<'_>,
    performed_event_map_bytes: &[u8],
    score: &symthaea_music_theory::Score,
    excluded_source_event_indices: &[usize],
    audio: &AudioView<'_>,
) -> Result<VerifiedRendererAudioBinding, CounterfactualInfluenceError> {
    let serialized_score = serde_json::to_vec(score)
        .map_err(|_| CounterfactualInfluenceError::SerializationFailure)?;
    if inputs.score_bytes != serialized_score.as_slice()
        || receipt.score_sha256 != sha256_hex(&serialized_score)
    {
        return Err(CounterfactualInfluenceError::RendererReceiptInvalid);
    }
    let expected_primary_midi_by_score_index: Vec<u8> = score
        .events()
        .iter()
        .map(|event| event.pitch.midi())
        .collect();
    verify_renderer_audio_binding_excluding_set(
        receipt,
        inputs,
        performed_event_map_bytes,
        expected_primary_midi_by_score_index.len(),
        &expected_primary_midi_by_score_index,
        excluded_source_event_indices,
        audio,
    )
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
    pub release_rms_threshold: f32,
    pub release_spectral_threshold: f32,
    /// Raw entry-threshold decision, before temporal qualification.
    pub effect_detected: bool,
    /// Lower hysteresis threshold decision. Considered only after an episode
    /// has already entered on an effect_detected window.
    pub release_detected: bool,
    /// Included in an accepted episode's qualified support mask.
    pub in_qualified_support: bool,
    /// A below-release gap bridged between two release-positive windows.
    pub bridged_gap: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CounterfactualInfluenceReport {
    pub version: String,
    pub source_event_id: String,
    pub intervention: EventIntervention,
    pub source_score_sha256: String,
    /// Hash of the exact serialized baseline map artifact.
    pub performed_event_map_sha256: String,
    pub performed_event_map_receipt_sha256: String,
    pub counterfactual_event_map_sha256: String,
    pub counterfactual_event_map_receipt_sha256: String,
    pub expected_primary_pitches_sha256: String,
    pub baseline_schedule_sha256: String,
    pub counterfactual_schedule_sha256: String,
    pub renderer_identity_sha256: String,
    pub renderer_config_sha256: String,
    pub baseline_audio_sha256: String,
    pub counterfactual_audio_sha256: String,
    pub baseline_renderer_receipt_sha256: String,
    pub counterfactual_renderer_receipt_sha256: String,
    pub repeatability_calibration_sha256: String,
    pub sample_rate_hz: u32,
    pub channels: usize,
    pub total_frames: usize,
    pub analysis: InfluenceAnalysisConfig,
    pub total_channel_windows: usize,
    /// Number of channel/windows crossing the raw enter threshold.
    pub affected_channel_windows: usize,
    /// Versioned policy used to qualify temporal support episodes.
    pub support_policy: InfluenceSupportPolicy,
    /// Complete ordered per-window/per-channel trace, including raw and release
    /// thresholds and the final hysteresis/bridge classification.
    pub window_measurements: Vec<InfluenceWindowMeasurement>,
    /// Historical field retained with its raw measured meaning for audit.
    pub measured_influence_intervals: Vec<FrameInterval>,
    /// Hysteresis/gap/minimum-duration qualified support plus bounded tail hold.
    /// This is the only interval set downstream planners should consume.
    pub qualified_support_intervals: Vec<FrameInterval>,
    pub qualified_channel_windows: usize,
    pub bridged_channel_windows: usize,
    pub tail_extension_frames_applied: u64,
    pub disposition: InfluenceDisposition,
    pub receipt_sha256: String,
}

/// Exact artifacts and audio supplied to the paired measurement. The caller
/// must create the counterfactual schedule by changing only the declared event;
/// this module binds the bytes but does not independently parse the schedule.
pub struct CounterfactualInfluenceInput<'view, 'audio> {
    pub source_event_id: &'view str,
    pub intervention: EventIntervention,
    pub source_score_bytes: &'view [u8],
    pub performed_event_map_bytes: &'view [u8],
    pub counterfactual_performed_event_map_bytes: &'view [u8],
    pub expected_score_event_count: usize,
    /// Expected MIDI pitch indexed by canonical source Score event index.
    pub expected_primary_midi_by_score_index: &'view [u8],
    pub baseline_schedule_bytes: &'view [u8],
    pub counterfactual_schedule_bytes: &'view [u8],
    pub baseline_renderer_evidence: &'view VerifiedRendererAudioBinding,
    pub counterfactual_renderer_evidence: &'view VerifiedRendererAudioBinding,
    pub baseline_audio: &'view AudioView<'audio>,
    pub counterfactual_audio: &'view AudioView<'audio>,
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
    RendererReceiptInvalid,
    RendererAudioMismatch,
    RendererIdentityMismatch,
    InvalidEventId,
    InvalidIntervention,
    EmptyArtifact(&'static str),
    ArtifactTooLarge(&'static str),
    IdenticalSchedules,
    SerializationFailure,
    InvalidInfluenceReceipt,
    InfluenceReplayMismatch,
    WindowIndexMismatch,
    InvalidSupportPolicy(&'static str),
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
            Self::RendererReceiptInvalid => {
                write!(f, "renderer execution receipt does not replay against the supplied artifacts")
            }
            Self::RendererAudioMismatch => {
                write!(f, "decoded renderer PCM does not match the supplied audio view")
            }
            Self::RendererIdentityMismatch => {
                write!(f, "renderer identity or configuration differs across paired renders")
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
            Self::InvalidSupportPolicy(reason) => write!(f, "invalid temporal support policy: {reason}"),
        }
    }
}

impl std::error::Error for CounterfactualInfluenceError {}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn parse_score_event_index(value: &str) -> Option<usize> {
    let suffix = value.strip_prefix("score_event_index:")?;
    if suffix.is_empty()
        || (suffix.len() > 1 && suffix.starts_with('0'))
        || !suffix.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    suffix.parse().ok()
}

pub(crate) fn audio_sha256(audio: &AudioView<'_>) -> String {
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

fn canonical_digest<T: Serialize>(
    value: &T,
    domain: &[u8],
) -> Result<String, CounterfactualInfluenceError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|_| CounterfactualInfluenceError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn calibration_digest_valid(
    calibration: &RepeatabilityCalibration,
) -> Result<bool, CounterfactualInfluenceError> {
    let mut unsealed = calibration.clone();
    let expected = std::mem::take(&mut unsealed.receipt_sha256);
    if expected.len() != 64 {
        return Ok(false);
    }
    Ok(canonical_digest(&unsealed, CALIBRATION_DOMAIN)? == expected)
}

fn influence_digest_valid(
    report: &CounterfactualInfluenceReport,
) -> Result<bool, CounterfactualInfluenceError> {
    let mut unsealed = report.clone();
    let expected = std::mem::take(&mut unsealed.receipt_sha256);
    if expected.len() != 64 {
        return Ok(false);
    }
    Ok(canonical_digest(&unsealed, INFLUENCE_DOMAIN)? == expected)
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

fn validate_support_policy(
    policy: &InfluenceSupportPolicy,
    total_frames: usize,
    window_count: usize,
) -> Result<(), CounterfactualInfluenceError> {
    if policy.version != INFLUENCE_SUPPORT_POLICY_VERSION {
        return Err(CounterfactualInfluenceError::InvalidSupportPolicy("unsupported policy version"));
    }
    if policy.release_threshold_ratio_microunits == 0
        || policy.release_threshold_ratio_microunits >= MICROS_PER_RATIO_UNIT
    {
        return Err(CounterfactualInfluenceError::InvalidSupportPolicy(
            "release threshold ratio must be strictly between zero and one",
        ));
    }
    if policy.minimum_active_windows == 0 || policy.minimum_active_windows > window_count {
        return Err(CounterfactualInfluenceError::InvalidSupportPolicy(
            "minimum_active_windows must be in 1..=window_count",
        ));
    }
    if policy.maximum_gap_windows > MAX_SUPPORT_GAP_WINDOWS {
        return Err(CounterfactualInfluenceError::InvalidSupportPolicy(
            "maximum_gap_windows exceeds the safety bound",
        ));
    }
    if policy.tail_extension_frames > total_frames as u64 {
        return Err(CounterfactualInfluenceError::InvalidSupportPolicy(
            "tail_extension_frames cannot exceed the rendered output horizon",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SupportQualificationCounts {
    qualified_channel_windows: usize,
    bridged_channel_windows: usize,
    tail_extension_frames_applied: u64,
}

/// Apply a temporal Schmitt-trigger-like support policy independently per
/// channel. A new episode starts only on an enter-threshold crossing. Once
/// active, lower release thresholds maintain it; a bounded number of
/// consecutive below-release windows may be bridged. Trailing gaps never extend
/// a measured episode. The configured tail extension is applied only after a
/// qualifying episode and is capped by the exact output-frame horizon.
fn qualify_support_windows(
    measurements: &mut [InfluenceWindowMeasurement],
    channels: usize,
    total_frames: usize,
    policy: &InfluenceSupportPolicy,
) -> Result<(Vec<FrameInterval>, SupportQualificationCounts), CounterfactualInfluenceError> {
    if channels == 0 || measurements.is_empty() || measurements.len() % channels != 0 {
        return Err(CounterfactualInfluenceError::WindowIndexMismatch);
    }
    let window_count = measurements.len() / channels;
    validate_support_policy(policy, total_frames, window_count)?;

    let mut intervals = Vec::new();
    let mut qualified_channel_windows = 0usize;
    let mut bridged_channel_windows = 0usize;
    let mut qualified_intervals = Vec::new();

    for channel in 0..channels {
        let mut episode_start: Option<usize> = None;
        let mut last_sustained_end = 0usize;
        let mut sustained_window_count = 0usize;
        let mut episode_indices = Vec::<usize>::new();
        let mut pending_gap_indices = Vec::<usize>::new();

        for index in (channel..measurements.len()).step_by(channels) {
            let measurement = &measurements[index];
            if measurement.channel != channel
                || measurement.start_frame >= measurement.end_frame
                || measurement.end_frame > total_frames
            {
                return Err(CounterfactualInfluenceError::WindowIndexMismatch);
            }

            if episode_start.is_none() {
                if measurement.effect_detected {
                    episode_start = Some(measurement.start_frame);
                    last_sustained_end = measurement.end_frame;
                    sustained_window_count = 1;
                    episode_indices.push(index);
                }
                continue;
            }

            if measurement.release_detected {
                if !pending_gap_indices.is_empty() {
                    episode_indices.extend(pending_gap_indices.drain(..));
                }
                episode_indices.push(index);
                sustained_window_count += 1;
                last_sustained_end = measurement.end_frame;
            } else {
                pending_gap_indices.push(index);
                if pending_gap_indices.len() > policy.maximum_gap_windows {
                    if sustained_window_count >= policy.minimum_active_windows {
                        let start = episode_start.expect("active support episode has a start");
                        for member_index in &episode_indices {
                            measurements[*member_index].in_qualified_support = true;
                        }
                        for gap_index in episode_indices.iter().copied() {
                            if measurements[gap_index].release_detected == false
                                && !measurements[gap_index].effect_detected
                            {
                                measurements[gap_index].bridged_gap = true;
                                bridged_channel_windows += 1;
                            }
                        }
                        qualified_channel_windows += episode_indices.len();
                        qualified_intervals.push(FrameInterval {
                            start_frame: start,
                            end_frame: last_sustained_end,
                        });
                    }
                    episode_start = None;
                    last_sustained_end = 0;
                    sustained_window_count = 0;
                    episode_indices.clear();
                    pending_gap_indices.clear();
                }
            }
        }

        if let Some(start) = episode_start
            && sustained_window_count >= policy.minimum_active_windows
        {
            for member_index in &episode_indices {
                measurements[*member_index].in_qualified_support = true;
            }
            for gap_index in episode_indices.iter().copied() {
                if !measurements[gap_index].release_detected
                    && !measurements[gap_index].effect_detected
                {
                    measurements[gap_index].bridged_gap = true;
                    bridged_channel_windows += 1;
                }
            }
            qualified_channel_windows += episode_indices.len();
            qualified_intervals.push(FrameInterval {
                start_frame: start,
                end_frame: last_sustained_end,
            });
        }
    }

    // Keep raw interval diagnostics separate from temporally qualified
    // episode windows. Tail extension is applied after channel union so shared
    // stereo/channel frames are counted only once in the report.
    let raw_measured = merge_intervals(intervals);
    let qualified_before_tail = merge_intervals(qualified_intervals);
    let baseline_support_frames: u64 = qualified_before_tail
        .iter()
        .map(|interval| (interval.end_frame - interval.start_frame) as u64)
        .sum();
    let mut tail_extended = Vec::with_capacity(qualified_before_tail.len());
    for interval in &qualified_before_tail {
        let end_frame = (interval.end_frame as u64)
            .saturating_add(policy.tail_extension_frames)
            .min(total_frames as u64) as usize;
        tail_extended.push(FrameInterval {
            start_frame: interval.start_frame,
            end_frame,
        });
    }
    let qualified_after_tail = merge_intervals(tail_extended);
    let qualified_support_frames: u64 = qualified_after_tail
        .iter()
        .map(|interval| (interval.end_frame - interval.start_frame) as u64)
        .sum();
    let tail_extension_frames_applied =
        qualified_support_frames.saturating_sub(baseline_support_frames);

    Ok((
        qualified_after_tail,
        SupportQualificationCounts {
            qualified_channel_windows,
            bridged_channel_windows,
            tail_extension_frames_applied,
        },
    ))
}

fn seal_calibration(
    mut calibration: RepeatabilityCalibration,
) -> Result<RepeatabilityCalibration, CounterfactualInfluenceError> {
    calibration.receipt_sha256.clear();
    let digest = canonical_digest(&calibration, CALIBRATION_DOMAIN)?;
    calibration.receipt_sha256 = digest;
    Ok(calibration)
}

/// Build an empirical no-op measurement floor from the exact same-render
/// anchor and at least three independent repeats with identical format and
/// duration. Each cell uses the largest observed no-op delta; this is a
/// conservative fixture-level floor, not a population quantile or guarantee.
pub fn calibrate_repeatability(
    anchor: &AudioView<'_>,
    noop_renders: &[AudioView<'_>],
    source_score_bytes: &[u8],
    performed_event_map_bytes: &[u8],
    baseline_schedule_bytes: &[u8],
    anchor_renderer_evidence: &VerifiedRendererAudioBinding,
    noop_renderer_evidence: &[VerifiedRendererAudioBinding],
    analysis: &InfluenceAnalysisConfig,
) -> Result<RepeatabilityCalibration, CounterfactualInfluenceError> {
    validate_config(analysis)?;
    for (name, bytes) in [
        ("source_score", source_score_bytes),
        ("performed_event_map", performed_event_map_bytes),
        ("baseline_schedule", baseline_schedule_bytes),
    ] {
        if bytes.is_empty() {
            return Err(CounterfactualInfluenceError::EmptyArtifact(name));
        }
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err(CounterfactualInfluenceError::ArtifactTooLarge(name));
        }
    }
    if noop_renderer_evidence.len() != noop_renders.len() {
        return Err(CounterfactualInfluenceError::CalibrationReplayMismatch);
    }
    let unique_noop_receipts: std::collections::BTreeSet<&str> = noop_renderer_evidence
        .iter()
        .map(|evidence| evidence.receipt_sha256.as_str())
        .collect();
    if unique_noop_receipts.len() != noop_renderer_evidence.len() {
        return Err(CounterfactualInfluenceError::CalibrationReplayMismatch);
    }
    if anchor_renderer_evidence.audio_sha256 != audio_sha256(anchor) {
        return Err(CounterfactualInfluenceError::RendererAudioMismatch);
    }
    let score_sha256 = sha256_hex(source_score_bytes);
    let event_map_sha256 = sha256_hex(performed_event_map_bytes);
    let baseline_schedule_sha256 = sha256_hex(baseline_schedule_bytes);
    if anchor_renderer_evidence.source_score_sha256 != score_sha256
        || anchor_renderer_evidence.input_artifact_sha256 != baseline_schedule_sha256
        || anchor_renderer_evidence.performed_event_map_artifact_sha256
            != event_map_sha256
    {
        return Err(CounterfactualInfluenceError::CalibrationConfigMismatch);
    }
    for (noop, evidence) in noop_renders.iter().zip(noop_renderer_evidence) {
        if evidence.audio_sha256 != audio_sha256(noop) {
            return Err(CounterfactualInfluenceError::RendererAudioMismatch);
        }
        if evidence.renderer_identity_sha256 != anchor_renderer_evidence.renderer_identity_sha256
            || evidence.renderer_config_sha256 != anchor_renderer_evidence.renderer_config_sha256
        {
            return Err(CounterfactualInfluenceError::RendererIdentityMismatch);
        }
        if evidence.source_score_sha256 != score_sha256
            || evidence.input_artifact_sha256 != baseline_schedule_sha256
            || evidence.performed_event_map_artifact_sha256 != event_map_sha256
            || evidence.performed_event_map_sha256
                != anchor_renderer_evidence.performed_event_map_sha256
            || evidence.expected_primary_pitches_sha256
                != anchor_renderer_evidence.expected_primary_pitches_sha256
        {
            return Err(CounterfactualInfluenceError::CalibrationConfigMismatch);
        }
    }
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
        source_score_sha256: sha256_hex(source_score_bytes),
        performed_event_map_sha256: event_map_sha256,
        performed_event_map_receipt_sha256: anchor_renderer_evidence
            .performed_event_map_sha256
            .clone(),
        expected_primary_pitches_sha256: anchor_renderer_evidence
            .expected_primary_pitches_sha256
            .clone(),
        baseline_schedule_sha256: sha256_hex(baseline_schedule_bytes),
        renderer_identity_sha256: anchor_renderer_evidence.renderer_identity_sha256.clone(),
        renderer_config_sha256: anchor_renderer_evidence.renderer_config_sha256.clone(),
        anchor_renderer_receipt_sha256: anchor_renderer_evidence.receipt_sha256.clone(),
        noop_renderer_receipt_sha256: noop_renderer_evidence
            .iter()
            .map(|evidence| evidence.receipt_sha256.clone())
            .collect(),
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
    source_score_bytes: &[u8],
    performed_event_map_bytes: &[u8],
    baseline_schedule_bytes: &[u8],
    anchor_renderer_evidence: &VerifiedRendererAudioBinding,
    noop_renderer_evidence: &[VerifiedRendererAudioBinding],
) -> Result<VerifiedRepeatabilityCalibration, CounterfactualInfluenceError> {
    if !calibration_digest_valid(calibration)? {
        return Err(CounterfactualInfluenceError::InvalidCalibrationReceipt);
    }
    let expected = calibrate_repeatability(
        anchor,
        noop_renders,
        source_score_bytes,
        performed_event_map_bytes,
        baseline_schedule_bytes,
        anchor_renderer_evidence,
        noop_renderer_evidence,
        &calibration.analysis,
    )?;
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
        // Other typed variants remain reserved until their operation-specific
        // MIDI obligations are implemented and independently tested.
        _ => Err(CounterfactualInfluenceError::InvalidIntervention),
    }
}

fn validate_input(
    input: &CounterfactualInfluenceInput<'_, '_>,
    calibration: &RepeatabilityCalibration,
) -> Result<usize, CounterfactualInfluenceError> {
    validate_intervention(&input.intervention)?;
    let excluded_index = parse_score_event_index(input.source_event_id)
        .ok_or(CounterfactualInfluenceError::InvalidEventId)?;
    if input.expected_score_event_count == 0
        || input.expected_score_event_count > 100_000
        || input.expected_primary_midi_by_score_index.len() != input.expected_score_event_count
        || excluded_index >= input.expected_score_event_count
    {
        return Err(CounterfactualInfluenceError::InvalidEventId);
    }
    for (name, bytes) in [
        ("source_score", input.source_score_bytes),
        ("performed_event_map", input.performed_event_map_bytes),
        ("counterfactual_performed_event_map", input.counterfactual_performed_event_map_bytes),
        ("baseline_schedule", input.baseline_schedule_bytes),
        ("counterfactual_schedule", input.counterfactual_schedule_bytes),
        ("baseline_renderer_receipt", input.baseline_renderer_evidence.receipt_sha256.as_bytes()),
        ("counterfactual_renderer_receipt", input.counterfactual_renderer_evidence.receipt_sha256.as_bytes()),
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
    if sha256_hex(input.source_score_bytes) != calibration.source_score_sha256
        || sha256_hex(input.performed_event_map_bytes) != calibration.performed_event_map_sha256
        || sha256_hex(input.baseline_schedule_bytes) != calibration.baseline_schedule_sha256
    {
        return Err(CounterfactualInfluenceError::CalibrationConfigMismatch);
    }
    if input.baseline_renderer_evidence.audio_sha256 != audio_sha256(input.baseline_audio)
        || input.counterfactual_renderer_evidence.audio_sha256
            != audio_sha256(input.counterfactual_audio)
    {
        return Err(CounterfactualInfluenceError::RendererAudioMismatch);
    }
    let expected_pitch_sha256 = sha256_hex(input.expected_primary_midi_by_score_index);
    if input.baseline_renderer_evidence.source_score_sha256
        != sha256_hex(input.source_score_bytes)
        || input.counterfactual_renderer_evidence.source_score_sha256
            != sha256_hex(input.source_score_bytes)
        || input.baseline_renderer_evidence.expected_primary_pitches_sha256
            != expected_pitch_sha256
        || input.counterfactual_renderer_evidence.expected_primary_pitches_sha256
            != expected_pitch_sha256
        || input.baseline_renderer_evidence.input_artifact_sha256
            != sha256_hex(input.baseline_schedule_bytes)
        || input.counterfactual_renderer_evidence.input_artifact_sha256
            != sha256_hex(input.counterfactual_schedule_bytes)
        || input.baseline_renderer_evidence.performed_event_map_artifact_sha256
            != sha256_hex(input.performed_event_map_bytes)
        || input.counterfactual_renderer_evidence.performed_event_map_artifact_sha256
            != sha256_hex(input.counterfactual_performed_event_map_bytes)
    {
        return Err(CounterfactualInfluenceError::CalibrationConfigMismatch);
    }

    let baseline_ids: std::collections::BTreeSet<usize> = input
        .baseline_renderer_evidence
        .performed_event_lineage
        .iter()
        .filter_map(|row| parse_score_event_index(&row.source_event_id))
        .collect();
    let counterfactual_ids: std::collections::BTreeSet<usize> = input
        .counterfactual_renderer_evidence
        .performed_event_lineage
        .iter()
        .filter_map(|row| parse_score_event_index(&row.source_event_id))
        .collect();
    let expected_baseline_ids: std::collections::BTreeSet<usize> =
        (0..input.expected_score_event_count).collect();
    let expected_counterfactual_ids: std::collections::BTreeSet<usize> =
        (0..input.expected_score_event_count)
            .filter(|index| *index != excluded_index)
            .collect();
    if baseline_ids != expected_baseline_ids
        || counterfactual_ids != expected_counterfactual_ids
    {
        return Err(CounterfactualInfluenceError::InvalidIntervention);
    }
    if input.baseline_renderer_evidence.performed_event_map_sha256
        != calibration.performed_event_map_receipt_sha256
    {
        return Err(CounterfactualInfluenceError::CalibrationConfigMismatch);
    }

    let baseline_target_rows: Vec<&PerformedEventLineage> = input
        .baseline_renderer_evidence
        .performed_event_lineage
        .iter()
        .filter(|row| row.source_event_id == input.source_event_id)
        .collect();
    let counterfactual_target_present = input
        .counterfactual_renderer_evidence
        .performed_event_lineage
        .iter()
        .any(|row| row.source_event_id == input.source_event_id);
    if baseline_target_rows.is_empty() || counterfactual_target_present {
        return Err(CounterfactualInfluenceError::InvalidIntervention);
    }

    // A true Remove operation must preserve every non-target descendant exactly.
    // MIDI ordinal is omitted because removing a prior note shifts later per-track ordinals.
    let baseline_protected: Vec<&PerformedEventLineage> = input
        .baseline_renderer_evidence
        .performed_event_lineage
        .iter()
        .filter(|row| row.source_event_id != input.source_event_id)
        .collect();
    let counterfactual_protected =
        &input.counterfactual_renderer_evidence.performed_event_lineage;
    if baseline_protected.len() != counterfactual_protected.len()
        || baseline_protected.iter().any(|baseline| {
            !counterfactual_protected.iter().any(|candidate| {
                candidate.source_event_id == baseline.source_event_id
                    && candidate.descendant_ordinal == baseline.descendant_ordinal
                    && candidate.midi_track_index == baseline.midi_track_index
                    && candidate.midi_channel == baseline.midi_channel
                    && candidate.midi_note == baseline.midi_note
                    && candidate.onset_micros == baseline.onset_micros
                    && candidate.end_micros == baseline.end_micros
                    && candidate.onset_frame == baseline.onset_frame
                    && candidate.end_frame == baseline.end_frame
            })
        })
    {
        return Err(CounterfactualInfluenceError::InvalidIntervention);
    }

    if input.baseline_renderer_evidence.receipt_sha256
        != calibration.anchor_renderer_receipt_sha256
        || input.baseline_renderer_evidence.renderer_identity_sha256
            != calibration.renderer_identity_sha256
        || input.baseline_renderer_evidence.renderer_config_sha256
            != calibration.renderer_config_sha256
        || input.counterfactual_renderer_evidence.renderer_identity_sha256
            != calibration.renderer_identity_sha256
        || input.counterfactual_renderer_evidence.renderer_config_sha256
            != calibration.renderer_config_sha256
    {
        return Err(CounterfactualInfluenceError::RendererIdentityMismatch);
    }
    Ok(frames)
}


fn seal_report(
    mut report: CounterfactualInfluenceReport,
) -> Result<CounterfactualInfluenceReport, CounterfactualInfluenceError> {
    report.receipt_sha256.clear();
    let digest = canonical_digest(&report, INFLUENCE_DOMAIN)?;
    report.receipt_sha256 = digest;
    Ok(report)
}

/// Measure the paired output difference using an already replay-verified
/// no-op calibration. The result is an empirical influence estimate for the
/// declared intervention and renderer configuration, not exclusive ownership
/// of affected audio samples.
pub fn measure_counterfactual_influence(
    input: &CounterfactualInfluenceInput<'_, '_>,
    verified_calibration: &VerifiedRepeatabilityCalibration,
) -> Result<CounterfactualInfluenceReport, CounterfactualInfluenceError> {
    measure_counterfactual_influence_with_policy(
        input,
        verified_calibration,
        &InfluenceSupportPolicy::default(),
    )
}

/// Measure paired output deltas and qualify their temporal support using an
/// explicit, replay-bound policy. The raw threshold intervals remain available
/// in measured_influence_intervals; the EVAL-025 planner adapter must consume
/// qualified_support_intervals.
pub fn measure_counterfactual_influence_with_policy(
    input: &CounterfactualInfluenceInput<'_, '_>,
    verified_calibration: &VerifiedRepeatabilityCalibration,
    support_policy: &InfluenceSupportPolicy,
) -> Result<CounterfactualInfluenceReport, CounterfactualInfluenceError> {
    let calibration = &verified_calibration.calibration;
    validate_config(&calibration.analysis)?;
    validate_input(input, calibration)?;
    let windows = frame_windows(calibration.total_frames, &calibration.analysis)?;
    validate_support_policy(support_policy, calibration.total_frames, windows.len())?;
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
    let mut raw_affected = Vec::new();
    let release_ratio =
        support_policy.release_threshold_ratio_microunits as f32 / MICROS_PER_RATIO_UNIT as f32;
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
        let release_rms_threshold = rms_threshold * release_ratio;
        let release_spectral_threshold = spectral_threshold * release_ratio;
        let effect_detected = metric.rms_delta > rms_threshold
            || metric.spectral_magnitude_distance > spectral_threshold;
        let release_detected = metric.rms_delta > release_rms_threshold
            || metric.spectral_magnitude_distance > release_spectral_threshold;
        if effect_detected {
            raw_affected.push(FrameInterval {
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
            release_rms_threshold,
            release_spectral_threshold,
            effect_detected,
            release_detected,
            in_qualified_support: false,
            bridged_gap: false,
        });
    }

    let affected_channel_windows = measurements
        .iter()
        .filter(|measurement| measurement.effect_detected)
        .count();
    let (qualified_support_intervals, qualification_counts) = qualify_support_windows(
        &mut measurements,
        calibration.channels,
        calibration.total_frames,
        support_policy,
    )?;
    let disposition = if qualified_support_intervals.is_empty() {
        InfluenceDisposition::NoEffectAboveNoiseFloor
    } else {
        InfluenceDisposition::EffectDetected
    };
    let measured_influence_intervals = merge_intervals(raw_affected);

    seal_report(CounterfactualInfluenceReport {
        version: COUNTERFACTUAL_INFLUENCE_VERSION.to_owned(),
        source_event_id: input.source_event_id.to_owned(),
        intervention: input.intervention.clone(),
        source_score_sha256: sha256_hex(input.source_score_bytes),
        performed_event_map_sha256: sha256_hex(input.performed_event_map_bytes),
        performed_event_map_receipt_sha256: calibration
            .performed_event_map_receipt_sha256
            .clone(),
        counterfactual_event_map_sha256: sha256_hex(
            input.counterfactual_performed_event_map_bytes,
        ),
        counterfactual_event_map_receipt_sha256: input
            .counterfactual_renderer_evidence
            .performed_event_map_sha256
            .clone(),
        expected_primary_pitches_sha256: calibration
            .expected_primary_pitches_sha256
            .clone(),
        baseline_schedule_sha256: sha256_hex(input.baseline_schedule_bytes),
        counterfactual_schedule_sha256: sha256_hex(input.counterfactual_schedule_bytes),
        renderer_identity_sha256: input.baseline_renderer_evidence.renderer_identity_sha256.clone(),
        renderer_config_sha256: input.baseline_renderer_evidence.renderer_config_sha256.clone(),
        baseline_audio_sha256: audio_sha256(input.baseline_audio),
        counterfactual_audio_sha256: audio_sha256(input.counterfactual_audio),
        baseline_renderer_receipt_sha256: input.baseline_renderer_evidence.receipt_sha256.clone(),
        counterfactual_renderer_receipt_sha256: input.counterfactual_renderer_evidence.receipt_sha256.clone(),
        repeatability_calibration_sha256: calibration.receipt_sha256.clone(),
        sample_rate_hz: calibration.sample_rate_hz,
        channels: calibration.channels,
        total_frames: calibration.total_frames,
        analysis: calibration.analysis.clone(),
        total_channel_windows: measurements.len(),
        affected_channel_windows,
        support_policy: support_policy.clone(),
        window_measurements: measurements,
        measured_influence_intervals,
        qualified_support_intervals,
        qualified_channel_windows: qualification_counts.qualified_channel_windows,
        bridged_channel_windows: qualification_counts.bridged_channel_windows,
        tail_extension_frames_applied: qualification_counts.tail_extension_frames_applied,
        disposition,
        receipt_sha256: String::new(),
    })
}

/// Replay the influence measurements and report receipt, including the frozen
/// temporal support policy and every per-window classification.
pub fn verify_counterfactual_influence(
    report: &CounterfactualInfluenceReport,
    input: &CounterfactualInfluenceInput<'_, '_>,
    verified_calibration: &VerifiedRepeatabilityCalibration,
) -> Result<bool, CounterfactualInfluenceError> {
    if !influence_digest_valid(report)? {
        return Err(CounterfactualInfluenceError::InvalidInfluenceReceipt);
    }
    let expected = measure_counterfactual_influence_with_policy(
        input,
        verified_calibration,
        &report.support_policy,
    )?;
    if &expected != report {
        return Err(CounterfactualInfluenceError::InfluenceReplayMismatch);
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer_execution_receipt::{
        RendererBackend, RendererProcessObservation,
    };

    fn config() -> InfluenceAnalysisConfig {
        InfluenceAnalysisConfig {
            window_size_frames: 16,
            hop_size_frames: 8,
            minimum_rms_delta: 1.0e-6,
            minimum_spectral_distance: 1.0e-4,
            noise_floor_multiplier: 2.0,
        }
    }

    fn support_window(
        window_index: usize,
        channel: usize,
        enter: bool,
        release: bool,
    ) -> InfluenceWindowMeasurement {
        InfluenceWindowMeasurement {
            start_frame: window_index * 10,
            end_frame: window_index * 10 + 10,
            channel,
            rms_delta: if enter { 2.0 } else if release { 0.8 } else { 0.1 },
            peak_sample_delta: if enter { 2.0 } else if release { 0.8 } else { 0.1 },
            spectral_magnitude_distance: if enter { 0.2 } else if release { 0.08 } else { 0.01 },
            calibrated_rms_threshold: 1.0,
            calibrated_spectral_threshold: 0.1,
            release_rms_threshold: 0.5,
            release_spectral_threshold: 0.05,
            effect_detected: enter,
            release_detected: release,
            in_qualified_support: false,
            bridged_gap: false,
        }
    }

    fn support_policy(
        minimum_active_windows: usize,
        maximum_gap_windows: usize,
        tail_extension_frames: u64,
    ) -> InfluenceSupportPolicy {
        InfluenceSupportPolicy {
            minimum_active_windows,
            maximum_gap_windows,
            tail_extension_frames,
            ..InfluenceSupportPolicy::default()
        }
    }

    fn audio(samples: &[f32]) -> AudioView<'_> {
        AudioView {
            interleaved_samples: samples,
            sample_rate_hz: 48_000,
            channels: 1,
        }
    }

    fn push_delta_event(track: &mut Vec<u8>, delta: u8, bytes: &[u8]) {
        assert!(delta < 0x80, "fixture uses one-byte MIDI delta times");
        track.push(delta);
        track.extend_from_slice(bytes);
    }

    fn push_lineage_marker(track: &mut Vec<u8>, delta: u8, source_index: usize) {
        let payload = crate::performed_event_map::encode_midi_lineage_marker_payload(
            source_index,
            0,
        );
        push_delta_event(track, delta, &[0xff, 0x7f, 0x10]);
        track.extend_from_slice(&payload);
    }

    /// Two-note, format-0 MIDI. Event 0 occupies ticks 0..1; event 1 occupies
    /// ticks 1..2. At 48 kHz/default 500,000 us per beat/480 TPB, each tick
    /// occupies 50 PCM frames.
    fn baseline_midi() -> Vec<u8> {
        let mut track = Vec::new();
        push_lineage_marker(&mut track, 0, 0);
        push_delta_event(&mut track, 0, &[0x90, 60, 90]);
        push_delta_event(&mut track, 1, &[0x80, 60, 0]);
        push_lineage_marker(&mut track, 0, 1);
        push_delta_event(&mut track, 0, &[0x90, 64, 90]);
        push_delta_event(&mut track, 1, &[0x80, 64, 0]);
        push_delta_event(&mut track, 0, &[0xff, 0x2f, 0x00]);

        let mut midi = vec![
            b'M', b'T', b'h', b'd', 0, 0, 0, 6, // header chunk / length
            0, 0, // format 0
            0, 1, // one track
            1, 0xe0, // 480 ticks per beat
        ];
        midi.extend_from_slice(b"MTrk");
        midi.extend_from_slice(&(track.len() as u32).to_be_bytes());
        midi.extend_from_slice(&track);
        midi
    }

    /// Counterfactual schedule with only event 0 removed. Event 1 retains its
    /// canonical Score identity even though its MIDI note-on ordinal changes.
    fn without_event_zero_midi() -> Vec<u8> {
        let mut track = Vec::new();
        push_lineage_marker(&mut track, 1, 1);
        push_delta_event(&mut track, 0, &[0x90, 64, 90]);
        push_delta_event(&mut track, 1, &[0x80, 64, 0]);
        push_delta_event(&mut track, 0, &[0xff, 0x2f, 0x00]);

        let mut midi = vec![
            b'M', b'T', b'h', b'd', 0, 0, 0, 6,
            0, 0,
            0, 1,
            1, 0xe0,
        ];
        midi.extend_from_slice(b"MTrk");
        midi.extend_from_slice(&(track.len() as u32).to_be_bytes());
        midi.extend_from_slice(&track);
        midi
    }

    /// Same declared removal of event 0, but event 1 is illegally transposed.
    /// The symbolic verifier must reject before acoustic metrics can bless it.
    fn without_event_zero_but_change_event_one_pitch() -> Vec<u8> {
        let mut track = Vec::new();
        push_lineage_marker(&mut track, 1, 1);
        push_delta_event(&mut track, 0, &[0x90, 65, 90]);
        push_delta_event(&mut track, 1, &[0x80, 65, 0]);
        push_delta_event(&mut track, 0, &[0xff, 0x2f, 0x00]);

        let mut midi = vec![
            b'M', b'T', b'h', b'd', 0, 0, 0, 6,
            0, 0,
            0, 1,
            1, 0xe0,
        ];
        midi.extend_from_slice(b"MTrk");
        midi.extend_from_slice(&(track.len() as u32).to_be_bytes());
        midi.extend_from_slice(&track);
        midi
    }

    fn float_wav(samples: &[f32]) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for sample in samples {
                writer.write_sample(*sample).unwrap();
            }
            writer.finalize().unwrap();
        }
        cursor.into_inner()
    }

    fn process_observation(render_id: &str, gain: f32) -> RendererProcessObservation {
        let executable = sha256_hex(b"fixture-fluidsynth-binary");
        let soundfont = sha256_hex(b"fixture-soundfont");
        RendererProcessObservation {
            executable_path: "/fixture/bin/fluidsynth".into(),
            executable_sha256_before: executable.clone(),
            executable_sha256_after: executable,
            soundfont_path: "/fixture/soundfont.sf2".into(),
            soundfont_sha256_before: soundfont.clone(),
            soundfont_sha256_after: soundfont,
            arguments: vec![
                "-g".into(),
                format!("{gain:.1}"),
                "-T".into(),
                "wav".into(),
                "-F".into(),
                format!("/tmp/{render_id}.wav"),
                "-r".into(),
                "48000".into(),
                "-o".into(),
                "synth.effects-groups=1".into(),
                "-ni".into(),
                "/fixture/soundfont.sf2".into(),
                format!("/tmp/{render_id}.mid"),
            ],
            environment_sha256: sha256_hex(b"fixture-environment"),
            stdout_sha256: sha256_hex(b"FluidSynth fixture"),
            stderr_sha256: sha256_hex(b""),
            stdout_preview: "FluidSynth fixture".into(),
            stderr_preview: String::new(),
            stdout_truncated: false,
            stderr_truncated: false,
            exit_code: 0,
        }
    }

    const SCORE_BYTES: &[u8] = br#"{"score":"two-event-fixture"}"#;
    const CONFIG_BYTES: &[u8] = br#"{"sample_rate_hz":48000,"fx":"fixed"}"#;

    struct BoundRender {
        receipt: RendererExecutionReceipt,
        wav: Vec<u8>,
        map: PerformedEventMap,
        map_bytes: Vec<u8>,
        midi: Vec<u8>,
        process: RendererProcessObservation,
    }

    fn build_render(
        render_id: &str,
        midi: &[u8],
        samples: &[f32],
    ) -> BoundRender {
        build_render_with_gain(render_id, midi, samples, 0.8)
    }

    fn build_render_with_gain(
        render_id: &str,
        midi: &[u8],
        samples: &[f32],
        gain: f32,
    ) -> BoundRender {
        let wav = float_wav(samples);
        let view = audio(samples);
        let score_sha256 = sha256_hex(SCORE_BYTES);
        let map = PerformedEventMap::from_midi_bytes(
            midi,
            &score_sha256,
            view.sample_rate_hz,
            samples.len() as u64,
        )
        .unwrap();
        let map_bytes = serde_json::to_vec(&map).unwrap();
        let process = process_observation(render_id, gain);
        let executable = process.executable_sha256_after.clone();
        let soundfont = process.soundfont_sha256_after.clone();
        // Studio only includes this receipt field after full-score coverage
        // verifies. A counterfactual map deliberately excludes one source ID,
        // so it must remain an explicit map artifact, not a complete-map claim.
        let complete_map_digest = if map.events.len() == 2 {
            Some(map.receipt_sha256.as_str())
        } else {
            None
        };
        let receipt = RendererExecutionReceipt::from_successful_render(
            RendererReceiptInputs {
                render_id,
                backend: RendererBackend::FluidSynth,
                score_bytes: SCORE_BYTES,
                render_config_bytes: CONFIG_BYTES,
                input_midi_bytes: Some(midi),
                performed_schedule_bytes: None,
                renderer_executable_sha256: Some(&executable),
                soundfont_sha256: Some(&soundfont),
                process_exit_code: Some(0),
                process_observation: Some(process.clone()),
                performed_event_map_sha256: complete_map_digest,
                output_wav_bytes: &wav,
            },
        )
        .unwrap();
        BoundRender {
            receipt,
            wav,
            map,
            map_bytes,
            midi: midi.to_vec(),
            process,
        }
    }

    fn verify_bound_render(
        render: &BoundRender,
        samples: &[f32],
        excluded_source_event_index: Option<usize>,
    ) -> VerifiedRendererAudioBinding {
        let executable = render.process.executable_sha256_after.as_str();
        let soundfont = render.process.soundfont_sha256_after.as_str();
        verify_renderer_audio_binding(
            &render.receipt,
            RendererReceiptInputs {
                render_id: &render.receipt.render_id,
                backend: RendererBackend::FluidSynth,
                score_bytes: SCORE_BYTES,
                render_config_bytes: CONFIG_BYTES,
                input_midi_bytes: Some(&render.midi),
                performed_schedule_bytes: None,
                renderer_executable_sha256: Some(executable),
                soundfont_sha256: Some(soundfont),
                process_exit_code: Some(0),
                process_observation: Some(render.process.clone()),
                performed_event_map_sha256: render.receipt.performed_event_map_sha256.as_deref(),
                output_wav_bytes: &render.wav,
            },
            &render.map_bytes,
            2,
            &[60, 64],
            excluded_source_event_index,
            &audio(samples),
        )
        .unwrap()
    }

    #[test]
    fn renderer_receipt_replays_midi_map_process_and_exact_pcm() {
        let samples = vec![0.0f32; 128];
        let render = build_render("render-0", &baseline_midi(), &samples);
        let verified = verify_bound_render(&render, &samples, None);
        assert_eq!(verified.audio_sha256(), audio_sha256(&audio(&samples)));
        assert_eq!(verified.receipt_sha256(), render.receipt.receipt_sha256);
        assert_eq!(verified.performed_event_lineage().len(), 2);
        assert_eq!(verified.fluid_synth_effects_groups(), Some(1));

        let mut altered = samples.clone();
        altered[10] = 0.25;
        let executable = render.process.executable_sha256_after.as_str();
        let soundfont = render.process.soundfont_sha256_after.as_str();
        assert_eq!(
            verify_renderer_audio_binding(
                &render.receipt,
                RendererReceiptInputs {
                    render_id: &render.receipt.render_id,
                    backend: RendererBackend::FluidSynth,
                    score_bytes: SCORE_BYTES,
                    render_config_bytes: CONFIG_BYTES,
                    input_midi_bytes: Some(&render.midi),
                    performed_schedule_bytes: None,
                    renderer_executable_sha256: Some(executable),
                    soundfont_sha256: Some(soundfont),
                    process_exit_code: Some(0),
                    process_observation: Some(render.process.clone()),
                    performed_event_map_sha256: Some(&render.map.receipt_sha256),
                    output_wav_bytes: &render.wav,
                },
                &render.map_bytes,
                2,
                &[60, 64],
                None,
                &audio(&altered),
            )
            .unwrap_err(),
            CounterfactualInfluenceError::RendererAudioMismatch
        );
    }

    #[test]
    fn repeated_reference_to_one_noop_receipt_does_not_fake_three_rerenders() {
        let samples = vec![0.0f32; 128];
        let midi = baseline_midi();
        let anchor = build_render("unique-anchor", &midi, &samples);
        let anchor_evidence = verify_bound_render(&anchor, &samples, None);
        let noop = build_render("one-noop-only", &midi, &samples);
        let noop_evidence = verify_bound_render(&noop, &samples, None);
        let repeated_audio = [
            audio(&samples),
            audio(&samples),
            audio(&samples),
        ];
        let repeated_receipts = [
            noop_evidence.clone(),
            noop_evidence.clone(),
            noop_evidence,
        ];

        assert_eq!(
            calibrate_repeatability(
                &audio(&samples),
                &repeated_audio,
                SCORE_BYTES,
                &anchor.map_bytes,
                &midi,
                &anchor_evidence,
                &repeated_receipts,
                &config(),
            )
            .unwrap_err(),
            CounterfactualInfluenceError::CalibrationReplayMismatch
        );
    }

    #[test]
    fn renderer_profile_normalizes_temporary_paths_but_binds_actual_options() {
        let samples = vec![0.0f32; 128];
        let midi = baseline_midi();
        let first = build_render("first-temp-path", &midi, &samples);
        let second = build_render("second-temp-path", &midi, &samples);
        let changed_gain = build_render_with_gain("changed-gain", &midi, &samples, 0.9);
        let first_evidence = verify_bound_render(&first, &samples, None);
        let second_evidence = verify_bound_render(&second, &samples, None);
        let changed_gain_evidence = verify_bound_render(&changed_gain, &samples, None);

        assert_ne!(
            first_evidence.receipt_sha256(),
            second_evidence.receipt_sha256()
        );
        assert_eq!(
            first_evidence.renderer_identity_sha256(),
            second_evidence.renderer_identity_sha256()
        );
        assert_eq!(
            first_evidence.renderer_config_sha256(),
            second_evidence.renderer_config_sha256()
        );
        assert_ne!(
            first_evidence.renderer_config_sha256(),
            changed_gain_evidence.renderer_config_sha256()
        );
    }

    #[test]
    fn repeatability_calibration_requires_three_noop_renders_and_replays() {
        let samples = vec![0.0f32; 128];
        let midi = baseline_midi();
        let anchor = build_render("anchor", &midi, &samples);
        let anchor_evidence = verify_bound_render(&anchor, &samples, None);

        let one = build_render("noop-0", &midi, &samples);
        let one_evidence = verify_bound_render(&one, &samples, None);
        assert_eq!(
            calibrate_repeatability(
                &audio(&samples),
                &[audio(&samples)],
                SCORE_BYTES,
                &anchor.map_bytes,
                &midi,
                &anchor_evidence,
                &[one_evidence],
                &config(),
            )
            .unwrap_err(),
            CounterfactualInfluenceError::TooFewNoopRenders
        );

        let repeats = [
            build_render("noop-1", &midi, &samples),
            build_render("noop-2", &midi, &samples),
            build_render("noop-3", &midi, &samples),
        ];
        let repeat_evidence: Vec<_> = repeats
            .iter()
            .map(|render| verify_bound_render(render, &samples, None))
            .collect();
        let repeat_audio: Vec<_> = (0..repeats.len())
            .map(|_| audio(&samples))
            .collect();
        let calibration = calibrate_repeatability(
            &audio(&samples),
            &repeat_audio,
            SCORE_BYTES,
            &anchor.map_bytes,
            &midi,
            &anchor_evidence,
            &repeat_evidence,
            &config(),
        )
        .unwrap();
        assert_eq!(calibration.noop_audio_sha256.len(), 3);
        assert_eq!(calibration.noop_renderer_receipt_sha256.len(), 3);
        let verified = verify_repeatability_calibration(
            &calibration,
            &audio(&samples),
            &repeat_audio,
            SCORE_BYTES,
            &anchor.map_bytes,
            &midi,
            &anchor_evidence,
            &repeat_evidence,
        )
        .unwrap();
        assert_eq!(
            verified.calibration().performed_event_map_receipt_sha256,
            anchor.map.receipt_sha256
        );
    }

    #[test]
    fn removing_one_note_preserves_all_other_symbolic_events_and_measures_pcm_effect() {
        let mut baseline_samples = vec![0.0f32; 128];
        for sample in &mut baseline_samples[0..50] {
            *sample = 0.25;
        }
        for sample in &mut baseline_samples[50..100] {
            *sample = 0.15;
        }
        let mut counterfactual_samples = baseline_samples.clone();
        counterfactual_samples[0..50].fill(0.0);

        let baseline_midi = baseline_midi();
        let counterfactual_midi = without_event_zero_midi();
        let anchor = build_render("baseline", &baseline_midi, &baseline_samples);
        let anchor_evidence = verify_bound_render(&anchor, &baseline_samples, None);

        let noops = [
            build_render("baseline-noop-1", &baseline_midi, &baseline_samples),
            build_render("baseline-noop-2", &baseline_midi, &baseline_samples),
            build_render("baseline-noop-3", &baseline_midi, &baseline_samples),
        ];
        let noop_evidence: Vec<_> = noops
            .iter()
            .map(|render| verify_bound_render(render, &baseline_samples, None))
            .collect();
        let noop_audio: Vec<_> = (0..noops.len())
            .map(|_| audio(&baseline_samples))
            .collect();
        let calibration = calibrate_repeatability(
            &audio(&baseline_samples),
            &noop_audio,
            SCORE_BYTES,
            &anchor.map_bytes,
            &baseline_midi,
            &anchor_evidence,
            &noop_evidence,
            &config(),
        )
        .unwrap();
        let calibration =
            verify_repeatability_calibration(
                &calibration,
                &audio(&baseline_samples),
                &noop_audio,
                SCORE_BYTES,
                &anchor.map_bytes,
                &baseline_midi,
                &anchor_evidence,
                &noop_evidence,
            )
            .unwrap();

        let counterfactual = build_render(
            "counterfactual-remove-event-0",
            &counterfactual_midi,
            &counterfactual_samples,
        );
        let counterfactual_evidence =
            verify_bound_render(&counterfactual, &counterfactual_samples, Some(0));

        let baseline_audio_view = audio(&baseline_samples);
        let counterfactual_audio_view = audio(&counterfactual_samples);
        let experiment = CounterfactualInfluenceInput {
            source_event_id: "score_event_index:0",
            intervention: EventIntervention::Remove,
            source_score_bytes: SCORE_BYTES,
            performed_event_map_bytes: &anchor.map_bytes,
            counterfactual_performed_event_map_bytes: &counterfactual.map_bytes,
            expected_score_event_count: 2,
            expected_primary_midi_by_score_index: &[60, 64],
            baseline_schedule_bytes: &baseline_midi,
            counterfactual_schedule_bytes: &counterfactual_midi,
            baseline_renderer_evidence: &anchor_evidence,
            counterfactual_renderer_evidence: &counterfactual_evidence,
            baseline_audio: &baseline_audio_view,
            counterfactual_audio: &counterfactual_audio_view,
        };

        let report = measure_counterfactual_influence(&experiment, &calibration).unwrap();
        assert_eq!(report.source_event_id, "score_event_index:0");
        assert_eq!(report.disposition, InfluenceDisposition::EffectDetected);
        assert!(report.affected_channel_windows > 0);
        assert_eq!(
            report.baseline_renderer_receipt_sha256,
            anchor_evidence.receipt_sha256()
        );
        assert_eq!(
            report.counterfactual_renderer_receipt_sha256,
            counterfactual_evidence.receipt_sha256()
        );
        assert_eq!(report.counterfactual_event_map_receipt_sha256, counterfactual.map.receipt_sha256);
        assert!(!report.measured_influence_intervals.is_empty());
        assert!(!report.qualified_support_intervals.is_empty());
        assert_eq!(report.window_measurements.len(), report.total_channel_windows);
        assert_eq!(report.disposition, InfluenceDisposition::EffectDetected);
        assert!(verify_counterfactual_influence(&report, &experiment, &calibration).unwrap());

        let mut policy_tampered = report.clone();
        policy_tampered.support_policy.tail_extension_frames = 1;
        assert_eq!(
            verify_counterfactual_influence(&policy_tampered, &experiment, &calibration)
                .unwrap_err(),
            CounterfactualInfluenceError::InvalidInfluenceReceipt,
        );
        let mut trace_tampered = report.clone();
        trace_tampered.window_measurements[0].effect_detected =
            !trace_tampered.window_measurements[0].effect_detected;
        assert_eq!(
            verify_counterfactual_influence(&trace_tampered, &experiment, &calibration)
                .unwrap_err(),
            CounterfactualInfluenceError::InvalidInfluenceReceipt,
        );
    }

    #[test]
    fn counterfactual_exclusion_does_not_authorize_a_protected_pitch_change() {
        let samples = vec![0.0f32; 128];
        let render = build_render(
            "changed-protected-pitch",
            &without_event_zero_but_change_event_one_pitch(),
            &samples,
        );
        let executable = render.process.executable_sha256_after.as_str();
        let soundfont = render.process.soundfont_sha256_after.as_str();
        assert_eq!(
            verify_renderer_audio_binding(
                &render.receipt,
                RendererReceiptInputs {
                    render_id: &render.receipt.render_id,
                    backend: RendererBackend::FluidSynth,
                    score_bytes: SCORE_BYTES,
                    render_config_bytes: CONFIG_BYTES,
                    input_midi_bytes: Some(&render.midi),
                    performed_schedule_bytes: None,
                    renderer_executable_sha256: Some(executable),
                    soundfont_sha256: Some(soundfont),
                    process_exit_code: Some(0),
                    process_observation: Some(render.process.clone()),
                    performed_event_map_sha256: None,
                    output_wav_bytes: &render.wav,
                },
                &render.map_bytes,
                2,
                &[60, 64],
                Some(0),
                &audio(&samples),
            )
            .unwrap_err(),
            CounterfactualInfluenceError::RendererReceiptInvalid
        );
    }

    #[test]
    fn calibration_and_intervention_reject_renderer_or_schedule_substitution() {
        let samples = vec![0.0f32; 128];
        let baseline = build_render("baseline", &baseline_midi(), &samples);
        let baseline_evidence = verify_bound_render(&baseline, &samples, None);
        let noops = [
            build_render("noop-1", &baseline.midi, &samples),
            build_render("noop-2", &baseline.midi, &samples),
            build_render("noop-3", &baseline.midi, &samples),
        ];
        let noop_evidence: Vec<_> = noops
            .iter()
            .map(|render| verify_bound_render(render, &samples, None))
            .collect();
        let noop_audio: Vec<_> = (0..3).map(|_| audio(&samples)).collect();
        let calibration = verify_repeatability_calibration(
            &calibrate_repeatability(
                &audio(&samples),
                &noop_audio,
                SCORE_BYTES,
                &baseline.map_bytes,
                &baseline.midi,
                &baseline_evidence,
                &noop_evidence,
                &config(),
            )
            .unwrap(),
            &audio(&samples),
            &noop_audio,
            SCORE_BYTES,
            &baseline.map_bytes,
            &baseline.midi,
            &baseline_evidence,
            &noop_evidence,
        )
        .unwrap();

        let mut altered_config_evidence =
            verify_bound_render(&baseline, &samples, None);
        altered_config_evidence.renderer_config_sha256 = sha256_hex(b"substituted-config");
        let counterfactual_samples = vec![0.0f32; 128];
        let counterfactual = build_render(
            "counterfactual",
            &without_event_zero_midi(),
            &counterfactual_samples,
        );
        let counterfactual_evidence =
            verify_bound_render(&counterfactual, &counterfactual_samples, Some(0));
        let experiment = CounterfactualInfluenceInput {
            source_event_id: "score_event_index:0",
            intervention: EventIntervention::Remove,
            source_score_bytes: SCORE_BYTES,
            performed_event_map_bytes: &baseline.map_bytes,
            counterfactual_performed_event_map_bytes: &counterfactual.map_bytes,
            expected_score_event_count: 2,
            expected_primary_midi_by_score_index: &[60, 64],
            baseline_schedule_bytes: &baseline.midi,
            counterfactual_schedule_bytes: &counterfactual.midi,
            baseline_renderer_evidence: &altered_config_evidence,
            counterfactual_renderer_evidence: &counterfactual_evidence,
            baseline_audio: &audio(&samples),
            counterfactual_audio: &audio(&counterfactual_samples),
        };
        assert_eq!(
            measure_counterfactual_influence(&experiment, &calibration).unwrap_err(),
            CounterfactualInfluenceError::RendererIdentityMismatch
        );
    }

    #[test]
    fn hysteresis_retains_a_sustained_effect_below_the_enter_threshold() {
        let mut measurements = vec![
            support_window(0, 0, true, true),
            support_window(1, 0, false, true),
            support_window(2, 0, false, true),
        ];
        let (intervals, counts) = qualify_support_windows(
            &mut measurements,
            1,
            30,
            &support_policy(2, 0, 0),
        )
        .unwrap();
        assert_eq!(intervals, vec![FrameInterval { start_frame: 0, end_frame: 30 }]);
        assert_eq!(counts.qualified_channel_windows, 3);
        assert!(measurements.iter().all(|window| window.in_qualified_support));
    }

    #[test]
    fn isolated_spike_is_preserved_as_raw_evidence_but_not_qualified_support() {
        let mut measurements = vec![
            support_window(0, 0, true, true),
            support_window(1, 0, false, false),
            support_window(2, 0, false, false),
        ];
        assert!(measurements[0].effect_detected);
        let (intervals, counts) = qualify_support_windows(
            &mut measurements,
            1,
            30,
            &support_policy(2, 1, 0),
        )
        .unwrap();
        assert!(intervals.is_empty());
        assert_eq!(counts.qualified_channel_windows, 0);
        assert!(!measurements[0].in_qualified_support);
    }

    #[test]
    fn bounded_single_window_gap_is_bridged_but_longer_gap_closes_episode() {
        let mut bridged = vec![
            support_window(0, 0, true, true),
            support_window(1, 0, false, false),
            support_window(2, 0, true, true),
        ];
        let (intervals, counts) = qualify_support_windows(
            &mut bridged,
            1,
            30,
            &support_policy(2, 1, 0),
        )
        .unwrap();
        assert_eq!(intervals, vec![FrameInterval { start_frame: 0, end_frame: 30 }]);
        assert_eq!(counts.bridged_channel_windows, 1);
        assert!(bridged[1].bridged_gap);
        assert!(bridged[1].in_qualified_support);

        let mut long_gap = vec![
            support_window(0, 0, true, true),
            support_window(1, 0, false, false),
            support_window(2, 0, false, false),
            support_window(3, 0, true, true),
        ];
        let (intervals, counts) = qualify_support_windows(
            &mut long_gap,
            1,
            40,
            &support_policy(2, 1, 0),
        )
        .unwrap();
        assert!(intervals.is_empty());
        assert_eq!(counts.qualified_channel_windows, 0);
    }

    #[test]
    fn explicit_tail_extension_is_clamped_to_rendered_output_horizon() {
        let mut measurements = vec![
            support_window(0, 0, true, true),
            support_window(1, 0, false, true),
        ];
        let (intervals, counts) = qualify_support_windows(
            &mut measurements,
            1,
            25,
            &support_policy(2, 0, 20),
        )
        .unwrap();
        assert_eq!(intervals, vec![FrameInterval { start_frame: 0, end_frame: 25 }]);
        assert_eq!(counts.tail_extension_frames_applied, 5);
    }

    #[test]
    fn measured_empty_scan_never_becomes_support_from_tail_policy() {
        let mut measurements = vec![
            support_window(0, 0, false, false),
            support_window(1, 0, false, false),
            support_window(2, 0, false, false),
        ];
        let (intervals, counts) = qualify_support_windows(
            &mut measurements,
            1,
            30,
            &support_policy(1, 1, 20),
        )
        .unwrap();
        assert!(intervals.is_empty());
        assert_eq!(counts.tail_extension_frames_applied, 0);
        assert!(measurements.iter().all(|window| !window.in_qualified_support));
    }

    #[test]
    fn channels_are_qualified_independently_before_interval_union() {
        let mut measurements = vec![
            support_window(0, 0, true, true),
            support_window(0, 1, false, false),
            support_window(1, 0, false, false),
            support_window(1, 1, true, true),
            support_window(2, 0, true, true),
            support_window(2, 1, false, false),
        ];
        let (intervals, counts) = qualify_support_windows(
            &mut measurements,
            2,
            30,
            &support_policy(1, 0, 0),
        )
        .unwrap();
        assert_eq!(
            intervals,
            vec![FrameInterval { start_frame: 0, end_frame: 30 }],
        );
        assert_eq!(counts.qualified_channel_windows, 3);
    }

    #[test]
    fn invalid_hysteresis_ratios_and_tail_bounds_fail_closed() {
        let mut invalid_ratio = InfluenceSupportPolicy::default();
        invalid_ratio.release_threshold_ratio_microunits = 1_000_000;
        assert!(validate_support_policy(&invalid_ratio, 100, 4).is_err());
        let mut invalid_tail = InfluenceSupportPolicy::default();
        invalid_tail.tail_extension_frames = 101;
        assert!(validate_support_policy(&invalid_tail, 100, 4).is_err());
        let mut invalid_minimum = InfluenceSupportPolicy::default();
        invalid_minimum.minimum_active_windows = 5;
        assert!(validate_support_policy(&invalid_minimum, 100, 4).is_err());
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
