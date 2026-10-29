// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Content-binding envelope for symbolic and rendered-audio edit evidence.
//!
//! This module is feature-gated with the theory capability. It binds the exact
//! source/target symbolic events, the symbolic verifier receipt, the source/
//! target audio buffers, the audio-only composite receipt, and a renderer
//! manifest.
//!
//! Important trust boundary: hashes prove equality/integrity of the supplied
//! content and reports. They do not prove that an untrusted renderer actually
//! consumed the target score. Callers should create this envelope at the
//! render boundary inside the renderer adapter; signed provenance or trusted
//! execution is a separate layer.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use symthaea_music_theory::edit_locality::{
    EditLocalityError, EditLocalityEvidence, LocalityEvent, MAX_EDIT_LOCALITY_EVENTS,
    verify_edit_locality_evidence,
};
use symthaea_music_theory::Score;

use crate::edit_contract::{
    AttributeValue, EditAttribute, EditObligation, FrozenEditContract, ObligationRequirement,
    ObligationScope, QuantizedTolerance, SupportedEditOperation,
};
use crate::audio_preservation::{
    AudioAuditInputError, AudioPreservationReport, AudioView, FrameInterval,
    OnsetPreservationConfig, OnsetPreservationReport, PreservationConfig,
    PreservationVerdict, audio_content_sha256,
    compose_audio_preservation_evidence,
    verify_audio_preservation_against_buffers, verify_onset_preservation_against_buffers,
};

/// Version for the symbolic/audio render-binding envelope.
pub const SYMBOLIC_AUDIO_BINDING_VERSION: &str = "mel-mr-eval-008-render-binding-v0.1";
/// Version of the renderer manifest serialized by this module.
pub const RENDERER_MANIFEST_VERSION: &str = "renderer-manifest-v0.1";

/// Reproducibility metadata required to interpret rendered audio.
///
/// Optional instrument/effects hashes must be explicitly absent or identify
/// exact configuration blobs. render_config_sha256 is always required.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RendererManifest {
    pub manifest_version: String,
    pub renderer_id: String,
    pub renderer_version: String,
    pub instrument_bank_sha256: Option<String>,
    pub effects_config_sha256: Option<String>,
    pub render_config_sha256: String,
    pub sample_rate_hz: u32,
    pub channels: usize,
    /// Signed frame offset from score-time positions to rendered-buffer positions.
    /// Positive means output events are delayed by this many frames.
    pub render_latency_frames: i64,
    pub latency_policy: String,
}

/// Content-bound join between symbolic and rendered-audio verification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SymbolicAudioRenderBinding {
    pub binding_version: String,
    pub source_score_sha256: String,
    pub target_score_sha256: String,
    pub symbolic_receipt_sha256: String,
    pub symbolic_verdict: PreservationVerdict,
    pub source_audio_sha256: String,
    pub target_audio_sha256: String,
    pub audio_composite_receipt_sha256: String,
    pub audio_verdict: PreservationVerdict,
    pub renderer_manifest: RendererManifest,
    pub renderer_manifest_sha256: String,
    pub sample_rate_hz: u32,
    pub channels: usize,
    pub total_frames: usize,
    pub overall_verdict: PreservationVerdict,
    pub reasons: Vec<String>,
    /// SHA-256 over the binding with this field blank.
    pub binding_receipt_sha256: String,
}

/// Fail-closed error while constructing or checking a cross-modal envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolicAudioBindingError {
    Audio(AudioAuditInputError),
    Symbolic(EditLocalityError),
    InvalidRendererManifest(&'static str),
    InvalidEvidence(&'static str),
    UnsupportedScore(&'static str),
    SerializationFailure,
}

impl std::fmt::Display for SymbolicAudioBindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Audio(error) => write!(f, "audio evidence error: {error}"),
            Self::Symbolic(error) => write!(f, "symbolic evidence error: {error:?}"),
            Self::InvalidRendererManifest(reason) => {
                write!(f, "invalid renderer manifest: {reason}")
            }
            Self::InvalidEvidence(reason) => write!(f, "invalid edit evidence: {reason}"),
            Self::UnsupportedScore(reason) => write!(f, "score cannot be represented by the current locality contract: {reason}"),
            Self::SerializationFailure => write!(f, "could not serialize binding receipt"),
        }
    }
}

impl std::error::Error for SymbolicAudioBindingError {}

impl From<AudioAuditInputError> for SymbolicAudioBindingError {
    fn from(value: AudioAuditInputError) -> Self {
        Self::Audio(value)
    }
}

impl From<EditLocalityError> for SymbolicAudioBindingError {
    fn from(value: EditLocalityError) -> Self {
        Self::Symbolic(value)
    }
}

fn is_lower_hex_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Digest a renderer manifest under a distinct domain-separated tag.
pub fn renderer_manifest_sha256(
    manifest: &RendererManifest,
) -> Result<String, SymbolicAudioBindingError> {
    validate_renderer_manifest(manifest)?;
    let bytes = serde_json::to_vec(manifest)
        .map_err(|_| SymbolicAudioBindingError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous-dynamics-renderer-manifest-v1\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_renderer_manifest(
    manifest: &RendererManifest,
) -> Result<(), SymbolicAudioBindingError> {
    if manifest.manifest_version != RENDERER_MANIFEST_VERSION {
        return Err(SymbolicAudioBindingError::InvalidRendererManifest(
            "unsupported manifest version",
        ));
    }
    if manifest.renderer_id.trim().is_empty()
        || manifest.renderer_version.trim().is_empty()
        || manifest.latency_policy.trim().is_empty()
    {
        return Err(SymbolicAudioBindingError::InvalidRendererManifest(
            "renderer identity, version, and latency policy must be non-empty",
        ));
    }
    if manifest.sample_rate_hz == 0 || manifest.channels == 0 {
        return Err(SymbolicAudioBindingError::InvalidRendererManifest(
            "sample rate and channel count must be non-zero",
        ));
    }
    if !is_lower_hex_sha256(&manifest.render_config_sha256) {
        return Err(SymbolicAudioBindingError::InvalidRendererManifest(
            "render_config_sha256 must be 64 lowercase hexadecimal characters",
        ));
    }
    if manifest
        .instrument_bank_sha256
        .as_ref()
        .is_some_and(|hash| !is_lower_hex_sha256(hash))
    {
        return Err(SymbolicAudioBindingError::InvalidRendererManifest(
            "instrument_bank_sha256 must be absent or a valid lowercase SHA-256 digest",
        ));
    }
    if manifest
        .effects_config_sha256
        .as_ref()
        .is_some_and(|hash| !is_lower_hex_sha256(hash))
    {
        return Err(SymbolicAudioBindingError::InvalidRendererManifest(
            "effects_config_sha256 must be absent or a valid lowercase SHA-256 digest",
        ));
    }
    Ok(())
}

fn binding_receipt_sha256(
    binding: &SymbolicAudioRenderBinding,
) -> Result<String, SymbolicAudioBindingError> {
    let mut unsealed = binding.clone();
    unsealed.binding_receipt_sha256.clear();
    let bytes = serde_json::to_vec(&unsealed)
        .map_err(|_| SymbolicAudioBindingError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous-dynamics-symbolic-audio-binding-v1\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Construct a cross-modal binding from the concrete score/audio inputs and
/// their independently verified reports. Source and target audio hashes are
/// recomputed from the passed buffers; a report cannot choose them by fiat.
///
/// The envelope is a caller-produced claim that the buffers correspond to the
/// scores under the supplied renderer manifest. It does not attest causal
/// execution by an untrusted renderer; generate it within a trusted adapter.
pub fn create_symbolic_audio_render_binding(
    source_score: &[LocalityEvent],
    target_score: &[LocalityEvent],
    symbolic_evidence: &EditLocalityEvidence,
    source_audio: AudioView<'_>,
    target_audio: AudioView<'_>,
    acoustic_evidence: &AudioPreservationReport,
    onset_evidence: &OnsetPreservationReport,
    renderer_manifest: &RendererManifest,
) -> Result<SymbolicAudioRenderBinding, SymbolicAudioBindingError> {
    if !verify_edit_locality_evidence(source_score, target_score, symbolic_evidence)? {
        return Err(SymbolicAudioBindingError::InvalidEvidence(
            "symbolic receipt or source/target score binding did not verify",
        ));
    }
    // Recompute both child reports against the exact buffers, using the
    // recorded authorization masks/configs. A sealed but fabricated metric is
    // not sufficient. Only then compose the child receipts.
    if !verify_audio_preservation_against_buffers(source_audio, target_audio, acoustic_evidence)?
        || !verify_onset_preservation_against_buffers(source_audio, target_audio, onset_evidence)?
    {
        return Err(SymbolicAudioBindingError::InvalidEvidence(
            "acoustic or onset report did not replay against the exact audio buffers",
        ));
    }
    let audio_evidence =
        compose_audio_preservation_evidence(acoustic_evidence, onset_evidence)?;

    let source_audio_sha256 = audio_content_sha256(source_audio)?;
    let target_audio_sha256 = audio_content_sha256(target_audio)?;
    let source_frames = source_audio.interleaved_samples.len() / source_audio.channels;
    let target_frames = target_audio.interleaved_samples.len() / target_audio.channels;

    if source_audio.sample_rate_hz != target_audio.sample_rate_hz
        || source_audio.channels != target_audio.channels
        || source_frames != target_frames
    {
        return Err(SymbolicAudioBindingError::InvalidEvidence(
            "source and target audio format/length must match for a paired edit",
        ));
    }
    if audio_evidence.reference_audio_sha256 != source_audio_sha256
        || audio_evidence.candidate_audio_sha256 != target_audio_sha256
        || audio_evidence.sample_rate_hz != source_audio.sample_rate_hz
        || audio_evidence.channels != source_audio.channels
        || audio_evidence.total_frames != source_frames
    {
        return Err(SymbolicAudioBindingError::InvalidEvidence(
            "audio composite report does not bind to the supplied source/target buffers",
        ));
    }

    let renderer_manifest_sha256 = renderer_manifest_sha256(renderer_manifest)?;
    if renderer_manifest.sample_rate_hz != source_audio.sample_rate_hz
        || renderer_manifest.channels != source_audio.channels
    {
        return Err(SymbolicAudioBindingError::InvalidRendererManifest(
            "manifest audio format does not match the rendered buffers",
        ));
    }

    // A concrete symbolic failure or audio failure is never compensated by a
    // passing sibling channel. An audio abstention prevents a composite pass.
    let mut reasons = Vec::new();
    let overall_verdict = if !symbolic_evidence.passed
        || audio_evidence.verdict == PreservationVerdict::Fail
    {
        if !symbolic_evidence.passed {
            reasons.push("symbolic edit-locality verifier failed".to_owned());
        }
        if audio_evidence.verdict == PreservationVerdict::Fail {
            reasons.push("audio preservation verifier failed".to_owned());
        }
        PreservationVerdict::Fail
    } else if audio_evidence.verdict == PreservationVerdict::Abstain {
        reasons.push("audio preservation evidence abstained".to_owned());
        PreservationVerdict::Abstain
    } else {
        PreservationVerdict::Pass
    };

    let mut binding = SymbolicAudioRenderBinding {
        binding_version: SYMBOLIC_AUDIO_BINDING_VERSION.to_owned(),
        source_score_sha256: symbolic_evidence.source_score_sha256.clone(),
        target_score_sha256: symbolic_evidence.target_score_sha256.clone(),
        symbolic_receipt_sha256: symbolic_evidence.receipt_sha256.clone(),
        symbolic_verdict: if symbolic_evidence.passed {
            PreservationVerdict::Pass
        } else {
            PreservationVerdict::Fail
        },
        source_audio_sha256,
        target_audio_sha256,
        audio_composite_receipt_sha256: audio_evidence.receipt_sha256.clone(),
        audio_verdict: audio_evidence.verdict,
        renderer_manifest: renderer_manifest.clone(),
        renderer_manifest_sha256,
        sample_rate_hz: source_audio.sample_rate_hz,
        channels: source_audio.channels,
        total_frames: source_frames,
        overall_verdict,
        reasons,
        binding_receipt_sha256: String::new(),
    };
    binding.binding_receipt_sha256 = binding_receipt_sha256(&binding)?;
    Ok(binding)
}


/// Full-Muse-Score wrapper around the narrower symbolic/audio binding.
///
/// The inner symbolic verifier intentionally supports only a single,
/// index-preserving monophonic line. This wrapper binds the entire serialized
/// Score as well, and rejects score metadata that the current locality
/// contract does not model instead of silently ignoring it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FullScoreRenderBinding {
    pub binding_version: String,
    pub source_score_content_sha256: String,
    pub target_score_content_sha256: String,
    pub preservation_profile_sha256: String,
    pub projected_binding: SymbolicAudioRenderBinding,
    pub receipt_sha256: String,
}

/// Explicit frozen detector/calibration profile required by the full-Score gate.
/// In production this must be loaded from a trusted allowlist/registry, not
/// assembled from the child reports being checked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioPreservationProfile {
    pub profile_version: String,
    pub profile_id: String,
    pub acoustic_config: PreservationConfig,
    pub onset_config: OnsetPreservationConfig,
}


/// Hash the canonical frozen profile. This identifies the profile bytes but
/// does not establish approval by itself; callers must resolve the profile
/// from their trusted calibration registry.
pub fn audio_preservation_profile_sha256(
    profile: &AudioPreservationProfile,
) -> Result<String, SymbolicAudioBindingError> {
    if profile.profile_version != "audio-preservation-profile-v0.1"
        || profile.profile_id.trim().is_empty()
    {
        return Err(SymbolicAudioBindingError::InvalidEvidence(
            "unsupported or unnamed audio preservation profile",
        ));
    }
    if profile.acoustic_config.calibration_id.as_deref() != Some(profile.profile_id.as_str())
        || profile.onset_config.calibration_id.as_deref() != Some(profile.profile_id.as_str())
    {
        return Err(SymbolicAudioBindingError::InvalidEvidence(
            "both child configurations must use the exact profile ID",
        ));
    }
    let bytes = serde_json::to_vec(profile)
        .map_err(|_| SymbolicAudioBindingError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous-dynamics-audio-preservation-profile-v1\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn require_profile_matches_reports(
    profile: &AudioPreservationProfile,
    acoustic: &AudioPreservationReport,
    onset: &OnsetPreservationReport,
) -> Result<String, SymbolicAudioBindingError> {
    let digest = audio_preservation_profile_sha256(profile)?;
    if acoustic.config != profile.acoustic_config || onset.config != profile.onset_config {
        return Err(SymbolicAudioBindingError::InvalidEvidence(
            "audio detector settings do not match the selected frozen preservation profile",
        ));
    }
    Ok(digest)
}

fn full_score_sha256(score: &Score) -> Result<String, SymbolicAudioBindingError> {
    let bytes = serde_json::to_vec(score)
        .map_err(|_| SymbolicAudioBindingError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous-dynamics-muse-score-content-v1\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn validate_score_projection(score: &Score) -> Result<Vec<LocalityEvent>, SymbolicAudioBindingError> {
    if score.notes.is_empty() {
        return Err(SymbolicAudioBindingError::UnsupportedScore("score has no notes"));
    }
    if score.notes.len() > MAX_EDIT_LOCALITY_EVENTS {
        return Err(SymbolicAudioBindingError::UnsupportedScore(
            "score exceeds the monophonic locality event budget",
        ));
    }
    if !score.tempo_bpm.is_finite() || score.tempo_bpm <= 0.0 || score.meter == 0 {
        return Err(SymbolicAudioBindingError::UnsupportedScore(
            "tempo must be finite and positive and meter must be non-zero",
        ));
    }

    let ordered = score.events();
    let first_part = ordered[0].part;
    let first_role = ordered[0].role;
    if ordered.iter().any(|note| note.part != first_part || note.role != first_role) {
        return Err(SymbolicAudioBindingError::UnsupportedScore(
            "current projection supports one part and one voice role only",
        ));
    }

    let mut projected: Vec<LocalityEvent> = Vec::with_capacity(ordered.len());
    let mut previous_end: Option<f64> = None;
    let mut maximum_end = 0.0f64;
    for note in ordered {
        let onset = note.onset.beats();
        let duration = note.duration.beats();
        let end = onset + duration;
        if !onset.is_finite() || onset < 0.0 || !duration.is_finite() || duration <= 0.0 {
            return Err(SymbolicAudioBindingError::UnsupportedScore(
                "onsets must be finite/non-negative and durations finite/positive",
            ));
        }
        if let Some(prior_end) = previous_end {
            let prior_onset = projected
                .last()
                .expect("previous end implies a prior projected event")
                .onset_beats;
            if onset <= prior_onset || onset + 1.0e-9 < prior_end {
                return Err(SymbolicAudioBindingError::UnsupportedScore(
                    "simultaneous or overlapping notes are outside the monophonic contract",
                ));
            }
        }
        previous_end = Some(end);
        maximum_end = maximum_end.max(end);
        projected.push(LocalityEvent {
            pitch_midi: note.pitch.midi() as i32,
            onset_beats: onset,
            duration_beats: duration,
        });
    }

    if (score.total_beats.beats() - maximum_end).abs() > 1.0e-9 {
        return Err(SymbolicAudioBindingError::UnsupportedScore(
            "score total_beats is inconsistent with the last note end",
        ));
    }
    Ok(projected)
}

fn require_only_locality_dimensions_changed(
    source_score: &Score,
    target_score: &Score,
) -> Result<(), SymbolicAudioBindingError> {
    if source_score.key != target_score.key
        || source_score.tempo_bpm.to_bits() != target_score.tempo_bpm.to_bits()
        || source_score.meter != target_score.meter
    {
        return Err(SymbolicAudioBindingError::UnsupportedScore(
            "key, tempo, and meter edits are not covered by the current locality contract",
        ));
    }
    let source_events = source_score.events();
    let target_events = target_score.events();
    if source_events.len() != target_events.len() {
        return Err(SymbolicAudioBindingError::UnsupportedScore(
            "event insertion/deletion is not covered by the current locality contract",
        ));
    }
    for (source, target) in source_events.iter().zip(&target_events) {
        if source.part != target.part
            || source.role != target.role
            || source.emphasis != target.emphasis
            || source.velocity.to_bits() != target.velocity.to_bits()
            || source.section_intensity.to_bits() != target.section_intensity.to_bits()
        {
            return Err(SymbolicAudioBindingError::UnsupportedScore(
                "part, role, velocity, emphasis, or section intensity changed outside the supported edit contract",
            ));
        }
    }
    Ok(())
}


/// Convert a beat value to a conservative audio-frame boundary without
/// clipping. The source/target score must fit fully inside the rendered buffer.
fn beat_to_frame(
    beat: f64,
    tempo_bpm: f64,
    sample_rate_hz: u32,
    total_frames: usize,
    round_up: bool,
) -> Result<usize, SymbolicAudioBindingError> {
    if !beat.is_finite() || beat < 0.0 || !tempo_bpm.is_finite() || tempo_bpm <= 0.0 {
        return Err(SymbolicAudioBindingError::UnsupportedScore(
            "non-finite or negative beat position cannot map to audio frames",
        ));
    }
    let frame = beat * 60.0 * sample_rate_hz as f64 / tempo_bpm;
    if !frame.is_finite() || frame > total_frames as f64 + 1.0e-6 {
        return Err(SymbolicAudioBindingError::UnsupportedScore(
            "score timing extends beyond the rendered audio buffer",
        ));
    }
    let rounded = if round_up { frame.ceil() } else { frame.floor() };
    Ok((rounded as usize).min(total_frames))
}

/// Derive the only authorized temporal mask implied by a supported symbolic
/// operation. This prevents a caller from widening the audio exemption mask
/// independently of the recorded edit request.
fn derive_authorized_audio_mask(
    source: &[LocalityEvent],
    target: &[LocalityEvent],
    request: symthaea_music_theory::edit_locality::EditRequest,
    tempo_bpm: f64,
    sample_rate_hz: u32,
    total_frames: usize,
    render_latency_frames: i64,
) -> Result<Vec<FrameInterval>, SymbolicAudioBindingError> {
    if source.is_empty() || source.len() != target.len() {
        return Err(SymbolicAudioBindingError::UnsupportedScore(
            "source/target symbolic events must be non-empty and index-aligned",
        ));
    }
    let indices: Vec<usize> = match request.operation {
        symthaea_music_theory::edit_locality::EditOperation::Transpose { .. }
        | symthaea_music_theory::edit_locality::EditOperation::ScaleDurations { .. }
        | symthaea_music_theory::edit_locality::EditOperation::ScaleOnsets { .. } =>
            (0..source.len()).collect(),
        symthaea_music_theory::edit_locality::EditOperation::PitchAt { index, .. }
        | symthaea_music_theory::edit_locality::EditOperation::DurationAt { index, .. } => {
            if index >= source.len() {
                return Err(SymbolicAudioBindingError::UnsupportedScore(
                    "requested edit index is outside the symbolic event sequence",
                ));
            }
            vec![index]
        }
    };

    let mut intervals = Vec::with_capacity(indices.len());
    for index in indices {
        let source_event = source[index];
        let target_event = target[index];
        let start_beat = source_event.onset_beats.min(target_event.onset_beats);
        let source_end = source_event.onset_beats + source_event.duration_beats;
        let target_end = target_event.onset_beats + target_event.duration_beats;
        let end_beat = source_end.max(target_end);
        let unshifted_start = beat_to_frame(
            start_beat, tempo_bpm, sample_rate_hz, total_frames, false,
        )?;
        let unshifted_end = beat_to_frame(
            end_beat, tempo_bpm, sample_rate_hz, total_frames, true,
        )?;
        let shift_frame = |frame: usize| -> Result<usize, SymbolicAudioBindingError> {
            let shifted = frame as i128 + render_latency_frames as i128;
            if shifted < 0 || shifted > total_frames as i128 {
                return Err(SymbolicAudioBindingError::UnsupportedScore(
                    "renderer latency shift places an authorized interval outside the audio buffer",
                ));
            }
            Ok(shifted as usize)
        };
        let start_frame = shift_frame(unshifted_start)?;
        let end_frame = shift_frame(unshifted_end)?;
        if start_frame >= end_frame {
            return Err(SymbolicAudioBindingError::UnsupportedScore(
                "authorized event does not map to a non-empty audio interval",
            ));
        }
        intervals.push(FrameInterval { start_frame, end_frame });
    }

    intervals.sort_by_key(|interval| interval.start_frame);
    let mut merged: Vec<FrameInterval> = Vec::with_capacity(intervals.len());
    for interval in intervals {
        if let Some(previous) = merged.last_mut() {
            if interval.start_frame <= previous.end_frame {
                previous.end_frame = previous.end_frame.max(interval.end_frame);
                continue;
            }
        }
        merged.push(interval);
    }
    Ok(merged)
}

/// Derive the onset-evidence mask independently from the acoustic edit mask.
/// A pitch or transpose edit changes spectral content, but the edited notes'
/// attacks are still expected to occur. Do not mask them out. ScaleOnsets is
/// fail-closed until the onset comparator supports a request-conditioned time
/// warp rather than a naive source/candidate onset equality check.
fn derive_authorized_onset_mask(
    request: symthaea_music_theory::edit_locality::EditRequest,
) -> Result<Vec<FrameInterval>, SymbolicAudioBindingError> {
    match request.operation {
        symthaea_music_theory::edit_locality::EditOperation::ScaleOnsets { .. } => {
            Err(SymbolicAudioBindingError::UnsupportedScore(
                "ScaleOnsets requires request-conditioned onset alignment and is not yet supported by this audio gate",
            ))
        }
        symthaea_music_theory::edit_locality::EditOperation::Transpose { .. }
        | symthaea_music_theory::edit_locality::EditOperation::ScaleDurations { .. }
        | symthaea_music_theory::edit_locality::EditOperation::PitchAt { .. }
        | symthaea_music_theory::edit_locality::EditOperation::DurationAt { .. } => Ok(Vec::new()),
    }
}

fn full_score_binding_receipt_sha256(
    binding: &FullScoreRenderBinding,
) -> Result<String, SymbolicAudioBindingError> {
    let mut unsealed = binding.clone();
    unsealed.receipt_sha256.clear();
    let bytes = serde_json::to_vec(&unsealed)
        .map_err(|_| SymbolicAudioBindingError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous-dynamics-full-score-render-binding-v1\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Create a binding directly from Muse's full Score type.
///
/// The supported subset is one non-overlapping part/role, identical key/tempo/
/// meter, and unchanged part/role/velocity/emphasis/section-intensity metadata.
/// The inner symbolic contract decides which pitch/onset/duration properties
/// may change. This wrapper hashes every serialized score field.
pub fn create_full_score_render_binding(
    source_score: &Score,
    target_score: &Score,
    symbolic_evidence: &EditLocalityEvidence,
    source_audio: AudioView<'_>,
    target_audio: AudioView<'_>,
    acoustic_evidence: &AudioPreservationReport,
    onset_evidence: &OnsetPreservationReport,
    renderer_manifest: &RendererManifest,
    preservation_profile: &AudioPreservationProfile,
) -> Result<FullScoreRenderBinding, SymbolicAudioBindingError> {
    require_only_locality_dimensions_changed(source_score, target_score)?;
    let preservation_profile_sha256 =
        require_profile_matches_reports(preservation_profile, acoustic_evidence, onset_evidence)?;
    let source_events = validate_score_projection(source_score)?;
    let target_events = validate_score_projection(target_score)?;
    if !verify_edit_locality_evidence(&source_events, &target_events, symbolic_evidence)? {
        return Err(SymbolicAudioBindingError::InvalidEvidence(
            "symbolic receipt does not replay against the supplied full Score projection",
        ));
    }
    // Validate buffers before using their dimensions in the score-to-audio map.
    audio_content_sha256(source_audio)?;
    audio_content_sha256(target_audio)?;
    if renderer_manifest.sample_rate_hz != source_audio.sample_rate_hz
        || renderer_manifest.channels != source_audio.channels
    {
        return Err(SymbolicAudioBindingError::InvalidRendererManifest(
            "manifest audio format does not match the rendered buffers",
        ));
    }
    // Validate the manifest before trusting its latency adjustment.
    renderer_manifest_sha256(renderer_manifest)?;
    let total_frames = source_audio.interleaved_samples.len() / source_audio.channels;
    let expected_acoustic_mask = derive_authorized_audio_mask(
        &source_events,
        &target_events,
        symbolic_evidence.request,
        source_score.tempo_bpm,
        source_audio.sample_rate_hz,
        total_frames,
        renderer_manifest.render_latency_frames,
    )?;
    let expected_onset_mask = derive_authorized_onset_mask(symbolic_evidence.request)?;
    if acoustic_evidence.authorized_edits != expected_acoustic_mask {
        return Err(SymbolicAudioBindingError::InvalidEvidence(
            "acoustic edit mask does not match the requested symbolic edit and score timing",
        ));
    }
    if onset_evidence.authorized_edits != expected_onset_mask {
        return Err(SymbolicAudioBindingError::InvalidEvidence(
            "onset mask must preserve edited-note attacks; only request-conditioned operations may authorize onset exclusions",
        ));
    }
    // This audio gate currently makes a calibrated claim only for pitch edits
    // and transposition: spectral content may change, but the edit's amplitude
    // envelope and all note attacks must remain. Duration/onset edits require a
    // request-conditioned temporal oracle and fail closed until implemented.
    if !matches!(
        symbolic_evidence.request.operation,
        symthaea_music_theory::edit_locality::EditOperation::Transpose { .. }
            | symthaea_music_theory::edit_locality::EditOperation::PitchAt { .. }
    ) {
        return Err(SymbolicAudioBindingError::UnsupportedScore(
            "full-score audio binding currently supports pitch edits/transposition only",
        ));
    }
    let expected_region_measurements =
        expected_acoustic_mask.len().saturating_mul(source_audio.channels);
    if acoustic_evidence.authorized_region_rms.len() != expected_region_measurements {
        return Err(SymbolicAudioBindingError::InvalidEvidence(
            "authorized-region RMS evidence does not cover every edited region and channel",
        ));
    }

    let mut projected_binding = create_symbolic_audio_render_binding(
        &source_events,
        &target_events,
        symbolic_evidence,
        source_audio,
        target_audio,
        acoustic_evidence,
        onset_evidence,
        renderer_manifest,
    )?;

    let envelope_violations = acoustic_evidence
        .authorized_region_rms
        .iter()
        .filter(|measurement| {
            !measurement.rms_delta_db.is_finite()
                || measurement.rms_delta_db > acoustic_evidence.config.rms_delta_db_tolerance
        })
        .count();
    if envelope_violations > 0 {
        projected_binding.overall_verdict = PreservationVerdict::Fail;
        projected_binding.reasons.push(format!(
            "{envelope_violations} edited-region/channel RMS envelopes exceeded the preserved-energy threshold"
        ));
        projected_binding.binding_receipt_sha256.clear();
        projected_binding.binding_receipt_sha256 =
            binding_receipt_sha256(&projected_binding)?;
    }

    let mut binding = FullScoreRenderBinding {
        binding_version: format!("{SYMBOLIC_AUDIO_BINDING_VERSION}+full-score-v0.1"),
        source_score_content_sha256: full_score_sha256(source_score)?,
        target_score_content_sha256: full_score_sha256(target_score)?,
        preservation_profile_sha256,
        projected_binding,
        receipt_sha256: String::new(),
    };
    binding.receipt_sha256 = full_score_binding_receipt_sha256(&binding)?;
    Ok(binding)
}

/// Verify a full Score envelope by recomputing the narrow projection and every
/// child receipt from the supplied source/target Scores and audio buffers.
pub fn verify_full_score_render_binding(
    source_score: &Score,
    target_score: &Score,
    symbolic_evidence: &EditLocalityEvidence,
    source_audio: AudioView<'_>,
    target_audio: AudioView<'_>,
    acoustic_evidence: &AudioPreservationReport,
    onset_evidence: &OnsetPreservationReport,
    renderer_manifest: &RendererManifest,
    preservation_profile: &AudioPreservationProfile,
    binding: &FullScoreRenderBinding,
) -> Result<bool, SymbolicAudioBindingError> {
    if binding.receipt_sha256.is_empty()
        || full_score_binding_receipt_sha256(binding)? != binding.receipt_sha256
    {
        return Ok(false);
    }
    let expected = create_full_score_render_binding(
        source_score,
        target_score,
        symbolic_evidence,
        source_audio,
        target_audio,
        acoustic_evidence,
        onset_evidence,
        renderer_manifest,
        preservation_profile,
    )?;
    Ok(expected == *binding)
}


/// Contract-bound full-score receipt. It preserves the existing audio binding
/// while committing to the frozen intent/obligation contract that was checked
/// against the exact source Score, target Score, profile, and renderer manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContractBoundFullScoreRenderBinding {
    pub binding_version: String,
    pub contract_sha256: String,
    pub instruction_sha256: String,
    pub inner: FullScoreRenderBinding,
    /// SHA-256 over this envelope with this field blank.
    pub receipt_sha256: String,
}

fn contract_bound_receipt_sha256(
    binding: &ContractBoundFullScoreRenderBinding,
) -> Result<String, SymbolicAudioBindingError> {
    let mut unsealed = binding.clone();
    unsealed.receipt_sha256.clear();
    let bytes = serde_json::to_vec(&unsealed)
        .map_err(|_| SymbolicAudioBindingError::SerializationFailure)?;
    let mut hasher = Sha256::new();
    hasher.update(b"luminous-dynamics-contract-bound-full-score-binding-v1\0");
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn profile_tolerance_microunits(
    value: f32,
) -> Result<u64, SymbolicAudioBindingError> {
    if !value.is_finite() || value < 0.0 {
        return Err(SymbolicAudioBindingError::InvalidEvidence(
            "preservation profile contains a non-finite or negative tolerance",
        ));
    }
    let scaled = (value as f64 * 1_000_000.0).round();
    if !scaled.is_finite() || scaled > u64::MAX as f64 {
        return Err(SymbolicAudioBindingError::InvalidEvidence(
            "preservation profile tolerance cannot be represented as integer microunits",
        ));
    }
    Ok(scaled as u64)
}

fn require_contract_profile_tolerance(
    contract: &FrozenEditContract,
    attribute: EditAttribute,
    comparator_id: &str,
    expected_unit: &str,
    expected_value_microunits: u64,
) -> Result<(), SymbolicAudioBindingError> {
    let matching: Vec<&EditObligation> = contract.obligations.iter()
        .filter(|obligation| obligation.attribute == attribute)
        .filter(|obligation| matches!(
            &obligation.requirement,
            ObligationRequirement::PreserveWithin { comparator_id: actual, .. }
                if actual == comparator_id
        ))
        .collect();
    if matching.len() != 1 {
        return Err(SymbolicAudioBindingError::InvalidEvidence(
            "contract must contain exactly one registered comparator for each profile tolerance",
        ));
    }
    match &matching[0].requirement {
        ObligationRequirement::PreserveWithin { comparator_id: actual, tolerance }
            if matching[0].critical
                && actual.as_str() == comparator_id
                && tolerance.unit == expected_unit
                && tolerance.value_microunits == expected_value_microunits => Ok(()),
        _ => Err(SymbolicAudioBindingError::InvalidEvidence(
            "contract tolerance does not exactly match the trusted preservation profile",
        )),
    }
}

/// Validate a frozen contract against the actual symbolic inputs before
/// rendering. This checks the full Score digests, raw-instruction digest,
/// approved-profile identity/digest, renderer-manifest digest, operation,
/// target event index, and requested pitch/delta.
///
/// The profile still needs to have been resolved from a trusted registry by
/// the caller. A matching caller-supplied digest is not profile approval and
/// a renderer manifest is not causal execution attestation by itself.
pub fn validate_frozen_edit_contract_for_rendering(
    contract: &FrozenEditContract,
    instruction_sha256: &str,
    source_score: &Score,
    target_score: &Score,
    symbolic_evidence: &EditLocalityEvidence,
    renderer_manifest: &RendererManifest,
    preservation_profile: &AudioPreservationProfile,
) -> Result<(), SymbolicAudioBindingError> {
    contract.validate_supported_operation().map_err(|_| {
        SymbolicAudioBindingError::InvalidEvidence(
            "frozen edit contract failed operation-specific obligation validation",
        )
    })?;

    if contract.instruction_sha256 != instruction_sha256 {
        return Err(SymbolicAudioBindingError::InvalidEvidence(
            "frozen contract is not bound to the exact user-instruction digest",
        ));
    }
    if contract.source_score_sha256 != full_score_sha256(source_score)?
        || contract.target_score_sha256 != full_score_sha256(target_score)?
    {
        return Err(SymbolicAudioBindingError::InvalidEvidence(
            "frozen contract source/target Score digests do not match the supplied Scores",
        ));
    }

    let profile_digest = audio_preservation_profile_sha256(preservation_profile)?;
    if contract.profile_id != preservation_profile.profile_id
        || contract.profile_sha256 != profile_digest
    {
        return Err(SymbolicAudioBindingError::InvalidEvidence(
            "frozen contract does not match the selected preservation profile",
        ));
    }

    // The operation policy requires the comparators and scopes; these checks
    // prevent a validly resealed contract from weakening their thresholds.
    let profile_onset_tolerance = u64::from(preservation_profile.onset_config.onset_time_tolerance_ms)
        .checked_mul(1_000)
        .ok_or(SymbolicAudioBindingError::InvalidEvidence(
            "onset tolerance cannot be represented in microseconds",
        ))?;
    require_contract_profile_tolerance(
        contract, EditAttribute::Onset, "onset-preservation-v1", "seconds",
        profile_onset_tolerance,
    )?;
    require_contract_profile_tolerance(
        contract, EditAttribute::Duration, "score-duration-exact-v1", "beats", 0,
    )?;
    let acoustic_rms_tolerance =
        profile_tolerance_microunits(preservation_profile.acoustic_config.rms_delta_db_tolerance)?;
    require_contract_profile_tolerance(
        contract, EditAttribute::RmsEnvelope, "rms-envelope-delta-v1", "db",
        acoustic_rms_tolerance,
    )?;
    require_contract_profile_tolerance(
        contract, EditAttribute::NeighboringContent, "protected-region-rms-v1", "db",
        acoustic_rms_tolerance,
    )?;
    let spectral_tolerance =
        profile_tolerance_microunits(preservation_profile.acoustic_config.spectral_distance_tolerance)?;
    require_contract_profile_tolerance(
        contract, EditAttribute::NeighboringContent, "protected-region-spectral-v1",
        "normalized_distance", spectral_tolerance,
    )?;

    let renderer_digest = renderer_manifest_sha256(renderer_manifest)?;
    let renderer_obligation = contract.obligations.iter()
        .find(|obligation| obligation.attribute == EditAttribute::RendererConfig)
        .ok_or(SymbolicAudioBindingError::InvalidEvidence(
            "frozen contract has no renderer configuration obligation",
        ))?;
    match &renderer_obligation.requirement {
        ObligationRequirement::BindDigest { expected_sha256 }
            if expected_sha256 == &renderer_digest => {}
        _ => return Err(SymbolicAudioBindingError::InvalidEvidence(
            "frozen contract renderer digest does not match the supplied renderer manifest",
        )),
    }

    require_only_locality_dimensions_changed(source_score, target_score)?;
    let source_events = validate_score_projection(source_score)?;
    let target_events = validate_score_projection(target_score)?;
    if !verify_edit_locality_evidence(&source_events, &target_events, symbolic_evidence)?
        || !symbolic_evidence.passed
    {
        return Err(SymbolicAudioBindingError::InvalidEvidence(
            "symbolic evidence does not pass against the exact source/target Score projection",
        ));
    }

    match (contract.validate_supported_operation().map_err(|_| {
        SymbolicAudioBindingError::InvalidEvidence(
            "frozen edit contract failed operation-specific obligation validation",
        )
    })?, symbolic_evidence.request.operation) {
        (
            SupportedEditOperation::PitchAt,
            symthaea_music_theory::edit_locality::EditOperation::PitchAt { index, semitones },
        ) => {
            if index >= source_events.len() || index >= target_events.len() {
                return Err(SymbolicAudioBindingError::InvalidEvidence(
                    "requested pitch event index is outside the bound Score",
                ));
            }
            let expected_event_id = format!("score_event_index:{index}");
            let pitch = contract.obligations.iter()
                .find(|obligation| obligation.attribute == EditAttribute::Pitch)
                .ok_or(SymbolicAudioBindingError::InvalidEvidence(
                    "frozen contract has no target-pitch obligation",
                ))?;
            match (&pitch.scope, &pitch.requirement) {
                (
                    ObligationScope::Event { event_id },
                    ObligationRequirement::ChangeTo {
                        expected: AttributeValue::Numeric { value_microunits, unit },
                    },
                ) if event_id == &expected_event_id
                    && unit == "midi_note"
                    && *value_microunits
                        == i64::from(target_events[index].pitch_midi) * 1_000_000
                    && target_events[index].pitch_midi - source_events[index].pitch_midi == semitones => {}
                _ => return Err(SymbolicAudioBindingError::InvalidEvidence(
                    "target-pitch obligation does not match the requested score event and exact target pitch",
                )),
            }
        }
        (
            SupportedEditOperation::Transpose,
            symthaea_music_theory::edit_locality::EditOperation::Transpose { semitones },
        ) => {
            let pitch = contract.obligations.iter()
                .find(|obligation| obligation.attribute == EditAttribute::Pitch)
                .ok_or(SymbolicAudioBindingError::InvalidEvidence(
                    "frozen contract has no transposition-delta obligation",
                ))?;
            match &pitch.requirement {
                ObligationRequirement::ChangeBy {
                    delta: AttributeValue::Numeric { value_microunits, unit },
                } if unit == "semitone"
                    && *value_microunits == i64::from(semitones) * 1_000_000 => {}
                _ => return Err(SymbolicAudioBindingError::InvalidEvidence(
                    "transposition obligation does not match the exact requested semitone delta",
                )),
            }
        }
        _ => return Err(SymbolicAudioBindingError::InvalidEvidence(
            "frozen contract operation does not match the independently verified symbolic request",
        )),
    }

    Ok(())
}

/// Create a full-score audio binding only after validating the frozen
/// operation/obligation contract against the actual symbolic inputs.
#[allow(clippy::too_many_arguments)]
pub fn create_contract_bound_full_score_render_binding(
    contract: &FrozenEditContract,
    instruction_sha256: &str,
    source_score: &Score,
    target_score: &Score,
    symbolic_evidence: &EditLocalityEvidence,
    source_audio: AudioView<'_>,
    target_audio: AudioView<'_>,
    acoustic_evidence: &AudioPreservationReport,
    onset_evidence: &OnsetPreservationReport,
    renderer_manifest: &RendererManifest,
    preservation_profile: &AudioPreservationProfile,
) -> Result<ContractBoundFullScoreRenderBinding, SymbolicAudioBindingError> {
    validate_frozen_edit_contract_for_rendering(
        contract,
        instruction_sha256,
        source_score,
        target_score,
        symbolic_evidence,
        renderer_manifest,
        preservation_profile,
    )?;
    let inner = create_full_score_render_binding(
        source_score,
        target_score,
        symbolic_evidence,
        source_audio,
        target_audio,
        acoustic_evidence,
        onset_evidence,
        renderer_manifest,
        preservation_profile,
    )?;
    let mut binding = ContractBoundFullScoreRenderBinding {
        binding_version: "mel-mr-eval-012-contract-bound-full-score-v0.1".to_string(),
        contract_sha256: contract.contract_sha256.clone(),
        instruction_sha256: instruction_sha256.to_string(),
        inner,
        receipt_sha256: String::new(),
    };
    binding.receipt_sha256 = contract_bound_receipt_sha256(&binding)?;
    Ok(binding)
}

/// Replay a contract-bound full-score binding from the original immutable
/// inputs. Receipt hashes are integrity checks; this function additionally
/// recomputes the contract context and inner full-score/audio evidence.
#[allow(clippy::too_many_arguments)]
pub fn verify_contract_bound_full_score_render_binding(
    contract: &FrozenEditContract,
    instruction_sha256: &str,
    source_score: &Score,
    target_score: &Score,
    symbolic_evidence: &EditLocalityEvidence,
    source_audio: AudioView<'_>,
    target_audio: AudioView<'_>,
    acoustic_evidence: &AudioPreservationReport,
    onset_evidence: &OnsetPreservationReport,
    renderer_manifest: &RendererManifest,
    preservation_profile: &AudioPreservationProfile,
    binding: &ContractBoundFullScoreRenderBinding,
) -> Result<bool, SymbolicAudioBindingError> {
    if binding.receipt_sha256.is_empty()
        || contract_bound_receipt_sha256(binding)? != binding.receipt_sha256
        || binding.contract_sha256 != contract.contract_sha256
        || binding.instruction_sha256 != instruction_sha256
    {
        return Ok(false);
    }
    let expected = create_contract_bound_full_score_render_binding(
        contract,
        instruction_sha256,
        source_score,
        target_score,
        symbolic_evidence,
        source_audio,
        target_audio,
        acoustic_evidence,
        onset_evidence,
        renderer_manifest,
        preservation_profile,
    )?;
    Ok(expected == *binding)
}

/// Recompute the full cross-modal envelope from the original inputs and require
/// exact equality, including child receipts, content digests, manifest, verdict,
/// reasons, and binding receipt. Hashes are unkeyed and do not authenticate a
/// renderer or publisher.
pub fn verify_symbolic_audio_render_binding(
    source_score: &[LocalityEvent],
    target_score: &[LocalityEvent],
    symbolic_evidence: &EditLocalityEvidence,
    source_audio: AudioView<'_>,
    target_audio: AudioView<'_>,
    acoustic_evidence: &AudioPreservationReport,
    onset_evidence: &OnsetPreservationReport,
    renderer_manifest: &RendererManifest,
    binding: &SymbolicAudioRenderBinding,
) -> Result<bool, SymbolicAudioBindingError> {
    if binding.binding_receipt_sha256.is_empty()
        || binding_receipt_sha256(binding)? != binding.binding_receipt_sha256
    {
        return Ok(false);
    }
    let expected = create_symbolic_audio_render_binding(
        source_score,
        target_score,
        symbolic_evidence,
        source_audio,
        target_audio,
        acoustic_evidence,
        onset_evidence,
        renderer_manifest,
    )?;
    Ok(expected == *binding)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_preservation::{
        FrameInterval, OnsetPreservationConfig, PreservationConfig,
        audit_onset_preservation, audit_protected_audio, compose_audio_preservation_evidence,
    };

    fn digest(label: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(label);
        format!("{:x}", hasher.finalize())
    }

    fn score_source() -> Vec<LocalityEvent> {
        vec![
            LocalityEvent { pitch_midi: 60, onset_beats: 0.0, duration_beats: 0.5 },
            LocalityEvent { pitch_midi: 62, onset_beats: 1.0, duration_beats: 0.5 },
            LocalityEvent { pitch_midi: 64, onset_beats: 2.0, duration_beats: 0.5 },
        ]
    }

    fn score_target() -> Vec<LocalityEvent> {
        let mut target = score_source();
        target[1].pitch_midi += 5;
        target
    }

    fn full_score_source() -> Score {
        let mut score = Score::new(
            symthaea_music_theory::Key::major(symthaea_music_theory::PitchClass::C),
            120.0,
            4,
        );
        for (onset_num, midi) in [(0i64, 60u8), (64, 62), (96, 64)] {
            score.push(symthaea_music_theory::ScoreNote {
                part: symthaea_music_theory::PartId(0),
                pitch: symthaea_music_theory::Pitch::from_midi(midi),
                onset: symthaea_music_theory::Duration::new(onset_num, 125),
                duration: symthaea_music_theory::Duration::new(16, 125),
                velocity: 0.7,
                role: symthaea_music_theory::VoiceRole::Melody,
                emphasis: symthaea_music_theory::Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        score
    }

    fn full_score_target() -> Score {
        let mut target = full_score_source();
        target.notes[1].pitch = symthaea_music_theory::Pitch::from_midi(67);
        target
    }

    fn full_score_symbolic_evidence(
        source: &Score,
        target: &Score,
    ) -> EditLocalityEvidence {
        let source_events = validate_score_projection(source).expect("source score projection");
        let target_events = validate_score_projection(target).expect("target score projection");
        let request = symthaea_music_theory::edit_locality::EditRequest {
            operation: symthaea_music_theory::edit_locality::EditOperation::PitchAt {
                index: 1,
                semitones: 5,
            },
        };
        symthaea_music_theory::edit_locality::verify_edit_locality(
            &source_events,
            &target_events,
            &request,
        ).expect("full-score pitch edit should produce symbolic evidence")
    }

    fn audio_pair() -> (Vec<f32>, Vec<f32>) {
        let total = 8_192usize;
        let sample_rate = 16_000.0f32;
        let mut source = vec![0.0f32; total];
        let mut target = vec![0.0f32; total];
        for (start, end) in [(1_024usize, 2_048usize), (4_096, 5_120), (6_144, 7_168)] {
            for frame in start..end {
                source[frame] = (std::f32::consts::TAU * 440.0 * frame as f32 / sample_rate).sin();
                let frequency = if (4_096..5_120).contains(&frame) { 880.0 } else { 440.0 };
                target[frame] = (std::f32::consts::TAU * frequency * frame as f32 / sample_rate).sin();
            }
        }
        (source, target)
    }

    fn audio_view(samples: &[f32]) -> AudioView<'_> {
        AudioView { interleaved_samples: samples, sample_rate_hz: 16_000, channels: 1 }
    }

    fn renderer_manifest() -> RendererManifest {
        RendererManifest {
            manifest_version: RENDERER_MANIFEST_VERSION.to_owned(),
            renderer_id: "synthetic-test-renderer".to_owned(),
            renderer_version: "1.0.0".to_owned(),
            instrument_bank_sha256: None,
            effects_config_sha256: None,
            render_config_sha256: digest(b"synthetic-render-config"),
            sample_rate_hz: 16_000,
            channels: 1,
            render_latency_frames: 0,
            latency_policy: "no-added-latency-synthetic-buffer".to_owned(),
        }
    }

    fn fixture_profile() -> AudioPreservationProfile {
        AudioPreservationProfile {
            profile_version: "audio-preservation-profile-v0.1".to_owned(),
            profile_id: "symbolic-audio-binding-test-fixture".to_owned(),
            acoustic_config: PreservationConfig {
                frame_size: 128,
                hop_size: 64,
                boundary_guard_frames: 0,
                rms_delta_db_tolerance: 0.5,
                spectral_distance_tolerance: 0.08,
                minimum_protected_windows: 4,
                calibration_id: Some("symbolic-audio-binding-test-fixture".to_owned()),
            },
            onset_config: OnsetPreservationConfig {
                frame_size: 128,
                hop_size: 32,
                novelty_threshold: 0.12,
                minimum_separation_ms: 8,
                onset_time_tolerance_ms: 4,
                boundary_guard_frames: 0,
                minimum_protected_duration_ms: 100,
                calibration_id: Some("symbolic-audio-binding-test-fixture".to_owned()),
            },
        }
    }

    fn complete_contract_for_full_score(
        instruction_sha256: &str,
        source: &Score,
        target: &Score,
        symbolic: &EditLocalityEvidence,
        manifest: &RendererManifest,
        profile: &AudioPreservationProfile,
    ) -> FrozenEditContract {
        let source_events = validate_score_projection(source).expect("source projection");
        let target_events = validate_score_projection(target).expect("target projection");
        let (operation_id, pitch_scope, pitch_requirement, rms_scope) =
            match symbolic.request.operation {
                symthaea_music_theory::edit_locality::EditOperation::PitchAt { index, semitones } => {
                    assert!(index < source_events.len());
                    assert_eq!(target_events[index].pitch_midi - source_events[index].pitch_midi, semitones);
                    (
                        "pitch-at",
                        ObligationScope::Event { event_id: format!("score_event_index:{index}") },
                        ObligationRequirement::ChangeTo {
                            expected: AttributeValue::Numeric {
                                value_microunits: i64::from(target_events[index].pitch_midi) * 1_000_000,
                                unit: "midi_note".to_string(),
                            },
                        },
                        ObligationScope::Event { event_id: format!("score_event_index:{index}") },
                    )
                }
                symthaea_music_theory::edit_locality::EditOperation::Transpose { semitones } => (
                    "transpose",
                    ObligationScope::ScoreWide,
                    ObligationRequirement::ChangeBy {
                        delta: AttributeValue::Numeric {
                            value_microunits: i64::from(semitones) * 1_000_000,
                            unit: "semitone".to_string(),
                        },
                    },
                    ObligationScope::ScoreWide,
                ),
                _ => panic!("only the currently supported pitch operations are valid in this fixture"),
            };
        let preservation = |id: &str,
                            attribute: EditAttribute,
                            scope: ObligationScope,
                            comparator_id: &str,
                            unit: &str,
                            value_microunits: u64| EditObligation {
            obligation_id: id.to_string(),
            attribute,
            scope,
            requirement: ObligationRequirement::PreserveWithin {
                comparator_id: comparator_id.to_string(),
                tolerance: QuantizedTolerance {
                    value_microunits,
                    unit: unit.to_string(),
                },
            },
            critical: true,
        };
        let renderer_digest = renderer_manifest_sha256(manifest).expect("manifest digest");
        let profile_digest = audio_preservation_profile_sha256(profile).expect("profile digest");
        FrozenEditContract::new(
            "contract-bound-test".to_string(),
            operation_id.to_string(),
            instruction_sha256.to_string(),
            full_score_sha256(source).expect("source score digest"),
            full_score_sha256(target).expect("target score digest"),
            profile.profile_id.clone(),
            profile_digest,
            vec![
                EditObligation {
                    obligation_id: "pitch-change".to_string(),
                    attribute: EditAttribute::Pitch,
                    scope: pitch_scope,
                    requirement: pitch_requirement,
                    critical: true,
                },
                preservation("onset", EditAttribute::Onset, ObligationScope::ScoreWide,
                    "onset-preservation-v1", "seconds", 10_000),
                preservation("duration", EditAttribute::Duration, ObligationScope::ScoreWide,
                    "score-duration-exact-v1", "beats", 0),
                preservation("edited-envelope", EditAttribute::RmsEnvelope, rms_scope,
                    "rms-envelope-delta-v1", "db", 1_000_000),
                preservation("protected-rms", EditAttribute::NeighboringContent,
                    ObligationScope::ProtectedRemainder, "protected-region-rms-v1", "db", 500_000),
                preservation("protected-spectrum", EditAttribute::NeighboringContent,
                    ObligationScope::ProtectedRemainder, "protected-region-spectral-v1",
                    "normalized_distance", 80_000),
                EditObligation {
                    obligation_id: "renderer-config".to_string(),
                    attribute: EditAttribute::RendererConfig,
                    scope: ObligationScope::ScoreWide,
                    requirement: ObligationRequirement::BindDigest { expected_sha256: renderer_digest },
                    critical: true,
                },
            ],
        ).seal().expect("complete contract should seal")
    }

    #[test]
    fn frozen_contract_binds_exact_instruction_scores_profile_renderer_and_target_event() {
        let source = full_score_source();
        let target = full_score_target();
        let symbolic = full_score_symbolic_evidence(&source, &target);
        let manifest = renderer_manifest();
        let profile = fixture_profile();
        let instruction = digest(b"raise the second note by five semitones");
        let contract = complete_contract_for_full_score(
            &instruction, &source, &target, &symbolic, &manifest, &profile,
        );
        assert!(validate_frozen_edit_contract_for_rendering(
            &contract, &instruction, &source, &target, &symbolic, &manifest, &profile,
        ).is_ok());
    }

    #[test]
    fn frozen_contract_rejects_wrong_target_score_hash() {
        let source = full_score_source();
        let target = full_score_target();
        let symbolic = full_score_symbolic_evidence(&source, &target);
        let manifest = renderer_manifest();
        let profile = fixture_profile();
        let instruction = digest(b"raise the second note by five semitones");
        let mut contract = complete_contract_for_full_score(
            &instruction, &source, &target, &symbolic, &manifest, &profile,
        );
        contract.target_score_sha256 = digest(b"different target score");
        let contract = contract.seal().expect("tampered contract can be resealed");
        assert!(validate_frozen_edit_contract_for_rendering(
            &contract, &instruction, &source, &target, &symbolic, &manifest, &profile,
        ).is_err());
    }

    #[test]
    fn frozen_contract_rejects_renderer_digest_substitution() {
        let source = full_score_source();
        let target = full_score_target();
        let symbolic = full_score_symbolic_evidence(&source, &target);
        let manifest = renderer_manifest();
        let profile = fixture_profile();
        let instruction = digest(b"raise the second note by five semitones");
        let mut contract = complete_contract_for_full_score(
            &instruction, &source, &target, &symbolic, &manifest, &profile,
        );
        for obligation in &mut contract.obligations {
            if obligation.attribute == EditAttribute::RendererConfig {
                obligation.requirement = ObligationRequirement::BindDigest {
                    expected_sha256: digest(b"substituted renderer"),
                };
            }
        }
        let contract = contract.seal().expect("renderer digest substitution is syntactically valid");
        assert!(validate_frozen_edit_contract_for_rendering(
            &contract, &instruction, &source, &target, &symbolic, &manifest, &profile,
        ).is_err());
    }

    #[test]
    fn frozen_contract_rejects_wrong_instruction_digest() {
        let source = full_score_source();
        let target = full_score_target();
        let symbolic = full_score_symbolic_evidence(&source, &target);
        let manifest = renderer_manifest();
        let profile = fixture_profile();
        let instruction = digest(b"raise the second note by five semitones");
        let contract = complete_contract_for_full_score(
            &instruction, &source, &target, &symbolic, &manifest, &profile,
        );
        assert!(validate_frozen_edit_contract_for_rendering(
            &contract, &digest(b"edit something else"), &source, &target, &symbolic, &manifest, &profile,
        ).is_err());
    }

    #[test]
    fn contract_bound_full_score_binding_replays_exact_contract_and_audio_evidence() {
        let source = full_score_source();
        let target = full_score_target();
        let symbolic = full_score_symbolic_evidence(&source, &target);
        let (source_audio, target_audio) = audio_pair();
        let manifest = renderer_manifest();
        let profile = fixture_profile();
        let instruction = digest(b"raise the second note by five semitones");
        let contract = complete_contract_for_full_score(
            &instruction, &source, &target, &symbolic, &manifest, &profile,
        );
        let mask = [FrameInterval { start_frame: 4_096, end_frame: 5_120 }];
        let acoustic = audit_protected_audio(
            audio_view(&source_audio),
            audio_view(&target_audio),
            &mask,
            &profile.acoustic_config,
        ).expect("acoustic evidence");
        let onset = audit_onset_preservation(
            audio_view(&source_audio),
            audio_view(&target_audio),
            &[],
            &profile.onset_config,
        ).expect("onset evidence");
        let binding = create_contract_bound_full_score_render_binding(
            &contract, &instruction, &source, &target, &symbolic,
            audio_view(&source_audio), audio_view(&target_audio), &acoustic, &onset,
            &manifest, &profile,
        ).expect("contract-bound binding should be constructed");
        assert!(verify_contract_bound_full_score_render_binding(
            &contract, &instruction, &source, &target, &symbolic,
            audio_view(&source_audio), audio_view(&target_audio), &acoustic, &onset,
            &manifest, &profile, &binding,
        ).expect("binding replay"));
        let mut tampered = binding.clone();
        tampered.contract_sha256 = digest(b"other contract");
        assert!(!verify_contract_bound_full_score_render_binding(
            &contract, &instruction, &source, &target, &symbolic,
            audio_view(&source_audio), audio_view(&target_audio), &acoustic, &onset,
            &manifest, &profile, &tampered,
        ).expect("tampered binding should return false"));
    }

    fn build_fixture() -> (
        Vec<LocalityEvent>,
        Vec<LocalityEvent>,
        EditLocalityEvidence,
        Vec<f32>,
        Vec<f32>,
        AudioPreservationReport,
        OnsetPreservationReport,
        RendererManifest,
    ) {
        let source_score = score_source();
        let target_score = score_target();
        let request = symthaea_music_theory::edit_locality::EditRequest {
            operation: symthaea_music_theory::edit_locality::EditOperation::PitchAt {
                index: 1,
                semitones: 5,
            },
        };
        let symbolic_evidence =
            symthaea_music_theory::edit_locality::verify_edit_locality(&source_score, &target_score, &request)
                .expect("synthetic symbolic edit should verify");

        let (source_audio, target_audio) = audio_pair();
        let edit_mask = [FrameInterval { start_frame: 4_096, end_frame: 5_120 }];
        let acoustic = audit_protected_audio(
            audio_view(&source_audio),
            audio_view(&target_audio),
            &edit_mask,
            &PreservationConfig {
                frame_size: 128,
                hop_size: 64,
                boundary_guard_frames: 0,
                rms_delta_db_tolerance: 0.5,
                spectral_distance_tolerance: 0.08,
                minimum_protected_windows: 4,
                calibration_id: Some("symbolic-audio-binding-test-fixture".to_owned()),
            },
        ).expect("synthetic acoustic report");
        let onset = audit_onset_preservation(
            audio_view(&source_audio),
            audio_view(&target_audio),
            &[],
            &OnsetPreservationConfig {
                frame_size: 128,
                hop_size: 32,
                novelty_threshold: 0.12,
                minimum_separation_ms: 8,
                onset_time_tolerance_ms: 4,
                boundary_guard_frames: 0,
                minimum_protected_duration_ms: 100,
                calibration_id: Some("symbolic-audio-binding-test-fixture".to_owned()),
            },
        ).expect("synthetic onset report");
        (
            source_score,
            target_score,
            symbolic_evidence,
            source_audio,
            target_audio,
            acoustic,
            onset,
            renderer_manifest(),
        )
    }

    #[test]
    fn binding_verifies_when_symbolic_audio_and_manifest_inputs_match() {
        let (source_score, target_score, symbolic, source_audio, target_audio, acoustic, onset, manifest) =
            build_fixture();
        let binding = create_symbolic_audio_render_binding(
            &source_score,
            &target_score,
            &symbolic,
            audio_view(&source_audio),
            audio_view(&target_audio),
            &acoustic, &onset,
            &manifest,
        ).expect("consistent evidence should bind");
        assert_eq!(binding.overall_verdict, PreservationVerdict::Pass);
        assert!(verify_symbolic_audio_render_binding(
            &source_score, &target_score, &symbolic,
            audio_view(&source_audio), audio_view(&target_audio), &acoustic, &onset, &manifest, &binding
        ).expect("binding should replay"));
    }

    #[test]
    fn binding_rejects_swapped_target_audio() {
        let (source_score, target_score, symbolic, source_audio, target_audio, acoustic, onset, manifest) =
            build_fixture();
        let binding = create_symbolic_audio_render_binding(
            &source_score, &target_score, &symbolic,
            audio_view(&source_audio), audio_view(&target_audio), &acoustic, &onset, &manifest
        ).expect("fixture binding");
        let altered = audio_pair().0;
        let result = verify_symbolic_audio_render_binding(
            &source_score, &target_score, &symbolic,
            audio_view(&source_audio), audio_view(&altered), &acoustic, &onset, &manifest, &binding
        );
        assert!(result.is_err(), "audio receipt must not be reused with another output buffer");
    }

    #[test]
    fn binding_rejects_modified_score_and_evidence() {
        let (source_score, target_score, symbolic, source_audio, target_audio, acoustic, onset, manifest) =
            build_fixture();
        let binding = create_symbolic_audio_render_binding(
            &source_score, &target_score, &symbolic,
            audio_view(&source_audio), audio_view(&target_audio), &acoustic, &onset, &manifest
        ).expect("fixture binding");
        let mut altered_target = target_score.clone();
        altered_target[0].pitch_midi += 1;
        let result = verify_symbolic_audio_render_binding(
            &source_score, &altered_target, &symbolic,
            audio_view(&source_audio), audio_view(&target_audio), &acoustic, &onset, &manifest, &binding
        );
        assert!(result.is_err(), "changed score must invalidate the symbolic receipt");
        let mut tampered_binding = binding.clone();
        tampered_binding.renderer_manifest.renderer_version = "2.0.0".to_owned();
        assert!(!verify_symbolic_audio_render_binding(
            &source_score, &target_score, &symbolic,
            audio_view(&source_audio), audio_view(&target_audio), &acoustic, &onset, &manifest, &tampered_binding
        ).expect("tampered binding should be rejected as a verdict"));
    }

    #[test]
    fn binding_rejects_tampered_audio_child_receipt() {
        let (source_score, target_score, symbolic, source_audio, target_audio, mut acoustic, onset, manifest) =
            build_fixture();
        let binding_attempt = create_symbolic_audio_render_binding(
            &source_score, &target_score, &symbolic,
            audio_view(&source_audio), audio_view(&target_audio), &acoustic, &onset, &manifest
        ).expect("fixture binding should be constructible before tampering");
        acoustic.config.spectral_distance_tolerance = 0.99;
        let result = verify_symbolic_audio_render_binding(
            &source_score, &target_score, &symbolic,
            audio_view(&source_audio), audio_view(&target_audio), &acoustic, &onset, &manifest, &binding_attempt
        );
        assert!(result.is_err(), "tampered acoustic child receipt must be rejected");
    }

    #[test]
    fn symbolic_failure_dominates_passing_audio_evidence() {
        let (source_score, target_score, _symbolic, source_audio, target_audio, acoustic, onset, manifest) =
            build_fixture();
        let no_change_request = symthaea_music_theory::edit_locality::EditRequest {
            operation: symthaea_music_theory::edit_locality::EditOperation::Transpose { semitones: 5 },
        };
        let failed_symbolic = symthaea_music_theory::edit_locality::verify_edit_locality(
            &source_score, &source_score, &no_change_request
        ).expect("well-formed but unsuccessful symbolic request still has evidence");
        assert!(!failed_symbolic.passed);
        let binding = create_symbolic_audio_render_binding(
            &source_score, &source_score, &failed_symbolic,
            audio_view(&source_audio), audio_view(&target_audio), &acoustic, &onset, &manifest
        ).expect("a valid failed result should still be reportable");
        assert_eq!(binding.overall_verdict, PreservationVerdict::Fail);
    }

    #[test]
    fn full_score_binding_verifies_full_serialized_scores() {
        let (_source_events, _target_events, _symbolic, source_audio, target_audio, acoustic, onset, manifest) =
            build_fixture();
        let source_score = full_score_source();
        let target_score = full_score_target();
        let symbolic = full_score_symbolic_evidence(&source_score, &target_score);
        let binding = create_full_score_render_binding(
            &source_score, &target_score, &symbolic,
            audio_view(&source_audio), audio_view(&target_audio),
            &acoustic, &onset, &manifest,
            &fixture_profile(),
        ).expect("supported full-score edit should bind");
        assert_eq!(binding.projected_binding.overall_verdict, PreservationVerdict::Pass);
        assert_ne!(
            binding.source_score_content_sha256,
            binding.target_score_content_sha256
        );
        assert!(verify_full_score_render_binding(
            &source_score, &target_score, &symbolic,
            audio_view(&source_audio), audio_view(&target_audio),
            &acoustic, &onset, &manifest,
            &fixture_profile(), &binding,
        ).expect("full score binding should replay"));

        let mut tampered = binding.clone();
        tampered.target_score_content_sha256 = tampered.source_score_content_sha256.clone();
        assert!(!verify_full_score_render_binding(
            &source_score, &target_score, &symbolic,
            audio_view(&source_audio), audio_view(&target_audio),
            &acoustic, &onset, &manifest,
            &fixture_profile(), &tampered,
        ).expect("modified full-score digest should be rejected as a verdict"));
    }


    #[test]
    fn full_score_binding_rejects_removed_edited_note_inside_authorized_mask() {
        let (_source_events, _target_events, _symbolic, source_audio, mut target_audio, _acoustic, _onset, manifest) =
            build_fixture();
        let source_score = full_score_source();
        let target_score = full_score_target();
        let symbolic = full_score_symbolic_evidence(&source_score, &target_score);
        target_audio[4096..5120].fill(0.0);
        let mask = [FrameInterval { start_frame: 4096, end_frame: 5120 }];
        let acoustic = audit_protected_audio(
            audio_view(&source_audio), audio_view(&target_audio), &mask,
            &PreservationConfig {
                frame_size: 128,
                hop_size: 64,
                boundary_guard_frames: 0,
                rms_delta_db_tolerance: 0.5,
                spectral_distance_tolerance: 0.08,
                minimum_protected_windows: 4,
                calibration_id: Some("symbolic-audio-binding-test-fixture".to_owned()),
            },
        ).expect("outside-mask acoustic checks alone can pass");
        let onset = audit_onset_preservation(
            audio_view(&source_audio), audio_view(&target_audio), &[],
            &OnsetPreservationConfig {
                frame_size: 128,
                hop_size: 32,
                novelty_threshold: 0.12,
                minimum_separation_ms: 8,
                onset_time_tolerance_ms: 4,
                boundary_guard_frames: 0,
                minimum_protected_duration_ms: 100,
                calibration_id: Some("symbolic-audio-binding-test-fixture".to_owned()),
            },
        ).expect("unmasked onset channel reports the missing attack");
        assert!(
            acoustic.authorized_region_rms.iter().any(|measurement|
                measurement.rms_delta_db > acoustic.config.rms_delta_db_tolerance
            ),
            "the edited-region envelope must detect the removed note"
        );
        let result = create_full_score_render_binding(
            &source_score, &target_score, &symbolic,
            audio_view(&source_audio), audio_view(&target_audio),
            &acoustic, &onset, &manifest,
            &fixture_profile(),
        ).expect("a failed edit result should be represented by a fail verdict");
        assert_eq!(result.projected_binding.overall_verdict, PreservationVerdict::Fail);
        assert!(result.projected_binding.reasons.iter().any(|reason| reason.contains("RMS envelopes")));
    }

    #[test]
    fn full_score_adapter_rejects_unverified_note_metadata_changes() {
        let (_source_events, _target_events, _symbolic, source_audio, target_audio, acoustic, onset, manifest) =
            build_fixture();
        let source_score = full_score_source();
        let mut target_score = full_score_target();
        let symbolic = full_score_symbolic_evidence(&source_score, &target_score);
        target_score.notes[0].velocity = 0.9;
        let result = create_full_score_render_binding(
            &source_score, &target_score, &symbolic,
            audio_view(&source_audio), audio_view(&target_audio),
            &acoustic, &onset, &manifest,
            &fixture_profile(),
        );
        assert!(matches!(
            result,
            Err(SymbolicAudioBindingError::UnsupportedScore(_))
        ));
    }

    #[test]
    fn full_score_adapter_rejects_simultaneous_polyphonic_events() {
        let (_source_events, _target_events, _symbolic, source_audio, target_audio, acoustic, onset, manifest) =
            build_fixture();
        let mut source_score = full_score_source();
        let mut extra_note = source_score.notes[0].clone();
        extra_note.pitch = symthaea_music_theory::Pitch::from_midi(67);
        source_score.push(extra_note);
        let mut target_score = full_score_target();
        let mut target_extra = target_score.notes[0].clone();
        target_extra.pitch = symthaea_music_theory::Pitch::from_midi(67);
        target_score.push(target_extra);
        let symbolic = full_score_symbolic_evidence(&source_score, &target_score);
        let result = create_full_score_render_binding(
            &source_score, &target_score, &symbolic,
            audio_view(&source_audio), audio_view(&target_audio),
            &acoustic, &onset, &manifest,
            &fixture_profile(),
        );
        assert!(matches!(
            result,
            Err(SymbolicAudioBindingError::UnsupportedScore(_))
        ));
    }

    #[test]
    fn renderer_latency_is_applied_when_deriving_the_authorized_mask() {
        let (_source_events, _target_events, _symbolic, source_audio, target_audio, acoustic, onset, mut manifest) =
            build_fixture();
        let source_score = full_score_source();
        let target_score = full_score_target();
        let symbolic = full_score_symbolic_evidence(&source_score, &target_score);
        manifest.render_latency_frames = 64;

        // Child reports still use the unshifted interval; the full-score adapter
        // must not silently accept it for a manifest that declares +64 frames.
        let result = create_full_score_render_binding(
            &source_score, &target_score, &symbolic,
            audio_view(&source_audio), audio_view(&target_audio),
            &acoustic, &onset, &manifest,
            &fixture_profile(),
        );
        assert!(matches!(
            result,
            Err(SymbolicAudioBindingError::InvalidEvidence(_))
        ));
    }

    #[test]
    fn full_score_adapter_rejects_broadened_audio_authorization_mask() {
        let (_source_events, _target_events, _symbolic, source_audio, target_audio, _acoustic, _onset, manifest) =
            build_fixture();
        let source_score = full_score_source();
        let target_score = full_score_target();
        let symbolic = full_score_symbolic_evidence(&source_score, &target_score);
        let overbroad_mask = [
            FrameInterval { start_frame: 0, end_frame: 1024 },
            FrameInterval { start_frame: 4096, end_frame: 5120 },
        ];
        let acoustic = audit_protected_audio(
            audio_view(&source_audio), audio_view(&target_audio), &overbroad_mask,
            &PreservationConfig {
                frame_size: 128,
                hop_size: 64,
                boundary_guard_frames: 0,
                rms_delta_db_tolerance: 0.5,
                spectral_distance_tolerance: 0.08,
                minimum_protected_windows: 4,
                calibration_id: Some("symbolic-audio-binding-test-fixture".to_owned()),
            },
        ).expect("overbroad acoustic report is internally valid");
        let onset = audit_onset_preservation(
            audio_view(&source_audio), audio_view(&target_audio), &overbroad_mask,
            &OnsetPreservationConfig {
                frame_size: 128,
                hop_size: 32,
                novelty_threshold: 0.12,
                minimum_separation_ms: 8,
                onset_time_tolerance_ms: 4,
                boundary_guard_frames: 0,
                minimum_protected_duration_ms: 100,
                calibration_id: Some("symbolic-audio-binding-test-fixture".to_owned()),
            },
        ).expect("overbroad onset report is internally valid");
        let result = create_full_score_render_binding(
            &source_score, &target_score, &symbolic,
            audio_view(&source_audio), audio_view(&target_audio),
            &acoustic, &onset, &manifest,
            &fixture_profile(),
        );
        assert!(matches!(
            result,
            Err(SymbolicAudioBindingError::InvalidEvidence(_))
        ));
    }

    #[test]
    fn binding_rejects_audio_manifest_format_mismatch() {
        let (source_score, target_score, symbolic, source_audio, target_audio, acoustic, onset, mut manifest) =
            build_fixture();
        manifest.sample_rate_hz = 48_000;
        let result = create_symbolic_audio_render_binding(
            &source_score, &target_score, &symbolic,
            audio_view(&source_audio), audio_view(&target_audio), &acoustic, &onset, &manifest
        );
        assert!(matches!(result, Err(SymbolicAudioBindingError::InvalidRendererManifest(_))));
    }
    #[test]
    fn frozen_profile_rejects_child_config_drift() {
        let (_source_events, _target_events, _symbolic, _source_audio, _target_audio, acoustic, onset, _manifest) =
            build_fixture();
        let profile = fixture_profile();
        let mut changed_profile = profile.clone();
        changed_profile.acoustic_config.rms_delta_db_tolerance = 1.0;
        assert!(require_profile_matches_reports(&changed_profile, &acoustic, &onset).is_err());
        assert!(require_profile_matches_reports(&profile, &acoustic, &onset).is_ok());
    }


}
