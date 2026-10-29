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
    EditLocalityError, EditLocalityEvidence, LocalityEvent, verify_edit_locality_evidence,
};

use crate::audio_preservation::{
    AudioAuditInputError, AudioPreservationReport, AudioView, OnsetPreservationReport,
    PreservationVerdict, audio_content_sha256, compose_audio_preservation_evidence,
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
            latency_policy: "no-added-latency-synthetic-buffer".to_owned(),
        }
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
            &edit_mask,
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
}
