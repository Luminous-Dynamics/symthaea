// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact native StereoF32 evidence primitive for the ProgSuite contextual
//! harmony survival protocol.
//!
//! This module does not execute the 64-subject lockbox. It defines the pure
//! measurement theorem the future runner must use once exact paired native
//! renders exist.

use super::prog_suite_contextual_harmony_audio_protocol::{
    PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_REPEATED_RENDERS_PER_ARM,
    PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_SURVIVAL_PROTOCOL_VERSION,
    ProgSuiteAudioSurvivalOutcomeV1,
};
use super::sha256_hex;
use serde::{Deserialize, Serialize};

pub const PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_EVIDENCE_VERSION: &str =
    "melothaea-prog-suite-contextual-harmony-audio-evidence-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteNativeAudioSurvivalEvidenceNonClaimV1 {
    PackageVersionDoesNotEstablishSourceRevision,
    WaveformDifferenceDoesNotEstablishAudibility,
    WaveformDifferenceDoesNotEstablishAcousticOnsetDifference,
    WaveformDifferenceDoesNotEstablishSpectralSalience,
    WaveformDifferenceDoesNotEstablishListenerPreference,
    WaveformDifferenceDoesNotEstablishArtisticQuality,
    WholeWorkDifferenceDoesNotLocalizeTheEffect,
    OneRendererDoesNotEstablishRendererGeneralization,
    LockboxPopulationDoesNotEstablishUniversalMusicGeneralization,
    FrameCountsDoNotEstablishStatisticalIndependence,
    EvidenceDoesNotGrantProductAuthority,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteNativeAudioIdentityV1 {
    pub frame_count: usize,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteNativeAudioDifferenceV1 {
    pub aligned_frame_count: usize,
    pub changed_aligned_frame_count: usize,
    pub mean_absolute_sample_delta: f64,
    pub rms_sample_delta: f64,
    pub peak_absolute_sample_delta: f64,
    pub unmatched_source_tail_frames: usize,
    pub unmatched_contextual_tail_frames: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteNativeRenderRepeatReceiptV1 {
    pub required_repeats_per_arm: u8,
    pub source_repeat_bit_exact: bool,
    pub contextual_repeat_bit_exact: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteNativeAudioSurvivalEvidenceV1 {
    pub version: String,
    pub protocol_version: String,
    pub subject_id: String,
    pub motif_id: String,
    pub plan_seed: u64,
    pub intent_seed: u64,
    pub source_score_sha256: String,
    pub contextual_score_sha256: String,
    pub symbolic_score_difference_present: bool,
    pub source_audio: ProgSuiteNativeAudioIdentityV1,
    pub contextual_audio: ProgSuiteNativeAudioIdentityV1,
    pub difference: ProgSuiteNativeAudioDifferenceV1,
    pub repeat_receipt: ProgSuiteNativeRenderRepeatReceiptV1,
    pub outcome: ProgSuiteAudioSurvivalOutcomeV1,
    pub nonclaims: Vec<ProgSuiteNativeAudioSurvivalEvidenceNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteNativeAudioSurvivalEvidenceErrorV1 {
    EmptySubjectId,
    EmptyMotifId,
    SeedBindingMismatch { plan_seed: u64, intent_seed: u64 },
    InvalidSourceScoreSha256,
    InvalidContextualScoreSha256,
    EmptySourceAudio,
    EmptyContextualAudio,
    NonFiniteSourceSample { frame_index: usize, channel_index: usize },
    NonFiniteContextualSample { frame_index: usize, channel_index: usize },
    SourceRepeatMismatch,
    ContextualRepeatMismatch,
    AudioChangedForIdenticalScoreIdentity,
    EvidenceMismatch,
}

#[allow(clippy::too_many_arguments)]
pub fn measure_prog_suite_native_audio_survival(
    subject_id: &str,
    motif_id: &str,
    plan_seed: u64,
    intent_seed: u64,
    source_score_sha256: &str,
    contextual_score_sha256: &str,
    source_render_a: &[[f32; 2]],
    source_render_b: &[[f32; 2]],
    contextual_render_a: &[[f32; 2]],
    contextual_render_b: &[[f32; 2]],
) -> Result<ProgSuiteNativeAudioSurvivalEvidenceV1, ProgSuiteNativeAudioSurvivalEvidenceErrorV1> {
    validate_identity_inputs(
        subject_id,
        motif_id,
        plan_seed,
        intent_seed,
        source_score_sha256,
        contextual_score_sha256,
    )?;
    validate_audio(source_render_a, true)?;
    validate_audio(source_render_b, true)?;
    validate_audio(contextual_render_a, false)?;
    validate_audio(contextual_render_b, false)?;

    if !frames_bit_equal(source_render_a, source_render_b) {
        return Err(ProgSuiteNativeAudioSurvivalEvidenceErrorV1::SourceRepeatMismatch);
    }
    if !frames_bit_equal(contextual_render_a, contextual_render_b) {
        return Err(ProgSuiteNativeAudioSurvivalEvidenceErrorV1::ContextualRepeatMismatch);
    }

    let source_audio = audio_identity(source_render_a);
    let contextual_audio = audio_identity(contextual_render_a);
    let difference = compare_audio(source_render_a, contextual_render_a);
    let symbolic_score_difference_present = source_score_sha256 != contextual_score_sha256;
    let exact_audio_equal = frames_bit_equal(source_render_a, contextual_render_a);

    if !symbolic_score_difference_present && !exact_audio_equal {
        return Err(
            ProgSuiteNativeAudioSurvivalEvidenceErrorV1::AudioChangedForIdenticalScoreIdentity,
        );
    }

    let outcome = if !symbolic_score_difference_present {
        ProgSuiteAudioSurvivalOutcomeV1::NoSymbolicIntervention
    } else if exact_audio_equal {
        ProgSuiteAudioSurvivalOutcomeV1::SymbolicInterventionRendererErased
    } else {
        ProgSuiteAudioSurvivalOutcomeV1::SymbolicInterventionSurvivedNativeRender
    };

    Ok(ProgSuiteNativeAudioSurvivalEvidenceV1 {
        version: PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_EVIDENCE_VERSION.into(),
        protocol_version: PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_SURVIVAL_PROTOCOL_VERSION.into(),
        subject_id: subject_id.into(),
        motif_id: motif_id.into(),
        plan_seed,
        intent_seed,
        source_score_sha256: source_score_sha256.into(),
        contextual_score_sha256: contextual_score_sha256.into(),
        symbolic_score_difference_present,
        source_audio,
        contextual_audio,
        difference,
        repeat_receipt: ProgSuiteNativeRenderRepeatReceiptV1 {
            required_repeats_per_arm:
                PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_REPEATED_RENDERS_PER_ARM,
            source_repeat_bit_exact: true,
            contextual_repeat_bit_exact: true,
        },
        outcome,
        nonclaims: required_nonclaims(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn verify_prog_suite_native_audio_survival(
    evidence: &ProgSuiteNativeAudioSurvivalEvidenceV1,
    source_render_a: &[[f32; 2]],
    source_render_b: &[[f32; 2]],
    contextual_render_a: &[[f32; 2]],
    contextual_render_b: &[[f32; 2]],
) -> Result<(), ProgSuiteNativeAudioSurvivalEvidenceErrorV1> {
    let canonical = measure_prog_suite_native_audio_survival(
        &evidence.subject_id,
        &evidence.motif_id,
        evidence.plan_seed,
        evidence.intent_seed,
        &evidence.source_score_sha256,
        &evidence.contextual_score_sha256,
        source_render_a,
        source_render_b,
        contextual_render_a,
        contextual_render_b,
    )?;
    if &canonical != evidence {
        return Err(ProgSuiteNativeAudioSurvivalEvidenceErrorV1::EvidenceMismatch);
    }
    Ok(())
}

fn validate_identity_inputs(
    subject_id: &str,
    motif_id: &str,
    plan_seed: u64,
    intent_seed: u64,
    source_score_sha256: &str,
    contextual_score_sha256: &str,
) -> Result<(), ProgSuiteNativeAudioSurvivalEvidenceErrorV1> {
    if subject_id.trim().is_empty() {
        return Err(ProgSuiteNativeAudioSurvivalEvidenceErrorV1::EmptySubjectId);
    }
    if motif_id.trim().is_empty() {
        return Err(ProgSuiteNativeAudioSurvivalEvidenceErrorV1::EmptyMotifId);
    }
    if plan_seed != intent_seed {
        return Err(ProgSuiteNativeAudioSurvivalEvidenceErrorV1::SeedBindingMismatch {
            plan_seed,
            intent_seed,
        });
    }
    if !is_canonical_sha256(source_score_sha256) {
        return Err(ProgSuiteNativeAudioSurvivalEvidenceErrorV1::InvalidSourceScoreSha256);
    }
    if !is_canonical_sha256(contextual_score_sha256) {
        return Err(ProgSuiteNativeAudioSurvivalEvidenceErrorV1::InvalidContextualScoreSha256);
    }
    Ok(())
}

fn validate_audio(
    frames: &[[f32; 2]],
    source: bool,
) -> Result<(), ProgSuiteNativeAudioSurvivalEvidenceErrorV1> {
    if frames.is_empty() {
        return Err(if source {
            ProgSuiteNativeAudioSurvivalEvidenceErrorV1::EmptySourceAudio
        } else {
            ProgSuiteNativeAudioSurvivalEvidenceErrorV1::EmptyContextualAudio
        });
    }
    for (frame_index, frame) in frames.iter().enumerate() {
        for (channel_index, sample) in frame.iter().enumerate() {
            if !sample.is_finite() {
                return Err(if source {
                    ProgSuiteNativeAudioSurvivalEvidenceErrorV1::NonFiniteSourceSample {
                        frame_index,
                        channel_index,
                    }
                } else {
                    ProgSuiteNativeAudioSurvivalEvidenceErrorV1::NonFiniteContextualSample {
                        frame_index,
                        channel_index,
                    }
                });
            }
        }
    }
    Ok(())
}

fn audio_identity(frames: &[[f32; 2]]) -> ProgSuiteNativeAudioIdentityV1 {
    let mut bytes = Vec::with_capacity(frames.len().saturating_mul(8));
    for frame in frames {
        bytes.extend_from_slice(&frame[0].to_bits().to_le_bytes());
        bytes.extend_from_slice(&frame[1].to_bits().to_le_bytes());
    }
    ProgSuiteNativeAudioIdentityV1 {
        frame_count: frames.len(),
        sha256: sha256_hex(&bytes),
    }
}

fn compare_audio(
    source: &[[f32; 2]],
    contextual: &[[f32; 2]],
) -> ProgSuiteNativeAudioDifferenceV1 {
    let aligned_frame_count = source.len().min(contextual.len());
    let mut changed_aligned_frame_count = 0usize;
    let mut absolute_sum = 0.0_f64;
    let mut squared_sum = 0.0_f64;
    let mut peak = 0.0_f64;

    for (left, right) in source.iter().zip(contextual.iter()) {
        if left[0].to_bits() != right[0].to_bits() || left[1].to_bits() != right[1].to_bits() {
            changed_aligned_frame_count += 1;
        }
        for channel in 0..2 {
            let delta = (f64::from(left[channel]) - f64::from(right[channel])).abs();
            absolute_sum += delta;
            squared_sum += delta * delta;
            peak = peak.max(delta);
        }
    }

    let aligned_sample_count = aligned_frame_count.saturating_mul(2);
    let (mean_absolute_sample_delta, rms_sample_delta) = if aligned_sample_count == 0 {
        (0.0, 0.0)
    } else {
        let denominator = aligned_sample_count as f64;
        (absolute_sum / denominator, (squared_sum / denominator).sqrt())
    };

    ProgSuiteNativeAudioDifferenceV1 {
        aligned_frame_count,
        changed_aligned_frame_count,
        mean_absolute_sample_delta,
        rms_sample_delta,
        peak_absolute_sample_delta: peak,
        unmatched_source_tail_frames: source.len().saturating_sub(aligned_frame_count),
        unmatched_contextual_tail_frames: contextual.len().saturating_sub(aligned_frame_count),
    }
}

fn frames_bit_equal(left: &[[f32; 2]], right: &[[f32; 2]]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right.iter()).all(|(a, b)| {
            a[0].to_bits() == b[0].to_bits() && a[1].to_bits() == b[1].to_bits()
        })
}

fn is_canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn required_nonclaims() -> Vec<ProgSuiteNativeAudioSurvivalEvidenceNonClaimV1> {
    vec![
        ProgSuiteNativeAudioSurvivalEvidenceNonClaimV1::PackageVersionDoesNotEstablishSourceRevision,
        ProgSuiteNativeAudioSurvivalEvidenceNonClaimV1::WaveformDifferenceDoesNotEstablishAudibility,
        ProgSuiteNativeAudioSurvivalEvidenceNonClaimV1::WaveformDifferenceDoesNotEstablishAcousticOnsetDifference,
        ProgSuiteNativeAudioSurvivalEvidenceNonClaimV1::WaveformDifferenceDoesNotEstablishSpectralSalience,
        ProgSuiteNativeAudioSurvivalEvidenceNonClaimV1::WaveformDifferenceDoesNotEstablishListenerPreference,
        ProgSuiteNativeAudioSurvivalEvidenceNonClaimV1::WaveformDifferenceDoesNotEstablishArtisticQuality,
        ProgSuiteNativeAudioSurvivalEvidenceNonClaimV1::WholeWorkDifferenceDoesNotLocalizeTheEffect,
        ProgSuiteNativeAudioSurvivalEvidenceNonClaimV1::OneRendererDoesNotEstablishRendererGeneralization,
        ProgSuiteNativeAudioSurvivalEvidenceNonClaimV1::LockboxPopulationDoesNotEstablishUniversalMusicGeneralization,
        ProgSuiteNativeAudioSurvivalEvidenceNonClaimV1::FrameCountsDoNotEstablishStatisticalIndependence,
        ProgSuiteNativeAudioSurvivalEvidenceNonClaimV1::EvidenceDoesNotGrantProductAuthority,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn frames(values: &[[f32; 2]]) -> Vec<[f32; 2]> {
        values.to_vec()
    }

    #[test]
    fn identical_score_and_audio_is_no_intervention() {
        let audio = frames(&[[0.0, 0.0], [0.25, -0.25]]);
        let evidence = measure_prog_suite_native_audio_survival(
            "motif-01:seed-3",
            "motif-01",
            3,
            3,
            &sha(1),
            &sha(1),
            &audio,
            &audio,
            &audio,
            &audio,
        )
        .unwrap();
        assert_eq!(
            evidence.outcome,
            ProgSuiteAudioSurvivalOutcomeV1::NoSymbolicIntervention
        );
        assert_eq!(evidence.difference.changed_aligned_frame_count, 0);
        assert_eq!(evidence.source_audio, evidence.contextual_audio);
    }

    #[test]
    fn symbolic_difference_with_equal_audio_is_renderer_erasure() {
        let audio = frames(&[[0.1, -0.1], [0.2, -0.2]]);
        let evidence = measure_prog_suite_native_audio_survival(
            "motif-02:seed-11",
            "motif-02",
            11,
            11,
            &sha(1),
            &sha(2),
            &audio,
            &audio,
            &audio,
            &audio,
        )
        .unwrap();
        assert_eq!(
            evidence.outcome,
            ProgSuiteAudioSurvivalOutcomeV1::SymbolicInterventionRendererErased
        );
    }

    #[test]
    fn symbolic_difference_with_waveform_change_survives_native_render() {
        let source = frames(&[[0.0, 0.0], [0.25, -0.25], [0.5, -0.5]]);
        let contextual = frames(&[[0.0, 0.0], [0.30, -0.25], [0.5, -0.45]]);
        let evidence = measure_prog_suite_native_audio_survival(
            "motif-03:seed-23",
            "motif-03",
            23,
            23,
            &sha(1),
            &sha(2),
            &source,
            &source,
            &contextual,
            &contextual,
        )
        .unwrap();
        assert_eq!(
            evidence.outcome,
            ProgSuiteAudioSurvivalOutcomeV1::SymbolicInterventionSurvivedNativeRender
        );
        assert_eq!(evidence.difference.changed_aligned_frame_count, 2);
        assert!(evidence.difference.mean_absolute_sample_delta > 0.0);
        assert!(evidence.difference.rms_sample_delta > 0.0);
        assert!(evidence.difference.peak_absolute_sample_delta > 0.0);
        verify_prog_suite_native_audio_survival(
            &evidence,
            &source,
            &source,
            &contextual,
            &contextual,
        )
        .unwrap();
    }

    #[test]
    fn unequal_lengths_are_accounted_without_invented_alignment() {
        let source = frames(&[[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]]);
        let contextual = frames(&[[0.0, 0.0], [0.1, 0.1]]);
        let evidence = measure_prog_suite_native_audio_survival(
            "motif-04:seed-41",
            "motif-04",
            41,
            41,
            &sha(1),
            &sha(2),
            &source,
            &source,
            &contextual,
            &contextual,
        )
        .unwrap();
        assert_eq!(evidence.difference.aligned_frame_count, 2);
        assert_eq!(evidence.difference.unmatched_source_tail_frames, 1);
        assert_eq!(evidence.difference.unmatched_contextual_tail_frames, 0);
    }

    #[test]
    fn within_arm_nondeterminism_fails_closed() {
        let source_a = frames(&[[0.0, 0.0], [0.1, 0.1]]);
        let source_b = frames(&[[0.0, 0.0], [0.2, 0.1]]);
        let contextual = frames(&[[0.0, 0.0], [0.3, 0.3]]);
        assert_eq!(
            measure_prog_suite_native_audio_survival(
                "motif-05:seed-59",
                "motif-05",
                59,
                59,
                &sha(1),
                &sha(2),
                &source_a,
                &source_b,
                &contextual,
                &contextual,
            ),
            Err(ProgSuiteNativeAudioSurvivalEvidenceErrorV1::SourceRepeatMismatch)
        );
    }

    #[test]
    fn non_finite_audio_fails_closed() {
        let source = frames(&[[0.0, f32::NAN]]);
        let contextual = frames(&[[0.0, 0.0]]);
        assert!(matches!(
            measure_prog_suite_native_audio_survival(
                "motif-06:seed-79",
                "motif-06",
                79,
                79,
                &sha(1),
                &sha(2),
                &source,
                &source,
                &contextual,
                &contextual,
            ),
            Err(ProgSuiteNativeAudioSurvivalEvidenceErrorV1::NonFiniteSourceSample { .. })
        ));
    }

    #[test]
    fn identical_score_identity_cannot_hide_an_audio_change() {
        let source = frames(&[[0.0, 0.0], [0.1, 0.1]]);
        let contextual = frames(&[[0.0, 0.0], [0.2, 0.1]]);
        assert_eq!(
            measure_prog_suite_native_audio_survival(
                "motif-07:seed-97",
                "motif-07",
                97,
                97,
                &sha(1),
                &sha(1),
                &source,
                &source,
                &contextual,
                &contextual,
            ),
            Err(
                ProgSuiteNativeAudioSurvivalEvidenceErrorV1::AudioChangedForIdenticalScoreIdentity
            )
        );
    }

    #[test]
    fn serialized_metric_tampering_is_rejected_when_waveforms_are_supplied() {
        let source = frames(&[[0.0, 0.0], [0.1, 0.1]]);
        let contextual = frames(&[[0.0, 0.0], [0.2, 0.1]]);
        let mut evidence = measure_prog_suite_native_audio_survival(
            "motif-08:seed-127",
            "motif-08",
            127,
            127,
            &sha(1),
            &sha(2),
            &source,
            &source,
            &contextual,
            &contextual,
        )
        .unwrap();
        evidence.difference.changed_aligned_frame_count = 0;
        assert_eq!(
            verify_prog_suite_native_audio_survival(
                &evidence,
                &source,
                &source,
                &contextual,
                &contextual,
            ),
            Err(ProgSuiteNativeAudioSurvivalEvidenceErrorV1::EvidenceMismatch)
        );
    }
}
