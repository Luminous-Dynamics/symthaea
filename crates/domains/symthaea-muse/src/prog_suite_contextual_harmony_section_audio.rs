// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Section-localized native-audio evidence for the frozen ProgSuite contextual
//! harmony experiment.
//!
//! Whole-work PCM evidence is verified first. Exact symbolic section spans are
//! then projected into performed-event envelopes using the four canonical
//! theory voices returned by `perform_with_spec`. The common source/contextual
//! envelope becomes one matched absolute-time waveform window.
//!
//! This is descriptive localization, not acoustic isolation: sustained notes,
//! reverb, renderer state, climax doubling, and neighboring material may cross
//! section boundaries.

use super::canonical_json_sha256;
use super::prog_suite_contextual_harmony_audio_evidence::{
    ProgSuiteNativeAudioSurvivalEvidenceErrorV1, ProgSuiteNativeAudioSurvivalEvidenceV1,
    verify_prog_suite_native_audio_survival,
};
use super::prog_suite_contextual_harmony_audio_protocol::{
    ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1,
    ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
};
use crate::theory_realize::{PerformedVoice, perform_with_spec};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use symthaea_music_theory::prog_suite_contextual_harmony_comparison::ProgSuiteContextualHarmonyComparisonV1;
use symthaea_music_theory::rhythm::Duration;
use symthaea_music_theory::score::{Score, VoiceRole as TheoryRole};

pub const PROG_SUITE_CONTEXTUAL_HARMONY_SECTION_AUDIO_VERSION: &str =
    "melothaea-prog-suite-contextual-harmony-section-audio-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteSectionAudioBoundaryV1 {
    CanonicalVoicePerformedEventEnvelope,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteSectionAudioNonClaimV1 {
    PerformedEnvelopeDoesNotEqualPrivateRendererTimeline,
    SectionWindowDoesNotIsolateReverbOrRendererState,
    SectionDifferenceDoesNotEstablishAudibility,
    SectionDifferenceDoesNotEstablishAcousticOnsetDifference,
    SectionDifferenceDoesNotEstablishSpectralOrHarmonicSalience,
    SectionDifferenceDoesNotEstablishListenerPreference,
    SectionDifferenceDoesNotEstablishArtisticQuality,
    SectionFramesDoNotEstablishStatisticalIndependence,
    FixedRendererDoesNotEstablishRendererGeneralization,
    LockboxDoesNotEstablishUniversalMusicGeneralization,
    EvidenceDoesNotGrantProductAuthority,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuitePerformedSectionEnvelopeV1 {
    pub event_count: usize,
    pub start_seconds: f32,
    pub end_seconds: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteSectionAudioDifferenceV1 {
    pub analysis_start_sample: usize,
    pub requested_end_sample: usize,
    pub aligned_end_sample: usize,
    pub analyzed_frame_count: usize,
    pub changed_frame_count: usize,
    pub mean_absolute_sample_delta: f64,
    pub rms_sample_delta: f64,
    pub peak_absolute_sample_delta: f64,
    pub unavailable_source_tail_frames: usize,
    pub unavailable_contextual_tail_frames: usize,
    pub complete_window_in_both_arms: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteSectionAudioEvidenceV1 {
    pub section_index: usize,
    pub symbolic_start: Duration,
    pub symbolic_end: Duration,
    pub progression_changed: bool,
    pub symbolic_event_stream_changed: bool,
    pub source_performed: ProgSuitePerformedSectionEnvelopeV1,
    pub contextual_performed: ProgSuitePerformedSectionEnvelopeV1,
    pub common_start_seconds: f32,
    pub common_end_seconds: f32,
    pub difference: ProgSuiteSectionAudioDifferenceV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteSectionAudioLocalizationV1 {
    pub version: String,
    pub subject_index: usize,
    pub subject_id: String,
    pub motif_id: String,
    pub plan_seed: u64,
    pub sample_rate: u32,
    pub boundary: ProgSuiteSectionAudioBoundaryV1,
    pub whole_work_evidence: ProgSuiteNativeAudioSurvivalEvidenceV1,
    pub sections: Vec<ProgSuiteSectionAudioEvidenceV1>,
    pub nonclaims: Vec<ProgSuiteSectionAudioNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteSectionAudioErrorV1 {
    Protocol(ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1),
    WholeWork(ProgSuiteNativeAudioSurvivalEvidenceErrorV1),
    InvalidComparison,
    SubjectIndexOutOfRange,
    SubjectMotifMissing,
    SubjectMotifMismatch,
    SubjectSeedMismatch,
    WholeWorkIdentityMismatch,
    ScoreDigest,
    ScoreDigestMismatch,
    MissingPerformedVoice { name: String },
    PerformedVoiceCardinalityMismatch {
        name: String,
        symbolic: usize,
        performed: usize,
    },
    InvalidPerformedNote { name: String, index: usize },
    EmptySectionEnvelope { section_index: usize },
    InvalidSampleRate,
    InvalidAnalysisWindow { section_index: usize },
    EvidenceMismatch,
}

#[allow(clippy::too_many_arguments)]
pub fn measure_prog_suite_section_audio_localization(
    protocol: &ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
    subject_index: usize,
    comparison: &ProgSuiteContextualHarmonyComparisonV1,
    whole_work_evidence: &ProgSuiteNativeAudioSurvivalEvidenceV1,
    source_render_a: &[[f32; 2]],
    source_render_b: &[[f32; 2]],
    contextual_render_a: &[[f32; 2]],
    contextual_render_b: &[[f32; 2]],
) -> Result<ProgSuiteSectionAudioLocalizationV1, ProgSuiteSectionAudioErrorV1> {
    protocol
        .validate()
        .map_err(ProgSuiteSectionAudioErrorV1::Protocol)?;
    comparison
        .validate()
        .map_err(|_| ProgSuiteSectionAudioErrorV1::InvalidComparison)?;
    verify_prog_suite_native_audio_survival(
        whole_work_evidence,
        source_render_a,
        source_render_b,
        contextual_render_a,
        contextual_render_b,
    )
    .map_err(ProgSuiteSectionAudioErrorV1::WholeWork)?;

    let expected = protocol
        .source_lockbox
        .subjects
        .get(subject_index)
        .ok_or(ProgSuiteSectionAudioErrorV1::SubjectIndexOutOfRange)?;
    let motif = protocol
        .source_lockbox
        .motifs
        .iter()
        .find(|motif| motif.motif_id == expected.motif_id)
        .ok_or(ProgSuiteSectionAudioErrorV1::SubjectMotifMissing)?;
    if comparison.source_motif != motif.motif {
        return Err(ProgSuiteSectionAudioErrorV1::SubjectMotifMismatch);
    }
    if comparison.intent.seed != expected.intent_seed
        || comparison.intervention.source_plan.source_seed != expected.plan_seed
    {
        return Err(ProgSuiteSectionAudioErrorV1::SubjectSeedMismatch);
    }

    let subject_id = format!("{}:seed-{}", expected.motif_id, expected.plan_seed);
    if whole_work_evidence.subject_id != subject_id
        || whole_work_evidence.motif_id != expected.motif_id
        || whole_work_evidence.plan_seed != expected.plan_seed
        || whole_work_evidence.intent_seed != expected.intent_seed
    {
        return Err(ProgSuiteSectionAudioErrorV1::WholeWorkIdentityMismatch);
    }

    let source_score_sha256 = canonical_json_sha256(&comparison.source_realization.score)
        .map_err(|_| ProgSuiteSectionAudioErrorV1::ScoreDigest)?;
    let contextual_score_sha256 = canonical_json_sha256(&comparison.contextual_realization.score)
        .map_err(|_| ProgSuiteSectionAudioErrorV1::ScoreDigest)?;
    if source_score_sha256 != whole_work_evidence.source_score_sha256
        || contextual_score_sha256 != whole_work_evidence.contextual_score_sha256
    {
        return Err(ProgSuiteSectionAudioErrorV1::ScoreDigestMismatch);
    }

    let sample_rate = protocol.render_policy.sample_rate;
    if sample_rate == 0 {
        return Err(ProgSuiteSectionAudioErrorV1::InvalidSampleRate);
    }
    let spec = &protocol.source_lockbox.spec;
    let state = &protocol.render_policy.render_state;
    let seed = expected.plan_seed;
    let source_performance =
        perform_with_spec(&comparison.source_realization.score, spec, seed, state);
    let contextual_performance =
        perform_with_spec(&comparison.contextual_realization.score, spec, seed, state);

    let mut sections = Vec::with_capacity(comparison.sections.len());
    for section in &comparison.sections {
        let source_performed = performed_event_envelope(
            &comparison.source_realization.score,
            &source_performance,
            section.section_index,
            section.start,
            section.end,
        )?;
        let contextual_performed = performed_event_envelope(
            &comparison.contextual_realization.score,
            &contextual_performance,
            section.section_index,
            section.start,
            section.end,
        )?;
        let common_start_seconds = source_performed
            .start_seconds
            .min(contextual_performed.start_seconds);
        let common_end_seconds = source_performed
            .end_seconds
            .max(contextual_performed.end_seconds);
        let difference = compare_section_window(
            section.section_index,
            source_render_a,
            contextual_render_a,
            sample_rate,
            common_start_seconds,
            common_end_seconds,
        )?;
        sections.push(ProgSuiteSectionAudioEvidenceV1 {
            section_index: section.section_index,
            symbolic_start: section.start,
            symbolic_end: section.end,
            progression_changed: section.progression_changed,
            symbolic_event_stream_changed: !section.all_events.exact_event_stream_match,
            source_performed,
            contextual_performed,
            common_start_seconds,
            common_end_seconds,
            difference,
        });
    }

    Ok(ProgSuiteSectionAudioLocalizationV1 {
        version: PROG_SUITE_CONTEXTUAL_HARMONY_SECTION_AUDIO_VERSION.into(),
        subject_index,
        subject_id,
        motif_id: expected.motif_id.clone(),
        plan_seed: expected.plan_seed,
        sample_rate,
        boundary: ProgSuiteSectionAudioBoundaryV1::CanonicalVoicePerformedEventEnvelope,
        whole_work_evidence: whole_work_evidence.clone(),
        sections,
        nonclaims: required_nonclaims(),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn verify_prog_suite_section_audio_localization(
    evidence: &ProgSuiteSectionAudioLocalizationV1,
    protocol: &ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
    comparison: &ProgSuiteContextualHarmonyComparisonV1,
    source_render_a: &[[f32; 2]],
    source_render_b: &[[f32; 2]],
    contextual_render_a: &[[f32; 2]],
    contextual_render_b: &[[f32; 2]],
) -> Result<(), ProgSuiteSectionAudioErrorV1> {
    let canonical = measure_prog_suite_section_audio_localization(
        protocol,
        evidence.subject_index,
        comparison,
        &evidence.whole_work_evidence,
        source_render_a,
        source_render_b,
        contextual_render_a,
        contextual_render_b,
    )?;
    if &canonical != evidence {
        return Err(ProgSuiteSectionAudioErrorV1::EvidenceMismatch);
    }
    Ok(())
}

fn performed_event_envelope(
    score: &Score,
    performed: &[PerformedVoice],
    section_index: usize,
    start: Duration,
    end: Duration,
) -> Result<ProgSuitePerformedSectionEnvelopeV1, ProgSuiteSectionAudioErrorV1> {
    let mappings = [
        (TheoryRole::Bass, "Bass"),
        (TheoryRole::Harmony, "Harmony"),
        (TheoryRole::CounterMelody, "Counter"),
        (TheoryRole::Melody, "Melody"),
    ];
    let mut event_count = 0usize;
    let mut envelope_start = f32::INFINITY;
    let mut envelope_end = f32::NEG_INFINITY;

    for (role, name) in mappings {
        let symbolic = score.voice(role);
        if symbolic.is_empty() {
            continue;
        }
        let voice = performed
            .iter()
            .find(|voice| voice.name == name)
            .ok_or_else(|| ProgSuiteSectionAudioErrorV1::MissingPerformedVoice {
                name: name.into(),
            })?;
        if symbolic.len() != voice.notes.len() {
            return Err(
                ProgSuiteSectionAudioErrorV1::PerformedVoiceCardinalityMismatch {
                    name: name.into(),
                    symbolic: symbolic.len(),
                    performed: voice.notes.len(),
                },
            );
        }
        for (index, (symbolic_note, performed_note)) in
            symbolic.iter().zip(&voice.notes).enumerate()
        {
            if !duration_in_half_open_span(symbolic_note.onset, start, end) {
                continue;
            }
            let event_start = performed_note.start_time;
            let event_end = performed_note.start_time + performed_note.duration;
            if !event_start.is_finite()
                || !performed_note.duration.is_finite()
                || performed_note.duration < 0.0
                || !event_end.is_finite()
                || event_start < 0.0
                || event_end < event_start
            {
                return Err(ProgSuiteSectionAudioErrorV1::InvalidPerformedNote {
                    name: name.into(),
                    index,
                });
            }
            envelope_start = envelope_start.min(event_start);
            envelope_end = envelope_end.max(event_end);
            event_count += 1;
        }
    }

    if event_count == 0 || !envelope_start.is_finite() || !envelope_end.is_finite() {
        return Err(ProgSuiteSectionAudioErrorV1::EmptySectionEnvelope { section_index });
    }
    Ok(ProgSuitePerformedSectionEnvelopeV1 {
        event_count,
        start_seconds: envelope_start,
        end_seconds: envelope_end,
    })
}

fn duration_in_half_open_span(value: Duration, start: Duration, end: Duration) -> bool {
    compare_duration(value, start) != Ordering::Less && compare_duration(value, end) == Ordering::Less
}

fn compare_duration(left: Duration, right: Duration) -> Ordering {
    (i128::from(left.num()) * i128::from(right.den()))
        .cmp(&(i128::from(right.num()) * i128::from(left.den())))
}

fn compare_section_window(
    section_index: usize,
    source: &[[f32; 2]],
    contextual: &[[f32; 2]],
    sample_rate: u32,
    start_seconds: f32,
    end_seconds: f32,
) -> Result<ProgSuiteSectionAudioDifferenceV1, ProgSuiteSectionAudioErrorV1> {
    if !start_seconds.is_finite()
        || !end_seconds.is_finite()
        || start_seconds < 0.0
        || end_seconds <= start_seconds
    {
        return Err(ProgSuiteSectionAudioErrorV1::InvalidAnalysisWindow { section_index });
    }
    let rate = f64::from(sample_rate);
    let start_value = f64::from(start_seconds) * rate;
    let end_value = f64::from(end_seconds) * rate;
    if !start_value.is_finite() || !end_value.is_finite() || start_value < 0.0 {
        return Err(ProgSuiteSectionAudioErrorV1::InvalidAnalysisWindow { section_index });
    }
    let analysis_start_sample = start_value.floor() as usize;
    let requested_end_sample = end_value.ceil() as usize;
    let aligned_end_sample = requested_end_sample.min(source.len()).min(contextual.len());
    if analysis_start_sample >= aligned_end_sample {
        return Err(ProgSuiteSectionAudioErrorV1::InvalidAnalysisWindow { section_index });
    }

    let source_window = &source[analysis_start_sample..aligned_end_sample];
    let contextual_window = &contextual[analysis_start_sample..aligned_end_sample];
    let mut changed_frame_count = 0usize;
    let mut absolute_sum = 0.0_f64;
    let mut squared_sum = 0.0_f64;
    let mut peak = 0.0_f64;
    for (left, right) in source_window.iter().zip(contextual_window) {
        if left[0].to_bits() != right[0].to_bits() || left[1].to_bits() != right[1].to_bits() {
            changed_frame_count += 1;
        }
        for channel in 0..2 {
            let delta = (f64::from(left[channel]) - f64::from(right[channel])).abs();
            absolute_sum += delta;
            squared_sum += delta * delta;
            peak = peak.max(delta);
        }
    }
    let analyzed_frame_count = source_window.len();
    let denominator = analyzed_frame_count.saturating_mul(2) as f64;
    let unavailable_source_tail_frames = requested_end_sample.saturating_sub(source.len());
    let unavailable_contextual_tail_frames = requested_end_sample.saturating_sub(contextual.len());

    Ok(ProgSuiteSectionAudioDifferenceV1 {
        analysis_start_sample,
        requested_end_sample,
        aligned_end_sample,
        analyzed_frame_count,
        changed_frame_count,
        mean_absolute_sample_delta: absolute_sum / denominator,
        rms_sample_delta: (squared_sum / denominator).sqrt(),
        peak_absolute_sample_delta: peak,
        unavailable_source_tail_frames,
        unavailable_contextual_tail_frames,
        complete_window_in_both_arms: unavailable_source_tail_frames == 0
            && unavailable_contextual_tail_frames == 0,
    })
}

fn required_nonclaims() -> Vec<ProgSuiteSectionAudioNonClaimV1> {
    vec![
        ProgSuiteSectionAudioNonClaimV1::PerformedEnvelopeDoesNotEqualPrivateRendererTimeline,
        ProgSuiteSectionAudioNonClaimV1::SectionWindowDoesNotIsolateReverbOrRendererState,
        ProgSuiteSectionAudioNonClaimV1::SectionDifferenceDoesNotEstablishAudibility,
        ProgSuiteSectionAudioNonClaimV1::SectionDifferenceDoesNotEstablishAcousticOnsetDifference,
        ProgSuiteSectionAudioNonClaimV1::SectionDifferenceDoesNotEstablishSpectralOrHarmonicSalience,
        ProgSuiteSectionAudioNonClaimV1::SectionDifferenceDoesNotEstablishListenerPreference,
        ProgSuiteSectionAudioNonClaimV1::SectionDifferenceDoesNotEstablishArtisticQuality,
        ProgSuiteSectionAudioNonClaimV1::SectionFramesDoNotEstablishStatisticalIndependence,
        ProgSuiteSectionAudioNonClaimV1::FixedRendererDoesNotEstablishRendererGeneralization,
        ProgSuiteSectionAudioNonClaimV1::LockboxDoesNotEstablishUniversalMusicGeneralization,
        ProgSuiteSectionAudioNonClaimV1::EvidenceDoesNotGrantProductAuthority,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_span_membership_is_exact() {
        let start = Duration::new(1, 3);
        let end = Duration::new(2, 3);
        assert!(duration_in_half_open_span(Duration::new(2, 6), start, end));
        assert!(duration_in_half_open_span(Duration::new(1, 2), start, end));
        assert!(!duration_in_half_open_span(Duration::new(2, 3), start, end));
    }

    #[test]
    fn clipped_section_tail_remains_visible() {
        let source = vec![[0.0, 0.0]; 100];
        let contextual = vec![[0.0, 0.0]; 90];
        let result = compare_section_window(3, &source, &contextual, 100, 0.5, 1.0).unwrap();
        assert_eq!(result.analysis_start_sample, 50);
        assert_eq!(result.requested_end_sample, 100);
        assert_eq!(result.aligned_end_sample, 90);
        assert_eq!(result.unavailable_source_tail_frames, 0);
        assert_eq!(result.unavailable_contextual_tail_frames, 10);
        assert!(!result.complete_window_in_both_arms);
    }

    #[test]
    fn section_delta_is_descriptive_not_a_gate() {
        let source = vec![[0.0, 0.0]; 100];
        let mut contextual = source.clone();
        contextual[60] = [0.25, -0.5];
        let result = compare_section_window(1, &source, &contextual, 100, 0.5, 0.8).unwrap();
        assert_eq!(result.changed_frame_count, 1);
        assert!(result.rms_sample_delta > 0.0);
        assert!(result.peak_absolute_sample_delta > 0.0);
    }
}
