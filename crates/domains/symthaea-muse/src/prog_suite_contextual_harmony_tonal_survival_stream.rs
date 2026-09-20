// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Streaming raw-PCM admission for the frozen ProgSuite tonal-survival panel.
//!
//! The canonical V1 panel builder accepts a slice of 64 runtime inputs. That is
//! semantically correct but operationally expensive because each input borrows
//! four complete repeated StereoF32 renders. A caller that constructs the slice
//! directly must retain every subject's raw PCM until the panel is finished,
//! even though verification itself is strictly subject-by-subject.
//!
//! This builder preserves the exact V1 evidence theorem while changing only
//! the lifetime shape: one subject is reverified against its raw repeated PCM,
//! immediately projected into the existing compact panel record, and then the
//! caller may drop that PCM before admitting the next subject.
//!
//! No new evidence schema or authority is introduced. Partial builder state is
//! deliberately not serializable and is never a panel. Only `finish()` after
//! all 64 canonical subjects have been PCM-verified returns the existing
//! [`ProgSuiteContextualHarmonyTonalSurvivalPanelV1`] schema, which is then
//! checked by its canonical `validate_shape()` implementation.

use super::prog_suite_contextual_harmony_audio_protocol::
    ProgSuiteContextualHarmonyAudioSurvivalProtocolV1;
use super::prog_suite_contextual_harmony_symbolic_acoustic_tonal::{
    ProgSuiteTonalSurvivalDispositionV1, verify_prog_suite_symbolic_acoustic_tonal_evidence,
};
use super::prog_suite_contextual_harmony_tonal_survival_panel::{
    PROG_SUITE_CONTEXTUAL_HARMONY_TONAL_SURVIVAL_PANEL_VERSION,
    PROG_SUITE_TONAL_SURVIVAL_SECTION_COUNT, ProgSuiteContextualHarmonyTonalSurvivalPanelV1,
    ProgSuiteFiniteDistributionV1, ProgSuiteTonalPanelAdmissionV1,
    ProgSuiteTonalPanelAggregationV1, ProgSuiteTonalPanelAnalysisPlanV1,
    ProgSuiteTonalPanelErrorV1, ProgSuiteTonalPanelNonClaimV1,
    ProgSuiteTonalPanelPrimaryUnitV1, ProgSuiteTonalPanelSectionProjectionV1,
    ProgSuiteTonalPanelSectionSummaryV1, ProgSuiteTonalPanelSubjectInputV1,
    ProgSuiteTonalPanelSubjectV1, ProgSuiteTonalPanelSummaryV1,
};
use std::collections::BTreeSet;
use symthaea_music_theory::PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT;

/// Runtime-only streaming admission state.
///
/// The builder retains only the frozen protocol plus compact subject
/// projections. It never retains raw PCM or references into caller-owned PCM.
#[derive(Debug, Clone)]
pub struct ProgSuiteTonalSurvivalPanelStreamBuilderV1 {
    protocol: ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
    subjects: Vec<ProgSuiteTonalPanelSubjectV1>,
}

impl ProgSuiteTonalSurvivalPanelStreamBuilderV1 {
    pub fn new(
        protocol: &ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
    ) -> Result<Self, ProgSuiteTonalPanelErrorV1> {
        protocol
            .validate()
            .map_err(ProgSuiteTonalPanelErrorV1::Protocol)?;
        if protocol.source_lockbox.subjects.len()
            != PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT
        {
            return Err(ProgSuiteTonalPanelErrorV1::WrongSubjectCount {
                found: protocol.source_lockbox.subjects.len(),
            });
        }
        Ok(Self {
            protocol: protocol.clone(),
            subjects: Vec::with_capacity(PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT),
        })
    }

    pub fn admitted_subject_count(&self) -> usize {
        self.subjects.len()
    }

    pub fn remaining_subject_count(&self) -> usize {
        PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT.saturating_sub(self.subjects.len())
    }

    /// Verify and compact exactly the next canonical lockbox subject.
    ///
    /// On success this method stores no references to `input`, so all four raw
    /// PCM buffers may be dropped immediately after the call returns.
    pub fn admit_next(
        &mut self,
        input: &ProgSuiteTonalPanelSubjectInputV1<'_>,
    ) -> Result<(), ProgSuiteTonalPanelErrorV1> {
        let position = self.subjects.len();
        if position >= PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT {
            return Err(ProgSuiteTonalPanelErrorV1::WrongSubjectCount {
                found: position.saturating_add(1),
            });
        }

        let expected = &self.protocol.source_lockbox.subjects[position];
        let expected_subject_id = format!("{}:seed-{}", expected.motif_id, expected.plan_seed);
        if input.evidence.subject_index != position
            || input.evidence.subject_id != expected_subject_id
        {
            return Err(ProgSuiteTonalPanelErrorV1::SubjectRosterMismatch { position });
        }

        verify_prog_suite_symbolic_acoustic_tonal_evidence(
            input.evidence,
            &self.protocol,
            input.comparison,
            input.source_render_a,
            input.source_render_b,
            input.contextual_render_a,
            input.contextual_render_b,
        )
        .map_err(ProgSuiteTonalPanelErrorV1::SubjectEvidence)?;

        if input.evidence.sections.len() != PROG_SUITE_TONAL_SURVIVAL_SECTION_COUNT {
            return Err(ProgSuiteTonalPanelErrorV1::WrongSectionCount {
                subject_index: position,
                found: input.evidence.sections.len(),
            });
        }

        let mut sections = Vec::with_capacity(PROG_SUITE_TONAL_SURVIVAL_SECTION_COUNT);
        for (section_position, section) in input.evidence.sections.iter().enumerate() {
            if section.section_index != section_position {
                return Err(ProgSuiteTonalPanelErrorV1::SectionRosterMismatch {
                    subject_index: position,
                    position: section_position,
                });
            }
            let projection = ProgSuiteTonalPanelSectionProjectionV1 {
                section_index: section.section_index,
                progression_changed: section.progression_changed,
                symbolic_event_stream_changed: section.symbolic_event_stream_changed,
                disposition: section.disposition,
                source_acoustic_profile_mass: section.source_acoustic.profile_mass,
                contextual_acoustic_profile_mass: section.contextual_acoustic.profile_mass,
                source_primary_to_acoustic_l1: section.source_primary_to_acoustic_shape.l1_distance,
                contextual_primary_to_acoustic_l1: section
                    .contextual_primary_to_acoustic_shape
                    .l1_distance,
                symbolic_change_l1_norm: section.change_agreement.symbolic_change_l1_norm,
                acoustic_change_l1_norm: section.change_agreement.acoustic_change_l1_norm,
                change_vector_l1_mismatch: section.change_agreement.change_vector_l1_mismatch,
                change_vector_cosine_similarity: section
                    .change_agreement
                    .change_vector_cosine_similarity,
            };
            validate_projection(&projection, position)?;
            sections.push(projection);
        }

        self.subjects.push(ProgSuiteTonalPanelSubjectV1 {
            subject_index: position,
            subject_id: expected_subject_id,
            motif_id: expected.motif_id.clone(),
            plan_seed: expected.plan_seed,
            intent_seed: expected.intent_seed,
            sections,
        });
        Ok(())
    }

    /// Finish only after every frozen motif×seed subject has been admitted.
    ///
    /// The returned value is the pre-existing V1 panel schema. `validate_shape`
    /// is run before return, so duplicated summary logic in this operational
    /// adapter cannot silently drift from the canonical panel validator.
    pub fn finish(
        self,
    ) -> Result<ProgSuiteContextualHarmonyTonalSurvivalPanelV1, ProgSuiteTonalPanelErrorV1> {
        if self.subjects.len() != PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT {
            return Err(ProgSuiteTonalPanelErrorV1::WrongSubjectCount {
                found: self.subjects.len(),
            });
        }
        let summary = summarize_panel(&self.subjects)?;
        let panel = ProgSuiteContextualHarmonyTonalSurvivalPanelV1 {
            version: PROG_SUITE_CONTEXTUAL_HARMONY_TONAL_SURVIVAL_PANEL_VERSION.into(),
            source_protocol: self.protocol,
            admission: ProgSuiteTonalPanelAdmissionV1::PcmVerifiedBeforePanelAdmission,
            analysis: canonical_analysis_plan(),
            subjects: self.subjects,
            summary,
            nonclaims: required_nonclaims(),
        };
        panel.validate_shape()?;
        Ok(panel)
    }
}

fn validate_projection(
    section: &ProgSuiteTonalPanelSectionProjectionV1,
    subject_index: usize,
) -> Result<(), ProgSuiteTonalPanelErrorV1> {
    let required = [
        section.source_acoustic_profile_mass,
        section.contextual_acoustic_profile_mass,
        section.source_primary_to_acoustic_l1,
        section.contextual_primary_to_acoustic_l1,
        section.symbolic_change_l1_norm,
        section.acoustic_change_l1_norm,
        section.change_vector_l1_mismatch,
    ];
    if required.iter().any(|value| !value.is_finite() || *value < 0.0)
        || section.source_acoustic_profile_mass > 1.0 + 1.0e-9
        || section.contextual_acoustic_profile_mass > 1.0 + 1.0e-9
        || section
            .change_vector_cosine_similarity
            .is_some_and(|value| !value.is_finite() || !(-1.0..=1.0).contains(&value))
    {
        return Err(ProgSuiteTonalPanelErrorV1::InvalidMetric {
            subject_index,
            section_index: section.section_index,
        });
    }
    Ok(())
}

fn summarize_panel(
    subjects: &[ProgSuiteTonalPanelSubjectV1],
) -> Result<ProgSuiteTonalPanelSummaryV1, ProgSuiteTonalPanelErrorV1> {
    let motifs = subjects
        .iter()
        .map(|subject| subject.motif_id.clone())
        .collect::<BTreeSet<_>>();
    let seeds = subjects
        .iter()
        .map(|subject| subject.plan_seed)
        .collect::<BTreeSet<_>>();
    let mut sections = Vec::with_capacity(PROG_SUITE_TONAL_SURVIVAL_SECTION_COUNT);
    for section_index in 0..PROG_SUITE_TONAL_SURVIVAL_SECTION_COUNT {
        let records = subjects
            .iter()
            .map(|subject| &subject.sections[section_index])
            .collect::<Vec<_>>();
        sections.push(summarize_section(section_index, &records)?);
    }
    Ok(ProgSuiteTonalPanelSummaryV1 {
        subject_count: subjects.len(),
        section_record_count: subjects.len() * PROG_SUITE_TONAL_SURVIVAL_SECTION_COUNT,
        distinct_motif_count: motifs.len(),
        distinct_plan_seed_count: seeds.len(),
        pcm_verified_subject_count: subjects.len(),
        sections,
    })
}

fn summarize_section(
    section_index: usize,
    records: &[&ProgSuiteTonalPanelSectionProjectionV1],
) -> Result<ProgSuiteTonalPanelSectionSummaryV1, ProgSuiteTonalPanelErrorV1> {
    let progression_changed_count = records
        .iter()
        .filter(|record| record.progression_changed)
        .count();
    let symbolic_event_stream_changed_count = records
        .iter()
        .filter(|record| record.symbolic_event_stream_changed)
        .count();
    let no_primary_symbolic_distribution_change_count = records
        .iter()
        .filter(|record| {
            record.disposition
                == ProgSuiteTonalSurvivalDispositionV1::NoPrimarySymbolicDistributionChange
        })
        .count();
    let primary_symbolic_change_equal_acoustic_mean_shape_count = records
        .iter()
        .filter(|record| {
            record.disposition
                == ProgSuiteTonalSurvivalDispositionV1::PrimarySymbolicChangeWithEqualAcousticMeanShape
        })
        .count();
    let primary_symbolic_and_acoustic_mean_shapes_changed_count = records
        .iter()
        .filter(|record| {
            record.disposition
                == ProgSuiteTonalSurvivalDispositionV1::PrimarySymbolicAndAcousticMeanShapesChanged
        })
        .count();

    Ok(ProgSuiteTonalPanelSectionSummaryV1 {
        section_index,
        subject_count: records.len(),
        progression_changed_count,
        symbolic_event_stream_changed_count,
        no_primary_symbolic_distribution_change_count,
        primary_symbolic_change_equal_acoustic_mean_shape_count,
        primary_symbolic_and_acoustic_mean_shapes_changed_count,
        source_acoustic_profile_mass: distribution(
            records.iter().map(|record| record.source_acoustic_profile_mass),
        )?,
        contextual_acoustic_profile_mass: distribution(
            records
                .iter()
                .map(|record| record.contextual_acoustic_profile_mass),
        )?,
        source_primary_to_acoustic_l1: distribution(
            records.iter().map(|record| record.source_primary_to_acoustic_l1),
        )?,
        contextual_primary_to_acoustic_l1: distribution(
            records
                .iter()
                .map(|record| record.contextual_primary_to_acoustic_l1),
        )?,
        symbolic_change_l1_norm: distribution(
            records.iter().map(|record| record.symbolic_change_l1_norm),
        )?,
        acoustic_change_l1_norm: distribution(
            records.iter().map(|record| record.acoustic_change_l1_norm),
        )?,
        change_vector_l1_mismatch: distribution(
            records.iter().map(|record| record.change_vector_l1_mismatch),
        )?,
        change_vector_cosine_similarity: distribution(
            records
                .iter()
                .filter_map(|record| record.change_vector_cosine_similarity),
        )?,
    })
}

fn distribution(
    values: impl IntoIterator<Item = f64>,
) -> Result<ProgSuiteFiniteDistributionV1, ProgSuiteTonalPanelErrorV1> {
    let values = values.into_iter().collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(ProgSuiteTonalPanelErrorV1::SummaryMismatch);
    }
    if values.is_empty() {
        return Ok(ProgSuiteFiniteDistributionV1 {
            count: 0,
            min: None,
            max: None,
            mean: None,
        });
    }
    let min = values
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .expect("nonempty values");
    let max = values
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .expect("nonempty values");
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if !mean.is_finite() {
        return Err(ProgSuiteTonalPanelErrorV1::SummaryMismatch);
    }
    Ok(ProgSuiteFiniteDistributionV1 {
        count: values.len(),
        min: Some(min),
        max: Some(max),
        mean: Some(mean),
    })
}

fn canonical_analysis_plan() -> ProgSuiteTonalPanelAnalysisPlanV1 {
    ProgSuiteTonalPanelAnalysisPlanV1 {
        primary_unit: ProgSuiteTonalPanelPrimaryUnitV1::MotifSeedSubject,
        expected_subject_count: PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT,
        expected_section_count_per_subject: PROG_SUITE_TONAL_SURVIVAL_SECTION_COUNT,
        aggregation:
            ProgSuiteTonalPanelAggregationV1::DescriptiveSubjectCountsAndSectionDistributionsOnly,
        section_level_inference_allowed: false,
        frame_level_inference_allowed: false,
        note_level_inference_allowed: false,
        fft_bin_level_inference_allowed: false,
    }
}

fn required_nonclaims() -> Vec<ProgSuiteTonalPanelNonClaimV1> {
    vec![
        ProgSuiteTonalPanelNonClaimV1::SerializedPanelDoesNotReverifyPcmWithoutRawWaveforms,
        ProgSuiteTonalPanelNonClaimV1::SectionRecordsAreNotIndependentSubjects,
        ProgSuiteTonalPanelNonClaimV1::FramesNotesAndBinsAreNotIndependentSubjects,
        ProgSuiteTonalPanelNonClaimV1::DescriptiveMeansDoNotEstablishStatisticalSignificance,
        ProgSuiteTonalPanelNonClaimV1::TonalAlignmentDoesNotEstablishHarmonicCorrectness,
        ProgSuiteTonalPanelNonClaimV1::TonalAlignmentDoesNotEstablishAudibility,
        ProgSuiteTonalPanelNonClaimV1::TonalAlignmentDoesNotEstablishListenerPreference,
        ProgSuiteTonalPanelNonClaimV1::TonalAlignmentDoesNotEstablishArtisticQuality,
        ProgSuiteTonalPanelNonClaimV1::HigherAlignmentDoesNotEstablishBetterMusic,
        ProgSuiteTonalPanelNonClaimV1::RepresentationViewsSharePcmAndAreNotIndependentReplications,
        ProgSuiteTonalPanelNonClaimV1::FixedRendererAndRepresentationDoNotEstablishGeneralization,
        ProgSuiteTonalPanelNonClaimV1::HeldOutLockboxDoesNotEstablishUniversalMusicGeneralization,
        ProgSuiteTonalPanelNonClaimV1::PanelDoesNotGrantProductAuthority,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::prog_suite_contextual_harmony_audio_protocol::{
        ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1,
        predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1,
    };

    fn projection(
        section_index: usize,
        disposition: ProgSuiteTonalSurvivalDispositionV1,
    ) -> ProgSuiteTonalPanelSectionProjectionV1 {
        ProgSuiteTonalPanelSectionProjectionV1 {
            section_index,
            progression_changed: true,
            symbolic_event_stream_changed: true,
            disposition,
            source_acoustic_profile_mass: 1.0,
            contextual_acoustic_profile_mass: 0.75,
            source_primary_to_acoustic_l1: 0.4,
            contextual_primary_to_acoustic_l1: 0.3,
            symbolic_change_l1_norm: 0.5,
            acoustic_change_l1_norm: 0.4,
            change_vector_l1_mismatch: 0.2,
            change_vector_cosine_similarity: Some(0.8),
        }
    }

    #[test]
    fn empty_stream_never_becomes_a_panel() {
        let protocol =
            predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1().unwrap();
        let builder = ProgSuiteTonalSurvivalPanelStreamBuilderV1::new(&protocol).unwrap();
        assert_eq!(builder.admitted_subject_count(), 0);
        assert_eq!(
            builder.remaining_subject_count(),
            PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT
        );
        assert!(matches!(
            builder.finish(),
            Err(ProgSuiteTonalPanelErrorV1::WrongSubjectCount { found: 0 })
        ));
    }

    #[test]
    fn invalid_protocol_is_rejected_before_any_admission() {
        let mut protocol =
            predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1().unwrap();
        protocol.version.push_str("-tampered");
        assert!(matches!(
            ProgSuiteTonalSurvivalPanelStreamBuilderV1::new(&protocol),
            Err(ProgSuiteTonalPanelErrorV1::Protocol(
                ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1::WrongVersion { .. }
            ))
        ));
    }

    #[test]
    fn streaming_summary_matches_existing_canonical_shape_validator() {
        let protocol =
            predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1().unwrap();
        let subjects = protocol
            .source_lockbox
            .subjects
            .iter()
            .enumerate()
            .map(|(subject_index, expected)| ProgSuiteTonalPanelSubjectV1 {
                subject_index,
                subject_id: format!("{}:seed-{}", expected.motif_id, expected.plan_seed),
                motif_id: expected.motif_id.clone(),
                plan_seed: expected.plan_seed,
                intent_seed: expected.intent_seed,
                sections: (0..PROG_SUITE_TONAL_SURVIVAL_SECTION_COUNT)
                    .map(|section_index| {
                        projection(
                            section_index,
                            match section_index % 3 {
                                0 => ProgSuiteTonalSurvivalDispositionV1::
                                    NoPrimarySymbolicDistributionChange,
                                1 => ProgSuiteTonalSurvivalDispositionV1::
                                    PrimarySymbolicChangeWithEqualAcousticMeanShape,
                                _ => ProgSuiteTonalSurvivalDispositionV1::
                                    PrimarySymbolicAndAcousticMeanShapesChanged,
                            },
                        )
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let panel = ProgSuiteContextualHarmonyTonalSurvivalPanelV1 {
            version: PROG_SUITE_CONTEXTUAL_HARMONY_TONAL_SURVIVAL_PANEL_VERSION.into(),
            source_protocol: protocol,
            admission: ProgSuiteTonalPanelAdmissionV1::PcmVerifiedBeforePanelAdmission,
            analysis: canonical_analysis_plan(),
            summary: summarize_panel(&subjects).unwrap(),
            subjects,
            nonclaims: required_nonclaims(),
        };
        panel.validate_shape().unwrap();
        assert_eq!(
            panel.summary.subject_count,
            PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT
        );
        assert_eq!(
            panel.summary.section_record_count,
            PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT
                * PROG_SUITE_TONAL_SURVIVAL_SECTION_COUNT
        );
    }
}
