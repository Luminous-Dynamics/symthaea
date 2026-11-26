// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Descriptive 64-subject panel for ProgSuite symbolic↔acoustic tonal survival.
//!
//! This is an aggregation layer, not a new source of PCM authority. Every
//! admitted subject must first pass the full raw-PCM verifier from
//! `prog_suite_contextual_harmony_symbolic_acoustic_tonal`. The serialized
//! panel retains compact subject/section projections and can revalidate their
//! roster, arithmetic, and summary consistency, but cannot recreate PCM after
//! the raw waveforms are absent.

use super::prog_suite_contextual_harmony_audio_protocol::{
    ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1,
    ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
};
use super::prog_suite_contextual_harmony_symbolic_acoustic_tonal::{
    ProgSuiteSymbolicAcousticTonalErrorV1, ProgSuiteSymbolicAcousticTonalEvidenceV1,
    ProgSuiteTonalSurvivalDispositionV1,
    verify_prog_suite_symbolic_acoustic_tonal_evidence,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_music_theory::PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT;
use symthaea_music_theory::prog_suite_contextual_harmony_comparison::
    ProgSuiteContextualHarmonyComparisonV1;

pub const PROG_SUITE_CONTEXTUAL_HARMONY_TONAL_SURVIVAL_PANEL_VERSION: &str =
    "melothaea-prog-suite-contextual-harmony-tonal-survival-panel-v1";
pub const PROG_SUITE_TONAL_SURVIVAL_SECTION_COUNT: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteTonalPanelAdmissionV1 {
    /// Every compact subject projection was created only after the retained
    /// symbolic↔acoustic evidence reverified against its raw repeated PCM.
    PcmVerifiedBeforePanelAdmission,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteTonalPanelPrimaryUnitV1 {
    MotifSeedSubject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteTonalPanelAggregationV1 {
    DescriptiveSubjectCountsAndSectionDistributionsOnly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteTonalPanelNonClaimV1 {
    SerializedPanelDoesNotReverifyPcmWithoutRawWaveforms,
    SectionRecordsAreNotIndependentSubjects,
    FramesNotesAndBinsAreNotIndependentSubjects,
    DescriptiveMeansDoNotEstablishStatisticalSignificance,
    TonalAlignmentDoesNotEstablishHarmonicCorrectness,
    TonalAlignmentDoesNotEstablishAudibility,
    TonalAlignmentDoesNotEstablishListenerPreference,
    TonalAlignmentDoesNotEstablishArtisticQuality,
    HigherAlignmentDoesNotEstablishBetterMusic,
    RepresentationViewsSharePcmAndAreNotIndependentReplications,
    FixedRendererAndRepresentationDoNotEstablishGeneralization,
    HeldOutLockboxDoesNotEstablishUniversalMusicGeneralization,
    PanelDoesNotGrantProductAuthority,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteFiniteDistributionV1 {
    pub count: usize,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub mean: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteTonalPanelSectionProjectionV1 {
    pub section_index: usize,
    pub progression_changed: bool,
    pub symbolic_event_stream_changed: bool,
    pub disposition: ProgSuiteTonalSurvivalDispositionV1,
    pub source_acoustic_profile_mass: f64,
    pub contextual_acoustic_profile_mass: f64,
    pub source_primary_to_acoustic_l1: f64,
    pub contextual_primary_to_acoustic_l1: f64,
    pub symbolic_change_l1_norm: f64,
    pub acoustic_change_l1_norm: f64,
    pub change_vector_l1_mismatch: f64,
    pub change_vector_cosine_similarity: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteTonalPanelSubjectV1 {
    pub subject_index: usize,
    pub subject_id: String,
    pub motif_id: String,
    pub plan_seed: u64,
    pub intent_seed: u64,
    pub sections: Vec<ProgSuiteTonalPanelSectionProjectionV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteTonalPanelSectionSummaryV1 {
    pub section_index: usize,
    pub subject_count: usize,
    pub progression_changed_count: usize,
    pub symbolic_event_stream_changed_count: usize,
    pub no_primary_symbolic_distribution_change_count: usize,
    pub primary_symbolic_change_equal_acoustic_mean_shape_count: usize,
    pub primary_symbolic_and_acoustic_mean_shapes_changed_count: usize,
    pub source_acoustic_profile_mass: ProgSuiteFiniteDistributionV1,
    pub contextual_acoustic_profile_mass: ProgSuiteFiniteDistributionV1,
    pub source_primary_to_acoustic_l1: ProgSuiteFiniteDistributionV1,
    pub contextual_primary_to_acoustic_l1: ProgSuiteFiniteDistributionV1,
    pub symbolic_change_l1_norm: ProgSuiteFiniteDistributionV1,
    pub acoustic_change_l1_norm: ProgSuiteFiniteDistributionV1,
    pub change_vector_l1_mismatch: ProgSuiteFiniteDistributionV1,
    /// Only subjects where both symbolic and acoustic change vectors have
    /// nonzero L2 norm contribute to this distribution.
    pub change_vector_cosine_similarity: ProgSuiteFiniteDistributionV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteTonalPanelSummaryV1 {
    pub subject_count: usize,
    pub section_record_count: usize,
    pub distinct_motif_count: usize,
    pub distinct_plan_seed_count: usize,
    pub pcm_verified_subject_count: usize,
    pub sections: Vec<ProgSuiteTonalPanelSectionSummaryV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteTonalPanelAnalysisPlanV1 {
    pub primary_unit: ProgSuiteTonalPanelPrimaryUnitV1,
    pub expected_subject_count: usize,
    pub expected_section_count_per_subject: usize,
    pub aggregation: ProgSuiteTonalPanelAggregationV1,
    pub section_level_inference_allowed: bool,
    pub frame_level_inference_allowed: bool,
    pub note_level_inference_allowed: bool,
    pub fft_bin_level_inference_allowed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteContextualHarmonyTonalSurvivalPanelV1 {
    pub version: String,
    pub source_protocol: ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
    pub admission: ProgSuiteTonalPanelAdmissionV1,
    pub analysis: ProgSuiteTonalPanelAnalysisPlanV1,
    pub subjects: Vec<ProgSuiteTonalPanelSubjectV1>,
    pub summary: ProgSuiteTonalPanelSummaryV1,
    pub nonclaims: Vec<ProgSuiteTonalPanelNonClaimV1>,
}

/// Runtime-only input. Raw PCM is deliberately not serialized into the panel.
pub struct ProgSuiteTonalPanelSubjectInputV1<'a> {
    pub comparison: &'a ProgSuiteContextualHarmonyComparisonV1,
    pub evidence: &'a ProgSuiteSymbolicAcousticTonalEvidenceV1,
    pub source_render_a: &'a [[f32; 2]],
    pub source_render_b: &'a [[f32; 2]],
    pub contextual_render_a: &'a [[f32; 2]],
    pub contextual_render_b: &'a [[f32; 2]],
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteTonalPanelErrorV1 {
    Protocol(ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1),
    SubjectEvidence(ProgSuiteSymbolicAcousticTonalErrorV1),
    WrongVersion { found: String },
    WrongSubjectCount { found: usize },
    WrongSectionCount { subject_index: usize, found: usize },
    SubjectRosterMismatch { position: usize },
    SectionRosterMismatch { subject_index: usize, position: usize },
    InvalidMetric { subject_index: usize, section_index: usize },
    NonCanonicalAdmission,
    NonCanonicalAnalysis,
    NonCanonicalNonClaims,
    SummaryMismatch,
}

pub fn build_prog_suite_tonal_survival_panel_v1(
    protocol: &ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
    inputs: &[ProgSuiteTonalPanelSubjectInputV1<'_>],
) -> Result<ProgSuiteContextualHarmonyTonalSurvivalPanelV1, ProgSuiteTonalPanelErrorV1> {
    protocol.validate().map_err(ProgSuiteTonalPanelErrorV1::Protocol)?;
    if inputs.len() != PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT {
        return Err(ProgSuiteTonalPanelErrorV1::WrongSubjectCount {
            found: inputs.len(),
        });
    }
    if protocol.source_lockbox.subjects.len()
        != PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT
    {
        return Err(ProgSuiteTonalPanelErrorV1::WrongSubjectCount {
            found: protocol.source_lockbox.subjects.len(),
        });
    }

    let mut subjects = Vec::with_capacity(inputs.len());
    for (position, input) in inputs.iter().enumerate() {
        let expected = &protocol.source_lockbox.subjects[position];
        let expected_subject_id = format!("{}:seed-{}", expected.motif_id, expected.plan_seed);
        if input.evidence.subject_index != position
            || input.evidence.subject_id != expected_subject_id
        {
            return Err(ProgSuiteTonalPanelErrorV1::SubjectRosterMismatch { position });
        }
        verify_prog_suite_symbolic_acoustic_tonal_evidence(
            input.evidence,
            protocol,
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
            let projection = project_section(section);
            validate_projection(&projection, position)?;
            sections.push(projection);
        }
        subjects.push(ProgSuiteTonalPanelSubjectV1 {
            subject_index: position,
            subject_id: expected_subject_id,
            motif_id: expected.motif_id.clone(),
            plan_seed: expected.plan_seed,
            intent_seed: expected.intent_seed,
            sections,
        });
    }

    let summary = summarize_panel(&subjects)?;
    let panel = ProgSuiteContextualHarmonyTonalSurvivalPanelV1 {
        version: PROG_SUITE_CONTEXTUAL_HARMONY_TONAL_SURVIVAL_PANEL_VERSION.into(),
        source_protocol: protocol.clone(),
        admission: ProgSuiteTonalPanelAdmissionV1::PcmVerifiedBeforePanelAdmission,
        analysis: canonical_analysis_plan(),
        subjects,
        summary,
        nonclaims: required_nonclaims(),
    };
    panel.validate_shape()?;
    Ok(panel)
}

impl ProgSuiteContextualHarmonyTonalSurvivalPanelV1 {
    /// Validate the serialized panel's roster, compact projections, and
    /// descriptive arithmetic. This does not reverify PCM; raw waveforms are
    /// required for that and are admitted only by `build_*` above.
    pub fn validate_shape(&self) -> Result<(), ProgSuiteTonalPanelErrorV1> {
        if self.version != PROG_SUITE_CONTEXTUAL_HARMONY_TONAL_SURVIVAL_PANEL_VERSION {
            return Err(ProgSuiteTonalPanelErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        self.source_protocol
            .validate()
            .map_err(ProgSuiteTonalPanelErrorV1::Protocol)?;
        if self.admission != ProgSuiteTonalPanelAdmissionV1::PcmVerifiedBeforePanelAdmission {
            return Err(ProgSuiteTonalPanelErrorV1::NonCanonicalAdmission);
        }
        if self.analysis != canonical_analysis_plan() {
            return Err(ProgSuiteTonalPanelErrorV1::NonCanonicalAnalysis);
        }
        if self.nonclaims != required_nonclaims() {
            return Err(ProgSuiteTonalPanelErrorV1::NonCanonicalNonClaims);
        }
        if self.subjects.len() != PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT {
            return Err(ProgSuiteTonalPanelErrorV1::WrongSubjectCount {
                found: self.subjects.len(),
            });
        }

        for (position, subject) in self.subjects.iter().enumerate() {
            let expected = &self.source_protocol.source_lockbox.subjects[position];
            let expected_subject_id = format!("{}:seed-{}", expected.motif_id, expected.plan_seed);
            if subject.subject_index != position
                || subject.subject_id != expected_subject_id
                || subject.motif_id != expected.motif_id
                || subject.plan_seed != expected.plan_seed
                || subject.intent_seed != expected.intent_seed
            {
                return Err(ProgSuiteTonalPanelErrorV1::SubjectRosterMismatch { position });
            }
            if subject.sections.len() != PROG_SUITE_TONAL_SURVIVAL_SECTION_COUNT {
                return Err(ProgSuiteTonalPanelErrorV1::WrongSectionCount {
                    subject_index: position,
                    found: subject.sections.len(),
                });
            }
            for (section_position, section) in subject.sections.iter().enumerate() {
                if section.section_index != section_position {
                    return Err(ProgSuiteTonalPanelErrorV1::SectionRosterMismatch {
                        subject_index: position,
                        position: section_position,
                    });
                }
                validate_projection(section, position)?;
            }
        }

        let expected_summary = summarize_panel(&self.subjects)?;
        if self.summary != expected_summary {
            return Err(ProgSuiteTonalPanelErrorV1::SummaryMismatch);
        }
        Ok(())
    }
}

fn project_section(
    section: &super::prog_suite_contextual_harmony_symbolic_acoustic_tonal::
        ProgSuiteSymbolicAcousticTonalSectionV1,
) -> ProgSuiteTonalPanelSectionProjectionV1 {
    ProgSuiteTonalPanelSectionProjectionV1 {
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
        aggregation: ProgSuiteTonalPanelAggregationV1::DescriptiveSubjectCountsAndSectionDistributionsOnly,
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

    fn projection(section_index: usize, disposition: ProgSuiteTonalSurvivalDispositionV1) -> ProgSuiteTonalPanelSectionProjectionV1 {
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
    fn finite_distribution_is_descriptive_only() {
        let summary = distribution([3.0, 1.0, 2.0]).unwrap();
        assert_eq!(summary.count, 3);
        assert_eq!(summary.min, Some(1.0));
        assert_eq!(summary.max, Some(3.0));
        assert_eq!(summary.mean, Some(2.0));
        let empty = distribution([]).unwrap();
        assert_eq!(empty.count, 0);
        assert_eq!(empty.mean, None);
    }

    #[test]
    fn section_summary_counts_dispositions_without_a_score() {
        let a = projection(
            2,
            ProgSuiteTonalSurvivalDispositionV1::PrimarySymbolicAndAcousticMeanShapesChanged,
        );
        let b = projection(
            2,
            ProgSuiteTonalSurvivalDispositionV1::PrimarySymbolicChangeWithEqualAcousticMeanShape,
        );
        let summary = summarize_section(2, &[&a, &b]).unwrap();
        assert_eq!(summary.subject_count, 2);
        assert_eq!(summary.progression_changed_count, 2);
        assert_eq!(summary.primary_symbolic_and_acoustic_mean_shapes_changed_count, 1);
        assert_eq!(summary.primary_symbolic_change_equal_acoustic_mean_shape_count, 1);
        assert_eq!(summary.change_vector_cosine_similarity.count, 2);
        assert_eq!(summary.change_vector_cosine_similarity.mean, Some(0.8));
    }

    #[test]
    fn invalid_alignment_fails_closed() {
        let mut value = projection(
            1,
            ProgSuiteTonalSurvivalDispositionV1::PrimarySymbolicAndAcousticMeanShapesChanged,
        );
        value.change_vector_cosine_similarity = Some(1.5);
        assert!(matches!(
            validate_projection(&value, 0),
            Err(ProgSuiteTonalPanelErrorV1::InvalidMetric { .. })
        ));
    }
}
