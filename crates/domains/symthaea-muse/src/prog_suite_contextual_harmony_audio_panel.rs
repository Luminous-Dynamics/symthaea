// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Descriptive 64-subject aggregation for native ProgSuite audio-survival
//! evidence.
//!
//! This panel is not a substitute for waveform verification. Each input record
//! must first have been remeasured with
//! `verify_prog_suite_native_audio_survival(...)` against its exact waveforms.
//! The panel then checks serialized-record consistency, exact lockbox subject
//! identity/order, and descriptive outcome/metric arithmetic.

use super::prog_suite_contextual_harmony_audio_evidence::{
    PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_EVIDENCE_VERSION,
    ProgSuiteNativeAudioSurvivalEvidenceNonClaimV1,
    ProgSuiteNativeAudioSurvivalEvidenceV1,
};
use super::prog_suite_contextual_harmony_audio_protocol::{
    PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_REPEATED_RENDERS_PER_ARM,
    PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_SURVIVAL_PROTOCOL_VERSION,
    ProgSuiteAudioSurvivalOutcomeV1,
    ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1,
    ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
};
use serde::{Deserialize, Serialize};
use symthaea_music_theory::PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT;

pub const PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_PANEL_VERSION: &str =
    "melothaea-prog-suite-contextual-harmony-audio-panel-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteAudioPanelAdmissionV1 {
    /// The panel assumes every record was waveform-verified before admission.
    /// Panel validation can verify serialized consistency but cannot recreate
    /// waveform hashes or deltas without the source PCM.
    WaveformVerifiedBeforeAdmission,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteAudioPanelAggregationV1 {
    DescriptiveSubjectCountsAndMetricDistributionsOnly,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteAudioMetricDistributionV1 {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteAudioSurvivalPanelSummaryV1 {
    pub subject_count: usize,
    pub no_symbolic_intervention_count: usize,
    pub renderer_erased_count: usize,
    pub renderer_survived_count: usize,
    pub changed_aligned_frame_count: ProgSuiteAudioMetricDistributionV1,
    pub mean_absolute_sample_delta: ProgSuiteAudioMetricDistributionV1,
    pub rms_sample_delta: ProgSuiteAudioMetricDistributionV1,
    pub peak_absolute_sample_delta: ProgSuiteAudioMetricDistributionV1,
    pub unmatched_source_tail_frames: ProgSuiteAudioMetricDistributionV1,
    pub unmatched_contextual_tail_frames: ProgSuiteAudioMetricDistributionV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteAudioPanelNonClaimV1 {
    PanelDoesNotReverifyWaveformsWithoutPcm,
    DescriptiveCountsDoNotEstablishStatisticalIndependence,
    SurvivalFractionDoesNotEstablishAudibility,
    SurvivalFractionDoesNotEstablishListenerPreference,
    SurvivalFractionDoesNotEstablishArtisticQuality,
    WholeWorkMetricsDoNotLocalizeTheEffect,
    OneRendererDoesNotEstablishRendererGeneralization,
    LockboxPopulationDoesNotEstablishUniversalMusicGeneralization,
    PanelDoesNotGrantProductAuthority,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteNativeAudioSurvivalPanelV1 {
    pub version: String,
    pub source_protocol: ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
    pub admission: ProgSuiteAudioPanelAdmissionV1,
    pub aggregation: ProgSuiteAudioPanelAggregationV1,
    /// Exact motif-major / seed-major ordering from the retained lockbox.
    pub records: Vec<ProgSuiteNativeAudioSurvivalEvidenceV1>,
    pub summary: ProgSuiteAudioSurvivalPanelSummaryV1,
    pub nonclaims: Vec<ProgSuiteAudioPanelNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteNativeAudioSurvivalPanelErrorV1 {
    SourceProtocol(ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1),
    WrongVersion { found: String },
    WrongRecordCount { found: usize },
    SubjectIdentityMismatch { index: usize },
    RecordVersionMismatch { index: usize },
    RecordProtocolMismatch { index: usize },
    InvalidRecordShape { index: usize },
    DuplicateSubjectIdentity { subject_id: String },
    NonFiniteSummary,
    CanonicalPanelMismatch,
}

pub fn build_prog_suite_native_audio_survival_panel(
    source_protocol: &ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
    records: Vec<ProgSuiteNativeAudioSurvivalEvidenceV1>,
) -> Result<ProgSuiteNativeAudioSurvivalPanelV1, ProgSuiteNativeAudioSurvivalPanelErrorV1> {
    source_protocol
        .validate()
        .map_err(ProgSuiteNativeAudioSurvivalPanelErrorV1::SourceProtocol)?;
    validate_records(source_protocol, &records)?;
    let summary = summarize(&records)?;
    Ok(ProgSuiteNativeAudioSurvivalPanelV1 {
        version: PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_PANEL_VERSION.into(),
        source_protocol: source_protocol.clone(),
        admission: ProgSuiteAudioPanelAdmissionV1::WaveformVerifiedBeforeAdmission,
        aggregation: ProgSuiteAudioPanelAggregationV1::DescriptiveSubjectCountsAndMetricDistributionsOnly,
        records,
        summary,
        nonclaims: required_nonclaims(),
    })
}

impl ProgSuiteNativeAudioSurvivalPanelV1 {
    /// Validate everything available from serialized evidence. This does not
    /// replace the per-record PCM-backed verifier from the evidence module.
    pub fn validate_serialized(
        &self,
    ) -> Result<(), ProgSuiteNativeAudioSurvivalPanelErrorV1> {
        if self.version != PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_PANEL_VERSION {
            return Err(ProgSuiteNativeAudioSurvivalPanelErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        self.source_protocol
            .validate()
            .map_err(ProgSuiteNativeAudioSurvivalPanelErrorV1::SourceProtocol)?;
        validate_records(&self.source_protocol, &self.records)?;
        let canonical = build_prog_suite_native_audio_survival_panel(
            &self.source_protocol,
            self.records.clone(),
        )?;
        if &canonical != self {
            return Err(ProgSuiteNativeAudioSurvivalPanelErrorV1::CanonicalPanelMismatch);
        }
        Ok(())
    }
}

fn validate_records(
    protocol: &ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
    records: &[ProgSuiteNativeAudioSurvivalEvidenceV1],
) -> Result<(), ProgSuiteNativeAudioSurvivalPanelErrorV1> {
    if records.len() != PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT {
        return Err(ProgSuiteNativeAudioSurvivalPanelErrorV1::WrongRecordCount {
            found: records.len(),
        });
    }
    let expected = &protocol.source_lockbox.subjects;
    if expected.len() != records.len() {
        return Err(ProgSuiteNativeAudioSurvivalPanelErrorV1::WrongRecordCount {
            found: records.len(),
        });
    }
    let mut seen = std::collections::BTreeSet::new();
    for (index, (subject, record)) in expected.iter().zip(records).enumerate() {
        let subject_id = canonical_subject_id(&subject.motif_id, subject.plan_seed);
        if record.subject_id != subject_id
            || record.motif_id != subject.motif_id
            || record.plan_seed != subject.plan_seed
            || record.intent_seed != subject.intent_seed
        {
            return Err(ProgSuiteNativeAudioSurvivalPanelErrorV1::SubjectIdentityMismatch {
                index,
            });
        }
        if !seen.insert(record.subject_id.clone()) {
            return Err(ProgSuiteNativeAudioSurvivalPanelErrorV1::DuplicateSubjectIdentity {
                subject_id: record.subject_id.clone(),
            });
        }
        if record.version != PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_EVIDENCE_VERSION {
            return Err(ProgSuiteNativeAudioSurvivalPanelErrorV1::RecordVersionMismatch {
                index,
            });
        }
        if record.protocol_version != PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_SURVIVAL_PROTOCOL_VERSION {
            return Err(ProgSuiteNativeAudioSurvivalPanelErrorV1::RecordProtocolMismatch {
                index,
            });
        }
        if !record_shape_is_consistent(record) {
            return Err(ProgSuiteNativeAudioSurvivalPanelErrorV1::InvalidRecordShape {
                index,
            });
        }
    }
    Ok(())
}

fn record_shape_is_consistent(record: &ProgSuiteNativeAudioSurvivalEvidenceV1) -> bool {
    if !is_canonical_sha256(&record.source_score_sha256)
        || !is_canonical_sha256(&record.contextual_score_sha256)
        || !is_canonical_sha256(&record.source_audio.sha256)
        || !is_canonical_sha256(&record.contextual_audio.sha256)
        || record.source_audio.frame_count == 0
        || record.contextual_audio.frame_count == 0
        || record.repeat_receipt.required_repeats_per_arm
            != PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_REPEATED_RENDERS_PER_ARM
        || !record.repeat_receipt.source_repeat_bit_exact
        || !record.repeat_receipt.contextual_repeat_bit_exact
        || record.nonclaims != required_evidence_nonclaims()
    {
        return false;
    }

    let expected_symbolic_difference = record.source_score_sha256 != record.contextual_score_sha256;
    if record.symbolic_score_difference_present != expected_symbolic_difference {
        return false;
    }

    let aligned = record.source_audio.frame_count.min(record.contextual_audio.frame_count);
    if record.difference.aligned_frame_count != aligned
        || record.difference.changed_aligned_frame_count > aligned
        || record.difference.unmatched_source_tail_frames
            != record.source_audio.frame_count.saturating_sub(aligned)
        || record.difference.unmatched_contextual_tail_frames
            != record.contextual_audio.frame_count.saturating_sub(aligned)
        || !finite_nonnegative(record.difference.mean_absolute_sample_delta)
        || !finite_nonnegative(record.difference.rms_sample_delta)
        || !finite_nonnegative(record.difference.peak_absolute_sample_delta)
    {
        return false;
    }

    let audio_equal = record.source_audio == record.contextual_audio;
    if audio_equal {
        if record.difference.changed_aligned_frame_count != 0
            || record.difference.unmatched_source_tail_frames != 0
            || record.difference.unmatched_contextual_tail_frames != 0
            || record.difference.mean_absolute_sample_delta != 0.0
            || record.difference.rms_sample_delta != 0.0
            || record.difference.peak_absolute_sample_delta != 0.0
        {
            return false;
        }
    } else if record.difference.changed_aligned_frame_count == 0
        && record.difference.unmatched_source_tail_frames == 0
        && record.difference.unmatched_contextual_tail_frames == 0
    {
        return false;
    }

    let expected_outcome = if !expected_symbolic_difference {
        if !audio_equal {
            return false;
        }
        ProgSuiteAudioSurvivalOutcomeV1::NoSymbolicIntervention
    } else if audio_equal {
        ProgSuiteAudioSurvivalOutcomeV1::SymbolicInterventionRendererErased
    } else {
        ProgSuiteAudioSurvivalOutcomeV1::SymbolicInterventionSurvivedNativeRender
    };
    record.outcome == expected_outcome
}

fn summarize(
    records: &[ProgSuiteNativeAudioSurvivalEvidenceV1],
) -> Result<ProgSuiteAudioSurvivalPanelSummaryV1, ProgSuiteNativeAudioSurvivalPanelErrorV1> {
    let no_symbolic_intervention_count = records
        .iter()
        .filter(|record| record.outcome == ProgSuiteAudioSurvivalOutcomeV1::NoSymbolicIntervention)
        .count();
    let renderer_erased_count = records
        .iter()
        .filter(|record| {
            record.outcome == ProgSuiteAudioSurvivalOutcomeV1::SymbolicInterventionRendererErased
        })
        .count();
    let renderer_survived_count = records
        .iter()
        .filter(|record| {
            record.outcome
                == ProgSuiteAudioSurvivalOutcomeV1::SymbolicInterventionSurvivedNativeRender
        })
        .count();

    let summary = ProgSuiteAudioSurvivalPanelSummaryV1 {
        subject_count: records.len(),
        no_symbolic_intervention_count,
        renderer_erased_count,
        renderer_survived_count,
        changed_aligned_frame_count: distribution(
            records
                .iter()
                .map(|record| record.difference.changed_aligned_frame_count as f64),
        )?,
        mean_absolute_sample_delta: distribution(
            records
                .iter()
                .map(|record| record.difference.mean_absolute_sample_delta),
        )?,
        rms_sample_delta: distribution(
            records.iter().map(|record| record.difference.rms_sample_delta),
        )?,
        peak_absolute_sample_delta: distribution(
            records
                .iter()
                .map(|record| record.difference.peak_absolute_sample_delta),
        )?,
        unmatched_source_tail_frames: distribution(
            records
                .iter()
                .map(|record| record.difference.unmatched_source_tail_frames as f64),
        )?,
        unmatched_contextual_tail_frames: distribution(
            records
                .iter()
                .map(|record| record.difference.unmatched_contextual_tail_frames as f64),
        )?,
    };
    if summary.no_symbolic_intervention_count
        + summary.renderer_erased_count
        + summary.renderer_survived_count
        != summary.subject_count
    {
        return Err(ProgSuiteNativeAudioSurvivalPanelErrorV1::NonFiniteSummary);
    }
    Ok(summary)
}

fn distribution(
    values: impl Iterator<Item = f64>,
) -> Result<ProgSuiteAudioMetricDistributionV1, ProgSuiteNativeAudioSurvivalPanelErrorV1> {
    let values: Vec<f64> = values.collect();
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(ProgSuiteNativeAudioSurvivalPanelErrorV1::NonFiniteSummary);
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if !min.is_finite() || !max.is_finite() || !mean.is_finite() {
        return Err(ProgSuiteNativeAudioSurvivalPanelErrorV1::NonFiniteSummary);
    }
    Ok(ProgSuiteAudioMetricDistributionV1 { min, max, mean })
}

fn canonical_subject_id(motif_id: &str, seed: u64) -> String {
    format!("{motif_id}:seed-{seed}")
}

fn finite_nonnegative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

fn is_canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn required_evidence_nonclaims() -> Vec<ProgSuiteNativeAudioSurvivalEvidenceNonClaimV1> {
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

fn required_nonclaims() -> Vec<ProgSuiteAudioPanelNonClaimV1> {
    vec![
        ProgSuiteAudioPanelNonClaimV1::PanelDoesNotReverifyWaveformsWithoutPcm,
        ProgSuiteAudioPanelNonClaimV1::DescriptiveCountsDoNotEstablishStatisticalIndependence,
        ProgSuiteAudioPanelNonClaimV1::SurvivalFractionDoesNotEstablishAudibility,
        ProgSuiteAudioPanelNonClaimV1::SurvivalFractionDoesNotEstablishListenerPreference,
        ProgSuiteAudioPanelNonClaimV1::SurvivalFractionDoesNotEstablishArtisticQuality,
        ProgSuiteAudioPanelNonClaimV1::WholeWorkMetricsDoNotLocalizeTheEffect,
        ProgSuiteAudioPanelNonClaimV1::OneRendererDoesNotEstablishRendererGeneralization,
        ProgSuiteAudioPanelNonClaimV1::LockboxPopulationDoesNotEstablishUniversalMusicGeneralization,
        ProgSuiteAudioPanelNonClaimV1::PanelDoesNotGrantProductAuthority,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::prog_suite_contextual_harmony_audio_evidence::measure_prog_suite_native_audio_survival;
    use crate::evidence_digest::prog_suite_contextual_harmony_audio_protocol::predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1;

    fn sha(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn synthetic_records(
        protocol: &ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
    ) -> Vec<ProgSuiteNativeAudioSurvivalEvidenceV1> {
        protocol
            .source_lockbox
            .subjects
            .iter()
            .map(|subject| {
                let source = [[0.0_f32, 0.0_f32], [0.10_f32, -0.10_f32]];
                let contextual = [[0.0_f32, 0.0_f32], [0.20_f32, -0.10_f32]];
                measure_prog_suite_native_audio_survival(
                    &canonical_subject_id(&subject.motif_id, subject.plan_seed),
                    &subject.motif_id,
                    subject.plan_seed,
                    subject.intent_seed,
                    &sha(1),
                    &sha(2),
                    &source,
                    &source,
                    &contextual,
                    &contextual,
                )
                .unwrap()
            })
            .collect()
    }

    #[test]
    fn panel_requires_the_exact_64_subject_lockbox_order() {
        let protocol =
            predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1().unwrap();
        let panel = build_prog_suite_native_audio_survival_panel(
            &protocol,
            synthetic_records(&protocol),
        )
        .unwrap();
        assert_eq!(panel.records.len(), 64);
        assert_eq!(panel.summary.subject_count, 64);
        assert_eq!(panel.summary.renderer_survived_count, 64);
        assert_eq!(panel.summary.renderer_erased_count, 0);
        assert_eq!(panel.summary.no_symbolic_intervention_count, 0);
        panel.validate_serialized().unwrap();
    }

    #[test]
    fn record_reordering_or_subject_substitution_fails() {
        let protocol =
            predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1().unwrap();
        let mut records = synthetic_records(&protocol);
        records.swap(0, 1);
        assert!(matches!(
            build_prog_suite_native_audio_survival_panel(&protocol, records),
            Err(ProgSuiteNativeAudioSurvivalPanelErrorV1::SubjectIdentityMismatch { .. })
        ));
    }

    #[test]
    fn malformed_serialized_record_is_not_laundered_by_panel() {
        let protocol =
            predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1().unwrap();
        let mut records = synthetic_records(&protocol);
        records[0].difference.changed_aligned_frame_count = 0;
        assert!(matches!(
            build_prog_suite_native_audio_survival_panel(&protocol, records),
            Err(ProgSuiteNativeAudioSurvivalPanelErrorV1::InvalidRecordShape { index: 0 })
        ));
    }

    #[test]
    fn serialized_panel_summary_tampering_fails() {
        let protocol =
            predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1().unwrap();
        let mut panel = build_prog_suite_native_audio_survival_panel(
            &protocol,
            synthetic_records(&protocol),
        )
        .unwrap();
        panel.summary.renderer_survived_count = 0;
        assert_eq!(
            panel.validate_serialized(),
            Err(ProgSuiteNativeAudioSurvivalPanelErrorV1::CanonicalPanelMismatch)
        );
    }

    #[test]
    fn panel_nonclaims_preserve_the_pcm_and_perceptual_boundaries() {
        let protocol =
            predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1().unwrap();
        let panel = build_prog_suite_native_audio_survival_panel(
            &protocol,
            synthetic_records(&protocol),
        )
        .unwrap();
        assert!(panel.nonclaims.contains(
            &ProgSuiteAudioPanelNonClaimV1::PanelDoesNotReverifyWaveformsWithoutPcm
        ));
        assert!(panel.nonclaims.contains(
            &ProgSuiteAudioPanelNonClaimV1::SurvivalFractionDoesNotEstablishAudibility
        ));
        assert!(panel.nonclaims.contains(
            &ProgSuiteAudioPanelNonClaimV1::SurvivalFractionDoesNotEstablishArtisticQuality
        ));
    }
}
