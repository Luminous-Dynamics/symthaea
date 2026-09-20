// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Descriptive cross-form observatory for validated FORM-003 evidence projections.
//!
//! This layer exists so tools can inspect Sonata and ProgSuite evidence through
//! one stable descriptive surface without flattening their native artifacts or
//! ranking their evidence profiles. Every retained source adapter is validated
//! before any counts are emitted. Counts are descriptive only.

use super::prog_suite_projection::{
    ProgSuiteWorkEvidenceProjectionErrorV1, ProgSuiteWorkObligationEvidenceProjectionV1,
};
use super::sonata_projection::{
    SonataWorkEvidenceProjectionErrorV1, SonataWorkObligationEvidenceProjectionV1,
};
use super::{
    WorkObligationEvidenceDispositionV1, WorkObligationEvidencePreservationV1,
    WorkObligationEvidenceProjectionLossV1, WorkObligationEvidenceProjectionSetV1,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const WORK_OBLIGATION_EVIDENCE_OBSERVATORY_VERSION: &str =
    "melothaea-work-obligation-evidence-observatory-v1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WorkObligationEvidenceObservedSourceV1 {
    Sonata(SonataWorkObligationEvidenceProjectionV1),
    ProgSuite(ProgSuiteWorkObligationEvidenceProjectionV1),
}

impl WorkObligationEvidenceObservedSourceV1 {
    pub fn source_id(&self) -> &'static str {
        match self {
            Self::Sonata(_) => "sonata",
            Self::ProgSuite(_) => "prog-suite",
        }
    }

    fn projection(&self) -> &WorkObligationEvidenceProjectionSetV1 {
        match self {
            Self::Sonata(source) => &source.projection,
            Self::ProgSuite(source) => &source.projection,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkObligationEvidenceDescriptiveSummaryV1 {
    pub total_obligations: usize,
    pub positive_under_profile: usize,
    pub negative_under_profile: usize,
    pub indeterminate_under_profile: usize,
    pub not_measured_under_profile: usize,
    pub exact_preservation: usize,
    pub projected_preservation: usize,
    /// Number of generic records contributed by each retained source namespace.
    pub source_namespace_counts: BTreeMap<String, usize>,
    /// Number of generic records interpreted under each exact source profile.
    pub source_profile_counts: BTreeMap<String, usize>,
    /// Descriptive incidence of explicit source-specific projection losses.
    pub projection_loss_counts: BTreeMap<WorkObligationEvidenceProjectionLossV1, usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkObligationEvidenceObservatoryNonClaimV1 {
    SummaryCountsDoNotEstablishEvidenceQuality,
    SummaryCountsDoNotRankForms,
    SummaryCountsDoNotRankEvidenceProfiles,
    MorePositiveRecordsDoNotMeanBetterMusic,
    MoreExactRecordsDoNotMeanBetterEvidenceForAnotherQuestion,
    ObservatoryDoesNotResolveWorkObligations,
    ObservatoryDoesNotEstablishListenerPerception,
    ObservatoryDoesNotEstablishArtisticQuality,
    ObservatoryDoesNotGrantProductAuthority,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkObligationEvidenceObservatoryV1 {
    pub version: String,
    /// Full source-authoritative adapter artifacts, keyed by stable form ID.
    pub sources: BTreeMap<String, WorkObligationEvidenceObservedSourceV1>,
    /// Canonically rederived descriptive summaries for the same sources.
    pub summaries: BTreeMap<String, WorkObligationEvidenceDescriptiveSummaryV1>,
    pub nonclaims: Vec<WorkObligationEvidenceObservatoryNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum WorkObligationEvidenceObservatoryErrorV1 {
    WrongVersion { found: String },
    SonataSource(SonataWorkEvidenceProjectionErrorV1),
    ProgSuiteSource(ProgSuiteWorkEvidenceProjectionErrorV1),
    EmptySourceId,
    SourceKeyMismatch { key: String, source_id: String },
    DuplicateSource { source_id: String },
    NonCanonicalNonClaims,
    CanonicalObservatoryMismatch,
}

pub fn observe_work_obligation_evidence(
    sources: Vec<WorkObligationEvidenceObservedSourceV1>,
) -> Result<WorkObligationEvidenceObservatoryV1, WorkObligationEvidenceObservatoryErrorV1> {
    let mut retained = BTreeMap::new();
    let mut summaries = BTreeMap::new();

    for source in sources {
        validate_source(&source)?;
        let source_id = source.source_id().to_string();
        if source_id.trim().is_empty() {
            return Err(WorkObligationEvidenceObservatoryErrorV1::EmptySourceId);
        }
        if retained.contains_key(&source_id) {
            return Err(WorkObligationEvidenceObservatoryErrorV1::DuplicateSource {
                source_id,
            });
        }
        summaries.insert(source_id.clone(), summarize(source.projection()));
        retained.insert(source_id, source);
    }

    Ok(WorkObligationEvidenceObservatoryV1 {
        version: WORK_OBLIGATION_EVIDENCE_OBSERVATORY_VERSION.into(),
        sources: retained,
        summaries,
        nonclaims: required_nonclaims(),
    })
}

impl WorkObligationEvidenceObservatoryV1 {
    pub fn validate(&self) -> Result<(), WorkObligationEvidenceObservatoryErrorV1> {
        if self.version != WORK_OBLIGATION_EVIDENCE_OBSERVATORY_VERSION {
            return Err(WorkObligationEvidenceObservatoryErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        if self.nonclaims != required_nonclaims() {
            return Err(WorkObligationEvidenceObservatoryErrorV1::NonCanonicalNonClaims);
        }
        for (key, source) in &self.sources {
            let source_id = source.source_id();
            if key != source_id {
                return Err(WorkObligationEvidenceObservatoryErrorV1::SourceKeyMismatch {
                    key: key.clone(),
                    source_id: source_id.into(),
                });
            }
            validate_source(source)?;
        }
        let canonical =
            observe_work_obligation_evidence(self.sources.values().cloned().collect())?;
        if &canonical != self {
            return Err(WorkObligationEvidenceObservatoryErrorV1::CanonicalObservatoryMismatch);
        }
        Ok(())
    }
}

fn validate_source(
    source: &WorkObligationEvidenceObservedSourceV1,
) -> Result<(), WorkObligationEvidenceObservatoryErrorV1> {
    match source {
        WorkObligationEvidenceObservedSourceV1::Sonata(source) => source
            .validate()
            .map_err(WorkObligationEvidenceObservatoryErrorV1::SonataSource),
        WorkObligationEvidenceObservedSourceV1::ProgSuite(source) => source
            .validate()
            .map_err(WorkObligationEvidenceObservatoryErrorV1::ProgSuiteSource),
    }
}

fn summarize(
    projection: &WorkObligationEvidenceProjectionSetV1,
) -> WorkObligationEvidenceDescriptiveSummaryV1 {
    let mut summary = WorkObligationEvidenceDescriptiveSummaryV1 {
        total_obligations: projection.records.len(),
        positive_under_profile: 0,
        negative_under_profile: 0,
        indeterminate_under_profile: 0,
        not_measured_under_profile: 0,
        exact_preservation: 0,
        projected_preservation: 0,
        source_namespace_counts: BTreeMap::new(),
        source_profile_counts: BTreeMap::new(),
        projection_loss_counts: BTreeMap::new(),
    };

    for record in projection.records.values() {
        match record.disposition {
            WorkObligationEvidenceDispositionV1::PositiveEvidenceUnderProfile => {
                summary.positive_under_profile += 1;
            }
            WorkObligationEvidenceDispositionV1::NegativeEvidenceUnderProfile => {
                summary.negative_under_profile += 1;
            }
            WorkObligationEvidenceDispositionV1::IndeterminateEvidenceUnderProfile => {
                summary.indeterminate_under_profile += 1;
            }
            WorkObligationEvidenceDispositionV1::NotMeasuredUnderProfile => {
                summary.not_measured_under_profile += 1;
            }
        }
        match record.preservation {
            WorkObligationEvidencePreservationV1::Exact => summary.exact_preservation += 1,
            WorkObligationEvidencePreservationV1::Projected => {
                summary.projected_preservation += 1;
            }
        }
        *summary
            .source_namespace_counts
            .entry(record.source.namespace.clone())
            .or_default() += 1;
        *summary
            .source_profile_counts
            .entry(record.source.profile_id.clone())
            .or_default() += 1;
        for loss in &record.projection_losses {
            *summary
                .projection_loss_counts
                .entry(loss.clone())
                .or_default() += 1;
        }
    }
    summary
}

fn required_nonclaims() -> Vec<WorkObligationEvidenceObservatoryNonClaimV1> {
    vec![
        WorkObligationEvidenceObservatoryNonClaimV1::SummaryCountsDoNotEstablishEvidenceQuality,
        WorkObligationEvidenceObservatoryNonClaimV1::SummaryCountsDoNotRankForms,
        WorkObligationEvidenceObservatoryNonClaimV1::SummaryCountsDoNotRankEvidenceProfiles,
        WorkObligationEvidenceObservatoryNonClaimV1::MorePositiveRecordsDoNotMeanBetterMusic,
        WorkObligationEvidenceObservatoryNonClaimV1::MoreExactRecordsDoNotMeanBetterEvidenceForAnotherQuestion,
        WorkObligationEvidenceObservatoryNonClaimV1::ObservatoryDoesNotResolveWorkObligations,
        WorkObligationEvidenceObservatoryNonClaimV1::ObservatoryDoesNotEstablishListenerPerception,
        WorkObligationEvidenceObservatoryNonClaimV1::ObservatoryDoesNotEstablishArtisticQuality,
        WorkObligationEvidenceObservatoryNonClaimV1::ObservatoryDoesNotGrantProductAuthority,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::work_obligation_evidence::prog_suite_projection::project_prog_suite_work_obligation_evidence;
    use crate::work_obligation_evidence::sonata_projection::project_sonata_work_obligation_evidence;
    use crate::{
        Duration, Key, Motif, MusicalIntent, PitchClass, Style,
        adjudicate_prog_suite_development, bind_prog_suite_subject,
        cite_prog_suite_development_obligations, cite_prog_suite_whole_section_coverage,
        measure_prog_suite_observed_relations, measure_prog_suite_section_coverage,
        plan_prog_suite, realize_prog_suite_subject_bound, realize_sonata_with_plan,
    };

    fn motif() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ])
    }

    fn sonata_source() -> WorkObligationEvidenceObservedSourceV1 {
        let realization = realize_sonata_with_plan(
            Key::major(PitchClass::C),
            96.0,
            4.0,
            &motif(),
            11,
            &MusicalIntent::default(),
        );
        WorkObligationEvidenceObservedSourceV1::Sonata(
            project_sonata_work_obligation_evidence(&realization).unwrap(),
        )
    }

    fn prog_suite_source() -> WorkObligationEvidenceObservedSourceV1 {
        let plan = plan_prog_suite(
            Key::major(PitchClass::C),
            100.0,
            5,
            &Style::ProgFolk.spec(),
        )
        .unwrap();
        let declaration = bind_prog_suite_subject(&plan, &motif()).unwrap();
        let subject = realize_prog_suite_subject_bound(
            &declaration,
            &MusicalIntent {
                energy: 0.73,
                seed: 9182,
                ..MusicalIntent::default()
            },
        )
        .unwrap();
        let observed = measure_prog_suite_observed_relations(&subject).unwrap();
        let adjudicated = adjudicate_prog_suite_development(&observed).unwrap();
        let base = cite_prog_suite_development_obligations(&adjudicated).unwrap();
        let coverage = measure_prog_suite_section_coverage(&observed).unwrap();
        let strong = cite_prog_suite_whole_section_coverage(&base, &coverage).unwrap();
        WorkObligationEvidenceObservedSourceV1::ProgSuite(
            project_prog_suite_work_obligation_evidence(&strong).unwrap(),
        )
    }

    #[test]
    fn sonata_and_prog_suite_are_summarized_without_cross_form_score() {
        let observatory =
            observe_work_obligation_evidence(vec![sonata_source(), prog_suite_source()]).unwrap();
        assert_eq!(observatory.sources.len(), 2);
        assert_eq!(observatory.summaries["sonata"].total_obligations, 9);
        assert_eq!(observatory.summaries["prog-suite"].total_obligations, 6);
        assert_eq!(observatory.summaries["sonata"].exact_preservation, 8);
        assert_eq!(observatory.summaries["sonata"].projected_preservation, 1);
        assert_eq!(observatory.summaries["prog-suite"].projected_preservation, 5);
        observatory.validate().unwrap();
    }

    #[test]
    fn source_specific_loss_vocabularies_remain_visible() {
        let observatory =
            observe_work_obligation_evidence(vec![sonata_source(), prog_suite_source()]).unwrap();
        let sonata_losses = &observatory.summaries["sonata"].projection_loss_counts;
        assert!(
            sonata_losses
                .keys()
                .any(|loss| loss.code == "cadential-specificity-to-tonal-center")
        );
        let prog_losses = &observatory.summaries["prog-suite"].projection_loss_counts;
        assert!(
            prog_losses
                .keys()
                .any(|loss| loss.code == "thematic-relation-anchored-at-first-statement")
        );
        assert!(
            prog_losses
                .keys()
                .any(|loss| loss.code == "tonic-anchor-proxy-not-full-tonal-analysis")
        );
    }

    #[test]
    fn duplicate_source_kind_is_rejected() {
        assert_eq!(
            observe_work_obligation_evidence(vec![sonata_source(), sonata_source()]),
            Err(WorkObligationEvidenceObservatoryErrorV1::DuplicateSource {
                source_id: "sonata".into(),
            })
        );
    }

    #[test]
    fn hand_edited_summary_fails_canonical_validation() {
        let mut observatory = observe_work_obligation_evidence(vec![sonata_source()]).unwrap();
        observatory
            .summaries
            .get_mut("sonata")
            .unwrap()
            .positive_under_profile += 1;
        assert_eq!(
            observatory.validate(),
            Err(WorkObligationEvidenceObservatoryErrorV1::CanonicalObservatoryMismatch)
        );
    }

    #[test]
    fn nonclaims_forbid_rank_and_quality_inference() {
        let observatory = observe_work_obligation_evidence(vec![sonata_source()]).unwrap();
        assert!(observatory.nonclaims.contains(
            &WorkObligationEvidenceObservatoryNonClaimV1::SummaryCountsDoNotRankForms
        ));
        assert!(observatory.nonclaims.contains(
            &WorkObligationEvidenceObservatoryNonClaimV1::SummaryCountsDoNotRankEvidenceProfiles
        ));
        assert!(observatory.nonclaims.contains(
            &WorkObligationEvidenceObservatoryNonClaimV1::MorePositiveRecordsDoNotMeanBetterMusic
        ));
    }
}
