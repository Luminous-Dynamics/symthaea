// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stronger FORM-003 citations requiring both relation admission and complete
//! ProgSuite section projection coverage.

use crate::prog_suite_obligation_citation::{
    ProgSuiteObligationCitationDispositionV1, ProgSuiteObligationCitationErrorV1,
    ProgSuiteObligationEvidenceCitationSetV1, cite_prog_suite_development_obligations,
};
use crate::prog_suite_section_coverage::{
    ProgSuiteSectionCoverageErrorV1, ProgSuiteSectionCoverageEvidenceV1,
    ProgSuiteSectionCoverageStatusV1, measure_prog_suite_section_coverage,
};
use serde::{Deserialize, Serialize};

pub const PROG_SUITE_COVERAGE_CITATION_VERSION: &str =
    "melothaea-prog-suite-coverage-citation-v1";
pub const PROG_SUITE_COVERAGE_CITATION_PROFILE: &str =
    "melothaea-prog-suite-whole-section-obligation-citation-profile-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteCoverageCitationDispositionV1 {
    PositiveEvidenceCitationUnderWholeSectionProfile,
    NegativeEvidenceCitationUnderWholeSectionProfile,
    IndeterminateEvidenceCitationUnderWholeSectionProfile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteCoverageCitationProjectionLossV1 {
    /// Ordered thematic identity is established at the first statement; later
    /// material is covered as canonical development/projection rather than
    /// independently re-proving the source transformation at every statement.
    ThematicRelationAnchoredAtFirstStatement,
    AdmissionDoesNotEqualResolution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteCoverageCitationNonClaimV1 {
    CitationDoesNotResolveWorkObligation,
    CitationDoesNotMutateWorkObligationPlan,
    WholeSectionCoverageDoesNotMeanWholeSectionExactTransformation,
    DoesNotEstablishHistoricalExecutionTrace,
    DoesNotEstablishListenerPerception,
    DoesNotEstablishArtisticQuality,
    DoesNotGrantProductAuthority,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteCoverageCitationRecordV1 {
    pub stage_id: String,
    pub obligation_id: String,
    pub base_citation_disposition: ProgSuiteObligationCitationDispositionV1,
    pub section_coverage_status: ProgSuiteSectionCoverageStatusV1,
    pub disposition: ProgSuiteCoverageCitationDispositionV1,
    pub projection_losses: Vec<ProgSuiteCoverageCitationProjectionLossV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteCoverageCitationSetV1 {
    pub version: String,
    pub profile_id: String,
    pub base_citations: ProgSuiteObligationEvidenceCitationSetV1,
    pub section_coverage: ProgSuiteSectionCoverageEvidenceV1,
    pub records: Vec<ProgSuiteCoverageCitationRecordV1>,
    pub nonclaims: Vec<ProgSuiteCoverageCitationNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteCoverageCitationErrorV1 {
    WrongVersion { found: String },
    WrongProfile { found: String },
    BaseCitation(ProgSuiteObligationCitationErrorV1),
    SectionCoverage(ProgSuiteSectionCoverageErrorV1),
    BaseCitationMismatch,
    SectionCoverageMismatch,
    SubjectMismatch,
    RecordCountMismatch { citations: usize, coverage: usize },
    MissingCoverageRecord { stage_id: String },
    EvidenceStateConflict { stage_id: String },
    NonCanonicalProjectionLosses { stage_id: String },
    NonCanonicalNonClaims,
    CanonicalCitationMismatch,
}

pub fn cite_prog_suite_whole_section_coverage(
    base_citations: &ProgSuiteObligationEvidenceCitationSetV1,
    section_coverage: &ProgSuiteSectionCoverageEvidenceV1,
) -> Result<ProgSuiteCoverageCitationSetV1, ProgSuiteCoverageCitationErrorV1> {
    let canonical_base = cite_prog_suite_development_obligations(&base_citations.adjudication)
        .map_err(ProgSuiteCoverageCitationErrorV1::BaseCitation)?;
    if &canonical_base != base_citations {
        return Err(ProgSuiteCoverageCitationErrorV1::BaseCitationMismatch);
    }
    let canonical_coverage = measure_prog_suite_section_coverage(&section_coverage.observed_relations)
        .map_err(ProgSuiteCoverageCitationErrorV1::SectionCoverage)?;
    if &canonical_coverage != section_coverage {
        return Err(ProgSuiteCoverageCitationErrorV1::SectionCoverageMismatch);
    }
    if base_citations.adjudication.observed_relations.subject
        != section_coverage.observed_relations.subject
    {
        return Err(ProgSuiteCoverageCitationErrorV1::SubjectMismatch);
    }
    if base_citations.records.len() != section_coverage.records.len() {
        return Err(ProgSuiteCoverageCitationErrorV1::RecordCountMismatch {
            citations: base_citations.records.len(),
            coverage: section_coverage.records.len(),
        });
    }

    let mut records = Vec::with_capacity(base_citations.records.len());
    for base in &base_citations.records {
        let coverage = section_coverage
            .records
            .iter()
            .find(|record| record.stage_id == base.stage_id)
            .ok_or_else(|| ProgSuiteCoverageCitationErrorV1::MissingCoverageRecord {
                stage_id: base.stage_id.clone(),
            })?;

        let disposition = match (base.citation_disposition, coverage.status) {
            (
                ProgSuiteObligationCitationDispositionV1::PositiveEvidenceCitationUnderProfile,
                ProgSuiteSectionCoverageStatusV1::CoveredUnderDeclaredAdaptation,
            ) => ProgSuiteCoverageCitationDispositionV1::PositiveEvidenceCitationUnderWholeSectionProfile,
            (
                ProgSuiteObligationCitationDispositionV1::NegativeEvidenceCitationUnderProfile,
                ProgSuiteSectionCoverageStatusV1::SourceRelationRejected,
            )
            | (
                ProgSuiteObligationCitationDispositionV1::PositiveEvidenceCitationUnderProfile,
                ProgSuiteSectionCoverageStatusV1::SectionProjectionFailed,
            ) => ProgSuiteCoverageCitationDispositionV1::NegativeEvidenceCitationUnderWholeSectionProfile,
            (
                ProgSuiteObligationCitationDispositionV1::IndeterminateEvidenceCitationUnderProfile,
                ProgSuiteSectionCoverageStatusV1::IndeterminateInsufficientMaterial,
            ) => ProgSuiteCoverageCitationDispositionV1::IndeterminateEvidenceCitationUnderWholeSectionProfile,
            _ => {
                return Err(ProgSuiteCoverageCitationErrorV1::EvidenceStateConflict {
                    stage_id: base.stage_id.clone(),
                });
            }
        };

        records.push(ProgSuiteCoverageCitationRecordV1 {
            stage_id: base.stage_id.clone(),
            obligation_id: base.obligation_id.clone(),
            base_citation_disposition: base.citation_disposition,
            section_coverage_status: coverage.status,
            disposition,
            projection_losses: required_projection_losses(),
        });
    }

    Ok(ProgSuiteCoverageCitationSetV1 {
        version: PROG_SUITE_COVERAGE_CITATION_VERSION.into(),
        profile_id: PROG_SUITE_COVERAGE_CITATION_PROFILE.into(),
        base_citations: base_citations.clone(),
        section_coverage: section_coverage.clone(),
        records,
        nonclaims: required_nonclaims(),
    })
}

impl ProgSuiteCoverageCitationSetV1 {
    pub fn validate(&self) -> Result<(), ProgSuiteCoverageCitationErrorV1> {
        if self.version != PROG_SUITE_COVERAGE_CITATION_VERSION {
            return Err(ProgSuiteCoverageCitationErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        if self.profile_id != PROG_SUITE_COVERAGE_CITATION_PROFILE {
            return Err(ProgSuiteCoverageCitationErrorV1::WrongProfile {
                found: self.profile_id.clone(),
            });
        }
        for record in &self.records {
            if record.projection_losses != required_projection_losses() {
                return Err(ProgSuiteCoverageCitationErrorV1::NonCanonicalProjectionLosses {
                    stage_id: record.stage_id.clone(),
                });
            }
        }
        if self.nonclaims != required_nonclaims() {
            return Err(ProgSuiteCoverageCitationErrorV1::NonCanonicalNonClaims);
        }
        let canonical = cite_prog_suite_whole_section_coverage(
            &self.base_citations,
            &self.section_coverage,
        )?;
        if &canonical != self {
            return Err(ProgSuiteCoverageCitationErrorV1::CanonicalCitationMismatch);
        }
        Ok(())
    }

    pub fn all_positive_under_whole_section_profile(&self) -> bool {
        self.records.iter().all(|record| {
            record.disposition
                == ProgSuiteCoverageCitationDispositionV1::PositiveEvidenceCitationUnderWholeSectionProfile
        })
    }
}

fn required_projection_losses() -> Vec<ProgSuiteCoverageCitationProjectionLossV1> {
    vec![
        ProgSuiteCoverageCitationProjectionLossV1::ThematicRelationAnchoredAtFirstStatement,
        ProgSuiteCoverageCitationProjectionLossV1::AdmissionDoesNotEqualResolution,
    ]
}

fn required_nonclaims() -> Vec<ProgSuiteCoverageCitationNonClaimV1> {
    vec![
        ProgSuiteCoverageCitationNonClaimV1::CitationDoesNotResolveWorkObligation,
        ProgSuiteCoverageCitationNonClaimV1::CitationDoesNotMutateWorkObligationPlan,
        ProgSuiteCoverageCitationNonClaimV1::WholeSectionCoverageDoesNotMeanWholeSectionExactTransformation,
        ProgSuiteCoverageCitationNonClaimV1::DoesNotEstablishHistoricalExecutionTrace,
        ProgSuiteCoverageCitationNonClaimV1::DoesNotEstablishListenerPerception,
        ProgSuiteCoverageCitationNonClaimV1::DoesNotEstablishArtisticQuality,
        ProgSuiteCoverageCitationNonClaimV1::DoesNotGrantProductAuthority,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Duration, Key, Motif, MusicalIntent, PitchClass, Style,
        adjudicate_prog_suite_development, bind_prog_suite_subject,
        measure_prog_suite_observed_relations, plan_prog_suite,
        realize_prog_suite_subject_bound,
    };

    fn motif() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ])
    }

    fn sources() -> (
        ProgSuiteObligationEvidenceCitationSetV1,
        ProgSuiteSectionCoverageEvidenceV1,
    ) {
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
        let citations = cite_prog_suite_development_obligations(&adjudicated).unwrap();
        let coverage = measure_prog_suite_section_coverage(&observed).unwrap();
        (citations, coverage)
    }

    #[test]
    fn canonical_subject_is_positive_only_when_both_axes_are_positive() {
        let (citations, coverage) = sources();
        let stronger = cite_prog_suite_whole_section_coverage(&citations, &coverage).unwrap();
        assert!(stronger.all_positive_under_whole_section_profile());
        stronger.validate().unwrap();
    }

    #[test]
    fn later_section_damage_cannot_be_hidden_by_positive_first_statement_citation() {
        let (citations, mut coverage) = sources();
        coverage.records[0].status = ProgSuiteSectionCoverageStatusV1::SectionProjectionFailed;
        assert_eq!(
            cite_prog_suite_whole_section_coverage(&citations, &coverage),
            Err(ProgSuiteCoverageCitationErrorV1::SectionCoverageMismatch)
        );
    }

    #[test]
    fn projection_loss_is_narrower_than_first_statement_only() {
        let (citations, coverage) = sources();
        let stronger = cite_prog_suite_whole_section_coverage(&citations, &coverage).unwrap();
        assert!(stronger.records.iter().all(|record| {
            record.projection_losses
                == vec![
                    ProgSuiteCoverageCitationProjectionLossV1::ThematicRelationAnchoredAtFirstStatement,
                    ProgSuiteCoverageCitationProjectionLossV1::AdmissionDoesNotEqualResolution,
                ]
        }));
    }

    #[test]
    fn nonclaim_boundary_remains_exact() {
        let (citations, coverage) = sources();
        let stronger = cite_prog_suite_whole_section_coverage(&citations, &coverage).unwrap();
        assert_eq!(stronger.nonclaims, required_nonclaims());
        assert!(stronger.nonclaims.contains(
            &ProgSuiteCoverageCitationNonClaimV1::CitationDoesNotResolveWorkObligation
        ));
    }
}
