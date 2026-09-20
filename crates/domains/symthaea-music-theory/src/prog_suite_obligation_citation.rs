// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Profile-relative evidence citations for ProgSuite FORM-003 obligations.
//!
//! [`crate::work_obligation::WorkObligationPlanV2`] is immutable prospective
//! memory: it records what a work promises and deliberately stores no mutable
//! resolution state. [`crate::prog_suite_development_adjudication`] likewise
//! stops at profile-relative admission. This module connects those two layers
//! without pretending that a citation is itself obligation resolution.
//!
//! A positive citation means only that the exact first-statement symbolic
//! profile admitted evidence relevant to the exact declared thematic-derivation
//! obligation. A future resolver may consume that citation under an explicit
//! policy; this module neither supplies such a policy nor mutates FORM-003.

use crate::development_program::DevelopmentEvidenceRequirementV1;
use crate::prog_suite_development_adjudication::{
    PROG_SUITE_FIRST_STATEMENT_EVIDENCE_PROFILE, ProgSuiteDevelopmentAdjudicationErrorV1,
    ProgSuiteDevelopmentAdjudicationV1, ProgSuiteDevelopmentDispositionV1,
    adjudicate_prog_suite_development,
};
use crate::rhythm::Duration;
use crate::work_obligation::WorkObligationKindV2;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

pub const PROG_SUITE_OBLIGATION_CITATION_VERSION: &str =
    "melothaea-prog-suite-obligation-citation-v1";
pub const PROG_SUITE_OBLIGATION_CITATION_PROFILE: &str =
    "melothaea-prog-suite-first-statement-obligation-citation-profile-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteObligationCitationDispositionV1 {
    /// Positive evidence exists and is citable under the exact retained profile.
    PositiveEvidenceCitationUnderProfile,
    /// The exact profile measured the relation and rejected it.
    NegativeEvidenceCitationUnderProfile,
    /// The exact profile could not determine the relation from available material.
    IndeterminateEvidenceCitationUnderProfile,
}

/// Information intentionally lost when profile-relative stage admission is
/// projected into an evidence citation about a broader FORM-003 obligation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteObligationCitationProjectionLossV1 {
    /// The source profile observes the first thematic statement, not every
    /// statement or event in the complete due-context section.
    FirstStatementCoverageOnly,
    /// Admission is evidence applicability, not a universal obligation-resolution
    /// theorem and not mutable obligation state.
    AdmissionDoesNotEqualResolution,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteObligationCitationNonClaimV1 {
    CitationDoesNotResolveWorkObligation,
    CitationDoesNotMutateWorkObligationPlan,
    PositiveCitationDoesNotEstablishUniversalSatisfaction,
    DoesNotEstablishWholeSectionExactTransformation,
    DoesNotEstablishHistoricalExecutionTrace,
    DoesNotEstablishListenerPerception,
    DoesNotEstablishArtisticQuality,
    DoesNotGrantProductAuthority,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteObligationEvidenceCitationRecordV1 {
    pub stage_id: String,
    pub obligation_id: String,
    pub derivation_id: String,
    pub due_context: String,
    pub due_earliest: Duration,
    pub due_latest: Duration,
    pub stage_end: Duration,
    pub required_evidence: DevelopmentEvidenceRequirementV1,
    pub source_evidence_profile_id: String,
    pub development_disposition: ProgSuiteDevelopmentDispositionV1,
    pub citation_disposition: ProgSuiteObligationCitationDispositionV1,
    pub projection_losses: Vec<ProgSuiteObligationCitationProjectionLossV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteObligationEvidenceCitationSetV1 {
    pub version: String,
    pub citation_profile_id: String,
    /// Canonical profile-relative development admission being cited. This is
    /// retained whole so citation authority cannot outlive or obscure its source.
    pub adjudication: ProgSuiteDevelopmentAdjudicationV1,
    /// Exactly one record for each development stage / thematic-derivation
    /// obligation represented by the ordered development program.
    pub records: Vec<ProgSuiteObligationEvidenceCitationRecordV1>,
    pub nonclaims: Vec<ProgSuiteObligationCitationNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteObligationCitationErrorV1 {
    WrongVersion { found: String },
    WrongCitationProfile { found: String },
    SourceAdjudication(ProgSuiteDevelopmentAdjudicationErrorV1),
    SourceAdjudicationMismatch,
    StageCountMismatch { program: usize, adjudicated: usize },
    MissingProgramStage { stage_id: String },
    StageIdentityMismatch { stage_id: String },
    MissingObligation { stage_id: String, obligation_id: String },
    ObligationKindMismatch { stage_id: String, obligation_id: String },
    ObligationDueContextMismatch { stage_id: String, obligation_id: String },
    StageEndOutsideDueWindow { stage_id: String, obligation_id: String },
    NonCanonicalProjectionLosses { stage_id: String },
    NonCanonicalNonClaims,
    CanonicalCitationMismatch,
}

pub fn cite_prog_suite_development_obligations(
    adjudication: &ProgSuiteDevelopmentAdjudicationV1,
) -> Result<ProgSuiteObligationEvidenceCitationSetV1, ProgSuiteObligationCitationErrorV1> {
    // Never trust serialized admission/disposition fields. Re-adjudicate from
    // the retained observation substrate and require exact identity first.
    let canonical = adjudicate_prog_suite_development(&adjudication.observed_relations)
        .map_err(ProgSuiteObligationCitationErrorV1::SourceAdjudication)?;
    if &canonical != adjudication {
        return Err(ProgSuiteObligationCitationErrorV1::SourceAdjudicationMismatch);
    }

    let program = &adjudication
        .observed_relations
        .subject
        .declaration
        .development_program
        .program;
    let obligation_plan = &adjudication
        .observed_relations
        .subject
        .declaration
        .work_binding
        .obligation_plan;

    if program.stages.len() != adjudication.stages.len() {
        return Err(ProgSuiteObligationCitationErrorV1::StageCountMismatch {
            program: program.stages.len(),
            adjudicated: adjudication.stages.len(),
        });
    }

    let mut records = Vec::with_capacity(program.stages.len());
    for admitted in &adjudication.stages {
        let stage = program
            .stages
            .iter()
            .find(|stage| stage.stage_id == admitted.stage_id)
            .ok_or_else(|| ProgSuiteObligationCitationErrorV1::MissingProgramStage {
                stage_id: admitted.stage_id.clone(),
            })?;
        if stage.derivation_id != admitted.derivation_id
            || stage.obligation_id != admitted.obligation_id
            || stage.evidence_requirement != admitted.required_evidence
        {
            return Err(ProgSuiteObligationCitationErrorV1::StageIdentityMismatch {
                stage_id: admitted.stage_id.clone(),
            });
        }

        let obligation = obligation_plan
            .obligations
            .get(&stage.obligation_id)
            .ok_or_else(|| ProgSuiteObligationCitationErrorV1::MissingObligation {
                stage_id: stage.stage_id.clone(),
                obligation_id: stage.obligation_id.clone(),
            })?;
        if !matches!(
            &obligation.kind,
            WorkObligationKindV2::RealizeThematicDerivation { derivation_id }
                if derivation_id == &stage.derivation_id
        ) {
            return Err(ProgSuiteObligationCitationErrorV1::ObligationKindMismatch {
                stage_id: stage.stage_id.clone(),
                obligation_id: stage.obligation_id.clone(),
            });
        }
        if obligation.due_context != stage.work_node_id {
            return Err(
                ProgSuiteObligationCitationErrorV1::ObligationDueContextMismatch {
                    stage_id: stage.stage_id.clone(),
                    obligation_id: stage.obligation_id.clone(),
                },
            );
        }
        if compare_duration(stage.end, obligation.due.earliest) == Ordering::Less
            || compare_duration(stage.end, obligation.due.latest) == Ordering::Greater
        {
            return Err(ProgSuiteObligationCitationErrorV1::StageEndOutsideDueWindow {
                stage_id: stage.stage_id.clone(),
                obligation_id: stage.obligation_id.clone(),
            });
        }

        let citation_disposition = match admitted.disposition {
            ProgSuiteDevelopmentDispositionV1::AdmittedObservedUnderProfile => {
                ProgSuiteObligationCitationDispositionV1::PositiveEvidenceCitationUnderProfile
            }
            ProgSuiteDevelopmentDispositionV1::RejectedObservedUnderProfile => {
                ProgSuiteObligationCitationDispositionV1::NegativeEvidenceCitationUnderProfile
            }
            ProgSuiteDevelopmentDispositionV1::IndeterminateInsufficientMaterial => {
                ProgSuiteObligationCitationDispositionV1::IndeterminateEvidenceCitationUnderProfile
            }
        };

        records.push(ProgSuiteObligationEvidenceCitationRecordV1 {
            stage_id: stage.stage_id.clone(),
            obligation_id: stage.obligation_id.clone(),
            derivation_id: stage.derivation_id.clone(),
            due_context: obligation.due_context.clone(),
            due_earliest: obligation.due.earliest,
            due_latest: obligation.due.latest,
            stage_end: stage.end,
            required_evidence: stage.evidence_requirement,
            source_evidence_profile_id: PROG_SUITE_FIRST_STATEMENT_EVIDENCE_PROFILE.into(),
            development_disposition: admitted.disposition,
            citation_disposition,
            projection_losses: required_projection_losses(),
        });
    }

    Ok(ProgSuiteObligationEvidenceCitationSetV1 {
        version: PROG_SUITE_OBLIGATION_CITATION_VERSION.into(),
        citation_profile_id: PROG_SUITE_OBLIGATION_CITATION_PROFILE.into(),
        adjudication: adjudication.clone(),
        records,
        nonclaims: required_nonclaims(),
    })
}

impl ProgSuiteObligationEvidenceCitationSetV1 {
    pub fn validate(&self) -> Result<(), ProgSuiteObligationCitationErrorV1> {
        if self.version != PROG_SUITE_OBLIGATION_CITATION_VERSION {
            return Err(ProgSuiteObligationCitationErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        if self.citation_profile_id != PROG_SUITE_OBLIGATION_CITATION_PROFILE {
            return Err(ProgSuiteObligationCitationErrorV1::WrongCitationProfile {
                found: self.citation_profile_id.clone(),
            });
        }
        for record in &self.records {
            if record.projection_losses != required_projection_losses() {
                return Err(
                    ProgSuiteObligationCitationErrorV1::NonCanonicalProjectionLosses {
                        stage_id: record.stage_id.clone(),
                    },
                );
            }
        }
        if self.nonclaims != required_nonclaims() {
            return Err(ProgSuiteObligationCitationErrorV1::NonCanonicalNonClaims);
        }
        let canonical = cite_prog_suite_development_obligations(&self.adjudication)?;
        if &canonical != self {
            return Err(ProgSuiteObligationCitationErrorV1::CanonicalCitationMismatch);
        }
        Ok(())
    }

    /// Convenience query only. It does not resolve obligations or mutate any
    /// FORM-003 state.
    pub fn all_development_obligations_have_positive_citations(&self) -> bool {
        self.records.iter().all(|record| {
            record.citation_disposition
                == ProgSuiteObligationCitationDispositionV1::PositiveEvidenceCitationUnderProfile
        })
    }
}

fn compare_duration(left: Duration, right: Duration) -> Ordering {
    (i128::from(left.num()) * i128::from(right.den()))
        .cmp(&(i128::from(right.num()) * i128::from(left.den())))
}

fn required_projection_losses() -> Vec<ProgSuiteObligationCitationProjectionLossV1> {
    vec![
        ProgSuiteObligationCitationProjectionLossV1::FirstStatementCoverageOnly,
        ProgSuiteObligationCitationProjectionLossV1::AdmissionDoesNotEqualResolution,
    ]
}

fn required_nonclaims() -> Vec<ProgSuiteObligationCitationNonClaimV1> {
    vec![
        ProgSuiteObligationCitationNonClaimV1::CitationDoesNotResolveWorkObligation,
        ProgSuiteObligationCitationNonClaimV1::CitationDoesNotMutateWorkObligationPlan,
        ProgSuiteObligationCitationNonClaimV1::PositiveCitationDoesNotEstablishUniversalSatisfaction,
        ProgSuiteObligationCitationNonClaimV1::DoesNotEstablishWholeSectionExactTransformation,
        ProgSuiteObligationCitationNonClaimV1::DoesNotEstablishHistoricalExecutionTrace,
        ProgSuiteObligationCitationNonClaimV1::DoesNotEstablishListenerPerception,
        ProgSuiteObligationCitationNonClaimV1::DoesNotEstablishArtisticQuality,
        ProgSuiteObligationCitationNonClaimV1::DoesNotGrantProductAuthority,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Duration, Key, Motif, MusicalIntent, Pitch, PitchClass, Style, VoiceRole,
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

    fn adjudication_with(motif: &Motif) -> ProgSuiteDevelopmentAdjudicationV1 {
        let plan = plan_prog_suite(
            Key::major(PitchClass::C),
            100.0,
            5,
            &Style::ProgFolk.spec(),
        )
        .unwrap();
        let declaration = bind_prog_suite_subject(&plan, motif).unwrap();
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
        adjudicate_prog_suite_development(&observed).unwrap()
    }

    #[test]
    fn canonical_development_admissions_become_profile_relative_obligation_citations() {
        let citations = cite_prog_suite_development_obligations(&adjudication_with(&motif())).unwrap();
        assert_eq!(citations.records.len(), 3);
        assert!(citations.all_development_obligations_have_positive_citations());
        for record in &citations.records {
            assert_eq!(
                record.citation_disposition,
                ProgSuiteObligationCitationDispositionV1::PositiveEvidenceCitationUnderProfile
            );
            assert_eq!(record.projection_losses, required_projection_losses());
            assert_ne!(compare_duration(record.stage_end, record.due_earliest), Ordering::Less);
            assert_ne!(compare_duration(record.stage_end, record.due_latest), Ordering::Greater);
        }
        citations.validate().unwrap();
    }

    #[test]
    fn damaged_completed_score_becomes_negative_evidence_not_resolution() {
        let mut adjudication = adjudication_with(&motif());
        let b_start = adjudication
            .observed_relations
            .subject
            .declaration
            .work_binding
            .native_plan
            .sections[1]
            .start;
        let note = adjudication
            .observed_relations
            .subject
            .realization
            .score
            .notes
            .iter_mut()
            .find(|note| note.role == VoiceRole::Melody && note.onset.beats() >= b_start.beats())
            .unwrap();
        note.pitch = Pitch::from_midi(note.pitch.midi().saturating_add(6).min(127));
        let observed = measure_prog_suite_observed_relations(&adjudication.observed_relations.subject)
            .unwrap();
        adjudication = adjudicate_prog_suite_development(&observed).unwrap();

        let citations = cite_prog_suite_development_obligations(&adjudication).unwrap();
        assert_eq!(
            citations.records[0].citation_disposition,
            ProgSuiteObligationCitationDispositionV1::NegativeEvidenceCitationUnderProfile
        );
        assert!(!citations.all_development_obligations_have_positive_citations());
    }

    #[test]
    fn insufficient_material_remains_indeterminate_at_obligation_boundary() {
        let one_note = Motif::from_degrees(&[(1, Duration::quarter())]);
        let citations = cite_prog_suite_development_obligations(&adjudication_with(&one_note)).unwrap();
        assert!(citations.records.iter().all(|record| {
            record.citation_disposition
                == ProgSuiteObligationCitationDispositionV1::IndeterminateEvidenceCitationUnderProfile
        }));
    }

    #[test]
    fn serialized_positive_citation_cannot_be_manufactured() {
        let mut citations = cite_prog_suite_development_obligations(&adjudication_with(&motif())).unwrap();
        citations.records[0].citation_disposition =
            ProgSuiteObligationCitationDispositionV1::NegativeEvidenceCitationUnderProfile;
        assert_eq!(
            citations.validate(),
            Err(ProgSuiteObligationCitationErrorV1::CanonicalCitationMismatch)
        );
    }

    #[test]
    fn projection_loss_and_nonclaim_boundaries_are_exact() {
        let citations = cite_prog_suite_development_obligations(&adjudication_with(&motif())).unwrap();
        assert_eq!(citations.nonclaims, required_nonclaims());
        assert!(citations.nonclaims.contains(
            &ProgSuiteObligationCitationNonClaimV1::CitationDoesNotResolveWorkObligation
        ));
        assert!(citations.records.iter().all(|record| {
            record.projection_losses
                .contains(&ProgSuiteObligationCitationProjectionLossV1::FirstStatementCoverageOnly)
        }));
    }
}
