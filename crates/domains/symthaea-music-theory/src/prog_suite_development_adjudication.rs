// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Profile-relative admission of ProgSuite observed-development evidence.
//!
//! `DevelopmentProgramV1` declares what symbolic evidence each development
//! stage requires. `ProgSuiteObservedRelationEvidenceV1` measures what survives
//! through the first thematic statement and into the completed score. This
//! module binds those contracts without widening either one's authority.
//!
//! A positive disposition is deliberately named `AdmittedObservedUnderProfile`:
//! it means the stage's declared relation is observed under this exact first-
//! statement symbolic profile. It does not mean the whole section is an exact
//! transformation, the FORM-003 obligation has been globally resolved, a human
//! heard the relation, or the music is artistically good.

use crate::development_program::DevelopmentEvidenceRequirementV1;
use crate::prog_suite_observed_relation::{
    ProgSuiteObservedRelationErrorV1, ProgSuiteObservedRelationEvidenceV1,
    ProgSuiteObservedRelationStatusV1, measure_prog_suite_observed_relations,
};
use serde::{Deserialize, Serialize};

pub const PROG_SUITE_DEVELOPMENT_ADJUDICATION_VERSION: &str =
    "melothaea-prog-suite-development-adjudication-v1";
pub const PROG_SUITE_FIRST_STATEMENT_EVIDENCE_PROFILE: &str =
    "melothaea-prog-suite-first-statement-observation-profile-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteDevelopmentDispositionV1 {
    AdmittedObservedUnderProfile,
    RejectedObservedUnderProfile,
    IndeterminateInsufficientMaterial,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteDevelopmentAdjudicationNonClaimV1 {
    AdmissionIsProfileRelativeNotUniversalProof,
    DoesNotEstablishWholeSectionExactTransformation,
    DoesNotResolveOrMutateWorkObligationState,
    DoesNotEstablishHistoricalExecutionTrace,
    DoesNotEstablishListenerPerception,
    DoesNotEstablishArtisticQuality,
    DoesNotGrantProductAuthority,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteDevelopmentStageAdjudicationV1 {
    pub stage_id: String,
    pub derivation_id: String,
    pub obligation_id: String,
    pub required_evidence: DevelopmentEvidenceRequirementV1,
    pub operation_count: usize,
    pub observed_status: ProgSuiteObservedRelationStatusV1,
    pub disposition: ProgSuiteDevelopmentDispositionV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteDevelopmentAdjudicationV1 {
    pub version: String,
    pub profile_id: String,
    /// Exact observation substrate being adjudicated. Replay evidence is not a
    /// prerequisite and is not imported into this authority class.
    pub observed_relations: ProgSuiteObservedRelationEvidenceV1,
    pub stages: Vec<ProgSuiteDevelopmentStageAdjudicationV1>,
    pub nonclaims: Vec<ProgSuiteDevelopmentAdjudicationNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteDevelopmentAdjudicationErrorV1 {
    WrongVersion { found: String },
    WrongProfile { found: String },
    ObservedRelation(ProgSuiteObservedRelationErrorV1),
    StageCountMismatch { program: usize, observed: usize },
    MissingObservedStage { stage_id: String },
    StageIdentityMismatch { stage_id: String },
    RequirementShapeMismatch {
        stage_id: String,
        requirement: DevelopmentEvidenceRequirementV1,
        operation_count: usize,
    },
    NonCanonicalNonClaims,
    CanonicalAdjudicationMismatch,
}

pub fn adjudicate_prog_suite_development(
    observed_relations: &ProgSuiteObservedRelationEvidenceV1,
) -> Result<ProgSuiteDevelopmentAdjudicationV1, ProgSuiteDevelopmentAdjudicationErrorV1> {
    // Re-derive the entire observation substrate rather than trusting a caller's
    // serialized status fields. This keeps adjudication downstream of exact
    // canonical observation identity.
    let canonical_observed = measure_prog_suite_observed_relations(&observed_relations.subject)
        .map_err(ProgSuiteDevelopmentAdjudicationErrorV1::ObservedRelation)?;
    if &canonical_observed != observed_relations {
        return Err(ProgSuiteDevelopmentAdjudicationErrorV1::ObservedRelation(
            ProgSuiteObservedRelationErrorV1::CanonicalEvidenceMismatch,
        ));
    }

    let program = &observed_relations
        .subject
        .declaration
        .development_program
        .program;
    if program.stages.len() != observed_relations.stages.len() {
        return Err(ProgSuiteDevelopmentAdjudicationErrorV1::StageCountMismatch {
            program: program.stages.len(),
            observed: observed_relations.stages.len(),
        });
    }

    let mut stages = Vec::with_capacity(program.stages.len());
    for stage in &program.stages {
        let observed = observed_relations
            .stages
            .iter()
            .find(|record| record.stage_id == stage.stage_id)
            .ok_or_else(|| ProgSuiteDevelopmentAdjudicationErrorV1::MissingObservedStage {
                stage_id: stage.stage_id.clone(),
            })?;
        if observed.derivation_id != stage.derivation_id
            || observed.work_node_id != stage.work_node_id
        {
            return Err(ProgSuiteDevelopmentAdjudicationErrorV1::StageIdentityMismatch {
                stage_id: stage.stage_id.clone(),
            });
        }

        validate_requirement_shape(
            &stage.stage_id,
            stage.evidence_requirement,
            stage.operations.len(),
        )?;

        let disposition = match observed.status {
            ProgSuiteObservedRelationStatusV1::ObservedWithDeclaredAdaptation => {
                ProgSuiteDevelopmentDispositionV1::AdmittedObservedUnderProfile
            }
            ProgSuiteObservedRelationStatusV1::InsufficientMaterial => {
                ProgSuiteDevelopmentDispositionV1::IndeterminateInsufficientMaterial
            }
            ProgSuiteObservedRelationStatusV1::OperationMechanismMismatch
            | ProgSuiteObservedRelationStatusV1::PhraseEventAlignmentFailed
            | ProgSuiteObservedRelationStatusV1::ScoreProjectionFailed => {
                ProgSuiteDevelopmentDispositionV1::RejectedObservedUnderProfile
            }
        };

        stages.push(ProgSuiteDevelopmentStageAdjudicationV1 {
            stage_id: stage.stage_id.clone(),
            derivation_id: stage.derivation_id.clone(),
            obligation_id: stage.obligation_id.clone(),
            required_evidence: stage.evidence_requirement,
            operation_count: stage.operations.len(),
            observed_status: observed.status,
            disposition,
        });
    }

    Ok(ProgSuiteDevelopmentAdjudicationV1 {
        version: PROG_SUITE_DEVELOPMENT_ADJUDICATION_VERSION.into(),
        profile_id: PROG_SUITE_FIRST_STATEMENT_EVIDENCE_PROFILE.into(),
        observed_relations: observed_relations.clone(),
        stages,
        nonclaims: required_nonclaims(),
    })
}

impl ProgSuiteDevelopmentAdjudicationV1 {
    pub fn validate(&self) -> Result<(), ProgSuiteDevelopmentAdjudicationErrorV1> {
        if self.version != PROG_SUITE_DEVELOPMENT_ADJUDICATION_VERSION {
            return Err(ProgSuiteDevelopmentAdjudicationErrorV1::WrongVersion {
                found: self.version.clone(),
            });
        }
        if self.profile_id != PROG_SUITE_FIRST_STATEMENT_EVIDENCE_PROFILE {
            return Err(ProgSuiteDevelopmentAdjudicationErrorV1::WrongProfile {
                found: self.profile_id.clone(),
            });
        }
        if self.nonclaims != required_nonclaims() {
            return Err(ProgSuiteDevelopmentAdjudicationErrorV1::NonCanonicalNonClaims);
        }
        let canonical = adjudicate_prog_suite_development(&self.observed_relations)?;
        if &canonical != self {
            return Err(
                ProgSuiteDevelopmentAdjudicationErrorV1::CanonicalAdjudicationMismatch,
            );
        }
        Ok(())
    }

    pub fn all_stages_admitted(&self) -> bool {
        self.stages.iter().all(|stage| {
            stage.disposition
                == ProgSuiteDevelopmentDispositionV1::AdmittedObservedUnderProfile
        })
    }
}

fn validate_requirement_shape(
    stage_id: &str,
    requirement: DevelopmentEvidenceRequirementV1,
    operation_count: usize,
) -> Result<(), ProgSuiteDevelopmentAdjudicationErrorV1> {
    let valid = match requirement {
        DevelopmentEvidenceRequirementV1::SingleOperationRelationMeasured => operation_count == 1,
        DevelopmentEvidenceRequirementV1::OrderedCompositeRelationMeasured => operation_count > 1,
    };
    if !valid {
        return Err(
            ProgSuiteDevelopmentAdjudicationErrorV1::RequirementShapeMismatch {
                stage_id: stage_id.into(),
                requirement,
                operation_count,
            },
        );
    }
    Ok(())
}

fn required_nonclaims() -> Vec<ProgSuiteDevelopmentAdjudicationNonClaimV1> {
    vec![
        ProgSuiteDevelopmentAdjudicationNonClaimV1::AdmissionIsProfileRelativeNotUniversalProof,
        ProgSuiteDevelopmentAdjudicationNonClaimV1::DoesNotEstablishWholeSectionExactTransformation,
        ProgSuiteDevelopmentAdjudicationNonClaimV1::DoesNotResolveOrMutateWorkObligationState,
        ProgSuiteDevelopmentAdjudicationNonClaimV1::DoesNotEstablishHistoricalExecutionTrace,
        ProgSuiteDevelopmentAdjudicationNonClaimV1::DoesNotEstablishListenerPerception,
        ProgSuiteDevelopmentAdjudicationNonClaimV1::DoesNotEstablishArtisticQuality,
        ProgSuiteDevelopmentAdjudicationNonClaimV1::DoesNotGrantProductAuthority,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Duration, Key, Motif, MusicalIntent, Pitch, PitchClass, Style, VoiceRole,
        bind_prog_suite_subject, measure_prog_suite_observed_relations, plan_prog_suite,
        realize_prog_suite_subject_bound,
    };

    fn subject_with(motif: &Motif) -> ProgSuiteSubjectBoundRealizationV1 {
        let plan = plan_prog_suite(
            Key::major(PitchClass::C),
            100.0,
            5,
            &Style::ProgFolk.spec(),
        )
        .unwrap();
        let declaration = bind_prog_suite_subject(&plan, motif).unwrap();
        realize_prog_suite_subject_bound(
            &declaration,
            &MusicalIntent {
                energy: 0.73,
                seed: 9182,
                ..MusicalIntent::default()
            },
        )
        .unwrap()
    }

    fn motif() -> Motif {
        Motif::from_degrees(&[
            (1, Duration::quarter()),
            (2, Duration::quarter()),
            (3, Duration::quarter()),
            (5, Duration::quarter()),
        ])
    }

    #[test]
    fn canonical_seed_five_program_is_admitted_stage_by_stage_under_profile() {
        let observed = measure_prog_suite_observed_relations(&subject_with(&motif())).unwrap();
        let adjudicated = adjudicate_prog_suite_development(&observed).unwrap();
        assert!(adjudicated.all_stages_admitted());
        assert_eq!(adjudicated.stages.len(), 3);
        assert_eq!(
            adjudicated.stages[0].required_evidence,
            DevelopmentEvidenceRequirementV1::OrderedCompositeRelationMeasured
        );
        assert_eq!(adjudicated.stages[0].operation_count, 2);
        adjudicated.validate().unwrap();
    }

    #[test]
    fn score_projection_failure_rejects_profile_admission() {
        let mut subject = subject_with(&motif());
        let b_start = subject.declaration.work_binding.native_plan.sections[1].start;
        let note = subject
            .realization
            .score
            .notes
            .iter_mut()
            .find(|note| note.role == VoiceRole::Melody && note.onset.beats() >= b_start.beats())
            .unwrap();
        note.pitch = Pitch::from_midi(note.pitch.midi().saturating_add(6).min(127));
        subject.validate_structure().unwrap();

        let observed = measure_prog_suite_observed_relations(&subject).unwrap();
        let adjudicated = adjudicate_prog_suite_development(&observed).unwrap();
        assert_eq!(
            adjudicated.stages[0].disposition,
            ProgSuiteDevelopmentDispositionV1::RejectedObservedUnderProfile
        );
        assert!(!adjudicated.all_stages_admitted());
    }

    #[test]
    fn one_note_subject_is_indeterminate_not_rejected() {
        let one_note = Motif::from_degrees(&[(1, Duration::quarter())]);
        let observed = measure_prog_suite_observed_relations(&subject_with(&one_note)).unwrap();
        let adjudicated = adjudicate_prog_suite_development(&observed).unwrap();
        assert!(adjudicated.stages.iter().all(|stage| {
            stage.disposition
                == ProgSuiteDevelopmentDispositionV1::IndeterminateInsufficientMaterial
        }));
    }

    #[test]
    fn serialized_disposition_cannot_be_hand_edited() {
        let observed = measure_prog_suite_observed_relations(&subject_with(&motif())).unwrap();
        let mut adjudicated = adjudicate_prog_suite_development(&observed).unwrap();
        adjudicated.stages[0].disposition =
            ProgSuiteDevelopmentDispositionV1::RejectedObservedUnderProfile;
        assert_eq!(
            adjudicated.validate(),
            Err(
                ProgSuiteDevelopmentAdjudicationErrorV1::CanonicalAdjudicationMismatch
            )
        );
    }

    #[test]
    fn profile_identity_and_nonclaims_are_exact_authority_boundaries() {
        let observed = measure_prog_suite_observed_relations(&subject_with(&motif())).unwrap();
        let adjudicated = adjudicate_prog_suite_development(&observed).unwrap();
        assert_eq!(
            adjudicated.profile_id,
            PROG_SUITE_FIRST_STATEMENT_EVIDENCE_PROFILE
        );
        assert_eq!(adjudicated.nonclaims, required_nonclaims());
        assert!(adjudicated.nonclaims.contains(
            &ProgSuiteDevelopmentAdjudicationNonClaimV1::DoesNotResolveOrMutateWorkObligationState
        ));
    }
}
