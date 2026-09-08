// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Composite-aware semantic skill contracts for humanoid execution.
//!
//! The DMC `HumanoidTask` enum is a benchmark/control substrate, not an ontology
//! for every operational behavior. This module keeps those layers separate:
//! operational skills compile into one or more subject-bound
//! `HumanoidCapabilityRequest`s, and every requirement must be admitted before
//! any motor synthesis begins.
//!
//! A composite skill such as `Carry` therefore requires both a manipulation
//! qualification (`Grasp`) and a locomotion qualification (`Walk` or `Run`) on
//! the same morphology/backend/actuation semantics. It does not create a fake
//! `HumanoidTask::Carry`.

use serde::{Deserialize, Serialize};

use crate::capability_envelope::HumanoidCapabilityEnvelope;
use crate::capability_request::{
    HumanoidCapabilityAdmission, HumanoidCapabilityRequest, admit_humanoid_capability_request,
};
use crate::qualification::HumanoidQualificationSubject;
use crate::types::{ActuationMode, HumanoidTask};
use crate::morphology::HumanoidMorphology;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidLocomotionMode {
    Walk,
    Run,
}

impl HumanoidLocomotionMode {
    pub const fn task(self) -> HumanoidTask {
        match self {
            Self::Walk => HumanoidTask::Walk,
            Self::Run => HumanoidTask::Run,
        }
    }
}

/// Operational skill intent. Numeric demands are requests, not automatically
/// clamped targets; capability admission may reject them and require replanning.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HumanoidSkillIntent {
    Stand,
    Locomote {
        mode: HumanoidLocomotionMode,
        horizontal_speed_mps: f64,
        turn_rate_rad_s: f64,
    },
    Reach {
        end_effector_speed_mps: f64,
    },
    Grasp {
        end_effector_speed_mps: f64,
        object_contact_force_n: f64,
        resulting_total_payload_kg: f64,
    },
    /// Composite manipulation + locomotion skill.
    Carry {
        mode: HumanoidLocomotionMode,
        horizontal_speed_mps: f64,
        turn_rate_rad_s: f64,
        manipulation_speed_mps: f64,
        retention_force_n: f64,
        resulting_total_payload_kg: f64,
    },
    /// Intentional physical interaction with a person. Current human-contact
    /// consent remains an independent runtime capability gate.
    AssistHuman {
        end_effector_speed_mps: f64,
        human_contact_force_n: f64,
    },
}

impl HumanoidSkillIntent {
    pub fn validate(self) -> bool {
        match self {
            Self::Stand => true,
            Self::Locomote {
                horizontal_speed_mps,
                turn_rate_rad_s,
                ..
            } => non_negative_finite(&[horizontal_speed_mps, turn_rate_rad_s]),
            Self::Reach {
                end_effector_speed_mps,
            } => non_negative_finite(&[end_effector_speed_mps]),
            Self::Grasp {
                end_effector_speed_mps,
                object_contact_force_n,
                resulting_total_payload_kg,
            } => non_negative_finite(&[
                end_effector_speed_mps,
                object_contact_force_n,
                resulting_total_payload_kg,
            ]),
            Self::Carry {
                horizontal_speed_mps,
                turn_rate_rad_s,
                manipulation_speed_mps,
                retention_force_n,
                resulting_total_payload_kg,
                ..
            } => {
                non_negative_finite(&[
                    horizontal_speed_mps,
                    turn_rate_rad_s,
                    manipulation_speed_mps,
                    retention_force_n,
                    resulting_total_payload_kg,
                ]) && resulting_total_payload_kg > 0.0
            }
            Self::AssistHuman {
                end_effector_speed_mps,
                human_contact_force_n,
            } => non_negative_finite(&[end_effector_speed_mps, human_contact_force_n]),
        }
    }
}

fn non_negative_finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite() && *value >= 0.0)
}

/// A runtime should hold one coherent qualification set for one physical body
/// and backend. Multiple tasks are allowed; mixed morphologies/backends or
/// duplicate task subjects are rejected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidSkillQualificationSet {
    subjects: Vec<HumanoidQualificationSubject>,
}

impl HumanoidSkillQualificationSet {
    pub fn new(subjects: Vec<HumanoidQualificationSubject>) -> Self {
        Self { subjects }
    }

    pub fn subjects(&self) -> &[HumanoidQualificationSubject] {
        &self.subjects
    }

    pub fn validate(&self) -> Result<(), HumanoidSkillCompileError> {
        let Some(first) = self.subjects.first() else {
            return Err(HumanoidSkillCompileError::EmptyQualificationSet);
        };
        if !first.validate() {
            return Err(HumanoidSkillCompileError::InvalidQualificationSubject);
        }
        for (index, subject) in self.subjects.iter().enumerate() {
            if !subject.validate() {
                return Err(HumanoidSkillCompileError::InvalidQualificationSubject);
            }
            if subject.morphology != first.morphology {
                return Err(HumanoidSkillCompileError::MixedMorphology);
            }
            if subject.actuation_mode != first.actuation_mode {
                return Err(HumanoidSkillCompileError::MixedActuationMode);
            }
            if subject.backend_profile_id != first.backend_profile_id {
                return Err(HumanoidSkillCompileError::MixedBackendProfile);
            }
            if self.subjects[..index]
                .iter()
                .any(|previous| previous.task == subject.task)
            {
                return Err(HumanoidSkillCompileError::DuplicateTask(subject.task));
            }
        }
        Ok(())
    }

    pub fn morphology(&self) -> Option<HumanoidMorphology> {
        self.subjects.first().map(|subject| subject.morphology)
    }

    pub fn actuation_mode(&self) -> Option<ActuationMode> {
        self.subjects.first().map(|subject| subject.actuation_mode)
    }

    pub fn backend_profile_id(&self) -> Option<&str> {
        self.subjects
            .first()
            .map(|subject| subject.backend_profile_id.as_str())
    }

    pub fn subject_for(
        &self,
        task: HumanoidTask,
    ) -> Result<&HumanoidQualificationSubject, HumanoidSkillCompileError> {
        self.validate()?;
        self.subjects
            .iter()
            .find(|subject| subject.task == task)
            .ok_or(HumanoidSkillCompileError::MissingTask(task))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidSkillRequirementRole {
    Posture,
    Locomotion,
    Manipulation,
    HumanInteraction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidSkillPrecondition {
    GoalExecutionAuthority,
    QualifiedSubject(HumanoidTask),
    LoadRetentionEvidence,
    HumanProximityEvidence,
    HumanContactConsent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidSkillInvariant {
    CapabilityEnvelopeRespected,
    QualificationSubjectStable,
    ProtectiveBehaviorMayPreemptGoal,
    LoadRetentionMaintained,
    HumanContactConsentMaintained,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidSkillRecoveryPolicy {
    Replan,
    SecureLoadThenReplan,
    WithdrawThenReplan,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidSkillRequirement {
    pub role: HumanoidSkillRequirementRole,
    pub task: HumanoidTask,
    pub request: HumanoidCapabilityRequest,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidSkillContract {
    pub intent: HumanoidSkillIntent,
    pub morphology: HumanoidMorphology,
    pub actuation_mode: ActuationMode,
    pub backend_profile_id: String,
    pub requirements: Vec<HumanoidSkillRequirement>,
    pub preconditions: Vec<HumanoidSkillPrecondition>,
    pub invariants: Vec<HumanoidSkillInvariant>,
    pub recovery: HumanoidSkillRecoveryPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidSkillCompileError {
    InvalidIntent,
    EmptyQualificationSet,
    InvalidQualificationSubject,
    MixedMorphology,
    MixedActuationMode,
    MixedBackendProfile,
    DuplicateTask(HumanoidTask),
    MissingTask(HumanoidTask),
}

/// Compile an operational skill into one or more independently subject-bound
/// capability requirements. This does not admit or execute the skill.
pub fn compile_humanoid_skill_contract(
    intent: HumanoidSkillIntent,
    qualifications: &HumanoidSkillQualificationSet,
) -> Result<HumanoidSkillContract, HumanoidSkillCompileError> {
    if !intent.validate() {
        return Err(HumanoidSkillCompileError::InvalidIntent);
    }
    qualifications.validate()?;
    let morphology = qualifications
        .morphology()
        .ok_or(HumanoidSkillCompileError::EmptyQualificationSet)?;
    let actuation_mode = qualifications
        .actuation_mode()
        .ok_or(HumanoidSkillCompileError::EmptyQualificationSet)?;
    let backend_profile_id = qualifications
        .backend_profile_id()
        .ok_or(HumanoidSkillCompileError::EmptyQualificationSet)?
        .to_string();

    let mut requirements = Vec::new();
    let mut preconditions = vec![HumanoidSkillPrecondition::GoalExecutionAuthority];
    let mut invariants = vec![
        HumanoidSkillInvariant::CapabilityEnvelopeRespected,
        HumanoidSkillInvariant::QualificationSubjectStable,
        HumanoidSkillInvariant::ProtectiveBehaviorMayPreemptGoal,
    ];
    let recovery;

    match intent {
        HumanoidSkillIntent::Stand => {
            let subject = qualifications.subject_for(HumanoidTask::Stand)?;
            push_requirement(
                &mut requirements,
                &mut preconditions,
                HumanoidSkillRequirementRole::Posture,
                subject,
                HumanoidCapabilityRequest::stationary(subject.fingerprint()),
            );
            recovery = HumanoidSkillRecoveryPolicy::Replan;
        }
        HumanoidSkillIntent::Locomote {
            mode,
            horizontal_speed_mps,
            turn_rate_rad_s,
        } => {
            let task = mode.task();
            let subject = qualifications.subject_for(task)?;
            let mut request = HumanoidCapabilityRequest::stationary(subject.fingerprint());
            request.horizontal_speed_mps = horizontal_speed_mps;
            request.turn_rate_rad_s = turn_rate_rad_s;
            push_requirement(
                &mut requirements,
                &mut preconditions,
                HumanoidSkillRequirementRole::Locomotion,
                subject,
                request,
            );
            recovery = HumanoidSkillRecoveryPolicy::Replan;
        }
        HumanoidSkillIntent::Reach {
            end_effector_speed_mps,
        } => {
            let subject = qualifications.subject_for(HumanoidTask::Reach)?;
            let mut request = HumanoidCapabilityRequest::stationary(subject.fingerprint());
            request.end_effector_speed_mps = end_effector_speed_mps;
            push_requirement(
                &mut requirements,
                &mut preconditions,
                HumanoidSkillRequirementRole::Manipulation,
                subject,
                request,
            );
            recovery = HumanoidSkillRecoveryPolicy::Replan;
        }
        HumanoidSkillIntent::Grasp {
            end_effector_speed_mps,
            object_contact_force_n,
            resulting_total_payload_kg,
        } => {
            let subject = qualifications.subject_for(HumanoidTask::Grasp)?;
            let mut request = HumanoidCapabilityRequest::stationary(subject.fingerprint());
            request.end_effector_speed_mps = end_effector_speed_mps;
            request.object_contact_force_n = object_contact_force_n;
            request.resulting_total_payload_kg = resulting_total_payload_kg;
            push_requirement(
                &mut requirements,
                &mut preconditions,
                HumanoidSkillRequirementRole::Manipulation,
                subject,
                request,
            );
            if resulting_total_payload_kg > 0.0 {
                preconditions.push(HumanoidSkillPrecondition::LoadRetentionEvidence);
                invariants.push(HumanoidSkillInvariant::LoadRetentionMaintained);
            }
            recovery = HumanoidSkillRecoveryPolicy::SecureLoadThenReplan;
        }
        HumanoidSkillIntent::Carry {
            mode,
            horizontal_speed_mps,
            turn_rate_rad_s,
            manipulation_speed_mps,
            retention_force_n,
            resulting_total_payload_kg,
        } => {
            // Manipulation qualification remains active while locomotion occurs;
            // these are concurrent skill requirements, not sequential permission.
            let grasp = qualifications.subject_for(HumanoidTask::Grasp)?;
            let locomotion = qualifications.subject_for(mode.task())?;

            let mut grasp_request = HumanoidCapabilityRequest::stationary(grasp.fingerprint());
            grasp_request.end_effector_speed_mps = manipulation_speed_mps;
            grasp_request.object_contact_force_n = retention_force_n;
            grasp_request.resulting_total_payload_kg = resulting_total_payload_kg;
            push_requirement(
                &mut requirements,
                &mut preconditions,
                HumanoidSkillRequirementRole::Manipulation,
                grasp,
                grasp_request,
            );

            let mut locomotion_request =
                HumanoidCapabilityRequest::stationary(locomotion.fingerprint());
            locomotion_request.horizontal_speed_mps = horizontal_speed_mps;
            locomotion_request.turn_rate_rad_s = turn_rate_rad_s;
            locomotion_request.resulting_total_payload_kg = resulting_total_payload_kg;
            push_requirement(
                &mut requirements,
                &mut preconditions,
                HumanoidSkillRequirementRole::Locomotion,
                locomotion,
                locomotion_request,
            );

            preconditions.push(HumanoidSkillPrecondition::LoadRetentionEvidence);
            invariants.push(HumanoidSkillInvariant::LoadRetentionMaintained);
            recovery = HumanoidSkillRecoveryPolicy::SecureLoadThenReplan;
        }
        HumanoidSkillIntent::AssistHuman {
            end_effector_speed_mps,
            human_contact_force_n,
        } => {
            let subject = qualifications.subject_for(HumanoidTask::Reach)?;
            let mut request = HumanoidCapabilityRequest::stationary(subject.fingerprint());
            request.end_effector_speed_mps = end_effector_speed_mps;
            request.intentional_human_contact = true;
            request.human_contact_force_n = human_contact_force_n;
            push_requirement(
                &mut requirements,
                &mut preconditions,
                HumanoidSkillRequirementRole::HumanInteraction,
                subject,
                request,
            );
            preconditions.push(HumanoidSkillPrecondition::HumanProximityEvidence);
            preconditions.push(HumanoidSkillPrecondition::HumanContactConsent);
            invariants.push(HumanoidSkillInvariant::HumanContactConsentMaintained);
            recovery = HumanoidSkillRecoveryPolicy::WithdrawThenReplan;
        }
    }

    Ok(HumanoidSkillContract {
        intent,
        morphology,
        actuation_mode,
        backend_profile_id,
        requirements,
        preconditions,
        invariants,
        recovery,
    })
}

fn push_requirement(
    requirements: &mut Vec<HumanoidSkillRequirement>,
    preconditions: &mut Vec<HumanoidSkillPrecondition>,
    role: HumanoidSkillRequirementRole,
    subject: &HumanoidQualificationSubject,
    request: HumanoidCapabilityRequest,
) {
    requirements.push(HumanoidSkillRequirement {
        role,
        task: subject.task,
        request,
    });
    preconditions.push(HumanoidSkillPrecondition::QualifiedSubject(subject.task));
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidSkillRequirementFailure {
    MissingCapabilityEnvelope,
    DuplicateCapabilityEnvelope,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidSkillRequirementAdmission {
    pub role: HumanoidSkillRequirementRole,
    pub task: HumanoidTask,
    pub subject_fingerprint: u64,
    pub capability_admission: Option<HumanoidCapabilityAdmission>,
    pub failure: Option<HumanoidSkillRequirementFailure>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidSkillAdmission {
    pub admitted: bool,
    pub contract: HumanoidSkillContract,
    pub requirements: Vec<HumanoidSkillRequirementAdmission>,
}

/// Admit every requirement of a compiled skill against current envelopes.
/// Composite skills are admitted only when every required subject is available
/// exactly once and every capability request is independently admitted.
pub fn admit_humanoid_skill_contract(
    contract: HumanoidSkillContract,
    envelopes: &[HumanoidCapabilityEnvelope],
) -> HumanoidSkillAdmission {
    let requirements = contract
        .requirements
        .iter()
        .map(|requirement| {
            let matches = envelopes
                .iter()
                .filter(|envelope| {
                    envelope.subject_fingerprint == requirement.request.subject_fingerprint
                })
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [] => HumanoidSkillRequirementAdmission {
                    role: requirement.role,
                    task: requirement.task,
                    subject_fingerprint: requirement.request.subject_fingerprint,
                    capability_admission: None,
                    failure: Some(HumanoidSkillRequirementFailure::MissingCapabilityEnvelope),
                },
                [envelope] => HumanoidSkillRequirementAdmission {
                    role: requirement.role,
                    task: requirement.task,
                    subject_fingerprint: requirement.request.subject_fingerprint,
                    capability_admission: Some(admit_humanoid_capability_request(
                        envelope,
                        requirement.request,
                    )),
                    failure: None,
                },
                _ => HumanoidSkillRequirementAdmission {
                    role: requirement.role,
                    task: requirement.task,
                    subject_fingerprint: requirement.request.subject_fingerprint,
                    capability_admission: None,
                    failure: Some(HumanoidSkillRequirementFailure::DuplicateCapabilityEnvelope),
                },
            }
        })
        .collect::<Vec<_>>();

    let admitted = requirements.iter().all(|requirement| {
        requirement.failure.is_none()
            && requirement
                .capability_admission
                .as_ref()
                .is_some_and(|admission| admission.admitted)
    });

    HumanoidSkillAdmission {
        admitted,
        contract,
        requirements,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability_envelope::{
        HumanInteractionEvidence, HumanoidNominalCapabilityProfile,
        derive_humanoid_capability_envelope,
    };
    use crate::execution::HumanoidAuthorityEnvelope;

    fn subject(task: HumanoidTask) -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            task,
            ActuationMode::NormalizedTorque,
            "skill-test-backend-v1",
        )
    }

    fn qualifications() -> HumanoidSkillQualificationSet {
        HumanoidSkillQualificationSet::new(vec![
            subject(HumanoidTask::Stand),
            subject(HumanoidTask::Walk),
            subject(HumanoidTask::Run),
            subject(HumanoidTask::Reach),
            subject(HumanoidTask::Grasp),
        ])
    }

    fn envelope_for(
        task: HumanoidTask,
        human: HumanInteractionEvidence,
    ) -> HumanoidCapabilityEnvelope {
        let subject = subject(task);
        // Test-only profile values, not production safety limits.
        let profile = HumanoidNominalCapabilityProfile::new(
            &subject, 2.0, 0.4, 1.0, 1.0, 0.2, 12.0, 80.0, 20.0, 0.8, true,
        );
        derive_humanoid_capability_envelope(
            &subject,
            &profile,
            HumanoidAuthorityEnvelope::fully_admitted(),
            human,
        )
    }

    fn all_envelopes() -> Vec<HumanoidCapabilityEnvelope> {
        [
            HumanoidTask::Stand,
            HumanoidTask::Walk,
            HumanoidTask::Run,
            HumanoidTask::Reach,
            HumanoidTask::Grasp,
        ]
        .into_iter()
        .map(|task| envelope_for(task, HumanInteractionEvidence::no_human_present()))
        .collect()
    }

    #[test]
    fn carry_compiles_to_manipulation_and_locomotion_requirements() {
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Carry {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.5,
                turn_rate_rad_s: 0.2,
                manipulation_speed_mps: 0.1,
                retention_force_n: 20.0,
                resulting_total_payload_kg: 4.0,
            },
            &qualifications(),
        )
        .unwrap();
        assert_eq!(contract.requirements.len(), 2);
        assert!(contract.requirements.iter().any(|requirement| {
            requirement.role == HumanoidSkillRequirementRole::Manipulation
                && requirement.task == HumanoidTask::Grasp
        }));
        assert!(contract.requirements.iter().any(|requirement| {
            requirement.role == HumanoidSkillRequirementRole::Locomotion
                && requirement.task == HumanoidTask::Walk
        }));
    }

    #[test]
    fn carry_is_admitted_only_when_both_subject_envelopes_admit() {
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Carry {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.5,
                turn_rate_rad_s: 0.2,
                manipulation_speed_mps: 0.1,
                retention_force_n: 20.0,
                resulting_total_payload_kg: 4.0,
            },
            &qualifications(),
        )
        .unwrap();
        let admission = admit_humanoid_skill_contract(contract, &all_envelopes());
        assert!(admission.admitted, "{:?}", admission.requirements);
    }

    #[test]
    fn carry_fails_when_manipulation_envelope_is_missing() {
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Carry {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.5,
                turn_rate_rad_s: 0.2,
                manipulation_speed_mps: 0.1,
                retention_force_n: 20.0,
                resulting_total_payload_kg: 4.0,
            },
            &qualifications(),
        )
        .unwrap();
        let envelopes = all_envelopes()
            .into_iter()
            .filter(|envelope| envelope.subject_fingerprint != subject(HumanoidTask::Grasp).fingerprint())
            .collect::<Vec<_>>();
        let admission = admit_humanoid_skill_contract(contract, &envelopes);
        assert!(!admission.admitted);
        assert!(admission.requirements.iter().any(|requirement| {
            requirement.task == HumanoidTask::Grasp
                && requirement.failure
                    == Some(HumanoidSkillRequirementFailure::MissingCapabilityEnvelope)
        }));
    }

    #[test]
    fn carry_fails_when_locomotion_request_exceeds_current_envelope() {
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Carry {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 1.5,
                turn_rate_rad_s: 0.2,
                manipulation_speed_mps: 0.1,
                retention_force_n: 20.0,
                resulting_total_payload_kg: 4.0,
            },
            &qualifications(),
        )
        .unwrap();
        let mut envelopes = all_envelopes();
        let walk_fingerprint = subject(HumanoidTask::Walk).fingerprint();
        let walk = envelopes
            .iter_mut()
            .find(|envelope| envelope.subject_fingerprint == walk_fingerprint)
            .unwrap();
        walk.limits.max_horizontal_speed_mps = 0.5;
        let admission = admit_humanoid_skill_contract(contract, &envelopes);
        assert!(!admission.admitted);
        assert!(admission.requirements.iter().any(|requirement| {
            requirement.task == HumanoidTask::Walk
                && requirement
                    .capability_admission
                    .as_ref()
                    .is_some_and(|capability| !capability.admitted)
        }));
    }

    #[test]
    fn assist_human_requires_live_contact_consent() {
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::AssistHuman {
                end_effector_speed_mps: 0.1,
                human_contact_force_n: 5.0,
            },
            &qualifications(),
        )
        .unwrap();
        let reach = envelope_for(
            HumanoidTask::Reach,
            HumanInteractionEvidence {
                human_present: true,
                proximity_valid: true,
                nearest_human_distance_m: 1.0,
                contact_consent: false,
            },
        );
        let admission = admit_humanoid_skill_contract(contract, &[reach]);
        assert!(!admission.admitted);
    }

    #[test]
    fn assist_human_can_be_admitted_when_contact_is_explicitly_consented() {
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::AssistHuman {
                end_effector_speed_mps: 0.1,
                human_contact_force_n: 5.0,
            },
            &qualifications(),
        )
        .unwrap();
        let reach = envelope_for(
            HumanoidTask::Reach,
            HumanInteractionEvidence {
                human_present: true,
                proximity_valid: true,
                nearest_human_distance_m: 1.0,
                contact_consent: true,
            },
        );
        assert!(admit_humanoid_skill_contract(contract, &[reach]).admitted);
    }

    #[test]
    fn mixed_backend_qualification_set_is_rejected() {
        let mut subjects = qualifications().subjects().to_vec();
        subjects.push(HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Walk,
            ActuationMode::NormalizedTorque,
            "other-backend",
        ));
        let set = HumanoidSkillQualificationSet::new(subjects);
        assert!(set.validate().is_err());
    }

    #[test]
    fn invalid_carry_with_zero_payload_is_rejected_before_compilation() {
        let result = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Carry {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.2,
                turn_rate_rad_s: 0.1,
                manipulation_speed_mps: 0.1,
                retention_force_n: 1.0,
                resulting_total_payload_kg: 0.0,
            },
            &qualifications(),
        );
        assert_eq!(result, Err(HumanoidSkillCompileError::InvalidIntent));
    }
}
