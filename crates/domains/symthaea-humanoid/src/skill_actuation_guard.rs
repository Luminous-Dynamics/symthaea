// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Per-requirement actuation-capability composition for operational skills.
//!
//! Composite skills can depend on physically different chains at the same time.
//! `Carry`, for example, needs both a manipulation qualification and a
//! locomotion qualification. A degraded hand chain should not be collapsed into
//! an undifferentiated whole-robot scalar, nor should healthy feet conceal lost
//! manipulation authority.
//!
//! This module binds each compiled skill requirement to exactly one typed
//! actuation policy, exactly one current controllability assessment, and exactly
//! one current capability envelope. It returns cloned envelopes with only the
//! affected subject tightened. The continuous skill executive can then re-run
//! ordinary capability admission over those guarded envelopes.

use serde::{Deserialize, Serialize};

use crate::actuator_controllability::HumanoidActuationControllabilityAssessment;
use crate::capability_envelope::{
    HumanoidCapabilityDisposition, HumanoidCapabilityEnvelope, HumanoidCapabilityLimits,
};
use crate::qualification::HumanoidQualificationSubject;
use crate::skill_runtime::{
    HumanoidSkillContract, HumanoidSkillRequirementRole,
};
use crate::typed_actuation_capability::{
    TypedHumanoidActuationCapabilityDecision, TypedHumanoidActuationCapabilityPolicy,
    assess_typed_humanoid_actuation_capability,
};
use crate::capability_envelope::HumanoidNominalCapabilityProfile;
use crate::types::HumanoidTask;

/// Qualified actuation policy for one exact requirement subject/role.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidSkillActuationPolicyEntry {
    pub role: HumanoidSkillRequirementRole,
    pub subject: HumanoidQualificationSubject,
    pub profile: HumanoidNominalCapabilityProfile,
    pub policy: TypedHumanoidActuationCapabilityPolicy,
}

impl HumanoidSkillActuationPolicyEntry {
    pub fn validate(&self) -> bool {
        self.subject.validate()
            && self.profile.validate_for(&self.subject)
            && self.policy.validate_for(&self.subject, &self.profile)
    }
}

/// Current actuation evidence for one exact qualification subject.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidSkillActuationEvidenceEntry {
    pub subject_fingerprint: u64,
    pub assessment: HumanoidActuationControllabilityAssessment,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidSkillActuationGuardFailure {
    InvalidPolicyEntry,
    PolicySubjectMismatch,
    PolicyRoleMismatch,
    PolicyTaskMismatch,
    MissingPolicy,
    DuplicatePolicy,
    MissingAssessment,
    DuplicateAssessment,
    MissingCapabilityEnvelope,
    DuplicateCapabilityEnvelope,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidSkillActuationRequirementReport {
    pub role: HumanoidSkillRequirementRole,
    pub task: HumanoidTask,
    pub subject_fingerprint: u64,
    pub failure: Option<HumanoidSkillActuationGuardFailure>,
    pub actuation_decision: Option<TypedHumanoidActuationCapabilityDecision>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidSkillActuationGuardReport {
    /// True only when every requirement was structurally bound to exactly one
    /// policy, assessment, and capability envelope. Individual actuation
    /// decisions can still intentionally Hold their subject envelope.
    pub composition_complete: bool,
    pub requirements: Vec<HumanoidSkillActuationRequirementReport>,
    /// Subject envelopes after task-local actuation restrictions were applied.
    pub guarded_envelopes: Vec<HumanoidCapabilityEnvelope>,
}

/// Apply current per-joint/contact actuation evidence independently to every
/// requirement of a compiled skill contract.
///
/// Missing/duplicate policy/evidence is never ignored. If a matching capability
/// envelope exists for a structurally failed requirement it is revoked to
/// `ProtectiveOnly`; if the envelope itself is missing/duplicated, the returned
/// report remains incomplete and downstream skill admission will also fail.
pub fn guard_humanoid_skill_actuation(
    contract: &HumanoidSkillContract,
    policies: &[HumanoidSkillActuationPolicyEntry],
    evidence: &[HumanoidSkillActuationEvidenceEntry],
    envelopes: &[HumanoidCapabilityEnvelope],
) -> HumanoidSkillActuationGuardReport {
    let mut guarded_envelopes = envelopes.to_vec();
    let mut reports = Vec::with_capacity(contract.requirements.len());

    for requirement in &contract.requirements {
        let subject_fingerprint = requirement.request.subject_fingerprint;
        let matching_policies = policies
            .iter()
            .filter(|entry| entry.subject.fingerprint() == subject_fingerprint)
            .collect::<Vec<_>>();
        let matching_evidence = evidence
            .iter()
            .filter(|entry| entry.subject_fingerprint == subject_fingerprint)
            .collect::<Vec<_>>();
        let envelope_indices = guarded_envelopes
            .iter()
            .enumerate()
            .filter_map(|(index, envelope)| {
                (envelope.subject_fingerprint == subject_fingerprint).then_some(index)
            })
            .collect::<Vec<_>>();

        let structural_failure = match matching_policies.as_slice() {
            [] => Some(HumanoidSkillActuationGuardFailure::MissingPolicy),
            [entry] if !entry.validate() => {
                Some(HumanoidSkillActuationGuardFailure::InvalidPolicyEntry)
            }
            [entry] if entry.subject.fingerprint() != subject_fingerprint => {
                Some(HumanoidSkillActuationGuardFailure::PolicySubjectMismatch)
            }
            [entry] if entry.role != requirement.role => {
                Some(HumanoidSkillActuationGuardFailure::PolicyRoleMismatch)
            }
            [entry] if entry.subject.task != requirement.task => {
                Some(HumanoidSkillActuationGuardFailure::PolicyTaskMismatch)
            }
            [_] => match matching_evidence.as_slice() {
                [] => Some(HumanoidSkillActuationGuardFailure::MissingAssessment),
                [_] => match envelope_indices.as_slice() {
                    [] => Some(HumanoidSkillActuationGuardFailure::MissingCapabilityEnvelope),
                    [_] => None,
                    _ => Some(HumanoidSkillActuationGuardFailure::DuplicateCapabilityEnvelope),
                },
                _ => Some(HumanoidSkillActuationGuardFailure::DuplicateAssessment),
            },
            _ => Some(HumanoidSkillActuationGuardFailure::DuplicatePolicy),
        };

        if let Some(failure) = structural_failure {
            if envelope_indices.len() == 1 {
                revoke_subject_envelope(&mut guarded_envelopes[envelope_indices[0]]);
            }
            reports.push(HumanoidSkillActuationRequirementReport {
                role: requirement.role,
                task: requirement.task,
                subject_fingerprint,
                failure: Some(failure),
                actuation_decision: None,
            });
            continue;
        }

        let policy = matching_policies[0];
        let assessment = &matching_evidence[0].assessment;
        let decision = assess_typed_humanoid_actuation_capability(
            &policy.subject,
            &policy.profile,
            &policy.policy,
            assessment,
        );
        decision.apply_to_envelope(&mut guarded_envelopes[envelope_indices[0]]);
        reports.push(HumanoidSkillActuationRequirementReport {
            role: requirement.role,
            task: requirement.task,
            subject_fingerprint,
            failure: None,
            actuation_decision: Some(decision),
        });
    }

    HumanoidSkillActuationGuardReport {
        composition_complete: reports.iter().all(|report| report.failure.is_none()),
        requirements: reports,
        guarded_envelopes,
    }
}

fn revoke_subject_envelope(envelope: &mut HumanoidCapabilityEnvelope) {
    envelope.goal_execution_allowed = false;
    envelope.human_contact_allowed = false;
    envelope.goal_authority_scale = 0.0;
    envelope.limits = HumanoidCapabilityLimits::zero();
    envelope.disposition = HumanoidCapabilityDisposition::ProtectiveOnly;
    envelope.restrictions_applied = envelope.restrictions_applied.saturating_add(1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actuator_controllability::{
        ContactWrenchAxis, ContactWrenchAxisMargin, ContactWrenchMarginAssessment,
    };
    use crate::capability_envelope::{
        HumanInteractionEvidence, HumanoidCapabilityRestriction,
        derive_humanoid_capability_envelope,
    };
    use crate::contact_site::HumanoidContactSite;
    use crate::execution::HumanoidAuthorityEnvelope;
    use crate::full_dynamics::DynamicsComponentSource;
    use crate::morphology::HumanoidMorphology;
    use crate::skill_runtime::{
        HumanoidLocomotionMode, HumanoidSkillIntent, HumanoidSkillQualificationSet,
        compile_humanoid_skill_contract,
    };
    use crate::typed_actuation_capability::{
        HUMANOID_TYPED_ACTUATION_POLICY_SCHEMA_VERSION,
        TypedHumanoidContactActuationRequirement,
    };
    use crate::types::ActuationMode;

    fn subject(task: HumanoidTask) -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            task,
            ActuationMode::NormalizedTorque,
            "skill-actuation-test-v1",
        )
    }

    fn profile(task: HumanoidTask) -> HumanoidNominalCapabilityProfile {
        HumanoidNominalCapabilityProfile::new(
            &subject(task), 2.0, 0.4, 1.0, 1.0, 0.2, 12.0, 80.0, 20.0, 0.8, true,
        )
    }

    fn envelope(task: HumanoidTask) -> HumanoidCapabilityEnvelope {
        derive_humanoid_capability_envelope(
            &subject(task),
            &profile(task),
            HumanoidAuthorityEnvelope::fully_admitted(),
            HumanInteractionEvidence::no_human_present(),
        )
    }

    fn policy_entry(
        task: HumanoidTask,
        role: HumanoidSkillRequirementRole,
        site: HumanoidContactSite,
    ) -> HumanoidSkillActuationPolicyEntry {
        let subject = subject(task);
        let profile = profile(task);
        let policy = TypedHumanoidActuationCapabilityPolicy {
            schema_version: HUMANOID_TYPED_ACTUATION_POLICY_SCHEMA_VERSION,
            subject_fingerprint: subject.fingerprint(),
            authority_profile_id: "joint-authority-v1".into(),
            calibration_fingerprint: 42,
            dynamics_model_id: "dynamics-v1".into(),
            requirements: vec![TypedHumanoidContactActuationRequirement {
                site,
                axis: ContactWrenchAxis::ForceZ,
                nominal_min_retained_fraction: 0.8,
                degraded_min_retained_fraction: 0.5,
                nominal_min_retained_wrench: 80.0,
                degraded_min_retained_wrench: 50.0,
            }],
            degraded_restriction: HumanoidCapabilityRestriction {
                max_horizontal_speed_mps: 0.5,
                max_turn_rate_rad_s: 0.3,
                max_end_effector_speed_mps: 0.3,
                max_payload_kg: 5.0,
                max_object_contact_force_n: 40.0,
                max_human_contact_force_n: 10.0,
            },
        };
        HumanoidSkillActuationPolicyEntry {
            role,
            subject,
            profile,
            policy,
        }
    }

    fn evidence_entry(
        task: HumanoidTask,
        site: HumanoidContactSite,
        fraction: f64,
        retained_wrench: f64,
    ) -> HumanoidSkillActuationEvidenceEntry {
        HumanoidSkillActuationEvidenceEntry {
            subject_fingerprint: subject(task).fingerprint(),
            assessment: HumanoidActuationControllabilityAssessment {
                morphology: HumanoidMorphology::Dexterous53,
                authority_sequence: 1,
                authority_age_s: 0.01,
                authority_profile_id: "joint-authority-v1".into(),
                calibration_fingerprint: 42,
                dynamics_model_id: "dynamics-v1".into(),
                sites: vec![ContactWrenchMarginAssessment {
                    site_id: site.canonical_id().into(),
                    contact_confidence: 1.0,
                    jacobian_source: DynamicsComponentSource::SimulatorSolver,
                    actuator_limit_source: DynamicsComponentSource::SystemIdentification,
                    axes: vec![ContactWrenchAxisMargin {
                        axis: ContactWrenchAxis::ForceZ,
                        actuated_support_present: true,
                        nominal_limit: Some(100.0),
                        retained_limit: Some(retained_wrench),
                        retained_fraction: fraction,
                        limiting_joint: Some(0),
                    }],
                    minimum_retained_fraction: fraction,
                }],
            },
        }
    }

    fn carry_contract() -> HumanoidSkillContract {
        let qualifications = HumanoidSkillQualificationSet::new(vec![
            subject(HumanoidTask::Walk),
            subject(HumanoidTask::Grasp),
        ]);
        compile_humanoid_skill_contract(
            HumanoidSkillIntent::Carry {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.4,
                turn_rate_rad_s: 0.1,
                manipulation_speed_mps: 0.1,
                retention_force_n: 10.0,
                resulting_total_payload_kg: 3.0,
            },
            &qualifications,
        )
        .unwrap()
    }

    fn policies() -> Vec<HumanoidSkillActuationPolicyEntry> {
        vec![
            policy_entry(
                HumanoidTask::Grasp,
                HumanoidSkillRequirementRole::Manipulation,
                HumanoidContactSite::RightHand,
            ),
            policy_entry(
                HumanoidTask::Walk,
                HumanoidSkillRequirementRole::Locomotion,
                HumanoidContactSite::RightFoot,
            ),
        ]
    }

    fn nominal_evidence() -> Vec<HumanoidSkillActuationEvidenceEntry> {
        vec![
            evidence_entry(HumanoidTask::Grasp, HumanoidContactSite::RightHand, 0.9, 90.0),
            evidence_entry(HumanoidTask::Walk, HumanoidContactSite::RightFoot, 0.9, 90.0),
        ]
    }

    #[test]
    fn composite_carry_guards_manipulation_and_locomotion_independently() {
        let report = guard_humanoid_skill_actuation(
            &carry_contract(),
            &policies(),
            &nominal_evidence(),
            &[envelope(HumanoidTask::Grasp), envelope(HumanoidTask::Walk)],
        );
        assert!(report.composition_complete);
        assert!(report.guarded_envelopes.iter().all(|e| e.goal_execution_allowed));
    }

    #[test]
    fn lost_hand_authority_revokes_only_manipulation_subject() {
        let evidence = vec![
            evidence_entry(HumanoidTask::Grasp, HumanoidContactSite::RightHand, 0.3, 30.0),
            evidence_entry(HumanoidTask::Walk, HumanoidContactSite::RightFoot, 0.9, 90.0),
        ];
        let report = guard_humanoid_skill_actuation(
            &carry_contract(),
            &policies(),
            &evidence,
            &[envelope(HumanoidTask::Grasp), envelope(HumanoidTask::Walk)],
        );
        let grasp = report
            .guarded_envelopes
            .iter()
            .find(|e| e.subject_fingerprint == subject(HumanoidTask::Grasp).fingerprint())
            .unwrap();
        let walk = report
            .guarded_envelopes
            .iter()
            .find(|e| e.subject_fingerprint == subject(HumanoidTask::Walk).fingerprint())
            .unwrap();
        assert!(!grasp.goal_execution_allowed);
        assert!(walk.goal_execution_allowed);
    }

    #[test]
    fn lost_foot_authority_revokes_only_locomotion_subject() {
        let evidence = vec![
            evidence_entry(HumanoidTask::Grasp, HumanoidContactSite::RightHand, 0.9, 90.0),
            evidence_entry(HumanoidTask::Walk, HumanoidContactSite::RightFoot, 0.3, 30.0),
        ];
        let report = guard_humanoid_skill_actuation(
            &carry_contract(),
            &policies(),
            &evidence,
            &[envelope(HumanoidTask::Grasp), envelope(HumanoidTask::Walk)],
        );
        let grasp = report
            .guarded_envelopes
            .iter()
            .find(|e| e.subject_fingerprint == subject(HumanoidTask::Grasp).fingerprint())
            .unwrap();
        let walk = report
            .guarded_envelopes
            .iter()
            .find(|e| e.subject_fingerprint == subject(HumanoidTask::Walk).fingerprint())
            .unwrap();
        assert!(grasp.goal_execution_allowed);
        assert!(!walk.goal_execution_allowed);
    }

    #[test]
    fn missing_manipulation_policy_is_explicit_and_revokes_its_envelope() {
        let policies = policies()
            .into_iter()
            .filter(|entry| entry.subject.task != HumanoidTask::Grasp)
            .collect::<Vec<_>>();
        let report = guard_humanoid_skill_actuation(
            &carry_contract(),
            &policies,
            &nominal_evidence(),
            &[envelope(HumanoidTask::Grasp), envelope(HumanoidTask::Walk)],
        );
        assert!(!report.composition_complete);
        assert!(report.requirements.iter().any(|requirement| {
            requirement.task == HumanoidTask::Grasp
                && requirement.failure == Some(HumanoidSkillActuationGuardFailure::MissingPolicy)
        }));
        let grasp = report
            .guarded_envelopes
            .iter()
            .find(|e| e.subject_fingerprint == subject(HumanoidTask::Grasp).fingerprint())
            .unwrap();
        assert!(!grasp.goal_execution_allowed);
    }

    #[test]
    fn duplicate_runtime_assessment_fails_closed() {
        let mut evidence = nominal_evidence();
        evidence.push(evidence_entry(
            HumanoidTask::Walk,
            HumanoidContactSite::RightFoot,
            0.9,
            90.0,
        ));
        let report = guard_humanoid_skill_actuation(
            &carry_contract(),
            &policies(),
            &evidence,
            &[envelope(HumanoidTask::Grasp), envelope(HumanoidTask::Walk)],
        );
        assert!(!report.composition_complete);
        assert!(report.requirements.iter().any(|requirement| {
            requirement.task == HumanoidTask::Walk
                && requirement.failure
                    == Some(HumanoidSkillActuationGuardFailure::DuplicateAssessment)
        }));
    }
}
