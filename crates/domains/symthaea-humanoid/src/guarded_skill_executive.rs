// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Preferred continuous skill executive with task-local actuation guarding.
//!
//! The lower `HumanoidSkillExecutive` continuously revalidates semantic
//! capability, authority, consent, and retention. This wrapper additionally
//! guarantees that current per-requirement actuation evidence is composed into
//! the capability envelopes before each Active admission/revalidation.
//!
//! It does not synthesize joint commands or add motor authority. The wrapper is
//! intended as the migration target for runtime callers; the lower executive is
//! retained temporarily as a compatibility primitive.

use serde::{Deserialize, Serialize};

use crate::capability_envelope::HumanoidCapabilityEnvelope;
use crate::skill_actuation_guard::{
    HumanoidSkillActuationEvidenceEntry, HumanoidSkillActuationGuardReport,
    HumanoidSkillActuationPolicyEntry, guard_humanoid_skill_actuation,
};
use crate::skill_executive::{
    HumanoidSkillExecutive, HumanoidSkillExecutiveReport, HumanoidSkillExecutiveState,
    HumanoidSkillRuntimeEvidence, HumanoidSkillStartRejection,
};
use crate::skill_runtime::HumanoidSkillContract;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidGuardedSkillExecutiveReport {
    pub actuation_guard: Option<HumanoidSkillActuationGuardReport>,
    pub executive: HumanoidSkillExecutiveReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidGuardedSkillStartRejection {
    pub actuation_guard: HumanoidSkillActuationGuardReport,
    /// Present when guard composition completed and the lower continuous
    /// executive rejected the skill for a semantic/runtime reason.
    pub executive: Option<HumanoidSkillStartRejection>,
}

/// Continuous skill executive whose public start/tick path always composes
/// current requirement-local actuation evidence first.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidGuardedSkillExecutive {
    inner: HumanoidSkillExecutive,
}

impl Default for HumanoidGuardedSkillExecutive {
    fn default() -> Self {
        Self::new()
    }
}

impl HumanoidGuardedSkillExecutive {
    pub const fn new() -> Self {
        Self {
            inner: HumanoidSkillExecutive::new(),
        }
    }

    pub const fn state(&self) -> HumanoidSkillExecutiveState {
        self.inner.state()
    }

    pub fn active_contract(&self) -> Option<&HumanoidSkillContract> {
        self.inner.active_contract()
    }

    /// Start a skill only after task-local actuation evidence has been composed
    /// into every required subject envelope.
    pub fn start(
        &mut self,
        contract: HumanoidSkillContract,
        envelopes: &[HumanoidCapabilityEnvelope],
        actuation_policies: &[HumanoidSkillActuationPolicyEntry],
        actuation_evidence: &[HumanoidSkillActuationEvidenceEntry],
        runtime_evidence: HumanoidSkillRuntimeEvidence,
    ) -> Result<HumanoidGuardedSkillExecutiveReport, HumanoidGuardedSkillStartRejection> {
        let guard = guard_humanoid_skill_actuation(
            &contract,
            actuation_policies,
            actuation_evidence,
            envelopes,
        );
        if !guard.composition_complete {
            return Err(HumanoidGuardedSkillStartRejection {
                actuation_guard: guard,
                executive: None,
            });
        }

        match self
            .inner
            .start(contract, &guard.guarded_envelopes, runtime_evidence)
        {
            Ok(executive) => Ok(HumanoidGuardedSkillExecutiveReport {
                actuation_guard: Some(guard),
                executive,
            }),
            Err(executive) => Err(HumanoidGuardedSkillStartRejection {
                actuation_guard: guard,
                executive: Some(executive),
            }),
        }
    }

    /// Revalidate the active contract after first recomputing current guarded
    /// envelopes. Structural actuation-composition failures are represented by
    /// fail-closed guarded envelopes where possible; the lower executive then
    /// transitions the goal into its declared recovery path.
    pub fn tick(
        &mut self,
        envelopes: &[HumanoidCapabilityEnvelope],
        actuation_policies: &[HumanoidSkillActuationPolicyEntry],
        actuation_evidence: &[HumanoidSkillActuationEvidenceEntry],
        runtime_evidence: HumanoidSkillRuntimeEvidence,
    ) -> HumanoidGuardedSkillExecutiveReport {
        if self.inner.state() != HumanoidSkillExecutiveState::Active {
            return HumanoidGuardedSkillExecutiveReport {
                actuation_guard: None,
                executive: self.inner.tick(envelopes, runtime_evidence),
            };
        }

        let contract = self
            .inner
            .active_contract()
            .expect("Active guarded executive must own an active contract")
            .clone();
        let guard = guard_humanoid_skill_actuation(
            &contract,
            actuation_policies,
            actuation_evidence,
            envelopes,
        );
        let executive = self
            .inner
            .tick(&guard.guarded_envelopes, runtime_evidence);
        HumanoidGuardedSkillExecutiveReport {
            actuation_guard: Some(guard),
            executive,
        }
    }

    pub fn acknowledge_recovery_complete(&mut self) -> HumanoidGuardedSkillExecutiveReport {
        HumanoidGuardedSkillExecutiveReport {
            actuation_guard: None,
            executive: self.inner.acknowledge_recovery_complete(),
        }
    }

    pub fn clear_completed(&mut self) {
        self.inner.clear_completed();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actuator_controllability::{
        ContactWrenchAxis, ContactWrenchAxisMargin, ContactWrenchMarginAssessment,
        HumanoidActuationControllabilityAssessment,
    };
    use crate::capability_envelope::{
        HumanInteractionEvidence, HumanoidCapabilityRestriction,
        HumanoidNominalCapabilityProfile, derive_humanoid_capability_envelope,
    };
    use crate::contact_site::HumanoidContactSite;
    use crate::execution::HumanoidAuthorityEnvelope;
    use crate::full_dynamics::DynamicsComponentSource;
    use crate::morphology::HumanoidMorphology;
    use crate::qualification::HumanoidQualificationSubject;
    use crate::skill_executive::{
        HumanoidSkillLifecycleViolation, HumanoidSkillTransition,
    };
    use crate::skill_runtime::{
        HumanoidLocomotionMode, HumanoidSkillIntent, HumanoidSkillQualificationSet,
        HumanoidSkillRecoveryPolicy, HumanoidSkillRequirementRole,
        compile_humanoid_skill_contract,
    };
    use crate::typed_actuation_capability::{
        HUMANOID_TYPED_ACTUATION_POLICY_SCHEMA_VERSION,
        TypedHumanoidActuationCapabilityPolicy, TypedHumanoidContactActuationRequirement,
    };
    use crate::types::{ActuationMode, HumanoidTask};

    fn subject(task: HumanoidTask) -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            task,
            ActuationMode::NormalizedTorque,
            "guarded-executive-test-v1",
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

    fn policy(
        task: HumanoidTask,
        role: HumanoidSkillRequirementRole,
        site: HumanoidContactSite,
    ) -> HumanoidSkillActuationPolicyEntry {
        let subject = subject(task);
        let profile = profile(task);
        HumanoidSkillActuationPolicyEntry {
            role,
            subject: subject.clone(),
            profile: profile.clone(),
            policy: TypedHumanoidActuationCapabilityPolicy {
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
            },
        }
    }

    fn assessment(
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
            policy(
                HumanoidTask::Grasp,
                HumanoidSkillRequirementRole::Manipulation,
                HumanoidContactSite::RightHand,
            ),
            policy(
                HumanoidTask::Walk,
                HumanoidSkillRequirementRole::Locomotion,
                HumanoidContactSite::RightFoot,
            ),
        ]
    }

    fn nominal_evidence() -> Vec<HumanoidSkillActuationEvidenceEntry> {
        vec![
            assessment(HumanoidTask::Grasp, HumanoidContactSite::RightHand, 0.9, 90.0),
            assessment(HumanoidTask::Walk, HumanoidContactSite::RightFoot, 0.9, 90.0),
        ]
    }

    fn runtime_evidence() -> HumanoidSkillRuntimeEvidence {
        HumanoidSkillRuntimeEvidence {
            goal_authority_valid: true,
            load_retained: true,
            human_proximity_valid: true,
            human_contact_consent: false,
            protective_preempted: false,
            objective_satisfied: false,
        }
    }

    #[test]
    fn nominal_carry_starts_through_guarded_path() {
        let mut executive = HumanoidGuardedSkillExecutive::new();
        let report = executive
            .start(
                carry_contract(),
                &[envelope(HumanoidTask::Grasp), envelope(HumanoidTask::Walk)],
                &policies(),
                &nominal_evidence(),
                runtime_evidence(),
            )
            .unwrap();
        assert!(report.actuation_guard.as_ref().unwrap().composition_complete);
        assert_eq!(report.executive.state, HumanoidSkillExecutiveState::Active);
    }

    #[test]
    fn missing_actuation_policy_rejects_before_lower_executive_start() {
        let mut policies = policies();
        policies.retain(|entry| entry.subject.task != HumanoidTask::Grasp);
        let mut executive = HumanoidGuardedSkillExecutive::new();
        let rejection = executive
            .start(
                carry_contract(),
                &[envelope(HumanoidTask::Grasp), envelope(HumanoidTask::Walk)],
                &policies,
                &nominal_evidence(),
                runtime_evidence(),
            )
            .unwrap_err();
        assert!(!rejection.actuation_guard.composition_complete);
        assert!(rejection.executive.is_none());
        assert_eq!(executive.state(), HumanoidSkillExecutiveState::Idle);
    }

    #[test]
    fn hand_authority_loss_during_carry_enters_secure_load_recovery() {
        let envelopes = [envelope(HumanoidTask::Grasp), envelope(HumanoidTask::Walk)];
        let mut executive = HumanoidGuardedSkillExecutive::new();
        executive
            .start(
                carry_contract(),
                &envelopes,
                &policies(),
                &nominal_evidence(),
                runtime_evidence(),
            )
            .unwrap();

        let degraded = vec![
            assessment(HumanoidTask::Grasp, HumanoidContactSite::RightHand, 0.3, 30.0),
            assessment(HumanoidTask::Walk, HumanoidContactSite::RightFoot, 0.9, 90.0),
        ];
        let report = executive.tick(
            &envelopes,
            &policies(),
            &degraded,
            runtime_evidence(),
        );
        assert_eq!(
            report.executive.state,
            HumanoidSkillExecutiveState::Recovering(
                HumanoidSkillRecoveryPolicy::SecureLoadThenReplan
            )
        );
        assert_eq!(
            report.executive.transition,
            HumanoidSkillTransition::EnteredRecovery(
                HumanoidSkillLifecycleViolation::CapabilityNoLongerAdmitted
            )
        );
    }

    #[test]
    fn foot_authority_loss_during_carry_also_invalidates_composite_goal() {
        let envelopes = [envelope(HumanoidTask::Grasp), envelope(HumanoidTask::Walk)];
        let mut executive = HumanoidGuardedSkillExecutive::new();
        executive
            .start(
                carry_contract(),
                &envelopes,
                &policies(),
                &nominal_evidence(),
                runtime_evidence(),
            )
            .unwrap();
        let degraded = vec![
            assessment(HumanoidTask::Grasp, HumanoidContactSite::RightHand, 0.9, 90.0),
            assessment(HumanoidTask::Walk, HumanoidContactSite::RightFoot, 0.3, 30.0),
        ];
        let report = executive.tick(
            &envelopes,
            &policies(),
            &degraded,
            runtime_evidence(),
        );
        assert!(matches!(
            report.executive.transition,
            HumanoidSkillTransition::EnteredRecovery(
                HumanoidSkillLifecycleViolation::CapabilityNoLongerAdmitted
            )
        ));
    }
}
