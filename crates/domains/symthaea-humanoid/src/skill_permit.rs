// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ephemeral process-local execution permits for continuously revalidated skills.
//!
//! A successful semantic/capability/actuation revalidation is still only an
//! upstream fact unless downstream execution can prove that it is consuming the
//! same fresh decision. This module turns a successful guarded validation into a
//! short-lived, non-serializable, non-cloneable capability token.
//!
//! The token is intentionally **skill-atomic**. Composite skills such as Carry
//! receive one permit covering every concurrently admitted requirement from one
//! validation epoch; locomotion and manipulation cannot be independently peeled
//! out as if they had been authorized in separate worlds.
//!
//! Freshness is partly enforced by Rust borrowing rather than by wall-clock
//! convention: `HumanoidSkillValidationCycle` retains an exclusive mutable borrow
//! of the permit executive. While a cycle (or a permit borrowing that cycle) is
//! alive, the executive cannot be revalidated or advanced. Dropping the cycle is
//! therefore required before a new validation epoch can be minted.

use crate::capability_envelope::HumanoidCapabilityEnvelope;
use crate::guarded_skill_executive::{
    HumanoidGuardedSkillExecutive, HumanoidGuardedSkillExecutiveReport,
    HumanoidGuardedSkillStartRejection,
};
use crate::skill_actuation_guard::{
    HumanoidSkillActuationEvidenceEntry, HumanoidSkillActuationPolicyEntry,
};
use crate::skill_executive::{
    HumanoidSkillExecutiveState, HumanoidSkillRuntimeEvidence,
};
use crate::skill_runtime::{
    HumanoidSkillContract, HumanoidSkillIntent, HumanoidSkillRequirement,
};
use crate::morphology::HumanoidMorphology;
use crate::types::ActuationMode;

/// Preferred process-local executive for callers that need proof of fresh
/// semantic admission before compiling a motor intent.
///
/// `validation_epoch` is monotonic for the lifetime of this object. It is not a
/// cryptographic nonce and must not be persisted or treated as distributed
/// authorization; Xenia/Mycelix task authority remains a separate boundary.
pub struct HumanoidPermitSkillExecutive {
    guarded: HumanoidGuardedSkillExecutive,
    validation_epoch: u64,
}

impl Default for HumanoidPermitSkillExecutive {
    fn default() -> Self {
        Self::new()
    }
}

impl HumanoidPermitSkillExecutive {
    pub const fn new() -> Self {
        Self {
            guarded: HumanoidGuardedSkillExecutive::new(),
            validation_epoch: 0,
        }
    }

    pub const fn state(&self) -> HumanoidSkillExecutiveState {
        self.guarded.state()
    }

    pub const fn validation_epoch(&self) -> u64 {
        self.validation_epoch
    }

    pub fn active_contract(&self) -> Option<&HumanoidSkillContract> {
        self.guarded.active_contract()
    }

    /// Start and fully validate a new skill, then mint the first execution cycle.
    /// Epoch exhaustion is checked before mutating the guarded executive.
    pub fn start_and_issue<'a>(
        &'a mut self,
        contract: HumanoidSkillContract,
        envelopes: &[HumanoidCapabilityEnvelope],
        actuation_policies: &[HumanoidSkillActuationPolicyEntry],
        actuation_evidence: &[HumanoidSkillActuationEvidenceEntry],
        runtime_evidence: HumanoidSkillRuntimeEvidence,
    ) -> Result<HumanoidSkillValidationCycle<'a>, HumanoidPermitStartError> {
        let next_epoch = self
            .validation_epoch
            .checked_add(1)
            .ok_or(HumanoidPermitStartError::EpochExhausted)?;

        let report = self
            .guarded
            .start(
                contract,
                envelopes,
                actuation_policies,
                actuation_evidence,
                runtime_evidence,
            )
            .map_err(HumanoidPermitStartError::Guarded)?;

        if report.executive.state != HumanoidSkillExecutiveState::Active {
            return Err(HumanoidPermitStartError::NoActiveContract);
        }
        let active_contract = self
            .guarded
            .active_contract()
            .cloned()
            .ok_or(HumanoidPermitStartError::NoActiveContract)?;

        self.validation_epoch = next_epoch;
        Ok(HumanoidSkillValidationCycle {
            executive: self,
            epoch: next_epoch,
            contract: active_contract,
            report,
        })
    }

    /// Revalidate the currently Active skill and mint exactly one new process-
    /// local validation cycle when it remains admitted.
    ///
    /// If revalidation enters recovery/completion, no permit is issued and the
    /// transition report is returned to the caller.
    pub fn revalidate_and_issue<'a>(
        &'a mut self,
        envelopes: &[HumanoidCapabilityEnvelope],
        actuation_policies: &[HumanoidSkillActuationPolicyEntry],
        actuation_evidence: &[HumanoidSkillActuationEvidenceEntry],
        runtime_evidence: HumanoidSkillRuntimeEvidence,
    ) -> Result<HumanoidSkillValidationCycle<'a>, HumanoidPermitRevalidationError> {
        if self.guarded.state() != HumanoidSkillExecutiveState::Active {
            return Err(HumanoidPermitRevalidationError::NotActive(
                self.guarded.state(),
            ));
        }
        let next_epoch = self
            .validation_epoch
            .checked_add(1)
            .ok_or(HumanoidPermitRevalidationError::EpochExhausted)?;

        let report = self.guarded.tick(
            envelopes,
            actuation_policies,
            actuation_evidence,
            runtime_evidence,
        );
        if report.executive.state != HumanoidSkillExecutiveState::Active {
            return Err(HumanoidPermitRevalidationError::Transitioned(report));
        }
        let active_contract = self
            .guarded
            .active_contract()
            .cloned()
            .ok_or(HumanoidPermitRevalidationError::NoActiveContract)?;

        self.validation_epoch = next_epoch;
        Ok(HumanoidSkillValidationCycle {
            executive: self,
            epoch: next_epoch,
            contract: active_contract,
            report,
        })
    }

    pub fn acknowledge_recovery_complete(&mut self) -> HumanoidGuardedSkillExecutiveReport {
        self.guarded.acknowledge_recovery_complete()
    }

    pub fn clear_completed(&mut self) {
        self.guarded.clear_completed();
    }
}

#[derive(Debug)]
pub enum HumanoidPermitStartError {
    EpochExhausted,
    NoActiveContract,
    Guarded(HumanoidGuardedSkillStartRejection),
}

#[derive(Debug)]
pub enum HumanoidPermitRevalidationError {
    EpochExhausted,
    NoActiveContract,
    NotActive(HumanoidSkillExecutiveState),
    Transitioned(HumanoidGuardedSkillExecutiveReport),
}

/// One successful guarded validation epoch.
///
/// The private mutable executive borrow is the process-local freshness lock.
/// There is deliberately no `Clone` or serialization implementation.
pub struct HumanoidSkillValidationCycle<'a> {
    #[allow(dead_code)]
    executive: &'a mut HumanoidPermitSkillExecutive,
    epoch: u64,
    contract: HumanoidSkillContract,
    report: HumanoidGuardedSkillExecutiveReport,
}

impl<'a> HumanoidSkillValidationCycle<'a> {
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn report(&self) -> &HumanoidGuardedSkillExecutiveReport {
        &self.report
    }

    pub fn contract(&self) -> &HumanoidSkillContract {
        &self.contract
    }

    /// Mint an opaque skill-atomic permit borrowing this exact validation cycle.
    /// The permit cannot outlive the cycle and cannot be cloned or serialized.
    pub fn skill_permit<'cycle>(&'cycle self) -> HumanoidSkillExecutionPermit<'cycle> {
        HumanoidSkillExecutionPermit {
            epoch: self.epoch,
            contract: &self.contract,
        }
    }
}

/// Opaque process-local proof that one complete skill contract remained admitted
/// at a specific guarded validation epoch.
///
/// Fields are private on purpose. Downstream code may inspect the semantic
/// demands through accessors but cannot construct a permit without going through
/// the guarded validation cycle.
pub struct HumanoidSkillExecutionPermit<'cycle> {
    epoch: u64,
    contract: &'cycle HumanoidSkillContract,
}

impl<'cycle> HumanoidSkillExecutionPermit<'cycle> {
    pub const fn epoch(&self) -> u64 {
        self.epoch
    }

    pub const fn morphology(&self) -> HumanoidMorphology {
        self.contract.morphology
    }

    pub const fn actuation_mode(&self) -> ActuationMode {
        self.contract.actuation_mode
    }

    pub fn backend_profile_id(&self) -> &str {
        &self.contract.backend_profile_id
    }

    pub const fn intent(&self) -> HumanoidSkillIntent {
        self.contract.intent
    }

    /// Every concurrently admitted requirement for this semantic skill.
    pub fn requirements(&self) -> &[HumanoidSkillRequirement] {
        &self.contract.requirements
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
    use crate::qualification::HumanoidQualificationSubject;
    use crate::skill_runtime::{
        HumanoidLocomotionMode, HumanoidSkillIntent, HumanoidSkillQualificationSet,
        HumanoidSkillRequirementRole, compile_humanoid_skill_contract,
    };
    use crate::typed_actuation_capability::{
        HUMANOID_TYPED_ACTUATION_POLICY_SCHEMA_VERSION,
        TypedHumanoidActuationCapabilityPolicy, TypedHumanoidContactActuationRequirement,
    };
    use crate::types::HumanoidTask;

    fn subject(task: HumanoidTask) -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            task,
            ActuationMode::NormalizedTorque,
            "permit-test-backend-v1",
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
        HumanoidSkillActuationPolicyEntry {
            role,
            subject: subject.clone(),
            profile: profile(task),
            policy: TypedHumanoidActuationCapabilityPolicy {
                schema_version: HUMANOID_TYPED_ACTUATION_POLICY_SCHEMA_VERSION,
                subject_fingerprint: subject.fingerprint(),
                authority_profile_id: "permit-joints-v1".into(),
                calibration_fingerprint: 7,
                dynamics_model_id: "permit-dynamics-v1".into(),
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

    fn evidence(
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
                authority_profile_id: "permit-joints-v1".into(),
                calibration_fingerprint: 7,
                dynamics_model_id: "permit-dynamics-v1".into(),
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
            evidence(HumanoidTask::Grasp, HumanoidContactSite::RightHand, 0.9, 90.0),
            evidence(HumanoidTask::Walk, HumanoidContactSite::RightFoot, 0.9, 90.0),
        ]
    }

    fn runtime() -> HumanoidSkillRuntimeEvidence {
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
    fn carry_receives_one_atomic_permit_covering_both_requirements() {
        let mut executive = HumanoidPermitSkillExecutive::new();
        let cycle = executive
            .start_and_issue(
                carry_contract(),
                &[envelope(HumanoidTask::Grasp), envelope(HumanoidTask::Walk)],
                &policies(),
                &nominal_evidence(),
                runtime(),
            )
            .unwrap();
        let permit = cycle.skill_permit();
        assert_eq!(permit.epoch(), 1);
        assert_eq!(permit.requirements().len(), 2);
        assert!(permit.requirements().iter().any(|r| r.task == HumanoidTask::Grasp));
        assert!(permit.requirements().iter().any(|r| r.task == HumanoidTask::Walk));
    }

    #[test]
    fn validation_epoch_advances_only_when_a_new_active_cycle_is_minted() {
        let envelopes = [envelope(HumanoidTask::Grasp), envelope(HumanoidTask::Walk)];
        let policies = policies();
        let evidence = nominal_evidence();
        let mut executive = HumanoidPermitSkillExecutive::new();
        {
            let first = executive
                .start_and_issue(carry_contract(), &envelopes, &policies, &evidence, runtime())
                .unwrap();
            assert_eq!(first.epoch(), 1);
        }
        {
            let second = executive
                .revalidate_and_issue(&envelopes, &policies, &evidence, runtime())
                .unwrap();
            assert_eq!(second.epoch(), 2);
        }
        assert_eq!(executive.validation_epoch(), 2);
    }

    #[test]
    fn degraded_required_chain_transitions_without_minting_a_new_epoch() {
        let envelopes = [envelope(HumanoidTask::Grasp), envelope(HumanoidTask::Walk)];
        let policies = policies();
        let mut executive = HumanoidPermitSkillExecutive::new();
        {
            let _first = executive
                .start_and_issue(
                    carry_contract(),
                    &envelopes,
                    &policies,
                    &nominal_evidence(),
                    runtime(),
                )
                .unwrap();
        }
        let degraded = vec![
            evidence(HumanoidTask::Grasp, HumanoidContactSite::RightHand, 0.3, 30.0),
            evidence(HumanoidTask::Walk, HumanoidContactSite::RightFoot, 0.9, 90.0),
        ];
        let result = executive.revalidate_and_issue(&envelopes, &policies, &degraded, runtime());
        assert!(matches!(
            result,
            Err(HumanoidPermitRevalidationError::Transitioned(_))
        ));
        assert_eq!(executive.validation_epoch(), 1);
        assert!(matches!(
            executive.state(),
            HumanoidSkillExecutiveState::Recovering(_)
        ));
    }
}
