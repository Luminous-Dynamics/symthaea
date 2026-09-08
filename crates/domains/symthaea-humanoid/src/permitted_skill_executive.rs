// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! One-validation-epoch execution permits for continuously guarded humanoid skills.
//!
//! The guarded skill executive proves that a semantic skill remains admitted by
//! current capability, authority, consent, load-retention, and task-local
//! actuation evidence. This module turns one successful guarded validation into
//! an opaque, process-local permit that can be consumed by a lower execution
//! boundary in a later integration layer.
//!
//! A permit is deliberately **not** motor authority. It proves only that the
//! exact closed skill contract was still semantically admitted at one monotonic
//! validation epoch. Prepared-command synthesis, final safety projection, HAL
//! session currentness, and exact actuator-command dispatch remain separate
//! authority boundaries.

use std::num::NonZeroU64;

use crate::capability_envelope::HumanoidCapabilityEnvelope;
use crate::guarded_skill_executive::{
    HumanoidGuardedSkillExecutive, HumanoidGuardedSkillExecutiveReport,
    HumanoidGuardedSkillStartRejection,
};
use crate::skill_actuation_guard::{
    HumanoidSkillActuationEvidenceEntry, HumanoidSkillActuationPolicyEntry,
};
use crate::skill_executive::{HumanoidSkillExecutiveState, HumanoidSkillRuntimeEvidence};
use crate::skill_runtime::HumanoidSkillContract;

/// Opaque proof that one exact skill contract survived one complete guarded
/// validation epoch.
///
/// Intentionally not `Clone`, `Copy`, `Serialize`, or `Deserialize`. Persisted
/// data must be revalidated after restore; serialized bytes must never recreate
/// live execution authority.
#[derive(Debug)]
pub struct HumanoidSkillExecutionPermit {
    validation_epoch: NonZeroU64,
    contract: HumanoidSkillContract,
}

impl HumanoidSkillExecutionPermit {
    pub const fn validation_epoch(&self) -> NonZeroU64 {
        self.validation_epoch
    }

    /// The exact closed contract admitted at this epoch. Cloning this contract
    /// clones data only; it does not clone this permit or execution authority.
    pub fn contract(&self) -> &HumanoidSkillContract {
        &self.contract
    }
}

/// Still-non-motor context returned only after a permit is consumed and checked
/// against the current permit owner. A later prepared-command boundary can
/// consume this value without accepting freely constructed skill data.
#[derive(Debug)]
pub struct HumanoidSkillExecutionContext {
    validation_epoch: NonZeroU64,
    contract: HumanoidSkillContract,
}

impl HumanoidSkillExecutionContext {
    pub const fn validation_epoch(&self) -> NonZeroU64 {
        self.validation_epoch
    }

    pub fn contract(&self) -> &HumanoidSkillContract {
        &self.contract
    }

    pub fn into_contract(self) -> HumanoidSkillContract {
        self.contract
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidSkillPermitError {
    EpochExhausted,
    NoActiveSkill,
    StaleValidationEpoch,
    ContractMismatch,
}

/// Result of one guarded lifecycle operation. A permit is present only when the
/// resulting executive state is `Active` and the validation epoch advanced
/// successfully.
#[derive(Debug)]
pub struct HumanoidPermittedSkillStep {
    pub report: HumanoidGuardedSkillExecutiveReport,
    pub permit: Option<HumanoidSkillExecutionPermit>,
}

/// Preferred semantic execution owner for callers that need a positive,
/// one-epoch proof before entering lower prepared-command execution.
///
/// This wrapper intentionally does not implement Serde or Clone. The underlying
/// guarded executive still has a compatibility serialization surface in its own
/// module, but restoring those bytes cannot recreate this wrapper's live epoch
/// or any outstanding permit.
#[derive(Debug)]
pub struct HumanoidPermittedSkillExecutive {
    inner: HumanoidGuardedSkillExecutive,
    validation_epoch: u64,
    epoch_exhausted: bool,
}

impl Default for HumanoidPermittedSkillExecutive {
    fn default() -> Self {
        Self::new()
    }
}

impl HumanoidPermittedSkillExecutive {
    pub const fn new() -> Self {
        Self {
            inner: HumanoidGuardedSkillExecutive::new(),
            validation_epoch: 0,
            epoch_exhausted: false,
        }
    }

    pub const fn state(&self) -> HumanoidSkillExecutiveState {
        self.inner.state()
    }

    pub fn active_contract(&self) -> Option<&HumanoidSkillContract> {
        self.inner.active_contract()
    }

    /// Last locally owned validation epoch. Zero means no guarded lifecycle
    /// operation has completed through this wrapper yet.
    pub const fn validation_epoch(&self) -> u64 {
        self.validation_epoch
    }

    pub const fn epoch_exhausted(&self) -> bool {
        self.epoch_exhausted
    }

    /// Start a skill through the complete #968 guarded path. A successful Active
    /// result advances the epoch and carries exactly one non-cloneable permit.
    pub fn start(
        &mut self,
        contract: HumanoidSkillContract,
        envelopes: &[HumanoidCapabilityEnvelope],
        actuation_policies: &[HumanoidSkillActuationPolicyEntry],
        actuation_evidence: &[HumanoidSkillActuationEvidenceEntry],
        runtime_evidence: HumanoidSkillRuntimeEvidence,
    ) -> Result<HumanoidPermittedSkillStep, HumanoidGuardedSkillStartRejection> {
        let report = self.inner.start(
            contract,
            envelopes,
            actuation_policies,
            actuation_evidence,
            runtime_evidence,
        )?;
        let epoch = self.advance_epoch();
        let permit = self.issue_if_active(epoch);
        Ok(HumanoidPermittedSkillStep { report, permit })
    }

    /// Revalidate an Active skill through current guarded envelopes. Every call
    /// advances the local epoch, including transitions into recovery/completion;
    /// therefore any permit from the previous epoch becomes stale immediately.
    pub fn tick(
        &mut self,
        envelopes: &[HumanoidCapabilityEnvelope],
        actuation_policies: &[HumanoidSkillActuationPolicyEntry],
        actuation_evidence: &[HumanoidSkillActuationEvidenceEntry],
        runtime_evidence: HumanoidSkillRuntimeEvidence,
    ) -> HumanoidPermittedSkillStep {
        let report = self.inner.tick(
            envelopes,
            actuation_policies,
            actuation_evidence,
            runtime_evidence,
        );
        let epoch = self.advance_epoch();
        let permit = self.issue_if_active(epoch);
        HumanoidPermittedSkillStep { report, permit }
    }

    /// Consume and recheck a permit against the current owner state. This is the
    /// only way to obtain a `HumanoidSkillExecutionContext`.
    pub fn accept_permit(
        &self,
        permit: HumanoidSkillExecutionPermit,
    ) -> Result<HumanoidSkillExecutionContext, HumanoidSkillPermitError> {
        if self.epoch_exhausted {
            return Err(HumanoidSkillPermitError::EpochExhausted);
        }
        if self.inner.state() != HumanoidSkillExecutiveState::Active {
            return Err(HumanoidSkillPermitError::NoActiveSkill);
        }
        if permit.validation_epoch.get() != self.validation_epoch {
            return Err(HumanoidSkillPermitError::StaleValidationEpoch);
        }
        let Some(active) = self.inner.active_contract() else {
            return Err(HumanoidSkillPermitError::NoActiveSkill);
        };
        if active != &permit.contract {
            return Err(HumanoidSkillPermitError::ContractMismatch);
        }
        Ok(HumanoidSkillExecutionContext {
            validation_epoch: permit.validation_epoch,
            contract: permit.contract,
        })
    }

    /// Recovery acknowledgement invalidates every older permit even though it
    /// does not itself issue a new one.
    pub fn acknowledge_recovery_complete(&mut self) -> HumanoidGuardedSkillExecutiveReport {
        let report = self.inner.acknowledge_recovery_complete();
        let _ = self.advance_epoch();
        report
    }

    /// Clearing completion similarly advances the epoch so no permit from the
    /// completed skill can survive into the next Idle period.
    pub fn clear_completed(&mut self) {
        self.inner.clear_completed();
        let _ = self.advance_epoch();
    }

    fn issue_if_active(
        &self,
        epoch: Option<NonZeroU64>,
    ) -> Option<HumanoidSkillExecutionPermit> {
        let epoch = epoch?;
        if self.inner.state() != HumanoidSkillExecutiveState::Active {
            return None;
        }
        let contract = self.inner.active_contract()?.clone();
        Some(HumanoidSkillExecutionPermit {
            validation_epoch: epoch,
            contract,
        })
    }

    fn advance_epoch(&mut self) -> Option<NonZeroU64> {
        if self.epoch_exhausted {
            return None;
        }
        let Some(next) = self.validation_epoch.checked_add(1) else {
            self.epoch_exhausted = true;
            return None;
        };
        self.validation_epoch = next;
        NonZeroU64::new(next)
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
    use crate::skill_runtime::{
        HumanoidLocomotionMode, HumanoidSkillIntent, HumanoidSkillQualificationSet,
        HumanoidSkillRequirementRole, compile_humanoid_skill_contract,
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
            profile: profile(task),
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
            subject,
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

    fn envelopes() -> Vec<HumanoidCapabilityEnvelope> {
        vec![envelope(HumanoidTask::Grasp), envelope(HumanoidTask::Walk)]
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

    fn start_carry(executive: &mut HumanoidPermittedSkillExecutive) -> HumanoidPermittedSkillStep {
        executive
            .start(
                carry_contract(),
                &envelopes(),
                &policies(),
                &nominal_evidence(),
                runtime_evidence(),
            )
            .unwrap()
    }

    #[test]
    fn successful_guarded_start_issues_epoch_one_permit() {
        let mut executive = HumanoidPermittedSkillExecutive::new();
        let step = start_carry(&mut executive);
        assert_eq!(executive.state(), HumanoidSkillExecutiveState::Active);
        assert_eq!(executive.validation_epoch(), 1);
        let permit = step.permit.expect("Active start should issue permit");
        assert_eq!(permit.validation_epoch().get(), 1);
        assert_eq!(permit.contract().requirements.len(), 2);
        assert_eq!(permit.contract(), executive.active_contract().unwrap());
    }

    #[test]
    fn newer_validation_epoch_rejects_older_permit() {
        let mut executive = HumanoidPermittedSkillExecutive::new();
        let old = start_carry(&mut executive).permit.unwrap();
        let current = executive.tick(
            &envelopes(),
            &policies(),
            &nominal_evidence(),
            runtime_evidence(),
        );
        assert_eq!(executive.validation_epoch(), 2);
        assert_eq!(
            executive.accept_permit(old).unwrap_err(),
            HumanoidSkillPermitError::StaleValidationEpoch
        );
        let context = executive.accept_permit(current.permit.unwrap()).unwrap();
        assert_eq!(context.validation_epoch().get(), 2);
        assert_eq!(context.contract().requirements.len(), 2);
    }

    #[test]
    fn recovery_transition_issues_no_permit_and_invalidates_old_epoch() {
        let mut executive = HumanoidPermittedSkillExecutive::new();
        let old = start_carry(&mut executive).permit.unwrap();
        let mut revoked = runtime_evidence();
        revoked.goal_authority_valid = false;
        let recovery = executive.tick(
            &envelopes(),
            &policies(),
            &nominal_evidence(),
            revoked,
        );
        assert!(matches!(
            recovery.report.executive.state,
            HumanoidSkillExecutiveState::Recovering(_)
        ));
        assert!(recovery.permit.is_none());
        assert_eq!(executive.validation_epoch(), 2);
        assert_eq!(
            executive.accept_permit(old).unwrap_err(),
            HumanoidSkillPermitError::NoActiveSkill
        );
    }

    #[test]
    fn accepted_permit_preserves_exact_composite_contract() {
        let expected = carry_contract();
        let mut executive = HumanoidPermittedSkillExecutive::new();
        let permit = executive
            .start(
                expected.clone(),
                &envelopes(),
                &policies(),
                &nominal_evidence(),
                runtime_evidence(),
            )
            .unwrap()
            .permit
            .unwrap();
        let context = executive.accept_permit(permit).unwrap();
        assert_eq!(context.contract(), &expected);
        assert_eq!(context.contract().requirements.len(), 2);
    }
}
