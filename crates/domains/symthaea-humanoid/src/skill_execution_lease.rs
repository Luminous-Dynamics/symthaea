// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Borrowed one-epoch skill authority for local command synthesis.
//!
//! `HumanoidSkillExecutionContext` is intentionally still non-motor data, but it
//! is detached from its owner after `accept_permit`. A caller could therefore
//! hold that context while another mutable operation advances the guarded skill
//! epoch. This module adds a stronger local option: an accepted execution lease
//! borrows the active contract directly from the permitted-skill owner.
//!
//! In safe Rust, the owner cannot be mutably borrowed for `tick`, recovery, or
//! completion while this lease remains alive. The lifetime therefore closes the
//! local permit-to-synthesis TOCTOU window without turning the skill layer into
//! motor authority or relying on a second wall-clock check.

use std::num::NonZeroU64;

use crate::permitted_skill_executive::{
    HumanoidPermittedSkillExecutive, HumanoidSkillExecutionPermit, HumanoidSkillPermitError,
};
use crate::skill_runtime::HumanoidSkillContract;

/// Process-local lease over one exact currently active skill contract.
///
/// The embedded contract reference is what carries the lifetime relationship to
/// the permit owner. This type deliberately does not implement `Clone`, `Copy`,
/// `Serialize`, or `Deserialize`.
#[derive(Debug)]
pub struct HumanoidSkillExecutionLease<'a> {
    validation_epoch: NonZeroU64,
    contract: &'a HumanoidSkillContract,
}

impl<'a> HumanoidSkillExecutionLease<'a> {
    pub const fn validation_epoch(&self) -> NonZeroU64 {
        self.validation_epoch
    }

    pub const fn contract(&self) -> &'a HumanoidSkillContract {
        self.contract
    }
}

/// Consume a one-epoch permit through the existing canonical owner check, then
/// retain a borrow of the exact active contract for the remainder of the local
/// synthesis window.
///
/// Because the returned lease borrows `owner`, safe Rust prevents a concurrent
/// or subsequent mutable epoch transition until the lease is consumed/dropped.
pub fn accept_humanoid_skill_execution_lease<'a>(
    owner: &'a HumanoidPermittedSkillExecutive,
    permit: HumanoidSkillExecutionPermit,
) -> Result<HumanoidSkillExecutionLease<'a>, HumanoidSkillPermitError> {
    let context = owner.accept_permit(permit)?;
    let validation_epoch = context.validation_epoch();
    let admitted_contract = context.into_contract();
    let active = owner
        .active_contract()
        .ok_or(HumanoidSkillPermitError::NoActiveSkill)?;
    if active != &admitted_contract {
        return Err(HumanoidSkillPermitError::ContractMismatch);
    }
    if owner.validation_epoch() != validation_epoch.get() {
        return Err(HumanoidSkillPermitError::StaleValidationEpoch);
    }

    Ok(HumanoidSkillExecutionLease {
        validation_epoch,
        contract: active,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actuator_controllability::{
        ContactWrenchAxis, ContactWrenchAxisMargin, ContactWrenchMarginAssessment,
        HumanoidActuationControllabilityAssessment,
    };
    use crate::capability_envelope::{
        HumanInteractionEvidence, HumanoidCapabilityEnvelope, HumanoidCapabilityRestriction,
        HumanoidNominalCapabilityProfile, derive_humanoid_capability_envelope,
    };
    use crate::contact_site::HumanoidContactSite;
    use crate::execution::HumanoidAuthorityEnvelope;
    use crate::full_dynamics::DynamicsComponentSource;
    use crate::morphology::HumanoidMorphology;
    use crate::qualification::HumanoidQualificationSubject;
    use crate::skill_actuation_guard::{
        HumanoidSkillActuationEvidenceEntry, HumanoidSkillActuationPolicyEntry,
    };
    use crate::skill_executive::HumanoidSkillRuntimeEvidence;
    use crate::skill_runtime::{
        HumanoidSkillIntent, HumanoidSkillQualificationSet, HumanoidSkillRequirementRole,
        compile_humanoid_skill_contract,
    };
    use crate::typed_actuation_capability::{
        HUMANOID_TYPED_ACTUATION_POLICY_SCHEMA_VERSION, TypedHumanoidActuationCapabilityPolicy,
        TypedHumanoidContactActuationRequirement,
    };
    use crate::types::{ActuationMode, HumanoidTask};

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dmc21,
            HumanoidTask::Stand,
            ActuationMode::NormalizedTorque,
            "lease-test-backend-v1",
        )
    }

    fn profile() -> HumanoidNominalCapabilityProfile {
        HumanoidNominalCapabilityProfile::new(
            &subject(), 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 80.0, 0.0, 0.8, true,
        )
    }

    fn envelope() -> HumanoidCapabilityEnvelope {
        derive_humanoid_capability_envelope(
            &subject(),
            &profile(),
            HumanoidAuthorityEnvelope::fully_admitted(),
            HumanInteractionEvidence::no_human_present(),
        )
    }

    fn policy() -> HumanoidSkillActuationPolicyEntry {
        let subject = subject();
        HumanoidSkillActuationPolicyEntry {
            role: HumanoidSkillRequirementRole::Posture,
            profile: profile(),
            policy: TypedHumanoidActuationCapabilityPolicy {
                schema_version: HUMANOID_TYPED_ACTUATION_POLICY_SCHEMA_VERSION,
                subject_fingerprint: subject.fingerprint(),
                authority_profile_id: "joint-authority-v1".into(),
                calibration_fingerprint: 42,
                dynamics_model_id: "dynamics-v1".into(),
                requirements: vec![TypedHumanoidContactActuationRequirement {
                    site: HumanoidContactSite::RightFoot,
                    axis: ContactWrenchAxis::ForceZ,
                    nominal_min_retained_fraction: 0.8,
                    degraded_min_retained_fraction: 0.5,
                    nominal_min_retained_wrench: 80.0,
                    degraded_min_retained_wrench: 50.0,
                }],
                degraded_restriction: HumanoidCapabilityRestriction {
                    max_horizontal_speed_mps: 0.0,
                    max_turn_rate_rad_s: 0.0,
                    max_end_effector_speed_mps: 0.0,
                    max_payload_kg: 0.0,
                    max_object_contact_force_n: 0.0,
                    max_human_contact_force_n: 0.0,
                },
            },
            subject,
        }
    }

    fn evidence() -> HumanoidSkillActuationEvidenceEntry {
        HumanoidSkillActuationEvidenceEntry {
            subject_fingerprint: subject().fingerprint(),
            assessment: HumanoidActuationControllabilityAssessment {
                morphology: HumanoidMorphology::Dmc21,
                authority_sequence: 1,
                authority_age_s: 0.01,
                authority_profile_id: "joint-authority-v1".into(),
                calibration_fingerprint: 42,
                dynamics_model_id: "dynamics-v1".into(),
                sites: vec![ContactWrenchMarginAssessment {
                    site_id: HumanoidContactSite::RightFoot.canonical_id().into(),
                    contact_confidence: 1.0,
                    jacobian_source: DynamicsComponentSource::SimulatorSolver,
                    actuator_limit_source: DynamicsComponentSource::SystemIdentification,
                    axes: vec![ContactWrenchAxisMargin {
                        axis: ContactWrenchAxis::ForceZ,
                        actuated_support_present: true,
                        nominal_limit: Some(100.0),
                        retained_limit: Some(90.0),
                        retained_fraction: 0.9,
                        limiting_joint: Some(0),
                    }],
                    minimum_retained_fraction: 0.9,
                }],
            },
        }
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

    fn started() -> (
        HumanoidPermittedSkillExecutive,
        HumanoidSkillExecutionPermit,
        Vec<HumanoidCapabilityEnvelope>,
        Vec<HumanoidSkillActuationPolicyEntry>,
        Vec<HumanoidSkillActuationEvidenceEntry>,
    ) {
        let qualifications = HumanoidSkillQualificationSet::new(vec![subject()]);
        let contract = compile_humanoid_skill_contract(HumanoidSkillIntent::Stand, &qualifications)
            .unwrap();
        let envelopes = vec![envelope()];
        let policies = vec![policy()];
        let evidence = vec![evidence()];
        let mut owner = HumanoidPermittedSkillExecutive::new();
        let permit = owner
            .start(contract, &envelopes, &policies, &evidence, runtime())
            .unwrap()
            .permit
            .unwrap();
        (owner, permit, envelopes, policies, evidence)
    }

    #[test]
    fn current_permit_becomes_borrowed_execution_lease() {
        let (owner, permit, _, _, _) = started();
        let lease = accept_humanoid_skill_execution_lease(&owner, permit).unwrap();
        assert_eq!(lease.validation_epoch().get(), 1);
        assert_eq!(lease.contract().intent, HumanoidSkillIntent::Stand);
    }

    #[test]
    fn older_permit_cannot_lease_after_epoch_advances() {
        let (mut owner, permit, envelopes, policies, evidence) = started();
        let _ = owner.tick(&envelopes, &policies, &evidence, runtime());
        assert_eq!(
            accept_humanoid_skill_execution_lease(&owner, permit).unwrap_err(),
            HumanoidSkillPermitError::StaleValidationEpoch
        );
    }
}
