// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Typed adapter over contact-wrench actuation capability policy.
//!
//! The lower policy type preserves its string wire identity for compatibility.
//! New qualification/runtime code should prefer this typed adapter so canonical
//! contact-site spelling, assessment morphology, and subject identity are
//! checked before delegating to the lower evaluator.

use serde::{Deserialize, Serialize};

use crate::actuation_capability::{
    HUMANOID_ACTUATION_CAPABILITY_POLICY_SCHEMA_VERSION,
    HumanoidActuationCapabilityAction, HumanoidActuationCapabilityDecision,
    HumanoidActuationCapabilityPolicy, HumanoidContactActuationRequirement,
    assess_humanoid_actuation_capability,
};
use crate::actuator_controllability::{
    ContactWrenchAxis, HumanoidActuationControllabilityAssessment,
};
use crate::capability_envelope::{
    HumanoidCapabilityEnvelope, HumanoidCapabilityRestriction, HumanoidNominalCapabilityProfile,
};
use crate::contact_site::HumanoidContactSite;
use crate::qualification::HumanoidQualificationSubject;

pub const HUMANOID_TYPED_ACTUATION_POLICY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedHumanoidContactActuationRequirement {
    pub site: HumanoidContactSite,
    pub axis: ContactWrenchAxis,
    pub nominal_min_retained_fraction: f64,
    pub degraded_min_retained_fraction: f64,
    pub nominal_min_retained_wrench: f64,
    pub degraded_min_retained_wrench: f64,
}

impl TypedHumanoidContactActuationRequirement {
    fn to_wire(&self) -> HumanoidContactActuationRequirement {
        HumanoidContactActuationRequirement {
            site_id: self.site.canonical_id().to_string(),
            axis: self.axis,
            nominal_min_retained_fraction: self.nominal_min_retained_fraction,
            degraded_min_retained_fraction: self.degraded_min_retained_fraction,
            nominal_min_retained_wrench: self.nominal_min_retained_wrench,
            degraded_min_retained_wrench: self.degraded_min_retained_wrench,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedHumanoidActuationCapabilityPolicy {
    pub schema_version: u32,
    pub subject_fingerprint: u64,
    pub authority_profile_id: String,
    pub calibration_fingerprint: u64,
    pub dynamics_model_id: String,
    pub requirements: Vec<TypedHumanoidContactActuationRequirement>,
    pub degraded_restriction: HumanoidCapabilityRestriction,
}

impl TypedHumanoidActuationCapabilityPolicy {
    pub fn from_subject(
        subject: &HumanoidQualificationSubject,
        authority_profile_id: impl Into<String>,
        calibration_fingerprint: u64,
        dynamics_model_id: impl Into<String>,
        requirements: Vec<TypedHumanoidContactActuationRequirement>,
        degraded_restriction: HumanoidCapabilityRestriction,
    ) -> Self {
        Self {
            schema_version: HUMANOID_TYPED_ACTUATION_POLICY_SCHEMA_VERSION,
            subject_fingerprint: subject.fingerprint(),
            authority_profile_id: authority_profile_id.into(),
            calibration_fingerprint,
            dynamics_model_id: dynamics_model_id.into(),
            requirements,
            degraded_restriction,
        }
    }

    fn to_wire_policy(&self) -> HumanoidActuationCapabilityPolicy {
        HumanoidActuationCapabilityPolicy {
            schema_version: HUMANOID_ACTUATION_CAPABILITY_POLICY_SCHEMA_VERSION,
            subject_fingerprint: self.subject_fingerprint,
            authority_profile_id: self.authority_profile_id.clone(),
            calibration_fingerprint: self.calibration_fingerprint,
            dynamics_model_id: self.dynamics_model_id.clone(),
            requirements: self.requirements.iter().map(|r| r.to_wire()).collect(),
            degraded_restriction: self.degraded_restriction,
        }
    }

    pub fn validate_for(
        &self,
        subject: &HumanoidQualificationSubject,
        profile: &HumanoidNominalCapabilityProfile,
    ) -> bool {
        self.schema_version == HUMANOID_TYPED_ACTUATION_POLICY_SCHEMA_VERSION
            && self.subject_fingerprint != 0
            && self.subject_fingerprint == subject.fingerprint()
            && self.to_wire_policy().validate_for(subject, profile)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypedHumanoidActuationCapabilityFailure {
    InvalidTypedPolicy,
    AssessmentMorphologyMismatch,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypedHumanoidActuationCapabilityDecision {
    pub action: HumanoidActuationCapabilityAction,
    pub guard_failure: Option<TypedHumanoidActuationCapabilityFailure>,
    pub underlying: Option<HumanoidActuationCapabilityDecision>,
}

impl TypedHumanoidActuationCapabilityDecision {
    pub fn apply_to_envelope(&self, envelope: &mut HumanoidCapabilityEnvelope) {
        if self.action == HumanoidActuationCapabilityAction::Hold && self.underlying.is_none() {
            // Use an invalid concrete restriction to invoke the envelope's
            // established fail-closed ProtectiveOnly behavior without duplicating
            // private limit-mutation logic here.
            envelope.apply_restriction(HumanoidCapabilityRestriction {
                max_horizontal_speed_mps: f64::NAN,
                max_turn_rate_rad_s: f64::NAN,
                max_end_effector_speed_mps: f64::NAN,
                max_payload_kg: f64::NAN,
                max_object_contact_force_n: f64::NAN,
                max_human_contact_force_n: f64::NAN,
            });
            return;
        }
        if let Some(underlying) = &self.underlying {
            underlying.apply_to_envelope(envelope);
        }
    }
}

/// Preferred typed policy evaluation boundary.
///
/// This independently checks that the measured actuation assessment describes
/// the same morphology as the requested qualification subject before invoking
/// the lower policy evaluator.
pub fn assess_typed_humanoid_actuation_capability(
    subject: &HumanoidQualificationSubject,
    profile: &HumanoidNominalCapabilityProfile,
    policy: &TypedHumanoidActuationCapabilityPolicy,
    assessment: &HumanoidActuationControllabilityAssessment,
) -> TypedHumanoidActuationCapabilityDecision {
    if !policy.validate_for(subject, profile) {
        return TypedHumanoidActuationCapabilityDecision {
            action: HumanoidActuationCapabilityAction::Hold,
            guard_failure: Some(TypedHumanoidActuationCapabilityFailure::InvalidTypedPolicy),
            underlying: None,
        };
    }
    if assessment.morphology != subject.morphology {
        return TypedHumanoidActuationCapabilityDecision {
            action: HumanoidActuationCapabilityAction::Hold,
            guard_failure: Some(
                TypedHumanoidActuationCapabilityFailure::AssessmentMorphologyMismatch,
            ),
            underlying: None,
        };
    }

    let underlying = assess_humanoid_actuation_capability(
        subject,
        profile,
        &policy.to_wire_policy(),
        assessment,
    );
    TypedHumanoidActuationCapabilityDecision {
        action: underlying.action,
        guard_failure: None,
        underlying: Some(underlying),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actuator_controllability::{
        ContactWrenchAxisMargin, ContactWrenchMarginAssessment,
    };
    use crate::capability_envelope::{
        HumanInteractionEvidence, HumanoidCapabilityDisposition,
        derive_humanoid_capability_envelope,
    };
    use crate::execution::HumanoidAuthorityEnvelope;
    use crate::full_dynamics::DynamicsComponentSource;
    use crate::morphology::HumanoidMorphology;
    use crate::types::{ActuationMode, HumanoidTask};

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dmc21,
            HumanoidTask::Walk,
            ActuationMode::NormalizedTorque,
            "typed-site-test-backend-v1",
        )
    }

    fn profile() -> HumanoidNominalCapabilityProfile {
        HumanoidNominalCapabilityProfile::new(
            &subject(), 2.0, 0.5, 1.0, 0.8, 0.2, 10.0, 80.0, 20.0, 0.8, true,
        )
    }

    fn typed_policy() -> TypedHumanoidActuationCapabilityPolicy {
        TypedHumanoidActuationCapabilityPolicy::from_subject(
            &subject(),
            "joint-authority-v1",
            42,
            "dynamics-v1",
            vec![TypedHumanoidContactActuationRequirement {
                site: HumanoidContactSite::RightFoot,
                axis: ContactWrenchAxis::ForceZ,
                nominal_min_retained_fraction: 0.8,
                degraded_min_retained_fraction: 0.5,
                nominal_min_retained_wrench: 80.0,
                degraded_min_retained_wrench: 50.0,
            }],
            HumanoidCapabilityRestriction {
                max_horizontal_speed_mps: 0.5,
                max_turn_rate_rad_s: 0.3,
                max_end_effector_speed_mps: 0.4,
                max_payload_kg: 5.0,
                max_object_contact_force_n: 40.0,
                max_human_contact_force_n: 10.0,
            },
        )
    }

    fn assessment(morphology: HumanoidMorphology) -> HumanoidActuationControllabilityAssessment {
        HumanoidActuationControllabilityAssessment {
            morphology,
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
                    limiting_joint: Some(8),
                }],
                minimum_retained_fraction: 0.9,
            }],
        }
    }

    #[test]
    fn typed_site_serializes_to_existing_canonical_wire_id() {
        let wire = typed_policy().to_wire_policy();
        assert_eq!(wire.requirements[0].site_id, "r_foot_site");
    }

    #[test]
    fn matching_assessment_is_delegated_to_underlying_policy() {
        let decision = assess_typed_humanoid_actuation_capability(
            &subject(),
            &profile(),
            &typed_policy(),
            &assessment(HumanoidMorphology::Dmc21),
        );
        assert_eq!(decision.action, HumanoidActuationCapabilityAction::Continue);
        assert_eq!(decision.guard_failure, None);
        assert!(decision.underlying.is_some());
    }

    #[test]
    fn assessment_morphology_mismatch_fails_closed_before_policy_evaluation() {
        let decision = assess_typed_humanoid_actuation_capability(
            &subject(),
            &profile(),
            &typed_policy(),
            &assessment(HumanoidMorphology::Dexterous53),
        );
        assert_eq!(decision.action, HumanoidActuationCapabilityAction::Hold);
        assert_eq!(
            decision.guard_failure,
            Some(TypedHumanoidActuationCapabilityFailure::AssessmentMorphologyMismatch)
        );
        assert!(decision.underlying.is_none());

        let mut envelope = derive_humanoid_capability_envelope(
            &subject(),
            &profile(),
            HumanoidAuthorityEnvelope::fully_admitted(),
            HumanInteractionEvidence::no_human_present(),
        );
        decision.apply_to_envelope(&mut envelope);
        assert!(!envelope.goal_execution_allowed);
        assert_eq!(envelope.disposition, HumanoidCapabilityDisposition::ProtectiveOnly);
    }

    #[test]
    fn invalid_typed_policy_fails_closed() {
        let mut policy = typed_policy();
        policy.requirements.clear();
        let decision = assess_typed_humanoid_actuation_capability(
            &subject(),
            &profile(),
            &policy,
            &assessment(HumanoidMorphology::Dmc21),
        );
        assert_eq!(decision.action, HumanoidActuationCapabilityAction::Hold);
        assert_eq!(
            decision.guard_failure,
            Some(TypedHumanoidActuationCapabilityFailure::InvalidTypedPolicy)
        );
    }
}
