// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Payload- and center-of-mass-derived runtime capability restrictions.
//!
//! A humanoid can become less capable simply by picking something up. This
//! module makes that change explicit without translating an abstract confidence
//! scalar into fabricated physics. Payload mass, load retention, COM offset, and
//! evidence age are evaluated against subject-bound qualification tiers.

use serde::{Deserialize, Serialize};

use crate::capability_envelope::{
    HumanoidCapabilityDisposition, HumanoidCapabilityEnvelope, HumanoidCapabilityRestriction,
    HumanoidNominalCapabilityProfile,
};
use crate::qualification::HumanoidQualificationSubject;

pub const HUMANOID_PAYLOAD_CAPABILITY_POLICY_SCHEMA_VERSION: u32 = 1;

/// Runtime load evidence. A zero-mass frame represents no carried payload.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HumanoidPayloadCapabilityEvidence {
    pub evidence_valid: bool,
    pub payload_mass_kg: f64,
    /// Payload COM relative to the qualified body reference frame.
    pub payload_com_offset_body_m: [f64; 3],
    /// Whether the currently carried load is known to be mechanically retained
    /// by the gripper/tool/fixture. Ignored only when payload mass is zero.
    pub load_retention_valid: bool,
    pub evidence_age_s: f64,
}

impl HumanoidPayloadCapabilityEvidence {
    pub fn validate(self) -> bool {
        self.evidence_valid
            && self.payload_mass_kg.is_finite()
            && self.payload_mass_kg >= 0.0
            && self
                .payload_com_offset_body_m
                .iter()
                .all(|value| value.is_finite())
            && self.evidence_age_s.is_finite()
            && self.evidence_age_s >= 0.0
    }

    pub fn com_offset_norm_m(self) -> f64 {
        self.payload_com_offset_body_m
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt()
    }

    pub fn retention_satisfied(self) -> bool {
        self.payload_mass_kg <= 1.0e-9 || self.load_retention_valid
    }
}

/// Subject-bound payload policy. There is intentionally no `Default` because
/// these load limits and degraded ceilings must come from qualification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidPayloadCapabilityPolicy {
    pub schema_version: u32,
    pub subject_fingerprint: u64,
    pub maximum_nominal_payload_kg: f64,
    pub maximum_nominal_com_offset_m: f64,
    pub maximum_nominal_evidence_age_s: f64,
    pub maximum_degraded_payload_kg: f64,
    pub maximum_degraded_com_offset_m: f64,
    pub maximum_degraded_evidence_age_s: f64,
    pub degraded_max_horizontal_speed_mps: f64,
    pub degraded_max_turn_rate_rad_s: f64,
    pub degraded_max_end_effector_speed_mps: f64,
}

impl HumanoidPayloadCapabilityPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &HumanoidQualificationSubject,
        maximum_nominal_payload_kg: f64,
        maximum_nominal_com_offset_m: f64,
        maximum_nominal_evidence_age_s: f64,
        maximum_degraded_payload_kg: f64,
        maximum_degraded_com_offset_m: f64,
        maximum_degraded_evidence_age_s: f64,
        degraded_max_horizontal_speed_mps: f64,
        degraded_max_turn_rate_rad_s: f64,
        degraded_max_end_effector_speed_mps: f64,
    ) -> Self {
        Self {
            schema_version: HUMANOID_PAYLOAD_CAPABILITY_POLICY_SCHEMA_VERSION,
            subject_fingerprint: subject.fingerprint(),
            maximum_nominal_payload_kg,
            maximum_nominal_com_offset_m,
            maximum_nominal_evidence_age_s,
            maximum_degraded_payload_kg,
            maximum_degraded_com_offset_m,
            maximum_degraded_evidence_age_s,
            degraded_max_horizontal_speed_mps,
            degraded_max_turn_rate_rad_s,
            degraded_max_end_effector_speed_mps,
        }
    }

    pub fn validate_for(
        &self,
        subject: &HumanoidQualificationSubject,
        profile: &HumanoidNominalCapabilityProfile,
    ) -> bool {
        if self.schema_version != HUMANOID_PAYLOAD_CAPABILITY_POLICY_SCHEMA_VERSION
            || self.subject_fingerprint == 0
            || self.subject_fingerprint != subject.fingerprint()
            || !profile.validate_for(subject)
        {
            return false;
        }

        let values = [
            self.maximum_nominal_payload_kg,
            self.maximum_nominal_com_offset_m,
            self.maximum_nominal_evidence_age_s,
            self.maximum_degraded_payload_kg,
            self.maximum_degraded_com_offset_m,
            self.maximum_degraded_evidence_age_s,
            self.degraded_max_horizontal_speed_mps,
            self.degraded_max_turn_rate_rad_s,
            self.degraded_max_end_effector_speed_mps,
        ];
        values.iter().all(|value| value.is_finite() && *value >= 0.0)
            && self.maximum_nominal_payload_kg <= self.maximum_degraded_payload_kg
            && self.maximum_degraded_payload_kg <= profile.max_payload_kg
            && self.maximum_nominal_com_offset_m <= self.maximum_degraded_com_offset_m
            && self.maximum_nominal_evidence_age_s <= self.maximum_degraded_evidence_age_s
            && self.degraded_max_horizontal_speed_mps <= profile.max_horizontal_speed_mps
            && self.degraded_max_turn_rate_rad_s <= profile.max_turn_rate_rad_s
            && self.degraded_max_end_effector_speed_mps <= profile.max_end_effector_speed_mps
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidPayloadCapabilityAction {
    Continue,
    Reduce,
    Hold,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HumanoidPayloadCapabilityAssessment {
    pub action: HumanoidPayloadCapabilityAction,
    pub evidence: HumanoidPayloadCapabilityEvidence,
    /// Remaining *additional* payload capacity under the qualified profile.
    pub remaining_payload_capacity_kg: f64,
    pub restriction: HumanoidCapabilityRestriction,
}

impl HumanoidPayloadCapabilityAssessment {
    pub fn apply_to(self, envelope: &mut HumanoidCapabilityEnvelope) {
        envelope.apply_restriction(self.restriction);
        if self.action == HumanoidPayloadCapabilityAction::Hold {
            envelope.goal_execution_allowed = false;
            envelope.human_contact_allowed = false;
            envelope.disposition = HumanoidCapabilityDisposition::ProtectiveOnly;
        }
    }
}

pub fn assess_humanoid_payload_capability(
    subject: &HumanoidQualificationSubject,
    profile: &HumanoidNominalCapabilityProfile,
    policy: &HumanoidPayloadCapabilityPolicy,
    evidence: HumanoidPayloadCapabilityEvidence,
    current: &HumanoidCapabilityEnvelope,
) -> HumanoidPayloadCapabilityAssessment {
    let preserve = HumanoidCapabilityRestriction {
        max_horizontal_speed_mps: current.limits.max_horizontal_speed_mps,
        max_turn_rate_rad_s: current.limits.max_turn_rate_rad_s,
        max_end_effector_speed_mps: current.limits.max_end_effector_speed_mps,
        max_payload_kg: current.limits.max_payload_kg,
        max_object_contact_force_n: current.limits.max_object_contact_force_n,
        max_human_contact_force_n: current.limits.max_human_contact_force_n,
    };

    let valid_binding = current.subject_fingerprint != 0
        && current.subject_fingerprint == subject.fingerprint()
        && policy.validate_for(subject, profile);
    let remaining_payload_capacity_kg = if evidence.payload_mass_kg.is_finite() {
        (profile.max_payload_kg - evidence.payload_mass_kg).max(0.0)
    } else {
        0.0
    };

    if !valid_binding || !evidence.validate() || !evidence.retention_satisfied() {
        return HumanoidPayloadCapabilityAssessment {
            action: HumanoidPayloadCapabilityAction::Hold,
            evidence,
            remaining_payload_capacity_kg,
            restriction: HumanoidCapabilityRestriction {
                max_horizontal_speed_mps: 0.0,
                max_turn_rate_rad_s: 0.0,
                max_end_effector_speed_mps: 0.0,
                ..preserve
            },
        };
    }

    let com_offset = evidence.com_offset_norm_m();
    if evidence.payload_mass_kg > profile.max_payload_kg {
        return HumanoidPayloadCapabilityAssessment {
            action: HumanoidPayloadCapabilityAction::Hold,
            evidence,
            remaining_payload_capacity_kg: 0.0,
            restriction: HumanoidCapabilityRestriction {
                max_horizontal_speed_mps: 0.0,
                max_turn_rate_rad_s: 0.0,
                max_end_effector_speed_mps: 0.0,
                max_payload_kg: profile.max_payload_kg,
                ..preserve
            },
        };
    }

    let nominal = evidence.payload_mass_kg <= policy.maximum_nominal_payload_kg
        && com_offset <= policy.maximum_nominal_com_offset_m
        && evidence.evidence_age_s <= policy.maximum_nominal_evidence_age_s;
    if nominal {
        return HumanoidPayloadCapabilityAssessment {
            action: HumanoidPayloadCapabilityAction::Continue,
            evidence,
            remaining_payload_capacity_kg,
            restriction: HumanoidCapabilityRestriction {
                max_payload_kg: current.limits.max_payload_kg.min(profile.max_payload_kg),
                ..preserve
            },
        };
    }

    let degraded = evidence.payload_mass_kg <= policy.maximum_degraded_payload_kg
        && com_offset <= policy.maximum_degraded_com_offset_m
        && evidence.evidence_age_s <= policy.maximum_degraded_evidence_age_s;
    if degraded {
        return HumanoidPayloadCapabilityAssessment {
            action: HumanoidPayloadCapabilityAction::Reduce,
            evidence,
            remaining_payload_capacity_kg,
            restriction: HumanoidCapabilityRestriction {
                max_horizontal_speed_mps: current
                    .limits
                    .max_horizontal_speed_mps
                    .min(policy.degraded_max_horizontal_speed_mps),
                max_turn_rate_rad_s: current
                    .limits
                    .max_turn_rate_rad_s
                    .min(policy.degraded_max_turn_rate_rad_s),
                max_end_effector_speed_mps: current
                    .limits
                    .max_end_effector_speed_mps
                    .min(policy.degraded_max_end_effector_speed_mps),
                max_payload_kg: current.limits.max_payload_kg.min(profile.max_payload_kg),
                ..preserve
            },
        };
    }

    HumanoidPayloadCapabilityAssessment {
        action: HumanoidPayloadCapabilityAction::Hold,
        evidence,
        remaining_payload_capacity_kg,
        restriction: HumanoidCapabilityRestriction {
            max_horizontal_speed_mps: 0.0,
            max_turn_rate_rad_s: 0.0,
            max_end_effector_speed_mps: 0.0,
            max_payload_kg: current.limits.max_payload_kg.min(profile.max_payload_kg),
            ..preserve
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability_envelope::{
        HumanInteractionEvidence, derive_humanoid_capability_envelope,
    };
    use crate::execution::HumanoidAuthorityEnvelope;
    use crate::morphology::HumanoidMorphology;
    use crate::types::{ActuationMode, HumanoidTask};

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dmc21,
            HumanoidTask::Carry,
            ActuationMode::NormalizedTorque,
            "test-backend-v1",
        )
    }

    fn profile(subject: &HumanoidQualificationSubject) -> HumanoidNominalCapabilityProfile {
        // Test-only limits; not production safety values.
        HumanoidNominalCapabilityProfile::new(
            subject, 2.0, 0.4, 1.0, 1.2, 0.2, 10.0, 80.0, 0.0, 0.8, false,
        )
    }

    fn policy(subject: &HumanoidQualificationSubject) -> HumanoidPayloadCapabilityPolicy {
        // Test-only tier boundaries and caps.
        HumanoidPayloadCapabilityPolicy::new(
            subject, 3.0, 0.10, 0.10, 8.0, 0.30, 0.50, 0.6, 0.3, 0.4,
        )
    }

    fn envelope(
        subject: &HumanoidQualificationSubject,
        profile: &HumanoidNominalCapabilityProfile,
    ) -> HumanoidCapabilityEnvelope {
        derive_humanoid_capability_envelope(
            subject,
            profile,
            HumanoidAuthorityEnvelope::fully_admitted(),
            HumanInteractionEvidence::no_human_present(),
        )
    }

    fn evidence(mass: f64, offset_x: f64, age_s: f64) -> HumanoidPayloadCapabilityEvidence {
        HumanoidPayloadCapabilityEvidence {
            evidence_valid: true,
            payload_mass_kg: mass,
            payload_com_offset_body_m: [offset_x, 0.0, 0.0],
            load_retention_valid: true,
            evidence_age_s: age_s,
        }
    }

    #[test]
    fn nominal_payload_preserves_motion_limits_and_reports_remaining_capacity() {
        let subject = subject();
        let profile = profile(&subject);
        let current = envelope(&subject, &profile);
        let assessment = assess_humanoid_payload_capability(
            &subject,
            &profile,
            &policy(&subject),
            evidence(2.0, 0.05, 0.05),
            &current,
        );
        assert_eq!(assessment.action, HumanoidPayloadCapabilityAction::Continue);
        assert_eq!(assessment.remaining_payload_capacity_kg, 8.0);
        assert_eq!(assessment.restriction.max_horizontal_speed_mps, 2.0);
    }

    #[test]
    fn degraded_payload_uses_explicit_qualified_caps() {
        let subject = subject();
        let profile = profile(&subject);
        let mut current = envelope(&subject, &profile);
        let assessment = assess_humanoid_payload_capability(
            &subject,
            &profile,
            &policy(&subject),
            evidence(6.0, 0.20, 0.20),
            &current,
        );
        assert_eq!(assessment.action, HumanoidPayloadCapabilityAction::Reduce);
        assessment.apply_to(&mut current);
        assert_eq!(current.limits.max_horizontal_speed_mps, 0.6);
        assert_eq!(current.limits.max_turn_rate_rad_s, 0.3);
        assert_eq!(current.limits.max_end_effector_speed_mps, 0.4);
        assert!(current.goal_execution_allowed);
    }

    #[test]
    fn excess_payload_requires_hold() {
        let subject = subject();
        let profile = profile(&subject);
        let mut current = envelope(&subject, &profile);
        let assessment = assess_humanoid_payload_capability(
            &subject,
            &profile,
            &policy(&subject),
            evidence(11.0, 0.05, 0.05),
            &current,
        );
        assert_eq!(assessment.remaining_payload_capacity_kg, 0.0);
        assessment.apply_to(&mut current);
        assert!(!current.goal_execution_allowed);
    }

    #[test]
    fn stale_or_large_com_shift_requires_hold() {
        let subject = subject();
        let profile = profile(&subject);
        let mut current = envelope(&subject, &profile);
        let assessment = assess_humanoid_payload_capability(
            &subject,
            &profile,
            &policy(&subject),
            evidence(4.0, 0.50, 1.0),
            &current,
        );
        assert_eq!(assessment.action, HumanoidPayloadCapabilityAction::Hold);
        assessment.apply_to(&mut current);
        assert!(!current.goal_execution_allowed);
    }

    #[test]
    fn unretained_nonzero_load_fails_closed() {
        let subject = subject();
        let profile = profile(&subject);
        let current = envelope(&subject, &profile);
        let mut sample = evidence(2.0, 0.05, 0.05);
        sample.load_retention_valid = false;
        let assessment = assess_humanoid_payload_capability(
            &subject,
            &profile,
            &policy(&subject),
            sample,
            &current,
        );
        assert_eq!(assessment.action, HumanoidPayloadCapabilityAction::Hold);
    }
}
