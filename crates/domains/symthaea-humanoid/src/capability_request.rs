// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pre-motor admission for semantic humanoid capability requests.
//!
//! Skills and planners should not discover capability violations only after
//! generating joint commands. This module defines a transport-neutral request IR
//! that can be checked against the composed runtime capability envelope before a
//! request reaches whole-body synthesis or the prepared-command boundary.
//!
//! Admission is reject-only: requested semantics are never silently clamped into
//! a different action. A planner may explicitly replan and submit a new request.

use serde::{Deserialize, Serialize};

use crate::capability_envelope::HumanoidCapabilityEnvelope;

/// Semantic physical demands compiled from a higher-level humanoid skill.
///
/// `resulting_total_payload_kg` is the total carried payload after the requested
/// action, not an additional payload increment.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HumanoidCapabilityRequest {
    pub subject_fingerprint: u64,
    pub horizontal_speed_mps: f64,
    pub turn_rate_rad_s: f64,
    pub end_effector_speed_mps: f64,
    pub resulting_total_payload_kg: f64,
    pub object_contact_force_n: f64,
    pub intentional_human_contact: bool,
    pub human_contact_force_n: f64,
}

impl HumanoidCapabilityRequest {
    pub const fn stationary(subject_fingerprint: u64) -> Self {
        Self {
            subject_fingerprint,
            horizontal_speed_mps: 0.0,
            turn_rate_rad_s: 0.0,
            end_effector_speed_mps: 0.0,
            resulting_total_payload_kg: 0.0,
            object_contact_force_n: 0.0,
            intentional_human_contact: false,
            human_contact_force_n: 0.0,
        }
    }

    pub fn validate(self) -> bool {
        self.subject_fingerprint != 0
            && [
                self.horizontal_speed_mps,
                self.turn_rate_rad_s,
                self.end_effector_speed_mps,
                self.resulting_total_payload_kg,
                self.object_contact_force_n,
                self.human_contact_force_n,
            ]
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
            && (self.intentional_human_contact || self.human_contact_force_n == 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidCapabilityViolationKind {
    InvalidRequest,
    SubjectMismatch,
    GoalExecutionRevoked,
    HorizontalSpeedExceeded,
    TurnRateExceeded,
    EndEffectorSpeedExceeded,
    PayloadExceeded,
    ObjectContactForceExceeded,
    HumanContactNotAllowed,
    HumanContactForceExceeded,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HumanoidCapabilityViolation {
    pub kind: HumanoidCapabilityViolationKind,
    pub requested: f64,
    pub admitted: f64,
}

impl HumanoidCapabilityViolation {
    const fn structural(kind: HumanoidCapabilityViolationKind) -> Self {
        Self {
            kind,
            requested: 0.0,
            admitted: 0.0,
        }
    }

    const fn limit(kind: HumanoidCapabilityViolationKind, requested: f64, admitted: f64) -> Self {
        Self {
            kind,
            requested,
            admitted,
        }
    }
}

/// Auditable pre-motor admission result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidCapabilityAdmission {
    pub subject_fingerprint: u64,
    pub admitted: bool,
    pub request: HumanoidCapabilityRequest,
    pub violations: Vec<HumanoidCapabilityViolation>,
}

/// Check one semantic request against the current runtime capability envelope.
///
/// No request field is modified. Rejected callers must explicitly replan.
pub fn admit_humanoid_capability_request(
    envelope: &HumanoidCapabilityEnvelope,
    request: HumanoidCapabilityRequest,
) -> HumanoidCapabilityAdmission {
    let mut violations = Vec::new();

    if !request.validate() {
        violations.push(HumanoidCapabilityViolation::structural(
            HumanoidCapabilityViolationKind::InvalidRequest,
        ));
    }
    if envelope.subject_fingerprint == 0
        || request.subject_fingerprint != envelope.subject_fingerprint
    {
        violations.push(HumanoidCapabilityViolation::structural(
            HumanoidCapabilityViolationKind::SubjectMismatch,
        ));
    }
    if !envelope.goal_execution_allowed {
        violations.push(HumanoidCapabilityViolation::structural(
            HumanoidCapabilityViolationKind::GoalExecutionRevoked,
        ));
    }

    check_limit(
        &mut violations,
        HumanoidCapabilityViolationKind::HorizontalSpeedExceeded,
        request.horizontal_speed_mps,
        envelope.limits.max_horizontal_speed_mps,
    );
    check_limit(
        &mut violations,
        HumanoidCapabilityViolationKind::TurnRateExceeded,
        request.turn_rate_rad_s,
        envelope.limits.max_turn_rate_rad_s,
    );
    check_limit(
        &mut violations,
        HumanoidCapabilityViolationKind::EndEffectorSpeedExceeded,
        request.end_effector_speed_mps,
        envelope.limits.max_end_effector_speed_mps,
    );
    check_limit(
        &mut violations,
        HumanoidCapabilityViolationKind::PayloadExceeded,
        request.resulting_total_payload_kg,
        envelope.limits.max_payload_kg,
    );
    check_limit(
        &mut violations,
        HumanoidCapabilityViolationKind::ObjectContactForceExceeded,
        request.object_contact_force_n,
        envelope.limits.max_object_contact_force_n,
    );

    if request.intentional_human_contact {
        if !envelope.human_contact_allowed {
            violations.push(HumanoidCapabilityViolation::structural(
                HumanoidCapabilityViolationKind::HumanContactNotAllowed,
            ));
        }
        check_limit(
            &mut violations,
            HumanoidCapabilityViolationKind::HumanContactForceExceeded,
            request.human_contact_force_n,
            envelope.limits.max_human_contact_force_n,
        );
    }

    HumanoidCapabilityAdmission {
        subject_fingerprint: envelope.subject_fingerprint,
        admitted: violations.is_empty(),
        request,
        violations,
    }
}

fn check_limit(
    violations: &mut Vec<HumanoidCapabilityViolation>,
    kind: HumanoidCapabilityViolationKind,
    requested: f64,
    admitted: f64,
) {
    if !requested.is_finite() || !admitted.is_finite() || requested > admitted {
        violations.push(HumanoidCapabilityViolation::limit(kind, requested, admitted));
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
    use crate::morphology::HumanoidMorphology;
    use crate::qualification::HumanoidQualificationSubject;
    use crate::types::{ActuationMode, HumanoidTask};

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dmc21,
            HumanoidTask::Grasp,
            ActuationMode::NormalizedTorque,
            "test-backend-v1",
        )
    }

    fn profile(subject: &HumanoidQualificationSubject) -> HumanoidNominalCapabilityProfile {
        // Test-only values, not production safety limits.
        HumanoidNominalCapabilityProfile::new(
            subject, 2.0, 0.5, 1.0, 1.2, 0.3, 10.0, 80.0, 20.0, 0.8, true,
        )
    }

    fn envelope_with_human(contact_consent: bool) -> HumanoidCapabilityEnvelope {
        let subject = subject();
        derive_humanoid_capability_envelope(
            &subject,
            &profile(&subject),
            HumanoidAuthorityEnvelope::fully_admitted(),
            HumanInteractionEvidence {
                human_present: true,
                proximity_valid: true,
                nearest_human_distance_m: 1.0,
                contact_consent,
            },
        )
    }

    #[test]
    fn request_inside_all_limits_is_admitted() {
        let envelope = envelope_with_human(true);
        let request = HumanoidCapabilityRequest {
            subject_fingerprint: envelope.subject_fingerprint,
            horizontal_speed_mps: 0.4,
            turn_rate_rad_s: 0.5,
            end_effector_speed_mps: 0.3,
            resulting_total_payload_kg: 4.0,
            object_contact_force_n: 30.0,
            intentional_human_contact: true,
            human_contact_force_n: 10.0,
        };
        let admission = admit_humanoid_capability_request(&envelope, request);
        assert!(admission.admitted, "{:?}", admission.violations);
    }

    #[test]
    fn request_is_rejected_not_clamped_when_limit_is_exceeded() {
        let envelope = envelope_with_human(true);
        let mut request = HumanoidCapabilityRequest::stationary(envelope.subject_fingerprint);
        request.horizontal_speed_mps = envelope.limits.max_horizontal_speed_mps + 0.1;
        let admission = admit_humanoid_capability_request(&envelope, request);
        assert!(!admission.admitted);
        assert_eq!(admission.request.horizontal_speed_mps, request.horizontal_speed_mps);
        assert!(admission.violations.iter().any(|violation| {
            violation.kind == HumanoidCapabilityViolationKind::HorizontalSpeedExceeded
        }));
    }

    #[test]
    fn revoked_goal_execution_rejects_even_zero_motion_request() {
        let subject = subject();
        let envelope = derive_humanoid_capability_envelope(
            &subject,
            &profile(&subject),
            HumanoidAuthorityEnvelope::default(),
            HumanInteractionEvidence::no_human_present(),
        );
        let request = HumanoidCapabilityRequest::stationary(envelope.subject_fingerprint);
        let admission = admit_humanoid_capability_request(&envelope, request);
        assert!(!admission.admitted);
        assert!(admission.violations.iter().any(|violation| {
            violation.kind == HumanoidCapabilityViolationKind::GoalExecutionRevoked
        }));
    }

    #[test]
    fn operator_authority_does_not_bypass_human_contact_gate() {
        let envelope = envelope_with_human(false);
        let mut request = HumanoidCapabilityRequest::stationary(envelope.subject_fingerprint);
        request.intentional_human_contact = true;
        request.human_contact_force_n = 1.0;
        let admission = admit_humanoid_capability_request(&envelope, request);
        assert!(!admission.admitted);
        assert!(admission.violations.iter().any(|violation| {
            violation.kind == HumanoidCapabilityViolationKind::HumanContactNotAllowed
        }));
    }

    #[test]
    fn subject_mismatch_rejects_request() {
        let envelope = envelope_with_human(true);
        let request = HumanoidCapabilityRequest::stationary(envelope.subject_fingerprint + 1);
        let admission = admit_humanoid_capability_request(&envelope, request);
        assert!(!admission.admitted);
        assert!(admission.violations.iter().any(|violation| {
            violation.kind == HumanoidCapabilityViolationKind::SubjectMismatch
        }));
    }

    #[test]
    fn non_finite_request_fails_closed() {
        let envelope = envelope_with_human(true);
        let mut request = HumanoidCapabilityRequest::stationary(envelope.subject_fingerprint);
        request.end_effector_speed_mps = f64::NAN;
        let admission = admit_humanoid_capability_request(&envelope, request);
        assert!(!admission.admitted);
        assert!(admission.violations.iter().any(|violation| {
            violation.kind == HumanoidCapabilityViolationKind::InvalidRequest
        }));
    }

    #[test]
    fn exact_limits_are_admitted() {
        let envelope = envelope_with_human(true);
        let request = HumanoidCapabilityRequest {
            subject_fingerprint: envelope.subject_fingerprint,
            horizontal_speed_mps: envelope.limits.max_horizontal_speed_mps,
            turn_rate_rad_s: envelope.limits.max_turn_rate_rad_s,
            end_effector_speed_mps: envelope.limits.max_end_effector_speed_mps,
            resulting_total_payload_kg: envelope.limits.max_payload_kg,
            object_contact_force_n: envelope.limits.max_object_contact_force_n,
            intentional_human_contact: true,
            human_contact_force_n: envelope.limits.max_human_contact_force_n,
        };
        assert!(admit_humanoid_capability_request(&envelope, request).admitted);
    }
}
