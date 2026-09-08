// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Subject-bound runtime capability envelopes for humanoid execution.
//!
//! Authority answers whether goal-directed execution is admitted. A capability
//! envelope answers a different question: what upper bounds remain applicable
//! to this exact qualified body/task/backend right now?
//!
//! The module deliberately does not invent universal safe speeds, payloads, or
//! contact forces. Nominal limits come from a subject-bound qualified profile;
//! runtime evidence may only preserve or tighten those limits. The resulting
//! numbers are upper bounds, not guarantees that a requested maneuver is
//! dynamically feasible.

use serde::{Deserialize, Serialize};

use crate::execution::HumanoidAuthorityEnvelope;
use crate::morphology::HumanoidMorphology;
use crate::qualification::{
    HUMANOID_QUALIFICATION_SUBJECT_SCHEMA_VERSION, HumanoidQualificationSubject,
};
use crate::types::{ActuationMode, HumanoidTask};

pub const HUMANOID_CAPABILITY_PROFILE_SCHEMA_VERSION: u32 = 1;

/// Qualified upper bounds for one exact humanoid execution subject.
///
/// These values must come from engineering/qualification evidence for the
/// identified body/backend. `HumanoidNominalCapabilityProfile` intentionally has
/// no `Default` implementation so callers cannot accidentally acquire invented
/// safety limits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidNominalCapabilityProfile {
    pub schema_version: u32,
    pub subject_fingerprint: u64,
    pub morphology: HumanoidMorphology,
    pub task: HumanoidTask,
    pub actuation_mode: ActuationMode,
    pub max_horizontal_speed_mps: f64,
    pub max_horizontal_speed_near_human_mps: f64,
    pub max_turn_rate_rad_s: f64,
    pub max_end_effector_speed_mps: f64,
    pub max_end_effector_speed_near_human_mps: f64,
    pub max_payload_kg: f64,
    pub max_object_contact_force_n: f64,
    pub max_human_contact_force_n: f64,
    pub minimum_unconsented_human_separation_m: f64,
    pub human_contact_permitted: bool,
}

impl HumanoidNominalCapabilityProfile {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &HumanoidQualificationSubject,
        max_horizontal_speed_mps: f64,
        max_horizontal_speed_near_human_mps: f64,
        max_turn_rate_rad_s: f64,
        max_end_effector_speed_mps: f64,
        max_end_effector_speed_near_human_mps: f64,
        max_payload_kg: f64,
        max_object_contact_force_n: f64,
        max_human_contact_force_n: f64,
        minimum_unconsented_human_separation_m: f64,
        human_contact_permitted: bool,
    ) -> Self {
        Self {
            schema_version: HUMANOID_CAPABILITY_PROFILE_SCHEMA_VERSION,
            subject_fingerprint: subject.fingerprint(),
            morphology: subject.morphology,
            task: subject.task,
            actuation_mode: subject.actuation_mode,
            max_horizontal_speed_mps,
            max_horizontal_speed_near_human_mps,
            max_turn_rate_rad_s,
            max_end_effector_speed_mps,
            max_end_effector_speed_near_human_mps,
            max_payload_kg,
            max_object_contact_force_n,
            max_human_contact_force_n,
            minimum_unconsented_human_separation_m,
            human_contact_permitted,
        }
    }

    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        if self.schema_version != HUMANOID_CAPABILITY_PROFILE_SCHEMA_VERSION
            || subject.schema_version != HUMANOID_QUALIFICATION_SUBJECT_SCHEMA_VERSION
            || !subject.validate()
            || self.subject_fingerprint == 0
            || self.subject_fingerprint != subject.fingerprint()
            || self.morphology != subject.morphology
            || self.task != subject.task
            || self.actuation_mode != subject.actuation_mode
        {
            return false;
        }

        let non_negative = [
            self.max_horizontal_speed_mps,
            self.max_horizontal_speed_near_human_mps,
            self.max_turn_rate_rad_s,
            self.max_end_effector_speed_mps,
            self.max_end_effector_speed_near_human_mps,
            self.max_payload_kg,
            self.max_object_contact_force_n,
            self.max_human_contact_force_n,
            self.minimum_unconsented_human_separation_m,
        ];
        non_negative.iter().all(|value| value.is_finite() && *value >= 0.0)
            && self.max_horizontal_speed_near_human_mps <= self.max_horizontal_speed_mps
            && self.max_end_effector_speed_near_human_mps <= self.max_end_effector_speed_mps
            && (self.human_contact_permitted || self.max_human_contact_force_n == 0.0)
    }

    fn limits(&self) -> HumanoidCapabilityLimits {
        HumanoidCapabilityLimits {
            max_horizontal_speed_mps: self.max_horizontal_speed_mps,
            max_turn_rate_rad_s: self.max_turn_rate_rad_s,
            max_end_effector_speed_mps: self.max_end_effector_speed_mps,
            max_payload_kg: self.max_payload_kg,
            max_object_contact_force_n: self.max_object_contact_force_n,
            max_human_contact_force_n: if self.human_contact_permitted {
                self.max_human_contact_force_n
            } else {
                0.0
            },
        }
    }
}

/// Explicit human-presence evidence. Human contact consent is separate from
/// operator/task authority and is never inferred from it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HumanInteractionEvidence {
    pub human_present: bool,
    /// Whether nearest-human separation is backed by valid/fresh perception.
    pub proximity_valid: bool,
    pub nearest_human_distance_m: f64,
    /// Explicit consent for intentional physical contact in the current task.
    pub contact_consent: bool,
}

impl HumanInteractionEvidence {
    pub const fn no_human_present() -> Self {
        Self {
            human_present: false,
            proximity_valid: true,
            nearest_human_distance_m: f64::INFINITY,
            contact_consent: false,
        }
    }

    fn valid_distance(self) -> bool {
        self.nearest_human_distance_m.is_finite() && self.nearest_human_distance_m >= 0.0
    }
}

/// Concrete upper bounds remaining after runtime restrictions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HumanoidCapabilityLimits {
    pub max_horizontal_speed_mps: f64,
    pub max_turn_rate_rad_s: f64,
    pub max_end_effector_speed_mps: f64,
    pub max_payload_kg: f64,
    pub max_object_contact_force_n: f64,
    pub max_human_contact_force_n: f64,
}

impl HumanoidCapabilityLimits {
    pub const fn zero() -> Self {
        Self {
            max_horizontal_speed_mps: 0.0,
            max_turn_rate_rad_s: 0.0,
            max_end_effector_speed_mps: 0.0,
            max_payload_kg: 0.0,
            max_object_contact_force_n: 0.0,
            max_human_contact_force_n: 0.0,
        }
    }

    fn tighten(&mut self, restriction: HumanoidCapabilityRestriction) {
        self.max_horizontal_speed_mps = self
            .max_horizontal_speed_mps
            .min(restriction.max_horizontal_speed_mps);
        self.max_turn_rate_rad_s = self.max_turn_rate_rad_s.min(restriction.max_turn_rate_rad_s);
        self.max_end_effector_speed_mps = self
            .max_end_effector_speed_mps
            .min(restriction.max_end_effector_speed_mps);
        self.max_payload_kg = self.max_payload_kg.min(restriction.max_payload_kg);
        self.max_object_contact_force_n = self
            .max_object_contact_force_n
            .min(restriction.max_object_contact_force_n);
        self.max_human_contact_force_n = self
            .max_human_contact_force_n
            .min(restriction.max_human_contact_force_n);
    }
}

/// Explicit additional bounds derived by terrain, payload, controllability,
/// thermal, workspace, or other evidence modules.
///
/// Constructing a restriction requires concrete upper bounds. This module does
/// not map an abstract confidence scalar into a fabricated speed/force limit.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HumanoidCapabilityRestriction {
    pub max_horizontal_speed_mps: f64,
    pub max_turn_rate_rad_s: f64,
    pub max_end_effector_speed_mps: f64,
    pub max_payload_kg: f64,
    pub max_object_contact_force_n: f64,
    pub max_human_contact_force_n: f64,
}

impl HumanoidCapabilityRestriction {
    pub fn validate(self) -> bool {
        [
            self.max_horizontal_speed_mps,
            self.max_turn_rate_rad_s,
            self.max_end_effector_speed_mps,
            self.max_payload_kg,
            self.max_object_contact_force_n,
            self.max_human_contact_force_n,
        ]
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidCapabilityDisposition {
    /// Full typed authority and no additional hold condition.
    Nominal,
    /// Goal execution remains possible but one or more authority sources restrict it.
    ReducedAuthority,
    /// Goal-directed execution is revoked; only independently admitted protective behavior remains.
    ProtectiveOnly,
    /// Human-proximity evidence requires goal motion to hold even though protective behavior remains.
    HumanProximityHold,
    /// The capability profile is not bound to the requested qualification subject.
    SubjectMismatch,
}

/// Auditable runtime capability envelope. Numeric limits are upper bounds only;
/// they do not claim maneuver feasibility or calibrated probability of success.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidCapabilityEnvelope {
    pub subject_fingerprint: u64,
    pub disposition: HumanoidCapabilityDisposition,
    pub goal_authority_scale: f32,
    pub goal_execution_allowed: bool,
    pub human_contact_allowed: bool,
    pub limits: HumanoidCapabilityLimits,
    pub restrictions_applied: usize,
}

impl HumanoidCapabilityEnvelope {
    /// Tighten the envelope with one concrete restriction. Invalid restrictions
    /// fail closed to protective-only rather than being ignored.
    pub fn apply_restriction(&mut self, restriction: HumanoidCapabilityRestriction) {
        if !restriction.validate() {
            self.goal_execution_allowed = false;
            self.human_contact_allowed = false;
            self.limits = HumanoidCapabilityLimits::zero();
            self.disposition = HumanoidCapabilityDisposition::ProtectiveOnly;
            return;
        }
        self.limits.tighten(restriction);
        self.restrictions_applied = self.restrictions_applied.saturating_add(1);
        if self.limits.max_human_contact_force_n <= 0.0 {
            self.human_contact_allowed = false;
        }
    }
}

/// Derive the runtime envelope without fabricating physics from confidence.
pub fn derive_humanoid_capability_envelope(
    subject: &HumanoidQualificationSubject,
    profile: &HumanoidNominalCapabilityProfile,
    authority: HumanoidAuthorityEnvelope,
    human: HumanInteractionEvidence,
) -> HumanoidCapabilityEnvelope {
    let subject_fingerprint = subject.fingerprint();
    if subject_fingerprint == 0 || !profile.validate_for(subject) {
        return HumanoidCapabilityEnvelope {
            subject_fingerprint,
            disposition: HumanoidCapabilityDisposition::SubjectMismatch,
            goal_authority_scale: 0.0,
            goal_execution_allowed: false,
            human_contact_allowed: false,
            limits: HumanoidCapabilityLimits::zero(),
            restrictions_applied: 0,
        };
    }

    let authority_scale = authority.effective_scale();
    if authority_scale <= 0.0 {
        return HumanoidCapabilityEnvelope {
            subject_fingerprint,
            disposition: HumanoidCapabilityDisposition::ProtectiveOnly,
            goal_authority_scale: 0.0,
            goal_execution_allowed: false,
            human_contact_allowed: false,
            limits: HumanoidCapabilityLimits::zero(),
            restrictions_applied: 0,
        };
    }

    let mut limits = profile.limits();
    let mut human_contact_allowed = false;
    let mut disposition = if authority_scale < 1.0 {
        HumanoidCapabilityDisposition::ReducedAuthority
    } else {
        HumanoidCapabilityDisposition::Nominal
    };

    if human.human_present {
        // Unknown/invalid separation is not interpreted as "far away".
        if !human.proximity_valid || !human.valid_distance() {
            limits.max_horizontal_speed_mps = 0.0;
            limits.max_turn_rate_rad_s = 0.0;
            limits.max_end_effector_speed_mps = 0.0;
            limits.max_human_contact_force_n = 0.0;
            disposition = HumanoidCapabilityDisposition::HumanProximityHold;
            return HumanoidCapabilityEnvelope {
                subject_fingerprint,
                disposition,
                goal_authority_scale: authority_scale,
                goal_execution_allowed: false,
                human_contact_allowed: false,
                limits,
                restrictions_applied: 1,
            };
        }

        // Presence of a human selects qualified near-human speed ceilings.
        limits.max_horizontal_speed_mps = limits
            .max_horizontal_speed_mps
            .min(profile.max_horizontal_speed_near_human_mps);
        limits.max_end_effector_speed_mps = limits
            .max_end_effector_speed_mps
            .min(profile.max_end_effector_speed_near_human_mps);

        let inside_unconsented_separation = human.nearest_human_distance_m
            < profile.minimum_unconsented_human_separation_m
            && !human.contact_consent;
        if inside_unconsented_separation {
            limits.max_horizontal_speed_mps = 0.0;
            limits.max_turn_rate_rad_s = 0.0;
            limits.max_end_effector_speed_mps = 0.0;
            limits.max_human_contact_force_n = 0.0;
            disposition = HumanoidCapabilityDisposition::HumanProximityHold;
            return HumanoidCapabilityEnvelope {
                subject_fingerprint,
                disposition,
                goal_authority_scale: authority_scale,
                goal_execution_allowed: false,
                human_contact_allowed: false,
                limits,
                restrictions_applied: 1,
            };
        }

        human_contact_allowed = profile.human_contact_permitted && human.contact_consent;
        if !human_contact_allowed {
            limits.max_human_contact_force_n = 0.0;
        }
    } else {
        // No human is currently present, so intentional human contact is not an
        // available capability regardless of a profile's nominal permission.
        limits.max_human_contact_force_n = 0.0;
    }

    HumanoidCapabilityEnvelope {
        subject_fingerprint,
        disposition,
        goal_authority_scale: authority_scale,
        goal_execution_allowed: true,
        human_contact_allowed,
        limits,
        restrictions_applied: usize::from(human.human_present),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subject(task: HumanoidTask) -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dmc21,
            task,
            ActuationMode::NormalizedTorque,
            "test-backend-v1",
        )
    }

    fn profile(subject: &HumanoidQualificationSubject) -> HumanoidNominalCapabilityProfile {
        // Test-only values: these exercise restriction algebra and are not
        // production safety limits.
        HumanoidNominalCapabilityProfile::new(
            subject, 2.0, 0.4, 1.0, 1.2, 0.2, 10.0, 80.0, 20.0, 0.8, true,
        )
    }

    #[test]
    fn profile_must_match_exact_qualification_subject() {
        let stand = subject(HumanoidTask::Stand);
        let run = subject(HumanoidTask::Run);
        let envelope = derive_humanoid_capability_envelope(
            &run,
            &profile(&stand),
            HumanoidAuthorityEnvelope::fully_admitted(),
            HumanInteractionEvidence::no_human_present(),
        );
        assert_eq!(envelope.disposition, HumanoidCapabilityDisposition::SubjectMismatch);
        assert!(!envelope.goal_execution_allowed);
        assert_eq!(envelope.limits, HumanoidCapabilityLimits::zero());
    }

    #[test]
    fn zero_authority_preserves_only_protective_execution() {
        let subject = subject(HumanoidTask::Stand);
        let envelope = derive_humanoid_capability_envelope(
            &subject,
            &profile(&subject),
            HumanoidAuthorityEnvelope::default(),
            HumanInteractionEvidence::no_human_present(),
        );
        assert_eq!(envelope.disposition, HumanoidCapabilityDisposition::ProtectiveOnly);
        assert!(!envelope.goal_execution_allowed);
        assert_eq!(envelope.goal_authority_scale, 0.0);
    }

    #[test]
    fn human_presence_selects_qualified_near_human_caps() {
        let subject = subject(HumanoidTask::Stand);
        let envelope = derive_humanoid_capability_envelope(
            &subject,
            &profile(&subject),
            HumanoidAuthorityEnvelope::fully_admitted(),
            HumanInteractionEvidence {
                human_present: true,
                proximity_valid: true,
                nearest_human_distance_m: 2.0,
                contact_consent: false,
            },
        );
        assert!(envelope.goal_execution_allowed);
        assert_eq!(envelope.limits.max_horizontal_speed_mps, 0.4);
        assert_eq!(envelope.limits.max_end_effector_speed_mps, 0.2);
        assert!(!envelope.human_contact_allowed);
        assert_eq!(envelope.limits.max_human_contact_force_n, 0.0);
    }

    #[test]
    fn unconsented_close_human_requires_hold() {
        let subject = subject(HumanoidTask::Stand);
        let envelope = derive_humanoid_capability_envelope(
            &subject,
            &profile(&subject),
            HumanoidAuthorityEnvelope::fully_admitted(),
            HumanInteractionEvidence {
                human_present: true,
                proximity_valid: true,
                nearest_human_distance_m: 0.5,
                contact_consent: false,
            },
        );
        assert_eq!(envelope.disposition, HumanoidCapabilityDisposition::HumanProximityHold);
        assert!(!envelope.goal_execution_allowed);
        assert_eq!(envelope.limits.max_horizontal_speed_mps, 0.0);
    }

    #[test]
    fn operator_authority_does_not_imply_contact_consent() {
        let subject = subject(HumanoidTask::Stand);
        let envelope = derive_humanoid_capability_envelope(
            &subject,
            &profile(&subject),
            HumanoidAuthorityEnvelope::fully_admitted(),
            HumanInteractionEvidence {
                human_present: true,
                proximity_valid: true,
                nearest_human_distance_m: 2.0,
                contact_consent: false,
            },
        );
        assert!(!envelope.human_contact_allowed);
    }

    #[test]
    fn explicit_consent_can_admit_profile_bounded_human_contact() {
        let subject = subject(HumanoidTask::Stand);
        let envelope = derive_humanoid_capability_envelope(
            &subject,
            &profile(&subject),
            HumanoidAuthorityEnvelope::fully_admitted(),
            HumanInteractionEvidence {
                human_present: true,
                proximity_valid: true,
                nearest_human_distance_m: 0.5,
                contact_consent: true,
            },
        );
        assert!(envelope.goal_execution_allowed);
        assert!(envelope.human_contact_allowed);
        assert_eq!(envelope.limits.max_human_contact_force_n, 20.0);
    }

    #[test]
    fn concrete_runtime_restrictions_are_monotonic() {
        let subject = subject(HumanoidTask::Stand);
        let mut envelope = derive_humanoid_capability_envelope(
            &subject,
            &profile(&subject),
            HumanoidAuthorityEnvelope::fully_admitted(),
            HumanInteractionEvidence::no_human_present(),
        );
        envelope.apply_restriction(HumanoidCapabilityRestriction {
            max_horizontal_speed_mps: 0.7,
            max_turn_rate_rad_s: 0.6,
            max_end_effector_speed_mps: 0.5,
            max_payload_kg: 3.0,
            max_object_contact_force_n: 40.0,
            max_human_contact_force_n: 0.0,
        });
        assert_eq!(envelope.limits.max_horizontal_speed_mps, 0.7);
        assert_eq!(envelope.limits.max_payload_kg, 3.0);
        assert_eq!(envelope.restrictions_applied, 1);
    }

    #[test]
    fn invalid_runtime_restriction_fails_closed() {
        let subject = subject(HumanoidTask::Stand);
        let mut envelope = derive_humanoid_capability_envelope(
            &subject,
            &profile(&subject),
            HumanoidAuthorityEnvelope::fully_admitted(),
            HumanInteractionEvidence::no_human_present(),
        );
        envelope.apply_restriction(HumanoidCapabilityRestriction {
            max_horizontal_speed_mps: f64::NAN,
            max_turn_rate_rad_s: 1.0,
            max_end_effector_speed_mps: 1.0,
            max_payload_kg: 1.0,
            max_object_contact_force_n: 1.0,
            max_human_contact_force_n: 0.0,
        });
        assert!(!envelope.goal_execution_allowed);
        assert_eq!(envelope.limits, HumanoidCapabilityLimits::zero());
    }
}
