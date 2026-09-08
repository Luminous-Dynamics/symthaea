// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Subject-bound capability restrictions from contact-wrench actuation margins.
//!
//! `actuator_controllability` measures what axis-aligned contact wrench authority
//! remains after per-joint degradation. This module answers the policy question:
//! is that measured authority sufficient for one exact qualified task/body/
//! backend, and if not, which concrete runtime capability limits must shrink?
//!
//! Thresholds are never invented here. The policy is explicit, subject-bound,
//! and has no `Default` implementation.

use serde::{Deserialize, Serialize};

use crate::actuator_controllability::{
    ContactWrenchAxis, HumanoidActuationControllabilityAssessment,
};
use crate::capability_envelope::{
    HumanoidCapabilityDisposition, HumanoidCapabilityEnvelope, HumanoidCapabilityLimits,
    HumanoidCapabilityRestriction, HumanoidNominalCapabilityProfile,
};
use crate::qualification::HumanoidQualificationSubject;

pub const HUMANOID_ACTUATION_CAPABILITY_POLICY_SCHEMA_VERSION: u32 = 1;

/// One qualified contact-wrench requirement for the current task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidContactActuationRequirement {
    pub site_id: String,
    pub axis: ContactWrenchAxis,
    /// Retained/nominal fraction required for full operation.
    pub nominal_min_retained_fraction: f64,
    /// Retained/nominal fraction required for degraded operation.
    pub degraded_min_retained_fraction: f64,
    /// Absolute axis-aligned retained wrench required for full operation.
    /// Units are N for force axes and N·m for moment axes.
    pub nominal_min_retained_wrench: f64,
    /// Absolute axis-aligned retained wrench required for degraded operation.
    pub degraded_min_retained_wrench: f64,
}

impl HumanoidContactActuationRequirement {
    pub fn validate(&self) -> bool {
        !self.site_id.trim().is_empty()
            && self.site_id == self.site_id.trim()
            && self.site_id.len() <= 256
            && self.nominal_min_retained_fraction.is_finite()
            && self.degraded_min_retained_fraction.is_finite()
            && (0.0..=1.0).contains(&self.nominal_min_retained_fraction)
            && (0.0..=1.0).contains(&self.degraded_min_retained_fraction)
            && self.nominal_min_retained_fraction > 0.0
            && self.degraded_min_retained_fraction > 0.0
            && self.degraded_min_retained_fraction <= self.nominal_min_retained_fraction
            && self.nominal_min_retained_wrench.is_finite()
            && self.degraded_min_retained_wrench.is_finite()
            && self.nominal_min_retained_wrench > 0.0
            && self.degraded_min_retained_wrench > 0.0
            && self.degraded_min_retained_wrench <= self.nominal_min_retained_wrench
    }
}

/// Qualified actuation requirements and degraded concrete limits for one exact
/// humanoid execution subject.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidActuationCapabilityPolicy {
    pub schema_version: u32,
    pub subject_fingerprint: u64,
    /// Exact producer/profile identity expected for per-joint authority evidence.
    pub authority_profile_id: String,
    /// Calibration/actuator-map identity that was qualified with this policy.
    pub calibration_fingerprint: u64,
    /// Exact full-dynamics model identity used to qualify these requirements.
    pub dynamics_model_id: String,
    pub requirements: Vec<HumanoidContactActuationRequirement>,
    /// Concrete limits applied when every requirement remains above the degraded
    /// floor but one or more fall below the nominal floor.
    pub degraded_restriction: HumanoidCapabilityRestriction,
}

impl HumanoidActuationCapabilityPolicy {
    pub fn validate_for(
        &self,
        subject: &HumanoidQualificationSubject,
        profile: &HumanoidNominalCapabilityProfile,
    ) -> bool {
        if self.schema_version != HUMANOID_ACTUATION_CAPABILITY_POLICY_SCHEMA_VERSION
            || !subject.validate()
            || self.subject_fingerprint == 0
            || self.subject_fingerprint != subject.fingerprint()
            || !profile.validate_for(subject)
            || self.authority_profile_id.trim().is_empty()
            || self.authority_profile_id != self.authority_profile_id.trim()
            || self.authority_profile_id.len() > 256
            || self.calibration_fingerprint == 0
            || self.dynamics_model_id.trim().is_empty()
            || self.dynamics_model_id != self.dynamics_model_id.trim()
            || self.dynamics_model_id.len() > 256
            || self.requirements.is_empty()
            || !self.degraded_restriction.validate()
        {
            return false;
        }

        if self.requirements.iter().any(|requirement| !requirement.validate()) {
            return false;
        }
        for (index, requirement) in self.requirements.iter().enumerate() {
            if self.requirements[..index].iter().any(|previous| {
                previous.site_id == requirement.site_id && previous.axis == requirement.axis
            }) {
                return false;
            }
        }

        let r = self.degraded_restriction;
        r.max_horizontal_speed_mps <= profile.max_horizontal_speed_mps
            && r.max_turn_rate_rad_s <= profile.max_turn_rate_rad_s
            && r.max_end_effector_speed_mps <= profile.max_end_effector_speed_mps
            && r.max_payload_kg <= profile.max_payload_kg
            && r.max_object_contact_force_n <= profile.max_object_contact_force_n
            && r.max_human_contact_force_n <= profile.max_human_contact_force_n
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidActuationCapabilityAction {
    Continue,
    Reduce,
    Hold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidActuationCapabilityFailureKind {
    InvalidPolicy,
    SubjectMismatch,
    AuthorityProfileMismatch,
    CalibrationMismatch,
    DynamicsModelMismatch,
    MissingRequiredSite,
    MissingRequiredAxis,
    UnsupportedRequiredAxis,
    BelowDegradedFraction,
    BelowDegradedAbsoluteWrench,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanoidActuationCapabilityFailure {
    pub requirement_index: Option<usize>,
    pub kind: HumanoidActuationCapabilityFailureKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidActuationCapabilityDecision {
    pub subject_fingerprint: u64,
    pub action: HumanoidActuationCapabilityAction,
    pub failures: Vec<HumanoidActuationCapabilityFailure>,
    pub degraded_requirements: Vec<usize>,
    pub restriction: Option<HumanoidCapabilityRestriction>,
}

impl HumanoidActuationCapabilityDecision {
    /// Apply the decision to the same subject-bound capability envelope.
    /// `Hold` revokes goal execution rather than merely setting numeric limits to
    /// zero, so even a zero-motion goal request cannot pass through stale intent.
    pub fn apply_to_envelope(&self, envelope: &mut HumanoidCapabilityEnvelope) {
        if envelope.subject_fingerprint != self.subject_fingerprint {
            revoke_goal(envelope);
            return;
        }
        match self.action {
            HumanoidActuationCapabilityAction::Continue => {}
            HumanoidActuationCapabilityAction::Reduce => {
                if let Some(restriction) = self.restriction {
                    envelope.apply_restriction(restriction);
                    if envelope.goal_execution_allowed
                        && envelope.disposition == HumanoidCapabilityDisposition::Nominal
                    {
                        envelope.disposition = HumanoidCapabilityDisposition::ReducedAuthority;
                    }
                } else {
                    revoke_goal(envelope);
                }
            }
            HumanoidActuationCapabilityAction::Hold => revoke_goal(envelope),
        }
    }
}

/// Evaluate measured contact-wrench authority against explicit qualified floors.
pub fn assess_humanoid_actuation_capability(
    subject: &HumanoidQualificationSubject,
    profile: &HumanoidNominalCapabilityProfile,
    policy: &HumanoidActuationCapabilityPolicy,
    assessment: &HumanoidActuationControllabilityAssessment,
) -> HumanoidActuationCapabilityDecision {
    let subject_fingerprint = subject.fingerprint();
    let mut failures = Vec::new();
    let mut degraded_requirements = Vec::new();

    if !policy.validate_for(subject, profile) {
        failures.push(HumanoidActuationCapabilityFailure {
            requirement_index: None,
            kind: HumanoidActuationCapabilityFailureKind::InvalidPolicy,
        });
    }
    if subject_fingerprint == 0 || policy.subject_fingerprint != subject_fingerprint {
        failures.push(HumanoidActuationCapabilityFailure {
            requirement_index: None,
            kind: HumanoidActuationCapabilityFailureKind::SubjectMismatch,
        });
    }
    if assessment.authority_profile_id != policy.authority_profile_id {
        failures.push(HumanoidActuationCapabilityFailure {
            requirement_index: None,
            kind: HumanoidActuationCapabilityFailureKind::AuthorityProfileMismatch,
        });
    }
    if assessment.calibration_fingerprint != policy.calibration_fingerprint {
        failures.push(HumanoidActuationCapabilityFailure {
            requirement_index: None,
            kind: HumanoidActuationCapabilityFailureKind::CalibrationMismatch,
        });
    }
    if assessment.dynamics_model_id != policy.dynamics_model_id {
        failures.push(HumanoidActuationCapabilityFailure {
            requirement_index: None,
            kind: HumanoidActuationCapabilityFailureKind::DynamicsModelMismatch,
        });
    }

    if failures.is_empty() {
        for (index, requirement) in policy.requirements.iter().enumerate() {
            let Some(site) = assessment.site(&requirement.site_id) else {
                failures.push(HumanoidActuationCapabilityFailure {
                    requirement_index: Some(index),
                    kind: HumanoidActuationCapabilityFailureKind::MissingRequiredSite,
                });
                continue;
            };
            let Some(axis) = site.axes.iter().find(|axis| axis.axis == requirement.axis) else {
                failures.push(HumanoidActuationCapabilityFailure {
                    requirement_index: Some(index),
                    kind: HumanoidActuationCapabilityFailureKind::MissingRequiredAxis,
                });
                continue;
            };
            if !axis.actuated_support_present {
                failures.push(HumanoidActuationCapabilityFailure {
                    requirement_index: Some(index),
                    kind: HumanoidActuationCapabilityFailureKind::UnsupportedRequiredAxis,
                });
                continue;
            }
            let retained_wrench = axis.retained_limit.unwrap_or(0.0);
            if axis.retained_fraction < requirement.degraded_min_retained_fraction {
                failures.push(HumanoidActuationCapabilityFailure {
                    requirement_index: Some(index),
                    kind: HumanoidActuationCapabilityFailureKind::BelowDegradedFraction,
                });
                continue;
            }
            if retained_wrench < requirement.degraded_min_retained_wrench {
                failures.push(HumanoidActuationCapabilityFailure {
                    requirement_index: Some(index),
                    kind: HumanoidActuationCapabilityFailureKind::BelowDegradedAbsoluteWrench,
                });
                continue;
            }
            if axis.retained_fraction < requirement.nominal_min_retained_fraction
                || retained_wrench < requirement.nominal_min_retained_wrench
            {
                degraded_requirements.push(index);
            }
        }
    }

    let action = if !failures.is_empty() {
        HumanoidActuationCapabilityAction::Hold
    } else if !degraded_requirements.is_empty() {
        HumanoidActuationCapabilityAction::Reduce
    } else {
        HumanoidActuationCapabilityAction::Continue
    };
    HumanoidActuationCapabilityDecision {
        subject_fingerprint,
        action,
        failures,
        degraded_requirements,
        restriction: (action == HumanoidActuationCapabilityAction::Reduce)
            .then_some(policy.degraded_restriction),
    }
}

fn revoke_goal(envelope: &mut HumanoidCapabilityEnvelope) {
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
        ContactWrenchAxisMargin, ContactWrenchMarginAssessment,
    };
    use crate::capability_envelope::{
        HumanInteractionEvidence, derive_humanoid_capability_envelope,
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
            "actuation-capability-test-v1",
        )
    }

    fn profile() -> HumanoidNominalCapabilityProfile {
        HumanoidNominalCapabilityProfile::new(
            &subject(), 2.0, 0.5, 1.0, 0.8, 0.2, 10.0, 80.0, 20.0, 0.8, true,
        )
    }

    fn policy() -> HumanoidActuationCapabilityPolicy {
        HumanoidActuationCapabilityPolicy {
            schema_version: HUMANOID_ACTUATION_CAPABILITY_POLICY_SCHEMA_VERSION,
            subject_fingerprint: subject().fingerprint(),
            authority_profile_id: "joint-authority-v1".into(),
            calibration_fingerprint: 42,
            dynamics_model_id: "dynamics-v1".into(),
            requirements: vec![HumanoidContactActuationRequirement {
                site_id: "right_foot".into(),
                axis: ContactWrenchAxis::ForceZ,
                nominal_min_retained_fraction: 0.8,
                degraded_min_retained_fraction: 0.5,
                nominal_min_retained_wrench: 80.0,
                degraded_min_retained_wrench: 50.0,
            }],
            degraded_restriction: HumanoidCapabilityRestriction {
                max_horizontal_speed_mps: 0.5,
                max_turn_rate_rad_s: 0.3,
                max_end_effector_speed_mps: 0.4,
                max_payload_kg: 5.0,
                max_object_contact_force_n: 40.0,
                max_human_contact_force_n: 10.0,
            },
        }
    }

    fn assessment(fraction: f64, retained_wrench: f64) -> HumanoidActuationControllabilityAssessment {
        HumanoidActuationControllabilityAssessment {
            morphology: HumanoidMorphology::Dmc21,
            authority_sequence: 1,
            authority_age_s: 0.01,
            authority_profile_id: "joint-authority-v1".into(),
            calibration_fingerprint: 42,
            dynamics_model_id: "dynamics-v1".into(),
            sites: vec![ContactWrenchMarginAssessment {
                site_id: "right_foot".into(),
                contact_confidence: 1.0,
                jacobian_source: DynamicsComponentSource::SimulatorSolver,
                actuator_limit_source: DynamicsComponentSource::SystemIdentification,
                axes: vec![ContactWrenchAxisMargin {
                    axis: ContactWrenchAxis::ForceZ,
                    actuated_support_present: true,
                    nominal_limit: Some(100.0),
                    retained_limit: Some(retained_wrench),
                    retained_fraction: fraction,
                    limiting_joint: Some(12),
                }],
                minimum_retained_fraction: fraction,
            }],
        }
    }

    fn envelope() -> HumanoidCapabilityEnvelope {
        derive_humanoid_capability_envelope(
            &subject(),
            &profile(),
            HumanoidAuthorityEnvelope::fully_admitted(),
            HumanInteractionEvidence::no_human_present(),
        )
    }

    #[test]
    fn nominal_fraction_and_absolute_margin_continue() {
        let decision = assess_humanoid_actuation_capability(
            &subject(),
            &profile(),
            &policy(),
            &assessment(0.9, 90.0),
        );
        assert_eq!(decision.action, HumanoidActuationCapabilityAction::Continue);
        assert!(decision.failures.is_empty());
    }

    #[test]
    fn degraded_but_qualified_margin_applies_explicit_caps() {
        let decision = assess_humanoid_actuation_capability(
            &subject(),
            &profile(),
            &policy(),
            &assessment(0.6, 60.0),
        );
        assert_eq!(decision.action, HumanoidActuationCapabilityAction::Reduce);
        let mut envelope = envelope();
        decision.apply_to_envelope(&mut envelope);
        assert!(envelope.goal_execution_allowed);
        assert_eq!(envelope.limits.max_horizontal_speed_mps, 0.5);
        assert_eq!(envelope.limits.max_payload_kg, 5.0);
        assert_eq!(envelope.disposition, HumanoidCapabilityDisposition::ReducedAuthority);
    }

    #[test]
    fn fraction_below_degraded_floor_holds_goal_execution() {
        let decision = assess_humanoid_actuation_capability(
            &subject(),
            &profile(),
            &policy(),
            &assessment(0.4, 90.0),
        );
        assert_eq!(decision.action, HumanoidActuationCapabilityAction::Hold);
        assert!(decision.failures.iter().any(|failure| {
            failure.kind == HumanoidActuationCapabilityFailureKind::BelowDegradedFraction
        }));
        let mut envelope = envelope();
        decision.apply_to_envelope(&mut envelope);
        assert!(!envelope.goal_execution_allowed);
        assert_eq!(envelope.limits, HumanoidCapabilityLimits::zero());
    }

    #[test]
    fn absolute_wrench_below_floor_holds_even_with_good_fraction() {
        let decision = assess_humanoid_actuation_capability(
            &subject(),
            &profile(),
            &policy(),
            &assessment(0.9, 40.0),
        );
        assert_eq!(decision.action, HumanoidActuationCapabilityAction::Hold);
        assert!(decision.failures.iter().any(|failure| {
            failure.kind
                == HumanoidActuationCapabilityFailureKind::BelowDegradedAbsoluteWrench
        }));
    }

    #[test]
    fn calibration_drift_fails_closed() {
        let mut assessment = assessment(1.0, 100.0);
        assessment.calibration_fingerprint = 43;
        let decision = assess_humanoid_actuation_capability(
            &subject(),
            &profile(),
            &policy(),
            &assessment,
        );
        assert_eq!(decision.action, HumanoidActuationCapabilityAction::Hold);
        assert!(decision.failures.iter().any(|failure| {
            failure.kind == HumanoidActuationCapabilityFailureKind::CalibrationMismatch
        }));
    }

    #[test]
    fn missing_required_site_fails_closed() {
        let mut assessment = assessment(1.0, 100.0);
        assessment.sites.clear();
        let decision = assess_humanoid_actuation_capability(
            &subject(),
            &profile(),
            &policy(),
            &assessment,
        );
        assert_eq!(decision.action, HumanoidActuationCapabilityAction::Hold);
        assert!(decision.failures.iter().any(|failure| {
            failure.kind == HumanoidActuationCapabilityFailureKind::MissingRequiredSite
        }));
    }
}
