// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Terrain-derived restrictions for the humanoid runtime capability envelope.
//!
//! The terrain planner already exposes concrete uncertainty/freshness evidence.
//! This module converts that evidence into subject-bound capability tiers without
//! inventing a linear confidence-to-speed relationship. The tier thresholds and
//! degraded locomotion ceilings must be supplied by qualification for the exact
//! humanoid subject.

use serde::{Deserialize, Serialize};

use crate::capability_envelope::{
    HumanoidCapabilityDisposition, HumanoidCapabilityEnvelope, HumanoidCapabilityRestriction,
    HumanoidNominalCapabilityProfile,
};
use crate::hierarchical::HierarchicalControlReport;
use crate::qualification::HumanoidQualificationSubject;

pub const HUMANOID_TERRAIN_CAPABILITY_POLICY_SCHEMA_VERSION: u32 = 1;

/// Worst-case terrain evidence retained for the active control horizon.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HumanoidTerrainCapabilityEvidence {
    pub horizon_steps: usize,
    pub minimum_confidence: f64,
    pub maximum_height_std_m: f64,
    pub maximum_evidence_age_s: f64,
}

impl HumanoidTerrainCapabilityEvidence {
    pub fn from_hierarchy_report(report: &HierarchicalControlReport) -> Self {
        Self {
            horizon_steps: report.terrain_horizon_steps,
            minimum_confidence: report.terrain_confidence,
            maximum_height_std_m: report.terrain_max_height_std_m,
            maximum_evidence_age_s: report.terrain_max_evidence_age_s,
        }
    }

    pub fn validate(self) -> bool {
        self.minimum_confidence.is_finite()
            && (0.0..=1.0).contains(&self.minimum_confidence)
            && self.maximum_height_std_m.is_finite()
            && self.maximum_height_std_m >= 0.0
            && self.maximum_evidence_age_s.is_finite()
            && self.maximum_evidence_age_s >= 0.0
    }
}

/// Subject-bound terrain policy. There is intentionally no `Default`: these
/// thresholds and degraded ceilings are engineering/qualification inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidTerrainCapabilityPolicy {
    pub schema_version: u32,
    pub subject_fingerprint: u64,
    pub require_active_horizon: bool,
    pub minimum_terrain_confidence: f64,
    pub maximum_nominal_height_std_m: f64,
    pub maximum_nominal_evidence_age_s: f64,
    pub maximum_degraded_height_std_m: f64,
    pub maximum_degraded_evidence_age_s: f64,
    pub degraded_max_horizontal_speed_mps: f64,
    pub degraded_max_turn_rate_rad_s: f64,
}

impl HumanoidTerrainCapabilityPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &HumanoidQualificationSubject,
        require_active_horizon: bool,
        minimum_terrain_confidence: f64,
        maximum_nominal_height_std_m: f64,
        maximum_nominal_evidence_age_s: f64,
        maximum_degraded_height_std_m: f64,
        maximum_degraded_evidence_age_s: f64,
        degraded_max_horizontal_speed_mps: f64,
        degraded_max_turn_rate_rad_s: f64,
    ) -> Self {
        Self {
            schema_version: HUMANOID_TERRAIN_CAPABILITY_POLICY_SCHEMA_VERSION,
            subject_fingerprint: subject.fingerprint(),
            require_active_horizon,
            minimum_terrain_confidence,
            maximum_nominal_height_std_m,
            maximum_nominal_evidence_age_s,
            maximum_degraded_height_std_m,
            maximum_degraded_evidence_age_s,
            degraded_max_horizontal_speed_mps,
            degraded_max_turn_rate_rad_s,
        }
    }

    pub fn validate_for(
        &self,
        subject: &HumanoidQualificationSubject,
        profile: &HumanoidNominalCapabilityProfile,
    ) -> bool {
        if self.schema_version != HUMANOID_TERRAIN_CAPABILITY_POLICY_SCHEMA_VERSION
            || self.subject_fingerprint == 0
            || self.subject_fingerprint != subject.fingerprint()
            || !profile.validate_for(subject)
        {
            return false;
        }

        let finite_non_negative = [
            self.maximum_nominal_height_std_m,
            self.maximum_nominal_evidence_age_s,
            self.maximum_degraded_height_std_m,
            self.maximum_degraded_evidence_age_s,
            self.degraded_max_horizontal_speed_mps,
            self.degraded_max_turn_rate_rad_s,
        ]
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0);

        self.minimum_terrain_confidence.is_finite()
            && (0.0..=1.0).contains(&self.minimum_terrain_confidence)
            && finite_non_negative
            && self.maximum_nominal_height_std_m <= self.maximum_degraded_height_std_m
            && self.maximum_nominal_evidence_age_s <= self.maximum_degraded_evidence_age_s
            && self.degraded_max_horizontal_speed_mps <= profile.max_horizontal_speed_mps
            && self.degraded_max_turn_rate_rad_s <= profile.max_turn_rate_rad_s
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidTerrainCapabilityAction {
    Continue,
    Reduce,
    Hold,
}

/// Auditable terrain decision plus the concrete min-wise restriction to apply.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HumanoidTerrainCapabilityAssessment {
    pub action: HumanoidTerrainCapabilityAction,
    pub evidence: HumanoidTerrainCapabilityEvidence,
    pub restriction: HumanoidCapabilityRestriction,
}

impl HumanoidTerrainCapabilityAssessment {
    /// Apply the concrete bounds. A terrain `Hold` also revokes goal-directed
    /// execution while leaving independently admitted protective behavior intact.
    pub fn apply_to(self, envelope: &mut HumanoidCapabilityEnvelope) {
        envelope.apply_restriction(self.restriction);
        if self.action == HumanoidTerrainCapabilityAction::Hold {
            envelope.goal_execution_allowed = false;
            envelope.human_contact_allowed = false;
            envelope.disposition = HumanoidCapabilityDisposition::ProtectiveOnly;
        }
    }
}

/// Evaluate terrain evidence into nominal/degraded/hold tiers.
///
/// No interpolation is performed. Qualification explicitly supplies both the
/// tier boundaries and the degraded speed/turn ceilings.
pub fn assess_humanoid_terrain_capability(
    subject: &HumanoidQualificationSubject,
    profile: &HumanoidNominalCapabilityProfile,
    policy: &HumanoidTerrainCapabilityPolicy,
    evidence: HumanoidTerrainCapabilityEvidence,
    current: &HumanoidCapabilityEnvelope,
) -> HumanoidTerrainCapabilityAssessment {
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
    let horizon_ok = !policy.require_active_horizon || evidence.horizon_steps > 0;
    if !valid_binding || !evidence.validate() || !horizon_ok {
        return HumanoidTerrainCapabilityAssessment {
            action: HumanoidTerrainCapabilityAction::Hold,
            evidence,
            restriction: HumanoidCapabilityRestriction {
                max_horizontal_speed_mps: 0.0,
                max_turn_rate_rad_s: 0.0,
                ..preserve
            },
        };
    }

    let confidence_ok = evidence.minimum_confidence >= policy.minimum_terrain_confidence;
    let nominal = confidence_ok
        && evidence.maximum_height_std_m <= policy.maximum_nominal_height_std_m
        && evidence.maximum_evidence_age_s <= policy.maximum_nominal_evidence_age_s;
    if nominal {
        return HumanoidTerrainCapabilityAssessment {
            action: HumanoidTerrainCapabilityAction::Continue,
            evidence,
            restriction: preserve,
        };
    }

    let degraded = confidence_ok
        && evidence.maximum_height_std_m <= policy.maximum_degraded_height_std_m
        && evidence.maximum_evidence_age_s <= policy.maximum_degraded_evidence_age_s;
    if degraded {
        return HumanoidTerrainCapabilityAssessment {
            action: HumanoidTerrainCapabilityAction::Reduce,
            evidence,
            restriction: HumanoidCapabilityRestriction {
                max_horizontal_speed_mps: current
                    .limits
                    .max_horizontal_speed_mps
                    .min(policy.degraded_max_horizontal_speed_mps),
                max_turn_rate_rad_s: current
                    .limits
                    .max_turn_rate_rad_s
                    .min(policy.degraded_max_turn_rate_rad_s),
                ..preserve
            },
        };
    }

    HumanoidTerrainCapabilityAssessment {
        action: HumanoidTerrainCapabilityAction::Hold,
        evidence,
        restriction: HumanoidCapabilityRestriction {
            max_horizontal_speed_mps: 0.0,
            max_turn_rate_rad_s: 0.0,
            ..preserve
        },
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
    use crate::types::{ActuationMode, HumanoidTask};

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dmc21,
            HumanoidTask::Walk,
            ActuationMode::NormalizedTorque,
            "test-backend-v1",
        )
    }

    fn profile(subject: &HumanoidQualificationSubject) -> HumanoidNominalCapabilityProfile {
        // Test-only values, not production safety limits.
        HumanoidNominalCapabilityProfile::new(
            subject, 2.0, 0.4, 1.0, 1.2, 0.2, 10.0, 80.0, 0.0, 0.8, false,
        )
    }

    fn policy(subject: &HumanoidQualificationSubject) -> HumanoidTerrainCapabilityPolicy {
        // Test-only tier boundaries and caps.
        HumanoidTerrainCapabilityPolicy::new(subject, true, 0.7, 0.02, 0.10, 0.08, 0.30, 0.5, 0.25)
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

    #[test]
    fn nominal_terrain_preserves_qualified_limits() {
        let subject = subject();
        let profile = profile(&subject);
        let current = envelope(&subject, &profile);
        let assessment = assess_humanoid_terrain_capability(
            &subject,
            &profile,
            &policy(&subject),
            HumanoidTerrainCapabilityEvidence {
                horizon_steps: 4,
                minimum_confidence: 0.95,
                maximum_height_std_m: 0.01,
                maximum_evidence_age_s: 0.05,
            },
            &current,
        );
        assert_eq!(assessment.action, HumanoidTerrainCapabilityAction::Continue);
        assert_eq!(assessment.restriction.max_horizontal_speed_mps, 2.0);
    }

    #[test]
    fn degraded_terrain_uses_explicit_qualified_caps() {
        let subject = subject();
        let profile = profile(&subject);
        let mut current = envelope(&subject, &profile);
        let assessment = assess_humanoid_terrain_capability(
            &subject,
            &profile,
            &policy(&subject),
            HumanoidTerrainCapabilityEvidence {
                horizon_steps: 4,
                minimum_confidence: 0.8,
                maximum_height_std_m: 0.05,
                maximum_evidence_age_s: 0.20,
            },
            &current,
        );
        assert_eq!(assessment.action, HumanoidTerrainCapabilityAction::Reduce);
        assessment.apply_to(&mut current);
        assert_eq!(current.limits.max_horizontal_speed_mps, 0.5);
        assert_eq!(current.limits.max_turn_rate_rad_s, 0.25);
        assert!(current.goal_execution_allowed);
    }

    #[test]
    fn stale_or_uncertain_terrain_requires_hold() {
        let subject = subject();
        let profile = profile(&subject);
        let mut current = envelope(&subject, &profile);
        let assessment = assess_humanoid_terrain_capability(
            &subject,
            &profile,
            &policy(&subject),
            HumanoidTerrainCapabilityEvidence {
                horizon_steps: 4,
                minimum_confidence: 0.8,
                maximum_height_std_m: 0.20,
                maximum_evidence_age_s: 1.0,
            },
            &current,
        );
        assert_eq!(assessment.action, HumanoidTerrainCapabilityAction::Hold);
        assessment.apply_to(&mut current);
        assert!(!current.goal_execution_allowed);
        assert_eq!(current.limits.max_horizontal_speed_mps, 0.0);
        assert_eq!(current.disposition, HumanoidCapabilityDisposition::ProtectiveOnly);
    }

    #[test]
    fn missing_required_horizon_fails_closed() {
        let subject = subject();
        let profile = profile(&subject);
        let mut current = envelope(&subject, &profile);
        let assessment = assess_humanoid_terrain_capability(
            &subject,
            &profile,
            &policy(&subject),
            HumanoidTerrainCapabilityEvidence {
                horizon_steps: 0,
                minimum_confidence: 1.0,
                maximum_height_std_m: 0.0,
                maximum_evidence_age_s: 0.0,
            },
            &current,
        );
        assessment.apply_to(&mut current);
        assert!(!current.goal_execution_allowed);
    }

    #[test]
    fn mismatched_subject_policy_fails_closed() {
        let subject = subject();
        let other = HumanoidQualificationSubject::new(
            HumanoidMorphology::Dmc21,
            HumanoidTask::Run,
            ActuationMode::NormalizedTorque,
            "test-backend-v1",
        );
        let profile = profile(&subject);
        let current = envelope(&subject, &profile);
        let assessment = assess_humanoid_terrain_capability(
            &subject,
            &profile,
            &policy(&other),
            HumanoidTerrainCapabilityEvidence {
                horizon_steps: 4,
                minimum_confidence: 1.0,
                maximum_height_std_m: 0.0,
                maximum_evidence_age_s: 0.0,
            },
            &current,
        );
        assert_eq!(assessment.action, HumanoidTerrainCapabilityAction::Hold);
    }
}
