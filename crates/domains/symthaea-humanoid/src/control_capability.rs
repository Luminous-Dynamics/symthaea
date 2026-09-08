// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Whole-body control-feasibility restrictions for humanoid capabilities.
//!
//! This is intentionally named control feasibility rather than actuator
//! controllability. The current humanoid stack exposes solver feasibility,
//! residuals, fallbacks, constraint utilization, and deadline evidence, but it
//! does not yet expose a full actuator-to-task controllability matrix like the
//! helicopter domain. This module uses only the evidence that actually exists.

use serde::{Deserialize, Serialize};

use crate::capability_envelope::{
    HumanoidCapabilityDisposition, HumanoidCapabilityEnvelope, HumanoidCapabilityRestriction,
    HumanoidNominalCapabilityProfile,
};
use crate::hierarchical::HierarchicalControlReport;
use crate::qualification::HumanoidQualificationSubject;

pub const HUMANOID_CONTROL_CAPABILITY_POLICY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HumanoidControlCapabilityEvidence {
    pub whole_body_feasible: bool,
    pub whole_body_joint_utilization: f64,
    pub whole_body_objective_residual: f64,
    pub inverse_dynamics_max_violation: f64,
    pub inverse_dynamics_fallback: bool,
    pub contact_dynamics_converged: bool,
    pub contact_dynamics_fallback: bool,
    pub contact_dynamics_residual_nm: f64,
    pub contact_solver_budget_missed: bool,
    pub floating_base_model_available: bool,
    pub floating_base_dynamics_converged: bool,
    pub floating_base_dynamics_fallback: bool,
    pub floating_base_dynamics_residual: f64,
    pub floating_base_solver_budget_missed: bool,
}

impl HumanoidControlCapabilityEvidence {
    pub fn from_hierarchy_report(report: &HierarchicalControlReport) -> Self {
        Self {
            whole_body_feasible: report.whole_body_feasible,
            whole_body_joint_utilization: report.whole_body_joint_utilization,
            whole_body_objective_residual: report.whole_body_objective_residual,
            inverse_dynamics_max_violation: report.inverse_dynamics_max_violation,
            inverse_dynamics_fallback: report.inverse_dynamics_fallback,
            contact_dynamics_converged: report.contact_dynamics_converged,
            contact_dynamics_fallback: report.contact_dynamics_fallback,
            contact_dynamics_residual_nm: report.contact_dynamics_residual_nm,
            contact_solver_budget_missed: report.contact_solver_budget_missed,
            floating_base_model_available: report.floating_base_model_available,
            floating_base_dynamics_converged: report.floating_base_dynamics_converged,
            floating_base_dynamics_fallback: report.floating_base_dynamics_fallback,
            floating_base_dynamics_residual: report.floating_base_dynamics_residual,
            floating_base_solver_budget_missed: report.floating_base_solver_budget_missed,
        }
    }

    pub fn validate(self) -> bool {
        [
            self.whole_body_joint_utilization,
            self.whole_body_objective_residual,
            self.inverse_dynamics_max_violation,
            self.contact_dynamics_residual_nm,
            self.floating_base_dynamics_residual,
        ]
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)
    }
}

/// Subject-bound solver/dynamics admission policy. No defaults are provided:
/// all residual limits and degraded motion ceilings must be qualified.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidControlCapabilityPolicy {
    pub schema_version: u32,
    pub subject_fingerprint: u64,
    pub require_contact_dynamics: bool,
    pub require_floating_base_dynamics: bool,
    pub allow_inverse_dynamics_fallback_degraded: bool,
    pub allow_contact_dynamics_fallback_degraded: bool,
    pub allow_floating_base_fallback_degraded: bool,
    pub maximum_nominal_joint_utilization: f64,
    pub maximum_degraded_joint_utilization: f64,
    pub maximum_nominal_objective_residual: f64,
    pub maximum_degraded_objective_residual: f64,
    pub maximum_nominal_inverse_dynamics_violation: f64,
    pub maximum_degraded_inverse_dynamics_violation: f64,
    pub maximum_nominal_contact_residual_nm: f64,
    pub maximum_degraded_contact_residual_nm: f64,
    pub maximum_nominal_floating_base_residual: f64,
    pub maximum_degraded_floating_base_residual: f64,
    pub degraded_max_horizontal_speed_mps: f64,
    pub degraded_max_turn_rate_rad_s: f64,
    pub degraded_max_end_effector_speed_mps: f64,
    pub degraded_max_object_contact_force_n: f64,
}

impl HumanoidControlCapabilityPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &HumanoidQualificationSubject,
        require_contact_dynamics: bool,
        require_floating_base_dynamics: bool,
        allow_inverse_dynamics_fallback_degraded: bool,
        allow_contact_dynamics_fallback_degraded: bool,
        allow_floating_base_fallback_degraded: bool,
        maximum_nominal_joint_utilization: f64,
        maximum_degraded_joint_utilization: f64,
        maximum_nominal_objective_residual: f64,
        maximum_degraded_objective_residual: f64,
        maximum_nominal_inverse_dynamics_violation: f64,
        maximum_degraded_inverse_dynamics_violation: f64,
        maximum_nominal_contact_residual_nm: f64,
        maximum_degraded_contact_residual_nm: f64,
        maximum_nominal_floating_base_residual: f64,
        maximum_degraded_floating_base_residual: f64,
        degraded_max_horizontal_speed_mps: f64,
        degraded_max_turn_rate_rad_s: f64,
        degraded_max_end_effector_speed_mps: f64,
        degraded_max_object_contact_force_n: f64,
    ) -> Self {
        Self {
            schema_version: HUMANOID_CONTROL_CAPABILITY_POLICY_SCHEMA_VERSION,
            subject_fingerprint: subject.fingerprint(),
            require_contact_dynamics,
            require_floating_base_dynamics,
            allow_inverse_dynamics_fallback_degraded,
            allow_contact_dynamics_fallback_degraded,
            allow_floating_base_fallback_degraded,
            maximum_nominal_joint_utilization,
            maximum_degraded_joint_utilization,
            maximum_nominal_objective_residual,
            maximum_degraded_objective_residual,
            maximum_nominal_inverse_dynamics_violation,
            maximum_degraded_inverse_dynamics_violation,
            maximum_nominal_contact_residual_nm,
            maximum_degraded_contact_residual_nm,
            maximum_nominal_floating_base_residual,
            maximum_degraded_floating_base_residual,
            degraded_max_horizontal_speed_mps,
            degraded_max_turn_rate_rad_s,
            degraded_max_end_effector_speed_mps,
            degraded_max_object_contact_force_n,
        }
    }

    pub fn validate_for(
        &self,
        subject: &HumanoidQualificationSubject,
        profile: &HumanoidNominalCapabilityProfile,
    ) -> bool {
        if self.schema_version != HUMANOID_CONTROL_CAPABILITY_POLICY_SCHEMA_VERSION
            || self.subject_fingerprint == 0
            || self.subject_fingerprint != subject.fingerprint()
            || !profile.validate_for(subject)
        {
            return false;
        }
        let values = [
            self.maximum_nominal_joint_utilization,
            self.maximum_degraded_joint_utilization,
            self.maximum_nominal_objective_residual,
            self.maximum_degraded_objective_residual,
            self.maximum_nominal_inverse_dynamics_violation,
            self.maximum_degraded_inverse_dynamics_violation,
            self.maximum_nominal_contact_residual_nm,
            self.maximum_degraded_contact_residual_nm,
            self.maximum_nominal_floating_base_residual,
            self.maximum_degraded_floating_base_residual,
            self.degraded_max_horizontal_speed_mps,
            self.degraded_max_turn_rate_rad_s,
            self.degraded_max_end_effector_speed_mps,
            self.degraded_max_object_contact_force_n,
        ];
        values.iter().all(|value| value.is_finite() && *value >= 0.0)
            && self.maximum_nominal_joint_utilization <= self.maximum_degraded_joint_utilization
            && self.maximum_nominal_objective_residual <= self.maximum_degraded_objective_residual
            && self.maximum_nominal_inverse_dynamics_violation
                <= self.maximum_degraded_inverse_dynamics_violation
            && self.maximum_nominal_contact_residual_nm <= self.maximum_degraded_contact_residual_nm
            && self.maximum_nominal_floating_base_residual
                <= self.maximum_degraded_floating_base_residual
            && self.degraded_max_horizontal_speed_mps <= profile.max_horizontal_speed_mps
            && self.degraded_max_turn_rate_rad_s <= profile.max_turn_rate_rad_s
            && self.degraded_max_end_effector_speed_mps <= profile.max_end_effector_speed_mps
            && self.degraded_max_object_contact_force_n <= profile.max_object_contact_force_n
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidControlCapabilityAction {
    Continue,
    Reduce,
    Hold,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HumanoidControlCapabilityAssessment {
    pub action: HumanoidControlCapabilityAction,
    pub evidence: HumanoidControlCapabilityEvidence,
    pub restriction: HumanoidCapabilityRestriction,
}

impl HumanoidControlCapabilityAssessment {
    pub fn apply_to(self, envelope: &mut HumanoidCapabilityEnvelope) {
        envelope.apply_restriction(self.restriction);
        if self.action == HumanoidControlCapabilityAction::Hold {
            envelope.goal_execution_allowed = false;
            envelope.human_contact_allowed = false;
            envelope.disposition = HumanoidCapabilityDisposition::ProtectiveOnly;
        }
    }
}

pub fn assess_humanoid_control_capability(
    subject: &HumanoidQualificationSubject,
    profile: &HumanoidNominalCapabilityProfile,
    policy: &HumanoidControlCapabilityPolicy,
    evidence: HumanoidControlCapabilityEvidence,
    current: &HumanoidCapabilityEnvelope,
) -> HumanoidControlCapabilityAssessment {
    let preserve = HumanoidCapabilityRestriction {
        max_horizontal_speed_mps: current.limits.max_horizontal_speed_mps,
        max_turn_rate_rad_s: current.limits.max_turn_rate_rad_s,
        max_end_effector_speed_mps: current.limits.max_end_effector_speed_mps,
        max_payload_kg: current.limits.max_payload_kg,
        max_object_contact_force_n: current.limits.max_object_contact_force_n,
        max_human_contact_force_n: current.limits.max_human_contact_force_n,
    };

    let binding_ok = current.subject_fingerprint != 0
        && current.subject_fingerprint == subject.fingerprint()
        && policy.validate_for(subject, profile);
    let required_models_ok = (!policy.require_contact_dynamics || evidence.contact_dynamics_converged)
        && (!policy.require_floating_base_dynamics
            || (evidence.floating_base_model_available && evidence.floating_base_dynamics_converged));
    let deadlines_ok = !evidence.contact_solver_budget_missed
        && !evidence.floating_base_solver_budget_missed;

    if !binding_ok
        || !evidence.validate()
        || !evidence.whole_body_feasible
        || !required_models_ok
        || !deadlines_ok
    {
        return hold(evidence, preserve);
    }

    let no_fallbacks = !evidence.inverse_dynamics_fallback
        && !evidence.contact_dynamics_fallback
        && !evidence.floating_base_dynamics_fallback;
    let nominal = no_fallbacks
        && evidence.whole_body_joint_utilization <= policy.maximum_nominal_joint_utilization
        && evidence.whole_body_objective_residual <= policy.maximum_nominal_objective_residual
        && evidence.inverse_dynamics_max_violation
            <= policy.maximum_nominal_inverse_dynamics_violation
        && evidence.contact_dynamics_residual_nm <= policy.maximum_nominal_contact_residual_nm
        && evidence.floating_base_dynamics_residual
            <= policy.maximum_nominal_floating_base_residual;
    if nominal {
        return HumanoidControlCapabilityAssessment {
            action: HumanoidControlCapabilityAction::Continue,
            evidence,
            restriction: preserve,
        };
    }

    let fallbacks_allowed = (!evidence.inverse_dynamics_fallback
        || policy.allow_inverse_dynamics_fallback_degraded)
        && (!evidence.contact_dynamics_fallback || policy.allow_contact_dynamics_fallback_degraded)
        && (!evidence.floating_base_dynamics_fallback
            || policy.allow_floating_base_fallback_degraded);
    let degraded = fallbacks_allowed
        && evidence.whole_body_joint_utilization <= policy.maximum_degraded_joint_utilization
        && evidence.whole_body_objective_residual <= policy.maximum_degraded_objective_residual
        && evidence.inverse_dynamics_max_violation
            <= policy.maximum_degraded_inverse_dynamics_violation
        && evidence.contact_dynamics_residual_nm <= policy.maximum_degraded_contact_residual_nm
        && evidence.floating_base_dynamics_residual
            <= policy.maximum_degraded_floating_base_residual;
    if degraded {
        return HumanoidControlCapabilityAssessment {
            action: HumanoidControlCapabilityAction::Reduce,
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
                max_end_effector_speed_mps: current
                    .limits
                    .max_end_effector_speed_mps
                    .min(policy.degraded_max_end_effector_speed_mps),
                max_object_contact_force_n: current
                    .limits
                    .max_object_contact_force_n
                    .min(policy.degraded_max_object_contact_force_n),
                ..preserve
            },
        };
    }

    hold(evidence, preserve)
}

fn hold(
    evidence: HumanoidControlCapabilityEvidence,
    preserve: HumanoidCapabilityRestriction,
) -> HumanoidControlCapabilityAssessment {
    HumanoidControlCapabilityAssessment {
        action: HumanoidControlCapabilityAction::Hold,
        evidence,
        restriction: HumanoidCapabilityRestriction {
            max_horizontal_speed_mps: 0.0,
            max_turn_rate_rad_s: 0.0,
            max_end_effector_speed_mps: 0.0,
            max_object_contact_force_n: 0.0,
            max_human_contact_force_n: 0.0,
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
            HumanoidTask::Walk,
            ActuationMode::NormalizedTorque,
            "test-backend-v1",
        )
    }

    fn profile(subject: &HumanoidQualificationSubject) -> HumanoidNominalCapabilityProfile {
        // Test-only values, not production limits.
        HumanoidNominalCapabilityProfile::new(
            subject, 2.0, 0.4, 1.0, 1.2, 0.2, 10.0, 80.0, 0.0, 0.8, false,
        )
    }

    fn policy(subject: &HumanoidQualificationSubject) -> HumanoidControlCapabilityPolicy {
        // Test-only solver thresholds and degraded caps.
        HumanoidControlCapabilityPolicy::new(
            subject,
            true,
            true,
            true,
            true,
            true,
            0.70,
            0.95,
            0.05,
            0.20,
            0.01,
            0.05,
            0.10,
            0.50,
            0.10,
            0.50,
            0.6,
            0.3,
            0.4,
            30.0,
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

    fn evidence() -> HumanoidControlCapabilityEvidence {
        HumanoidControlCapabilityEvidence {
            whole_body_feasible: true,
            whole_body_joint_utilization: 0.5,
            whole_body_objective_residual: 0.02,
            inverse_dynamics_max_violation: 0.005,
            inverse_dynamics_fallback: false,
            contact_dynamics_converged: true,
            contact_dynamics_fallback: false,
            contact_dynamics_residual_nm: 0.05,
            contact_solver_budget_missed: false,
            floating_base_model_available: true,
            floating_base_dynamics_converged: true,
            floating_base_dynamics_fallback: false,
            floating_base_dynamics_residual: 0.05,
            floating_base_solver_budget_missed: false,
        }
    }

    #[test]
    fn nominal_feasibility_preserves_capabilities() {
        let subject = subject();
        let profile = profile(&subject);
        let current = envelope(&subject, &profile);
        let result = assess_humanoid_control_capability(
            &subject,
            &profile,
            &policy(&subject),
            evidence(),
            &current,
        );
        assert_eq!(result.action, HumanoidControlCapabilityAction::Continue);
        assert_eq!(result.restriction.max_horizontal_speed_mps, 2.0);
    }

    #[test]
    fn degraded_feasibility_selects_explicit_caps() {
        let subject = subject();
        let profile = profile(&subject);
        let mut current = envelope(&subject, &profile);
        let mut sample = evidence();
        sample.whole_body_joint_utilization = 0.85;
        sample.inverse_dynamics_fallback = true;
        let result = assess_humanoid_control_capability(
            &subject,
            &profile,
            &policy(&subject),
            sample,
            &current,
        );
        assert_eq!(result.action, HumanoidControlCapabilityAction::Reduce);
        result.apply_to(&mut current);
        assert_eq!(current.limits.max_horizontal_speed_mps, 0.6);
        assert_eq!(current.limits.max_object_contact_force_n, 30.0);
    }

    #[test]
    fn infeasible_solution_requires_hold() {
        let subject = subject();
        let profile = profile(&subject);
        let mut current = envelope(&subject, &profile);
        let mut sample = evidence();
        sample.whole_body_feasible = false;
        let result = assess_humanoid_control_capability(
            &subject,
            &profile,
            &policy(&subject),
            sample,
            &current,
        );
        result.apply_to(&mut current);
        assert!(!current.goal_execution_allowed);
        assert_eq!(current.disposition, HumanoidCapabilityDisposition::ProtectiveOnly);
    }

    #[test]
    fn solver_deadline_miss_requires_hold() {
        let subject = subject();
        let profile = profile(&subject);
        let current = envelope(&subject, &profile);
        let mut sample = evidence();
        sample.floating_base_solver_budget_missed = true;
        let result = assess_humanoid_control_capability(
            &subject,
            &profile,
            &policy(&subject),
            sample,
            &current,
        );
        assert_eq!(result.action, HumanoidControlCapabilityAction::Hold);
    }

    #[test]
    fn required_model_loss_requires_hold() {
        let subject = subject();
        let profile = profile(&subject);
        let current = envelope(&subject, &profile);
        let mut sample = evidence();
        sample.floating_base_model_available = false;
        sample.floating_base_dynamics_converged = false;
        let result = assess_humanoid_control_capability(
            &subject,
            &profile,
            &policy(&subject),
            sample,
            &current,
        );
        assert_eq!(result.action, HumanoidControlCapabilityAction::Hold);
    }
}
