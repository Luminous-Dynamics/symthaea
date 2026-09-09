// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Subject-bound command-side evidence for permit-authorized Reach execution.
//!
//! This module deliberately stops short of claiming Reach task success. The
//! Cartesian error available during preparation is a **pre-command** error. A
//! qualification verdict about tracking requires a later post-step observation
//! bound to the exact same target. Until that target binding is carried through
//! the execution result, this layer can only assess whether command generation,
//! dynamics solving, authority composition, and final safety projection met an
//! explicit policy.

use crate::execution::HumanoidAuthorityEnvelope;
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_execution::HumanoidPermittedReachExecutionResult;
use crate::types::HumanoidTask;

pub const HUMANOID_REACH_COMMAND_EVIDENCE_SCHEMA_VERSION: u32 = 1;

/// Immutable evidence receipt for one finalized Reach command.
///
/// This is evidence, not authority. It cannot be used to construct a permit or
/// bypass the continuous executive.
#[derive(Debug, Clone)]
pub struct HumanoidReachCommandEvidence {
    pub schema_version: u32,
    pub subject: HumanoidQualificationSubject,
    pub subject_fingerprint: u64,
    pub validation_epoch: u64,
    pub goal_id: String,

    // Spatial/reference evidence available before the command was applied.
    pub pre_command_position_error_norm_m: f64,
    pub desired_cartesian_speed_mps: f64,
    pub jacobian_confidence: f64,
    pub cartesian_reference_authority_used: f32,

    // Frozen dynamics lineage used by the same preparation cycle.
    pub rigid_model_id: Option<String>,
    pub rigid_sampled_at_s: Option<f64>,
    pub full_model_id: String,
    pub full_sampled_at_s: f64,
    pub floating_model_id: Option<String>,
    pub floating_sampled_at_s: Option<f64>,
    pub full_dynamics_age_s: f64,

    // Existing hierarchy / solver evidence.
    pub whole_body_feasible: bool,
    pub whole_body_objective_residual: f64,
    pub whole_body_joint_utilization: f64,
    pub inverse_dynamics_iterations: usize,
    pub inverse_dynamics_max_violation: f64,
    pub inverse_dynamics_fallback: bool,
    pub contact_dynamics_converged: bool,
    pub contact_dynamics_fallback: bool,
    pub contact_dynamics_residual_nm: f64,
    pub contact_acceleration_residual: f64,
    pub contact_friction_utilization: f64,
    pub contact_solver_budget_missed: bool,
    pub floating_base_model_available: bool,
    pub floating_base_dynamics_converged: bool,
    pub floating_base_dynamics_fallback: bool,
    pub floating_base_dynamics_residual: f64,
    pub floating_base_solver_budget_missed: bool,

    // Final authority and physical projection evidence.
    pub authority: HumanoidAuthorityEnvelope,
    pub authority_scale: f32,
    pub final_safety_rejected: bool,
    pub final_safety_non_finite_values: usize,
    pub final_safety_magnitude_clips: usize,
    pub final_safety_slew_clips: usize,
    pub final_safety_joint_limit_interventions: usize,
    pub final_safety_velocity_interventions: usize,
    pub final_command_effort: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachCommandEvidenceBindFailure {
    InvalidSubject,
    SubjectIsNotReach,
    SubjectFingerprintMismatch,
    ValidationEpochMismatch,
    GoalIdentityMismatch,
    FullDynamicsIdentityMismatch,
    InvalidReceiptNumericEvidence,
    FinalCommandMorphologyMismatch,
}

/// Bind one finalized Reach result to the exact qualification subject it claims
/// to provide command-side evidence for.
pub fn bind_humanoid_reach_command_evidence(
    subject: &HumanoidQualificationSubject,
    result: &HumanoidPermittedReachExecutionResult,
) -> Result<HumanoidReachCommandEvidence, HumanoidReachCommandEvidenceBindFailure> {
    if !subject.validate() {
        return Err(HumanoidReachCommandEvidenceBindFailure::InvalidSubject);
    }
    if subject.task != HumanoidTask::Reach {
        return Err(HumanoidReachCommandEvidenceBindFailure::SubjectIsNotReach);
    }
    let subject_fingerprint = subject.fingerprint();
    let cartesian = &result.preparation.cartesian_reference;
    if subject_fingerprint == 0 || subject_fingerprint != cartesian.subject_fingerprint {
        return Err(HumanoidReachCommandEvidenceBindFailure::SubjectFingerprintMismatch);
    }
    if result.preparation.validation_epoch == 0
        || result.preparation.validation_epoch != cartesian.validation_epoch
    {
        return Err(HumanoidReachCommandEvidenceBindFailure::ValidationEpochMismatch);
    }
    if result.preparation.goal_id.trim().is_empty()
        || result.preparation.goal_id != cartesian.goal_id
    {
        return Err(HumanoidReachCommandEvidenceBindFailure::GoalIdentityMismatch);
    }
    if result.preparation.dynamics.full_model_id != cartesian.dynamics_model_id
        || result.preparation.dynamics.full_sampled_at_s > f64::MAX
    {
        return Err(HumanoidReachCommandEvidenceBindFailure::FullDynamicsIdentityMismatch);
    }
    if result.execution.command.num_actuators() != subject.morphology.num_actuators() {
        return Err(HumanoidReachCommandEvidenceBindFailure::FinalCommandMorphologyMismatch);
    }

    let hierarchy = &result.execution.report.hierarchy;
    let safety = &result.execution.report.safety;
    let finite = [
        cartesian.position_error_norm_m,
        cartesian.desired_cartesian_speed_mps,
        cartesian.jacobian_confidence,
        cartesian.dynamics_age_s,
        hierarchy.whole_body_objective_residual,
        hierarchy.whole_body_joint_utilization,
        hierarchy.inverse_dynamics_max_violation,
        hierarchy.contact_dynamics_residual_nm,
        hierarchy.contact_acceleration_residual,
        hierarchy.contact_friction_utilization,
        hierarchy.floating_base_dynamics_residual,
        result.execution.report.authority_scale as f64,
        result.execution.command.control_effort() as f64,
        result.preparation.dynamics.full_sampled_at_s,
    ]
    .into_iter()
    .all(f64::is_finite);
    if !finite
        || cartesian.position_error_norm_m < 0.0
        || cartesian.desired_cartesian_speed_mps < 0.0
        || !(0.0..=1.0).contains(&cartesian.jacobian_confidence)
        || cartesian.dynamics_age_s < 0.0
        || !(0.0..=1.0).contains(&result.execution.report.authority_scale)
    {
        return Err(HumanoidReachCommandEvidenceBindFailure::InvalidReceiptNumericEvidence);
    }

    Ok(HumanoidReachCommandEvidence {
        schema_version: HUMANOID_REACH_COMMAND_EVIDENCE_SCHEMA_VERSION,
        subject: subject.clone(),
        subject_fingerprint,
        validation_epoch: result.preparation.validation_epoch,
        goal_id: result.preparation.goal_id.clone(),
        pre_command_position_error_norm_m: cartesian.position_error_norm_m,
        desired_cartesian_speed_mps: cartesian.desired_cartesian_speed_mps,
        jacobian_confidence: cartesian.jacobian_confidence,
        cartesian_reference_authority_used: cartesian.maximum_normalized_correction_used,
        rigid_model_id: result.preparation.dynamics.rigid_model_id.clone(),
        rigid_sampled_at_s: result.preparation.dynamics.rigid_sampled_at_s,
        full_model_id: result.preparation.dynamics.full_model_id.clone(),
        full_sampled_at_s: result.preparation.dynamics.full_sampled_at_s,
        floating_model_id: result.preparation.dynamics.floating_model_id.clone(),
        floating_sampled_at_s: result.preparation.dynamics.floating_sampled_at_s,
        full_dynamics_age_s: cartesian.dynamics_age_s,
        whole_body_feasible: hierarchy.whole_body_feasible,
        whole_body_objective_residual: hierarchy.whole_body_objective_residual,
        whole_body_joint_utilization: hierarchy.whole_body_joint_utilization,
        inverse_dynamics_iterations: hierarchy.inverse_dynamics_iterations,
        inverse_dynamics_max_violation: hierarchy.inverse_dynamics_max_violation,
        inverse_dynamics_fallback: hierarchy.inverse_dynamics_fallback,
        contact_dynamics_converged: hierarchy.contact_dynamics_converged,
        contact_dynamics_fallback: hierarchy.contact_dynamics_fallback,
        contact_dynamics_residual_nm: hierarchy.contact_dynamics_residual_nm,
        contact_acceleration_residual: hierarchy.contact_acceleration_residual,
        contact_friction_utilization: hierarchy.contact_friction_utilization,
        contact_solver_budget_missed: hierarchy.contact_solver_budget_missed,
        floating_base_model_available: hierarchy.floating_base_model_available,
        floating_base_dynamics_converged: hierarchy.floating_base_dynamics_converged,
        floating_base_dynamics_fallback: hierarchy.floating_base_dynamics_fallback,
        floating_base_dynamics_residual: hierarchy.floating_base_dynamics_residual,
        floating_base_solver_budget_missed: hierarchy.floating_base_solver_budget_missed,
        authority: result.execution.report.authority,
        authority_scale: result.execution.report.authority_scale,
        final_safety_rejected: safety.rejected,
        final_safety_non_finite_values: safety.non_finite_values,
        final_safety_magnitude_clips: safety.magnitude_clips,
        final_safety_slew_clips: safety.slew_clips,
        final_safety_joint_limit_interventions: safety.joint_limit_interventions,
        final_safety_velocity_interventions: safety.velocity_interventions,
        final_command_effort: result.execution.command.control_effort(),
    })
}

/// Explicit command-generation policy for one Reach qualification campaign.
/// There is intentionally no Default implementation: thresholds and fallback
/// allowances must come from the campaign's declared qualification protocol.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachCommandEvidencePolicy {
    pub policy_id: String,
    pub subject_fingerprint: u64,
    pub maximum_full_dynamics_age_s: f64,
    pub minimum_jacobian_confidence: f64,
    pub minimum_goal_authority_scale: f32,
    pub maximum_whole_body_objective_residual: f64,
    pub maximum_joint_utilization: f64,
    pub maximum_inverse_dynamics_violation: f64,
    pub allow_inverse_dynamics_fallback: bool,
    pub maximum_contact_dynamics_residual_nm: f64,
    pub maximum_contact_acceleration_residual: f64,
    pub maximum_contact_friction_utilization: f64,
    pub allow_contact_dynamics_fallback: bool,
    pub require_floating_base_model: bool,
    pub maximum_floating_base_dynamics_residual: f64,
    pub allow_floating_base_fallback: bool,
    pub maximum_final_safety_interventions: usize,
}

impl HumanoidReachCommandEvidencePolicy {
    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        !self.policy_id.trim().is_empty()
            && self.policy_id == self.policy_id.trim()
            && self.policy_id.len() <= 256
            && subject.validate()
            && subject.task == HumanoidTask::Reach
            && self.subject_fingerprint != 0
            && self.subject_fingerprint == subject.fingerprint()
            && self.maximum_full_dynamics_age_s.is_finite()
            && self.maximum_full_dynamics_age_s > 0.0
            && self.minimum_jacobian_confidence.is_finite()
            && (0.0..=1.0).contains(&self.minimum_jacobian_confidence)
            && self.minimum_goal_authority_scale.is_finite()
            && (0.0..=1.0).contains(&self.minimum_goal_authority_scale)
            && self.maximum_whole_body_objective_residual.is_finite()
            && self.maximum_whole_body_objective_residual >= 0.0
            && self.maximum_joint_utilization.is_finite()
            && self.maximum_joint_utilization >= 0.0
            && self.maximum_inverse_dynamics_violation.is_finite()
            && self.maximum_inverse_dynamics_violation >= 0.0
            && self.maximum_contact_dynamics_residual_nm.is_finite()
            && self.maximum_contact_dynamics_residual_nm >= 0.0
            && self.maximum_contact_acceleration_residual.is_finite()
            && self.maximum_contact_acceleration_residual >= 0.0
            && self.maximum_contact_friction_utilization.is_finite()
            && self.maximum_contact_friction_utilization >= 0.0
            && self.maximum_floating_base_dynamics_residual.is_finite()
            && self.maximum_floating_base_dynamics_residual >= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachCommandEvidenceFailureKind {
    InvalidPolicy,
    SubjectMismatch,
    DynamicsTooOld,
    JacobianConfidenceTooLow,
    GoalAuthorityTooLow,
    WholeBodyInfeasible,
    WholeBodyResidualTooLarge,
    JointUtilizationTooHigh,
    InverseDynamicsViolationTooLarge,
    InverseDynamicsFallbackForbidden,
    ContactDynamicsNotConverged,
    ContactDynamicsResidualTooLarge,
    ContactAccelerationResidualTooLarge,
    ContactFrictionUtilizationTooHigh,
    ContactDynamicsFallbackForbidden,
    ContactSolverBudgetMissed,
    FloatingBaseModelRequired,
    FloatingBaseNotConverged,
    FloatingBaseResidualTooLarge,
    FloatingBaseFallbackForbidden,
    FloatingBaseSolverBudgetMissed,
    FinalSafetyRejected,
    FinalSafetyInterventionsExceeded,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachCommandEvidenceAssessment {
    pub policy_id: String,
    pub subject_fingerprint: u64,
    pub validation_epoch: u64,
    pub goal_id: String,
    /// `true` means only that the command-side execution path met this policy.
    /// It is **not** a Reach task-success or qualification verdict.
    pub command_path_accepted: bool,
    pub failures: Vec<HumanoidReachCommandEvidenceFailureKind>,
}

pub fn assess_humanoid_reach_command_evidence(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachCommandEvidencePolicy,
    evidence: &HumanoidReachCommandEvidence,
) -> HumanoidReachCommandEvidenceAssessment {
    let mut failures = Vec::new();
    if !policy.validate_for(subject) {
        failures.push(HumanoidReachCommandEvidenceFailureKind::InvalidPolicy);
    }
    let fingerprint = subject.fingerprint();
    if fingerprint == 0
        || evidence.subject_fingerprint != fingerprint
        || evidence.subject != *subject
        || policy.subject_fingerprint != fingerprint
    {
        failures.push(HumanoidReachCommandEvidenceFailureKind::SubjectMismatch);
    }
    if evidence.full_dynamics_age_s > policy.maximum_full_dynamics_age_s {
        failures.push(HumanoidReachCommandEvidenceFailureKind::DynamicsTooOld);
    }
    if evidence.jacobian_confidence < policy.minimum_jacobian_confidence {
        failures.push(HumanoidReachCommandEvidenceFailureKind::JacobianConfidenceTooLow);
    }
    if evidence.authority_scale < policy.minimum_goal_authority_scale {
        failures.push(HumanoidReachCommandEvidenceFailureKind::GoalAuthorityTooLow);
    }
    if !evidence.whole_body_feasible {
        failures.push(HumanoidReachCommandEvidenceFailureKind::WholeBodyInfeasible);
    }
    if evidence.whole_body_objective_residual > policy.maximum_whole_body_objective_residual {
        failures.push(HumanoidReachCommandEvidenceFailureKind::WholeBodyResidualTooLarge);
    }
    if evidence.whole_body_joint_utilization > policy.maximum_joint_utilization {
        failures.push(HumanoidReachCommandEvidenceFailureKind::JointUtilizationTooHigh);
    }
    if evidence.inverse_dynamics_max_violation > policy.maximum_inverse_dynamics_violation {
        failures.push(HumanoidReachCommandEvidenceFailureKind::InverseDynamicsViolationTooLarge);
    }
    if evidence.inverse_dynamics_fallback && !policy.allow_inverse_dynamics_fallback {
        failures.push(HumanoidReachCommandEvidenceFailureKind::InverseDynamicsFallbackForbidden);
    }
    if !evidence.contact_dynamics_converged {
        failures.push(HumanoidReachCommandEvidenceFailureKind::ContactDynamicsNotConverged);
    }
    if evidence.contact_dynamics_residual_nm > policy.maximum_contact_dynamics_residual_nm {
        failures.push(HumanoidReachCommandEvidenceFailureKind::ContactDynamicsResidualTooLarge);
    }
    if evidence.contact_acceleration_residual > policy.maximum_contact_acceleration_residual {
        failures.push(HumanoidReachCommandEvidenceFailureKind::ContactAccelerationResidualTooLarge);
    }
    if evidence.contact_friction_utilization > policy.maximum_contact_friction_utilization {
        failures.push(HumanoidReachCommandEvidenceFailureKind::ContactFrictionUtilizationTooHigh);
    }
    if evidence.contact_dynamics_fallback && !policy.allow_contact_dynamics_fallback {
        failures.push(HumanoidReachCommandEvidenceFailureKind::ContactDynamicsFallbackForbidden);
    }
    if evidence.contact_solver_budget_missed {
        failures.push(HumanoidReachCommandEvidenceFailureKind::ContactSolverBudgetMissed);
    }
    if policy.require_floating_base_model && !evidence.floating_base_model_available {
        failures.push(HumanoidReachCommandEvidenceFailureKind::FloatingBaseModelRequired);
    }
    if evidence.floating_base_model_available {
        if !evidence.floating_base_dynamics_converged {
            failures.push(HumanoidReachCommandEvidenceFailureKind::FloatingBaseNotConverged);
        }
        if evidence.floating_base_dynamics_residual > policy.maximum_floating_base_dynamics_residual {
            failures.push(HumanoidReachCommandEvidenceFailureKind::FloatingBaseResidualTooLarge);
        }
        if evidence.floating_base_dynamics_fallback && !policy.allow_floating_base_fallback {
            failures.push(HumanoidReachCommandEvidenceFailureKind::FloatingBaseFallbackForbidden);
        }
        if evidence.floating_base_solver_budget_missed {
            failures.push(HumanoidReachCommandEvidenceFailureKind::FloatingBaseSolverBudgetMissed);
        }
    }
    if evidence.final_safety_rejected || evidence.final_safety_non_finite_values > 0 {
        failures.push(HumanoidReachCommandEvidenceFailureKind::FinalSafetyRejected);
    }
    let safety_interventions = evidence
        .final_safety_magnitude_clips
        .saturating_add(evidence.final_safety_slew_clips)
        .saturating_add(evidence.final_safety_joint_limit_interventions)
        .saturating_add(evidence.final_safety_velocity_interventions);
    if safety_interventions > policy.maximum_final_safety_interventions {
        failures.push(HumanoidReachCommandEvidenceFailureKind::FinalSafetyInterventionsExceeded);
    }

    HumanoidReachCommandEvidenceAssessment {
        policy_id: policy.policy_id.clone(),
        subject_fingerprint: evidence.subject_fingerprint,
        validation_epoch: evidence.validation_epoch,
        goal_id: evidence.goal_id.clone(),
        command_path_accepted: failures.is_empty(),
        failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::HumanoidMorphology;
    use crate::types::ActuationMode;

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Reach,
            ActuationMode::NormalizedTorque,
            "reach-evidence-test-v1",
        )
    }

    fn policy() -> HumanoidReachCommandEvidencePolicy {
        HumanoidReachCommandEvidencePolicy {
            policy_id: "reach-command-policy-test-v1".into(),
            subject_fingerprint: subject().fingerprint(),
            maximum_full_dynamics_age_s: 0.05,
            minimum_jacobian_confidence: 0.8,
            minimum_goal_authority_scale: 0.5,
            maximum_whole_body_objective_residual: 0.2,
            maximum_joint_utilization: 1.0,
            maximum_inverse_dynamics_violation: 0.01,
            allow_inverse_dynamics_fallback: false,
            maximum_contact_dynamics_residual_nm: 0.05,
            maximum_contact_acceleration_residual: 0.05,
            maximum_contact_friction_utilization: 1.0,
            allow_contact_dynamics_fallback: false,
            require_floating_base_model: true,
            maximum_floating_base_dynamics_residual: 0.05,
            allow_floating_base_fallback: false,
            maximum_final_safety_interventions: 0,
        }
    }

    fn evidence() -> HumanoidReachCommandEvidence {
        HumanoidReachCommandEvidence {
            schema_version: HUMANOID_REACH_COMMAND_EVIDENCE_SCHEMA_VERSION,
            subject: subject(),
            subject_fingerprint: subject().fingerprint(),
            validation_epoch: 7,
            goal_id: "cup-7".into(),
            pre_command_position_error_norm_m: 0.1,
            desired_cartesian_speed_mps: 0.2,
            jacobian_confidence: 0.95,
            cartesian_reference_authority_used: 0.2,
            rigid_model_id: Some("rigid-v1".into()),
            rigid_sampled_at_s: Some(1.0),
            full_model_id: "full-v1".into(),
            full_sampled_at_s: 1.0,
            floating_model_id: Some("floating-v1".into()),
            floating_sampled_at_s: Some(1.0),
            full_dynamics_age_s: 0.01,
            whole_body_feasible: true,
            whole_body_objective_residual: 0.01,
            whole_body_joint_utilization: 0.5,
            inverse_dynamics_iterations: 4,
            inverse_dynamics_max_violation: 0.001,
            inverse_dynamics_fallback: false,
            contact_dynamics_converged: true,
            contact_dynamics_fallback: false,
            contact_dynamics_residual_nm: 0.001,
            contact_acceleration_residual: 0.001,
            contact_friction_utilization: 0.5,
            contact_solver_budget_missed: false,
            floating_base_model_available: true,
            floating_base_dynamics_converged: true,
            floating_base_dynamics_fallback: false,
            floating_base_dynamics_residual: 0.001,
            floating_base_solver_budget_missed: false,
            authority: HumanoidAuthorityEnvelope::fully_admitted(),
            authority_scale: 1.0,
            final_safety_rejected: false,
            final_safety_non_finite_values: 0,
            final_safety_magnitude_clips: 0,
            final_safety_slew_clips: 0,
            final_safety_joint_limit_interventions: 0,
            final_safety_velocity_interventions: 0,
            final_command_effort: 0.1,
        }
    }

    #[test]
    fn nominal_command_path_evidence_is_accepted_without_claiming_task_success() {
        let assessment = assess_humanoid_reach_command_evidence(&subject(), &policy(), &evidence());
        assert!(assessment.command_path_accepted);
        assert!(assessment.failures.is_empty());
    }

    #[test]
    fn safety_intervention_can_fail_explicit_policy() {
        let mut evidence = evidence();
        evidence.final_safety_slew_clips = 1;
        let assessment = assess_humanoid_reach_command_evidence(&subject(), &policy(), &evidence);
        assert!(!assessment.command_path_accepted);
        assert!(assessment
            .failures
            .contains(&HumanoidReachCommandEvidenceFailureKind::FinalSafetyInterventionsExceeded));
    }

    #[test]
    fn fallback_is_not_silently_accepted() {
        let mut evidence = evidence();
        evidence.inverse_dynamics_fallback = true;
        let assessment = assess_humanoid_reach_command_evidence(&subject(), &policy(), &evidence);
        assert!(assessment
            .failures
            .contains(&HumanoidReachCommandEvidenceFailureKind::InverseDynamicsFallbackForbidden));
    }
}
