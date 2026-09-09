// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Target-bound post-step evidence for permit-authorized Reach execution.
//!
//! Command-side evidence answers whether the controller/solver/safety path met a
//! declared policy. This module answers a different question: after that command,
//! did a fresh body observation show progress toward the **exact same world-space
//! target observation** that was admitted by the spatial permit?
//!
//! A single accepted step is still not a qualification certificate. It is one
//! evidence case suitable for later aggregation across scenarios, seeds,
//! perturbations, hardware states, and repeated trials.

use crate::morphology::HandSide;
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_execution::HumanoidPermittedReachExecutionResult;
use crate::reach_execution_evidence::{
    HumanoidReachCommandEvidence, HumanoidReachCommandEvidenceAssessment,
    HumanoidReachCommandEvidencePolicy, assess_humanoid_reach_command_evidence,
};
use crate::types::{HumanoidState, HumanoidTask};

pub const HUMANOID_REACH_OUTCOME_EVIDENCE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachOutcomeEvidence {
    pub schema_version: u32,
    pub subject_fingerprint: u64,
    pub validation_epoch: u64,
    pub goal_id: String,
    pub spatial_goal_fingerprint: u64,
    pub hand: HandSide,
    pub target_world_m: [f64; 3],
    pub prepared_at_s: f64,
    pub observed_at_s: f64,
    pub received_at_s: f64,
    pub observation_age_s: f64,
    pub elapsed_since_preparation_s: f64,
    pub hand_world_m: [f64; 3],
    pub pre_command_error_m: f64,
    pub post_command_error_m: f64,
    /// Positive means the hand moved closer to the target.
    pub progress_m: f64,
    /// Progress divided by pre-command error when that error is nonzero.
    pub fractional_progress: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachOutcomeBindFailure {
    InvalidSubject,
    SubjectIsNotReach,
    SubjectMismatch,
    CommandExecutionIdentityMismatch,
    InvalidSpatialGoalIdentity,
    InvalidPreparationTime,
    InvalidPostState,
    ObservationBeforePreparation,
    ObservationTimestampInFuture,
    InvalidReceivedTime,
    InvalidOutcomeNumericEvidence,
}

/// Bind a fresh post-step body state to the exact finalized Reach command and its
/// command-side receipt.
pub fn bind_humanoid_reach_outcome_evidence(
    subject: &HumanoidQualificationSubject,
    command: &HumanoidReachCommandEvidence,
    result: &HumanoidPermittedReachExecutionResult,
    post_state: &HumanoidState,
    received_at_s: f64,
) -> Result<HumanoidReachOutcomeEvidence, HumanoidReachOutcomeBindFailure> {
    if !subject.validate() {
        return Err(HumanoidReachOutcomeBindFailure::InvalidSubject);
    }
    if subject.task != HumanoidTask::Reach {
        return Err(HumanoidReachOutcomeBindFailure::SubjectIsNotReach);
    }
    let fingerprint = subject.fingerprint();
    if fingerprint == 0
        || command.subject_fingerprint != fingerprint
        || command.subject != *subject
    {
        return Err(HumanoidReachOutcomeBindFailure::SubjectMismatch);
    }
    if command.validation_epoch != result.preparation.validation_epoch
        || command.goal_id != result.preparation.goal_id
        || command.full_model_id != result.preparation.dynamics.full_model_id
        || command.authority_scale.to_bits() != result.execution.report.authority_scale.to_bits()
    {
        return Err(HumanoidReachOutcomeBindFailure::CommandExecutionIdentityMismatch);
    }
    if result.preparation.spatial_goal_fingerprint == 0
        || result.preparation.goal_id.trim().is_empty()
        || !result.preparation.target_world_m.iter().all(|value| value.is_finite())
    {
        return Err(HumanoidReachOutcomeBindFailure::InvalidSpatialGoalIdentity);
    }
    if !result.preparation.prepared_at_s.is_finite() || result.preparation.prepared_at_s < 0.0 {
        return Err(HumanoidReachOutcomeBindFailure::InvalidPreparationTime);
    }
    if post_state.validate_for(subject.morphology).is_err() || !post_state.timestamp.is_finite() {
        return Err(HumanoidReachOutcomeBindFailure::InvalidPostState);
    }
    if post_state.timestamp < result.preparation.prepared_at_s {
        return Err(HumanoidReachOutcomeBindFailure::ObservationBeforePreparation);
    }
    if !received_at_s.is_finite() || received_at_s < 0.0 || received_at_s < post_state.timestamp {
        return Err(HumanoidReachOutcomeBindFailure::InvalidReceivedTime);
    }
    if post_state.timestamp > received_at_s {
        return Err(HumanoidReachOutcomeBindFailure::ObservationTimestampInFuture);
    }

    let hand_world_m = hand_world_position(post_state, result.preparation.hand)
        .ok_or(HumanoidReachOutcomeBindFailure::InvalidPostState)?;
    let post_error = norm3(sub3(result.preparation.target_world_m, hand_world_m));
    let pre_error = command.pre_command_position_error_norm_m;
    let progress_m = pre_error - post_error;
    let fractional_progress = if pre_error > 1.0e-12 {
        progress_m / pre_error
    } else if post_error <= 1.0e-12 {
        1.0
    } else {
        f64::NEG_INFINITY
    };
    let observation_age_s = received_at_s - post_state.timestamp;
    let elapsed_since_preparation_s = post_state.timestamp - result.preparation.prepared_at_s;

    if ![
        pre_error,
        post_error,
        progress_m,
        fractional_progress,
        observation_age_s,
        elapsed_since_preparation_s,
    ]
    .into_iter()
    .all(f64::is_finite)
        || pre_error < 0.0
        || post_error < 0.0
        || observation_age_s < 0.0
        || elapsed_since_preparation_s < 0.0
    {
        return Err(HumanoidReachOutcomeBindFailure::InvalidOutcomeNumericEvidence);
    }

    Ok(HumanoidReachOutcomeEvidence {
        schema_version: HUMANOID_REACH_OUTCOME_EVIDENCE_SCHEMA_VERSION,
        subject_fingerprint: fingerprint,
        validation_epoch: command.validation_epoch,
        goal_id: command.goal_id.clone(),
        spatial_goal_fingerprint: result.preparation.spatial_goal_fingerprint,
        hand: result.preparation.hand,
        target_world_m: result.preparation.target_world_m,
        prepared_at_s: result.preparation.prepared_at_s,
        observed_at_s: post_state.timestamp,
        received_at_s,
        observation_age_s,
        elapsed_since_preparation_s,
        hand_world_m,
        pre_command_error_m: pre_error,
        post_command_error_m: post_error,
        progress_m,
        fractional_progress,
    })
}

/// Explicit outcome policy for one Reach qualification campaign. There is no
/// Default implementation; thresholds must be declared by the campaign.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachOutcomeEvidencePolicy {
    pub policy_id: String,
    pub subject_fingerprint: u64,
    pub maximum_observation_age_s: f64,
    pub maximum_elapsed_since_preparation_s: f64,
    pub maximum_post_command_error_m: f64,
    pub minimum_progress_m: f64,
    pub minimum_fractional_progress: f64,
    /// If the hand was already inside final tolerance, do not require artificial
    /// movement simply to satisfy a progress floor.
    pub allow_already_within_tolerance: bool,
}

impl HumanoidReachOutcomeEvidencePolicy {
    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        !self.policy_id.trim().is_empty()
            && self.policy_id == self.policy_id.trim()
            && self.policy_id.len() <= 256
            && subject.validate()
            && subject.task == HumanoidTask::Reach
            && self.subject_fingerprint != 0
            && self.subject_fingerprint == subject.fingerprint()
            && self.maximum_observation_age_s.is_finite()
            && self.maximum_observation_age_s >= 0.0
            && self.maximum_elapsed_since_preparation_s.is_finite()
            && self.maximum_elapsed_since_preparation_s > 0.0
            && self.maximum_post_command_error_m.is_finite()
            && self.maximum_post_command_error_m >= 0.0
            && self.minimum_progress_m.is_finite()
            && self.minimum_progress_m >= 0.0
            && self.minimum_fractional_progress.is_finite()
            && (0.0..=1.0).contains(&self.minimum_fractional_progress)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachOutcomeFailureKind {
    InvalidPolicy,
    InvalidEvidenceSchema,
    SubjectMismatch,
    ObservationTooOld,
    ObservationTooLate,
    FinalErrorTooLarge,
    ProgressTooSmall,
    FractionalProgressTooSmall,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachOutcomeEvidenceAssessment {
    pub policy_id: String,
    pub subject_fingerprint: u64,
    pub validation_epoch: u64,
    pub goal_id: String,
    pub spatial_goal_fingerprint: u64,
    /// Accepted means this single post-step observation met the declared outcome
    /// policy; it is not a campaign qualification verdict.
    pub outcome_accepted: bool,
    pub failures: Vec<HumanoidReachOutcomeFailureKind>,
}

pub fn assess_humanoid_reach_outcome_evidence(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachOutcomeEvidencePolicy,
    evidence: &HumanoidReachOutcomeEvidence,
) -> HumanoidReachOutcomeEvidenceAssessment {
    let mut failures = Vec::new();
    if !policy.validate_for(subject) {
        failures.push(HumanoidReachOutcomeFailureKind::InvalidPolicy);
    }
    if evidence.schema_version != HUMANOID_REACH_OUTCOME_EVIDENCE_SCHEMA_VERSION {
        failures.push(HumanoidReachOutcomeFailureKind::InvalidEvidenceSchema);
    }
    let fingerprint = subject.fingerprint();
    if fingerprint == 0
        || evidence.subject_fingerprint != fingerprint
        || policy.subject_fingerprint != fingerprint
    {
        failures.push(HumanoidReachOutcomeFailureKind::SubjectMismatch);
    }
    if evidence.observation_age_s > policy.maximum_observation_age_s {
        failures.push(HumanoidReachOutcomeFailureKind::ObservationTooOld);
    }
    if evidence.elapsed_since_preparation_s > policy.maximum_elapsed_since_preparation_s {
        failures.push(HumanoidReachOutcomeFailureKind::ObservationTooLate);
    }
    if evidence.post_command_error_m > policy.maximum_post_command_error_m {
        failures.push(HumanoidReachOutcomeFailureKind::FinalErrorTooLarge);
    }

    let already_inside = evidence.pre_command_error_m <= policy.maximum_post_command_error_m;
    if !(policy.allow_already_within_tolerance && already_inside) {
        if evidence.progress_m < policy.minimum_progress_m {
            failures.push(HumanoidReachOutcomeFailureKind::ProgressTooSmall);
        }
        if evidence.fractional_progress < policy.minimum_fractional_progress {
            failures.push(HumanoidReachOutcomeFailureKind::FractionalProgressTooSmall);
        }
    }

    HumanoidReachOutcomeEvidenceAssessment {
        policy_id: policy.policy_id.clone(),
        subject_fingerprint: evidence.subject_fingerprint,
        validation_epoch: evidence.validation_epoch,
        goal_id: evidence.goal_id.clone(),
        spatial_goal_fingerprint: evidence.spatial_goal_fingerprint,
        outcome_accepted: failures.is_empty(),
        failures,
    }
}

/// One fully bound single-step Reach evidence case. This combines command-path
/// and post-step outcome assessment while preserving their separate verdicts.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachStepEvidenceAssessment {
    pub subject_fingerprint: u64,
    pub validation_epoch: u64,
    pub goal_id: String,
    pub spatial_goal_fingerprint: u64,
    pub command: HumanoidReachCommandEvidenceAssessment,
    pub outcome: HumanoidReachOutcomeEvidenceAssessment,
    /// One step is accepted only when both independent sides accepted it.
    pub step_accepted: bool,
}

/// Build and assess one target-bound Reach step without allowing callers to mix a
/// command assessment from one execution with outcome evidence from another.
#[allow(clippy::too_many_arguments)]
pub fn assess_humanoid_reach_step(
    subject: &HumanoidQualificationSubject,
    command_policy: &HumanoidReachCommandEvidencePolicy,
    outcome_policy: &HumanoidReachOutcomeEvidencePolicy,
    command_evidence: &HumanoidReachCommandEvidence,
    result: &HumanoidPermittedReachExecutionResult,
    post_state: &HumanoidState,
    received_at_s: f64,
) -> Result<HumanoidReachStepEvidenceAssessment, HumanoidReachOutcomeBindFailure> {
    let command = assess_humanoid_reach_command_evidence(subject, command_policy, command_evidence);
    let outcome_evidence = bind_humanoid_reach_outcome_evidence(
        subject,
        command_evidence,
        result,
        post_state,
        received_at_s,
    )?;
    let outcome = assess_humanoid_reach_outcome_evidence(subject, outcome_policy, &outcome_evidence);
    let step_accepted = command.command_path_accepted && outcome.outcome_accepted;
    Ok(HumanoidReachStepEvidenceAssessment {
        subject_fingerprint: outcome.subject_fingerprint,
        validation_epoch: outcome.validation_epoch,
        goal_id: outcome.goal_id.clone(),
        spatial_goal_fingerprint: outcome.spatial_goal_fingerprint,
        command,
        outcome,
        step_accepted,
    })
}

fn hand_world_position(state: &HumanoidState, hand: HandSide) -> Option<[f64; 3]> {
    let start = match hand {
        HandSide::Right => 0,
        HandSide::Left => 3,
    };
    let values = state.extremities.get(start..start + 3)?;
    values.iter().all(|value| value.is_finite()).then_some([
        values[0], values[1], values[2],
    ])
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
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
            "reach-outcome-test-v1",
        )
    }

    fn outcome_policy() -> HumanoidReachOutcomeEvidencePolicy {
        HumanoidReachOutcomeEvidencePolicy {
            policy_id: "reach-outcome-policy-test-v1".into(),
            subject_fingerprint: subject().fingerprint(),
            maximum_observation_age_s: 0.02,
            maximum_elapsed_since_preparation_s: 0.1,
            maximum_post_command_error_m: 0.05,
            minimum_progress_m: 0.01,
            minimum_fractional_progress: 0.2,
            allow_already_within_tolerance: true,
        }
    }

    fn evidence() -> HumanoidReachOutcomeEvidence {
        HumanoidReachOutcomeEvidence {
            schema_version: HUMANOID_REACH_OUTCOME_EVIDENCE_SCHEMA_VERSION,
            subject_fingerprint: subject().fingerprint(),
            validation_epoch: 9,
            goal_id: "cup-9".into(),
            spatial_goal_fingerprint: 99,
            hand: HandSide::Right,
            target_world_m: [0.4, -0.2, 1.0],
            prepared_at_s: 1.0,
            observed_at_s: 1.02,
            received_at_s: 1.025,
            observation_age_s: 0.005,
            elapsed_since_preparation_s: 0.02,
            hand_world_m: [0.36, -0.2, 1.0],
            pre_command_error_m: 0.08,
            post_command_error_m: 0.04,
            progress_m: 0.04,
            fractional_progress: 0.5,
        }
    }

    #[test]
    fn target_bound_progress_can_pass_single_step_policy() {
        let assessment =
            assess_humanoid_reach_outcome_evidence(&subject(), &outcome_policy(), &evidence());
        assert!(assessment.outcome_accepted);
    }

    #[test]
    fn moving_away_from_target_fails_progress_policy() {
        let mut evidence = evidence();
        evidence.post_command_error_m = 0.10;
        evidence.progress_m = -0.02;
        evidence.fractional_progress = -0.25;
        let assessment =
            assess_humanoid_reach_outcome_evidence(&subject(), &outcome_policy(), &evidence);
        assert!(!assessment.outcome_accepted);
        assert!(assessment.failures.contains(&HumanoidReachOutcomeFailureKind::ProgressTooSmall));
    }

    #[test]
    fn already_inside_tolerance_does_not_require_motion_for_its_own_sake() {
        let mut evidence = evidence();
        evidence.pre_command_error_m = 0.02;
        evidence.post_command_error_m = 0.02;
        evidence.progress_m = 0.0;
        evidence.fractional_progress = 0.0;
        let assessment =
            assess_humanoid_reach_outcome_evidence(&subject(), &outcome_policy(), &evidence);
        assert!(assessment.outcome_accepted);
    }
}
