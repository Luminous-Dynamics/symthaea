// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Episode-level evidence for permit-bound Reach.
//!
//! A good controller step is not the same claim as a completed Reach. This layer
//! binds repeated fresh validation/authority cycles into one target-stable episode
//! and requires final target tolerance, bounded target drift, bounded duration,
//! chronological execution, and explicit evidence-policy identity.
//!
//! This is engineering qualification evidence, not legal/product-safety
//! certification and not actuator authority.

use std::collections::BTreeSet;

use crate::execution_authority_scope::{
    HumanoidExecutionPurpose, HumanoidQualificationAuthorityBasis,
};
use crate::morphology::HandSide;
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_execution::HumanoidPermittedReachExecutionResult;
use crate::reach_execution_evidence::{
    HumanoidReachCommandEvidence, HumanoidReachCommandEvidencePolicy,
    assess_humanoid_reach_command_evidence,
};
use crate::reach_outcome_evidence::{
    HumanoidReachOutcomeBindFailure, HumanoidReachOutcomeEvidencePolicy,
    assess_humanoid_reach_outcome_evidence, bind_humanoid_reach_outcome_evidence,
};
use crate::reach_policy_identity::{
    humanoid_reach_command_policy_fingerprint, humanoid_reach_outcome_policy_fingerprint,
};
use crate::types::{HumanoidState, HumanoidTask};

pub const HUMANOID_REACH_EPISODE_EVIDENCE_SCHEMA_VERSION: u32 = 1;

/// One completed control step suitable for episode aggregation.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachEpisodeStepEvidence {
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
    pub pre_error_m: f64,
    pub post_error_m: f64,
    pub progress_m: f64,
    pub command_policy_id: String,
    pub outcome_policy_id: String,
    pub command_policy_fingerprint: u64,
    pub outcome_policy_fingerprint: u64,
    pub authority_receipt_fingerprint: u64,
    pub authority_scope_fingerprint: u64,
    pub authority_scope_id: String,
    pub execution_purpose: HumanoidExecutionPurpose,
    pub qualification_basis: HumanoidQualificationAuthorityBasis,
    pub step_accepted: bool,
    pub step_fingerprint: u64,
}

impl HumanoidReachEpisodeStepEvidence {
    pub fn validate(&self) -> bool {
        self.schema_version == HUMANOID_REACH_EPISODE_EVIDENCE_SCHEMA_VERSION
            && self.subject_fingerprint != 0
            && self.validation_epoch != 0
            && valid_id(&self.goal_id)
            && self.spatial_goal_fingerprint != 0
            && self.target_world_m.iter().all(|value| value.is_finite())
            && self.prepared_at_s.is_finite()
            && self.observed_at_s.is_finite()
            && self.received_at_s.is_finite()
            && self.prepared_at_s >= 0.0
            && self.observed_at_s >= self.prepared_at_s
            && self.received_at_s >= self.observed_at_s
            && self.pre_error_m.is_finite()
            && self.pre_error_m >= 0.0
            && self.post_error_m.is_finite()
            && self.post_error_m >= 0.0
            && self.progress_m.is_finite()
            && approximately_equal(self.progress_m, self.pre_error_m - self.post_error_m)
            && valid_id(&self.command_policy_id)
            && valid_id(&self.outcome_policy_id)
            && self.command_policy_fingerprint != 0
            && self.outcome_policy_fingerprint != 0
            && self.authority_receipt_fingerprint != 0
            && self.authority_scope_fingerprint != 0
            && valid_id(&self.authority_scope_id)
            && self.execution_purpose.is_qualification()
            && self.qualification_basis == HumanoidQualificationAuthorityBasis::TrialProtocol
            && self.step_fingerprint != 0
            && self.step_fingerprint == fingerprint_episode_step(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachEpisodeStepBindFailure {
    InvalidSubject,
    SubjectIsNotReach,
    InvalidCommandPolicy,
    InvalidOutcomePolicy,
    PolicyFingerprintInvalid,
    CommandIdentityMismatch,
    NonQualificationAuthorityScope,
    InvalidAuthorityLineage,
    Outcome(HumanoidReachOutcomeBindFailure),
    InvalidStepEvidence,
}

/// Bind one actual post-step observation to one finalized Reach execution while
/// retaining enough evidence for later episode-level continuity checks.
#[allow(clippy::too_many_arguments)]
pub fn bind_humanoid_reach_episode_step(
    subject: &HumanoidQualificationSubject,
    command_policy: &HumanoidReachCommandEvidencePolicy,
    outcome_policy: &HumanoidReachOutcomeEvidencePolicy,
    command_evidence: &HumanoidReachCommandEvidence,
    result: &HumanoidPermittedReachExecutionResult,
    post_state: &HumanoidState,
    received_at_s: f64,
) -> Result<HumanoidReachEpisodeStepEvidence, HumanoidReachEpisodeStepBindFailure> {
    if !subject.validate() {
        return Err(HumanoidReachEpisodeStepBindFailure::InvalidSubject);
    }
    if subject.task != HumanoidTask::Reach {
        return Err(HumanoidReachEpisodeStepBindFailure::SubjectIsNotReach);
    }
    if !command_policy.validate_for(subject) {
        return Err(HumanoidReachEpisodeStepBindFailure::InvalidCommandPolicy);
    }
    if !outcome_policy.validate_for(subject) {
        return Err(HumanoidReachEpisodeStepBindFailure::InvalidOutcomePolicy);
    }

    let command_policy_fingerprint = humanoid_reach_command_policy_fingerprint(command_policy);
    let outcome_policy_fingerprint = humanoid_reach_outcome_policy_fingerprint(outcome_policy);
    if command_policy_fingerprint == 0 || outcome_policy_fingerprint == 0 {
        return Err(HumanoidReachEpisodeStepBindFailure::PolicyFingerprintInvalid);
    }

    let subject_fingerprint = subject.fingerprint();
    if command_evidence.subject_fingerprint != subject_fingerprint
        || command_evidence.validation_epoch != result.preparation.validation_epoch
        || command_evidence.goal_id != result.preparation.goal_id
        || command_evidence.authority_scale.to_bits()
            != result.execution.report.authority_scale.to_bits()
    {
        return Err(HumanoidReachEpisodeStepBindFailure::CommandIdentityMismatch);
    }

    let authority = &result.authority_receipt;
    if !authority.execution_purpose.is_qualification()
        || authority.qualification_basis != HumanoidQualificationAuthorityBasis::TrialProtocol
    {
        return Err(HumanoidReachEpisodeStepBindFailure::NonQualificationAuthorityScope);
    }
    if authority.validation_epoch != result.preparation.validation_epoch
        || authority.requirement_subject_fingerprints.as_slice() != [subject_fingerprint]
        || authority.receipt_fingerprint == 0
        || authority.scope_fingerprint == 0
        || !valid_id(&authority.scope_id)
    {
        return Err(HumanoidReachEpisodeStepBindFailure::InvalidAuthorityLineage);
    }

    let command = assess_humanoid_reach_command_evidence(subject, command_policy, command_evidence);
    let outcome = bind_humanoid_reach_outcome_evidence(
        subject,
        command_evidence,
        result,
        post_state,
        received_at_s,
    )
    .map_err(HumanoidReachEpisodeStepBindFailure::Outcome)?;
    let outcome_assessment = assess_humanoid_reach_outcome_evidence(subject, outcome_policy, &outcome);

    let mut step = HumanoidReachEpisodeStepEvidence {
        schema_version: HUMANOID_REACH_EPISODE_EVIDENCE_SCHEMA_VERSION,
        subject_fingerprint,
        validation_epoch: result.preparation.validation_epoch,
        goal_id: result.preparation.goal_id.clone(),
        spatial_goal_fingerprint: result.preparation.spatial_goal_fingerprint,
        hand: result.preparation.hand,
        target_world_m: result.preparation.target_world_m,
        prepared_at_s: result.preparation.prepared_at_s,
        observed_at_s: outcome.observed_at_s,
        received_at_s: outcome.received_at_s,
        pre_error_m: outcome.pre_command_error_m,
        post_error_m: outcome.post_command_error_m,
        progress_m: outcome.progress_m,
        command_policy_id: command.policy_id.clone(),
        outcome_policy_id: outcome_assessment.policy_id.clone(),
        command_policy_fingerprint,
        outcome_policy_fingerprint,
        authority_receipt_fingerprint: authority.receipt_fingerprint,
        authority_scope_fingerprint: authority.scope_fingerprint,
        authority_scope_id: authority.scope_id.clone(),
        execution_purpose: authority.execution_purpose,
        qualification_basis: authority.qualification_basis,
        step_accepted: command.command_path_accepted && outcome_assessment.outcome_accepted,
        step_fingerprint: 0,
    };
    step.step_fingerprint = fingerprint_episode_step(&step);
    if !step.validate() {
        return Err(HumanoidReachEpisodeStepBindFailure::InvalidStepEvidence);
    }
    Ok(step)
}

/// Episode semantics for one exact qualification subject and evidence-policy pair.
/// There is intentionally no Default implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachEpisodePolicy {
    pub schema_version: u32,
    pub policy_id: String,
    pub subject_fingerprint: u64,
    pub required_execution_purpose: HumanoidExecutionPurpose,
    pub required_authority_scope_id: String,
    pub command_policy_fingerprint: u64,
    pub outcome_policy_fingerprint: u64,
    pub maximum_steps: usize,
    pub maximum_episode_duration_s: f64,
    pub maximum_inter_step_gap_s: f64,
    pub maximum_target_drift_m: f64,
    pub maximum_final_error_m: f64,
    pub minimum_net_progress_m: f64,
    pub minimum_distinct_authority_receipts: usize,
    pub require_every_step_accepted: bool,
}

impl HumanoidReachEpisodePolicy {
    pub fn from_exact_policies(
        subject: &HumanoidQualificationSubject,
        policy_id: impl Into<String>,
        required_execution_purpose: HumanoidExecutionPurpose,
        required_authority_scope_id: impl Into<String>,
        command_policy: &HumanoidReachCommandEvidencePolicy,
        outcome_policy: &HumanoidReachOutcomeEvidencePolicy,
        maximum_steps: usize,
        maximum_episode_duration_s: f64,
        maximum_inter_step_gap_s: f64,
        maximum_target_drift_m: f64,
        maximum_final_error_m: f64,
        minimum_net_progress_m: f64,
        minimum_distinct_authority_receipts: usize,
        require_every_step_accepted: bool,
    ) -> Option<Self> {
        if !command_policy.validate_for(subject) || !outcome_policy.validate_for(subject) {
            return None;
        }
        let policy = Self {
            schema_version: HUMANOID_REACH_EPISODE_EVIDENCE_SCHEMA_VERSION,
            policy_id: policy_id.into(),
            subject_fingerprint: subject.fingerprint(),
            required_execution_purpose,
            required_authority_scope_id: required_authority_scope_id.into(),
            command_policy_fingerprint: humanoid_reach_command_policy_fingerprint(command_policy),
            outcome_policy_fingerprint: humanoid_reach_outcome_policy_fingerprint(outcome_policy),
            maximum_steps,
            maximum_episode_duration_s,
            maximum_inter_step_gap_s,
            maximum_target_drift_m,
            maximum_final_error_m,
            minimum_net_progress_m,
            minimum_distinct_authority_receipts,
            require_every_step_accepted,
        };
        policy.validate_for(subject).then_some(policy)
    }

    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.schema_version == HUMANOID_REACH_EPISODE_EVIDENCE_SCHEMA_VERSION
            && valid_id(&self.policy_id)
            && subject.validate()
            && subject.task == HumanoidTask::Reach
            && self.subject_fingerprint != 0
            && self.subject_fingerprint == subject.fingerprint()
            && self.required_execution_purpose.is_qualification()
            && valid_id(&self.required_authority_scope_id)
            && self.command_policy_fingerprint != 0
            && self.outcome_policy_fingerprint != 0
            && self.maximum_steps > 0
            && self.maximum_episode_duration_s.is_finite()
            && self.maximum_episode_duration_s > 0.0
            && self.maximum_inter_step_gap_s.is_finite()
            && self.maximum_inter_step_gap_s >= 0.0
            && self.maximum_target_drift_m.is_finite()
            && self.maximum_target_drift_m >= 0.0
            && self.maximum_final_error_m.is_finite()
            && self.maximum_final_error_m >= 0.0
            && self.minimum_net_progress_m.is_finite()
            && self.minimum_net_progress_m >= 0.0
            && self.minimum_distinct_authority_receipts > 0
            && self.minimum_distinct_authority_receipts <= self.maximum_steps
    }

    pub fn fingerprint(&self) -> u64 {
        if self.schema_version != HUMANOID_REACH_EPISODE_EVIDENCE_SCHEMA_VERSION {
            return 0;
        }
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        feed_u64(&mut hash, self.schema_version as u64);
        feed_bytes(&mut hash, self.policy_id.as_bytes());
        feed_u64(&mut hash, self.subject_fingerprint);
        feed_u64(&mut hash, purpose_id(self.required_execution_purpose));
        feed_bytes(&mut hash, self.required_authority_scope_id.as_bytes());
        feed_u64(&mut hash, self.command_policy_fingerprint);
        feed_u64(&mut hash, self.outcome_policy_fingerprint);
        feed_u64(&mut hash, self.maximum_steps as u64);
        feed_u64(&mut hash, self.maximum_episode_duration_s.to_bits());
        feed_u64(&mut hash, self.maximum_inter_step_gap_s.to_bits());
        feed_u64(&mut hash, self.maximum_target_drift_m.to_bits());
        feed_u64(&mut hash, self.maximum_final_error_m.to_bits());
        feed_u64(&mut hash, self.minimum_net_progress_m.to_bits());
        feed_u64(&mut hash, self.minimum_distinct_authority_receipts as u64);
        feed_u64(&mut hash, self.require_every_step_accepted as u64);
        if self.validate_shape() && hash != 0 { hash } else { 0 }
    }

    fn validate_shape(&self) -> bool {
        valid_id(&self.policy_id)
            && self.subject_fingerprint != 0
            && self.required_execution_purpose.is_qualification()
            && valid_id(&self.required_authority_scope_id)
            && self.command_policy_fingerprint != 0
            && self.outcome_policy_fingerprint != 0
            && self.maximum_steps > 0
            && self.maximum_episode_duration_s.is_finite()
            && self.maximum_episode_duration_s > 0.0
            && self.maximum_inter_step_gap_s.is_finite()
            && self.maximum_inter_step_gap_s >= 0.0
            && self.maximum_target_drift_m.is_finite()
            && self.maximum_target_drift_m >= 0.0
            && self.maximum_final_error_m.is_finite()
            && self.maximum_final_error_m >= 0.0
            && self.minimum_net_progress_m.is_finite()
            && self.minimum_net_progress_m >= 0.0
            && self.minimum_distinct_authority_receipts > 0
            && self.minimum_distinct_authority_receipts <= self.maximum_steps
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HumanoidReachEpisodeFailureKind {
    InvalidPolicy,
    EmptyEpisode,
    TooManySteps,
    InvalidStep,
    SubjectMismatch,
    GoalIdentityMismatch,
    HandMismatch,
    ExecutionPurposeMismatch,
    AuthorityScopeMismatch,
    EvidencePolicyMismatch,
    ValidationEpochNotIncreasing,
    TimeOrderInvalid,
    InterStepGapTooLarge,
    TargetDriftTooLarge,
    RejectedStep,
    InsufficientDistinctAuthorityReceipts,
    EpisodeDurationTooLong,
    FinalErrorTooLarge,
    NetProgressTooSmall,
    InvalidEpisodeFingerprint,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachEpisodeAssessment {
    pub schema_version: u32,
    pub policy_id: String,
    pub policy_fingerprint: u64,
    pub subject_fingerprint: u64,
    pub goal_id: String,
    pub hand: Option<HandSide>,
    pub step_count: usize,
    pub distinct_authority_receipts: usize,
    pub started_at_s: f64,
    pub completed_at_s: f64,
    pub duration_s: f64,
    pub initial_error_m: f64,
    pub final_error_m: f64,
    pub net_progress_m: f64,
    pub maximum_target_drift_m: f64,
    pub episode_fingerprint: u64,
    pub episode_accepted: bool,
    pub failures: Vec<HumanoidReachEpisodeFailureKind>,
}

pub fn assess_humanoid_reach_episode(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachEpisodePolicy,
    steps: &[HumanoidReachEpisodeStepEvidence],
) -> HumanoidReachEpisodeAssessment {
    let mut failures = Vec::new();
    if !policy.validate_for(subject) || policy.fingerprint() == 0 {
        failures.push(HumanoidReachEpisodeFailureKind::InvalidPolicy);
    }
    if steps.is_empty() {
        failures.push(HumanoidReachEpisodeFailureKind::EmptyEpisode);
    }
    if steps.len() > policy.maximum_steps {
        failures.push(HumanoidReachEpisodeFailureKind::TooManySteps);
    }

    let subject_fingerprint = subject.fingerprint();
    let first = steps.first();
    let goal_id = first.map(|step| step.goal_id.clone()).unwrap_or_default();
    let hand = first.map(|step| step.hand);
    let first_target = first.map(|step| step.target_world_m).unwrap_or([0.0; 3]);

    let mut maximum_target_drift_m = 0.0f64;
    let mut previous: Option<&HumanoidReachEpisodeStepEvidence> = None;
    for step in steps {
        if !step.validate() {
            failures.push(HumanoidReachEpisodeFailureKind::InvalidStep);
        }
        if step.subject_fingerprint != subject_fingerprint {
            failures.push(HumanoidReachEpisodeFailureKind::SubjectMismatch);
        }
        if step.goal_id != goal_id {
            failures.push(HumanoidReachEpisodeFailureKind::GoalIdentityMismatch);
        }
        if hand.is_some_and(|expected| step.hand != expected) {
            failures.push(HumanoidReachEpisodeFailureKind::HandMismatch);
        }
        if step.execution_purpose != policy.required_execution_purpose {
            failures.push(HumanoidReachEpisodeFailureKind::ExecutionPurposeMismatch);
        }
        if step.qualification_basis != HumanoidQualificationAuthorityBasis::TrialProtocol
            || step.authority_scope_id != policy.required_authority_scope_id
        {
            failures.push(HumanoidReachEpisodeFailureKind::AuthorityScopeMismatch);
        }
        if step.command_policy_fingerprint != policy.command_policy_fingerprint
            || step.outcome_policy_fingerprint != policy.outcome_policy_fingerprint
        {
            failures.push(HumanoidReachEpisodeFailureKind::EvidencePolicyMismatch);
        }
        if policy.require_every_step_accepted && !step.step_accepted {
            failures.push(HumanoidReachEpisodeFailureKind::RejectedStep);
        }

        maximum_target_drift_m = maximum_target_drift_m.max(norm3(sub3(step.target_world_m, first_target)));
        if maximum_target_drift_m > policy.maximum_target_drift_m {
            failures.push(HumanoidReachEpisodeFailureKind::TargetDriftTooLarge);
        }

        if let Some(previous) = previous {
            if step.validation_epoch <= previous.validation_epoch {
                failures.push(HumanoidReachEpisodeFailureKind::ValidationEpochNotIncreasing);
            }
            if step.prepared_at_s < previous.observed_at_s
                || step.observed_at_s < step.prepared_at_s
                || step.received_at_s < step.observed_at_s
            {
                failures.push(HumanoidReachEpisodeFailureKind::TimeOrderInvalid);
            } else if step.prepared_at_s - previous.observed_at_s > policy.maximum_inter_step_gap_s {
                failures.push(HumanoidReachEpisodeFailureKind::InterStepGapTooLarge);
            }
        }
        previous = Some(step);
    }

    let distinct_authority_receipts = steps
        .iter()
        .map(|step| step.authority_receipt_fingerprint)
        .collect::<BTreeSet<_>>()
        .len();
    if distinct_authority_receipts < policy.minimum_distinct_authority_receipts {
        failures.push(HumanoidReachEpisodeFailureKind::InsufficientDistinctAuthorityReceipts);
    }

    let started_at_s = first.map(|step| step.prepared_at_s).unwrap_or(0.0);
    let completed_at_s = steps.last().map(|step| step.observed_at_s).unwrap_or(0.0);
    let duration_s = completed_at_s - started_at_s;
    if !duration_s.is_finite() || duration_s < 0.0 {
        failures.push(HumanoidReachEpisodeFailureKind::TimeOrderInvalid);
    } else if duration_s > policy.maximum_episode_duration_s {
        failures.push(HumanoidReachEpisodeFailureKind::EpisodeDurationTooLong);
    }

    let initial_error_m = first.map(|step| step.pre_error_m).unwrap_or(f64::INFINITY);
    let final_error_m = steps.last().map(|step| step.post_error_m).unwrap_or(f64::INFINITY);
    let net_progress_m = initial_error_m - final_error_m;
    if !final_error_m.is_finite() || final_error_m > policy.maximum_final_error_m {
        failures.push(HumanoidReachEpisodeFailureKind::FinalErrorTooLarge);
    }
    let initially_outside_tolerance = initial_error_m.is_finite()
        && initial_error_m > policy.maximum_final_error_m;
    if initially_outside_tolerance
        && (!net_progress_m.is_finite() || net_progress_m < policy.minimum_net_progress_m)
    {
        failures.push(HumanoidReachEpisodeFailureKind::NetProgressTooSmall);
    }

    failures.sort();
    failures.dedup();
    let policy_fingerprint = policy.fingerprint();
    let episode_fingerprint = fingerprint_episode(policy_fingerprint, steps);
    if episode_fingerprint == 0 {
        failures.push(HumanoidReachEpisodeFailureKind::InvalidEpisodeFingerprint);
    }

    HumanoidReachEpisodeAssessment {
        schema_version: HUMANOID_REACH_EPISODE_EVIDENCE_SCHEMA_VERSION,
        policy_id: policy.policy_id.clone(),
        policy_fingerprint,
        subject_fingerprint,
        goal_id,
        hand,
        step_count: steps.len(),
        distinct_authority_receipts,
        started_at_s,
        completed_at_s,
        duration_s,
        initial_error_m,
        final_error_m,
        net_progress_m,
        maximum_target_drift_m,
        episode_fingerprint,
        episode_accepted: failures.is_empty(),
        failures,
    }
}

fn fingerprint_episode_step(step: &HumanoidReachEpisodeStepEvidence) -> u64 {
    if step.subject_fingerprint == 0
        || step.validation_epoch == 0
        || !valid_id(&step.goal_id)
        || step.spatial_goal_fingerprint == 0
        || !valid_id(&step.command_policy_id)
        || !valid_id(&step.outcome_policy_id)
        || step.command_policy_fingerprint == 0
        || step.outcome_policy_fingerprint == 0
        || step.authority_receipt_fingerprint == 0
        || step.authority_scope_fingerprint == 0
        || !valid_id(&step.authority_scope_id)
    {
        return 0;
    }
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    feed_u64(&mut hash, step.schema_version as u64);
    feed_u64(&mut hash, step.subject_fingerprint);
    feed_u64(&mut hash, step.validation_epoch);
    feed_bytes(&mut hash, step.goal_id.as_bytes());
    feed_u64(&mut hash, step.spatial_goal_fingerprint);
    feed_u64(&mut hash, hand_id(step.hand));
    for value in step.target_world_m {
        feed_u64(&mut hash, value.to_bits());
    }
    feed_u64(&mut hash, step.prepared_at_s.to_bits());
    feed_u64(&mut hash, step.observed_at_s.to_bits());
    feed_u64(&mut hash, step.received_at_s.to_bits());
    feed_u64(&mut hash, step.pre_error_m.to_bits());
    feed_u64(&mut hash, step.post_error_m.to_bits());
    feed_u64(&mut hash, step.progress_m.to_bits());
    feed_bytes(&mut hash, step.command_policy_id.as_bytes());
    feed_bytes(&mut hash, step.outcome_policy_id.as_bytes());
    feed_u64(&mut hash, step.command_policy_fingerprint);
    feed_u64(&mut hash, step.outcome_policy_fingerprint);
    feed_u64(&mut hash, step.authority_receipt_fingerprint);
    feed_u64(&mut hash, step.authority_scope_fingerprint);
    feed_bytes(&mut hash, step.authority_scope_id.as_bytes());
    feed_u64(&mut hash, purpose_id(step.execution_purpose));
    feed_u64(&mut hash, step.step_accepted as u64);
    if hash == 0 { 1 } else { hash }
}

fn fingerprint_episode(policy_fingerprint: u64, steps: &[HumanoidReachEpisodeStepEvidence]) -> u64 {
    if policy_fingerprint == 0 || steps.is_empty() || steps.iter().any(|step| !step.validate()) {
        return 0;
    }
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    feed_u64(&mut hash, HUMANOID_REACH_EPISODE_EVIDENCE_SCHEMA_VERSION as u64);
    feed_u64(&mut hash, policy_fingerprint);
    for step in steps {
        feed_u64(&mut hash, step.step_fingerprint);
    }
    if hash == 0 { 1 } else { hash }
}

fn approximately_equal(a: f64, b: f64) -> bool {
    let scale = 1.0 + a.abs().max(b.abs());
    (a - b).abs() <= 1.0e-10 * scale
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
}

fn hand_id(hand: HandSide) -> u64 {
    match hand {
        HandSide::Right => 1,
        HandSide::Left => 2,
    }
}

fn purpose_id(purpose: HumanoidExecutionPurpose) -> u64 {
    match purpose {
        HumanoidExecutionPurpose::SimulationQualification => 1,
        HumanoidExecutionPurpose::HilQualification => 2,
        HumanoidExecutionPurpose::PhysicalQualification => 3,
        HumanoidExecutionPurpose::Operational => 4,
    }
}

fn feed_u64(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *hash ^= byte as u64;
        *hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
}

fn feed_bytes(hash: &mut u64, bytes: &[u8]) {
    feed_u64(hash, bytes.len() as u64);
    for byte in bytes {
        *hash ^= *byte as u64;
        *hash = hash.wrapping_mul(0x1000_0000_01b3);
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
            "episode-test-v1",
        )
    }

    fn step(epoch: u64, prepared: f64, observed: f64, pre: f64, post: f64) -> HumanoidReachEpisodeStepEvidence {
        let mut step = HumanoidReachEpisodeStepEvidence {
            schema_version: HUMANOID_REACH_EPISODE_EVIDENCE_SCHEMA_VERSION,
            subject_fingerprint: subject().fingerprint(),
            validation_epoch: epoch,
            goal_id: "cup-episode".into(),
            spatial_goal_fingerprint: 100 + epoch,
            hand: HandSide::Right,
            target_world_m: [0.4, -0.2, 1.0],
            prepared_at_s: prepared,
            observed_at_s: observed,
            received_at_s: observed + 0.001,
            pre_error_m: pre,
            post_error_m: post,
            progress_m: pre - post,
            command_policy_id: "command-v1".into(),
            outcome_policy_id: "outcome-v1".into(),
            command_policy_fingerprint: 111,
            outcome_policy_fingerprint: 222,
            authority_receipt_fingerprint: 1_000 + epoch,
            authority_scope_fingerprint: 2_000 + epoch,
            authority_scope_id: "reach-sim-episode-v1".into(),
            execution_purpose: HumanoidExecutionPurpose::SimulationQualification,
            qualification_basis: HumanoidQualificationAuthorityBasis::TrialProtocol,
            step_accepted: true,
            step_fingerprint: 0,
        };
        step.step_fingerprint = fingerprint_episode_step(&step);
        step
    }

    fn policy() -> HumanoidReachEpisodePolicy {
        HumanoidReachEpisodePolicy {
            schema_version: HUMANOID_REACH_EPISODE_EVIDENCE_SCHEMA_VERSION,
            policy_id: "episode-policy-v1".into(),
            subject_fingerprint: subject().fingerprint(),
            required_execution_purpose: HumanoidExecutionPurpose::SimulationQualification,
            required_authority_scope_id: "reach-sim-episode-v1".into(),
            command_policy_fingerprint: 111,
            outcome_policy_fingerprint: 222,
            maximum_steps: 8,
            maximum_episode_duration_s: 1.0,
            maximum_inter_step_gap_s: 0.05,
            maximum_target_drift_m: 0.01,
            maximum_final_error_m: 0.02,
            minimum_net_progress_m: 0.04,
            minimum_distinct_authority_receipts: 2,
            require_every_step_accepted: true,
        }
    }

    #[test]
    fn multiple_good_steps_can_establish_episode_completion() {
        let steps = vec![
            step(1, 1.00, 1.02, 0.08, 0.05),
            step(2, 1.03, 1.05, 0.05, 0.015),
        ];
        let assessment = assess_humanoid_reach_episode(&subject(), &policy(), &steps);
        assert!(assessment.episode_accepted);
        assert_eq!(assessment.step_count, 2);
    }

    #[test]
    fn progress_without_final_reach_does_not_qualify_episode() {
        let steps = vec![
            step(1, 1.00, 1.02, 0.08, 0.06),
            step(2, 1.03, 1.05, 0.06, 0.04),
        ];
        let assessment = assess_humanoid_reach_episode(&subject(), &policy(), &steps);
        assert!(!assessment.episode_accepted);
        assert!(assessment.failures.contains(&HumanoidReachEpisodeFailureKind::FinalErrorTooLarge));
    }

    #[test]
    fn stale_validation_epoch_cannot_be_replayed_inside_episode() {
        let steps = vec![
            step(2, 1.00, 1.02, 0.08, 0.05),
            step(2, 1.03, 1.05, 0.05, 0.015),
        ];
        let assessment = assess_humanoid_reach_episode(&subject(), &policy(), &steps);
        assert!(assessment.failures.contains(&HumanoidReachEpisodeFailureKind::ValidationEpochNotIncreasing));
    }

    #[test]
    fn target_identity_can_refresh_but_world_target_drift_is_bounded() {
        let a = step(1, 1.00, 1.02, 0.08, 0.05);
        let mut b = step(2, 1.03, 1.05, 0.05, 0.015);
        b.target_world_m = [0.45, -0.2, 1.0];
        b.step_fingerprint = fingerprint_episode_step(&b);
        let assessment = assess_humanoid_reach_episode(&subject(), &policy(), &[a, b]);
        assert!(assessment.failures.contains(&HumanoidReachEpisodeFailureKind::TargetDriftTooLarge));
    }
}
