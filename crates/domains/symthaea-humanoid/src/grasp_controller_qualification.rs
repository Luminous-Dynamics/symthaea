// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-first qualification contracts for a future humanoid Grasp controller.
//!
//! This module intentionally contains no controller and no actuator lowering.
//! It defines what a candidate controller must prove before a later promotion
//! layer may even consider it. Trials are rebuilt from canonical measured-contact
//! assessments and the existing temporal retention evaluator; a controller's own
//! "retained" flag is only a claim and cannot create evidence.
//!
//! Campaign acceptance is coverage-complete and scenario-local. Easy trials may
//! not average away a missing hand/workspace/fixture/perturbation cell. Acceptance
//! here remains internal engineering evidence, not operational motor authority or
//! legal/product-safety certification.

use std::collections::BTreeSet;

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::execution_authority_scope::HumanoidExecutionPurpose;
use crate::grasp_contact_evidence::{
    HumanoidGraspContactAssessment, HumanoidGraspContactObservation, HumanoidGraspContactPolicy,
};
use crate::grasp_retention_evidence::{
    HumanoidGraspRetentionPolicy, HumanoidGraspRetentionSample,
    evaluate_humanoid_grasp_retention,
};
use crate::morphology::HandSide;
use crate::qualification::HumanoidQualificationSubject;
use crate::types::{ActuationMode, HumanoidTask};

/// v2 makes qualification trials independently self-auditing against the exact
/// subject and scenario-policy metadata. v1 trial digests must not be interpreted
/// under these stronger semantics.
pub const HUMANOID_GRASP_CONTROLLER_QUALIFICATION_SCHEMA_VERSION: u32 = 2;

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspControllerCandidate {
    controller_id: String,
    controller_artifact_digest: HumanoidEvidenceDigest,
    configuration_digest: HumanoidEvidenceDigest,
    candidate_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspControllerCandidate {
    pub fn new(
        controller_id: impl Into<String>,
        controller_artifact_digest: HumanoidEvidenceDigest,
        configuration_digest: HumanoidEvidenceDigest,
    ) -> Option<Self> {
        let mut value = Self {
            controller_id: controller_id.into(),
            controller_artifact_digest,
            configuration_digest,
            candidate_digest: HumanoidEvidenceDigest::ZERO,
        };
        if !value.base_valid() {
            return None;
        }
        value.candidate_digest = digest_candidate(&value);
        value.validate().then_some(value)
    }

    pub fn validate(&self) -> bool {
        self.base_valid()
            && !self.candidate_digest.is_zero()
            && self.candidate_digest == digest_candidate(self)
    }

    fn base_valid(&self) -> bool {
        valid_id(&self.controller_id)
            && !self.controller_artifact_digest.is_zero()
            && !self.configuration_digest.is_zero()
    }

    pub fn controller_id(&self) -> &str {
        &self.controller_id
    }

    pub const fn candidate_digest(&self) -> HumanoidEvidenceDigest {
        self.candidate_digest
    }
}

/// One precommitted qualification cell.
///
/// Profile IDs are human/operator-facing selectors. Their SHA-256 digests bind
/// the exact class/fault/perturbation configurations so an unchanged ID cannot be
/// reused for different test semantics.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspControllerScenarioCell {
    pub scenario_id: String,
    pub hand: HandSide,
    pub minimum_workspace_utilization_sq: f64,
    pub maximum_workspace_utilization_sq: f64,
    pub object_fixture_class_id: String,
    pub object_fixture_class_digest: HumanoidEvidenceDigest,
    pub perturbation_profile_id: String,
    pub perturbation_profile_digest: HumanoidEvidenceDigest,
    pub sensor_fault_profile_id: String,
    pub sensor_fault_profile_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspControllerScenarioCell {
    pub fn validate(&self) -> bool {
        valid_id(&self.scenario_id)
            && valid_id(&self.object_fixture_class_id)
            && !self.object_fixture_class_digest.is_zero()
            && valid_id(&self.perturbation_profile_id)
            && !self.perturbation_profile_digest.is_zero()
            && valid_id(&self.sensor_fault_profile_id)
            && !self.sensor_fault_profile_digest.is_zero()
            && self.minimum_workspace_utilization_sq.is_finite()
            && self.maximum_workspace_utilization_sq.is_finite()
            && self.minimum_workspace_utilization_sq >= 0.0
            && self.maximum_workspace_utilization_sq >= self.minimum_workspace_utilization_sq
            && self.maximum_workspace_utilization_sq <= 1.0
    }

    fn admits(&self, context: &HumanoidGraspControllerTrialContext) -> bool {
        self.validate()
            && context.workspace_utilization_sq >= self.minimum_workspace_utilization_sq
            && context.workspace_utilization_sq <= self.maximum_workspace_utilization_sq
            && context.object_fixture_class_id == self.object_fixture_class_id
            && context.object_fixture_class_digest == self.object_fixture_class_digest
            && context.perturbation_profile_id == self.perturbation_profile_id
            && context.perturbation_profile_digest == self.perturbation_profile_digest
            && context.sensor_fault_profile_id == self.sensor_fault_profile_id
            && context.sensor_fault_profile_digest == self.sensor_fault_profile_digest
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspControllerScenarioRequirement {
    pub cell: HumanoidGraspControllerScenarioCell,
    pub minimum_trials: usize,
    pub maximum_trial_failure_rate: f64,
    pub maximum_false_retention_rate: f64,
    pub maximum_false_negative_rate: f64,
    pub maximum_retention_loss_rate: f64,
    pub maximum_continuity_break_trial_rate: f64,
    pub minimum_distinct_object_fixtures: usize,
    pub minimum_distinct_environment_digests: usize,
    pub require_unique_trial_seeds: bool,
}

impl HumanoidGraspControllerScenarioRequirement {
    pub fn validate(&self) -> bool {
        self.cell.validate()
            && self.minimum_trials > 0
            && valid_rate(self.maximum_trial_failure_rate)
            && valid_rate(self.maximum_false_retention_rate)
            && valid_rate(self.maximum_false_negative_rate)
            && valid_rate(self.maximum_retention_loss_rate)
            && valid_rate(self.maximum_continuity_break_trial_rate)
            && self.minimum_distinct_object_fixtures > 0
            && self.minimum_distinct_object_fixtures <= self.minimum_trials
            && self.minimum_distinct_environment_digests > 0
            && self.minimum_distinct_environment_digests <= self.minimum_trials
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspControllerQualificationPolicy {
    schema_version: u32,
    campaign_id: String,
    subject_digest: HumanoidEvidenceDigest,
    candidate_digest: HumanoidEvidenceDigest,
    contact_policy_digest: HumanoidEvidenceDigest,
    retention_policy_digest: HumanoidEvidenceDigest,
    required_execution_purpose: HumanoidExecutionPurpose,
    required_scenarios: Vec<HumanoidGraspControllerScenarioRequirement>,
    policy_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspControllerQualificationPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        subject: &HumanoidQualificationSubject,
        candidate: &HumanoidGraspControllerCandidate,
        contact_policy: &HumanoidGraspContactPolicy,
        retention_policy: &HumanoidGraspRetentionPolicy,
        campaign_id: impl Into<String>,
        required_execution_purpose: HumanoidExecutionPurpose,
        required_scenarios: Vec<HumanoidGraspControllerScenarioRequirement>,
    ) -> Option<Self> {
        if !subject.validate()
            || subject.task != HumanoidTask::Grasp
            || !candidate.validate()
            || !contact_policy.validate_for(subject)
            || !retention_policy.validate_for(subject, contact_policy)
            || !is_qualification_purpose(required_execution_purpose)
        {
            return None;
        }
        let mut value = Self {
            schema_version: HUMANOID_GRASP_CONTROLLER_QUALIFICATION_SCHEMA_VERSION,
            campaign_id: campaign_id.into(),
            subject_digest: digest_subject(subject)?,
            candidate_digest: candidate.candidate_digest(),
            contact_policy_digest: contact_policy.policy_digest(),
            retention_policy_digest: retention_policy.policy_digest(),
            required_execution_purpose,
            required_scenarios,
            policy_digest: HumanoidEvidenceDigest::ZERO,
        };
        value.policy_digest = digest_policy(&value);
        value
            .validate_for(subject, candidate, contact_policy, retention_policy)
            .then_some(value)
    }

    fn structural_validate_for_subject(&self, subject: &HumanoidQualificationSubject) -> bool {
        if self.schema_version != HUMANOID_GRASP_CONTROLLER_QUALIFICATION_SCHEMA_VERSION
            || digest_subject(subject) != Some(self.subject_digest)
            || subject.task != HumanoidTask::Grasp
            || !valid_id(&self.campaign_id)
            || self.candidate_digest.is_zero()
            || self.contact_policy_digest.is_zero()
            || self.retention_policy_digest.is_zero()
            || !is_qualification_purpose(self.required_execution_purpose)
            || self.required_scenarios.is_empty()
            || self
                .required_scenarios
                .iter()
                .any(|requirement| !requirement.validate())
            || self.policy_digest.is_zero()
            || self.policy_digest != digest_policy(self)
        {
            return false;
        }
        let mut ids = BTreeSet::new();
        self.required_scenarios
            .iter()
            .all(|requirement| ids.insert(requirement.cell.scenario_id.clone()))
    }

    pub fn validate_for(
        &self,
        subject: &HumanoidQualificationSubject,
        candidate: &HumanoidGraspControllerCandidate,
        contact_policy: &HumanoidGraspContactPolicy,
        retention_policy: &HumanoidGraspRetentionPolicy,
    ) -> bool {
        self.structural_validate_for_subject(subject)
            && candidate.validate()
            && self.candidate_digest == candidate.candidate_digest()
            && contact_policy.validate_for(subject)
            && self.contact_policy_digest == contact_policy.policy_digest()
            && retention_policy.validate_for(subject, contact_policy)
            && self.retention_policy_digest == retention_policy.policy_digest()
            // One canonical contact policy is hand-specific. Mixed-hand campaigns
            // therefore require separate qualification policies/campaigns today.
            && self
                .required_scenarios
                .iter()
                .all(|requirement| requirement.cell.hand == contact_policy.hand())
    }

    fn scenario(
        &self,
        scenario_id: &str,
    ) -> Option<&HumanoidGraspControllerScenarioRequirement> {
        self.required_scenarios
            .iter()
            .find(|requirement| requirement.cell.scenario_id == scenario_id)
    }

    pub fn campaign_id(&self) -> &str {
        &self.campaign_id
    }

    pub const fn required_execution_purpose(&self) -> HumanoidExecutionPurpose {
        self.required_execution_purpose
    }

    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest {
        self.policy_digest
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspControllerTrialContext {
    pub trial_id: String,
    pub trial_seed: u64,
    pub scenario_id: String,
    pub execution_purpose: HumanoidExecutionPurpose,
    pub backend_profile_id: String,
    pub workspace_utilization_sq: f64,
    pub object_fixture_id: String,
    pub object_fixture_class_id: String,
    pub object_fixture_class_digest: HumanoidEvidenceDigest,
    pub object_fixture_digest: HumanoidEvidenceDigest,
    pub environment_digest: HumanoidEvidenceDigest,
    pub perturbation_profile_id: String,
    pub perturbation_profile_digest: HumanoidEvidenceDigest,
    pub sensor_fault_profile_id: String,
    pub sensor_fault_profile_digest: HumanoidEvidenceDigest,
    pub controller_reported_retained: bool,
}

impl HumanoidGraspControllerTrialContext {
    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        subject.validate()
            && subject.task == HumanoidTask::Grasp
            && valid_id(&self.trial_id)
            && valid_id(&self.scenario_id)
            && is_qualification_purpose(self.execution_purpose)
            && self.backend_profile_id == subject.backend_profile_id
            && self.workspace_utilization_sq.is_finite()
            && (0.0..=1.0).contains(&self.workspace_utilization_sq)
            && valid_id(&self.object_fixture_id)
            && valid_id(&self.object_fixture_class_id)
            && !self.object_fixture_class_digest.is_zero()
            && !self.object_fixture_digest.is_zero()
            && !self.environment_digest.is_zero()
            && valid_id(&self.perturbation_profile_id)
            && !self.perturbation_profile_digest.is_zero()
            && valid_id(&self.sensor_fault_profile_id)
            && !self.sensor_fault_profile_digest.is_zero()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspControllerTrialBindFailure {
    InvalidSubject,
    InvalidCandidate,
    InvalidPolicy,
    InvalidContext,
    UnknownScenario,
    ScenarioMismatch,
    EvidenceLengthMismatch,
    EmptyEvidence,
    InvalidObservationOrAssessment,
    HandMismatch,
    RetentionSampleBindingFailed,
    RetentionEvaluationFailed,
    InvalidDigest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspControllerQualificationTrial {
    schema_version: u32,
    campaign_policy_digest: HumanoidEvidenceDigest,
    subject_digest: HumanoidEvidenceDigest,
    candidate_digest: HumanoidEvidenceDigest,
    contact_policy_digest: HumanoidEvidenceDigest,
    retention_policy_digest: HumanoidEvidenceDigest,
    scenario_id: String,
    scenario_requirement_digest: HumanoidEvidenceDigest,
    trial_id: String,
    trial_seed: u64,
    execution_purpose: HumanoidExecutionPurpose,
    backend_profile_id: String,
    hand: HandSide,
    workspace_utilization_sq: f64,
    object_id: String,
    object_fixture_id: String,
    object_fixture_class_id: String,
    object_fixture_class_digest: HumanoidEvidenceDigest,
    object_fixture_digest: HumanoidEvidenceDigest,
    environment_digest: HumanoidEvidenceDigest,
    perturbation_profile_id: String,
    perturbation_profile_digest: HumanoidEvidenceDigest,
    sensor_fault_profile_id: String,
    sensor_fault_profile_digest: HumanoidEvidenceDigest,
    observation_digests: Vec<HumanoidEvidenceDigest>,
    assessment_digests: Vec<HumanoidEvidenceDigest>,
    retention_episode_digest: HumanoidEvidenceDigest,
    sample_count: usize,
    accepted_contact_samples: usize,
    in_contact_policy_violation_samples: usize,
    distinct_object_states: usize,
    continuity_breaks: usize,
    reacquisitions: usize,
    ever_retained: bool,
    evidence_retained_at_end: bool,
    controller_reported_retained: bool,
    false_retention: bool,
    false_negative: bool,
    retention_lost_after_success: bool,
    trial_accepted: bool,
    trial_digest: HumanoidEvidenceDigest,
}

impl HumanoidGraspControllerQualificationTrial {
    /// Promotion-grade self-validation against the exact qualification subject
    /// and exact selected campaign cell.
    ///
    /// Recomputing `trial_digest` after tampering is insufficient: subject,
    /// backend, hand, workspace cell, fixture class and perturbation/fault profile
    /// identities must still match the precommitted policy.
    pub fn validate_for(
        &self,
        subject: &HumanoidQualificationSubject,
        policy: &HumanoidGraspControllerQualificationPolicy,
    ) -> bool {
        let Some(subject_digest) = digest_subject(subject) else {
            return false;
        };
        if !policy.structural_validate_for_subject(subject) {
            return false;
        }
        let Some(requirement) = policy.scenario(&self.scenario_id) else {
            return false;
        };

        self.schema_version == HUMANOID_GRASP_CONTROLLER_QUALIFICATION_SCHEMA_VERSION
            && self.campaign_policy_digest == policy.policy_digest()
            && !self.campaign_policy_digest.is_zero()
            && self.subject_digest == subject_digest
            && self.subject_digest == policy.subject_digest
            && self.candidate_digest == policy.candidate_digest
            && !self.candidate_digest.is_zero()
            && self.contact_policy_digest == policy.contact_policy_digest
            && self.retention_policy_digest == policy.retention_policy_digest
            && self.scenario_requirement_digest == digest_scenario_requirement(requirement)
            && !self.scenario_requirement_digest.is_zero()
            && valid_id(&self.scenario_id)
            && self.scenario_id == requirement.cell.scenario_id
            && valid_id(&self.trial_id)
            && self.execution_purpose == policy.required_execution_purpose
            && self.backend_profile_id == subject.backend_profile_id
            && self.hand == requirement.cell.hand
            && self.workspace_utilization_sq.is_finite()
            && self.workspace_utilization_sq >= requirement.cell.minimum_workspace_utilization_sq
            && self.workspace_utilization_sq <= requirement.cell.maximum_workspace_utilization_sq
            && valid_id(&self.object_id)
            && valid_id(&self.object_fixture_id)
            && self.object_fixture_class_id == requirement.cell.object_fixture_class_id
            && self.object_fixture_class_digest == requirement.cell.object_fixture_class_digest
            && !self.object_fixture_digest.is_zero()
            && !self.environment_digest.is_zero()
            && self.perturbation_profile_id == requirement.cell.perturbation_profile_id
            && self.perturbation_profile_digest == requirement.cell.perturbation_profile_digest
            && self.sensor_fault_profile_id == requirement.cell.sensor_fault_profile_id
            && self.sensor_fault_profile_digest == requirement.cell.sensor_fault_profile_digest
            && self.sample_count > 0
            && self.observation_digests.len() == self.sample_count
            && self.assessment_digests.len() == self.sample_count
            && self.observation_digests.iter().all(|digest| !digest.is_zero())
            && self.assessment_digests.iter().all(|digest| !digest.is_zero())
            && !self.retention_episode_digest.is_zero()
            && self.accepted_contact_samples <= self.sample_count
            && self.in_contact_policy_violation_samples <= self.sample_count
            && self.distinct_object_states > 0
            && self.distinct_object_states <= self.sample_count
            && self.false_retention
                == (self.controller_reported_retained && !self.evidence_retained_at_end)
            && self.false_negative
                == (!self.controller_reported_retained && self.evidence_retained_at_end)
            && self.retention_lost_after_success
                == (self.ever_retained && !self.evidence_retained_at_end)
            && self.trial_accepted
                == (self.evidence_retained_at_end
                    && self.controller_reported_retained
                    && !self.false_retention
                    && !self.false_negative
                    && self.in_contact_policy_violation_samples == 0)
            && !self.trial_digest.is_zero()
            && self.trial_digest == digest_trial(self)
    }

    pub const fn trial_accepted(&self) -> bool {
        self.trial_accepted
    }

    pub const fn false_retention(&self) -> bool {
        self.false_retention
    }

    pub const fn false_negative(&self) -> bool {
        self.false_negative
    }

    pub const fn in_contact_policy_violation_samples(&self) -> usize {
        self.in_contact_policy_violation_samples
    }

    pub const fn retention_lost_after_success(&self) -> bool {
        self.retention_lost_after_success
    }

    pub const fn execution_purpose(&self) -> HumanoidExecutionPurpose {
        self.execution_purpose
    }

    pub const fn trial_digest(&self) -> HumanoidEvidenceDigest {
        self.trial_digest
    }
}

#[allow(clippy::too_many_arguments)]
pub fn bind_humanoid_grasp_controller_qualification_trial(
    subject: &HumanoidQualificationSubject,
    candidate: &HumanoidGraspControllerCandidate,
    policy: &HumanoidGraspControllerQualificationPolicy,
    contact_policy: &HumanoidGraspContactPolicy,
    retention_policy: &HumanoidGraspRetentionPolicy,
    context: &HumanoidGraspControllerTrialContext,
    observations: &[HumanoidGraspContactObservation],
    assessments: &[HumanoidGraspContactAssessment],
) -> Result<HumanoidGraspControllerQualificationTrial, HumanoidGraspControllerTrialBindFailure> {
    if !subject.validate() || subject.task != HumanoidTask::Grasp {
        return Err(HumanoidGraspControllerTrialBindFailure::InvalidSubject);
    }
    if !candidate.validate() {
        return Err(HumanoidGraspControllerTrialBindFailure::InvalidCandidate);
    }
    if !policy.validate_for(subject, candidate, contact_policy, retention_policy) {
        return Err(HumanoidGraspControllerTrialBindFailure::InvalidPolicy);
    }
    if !context.validate_for(subject)
        || context.execution_purpose != policy.required_execution_purpose
    {
        return Err(HumanoidGraspControllerTrialBindFailure::InvalidContext);
    }
    let Some(requirement) = policy.scenario(&context.scenario_id) else {
        return Err(HumanoidGraspControllerTrialBindFailure::UnknownScenario);
    };
    if !requirement.cell.admits(context) {
        return Err(HumanoidGraspControllerTrialBindFailure::ScenarioMismatch);
    }
    if observations.len() != assessments.len() {
        return Err(HumanoidGraspControllerTrialBindFailure::EvidenceLengthMismatch);
    }
    if observations.is_empty() {
        return Err(HumanoidGraspControllerTrialBindFailure::EmptyEvidence);
    }

    let mut samples = Vec::with_capacity(observations.len());
    let mut observation_digests = Vec::with_capacity(observations.len());
    let mut assessment_digests = Vec::with_capacity(observations.len());
    let mut accepted_contact_samples = 0usize;
    let mut in_contact_policy_violation_samples = 0usize;
    let mut object_states = BTreeSet::new();

    for (observation, assessment) in observations.iter().zip(assessments) {
        if !observation.validate_for(subject)
            || !assessment.validate(subject, observation, contact_policy)
        {
            return Err(HumanoidGraspControllerTrialBindFailure::InvalidObservationOrAssessment);
        }
        if observation.hand() != requirement.cell.hand {
            return Err(HumanoidGraspControllerTrialBindFailure::HandMismatch);
        }
        if assessment.accepted() {
            accepted_contact_samples += 1;
        }
        if observation.in_contact() && !assessment.accepted() {
            in_contact_policy_violation_samples += 1;
        }
        object_states.insert(observation.object_state_digest());
        observation_digests.push(observation.observation_digest());
        assessment_digests.push(assessment.assessment_digest());
        samples.push(
            HumanoidGraspRetentionSample::bind(subject, observation, assessment, contact_policy)
                .map_err(|_| {
                    HumanoidGraspControllerTrialBindFailure::RetentionSampleBindingFailed
                })?,
        );
    }

    let episode = evaluate_humanoid_grasp_retention(
        subject,
        &samples,
        contact_policy,
        retention_policy,
    )
    .map_err(|_| HumanoidGraspControllerTrialBindFailure::RetentionEvaluationFailed)?;

    let evidence_retained_at_end = episode.retained_at_end();
    let false_retention = context.controller_reported_retained && !evidence_retained_at_end;
    let false_negative = !context.controller_reported_retained && evidence_retained_at_end;
    let retention_lost_after_success = episode.ever_retained() && !evidence_retained_at_end;
    let trial_accepted = evidence_retained_at_end
        && context.controller_reported_retained
        && !false_retention
        && !false_negative
        && in_contact_policy_violation_samples == 0;

    let mut trial = HumanoidGraspControllerQualificationTrial {
        schema_version: HUMANOID_GRASP_CONTROLLER_QUALIFICATION_SCHEMA_VERSION,
        campaign_policy_digest: policy.policy_digest(),
        subject_digest: digest_subject(subject)
            .ok_or(HumanoidGraspControllerTrialBindFailure::InvalidSubject)?,
        candidate_digest: candidate.candidate_digest(),
        contact_policy_digest: contact_policy.policy_digest(),
        retention_policy_digest: retention_policy.policy_digest(),
        scenario_id: context.scenario_id.clone(),
        scenario_requirement_digest: digest_scenario_requirement(requirement),
        trial_id: context.trial_id.clone(),
        trial_seed: context.trial_seed,
        execution_purpose: context.execution_purpose,
        backend_profile_id: context.backend_profile_id.clone(),
        hand: requirement.cell.hand,
        workspace_utilization_sq: context.workspace_utilization_sq,
        object_id: observations[0].object_id().to_string(),
        object_fixture_id: context.object_fixture_id.clone(),
        object_fixture_class_id: context.object_fixture_class_id.clone(),
        object_fixture_class_digest: context.object_fixture_class_digest,
        object_fixture_digest: context.object_fixture_digest,
        environment_digest: context.environment_digest,
        perturbation_profile_id: context.perturbation_profile_id.clone(),
        perturbation_profile_digest: context.perturbation_profile_digest,
        sensor_fault_profile_id: context.sensor_fault_profile_id.clone(),
        sensor_fault_profile_digest: context.sensor_fault_profile_digest,
        observation_digests,
        assessment_digests,
        retention_episode_digest: episode.episode_digest(),
        sample_count: observations.len(),
        accepted_contact_samples,
        in_contact_policy_violation_samples,
        distinct_object_states: object_states.len(),
        continuity_breaks: episode.continuity_breaks(),
        reacquisitions: episode.reacquisitions(),
        ever_retained: episode.ever_retained(),
        evidence_retained_at_end,
        controller_reported_retained: context.controller_reported_retained,
        false_retention,
        false_negative,
        retention_lost_after_success,
        trial_accepted,
        trial_digest: HumanoidEvidenceDigest::ZERO,
    };
    trial.trial_digest = digest_trial(&trial);
    if !trial.validate_for(subject, policy) {
        return Err(HumanoidGraspControllerTrialBindFailure::InvalidDigest);
    }
    Ok(trial)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspControllerScenarioFailureKind {
    MissingTrials,
    FailureRateTooHigh,
    FalseRetentionRateTooHigh,
    FalseNegativeRateTooHigh,
    RetentionLossRateTooHigh,
    ContinuityBreakRateTooHigh,
    InsufficientDistinctObjectFixtures,
    InsufficientDistinctEnvironments,
    DuplicateTrialSeed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspControllerScenarioAssessment {
    pub scenario_id: String,
    pub total_trials: usize,
    pub accepted_trials: usize,
    pub failure_rate: f64,
    pub false_retention_rate: f64,
    pub false_negative_rate: f64,
    pub retention_loss_rate: f64,
    pub continuity_break_trial_rate: f64,
    pub distinct_object_fixtures: usize,
    pub distinct_environments: usize,
    pub distinct_trial_seeds: usize,
    pub accepted: bool,
    pub failures: Vec<HumanoidGraspControllerScenarioFailureKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidGraspControllerCampaignFailureKind {
    InvalidPolicy,
    EmptyCorpus,
    InvalidTrial,
    DuplicateTrialId,
    RequiredScenarioFailed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidGraspControllerQualificationCampaignAssessment {
    pub schema_version: u32,
    pub campaign_id: String,
    pub policy_digest: HumanoidEvidenceDigest,
    pub campaign_digest: HumanoidEvidenceDigest,
    pub total_trials: usize,
    pub total_accepted_trials: usize,
    pub scenarios: Vec<HumanoidGraspControllerScenarioAssessment>,
    pub campaign_accepted: bool,
    pub failures: Vec<HumanoidGraspControllerCampaignFailureKind>,
}

pub fn assess_humanoid_grasp_controller_qualification_campaign(
    subject: &HumanoidQualificationSubject,
    candidate: &HumanoidGraspControllerCandidate,
    policy: &HumanoidGraspControllerQualificationPolicy,
    contact_policy: &HumanoidGraspContactPolicy,
    retention_policy: &HumanoidGraspRetentionPolicy,
    trials: &[HumanoidGraspControllerQualificationTrial],
) -> HumanoidGraspControllerQualificationCampaignAssessment {
    let mut failures = Vec::new();
    if !policy.validate_for(subject, candidate, contact_policy, retention_policy) {
        failures.push(HumanoidGraspControllerCampaignFailureKind::InvalidPolicy);
    }
    if trials.is_empty() {
        failures.push(HumanoidGraspControllerCampaignFailureKind::EmptyCorpus);
    }

    let mut seen_trial_ids = BTreeSet::new();
    for trial in trials {
        if !trial.validate_for(subject, policy) {
            failures.push(HumanoidGraspControllerCampaignFailureKind::InvalidTrial);
            continue;
        }
        if !seen_trial_ids.insert((trial.scenario_id.clone(), trial.trial_id.clone())) {
            failures.push(HumanoidGraspControllerCampaignFailureKind::DuplicateTrialId);
        }
    }

    let mut scenario_assessments = Vec::with_capacity(policy.required_scenarios.len());
    for requirement in &policy.required_scenarios {
        let cell_trials = trials
            .iter()
            .filter(|trial| trial.scenario_id == requirement.cell.scenario_id)
            .collect::<Vec<_>>();
        let total_trials = cell_trials.len();
        let accepted_trials = cell_trials
            .iter()
            .filter(|trial| trial.trial_accepted)
            .count();
        let failure_rate = rate(total_trials - accepted_trials, total_trials, 1.0);
        let false_retention_rate = rate(
            cell_trials
                .iter()
                .filter(|trial| trial.false_retention)
                .count(),
            total_trials,
            1.0,
        );
        let false_negative_rate = rate(
            cell_trials
                .iter()
                .filter(|trial| trial.false_negative)
                .count(),
            total_trials,
            1.0,
        );
        let retention_loss_rate = rate(
            cell_trials
                .iter()
                .filter(|trial| trial.retention_lost_after_success)
                .count(),
            total_trials,
            1.0,
        );
        let continuity_break_trial_rate = rate(
            cell_trials
                .iter()
                .filter(|trial| trial.continuity_breaks > 0)
                .count(),
            total_trials,
            1.0,
        );
        let distinct_object_fixtures = cell_trials
            .iter()
            .map(|trial| trial.object_fixture_digest)
            .collect::<BTreeSet<_>>()
            .len();
        let distinct_environments = cell_trials
            .iter()
            .map(|trial| trial.environment_digest)
            .collect::<BTreeSet<_>>()
            .len();
        let distinct_trial_seeds = cell_trials
            .iter()
            .map(|trial| trial.trial_seed)
            .collect::<BTreeSet<_>>()
            .len();

        let mut scenario_failures = Vec::new();
        if total_trials < requirement.minimum_trials {
            scenario_failures.push(HumanoidGraspControllerScenarioFailureKind::MissingTrials);
        }
        if failure_rate > requirement.maximum_trial_failure_rate {
            scenario_failures.push(HumanoidGraspControllerScenarioFailureKind::FailureRateTooHigh);
        }
        if false_retention_rate > requirement.maximum_false_retention_rate {
            scenario_failures.push(
                HumanoidGraspControllerScenarioFailureKind::FalseRetentionRateTooHigh,
            );
        }
        if false_negative_rate > requirement.maximum_false_negative_rate {
            scenario_failures.push(
                HumanoidGraspControllerScenarioFailureKind::FalseNegativeRateTooHigh,
            );
        }
        if retention_loss_rate > requirement.maximum_retention_loss_rate {
            scenario_failures
                .push(HumanoidGraspControllerScenarioFailureKind::RetentionLossRateTooHigh);
        }
        if continuity_break_trial_rate > requirement.maximum_continuity_break_trial_rate {
            scenario_failures
                .push(HumanoidGraspControllerScenarioFailureKind::ContinuityBreakRateTooHigh);
        }
        if distinct_object_fixtures < requirement.minimum_distinct_object_fixtures {
            scenario_failures.push(
                HumanoidGraspControllerScenarioFailureKind::InsufficientDistinctObjectFixtures,
            );
        }
        if distinct_environments < requirement.minimum_distinct_environment_digests {
            scenario_failures.push(
                HumanoidGraspControllerScenarioFailureKind::InsufficientDistinctEnvironments,
            );
        }
        if requirement.require_unique_trial_seeds && distinct_trial_seeds != total_trials {
            scenario_failures.push(HumanoidGraspControllerScenarioFailureKind::DuplicateTrialSeed);
        }
        let accepted = scenario_failures.is_empty();
        if !accepted {
            failures.push(HumanoidGraspControllerCampaignFailureKind::RequiredScenarioFailed);
        }
        scenario_assessments.push(HumanoidGraspControllerScenarioAssessment {
            scenario_id: requirement.cell.scenario_id.clone(),
            total_trials,
            accepted_trials,
            failure_rate,
            false_retention_rate,
            false_negative_rate,
            retention_loss_rate,
            continuity_break_trial_rate,
            distinct_object_fixtures,
            distinct_environments,
            distinct_trial_seeds,
            accepted,
            failures: scenario_failures,
        });
    }

    let campaign_digest = digest_campaign(policy, trials);
    HumanoidGraspControllerQualificationCampaignAssessment {
        schema_version: HUMANOID_GRASP_CONTROLLER_QUALIFICATION_SCHEMA_VERSION,
        campaign_id: policy.campaign_id.clone(),
        policy_digest: policy.policy_digest(),
        campaign_digest,
        total_trials: trials.len(),
        total_accepted_trials: trials
            .iter()
            .filter(|trial| trial.trial_accepted)
            .count(),
        scenarios: scenario_assessments,
        campaign_accepted: failures.is_empty() && !campaign_digest.is_zero(),
        failures,
    }
}

fn digest_candidate(value: &HumanoidGraspControllerCandidate) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-controller-candidate.v1");
    h.string(&value.controller_id)
        .digest(value.controller_artifact_digest)
        .digest(value.configuration_digest);
    h.finish()
}

fn digest_scenario_requirement(
    value: &HumanoidGraspControllerScenarioRequirement,
) -> HumanoidEvidenceDigest {
    if !value.validate() {
        return HumanoidEvidenceDigest::ZERO;
    }
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-controller-scenario-requirement.v2");
    h.string(&value.cell.scenario_id)
        .u64(hand_id(value.cell.hand))
        .f64(value.cell.minimum_workspace_utilization_sq)
        .f64(value.cell.maximum_workspace_utilization_sq)
        .string(&value.cell.object_fixture_class_id)
        .digest(value.cell.object_fixture_class_digest)
        .string(&value.cell.perturbation_profile_id)
        .digest(value.cell.perturbation_profile_digest)
        .string(&value.cell.sensor_fault_profile_id)
        .digest(value.cell.sensor_fault_profile_digest)
        .usize(value.minimum_trials)
        .f64(value.maximum_trial_failure_rate)
        .f64(value.maximum_false_retention_rate)
        .f64(value.maximum_false_negative_rate)
        .f64(value.maximum_retention_loss_rate)
        .f64(value.maximum_continuity_break_trial_rate)
        .usize(value.minimum_distinct_object_fixtures)
        .usize(value.minimum_distinct_environment_digests)
        .bool(value.require_unique_trial_seeds);
    h.finish()
}

fn digest_policy(value: &HumanoidGraspControllerQualificationPolicy) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-controller-qualification-policy.v2");
    h.u32(value.schema_version)
        .string(&value.campaign_id)
        .digest(value.subject_digest)
        .digest(value.candidate_digest)
        .digest(value.contact_policy_digest)
        .digest(value.retention_policy_digest)
        .u64(purpose_id(value.required_execution_purpose))
        .usize(value.required_scenarios.len());
    for requirement in &value.required_scenarios {
        h.digest(digest_scenario_requirement(requirement));
    }
    h.finish()
}

fn digest_trial(value: &HumanoidGraspControllerQualificationTrial) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-controller-qualification-trial.v2");
    h.u32(value.schema_version)
        .digest(value.campaign_policy_digest)
        .digest(value.subject_digest)
        .digest(value.candidate_digest)
        .digest(value.contact_policy_digest)
        .digest(value.retention_policy_digest)
        .string(&value.scenario_id)
        .digest(value.scenario_requirement_digest)
        .string(&value.trial_id)
        .u64(value.trial_seed)
        .u64(purpose_id(value.execution_purpose))
        .string(&value.backend_profile_id)
        .u64(hand_id(value.hand))
        .f64(value.workspace_utilization_sq)
        .string(&value.object_id)
        .string(&value.object_fixture_id)
        .string(&value.object_fixture_class_id)
        .digest(value.object_fixture_class_digest)
        .digest(value.object_fixture_digest)
        .digest(value.environment_digest)
        .string(&value.perturbation_profile_id)
        .digest(value.perturbation_profile_digest)
        .string(&value.sensor_fault_profile_id)
        .digest(value.sensor_fault_profile_digest)
        .usize(value.observation_digests.len());
    for digest in &value.observation_digests {
        h.digest(*digest);
    }
    h.usize(value.assessment_digests.len());
    for digest in &value.assessment_digests {
        h.digest(*digest);
    }
    h.digest(value.retention_episode_digest)
        .usize(value.sample_count)
        .usize(value.accepted_contact_samples)
        .usize(value.in_contact_policy_violation_samples)
        .usize(value.distinct_object_states)
        .usize(value.continuity_breaks)
        .usize(value.reacquisitions)
        .bool(value.ever_retained)
        .bool(value.evidence_retained_at_end)
        .bool(value.controller_reported_retained)
        .bool(value.false_retention)
        .bool(value.false_negative)
        .bool(value.retention_lost_after_success)
        .bool(value.trial_accepted);
    h.finish()
}

fn digest_campaign(
    policy: &HumanoidGraspControllerQualificationPolicy,
    trials: &[HumanoidGraspControllerQualificationTrial],
) -> HumanoidEvidenceDigest {
    if trials.is_empty() {
        return HumanoidEvidenceDigest::ZERO;
    }
    let mut sorted = trials.iter().collect::<Vec<_>>();
    sorted.sort_by(|left, right| {
        left.scenario_id
            .cmp(&right.scenario_id)
            .then(left.trial_id.cmp(&right.trial_id))
            .then(left.trial_seed.cmp(&right.trial_seed))
    });
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-controller-qualification-campaign.v2");
    h.digest(policy.policy_digest()).usize(sorted.len());
    for trial in sorted {
        h.digest(trial.trial_digest);
    }
    h.finish()
}

fn digest_subject(subject: &HumanoidQualificationSubject) -> Option<HumanoidEvidenceDigest> {
    if !subject.validate() || subject.task != HumanoidTask::Grasp {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("humanoid.grasp-controller-qualification-subject.v1");
    h.u32(subject.schema_version)
        .string(subject.morphology.schema_id())
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    Some(h.finish())
}

fn rate(numerator: usize, denominator: usize, empty: f64) -> f64 {
    if denominator == 0 {
        empty
    } else {
        numerator as f64 / denominator as f64
    }
}

fn valid_rate(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
}

fn is_qualification_purpose(purpose: HumanoidExecutionPurpose) -> bool {
    matches!(
        purpose,
        HumanoidExecutionPurpose::SimulationQualification
            | HumanoidExecutionPurpose::HilQualification
            | HumanoidExecutionPurpose::PhysicalQualification
    )
}

fn purpose_id(purpose: HumanoidExecutionPurpose) -> u64 {
    match purpose {
        HumanoidExecutionPurpose::SimulationQualification => 1,
        HumanoidExecutionPurpose::HilQualification => 2,
        HumanoidExecutionPurpose::PhysicalQualification => 3,
        HumanoidExecutionPurpose::Operational => 4,
    }
}

fn hand_id(hand: HandSide) -> u64 {
    match hand {
        HandSide::Right => 1,
        HandSide::Left => 2,
    }
}

fn task_id(task: HumanoidTask) -> u64 {
    match task {
        HumanoidTask::Stand => 1,
        HumanoidTask::Walk => 2,
        HumanoidTask::Run => 3,
        HumanoidTask::Reach => 4,
        HumanoidTask::Grasp => 5,
    }
}

fn actuation_mode_id(mode: ActuationMode) -> u64 {
    match mode {
        ActuationMode::NormalizedTorque => 1,
        ActuationMode::TorqueNewtonMetres => 2,
        ActuationMode::NormalizedPosition => 3,
        ActuationMode::PositionTargetRadians => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contact_site::HumanoidContactSite;
    use crate::grasp_contact_evidence::{
        HumanoidManipulationContactSource, assess_humanoid_grasp_contact,
    };
    use crate::morphology::HumanoidMorphology;

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Grasp,
            ActuationMode::NormalizedTorque,
            "grasp-controller-qualification-test-backend",
        )
    }

    fn candidate() -> HumanoidGraspControllerCandidate {
        HumanoidGraspControllerCandidate::new(
            "candidate-v1",
            HumanoidEvidenceDigest::from_bytes([1; 32]),
            HumanoidEvidenceDigest::from_bytes([2; 32]),
        )
        .unwrap()
    }

    fn contact_policy() -> HumanoidGraspContactPolicy {
        HumanoidGraspContactPolicy::new(
            &subject(),
            HandSide::Right,
            0.05,
            0.8,
            HumanoidManipulationContactSource::SolverWrench.quality_rank(),
            true,
            2.0,
            50.0,
            15.0,
            2.0,
            0.02,
            0.01,
            0.02,
        )
        .unwrap()
    }

    fn retention_policy(contact: &HumanoidGraspContactPolicy) -> HumanoidGraspRetentionPolicy {
        HumanoidGraspRetentionPolicy::new(&subject(), contact, 3, 0.04, 5, 0.08, 0.03, 64, 2.0)
            .unwrap()
    }

    fn scenario(id: &str, min_u: f64, max_u: f64) -> HumanoidGraspControllerScenarioRequirement {
        HumanoidGraspControllerScenarioRequirement {
            cell: HumanoidGraspControllerScenarioCell {
                scenario_id: id.into(),
                hand: HandSide::Right,
                minimum_workspace_utilization_sq: min_u,
                maximum_workspace_utilization_sq: max_u,
                object_fixture_class_id: "rigid-small-v1".into(),
                object_fixture_class_digest: HumanoidEvidenceDigest::from_bytes([7; 32]),
                perturbation_profile_id: "nominal-v1".into(),
                perturbation_profile_digest: HumanoidEvidenceDigest::from_bytes([5; 32]),
                sensor_fault_profile_id: "sensors-nominal-v1".into(),
                sensor_fault_profile_digest: HumanoidEvidenceDigest::from_bytes([6; 32]),
            },
            minimum_trials: 1,
            maximum_trial_failure_rate: 0.0,
            maximum_false_retention_rate: 0.0,
            maximum_false_negative_rate: 0.0,
            maximum_retention_loss_rate: 0.0,
            maximum_continuity_break_trial_rate: 0.0,
            minimum_distinct_object_fixtures: 1,
            minimum_distinct_environment_digests: 1,
            require_unique_trial_seeds: true,
        }
    }

    fn policy_with(
        scenarios: Vec<HumanoidGraspControllerScenarioRequirement>,
        purpose: HumanoidExecutionPurpose,
    ) -> Option<HumanoidGraspControllerQualificationPolicy> {
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        HumanoidGraspControllerQualificationPolicy::new(
            &subject(),
            &candidate(),
            &contact,
            &retention,
            "grasp-controller-campaign-v1",
            purpose,
            scenarios,
        )
    }

    fn context(
        scenario_id: &str,
        workspace_utilization_sq: f64,
        reported: bool,
    ) -> HumanoidGraspControllerTrialContext {
        HumanoidGraspControllerTrialContext {
            trial_id: format!("trial-{scenario_id}"),
            trial_seed: 7,
            scenario_id: scenario_id.into(),
            execution_purpose: HumanoidExecutionPurpose::SimulationQualification,
            backend_profile_id: subject().backend_profile_id,
            workspace_utilization_sq,
            object_fixture_id: "fixture-a".into(),
            object_fixture_class_id: "rigid-small-v1".into(),
            object_fixture_class_digest: HumanoidEvidenceDigest::from_bytes([7; 32]),
            object_fixture_digest: HumanoidEvidenceDigest::from_bytes([3; 32]),
            environment_digest: HumanoidEvidenceDigest::from_bytes([4; 32]),
            perturbation_profile_id: "nominal-v1".into(),
            perturbation_profile_digest: HumanoidEvidenceDigest::from_bytes([5; 32]),
            sensor_fault_profile_id: "sensors-nominal-v1".into(),
            sensor_fault_profile_digest: HumanoidEvidenceDigest::from_bytes([6; 32]),
            controller_reported_retained: reported,
        }
    }

    fn evidence(
        force_n: f64,
        in_contact: bool,
        times: &[f64],
    ) -> (
        Vec<HumanoidGraspContactObservation>,
        Vec<HumanoidGraspContactAssessment>,
    ) {
        let contact = contact_policy();
        let mut observations = Vec::new();
        let mut assessments = Vec::new();
        for (index, time) in times.iter().copied().enumerate() {
            let observation = HumanoidGraspContactObservation::new(
                &subject(),
                "object-a",
                HumanoidEvidenceDigest::from_bytes([index as u8 + 10; 32]),
                HandSide::Right,
                HumanoidContactSite::RightHand,
                in_contact,
                [0.2, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                if in_contact {
                    [-force_n, 0.1, 0.0]
                } else {
                    [0.0; 3]
                },
                [0.0; 3],
                [0.0; 3],
                0.95,
                if in_contact {
                    HumanoidManipulationContactSource::ForceTorqueSensor
                } else {
                    HumanoidManipulationContactSource::KinematicEstimate
                },
                time,
            )
            .unwrap();
            let assessment =
                assess_humanoid_grasp_contact(&subject(), &observation, &contact, time + 0.001)
                    .unwrap();
            observations.push(observation);
            assessments.push(assessment);
        }
        (observations, assessments)
    }

    fn bind_trial(
        policy: &HumanoidGraspControllerQualificationPolicy,
        context: &HumanoidGraspControllerTrialContext,
        observations: &[HumanoidGraspContactObservation],
        assessments: &[HumanoidGraspContactAssessment],
    ) -> HumanoidGraspControllerQualificationTrial {
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        bind_humanoid_grasp_controller_qualification_trial(
            &subject(),
            &candidate(),
            policy,
            &contact,
            &retention,
            context,
            observations,
            assessments,
        )
        .unwrap()
    }

    fn accepted_trial() -> (
        HumanoidGraspControllerQualificationPolicy,
        HumanoidGraspControllerQualificationTrial,
    ) {
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        let policy = HumanoidGraspControllerQualificationPolicy::new(
            &subject(),
            &candidate(),
            &contact,
            &retention,
            "grasp-controller-campaign-v1",
            HumanoidExecutionPurpose::SimulationQualification,
            vec![scenario("center", 0.0, 0.5)],
        )
        .unwrap();
        let (observations, assessments) =
            evidence(10.0, true, &[1.00, 1.02, 1.04, 1.06, 1.085]);
        let trial = bind_trial(
            &policy,
            &context("center", 0.25, true),
            &observations,
            &assessments,
        );
        (policy, trial)
    }

    #[test]
    fn operational_purpose_cannot_be_a_qualification_campaign() {
        assert!(
            policy_with(
                vec![scenario("center", 0.0, 0.5)],
                HumanoidExecutionPurpose::Operational,
            )
            .is_none()
        );
    }

    #[test]
    fn measured_continuous_retention_and_matching_controller_claim_passes_trial() {
        let (policy, trial) = accepted_trial();
        assert!(trial.validate_for(&subject(), &policy));
        assert!(trial.trial_accepted());
        assert!(!trial.false_retention());
        assert!(!trial.false_negative());
        assert_eq!(trial.in_contact_policy_violation_samples(), 0);
    }

    #[test]
    fn controller_cannot_self_certify_retention() {
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        let policy = HumanoidGraspControllerQualificationPolicy::new(
            &subject(),
            &candidate(),
            &contact,
            &retention,
            "grasp-controller-campaign-v1",
            HumanoidExecutionPurpose::SimulationQualification,
            vec![scenario("center", 0.0, 0.5)],
        )
        .unwrap();
        let (observations, assessments) =
            evidence(0.0, false, &[1.00, 1.02, 1.04, 1.06, 1.085]);
        let trial = bind_trial(
            &policy,
            &context("center", 0.25, true),
            &observations,
            &assessments,
        );
        assert!(!trial.trial_accepted());
        assert!(trial.false_retention());
    }

    #[test]
    fn unsafe_contact_cannot_be_averaged_away_by_later_good_samples() {
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        let policy = HumanoidGraspControllerQualificationPolicy::new(
            &subject(),
            &candidate(),
            &contact,
            &retention,
            "grasp-controller-campaign-v1",
            HumanoidExecutionPurpose::SimulationQualification,
            vec![scenario("center", 0.0, 0.5)],
        )
        .unwrap();

        let (mut observations, mut assessments) = evidence(80.0, true, &[0.98]);
        let (good_observations, good_assessments) =
            evidence(10.0, true, &[1.00, 1.02, 1.04, 1.06, 1.085]);
        observations.extend(good_observations);
        assessments.extend(good_assessments);

        let trial = bind_trial(
            &policy,
            &context("center", 0.25, true),
            &observations,
            &assessments,
        );
        assert!(!trial.trial_accepted());
        assert_eq!(trial.in_contact_policy_violation_samples(), 1);
    }

    #[test]
    fn missing_required_scenario_fails_campaign_cell_locally() {
        let contact = contact_policy();
        let retention = retention_policy(&contact);
        let policy = HumanoidGraspControllerQualificationPolicy::new(
            &subject(),
            &candidate(),
            &contact,
            &retention,
            "grasp-controller-campaign-v1",
            HumanoidExecutionPurpose::SimulationQualification,
            vec![
                scenario("center", 0.0, 0.5),
                scenario("boundary", 0.8, 1.0),
            ],
        )
        .unwrap();
        let (observations, assessments) =
            evidence(10.0, true, &[1.00, 1.02, 1.04, 1.06, 1.085]);
        let center = bind_trial(
            &policy,
            &context("center", 0.25, true),
            &observations,
            &assessments,
        );
        let campaign = assess_humanoid_grasp_controller_qualification_campaign(
            &subject(),
            &candidate(),
            &policy,
            &contact,
            &retention,
            &[center],
        );
        assert!(!campaign.campaign_accepted);
        let boundary = campaign
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == "boundary")
            .unwrap();
        assert!(
            boundary
                .failures
                .contains(&HumanoidGraspControllerScenarioFailureKind::MissingTrials)
        );
    }

    #[test]
    fn exact_subject_substitution_is_rejected() {
        let (policy, trial) = accepted_trial();
        let substituted = HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Grasp,
            ActuationMode::PositionTargetRadians,
            "grasp-controller-qualification-test-backend",
        );
        assert!(!trial.validate_for(&substituted, &policy));
        assert!(trial.validate_for(&subject(), &policy));
    }

    #[test]
    fn backend_substitution_is_rejected_even_after_recomputing_trial_digest() {
        let (policy, mut trial) = accepted_trial();
        trial.backend_profile_id = "other-backend".into();
        trial.trial_digest = digest_trial(&trial);
        assert!(!trial.validate_for(&subject(), &policy));
    }

    #[test]
    fn scenario_metadata_substitution_is_rejected_even_after_recomputing_trial_digest() {
        let (policy, mut trial) = accepted_trial();
        trial.object_fixture_class_id = "other-fixture-class".into();
        trial.object_fixture_class_digest = HumanoidEvidenceDigest::from_bytes([8; 32]);
        trial.trial_digest = digest_trial(&trial);
        assert!(!trial.validate_for(&subject(), &policy));
    }

    #[test]
    fn perturbation_configuration_substitution_is_rejected_even_after_recomputing_trial_digest() {
        let (policy, mut trial) = accepted_trial();
        trial.perturbation_profile_digest = HumanoidEvidenceDigest::from_bytes([9; 32]);
        trial.trial_digest = digest_trial(&trial);
        assert!(!trial.validate_for(&subject(), &policy));
    }

    #[test]
    fn mixed_hand_campaign_is_rejected_for_one_hand_contact_policy() {
        let mut left = scenario("left", 0.0, 0.5);
        left.cell.hand = HandSide::Left;
        assert!(
            policy_with(
                vec![scenario("right", 0.0, 0.5), left],
                HumanoidExecutionPurpose::SimulationQualification,
            )
            .is_none()
        );
    }
}
