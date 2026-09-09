// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coverage-complete Reach qualification campaign evidence.
//!
//! A single accepted Reach step is not a capability qualification. This module
//! aggregates target-bound step evidence across an explicitly declared scenario
//! matrix and fails closed when any required hand/workspace/perturbation cell is
//! missing or under-sampled.
//!
//! Acceptance is intentionally cell-local. A large number of easy center-workspace
//! successes cannot average away missing boundary coverage or a failing opposite
//! hand. This is internal engineering qualification evidence, not legal/product
//! safety certification.

use std::collections::{BTreeMap, BTreeSet};

use crate::execution_authority_scope::{
    HumanoidExecutionPurpose, HumanoidQualificationAuthorityBasis,
};
use crate::morphology::HandSide;
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_execution::HumanoidPermittedReachExecutionResult;
use crate::reach_outcome_evidence::HumanoidReachStepEvidenceAssessment;
use crate::types::HumanoidTask;

pub const HUMANOID_REACH_QUALIFICATION_CAMPAIGN_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachScenarioCell {
    pub scenario_id: String,
    pub hand: HandSide,
    pub minimum_workspace_utilization_sq: f64,
    pub maximum_workspace_utilization_sq: f64,
    pub perturbation_profile_id: String,
}

impl HumanoidReachScenarioCell {
    pub fn validate(&self) -> bool {
        valid_id(&self.scenario_id)
            && valid_id(&self.perturbation_profile_id)
            && self.minimum_workspace_utilization_sq.is_finite()
            && self.maximum_workspace_utilization_sq.is_finite()
            && self.minimum_workspace_utilization_sq >= 0.0
            && self.maximum_workspace_utilization_sq >= self.minimum_workspace_utilization_sq
            && self.maximum_workspace_utilization_sq <= 1.0
    }

    pub fn admits(&self, hand: HandSide, workspace_utilization_sq: f64, perturbation: &str) -> bool {
        self.validate()
            && self.hand == hand
            && perturbation == self.perturbation_profile_id
            && workspace_utilization_sq.is_finite()
            && workspace_utilization_sq >= self.minimum_workspace_utilization_sq
            && workspace_utilization_sq <= self.maximum_workspace_utilization_sq
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachScenarioRequirement {
    pub cell: HumanoidReachScenarioCell,
    pub minimum_trials: usize,
    pub maximum_failure_rate: f64,
    pub minimum_distinct_spatial_goals: usize,
    pub minimum_distinct_authority_receipts: usize,
    pub require_unique_trial_seeds: bool,
}

impl HumanoidReachScenarioRequirement {
    pub fn validate(&self) -> bool {
        self.cell.validate()
            && self.minimum_trials > 0
            && self.maximum_failure_rate.is_finite()
            && (0.0..=1.0).contains(&self.maximum_failure_rate)
            && self.minimum_distinct_spatial_goals > 0
            && self.minimum_distinct_spatial_goals <= self.minimum_trials
            && self.minimum_distinct_authority_receipts > 0
            && self.minimum_distinct_authority_receipts <= self.minimum_trials
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachQualificationCampaignPolicy {
    pub schema_version: u32,
    pub campaign_id: String,
    pub subject_fingerprint: u64,
    /// Campaigns are qualification evidence, so Operational purpose is rejected.
    pub required_execution_purpose: HumanoidExecutionPurpose,
    pub required_authority_scope_id: String,
    pub required_command_policy_id: String,
    pub required_outcome_policy_id: String,
    pub required_scenarios: Vec<HumanoidReachScenarioRequirement>,
}

impl HumanoidReachQualificationCampaignPolicy {
    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        if self.schema_version != HUMANOID_REACH_QUALIFICATION_CAMPAIGN_SCHEMA_VERSION
            || !valid_id(&self.campaign_id)
            || !subject.validate()
            || subject.task != HumanoidTask::Reach
            || self.subject_fingerprint == 0
            || self.subject_fingerprint != subject.fingerprint()
            || !is_qualification_purpose(self.required_execution_purpose)
            || !valid_id(&self.required_authority_scope_id)
            || !valid_id(&self.required_command_policy_id)
            || !valid_id(&self.required_outcome_policy_id)
            || self.required_scenarios.is_empty()
            || self.required_scenarios.iter().any(|requirement| !requirement.validate())
        {
            return false;
        }
        let mut ids = BTreeSet::new();
        self.required_scenarios
            .iter()
            .all(|requirement| ids.insert(requirement.cell.scenario_id.clone()))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachQualificationTrial {
    pub schema_version: u32,
    pub subject_fingerprint: u64,
    pub scenario_id: String,
    pub perturbation_profile_id: String,
    pub trial_id: String,
    pub trial_seed: u64,
    pub hand: HandSide,
    pub workspace_utilization_sq: f64,
    pub validation_epoch: u64,
    pub goal_id: String,
    pub spatial_goal_fingerprint: u64,
    pub command_policy_id: String,
    pub outcome_policy_id: String,
    pub authority_receipt_fingerprint: u64,
    pub authority_scope_fingerprint: u64,
    pub authority_scope_id: String,
    pub execution_purpose: HumanoidExecutionPurpose,
    pub qualification_basis: HumanoidQualificationAuthorityBasis,
    pub authority_effective_scale: f32,
    pub step_accepted: bool,
    pub trial_fingerprint: u64,
}

impl HumanoidReachQualificationTrial {
    pub fn validate(&self) -> bool {
        self.schema_version == HUMANOID_REACH_QUALIFICATION_CAMPAIGN_SCHEMA_VERSION
            && self.subject_fingerprint != 0
            && valid_id(&self.scenario_id)
            && valid_id(&self.perturbation_profile_id)
            && valid_id(&self.trial_id)
            && self.workspace_utilization_sq.is_finite()
            && (0.0..=1.0).contains(&self.workspace_utilization_sq)
            && self.validation_epoch != 0
            && valid_id(&self.goal_id)
            && self.spatial_goal_fingerprint != 0
            && valid_id(&self.command_policy_id)
            && valid_id(&self.outcome_policy_id)
            && self.authority_receipt_fingerprint != 0
            && self.authority_scope_fingerprint != 0
            && valid_id(&self.authority_scope_id)
            && is_qualification_purpose(self.execution_purpose)
            && self.qualification_basis == HumanoidQualificationAuthorityBasis::TrialProtocol
            && self.authority_effective_scale.is_finite()
            && (0.0..=1.0).contains(&self.authority_effective_scale)
            && self.trial_fingerprint != 0
            && self.trial_fingerprint == fingerprint_trial(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachQualificationTrialBindFailure {
    InvalidSubject,
    SubjectIsNotReach,
    InvalidScenario,
    InvalidPerturbationProfile,
    ScenarioDoesNotMatchExecution,
    InvalidTrialId,
    StepIdentityMismatch,
    StepPolicyIdentityInvalid,
    AuthorityIdentityMismatch,
    AuthorityEnvelopeMismatch,
    InvalidAuthorityLineage,
    NonQualificationAuthorityScope,
}

#[allow(clippy::too_many_arguments)]
pub fn bind_humanoid_reach_qualification_trial(
    subject: &HumanoidQualificationSubject,
    scenario: &HumanoidReachScenarioCell,
    perturbation_profile_id: impl Into<String>,
    trial_id: impl Into<String>,
    trial_seed: u64,
    result: &HumanoidPermittedReachExecutionResult,
    step: &HumanoidReachStepEvidenceAssessment,
) -> Result<HumanoidReachQualificationTrial, HumanoidReachQualificationTrialBindFailure> {
    if !subject.validate() {
        return Err(HumanoidReachQualificationTrialBindFailure::InvalidSubject);
    }
    if subject.task != HumanoidTask::Reach {
        return Err(HumanoidReachQualificationTrialBindFailure::SubjectIsNotReach);
    }
    if !scenario.validate() {
        return Err(HumanoidReachQualificationTrialBindFailure::InvalidScenario);
    }
    let perturbation_profile_id = perturbation_profile_id.into();
    if !valid_id(&perturbation_profile_id) {
        return Err(HumanoidReachQualificationTrialBindFailure::InvalidPerturbationProfile);
    }
    let trial_id = trial_id.into();
    if !valid_id(&trial_id) {
        return Err(HumanoidReachQualificationTrialBindFailure::InvalidTrialId);
    }
    if !scenario.admits(
        result.preparation.hand,
        result.preparation.workspace_utilization_sq,
        &perturbation_profile_id,
    ) {
        return Err(HumanoidReachQualificationTrialBindFailure::ScenarioDoesNotMatchExecution);
    }

    let subject_fingerprint = subject.fingerprint();
    if step.subject_fingerprint != subject_fingerprint
        || step.validation_epoch != result.preparation.validation_epoch
        || step.goal_id != result.preparation.goal_id
        || step.spatial_goal_fingerprint != result.preparation.spatial_goal_fingerprint
        || step.command.subject_fingerprint != subject_fingerprint
        || step.command.validation_epoch != result.preparation.validation_epoch
        || step.command.goal_id != result.preparation.goal_id
        || step.outcome.subject_fingerprint != subject_fingerprint
        || step.outcome.validation_epoch != result.preparation.validation_epoch
        || step.outcome.goal_id != result.preparation.goal_id
        || step.outcome.spatial_goal_fingerprint != result.preparation.spatial_goal_fingerprint
    {
        return Err(HumanoidReachQualificationTrialBindFailure::StepIdentityMismatch);
    }
    if !valid_id(&step.command.policy_id) || !valid_id(&step.outcome.policy_id) {
        return Err(HumanoidReachQualificationTrialBindFailure::StepPolicyIdentityInvalid);
    }

    let authority = &result.authority_receipt;
    if authority.receipt_fingerprint == 0
        || authority.scope_fingerprint == 0
        || authority.validation_epoch != result.preparation.validation_epoch
        || authority.requirement_subject_fingerprints.as_slice() != [subject_fingerprint]
    {
        return Err(HumanoidReachQualificationTrialBindFailure::AuthorityIdentityMismatch);
    }
    if !is_qualification_purpose(authority.execution_purpose)
        || authority.qualification_basis != HumanoidQualificationAuthorityBasis::TrialProtocol
    {
        return Err(HumanoidReachQualificationTrialBindFailure::NonQualificationAuthorityScope);
    }
    if !valid_id(&authority.scope_id)
        || !authority.issued_at_s.is_finite()
        || !authority.valid_until_s.is_finite()
        || !authority.finalized_at_s.is_finite()
        || authority.issued_at_s < 0.0
        || authority.valid_until_s < authority.issued_at_s
        || authority.finalized_at_s < authority.issued_at_s
        || authority.finalized_at_s > authority.valid_until_s
        || authority.finalized_at_s < result.preparation.prepared_at_s
        || ![
            authority.operator_scale,
            authority.qualification_scale,
            authority.physical_scale,
            authority.epistemic_scale,
            authority.cognitive_scale,
        ]
        .into_iter()
        .all(|scale| scale.is_finite() && (0.0..=1.0).contains(&scale))
        || ![
            authority.operator_evidence_id.as_str(),
            authority.qualification_evidence_id.as_str(),
            authority.physical_evidence_id.as_str(),
            authority.epistemic_evidence_id.as_str(),
            authority.cognitive_evidence_id.as_str(),
        ]
        .into_iter()
        .all(valid_id)
    {
        return Err(HumanoidReachQualificationTrialBindFailure::InvalidAuthorityLineage);
    }

    let reported = result.execution.report.authority;
    let expected_scales = [
        authority.operator_scale,
        authority.qualification_scale,
        authority.physical_scale,
        authority.epistemic_scale,
        authority.cognitive_scale,
    ];
    let reported_scales = [
        reported.operator,
        reported.qualification,
        reported.physical,
        reported.epistemic,
        reported.cognitive,
    ];
    if !expected_scales
        .into_iter()
        .zip(reported_scales)
        .all(|(expected, actual)| expected.to_bits() == actual.to_bits())
    {
        return Err(HumanoidReachQualificationTrialBindFailure::AuthorityEnvelopeMismatch);
    }
    let authority_effective_scale = reported.effective_scale();
    if authority_effective_scale.to_bits() != result.execution.report.authority_scale.to_bits() {
        return Err(HumanoidReachQualificationTrialBindFailure::AuthorityEnvelopeMismatch);
    }

    let mut trial = HumanoidReachQualificationTrial {
        schema_version: HUMANOID_REACH_QUALIFICATION_CAMPAIGN_SCHEMA_VERSION,
        subject_fingerprint,
        scenario_id: scenario.scenario_id.clone(),
        perturbation_profile_id,
        trial_id,
        trial_seed,
        hand: result.preparation.hand,
        workspace_utilization_sq: result.preparation.workspace_utilization_sq,
        validation_epoch: result.preparation.validation_epoch,
        goal_id: result.preparation.goal_id.clone(),
        spatial_goal_fingerprint: result.preparation.spatial_goal_fingerprint,
        command_policy_id: step.command.policy_id.clone(),
        outcome_policy_id: step.outcome.policy_id.clone(),
        authority_receipt_fingerprint: authority.receipt_fingerprint,
        authority_scope_fingerprint: authority.scope_fingerprint,
        authority_scope_id: authority.scope_id.clone(),
        execution_purpose: authority.execution_purpose,
        qualification_basis: authority.qualification_basis,
        authority_effective_scale,
        step_accepted: step.step_accepted,
        trial_fingerprint: 0,
    };
    trial.trial_fingerprint = fingerprint_trial(&trial);
    Ok(trial)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachScenarioAssessmentFailureKind {
    MissingTrials,
    FailureRateTooHigh,
    InsufficientDistinctSpatialGoals,
    InsufficientDistinctAuthorityReceipts,
    DuplicateTrialSeed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachScenarioAssessment {
    pub scenario_id: String,
    pub total_trials: usize,
    pub accepted_trials: usize,
    pub failure_rate: f64,
    pub distinct_spatial_goals: usize,
    pub distinct_authority_receipts: usize,
    pub distinct_trial_seeds: usize,
    pub accepted: bool,
    pub failures: Vec<HumanoidReachScenarioAssessmentFailureKind>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachCampaignFailureKind {
    InvalidPolicy,
    EmptyCorpus,
    InvalidTrial,
    SubjectMismatch,
    UnknownScenario,
    ScenarioMetadataMismatch,
    EvidencePolicyMismatch,
    AuthorityScopeMismatch,
    DuplicateTrialId,
    RequiredScenarioFailed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachQualificationCampaignAssessment {
    pub schema_version: u32,
    pub campaign_id: String,
    pub subject_fingerprint: u64,
    pub campaign_fingerprint: u64,
    pub total_trials: usize,
    pub total_accepted_trials: usize,
    pub scenarios: Vec<HumanoidReachScenarioAssessment>,
    pub campaign_accepted: bool,
    pub failures: Vec<HumanoidReachCampaignFailureKind>,
}

pub fn assess_humanoid_reach_qualification_campaign(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachQualificationCampaignPolicy,
    trials: &[HumanoidReachQualificationTrial],
) -> HumanoidReachQualificationCampaignAssessment {
    let subject_fingerprint = subject.fingerprint();
    let mut failures = Vec::new();
    if !policy.validate_for(subject) {
        failures.push(HumanoidReachCampaignFailureKind::InvalidPolicy);
    }
    if trials.is_empty() {
        failures.push(HumanoidReachCampaignFailureKind::EmptyCorpus);
    }

    let requirements = policy
        .required_scenarios
        .iter()
        .map(|requirement| (requirement.cell.scenario_id.clone(), requirement))
        .collect::<BTreeMap<_, _>>();

    let mut seen_trial_ids = BTreeSet::new();
    for trial in trials {
        if !trial.validate() {
            failures.push(HumanoidReachCampaignFailureKind::InvalidTrial);
            continue;
        }
        if trial.subject_fingerprint != subject_fingerprint {
            failures.push(HumanoidReachCampaignFailureKind::SubjectMismatch);
        }
        let Some(requirement) = requirements.get(&trial.scenario_id) else {
            failures.push(HumanoidReachCampaignFailureKind::UnknownScenario);
            continue;
        };
        if !requirement.cell.admits(
            trial.hand,
            trial.workspace_utilization_sq,
            &trial.perturbation_profile_id,
        ) {
            failures.push(HumanoidReachCampaignFailureKind::ScenarioMetadataMismatch);
        }
        if trial.command_policy_id != policy.required_command_policy_id
            || trial.outcome_policy_id != policy.required_outcome_policy_id
        {
            failures.push(HumanoidReachCampaignFailureKind::EvidencePolicyMismatch);
        }
        if trial.execution_purpose != policy.required_execution_purpose
            || trial.qualification_basis != HumanoidQualificationAuthorityBasis::TrialProtocol
            || trial.authority_scope_id != policy.required_authority_scope_id
        {
            failures.push(HumanoidReachCampaignFailureKind::AuthorityScopeMismatch);
        }
        if !seen_trial_ids.insert((trial.scenario_id.clone(), trial.trial_id.clone())) {
            failures.push(HumanoidReachCampaignFailureKind::DuplicateTrialId);
        }
    }

    let mut scenario_assessments = Vec::with_capacity(policy.required_scenarios.len());
    for requirement in &policy.required_scenarios {
        let cell_trials = trials
            .iter()
            .filter(|trial| trial.scenario_id == requirement.cell.scenario_id)
            .collect::<Vec<_>>();
        let total_trials = cell_trials.len();
        let accepted_trials = cell_trials.iter().filter(|trial| trial.step_accepted).count();
        let failure_rate = if total_trials == 0 {
            1.0
        } else {
            (total_trials - accepted_trials) as f64 / total_trials as f64
        };
        let distinct_spatial_goals = cell_trials
            .iter()
            .map(|trial| trial.spatial_goal_fingerprint)
            .collect::<BTreeSet<_>>()
            .len();
        let distinct_authority_receipts = cell_trials
            .iter()
            .map(|trial| trial.authority_receipt_fingerprint)
            .collect::<BTreeSet<_>>()
            .len();
        let distinct_trial_seeds = cell_trials
            .iter()
            .map(|trial| trial.trial_seed)
            .collect::<BTreeSet<_>>()
            .len();

        let mut scenario_failures = Vec::new();
        if total_trials < requirement.minimum_trials {
            scenario_failures.push(HumanoidReachScenarioAssessmentFailureKind::MissingTrials);
        }
        if failure_rate > requirement.maximum_failure_rate {
            scenario_failures.push(HumanoidReachScenarioAssessmentFailureKind::FailureRateTooHigh);
        }
        if distinct_spatial_goals < requirement.minimum_distinct_spatial_goals {
            scenario_failures.push(
                HumanoidReachScenarioAssessmentFailureKind::InsufficientDistinctSpatialGoals,
            );
        }
        if distinct_authority_receipts < requirement.minimum_distinct_authority_receipts {
            scenario_failures.push(
                HumanoidReachScenarioAssessmentFailureKind::InsufficientDistinctAuthorityReceipts,
            );
        }
        if requirement.require_unique_trial_seeds && distinct_trial_seeds != total_trials {
            scenario_failures.push(HumanoidReachScenarioAssessmentFailureKind::DuplicateTrialSeed);
        }
        let accepted = scenario_failures.is_empty();
        if !accepted {
            failures.push(HumanoidReachCampaignFailureKind::RequiredScenarioFailed);
        }
        scenario_assessments.push(HumanoidReachScenarioAssessment {
            scenario_id: requirement.cell.scenario_id.clone(),
            total_trials,
            accepted_trials,
            failure_rate,
            distinct_spatial_goals,
            distinct_authority_receipts,
            distinct_trial_seeds,
            accepted,
            failures: scenario_failures,
        });
    }

    let total_accepted_trials = trials.iter().filter(|trial| trial.step_accepted).count();
    let campaign_fingerprint = fingerprint_campaign(policy, trials);
    HumanoidReachQualificationCampaignAssessment {
        schema_version: HUMANOID_REACH_QUALIFICATION_CAMPAIGN_SCHEMA_VERSION,
        campaign_id: policy.campaign_id.clone(),
        subject_fingerprint,
        campaign_fingerprint,
        total_trials: trials.len(),
        total_accepted_trials,
        scenarios: scenario_assessments,
        campaign_accepted: failures.is_empty() && campaign_fingerprint != 0,
        failures,
    }
}

fn fingerprint_trial(trial: &HumanoidReachQualificationTrial) -> u64 {
    if trial.schema_version != HUMANOID_REACH_QUALIFICATION_CAMPAIGN_SCHEMA_VERSION
        || trial.subject_fingerprint == 0
        || !valid_id(&trial.scenario_id)
        || !valid_id(&trial.perturbation_profile_id)
        || !valid_id(&trial.trial_id)
        || !valid_id(&trial.goal_id)
        || !valid_id(&trial.command_policy_id)
        || !valid_id(&trial.outcome_policy_id)
        || !valid_id(&trial.authority_scope_id)
        || trial.validation_epoch == 0
        || trial.spatial_goal_fingerprint == 0
        || trial.authority_receipt_fingerprint == 0
        || trial.authority_scope_fingerprint == 0
        || !trial.workspace_utilization_sq.is_finite()
        || !trial.authority_effective_scale.is_finite()
        || !is_qualification_purpose(trial.execution_purpose)
        || trial.qualification_basis != HumanoidQualificationAuthorityBasis::TrialProtocol
    {
        return 0;
    }
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    feed_u64(&mut hash, trial.schema_version as u64);
    feed_u64(&mut hash, trial.subject_fingerprint);
    feed_bytes(&mut hash, trial.scenario_id.as_bytes());
    feed_bytes(&mut hash, trial.perturbation_profile_id.as_bytes());
    feed_bytes(&mut hash, trial.trial_id.as_bytes());
    feed_u64(&mut hash, trial.trial_seed);
    feed_u64(&mut hash, hand_id(trial.hand));
    feed_u64(&mut hash, trial.workspace_utilization_sq.to_bits());
    feed_u64(&mut hash, trial.validation_epoch);
    feed_bytes(&mut hash, trial.goal_id.as_bytes());
    feed_u64(&mut hash, trial.spatial_goal_fingerprint);
    feed_bytes(&mut hash, trial.command_policy_id.as_bytes());
    feed_bytes(&mut hash, trial.outcome_policy_id.as_bytes());
    feed_u64(&mut hash, trial.authority_receipt_fingerprint);
    feed_u64(&mut hash, trial.authority_scope_fingerprint);
    feed_bytes(&mut hash, trial.authority_scope_id.as_bytes());
    feed_u64(&mut hash, purpose_id(trial.execution_purpose));
    feed_u64(&mut hash, basis_id(trial.qualification_basis));
    feed_u64(&mut hash, trial.authority_effective_scale.to_bits() as u64);
    feed_u64(&mut hash, trial.step_accepted as u64);
    if hash == 0 { 1 } else { hash }
}

fn fingerprint_campaign(
    policy: &HumanoidReachQualificationCampaignPolicy,
    trials: &[HumanoidReachQualificationTrial],
) -> u64 {
    if !valid_id(&policy.campaign_id)
        || policy.subject_fingerprint == 0
        || !valid_id(&policy.required_authority_scope_id)
        || !valid_id(&policy.required_command_policy_id)
        || !valid_id(&policy.required_outcome_policy_id)
        || trials.is_empty()
    {
        return 0;
    }
    let mut sorted = trials.iter().collect::<Vec<_>>();
    sorted.sort_by(|left, right| {
        left.scenario_id
            .cmp(&right.scenario_id)
            .then(left.trial_id.cmp(&right.trial_id))
            .then(left.trial_seed.cmp(&right.trial_seed))
    });
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    feed_u64(&mut hash, policy.schema_version as u64);
    feed_bytes(&mut hash, policy.campaign_id.as_bytes());
    feed_u64(&mut hash, policy.subject_fingerprint);
    feed_u64(&mut hash, purpose_id(policy.required_execution_purpose));
    feed_bytes(&mut hash, policy.required_authority_scope_id.as_bytes());
    feed_bytes(&mut hash, policy.required_command_policy_id.as_bytes());
    feed_bytes(&mut hash, policy.required_outcome_policy_id.as_bytes());
    for requirement in &policy.required_scenarios {
        feed_bytes(&mut hash, requirement.cell.scenario_id.as_bytes());
        feed_u64(&mut hash, hand_id(requirement.cell.hand));
        feed_u64(
            &mut hash,
            requirement.cell.minimum_workspace_utilization_sq.to_bits(),
        );
        feed_u64(
            &mut hash,
            requirement.cell.maximum_workspace_utilization_sq.to_bits(),
        );
        feed_bytes(&mut hash, requirement.cell.perturbation_profile_id.as_bytes());
        feed_u64(&mut hash, requirement.minimum_trials as u64);
        feed_u64(&mut hash, requirement.maximum_failure_rate.to_bits());
        feed_u64(&mut hash, requirement.minimum_distinct_spatial_goals as u64);
        feed_u64(
            &mut hash,
            requirement.minimum_distinct_authority_receipts as u64,
        );
        feed_u64(&mut hash, requirement.require_unique_trial_seeds as u64);
    }
    for trial in sorted {
        feed_u64(&mut hash, trial.trial_fingerprint);
    }
    if hash == 0 { 1 } else { hash }
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

fn basis_id(basis: HumanoidQualificationAuthorityBasis) -> u64 {
    match basis {
        HumanoidQualificationAuthorityBasis::TrialProtocol => 1,
        HumanoidQualificationAuthorityBasis::QualifiedCapability => 2,
    }
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
            "reach-campaign-test-v1",
        )
    }

    fn requirement(
        id: &str,
        hand: HandSide,
        min_u: f64,
        max_u: f64,
    ) -> HumanoidReachScenarioRequirement {
        HumanoidReachScenarioRequirement {
            cell: HumanoidReachScenarioCell {
                scenario_id: id.into(),
                hand,
                minimum_workspace_utilization_sq: min_u,
                maximum_workspace_utilization_sq: max_u,
                perturbation_profile_id: "nominal-v1".into(),
            },
            minimum_trials: 2,
            maximum_failure_rate: 0.0,
            minimum_distinct_spatial_goals: 2,
            minimum_distinct_authority_receipts: 2,
            require_unique_trial_seeds: true,
        }
    }

    fn policy() -> HumanoidReachQualificationCampaignPolicy {
        HumanoidReachQualificationCampaignPolicy {
            schema_version: HUMANOID_REACH_QUALIFICATION_CAMPAIGN_SCHEMA_VERSION,
            campaign_id: "reach-campaign-test-v1".into(),
            subject_fingerprint: subject().fingerprint(),
            required_execution_purpose: HumanoidExecutionPurpose::SimulationQualification,
            required_authority_scope_id: "reach-sim-qualification-v1".into(),
            required_command_policy_id: "reach-command-policy-test-v1".into(),
            required_outcome_policy_id: "reach-outcome-policy-test-v1".into(),
            required_scenarios: vec![
                requirement("right-interior", HandSide::Right, 0.0, 0.49),
                requirement("left-boundary", HandSide::Left, 0.81, 1.0),
            ],
        }
    }

    fn trial(
        scenario_id: &str,
        hand: HandSide,
        utilization: f64,
        index: u64,
        accepted: bool,
    ) -> HumanoidReachQualificationTrial {
        let mut trial = HumanoidReachQualificationTrial {
            schema_version: HUMANOID_REACH_QUALIFICATION_CAMPAIGN_SCHEMA_VERSION,
            subject_fingerprint: subject().fingerprint(),
            scenario_id: scenario_id.into(),
            perturbation_profile_id: "nominal-v1".into(),
            trial_id: format!("trial-{index}"),
            trial_seed: 100 + index,
            hand,
            workspace_utilization_sq: utilization,
            validation_epoch: index + 1,
            goal_id: format!("goal-{index}"),
            spatial_goal_fingerprint: 1_000 + index,
            command_policy_id: "reach-command-policy-test-v1".into(),
            outcome_policy_id: "reach-outcome-policy-test-v1".into(),
            authority_receipt_fingerprint: 2_000 + index,
            authority_scope_fingerprint: 3_000 + index,
            authority_scope_id: "reach-sim-qualification-v1".into(),
            execution_purpose: HumanoidExecutionPurpose::SimulationQualification,
            qualification_basis: HumanoidQualificationAuthorityBasis::TrialProtocol,
            authority_effective_scale: 1.0,
            step_accepted: accepted,
            trial_fingerprint: 0,
        };
        trial.trial_fingerprint = fingerprint_trial(&trial);
        trial
    }

    #[test]
    fn complete_required_matrix_can_pass() {
        let trials = vec![
            trial("right-interior", HandSide::Right, 0.25, 1, true),
            trial("right-interior", HandSide::Right, 0.35, 2, true),
            trial("left-boundary", HandSide::Left, 0.85, 3, true),
            trial("left-boundary", HandSide::Left, 0.95, 4, true),
        ];
        let assessment = assess_humanoid_reach_qualification_campaign(&subject(), &policy(), &trials);
        assert!(assessment.campaign_accepted);
        assert!(assessment.scenarios.iter().all(|scenario| scenario.accepted));
    }

    #[test]
    fn easy_center_successes_cannot_replace_missing_boundary_coverage() {
        let trials = vec![
            trial("right-interior", HandSide::Right, 0.25, 1, true),
            trial("right-interior", HandSide::Right, 0.35, 2, true),
            trial("right-interior", HandSide::Right, 0.30, 3, true),
            trial("right-interior", HandSide::Right, 0.40, 4, true),
        ];
        let assessment = assess_humanoid_reach_qualification_campaign(&subject(), &policy(), &trials);
        assert!(!assessment.campaign_accepted);
        let boundary = assessment
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == "left-boundary")
            .unwrap();
        assert!(!boundary.accepted);
        assert!(boundary
            .failures
            .contains(&HumanoidReachScenarioAssessmentFailureKind::MissingTrials));
    }

    #[test]
    fn pooled_success_rate_cannot_hide_failing_required_cell() {
        let mut trials = Vec::new();
        for index in 1..=20 {
            trials.push(trial("right-interior", HandSide::Right, 0.3, index, true));
        }
        trials.push(trial("left-boundary", HandSide::Left, 0.85, 30, false));
        trials.push(trial("left-boundary", HandSide::Left, 0.95, 31, false));
        let assessment = assess_humanoid_reach_qualification_campaign(&subject(), &policy(), &trials);
        assert!(!assessment.campaign_accepted);
        let boundary = assessment
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == "left-boundary")
            .unwrap();
        assert_eq!(boundary.failure_rate, 1.0);
    }

    #[test]
    fn reused_authority_receipt_cannot_satisfy_independent_trial_requirement() {
        let a = trial("right-interior", HandSide::Right, 0.25, 1, true);
        let mut b = trial("right-interior", HandSide::Right, 0.35, 2, true);
        b.authority_receipt_fingerprint = a.authority_receipt_fingerprint;
        b.trial_fingerprint = fingerprint_trial(&b);
        let trials = vec![
            a,
            b,
            trial("left-boundary", HandSide::Left, 0.85, 3, true),
            trial("left-boundary", HandSide::Left, 0.95, 4, true),
        ];
        let assessment = assess_humanoid_reach_qualification_campaign(&subject(), &policy(), &trials);
        assert!(!assessment.campaign_accepted);
        let interior = assessment
            .scenarios
            .iter()
            .find(|scenario| scenario.scenario_id == "right-interior")
            .unwrap();
        assert!(interior.failures.contains(
            &HumanoidReachScenarioAssessmentFailureKind::InsufficientDistinctAuthorityReceipts
        ));
    }

    #[test]
    fn operational_authority_cannot_enter_qualification_campaign() {
        let mut trials = vec![
            trial("right-interior", HandSide::Right, 0.25, 1, true),
            trial("right-interior", HandSide::Right, 0.35, 2, true),
            trial("left-boundary", HandSide::Left, 0.85, 3, true),
            trial("left-boundary", HandSide::Left, 0.95, 4, true),
        ];
        trials[0].execution_purpose = HumanoidExecutionPurpose::Operational;
        trials[0].qualification_basis = HumanoidQualificationAuthorityBasis::QualifiedCapability;
        trials[0].trial_fingerprint = fingerprint_trial(&trials[0]);
        let assessment = assess_humanoid_reach_qualification_campaign(&subject(), &policy(), &trials);
        assert!(!assessment.campaign_accepted);
    }

    #[test]
    fn mixed_evidence_policies_fail_closed() {
        let mut trials = vec![
            trial("right-interior", HandSide::Right, 0.25, 1, true),
            trial("right-interior", HandSide::Right, 0.35, 2, true),
            trial("left-boundary", HandSide::Left, 0.85, 3, true),
            trial("left-boundary", HandSide::Left, 0.95, 4, true),
        ];
        trials[0].outcome_policy_id = "easier-policy-v2".into();
        trials[0].trial_fingerprint = fingerprint_trial(&trials[0]);
        let assessment = assess_humanoid_reach_qualification_campaign(&subject(), &policy(), &trials);
        assert!(!assessment.campaign_accepted);
        assert!(assessment.failures.contains(&HumanoidReachCampaignFailureKind::EvidencePolicyMismatch));
    }

    #[test]
    fn workspace_band_is_checked_from_admitted_geometry() {
        let cell = HumanoidReachScenarioCell {
            scenario_id: "boundary".into(),
            hand: HandSide::Left,
            minimum_workspace_utilization_sq: 0.81,
            maximum_workspace_utilization_sq: 1.0,
            perturbation_profile_id: "nominal-v1".into(),
        };
        assert!(cell.admits(HandSide::Left, 0.9, "nominal-v1"));
        assert!(!cell.admits(HandSide::Left, 0.4, "nominal-v1"));
        assert!(!cell.admits(HandSide::Right, 0.9, "nominal-v1"));
        assert!(!cell.admits(HandSide::Left, 0.9, "different-perturbation"));
    }
}
