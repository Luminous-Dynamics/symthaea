// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Episode-complete promotion for operational Reach qualification.
//!
//! Step-level evidence proves individual Reach control cycles satisfy declared
//! policies. It does not prove completion of a Reach episode. Operational
//! promotion therefore requires both step-level and episode-completion evidence
//! for Simulation, HIL, and Physical stages.

use std::collections::{BTreeMap, BTreeSet};

use crate::execution_authority_scope::{
    HumanoidExecutionAuthorityScopeIssueFailure, HumanoidExecutionPurpose,
    HumanoidScopedSkillAuthorityReceipt, scope_verified_operational_authority_receipt,
};
use crate::morphology::HandSide;
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_episode_evidence::{
    HumanoidReachEpisodeAssessment, HumanoidReachEpisodePolicy,
    HumanoidReachEpisodeStepEvidence, assess_humanoid_reach_episode,
};
use crate::reach_qualification_promotion::HumanoidReachQualificationStageArtifact;
use crate::skill_authority_receipt::{
    HumanoidAuthoritySourceSnapshot, HumanoidSkillAuthorityEvidence,
    HumanoidSkillAuthorityReceiptIssueFailure, issue_humanoid_skill_authority_receipt,
};
use crate::skill_permit::HumanoidSkillExecutionPermit;
use crate::types::HumanoidTask;

pub use crate::reach_qualification_promotion::{
    HumanoidReachQualificationStageArtifact as HumanoidReachStepQualificationStageArtifact,
    HumanoidReachQualificationStageIssueFailure as HumanoidReachStepQualificationStageIssueFailure,
    issue_humanoid_reach_qualification_stage_artifact as issue_humanoid_reach_step_qualification_stage_artifact,
};

pub const HUMANOID_REACH_EPISODE_CAMPAIGN_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_EPISODE_STAGE_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_OPERATIONAL_EPISODE_QUALIFICATION_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_OPERATIONAL_EPISODE_PROMOTION_POLICY_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachEpisodeScenarioRequirement {
    pub scenario_id: String,
    pub hand: HandSide,
    pub perturbation_profile_id: String,
    pub perturbation_configuration_fingerprint: u64,
    pub minimum_episodes: usize,
    pub maximum_failure_rate: f64,
    pub minimum_distinct_goals: usize,
    pub minimum_distinct_authority_receipts: usize,
    pub require_unique_episode_seeds: bool,
}

impl HumanoidReachEpisodeScenarioRequirement {
    pub fn validate(&self) -> bool {
        valid_id(&self.scenario_id)
            && valid_id(&self.perturbation_profile_id)
            && self.perturbation_configuration_fingerprint != 0
            && self.minimum_episodes > 0
            && self.maximum_failure_rate.is_finite()
            && (0.0..=1.0).contains(&self.maximum_failure_rate)
            && self.minimum_distinct_goals > 0
            && self.minimum_distinct_goals <= self.minimum_episodes
            && self.minimum_distinct_authority_receipts > 0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachEpisodeQualificationCase {
    pub scenario_id: String,
    pub perturbation_profile_id: String,
    pub perturbation_configuration_fingerprint: u64,
    pub episode_id: String,
    pub episode_seed: u64,
    pub steps: Vec<HumanoidReachEpisodeStepEvidence>,
}

impl HumanoidReachEpisodeQualificationCase {
    pub fn validate_shape(&self) -> bool {
        valid_id(&self.scenario_id)
            && valid_id(&self.perturbation_profile_id)
            && self.perturbation_configuration_fingerprint != 0
            && valid_id(&self.episode_id)
            && !self.steps.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachEpisodeCampaignPolicy {
    pub schema_version: u32,
    pub campaign_id: String,
    pub subject_fingerprint: u64,
    pub required_execution_purpose: HumanoidExecutionPurpose,
    pub required_authority_scope_id: String,
    pub episode_policy_fingerprint: u64,
    pub required_scenarios: Vec<HumanoidReachEpisodeScenarioRequirement>,
}

impl HumanoidReachEpisodeCampaignPolicy {
    pub fn from_episode_policy(
        subject: &HumanoidQualificationSubject,
        campaign_id: impl Into<String>,
        episode_policy: &HumanoidReachEpisodePolicy,
        required_scenarios: Vec<HumanoidReachEpisodeScenarioRequirement>,
    ) -> Option<Self> {
        if !episode_policy.validate_for(subject) || episode_policy.fingerprint() == 0 {
            return None;
        }
        let policy = Self {
            schema_version: HUMANOID_REACH_EPISODE_CAMPAIGN_SCHEMA_VERSION,
            campaign_id: campaign_id.into(),
            subject_fingerprint: subject.fingerprint(),
            required_execution_purpose: episode_policy.required_execution_purpose,
            required_authority_scope_id: episode_policy.required_authority_scope_id.clone(),
            episode_policy_fingerprint: episode_policy.fingerprint(),
            required_scenarios,
        };
        policy.validate_for(subject, episode_policy).then_some(policy)
    }

    pub fn validate_for(
        &self,
        subject: &HumanoidQualificationSubject,
        episode_policy: &HumanoidReachEpisodePolicy,
    ) -> bool {
        if self.schema_version != HUMANOID_REACH_EPISODE_CAMPAIGN_SCHEMA_VERSION
            || !valid_id(&self.campaign_id)
            || !subject.validate()
            || subject.task != HumanoidTask::Reach
            || self.subject_fingerprint == 0
            || self.subject_fingerprint != subject.fingerprint()
            || !episode_policy.validate_for(subject)
            || self.required_execution_purpose != episode_policy.required_execution_purpose
            || self.required_authority_scope_id != episode_policy.required_authority_scope_id
            || self.episode_policy_fingerprint == 0
            || self.episode_policy_fingerprint != episode_policy.fingerprint()
            || self.required_scenarios.is_empty()
            || self.required_scenarios.iter().any(|item| !item.validate())
        {
            return false;
        }
        let mut ids = BTreeSet::new();
        self.required_scenarios
            .iter()
            .all(|item| ids.insert(item.scenario_id.clone()))
    }

    pub fn fingerprint(&self) -> u64 {
        if self.schema_version != HUMANOID_REACH_EPISODE_CAMPAIGN_SCHEMA_VERSION
            || !valid_id(&self.campaign_id)
            || self.subject_fingerprint == 0
            || !self.required_execution_purpose.is_qualification()
            || !valid_id(&self.required_authority_scope_id)
            || self.episode_policy_fingerprint == 0
            || self.required_scenarios.is_empty()
        {
            return 0;
        }
        let mut entries = self.required_scenarios.iter().collect::<Vec<_>>();
        entries.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        feed_u64(&mut hash, self.schema_version as u64);
        feed_bytes(&mut hash, self.campaign_id.as_bytes());
        feed_u64(&mut hash, self.subject_fingerprint);
        feed_u64(&mut hash, purpose_id(self.required_execution_purpose));
        feed_bytes(&mut hash, self.required_authority_scope_id.as_bytes());
        feed_u64(&mut hash, self.episode_policy_fingerprint);
        for item in entries {
            feed_bytes(&mut hash, item.scenario_id.as_bytes());
            feed_u64(&mut hash, hand_id(item.hand));
            feed_bytes(&mut hash, item.perturbation_profile_id.as_bytes());
            feed_u64(&mut hash, item.perturbation_configuration_fingerprint);
            feed_u64(&mut hash, item.minimum_episodes as u64);
            feed_u64(&mut hash, item.maximum_failure_rate.to_bits());
            feed_u64(&mut hash, item.minimum_distinct_goals as u64);
            feed_u64(&mut hash, item.minimum_distinct_authority_receipts as u64);
            feed_u64(&mut hash, item.require_unique_episode_seeds as u64);
        }
        if hash == 0 { 1 } else { hash }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HumanoidReachEpisodeCampaignFailureKind {
    InvalidPolicy,
    EmptyCorpus,
    InvalidCase,
    UnknownScenario,
    ScenarioMetadataMismatch,
    DuplicateEpisodeId,
    RequiredScenarioFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum HumanoidReachEpisodeScenarioFailureKind {
    MissingEpisodes,
    FailureRateTooHigh,
    InsufficientDistinctGoals,
    InsufficientDistinctAuthorityReceipts,
    DuplicateEpisodeSeed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachEpisodeScenarioAssessment {
    pub scenario_id: String,
    pub total_episodes: usize,
    pub accepted_episodes: usize,
    pub failure_rate: f64,
    pub distinct_goals: usize,
    pub distinct_authority_receipts: usize,
    pub distinct_episode_seeds: usize,
    pub accepted: bool,
    pub failures: Vec<HumanoidReachEpisodeScenarioFailureKind>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachEpisodeCampaignAssessment {
    pub schema_version: u32,
    pub campaign_id: String,
    pub subject_fingerprint: u64,
    pub campaign_policy_fingerprint: u64,
    pub corpus_fingerprint: u64,
    pub total_episodes: usize,
    pub accepted_episodes: usize,
    pub scenarios: Vec<HumanoidReachEpisodeScenarioAssessment>,
    pub campaign_accepted: bool,
    pub failures: Vec<HumanoidReachEpisodeCampaignFailureKind>,
}

pub fn assess_humanoid_reach_episode_campaign(
    subject: &HumanoidQualificationSubject,
    campaign_policy: &HumanoidReachEpisodeCampaignPolicy,
    episode_policy: &HumanoidReachEpisodePolicy,
    cases: &[HumanoidReachEpisodeQualificationCase],
) -> HumanoidReachEpisodeCampaignAssessment {
    let mut failures = Vec::new();
    if !campaign_policy.validate_for(subject, episode_policy) || campaign_policy.fingerprint() == 0 {
        failures.push(HumanoidReachEpisodeCampaignFailureKind::InvalidPolicy);
    }
    if cases.is_empty() {
        failures.push(HumanoidReachEpisodeCampaignFailureKind::EmptyCorpus);
    }

    let requirements = campaign_policy
        .required_scenarios
        .iter()
        .map(|item| (item.scenario_id.as_str(), item))
        .collect::<BTreeMap<_, _>>();
    let mut seen_ids = BTreeSet::new();
    let mut assessed = Vec::new();

    for case in cases {
        if !case.validate_shape() {
            failures.push(HumanoidReachEpisodeCampaignFailureKind::InvalidCase);
            continue;
        }
        let Some(requirement) = requirements.get(case.scenario_id.as_str()) else {
            failures.push(HumanoidReachEpisodeCampaignFailureKind::UnknownScenario);
            continue;
        };
        let episode = assess_humanoid_reach_episode(subject, episode_policy, &case.steps);
        if episode.hand != Some(requirement.hand)
            || case.perturbation_profile_id != requirement.perturbation_profile_id
            || case.perturbation_configuration_fingerprint
                != requirement.perturbation_configuration_fingerprint
        {
            failures.push(HumanoidReachEpisodeCampaignFailureKind::ScenarioMetadataMismatch);
        }
        if !seen_ids.insert((case.scenario_id.clone(), case.episode_id.clone())) {
            failures.push(HumanoidReachEpisodeCampaignFailureKind::DuplicateEpisodeId);
        }
        assessed.push((case, episode));
    }

    let mut scenario_assessments = Vec::with_capacity(campaign_policy.required_scenarios.len());
    for requirement in &campaign_policy.required_scenarios {
        let entries = assessed
            .iter()
            .filter(|(case, _)| case.scenario_id == requirement.scenario_id)
            .collect::<Vec<_>>();
        let total_episodes = entries.len();
        let accepted_episodes = entries
            .iter()
            .filter(|(_, episode)| episode.episode_accepted)
            .count();
        let failure_rate = if total_episodes == 0 {
            1.0
        } else {
            (total_episodes - accepted_episodes) as f64 / total_episodes as f64
        };
        let distinct_goals = entries
            .iter()
            .map(|(_, episode)| episode.goal_id.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        let distinct_authority_receipts = entries
            .iter()
            .flat_map(|(case, _)| case.steps.iter().map(|step| step.authority_receipt_fingerprint))
            .collect::<BTreeSet<_>>()
            .len();
        let distinct_episode_seeds = entries
            .iter()
            .map(|(case, _)| case.episode_seed)
            .collect::<BTreeSet<_>>()
            .len();

        let mut scenario_failures = Vec::new();
        if total_episodes < requirement.minimum_episodes {
            scenario_failures.push(HumanoidReachEpisodeScenarioFailureKind::MissingEpisodes);
        }
        if failure_rate > requirement.maximum_failure_rate {
            scenario_failures.push(HumanoidReachEpisodeScenarioFailureKind::FailureRateTooHigh);
        }
        if distinct_goals < requirement.minimum_distinct_goals {
            scenario_failures.push(HumanoidReachEpisodeScenarioFailureKind::InsufficientDistinctGoals);
        }
        if distinct_authority_receipts < requirement.minimum_distinct_authority_receipts {
            scenario_failures.push(
                HumanoidReachEpisodeScenarioFailureKind::InsufficientDistinctAuthorityReceipts,
            );
        }
        if requirement.require_unique_episode_seeds && distinct_episode_seeds != total_episodes {
            scenario_failures.push(HumanoidReachEpisodeScenarioFailureKind::DuplicateEpisodeSeed);
        }
        let accepted = scenario_failures.is_empty();
        if !accepted {
            failures.push(HumanoidReachEpisodeCampaignFailureKind::RequiredScenarioFailed);
        }
        scenario_assessments.push(HumanoidReachEpisodeScenarioAssessment {
            scenario_id: requirement.scenario_id.clone(),
            total_episodes,
            accepted_episodes,
            failure_rate,
            distinct_goals,
            distinct_authority_receipts,
            distinct_episode_seeds,
            accepted,
            failures: scenario_failures,
        });
    }

    failures.sort();
    failures.dedup();
    let campaign_policy_fingerprint = campaign_policy.fingerprint();
    let corpus_fingerprint = fingerprint_episode_corpus(episode_policy.fingerprint(), &assessed);
    HumanoidReachEpisodeCampaignAssessment {
        schema_version: HUMANOID_REACH_EPISODE_CAMPAIGN_SCHEMA_VERSION,
        campaign_id: campaign_policy.campaign_id.clone(),
        subject_fingerprint: subject.fingerprint(),
        campaign_policy_fingerprint,
        corpus_fingerprint,
        total_episodes: assessed.len(),
        accepted_episodes: assessed
            .iter()
            .filter(|(_, episode)| episode.episode_accepted)
            .count(),
        scenarios: scenario_assessments,
        campaign_accepted: failures.is_empty()
            && campaign_policy_fingerprint != 0
            && corpus_fingerprint != 0,
        failures,
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachEpisodeQualificationStageArtifact {
    pub schema_version: u32,
    pub subject: HumanoidQualificationSubject,
    pub subject_fingerprint: u64,
    pub execution_purpose: HumanoidExecutionPurpose,
    pub campaign_id: String,
    pub campaign_policy_fingerprint: u64,
    pub episode_policy_fingerprint: u64,
    pub corpus_fingerprint: u64,
    pub issued_unix_millis: u64,
    pub artifact_fingerprint: u64,
}

impl HumanoidReachEpisodeQualificationStageArtifact {
    pub fn validate(&self) -> bool {
        self.schema_version == HUMANOID_REACH_EPISODE_STAGE_ARTIFACT_SCHEMA_VERSION
            && self.subject.validate()
            && self.subject.task == HumanoidTask::Reach
            && self.subject_fingerprint != 0
            && self.subject_fingerprint == self.subject.fingerprint()
            && self.execution_purpose.is_qualification()
            && valid_id(&self.campaign_id)
            && self.campaign_policy_fingerprint != 0
            && self.episode_policy_fingerprint != 0
            && self.corpus_fingerprint != 0
            && self.issued_unix_millis != 0
            && self.artifact_fingerprint != 0
            && self.artifact_fingerprint == fingerprint_episode_stage(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachEpisodeStageIssueFailure {
    InvalidSubject,
    WrongCampaignPurpose,
    CampaignNotAccepted,
    InvalidIssueTime,
    InvalidArtifactFingerprint,
}

pub fn issue_humanoid_reach_episode_stage_artifact(
    subject: &HumanoidQualificationSubject,
    campaign_policy: &HumanoidReachEpisodeCampaignPolicy,
    episode_policy: &HumanoidReachEpisodePolicy,
    cases: &[HumanoidReachEpisodeQualificationCase],
    issued_unix_millis: u64,
) -> Result<HumanoidReachEpisodeQualificationStageArtifact, HumanoidReachEpisodeStageIssueFailure> {
    if !subject.validate() || subject.task != HumanoidTask::Reach {
        return Err(HumanoidReachEpisodeStageIssueFailure::InvalidSubject);
    }
    if !campaign_policy.required_execution_purpose.is_qualification() {
        return Err(HumanoidReachEpisodeStageIssueFailure::WrongCampaignPurpose);
    }
    if issued_unix_millis == 0 {
        return Err(HumanoidReachEpisodeStageIssueFailure::InvalidIssueTime);
    }
    let assessment = assess_humanoid_reach_episode_campaign(
        subject,
        campaign_policy,
        episode_policy,
        cases,
    );
    if !assessment.campaign_accepted {
        return Err(HumanoidReachEpisodeStageIssueFailure::CampaignNotAccepted);
    }
    let mut artifact = HumanoidReachEpisodeQualificationStageArtifact {
        schema_version: HUMANOID_REACH_EPISODE_STAGE_ARTIFACT_SCHEMA_VERSION,
        subject: subject.clone(),
        subject_fingerprint: subject.fingerprint(),
        execution_purpose: campaign_policy.required_execution_purpose,
        campaign_id: campaign_policy.campaign_id.clone(),
        campaign_policy_fingerprint: assessment.campaign_policy_fingerprint,
        episode_policy_fingerprint: episode_policy.fingerprint(),
        corpus_fingerprint: assessment.corpus_fingerprint,
        issued_unix_millis,
        artifact_fingerprint: 0,
    };
    artifact.artifact_fingerprint = fingerprint_episode_stage(&artifact);
    if !artifact.validate() {
        return Err(HumanoidReachEpisodeStageIssueFailure::InvalidArtifactFingerprint);
    }
    Ok(artifact)
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachEpisodeCompletePromotionPolicy {
    pub schema_version: u32,
    pub policy_id: String,
    pub subject_fingerprint: u64,
    pub maximum_simulation_stage_age_millis: u64,
    pub maximum_hil_stage_age_millis: u64,
    pub maximum_physical_stage_age_millis: u64,
    pub maximum_stage_pair_skew_millis: u64,
    pub operational_artifact_validity_millis: u64,
}

impl HumanoidReachEpisodeCompletePromotionPolicy {
    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.schema_version == HUMANOID_REACH_OPERATIONAL_EPISODE_PROMOTION_POLICY_SCHEMA_VERSION
            && valid_id(&self.policy_id)
            && subject.validate()
            && subject.task == HumanoidTask::Reach
            && self.subject_fingerprint != 0
            && self.subject_fingerprint == subject.fingerprint()
            && self.maximum_simulation_stage_age_millis > 0
            && self.maximum_hil_stage_age_millis > 0
            && self.maximum_physical_stage_age_millis > 0
            && self.maximum_stage_pair_skew_millis > 0
            && self.operational_artifact_validity_millis > 0
    }

    pub fn fingerprint(&self) -> u64 {
        if !valid_id(&self.policy_id)
            || self.subject_fingerprint == 0
            || self.maximum_simulation_stage_age_millis == 0
            || self.maximum_hil_stage_age_millis == 0
            || self.maximum_physical_stage_age_millis == 0
            || self.maximum_stage_pair_skew_millis == 0
            || self.operational_artifact_validity_millis == 0
        {
            return 0;
        }
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        feed_u64(&mut hash, self.schema_version as u64);
        feed_bytes(&mut hash, self.policy_id.as_bytes());
        feed_u64(&mut hash, self.subject_fingerprint);
        feed_u64(&mut hash, self.maximum_simulation_stage_age_millis);
        feed_u64(&mut hash, self.maximum_hil_stage_age_millis);
        feed_u64(&mut hash, self.maximum_physical_stage_age_millis);
        feed_u64(&mut hash, self.maximum_stage_pair_skew_millis);
        feed_u64(&mut hash, self.operational_artifact_validity_millis);
        if hash == 0 { 1 } else { hash }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachEpisodeCompleteOperationalQualificationArtifact {
    pub schema_version: u32,
    pub subject: HumanoidQualificationSubject,
    pub subject_fingerprint: u64,
    pub promotion_policy_id: String,
    pub promotion_policy_fingerprint: u64,
    pub simulation_step_stage_fingerprint: u64,
    pub simulation_episode_stage_fingerprint: u64,
    pub hil_step_stage_fingerprint: u64,
    pub hil_episode_stage_fingerprint: u64,
    pub physical_step_stage_fingerprint: u64,
    pub physical_episode_stage_fingerprint: u64,
    pub issued_unix_millis: u64,
    pub valid_until_unix_millis: u64,
    pub artifact_fingerprint: u64,
}

impl HumanoidReachEpisodeCompleteOperationalQualificationArtifact {
    pub fn validate_at(&self, subject: &HumanoidQualificationSubject, now_unix_millis: u64) -> bool {
        self.schema_version == HUMANOID_REACH_OPERATIONAL_EPISODE_QUALIFICATION_SCHEMA_VERSION
            && subject.validate()
            && subject.task == HumanoidTask::Reach
            && self.subject == *subject
            && self.subject_fingerprint == subject.fingerprint()
            && self.subject_fingerprint != 0
            && valid_id(&self.promotion_policy_id)
            && self.promotion_policy_fingerprint != 0
            && self.simulation_step_stage_fingerprint != 0
            && self.simulation_episode_stage_fingerprint != 0
            && self.hil_step_stage_fingerprint != 0
            && self.hil_episode_stage_fingerprint != 0
            && self.physical_step_stage_fingerprint != 0
            && self.physical_episode_stage_fingerprint != 0
            && self.issued_unix_millis != 0
            && self.valid_until_unix_millis >= self.issued_unix_millis
            && now_unix_millis >= self.issued_unix_millis
            && now_unix_millis <= self.valid_until_unix_millis
            && self.artifact_fingerprint != 0
            && self.artifact_fingerprint == fingerprint_operational_episode_artifact(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachEpisodeCompletePromotionFailure {
    InvalidSubject,
    InvalidPolicy,
    InvalidStepStage,
    InvalidEpisodeStage,
    StageSubjectMismatch,
    StagePurposeMismatch,
    StagePairSkewTooLarge,
    StageOrderInvalid,
    StageEvidenceStale,
    InvalidPromotionTime,
    ExpiryOverflow,
    InvalidArtifactFingerprint,
}

#[allow(clippy::too_many_arguments)]
pub fn promote_humanoid_reach_episode_complete_to_operational(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachEpisodeCompletePromotionPolicy,
    simulation_step: &HumanoidReachQualificationStageArtifact,
    simulation_episode: &HumanoidReachEpisodeQualificationStageArtifact,
    hil_step: &HumanoidReachQualificationStageArtifact,
    hil_episode: &HumanoidReachEpisodeQualificationStageArtifact,
    physical_step: &HumanoidReachQualificationStageArtifact,
    physical_episode: &HumanoidReachEpisodeQualificationStageArtifact,
    now_unix_millis: u64,
) -> Result<HumanoidReachEpisodeCompleteOperationalQualificationArtifact, HumanoidReachEpisodeCompletePromotionFailure> {
    if !subject.validate() || subject.task != HumanoidTask::Reach {
        return Err(HumanoidReachEpisodeCompletePromotionFailure::InvalidSubject);
    }
    if !policy.validate_for(subject) || policy.fingerprint() == 0 {
        return Err(HumanoidReachEpisodeCompletePromotionFailure::InvalidPolicy);
    }
    if now_unix_millis == 0 {
        return Err(HumanoidReachEpisodeCompletePromotionFailure::InvalidPromotionTime);
    }

    for step in [simulation_step, hil_step, physical_step] {
        if !step.validate() {
            return Err(HumanoidReachEpisodeCompletePromotionFailure::InvalidStepStage);
        }
        if step.subject != *subject || step.subject_fingerprint != subject.fingerprint() {
            return Err(HumanoidReachEpisodeCompletePromotionFailure::StageSubjectMismatch);
        }
    }
    for episode in [simulation_episode, hil_episode, physical_episode] {
        if !episode.validate() {
            return Err(HumanoidReachEpisodeCompletePromotionFailure::InvalidEpisodeStage);
        }
        if episode.subject != *subject || episode.subject_fingerprint != subject.fingerprint() {
            return Err(HumanoidReachEpisodeCompletePromotionFailure::StageSubjectMismatch);
        }
    }

    let expected = [
        HumanoidExecutionPurpose::SimulationQualification,
        HumanoidExecutionPurpose::HilQualification,
        HumanoidExecutionPurpose::PhysicalQualification,
    ];
    for ((step, episode), purpose) in [
        (simulation_step, simulation_episode),
        (hil_step, hil_episode),
        (physical_step, physical_episode),
    ]
    .into_iter()
    .zip(expected)
    {
        if step.execution_purpose != purpose || episode.execution_purpose != purpose {
            return Err(HumanoidReachEpisodeCompletePromotionFailure::StagePurposeMismatch);
        }
        if step.issued_unix_millis.abs_diff(episode.issued_unix_millis)
            > policy.maximum_stage_pair_skew_millis
        {
            return Err(HumanoidReachEpisodeCompletePromotionFailure::StagePairSkewTooLarge);
        }
    }

    let simulation_time = simulation_step
        .issued_unix_millis
        .max(simulation_episode.issued_unix_millis);
    let hil_time = hil_step.issued_unix_millis.max(hil_episode.issued_unix_millis);
    let physical_time = physical_step
        .issued_unix_millis
        .max(physical_episode.issued_unix_millis);
    if !(simulation_time <= hil_time && hil_time <= physical_time && physical_time <= now_unix_millis)
    {
        return Err(HumanoidReachEpisodeCompletePromotionFailure::StageOrderInvalid);
    }

    if stage_pair_age(
        now_unix_millis,
        simulation_step.issued_unix_millis,
        simulation_episode.issued_unix_millis,
    ) > policy.maximum_simulation_stage_age_millis
        || stage_pair_age(now_unix_millis, hil_step.issued_unix_millis, hil_episode.issued_unix_millis)
            > policy.maximum_hil_stage_age_millis
        || stage_pair_age(
            now_unix_millis,
            physical_step.issued_unix_millis,
            physical_episode.issued_unix_millis,
        ) > policy.maximum_physical_stage_age_millis
    {
        return Err(HumanoidReachEpisodeCompletePromotionFailure::StageEvidenceStale);
    }

    let valid_until_unix_millis = now_unix_millis
        .checked_add(policy.operational_artifact_validity_millis)
        .ok_or(HumanoidReachEpisodeCompletePromotionFailure::ExpiryOverflow)?;
    let mut artifact = HumanoidReachEpisodeCompleteOperationalQualificationArtifact {
        schema_version: HUMANOID_REACH_OPERATIONAL_EPISODE_QUALIFICATION_SCHEMA_VERSION,
        subject: subject.clone(),
        subject_fingerprint: subject.fingerprint(),
        promotion_policy_id: policy.policy_id.clone(),
        promotion_policy_fingerprint: policy.fingerprint(),
        simulation_step_stage_fingerprint: simulation_step.artifact_fingerprint,
        simulation_episode_stage_fingerprint: simulation_episode.artifact_fingerprint,
        hil_step_stage_fingerprint: hil_step.artifact_fingerprint,
        hil_episode_stage_fingerprint: hil_episode.artifact_fingerprint,
        physical_step_stage_fingerprint: physical_step.artifact_fingerprint,
        physical_episode_stage_fingerprint: physical_episode.artifact_fingerprint,
        issued_unix_millis: now_unix_millis,
        valid_until_unix_millis,
        artifact_fingerprint: 0,
    };
    artifact.artifact_fingerprint = fingerprint_operational_episode_artifact(&artifact);
    if !artifact.validate_at(subject, now_unix_millis) {
        return Err(HumanoidReachEpisodeCompletePromotionFailure::InvalidArtifactFingerprint);
    }
    Ok(artifact)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachEpisodeOperationalAuthorityIssueFailure {
    InvalidQualificationArtifact,
    PermitSubjectMismatch,
    TimeDomainInvalid,
    SourceValidityTooShort,
    Inner(HumanoidSkillAuthorityReceiptIssueFailure),
    Scope(HumanoidExecutionAuthorityScopeIssueFailure),
}

#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_episode_operational_authority_receipt(
    permit: &HumanoidSkillExecutionPermit<'_>,
    qualification: &HumanoidReachEpisodeCompleteOperationalQualificationArtifact,
    operator: HumanoidAuthoritySourceSnapshot,
    physical: HumanoidAuthoritySourceSnapshot,
    epistemic: HumanoidAuthoritySourceSnapshot,
    cognitive: HumanoidAuthoritySourceSnapshot,
    operational_scope_id: impl Into<String>,
    now_s: f64,
    now_unix_millis: u64,
) -> Result<HumanoidScopedSkillAuthorityReceipt, HumanoidReachEpisodeOperationalAuthorityIssueFailure> {
    if !qualification.validate_at(&qualification.subject, now_unix_millis) {
        return Err(HumanoidReachEpisodeOperationalAuthorityIssueFailure::InvalidQualificationArtifact);
    }
    let subjects = permit
        .requirements()
        .iter()
        .map(|requirement| requirement.request.subject_fingerprint)
        .collect::<Vec<_>>();
    if subjects.as_slice() != &[qualification.subject_fingerprint]
        || permit.morphology() != qualification.subject.morphology
        || permit.actuation_mode() != qualification.subject.actuation_mode
        || permit.backend_profile_id() != qualification.subject.backend_profile_id.as_str()
    {
        return Err(HumanoidReachEpisodeOperationalAuthorityIssueFailure::PermitSubjectMismatch);
    }
    if !now_s.is_finite() || now_s < 0.0 || now_unix_millis == 0 {
        return Err(HumanoidReachEpisodeOperationalAuthorityIssueFailure::TimeDomainInvalid);
    }
    let remaining_millis = qualification
        .valid_until_unix_millis
        .checked_sub(now_unix_millis)
        .ok_or(HumanoidReachEpisodeOperationalAuthorityIssueFailure::InvalidQualificationArtifact)?;
    let remaining_s = remaining_millis as f64 / 1000.0;
    if !remaining_s.is_finite() || remaining_s <= 0.0 {
        return Err(HumanoidReachEpisodeOperationalAuthorityIssueFailure::SourceValidityTooShort);
    }
    let valid_until_s = now_s + remaining_s;
    if !valid_until_s.is_finite() || valid_until_s <= now_s {
        return Err(HumanoidReachEpisodeOperationalAuthorityIssueFailure::SourceValidityTooShort);
    }

    let qualification_source = HumanoidAuthoritySourceSnapshot {
        evidence_id: format!(
            "reach-episode-qualified:{:016x}",
            qualification.artifact_fingerprint
        ),
        scale: 1.0,
        evaluated_at_s: now_s,
        valid_until_s,
    };
    let inner = issue_humanoid_skill_authority_receipt(
        permit,
        HumanoidSkillAuthorityEvidence {
            operator,
            qualification: qualification_source,
            physical,
            epistemic,
            cognitive,
        },
        now_s,
    )
    .map_err(HumanoidReachEpisodeOperationalAuthorityIssueFailure::Inner)?;
    scope_verified_operational_authority_receipt(inner, permit, operational_scope_id, now_s)
        .map_err(HumanoidReachEpisodeOperationalAuthorityIssueFailure::Scope)
}

fn fingerprint_episode_stage(artifact: &HumanoidReachEpisodeQualificationStageArtifact) -> u64 {
    if artifact.schema_version != HUMANOID_REACH_EPISODE_STAGE_ARTIFACT_SCHEMA_VERSION
        || artifact.subject_fingerprint == 0
        || !artifact.execution_purpose.is_qualification()
        || !valid_id(&artifact.campaign_id)
        || artifact.campaign_policy_fingerprint == 0
        || artifact.episode_policy_fingerprint == 0
        || artifact.corpus_fingerprint == 0
        || artifact.issued_unix_millis == 0
    {
        return 0;
    }
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    feed_u64(&mut hash, artifact.schema_version as u64);
    feed_u64(&mut hash, artifact.subject_fingerprint);
    feed_u64(&mut hash, purpose_id(artifact.execution_purpose));
    feed_bytes(&mut hash, artifact.campaign_id.as_bytes());
    feed_u64(&mut hash, artifact.campaign_policy_fingerprint);
    feed_u64(&mut hash, artifact.episode_policy_fingerprint);
    feed_u64(&mut hash, artifact.corpus_fingerprint);
    feed_u64(&mut hash, artifact.issued_unix_millis);
    if hash == 0 { 1 } else { hash }
}

fn fingerprint_operational_episode_artifact(
    artifact: &HumanoidReachEpisodeCompleteOperationalQualificationArtifact,
) -> u64 {
    if artifact.schema_version != HUMANOID_REACH_OPERATIONAL_EPISODE_QUALIFICATION_SCHEMA_VERSION
        || artifact.subject_fingerprint == 0
        || !valid_id(&artifact.promotion_policy_id)
        || artifact.promotion_policy_fingerprint == 0
        || artifact.simulation_step_stage_fingerprint == 0
        || artifact.simulation_episode_stage_fingerprint == 0
        || artifact.hil_step_stage_fingerprint == 0
        || artifact.hil_episode_stage_fingerprint == 0
        || artifact.physical_step_stage_fingerprint == 0
        || artifact.physical_episode_stage_fingerprint == 0
        || artifact.issued_unix_millis == 0
        || artifact.valid_until_unix_millis < artifact.issued_unix_millis
    {
        return 0;
    }
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    feed_u64(&mut hash, artifact.schema_version as u64);
    feed_u64(&mut hash, artifact.subject_fingerprint);
    feed_bytes(&mut hash, artifact.promotion_policy_id.as_bytes());
    feed_u64(&mut hash, artifact.promotion_policy_fingerprint);
    feed_u64(&mut hash, artifact.simulation_step_stage_fingerprint);
    feed_u64(&mut hash, artifact.simulation_episode_stage_fingerprint);
    feed_u64(&mut hash, artifact.hil_step_stage_fingerprint);
    feed_u64(&mut hash, artifact.hil_episode_stage_fingerprint);
    feed_u64(&mut hash, artifact.physical_step_stage_fingerprint);
    feed_u64(&mut hash, artifact.physical_episode_stage_fingerprint);
    feed_u64(&mut hash, artifact.issued_unix_millis);
    feed_u64(&mut hash, artifact.valid_until_unix_millis);
    if hash == 0 { 1 } else { hash }
}

fn fingerprint_episode_corpus(
    episode_policy_fingerprint: u64,
    assessed: &[(&HumanoidReachEpisodeQualificationCase, HumanoidReachEpisodeAssessment)],
) -> u64 {
    if episode_policy_fingerprint == 0 || assessed.is_empty() {
        return 0;
    }
    let mut entries = assessed.iter().collect::<Vec<_>>();
    entries.sort_by(|left, right| {
        left.0
            .scenario_id
            .cmp(&right.0.scenario_id)
            .then(left.0.episode_id.cmp(&right.0.episode_id))
            .then(left.0.episode_seed.cmp(&right.0.episode_seed))
    });
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    feed_u64(&mut hash, episode_policy_fingerprint);
    for entry in entries {
        let (case, episode) = entry;
        if episode.episode_fingerprint == 0 {
            return 0;
        }
        feed_bytes(&mut hash, case.scenario_id.as_bytes());
        feed_bytes(&mut hash, case.perturbation_profile_id.as_bytes());
        feed_u64(&mut hash, case.perturbation_configuration_fingerprint);
        feed_bytes(&mut hash, case.episode_id.as_bytes());
        feed_u64(&mut hash, case.episode_seed);
        feed_u64(&mut hash, episode.episode_fingerprint);
        feed_u64(&mut hash, episode.episode_accepted as u64);
    }
    if hash == 0 { 1 } else { hash }
}

fn stage_pair_age(now: u64, a: u64, b: u64) -> u64 {
    now.saturating_sub(a.min(b))
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
            "episode-promotion-test-v1",
        )
    }

    fn episode_policy() -> HumanoidReachEpisodePolicy {
        HumanoidReachEpisodePolicy {
            schema_version: crate::reach_episode_evidence::HUMANOID_REACH_EPISODE_EVIDENCE_SCHEMA_VERSION,
            policy_id: "episode-v1".into(),
            subject_fingerprint: subject().fingerprint(),
            required_execution_purpose: HumanoidExecutionPurpose::SimulationQualification,
            required_authority_scope_id: "sim-episode-v1".into(),
            command_policy_fingerprint: 11,
            outcome_policy_fingerprint: 22,
            maximum_steps: 8,
            maximum_episode_duration_s: 1.0,
            maximum_inter_step_gap_s: 0.1,
            maximum_target_drift_m: 0.01,
            maximum_final_error_m: 0.02,
            minimum_net_progress_m: 0.04,
            minimum_distinct_authority_receipts: 2,
            require_every_step_accepted: true,
        }
    }

    #[test]
    fn episode_campaign_policy_rejects_missing_scenarios() {
        let policy = HumanoidReachEpisodeCampaignPolicy::from_episode_policy(
            &subject(),
            "campaign-v1",
            &episode_policy(),
            vec![],
        );
        assert!(policy.is_none());
    }

    #[test]
    fn promotion_policy_fingerprint_changes_with_pair_skew() {
        let mut policy = HumanoidReachEpisodeCompletePromotionPolicy {
            schema_version: HUMANOID_REACH_OPERATIONAL_EPISODE_PROMOTION_POLICY_SCHEMA_VERSION,
            policy_id: "promotion-v1".into(),
            subject_fingerprint: subject().fingerprint(),
            maximum_simulation_stage_age_millis: 10_000,
            maximum_hil_stage_age_millis: 10_000,
            maximum_physical_stage_age_millis: 10_000,
            maximum_stage_pair_skew_millis: 1_000,
            operational_artifact_validity_millis: 5_000,
        };
        let first = policy.fingerprint();
        policy.maximum_stage_pair_skew_millis = 2_000;
        assert_ne!(first, policy.fingerprint());
    }
}
