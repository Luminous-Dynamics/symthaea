// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Preferred operational Reach promotion with SHA-256 authority lineage.
//!
//! The lower cryptographic facade binds exact protocols and behavioral corpora.
//! This facade additionally requires every contributing trial/episode step to be
//! created from `HumanoidReachAuthorityCommitment` evidence before promotion.

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::execution_authority_scope::{HumanoidExecutionPurpose, HumanoidScopedSkillAuthorityReceipt};
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_authority_committed_evidence::{
    HumanoidReachAuthorityCommittedEpisodeCase, HumanoidReachAuthorityCommittedTrial,
};
use crate::reach_cryptographic_authority::{
    HumanoidReachCryptographicEpisodeStageArtifact, HumanoidReachCryptographicOperationalArtifact,
    HumanoidReachCryptographicStepStageArtifact,
    issue_humanoid_reach_cryptographic_episode_stage,
    issue_humanoid_reach_cryptographic_operational_authority_receipt,
    issue_humanoid_reach_cryptographic_step_stage,
    promote_humanoid_reach_cryptographic_to_operational,
};
use crate::reach_episode_evidence::HumanoidReachEpisodePolicy;
use crate::reach_episode_promotion::HumanoidReachEpisodeCampaignPolicy;
use crate::reach_execution_evidence::HumanoidReachCommandEvidencePolicy;
use crate::reach_outcome_evidence::HumanoidReachOutcomeEvidencePolicy;
use crate::reach_qualification_lineage::HumanoidReachLineageCampaignPolicy;
use crate::skill_authority_receipt::HumanoidAuthoritySourceSnapshot;
use crate::skill_permit::HumanoidSkillExecutionPermit;

pub use crate::reach_cryptographic_authority::{
    HumanoidReachCryptographicAuthorityIssueFailure, HumanoidReachCryptographicOperatorApproval,
    HumanoidReachCryptographicOperationalPolicy, HumanoidReachCryptographicPromotionFailure,
    HumanoidReachCryptographicStageIssueFailure, HumanoidReachCryptographicStageRequirement,
};
pub use crate::reach_episode_promotion::{
    HumanoidReachEpisodeCampaignAssessment, HumanoidReachEpisodeCampaignFailureKind,
    HumanoidReachEpisodeCampaignPolicy as HumanoidReachAuthorityCommittedEpisodeCampaignPolicy,
    HumanoidReachEpisodeScenarioAssessment, HumanoidReachEpisodeScenarioFailureKind,
    HumanoidReachEpisodeScenarioRequirement, assess_humanoid_reach_episode_campaign,
};

pub const HUMANOID_REACH_AUTHORITY_COMMITTED_STAGE_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_AUTHORITY_COMMITTED_OPERATIONAL_SCHEMA_VERSION: u32 = 1;

pub struct HumanoidReachAuthorityCommittedStepStageArtifact {
    purpose: HumanoidExecutionPurpose,
    authority_corpus_digest: HumanoidEvidenceDigest,
    lower_artifact_digest: HumanoidEvidenceDigest,
    artifact_digest: HumanoidEvidenceDigest,
    inner: HumanoidReachCryptographicStepStageArtifact,
}

impl std::fmt::Debug for HumanoidReachAuthorityCommittedStepStageArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidReachAuthorityCommittedStepStageArtifact")
            .field("purpose", &self.purpose)
            .field("authority_corpus_digest", &self.authority_corpus_digest)
            .field("artifact_digest", &self.artifact_digest)
            .finish()
    }
}

impl HumanoidReachAuthorityCommittedStepStageArtifact {
    pub const fn purpose(&self) -> HumanoidExecutionPurpose { self.purpose }
    pub const fn authority_corpus_digest(&self) -> HumanoidEvidenceDigest { self.authority_corpus_digest }
    pub const fn artifact_digest(&self) -> HumanoidEvidenceDigest { self.artifact_digest }

    fn validate_shape(&self) -> bool {
        self.purpose.is_qualification()
            && !self.authority_corpus_digest.is_zero()
            && self.lower_artifact_digest == self.inner.artifact_digest()
            && !self.lower_artifact_digest.is_zero()
            && !self.artifact_digest.is_zero()
            && self.artifact_digest == digest_step_stage(self)
    }
}

pub struct HumanoidReachAuthorityCommittedEpisodeStageArtifact {
    purpose: HumanoidExecutionPurpose,
    authority_corpus_digest: HumanoidEvidenceDigest,
    lower_artifact_digest: HumanoidEvidenceDigest,
    artifact_digest: HumanoidEvidenceDigest,
    inner: HumanoidReachCryptographicEpisodeStageArtifact,
}

impl std::fmt::Debug for HumanoidReachAuthorityCommittedEpisodeStageArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidReachAuthorityCommittedEpisodeStageArtifact")
            .field("purpose", &self.purpose)
            .field("authority_corpus_digest", &self.authority_corpus_digest)
            .field("artifact_digest", &self.artifact_digest)
            .finish()
    }
}

impl HumanoidReachAuthorityCommittedEpisodeStageArtifact {
    pub const fn purpose(&self) -> HumanoidExecutionPurpose { self.purpose }
    pub const fn authority_corpus_digest(&self) -> HumanoidEvidenceDigest { self.authority_corpus_digest }
    pub const fn artifact_digest(&self) -> HumanoidEvidenceDigest { self.artifact_digest }

    fn validate_shape(&self) -> bool {
        self.purpose.is_qualification()
            && !self.authority_corpus_digest.is_zero()
            && self.lower_artifact_digest == self.inner.artifact_digest()
            && !self.lower_artifact_digest.is_zero()
            && !self.artifact_digest.is_zero()
            && self.artifact_digest == digest_episode_stage(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachAuthorityCommittedStageIssueFailure {
    InvalidAuthorityCorpus,
    LowerStageIssue,
    InvalidArtifact,
}

#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_authority_committed_step_stage(
    subject: &HumanoidQualificationSubject,
    step_policy: &HumanoidReachLineageCampaignPolicy,
    command_policy: &HumanoidReachCommandEvidencePolicy,
    outcome_policy: &HumanoidReachOutcomeEvidencePolicy,
    trials: &[HumanoidReachAuthorityCommittedTrial],
    issued_unix_millis: u64,
) -> Result<HumanoidReachAuthorityCommittedStepStageArtifact, HumanoidReachAuthorityCommittedStageIssueFailure> {
    let authority_corpus_digest = digest_trial_corpus(trials)
        .ok_or(HumanoidReachAuthorityCommittedStageIssueFailure::InvalidAuthorityCorpus)?;
    let lower_trials = trials.iter().map(|trial| trial.inner.clone()).collect::<Vec<_>>();
    let inner = issue_humanoid_reach_cryptographic_step_stage(
        subject, step_policy, command_policy, outcome_policy, &lower_trials, issued_unix_millis,
    )
    .map_err(|_| HumanoidReachAuthorityCommittedStageIssueFailure::LowerStageIssue)?;
    let mut artifact = HumanoidReachAuthorityCommittedStepStageArtifact {
        purpose: inner.purpose(), authority_corpus_digest,
        lower_artifact_digest: inner.artifact_digest(),
        artifact_digest: HumanoidEvidenceDigest::ZERO, inner,
    };
    artifact.artifact_digest = digest_step_stage(&artifact);
    if !artifact.validate_shape() {
        return Err(HumanoidReachAuthorityCommittedStageIssueFailure::InvalidArtifact);
    }
    Ok(artifact)
}

#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_authority_committed_episode_stage(
    subject: &HumanoidQualificationSubject,
    campaign_policy: &HumanoidReachEpisodeCampaignPolicy,
    episode_policy: &HumanoidReachEpisodePolicy,
    command_policy: &HumanoidReachCommandEvidencePolicy,
    outcome_policy: &HumanoidReachOutcomeEvidencePolicy,
    cases: &[HumanoidReachAuthorityCommittedEpisodeCase],
    issued_unix_millis: u64,
) -> Result<HumanoidReachAuthorityCommittedEpisodeStageArtifact, HumanoidReachAuthorityCommittedStageIssueFailure> {
    let authority_corpus_digest = digest_episode_corpus(cases)
        .ok_or(HumanoidReachAuthorityCommittedStageIssueFailure::InvalidAuthorityCorpus)?;
    let lower_cases = cases.iter().map(|case| case.inner.clone()).collect::<Vec<_>>();
    let inner = issue_humanoid_reach_cryptographic_episode_stage(
        subject, campaign_policy, episode_policy, command_policy, outcome_policy,
        &lower_cases, issued_unix_millis,
    )
    .map_err(|_| HumanoidReachAuthorityCommittedStageIssueFailure::LowerStageIssue)?;
    let mut artifact = HumanoidReachAuthorityCommittedEpisodeStageArtifact {
        purpose: inner.purpose(), authority_corpus_digest,
        lower_artifact_digest: inner.artifact_digest(),
        artifact_digest: HumanoidEvidenceDigest::ZERO, inner,
    };
    artifact.artifact_digest = digest_episode_stage(&artifact);
    if !artifact.validate_shape() {
        return Err(HumanoidReachAuthorityCommittedStageIssueFailure::InvalidArtifact);
    }
    Ok(artifact)
}

pub struct HumanoidReachAuthorityCommittedOperationalArtifact {
    policy_digest: HumanoidEvidenceDigest,
    simulation_step_digest: HumanoidEvidenceDigest,
    simulation_episode_digest: HumanoidEvidenceDigest,
    hil_step_digest: HumanoidEvidenceDigest,
    hil_episode_digest: HumanoidEvidenceDigest,
    physical_step_digest: HumanoidEvidenceDigest,
    physical_episode_digest: HumanoidEvidenceDigest,
    lower_artifact_digest: HumanoidEvidenceDigest,
    artifact_digest: HumanoidEvidenceDigest,
    inner: HumanoidReachCryptographicOperationalArtifact,
}

impl std::fmt::Debug for HumanoidReachAuthorityCommittedOperationalArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidReachAuthorityCommittedOperationalArtifact")
            .field("policy_digest", &self.policy_digest)
            .field("artifact_digest", &self.artifact_digest)
            .finish()
    }
}

impl HumanoidReachAuthorityCommittedOperationalArtifact {
    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest { self.policy_digest }
    pub const fn artifact_digest(&self) -> HumanoidEvidenceDigest { self.artifact_digest }

    pub fn validate_at(
        &self,
        subject: &HumanoidQualificationSubject,
        policy: &HumanoidReachCryptographicOperationalPolicy,
        now_unix_millis: u64,
    ) -> bool {
        policy.validate_for(subject)
            && self.policy_digest == policy.policy_digest()
            && self.inner.validate_at(subject, policy, now_unix_millis)
            && self.lower_artifact_digest == self.inner.artifact_digest()
            && !self.lower_artifact_digest.is_zero()
            && !self.artifact_digest.is_zero()
            && self.artifact_digest == digest_operational(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachAuthorityCommittedPromotionFailure {
    InvalidPolicy,
    InvalidStage,
    LowerPromotion,
    InvalidArtifact,
}

#[allow(clippy::too_many_arguments)]
pub fn promote_humanoid_reach_authority_committed_to_operational(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachCryptographicOperationalPolicy,
    simulation_step: &HumanoidReachAuthorityCommittedStepStageArtifact,
    simulation_episode: &HumanoidReachAuthorityCommittedEpisodeStageArtifact,
    hil_step: &HumanoidReachAuthorityCommittedStepStageArtifact,
    hil_episode: &HumanoidReachAuthorityCommittedEpisodeStageArtifact,
    physical_step: &HumanoidReachAuthorityCommittedStepStageArtifact,
    physical_episode: &HumanoidReachAuthorityCommittedEpisodeStageArtifact,
    now_unix_millis: u64,
) -> Result<HumanoidReachAuthorityCommittedOperationalArtifact, HumanoidReachAuthorityCommittedPromotionFailure> {
    if !policy.validate_for(subject) { return Err(HumanoidReachAuthorityCommittedPromotionFailure::InvalidPolicy); }
    if ![simulation_step, hil_step, physical_step].into_iter().all(|stage| stage.validate_shape())
        || ![simulation_episode, hil_episode, physical_episode].into_iter().all(|stage| stage.validate_shape())
    {
        return Err(HumanoidReachAuthorityCommittedPromotionFailure::InvalidStage);
    }
    let inner = promote_humanoid_reach_cryptographic_to_operational(
        subject, policy,
        &simulation_step.inner, &simulation_episode.inner,
        &hil_step.inner, &hil_episode.inner,
        &physical_step.inner, &physical_episode.inner,
        now_unix_millis,
    )
    .map_err(|_| HumanoidReachAuthorityCommittedPromotionFailure::LowerPromotion)?;
    let mut artifact = HumanoidReachAuthorityCommittedOperationalArtifact {
        policy_digest: policy.policy_digest(),
        simulation_step_digest: simulation_step.artifact_digest,
        simulation_episode_digest: simulation_episode.artifact_digest,
        hil_step_digest: hil_step.artifact_digest,
        hil_episode_digest: hil_episode.artifact_digest,
        physical_step_digest: physical_step.artifact_digest,
        physical_episode_digest: physical_episode.artifact_digest,
        lower_artifact_digest: inner.artifact_digest(),
        artifact_digest: HumanoidEvidenceDigest::ZERO,
        inner,
    };
    artifact.artifact_digest = digest_operational(&artifact);
    if !artifact.validate_at(subject, policy, now_unix_millis) {
        return Err(HumanoidReachAuthorityCommittedPromotionFailure::InvalidArtifact);
    }
    Ok(artifact)
}

#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_authority_committed_operational_authority_receipt(
    subject: &HumanoidQualificationSubject,
    permit: &HumanoidSkillExecutionPermit<'_>,
    qualification: &HumanoidReachAuthorityCommittedOperationalArtifact,
    policy: &HumanoidReachCryptographicOperationalPolicy,
    operator_approval: &HumanoidReachCryptographicOperatorApproval,
    physical: HumanoidAuthoritySourceSnapshot,
    epistemic: HumanoidAuthoritySourceSnapshot,
    cognitive: HumanoidAuthoritySourceSnapshot,
    now_s: f64,
    now_unix_millis: u64,
) -> Result<HumanoidScopedSkillAuthorityReceipt, HumanoidReachCryptographicAuthorityIssueFailure> {
    if !qualification.validate_at(subject, policy, now_unix_millis) {
        return Err(HumanoidReachCryptographicAuthorityIssueFailure::InvalidQualification);
    }
    issue_humanoid_reach_cryptographic_operational_authority_receipt(
        permit, &qualification.inner, policy, operator_approval,
        physical, epistemic, cognitive, now_s, now_unix_millis,
    )
}

fn digest_trial_corpus(trials: &[HumanoidReachAuthorityCommittedTrial]) -> Option<HumanoidEvidenceDigest> {
    if trials.is_empty() || trials.iter().any(|trial| !trial.validate()) { return None; }
    let mut ordered = trials.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| a.inner.trial.scenario_id.cmp(&b.inner.trial.scenario_id)
        .then(a.inner.trial.trial_id.cmp(&b.inner.trial.trial_id))
        .then(a.inner.trial.trial_seed.cmp(&b.inner.trial.trial_seed)));
    let mut h = HumanoidEvidenceHasher::new("reach.authority-trial-corpus.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITTED_STAGE_SCHEMA_VERSION).usize(ordered.len());
    for trial in ordered { h.digest(trial.binding_digest()); }
    Some(h.finish())
}

fn digest_episode_corpus(cases: &[HumanoidReachAuthorityCommittedEpisodeCase]) -> Option<HumanoidEvidenceDigest> {
    if cases.is_empty() || cases.iter().any(|case| !case.validate()) { return None; }
    let mut ordered = cases.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| a.inner.scenario_id.cmp(&b.inner.scenario_id)
        .then(a.inner.episode_id.cmp(&b.inner.episode_id))
        .then(a.inner.episode_seed.cmp(&b.inner.episode_seed)));
    let mut h = HumanoidEvidenceHasher::new("reach.authority-episode-corpus.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITTED_STAGE_SCHEMA_VERSION).usize(ordered.len());
    for case in ordered { h.digest(case.case_digest()); }
    Some(h.finish())
}

fn digest_step_stage(stage: &HumanoidReachAuthorityCommittedStepStageArtifact) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.authority-step-stage.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITTED_STAGE_SCHEMA_VERSION)
        .u64(purpose_id(stage.purpose)).digest(stage.authority_corpus_digest)
        .digest(stage.lower_artifact_digest);
    h.finish()
}

fn digest_episode_stage(stage: &HumanoidReachAuthorityCommittedEpisodeStageArtifact) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.authority-episode-stage.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITTED_STAGE_SCHEMA_VERSION)
        .u64(purpose_id(stage.purpose)).digest(stage.authority_corpus_digest)
        .digest(stage.lower_artifact_digest);
    h.finish()
}

fn digest_operational(artifact: &HumanoidReachAuthorityCommittedOperationalArtifact) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.authority-operational-artifact.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITTED_OPERATIONAL_SCHEMA_VERSION)
        .digest(artifact.policy_digest)
        .digest(artifact.simulation_step_digest).digest(artifact.simulation_episode_digest)
        .digest(artifact.hil_step_digest).digest(artifact.hil_episode_digest)
        .digest(artifact.physical_step_digest).digest(artifact.physical_episode_digest)
        .digest(artifact.lower_artifact_digest);
    h.finish()
}

fn purpose_id(purpose: HumanoidExecutionPurpose) -> u64 {
    match purpose {
        HumanoidExecutionPurpose::SimulationQualification => 1,
        HumanoidExecutionPurpose::HilQualification => 2,
        HumanoidExecutionPurpose::PhysicalQualification => 3,
        HumanoidExecutionPurpose::Operational => 4,
    }
}
