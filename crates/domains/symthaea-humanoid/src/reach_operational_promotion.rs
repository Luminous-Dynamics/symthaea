// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Preferred operational Reach promotion facade.
//!
//! Passing qualification evidence is not enough unless the evidence was produced
//! under the exact qualification protocols that operational policy intended to
//! trust. This module binds those protocols before promotion and keeps the older
//! step-only/episode-pair promotion engines crate-private implementation details.
//!
//! The public chain is therefore:
//! exact step protocol + exact episode protocol -> purpose-bound stage evidence
//! -> Simulation/HIL/Physical protocol policy -> operational capability artifact
//! -> fresh skill-atomic operational authority receipt.

use crate::execution_authority_scope::{
    HumanoidExecutionAuthorityScopeIssueFailure, HumanoidExecutionPurpose,
    HumanoidScopedSkillAuthorityReceipt, scope_verified_operational_authority_receipt,
};
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_episode_evidence::HumanoidReachEpisodePolicy;
use crate::reach_episode_promotion::{
    HumanoidReachEpisodeCampaignPolicy, HumanoidReachEpisodeQualificationCase,
    HumanoidReachEpisodeQualificationStageArtifact,
    HumanoidReachEpisodeStageIssueFailure,
    HumanoidReachEpisodeCompleteOperationalQualificationArtifact,
    HumanoidReachEpisodeCompletePromotionPolicy,
    issue_humanoid_reach_episode_stage_artifact,
    promote_humanoid_reach_episode_complete_to_operational,
};
use crate::reach_qualification_lineage::{
    HumanoidReachLineageBoundTrial, HumanoidReachLineageCampaignPolicy,
};
use crate::reach_qualification_promotion::{
    HumanoidReachQualificationStageArtifact, HumanoidReachQualificationStageIssueFailure,
    issue_humanoid_reach_qualification_stage_artifact,
};
use crate::skill_authority_receipt::{
    HumanoidAuthoritySourceSnapshot, HumanoidSkillAuthorityEvidence,
    HumanoidSkillAuthorityReceiptIssueFailure, issue_humanoid_skill_authority_receipt,
};
use crate::skill_permit::HumanoidSkillExecutionPermit;
use crate::types::HumanoidTask;

pub use crate::reach_episode_promotion::{
    HumanoidReachEpisodeCampaignAssessment, HumanoidReachEpisodeCampaignFailureKind,
    HumanoidReachEpisodeCampaignPolicy as HumanoidReachOperationalEpisodeCampaignPolicy,
    HumanoidReachEpisodeQualificationCase as HumanoidReachOperationalEpisodeCase,
    HumanoidReachEpisodeScenarioAssessment, HumanoidReachEpisodeScenarioFailureKind,
    HumanoidReachEpisodeScenarioRequirement,
    assess_humanoid_reach_episode_campaign,
};

pub const HUMANOID_REACH_PROTOCOL_STEP_STAGE_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_PROTOCOL_EPISODE_STAGE_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_OPERATIONAL_PROTOCOL_POLICY_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_OPERATIONAL_PROTOCOL_ARTIFACT_SCHEMA_VERSION: u32 = 1;

/// Step-level qualification stage bound to the complete lineage campaign policy,
/// including coverage thresholds, authority scope, evidence policy fingerprints,
/// and perturbation configuration fingerprints.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachProtocolBoundStepStageArtifact {
    pub schema_version: u32,
    pub subject_fingerprint: u64,
    pub execution_purpose: HumanoidExecutionPurpose,
    pub protocol_fingerprint: u64,
    pub inner_artifact_fingerprint: u64,
    pub issued_unix_millis: u64,
    pub artifact_fingerprint: u64,
    inner: HumanoidReachQualificationStageArtifact,
}

impl HumanoidReachProtocolBoundStepStageArtifact {
    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.schema_version == HUMANOID_REACH_PROTOCOL_STEP_STAGE_SCHEMA_VERSION
            && subject.validate()
            && subject.task == HumanoidTask::Reach
            && self.subject_fingerprint != 0
            && self.subject_fingerprint == subject.fingerprint()
            && self.execution_purpose.is_qualification()
            && self.protocol_fingerprint != 0
            && self.inner.validate()
            && self.inner.subject == *subject
            && self.inner.subject_fingerprint == self.subject_fingerprint
            && self.inner.execution_purpose == self.execution_purpose
            && self.inner.artifact_fingerprint == self.inner_artifact_fingerprint
            && self.inner.issued_unix_millis == self.issued_unix_millis
            && self.artifact_fingerprint != 0
            && self.artifact_fingerprint == fingerprint_step_stage_wrapper(self)
    }

    pub const fn inner_artifact_fingerprint(&self) -> u64 {
        self.inner_artifact_fingerprint
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachProtocolStepStageIssueFailure {
    InvalidProtocol,
    Inner(HumanoidReachQualificationStageIssueFailure),
    InvalidArtifact,
}

pub fn issue_humanoid_reach_protocol_bound_step_stage(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachLineageCampaignPolicy,
    trials: &[HumanoidReachLineageBoundTrial],
    issued_unix_millis: u64,
) -> Result<HumanoidReachProtocolBoundStepStageArtifact, HumanoidReachProtocolStepStageIssueFailure> {
    let protocol_fingerprint = fingerprint_step_protocol(subject, policy);
    if protocol_fingerprint == 0 {
        return Err(HumanoidReachProtocolStepStageIssueFailure::InvalidProtocol);
    }
    let inner = issue_humanoid_reach_qualification_stage_artifact(
        subject,
        policy,
        trials,
        issued_unix_millis,
    )
    .map_err(HumanoidReachProtocolStepStageIssueFailure::Inner)?;
    let mut artifact = HumanoidReachProtocolBoundStepStageArtifact {
        schema_version: HUMANOID_REACH_PROTOCOL_STEP_STAGE_SCHEMA_VERSION,
        subject_fingerprint: subject.fingerprint(),
        execution_purpose: inner.execution_purpose,
        protocol_fingerprint,
        inner_artifact_fingerprint: inner.artifact_fingerprint,
        issued_unix_millis: inner.issued_unix_millis,
        artifact_fingerprint: 0,
        inner,
    };
    artifact.artifact_fingerprint = fingerprint_step_stage_wrapper(&artifact);
    if !artifact.validate_for(subject) {
        return Err(HumanoidReachProtocolStepStageIssueFailure::InvalidArtifact);
    }
    Ok(artifact)
}

/// Episode-completion qualification stage bound to the complete episode campaign
/// policy plus the exact episode semantic policy.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachProtocolBoundEpisodeStageArtifact {
    pub schema_version: u32,
    pub subject_fingerprint: u64,
    pub execution_purpose: HumanoidExecutionPurpose,
    pub protocol_fingerprint: u64,
    pub inner_artifact_fingerprint: u64,
    pub issued_unix_millis: u64,
    pub artifact_fingerprint: u64,
    inner: HumanoidReachEpisodeQualificationStageArtifact,
}

impl HumanoidReachProtocolBoundEpisodeStageArtifact {
    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.schema_version == HUMANOID_REACH_PROTOCOL_EPISODE_STAGE_SCHEMA_VERSION
            && subject.validate()
            && subject.task == HumanoidTask::Reach
            && self.subject_fingerprint != 0
            && self.subject_fingerprint == subject.fingerprint()
            && self.execution_purpose.is_qualification()
            && self.protocol_fingerprint != 0
            && self.inner.validate()
            && self.inner.subject == *subject
            && self.inner.subject_fingerprint == self.subject_fingerprint
            && self.inner.execution_purpose == self.execution_purpose
            && self.inner.artifact_fingerprint == self.inner_artifact_fingerprint
            && self.inner.issued_unix_millis == self.issued_unix_millis
            && self.artifact_fingerprint != 0
            && self.artifact_fingerprint == fingerprint_episode_stage_wrapper(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachProtocolEpisodeStageIssueFailure {
    InvalidProtocol,
    Inner(HumanoidReachEpisodeStageIssueFailure),
    InvalidArtifact,
}

pub fn issue_humanoid_reach_protocol_bound_episode_stage(
    subject: &HumanoidQualificationSubject,
    campaign_policy: &HumanoidReachEpisodeCampaignPolicy,
    episode_policy: &HumanoidReachEpisodePolicy,
    cases: &[HumanoidReachEpisodeQualificationCase],
    issued_unix_millis: u64,
) -> Result<HumanoidReachProtocolBoundEpisodeStageArtifact, HumanoidReachProtocolEpisodeStageIssueFailure> {
    let protocol_fingerprint = fingerprint_episode_protocol(subject, campaign_policy, episode_policy);
    if protocol_fingerprint == 0 {
        return Err(HumanoidReachProtocolEpisodeStageIssueFailure::InvalidProtocol);
    }
    let inner = issue_humanoid_reach_episode_stage_artifact(
        subject,
        campaign_policy,
        episode_policy,
        cases,
        issued_unix_millis,
    )
    .map_err(HumanoidReachProtocolEpisodeStageIssueFailure::Inner)?;
    let mut artifact = HumanoidReachProtocolBoundEpisodeStageArtifact {
        schema_version: HUMANOID_REACH_PROTOCOL_EPISODE_STAGE_SCHEMA_VERSION,
        subject_fingerprint: subject.fingerprint(),
        execution_purpose: inner.execution_purpose,
        protocol_fingerprint,
        inner_artifact_fingerprint: inner.artifact_fingerprint,
        issued_unix_millis: inner.issued_unix_millis,
        artifact_fingerprint: 0,
        inner,
    };
    artifact.artifact_fingerprint = fingerprint_episode_stage_wrapper(&artifact);
    if !artifact.validate_for(subject) {
        return Err(HumanoidReachProtocolEpisodeStageIssueFailure::InvalidArtifact);
    }
    Ok(artifact)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HumanoidReachStageProtocolRequirement {
    pub purpose: HumanoidExecutionPurpose,
    pub step_protocol_fingerprint: u64,
    pub episode_protocol_fingerprint: u64,
}

impl HumanoidReachStageProtocolRequirement {
    fn validate_for(self, expected: HumanoidExecutionPurpose) -> bool {
        self.purpose == expected
            && expected.is_qualification()
            && self.step_protocol_fingerprint != 0
            && self.episode_protocol_fingerprint != 0
    }
}

/// Operational policy precommits to the exact Simulation/HIL/Physical protocols
/// whose evidence may be promoted. There is intentionally no Default.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachOperationalProtocolPolicy {
    pub schema_version: u32,
    pub policy_id: String,
    pub subject_fingerprint: u64,
    pub simulation: HumanoidReachStageProtocolRequirement,
    pub hil: HumanoidReachStageProtocolRequirement,
    pub physical: HumanoidReachStageProtocolRequirement,
    pub maximum_simulation_stage_age_millis: u64,
    pub maximum_hil_stage_age_millis: u64,
    pub maximum_physical_stage_age_millis: u64,
    pub maximum_stage_pair_skew_millis: u64,
    pub operational_artifact_validity_millis: u64,
}

impl HumanoidReachOperationalProtocolPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn from_exact_protocols(
        subject: &HumanoidQualificationSubject,
        policy_id: impl Into<String>,
        simulation_step: &HumanoidReachLineageCampaignPolicy,
        simulation_episode_campaign: &HumanoidReachEpisodeCampaignPolicy,
        simulation_episode: &HumanoidReachEpisodePolicy,
        hil_step: &HumanoidReachLineageCampaignPolicy,
        hil_episode_campaign: &HumanoidReachEpisodeCampaignPolicy,
        hil_episode: &HumanoidReachEpisodePolicy,
        physical_step: &HumanoidReachLineageCampaignPolicy,
        physical_episode_campaign: &HumanoidReachEpisodeCampaignPolicy,
        physical_episode: &HumanoidReachEpisodePolicy,
        maximum_simulation_stage_age_millis: u64,
        maximum_hil_stage_age_millis: u64,
        maximum_physical_stage_age_millis: u64,
        maximum_stage_pair_skew_millis: u64,
        operational_artifact_validity_millis: u64,
    ) -> Option<Self> {
        let simulation = protocol_requirement(
            subject,
            HumanoidExecutionPurpose::SimulationQualification,
            simulation_step,
            simulation_episode_campaign,
            simulation_episode,
        )?;
        let hil = protocol_requirement(
            subject,
            HumanoidExecutionPurpose::HilQualification,
            hil_step,
            hil_episode_campaign,
            hil_episode,
        )?;
        let physical = protocol_requirement(
            subject,
            HumanoidExecutionPurpose::PhysicalQualification,
            physical_step,
            physical_episode_campaign,
            physical_episode,
        )?;
        let policy = Self {
            schema_version: HUMANOID_REACH_OPERATIONAL_PROTOCOL_POLICY_SCHEMA_VERSION,
            policy_id: policy_id.into(),
            subject_fingerprint: subject.fingerprint(),
            simulation,
            hil,
            physical,
            maximum_simulation_stage_age_millis,
            maximum_hil_stage_age_millis,
            maximum_physical_stage_age_millis,
            maximum_stage_pair_skew_millis,
            operational_artifact_validity_millis,
        };
        policy.validate_for(subject).then_some(policy)
    }

    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.schema_version == HUMANOID_REACH_OPERATIONAL_PROTOCOL_POLICY_SCHEMA_VERSION
            && valid_id(&self.policy_id)
            && subject.validate()
            && subject.task == HumanoidTask::Reach
            && self.subject_fingerprint != 0
            && self.subject_fingerprint == subject.fingerprint()
            && self.simulation.validate_for(HumanoidExecutionPurpose::SimulationQualification)
            && self.hil.validate_for(HumanoidExecutionPurpose::HilQualification)
            && self.physical.validate_for(HumanoidExecutionPurpose::PhysicalQualification)
            && self.maximum_simulation_stage_age_millis > 0
            && self.maximum_hil_stage_age_millis > 0
            && self.maximum_physical_stage_age_millis > 0
            && self.maximum_stage_pair_skew_millis > 0
            && self.operational_artifact_validity_millis > 0
    }

    pub fn fingerprint(&self) -> u64 {
        if self.schema_version != HUMANOID_REACH_OPERATIONAL_PROTOCOL_POLICY_SCHEMA_VERSION
            || !valid_id(&self.policy_id)
            || self.subject_fingerprint == 0
            || !self.simulation.validate_for(HumanoidExecutionPurpose::SimulationQualification)
            || !self.hil.validate_for(HumanoidExecutionPurpose::HilQualification)
            || !self.physical.validate_for(HumanoidExecutionPurpose::PhysicalQualification)
            || self.maximum_simulation_stage_age_millis == 0
            || self.maximum_hil_stage_age_millis == 0
            || self.maximum_physical_stage_age_millis == 0
            || self.maximum_stage_pair_skew_millis == 0
            || self.operational_artifact_validity_millis == 0
        {
            return 0;
        }
        let mut hash = fnv_start();
        feed_u64(&mut hash, self.schema_version as u64);
        feed_bytes(&mut hash, self.policy_id.as_bytes());
        feed_u64(&mut hash, self.subject_fingerprint);
        for requirement in [self.simulation, self.hil, self.physical] {
            feed_u64(&mut hash, purpose_id(requirement.purpose));
            feed_u64(&mut hash, requirement.step_protocol_fingerprint);
            feed_u64(&mut hash, requirement.episode_protocol_fingerprint);
        }
        feed_u64(&mut hash, self.maximum_simulation_stage_age_millis);
        feed_u64(&mut hash, self.maximum_hil_stage_age_millis);
        feed_u64(&mut hash, self.maximum_physical_stage_age_millis);
        feed_u64(&mut hash, self.maximum_stage_pair_skew_millis);
        feed_u64(&mut hash, self.operational_artifact_validity_millis);
        nonzero(hash)
    }
}

/// Public operational capability artifact. The older internal promotion artifact
/// is retained only so its subject/stage-order/freshness checks are still reused;
/// this wrapper additionally binds the exact precommitted protocol identities.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachOperationalProtocolArtifact {
    pub schema_version: u32,
    pub subject: HumanoidQualificationSubject,
    pub subject_fingerprint: u64,
    pub policy_id: String,
    pub policy_fingerprint: u64,
    pub simulation_step_stage_fingerprint: u64,
    pub simulation_episode_stage_fingerprint: u64,
    pub hil_step_stage_fingerprint: u64,
    pub hil_episode_stage_fingerprint: u64,
    pub physical_step_stage_fingerprint: u64,
    pub physical_episode_stage_fingerprint: u64,
    pub issued_unix_millis: u64,
    pub valid_until_unix_millis: u64,
    pub artifact_fingerprint: u64,
    inner: HumanoidReachEpisodeCompleteOperationalQualificationArtifact,
}

impl HumanoidReachOperationalProtocolArtifact {
    pub fn validate_at(&self, subject: &HumanoidQualificationSubject, now_unix_millis: u64) -> bool {
        self.schema_version == HUMANOID_REACH_OPERATIONAL_PROTOCOL_ARTIFACT_SCHEMA_VERSION
            && subject.validate()
            && subject.task == HumanoidTask::Reach
            && self.subject == *subject
            && self.subject_fingerprint == subject.fingerprint()
            && self.subject_fingerprint != 0
            && valid_id(&self.policy_id)
            && self.policy_fingerprint != 0
            && self.simulation_step_stage_fingerprint != 0
            && self.simulation_episode_stage_fingerprint != 0
            && self.hil_step_stage_fingerprint != 0
            && self.hil_episode_stage_fingerprint != 0
            && self.physical_step_stage_fingerprint != 0
            && self.physical_episode_stage_fingerprint != 0
            && self.inner.validate_at(subject, now_unix_millis)
            && self.issued_unix_millis == self.inner.issued_unix_millis
            && self.valid_until_unix_millis == self.inner.valid_until_unix_millis
            && now_unix_millis >= self.issued_unix_millis
            && now_unix_millis <= self.valid_until_unix_millis
            && self.artifact_fingerprint != 0
            && self.artifact_fingerprint == fingerprint_operational_wrapper(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachOperationalProtocolPromotionFailure {
    InvalidSubject,
    InvalidPolicy,
    InvalidStepStage,
    InvalidEpisodeStage,
    ProtocolMismatch,
    StagePurposeMismatch,
    StagePairSkewTooLarge,
    InnerPromotionFailed,
    InvalidArtifact,
}

#[allow(clippy::too_many_arguments)]
pub fn promote_humanoid_reach_protocol_to_operational(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachOperationalProtocolPolicy,
    simulation_step: &HumanoidReachProtocolBoundStepStageArtifact,
    simulation_episode: &HumanoidReachProtocolBoundEpisodeStageArtifact,
    hil_step: &HumanoidReachProtocolBoundStepStageArtifact,
    hil_episode: &HumanoidReachProtocolBoundEpisodeStageArtifact,
    physical_step: &HumanoidReachProtocolBoundStepStageArtifact,
    physical_episode: &HumanoidReachProtocolBoundEpisodeStageArtifact,
    now_unix_millis: u64,
) -> Result<HumanoidReachOperationalProtocolArtifact, HumanoidReachOperationalProtocolPromotionFailure> {
    if !subject.validate() || subject.task != HumanoidTask::Reach {
        return Err(HumanoidReachOperationalProtocolPromotionFailure::InvalidSubject);
    }
    if !policy.validate_for(subject) || policy.fingerprint() == 0 {
        return Err(HumanoidReachOperationalProtocolPromotionFailure::InvalidPolicy);
    }
    for stage in [simulation_step, hil_step, physical_step] {
        if !stage.validate_for(subject) {
            return Err(HumanoidReachOperationalProtocolPromotionFailure::InvalidStepStage);
        }
    }
    for stage in [simulation_episode, hil_episode, physical_episode] {
        if !stage.validate_for(subject) {
            return Err(HumanoidReachOperationalProtocolPromotionFailure::InvalidEpisodeStage);
        }
    }

    let expected = [
        (policy.simulation, simulation_step, simulation_episode),
        (policy.hil, hil_step, hil_episode),
        (policy.physical, physical_step, physical_episode),
    ];
    for (requirement, step, episode) in expected {
        if step.execution_purpose != requirement.purpose
            || episode.execution_purpose != requirement.purpose
        {
            return Err(HumanoidReachOperationalProtocolPromotionFailure::StagePurposeMismatch);
        }
        if step.protocol_fingerprint != requirement.step_protocol_fingerprint
            || episode.protocol_fingerprint != requirement.episode_protocol_fingerprint
        {
            return Err(HumanoidReachOperationalProtocolPromotionFailure::ProtocolMismatch);
        }
        if step.issued_unix_millis.abs_diff(episode.issued_unix_millis)
            > policy.maximum_stage_pair_skew_millis
        {
            return Err(HumanoidReachOperationalProtocolPromotionFailure::StagePairSkewTooLarge);
        }
    }

    let inner_policy = HumanoidReachEpisodeCompletePromotionPolicy {
        schema_version: crate::reach_episode_promotion::HUMANOID_REACH_OPERATIONAL_EPISODE_PROMOTION_POLICY_SCHEMA_VERSION,
        policy_id: policy.policy_id.clone(),
        subject_fingerprint: policy.subject_fingerprint,
        maximum_simulation_stage_age_millis: policy.maximum_simulation_stage_age_millis,
        maximum_hil_stage_age_millis: policy.maximum_hil_stage_age_millis,
        maximum_physical_stage_age_millis: policy.maximum_physical_stage_age_millis,
        maximum_stage_pair_skew_millis: policy.maximum_stage_pair_skew_millis,
        operational_artifact_validity_millis: policy.operational_artifact_validity_millis,
    };
    let inner = promote_humanoid_reach_episode_complete_to_operational(
        subject,
        &inner_policy,
        &simulation_step.inner,
        &simulation_episode.inner,
        &hil_step.inner,
        &hil_episode.inner,
        &physical_step.inner,
        &physical_episode.inner,
        now_unix_millis,
    )
    .map_err(|_| HumanoidReachOperationalProtocolPromotionFailure::InnerPromotionFailed)?;

    let mut artifact = HumanoidReachOperationalProtocolArtifact {
        schema_version: HUMANOID_REACH_OPERATIONAL_PROTOCOL_ARTIFACT_SCHEMA_VERSION,
        subject: subject.clone(),
        subject_fingerprint: subject.fingerprint(),
        policy_id: policy.policy_id.clone(),
        policy_fingerprint: policy.fingerprint(),
        simulation_step_stage_fingerprint: simulation_step.artifact_fingerprint,
        simulation_episode_stage_fingerprint: simulation_episode.artifact_fingerprint,
        hil_step_stage_fingerprint: hil_step.artifact_fingerprint,
        hil_episode_stage_fingerprint: hil_episode.artifact_fingerprint,
        physical_step_stage_fingerprint: physical_step.artifact_fingerprint,
        physical_episode_stage_fingerprint: physical_episode.artifact_fingerprint,
        issued_unix_millis: inner.issued_unix_millis,
        valid_until_unix_millis: inner.valid_until_unix_millis,
        artifact_fingerprint: 0,
        inner,
    };
    artifact.artifact_fingerprint = fingerprint_operational_wrapper(&artifact);
    if !artifact.validate_at(subject, now_unix_millis) {
        return Err(HumanoidReachOperationalProtocolPromotionFailure::InvalidArtifact);
    }
    Ok(artifact)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachOperationalProtocolAuthorityIssueFailure {
    InvalidQualificationArtifact,
    PermitSubjectMismatch,
    TimeDomainInvalid,
    SourceValidityTooShort,
    Inner(HumanoidSkillAuthorityReceiptIssueFailure),
    Scope(HumanoidExecutionAuthorityScopeIssueFailure),
}

#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_operational_protocol_authority_receipt(
    permit: &HumanoidSkillExecutionPermit<'_>,
    qualification: &HumanoidReachOperationalProtocolArtifact,
    operator: HumanoidAuthoritySourceSnapshot,
    physical: HumanoidAuthoritySourceSnapshot,
    epistemic: HumanoidAuthoritySourceSnapshot,
    cognitive: HumanoidAuthoritySourceSnapshot,
    operational_scope_id: impl Into<String>,
    now_s: f64,
    now_unix_millis: u64,
) -> Result<HumanoidScopedSkillAuthorityReceipt, HumanoidReachOperationalProtocolAuthorityIssueFailure> {
    if !qualification.validate_at(&qualification.subject, now_unix_millis) {
        return Err(HumanoidReachOperationalProtocolAuthorityIssueFailure::InvalidQualificationArtifact);
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
        return Err(HumanoidReachOperationalProtocolAuthorityIssueFailure::PermitSubjectMismatch);
    }
    if !now_s.is_finite() || now_s < 0.0 || now_unix_millis == 0 {
        return Err(HumanoidReachOperationalProtocolAuthorityIssueFailure::TimeDomainInvalid);
    }
    let remaining_millis = qualification
        .valid_until_unix_millis
        .checked_sub(now_unix_millis)
        .ok_or(HumanoidReachOperationalProtocolAuthorityIssueFailure::InvalidQualificationArtifact)?;
    let remaining_s = remaining_millis as f64 / 1000.0;
    if !remaining_s.is_finite() || remaining_s <= 0.0 {
        return Err(HumanoidReachOperationalProtocolAuthorityIssueFailure::SourceValidityTooShort);
    }
    let qualification_valid_until_s = now_s + remaining_s;
    if !qualification_valid_until_s.is_finite() || qualification_valid_until_s <= now_s {
        return Err(HumanoidReachOperationalProtocolAuthorityIssueFailure::SourceValidityTooShort);
    }

    let qualification_source = HumanoidAuthoritySourceSnapshot {
        evidence_id: format!(
            "reach-operational-protocol-qualified:{:016x}",
            qualification.artifact_fingerprint
        ),
        scale: 1.0,
        evaluated_at_s: now_s,
        valid_until_s: qualification_valid_until_s,
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
    .map_err(HumanoidReachOperationalProtocolAuthorityIssueFailure::Inner)?;
    scope_verified_operational_authority_receipt(inner, permit, operational_scope_id, now_s)
        .map_err(HumanoidReachOperationalProtocolAuthorityIssueFailure::Scope)
}

fn protocol_requirement(
    subject: &HumanoidQualificationSubject,
    purpose: HumanoidExecutionPurpose,
    step: &HumanoidReachLineageCampaignPolicy,
    episode_campaign: &HumanoidReachEpisodeCampaignPolicy,
    episode: &HumanoidReachEpisodePolicy,
) -> Option<HumanoidReachStageProtocolRequirement> {
    if !step.validate_for(subject)
        || step.campaign.required_execution_purpose != purpose
        || !episode_campaign.validate_for(subject, episode)
        || episode_campaign.required_execution_purpose != purpose
        || episode.required_execution_purpose != purpose
    {
        return None;
    }
    let step_protocol_fingerprint = fingerprint_step_protocol(subject, step);
    let episode_protocol_fingerprint = fingerprint_episode_protocol(subject, episode_campaign, episode);
    if step_protocol_fingerprint == 0 || episode_protocol_fingerprint == 0 {
        return None;
    }
    Some(HumanoidReachStageProtocolRequirement {
        purpose,
        step_protocol_fingerprint,
        episode_protocol_fingerprint,
    })
}

fn fingerprint_step_protocol(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachLineageCampaignPolicy,
) -> u64 {
    if !policy.validate_for(subject) {
        return 0;
    }
    let campaign = &policy.campaign;
    let mut hash = fnv_start();
    feed_u64(&mut hash, policy.schema_version as u64);
    feed_u64(&mut hash, campaign.schema_version as u64);
    feed_bytes(&mut hash, campaign.campaign_id.as_bytes());
    feed_u64(&mut hash, campaign.subject_fingerprint);
    feed_u64(&mut hash, purpose_id(campaign.required_execution_purpose));
    feed_bytes(&mut hash, campaign.required_authority_scope_id.as_bytes());
    feed_bytes(&mut hash, campaign.required_command_policy_id.as_bytes());
    feed_bytes(&mut hash, campaign.required_outcome_policy_id.as_bytes());
    feed_u64(&mut hash, policy.command_policy_fingerprint);
    feed_u64(&mut hash, policy.outcome_policy_fingerprint);

    let mut requirements = campaign.required_scenarios.iter().collect::<Vec<_>>();
    requirements.sort_by(|left, right| left.cell.scenario_id.cmp(&right.cell.scenario_id));
    for requirement in requirements {
        feed_bytes(&mut hash, requirement.cell.scenario_id.as_bytes());
        feed_u64(&mut hash, hand_id(requirement.cell.hand));
        feed_u64(&mut hash, requirement.cell.minimum_workspace_utilization_sq.to_bits());
        feed_u64(&mut hash, requirement.cell.maximum_workspace_utilization_sq.to_bits());
        feed_bytes(&mut hash, requirement.cell.perturbation_profile_id.as_bytes());
        feed_u64(&mut hash, requirement.minimum_trials as u64);
        feed_u64(&mut hash, requirement.maximum_failure_rate.to_bits());
        feed_u64(&mut hash, requirement.minimum_distinct_spatial_goals as u64);
        feed_u64(&mut hash, requirement.minimum_distinct_authority_receipts as u64);
        feed_u64(&mut hash, requirement.require_unique_trial_seeds as u64);
    }

    let mut lineage = policy.scenario_lineage.iter().collect::<Vec<_>>();
    lineage.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    for item in lineage {
        feed_bytes(&mut hash, item.scenario_id.as_bytes());
        feed_bytes(&mut hash, item.perturbation_profile_id.as_bytes());
        feed_u64(&mut hash, item.perturbation_configuration_fingerprint);
    }
    nonzero(hash)
}

fn fingerprint_episode_protocol(
    subject: &HumanoidQualificationSubject,
    campaign: &HumanoidReachEpisodeCampaignPolicy,
    episode: &HumanoidReachEpisodePolicy,
) -> u64 {
    if !campaign.validate_for(subject, episode) {
        return 0;
    }
    let mut hash = fnv_start();
    feed_u64(&mut hash, campaign.fingerprint());
    feed_u64(&mut hash, episode.fingerprint());
    nonzero(hash)
}

fn fingerprint_step_stage_wrapper(stage: &HumanoidReachProtocolBoundStepStageArtifact) -> u64 {
    if stage.schema_version != HUMANOID_REACH_PROTOCOL_STEP_STAGE_SCHEMA_VERSION
        || stage.subject_fingerprint == 0
        || !stage.execution_purpose.is_qualification()
        || stage.protocol_fingerprint == 0
        || stage.inner_artifact_fingerprint == 0
        || stage.issued_unix_millis == 0
    {
        return 0;
    }
    let mut hash = fnv_start();
    feed_u64(&mut hash, stage.schema_version as u64);
    feed_u64(&mut hash, stage.subject_fingerprint);
    feed_u64(&mut hash, purpose_id(stage.execution_purpose));
    feed_u64(&mut hash, stage.protocol_fingerprint);
    feed_u64(&mut hash, stage.inner_artifact_fingerprint);
    feed_u64(&mut hash, stage.issued_unix_millis);
    nonzero(hash)
}

fn fingerprint_episode_stage_wrapper(stage: &HumanoidReachProtocolBoundEpisodeStageArtifact) -> u64 {
    if stage.schema_version != HUMANOID_REACH_PROTOCOL_EPISODE_STAGE_SCHEMA_VERSION
        || stage.subject_fingerprint == 0
        || !stage.execution_purpose.is_qualification()
        || stage.protocol_fingerprint == 0
        || stage.inner_artifact_fingerprint == 0
        || stage.issued_unix_millis == 0
    {
        return 0;
    }
    let mut hash = fnv_start();
    feed_u64(&mut hash, stage.schema_version as u64);
    feed_u64(&mut hash, stage.subject_fingerprint);
    feed_u64(&mut hash, purpose_id(stage.execution_purpose));
    feed_u64(&mut hash, stage.protocol_fingerprint);
    feed_u64(&mut hash, stage.inner_artifact_fingerprint);
    feed_u64(&mut hash, stage.issued_unix_millis);
    nonzero(hash)
}

fn fingerprint_operational_wrapper(artifact: &HumanoidReachOperationalProtocolArtifact) -> u64 {
    if artifact.schema_version != HUMANOID_REACH_OPERATIONAL_PROTOCOL_ARTIFACT_SCHEMA_VERSION
        || artifact.subject_fingerprint == 0
        || !valid_id(&artifact.policy_id)
        || artifact.policy_fingerprint == 0
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
    let mut hash = fnv_start();
    feed_u64(&mut hash, artifact.schema_version as u64);
    feed_u64(&mut hash, artifact.subject_fingerprint);
    feed_bytes(&mut hash, artifact.policy_id.as_bytes());
    feed_u64(&mut hash, artifact.policy_fingerprint);
    feed_u64(&mut hash, artifact.simulation_step_stage_fingerprint);
    feed_u64(&mut hash, artifact.simulation_episode_stage_fingerprint);
    feed_u64(&mut hash, artifact.hil_step_stage_fingerprint);
    feed_u64(&mut hash, artifact.hil_episode_stage_fingerprint);
    feed_u64(&mut hash, artifact.physical_step_stage_fingerprint);
    feed_u64(&mut hash, artifact.physical_episode_stage_fingerprint);
    feed_u64(&mut hash, artifact.issued_unix_millis);
    feed_u64(&mut hash, artifact.valid_until_unix_millis);
    feed_u64(&mut hash, artifact.inner.artifact_fingerprint);
    nonzero(hash)
}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
}

fn hand_id(hand: crate::morphology::HandSide) -> u64 {
    match hand {
        crate::morphology::HandSide::Right => 1,
        crate::morphology::HandSide::Left => 2,
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

const fn fnv_start() -> u64 {
    0xcbf2_9ce4_8422_2325u64
}

fn nonzero(hash: u64) -> u64 {
    if hash == 0 { 1 } else { hash }
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
            "operational-protocol-test-v1",
        )
    }

    #[test]
    fn stage_requirement_rejects_purpose_substitution() {
        let requirement = HumanoidReachStageProtocolRequirement {
            purpose: HumanoidExecutionPurpose::SimulationQualification,
            step_protocol_fingerprint: 1,
            episode_protocol_fingerprint: 2,
        };
        assert!(!requirement.validate_for(HumanoidExecutionPurpose::HilQualification));
    }

    #[test]
    fn operational_policy_requires_all_three_protocol_stages() {
        let policy = HumanoidReachOperationalProtocolPolicy {
            schema_version: HUMANOID_REACH_OPERATIONAL_PROTOCOL_POLICY_SCHEMA_VERSION,
            policy_id: "operational-v1".into(),
            subject_fingerprint: subject().fingerprint(),
            simulation: HumanoidReachStageProtocolRequirement {
                purpose: HumanoidExecutionPurpose::SimulationQualification,
                step_protocol_fingerprint: 1,
                episode_protocol_fingerprint: 2,
            },
            hil: HumanoidReachStageProtocolRequirement {
                purpose: HumanoidExecutionPurpose::HilQualification,
                step_protocol_fingerprint: 3,
                episode_protocol_fingerprint: 4,
            },
            physical: HumanoidReachStageProtocolRequirement {
                purpose: HumanoidExecutionPurpose::PhysicalQualification,
                step_protocol_fingerprint: 5,
                episode_protocol_fingerprint: 6,
            },
            maximum_simulation_stage_age_millis: 10_000,
            maximum_hil_stage_age_millis: 10_000,
            maximum_physical_stage_age_millis: 10_000,
            maximum_stage_pair_skew_millis: 1_000,
            operational_artifact_validity_millis: 5_000,
        };
        assert!(policy.validate_for(&subject()));
        assert_ne!(policy.fingerprint(), 0);
    }
}
