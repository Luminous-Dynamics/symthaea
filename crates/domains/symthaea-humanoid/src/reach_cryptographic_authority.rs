// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! SHA-256 committed operational Reach qualification and authority.
//!
//! Compact 64-bit fingerprints remain inside the lower Reach qualification stack
//! as deterministic checksums. They are not used here as adversarial authority
//! identities. This facade adds domain-separated SHA-256 commitments over the
//! exact qualification protocols and evidence corpora before any operational
//! qualification source can be minted.
//!
//! The lower promotion engines remain defense-in-depth: they still enforce their
//! subject, purpose, stage-order, freshness, coverage, and FNV checksum rules.
//! This layer additionally makes protocol substitution and evidence-corpus
//! substitution require breaking SHA-256 rather than finding a 64-bit collision.

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::execution_authority_scope::{
    HumanoidExecutionAuthorityScopeIssueFailure, HumanoidExecutionPurpose,
    HumanoidQualificationAuthorityBasis, HumanoidScopedSkillAuthorityReceipt,
    scope_verified_operational_authority_receipt,
};
use crate::morphology::{HandSide, HumanoidMorphology};
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_episode_evidence::{
    HumanoidReachEpisodePolicy, HumanoidReachEpisodeStepEvidence,
};
use crate::reach_episode_promotion::{
    HumanoidReachEpisodeCampaignPolicy, HumanoidReachEpisodeQualificationCase,
};
use crate::reach_execution_evidence::HumanoidReachCommandEvidencePolicy;
use crate::reach_operational_promotion::{
    HUMANOID_REACH_OPERATIONAL_PROTOCOL_POLICY_SCHEMA_VERSION,
    HumanoidReachOperationalProtocolArtifact, HumanoidReachOperationalProtocolPolicy,
    HumanoidReachProtocolBoundEpisodeStageArtifact, HumanoidReachProtocolBoundStepStageArtifact,
    HumanoidReachStageProtocolRequirement, issue_humanoid_reach_protocol_bound_episode_stage,
    issue_humanoid_reach_protocol_bound_step_stage,
    promote_humanoid_reach_protocol_to_operational,
};
use crate::reach_outcome_evidence::HumanoidReachOutcomeEvidencePolicy;
use crate::reach_policy_identity::{
    humanoid_reach_command_policy_fingerprint, humanoid_reach_outcome_policy_fingerprint,
};
use crate::reach_qualification_lineage::{
    HumanoidReachLineageBoundTrial, HumanoidReachLineageCampaignPolicy,
};
use crate::skill_authority_receipt::{
    HumanoidAuthoritySourceSnapshot, HumanoidSkillAuthorityEvidence,
    HumanoidSkillAuthorityReceiptIssueFailure, issue_humanoid_skill_authority_receipt,
};
use crate::skill_permit::HumanoidSkillExecutionPermit;
use crate::types::{ActuationMode, HumanoidTask};

pub use crate::reach_episode_promotion::{
    HumanoidReachEpisodeCampaignAssessment, HumanoidReachEpisodeCampaignFailureKind,
    HumanoidReachEpisodeCampaignPolicy as HumanoidReachCryptographicEpisodeCampaignPolicy,
    HumanoidReachEpisodeQualificationCase as HumanoidReachCryptographicEpisodeCase,
    HumanoidReachEpisodeScenarioAssessment, HumanoidReachEpisodeScenarioFailureKind,
    HumanoidReachEpisodeScenarioRequirement, assess_humanoid_reach_episode_campaign,
};
pub use crate::reach_operational_promotion::{
    HumanoidReachOperationalProtocolPromotionFailure,
};

pub const HUMANOID_REACH_CRYPTOGRAPHIC_STAGE_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_CRYPTOGRAPHIC_POLICY_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_CRYPTOGRAPHIC_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_CRYPTOGRAPHIC_OPERATOR_APPROVAL_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HumanoidReachCryptographicStageRequirement {
    pub purpose: HumanoidExecutionPurpose,
    pub step_protocol_digest: HumanoidEvidenceDigest,
    pub episode_protocol_digest: HumanoidEvidenceDigest,
}

impl HumanoidReachCryptographicStageRequirement {
    #[allow(clippy::too_many_arguments)]
    pub fn from_exact_protocols(
        subject: &HumanoidQualificationSubject,
        purpose: HumanoidExecutionPurpose,
        step_policy: &HumanoidReachLineageCampaignPolicy,
        command_policy: &HumanoidReachCommandEvidencePolicy,
        outcome_policy: &HumanoidReachOutcomeEvidencePolicy,
        episode_campaign: &HumanoidReachEpisodeCampaignPolicy,
        episode_policy: &HumanoidReachEpisodePolicy,
    ) -> Option<Self> {
        if !purpose.is_qualification()
            || step_policy.campaign.required_execution_purpose != purpose
            || episode_campaign.required_execution_purpose != purpose
            || episode_policy.required_execution_purpose != purpose
        {
            return None;
        }
        let step_protocol_digest = digest_step_protocol(
            subject,
            step_policy,
            command_policy,
            outcome_policy,
        )?;
        let episode_protocol_digest = digest_episode_protocol(
            subject,
            episode_campaign,
            episode_policy,
            command_policy,
            outcome_policy,
        )?;
        Some(Self {
            purpose,
            step_protocol_digest,
            episode_protocol_digest,
        })
    }

    fn validate_for(self, purpose: HumanoidExecutionPurpose) -> bool {
        self.purpose == purpose
            && purpose.is_qualification()
            && !self.step_protocol_digest.is_zero()
            && !self.episode_protocol_digest.is_zero()
    }
}

/// Operational policy whose trusted qualification protocols are committed with
/// SHA-256 before evidence is promoted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HumanoidReachCryptographicOperationalPolicy {
    schema_version: u32,
    policy_id: String,
    subject_digest: HumanoidEvidenceDigest,
    simulation: HumanoidReachCryptographicStageRequirement,
    hil: HumanoidReachCryptographicStageRequirement,
    physical: HumanoidReachCryptographicStageRequirement,
    maximum_simulation_stage_age_millis: u64,
    maximum_hil_stage_age_millis: u64,
    maximum_physical_stage_age_millis: u64,
    maximum_stage_pair_skew_millis: u64,
    operational_artifact_validity_millis: u64,
    policy_digest: HumanoidEvidenceDigest,
}

impl HumanoidReachCryptographicOperationalPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn from_exact_protocols(
        subject: &HumanoidQualificationSubject,
        policy_id: impl Into<String>,
        simulation: HumanoidReachCryptographicStageRequirement,
        hil: HumanoidReachCryptographicStageRequirement,
        physical: HumanoidReachCryptographicStageRequirement,
        maximum_simulation_stage_age_millis: u64,
        maximum_hil_stage_age_millis: u64,
        maximum_physical_stage_age_millis: u64,
        maximum_stage_pair_skew_millis: u64,
        operational_artifact_validity_millis: u64,
    ) -> Option<Self> {
        let policy_id = policy_id.into();
        let subject_digest = digest_subject(subject)?;
        let mut policy = Self {
            schema_version: HUMANOID_REACH_CRYPTOGRAPHIC_POLICY_SCHEMA_VERSION,
            policy_id,
            subject_digest,
            simulation,
            hil,
            physical,
            maximum_simulation_stage_age_millis,
            maximum_hil_stage_age_millis,
            maximum_physical_stage_age_millis,
            maximum_stage_pair_skew_millis,
            operational_artifact_validity_millis,
            policy_digest: HumanoidEvidenceDigest::ZERO,
        };
        policy.policy_digest = digest_operational_policy(&policy);
        policy.validate_for(subject).then_some(policy)
    }

    pub fn policy_id(&self) -> &str {
        &self.policy_id
    }

    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest {
        self.policy_digest
    }

    pub const fn simulation(&self) -> HumanoidReachCryptographicStageRequirement {
        self.simulation
    }

    pub const fn hil(&self) -> HumanoidReachCryptographicStageRequirement {
        self.hil
    }

    pub const fn physical(&self) -> HumanoidReachCryptographicStageRequirement {
        self.physical
    }

    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.schema_version == HUMANOID_REACH_CRYPTOGRAPHIC_POLICY_SCHEMA_VERSION
            && valid_id(&self.policy_id)
            && digest_subject(subject) == Some(self.subject_digest)
            && self.simulation
                .validate_for(HumanoidExecutionPurpose::SimulationQualification)
            && self.hil.validate_for(HumanoidExecutionPurpose::HilQualification)
            && self.physical
                .validate_for(HumanoidExecutionPurpose::PhysicalQualification)
            && self.maximum_simulation_stage_age_millis > 0
            && self.maximum_hil_stage_age_millis > 0
            && self.maximum_physical_stage_age_millis > 0
            && self.maximum_stage_pair_skew_millis > 0
            && self.operational_artifact_validity_millis > 0
            && !self.policy_digest.is_zero()
            && self.policy_digest == digest_operational_policy(self)
    }
}

/// Opaque step-stage artifact. Its protocol and corpus commitments are computed
/// while the full exact policies and trial corpus are still available.
pub struct HumanoidReachCryptographicStepStageArtifact {
    purpose: HumanoidExecutionPurpose,
    subject_digest: HumanoidEvidenceDigest,
    protocol_digest: HumanoidEvidenceDigest,
    corpus_digest: HumanoidEvidenceDigest,
    issued_unix_millis: u64,
    artifact_digest: HumanoidEvidenceDigest,
    inner: HumanoidReachProtocolBoundStepStageArtifact,
}

impl std::fmt::Debug for HumanoidReachCryptographicStepStageArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidReachCryptographicStepStageArtifact")
            .field("purpose", &self.purpose)
            .field("protocol_digest", &self.protocol_digest)
            .field("corpus_digest", &self.corpus_digest)
            .field("issued_unix_millis", &self.issued_unix_millis)
            .field("artifact_digest", &self.artifact_digest)
            .finish()
    }
}

impl HumanoidReachCryptographicStepStageArtifact {
    pub const fn purpose(&self) -> HumanoidExecutionPurpose {
        self.purpose
    }

    pub const fn protocol_digest(&self) -> HumanoidEvidenceDigest {
        self.protocol_digest
    }

    pub const fn corpus_digest(&self) -> HumanoidEvidenceDigest {
        self.corpus_digest
    }

    pub const fn artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.artifact_digest
    }

    fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        digest_subject(subject) == Some(self.subject_digest)
            && self.purpose.is_qualification()
            && !self.protocol_digest.is_zero()
            && !self.corpus_digest.is_zero()
            && self.issued_unix_millis != 0
            && self.inner.validate_for(subject)
            && self.inner.execution_purpose == self.purpose
            && self.inner.issued_unix_millis == self.issued_unix_millis
            && !self.artifact_digest.is_zero()
            && self.artifact_digest == digest_step_stage_artifact(self)
    }
}

/// Opaque episode-stage artifact. Unlike the historical step summary, the episode
/// corpus contains exact world targets, errors, timestamps, policy identities and
/// authority receipts, so this digest cryptographically commits the behavioral
/// outcome evidence used for operational promotion.
pub struct HumanoidReachCryptographicEpisodeStageArtifact {
    purpose: HumanoidExecutionPurpose,
    subject_digest: HumanoidEvidenceDigest,
    protocol_digest: HumanoidEvidenceDigest,
    corpus_digest: HumanoidEvidenceDigest,
    issued_unix_millis: u64,
    artifact_digest: HumanoidEvidenceDigest,
    inner: HumanoidReachProtocolBoundEpisodeStageArtifact,
}

impl std::fmt::Debug for HumanoidReachCryptographicEpisodeStageArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidReachCryptographicEpisodeStageArtifact")
            .field("purpose", &self.purpose)
            .field("protocol_digest", &self.protocol_digest)
            .field("corpus_digest", &self.corpus_digest)
            .field("issued_unix_millis", &self.issued_unix_millis)
            .field("artifact_digest", &self.artifact_digest)
            .finish()
    }
}

impl HumanoidReachCryptographicEpisodeStageArtifact {
    pub const fn purpose(&self) -> HumanoidExecutionPurpose {
        self.purpose
    }

    pub const fn protocol_digest(&self) -> HumanoidEvidenceDigest {
        self.protocol_digest
    }

    pub const fn corpus_digest(&self) -> HumanoidEvidenceDigest {
        self.corpus_digest
    }

    pub const fn artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.artifact_digest
    }

    fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        digest_subject(subject) == Some(self.subject_digest)
            && self.purpose.is_qualification()
            && !self.protocol_digest.is_zero()
            && !self.corpus_digest.is_zero()
            && self.issued_unix_millis != 0
            && self.inner.validate_for(subject)
            && self.inner.execution_purpose == self.purpose
            && self.inner.issued_unix_millis == self.issued_unix_millis
            && !self.artifact_digest.is_zero()
            && self.artifact_digest == digest_episode_stage_artifact(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachCryptographicStageIssueFailure {
    InvalidProtocol,
    InvalidCorpus,
    LowerStageIssue,
    InvalidArtifact,
}

#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_cryptographic_step_stage(
    subject: &HumanoidQualificationSubject,
    step_policy: &HumanoidReachLineageCampaignPolicy,
    command_policy: &HumanoidReachCommandEvidencePolicy,
    outcome_policy: &HumanoidReachOutcomeEvidencePolicy,
    trials: &[HumanoidReachLineageBoundTrial],
    issued_unix_millis: u64,
) -> Result<HumanoidReachCryptographicStepStageArtifact, HumanoidReachCryptographicStageIssueFailure> {
    let protocol_digest = digest_step_protocol(subject, step_policy, command_policy, outcome_policy)
        .ok_or(HumanoidReachCryptographicStageIssueFailure::InvalidProtocol)?;
    let corpus_digest = digest_step_corpus(trials)
        .ok_or(HumanoidReachCryptographicStageIssueFailure::InvalidCorpus)?;
    let inner = issue_humanoid_reach_protocol_bound_step_stage(
        subject,
        step_policy,
        trials,
        issued_unix_millis,
    )
    .map_err(|_| HumanoidReachCryptographicStageIssueFailure::LowerStageIssue)?;
    let mut artifact = HumanoidReachCryptographicStepStageArtifact {
        purpose: step_policy.campaign.required_execution_purpose,
        subject_digest: digest_subject(subject).unwrap_or(HumanoidEvidenceDigest::ZERO),
        protocol_digest,
        corpus_digest,
        issued_unix_millis,
        artifact_digest: HumanoidEvidenceDigest::ZERO,
        inner,
    };
    artifact.artifact_digest = digest_step_stage_artifact(&artifact);
    if !artifact.validate_for(subject) {
        return Err(HumanoidReachCryptographicStageIssueFailure::InvalidArtifact);
    }
    Ok(artifact)
}

#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_cryptographic_episode_stage(
    subject: &HumanoidQualificationSubject,
    campaign_policy: &HumanoidReachEpisodeCampaignPolicy,
    episode_policy: &HumanoidReachEpisodePolicy,
    command_policy: &HumanoidReachCommandEvidencePolicy,
    outcome_policy: &HumanoidReachOutcomeEvidencePolicy,
    cases: &[HumanoidReachEpisodeQualificationCase],
    issued_unix_millis: u64,
) -> Result<HumanoidReachCryptographicEpisodeStageArtifact, HumanoidReachCryptographicStageIssueFailure> {
    let protocol_digest = digest_episode_protocol(
        subject,
        campaign_policy,
        episode_policy,
        command_policy,
        outcome_policy,
    )
    .ok_or(HumanoidReachCryptographicStageIssueFailure::InvalidProtocol)?;
    let corpus_digest = digest_episode_corpus(cases)
        .ok_or(HumanoidReachCryptographicStageIssueFailure::InvalidCorpus)?;
    let inner = issue_humanoid_reach_protocol_bound_episode_stage(
        subject,
        campaign_policy,
        episode_policy,
        cases,
        issued_unix_millis,
    )
    .map_err(|_| HumanoidReachCryptographicStageIssueFailure::LowerStageIssue)?;
    let mut artifact = HumanoidReachCryptographicEpisodeStageArtifact {
        purpose: campaign_policy.required_execution_purpose,
        subject_digest: digest_subject(subject).unwrap_or(HumanoidEvidenceDigest::ZERO),
        protocol_digest,
        corpus_digest,
        issued_unix_millis,
        artifact_digest: HumanoidEvidenceDigest::ZERO,
        inner,
    };
    artifact.artifact_digest = digest_episode_stage_artifact(&artifact);
    if !artifact.validate_for(subject) {
        return Err(HumanoidReachCryptographicStageIssueFailure::InvalidArtifact);
    }
    Ok(artifact)
}

/// Operational Reach artifact whose authorization identity is a SHA-256 digest of
/// the trusted policy plus all six protocol/corpus-bound stage artifacts.
pub struct HumanoidReachCryptographicOperationalArtifact {
    subject_digest: HumanoidEvidenceDigest,
    policy_digest: HumanoidEvidenceDigest,
    simulation_step_digest: HumanoidEvidenceDigest,
    simulation_episode_digest: HumanoidEvidenceDigest,
    hil_step_digest: HumanoidEvidenceDigest,
    hil_episode_digest: HumanoidEvidenceDigest,
    physical_step_digest: HumanoidEvidenceDigest,
    physical_episode_digest: HumanoidEvidenceDigest,
    issued_unix_millis: u64,
    valid_until_unix_millis: u64,
    artifact_digest: HumanoidEvidenceDigest,
    inner: HumanoidReachOperationalProtocolArtifact,
}

impl std::fmt::Debug for HumanoidReachCryptographicOperationalArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidReachCryptographicOperationalArtifact")
            .field("policy_digest", &self.policy_digest)
            .field("issued_unix_millis", &self.issued_unix_millis)
            .field("valid_until_unix_millis", &self.valid_until_unix_millis)
            .field("artifact_digest", &self.artifact_digest)
            .finish()
    }
}

impl HumanoidReachCryptographicOperationalArtifact {
    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest {
        self.policy_digest
    }

    pub const fn artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.artifact_digest
    }

    pub const fn valid_until_unix_millis(&self) -> u64 {
        self.valid_until_unix_millis
    }

    pub fn validate_at(
        &self,
        subject: &HumanoidQualificationSubject,
        policy: &HumanoidReachCryptographicOperationalPolicy,
        now_unix_millis: u64,
    ) -> bool {
        digest_subject(subject) == Some(self.subject_digest)
            && policy.validate_for(subject)
            && self.policy_digest == policy.policy_digest
            && self.inner.validate_at(subject, now_unix_millis)
            && self.issued_unix_millis == self.inner.issued_unix_millis
            && self.valid_until_unix_millis == self.inner.valid_until_unix_millis
            && now_unix_millis >= self.issued_unix_millis
            && now_unix_millis <= self.valid_until_unix_millis
            && !self.artifact_digest.is_zero()
            && self.artifact_digest == digest_operational_artifact(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachCryptographicPromotionFailure {
    InvalidPolicy,
    InvalidStage,
    ProtocolDigestMismatch,
    LowerPromotionFailed,
    InvalidArtifact,
}

#[allow(clippy::too_many_arguments)]
pub fn promote_humanoid_reach_cryptographic_to_operational(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachCryptographicOperationalPolicy,
    simulation_step: &HumanoidReachCryptographicStepStageArtifact,
    simulation_episode: &HumanoidReachCryptographicEpisodeStageArtifact,
    hil_step: &HumanoidReachCryptographicStepStageArtifact,
    hil_episode: &HumanoidReachCryptographicEpisodeStageArtifact,
    physical_step: &HumanoidReachCryptographicStepStageArtifact,
    physical_episode: &HumanoidReachCryptographicEpisodeStageArtifact,
    now_unix_millis: u64,
) -> Result<HumanoidReachCryptographicOperationalArtifact, HumanoidReachCryptographicPromotionFailure> {
    if !policy.validate_for(subject) {
        return Err(HumanoidReachCryptographicPromotionFailure::InvalidPolicy);
    }
    for stage in [simulation_step, hil_step, physical_step] {
        if !stage.validate_for(subject) {
            return Err(HumanoidReachCryptographicPromotionFailure::InvalidStage);
        }
    }
    for stage in [simulation_episode, hil_episode, physical_episode] {
        if !stage.validate_for(subject) {
            return Err(HumanoidReachCryptographicPromotionFailure::InvalidStage);
        }
    }
    let stage_bindings = [
        (policy.simulation, simulation_step, simulation_episode),
        (policy.hil, hil_step, hil_episode),
        (policy.physical, physical_step, physical_episode),
    ];
    for (required, step, episode) in stage_bindings {
        if step.purpose != required.purpose
            || episode.purpose != required.purpose
            || step.protocol_digest != required.step_protocol_digest
            || episode.protocol_digest != required.episode_protocol_digest
        {
            return Err(HumanoidReachCryptographicPromotionFailure::ProtocolDigestMismatch);
        }
    }

    // Reconstruct the lower policy only after cryptographic protocol identity has
    // matched. Its u64 fingerprints remain secondary defense-in-depth checks.
    let lower_policy = HumanoidReachOperationalProtocolPolicy {
        schema_version: HUMANOID_REACH_OPERATIONAL_PROTOCOL_POLICY_SCHEMA_VERSION,
        policy_id: policy.policy_id.clone(),
        subject_fingerprint: subject.fingerprint(),
        simulation: HumanoidReachStageProtocolRequirement {
            purpose: HumanoidExecutionPurpose::SimulationQualification,
            step_protocol_fingerprint: simulation_step.inner.protocol_fingerprint,
            episode_protocol_fingerprint: simulation_episode.inner.protocol_fingerprint,
        },
        hil: HumanoidReachStageProtocolRequirement {
            purpose: HumanoidExecutionPurpose::HilQualification,
            step_protocol_fingerprint: hil_step.inner.protocol_fingerprint,
            episode_protocol_fingerprint: hil_episode.inner.protocol_fingerprint,
        },
        physical: HumanoidReachStageProtocolRequirement {
            purpose: HumanoidExecutionPurpose::PhysicalQualification,
            step_protocol_fingerprint: physical_step.inner.protocol_fingerprint,
            episode_protocol_fingerprint: physical_episode.inner.protocol_fingerprint,
        },
        maximum_simulation_stage_age_millis: policy.maximum_simulation_stage_age_millis,
        maximum_hil_stage_age_millis: policy.maximum_hil_stage_age_millis,
        maximum_physical_stage_age_millis: policy.maximum_physical_stage_age_millis,
        maximum_stage_pair_skew_millis: policy.maximum_stage_pair_skew_millis,
        operational_artifact_validity_millis: policy.operational_artifact_validity_millis,
    };
    let inner = promote_humanoid_reach_protocol_to_operational(
        subject,
        &lower_policy,
        &simulation_step.inner,
        &simulation_episode.inner,
        &hil_step.inner,
        &hil_episode.inner,
        &physical_step.inner,
        &physical_episode.inner,
        now_unix_millis,
    )
    .map_err(|_| HumanoidReachCryptographicPromotionFailure::LowerPromotionFailed)?;

    let mut artifact = HumanoidReachCryptographicOperationalArtifact {
        subject_digest: digest_subject(subject).unwrap_or(HumanoidEvidenceDigest::ZERO),
        policy_digest: policy.policy_digest,
        simulation_step_digest: simulation_step.artifact_digest,
        simulation_episode_digest: simulation_episode.artifact_digest,
        hil_step_digest: hil_step.artifact_digest,
        hil_episode_digest: hil_episode.artifact_digest,
        physical_step_digest: physical_step.artifact_digest,
        physical_episode_digest: physical_episode.artifact_digest,
        issued_unix_millis: inner.issued_unix_millis,
        valid_until_unix_millis: inner.valid_until_unix_millis,
        artifact_digest: HumanoidEvidenceDigest::ZERO,
        inner,
    };
    artifact.artifact_digest = digest_operational_artifact(&artifact);
    if !artifact.validate_at(subject, policy, now_unix_millis) {
        return Err(HumanoidReachCryptographicPromotionFailure::InvalidArtifact);
    }
    Ok(artifact)
}

/// Upstream operator authority bound to the cryptographic policy commitment and
/// exact deployment scope. This is still structural provenance binding, not
/// authentication/signature verification.
pub struct HumanoidReachCryptographicOperatorApproval {
    approval_id: String,
    subject_digest: HumanoidEvidenceDigest,
    policy_digest: HumanoidEvidenceDigest,
    operational_scope_id: String,
    approved_at_s: f64,
    valid_until_s: f64,
    operator_source: HumanoidAuthoritySourceSnapshot,
    approval_digest: HumanoidEvidenceDigest,
}

impl std::fmt::Debug for HumanoidReachCryptographicOperatorApproval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidReachCryptographicOperatorApproval")
            .field("approval_id", &self.approval_id)
            .field("policy_digest", &self.policy_digest)
            .field("operational_scope_id", &self.operational_scope_id)
            .field("valid_until_s", &self.valid_until_s)
            .field("approval_digest", &self.approval_digest)
            .finish()
    }
}

impl HumanoidReachCryptographicOperatorApproval {
    pub fn bind_upstream(
        subject: &HumanoidQualificationSubject,
        policy: &HumanoidReachCryptographicOperationalPolicy,
        operational_scope_id: impl Into<String>,
        approval_id: impl Into<String>,
        operator_source: HumanoidAuthoritySourceSnapshot,
        approved_at_s: f64,
        valid_until_s: f64,
    ) -> Option<Self> {
        let operational_scope_id = operational_scope_id.into();
        let approval_id = approval_id.into();
        if !policy.validate_for(subject)
            || !valid_id(&operational_scope_id)
            || !valid_id(&approval_id)
            || !approved_at_s.is_finite()
            || approved_at_s < 0.0
            || !valid_until_s.is_finite()
            || valid_until_s < approved_at_s
            || !operator_source.validate_at(approved_at_s)
            || valid_until_s > operator_source.valid_until_s
        {
            return None;
        }
        let mut approval = Self {
            approval_id,
            subject_digest: digest_subject(subject)?,
            policy_digest: policy.policy_digest,
            operational_scope_id,
            approved_at_s,
            valid_until_s,
            operator_source,
            approval_digest: HumanoidEvidenceDigest::ZERO,
        };
        approval.approval_digest = digest_operator_approval(&approval);
        approval.validate_at(subject, policy, approved_at_s).then_some(approval)
    }

    pub fn operational_scope_id(&self) -> &str {
        &self.operational_scope_id
    }

    pub const fn approval_digest(&self) -> HumanoidEvidenceDigest {
        self.approval_digest
    }

    pub fn validate_at(
        &self,
        subject: &HumanoidQualificationSubject,
        policy: &HumanoidReachCryptographicOperationalPolicy,
        now_s: f64,
    ) -> bool {
        digest_subject(subject) == Some(self.subject_digest)
            && policy.validate_for(subject)
            && self.policy_digest == policy.policy_digest
            && valid_id(&self.approval_id)
            && valid_id(&self.operational_scope_id)
            && self.approved_at_s.is_finite()
            && self.valid_until_s.is_finite()
            && now_s.is_finite()
            && now_s >= self.approved_at_s
            && now_s <= self.valid_until_s
            && self.operator_source.validate_at(now_s)
            && self.valid_until_s <= self.operator_source.valid_until_s
            && !self.approval_digest.is_zero()
            && self.approval_digest == digest_operator_approval(self)
    }

    fn derived_operator_source(&self, now_s: f64) -> Option<HumanoidAuthoritySourceSnapshot> {
        if now_s < self.approved_at_s || now_s > self.valid_until_s {
            return None;
        }
        let source = HumanoidAuthoritySourceSnapshot {
            evidence_id: format!("reach-operator-sha256:{}", self.approval_digest.to_hex()),
            scale: self.operator_source.scale,
            evaluated_at_s: self.operator_source.evaluated_at_s.max(self.approved_at_s),
            valid_until_s: self.operator_source.valid_until_s.min(self.valid_until_s),
        };
        source.validate_at(now_s).then_some(source)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachCryptographicAuthorityIssueFailure {
    InvalidQualification,
    InvalidApproval,
    PermitSubjectMismatch,
    InvalidTime,
    SourceValidityTooShort,
    Inner(HumanoidSkillAuthorityReceiptIssueFailure),
    Scope(HumanoidExecutionAuthorityScopeIssueFailure),
}

#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_cryptographic_operational_authority_receipt(
    permit: &HumanoidSkillExecutionPermit<'_>,
    qualification: &HumanoidReachCryptographicOperationalArtifact,
    policy: &HumanoidReachCryptographicOperationalPolicy,
    operator_approval: &HumanoidReachCryptographicOperatorApproval,
    physical: HumanoidAuthoritySourceSnapshot,
    epistemic: HumanoidAuthoritySourceSnapshot,
    cognitive: HumanoidAuthoritySourceSnapshot,
    now_s: f64,
    now_unix_millis: u64,
) -> Result<HumanoidScopedSkillAuthorityReceipt, HumanoidReachCryptographicAuthorityIssueFailure> {
    if !qualification.validate_at(&qualification.inner.subject, policy, now_unix_millis) {
        return Err(HumanoidReachCryptographicAuthorityIssueFailure::InvalidQualification);
    }
    let subject = &qualification.inner.subject;
    if !operator_approval.validate_at(subject, policy, now_s) {
        return Err(HumanoidReachCryptographicAuthorityIssueFailure::InvalidApproval);
    }
    let subjects = permit
        .requirements()
        .iter()
        .map(|requirement| requirement.request.subject_fingerprint)
        .collect::<Vec<_>>();
    if subjects.as_slice() != &[subject.fingerprint()]
        || permit.morphology() != subject.morphology
        || permit.actuation_mode() != subject.actuation_mode
        || permit.backend_profile_id() != subject.backend_profile_id.as_str()
    {
        return Err(HumanoidReachCryptographicAuthorityIssueFailure::PermitSubjectMismatch);
    }
    if !now_s.is_finite() || now_s < 0.0 || now_unix_millis == 0 {
        return Err(HumanoidReachCryptographicAuthorityIssueFailure::InvalidTime);
    }
    let remaining_millis = qualification
        .valid_until_unix_millis
        .checked_sub(now_unix_millis)
        .ok_or(HumanoidReachCryptographicAuthorityIssueFailure::InvalidQualification)?;
    let remaining_s = remaining_millis as f64 / 1000.0;
    if !remaining_s.is_finite() || remaining_s <= 0.0 {
        return Err(HumanoidReachCryptographicAuthorityIssueFailure::SourceValidityTooShort);
    }
    let qualification_valid_until_s = now_s + remaining_s;
    if !qualification_valid_until_s.is_finite() || qualification_valid_until_s <= now_s {
        return Err(HumanoidReachCryptographicAuthorityIssueFailure::SourceValidityTooShort);
    }
    let operator = operator_approval
        .derived_operator_source(now_s)
        .ok_or(HumanoidReachCryptographicAuthorityIssueFailure::InvalidApproval)?;
    let qualification_source = HumanoidAuthoritySourceSnapshot {
        evidence_id: format!(
            "reach-qualified-sha256:{}",
            qualification.artifact_digest.to_hex()
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
    .map_err(HumanoidReachCryptographicAuthorityIssueFailure::Inner)?;
    scope_verified_operational_authority_receipt(
        inner,
        permit,
        operator_approval.operational_scope_id.clone(),
        now_s,
    )
    .map_err(HumanoidReachCryptographicAuthorityIssueFailure::Scope)
}

fn digest_subject(subject: &HumanoidQualificationSubject) -> Option<HumanoidEvidenceDigest> {
    if !subject.validate() || subject.task != HumanoidTask::Reach {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("reach.subject.v1");
    h.u32(subject.schema_version)
        .u64(morphology_id(subject.morphology))
        .u64(task_id(subject.task))
        .u64(actuation_mode_id(subject.actuation_mode))
        .string(&subject.backend_profile_id);
    Some(h.finish())
}

fn digest_command_policy(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachCommandEvidencePolicy,
) -> Option<HumanoidEvidenceDigest> {
    if !policy.validate_for(subject) {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("reach.command-policy.v1");
    h.digest(digest_subject(subject)?)
        .string(&policy.policy_id)
        .f64(policy.maximum_full_dynamics_age_s)
        .f64(policy.minimum_jacobian_confidence)
        .f32(policy.minimum_goal_authority_scale)
        .f64(policy.maximum_whole_body_objective_residual)
        .f64(policy.maximum_joint_utilization)
        .f64(policy.maximum_inverse_dynamics_violation)
        .bool(policy.allow_inverse_dynamics_fallback)
        .f64(policy.maximum_contact_dynamics_residual_nm)
        .f64(policy.maximum_contact_acceleration_residual)
        .f64(policy.maximum_contact_friction_utilization)
        .bool(policy.allow_contact_dynamics_fallback)
        .bool(policy.require_floating_base_model)
        .f64(policy.maximum_floating_base_dynamics_residual)
        .bool(policy.allow_floating_base_fallback)
        .usize(policy.maximum_final_safety_interventions);
    Some(h.finish())
}

fn digest_outcome_policy(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachOutcomeEvidencePolicy,
) -> Option<HumanoidEvidenceDigest> {
    if !policy.validate_for(subject) {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("reach.outcome-policy.v1");
    h.digest(digest_subject(subject)?)
        .string(&policy.policy_id)
        .f64(policy.maximum_observation_age_s)
        .f64(policy.maximum_elapsed_since_preparation_s)
        .f64(policy.maximum_post_command_error_m)
        .f64(policy.minimum_progress_m)
        .f64(policy.minimum_fractional_progress)
        .bool(policy.allow_already_within_tolerance);
    Some(h.finish())
}

fn digest_step_protocol(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachLineageCampaignPolicy,
    command_policy: &HumanoidReachCommandEvidencePolicy,
    outcome_policy: &HumanoidReachOutcomeEvidencePolicy,
) -> Option<HumanoidEvidenceDigest> {
    if !policy.validate_for(subject)
        || humanoid_reach_command_policy_fingerprint(command_policy) != policy.command_policy_fingerprint
        || humanoid_reach_outcome_policy_fingerprint(outcome_policy) != policy.outcome_policy_fingerprint
        || policy.campaign.required_command_policy_id != command_policy.policy_id
        || policy.campaign.required_outcome_policy_id != outcome_policy.policy_id
    {
        return None;
    }
    let campaign = &policy.campaign;
    let mut h = HumanoidEvidenceHasher::new("reach.step-protocol.v1");
    h.digest(digest_subject(subject)?)
        .digest(digest_command_policy(subject, command_policy)?)
        .digest(digest_outcome_policy(subject, outcome_policy)?)
        .u32(policy.schema_version)
        .u32(campaign.schema_version)
        .string(&campaign.campaign_id)
        .u64(purpose_id(campaign.required_execution_purpose))
        .string(&campaign.required_authority_scope_id)
        .usize(campaign.required_scenarios.len());
    let mut requirements = campaign.required_scenarios.iter().collect::<Vec<_>>();
    requirements.sort_by(|a, b| a.cell.scenario_id.cmp(&b.cell.scenario_id));
    for requirement in requirements {
        h.string(&requirement.cell.scenario_id)
            .u64(hand_id(requirement.cell.hand))
            .f64(requirement.cell.minimum_workspace_utilization_sq)
            .f64(requirement.cell.maximum_workspace_utilization_sq)
            .string(&requirement.cell.perturbation_profile_id)
            .usize(requirement.minimum_trials)
            .f64(requirement.maximum_failure_rate)
            .usize(requirement.minimum_distinct_spatial_goals)
            .usize(requirement.minimum_distinct_authority_receipts)
            .bool(requirement.require_unique_trial_seeds);
    }
    let mut lineage = policy.scenario_lineage.iter().collect::<Vec<_>>();
    lineage.sort_by(|a, b| a.scenario_id.cmp(&b.scenario_id));
    h.usize(lineage.len());
    for item in lineage {
        h.string(&item.scenario_id)
            .string(&item.perturbation_profile_id)
            .u64(item.perturbation_configuration_fingerprint);
    }
    Some(h.finish())
}

fn digest_episode_protocol(
    subject: &HumanoidQualificationSubject,
    campaign: &HumanoidReachEpisodeCampaignPolicy,
    episode: &HumanoidReachEpisodePolicy,
    command_policy: &HumanoidReachCommandEvidencePolicy,
    outcome_policy: &HumanoidReachOutcomeEvidencePolicy,
) -> Option<HumanoidEvidenceDigest> {
    if !campaign.validate_for(subject, episode)
        || humanoid_reach_command_policy_fingerprint(command_policy) != episode.command_policy_fingerprint
        || humanoid_reach_outcome_policy_fingerprint(outcome_policy) != episode.outcome_policy_fingerprint
    {
        return None;
    }
    let mut h = HumanoidEvidenceHasher::new("reach.episode-protocol.v1");
    h.digest(digest_subject(subject)?)
        .digest(digest_command_policy(subject, command_policy)?)
        .digest(digest_outcome_policy(subject, outcome_policy)?)
        .u32(campaign.schema_version)
        .string(&campaign.campaign_id)
        .u64(purpose_id(campaign.required_execution_purpose))
        .string(&campaign.required_authority_scope_id)
        .u32(episode.schema_version)
        .string(&episode.policy_id)
        .usize(episode.maximum_steps)
        .f64(episode.maximum_episode_duration_s)
        .f64(episode.maximum_inter_step_gap_s)
        .f64(episode.maximum_target_drift_m)
        .f64(episode.maximum_final_error_m)
        .f64(episode.minimum_net_progress_m)
        .usize(episode.minimum_distinct_authority_receipts)
        .bool(episode.require_every_step_accepted)
        .usize(campaign.required_scenarios.len());
    let mut scenarios = campaign.required_scenarios.iter().collect::<Vec<_>>();
    scenarios.sort_by(|a, b| a.scenario_id.cmp(&b.scenario_id));
    for item in scenarios {
        h.string(&item.scenario_id)
            .u64(hand_id(item.hand))
            .string(&item.perturbation_profile_id)
            .u64(item.perturbation_configuration_fingerprint)
            .usize(item.minimum_episodes)
            .f64(item.maximum_failure_rate)
            .usize(item.minimum_distinct_goals)
            .usize(item.minimum_distinct_authority_receipts)
            .bool(item.require_unique_episode_seeds);
    }
    Some(h.finish())
}

fn digest_step_corpus(trials: &[HumanoidReachLineageBoundTrial]) -> Option<HumanoidEvidenceDigest> {
    if trials.is_empty() || trials.iter().any(|trial| !trial.validate()) {
        return None;
    }
    let mut ordered = trials.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| {
        a.trial
            .scenario_id
            .cmp(&b.trial.scenario_id)
            .then(a.trial.trial_id.cmp(&b.trial.trial_id))
            .then(a.trial.trial_seed.cmp(&b.trial.trial_seed))
    });
    let mut h = HumanoidEvidenceHasher::new("reach.step-corpus.v1");
    h.usize(ordered.len());
    for bound in ordered {
        let trial = &bound.trial;
        h.u32(bound.schema_version)
            .u32(trial.schema_version)
            .u64(trial.subject_fingerprint)
            .string(&trial.scenario_id)
            .string(&trial.perturbation_profile_id)
            .string(&trial.trial_id)
            .u64(trial.trial_seed)
            .u64(hand_id(trial.hand))
            .f64(trial.workspace_utilization_sq)
            .u64(trial.validation_epoch)
            .string(&trial.goal_id)
            .u64(trial.spatial_goal_fingerprint)
            .string(&trial.command_policy_id)
            .string(&trial.outcome_policy_id)
            .u64(trial.authority_receipt_fingerprint)
            .u64(trial.authority_scope_fingerprint)
            .string(&trial.authority_scope_id)
            .u64(purpose_id(trial.execution_purpose))
            .u64(basis_id(trial.qualification_basis))
            .f32(trial.authority_effective_scale)
            .bool(trial.step_accepted)
            .u64(bound.command_policy_fingerprint)
            .u64(bound.outcome_policy_fingerprint)
            .u64(bound.perturbation_configuration_fingerprint);
    }
    Some(h.finish())
}

fn digest_episode_corpus(cases: &[HumanoidReachEpisodeQualificationCase]) -> Option<HumanoidEvidenceDigest> {
    if cases.is_empty() || cases.iter().any(|case| !case.validate_shape()) {
        return None;
    }
    let mut ordered = cases.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| {
        a.scenario_id
            .cmp(&b.scenario_id)
            .then(a.episode_id.cmp(&b.episode_id))
            .then(a.episode_seed.cmp(&b.episode_seed))
    });
    let mut h = HumanoidEvidenceHasher::new("reach.episode-corpus.v1");
    h.usize(ordered.len());
    for case in ordered {
        h.string(&case.scenario_id)
            .string(&case.perturbation_profile_id)
            .u64(case.perturbation_configuration_fingerprint)
            .string(&case.episode_id)
            .u64(case.episode_seed)
            .usize(case.steps.len());
        for step in &case.steps {
            if !step.validate() {
                return None;
            }
            hash_episode_step(&mut h, step);
        }
    }
    Some(h.finish())
}

fn hash_episode_step(h: &mut HumanoidEvidenceHasher, step: &HumanoidReachEpisodeStepEvidence) {
    h.u32(step.schema_version)
        .u64(step.subject_fingerprint)
        .u64(step.validation_epoch)
        .string(&step.goal_id)
        .u64(step.spatial_goal_fingerprint)
        .u64(hand_id(step.hand));
    for value in step.target_world_m {
        h.f64(value);
    }
    h.f64(step.prepared_at_s)
        .f64(step.observed_at_s)
        .f64(step.received_at_s)
        .f64(step.pre_error_m)
        .f64(step.post_error_m)
        .f64(step.progress_m)
        .string(&step.command_policy_id)
        .string(&step.outcome_policy_id)
        .u64(step.command_policy_fingerprint)
        .u64(step.outcome_policy_fingerprint)
        .u64(step.authority_receipt_fingerprint)
        .u64(step.authority_scope_fingerprint)
        .string(&step.authority_scope_id)
        .u64(purpose_id(step.execution_purpose))
        .u64(basis_id(step.qualification_basis))
        .bool(step.step_accepted);
}

fn digest_operational_policy(policy: &HumanoidReachCryptographicOperationalPolicy) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.operational-policy.v1");
    h.u32(policy.schema_version)
        .string(&policy.policy_id)
        .digest(policy.subject_digest);
    for stage in [policy.simulation, policy.hil, policy.physical] {
        h.u64(purpose_id(stage.purpose))
            .digest(stage.step_protocol_digest)
            .digest(stage.episode_protocol_digest);
    }
    h.u64(policy.maximum_simulation_stage_age_millis)
        .u64(policy.maximum_hil_stage_age_millis)
        .u64(policy.maximum_physical_stage_age_millis)
        .u64(policy.maximum_stage_pair_skew_millis)
        .u64(policy.operational_artifact_validity_millis);
    h.finish()
}

fn digest_step_stage_artifact(stage: &HumanoidReachCryptographicStepStageArtifact) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.step-stage.v1");
    h.u64(purpose_id(stage.purpose))
        .digest(stage.subject_digest)
        .digest(stage.protocol_digest)
        .digest(stage.corpus_digest)
        .u64(stage.issued_unix_millis)
        .u64(stage.inner.artifact_fingerprint);
    h.finish()
}

fn digest_episode_stage_artifact(stage: &HumanoidReachCryptographicEpisodeStageArtifact) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.episode-stage.v1");
    h.u64(purpose_id(stage.purpose))
        .digest(stage.subject_digest)
        .digest(stage.protocol_digest)
        .digest(stage.corpus_digest)
        .u64(stage.issued_unix_millis)
        .u64(stage.inner.artifact_fingerprint);
    h.finish()
}

fn digest_operational_artifact(artifact: &HumanoidReachCryptographicOperationalArtifact) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.operational-artifact.v1");
    h.digest(artifact.subject_digest)
        .digest(artifact.policy_digest)
        .digest(artifact.simulation_step_digest)
        .digest(artifact.simulation_episode_digest)
        .digest(artifact.hil_step_digest)
        .digest(artifact.hil_episode_digest)
        .digest(artifact.physical_step_digest)
        .digest(artifact.physical_episode_digest)
        .u64(artifact.issued_unix_millis)
        .u64(artifact.valid_until_unix_millis)
        .u64(artifact.inner.artifact_fingerprint);
    h.finish()
}

fn digest_operator_approval(approval: &HumanoidReachCryptographicOperatorApproval) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.operator-approval.v1");
    h.string(&approval.approval_id)
        .digest(approval.subject_digest)
        .digest(approval.policy_digest)
        .string(&approval.operational_scope_id)
        .f64(approval.approved_at_s)
        .f64(approval.valid_until_s)
        .string(&approval.operator_source.evidence_id)
        .f32(approval.operator_source.scale)
        .f64(approval.operator_source.evaluated_at_s)
        .f64(approval.operator_source.valid_until_s);
    h.finish()
}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
}

fn morphology_id(value: HumanoidMorphology) -> u64 {
    match value {
        HumanoidMorphology::Dmc21 => 1,
        HumanoidMorphology::WithNeckWrist => 2,
        HumanoidMorphology::Dexterous53 => 3,
        HumanoidMorphology::FullSpine => 4,
    }
}

fn task_id(value: HumanoidTask) -> u64 {
    match value {
        HumanoidTask::Stand => 1,
        HumanoidTask::Walk => 2,
        HumanoidTask::Run => 3,
        HumanoidTask::Reach => 4,
        HumanoidTask::Grasp => 5,
    }
}

fn actuation_mode_id(value: ActuationMode) -> u64 {
    match value {
        ActuationMode::NormalizedTorque => 1,
        ActuationMode::TorqueNewtonMetres => 2,
        ActuationMode::NormalizedPosition => 3,
        ActuationMode::PositionTargetRadians => 4,
    }
}

fn hand_id(value: HandSide) -> u64 {
    match value {
        HandSide::Right => 1,
        HandSide::Left => 2,
    }
}

fn purpose_id(value: HumanoidExecutionPurpose) -> u64 {
    match value {
        HumanoidExecutionPurpose::SimulationQualification => 1,
        HumanoidExecutionPurpose::HilQualification => 2,
        HumanoidExecutionPurpose::PhysicalQualification => 3,
        HumanoidExecutionPurpose::Operational => 4,
    }
}

fn basis_id(value: HumanoidQualificationAuthorityBasis) -> u64 {
    match value {
        HumanoidQualificationAuthorityBasis::TrialProtocol => 1,
        HumanoidQualificationAuthorityBasis::QualifiedCapability => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Reach,
            ActuationMode::NormalizedTorque,
            "crypto-reach-test-v1",
        )
    }

    #[test]
    fn subject_digest_is_not_the_legacy_u64_identity() {
        let digest = digest_subject(&subject()).unwrap();
        assert!(!digest.is_zero());
        assert_eq!(digest.to_hex().len(), 64);
    }

    #[test]
    fn operational_policy_digest_changes_when_stage_requirement_changes() {
        let a = HumanoidReachCryptographicOperationalPolicy::from_exact_protocols(
            &subject(),
            "crypto-policy-v1",
            HumanoidReachCryptographicStageRequirement {
                purpose: HumanoidExecutionPurpose::SimulationQualification,
                step_protocol_digest: HumanoidEvidenceDigest::from_bytes([1; 32]),
                episode_protocol_digest: HumanoidEvidenceDigest::from_bytes([2; 32]),
            },
            HumanoidReachCryptographicStageRequirement {
                purpose: HumanoidExecutionPurpose::HilQualification,
                step_protocol_digest: HumanoidEvidenceDigest::from_bytes([3; 32]),
                episode_protocol_digest: HumanoidEvidenceDigest::from_bytes([4; 32]),
            },
            HumanoidReachCryptographicStageRequirement {
                purpose: HumanoidExecutionPurpose::PhysicalQualification,
                step_protocol_digest: HumanoidEvidenceDigest::from_bytes([5; 32]),
                episode_protocol_digest: HumanoidEvidenceDigest::from_bytes([6; 32]),
            },
            10_000,
            10_000,
            10_000,
            1_000,
            5_000,
        )
        .unwrap();
        let mut physical = a.physical();
        physical.episode_protocol_digest = HumanoidEvidenceDigest::from_bytes([7; 32]);
        let b = HumanoidReachCryptographicOperationalPolicy::from_exact_protocols(
            &subject(),
            "crypto-policy-v1",
            a.simulation(),
            a.hil(),
            physical,
            10_000,
            10_000,
            10_000,
            1_000,
            5_000,
        )
        .unwrap();
        assert_ne!(a.policy_digest(), b.policy_digest());
    }
}
