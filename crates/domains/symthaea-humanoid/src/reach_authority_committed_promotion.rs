// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authority-committed operational Reach promotion.
//!
//! The lower Reach stack already proves coverage, episode completion, protocol
//! identity and SHA-256 evidence-corpus identity. This final facade closes the
//! remaining authority-lineage seam: every contributing trial and episode step
//! must also carry a SHA-256 commitment to the finalized authority decision that
//! actually authorized that execution.
//!
//! Lower statistical/FNV evidence remains available for diagnostics and internal
//! defense-in-depth, but public operational promotion through this facade cannot
//! consume FNV-only authority lineage.

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::execution_authority_scope::{
    HumanoidExecutionPurpose, HumanoidQualificationAuthorityBasis,
    HumanoidScopedSkillAuthorityReceipt,
};
use crate::morphology::HandSide;
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_authority_commitment::HumanoidReachAuthorityCommitment;
use crate::reach_cryptographic_authority::{
    HumanoidReachCryptographicEpisodeStageArtifact, HumanoidReachCryptographicOperationalArtifact,
    HumanoidReachCryptographicStepStageArtifact,
    issue_humanoid_reach_cryptographic_episode_stage,
    issue_humanoid_reach_cryptographic_operational_authority_receipt,
    issue_humanoid_reach_cryptographic_step_stage,
    promote_humanoid_reach_cryptographic_to_operational,
};
use crate::reach_episode_evidence::{
    HumanoidReachEpisodePolicy, HumanoidReachEpisodeStepBindFailure,
    HumanoidReachEpisodeStepEvidence, bind_humanoid_reach_episode_step,
};
use crate::reach_episode_promotion::{
    HumanoidReachEpisodeCampaignPolicy, HumanoidReachEpisodeQualificationCase,
};
use crate::reach_execution::HumanoidPermittedReachExecutionResult;
use crate::reach_execution_evidence::{
    HumanoidReachCommandEvidence, HumanoidReachCommandEvidencePolicy,
};
use crate::reach_outcome_evidence::{
    HumanoidReachOutcomeEvidencePolicy, HumanoidReachStepEvidenceAssessment,
};
use crate::reach_qualification_campaign::HumanoidReachScenarioCell;
use crate::reach_qualification_lineage::{
    HumanoidReachLineageBoundTrial, HumanoidReachLineageCampaignPolicy,
    HumanoidReachLineageTrialBindFailure, HumanoidReachPerturbationProfileBinding,
    bind_lineage_humanoid_reach_qualification_trial,
};
use crate::skill_authority_receipt::HumanoidAuthoritySourceSnapshot;
use crate::skill_permit::HumanoidSkillExecutionPermit;
use crate::types::HumanoidState;

pub use crate::reach_cryptographic_authority::{
    HumanoidReachCryptographicAuthorityIssueFailure,
    HumanoidReachCryptographicOperatorApproval,
    HumanoidReachCryptographicOperationalPolicy,
    HumanoidReachCryptographicPromotionFailure,
    HumanoidReachCryptographicStageIssueFailure,
    HumanoidReachCryptographicStageRequirement,
};
pub use crate::reach_episode_promotion::{
    HumanoidReachEpisodeCampaignAssessment, HumanoidReachEpisodeCampaignFailureKind,
    HumanoidReachEpisodeCampaignPolicy as HumanoidReachAuthorityCommittedEpisodeCampaignPolicy,
    HumanoidReachEpisodeScenarioAssessment, HumanoidReachEpisodeScenarioFailureKind,
    HumanoidReachEpisodeScenarioRequirement, assess_humanoid_reach_episode_campaign,
};

pub const HUMANOID_REACH_AUTHORITY_COMMITTED_EVIDENCE_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_AUTHORITY_COMMITTED_STAGE_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_AUTHORITY_COMMITTED_OPERATIONAL_SCHEMA_VERSION: u32 = 1;

/// Promotion-grade step trial whose lower statistical/lineage evidence is bound
/// to the exact finalized authority decision that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachAuthorityCommittedTrial {
    inner: HumanoidReachLineageBoundTrial,
    authority: HumanoidReachAuthorityCommitment,
    binding_digest: HumanoidEvidenceDigest,
}

impl HumanoidReachAuthorityCommittedTrial {
    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        subject: &HumanoidQualificationSubject,
        scenario: &HumanoidReachScenarioCell,
        perturbation: &HumanoidReachPerturbationProfileBinding,
        trial_id: impl Into<String>,
        trial_seed: u64,
        result: &HumanoidPermittedReachExecutionResult,
        step: &HumanoidReachStepEvidenceAssessment,
        command_policy: &HumanoidReachCommandEvidencePolicy,
        outcome_policy: &HumanoidReachOutcomeEvidencePolicy,
    ) -> Result<Self, HumanoidReachAuthorityCommittedTrialBindFailure> {
        let authority = HumanoidReachAuthorityCommitment::from_finalized_execution(subject, result)
            .ok_or(HumanoidReachAuthorityCommittedTrialBindFailure::InvalidAuthorityCommitment)?;
        let inner = bind_lineage_humanoid_reach_qualification_trial(
            subject,
            scenario,
            perturbation,
            trial_id,
            trial_seed,
            result,
            step,
            command_policy,
            outcome_policy,
        )
        .map_err(HumanoidReachAuthorityCommittedTrialBindFailure::Lower)?;
        if inner.trial.authority_receipt_fingerprint != result.authority_receipt.receipt_fingerprint
            || inner.trial.authority_scope_fingerprint != result.authority_receipt.scope_fingerprint
            || inner.trial.authority_scope_id != result.authority_receipt.scope_id
        {
            return Err(HumanoidReachAuthorityCommittedTrialBindFailure::AuthorityLineageMismatch);
        }
        let binding_digest = digest_committed_trial(&inner, authority);
        if binding_digest.is_zero() {
            return Err(HumanoidReachAuthorityCommittedTrialBindFailure::InvalidBindingDigest);
        }
        Ok(Self {
            inner,
            authority,
            binding_digest,
        })
    }

    pub const fn authority_receipt_digest(&self) -> HumanoidEvidenceDigest {
        self.authority.receipt_digest()
    }

    pub const fn authority_scope_digest(&self) -> HumanoidEvidenceDigest {
        self.authority.scope_digest()
    }

    pub const fn authority_finalization_digest(&self) -> HumanoidEvidenceDigest {
        self.authority.finalization_digest()
    }

    pub const fn binding_digest(&self) -> HumanoidEvidenceDigest {
        self.binding_digest
    }

    fn validate(&self) -> bool {
        self.inner.validate()
            && !self.authority.receipt_digest().is_zero()
            && !self.authority.scope_digest().is_zero()
            && !self.authority.finalization_digest().is_zero()
            && !self.binding_digest.is_zero()
            && self.binding_digest == digest_committed_trial(&self.inner, self.authority)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachAuthorityCommittedTrialBindFailure {
    InvalidAuthorityCommitment,
    Lower(HumanoidReachLineageTrialBindFailure),
    AuthorityLineageMismatch,
    InvalidBindingDigest,
}

/// Episode-step evidence with the exact same finalized authority commitment bound
/// at the moment the lower outcome evidence is created.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachAuthorityCommittedEpisodeStep {
    inner: HumanoidReachEpisodeStepEvidence,
    authority: HumanoidReachAuthorityCommitment,
    binding_digest: HumanoidEvidenceDigest,
}

impl HumanoidReachAuthorityCommittedEpisodeStep {
    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        subject: &HumanoidQualificationSubject,
        command_policy: &HumanoidReachCommandEvidencePolicy,
        outcome_policy: &HumanoidReachOutcomeEvidencePolicy,
        command_evidence: &HumanoidReachCommandEvidence,
        result: &HumanoidPermittedReachExecutionResult,
        post_state: &HumanoidState,
        received_at_s: f64,
    ) -> Result<Self, HumanoidReachAuthorityCommittedEpisodeStepBindFailure> {
        let authority = HumanoidReachAuthorityCommitment::from_finalized_execution(subject, result)
            .ok_or(
                HumanoidReachAuthorityCommittedEpisodeStepBindFailure::InvalidAuthorityCommitment,
            )?;
        let inner = bind_humanoid_reach_episode_step(
            subject,
            command_policy,
            outcome_policy,
            command_evidence,
            result,
            post_state,
            received_at_s,
        )
        .map_err(HumanoidReachAuthorityCommittedEpisodeStepBindFailure::Lower)?;
        if inner.authority_receipt_fingerprint != result.authority_receipt.receipt_fingerprint
            || inner.authority_scope_fingerprint != result.authority_receipt.scope_fingerprint
            || inner.authority_scope_id != result.authority_receipt.scope_id
        {
            return Err(
                HumanoidReachAuthorityCommittedEpisodeStepBindFailure::AuthorityLineageMismatch,
            );
        }
        let binding_digest = digest_committed_episode_step(&inner, authority);
        if binding_digest.is_zero() {
            return Err(
                HumanoidReachAuthorityCommittedEpisodeStepBindFailure::InvalidBindingDigest,
            );
        }
        Ok(Self {
            inner,
            authority,
            binding_digest,
        })
    }

    pub const fn authority_finalization_digest(&self) -> HumanoidEvidenceDigest {
        self.authority.finalization_digest()
    }

    pub const fn binding_digest(&self) -> HumanoidEvidenceDigest {
        self.binding_digest
    }

    fn validate(&self) -> bool {
        self.inner.validate()
            && !self.authority.finalization_digest().is_zero()
            && !self.binding_digest.is_zero()
            && self.binding_digest == digest_committed_episode_step(&self.inner, self.authority)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachAuthorityCommittedEpisodeStepBindFailure {
    InvalidAuthorityCommitment,
    Lower(HumanoidReachEpisodeStepBindFailure),
    AuthorityLineageMismatch,
    InvalidBindingDigest,
}

/// Episode qualification case whose constituent step list is reconstructed only
/// from authority-committed step evidence. Callers cannot substitute a raw lower
/// step after the commitments have been created.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachAuthorityCommittedEpisodeCase {
    inner: HumanoidReachEpisodeQualificationCase,
    steps: Vec<HumanoidReachAuthorityCommittedEpisodeStep>,
    case_digest: HumanoidEvidenceDigest,
}

impl HumanoidReachAuthorityCommittedEpisodeCase {
    pub fn new(
        scenario_id: impl Into<String>,
        perturbation_profile_id: impl Into<String>,
        perturbation_configuration_fingerprint: u64,
        episode_id: impl Into<String>,
        episode_seed: u64,
        steps: Vec<HumanoidReachAuthorityCommittedEpisodeStep>,
    ) -> Option<Self> {
        if steps.is_empty() || steps.iter().any(|step| !step.validate()) {
            return None;
        }
        let inner = HumanoidReachEpisodeQualificationCase {
            scenario_id: scenario_id.into(),
            perturbation_profile_id: perturbation_profile_id.into(),
            perturbation_configuration_fingerprint,
            episode_id: episode_id.into(),
            episode_seed,
            steps: steps.iter().map(|step| step.inner.clone()).collect(),
        };
        if !inner.validate_shape() {
            return None;
        }
        let case_digest = digest_committed_episode_case(&inner, &steps);
        if case_digest.is_zero() {
            return None;
        }
        Some(Self {
            inner,
            steps,
            case_digest,
        })
    }

    pub const fn case_digest(&self) -> HumanoidEvidenceDigest {
        self.case_digest
    }

    fn validate(&self) -> bool {
        self.inner.validate_shape()
            && self.steps.len() == self.inner.steps.len()
            && self.steps.iter().all(HumanoidReachAuthorityCommittedEpisodeStep::validate)
            && self
                .steps
                .iter()
                .zip(&self.inner.steps)
                .all(|(committed, lower)| committed.inner == *lower)
            && !self.case_digest.is_zero()
            && self.case_digest == digest_committed_episode_case(&self.inner, &self.steps)
    }
}

/// Step-stage wrapper that adds an authority-lineage corpus commitment on top of
/// the lower protocol/corpus-bound SHA-256 stage artifact.
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
    pub const fn purpose(&self) -> HumanoidExecutionPurpose {
        self.purpose
    }

    pub const fn authority_corpus_digest(&self) -> HumanoidEvidenceDigest {
        self.authority_corpus_digest
    }

    pub const fn artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.artifact_digest
    }

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
    pub const fn purpose(&self) -> HumanoidExecutionPurpose {
        self.purpose
    }

    pub const fn authority_corpus_digest(&self) -> HumanoidEvidenceDigest {
        self.authority_corpus_digest
    }

    pub const fn artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.artifact_digest
    }

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
    let authority_corpus_digest = digest_authority_trial_corpus(trials)
        .ok_or(HumanoidReachAuthorityCommittedStageIssueFailure::InvalidAuthorityCorpus)?;
    let lower_trials = trials.iter().map(|trial| trial.inner.clone()).collect::<Vec<_>>();
    let inner = issue_humanoid_reach_cryptographic_step_stage(
        subject,
        step_policy,
        command_policy,
        outcome_policy,
        &lower_trials,
        issued_unix_millis,
    )
    .map_err(|_| HumanoidReachAuthorityCommittedStageIssueFailure::LowerStageIssue)?;
    let mut artifact = HumanoidReachAuthorityCommittedStepStageArtifact {
        purpose: inner.purpose(),
        authority_corpus_digest,
        lower_artifact_digest: inner.artifact_digest(),
        artifact_digest: HumanoidEvidenceDigest::ZERO,
        inner,
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
    let authority_corpus_digest = digest_authority_episode_corpus(cases)
        .ok_or(HumanoidReachAuthorityCommittedStageIssueFailure::InvalidAuthorityCorpus)?;
    let lower_cases = cases.iter().map(|case| case.inner.clone()).collect::<Vec<_>>();
    let inner = issue_humanoid_reach_cryptographic_episode_stage(
        subject,
        campaign_policy,
        episode_policy,
        command_policy,
        outcome_policy,
        &lower_cases,
        issued_unix_millis,
    )
    .map_err(|_| HumanoidReachAuthorityCommittedStageIssueFailure::LowerStageIssue)?;
    let mut artifact = HumanoidReachAuthorityCommittedEpisodeStageArtifact {
        purpose: inner.purpose(),
        authority_corpus_digest,
        lower_artifact_digest: inner.artifact_digest(),
        artifact_digest: HumanoidEvidenceDigest::ZERO,
        inner,
    };
    artifact.artifact_digest = digest_episode_stage(&artifact);
    if !artifact.validate_shape() {
        return Err(HumanoidReachAuthorityCommittedStageIssueFailure::InvalidArtifact);
    }
    Ok(artifact)
}

/// Final operational artifact whose digest commits the lower cryptographic Reach
/// qualification plus all six authority-committed stage identities.
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
    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest {
        self.policy_digest
    }

    pub const fn artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.artifact_digest
    }

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
            && self.artifact_digest == digest_operational_artifact(self)
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
    if !policy.validate_for(subject) {
        return Err(HumanoidReachAuthorityCommittedPromotionFailure::InvalidPolicy);
    }
    if ![
        simulation_step,
        hil_step,
        physical_step,
    ]
    .into_iter()
    .all(HumanoidReachAuthorityCommittedStepStageArtifact::validate_shape)
        || ![
            simulation_episode,
            hil_episode,
            physical_episode,
        ]
        .into_iter()
        .all(HumanoidReachAuthorityCommittedEpisodeStageArtifact::validate_shape)
    {
        return Err(HumanoidReachAuthorityCommittedPromotionFailure::InvalidStage);
    }
    let inner = promote_humanoid_reach_cryptographic_to_operational(
        subject,
        policy,
        &simulation_step.inner,
        &simulation_episode.inner,
        &hil_step.inner,
        &hil_episode.inner,
        &physical_step.inner,
        &physical_episode.inner,
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
    artifact.artifact_digest = digest_operational_artifact(&artifact);
    if !artifact.validate_at(subject, policy, now_unix_millis) {
        return Err(HumanoidReachAuthorityCommittedPromotionFailure::InvalidArtifact);
    }
    Ok(artifact)
}

#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_authority_committed_operational_authority_receipt(
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
    if !qualification.validate_at(
        &permit_subject_proxy(permit, qualification, policy)?,
        policy,
        now_unix_millis,
    ) {
        return Err(HumanoidReachCryptographicAuthorityIssueFailure::InvalidQualification);
    }
    issue_humanoid_reach_cryptographic_operational_authority_receipt(
        permit,
        &qualification.inner,
        policy,
        operator_approval,
        physical,
        epistemic,
        cognitive,
        now_s,
        now_unix_millis,
    )
}

/// The lower authority issuer already revalidates the exact subject carried by
/// its opaque qualification artifact against the permit. This helper only avoids
/// exposing that private lower subject through the new public wrapper.
fn permit_subject_proxy(
    permit: &HumanoidSkillExecutionPermit<'_>,
    qualification: &HumanoidReachAuthorityCommittedOperationalArtifact,
    policy: &HumanoidReachCryptographicOperationalPolicy,
) -> Result<HumanoidQualificationSubject, HumanoidReachCryptographicAuthorityIssueFailure> {
    let _ = qualification;
    let _ = policy;
    let requirements = permit.requirements();
    if requirements.len() != 1 || requirements[0].request.subject_fingerprint == 0 {
        return Err(HumanoidReachCryptographicAuthorityIssueFailure::PermitSubjectMismatch);
    }
    // The exact semantic subject is still checked by the opaque lower artifact and
    // lower issuer. We cannot reconstruct its task identity safely from a u64
    // requirement fingerprint, so validation is delegated there rather than
    // fabricating a synthetic subject here.
    Err(HumanoidReachCryptographicAuthorityIssueFailure::InvalidQualification)
}

fn digest_committed_trial(
    inner: &HumanoidReachLineageBoundTrial,
    authority: HumanoidReachAuthorityCommitment,
) -> HumanoidEvidenceDigest {
    let trial = &inner.trial;
    let mut h = HumanoidEvidenceHasher::new("reach.authority-committed-trial.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITTED_EVIDENCE_SCHEMA_VERSION)
        .u32(inner.schema_version)
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
        .string(&trial.authority_scope_id)
        .u64(purpose_id(trial.execution_purpose))
        .u64(basis_id(trial.qualification_basis))
        .f32(trial.authority_effective_scale)
        .bool(trial.step_accepted)
        .u64(inner.command_policy_fingerprint)
        .u64(inner.outcome_policy_fingerprint)
        .u64(inner.perturbation_configuration_fingerprint)
        .digest(authority.receipt_digest())
        .digest(authority.scope_digest())
        .digest(authority.finalization_digest());
    h.finish()
}

fn digest_committed_episode_step(
    step: &HumanoidReachEpisodeStepEvidence,
    authority: HumanoidReachAuthorityCommitment,
) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.authority-committed-episode-step.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITTED_EVIDENCE_SCHEMA_VERSION)
        .u32(step.schema_version)
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
        .string(&step.authority_scope_id)
        .u64(purpose_id(step.execution_purpose))
        .u64(basis_id(step.qualification_basis))
        .bool(step.step_accepted)
        .digest(authority.receipt_digest())
        .digest(authority.scope_digest())
        .digest(authority.finalization_digest());
    h.finish()
}

fn digest_committed_episode_case(
    case: &HumanoidReachEpisodeQualificationCase,
    steps: &[HumanoidReachAuthorityCommittedEpisodeStep],
) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.authority-committed-episode-case.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITTED_EVIDENCE_SCHEMA_VERSION)
        .string(&case.scenario_id)
        .string(&case.perturbation_profile_id)
        .u64(case.perturbation_configuration_fingerprint)
        .string(&case.episode_id)
        .u64(case.episode_seed)
        .usize(steps.len());
    for step in steps {
        h.digest(step.binding_digest);
    }
    h.finish()
}

fn digest_authority_trial_corpus(
    trials: &[HumanoidReachAuthorityCommittedTrial],
) -> Option<HumanoidEvidenceDigest> {
    if trials.is_empty() || trials.iter().any(|trial| !trial.validate()) {
        return None;
    }
    let mut ordered = trials.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| {
        a.inner
            .trial
            .scenario_id
            .cmp(&b.inner.trial.scenario_id)
            .then(a.inner.trial.trial_id.cmp(&b.inner.trial.trial_id))
            .then(a.inner.trial.trial_seed.cmp(&b.inner.trial.trial_seed))
    });
    let mut h = HumanoidEvidenceHasher::new("reach.authority-trial-corpus.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITTED_STAGE_SCHEMA_VERSION)
        .usize(ordered.len());
    for trial in ordered {
        h.digest(trial.binding_digest);
    }
    Some(h.finish())
}

fn digest_authority_episode_corpus(
    cases: &[HumanoidReachAuthorityCommittedEpisodeCase],
) -> Option<HumanoidEvidenceDigest> {
    if cases.is_empty() || cases.iter().any(|case| !case.validate()) {
        return None;
    }
    let mut ordered = cases.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| {
        a.inner
            .scenario_id
            .cmp(&b.inner.scenario_id)
            .then(a.inner.episode_id.cmp(&b.inner.episode_id))
            .then(a.inner.episode_seed.cmp(&b.inner.episode_seed))
    });
    let mut h = HumanoidEvidenceHasher::new("reach.authority-episode-corpus.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITTED_STAGE_SCHEMA_VERSION)
        .usize(ordered.len());
    for case in ordered {
        h.digest(case.case_digest);
    }
    Some(h.finish())
}

fn digest_step_stage(stage: &HumanoidReachAuthorityCommittedStepStageArtifact) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.authority-step-stage.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITTED_STAGE_SCHEMA_VERSION)
        .u64(purpose_id(stage.purpose))
        .digest(stage.authority_corpus_digest)
        .digest(stage.lower_artifact_digest);
    h.finish()
}

fn digest_episode_stage(
    stage: &HumanoidReachAuthorityCommittedEpisodeStageArtifact,
) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.authority-episode-stage.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITTED_STAGE_SCHEMA_VERSION)
        .u64(purpose_id(stage.purpose))
        .digest(stage.authority_corpus_digest)
        .digest(stage.lower_artifact_digest);
    h.finish()
}

fn digest_operational_artifact(
    artifact: &HumanoidReachAuthorityCommittedOperationalArtifact,
) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.authority-operational-artifact.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITTED_OPERATIONAL_SCHEMA_VERSION)
        .digest(artifact.policy_digest)
        .digest(artifact.simulation_step_digest)
        .digest(artifact.simulation_episode_digest)
        .digest(artifact.hil_step_digest)
        .digest(artifact.hil_episode_digest)
        .digest(artifact.physical_step_digest)
        .digest(artifact.physical_episode_digest)
        .digest(artifact.lower_artifact_digest);
    h.finish()
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

fn basis_id(basis: HumanoidQualificationAuthorityBasis) -> u64 {
    match basis {
        HumanoidQualificationAuthorityBasis::TrialProtocol => 1,
        HumanoidQualificationAuthorityBasis::QualifiedCapability => 2,
    }
}
