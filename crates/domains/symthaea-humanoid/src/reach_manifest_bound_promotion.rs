// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Preferred operational Reach promotion with exact perturbation-manifest binding.
//!
//! This facade sits above SHA-256 protocol/corpus, finalized-authority, spatial,
//! and diversity gates. It closes the remaining perturbation substitution seam by
//! requiring each Simulation/HIL/Physical scenario to precommit to the exact
//! perturbation manifest that generated its evidence.

use std::collections::{BTreeMap, BTreeSet};

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::execution_authority_scope::{HumanoidExecutionPurpose, HumanoidScopedSkillAuthorityReceipt};
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_authority_committed_evidence::{
    HumanoidReachAuthorityCommittedEpisodeStep, HumanoidReachAuthorityCommittedTrial,
    HumanoidReachAuthorityCommittedTrialBindFailure,
};
use crate::reach_execution::HumanoidPermittedReachExecutionResult;
use crate::reach_execution_evidence::{HumanoidReachCommandEvidence, HumanoidReachCommandEvidencePolicy};
use crate::reach_outcome_evidence::{HumanoidReachOutcomeEvidencePolicy, HumanoidReachStepEvidenceAssessment};
use crate::reach_perturbation_manifest::HumanoidReachPerturbationManifest;
use crate::reach_qualification_campaign::HumanoidReachScenarioCell;
use crate::reach_qualification_lineage::HumanoidReachLineageCampaignPolicy;
use crate::reach_strong_diversity_promotion::{
    HumanoidReachStrongEpisodeStageArtifact, HumanoidReachStrongOperationalArtifact,
    HumanoidReachStrongPromotionFailure, HumanoidReachStrongStageIssueFailure,
    HumanoidReachStrongStepStageArtifact, issue_humanoid_reach_strong_episode_stage,
    issue_humanoid_reach_strong_operational_authority_receipt,
    issue_humanoid_reach_strong_step_stage, promote_humanoid_reach_strong_to_operational,
};
use crate::skill_authority_receipt::HumanoidAuthoritySourceSnapshot;
use crate::skill_permit::HumanoidSkillExecutionPermit;
use crate::types::HumanoidState;

pub use crate::reach_cryptographic_authority::{
    HumanoidReachCryptographicAuthorityIssueFailure,
    HumanoidReachCryptographicOperationalPolicy as HumanoidReachBaseOperationalPolicy,
    HumanoidReachCryptographicStageRequirement as HumanoidReachBaseStageRequirement,
};
pub use crate::reach_episode_evidence::HumanoidReachEpisodePolicy;
pub use crate::reach_episode_promotion::{
    HumanoidReachEpisodeCampaignAssessment, HumanoidReachEpisodeCampaignFailureKind,
    HumanoidReachEpisodeCampaignPolicy, HumanoidReachEpisodeScenarioAssessment,
    HumanoidReachEpisodeScenarioFailureKind, HumanoidReachEpisodeScenarioRequirement,
    assess_humanoid_reach_episode_campaign,
};

pub const HUMANOID_REACH_MANIFEST_STAGE_REQUIREMENT_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_MANIFEST_OPERATIONAL_POLICY_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_MANIFEST_BOUND_EVIDENCE_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_MANIFEST_STAGE_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_MANIFEST_OPERATIONAL_ARTIFACT_SCHEMA_VERSION: u32 = 1;
pub const HUMANOID_REACH_MANIFEST_OPERATOR_APPROVAL_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HumanoidReachScenarioManifestRequirement {
    pub scenario_id: String,
    pub manifest_digest: HumanoidEvidenceDigest,
}

impl HumanoidReachScenarioManifestRequirement {
    pub fn validate(&self) -> bool {
        valid_id(&self.scenario_id) && !self.manifest_digest.is_zero()
    }
}

/// Exact manifest requirements for one qualification stage. Step and episode
/// scenario sets are separate because their campaign matrices need not be equal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HumanoidReachManifestStageRequirement {
    schema_version: u32,
    purpose: HumanoidExecutionPurpose,
    step_scenarios: Vec<HumanoidReachScenarioManifestRequirement>,
    episode_scenarios: Vec<HumanoidReachScenarioManifestRequirement>,
    requirement_digest: HumanoidEvidenceDigest,
}

impl HumanoidReachManifestStageRequirement {
    pub fn new(
        purpose: HumanoidExecutionPurpose,
        step_scenarios: Vec<HumanoidReachScenarioManifestRequirement>,
        episode_scenarios: Vec<HumanoidReachScenarioManifestRequirement>,
    ) -> Option<Self> {
        let mut value = Self {
            schema_version: HUMANOID_REACH_MANIFEST_STAGE_REQUIREMENT_SCHEMA_VERSION,
            purpose,
            step_scenarios,
            episode_scenarios,
            requirement_digest: HumanoidEvidenceDigest::ZERO,
        };
        if !value.validate_shape() {
            return None;
        }
        value.requirement_digest = digest_stage_requirement(&value);
        value.validate_shape().then_some(value)
    }

    pub const fn purpose(&self) -> HumanoidExecutionPurpose {
        self.purpose
    }

    pub const fn requirement_digest(&self) -> HumanoidEvidenceDigest {
        self.requirement_digest
    }

    pub fn step_scenarios(&self) -> &[HumanoidReachScenarioManifestRequirement] {
        &self.step_scenarios
    }

    pub fn episode_scenarios(&self) -> &[HumanoidReachScenarioManifestRequirement] {
        &self.episode_scenarios
    }

    fn validate_shape(&self) -> bool {
        self.schema_version == HUMANOID_REACH_MANIFEST_STAGE_REQUIREMENT_SCHEMA_VERSION
            && self.purpose.is_qualification()
            && valid_manifest_requirements(&self.step_scenarios)
            && valid_manifest_requirements(&self.episode_scenarios)
            && (self.requirement_digest.is_zero()
                || self.requirement_digest == digest_stage_requirement(self))
    }
}

/// Operational policy that precommits the lower exact Reach protocol plus exact
/// perturbation manifests for Simulation, HIL and Physical evidence.
pub struct HumanoidReachManifestOperationalPolicy {
    schema_version: u32,
    policy_id: String,
    base: HumanoidReachBaseOperationalPolicy,
    simulation: HumanoidReachManifestStageRequirement,
    hil: HumanoidReachManifestStageRequirement,
    physical: HumanoidReachManifestStageRequirement,
    policy_digest: HumanoidEvidenceDigest,
}

impl std::fmt::Debug for HumanoidReachManifestOperationalPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidReachManifestOperationalPolicy")
            .field("policy_id", &self.policy_id)
            .field("base_policy_digest", &self.base.policy_digest())
            .field("policy_digest", &self.policy_digest)
            .finish()
    }
}

impl HumanoidReachManifestOperationalPolicy {
    pub fn from_base(
        subject: &HumanoidQualificationSubject,
        policy_id: impl Into<String>,
        base: HumanoidReachBaseOperationalPolicy,
        simulation: HumanoidReachManifestStageRequirement,
        hil: HumanoidReachManifestStageRequirement,
        physical: HumanoidReachManifestStageRequirement,
    ) -> Option<Self> {
        let mut policy = Self {
            schema_version: HUMANOID_REACH_MANIFEST_OPERATIONAL_POLICY_SCHEMA_VERSION,
            policy_id: policy_id.into(),
            base,
            simulation,
            hil,
            physical,
            policy_digest: HumanoidEvidenceDigest::ZERO,
        };
        if !policy.validate_shape(subject) {
            return None;
        }
        policy.policy_digest = digest_operational_policy(&policy);
        policy.validate_for(subject).then_some(policy)
    }

    pub fn policy_id(&self) -> &str {
        &self.policy_id
    }

    pub const fn policy_digest(&self) -> HumanoidEvidenceDigest {
        self.policy_digest
    }

    pub fn base_policy(&self) -> &HumanoidReachBaseOperationalPolicy {
        &self.base
    }

    pub fn simulation(&self) -> &HumanoidReachManifestStageRequirement {
        &self.simulation
    }

    pub fn hil(&self) -> &HumanoidReachManifestStageRequirement {
        &self.hil
    }

    pub fn physical(&self) -> &HumanoidReachManifestStageRequirement {
        &self.physical
    }

    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.validate_shape(subject)
            && !self.policy_digest.is_zero()
            && self.policy_digest == digest_operational_policy(self)
    }

    fn validate_shape(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.schema_version == HUMANOID_REACH_MANIFEST_OPERATIONAL_POLICY_SCHEMA_VERSION
            && valid_id(&self.policy_id)
            && self.base.validate_for(subject)
            && self.simulation.validate_shape()
            && self.hil.validate_shape()
            && self.physical.validate_shape()
            && self.simulation.purpose == HumanoidExecutionPurpose::SimulationQualification
            && self.hil.purpose == HumanoidExecutionPurpose::HilQualification
            && self.physical.purpose == HumanoidExecutionPurpose::PhysicalQualification
            && self.base.simulation().purpose == self.simulation.purpose
            && self.base.hil().purpose == self.hil.purpose
            && self.base.physical().purpose == self.physical.purpose
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachManifestBoundTrial {
    pub(crate) inner: HumanoidReachAuthorityCommittedTrial,
    scenario_id: String,
    manifest_digest: HumanoidEvidenceDigest,
    binding_digest: HumanoidEvidenceDigest,
}

impl HumanoidReachManifestBoundTrial {
    #[allow(clippy::too_many_arguments)]
    pub fn bind(
        subject: &HumanoidQualificationSubject,
        scenario: &HumanoidReachScenarioCell,
        manifest: &HumanoidReachPerturbationManifest,
        trial_id: impl Into<String>,
        trial_seed: u64,
        result: &HumanoidPermittedReachExecutionResult,
        step: &HumanoidReachStepEvidenceAssessment,
        command_policy: &HumanoidReachCommandEvidencePolicy,
        outcome_policy: &HumanoidReachOutcomeEvidencePolicy,
    ) -> Result<Self, HumanoidReachManifestTrialBindFailure> {
        if !manifest.validate_for(subject)
            || !manifest.seed_scheme().supports_trial_evidence()
            || manifest.profile_id() != scenario.perturbation_profile_id
        {
            return Err(HumanoidReachManifestTrialBindFailure::InvalidManifest);
        }
        let compatibility = manifest.legacy_binding();
        let inner = HumanoidReachAuthorityCommittedTrial::bind(
            subject,
            scenario,
            &compatibility,
            trial_id,
            trial_seed,
            result,
            step,
            command_policy,
            outcome_policy,
        )
        .map_err(HumanoidReachManifestTrialBindFailure::Lower)?;
        let scenario_id = scenario.scenario_id.clone();
        let manifest_digest = manifest.manifest_digest();
        let binding_digest = digest_manifest_trial(&scenario_id, manifest_digest, inner.binding_digest());
        if binding_digest.is_zero() {
            return Err(HumanoidReachManifestTrialBindFailure::InvalidBindingDigest);
        }
        Ok(Self {
            inner,
            scenario_id,
            manifest_digest,
            binding_digest,
        })
    }

    pub fn scenario_id(&self) -> &str {
        &self.scenario_id
    }

    pub const fn manifest_digest(&self) -> HumanoidEvidenceDigest {
        self.manifest_digest
    }

    pub const fn binding_digest(&self) -> HumanoidEvidenceDigest {
        self.binding_digest
    }

    fn validate(&self) -> bool {
        self.inner.validate()
            && valid_id(&self.scenario_id)
            && !self.manifest_digest.is_zero()
            && !self.binding_digest.is_zero()
            && self.binding_digest
                == digest_manifest_trial(
                    &self.scenario_id,
                    self.manifest_digest,
                    self.inner.binding_digest(),
                )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachManifestTrialBindFailure {
    InvalidManifest,
    Lower(HumanoidReachAuthorityCommittedTrialBindFailure),
    InvalidBindingDigest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachManifestBoundEpisodeCase {
    pub(crate) inner: crate::reach_authority_committed_evidence::HumanoidReachAuthorityCommittedEpisodeCase,
    scenario_id: String,
    manifest_digest: HumanoidEvidenceDigest,
    binding_digest: HumanoidEvidenceDigest,
}

impl HumanoidReachManifestBoundEpisodeCase {
    pub fn new(
        subject: &HumanoidQualificationSubject,
        manifest: &HumanoidReachPerturbationManifest,
        scenario_id: impl Into<String>,
        episode_id: impl Into<String>,
        episode_seed: u64,
        steps: Vec<HumanoidReachAuthorityCommittedEpisodeStep>,
    ) -> Option<Self> {
        if !manifest.validate_for(subject) || !manifest.seed_scheme().supports_episode_evidence() {
            return None;
        }
        let scenario_id = scenario_id.into();
        if !valid_id(&scenario_id) {
            return None;
        }
        let compatibility = manifest.legacy_binding();
        let inner = crate::reach_authority_committed_evidence::HumanoidReachAuthorityCommittedEpisodeCase::new(
            scenario_id.clone(),
            manifest.profile_id(),
            compatibility.configuration_fingerprint,
            episode_id,
            episode_seed,
            steps,
        )?;
        let manifest_digest = manifest.manifest_digest();
        let binding_digest = digest_manifest_episode_case(
            &scenario_id,
            manifest_digest,
            inner.case_digest(),
        );
        if binding_digest.is_zero() {
            return None;
        }
        Some(Self {
            inner,
            scenario_id,
            manifest_digest,
            binding_digest,
        })
    }

    pub fn scenario_id(&self) -> &str {
        &self.scenario_id
    }

    pub const fn manifest_digest(&self) -> HumanoidEvidenceDigest {
        self.manifest_digest
    }

    pub const fn binding_digest(&self) -> HumanoidEvidenceDigest {
        self.binding_digest
    }

    fn validate(&self) -> bool {
        self.inner.validate()
            && valid_id(&self.scenario_id)
            && !self.manifest_digest.is_zero()
            && !self.binding_digest.is_zero()
            && self.binding_digest
                == digest_manifest_episode_case(
                    &self.scenario_id,
                    self.manifest_digest,
                    self.inner.case_digest(),
                )
    }
}

pub struct HumanoidReachManifestStepStageArtifact {
    purpose: HumanoidExecutionPurpose,
    manifest_requirement_digest: HumanoidEvidenceDigest,
    manifest_corpus_digest: HumanoidEvidenceDigest,
    lower_artifact_digest: HumanoidEvidenceDigest,
    artifact_digest: HumanoidEvidenceDigest,
    inner: HumanoidReachStrongStepStageArtifact,
}

impl std::fmt::Debug for HumanoidReachManifestStepStageArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidReachManifestStepStageArtifact")
            .field("purpose", &self.purpose)
            .field("manifest_requirement_digest", &self.manifest_requirement_digest)
            .field("manifest_corpus_digest", &self.manifest_corpus_digest)
            .field("artifact_digest", &self.artifact_digest)
            .finish()
    }
}

impl HumanoidReachManifestStepStageArtifact {
    pub const fn artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.artifact_digest
    }

    fn validate_shape(&self) -> bool {
        self.purpose.is_qualification()
            && !self.manifest_requirement_digest.is_zero()
            && !self.manifest_corpus_digest.is_zero()
            && self.lower_artifact_digest == self.inner.artifact_digest()
            && !self.lower_artifact_digest.is_zero()
            && !self.artifact_digest.is_zero()
            && self.artifact_digest == digest_step_stage(self)
    }
}

pub struct HumanoidReachManifestEpisodeStageArtifact {
    purpose: HumanoidExecutionPurpose,
    manifest_requirement_digest: HumanoidEvidenceDigest,
    manifest_corpus_digest: HumanoidEvidenceDigest,
    lower_artifact_digest: HumanoidEvidenceDigest,
    artifact_digest: HumanoidEvidenceDigest,
    inner: HumanoidReachStrongEpisodeStageArtifact,
}

impl std::fmt::Debug for HumanoidReachManifestEpisodeStageArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidReachManifestEpisodeStageArtifact")
            .field("purpose", &self.purpose)
            .field("manifest_requirement_digest", &self.manifest_requirement_digest)
            .field("manifest_corpus_digest", &self.manifest_corpus_digest)
            .field("artifact_digest", &self.artifact_digest)
            .finish()
    }
}

impl HumanoidReachManifestEpisodeStageArtifact {
    pub const fn artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.artifact_digest
    }

    fn validate_shape(&self) -> bool {
        self.purpose.is_qualification()
            && !self.manifest_requirement_digest.is_zero()
            && !self.manifest_corpus_digest.is_zero()
            && self.lower_artifact_digest == self.inner.artifact_digest()
            && !self.lower_artifact_digest.is_zero()
            && !self.artifact_digest.is_zero()
            && self.artifact_digest == digest_episode_stage(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachManifestStageIssueFailure {
    InvalidManifestEvidence,
    ManifestPolicyMismatch,
    LowerStageIssue,
    InvalidArtifact,
}

#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_manifest_step_stage(
    subject: &HumanoidQualificationSubject,
    requirement: &HumanoidReachManifestStageRequirement,
    step_policy: &HumanoidReachLineageCampaignPolicy,
    command_policy: &HumanoidReachCommandEvidencePolicy,
    outcome_policy: &HumanoidReachOutcomeEvidencePolicy,
    trials: &[HumanoidReachManifestBoundTrial],
    issued_unix_millis: u64,
) -> Result<HumanoidReachManifestStepStageArtifact, HumanoidReachManifestStageIssueFailure> {
    if requirement.purpose != step_policy.campaign.required_execution_purpose
        || !validate_trial_manifests(requirement.step_scenarios(), trials)
    {
        return Err(HumanoidReachManifestStageIssueFailure::ManifestPolicyMismatch);
    }
    let manifest_corpus_digest = digest_manifest_trial_corpus(trials)
        .ok_or(HumanoidReachManifestStageIssueFailure::InvalidManifestEvidence)?;
    let lower_trials = trials.iter().map(|trial| trial.inner.clone()).collect::<Vec<_>>();
    let inner = issue_humanoid_reach_strong_step_stage(
        subject,
        step_policy,
        command_policy,
        outcome_policy,
        &lower_trials,
        issued_unix_millis,
    )
    .map_err(|_: HumanoidReachStrongStageIssueFailure| HumanoidReachManifestStageIssueFailure::LowerStageIssue)?;
    let mut artifact = HumanoidReachManifestStepStageArtifact {
        purpose: requirement.purpose,
        manifest_requirement_digest: requirement.requirement_digest,
        manifest_corpus_digest,
        lower_artifact_digest: inner.artifact_digest(),
        artifact_digest: HumanoidEvidenceDigest::ZERO,
        inner,
    };
    artifact.artifact_digest = digest_step_stage(&artifact);
    if !artifact.validate_shape() {
        return Err(HumanoidReachManifestStageIssueFailure::InvalidArtifact);
    }
    Ok(artifact)
}

#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_manifest_episode_stage(
    subject: &HumanoidQualificationSubject,
    requirement: &HumanoidReachManifestStageRequirement,
    campaign_policy: &HumanoidReachEpisodeCampaignPolicy,
    episode_policy: &HumanoidReachEpisodePolicy,
    command_policy: &HumanoidReachCommandEvidencePolicy,
    outcome_policy: &HumanoidReachOutcomeEvidencePolicy,
    cases: &[HumanoidReachManifestBoundEpisodeCase],
    issued_unix_millis: u64,
) -> Result<HumanoidReachManifestEpisodeStageArtifact, HumanoidReachManifestStageIssueFailure> {
    if requirement.purpose != campaign_policy.required_execution_purpose
        || !validate_episode_manifests(requirement.episode_scenarios(), cases)
    {
        return Err(HumanoidReachManifestStageIssueFailure::ManifestPolicyMismatch);
    }
    let manifest_corpus_digest = digest_manifest_episode_corpus(cases)
        .ok_or(HumanoidReachManifestStageIssueFailure::InvalidManifestEvidence)?;
    let lower_cases = cases.iter().map(|case| case.inner.clone()).collect::<Vec<_>>();
    let inner = issue_humanoid_reach_strong_episode_stage(
        subject,
        campaign_policy,
        episode_policy,
        command_policy,
        outcome_policy,
        &lower_cases,
        issued_unix_millis,
    )
    .map_err(|_: HumanoidReachStrongStageIssueFailure| HumanoidReachManifestStageIssueFailure::LowerStageIssue)?;
    let mut artifact = HumanoidReachManifestEpisodeStageArtifact {
        purpose: requirement.purpose,
        manifest_requirement_digest: requirement.requirement_digest,
        manifest_corpus_digest,
        lower_artifact_digest: inner.artifact_digest(),
        artifact_digest: HumanoidEvidenceDigest::ZERO,
        inner,
    };
    artifact.artifact_digest = digest_episode_stage(&artifact);
    if !artifact.validate_shape() {
        return Err(HumanoidReachManifestStageIssueFailure::InvalidArtifact);
    }
    Ok(artifact)
}

pub struct HumanoidReachManifestOperationalArtifact {
    policy_digest: HumanoidEvidenceDigest,
    simulation_step_digest: HumanoidEvidenceDigest,
    simulation_episode_digest: HumanoidEvidenceDigest,
    hil_step_digest: HumanoidEvidenceDigest,
    hil_episode_digest: HumanoidEvidenceDigest,
    physical_step_digest: HumanoidEvidenceDigest,
    physical_episode_digest: HumanoidEvidenceDigest,
    lower_artifact_digest: HumanoidEvidenceDigest,
    artifact_digest: HumanoidEvidenceDigest,
    inner: HumanoidReachStrongOperationalArtifact,
}

impl std::fmt::Debug for HumanoidReachManifestOperationalArtifact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidReachManifestOperationalArtifact")
            .field("policy_digest", &self.policy_digest)
            .field("artifact_digest", &self.artifact_digest)
            .finish()
    }
}

impl HumanoidReachManifestOperationalArtifact {
    pub const fn artifact_digest(&self) -> HumanoidEvidenceDigest {
        self.artifact_digest
    }

    pub fn validate_at(
        &self,
        subject: &HumanoidQualificationSubject,
        policy: &HumanoidReachManifestOperationalPolicy,
        now_unix_millis: u64,
    ) -> bool {
        policy.validate_for(subject)
            && self.policy_digest == policy.policy_digest
            && self.inner.validate_at(subject, &policy.base, now_unix_millis)
            && self.lower_artifact_digest == self.inner.artifact_digest()
            && !self.lower_artifact_digest.is_zero()
            && !self.artifact_digest.is_zero()
            && self.artifact_digest == digest_operational_artifact(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachManifestPromotionFailure {
    InvalidPolicy,
    InvalidStage,
    LowerPromotion,
    InvalidArtifact,
}

#[allow(clippy::too_many_arguments)]
pub fn promote_humanoid_reach_manifest_to_operational(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachManifestOperationalPolicy,
    simulation_step: &HumanoidReachManifestStepStageArtifact,
    simulation_episode: &HumanoidReachManifestEpisodeStageArtifact,
    hil_step: &HumanoidReachManifestStepStageArtifact,
    hil_episode: &HumanoidReachManifestEpisodeStageArtifact,
    physical_step: &HumanoidReachManifestStepStageArtifact,
    physical_episode: &HumanoidReachManifestEpisodeStageArtifact,
    now_unix_millis: u64,
) -> Result<HumanoidReachManifestOperationalArtifact, HumanoidReachManifestPromotionFailure> {
    if !policy.validate_for(subject) {
        return Err(HumanoidReachManifestPromotionFailure::InvalidPolicy);
    }
    for artifact in [simulation_step, hil_step, physical_step] {
        if !artifact.validate_shape() {
            return Err(HumanoidReachManifestPromotionFailure::InvalidStage);
        }
    }
    for artifact in [simulation_episode, hil_episode, physical_episode] {
        if !artifact.validate_shape() {
            return Err(HumanoidReachManifestPromotionFailure::InvalidStage);
        }
    }
    for (required, step, episode) in [
        (&policy.simulation, simulation_step, simulation_episode),
        (&policy.hil, hil_step, hil_episode),
        (&policy.physical, physical_step, physical_episode),
    ] {
        if step.purpose != required.purpose
            || episode.purpose != required.purpose
            || step.manifest_requirement_digest != required.requirement_digest
            || episode.manifest_requirement_digest != required.requirement_digest
        {
            return Err(HumanoidReachManifestPromotionFailure::InvalidStage);
        }
    }
    let inner = promote_humanoid_reach_strong_to_operational(
        subject,
        &policy.base,
        &simulation_step.inner,
        &simulation_episode.inner,
        &hil_step.inner,
        &hil_episode.inner,
        &physical_step.inner,
        &physical_episode.inner,
        now_unix_millis,
    )
    .map_err(|_: HumanoidReachStrongPromotionFailure| HumanoidReachManifestPromotionFailure::LowerPromotion)?;
    let mut artifact = HumanoidReachManifestOperationalArtifact {
        policy_digest: policy.policy_digest,
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
        return Err(HumanoidReachManifestPromotionFailure::InvalidArtifact);
    }
    Ok(artifact)
}

/// Operator approval bound to the exact manifest-aware operational policy.
pub struct HumanoidReachManifestOperatorApproval {
    policy_digest: HumanoidEvidenceDigest,
    approval_digest: HumanoidEvidenceDigest,
    inner: crate::reach_cryptographic_authority::HumanoidReachCryptographicOperatorApproval,
}

impl std::fmt::Debug for HumanoidReachManifestOperatorApproval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HumanoidReachManifestOperatorApproval")
            .field("policy_digest", &self.policy_digest)
            .field("approval_digest", &self.approval_digest)
            .finish()
    }
}

impl HumanoidReachManifestOperatorApproval {
    #[allow(clippy::too_many_arguments)]
    pub fn bind_upstream(
        subject: &HumanoidQualificationSubject,
        policy: &HumanoidReachManifestOperationalPolicy,
        operational_scope_id: impl Into<String>,
        approval_id: impl Into<String>,
        operator_source: HumanoidAuthoritySourceSnapshot,
        approved_at_s: f64,
        valid_until_s: f64,
    ) -> Option<Self> {
        if !policy.validate_for(subject) {
            return None;
        }
        let inner = crate::reach_cryptographic_authority::HumanoidReachCryptographicOperatorApproval::bind_upstream(
            subject,
            &policy.base,
            operational_scope_id,
            approval_id,
            operator_source,
            approved_at_s,
            valid_until_s,
        )?;
        let policy_digest = policy.policy_digest;
        let approval_digest = digest_operator_approval(policy_digest, inner.approval_digest());
        if approval_digest.is_zero() {
            return None;
        }
        Some(Self {
            policy_digest,
            approval_digest,
            inner,
        })
    }

    pub const fn approval_digest(&self) -> HumanoidEvidenceDigest {
        self.approval_digest
    }

    pub fn operational_scope_id(&self) -> &str {
        self.inner.operational_scope_id()
    }

    pub fn validate_at(
        &self,
        subject: &HumanoidQualificationSubject,
        policy: &HumanoidReachManifestOperationalPolicy,
        now_s: f64,
    ) -> bool {
        policy.validate_for(subject)
            && self.policy_digest == policy.policy_digest
            && self.inner.validate_at(subject, &policy.base, now_s)
            && !self.approval_digest.is_zero()
            && self.approval_digest
                == digest_operator_approval(self.policy_digest, self.inner.approval_digest())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_manifest_operational_authority_receipt(
    subject: &HumanoidQualificationSubject,
    permit: &HumanoidSkillExecutionPermit<'_>,
    qualification: &HumanoidReachManifestOperationalArtifact,
    policy: &HumanoidReachManifestOperationalPolicy,
    operator_approval: &HumanoidReachManifestOperatorApproval,
    physical: HumanoidAuthoritySourceSnapshot,
    epistemic: HumanoidAuthoritySourceSnapshot,
    cognitive: HumanoidAuthoritySourceSnapshot,
    now_s: f64,
    now_unix_millis: u64,
) -> Result<HumanoidScopedSkillAuthorityReceipt, HumanoidReachCryptographicAuthorityIssueFailure> {
    if !qualification.validate_at(subject, policy, now_unix_millis)
        || !operator_approval.validate_at(subject, policy, now_s)
    {
        return Err(HumanoidReachCryptographicAuthorityIssueFailure::InvalidQualification);
    }
    issue_humanoid_reach_strong_operational_authority_receipt(
        subject,
        permit,
        &qualification.inner,
        &policy.base,
        &operator_approval.inner,
        physical,
        epistemic,
        cognitive,
        now_s,
        now_unix_millis,
    )
}

fn validate_manifest_requirements(values: &[HumanoidReachScenarioManifestRequirement]) -> bool {
    if values.is_empty() || values.iter().any(|value| !value.validate()) {
        return false;
    }
    let mut seen = BTreeSet::new();
    values.iter().all(|value| seen.insert(value.scenario_id.as_str()))
}

fn validate_trial_manifests(
    requirements: &[HumanoidReachScenarioManifestRequirement],
    trials: &[HumanoidReachManifestBoundTrial],
) -> bool {
    if trials.is_empty() || trials.iter().any(|trial| !trial.validate()) {
        return false;
    }
    let required = requirements
        .iter()
        .map(|item| (item.scenario_id.as_str(), item.manifest_digest))
        .collect::<BTreeMap<_, _>>();
    if required.len() != requirements.len() {
        return false;
    }
    let mut seen = BTreeSet::new();
    for trial in trials {
        let Some(expected) = required.get(trial.scenario_id.as_str()) else {
            return false;
        };
        if *expected != trial.manifest_digest {
            return false;
        }
        seen.insert(trial.scenario_id.as_str());
    }
    seen.len() == required.len()
}

fn validate_episode_manifests(
    requirements: &[HumanoidReachScenarioManifestRequirement],
    cases: &[HumanoidReachManifestBoundEpisodeCase],
) -> bool {
    if cases.is_empty() || cases.iter().any(|case| !case.validate()) {
        return false;
    }
    let required = requirements
        .iter()
        .map(|item| (item.scenario_id.as_str(), item.manifest_digest))
        .collect::<BTreeMap<_, _>>();
    if required.len() != requirements.len() {
        return false;
    }
    let mut seen = BTreeSet::new();
    for case in cases {
        let Some(expected) = required.get(case.scenario_id.as_str()) else {
            return false;
        };
        if *expected != case.manifest_digest {
            return false;
        }
        seen.insert(case.scenario_id.as_str());
    }
    seen.len() == required.len()
}

fn digest_stage_requirement(requirement: &HumanoidReachManifestStageRequirement) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.manifest-stage-requirement.v1");
    h.u32(requirement.schema_version)
        .u64(purpose_id(requirement.purpose));
    hash_manifest_requirements(&mut h, &requirement.step_scenarios);
    hash_manifest_requirements(&mut h, &requirement.episode_scenarios);
    h.finish()
}

fn hash_manifest_requirements(
    h: &mut HumanoidEvidenceHasher,
    values: &[HumanoidReachScenarioManifestRequirement],
) {
    let mut ordered = values.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    h.usize(ordered.len());
    for item in ordered {
        h.string(&item.scenario_id).digest(item.manifest_digest);
    }
}

fn digest_operational_policy(policy: &HumanoidReachManifestOperationalPolicy) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.manifest-operational-policy.v1");
    h.u32(policy.schema_version)
        .string(&policy.policy_id)
        .digest(policy.base.policy_digest())
        .digest(policy.simulation.requirement_digest)
        .digest(policy.hil.requirement_digest)
        .digest(policy.physical.requirement_digest);
    h.finish()
}

fn digest_manifest_trial(
    scenario_id: &str,
    manifest_digest: HumanoidEvidenceDigest,
    lower_binding_digest: HumanoidEvidenceDigest,
) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.manifest-bound-trial.v1");
    h.u32(HUMANOID_REACH_MANIFEST_BOUND_EVIDENCE_SCHEMA_VERSION)
        .string(scenario_id)
        .digest(manifest_digest)
        .digest(lower_binding_digest);
    h.finish()
}

fn digest_manifest_episode_case(
    scenario_id: &str,
    manifest_digest: HumanoidEvidenceDigest,
    lower_case_digest: HumanoidEvidenceDigest,
) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.manifest-bound-episode-case.v1");
    h.u32(HUMANOID_REACH_MANIFEST_BOUND_EVIDENCE_SCHEMA_VERSION)
        .string(scenario_id)
        .digest(manifest_digest)
        .digest(lower_case_digest);
    h.finish()
}

fn digest_manifest_trial_corpus(
    trials: &[HumanoidReachManifestBoundTrial],
) -> Option<HumanoidEvidenceDigest> {
    if trials.is_empty() || trials.iter().any(|trial| !trial.validate()) {
        return None;
    }
    let mut ordered = trials.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        left.scenario_id
            .cmp(&right.scenario_id)
            .then(left.binding_digest.cmp(&right.binding_digest))
    });
    let mut h = HumanoidEvidenceHasher::new("reach.manifest-trial-corpus.v1");
    h.usize(ordered.len());
    for trial in ordered {
        h.digest(trial.binding_digest);
    }
    Some(h.finish())
}

fn digest_manifest_episode_corpus(
    cases: &[HumanoidReachManifestBoundEpisodeCase],
) -> Option<HumanoidEvidenceDigest> {
    if cases.is_empty() || cases.iter().any(|case| !case.validate()) {
        return None;
    }
    let mut ordered = cases.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        left.scenario_id
            .cmp(&right.scenario_id)
            .then(left.binding_digest.cmp(&right.binding_digest))
    });
    let mut h = HumanoidEvidenceHasher::new("reach.manifest-episode-corpus.v1");
    h.usize(ordered.len());
    for case in ordered {
        h.digest(case.binding_digest);
    }
    Some(h.finish())
}

fn digest_step_stage(stage: &HumanoidReachManifestStepStageArtifact) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.manifest-step-stage.v1");
    h.u32(HUMANOID_REACH_MANIFEST_STAGE_ARTIFACT_SCHEMA_VERSION)
        .u64(purpose_id(stage.purpose))
        .digest(stage.manifest_requirement_digest)
        .digest(stage.manifest_corpus_digest)
        .digest(stage.lower_artifact_digest);
    h.finish()
}

fn digest_episode_stage(stage: &HumanoidReachManifestEpisodeStageArtifact) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.manifest-episode-stage.v1");
    h.u32(HUMANOID_REACH_MANIFEST_STAGE_ARTIFACT_SCHEMA_VERSION)
        .u64(purpose_id(stage.purpose))
        .digest(stage.manifest_requirement_digest)
        .digest(stage.manifest_corpus_digest)
        .digest(stage.lower_artifact_digest);
    h.finish()
}

fn digest_operational_artifact(artifact: &HumanoidReachManifestOperationalArtifact) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.manifest-operational-artifact.v1");
    h.u32(HUMANOID_REACH_MANIFEST_OPERATIONAL_ARTIFACT_SCHEMA_VERSION)
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

fn digest_operator_approval(
    policy_digest: HumanoidEvidenceDigest,
    lower_approval_digest: HumanoidEvidenceDigest,
) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.manifest-operator-approval.v1");
    h.u32(HUMANOID_REACH_MANIFEST_OPERATOR_APPROVAL_SCHEMA_VERSION)
        .digest(policy_digest)
        .digest(lower_approval_digest);
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

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_requirement_is_order_independent() {
        let a = HumanoidReachScenarioManifestRequirement {
            scenario_id: "left-edge".into(),
            manifest_digest: HumanoidEvidenceDigest::from_bytes([1; 32]),
        };
        let b = HumanoidReachScenarioManifestRequirement {
            scenario_id: "right-edge".into(),
            manifest_digest: HumanoidEvidenceDigest::from_bytes([2; 32]),
        };
        let left = HumanoidReachManifestStageRequirement::new(
            HumanoidExecutionPurpose::SimulationQualification,
            vec![a.clone(), b.clone()],
            vec![a.clone(), b.clone()],
        )
        .unwrap();
        let right = HumanoidReachManifestStageRequirement::new(
            HumanoidExecutionPurpose::SimulationQualification,
            vec![b.clone(), a.clone()],
            vec![b, a],
        )
        .unwrap();
        assert_eq!(left.requirement_digest(), right.requirement_digest());
    }

    #[test]
    fn manifest_change_changes_stage_requirement() {
        let a = HumanoidReachManifestStageRequirement::new(
            HumanoidExecutionPurpose::PhysicalQualification,
            vec![HumanoidReachScenarioManifestRequirement {
                scenario_id: "boundary".into(),
                manifest_digest: HumanoidEvidenceDigest::from_bytes([3; 32]),
            }],
            vec![HumanoidReachScenarioManifestRequirement {
                scenario_id: "boundary".into(),
                manifest_digest: HumanoidEvidenceDigest::from_bytes([3; 32]),
            }],
        )
        .unwrap();
        let b = HumanoidReachManifestStageRequirement::new(
            HumanoidExecutionPurpose::PhysicalQualification,
            vec![HumanoidReachScenarioManifestRequirement {
                scenario_id: "boundary".into(),
                manifest_digest: HumanoidEvidenceDigest::from_bytes([4; 32]),
            }],
            vec![HumanoidReachScenarioManifestRequirement {
                scenario_id: "boundary".into(),
                manifest_digest: HumanoidEvidenceDigest::from_bytes([4; 32]),
            }],
        )
        .unwrap();
        assert_ne!(a.requirement_digest(), b.requirement_digest());
    }
}
