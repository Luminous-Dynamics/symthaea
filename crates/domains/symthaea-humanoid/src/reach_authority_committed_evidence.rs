// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reach evidence wrappers that require SHA-256 finalized-authority commitments.

use crate::evidence_digest::{HumanoidEvidenceDigest, HumanoidEvidenceHasher};
use crate::execution_authority_scope::{HumanoidExecutionPurpose, HumanoidQualificationAuthorityBasis};
use crate::morphology::HandSide;
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_authority_commitment::HumanoidReachAuthorityCommitment;
use crate::reach_episode_evidence::{
    HumanoidReachEpisodeStepBindFailure, HumanoidReachEpisodeStepEvidence,
    bind_humanoid_reach_episode_step,
};
use crate::reach_episode_promotion::HumanoidReachEpisodeQualificationCase;
use crate::reach_execution::HumanoidPermittedReachExecutionResult;
use crate::reach_execution_evidence::{HumanoidReachCommandEvidence, HumanoidReachCommandEvidencePolicy};
use crate::reach_outcome_evidence::{HumanoidReachOutcomeEvidencePolicy, HumanoidReachStepEvidenceAssessment};
use crate::reach_qualification_campaign::HumanoidReachScenarioCell;
use crate::reach_qualification_lineage::{
    HumanoidReachLineageBoundTrial, HumanoidReachLineageTrialBindFailure,
    HumanoidReachPerturbationProfileBinding, bind_lineage_humanoid_reach_qualification_trial,
};
use crate::reach_spatial_goal_commitment::HumanoidReachSpatialGoalCommitment;
use crate::types::HumanoidState;

pub const HUMANOID_REACH_AUTHORITY_COMMITTED_EVIDENCE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachAuthorityCommittedTrial {
    pub(crate) inner: HumanoidReachLineageBoundTrial,
    authority: HumanoidReachAuthorityCommitment,
    spatial: HumanoidReachSpatialGoalCommitment,
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
        let spatial = HumanoidReachSpatialGoalCommitment::from_preparation(subject, &result.preparation)
            .ok_or(HumanoidReachAuthorityCommittedTrialBindFailure::InvalidSpatialGoalCommitment)?;
        let inner = bind_lineage_humanoid_reach_qualification_trial(
            subject, scenario, perturbation, trial_id, trial_seed, result, step,
            command_policy, outcome_policy,
        )
        .map_err(HumanoidReachAuthorityCommittedTrialBindFailure::Lower)?;
        if inner.trial.authority_receipt_fingerprint != result.authority_receipt.receipt_fingerprint
            || inner.trial.authority_scope_fingerprint != result.authority_receipt.scope_fingerprint
            || inner.trial.authority_scope_id.as_str()
                != result.authority_receipt.scope_id.as_str()
            || inner.trial.spatial_goal_fingerprint != result.preparation.spatial_goal_fingerprint
        {
            return Err(HumanoidReachAuthorityCommittedTrialBindFailure::AuthorityLineageMismatch);
        }
        let binding_digest = digest_trial(&inner, authority, spatial);
        if binding_digest.is_zero() {
            return Err(HumanoidReachAuthorityCommittedTrialBindFailure::InvalidBindingDigest);
        }
        Ok(Self { inner, authority, spatial, binding_digest })
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

    pub const fn spatial_goal_digest(&self) -> HumanoidEvidenceDigest {
        self.spatial.goal_digest()
    }

    pub const fn binding_digest(&self) -> HumanoidEvidenceDigest {
        self.binding_digest
    }

    pub(crate) fn validate(&self) -> bool {
        self.inner.validate()
            && !self.authority.finalization_digest().is_zero()
            && !self.spatial.goal_digest().is_zero()
            && !self.binding_digest.is_zero()
            && self.binding_digest == digest_trial(&self.inner, self.authority, self.spatial)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachAuthorityCommittedTrialBindFailure {
    InvalidAuthorityCommitment,
    InvalidSpatialGoalCommitment,
    Lower(HumanoidReachLineageTrialBindFailure),
    AuthorityLineageMismatch,
    InvalidBindingDigest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachAuthorityCommittedEpisodeStep {
    pub(crate) inner: HumanoidReachEpisodeStepEvidence,
    authority: HumanoidReachAuthorityCommitment,
    spatial: HumanoidReachSpatialGoalCommitment,
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
            .ok_or(HumanoidReachAuthorityCommittedEpisodeStepBindFailure::InvalidAuthorityCommitment)?;
        let spatial = HumanoidReachSpatialGoalCommitment::from_preparation(subject, &result.preparation)
            .ok_or(HumanoidReachAuthorityCommittedEpisodeStepBindFailure::InvalidSpatialGoalCommitment)?;
        let inner = bind_humanoid_reach_episode_step(
            subject, command_policy, outcome_policy, command_evidence, result, post_state, received_at_s,
        )
        .map_err(HumanoidReachAuthorityCommittedEpisodeStepBindFailure::Lower)?;
        if inner.authority_receipt_fingerprint != result.authority_receipt.receipt_fingerprint
            || inner.authority_scope_fingerprint != result.authority_receipt.scope_fingerprint
            || inner.authority_scope_id.as_str() != result.authority_receipt.scope_id.as_str()
            || inner.spatial_goal_fingerprint != result.preparation.spatial_goal_fingerprint
        {
            return Err(HumanoidReachAuthorityCommittedEpisodeStepBindFailure::AuthorityLineageMismatch);
        }
        let binding_digest = digest_episode_step(&inner, authority, spatial);
        if binding_digest.is_zero() {
            return Err(HumanoidReachAuthorityCommittedEpisodeStepBindFailure::InvalidBindingDigest);
        }
        Ok(Self { inner, authority, spatial, binding_digest })
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

    pub const fn spatial_goal_digest(&self) -> HumanoidEvidenceDigest {
        self.spatial.goal_digest()
    }

    pub const fn binding_digest(&self) -> HumanoidEvidenceDigest {
        self.binding_digest
    }

    pub(crate) fn validate(&self) -> bool {
        self.inner.validate()
            && !self.authority.finalization_digest().is_zero()
            && !self.spatial.goal_digest().is_zero()
            && !self.binding_digest.is_zero()
            && self.binding_digest == digest_episode_step(&self.inner, self.authority, self.spatial)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachAuthorityCommittedEpisodeStepBindFailure {
    InvalidAuthorityCommitment,
    InvalidSpatialGoalCommitment,
    Lower(HumanoidReachEpisodeStepBindFailure),
    AuthorityLineageMismatch,
    InvalidBindingDigest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachAuthorityCommittedEpisodeCase {
    pub(crate) inner: HumanoidReachEpisodeQualificationCase,
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
        let case_digest = digest_case(&inner, &steps);
        (!case_digest.is_zero()).then_some(Self { inner, steps, case_digest })
    }

    pub const fn case_digest(&self) -> HumanoidEvidenceDigest {
        self.case_digest
    }

    pub(crate) fn steps(&self) -> &[HumanoidReachAuthorityCommittedEpisodeStep] {
        &self.steps
    }

    pub(crate) fn validate(&self) -> bool {
        self.inner.validate_shape()
            && self.steps.len() == self.inner.steps.len()
            && self.steps.iter().all(HumanoidReachAuthorityCommittedEpisodeStep::validate)
            && self.steps.iter().zip(&self.inner.steps).all(|(a, b)| &a.inner == b)
            && !self.case_digest.is_zero()
            && self.case_digest == digest_case(&self.inner, &self.steps)
    }
}

fn digest_trial(
    bound: &HumanoidReachLineageBoundTrial,
    authority: HumanoidReachAuthorityCommitment,
    spatial: HumanoidReachSpatialGoalCommitment,
) -> HumanoidEvidenceDigest {
    let t = &bound.trial;
    let mut h = HumanoidEvidenceHasher::new("reach.authority-committed-trial.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITTED_EVIDENCE_SCHEMA_VERSION)
        .u32(bound.schema_version).u32(t.schema_version).u64(t.subject_fingerprint)
        .string(&t.scenario_id).string(&t.perturbation_profile_id).string(&t.trial_id)
        .u64(t.trial_seed).u64(hand_id(t.hand)).f64(t.workspace_utilization_sq)
        .u64(t.validation_epoch).string(&t.goal_id).u64(t.spatial_goal_fingerprint)
        .string(&t.command_policy_id).string(&t.outcome_policy_id)
        .string(&t.authority_scope_id).u64(purpose_id(t.execution_purpose))
        .u64(basis_id(t.qualification_basis)).f32(t.authority_effective_scale)
        .bool(t.step_accepted).u64(bound.command_policy_fingerprint)
        .u64(bound.outcome_policy_fingerprint)
        .u64(bound.perturbation_configuration_fingerprint)
        .digest(spatial.goal_digest())
        .digest(authority.receipt_digest()).digest(authority.scope_digest())
        .digest(authority.finalization_digest());
    h.finish()
}

fn digest_episode_step(
    step: &HumanoidReachEpisodeStepEvidence,
    authority: HumanoidReachAuthorityCommitment,
    spatial: HumanoidReachSpatialGoalCommitment,
) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.authority-committed-episode-step.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITTED_EVIDENCE_SCHEMA_VERSION)
        .u32(step.schema_version).u64(step.subject_fingerprint).u64(step.validation_epoch)
        .string(&step.goal_id).u64(step.spatial_goal_fingerprint).u64(hand_id(step.hand));
    for value in step.target_world_m { h.f64(value); }
    h.f64(step.prepared_at_s).f64(step.observed_at_s).f64(step.received_at_s)
        .f64(step.pre_error_m).f64(step.post_error_m).f64(step.progress_m)
        .string(&step.command_policy_id).string(&step.outcome_policy_id)
        .string(&step.authority_scope_id).u64(purpose_id(step.execution_purpose))
        .u64(basis_id(step.qualification_basis)).bool(step.step_accepted)
        .digest(spatial.goal_digest())
        .digest(authority.receipt_digest()).digest(authority.scope_digest())
        .digest(authority.finalization_digest());
    h.finish()
}

fn digest_case(
    case: &HumanoidReachEpisodeQualificationCase,
    steps: &[HumanoidReachAuthorityCommittedEpisodeStep],
) -> HumanoidEvidenceDigest {
    let mut h = HumanoidEvidenceHasher::new("reach.authority-committed-episode-case.v1");
    h.u32(HUMANOID_REACH_AUTHORITY_COMMITTED_EVIDENCE_SCHEMA_VERSION)
        .string(&case.scenario_id).string(&case.perturbation_profile_id)
        .u64(case.perturbation_configuration_fingerprint).string(&case.episode_id)
        .u64(case.episode_seed).usize(steps.len());
    for step in steps { h.digest(step.binding_digest()); }
    h.finish()
}

fn hand_id(hand: HandSide) -> u64 {
    match hand { HandSide::Right => 1, HandSide::Left => 2 }
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
