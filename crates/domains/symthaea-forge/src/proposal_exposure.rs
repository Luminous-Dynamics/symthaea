// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Proposal/exposure evidence required before Forge outcomes may inform a learned search policy.
//!
//! Forge's current mutator samples uniformly over eligible `(operator, site)` pairs. Operator
//! families therefore have unequal proposal exposure when their eligible-site counts differ. Raw
//! family outcome rates remain policy-confounded unless the opportunity set and selected pair are
//! retained. This module records that information without granting search, promotion, or runtime
//! authority.

use crate::family_learning::{
    ForgeFamilyLearningError, ForgeFamilyTrialSet, ForgeTransformationFamilyId,
};
use crate::learning::{ForgeLearningError, ForgeTrialBatch};
use crate::trace::{ForgeAttemptId, ForgeTraceEvent};
use crate::trial_semantics::NO_CANDIDATE_SCHEMA;
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use symthaea_algorithms::discovery::DiscoveryRun;
use symthaea_algorithms::ledger::DiscoveryEventKind;
use symthaea_algorithms::observation::ObservationStore;
use symthaea_algorithms::{ContentId, ImplementationRecord};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalExposureError {
    #[error(transparent)]
    Family(#[from] ForgeFamilyLearningError),
    #[error(transparent)]
    Learning(#[from] ForgeLearningError),
    #[error("proposal policy requires at least one registered family")]
    EmptyPolicy,
    #[error("proposal policy contains a duplicate family")]
    DuplicatePolicyFamily,
    #[error("proposal policy mixes generator identities")]
    PolicyGeneratorMismatch,
    #[error("proposal policy identity does not match canonical fields")]
    PolicyIdentityMismatch,
    #[error("proposal opportunity profile does not exactly match policy family order")]
    OpportunityPolicyMismatch,
    #[error("proposal opportunity count overflow")]
    OpportunityCountOverflow,
    #[error("proposal decision is inconsistent with its opportunity profile")]
    InvalidDecision,
    #[error("proposal exposure attempt generation does not match attempt identity")]
    GenerationMismatch,
    #[error("proposal exposure identity does not match canonical fields")]
    ExposureIdentityMismatch,
    #[error("proposal exposure archive contains duplicate/non-canonical attempts")]
    NonCanonicalAttemptOrder,
    #[error("proposal exposure archive mixes discovery runs or policies")]
    ArchiveScopeMismatch,
    #[error("proposal exposure archive identity does not match canonical fields")]
    ArchiveIdentityMismatch,
    #[error("proposal-learning qualification requires exposure for every proposal-stage attempt")]
    MissingExposure,
    #[error("proposal-learning qualification contains exposure for an unknown attempt")]
    ExtraExposure,
    #[error("candidate exposure did not select the exact observed transformation family")]
    SelectedFamilyMismatch,
    #[error("proposal exposure parent artifact disagrees with observed parent")]
    ParentArtifactMismatch,
    #[error("GeneratorNoOp observation is invalid or unsupported for policy-learning qualification")]
    InvalidNoOp,
    #[error("attempt failed before proposal opportunity enumeration and is ineligible for policy learning")]
    PreProposalAttempt,
    #[error("family and concrete trial projections do not share the same run-bound source")]
    TrialProjectionMismatch,
    #[error("policy-learning qualification identity does not match canonical fields")]
    QualificationIdentityMismatch,
}

/// Exact proposal mechanism represented by v1 exposure evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ForgeProposalRule {
    /// Uniform choice over every currently eligible `(operator, site)` pair.
    UniformEligibleSiteV1,
}

impl ForgeProposalRule {
    fn tag(self) -> &'static [u8] {
        match self {
            Self::UniformEligibleSiteV1 => b"uniform-eligible-site-v1",
        }
    }
}

/// Ordered operator-family registry plus exact proposal rule under one generator implementation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalPolicy {
    id: ContentId,
    generator_id: ContentId,
    rule: ForgeProposalRule,
    families: Vec<ForgeTransformationFamilyId>,
}

impl ForgeProposalPolicy {
    pub fn uniform_eligible_site<I, S>(
        generator_id: ContentId,
        operator_names: I,
    ) -> Result<Self, ForgeProposalExposureError>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let mut families = Vec::new();
        let mut seen = BTreeSet::new();
        for operator in operator_names {
            let family = ForgeTransformationFamilyId::new(generator_id.clone(), operator.into())?;
            if !seen.insert(family.as_content_id().as_str().to_string()) {
                return Err(ForgeProposalExposureError::DuplicatePolicyFamily);
            }
            families.push(family);
        }
        if families.is_empty() {
            return Err(ForgeProposalExposureError::EmptyPolicy);
        }
        let rule = ForgeProposalRule::UniformEligibleSiteV1;
        let id = derive_policy_id(&generator_id, rule, &families);
        Ok(Self {
            id,
            generator_id,
            rule,
            families,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn generator_id(&self) -> &ContentId { &self.generator_id }
    pub fn rule(&self) -> ForgeProposalRule { self.rule }
    pub fn families(&self) -> &[ForgeTransformationFamilyId] { &self.families }

    pub fn validate(&self) -> Result<(), ForgeProposalExposureError> {
        if self.families.is_empty() {
            return Err(ForgeProposalExposureError::EmptyPolicy);
        }
        let mut seen = BTreeSet::new();
        for family in &self.families {
            family.validate()?;
            if family.generator_id() != &self.generator_id {
                return Err(ForgeProposalExposureError::PolicyGeneratorMismatch);
            }
            if !seen.insert(family.as_content_id().as_str().to_string()) {
                return Err(ForgeProposalExposureError::DuplicatePolicyFamily);
            }
        }
        if derive_policy_id(&self.generator_id, self.rule, &self.families) == self.id {
            Ok(())
        } else {
            Err(ForgeProposalExposureError::PolicyIdentityMismatch)
        }
    }
}

fn derive_policy_id(
    generator_id: &ContentId,
    rule: ForgeProposalRule,
    families: &[ForgeTransformationFamilyId],
) -> ContentId {
    let count = (families.len() as u64).to_be_bytes();
    let mut parts = vec![
        generator_id.as_str().as_bytes().to_vec(),
        rule.tag().to_vec(),
        count.to_vec(),
    ];
    parts.extend(
        families
            .iter()
            .map(|family| family.as_content_id().as_str().as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.forge-proposal-policy.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeFamilyOpportunity {
    family_id: ForgeTransformationFamilyId,
    eligible_sites: u64,
}

impl ForgeFamilyOpportunity {
    pub fn new(family_id: ForgeTransformationFamilyId, eligible_sites: u64) -> Self {
        Self { family_id, eligible_sites }
    }

    pub fn family_id(&self) -> &ForgeTransformationFamilyId { &self.family_id }
    pub fn eligible_sites(&self) -> u64 { self.eligible_sites }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalSelection {
    family_id: ForgeTransformationFamilyId,
    site_index: u64,
    global_pair_index: u64,
}

impl ForgeProposalSelection {
    pub fn new(
        family_id: ForgeTransformationFamilyId,
        site_index: u64,
        global_pair_index: u64,
    ) -> Self {
        Self { family_id, site_index, global_pair_index }
    }

    pub fn family_id(&self) -> &ForgeTransformationFamilyId { &self.family_id }
    pub fn site_index(&self) -> u64 { self.site_index }
    pub fn global_pair_index(&self) -> u64 { self.global_pair_index }
}

/// Result of proposal enumeration. A selected pair may still render identical source later.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ForgeProposalDecision {
    NoEligibleSites,
    Selected(ForgeProposalSelection),
}

/// Opportunity/exposure record for one attempt that reached mutation proposal enumeration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalExposure {
    id: ContentId,
    run_id: ContentId,
    attempt_id: ForgeAttemptId,
    generation: u64,
    parent_artifact_id: ContentId,
    policy: ForgeProposalPolicy,
    opportunities: Vec<ForgeFamilyOpportunity>,
    total_eligible_sites: u64,
    decision: ForgeProposalDecision,
}

impl ForgeProposalExposure {
    pub fn new(
        run_id: ContentId,
        attempt_id: ForgeAttemptId,
        parent_artifact_id: ContentId,
        policy: ForgeProposalPolicy,
        opportunities: Vec<ForgeFamilyOpportunity>,
        decision: ForgeProposalDecision,
    ) -> Result<Self, ForgeProposalExposureError> {
        policy.validate()?;
        attempt_id
            .validate()
            .map_err(|_| ForgeProposalExposureError::ExposureIdentityMismatch)?;
        validate_opportunities(&policy, &opportunities)?;
        let total_eligible_sites = total_sites(&opportunities)?;
        validate_decision(&policy, &opportunities, total_eligible_sites, &decision)?;
        let generation = attempt_id.generation();
        let id = derive_exposure_id(
            &run_id,
            &attempt_id,
            generation,
            &parent_artifact_id,
            &policy,
            &opportunities,
            total_eligible_sites,
            &decision,
        );
        Ok(Self {
            id,
            run_id,
            attempt_id,
            generation,
            parent_artifact_id,
            policy,
            opportunities,
            total_eligible_sites,
            decision,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn run_id(&self) -> &ContentId { &self.run_id }
    pub fn attempt_id(&self) -> &ForgeAttemptId { &self.attempt_id }
    pub fn generation(&self) -> u64 { self.generation }
    pub fn parent_artifact_id(&self) -> &ContentId { &self.parent_artifact_id }
    pub fn policy(&self) -> &ForgeProposalPolicy { &self.policy }
    pub fn opportunities(&self) -> &[ForgeFamilyOpportunity] { &self.opportunities }
    pub fn total_eligible_sites(&self) -> u64 { self.total_eligible_sites }
    pub fn decision(&self) -> &ForgeProposalDecision { &self.decision }

    /// Exact family proposal mass `(eligible family sites, all eligible sites)`.
    pub fn family_exposure_weight(
        &self,
        family: &ForgeTransformationFamilyId,
    ) -> Option<(u64, u64)> {
        if self.total_eligible_sites == 0 {
            return None;
        }
        self.opportunities
            .iter()
            .find(|opportunity| opportunity.family_id() == family)
            .map(|opportunity| (opportunity.eligible_sites(), self.total_eligible_sites))
    }

    /// Exact selected pair propensity `(1, total eligible pairs)` under the recorded rule.
    pub fn selected_pair_weight(&self) -> Option<(u64, u64)> {
        matches!(self.decision, ForgeProposalDecision::Selected(_))
            .then_some((1, self.total_eligible_sites))
    }

    pub fn validate(&self) -> Result<(), ForgeProposalExposureError> {
        self.policy.validate()?;
        self.attempt_id
            .validate()
            .map_err(|_| ForgeProposalExposureError::ExposureIdentityMismatch)?;
        if self.attempt_id.generation() != self.generation {
            return Err(ForgeProposalExposureError::GenerationMismatch);
        }
        validate_opportunities(&self.policy, &self.opportunities)?;
        let total = total_sites(&self.opportunities)?;
        if total != self.total_eligible_sites {
            return Err(ForgeProposalExposureError::OpportunityPolicyMismatch);
        }
        validate_decision(
            &self.policy,
            &self.opportunities,
            self.total_eligible_sites,
            &self.decision,
        )?;
        let expected = derive_exposure_id(
            &self.run_id,
            &self.attempt_id,
            self.generation,
            &self.parent_artifact_id,
            &self.policy,
            &self.opportunities,
            self.total_eligible_sites,
            &self.decision,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalExposureError::ExposureIdentityMismatch)
        }
    }
}

fn validate_opportunities(
    policy: &ForgeProposalPolicy,
    opportunities: &[ForgeFamilyOpportunity],
) -> Result<(), ForgeProposalExposureError> {
    if opportunities.len() != policy.families().len() {
        return Err(ForgeProposalExposureError::OpportunityPolicyMismatch);
    }
    for (opportunity, family) in opportunities.iter().zip(policy.families()) {
        opportunity.family_id().validate()?;
        if opportunity.family_id() != family {
            return Err(ForgeProposalExposureError::OpportunityPolicyMismatch);
        }
    }
    Ok(())
}

fn total_sites(opportunities: &[ForgeFamilyOpportunity]) -> Result<u64, ForgeProposalExposureError> {
    opportunities.iter().try_fold(0u64, |sum, opportunity| {
        sum.checked_add(opportunity.eligible_sites())
            .ok_or(ForgeProposalExposureError::OpportunityCountOverflow)
    })
}

fn validate_decision(
    policy: &ForgeProposalPolicy,
    opportunities: &[ForgeFamilyOpportunity],
    total: u64,
    decision: &ForgeProposalDecision,
) -> Result<(), ForgeProposalExposureError> {
    match (total, decision) {
        (0, ForgeProposalDecision::NoEligibleSites) => Ok(()),
        (0, ForgeProposalDecision::Selected(_)) | (_, ForgeProposalDecision::NoEligibleSites) => {
            Err(ForgeProposalExposureError::InvalidDecision)
        }
        (_, ForgeProposalDecision::Selected(selection)) => {
            selection.family_id().validate()?;
            if selection.family_id().generator_id() != policy.generator_id()
                || selection.global_pair_index() >= total
            {
                return Err(ForgeProposalExposureError::InvalidDecision);
            }
            let mut offset = 0u64;
            for opportunity in opportunities {
                if opportunity.family_id() == selection.family_id() {
                    if selection.site_index() >= opportunity.eligible_sites() {
                        return Err(ForgeProposalExposureError::InvalidDecision);
                    }
                    let expected_global = offset
                        .checked_add(selection.site_index())
                        .ok_or(ForgeProposalExposureError::OpportunityCountOverflow)?;
                    return if expected_global == selection.global_pair_index() {
                        Ok(())
                    } else {
                        Err(ForgeProposalExposureError::InvalidDecision)
                    };
                }
                offset = offset
                    .checked_add(opportunity.eligible_sites())
                    .ok_or(ForgeProposalExposureError::OpportunityCountOverflow)?;
            }
            Err(ForgeProposalExposureError::InvalidDecision)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_exposure_id(
    run_id: &ContentId,
    attempt_id: &ForgeAttemptId,
    generation: u64,
    parent_artifact_id: &ContentId,
    policy: &ForgeProposalPolicy,
    opportunities: &[ForgeFamilyOpportunity],
    total: u64,
    decision: &ForgeProposalDecision,
) -> ContentId {
    let generation = generation.to_be_bytes();
    let count = (opportunities.len() as u64).to_be_bytes();
    let total = total.to_be_bytes();
    let mut parts = vec![
        run_id.as_str().as_bytes().to_vec(),
        attempt_id.as_content_id().as_str().as_bytes().to_vec(),
        generation.to_vec(),
        parent_artifact_id.as_str().as_bytes().to_vec(),
        policy.id().as_str().as_bytes().to_vec(),
        count.to_vec(),
    ];
    for opportunity in opportunities {
        parts.push(opportunity.family_id().as_content_id().as_str().as_bytes().to_vec());
        parts.push(opportunity.eligible_sites().to_be_bytes().to_vec());
    }
    parts.push(total.to_vec());
    match decision {
        ForgeProposalDecision::NoEligibleSites => parts.push(b"no-eligible-sites".to_vec()),
        ForgeProposalDecision::Selected(selection) => {
            parts.push(b"selected".to_vec());
            parts.push(
                selection
                    .family_id()
                    .as_content_id()
                    .as_str()
                    .as_bytes()
                    .to_vec(),
            );
            parts.push(selection.site_index().to_be_bytes().to_vec());
            parts.push(selection.global_pair_index().to_be_bytes().to_vec());
        }
    }
    ContentId::derive(
        "symthaea.forge-proposal-exposure.v1",
        parts.iter().map(Vec::as_slice),
    )
}

/// Canonical exposure archive for one semantic discovery run and one exact proposal policy.
#[derive(Debug, Clone, Serialize)]
pub struct ForgeProposalExposureArchive {
    id: ContentId,
    run_id: ContentId,
    policy_id: ContentId,
    exposures: Vec<ForgeProposalExposure>,
}

impl ForgeProposalExposureArchive {
    pub fn new(
        run: &DiscoveryRun,
        policy: &ForgeProposalPolicy,
        mut exposures: Vec<ForgeProposalExposure>,
    ) -> Result<Self, ForgeProposalExposureError> {
        run.validate().map_err(ForgeLearningError::from)?;
        policy.validate()?;
        if policy.generator_id() != &run.generator_id {
            return Err(ForgeProposalExposureError::PolicyGeneratorMismatch);
        }
        exposures.sort_by_key(|exposure| exposure.attempt_id().ordinal());
        let mut previous = None;
        for exposure in &exposures {
            exposure.validate()?;
            if exposure.run_id() != &run.id || exposure.policy().id() != policy.id() {
                return Err(ForgeProposalExposureError::ArchiveScopeMismatch);
            }
            if exposure.attempt_id().seed() != run.seed
                || exposure.attempt_id().generation() >= run.budget.max_generations
                || previous.is_some_and(|ordinal| exposure.attempt_id().ordinal() <= ordinal)
            {
                return Err(ForgeProposalExposureError::NonCanonicalAttemptOrder);
            }
            previous = Some(exposure.attempt_id().ordinal());
        }
        let id = derive_archive_id(&run.id, policy.id(), &exposures);
        Ok(Self {
            id,
            run_id: run.id.clone(),
            policy_id: policy.id().clone(),
            exposures,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn run_id(&self) -> &ContentId { &self.run_id }
    pub fn policy_id(&self) -> &ContentId { &self.policy_id }
    pub fn exposures(&self) -> &[ForgeProposalExposure] { &self.exposures }

    pub fn get(&self, attempt: &ForgeAttemptId) -> Option<&ForgeProposalExposure> {
        self.exposures
            .binary_search_by_key(&attempt.ordinal(), |exposure| exposure.attempt_id().ordinal())
            .ok()
            .map(|index| &self.exposures[index])
            .filter(|exposure| exposure.attempt_id() == attempt)
    }

    pub fn validate_for(
        &self,
        run: &DiscoveryRun,
        policy: &ForgeProposalPolicy,
    ) -> Result<(), ForgeProposalExposureError> {
        let rebuilt = Self::new(run, policy, self.exposures.clone())?;
        if rebuilt.id == self.id
            && rebuilt.run_id == self.run_id
            && rebuilt.policy_id == self.policy_id
        {
            Ok(())
        } else {
            Err(ForgeProposalExposureError::ArchiveIdentityMismatch)
        }
    }
}

fn derive_archive_id(
    run_id: &ContentId,
    policy_id: &ContentId,
    exposures: &[ForgeProposalExposure],
) -> ContentId {
    let count = (exposures.len() as u64).to_be_bytes();
    let mut parts = vec![
        run_id.as_str().as_bytes().to_vec(),
        policy_id.as_str().as_bytes().to_vec(),
        count.to_vec(),
    ];
    parts.extend(
        exposures
            .iter()
            .map(|exposure| exposure.id().as_str().as_bytes().to_vec()),
    );
    ContentId::derive(
        "symthaea.forge-proposal-exposure-archive.v1",
        parts.iter().map(Vec::as_slice),
    )
}

/// Receipt proving one run has complete attributable proposal exposure for later policy study.
/// It performs no inverse-propensity estimator and grants no authority to alter search.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ForgeProposalLearningQualification {
    id: ContentId,
    run_id: ContentId,
    policy_id: ContentId,
    exposure_archive_id: ContentId,
    concrete_batch_id: ContentId,
    family_trial_set_id: ContentId,
    proposal_stage_attempts: u64,
}

impl ForgeProposalLearningQualification {
    pub fn qualify(
        run: &DiscoveryRun,
        baseline: &ImplementationRecord,
        trace: &[ForgeTraceEvent],
        observations: &ObservationStore,
        policy: &ForgeProposalPolicy,
        archive: &ForgeProposalExposureArchive,
    ) -> Result<Self, ForgeProposalExposureError> {
        archive.validate_for(run, policy)?;
        let concrete = ForgeTrialBatch::from_trace(run, baseline, trace, observations)?;
        let families = ForgeFamilyTrialSet::from_trace(run, baseline, trace, observations)?;
        if families.source_batch_id() != concrete.id() {
            return Err(ForgeProposalExposureError::TrialProjectionMismatch);
        }

        let family_by_attempt = families
            .trials()
            .iter()
            .map(|trial| {
                (
                    trial.attempt_id().as_content_id().as_str().to_string(),
                    trial,
                )
            })
            .collect::<BTreeMap<_, _>>();
        let concrete_by_attempt = concrete
            .trials()
            .trials()
            .iter()
            .map(|trial| {
                (
                    trial.attempt_id().as_content_id().as_str().to_string(),
                    trial,
                )
            })
            .collect::<BTreeMap<_, _>>();

        let mut expected_attempts = BTreeSet::new();
        for event in trace {
            match event.kind {
                DiscoveryEventKind::CandidateGenerated => {
                    let attempt = event
                        .attempt_id
                        .as_ref()
                        .ok_or(ForgeProposalExposureError::MissingExposure)?;
                    let exposure = archive
                        .get(attempt)
                        .ok_or(ForgeProposalExposureError::MissingExposure)?;
                    let family_trial = family_by_attempt
                        .get(attempt.as_content_id().as_str())
                        .ok_or(ForgeProposalExposureError::SelectedFamilyMismatch)?;
                    let concrete_trial = concrete_by_attempt
                        .get(attempt.as_content_id().as_str())
                        .ok_or(ForgeProposalExposureError::TrialProjectionMismatch)?;
                    let ForgeProposalDecision::Selected(selection) = exposure.decision() else {
                        return Err(ForgeProposalExposureError::SelectedFamilyMismatch);
                    };
                    if selection.family_id() != family_trial.family_id() {
                        return Err(ForgeProposalExposureError::SelectedFamilyMismatch);
                    }
                    if exposure.parent_artifact_id() != concrete_trial.parent_artifact_id() {
                        return Err(ForgeProposalExposureError::ParentArtifactMismatch);
                    }
                    expected_attempts.insert(attempt.as_content_id().as_str().to_string());
                }
                DiscoveryEventKind::GeneratorNoOp => {
                    let attempt = event
                        .attempt_id
                        .as_ref()
                        .ok_or(ForgeProposalExposureError::InvalidNoOp)?;
                    let object = observations
                        .get(&event.observation_id)
                        .ok_or(ForgeProposalExposureError::InvalidNoOp)?;
                    if object.schema() != NO_CANDIDATE_SCHEMA {
                        return Err(ForgeProposalExposureError::InvalidNoOp);
                    }
                    let payload: Value = serde_json::from_slice(object.payload())
                        .map_err(|_| ForgeProposalExposureError::InvalidNoOp)?;
                    let reason = payload
                        .get("reason")
                        .and_then(Value::as_str)
                        .ok_or(ForgeProposalExposureError::InvalidNoOp)?;
                    if reason == "current-best-not-syn-parseable" {
                        return Err(ForgeProposalExposureError::PreProposalAttempt);
                    }
                    let exposure = archive
                        .get(attempt)
                        .ok_or(ForgeProposalExposureError::MissingExposure)?;
                    let observed_parent = payload
                        .get("parent_artifact_id")
                        .and_then(Value::as_str)
                        .ok_or(ForgeProposalExposureError::InvalidNoOp)?;
                    if observed_parent != exposure.parent_artifact_id().as_str() {
                        return Err(ForgeProposalExposureError::ParentArtifactMismatch);
                    }
                    match (reason, exposure.decision()) {
                        ("no-eligible-ast-mutation", ForgeProposalDecision::NoEligibleSites) => {}
                        ("mutation-rendered-identical-source", ForgeProposalDecision::Selected(_)) => {}
                        _ => return Err(ForgeProposalExposureError::InvalidNoOp),
                    }
                    expected_attempts.insert(attempt.as_content_id().as_str().to_string());
                }
                _ => {}
            }
        }

        if archive.exposures().len() != expected_attempts.len()
            || archive.exposures().iter().any(|exposure| {
                !expected_attempts.contains(exposure.attempt_id().as_content_id().as_str())
            })
        {
            return Err(ForgeProposalExposureError::ExtraExposure);
        }

        let proposal_stage_attempts = u64::try_from(expected_attempts.len())
            .map_err(|_| ForgeProposalExposureError::OpportunityCountOverflow)?;
        let id = derive_qualification_id(
            &run.id,
            policy.id(),
            archive.id(),
            concrete.id(),
            families.id(),
            proposal_stage_attempts,
        );
        Ok(Self {
            id,
            run_id: run.id.clone(),
            policy_id: policy.id().clone(),
            exposure_archive_id: archive.id().clone(),
            concrete_batch_id: concrete.id().clone(),
            family_trial_set_id: families.id().clone(),
            proposal_stage_attempts,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn run_id(&self) -> &ContentId { &self.run_id }
    pub fn policy_id(&self) -> &ContentId { &self.policy_id }
    pub fn exposure_archive_id(&self) -> &ContentId { &self.exposure_archive_id }
    pub fn concrete_batch_id(&self) -> &ContentId { &self.concrete_batch_id }
    pub fn family_trial_set_id(&self) -> &ContentId { &self.family_trial_set_id }
    pub fn proposal_stage_attempts(&self) -> u64 { self.proposal_stage_attempts }

    pub fn validate_identity(&self) -> Result<(), ForgeProposalExposureError> {
        let expected = derive_qualification_id(
            &self.run_id,
            &self.policy_id,
            &self.exposure_archive_id,
            &self.concrete_batch_id,
            &self.family_trial_set_id,
            self.proposal_stage_attempts,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(ForgeProposalExposureError::QualificationIdentityMismatch)
        }
    }
}

fn derive_qualification_id(
    run_id: &ContentId,
    policy_id: &ContentId,
    archive_id: &ContentId,
    concrete_batch_id: &ContentId,
    family_trial_set_id: &ContentId,
    proposal_stage_attempts: u64,
) -> ContentId {
    let attempts = proposal_stage_attempts.to_be_bytes();
    ContentId::derive(
        "symthaea.forge-proposal-learning-qualification.v1",
        [
            run_id.as_str().as_bytes(),
            policy_id.as_str().as_bytes(),
            archive_id.as_str().as_bytes(),
            concrete_batch_id.as_str().as_bytes(),
            family_trial_set_id.as_str().as_bytes(),
            attempts.as_slice(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::full_source_artifact_id;
    use symthaea_algorithms::discovery::{DiscoveryPolicy, SearchBudget};
    use symthaea_algorithms::{
        DeterminismRequirement, DiscoveryRisk, ProblemSpec, SemanticGuarantee,
    };

    fn run(generator: ContentId) -> DiscoveryRun {
        let problem = ProblemSpec::new(
            "proposal-test",
            "Exact test contract.",
            SemanticGuarantee::Exact,
            DeterminismRequirement::Required,
            vec!["exact".into()],
            DiscoveryRisk::Ordinary,
        )
        .unwrap();
        DiscoveryRun::new(
            &problem,
            DiscoveryPolicy::default(),
            generator,
            "abc123",
            SearchBudget::new(10, 2, 10).unwrap(),
            7,
        )
        .unwrap()
    }

    #[test]
    fn selected_pair_maps_to_exact_family_site_and_weights() {
        let generator = ContentId::derive("generator", [b"forge-v1".as_slice()]);
        let run = run(generator.clone());
        let policy = ForgeProposalPolicy::uniform_eligible_site(generator, ["A", "B", "C"]).unwrap();
        let opportunities = vec![
            ForgeFamilyOpportunity::new(policy.families()[0].clone(), 2),
            ForgeFamilyOpportunity::new(policy.families()[1].clone(), 5),
            ForgeFamilyOpportunity::new(policy.families()[2].clone(), 0),
        ];
        let attempt = ForgeAttemptId::derive(
            &full_source_artifact_id("baseline"),
            run.seed,
            0,
            0,
        );
        let exposure = ForgeProposalExposure::new(
            run.id,
            attempt,
            full_source_artifact_id("parent"),
            policy.clone(),
            opportunities,
            ForgeProposalDecision::Selected(ForgeProposalSelection::new(
                policy.families()[1].clone(),
                3,
                5,
            )),
        )
        .unwrap();
        assert_eq!(exposure.total_eligible_sites(), 7);
        assert_eq!(
            exposure.family_exposure_weight(&policy.families()[1]),
            Some((5, 7))
        );
        assert_eq!(exposure.selected_pair_weight(), Some((1, 7)));
        assert!(exposure.validate().is_ok());
    }

    #[test]
    fn zero_opportunities_requires_no_eligible_sites_decision() {
        let generator = ContentId::derive("generator", [b"forge-v1".as_slice()]);
        let run = run(generator.clone());
        let policy = ForgeProposalPolicy::uniform_eligible_site(generator, ["A", "B"]).unwrap();
        let opportunities = policy
            .families()
            .iter()
            .cloned()
            .map(|family| ForgeFamilyOpportunity::new(family, 0))
            .collect();
        let attempt = ForgeAttemptId::derive(
            &full_source_artifact_id("baseline"),
            run.seed,
            0,
            0,
        );
        let exposure = ForgeProposalExposure::new(
            run.id,
            attempt,
            full_source_artifact_id("parent"),
            policy,
            opportunities,
            ForgeProposalDecision::NoEligibleSites,
        )
        .unwrap();
        assert_eq!(exposure.total_eligible_sites(), 0);
        assert_eq!(exposure.selected_pair_weight(), None);
        assert!(exposure.validate().is_ok());
    }

    #[test]
    fn wrong_global_pair_index_is_rejected() {
        let generator = ContentId::derive("generator", [b"forge-v1".as_slice()]);
        let run = run(generator.clone());
        let policy = ForgeProposalPolicy::uniform_eligible_site(generator, ["A", "B"]).unwrap();
        let opportunities = vec![
            ForgeFamilyOpportunity::new(policy.families()[0].clone(), 2),
            ForgeFamilyOpportunity::new(policy.families()[1].clone(), 3),
        ];
        let attempt = ForgeAttemptId::derive(
            &full_source_artifact_id("baseline"),
            run.seed,
            0,
            0,
        );
        assert!(matches!(
            ForgeProposalExposure::new(
                run.id,
                attempt,
                full_source_artifact_id("parent"),
                policy.clone(),
                opportunities,
                ForgeProposalDecision::Selected(ForgeProposalSelection::new(
                    policy.families()[1].clone(),
                    1,
                    1,
                )),
            ),
            Err(ForgeProposalExposureError::InvalidDecision)
        ));
    }
}
