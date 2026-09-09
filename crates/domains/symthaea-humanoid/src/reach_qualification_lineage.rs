// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact evidence-lineage binding for Reach qualification campaigns.
//!
//! `reach_qualification_campaign` owns cell coverage and statistics. This module
//! adds the promotion-grade lineage boundary: every trial is bound at creation
//! time to the complete command/outcome policy contents and to an upstream
//! perturbation-configuration fingerprint. Human-readable IDs alone are not
//! sufficient for promotion.

use std::collections::{BTreeMap, BTreeSet};

use crate::qualification::HumanoidQualificationSubject;
use crate::reach_execution::HumanoidPermittedReachExecutionResult;
use crate::reach_execution_evidence::HumanoidReachCommandEvidencePolicy;
use crate::reach_outcome_evidence::{
    HumanoidReachOutcomeEvidencePolicy, HumanoidReachStepEvidenceAssessment,
};
use crate::reach_policy_identity::{
    humanoid_reach_command_policy_fingerprint, humanoid_reach_outcome_policy_fingerprint,
};
use crate::reach_qualification_campaign::{
    HumanoidReachCampaignFailureKind, HumanoidReachQualificationCampaignAssessment,
    HumanoidReachQualificationCampaignPolicy, HumanoidReachQualificationTrial,
    HumanoidReachQualificationTrialBindFailure, HumanoidReachScenarioCell,
    assess_humanoid_reach_qualification_campaign, bind_humanoid_reach_qualification_trial,
};

pub const HUMANOID_REACH_QUALIFICATION_LINEAGE_SCHEMA_VERSION: u32 = 1;

/// Identity supplied by the perturbation/randomization producer.
///
/// The humanoid crate does not currently own a canonical domain-randomization
/// schema, so it does not invent one here. The upstream producer must canonicalize
/// its complete perturbation configuration and provide a stable non-zero
/// fingerprint. When a first-class perturbation profile lands, it can own this
/// fingerprint without changing the campaign contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HumanoidReachPerturbationProfileBinding {
    pub profile_id: String,
    pub configuration_fingerprint: u64,
}

impl HumanoidReachPerturbationProfileBinding {
    pub fn validate(&self) -> bool {
        valid_id(&self.profile_id) && self.configuration_fingerprint != 0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachLineageBoundTrial {
    pub schema_version: u32,
    pub trial: HumanoidReachQualificationTrial,
    pub command_policy_fingerprint: u64,
    pub outcome_policy_fingerprint: u64,
    pub perturbation_configuration_fingerprint: u64,
    pub lineage_fingerprint: u64,
}

impl HumanoidReachLineageBoundTrial {
    pub fn validate(&self) -> bool {
        self.schema_version == HUMANOID_REACH_QUALIFICATION_LINEAGE_SCHEMA_VERSION
            && self.trial.validate()
            && self.command_policy_fingerprint != 0
            && self.outcome_policy_fingerprint != 0
            && self.perturbation_configuration_fingerprint != 0
            && self.lineage_fingerprint != 0
            && self.lineage_fingerprint == fingerprint_lineage_trial(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachLineageTrialBindFailure {
    InvalidCommandPolicy,
    InvalidOutcomePolicy,
    InvalidPerturbationBinding,
    PerturbationProfileMismatch,
    StepCommandPolicyMismatch,
    StepOutcomePolicyMismatch,
    PolicyFingerprintInvalid,
    Base(HumanoidReachQualificationTrialBindFailure),
}

/// Preferred trial binder for qualification evidence that may later be promoted.
///
/// It binds the full policy contents before delegating to the cell/authority
/// checks in `bind_humanoid_reach_qualification_trial`.
#[allow(clippy::too_many_arguments)]
pub fn bind_lineage_humanoid_reach_qualification_trial(
    subject: &HumanoidQualificationSubject,
    scenario: &HumanoidReachScenarioCell,
    perturbation: &HumanoidReachPerturbationProfileBinding,
    trial_id: impl Into<String>,
    trial_seed: u64,
    result: &HumanoidPermittedReachExecutionResult,
    step: &HumanoidReachStepEvidenceAssessment,
    command_policy: &HumanoidReachCommandEvidencePolicy,
    outcome_policy: &HumanoidReachOutcomeEvidencePolicy,
) -> Result<HumanoidReachLineageBoundTrial, HumanoidReachLineageTrialBindFailure> {
    if !command_policy.validate_for(subject) {
        return Err(HumanoidReachLineageTrialBindFailure::InvalidCommandPolicy);
    }
    if !outcome_policy.validate_for(subject) {
        return Err(HumanoidReachLineageTrialBindFailure::InvalidOutcomePolicy);
    }
    if !perturbation.validate() {
        return Err(HumanoidReachLineageTrialBindFailure::InvalidPerturbationBinding);
    }
    if scenario.perturbation_profile_id != perturbation.profile_id {
        return Err(HumanoidReachLineageTrialBindFailure::PerturbationProfileMismatch);
    }
    if step.command.policy_id != command_policy.policy_id {
        return Err(HumanoidReachLineageTrialBindFailure::StepCommandPolicyMismatch);
    }
    if step.outcome.policy_id != outcome_policy.policy_id {
        return Err(HumanoidReachLineageTrialBindFailure::StepOutcomePolicyMismatch);
    }

    let command_policy_fingerprint = humanoid_reach_command_policy_fingerprint(command_policy);
    let outcome_policy_fingerprint = humanoid_reach_outcome_policy_fingerprint(outcome_policy);
    if command_policy_fingerprint == 0 || outcome_policy_fingerprint == 0 {
        return Err(HumanoidReachLineageTrialBindFailure::PolicyFingerprintInvalid);
    }

    let trial = bind_humanoid_reach_qualification_trial(
        subject,
        scenario,
        perturbation.profile_id.clone(),
        trial_id,
        trial_seed,
        result,
        step,
    )
    .map_err(HumanoidReachLineageTrialBindFailure::Base)?;

    let mut bound = HumanoidReachLineageBoundTrial {
        schema_version: HUMANOID_REACH_QUALIFICATION_LINEAGE_SCHEMA_VERSION,
        trial,
        command_policy_fingerprint,
        outcome_policy_fingerprint,
        perturbation_configuration_fingerprint: perturbation.configuration_fingerprint,
        lineage_fingerprint: 0,
    };
    bound.lineage_fingerprint = fingerprint_lineage_trial(&bound);
    if !bound.validate() {
        return Err(HumanoidReachLineageTrialBindFailure::PolicyFingerprintInvalid);
    }
    Ok(bound)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HumanoidReachScenarioLineageRequirement {
    pub scenario_id: String,
    pub perturbation_profile_id: String,
    pub perturbation_configuration_fingerprint: u64,
}

impl HumanoidReachScenarioLineageRequirement {
    pub fn validate(&self) -> bool {
        valid_id(&self.scenario_id)
            && valid_id(&self.perturbation_profile_id)
            && self.perturbation_configuration_fingerprint != 0
    }
}

/// Promotion-grade wrapper around the statistical campaign policy.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachLineageCampaignPolicy {
    pub schema_version: u32,
    pub campaign: HumanoidReachQualificationCampaignPolicy,
    pub command_policy_fingerprint: u64,
    pub outcome_policy_fingerprint: u64,
    pub scenario_lineage: Vec<HumanoidReachScenarioLineageRequirement>,
}

impl HumanoidReachLineageCampaignPolicy {
    pub fn from_exact_policies(
        subject: &HumanoidQualificationSubject,
        campaign: HumanoidReachQualificationCampaignPolicy,
        command_policy: &HumanoidReachCommandEvidencePolicy,
        outcome_policy: &HumanoidReachOutcomeEvidencePolicy,
        scenario_lineage: Vec<HumanoidReachScenarioLineageRequirement>,
    ) -> Option<Self> {
        if !campaign.validate_for(subject)
            || !command_policy.validate_for(subject)
            || !outcome_policy.validate_for(subject)
            || campaign.required_command_policy_id != command_policy.policy_id
            || campaign.required_outcome_policy_id != outcome_policy.policy_id
        {
            return None;
        }
        let policy = Self {
            schema_version: HUMANOID_REACH_QUALIFICATION_LINEAGE_SCHEMA_VERSION,
            campaign,
            command_policy_fingerprint: humanoid_reach_command_policy_fingerprint(command_policy),
            outcome_policy_fingerprint: humanoid_reach_outcome_policy_fingerprint(outcome_policy),
            scenario_lineage,
        };
        policy.validate_for(subject).then_some(policy)
    }

    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        if self.schema_version != HUMANOID_REACH_QUALIFICATION_LINEAGE_SCHEMA_VERSION
            || !self.campaign.validate_for(subject)
            || self.command_policy_fingerprint == 0
            || self.outcome_policy_fingerprint == 0
            || self.scenario_lineage.is_empty()
            || self.scenario_lineage.iter().any(|item| !item.validate())
        {
            return false;
        }

        let required = self
            .campaign
            .required_scenarios
            .iter()
            .map(|requirement| {
                (
                    requirement.cell.scenario_id.clone(),
                    requirement.cell.perturbation_profile_id.clone(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        if required.len() != self.scenario_lineage.len() {
            return false;
        }
        let mut seen = BTreeSet::new();
        self.scenario_lineage.iter().all(|lineage| {
            seen.insert(lineage.scenario_id.clone())
                && required
                    .get(&lineage.scenario_id)
                    .is_some_and(|profile| profile == &lineage.perturbation_profile_id)
        }) && seen.len() == required.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachLineageCampaignFailureKind {
    InvalidLineagePolicy,
    InvalidLineageTrial,
    StatisticalCampaignFailed,
    CommandPolicyFingerprintMismatch,
    OutcomePolicyFingerprintMismatch,
    MissingScenarioLineage,
    PerturbationFingerprintMismatch,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachLineageCampaignAssessment {
    pub base: HumanoidReachQualificationCampaignAssessment,
    pub lineage_policy_fingerprint: u64,
    pub corpus_lineage_fingerprint: u64,
    /// Promotion eligibility requires both statistical coverage and exact lineage.
    pub promotion_eligible: bool,
    pub failures: Vec<HumanoidReachLineageCampaignFailureKind>,
}

pub fn assess_lineage_humanoid_reach_qualification_campaign(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachLineageCampaignPolicy,
    trials: &[HumanoidReachLineageBoundTrial],
) -> HumanoidReachLineageCampaignAssessment {
    let mut failures = Vec::new();
    if !policy.validate_for(subject) {
        failures.push(HumanoidReachLineageCampaignFailureKind::InvalidLineagePolicy);
    }
    if trials.iter().any(|trial| !trial.validate()) {
        failures.push(HumanoidReachLineageCampaignFailureKind::InvalidLineageTrial);
    }

    let base_trials = trials.iter().map(|trial| trial.trial.clone()).collect::<Vec<_>>();
    let base = assess_humanoid_reach_qualification_campaign(subject, &policy.campaign, &base_trials);
    if !base.campaign_accepted {
        failures.push(HumanoidReachLineageCampaignFailureKind::StatisticalCampaignFailed);
    }

    let scenario_lineage = policy
        .scenario_lineage
        .iter()
        .map(|item| (item.scenario_id.as_str(), item))
        .collect::<BTreeMap<_, _>>();
    for trial in trials {
        if trial.command_policy_fingerprint != policy.command_policy_fingerprint {
            failures.push(
                HumanoidReachLineageCampaignFailureKind::CommandPolicyFingerprintMismatch,
            );
        }
        if trial.outcome_policy_fingerprint != policy.outcome_policy_fingerprint {
            failures.push(
                HumanoidReachLineageCampaignFailureKind::OutcomePolicyFingerprintMismatch,
            );
        }
        match scenario_lineage.get(trial.trial.scenario_id.as_str()) {
            None => failures.push(HumanoidReachLineageCampaignFailureKind::MissingScenarioLineage),
            Some(required)
                if required.perturbation_configuration_fingerprint
                    != trial.perturbation_configuration_fingerprint =>
            {
                failures.push(
                    HumanoidReachLineageCampaignFailureKind::PerturbationFingerprintMismatch,
                );
            }
            Some(_) => {}
        }
    }

    failures.sort_by_key(|failure| *failure as u8);
    failures.dedup();
    let lineage_policy_fingerprint = fingerprint_lineage_policy(policy);
    let corpus_lineage_fingerprint = fingerprint_lineage_corpus(trials);
    HumanoidReachLineageCampaignAssessment {
        promotion_eligible: failures.is_empty()
            && lineage_policy_fingerprint != 0
            && corpus_lineage_fingerprint != 0,
        base,
        lineage_policy_fingerprint,
        corpus_lineage_fingerprint,
        failures,
    }
}

fn fingerprint_lineage_trial(trial: &HumanoidReachLineageBoundTrial) -> u64 {
    if trial.schema_version != HUMANOID_REACH_QUALIFICATION_LINEAGE_SCHEMA_VERSION
        || trial.trial.trial_fingerprint == 0
        || trial.command_policy_fingerprint == 0
        || trial.outcome_policy_fingerprint == 0
        || trial.perturbation_configuration_fingerprint == 0
    {
        return 0;
    }
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    feed_u64(&mut hash, trial.schema_version as u64);
    feed_u64(&mut hash, trial.trial.trial_fingerprint);
    feed_u64(&mut hash, trial.command_policy_fingerprint);
    feed_u64(&mut hash, trial.outcome_policy_fingerprint);
    feed_u64(&mut hash, trial.perturbation_configuration_fingerprint);
    if hash == 0 { 1 } else { hash }
}

fn fingerprint_lineage_policy(policy: &HumanoidReachLineageCampaignPolicy) -> u64 {
    if policy.schema_version != HUMANOID_REACH_QUALIFICATION_LINEAGE_SCHEMA_VERSION
        || policy.command_policy_fingerprint == 0
        || policy.outcome_policy_fingerprint == 0
        || policy.scenario_lineage.is_empty()
    {
        return 0;
    }
    let mut entries = policy.scenario_lineage.iter().collect::<Vec<_>>();
    entries.sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    feed_u64(&mut hash, policy.schema_version as u64);
    feed_bytes(&mut hash, policy.campaign.campaign_id.as_bytes());
    feed_u64(&mut hash, policy.campaign.subject_fingerprint);
    feed_u64(&mut hash, policy.command_policy_fingerprint);
    feed_u64(&mut hash, policy.outcome_policy_fingerprint);
    for entry in entries {
        feed_bytes(&mut hash, entry.scenario_id.as_bytes());
        feed_bytes(&mut hash, entry.perturbation_profile_id.as_bytes());
        feed_u64(&mut hash, entry.perturbation_configuration_fingerprint);
    }
    if hash == 0 { 1 } else { hash }
}

fn fingerprint_lineage_corpus(trials: &[HumanoidReachLineageBoundTrial]) -> u64 {
    if trials.is_empty() || trials.iter().any(|trial| !trial.validate()) {
        return 0;
    }
    let mut entries = trials.iter().collect::<Vec<_>>();
    entries.sort_by(|left, right| {
        left.trial
            .scenario_id
            .cmp(&right.trial.scenario_id)
            .then(left.trial.trial_id.cmp(&right.trial.trial_id))
    });
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for entry in entries {
        feed_u64(&mut hash, entry.lineage_fingerprint);
    }
    if hash == 0 { 1 } else { hash }
}

fn valid_id(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && value.len() <= 256
        && value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && !byte.is_ascii_whitespace())
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
    use crate::execution_authority_scope::{
        HumanoidExecutionPurpose, HumanoidQualificationAuthorityBasis,
    };
    use crate::morphology::{HandSide, HumanoidMorphology};
    use crate::reach_qualification_campaign::{
        HumanoidReachScenarioRequirement, HUMANOID_REACH_QUALIFICATION_CAMPAIGN_SCHEMA_VERSION,
    };
    use crate::types::{ActuationMode, HumanoidTask};

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Reach,
            ActuationMode::NormalizedTorque,
            "lineage-test-v1",
        )
    }

    fn command_policy() -> HumanoidReachCommandEvidencePolicy {
        HumanoidReachCommandEvidencePolicy {
            policy_id: "command-policy-v1".into(),
            subject_fingerprint: subject().fingerprint(),
            maximum_full_dynamics_age_s: 0.05,
            minimum_jacobian_confidence: 0.8,
            minimum_goal_authority_scale: 0.5,
            maximum_whole_body_objective_residual: 0.2,
            maximum_joint_utilization: 1.0,
            maximum_inverse_dynamics_violation: 0.01,
            allow_inverse_dynamics_fallback: false,
            maximum_contact_dynamics_residual_nm: 0.05,
            maximum_contact_acceleration_residual: 0.05,
            maximum_contact_friction_utilization: 1.0,
            allow_contact_dynamics_fallback: false,
            require_floating_base_model: true,
            maximum_floating_base_dynamics_residual: 0.05,
            allow_floating_base_fallback: false,
            maximum_final_safety_interventions: 0,
        }
    }

    fn outcome_policy() -> HumanoidReachOutcomeEvidencePolicy {
        HumanoidReachOutcomeEvidencePolicy {
            policy_id: "outcome-policy-v1".into(),
            subject_fingerprint: subject().fingerprint(),
            maximum_observation_age_s: 0.02,
            maximum_elapsed_since_preparation_s: 0.1,
            maximum_post_command_error_m: 0.05,
            minimum_progress_m: 0.01,
            minimum_fractional_progress: 0.2,
            allow_already_within_tolerance: true,
        }
    }

    fn base_campaign() -> HumanoidReachQualificationCampaignPolicy {
        HumanoidReachQualificationCampaignPolicy {
            schema_version: HUMANOID_REACH_QUALIFICATION_CAMPAIGN_SCHEMA_VERSION,
            campaign_id: "lineage-campaign-v1".into(),
            subject_fingerprint: subject().fingerprint(),
            required_execution_purpose: HumanoidExecutionPurpose::SimulationQualification,
            required_authority_scope_id: "sim-trial-v1".into(),
            required_command_policy_id: "command-policy-v1".into(),
            required_outcome_policy_id: "outcome-policy-v1".into(),
            required_scenarios: vec![HumanoidReachScenarioRequirement {
                cell: HumanoidReachScenarioCell {
                    scenario_id: "right-interior".into(),
                    hand: HandSide::Right,
                    minimum_workspace_utilization_sq: 0.0,
                    maximum_workspace_utilization_sq: 0.5,
                    perturbation_profile_id: "nominal-v1".into(),
                },
                minimum_trials: 1,
                maximum_failure_rate: 0.0,
                minimum_distinct_spatial_goals: 1,
                minimum_distinct_authority_receipts: 1,
                require_unique_trial_seeds: true,
            }],
        }
    }

    fn base_trial() -> HumanoidReachQualificationTrial {
        let mut trial = HumanoidReachQualificationTrial {
            schema_version: HUMANOID_REACH_QUALIFICATION_CAMPAIGN_SCHEMA_VERSION,
            subject_fingerprint: subject().fingerprint(),
            scenario_id: "right-interior".into(),
            perturbation_profile_id: "nominal-v1".into(),
            trial_id: "trial-1".into(),
            trial_seed: 1,
            hand: HandSide::Right,
            workspace_utilization_sq: 0.25,
            validation_epoch: 1,
            goal_id: "goal-1".into(),
            spatial_goal_fingerprint: 100,
            command_policy_id: "command-policy-v1".into(),
            outcome_policy_id: "outcome-policy-v1".into(),
            authority_receipt_fingerprint: 200,
            authority_scope_fingerprint: 300,
            authority_scope_id: "sim-trial-v1".into(),
            execution_purpose: HumanoidExecutionPurpose::SimulationQualification,
            qualification_basis: HumanoidQualificationAuthorityBasis::TrialProtocol,
            authority_effective_scale: 1.0,
            step_accepted: true,
            trial_fingerprint: 0,
        };
        // Test fixture cannot call the private base fingerprint helper; build a
        // lineage test around validation failure/identity functions instead.
        trial
    }

    #[test]
    fn reused_policy_id_with_changed_threshold_has_different_lineage_identity() {
        let a = command_policy();
        let mut b = command_policy();
        b.maximum_inverse_dynamics_violation = 0.02;
        assert_ne!(
            humanoid_reach_command_policy_fingerprint(&a),
            humanoid_reach_command_policy_fingerprint(&b)
        );
    }

    #[test]
    fn campaign_policy_requires_one_perturbation_fingerprint_per_cell() {
        let policy = HumanoidReachLineageCampaignPolicy::from_exact_policies(
            &subject(),
            base_campaign(),
            &command_policy(),
            &outcome_policy(),
            vec![],
        );
        assert!(policy.is_none());
    }

    #[test]
    fn matching_exact_policy_and_perturbation_lineage_is_constructible() {
        let policy = HumanoidReachLineageCampaignPolicy::from_exact_policies(
            &subject(),
            base_campaign(),
            &command_policy(),
            &outcome_policy(),
            vec![HumanoidReachScenarioLineageRequirement {
                scenario_id: "right-interior".into(),
                perturbation_profile_id: "nominal-v1".into(),
                perturbation_configuration_fingerprint: 777,
            }],
        );
        assert!(policy.is_some());
    }
}
