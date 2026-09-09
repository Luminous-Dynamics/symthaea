// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Preferred Reach promotion facade with cryptographic diversity gates.
//!
//! Lower Reach campaigns retain compact 64-bit identities for compatibility and
//! deterministic diagnostics. Operational promotion must not rely on those values
//! for evidence-diversity claims. This facade independently counts collision-
//! resistant spatial-goal and authority-receipt commitments before delegating to
//! the authority-committed promotion engine.

use std::collections::BTreeSet;

use crate::execution_authority_scope::HumanoidScopedSkillAuthorityReceipt;
use crate::qualification::HumanoidQualificationSubject;
use crate::reach_authority_committed_evidence::{
    HumanoidReachAuthorityCommittedEpisodeCase, HumanoidReachAuthorityCommittedTrial,
};
use crate::reach_authority_committed_promotion::{
    issue_humanoid_reach_authority_committed_episode_stage,
    issue_humanoid_reach_authority_committed_operational_authority_receipt,
    issue_humanoid_reach_authority_committed_step_stage,
    promote_humanoid_reach_authority_committed_to_operational,
};
use crate::reach_episode_evidence::HumanoidReachEpisodePolicy;
use crate::reach_execution_evidence::HumanoidReachCommandEvidencePolicy;
use crate::reach_outcome_evidence::HumanoidReachOutcomeEvidencePolicy;
use crate::reach_qualification_lineage::HumanoidReachLineageCampaignPolicy;
use crate::skill_authority_receipt::HumanoidAuthoritySourceSnapshot;
use crate::skill_permit::HumanoidSkillExecutionPermit;

pub use crate::reach_authority_committed_promotion::{
    HumanoidReachAuthorityCommittedEpisodeStageArtifact as HumanoidReachStrongEpisodeStageArtifact,
    HumanoidReachAuthorityCommittedOperationalArtifact as HumanoidReachStrongOperationalArtifact,
    HumanoidReachAuthorityCommittedStepStageArtifact as HumanoidReachStrongStepStageArtifact,
};
pub use crate::reach_cryptographic_authority::{
    HumanoidReachCryptographicAuthorityIssueFailure,
    HumanoidReachCryptographicOperatorApproval,
    HumanoidReachCryptographicOperationalPolicy,
    HumanoidReachCryptographicStageRequirement,
};
pub use crate::reach_episode_promotion::{
    HumanoidReachEpisodeCampaignAssessment,
    HumanoidReachEpisodeCampaignFailureKind,
    HumanoidReachEpisodeCampaignPolicy,
    HumanoidReachEpisodeScenarioAssessment,
    HumanoidReachEpisodeScenarioFailureKind,
    HumanoidReachEpisodeScenarioRequirement,
    assess_humanoid_reach_episode_campaign,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachStrongDiversityFailure {
    InvalidEvidence,
    UnknownScenario,
    InsufficientDistinctSpatialGoals,
    InsufficientDistinctAuthorityReceipts,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachStrongStageIssueFailure {
    Diversity(HumanoidReachStrongDiversityFailure),
    LowerStageIssue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidReachStrongPromotionFailure {
    LowerPromotion,
}

/// Issue a step-stage artifact only when every required scenario satisfies the
/// campaign's diversity thresholds using SHA-256 identities.
#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_strong_step_stage(
    subject: &HumanoidQualificationSubject,
    step_policy: &HumanoidReachLineageCampaignPolicy,
    command_policy: &HumanoidReachCommandEvidencePolicy,
    outcome_policy: &HumanoidReachOutcomeEvidencePolicy,
    trials: &[HumanoidReachAuthorityCommittedTrial],
    issued_unix_millis: u64,
) -> Result<HumanoidReachStrongStepStageArtifact, HumanoidReachStrongStageIssueFailure> {
    validate_step_diversity(step_policy, trials)
        .map_err(HumanoidReachStrongStageIssueFailure::Diversity)?;
    issue_humanoid_reach_authority_committed_step_stage(
        subject,
        step_policy,
        command_policy,
        outcome_policy,
        trials,
        issued_unix_millis,
    )
    .map_err(|_| HumanoidReachStrongStageIssueFailure::LowerStageIssue)
}

/// Issue an episode-stage artifact only when authority diversity is satisfied by
/// SHA-256 receipt commitments across the actual committed steps in each scenario.
#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_strong_episode_stage(
    subject: &HumanoidQualificationSubject,
    campaign_policy: &HumanoidReachEpisodeCampaignPolicy,
    episode_policy: &HumanoidReachEpisodePolicy,
    command_policy: &HumanoidReachCommandEvidencePolicy,
    outcome_policy: &HumanoidReachOutcomeEvidencePolicy,
    cases: &[HumanoidReachAuthorityCommittedEpisodeCase],
    issued_unix_millis: u64,
) -> Result<HumanoidReachStrongEpisodeStageArtifact, HumanoidReachStrongStageIssueFailure> {
    validate_episode_authority_diversity(campaign_policy, cases)
        .map_err(HumanoidReachStrongStageIssueFailure::Diversity)?;
    issue_humanoid_reach_authority_committed_episode_stage(
        subject,
        campaign_policy,
        episode_policy,
        command_policy,
        outcome_policy,
        cases,
        issued_unix_millis,
    )
    .map_err(|_| HumanoidReachStrongStageIssueFailure::LowerStageIssue)
}

#[allow(clippy::too_many_arguments)]
pub fn promote_humanoid_reach_strong_to_operational(
    subject: &HumanoidQualificationSubject,
    policy: &HumanoidReachCryptographicOperationalPolicy,
    simulation_step: &HumanoidReachStrongStepStageArtifact,
    simulation_episode: &HumanoidReachStrongEpisodeStageArtifact,
    hil_step: &HumanoidReachStrongStepStageArtifact,
    hil_episode: &HumanoidReachStrongEpisodeStageArtifact,
    physical_step: &HumanoidReachStrongStepStageArtifact,
    physical_episode: &HumanoidReachStrongEpisodeStageArtifact,
    now_unix_millis: u64,
) -> Result<HumanoidReachStrongOperationalArtifact, HumanoidReachStrongPromotionFailure> {
    promote_humanoid_reach_authority_committed_to_operational(
        subject,
        policy,
        simulation_step,
        simulation_episode,
        hil_step,
        hil_episode,
        physical_step,
        physical_episode,
        now_unix_millis,
    )
    .map_err(|_| HumanoidReachStrongPromotionFailure::LowerPromotion)
}

#[allow(clippy::too_many_arguments)]
pub fn issue_humanoid_reach_strong_operational_authority_receipt(
    subject: &HumanoidQualificationSubject,
    permit: &HumanoidSkillExecutionPermit<'_>,
    qualification: &HumanoidReachStrongOperationalArtifact,
    policy: &HumanoidReachCryptographicOperationalPolicy,
    operator_approval: &HumanoidReachCryptographicOperatorApproval,
    physical: HumanoidAuthoritySourceSnapshot,
    epistemic: HumanoidAuthoritySourceSnapshot,
    cognitive: HumanoidAuthoritySourceSnapshot,
    now_s: f64,
    now_unix_millis: u64,
) -> Result<HumanoidScopedSkillAuthorityReceipt, HumanoidReachCryptographicAuthorityIssueFailure> {
    issue_humanoid_reach_authority_committed_operational_authority_receipt(
        subject,
        permit,
        qualification,
        policy,
        operator_approval,
        physical,
        epistemic,
        cognitive,
        now_s,
        now_unix_millis,
    )
}

fn validate_step_diversity(
    policy: &HumanoidReachLineageCampaignPolicy,
    trials: &[HumanoidReachAuthorityCommittedTrial],
) -> Result<(), HumanoidReachStrongDiversityFailure> {
    if trials.is_empty() || trials.iter().any(|trial| !trial.validate()) {
        return Err(HumanoidReachStrongDiversityFailure::InvalidEvidence);
    }

    let known = policy
        .campaign
        .required_scenarios
        .iter()
        .map(|requirement| requirement.cell.scenario_id.as_str())
        .collect::<BTreeSet<_>>();
    if trials
        .iter()
        .any(|trial| !known.contains(trial.inner.trial.scenario_id.as_str()))
    {
        return Err(HumanoidReachStrongDiversityFailure::UnknownScenario);
    }

    for requirement in &policy.campaign.required_scenarios {
        let scenario_trials = trials
            .iter()
            .filter(|trial| trial.inner.trial.scenario_id == requirement.cell.scenario_id)
            .collect::<Vec<_>>();

        let spatial = scenario_trials
            .iter()
            .map(|trial| trial.spatial_goal_digest())
            .filter(|digest| !digest.is_zero())
            .collect::<BTreeSet<_>>()
            .len();
        if spatial < requirement.minimum_distinct_spatial_goals {
            return Err(HumanoidReachStrongDiversityFailure::InsufficientDistinctSpatialGoals);
        }

        let authority = scenario_trials
            .iter()
            .map(|trial| trial.authority_receipt_digest())
            .filter(|digest| !digest.is_zero())
            .collect::<BTreeSet<_>>()
            .len();
        if authority < requirement.minimum_distinct_authority_receipts {
            return Err(
                HumanoidReachStrongDiversityFailure::InsufficientDistinctAuthorityReceipts,
            );
        }
    }
    Ok(())
}

fn validate_episode_authority_diversity(
    policy: &HumanoidReachEpisodeCampaignPolicy,
    cases: &[HumanoidReachAuthorityCommittedEpisodeCase],
) -> Result<(), HumanoidReachStrongDiversityFailure> {
    if cases.is_empty() || cases.iter().any(|case| !case.validate()) {
        return Err(HumanoidReachStrongDiversityFailure::InvalidEvidence);
    }

    let known = policy
        .required_scenarios
        .iter()
        .map(|requirement| requirement.scenario_id.as_str())
        .collect::<BTreeSet<_>>();
    if cases
        .iter()
        .any(|case| !known.contains(case.inner.scenario_id.as_str()))
    {
        return Err(HumanoidReachStrongDiversityFailure::UnknownScenario);
    }

    for requirement in &policy.required_scenarios {
        let authority = cases
            .iter()
            .filter(|case| case.inner.scenario_id == requirement.scenario_id)
            .flat_map(|case| case.steps().iter())
            .map(|step| step.authority_receipt_digest())
            .filter(|digest| !digest.is_zero())
            .collect::<BTreeSet<_>>()
            .len();
        if authority < requirement.minimum_distinct_authority_receipts {
            return Err(
                HumanoidReachStrongDiversityFailure::InsufficientDistinctAuthorityReceipts,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::HumanoidEvidenceDigest;

    #[test]
    fn digest_diversity_uses_full_sha_identity() {
        let a = HumanoidEvidenceDigest::from_bytes([1; 32]);
        let b = HumanoidEvidenceDigest::from_bytes([2; 32]);
        let values = [a, a, b].into_iter().collect::<BTreeSet<_>>();
        assert_eq!(values.len(), 2);
    }
}
