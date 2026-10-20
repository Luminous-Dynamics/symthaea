//! Held-out replay for evaluating experiment-priority decisions against realized outcomes.
//!
//! This evaluator is deliberately downstream of the selector: estimates choose a proposal, while
//! independently sealed outcome receipts score it. It is not a marine-performance claim, a causal
//! value-of-information estimator, or proof that caller-attested split/freeze declarations are true.

use super::experiment_priority::{
    ExperimentPriorityDisposition, ExperimentPriorityReport, ProposalAssessment, ProposalDisposition,
};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RealizedOutcomeStatus {
    /// An evaluated result and its signed decision-loss change were recorded.
    Completed,
    /// The attempt failed before a usable outcome could be measured.
    Failed,
    /// The attempt was stopped or its result was censored.
    Censored,
    /// The proposed evaluation fell outside its declared applicability domain.
    OutOfDomain,
    /// The result did not meet the frozen validity protocol.
    InvalidResult,
}

impl RealizedOutcomeStatus {
    fn canonical_tag(self) -> u8 {
        match self {
            Self::Completed => 1,
            Self::Failed => 2,
            Self::Censored => 3,
            Self::OutOfDomain => 4,
            Self::InvalidResult => 5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct RealizedExperimentOutcome {
    pub proposal_id: String,
    /// Digest of the frozen, common evaluation-group roster used to compare all candidates.
    pub evaluation_group_roster_sha256: String,
    pub status: RealizedOutcomeStatus,
    /// Positive means predeclared decision loss decreased; negative means it increased.
    /// Only Completed outcomes may supply this field.
    pub realized_decision_loss_reduction_units: Option<i64>,
    /// Actual end-to-end cost, including failed work and required follow-up validation.
    pub actual_end_to_end_cost_units: Option<u64>,
    /// SHA-256 of the frozen outcome receipt/artifact; format and identity only, not publisher auth.
    pub outcome_artifact_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PriorityReplayProfile {
    pub replay_id: String,
    pub profile_id: String,
    pub campaign_id: String,
    pub cost_basis_id: String,
    pub decision_value_basis_id: String,
    pub decision_value_estimator_method_id: String,
    pub decision_value_estimator_method_revision: String,
    /// Digest of the preregistered independent evaluation-group roster shared by every outcome.
    pub evaluation_group_roster_sha256: String,
    /// Digest over the complete held-out outcome roster, computed with
    /// canonical_outcome_roster_sha256; not a substitute for source authenticity.
    pub outcome_roster_sha256: String,
    pub budget_units: u64,
    /// These are independent-review / freeze attestations supplied by the caller.
    pub candidate_roster_frozen_before_outcomes: bool,
    pub policy_frozen_before_outcomes: bool,
    pub evaluation_groups_independent_of_estimation: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReplayPolicyKind {
    EvidenceAwarePriority,
    CheapestEstimatedCost,
    HighestEstimatedDecisionValue,
    HindsightOracle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayDisposition {
    Evaluated,
    PrioritySelectionExceededBudget,
    InvalidInput,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RatioRelation {
    BelowOracle,
    TiedWithOracle,
    AboveOracle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicyReplayScore {
    pub policy: ReplayPolicyKind,
    pub selected_proposal_id: Option<String>,
    pub outcome_status: Option<RealizedOutcomeStatus>,
    /// None means no selected outcome was observed as complete. It is not treated as measured zero.
    pub observed_decision_loss_reduction_units: Option<i64>,
    /// Conservative score for comparisons: incomplete outcomes get zero credit; over-budget
    /// outcomes also get zero credit. The cost is still reported and charged.
    pub budget_adjusted_gain_units: Option<i64>,
    pub actual_end_to_end_cost_units: Option<u64>,
    pub within_budget: bool,
    pub gain_per_cost_numerator: Option<i64>,
    pub gain_per_cost_denominator: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PriorityReplayIssue {
    EmptyIdentity { field: &'static str },
    InvalidDigest { field: &'static str },
    ProfileCampaignMismatch,
    CostBasisMismatch,
    DecisionValueContractMismatch,
    EvaluationGroupRosterMismatch { proposal_id: String },
    DuplicateCandidateProposalId { proposal_id: String },
    ReportBudgetMismatch,
    SelectorDidNotSelectOptionalExperiment,
    MandatoryRequirementStillDue,
    RequiredAttestationMissing { field: &'static str },
    OutcomeRosterDigestMismatch,
    DuplicateOutcomeId { proposal_id: String },
    MissingOutcome { proposal_id: String },
    UnexpectedOutcome { proposal_id: String },
    InvalidOutcomeReceipt { proposal_id: String },
    MissingOrZeroActualCost { proposal_id: String },
    OutcomeValueStatusMismatch { proposal_id: String },
    SelectedProposalNotEligible { proposal_id: String },
    NoEligibleOptionalProposals,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExperimentPriorityReplayReport {
    pub replay_id: String,
    pub profile_id: String,
    pub campaign_id: String,
    pub outcome_roster_sha256: String,
    pub budget_units: u64,
    pub disposition: ReplayDisposition,
    pub scores: Vec<PolicyReplayScore>,
    pub hindsight_oracle_proposal_id: Option<String>,
    /// Oracle measured gain minus the selector's credited gain, floored at zero. The value is
    /// absent if no positive, complete, in-budget oracle outcome exists.
    pub absolute_decision_loss_regret_units: Option<u128>,
    pub priority_ratio_relation_to_oracle: Option<RatioRelation>,
    pub issues: Vec<PriorityReplayIssue>,
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.as_bytes().iter().all(u8::is_ascii_hexdigit)
}

fn push_field(hash: &mut Sha256, value: &[u8]) {
    hash.update((value.len() as u64).to_le_bytes());
    hash.update(value);
}

fn push_optional_i64(hash: &mut Sha256, value: Option<i64>) {
    match value {
        Some(value) => {
            hash.update([1]);
            hash.update(value.to_le_bytes());
        }
        None => hash.update([0]),
    }
}

fn push_optional_u64(hash: &mut Sha256, value: Option<u64>) {
    match value {
        Some(value) => {
            hash.update([1]);
            hash.update(value.to_le_bytes());
        }
        None => hash.update([0]),
    }
}

/// Computes a versioned digest of all outcome fields, with order-independent canonical sorting.
/// Duplicate proposal IDs are still invalid input even though the digest function is deterministic.
pub fn canonical_outcome_roster_sha256(outcomes: &[RealizedExperimentOutcome]) -> String {
    let mut canonical = outcomes.to_vec();
    canonical.sort();
    let mut hash = Sha256::new();
    push_field(&mut hash, b"marine-priority-outcome-roster-v1");
    hash.update((canonical.len() as u64).to_le_bytes());
    for outcome in &canonical {
        push_field(&mut hash, outcome.proposal_id.as_bytes());
        push_field(&mut hash, outcome.evaluation_group_roster_sha256.as_bytes());
        hash.update([outcome.status.canonical_tag()]);
        push_optional_i64(&mut hash, outcome.realized_decision_loss_reduction_units);
        push_optional_u64(&mut hash, outcome.actual_end_to_end_cost_units);
        push_field(&mut hash, outcome.outcome_artifact_sha256.as_bytes());
    }
    let digest = hash.finalize();
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

fn eligible_optional<'a>(
    report: &'a ExperimentPriorityReport,
) -> Vec<&'a ProposalAssessment> {
    let mut candidates: Vec<_> = report
        .assessments
        .iter()
        .filter(|assessment| {
            assessment.disposition == ProposalDisposition::EligibleForOptionalRanking
                && assessment.decision_value_usable
                && assessment
                    .decision_value_lower_bound_units
                    .is_some_and(|value| value > 0)
                && assessment
                    .estimated_end_to_end_cost_units
                    .is_some_and(|cost| cost > 0 && cost <= report.budget_remaining_units)
        })
        .collect();
    candidates.sort_by(|a, b| a.proposal_id.cmp(&b.proposal_id));
    candidates
}

fn ratio_cmp(a_gain: i64, a_cost: u64, b_gain: i64, b_cost: u64) -> Ordering {
    // i64 × u64 fits in i128, including the signed lower endpoint.
    ((a_gain as i128) * (b_cost as i128)).cmp(&((b_gain as i128) * (a_cost as i128)))
}

fn outcome_for<'a>(
    outcomes: &'a BTreeMap<&str, &RealizedExperimentOutcome>,
    proposal_id: &str,
) -> Option<&'a RealizedExperimentOutcome> {
    outcomes.get(proposal_id).copied()
}

fn score(
    kind: ReplayPolicyKind,
    proposal_id: Option<&str>,
    outcomes: &BTreeMap<&str, &RealizedExperimentOutcome>,
    budget_units: u64,
) -> PolicyReplayScore {
    let Some(proposal_id) = proposal_id else {
        return PolicyReplayScore {
            policy: kind,
            selected_proposal_id: None,
            outcome_status: None,
            observed_decision_loss_reduction_units: None,
            budget_adjusted_gain_units: None,
            actual_end_to_end_cost_units: None,
            within_budget: false,
            gain_per_cost_numerator: None,
            gain_per_cost_denominator: None,
        };
    };
    let Some(outcome) = outcome_for(outcomes, proposal_id) else {
        return PolicyReplayScore {
            policy: kind,
            selected_proposal_id: Some(proposal_id.to_string()),
            outcome_status: None,
            observed_decision_loss_reduction_units: None,
            budget_adjusted_gain_units: None,
            actual_end_to_end_cost_units: None,
            within_budget: false,
            gain_per_cost_numerator: None,
            gain_per_cost_denominator: None,
        };
    };

    let actual_cost = outcome.actual_end_to_end_cost_units;
    let within_budget = actual_cost.is_some_and(|cost| cost > 0 && cost <= budget_units);
    let observed_gain = (outcome.status == RealizedOutcomeStatus::Completed)
        .then_some(outcome.realized_decision_loss_reduction_units)
        .flatten();
    let credited_gain = if within_budget {
        observed_gain.or(Some(0))
    } else {
        Some(0)
    };
    let ratio_denominator = actual_cost.filter(|cost| *cost > 0);
    PolicyReplayScore {
        policy: kind,
        selected_proposal_id: Some(proposal_id.to_string()),
        outcome_status: Some(outcome.status),
        observed_decision_loss_reduction_units: observed_gain,
        budget_adjusted_gain_units: credited_gain,
        actual_end_to_end_cost_units: actual_cost,
        within_budget,
        gain_per_cost_numerator: credited_gain,
        gain_per_cost_denominator: ratio_denominator,
    }
}

fn choose_cheapest<'a>(candidates: &[&'a ProposalAssessment]) -> Option<&'a ProposalAssessment> {
    candidates.iter().copied().min_by(|a, b| {
        a.estimated_end_to_end_cost_units
            .cmp(&b.estimated_end_to_end_cost_units)
            .then_with(|| a.proposal_id.cmp(&b.proposal_id))
    })
}

fn choose_highest_estimated_value<'a>(
    candidates: &[&'a ProposalAssessment],
) -> Option<&'a ProposalAssessment> {
    candidates.iter().copied().min_by(|a, b| {
        b.decision_value_lower_bound_units
            .cmp(&a.decision_value_lower_bound_units)
            .then_with(|| a.proposal_id.cmp(&b.proposal_id))
    })
}

fn choose_hindsight_oracle<'a>(
    candidates: &[&'a ProposalAssessment],
    outcomes: &BTreeMap<&str, &RealizedExperimentOutcome>,
    budget_units: u64,
) -> Option<&'a ProposalAssessment> {
    candidates
        .iter()
        .copied()
        .filter_map(|candidate| {
            let outcome = outcome_for(outcomes, &candidate.proposal_id)?;
            let gain = outcome.realized_decision_loss_reduction_units?;
            let cost = outcome.actual_end_to_end_cost_units?;
            (outcome.status == RealizedOutcomeStatus::Completed
                && gain > 0
                && cost > 0
                && cost <= budget_units)
                .then_some((candidate, gain, cost))
        })
        .max_by(|(a, a_gain, a_cost), (b, b_gain, b_cost)| {
            ratio_cmp(*a_gain, *a_cost, *b_gain, *b_cost)
                // Deterministic lexicographically smallest ID wins ties.
                .then_with(|| b.proposal_id.cmp(&a.proposal_id))
        })
        .map(|(candidate, _, _)| candidate)
}

fn ratio_relation(
    priority: &PolicyReplayScore,
    oracle: &PolicyReplayScore,
) -> Option<RatioRelation> {
    let priority_gain = priority.gain_per_cost_numerator?;
    let priority_cost = priority.gain_per_cost_denominator?;
    let oracle_gain = oracle.gain_per_cost_numerator?;
    let oracle_cost = oracle.gain_per_cost_denominator?;
    if !priority.within_budget || !oracle.within_budget || oracle_gain <= 0 {
        return None;
    }
    Some(match ratio_cmp(priority_gain, priority_cost, oracle_gain, oracle_cost) {
        Ordering::Less => RatioRelation::BelowOracle,
        Ordering::Equal => RatioRelation::TiedWithOracle,
        Ordering::Greater => RatioRelation::AboveOracle,
    })
}

/// Scores a frozen optional-selection decision against a full held-out outcome roster.
///
/// Required gates, evidence admissibility and estimated-budget eligibility must already have been
/// handled by prioritize_next_experiment. This replay rejects mandatory rounds, scope mismatches,
/// missing candidates/outcomes, incomplete outcome manifests, and absent freeze/independence
/// attestations. The attestations themselves require external review/access controls.
///
/// Baselines are deliberately simple: lowest estimated total cost and highest estimated absolute
/// value. The hindsight oracle is only an evaluation ceiling, chosen from completed, in-budget
/// realized outcomes; it must never be used as an input to the deployed selector.
pub fn replay_experiment_priority(
    profile: &PriorityReplayProfile,
    priority_report: &ExperimentPriorityReport,
    outcomes: &[RealizedExperimentOutcome],
) -> ExperimentPriorityReplayReport {
    let mut issues = Vec::new();
    for (value, field) in [
        (profile.replay_id.as_str(), "replay_id"),
        (profile.profile_id.as_str(), "profile_id"),
        (profile.campaign_id.as_str(), "campaign_id"),
        (profile.cost_basis_id.as_str(), "cost_basis_id"),
        (profile.decision_value_basis_id.as_str(), "decision_value_basis_id"),
        (profile.decision_value_estimator_method_id.as_str(), "decision_value_estimator_method_id"),
        (profile.decision_value_estimator_method_revision.as_str(), "decision_value_estimator_method_revision"),
    ] {
        if value.trim().is_empty() {
            issues.push(PriorityReplayIssue::EmptyIdentity { field });
        }
    }
    if !valid_sha256(&profile.outcome_roster_sha256) {
        issues.push(PriorityReplayIssue::InvalidDigest {
            field: "outcome_roster_sha256",
        });
    }
    if profile.profile_id != priority_report.profile_id
        || profile.campaign_id != priority_report.campaign_id
    {
        issues.push(PriorityReplayIssue::ProfileCampaignMismatch);
    }
    if profile.cost_basis_id != priority_report.cost_basis_id {
        issues.push(PriorityReplayIssue::CostBasisMismatch);
    }
    if profile.decision_value_basis_id != priority_report.decision_value_basis_id
        || profile.decision_value_estimator_method_id != priority_report.decision_value_estimator_method_id
        || profile.decision_value_estimator_method_revision != priority_report.decision_value_estimator_method_revision
    {
        issues.push(PriorityReplayIssue::DecisionValueContractMismatch);
    }
    if !valid_sha256(&profile.evaluation_group_roster_sha256) {
        issues.push(PriorityReplayIssue::InvalidDigest {
            field: "evaluation_group_roster_sha256",
        });
    }
    if profile.budget_units != priority_report.budget_remaining_units {
        issues.push(PriorityReplayIssue::ReportBudgetMismatch);
    }
    if priority_report.disposition != ExperimentPriorityDisposition::SelectedOptionalExperiment {
        issues.push(PriorityReplayIssue::SelectorDidNotSelectOptionalExperiment);
    }
    if !priority_report.mandatory_requirements_due.is_empty() {
        issues.push(PriorityReplayIssue::MandatoryRequirementStillDue);
    }
    for (attested, field) in [
        (
            profile.candidate_roster_frozen_before_outcomes,
            "candidate_roster_frozen_before_outcomes",
        ),
        (profile.policy_frozen_before_outcomes, "policy_frozen_before_outcomes"),
        (
            profile.evaluation_groups_independent_of_estimation,
            "evaluation_groups_independent_of_estimation",
        ),
    ] {
        if !attested {
            issues.push(PriorityReplayIssue::RequiredAttestationMissing { field });
        }
    }

    let observed_roster_sha256 = canonical_outcome_roster_sha256(outcomes);
    if profile.outcome_roster_sha256 != observed_roster_sha256 {
        issues.push(PriorityReplayIssue::OutcomeRosterDigestMismatch);
    }

    let mut outcome_counts = BTreeMap::<&str, usize>::new();
    for outcome in outcomes {
        *outcome_counts.entry(outcome.proposal_id.as_str()).or_default() += 1;
        if outcome.proposal_id.trim().is_empty()
            || !valid_sha256(&outcome.outcome_artifact_sha256)
            || !valid_sha256(&outcome.evaluation_group_roster_sha256)
        {
            issues.push(PriorityReplayIssue::InvalidOutcomeReceipt {
                proposal_id: outcome.proposal_id.clone(),
            });
        }
        if outcome.evaluation_group_roster_sha256 != profile.evaluation_group_roster_sha256 {
            issues.push(PriorityReplayIssue::EvaluationGroupRosterMismatch {
                proposal_id: outcome.proposal_id.clone(),
            });
        }
        if outcome.actual_end_to_end_cost_units.is_none_or(|cost| cost == 0) {
            issues.push(PriorityReplayIssue::MissingOrZeroActualCost {
                proposal_id: outcome.proposal_id.clone(),
            });
        }
        let value_status_matches = match outcome.status {
            RealizedOutcomeStatus::Completed => {
                outcome.realized_decision_loss_reduction_units.is_some()
            }
            RealizedOutcomeStatus::Failed
            | RealizedOutcomeStatus::Censored
            | RealizedOutcomeStatus::OutOfDomain
            | RealizedOutcomeStatus::InvalidResult => {
                outcome.realized_decision_loss_reduction_units.is_none()
            }
        };
        if !value_status_matches {
            issues.push(PriorityReplayIssue::OutcomeValueStatusMismatch {
                proposal_id: outcome.proposal_id.clone(),
            });
        }
    }
    for (proposal_id, count) in &outcome_counts {
        if *count > 1 {
            issues.push(PriorityReplayIssue::DuplicateOutcomeId {
                proposal_id: (*proposal_id).to_string(),
            });
        }
    }

    let candidates = eligible_optional(priority_report);
    if candidates.is_empty() {
        issues.push(PriorityReplayIssue::NoEligibleOptionalProposals);
    }
    let mut candidate_counts = BTreeMap::<&str, usize>::new();
    for candidate in &candidates {
        *candidate_counts.entry(candidate.proposal_id.as_str()).or_default() += 1;
    }
    for (proposal_id, count) in &candidate_counts {
        if *count > 1 {
            issues.push(PriorityReplayIssue::DuplicateCandidateProposalId {
                proposal_id: (*proposal_id).to_string(),
            });
        }
    }
    let candidate_ids: BTreeSet<&str> = candidates
        .iter()
        .map(|candidate| candidate.proposal_id.as_str())
        .collect();
    let outcome_ids: BTreeSet<&str> = outcome_counts.keys().copied().collect();
    for proposal_id in candidate_ids.difference(&outcome_ids) {
        issues.push(PriorityReplayIssue::MissingOutcome {
            proposal_id: (*proposal_id).to_string(),
        });
    }
    for proposal_id in outcome_ids.difference(&candidate_ids) {
        issues.push(PriorityReplayIssue::UnexpectedOutcome {
            proposal_id: (*proposal_id).to_string(),
        });
    }

    let selected_id = priority_report.selected_proposal_id.as_deref();
    if let Some(selected_id) = selected_id {
        if !candidate_ids.contains(selected_id) {
            issues.push(PriorityReplayIssue::SelectedProposalNotEligible {
                proposal_id: selected_id.to_string(),
            });
        }
    } else {
        issues.push(PriorityReplayIssue::SelectorDidNotSelectOptionalExperiment);
    }

    issues.sort();
    issues.dedup();
    let mut report = ExperimentPriorityReplayReport {
        replay_id: profile.replay_id.clone(),
        profile_id: profile.profile_id.clone(),
        campaign_id: profile.campaign_id.clone(),
        outcome_roster_sha256: observed_roster_sha256,
        budget_units: profile.budget_units,
        disposition: ReplayDisposition::InvalidInput,
        scores: Vec::new(),
        hindsight_oracle_proposal_id: None,
        absolute_decision_loss_regret_units: None,
        priority_ratio_relation_to_oracle: None,
        issues,
    };
    if !report.issues.is_empty() {
        return report;
    }

    let outcomes_by_id: BTreeMap<&str, &RealizedExperimentOutcome> = outcomes
        .iter()
        .map(|outcome| (outcome.proposal_id.as_str(), outcome))
        .collect();
    let cheapest = choose_cheapest(&candidates).map(|item| item.proposal_id.as_str());
    let highest_value =
        choose_highest_estimated_value(&candidates).map(|item| item.proposal_id.as_str());
    let oracle = choose_hindsight_oracle(&candidates, &outcomes_by_id, profile.budget_units)
        .map(|item| item.proposal_id.as_str());

    let priority_score = score(
        ReplayPolicyKind::EvidenceAwarePriority,
        selected_id,
        &outcomes_by_id,
        profile.budget_units,
    );
    let cheapest_score = score(
        ReplayPolicyKind::CheapestEstimatedCost,
        cheapest,
        &outcomes_by_id,
        profile.budget_units,
    );
    let highest_value_score = score(
        ReplayPolicyKind::HighestEstimatedDecisionValue,
        highest_value,
        &outcomes_by_id,
        profile.budget_units,
    );
    let oracle_score = score(
        ReplayPolicyKind::HindsightOracle,
        oracle,
        &outcomes_by_id,
        profile.budget_units,
    );

    report.hindsight_oracle_proposal_id = oracle.map(str::to_string);
    if let (Some(oracle_gain), Some(priority_gain)) = (
        oracle_score.budget_adjusted_gain_units,
        priority_score.budget_adjusted_gain_units,
    ) {
        report.absolute_decision_loss_regret_units =
            Some((oracle_gain as i128 - priority_gain as i128).max(0) as u128);
        report.priority_ratio_relation_to_oracle = ratio_relation(&priority_score, &oracle_score);
    }
    report.disposition = if !priority_score.within_budget {
        ReplayDisposition::PrioritySelectionExceededBudget
    } else {
        ReplayDisposition::Evaluated
    };
    report.scores = vec![
        priority_score,
        cheapest_score,
        highest_value_score,
        oracle_score,
    ];
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const SHA_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SHA_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn assessment(id: &str, estimated_cost: u64, estimated_gain: i64) -> ProposalAssessment {
        ProposalAssessment {
            proposal_id: id.to_string(),
            disposition: ProposalDisposition::EligibleForOptionalRanking,
            is_mandatory: false,
            requirement_id: None,
            cost_units: Some(estimated_cost),
            estimated_end_to_end_cost_units: Some(estimated_cost),
            decision_value_lower_bound_units: Some(estimated_gain),
            decision_value_usable: true,
            issues: Vec::new(),
        }
    }

    fn outcome(
        id: &str,
        status: RealizedOutcomeStatus,
        gain: Option<i64>,
        cost: u64,
        digest: &str,
    ) -> RealizedExperimentOutcome {
        RealizedExperimentOutcome {
            proposal_id: id.to_string(),
            evaluation_group_roster_sha256: SHA_A.to_string(),
            status,
            realized_decision_loss_reduction_units: gain,
            actual_end_to_end_cost_units: Some(cost),
            outcome_artifact_sha256: digest.to_string(),
        }
    }

    fn fixture() -> (
        PriorityReplayProfile,
        ExperimentPriorityReport,
        Vec<RealizedExperimentOutcome>,
    ) {
        let outcomes = vec![
            outcome("predicted-high", RealizedOutcomeStatus::Completed, Some(4), 10, SHA_B),
            outcome("actually-good", RealizedOutcomeStatus::Completed, Some(8), 5, SHA_C),
        ];
        let report = ExperimentPriorityReport {
            profile_id: "profile-1".to_string(),
            campaign_id: "campaign-1".to_string(),
            cost_basis_id: "lab-hours-v1".to_string(),
            decision_value_basis_id: "loss-v1".to_string(),
            decision_value_estimator_method_id: "estimator-v1".to_string(),
            decision_value_estimator_method_revision: "r1".to_string(),
            disposition: ExperimentPriorityDisposition::SelectedOptionalExperiment,
            budget_remaining_units: 12,
            campaign_spent_units: Some(0),
            mandatory_requirements_due: Vec::new(),
            selected_proposal_id: Some("predicted-high".to_string()),
            selected_requirement_id: None,
            selected_cost_units: Some(10),
            selected_end_to_end_cost_units: Some(10),
            selected_decision_value_lower_bound_units: Some(100),
            selected_value_per_cost_numerator: Some(100),
            selected_value_per_cost_denominator: Some(10),
            assessments: vec![assessment("predicted-high", 10, 100), assessment("actually-good", 5, 20)],
            issues: Vec::new(),
        };
        let profile = PriorityReplayProfile {
            replay_id: "replay-001".to_string(),
            profile_id: "profile-1".to_string(),
            campaign_id: "campaign-1".to_string(),
            cost_basis_id: "lab-hours-v1".to_string(),
            decision_value_basis_id: "loss-v1".to_string(),
            decision_value_estimator_method_id: "estimator-v1".to_string(),
            decision_value_estimator_method_revision: "r1".to_string(),
            evaluation_group_roster_sha256: SHA_A.to_string(),
            outcome_roster_sha256: canonical_outcome_roster_sha256(&outcomes),
            budget_units: 12,
            candidate_roster_frozen_before_outcomes: true,
            policy_frozen_before_outcomes: true,
            evaluation_groups_independent_of_estimation: true,
        };
        (profile, report, outcomes)
    }

    #[test]
    fn priority_and_simple_baselines_are_scored_on_same_held_out_roster() {
        let (profile, report, outcomes) = fixture();
        let replay = replay_experiment_priority(&profile, &report, &outcomes);
        assert_eq!(replay.disposition, ReplayDisposition::Evaluated);
        assert_eq!(replay.hindsight_oracle_proposal_id.as_deref(), Some("actually-good"));
        assert_eq!(replay.absolute_decision_loss_regret_units, Some(4));
        assert_eq!(replay.priority_ratio_relation_to_oracle, Some(RatioRelation::BelowOracle));
        assert_eq!(replay.scores.len(), 4);
        assert_eq!(
            replay.scores.iter().find(|s| s.policy == ReplayPolicyKind::CheapestEstimatedCost)
                .and_then(|s| s.selected_proposal_id.as_deref()),
            Some("actually-good")
        );
        assert_eq!(
            replay.scores.iter().find(|s| s.policy == ReplayPolicyKind::HighestEstimatedDecisionValue)
                .and_then(|s| s.selected_proposal_id.as_deref()),
            Some("predicted-high")
        );
    }

    #[test]
    fn outcome_roster_digest_is_order_independent_but_content_bound() {
        let (_, _, outcomes) = fixture();
        let mut reversed = outcomes.clone();
        reversed.reverse();
        assert_eq!(canonical_outcome_roster_sha256(&outcomes), canonical_outcome_roster_sha256(&reversed));
        reversed[0].realized_decision_loss_reduction_units = Some(999);
        assert_ne!(canonical_outcome_roster_sha256(&outcomes), canonical_outcome_roster_sha256(&reversed));
    }

    #[test]
    fn missing_outcome_blocks_replay_instead_of_selective_scoring() {
        let (mut profile, report, mut outcomes) = fixture();
        outcomes.pop();
        profile.outcome_roster_sha256 = canonical_outcome_roster_sha256(&outcomes);
        let replay = replay_experiment_priority(&profile, &report, &outcomes);
        assert_eq!(replay.disposition, ReplayDisposition::InvalidInput);
        assert!(replay.issues.iter().any(|issue| matches!(issue, PriorityReplayIssue::MissingOutcome { .. })));
    }

    #[test]
    fn stale_outcome_digest_blocks_replay() {
        let (profile, report, outcomes) = fixture();
        let mut tampered = profile;
        tampered.outcome_roster_sha256 = SHA_A.to_string();
        let replay = replay_experiment_priority(&tampered, &report, &outcomes);
        assert_eq!(replay.disposition, ReplayDisposition::InvalidInput);
        assert!(replay.issues.contains(&PriorityReplayIssue::OutcomeRosterDigestMismatch));
    }

    #[test]
    fn duplicate_outcome_ids_block_replay() {
        let (mut profile, report, mut outcomes) = fixture();
        outcomes.push(outcomes[0].clone());
        profile.outcome_roster_sha256 = canonical_outcome_roster_sha256(&outcomes);
        let replay = replay_experiment_priority(&profile, &report, &outcomes);
        assert_eq!(replay.disposition, ReplayDisposition::InvalidInput);
        assert!(replay.issues.iter().any(|issue| matches!(issue, PriorityReplayIssue::DuplicateOutcomeId { .. })));
    }

    #[test]
    fn failed_outcome_gets_no_gain_credit_but_its_cost_remains_visible() {
        let (mut profile, report, mut outcomes) = fixture();
        outcomes[0] = outcome("predicted-high", RealizedOutcomeStatus::Failed, None, 10, SHA_B);
        profile.outcome_roster_sha256 = canonical_outcome_roster_sha256(&outcomes);
        let replay = replay_experiment_priority(&profile, &report, &outcomes);
        assert_eq!(replay.disposition, ReplayDisposition::Evaluated);
        let priority = replay.scores.iter().find(|s| s.policy == ReplayPolicyKind::EvidenceAwarePriority).unwrap();
        assert_eq!(priority.observed_decision_loss_reduction_units, None);
        assert_eq!(priority.budget_adjusted_gain_units, Some(0));
        assert_eq!(priority.actual_end_to_end_cost_units, Some(10));
    }

    #[test]
    fn actual_budget_overrun_is_not_hidden_by_a_cheaper_estimate() {
        let (mut profile, report, mut outcomes) = fixture();
        outcomes[0] = outcome("predicted-high", RealizedOutcomeStatus::Completed, Some(100), 13, SHA_B);
        profile.outcome_roster_sha256 = canonical_outcome_roster_sha256(&outcomes);
        let replay = replay_experiment_priority(&profile, &report, &outcomes);
        assert_eq!(replay.disposition, ReplayDisposition::PrioritySelectionExceededBudget);
        let priority = replay.scores.iter().find(|s| s.policy == ReplayPolicyKind::EvidenceAwarePriority).unwrap();
        assert!(!priority.within_budget);
        assert_eq!(priority.budget_adjusted_gain_units, Some(0));
        assert_eq!(priority.observed_decision_loss_reduction_units, Some(100));
    }

    #[test]
    fn non_completed_outcome_cannot_claim_measured_gain() {
        let (mut profile, report, mut outcomes) = fixture();
        outcomes[0] = outcome("predicted-high", RealizedOutcomeStatus::Censored, Some(4), 10, SHA_B);
        profile.outcome_roster_sha256 = canonical_outcome_roster_sha256(&outcomes);
        let replay = replay_experiment_priority(&profile, &report, &outcomes);
        assert_eq!(replay.disposition, ReplayDisposition::InvalidInput);
        assert!(replay.issues.iter().any(|issue| matches!(issue, PriorityReplayIssue::OutcomeValueStatusMismatch { .. })));
    }

    #[test]
    fn absent_freeze_or_independence_attestations_block_claim() {
        let (mut profile, report, outcomes) = fixture();
        profile.policy_frozen_before_outcomes = false;
        let replay = replay_experiment_priority(&profile, &report, &outcomes);
        assert_eq!(replay.disposition, ReplayDisposition::InvalidInput);
        assert!(replay.issues.iter().any(|issue| matches!(issue, PriorityReplayIssue::RequiredAttestationMissing { .. })));
    }

    #[test]
    fn mandatory_round_cannot_be_relabelled_as_optional_policy_replay() {
        let (profile, mut report, outcomes) = fixture();
        report.disposition = ExperimentPriorityDisposition::SelectedMandatoryEvidence;
        let replay = replay_experiment_priority(&profile, &report, &outcomes);
        assert_eq!(replay.disposition, ReplayDisposition::InvalidInput);
        assert!(replay.issues.contains(&PriorityReplayIssue::SelectorDidNotSelectOptionalExperiment));
    }

    #[test]
    fn no_positive_in_budget_realized_outcome_has_no_hindsight_oracle_or_regret() {
        let (mut profile, report, mut outcomes) = fixture();
        outcomes[0] = outcome("predicted-high", RealizedOutcomeStatus::Completed, Some(-5), 10, SHA_B);
        outcomes[1] = outcome("actually-good", RealizedOutcomeStatus::Completed, Some(0), 5, SHA_C);
        profile.outcome_roster_sha256 = canonical_outcome_roster_sha256(&outcomes);
        let replay = replay_experiment_priority(&profile, &report, &outcomes);
        assert_eq!(replay.disposition, ReplayDisposition::Evaluated);
        assert_eq!(replay.hindsight_oracle_proposal_id, None);
        assert_eq!(replay.absolute_decision_loss_regret_units, None);
    }
    #[test]
    fn outcomes_from_different_evaluation_group_rosters_cannot_be_compared() {
        let (mut profile, report, mut outcomes) = fixture();
        outcomes[1].evaluation_group_roster_sha256 = SHA_B.to_string();
        profile.outcome_roster_sha256 = canonical_outcome_roster_sha256(&outcomes);
        let replay = replay_experiment_priority(&profile, &report, &outcomes);
        assert_eq!(replay.disposition, ReplayDisposition::InvalidInput);
        assert!(replay.issues.iter().any(|issue| matches!(
            issue,
            PriorityReplayIssue::EvaluationGroupRosterMismatch { proposal_id }
                if proposal_id == "actually-good"
        )));
    }

    #[test]
    fn cost_and_decision_value_units_are_frozen_by_the_replay_profile() {
        let (mut profile, report, outcomes) = fixture();
        profile.cost_basis_id = "incompatible-cost-units".to_string();
        profile.decision_value_basis_id = "incompatible-loss-units".to_string();
        let replay = replay_experiment_priority(&profile, &report, &outcomes);
        assert_eq!(replay.disposition, ReplayDisposition::InvalidInput);
        assert!(replay.issues.contains(&PriorityReplayIssue::CostBasisMismatch));
        assert!(replay.issues.contains(&PriorityReplayIssue::DecisionValueContractMismatch));
    }

}
