// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic, zero-I/O disconfirmation and Pareto planning for Symthaea investigations.
//!
//! This crate deliberately produces analysis candidates only. It has no network, persistence,
//! browser, model-provider, Mycelix, Xenia, credential, subprocess, or action API.
//!
//! Core non-equivalences:
//!
//! ```text
//! preferred hypothesis != true
//! failed falsifier search != confirmation
//! Pareto-nondominated != authorized
//! blocked candidate != analytically useless
//! planner output != search attempt != collection authority
//! ```

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use symthaea_investigation::{
    ExpectedRelationV1, HypothesisId, InformationProposalId, InvestigationAuthorityScopeV1,
    NextBestInformationProposalV1, PlanningBurdenV1, PlanningDispositionV1, PlanningValueV1,
};

pub const NEXT_INFORMATION_PROFILE_V1: &str = "symthaea:next-information:pareto-front:v1";

#[derive(Clone, PartialEq, Eq)]
pub enum PlanningError {
    DuplicateProposalId(String),
}

impl fmt::Display for PlanningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateProposalId(_) => write!(f, "duplicate proposal id: <redacted>"),
        }
    }
}

impl fmt::Debug for PlanningError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PlanningError({self})")
    }
}

impl Error for PlanningError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannerCandidateV1 {
    pub proposal: NextBestInformationProposalV1,
    pub relation_to_preferred_hypothesis: ExpectedRelationV1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockedReasonV1 {
    Privacy,
    Opsec,
    Policy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockedCandidateV1 {
    pub proposal_id: InformationProposalId,
    pub reason: BlockedReasonV1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DominationWitnessV1 {
    pub dominated_proposal_id: InformationProposalId,
    pub dominating_proposal_id: InformationProposalId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanningLimitationV1 {
    PriorFalsifierSearchUnknownCoverage,
    NoAvailableDisconfirmationCandidate,
    NoEligibleDisconfirmationCandidate,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanningSummaryV1 {
    pub profile: &'static str,
    pub preferred_hypothesis_id: HypothesisId,
    pub disconfirmation_candidate_ids: Vec<InformationProposalId>,
    pub eligible_disconfirmation_candidate_ids: Vec<InformationProposalId>,
    pub eligible_pareto_front_ids: Vec<InformationProposalId>,
    pub blocked_candidates: Vec<BlockedCandidateV1>,
    pub domination_witnesses: Vec<DominationWitnessV1>,
    pub limitations: Vec<PlanningLimitationV1>,
}

impl PlanningSummaryV1 {
    pub fn authority_scope(&self) -> InvestigationAuthorityScopeV1 {
        InvestigationAuthorityScopeV1::CandidateAnalysisOnly
    }
}

/// Produce a deterministic, candidate-only planning summary.
///
/// `prior_falsifier_search_unknown_coverage` is recorded only as a limitation. It never changes
/// hypothesis confidence and cannot make the preferred hypothesis stronger.
pub fn plan_next_information(
    preferred_hypothesis_id: HypothesisId,
    candidates: Vec<PlannerCandidateV1>,
    prior_falsifier_search_unknown_coverage: bool,
) -> Result<PlanningSummaryV1, PlanningError> {
    let mut seen = BTreeSet::new();
    for candidate in &candidates {
        if !seen.insert(candidate.proposal.id.clone()) {
            return Err(PlanningError::DuplicateProposalId(
                candidate.proposal.id.as_str().to_string(),
            ));
        }
    }

    let mut disconfirmation_candidate_ids = Vec::new();
    let mut eligible_disconfirmation_candidate_ids = Vec::new();
    let mut eligible = Vec::new();
    let mut blocked_candidates = Vec::new();

    for candidate in &candidates {
        let is_disconfirmation = is_disconfirmation_relation(candidate.relation_to_preferred_hypothesis);
        if is_disconfirmation {
            disconfirmation_candidate_ids.push(candidate.proposal.id.clone());
        }

        match candidate.proposal.disposition {
            PlanningDispositionV1::BlockedByPrivacy => {
                blocked_candidates.push(BlockedCandidateV1 {
                    proposal_id: candidate.proposal.id.clone(),
                    reason: BlockedReasonV1::Privacy,
                });
            }
            PlanningDispositionV1::BlockedByOpsec => {
                blocked_candidates.push(BlockedCandidateV1 {
                    proposal_id: candidate.proposal.id.clone(),
                    reason: BlockedReasonV1::Opsec,
                });
            }
            PlanningDispositionV1::BlockedByPolicy => {
                blocked_candidates.push(BlockedCandidateV1 {
                    proposal_id: candidate.proposal.id.clone(),
                    reason: BlockedReasonV1::Policy,
                });
            }
            PlanningDispositionV1::ProposalOnly
            | PlanningDispositionV1::NeedsExternalAuthorization => {
                if is_disconfirmation {
                    eligible_disconfirmation_candidate_ids.push(candidate.proposal.id.clone());
                }
                eligible.push(candidate);
            }
        }
    }

    let mut eligible_pareto_front_ids = Vec::new();
    let mut domination_witnesses = Vec::new();

    for candidate in &eligible {
        let mut dominated = false;
        for other in &eligible {
            if candidate.proposal.id == other.proposal.id {
                continue;
            }
            if dominates(&other.proposal, &candidate.proposal) {
                dominated = true;
                domination_witnesses.push(DominationWitnessV1 {
                    dominated_proposal_id: candidate.proposal.id.clone(),
                    dominating_proposal_id: other.proposal.id.clone(),
                });
            }
        }
        if !dominated {
            eligible_pareto_front_ids.push(candidate.proposal.id.clone());
        }
    }

    disconfirmation_candidate_ids.sort();
    eligible_disconfirmation_candidate_ids.sort();
    eligible_pareto_front_ids.sort();
    blocked_candidates.sort_by(|a, b| a.proposal_id.cmp(&b.proposal_id));
    domination_witnesses.sort_by(|a, b| {
        a.dominated_proposal_id
            .cmp(&b.dominated_proposal_id)
            .then_with(|| a.dominating_proposal_id.cmp(&b.dominating_proposal_id))
    });

    let mut limitations = Vec::new();
    if prior_falsifier_search_unknown_coverage {
        limitations.push(PlanningLimitationV1::PriorFalsifierSearchUnknownCoverage);
    }
    if disconfirmation_candidate_ids.is_empty() {
        limitations.push(PlanningLimitationV1::NoAvailableDisconfirmationCandidate);
    } else if eligible_disconfirmation_candidate_ids.is_empty() {
        limitations.push(PlanningLimitationV1::NoEligibleDisconfirmationCandidate);
    }

    Ok(PlanningSummaryV1 {
        profile: NEXT_INFORMATION_PROFILE_V1,
        preferred_hypothesis_id,
        disconfirmation_candidate_ids,
        eligible_disconfirmation_candidate_ids,
        eligible_pareto_front_ids,
        blocked_candidates,
        domination_witnesses,
        limitations,
    })
}

fn is_disconfirmation_relation(relation: ExpectedRelationV1) -> bool {
    matches!(
        relation,
        ExpectedRelationV1::DiscriminatesAgainst
            | ExpectedRelationV1::StronglyDiscriminatesAgainst
    )
}

fn dominates(a: &NextBestInformationProposalV1, b: &NextBestInformationProposalV1) -> bool {
    let benefit_pairs = [
        (a.discriminating_power, b.discriminating_power),
        (a.contradiction_value, b.contradiction_value),
        (a.dependency_reduction, b.dependency_reduction),
        (a.currentness_gap_reduction, b.currentness_gap_reduction),
        (a.coverage_gap_reduction, b.coverage_gap_reduction),
        (a.expected_reproducibility, b.expected_reproducibility),
    ];
    let burden_pairs = [
        (a.expected_collection_cost, b.expected_collection_cost),
        (a.privacy_sensitivity, b.privacy_sensitivity),
        (a.opsec_disclosure_cost, b.opsec_disclosure_cost),
    ];

    let mut strictly_better = false;

    for (left, right) in benefit_pairs {
        match compare_value(left, right) {
            Some(Ordering::Less) | None => return false,
            Some(Ordering::Greater) => strictly_better = true,
            Some(Ordering::Equal) => {}
        }
    }

    for (left, right) in burden_pairs {
        match compare_burden(left, right) {
            Some(Ordering::Greater) | None => return false,
            Some(Ordering::Less) => strictly_better = true,
            Some(Ordering::Equal) => {}
        }
    }

    strictly_better
}

fn compare_value(a: PlanningValueV1, b: PlanningValueV1) -> Option<Ordering> {
    use PlanningValueV1::*;
    match (a, b) {
        (Unknown, Unknown) => Some(Ordering::Equal),
        (Unknown, _) | (_, Unknown) => None,
        _ => Some(value_rank(a).cmp(&value_rank(b))),
    }
}

fn compare_burden(a: PlanningBurdenV1, b: PlanningBurdenV1) -> Option<Ordering> {
    use PlanningBurdenV1::*;
    match (a, b) {
        (Unknown, Unknown) => Some(Ordering::Equal),
        (Unknown, _) | (_, Unknown) => None,
        _ => Some(burden_rank(a).cmp(&burden_rank(b))),
    }
}

fn value_rank(value: PlanningValueV1) -> u8 {
    use PlanningValueV1::*;
    match value {
        None => 0,
        Low => 1,
        Medium => 2,
        High => 3,
        VeryHigh => 4,
        Unknown => unreachable!("unknown values are handled before ranking"),
    }
}

fn burden_rank(value: PlanningBurdenV1) -> u8 {
    use PlanningBurdenV1::*;
    match value {
        None => 0,
        Low => 1,
        Medium => 2,
        High => 3,
        VeryHigh => 4,
        Unknown => unreachable!("unknown burdens are handled before ranking"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_investigation::{
        ObservationCandidateId, PlanningBurdenV1 as Burden, PlanningDispositionV1,
        PlanningValueV1 as Value, PropositionRef,
    };

    #[allow(clippy::too_many_arguments)]
    fn candidate(
        id: &str,
        relation: ExpectedRelationV1,
        disposition: PlanningDispositionV1,
        discriminating: PlanningValueV1,
        contradiction: PlanningValueV1,
        dependency: PlanningValueV1,
        currentness: PlanningValueV1,
        coverage: PlanningValueV1,
        reproducibility: PlanningValueV1,
        collection_cost: PlanningBurdenV1,
        privacy: PlanningBurdenV1,
        opsec: PlanningBurdenV1,
    ) -> PlannerCandidateV1 {
        PlannerCandidateV1 {
            proposal: NextBestInformationProposalV1 {
                id: InformationProposalId::new(id).unwrap(),
                target_observation_id: ObservationCandidateId::new(id).unwrap(),
                discriminating_power: discriminating,
                dependency_reduction: dependency,
                contradiction_value: contradiction,
                currentness_gap_reduction: currentness,
                coverage_gap_reduction: coverage,
                expected_collection_cost: collection_cost,
                privacy_sensitivity: privacy,
                opsec_disclosure_cost: opsec,
                expected_reproducibility: reproducibility,
                disposition,
                rationale_ref: PropositionRef::new(format!("rationale:{id}")).unwrap(),
            },
            relation_to_preferred_hypothesis: relation,
        }
    }

    fn reservoir_candidates() -> Vec<PlannerCandidateV1> {
        vec![
            candidate(
                "D1",
                ExpectedRelationV1::StronglyDiscriminatesAgainst,
                PlanningDispositionV1::NeedsExternalAuthorization,
                Value::VeryHigh,
                Value::VeryHigh,
                Value::Low,
                Value::High,
                Value::High,
                Value::VeryHigh,
                Burden::Medium,
                Burden::Low,
                Burden::Low,
            ),
            candidate(
                "D2",
                ExpectedRelationV1::StronglyDiscriminatesAgainst,
                PlanningDispositionV1::NeedsExternalAuthorization,
                Value::VeryHigh,
                Value::VeryHigh,
                Value::Low,
                Value::Medium,
                Value::Medium,
                Value::VeryHigh,
                Burden::Low,
                Burden::Low,
                Burden::Low,
            ),
            candidate(
                "D3",
                ExpectedRelationV1::NeutralUnderHypothesis,
                PlanningDispositionV1::NeedsExternalAuthorization,
                Value::Medium,
                Value::Low,
                Value::VeryHigh,
                Value::Low,
                Value::Medium,
                Value::VeryHigh,
                Burden::Low,
                Burden::Low,
                Burden::Low,
            ),
            candidate(
                "D4",
                ExpectedRelationV1::StronglyDiscriminatesAgainst,
                PlanningDispositionV1::BlockedByPrivacy,
                Value::VeryHigh,
                Value::VeryHigh,
                Value::Medium,
                Value::VeryHigh,
                Value::High,
                Value::High,
                Burden::Medium,
                Burden::VeryHigh,
                Burden::High,
            ),
            candidate(
                "D5",
                ExpectedRelationV1::WeaklyDiscriminatesFor,
                PlanningDispositionV1::NeedsExternalAuthorization,
                Value::Low,
                Value::None,
                Value::None,
                Value::Low,
                Value::None,
                Value::Medium,
                Burden::Medium,
                Burden::Low,
                Burden::Low,
            ),
        ]
    }

    fn ids(values: &[InformationProposalId]) -> Vec<&str> {
        values.iter().map(|id| id.as_str()).collect()
    }

    #[test]
    fn reservoir_fixture_matches_frozen_planner_theorem() {
        let summary = plan_next_information(
            HypothesisId::new("H1").unwrap(),
            reservoir_candidates(),
            true,
        )
        .unwrap();

        assert_eq!(summary.profile, NEXT_INFORMATION_PROFILE_V1);
        assert_eq!(ids(&summary.disconfirmation_candidate_ids), vec!["D1", "D2", "D4"]);
        assert_eq!(
            ids(&summary.eligible_disconfirmation_candidate_ids),
            vec!["D1", "D2"]
        );
        assert_eq!(ids(&summary.eligible_pareto_front_ids), vec!["D1", "D2", "D3"]);
        assert!(summary.blocked_candidates.iter().any(|blocked| {
            blocked.proposal_id.as_str() == "D4" && blocked.reason == BlockedReasonV1::Privacy
        }));
        assert!(summary.domination_witnesses.iter().any(|witness| {
            witness.dominated_proposal_id.as_str() == "D5"
                && witness.dominating_proposal_id.as_str() == "D2"
        }));
        assert!(summary
            .limitations
            .contains(&PlanningLimitationV1::PriorFalsifierSearchUnknownCoverage));
        assert_eq!(
            summary.authority_scope(),
            InvestigationAuthorityScopeV1::CandidateAnalysisOnly
        );
    }

    #[test]
    fn candidate_input_order_does_not_change_set_semantic_output() {
        let forward = plan_next_information(
            HypothesisId::new("H1").unwrap(),
            reservoir_candidates(),
            true,
        )
        .unwrap();
        let mut reversed_candidates = reservoir_candidates();
        reversed_candidates.reverse();
        let reversed = plan_next_information(
            HypothesisId::new("H1").unwrap(),
            reversed_candidates,
            true,
        )
        .unwrap();

        assert_eq!(forward, reversed);
    }

    #[test]
    fn removing_all_disconfirmation_candidates_records_limitation_without_confidence_update() {
        let candidates = reservoir_candidates()
            .into_iter()
            .filter(|candidate| {
                !matches!(
                    candidate.relation_to_preferred_hypothesis,
                    ExpectedRelationV1::DiscriminatesAgainst
                        | ExpectedRelationV1::StronglyDiscriminatesAgainst
                )
            })
            .collect();

        let summary = plan_next_information(
            HypothesisId::new("H1").unwrap(),
            candidates,
            false,
        )
        .unwrap();

        assert!(summary.disconfirmation_candidate_ids.is_empty());
        assert!(summary
            .limitations
            .contains(&PlanningLimitationV1::NoAvailableDisconfirmationCandidate));
    }

    #[test]
    fn blocked_only_disconfirmation_is_not_misreported_as_eligible() {
        let blocked = candidate(
            "D4",
            ExpectedRelationV1::StronglyDiscriminatesAgainst,
            PlanningDispositionV1::BlockedByPrivacy,
            Value::VeryHigh,
            Value::VeryHigh,
            Value::High,
            Value::High,
            Value::High,
            Value::High,
            Burden::Medium,
            Burden::VeryHigh,
            Burden::High,
        );
        let summary = plan_next_information(
            HypothesisId::new("H1").unwrap(),
            vec![blocked],
            false,
        )
        .unwrap();

        assert_eq!(ids(&summary.disconfirmation_candidate_ids), vec!["D4"]);
        assert!(summary.eligible_disconfirmation_candidate_ids.is_empty());
        assert!(summary
            .limitations
            .contains(&PlanningLimitationV1::NoEligibleDisconfirmationCandidate));
    }

    #[test]
    fn blocked_high_value_candidate_never_enters_eligible_pareto_front() {
        let summary = plan_next_information(
            HypothesisId::new("H1").unwrap(),
            reservoir_candidates(),
            false,
        )
        .unwrap();

        assert!(!ids(&summary.eligible_pareto_front_ids).contains(&"D4"));
        assert!(summary.blocked_candidates.iter().any(|blocked| {
            blocked.proposal_id.as_str() == "D4" && blocked.reason == BlockedReasonV1::Privacy
        }));
    }

    #[test]
    fn unknown_coordinate_is_incomparable_not_best_or_worst() {
        let a = candidate(
            "A",
            ExpectedRelationV1::NeutralUnderHypothesis,
            PlanningDispositionV1::ProposalOnly,
            Value::Unknown,
            Value::High,
            Value::High,
            Value::High,
            Value::High,
            Value::High,
            Burden::Low,
            Burden::Low,
            Burden::Low,
        );
        let b = candidate(
            "B",
            ExpectedRelationV1::NeutralUnderHypothesis,
            PlanningDispositionV1::ProposalOnly,
            Value::High,
            Value::High,
            Value::High,
            Value::High,
            Value::High,
            Value::High,
            Burden::Low,
            Burden::Low,
            Burden::Low,
        );

        let summary = plan_next_information(
            HypothesisId::new("H1").unwrap(),
            vec![a, b],
            false,
        )
        .unwrap();

        assert_eq!(ids(&summary.eligible_pareto_front_ids), vec!["A", "B"]);
        assert!(summary.domination_witnesses.is_empty());
    }

    #[test]
    fn duplicate_proposal_id_rejects_without_diagnostic_leakage() {
        let first = candidate(
            "sensitive-proposal-7f8a",
            ExpectedRelationV1::NeutralUnderHypothesis,
            PlanningDispositionV1::ProposalOnly,
            Value::Medium,
            Value::Medium,
            Value::Medium,
            Value::Medium,
            Value::Medium,
            Value::Medium,
            Burden::Low,
            Burden::Low,
            Burden::Low,
        );
        let second = first.clone();

        let error = plan_next_information(
            HypothesisId::new("H1").unwrap(),
            vec![first, second],
            false,
        )
        .unwrap_err();

        assert!(!format!("{error}").contains("sensitive-proposal-7f8a"));
        assert!(!format!("{error:?}").contains("sensitive-proposal-7f8a"));
    }
}
