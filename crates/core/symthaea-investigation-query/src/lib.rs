// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic, zero-I/O anti-confirmation query methodology for Symthaea investigations.
//!
//! This crate models query-strategy candidates only. It contains no raw query body type and no
//! network, browser, credential, model-provider, persistence, Mycelix, Xenia, subprocess, or
//! action API.
//!
//! Core non-equivalences:
//!
//! ```text
//! query matches preferred hypothesis != evidence supports it
//! query count != evidence count
//! wording diversity != methodological diversity
//! query commitment != disclosure permission
//! blocked query != executable query
//! zero results under UnknownCoverage != preferred-hypothesis support
//! ```

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use symthaea_investigation::{
    HypothesisId, ObservationCandidateId, ProfileRef, QueryCommitmentRef,
    SearchPlanAuthorityScopeV1,
};

pub const QUERY_STRATEGY_PROFILE_V1: &str = "symthaea:osint-query-strategy:v1";
pub const QUERY_SEMANTIC_NORMALIZATION_PROFILE_V1: &str =
    "symthaea:query-semantic-normalization:v1";
pub const MAX_QUERY_REF_BYTES: usize = 256;

fn validate_ref(role: &'static str, value: &str) -> Result<(), QueryStrategyError> {
    if value.is_empty() {
        return Err(QueryStrategyError::InvalidReference {
            role,
            reason: "empty",
        });
    }
    if value.len() > MAX_QUERY_REF_BYTES {
        return Err(QueryStrategyError::InvalidReference {
            role,
            reason: "too long",
        });
    }
    if !value.bytes().all(|b| (0x21..=0x7e).contains(&b)) {
        return Err(QueryStrategyError::InvalidReference {
            role,
            reason: "must contain graphic ASCII only",
        });
    }
    Ok(())
}

macro_rules! query_ref {
    ($name:ident) => {
        #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, QueryStrategyError> {
                let value = value.into();
                validate_ref(stringify!($name), &value)?;
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}(<redacted>)", stringify!($name))
            }
        }
    };
}

query_ref!(QueryCandidateId);
query_ref!(QuerySemanticRef);
query_ref!(QueryDuplicateGroupRef);

#[derive(Clone, PartialEq, Eq)]
pub enum QueryStrategyError {
    InvalidReference {
        role: &'static str,
        reason: &'static str,
    },
    DuplicateQueryId(String),
    DuplicateQueryCommitment(String),
    PreferredHypothesisListedAsAlternative(String),
    EmptyTargetHypotheses(String),
    EmptyTargetObservations(String),
}

impl fmt::Display for QueryStrategyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidReference { role, reason } => {
                write!(f, "invalid {role} reference: {reason}")
            }
            Self::DuplicateQueryId(_) => write!(f, "duplicate query id: <redacted>"),
            Self::DuplicateQueryCommitment(_) => {
                write!(f, "duplicate query commitment: <redacted>")
            }
            Self::PreferredHypothesisListedAsAlternative(_) => {
                write!(f, "preferred hypothesis cannot also be a required alternative: <redacted>")
            }
            Self::EmptyTargetHypotheses(_) => {
                write!(f, "query candidate requires at least one target hypothesis: <redacted>")
            }
            Self::EmptyTargetObservations(_) => {
                write!(f, "query candidate requires at least one target observation: <redacted>")
            }
        }
    }
}

impl fmt::Debug for QueryStrategyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "QueryStrategyError({self})")
    }
}

impl Error for QueryStrategyError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum QueryFamilyV1 {
    NeutralDescriptive,
    DisconfirmationSeeking,
    AlternativeExplanation,
    DependencyLineage,
    ExactIdentifier,
    SupportSeeking,
    SourceClassSpecific,
    TemporalCurrentness,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryPolicyDispositionV1 {
    ProposalOnly,
    BlockedByPrivacy,
    BlockedByOpsec,
    BlockedByPolicy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryCandidateV1 {
    pub id: QueryCandidateId,
    pub family: QueryFamilyV1,
    pub semantic_ref: QuerySemanticRef,
    pub query_commitment: QueryCommitmentRef,
    pub target_observations: Vec<ObservationCandidateId>,
    pub target_hypotheses: Vec<HypothesisId>,
    pub tool_capability_refs: Vec<ProfileRef>,
    pub expected_disclosure_surfaces: Vec<ProfileRef>,
    pub disposition: QueryPolicyDispositionV1,
    pub semantic_duplicate_group: Option<QueryDuplicateGroupRef>,
}

impl QueryCandidateV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: QueryCandidateId,
        family: QueryFamilyV1,
        semantic_ref: QuerySemanticRef,
        query_commitment: QueryCommitmentRef,
        target_observations: Vec<ObservationCandidateId>,
        target_hypotheses: Vec<HypothesisId>,
        tool_capability_refs: Vec<ProfileRef>,
        expected_disclosure_surfaces: Vec<ProfileRef>,
        disposition: QueryPolicyDispositionV1,
        semantic_duplicate_group: Option<QueryDuplicateGroupRef>,
    ) -> Result<Self, QueryStrategyError> {
        if target_hypotheses.is_empty() {
            return Err(QueryStrategyError::EmptyTargetHypotheses(
                id.as_str().to_string(),
            ));
        }
        if target_observations.is_empty() {
            return Err(QueryStrategyError::EmptyTargetObservations(
                id.as_str().to_string(),
            ));
        }
        Ok(Self {
            id,
            family,
            semantic_ref,
            query_commitment,
            target_observations,
            target_hypotheses,
            tool_capability_refs,
            expected_disclosure_surfaces,
            disposition,
            semantic_duplicate_group,
        })
    }

    pub fn authority_scope(&self) -> SearchPlanAuthorityScopeV1 {
        SearchPlanAuthorityScopeV1::ProposalOnly
    }

    pub fn is_policy_blocked(&self) -> bool {
        matches!(
            self.disposition,
            QueryPolicyDispositionV1::BlockedByPrivacy
                | QueryPolicyDispositionV1::BlockedByOpsec
                | QueryPolicyDispositionV1::BlockedByPolicy
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum BlockedQueryReasonV1 {
    Privacy,
    Opsec,
    Policy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockedQueryV1 {
    pub query_id: QueryCandidateId,
    pub reason: BlockedQueryReasonV1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QueryStrategyLimitationV1 {
    PriorSearchUnknownCoverage,
    NoAvailableDisconfirmationQueryCandidate,
    NoEligibleDisconfirmationQueryCandidate,
    MissingNeutralFamily,
    MissingDependencyLineageFamily,
    MissingAlternativeCoverage(HypothesisId),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryStrategySummaryV1 {
    pub preferred_hypothesis_id: HypothesisId,
    pub disconfirmation_query_ids: Vec<QueryCandidateId>,
    pub eligible_disconfirmation_query_ids: Vec<QueryCandidateId>,
    pub alternative_query_ids: Vec<QueryCandidateId>,
    pub neutral_query_ids: Vec<QueryCandidateId>,
    pub dependency_query_ids: Vec<QueryCandidateId>,
    pub blocked_queries: Vec<BlockedQueryV1>,
    pub represented_families: Vec<QueryFamilyV1>,
    pub eligible_families: Vec<QueryFamilyV1>,
    pub methodological_family_count: usize,
    pub eligible_semantic_candidate_count: usize,
    pub limitations: Vec<QueryStrategyLimitationV1>,
}

impl QueryStrategySummaryV1 {
    pub fn authority_scope(&self) -> SearchPlanAuthorityScopeV1 {
        SearchPlanAuthorityScopeV1::ProposalOnly
    }
}

/// Analyze a bounded query-methodology plan without generating or executing query text.
///
/// `prior_search_unknown_coverage` records only a methodological limitation. It cannot strengthen
/// the preferred hypothesis and does not create evidence.
pub fn analyze_query_strategy(
    preferred_hypothesis_id: HypothesisId,
    required_alternative_hypotheses: Vec<HypothesisId>,
    candidates: Vec<QueryCandidateV1>,
    prior_search_unknown_coverage: bool,
) -> Result<QueryStrategySummaryV1, QueryStrategyError> {
    let mut alternatives = BTreeSet::new();
    for alternative in required_alternative_hypotheses {
        if alternative == preferred_hypothesis_id {
            return Err(QueryStrategyError::PreferredHypothesisListedAsAlternative(
                alternative.as_str().to_string(),
            ));
        }
        alternatives.insert(alternative);
    }

    let mut seen_ids = BTreeSet::new();
    let mut seen_commitments = BTreeSet::new();
    for candidate in &candidates {
        if !seen_ids.insert(candidate.id.clone()) {
            return Err(QueryStrategyError::DuplicateQueryId(
                candidate.id.as_str().to_string(),
            ));
        }
        if !seen_commitments.insert(candidate.query_commitment.clone()) {
            return Err(QueryStrategyError::DuplicateQueryCommitment(
                candidate.query_commitment.as_str().to_string(),
            ));
        }
    }

    let mut disconfirmation = BTreeSet::new();
    let mut eligible_disconfirmation = BTreeSet::new();
    let mut alternative_queries = BTreeSet::new();
    let mut neutral_queries = BTreeSet::new();
    let mut dependency_queries = BTreeSet::new();
    let mut blocked_queries = Vec::new();
    let mut represented_families = BTreeSet::new();
    let mut eligible_families = BTreeSet::new();
    let mut eligible_semantic_refs = BTreeSet::new();
    let mut covered_alternatives = BTreeSet::new();

    for candidate in &candidates {
        represented_families.insert(candidate.family);

        let targets_preferred = candidate
            .target_hypotheses
            .iter()
            .any(|h| h == &preferred_hypothesis_id);
        let is_disconfirmation =
            candidate.family == QueryFamilyV1::DisconfirmationSeeking && targets_preferred;

        if is_disconfirmation {
            disconfirmation.insert(candidate.id.clone());
        }

        if candidate.is_policy_blocked() {
            let reason = match candidate.disposition {
                QueryPolicyDispositionV1::BlockedByPrivacy => BlockedQueryReasonV1::Privacy,
                QueryPolicyDispositionV1::BlockedByOpsec => BlockedQueryReasonV1::Opsec,
                QueryPolicyDispositionV1::BlockedByPolicy => BlockedQueryReasonV1::Policy,
                QueryPolicyDispositionV1::ProposalOnly => unreachable!(),
            };
            blocked_queries.push(BlockedQueryV1 {
                query_id: candidate.id.clone(),
                reason,
            });
            continue;
        }

        eligible_families.insert(candidate.family);
        eligible_semantic_refs.insert(candidate.semantic_ref.clone());

        if is_disconfirmation {
            eligible_disconfirmation.insert(candidate.id.clone());
        }

        match candidate.family {
            QueryFamilyV1::NeutralDescriptive => {
                neutral_queries.insert(candidate.id.clone());
            }
            QueryFamilyV1::AlternativeExplanation => {
                alternative_queries.insert(candidate.id.clone());
                for hypothesis in &candidate.target_hypotheses {
                    if alternatives.contains(hypothesis) {
                        covered_alternatives.insert(hypothesis.clone());
                    }
                }
            }
            QueryFamilyV1::DependencyLineage => {
                dependency_queries.insert(candidate.id.clone());
                alternative_queries.insert(candidate.id.clone());
                for hypothesis in &candidate.target_hypotheses {
                    if alternatives.contains(hypothesis) {
                        covered_alternatives.insert(hypothesis.clone());
                    }
                }
            }
            QueryFamilyV1::DisconfirmationSeeking
            | QueryFamilyV1::ExactIdentifier
            | QueryFamilyV1::SupportSeeking
            | QueryFamilyV1::SourceClassSpecific
            | QueryFamilyV1::TemporalCurrentness => {}
        }
    }

    blocked_queries.sort_by(|a, b| {
        a.query_id
            .as_str()
            .cmp(b.query_id.as_str())
            .then(a.reason.cmp(&b.reason))
    });

    let mut limitations = Vec::new();
    if prior_search_unknown_coverage {
        limitations.push(QueryStrategyLimitationV1::PriorSearchUnknownCoverage);
    }
    if disconfirmation.is_empty() {
        limitations.push(QueryStrategyLimitationV1::NoAvailableDisconfirmationQueryCandidate);
    } else if eligible_disconfirmation.is_empty() {
        limitations.push(QueryStrategyLimitationV1::NoEligibleDisconfirmationQueryCandidate);
    }
    if neutral_queries.is_empty() {
        limitations.push(QueryStrategyLimitationV1::MissingNeutralFamily);
    }
    if dependency_queries.is_empty() {
        limitations.push(QueryStrategyLimitationV1::MissingDependencyLineageFamily);
    }
    for alternative in alternatives {
        if !covered_alternatives.contains(&alternative) {
            limitations.push(QueryStrategyLimitationV1::MissingAlternativeCoverage(alternative));
        }
    }

    Ok(QueryStrategySummaryV1 {
        preferred_hypothesis_id,
        disconfirmation_query_ids: disconfirmation.into_iter().collect(),
        eligible_disconfirmation_query_ids: eligible_disconfirmation.into_iter().collect(),
        alternative_query_ids: alternative_queries.into_iter().collect(),
        neutral_query_ids: neutral_queries.into_iter().collect(),
        dependency_query_ids: dependency_queries.into_iter().collect(),
        blocked_queries,
        represented_families: represented_families.into_iter().collect(),
        eligible_families: eligible_families.iter().copied().collect(),
        methodological_family_count: eligible_families.len(),
        eligible_semantic_candidate_count: eligible_semantic_refs.len(),
        limitations,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(
        id: &str,
        family: QueryFamilyV1,
        semantic_ref: &str,
        commitment: &str,
        observation: &str,
        hypotheses: &[&str],
        disposition: QueryPolicyDispositionV1,
        duplicate_group: Option<&str>,
    ) -> QueryCandidateV1 {
        QueryCandidateV1::new(
            QueryCandidateId::new(id).unwrap(),
            family,
            QuerySemanticRef::new(semantic_ref).unwrap(),
            QueryCommitmentRef::new(commitment).unwrap(),
            vec![ObservationCandidateId::new(observation).unwrap()],
            hypotheses
                .iter()
                .map(|h| HypothesisId::new(*h).unwrap())
                .collect(),
            vec![ProfileRef::new("capability:synthetic").unwrap()],
            vec![ProfileRef::new("surface:query-terms").unwrap()],
            disposition,
            duplicate_group.map(|g| QueryDuplicateGroupRef::new(g).unwrap()),
        )
        .unwrap()
    }

    fn fixture_queries() -> Vec<QueryCandidateV1> {
        vec![
            q(
                "Q1",
                QueryFamilyV1::NeutralDescriptive,
                "query:maintenance-calibration-neutral",
                "sha256:q1-neutral-maintenance",
                "D2",
                &["H1", "H2"],
                QueryPolicyDispositionV1::ProposalOnly,
                None,
            ),
            q(
                "Q2",
                QueryFamilyV1::DisconfirmationSeeking,
                "query:sensor-fault-disconfirm-h1",
                "sha256:q2-sensor-fault",
                "D2",
                &["H1", "H2"],
                QueryPolicyDispositionV1::ProposalOnly,
                None,
            ),
            q(
                "Q3",
                QueryFamilyV1::AlternativeExplanation,
                "query:sensor-drift-alternative-h2",
                "sha256:q3-sensor-drift",
                "D2",
                &["H2"],
                QueryPolicyDispositionV1::ProposalOnly,
                None,
            ),
            q(
                "Q4",
                QueryFamilyV1::DependencyLineage,
                "query:upstream-syndication-lineage-h3",
                "sha256:q4-lineage",
                "D3",
                &["H3"],
                QueryPolicyDispositionV1::ProposalOnly,
                None,
            ),
            q(
                "Q5",
                QueryFamilyV1::ExactIdentifier,
                "query:exact-maintenance-record-id",
                "sha256:q5-exact-record",
                "D2",
                &["H1", "H2"],
                QueryPolicyDispositionV1::ProposalOnly,
                None,
            ),
            q(
                "Q6",
                QueryFamilyV1::SupportSeeking,
                "query:level-change-confirmation-wording-clone",
                "sha256:q6-confirmation-clone",
                "D1",
                &["H1"],
                QueryPolicyDispositionV1::ProposalOnly,
                Some("DG1"),
            ),
            q(
                "Q7",
                QueryFamilyV1::SourceClassSpecific,
                "query:protected-device-personnel-record",
                "sha256:q7-protected",
                "D2",
                &["H1", "H2"],
                QueryPolicyDispositionV1::BlockedByPrivacy,
                None,
            ),
            q(
                "Q8",
                QueryFamilyV1::TemporalCurrentness,
                "query:recent-calibration-currentness",
                "sha256:q8-currentness",
                "D2",
                &["H1", "H2"],
                QueryPolicyDispositionV1::ProposalOnly,
                None,
            ),
        ]
    }

    fn ids(values: &[QueryCandidateId]) -> Vec<&str> {
        values.iter().map(QueryCandidateId::as_str).collect()
    }

    #[test]
    fn reservoir_fixture_matches_frozen_query_methodology() {
        let summary = analyze_query_strategy(
            HypothesisId::new("H1").unwrap(),
            vec![HypothesisId::new("H2").unwrap(), HypothesisId::new("H3").unwrap()],
            fixture_queries(),
            true,
        )
        .unwrap();

        assert_eq!(ids(&summary.disconfirmation_query_ids), vec!["Q2"]);
        assert_eq!(ids(&summary.eligible_disconfirmation_query_ids), vec!["Q2"]);
        assert_eq!(ids(&summary.alternative_query_ids), vec!["Q3", "Q4"]);
        assert_eq!(ids(&summary.neutral_query_ids), vec!["Q1"]);
        assert_eq!(ids(&summary.dependency_query_ids), vec!["Q4"]);
        assert_eq!(summary.blocked_queries.len(), 1);
        assert_eq!(summary.blocked_queries[0].query_id.as_str(), "Q7");
        assert_eq!(summary.blocked_queries[0].reason, BlockedQueryReasonV1::Privacy);
        assert!(summary
            .limitations
            .contains(&QueryStrategyLimitationV1::PriorSearchUnknownCoverage));
        assert_eq!(summary.authority_scope(), SearchPlanAuthorityScopeV1::ProposalOnly);
    }

    #[test]
    fn input_reordering_does_not_change_set_semantic_summary() {
        let mut reversed = fixture_queries();
        reversed.reverse();

        let a = analyze_query_strategy(
            HypothesisId::new("H1").unwrap(),
            vec![HypothesisId::new("H2").unwrap(), HypothesisId::new("H3").unwrap()],
            fixture_queries(),
            true,
        )
        .unwrap();
        let b = analyze_query_strategy(
            HypothesisId::new("H1").unwrap(),
            vec![HypothesisId::new("H3").unwrap(), HypothesisId::new("H2").unwrap()],
            reversed,
            true,
        )
        .unwrap();

        assert_eq!(a, b);
    }

    #[test]
    fn wording_clones_do_not_inflate_methodological_family_diversity() {
        let base = analyze_query_strategy(
            HypothesisId::new("H1").unwrap(),
            vec![HypothesisId::new("H2").unwrap(), HypothesisId::new("H3").unwrap()],
            fixture_queries(),
            false,
        )
        .unwrap();

        let mut expanded = fixture_queries();
        for i in 0..10 {
            expanded.push(q(
                &format!("Q6C{i}"),
                QueryFamilyV1::SupportSeeking,
                "query:level-change-confirmation-wording-clone",
                &format!("sha256:q6-clone-{i}"),
                "D1",
                &["H1"],
                QueryPolicyDispositionV1::ProposalOnly,
                Some("DG1"),
            ));
        }

        let expanded = analyze_query_strategy(
            HypothesisId::new("H1").unwrap(),
            vec![HypothesisId::new("H2").unwrap(), HypothesisId::new("H3").unwrap()],
            expanded,
            false,
        )
        .unwrap();

        assert_eq!(base.methodological_family_count, expanded.methodological_family_count);
        assert_eq!(base.eligible_semantic_candidate_count, expanded.eligible_semantic_candidate_count);
    }

    #[test]
    fn no_disconfirmation_candidate_is_a_limitation_not_support() {
        let candidates: Vec<_> = fixture_queries()
            .into_iter()
            .filter(|q| q.family != QueryFamilyV1::DisconfirmationSeeking)
            .collect();
        let summary = analyze_query_strategy(
            HypothesisId::new("H1").unwrap(),
            vec![HypothesisId::new("H2").unwrap(), HypothesisId::new("H3").unwrap()],
            candidates,
            false,
        )
        .unwrap();

        assert!(summary.disconfirmation_query_ids.is_empty());
        assert!(summary.limitations.contains(
            &QueryStrategyLimitationV1::NoAvailableDisconfirmationQueryCandidate
        ));
    }

    #[test]
    fn blocked_only_disconfirmation_is_not_safe_disconfirmation_coverage() {
        let candidates = vec![q(
            "QX",
            QueryFamilyV1::DisconfirmationSeeking,
            "query:blocked-challenger",
            "sha256:blocked-challenger",
            "D4",
            &["H1"],
            QueryPolicyDispositionV1::BlockedByPrivacy,
            None,
        )];
        let summary = analyze_query_strategy(
            HypothesisId::new("H1").unwrap(),
            vec![],
            candidates,
            false,
        )
        .unwrap();

        assert_eq!(ids(&summary.disconfirmation_query_ids), vec!["QX"]);
        assert!(summary.eligible_disconfirmation_query_ids.is_empty());
        assert!(summary.limitations.contains(
            &QueryStrategyLimitationV1::NoEligibleDisconfirmationQueryCandidate
        ));
    }

    #[test]
    fn missing_alternative_and_method_families_are_explicit() {
        let summary = analyze_query_strategy(
            HypothesisId::new("H1").unwrap(),
            vec![HypothesisId::new("H2").unwrap()],
            vec![q(
                "Q2",
                QueryFamilyV1::DisconfirmationSeeking,
                "query:disconfirm",
                "sha256:disconfirm",
                "D2",
                &["H1"],
                QueryPolicyDispositionV1::ProposalOnly,
                None,
            )],
            false,
        )
        .unwrap();

        assert!(summary
            .limitations
            .contains(&QueryStrategyLimitationV1::MissingNeutralFamily));
        assert!(summary
            .limitations
            .contains(&QueryStrategyLimitationV1::MissingDependencyLineageFamily));
        assert!(summary.limitations.iter().any(|limit| matches!(
            limit,
            QueryStrategyLimitationV1::MissingAlternativeCoverage(h) if h.as_str() == "H2"
        )));
    }

    #[test]
    fn duplicate_ids_and_commitments_reject_with_redacted_diagnostics() {
        let a = q(
            "Q1",
            QueryFamilyV1::NeutralDescriptive,
            "query:a",
            "sha256:a",
            "D1",
            &["H1"],
            QueryPolicyDispositionV1::ProposalOnly,
            None,
        );
        let mut b = a.clone();
        b.query_commitment = QueryCommitmentRef::new("sha256:b").unwrap();
        let err = analyze_query_strategy(
            HypothesisId::new("H1").unwrap(),
            vec![],
            vec![a.clone(), b],
            false,
        )
        .unwrap_err();
        assert!(!format!("{err}").contains("Q1"));
        assert!(!format!("{err:?}").contains("Q1"));

        let c = q(
            "Q2",
            QueryFamilyV1::SupportSeeking,
            "query:c",
            "sha256:a",
            "D1",
            &["H1"],
            QueryPolicyDispositionV1::ProposalOnly,
            None,
        );
        let err = analyze_query_strategy(
            HypothesisId::new("H1").unwrap(),
            vec![],
            vec![a, c],
            false,
        )
        .unwrap_err();
        assert!(!format!("{err}").contains("sha256:a"));
        assert!(!format!("{err:?}").contains("sha256:a"));
    }

    #[test]
    fn preferred_hypothesis_cannot_be_relabelled_as_required_alternative() {
        let err = analyze_query_strategy(
            HypothesisId::new("H1").unwrap(),
            vec![HypothesisId::new("H1").unwrap()],
            fixture_queries(),
            false,
        )
        .unwrap_err();
        assert!(!format!("{err}").contains("H1"));
    }

    #[test]
    fn candidate_contains_no_execution_authority() {
        let candidate = &fixture_queries()[0];
        assert_eq!(candidate.authority_scope(), SearchPlanAuthorityScopeV1::ProposalOnly);
    }
}
