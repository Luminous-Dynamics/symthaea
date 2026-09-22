// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic closed-world OSINT investigation composition.
//!
//! All inputs are synthetic and committed. This crate has no network, persistence, browser,
//! model, credential, Mycelix mutation, Xenia, OPSEC-permit, or action API.

use std::error::Error;
use std::fmt;

use symthaea_investigation::{
    ExpectedRelationV1, FrontierRef, HypothesisId, InformationProposalId,
    InvestigationAuthorityScopeV1, NextBestInformationProposalV1, ObservationCandidateId,
    PlanningBurdenV1, PlanningDispositionV1, PlanningValueV1, ProfileRef, PropositionRef,
    QueryCommitmentRef, SearchEvidenceRef, SearchPlanAuthorityScopeV1, ToolProfileRef,
};
use symthaea_investigation_analysis::{
    interpret_negative_search, ArtifactProjectionRef, CorroborationAnalyzerV1,
    DependencyGroupRef, DependencyObservationV1, DependencyProjectionStateV1,
    NegativeFindingProjectionV1, NegativeSearchInterpretationV1, NegativeSearchObservationV1,
    SearchCoverageProjectionV1,
};
use symthaea_investigation_planning::{
    plan_next_information, PlannerCandidateV1, PlanningSummaryV1,
};
use symthaea_investigation_query::{
    analyze_query_strategy, QueryCandidateId, QueryCandidateV1, QueryFamilyV1,
    QueryPolicyDispositionV1, QuerySemanticRef, QueryStrategySummaryV1,
};
use symthaea_investigation_tool_selection::{
    select_tool_profiles, CoverageClassV1, CoverageRequirementV1, CurrentnessRequirementV1,
    DisclosureSurfaceV1, SideEffectClassV1, ToolCapabilityV1, ToolCurrentnessV1,
    ToolProfileProjectionV1, ToolSelectionSummaryV1, ToolSelectionTaskV1, ToolTransformV1,
    ToolViewClassV1,
};

pub const CLOSED_WORLD_LOOP_PROFILE_V1: &str = "symthaea:closed-world-investigation-loop:v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordProjectionAuthorityV1 {
    RecordProjectionOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MethodExecutionStateV1 {
    NotExecuted,
}

#[derive(Clone, PartialEq, Eq)]
pub struct ClosedWorldInvestigationRecordV1 {
    pub profile: &'static str,
    pub frontier_ref: FrontierRef,
    pub preferred_hypothesis_id: HypothesisId,
    pub corroboration_artifact_count: usize,
    pub observed_dependency_group_count: usize,
    pub negative_search_interpretation: NegativeSearchInterpretationV1,
    pub planning: PlanningSummaryV1,
    pub query_strategy: QueryStrategySummaryV1,
    pub tool_selection: ToolSelectionSummaryV1,
    pub selected_methodology_refs: Vec<ToolProfileRef>,
    pub execution_state: MethodExecutionStateV1,
}

impl ClosedWorldInvestigationRecordV1 {
    pub fn authority_scope(&self) -> RecordProjectionAuthorityV1 {
        RecordProjectionAuthorityV1::RecordProjectionOnly
    }

    pub fn investigation_authority_scope(&self) -> InvestigationAuthorityScopeV1 {
        InvestigationAuthorityScopeV1::CandidateAnalysisOnly
    }

    pub fn search_authority_scope(&self) -> SearchPlanAuthorityScopeV1 {
        SearchPlanAuthorityScopeV1::ProposalOnly
    }
}

impl fmt::Debug for ClosedWorldInvestigationRecordV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClosedWorldInvestigationRecordV1")
            .field("profile", &self.profile)
            .field("corroboration_artifact_count", &self.corroboration_artifact_count)
            .field("observed_dependency_group_count", &self.observed_dependency_group_count)
            .field("execution_state", &self.execution_state)
            .field("authority", &self.authority_scope())
            .finish()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClosedWorldLoopStageV1 {
    DependencyAnalysis,
    NegativeSearchInterpretation,
    Planning,
    QueryMethodology,
    ToolSelection,
}

#[derive(Clone, PartialEq, Eq)]
pub struct ClosedWorldLoopErrorV1 {
    pub stage: ClosedWorldLoopStageV1,
}

impl fmt::Display for ClosedWorldLoopErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "closed-world investigation stage failed: {:?}", self.stage)
    }
}

impl fmt::Debug for ClosedWorldLoopErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl Error for ClosedWorldLoopErrorV1 {}

fn stage(stage: ClosedWorldLoopStageV1) -> ClosedWorldLoopErrorV1 {
    ClosedWorldLoopErrorV1 { stage }
}

pub fn run_synthetic_reservoir_investigation(
) -> Result<ClosedWorldInvestigationRecordV1, ClosedWorldLoopErrorV1> {
    let frontier_ref = FrontierRef::new("F2")
        .map_err(|_| stage(ClosedWorldLoopStageV1::DependencyAnalysis))?;
    let preferred_hypothesis_id = HypothesisId::new("H1")
        .map_err(|_| stage(ClosedWorldLoopStageV1::Planning))?;

    let corroboration = CorroborationAnalyzerV1::summarize(
        frontier_ref.clone(),
        ProfileRef::new("analysis:dependency:v1")
            .map_err(|_| stage(ClosedWorldLoopStageV1::DependencyAnalysis))?,
        vec![
            dependency_observation("A1", "G1")?,
            dependency_observation("A2", "G1")?,
            dependency_observation("A3", "G1")?,
        ],
    )
    .map_err(|_| stage(ClosedWorldLoopStageV1::DependencyAnalysis))?;

    let negative_search = interpret_negative_search(NegativeSearchObservationV1 {
        search_evidence_ref: SearchEvidenceRef::new("S_D2")
            .map_err(|_| stage(ClosedWorldLoopStageV1::NegativeSearchInterpretation))?,
        frontier_ref: frontier_ref.clone(),
        result_count: 0,
        coverage: SearchCoverageProjectionV1::UnknownCoverage,
        finding: NegativeFindingProjectionV1::NoMatchObservedUnderSearchProfile,
    })
    .map_err(|_| stage(ClosedWorldLoopStageV1::NegativeSearchInterpretation))?;

    let prior_unknown = matches!(
        negative_search.interpretation,
        NegativeSearchInterpretationV1::UnresolvedDueToUnknownCoverage
    );

    let planning = plan_next_information(
        preferred_hypothesis_id.clone(),
        reservoir_planning_candidates()?,
        prior_unknown,
    )
    .map_err(|_| stage(ClosedWorldLoopStageV1::Planning))?;

    let query_strategy = analyze_query_strategy(
        preferred_hypothesis_id.clone(),
        vec![
            hypothesis("H2", ClosedWorldLoopStageV1::QueryMethodology)?,
            hypothesis("H3", ClosedWorldLoopStageV1::QueryMethodology)?,
        ],
        reservoir_query_candidates()?,
        prior_unknown,
    )
    .map_err(|_| stage(ClosedWorldLoopStageV1::QueryMethodology))?;

    let tool_selection = select_tool_profiles(
        &ToolSelectionTaskV1 {
            required_capabilities: vec![ToolCapabilityV1::WebSearch],
            coverage_requirement: CoverageRequirementV1::BestEffortOrBetter,
            required_corpus_commitment: None,
            currentness_requirement: CurrentnessRequirementV1::CurrentQualified,
            allowed_views: vec![ToolViewClassV1::AnonymousPublicView],
            forbidden_views: vec![ToolViewClassV1::AuthenticatedAccountView],
            forbidden_disclosure_surfaces: vec![],
            observation_only_required: false,
            required_transforms: vec![],
            external_authorization_review_required: true,
        },
        reservoir_tool_profiles()?,
    )
    .map_err(|_| stage(ClosedWorldLoopStageV1::ToolSelection))?;

    Ok(ClosedWorldInvestigationRecordV1 {
        profile: CLOSED_WORLD_LOOP_PROFILE_V1,
        frontier_ref,
        preferred_hypothesis_id,
        corroboration_artifact_count: corroboration.asserting_artifact_count(),
        observed_dependency_group_count: corroboration.observed_dependency_group_count(),
        negative_search_interpretation: negative_search.interpretation,
        selected_methodology_refs: tool_selection.pareto_front_profiles.clone(),
        planning,
        query_strategy,
        tool_selection,
        execution_state: MethodExecutionStateV1::NotExecuted,
    })
}

fn hypothesis(
    value: &str,
    at: ClosedWorldLoopStageV1,
) -> Result<HypothesisId, ClosedWorldLoopErrorV1> {
    HypothesisId::new(value).map_err(|_| stage(at))
}

fn dependency_observation(
    artifact: &str,
    group: &str,
) -> Result<DependencyObservationV1, ClosedWorldLoopErrorV1> {
    Ok(DependencyObservationV1 {
        artifact_ref: ArtifactProjectionRef::new(format!("artifact:{artifact}"))
            .map_err(|_| stage(ClosedWorldLoopStageV1::DependencyAnalysis))?,
        state: DependencyProjectionStateV1::ObservedLineageGroup(
            DependencyGroupRef::new(format!("dependency-group:{group}"))
                .map_err(|_| stage(ClosedWorldLoopStageV1::DependencyAnalysis))?,
        ),
    })
}

#[allow(clippy::too_many_arguments)]
fn planning_candidate(
    id: &str,
    relation: ExpectedRelationV1,
    disposition: PlanningDispositionV1,
    values: [PlanningValueV1; 6],
    burdens: [PlanningBurdenV1; 3],
) -> Result<PlannerCandidateV1, ClosedWorldLoopErrorV1> {
    Ok(PlannerCandidateV1 {
        proposal: NextBestInformationProposalV1 {
            id: InformationProposalId::new(id)
                .map_err(|_| stage(ClosedWorldLoopStageV1::Planning))?,
            target_observation_id: ObservationCandidateId::new(id)
                .map_err(|_| stage(ClosedWorldLoopStageV1::Planning))?,
            discriminating_power: values[0],
            contradiction_value: values[1],
            dependency_reduction: values[2],
            currentness_gap_reduction: values[3],
            coverage_gap_reduction: values[4],
            expected_reproducibility: values[5],
            expected_collection_cost: burdens[0],
            privacy_sensitivity: burdens[1],
            opsec_disclosure_cost: burdens[2],
            disposition,
            rationale_ref: PropositionRef::new(format!("rationale:{id}"))
                .map_err(|_| stage(ClosedWorldLoopStageV1::Planning))?,
        },
        relation_to_preferred_hypothesis: relation,
    })
}

fn reservoir_planning_candidates() -> Result<Vec<PlannerCandidateV1>, ClosedWorldLoopErrorV1> {
    use PlanningBurdenV1 as B;
    use PlanningValueV1 as V;

    Ok(vec![
        planning_candidate(
            "D1",
            ExpectedRelationV1::StronglyDiscriminatesAgainst,
            PlanningDispositionV1::NeedsExternalAuthorization,
            [V::VeryHigh, V::VeryHigh, V::Low, V::High, V::High, V::VeryHigh],
            [B::Medium, B::Low, B::Low],
        )?,
        planning_candidate(
            "D2",
            ExpectedRelationV1::StronglyDiscriminatesAgainst,
            PlanningDispositionV1::NeedsExternalAuthorization,
            [V::VeryHigh, V::VeryHigh, V::Low, V::Medium, V::Medium, V::VeryHigh],
            [B::Low, B::Low, B::Low],
        )?,
        planning_candidate(
            "D3",
            ExpectedRelationV1::NeutralUnderHypothesis,
            PlanningDispositionV1::NeedsExternalAuthorization,
            [V::Medium, V::Low, V::VeryHigh, V::Low, V::Medium, V::VeryHigh],
            [B::Low, B::Low, B::Low],
        )?,
        planning_candidate(
            "D4",
            ExpectedRelationV1::StronglyDiscriminatesAgainst,
            PlanningDispositionV1::BlockedByPrivacy,
            [V::VeryHigh, V::VeryHigh, V::Medium, V::VeryHigh, V::High, V::High],
            [B::Medium, B::VeryHigh, B::High],
        )?,
        planning_candidate(
            "D5",
            ExpectedRelationV1::WeaklyDiscriminatesFor,
            PlanningDispositionV1::NeedsExternalAuthorization,
            [V::Low, V::None, V::None, V::Low, V::None, V::Medium],
            [B::Medium, B::Low, B::Low],
        )?,
    ])
}

fn query_candidate(
    id: &str,
    family: QueryFamilyV1,
    semantic_ref: &str,
    commitment: &str,
    observation: &str,
    hypotheses: &[&str],
    disposition: QueryPolicyDispositionV1,
) -> Result<QueryCandidateV1, ClosedWorldLoopErrorV1> {
    QueryCandidateV1::new(
        QueryCandidateId::new(id).map_err(|_| stage(ClosedWorldLoopStageV1::QueryMethodology))?,
        family,
        QuerySemanticRef::new(semantic_ref)
            .map_err(|_| stage(ClosedWorldLoopStageV1::QueryMethodology))?,
        QueryCommitmentRef::new(commitment)
            .map_err(|_| stage(ClosedWorldLoopStageV1::QueryMethodology))?,
        vec![ObservationCandidateId::new(observation)
            .map_err(|_| stage(ClosedWorldLoopStageV1::QueryMethodology))?],
        hypotheses
            .iter()
            .map(|value| hypothesis(value, ClosedWorldLoopStageV1::QueryMethodology))
            .collect::<Result<Vec<_>, _>>()?,
        vec![ProfileRef::new("capability:synthetic")
            .map_err(|_| stage(ClosedWorldLoopStageV1::QueryMethodology))?],
        vec![ProfileRef::new("surface:query-terms")
            .map_err(|_| stage(ClosedWorldLoopStageV1::QueryMethodology))?],
        disposition,
        None,
    )
    .map_err(|_| stage(ClosedWorldLoopStageV1::QueryMethodology))
}

fn reservoir_query_candidates() -> Result<Vec<QueryCandidateV1>, ClosedWorldLoopErrorV1> {
    Ok(vec![
        query_candidate(
            "Q1",
            QueryFamilyV1::NeutralDescriptive,
            "query:maintenance-calibration-neutral",
            "sha256:q1-neutral-maintenance",
            "D2",
            &["H1", "H2"],
            QueryPolicyDispositionV1::ProposalOnly,
        )?,
        query_candidate(
            "Q2",
            QueryFamilyV1::DisconfirmationSeeking,
            "query:sensor-fault-disconfirm-h1",
            "sha256:q2-sensor-fault",
            "D2",
            &["H1", "H2"],
            QueryPolicyDispositionV1::ProposalOnly,
        )?,
        query_candidate(
            "Q3",
            QueryFamilyV1::AlternativeExplanation,
            "query:sensor-drift-alternative-h2",
            "sha256:q3-sensor-drift",
            "D2",
            &["H2"],
            QueryPolicyDispositionV1::ProposalOnly,
        )?,
        query_candidate(
            "Q4",
            QueryFamilyV1::DependencyLineage,
            "query:upstream-syndication-lineage-h3",
            "sha256:q4-lineage",
            "D3",
            &["H3"],
            QueryPolicyDispositionV1::ProposalOnly,
        )?,
        query_candidate(
            "Q7",
            QueryFamilyV1::SourceClassSpecific,
            "query:protected-device-personnel-record",
            "sha256:q7-protected",
            "D2",
            &["H1", "H2"],
            QueryPolicyDispositionV1::BlockedByPrivacy,
        )?,
    ])
}

fn profile_ref(value: &str) -> Result<ProfileRef, ClosedWorldLoopErrorV1> {
    ProfileRef::new(value).map_err(|_| stage(ClosedWorldLoopStageV1::ToolSelection))
}

fn tool_ref(value: &str) -> Result<ToolProfileRef, ClosedWorldLoopErrorV1> {
    ToolProfileRef::new(value).map_err(|_| stage(ClosedWorldLoopStageV1::ToolSelection))
}

fn reservoir_tool_profiles() -> Result<Vec<ToolProfileProjectionV1>, ClosedWorldLoopErrorV1> {
    Ok(vec![
        ToolProfileProjectionV1 {
            profile_ref: tool_ref("T_WEB_PUBLIC_TOPK")?,
            capabilities: vec![ToolCapabilityV1::WebSearch],
            coverage: CoverageClassV1::TopKOnly { result_cap: 10 },
            view: ToolViewClassV1::AnonymousPublicView,
            provider_ranking_ref: profile_ref("ranking:provider-ranked")?,
            transforms: vec![ToolTransformV1::ProviderIndexing, ToolTransformV1::SnippetGeneration],
            disclosure_surfaces: vec![
                DisclosureSurfaceV1::QueryTerms,
                DisclosureSurfaceV1::DnsTransportMetadata,
                DisclosureSurfaceV1::IpNetworkOrigin,
                DisclosureSurfaceV1::ProviderAnalytics,
            ],
            side_effect: SideEffectClassV1::ReadWithRemoteDisclosure,
            currentness: ToolCurrentnessV1::CurrentQualified,
            limitations: vec![
                profile_ref("limitation:ResultCap")?,
                profile_ref("limitation:ProviderRankingOpaque")?,
                profile_ref("limitation:CoverageUnknown")?,
            ],
        },
        ToolProfileProjectionV1 {
            profile_ref: tool_ref("T_WEB_AUTH_VIEW")?,
            capabilities: vec![ToolCapabilityV1::WebSearch],
            coverage: CoverageClassV1::PersonalizedView { result_cap: 10 },
            view: ToolViewClassV1::AuthenticatedAccountView,
            provider_ranking_ref: profile_ref("ranking:provider-ranked-personalized")?,
            transforms: vec![
                ToolTransformV1::ProviderIndexing,
                ToolTransformV1::SnippetGeneration,
                ToolTransformV1::PersonalizedRanking,
            ],
            disclosure_surfaces: vec![
                DisclosureSurfaceV1::QueryTerms,
                DisclosureSurfaceV1::AccountProviderIdentity,
                DisclosureSurfaceV1::DnsTransportMetadata,
                DisclosureSurfaceV1::IpNetworkOrigin,
                DisclosureSurfaceV1::StoredSearchHistory,
                DisclosureSurfaceV1::ProviderAnalytics,
            ],
            side_effect: SideEffectClassV1::ReadWithRemoteDisclosure,
            currentness: ToolCurrentnessV1::CurrentQualified,
            limitations: vec![
                profile_ref("limitation:AuthenticationRequired")?,
                profile_ref("limitation:PersonalizationPossible")?,
                profile_ref("limitation:ResultCap")?,
            ],
        },
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_investigation_planning::{BlockedReasonV1, PlanningLimitationV1};
    use symthaea_investigation_query::{BlockedQueryReasonV1, QueryStrategyLimitationV1};
    use symthaea_investigation_tool_selection::{IneligibilityReasonV1, ToolMethodDispositionV1};

    fn id_strings<T>(items: &[T], f: impl Fn(&T) -> String) -> Vec<String> {
        items.iter().map(f).collect()
    }

    #[test]
    fn closed_world_reservoir_loop_matches_frozen_theorem() {
        let record = run_synthetic_reservoir_investigation().unwrap();

        assert_eq!(record.profile, CLOSED_WORLD_LOOP_PROFILE_V1);
        assert_eq!(record.frontier_ref.as_str(), "F2");
        assert_eq!(record.preferred_hypothesis_id.as_str(), "H1");
        assert_eq!(record.corroboration_artifact_count, 3);
        assert_eq!(record.observed_dependency_group_count, 1);
        assert_eq!(
            record.negative_search_interpretation,
            NegativeSearchInterpretationV1::UnresolvedDueToUnknownCoverage
        );

        assert_eq!(
            id_strings(&record.planning.disconfirmation_candidate_ids, |id| id.as_str().to_owned()),
            vec!["D1", "D2", "D4"]
        );
        assert_eq!(
            id_strings(&record.planning.eligible_disconfirmation_candidate_ids, |id| id.as_str().to_owned()),
            vec!["D1", "D2"]
        );
        assert_eq!(
            id_strings(&record.planning.eligible_pareto_front_ids, |id| id.as_str().to_owned()),
            vec!["D1", "D2", "D3"]
        );
        assert!(record.planning.blocked_candidates.iter().any(|blocked| {
            blocked.proposal_id.as_str() == "D4" && blocked.reason == BlockedReasonV1::Privacy
        }));
        assert!(record
            .planning
            .limitations
            .contains(&PlanningLimitationV1::PriorFalsifierSearchUnknownCoverage));

        assert_eq!(
            id_strings(&record.query_strategy.disconfirmation_query_ids, |id| id.as_str().to_owned()),
            vec!["Q2"]
        );
        assert_eq!(
            id_strings(&record.query_strategy.alternative_query_ids, |id| id.as_str().to_owned()),
            vec!["Q3", "Q4"]
        );
        assert_eq!(
            id_strings(&record.query_strategy.neutral_query_ids, |id| id.as_str().to_owned()),
            vec!["Q1"]
        );
        assert_eq!(
            id_strings(&record.query_strategy.dependency_query_ids, |id| id.as_str().to_owned()),
            vec!["Q4"]
        );
        assert!(record.query_strategy.blocked_queries.iter().any(|blocked| {
            blocked.query_id.as_str() == "Q7" && blocked.reason == BlockedQueryReasonV1::Privacy
        }));
        assert!(record
            .query_strategy
            .limitations
            .contains(&QueryStrategyLimitationV1::PriorSearchUnknownCoverage));

        assert_eq!(record.tool_selection.eligible_profiles.len(), 1);
        assert_eq!(record.tool_selection.eligible_profiles[0].profile_ref.as_str(), "T_WEB_PUBLIC_TOPK");
        assert_eq!(
            record.tool_selection.eligible_profiles[0].disposition,
            ToolMethodDispositionV1::NeedsExternalAuthorization
        );
        assert!(record.tool_selection.blocked_profiles.iter().any(|blocked| {
            blocked.profile_ref.as_str() == "T_WEB_AUTH_VIEW"
                && blocked.reasons.contains(&IneligibilityReasonV1::BlockedViewClass)
        }));
        assert_eq!(
            id_strings(&record.selected_methodology_refs, |id| id.as_str().to_owned()),
            vec!["T_WEB_PUBLIC_TOPK"]
        );
        assert_eq!(record.execution_state, MethodExecutionStateV1::NotExecuted);
        assert_eq!(record.authority_scope(), RecordProjectionAuthorityV1::RecordProjectionOnly);
        assert_eq!(
            record.investigation_authority_scope(),
            InvestigationAuthorityScopeV1::CandidateAnalysisOnly
        );
        assert_eq!(record.search_authority_scope(), SearchPlanAuthorityScopeV1::ProposalOnly);
    }

    #[test]
    fn record_debug_does_not_dump_investigation_identifiers() {
        let record = run_synthetic_reservoir_investigation().unwrap();
        let debug = format!("{record:?}");
        assert!(!debug.contains("Q7"));
        assert!(!debug.contains("T_WEB_PUBLIC_TOPK"));
        assert!(!debug.contains("artifact:A1"));
    }
}
