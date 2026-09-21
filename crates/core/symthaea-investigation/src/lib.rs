// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bounded, dependency-free investigation semantics for Symthaea.
//!
//! This crate deliberately models candidate analysis and investigation planning only.
//! It has no network, persistence, browser, model-provider, Mycelix, Xenia, or action API.
//!
//! Core non-equivalences:
//!
//! ```text
//! hypothesis preferred under a profile != hypothesis true
//! no falsifier found != hypothesis confirmed
//! search plan != search attempt != collection authority
//! next-best-information proposal != permission to collect
//! candidate analysis != canonical evidence admission
//! ```

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const INVESTIGATION_PROFILE_V1: &str = "symthaea:bounded-investigation:v1";
pub const MAX_REF_BYTES: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvestigationAuthorityScopeV1 {
    CandidateAnalysisOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchPlanAuthorityScopeV1 {
    ProposalOnly,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InvestigationError {
    InvalidReference {
        role: &'static str,
        reason: &'static str,
    },
    HypothesisSetTooSmall,
    MissingInsufficientEvidenceHypothesis,
    MultipleInsufficientEvidenceHypotheses,
    DuplicateHypothesisId(String),
    TooFewDiscriminatorExpectations,
    DuplicateDiscriminatorHypothesis(String),
    UnknownHypothesis(String),
    NonDiscriminatingObservation,
    EmptySearchTargets,
    EmptySearchSources,
    ZeroSearchBound(&'static str),
    DuplicateAssumptionAssessmentId(String),
    UnexpectedInitialAssumptionSupersession,
    AssumptionSupersessionRequired,
    AssumptionSupersessionMismatch,
    EmptyHypothesisAssessments,
}

impl fmt::Display for InvestigationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidReference { role, reason } => {
                write!(f, "invalid {role} reference: {reason}")
            }
            Self::HypothesisSetTooSmall => write!(f, "hypothesis set requires at least two entries"),
            Self::MissingInsufficientEvidenceHypothesis => {
                write!(f, "hypothesis set requires one explicit insufficient-evidence alternative")
            }
            Self::MultipleInsufficientEvidenceHypotheses => {
                write!(f, "hypothesis set permits exactly one insufficient-evidence alternative")
            }
            Self::DuplicateHypothesisId(id) => write!(f, "duplicate hypothesis id: {id}"),
            Self::TooFewDiscriminatorExpectations => {
                write!(f, "a discriminating observation must compare at least two hypotheses")
            }
            Self::DuplicateDiscriminatorHypothesis(id) => {
                write!(f, "duplicate discriminator hypothesis: {id}")
            }
            Self::UnknownHypothesis(id) => write!(f, "unknown hypothesis: {id}"),
            Self::NonDiscriminatingObservation => {
                write!(f, "candidate assigns the same expected relation to every hypothesis")
            }
            Self::EmptySearchTargets => write!(f, "search plan requires at least one target discriminator"),
            Self::EmptySearchSources => write!(f, "search plan requires at least one source/tool profile"),
            Self::ZeroSearchBound(name) => write!(f, "search bound {name} must be greater than zero"),
            Self::DuplicateAssumptionAssessmentId(id) => {
                write!(f, "duplicate assumption assessment id: {id}")
            }
            Self::UnexpectedInitialAssumptionSupersession => {
                write!(f, "first assessment for an assumption cannot supersede a prior assessment")
            }
            Self::AssumptionSupersessionRequired => {
                write!(f, "later assumption assessment must explicitly supersede the latest prior assessment")
            }
            Self::AssumptionSupersessionMismatch => {
                write!(f, "assumption assessment does not supersede the latest prior assessment")
            }
            Self::EmptyHypothesisAssessments => {
                write!(f, "analysis step requires at least one hypothesis assessment")
            }
        }
    }
}

impl Error for InvestigationError {}

fn validate_ref(role: &'static str, value: &str) -> Result<(), InvestigationError> {
    if value.is_empty() {
        return Err(InvestigationError::InvalidReference {
            role,
            reason: "empty",
        });
    }
    if value.len() > MAX_REF_BYTES {
        return Err(InvestigationError::InvalidReference {
            role,
            reason: "too long",
        });
    }
    if !value.bytes().all(|b| (0x21..=0x7e).contains(&b)) {
        return Err(InvestigationError::InvalidReference {
            role,
            reason: "must contain graphic ASCII only",
        });
    }
    Ok(())
}

macro_rules! role_ref {
    ($name:ident) => {
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self, InvestigationError> {
                let value = value.into();
                validate_ref(stringify!($name), &value)?;
                Ok(Self(value))
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }
        }
    };
}

role_ref!(InvestigationId);
role_ref!(QuestionRef);
role_ref!(PurposeRef);
role_ref!(PolicyRef);
role_ref!(ProfileRef);
role_ref!(FrontierRef);
role_ref!(SubjectRef);
role_ref!(HypothesisId);
role_ref!(PropositionRef);
role_ref!(AssumptionId);
role_ref!(AssumptionAssessmentId);
role_ref!(ChallengeTriggerRef);
role_ref!(ObservationCandidateId);
role_ref!(SearchPlanId);
role_ref!(QueryCommitmentRef);
role_ref!(ToolProfileRef);
role_ref!(EvidenceRef);
role_ref!(RelationRef);
role_ref!(DependencyAssessmentRef);
role_ref!(SearchEvidenceRef);
role_ref!(TemporalAssessmentRef);
role_ref!(AnalysisStepId);
role_ref!(ExplicitUnknownId);
role_ref!(InformationProposalId);
role_ref!(DerivationRef);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HypothesisRoleV1 {
    Primary,
    Null,
    Alternative,
    InsufficientEvidence,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HypothesisCandidateV1 {
    pub id: HypothesisId,
    pub proposition_ref: PropositionRef,
    pub role: HypothesisRoleV1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HypothesisSetV1 {
    hypotheses: Vec<HypothesisCandidateV1>,
}

impl HypothesisSetV1 {
    pub fn new(hypotheses: Vec<HypothesisCandidateV1>) -> Result<Self, InvestigationError> {
        if hypotheses.len() < 2 {
            return Err(InvestigationError::HypothesisSetTooSmall);
        }

        let mut ids = BTreeSet::new();
        for hypothesis in &hypotheses {
            if !ids.insert(hypothesis.id.clone()) {
                return Err(InvestigationError::DuplicateHypothesisId(
                    hypothesis.id.as_str().to_string(),
                ));
            }
        }

        let insufficiency_count = hypotheses
            .iter()
            .filter(|h| h.role == HypothesisRoleV1::InsufficientEvidence)
            .count();

        match insufficiency_count {
            0 => return Err(InvestigationError::MissingInsufficientEvidenceHypothesis),
            1 => {}
            _ => return Err(InvestigationError::MultipleInsufficientEvidenceHypotheses),
        }

        Ok(Self { hypotheses })
    }

    pub fn hypotheses(&self) -> &[HypothesisCandidateV1] {
        &self.hypotheses
    }

    pub fn contains(&self, id: &HypothesisId) -> bool {
        self.hypotheses.iter().any(|h| &h.id == id)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvestigationManifestV1 {
    pub investigation_id: InvestigationId,
    pub question_ref: QuestionRef,
    pub purpose_ref: PurposeRef,
    pub policy_ref: PolicyRef,
    pub initial_frontier_ref: FrontierRef,
    pub subjects: Vec<SubjectRef>,
    pub hypotheses: HypothesisSetV1,
    pub resource_budget_profile_ref: ProfileRef,
    pub privacy_profile_ref: ProfileRef,
    pub opsec_profile_ref: ProfileRef,
    pub analysis_profile_ref: ProfileRef,
}

impl InvestigationManifestV1 {
    pub fn authority_scope(&self) -> InvestigationAuthorityScopeV1 {
        InvestigationAuthorityScopeV1::CandidateAnalysisOnly
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AssumptionStatusV1 {
    DeclaredWorkingAssumption,
    WeaklySupportedAssumption,
    ContestedAssumption,
    UnsupportedAssumption,
    InvalidatedWithinProfile,
    Retired,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AssumptionAssessmentV1 {
    pub assessment_id: AssumptionAssessmentId,
    pub assumption_id: AssumptionId,
    pub proposition_ref: PropositionRef,
    pub frontier_ref: FrontierRef,
    pub status: AssumptionStatusV1,
    pub supporting_evidence: Vec<EvidenceRef>,
    pub contradicting_evidence: Vec<EvidenceRef>,
    pub challenge_triggers: Vec<ChallengeTriggerRef>,
    pub supersedes: Option<AssumptionAssessmentId>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AssumptionLedgerV1 {
    assessments: Vec<AssumptionAssessmentV1>,
}

impl AssumptionLedgerV1 {
    pub fn assessments(&self) -> &[AssumptionAssessmentV1] {
        &self.assessments
    }

    pub fn append(&mut self, assessment: AssumptionAssessmentV1) -> Result<(), InvestigationError> {
        if self
            .assessments
            .iter()
            .any(|existing| existing.assessment_id == assessment.assessment_id)
        {
            return Err(InvestigationError::DuplicateAssumptionAssessmentId(
                assessment.assessment_id.as_str().to_string(),
            ));
        }

        let latest = self
            .assessments
            .iter()
            .rev()
            .find(|existing| existing.assumption_id == assessment.assumption_id);

        match latest {
            None if assessment.supersedes.is_some() => {
                return Err(InvestigationError::UnexpectedInitialAssumptionSupersession)
            }
            None => {}
            Some(_) if assessment.supersedes.is_none() => {
                return Err(InvestigationError::AssumptionSupersessionRequired)
            }
            Some(previous) if assessment.supersedes.as_ref() != Some(&previous.assessment_id) => {
                return Err(InvestigationError::AssumptionSupersessionMismatch)
            }
            Some(_) => {}
        }

        self.assessments.push(assessment);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpectedRelationV1 {
    StronglyDiscriminatesFor,
    DiscriminatesFor,
    WeaklyDiscriminatesFor,
    NeutralUnderHypothesis,
    DiscriminatesAgainst,
    StronglyDiscriminatesAgainst,
    UnknownRelation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HypothesisExpectationV1 {
    pub hypothesis_id: HypothesisId,
    pub expected_relation: ExpectedRelationV1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiscriminatingObservationCandidateV1 {
    pub id: ObservationCandidateId,
    pub observation_ref: PropositionRef,
    pub expectations: Vec<HypothesisExpectationV1>,
    pub temporal_scope_ref: Option<ProfileRef>,
    pub evidence_class_ref: ProfileRef,
    pub coverage_assumption_ref: Option<ProfileRef>,
    pub dependency_constraints: Vec<DependencyAssessmentRef>,
    pub privacy_constraint_ref: PolicyRef,
    pub derivation_ref: DerivationRef,
}

impl DiscriminatingObservationCandidateV1 {
    pub fn new(
        id: ObservationCandidateId,
        observation_ref: PropositionRef,
        expectations: Vec<HypothesisExpectationV1>,
        temporal_scope_ref: Option<ProfileRef>,
        evidence_class_ref: ProfileRef,
        coverage_assumption_ref: Option<ProfileRef>,
        dependency_constraints: Vec<DependencyAssessmentRef>,
        privacy_constraint_ref: PolicyRef,
        derivation_ref: DerivationRef,
    ) -> Result<Self, InvestigationError> {
        if expectations.len() < 2 {
            return Err(InvestigationError::TooFewDiscriminatorExpectations);
        }

        let mut seen = BTreeSet::new();
        for expectation in &expectations {
            if !seen.insert(expectation.hypothesis_id.clone()) {
                return Err(InvestigationError::DuplicateDiscriminatorHypothesis(
                    expectation.hypothesis_id.as_str().to_string(),
                ));
            }
        }

        let first = expectations[0].expected_relation;
        if expectations.iter().all(|e| e.expected_relation == first) {
            return Err(InvestigationError::NonDiscriminatingObservation);
        }

        Ok(Self {
            id,
            observation_ref,
            expectations,
            temporal_scope_ref,
            evidence_class_ref,
            coverage_assumption_ref,
            dependency_constraints,
            privacy_constraint_ref,
            derivation_ref,
        })
    }

    pub fn validate_against(&self, hypotheses: &HypothesisSetV1) -> Result<(), InvestigationError> {
        for expectation in &self.expectations {
            if !hypotheses.contains(&expectation.hypothesis_id) {
                return Err(InvestigationError::UnknownHypothesis(
                    expectation.hypothesis_id.as_str().to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchPlanLimitsV1 {
    pub max_results: Option<u32>,
    pub max_pages: Option<u32>,
}

impl SearchPlanLimitsV1 {
    pub fn new(max_results: Option<u32>, max_pages: Option<u32>) -> Result<Self, InvestigationError> {
        if max_results == Some(0) {
            return Err(InvestigationError::ZeroSearchBound("max_results"));
        }
        if max_pages == Some(0) {
            return Err(InvestigationError::ZeroSearchBound("max_pages"));
        }
        Ok(Self {
            max_results,
            max_pages,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SearchPlanCandidateV1 {
    pub id: SearchPlanId,
    pub investigation_id: InvestigationId,
    pub frontier_ref: FrontierRef,
    pub target_observations: Vec<ObservationCandidateId>,
    pub query_commitments: Vec<QueryCommitmentRef>,
    pub source_tool_profiles: Vec<ToolProfileRef>,
    pub coverage_goal_ref: ProfileRef,
    pub limits: SearchPlanLimitsV1,
    pub expected_disclosure_surfaces: Vec<ProfileRef>,
    pub purpose_ref: PurposeRef,
    pub privacy_profile_ref: ProfileRef,
    pub stop_condition_refs: Vec<ProfileRef>,
    pub derivation_ref: DerivationRef,
}

impl SearchPlanCandidateV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: SearchPlanId,
        investigation_id: InvestigationId,
        frontier_ref: FrontierRef,
        target_observations: Vec<ObservationCandidateId>,
        query_commitments: Vec<QueryCommitmentRef>,
        source_tool_profiles: Vec<ToolProfileRef>,
        coverage_goal_ref: ProfileRef,
        limits: SearchPlanLimitsV1,
        expected_disclosure_surfaces: Vec<ProfileRef>,
        purpose_ref: PurposeRef,
        privacy_profile_ref: ProfileRef,
        stop_condition_refs: Vec<ProfileRef>,
        derivation_ref: DerivationRef,
    ) -> Result<Self, InvestigationError> {
        if target_observations.is_empty() {
            return Err(InvestigationError::EmptySearchTargets);
        }
        if source_tool_profiles.is_empty() {
            return Err(InvestigationError::EmptySearchSources);
        }

        Ok(Self {
            id,
            investigation_id,
            frontier_ref,
            target_observations,
            query_commitments,
            source_tool_profiles,
            coverage_goal_ref,
            limits,
            expected_disclosure_surfaces,
            purpose_ref,
            privacy_profile_ref,
            stop_condition_refs,
            derivation_ref,
        })
    }

    pub fn authority_scope(&self) -> SearchPlanAuthorityScopeV1 {
        SearchPlanAuthorityScopeV1::ProposalOnly
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnknownKindV1 {
    SourceLineage,
    Coverage,
    Currentness,
    Identity,
    TemporalRelation,
    Dependency,
    MeasurementQuality,
    AlternativeExplanation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExplicitUnknownV1 {
    pub id: ExplicitUnknownId,
    pub kind: UnknownKindV1,
    pub subject_ref: PropositionRef,
    pub frontier_ref: FrontierRef,
    pub limitation_ref: Option<ProfileRef>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HypothesisAssessmentDispositionV1 {
    Live,
    PreferredWithinProfile,
    DisfavoredWithinProfile,
    InsufficientlyDiscriminated,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HypothesisAssessmentCandidateV1 {
    pub hypothesis_id: HypothesisId,
    pub frontier_ref: FrontierRef,
    pub disposition: HypothesisAssessmentDispositionV1,
    pub supporting_relation_refs: Vec<RelationRef>,
    pub contradicting_relation_refs: Vec<RelationRef>,
    pub contextual_relation_refs: Vec<RelationRef>,
    pub dependency_assessment_refs: Vec<DependencyAssessmentRef>,
    pub search_evidence_refs: Vec<SearchEvidenceRef>,
    pub temporal_currentness_refs: Vec<TemporalAssessmentRef>,
    pub unresolved_conflict_refs: Vec<EvidenceRef>,
    pub missing_discriminating_observations: Vec<ObservationCandidateId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlanningValueV1 {
    None,
    Low,
    Medium,
    High,
    VeryHigh,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlanningBurdenV1 {
    None,
    Low,
    Medium,
    High,
    VeryHigh,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanningDispositionV1 {
    ProposalOnly,
    NeedsExternalAuthorization,
    BlockedByPrivacy,
    BlockedByOpsec,
    BlockedByPolicy,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NextBestInformationProposalV1 {
    pub id: InformationProposalId,
    pub target_observation_id: ObservationCandidateId,
    pub discriminating_power: PlanningValueV1,
    pub dependency_reduction: PlanningValueV1,
    pub contradiction_value: PlanningValueV1,
    pub currentness_gap_reduction: PlanningValueV1,
    pub coverage_gap_reduction: PlanningValueV1,
    pub expected_collection_cost: PlanningBurdenV1,
    pub privacy_sensitivity: PlanningBurdenV1,
    pub opsec_disclosure_cost: PlanningBurdenV1,
    pub expected_reproducibility: PlanningValueV1,
    pub disposition: PlanningDispositionV1,
    pub rationale_ref: PropositionRef,
}

impl NextBestInformationProposalV1 {
    pub fn authority_scope(&self) -> SearchPlanAuthorityScopeV1 {
        SearchPlanAuthorityScopeV1::ProposalOnly
    }

    pub fn is_policy_blocked(&self) -> bool {
        matches!(
            self.disposition,
            PlanningDispositionV1::BlockedByPrivacy
                | PlanningDispositionV1::BlockedByOpsec
                | PlanningDispositionV1::BlockedByPolicy
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InvestigationStopDispositionV1 {
    SufficientlyDiscriminatedWithinProfile,
    InsufficientEvidence,
    BlockedByCoverage,
    BlockedByDependency,
    BlockedByCurrentness,
    BlockedByPrivacyPolicy,
    BlockedByOpsec,
    ResourceBudgetReached,
    NoAuthorizedNextStep,
    ConflictingEvidence,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvestigationAnalysisStepV1 {
    pub id: AnalysisStepId,
    pub frontier_ref: FrontierRef,
    pub analysis_profile_ref: ProfileRef,
    pub prior_step_ref: Option<AnalysisStepId>,
    pub hypothesis_assessments: Vec<HypothesisAssessmentCandidateV1>,
    pub explicit_unknowns: Vec<ExplicitUnknownV1>,
    pub stop_disposition: InvestigationStopDispositionV1,
}

impl InvestigationAnalysisStepV1 {
    pub fn new(
        id: AnalysisStepId,
        frontier_ref: FrontierRef,
        analysis_profile_ref: ProfileRef,
        prior_step_ref: Option<AnalysisStepId>,
        hypothesis_assessments: Vec<HypothesisAssessmentCandidateV1>,
        explicit_unknowns: Vec<ExplicitUnknownV1>,
        stop_disposition: InvestigationStopDispositionV1,
    ) -> Result<Self, InvestigationError> {
        if hypothesis_assessments.is_empty() {
            return Err(InvestigationError::EmptyHypothesisAssessments);
        }
        Ok(Self {
            id,
            frontier_ref,
            analysis_profile_ref,
            prior_step_ref,
            hypothesis_assessments,
            explicit_unknowns,
            stop_disposition,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvestigationCandidateBundleV1 {
    pub investigation_id: InvestigationId,
    pub input_frontier_ref: FrontierRef,
    pub analysis_step: InvestigationAnalysisStepV1,
    pub discriminating_observations: Vec<DiscriminatingObservationCandidateV1>,
    pub search_plans: Vec<SearchPlanCandidateV1>,
    pub next_information_proposals: Vec<NextBestInformationProposalV1>,
}

impl InvestigationCandidateBundleV1 {
    pub fn authority_scope(&self) -> InvestigationAuthorityScopeV1 {
        InvestigationAuthorityScopeV1::CandidateAnalysisOnly
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::TypeId;

    fn h(id: &str, role: HypothesisRoleV1) -> HypothesisCandidateV1 {
        HypothesisCandidateV1 {
            id: HypothesisId::new(id).unwrap(),
            proposition_ref: PropositionRef::new(format!("prop:{id}")).unwrap(),
            role,
        }
    }

    fn hypothesis_set() -> HypothesisSetV1 {
        HypothesisSetV1::new(vec![
            h("H1", HypothesisRoleV1::Primary),
            h("H2", HypothesisRoleV1::Alternative),
            h("HU", HypothesisRoleV1::InsufficientEvidence),
        ])
        .unwrap()
    }

    #[test]
    fn bounded_refs_reject_empty_space_and_oversize() {
        assert!(InvestigationId::new("").is_err());
        assert!(InvestigationId::new("has space").is_err());
        assert!(InvestigationId::new("x".repeat(MAX_REF_BYTES + 1)).is_err());
        assert!(InvestigationId::new("investigation:1").is_ok());
    }

    #[test]
    fn equal_text_in_different_roles_keeps_type_identity() {
        assert_ne!(TypeId::of::<HypothesisId>(), TypeId::of::<PropositionRef>());
        assert_ne!(TypeId::of::<SearchPlanId>(), TypeId::of::<InvestigationId>());
    }

    #[test]
    fn hypothesis_set_requires_exactly_one_insufficiency_alternative() {
        let missing = HypothesisSetV1::new(vec![
            h("H1", HypothesisRoleV1::Primary),
            h("H2", HypothesisRoleV1::Alternative),
        ]);
        assert_eq!(missing, Err(InvestigationError::MissingInsufficientEvidenceHypothesis));

        let duplicate = HypothesisSetV1::new(vec![
            h("H1", HypothesisRoleV1::Primary),
            h("HU1", HypothesisRoleV1::InsufficientEvidence),
            h("HU2", HypothesisRoleV1::InsufficientEvidence),
        ]);
        assert_eq!(
            duplicate,
            Err(InvestigationError::MultipleInsufficientEvidenceHypotheses)
        );
    }

    #[test]
    fn hypothesis_set_rejects_duplicate_ids() {
        let result = HypothesisSetV1::new(vec![
            h("H1", HypothesisRoleV1::Primary),
            h("H1", HypothesisRoleV1::Alternative),
            h("HU", HypothesisRoleV1::InsufficientEvidence),
        ]);
        assert_eq!(result, Err(InvestigationError::DuplicateHypothesisId("H1".into())));
    }

    #[test]
    fn discriminator_must_actually_discriminate() {
        let result = DiscriminatingObservationCandidateV1::new(
            ObservationCandidateId::new("D1").unwrap(),
            PropositionRef::new("prop:gauge-reading").unwrap(),
            vec![
                HypothesisExpectationV1 {
                    hypothesis_id: HypothesisId::new("H1").unwrap(),
                    expected_relation: ExpectedRelationV1::NeutralUnderHypothesis,
                },
                HypothesisExpectationV1 {
                    hypothesis_id: HypothesisId::new("H2").unwrap(),
                    expected_relation: ExpectedRelationV1::NeutralUnderHypothesis,
                },
            ],
            None,
            ProfileRef::new("evidence-class:gauge").unwrap(),
            None,
            vec![],
            PolicyRef::new("privacy:public-synthetic").unwrap(),
            DerivationRef::new("derivation:test").unwrap(),
        );
        assert_eq!(result, Err(InvestigationError::NonDiscriminatingObservation));
    }

    #[test]
    fn discriminator_rejects_unknown_hypothesis_on_validation() {
        let candidate = DiscriminatingObservationCandidateV1::new(
            ObservationCandidateId::new("D1").unwrap(),
            PropositionRef::new("prop:gauge-reading").unwrap(),
            vec![
                HypothesisExpectationV1 {
                    hypothesis_id: HypothesisId::new("H1").unwrap(),
                    expected_relation: ExpectedRelationV1::DiscriminatesFor,
                },
                HypothesisExpectationV1 {
                    hypothesis_id: HypothesisId::new("HX").unwrap(),
                    expected_relation: ExpectedRelationV1::DiscriminatesAgainst,
                },
            ],
            None,
            ProfileRef::new("evidence-class:gauge").unwrap(),
            None,
            vec![],
            PolicyRef::new("privacy:public-synthetic").unwrap(),
            DerivationRef::new("derivation:test").unwrap(),
        )
        .unwrap();

        assert_eq!(
            candidate.validate_against(&hypothesis_set()),
            Err(InvestigationError::UnknownHypothesis("HX".into()))
        );
    }

    #[test]
    fn assumption_history_is_append_only_and_explicitly_superseded() {
        let mut ledger = AssumptionLedgerV1::default();
        let first = AssumptionAssessmentV1 {
            assessment_id: AssumptionAssessmentId::new("AA1").unwrap(),
            assumption_id: AssumptionId::new("A1").unwrap(),
            proposition_ref: PropositionRef::new("prop:reports-independent").unwrap(),
            frontier_ref: FrontierRef::new("F1").unwrap(),
            status: AssumptionStatusV1::DeclaredWorkingAssumption,
            supporting_evidence: vec![],
            contradicting_evidence: vec![],
            challenge_triggers: vec![ChallengeTriggerRef::new("trigger:lineage").unwrap()],
            supersedes: None,
        };
        ledger.append(first).unwrap();

        let missing_supersession = AssumptionAssessmentV1 {
            assessment_id: AssumptionAssessmentId::new("AA2").unwrap(),
            assumption_id: AssumptionId::new("A1").unwrap(),
            proposition_ref: PropositionRef::new("prop:reports-independent").unwrap(),
            frontier_ref: FrontierRef::new("F2").unwrap(),
            status: AssumptionStatusV1::InvalidatedWithinProfile,
            supporting_evidence: vec![],
            contradicting_evidence: vec![EvidenceRef::new("evidence:dependency-edge").unwrap()],
            challenge_triggers: vec![],
            supersedes: None,
        };
        assert_eq!(
            ledger.append(missing_supersession),
            Err(InvestigationError::AssumptionSupersessionRequired)
        );

        let second = AssumptionAssessmentV1 {
            assessment_id: AssumptionAssessmentId::new("AA2").unwrap(),
            assumption_id: AssumptionId::new("A1").unwrap(),
            proposition_ref: PropositionRef::new("prop:reports-independent").unwrap(),
            frontier_ref: FrontierRef::new("F2").unwrap(),
            status: AssumptionStatusV1::InvalidatedWithinProfile,
            supporting_evidence: vec![],
            contradicting_evidence: vec![EvidenceRef::new("evidence:dependency-edge").unwrap()],
            challenge_triggers: vec![],
            supersedes: Some(AssumptionAssessmentId::new("AA1").unwrap()),
        };
        ledger.append(second).unwrap();
        assert_eq!(ledger.assessments().len(), 2);
        assert_eq!(ledger.assessments()[0].frontier_ref.as_str(), "F1");
        assert_eq!(ledger.assessments()[1].frontier_ref.as_str(), "F2");
    }

    #[test]
    fn search_plan_is_proposal_only_and_zero_bounds_fail() {
        assert_eq!(
            SearchPlanLimitsV1::new(Some(0), Some(1)),
            Err(InvestigationError::ZeroSearchBound("max_results"))
        );

        let plan = SearchPlanCandidateV1::new(
            SearchPlanId::new("S1").unwrap(),
            InvestigationId::new("I1").unwrap(),
            FrontierRef::new("F1").unwrap(),
            vec![ObservationCandidateId::new("D1").unwrap()],
            vec![QueryCommitmentRef::new("query:sha256:abc").unwrap()],
            vec![ToolProfileRef::new("tool:synthetic-search:v1").unwrap()],
            ProfileRef::new("coverage:bounded").unwrap(),
            SearchPlanLimitsV1::new(Some(10), Some(2)).unwrap(),
            vec![ProfileRef::new("disclosure:query").unwrap()],
            PurposeRef::new("purpose:synthetic-test").unwrap(),
            ProfileRef::new("privacy:synthetic").unwrap(),
            vec![ProfileRef::new("stop:page-limit").unwrap()],
            DerivationRef::new("derivation:test").unwrap(),
        )
        .unwrap();

        assert_eq!(plan.authority_scope(), SearchPlanAuthorityScopeV1::ProposalOnly);
    }

    #[test]
    fn blocked_information_proposal_remains_candidate_only() {
        let proposal = NextBestInformationProposalV1 {
            id: InformationProposalId::new("NBI1").unwrap(),
            target_observation_id: ObservationCandidateId::new("D2").unwrap(),
            discriminating_power: PlanningValueV1::VeryHigh,
            dependency_reduction: PlanningValueV1::Medium,
            contradiction_value: PlanningValueV1::High,
            currentness_gap_reduction: PlanningValueV1::High,
            coverage_gap_reduction: PlanningValueV1::High,
            expected_collection_cost: PlanningBurdenV1::Medium,
            privacy_sensitivity: PlanningBurdenV1::VeryHigh,
            opsec_disclosure_cost: PlanningBurdenV1::High,
            expected_reproducibility: PlanningValueV1::Medium,
            disposition: PlanningDispositionV1::BlockedByPrivacy,
            rationale_ref: PropositionRef::new("rationale:protected-telemetry").unwrap(),
        };

        assert!(proposal.is_policy_blocked());
        assert_eq!(
            proposal.authority_scope(),
            SearchPlanAuthorityScopeV1::ProposalOnly
        );
    }

    #[test]
    fn analysis_and_bundle_keep_candidate_authority() {
        let assessment = HypothesisAssessmentCandidateV1 {
            hypothesis_id: HypothesisId::new("H1").unwrap(),
            frontier_ref: FrontierRef::new("F2").unwrap(),
            disposition: HypothesisAssessmentDispositionV1::InsufficientlyDiscriminated,
            supporting_relation_refs: vec![RelationRef::new("relation:r1").unwrap()],
            contradicting_relation_refs: vec![RelationRef::new("relation:r2").unwrap()],
            contextual_relation_refs: vec![],
            dependency_assessment_refs: vec![DependencyAssessmentRef::new("dep:d1").unwrap()],
            search_evidence_refs: vec![SearchEvidenceRef::new("search:s1").unwrap()],
            temporal_currentness_refs: vec![],
            unresolved_conflict_refs: vec![EvidenceRef::new("conflict:c1").unwrap()],
            missing_discriminating_observations: vec![ObservationCandidateId::new("D1").unwrap()],
        };

        let step = InvestigationAnalysisStepV1::new(
            AnalysisStepId::new("STEP2").unwrap(),
            FrontierRef::new("F2").unwrap(),
            ProfileRef::new("analysis:deterministic:v1").unwrap(),
            Some(AnalysisStepId::new("STEP1").unwrap()),
            vec![assessment],
            vec![ExplicitUnknownV1 {
                id: ExplicitUnknownId::new("U1").unwrap(),
                kind: UnknownKindV1::Coverage,
                subject_ref: PropositionRef::new("prop:independent-gauge-exists").unwrap(),
                frontier_ref: FrontierRef::new("F2").unwrap(),
                limitation_ref: Some(ProfileRef::new("coverage:unknown").unwrap()),
            }],
            InvestigationStopDispositionV1::ConflictingEvidence,
        )
        .unwrap();

        let bundle = InvestigationCandidateBundleV1 {
            investigation_id: InvestigationId::new("I1").unwrap(),
            input_frontier_ref: FrontierRef::new("F2").unwrap(),
            analysis_step: step,
            discriminating_observations: vec![],
            search_plans: vec![],
            next_information_proposals: vec![],
        };

        assert_eq!(
            bundle.authority_scope(),
            InvestigationAuthorityScopeV1::CandidateAnalysisOnly
        );
    }
}
