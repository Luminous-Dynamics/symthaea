// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-backed competition between plausible reasoning contexts.
//!
//! Context inference is intentionally adapter-owned. This module does not inspect natural-language
//! queries or invent semantic labels from keywords. It accepts explicit, evidence-referenced
//! hypotheses, validates them, preserves ambiguity, and selects candidates conservatively across
//! every context that remains plausible under a frozen policy.
//!
//! When multiple contexts are near-leading, candidate selection uses maximin utility across their
//! objective policies. This avoids silently collapsing a mixed safety/technical/scientific problem
//! to one label merely because one support score is slightly larger.

use super::reasoning_objective_core::{ObjectiveCoreError, ObjectiveVector, ObjectiveWeights};
use crate::consciousness::context_aware_evolution::ReasoningContext;
use crate::consciousness::primitive_evolution::CandidatePrimitive;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

pub const CONTEXT_COMPETITION_VERSION: &str = "rq-006-context-competition-v1";
const POLICY_TOLERANCE: f64 = 1.0e-12;

/// One adapter-produced context hypothesis.
///
/// `support` is a bounded support measurement, not a calibrated probability that the context is
/// "true". Multiple contexts may simultaneously have high support.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextHypothesis {
    pub context: ReasoningContext,
    pub support: f64,
    /// Human/machine-readable adapter identity. This is provenance, not authority.
    pub source: String,
    /// Evidence identifiers from the episode evidence set that support this hypothesis.
    pub evidence_refs: Vec<String>,
}

/// Frozen policy for deciding which context hypotheses remain plausible.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ContextCompetitionPolicy {
    /// A context below this support is not admitted into the active set.
    pub minimum_support: f64,
    /// Contexts within this absolute support distance of the leader remain jointly plausible.
    pub ambiguity_band: f64,
}

impl ContextCompetitionPolicy {
    pub fn try_new(minimum_support: f64, ambiguity_band: f64) -> Result<Self, ContextCompetitionError> {
        validate_unit("minimum_support", minimum_support)?;
        validate_unit("ambiguity_band", ambiguity_band)?;
        Ok(Self {
            minimum_support,
            ambiguity_band,
        })
    }

    /// Frozen public-development policy. These thresholds are policy parameters, not scientific
    /// constants, and must be qualified before gaining control authority.
    pub fn development_v1() -> Self {
        Self::try_new(0.55, 0.10).expect("built-in context competition policy must remain valid")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContextResolution {
    Resolved,
    Ambiguous,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextAssessmentReport {
    pub evaluator_version: String,
    pub policy: ContextCompetitionPolicy,
    /// Deterministically sorted by descending support and then canonical context rank.
    pub hypotheses: Vec<ContextHypothesis>,
    pub primary_context: ReasoningContext,
    pub top_support: f64,
    pub second_support: Option<f64>,
    pub margin: Option<f64>,
    pub active_contexts: Vec<ReasoningContext>,
    pub resolution: ContextResolution,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerContextCandidateScore {
    pub context: ReasoningContext,
    pub weights: ObjectiveWeights,
    pub weighted_score: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobustCandidateEvaluation {
    pub source_index: usize,
    pub candidate_name: String,
    pub vector: ObjectiveVector,
    pub context_scores: Vec<PerContextCandidateScore>,
    /// Minimum score across every active plausible context.
    pub worst_case_score: f64,
    /// Unweighted mean across active contexts, used only as a deterministic secondary criterion.
    pub mean_score: f64,
    pub pareto_optimal: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RobustContextSelectionReport {
    pub evaluator_version: String,
    pub assessment: ContextAssessmentReport,
    pub evaluations: Vec<RobustCandidateEvaluation>,
    pub selected_source_index: usize,
    pub selected_candidate_name: String,
    pub selected_vector: ObjectiveVector,
    pub selected_worst_case_score: f64,
    pub selected_mean_score: f64,
}

/// Validate and deterministically order context hypotheses.
pub fn assess_contexts(
    hypotheses: &[ContextHypothesis],
    policy: ContextCompetitionPolicy,
) -> Result<ContextAssessmentReport, ContextCompetitionError> {
    validate_unit("minimum_support", policy.minimum_support)?;
    validate_unit("ambiguity_band", policy.ambiguity_band)?;
    if hypotheses.is_empty() {
        return Err(ContextCompetitionError::EmptyHypothesisSet);
    }

    let mut seen_contexts = HashSet::with_capacity(hypotheses.len());
    let mut validated = Vec::with_capacity(hypotheses.len());
    for hypothesis in hypotheses {
        validate_hypothesis(hypothesis)?;
        if !seen_contexts.insert(hypothesis.context) {
            return Err(ContextCompetitionError::DuplicateContext(hypothesis.context));
        }
        validated.push(hypothesis.clone());
    }

    validated.sort_by(|left, right| {
        right
            .support
            .total_cmp(&left.support)
            .then_with(|| context_rank(left.context).cmp(&context_rank(right.context)))
    });

    let top_support = validated[0].support;
    if top_support + POLICY_TOLERANCE < policy.minimum_support {
        return Err(ContextCompetitionError::InsufficientContextSupport {
            top_support,
            minimum_support: policy.minimum_support,
        });
    }

    let primary_context = validated[0].context;
    let second_support = validated.get(1).map(|hypothesis| hypothesis.support);
    let margin = second_support.map(|second| top_support - second);

    let mut active_contexts = validated
        .iter()
        .filter(|hypothesis| {
            hypothesis.support + POLICY_TOLERANCE >= policy.minimum_support
                && top_support - hypothesis.support <= policy.ambiguity_band + POLICY_TOLERANCE
        })
        .map(|hypothesis| hypothesis.context)
        .collect::<Vec<_>>();
    active_contexts.sort_by_key(|context| context_rank(*context));

    if active_contexts.is_empty() {
        return Err(ContextCompetitionError::InsufficientContextSupport {
            top_support,
            minimum_support: policy.minimum_support,
        });
    }

    let resolution = if active_contexts.len() == 1 {
        ContextResolution::Resolved
    } else {
        ContextResolution::Ambiguous
    };

    Ok(ContextAssessmentReport {
        evaluator_version: CONTEXT_COMPETITION_VERSION.into(),
        policy,
        hypotheses: validated,
        primary_context,
        top_support,
        second_support,
        margin,
        active_contexts,
        resolution,
    })
}

/// Select a candidate robustly across every context retained by `assess_contexts`.
///
/// Pareto-dominated candidates are never selected. Among Pareto-optimal candidates, the primary
/// criterion is the worst weighted objective score across active contexts (maximin). Mean score is
/// only a secondary tie-break, followed by stable candidate metadata and source index.
pub fn select_candidate_robustly(
    assessment: ContextAssessmentReport,
    candidates: &[CandidatePrimitive],
) -> Result<RobustContextSelectionReport, ContextCompetitionError> {
    if candidates.is_empty() {
        return Err(ContextCompetitionError::EmptyCandidateSet);
    }
    if assessment.active_contexts.is_empty() {
        return Err(ContextCompetitionError::EmptyActiveContextSet);
    }

    let mut evaluations = candidates
        .iter()
        .enumerate()
        .map(|(source_index, candidate)| {
            let vector = ObjectiveVector::from_candidate(candidate)?;
            let context_scores = assessment
                .active_contexts
                .iter()
                .map(|context| {
                    let weights = ObjectiveWeights::for_context(*context);
                    PerContextCandidateScore {
                        context: *context,
                        weights,
                        weighted_score: weights.score(vector),
                    }
                })
                .collect::<Vec<_>>();
            let worst_case_score = context_scores
                .iter()
                .map(|score| score.weighted_score)
                .fold(f64::INFINITY, f64::min);
            let mean_score = context_scores
                .iter()
                .map(|score| score.weighted_score)
                .sum::<f64>()
                / context_scores.len() as f64;
            Ok(RobustCandidateEvaluation {
                source_index,
                candidate_name: candidate.name.clone(),
                vector,
                context_scores,
                worst_case_score,
                mean_score,
                pareto_optimal: false,
            })
        })
        .collect::<Result<Vec<_>, ObjectiveCoreError>>()?;

    for index in 0..evaluations.len() {
        let vector = evaluations[index].vector;
        evaluations[index].pareto_optimal = !evaluations
            .iter()
            .enumerate()
            .any(|(other_index, other)| other_index != index && other.vector.dominates(vector));
    }

    let mut frontier = evaluations
        .iter()
        .enumerate()
        .filter_map(|(index, evaluation)| evaluation.pareto_optimal.then_some(index))
        .collect::<Vec<_>>();
    if frontier.is_empty() {
        return Err(ContextCompetitionError::EmptyParetoFrontier);
    }

    frontier.sort_by(|left, right| {
        let a = &evaluations[*left];
        let b = &evaluations[*right];
        b.worst_case_score
            .total_cmp(&a.worst_case_score)
            .then_with(|| b.mean_score.total_cmp(&a.mean_score))
            .then_with(|| candidates[*left].name.cmp(&candidates[*right].name))
            .then_with(|| candidates[*left].definition.cmp(&candidates[*right].definition))
            .then_with(|| left.cmp(right))
    });

    let winner = evaluations[frontier[0]].clone();
    Ok(RobustContextSelectionReport {
        evaluator_version: CONTEXT_COMPETITION_VERSION.into(),
        assessment,
        evaluations,
        selected_source_index: winner.source_index,
        selected_candidate_name: winner.candidate_name,
        selected_vector: winner.vector,
        selected_worst_case_score: winner.worst_case_score,
        selected_mean_score: winner.mean_score,
    })
}

pub fn assess_and_select(
    hypotheses: &[ContextHypothesis],
    policy: ContextCompetitionPolicy,
    candidates: &[CandidatePrimitive],
) -> Result<RobustContextSelectionReport, ContextCompetitionError> {
    let assessment = assess_contexts(hypotheses, policy)?;
    select_candidate_robustly(assessment, candidates)
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContextCompetitionError {
    EmptyHypothesisSet,
    EmptyCandidateSet,
    EmptyActiveContextSet,
    EmptyParetoFrontier,
    EmptyField(&'static str),
    EmptyEvidenceRefs(ReasoningContext),
    EmptyEvidenceRef(ReasoningContext),
    DuplicateEvidenceRef {
        context: ReasoningContext,
        evidence_ref: String,
    },
    DuplicateContext(ReasoningContext),
    InvalidUnitValue {
        field: &'static str,
        value: f64,
    },
    InsufficientContextSupport {
        top_support: f64,
        minimum_support: f64,
    },
    Objective(ObjectiveCoreError),
}

impl fmt::Display for ContextCompetitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyHypothesisSet => write!(f, "context competition requires at least one hypothesis"),
            Self::EmptyCandidateSet => write!(f, "context competition requires at least one candidate"),
            Self::EmptyActiveContextSet => write!(f, "context assessment contains no active contexts"),
            Self::EmptyParetoFrontier => write!(f, "context competition produced an empty Pareto frontier"),
            Self::EmptyField(field) => write!(f, "required field `{field}` is empty"),
            Self::EmptyEvidenceRefs(context) => write!(
                f,
                "context hypothesis {:?} must reference at least one evidence item",
                context
            ),
            Self::EmptyEvidenceRef(context) => write!(
                f,
                "context hypothesis {:?} contains an empty evidence reference",
                context
            ),
            Self::DuplicateEvidenceRef {
                context,
                evidence_ref,
            } => write!(
                f,
                "context hypothesis {:?} repeats evidence reference `{evidence_ref}`",
                context
            ),
            Self::DuplicateContext(context) => {
                write!(f, "context {:?} appears more than once in the hypothesis set", context)
            }
            Self::InvalidUnitValue { field, value } => {
                write!(f, "`{field}` must be finite and within [0, 1], got {value}")
            }
            Self::InsufficientContextSupport {
                top_support,
                minimum_support,
            } => write!(
                f,
                "top context support {top_support} is below required minimum {minimum_support}"
            ),
            Self::Objective(err) => write!(f, "objective admission failed: {err}"),
        }
    }
}

impl std::error::Error for ContextCompetitionError {}

impl From<ObjectiveCoreError> for ContextCompetitionError {
    fn from(value: ObjectiveCoreError) -> Self {
        Self::Objective(value)
    }
}

fn validate_hypothesis(hypothesis: &ContextHypothesis) -> Result<(), ContextCompetitionError> {
    validate_unit("context.support", hypothesis.support)?;
    if hypothesis.source.trim().is_empty() {
        return Err(ContextCompetitionError::EmptyField("context.source"));
    }
    if hypothesis.evidence_refs.is_empty() {
        return Err(ContextCompetitionError::EmptyEvidenceRefs(hypothesis.context));
    }
    let mut seen = HashSet::with_capacity(hypothesis.evidence_refs.len());
    for evidence_ref in &hypothesis.evidence_refs {
        if evidence_ref.trim().is_empty() {
            return Err(ContextCompetitionError::EmptyEvidenceRef(hypothesis.context));
        }
        if !seen.insert(evidence_ref.as_str()) {
            return Err(ContextCompetitionError::DuplicateEvidenceRef {
                context: hypothesis.context,
                evidence_ref: evidence_ref.clone(),
            });
        }
    }
    Ok(())
}

fn validate_unit(field: &'static str, value: f64) -> Result<(), ContextCompetitionError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ContextCompetitionError::InvalidUnitValue { field, value })
    }
}

const fn context_rank(context: ReasoningContext) -> u8 {
    match context {
        ReasoningContext::CriticalSafety => 0,
        ReasoningContext::ScientificReasoning => 1,
        ReasoningContext::TechnicalImplementation => 2,
        ReasoningContext::Learning => 3,
        ReasoningContext::SocialInteraction => 4,
        ReasoningContext::PhilosophicalInquiry => 5,
        ReasoningContext::CreativeExploration => 6,
        ReasoningContext::GeneralReasoning => 7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::epistemic_tiers::EpistemicCoordinate;
    use crate::hdc::BinaryHV;
    use symthaea_core::hdc::primitive_system::PrimitiveTier;

    fn hypothesis(context: ReasoningContext, support: f64, evidence: &str) -> ContextHypothesis {
        ContextHypothesis {
            context,
            support,
            source: "fixture-adapter-v1".into(),
            evidence_refs: vec![evidence.into()],
        }
    }

    fn candidate(
        name: &str,
        fitness: f64,
        harmonic_alignment: f64,
        epistemic_coordinate: EpistemicCoordinate,
    ) -> CandidatePrimitive {
        CandidatePrimitive {
            name: name.into(),
            tier: PrimitiveTier::Physical,
            definition: format!("fixture-{name}"),
            fitness,
            encoding: BinaryHV::random(name.len() as u64 + 700),
            epistemic_coordinate,
            harmonic_alignment,
        }
    }

    #[test]
    fn unique_strong_context_resolves() {
        let report = assess_contexts(
            &[
                hypothesis(ReasoningContext::TechnicalImplementation, 0.90, "query"),
                hypothesis(ReasoningContext::ScientificReasoning, 0.60, "query"),
            ],
            ContextCompetitionPolicy::development_v1(),
        )
        .unwrap();
        assert_eq!(report.resolution, ContextResolution::Resolved);
        assert_eq!(report.primary_context, ReasoningContext::TechnicalImplementation);
        assert_eq!(report.active_contexts, vec![ReasoningContext::TechnicalImplementation]);
    }

    #[test]
    fn near_leading_contexts_remain_ambiguous() {
        let report = assess_contexts(
            &[
                hypothesis(ReasoningContext::TechnicalImplementation, 0.88, "query"),
                hypothesis(ReasoningContext::CriticalSafety, 0.84, "query"),
            ],
            ContextCompetitionPolicy::development_v1(),
        )
        .unwrap();
        assert_eq!(report.resolution, ContextResolution::Ambiguous);
        assert_eq!(
            report.active_contexts,
            vec![
                ReasoningContext::CriticalSafety,
                ReasoningContext::TechnicalImplementation,
            ]
        );
    }

    #[test]
    fn ambiguous_contexts_use_maximin_selection() {
        let report = assess_and_select(
            &[
                hypothesis(ReasoningContext::TechnicalImplementation, 0.88, "query"),
                hypothesis(ReasoningContext::CriticalSafety, 0.84, "query"),
            ],
            ContextCompetitionPolicy::development_v1(),
            &[
                candidate("safety-only", 0.5, 0.90, EpistemicCoordinate::null()),
                candidate("robust", 0.5, 0.55, EpistemicCoordinate::axiom()),
            ],
        )
        .unwrap();
        assert_eq!(report.selected_candidate_name, "robust");
        assert_eq!(report.assessment.resolution, ContextResolution::Ambiguous);
        assert_eq!(report.evaluations[report.selected_source_index].vector.epistemic_grounding, 1.0);
    }

    #[test]
    fn insufficient_support_fails_closed() {
        let err = assess_contexts(
            &[hypothesis(ReasoningContext::GeneralReasoning, 0.40, "query")],
            ContextCompetitionPolicy::development_v1(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ContextCompetitionError::InsufficientContextSupport { .. }
        ));
    }

    #[test]
    fn duplicate_context_fails_closed() {
        let err = assess_contexts(
            &[
                hypothesis(ReasoningContext::Learning, 0.8, "a"),
                hypothesis(ReasoningContext::Learning, 0.7, "b"),
            ],
            ContextCompetitionPolicy::development_v1(),
        )
        .unwrap_err();
        assert!(matches!(err, ContextCompetitionError::DuplicateContext(_)));
    }

    #[test]
    fn duplicate_evidence_ref_fails_closed() {
        let err = assess_contexts(
            &[ContextHypothesis {
                context: ReasoningContext::Learning,
                support: 0.8,
                source: "fixture-adapter-v1".into(),
                evidence_refs: vec!["query".into(), "query".into()],
            }],
            ContextCompetitionPolicy::development_v1(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ContextCompetitionError::DuplicateEvidenceRef { .. }
        ));
    }

    #[test]
    fn equal_candidates_tie_break_deterministically() {
        let report = assess_and_select(
            &[hypothesis(ReasoningContext::GeneralReasoning, 0.9, "query")],
            ContextCompetitionPolicy::development_v1(),
            &[
                candidate("zeta", 0.5, 0.5, EpistemicCoordinate::null()),
                candidate("alpha", 0.5, 0.5, EpistemicCoordinate::null()),
            ],
        )
        .unwrap();
        assert_eq!(report.selected_candidate_name, "alpha");
    }
}
