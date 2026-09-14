// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-seeking canonical reasoning planner V3.
//!
//! V2 can preserve context ambiguity, but it still assumes every candidate objective is already
//! represented by a point value. This planner composes context competition with the evidence-
//! interval selector so that missing measurements become explicit requests rather than fabricated
//! neutral values.
//!
//! The planner has three legitimate outcomes:
//! - `Selected`: available evidence strictly identifies one candidate;
//! - `NeedEvidence`: missing context/objective evidence could change the decision;
//! - `Abstained`: the represented evidence is complete but still does not identify a winner.
//!
//! Evidence-request ranking is a deterministic decision-relevance heuristic, not expected value of
//! information and not a calibrated probability of usefulness.

use super::reasoning_context_competition::{
    assess_contexts, ContextAssessmentReport, ContextCompetitionError, ContextCompetitionPolicy,
    ContextHypothesis,
};
use super::reasoning_objective_core::{ObjectiveKind, ObjectiveWeights};
use super::reasoning_objective_evidence::{
    select_from_objective_evidence, CandidateObjectiveEvidence, ObjectiveEvidenceError,
    ObjectiveEvidenceSelection, ObjectiveEvidenceSelectionReport, ObjectiveEvidenceStatus,
    ObjectiveUnknownReason,
};
use crate::consciousness::context_aware_evolution::ReasoningContext;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

pub const EVIDENCE_SEEKING_PLANNER_VERSION: &str = "rq-006-evidence-seeking-v3";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvidenceRequestKind {
    /// No usable context hypothesis was supplied at all.
    ContextHypotheses {
        minimum_support: f64,
    },
    /// Context hypotheses exist, but none meets the frozen admission threshold.
    ContextSupport {
        current_top_support: f64,
        minimum_support: f64,
        candidate_contexts: Vec<ReasoningContext>,
    },
    /// One candidate objective needed for decision identification has not been measured.
    ObjectiveMeasurement {
        candidate_id: String,
        objective: ObjectiveKind,
        reason: ObjectiveUnknownReason,
        source: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceRequest {
    /// Deterministic public identity, stable for the same semantic request.
    pub request_id: String,
    pub kind: EvidenceRequestKind,
    /// Bounded decision-relevance heuristic used only to order requests.
    pub decision_relevance: f64,
    /// Existing evidence refs relevant to this request. Empty is legitimate for genuinely
    /// unmeasured objectives or a completely missing context-hypothesis set.
    pub evidence_refs: Vec<String>,
    pub rationale: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceSeekingAbstention {
    /// All represented contender objective axes are observed, but the conservative robust scores
    /// still do not strictly identify one winner. Repeating the same measurements cannot resolve
    /// the decision without adding a new discriminating objective/model.
    FullyObservedUnderdetermination { contender_ids: Vec<String> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EvidenceSeekingOutcome {
    Selected {
        source_index: usize,
        candidate_id: String,
    },
    NeedEvidence {
        contender_ids: Vec<String>,
        requests: Vec<EvidenceRequest>,
    },
    Abstained {
        reason: EvidenceSeekingAbstention,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceSeekingPlanReport {
    pub planner_version: String,
    pub context_policy: ContextCompetitionPolicy,
    pub context_assessment: Option<ContextAssessmentReport>,
    pub objective_report: Option<ObjectiveEvidenceSelectionReport>,
    pub outcome: EvidenceSeekingOutcome,
}

/// Plan a reasoning decision without inventing missing evidence.
///
/// Malformed evidence remains an error. Missing-but-well-formed evidence becomes `NeedEvidence`.
pub fn plan_with_evidence(
    hypotheses: &[ContextHypothesis],
    context_policy: ContextCompetitionPolicy,
    candidates: &[CandidateObjectiveEvidence],
) -> Result<EvidenceSeekingPlanReport, EvidenceSeekingPlannerError> {
    validate_candidate_set(candidates)?;

    let assessment = match assess_contexts(hypotheses, context_policy) {
        Ok(assessment) => assessment,
        Err(ContextCompetitionError::EmptyHypothesisSet) => {
            return Ok(EvidenceSeekingPlanReport {
                planner_version: EVIDENCE_SEEKING_PLANNER_VERSION.into(),
                context_policy,
                context_assessment: None,
                objective_report: None,
                outcome: EvidenceSeekingOutcome::NeedEvidence {
                    contender_ids: sorted_candidate_ids(candidates),
                    requests: vec![EvidenceRequest {
                        request_id: "context:hypotheses".into(),
                        kind: EvidenceRequestKind::ContextHypotheses {
                            minimum_support: context_policy.minimum_support,
                        },
                        decision_relevance: 1.0,
                        evidence_refs: Vec::new(),
                        rationale: "No evidence-backed reasoning-context hypothesis was supplied"
                            .into(),
                    }],
                },
            });
        }
        Err(ContextCompetitionError::InsufficientContextSupport {
            top_support,
            minimum_support,
        }) => {
            let mut ordered = hypotheses.to_vec();
            ordered.sort_by(|left, right| {
                right
                    .support
                    .total_cmp(&left.support)
                    .then_with(|| context_rank(left.context).cmp(&context_rank(right.context)))
            });
            let candidate_contexts = ordered.iter().map(|item| item.context).collect::<Vec<_>>();
            let mut evidence_refs = ordered
                .iter()
                .flat_map(|item| item.evidence_refs.iter().cloned())
                .collect::<Vec<_>>();
            evidence_refs.sort();
            evidence_refs.dedup();
            return Ok(EvidenceSeekingPlanReport {
                planner_version: EVIDENCE_SEEKING_PLANNER_VERSION.into(),
                context_policy,
                context_assessment: None,
                objective_report: None,
                outcome: EvidenceSeekingOutcome::NeedEvidence {
                    contender_ids: sorted_candidate_ids(candidates),
                    requests: vec![EvidenceRequest {
                        request_id: "context:support".into(),
                        kind: EvidenceRequestKind::ContextSupport {
                            current_top_support: top_support,
                            minimum_support,
                            candidate_contexts,
                        },
                        decision_relevance: 1.0,
                        evidence_refs,
                        rationale: format!(
                            "Top context support {top_support:.6} is below the frozen admission threshold {minimum_support:.6}"
                        ),
                    }],
                },
            });
        }
        Err(err) => return Err(EvidenceSeekingPlannerError::Context(err)),
    };

    let objective_report = select_from_objective_evidence(&assessment, candidates)?;
    let outcome = match &objective_report.outcome {
        ObjectiveEvidenceSelection::Selected {
            source_index,
            candidate_id,
        } => EvidenceSeekingOutcome::Selected {
            source_index: *source_index,
            candidate_id: candidate_id.clone(),
        },
        ObjectiveEvidenceSelection::Underdetermined {
            contender_source_indices,
            contender_ids,
        } => {
            let requests = objective_requests(
                &assessment,
                candidates,
                contender_source_indices,
            );
            if requests.is_empty() {
                EvidenceSeekingOutcome::Abstained {
                    reason: EvidenceSeekingAbstention::FullyObservedUnderdetermination {
                        contender_ids: contender_ids.clone(),
                    },
                }
            } else {
                EvidenceSeekingOutcome::NeedEvidence {
                    contender_ids: contender_ids.clone(),
                    requests,
                }
            }
        }
    };

    Ok(EvidenceSeekingPlanReport {
        planner_version: EVIDENCE_SEEKING_PLANNER_VERSION.into(),
        context_policy,
        context_assessment: Some(assessment),
        objective_report: Some(objective_report),
        outcome,
    })
}

fn validate_candidate_set(
    candidates: &[CandidateObjectiveEvidence],
) -> Result<(), EvidenceSeekingPlannerError> {
    if candidates.is_empty() {
        return Err(EvidenceSeekingPlannerError::Objective(
            ObjectiveEvidenceError::EmptyCandidateSet,
        ));
    }
    let mut seen = HashSet::with_capacity(candidates.len());
    for candidate in candidates {
        candidate.validate()?;
        if !seen.insert(candidate.candidate_id.as_str()) {
            return Err(EvidenceSeekingPlannerError::Objective(
                ObjectiveEvidenceError::DuplicateCandidateId(candidate.candidate_id.clone()),
            ));
        }
    }
    Ok(())
}

fn sorted_candidate_ids(candidates: &[CandidateObjectiveEvidence]) -> Vec<String> {
    let mut ids = candidates
        .iter()
        .map(|candidate| candidate.candidate_id.clone())
        .collect::<Vec<_>>();
    ids.sort();
    ids
}

fn objective_requests(
    assessment: &ContextAssessmentReport,
    candidates: &[CandidateObjectiveEvidence],
    contender_source_indices: &[usize],
) -> Vec<EvidenceRequest> {
    let mut requests = Vec::new();
    for source_index in contender_source_indices {
        let Some(candidate) = candidates.get(*source_index) else {
            continue;
        };
        push_unknown_axis_request(
            &mut requests,
            assessment,
            candidate,
            ObjectiveKind::IntegrationProxy,
            &candidate.integration_proxy,
        );
        push_unknown_axis_request(
            &mut requests,
            assessment,
            candidate,
            ObjectiveKind::HarmonicAlignment,
            &candidate.harmonic_alignment,
        );
        push_unknown_axis_request(
            &mut requests,
            assessment,
            candidate,
            ObjectiveKind::EpistemicGrounding,
            &candidate.epistemic_grounding,
        );
    }

    requests.sort_by(|left, right| {
        right
            .decision_relevance
            .total_cmp(&left.decision_relevance)
            .then_with(|| left.request_id.cmp(&right.request_id))
    });
    requests
}

fn push_unknown_axis_request(
    requests: &mut Vec<EvidenceRequest>,
    assessment: &ContextAssessmentReport,
    candidate: &CandidateObjectiveEvidence,
    objective: ObjectiveKind,
    evidence: &super::reasoning_objective_evidence::ObjectiveEvidence,
) {
    let ObjectiveEvidenceStatus::Unknown { reason } = &evidence.status else {
        return;
    };
    let decision_relevance = assessment
        .active_contexts
        .iter()
        .map(|context| objective_weight(ObjectiveWeights::for_context(*context), objective))
        .fold(0.0_f64, f64::max);
    requests.push(EvidenceRequest {
        request_id: format!(
            "objective:{}:{}",
            candidate.candidate_id,
            objective_slug(objective)
        ),
        kind: EvidenceRequestKind::ObjectiveMeasurement {
            candidate_id: candidate.candidate_id.clone(),
            objective,
            reason: *reason,
            source: evidence.source.clone(),
        },
        decision_relevance,
        evidence_refs: evidence.evidence_refs.clone(),
        rationale: format!(
            "Candidate `{}` has no observed {} value; this axis carries up to {:.3} policy weight across the active contexts",
            candidate.candidate_id,
            objective.label(),
            decision_relevance
        ),
    });
}

const fn objective_weight(weights: ObjectiveWeights, objective: ObjectiveKind) -> f64 {
    match objective {
        ObjectiveKind::IntegrationProxy => weights.integration_proxy,
        ObjectiveKind::HarmonicAlignment => weights.harmonic_alignment,
        ObjectiveKind::EpistemicGrounding => weights.epistemic_grounding,
    }
}

const fn objective_slug(objective: ObjectiveKind) -> &'static str {
    match objective {
        ObjectiveKind::IntegrationProxy => "integration",
        ObjectiveKind::HarmonicAlignment => "harmonic",
        ObjectiveKind::EpistemicGrounding => "epistemic",
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

#[derive(Debug, Clone, PartialEq)]
pub enum EvidenceSeekingPlannerError {
    Context(ContextCompetitionError),
    Objective(ObjectiveEvidenceError),
}

impl fmt::Display for EvidenceSeekingPlannerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Context(err) => write!(f, "context evidence is malformed: {err}"),
            Self::Objective(err) => write!(f, "objective evidence is malformed: {err}"),
        }
    }
}

impl std::error::Error for EvidenceSeekingPlannerError {}

impl From<ObjectiveEvidenceError> for EvidenceSeekingPlannerError {
    fn from(value: ObjectiveEvidenceError) -> Self {
        Self::Objective(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::reasoning_objective_evidence::{ObjectiveEvidence, ObjectiveUnknownReason};

    fn hypothesis(context: ReasoningContext, support: f64, id: &str) -> ContextHypothesis {
        ContextHypothesis {
            context,
            support,
            source: "fixture-context-adapter".into(),
            evidence_refs: vec![id.into()],
        }
    }

    fn observed(value: f64, id: &str) -> ObjectiveEvidence {
        ObjectiveEvidence::observed("fixture-objective-adapter", vec![id.into()], value).unwrap()
    }

    fn unknown(reason: ObjectiveUnknownReason) -> ObjectiveEvidence {
        ObjectiveEvidence::unknown("fixture-objective-adapter", Vec::new(), reason).unwrap()
    }

    fn candidate(
        id: &str,
        integration: ObjectiveEvidence,
        harmonic: ObjectiveEvidence,
        epistemic: ObjectiveEvidence,
    ) -> CandidateObjectiveEvidence {
        CandidateObjectiveEvidence {
            candidate_id: id.into(),
            candidate_label: id.into(),
            integration_proxy: integration,
            harmonic_alignment: harmonic,
            epistemic_grounding: epistemic,
        }
    }

    #[test]
    fn missing_context_hypotheses_requests_context_evidence() {
        let report = plan_with_evidence(
            &[],
            ContextCompetitionPolicy::development_v1(),
            &[candidate(
                "a",
                observed(0.8, "a-i"),
                observed(0.8, "a-h"),
                observed(0.8, "a-e"),
            )],
        )
        .unwrap();
        let EvidenceSeekingOutcome::NeedEvidence { requests, .. } = report.outcome else {
            panic!("expected NeedEvidence");
        };
        assert_eq!(requests.len(), 1);
        assert!(matches!(
            &requests[0].kind,
            EvidenceRequestKind::ContextHypotheses { .. }
        ));
    }

    #[test]
    fn weak_context_support_requests_better_context_evidence() {
        let report = plan_with_evidence(
            &[hypothesis(ReasoningContext::GeneralReasoning, 0.5, "query")],
            ContextCompetitionPolicy::development_v1(),
            &[candidate(
                "a",
                observed(0.8, "a-i"),
                observed(0.8, "a-h"),
                observed(0.8, "a-e"),
            )],
        )
        .unwrap();
        let EvidenceSeekingOutcome::NeedEvidence { requests, .. } = report.outcome else {
            panic!("expected NeedEvidence");
        };
        assert!(matches!(
            &requests[0].kind,
            EvidenceRequestKind::ContextSupport { .. }
        ));
    }

    #[test]
    fn strict_observed_winner_is_selected() {
        let report = plan_with_evidence(
            &[hypothesis(ReasoningContext::GeneralReasoning, 0.9, "query")],
            ContextCompetitionPolicy::development_v1(),
            &[
                candidate(
                    "strong",
                    observed(0.9, "s-i"),
                    observed(0.9, "s-h"),
                    observed(0.9, "s-e"),
                ),
                candidate(
                    "weak",
                    observed(0.1, "w-i"),
                    observed(0.1, "w-h"),
                    observed(0.1, "w-e"),
                ),
            ],
        )
        .unwrap();
        assert!(matches!(
            report.outcome,
            EvidenceSeekingOutcome::Selected { ref candidate_id, .. } if candidate_id == "strong"
        ));
    }

    #[test]
    fn missing_objectives_become_ranked_evidence_requests() {
        let report = plan_with_evidence(
            &[hypothesis(
                ReasoningContext::TechnicalImplementation,
                0.9,
                "query",
            )],
            ContextCompetitionPolicy::development_v1(),
            &[
                candidate(
                    "a",
                    observed(0.7, "a-i"),
                    unknown(ObjectiveUnknownReason::NotMeasured),
                    unknown(ObjectiveUnknownReason::NotMeasured),
                ),
                candidate(
                    "b",
                    observed(0.7, "b-i"),
                    unknown(ObjectiveUnknownReason::NotMeasured),
                    unknown(ObjectiveUnknownReason::NotMeasured),
                ),
            ],
        )
        .unwrap();
        let EvidenceSeekingOutcome::NeedEvidence { requests, .. } = report.outcome else {
            panic!("expected NeedEvidence");
        };
        assert_eq!(requests.len(), 4);
        assert_eq!(requests[0].decision_relevance, 0.60);
        assert!(matches!(
            &requests[0].kind,
            EvidenceRequestKind::ObjectiveMeasurement {
                objective: ObjectiveKind::EpistemicGrounding,
                ..
            }
        ));
    }

    #[test]
    fn complete_exact_tie_abstains_instead_of_requesting_same_measurements() {
        let report = plan_with_evidence(
            &[hypothesis(ReasoningContext::GeneralReasoning, 0.9, "query")],
            ContextCompetitionPolicy::development_v1(),
            &[
                candidate(
                    "a",
                    observed(0.5, "a-i"),
                    observed(0.5, "a-h"),
                    observed(0.5, "a-e"),
                ),
                candidate(
                    "b",
                    observed(0.5, "b-i"),
                    observed(0.5, "b-h"),
                    observed(0.5, "b-e"),
                ),
            ],
        )
        .unwrap();
        assert!(matches!(
            report.outcome,
            EvidenceSeekingOutcome::Abstained {
                reason: EvidenceSeekingAbstention::FullyObservedUnderdetermination { .. }
            }
        ));
    }

    #[test]
    fn malformed_objective_evidence_still_fails_closed() {
        let invalid = CandidateObjectiveEvidence {
            candidate_id: "a".into(),
            candidate_label: "a".into(),
            integration_proxy: ObjectiveEvidence {
                source: "fixture".into(),
                evidence_refs: Vec::new(),
                status: ObjectiveEvidenceStatus::Observed { value: 0.5 },
            },
            harmonic_alignment: unknown(ObjectiveUnknownReason::NotMeasured),
            epistemic_grounding: unknown(ObjectiveUnknownReason::NotMeasured),
        };
        assert!(matches!(
            plan_with_evidence(
                &[hypothesis(ReasoningContext::GeneralReasoning, 0.9, "query")],
                ContextCompetitionPolicy::development_v1(),
                &[invalid],
            ),
            Err(EvidenceSeekingPlannerError::Objective(
                ObjectiveEvidenceError::ObservedWithoutEvidence
            ))
        ));
    }
}
