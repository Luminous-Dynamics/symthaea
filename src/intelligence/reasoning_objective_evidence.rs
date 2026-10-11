// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-bearing objective selection with explicit unknowns and bounded measurements.
//!
//! Numeric defaults are not evidence. This module separates exact observed points, bounded
//! observed intervals, and unknown/unavailable objectives. Those bounds propagate through the
//! context-weighted robust selector without being collapsed to a midpoint. A candidate is selected
//! only when its conservative lower bound is strictly above every competitor's possible upper
//! bound. Otherwise the result is explicitly underdetermined.
//!
//! An observed interval is only a bounded measurement claim. It is not called a confidence or
//! credible interval unless the producing adapter independently establishes that stronger meaning.

use super::reasoning_context_competition::ContextAssessmentReport;
use super::reasoning_objective_core::{ObjectiveKind, ObjectiveWeights};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt;

pub const OBJECTIVE_EVIDENCE_SELECTOR_VERSION: &str = "rq-006-objective-evidence-v2";
const SELECTION_EPSILON: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectiveUnknownReason {
    NotMeasured,
    Unavailable,
    Invalidated,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ObjectiveEvidenceStatus {
    /// The producer claims an exact point value under its declared measurement semantics.
    Observed { value: f64 },
    /// The producer can bound the objective but cannot justify collapsing it to an exact point.
    ObservedInterval { lower: f64, upper: f64 },
    Unknown { reason: ObjectiveUnknownReason },
}

/// Evidence for one objective axis.
///
/// `source` identifies the measuring adapter. Observed points and intervals require at least one
/// evidence ref. Unknown values may have zero refs when the point is precisely that no measurement
/// exists.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveEvidence {
    pub source: String,
    pub evidence_refs: Vec<String>,
    pub status: ObjectiveEvidenceStatus,
}

impl ObjectiveEvidence {
    pub fn observed(
        source: impl Into<String>,
        evidence_refs: Vec<String>,
        value: f64,
    ) -> Result<Self, ObjectiveEvidenceError> {
        let evidence = Self {
            source: source.into(),
            evidence_refs,
            status: ObjectiveEvidenceStatus::Observed { value },
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub fn observed_interval(
        source: impl Into<String>,
        evidence_refs: Vec<String>,
        lower: f64,
        upper: f64,
    ) -> Result<Self, ObjectiveEvidenceError> {
        let evidence = Self {
            source: source.into(),
            evidence_refs,
            status: ObjectiveEvidenceStatus::ObservedInterval { lower, upper },
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub fn unknown(
        source: impl Into<String>,
        evidence_refs: Vec<String>,
        reason: ObjectiveUnknownReason,
    ) -> Result<Self, ObjectiveEvidenceError> {
        let evidence = Self {
            source: source.into(),
            evidence_refs,
            status: ObjectiveEvidenceStatus::Unknown { reason },
        };
        evidence.validate()?;
        Ok(evidence)
    }

    pub fn validate(&self) -> Result<(), ObjectiveEvidenceError> {
        if self.source.trim().is_empty() {
            return Err(ObjectiveEvidenceError::EmptyField("objective.source"));
        }
        let mut seen = HashSet::with_capacity(self.evidence_refs.len());
        for evidence_ref in &self.evidence_refs {
            if evidence_ref.trim().is_empty() {
                return Err(ObjectiveEvidenceError::EmptyEvidenceRef);
            }
            if !seen.insert(evidence_ref.as_str()) {
                return Err(ObjectiveEvidenceError::DuplicateEvidenceRef(
                    evidence_ref.clone(),
                ));
            }
        }
        match self.status {
            ObjectiveEvidenceStatus::Observed { value } => {
                validate_unit("objective.value", value)?;
                require_observed_evidence_refs(&self.evidence_refs)?;
            }
            ObjectiveEvidenceStatus::ObservedInterval { lower, upper } => {
                validate_unit("objective.interval.lower", lower)?;
                validate_unit("objective.interval.upper", upper)?;
                if lower > upper {
                    return Err(ObjectiveEvidenceError::InvalidIntervalBounds { lower, upper });
                }
                require_observed_evidence_refs(&self.evidence_refs)?;
            }
            ObjectiveEvidenceStatus::Unknown { .. } => {}
        }
        Ok(())
    }

    pub fn interval(&self) -> ScoreInterval {
        match self.status {
            ObjectiveEvidenceStatus::Observed { value } => ScoreInterval::point(value),
            ObjectiveEvidenceStatus::ObservedInterval { lower, upper } => {
                ScoreInterval { lower, upper }
            }
            ObjectiveEvidenceStatus::Unknown { .. } => ScoreInterval {
                lower: 0.0,
                upper: 1.0,
            },
        }
    }

    pub fn is_observed(&self) -> bool {
        matches!(
            self.status,
            ObjectiveEvidenceStatus::Observed { .. }
                | ObjectiveEvidenceStatus::ObservedInterval { .. }
        )
    }

    pub fn is_exact_point(&self) -> bool {
        matches!(self.status, ObjectiveEvidenceStatus::Observed { .. })
    }
}

fn require_observed_evidence_refs(
    evidence_refs: &[String],
) -> Result<(), ObjectiveEvidenceError> {
    if evidence_refs.is_empty() {
        Err(ObjectiveEvidenceError::ObservedWithoutEvidence)
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateObjectiveEvidence {
    pub candidate_id: String,
    pub candidate_label: String,
    pub integration_proxy: ObjectiveEvidence,
    pub harmonic_alignment: ObjectiveEvidence,
    pub epistemic_grounding: ObjectiveEvidence,
}

impl CandidateObjectiveEvidence {
    pub fn validate(&self) -> Result<(), ObjectiveEvidenceError> {
        if self.candidate_id.trim().is_empty() {
            return Err(ObjectiveEvidenceError::EmptyField("candidate_id"));
        }
        if self.candidate_label.trim().is_empty() {
            return Err(ObjectiveEvidenceError::EmptyField("candidate_label"));
        }
        self.integration_proxy.validate()?;
        self.harmonic_alignment.validate()?;
        self.epistemic_grounding.validate()?;
        Ok(())
    }

    pub fn axis(&self, kind: ObjectiveKind) -> &ObjectiveEvidence {
        match kind {
            ObjectiveKind::IntegrationProxy => &self.integration_proxy,
            ObjectiveKind::HarmonicAlignment => &self.harmonic_alignment,
            ObjectiveKind::EpistemicGrounding => &self.epistemic_grounding,
        }
    }

    pub fn observed_axes(&self) -> usize {
        [
            ObjectiveKind::IntegrationProxy,
            ObjectiveKind::HarmonicAlignment,
            ObjectiveKind::EpistemicGrounding,
        ]
        .into_iter()
        .filter(|kind| self.axis(*kind).is_observed())
        .count()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScoreInterval {
    pub lower: f64,
    pub upper: f64,
}

impl ScoreInterval {
    pub const fn point(value: f64) -> Self {
        Self {
            lower: value,
            upper: value,
        }
    }

    pub fn width(self) -> f64 {
        self.upper - self.lower
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContextualEvidenceScore {
    pub context: crate::consciousness::context_aware_evolution::ReasoningContext,
    pub weights: ObjectiveWeights,
    pub interval: ScoreInterval,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateEvidenceEvaluation {
    pub source_index: usize,
    pub candidate_id: String,
    pub candidate_label: String,
    pub observed_axes: usize,
    pub context_scores: Vec<ContextualEvidenceScore>,
    /// Conservative interval for the minimum score across all active plausible contexts.
    pub robust_interval: ScoreInterval,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ObjectiveEvidenceSelection {
    Selected {
        source_index: usize,
        candidate_id: String,
    },
    Underdetermined {
        contender_source_indices: Vec<usize>,
        contender_ids: Vec<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveEvidenceSelectionReport {
    pub evaluator_version: String,
    pub active_contexts: Vec<crate::consciousness::context_aware_evolution::ReasoningContext>,
    pub evaluations: Vec<CandidateEvidenceEvaluation>,
    pub outcome: ObjectiveEvidenceSelection,
}

/// Select only when the available objective evidence proves one candidate strictly better under
/// the conservative robust interval. Unknown axes and bounded observations therefore widen
/// uncertainty rather than becoming fabricated neutral or exact measurements.
pub fn select_from_objective_evidence(
    assessment: &ContextAssessmentReport,
    candidates: &[CandidateObjectiveEvidence],
) -> Result<ObjectiveEvidenceSelectionReport, ObjectiveEvidenceError> {
    if assessment.active_contexts.is_empty() {
        return Err(ObjectiveEvidenceError::EmptyActiveContextSet);
    }
    if candidates.is_empty() {
        return Err(ObjectiveEvidenceError::EmptyCandidateSet);
    }

    let mut context_seen = HashSet::with_capacity(assessment.active_contexts.len());
    for context in &assessment.active_contexts {
        if !context_seen.insert(*context) {
            return Err(ObjectiveEvidenceError::DuplicateActiveContext(*context));
        }
    }

    let mut candidate_seen = HashSet::with_capacity(candidates.len());
    let mut evaluations = Vec::with_capacity(candidates.len());
    for (source_index, candidate) in candidates.iter().enumerate() {
        candidate.validate()?;
        if !candidate_seen.insert(candidate.candidate_id.as_str()) {
            return Err(ObjectiveEvidenceError::DuplicateCandidateId(
                candidate.candidate_id.clone(),
            ));
        }

        let context_scores = assessment
            .active_contexts
            .iter()
            .map(|context| {
                let weights = ObjectiveWeights::for_context(*context);
                let interval = weighted_interval(candidate, weights);
                ContextualEvidenceScore {
                    context: *context,
                    weights,
                    interval,
                }
            })
            .collect::<Vec<_>>();

        let robust_interval = ScoreInterval {
            lower: context_scores
                .iter()
                .map(|score| score.interval.lower)
                .fold(f64::INFINITY, f64::min),
            upper: context_scores
                .iter()
                .map(|score| score.interval.upper)
                .fold(f64::INFINITY, f64::min),
        };

        evaluations.push(CandidateEvidenceEvaluation {
            source_index,
            candidate_id: candidate.candidate_id.clone(),
            candidate_label: candidate.candidate_label.clone(),
            observed_axes: candidate.observed_axes(),
            context_scores,
            robust_interval,
        });
    }

    let best_lower = evaluations
        .iter()
        .map(|evaluation| evaluation.robust_interval.lower)
        .fold(f64::NEG_INFINITY, f64::max);

    let mut lower_leaders = evaluations
        .iter()
        .enumerate()
        .filter_map(|(index, evaluation)| {
            ((evaluation.robust_interval.lower - best_lower).abs() <= SELECTION_EPSILON)
                .then_some(index)
        })
        .collect::<Vec<_>>();
    lower_leaders.sort_by(|left, right| {
        evaluations[*left]
            .candidate_id
            .cmp(&evaluations[*right].candidate_id)
            .then_with(|| left.cmp(right))
    });

    let leader_index = lower_leaders[0];
    let leader = &evaluations[leader_index];
    let max_competing_upper = evaluations
        .iter()
        .enumerate()
        .filter_map(|(index, evaluation)| {
            (index != leader_index).then_some(evaluation.robust_interval.upper)
        })
        .fold(f64::NEG_INFINITY, f64::max);

    let outcome = if evaluations.len() == 1
        || leader.robust_interval.lower > max_competing_upper + SELECTION_EPSILON
    {
        ObjectiveEvidenceSelection::Selected {
            source_index: leader.source_index,
            candidate_id: leader.candidate_id.clone(),
        }
    } else {
        let mut contenders = evaluations
            .iter()
            .filter(|evaluation| {
                evaluation.robust_interval.upper + SELECTION_EPSILON >= best_lower
            })
            .map(|evaluation| (evaluation.source_index, evaluation.candidate_id.clone()))
            .collect::<Vec<_>>();
        contenders.sort_by(|left, right| left.1.cmp(&right.1).then_with(|| left.0.cmp(&right.0)));
        ObjectiveEvidenceSelection::Underdetermined {
            contender_source_indices: contenders.iter().map(|(index, _)| *index).collect(),
            contender_ids: contenders.into_iter().map(|(_, id)| id).collect(),
        }
    };

    Ok(ObjectiveEvidenceSelectionReport {
        evaluator_version: OBJECTIVE_EVIDENCE_SELECTOR_VERSION.into(),
        active_contexts: assessment.active_contexts.clone(),
        evaluations,
        outcome,
    })
}

fn weighted_interval(candidate: &CandidateObjectiveEvidence, weights: ObjectiveWeights) -> ScoreInterval {
    let axes = [
        (
            weights.integration_proxy,
            candidate.integration_proxy.interval(),
        ),
        (
            weights.harmonic_alignment,
            candidate.harmonic_alignment.interval(),
        ),
        (
            weights.epistemic_grounding,
            candidate.epistemic_grounding.interval(),
        ),
    ];

    ScoreInterval {
        lower: axes
            .iter()
            .map(|(weight, interval)| weight * interval.lower)
            .sum(),
        upper: axes
            .iter()
            .map(|(weight, interval)| weight * interval.upper)
            .sum(),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ObjectiveEvidenceError {
    EmptyField(&'static str),
    EmptyEvidenceRef,
    DuplicateEvidenceRef(String),
    ObservedWithoutEvidence,
    InvalidUnitValue {
        field: &'static str,
        value: f64,
    },
    InvalidIntervalBounds {
        lower: f64,
        upper: f64,
    },
    EmptyActiveContextSet,
    EmptyCandidateSet,
    DuplicateActiveContext(crate::consciousness::context_aware_evolution::ReasoningContext),
    DuplicateCandidateId(String),
}

impl fmt::Display for ObjectiveEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "required field `{field}` is empty"),
            Self::EmptyEvidenceRef => write!(f, "objective evidence reference cannot be empty"),
            Self::DuplicateEvidenceRef(reference) => {
                write!(f, "objective evidence reference `{reference}` is duplicated")
            }
            Self::ObservedWithoutEvidence => {
                write!(f, "observed objective value requires at least one evidence reference")
            }
            Self::InvalidUnitValue { field, value } => {
                write!(f, "`{field}` must be finite and within [0, 1], got {value}")
            }
            Self::InvalidIntervalBounds { lower, upper } => write!(
                f,
                "objective interval lower bound {lower} must not exceed upper bound {upper}"
            ),
            Self::EmptyActiveContextSet => {
                write!(f, "objective evidence selection requires active contexts")
            }
            Self::EmptyCandidateSet => write!(f, "objective evidence selection requires candidates"),
            Self::DuplicateActiveContext(context) => {
                write!(f, "active context {:?} is duplicated", context)
            }
            Self::DuplicateCandidateId(candidate_id) => {
                write!(f, "candidate id `{candidate_id}` is duplicated")
            }
        }
    }
}

impl std::error::Error for ObjectiveEvidenceError {}

fn validate_unit(field: &'static str, value: f64) -> Result<(), ObjectiveEvidenceError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ObjectiveEvidenceError::InvalidUnitValue { field, value })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::reasoning_context_competition::{
        assess_contexts, ContextCompetitionPolicy, ContextHypothesis,
    };
    use crate::consciousness::context_aware_evolution::ReasoningContext;

    fn assessment(contexts: &[(ReasoningContext, f64)]) -> ContextAssessmentReport {
        assess_contexts(
            &contexts
                .iter()
                .enumerate()
                .map(|(index, (context, support))| ContextHypothesis {
                    context: *context,
                    support: *support,
                    source: "fixture-context-source".into(),
                    evidence_refs: vec![format!("context-evidence-{index}")],
                })
                .collect::<Vec<_>>(),
            ContextCompetitionPolicy::development_v1(),
        )
        .unwrap()
    }

    fn observed(value: f64, id: &str) -> ObjectiveEvidence {
        ObjectiveEvidence::observed("fixture-objective-source", vec![id.into()], value).unwrap()
    }

    fn observed_interval(lower: f64, upper: f64, id: &str) -> ObjectiveEvidence {
        ObjectiveEvidence::observed_interval(
            "fixture-objective-source",
            vec![id.into()],
            lower,
            upper,
        )
        .unwrap()
    }

    fn unknown(reason: ObjectiveUnknownReason) -> ObjectiveEvidence {
        ObjectiveEvidence::unknown("fixture-objective-source", Vec::new(), reason).unwrap()
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
    fn fully_observed_strict_winner_is_selected() {
        let report = select_from_objective_evidence(
            &assessment(&[(ReasoningContext::GeneralReasoning, 0.9)]),
            &[
                candidate(
                    "weak",
                    observed(0.2, "w-i"),
                    observed(0.2, "w-h"),
                    observed(0.2, "w-e"),
                ),
                candidate(
                    "strong",
                    observed(0.9, "s-i"),
                    observed(0.9, "s-h"),
                    observed(0.9, "s-e"),
                ),
            ],
        )
        .unwrap();
        assert!(matches!(
            report.outcome,
            ObjectiveEvidenceSelection::Selected { ref candidate_id, .. } if candidate_id == "strong"
        ));
    }

    #[test]
    fn unknown_axes_widen_interval_instead_of_becoming_neutral() {
        let report = select_from_objective_evidence(
            &assessment(&[(ReasoningContext::TechnicalImplementation, 0.9)]),
            &[
                candidate(
                    "a",
                    observed(0.8, "a-i"),
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

        assert!(matches!(
            report.outcome,
            ObjectiveEvidenceSelection::Underdetermined { .. }
        ));
        for evaluation in &report.evaluations {
            assert!(evaluation.robust_interval.width() > 0.0);
        }
    }

    #[test]
    fn separated_observed_intervals_can_identify_winner() {
        let report = select_from_objective_evidence(
            &assessment(&[(ReasoningContext::GeneralReasoning, 0.9)]),
            &[
                candidate(
                    "weak",
                    observed_interval(0.10, 0.20, "w-i"),
                    observed_interval(0.10, 0.20, "w-h"),
                    observed_interval(0.10, 0.20, "w-e"),
                ),
                candidate(
                    "strong",
                    observed_interval(0.70, 0.80, "s-i"),
                    observed_interval(0.70, 0.80, "s-h"),
                    observed_interval(0.70, 0.80, "s-e"),
                ),
            ],
        )
        .unwrap();
        assert!(matches!(
            report.outcome,
            ObjectiveEvidenceSelection::Selected { ref candidate_id, .. } if candidate_id == "strong"
        ));
    }

    #[test]
    fn overlapping_observed_intervals_remain_underdetermined() {
        let report = select_from_objective_evidence(
            &assessment(&[(ReasoningContext::GeneralReasoning, 0.9)]),
            &[
                candidate(
                    "a",
                    observed_interval(0.40, 0.70, "a-i"),
                    observed(0.5, "a-h"),
                    observed(0.5, "a-e"),
                ),
                candidate(
                    "b",
                    observed_interval(0.45, 0.65, "b-i"),
                    observed(0.5, "b-h"),
                    observed(0.5, "b-e"),
                ),
            ],
        )
        .unwrap();
        assert!(matches!(
            report.outcome,
            ObjectiveEvidenceSelection::Underdetermined { .. }
        ));
    }

    #[test]
    fn malformed_observed_interval_fails_closed() {
        assert!(matches!(
            ObjectiveEvidence::observed_interval("fixture", vec!["e".into()], 0.8, 0.2),
            Err(ObjectiveEvidenceError::InvalidIntervalBounds {
                lower: 0.8,
                upper: 0.2
            })
        ));
        assert!(matches!(
            ObjectiveEvidence::observed_interval("fixture", vec!["e".into()], -0.1, 0.2),
            Err(ObjectiveEvidenceError::InvalidUnitValue {
                field: "objective.interval.lower",
                ..
            })
        ));
    }

    #[test]
    fn exact_known_tie_is_underdetermined_not_arbitrary() {
        let report = select_from_objective_evidence(
            &assessment(&[(ReasoningContext::GeneralReasoning, 0.9)]),
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
            ObjectiveEvidenceSelection::Underdetermined { ref contender_ids, .. }
                if contender_ids == &vec!["a".to_string(), "b".to_string()]
        ));
    }

    #[test]
    fn ambiguity_is_robust_across_all_active_contexts() {
        let report = select_from_objective_evidence(
            &assessment(&[
                (ReasoningContext::CriticalSafety, 0.9),
                (ReasoningContext::TechnicalImplementation, 0.86),
            ]),
            &[
                candidate(
                    "balanced",
                    observed(0.7, "b-i"),
                    observed(0.8, "b-h"),
                    observed(0.8, "b-e"),
                ),
                candidate(
                    "technical-only",
                    observed(0.7, "t-i"),
                    observed(0.1, "t-h"),
                    observed(1.0, "t-e"),
                ),
            ],
        )
        .unwrap();
        assert!(matches!(
            report.outcome,
            ObjectiveEvidenceSelection::Selected { ref candidate_id, .. } if candidate_id == "balanced"
        ));
    }

    #[test]
    fn observed_value_without_provenance_is_rejected() {
        let err = ObjectiveEvidence::observed("fixture", Vec::new(), 0.5).unwrap_err();
        assert_eq!(err, ObjectiveEvidenceError::ObservedWithoutEvidence);
        let interval_err =
            ObjectiveEvidence::observed_interval("fixture", Vec::new(), 0.4, 0.6).unwrap_err();
        assert_eq!(interval_err, ObjectiveEvidenceError::ObservedWithoutEvidence);
    }

    #[test]
    fn duplicate_candidate_identity_fails_closed() {
        let evidence = candidate(
            "same",
            observed(0.5, "i"),
            observed(0.5, "h"),
            observed(0.5, "e"),
        );
        let err = select_from_objective_evidence(
            &assessment(&[(ReasoningContext::GeneralReasoning, 0.9)]),
            &[evidence.clone(), evidence],
        )
        .unwrap_err();
        assert!(matches!(err, ObjectiveEvidenceError::DuplicateCandidateId(_)));
    }
}
