// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical typed objective substrate for reasoning selection.
//!
//! This layer keeps measured coordinates separate from stronger interpretations:
//! primitive fitness is an integration/fitness proxy, harmonic alignment is harmonic alignment,
//! and epistemic-coordinate quality is epistemic grounding. Context policy may weight these
//! measurements, but may not rename them into stronger claims.

use crate::consciousness::context_aware_evolution::ReasoningContext;
use crate::consciousness::primitive_evolution::CandidatePrimitive;
use serde::{Deserialize, Serialize};
use std::fmt;

pub const REASONING_OBJECTIVE_CORE_VERSION: &str = "rq-006-objective-core-v1";
const WEIGHT_SUM_TOLERANCE: f64 = 1.0e-9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectiveKind {
    IntegrationProxy,
    HarmonicAlignment,
    EpistemicGrounding,
}

impl ObjectiveKind {
    pub const fn label(self) -> &'static str {
        match self {
            Self::IntegrationProxy => "integration proxy",
            Self::HarmonicAlignment => "harmonic alignment",
            Self::EpistemicGrounding => "epistemic grounding",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveVector {
    pub integration_proxy: f64,
    pub harmonic_alignment: f64,
    pub epistemic_grounding: f64,
}

impl ObjectiveVector {
    pub fn try_new(
        integration_proxy: f64,
        harmonic_alignment: f64,
        epistemic_grounding: f64,
    ) -> Result<Self, ObjectiveCoreError> {
        validate_unit("integration_proxy", integration_proxy, None)?;
        validate_unit("harmonic_alignment", harmonic_alignment, None)?;
        validate_unit("epistemic_grounding", epistemic_grounding, None)?;
        Ok(Self {
            integration_proxy,
            harmonic_alignment,
            epistemic_grounding,
        })
    }

    pub fn from_candidate(candidate: &CandidatePrimitive) -> Result<Self, ObjectiveCoreError> {
        let name = Some(candidate.name.as_str());
        validate_unit("fitness", candidate.fitness, name)?;
        validate_unit("harmonic_alignment", candidate.harmonic_alignment, name)?;
        let epistemic_grounding = candidate.epistemic_coordinate.quality_score();
        validate_unit("epistemic_grounding", epistemic_grounding, name)?;
        Ok(Self {
            integration_proxy: candidate.fitness,
            harmonic_alignment: candidate.harmonic_alignment,
            epistemic_grounding,
        })
    }

    pub const fn value(self, kind: ObjectiveKind) -> f64 {
        match kind {
            ObjectiveKind::IntegrationProxy => self.integration_proxy,
            ObjectiveKind::HarmonicAlignment => self.harmonic_alignment,
            ObjectiveKind::EpistemicGrounding => self.epistemic_grounding,
        }
    }

    pub fn dominates(self, other: Self) -> bool {
        let no_worse = self.integration_proxy >= other.integration_proxy
            && self.harmonic_alignment >= other.harmonic_alignment
            && self.epistemic_grounding >= other.epistemic_grounding;
        let strictly_better = self.integration_proxy > other.integration_proxy
            || self.harmonic_alignment > other.harmonic_alignment
            || self.epistemic_grounding > other.epistemic_grounding;
        no_worse && strictly_better
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveWeights {
    pub integration_proxy: f64,
    pub harmonic_alignment: f64,
    pub epistemic_grounding: f64,
}

impl ObjectiveWeights {
    pub fn try_new(
        integration_proxy: f64,
        harmonic_alignment: f64,
        epistemic_grounding: f64,
    ) -> Result<Self, ObjectiveCoreError> {
        validate_unit("weight.integration_proxy", integration_proxy, None)?;
        validate_unit("weight.harmonic_alignment", harmonic_alignment, None)?;
        validate_unit("weight.epistemic_grounding", epistemic_grounding, None)?;
        let sum = integration_proxy + harmonic_alignment + epistemic_grounding;
        if (sum - 1.0).abs() > WEIGHT_SUM_TOLERANCE {
            return Err(ObjectiveCoreError::InvalidWeightSum(sum));
        }
        Ok(Self {
            integration_proxy,
            harmonic_alignment,
            epistemic_grounding,
        })
    }

    pub fn for_context(context: ReasoningContext) -> Self {
        let values = match context {
            ReasoningContext::CriticalSafety => (0.10, 0.70, 0.20),
            ReasoningContext::ScientificReasoning => (0.30, 0.10, 0.60),
            ReasoningContext::CreativeExploration => (0.70, 0.15, 0.15),
            ReasoningContext::GeneralReasoning => (0.40, 0.30, 0.30),
            ReasoningContext::Learning => (0.35, 0.25, 0.40),
            ReasoningContext::SocialInteraction => (0.30, 0.45, 0.25),
            ReasoningContext::PhilosophicalInquiry => (0.45, 0.30, 0.25),
            ReasoningContext::TechnicalImplementation => (0.25, 0.15, 0.60),
        };
        Self::try_new(values.0, values.1, values.2)
            .expect("built-in reasoning objective weights must remain normalized")
    }

    pub fn score(self, vector: ObjectiveVector) -> f64 {
        self.integration_proxy * vector.integration_proxy
            + self.harmonic_alignment * vector.harmonic_alignment
            + self.epistemic_grounding * vector.epistemic_grounding
    }

    pub fn dominant_objectives(self) -> Vec<ObjectiveKind> {
        let maximum = self
            .integration_proxy
            .max(self.harmonic_alignment)
            .max(self.epistemic_grounding);
        let mut result = Vec::with_capacity(3);
        if self.integration_proxy == maximum {
            result.push(ObjectiveKind::IntegrationProxy);
        }
        if self.harmonic_alignment == maximum {
            result.push(ObjectiveKind::HarmonicAlignment);
        }
        if self.epistemic_grounding == maximum {
            result.push(ObjectiveKind::EpistemicGrounding);
        }
        result
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateObjectiveEvaluation {
    pub source_index: usize,
    pub candidate_name: String,
    pub vector: ObjectiveVector,
    pub weighted_score: f64,
    pub pareto_optimal: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveSelectionReport {
    pub evaluator_version: String,
    pub context: ReasoningContext,
    pub weights: ObjectiveWeights,
    pub dominant_objectives: Vec<ObjectiveKind>,
    pub evaluations: Vec<CandidateObjectiveEvaluation>,
    pub selected_source_index: usize,
    pub selected_candidate_name: String,
    pub selected_vector: ObjectiveVector,
    pub selected_weighted_score: f64,
}

pub fn select_candidate_by_objectives(
    context: ReasoningContext,
    candidates: &[CandidatePrimitive],
) -> Result<ObjectiveSelectionReport, ObjectiveCoreError> {
    if candidates.is_empty() {
        return Err(ObjectiveCoreError::EmptyCandidateSet);
    }

    let weights = ObjectiveWeights::for_context(context);
    let mut evaluations = candidates
        .iter()
        .enumerate()
        .map(|(source_index, candidate)| {
            let vector = ObjectiveVector::from_candidate(candidate)?;
            Ok(CandidateObjectiveEvaluation {
                source_index,
                candidate_name: candidate.name.clone(),
                vector,
                weighted_score: weights.score(vector),
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
        return Err(ObjectiveCoreError::EmptyParetoFrontier);
    }

    frontier.sort_by(|left, right| {
        let a = &evaluations[*left];
        let b = &evaluations[*right];
        b.weighted_score
            .total_cmp(&a.weighted_score)
            .then_with(|| candidates[*left].name.cmp(&candidates[*right].name))
            .then_with(|| candidates[*left].definition.cmp(&candidates[*right].definition))
            .then_with(|| left.cmp(right))
    });

    let winner = evaluations[frontier[0]].clone();
    Ok(ObjectiveSelectionReport {
        evaluator_version: REASONING_OBJECTIVE_CORE_VERSION.into(),
        context,
        weights,
        dominant_objectives: weights.dominant_objectives(),
        evaluations,
        selected_source_index: winner.source_index,
        selected_candidate_name: winner.candidate_name,
        selected_vector: winner.vector,
        selected_weighted_score: winner.weighted_score,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub enum ObjectiveCoreError {
    EmptyCandidateSet,
    EmptyParetoFrontier,
    InvalidObjective {
        field: &'static str,
        candidate_name: Option<String>,
        value: f64,
    },
    InvalidWeightSum(f64),
}

impl fmt::Display for ObjectiveCoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCandidateSet => write!(f, "objective selection requires at least one candidate"),
            Self::EmptyParetoFrontier => write!(f, "objective selection produced an empty Pareto frontier"),
            Self::InvalidObjective {
                field,
                candidate_name,
                value,
            } => match candidate_name {
                Some(name) => write!(
                    f,
                    "candidate `{name}` objective `{field}` must be finite and normalized to [0, 1], got {value}"
                ),
                None => write!(
                    f,
                    "objective `{field}` must be finite and normalized to [0, 1], got {value}"
                ),
            },
            Self::InvalidWeightSum(sum) => write!(f, "objective weights must sum to 1.0, got {sum}"),
        }
    }
}

impl std::error::Error for ObjectiveCoreError {}

fn validate_unit(
    field: &'static str,
    value: f64,
    candidate_name: Option<&str>,
) -> Result<(), ObjectiveCoreError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(ObjectiveCoreError::InvalidObjective {
            field,
            candidate_name: candidate_name.map(str::to_owned),
            value,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::epistemic_tiers::EpistemicCoordinate;
    use crate::hdc::BinaryHV;
    use symthaea_core::hdc::primitive_system::PrimitiveTier;

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
            encoding: BinaryHV::random(name.len() as u64 + 100),
            epistemic_coordinate,
            harmonic_alignment,
        }
    }

    #[test]
    fn objective_identity_is_typed() {
        assert_eq!(
            ObjectiveWeights::for_context(ReasoningContext::CriticalSafety).dominant_objectives(),
            vec![ObjectiveKind::HarmonicAlignment]
        );
    }

    #[test]
    fn social_context_uses_real_harmonic_alignment() {
        let report = select_candidate_by_objectives(
            ReasoningContext::SocialInteraction,
            &[
                candidate("low", 0.5, 0.1, EpistemicCoordinate::null()),
                candidate("high", 0.5, 0.9, EpistemicCoordinate::null()),
            ],
        )
        .unwrap();
        assert_eq!(report.selected_candidate_name, "high");
        assert_eq!(report.selected_vector.harmonic_alignment, 0.9);
    }

    #[test]
    fn technical_context_uses_real_epistemic_grounding() {
        let report = select_candidate_by_objectives(
            ReasoningContext::TechnicalImplementation,
            &[
                candidate("low", 0.5, 0.5, EpistemicCoordinate::null()),
                candidate("high", 0.5, 0.5, EpistemicCoordinate::axiom()),
            ],
        )
        .unwrap();
        assert_eq!(report.selected_candidate_name, "high");
        assert_eq!(report.selected_vector.epistemic_grounding, 1.0);
    }

    #[test]
    fn malformed_candidate_fails_closed() {
        let err = select_candidate_by_objectives(
            ReasoningContext::GeneralReasoning,
            &[candidate("bad", f64::NAN, 0.5, EpistemicCoordinate::null())],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ObjectiveCoreError::InvalidObjective { field: "fitness", .. }
        ));
    }

    #[test]
    fn ties_are_deterministic() {
        let report = select_candidate_by_objectives(
            ReasoningContext::GeneralReasoning,
            &[
                candidate("zeta", 0.5, 0.5, EpistemicCoordinate::null()),
                candidate("alpha", 0.5, 0.5, EpistemicCoordinate::null()),
            ],
        )
        .unwrap();
        assert_eq!(report.selected_candidate_name, "alpha");
    }
}
