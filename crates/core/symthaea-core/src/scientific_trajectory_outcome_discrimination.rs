// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bridge competing scientific trajectories to explicit measurable outcomes.
//!
//! This module connects trajectory divergence to the existing outcome-prototype
//! likelihood representation. The resulting separations are computational
//! discrimination candidates only: they are not experimental probabilities,
//! observations, evidence, falsification, truth, or Millennium criterion
//! satisfaction.

use crate::scientific_active_inference::ScientificTrajectoryForecast;
use crate::scientific_outcome_likelihood::{
    OutcomeLikelihoodError, OutcomeLikelihoodModel, ScientificOutcomePrototype,
};
use crate::scientific_trajectory_divergence::{
    discover as discover_trajectory_divergence, TrajectoryDivergence,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

/// Model-implied separation for one explicit outcome and one pair of models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryOutcomeDiscrimination {
    pub left_model_id: String,
    pub right_model_id: String,
    pub left_lineage: String,
    pub right_lineage: String,
    pub horizon_seconds: f32,
    pub outcome_id: String,
    /// Pairwise trajectory similarity before outcome projection.
    pub trajectory_similarity: f32,
    /// Pairwise trajectory divergence before outcome projection.
    pub trajectory_divergence: f32,
    pub left_similarity: f32,
    pub right_similarity: f32,
    /// Model-implied normalized weight for this outcome.
    pub left_likelihood: f64,
    /// Model-implied normalized weight for this outcome.
    pub right_likelihood: f64,
    /// Absolute separation between model-implied outcome weights.
    pub likelihood_separation: f64,
}

/// Deterministically ranked outcome candidates for discriminating competing models.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscriminatingOutcomeSet {
    pub candidates: Vec<TrajectoryOutcomeDiscrimination>,
}

/// Validation and projection failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrajectoryOutcomeDiscriminationError {
    EmptyForecasts,
    EmptyModelId,
    EmptyLineage,
    DuplicateModelId,
    InvalidHorizon,
    HorizonMismatch,
    DimensionMismatch,
    NonFiniteState,
    EmptyOutcomes,
    EmptyOutcomeId,
    DuplicateOutcomeId,
    OutcomeLikelihood(OutcomeLikelihoodError),
}

impl fmt::Display for TrajectoryOutcomeDiscriminationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for TrajectoryOutcomeDiscriminationError {}

impl From<OutcomeLikelihoodError> for TrajectoryOutcomeDiscriminationError {
    fn from(error: OutcomeLikelihoodError) -> Self {
        Self::OutcomeLikelihood(error)
    }
}

/// Rank explicit measurable outcomes by how strongly they separate each pair
/// of competing model-implied outcome distributions.
///
/// The outcome prototypes are supplied by the caller; this function never
/// invents observations or converts a computational candidate into evidence.
pub fn discover(
    forecasts: &[ScientificTrajectoryForecast],
    outcomes: &[ScientificOutcomePrototype],
    likelihood_model: &OutcomeLikelihoodModel,
) -> Result<DiscriminatingOutcomeSet, TrajectoryOutcomeDiscriminationError> {
    validate_forecasts(forecasts)?;
    validate_outcomes(outcomes, forecasts[0].predicted_state.dim())?;

    let trajectory_set = discover_trajectory_divergence(forecasts)
        .map_err(|error| match error {
            crate::scientific_trajectory_divergence::TrajectoryDivergenceError::EmptyForecasts => SelfError::EmptyForecasts,
            crate::scientific_trajectory_divergence::TrajectoryDivergenceError::EmptyModelId => SelfError::EmptyModelId,
            crate::scientific_trajectory_divergence::TrajectoryDivergenceError::EmptyLineage => SelfError::EmptyLineage,
            crate::scientific_trajectory_divergence::TrajectoryDivergenceError::DuplicateModelId => SelfError::DuplicateModelId,
            crate::scientific_trajectory_divergence::TrajectoryDivergenceError::InvalidHorizon => SelfError::InvalidHorizon,
            crate::scientific_trajectory_divergence::TrajectoryDivergenceError::HorizonMismatch => SelfError::HorizonMismatch,
            crate::scientific_trajectory_divergence::TrajectoryDivergenceError::DimensionMismatch => SelfError::DimensionMismatch,
            crate::scientific_trajectory_divergence::TrajectoryDivergenceError::NonFiniteState => SelfError::NonFiniteState,
        })
        .map_err(TrajectoryOutcomeDiscriminationError::from)?;

    let likelihoods: Vec<Vec<_>> = forecasts
        .iter()
        .map(|forecast| likelihood_model.infer(&forecast.predicted_state, outcomes))
        .collect::<Result<_, _>>()?;

    let by_model: std::collections::BTreeMap<&str, usize> = forecasts
        .iter()
        .enumerate()
        .map(|(index, forecast)| (forecast.model_id.as_str(), index))
        .collect();

    let mut candidates = Vec::with_capacity(
        trajectory_set
            .divergences
            .len()
            .saturating_mul(outcomes.len()),
    );

    for divergence in trajectory_set.divergences {
        let left_index = by_model[divergence.left_model_id.as_str()];
        let right_index = by_model[divergence.right_model_id.as_str()];
        for outcome in outcomes {
            let left = likelihoods[left_index]
                .iter()
                .find(|value| value.outcome_id == outcome.outcome_id)
                .expect("validated outcome IDs are emitted by infer");
            let right = likelihoods[right_index]
                .iter()
                .find(|value| value.outcome_id == outcome.outcome_id)
                .expect("validated outcome IDs are emitted by infer");

            candidates.push(TrajectoryOutcomeDiscrimination {
                left_model_id: divergence.left_model_id.clone(),
                right_model_id: divergence.right_model_id.clone(),
                left_lineage: divergence.left_lineage.clone(),
                right_lineage: divergence.right_lineage.clone(),
                horizon_seconds: divergence.horizon_seconds,
                outcome_id: outcome.outcome_id.clone(),
                trajectory_similarity: divergence.similarity,
                trajectory_divergence: divergence.divergence,
                left_similarity: left.similarity,
                right_similarity: right.similarity,
                left_likelihood: left.likelihood,
                right_likelihood: right.likelihood,
                likelihood_separation: (left.likelihood - right.likelihood).abs(),
            });
        }
    }

    candidates.sort_by(|a, b| {
        b.likelihood_separation
            .total_cmp(&a.likelihood_separation)
            .then_with(|| b.trajectory_divergence.total_cmp(&a.trajectory_divergence))
            .then_with(|| a.outcome_id.cmp(&b.outcome_id))
            .then_with(|| a.left_model_id.cmp(&b.left_model_id))
            .then_with(|| a.right_model_id.cmp(&b.right_model_id))
    });

    Ok(DiscriminatingOutcomeSet { candidates })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelfError {
    EmptyForecasts,
    EmptyModelId,
    EmptyLineage,
    DuplicateModelId,
    InvalidHorizon,
    HorizonMismatch,
    DimensionMismatch,
    NonFiniteState,
}

impl From<SelfError> for TrajectoryOutcomeDiscriminationError {
    fn from(error: SelfError) -> Self {
        match error {
            SelfError::EmptyForecasts => Self::EmptyForecasts,
            SelfError::EmptyModelId => Self::EmptyModelId,
            SelfError::EmptyLineage => Self::EmptyLineage,
            SelfError::DuplicateModelId => Self::DuplicateModelId,
            SelfError::InvalidHorizon => Self::InvalidHorizon,
            SelfError::HorizonMismatch => Self::HorizonMismatch,
            SelfError::DimensionMismatch => Self::DimensionMismatch,
            SelfError::NonFiniteState => Self::NonFiniteState,
        }
    }
}

fn validate_forecasts(
    forecasts: &[ScientificTrajectoryForecast],
) -> Result<(), TrajectoryOutcomeDiscriminationError> {
    if forecasts.len() < 2 {
        return Err(TrajectoryOutcomeDiscriminationError::EmptyForecasts);
    }

    let mut model_ids = BTreeSet::new();
    let mut dimension = None;
    let mut horizon = None;

    for forecast in forecasts {
        if forecast.model_id.trim().is_empty() {
            return Err(TrajectoryOutcomeDiscriminationError::EmptyModelId);
        }
        if !model_ids.insert(forecast.model_id.as_str()) {
            return Err(TrajectoryOutcomeDiscriminationError::DuplicateModelId);
        }
        if forecast.lineage.trim().is_empty() {
            return Err(TrajectoryOutcomeDiscriminationError::EmptyLineage);
        }
        if !forecast.horizon_seconds.is_finite() || forecast.horizon_seconds <= 0.0 {
            return Err(TrajectoryOutcomeDiscriminationError::InvalidHorizon);
        }
        if let Some(expected) = horizon {
            if (expected - forecast.horizon_seconds).abs() > f32::EPSILON {
                return Err(TrajectoryOutcomeDiscriminationError::HorizonMismatch);
            }
        } else {
            horizon = Some(forecast.horizon_seconds);
        }
        if let Some(expected) = dimension {
            if expected != forecast.predicted_state.dim() {
                return Err(TrajectoryOutcomeDiscriminationError::DimensionMismatch);
            }
        } else {
            dimension = Some(forecast.predicted_state.dim());
        }
        if forecast.predicted_state.values.iter().any(|v| !v.is_finite()) {
            return Err(TrajectoryOutcomeDiscriminationError::NonFiniteState);
        }
    }

    Ok(())
}

fn validate_outcomes(
    outcomes: &[ScientificOutcomePrototype],
    dimension: usize,
) -> Result<(), TrajectoryOutcomeDiscriminationError> {
    if outcomes.is_empty() {
        return Err(TrajectoryOutcomeDiscriminationError::EmptyOutcomes);
    }

    let mut ids = BTreeSet::new();
    for outcome in outcomes {
        if outcome.outcome_id.trim().is_empty() {
            return Err(TrajectoryOutcomeDiscriminationError::EmptyOutcomeId);
        }
        if !ids.insert(outcome.outcome_id.as_str()) {
            return Err(TrajectoryOutcomeDiscriminationError::DuplicateOutcomeId);
        }
        if outcome.prototype.dim() != dimension {
            return Err(TrajectoryOutcomeDiscriminationError::DimensionMismatch);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

    fn state(seed: u64) -> ContinuousHV {
        ContinuousHV::random(HDC_DIMENSION, seed)
    }

    fn forecasts() -> Vec<ScientificTrajectoryForecast> {
        vec![
            ScientificTrajectoryForecast {
                model_id: "model-a".into(),
                lineage: "lineage-a".into(),
                prior: 0.5,
                horizon_seconds: 10.0,
                predicted_state: state(1),
                outcome_probabilities: vec![],
            },
            ScientificTrajectoryForecast {
                model_id: "model-b".into(),
                lineage: "lineage-b".into(),
                prior: 0.5,
                horizon_seconds: 10.0,
                predicted_state: state(2),
                outcome_probabilities: vec![],
            },
        ]
    }

    fn outcomes() -> Vec<ScientificOutcomePrototype> {
        vec![
            ScientificOutcomePrototype { outcome_id: "outcome-a".into(), prototype: state(1) },
            ScientificOutcomePrototype { outcome_id: "outcome-b".into(), prototype: state(2) },
        ]
    }

    #[test]
    fn ranks_outcome_that_separates_models() {
        let result = discover(&forecasts(), &outcomes(), &OutcomeLikelihoodModel::default()).unwrap();
        assert_eq!(result.candidates.len(), 2);
        assert_eq!(result.candidates[0].outcome_id, "outcome-a");
        assert!(result.candidates[0].likelihood_separation > 0.0);
    }

    #[test]
    fn preserves_trajectory_divergence_metadata() {
        let result = discover(&forecasts(), &outcomes(), &OutcomeLikelihoodModel::default()).unwrap();
        assert!(result.candidates.iter().all(|candidate| candidate.trajectory_divergence.is_finite()));
        assert_eq!(result.candidates[0].horizon_seconds, 10.0);
    }

    #[test]
    fn identical_models_have_zero_outcome_separation() {
        let mut f = forecasts();
        f[1].predicted_state = f[0].predicted_state.clone();
        let result = discover(&f, &outcomes(), &OutcomeLikelihoodModel::default()).unwrap();
        assert!(result.candidates.iter().all(|candidate| candidate.likelihood_separation.abs() < 1e-12));
    }

    #[test]
    fn duplicate_outcomes_are_rejected() {
        let mut o = outcomes();
        o[1].outcome_id = o[0].outcome_id.clone();
        assert!(matches!(
            discover(&forecasts(), &o, &OutcomeLikelihoodModel::default()),
            Err(TrajectoryOutcomeDiscriminationError::DuplicateOutcomeId)
        ));
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let o = vec![ScientificOutcomePrototype {
            outcome_id: "bad".into(),
            prototype: ContinuousHV::random(64, 1),
        }];
        assert!(matches!(
            discover(&forecasts(), &o, &OutcomeLikelihoodModel::default()),
            Err(TrajectoryOutcomeDiscriminationError::DimensionMismatch)
        ));
    }

    #[test]
    fn shared_lineage_is_not_independence() {
        let mut f = forecasts();
        f[1].lineage = f[0].lineage.clone();
        let result = discover(&f, &outcomes(), &OutcomeLikelihoodModel::default()).unwrap();
        assert_eq!(result.candidates.len(), 2);
        assert_eq!(result.candidates[0].left_lineage, result.candidates[0].right_lineage);
    }
}
