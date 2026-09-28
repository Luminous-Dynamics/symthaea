// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pairwise divergence discovery over competing scientific trajectories.
//!
//! Divergence identifies computationally distinct predictions that may be
//! useful when designing a discriminating observation. It is not falsification,
//! truth, confidence, or experimental evidence.

use crate::hdc::unified_hv::ContinuousHV;
use crate::scientific_active_inference::ScientificTrajectoryForecast;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

/// A pairwise separation between two model-predicted states.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrajectoryDivergence {
    pub left_model_id: String,
    pub right_model_id: String,
    pub left_lineage: String,
    pub right_lineage: String,
    pub horizon_seconds: f32,
    /// Cosine similarity between the two predicted states.
    pub similarity: f32,
    /// One minus similarity; a representational separation diagnostic.
    pub divergence: f32,
}

/// Deterministically ranked computational candidates for discrimination.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiscriminatingTrajectorySet {
    pub divergences: Vec<TrajectoryDivergence>,
}

/// Validation failures for trajectory divergence discovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrajectoryDivergenceError {
    EmptyForecasts,
    EmptyModelId,
    EmptyLineage,
    DuplicateModelId,
    InvalidHorizon,
    HorizonMismatch,
    DimensionMismatch,
    NonFiniteState,
}

impl fmt::Display for TrajectoryDivergenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for TrajectoryDivergenceError {}

/// Compute pairwise divergence between competing trajectory forecasts.
pub fn discover(
    forecasts: &[ScientificTrajectoryForecast],
) -> Result<DiscriminatingTrajectorySet, TrajectoryDivergenceError> {
    validate(forecasts)?;

    let mut divergences = Vec::with_capacity(forecasts.len().saturating_mul(forecasts.len().saturating_sub(1)) / 2));

    for left in 0..forecasts.len() {
        for right in (left + 1)..forecasts.len() {
            let a = &forecasts[left];
            let b = &forecasts[right];
            let similarity = a.predicted_state.similarity(&b.predicted_state);
            if !similarity.is_finite() {
                return Err(TrajectoryDivergenceError::NonFiniteState);
            }

            divergences.push(TrajectoryDivergence {
                left_model_id: a.model_id.clone(),
                right_model_id: b.model_id.clone(),
                left_lineage: a.lineage.clone(),
                right_lineage: b.lineage.clone(),
                horizon_seconds: a.horizon_seconds,
                similarity,
                divergence: 1.0 - similarity,
            });
        }
    }

    divergences.sort_by(|a, b| {
        b.divergence
            .total_cmp(&a.divergence)
            .then_with(|| a.left_model_id.cmp(&b.left_model_id))
            .then_with(|| a.right_model_id.cmp(&b.right_model_id))
    });

    Ok(DiscriminatingTrajectorySet { divergences })
}

fn validate(
    forecasts: &[ScientificTrajectoryForecast],
) -> Result<(), TrajectoryDivergenceError> {
    if forecasts.len() < 2 {
        return Err(TrajectoryDivergenceError::EmptyForecasts);
    }

    let mut model_ids = BTreeSet::new();
    let mut dimension = None;
    let mut horizon = None;

    for forecast in forecasts {
        if forecast.model_id.trim().is_empty() {
            return Err(TrajectoryDivergenceError::EmptyModelId);
        }
        if !model_ids.insert(forecast.model_id.as_str()) {
            return Err(TrajectoryDivergenceError::DuplicateModelId);
        }
        if forecast.lineage.trim().is_empty() {
            return Err(TrajectoryDivergenceError::EmptyLineage);
        }
        if !forecast.horizon_seconds.is_finite() || forecast.horizon_seconds <= 0.0 {
            return Err(TrajectoryDivergenceError::InvalidHorizon);
        }
        if let Some(expected) = horizon {
            if (expected - forecast.horizon_seconds).abs() > f32::EPSILON {
                return Err(TrajectoryDivergenceError::HorizonMismatch);
            }
        } else {
            horizon = Some(forecast.horizon_seconds);
        }

        if let Some(expected) = dimension {
            if expected != forecast.predicted_state.dim() {
                return Err(TrajectoryDivergenceError::DimensionMismatch);
            }
        } else {
            dimension = Some(forecast.predicted_state.dim());
        }

        if forecast.predicted_state.values.iter().any(|v| !v.is_finite()) {
            return Err(TrajectoryDivergenceError::NonFiniteState);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::unified_hv::HDC_DIMENSION;

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
                outcome_probabilities: vec![1.0, 0.0],
            },
            ScientificTrajectoryForecast {
                model_id: "model-b".into(),
                lineage: "lineage-b".into(),
                prior: 0.5,
                horizon_seconds: 10.0,
                predicted_state: state(2),
                outcome_probabilities: vec![0.0, 1.0],
            },
        ]
    }

    #[test]
    fn discovers_pairwise_divergence() {
        let result = discover(&forecasts()).unwrap();
        assert_eq!(result.divergences.len(), 1);
        assert!(result.divergences[0].divergence >= 0.0);
    }

    #[test]
    fn identical_states_have_zero_divergence() {
        let mut f = forecasts();
        f[1].predicted_state = f[0].predicted_state.clone();
        let result = discover(&f).unwrap();
        assert!(result.divergences[0].divergence.abs() < f32::EPSILON);
    }

    #[test]
    fn most_separated_pair_is_first() {
        let mut f = forecasts();
        f.push(ScientificTrajectoryForecast {
            model_id: "model-c".into(),
            lineage: "lineage-c".into(),
            prior: 0.0,
            horizon_seconds: 10.0,
            predicted_state: state(1),
            outcome_probabilities: vec![0.5, 0.5],
        });
        let result = discover(&f).unwrap();
        assert_eq!(result.divergences[0].left_model_id, "model-a");
        assert_eq!(result.divergences[0].right_model_id, "model-b");
    }

    #[test]
    fn model_id_duplicates_are_rejected() {
        let mut f = forecasts();
        f[1].model_id = f[0].model_id.clone();
        assert!(matches!(discover(&f), Err(TrajectoryDivergenceError::DuplicateModelId)));
    }

    #[test]
    fn horizon_mismatch_is_rejected() {
        let mut f = forecasts();
        f[1].horizon_seconds = 11.0;
        assert!(matches!(discover(&f), Err(TrajectoryDivergenceError::HorizonMismatch)));
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let mut f = forecasts();
        f[1].predicted_state = ContinuousHV::random(64, 9);
        assert!(matches!(discover(&f), Err(TrajectoryDivergenceError::DimensionMismatch)));
    }

    #[test]
    fn lineage_is_metadata_not_independence() {
        let mut f = forecasts();
        f[1].lineage = f[0].lineage.clone();
        let result = discover(&f).unwrap();
        assert_eq!(result.divergences.len(), 1);
        assert_eq!(result.divergences[0].left_lineage, result.divergences[0].right_lineage);
    }
}
