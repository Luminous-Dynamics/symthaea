// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Model-implied outcome likelihoods from predicted scientific HDC states.
//!
//! These values are a contrastive representation model: similarities to
//! explicit outcome prototypes are converted to a normalized distribution.
//! They are not calibrated experimental probabilities and never constitute
//! observations or evidence.

use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use std::fmt;

/// An explicit representational prototype for one observable test outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificOutcomePrototype {
    pub outcome_id: String,
    pub prototype: ContinuousHV,
}

/// Contrastive likelihood model configuration.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OutcomeLikelihoodModel {
    /// Softmax temperature. Smaller values sharpen similarity differences.
    pub temperature: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutcomeLikelihood {
    pub outcome_id: String,
    /// Model-implied normalized weight, not an experimentally calibrated probability.
    pub likelihood: f64,
    pub similarity: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutcomeLikelihoodError {
    EmptyOutcomes,
    EmptyOutcomeId,
    DimensionMismatch,
    InvalidTemperature,
    NonFiniteSimilarity,
}

impl fmt::Display for OutcomeLikelihoodError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for OutcomeLikelihoodError {}

impl Default for OutcomeLikelihoodModel {
    fn default() -> Self {
        Self { temperature: 0.1 }
    }
}

impl OutcomeLikelihoodModel {
    /// Convert predicted-state/prototype similarities into a normalized model distribution.
    pub fn infer(
        &self,
        predicted_state: &ContinuousHV,
        outcomes: &[ScientificOutcomePrototype],
    ) -> Result<Vec<OutcomeLikelihood>, OutcomeLikelihoodError> {
        validate(self, predicted_state, outcomes)?;

        let similarities: Vec<f32> = outcomes
            .iter()
            .map(|outcome| predicted_state.similarity(&outcome.prototype))
            .collect();

        if similarities.iter().any(|s| !s.is_finite()) {
            return Err(OutcomeLikelihoodError::NonFiniteSimilarity);
        }

        // Stable softmax: subtract the largest logit before exponentiation.
        let logits: Vec<f64> = similarities
            .iter()
            .map(|s| f64::from(*s) / f64::from(self.temperature))
            .collect();
        let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = logits.iter().map(|x| (*x - max_logit).exp()).collect();
        let total: f64 = weights.iter().sum();

        Ok(outcomes
            .iter()
            .zip(similarities)
            .zip(weights)
            .map(|((outcome, similarity), weight)| OutcomeLikelihood {
                outcome_id: outcome.outcome_id.clone(),
                likelihood: weight / total,
                similarity,
            })
            .collect())
    }
}

fn validate(
    model: &OutcomeLikelihoodModel,
    predicted_state: &ContinuousHV,
    outcomes: &[ScientificOutcomePrototype],
) -> Result<(), OutcomeLikelihoodError> {
    if outcomes.is_empty() {
        return Err(OutcomeLikelihoodError::EmptyOutcomes);
    }
    if !model.temperature.is_finite() || model.temperature <= 0.0 {
        return Err(OutcomeLikelihoodError::InvalidTemperature);
    }

    for outcome in outcomes {
        if outcome.outcome_id.trim().is_empty() {
            return Err(OutcomeLikelihoodError::EmptyOutcomeId);
        }
        if outcome.prototype.dim() != predicted_state.dim() {
            return Err(OutcomeLikelihoodError::DimensionMismatch);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hv(seed: u64) -> ContinuousHV {
        ContinuousHV::random(128, seed)
    }

    fn outcomes() -> Vec<ScientificOutcomePrototype> {
        vec![
            ScientificOutcomePrototype { outcome_id: "a".into(), prototype: hv(1) },
            ScientificOutcomePrototype { outcome_id: "b".into(), prototype: hv(2) },
        ]
    }

    #[test]
    fn deterministic() {
        let model = OutcomeLikelihoodModel::default();
        assert_eq!(model.infer(&hv(3), &outcomes()), model.infer(&hv(3), &outcomes()));
    }

    #[test]
    fn likelihoods_normalize() {
        let result = OutcomeLikelihoodModel::default().infer(&hv(3), &outcomes()).unwrap();
        let sum: f64 = result.iter().map(|x| x.likelihood).sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn one_outcome_is_certain_under_the_model() {
        let result = OutcomeLikelihoodModel::default()
            .infer(&hv(3), &[ScientificOutcomePrototype { outcome_id: "only".into(), prototype: hv(1) }])
            .unwrap();
        assert!((result[0].likelihood - 1.0).abs() < 1e-12);
    }

    #[test]
    fn temperature_is_validated() {
        let model = OutcomeLikelihoodModel { temperature: 0.0 };
        assert!(matches!(
            model.infer(&hv(3), &outcomes()),
            Err(OutcomeLikelihoodError::InvalidTemperature)
        ));
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let outcomes = vec![
            ScientificOutcomePrototype { outcome_id: "bad".into(), prototype: ContinuousHV::random(64, 1) }
        ];
        assert!(matches!(
            OutcomeLikelihoodModel::default().infer(&hv(3), &outcomes),
            Err(OutcomeLikelihoodError::DimensionMismatch)
        ));
    }

    #[test]
    fn swapping_outcomes_swaps_outputs() {
        let model = OutcomeLikelihoodModel::default();
        let a = model.infer(&hv(3), &outcomes()).unwrap();
        let mut swapped = outcomes();
        swapped.swap(0, 1);
        let b = model.infer(&hv(3), &swapped).unwrap();
        assert_eq!(a[0].likelihood, b[1].likelihood);
        assert_eq!(a[1].likelihood, b[0].likelihood);
    }

    #[test]
    fn zero_temperature_is_not_used_as_a_hidden_argmax() {
        let model = OutcomeLikelihoodModel { temperature: f32::MIN_POSITIVE };
        let result = model.infer(&hv(3), &outcomes()).unwrap();
        let sum: f64 = result.iter().map(|x| x.likelihood).sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }
}
