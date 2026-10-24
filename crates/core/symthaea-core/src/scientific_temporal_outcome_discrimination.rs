// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Horizon-aligned computational outcome discrimination over scientific trajectories.
//!
//! This bridges the 002L multi-horizon representation to the 002D outcome
//! prototype likelihood model. The output answers a representational question:
//! at which supplied horizon do competing trajectories differ with respect to
//! an explicit measurable outcome? It does not turn model-implied likelihoods
//! into experimental probabilities or evidence.

use crate::scientific_multihorizon_divergence::{
    discover as discover_multihorizon_divergence, MultiHorizonDivergenceError,
    ScientificTrajectorySnapshot,
};
use crate::scientific_outcome_likelihood::{
    OutcomeLikelihoodError, OutcomeLikelihoodModel, ScientificOutcomePrototype,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// One outcome-discrimination candidate at one explicit horizon.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalOutcomeDiscrimination {
    pub left_model_id: String,
    pub right_model_id: String,
    pub left_lineage: String,
    pub right_lineage: String,
    pub horizon_seconds: f32,
    pub outcome_id: String,
    pub trajectory_similarity: f32,
    pub trajectory_divergence: f32,
    pub left_similarity: f32,
    pub right_similarity: f32,
    /// Model-implied normalized outcome weight, not an experimental probability.
    pub left_likelihood: f64,
    /// Model-implied normalized outcome weight, not an experimental probability.
    pub right_likelihood: f64,
    /// Absolute separation between model-implied outcome weights.
    pub likelihood_separation: f64,
}

/// Deterministically ranked temporal outcome-discrimination candidates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalOutcomeDiscriminationSet {
    pub candidates: Vec<TemporalOutcomeDiscrimination>,
}

/// Validation and projection failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalOutcomeDiscriminationError {
    MultiHorizon(MultiHorizonDivergenceError),
    EmptyOutcomes,
    EmptyOutcomeId,
    DuplicateOutcomeId,
    OutcomeLikelihood(OutcomeLikelihoodError),
}

impl fmt::Display for TemporalOutcomeDiscriminationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for TemporalOutcomeDiscriminationError {}

impl From<MultiHorizonDivergenceError> for TemporalOutcomeDiscriminationError {
    fn from(error: MultiHorizonDivergenceError) -> Self {
        Self::MultiHorizon(error)
    }
}

impl From<OutcomeLikelihoodError> for TemporalOutcomeDiscriminationError {
    fn from(error: OutcomeLikelihoodError) -> Self {
        Self::OutcomeLikelihood(error)
    }
}

/// Compute outcome discrimination for every model pair, explicit horizon, and
/// caller-supplied measurable outcome prototype.
///
/// Inputs are canonicalized through the 002L multi-horizon divergence contract.
/// Outcome likelihoods reuse the 002D representation exactly.
pub fn discover(
    snapshots: &[ScientificTrajectorySnapshot],
    outcomes: &[ScientificOutcomePrototype],
    likelihood_model: &OutcomeLikelihoodModel,
) -> Result<TemporalOutcomeDiscriminationSet, TemporalOutcomeDiscriminationError> {
    let divergence_set = discover_multihorizon_divergence(snapshots)?;
    validate_outcomes(outcomes, snapshots[0].predicted_state.dim())?;

    let mut likelihoods: BTreeMap<(String, u32), Vec<_>> = BTreeMap::new();
    for snapshot in snapshots {
        let inferred = likelihood_model.infer(&snapshot.predicted_state, outcomes)?;
        likelihoods.insert(
            (snapshot.model_id.clone(), snapshot.horizon_seconds.to_bits()),
            inferred,
        );
    }

    let by_outcome: BTreeMap<&str, &ScientificOutcomePrototype> = outcomes
        .iter()
        .map(|outcome| (outcome.outcome_id.as_str(), outcome))
        .collect();

    let mut candidates = Vec::new();
    for pair in divergence_set.pairs {
        for (horizon_index, &horizon) in pair.horizons.iter().enumerate() {
            let left = likelihoods
                .get(&(pair.left_model_id.clone(), horizon.to_bits()))
                .expect("validated complete horizon grid");
            let right = likelihoods
                .get(&(pair.right_model_id.clone(), horizon.to_bits()))
                .expect("validated complete horizon grid");

            let trajectory_divergence = pair.divergences[horizon_index];
            let trajectory_similarity = 1.0 - trajectory_divergence;

            for outcome in outcomes {
                let left_value = left
                    .iter()
                    .find(|value| value.outcome_id == outcome.outcome_id)
                    .expect("validated outcome IDs are emitted by infer");
                let right_value = right
                    .iter()
                    .find(|value| value.outcome_id == outcome.outcome_id)
                    .expect("validated outcome IDs are emitted by infer");

                debug_assert!(by_outcome.contains_key(outcome.outcome_id.as_str()));

                candidates.push(TemporalOutcomeDiscrimination {
                    left_model_id: pair.left_model_id.clone(),
                    right_model_id: pair.right_model_id.clone(),
                    left_lineage: pair.left_lineage.clone(),
                    right_lineage: pair.right_lineage.clone(),
                    horizon_seconds: horizon,
                    outcome_id: outcome.outcome_id.clone(),
                    trajectory_similarity,
                    trajectory_divergence,
                    left_similarity: left_value.similarity,
                    right_similarity: right_value.similarity,
                    left_likelihood: left_value.likelihood,
                    right_likelihood: right_value.likelihood,
                    likelihood_separation: (left_value.likelihood - right_value.likelihood).abs(),
                });
            }
        }
    }

    candidates.sort_by(|a, b| {
        b.likelihood_separation
            .total_cmp(&a.likelihood_separation)
            .then_with(|| b.trajectory_divergence.total_cmp(&a.trajectory_divergence))
            .then_with(|| a.horizon_seconds.total_cmp(&b.horizon_seconds))
            .then_with(|| a.outcome_id.cmp(&b.outcome_id))
            .then_with(|| a.left_model_id.cmp(&b.left_model_id))
            .then_with(|| a.right_model_id.cmp(&b.right_model_id))
    });

    Ok(TemporalOutcomeDiscriminationSet { candidates })
}

fn validate_outcomes(
    outcomes: &[ScientificOutcomePrototype],
    dimension: usize,
) -> Result<(), TemporalOutcomeDiscriminationError> {
    if outcomes.is_empty() {
        return Err(TemporalOutcomeDiscriminationError::EmptyOutcomes);
    }

    let mut ids = BTreeSet::new();
    for outcome in outcomes {
        if outcome.outcome_id.trim().is_empty() {
            return Err(TemporalOutcomeDiscriminationError::EmptyOutcomeId);
        }
        if !ids.insert(outcome.outcome_id.as_str()) {
            return Err(TemporalOutcomeDiscriminationError::DuplicateOutcomeId);
        }
        if outcome.prototype.dim() != dimension {
            return Err(TemporalOutcomeDiscriminationError::OutcomeLikelihood(
                OutcomeLikelihoodError::DimensionMismatch,
            ));
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

    fn snapshots() -> Vec<ScientificTrajectorySnapshot> {
        vec![
            ScientificTrajectorySnapshot {
                model_id: "a".into(),
                lineage: "la".into(),
                horizon_seconds: 1.0,
                predicted_state: state(1),
            },
            ScientificTrajectorySnapshot {
                model_id: "a".into(),
                lineage: "la".into(),
                horizon_seconds: 2.0,
                predicted_state: state(1),
            },
            ScientificTrajectorySnapshot {
                model_id: "b".into(),
                lineage: "lb".into(),
                horizon_seconds: 1.0,
                predicted_state: state(1),
            },
            ScientificTrajectorySnapshot {
                model_id: "b".into(),
                lineage: "lb".into(),
                horizon_seconds: 2.0,
                predicted_state: state(2),
            },
        ]
    }

    fn outcomes() -> Vec<ScientificOutcomePrototype> {
        vec![
            ScientificOutcomePrototype {
                outcome_id: "outcome-a".into(),
                prototype: state(1),
            },
            ScientificOutcomePrototype {
                outcome_id: "outcome-b".into(),
                prototype: state(2),
            },
        ]
    }

    #[test]
    fn produces_horizon_aligned_candidates() {
        let result =
            discover(&snapshots(), &outcomes(), &OutcomeLikelihoodModel::default()).unwrap();
        assert_eq!(result.candidates.len(), 4);
        assert!(result
            .candidates
            .iter()
            .any(|candidate| candidate.horizon_seconds == 1.0));
        assert!(result
            .candidates
            .iter()
            .any(|candidate| candidate.horizon_seconds == 2.0));
    }

    #[test]
    fn later_divergence_can_create_larger_outcome_separation() {
        let result =
            discover(&snapshots(), &outcomes(), &OutcomeLikelihoodModel::default()).unwrap();
        let later = result
            .candidates
            .iter()
            .filter(|candidate| candidate.horizon_seconds == 2.0)
            .map(|candidate| candidate.likelihood_separation)
            .fold(0.0, f64::max);
        let early = result
            .candidates
            .iter()
            .filter(|candidate| candidate.horizon_seconds == 1.0)
            .map(|candidate| candidate.likelihood_separation)
            .fold(0.0, f64::max);
        assert!(later > early);
    }

    #[test]
    fn input_permutation_is_invariant() {
        let a =
            discover(&snapshots(), &outcomes(), &OutcomeLikelihoodModel::default()).unwrap();
        let mut reversed = snapshots();
        reversed.reverse();
        let b =
            discover(&reversed, &outcomes(), &OutcomeLikelihoodModel::default()).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn duplicate_outcomes_are_rejected() {
        let mut o = outcomes();
        o[1].outcome_id = o[0].outcome_id.clone();
        assert!(matches!(
            discover(&snapshots(), &o, &OutcomeLikelihoodModel::default()),
            Err(TemporalOutcomeDiscriminationError::DuplicateOutcomeId)
        ));
    }

    #[test]
    fn shared_lineage_is_not_independence() {
        let mut s = snapshots();
        s[2].lineage = "la".into();
        let result =
            discover(&s, &outcomes(), &OutcomeLikelihoodModel::default()).unwrap();
        assert_eq!(result.candidates[0].left_lineage, result.candidates[0].right_lineage);
    }
}
