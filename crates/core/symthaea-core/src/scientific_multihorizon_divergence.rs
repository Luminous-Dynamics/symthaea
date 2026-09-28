// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Multi-horizon computational divergence over explicit trajectory snapshots.
//!
//! This extends single-horizon discrimination without changing its epistemic
//! status: separation is a representational diagnostic, not falsification,
//! evidence, truth, confidence, or independence.

use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// One explicit predicted state for a model at one horizon.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScientificTrajectorySnapshot {
    pub model_id: String,
    pub lineage: String,
    pub horizon_seconds: f32,
    pub predicted_state: ContinuousHV,
}

/// Pairwise divergence evaluated at every supplied horizon.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiHorizonPair {
    pub left_model_id: String,
    pub right_model_id: String,
    pub left_lineage: String,
    pub right_lineage: String,
    pub horizons: Vec<f32>,
    pub divergences: Vec<f32>,
    pub max_divergence: f32,
    pub mean_divergence: f32,
    pub horizon_at_max_divergence: f32,
}

/// Deterministically ordered multi-horizon divergence results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultiHorizonDivergenceSet {
    pub pairs: Vec<MultiHorizonPair>,
}

/// Validation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiHorizonDivergenceError {
    EmptySnapshots,
    EmptyModelId,
    EmptyLineage,
    DuplicateSnapshot,
    FewerThanTwoModels,
    InvalidHorizon,
    DimensionMismatch,
    NonFiniteState,
    IncompleteHorizonGrid,
    NonFiniteDivergence,
}

impl fmt::Display for MultiHorizonDivergenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for MultiHorizonDivergenceError {}

/// Compute pairwise divergence over a complete, shared horizon grid.
///
/// Every model must provide exactly one snapshot at every horizon. Inputs may
/// arrive in any order; the output is canonicalized by model IDs and horizon.
pub fn discover(
    snapshots: &[ScientificTrajectorySnapshot],
) -> Result<MultiHorizonDivergenceSet, MultiHorizonDivergenceError> {
    let horizons = validate(snapshots)?;

    let mut by_model: BTreeMap<&str, Vec<&ScientificTrajectorySnapshot>> = BTreeMap::new();
    for snapshot in snapshots {
        by_model.entry(snapshot.model_id.as_str()).or_default().push(snapshot);
    }

    let model_ids: Vec<&str> = by_model.keys().copied().collect();
    let mut pairs = Vec::with_capacity(model_ids.len().saturating_mul(model_ids.len().saturating_sub(1)) / 2);

    for left_index in 0..model_ids.len() {
        for right_index in (left_index + 1)..model_ids.len() {
            let left = &by_model[model_ids[left_index]];
            let right = &by_model[model_ids[right_index]];
            let mut divergences = Vec::with_capacity(horizons.len());

            for horizon_index in 0..horizons.len() {
                let similarity = left[horizon_index]
                    .predicted_state
                    .similarity(&right[horizon_index].predicted_state);
                if !similarity.is_finite() {
                    return Err(MultiHorizonDivergenceError::NonFiniteDivergence);
                }
                divergences.push(1.0 - similarity);
            }

            let (max_index, &max_divergence) = divergences
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b).then_with(|| {
                    horizons[0].total_cmp(&horizons[0])
                }))
                .expect("validated non-empty horizon grid");
            let mean_divergence =
                divergences.iter().copied().sum::<f32>() / divergences.len() as f32;

            pairs.push(MultiHorizonPair {
                left_model_id: model_ids[left_index].into(),
                right_model_id: model_ids[right_index].into(),
                left_lineage: left[0].lineage.clone(),
                right_lineage: right[0].lineage.clone(),
                horizons: horizons.clone(),
                divergences,
                max_divergence,
                mean_divergence,
                horizon_at_max_divergence: horizons[max_index],
            });
        }
    }

    pairs.sort_by(|a, b| {
        b.max_divergence
            .total_cmp(&a.max_divergence)
            .then_with(|| b.mean_divergence.total_cmp(&a.mean_divergence))
            .then_with(|| a.horizon_at_max_divergence.total_cmp(&b.horizon_at_max_divergence))
            .then_with(|| a.left_model_id.cmp(&b.left_model_id))
            .then_with(|| a.right_model_id.cmp(&b.right_model_id))
    });

    Ok(MultiHorizonDivergenceSet { pairs })
}

fn validate(
    snapshots: &[ScientificTrajectorySnapshot],
) -> Result<Vec<f32>, MultiHorizonDivergenceError> {
    if snapshots.is_empty() {
        return Err(MultiHorizonDivergenceError::EmptySnapshots);
    }

    let mut horizons = BTreeSet::new();
    let mut model_ids = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut dimension = None;

    for snapshot in snapshots {
        if snapshot.model_id.trim().is_empty() {
            return Err(MultiHorizonDivergenceError::EmptyModelId);
        }
        if snapshot.lineage.trim().is_empty() {
            return Err(MultiHorizonDivergenceError::EmptyLineage);
        }
        if !snapshot.horizon_seconds.is_finite() || snapshot.horizon_seconds <= 0.0 {
            return Err(MultiHorizonDivergenceError::InvalidHorizon);
        }
        if !horizons.insert(snapshot.horizon_seconds.to_bits()) {
            // Same model/horizon is the actual duplicate; other models may
            // legitimately share the horizon.
        }
        let key = (snapshot.model_id.clone(), snapshot.horizon_seconds.to_bits());
        if !seen.insert(key) {
            return Err(MultiHorizonDivergenceError::DuplicateSnapshot);
        }
        model_ids.insert(snapshot.model_id.as_str());
        if let Some(expected) = dimension {
            if expected != snapshot.predicted_state.dim() {
                return Err(MultiHorizonDivergenceError::DimensionMismatch);
            }
        } else {
            dimension = Some(snapshot.predicted_state.dim());
        }
        if snapshot.predicted_state.values.iter().any(|v| !v.is_finite()) {
            return Err(MultiHorizonDivergenceError::NonFiniteState);
        }
    }

    if model_ids.len() < 2 {
        return Err(MultiHorizonDivergenceError::FewerThanTwoModels);
    }

    let mut ordered_horizons: Vec<f32> =
        snapshots.iter().map(|s| s.horizon_seconds).collect();
    ordered_horizons.sort_by(|a, b| a.total_cmp(b));
    ordered_horizons.dedup_by(|a, b| a.to_bits() == b.to_bits());

    let expected = ordered_horizons.len();
    if snapshots.len() != expected * model_ids.len() {
        return Err(MultiHorizonDivergenceError::IncompleteHorizonGrid);
    }

    Ok(ordered_horizons)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::unified_hv::HDC_DIMENSION;

    fn state(seed: u64) -> ContinuousHV {
        ContinuousHV::random(HDC_DIMENSION, seed)
    }

    fn snapshots() -> Vec<ScientificTrajectorySnapshot> {
        vec![
            ScientificTrajectorySnapshot { model_id: "a".into(), lineage: "la".into(), horizon_seconds: 1.0, predicted_state: state(1) },
            ScientificTrajectorySnapshot { model_id: "a".into(), lineage: "la".into(), horizon_seconds: 2.0, predicted_state: state(2) },
            ScientificTrajectorySnapshot { model_id: "b".into(), lineage: "lb".into(), horizon_seconds: 1.0, predicted_state: state(1) },
            ScientificTrajectorySnapshot { model_id: "b".into(), lineage: "lb".into(), horizon_seconds: 2.0, predicted_state: state(3) },
        ]
    }

    #[test]
    fn discovers_temporal_divergence() {
        let result = discover(&snapshots()).unwrap();
        assert_eq!(result.pairs.len(), 1);
        assert_eq!(result.pairs[0].horizons, vec![1.0, 2.0]);
        assert!(result.pairs[0].divergences[0].abs() < f32::EPSILON);
        assert!(result.pairs[0].divergences[1] > 0.0);
        assert_eq!(result.pairs[0].horizon_at_max_divergence, 2.0);
    }

    #[test]
    fn input_permutation_is_invariant() {
        let a = discover(&snapshots()).unwrap();
        let mut reversed = snapshots();
        reversed.reverse();
        assert_eq!(a, discover(&reversed).unwrap());
    }

    #[test]
    fn incomplete_grid_is_rejected() {
        let mut s = snapshots();
        s.pop();
        assert!(matches!(
            discover(&s),
            Err(MultiHorizonDivergenceError::IncompleteHorizonGrid)
        ));
    }

    #[test]
    fn duplicate_snapshot_is_rejected() {
        let mut s = snapshots();
        s.push(s[0].clone());
        assert!(matches!(
            discover(&s),
            Err(MultiHorizonDivergenceError::DuplicateSnapshot)
        ));
    }

    #[test]
    fn shared_lineage_is_not_independence() {
        let mut s = snapshots();
        s[2].lineage = "la".into();
        let result = discover(&s).unwrap();
        assert_eq!(result.pairs[0].left_lineage, result.pairs[0].right_lineage);
    }
}
