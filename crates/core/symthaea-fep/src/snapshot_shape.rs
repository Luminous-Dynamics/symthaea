// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only shape evidence for validated FEP persistence capabilities.
//!
//! Cross-crate resume layers need to prove that an enclosing organism/population configuration
//! agrees with the cognition and Markov-boundary state it contains. They should not need raw
//! access to either snapshot's authority-bearing internals to do that. This module exposes only
//! the minimal structural facts required for such cross-checks, and only from already-validated
//! capabilities.

use crate::{ValidatedActiveInferenceAgentSnapshotV1, ValidatedMarkovBoundarySnapshotV1};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActiveInferenceAgentSnapshotShapeV1 {
    pub state_dim: usize,
    pub obs_dim: usize,
    pub num_actions: usize,
    pub action_temperature: f64,
    pub td_learning_enabled: bool,
    pub timestamp: u64,
    pub perception_cycles: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MarkovBoundarySnapshotShapeV1 {
    pub internal_dim: usize,
    pub sensory_dim: usize,
    pub active_dim: usize,
    pub alpha: f64,
    pub history_count: usize,
}

/// Minimal structural evidence from an already-validated active-inference snapshot.
pub fn active_inference_snapshot_shape_v1(
    validated: &ValidatedActiveInferenceAgentSnapshotV1,
) -> ActiveInferenceAgentSnapshotShapeV1 {
    let snapshot = validated.as_snapshot();
    ActiveInferenceAgentSnapshotShapeV1 {
        state_dim: snapshot.config.state_dim,
        obs_dim: snapshot.config.obs_dim,
        num_actions: snapshot.config.num_actions,
        action_temperature: snapshot.config.action_temperature,
        td_learning_enabled: snapshot.config.enable_td_learning,
        timestamp: snapshot.timestamp,
        perception_cycles: snapshot.stats.perception_cycles,
    }
}

/// Minimal structural evidence from an already-validated Markov-boundary snapshot.
pub fn markov_boundary_snapshot_shape_v1(
    validated: &ValidatedMarkovBoundarySnapshotV1,
) -> MarkovBoundarySnapshotShapeV1 {
    let snapshot = validated.as_snapshot();
    MarkovBoundarySnapshotShapeV1 {
        internal_dim: snapshot.partition.internal_dim,
        sensory_dim: snapshot.partition.sensory_dim,
        active_dim: snapshot.partition.active_dim,
        alpha: snapshot.alpha,
        history_count: snapshot.history_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ActiveInferenceAgentConfig, ActiveInferenceAgentSnapshotV1, ActiveInferenceAgentStats,
        BlanketPermeability, ExpectedFreeEnergyComputer, FreeEnergyCalculator, GenerativeModel,
        HiddenState, MarkovBoundarySnapshotV1, MarkovPartition, PrecisionEstimator,
        TemporalDifferenceLearner,
    };

    #[test]
    fn active_inference_shape_is_available_only_after_validation() {
        let config = ActiveInferenceAgentConfig {
            state_dim: 6,
            obs_dim: 6,
            num_actions: 3,
            action_temperature: 0.7,
            ..Default::default()
        };
        let raw = ActiveInferenceAgentSnapshotV1 {
            belief: HiddenState::new(config.state_dim),
            previous_state: None,
            last_action: None,
            model: GenerativeModel::new(config.state_dim, config.obs_dim, config.num_actions),
            free_energy_calc: FreeEnergyCalculator::new(500),
            precision: PrecisionEstimator::new(),
            efe_computer: ExpectedFreeEnergyComputer::new(config.obs_dim),
            td_learner: Some(TemporalDifferenceLearner::new(
                config.td_config.clone(),
                config.num_actions,
                config.state_dim,
                config.obs_dim,
            )),
            last_fe_components: None,
            stats: ActiveInferenceAgentStats::default(),
            timestamp: 0,
            rng_state: 0x9E37_79B9_7F4A_7C15,
            config,
        };
        let validated = raw.validate().expect("valid agent snapshot");
        let shape = active_inference_snapshot_shape_v1(&validated);
        assert_eq!(shape.state_dim, 6);
        assert_eq!(shape.obs_dim, 6);
        assert_eq!(shape.num_actions, 3);
        assert_eq!(shape.action_temperature.to_bits(), 0.7f64.to_bits());
        assert_eq!(shape.timestamp, shape.perception_cycles);
    }

    #[test]
    fn boundary_shape_is_available_only_after_validation() {
        let raw = MarkovBoundarySnapshotV1 {
            partition: MarkovPartition {
                internal_dim: 6,
                sensory_dim: 1,
                active_dim: 1,
            },
            permeability: BlanketPermeability::default(),
            permeability_ema: BlanketPermeability::default(),
            alpha: 0.1,
            history: vec![0.5; crate::MARKOV_BOUNDARY_HISTORY_CAP_V1],
            history_idx: 0,
            history_count: 0,
        };
        let validated = raw.validate().expect("valid boundary snapshot");
        let shape = markov_boundary_snapshot_shape_v1(&validated);
        assert_eq!(shape.internal_dim, 6);
        assert_eq!(shape.sensory_dim, 1);
        assert_eq!(shape.active_dim, 1);
        assert_eq!(shape.alpha.to_bits(), 0.1f64.to_bits());
        assert_eq!(shape.history_count, 0);
    }
}
