// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Primitive-Belief Bridge: Connecting Ontological Primitives to Active Inference
//!
//! Maps between the 9-tier primitive system and active inference belief states,
//! enabling primitives to participate in free energy minimization.
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────┐        ┌──────────────────────┐
//! │  Primitive          │        │  Active Inference     │
//! │  Consciousness      │◄──────►│  Agent               │
//! │  State              │        │  (FEP beliefs)       │
//! │  (9 tiers)          │        │                      │
//! └────────────────────┘        └──────────────────────┘
//!          │                              │
//!          └──────── Bridge ──────────────┘
//!            state_to_beliefs()
//!            beliefs_to_tier_targets()
//!            compute_prediction_error()
//!            td_error_signal()
//! ```
//!
//! ## Bidirectional Flow
//!
//! **Bottom-up (primitives → beliefs)**: Active primitive states are converted
//! to a belief vector that the active inference agent uses as observations.
//!
//! **Top-down (beliefs → primitives)**: The agent's posterior beliefs are
//! converted to target activations that modulate primitive selection.
//!
//! **Error signal**: Prediction errors at the primitive level feed into
//! TD(λ) learning, creating a reward signal for accurate primitive prediction.

use crate::consciousness::primitive_consciousness::PrimitiveConsciousnessState;
use serde::{Deserialize, Serialize};
use symthaea_core::hdc::primitive_system::PrimitiveTier;

// ═══════════════════════════════════════════════════════════════════════════════
// TIER ORDERING
// ═══════════════════════════════════════════════════════════════════════════════

/// Canonical ordering of primitive tiers for consistent belief vector indexing.
const TIER_ORDER: [PrimitiveTier; 9] = [
    PrimitiveTier::NSM,
    PrimitiveTier::Mathematical,
    PrimitiveTier::Physical,
    PrimitiveTier::Geometric,
    PrimitiveTier::Strategic,
    PrimitiveTier::MetaCognitive,
    PrimitiveTier::Temporal,
    PrimitiveTier::Compositional,
    PrimitiveTier::Consciousness,
];

// ═══════════════════════════════════════════════════════════════════════════════
// BRIDGE
// ═══════════════════════════════════════════════════════════════════════════════

/// Bridges the 9-tier primitive system with active inference belief states.
///
/// Converts between primitive consciousness states and flat belief vectors,
/// computes prediction errors at the primitive level, and generates TD-compatible
/// error signals for temporal difference learning.
pub struct PrimitiveBeliefBridge {
    /// Number of belief dimensions (one per tier)
    belief_dim: usize,
    /// Running EMA of per-tier prediction errors
    tier_prediction_errors: [f64; 9],
    /// EMA decay factor (closer to 1.0 = slower adaptation)
    ema_decay: f64,
    /// Total updates processed
    updates: u64,
}

impl PrimitiveBeliefBridge {
    /// Create a new bridge with default parameters.
    pub fn new() -> Self {
        Self {
            belief_dim: 9,
            tier_prediction_errors: [0.0; 9],
            ema_decay: 0.95,
            updates: 0,
        }
    }

    /// Create with custom EMA decay rate.
    pub fn with_ema_decay(ema_decay: f64) -> Self {
        Self {
            ema_decay: ema_decay.clamp(0.0, 0.999),
            ..Self::new()
        }
    }

    /// Convert a primitive consciousness state to a belief vector.
    ///
    /// Each dimension represents the mean activation of one primitive tier.
    /// This vector can be used as an observation for active inference.
    ///
    /// Returns a 9-element vector indexed by `TIER_ORDER`.
    pub fn state_to_beliefs(&self, state: &PrimitiveConsciousnessState) -> Vec<f64> {
        TIER_ORDER
            .iter()
            .map(|tier| {
                state
                    .active_by_tier
                    .get(tier)
                    .map(|actives| {
                        if actives.is_empty() {
                            0.0
                        } else {
                            actives.iter().map(|a| a.activation).sum::<f64>() / actives.len() as f64
                        }
                    })
                    .unwrap_or(0.0)
            })
            .collect()
    }

    /// Convert a belief vector back to primitive tier activation targets.
    ///
    /// Returns (tier, target_activation) pairs that can be used to
    /// top-down modulate primitive selection via attention allocation.
    pub fn beliefs_to_tier_targets(&self, beliefs: &[f64]) -> Vec<(PrimitiveTier, f64)> {
        TIER_ORDER
            .iter()
            .zip(beliefs.iter())
            .map(|(&tier, &belief)| (tier, belief.clamp(0.0, 1.0)))
            .collect()
    }

    /// Compute prediction error between predicted and actual primitive states.
    ///
    /// Returns per-tier absolute errors and an aggregate scalar. Also updates
    /// the running EMA of per-tier errors for profiling.
    pub fn compute_prediction_error(
        &mut self,
        predicted: &PrimitiveConsciousnessState,
        actual: &PrimitiveConsciousnessState,
    ) -> PrimitivePredictionError {
        let predicted_beliefs = self.state_to_beliefs(predicted);
        let actual_beliefs = self.state_to_beliefs(actual);

        let mut tier_errors = Vec::with_capacity(9);
        let mut total_error = 0.0;

        for (i, (&pred, &act)) in predicted_beliefs
            .iter()
            .zip(actual_beliefs.iter())
            .enumerate()
        {
            let error = (pred - act).abs();
            tier_errors.push((TIER_ORDER[i], error));
            total_error += error;

            // Update running EMA of per-tier errors
            self.tier_prediction_errors[i] =
                self.tier_prediction_errors[i] * self.ema_decay + error * (1.0 - self.ema_decay);
        }

        let n = tier_errors.len().max(1) as f64;
        self.updates += 1;

        PrimitivePredictionError {
            tier_errors,
            aggregate_error: total_error / n,
            phi_delta: actual.phi - predicted.phi,
        }
    }

    /// Generate a TD-compatible error signal from a prediction error.
    ///
    /// Returns a scalar suitable for temporal difference learning:
    /// - Positive: actual state was better than predicted (Phi increased)
    /// - Negative: actual state was worse (Phi decreased or large prediction error)
    pub fn td_error_signal(&self, prediction_error: &PrimitivePredictionError) -> f64 {
        // Phi improvement is rewarding; prediction errors are costly
        let phi_reward = prediction_error.phi_delta;
        let error_cost = -prediction_error.aggregate_error * 0.5;
        phi_reward + error_cost
    }

    /// Get running average prediction errors per tier.
    ///
    /// Useful for identifying which tiers are hardest to predict,
    /// guiding attention allocation and learning effort.
    pub fn tier_error_profile(&self) -> Vec<(PrimitiveTier, f64)> {
        TIER_ORDER
            .iter()
            .enumerate()
            .map(|(i, &tier)| (tier, self.tier_prediction_errors[i]))
            .collect()
    }

    /// Get the tier with the highest prediction error (needs most learning).
    pub fn most_surprising_tier(&self) -> Option<(PrimitiveTier, f64)> {
        self.tier_error_profile()
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get the number of belief dimensions.
    pub fn belief_dim(&self) -> usize {
        self.belief_dim
    }

    /// Get total updates processed.
    pub fn updates(&self) -> u64 {
        self.updates
    }
}

impl Default for PrimitiveBeliefBridge {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PREDICTION ERROR
// ═══════════════════════════════════════════════════════════════════════════════

/// Prediction error decomposed by primitive tier.
///
/// Captures both the per-tier absolute errors and the overall Phi change,
/// enabling fine-grained credit assignment during learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitivePredictionError {
    /// Per-tier absolute prediction errors
    pub tier_errors: Vec<(PrimitiveTier, f64)>,
    /// Aggregate prediction error (mean across tiers)
    pub aggregate_error: f64,
    /// Change in Phi between predicted and actual states
    pub phi_delta: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let bridge = PrimitiveBeliefBridge::new();
        assert_eq!(bridge.belief_dim(), 9);
        assert_eq!(bridge.updates(), 0);
    }

    #[test]
    fn test_empty_state_to_beliefs() {
        let bridge = PrimitiveBeliefBridge::new();
        let state = PrimitiveConsciousnessState::new(0.0);
        let beliefs = bridge.state_to_beliefs(&state);
        assert_eq!(beliefs.len(), 9);
        assert!(beliefs.iter().all(|&b| b == 0.0));
    }

    #[test]
    fn test_prediction_error_identical_states() {
        let mut bridge = PrimitiveBeliefBridge::new();
        let state = PrimitiveConsciousnessState::new(0.0);
        let error = bridge.compute_prediction_error(&state, &state);
        assert!(error.aggregate_error < 1e-6);
        assert!(error.phi_delta.abs() < 1e-6);
    }

    #[test]
    fn test_beliefs_roundtrip() {
        let bridge = PrimitiveBeliefBridge::new();
        let beliefs = vec![0.5; 9];
        let targets = bridge.beliefs_to_tier_targets(&beliefs);
        assert_eq!(targets.len(), 9);
        for (_, activation) in &targets {
            assert!((*activation - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_beliefs_clamping() {
        let bridge = PrimitiveBeliefBridge::new();
        let beliefs = vec![-1.0, 2.0, 0.5, 0.0, 1.0, 0.3, 0.7, -0.5, 1.5];
        let targets = bridge.beliefs_to_tier_targets(&beliefs);
        for (_, activation) in &targets {
            assert!(*activation >= 0.0 && *activation <= 1.0);
        }
    }

    #[test]
    fn test_td_error_positive_phi_improvement() {
        let bridge = PrimitiveBeliefBridge::new();
        let error = PrimitivePredictionError {
            tier_errors: vec![],
            aggregate_error: 0.1,
            phi_delta: 0.3,
        };
        let td = bridge.td_error_signal(&error);
        assert!(td > 0.0, "Phi improvement should yield positive TD signal");
    }

    #[test]
    fn test_td_error_negative_phi_decline() {
        let bridge = PrimitiveBeliefBridge::new();
        let error = PrimitivePredictionError {
            tier_errors: vec![],
            aggregate_error: 0.5,
            phi_delta: -0.3,
        };
        let td = bridge.td_error_signal(&error);
        assert!(
            td < 0.0,
            "Phi decline + high error should yield negative TD signal"
        );
    }

    #[test]
    fn test_ema_updates() {
        let mut bridge = PrimitiveBeliefBridge::new();

        // Create two different states
        let state1 = PrimitiveConsciousnessState::new(0.0);
        let mut state2 = PrimitiveConsciousnessState::new(0.0);
        state2.phi = 0.5;

        // First update
        bridge.compute_prediction_error(&state1, &state2);
        assert_eq!(bridge.updates(), 1);

        // Profile should reflect the errors
        let profile = bridge.tier_error_profile();
        assert_eq!(profile.len(), 9);
    }

    #[test]
    fn test_most_surprising_tier_empty() {
        let bridge = PrimitiveBeliefBridge::new();
        let result = bridge.most_surprising_tier();
        // All errors are 0.0, so the max is still valid but zero
        assert!(result.is_some());
        assert!(result.unwrap().1 < 1e-6);
    }
}
