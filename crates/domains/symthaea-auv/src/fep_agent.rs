// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Active Inference AUV agent: depth/navigation priors + blackout uncertainty.

use crate::types::{AuvState, NUM_STATE_CHANNELS};
use symthaea_fep::Observation;

/// FEP modulation result for AUV controller.
#[derive(Debug, Clone)]
pub struct AuvFepResult {
    pub tau_factor: f32,
    pub learning_rate_factor: f32,
    /// Hand-derived depth/angular-rate/blackout deviation estimate -- NOT
    /// the agent's free energy. Kept under its original name and formula
    /// for backward compatibility with existing tau/lr gating and the
    /// `test_blackout_increases_uncertainty` relative-comparison test. See
    /// `perceived_free_energy` for the genuine FEP output (added by the
    /// SYMTHAEA_CLASSIC_PLATFORMS_FEP_HONESTY_2026-07-09.md fix).
    pub free_energy: f64,
    /// The wrapped agent's real, perceive()-computed free energy.
    /// `agent` was previously constructed and never perceived at all (pure
    /// dead weight); this is now genuine.
    pub perceived_free_energy: f64,
    /// Whether to widen priors due to communication blackout.
    pub blackout_uncertainty_boost: bool,
}

/// Active Inference agent for AUV navigation and water monitoring.
pub struct ActiveInferenceAuvAgent {
    agent: symthaea_fep::ActiveInferenceAgent,
    /// Whether currently in communication blackout.
    in_blackout: bool,
    tick_count: u64,
}

impl ActiveInferenceAuvAgent {
    pub fn new() -> Self {
        let config = symthaea_fep::ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim: NUM_STATE_CHANNELS,
            num_actions: 6,
            belief_learning_rate: 0.1,
            planning_horizon: 1,
            action_temperature: 0.5,
            ..symthaea_fep::ActiveInferenceAgentConfig::default()
        };
        Self {
            agent: symthaea_fep::ActiveInferenceAgent::new(config),
            in_blackout: false,
            tick_count: 0,
        }
    }

    /// Update blackout status (affects uncertainty).
    pub fn set_blackout(&mut self, in_blackout: bool) {
        self.in_blackout = in_blackout;
    }

    /// Cognitive tick: observe state, modulate controller.
    pub fn tick(&mut self, state: &AuvState, target_depth: f64) -> AuvFepResult {
        self.tick_count += 1;
        let values: Vec<f64> = state.to_channels().iter().map(|v| *v as f64).collect();
        // Blackout genuinely degrades sensor trust, so it lowers the
        // observation's precision rather than just padding a heuristic --
        // a legitimate use of perceive()'s precision weighting, not a
        // fabricated add-on.
        let precision = if self.in_blackout { 0.3 } else { 1.0 };
        let observation = Observation {
            values,
            precision,
            timestamp: self.tick_count,
            modality: "auv".to_string(),
        };
        self.agent.perceive(&observation);

        let depth_error = (state.depth - target_depth).abs();
        let angular_speed = state
            .angular_velocity
            .iter()
            .map(|w| w * w)
            .sum::<f64>()
            .sqrt();
        let speed = state.speed();

        let tau_factor = if depth_error > 10.0 || angular_speed > 1.0 {
            0.85
        } else if depth_error < 1.0 && angular_speed < 0.2 {
            1.15
        } else {
            1.0
        };

        let lr_factor = if depth_error > 20.0 || self.in_blackout {
            1.5 // High uncertainty: learn faster
        } else if depth_error < 0.5 {
            0.6
        } else {
            1.0
        };

        let free_energy = depth_error
            + angular_speed * 2.0
            + speed * 0.5
            + if self.in_blackout { 5.0 } else { 0.0 };

        AuvFepResult {
            tau_factor,
            learning_rate_factor: lr_factor,
            free_energy,
            perceived_free_energy: self.agent.current_free_energy(),
            blackout_uncertainty_boost: self.in_blackout,
        }
    }

    pub fn reset(&mut self) {
        self.agent.reset();
        self.in_blackout = false;
    }
}

impl Default for ActiveInferenceAuvAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_at_target_depth() {
        let mut agent = ActiveInferenceAuvAgent::new();
        let state = AuvState::neutral_buoyancy(50.0);
        let result = agent.tick(&state, 50.0);
        assert!(result.tau_factor >= 1.0);
        assert!(result.free_energy < 5.0);
    }

    #[test]
    fn test_blackout_increases_uncertainty() {
        let mut agent = ActiveInferenceAuvAgent::new();
        let state = AuvState::neutral_buoyancy(50.0);

        let normal = agent.tick(&state, 50.0);
        agent.set_blackout(true);
        let blackout = agent.tick(&state, 50.0);

        assert!(blackout.free_energy > normal.free_energy);
        assert!(blackout.blackout_uncertainty_boost);
        assert!(blackout.learning_rate_factor > normal.learning_rate_factor);
    }

    #[test]
    fn test_perceived_free_energy_is_real_and_finite() {
        let mut agent = ActiveInferenceAuvAgent::new();
        let state = AuvState::neutral_buoyancy(50.0);
        let result = agent.tick(&state, 50.0);
        assert!(result.perceived_free_energy.is_finite());
    }

    #[test]
    fn test_fep_perceiving_changes_agent_belief() {
        let mut agent = ActiveInferenceAuvAgent::new();
        let belief_before = agent.agent.belief.mean.clone();
        let deep_state = AuvState::neutral_buoyancy(500.0);
        for _ in 0..10 {
            agent.tick(&deep_state, 10.0);
        }
        assert_ne!(
            belief_before, agent.agent.belief.mean,
            "belief must change after repeated perception of an extreme observation"
        );
    }
}
