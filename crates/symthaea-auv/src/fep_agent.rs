// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Active Inference AUV agent: depth/navigation priors + blackout uncertainty.

use crate::types::AuvState;

/// FEP modulation result for AUV controller.
#[derive(Debug, Clone)]
pub struct AuvFepResult {
    pub tau_factor: f32,
    pub learning_rate_factor: f32,
    pub free_energy: f64,
    /// Whether to widen priors due to communication blackout.
    pub blackout_uncertainty_boost: bool,
}

/// Active Inference agent for AUV navigation and water monitoring.
pub struct ActiveInferenceAuvAgent {
    agent: symthaea_fep::ActiveInferenceAgent,
    /// Whether currently in communication blackout.
    in_blackout: bool,
}

impl ActiveInferenceAuvAgent {
    pub fn new() -> Self {
        let config = symthaea_fep::ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim: 8,
            num_actions: 6,
            belief_learning_rate: 0.1,
            planning_horizon: 1,
            action_temperature: 0.5,
            ..symthaea_fep::ActiveInferenceAgentConfig::default()
        };
        Self {
            agent: symthaea_fep::ActiveInferenceAgent::new(config),
            in_blackout: false,
        }
    }

    /// Update blackout status (affects uncertainty).
    pub fn set_blackout(&mut self, in_blackout: bool) {
        self.in_blackout = in_blackout;
    }

    /// Cognitive tick: observe state, modulate controller.
    pub fn tick(&mut self, state: &AuvState, target_depth: f64) -> AuvFepResult {
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
}
