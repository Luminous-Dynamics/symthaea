// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Active Inference helicopter agent: precision-weighted meta-control at 20Hz cognitive tick.
//!
//! Each `tick()` builds a 6D normalized observation (altitude error, attitude
//! error, speed, angular speed, signed altitude offset, rotor RPM deviation)
//! and steps the real `symthaea_fep::ActiveInferenceAgent` (`perceive()` →
//! variational belief update). Free energy and prediction error come from the
//! agent, not a hand-coded formula. Action selection (τ / learning-rate
//! modulation) uses a rule-based policy over the same observation channels —
//! the same default as symthaea-multirotor's `use_rule_based_policy: true`.

use symthaea_fep::Observation;

use crate::types::HelicopterState;

/// Active Inference modulation result.
#[derive(Debug, Clone)]
pub struct HelicopterFepResult {
    /// Time constant modulation (0.85 = faster, 1.15 = slower).
    pub tau_factor: f32,
    /// Learning rate multiplier.
    pub learning_rate_factor: f32,
    /// Current variational free energy (from the FEP agent's perceive()).
    pub free_energy: f64,
    /// Current prediction error (from the FEP agent's perceive()).
    pub prediction_error: f64,
}

/// Active Inference helicopter agent.
pub struct ActiveInferenceHelicopterAgent {
    /// FEP agent from symthaea-fep.
    agent: symthaea_fep::ActiveInferenceAgent,
}

impl ActiveInferenceHelicopterAgent {
    /// Create a new agent with default SAR configuration.
    pub fn new() -> Self {
        let config = symthaea_fep::ActiveInferenceAgentConfig {
            state_dim: 6,
            obs_dim: 6,
            num_actions: 6,
            belief_learning_rate: 0.1,
            planning_horizon: 1,
            action_temperature: 0.5,
            ..symthaea_fep::ActiveInferenceAgentConfig::default()
        };
        Self {
            agent: symthaea_fep::ActiveInferenceAgent::new(config),
        }
    }

    /// Build the 6D normalized observation vector.
    fn build_observation(state: &HelicopterState, target_altitude: f64) -> Vec<f64> {
        let altitude_error = (state.altitude() - target_altitude).abs();
        let (roll, pitch, _yaw) = state.euler_angles();
        let att_err = (roll * roll + pitch * pitch).sqrt();

        vec![
            (altitude_error / 20.0).min(1.0),
            (att_err / 1.0).min(1.0),
            (state.speed() / 10.0).min(1.0),
            (state.angular_speed() / 5.0).min(1.0),
            (((state.altitude() - target_altitude) / 20.0) + 0.5).clamp(0.0, 1.0),
            (((state.main_rotor_rpm - 3500.0) / 3500.0) * 0.5 + 0.5).clamp(0.0, 1.0),
        ]
    }

    /// Cognitive tick: observe state, update beliefs (real variational
    /// inference via `ActiveInferenceAgent::perceive`), modulate controller.
    ///
    /// Free energy / prediction error are the agent's variational estimates.
    /// The τ / learning-rate modulation is a rule-based policy over the raw
    /// error channels (mirroring multirotor's default rule-based policy).
    pub fn tick(&mut self, state: &HelicopterState, target_altitude: f64) -> HelicopterFepResult {
        let altitude_error = (state.altitude() - target_altitude).abs();
        let angular_speed = state.angular_speed();

        // Step the real FEP agent: perceive → variational belief update.
        let obs = Observation::new(
            Self::build_observation(state, target_altitude),
            1.0,
            "helicopter",
        );
        let perception = self.agent.perceive(&obs);
        let free_energy = perception.free_energy.total;
        let prediction_error = perception.free_energy.prediction_error;

        // Rule-based modulation policy over error channels.
        let tau_factor = if altitude_error > 5.0 || angular_speed > 1.0 {
            0.85 // Speed up response
        } else if altitude_error < 1.0 && angular_speed < 0.2 {
            1.15 // Slow down, conserve
        } else {
            1.0
        };

        let lr_factor = if altitude_error > 10.0 {
            1.5 // High error: learn faster
        } else if altitude_error < 0.5 {
            0.6 // Low error: reduce plasticity
        } else {
            1.0
        };

        HelicopterFepResult {
            tau_factor,
            learning_rate_factor: lr_factor,
            free_energy,
            prediction_error,
        }
    }

    /// Reset agent state.
    pub fn reset(&mut self) {
        self.agent.reset();
    }
}

impl Default for ActiveInferenceHelicopterAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_hover_stable() {
        let mut agent = ActiveInferenceHelicopterAgent::new();
        let state = HelicopterState::hover(20.0);
        let result = agent.tick(&state, 20.0);
        // At target altitude, should slow down (tau > 1)
        assert!(result.tau_factor >= 1.0);
        assert!(result.free_energy.is_finite());
        assert!(result.prediction_error.is_finite());
    }

    #[test]
    fn test_agent_high_error_speeds_up() {
        let mut agent = ActiveInferenceHelicopterAgent::new();
        let state = HelicopterState::hover(50.0); // 30m above target
        let result = agent.tick(&state, 20.0);
        assert!(
            result.tau_factor < 1.0,
            "High error should speed up response"
        );
        assert!(
            result.learning_rate_factor > 1.0,
            "High error should boost learning"
        );
        assert!(result.free_energy.is_finite());
    }

    #[test]
    fn test_agent_beliefs_converge_at_constant_state() {
        // The real variational agent must reduce prediction error when fed
        // the same observation repeatedly (belief convergence). This test
        // fails against the old hand-coded rule table, which had constant
        // "free energy" regardless of belief state.
        let mut agent = ActiveInferenceHelicopterAgent::new();
        let state = HelicopterState::hover(35.0); // Persistent 15m offset
        let first = agent.tick(&state, 20.0).prediction_error;
        let mut last = first;
        for _ in 0..20 {
            last = agent.tick(&state, 20.0).prediction_error;
        }
        assert!(
            last < first,
            "prediction error must shrink as beliefs converge: first {first:.5} -> last {last:.5}"
        );
    }
}
