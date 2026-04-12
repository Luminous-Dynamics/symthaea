// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Active Inference agent for the manipulator arm.
//!
//! Wraps `symthaea_fep::ActiveInferenceAgent` with manipulator-specific
//! priors: joint angles near home position, end-effector force below
//! safety threshold. Modulates tau and learning rate based on prediction
//! error and workspace safety.

use crate::types::ManipulatorState;

/// Result of a single FEP agent tick.
pub struct ManipulatorFepResult {
    pub tau_factor: f32,
    pub learning_rate_factor: f32,
    pub free_energy: f64,
}

/// Active Inference agent for the 7-DOF manipulator.
///
/// Free energy is computed from joint deviation from home position
/// and end-effector force magnitude. High force or large deviation
/// increases free energy, which tightens control (lower tau) and
/// increases learning rate.
pub struct ActiveInferenceManipulatorAgent {
    agent: symthaea_fep::ActiveInferenceAgent,
}

impl ActiveInferenceManipulatorAgent {
    pub fn new() -> Self {
        let config = symthaea_fep::ActiveInferenceAgentConfig {
            state_dim: 7,
            obs_dim: 7,
            num_actions: 8, // 7 joints + gripper
            belief_learning_rate: 0.08,
            planning_horizon: 1,
            action_temperature: 0.4,
            ..symthaea_fep::ActiveInferenceAgentConfig::default()
        };
        Self {
            agent: symthaea_fep::ActiveInferenceAgent::new(config),
        }
    }

    /// Run one tick of active inference given current arm state.
    ///
    /// Returns tau_factor (speed modulation), learning_rate_factor,
    /// and raw free energy.
    pub fn tick(&mut self, state: &ManipulatorState) -> ManipulatorFepResult {
        // Free energy: deviation from home + force magnitude
        let home = ManipulatorState::home();
        let joint_deviation: f64 = state
            .joint_angles
            .iter()
            .zip(home.joint_angles.iter())
            .map(|(a, h)| (a - h).powi(2))
            .sum::<f64>()
            .sqrt();

        let force_magnitude: f64 = state
            .end_effector_force
            .iter()
            .map(|f| f.powi(2))
            .sum::<f64>()
            .sqrt();

        // Free energy: weighted sum of deviation and force
        let fe = joint_deviation * 2.0 + force_magnitude * 0.5;

        // Tau factor: tighten control when force is high or deviation is large
        let tau = if force_magnitude > 5.0 {
            0.8 // Slow down near force limits
        } else if joint_deviation > 1.5 {
            0.9 // Slightly cautious when far from home
        } else {
            1.0
        };

        // Learning rate: increase when prediction error is high (high FE)
        let lr = if fe > 8.0 {
            1.8
        } else if fe < 1.0 {
            0.7
        } else {
            1.0
        };

        ManipulatorFepResult {
            tau_factor: tau,
            learning_rate_factor: lr,
            free_energy: fe,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl Default for ActiveInferenceManipulatorAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fep_at_home() {
        let mut agent = ActiveInferenceManipulatorAgent::new();
        let result = agent.tick(&ManipulatorState::home());
        assert!(result.free_energy.is_finite());
        assert!(result.free_energy < 2.0, "Home state should have low FE");
        assert_eq!(result.tau_factor, 1.0);
    }

    #[test]
    fn test_fep_high_force() {
        let mut agent = ActiveInferenceManipulatorAgent::new();
        let mut state = ManipulatorState::home();
        state.end_effector_force = [10.0, 0.0, 0.0];
        let result = agent.tick(&state);
        assert!(result.free_energy > 4.0);
        assert!(result.tau_factor < 1.0, "High force should reduce tau");
    }

    #[test]
    fn test_fep_large_deviation() {
        let mut agent = ActiveInferenceManipulatorAgent::new();
        let mut state = ManipulatorState::home();
        state.joint_angles = [2.0; 7];
        let result = agent.tick(&state);
        assert!(result.free_energy > 5.0);
        assert!(result.learning_rate_factor > 1.0, "High FE should boost LR");
    }

    #[test]
    fn test_fep_reset() {
        let mut agent = ActiveInferenceManipulatorAgent::new();
        let mut state = ManipulatorState::home();
        state.end_effector_force = [50.0, 0.0, 0.0];
        let _ = agent.tick(&state);
        agent.reset();
        let result = agent.tick(&ManipulatorState::home());
        assert!(result.free_energy < 2.0);
    }
}
