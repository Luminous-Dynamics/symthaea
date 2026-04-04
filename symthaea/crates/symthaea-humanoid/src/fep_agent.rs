// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Active Inference humanoid agent: precision/tau modulation at cognitive rate (10Hz).
//!
//! Wraps `symthaea_fep::ActiveInferenceAgent` with humanoid-specific balance priors.
//! The FEP agent observes balance/locomotion error channels and modulates controller
//! parameters: tau (time constants), learning rate, and encoding attention weights.

use symthaea_fep::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation, TemporalDifferenceLearningConfig,
};

use crate::types::{HumanoidCommand, HumanoidState, HumanoidTask, NUM_ACTUATORS};

/// Result of a cognitive tick from the FEP agent.
#[derive(Debug, Clone)]
pub struct HumanoidFepResult {
    /// Multiply all tau by this (1.0 = no change).
    pub tau_factor: f32,
    /// Multiply learning rate by this.
    pub learning_rate_factor: f32,
    /// Updated prior precision weight.
    pub prior_precision: f64,
    /// Exploration noise (only if exploration triggered).
    pub exploration_noise: Option<[f32; NUM_ACTUATORS]>,
    /// Current free energy.
    pub free_energy: f64,
    /// Current prediction error.
    pub prediction_error: f64,
}

impl Default for HumanoidFepResult {
    fn default() -> Self {
        Self {
            tau_factor: 1.0,
            learning_rate_factor: 1.0,
            prior_precision: 1.0,
            exploration_noise: None,
            free_energy: 0.0,
            prediction_error: 0.0,
        }
    }
}

/// 6 cognitive actions the FEP agent can select.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidAction {
    /// Multiply tau by 0.85 — faster adaptation (falling!).
    DropTau = 0,
    /// Multiply tau by 1.15 — stable standing.
    RaiseTau = 1,
    /// LR x 1.5 — high uncertainty, model outdated.
    BoostLearningRate = 2,
    /// LR x 0.6 — converged, consolidate.
    ReduceLearningRate = 3,
    /// Shift attention to posture channels.
    ShiftToPosture = 4,
    /// Shift attention to locomotion channels.
    ShiftToLocomotion = 5,
}

impl HumanoidAction {
    fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::DropTau,
            1 => Self::RaiseTau,
            2 => Self::BoostLearningRate,
            3 => Self::ReduceLearningRate,
            4 => Self::ShiftToPosture,
            _ => Self::ShiftToLocomotion,
        }
    }
}

/// Configuration for the humanoid FEP agent.
#[derive(Debug, Clone)]
pub struct HumanoidFepConfig {
    /// Number of inference iterations per perception step.
    pub inference_iterations: usize,
    /// Softmax temperature for action selection.
    pub action_temperature: f64,
    /// High FE threshold to trigger exploration.
    pub exploration_fe_threshold: f64,
    /// Cognitive ticks with high FE before exploration triggers.
    pub exploration_patience: usize,
    /// Magnitude of exploration noise.
    pub exploration_magnitude: f32,
    /// Decay rate for high-FE counter when FE drops below threshold.
    /// 0.0 = never decays (pure accumulator), 1.0 ≈ consecutive behavior.
    /// Default: 0.5
    pub exploration_decay_rate: f64,
    /// Enable TD learning.
    pub enable_td_learning: bool,
    /// TD discount factor.
    pub td_discount: f64,
    /// TD eligibility trace lambda.
    pub td_lambda: f64,
    /// Use rule-based policy (default: true).
    pub use_rule_based_policy: bool,
}

impl Default for HumanoidFepConfig {
    fn default() -> Self {
        Self {
            inference_iterations: 5,
            action_temperature: 0.5,
            exploration_fe_threshold: 1.2,
            exploration_patience: 5,
            exploration_magnitude: 0.03,
            exploration_decay_rate: 0.5,
            enable_td_learning: true,
            td_discount: 0.99,
            td_lambda: 0.8,
            use_rule_based_policy: true,
        }
    }
}

/// Active Inference humanoid agent operating at cognitive rate (10Hz).
///
/// At each cognitive tick, the agent:
/// 1. Observes 10D balance/locomotion error vector
/// 2. Updates beliefs via variational inference
/// 3. Selects one of 6 precision-modulation actions
/// 4. Returns modulation parameters for the motor controller
pub struct ActiveInferenceHumanoidAgent {
    /// Inner FEP agent.
    agent: ActiveInferenceAgent,
    /// Configuration.
    config: HumanoidFepConfig,
    /// Leaky accumulator for high-FE ticks (tolerates transient FE dips).
    high_fe_ticks: f64,
    /// Previous head height error for trend detection.
    prev_head_height_error: f64,
    /// EMA of applied tau factors.
    tau_ema: f64,
    /// Current free energy.
    current_fe: f64,
    /// Previous free energy.
    prev_fe: f64,
    /// Current task.
    task: HumanoidTask,
}

impl ActiveInferenceHumanoidAgent {
    /// Create a new humanoid FEP agent.
    pub fn new(config: HumanoidFepConfig, task: HumanoidTask) -> Self {
        let td_config = TemporalDifferenceLearningConfig {
            gamma: config.td_discount,
            lambda: config.td_lambda,
            use_eligibility_traces: true,
            ..TemporalDifferenceLearningConfig::default()
        };

        let agent_config = ActiveInferenceAgentConfig {
            state_dim: 10,
            obs_dim: 10,
            num_actions: 6,
            inference_iterations: config.inference_iterations,
            belief_learning_rate: 0.1,
            planning_horizon: 1,
            action_temperature: config.action_temperature,
            enable_model_learning: true,
            enable_td_learning: config.enable_td_learning,
            td_config,
        };

        let agent = ActiveInferenceAgent::new(agent_config);

        Self {
            agent,
            config,
            high_fe_ticks: 0.0,
            prev_head_height_error: 0.0,
            tau_ema: 1.0,
            current_fe: 0.0,
            prev_fe: 0.0,
            task,
        }
    }

    /// Build the 10D observation vector from humanoid state.
    ///
    /// Channels:
    /// 0. head_height_error — deviation from 1.4m target
    /// 1. uprightness — torso_vertical[2] (1.0 = upright)
    /// 2. com_speed — horizontal COM velocity magnitude
    /// 3. speed_error — deviation from target speed
    /// 4. angular_momentum — total body angular momentum
    /// 5. position_error_trend — worsening vs improving
    /// 6. tau_ema — current tau factor
    /// 7. free_energy_trend — rising vs falling
    /// 8. control_effort — mean |torque|
    /// 9. joint_limit_proximity — how close to mechanical stops
    fn build_observation(&self, state: &HumanoidState, cmd: &HumanoidCommand) -> Vec<f64> {
        let target_speed = self.task.target_speed();

        // 0. Head height error (normalized)
        let head_err = ((1.4 - state.head_height) / 1.4).clamp(0.0, 1.0);

        // 1. Uprightness (already in [0, 1] range for upright)
        let uprightness = state.uprightness().clamp(0.0, 1.0);

        // 2. COM speed (normalized)
        let com_speed = (state.horizontal_speed() / 12.0).clamp(0.0, 1.0);

        // 3. Speed error (normalized)
        let speed_error = ((state.horizontal_speed() - target_speed).abs() / target_speed.max(1.0))
            .clamp(0.0, 1.0);

        // 4. Angular momentum (normalized, high = falling)
        let angular_mom = (state.angular_momentum() / 10.0).clamp(0.0, 1.0);

        // 5. Position error trend (>0.5 = getting worse)
        let pe_trend = if self.prev_head_height_error > 0.0 {
            let current_err = (1.4 - state.head_height).abs();
            let delta = current_err - self.prev_head_height_error;
            ((delta / 0.5) + 0.5).clamp(0.0, 1.0)
        } else {
            0.5
        };

        // 6. Tau EMA (normalized)
        let tau_norm = (self.tau_ema / 3.0).clamp(0.0, 1.0);

        // 7. Free energy trend
        let fe_trend = if self.prev_fe > 0.0 {
            let delta = self.current_fe - self.prev_fe;
            ((delta / 2.0) + 0.5).clamp(0.0, 1.0)
        } else {
            0.5
        };

        // 8. Control effort
        let effort = (cmd.control_effort() as f64).clamp(0.0, 1.0);

        // 9. Joint limit proximity (fraction of joints near limits)
        let near_limit_count = state.joint_angles.iter().filter(|a| a.abs() > 2.0).count();
        let limit_prox = near_limit_count as f64 / NUM_ACTUATORS as f64;

        vec![
            head_err,
            uprightness,
            com_speed,
            speed_error,
            angular_mom,
            pe_trend,
            tau_norm,
            fe_trend,
            effort,
            limit_prox,
        ]
    }

    /// Rule-based action selection from observation channels.
    fn rule_based_action(&self, obs: &[f64]) -> HumanoidAction {
        let head_err = obs[0];
        let angular_mom = obs[4];
        let pe_trend = obs[5];
        let fe_trend = obs[7];

        // Falling: high angular momentum or large head height error trend
        if angular_mom > 0.3 || (pe_trend > 0.6 && head_err > 0.1) {
            HumanoidAction::DropTau
        } else if pe_trend < 0.4 && head_err < 0.05 {
            HumanoidAction::RaiseTau
        } else if fe_trend > 0.65 {
            HumanoidAction::BoostLearningRate
        } else if fe_trend < 0.35 && head_err < 0.1 {
            HumanoidAction::ReduceLearningRate
        } else {
            HumanoidAction::ShiftToPosture
        }
    }

    /// Perform a cognitive tick: observe errors, update beliefs, select action.
    ///
    /// Called at 10Hz (every 4th physics step).
    pub fn step(&mut self, state: &HumanoidState, cmd: &HumanoidCommand) -> HumanoidFepResult {
        self.step_with_encoder_pe(state, cmd, None)
    }

    /// Called at 10Hz with optional encoder-derived prediction error.
    ///
    /// When `encoder_pe` is Some, it's used as the prediction error signal
    /// (from the predictive HDC-LTC encoder layer). This replaces the FEP
    /// agent's internal PE computation with emergent HDC-derived surprise.
    pub fn step_with_encoder_pe(
        &mut self,
        state: &HumanoidState,
        cmd: &HumanoidCommand,
        encoder_pe: Option<f32>,
    ) -> HumanoidFepResult {
        let obs_values = self.build_observation(state, cmd);

        let obs = Observation::new(obs_values.clone(), 1.0, "humanoid");

        let perception = self.agent.perceive(&obs);
        let free_energy = perception.free_energy.total;
        // Use encoder PE if provided, otherwise use FEP agent's internal PE
        let prediction_error = encoder_pe
            .map(|pe| pe as f64)
            .unwrap_or(perception.free_energy.prediction_error);

        let action = if self.config.use_rule_based_policy {
            self.rule_based_action(&obs_values)
        } else {
            let action_result = self.agent.select_action();
            self.agent.act(action_result.action);
            if self.prev_head_height_error > 0.0 {
                self.agent.learn_from_outcome(action_result.action, &obs);
            }
            HumanoidAction::from_index(action_result.action)
        };

        let mut result = HumanoidFepResult {
            free_energy,
            prediction_error,
            prior_precision: self.agent.precision.prior_precision,
            ..HumanoidFepResult::default()
        };

        match action {
            HumanoidAction::DropTau => {
                result.tau_factor = 0.85;
            }
            HumanoidAction::RaiseTau => {
                result.tau_factor = 1.15;
            }
            HumanoidAction::BoostLearningRate => {
                result.learning_rate_factor = 1.5;
            }
            HumanoidAction::ReduceLearningRate => {
                result.learning_rate_factor = 0.6;
            }
            HumanoidAction::ShiftToPosture => {
                result.tau_factor = 1.0;
                result.learning_rate_factor = 1.0;
            }
            HumanoidAction::ShiftToLocomotion => {
                result.tau_factor = 1.0;
                result.learning_rate_factor = 1.0;
            }
        }

        // Exploration patience: leaky accumulator (tolerates transient FE dips)
        if free_energy > self.config.exploration_fe_threshold {
            self.high_fe_ticks += 1.0;
            if self.high_fe_ticks >= self.config.exploration_patience as f64
                && result.exploration_noise.is_none()
            {
                let mag = self.config.exploration_magnitude;
                let noise = [mag * 0.5; NUM_ACTUATORS];
                result.exploration_noise = Some(noise);
            }
        } else {
            // Slow decay instead of hard reset
            self.high_fe_ticks = (self.high_fe_ticks - self.config.exploration_decay_rate).max(0.0);
        }

        // Update tracking state
        self.prev_head_height_error = (1.4 - state.head_height).abs();
        self.tau_ema = 0.8 * self.tau_ema + 0.2 * result.tau_factor as f64;
        self.prev_fe = self.current_fe;
        self.current_fe = free_energy;

        result
    }

    /// Reset the agent (for new episode).
    pub fn reset(&mut self) {
        self.agent.reset();
        self.high_fe_ticks = 0.0;
        self.prev_head_height_error = 0.0;
        self.tau_ema = 1.0;
        self.current_fe = 0.0;
        self.prev_fe = 0.0;
    }

    /// Get the current free energy.
    pub fn current_free_energy(&self) -> f64 {
        self.agent.current_free_energy()
    }

    /// Check if the agent is surprised.
    pub fn is_surprised(&self) -> bool {
        self.agent.is_surprised()
    }

    /// Set the task (for curriculum transitions).
    pub fn set_task(&mut self, task: HumanoidTask) {
        self.task = task;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fep_agent_creation() {
        let config = HumanoidFepConfig::default();
        let agent = ActiveInferenceHumanoidAgent::new(config, HumanoidTask::Stand);
        assert!(!agent.is_surprised());
    }

    #[test]
    fn test_fep_step_at_standing() {
        let config = HumanoidFepConfig::default();
        let mut agent = ActiveInferenceHumanoidAgent::new(config, HumanoidTask::Stand);

        let state = HumanoidState::standing();
        let cmd = HumanoidCommand::zero();

        let result = agent.step(&state, &cmd);
        assert!(result.free_energy.is_finite());
        assert!(result.prediction_error.is_finite());
    }

    #[test]
    fn test_fep_step_falling() {
        let config = HumanoidFepConfig::default();
        let mut agent = ActiveInferenceHumanoidAgent::new(config, HumanoidTask::Stand);

        let mut state = HumanoidState::standing();
        state.head_height = 0.5;
        state.torso_vertical = [0.5, 0.0, 0.5];
        state.root_angular_velocity = [5.0, 3.0, 1.0];

        let cmd = HumanoidCommand::zero();
        let result = agent.step(&state, &cmd);
        assert!(result.free_energy.is_finite());
    }

    #[test]
    fn test_fep_exploration_patience() {
        let config = HumanoidFepConfig {
            exploration_patience: 3,
            exploration_fe_threshold: 0.0, // Always exceeds
            ..HumanoidFepConfig::default()
        };
        let mut agent = ActiveInferenceHumanoidAgent::new(config, HumanoidTask::Stand);

        let mut state = HumanoidState::standing();
        state.head_height = 0.5;
        let cmd = HumanoidCommand::zero();

        let mut had_exploration = false;
        for _ in 0..10 {
            let result = agent.step(&state, &cmd);
            if result.exploration_noise.is_some() {
                had_exploration = true;
            }
        }
        assert!(had_exploration, "Should trigger exploration after patience");
    }

    #[test]
    fn test_fep_reset() {
        let config = HumanoidFepConfig::default();
        let mut agent = ActiveInferenceHumanoidAgent::new(config, HumanoidTask::Stand);

        let state = HumanoidState::standing();
        let cmd = HumanoidCommand::zero();
        agent.step(&state, &cmd);

        agent.reset();
        assert!(agent.high_fe_ticks.abs() < 1e-10);
        assert!((agent.prev_head_height_error - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_humanoid_action_roundtrip() {
        for i in 0..6 {
            let action = HumanoidAction::from_index(i);
            assert_eq!(action as usize, i);
        }
    }

    #[test]
    fn test_fep_result_default() {
        let result = HumanoidFepResult::default();
        assert!((result.tau_factor - 1.0).abs() < 1e-6);
        assert!((result.learning_rate_factor - 1.0).abs() < 1e-6);
        assert!(result.exploration_noise.is_none());
    }
}
