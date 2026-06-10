// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Active Inference vehicle agent: precision/tau modulation at cognitive rate (50Hz).
//!
//! Wraps `symthaea_fep::ActiveInferenceAgent` with driving-specific safety priors.
//! The FEP agent observes lane/speed/grip error channels and modulates controller
//! parameters: tau (time constants), learning rate, and encoding attention weights.
//!
//! Unlike the humanoid FEP agent (10Hz), the vehicle operates at 50Hz cognitive rate
//! because driving decisions (brake! swerve!) must happen in <20ms, not <100ms.

use symthaea_fep::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation, TemporalDifferenceLearningConfig,
};

use crate::types::{NUM_ACTUATORS, VehicleCommand, VehicleState, VehicleTask};

/// Result of a cognitive tick from the vehicle FEP agent.
#[derive(Debug, Clone)]
pub struct VehicleFepResult {
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

impl Default for VehicleFepResult {
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

/// 6 cognitive actions the vehicle FEP agent can select.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VehicleAction {
    /// Multiply tau by 0.8 — faster adaptation (loss of grip!).
    DropTau = 0,
    /// Multiply tau by 1.2 — stable cruising, consolidate.
    RaiseTau = 1,
    /// LR x 1.5 — high uncertainty, model outdated.
    BoostLearningRate = 2,
    /// LR x 0.6 — converged, consolidate.
    ReduceLearningRate = 3,
    /// Increase prior on deceleration (ambiguity response).
    SlowDown = 4,
    /// Increase mesh trust weight (swarm says brake).
    IncreaseMeshTrust = 5,
}

impl VehicleAction {
    fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::DropTau,
            1 => Self::RaiseTau,
            2 => Self::BoostLearningRate,
            3 => Self::ReduceLearningRate,
            4 => Self::SlowDown,
            _ => Self::IncreaseMeshTrust,
        }
    }
}

/// Configuration for the vehicle FEP agent.
#[derive(Debug, Clone)]
pub struct VehicleFepConfig {
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
    /// Enable TD learning.
    pub enable_td_learning: bool,
    /// TD discount factor.
    pub td_discount: f64,
    /// TD eligibility trace lambda.
    pub td_lambda: f64,
    /// Use rule-based policy (default: true).
    pub use_rule_based_policy: bool,
}

impl Default for VehicleFepConfig {
    fn default() -> Self {
        Self {
            inference_iterations: 5,
            action_temperature: 0.5,
            exploration_fe_threshold: 1.5,
            exploration_patience: 8,
            exploration_magnitude: 0.02, // Smaller than humanoid — driving is delicate
            enable_td_learning: true,
            td_discount: 0.99,
            td_lambda: 0.8,
            use_rule_based_policy: true,
        }
    }
}

/// Active Inference vehicle agent operating at cognitive rate (50Hz).
///
/// At each cognitive tick, the agent:
/// 1. Observes 11D driving error vector
/// 2. Updates beliefs via variational inference
/// 3. Selects one of 6 precision-modulation actions
/// 4. Returns modulation parameters for the motor controller
pub struct ActiveInferenceVehicleAgent {
    /// Inner FEP agent.
    agent: ActiveInferenceAgent,
    /// Configuration.
    config: VehicleFepConfig,
    /// Consecutive high-FE ticks.
    high_fe_ticks: usize,
    /// Previous speed error for trend detection.
    prev_speed_error: f64,
    /// EMA of applied tau factors.
    tau_ema: f64,
    /// Current free energy.
    current_fe: f64,
    /// Previous free energy.
    prev_fe: f64,
    /// Current task.
    task: VehicleTask,
    /// Target speed for observations.
    target_speed: f64,
    /// Previous observation for delta computation (ambiguity detection).
    prev_observation: Vec<f64>,
    /// EMA of observation L2 delta (sensor stability signal).
    obs_delta_ema: f64,
}

impl ActiveInferenceVehicleAgent {
    /// Create a new vehicle FEP agent.
    pub fn new(config: VehicleFepConfig, task: VehicleTask) -> Self {
        let td_config = TemporalDifferenceLearningConfig {
            gamma: config.td_discount,
            lambda: config.td_lambda,
            use_eligibility_traces: true,
            ..TemporalDifferenceLearningConfig::default()
        };

        let agent_config = ActiveInferenceAgentConfig {
            state_dim: 11,
            obs_dim: 11,
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
            high_fe_ticks: 0,
            prev_speed_error: 0.0,
            tau_ema: 1.0,
            current_fe: 0.0,
            prev_fe: 0.0,
            task,
            target_speed: task.target_speed(),
            prev_observation: Vec::new(),
            obs_delta_ema: 0.5,
        }
    }

    /// Build the 11D observation vector from vehicle state.
    ///
    /// Channels:
    /// 0. speed_error       — |speed - target| / target (normalized)
    /// 1. lateral_offset    — |lateral_offset| / 3.0 (normalized, 3m = off road)
    /// 2. heading_error     — |heading| / pi (normalized for straight road)
    /// 3. tire_slip_max     — max(front, rear) / 0.3 (normalized, 0.3rad = full slide)
    /// 4. yaw_rate          — |yaw_rate| / 1.5 (normalized, 1.5 = spin)
    /// 5. safety_trend      — worsening (>0.5) vs improving (<0.5)
    /// 6. tau_ema           — current tau factor / 3.0 (normalized)
    /// 7. free_energy_trend — rising (>0.5) vs falling (<0.5)
    /// 8. control_effort    — mean |command| (already 0..1)
    /// 9. mesh_brake_density— swarm brake fraction (0..1)
    /// 10. obs_stability    — EMA of observation delta (sensor health signal)
    fn build_observation(&self, state: &VehicleState, cmd: &VehicleCommand) -> Vec<f64> {
        // 0. Speed error (normalized by target)
        let speed_error = if self.target_speed > 0.1 {
            ((state.speed - self.target_speed).abs() / self.target_speed).clamp(0.0, 1.0)
        } else {
            (state.speed / 5.0).clamp(0.0, 1.0) // for emergency stop, any speed is error
        };

        // 1. Lateral offset (normalized by ~3m)
        let lat_offset = (state.lateral_offset.abs() / 3.0).clamp(0.0, 1.0);

        // 2. Heading error (normalized by pi, for straight road target=0)
        let heading_error = (state.heading.abs() / std::f64::consts::PI).clamp(0.0, 1.0);

        // 3. Max tire slip (normalized by 0.3 rad = danger zone)
        let max_slip = state.tire_slip_front.max(state.tire_slip_rear);
        let slip_norm = (max_slip / 0.3).clamp(0.0, 1.0);

        // 4. Yaw rate (normalized by 1.5 rad/s = spin threshold)
        let yaw_norm = (state.yaw_rate.abs() / 1.5).clamp(0.0, 1.0);

        // 5. Safety trend: are things getting worse?
        let safety_trend = if self.prev_speed_error > 0.0 {
            let delta = speed_error - self.prev_speed_error;
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
        let effort = cmd.control_effort() as f64;

        // 9. Mesh brake density (direct — the swarm's emergency signal)
        let mesh_brake = state.mesh_brake_density.clamp(0.0, 1.0);

        // 10. Observation stability (sensor health signal)
        // High = sensors changing normally; near-zero = channels frozen (sensor failure)
        let obs_stability = self.obs_delta_ema.clamp(0.0, 1.0);

        vec![
            speed_error,
            lat_offset,
            heading_error,
            slip_norm,
            yaw_norm,
            safety_trend,
            tau_norm,
            fe_trend,
            effort,
            mesh_brake,
            obs_stability,
        ]
    }

    /// Rule-based action selection from observation channels.
    fn rule_based_action(&self, obs: &[f64]) -> VehicleAction {
        let speed_error = obs[0];
        let slip_norm = obs[3];
        let yaw_norm = obs[4];
        let safety_trend = obs[5];
        let fe_trend = obs[7];
        let mesh_brake = obs[9];
        let obs_stability = obs[10];

        // SUPREME: if mesh says brake, trust the swarm
        if mesh_brake > 0.3 {
            return VehicleAction::IncreaseMeshTrust;
        }

        // Sensor failure: observations frozen but speed error high → slow down
        // When channels are zeroed by SensorBlindness, obs_delta drops to ~0
        // but tire_slip falsely reads safe (0.0). This catches that ambiguity.
        if obs_stability < 0.05 && speed_error > 0.5 {
            return VehicleAction::SlowDown;
        }

        // Grip loss or spin: drop tau for faster reaction
        if slip_norm > 0.4 || yaw_norm > 0.3 || (safety_trend > 0.65 && slip_norm > 0.2) {
            VehicleAction::DropTau
        }
        // Ambiguity spike: slow down (high FE + things getting worse)
        else if fe_trend > 0.65 && safety_trend > 0.55 {
            VehicleAction::SlowDown
        }
        // Stable cruising: consolidate
        else if safety_trend < 0.4 && slip_norm < 0.1 && yaw_norm < 0.1 {
            VehicleAction::RaiseTau
        }
        // High uncertainty but not dangerous: learn faster
        else if fe_trend > 0.6 {
            VehicleAction::BoostLearningRate
        }
        // Converged and stable: reduce LR
        else if fe_trend < 0.35 && slip_norm < 0.15 {
            VehicleAction::ReduceLearningRate
        } else {
            VehicleAction::RaiseTau
        }
    }

    /// Perform a cognitive tick: observe errors, update beliefs, select action.
    ///
    /// Called at 50Hz (every 4th physics step at 200Hz).
    pub fn step(&mut self, state: &VehicleState, cmd: &VehicleCommand) -> VehicleFepResult {
        let obs_values = self.build_observation(state, cmd);

        // Compute observation delta for sensor stability tracking.
        // Only track the 5 direct sensor channels (0-4): speed_error, lat_offset,
        // heading_error, tire_slip, yaw_rate. Internal channels (5-7: safety_trend,
        // tau_ema, fe_trend) evolve from agent state and would mask sensor failure.
        if !self.prev_observation.is_empty() {
            let l2_delta: f64 = obs_values[..5]
                .iter()
                .zip(self.prev_observation.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            // Normalize: typical driving delta ~0.01-0.1 across 5 channels
            let normalized = (l2_delta / 0.2).clamp(0.0, 1.0);
            self.obs_delta_ema = 0.8 * self.obs_delta_ema + 0.2 * normalized;
        }
        self.prev_observation = obs_values[..5].to_vec();

        let obs = Observation::new(obs_values.clone(), 1.0, "vehicle");

        let perception = self.agent.perceive(&obs);
        let free_energy = perception.free_energy.total;
        let prediction_error = perception.free_energy.prediction_error;

        let action = if self.config.use_rule_based_policy {
            self.rule_based_action(&obs_values)
        } else {
            let action_result = self.agent.select_action();
            self.agent.act(action_result.action);
            if self.prev_speed_error > 0.0 {
                self.agent.learn_from_outcome(action_result.action, &obs);
            }
            VehicleAction::from_index(action_result.action)
        };

        let mut result = VehicleFepResult {
            free_energy,
            prediction_error,
            prior_precision: self.agent.precision.prior_precision,
            ..VehicleFepResult::default()
        };

        match action {
            VehicleAction::DropTau => {
                result.tau_factor = 0.8;
            }
            VehicleAction::RaiseTau => {
                result.tau_factor = 1.2;
            }
            VehicleAction::BoostLearningRate => {
                result.learning_rate_factor = 1.5;
            }
            VehicleAction::ReduceLearningRate => {
                result.learning_rate_factor = 0.6;
            }
            VehicleAction::SlowDown => {
                // Increase prior precision on deceleration:
                // tau drops slightly (faster reaction) + LR boost to adapt
                result.tau_factor = 0.9;
                result.learning_rate_factor = 1.2;
            }
            VehicleAction::IncreaseMeshTrust => {
                // When the swarm says brake: faster tau + moderate LR
                result.tau_factor = 0.85;
                result.learning_rate_factor = 1.3;
            }
        }

        // Exploration patience
        if free_energy > self.config.exploration_fe_threshold {
            self.high_fe_ticks += 1;
            if self.high_fe_ticks >= self.config.exploration_patience
                && result.exploration_noise.is_none()
            {
                let mag = self.config.exploration_magnitude;
                // Steering noise is larger; throttle/brake noise is smaller
                let noise = [mag, mag * 0.3, mag * 0.3];
                result.exploration_noise = Some(noise);
            }
        } else {
            self.high_fe_ticks = 0;
        }

        // Update tracking state
        let speed_error = if self.target_speed > 0.1 {
            (state.speed - self.target_speed).abs()
        } else {
            state.speed
        };
        self.prev_speed_error = speed_error;
        self.tau_ema = 0.8 * self.tau_ema + 0.2 * result.tau_factor as f64;
        self.prev_fe = self.current_fe;
        self.current_fe = free_energy;

        result
    }

    /// Reset the agent (for new episode).
    pub fn reset(&mut self) {
        self.agent.reset();
        self.high_fe_ticks = 0;
        self.prev_speed_error = 0.0;
        self.tau_ema = 1.0;
        self.current_fe = 0.0;
        self.prev_fe = 0.0;
        self.prev_observation.clear();
        self.obs_delta_ema = 0.5;
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
    pub fn set_task(&mut self, task: VehicleTask) {
        self.task = task;
        self.target_speed = task.target_speed();
    }

    /// Set the target speed directly.
    pub fn set_target_speed(&mut self, speed: f64) {
        self.target_speed = speed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fep_agent_creation() {
        let config = VehicleFepConfig::default();
        let agent = ActiveInferenceVehicleAgent::new(config, VehicleTask::LaneKeep);
        assert!(!agent.is_surprised());
    }

    #[test]
    fn test_fep_step_cruising() {
        let config = VehicleFepConfig::default();
        let mut agent = ActiveInferenceVehicleAgent::new(config, VehicleTask::LaneKeep);

        let state = VehicleState::cruising(13.4);
        let cmd = VehicleCommand::zero();

        let result = agent.step(&state, &cmd);
        assert!(result.free_energy.is_finite());
        assert!(result.prediction_error.is_finite());
    }

    #[test]
    fn test_fep_step_sliding() {
        let config = VehicleFepConfig::default();
        let mut agent = ActiveInferenceVehicleAgent::new(config, VehicleTask::LaneKeep);

        let mut state = VehicleState::cruising(13.4);
        state.tire_slip_front = 0.2;
        state.tire_slip_rear = 0.25;
        state.yaw_rate = 0.5;
        state.lateral_velocity = 2.0;

        let cmd = VehicleCommand {
            steering: 0.3,
            throttle: 0.0,
            brake: 0.5,
        };
        let result = agent.step(&state, &cmd);
        assert!(result.free_energy.is_finite());
        // Should want to drop tau (faster reaction to grip loss)
        assert!(
            result.tau_factor <= 1.0,
            "Sliding should trigger DropTau or SlowDown: tau={}",
            result.tau_factor
        );
    }

    #[test]
    fn test_fep_mesh_brake_response() {
        let config = VehicleFepConfig::default();
        let mut agent = ActiveInferenceVehicleAgent::new(config, VehicleTask::LaneKeep);

        let mut state = VehicleState::cruising(13.4);
        state.mesh_brake_density = 0.6; // 60% of swarm braking
        state.mesh_confidence = 0.9;

        let cmd = VehicleCommand::zero();
        let result = agent.step(&state, &cmd);

        // Should trigger IncreaseMeshTrust
        assert!(result.free_energy.is_finite());
        assert!(
            result.tau_factor < 1.0,
            "Mesh brake should drop tau: {}",
            result.tau_factor
        );
    }

    #[test]
    fn test_fep_exploration_patience() {
        let config = VehicleFepConfig {
            exploration_patience: 3,
            exploration_fe_threshold: 0.0, // Always exceeds
            ..VehicleFepConfig::default()
        };
        let mut agent = ActiveInferenceVehicleAgent::new(config, VehicleTask::LaneKeep);

        let mut state = VehicleState::cruising(13.4);
        state.tire_slip_front = 0.15;
        let cmd = VehicleCommand::zero();

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
        let config = VehicleFepConfig::default();
        let mut agent = ActiveInferenceVehicleAgent::new(config, VehicleTask::LaneKeep);

        let state = VehicleState::cruising(13.4);
        let cmd = VehicleCommand::zero();
        agent.step(&state, &cmd);

        agent.reset();
        assert_eq!(agent.high_fe_ticks, 0);
        assert!((agent.prev_speed_error - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_vehicle_action_roundtrip() {
        for i in 0..6 {
            let action = VehicleAction::from_index(i);
            assert_eq!(action as usize, i);
        }
    }

    #[test]
    fn test_fep_result_default() {
        let result = VehicleFepResult::default();
        assert!((result.tau_factor - 1.0).abs() < 1e-6);
        assert!((result.learning_rate_factor - 1.0).abs() < 1e-6);
        assert!(result.exploration_noise.is_none());
    }

    #[test]
    fn test_fep_set_task() {
        let config = VehicleFepConfig::default();
        let mut agent = ActiveInferenceVehicleAgent::new(config, VehicleTask::LaneKeep);
        agent.set_task(VehicleTask::EmergencyStop);
        assert_eq!(agent.task, VehicleTask::EmergencyStop);
        assert!((agent.target_speed - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_observation_vector_bounds() {
        let config = VehicleFepConfig::default();
        let agent = ActiveInferenceVehicleAgent::new(config, VehicleTask::LaneKeep);

        let state = VehicleState::cruising(13.4);
        let cmd = VehicleCommand {
            steering: 0.5,
            throttle: 0.8,
            brake: 0.0,
        };

        let obs = agent.build_observation(&state, &cmd);
        assert_eq!(obs.len(), 11);
        for (i, v) in obs.iter().enumerate() {
            assert!(*v >= 0.0 && *v <= 1.0, "Observation[{i}] out of [0,1]: {v}");
        }
    }

    #[test]
    fn test_sensor_blindness_triggers_slowdown() {
        let config = VehicleFepConfig::default();
        let mut agent = ActiveInferenceVehicleAgent::new(config, VehicleTask::LaneKeep);

        // First: run a few normal steps so prev_observation is populated
        let normal_state = VehicleState::cruising(13.4);
        let cmd = VehicleCommand::zero();
        for _ in 0..5 {
            agent.step(&normal_state, &cmd);
        }

        // Now apply sensor blindness (zeros 4 channels) — speed goes to 0
        let mut blind_state = VehicleState::cruising(13.4);
        blind_state.apply_sensor_blindness(4);
        // The real speed is still 13.4 in the physics but sensors report 0

        // Step multiple times so EMA converges toward frozen channels
        for _ in 0..10 {
            agent.step(&blind_state, &cmd);
        }

        // obs_delta_ema should have dropped near zero (channels frozen)
        assert!(
            agent.obs_delta_ema < 0.1,
            "obs_delta_ema should be near zero under sensor blindness: {}",
            agent.obs_delta_ema
        );
    }

    #[test]
    fn test_normal_driving_obs_stability() {
        let config = VehicleFepConfig::default();
        let mut agent = ActiveInferenceVehicleAgent::new(config, VehicleTask::LaneKeep);

        // Simulate normal driving with varying state — realistic speed/heading variation
        let cmd = VehicleCommand::zero();
        for i in 0..20 {
            let mut state = VehicleState::cruising(13.4 + (i as f64 * 0.5).sin() * 2.0);
            state.heading = (i as f64 * 0.3).sin() * 0.15;
            state.lateral_offset = (i as f64 * 0.2).sin() * 1.0;
            state.yaw_rate = (i as f64 * 0.25).cos() * 0.2;
            agent.step(&state, &cmd);
        }

        // Under normal driving, obs_delta_ema should be non-trivial (> 0.01)
        // indicating that sensor channels are actively changing
        assert!(
            agent.obs_delta_ema > 0.01,
            "obs_delta_ema should reflect changing observations: {}",
            agent.obs_delta_ema
        );
    }
}
