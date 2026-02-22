//! Active Inference flight agent: precision/τ modulation at cognitive rate (25Hz).
//!
//! Wraps `symthaea_fep::ActiveInferenceAgent` with flight-specific precision
//! modulation. Instead of injecting random exploration noise, the FEP agent
//! modulates controller parameters: τ (time constants), learning rate, and
//! prior precision — true Active Inference as precision-weighted meta-learning.

use symthaea_fep::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation,
    TemporalDifferenceLearningConfig,
};

use crate::types::{FlightState, FlightSetpoint, QuadrotorCommand};

/// Result of a cognitive tick from the FEP agent.
#[derive(Debug, Clone)]
pub struct FlightFepResult {
    /// Multiply all τ by this (1.0 = no change).
    pub tau_factor: f32,
    /// Multiply learning rate by this.
    pub learning_rate_factor: f32,
    /// Updated prior precision weight.
    pub prior_precision: f64,
    /// Exploration noise (only if exploration triggered).
    pub exploration_noise: Option<QuadrotorCommand>,
    /// Current free energy.
    pub free_energy: f64,
    /// Current prediction error.
    pub prediction_error: f64,
}

impl Default for FlightFepResult {
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
pub enum FlightAction {
    /// Multiply τ by 0.85 — faster adaptation (wind gust, rotor loss).
    DropTau = 0,
    /// Multiply τ by 1.15 — more stable (low surprise hover).
    RaiseTau = 1,
    /// LR × 1.5 — high epistemic uncertainty, model outdated.
    BoostLearningRate = 2,
    /// LR × 0.6 — low error, consolidate patterns.
    ReduceLearningRate = 3,
    /// Weight position vs attitude errors differently.
    ShiftAttention = 4,
    /// Add bounded noise to commands — last resort after τ/LR modulation fails.
    ExplorationBurst = 5,
}

impl FlightAction {
    fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::DropTau,
            1 => Self::RaiseTau,
            2 => Self::BoostLearningRate,
            3 => Self::ReduceLearningRate,
            4 => Self::ShiftAttention,
            _ => Self::ExplorationBurst,
        }
    }
}

/// Configuration for the flight FEP agent.
#[derive(Debug, Clone)]
pub struct FlightFepConfig {
    /// Number of inference iterations per perception step.
    pub inference_iterations: usize,
    /// Softmax temperature for action selection (lower = more deterministic).
    pub action_temperature: f64,
    /// High FE threshold to trigger exploration (after patience runs out).
    pub exploration_fe_threshold: f64,
    /// Number of cognitive ticks with high FE before exploration triggers.
    pub exploration_patience: usize,
    /// Magnitude of exploration noise (fraction of max command range).
    pub exploration_magnitude: f32,
    /// Whether to enable temporal difference learning (default: true).
    pub enable_td_learning: bool,
    /// TD discount factor γ (default: 0.99).
    pub td_discount: f64,
    /// TD eligibility trace λ (default: 0.8).
    pub td_lambda: f64,
    /// Use rule-based policy instead of softmax action selection (default: true).
    /// Rules use observation channels directly for reliable modulation.
    pub use_rule_based_policy: bool,
}

impl Default for FlightFepConfig {
    fn default() -> Self {
        Self {
            inference_iterations: 5,
            action_temperature: 0.5,
            exploration_fe_threshold: 2.0,
            exploration_patience: 20,    // Wait longer before injecting noise
            exploration_magnitude: 0.02, // Subtle noise — avoid destabilizing hover
            enable_td_learning: true,
            td_discount: 0.99,
            td_lambda: 0.8,
            use_rule_based_policy: true,
        }
    }
}

/// Active Inference flight agent operating at cognitive rate (25Hz).
///
/// At each cognitive tick, the agent:
/// 1. Observes normalized error channels (position, attitude, velocity)
/// 2. Updates beliefs via variational inference
/// 3. Selects one of 6 precision-modulation actions
/// 4. Returns modulation parameters for the motor controller
pub struct ActiveInferenceFlightAgent {
    /// Inner FEP agent.
    agent: ActiveInferenceAgent,
    /// Configuration.
    config: FlightFepConfig,
    /// Consecutive high-FE ticks (for exploration patience).
    high_fe_ticks: usize,
    /// Running prediction error for trend detection.
    prev_prediction_error: f64,
    /// Exponential moving average of applied tau factors.
    tau_ema: f64,
    /// Current free energy (for surprise rate computation).
    current_fe: f64,
    /// Previous free energy.
    prev_fe: f64,
}

impl ActiveInferenceFlightAgent {
    /// Create a new flight FEP agent.
    pub fn new(config: FlightFepConfig) -> Self {
        let td_config = TemporalDifferenceLearningConfig {
            gamma: config.td_discount,
            lambda: config.td_lambda,
            use_eligibility_traces: true,
            ..TemporalDifferenceLearningConfig::default()
        };

        let agent_config = ActiveInferenceAgentConfig {
            state_dim: 8,       // expanded observation space
            obs_dim: 8,
            num_actions: 6,     // The 6 FlightActions
            inference_iterations: config.inference_iterations,
            belief_learning_rate: 0.1,
            planning_horizon: 1, // Single-step (fast cognitive tick)
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
            prev_prediction_error: 0.0,
            tau_ema: 1.0,
            current_fe: 0.0,
            prev_fe: 0.0,
        }
    }

    /// Build the 8D observation vector from flight state.
    fn build_observation(&self, state: &FlightState, setpoint: &FlightSetpoint) -> Vec<f64> {
        let pos_err = setpoint.position_error_magnitude(state);
        let (roll, pitch, _yaw) = state.euler_angles();
        let att_err = (roll * roll + pitch * pitch).sqrt();
        let vel_err = state.speed();

        // Original 3 channels
        let norm_pos = (pos_err / 1.0).min(1.0);
        let norm_att = (att_err / 1.0).min(1.0);
        let norm_vel = (vel_err / 5.0).min(1.0);

        // New channels
        let pe_trend = if self.prev_prediction_error > 0.0 {
            let delta = pos_err - self.prev_prediction_error;
            ((delta / 0.5) + 0.5).clamp(0.0, 1.0) // >0.5 = getting worse
        } else {
            0.5
        };
        let tau_ema_norm = (self.tau_ema / 3.0).clamp(0.0, 1.0);
        let surprise_rate = if self.prev_fe > 0.0 {
            let delta = self.current_fe - self.prev_fe;
            ((delta / 2.0) + 0.5).clamp(0.0, 1.0)
        } else {
            0.5
        };
        let alt_err_signed = {
            let raw = state.altitude() - setpoint.position[2]; // positive = above
            ((raw / 1.0) + 0.5).clamp(0.0, 1.0)
        };
        let angular_speed = (state.angular_speed() / 20.0).min(1.0);

        vec![
            norm_pos, norm_att, norm_vel,
            pe_trend, tau_ema_norm, surprise_rate, alt_err_signed, angular_speed,
        ]
    }

    /// Rule-based action selection from observation channels.
    ///
    /// Uses direct thresholds on the 8D observation vector:
    /// - `obs[0]` = norm_pos (position error, 0=perfect, 1=bad)
    /// - `obs[3]` = pe_trend (>0.5 = getting worse, <0.5 = improving)
    /// - `obs[5]` = surprise_rate (>0.5 = FE rising, <0.5 = FE falling)
    fn rule_based_action(&self, obs: &[f64]) -> FlightAction {
        let norm_pos = obs[0];
        let pe_trend = obs[3];
        let surprise_rate = obs[5];

        if pe_trend > 0.6 && norm_pos > 0.1 {
            // Error growing and significant → speed up adaptation
            FlightAction::DropTau
        } else if pe_trend < 0.4 && norm_pos < 0.05 {
            // Error shrinking and small → consolidate, slow down
            FlightAction::RaiseTau
        } else if surprise_rate > 0.65 {
            // Free energy rising → model outdated, boost learning
            FlightAction::BoostLearningRate
        } else if surprise_rate < 0.35 && norm_pos < 0.1 {
            // FE falling and low error → reduce learning, stabilize
            FlightAction::ReduceLearningRate
        } else {
            // Neutral — no modulation
            FlightAction::ShiftAttention
        }
    }

    /// Perform a cognitive tick: observe errors, update beliefs, select action.
    ///
    /// Called at 25Hz (every 20th motor step).
    pub fn step(
        &mut self,
        state: &FlightState,
        setpoint: &FlightSetpoint,
    ) -> FlightFepResult {
        let obs_values = self.build_observation(state, setpoint);

        let obs = Observation::new(
            obs_values.clone(),
            1.0, // Full precision
            "flight",
        );

        // Perceive: update beliefs
        let perception = self.agent.perceive(&obs);
        let free_energy = perception.free_energy.total;
        let prediction_error = perception.free_energy.prediction_error;

        // Select action: rule-based policy or softmax
        let action = if self.config.use_rule_based_policy {
            self.rule_based_action(&obs_values)
        } else {
            let action_result = self.agent.select_action();
            self.agent.act(action_result.action);
            // Learn from the transition if we have previous state
            if self.prev_prediction_error > 0.0 {
                self.agent.learn_from_outcome(action_result.action, &obs);
            }
            FlightAction::from_index(action_result.action)
        };

        // Build result based on selected action
        let mut result = FlightFepResult {
            free_energy,
            prediction_error,
            prior_precision: self.agent.precision.prior_precision,
            ..FlightFepResult::default()
        };

        match action {
            FlightAction::DropTau => {
                result.tau_factor = 0.85;
            }
            FlightAction::RaiseTau => {
                result.tau_factor = 1.15;
            }
            FlightAction::BoostLearningRate => {
                result.learning_rate_factor = 1.5;
            }
            FlightAction::ReduceLearningRate => {
                result.learning_rate_factor = 0.6;
            }
            FlightAction::ShiftAttention => {
                result.tau_factor = 1.0;
                result.learning_rate_factor = 1.0;
            }
            FlightAction::ExplorationBurst => {
                let mag = self.config.exploration_magnitude;
                result.exploration_noise = Some(QuadrotorCommand {
                    thrust: mag * QuadrotorCommand::MAX_THRUST * 0.1,
                    roll_moment: mag * QuadrotorCommand::MAX_MOMENT_RP,
                    pitch_moment: mag * QuadrotorCommand::MAX_MOMENT_RP,
                    yaw_moment: mag * QuadrotorCommand::MAX_MOMENT_YAW,
                });
            }
        }

        // Exploration patience: override to exploration if FE stays high
        if free_energy > self.config.exploration_fe_threshold {
            self.high_fe_ticks += 1;
            if self.high_fe_ticks >= self.config.exploration_patience
                && result.exploration_noise.is_none()
            {
                let mag = self.config.exploration_magnitude;
                result.exploration_noise = Some(QuadrotorCommand {
                    thrust: mag * QuadrotorCommand::MAX_THRUST * 0.05,
                    roll_moment: mag * QuadrotorCommand::MAX_MOMENT_RP * 0.5,
                    pitch_moment: mag * QuadrotorCommand::MAX_MOMENT_RP * 0.5,
                    yaw_moment: mag * QuadrotorCommand::MAX_MOMENT_YAW * 0.5,
                });
            }
        } else {
            self.high_fe_ticks = 0;
        }

        // Update tracking state
        self.prev_prediction_error = setpoint.position_error_magnitude(state);
        self.tau_ema = 0.8 * self.tau_ema + 0.2 * result.tau_factor as f64;
        self.prev_fe = self.current_fe;
        self.current_fe = free_energy;

        result
    }

    /// Initial step: prime the agent with starting state.
    pub fn initial_step(
        &mut self,
        state: &FlightState,
        setpoint: &FlightSetpoint,
    ) -> FlightFepResult {
        self.step(state, setpoint)
    }

    /// Reset the agent (for new episode).
    pub fn reset(&mut self) {
        self.agent.reset();
        self.high_fe_ticks = 0;
        self.prev_prediction_error = 0.0;
        self.tau_ema = 1.0;
        self.current_fe = 0.0;
        self.prev_fe = 0.0;
    }

    /// Get the current free energy.
    pub fn current_free_energy(&self) -> f64 {
        self.agent.current_free_energy()
    }

    /// Check if the agent is surprised (high prediction error).
    pub fn is_surprised(&self) -> bool {
        self.agent.is_surprised()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fep_agent_creation() {
        let config = FlightFepConfig::default();
        let agent = ActiveInferenceFlightAgent::new(config);
        assert!(!agent.is_surprised());
    }

    #[test]
    fn test_fep_step_at_hover() {
        let config = FlightFepConfig::default();
        let mut agent = ActiveInferenceFlightAgent::new(config);

        let state = FlightState::hover(0.1);
        let setpoint = FlightSetpoint::hover();

        let result = agent.step(&state, &setpoint);
        // At hover (low error): should tend toward stability
        assert!(result.free_energy.is_finite());
        assert!(result.prediction_error.is_finite());
    }

    #[test]
    fn test_fep_step_high_error() {
        let config = FlightFepConfig::default();
        let mut agent = ActiveInferenceFlightAgent::new(config);

        // Far from setpoint
        let state = FlightState {
            position: [1.0, 1.0, 0.0],
            quaternion: [0.707, 0.707, 0.0, 0.0], // Tilted
            linear_velocity: [2.0, 2.0, -1.0],
            angular_velocity: [5.0, 5.0, 5.0],
            timestamp: 0.0,
        };
        let setpoint = FlightSetpoint::hover();

        let result = agent.step(&state, &setpoint);
        assert!(result.free_energy.is_finite());
        assert!(result.prediction_error.is_finite());
    }

    #[test]
    fn test_fep_exploration_patience() {
        let config = FlightFepConfig {
            exploration_patience: 3,
            exploration_fe_threshold: 0.0, // Always exceeds
            ..FlightFepConfig::default()
        };
        let mut agent = ActiveInferenceFlightAgent::new(config);

        let state = FlightState {
            position: [1.0, 1.0, 0.0],
            linear_velocity: [2.0, 2.0, -1.0],
            ..FlightState::hover(0.0)
        };
        let setpoint = FlightSetpoint::hover();

        // Step until patience runs out
        let mut had_exploration = false;
        for _ in 0..10 {
            let result = agent.step(&state, &setpoint);
            if result.exploration_noise.is_some() {
                had_exploration = true;
            }
        }
        assert!(had_exploration, "Should eventually trigger exploration after patience exhausted");
    }

    #[test]
    fn test_fep_reset() {
        let config = FlightFepConfig::default();
        let mut agent = ActiveInferenceFlightAgent::new(config);

        let state = FlightState::hover(0.1);
        let setpoint = FlightSetpoint::hover();
        agent.step(&state, &setpoint);

        agent.reset();
        assert_eq!(agent.high_fe_ticks, 0);
        assert!((agent.prev_prediction_error - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_flight_action_roundtrip() {
        for i in 0..6 {
            let action = FlightAction::from_index(i);
            assert_eq!(action as usize, i);
        }
    }

    #[test]
    fn test_fep_result_default() {
        let result = FlightFepResult::default();
        assert!((result.tau_factor - 1.0).abs() < 1e-6);
        assert!((result.learning_rate_factor - 1.0).abs() < 1e-6);
        assert!(result.exploration_noise.is_none());
    }
}
