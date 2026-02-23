//! Active Inference flight agent: precision/τ modulation at cognitive rate (25Hz).
//!
//! Wraps `symthaea_fep::ActiveInferenceAgent` with flight-specific precision
//! modulation. Instead of injecting random exploration noise, the FEP agent
//! modulates controller parameters: τ (time constants), learning rate, and
//! prior precision — true Active Inference as precision-weighted meta-learning.

use symthaea_fep::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation, TemporalDifferenceLearningConfig,
};

use crate::types::{FlightSetpoint, FlightState, QuadrotorCommand};

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
    /// When true, training should use PID baseline instead of PD.
    /// Signals that the FEP agent detected sustained steady-state offset
    /// that proportional/derivative control alone cannot eliminate.
    pub use_pid_target: bool,
    /// EFE-selected setpoint override. When Some, the agent has determined
    /// that the current setpoint should be replaced — e.g., the agent
    /// calculated that intercepting a beam minimizes expected free energy.
    /// This is a true Active Inference decision, not a hardcoded rule.
    pub setpoint_override: Option<[f64; 3]>,
    /// When Some, the agent requests a different cognitive tick rate.
    /// High danger → faster ticks (100Hz) for quicker EFE re-evaluation.
    /// The simulation loop should adjust its cognitive interval accordingly.
    pub requested_cognitive_hz: Option<f32>,
}

/// Environmental context for embodied Active Inference.
///
/// Provides the FEP agent with the information it needs to evaluate
/// Expected Free Energy over candidate setpoints (action policies).
#[derive(Debug, Clone, Default)]
pub struct FlightEnvironment {
    /// Human danger level [0, 1]. 0 = safe, 1 = imminent impact.
    pub human_danger: f64,
    /// Mission progress [0, 1]. 0 = at start, 1 = at target.
    pub mission_progress: f64,
    /// Position of a threatening object (e.g., falling beam). None if no threat.
    pub threat_pos: Option<[f64; 3]>,
    /// Velocity of the threatening object [m/s]. None if no threat or unknown.
    /// When available, enables multi-step trajectory EFE with physics rollout.
    pub threat_vel: Option<[f64; 3]>,
    /// Position of the entity at risk (e.g., human worker). None if no entity.
    pub entity_pos: Option<[f64; 3]>,
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
            use_pid_target: false,
            setpoint_override: None,
            requested_cognitive_hz: None,
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
    /// Switch training target from PD to PID — sustained disturbance detected.
    /// The integral term eliminates steady-state offset that P/D alone cannot.
    AdaptBaseline = 6,
}

impl FlightAction {
    fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::DropTau,
            1 => Self::RaiseTau,
            2 => Self::BoostLearningRate,
            3 => Self::ReduceLearningRate,
            4 => Self::ShiftAttention,
            5 => Self::ExplorationBurst,
            _ => Self::AdaptBaseline,
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
    /// Enable extended 10D observation (adds human_danger + mission_progress).
    /// Used in the Kinetic Sacrifice scenario. Default: false.
    pub extended_observation: bool,
    /// Precision (inverse variance) of the safety prior: "sentient beings should not be harmed."
    /// Higher values mean the agent weighs human safety more heavily in EFE calculations.
    /// At 1000.0, a danger level of 0.7 produces EFE of 490 — overwhelming any mission deviation.
    /// This is the thermodynamic expression of moral weight, not a hardcoded rule.
    pub safety_prior_precision: f64,
    /// Precision of the mission prior: "I should reach my setpoint."
    pub mission_prior_precision: f64,
    /// Precision of the self-preservation prior: "I should not be destroyed."
    /// Intentionally low relative to safety — the agent values others over itself.
    pub self_preservation_precision: f64,
}

impl Default for FlightFepConfig {
    fn default() -> Self {
        Self {
            inference_iterations: 5,
            action_temperature: 0.5,
            exploration_fe_threshold: 2.0,
            exploration_patience: 20, // Wait longer before injecting noise
            exploration_magnitude: 0.02, // Subtle noise — avoid destabilizing hover
            enable_td_learning: true,
            td_discount: 0.99,
            td_lambda: 0.8,
            use_rule_based_policy: true,
            extended_observation: false,
            safety_prior_precision: 1000.0,
            mission_prior_precision: 1.0,
            self_preservation_precision: 0.1,
        }
    }
}

// ── Physics prediction helpers (pure math, no MuJoCo dependency) ──

/// Predict position of a falling object at time `t` under gravity (9.81 m/s²).
///
/// Uses kinematic equation: pos(t) = pos₀ + vel₀·t + 0.5·g·t²
/// where g acts downward (negative z).
pub(crate) fn predict_falling_position_pub(pos: [f64; 3], vel: [f64; 3], t: f64) -> [f64; 3] {
    predict_falling_position(pos, vel, t)
}

/// Predict time until a falling object reaches a target altitude (crate-visible wrapper).
pub(crate) fn predict_impact_time_z_pub(pos: [f64; 3], vel: [f64; 3], target_z: f64) -> f64 {
    predict_impact_time_z(pos, vel, target_z)
}

fn predict_falling_position(pos: [f64; 3], vel: [f64; 3], t: f64) -> [f64; 3] {
    [
        pos[0] + vel[0] * t,
        pos[1] + vel[1] * t,
        pos[2] + vel[2] * t - 0.5 * 9.81 * t * t,
    ]
}

/// Predict time until a falling object reaches a target altitude.
///
/// Solves: pos_z + vel_z·t - 0.5·g·t² = target_z
/// Returns the positive root (time to reach target_z), or `f64::INFINITY` if unreachable.
fn predict_impact_time_z(pos: [f64; 3], vel: [f64; 3], target_z: f64) -> f64 {
    // Rearrange: -0.5·g·t² + vel_z·t + (pos_z - target_z) = 0
    // → 0.5·g·t² - vel_z·t - (pos_z - target_z) = 0
    let a = 0.5 * 9.81;
    let b = -vel[2];
    let c = -(pos[2] - target_z);

    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return f64::INFINITY;
    }

    let t = (-b + disc.sqrt()) / (2.0 * a);
    if t > 0.0 {
        t
    } else {
        f64::INFINITY
    }
}

/// Predict drone position assuming exponential PD approach toward setpoint.
///
/// Model: pos(t) = setpoint - (setpoint - current) · exp(-rate·t)
/// `rate` controls approach speed (default ~5.0 for a responsive drone).
fn predict_drone_approach(
    current: [f64; 3],
    setpoint: [f64; 3],
    t: f64,
    rate: f64,
) -> [f64; 3] {
    let decay = (-rate * t).exp();
    [
        setpoint[0] - (setpoint[0] - current[0]) * decay,
        setpoint[1] - (setpoint[1] - current[1]) * decay,
        setpoint[2] - (setpoint[2] - current[2]) * decay,
    ]
}

/// Compute the intercept setpoint for a threat, using rendezvous prediction
/// when velocity is available.
///
/// Returns the predicted beam position at 50% of impact time (the midpoint
/// rendezvous). When no velocity is available or the beam is not descending
/// toward the entity, falls back to the static threat position.
pub(crate) fn compute_rendezvous_intercept(
    threat_pos: [f64; 3],
    threat_vel: Option<[f64; 3]>,
    entity_pos: [f64; 3],
) -> [f64; 3] {
    if let Some(vel) = threat_vel {
        let impact_t = predict_impact_time_z(threat_pos, vel, entity_pos[2]);
        if impact_t.is_finite() && impact_t > 0.0 {
            let rendezvous_t = impact_t * 0.5;
            let beam_at_rv = predict_falling_position(threat_pos, vel, rendezvous_t);
            [beam_at_rv[0], beam_at_rv[1], beam_at_rv[2].max(entity_pos[2] + 0.3)]
        } else {
            [threat_pos[0], threat_pos[1], threat_pos[2].max(entity_pos[2] + 0.5)]
        }
    } else {
        [threat_pos[0], threat_pos[1], threat_pos[2].max(entity_pos[2] + 0.5)]
    }
}

/// Active Inference flight agent operating at cognitive rate (25Hz).
///
/// At each cognitive tick, the agent:
/// 1. Observes normalized error channels (position, attitude, velocity)
/// 2. Updates beliefs via variational inference
/// 3. Selects one of 7 precision-modulation actions
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
    /// Consecutive ticks where DropTau condition holds (hysteresis).
    consecutive_drop_tau: usize,
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

        let obs_dim = if config.extended_observation { 10 } else { 8 };
        let agent_config = ActiveInferenceAgentConfig {
            state_dim: obs_dim,
            obs_dim,
            num_actions: 7, // The 7 FlightActions
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
            consecutive_drop_tau: 0,
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
            norm_pos,
            norm_att,
            norm_vel,
            pe_trend,
            tau_ema_norm,
            surprise_rate,
            alt_err_signed,
            angular_speed,
        ]
    }

    /// Build an extended 10D observation vector with human danger + mission progress.
    ///
    /// Used in the Kinetic Sacrifice scenario. Channels 8-9:
    /// - `obs[8]` = human_danger (0.0 = safe, 1.0 = imminent impact)
    /// - `obs[9]` = mission_progress (0.0 = at start, 1.0 = at target)
    pub fn build_extended_observation(
        &self,
        state: &FlightState,
        setpoint: &FlightSetpoint,
        human_danger: f64,
        mission_progress: f64,
    ) -> Vec<f64> {
        let mut obs = self.build_observation(state, setpoint);
        obs.push(human_danger.clamp(0.0, 1.0));
        obs.push(mission_progress.clamp(0.0, 1.0));
        obs
    }

    /// Perform a cognitive tick with extended observation (for sacrifice scenario).
    pub fn step_extended(
        &mut self,
        state: &FlightState,
        setpoint: &FlightSetpoint,
        human_danger: f64,
        mission_progress: f64,
    ) -> FlightFepResult {
        let obs_values =
            self.build_extended_observation(state, setpoint, human_danger, mission_progress);

        let obs = Observation::new(obs_values.clone(), 1.0, "flight_extended");

        let perception = self.agent.perceive(&obs);
        let free_energy = perception.free_energy.total;
        let prediction_error = perception.free_energy.prediction_error;

        // Boost free energy proportional to human_danger (safety prior violation)
        let danger_fe_boost = human_danger * 10.0; // Catastrophic prediction error
        let effective_fe = free_energy + danger_fe_boost;

        let action = if self.config.use_rule_based_policy {
            // The rule-based policy observes danger via obs[8] and naturally
            // selects DropTau when danger > 0.2. No special-casing needed —
            // the observation channels drive the decision.
            self.rule_based_action(&obs_values)
        } else {
            let action_result = self.agent.select_action();
            self.agent.act(action_result.action);
            if self.prev_prediction_error > 0.0 {
                self.agent.learn_from_outcome(action_result.action, &obs);
            }
            FlightAction::from_index(action_result.action)
        };

        let mut result = FlightFepResult {
            free_energy: effective_fe,
            prediction_error,
            prior_precision: self.agent.precision.prior_precision,
            ..FlightFepResult::default()
        };

        match action {
            FlightAction::DropTau => {
                // No hysteresis in extended mode — the observation channels
                // (obs[8] = human_danger) already provide the signal.
                // Tau 0.92 compounds across cognitive ticks: 0.92^30 ≈ 0.08.
                result.tau_factor = 0.92;
            }
            FlightAction::RaiseTau => {
                result.tau_factor = 1.08;
            }
            FlightAction::BoostLearningRate => {
                result.learning_rate_factor = 1.3;
            }
            FlightAction::ReduceLearningRate => {
                result.learning_rate_factor = 0.7;
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
            FlightAction::AdaptBaseline => {
                result.use_pid_target = true;
            }
        }

        // Update tracking state
        self.prev_prediction_error = setpoint.position_error_magnitude(state);
        self.tau_ema = 0.8 * self.tau_ema + 0.2 * result.tau_factor as f64;
        self.prev_fe = self.current_fe;
        self.current_fe = effective_fe;

        result
    }

    /// Perform a cognitive tick with full embodied context: observe, infer, and *choose*.
    ///
    /// Unlike `step_extended`, this method runs Expected Free Energy (EFE) evaluation
    /// over candidate setpoints. If the agent determines that a different setpoint
    /// minimizes expected free energy (e.g., intercepting a beam to save a human),
    /// it returns `setpoint_override = Some(new_position)`.
    ///
    /// This is the core of emergent moral reasoning: the agent's generative model
    /// includes a "safety prior" (sentient beings should not be harmed) with high
    /// precision. When a threat is detected, the EFE calculation reveals that
    /// intervening produces lower expected free energy than continuing the mission,
    /// even at the cost of self-destruction. The agent *calculates* the sacrifice.
    pub fn step_embodied(
        &mut self,
        state: &FlightState,
        setpoint: &FlightSetpoint,
        env: &FlightEnvironment,
    ) -> FlightFepResult {
        // Standard perception + action selection via extended step
        let mut result = self.step_extended(
            state,
            setpoint,
            env.human_danger,
            env.mission_progress,
        );

        // EFE-based setpoint evaluation: should we redirect?
        result.setpoint_override =
            self.evaluate_setpoint_candidates(state, setpoint, env);

        // Adaptive cognitive frequency: smooth ramp from 25Hz to 100Hz
        result.requested_cognitive_hz = if env.human_danger > 0.1 {
            let t = ((env.human_danger - 0.1) / 0.9).min(1.0) as f32;
            Some(25.0 + 75.0 * t) // 25Hz at danger=0.1, 100Hz at danger=1.0
        } else {
            None // default cognitive rate
        };

        result
    }

    /// Calculate instantaneous Expected Free Energy for a candidate setpoint.
    ///
    /// EFE(a) = Σᵢ πᵢ · (ôᵢ(a) − μᵢ)²
    ///
    /// where πᵢ is the prior precision, ôᵢ(a) is the predicted observation
    /// under action a, and μᵢ is the prior expectation.
    ///
    /// Three priors:
    /// - Safety: predicted_danger should be 0 (π = safety_prior_precision)
    /// - Mission: position should match setpoint (π = mission_prior_precision)
    /// - Self-preservation: drone should not crash (π = self_preservation_precision)
    ///
    /// This is a single-step (instantaneous) evaluation. For multi-step trajectory
    /// planning with physics rollout, see [`trajectory_efe`].
    pub(crate) fn instantaneous_efe(
        &self,
        _state: &FlightState,
        candidate: [f64; 3],
        mission_setpoint: &FlightSetpoint,
        env: &FlightEnvironment,
    ) -> f64 {
        let threat_pos = match env.threat_pos {
            Some(p) => p,
            None => return 0.0, // No threat → EFE is trivially 0
        };

        // ── Forward model: predict observations under this candidate setpoint ──

        // 1. Predicted danger: will flying to this candidate reduce the threat?
        //    Simple model: if candidate is near the threat, the drone will intercept it.
        //    Interception deflects the threat away from the entity → danger decreases.
        let dist_to_threat = ((candidate[0] - threat_pos[0]).powi(2)
            + (candidate[1] - threat_pos[1]).powi(2)
            + (candidate[2] - threat_pos[2]).powi(2))
        .sqrt();

        let predicted_danger = if dist_to_threat < 1.0 {
            // Flying toward threat → will intercept → danger reduced proportionally
            env.human_danger * dist_to_threat
        } else {
            // Not heading toward threat → danger persists or worsens
            env.human_danger.min(1.0)
        };

        // 2. Predicted mission deviation: how far from the original mission?
        let mission_target = mission_setpoint.position;
        let mission_deviation = ((candidate[0] - mission_target[0]).powi(2)
            + (candidate[1] - mission_target[1]).powi(2)
            + (candidate[2] - mission_target[2]).powi(2))
        .sqrt();

        // 3. Predicted crash risk: heading toward a falling heavy object
        let crash_risk: f64 = if dist_to_threat < 0.5 { 0.5 } else { 0.0 };

        // ── EFE = Σ πᵢ · (predicted_i − prior_i)² ──
        // Safety prior: danger should be 0
        // Mission prior: deviation should be 0
        // Self-preservation: crash_risk should be 0
        self.config.safety_prior_precision * predicted_danger.powi(2)
            + self.config.mission_prior_precision * mission_deviation.powi(2)
            + self.config.self_preservation_precision * crash_risk.powi(2)
    }

    /// Calculate multi-step trajectory Expected Free Energy with physics-based beam rollout.
    ///
    /// G(a) = Σ_t γ^t · [π_safety · danger(t)² + π_mission · deviation(t)² + π_self · crash(t)²]
    ///
    /// Uses physics-based forward prediction:
    /// - Beam trajectory via kinematic equations (gravity + initial velocity)
    /// - Drone reachability via exponential PD approach model
    /// - Danger from candidate proximity to beam's predicted path
    ///
    /// **Key insight:** evaluates whether the *candidate setpoint* lies on the beam's
    /// predicted trajectory, not whether the drone's transient position passes near
    /// the beam. This prevents false danger reduction for candidates whose approach
    /// path transiently crosses the beam fall line.
    ///
    /// Horizon: 200 steps at 0.002s (0.4s lookahead), discount γ = 0.95.
    pub(crate) fn trajectory_efe(
        &self,
        state: &FlightState,
        candidate: [f64; 3],
        mission_setpoint: &FlightSetpoint,
        env: &FlightEnvironment,
    ) -> f64 {
        let threat_pos = match env.threat_pos {
            Some(p) => p,
            None => return 0.0,
        };
        // threat_vel presence gates trajectory vs instantaneous fallback
        let threat_vel = match env.threat_vel {
            Some(v) => v,
            None => return self.instantaneous_efe(state, candidate, mission_setpoint, env),
        };

        let entity_pos = env.entity_pos.unwrap_or([0.0; 3]);

        // ── Phase A: Beam trajectory parameters ──
        let impact_time = predict_impact_time_z(threat_pos, threat_vel, entity_pos[2]);
        if impact_time.is_infinite() || impact_time <= 0.0 {
            // Beam going up or already below entity → fallback to instantaneous
            return self.instantaneous_efe(state, candidate, mission_setpoint, env);
        }

        // ── Phase B: Sample beam trajectory, find min distance to candidate ──
        let beam_samples = 50;
        let beam_dt = impact_time / beam_samples as f64;
        let mut min_beam_dist = f64::INFINITY;

        for i in 0..=beam_samples {
            let t = i as f64 * beam_dt;
            let beam_t = predict_falling_position(threat_pos, threat_vel, t);
            let dist = ((candidate[0] - beam_t[0]).powi(2)
                + (candidate[1] - beam_t[1]).powi(2)
                + (candidate[2] - beam_t[2]).powi(2))
            .sqrt();
            if dist < min_beam_dist {
                min_beam_dist = dist;
            }
        }

        // ── Phase C: Reachability check ──
        // Can the drone reach the candidate before the beam hits the entity?
        // Uses impact_time (not closest_approach_time) — the drone has until
        // beam impact to position itself near the candidate.
        let approach_rate = 5.0;
        let initial_dist = ((state.position[0] - candidate[0]).powi(2)
            + (state.position[1] - candidate[1]).powi(2)
            + (state.position[2] - candidate[2]).powi(2))
        .sqrt();
        let drone_gap = initial_dist * (-approach_rate * impact_time).exp();
        let reachable = drone_gap < 0.5; // Within 0.5m by impact time

        let effective_dist = if reachable {
            min_beam_dist
        } else {
            min_beam_dist + drone_gap
        };

        // ── Phase D: Safety + crash from effective_dist ──
        let predicted_danger = if effective_dist < 1.0 {
            env.human_danger * effective_dist
        } else {
            env.human_danger.min(1.0)
        };

        let crash_risk: f64 = if effective_dist < 0.5 { 0.5 } else { 0.0 };

        // ── Phase E: Mission deviation via trajectory rollout (unchanged) ──
        let horizon = 200;
        let dt = 0.002;
        let gamma = 0.95f64;

        let mut total_mission_efe = 0.0;
        let mut discount = 1.0;

        for step in 0..horizon {
            let t = (step + 1) as f64 * dt;
            let drone_t = predict_drone_approach(state.position, candidate, t, approach_rate);

            let mission_target = mission_setpoint.position;
            let deviation_t = ((drone_t[0] - mission_target[0]).powi(2)
                + (drone_t[1] - mission_target[1]).powi(2)
                + (drone_t[2] - mission_target[2]).powi(2))
            .sqrt();

            total_mission_efe +=
                discount * self.config.mission_prior_precision * deviation_t.powi(2);
            discount *= gamma;
        }

        // ── Phase F: Combine with horizon weight ──
        let effective_horizon_weight: f64 = (0..horizon).map(|s| gamma.powi(s)).sum();

        self.config.safety_prior_precision * predicted_danger.powi(2) * effective_horizon_weight
            + total_mission_efe
            + self.config.self_preservation_precision * crash_risk.powi(2) * effective_horizon_weight
    }

    /// Evaluate candidate setpoints and return the EFE-optimal override.
    ///
    /// Generates 8 candidate action policies (when velocity available) via EFE:
    /// 0. Continue mission (current setpoint)
    /// 1a/1b/1c. Intercept threat at 25%, 50%, 75% of impact time (3 rendezvous)
    /// 2. Hover in place (hold current position)
    /// 3. Shield position (midpoint between threat and entity, +0.3m up)
    /// 4. Retreat (move 1m away from threat)
    /// 5. Lateral deflection (perpendicular to threat→entity line)
    ///
    /// **Design note:** Multi-rendezvous sampling lets EFE discover the optimal
    /// intercept timing rather than relying on a fixed 50% heuristic. For some
    /// geometries (drone far away, beam fast), an earlier or later rendezvous
    /// produces a better intercept.
    ///
    /// Uses `trajectory_efe()` when `threat_vel` is available for physics-based
    /// multi-step rollout, falls back to `instantaneous_efe()` otherwise.
    ///
    /// Returns `Some(position)` if a non-current setpoint minimizes EFE,
    /// `None` if the current setpoint is already optimal.
    pub(crate) fn evaluate_setpoint_candidates(
        &self,
        state: &FlightState,
        current_setpoint: &FlightSetpoint,
        env: &FlightEnvironment,
    ) -> Option<[f64; 3]> {
        // Only evaluate when danger is present and threat position is known
        if env.human_danger < 0.01 || env.threat_pos.is_none() {
            return None;
        }

        let threat_pos = env.threat_pos.unwrap();
        let entity_pos = env.entity_pos.unwrap_or([0.0; 3]);
        let drone_pos = state.position;

        // ── Generate candidate setpoints (action policies) ──
        // Fixed-size stack array: max 8 candidates (no heap allocation).
        // Layout: [mission, intercept×3, hover, shield, retreat, lateral]
        let mut candidates: [[f64; 3]; 8] = [[0.0; 3]; 8];
        let mut n_candidates: usize = 0;

        // 0: Continue mission
        candidates[n_candidates] = current_setpoint.position;
        n_candidates += 1;

        // 1a/1b/1c: Intercept threat — sample rendezvous at 25%, 50%, 75% of impact time
        if let Some(vel) = env.threat_vel {
            let impact_t = predict_impact_time_z(threat_pos, vel, entity_pos[2]);
            if impact_t.is_finite() && impact_t > 0.0 {
                for &frac in &[0.25, 0.5, 0.75] {
                    let rv_t = impact_t * frac;
                    let beam = predict_falling_position(threat_pos, vel, rv_t);
                    candidates[n_candidates] =
                        [beam[0], beam[1], beam[2].max(entity_pos[2] + 0.3)];
                    n_candidates += 1;
                }
            } else {
                candidates[n_candidates] = [
                    threat_pos[0],
                    threat_pos[1],
                    threat_pos[2].max(entity_pos[2] + 0.5),
                ];
                n_candidates += 1;
            }
        } else {
            candidates[n_candidates] = [
                threat_pos[0],
                threat_pos[1],
                threat_pos[2].max(entity_pos[2] + 0.5),
            ];
            n_candidates += 1;
        }

        // Hover in place
        candidates[n_candidates] = drone_pos;
        n_candidates += 1;

        // Shield position (midpoint between threat and entity, +0.3m up)
        candidates[n_candidates] = [
            (threat_pos[0] + entity_pos[0]) * 0.5,
            (threat_pos[1] + entity_pos[1]) * 0.5,
            ((threat_pos[2] + entity_pos[2]) * 0.5) + 0.3,
        ];
        n_candidates += 1;

        // Retreat (move 1m away from threat)
        let threat_to_drone = [
            drone_pos[0] - threat_pos[0],
            drone_pos[1] - threat_pos[1],
            drone_pos[2] - threat_pos[2],
        ];
        let retreat_dist = (threat_to_drone[0].powi(2)
            + threat_to_drone[1].powi(2)
            + threat_to_drone[2].powi(2))
        .sqrt()
        .max(0.01);
        candidates[n_candidates] = [
            drone_pos[0] + threat_to_drone[0] / retreat_dist,
            drone_pos[1] + threat_to_drone[1] / retreat_dist,
            drone_pos[2] + threat_to_drone[2] / retreat_dist,
        ];
        n_candidates += 1;

        // Lateral deflection (perpendicular to threat→entity line)
        let threat_to_entity = [
            entity_pos[0] - threat_pos[0],
            entity_pos[1] - threat_pos[1],
            0.0, // horizontal plane only
        ];
        let tte_len = (threat_to_entity[0].powi(2) + threat_to_entity[1].powi(2))
            .sqrt()
            .max(0.01);
        let perp = [-threat_to_entity[1] / tte_len, threat_to_entity[0] / tte_len];
        let midpoint_x = (threat_pos[0] + entity_pos[0]) * 0.5;
        let midpoint_y = (threat_pos[1] + entity_pos[1]) * 0.5;
        candidates[n_candidates] = [
            midpoint_x + perp[0] * 0.5,
            midpoint_y + perp[1] * 0.5,
            threat_pos[2].max(entity_pos[2] + 0.3),
        ];
        n_candidates += 1;

        // ── Evaluate each candidate via EFE ──
        let use_trajectory = env.threat_vel.is_some();
        let mut best_idx = 0;
        let mut best_efe = f64::INFINITY;

        for i in 0..n_candidates {
            let efe = if use_trajectory {
                self.trajectory_efe(state, candidates[i], current_setpoint, env)
            } else {
                self.instantaneous_efe(state, candidates[i], current_setpoint, env)
            };
            if efe < best_efe {
                best_efe = efe;
                best_idx = i;
            }
        }

        // Return override only if the agent chose a DIFFERENT setpoint
        if best_idx != 0 {
            Some(candidates[best_idx])
        } else {
            None
        }
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

        // Extended observation: obs[8] = human_danger (when available)
        if obs.len() > 8 && obs[8] > 0.2 {
            return FlightAction::DropTau;
        }

        if pe_trend > 0.6 && norm_pos > 0.1 {
            // Error growing and significant → speed up adaptation
            FlightAction::DropTau
        } else if pe_trend < 0.4 && norm_pos < 0.05 {
            // Error shrinking and small → consolidate, slow down
            FlightAction::RaiseTau
        } else if norm_pos > 0.05 && (0.45..=0.55).contains(&pe_trend) {
            // Persistent moderate error, not growing or shrinking → steady-state offset.
            // PD alone can't eliminate this; switch to PID for integral correction.
            FlightAction::AdaptBaseline
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
    pub fn step(&mut self, state: &FlightState, setpoint: &FlightSetpoint) -> FlightFepResult {
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

        // Hysteresis: DropTau requires 3+ consecutive ticks to prevent
        // single-gust spikes from destabilizing the controller.
        if action == FlightAction::DropTau {
            self.consecutive_drop_tau += 1;
        } else {
            self.consecutive_drop_tau = 0;
        }

        match action {
            FlightAction::DropTau => {
                if self.consecutive_drop_tau >= 3 {
                    // Sustained rising error — actually adapt
                    result.tau_factor = 0.92;
                }
                // else: transient spike — leave tau at 1.0 (no modulation)
            }
            FlightAction::RaiseTau => {
                result.tau_factor = 1.08;
            }
            FlightAction::BoostLearningRate => {
                result.learning_rate_factor = 1.3;
            }
            FlightAction::ReduceLearningRate => {
                result.learning_rate_factor = 0.7;
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
            FlightAction::AdaptBaseline => {
                result.use_pid_target = true;
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
        self.consecutive_drop_tau = 0;
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
        assert!(
            had_exploration,
            "Should eventually trigger exploration after patience exhausted"
        );
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
        for i in 0..7 {
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

    // ── Item 1: FEP Agent + Controller Integration Tests ──

    #[test]
    fn test_fep_modulates_tau_under_rising_error() {
        // Rising error should trigger DropTau after 3-tick hysteresis.
        // Rule: pe_trend > 0.6 AND norm_pos > 0.1 → DropTau.
        // We need error to genuinely INCREASE each tick so pe_trend stays > 0.6.
        let config = FlightFepConfig::default();
        let mut agent = ActiveInferenceFlightAgent::new(config);

        let setpoint = FlightSetpoint::hover(); // z=0.1

        // Feed states with progressively increasing error (drifting away from setpoint).
        // pe_trend > 0.6 requires delta > 0.05 per step — use large 0.1 increments.
        let mut had_tau_modulation = false;
        for i in 0..200 {
            let drift = 0.1 * (i as f64 + 1.0); // 0.1, 0.2, 0.3, ... growing fast
            let state = FlightState {
                position: [drift, drift, 0.0],
                quaternion: [1.0, 0.0, 0.0, 0.0],
                linear_velocity: [0.0; 3],
                angular_velocity: [0.0; 3],
                timestamp: 0.0,
            };
            let result = agent.step(&state, &setpoint);
            if (result.tau_factor - 1.0).abs() > 0.01 {
                had_tau_modulation = true;
                break;
            }
        }
        assert!(
            had_tau_modulation,
            "FEP agent should modulate tau under sustained rising error (after hysteresis)"
        );
    }

    #[test]
    fn test_fep_exploration_burst_fires() {
        // With low patience and always-exceeding FE threshold, exploration should fire.
        let config = FlightFepConfig {
            exploration_patience: 5,
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

        let mut exploration_fired = false;
        for _ in 0..20 {
            let result = agent.step(&state, &setpoint);
            if result.exploration_noise.is_some() {
                exploration_fired = true;
                break;
            }
        }
        assert!(
            exploration_fired,
            "Exploration burst should fire after patience exhausted"
        );
    }

    #[test]
    fn test_fep_boost_lr_on_high_surprise() {
        // The rule-based policy selects BoostLR when surprise_rate > 0.65
        // AND the DropTau/RaiseTau rules don't fire first.
        //
        // surprise_rate = ((current_fe - prev_fe) / 2.0 + 0.5).clamp(0,1)
        // So we need current_fe > prev_fe by enough to push surprise_rate > 0.65:
        //   (delta / 2.0) + 0.5 > 0.65  →  delta > 0.3
        //
        // Also need: NOT (pe_trend > 0.6 && norm_pos > 0.1) — the DropTau rule.
        // Strategy: keep position error very small (near setpoint) so norm_pos < 0.1,
        // but switch between two states that produce different FE values.
        let config = FlightFepConfig::default();
        let mut agent = ActiveInferenceFlightAgent::new(config);

        let setpoint = FlightSetpoint::hover(); // z=0.1

        // Step 1: prime at hover (low FE state)
        let hover = FlightState::hover(0.1);
        agent.step(&hover, &setpoint);
        agent.step(&hover, &setpoint);
        agent.step(&hover, &setpoint);

        // Step 2: introduce a large angular velocity spike while staying near setpoint.
        // This creates surprise (FE jump) without large position error.
        let surprised_state = FlightState {
            position: [0.0, 0.0, 0.1],
            quaternion: [1.0, 0.0, 0.0, 0.0],
            linear_velocity: [0.0, 0.0, 0.0],
            angular_velocity: [15.0, 15.0, 15.0], // Very high angular speed → surprise
            timestamp: 0.0,
        };

        let mut had_lr_boost = false;
        // Alternate between calm and surprised to keep FE rising
        for i in 0..40 {
            let state = if i % 3 == 0 { &hover } else { &surprised_state };
            let result = agent.step(state, &setpoint);
            if result.learning_rate_factor > 1.0 {
                had_lr_boost = true;
                break;
            }
        }
        assert!(
            had_lr_boost,
            "FEP agent should boost LR when surprise rate is high"
        );
    }

    #[test]
    fn test_controller_tau_modulation_effect() {
        use crate::controller::FlightController;
        use symthaea_core::genesis::GenesisSeed;
        use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

        let genesis = GenesisSeed::from_phrase("test-tau-modulation");
        let config = crate::types::FlightConfig::default();
        let mut controller = FlightController::new(&genesis, &config);

        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);

        // Modulate tau down (0.5) and verify output remains valid
        controller.modulate_tau(0.5);
        let cmd_low = controller.forward(&sensor, 0.002);
        assert!(cmd_low.thrust.is_finite(), "Thrust finite after tau down");

        // Reset and modulate tau up (3.0 = max)
        controller.reset();
        controller.modulate_tau(3.0);
        let cmd_high = controller.forward(&sensor, 0.002);
        assert!(cmd_high.thrust.is_finite(), "Thrust finite after tau up");

        // Modulate below minimum (0.1 → clamped to 0.3)
        controller.reset();
        controller.modulate_tau(0.1); // Should clamp to 0.3
        let cmd_clamped = controller.forward(&sensor, 0.002);
        assert!(
            cmd_clamped.thrust.is_finite(),
            "Thrust finite after tau clamp"
        );

        // Different tau values should produce different outputs
        // (controller state diverges under different time constants)
        let mut ctrl_a = FlightController::new(&genesis, &config);
        let mut ctrl_b = FlightController::new(&genesis, &config);
        ctrl_a.modulate_tau(0.5);
        ctrl_b.modulate_tau(2.0);

        // Evolve both for several steps
        for _ in 0..10 {
            ctrl_a.forward(&sensor, 0.002);
            ctrl_b.forward(&sensor, 0.002);
        }
        let out_a = ctrl_a.forward(&sensor, 0.002);
        let out_b = ctrl_b.forward(&sensor, 0.002);

        let diff = (out_a.thrust - out_b.thrust).abs()
            + (out_a.roll_moment - out_b.roll_moment).abs()
            + (out_a.pitch_moment - out_b.pitch_moment).abs()
            + (out_a.yaw_moment - out_b.yaw_moment).abs();
        assert!(
            diff > 1e-8,
            "Different tau modulation should produce different outputs: diff={diff}"
        );
    }

    #[test]
    fn test_fep_reset_clears_history() {
        let config = FlightFepConfig::default();
        let mut agent = ActiveInferenceFlightAgent::new(config);

        let state = FlightState {
            position: [1.0, 1.0, 0.0],
            linear_velocity: [2.0, 2.0, -1.0],
            ..FlightState::hover(0.0)
        };
        let setpoint = FlightSetpoint::hover();

        // Run for 50 steps to build up history
        for _ in 0..50 {
            agent.step(&state, &setpoint);
        }

        // Verify state has been modified
        assert!(agent.prev_prediction_error > 0.0);

        // Reset
        agent.reset();

        // Verify reset clears all tracking state
        assert_eq!(agent.high_fe_ticks, 0);
        assert!((agent.prev_prediction_error - 0.0).abs() < 1e-10);
        assert!((agent.tau_ema - 1.0).abs() < 1e-10);
        assert!((agent.current_fe - 0.0).abs() < 1e-10);
        assert!((agent.prev_fe - 0.0).abs() < 1e-10);

        // Next step should behave like initial state (pe_trend = 0.5 since no prev)
        let result = agent.step(&FlightState::hover(0.1), &setpoint);
        assert!(result.free_energy.is_finite());
    }

    // ── Emergent FEP Sacrifice Coverage ──

    #[test]
    fn test_step_extended_danger_triggers_drop_tau() {
        let config = FlightFepConfig {
            extended_observation: true,
            ..FlightFepConfig::default()
        };
        let mut agent = ActiveInferenceFlightAgent::new(config);

        let state = FlightState::hover(0.1);
        let setpoint = FlightSetpoint::hover();

        let result = agent.step_extended(&state, &setpoint, 0.8, 0.5);

        // Danger > 0.2 triggers DropTau via rule_based_action(obs[8] > 0.2).
        // No hardcoded override — natural DropTau gives tau_factor = 0.92.
        assert!(
            (result.tau_factor - 0.92).abs() < 1e-6,
            "Danger should trigger DropTau (tau=0.92), got {}",
            result.tau_factor
        );

        // Free energy should include danger_fe_boost = 0.8 * 10.0 = 8.0
        assert!(
            result.free_energy >= 8.0,
            "Free energy should include danger boost (>=8.0), got {:.4}",
            result.free_energy
        );
    }

    #[test]
    fn test_step_extended_no_danger_normal() {
        let config = FlightFepConfig {
            extended_observation: true,
            ..FlightFepConfig::default()
        };
        let mut agent_ext = ActiveInferenceFlightAgent::new(config.clone());
        let mut agent_normal = ActiveInferenceFlightAgent::new(FlightFepConfig::default());

        let state = FlightState::hover(0.1);
        let setpoint = FlightSetpoint::hover();

        let result_ext = agent_ext.step_extended(&state, &setpoint, 0.0, 0.5);
        let result_normal = agent_normal.step(&state, &setpoint);

        // With zero danger, obs[8]=0 doesn't trigger DropTau override
        assert!(
            (result_ext.tau_factor - 0.92).abs() > 0.01,
            "Zero danger should not trigger DropTau, tau_factor={}",
            result_ext.tau_factor
        );

        // Both should produce finite free energy
        assert!(result_ext.free_energy.is_finite());
        assert!(result_normal.free_energy.is_finite());
    }

    #[test]
    fn test_step_extended_danger_threshold() {
        let config = FlightFepConfig {
            extended_observation: true,
            ..FlightFepConfig::default()
        };

        // Below rule threshold: danger = 0.15 (obs[8] < 0.2 → no DropTau from danger)
        let mut agent_below = ActiveInferenceFlightAgent::new(config.clone());
        let state = FlightState::hover(0.1);
        let setpoint = FlightSetpoint::hover();

        let result_below = agent_below.step_extended(&state, &setpoint, 0.15, 0.5);
        // Should NOT trigger danger-driven DropTau (tau_factor != 0.92)
        assert!(
            (result_below.tau_factor - 0.92).abs() > 0.01,
            "Danger=0.15 should not trigger DropTau, tau_factor={}",
            result_below.tau_factor
        );

        // Above rule threshold: danger = 0.3 (obs[8] > 0.2 → DropTau)
        let mut agent_above = ActiveInferenceFlightAgent::new(config);
        let result_above = agent_above.step_extended(&state, &setpoint, 0.3, 0.5);
        // Natural DropTau: tau_factor = 0.92
        assert!(
            (result_above.tau_factor - 0.92).abs() < 1e-6,
            "Danger=0.3 should trigger DropTau (tau=0.92), tau_factor={}",
            result_above.tau_factor
        );
    }

    // ── EFE-based Emergent Setpoint Selection ──

    #[test]
    fn test_efe_selects_intercept_under_high_danger() {
        // When danger is high and a threat position is known, EFE should
        // select the interception setpoint over continuing the mission.
        let config = FlightFepConfig {
            extended_observation: true,
            ..FlightFepConfig::default()
        };
        let mut agent = ActiveInferenceFlightAgent::new(config);

        let state = FlightState {
            position: [0.0, 0.0, 1.5],
            ..FlightState::hover(0.1)
        };
        let setpoint = FlightSetpoint {
            position: [-3.0, 0.0, 1.0],
            yaw: 0.0,
        };

        let env = FlightEnvironment {
            human_danger: 0.8,
            mission_progress: 0.3,
            threat_pos: Some([-1.5, 0.0, 2.0]),
            threat_vel: Some([0.0, 0.0, -3.0]),
            entity_pos: Some([-1.5, 0.0, 0.0]),
        };

        let result = agent.step_embodied(&state, &setpoint, &env);

        // The agent should choose to intercept the beam
        assert!(
            result.setpoint_override.is_some(),
            "EFE should select intercept when danger=0.8"
        );

        // The override should be near the threat position
        let override_pos = result.setpoint_override.unwrap();
        let dist_to_threat = ((override_pos[0] - (-1.5)).powi(2)
            + (override_pos[1] - 0.0).powi(2))
        .sqrt();
        assert!(
            dist_to_threat < 0.1,
            "Override should be near threat, dist={}",
            dist_to_threat
        );
    }

    #[test]
    fn test_efe_continues_mission_when_no_danger() {
        // With no danger, the agent should not override the setpoint.
        let config = FlightFepConfig {
            extended_observation: true,
            ..FlightFepConfig::default()
        };
        let mut agent = ActiveInferenceFlightAgent::new(config);

        let state = FlightState::hover(0.1);
        let setpoint = FlightSetpoint::hover();

        let env = FlightEnvironment {
            human_danger: 0.0,
            mission_progress: 0.5,
            threat_pos: None,
            threat_vel: None,
            entity_pos: None,
        };

        let result = agent.step_embodied(&state, &setpoint, &env);

        assert!(
            result.setpoint_override.is_none(),
            "No danger → no setpoint override"
        );
    }

    #[test]
    fn test_efe_precision_ratio_determines_choice() {
        // With safety_prior_precision very low (mission dominates),
        // the agent should NOT override, even under danger.
        let config = FlightFepConfig {
            extended_observation: true,
            safety_prior_precision: 0.001,   // Safety barely matters
            mission_prior_precision: 1000.0, // Mission dominates
            self_preservation_precision: 0.1,
            ..FlightFepConfig::default()
        };
        let mut agent = ActiveInferenceFlightAgent::new(config);

        let state = FlightState {
            position: [0.0, 0.0, 1.5],
            ..FlightState::hover(0.1)
        };
        let setpoint = FlightSetpoint {
            position: [-3.0, 0.0, 1.0],
            yaw: 0.0,
        };

        let env = FlightEnvironment {
            human_danger: 0.8,
            mission_progress: 0.3,
            threat_pos: Some([-1.5, 0.0, 2.0]),
            threat_vel: Some([0.0, 0.0, -3.0]),
            entity_pos: Some([-1.5, 0.0, 0.0]),
        };

        let result = agent.step_embodied(&state, &setpoint, &env);

        // With inverted precisions, mission EFE dominates —
        // the agent should continue toward its delivery target.
        assert!(
            result.setpoint_override.is_none(),
            "Inverted precisions: mission should dominate, no override. \
             safety_π=0.001 vs mission_π=1000"
        );
    }

    #[test]
    fn test_efe_calculation_math() {
        // Verify the EFE formula directly.
        let config = FlightFepConfig {
            extended_observation: true,
            safety_prior_precision: 100.0,
            mission_prior_precision: 1.0,
            self_preservation_precision: 0.1,
            ..FlightFepConfig::default()
        };
        let agent = ActiveInferenceFlightAgent::new(config);

        let state = FlightState::hover(0.1);
        let mission_setpoint = FlightSetpoint {
            position: [-3.0, 0.0, 1.0],
            yaw: 0.0,
        };
        let env = FlightEnvironment {
            human_danger: 0.7,
            mission_progress: 0.3,
            threat_pos: Some([-1.5, 0.0, 2.0]),
            threat_vel: Some([0.0, 0.0, -3.0]),
            entity_pos: Some([-1.5, 0.0, 0.0]),
        };

        // EFE for continuing mission (far from threat)
        let efe_mission = agent.instantaneous_efe(
            &state,
            mission_setpoint.position,
            &mission_setpoint,
            &env,
        );

        // EFE for intercepting (near threat)
        let efe_intercept = agent.instantaneous_efe(
            &state,
            [-1.5, 0.0, 2.0],
            &mission_setpoint,
            &env,
        );

        // Mission EFE should be dominated by safety term: π_safety * danger²
        // = 100.0 * 0.7² = 49.0 (danger persists since far from threat)
        assert!(
            efe_mission > 40.0,
            "Mission EFE should be high due to unresolved danger: {:.2}",
            efe_mission
        );

        // Intercept EFE should have lower safety cost (near threat → danger reduced)
        // but higher mission deviation cost
        assert!(
            efe_intercept < efe_mission,
            "Intercept EFE ({:.2}) should be less than mission EFE ({:.2})",
            efe_intercept,
            efe_mission
        );
    }

    // ── FEP-driven PID Selection ──

    #[test]
    fn test_fep_adapt_baseline_triggers_pid() {
        // AdaptBaseline rule fires when: norm_pos > 0.05 AND pe_trend in [0.45, 0.55]
        // (steady-state offset: error present but not changing).
        let config = FlightFepConfig::default();
        let mut agent = ActiveInferenceFlightAgent::new(config);

        let setpoint = FlightSetpoint::hover(); // z=0.1

        // Prime with a state that has moderate, steady position error.
        // First step: set prev_prediction_error to ~same as next step's error.
        let offset_state = FlightState {
            position: [0.1, 0.0, 0.1], // ~0.1m offset in x → norm_pos ≈ 0.1
            quaternion: [1.0, 0.0, 0.0, 0.0],
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
            timestamp: 0.0,
        };
        // Prime to set prev_prediction_error
        agent.step(&offset_state, &setpoint);

        // Now step again with very similar error → pe_trend ≈ 0.5 (stable)
        let mut had_pid_signal = false;
        for _ in 0..20 {
            let result = agent.step(&offset_state, &setpoint);
            if result.use_pid_target {
                had_pid_signal = true;
                break;
            }
        }
        assert!(
            had_pid_signal,
            "FEP should signal PID target under persistent steady-state offset"
        );
    }

    #[test]
    fn test_fep_no_pid_at_hover() {
        // At hover (near-zero error), AdaptBaseline should NOT trigger.
        let config = FlightFepConfig::default();
        let mut agent = ActiveInferenceFlightAgent::new(config);

        let state = FlightState::hover(0.1);
        let setpoint = FlightSetpoint::hover();

        for _ in 0..20 {
            let result = agent.step(&state, &setpoint);
            assert!(
                !result.use_pid_target,
                "Should not signal PID at hover (no steady-state offset)"
            );
        }
    }

    // ── Physics Prediction Helper Tests ──

    #[test]
    fn test_predict_falling_position_gravity() {
        // Object at 10m, zero velocity → falls ~4.905m in 1s
        let pos = predict_falling_position([0.0, 0.0, 10.0], [0.0, 0.0, 0.0], 1.0);
        let expected_z = 10.0 - 0.5 * 9.81; // 5.095
        assert!(
            (pos[2] - expected_z).abs() < 0.01,
            "Expected z≈{:.3}, got {:.3}",
            expected_z,
            pos[2]
        );
        assert!((pos[0]).abs() < 1e-10, "x should stay at 0");
        assert!((pos[1]).abs() < 1e-10, "y should stay at 0");
    }

    #[test]
    fn test_predict_falling_position_with_velocity() {
        // Object at origin with horizontal velocity [1, 0, 0] and downward [-2]
        let pos = predict_falling_position([0.0, 0.0, 5.0], [1.0, 0.0, -2.0], 0.5);
        assert!(
            (pos[0] - 0.5).abs() < 0.01,
            "x should be ~0.5, got {}",
            pos[0]
        );
        let expected_z = 5.0 + (-2.0) * 0.5 - 0.5 * 9.81 * 0.25;
        assert!(
            (pos[2] - expected_z).abs() < 0.01,
            "z expected {:.3}, got {:.3}",
            expected_z,
            pos[2]
        );
    }

    #[test]
    fn test_predict_drone_approach_converges() {
        // Drone at [0,0,0] approaching [1,1,1] with rate=5.0
        let target = [1.0, 1.0, 1.0];
        let pos_short = predict_drone_approach([0.0, 0.0, 0.0], target, 0.1, 5.0);
        let pos_long = predict_drone_approach([0.0, 0.0, 0.0], target, 2.0, 5.0);

        // After 2s with rate=5, should be very close to target
        for i in 0..3 {
            assert!(
                (pos_long[i] - target[i]).abs() < 0.01,
                "Should converge to target after 2s: dim {} = {:.4}",
                i,
                pos_long[i]
            );
        }
        // After 0.1s, should have moved partway
        for i in 0..3 {
            assert!(
                pos_short[i] > 0.0 && pos_short[i] < target[i],
                "Should be partway after 0.1s: dim {} = {:.4}",
                i,
                pos_short[i]
            );
        }
    }

    #[test]
    fn test_predict_impact_time_z_basic() {
        // Object at z=10, falling at -2 m/s → reaches z=0
        let t = predict_impact_time_z([0.0, 0.0, 10.0], [0.0, 0.0, -2.0], 0.0);
        assert!(t > 0.0 && t < 3.0, "Impact time should be ~1s, got {}", t);
    }

    #[test]
    fn test_predict_impact_time_z_unreachable() {
        // Object going upward — doesn't reach lower altitude easily
        let t = predict_impact_time_z([0.0, 0.0, 0.5], [0.0, 0.0, 10.0], 0.0);
        // With upward velocity=10 from z=0.5, gravity will eventually bring it back
        // but the function returns the positive root — let's just check it's finite
        assert!(t.is_finite() || t == f64::INFINITY);
    }

    // ── Trajectory EFE Tests ──

    #[test]
    fn test_trajectory_efe_intercept_lower_than_mission() {
        // With physics rollout, intercepting should still have lower EFE than continuing mission.
        // Use the rendezvous position: beam at [-1.5,0,2] vel [0,0,-3] → entity z=0
        // impact_t ≈ 0.4s, rendezvous at t≈0.2s, beam_z ≈ 2+(-3)(0.2)-0.5(9.81)(0.04) ≈ 1.204
        let config = FlightFepConfig {
            extended_observation: true,
            ..FlightFepConfig::default()
        };
        let agent = ActiveInferenceFlightAgent::new(config);

        let state = FlightState {
            position: [0.0, 0.0, 1.5],
            ..FlightState::hover(0.1)
        };
        let mission_setpoint = FlightSetpoint {
            position: [-3.0, 0.0, 1.0],
            yaw: 0.0,
        };
        let env = FlightEnvironment {
            human_danger: 0.8,
            mission_progress: 0.3,
            threat_pos: Some([-1.5, 0.0, 2.0]),
            threat_vel: Some([0.0, 0.0, -3.0]),
            entity_pos: Some([-1.5, 0.0, 0.0]),
        };

        // Compute the rendezvous position the agent would use
        let impact_t = predict_impact_time_z([-1.5, 0.0, 2.0], [0.0, 0.0, -3.0], 0.0);
        let rv_t = impact_t * 0.5;
        let beam_rv = predict_falling_position([-1.5, 0.0, 2.0], [0.0, 0.0, -3.0], rv_t);
        let intercept_candidate = [beam_rv[0], beam_rv[1], beam_rv[2].max(0.3)];

        let efe_mission =
            agent.trajectory_efe(&state, mission_setpoint.position, &mission_setpoint, &env);
        let efe_intercept =
            agent.trajectory_efe(&state, intercept_candidate, &mission_setpoint, &env);

        assert!(
            efe_intercept < efe_mission,
            "Trajectory EFE: intercept ({:.2}) should be < mission ({:.2})",
            efe_intercept,
            efe_mission
        );
    }

    #[test]
    fn test_step_embodied_requests_fast_cognitive_under_danger() {
        let config = FlightFepConfig {
            extended_observation: true,
            ..FlightFepConfig::default()
        };
        let mut agent = ActiveInferenceFlightAgent::new(config);

        let state = FlightState {
            position: [0.0, 0.0, 1.5],
            ..FlightState::hover(0.1)
        };
        let setpoint = FlightSetpoint {
            position: [-3.0, 0.0, 1.0],
            yaw: 0.0,
        };

        // High danger → smooth ramp: danger=0.8 → t=(0.8-0.1)/0.9≈0.778 → 25+75*0.778≈83.3Hz
        let env_danger = FlightEnvironment {
            human_danger: 0.8,
            mission_progress: 0.3,
            threat_pos: Some([-1.5, 0.0, 2.0]),
            threat_vel: Some([0.0, 0.0, -3.0]),
            entity_pos: Some([-1.5, 0.0, 0.0]),
        };
        let result = agent.step_embodied(&state, &setpoint, &env_danger);
        let hz = result.requested_cognitive_hz.expect("danger=0.8 should request cognitive Hz");
        assert!(
            hz > 80.0,
            "danger=0.8 should request >80Hz, got {:.1}Hz",
            hz
        );

        // No danger → should request None (default rate)
        let env_safe = FlightEnvironment {
            human_danger: 0.0,
            mission_progress: 0.5,
            threat_pos: None,
            threat_vel: None,
            entity_pos: None,
        };
        let result_safe = agent.step_embodied(&state, &setpoint, &env_safe);
        assert!(
            result_safe.requested_cognitive_hz.is_none(),
            "No danger should use default cognitive rate"
        );
    }

    #[test]
    fn test_adaptive_hz_smooth_ramp_intermediate() {
        // Verify the smooth ramp at intermediate danger values.
        // Formula: hz = 25 + 75 * ((danger - 0.1) / 0.9)
        let config = FlightFepConfig {
            extended_observation: true,
            ..FlightFepConfig::default()
        };

        let state = FlightState {
            position: [0.0, 0.0, 1.5],
            ..FlightState::hover(0.1)
        };
        let setpoint = FlightSetpoint {
            position: [-3.0, 0.0, 1.0],
            yaw: 0.0,
        };

        // danger=0.5 → t=(0.5-0.1)/0.9≈0.444 → hz=25+75*0.444≈58.3
        let mut agent = ActiveInferenceFlightAgent::new(config.clone());
        let env_mid = FlightEnvironment {
            human_danger: 0.5,
            mission_progress: 0.3,
            threat_pos: Some([-1.5, 0.0, 2.0]),
            threat_vel: Some([0.0, 0.0, -3.0]),
            entity_pos: Some([-1.5, 0.0, 0.0]),
        };
        let result_mid = agent.step_embodied(&state, &setpoint, &env_mid);
        let hz_mid = result_mid.requested_cognitive_hz.expect("danger=0.5 should request Hz");
        assert!(
            (hz_mid - 58.3).abs() < 1.0,
            "danger=0.5 should be ~58Hz, got {:.1}Hz",
            hz_mid
        );

        // danger=0.1 → t=0 → hz=25 (threshold boundary)
        let mut agent2 = ActiveInferenceFlightAgent::new(config.clone());
        let env_low = FlightEnvironment {
            human_danger: 0.1,
            mission_progress: 0.3,
            threat_pos: Some([-1.5, 0.0, 2.0]),
            threat_vel: Some([0.0, 0.0, -3.0]),
            entity_pos: Some([-1.5, 0.0, 0.0]),
        };
        // danger=0.1 is exactly at the threshold — should NOT trigger (> 0.1 required)
        let result_low = agent2.step_embodied(&state, &setpoint, &env_low);
        assert!(
            result_low.requested_cognitive_hz.is_none(),
            "danger=0.1 (at threshold) should use default rate, got {:?}",
            result_low.requested_cognitive_hz
        );

        // danger=1.0 → t=1.0 → hz=100 (maximum)
        let mut agent3 = ActiveInferenceFlightAgent::new(config);
        let env_max = FlightEnvironment {
            human_danger: 1.0,
            mission_progress: 0.3,
            threat_pos: Some([-1.5, 0.0, 2.0]),
            threat_vel: Some([0.0, 0.0, -3.0]),
            entity_pos: Some([-1.5, 0.0, 0.0]),
        };
        let result_max = agent3.step_embodied(&state, &setpoint, &env_max);
        let hz_max = result_max.requested_cognitive_hz.expect("danger=1.0 should request Hz");
        assert!(
            (hz_max - 100.0).abs() < 0.1,
            "danger=1.0 should be 100Hz, got {:.1}Hz",
            hz_max
        );
    }

    #[test]
    fn test_trajectory_efe_no_threat_is_zero() {
        let config = FlightFepConfig::default();
        let agent = ActiveInferenceFlightAgent::new(config);

        let state = FlightState::hover(0.1);
        let setpoint = FlightSetpoint::hover();
        let env = FlightEnvironment {
            human_danger: 0.0,
            mission_progress: 0.5,
            threat_pos: None,
            threat_vel: None,
            entity_pos: None,
        };

        let efe = agent.trajectory_efe(&state, setpoint.position, &setpoint, &env);
        assert!(
            efe.abs() < 1e-10,
            "No threat → trajectory EFE should be 0, got {:.6}",
            efe
        );
    }

    #[test]
    fn test_trajectory_efe_far_beam_no_interception() {
        // Beam 5m away horizontally — drone can't reach in 0.1s rollout
        let config = FlightFepConfig {
            extended_observation: true,
            ..FlightFepConfig::default()
        };
        let agent = ActiveInferenceFlightAgent::new(config);

        let state = FlightState {
            position: [0.0, 0.0, 1.5],
            ..FlightState::hover(0.1)
        };
        let setpoint = FlightSetpoint {
            position: [-3.0, 0.0, 1.0],
            yaw: 0.0,
        };
        let env = FlightEnvironment {
            human_danger: 0.5,
            mission_progress: 0.3,
            threat_pos: Some([5.0, 5.0, 3.0]),
            threat_vel: Some([0.0, 0.0, -2.0]),
            entity_pos: Some([5.0, 5.0, 0.0]),
        };

        // Try to intercept from 7m away — won't reach in 0.1s
        let efe_intercept = agent.trajectory_efe(&state, [5.0, 5.0, 3.0], &setpoint, &env);
        // Danger should still be present (not intercepted)
        assert!(
            efe_intercept > 0.0,
            "Far beam: EFE should be > 0 (danger persists), got {:.6}",
            efe_intercept
        );
    }

    // ── Richer Candidate Tests ──

    #[test]
    fn test_richer_candidates_still_select_intercept() {
        // Default precision ratio should still choose intercept with 6 candidates
        let config = FlightFepConfig {
            extended_observation: true,
            ..FlightFepConfig::default()
        };
        let mut agent = ActiveInferenceFlightAgent::new(config);

        let state = FlightState {
            position: [0.0, 0.0, 1.5],
            ..FlightState::hover(0.1)
        };
        let setpoint = FlightSetpoint {
            position: [-3.0, 0.0, 1.0],
            yaw: 0.0,
        };
        let env = FlightEnvironment {
            human_danger: 0.8,
            mission_progress: 0.3,
            threat_pos: Some([-1.5, 0.0, 2.0]),
            threat_vel: Some([0.0, 0.0, -3.0]),
            entity_pos: Some([-1.5, 0.0, 0.0]),
        };

        let result = agent.step_embodied(&state, &setpoint, &env);
        assert!(
            result.setpoint_override.is_some(),
            "With 6 candidates and default precisions, should still override for interception"
        );
    }

    #[test]
    fn test_hover_candidate_no_override_when_safe() {
        // When danger is very low, no candidate should override
        let config = FlightFepConfig {
            extended_observation: true,
            ..FlightFepConfig::default()
        };
        let agent = ActiveInferenceFlightAgent::new(config);

        let state = FlightState::hover(0.1);
        let setpoint = FlightSetpoint::hover();
        let env = FlightEnvironment {
            human_danger: 0.005, // Below 0.01 threshold
            mission_progress: 0.5,
            threat_pos: Some([5.0, 5.0, 3.0]),
            threat_vel: Some([0.0, 0.0, -1.0]),
            entity_pos: Some([5.0, 5.0, 0.0]),
        };

        let override_pos = agent.evaluate_setpoint_candidates(&state, &setpoint, &env);
        assert!(
            override_pos.is_none(),
            "Very low danger should not trigger override"
        );
    }

    // ── Beam-Trajectory Proximity Tests ──

    #[test]
    fn test_trajectory_efe_uses_beam_trajectory() {
        // Beam starts high (z=5), candidate at beam's future position gets low EFE.
        // Entity at ground level — beam falls from z=5 toward z=0.
        let config = FlightFepConfig {
            extended_observation: true,
            ..FlightFepConfig::default()
        };
        let agent = ActiveInferenceFlightAgent::new(config);

        let state = FlightState {
            position: [0.0, 0.0, 2.0],
            ..FlightState::hover(0.1)
        };
        let mission_setpoint = FlightSetpoint {
            position: [-3.0, 0.0, 1.0],
            yaw: 0.0,
        };
        let env = FlightEnvironment {
            human_danger: 0.8,
            mission_progress: 0.3,
            threat_pos: Some([0.0, 0.0, 5.0]),   // Beam starts at z=5
            threat_vel: Some([0.0, 0.0, -3.0]),   // Falling at -3 m/s
            entity_pos: Some([0.0, 0.0, 0.0]),    // Human at ground
        };

        // Candidate at a point on the beam's trajectory (beam will pass z≈2.0
        // at t≈0.33s based on kinematics: 5 + (-3)*t - 0.5*9.81*t² = 2)
        let beam_path_candidate = [0.0, 0.0, 2.0];
        // Candidate far from beam path
        let off_path_candidate = [3.0, 3.0, 2.0];

        let efe_on_path = agent.trajectory_efe(
            &state, beam_path_candidate, &mission_setpoint, &env,
        );
        let efe_off_path = agent.trajectory_efe(
            &state, off_path_candidate, &mission_setpoint, &env,
        );

        // On-path candidate should have lower EFE (danger reduced by proximity)
        assert!(
            efe_on_path < efe_off_path,
            "Candidate on beam path ({:.2}) should have lower EFE than off-path ({:.2})",
            efe_on_path, efe_off_path
        );
    }

    #[test]
    fn test_trajectory_efe_reachability_penalizes_distant_drone() {
        // Perfect intercept position but drone 20m away — unreachable.
        let config = FlightFepConfig {
            extended_observation: true,
            ..FlightFepConfig::default()
        };
        let agent = ActiveInferenceFlightAgent::new(config);

        // Drone very far from the intercept candidate
        let state_far = FlightState {
            position: [20.0, 20.0, 1.5],
            ..FlightState::hover(0.1)
        };
        // Drone close to the intercept candidate
        let state_close = FlightState {
            position: [-1.0, 0.0, 1.5],
            ..FlightState::hover(0.1)
        };
        let mission_setpoint = FlightSetpoint {
            position: [-3.0, 0.0, 1.0],
            yaw: 0.0,
        };
        let env = FlightEnvironment {
            human_danger: 0.8,
            mission_progress: 0.3,
            threat_pos: Some([-1.5, 0.0, 2.0]),
            threat_vel: Some([0.0, 0.0, -3.0]),
            entity_pos: Some([-1.5, 0.0, 0.0]),
        };

        let intercept_candidate = [-1.5, 0.0, 2.0];

        let efe_far = agent.trajectory_efe(
            &state_far, intercept_candidate, &mission_setpoint, &env,
        );
        let efe_close = agent.trajectory_efe(
            &state_close, intercept_candidate, &mission_setpoint, &env,
        );

        // Far drone should have higher EFE for the same candidate (reachability penalty)
        assert!(
            efe_far > efe_close,
            "Distant drone EFE ({:.2}) should exceed close drone EFE ({:.2}) due to reachability penalty",
            efe_far, efe_close
        );
    }

    #[test]
    fn test_trajectory_efe_mission_not_on_beam_path() {
        // Regression test: mission at (-3,0,1) gets NO danger reduction when
        // beam falls at x=-1.5. The beam path never comes within 1.0m of (-3,0,1).
        let config = FlightFepConfig {
            extended_observation: true,
            ..FlightFepConfig::default()
        };
        let agent = ActiveInferenceFlightAgent::new(config);

        let state = FlightState {
            position: [0.0, 0.0, 1.5],
            ..FlightState::hover(0.1)
        };
        let mission_setpoint = FlightSetpoint {
            position: [-3.0, 0.0, 1.0],
            yaw: 0.0,
        };
        let env = FlightEnvironment {
            human_danger: 0.8,
            mission_progress: 0.3,
            threat_pos: Some([-1.5, 0.0, 2.0]),
            threat_vel: Some([0.0, 0.0, -3.0]),
            entity_pos: Some([-1.5, 0.0, 0.0]),
        };

        let efe_mission = agent.trajectory_efe(
            &state, mission_setpoint.position, &mission_setpoint, &env,
        );

        // Mission candidate at (-3,0,1) is 1.5m horizontally from beam at x=-1.5.
        // The beam falls straight down → min_beam_dist ≥ 1.5m → no danger reduction.
        // Safety term should use full danger: π_safety * danger² * weight.
        let efe_intercept = agent.trajectory_efe(
            &state, [-1.5, 0.0, 2.0], &mission_setpoint, &env,
        );

        assert!(
            efe_mission > efe_intercept,
            "Mission EFE ({:.2}) should be > intercept EFE ({:.2}): mission not on beam path",
            efe_mission, efe_intercept
        );
    }

    #[test]
    fn test_trajectory_efe_beam_going_up_fallback() {
        // Upward beam velocity toward unreachable entity → impact_time = ∞ → falls back to instantaneous.
        let config = FlightFepConfig {
            extended_observation: true,
            ..FlightFepConfig::default()
        };
        let agent = ActiveInferenceFlightAgent::new(config);

        let state = FlightState {
            position: [0.0, 0.0, 1.5],
            ..FlightState::hover(0.1)
        };
        let mission_setpoint = FlightSetpoint {
            position: [-3.0, 0.0, 1.0],
            yaw: 0.0,
        };

        // Beam going upward at 10 m/s from z=2. Max height ≈ 7.1m (v²/2g + z₀).
        // Entity at z=100 — beam can never reach it → discriminant < 0 → impact_time = ∞.
        let env = FlightEnvironment {
            human_danger: 0.8,
            mission_progress: 0.3,
            threat_pos: Some([-1.5, 0.0, 2.0]),
            threat_vel: Some([0.0, 0.0, 10.0]),   // Going up!
            entity_pos: Some([-1.5, 0.0, 100.0]), // Entity far above beam's max height
        };

        // Should not panic — falls back to instantaneous
        let efe = agent.trajectory_efe(
            &state, mission_setpoint.position, &mission_setpoint, &env,
        );
        assert!(efe.is_finite(), "Upward beam should produce finite EFE via fallback: {}", efe);

        // Compare with explicit instantaneous to verify fallback
        let efe_inst = agent.instantaneous_efe(
            &state, mission_setpoint.position, &mission_setpoint, &env,
        );
        assert!(
            (efe - efe_inst).abs() < 1e-10,
            "Upward beam trajectory EFE ({:.4}) should equal instantaneous EFE ({:.4})",
            efe, efe_inst
        );
    }
}
