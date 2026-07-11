// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # FEP Module — Consolidated Free Energy Principle / Active Inference State
//!
//! Consolidates 10 previously scattered FEP fields from CognitiveLoopService
//! into a single coherent module.
//!
//! ## Trajectory Planning (ODE-based)
//!
//! The `GenerativeModelOde` adapter wraps the FEP generative model's transition
//! matrices as a continuous-time ODE system: `ds/dt = (A_action * s - s) / tau`.
//! This enables Dormand-Prince adaptive rollouts for computing expected free
//! energy along continuous trajectories — genuine planning through simulation.
//!
//! Science: Friston, K. (2010) — Active Inference requires trajectory simulation
//! to compute expected free energy over future time horizons.

use crate::consciousness::fep_active_inference::{ActiveInferenceAgent, EnhancedFEPBridge};
use crate::dynamics::ode_solvers::{OdeConfig, OdeResult, OdeSolver, OdeSolverEngine, OdeSystem};
use crate::exploration::SurpriseExplorationBridge;
use std::collections::VecDeque;
use symthaea_core::embodiment::EmbodimentBridge;
use symthaea_core::physics::thermodynamics::ThermodynamicLedger;

use super::goal_world::{GoalSystemBridge, WorldModelBridge};
use super::learning::ClosedLearningLoop;
use super::memory_bridge::EpisodicMemoryBridge;
use super::routing::ActiveInferenceBridge;

// ─── Trajectory Planning Constants ──────────────────────────────────────────

/// Default trajectory planning horizon in seconds (~10 cycles at 20Hz).
const TRAJECTORY_HORIZON_SECONDS: f64 = 0.5;

/// Default ODE tolerance for trajectory rollouts (planning doesn't need
/// control-law precision).
const TRAJECTORY_ODE_TOLERANCE: f64 = 1e-4;

/// Time constant for continuous-time transition dynamics.
/// Controls how quickly the discrete transition is "spread" into continuous time.
/// Science: Friston (2017) — τ governs the rate of belief relaxation.
const TRAJECTORY_TAU: f64 = 0.1;

/// Maximum ODE steps per trajectory (budget cap to prevent runaway).
const TRAJECTORY_MAX_STEPS: usize = 200;

/// Cycle interval for trajectory planning (not every cycle — too expensive).
const TRAJECTORY_PLANNING_INTERVAL: u64 = 10;

/// Ring buffer capacity for trajectory history.
const TRAJECTORY_HISTORY_CAP: usize = 16;

// ─── ODE System Adapter ─────────────────────────────────────────────────────

/// Wraps a generative model's transition matrix for a specific action
/// as a continuous-time ODE system.
///
/// The discrete transition `s' = A * s` is converted to continuous dynamics:
/// `ds/dt = (A * s - s) / tau`
///
/// This is the standard continuous-time embedding of a discrete Markov chain
/// (matrix exponential: `exp(t * (A-I)/tau)` recovers the discrete step at `t=tau`).
pub struct GenerativeModelOde {
    /// Transition matrix for the selected action (row-major: transition[from][to]).
    transition: Vec<Vec<f64>>,
    /// State dimension.
    dim: usize,
    /// Time constant controlling relaxation rate.
    tau: f64,
}

impl GenerativeModelOde {
    /// Create an ODE system from a generative model's transition matrix for a given action.
    pub fn from_transition(transition: &[Vec<f64>], tau: f64) -> Self {
        Self {
            transition: transition.to_vec(),
            dim: transition.len(),
            tau,
        }
    }
}

impl OdeSystem for GenerativeModelOde {
    fn dimension(&self) -> usize {
        self.dim
    }

    fn evaluate(&self, _t: f64, state: &[f64], derivative: &mut [f64]) {
        // ds_i/dt = (sum_j A[j][i] * s[j] - s[i]) / tau
        for i in 0..self.dim {
            let mut transition_sum = 0.0;
            for j in 0..self.dim.min(state.len()) {
                transition_sum += self.transition[j][i] * state[j];
            }
            derivative[i] = (transition_sum - state[i]) / self.tau;
        }
    }
}

// ─── Trajectory Planning Types ──────────────────────────────────────────────

/// Configuration for ODE-based trajectory planning.
#[derive(Debug, Clone)]
pub struct TrajectoryPlanningConfig {
    /// Whether trajectory planning is enabled.
    pub enabled: bool,
    /// Planning horizon in seconds.
    pub horizon_seconds: f64,
    /// ODE solver tolerance.
    pub ode_tolerance: f64,
    /// Maximum ODE steps per trajectory.
    pub max_steps: usize,
    /// Time constant for continuous dynamics.
    pub tau: f64,
    /// Cycle interval: run planning every N cycles.
    pub planning_interval: u64,
}

impl Default for TrajectoryPlanningConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            horizon_seconds: TRAJECTORY_HORIZON_SECONDS,
            ode_tolerance: TRAJECTORY_ODE_TOLERANCE,
            max_steps: TRAJECTORY_MAX_STEPS,
            tau: TRAJECTORY_TAU,
            planning_interval: TRAJECTORY_PLANNING_INTERVAL,
        }
    }
}

/// Result of a single action's trajectory rollout.
#[derive(Debug, Clone)]
pub struct TrajectoryRollout {
    /// Action index that was simulated.
    pub action: usize,
    /// Cumulative expected free energy along the trajectory.
    pub expected_free_energy: f64,
    /// Epistemic value (information gain from trajectory).
    pub information_gain: f64,
    /// Pragmatic value (goal-directed value from trajectory).
    pub pragmatic_value: f64,
    /// Integrated surprise along the trajectory path.
    pub trajectory_surprise: f64,
    /// Number of ODE steps taken.
    pub ode_steps: usize,
    /// Final state at end of trajectory.
    pub final_state: Vec<f64>,
}

/// Telemetry from trajectory planning.
#[derive(Debug, Clone, Default)]
pub struct TrajectoryTelemetry {
    /// Expected free energy of the best trajectory.
    pub best_efe: f64,
    /// Action selected by trajectory planning.
    pub best_action: usize,
    /// Planning horizon used.
    pub horizon: f64,
    /// Total ODE steps across all action rollouts.
    pub total_ode_steps: usize,
    /// Number of actions evaluated.
    pub actions_evaluated: usize,
    /// Surprise from the best trajectory.
    pub best_trajectory_surprise: f64,
}

/// Record of a past trajectory planning decision.
#[derive(Debug, Clone)]
pub struct TrajectoryRecord {
    /// Cycle at which planning occurred.
    pub cycle: u64,
    /// Best action selected.
    pub action: usize,
    /// Expected free energy of best trajectory.
    pub efe: f64,
    /// Surprise of best trajectory.
    pub surprise: f64,
}

// ─── FEP Module ─────────────────────────────────────────────────────────────

/// Consolidated Free Energy Principle module.
///
/// Groups all FEP/active-inference state that was previously scattered across
/// 10 separate fields in CognitiveLoopService.
pub struct FepModule {
    /// Active Inference Bridge for precision-weighted prediction.
    pub active_inference_bridge: ActiveInferenceBridge,

    /// Thermodynamic ledger for tracking energy costs of inference.
    pub ledger: ThermodynamicLedger,

    /// Closed Learning Loop for strategy-based behavioral adaptation.
    pub closed_learning_loop: ClosedLearningLoop,

    /// Episodic Memory Bridge for memory encoding and recall during cycles.
    pub episodic_memory: EpisodicMemoryBridge,

    /// Goal System Bridge for goal-directed attention modulation.
    pub goal_system: GoalSystemBridge,

    /// World Model Bridge for hierarchical grounded prediction.
    pub world_model: WorldModelBridge,

    /// FEP Active Inference Agent for full perception-action loop.
    pub agent: ActiveInferenceAgent,

    /// Proprioceptive-Semantic binder for somatic grounding.
    pub haptic_semantic_binder: symthaea_fep::HapticSemanticBinder,

    /// Enhanced FEP Bridge with motor system integration.
    pub enhanced_bridge: EnhancedFEPBridge,

    /// Current learning signal from FEP (for downstream systems).
    pub learning_signal: f32,
    pub last_action_idx: u8,

    /// FEP-driven learning rate boost (applied during CfC training step).
    /// Range: [1.0, 3.0].
    pub lr_boost: f64,

    /// Surprise-driven exploration bridge for FEP-based exploration.
    pub surprise_bridge: Option<SurpriseExplorationBridge>,

    /// Configuration for ODE-based trajectory planning.
    pub trajectory_config: TrajectoryPlanningConfig,

    /// Latest trajectory planning telemetry.
    pub trajectory_telemetry: TrajectoryTelemetry,

    /// Ring buffer of recent trajectory planning decisions.
    pub trajectory_history: VecDeque<TrajectoryRecord>,
}

impl FepModule {
    /// Create a new FEP module.
    pub fn new(
        agent: ActiveInferenceAgent,
        ledger: ThermodynamicLedger,
        closed_learning_loop: ClosedLearningLoop,
        episodic_memory: EpisodicMemoryBridge,
        goal_system: GoalSystemBridge,
        world_model: WorldModelBridge,
        haptic_semantic_binder: symthaea_fep::HapticSemanticBinder,
        enhanced_bridge: EnhancedFEPBridge,
    ) -> Self {
        Self {
            active_inference_bridge: ActiveInferenceBridge::default(),
            ledger,
            closed_learning_loop,
            episodic_memory,
            goal_system,
            world_model,
            agent,
            haptic_semantic_binder,
            enhanced_bridge,
            learning_signal: 0.0,
            last_action_idx: 0,
            lr_boost: 1.0,
            surprise_bridge: None,
            trajectory_config: TrajectoryPlanningConfig::default(),
            trajectory_telemetry: TrajectoryTelemetry::default(),
            trajectory_history: VecDeque::new(),
        }
    }

    /// Run ODE-based trajectory planning if enabled and at the right interval.
    ///
    /// Simulates forward trajectories for each available action using
    /// Dormand-Prince adaptive ODE integration on the generative model's
    /// transition dynamics. Selects the action with lowest cumulative
    /// expected free energy.
    ///
    /// Returns `Some(best_action)` if planning ran, `None` if skipped.
    pub fn plan_trajectories(&mut self, cycle: u64) -> Option<usize> {
        if !self.trajectory_config.enabled {
            return None;
        }
        if cycle % self.trajectory_config.planning_interval != 0 {
            return None;
        }

        let num_actions = self.agent.config.num_actions;
        let state_dim = self.agent.config.state_dim;
        let belief_state = &self.agent.belief.mean;
        let model = &self.agent.model;

        // Set up ODE solver with Dormand-Prince adaptive stepping
        let ode_config = OdeConfig {
            solver: OdeSolver::DormandPrince,
            dt: 0.01,
            tolerance: self.trajectory_config.ode_tolerance,
            max_step: self.trajectory_config.horizon_seconds / 5.0,
            min_step: 1e-8,
            max_iterations: self.trajectory_config.max_steps,
        };
        let solver = OdeSolverEngine::new(ode_config, state_dim);

        let mut rollouts = Vec::with_capacity(num_actions);
        let mut total_ode_steps = 0;

        for action in 0..num_actions {
            let action_idx = action.min(model.num_actions - 1);
            let ode_system = GenerativeModelOde::from_transition(
                &model.transition_matrices[action_idx],
                self.trajectory_config.tau,
            );

            // Integrate from current belief state over planning horizon
            let result: OdeResult = solver.solve(
                &ode_system,
                belief_state,
                (0.0, self.trajectory_config.horizon_seconds),
            );

            let ode_steps = result.times.len();
            total_ode_steps += ode_steps;

            // Compute expected free energy along trajectory via trapezoidal rule
            let (efe, info_gain, pragmatic, surprise) =
                Self::compute_trajectory_efe(&result, model, belief_state);

            let final_state = result
                .states
                .last()
                .cloned()
                .unwrap_or_else(|| belief_state.to_vec());

            rollouts.push(TrajectoryRollout {
                action,
                expected_free_energy: efe,
                information_gain: info_gain,
                pragmatic_value: pragmatic,
                trajectory_surprise: surprise,
                ode_steps,
                final_state,
            });
        }

        // Select action with lowest EFE (most preferred under active inference)
        let best = rollouts
            .iter()
            .enumerate()
            .min_by(|a, b| {
                a.1.expected_free_energy
                    .partial_cmp(&b.1.expected_free_energy)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let best_rollout = &rollouts[best];

        // Update telemetry
        self.trajectory_telemetry = TrajectoryTelemetry {
            best_efe: best_rollout.expected_free_energy,
            best_action: best_rollout.action,
            horizon: self.trajectory_config.horizon_seconds,
            total_ode_steps,
            actions_evaluated: num_actions,
            best_trajectory_surprise: best_rollout.trajectory_surprise,
        };

        // Record in history ring buffer
        if self.trajectory_history.len() >= TRAJECTORY_HISTORY_CAP {
            self.trajectory_history.pop_front();
        }
        self.trajectory_history.push_back(TrajectoryRecord {
            cycle,
            action: best_rollout.action,
            efe: best_rollout.expected_free_energy,
            surprise: best_rollout.trajectory_surprise,
        });

        Some(best_rollout.action)
    }

    /// Compute expected free energy along an ODE trajectory.
    ///
    /// EFE decomposes into:
    /// - **Pragmatic value**: distance from preferred states (goal-directedness)
    /// - **Information gain**: reduction in belief uncertainty (curiosity)
    /// - **Surprise**: prediction error at each trajectory point
    ///
    /// Uses trapezoidal integration over the ODE output points.
    fn compute_trajectory_efe(
        result: &OdeResult,
        model: &symthaea_fep::GenerativeModel,
        initial_state: &[f64],
    ) -> (f64, f64, f64, f64) {
        let n_points = result.states.len();
        if n_points < 2 {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let mut cumulative_efe = 0.0;
        let mut cumulative_info_gain = 0.0;
        let mut cumulative_pragmatic = 0.0;
        let mut cumulative_surprise = 0.0;

        for i in 1..n_points {
            let dt = result.times[i] - result.times[i - 1];
            if dt <= 0.0 {
                continue;
            }

            let state = &result.states[i];
            let prev_state = &result.states[i - 1];

            // Pragmatic value: divergence from prior (preferred) states
            let pragmatic: f64 = state
                .iter()
                .zip(model.prior_mean.iter())
                .map(|(s, p)| (s - p).powi(2))
                .sum::<f64>()
                .sqrt();

            // Information gain: change in state entropy (more concentrated = less uncertainty)
            let state_entropy: f64 = state
                .iter()
                .map(|s| {
                    let p = s.abs().clamp(1e-10, 1.0);
                    -p * p.ln()
                })
                .sum();
            let prev_entropy: f64 = prev_state
                .iter()
                .map(|s| {
                    let p = s.abs().clamp(1e-10, 1.0);
                    -p * p.ln()
                })
                .sum();
            let info_gain = (prev_entropy - state_entropy).max(0.0);

            // Surprise: prediction error relative to initial belief
            let surprise: f64 = state
                .iter()
                .zip(initial_state.iter())
                .map(|(s, i)| (s - i).powi(2))
                .sum::<f64>()
                .sqrt();

            // EFE = pragmatic_risk - information_gain (lower is better)
            let local_efe = pragmatic - info_gain;

            // Trapezoidal integration
            cumulative_efe += local_efe * dt;
            cumulative_info_gain += info_gain * dt;
            cumulative_pragmatic += pragmatic * dt;
            cumulative_surprise += surprise * dt;
        }

        (
            cumulative_efe,
            cumulative_info_gain,
            cumulative_pragmatic,
            cumulative_surprise,
        )
    }

    /// Enable trajectory planning with default configuration.
    pub fn enable_trajectory_planning(&mut self) {
        self.trajectory_config.enabled = true;
    }

    /// Get the latest trajectory telemetry.
    pub fn trajectory_telemetry(&self) -> &TrajectoryTelemetry {
        &self.trajectory_telemetry
    }

    /// Get the trajectory planning history.
    pub fn trajectory_history(&self) -> &VecDeque<TrajectoryRecord> {
        &self.trajectory_history
    }

    // ── Markov Blanket Integration ───────────────────────────────────────

    /// Update the Markov blanket permeability from neuromodulator state.
    pub fn update_blanket_permeability(
        &mut self,
        acetylcholine: f64,
        noradrenaline: f64,
        serotonin: f64,
        oxytocin: f64,
        threat_level: f64,
        flow_state: f64,
        peer_trust: f64,
    ) {
        use crate::consciousness::fep_active_inference::PermeabilityInputs;
        let inputs = PermeabilityInputs {
            acetylcholine,
            noradrenaline,
            serotonin,
            oxytocin,
            threat_level,
            peer_trust,
            flow_state,
        };
        self.enhanced_bridge.update_blanket_permeability(&inputs);
    }

    /// Apply topological boundary constraints to the Markov blanket.
    pub fn apply_topology_constraints(
        &mut self,
        boundary_thickness: f64,
        fiedler_value: f64,
        boundary_components: usize,
    ) {
        use crate::consciousness::fep_active_inference::TopologyBoundaryInputs;
        let topo = TopologyBoundaryInputs {
            boundary_thickness,
            fiedler_value,
            boundary_components,
        };
        self.enhanced_bridge
            .blanket
            .apply_topology_constraints(&topo);
    }

    /// Apply active inference motor reflexes based on prediction error (Channel 1/3).
    ///
    /// When surprise is high, triggers reflexive motor actions to change the physical
    /// world to match expectations, minimizing variational free energy.
    /// Incorporates haptic-semantic constraints to bias reflexes away from fragile configurations.
    pub fn apply_active_reflex(
        &mut self,
        bridge: &mut dyn symthaea_core::embodiment::EmbodimentBridge,
        prediction: &symthaea_core::hdc::unified_hv::ContinuousHV,
        observation: &symthaea_core::hdc::unified_hv::ContinuousHV,
        proprioception: &[f32],
    ) {
        // Project proprioceptive state into semantic space, adaptively dampened by
        // actuator health. Actuator indices don't align with proprioception state
        // dims (state layouts are platform-specific), so degraded hardware dampens
        // the haptic constraint uniformly via mean health rather than per-index.
        let reported = bridge.actuator_health();
        let mean_health = if reported.is_empty() {
            1.0
        } else {
            (reported.iter().sum::<f32>() / reported.len() as f32).clamp(0.0, 1.0)
        };
        let health = vec![mean_health; proprioception.len()];
        let haptic_constraint = self
            .haptic_semantic_binder
            .bind_with_health(proprioception, &health);

        // Calculate raw prediction error vector
        let error_hv = symthaea_core::core::ContinuousHV::encode_weighted(
            &[prediction.clone(), observation.clone()],
            &[1.0, -1.0],
        );

        // Modulate reflexes by somatic constraint (bind error with haptic prior)
        // High strain/instability → higher gain on reflexes that counteract the instability
        let modulated_error = error_hv.bind(&haptic_constraint);

        // Trigger reflexive motor impulse via bridge
        bridge.apply_active_inference(&modulated_error);
    }

    /// Deduct energy from the ledger for cognitive operations.
    pub fn deduct_cognitive_cost(&mut self, cost_j: f64) -> bool {
        self.ledger.deduct(cost_j)
    }

    /// Get the current effective blanket permeability.
    pub fn blanket_effective_permeability(&self) -> f64 {
        self.enhanced_bridge.blanket.permeability().effective
    }
}
