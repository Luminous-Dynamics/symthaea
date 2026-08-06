// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Full Active Inference Agent implementing the perception-action loop.

use serde::{Deserialize, Serialize};

use super::free_energy::{ExpectedFreeEnergyComputer, FreeEnergyCalculator, PrecisionEstimator};
use super::generative_model::GenerativeModel;
use super::td_learning::{
    TemporalDifferenceLearner, TemporalDifferenceLearningConfig, TemporalDifferenceLearningStats,
};
use super::types::{
    ActionOutcome, ActionSelectionResult, ActiveInferenceAgentStats, ActiveInferenceSummary,
    FreeEnergyComponents, HiddenState, Observation, PerceptionResult,
};

// =============================================================================
// ACTIVE INFERENCE AGENT CONFIG
// =============================================================================

/// Configuration for Active Inference Agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveInferenceAgentConfig {
    /// Hidden state dimension
    pub state_dim: usize,
    /// Observation dimension
    pub obs_dim: usize,
    /// Number of available actions
    pub num_actions: usize,
    /// Belief update iterations per perception step
    pub inference_iterations: usize,
    /// Learning rate for belief updates
    pub belief_learning_rate: f64,
    /// Planning horizon for action selection
    pub planning_horizon: usize,
    /// Action selection temperature (softmax)
    pub action_temperature: f64,
    /// Whether to enable model learning
    pub enable_model_learning: bool,
    /// Whether to enable temporal difference learning
    pub enable_td_learning: bool,
    /// Temporal difference learning configuration
    pub td_config: TemporalDifferenceLearningConfig,
}

impl Default for ActiveInferenceAgentConfig {
    fn default() -> Self {
        Self {
            state_dim: 8,
            obs_dim: 4,
            num_actions: 6,
            inference_iterations: 5,
            belief_learning_rate: 0.1,
            planning_horizon: 3,
            action_temperature: 1.0,
            enable_model_learning: true,
            enable_td_learning: true,
            td_config: TemporalDifferenceLearningConfig::default(),
        }
    }
}

// =============================================================================
// ACTIVE INFERENCE AGENT
// =============================================================================

/// Full Active Inference Agent implementing perception-action loop
#[derive(Debug, Clone)]
pub struct ActiveInferenceAgent {
    /// Configuration
    pub config: ActiveInferenceAgentConfig,
    /// Current belief state
    pub belief: HiddenState,
    /// Previous belief state (for temporal difference learning)
    pub previous_state: Option<HiddenState>,
    /// Last action taken (for temporal difference learning)
    pub last_action: Option<usize>,
    /// Generative model
    pub model: GenerativeModel,
    /// Free energy calculator
    pub free_energy_calc: FreeEnergyCalculator,
    /// Precision estimator
    pub precision: PrecisionEstimator,
    /// Expected free energy computer
    pub efe_computer: ExpectedFreeEnergyComputer,
    /// Temporal difference learner
    pub td_learner: Option<TemporalDifferenceLearner>,
    /// Last free energy components
    pub last_fe_components: Option<FreeEnergyComponents>,
    /// Statistics
    pub stats: ActiveInferenceAgentStats,
    /// Current timestamp counter
    timestamp: u64,
    /// RNG state for stochastic action sampling
    rng_state: u64,
}

impl ActiveInferenceAgent {
    /// Create new active inference agent
    pub fn new(config: ActiveInferenceAgentConfig) -> Self {
        let model = GenerativeModel::new(config.state_dim, config.obs_dim, config.num_actions);
        let belief = HiddenState::new(config.state_dim);
        let free_energy_calc = FreeEnergyCalculator::new(500);
        let precision = PrecisionEstimator::new();
        let efe_computer = ExpectedFreeEnergyComputer::new(config.obs_dim);

        let td_learner = if config.enable_td_learning {
            Some(TemporalDifferenceLearner::new(
                config.td_config.clone(),
                config.num_actions,
                config.state_dim,
                config.obs_dim,
            ))
        } else {
            None
        };

        Self {
            config,
            belief,
            previous_state: None,
            last_action: None,
            model,
            free_energy_calc,
            precision,
            efe_computer,
            td_learner,
            last_fe_components: None,
            stats: ActiveInferenceAgentStats::default(),
            timestamp: 0,
            rng_state: 0x9E3779B97F4A7C15, // Golden ratio hash — good default seed
        }
    }

    /// Re-seed this agent's internal action-selection RNG.
    ///
    /// `new()` always starts from the same fixed golden-ratio constant, so
    /// by default every agent instance's stochastic action selection
    /// follows an identical sequence regardless of anything caller-specific
    /// -- fine for a single agent, but a real confound for experiments that
    /// construct many independent agent instances and expect their
    /// stochastic decisions to vary independently (e.g. one agent per
    /// simulation replicate, seeded from that replicate's own seed). Purely
    /// additive: existing callers that never call this see no change in
    /// behavior.
    pub fn set_rng_seed(&mut self, seed: u64) {
        // xorshift64 requires a non-zero state; a zero seed would silently
        // produce a degenerate all-zero stream.
        self.rng_state = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
    }

    /// Perception step: Update beliefs to minimize free energy
    ///
    /// This implements variational inference:
    /// q(s) ← argmin_q F[q, o]
    pub fn perceive(&mut self, observation: &Observation) -> PerceptionResult {
        self.timestamp += 1;

        // Store previous state for TD learning before updating
        let old_state = self.belief.clone();

        // Run belief update iterations
        let mut total_belief_change = 0.0;
        for _ in 0..self.config.inference_iterations {
            let change = self.update_belief(observation);
            total_belief_change += change;
        }

        // Compute free energy
        let fe_components = self
            .free_energy_calc
            .compute(&self.belief, observation, &self.model);
        self.last_fe_components = Some(fe_components.clone());

        // Update precision based on prediction error
        self.precision
            .update_from_error(fe_components.prediction_error, self.timestamp);

        // Temporal difference learning: if we have a previous state and action
        if let (Some(prev_state), Some(action)) = (&self.previous_state, self.last_action)
            && let Some(ref mut td_learner) = self.td_learner
        {
            // Observe the transition and compute TD error
            let td_error = td_learner.observe_transition(
                prev_state,
                action,
                &self.belief,
                observation,
                &self.model,
                self.timestamp,
            );

            // Update generative model using TD learning
            td_learner.update_model(
                &mut self.model,
                prev_state,
                action,
                &self.belief,
                observation,
                td_error,
            );

            // Update stats
            self.stats.td_updates += 1;
            let n = self.stats.td_updates as f64;
            self.stats.avg_td_error = (self.stats.avg_td_error * (n - 1.0) + td_error.abs()) / n;
            self.stats.transition_accuracy = td_learner.avg_prediction_accuracy;
        }

        // Also use direct model learning (Hebbian-like), but only if TD learning is not active
        // to avoid conflicting updates to the same matrices
        if self.config.enable_model_learning && self.td_learner.is_none() {
            self.model
                .learn(&self.belief, observation, self.last_action);
            self.stats.model_updates += 1;
        }

        // Store pre-update state for next TD update (old_state captured before belief update)
        self.previous_state = Some(old_state);

        // Update stats
        self.stats.perception_cycles += 1;
        let n = self.stats.perception_cycles as f64;
        self.stats.avg_free_energy =
            (self.stats.avg_free_energy * (n - 1.0) + fe_components.total) / n;
        self.stats.avg_prediction_error =
            (self.stats.avg_prediction_error * (n - 1.0) + fe_components.prediction_error) / n;
        self.stats.avg_precision =
            (self.stats.avg_precision * (n - 1.0) + self.precision.perceptual_precision()) / n;

        PerceptionResult {
            updated_belief: self.belief.clone(),
            free_energy: fe_components,
            precision: self.precision.perceptual_precision(),
            belief_change: total_belief_change,
            timestamp: self.timestamp,
        }
    }

    /// Update belief based on observation (single iteration)
    fn update_belief(&mut self, observation: &Observation) -> f64 {
        // Compute prediction error
        let prediction_error = self.model.prediction_error(&self.belief, observation);

        // Compute precision-weighted error
        let weighted_error: Vec<f64> = prediction_error
            .iter()
            .map(|e| e * self.precision.sensory_precision)
            .collect();

        // Update belief mean (gradient descent on free energy)
        let mut total_change = 0.0;
        for i in 0..self.belief.mean.len() {
            // Aggregate error from likelihood matrix
            let mut grad = 0.0;
            for j in 0..weighted_error.len() {
                if i < self.model.likelihood_matrix.len()
                    && j < self.model.likelihood_matrix[i].len()
                {
                    grad += self.model.likelihood_matrix[i][j] * weighted_error[j];
                }
            }

            // Add prior gradient (pull toward prior mean)
            let prior_grad = self.precision.prior_precision
                * (self.model.prior_mean.get(i).copied().unwrap_or(0.5) - self.belief.mean[i]);

            // Update
            let delta = self.config.belief_learning_rate * (grad + prior_grad * 0.1);
            self.belief.mean[i] += delta;
            self.belief.mean[i] = self.belief.mean[i].clamp(0.0, 1.0);
            total_change += delta.abs();
        }

        // Update belief precision based on prediction accuracy
        for i in 0..self.belief.precision.len() {
            let error_i = if i < prediction_error.len() {
                prediction_error[i].abs()
            } else {
                0.5
            };
            // Higher error → lower precision
            let new_precision = 1.0 / (1.0 + error_i);
            self.belief.precision[i] = 0.9 * self.belief.precision[i] + 0.1 * new_precision;
        }

        total_change
    }

    /// Action selection: Choose action that minimizes expected free energy
    ///
    /// a* = argmin_a G(a)
    pub fn select_action(&mut self) -> ActionSelectionResult {
        // Compute expected free energy for each action
        let mut efe_results: Vec<super::types::ExpectedFreeEnergyResult> = Vec::new();

        for action in 0..self.config.num_actions {
            let efe = self.efe_computer.compute(action, &self.belief, &self.model);
            efe_results.push(efe);
        }

        // Softmax action selection
        let max_efe = efe_results
            .iter()
            .map(|r| -r.total)
            .fold(f64::NEG_INFINITY, f64::max);
        let exp_values: Vec<f64> = efe_results
            .iter()
            .map(|r| ((-r.total - max_efe) / self.config.action_temperature).exp())
            .collect();
        let sum_exp: f64 = exp_values.iter().sum();
        let probabilities: Vec<f64> = exp_values.iter().map(|e| e / sum_exp).collect();

        // Stochastic action selection: sample from softmax distribution
        let selected_idx = {
            // xorshift64 step
            self.rng_state ^= self.rng_state << 13;
            self.rng_state ^= self.rng_state >> 7;
            self.rng_state ^= self.rng_state << 17;
            let u = (self.rng_state as f64) / (u64::MAX as f64); // uniform [0, 1)
            let mut cumulative = 0.0;
            let mut idx = probabilities.len() - 1;
            for (i, &p) in probabilities.iter().enumerate() {
                cumulative += p;
                if u < cumulative {
                    idx = i;
                    break;
                }
            }
            idx
        };

        let selected = &efe_results[selected_idx];

        // Determine if this is exploratory (high epistemic value)
        let is_exploratory = selected.epistemic.abs() > selected.pragmatic.abs();
        if is_exploratory {
            self.stats.epistemic_actions += 1;
        }

        // Update stats
        self.stats.actions_taken += 1;
        self.stats.exploration_rate =
            self.stats.epistemic_actions as f64 / self.stats.actions_taken as f64;

        ActionSelectionResult {
            action: selected.action,
            expected_free_energy: selected.total,
            action_probabilities: probabilities,
            is_exploratory,
            pragmatic_value: selected.pragmatic,
            epistemic_value: selected.epistemic,
            predicted_state: selected.predicted_state.clone(),
        }
    }

    /// Execute action and observe outcome
    pub fn act(&mut self, action: usize) -> ActionOutcome {
        // Track the action for temporal difference learning
        self.last_action = Some(action);

        let predicted_state = self.model.predict_next_state(&self.belief, action);
        let expected_obs = self.model.predict_observation(&predicted_state);

        // In a real system, the action would be executed and observation received
        // Here we just predict what would happen
        ActionOutcome {
            action,
            predicted_next_state: predicted_state,
            expected_observation: expected_obs,
            timestamp: self.timestamp,
        }
    }

    /// Learn from action outcome
    pub fn learn_from_outcome(&mut self, action: usize, actual_observation: &Observation) {
        // Track action for TD learning
        self.last_action = Some(action);

        // Update belief to incorporate new observation
        // This will trigger TD learning internally
        let _ = self.perceive(actual_observation);

        // Update generative model with action information
        if self.config.enable_model_learning {
            self.model
                .learn(&self.belief, actual_observation, Some(action));
        }

        // Update action precision
        let expected = self.model.predict_observation(&self.belief);
        let expected_phi = expected.first().copied().unwrap_or(0.5);
        let actual_phi = actual_observation.values.first().copied().unwrap_or(0.5);
        self.precision
            .update_from_action(expected_phi, actual_phi, self.timestamp);
    }

    /// Observe a full transition (old_state, action, new_state, observation)
    ///
    /// This is the primary interface for temporal difference learning,
    /// allowing external systems to provide complete transition information.
    pub fn observe_transition(
        &mut self,
        old_state: &HiddenState,
        action: usize,
        new_state: &HiddenState,
        observation: &Observation,
    ) -> Option<f64> {
        if let Some(ref mut td_learner) = self.td_learner {
            // Observe transition and compute TD error
            let td_error = td_learner.observe_transition(
                old_state,
                action,
                new_state,
                observation,
                &self.model,
                self.timestamp,
            );

            // Update model
            td_learner.update_model(
                &mut self.model,
                old_state,
                action,
                new_state,
                observation,
                td_error,
            );

            // Update stats
            self.stats.td_updates += 1;
            let n = self.stats.td_updates as f64;
            self.stats.avg_td_error = (self.stats.avg_td_error * (n - 1.0) + td_error.abs()) / n;
            self.stats.transition_accuracy = td_learner.avg_prediction_accuracy;

            Some(td_error)
        } else {
            // Fallback: use direct model learning
            self.model
                .learn_transition(old_state, action, new_state, observation);
            None
        }
    }

    /// Signal end of episode (for learning rate decay and trace reset)
    pub fn end_episode(&mut self) {
        if let Some(ref mut td_learner) = self.td_learner {
            td_learner.decay_learning_rate();
            td_learner.reset_traces();
        }
    }

    /// Get temporal difference learning statistics
    pub fn td_stats(&self) -> Option<TemporalDifferenceLearningStats> {
        self.td_learner.as_ref().map(|td| td.stats())
    }

    /// Set goal preferences for the agent
    pub fn set_goals(&mut self, preferences: Vec<f64>, precision: f64) {
        self.efe_computer.set_preferences(preferences, precision);
    }

    /// Set goal preferences with an explicit per-dimension precision -- see
    /// [`super::free_energy::ExpectedFreeEnergyComputer::set_preferences_with_precisions`] for
    /// why this exists (a shared scalar precision can't express "no preference on this specific
    /// channel").
    pub fn set_goals_with_precisions(&mut self, preferences: Vec<f64>, precisions: Vec<f64>) {
        self.efe_computer
            .set_preferences_with_precisions(preferences, precisions);
    }

    /// Get current free energy
    pub fn current_free_energy(&self) -> f64 {
        self.last_fe_components
            .as_ref()
            .map(|c| c.total)
            .unwrap_or(0.0)
    }

    /// Check if agent is in surprised state (high free energy)
    pub fn is_surprised(&self) -> bool {
        self.current_free_energy() > 2.0 || self.stats.avg_prediction_error > 0.5
    }

    /// Get summary of agent state
    pub fn summary(&self) -> ActiveInferenceSummary {
        ActiveInferenceSummary {
            belief_mean: self.belief.mean.clone(),
            belief_confidence: self.belief.confidence(),
            free_energy: self.current_free_energy(),
            precision: self.precision.perceptual_precision(),
            exploration_rate: self.stats.exploration_rate,
            total_cycles: self.stats.perception_cycles,
        }
    }

    /// Inject explicit priors into the generative model (Passport Route).
    pub fn inject_priors(&mut self, mean: Vec<f64>, precision: Vec<f64>) {
        self.model.inject_priors(mean, precision);
    }

    /// Set the additive transition bias for `action` (Passport Route for
    /// transition dynamics — see `GenerativeModel::transition_bias`'s doc).
    pub fn set_transition_bias(&mut self, action: usize, bias: Vec<f64>) {
        self.model.set_transition_bias(action, bias);
    }

    /// Reset agent state
    pub fn reset(&mut self) {
        self.belief = HiddenState::new(self.config.state_dim);
        self.previous_state = None;
        self.last_action = None;
        self.precision = PrecisionEstimator::new();
        self.free_energy_calc = FreeEnergyCalculator::new(500);
        self.efe_computer = ExpectedFreeEnergyComputer::new(self.config.obs_dim);

        // Reset TD learner but preserve configuration
        if self.config.enable_td_learning {
            self.td_learner = Some(TemporalDifferenceLearner::new(
                self.config.td_config.clone(),
                self.config.num_actions,
                self.config.state_dim,
                self.config.obs_dim,
            ));
        }

        self.last_fe_components = None;
        self.stats = ActiveInferenceAgentStats::default();
        self.timestamp = 0;
    }
}
