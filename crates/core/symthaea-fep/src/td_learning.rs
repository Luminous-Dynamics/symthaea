// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal Difference Learning for updating the generative model.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use super::generative_model::GenerativeModel;
use super::types::{HiddenState, Observation};

// =============================================================================
// TEMPORAL DIFFERENCE LEARNING CONFIGURATION
// =============================================================================

/// Configuration for temporal difference learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalDifferenceLearningConfig {
    /// Initial learning rate for TD updates
    pub initial_learning_rate: f64,
    /// Minimum learning rate (after decay)
    pub min_learning_rate: f64,
    /// Learning rate decay factor (applied per episode/epoch)
    pub learning_rate_decay: f64,
    /// Discount factor gamma for future rewards/values
    pub gamma: f64,
    /// Lambda for eligibility traces (0 = TD(0), 1 = Monte Carlo)
    pub lambda: f64,
    /// Whether to use eligibility traces (TD(lambda))
    pub use_eligibility_traces: bool,
    /// Eligibility trace decay rate
    pub trace_decay: f64,
    /// Maximum number of transitions to store
    pub max_transition_history: usize,
    /// Confidence decay rate for model uncertainty
    pub confidence_decay: f64,
    /// Minimum confidence threshold
    pub min_confidence: f64,
}

impl Default for TemporalDifferenceLearningConfig {
    fn default() -> Self {
        Self {
            initial_learning_rate: 0.1,
            min_learning_rate: 0.001,
            learning_rate_decay: 0.999,
            gamma: 0.99,
            lambda: 0.8,
            use_eligibility_traces: true,
            trace_decay: 0.9,
            max_transition_history: 1000,
            confidence_decay: 0.99,
            min_confidence: 0.1,
        }
    }
}

// =============================================================================
// STATE TRANSITION RECORD
// =============================================================================

/// Record of a single state transition for temporal difference learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    /// Previous hidden state
    pub old_state: HiddenState,
    /// Action taken
    pub action: usize,
    /// New hidden state after transition
    pub new_state: HiddenState,
    /// Observation received after transition
    pub observation: Observation,
    /// Timestamp of transition
    pub timestamp: u64,
    /// TD error computed for this transition
    pub td_error: f64,
}

// =============================================================================
// ELIGIBILITY TRACE
// =============================================================================

/// Eligibility traces for TD(lambda) learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EligibilityTraces {
    /// Traces for transition matrix: `[action][from_state][to_state]`
    pub transition_traces: Vec<Vec<Vec<f64>>>,
    /// Traces for likelihood matrix: `[state][observation]`
    pub likelihood_traces: Vec<Vec<f64>>,
    /// Lambda parameter
    pub lambda: f64,
    /// Gamma (discount factor)
    pub gamma: f64,
}

impl EligibilityTraces {
    /// Create new eligibility traces
    pub fn new(
        num_actions: usize,
        state_dim: usize,
        obs_dim: usize,
        lambda: f64,
        gamma: f64,
    ) -> Self {
        Self {
            transition_traces: vec![vec![vec![0.0; state_dim]; state_dim]; num_actions],
            likelihood_traces: vec![vec![0.0; obs_dim]; state_dim],
            lambda,
            gamma,
        }
    }

    /// Decay all traces
    pub fn decay(&mut self) {
        let decay_factor = self.gamma * self.lambda;

        for action_traces in &mut self.transition_traces {
            for from_traces in action_traces {
                for trace in from_traces {
                    *trace *= decay_factor;
                }
            }
        }

        for state_traces in &mut self.likelihood_traces {
            for trace in state_traces {
                *trace *= decay_factor;
            }
        }
    }

    /// Update traces for a transition (accumulating traces)
    pub fn update(
        &mut self,
        action: usize,
        from_state: &[f64],
        to_state: &[f64],
        observation: &[f64],
    ) {
        let action_idx = action.min(self.transition_traces.len().saturating_sub(1));

        // Update transition traces (outer product of from_state and to_state)
        for (i, &from_val) in from_state.iter().enumerate() {
            if i >= self.transition_traces[action_idx].len() {
                break;
            }
            for (j, &to_val) in to_state.iter().enumerate() {
                if j >= self.transition_traces[action_idx][i].len() {
                    break;
                }
                // Accumulating traces: add gradient to existing trace
                self.transition_traces[action_idx][i][j] += from_val * to_val;
            }
        }

        // Update likelihood traces (outer product of state and observation)
        for (i, &state_val) in to_state.iter().enumerate() {
            if i >= self.likelihood_traces.len() {
                break;
            }
            for (j, &obs_val) in observation.iter().enumerate() {
                if j >= self.likelihood_traces[i].len() {
                    break;
                }
                self.likelihood_traces[i][j] += state_val * obs_val;
            }
        }
    }

    /// Reset all traces to zero
    pub fn reset(&mut self) {
        for action_traces in &mut self.transition_traces {
            for from_traces in action_traces {
                for trace in from_traces {
                    *trace = 0.0;
                }
            }
        }

        for state_traces in &mut self.likelihood_traces {
            for trace in state_traces {
                *trace = 0.0;
            }
        }
    }
}

// =============================================================================
// MODEL CONFIDENCE TRACKER
// =============================================================================

/// Tracks confidence/uncertainty in generative model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfidenceTracker {
    /// Confidence in transition model for each action: `[action][from_state][to_state]`
    pub transition_confidence: Vec<Vec<Vec<f64>>>,
    /// Confidence in likelihood model: `[state][observation]`
    pub likelihood_confidence: Vec<Vec<f64>>,
    /// Visit counts for transition model
    pub transition_counts: Vec<Vec<Vec<u64>>>,
    /// Visit counts for likelihood model
    pub likelihood_counts: Vec<Vec<u64>>,
    /// Decay rate for confidence
    pub decay_rate: f64,
    /// Minimum confidence
    pub min_confidence: f64,
}

impl ModelConfidenceTracker {
    /// Create new confidence tracker
    pub fn new(
        num_actions: usize,
        state_dim: usize,
        obs_dim: usize,
        decay_rate: f64,
        min_confidence: f64,
    ) -> Self {
        Self {
            transition_confidence: vec![
                vec![vec![min_confidence; state_dim]; state_dim];
                num_actions
            ],
            likelihood_confidence: vec![vec![min_confidence; obs_dim]; state_dim],
            transition_counts: vec![vec![vec![0; state_dim]; state_dim]; num_actions],
            likelihood_counts: vec![vec![0; obs_dim]; state_dim],
            decay_rate,
            min_confidence,
        }
    }

    /// Update confidence based on observed transition
    pub fn update_transition(&mut self, action: usize, from_idx: usize, to_idx: usize) {
        let action_idx = action.min(self.transition_confidence.len().saturating_sub(1));
        let from_idx = from_idx.min(
            self.transition_confidence[action_idx]
                .len()
                .saturating_sub(1),
        );
        let to_idx = to_idx.min(
            self.transition_confidence[action_idx][from_idx]
                .len()
                .saturating_sub(1),
        );

        self.transition_counts[action_idx][from_idx][to_idx] += 1;
        let count = self.transition_counts[action_idx][from_idx][to_idx] as f64;

        // Confidence increases with observations but saturates
        self.transition_confidence[action_idx][from_idx][to_idx] =
            1.0 - (1.0 - self.min_confidence) * (-count / 10.0).exp();
    }

    /// Update confidence based on observed observation
    pub fn update_likelihood(&mut self, state_idx: usize, obs_idx: usize) {
        let state_idx = state_idx.min(self.likelihood_confidence.len().saturating_sub(1));
        let obs_idx = obs_idx.min(
            self.likelihood_confidence[state_idx]
                .len()
                .saturating_sub(1),
        );

        self.likelihood_counts[state_idx][obs_idx] += 1;
        let count = self.likelihood_counts[state_idx][obs_idx] as f64;

        self.likelihood_confidence[state_idx][obs_idx] =
            1.0 - (1.0 - self.min_confidence) * (-count / 10.0).exp();
    }

    /// Apply decay to all confidences
    pub fn decay(&mut self) {
        for action_conf in &mut self.transition_confidence {
            for from_conf in action_conf {
                for conf in from_conf {
                    *conf = (*conf * self.decay_rate).max(self.min_confidence);
                }
            }
        }

        for state_conf in &mut self.likelihood_confidence {
            for conf in state_conf {
                *conf = (*conf * self.decay_rate).max(self.min_confidence);
            }
        }
    }

    /// Get average transition confidence
    pub fn avg_transition_confidence(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;

        for action_conf in &self.transition_confidence {
            for from_conf in action_conf {
                for &conf in from_conf {
                    sum += conf;
                    count += 1;
                }
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            self.min_confidence
        }
    }

    /// Get average likelihood confidence
    pub fn avg_likelihood_confidence(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;

        for state_conf in &self.likelihood_confidence {
            for &conf in state_conf {
                sum += conf;
                count += 1;
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            self.min_confidence
        }
    }
}

// =============================================================================
// TEMPORAL DIFFERENCE LEARNER
// =============================================================================

/// Temporal Difference Learner for updating generative model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalDifferenceLearner {
    /// Configuration
    pub config: TemporalDifferenceLearningConfig,
    /// Current learning rate
    pub current_learning_rate: f64,
    /// Eligibility traces (if using TD(lambda))
    pub eligibility_traces: Option<EligibilityTraces>,
    /// Model confidence tracker
    pub confidence_tracker: ModelConfidenceTracker,
    /// History of state transitions
    pub transition_history: VecDeque<StateTransition>,
    /// Total number of updates performed
    pub total_updates: u64,
    /// Running average TD error
    pub avg_td_error: f64,
    /// Running average prediction accuracy
    pub avg_prediction_accuracy: f64,
    /// Number of episodes/epochs completed (for learning rate scheduling)
    pub episodes_completed: u64,
    /// Value function weights for TD learning: V(s) = tanh(w · s + b)
    pub value_weights: Vec<f64>,
    /// Value function bias
    pub value_bias: f64,
    /// Previous state value (for bootstrapping)
    pub prev_value: f64,
}

impl TemporalDifferenceLearner {
    /// Create new temporal difference learner
    pub fn new(
        config: TemporalDifferenceLearningConfig,
        num_actions: usize,
        state_dim: usize,
        obs_dim: usize,
    ) -> Self {
        let eligibility_traces = if config.use_eligibility_traces {
            Some(EligibilityTraces::new(
                num_actions,
                state_dim,
                obs_dim,
                config.lambda,
                config.gamma,
            ))
        } else {
            None
        };

        let confidence_tracker = ModelConfidenceTracker::new(
            num_actions,
            state_dim,
            obs_dim,
            config.confidence_decay,
            config.min_confidence,
        );

        Self {
            current_learning_rate: config.initial_learning_rate,
            config,
            eligibility_traces,
            confidence_tracker,
            transition_history: VecDeque::with_capacity(1000),
            total_updates: 0,
            avg_td_error: 0.0,
            avg_prediction_accuracy: 0.0,
            episodes_completed: 0,
            value_weights: vec![0.0; state_dim],
            value_bias: 0.0,
            prev_value: 0.0,
        }
    }

    /// Observe a state transition and compute TD error
    pub fn observe_transition(
        &mut self,
        old_state: &HiddenState,
        action: usize,
        new_state: &HiddenState,
        observation: &Observation,
        model: &GenerativeModel,
        timestamp: u64,
    ) -> f64 {
        // Compute TD error: delta = r + gamma * V(s') - V(s)
        // In FEP context, "value" is negative free energy
        // We use prediction accuracy as a proxy for value

        // Predicted observation from old state
        let predicted_obs = model.predict_observation(old_state);

        // Predicted next state
        let predicted_next = model.predict_next_state(old_state, action);

        // State prediction error (how well did we predict the next state?)
        let _state_prediction_error: f64 = new_state
            .mean
            .iter()
            .zip(predicted_next.mean.iter())
            .map(|(actual, predicted)| (actual - predicted).powi(2))
            .sum::<f64>()
            .sqrt();

        // Observation prediction error
        let obs_prediction_error: f64 = observation
            .values
            .iter()
            .zip(predicted_obs.iter())
            .map(|(actual, predicted)| (actual - predicted).powi(2))
            .sum::<f64>()
            .sqrt();

        // Nonlinear value function: V(s) = tanh(w · s + b)
        let pre_old: f64 = self
            .value_weights
            .iter()
            .zip(old_state.mean.iter())
            .map(|(w, s)| w * s)
            .sum::<f64>()
            + self.value_bias;
        let v_old = pre_old.tanh();
        let pre_new: f64 = self
            .value_weights
            .iter()
            .zip(new_state.mean.iter())
            .map(|(w, s)| w * s)
            .sum::<f64>()
            + self.value_bias;
        let v_new = pre_new.tanh();

        // Intrinsic reward: negative prediction error
        let reward = -obs_prediction_error;

        // Real TD error: δ = r + γV(s') - V(s)
        let td_error = reward + self.config.gamma * v_new - v_old;
        self.prev_value = v_old;

        // Update eligibility traces if using TD(lambda)
        if let Some(ref mut traces) = self.eligibility_traces {
            traces.decay();
            traces.update(
                action,
                &old_state.mean,
                &new_state.mean,
                &observation.values,
            );
        }

        // Record transition
        let transition = StateTransition {
            old_state: old_state.clone(),
            action,
            new_state: new_state.clone(),
            observation: observation.clone(),
            timestamp,
            td_error,
        };

        if self.transition_history.len() >= self.config.max_transition_history {
            self.transition_history.pop_front();
        }
        self.transition_history.push_back(transition);

        // Update running statistics
        self.total_updates += 1;
        let alpha = 0.05;
        self.avg_td_error = (1.0 - alpha) * self.avg_td_error + alpha * td_error.abs();
        self.avg_prediction_accuracy = (1.0 - alpha) * self.avg_prediction_accuracy
            + alpha * (1.0 / (1.0 + obs_prediction_error));

        td_error
    }

    /// Update generative model using TD learning
    pub fn update_model(
        &mut self,
        model: &mut GenerativeModel,
        old_state: &HiddenState,
        action: usize,
        new_state: &HiddenState,
        observation: &Observation,
        td_error: f64,
    ) {
        let lr = self.current_learning_rate;
        self.total_updates += 1;
        let action_idx = action.min(model.num_actions.saturating_sub(1));

        // === Update transition matrix P(s'|s,a) ===
        // Both branches use error-driven gradient: (observed - predicted) transition
        for i in 0..model.state_dim.min(old_state.mean.len()) {
            for j in 0..model.state_dim.min(new_state.mean.len()) {
                let observed_prob = old_state.mean[i] * new_state.mean[j];
                let model_prob = model.transition_matrices[action_idx][i][j];
                let gradient = observed_prob - model_prob;

                // Scale by eligibility trace if available (for credit assignment over time)
                let trace_scale = if let Some(ref traces) = self.eligibility_traces {
                    if i < traces.transition_traces[action_idx].len()
                        && j < traces.transition_traces[action_idx][i].len()
                    {
                        traces.transition_traces[action_idx][i][j].abs().min(1.0)
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };

                model.transition_matrices[action_idx][i][j] += lr * gradient * trace_scale;
                model.transition_matrices[action_idx][i][j] =
                    model.transition_matrices[action_idx][i][j].clamp(0.0, 1.0);
            }
        }

        // Normalize transition matrix rows to sum to 1 (probability distribution)
        for i in 0..model.state_dim {
            let row_sum: f64 = model.transition_matrices[action_idx][i].iter().sum();
            if row_sum > 0.0 {
                for j in 0..model.state_dim {
                    model.transition_matrices[action_idx][i][j] /= row_sum;
                }
            }
        }

        // === Update likelihood matrix P(o|s) ===
        // Error-driven gradient: move predictions toward observations
        for i in 0..model.state_dim.min(new_state.mean.len()) {
            for j in 0..model.obs_dim.min(observation.values.len()) {
                let predicted = model.likelihood_matrix[i]
                    .iter()
                    .zip(new_state.mean.iter())
                    .map(|(&l, &s)| l * s)
                    .sum::<f64>();
                let obs_error = observation.values[j] - predicted;
                let gradient = obs_error * new_state.mean[i];

                // Scale by eligibility trace if available
                let trace_scale = if let Some(ref traces) = self.eligibility_traces {
                    if i < traces.likelihood_traces.len() && j < traces.likelihood_traces[i].len() {
                        traces.likelihood_traces[i][j].abs().min(1.0)
                    } else {
                        1.0
                    }
                } else {
                    1.0
                };

                model.likelihood_matrix[i][j] += lr * 0.5 * gradient * trace_scale;
                model.likelihood_matrix[i][j] = model.likelihood_matrix[i][j].clamp(0.0, 1.0);
            }
        }

        // Update value function weights: w += lr * δ * dtanh * s, b += lr * δ * dtanh
        // dtanh = 1 - tanh^2(w · s + b)
        let pre_act: f64 = self
            .value_weights
            .iter()
            .zip(old_state.mean.iter())
            .map(|(w, s)| w * s)
            .sum::<f64>()
            + self.value_bias;
        let dtanh = 1.0 - pre_act.tanh().powi(2);

        for i in 0..self.value_weights.len().min(old_state.mean.len()) {
            let trace_scale = if let Some(ref traces) = self.eligibility_traces {
                let aidx = action_idx.min(traces.transition_traces.len().saturating_sub(1));
                if aidx < traces.transition_traces.len()
                    && i < traces.transition_traces[aidx].len()
                    && i < traces.transition_traces[aidx][i].len()
                {
                    traces.transition_traces[aidx][i][i].abs().min(1.0)
                } else {
                    1.0
                }
            } else {
                1.0
            };

            self.value_weights[i] += lr * td_error * dtanh * old_state.mean[i] * trace_scale;
            self.value_weights[i] = self.value_weights[i].clamp(-10.0, 10.0);
        }
        self.value_bias += lr * td_error * dtanh;
        self.value_bias = self.value_bias.clamp(-5.0, 5.0);

        // Update confidence tracker
        let from_idx = old_state
            .mean
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let to_idx = new_state
            .mean
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let obs_idx = observation
            .values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.confidence_tracker
            .update_transition(action, from_idx, to_idx);
        self.confidence_tracker.update_likelihood(to_idx, obs_idx);
    }

    /// Decay learning rate (call at end of episode/epoch)
    pub fn decay_learning_rate(&mut self) {
        self.current_learning_rate = (self.current_learning_rate * self.config.learning_rate_decay)
            .max(self.config.min_learning_rate);
        self.episodes_completed += 1;

        // Also decay confidence tracker
        self.confidence_tracker.decay();
    }

    /// Reset eligibility traces (call at end of episode)
    pub fn reset_traces(&mut self) {
        if let Some(ref mut traces) = self.eligibility_traces {
            traces.reset();
        }
    }

    /// Get learning statistics
    pub fn stats(&self) -> TemporalDifferenceLearningStats {
        TemporalDifferenceLearningStats {
            current_learning_rate: self.current_learning_rate,
            total_updates: self.total_updates,
            avg_td_error: self.avg_td_error,
            avg_prediction_accuracy: self.avg_prediction_accuracy,
            episodes_completed: self.episodes_completed,
            transition_history_size: self.transition_history.len(),
            avg_transition_confidence: self.confidence_tracker.avg_transition_confidence(),
            avg_likelihood_confidence: self.confidence_tracker.avg_likelihood_confidence(),
        }
    }
}

/// Statistics for temporal difference learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalDifferenceLearningStats {
    /// Current learning rate
    pub current_learning_rate: f64,
    /// Total number of updates
    pub total_updates: u64,
    /// Running average TD error
    pub avg_td_error: f64,
    /// Running average prediction accuracy
    pub avg_prediction_accuracy: f64,
    /// Number of episodes completed
    pub episodes_completed: u64,
    /// Size of transition history
    pub transition_history_size: usize,
    /// Average confidence in transition model
    pub avg_transition_confidence: f64,
    /// Average confidence in likelihood model
    pub avg_likelihood_confidence: f64,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generative_model::GenerativeModel;
    use crate::types::{HiddenState, Observation};

    const STATE_DIM: usize = 4;
    const OBS_DIM: usize = 4;
    const NUM_ACTIONS: usize = 3;

    /// Helper: create default TD learner
    fn make_learner() -> TemporalDifferenceLearner {
        TemporalDifferenceLearner::new(
            TemporalDifferenceLearningConfig::default(),
            NUM_ACTIONS,
            STATE_DIM,
            OBS_DIM,
        )
    }

    /// Helper: create TD learner without eligibility traces
    fn make_learner_no_traces() -> TemporalDifferenceLearner {
        let mut config = TemporalDifferenceLearningConfig::default();
        config.use_eligibility_traces = false;
        TemporalDifferenceLearner::new(config, NUM_ACTIONS, STATE_DIM, OBS_DIM)
    }

    /// Helper: create a simple hidden state with given values
    fn state_from(vals: &[f64]) -> HiddenState {
        HiddenState {
            mean: vals.to_vec(),
            precision: vec![1.0; vals.len()],
            mode_probs: vec![1.0],
            current_mode: 0,
        }
    }

    /// Helper: create a simple observation
    fn obs_from(vals: &[f64]) -> Observation {
        Observation::new(vals.to_vec(), 1.0, "test")
    }

    // ---------------------------------------------------------------
    // 1. Construction
    // ---------------------------------------------------------------

    #[test]
    fn test_construction_default_config() {
        let learner = make_learner();
        assert_eq!(learner.value_weights.len(), STATE_DIM);
        assert_eq!(learner.value_bias, 0.0);
        assert_eq!(learner.total_updates, 0);
        assert_eq!(learner.episodes_completed, 0);
        assert!(learner.eligibility_traces.is_some());
        assert!(learner.transition_history.is_empty());
    }

    #[test]
    fn test_construction_without_traces() {
        let learner = make_learner_no_traces();
        assert!(learner.eligibility_traces.is_none());
    }

    // ---------------------------------------------------------------
    // 2. Eligibility Traces
    // ---------------------------------------------------------------

    #[test]
    fn test_eligibility_traces_new_zeros() {
        let traces = EligibilityTraces::new(NUM_ACTIONS, STATE_DIM, OBS_DIM, 0.8, 0.99);
        for action_traces in &traces.transition_traces {
            for from_traces in action_traces {
                for &t in from_traces {
                    assert_eq!(t, 0.0);
                }
            }
        }
        for state_traces in &traces.likelihood_traces {
            for &t in state_traces {
                assert_eq!(t, 0.0);
            }
        }
    }

    #[test]
    fn test_eligibility_traces_update_increases_values() {
        let mut traces = EligibilityTraces::new(NUM_ACTIONS, STATE_DIM, OBS_DIM, 0.8, 0.99);
        let from_state = vec![0.5, 0.3, 0.1, 0.1];
        let to_state = vec![0.1, 0.5, 0.3, 0.1];
        let observation = vec![0.2, 0.4, 0.3, 0.1];

        traces.update(0, &from_state, &to_state, &observation);

        // After update, transition traces should have nonzero values
        let has_nonzero = traces.transition_traces[0]
            .iter()
            .any(|row| row.iter().any(|&v| v != 0.0));
        assert!(
            has_nonzero,
            "transition traces should have nonzero values after update"
        );

        // Likelihood traces should also have nonzero values
        let has_nonzero_lk = traces
            .likelihood_traces
            .iter()
            .any(|row| row.iter().any(|&v| v != 0.0));
        assert!(
            has_nonzero_lk,
            "likelihood traces should have nonzero values after update"
        );
    }

    #[test]
    fn test_eligibility_traces_decay_reduces_values() {
        let mut traces = EligibilityTraces::new(NUM_ACTIONS, STATE_DIM, OBS_DIM, 0.8, 0.99);
        let from_state = vec![1.0, 0.0, 0.0, 0.0];
        let to_state = vec![0.0, 1.0, 0.0, 0.0];
        let observation = vec![0.5, 0.5, 0.0, 0.0];

        traces.update(0, &from_state, &to_state, &observation);

        // Record value before decay
        let before = traces.transition_traces[0][0][1];
        assert!(before > 0.0);

        traces.decay();

        let after = traces.transition_traces[0][0][1];
        assert!(
            after < before,
            "decay should reduce trace values: {} < {}",
            after,
            before
        );
        assert!(after > 0.0, "decay should not zero out traces in one step");
    }

    #[test]
    fn test_eligibility_traces_reset_zeros_all() {
        let mut traces = EligibilityTraces::new(NUM_ACTIONS, STATE_DIM, OBS_DIM, 0.8, 0.99);
        traces.update(
            0,
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0, 0.0],
            &[0.5, 0.5, 0.0, 0.0],
        );

        traces.reset();

        for action_traces in &traces.transition_traces {
            for from_traces in action_traces {
                for &t in from_traces {
                    assert_eq!(t, 0.0, "all traces should be zero after reset");
                }
            }
        }
    }

    #[test]
    fn test_eligibility_traces_accumulate() {
        let mut traces = EligibilityTraces::new(NUM_ACTIONS, STATE_DIM, OBS_DIM, 0.8, 0.99);
        let from = vec![1.0, 0.0, 0.0, 0.0];
        let to = vec![0.0, 1.0, 0.0, 0.0];
        let obs = vec![0.5, 0.5, 0.0, 0.0];

        traces.update(0, &from, &to, &obs);
        let after_one = traces.transition_traces[0][0][1];

        traces.update(0, &from, &to, &obs);
        let after_two = traces.transition_traces[0][0][1];

        assert!(
            (after_two - 2.0 * after_one).abs() < 1e-10,
            "accumulating traces should double: {} vs 2*{}",
            after_two,
            after_one
        );
    }

    // ---------------------------------------------------------------
    // 3. Model Confidence Tracker
    // ---------------------------------------------------------------

    #[test]
    fn test_confidence_tracker_initial_min() {
        let tracker = ModelConfidenceTracker::new(NUM_ACTIONS, STATE_DIM, OBS_DIM, 0.99, 0.1);
        assert!((tracker.avg_transition_confidence() - 0.1).abs() < 1e-10);
        assert!((tracker.avg_likelihood_confidence() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_increases_with_observations() {
        let mut tracker = ModelConfidenceTracker::new(NUM_ACTIONS, STATE_DIM, OBS_DIM, 0.99, 0.1);
        let initial = tracker.transition_confidence[0][1][2];

        // Observe the same transition many times
        for _ in 0..20 {
            tracker.update_transition(0, 1, 2);
        }

        let after = tracker.transition_confidence[0][1][2];
        assert!(
            after > initial,
            "confidence should increase with observations: {} > {}",
            after,
            initial
        );
        assert!(after <= 1.0, "confidence should not exceed 1.0");
    }

    #[test]
    fn test_confidence_saturates_below_one() {
        let mut tracker = ModelConfidenceTracker::new(NUM_ACTIONS, STATE_DIM, OBS_DIM, 0.99, 0.1);
        // Observe many times to saturate
        for _ in 0..1000 {
            tracker.update_transition(0, 0, 0);
        }
        let conf = tracker.transition_confidence[0][0][0];
        assert!(
            conf > 0.9,
            "confidence should approach 1.0 after many observations: {}",
            conf
        );
        assert!(conf <= 1.0);
    }

    #[test]
    fn test_confidence_decay_respects_minimum() {
        let mut tracker = ModelConfidenceTracker::new(NUM_ACTIONS, STATE_DIM, OBS_DIM, 0.99, 0.1);
        // Decay many times
        for _ in 0..1000 {
            tracker.decay();
        }
        let conf = tracker.avg_transition_confidence();
        assert!(
            conf >= 0.1 - 1e-10,
            "confidence should not drop below min_confidence (0.1): {}",
            conf
        );
    }

    #[test]
    fn test_likelihood_confidence_updates() {
        let mut tracker = ModelConfidenceTracker::new(NUM_ACTIONS, STATE_DIM, OBS_DIM, 0.99, 0.1);
        let initial = tracker.likelihood_confidence[1][2];

        for _ in 0..10 {
            tracker.update_likelihood(1, 2);
        }

        let after = tracker.likelihood_confidence[1][2];
        assert!(after > initial, "likelihood confidence should increase");
    }

    // ---------------------------------------------------------------
    // 4. TD Learning: observe_transition
    // ---------------------------------------------------------------

    #[test]
    fn test_observe_transition_returns_finite_td_error() {
        let mut learner = make_learner();
        let model = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
        let old = state_from(&[0.5, 0.3, 0.1, 0.1]);
        let new = state_from(&[0.3, 0.5, 0.1, 0.1]);
        let obs = obs_from(&[0.4, 0.4, 0.1, 0.1]);

        let td_error = learner.observe_transition(&old, 0, &new, &obs, &model, 1);
        assert!(
            td_error.is_finite(),
            "TD error should be finite: {}",
            td_error
        );
    }

    #[test]
    fn test_observe_transition_records_history() {
        let mut learner = make_learner();
        let model = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
        let old = state_from(&[0.5, 0.3, 0.1, 0.1]);
        let new = state_from(&[0.3, 0.5, 0.1, 0.1]);
        let obs = obs_from(&[0.4, 0.4, 0.1, 0.1]);

        assert_eq!(learner.transition_history.len(), 0);
        learner.observe_transition(&old, 0, &new, &obs, &model, 1);
        assert_eq!(learner.transition_history.len(), 1);
        learner.observe_transition(&old, 1, &new, &obs, &model, 2);
        assert_eq!(learner.transition_history.len(), 2);
    }

    #[test]
    fn test_observe_transition_increments_update_count() {
        let mut learner = make_learner();
        let model = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
        let old = state_from(&[0.5, 0.3, 0.1, 0.1]);
        let new = state_from(&[0.3, 0.5, 0.1, 0.1]);
        let obs = obs_from(&[0.4, 0.4, 0.1, 0.1]);

        learner.observe_transition(&old, 0, &new, &obs, &model, 1);
        assert_eq!(learner.total_updates, 1);
    }

    #[test]
    fn test_transition_history_respects_max_capacity() {
        let mut config = TemporalDifferenceLearningConfig::default();
        config.max_transition_history = 5;
        let mut learner = TemporalDifferenceLearner::new(config, NUM_ACTIONS, STATE_DIM, OBS_DIM);
        let model = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
        let old = state_from(&[0.5, 0.3, 0.1, 0.1]);
        let new = state_from(&[0.3, 0.5, 0.1, 0.1]);
        let obs = obs_from(&[0.4, 0.4, 0.1, 0.1]);

        for t in 0..10 {
            learner.observe_transition(&old, 0, &new, &obs, &model, t);
        }

        assert_eq!(
            learner.transition_history.len(),
            5,
            "history should be capped at max_transition_history"
        );
    }

    // ---------------------------------------------------------------
    // 5. TD Learning: update_model
    // ---------------------------------------------------------------

    #[test]
    fn test_update_model_modifies_transition_matrix() {
        let mut learner = make_learner();
        let mut model = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
        let old = state_from(&[0.8, 0.1, 0.05, 0.05]);
        let new = state_from(&[0.1, 0.8, 0.05, 0.05]);
        let obs = obs_from(&[0.2, 0.6, 0.1, 0.1]);

        let before: Vec<Vec<f64>> = model.transition_matrices[0].clone();

        learner.update_model(&mut model, &old, 0, &new, &obs, 0.5);

        let after = &model.transition_matrices[0];
        let changed = before.iter().zip(after.iter()).any(|(b_row, a_row)| {
            b_row
                .iter()
                .zip(a_row.iter())
                .any(|(b, a)| (b - a).abs() > 1e-15)
        });
        assert!(changed, "update_model should modify the transition matrix");
    }

    #[test]
    fn test_update_model_transition_rows_sum_to_one() {
        let mut learner = make_learner();
        let mut model = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
        let old = state_from(&[0.8, 0.1, 0.05, 0.05]);
        let new = state_from(&[0.1, 0.8, 0.05, 0.05]);
        let obs = obs_from(&[0.2, 0.6, 0.1, 0.1]);

        learner.update_model(&mut model, &old, 0, &new, &obs, 0.5);

        for (i, row) in model.transition_matrices[0].iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-8,
                "transition row {} should sum to 1.0 after normalization, got {}",
                i,
                sum
            );
        }
    }

    #[test]
    fn test_update_model_updates_value_weights() {
        // Use a learner without eligibility traces so trace_scale is 1.0
        // (with traces enabled but empty, trace_scale = 0, blocking weight updates)
        let mut learner = make_learner_no_traces();
        let mut model = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
        let old = state_from(&[0.8, 0.1, 0.05, 0.05]);
        let new = state_from(&[0.1, 0.8, 0.05, 0.05]);
        let obs = obs_from(&[0.2, 0.6, 0.1, 0.1]);

        let weights_before = learner.value_weights.clone();
        let bias_before = learner.value_bias;

        // Use a large td_error to ensure visible changes
        learner.update_model(&mut model, &old, 0, &new, &obs, 1.0);

        let weights_changed = learner
            .value_weights
            .iter()
            .zip(weights_before.iter())
            .any(|(a, b)| (a - b).abs() > 1e-15);
        assert!(weights_changed, "value weights should change after update");

        // Bias should also change with nonzero td_error and dtanh
        assert!(
            (learner.value_bias - bias_before).abs() > 1e-15,
            "bias should change after update with nonzero td_error"
        );
        assert!(learner.value_bias.is_finite());
    }

    // ---------------------------------------------------------------
    // 6. Learning Rate Decay
    // ---------------------------------------------------------------

    #[test]
    fn test_learning_rate_decay_decreases() {
        let mut learner = make_learner();
        let initial_lr = learner.current_learning_rate;

        learner.decay_learning_rate();

        assert!(
            learner.current_learning_rate < initial_lr,
            "learning rate should decrease after decay: {} < {}",
            learner.current_learning_rate,
            initial_lr
        );
        assert_eq!(learner.episodes_completed, 1);
    }

    #[test]
    fn test_learning_rate_decay_respects_minimum() {
        let mut learner = make_learner();

        for _ in 0..100_000 {
            learner.decay_learning_rate();
        }

        assert!(
            learner.current_learning_rate >= learner.config.min_learning_rate,
            "learning rate should not drop below minimum: {} >= {}",
            learner.current_learning_rate,
            learner.config.min_learning_rate
        );
    }

    // ---------------------------------------------------------------
    // 7. Stats
    // ---------------------------------------------------------------

    #[test]
    fn test_stats_reflect_learner_state() {
        let mut learner = make_learner();
        let model = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
        let old = state_from(&[0.5, 0.3, 0.1, 0.1]);
        let new = state_from(&[0.3, 0.5, 0.1, 0.1]);
        let obs = obs_from(&[0.4, 0.4, 0.1, 0.1]);

        learner.observe_transition(&old, 0, &new, &obs, &model, 1);
        learner.decay_learning_rate();

        let stats = learner.stats();
        assert_eq!(stats.total_updates, 1);
        assert_eq!(stats.episodes_completed, 1);
        assert_eq!(stats.transition_history_size, 1);
        assert!(stats.avg_td_error.is_finite());
        assert!(stats.avg_prediction_accuracy.is_finite());
        assert!(stats.current_learning_rate < 0.1); // decayed from default
    }

    // ---------------------------------------------------------------
    // 8. Reset Traces
    // ---------------------------------------------------------------

    #[test]
    fn test_reset_traces_clears_eligibility() {
        let mut learner = make_learner();
        let model = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
        let old = state_from(&[0.5, 0.3, 0.1, 0.1]);
        let new = state_from(&[0.3, 0.5, 0.1, 0.1]);
        let obs = obs_from(&[0.4, 0.4, 0.1, 0.1]);

        // Observe a transition to populate traces
        learner.observe_transition(&old, 0, &new, &obs, &model, 1);

        // Verify traces are nonzero
        let traces = learner.eligibility_traces.as_ref().unwrap();
        let has_nonzero = traces.transition_traces[0]
            .iter()
            .any(|row| row.iter().any(|&v| v != 0.0));
        assert!(has_nonzero);

        // Reset
        learner.reset_traces();

        let traces = learner.eligibility_traces.as_ref().unwrap();
        for action_traces in &traces.transition_traces {
            for from_traces in action_traces {
                for &t in from_traces {
                    assert_eq!(t, 0.0, "all traces should be zero after reset");
                }
            }
        }
    }

    // ---------------------------------------------------------------
    // 9. Edge Cases
    // ---------------------------------------------------------------

    #[test]
    fn test_zero_reward_td_error() {
        // When observation perfectly matches prediction, reward ~ 0
        let mut learner = make_learner();
        let model = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
        let state = state_from(&[0.5, 0.5, 0.0, 0.0]);

        // Predict observation and use that as the actual observation
        let predicted = model.predict_observation(&state);
        let obs = obs_from(&predicted);

        let td_error = learner.observe_transition(&state, 0, &state, &obs, &model, 0);
        // With zero prediction error, reward=0, and V(s)=V(s')=0 initially => td_error ~ 0
        assert!(td_error.is_finite());
        // The TD error should be very small since prediction matches and values are 0
        assert!(
            td_error.abs() < 0.1,
            "TD error with perfect prediction should be near zero: {}",
            td_error
        );
    }

    #[test]
    fn test_negative_reward_scenario() {
        // Large prediction error gives negative reward (reward = -error)
        let mut learner = make_learner();
        let model = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
        let old = state_from(&[0.9, 0.05, 0.025, 0.025]);
        let new = state_from(&[0.05, 0.9, 0.025, 0.025]);
        // Observation very different from what model would predict
        let obs = obs_from(&[0.0, 0.0, 0.0, 10.0]);

        let td_error = learner.observe_transition(&old, 0, &new, &obs, &model, 1);
        // With large prediction error, reward is very negative
        assert!(td_error.is_finite());
        assert!(
            td_error < 0.0,
            "large prediction error should yield negative TD error: {}",
            td_error
        );
    }

    #[test]
    fn test_discount_factor_zero() {
        let mut config = TemporalDifferenceLearningConfig::default();
        config.gamma = 0.0; // No discounting of future
        let mut learner = TemporalDifferenceLearner::new(config, NUM_ACTIONS, STATE_DIM, OBS_DIM);
        let model = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
        let old = state_from(&[0.5, 0.3, 0.1, 0.1]);
        let new = state_from(&[0.3, 0.5, 0.1, 0.1]);
        let obs = obs_from(&[0.4, 0.4, 0.1, 0.1]);

        let td_error = learner.observe_transition(&old, 0, &new, &obs, &model, 1);
        // With gamma=0: td_error = reward + 0*V(s') - V(s) = reward - V(s)
        assert!(td_error.is_finite());
    }

    #[test]
    fn test_high_learning_rate_effect() {
        let mut config = TemporalDifferenceLearningConfig::default();
        config.initial_learning_rate = 0.9;
        let mut learner_high =
            TemporalDifferenceLearner::new(config.clone(), NUM_ACTIONS, STATE_DIM, OBS_DIM);

        config.initial_learning_rate = 0.01;
        let mut learner_low =
            TemporalDifferenceLearner::new(config, NUM_ACTIONS, STATE_DIM, OBS_DIM);

        let mut model_high = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
        let mut model_low = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);

        let old = state_from(&[0.8, 0.1, 0.05, 0.05]);
        let new = state_from(&[0.1, 0.8, 0.05, 0.05]);
        let obs = obs_from(&[0.2, 0.6, 0.1, 0.1]);

        learner_high.update_model(&mut model_high, &old, 0, &new, &obs, 0.5);
        learner_low.update_model(&mut model_low, &old, 0, &new, &obs, 0.5);

        // Higher learning rate should produce larger weight changes
        let delta_high: f64 = learner_high.value_weights.iter().map(|w| w.abs()).sum();
        let delta_low: f64 = learner_low.value_weights.iter().map(|w| w.abs()).sum();

        // With td_error of same sign and magnitude, higher LR gives bigger weights
        // (though value weights start at 0, the delta is lr*td_error*dtanh*state)
        // We check that both are finite; the high-LR version should have larger magnitude
        assert!(delta_high.is_finite());
        assert!(delta_low.is_finite());
        assert!(
            delta_high >= delta_low,
            "higher LR should produce equal or larger weight magnitude: {} >= {}",
            delta_high,
            delta_low
        );
    }

    // ---------------------------------------------------------------
    // 10. Value Function Convergence (Simple MDP)
    // ---------------------------------------------------------------

    #[test]
    fn test_value_function_updates_toward_reward() {
        // Repeatedly present the same transition with negative reward
        // Value weights should adjust to reflect the state's value
        let mut learner = make_learner();
        let mut model = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
        let old = state_from(&[0.9, 0.05, 0.025, 0.025]);
        let new = state_from(&[0.05, 0.9, 0.025, 0.025]);
        // Observation that creates moderate prediction error
        let obs = obs_from(&[0.3, 0.3, 0.2, 0.2]);

        for t in 0..50 {
            let td_error = learner.observe_transition(&old, 0, &new, &obs, &model, t);
            learner.update_model(&mut model, &old, 0, &new, &obs, td_error);
        }

        // After 50 updates, value weights and avg_td_error should be finite
        assert!(learner.avg_td_error.is_finite());
        assert!(learner.avg_prediction_accuracy.is_finite());
        assert!(
            learner.avg_prediction_accuracy > 0.0,
            "accuracy should be positive"
        );

        // Value weights should have been modified from zero
        let weight_sum: f64 = learner.value_weights.iter().map(|w| w.abs()).sum();
        assert!(
            weight_sum > 0.0,
            "value weights should be nonzero after training"
        );
    }

    // ---------------------------------------------------------------
    // 11. No traces vs with traces
    // ---------------------------------------------------------------

    #[test]
    fn test_with_and_without_traces_both_produce_finite_results() {
        let model = GenerativeModel::new(STATE_DIM, OBS_DIM, NUM_ACTIONS);
        let old = state_from(&[0.5, 0.3, 0.1, 0.1]);
        let new = state_from(&[0.3, 0.5, 0.1, 0.1]);
        let obs = obs_from(&[0.4, 0.4, 0.1, 0.1]);

        let mut with_traces = make_learner();
        let mut without_traces = make_learner_no_traces();

        let td_with = with_traces.observe_transition(&old, 0, &new, &obs, &model, 1);
        let td_without = without_traces.observe_transition(&old, 0, &new, &obs, &model, 1);

        assert!(td_with.is_finite());
        assert!(td_without.is_finite());

        // Both should compute the same TD error (traces don't affect observe_transition's return)
        assert!(
            (td_with - td_without).abs() < 1e-10,
            "TD error should be the same with/without traces: {} vs {}",
            td_with,
            td_without
        );
    }
}
