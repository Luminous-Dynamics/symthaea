// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generative model: P(o, s) = P(o|s) * P(s)

use serde::{Deserialize, Serialize};

use super::types::{HiddenState, Observation};

/// Generative model: P(o, s) = P(o|s) * P(s)
///
/// The generative model defines how hidden states generate observations
/// and how states transition over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerativeModel {
    /// Observation model: likelihood P(o|s)
    /// Maps hidden state dimension → observation dimension
    pub likelihood_matrix: Vec<Vec<f64>>,

    /// State transition model: P(s'|s, a)
    /// Maps (state, action) → next state distribution
    pub transition_matrices: Vec<Vec<Vec<f64>>>,

    /// Prior over initial states P(s_0)
    pub prior_mean: Vec<f64>,
    pub prior_precision: Vec<f64>,

    /// Observation noise precision
    pub observation_precision: f64,

    /// Transition noise precision
    pub transition_precision: f64,

    /// Hidden state dimension
    pub state_dim: usize,

    /// Observation dimension
    pub obs_dim: usize,

    /// Number of actions
    pub num_actions: usize,

    /// Learning rate for model parameters
    pub learning_rate: f64,
}

impl GenerativeModel {
    /// Create new generative model
    pub fn new(state_dim: usize, obs_dim: usize, num_actions: usize) -> Self {
        // Initialize likelihood matrix (near-diagonal with some spread)
        let mut likelihood_matrix = vec![vec![0.0; obs_dim]; state_dim];
        for i in 0..state_dim {
            for j in 0..obs_dim {
                if i == j && i < obs_dim {
                    likelihood_matrix[i][j] = 0.7;
                } else if (i as isize - j as isize).abs() == 1 {
                    likelihood_matrix[i][j] = 0.15;
                } else {
                    likelihood_matrix[i][j] = 0.05 / (state_dim.max(obs_dim) - 2).max(1) as f64;
                }
            }
        }

        // Initialize transition matrices (one per action)
        let mut transition_matrices = Vec::with_capacity(num_actions);
        for action_idx in 0..num_actions {
            let mut transition = vec![vec![0.0; state_dim]; state_dim];
            for i in 0..state_dim {
                // Self-transition
                transition[i][i] = 0.7;
                // Action-dependent bias
                let bias_direction = if action_idx % 2 == 0 { -1 } else { 1 };
                let next_i = ((i as isize + bias_direction).max(0) as usize).min(state_dim - 1);
                transition[i][next_i] += 0.2;
                // Small transitions to other states
                for j in 0..state_dim {
                    if j != i && j != next_i {
                        transition[i][j] = 0.1 / (state_dim - 2).max(1) as f64;
                    }
                }
            }
            transition_matrices.push(transition);
        }

        Self {
            likelihood_matrix,
            transition_matrices,
            prior_mean: vec![0.5; state_dim],
            prior_precision: vec![1.0; state_dim],
            observation_precision: 5.0,
            transition_precision: 10.0,
            state_dim,
            obs_dim,
            num_actions,
            learning_rate: 0.01,
        }
    }

    /// Predict observation given hidden state: E[o|s]
    pub fn predict_observation(&self, state: &HiddenState) -> Vec<f64> {
        let mut predicted = vec![0.0; self.obs_dim];
        for i in 0..self.state_dim.min(state.mean.len()) {
            for j in 0..self.obs_dim {
                predicted[j] += self.likelihood_matrix[i][j] * state.mean[i];
            }
        }
        predicted
    }

    /// Predict next state given current state and action: E[s'|s, a]
    pub fn predict_next_state(&self, state: &HiddenState, action: usize) -> HiddenState {
        let action_idx = action.min(self.num_actions - 1);
        let transition = &self.transition_matrices[action_idx];

        let mut next_mean = vec![0.0; self.state_dim];
        for i in 0..self.state_dim {
            for j in 0..self.state_dim.min(state.mean.len()) {
                next_mean[i] += transition[j][i] * state.mean[j];
            }
        }

        // Precision decreases due to transition uncertainty
        let next_precision: Vec<f64> = state
            .precision
            .iter()
            .map(|p| (p * self.transition_precision) / (p + self.transition_precision))
            .collect();

        HiddenState {
            mean: next_mean,
            precision: next_precision,
            mode_probs: state.mode_probs.clone(),
            current_mode: state.current_mode,
        }
    }

    /// Compute prediction error (observation - predicted)
    pub fn prediction_error(&self, state: &HiddenState, observation: &Observation) -> Vec<f64> {
        let predicted = self.predict_observation(state);
        predicted
            .iter()
            .zip(observation.values.iter())
            .map(|(pred, obs)| obs - pred)
            .collect()
    }

    /// Update model parameters based on prediction error (Hebbian-like learning)
    ///
    /// Note: For full temporal difference learning with next state observations,
    /// use `TemporalDifferenceLearner::update_model()` instead.
    pub fn learn(&mut self, state: &HiddenState, observation: &Observation, action: Option<usize>) {
        let predicted = self.predict_observation(state);

        // Update likelihood matrix
        for i in 0..self.state_dim {
            for j in 0..self.obs_dim.min(observation.values.len()) {
                let error = observation.values[j] - predicted.get(j).copied().unwrap_or(0.0);
                let gradient = error * state.mean.get(i).copied().unwrap_or(0.0);
                self.likelihood_matrix[i][j] += self.learning_rate * gradient;
                // Keep bounded
                self.likelihood_matrix[i][j] = self.likelihood_matrix[i][j].max(0.0).min(1.0);
            }
        }

        // Update transition matrix if action was taken
        // Note: This is a simplified update without next-state information.
        // For proper TD learning, use TemporalDifferenceLearner::update_model()
        // which takes both old_state and new_state.
        if let Some(action_idx) = action {
            let idx = action_idx.min(self.num_actions - 1);
            // Self-reinforcement: strengthen transitions that maintain state
            for i in 0..self.state_dim.min(state.mean.len()) {
                // Increase self-transition probability slightly for active states
                let state_activity = state.mean[i];
                if state_activity > 0.5 {
                    self.transition_matrices[idx][i][i] +=
                        self.learning_rate * 0.1 * (state_activity - 0.5);
                    self.transition_matrices[idx][i][i] =
                        self.transition_matrices[idx][i][i].clamp(0.0, 1.0);
                }
            }

            // Normalize rows
            for i in 0..self.state_dim {
                let row_sum: f64 = self.transition_matrices[idx][i].iter().sum();
                if row_sum > 0.0 {
                    for j in 0..self.state_dim {
                        self.transition_matrices[idx][i][j] /= row_sum;
                    }
                }
            }
        }
    }

    /// Learn transition dynamics from observed state transition
    ///
    /// This is the full temporal difference learning update that requires
    /// both the previous and next state observations.
    pub fn learn_transition(
        &mut self,
        old_state: &HiddenState,
        action: usize,
        new_state: &HiddenState,
        observation: &Observation,
    ) {
        let action_idx = action.min(self.num_actions.saturating_sub(1));

        // Compute prediction error for the transition
        let predicted_next = self.predict_next_state(old_state, action);
        let transition_error: f64 = new_state
            .mean
            .iter()
            .zip(predicted_next.mean.iter())
            .map(|(actual, predicted)| (actual - predicted).powi(2))
            .sum::<f64>()
            .sqrt();

        // Learning rate modulated by prediction error (larger errors = more learning)
        let effective_lr = self.learning_rate * (1.0 + transition_error).min(2.0);

        // Update transition matrix P(s'|s,a)
        for i in 0..self.state_dim.min(old_state.mean.len()) {
            for j in 0..self.state_dim.min(new_state.mean.len()) {
                // The observed transition probability
                let observed_prob = old_state.mean[i] * new_state.mean[j];
                // Current model probability
                let model_prob = self.transition_matrices[action_idx][i][j];
                // Gradient: move toward observed transition
                let gradient = observed_prob - model_prob;

                self.transition_matrices[action_idx][i][j] += effective_lr * gradient;
                self.transition_matrices[action_idx][i][j] =
                    self.transition_matrices[action_idx][i][j].clamp(0.0, 1.0);
            }
        }

        // Normalize transition matrix rows
        for i in 0..self.state_dim {
            let row_sum: f64 = self.transition_matrices[action_idx][i].iter().sum();
            if row_sum > 0.0 {
                for j in 0..self.state_dim {
                    self.transition_matrices[action_idx][i][j] /= row_sum;
                }
            }
        }

        // Update likelihood matrix P(o|s) based on new state and observation
        let predicted_obs = self.predict_observation(new_state);
        for i in 0..self.state_dim.min(new_state.mean.len()) {
            for j in 0..self.obs_dim.min(observation.values.len()) {
                let obs_error =
                    observation.values[j] - predicted_obs.get(j).copied().unwrap_or(0.0);
                let gradient = obs_error * new_state.mean[i];
                self.likelihood_matrix[i][j] += effective_lr * gradient;
                self.likelihood_matrix[i][j] = self.likelihood_matrix[i][j].clamp(0.0, 1.0);
            }
        }
    }

    /// Inject explicit priors into the generative model (Passport Route).
    pub fn inject_priors(&mut self, mean: Vec<f64>, precision: Vec<f64>) {
        if mean.len() == self.state_dim {
            self.prior_mean = mean;
        }
        if precision.len() == self.state_dim {
            self.prior_precision = precision;
        }
    }
}
