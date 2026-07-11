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

    /// Optional additive bias per `[action][state_dim]`, added on top of the
    /// transition matrices' multiplicative sum in `predict_next_state`.
    /// Defaults to all zeros — **no behavior change for any existing caller**.
    ///
    /// Without this, `predict_next_state` can never predict a value larger
    /// than a weighted average of the *current* state (a bias-free linear
    /// map) — fine for representing decay toward zero, but unable to
    /// represent genuine self-reinforcing growth. Found and root-caused via
    /// `symthaea-culinary`'s active-inference palate
    /// (`CULINARY_PLAN_2026-07-09.md` Phase 3/4): a kitchen's φ rises under
    /// sustained heat regardless of its current value, which no seeding of
    /// `transition_matrices` alone could represent. Set via
    /// [`GenerativeModel::set_transition_bias`] — the transition-model
    /// analogue of `ActiveInferenceAgent::inject_priors`'s "Passport Route"
    /// for belief priors. Deliberately **not** touched by `learn()` or TD
    /// learning in this version — an externally-set structural prior, same
    /// scope as `inject_priors`, not a claim that bias learning is implemented.
    #[serde(default)]
    pub transition_bias: Vec<Vec<f64>>,

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
            transition_bias: vec![vec![0.0; state_dim]; num_actions],
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
            if let Some(bias) = self.transition_bias.get(action_idx)
                && let Some(b) = bias.get(i)
            {
                next_mean[i] += b;
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

    /// Set the additive transition bias for `action` — the transition-model
    /// analogue of [`GenerativeModel::inject_priors`]'s Passport Route, letting
    /// `predict_next_state` represent genuine growth/decay that a bias-free
    /// multiplicative transition matrix cannot (see `transition_bias`'s doc for
    /// why this was needed). No-op if `action` is out of range or `bias`'s
    /// length doesn't match `state_dim`.
    pub fn set_transition_bias(&mut self, action: usize, bias: Vec<f64>) {
        if bias.len() == self.state_dim
            && let Some(slot) = self.transition_bias.get_mut(action)
        {
            *slot = bias;
        }
    }
}
