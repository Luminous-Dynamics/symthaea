// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Free energy computation, precision estimation, and expected free energy for action selection.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use super::generative_model::GenerativeModel;
use super::types::{
    ExpectedFreeEnergyResult, FreeEnergyComponents, HiddenState, Observation, PrecisionSnapshot,
};

// =============================================================================
// FREE ENERGY CALCULATOR
// =============================================================================

/// Computes variational free energy and its components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeEnergyCalculator {
    /// History of free energy values
    pub history: VecDeque<f64>,
    /// Maximum history size
    pub max_history: usize,
    /// Running average free energy
    pub running_average: f64,
}

impl FreeEnergyCalculator {
    pub fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history),
            max_history,
            running_average: 0.0,
        }
    }

    /// Compute variational free energy: F = Accuracy - Complexity
    ///
    /// F = E_q[ln q(s)] - E_q[ln p(o,s)]
    ///   = E_q[ln q(s)] - E_q[ln p(o|s)] - E_q[ln p(s)]
    ///   = Complexity - Accuracy
    ///
    /// Returns (total_F, accuracy_term, complexity_term)
    pub fn compute(
        &mut self,
        state: &HiddenState,
        observation: &Observation,
        model: &GenerativeModel,
    ) -> FreeEnergyComponents {
        // Accuracy: Expected log likelihood E_q[ln p(o|s)]
        // Approximated as -0.5 * precision * prediction_error^2
        let prediction = model.predict_observation(state);
        let accuracy = self.compute_accuracy(&prediction, observation, model.observation_precision);

        // Complexity: KL divergence D_KL[q(s) || p(s)]
        let complexity = self.compute_complexity(state, &model.prior_mean, &model.prior_precision);

        // Free energy F = Complexity - Accuracy
        let total = complexity - accuracy;

        // Update history
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(total);

        // Update running average
        let alpha = 0.1;
        self.running_average = (1.0 - alpha) * self.running_average + alpha * total;

        FreeEnergyComponents {
            total,
            accuracy,
            complexity,
            surprise: -accuracy, // Surprise ≈ -log p(o) ≈ -accuracy
            prediction_error: self.compute_prediction_error_magnitude(&prediction, observation),
        }
    }

    /// Compute accuracy term: E_q[ln p(o|s)] ≈ -0.5 * π * ε^2
    fn compute_accuracy(
        &self,
        prediction: &[f64],
        observation: &Observation,
        precision: f64,
    ) -> f64 {
        let mut sum_sq_error = 0.0;
        for (pred, obs) in prediction.iter().zip(observation.values.iter()) {
            sum_sq_error += (pred - obs).powi(2);
        }
        -0.5 * precision * sum_sq_error
    }

    /// Compute complexity term: D_KL[q(s) || p(s)]
    fn compute_complexity(
        &self,
        state: &HiddenState,
        prior_mean: &[f64],
        prior_precision: &[f64],
    ) -> f64 {
        let mut kl = 0.0;
        for i in 0..state.mean.len().min(prior_mean.len()) {
            let var_q = 1.0 / state.precision[i].max(0.001);
            let var_p = 1.0 / prior_precision.get(i).copied().unwrap_or(1.0).max(0.001);
            let mean_diff = state.mean[i] - prior_mean.get(i).copied().unwrap_or(0.5);

            // KL for univariate Gaussians
            kl += 0.5 * (var_q / var_p + mean_diff.powi(2) / var_p - 1.0 + (var_p / var_q).ln());
        }
        kl.max(0.0)
    }

    /// Compute magnitude of prediction error
    fn compute_prediction_error_magnitude(
        &self,
        prediction: &[f64],
        observation: &Observation,
    ) -> f64 {
        let mut sum_sq = 0.0;
        for (pred, obs) in prediction.iter().zip(observation.values.iter()) {
            sum_sq += (pred - obs).powi(2);
        }
        sum_sq.sqrt()
    }

    /// Get surprise trend (positive = increasing surprise)
    pub fn surprise_trend(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.0;
        }

        let recent: Vec<f64> = self.history.iter().rev().take(10).cloned().collect();
        let older: Vec<f64> = self
            .history
            .iter()
            .rev()
            .skip(10)
            .take(10)
            .cloned()
            .collect();

        if recent.is_empty() || older.is_empty() {
            return 0.0;
        }

        let recent_avg: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let older_avg: f64 = older.iter().sum::<f64>() / older.len() as f64;

        recent_avg - older_avg
    }
}

// =============================================================================
// LEVIN MORPHOSPACE REMAPPING CONTROLLER
// =============================================================================

/// Tracks persistent unresolvable surprise to trigger representational remapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevinRemappingController {
    pub persistent_error_ticks: usize,
    pub stall_threshold_ticks: usize,
    pub remapping_trigger_count: u64,
}

impl LevinRemappingController {
    pub fn new(stall_threshold_ticks: usize) -> Self {
        Self {
            persistent_error_ticks: 0,
            stall_threshold_ticks,
            remapping_trigger_count: 0,
        }
    }

    /// Evaluates if high-dimensional variational free energy spikes require
    /// an algebraic rotation/remapping of the underlying embedding space.
    pub fn evaluate_remapping_necessity(&mut self, components: &FreeEnergyComponents, running_avg: f64) -> bool {
        if components.surprise > (running_avg * 2.5).max(5.0) {
            self.persistent_error_ticks += 1;
        } else {
            self.persistent_error_ticks = self.persistent_error_ticks.saturating_sub(1);
        }

        if self.persistent_error_ticks >= self.stall_threshold_ticks {
            self.persistent_error_ticks = 0;
            self.remapping_trigger_count += 1;
            true
        } else {
            false
        }
    }
}

// =============================================================================
// PRECISION ESTIMATOR
// =============================================================================

/// Dynamic precision estimation for confidence-weighted prediction errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionEstimator {
    /// Sensory precision (confidence in observations)
    pub sensory_precision: f64,
    /// Prior precision (confidence in predictions)
    pub prior_precision: f64,
    /// State precision (confidence in current belief)
    pub state_precision: f64,
    /// Action precision (confidence in action outcomes)
    pub action_precision: f64,
    /// Precision learning rate
    pub learning_rate: f64,
    /// History of precision estimates
    pub history: VecDeque<PrecisionSnapshot>,
    /// Maximum history size
    pub max_history: usize,
}

impl PrecisionEstimator {
    pub fn new() -> Self {
        Self {
            sensory_precision: 1.0,
            prior_precision: 1.0,
            state_precision: 1.0,
            action_precision: 1.0,
            learning_rate: 0.05,
            history: VecDeque::with_capacity(100),
            max_history: 100,
        }
    }

    /// Update precision based on prediction error
    pub fn update_from_error(&mut self, prediction_error: f64, timestamp: u64) {
        // Precision inversely related to prediction error variance
        // High error → reduce precision, low error → increase precision
        let error_factor = (1.0 + prediction_error.abs()).recip();

        // Update sensory precision (how much to trust observations)
        if prediction_error.abs() > 0.5 {
            // High error: increase sensory precision (trust observations more)
            self.sensory_precision =
                (self.sensory_precision * (1.0 + self.learning_rate * error_factor)).min(5.0);
            // Decrease prior precision (trust predictions less)
            self.prior_precision =
                (self.prior_precision * (1.0 - self.learning_rate * 0.5)).max(0.1);
        } else {
            // Low error: can rely more on predictions
            self.prior_precision =
                (self.prior_precision * (1.0 + self.learning_rate * error_factor)).min(5.0);
            // Slight decrease in sensory precision (predictions are good)
            self.sensory_precision =
                (self.sensory_precision * (1.0 - self.learning_rate * 0.1)).max(0.5);
        }

        // State precision based on running average
        self.state_precision = (self.sensory_precision + self.prior_precision) / 2.0;

        // Record snapshot
        self.record_snapshot(timestamp);
    }

    /// Update precision based on action outcome
    pub fn update_from_action(
        &mut self,
        expected_outcome: f64,
        actual_outcome: f64,
        timestamp: u64,
    ) {
        let action_error = (expected_outcome - actual_outcome).abs();
        let error_factor = (1.0 + action_error).recip();

        // Update action precision
        self.action_precision = 0.9 * self.action_precision + 0.1 * error_factor * 2.0;
        self.action_precision = self.action_precision.max(0.1).min(5.0);

        self.record_snapshot(timestamp);
    }

    /// Record precision snapshot
    fn record_snapshot(&mut self, timestamp: u64) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(PrecisionSnapshot {
            sensory: self.sensory_precision,
            prior: self.prior_precision,
            state: self.state_precision,
            action: self.action_precision,
            timestamp,
        });
    }

    /// Get effective precision for perception
    pub fn perceptual_precision(&self) -> f64 {
        (self.sensory_precision + self.prior_precision) / 2.0
    }

    /// Get precision-weighted prediction error
    pub fn weight_error(&self, error: f64) -> f64 {
        error * self.sensory_precision
    }

    /// Get precision stability (low variance = high stability)
    pub fn stability(&self) -> f64 {
        if self.history.len() < 2 {
            return 0.5;
        }

        let precisions: Vec<f64> = self.history.iter().map(|s| s.state).collect();
        let mean = precisions.iter().sum::<f64>() / precisions.len() as f64;
        let variance =
            precisions.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / precisions.len() as f64;

        // High variance = low stability
        1.0 - (variance.sqrt() / 2.0).min(1.0)
    }
}

impl Default for PrecisionEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// EXPECTED FREE ENERGY (for action selection)
// =============================================================================

/// Expected Free Energy for action selection
///
/// G = E_q[ln q(s|π) - ln p(o,s|π)]
///   = Epistemic value + Pragmatic value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedFreeEnergyComputer {
    /// Weight for pragmatic (goal-directed) value
    pub pragmatic_weight: f64,
    /// Weight for epistemic (uncertainty-reducing) value
    pub epistemic_weight: f64,
    /// Weight for novelty (exploration) bonus
    pub novelty_weight: f64,
    /// Preferred observations (goals)
    pub preferences: Vec<f64>,
    /// Preference precision (how strongly to pursue goals)
    pub preference_precision: f64,
    /// Action history for novelty computation
    pub action_history: VecDeque<usize>,
}

impl ExpectedFreeEnergyComputer {
    pub fn new(obs_dim: usize) -> Self {
        Self {
            pragmatic_weight: 1.0,
            epistemic_weight: 0.5,
            novelty_weight: 0.1,
            preferences: vec![0.8; obs_dim], // Default: prefer high values
            preference_precision: 2.0,
            action_history: VecDeque::with_capacity(100),
        }
    }

    /// Set goal preferences
    pub fn set_preferences(&mut self, preferences: Vec<f64>, precision: f64) {
        self.preferences = preferences;
        self.preference_precision = precision;
    }

    /// Compute expected free energy for an action
    ///
    /// G(π) = Pragmatic + Epistemic
    ///      = E_q[D_KL[q(o|s) || p̃(o)]] + E_q[H[p(o|s)]]
    pub fn compute(
        &mut self,
        action: usize,
        state: &HiddenState,
        model: &GenerativeModel,
    ) -> ExpectedFreeEnergyResult {
        // Predict next state under this action
        let predicted_state = model.predict_next_state(state, action);

        // Predict expected observation
        let expected_obs = model.predict_observation(&predicted_state);

        // Pragmatic value: How close to preferences?
        // Lower distance = better (negate for minimization)
        let pragmatic = self.compute_pragmatic_value(&expected_obs);

        // Epistemic value: How much uncertainty reduction?
        // Higher uncertainty reduction = better (negate for minimization)
        let epistemic = self.compute_epistemic_value(&predicted_state, state);

        // Novelty: How often have we taken this action?
        let novelty = self.compute_novelty(action);

        // Record action in history
        self.action_history.push_back(action);
        if self.action_history.len() > 100 {
            self.action_history.pop_front();
        }

        // Total expected free energy (lower is better)
        let total = self.pragmatic_weight * pragmatic + self.epistemic_weight * epistemic
            - self.novelty_weight * novelty; // Novelty encourages exploration

        ExpectedFreeEnergyResult {
            action,
            total,
            pragmatic,
            epistemic,
            novelty,
            predicted_state,
            expected_observation: expected_obs,
        }
    }

    /// Compute pragmatic value (divergence from preferences)
    fn compute_pragmatic_value(&self, expected_obs: &[f64]) -> f64 {
        let mut divergence = 0.0;
        for (obs, pref) in expected_obs.iter().zip(self.preferences.iter()) {
            divergence += self.preference_precision * (obs - pref).powi(2);
        }
        divergence
    }

    /// Compute epistemic value (uncertainty about outcomes)
    fn compute_epistemic_value(
        &self,
        predicted_state: &HiddenState,
        current_state: &HiddenState,
    ) -> f64 {
        // Epistemic value = reduction in uncertainty
        let current_entropy = current_state.entropy();
        let predicted_entropy = predicted_state.entropy();

        // If entropy decreases, that's good (negative contribution to G)
        predicted_entropy - current_entropy
    }

    /// Compute novelty bonus for action
    fn compute_novelty(&self, action: usize) -> f64 {
        let count = self.action_history.iter().filter(|a| **a == action).count();
        1.0 / (1.0 + count as f64)
    }
}
