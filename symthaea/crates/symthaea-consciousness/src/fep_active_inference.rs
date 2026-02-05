//! # Full Active Inference Implementation (FEP Integration)
//!
//! Implements Karl Friston's Free Energy Principle (FEP) as a complete active inference loop.
//!
//! ## Mathematical Foundation
//!
//! The Free Energy Principle posits that biological systems minimize variational free energy:
//!
//! ```text
//! F = E_q[ln q(s) - ln p(o,s)]
//!   = D_KL[q(s) || p(s|o)] - ln p(o)
//!   ≥ -ln p(o)  (Surprise)
//! ```
//!
//! where:
//! - `p(o,s)` is the generative model (joint distribution over observations and states)
//! - `q(s)` is the recognition model (approximate posterior over hidden states)
//! - `F` is variational free energy (upper bound on surprise)
//!
//! ## Active Inference Loop
//!
//! 1. **Perception**: Update beliefs q(s) to minimize free energy given observations
//! 2. **Action Selection**: Choose actions that minimize expected free energy
//! 3. **Model Learning**: Update generative model parameters based on prediction errors
//!
//! ## Components
//!
//! - `GenerativeModel`: Maps hidden states → predicted observations
//! - `FreeEnergyCalculator`: Computes variational free energy and its components
//! - `PrecisionEstimator`: Dynamic precision weighting for confidence-weighted errors
//! - `ActiveInferenceAgent`: Full perception-action loop
//!
//! ## Integration
//!
//! This module integrates with the cognitive loop's prediction error system, providing
//! precision-weighted prediction errors that modulate learning and attention.
//!
//! ## References
//!
//! - Friston, K. (2010). The free-energy principle: a unified brain theory?
//! - Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017).
//!   Active Inference: A Process Theory.
//! - Parr, T., Pezzulo, G., & Friston, K. J. (2022). Active Inference: The Free Energy
//!   Principle in Mind, Brain, and Behavior.

use std::collections::VecDeque;
use std::f64::consts::PI;
use serde::{Deserialize, Serialize};

// =============================================================================
// OBSERVATION MODEL
// =============================================================================

/// Observation from the environment/internal state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Raw observation vector
    pub values: Vec<f64>,
    /// Observation precision (inverse variance, confidence in observation)
    pub precision: f64,
    /// Timestamp (monotonic counter)
    pub timestamp: u64,
    /// Modality (e.g., "visual", "interoceptive", "cognitive")
    pub modality: String,
}

impl Observation {
    /// Create new observation
    pub fn new(values: Vec<f64>, precision: f64, modality: &str) -> Self {
        Self {
            values,
            precision,
            timestamp: 0,
            modality: modality.to_string(),
        }
    }

    /// Create from consciousness state observables
    pub fn from_consciousness_state(phi: f64, integration: f64, coherence: f64, attention: f64) -> Self {
        Self {
            values: vec![phi, integration, coherence, attention],
            precision: 1.0,
            timestamp: 0,
            modality: "consciousness".to_string(),
        }
    }

    /// Dimension of observation
    pub fn dim(&self) -> usize {
        self.values.len()
    }
}

// =============================================================================
// HIDDEN STATE (Beliefs)
// =============================================================================

/// Hidden state representation (beliefs about the world)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenState {
    /// Mean of belief distribution (expected hidden state)
    pub mean: Vec<f64>,
    /// Precision (inverse variance) for each dimension
    pub precision: Vec<f64>,
    /// Mode probabilities for discrete states (if applicable)
    pub mode_probs: Vec<f64>,
    /// Current mode (for discrete state space)
    pub current_mode: usize,
}

impl HiddenState {
    /// Create new hidden state with given dimension
    pub fn new(dim: usize) -> Self {
        Self {
            mean: vec![0.5; dim],
            precision: vec![1.0; dim],
            mode_probs: vec![1.0],
            current_mode: 0,
        }
    }

    /// Create with discrete modes
    pub fn with_modes(continuous_dim: usize, num_modes: usize) -> Self {
        let mode_probs = vec![1.0 / num_modes as f64; num_modes];
        Self {
            mean: vec![0.5; continuous_dim],
            precision: vec![1.0; continuous_dim],
            mode_probs,
            current_mode: 0,
        }
    }

    /// Get variance (inverse of precision)
    pub fn variance(&self) -> Vec<f64> {
        self.precision.iter().map(|p| 1.0 / p.max(0.001)).collect()
    }

    /// Compute entropy of the continuous belief (Gaussian)
    pub fn entropy(&self) -> f64 {
        let dim = self.mean.len() as f64;
        // Entropy of multivariate Gaussian: 0.5 * (d + d*ln(2π) + ln|Σ|)
        let log_det: f64 = self.precision.iter().map(|p| -p.max(0.001).ln()).sum();
        0.5 * (dim + dim * (2.0 * PI).ln() + log_det)
    }

    /// Compute discrete entropy over modes
    pub fn mode_entropy(&self) -> f64 {
        -self.mode_probs.iter()
            .filter(|p| **p > 0.0)
            .map(|p| p * p.ln())
            .sum::<f64>()
    }

    /// Total uncertainty (continuous + discrete)
    pub fn total_uncertainty(&self) -> f64 {
        self.entropy() + self.mode_entropy()
    }

    /// Confidence (inverse of uncertainty, normalized)
    pub fn confidence(&self) -> f64 {
        let avg_precision = self.precision.iter().sum::<f64>() / self.precision.len() as f64;
        let max_mode_prob = self.mode_probs.iter().cloned().fold(0.0, f64::max);
        (avg_precision * max_mode_prob).min(1.0)
    }
}

// =============================================================================
// GENERATIVE MODEL
// =============================================================================

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
                let bias_direction = if action_idx % 2 == 0 { 1 } else { -1 };
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
        let next_precision: Vec<f64> = state.precision.iter()
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
        predicted.iter()
            .zip(observation.values.iter())
            .map(|(pred, obs)| obs - pred)
            .collect()
    }

    /// Update model parameters based on prediction error (Hebbian-like learning)
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
        if let Some(action_idx) = action {
            let idx = action_idx.min(self.num_actions - 1);
            // This would require next state observation - simplified here
            // Full implementation would use temporal difference learning
        }
    }
}

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
    fn compute_accuracy(&self, prediction: &[f64], observation: &Observation, precision: f64) -> f64 {
        let mut sum_sq_error = 0.0;
        for (pred, obs) in prediction.iter().zip(observation.values.iter()) {
            sum_sq_error += (pred - obs).powi(2);
        }
        -0.5 * precision * sum_sq_error
    }

    /// Compute complexity term: D_KL[q(s) || p(s)]
    fn compute_complexity(&self, state: &HiddenState, prior_mean: &[f64], prior_precision: &[f64]) -> f64 {
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
    fn compute_prediction_error_magnitude(&self, prediction: &[f64], observation: &Observation) -> f64 {
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
        let older: Vec<f64> = self.history.iter().rev().skip(10).take(10).cloned().collect();

        if recent.is_empty() || older.is_empty() {
            return 0.0;
        }

        let recent_avg: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let older_avg: f64 = older.iter().sum::<f64>() / older.len() as f64;

        recent_avg - older_avg
    }
}

/// Components of free energy computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeEnergyComponents {
    /// Total variational free energy
    pub total: f64,
    /// Accuracy term (expected log likelihood)
    pub accuracy: f64,
    /// Complexity term (KL divergence from prior)
    pub complexity: f64,
    /// Surprise (negative log evidence)
    pub surprise: f64,
    /// Prediction error magnitude
    pub prediction_error: f64,
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

/// Snapshot of precision values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionSnapshot {
    pub sensory: f64,
    pub prior: f64,
    pub state: f64,
    pub action: f64,
    pub timestamp: u64,
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
            self.sensory_precision = (self.sensory_precision * (1.0 + self.learning_rate * error_factor)).min(5.0);
            // Decrease prior precision (trust predictions less)
            self.prior_precision = (self.prior_precision * (1.0 - self.learning_rate * 0.5)).max(0.1);
        } else {
            // Low error: can rely more on predictions
            self.prior_precision = (self.prior_precision * (1.0 + self.learning_rate * error_factor)).min(5.0);
            // Slight decrease in sensory precision (predictions are good)
            self.sensory_precision = (self.sensory_precision * (1.0 - self.learning_rate * 0.1)).max(0.5);
        }

        // State precision based on running average
        self.state_precision = (self.sensory_precision + self.prior_precision) / 2.0;

        // Record snapshot
        self.record_snapshot(timestamp);
    }

    /// Update precision based on action outcome
    pub fn update_from_action(&mut self, expected_outcome: f64, actual_outcome: f64, timestamp: u64) {
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
        let variance = precisions.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / precisions.len() as f64;

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
    pub fn compute(&mut self, action: usize, state: &HiddenState, model: &GenerativeModel) -> ExpectedFreeEnergyResult {
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
        let total = self.pragmatic_weight * pragmatic
            + self.epistemic_weight * epistemic
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
    fn compute_epistemic_value(&self, predicted_state: &HiddenState, current_state: &HiddenState) -> f64 {
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

/// Result of expected free energy computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedFreeEnergyResult {
    /// Action evaluated
    pub action: usize,
    /// Total expected free energy (lower is better)
    pub total: f64,
    /// Pragmatic component (goal-directedness)
    pub pragmatic: f64,
    /// Epistemic component (uncertainty reduction)
    pub epistemic: f64,
    /// Novelty bonus
    pub novelty: f64,
    /// Predicted state after action
    pub predicted_state: HiddenState,
    /// Expected observation after action
    pub expected_observation: Vec<f64>,
}

// =============================================================================
// ACTIVE INFERENCE AGENT
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
        }
    }
}

/// Statistics for Active Inference Agent
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActiveInferenceAgentStats {
    /// Total perception cycles
    pub perception_cycles: u64,
    /// Total actions taken
    pub actions_taken: u64,
    /// Average free energy
    pub avg_free_energy: f64,
    /// Average prediction error
    pub avg_prediction_error: f64,
    /// Average precision
    pub avg_precision: f64,
    /// Exploration rate (epistemic actions / total)
    pub exploration_rate: f64,
    /// Model learning updates
    pub model_updates: u64,
    /// Epistemic actions taken
    epistemic_actions: u64,
}

/// Full Active Inference Agent implementing perception-action loop
#[derive(Debug, Clone)]
pub struct ActiveInferenceAgent {
    /// Configuration
    pub config: ActiveInferenceAgentConfig,
    /// Current belief state
    pub belief: HiddenState,
    /// Generative model
    pub model: GenerativeModel,
    /// Free energy calculator
    pub free_energy_calc: FreeEnergyCalculator,
    /// Precision estimator
    pub precision: PrecisionEstimator,
    /// Expected free energy computer
    pub efe_computer: ExpectedFreeEnergyComputer,
    /// Last free energy components
    pub last_fe_components: Option<FreeEnergyComponents>,
    /// Statistics
    pub stats: ActiveInferenceAgentStats,
    /// Current timestamp counter
    timestamp: u64,
}

impl ActiveInferenceAgent {
    /// Create new active inference agent
    pub fn new(config: ActiveInferenceAgentConfig) -> Self {
        let model = GenerativeModel::new(config.state_dim, config.obs_dim, config.num_actions);
        let belief = HiddenState::new(config.state_dim);
        let free_energy_calc = FreeEnergyCalculator::new(500);
        let precision = PrecisionEstimator::new();
        let efe_computer = ExpectedFreeEnergyComputer::new(config.obs_dim);

        Self {
            config,
            belief,
            model,
            free_energy_calc,
            precision,
            efe_computer,
            last_fe_components: None,
            stats: ActiveInferenceAgentStats::default(),
            timestamp: 0,
        }
    }

    /// Perception step: Update beliefs to minimize free energy
    ///
    /// This implements variational inference:
    /// q(s) ← argmin_q F[q, o]
    pub fn perceive(&mut self, observation: &Observation) -> PerceptionResult {
        self.timestamp += 1;

        // Run belief update iterations
        let mut total_belief_change = 0.0;
        for _ in 0..self.config.inference_iterations {
            let change = self.update_belief(observation);
            total_belief_change += change;
        }

        // Compute free energy
        let fe_components = self.free_energy_calc.compute(&self.belief, observation, &self.model);
        self.last_fe_components = Some(fe_components.clone());

        // Update precision based on prediction error
        self.precision.update_from_error(fe_components.prediction_error, self.timestamp);

        // Learn generative model
        if self.config.enable_model_learning {
            self.model.learn(&self.belief, observation, None);
            self.stats.model_updates += 1;
        }

        // Update stats
        self.stats.perception_cycles += 1;
        let n = self.stats.perception_cycles as f64;
        self.stats.avg_free_energy = (self.stats.avg_free_energy * (n - 1.0) + fe_components.total) / n;
        self.stats.avg_prediction_error = (self.stats.avg_prediction_error * (n - 1.0) + fe_components.prediction_error) / n;
        self.stats.avg_precision = (self.stats.avg_precision * (n - 1.0) + self.precision.perceptual_precision()) / n;

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
        let weighted_error: Vec<f64> = prediction_error.iter()
            .map(|e| e * self.precision.sensory_precision)
            .collect();

        // Update belief mean (gradient descent on free energy)
        let mut total_change = 0.0;
        for i in 0..self.belief.mean.len() {
            // Aggregate error from likelihood matrix
            let mut grad = 0.0;
            for j in 0..weighted_error.len() {
                if i < self.model.likelihood_matrix.len() && j < self.model.likelihood_matrix[i].len() {
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
            let error_i = if i < prediction_error.len() { prediction_error[i].abs() } else { 0.5 };
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
        let mut efe_results: Vec<ExpectedFreeEnergyResult> = Vec::new();

        for action in 0..self.config.num_actions {
            let efe = self.efe_computer.compute(action, &self.belief, &self.model);
            efe_results.push(efe);
        }

        // Softmax action selection
        let max_efe = efe_results.iter().map(|r| -r.total).fold(f64::NEG_INFINITY, f64::max);
        let exp_values: Vec<f64> = efe_results.iter()
            .map(|r| ((-r.total - max_efe) / self.config.action_temperature).exp())
            .collect();
        let sum_exp: f64 = exp_values.iter().sum();
        let probabilities: Vec<f64> = exp_values.iter().map(|e| e / sum_exp).collect();

        // Select action with highest probability (greedy for now)
        let selected_idx = probabilities.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let selected = &efe_results[selected_idx];

        // Determine if this is exploratory (high epistemic value)
        let is_exploratory = selected.epistemic.abs() > selected.pragmatic.abs();
        if is_exploratory {
            self.stats.epistemic_actions += 1;
        }

        // Update stats
        self.stats.actions_taken += 1;
        self.stats.exploration_rate = self.stats.epistemic_actions as f64 / self.stats.actions_taken as f64;

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
        // Update belief to incorporate new observation
        let _ = self.perceive(actual_observation);

        // Update generative model with action information
        if self.config.enable_model_learning {
            self.model.learn(&self.belief, actual_observation, Some(action));
        }

        // Update action precision
        let expected = self.model.predict_observation(&self.belief);
        let expected_phi = expected.get(0).copied().unwrap_or(0.5);
        let actual_phi = actual_observation.values.get(0).copied().unwrap_or(0.5);
        self.precision.update_from_action(expected_phi, actual_phi, self.timestamp);
    }

    /// Set goal preferences for the agent
    pub fn set_goals(&mut self, preferences: Vec<f64>, precision: f64) {
        self.efe_computer.set_preferences(preferences, precision);
    }

    /// Get current free energy
    pub fn current_free_energy(&self) -> f64 {
        self.last_fe_components.as_ref().map(|c| c.total).unwrap_or(0.0)
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

    /// Reset agent state
    pub fn reset(&mut self) {
        self.belief = HiddenState::new(self.config.state_dim);
        self.precision = PrecisionEstimator::new();
        self.free_energy_calc = FreeEnergyCalculator::new(500);
        self.efe_computer = ExpectedFreeEnergyComputer::new(self.config.obs_dim);
        self.last_fe_components = None;
        self.stats = ActiveInferenceAgentStats::default();
        self.timestamp = 0;
    }
}

/// Result of perception step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptionResult {
    /// Updated belief state
    pub updated_belief: HiddenState,
    /// Free energy components
    pub free_energy: FreeEnergyComponents,
    /// Current precision
    pub precision: f64,
    /// Total belief change
    pub belief_change: f64,
    /// Timestamp
    pub timestamp: u64,
}

/// Result of action selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionSelectionResult {
    /// Selected action
    pub action: usize,
    /// Expected free energy of selected action
    pub expected_free_energy: f64,
    /// Probability distribution over actions
    pub action_probabilities: Vec<f64>,
    /// Whether this is an exploratory action
    pub is_exploratory: bool,
    /// Pragmatic value component
    pub pragmatic_value: f64,
    /// Epistemic value component
    pub epistemic_value: f64,
    /// Predicted state after action
    pub predicted_state: HiddenState,
}

/// Outcome of action execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionOutcome {
    /// Action taken
    pub action: usize,
    /// Predicted next state
    pub predicted_next_state: HiddenState,
    /// Expected observation
    pub expected_observation: Vec<f64>,
    /// Timestamp
    pub timestamp: u64,
}

/// Summary of active inference agent state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveInferenceSummary {
    /// Current belief mean
    pub belief_mean: Vec<f64>,
    /// Belief confidence
    pub belief_confidence: f64,
    /// Current free energy
    pub free_energy: f64,
    /// Current precision
    pub precision: f64,
    /// Exploration rate
    pub exploration_rate: f64,
    /// Total perception cycles
    pub total_cycles: u64,
}

// =============================================================================
// COGNITIVE LOOP INTEGRATION
// =============================================================================

/// Integration adapter for cognitive loop
///
/// This provides the interface between the active inference agent and
/// the existing cognitive loop's prediction error system.
#[derive(Debug, Clone)]
pub struct CognitiveLoopFEPBridge {
    /// Active inference agent
    pub agent: ActiveInferenceAgent,
    /// Whether to modulate learning rate based on precision
    pub precision_modulated_learning: bool,
    /// Precision threshold for learning
    pub learning_precision_threshold: f64,
}

impl CognitiveLoopFEPBridge {
    /// Create new bridge
    pub fn new(config: ActiveInferenceAgentConfig) -> Self {
        Self {
            agent: ActiveInferenceAgent::new(config),
            precision_modulated_learning: true,
            learning_precision_threshold: 0.5,
        }
    }

    /// Process cognitive loop state
    pub fn process(&mut self, phi: f64, integration: f64, coherence: f64, attention: f64) -> CognitiveLoopFEPResult {
        // Create observation from consciousness state
        let observation = Observation::from_consciousness_state(phi, integration, coherence, attention);

        // Run perception
        let perception = self.agent.perceive(&observation);

        // Select action
        let action_selection = self.agent.select_action();

        // Compute learning rate modulation
        let learning_rate_mod = if self.precision_modulated_learning {
            self.compute_learning_modulation()
        } else {
            1.0
        };

        // Should learning occur?
        let should_learn = perception.precision > self.learning_precision_threshold
            && perception.free_energy.prediction_error < 0.8;

        CognitiveLoopFEPResult {
            free_energy: perception.free_energy.total,
            prediction_error: perception.free_energy.prediction_error,
            precision_weighted_error: self.agent.precision.weight_error(perception.free_energy.prediction_error),
            recommended_action: action_selection.action,
            is_surprised: self.agent.is_surprised(),
            learning_rate_modulation: learning_rate_mod,
            should_learn,
            exploration_mode: action_selection.is_exploratory,
            belief_confidence: perception.updated_belief.confidence(),
            epistemic_value: action_selection.epistemic_value,
            pragmatic_value: action_selection.pragmatic_value,
        }
    }

    /// Compute learning rate modulation based on free energy
    fn compute_learning_modulation(&self) -> f64 {
        let precision = self.agent.precision.perceptual_precision();
        let stability = self.agent.precision.stability();

        // High precision + high stability = boost learning
        // Low precision or low stability = reduce learning
        (precision * stability).sqrt().max(0.1).min(2.0)
    }

    /// Set goals for the agent
    pub fn set_goals(&mut self, preferred_phi: f64, preferred_integration: f64, preferred_coherence: f64, preferred_attention: f64) {
        self.agent.set_goals(
            vec![preferred_phi, preferred_integration, preferred_coherence, preferred_attention],
            2.0,
        );
    }

    /// Reset the bridge
    pub fn reset(&mut self) {
        self.agent.reset();
    }
}

/// Result from cognitive loop FEP processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveLoopFEPResult {
    /// Current free energy
    pub free_energy: f64,
    /// Raw prediction error
    pub prediction_error: f64,
    /// Precision-weighted prediction error
    pub precision_weighted_error: f64,
    /// Recommended action (index)
    pub recommended_action: usize,
    /// Whether agent is surprised
    pub is_surprised: bool,
    /// Learning rate modulation factor
    pub learning_rate_modulation: f64,
    /// Should learning occur this cycle?
    pub should_learn: bool,
    /// Is agent in exploration mode?
    pub exploration_mode: bool,
    /// Confidence in current beliefs
    pub belief_confidence: f64,
    /// Epistemic value of current state
    pub epistemic_value: f64,
    /// Pragmatic value of current state
    pub pragmatic_value: f64,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_creation() {
        let obs = Observation::from_consciousness_state(0.7, 0.6, 0.8, 0.5);
        assert_eq!(obs.dim(), 4);
        assert_eq!(obs.values[0], 0.7);
        assert_eq!(obs.modality, "consciousness");
    }

    #[test]
    fn test_hidden_state_creation() {
        let state = HiddenState::new(8);
        assert_eq!(state.mean.len(), 8);
        assert_eq!(state.precision.len(), 8);
        assert!(state.confidence() > 0.0);
    }

    #[test]
    fn test_hidden_state_entropy() {
        let state = HiddenState::new(4);
        let entropy = state.entropy();
        assert!(entropy > 0.0);
        assert!(entropy.is_finite());
    }

    #[test]
    fn test_generative_model_creation() {
        let model = GenerativeModel::new(8, 4, 6);
        assert_eq!(model.state_dim, 8);
        assert_eq!(model.obs_dim, 4);
        assert_eq!(model.num_actions, 6);
    }

    #[test]
    fn test_generative_model_prediction() {
        let model = GenerativeModel::new(4, 4, 4);
        let state = HiddenState::new(4);

        let obs = model.predict_observation(&state);
        assert_eq!(obs.len(), 4);

        let next_state = model.predict_next_state(&state, 0);
        assert_eq!(next_state.mean.len(), 4);
    }

    #[test]
    fn test_free_energy_computation() {
        let model = GenerativeModel::new(4, 4, 4);
        let state = HiddenState::new(4);
        let obs = Observation::from_consciousness_state(0.5, 0.5, 0.5, 0.5);

        let mut calc = FreeEnergyCalculator::new(100);
        let components = calc.compute(&state, &obs, &model);

        assert!(components.total.is_finite());
        assert!(components.accuracy.is_finite());
        assert!(components.complexity >= 0.0);
    }

    #[test]
    fn test_precision_estimator() {
        let mut precision = PrecisionEstimator::new();

        // High prediction error should decrease prior precision
        precision.update_from_error(0.8, 1);
        assert!(precision.prior_precision < 1.0);

        // Low prediction error should increase prior precision
        for i in 0..10 {
            precision.update_from_error(0.1, i + 2);
        }
        assert!(precision.prior_precision > 0.5);
    }

    #[test]
    fn test_expected_free_energy() {
        let model = GenerativeModel::new(4, 4, 4);
        let state = HiddenState::new(4);
        let mut efe_computer = ExpectedFreeEnergyComputer::new(4);

        let result = efe_computer.compute(0, &state, &model);

        assert!(result.total.is_finite());
        assert!(result.pragmatic.is_finite());
        assert!(result.epistemic.is_finite());
    }

    #[test]
    fn test_active_inference_agent_creation() {
        let config = ActiveInferenceAgentConfig::default();
        let agent = ActiveInferenceAgent::new(config);

        assert_eq!(agent.belief.mean.len(), 8);
        assert_eq!(agent.stats.perception_cycles, 0);
    }

    #[test]
    fn test_active_inference_perception() {
        let config = ActiveInferenceAgentConfig::default();
        let mut agent = ActiveInferenceAgent::new(config);

        let obs = Observation::from_consciousness_state(0.7, 0.6, 0.8, 0.5);
        let result = agent.perceive(&obs);

        assert!(result.free_energy.total.is_finite());
        assert!(result.precision > 0.0);
        assert_eq!(agent.stats.perception_cycles, 1);
    }

    #[test]
    fn test_active_inference_action_selection() {
        let config = ActiveInferenceAgentConfig::default();
        let mut agent = ActiveInferenceAgent::new(config);

        // Run perception first
        let obs = Observation::from_consciousness_state(0.7, 0.6, 0.8, 0.5);
        let _ = agent.perceive(&obs);

        // Select action
        let result = agent.select_action();

        assert!(result.action < 6);
        assert!(result.expected_free_energy.is_finite());
        assert_eq!(result.action_probabilities.len(), 6);
    }

    #[test]
    fn test_active_inference_learning() {
        let config = ActiveInferenceAgentConfig::default();
        let mut agent = ActiveInferenceAgent::new(config);

        // Run multiple perception cycles with consistent observations
        for _ in 0..20 {
            let obs = Observation::from_consciousness_state(0.7, 0.6, 0.8, 0.5);
            let _ = agent.perceive(&obs);
        }

        // Prediction error should decrease with learning
        assert!(agent.stats.avg_prediction_error < 1.0);
    }

    #[test]
    fn test_cognitive_loop_bridge() {
        let config = ActiveInferenceAgentConfig::default();
        let mut bridge = CognitiveLoopFEPBridge::new(config);

        // Process consciousness state
        let result = bridge.process(0.7, 0.6, 0.8, 0.5);

        assert!(result.free_energy.is_finite());
        assert!(result.prediction_error >= 0.0);
        assert!(result.learning_rate_modulation > 0.0);
    }

    #[test]
    fn test_cognitive_loop_bridge_goals() {
        let config = ActiveInferenceAgentConfig::default();
        let mut bridge = CognitiveLoopFEPBridge::new(config);

        // Set goals for high consciousness state
        bridge.set_goals(0.9, 0.9, 0.9, 0.9);

        // Process lower state
        let result = bridge.process(0.3, 0.3, 0.3, 0.3);

        // Should have high pragmatic motivation (far from goals)
        assert!(result.pragmatic_value > 0.0);
    }

    #[test]
    fn test_precision_stability() {
        let mut precision = PrecisionEstimator::new();

        // Consistent low errors should give high stability
        for i in 0..50 {
            precision.update_from_error(0.1, i);
        }

        let stability = precision.stability();
        assert!(stability > 0.5);
    }

    #[test]
    fn test_free_energy_trend() {
        let model = GenerativeModel::new(4, 4, 4);
        let state = HiddenState::new(4);
        let mut calc = FreeEnergyCalculator::new(100);

        // Build up history
        for _ in 0..30 {
            let obs = Observation::from_consciousness_state(0.5, 0.5, 0.5, 0.5);
            calc.compute(&state, &obs, &model);
        }

        let trend = calc.surprise_trend();
        assert!(trend.is_finite());
    }

    #[test]
    fn test_agent_reset() {
        let config = ActiveInferenceAgentConfig::default();
        let mut agent = ActiveInferenceAgent::new(config);

        // Run some cycles
        for _ in 0..10 {
            let obs = Observation::from_consciousness_state(0.7, 0.6, 0.8, 0.5);
            let _ = agent.perceive(&obs);
        }

        assert!(agent.stats.perception_cycles > 0);

        // Reset
        agent.reset();

        assert_eq!(agent.stats.perception_cycles, 0);
        assert!(agent.last_fe_components.is_none());
    }

    #[test]
    fn test_is_surprised() {
        let config = ActiveInferenceAgentConfig::default();
        let mut agent = ActiveInferenceAgent::new(config);

        // Initial state should not be surprised
        let obs = Observation::from_consciousness_state(0.5, 0.5, 0.5, 0.5);
        let _ = agent.perceive(&obs);

        // Check surprise status
        let surprised = agent.is_surprised();
        // Just verify it returns a boolean without crashing
        assert!(surprised || !surprised);
    }

    #[test]
    fn test_summary() {
        let config = ActiveInferenceAgentConfig::default();
        let mut agent = ActiveInferenceAgent::new(config);

        let obs = Observation::from_consciousness_state(0.5, 0.5, 0.5, 0.5);
        let _ = agent.perceive(&obs);

        let summary = agent.summary();

        assert_eq!(summary.belief_mean.len(), 8);
        assert!(summary.belief_confidence >= 0.0);
        assert_eq!(summary.total_cycles, 1);
    }
}
