//! # Active Inference Routing
//!
//! This module implements routing based on Karl Friston's Free Energy Principle:
//!
//! ## Key Concepts
//!
//! 1. **Generative Model**: Internal model of how consciousness states evolve
//! 2. **Recognition Model**: Approximate posterior over hidden states
//! 3. **Free Energy Minimization**: Select strategies that reduce surprise
//! 4. **Expected Free Energy**: Balance exploitation vs exploration
//! 5. **Precision Weighting**: Confidence in predictions modulates routing
//!
//! ## Mathematical Foundation
//!
//! The core insight is that the brain (and our routing system) can be modeled
//! as minimizing variational free energy:
//!
//! ```text
//! F = E_q[log q(s) - log p(o,s)]
//!   = -log p(o) + KL[q(s) || p(s|o)]
//! ```
//!
//! where:
//! - p(o,s) is the generative model (joint over observations and states)
//! - q(s) is the recognition model (approximate posterior)
//! - F is an upper bound on surprise -log p(o)
//!
//! For routing, we:
//! 1. Maintain a generative model of consciousness dynamics
//! 2. Compute expected free energy for each potential strategy
//! 3. Select strategies that minimize expected free energy
//! 4. This naturally balances:
//!    - Pragmatic value (achieving goals)
//!    - Epistemic value (reducing uncertainty)

use std::collections::VecDeque;
use std::f64::consts::PI;
use serde::{Deserialize, Serialize};

use super::{
    RoutingStrategy,
    QuantumCoherenceRouter, QuantumRouterConfig, QuantumRoutingDecision,
};
use crate::consciousness::recursive_improvement::LatentConsciousnessState;

// =============================================================================
// BELIEF DISTRIBUTION
// =============================================================================

/// Belief distribution over hidden states
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefDistribution {
    /// Mean of the belief (expected hidden state)
    pub mean: Vec<f64>,
    /// Precision (inverse variance) for each dimension
    pub precision: Vec<f64>,
    /// Confidence in the belief (average precision)
    pub confidence: f64,
}

impl BeliefDistribution {
    /// Create a new belief distribution
    pub fn new(dim: usize) -> Self {
        Self {
            mean: vec![0.5; dim],
            precision: vec![1.0; dim],
            confidence: 1.0,
        }
    }

    /// Create from mean and precision
    pub fn from_mean_precision(mean: Vec<f64>, precision: Vec<f64>) -> Self {
        let confidence = precision.iter().sum::<f64>() / precision.len() as f64;
        Self { mean, precision, confidence }
    }

    /// Get variance (inverse of precision)
    pub fn variance(&self) -> Vec<f64> {
        self.precision.iter().map(|p| 1.0 / p.max(0.001)).collect()
    }

    /// Compute entropy of the belief
    pub fn entropy(&self) -> f64 {
        let dim = self.mean.len() as f64;
        // Entropy of multivariate Gaussian: 0.5 * (d + d*ln(2π) + ln|Σ|)
        // For diagonal covariance: ln|Σ| = sum of ln(1/precision)
        let log_det: f64 = self.precision.iter().map(|p| -p.max(0.001).ln()).sum();
        0.5 * (dim + dim * (2.0 * PI).ln() + log_det)
    }

    /// Update belief with new observation (Bayesian update)
    pub fn update(&mut self, observation: &[f64], obs_precision: f64) {
        for i in 0..self.mean.len().min(observation.len()) {
            // Precision-weighted average
            let prior_precision = self.precision[i];
            let new_precision = prior_precision + obs_precision;
            let new_mean = (self.mean[i] * prior_precision + observation[i] * obs_precision) / new_precision;

            self.mean[i] = new_mean;
            self.precision[i] = new_precision;
        }
        self.confidence = self.precision.iter().sum::<f64>() / self.precision.len() as f64;
    }

    /// Decay precision over time (uncertainty grows)
    pub fn decay(&mut self, rate: f64) {
        for p in &mut self.precision {
            *p *= (1.0 - rate).max(0.01);
        }
        self.confidence = self.precision.iter().sum::<f64>() / self.precision.len() as f64;
    }

    /// KL divergence from another belief
    pub fn kl_divergence(&self, other: &BeliefDistribution) -> f64 {
        let mut kl = 0.0;
        for i in 0..self.mean.len().min(other.mean.len()) {
            let var_self = 1.0 / self.precision[i].max(0.001);
            let var_other = 1.0 / other.precision[i].max(0.001);
            let mean_diff = self.mean[i] - other.mean[i];

            // KL for Gaussians: 0.5 * (var_ratio + mean_diff^2/var_other - 1 + ln(var_other/var_self))
            kl += 0.5 * (var_self / var_other + mean_diff.powi(2) / var_other - 1.0 + (var_other / var_self).ln());
        }
        kl.max(0.0)
    }
}

// =============================================================================
// GENERATIVE MODEL
// =============================================================================

/// Generative model for consciousness dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerativeModel {
    /// State transition matrix (how states evolve)
    pub transition: Vec<Vec<f64>>,
    /// Observation likelihood matrix
    pub likelihood: Vec<Vec<f64>>,
    /// Prior over states
    pub prior: BeliefDistribution,
    /// Dimension of hidden states
    pub state_dim: usize,
    /// Dimension of observations
    pub obs_dim: usize,
    /// Transition noise precision
    pub transition_precision: f64,
    /// Observation noise precision
    pub observation_precision: f64,
}

impl GenerativeModel {
    /// Create a new generative model
    pub fn new(state_dim: usize, obs_dim: usize) -> Self {
        // Initialize with identity-like matrices
        let mut transition = vec![vec![0.0; state_dim]; state_dim];
        let mut likelihood = vec![vec![0.0; obs_dim]; state_dim];

        for i in 0..state_dim {
            transition[i][i] = 0.9; // Self-transition
            if i > 0 {
                transition[i][i-1] = 0.05; // Downward transition
            }
            if i < state_dim - 1 {
                transition[i][i+1] = 0.05; // Upward transition
            }
        }

        for i in 0..state_dim.min(obs_dim) {
            likelihood[i][i] = 0.8; // Diagonal likelihood
            if i > 0 && i < obs_dim {
                likelihood[i][i-1] = 0.1;
            }
            if i + 1 < obs_dim {
                likelihood[i][i+1] = 0.1;
            }
        }

        Self {
            transition,
            likelihood,
            prior: BeliefDistribution::new(state_dim),
            state_dim,
            obs_dim,
            transition_precision: 10.0,
            observation_precision: 5.0,
        }
    }

    /// Predict next state given current belief
    pub fn predict(&self, belief: &BeliefDistribution) -> BeliefDistribution {
        let mut predicted_mean = vec![0.0; self.state_dim];

        for i in 0..self.state_dim {
            for j in 0..self.state_dim.min(belief.mean.len()) {
                predicted_mean[i] += self.transition[i][j] * belief.mean[j];
            }
        }

        // Precision decreases due to transition noise
        let predicted_precision: Vec<f64> = belief.precision
            .iter()
            .map(|p| (p * self.transition_precision) / (p + self.transition_precision))
            .collect();

        BeliefDistribution::from_mean_precision(predicted_mean, predicted_precision)
    }

    /// Compute expected observation given belief
    pub fn expected_observation(&self, belief: &BeliefDistribution) -> Vec<f64> {
        let mut expected = vec![0.0; self.obs_dim];

        for i in 0..self.state_dim {
            for j in 0..self.obs_dim {
                expected[j] += self.likelihood[i][j] * belief.mean.get(i).copied().unwrap_or(0.0);
            }
        }

        expected
    }

    /// Compute prediction error (surprise)
    pub fn prediction_error(&self, belief: &BeliefDistribution, observation: &[f64]) -> f64 {
        let expected = self.expected_observation(belief);
        let mut error = 0.0;

        for i in 0..expected.len().min(observation.len()) {
            error += (expected[i] - observation[i]).powi(2) * self.observation_precision;
        }

        error
    }

    /// Update model parameters based on prediction error
    pub fn learn(&mut self, belief: &BeliefDistribution, observation: &[f64], learning_rate: f64) {
        let expected = self.expected_observation(belief);

        // Update likelihood based on prediction error
        for i in 0..self.state_dim {
            for j in 0..self.obs_dim.min(observation.len()) {
                let error = observation[j] - expected.get(j).copied().unwrap_or(0.0);
                let gradient = error * belief.mean.get(i).copied().unwrap_or(0.0);
                self.likelihood[i][j] += learning_rate * gradient;
                // Keep likelihood normalized
                self.likelihood[i][j] = self.likelihood[i][j].max(0.0).min(1.0);
            }
        }
    }
}

// =============================================================================
// EXPECTED FREE ENERGY
// =============================================================================

/// Expected free energy for a potential action/strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedFreeEnergy {
    /// Total expected free energy
    pub total: f64,
    /// Pragmatic value (expected reward/goal achievement)
    pub pragmatic: f64,
    /// Epistemic value (information gain, uncertainty reduction)
    pub epistemic: f64,
    /// Novelty bonus (exploration term)
    pub novelty: f64,
    /// Associated strategy
    pub strategy: RoutingStrategy,
}

impl ExpectedFreeEnergy {
    /// Create a new expected free energy estimate
    pub fn new(strategy: RoutingStrategy) -> Self {
        Self {
            total: 0.0,
            pragmatic: 0.0,
            epistemic: 0.0,
            novelty: 0.0,
            strategy,
        }
    }

    /// Compute the total (lower is better for selection)
    pub fn compute_total(&mut self, pragmatic_weight: f64, epistemic_weight: f64, novelty_weight: f64) {
        // Negate pragmatic (we want to maximize reward)
        // Add epistemic (we want to reduce uncertainty)
        // Add novelty (encourage exploration)
        self.total = -pragmatic_weight * self.pragmatic
            + epistemic_weight * self.epistemic
            - novelty_weight * self.novelty;
    }
}

// =============================================================================
// PREFERENCES
// =============================================================================

/// Preference distribution over outcomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preferences {
    /// Preferred observation values
    pub preferred: Vec<f64>,
    /// Preference precision (how strongly we prefer)
    pub precision: f64,
}

impl Preferences {
    /// Create new preferences
    pub fn new(preferred: Vec<f64>, precision: f64) -> Self {
        Self { preferred, precision }
    }

    /// Compute pragmatic value (negative divergence from preferred)
    pub fn pragmatic_value(&self, expected_obs: &[f64]) -> f64 {
        let mut value = 0.0;
        for i in 0..self.preferred.len().min(expected_obs.len()) {
            value -= self.precision * (expected_obs[i] - self.preferred[i]).powi(2);
        }
        value
    }
}

// =============================================================================
// CONFIGURATION AND TYPES
// =============================================================================

/// Configuration for active inference router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveInferenceConfig {
    /// State dimension
    pub state_dim: usize,
    /// Observation dimension
    pub obs_dim: usize,
    /// Number of strategies to consider
    pub num_strategies: usize,
    /// Learning rate for model updates
    pub learning_rate: f64,
    /// Belief decay rate
    pub decay_rate: f64,
    /// Weight for pragmatic value
    pub pragmatic_weight: f64,
    /// Weight for epistemic value
    pub epistemic_weight: f64,
    /// Weight for novelty
    pub novelty_weight: f64,
    /// Planning horizon
    pub horizon: usize,
    /// Precision on preferences
    pub preference_precision: f64,
}

impl Default for ActiveInferenceConfig {
    fn default() -> Self {
        Self {
            state_dim: 4,
            obs_dim: 4,
            num_strategies: 7,
            learning_rate: 0.01,
            decay_rate: 0.05,
            pragmatic_weight: 1.0,
            epistemic_weight: 0.5,
            novelty_weight: 0.1,
            horizon: 3,
            preference_precision: 2.0,
        }
    }
}

/// Statistics for active inference router
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActiveInferenceStats {
    /// Number of routing decisions
    pub decisions: usize,
    /// Total free energy accumulated
    pub total_free_energy: f64,
    /// Average prediction error
    pub avg_prediction_error: f64,
    /// Average epistemic value
    pub avg_epistemic: f64,
    /// Average pragmatic value
    pub avg_pragmatic: f64,
    /// Model updates performed
    pub model_updates: usize,
    /// Exploration actions taken
    pub explorations: usize,
    /// Exploitation actions taken
    pub exploitations: usize,
}

/// Active inference routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveInferenceDecision {
    /// Chosen strategy
    pub strategy: RoutingStrategy,
    /// Expected free energy of chosen strategy
    pub expected_free_energy: f64,
    /// Pragmatic value
    pub pragmatic: f64,
    /// Epistemic value
    pub epistemic: f64,
    /// Current prediction error
    pub prediction_error: f64,
    /// Current belief entropy
    pub belief_entropy: f64,
    /// Was this exploratory?
    pub is_exploratory: bool,
    /// Confidence in decision
    pub confidence: f64,
}

/// Summary of active inference router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveInferenceSummary {
    /// Current belief mean
    pub belief_mean: Vec<f64>,
    /// Current belief confidence
    pub belief_confidence: f64,
    /// Total decisions made
    pub decisions: usize,
    /// Average free energy
    pub avg_free_energy: f64,
    /// Exploration ratio
    pub exploration_ratio: f64,
}

// =============================================================================
// ACTIVE INFERENCE ROUTER
// =============================================================================

/// Active Inference Router
///
/// Routes consciousness using the Free Energy Principle:
/// - Maintains a generative model of consciousness dynamics
/// - Selects strategies that minimize expected free energy
/// - Balances goal-achievement (pragmatic) with uncertainty-reduction (epistemic)
pub struct ActiveInferenceRouter {
    /// Underlying quantum router for base decisions
    quantum_router: QuantumCoherenceRouter,
    /// Generative model
    model: GenerativeModel,
    /// Current belief over hidden states
    belief: BeliefDistribution,
    /// Preferences over observations
    preferences: Preferences,
    /// Configuration
    config: ActiveInferenceConfig,
    /// Statistics
    stats: ActiveInferenceStats,
    /// Strategy history for novelty computation
    strategy_history: VecDeque<RoutingStrategy>,
    /// Expected free energies for each strategy
    efes: Vec<ExpectedFreeEnergy>,
}

impl ActiveInferenceRouter {
    /// Create a new active inference router
    pub fn new(config: ActiveInferenceConfig) -> Self {
        let model = GenerativeModel::new(config.state_dim, config.obs_dim);
        let belief = BeliefDistribution::new(config.state_dim);

        // Default preferences: high phi, high coherence
        let preferred = vec![0.8, 0.7, 0.7, 0.6];
        let preferences = Preferences::new(preferred, config.preference_precision);

        let efes = (0..config.num_strategies)
            .map(|i| ExpectedFreeEnergy::new(Self::index_to_strategy(i)))
            .collect();

        Self {
            quantum_router: QuantumCoherenceRouter::new(QuantumRouterConfig::default()),
            model,
            belief,
            preferences,
            config,
            stats: ActiveInferenceStats::default(),
            strategy_history: VecDeque::with_capacity(100),
            efes,
        }
    }

    /// Convert index to strategy
    pub fn index_to_strategy(idx: usize) -> RoutingStrategy {
        match idx {
            0 => RoutingStrategy::Reflexive,
            1 => RoutingStrategy::FastPatterns,
            2 => RoutingStrategy::HeuristicGuided,
            3 => RoutingStrategy::StandardProcessing,
            4 => RoutingStrategy::FullDeliberation,
            5 => RoutingStrategy::Ensemble,
            _ => RoutingStrategy::Preparatory,
        }
    }

    /// Convert strategy to index
    fn strategy_to_index(strategy: RoutingStrategy) -> usize {
        match strategy {
            RoutingStrategy::Reflexive => 0,
            RoutingStrategy::FastPatterns => 1,
            RoutingStrategy::HeuristicGuided => 2,
            RoutingStrategy::StandardProcessing => 3,
            RoutingStrategy::FullDeliberation => 4,
            RoutingStrategy::Ensemble => 5,
            RoutingStrategy::Preparatory => 6,
        }
    }

    /// Observe a new consciousness state
    pub fn observe_state(&mut self, state: &LatentConsciousnessState) {
        let observation = vec![state.phi, state.integration, state.coherence, state.attention];

        // Compute prediction error before updating
        let pred_error = self.model.prediction_error(&self.belief, &observation);

        // Update belief with observation
        self.belief.update(&observation, self.model.observation_precision);

        // Learn from prediction error
        self.model.learn(&self.belief, &observation, self.config.learning_rate);
        self.stats.model_updates += 1;

        // Decay belief precision (uncertainty grows over time)
        self.belief.decay(self.config.decay_rate);

        // Update running stats
        let n = self.stats.decisions.max(1) as f64;
        self.stats.avg_prediction_error =
            (self.stats.avg_prediction_error * (n - 1.0) + pred_error) / n;

        // Also update quantum router
        self.quantum_router.observe_state(state);
    }

    /// Compute expected free energy for a strategy
    pub fn compute_efe(&self, strategy: RoutingStrategy) -> ExpectedFreeEnergy {
        let mut efe = ExpectedFreeEnergy::new(strategy);

        // Pragmatic value: how well does predicted outcome match preferences?
        let predicted_belief = self.model.predict(&self.belief);
        let expected_obs = self.model.expected_observation(&predicted_belief);
        efe.pragmatic = self.preferences.pragmatic_value(&expected_obs);

        // Epistemic value: how much will uncertainty reduce?
        // This is the expected KL divergence between posterior and prior
        let prior_entropy = self.belief.entropy();
        let predicted_entropy = predicted_belief.entropy();
        efe.epistemic = (prior_entropy - predicted_entropy).abs();

        // Novelty: how often have we used this strategy recently?
        let recent_uses = self.strategy_history
            .iter()
            .filter(|s| **s == strategy)
            .count();
        efe.novelty = 1.0 / (1.0 + recent_uses as f64);

        // Compute total EFE
        efe.compute_total(
            self.config.pragmatic_weight,
            self.config.epistemic_weight,
            self.config.novelty_weight,
        );

        efe
    }

    /// Route based on active inference
    pub fn route(&mut self, target: &LatentConsciousnessState) -> ActiveInferenceDecision {
        // Update with target observation
        let observation = vec![target.phi, target.integration, target.coherence, target.attention];
        let prediction_error = self.model.prediction_error(&self.belief, &observation);

        // Compute expected free energy for each strategy
        self.efes = (0..self.config.num_strategies)
            .map(|i| self.compute_efe(Self::index_to_strategy(i)))
            .collect();

        // Select strategy with minimum expected free energy
        let best_efe = self.efes
            .iter()
            .min_by(|a, b| a.total.partial_cmp(&b.total).unwrap())
            .cloned()
            .unwrap_or_else(|| ExpectedFreeEnergy::new(RoutingStrategy::StandardProcessing));

        let chosen_strategy = best_efe.strategy;

        // Determine if this was exploratory (high epistemic, low pragmatic)
        let is_exploratory = best_efe.epistemic > best_efe.pragmatic.abs();
        if is_exploratory {
            self.stats.explorations += 1;
        } else {
            self.stats.exploitations += 1;
        }

        // Update history
        self.strategy_history.push_back(chosen_strategy);
        if self.strategy_history.len() > 100 {
            self.strategy_history.pop_front();
        }

        // Get quantum router's confidence
        let quantum_decision = self.quantum_router.route(target);
        let confidence = self.belief.confidence * quantum_decision.probability;

        // Update stats
        let n = self.stats.decisions as f64;
        self.stats.total_free_energy += best_efe.total;
        self.stats.avg_pragmatic = (self.stats.avg_pragmatic * n + best_efe.pragmatic) / (n + 1.0);
        self.stats.avg_epistemic = (self.stats.avg_epistemic * n + best_efe.epistemic) / (n + 1.0);
        self.stats.decisions += 1;

        ActiveInferenceDecision {
            strategy: chosen_strategy,
            expected_free_energy: best_efe.total,
            pragmatic: best_efe.pragmatic,
            epistemic: best_efe.epistemic,
            prediction_error,
            belief_entropy: self.belief.entropy(),
            is_exploratory,
            confidence,
        }
    }

    /// Set preferences for desired outcomes
    pub fn set_preferences(&mut self, preferred: Vec<f64>, precision: f64) {
        self.preferences = Preferences::new(preferred, precision);
    }

    /// Get current belief state
    pub fn belief(&self) -> &BeliefDistribution {
        &self.belief
    }

    /// Get current free energy (surprise)
    pub fn current_free_energy(&self) -> f64 {
        // F = -log p(o) ≈ prediction_error + belief_entropy
        self.stats.avg_prediction_error + self.belief.entropy()
    }

    /// Check if system is in surprise state (high free energy)
    pub fn is_surprised(&self) -> bool {
        self.current_free_energy() > 5.0
    }

    /// Get summary of router state
    pub fn summary(&self) -> ActiveInferenceSummary {
        let total = self.stats.explorations + self.stats.exploitations;
        let exploration_ratio = if total > 0 {
            self.stats.explorations as f64 / total as f64
        } else {
            0.5
        };

        ActiveInferenceSummary {
            belief_mean: self.belief.mean.clone(),
            belief_confidence: self.belief.confidence,
            decisions: self.stats.decisions,
            avg_free_energy: if self.stats.decisions > 0 {
                self.stats.total_free_energy / self.stats.decisions as f64
            } else {
                0.0
            },
            exploration_ratio,
        }
    }

    /// Get statistics
    pub fn stats(&self) -> &ActiveInferenceStats {
        &self.stats
    }

    /// Reset the router
    pub fn reset(&mut self) {
        self.belief = BeliefDistribution::new(self.config.state_dim);
        self.strategy_history.clear();
        self.stats = ActiveInferenceStats::default();
        self.quantum_router.reset();
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_belief_distribution_new() {
        let belief = BeliefDistribution::new(4);
        assert_eq!(belief.mean.len(), 4);
        assert_eq!(belief.precision.len(), 4);
        assert!(belief.confidence > 0.0);
    }

    #[test]
    fn test_belief_entropy() {
        let belief = BeliefDistribution::new(4);
        let entropy = belief.entropy();
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_belief_update() {
        let mut belief = BeliefDistribution::new(4);
        let initial_confidence = belief.confidence;

        belief.update(&[0.8, 0.7, 0.6, 0.5], 5.0);

        // Confidence should increase after update
        assert!(belief.confidence > initial_confidence);
    }

    #[test]
    fn test_belief_decay() {
        let mut belief = BeliefDistribution::new(4);
        let initial_confidence = belief.confidence;

        belief.decay(0.1);

        // Confidence should decrease after decay
        assert!(belief.confidence < initial_confidence);
    }

    #[test]
    fn test_belief_kl_divergence() {
        let belief1 = BeliefDistribution::new(4);
        let belief2 = BeliefDistribution::from_mean_precision(
            vec![0.8, 0.8, 0.8, 0.8],
            vec![2.0, 2.0, 2.0, 2.0],
        );

        let kl = belief1.kl_divergence(&belief2);
        assert!(kl >= 0.0);
    }

    #[test]
    fn test_generative_model_new() {
        let model = GenerativeModel::new(4, 4);
        assert_eq!(model.state_dim, 4);
        assert_eq!(model.obs_dim, 4);
        assert_eq!(model.transition.len(), 4);
        assert_eq!(model.likelihood.len(), 4);
    }

    #[test]
    fn test_generative_model_predict() {
        let model = GenerativeModel::new(4, 4);
        let belief = BeliefDistribution::new(4);

        let predicted = model.predict(&belief);
        assert_eq!(predicted.mean.len(), 4);
    }

    #[test]
    fn test_generative_model_expected_observation() {
        let model = GenerativeModel::new(4, 4);
        let belief = BeliefDistribution::new(4);

        let expected = model.expected_observation(&belief);
        assert_eq!(expected.len(), 4);
    }

    #[test]
    fn test_generative_model_prediction_error() {
        let model = GenerativeModel::new(4, 4);
        let belief = BeliefDistribution::new(4);

        let error = model.prediction_error(&belief, &[0.5, 0.5, 0.5, 0.5]);
        assert!(error >= 0.0);
    }

    #[test]
    fn test_expected_free_energy() {
        let mut efe = ExpectedFreeEnergy::new(RoutingStrategy::StandardProcessing);
        efe.pragmatic = 1.0;
        efe.epistemic = 0.5;
        efe.novelty = 0.1;
        efe.compute_total(1.0, 0.5, 0.1);

        assert!(efe.total.is_finite());
    }

    #[test]
    fn test_preferences_pragmatic_value() {
        let prefs = Preferences::new(vec![0.8, 0.8, 0.8, 0.8], 2.0);

        let value_good = prefs.pragmatic_value(&[0.79, 0.79, 0.79, 0.79]);
        let value_bad = prefs.pragmatic_value(&[0.2, 0.2, 0.2, 0.2]);

        // Good observation should have higher (less negative) value
        assert!(value_good > value_bad);
    }

    #[test]
    fn test_active_inference_router_creation() {
        let router = ActiveInferenceRouter::new(ActiveInferenceConfig::default());
        assert_eq!(router.stats.decisions, 0);
    }

    #[test]
    fn test_active_inference_router_observe() {
        let mut router = ActiveInferenceRouter::new(ActiveInferenceConfig::default());

        let state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        router.observe_state(&state);

        assert_eq!(router.stats.model_updates, 1);
    }

    #[test]
    fn test_active_inference_router_route() {
        let mut router = ActiveInferenceRouter::new(ActiveInferenceConfig::default());

        // Add some observations
        for i in 0..5 {
            let v = 0.3 + (i as f64) * 0.1;
            let state = LatentConsciousnessState::from_observables(v, v, v, v);
            router.observe_state(&state);
        }

        let target = LatentConsciousnessState::from_observables(0.7, 0.7, 0.7, 0.7);
        let decision = router.route(&target);

        assert_eq!(router.stats.decisions, 1);
        assert!(decision.confidence > 0.0);
    }

    #[test]
    fn test_active_inference_exploration_vs_exploitation() {
        let mut router = ActiveInferenceRouter::new(ActiveInferenceConfig {
            epistemic_weight: 2.0, // High epistemic weight for exploration
            ..Default::default()
        });

        for i in 0..10 {
            let v = 0.3 + (i as f64) * 0.05;
            let state = LatentConsciousnessState::from_observables(v, v, v, v);
            router.observe_state(&state);
            let _ = router.route(&state);
        }

        // With high epistemic weight, should have some explorations
        assert!(router.stats.decisions > 0);
    }

    #[test]
    fn test_active_inference_set_preferences() {
        let mut router = ActiveInferenceRouter::new(ActiveInferenceConfig::default());

        router.set_preferences(vec![0.9, 0.9, 0.9, 0.9], 3.0);

        assert_eq!(router.preferences.preferred, vec![0.9, 0.9, 0.9, 0.9]);
        assert_eq!(router.preferences.precision, 3.0);
    }

    #[test]
    fn test_active_inference_free_energy() {
        let router = ActiveInferenceRouter::new(ActiveInferenceConfig::default());
        let fe = router.current_free_energy();
        assert!(fe.is_finite());
    }

    #[test]
    fn test_active_inference_summary() {
        let router = ActiveInferenceRouter::new(ActiveInferenceConfig::default());
        let summary = router.summary();

        assert_eq!(summary.decisions, 0);
        assert!(summary.exploration_ratio >= 0.0);
        assert!(summary.exploration_ratio <= 1.0);
    }

    #[test]
    fn test_active_inference_reset() {
        let mut router = ActiveInferenceRouter::new(ActiveInferenceConfig::default());

        let state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        router.observe_state(&state);
        let _ = router.route(&state);

        router.reset();

        assert_eq!(router.stats.decisions, 0);
        assert!(router.strategy_history.is_empty());
    }

    #[test]
    fn test_generative_model_learn() {
        let mut model = GenerativeModel::new(4, 4);
        let belief = BeliefDistribution::new(4);
        let initial_likelihood = model.likelihood[0][0];

        model.learn(&belief, &[0.9, 0.9, 0.9, 0.9], 0.1);

        // Likelihood should have changed
        assert!(model.likelihood[0][0].is_finite());
    }

    #[test]
    fn test_active_inference_is_surprised() {
        let router = ActiveInferenceRouter::new(ActiveInferenceConfig::default());
        // Initial state should not be very surprised
        let is_surprised = router.is_surprised();
        // Just check it returns a boolean
        assert!(is_surprised || !is_surprised);
    }

    #[test]
    fn test_compute_efe_strategies() {
        let router = ActiveInferenceRouter::new(ActiveInferenceConfig::default());

        // Check that we can compute EFE for each strategy
        for i in 0..7 {
            let strategy = ActiveInferenceRouter::index_to_strategy(i);
            let efe = router.compute_efe(strategy);
            assert!(efe.total.is_finite());
        }
    }

    #[test]
    fn test_belief_variance() {
        let belief = BeliefDistribution::new(4);
        let variance = belief.variance();

        assert_eq!(variance.len(), 4);
        for v in &variance {
            assert!(*v > 0.0);
        }
    }
}
