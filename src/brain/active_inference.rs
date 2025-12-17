//! # Active Inference Engine
//!
//! **Week 18: Paradigm-Shifting Foundation - Free Energy Principle**
//!
//! This module implements Karl Friston's Active Inference framework, providing
//! the theoretical foundation that unifies perception, action, and learning
//! under a single objective: minimize variational free energy (surprise).
//!
//! ## Revolutionary Concepts
//!
//! ### 1. Free Energy Principle
//! All adaptive systems minimize surprise by either:
//! - **Perception**: Update beliefs to match observations (perceptual inference)
//! - **Action**: Change the world to match predictions (active inference)
//!
//! ### 2. Precision-Weighted Prediction Errors
//! Not all prediction errors are equal. Precision (confidence in predictions)
//! determines how much weight errors receive. High precision = high importance.
//!
//! ### 3. Epistemic vs Pragmatic Actions
//! - **Pragmatic**: Actions that achieve goals (reduce expected free energy)
//! - **Epistemic**: Actions that reduce uncertainty (curiosity-driven exploration)
//!
//! ### 4. Generative Models
//! The system maintains internal models that predict sensory inputs.
//! Learning = updating these models to better predict reality.
//!
//! ## Mathematical Foundation
//!
//! Free Energy F = D_KL[q(s) || p(s|o)] + E_q[-ln p(o)]
//!             ≈ complexity + inaccuracy
//!             ≈ divergence from prior + prediction error
//!
//! The system minimizes F by:
//! 1. Updating beliefs q(s) to match observations (perception)
//! 2. Selecting actions that minimize expected F (action)
//! 3. Learning model parameters to improve predictions (learning)

use std::collections::{HashMap, VecDeque};
use serde::{Serialize, Deserialize};

/// Default instant for deserialization
fn default_instant() -> std::time::Instant {
    std::time::Instant::now()
}

/// Precision-weighted prediction error
///
/// The core unit of active inference. Represents the mismatch between
/// predicted and actual observations, weighted by confidence (precision).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionError {
    /// What was predicted
    pub expected: f32,
    /// What was observed
    pub observed: f32,
    /// Raw error (observed - expected)
    pub error: f32,
    /// Precision (confidence in prediction, 0.0 to 1.0)
    /// Higher precision = this error matters more
    pub precision: f32,
    /// Precision-weighted error (the actual signal)
    pub weighted_error: f32,
    /// Domain of this prediction (coherence, performance, safety, etc.)
    pub domain: PredictionDomain,
    /// Timestamp for temporal tracking (skipped for serialization)
    #[serde(skip, default = "default_instant")]
    pub timestamp: std::time::Instant,
}

/// Domains where predictions can be made
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PredictionDomain {
    /// Coherence level predictions
    Coherence,
    /// Task success predictions
    TaskSuccess,
    /// User state predictions
    UserState,
    /// System performance predictions
    Performance,
    /// Safety threat predictions
    Safety,
    /// Energy/resource predictions
    Energy,
    /// Social coherence predictions
    Social,
    /// Time perception predictions
    Temporal,
}

impl PredictionError {
    /// Create a new prediction error
    pub fn new(expected: f32, observed: f32, precision: f32, domain: PredictionDomain) -> Self {
        let error = observed - expected;
        let weighted_error = error * precision;

        Self {
            expected,
            observed,
            error,
            precision,
            weighted_error,
            domain,
            timestamp: std::time::Instant::now(),
        }
    }

    /// Surprise value (unsigned magnitude of precision-weighted error)
    pub fn surprise(&self) -> f32 {
        self.weighted_error.abs()
    }

    /// Was this a positive surprise (better than expected)?
    pub fn is_positive_surprise(&self) -> bool {
        // For most domains, higher observed = better
        // For Safety domain, lower observed (threat) = worse
        match self.domain {
            PredictionDomain::Safety => self.error < 0.0,
            _ => self.error > 0.0,
        }
    }
}

/// Generative model for a prediction domain
///
/// Maintains beliefs about the hidden state and generates predictions.
/// Updates through variational inference when observations arrive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerativeModel {
    /// Domain this model predicts
    pub domain: PredictionDomain,
    /// Current belief about hidden state (mean)
    pub belief_mean: f32,
    /// Uncertainty in belief (variance)
    pub belief_variance: f32,
    /// Prior mean (what we expect by default)
    pub prior_mean: f32,
    /// Prior variance (how uncertain we are by default)
    pub prior_variance: f32,
    /// Observation noise (how noisy are observations)
    pub observation_variance: f32,
    /// Learning rate for model updates
    pub learning_rate: f32,
    /// History of prediction errors for tracking
    #[serde(skip)]
    pub error_history: VecDeque<f32>,
    /// Maximum history size
    pub max_history: usize,
    /// Total observations
    pub observation_count: u64,
}

impl GenerativeModel {
    /// Create a new generative model
    pub fn new(domain: PredictionDomain) -> Self {
        let (prior_mean, prior_variance) = match domain {
            PredictionDomain::Coherence => (0.7, 0.1),      // Expect ~70% coherence
            PredictionDomain::TaskSuccess => (0.8, 0.15),   // Expect ~80% success
            PredictionDomain::UserState => (0.5, 0.2),      // Neutral user state
            PredictionDomain::Performance => (0.6, 0.1),    // Expect ~60% performance
            PredictionDomain::Safety => (0.1, 0.05),        // Expect low threat
            PredictionDomain::Energy => (0.5, 0.15),        // Expect ~50% energy
            PredictionDomain::Social => (0.5, 0.2),         // Neutral social
            PredictionDomain::Temporal => (1.0, 0.1),       // Normal time flow
        };

        Self {
            domain,
            belief_mean: prior_mean,
            belief_variance: prior_variance,
            prior_mean,
            prior_variance,
            observation_variance: 0.1,
            learning_rate: 0.1,
            error_history: VecDeque::with_capacity(100),
            max_history: 100,
            observation_count: 0,
        }
    }

    /// Generate a prediction from current beliefs
    pub fn predict(&self) -> f32 {
        self.belief_mean
    }

    /// Calculate precision (inverse variance) of prediction
    pub fn precision(&self) -> f32 {
        // Precision increases with:
        // 1. Lower belief variance (more certain)
        // 2. Higher observation count (more experience)
        let base_precision = 1.0 / (self.belief_variance + 0.01);
        let experience_factor = (self.observation_count as f32 / 100.0).min(1.0);

        (base_precision * (0.5 + 0.5 * experience_factor)).min(10.0)
    }

    /// Update beliefs given new observation (variational inference)
    ///
    /// Uses Kalman filter-style update:
    /// posterior = prior + K * (observation - prior)
    /// where K = prior_var / (prior_var + obs_var)
    pub fn update(&mut self, observation: f32) -> PredictionError {
        let prediction = self.predict();
        let precision = self.precision();

        // Create prediction error
        let error = PredictionError::new(prediction, observation, precision, self.domain);

        // Kalman gain (how much to update)
        let kalman_gain = self.belief_variance /
            (self.belief_variance + self.observation_variance);

        // Update belief mean
        self.belief_mean += kalman_gain * error.error * self.learning_rate;

        // Update belief variance (becomes more certain with observations)
        self.belief_variance = (1.0 - kalman_gain) * self.belief_variance;

        // Prevent variance from going to zero (always some uncertainty)
        self.belief_variance = self.belief_variance.max(0.01);

        // Drift back toward prior (prevents over-specialization)
        let prior_pull = 0.01;
        self.belief_mean = self.belief_mean * (1.0 - prior_pull) +
                          self.prior_mean * prior_pull;
        self.belief_variance = self.belief_variance * (1.0 - prior_pull) +
                              self.prior_variance * prior_pull;

        // Track history
        self.error_history.push_back(error.weighted_error);
        if self.error_history.len() > self.max_history {
            self.error_history.pop_front();
        }

        self.observation_count += 1;

        error
    }

    /// Calculate current free energy contribution from this model
    ///
    /// F = complexity + inaccuracy
    ///   = KL(posterior || prior) + expected_prediction_error
    pub fn free_energy(&self) -> f32 {
        // Complexity: KL divergence from prior
        // For Gaussians: KL = 0.5 * (var_q/var_p + (mu_q - mu_p)^2/var_p - 1 + ln(var_p/var_q))
        let complexity = 0.5 * (
            self.belief_variance / self.prior_variance +
            (self.belief_mean - self.prior_mean).powi(2) / self.prior_variance -
            1.0 +
            (self.prior_variance / self.belief_variance).ln()
        );

        // Inaccuracy: average prediction error (approximated from history)
        let inaccuracy = if self.error_history.is_empty() {
            0.1 // Default inaccuracy
        } else {
            let sum: f32 = self.error_history.iter().map(|e| e.abs()).sum();
            sum / self.error_history.len() as f32
        };

        complexity + inaccuracy
    }

    /// Calculate uncertainty (epistemic value of observing this domain)
    pub fn uncertainty(&self) -> f32 {
        // Higher variance = more uncertainty = more epistemic value
        // Also consider inverse of observation count
        let variance_term = self.belief_variance;
        let novelty_term = 1.0 / (1.0 + self.observation_count as f32 * 0.01);

        variance_term + novelty_term * 0.5
    }
}

/// Action types for active inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    /// Epistemic action: reduces uncertainty
    Epistemic {
        target_domain: PredictionDomain,
        expected_information_gain: f32,
    },
    /// Pragmatic action: achieves goals
    Pragmatic {
        goal: String,
        expected_utility: f32,
    },
    /// Centering action: reduces free energy through internal adjustment
    Centering {
        target_coherence: f32,
    },
    /// Social action: coordinate with others
    Social {
        target: String,
        expected_resonance_gain: f32,
    },
}

/// Active Inference Engine
///
/// The central system that implements the Free Energy Principle.
/// Manages generative models, tracks prediction errors, and
/// suggests actions that minimize expected free energy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveInferenceEngine {
    /// Generative models for each domain
    pub models: HashMap<PredictionDomain, GenerativeModel>,
    /// Recent prediction errors across all domains
    #[serde(skip)]
    pub recent_errors: VecDeque<PredictionError>,
    /// Maximum recent errors to track
    pub max_recent_errors: usize,
    /// Total free energy (surprise accumulated)
    pub total_free_energy: f32,
    /// Curiosity weight (how much to value epistemic actions)
    pub curiosity_weight: f32,
    /// Goal weight (how much to value pragmatic actions)
    pub goal_weight: f32,
    /// Temperature for action selection (exploration vs exploitation)
    pub temperature: f32,
    /// Statistics
    pub stats: ActiveInferenceStats,
}

/// Statistics for the active inference engine
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActiveInferenceStats {
    pub total_observations: u64,
    pub total_predictions: u64,
    pub epistemic_actions_suggested: u64,
    pub pragmatic_actions_suggested: u64,
    pub centering_actions_suggested: u64,
    pub average_surprise: f32,
    pub cumulative_free_energy: f32,
}

impl Default for ActiveInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ActiveInferenceEngine {
    /// Create a new active inference engine with default models
    pub fn new() -> Self {
        let mut models = HashMap::new();

        // Initialize all domain models
        for domain in [
            PredictionDomain::Coherence,
            PredictionDomain::TaskSuccess,
            PredictionDomain::UserState,
            PredictionDomain::Performance,
            PredictionDomain::Safety,
            PredictionDomain::Energy,
            PredictionDomain::Social,
            PredictionDomain::Temporal,
        ] {
            models.insert(domain, GenerativeModel::new(domain));
        }

        Self {
            models,
            recent_errors: VecDeque::with_capacity(100),
            max_recent_errors: 100,
            total_free_energy: 0.0,
            curiosity_weight: 0.3,  // Balance: 30% curiosity
            goal_weight: 0.7,       // 70% goal-directed
            temperature: 1.0,
            stats: ActiveInferenceStats::default(),
        }
    }

    /// Observe a value in a domain, update beliefs, return prediction error
    pub fn observe(&mut self, domain: PredictionDomain, value: f32) -> PredictionError {
        self.stats.total_observations += 1;

        let model = self.models.get_mut(&domain)
            .expect("All domains should have models");

        let error = model.update(value);

        // Track in recent errors
        self.recent_errors.push_back(error.clone());
        if self.recent_errors.len() > self.max_recent_errors {
            self.recent_errors.pop_front();
        }

        // Update total free energy
        self.total_free_energy += error.surprise();
        self.stats.cumulative_free_energy += error.surprise();

        // Update average surprise
        let alpha = 0.1;
        self.stats.average_surprise = self.stats.average_surprise * (1.0 - alpha) +
                                      error.surprise() * alpha;

        error
    }

    /// Generate prediction for a domain
    pub fn predict(&mut self, domain: PredictionDomain) -> (f32, f32) {
        self.stats.total_predictions += 1;

        let model = self.models.get(&domain)
            .expect("All domains should have models");

        (model.predict(), model.precision())
    }

    /// Calculate total free energy across all models
    pub fn total_free_energy_estimate(&self) -> f32 {
        self.models.values().map(|m| m.free_energy()).sum()
    }

    /// Calculate expected free energy for an action
    ///
    /// G = E[F after action] + pragmatic_value - epistemic_value
    ///
    /// We want to minimize G, so:
    /// - Actions that reduce expected surprise are good (pragmatic)
    /// - Actions that reduce uncertainty are good (epistemic)
    pub fn expected_free_energy(&self, action: &ActionType) -> f32 {
        match action {
            ActionType::Epistemic { target_domain, expected_information_gain } => {
                // Epistemic actions: value = uncertainty reduced
                let model = self.models.get(target_domain).unwrap();
                let current_uncertainty = model.uncertainty();

                // Expected free energy = -information_gain * curiosity_weight
                // Negative because we want to minimize, and info gain is good
                -expected_information_gain * self.curiosity_weight * current_uncertainty
            }

            ActionType::Pragmatic { expected_utility, .. } => {
                // Pragmatic actions: value = expected utility
                // Expected free energy = -utility * goal_weight
                -expected_utility * self.goal_weight
            }

            ActionType::Centering { target_coherence } => {
                // Centering: reduces internal free energy
                let coherence_model = self.models.get(&PredictionDomain::Coherence).unwrap();
                let current = coherence_model.belief_mean;

                // Value = how much closer we get to target
                let improvement = (target_coherence - current).max(0.0);
                -improvement * 0.5
            }

            ActionType::Social { expected_resonance_gain, .. } => {
                // Social actions: value = resonance gain + uncertainty reduction
                let social_model = self.models.get(&PredictionDomain::Social).unwrap();
                let uncertainty_reduction = social_model.uncertainty() * 0.3;

                -(expected_resonance_gain + uncertainty_reduction)
            }
        }
    }

    /// Suggest the best action from a set of candidates
    pub fn suggest_action(&mut self, candidates: &[ActionType]) -> Option<ActionType> {
        if candidates.is_empty() {
            return None;
        }

        // Calculate expected free energy for each
        let mut scored: Vec<(f32, &ActionType)> = candidates.iter()
            .map(|a| (self.expected_free_energy(a), a))
            .collect();

        // Sort by expected free energy (lower is better)
        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Track stats
        match scored[0].1 {
            ActionType::Epistemic { .. } => self.stats.epistemic_actions_suggested += 1,
            ActionType::Pragmatic { .. } => self.stats.pragmatic_actions_suggested += 1,
            ActionType::Centering { .. } => self.stats.centering_actions_suggested += 1,
            ActionType::Social { .. } => self.stats.pragmatic_actions_suggested += 1,
        }

        Some(scored[0].1.clone())
    }

    /// Get domain with highest uncertainty (most curiosity-worthy)
    pub fn most_uncertain_domain(&self) -> (PredictionDomain, f32) {
        self.models.iter()
            .map(|(domain, model)| (*domain, model.uncertainty()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
    }

    /// Get domain with highest prediction error (most surprising)
    pub fn most_surprising_domain(&self) -> Option<(PredictionDomain, f32)> {
        // Average recent errors by domain
        let mut domain_errors: HashMap<PredictionDomain, Vec<f32>> = HashMap::new();

        for error in &self.recent_errors {
            domain_errors.entry(error.domain)
                .or_default()
                .push(error.surprise());
        }

        domain_errors.iter()
            .map(|(domain, errors)| {
                let avg = errors.iter().sum::<f32>() / errors.len() as f32;
                (*domain, avg)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }

    /// Decay free energy over time (natural entropy reduction)
    pub fn decay(&mut self, delta_time: f32) {
        let decay_rate = 0.1;
        self.total_free_energy *= (1.0 - decay_rate * delta_time).max(0.0);
    }

    /// Generate curiosity-driven exploration suggestions
    pub fn curiosity_suggestions(&self, max_suggestions: usize) -> Vec<ActionType> {
        let mut suggestions = Vec::new();

        // Get domains sorted by uncertainty
        let mut domains: Vec<_> = self.models.iter()
            .map(|(d, m)| (*d, m.uncertainty()))
            .collect();
        domains.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Generate epistemic actions for most uncertain domains
        for (domain, uncertainty) in domains.into_iter().take(max_suggestions) {
            suggestions.push(ActionType::Epistemic {
                target_domain: domain,
                expected_information_gain: uncertainty,
            });
        }

        suggestions
    }

    /// Get a summary of the current state
    pub fn summary(&self) -> ActiveInferenceSummary {
        let (most_uncertain, uncertainty) = self.most_uncertain_domain();
        let most_surprising = self.most_surprising_domain();

        ActiveInferenceSummary {
            total_free_energy: self.total_free_energy_estimate(),
            average_surprise: self.stats.average_surprise,
            most_uncertain_domain: most_uncertain,
            uncertainty_level: uncertainty,
            most_surprising_domain: most_surprising.map(|(d, _)| d),
            surprise_level: most_surprising.map(|(_, s)| s).unwrap_or(0.0),
            curiosity_pressure: uncertainty > 0.3, // High uncertainty = should explore
            observations_total: self.stats.total_observations,
        }
    }
}

/// Summary of active inference state
#[derive(Debug, Clone)]
pub struct ActiveInferenceSummary {
    pub total_free_energy: f32,
    pub average_surprise: f32,
    pub most_uncertain_domain: PredictionDomain,
    pub uncertainty_level: f32,
    pub most_surprising_domain: Option<PredictionDomain>,
    pub surprise_level: f32,
    pub curiosity_pressure: bool,
    pub observations_total: u64,
}

impl std::fmt::Display for ActiveInferenceSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FE={:.2} | Surprise={:.2} | Uncertain={:?}({:.2}) | Curious={}",
            self.total_free_energy,
            self.average_surprise,
            self.most_uncertain_domain,
            self.uncertainty_level,
            if self.curiosity_pressure { "yes" } else { "no" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_error_creation() {
        let error = PredictionError::new(0.7, 0.8, 0.9, PredictionDomain::Coherence);

        assert!((error.error - 0.1).abs() < 0.001);
        assert!((error.weighted_error - 0.09).abs() < 0.001);
        assert!((error.surprise() - 0.09).abs() < 0.001);
        assert!(error.is_positive_surprise()); // Higher coherence = good
    }

    #[test]
    fn test_safety_surprise_inversion() {
        // For safety, higher observed threat is BAD (negative surprise)
        let error = PredictionError::new(0.1, 0.5, 0.9, PredictionDomain::Safety);

        assert!(!error.is_positive_surprise()); // Higher threat = bad
    }

    #[test]
    fn test_generative_model_prediction() {
        let model = GenerativeModel::new(PredictionDomain::Coherence);

        let prediction = model.predict();
        assert!((prediction - 0.7).abs() < 0.001); // Prior mean
    }

    #[test]
    fn test_generative_model_update() {
        let mut model = GenerativeModel::new(PredictionDomain::Coherence);

        // Observe consistently higher values
        for _ in 0..20 {
            model.update(0.9);
        }

        // Belief should move toward observations
        assert!(model.belief_mean > 0.7);
        // Variance should decrease
        assert!(model.belief_variance < model.prior_variance);
    }

    #[test]
    fn test_generative_model_free_energy() {
        let model = GenerativeModel::new(PredictionDomain::Coherence);

        // Initial free energy should be low (at prior)
        let fe = model.free_energy();
        assert!(fe < 1.0);
    }

    #[test]
    fn test_active_inference_engine_creation() {
        let engine = ActiveInferenceEngine::new();

        // Should have models for all domains
        assert_eq!(engine.models.len(), 8);
        assert!(engine.models.contains_key(&PredictionDomain::Coherence));
    }

    #[test]
    fn test_observe_updates_beliefs() {
        let mut engine = ActiveInferenceEngine::new();

        // Observe high coherence repeatedly
        for _ in 0..10 {
            let error = engine.observe(PredictionDomain::Coherence, 0.9);
            assert!(error.surprise() >= 0.0);
        }

        // Prediction should move toward observed
        let (prediction, _precision) = engine.predict(PredictionDomain::Coherence);
        assert!(prediction > 0.7);
    }

    #[test]
    fn test_most_uncertain_domain() {
        let engine = ActiveInferenceEngine::new();

        let (domain, uncertainty) = engine.most_uncertain_domain();

        // UserState and Social both have highest initial variance (0.2)
        // Either is acceptable as "most uncertain"
        assert!(
            domain == PredictionDomain::UserState || domain == PredictionDomain::Social,
            "Expected UserState or Social (both have highest variance), got {:?}",
            domain
        );
        assert!(uncertainty > 0.0);
    }

    #[test]
    fn test_expected_free_energy_epistemic() {
        let engine = ActiveInferenceEngine::new();

        let action = ActionType::Epistemic {
            target_domain: PredictionDomain::UserState,
            expected_information_gain: 0.5,
        };

        let efg = engine.expected_free_energy(&action);

        // Should be negative (good action reduces expected FE)
        assert!(efg < 0.0);
    }

    #[test]
    fn test_expected_free_energy_pragmatic() {
        let engine = ActiveInferenceEngine::new();

        let action = ActionType::Pragmatic {
            goal: "install nginx".to_string(),
            expected_utility: 0.8,
        };

        let efg = engine.expected_free_energy(&action);

        // Should be negative (good action)
        assert!(efg < 0.0);
    }

    #[test]
    fn test_suggest_action_prefers_lower_efg() {
        let mut engine = ActiveInferenceEngine::new();

        let candidates = vec![
            ActionType::Pragmatic {
                goal: "low value".to_string(),
                expected_utility: 0.1,
            },
            ActionType::Pragmatic {
                goal: "high value".to_string(),
                expected_utility: 0.9,
            },
        ];

        let suggested = engine.suggest_action(&candidates);

        assert!(suggested.is_some());
        match suggested.unwrap() {
            ActionType::Pragmatic { goal, .. } => {
                assert_eq!(goal, "high value");
            }
            _ => panic!("Expected pragmatic action"),
        }
    }

    #[test]
    fn test_curiosity_suggestions() {
        let engine = ActiveInferenceEngine::new();

        let suggestions = engine.curiosity_suggestions(3);

        assert_eq!(suggestions.len(), 3);
        for suggestion in suggestions {
            match suggestion {
                ActionType::Epistemic { .. } => {}
                _ => panic!("Expected epistemic actions"),
            }
        }
    }

    #[test]
    fn test_decay_reduces_free_energy() {
        let mut engine = ActiveInferenceEngine::new();

        // Accumulate some free energy
        engine.total_free_energy = 10.0;

        // Decay
        engine.decay(1.0);

        // Should be lower
        assert!(engine.total_free_energy < 10.0);
    }

    #[test]
    fn test_summary_generation() {
        let engine = ActiveInferenceEngine::new();

        let summary = engine.summary();

        assert!(summary.total_free_energy >= 0.0);
        assert!(summary.uncertainty_level >= 0.0);
    }

    #[test]
    fn test_precision_increases_with_observations() {
        let mut model = GenerativeModel::new(PredictionDomain::Coherence);

        let initial_precision = model.precision();

        // Make many observations
        for _ in 0..50 {
            model.update(0.75);
        }

        let final_precision = model.precision();

        // Precision should increase with experience
        assert!(final_precision > initial_precision);
    }

    #[test]
    fn test_prediction_errors_small_with_accurate_observations() {
        let mut model = GenerativeModel::new(PredictionDomain::Coherence);
        let target = model.prior_mean; // Save before updates

        // Make several predictions that match observations
        // Track prediction errors
        let mut errors: Vec<f32> = Vec::new();

        for _ in 0..20 {
            // Observe the prior mean (accurate prediction)
            let error = model.update(target);
            errors.push(error.weighted_error.abs());
        }

        // With consistent observations at prior_mean, errors should remain small
        // (The model should predict well when observations match expectations)
        let avg_error: f32 = errors.iter().sum::<f32>() / errors.len() as f32;
        assert!(
            avg_error < 1.0,
            "Average prediction errors should be small when observations match predictions: {}",
            avg_error
        );

        // Belief mean should stay close to target (not drift away)
        let drift = (model.belief_mean - target).abs();
        assert!(
            drift < 0.1,
            "Belief mean should stay close to target: drift={}",
            drift
        );
    }

    #[test]
    fn test_centering_action_efg() {
        let engine = ActiveInferenceEngine::new();

        let action = ActionType::Centering {
            target_coherence: 0.9,
        };

        let efg = engine.expected_free_energy(&action);

        // Should be negative (improvement)
        assert!(efg < 0.0);
    }
}
