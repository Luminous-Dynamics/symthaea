//! # Predictive Processing Router
//!
//! Implements Karl Friston's Predictive Processing framework where consciousness
//! emerges from hierarchical prediction error minimization. The brain is a
//! hypothesis-testing machine that continuously generates and updates predictions
//! about sensory input.
//!
//! ## Key Concepts
//!
//! 1. **Hierarchical generative models** - Multiple levels of abstraction
//! 2. **Prediction errors** - Mismatch between prediction and observation
//! 3. **Precision weighting** - Confidence-weighted error propagation
//! 4. **Top-down predictions** - Higher levels predict lower level activity
//! 5. **Bottom-up errors** - Prediction errors propagate upward
//!
//! This extends Active Inference by adding explicit hierarchical structure
//! and precision-weighted prediction error minimization across levels.

use std::collections::{HashMap, VecDeque};
use serde::{Deserialize, Serialize};

use super::{
    RoutingStrategy,
    ActiveInferenceRouter, ActiveInferenceConfig,
};
use crate::consciousness::recursive_improvement::LatentConsciousnessState;

// =============================================================================
// PREDICTIVE LEVEL
// =============================================================================

/// A single level in the predictive hierarchy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveLevel {
    /// Level index (0 = lowest/sensory)
    pub level: usize,
    /// Current representation at this level
    pub representation: Vec<f64>,
    /// Prediction of the level below
    pub prediction: Vec<f64>,
    /// Prediction error from level below
    pub prediction_error: Vec<f64>,
    /// Precision (confidence) at this level
    pub precision: f64,
    /// Temporal smoothing factor
    pub temporal_smoothing: f64,
    /// Learning rate for this level
    pub learning_rate: f64,
}

impl PredictiveLevel {
    pub fn new(level: usize, dim: usize) -> Self {
        // Higher levels have slower dynamics and lower learning rates
        let temporal_smoothing = 0.8 + 0.05 * level as f64;
        let learning_rate = 0.1 / (1.0 + 0.5 * level as f64);

        Self {
            level,
            representation: vec![0.5; dim],
            prediction: vec![0.5; dim],
            prediction_error: vec![0.0; dim],
            precision: 1.0,
            temporal_smoothing: temporal_smoothing.min(0.99),
            learning_rate,
        }
    }

    /// Generate prediction for level below
    pub fn predict(&self, weights: &[Vec<f64>]) -> Vec<f64> {
        let mut prediction = vec![0.0; self.prediction.len()];

        for (i, row) in weights.iter().enumerate() {
            if i < self.representation.len() {
                for (j, &w) in row.iter().enumerate() {
                    if j < prediction.len() {
                        prediction[j] += self.representation[i] * w;
                    }
                }
            }
        }

        // Sigmoid activation
        prediction.iter_mut().for_each(|p| *p = 1.0 / (1.0 + (-*p).exp()));
        prediction
    }

    /// Compute prediction error given observation
    pub fn compute_error(&mut self, observation: &[f64]) {
        for (i, (pred, obs)) in self.prediction.iter().zip(observation.iter()).enumerate() {
            if i < self.prediction_error.len() {
                self.prediction_error[i] = obs - pred;
            }
        }
    }

    /// Update representation based on error from below and prediction from above
    pub fn update(&mut self, error_from_below: Option<&[f64]>, prediction_from_above: Option<&[f64]>) {
        let dim = self.representation.len();

        for i in 0..dim {
            let mut delta = 0.0;

            // Bottom-up: reduce prediction error from level below
            if let Some(error) = error_from_below {
                if i < error.len() {
                    delta += self.learning_rate * error[i] * self.precision;
                }
            }

            // Top-down: conform to predictions from level above
            if let Some(pred) = prediction_from_above {
                if i < pred.len() {
                    delta += self.learning_rate * (pred[i] - self.representation[i]) * self.precision;
                }
            }

            // Update with temporal smoothing
            self.representation[i] = self.temporal_smoothing * self.representation[i]
                + (1.0 - self.temporal_smoothing) * (self.representation[i] + delta);

            // Clamp to valid range
            self.representation[i] = self.representation[i].max(0.0).min(1.0);
        }
    }

    /// Update precision based on prediction error
    pub fn update_precision(&mut self) {
        let error_magnitude: f64 = self.prediction_error.iter().map(|e| e * e).sum::<f64>().sqrt();

        // Precision inversely related to error magnitude
        // Higher error = lower precision (less confidence in predictions)
        let new_precision = 1.0 / (1.0 + error_magnitude);

        // Slow adaptation of precision
        self.precision = 0.9 * self.precision + 0.1 * new_precision;
    }

    /// Free energy at this level
    pub fn free_energy(&self) -> f64 {
        let prediction_error_term: f64 = self.prediction_error
            .iter()
            .map(|e| 0.5 * self.precision * e * e)
            .sum();

        // Complexity term (deviation from prior)
        let complexity: f64 = self.representation
            .iter()
            .map(|r| (r - 0.5).powi(2))
            .sum::<f64>()
            * 0.1;

        prediction_error_term + complexity
    }
}

// =============================================================================
// HIERARCHICAL WEIGHTS
// =============================================================================

/// Hierarchical weights between levels
#[derive(Debug, Clone)]
pub struct HierarchicalWeights {
    /// Weights from level i to level i-1 (top-down predictions)
    pub top_down: Vec<Vec<Vec<f64>>>,
    /// Weights from level i-1 to level i (bottom-up errors)
    pub bottom_up: Vec<Vec<Vec<f64>>>,
}

impl HierarchicalWeights {
    pub fn new(level_dims: &[usize]) -> Self {
        let mut top_down = Vec::new();
        let mut bottom_up = Vec::new();

        for i in 1..level_dims.len() {
            let higher_dim = level_dims[i];
            let lower_dim = level_dims[i - 1];

            // Top-down: higher -> lower
            let td: Vec<Vec<f64>> = (0..higher_dim)
                .map(|_| {
                    (0..lower_dim)
                        .map(|_| (rand::random::<f64>() - 0.5) * 0.2)
                        .collect()
                })
                .collect();
            top_down.push(td);

            // Bottom-up: lower -> higher
            let bu: Vec<Vec<f64>> = (0..lower_dim)
                .map(|_| {
                    (0..higher_dim)
                        .map(|_| (rand::random::<f64>() - 0.5) * 0.2)
                        .collect()
                })
                .collect();
            bottom_up.push(bu);
        }

        Self { top_down, bottom_up }
    }

    /// Learn weights to reduce prediction error
    pub fn learn(&mut self, levels: &[PredictiveLevel], learning_rate: f64) {
        for i in 0..self.top_down.len() {
            let higher_level = &levels[i + 1];
            let lower_level = &levels[i];

            // Update top-down weights
            for j in 0..self.top_down[i].len() {
                for k in 0..self.top_down[i][j].len() {
                    if j < higher_level.representation.len() && k < lower_level.prediction_error.len() {
                        let delta = learning_rate
                            * higher_level.representation[j]
                            * lower_level.prediction_error[k]
                            * higher_level.precision;
                        self.top_down[i][j][k] += delta;
                    }
                }
            }

            // Update bottom-up weights
            for j in 0..self.bottom_up[i].len() {
                for k in 0..self.bottom_up[i][j].len() {
                    if j < lower_level.prediction_error.len() && k < higher_level.representation.len() {
                        let delta = learning_rate
                            * lower_level.prediction_error[j]
                            * (1.0 - higher_level.representation[k])
                            * lower_level.precision;
                        self.bottom_up[i][j][k] += delta;
                    }
                }
            }
        }
    }
}

// =============================================================================
// CONFIGURATION AND TYPES
// =============================================================================

/// Configuration for predictive processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveProcessingConfig {
    /// Number of hierarchical levels
    pub num_levels: usize,
    /// Dimension at each level
    pub level_dims: Vec<usize>,
    /// Global precision gain
    pub precision_gain: f64,
    /// How much to weight prediction errors
    pub error_weight: f64,
    /// Temperature for strategy selection
    pub temperature: f64,
    /// Enable precision weighting
    pub enable_precision_weighting: bool,
}

impl Default for PredictiveProcessingConfig {
    fn default() -> Self {
        Self {
            num_levels: 4,
            level_dims: vec![8, 6, 4, 2], // Pyramid structure
            precision_gain: 1.0,
            error_weight: 1.0,
            temperature: 1.0,
            enable_precision_weighting: true,
        }
    }
}

/// Statistics for predictive processing
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PredictiveProcessingStats {
    pub updates: usize,
    pub total_free_energy: f64,
    pub average_precision: f64,
    pub prediction_accuracy: f64,
    pub level_errors: Vec<f64>,
}

/// Decision from predictive processing router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveProcessingDecision {
    /// Selected strategy
    pub strategy: RoutingStrategy,
    /// Precision-weighted confidence
    pub confidence: f64,
    /// Total free energy
    pub free_energy: f64,
    /// Prediction errors at each level
    pub prediction_errors: Vec<f64>,
    /// Precision at each level
    pub level_precisions: Vec<f64>,
    /// Top level representation
    pub top_representation: Vec<f64>,
    /// Probability of selected strategy
    pub selected_probability: f64,
}

// =============================================================================
// PREDICTIVE PROCESSING ROUTER
// =============================================================================

/// Predictive Processing Router
/// Routes consciousness using hierarchical prediction error minimization
pub struct PredictiveProcessingRouter {
    /// Active inference router (for base routing)
    active_inference_router: ActiveInferenceRouter,
    /// Hierarchical levels
    levels: Vec<PredictiveLevel>,
    /// Inter-level weights
    weights: HierarchicalWeights,
    /// Configuration
    config: PredictiveProcessingConfig,
    /// Statistics
    stats: PredictiveProcessingStats,
    /// Strategy predictions at top level
    strategy_predictions: HashMap<RoutingStrategy, f64>,
    /// History of prediction errors
    error_history: VecDeque<f64>,
}

impl PredictiveProcessingRouter {
    pub fn new(config: PredictiveProcessingConfig) -> Self {
        let levels: Vec<PredictiveLevel> = config.level_dims
            .iter()
            .enumerate()
            .map(|(i, &dim)| PredictiveLevel::new(i, dim))
            .collect();

        let weights = HierarchicalWeights::new(&config.level_dims);

        let mut strategy_predictions = HashMap::new();
        for i in 0..7 {
            let strategy = ActiveInferenceRouter::index_to_strategy(i);
            strategy_predictions.insert(strategy, 1.0 / 7.0);
        }

        Self {
            active_inference_router: ActiveInferenceRouter::new(ActiveInferenceConfig::default()),
            levels,
            weights,
            config,
            stats: PredictiveProcessingStats::default(),
            strategy_predictions,
            error_history: VecDeque::with_capacity(100),
        }
    }

    /// Encode consciousness state as sensory input (lowest level)
    pub fn encode_state(&self, state: &LatentConsciousnessState) -> Vec<f64> {
        let base = vec![state.phi, state.integration, state.coherence, state.attention];

        // Expand to match lowest level dimension
        let dim = self.config.level_dims[0];
        let mut encoded = Vec::with_capacity(dim);

        for i in 0..dim {
            if i < base.len() {
                encoded.push(base[i]);
            } else {
                // Derived features
                let idx = i % base.len();
                let next_idx = (i + 1) % base.len();
                encoded.push((base[idx] * base[next_idx]).sqrt());
            }
        }

        encoded
    }

    /// Run one step of predictive processing
    fn process_step(&mut self, observation: &[f64]) {
        let num_levels = self.levels.len();

        // 1. Bottom-up pass: compute prediction errors
        self.levels[0].compute_error(observation);

        for i in 1..num_levels {
            // Get prediction from level above
            let prediction = if i + 1 < num_levels {
                Some(self.levels[i + 1].predict(&self.weights.top_down[i]))
            } else {
                None
            };

            // Compute error at this level
            if let Some(pred) = prediction {
                self.levels[i].compute_error(&pred);
            }
        }

        // 2. Top-down pass: generate predictions and update representations
        for i in (0..num_levels).rev() {
            // Get prediction from above
            let prediction_from_above = if i + 1 < num_levels {
                Some(self.levels[i + 1].predict(&self.weights.top_down[i]))
            } else {
                None
            };

            // Get error from below (transformed through bottom-up weights)
            let error_from_below = if i > 0 {
                let lower_error = &self.levels[i - 1].prediction_error;
                let mut transformed = vec![0.0; self.levels[i].representation.len()];

                for (j, row) in self.weights.bottom_up[i - 1].iter().enumerate() {
                    if j < lower_error.len() {
                        for (k, &w) in row.iter().enumerate() {
                            if k < transformed.len() {
                                transformed[k] += lower_error[j] * w;
                            }
                        }
                    }
                }
                Some(transformed)
            } else {
                None
            };

            // Update level
            self.levels[i].update(
                error_from_below.as_deref(),
                prediction_from_above.as_deref()
            );

            // Update precision
            if self.config.enable_precision_weighting {
                self.levels[i].update_precision();
            }

            // Generate predictions for level below
            if i > 0 {
                self.levels[i].prediction = self.levels[i].predict(&self.weights.top_down[i - 1]);
            }
        }

        // 3. Learn weights
        self.weights.learn(&self.levels, 0.01);

        // 4. Update statistics
        self.stats.updates += 1;
        self.stats.total_free_energy = self.total_free_energy();
        self.stats.average_precision = self.levels.iter().map(|l| l.precision).sum::<f64>()
            / self.levels.len() as f64;
        self.stats.level_errors = self.levels.iter()
            .map(|l| l.prediction_error.iter().map(|e| e.abs()).sum::<f64>())
            .collect();

        // Track error history
        let total_error: f64 = self.stats.level_errors.iter().sum();
        if self.error_history.len() >= 100 {
            self.error_history.pop_front();
        }
        self.error_history.push_back(total_error);
    }

    /// Total free energy across all levels
    pub fn total_free_energy(&self) -> f64 {
        self.levels.iter().map(|l| l.free_energy()).sum()
    }

    /// Route using predictive processing
    pub fn route(&mut self, target: &LatentConsciousnessState) -> PredictiveProcessingDecision {
        // Encode and process
        let observation = self.encode_state(target);
        self.process_step(&observation);

        // Get base routing from active inference
        let ai_decision = self.active_inference_router.route(target);

        // Use highest level representation to modulate strategy selection
        let top_level = &self.levels[self.levels.len() - 1];

        // Map top level to strategy predictions
        let mut strategy_probs = HashMap::new();
        let strategies = [
            RoutingStrategy::FullDeliberation,
            RoutingStrategy::StandardProcessing,
            RoutingStrategy::HeuristicGuided,
            RoutingStrategy::FastPatterns,
            RoutingStrategy::Reflexive,
            RoutingStrategy::Ensemble,
            RoutingStrategy::Preparatory,
        ];

        let mut total: f64 = 0.0;
        for (i, &strategy) in strategies.iter().enumerate() {
            // Combine AI decision with top-level predictions
            let ai_weight = if ai_decision.strategy == strategy { ai_decision.confidence } else { 0.1 };

            // Use top-level representation as prior
            let level_idx = i % top_level.representation.len();
            let level_weight = top_level.representation[level_idx] * top_level.precision;

            let combined = ai_weight * 0.6 + level_weight * 0.4;
            strategy_probs.insert(strategy, combined);
            total += combined;
        }

        // Normalize
        for prob in strategy_probs.values_mut() {
            *prob /= total.max(0.001);
        }

        // Apply softmax with temperature
        let max_logit = strategy_probs.values().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut exp_sum = 0.0;
        for prob in strategy_probs.values_mut() {
            *prob = ((*prob - max_logit) / self.config.temperature).exp();
            exp_sum += *prob;
        }
        for prob in strategy_probs.values_mut() {
            *prob /= exp_sum.max(0.001);
        }

        // Select strategy
        let (selected_strategy, selected_prob) = strategy_probs
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(s, p)| (*s, *p))
            .unwrap_or((RoutingStrategy::StandardProcessing, 0.5));

        // Calculate precision-weighted confidence
        let confidence = selected_prob * self.stats.average_precision;

        // Update strategy predictions for next round
        self.strategy_predictions = strategy_probs.clone();

        PredictiveProcessingDecision {
            strategy: selected_strategy,
            confidence,
            free_energy: self.stats.total_free_energy,
            prediction_errors: self.stats.level_errors.clone(),
            level_precisions: self.levels.iter().map(|l| l.precision).collect(),
            top_representation: top_level.representation.clone(),
            selected_probability: selected_prob,
        }
    }

    /// Get current prediction accuracy (how well predictions match observations)
    pub fn prediction_accuracy(&self) -> f64 {
        if self.error_history.len() < 2 {
            return 0.5;
        }

        let recent_error: f64 = self.error_history.iter().rev().take(10).sum::<f64>() / 10.0;
        let max_error = self.config.level_dims.iter().sum::<usize>() as f64;

        1.0 - (recent_error / max_error.max(1.0)).min(1.0)
    }

    /// Reset the router
    pub fn reset(&mut self) {
        for level in &mut self.levels {
            for v in &mut level.representation {
                *v = 0.5;
            }
            for v in &mut level.prediction {
                *v = 0.5;
            }
            for v in &mut level.prediction_error {
                *v = 0.0;
            }
            level.precision = 1.0;
        }

        self.weights = HierarchicalWeights::new(&self.config.level_dims);
        self.stats = PredictiveProcessingStats::default();
        self.error_history.clear();
        self.active_inference_router.reset();
    }

    /// Get statistics
    pub fn stats(&self) -> &PredictiveProcessingStats {
        &self.stats
    }

    /// Get level representations
    pub fn level_representations(&self) -> Vec<Vec<f64>> {
        self.levels.iter().map(|l| l.representation.clone()).collect()
    }

    /// Summary of current state
    pub fn summary(&self) -> String {
        format!(
            "PredictiveProcessingRouter: {} levels, FE={:.4}, accuracy={:.2}%, avg_precision={:.3}",
            self.levels.len(),
            self.stats.total_free_energy,
            self.prediction_accuracy() * 100.0,
            self.stats.average_precision
        )
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictive_level_creation() {
        let level = PredictiveLevel::new(2, 8);

        assert_eq!(level.level, 2);
        assert_eq!(level.representation.len(), 8);
        assert_eq!(level.prediction.len(), 8);
        assert!(level.precision > 0.0);
    }

    #[test]
    fn test_predictive_level_compute_error() {
        let mut level = PredictiveLevel::new(0, 4);
        level.prediction = vec![0.5, 0.5, 0.5, 0.5];

        level.compute_error(&[0.7, 0.3, 0.6, 0.4]);

        assert!((level.prediction_error[0] - 0.2).abs() < 0.001);
        assert!((level.prediction_error[1] - (-0.2)).abs() < 0.001);
    }

    #[test]
    fn test_predictive_level_free_energy() {
        let mut level = PredictiveLevel::new(0, 4);
        level.prediction_error = vec![0.1, -0.1, 0.05, -0.05];

        let fe = level.free_energy();
        assert!(fe > 0.0);
        assert!(fe.is_finite());
    }

    #[test]
    fn test_predictive_level_update_precision() {
        let mut level = PredictiveLevel::new(0, 4);
        let initial_precision = level.precision;

        level.prediction_error = vec![0.5, 0.5, 0.5, 0.5]; // High error
        level.update_precision();

        // Precision should decrease with high error
        assert!(level.precision < initial_precision);
    }

    #[test]
    fn test_predictive_processing_router_creation() {
        let router = PredictiveProcessingRouter::new(PredictiveProcessingConfig::default());

        assert_eq!(router.levels.len(), 4);
        assert!(router.stats.updates == 0);
    }

    #[test]
    fn test_predictive_processing_encode_state() {
        let router = PredictiveProcessingRouter::new(PredictiveProcessingConfig::default());
        let state = LatentConsciousnessState::from_observables(0.5, 0.6, 0.7, 0.8);

        let encoded = router.encode_state(&state);
        assert_eq!(encoded.len(), 8); // Default lowest level dim
    }

    #[test]
    fn test_predictive_processing_route() {
        let mut router = PredictiveProcessingRouter::new(PredictiveProcessingConfig::default());
        let state = LatentConsciousnessState::from_observables(0.5, 0.6, 0.7, 0.8);

        let decision = router.route(&state);

        assert!(decision.confidence > 0.0);
        assert!(decision.free_energy >= 0.0);
        assert_eq!(decision.prediction_errors.len(), 4);
    }

    #[test]
    fn test_predictive_processing_learns() {
        let mut router = PredictiveProcessingRouter::new(PredictiveProcessingConfig::default());

        // Run multiple steps
        for i in 0..20 {
            let phi = 0.5 + 0.3 * (i as f64 * 0.1).sin();
            let state = LatentConsciousnessState::from_observables(phi, 0.6, 0.7, 0.8);
            let _ = router.route(&state);
        }

        assert_eq!(router.stats.updates, 20);
        assert!(router.error_history.len() > 0);
    }

    #[test]
    fn test_predictive_processing_prediction_accuracy() {
        let mut router = PredictiveProcessingRouter::new(PredictiveProcessingConfig::default());

        // Consistent states should lead to good accuracy
        for _ in 0..30 {
            let state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
            let _ = router.route(&state);
        }

        let accuracy = router.prediction_accuracy();
        assert!(accuracy >= 0.0);
        assert!(accuracy <= 1.0);
    }

    #[test]
    fn test_predictive_processing_reset() {
        let mut router = PredictiveProcessingRouter::new(PredictiveProcessingConfig::default());

        // Make some decisions
        let state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);
        for _ in 0..5 {
            let _ = router.route(&state);
        }

        router.reset();

        assert_eq!(router.stats.updates, 0);
        assert!(router.error_history.is_empty());
    }

    #[test]
    fn test_predictive_processing_summary() {
        let router = PredictiveProcessingRouter::new(PredictiveProcessingConfig::default());
        let summary = router.summary();

        assert!(summary.contains("PredictiveProcessingRouter"));
        assert!(summary.contains("levels"));
    }

    #[test]
    fn test_hierarchical_weights_creation() {
        let level_dims = vec![8, 6, 4, 2];
        let weights = HierarchicalWeights::new(&level_dims);

        assert_eq!(weights.top_down.len(), 3); // 3 transitions
        assert_eq!(weights.bottom_up.len(), 3);
    }

    #[test]
    fn test_hierarchical_weights_dimensions() {
        let level_dims = vec![8, 6, 4, 2];
        let weights = HierarchicalWeights::new(&level_dims);

        // Top-down from level 1 (dim 6) to level 0 (dim 8)
        assert_eq!(weights.top_down[0].len(), 6);
        assert_eq!(weights.top_down[0][0].len(), 8);
    }

    #[test]
    fn test_predictive_processing_level_representations() {
        let mut router = PredictiveProcessingRouter::new(PredictiveProcessingConfig::default());
        let state = LatentConsciousnessState::from_observables(0.5, 0.6, 0.7, 0.8);

        let _ = router.route(&state);
        let reps = router.level_representations();

        assert_eq!(reps.len(), 4);
        assert_eq!(reps[0].len(), 8);
        assert_eq!(reps[1].len(), 6);
        assert_eq!(reps[2].len(), 4);
        assert_eq!(reps[3].len(), 2);
    }

    #[test]
    fn test_predictive_processing_decision_structure() {
        let mut router = PredictiveProcessingRouter::new(PredictiveProcessingConfig::default());
        let state = LatentConsciousnessState::from_observables(0.5, 0.5, 0.5, 0.5);

        let decision = router.route(&state);

        assert!(decision.selected_probability > 0.0);
        assert!(decision.selected_probability <= 1.0);
        assert_eq!(decision.level_precisions.len(), 4);
        assert_eq!(decision.top_representation.len(), 2);
    }

    #[test]
    fn test_predictive_config_default() {
        let config = PredictiveProcessingConfig::default();

        assert_eq!(config.num_levels, 4);
        assert_eq!(config.level_dims.len(), 4);
        assert!(config.enable_precision_weighting);
    }

    #[test]
    fn test_predictive_level_predict() {
        let level = PredictiveLevel::new(1, 4);
        let weights: Vec<Vec<f64>> = vec![
            vec![0.5, 0.3, 0.2, 0.1],
            vec![0.1, 0.5, 0.3, 0.2],
            vec![0.2, 0.1, 0.5, 0.3],
            vec![0.3, 0.2, 0.1, 0.5],
        ];

        let prediction = level.predict(&weights);

        assert_eq!(prediction.len(), 4);
        for p in &prediction {
            assert!(*p >= 0.0);
            assert!(*p <= 1.0);
        }
    }
}
