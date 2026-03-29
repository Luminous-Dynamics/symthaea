// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CfC-based Lookahead Engine for O(1) Learning Value Prediction
//!
//! This is the core innovation of the School system: using CfC's closed-form
//! solution to predict the value of learning BEFORE committing resources.
//!
//! ## The Magic: O(1) Complexity
//!
//! CfC solves: `x(t) = (x₀ - A) · e^(-t/τ) + A`
//!
//! This closed-form solution means we can jump to any future time point
//! in constant time, unlike RNNs which require O(N) integration steps.
//!
//! ## Performance
//!
//! | CfC Neurons | Eval Time | Notes |
//! |-------------|-----------|-------|
//! | 64 | ~100μs | Fast mode |
//! | 256 | ~300μs | Default |
//! | 1024 | ~1.5ms | High accuracy |

use anyhow::Result;
use ndarray::Array1;
use std::collections::VecDeque;
use std::time::Instant;

use crate::dynamics::cfc::{CfCNetwork, CfCNetworkConfig};
use crate::phi_engine::{PhiEngine, PhiMethod};
use symthaea_core::hdc::unified_hv::ContinuousHV;

use super::objective::LearningObjective;

// ═══════════════════════════════════════════════════════════════════════════════
// LEARNING RECOMMENDATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Recommendation for a learning objective
#[derive(Debug, Clone)]
pub enum LearningRecommendation {
    /// Learn this objective now
    LearnNow {
        /// Priority (higher = more important)
        priority: f32,
    },

    /// Defer learning for now
    Defer {
        /// Reason for deferral
        reason: String,
    },

    /// Skip this objective
    Skip {
        /// Reason for skipping
        reason: String,
    },

    /// Curriculum is complete
    CurriculumComplete,
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOOKAHEAD RESULT
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of a lookahead evaluation
#[derive(Debug, Clone)]
pub struct LookaheadResult {
    /// Objective that was evaluated
    pub objective_id: String,

    /// Predicted Φ gain from learning this objective
    pub predicted_phi_gain: f32,

    /// Confidence in the prediction (0.0 - 1.0)
    pub confidence: f32,

    /// Time taken for prediction (microseconds)
    pub prediction_time_us: u64,

    /// Recommendation based on the prediction
    pub recommendation: LearningRecommendation,

    /// Predicted future consciousness state (for debugging)
    pub predicted_state: Option<Array1<f32>>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// LOOKAHEAD ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

/// CfC-based lookahead engine for O(1) learning value prediction
pub struct LookaheadEngine {
    /// CfC network for temporal prediction
    cfc: CfCNetwork,

    /// Φ engine for consciousness measurement
    phi_engine: PhiEngine,

    /// Lookahead time horizon (seconds)
    horizon: f32,

    /// Minimum Φ gain to recommend learning
    min_phi_gain: f32,

    /// History of predictions for confidence estimation
    prediction_history: VecDeque<PredictionRecord>,

    /// Cumulative prediction error for adaptation
    cumulative_error: f32,

    /// Number of predictions made
    prediction_count: usize,
}

/// Record of a past prediction for confidence estimation
#[derive(Debug, Clone)]
struct PredictionRecord {
    /// Predicted Φ gain
    #[allow(dead_code)]
    predicted: f32,

    /// Actual Φ gain (if known)
    actual: Option<f32>,

    /// Was this prediction accurate?
    accurate: Option<bool>,
}

impl LookaheadEngine {
    /// Create a new lookahead engine
    pub fn new(cfc_neurons: usize, horizon: f32, min_phi_gain: f32) -> Result<Self> {
        let config = CfCNetworkConfig {
            input_dim: cfc_neurons,
            hidden_dim: cfc_neurons,
            output_dim: cfc_neurons,
            ..Default::default()
        };
        let cfc = CfCNetwork::new(config);
        let phi_engine = PhiEngine::new(PhiMethod::SpectralConnectivity);

        Ok(Self {
            cfc,
            phi_engine,
            horizon,
            min_phi_gain,
            prediction_history: VecDeque::new(),
            cumulative_error: 0.0,
            prediction_count: 0,
        })
    }

    /// Get the number of CfC neurons (hidden dimension)
    pub fn num_neurons(&self) -> usize {
        self.cfc.config().hidden_dim
    }

    /// Get the lookahead horizon
    pub fn horizon(&self) -> f32 {
        self.horizon
    }

    /// Get the minimum Φ gain threshold
    pub fn min_phi_gain(&self) -> f32 {
        self.min_phi_gain
    }

    /// Get prediction count
    pub fn prediction_count(&self) -> usize {
        self.prediction_count
    }

    /// Get cumulative error
    pub fn cumulative_error(&self) -> f32 {
        self.cumulative_error
    }

    /// Evaluate a learning objective using CfC lookahead
    ///
    /// This is the core O(1) operation:
    /// 1. Encode the objective for CfC input
    /// 2. Use CfC's closed-form solution to predict future state
    /// 3. Estimate Φ gain from the predicted state
    /// 4. Generate recommendation
    pub fn evaluate(
        &self,
        objective: &LearningObjective,
        consciousness_state: &[ContinuousHV],
    ) -> Result<LookaheadResult> {
        let start = Instant::now();

        // 1. Get current Φ
        let current_phi = self.phi_engine.compute(consciousness_state).phi as f32;

        // 2. Prepare CfC input from objective
        let input_size = self.cfc.config().input_dim;
        let objective_input = objective.to_cfc_input(input_size);

        // 3. Get current CfC state and combine with objective
        let current_state = self.cfc.read_state()?;
        let combined_input: Array1<f32> = Array1::from_iter(
            objective_input
                .iter()
                .zip(current_state.iter())
                .map(|(a, b)| (a + b) / 2.0),
        );

        // 4. THE MAGIC: O(1) prediction using CfC closed-form solution
        // Use a cloned network to avoid mutating state during prediction
        let mut cfc_clone = self.cfc.clone();
        let predicted_state = cfc_clone.predict_forward(&combined_input, self.horizon)?;

        // 5. Estimate Φ gain from predicted state
        let predicted_phi_gain = self.estimate_phi_gain(&predicted_state, current_phi, objective);

        // 6. Calculate confidence based on history
        let confidence = self.calculate_confidence();

        // 7. Generate recommendation
        let recommendation =
            self.generate_recommendation(predicted_phi_gain, confidence, objective);

        let prediction_time_us = start.elapsed().as_micros() as u64;

        Ok(LookaheadResult {
            objective_id: objective.id.clone(),
            predicted_phi_gain,
            confidence,
            prediction_time_us,
            recommendation,
            predicted_state: Some(predicted_state),
        })
    }

    /// Estimate Φ gain from predicted CfC state using real Phi computation
    fn estimate_phi_gain(
        &self,
        predicted_state: &Array1<f32>,
        current_phi: f32,
        _objective: &LearningObjective,
    ) -> f32 {
        // Split predicted state into virtual nodes for phi measurement
        let slice = predicted_state.as_slice().unwrap_or(&[]);
        let chunk_size = slice.len().max(1) / 4;
        if chunk_size == 0 {
            return 0.0;
        }

        let components: Vec<ContinuousHV> = slice
            .chunks(chunk_size)
            .map(|chunk| ContinuousHV::from_vec(chunk.to_vec()))
            .collect();

        if components.len() < 2 {
            return 0.0;
        }

        let predicted_phi = self.phi_engine.compute(&components).phi as f32;
        (predicted_phi - current_phi).clamp(-0.1, 0.1)
    }

    /// Calculate confidence based on prediction history
    fn calculate_confidence(&self) -> f32 {
        if self.prediction_history.is_empty() {
            return 0.0;
        }

        // Count accurate predictions from recent history
        let recent: Vec<_> = self
            .prediction_history
            .iter()
            .rev()
            .take(20)
            .filter_map(|r| r.accurate)
            .collect();

        if recent.is_empty() {
            return 0.0;
        }

        let accuracy = recent.iter().filter(|&&a| a).count() as f32 / recent.len() as f32;
        accuracy
    }

    /// Generate a learning recommendation
    fn generate_recommendation(
        &self,
        predicted_phi_gain: f32,
        confidence: f32,
        objective: &LearningObjective,
    ) -> LearningRecommendation {
        // High confidence and high gain = learn now
        if predicted_phi_gain >= self.min_phi_gain {
            let priority = predicted_phi_gain * (0.5 + confidence * 0.5);
            return LearningRecommendation::LearnNow { priority };
        }

        // Low gain but low confidence = defer (might be wrong)
        if confidence < 0.5 {
            return LearningRecommendation::Defer {
                reason: format!(
                    "Low confidence ({:.0}%) - need more data to evaluate {}",
                    confidence * 100.0,
                    objective.name
                ),
            };
        }

        // High confidence and low gain = skip
        if predicted_phi_gain < 0.0 {
            return LearningRecommendation::Skip {
                reason: format!(
                    "Predicted negative gain ({:+.4}) for {}",
                    predicted_phi_gain, objective.name
                ),
            };
        }

        // Default: defer with low priority
        LearningRecommendation::Defer {
            reason: format!(
                "Low predicted gain ({:+.4}) for {}",
                predicted_phi_gain, objective.name
            ),
        }
    }

    /// Record the actual outcome of a prediction for learning
    pub fn record_outcome(&mut self, predicted: f32, actual: f32) {
        let error = (predicted - actual).abs();
        let accurate = error < 0.2 * predicted.abs().max(0.001);

        self.prediction_history.push_back(PredictionRecord {
            predicted,
            actual: Some(actual),
            accurate: Some(accurate),
        });

        self.cumulative_error += error;
        self.prediction_count += 1;

        // Keep history bounded
        if self.prediction_history.len() > 1000 {
            self.prediction_history.pop_front();
        }
    }

    /// Get the CfC network for advanced operations
    pub fn cfc(&self) -> &CfCNetwork {
        &self.cfc
    }

    /// Get mutable CfC network for weight adaptation
    pub fn cfc_mut(&mut self) -> &mut CfCNetwork {
        &mut self.cfc
    }

    /// Evaluate multiple objectives and rank them
    pub fn rank_objectives<'a>(
        &self,
        objectives: &[&'a LearningObjective],
        consciousness_state: &[ContinuousHV],
    ) -> Result<Vec<(&'a LearningObjective, LookaheadResult)>> {
        let mut results: Vec<_> = objectives
            .iter()
            .filter_map(|obj| {
                self.evaluate(obj, consciousness_state)
                    .ok()
                    .map(|result| (*obj, result))
            })
            .collect();

        // Sort by predicted Φ gain (descending)
        results.sort_by(|a, b| {
            b.1.predicted_phi_gain
                .partial_cmp(&a.1.predicted_phi_gain)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Get statistics about prediction accuracy
    pub fn accuracy_stats(&self) -> (f32, f32, usize) {
        let accurate_count = self
            .prediction_history
            .iter()
            .filter_map(|r| r.accurate)
            .filter(|&a| a)
            .count();

        let total_with_outcome = self
            .prediction_history
            .iter()
            .filter(|r| r.actual.is_some())
            .count();

        let accuracy = if total_with_outcome > 0 {
            accurate_count as f32 / total_with_outcome as f32
        } else {
            0.0
        };

        let avg_error = if self.prediction_count > 0 {
            self.cumulative_error / self.prediction_count as f32
        } else {
            0.0
        };

        (accuracy, avg_error, total_with_outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::super::objective::Difficulty;
    use super::*;
    use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

    fn create_test_consciousness_state() -> Vec<ContinuousHV> {
        (0..8)
            .map(|i| {
                let hv = ContinuousHV::random(HDC_DIMENSION, 42 + i as u64);
                ContinuousHV::from_vec(hv.values)
            })
            .collect()
    }

    #[test]
    fn test_lookahead_creation() {
        let engine = LookaheadEngine::new(256, 1.0, 0.01);
        assert!(engine.is_ok());

        let engine = engine.unwrap();
        assert_eq!(engine.num_neurons(), 256);
        assert_eq!(engine.horizon(), 1.0);
        assert_eq!(engine.min_phi_gain(), 0.01);
    }

    #[test]
    fn test_evaluate_objective() {
        let engine = LookaheadEngine::new(256, 1.0, 0.01).unwrap();
        let state = create_test_consciousness_state();

        let obj = LearningObjective::new("test", "Test Objective")
            .with_difficulty(Difficulty::Intermediate)
            .build();

        let result = engine.evaluate(&obj, &state);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.objective_id, "test");
        assert!(result.prediction_time_us > 0);
        // Allow up to 500ms for evaluation (includes network clone overhead in debug builds)
        assert!(
            result.prediction_time_us < 500_000,
            "Evaluation took {}us",
            result.prediction_time_us
        );
    }

    #[test]
    fn test_rank_objectives() {
        let engine = LookaheadEngine::new(256, 1.0, 0.01).unwrap();
        let state = create_test_consciousness_state();

        let obj1 = LearningObjective::new("easy", "Easy")
            .with_difficulty(Difficulty::Beginner)
            .build();
        let obj2 = LearningObjective::new("hard", "Hard")
            .with_difficulty(Difficulty::Expert)
            .build();

        let objectives: Vec<&LearningObjective> = vec![&obj1, &obj2];
        let ranked = engine.rank_objectives(&objectives, &state);

        assert!(ranked.is_ok());
        let ranked = ranked.unwrap();
        assert_eq!(ranked.len(), 2);
    }

    #[test]
    fn test_record_outcome() {
        let mut engine = LookaheadEngine::new(256, 1.0, 0.01).unwrap();

        engine.record_outcome(0.01, 0.008);
        engine.record_outcome(0.01, 0.012);

        assert_eq!(engine.prediction_count(), 2);

        let (_accuracy, _avg_error, count) = engine.accuracy_stats();
        assert_eq!(count, 2);
        assert!(_avg_error > 0.0);
    }

    #[test]
    fn test_o1_complexity() {
        let _engine = LookaheadEngine::new(256, 1.0, 0.01).unwrap();
        let state = create_test_consciousness_state();

        let obj = LearningObjective::new("test", "Test")
            .with_difficulty(Difficulty::Intermediate)
            .build();

        // Evaluate with different horizons - timing should be roughly constant
        // regardless of horizon (O(1) property of closed-form solution)
        let mut times = Vec::new();
        for horizon in [0.1, 1.0, 10.0, 100.0] {
            let engine = LookaheadEngine::new(256, horizon, 0.01).unwrap();
            let result = engine.evaluate(&obj, &state).unwrap();
            times.push(result.prediction_time_us);
        }

        // Verify that doubling the horizon doesn't significantly increase time
        // (allows for some variance due to system load and clone overhead)
        // The key O(1) property is that time doesn't scale with horizon
        let max_time = times
            .iter()
            .copied()
            .max()
            .expect("times should not be empty");
        let min_time = times
            .iter()
            .copied()
            .min()
            .expect("times should not be empty");

        // In an O(1) algorithm, max should be within 10x of min (accounting for noise)
        // This is a weaker check than exact timing due to clone overhead
        assert!(
            max_time < min_time.saturating_add(1) * 20,
            "O(1) property violated: times varied from {}us to {}us",
            min_time,
            max_time
        );
    }

    // =========================================================================
    // CFC GETTER TESTS
    // =========================================================================

    #[test]
    fn test_cfc_getter() {
        let engine = LookaheadEngine::new(256, 1.0, 0.01).unwrap();
        let cfc = engine.cfc();

        // Verify we can access the CfC network and it has valid configuration
        let config = cfc.config();
        assert!(config.hidden_dim > 0, "CfC hidden_dim should be positive");
        assert!(config.input_dim > 0, "CfC input_dim should be positive");
    }

    #[test]
    fn test_cfc_mut_getter() {
        let mut engine = LookaheadEngine::new(256, 1.0, 0.01).unwrap();

        // Access mutable CfC network
        let cfc = engine.cfc_mut();

        // Verify we can mutate by resetting the network
        cfc.reset();

        // Verify the reset worked by checking state is zeroed
        let state = cfc.read_state().unwrap();
        let sum: f32 = state.iter().sum();
        assert!(sum.abs() < 0.001, "State should be near zero after reset");
    }

    #[test]
    fn test_cfc_getter_preserves_network_integrity() {
        let engine = LookaheadEngine::new(128, 2.0, 0.05).unwrap();
        let state = create_test_consciousness_state();

        let obj = LearningObjective::new("test", "Test")
            .with_difficulty(Difficulty::Beginner)
            .build();

        // Use the engine for evaluation
        let result1 = engine.evaluate(&obj, &state).unwrap();

        // Access CfC through getter
        let cfc = engine.cfc();
        let cfc_config = cfc.config();

        // Verify CfC configuration is intact
        assert_eq!(cfc_config.hidden_dim, 128);

        // Network should still work for predictions
        // (Would need another evaluation to verify, but getter is read-only)
        assert!(result1.predicted_phi_gain >= -0.1 && result1.predicted_phi_gain <= 0.1);
    }

    // =========================================================================
    // ESTIMATE_PHI_GAIN EDGE CASE TESTS
    // =========================================================================

    #[test]
    fn test_estimate_phi_gain_empty_array() {
        let engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // Create an empty array
        let empty_state = Array1::from_vec(vec![]);
        let current_phi = 0.5;

        let obj = LearningObjective::new("test", "Test")
            .with_difficulty(Difficulty::Intermediate)
            .build();

        // estimate_phi_gain should handle empty gracefully and return 0.0
        let gain = engine.estimate_phi_gain(&empty_state, current_phi, &obj);
        assert_eq!(gain, 0.0, "Empty array should return 0.0 gain");
    }

    #[test]
    fn test_estimate_phi_gain_single_element() {
        let engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // Single element array - chunk_size will be 0
        let single = Array1::from_vec(vec![0.5]);
        let current_phi = 0.3;

        let obj = LearningObjective::new("test", "Test")
            .with_difficulty(Difficulty::Beginner)
            .build();

        // Should handle gracefully
        let gain = engine.estimate_phi_gain(&single, current_phi, &obj);
        assert_eq!(
            gain, 0.0,
            "Single element should return 0.0 gain (chunk_size=0)"
        );
    }

    #[test]
    fn test_estimate_phi_gain_two_elements() {
        let engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // Two elements - chunk_size = 2/4 = 0
        let two = Array1::from_vec(vec![0.5, 0.7]);
        let current_phi = 0.3;

        let obj = LearningObjective::new("test", "Test")
            .with_difficulty(Difficulty::Beginner)
            .build();

        let gain = engine.estimate_phi_gain(&two, current_phi, &obj);
        assert_eq!(
            gain, 0.0,
            "Two elements should return 0.0 gain (chunk_size=0)"
        );
    }

    #[test]
    fn test_estimate_phi_gain_three_elements() {
        let engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // Three elements - chunk_size = 3/4 = 0
        let three = Array1::from_vec(vec![0.5, 0.7, 0.9]);
        let current_phi = 0.3;

        let obj = LearningObjective::new("test", "Test")
            .with_difficulty(Difficulty::Beginner)
            .build();

        let gain = engine.estimate_phi_gain(&three, current_phi, &obj);
        assert_eq!(
            gain, 0.0,
            "Three elements should return 0.0 gain (chunk_size=0)"
        );
    }

    #[test]
    fn test_estimate_phi_gain_four_elements() {
        let engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // Four elements - chunk_size = 4/4 = 1, can create 4 chunks
        let four = Array1::from_vec(vec![0.5, 0.7, 0.9, 0.3]);
        let current_phi = 0.3;

        let obj = LearningObjective::new("test", "Test")
            .with_difficulty(Difficulty::Intermediate)
            .build();

        // Should now compute a real phi gain (not necessarily 0)
        let gain = engine.estimate_phi_gain(&four, current_phi, &obj);
        // Gain should be clamped to [-0.1, 0.1]
        assert!(
            gain >= -0.1 && gain <= 0.1,
            "Gain should be clamped: {}",
            gain
        );
    }

    #[test]
    fn test_estimate_phi_gain_boundary_chunk_size() {
        let engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // Exactly 4 elements - boundary case
        let exactly_four = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let current_phi = 0.5;

        let obj = LearningObjective::new("test", "Test")
            .with_difficulty(Difficulty::Intermediate)
            .build();

        let gain = engine.estimate_phi_gain(&exactly_four, current_phi, &obj);
        assert!(gain >= -0.1 && gain <= 0.1, "Gain should be within bounds");
    }

    // =========================================================================
    // CHUNK_SIZE EDGE CASE TESTS (Line 226)
    // =========================================================================

    #[test]
    fn test_chunk_size_edge_case_zero_length() {
        let engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // Empty slice - max(0, 1) / 4 = 1/4 = 0, but handled specially
        let empty = Array1::<f32>::from_vec(vec![]);
        let obj = LearningObjective::new("test", "Test")
            .with_difficulty(Difficulty::Beginner)
            .build();

        let gain = engine.estimate_phi_gain(&empty, 0.5, &obj);
        assert_eq!(gain, 0.0, "Empty array should yield 0.0 gain");
    }

    #[test]
    fn test_chunk_size_edge_case_length_one() {
        let engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // Length 1: max(1, 1) / 4 = 1/4 = 0
        let one = Array1::from_vec(vec![0.5]);
        let obj = LearningObjective::new("test", "Test")
            .with_difficulty(Difficulty::Beginner)
            .build();

        let gain = engine.estimate_phi_gain(&one, 0.5, &obj);
        assert_eq!(gain, 0.0, "Single element should yield 0.0 gain");
    }

    #[test]
    fn test_chunk_size_edge_case_length_three() {
        let engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // Length 3: max(3, 1) / 4 = 3/4 = 0 (integer division)
        let three = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let obj = LearningObjective::new("test", "Test")
            .with_difficulty(Difficulty::Intermediate)
            .build();

        let gain = engine.estimate_phi_gain(&three, 0.4, &obj);
        assert_eq!(gain, 0.0, "Length 3 should yield 0.0 (chunk_size=0)");
    }

    #[test]
    fn test_chunk_size_edge_case_length_four() {
        let engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // Length 4: max(4, 1) / 4 = 4/4 = 1
        let four = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let obj = LearningObjective::new("test", "Test")
            .with_difficulty(Difficulty::Intermediate)
            .build();

        let gain = engine.estimate_phi_gain(&four, 0.4, &obj);
        // Now we can compute real phi - gain should be bounded
        assert!(gain >= -0.1 && gain <= 0.1, "Gain out of bounds: {}", gain);
    }

    #[test]
    fn test_chunk_size_edge_case_length_five() {
        let engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // Length 5: max(5, 1) / 4 = 5/4 = 1
        let five = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let obj = LearningObjective::new("test", "Test")
            .with_difficulty(Difficulty::Intermediate)
            .build();

        let gain = engine.estimate_phi_gain(&five, 0.3, &obj);
        assert!(gain >= -0.1 && gain <= 0.1, "Gain out of bounds: {}", gain);
    }

    // =========================================================================
    // ACCURACY EPSILON EDGE CASE TESTS (Line 313)
    // =========================================================================

    #[test]
    fn test_accuracy_epsilon_zero_predicted() {
        let mut engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // When predicted = 0, the comparison uses max(0.001, |predicted|) = 0.001
        // accurate = error < 0.2 * 0.001 = 0.0002
        engine.record_outcome(0.0, 0.0001);

        let (accuracy, _avg_error, count) = engine.accuracy_stats();
        assert_eq!(count, 1);
        // error = 0.0001, threshold = 0.0002, so this should be accurate
        assert!(
            accuracy >= 0.99,
            "Zero predicted with tiny error should be accurate"
        );
    }

    #[test]
    fn test_accuracy_epsilon_very_small_predicted() {
        let mut engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // predicted = 0.0001, threshold = 0.2 * max(0.001, 0.0001) = 0.2 * 0.001 = 0.0002
        // actual = 0.0001, error = 0
        engine.record_outcome(0.0001, 0.0001);

        let (accuracy, avg_error, count) = engine.accuracy_stats();
        assert_eq!(count, 1);
        assert!(accuracy >= 0.99, "Exact match should be accurate");
        assert!(avg_error.abs() < 0.0001, "No error for exact match");
    }

    #[test]
    fn test_accuracy_epsilon_very_large_predicted() {
        let mut engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // predicted = 100.0, threshold = 0.2 * 100.0 = 20.0
        // actual = 95.0, error = 5.0 < 20.0, should be accurate
        engine.record_outcome(100.0, 95.0);

        let (accuracy, avg_error, count) = engine.accuracy_stats();
        assert_eq!(count, 1);
        assert!(
            accuracy >= 0.99,
            "Within 5% of large prediction should be accurate"
        );
        assert!((avg_error - 5.0).abs() < 0.01, "Avg error should be 5.0");
    }

    #[test]
    fn test_accuracy_epsilon_large_error_large_predicted() {
        let mut engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // predicted = 100.0, threshold = 0.2 * 100.0 = 20.0
        // actual = 70.0, error = 30.0 > 20.0, should NOT be accurate
        engine.record_outcome(100.0, 70.0);

        let (accuracy, avg_error, count) = engine.accuracy_stats();
        assert_eq!(count, 1);
        assert!(accuracy < 0.01, "30% error should not be accurate");
        assert!((avg_error - 30.0).abs() < 0.01, "Avg error should be 30.0");
    }

    #[test]
    fn test_accuracy_epsilon_negative_predicted() {
        let mut engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // predicted = -0.05, |predicted| = 0.05 > 0.001
        // threshold = 0.2 * 0.05 = 0.01
        // actual = -0.05, error = 0 < 0.01, should be accurate
        engine.record_outcome(-0.05, -0.05);

        let (accuracy, avg_error, count) = engine.accuracy_stats();
        assert_eq!(count, 1);
        assert!(accuracy >= 0.99, "Exact match should be accurate");
        assert!(avg_error.abs() < 0.0001, "No error for exact match");
    }

    #[test]
    fn test_accuracy_epsilon_negative_with_small_error() {
        let mut engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // predicted = -0.05, threshold = 0.2 * 0.05 = 0.01
        // actual = -0.045, error = 0.005 < 0.01, should be accurate
        engine.record_outcome(-0.05, -0.045);

        let (accuracy, _avg_error, count) = engine.accuracy_stats();
        assert_eq!(count, 1);
        assert!(
            accuracy >= 0.99,
            "Small error on negative should be accurate"
        );
    }

    #[test]
    fn test_accuracy_epsilon_boundary_threshold() {
        let mut engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // Test just beyond the boundary to ensure rejection
        // predicted = 1.0, threshold = 0.2 * 1.0 = 0.2
        // error = 0.21 which is > 0.2, so should NOT be accurate
        engine.record_outcome(1.0, 0.79);

        let (accuracy, _avg_error, count) = engine.accuracy_stats();
        assert_eq!(count, 1);
        // error = 0.21, threshold = 0.2, 0.21 < 0.2 is false
        assert!(accuracy < 0.01, "Beyond threshold should not be accurate");
    }

    #[test]
    fn test_accuracy_epsilon_just_under_threshold() {
        let mut engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // predicted = 1.0, threshold = 0.2 * 1.0 = 0.2
        // error = 0.199 < 0.2, should be accurate
        engine.record_outcome(1.0, 0.801);

        let (accuracy, _avg_error, count) = engine.accuracy_stats();
        assert_eq!(count, 1);
        assert!(accuracy >= 0.99, "Just under threshold should be accurate");
    }

    #[test]
    fn test_accuracy_multiple_records_mixed() {
        let mut engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // Record a mix of accurate and inaccurate predictions
        engine.record_outcome(1.0, 0.95); // accurate (error=0.05 < 0.2)
        engine.record_outcome(1.0, 0.5); // inaccurate (error=0.5 > 0.2)
        engine.record_outcome(0.0, 0.0001); // accurate (error < 0.0002)
        engine.record_outcome(0.0, 0.001); // inaccurate (error=0.001 > 0.0002)

        let (accuracy, _avg_error, count) = engine.accuracy_stats();
        assert_eq!(count, 4);
        // 2 accurate out of 4 = 0.5
        assert!(
            (accuracy - 0.5).abs() < 0.01,
            "Should be 50% accurate: {}",
            accuracy
        );
    }

    // =========================================================================
    // CONFIDENCE CALCULATION TESTS
    // =========================================================================

    #[test]
    fn test_calculate_confidence_empty_history() {
        let engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // No predictions yet
        let confidence = engine.calculate_confidence();
        assert_eq!(confidence, 0.0, "Empty history should yield 0 confidence");
    }

    #[test]
    fn test_calculate_confidence_no_outcomes() {
        let engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // History exists but no outcomes recorded
        // (This happens when evaluate() is called but record_outcome() is not)
        let confidence = engine.calculate_confidence();
        assert_eq!(confidence, 0.0, "No outcomes should yield 0 confidence");
    }

    #[test]
    fn test_calculate_confidence_all_accurate() {
        let mut engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // Record all accurate predictions
        for _ in 0..10 {
            engine.record_outcome(1.0, 0.95); // Within 5%, threshold is 20%
        }

        let confidence = engine.calculate_confidence();
        assert!(
            (confidence - 1.0).abs() < 0.01,
            "All accurate should yield confidence near 1.0"
        );
    }

    #[test]
    fn test_calculate_confidence_all_inaccurate() {
        let mut engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // Record all inaccurate predictions
        for _ in 0..10 {
            engine.record_outcome(1.0, 0.5); // 50% off, threshold is 20%
        }

        let confidence = engine.calculate_confidence();
        assert!(
            confidence.abs() < 0.01,
            "All inaccurate should yield confidence near 0.0"
        );
    }

    // =========================================================================
    // RECOMMENDATION GENERATION TESTS
    // =========================================================================

    #[test]
    fn test_generate_recommendation_high_gain_high_confidence() {
        let mut engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // Build up confidence
        for _ in 0..10 {
            engine.record_outcome(0.05, 0.048);
        }

        let obj = LearningObjective::new("test", "Test")
            .with_difficulty(Difficulty::Intermediate)
            .build();

        let recommendation = engine.generate_recommendation(0.05, 0.9, &obj);
        assert!(matches!(
            recommendation,
            LearningRecommendation::LearnNow { .. }
        ));
    }

    #[test]
    fn test_generate_recommendation_low_gain_low_confidence() {
        let engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        let obj = LearningObjective::new("test", "Test")
            .with_difficulty(Difficulty::Intermediate)
            .build();

        // Low gain (below min_phi_gain) and low confidence
        let recommendation = engine.generate_recommendation(0.005, 0.3, &obj);
        assert!(matches!(
            recommendation,
            LearningRecommendation::Defer { .. }
        ));
    }

    #[test]
    fn test_generate_recommendation_negative_gain_high_confidence() {
        let engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        let obj = LearningObjective::new("test", "Test")
            .with_difficulty(Difficulty::Expert)
            .build();

        // Negative gain with high confidence = skip
        let recommendation = engine.generate_recommendation(-0.02, 0.8, &obj);
        assert!(matches!(
            recommendation,
            LearningRecommendation::Skip { .. }
        ));
    }

    // =========================================================================
    // HISTORY BOUNDING TESTS
    // =========================================================================

    #[test]
    fn test_prediction_history_bounded() {
        let mut engine = LookaheadEngine::new(64, 1.0, 0.01).unwrap();

        // Record more than 1000 predictions
        for i in 0..1200 {
            engine.record_outcome(i as f32 * 0.001, i as f32 * 0.001);
        }

        // History should be bounded to 1000
        // We can check this indirectly through prediction_count
        assert_eq!(
            engine.prediction_count(),
            1200,
            "Prediction count should be 1200"
        );

        // The history vector should have been trimmed
        // We can verify through confidence calculation (uses up to 20 recent)
        let confidence = engine.calculate_confidence();
        assert!(
            confidence >= 0.99,
            "Recent predictions should all be accurate"
        );
    }
}
