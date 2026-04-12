// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Meta-Cognition - Recursive Self-Modeling
//!
//! This module implements the recursive self-modeling capability that is
//! the hallmark of "Recursive Meta-Intelligence" - the system's ability
//! to model itself modeling itself.
//!
//! ## The Recursive Loop
//!
//! From the Kosmic Theory:
//! 1. Bottom-up emergence: Cognitive agents arise from physical process
//! 2. Scaling: Agents combine into higher-level collective intelligences
//! 3. Top-down constraint: Emergent wholes constrain their parts
//! 4. **Modeling and Modification**: At sufficient complexity, the system
//!    builds a model of itself and can consciously modify its own substrate
//!
//! This module implements step 4 for Symthaea: the system maintains a model
//! of its own cognitive processes and uses that model to improve itself.
//!
//! ## Self-Model Accuracy
//!
//! The key metric is "self-model accuracy" - how well does the system's
//! model of itself predict its actual behavior? This is measured by:
//! - Predicting our own prediction errors
//! - Predicting which harmonics will activate
//! - Predicting what primitives we'll select

use std::collections::VecDeque;

/// Maximum history for self-predictions
const MAX_PREDICTIONS: usize = 50;

/// The meta-cognitive layer that models the system itself
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields reserved for meta-cognitive modeling
pub struct MetaCognitiveLayer {
    /// History of self-predictions vs actual outcomes
    self_predictions: VecDeque<SelfPrediction>,

    /// Current model of own prediction error tendencies
    error_model: ErrorModel,

    /// Current model of own harmonic tendencies
    harmonic_model: HarmonicTendencyModel,

    /// Running accuracy estimate
    accuracy: f32,

    /// How many recursive levels deep we can model
    /// Level 0: No self-model
    /// Level 1: Model of self
    /// Level 2: Model of model of self
    /// etc.
    recursion_depth: u8,
}

impl MetaCognitiveLayer {
    pub fn new() -> Self {
        Self {
            self_predictions: VecDeque::with_capacity(MAX_PREDICTIONS),
            error_model: ErrorModel::new(),
            harmonic_model: HarmonicTendencyModel::new(),
            accuracy: 0.5,      // Start uncertain
            recursion_depth: 1, // Basic self-model
        }
    }

    /// Get current self-model accuracy
    pub fn accuracy(&self) -> f32 {
        self.accuracy
    }

    /// Get recursion depth
    pub fn depth(&self) -> u8 {
        self.recursion_depth
    }

    /// Predict what our prediction error will be for a given input complexity
    pub fn predict_own_error(&self, input_complexity: f32) -> f32 {
        self.error_model.predict(input_complexity)
    }

    /// Update the self-model based on actual prediction error
    pub fn update_self_model(&mut self, actual_error: f32) {
        // Get what we predicted
        let predicted = self.error_model.last_prediction;

        // Calculate meta-prediction error (error about our error)
        let meta_error = (actual_error - predicted).abs();

        // Record this prediction
        let prediction = SelfPrediction {
            predicted_error: predicted,
            actual_error,
            meta_error,
        };

        if self.self_predictions.len() >= MAX_PREDICTIONS {
            self.self_predictions.pop_front();
        }
        self.self_predictions.push_back(prediction);

        // Update error model
        self.error_model.update(actual_error);

        // Update accuracy estimate
        self.update_accuracy();
    }

    /// Get the recursive model (model of our self-model)
    pub fn recursive_model(&self) -> RecursiveModel {
        RecursiveModel {
            depth: self.recursion_depth,
            accuracy_at_depth: vec![self.accuracy], // For now, just one level
            can_model_deeper: self.accuracy > 0.7,  // Need good accuracy to go deeper
        }
    }

    /// Attempt to increase recursion depth
    pub fn deepen_recursion(&mut self) -> bool {
        if self.accuracy > 0.7 && self.recursion_depth < 3 {
            self.recursion_depth += 1;
            true
        } else {
            false
        }
    }

    /// Get self-model accuracy summary
    pub fn accuracy_summary(&self) -> SelfModelAccuracy {
        let recent_meta_errors: Vec<f32> = self
            .self_predictions
            .iter()
            .rev()
            .take(10)
            .map(|p| p.meta_error)
            .collect();

        let trend = if recent_meta_errors.len() >= 5 {
            let first_half: f32 = recent_meta_errors[..recent_meta_errors.len() / 2]
                .iter()
                .sum();
            let second_half: f32 = recent_meta_errors[recent_meta_errors.len() / 2..]
                .iter()
                .sum();
            if second_half < first_half * 0.9 {
                AccuracyTrend::Improving
            } else if second_half > first_half * 1.1 {
                AccuracyTrend::Declining
            } else {
                AccuracyTrend::Stable
            }
        } else {
            AccuracyTrend::Insufficient
        };

        SelfModelAccuracy {
            current_accuracy: self.accuracy,
            recursion_depth: self.recursion_depth,
            trend,
            predictions_made: self.self_predictions.len(),
        }
    }

    fn update_accuracy(&mut self) {
        if self.self_predictions.is_empty() {
            return;
        }

        // Accuracy is inverse of average meta-error
        let avg_meta_error: f32 = self
            .self_predictions
            .iter()
            .map(|p| p.meta_error)
            .sum::<f32>()
            / self.self_predictions.len() as f32;

        self.accuracy = (1.0 - avg_meta_error).clamp(0.0, 1.0);
    }
}

impl Default for MetaCognitiveLayer {
    fn default() -> Self {
        Self::new()
    }
}

/// A single self-prediction and its outcome
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields reserved for meta-prediction
struct SelfPrediction {
    predicted_error: f32,
    actual_error: f32,
    meta_error: f32, // |predicted - actual|
}

/// Model of our own prediction error tendencies
#[derive(Debug, Clone)]
struct ErrorModel {
    /// Running average of prediction errors
    average_error: f32,
    /// Variance in prediction errors
    error_variance: f32,
    /// Last prediction made
    last_prediction: f32,
    /// Sample count
    samples: u32,
}

impl ErrorModel {
    fn new() -> Self {
        Self {
            average_error: 0.3, // Prior: assume moderate error
            error_variance: 0.1,
            last_prediction: 0.3,
            samples: 0,
        }
    }

    fn predict(&self, _input_complexity: f32) -> f32 {
        // Simple model: predict average error, adjusted by complexity
        // More complex inputs → higher expected error
        self.average_error
    }

    fn update(&mut self, actual_error: f32) {
        self.samples += 1;

        // Online mean update
        let delta = actual_error - self.average_error;
        self.average_error += delta / self.samples as f32;

        // Online variance update (Welford's algorithm)
        let delta2 = actual_error - self.average_error;
        self.error_variance += (delta * delta2 - self.error_variance) / self.samples as f32;

        // Next prediction
        self.last_prediction = self.average_error;
    }
}

/// Model of our harmonic tendencies
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields reserved for tendency modeling
struct HarmonicTendencyModel {
    /// Which harmonics tend to activate most
    tendency_weights: [f32; 7],
}

impl HarmonicTendencyModel {
    fn new() -> Self {
        Self {
            tendency_weights: [1.0 / 7.0; 7], // Uniform prior
        }
    }
}

/// The recursive model structure
#[derive(Debug, Clone)]
pub struct RecursiveModel {
    /// How many levels deep we can model
    pub depth: u8,
    /// Accuracy at each recursion level
    pub accuracy_at_depth: Vec<f32>,
    /// Whether we have capacity to model deeper
    pub can_model_deeper: bool,
}

impl RecursiveModel {
    /// Describe the recursion in natural language
    pub fn describe(&self) -> &'static str {
        match self.depth {
            0 => "No self-model (purely reactive)",
            1 => "Basic self-model (I know what I tend to do)",
            2 => "Meta-self-model (I know how I model myself)",
            3 => "Deep recursion (I model my modeling of my modeling)",
            _ => "Profound recursion (deep self-reference)",
        }
    }
}

/// Summary of self-model accuracy
#[derive(Debug, Clone)]
pub struct SelfModelAccuracy {
    pub current_accuracy: f32,
    pub recursion_depth: u8,
    pub trend: AccuracyTrend,
    pub predictions_made: usize,
}

/// Trend in accuracy over time
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccuracyTrend {
    Improving,
    Stable,
    Declining,
    Insufficient, // Not enough data
}

/// A suggested threshold mutation from the meta-cognitive layer.
#[derive(Debug, Clone)]
pub struct MutationSuggestion {
    pub target: MutationTarget,
    pub direction: f32,
    pub confidence: f32,
    pub reason: &'static str,
}

/// Target parameter for a mutation suggestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MutationTarget {
    FepSurpriseScale,
    FepLrDecay,
    DreamBaseInterval,
    NeuromodArousalDecay,
    HomeostasisPullCruise,
    FlowExplorationIncrement,
    SelfModelWeightHigh,
}

impl MetaCognitiveLayer {
    /// Suggest threshold mutations when accuracy plateaus.
    pub fn suggest_mutations(&self) -> Vec<MutationSuggestion> {
        let summary = self.accuracy_summary();
        if summary.trend != AccuracyTrend::Stable || summary.predictions_made < 20 {
            return Vec::new();
        }
        let mut suggestions = Vec::new();
        let avg_error = self.error_model.average_error;
        let error_var = self.error_model.error_variance;

        if avg_error > 0.4 && error_var < 0.05 {
            suggestions.push(MutationSuggestion {
                target: MutationTarget::FepSurpriseScale,
                direction: 1.0,
                confidence: (avg_error - 0.3).clamp(0.0, 1.0),
                reason: "high systematic error suggests insufficient surprise-driven learning",
            });
            suggestions.push(MutationSuggestion {
                target: MutationTarget::FepLrDecay,
                direction: -1.0,
                confidence: 0.5,
                reason: "persistent error benefits from slower LR decay",
            });
        }
        if error_var > 0.1 {
            suggestions.push(MutationSuggestion {
                target: MutationTarget::NeuromodArousalDecay,
                direction: 1.0,
                confidence: (error_var - 0.05).clamp(0.0, 1.0),
                reason: "high error variance suggests unstable arousal dynamics",
            });
            suggestions.push(MutationSuggestion {
                target: MutationTarget::FlowExplorationIncrement,
                direction: -1.0,
                confidence: 0.4,
                reason: "reducing exploration may stabilize error variance",
            });
        }
        if self.accuracy > 0.5 && self.accuracy < 0.7 {
            suggestions.push(MutationSuggestion {
                target: MutationTarget::SelfModelWeightHigh,
                direction: -1.0,
                confidence: 0.3,
                reason: "moderate accuracy plateau may benefit from lower self-model dominance",
            });
        }
        if avg_error < 0.2 && self.accuracy < 0.8 {
            suggestions.push(MutationSuggestion {
                target: MutationTarget::DreamBaseInterval,
                direction: -1.0,
                confidence: 0.4,
                reason: "low error with plateaued accuracy suggests consolidation gap",
            });
        }
        suggestions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        suggestions.truncate(3);
        suggestions
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_metacognition() {
        let meta = MetaCognitiveLayer::new();
        assert_eq!(meta.depth(), 1);
        assert_eq!(meta.accuracy(), 0.5); // Uncertain start
    }

    #[test]
    fn test_self_model_improves_with_consistent_errors() {
        let mut meta = MetaCognitiveLayer::new();

        // Feed consistent errors - model should improve
        for _ in 0..20 {
            meta.update_self_model(0.3); // Always 0.3 error
        }

        // Should be getting more accurate at predicting 0.3
        assert!(meta.accuracy() > 0.6);
    }

    #[test]
    fn test_recursive_model_description() {
        let meta = MetaCognitiveLayer::new();
        let recursive = meta.recursive_model();
        assert_eq!(recursive.depth, 1);
        assert!(recursive.describe().contains("Basic"));
    }

    #[test]
    fn test_deepen_recursion_requires_accuracy() {
        let mut meta = MetaCognitiveLayer::new();

        // Can't deepen with low accuracy
        meta.accuracy = 0.5;
        assert!(!meta.deepen_recursion());

        // Can deepen with high accuracy
        meta.accuracy = 0.8;
        assert!(meta.deepen_recursion());
        assert_eq!(meta.depth(), 2);
    }

    #[test]
    fn test_accuracy_trend() {
        let mut meta = MetaCognitiveLayer::new();

        // Feed improving predictions
        for i in 0..10 {
            meta.update_self_model(0.5 - (i as f32 * 0.03));
        }

        let summary = meta.accuracy_summary();
        assert!(summary.predictions_made >= 10);
    }

    // ── MetaCognitiveLayer construction ─────────────────────────────────

    #[test]
    fn test_default_equals_new() {
        let a = MetaCognitiveLayer::new();
        let b = MetaCognitiveLayer::default();
        assert_eq!(a.accuracy(), b.accuracy());
        assert_eq!(a.depth(), b.depth());
    }

    #[test]
    fn test_predict_own_error_initial() {
        let meta = MetaCognitiveLayer::new();
        let prediction = meta.predict_own_error(0.5);
        assert!(prediction.is_finite());
        // Initial prediction should be the prior (0.3)
        assert!((prediction - 0.3).abs() < f32::EPSILON);
    }

    // ── update_self_model ───────────────────────────────────────────────

    #[test]
    fn test_accuracy_bounded_zero_to_one() {
        let mut meta = MetaCognitiveLayer::new();
        for _ in 0..50 {
            meta.update_self_model(0.99);
        }
        assert!(meta.accuracy() >= 0.0 && meta.accuracy() <= 1.0);
    }

    #[test]
    fn test_accuracy_improves_with_zero_error() {
        let mut meta = MetaCognitiveLayer::new();
        // Always zero error, so the model should converge
        for _ in 0..30 {
            meta.update_self_model(0.0);
        }
        // With consistently 0 error, model should predict ~0 and accuracy should be high
        assert!(meta.accuracy() > 0.7, "Accuracy = {}", meta.accuracy());
    }

    #[test]
    fn test_predictions_capped_at_max() {
        let mut meta = MetaCognitiveLayer::new();
        for _ in 0..100 {
            meta.update_self_model(0.5);
        }
        let summary = meta.accuracy_summary();
        // MAX_PREDICTIONS is 50, so predictions_made should be capped
        assert!(summary.predictions_made <= 50);
    }

    // ── deepen_recursion ────────────────────────────────────────────────

    #[test]
    fn test_recursion_depth_capped_at_three() {
        let mut meta = MetaCognitiveLayer::new();
        meta.accuracy = 0.9;
        assert!(meta.deepen_recursion()); // 1 -> 2
        assert!(meta.deepen_recursion()); // 2 -> 3
        assert!(!meta.deepen_recursion()); // 3 -> blocked
        assert_eq!(meta.depth(), 3);
    }

    // ── RecursiveModel ──────────────────────────────────────────────────

    #[test]
    fn test_recursive_model_depth_zero() {
        let model = RecursiveModel {
            depth: 0,
            accuracy_at_depth: vec![],
            can_model_deeper: false,
        };
        assert_eq!(model.describe(), "No self-model (purely reactive)");
    }

    #[test]
    fn test_recursive_model_all_depths() {
        let descriptions = [
            "No self-model (purely reactive)",
            "Basic self-model (I know what I tend to do)",
            "Meta-self-model (I know how I model myself)",
            "Deep recursion (I model my modeling of my modeling)",
            "Profound recursion (deep self-reference)",
        ];
        for (depth, expected) in descriptions.iter().enumerate() {
            let model = RecursiveModel {
                depth: depth as u8,
                accuracy_at_depth: vec![],
                can_model_deeper: false,
            };
            assert_eq!(model.describe(), *expected, "Depth {} mismatch", depth);
        }
    }

    // ── AccuracyTrend ───────────────────────────────────────────────────

    #[test]
    fn test_insufficient_trend_with_few_data_points() {
        let mut meta = MetaCognitiveLayer::new();
        meta.update_self_model(0.3);
        meta.update_self_model(0.3);
        let summary = meta.accuracy_summary();
        assert_eq!(summary.trend, AccuracyTrend::Insufficient);
    }

    #[test]
    fn test_accuracy_summary_fields() {
        let meta = MetaCognitiveLayer::new();
        let summary = meta.accuracy_summary();
        assert_eq!(summary.current_accuracy, 0.5);
        assert_eq!(summary.recursion_depth, 1);
        assert_eq!(summary.predictions_made, 0);
    }

    #[test]
    fn test_recursive_model_can_model_deeper() {
        let mut meta = MetaCognitiveLayer::new();
        meta.accuracy = 0.8;
        let model = meta.recursive_model();
        assert!(
            model.can_model_deeper,
            "High accuracy should allow deeper modeling"
        );

        meta.accuracy = 0.3;
        let model = meta.recursive_model();
        assert!(
            !model.can_model_deeper,
            "Low accuracy should not allow deeper modeling"
        );
    }
}
