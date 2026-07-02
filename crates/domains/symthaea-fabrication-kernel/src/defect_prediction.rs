// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Defect prediction from manufacturing data
//!
//! Online-learning defect predictor that ingests manufacturing telemetry
//! (tolerance, surface quality, throughput, energy cost, anomaly count,
//! layer count) and predicts part quality via a simple linear model
//! with Welford online statistics and gradient-descent weight updates.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Feature schema
// ---------------------------------------------------------------------------

/// Canonical names for the 6 input features, in index order.
const FEATURE_NAMES: [&str; 6] = [
    "tolerance",
    "surface_quality",
    "throughput",
    "energy_efficiency",
    "anomaly_free",
    "layer_density",
];

const NUM_FEATURES: usize = 6;

/// Learning rate for online weight updates.
const LEARNING_RATE: f64 = 0.01;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Raw manufacturing telemetry for a single part or batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionInput {
    /// Dimensional tolerance (0.0 = worst, 1.0 = best).
    pub tolerance: f64,
    /// Surface finish quality (0.0 = worst, 1.0 = best).
    pub surface_quality: f64,
    /// Production throughput (0.0 = worst, 1.0 = best).
    pub throughput: f64,
    /// Energy cost (0.0 = cheapest, 1.0 = most expensive).
    pub energy_cost: f64,
    /// Number of detected anomalies in the run.
    pub anomaly_count: u32,
    /// Number of additive-manufacturing layers.
    pub layer_count: u32,
}

/// Predicted quality outcome with confidence and risk breakdown.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionOutput {
    /// Predicted quality score in [0, 1].
    pub predicted_quality: f64,
    /// Confidence in the prediction based on training data volume.
    pub confidence: f64,
    /// Individual risk factors that contribute to lower quality.
    pub risk_factors: Vec<RiskFactor>,
}

/// A single risk factor identified during prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    /// Human-readable feature name.
    pub name: String,
    /// How much this factor contributes to reduced quality (higher = worse).
    pub contribution: f64,
    /// Severity level in [0, 1].
    pub severity: f32,
}

/// A suggested parameter adjustment to improve quality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adjustment {
    /// Name of the parameter to adjust.
    pub parameter: String,
    /// Current feature value.
    pub current_value: f64,
    /// Suggested feature value.
    pub suggested_value: f64,
    /// Expected quality improvement from this change.
    pub expected_improvement: f64,
}

/// Online-learning defect predictor with 6 features.
///
/// Uses a weighted linear model with Welford online mean/variance tracking
/// and gradient-descent weight updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefectPredictor {
    /// Model weights, one per feature.
    pub weights: Vec<f64>,
    /// Number of training samples seen.
    pub training_count: u64,
    /// Running feature means (Welford).
    pub feature_means: Vec<f64>,
    /// Running feature variances (Welford).
    pub feature_vars: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------

/// Extract the 6 normalized features from a [`PredictionInput`].
fn extract_features(input: &PredictionInput) -> [f64; NUM_FEATURES] {
    [
        input.tolerance,
        input.surface_quality,
        input.throughput,
        1.0 - input.energy_cost,
        1.0 / (1.0 + input.anomaly_count as f64),
        (input.layer_count.min(500) as f64) / 500.0,
    ]
}

// ---------------------------------------------------------------------------
// DefectPredictor
// ---------------------------------------------------------------------------

impl DefectPredictor {
    /// Create a new predictor with equal weights (1/6 each), means at 0.5,
    /// and small initial variance.
    pub fn new() -> Self {
        Self {
            weights: vec![1.0 / NUM_FEATURES as f64; NUM_FEATURES],
            training_count: 0,
            feature_means: vec![0.5; NUM_FEATURES],
            feature_vars: vec![0.01; NUM_FEATURES],
        }
    }

    /// Predict quality from manufacturing input.
    ///
    /// Quality = dot(normalized_features, weights), clamped to [0, 1].
    /// Confidence = min(training_count / 100, 1.0).
    /// Risk factors are generated for features that fall below their running mean.
    pub fn predict(&self, input: &PredictionInput) -> PredictionOutput {
        let features = extract_features(input);

        // Normalize features relative to running stats
        let normalized: Vec<f64> = features
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                let std_dev = self.feature_vars[i].sqrt().max(1e-8);
                (f - self.feature_means[i]) / std_dev
            })
            .collect();

        // Weighted sum → quality prediction
        let raw: f64 = normalized
            .iter()
            .zip(self.weights.iter())
            .map(|(n, w)| n * w)
            .sum();

        // Map through sigmoid-like transform and clamp
        let predicted_quality = (0.5 + raw * 0.25).clamp(0.0, 1.0);

        // Confidence from data volume
        let confidence = (self.training_count as f64 / 100.0).min(1.0);

        // Identify risk factors: features below their mean
        let mut risk_factors = Vec::new();
        for (i, &f) in features.iter().enumerate() {
            if f < self.feature_means[i] {
                let deficit = self.feature_means[i] - f;
                let severity = (deficit / self.feature_means[i].max(1e-8)).min(1.0) as f32;
                risk_factors.push(RiskFactor {
                    name: FEATURE_NAMES[i].to_string(),
                    contribution: deficit * self.weights[i].abs(),
                    severity,
                });
            }
        }

        // Sort by contribution descending
        risk_factors.sort_by(|a, b| {
            b.contribution
                .partial_cmp(&a.contribution)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        PredictionOutput {
            predicted_quality,
            confidence,
            risk_factors,
        }
    }

    /// Update the model with a single training example (online learning).
    ///
    /// Updates running means/variances via Welford's algorithm and adjusts
    /// weights via gradient descent: `w -= lr * (predicted - actual) * feature`.
    pub fn train(&mut self, input: &PredictionInput, actual_quality: f64) {
        let features = extract_features(input);
        self.training_count += 1;
        let n = self.training_count as f64;

        // Welford online update for means and variances
        for (i, &f) in features.iter().enumerate() {
            let delta = f - self.feature_means[i];
            self.feature_means[i] += delta / n;
            let delta2 = f - self.feature_means[i];
            // Running variance (population)
            if n > 1.0 {
                self.feature_vars[i] += (delta * delta2 - self.feature_vars[i]) / n;
            }
        }

        // Predict with current weights (before update)
        let prediction = self.predict(input);
        let error = prediction.predicted_quality - actual_quality;

        // Normalize features for gradient computation
        let normalized: Vec<f64> = features
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                let std_dev = self.feature_vars[i].sqrt().max(1e-8);
                (f - self.feature_means[i]) / std_dev
            })
            .collect();

        // Gradient descent step
        for (i, w) in self.weights.iter_mut().enumerate() {
            *w -= LEARNING_RATE * error * normalized[i];
        }
    }

    /// Return feature importance ranked by absolute weight magnitude (descending).
    pub fn feature_importance(&self) -> Vec<(String, f64)> {
        let mut importance: Vec<(String, f64)> = FEATURE_NAMES
            .iter()
            .zip(self.weights.iter())
            .map(|(name, &w)| (name.to_string(), w))
            .collect();
        importance.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        importance
    }

    /// Suggest parameter adjustments to move predicted quality toward `target_quality`.
    ///
    /// For each feature whose current contribution is below the proportional
    /// target, suggest increasing its value.
    pub fn suggest_adjustments(
        &self,
        input: &PredictionInput,
        target_quality: f64,
    ) -> Vec<Adjustment> {
        let features = extract_features(input);
        let prediction = self.predict(input);
        let quality_gap = target_quality - prediction.predicted_quality;

        if quality_gap <= 0.0 {
            // Already at or above target
            return Vec::new();
        }

        let mut adjustments = Vec::new();
        let total_weight: f64 = self.weights.iter().map(|w| w.abs()).sum();

        for (i, &f) in features.iter().enumerate() {
            let weight_frac = self.weights[i].abs() / total_weight.max(1e-8);
            let target_contribution = weight_frac * quality_gap;

            // If the feature is below its mean, suggest raising it
            if f < self.feature_means[i] || target_contribution > 0.01 {
                let suggested = (f + target_contribution * 2.0).min(1.0);
                if (suggested - f).abs() > 1e-6 {
                    let expected_improvement = (suggested - f) * self.weights[i].abs() * 0.25;
                    adjustments.push(Adjustment {
                        parameter: FEATURE_NAMES[i].to_string(),
                        current_value: f,
                        suggested_value: suggested,
                        expected_improvement,
                    });
                }
            }
        }

        // Sort by expected improvement descending
        adjustments.sort_by(|a, b| b.expected_improvement.total_cmp(&a.expected_improvement));

        adjustments
    }
}

impl Default for DefectPredictor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_input() -> PredictionInput {
        PredictionInput {
            tolerance: 0.8,
            surface_quality: 0.9,
            throughput: 0.7,
            energy_cost: 0.3,
            anomaly_count: 0,
            layer_count: 100,
        }
    }

    fn poor_input() -> PredictionInput {
        PredictionInput {
            tolerance: 0.1,
            surface_quality: 0.2,
            throughput: 0.1,
            energy_cost: 0.9,
            anomaly_count: 10,
            layer_count: 5,
        }
    }

    #[test]
    fn new_predictor_defaults() {
        let p = DefectPredictor::new();
        assert_eq!(p.weights.len(), 6);
        assert_eq!(p.training_count, 0);
        assert_eq!(p.feature_means.len(), 6);
        assert_eq!(p.feature_vars.len(), 6);
        // All weights should start equal
        let expected_w = 1.0 / 6.0;
        for w in &p.weights {
            assert!((w - expected_w).abs() < 1e-10);
        }
    }

    #[test]
    fn predict_untrained() {
        let p = DefectPredictor::new();
        let output = p.predict(&sample_input());
        // Should produce a valid prediction even with no training
        assert!(output.predicted_quality >= 0.0 && output.predicted_quality <= 1.0);
        assert_eq!(output.confidence, 0.0); // No training data
    }

    #[test]
    fn train_improves() {
        let mut p = DefectPredictor::new();
        let good = sample_input();

        // Train on good samples with high quality
        for _ in 0..200 {
            p.train(&good, 0.95);
        }

        let output_good = p.predict(&good);
        let output_poor = p.predict(&poor_input());

        // After training on good data, good input should score higher than poor input
        assert!(
            output_good.predicted_quality > output_poor.predicted_quality,
            "Good: {}, Poor: {}",
            output_good.predicted_quality,
            output_poor.predicted_quality
        );
    }

    #[test]
    fn feature_importance_sorted() {
        let mut p = DefectPredictor::new();
        let good = sample_input();
        for _ in 0..50 {
            p.train(&good, 0.9);
        }
        let importance = p.feature_importance();
        assert_eq!(importance.len(), 6);
        // Should be sorted by absolute weight descending
        for i in 1..importance.len() {
            assert!(
                importance[i - 1].1.abs() >= importance[i].1.abs(),
                "Not sorted at index {}: {} vs {}",
                i,
                importance[i - 1].1.abs(),
                importance[i].1.abs()
            );
        }
    }

    #[test]
    fn adjustment_suggestions() {
        let mut p = DefectPredictor::new();
        let input = poor_input();
        for _ in 0..100 {
            p.train(&sample_input(), 0.95);
        }
        let adjustments = p.suggest_adjustments(&input, 0.99);
        assert!(
            !adjustments.is_empty(),
            "Should suggest at least one adjustment"
        );
        for adj in &adjustments {
            assert!(
                adj.suggested_value >= adj.current_value,
                "Suggestion should not decrease: {} -> {}",
                adj.current_value,
                adj.suggested_value
            );
            assert!(adj.expected_improvement >= 0.0);
        }
    }

    #[test]
    fn confidence_increases() {
        let mut p = DefectPredictor::new();
        let input = sample_input();
        assert_eq!(p.predict(&input).confidence, 0.0);

        for _ in 0..50 {
            p.train(&input, 0.8);
        }
        let c50 = p.predict(&input).confidence;
        assert!(c50 > 0.0 && c50 <= 0.5);

        for _ in 0..50 {
            p.train(&input, 0.8);
        }
        let c100 = p.predict(&input).confidence;
        assert!(c100 >= c50);
        assert!(
            (c100 - 1.0).abs() < 1e-6,
            "100 samples should give confidence=1.0"
        );
    }

    #[test]
    fn risk_factor_identification() {
        let p = DefectPredictor::new();
        // With default means at 0.5, a poor input should trigger risk factors
        let output = p.predict(&poor_input());
        // At least some features are below the mean of 0.5
        assert!(
            !output.risk_factors.is_empty(),
            "Poor input should produce risk factors"
        );
        for rf in &output.risk_factors {
            assert!(rf.severity >= 0.0 && rf.severity <= 1.0);
            assert!(rf.contribution >= 0.0);
        }
    }

    #[test]
    fn serde_roundtrip() {
        let mut p = DefectPredictor::new();
        p.train(&sample_input(), 0.85);
        p.train(&sample_input(), 0.90);

        let json = serde_json::to_string(&p).unwrap();
        let restored: DefectPredictor = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.training_count, p.training_count);
        assert_eq!(restored.weights.len(), p.weights.len());
        for (a, b) in restored.weights.iter().zip(p.weights.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_training_convergence() {
        let mut predictor = DefectPredictor::new();
        // Train on 50 samples where quality = tolerance (simple linear)
        for i in 0..50 {
            let t = (i as f64) / 50.0;
            let input = PredictionInput {
                tolerance: t,
                surface_quality: 0.5,
                throughput: 0.5,
                energy_cost: 0.5,
                anomaly_count: 0,
                layer_count: 100,
            };
            predictor.train(&input, t); // quality = tolerance
        }
        // Predict: high tolerance should give higher quality than low
        let high = predictor.predict(&PredictionInput {
            tolerance: 0.9,
            surface_quality: 0.5,
            throughput: 0.5,
            energy_cost: 0.5,
            anomaly_count: 0,
            layer_count: 100,
        });
        let low = predictor.predict(&PredictionInput {
            tolerance: 0.1,
            surface_quality: 0.5,
            throughput: 0.5,
            energy_cost: 0.5,
            anomaly_count: 0,
            layer_count: 100,
        });
        assert!(
            high.predicted_quality > low.predicted_quality,
            "High tolerance ({:.3}) should predict higher quality than low ({:.3})",
            high.predicted_quality,
            low.predicted_quality
        );
    }

    #[test]
    fn no_adjustments_when_above_target() {
        let p = DefectPredictor::new();
        let input = sample_input();
        let pred = p.predict(&input);
        // Ask for target below current prediction
        let adjustments = p.suggest_adjustments(&input, pred.predicted_quality - 0.1);
        assert!(
            adjustments.is_empty(),
            "Should not suggest adjustments when already above target"
        );
    }

    #[test]
    fn feature_extraction_bounds() {
        let input = PredictionInput {
            tolerance: 1.0,
            surface_quality: 1.0,
            throughput: 1.0,
            energy_cost: 0.0,
            anomaly_count: 0,
            layer_count: 1000,
        };
        let features = extract_features(&input);
        for f in &features {
            assert!(*f >= 0.0 && *f <= 1.0, "Feature out of bounds: {}", f);
        }
    }
}
