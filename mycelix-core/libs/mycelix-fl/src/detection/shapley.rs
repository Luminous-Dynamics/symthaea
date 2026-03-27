// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Shapley-value based Byzantine detection.
//!
//! Computes each node's marginal contribution to overall model quality
//! using leave-one-out evaluation. Nodes with low or negative Shapley
//! values are flagged as suspicious (they hurt the aggregate).
//!
//! # Quality Metrics
//!
//! - **NegativeVariance**: Quality = -Var(aggregated). Lower variance among
//!   the remaining gradients means more agreement, which is better.
//! - **CosineToMean**: Quality = average cosine similarity of each remaining
//!   gradient to the mean of the remaining set.
//!
//! # Complexity
//!
//! The leave-one-out computation is O(n^2 * d) where n is the number of
//! nodes and d is the gradient dimensionality.

use serde::{Deserialize, Serialize};

use crate::error::FlError;
use crate::types::{cosine_similarity, Gradient};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Quality metric used for Shapley value computation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum QualityMetric {
    /// Quality = -Var(aggregated gradient). Lower variance = more agreement.
    NegativeVariance,
    /// Quality = average cosine similarity of each gradient to the coalition mean.
    CosineToMean,
}

/// Configuration for the Shapley detector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShapleyConfig {
    /// Shapley value below this threshold is suspicious. Default: -0.1
    pub threshold: f64,
    /// Quality metric to use for leave-one-out evaluation.
    pub quality_metric: QualityMetric,
}

impl Default for ShapleyConfig {
    fn default() -> Self {
        Self {
            threshold: -0.1,
            quality_metric: QualityMetric::NegativeVariance,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Result of Shapley-based detection.
#[derive(Clone, Debug)]
pub struct ShapleyResult {
    /// (node_id, shapley_value) pairs for every node.
    pub values: Vec<(String, f64)>,
    /// Node IDs whose Shapley value fell below the threshold.
    pub suspicious: Vec<String>,
}

// ---------------------------------------------------------------------------
// Detector
// ---------------------------------------------------------------------------

/// Shapley-value based Byzantine detector.
///
/// For each node i, computes:
///   Shapley(i) = Quality(all) - Quality(all \ {i})
///
/// A negative Shapley value means removing the node *improves* quality,
/// i.e. the node is harmful.
pub struct ShapleyDetector {
    config: ShapleyConfig,
}

impl ShapleyDetector {
    /// Create a new detector with the given configuration.
    pub fn new(config: ShapleyConfig) -> Self {
        Self { config }
    }

    /// Create a detector with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ShapleyConfig::default())
    }

    /// Get the configuration.
    pub fn config(&self) -> &ShapleyConfig {
        &self.config
    }

    /// Detect suspicious nodes via leave-one-out Shapley values.
    ///
    /// # Errors
    ///
    /// Returns [`FlError::EmptyGradients`] if the slice is empty.
    /// Returns [`FlError::DimensionMismatch`] if gradients differ in length.
    pub fn detect(&self, gradients: &[Gradient]) -> Result<ShapleyResult, FlError> {
        if gradients.is_empty() {
            return Err(FlError::EmptyGradients);
        }

        let dim = gradients[0].dim();
        if dim == 0 {
            return Err(FlError::EmptyGradient);
        }
        for g in &gradients[1..] {
            if g.dim() != dim {
                return Err(FlError::DimensionMismatch {
                    expected: dim,
                    got: g.dim(),
                });
            }
        }

        // Single gradient: its Shapley value is just the quality of itself.
        if gradients.len() == 1 {
            let q = self.quality_of(&gradients[0..1]);
            return Ok(ShapleyResult {
                values: vec![(gradients[0].node_id.clone(), q)],
                suspicious: if q < self.config.threshold {
                    vec![gradients[0].node_id.clone()]
                } else {
                    vec![]
                },
            });
        }

        // Quality of the full coalition.
        let q_all = self.quality_of(gradients);

        // Leave-one-out: for each node, compute quality without it.
        let mut values = Vec::with_capacity(gradients.len());
        let mut suspicious = Vec::new();

        for i in 0..gradients.len() {
            let without: Vec<&Gradient> = gradients
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, g)| g)
                .collect();

            let q_without = self.quality_of_refs(&without);
            let shapley = q_all - q_without;

            if shapley < self.config.threshold {
                suspicious.push(gradients[i].node_id.clone());
            }
            values.push((gradients[i].node_id.clone(), shapley));
        }

        Ok(ShapleyResult {
            values,
            suspicious,
        })
    }

    // -----------------------------------------------------------------------
    // Quality metrics
    // -----------------------------------------------------------------------

    /// Compute quality of a set of gradients (owned slice).
    fn quality_of(&self, gradients: &[Gradient]) -> f64 {
        let refs: Vec<&Gradient> = gradients.iter().collect();
        self.quality_of_refs(&refs)
    }

    /// Compute quality of a set of gradient references.
    fn quality_of_refs(&self, gradients: &[&Gradient]) -> f64 {
        if gradients.is_empty() {
            return 0.0;
        }
        if gradients.len() == 1 {
            // Single gradient: variance is 0 -> quality is 0 (negative variance)
            // or cosine to self is 1.0.
            return match self.config.quality_metric {
                QualityMetric::NegativeVariance => 0.0,
                QualityMetric::CosineToMean => 1.0,
            };
        }

        match self.config.quality_metric {
            QualityMetric::NegativeVariance => self.negative_variance(gradients),
            QualityMetric::CosineToMean => self.cosine_to_mean(gradients),
        }
    }

    /// Quality = -Var(mean_gradient).
    ///
    /// Compute the mean gradient, then return the negative of its variance
    /// (average squared deviation of coordinates from their mean).
    /// Lower variance in the mean = gradients agree more = better quality.
    fn negative_variance(&self, gradients: &[&Gradient]) -> f64 {
        let dim = gradients[0].dim();
        let n = gradients.len() as f64;

        // Compute element-wise mean.
        let mut mean = vec![0.0f64; dim];
        for g in gradients {
            for (i, &v) in g.values.iter().enumerate() {
                mean[i] += v as f64;
            }
        }
        for m in &mut mean {
            *m /= n;
        }

        // Variance of the mean gradient coordinates.
        let grand_mean: f64 = mean.iter().sum::<f64>() / dim as f64;
        let var: f64 = mean.iter().map(|&m| (m - grand_mean).powi(2)).sum::<f64>() / dim as f64;

        // We want *agreement*: when all gradients point the same way,
        // the per-node deviations from the mean are small. Compute
        // the average per-coordinate variance across nodes.
        let mut coord_var_sum = 0.0f64;
        for d in 0..dim {
            let coord_mean = mean[d];
            let coord_var: f64 = gradients
                .iter()
                .map(|g| {
                    let diff = g.values[d] as f64 - coord_mean;
                    diff * diff
                })
                .sum::<f64>()
                / n;
            coord_var_sum += coord_var;
        }
        let avg_coord_var = coord_var_sum / dim as f64;

        // Negative variance: higher (less negative) is better.
        let _ = var; // unused — we use per-coordinate variance across nodes
        -avg_coord_var
    }

    /// Quality = average cosine similarity of each gradient to the coalition mean.
    fn cosine_to_mean(&self, gradients: &[&Gradient]) -> f64 {
        let dim = gradients[0].dim();
        let n = gradients.len() as f64;

        // Compute element-wise mean.
        let mut mean = vec![0.0f32; dim];
        for g in gradients {
            for (i, &v) in g.values.iter().enumerate() {
                mean[i] += v / n as f32;
            }
        }

        // Average cosine similarity to mean.
        let sum_cos: f64 = gradients
            .iter()
            .map(|g| cosine_similarity(&g.values, &mean))
            .sum();

        sum_cos / n
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Gradient;

    fn grad(id: &str, values: Vec<f32>) -> Gradient {
        Gradient::new(id, values, 0)
    }

    /// 5 honest (similar direction) + 1 Byzantine (opposite) ->
    /// Byzantine has negative Shapley value.
    #[test]
    fn test_byzantine_has_negative_shapley() {
        let base: Vec<f32> = (0..50).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut gradients: Vec<Gradient> = (0..5)
            .map(|i| {
                let vals: Vec<f32> = base
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| v + 0.01 * ((i * 7 + j * 3) % 13) as f32)
                    .collect();
                grad(&format!("honest-{}", i), vals)
            })
            .collect();

        // Byzantine: opposite direction, large magnitude
        let byz_vals: Vec<f32> = base.iter().map(|&v| -v * 5.0).collect();
        gradients.push(grad("byzantine", byz_vals));

        let detector = ShapleyDetector::new(ShapleyConfig {
            threshold: -0.1,
            quality_metric: QualityMetric::NegativeVariance,
        });
        let result = detector.detect(&gradients).unwrap();

        // Byzantine should be in the suspicious list.
        assert!(
            result.suspicious.contains(&"byzantine".to_string()),
            "Byzantine node should be suspicious. Values: {:?}",
            result.values
        );

        // Byzantine should have the lowest Shapley value.
        let byz_sv = result
            .values
            .iter()
            .find(|(id, _)| id == "byzantine")
            .unwrap()
            .1;
        for (id, sv) in &result.values {
            if id != "byzantine" {
                assert!(
                    byz_sv <= *sv,
                    "Byzantine Shapley ({}) should be <= honest {} ({})",
                    byz_sv,
                    id,
                    sv
                );
            }
        }
    }

    /// All identical gradients -> all have equal (near-zero) Shapley values.
    #[test]
    fn test_all_identical_equal_shapley() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gradients: Vec<Gradient> = (0..5)
            .map(|i| grad(&format!("node-{}", i), vals.clone()))
            .collect();

        let detector = ShapleyDetector::with_defaults();
        let result = detector.detect(&gradients).unwrap();

        // All Shapley values should be equal.
        let first_sv = result.values[0].1;
        for (id, sv) in &result.values {
            assert!(
                (sv - first_sv).abs() < 1e-9,
                "Node {} Shapley {} differs from first {}",
                id,
                sv,
                first_sv
            );
        }

        // With identical gradients removing any one doesn't change quality,
        // so Shapley should be ~0. None should be suspicious with default threshold -0.1.
        assert!(
            result.suspicious.is_empty(),
            "No node should be suspicious when all are identical"
        );
    }

    /// Single gradient -> Shapley = quality of that gradient.
    #[test]
    fn test_single_gradient() {
        let g = grad("only", vec![1.0, 2.0, 3.0]);
        let detector = ShapleyDetector::with_defaults();
        let result = detector.detect(&[g]).unwrap();

        assert_eq!(result.values.len(), 1);
        assert_eq!(result.values[0].0, "only");
    }

    /// Serde roundtrip for ShapleyConfig.
    #[test]
    fn test_config_serde_roundtrip() {
        let cfg = ShapleyConfig {
            threshold: -0.05,
            quality_metric: QualityMetric::CosineToMean,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: ShapleyConfig = serde_json::from_str(&json).unwrap();
        assert!((cfg.threshold - cfg2.threshold).abs() < 1e-12);
        assert_eq!(cfg.quality_metric, cfg2.quality_metric);
    }

    /// CosineToMean metric also detects Byzantine.
    #[test]
    fn test_cosine_metric_detects_byzantine() {
        let base: Vec<f32> = (0..20).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let mut gradients: Vec<Gradient> = (0..5)
            .map(|i| {
                let vals: Vec<f32> = base
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| v + 0.005 * ((i * 3 + j) % 7) as f32)
                    .collect();
                grad(&format!("h-{}", i), vals)
            })
            .collect();

        let byz_vals: Vec<f32> = base.iter().map(|&v| -v * 3.0).collect();
        gradients.push(grad("byz", byz_vals));

        let detector = ShapleyDetector::new(ShapleyConfig {
            threshold: -0.01,
            quality_metric: QualityMetric::CosineToMean,
        });
        let result = detector.detect(&gradients).unwrap();

        assert!(
            result.suspicious.contains(&"byz".to_string()),
            "CosineToMean should also detect Byzantine. Values: {:?}",
            result.values
        );
    }
}
