// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Terrain Federated Learning Types
//!
//! Rust types and utilities for federated learning of terrain traversability
//! models across lunar rovers. This module provides the wire-compatible types
//! needed for gradient exchange via Holochain DHT; the actual training loop
//! lives in the 0TML Python/Rust stack.
//!
//! ## Design
//!
//! - **`TraversabilityModel`**: a lightweight linear model that each rover
//!   trains locally on its own terrain observations.
//! - **`GradientUpdate`**: the delta a rover publishes to the DHT after a
//!   local training step.
//! - **`FederatedAggregator`**: coordinate-wise median aggregation (Byzantine-
//!   robust) that produces an `AggregatedGradient` once enough updates arrive.
//! - **`IsruYieldModel`**: predicts oxygen yield from regolith type, temperature,
//!   and energy input for ISRU planning.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Terrain features
// ============================================================================

/// Terrain features observed by a lunar rover. Each feature is a scalar
/// measurement normalized to a domain-specific range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TerrainFeature {
    /// Surface slope in degrees.
    Slope,
    /// Surface roughness index (0 = smooth, 1 = impassable).
    Roughness,
    /// Dust/regolith depth in centimeters.
    DustDepth,
    /// Boulder density (boulders per square meter).
    BoulderDensity,
    /// Distance to nearest crater rim (meters, inverse-scaled).
    CraterProximity,
    /// Solar exposure factor (0 = permanent shadow, 1 = full illumination).
    SolarExposure,
    /// Thermal gradient (K/m).
    ThermalGradient,
    /// Regolith compaction (0 = loose, 1 = fully compacted).
    RegolithCompaction,
}

impl TerrainFeature {
    /// All feature variants for iteration.
    pub const ALL: [TerrainFeature; 8] = [
        TerrainFeature::Slope,
        TerrainFeature::Roughness,
        TerrainFeature::DustDepth,
        TerrainFeature::BoulderDensity,
        TerrainFeature::CraterProximity,
        TerrainFeature::SolarExposure,
        TerrainFeature::ThermalGradient,
        TerrainFeature::RegolithCompaction,
    ];
}

// ============================================================================
// Observation
// ============================================================================

/// A single terrain observation from a rover at a specific location.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerrainObservation {
    /// (latitude, longitude) in decimal degrees.
    pub location: (f64, f64),
    /// Feature measurements.
    pub features: Vec<(TerrainFeature, f64)>,
    /// Ground-truth traversability score (0.0 = impassable, 1.0 = easy).
    pub traversability_score: f64,
    /// Observer agent identifier.
    pub observer_id: String,
    /// Observation confidence (0.0–1.0).
    pub confidence: f64,
    /// Timestamp (microseconds since epoch).
    pub timestamp_us: u64,
}

// ============================================================================
// Traversability model
// ============================================================================

/// A lightweight linear traversability model: `score = clamp(sum(w_i * x_i) + bias, 0, 1)`.
/// Each rover trains one locally and publishes gradient updates to the DHT.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraversabilityModel {
    /// Unique model identifier.
    pub model_id: String,
    /// Model version (incremented on each weight update).
    pub version: u32,
    /// Learned feature weights.
    pub feature_weights: Vec<(TerrainFeature, f64)>,
    /// Bias term.
    pub bias: f64,
    /// Number of training samples seen by this model.
    pub training_samples: u64,
    /// Current accuracy estimate (0.0–1.0).
    pub accuracy: f64,
    /// Last update timestamp (microseconds since epoch).
    pub last_updated_us: u64,
}

impl TraversabilityModel {
    /// Create a new model with zero weights.
    pub fn new(model_id: String) -> Self {
        Self {
            model_id,
            version: 0,
            feature_weights: TerrainFeature::ALL
                .iter()
                .map(|&f| (f, 0.0))
                .collect(),
            bias: 0.0,
            training_samples: 0,
            accuracy: 0.0,
            last_updated_us: 0,
        }
    }

    /// Predict traversability for a set of feature values.
    /// Linear model clamped to [0.0, 1.0].
    pub fn predict(&self, features: &[(TerrainFeature, f64)]) -> f64 {
        let weight_map: HashMap<TerrainFeature, f64> =
            self.feature_weights.iter().copied().collect();
        let sum: f64 = features
            .iter()
            .map(|(feat, val)| weight_map.get(feat).copied().unwrap_or(0.0) * val)
            .sum();
        (sum + self.bias).clamp(0.0, 1.0)
    }

    /// Replace weights and bias with new values from an aggregated gradient
    /// application.
    pub fn update_weights(
        &mut self,
        new_weights: &[(TerrainFeature, f64)],
        new_bias: f64,
        new_samples: u64,
    ) {
        self.feature_weights = new_weights.to_vec();
        self.bias = new_bias;
        self.training_samples += new_samples;
        self.version += 1;
    }
}

// ============================================================================
// Gradient exchange types
// ============================================================================

/// A gradient update published by a rover after local training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientUpdate {
    /// Model this gradient applies to.
    pub model_id: String,
    /// Rover that computed this gradient.
    pub source_agent: String,
    /// Per-feature gradient values.
    pub feature_gradients: Vec<(TerrainFeature, f64)>,
    /// Bias gradient.
    pub bias_gradient: f64,
    /// Number of local samples used.
    pub n_samples: u64,
    /// Timestamp (microseconds since epoch).
    pub timestamp_us: u64,
}

/// Result of aggregating multiple gradient updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedGradient {
    /// Coordinate-wise median gradient per feature.
    pub feature_gradients: Vec<(TerrainFeature, f64)>,
    /// Median bias gradient.
    pub bias_gradient: f64,
    /// Number of contributing rovers.
    pub n_contributors: usize,
    /// Number of updates rejected as Byzantine outliers.
    pub n_rejected: usize,
}

// ============================================================================
// Federated aggregator
// ============================================================================

/// Byzantine-robust federated aggregator using coordinate-wise median.
/// Rejects gradients whose magnitude is more than `byzantine_threshold` standard
/// deviations from the median on any coordinate.
#[derive(Debug, Clone)]
pub struct FederatedAggregator {
    /// Pending gradient updates awaiting aggregation.
    pending_gradients: Vec<GradientUpdate>,
    /// Minimum number of updates required before aggregation proceeds.
    pub min_updates_for_round: usize,
    /// Standard-deviation multiplier for Byzantine rejection (default 3.0).
    pub byzantine_threshold: f64,
}

impl FederatedAggregator {
    /// Create a new aggregator.
    pub fn new() -> Self {
        Self {
            pending_gradients: Vec::new(),
            min_updates_for_round: 2,
            byzantine_threshold: 3.0,
        }
    }

    /// Submit a gradient update from a rover.
    pub fn submit_gradient(&mut self, update: GradientUpdate) {
        self.pending_gradients.push(update);
    }

    /// Attempt to aggregate pending gradients. Returns `None` if fewer than
    /// `min_updates_for_round` updates are available. Uses coordinate-wise
    /// median for robustness against Byzantine faults, then rejects any
    /// update whose gradient on any feature is more than
    /// `byzantine_threshold` standard deviations from the median.
    pub fn aggregate(&mut self) -> Option<AggregatedGradient> {
        if self.pending_gradients.len() < self.min_updates_for_round {
            return None;
        }

        let all_features: Vec<TerrainFeature> = TerrainFeature::ALL.to_vec();
        let n = self.pending_gradients.len();

        // Collect per-feature gradient vectors.
        let mut feature_vecs: HashMap<TerrainFeature, Vec<f64>> = HashMap::new();
        let mut bias_vec: Vec<f64> = Vec::with_capacity(n);

        for feat in &all_features {
            feature_vecs.insert(*feat, Vec::with_capacity(n));
        }

        for update in &self.pending_gradients {
            let grad_map: HashMap<TerrainFeature, f64> =
                update.feature_gradients.iter().copied().collect();
            for feat in &all_features {
                feature_vecs
                    .get_mut(feat)
                    .unwrap()
                    .push(grad_map.get(feat).copied().unwrap_or(0.0));
            }
            bias_vec.push(update.bias_gradient);
        }

        // Compute coordinate-wise medians.
        let medians: HashMap<TerrainFeature, f64> = feature_vecs
            .iter()
            .map(|(&feat, vals)| (feat, median_of(vals)))
            .collect();
        let bias_median = median_of(&bias_vec);

        // Compute MAD (Median Absolute Deviation) for robust Byzantine rejection.
        // MAD is not inflated by outliers, unlike standard deviation.
        // Scale factor 1.4826 converts MAD to equivalent standard deviation
        // for normally distributed data.
        let std_devs: HashMap<TerrainFeature, f64> = feature_vecs
            .iter()
            .map(|(&feat, vals)| {
                let med = medians[&feat];
                let mut abs_devs: Vec<f64> = vals.iter().map(|v| (v - med).abs()).collect();
                abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
                let mad = median_of(&abs_devs);
                (feat, mad * 1.4826) // MAD → σ equivalent
            })
            .collect();

        let bias_std = {
            let mut abs_devs: Vec<f64> = bias_vec.iter().map(|v| (v - bias_median).abs()).collect();
            abs_devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
            median_of(&abs_devs) * 1.4826
        };

        // Identify Byzantine updates.
        let mut rejected = vec![false; n];
        for (i, update) in self.pending_gradients.iter().enumerate() {
            let grad_map: HashMap<TerrainFeature, f64> =
                update.feature_gradients.iter().copied().collect();
            for feat in &all_features {
                let val = grad_map.get(feat).copied().unwrap_or(0.0);
                let med = medians[&feat];
                let sd = std_devs[&feat];
                if sd > 1e-12 && (val - med).abs() > self.byzantine_threshold * sd {
                    rejected[i] = true;
                    break;
                }
            }
            if !rejected[i] && bias_std > 1e-12 {
                if (update.bias_gradient - bias_median).abs() > self.byzantine_threshold * bias_std
                {
                    rejected[i] = true;
                }
            }
        }

        let n_rejected = rejected.iter().filter(|&&r| r).count();

        // Recompute medians from non-rejected updates only.
        let mut clean_feature_vecs: HashMap<TerrainFeature, Vec<f64>> = HashMap::new();
        let mut clean_bias_vec: Vec<f64> = Vec::new();
        for feat in &all_features {
            clean_feature_vecs.insert(*feat, Vec::new());
        }

        for (i, update) in self.pending_gradients.iter().enumerate() {
            if rejected[i] {
                continue;
            }
            let grad_map: HashMap<TerrainFeature, f64> =
                update.feature_gradients.iter().copied().collect();
            for feat in &all_features {
                clean_feature_vecs
                    .get_mut(feat)
                    .unwrap()
                    .push(grad_map.get(feat).copied().unwrap_or(0.0));
            }
            clean_bias_vec.push(update.bias_gradient);
        }

        // If all were rejected, fall back to original medians.
        let final_gradients: Vec<(TerrainFeature, f64)> = if clean_bias_vec.is_empty() {
            all_features
                .iter()
                .map(|&f| (f, medians[&f]))
                .collect()
        } else {
            all_features
                .iter()
                .map(|&f| (f, median_of(clean_feature_vecs.get(&f).unwrap())))
                .collect()
        };

        let final_bias = if clean_bias_vec.is_empty() {
            bias_median
        } else {
            median_of(&clean_bias_vec)
        };

        Some(AggregatedGradient {
            feature_gradients: final_gradients,
            bias_gradient: final_bias,
            n_contributors: n - n_rejected,
            n_rejected,
        })
    }

    /// Clear all pending gradients (typically called after aggregation).
    pub fn clear(&mut self) {
        self.pending_gradients.clear();
    }

    /// Number of pending gradient updates.
    pub fn pending_count(&self) -> usize {
        self.pending_gradients.len()
    }
}

impl Default for FederatedAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute the median of a slice. Returns 0.0 for empty slices.
fn median_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

// ============================================================================
// ISRU yield model
// ============================================================================

/// Predicts oxygen yield from in-situ resource utilization (ISRU) given
/// regolith type, temperature, and energy input. Used for planning ISRU
/// operations and sharing learned models across colony sites.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsruYieldModel {
    /// Model identifier.
    pub model_id: String,
    /// Learned weights per regolith type (e.g., "highland", "mare", "polar").
    pub regolith_type_weights: HashMap<String, f64>,
    /// Temperature scaling factor (yield increases with temperature up to a point).
    pub temperature_factor: f64,
    /// Energy efficiency factor (yield per watt-hour).
    pub energy_factor: f64,
}

impl IsruYieldModel {
    /// Create a new ISRU model with default factors.
    pub fn new(model_id: String) -> Self {
        Self {
            model_id,
            regolith_type_weights: HashMap::new(),
            temperature_factor: 1.0,
            energy_factor: 1.0,
        }
    }

    /// Predict O2 yield in kilograms.
    ///
    /// Formula: `regolith_weight * (temperature_k / 1000.0) * temperature_factor * energy_wh * energy_factor`
    ///
    /// Unknown regolith types default to weight 0.5.
    pub fn predict_yield(&self, regolith_type: &str, temperature_k: f64, energy_wh: f64) -> f64 {
        let base_weight = self
            .regolith_type_weights
            .get(regolith_type)
            .copied()
            .unwrap_or(0.5);
        let temp_scaled = (temperature_k / 1000.0) * self.temperature_factor;
        let energy_scaled = energy_wh * self.energy_factor;
        (base_weight * temp_scaled * energy_scaled).max(0.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_features() -> Vec<(TerrainFeature, f64)> {
        vec![
            (TerrainFeature::Slope, 0.3),
            (TerrainFeature::Roughness, 0.5),
            (TerrainFeature::DustDepth, 0.2),
            (TerrainFeature::SolarExposure, 0.8),
        ]
    }

    fn make_gradient(agent: &str, slope_grad: f64, bias_grad: f64) -> GradientUpdate {
        GradientUpdate {
            model_id: "model-1".to_string(),
            source_agent: agent.to_string(),
            feature_gradients: vec![
                (TerrainFeature::Slope, slope_grad),
                (TerrainFeature::Roughness, 0.01),
                (TerrainFeature::DustDepth, -0.005),
                (TerrainFeature::BoulderDensity, 0.0),
                (TerrainFeature::CraterProximity, 0.0),
                (TerrainFeature::SolarExposure, 0.02),
                (TerrainFeature::ThermalGradient, 0.0),
                (TerrainFeature::RegolithCompaction, 0.0),
            ],
            bias_gradient: bias_grad,
            n_samples: 100,
            timestamp_us: 1_000_000,
        }
    }

    #[test]
    fn test_model_prediction() {
        let mut model = TraversabilityModel::new("test-model".to_string());
        // Set some weights.
        model.feature_weights = vec![
            (TerrainFeature::Slope, -0.5),
            (TerrainFeature::Roughness, -0.3),
            (TerrainFeature::DustDepth, -0.1),
            (TerrainFeature::SolarExposure, 0.4),
        ];
        model.bias = 0.5;

        let score = model.predict(&sample_features());
        // -0.5*0.3 + -0.3*0.5 + -0.1*0.2 + 0.4*0.8 + 0.5
        // = -0.15 + -0.15 + -0.02 + 0.32 + 0.5 = 0.5
        assert!((score - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_prediction_clamped() {
        let mut model = TraversabilityModel::new("test".to_string());
        model.feature_weights = vec![(TerrainFeature::Slope, 10.0)];
        model.bias = 5.0;

        let score = model.predict(&[(TerrainFeature::Slope, 1.0)]);
        assert_eq!(score, 1.0); // Clamped to max.

        model.feature_weights = vec![(TerrainFeature::Slope, -10.0)];
        model.bias = -5.0;
        let score = model.predict(&[(TerrainFeature::Slope, 1.0)]);
        assert_eq!(score, 0.0); // Clamped to min.
    }

    #[test]
    fn test_weight_update() {
        let mut model = TraversabilityModel::new("test".to_string());
        assert_eq!(model.version, 0);
        assert_eq!(model.training_samples, 0);

        let new_weights = vec![
            (TerrainFeature::Slope, -0.4),
            (TerrainFeature::Roughness, -0.2),
        ];
        model.update_weights(&new_weights, 0.3, 50);

        assert_eq!(model.version, 1);
        assert_eq!(model.training_samples, 50);
        assert_eq!(model.bias, 0.3);
        assert_eq!(model.feature_weights.len(), 2);
    }

    #[test]
    fn test_gradient_aggregation_median() {
        let mut agg = FederatedAggregator::new();
        agg.min_updates_for_round = 3;

        agg.submit_gradient(make_gradient("rover-1", 0.10, 0.01));
        agg.submit_gradient(make_gradient("rover-2", 0.12, 0.02));
        agg.submit_gradient(make_gradient("rover-3", 0.08, 0.015));

        let result = agg.aggregate().expect("should aggregate with 3 updates");
        assert_eq!(result.n_contributors, 3);
        assert_eq!(result.n_rejected, 0);

        // Median of [0.10, 0.12, 0.08] = 0.10 (middle value).
        let slope_grad = result
            .feature_gradients
            .iter()
            .find(|(f, _)| *f == TerrainFeature::Slope)
            .unwrap()
            .1;
        assert!((slope_grad - 0.10).abs() < 1e-9);

        // Median of [0.01, 0.02, 0.015] = 0.015.
        assert!((result.bias_gradient - 0.015).abs() < 1e-9);
    }

    #[test]
    fn test_byzantine_rejection() {
        let mut agg = FederatedAggregator::new();
        agg.min_updates_for_round = 3;
        agg.byzantine_threshold = 3.0;

        // Three honest rovers with similar gradients.
        agg.submit_gradient(make_gradient("rover-1", 0.10, 0.01));
        agg.submit_gradient(make_gradient("rover-2", 0.12, 0.02));
        agg.submit_gradient(make_gradient("rover-3", 0.11, 0.015));
        // One Byzantine rover with a wildly different gradient.
        agg.submit_gradient(make_gradient("byzantine", 50.0, 50.0));

        let result = agg.aggregate().expect("should aggregate");
        // The Byzantine update should be rejected.
        assert_eq!(result.n_rejected, 1);
        assert_eq!(result.n_contributors, 3);

        // Median of honest gradients only.
        let slope_grad = result
            .feature_gradients
            .iter()
            .find(|(f, _)| *f == TerrainFeature::Slope)
            .unwrap()
            .1;
        assert!((slope_grad - 0.11).abs() < 1e-9);
    }

    #[test]
    fn test_insufficient_updates_returns_none() {
        let mut agg = FederatedAggregator::new();
        agg.min_updates_for_round = 3;

        agg.submit_gradient(make_gradient("rover-1", 0.10, 0.01));
        assert!(agg.aggregate().is_none());
        assert_eq!(agg.pending_count(), 1);

        agg.submit_gradient(make_gradient("rover-2", 0.12, 0.02));
        assert!(agg.aggregate().is_none());
        assert_eq!(agg.pending_count(), 2);
    }

    #[test]
    fn test_terrain_feature_classification() {
        // Verify all 8 features are distinct and enumerable.
        assert_eq!(TerrainFeature::ALL.len(), 8);
        let mut seen = std::collections::HashSet::new();
        for f in &TerrainFeature::ALL {
            assert!(seen.insert(*f), "duplicate feature: {:?}", f);
        }
    }

    #[test]
    fn test_isru_yield_prediction() {
        let mut model = IsruYieldModel::new("isru-1".to_string());
        model
            .regolith_type_weights
            .insert("highland".to_string(), 0.8);
        model
            .regolith_type_weights
            .insert("mare".to_string(), 0.4);
        model.temperature_factor = 1.2;
        model.energy_factor = 0.001;

        let yield_highland = model.predict_yield("highland", 500.0, 1000.0);
        // 0.8 * (500/1000) * 1.2 * 1000 * 0.001 = 0.8 * 0.5 * 1.2 * 1.0 = 0.48
        assert!((yield_highland - 0.48).abs() < 0.01);

        let yield_mare = model.predict_yield("mare", 500.0, 1000.0);
        assert!(yield_mare < yield_highland);

        // Unknown regolith type defaults to 0.5.
        let yield_unknown = model.predict_yield("polar_ice", 500.0, 1000.0);
        let expected = 0.5 * 0.5 * 1.2 * 1.0;
        assert!((yield_unknown - expected).abs() < 0.01);
    }

    #[test]
    fn test_empty_aggregator() {
        let mut agg = FederatedAggregator::new();
        assert_eq!(agg.pending_count(), 0);
        assert!(agg.aggregate().is_none());

        agg.submit_gradient(make_gradient("rover-1", 0.1, 0.01));
        agg.submit_gradient(make_gradient("rover-2", 0.2, 0.02));
        assert_eq!(agg.pending_count(), 2);

        let _ = agg.aggregate(); // min_updates_for_round defaults to 2
        agg.clear();
        assert_eq!(agg.pending_count(), 0);
    }

    #[test]
    fn test_observation_roundtrip() {
        let obs = TerrainObservation {
            location: (-89.9, 45.0),
            features: sample_features(),
            traversability_score: 0.75,
            observer_id: "rover-1".to_string(),
            confidence: 0.9,
            timestamp_us: 1_000_000,
        };
        let json = serde_json::to_string(&obs).expect("serialize");
        let deser: TerrainObservation = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(deser.observer_id, "rover-1");
        assert!((deser.traversability_score - 0.75).abs() < 1e-9);
    }
}
