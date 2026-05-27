// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Visual Cortex: Hierarchical Visual Processing
//!
//! Simulates visual cortex processing with hierarchical feature extraction
//! and attention mechanisms.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use symthaea_core::hdc::ContinuousHV;

/// Configuration for visual cortex
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualCortexConfig {
    /// Embedding dimension
    pub dimension: usize,
    /// Number of processing layers
    pub layers: usize,
    /// Receptive field size
    pub receptive_field: usize,
    /// Use attention
    pub use_attention: bool,
}

impl Default for VisualCortexConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            layers: 6,
            receptive_field: 3,
            use_attention: true,
        }
    }
}

/// Feature extraction result
#[derive(Debug, Clone)]
pub struct FeatureExtractionResult {
    /// Extracted features per layer
    pub layer_features: Vec<ContinuousHV>,
    /// Final aggregated features
    pub final_features: ContinuousHV,
    /// Attention maps (if enabled)
    pub attention_maps: Option<Vec<Vec<f32>>>,
    /// Processing stats
    pub processing_time_ms: f64,
}

/// The visual cortex system
#[derive(Debug)]
pub struct VisualCortex {
    /// Configuration
    config: VisualCortexConfig,
    /// Layer filters
    layer_filters: Vec<Vec<ContinuousHV>>,
    /// Attention weights
    attention_weights: Vec<f32>,
    /// Statistics
    stats: VisualCortexStats,
}

/// Statistics for visual cortex
#[derive(Debug, Clone, Default)]
pub struct VisualCortexStats {
    /// Total processing operations
    pub total_operations: u64,
    /// Average processing time
    pub avg_processing_time_ms: f64,
}

impl VisualCortex {
    /// Create a new visual cortex
    pub fn new(config: VisualCortexConfig) -> Self {
        let dim = config.dimension;

        // Initialize layer filters
        let mut layer_filters = Vec::with_capacity(config.layers);
        for layer in 0..config.layers {
            let num_filters = 2usize.pow((layer + 2) as u32).min(64);
            let filters: Vec<ContinuousHV> = (0..num_filters)
                .map(|i| ContinuousHV::random(dim, (layer * 1000 + i) as u64))
                .collect();
            layer_filters.push(filters);
        }

        // Initialize attention weights
        let attention_weights = vec![1.0 / config.layers as f32; config.layers];

        Self {
            config,
            layer_filters,
            attention_weights,
            stats: VisualCortexStats::default(),
        }
    }

    /// Create a visual cortex with deterministic RNG from a genesis seed.
    pub fn from_genesis(
        config: VisualCortexConfig,
        genesis: &symthaea_core::genesis::GenesisSeed,
        label: &str,
    ) -> Self {
        let dim = config.dimension;
        let mut rng = genesis.domain(&format!("{label}::visual_cortex"));

        let mut layer_filters = Vec::with_capacity(config.layers);
        for _layer in 0..config.layers {
            let num_filters = 2usize.pow((_layer + 2) as u32).min(64);
            let filters: Vec<ContinuousHV> = (0..num_filters)
                .map(|_| {
                    let seed: u64 = rand::Rng::r#gen(&mut rng);
                    ContinuousHV::random(dim, seed)
                })
                .collect();
            layer_filters.push(filters);
        }

        let attention_weights = vec![1.0 / config.layers as f32; config.layers];

        Self {
            config,
            layer_filters,
            attention_weights,
            stats: VisualCortexStats::default(),
        }
    }

    /// Process visual input through the cortex
    pub fn process(&mut self, input: &ContinuousHV) -> FeatureExtractionResult {
        let start = std::time::Instant::now();
        self.stats.total_operations += 1;

        let mut layer_features = Vec::with_capacity(self.config.layers);
        let mut current = input.clone();

        // Process through each layer
        for (layer_idx, filters) in self.layer_filters.iter().enumerate() {
            let layer_output = self.process_layer(&current, filters, layer_idx);
            layer_features.push(layer_output.clone());
            current = layer_output;
        }

        // Update attention weights based on layer feature energies
        if self.config.use_attention && !layer_features.is_empty() {
            let energies: Vec<f32> = layer_features
                .iter()
                .map(|f| f.as_slice().iter().map(|v| v * v).sum::<f32>().sqrt())
                .collect();
            let total_energy: f32 = energies.iter().sum();
            if total_energy > 0.0 {
                for (w, e) in self.attention_weights.iter_mut().zip(energies.iter()) {
                    // Exponential moving average: 80% old + 20% new
                    *w = 0.8 * *w + 0.2 * (e / total_energy);
                }
            }
        }

        // Aggregate features
        let final_features = self.aggregate_features(&layer_features);

        // Generate attention maps if enabled
        let attention_maps = if self.config.use_attention {
            Some(self.generate_attention_maps(&layer_features))
        } else {
            None
        };

        let processing_time = start.elapsed().as_secs_f64() * 1000.0;
        self.update_avg_time(processing_time);

        FeatureExtractionResult {
            layer_features,
            final_features,
            attention_maps,
            processing_time_ms: processing_time,
        }
    }

    /// Process through a single layer
    fn process_layer(
        &self,
        input: &ContinuousHV,
        filters: &[ContinuousHV],
        _layer_idx: usize,
    ) -> ContinuousHV {
        // Apply filters and aggregate
        let filter_outputs: Vec<ContinuousHV> =
            filters.iter().map(|filter| input.bind(filter)).collect();

        if filter_outputs.is_empty() {
            input.clone()
        } else {
            ContinuousHV::bundle_owned(&filter_outputs)
        }
    }

    /// Aggregate features from all layers
    fn aggregate_features(&self, layer_features: &[ContinuousHV]) -> ContinuousHV {
        if layer_features.is_empty() {
            return ContinuousHV::zero(self.config.dimension);
        }

        // Weighted sum based on attention weights
        let mut result = ContinuousHV::zero(self.config.dimension);
        for (feat, &weight) in layer_features.iter().zip(self.attention_weights.iter()) {
            let weighted = feat.scale(weight);
            result = ContinuousHV::bundle_owned(&[result, weighted]);
        }

        result
    }

    /// Generate attention maps
    fn generate_attention_maps(&self, layer_features: &[ContinuousHV]) -> Vec<Vec<f32>> {
        layer_features
            .iter()
            .map(|feat| {
                // Simple attention based on feature magnitudes
                let slice = feat.as_slice();
                let max_val = slice.iter().map(|v| v.abs()).fold(0.0f32, |a, b| a.max(b));
                if max_val > 0.0 {
                    slice.iter().map(|v| v.abs() / max_val).collect()
                } else {
                    vec![0.0; slice.len()]
                }
            })
            .collect()
    }

    /// Update average processing time
    fn update_avg_time(&mut self, time: f64) {
        let n = self.stats.total_operations as f64;
        self.stats.avg_processing_time_ms =
            (self.stats.avg_processing_time_ms * (n - 1.0) + time) / n;
    }

    /// Get statistics
    pub fn stats(&self) -> &VisualCortexStats {
        &self.stats
    }
}

impl Default for VisualCortex {
    fn default() -> Self {
        Self::new(VisualCortexConfig::default())
    }
}

/// Visual feature extractor component
#[derive(Debug)]
pub struct VisualFeatureExtractor {
    /// Dimension
    dimension: usize,
    /// Feature bases
    feature_bases: HashMap<String, ContinuousHV>,
}

impl VisualFeatureExtractor {
    /// Create a new feature extractor
    pub fn new(dimension: usize) -> Self {
        let mut feature_bases = HashMap::new();

        // Initialize bases for common visual features
        for (i, feature) in ["edge", "color", "texture", "shape", "motion", "depth"]
            .iter()
            .enumerate()
        {
            feature_bases.insert(
                feature.to_string(),
                ContinuousHV::random(dimension, (i + 1) as u64),
            );
        }

        Self {
            dimension,
            feature_bases,
        }
    }

    /// Extract features from raw visual data
    pub fn extract(&self, data: &[u8]) -> HashMap<String, ContinuousHV> {
        let mut features = HashMap::new();

        // Extract pseudo-features (would use real CV in production)
        for (name, basis) in &self.feature_bases {
            let feature = self.extract_feature_type(data, basis);
            features.insert(name.clone(), feature);
        }

        features
    }

    /// Extract a specific feature type
    fn extract_feature_type(&self, data: &[u8], basis: &ContinuousHV) -> ContinuousHV {
        // Simple hash-based feature extraction
        let mut values = vec![0.0f32; self.dimension];

        for (i, &byte) in data.iter().enumerate() {
            let idx = (byte as usize + i) % self.dimension;
            values[idx] += 1.0;
        }

        // Normalize and bind with basis
        let magnitude: f32 = values.iter().map(|v| v * v).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in values.iter_mut() {
                *v /= magnitude;
            }
        }

        let data_hv = ContinuousHV::from_slice(&values);
        data_hv.bind(basis)
    }
}

impl Default for VisualFeatureExtractor {
    fn default() -> Self {
        Self::new(512)
    }
}

/// Visual attention mechanism
#[derive(Debug)]
pub struct VisualAttention {
    /// Dimension
    dimension: usize,
    /// Query transform
    query_transform: ContinuousHV,
    /// Key transform
    key_transform: ContinuousHV,
    /// Value transform
    value_transform: ContinuousHV,
}

impl VisualAttention {
    /// Create a new visual attention mechanism
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            query_transform: ContinuousHV::random(dimension, 100),
            key_transform: ContinuousHV::random(dimension, 101),
            value_transform: ContinuousHV::random(dimension, 102),
        }
    }

    /// Apply attention to features
    pub fn attend(
        &self,
        query: &ContinuousHV,
        keys: &[ContinuousHV],
        values: &[ContinuousHV],
    ) -> ContinuousHV {
        if keys.is_empty() || values.is_empty() || keys.len() != values.len() {
            return query.clone();
        }

        // Transform query
        let q = query.bind(&self.query_transform);

        // Calculate attention scores
        let scores: Vec<f32> = keys
            .iter()
            .map(|k| {
                let k_transformed = k.bind(&self.key_transform);
                q.similarity(&k_transformed)
            })
            .collect();

        // Softmax
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        let attention_weights: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

        // Weighted sum of values
        let mut result = ContinuousHV::zero(self.dimension);
        for (v, &weight) in values.iter().zip(attention_weights.iter()) {
            let v_transformed = v.bind(&self.value_transform);
            let weighted = v_transformed.scale(weight);
            result = ContinuousHV::bundle(&[&result, &weighted]);
        }

        result
    }

    /// Get attention weights for visualization
    pub fn get_attention_weights(&self, query: &ContinuousHV, keys: &[ContinuousHV]) -> Vec<f32> {
        if keys.is_empty() {
            return Vec::new();
        }

        let q = query.bind(&self.query_transform);

        let scores: Vec<f32> = keys
            .iter()
            .map(|k| {
                let k_transformed = k.bind(&self.key_transform);
                q.similarity(&k_transformed)
            })
            .collect();

        // Softmax
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_scores: Vec<f32> = scores.iter().map(|s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();

        exp_scores.iter().map(|e| e / sum_exp).collect()
    }
}

impl Default for VisualAttention {
    fn default() -> Self {
        Self::new(512)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visual_cortex_creation() {
        let cortex = VisualCortex::default();
        assert_eq!(cortex.stats.total_operations, 0);
    }

    #[test]
    fn test_visual_processing() {
        let mut cortex = VisualCortex::default();
        let input = ContinuousHV::random(512, 1);

        let result = cortex.process(&input);
        assert!(!result.layer_features.is_empty());
        assert_eq!(result.final_features.dim(), 512);
    }

    #[test]
    fn test_feature_extractor() {
        let extractor = VisualFeatureExtractor::default();
        let data = vec![0u8; 1024];

        let features = extractor.extract(&data);
        assert!(!features.is_empty());
    }

    #[test]
    fn test_visual_attention() {
        let attention = VisualAttention::default();

        let query = ContinuousHV::random(512, 10);
        let keys = vec![ContinuousHV::random(512, 11), ContinuousHV::random(512, 12)];
        let values = vec![ContinuousHV::random(512, 13), ContinuousHV::random(512, 14)];

        let result = attention.attend(&query, &keys, &values);
        assert_eq!(result.dim(), 512);
    }

    #[test]
    fn test_attention_weights() {
        let attention = VisualAttention::default();

        let query = ContinuousHV::random(512, 20);
        let keys = vec![
            ContinuousHV::random(512, 21),
            ContinuousHV::random(512, 22),
            ContinuousHV::random(512, 23),
        ];

        let weights = attention.get_attention_weights(&query, &keys);
        assert_eq!(weights.len(), 3);

        // Weights should sum to 1
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }
}
