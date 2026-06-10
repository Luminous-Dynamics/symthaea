// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Hierarchical World Model
//!
//! A multi-level world model using CfC networks for continuous-time
//! dynamics at each level of abstraction.
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────┐
//! │                  HIERARCHICAL WORLD MODEL                   │
//! │                                                             │
//! │  Level 3 (Abstract):  Planning & Goals                      │
//! │       ↑↓                                                    │
//! │  Level 2 (Concepts):  Object & Event Representations        │
//! │       ↑↓                                                    │
//! │  Level 1 (Features):  Spatial & Temporal Features           │
//! │       ↑↓                                                    │
//! │  Level 0 (Sensory):   Raw Sensory Input                     │
//! └────────────────────────────────────────────────────────────┘
//! ```

use crate::dynamics::CrystalizedConcept;
use crate::dynamics::cfc::{CfCConfig, CfCNetwork, CfCNetworkConfig};
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use symthaea_core::genesis::GenesisSeed;

/// Configuration for the hierarchical world model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelConfig {
    /// Number of hierarchical levels
    pub num_levels: usize,

    /// Dimension at each level (ordered from lowest to highest)
    pub level_dims: Vec<usize>,

    /// Time scale multiplier at each level (higher levels = slower dynamics)
    pub time_scales: Vec<f32>,

    /// Whether to use bidirectional connections between levels
    pub bidirectional: bool,

    /// Maximum concepts to store
    pub max_concepts: usize,

    /// Prediction horizon steps
    pub prediction_horizon: usize,
}

impl Default for WorldModelConfig {
    fn default() -> Self {
        Self {
            num_levels: 4,
            level_dims: vec![64, 128, 256, 128],
            time_scales: vec![1.0, 2.0, 5.0, 10.0],
            bidirectional: true,
            max_concepts: 10000,
            prediction_horizon: 10,
        }
    }
}

/// A single layer in the world model hierarchy
#[derive(Debug, Clone)]
#[allow(dead_code)] // RESERVED(future): predictive coding layer fields
pub struct WorldModelLayer {
    /// Layer index (0 = lowest/sensory)
    level: usize,

    /// CfC network for temporal dynamics
    cfc: CfCNetwork,

    /// Current state representation
    state: Array1<f32>,

    /// Prediction of next state
    prediction: Array1<f32>,

    /// Prediction error from lower level
    prediction_error: Array1<f32>,

    /// Time scale for this level
    time_scale: f32,

    /// Statistics
    stats: LayerStats,
}

/// Statistics for a layer
#[derive(Debug, Clone, Default)]
pub struct LayerStats {
    /// Total updates
    pub updates: u64,

    /// Average prediction error
    pub avg_prediction_error: f32,

    /// Maximum prediction error seen
    pub max_prediction_error: f32,
}

impl WorldModelLayer {
    /// Create a new world model layer
    pub fn new(level: usize, input_dim: usize, hidden_dim: usize, time_scale: f32) -> Self {
        let cfc_config = CfCNetworkConfig {
            input_dim,
            hidden_dim,
            output_dim: hidden_dim,
            num_layers: 2,
            cell_config: CfCConfig {
                input_dim,
                hidden_dim,
                ..Default::default()
            },
            ..Default::default()
        };

        Self {
            level,
            cfc: CfCNetwork::new(cfc_config),
            state: Array1::zeros(hidden_dim),
            prediction: Array1::zeros(hidden_dim),
            prediction_error: Array1::zeros(hidden_dim),
            time_scale,
            stats: LayerStats::default(),
        }
    }

    /// Create a new world model layer with deterministic initialization from a genesis seed.
    pub fn from_genesis(
        level: usize,
        input_dim: usize,
        hidden_dim: usize,
        time_scale: f32,
        genesis: &GenesisSeed,
        label: &str,
    ) -> Self {
        let cfc_config = CfCNetworkConfig {
            input_dim,
            hidden_dim,
            output_dim: hidden_dim,
            num_layers: 2,
            cell_config: CfCConfig {
                input_dim,
                hidden_dim,
                ..Default::default()
            },
            ..Default::default()
        };

        Self {
            level,
            cfc: CfCNetwork::from_genesis(cfc_config, genesis, label),
            state: Array1::zeros(hidden_dim),
            prediction: Array1::zeros(hidden_dim),
            prediction_error: Array1::zeros(hidden_dim),
            time_scale,
            stats: LayerStats::default(),
        }
    }

    /// Update the layer with new input
    pub fn update(&mut self, input: &Array1<f32>, dt: f32) -> Array1<f32> {
        // Adjust time step by level's time scale
        let adjusted_dt = dt * self.time_scale;

        // Generate prediction error
        self.prediction_error = &self.state - input;

        // Forward through CfC
        let output = self.cfc.forward(input, adjusted_dt);

        // Update state
        self.state = output.clone();

        // Update statistics
        self.stats.updates += 1;
        let error_norm: f32 = self
            .prediction_error
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        // Use exponential moving average (EMA) instead of running average.
        // Running average (n-1)/n freezes when n > ~16M because (n-1)/n == 1.0 in f32.
        const EMA_ALPHA: f32 = 0.001;
        if self.stats.updates == 1 {
            self.stats.avg_prediction_error = error_norm;
        } else {
            self.stats.avg_prediction_error =
                (1.0 - EMA_ALPHA) * self.stats.avg_prediction_error + EMA_ALPHA * error_norm;
        }
        self.stats.max_prediction_error = self.stats.max_prediction_error.max(error_norm);

        output
    }

    /// Generate prediction for next state
    pub fn predict(&mut self, dt: f32) -> Array1<f32> {
        let adjusted_dt = dt * self.time_scale;

        // Use current state as input to predict next
        self.prediction = self.cfc.forward(&self.state, adjusted_dt);
        self.prediction.clone()
    }

    /// Get current state
    pub fn state(&self) -> &Array1<f32> {
        &self.state
    }

    /// Get prediction error
    pub fn prediction_error(&self) -> &Array1<f32> {
        &self.prediction_error
    }

    /// Get statistics
    pub fn stats(&self) -> &LayerStats {
        &self.stats
    }

    /// Reset the layer
    pub fn reset(&mut self) {
        self.cfc.reset();
        self.state.fill(0.0);
        self.prediction.fill(0.0);
        self.prediction_error.fill(0.0);
    }
}

/// Hierarchical CfC-based world model
#[derive(Debug)]
pub struct HierarchicalCfCWorldModel {
    /// Configuration
    config: WorldModelConfig,

    /// Layers in the hierarchy (index 0 = lowest level)
    layers: Vec<WorldModelLayer>,

    /// Inter-level projection matrices (upward)
    up_projections: Vec<Array2<f32>>,

    /// Inter-level projection matrices (downward)
    down_projections: Vec<Array2<f32>>,

    /// Stored concepts
    concepts: HashMap<u64, CrystalizedConcept>,

    /// Next concept ID
    next_concept_id: u64,

    /// Global statistics
    stats: WorldModelStats,
}

/// Statistics for the world model
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorldModelStats {
    /// Total updates across all levels
    pub total_updates: u64,

    /// Total concepts stored
    pub concepts_stored: usize,

    /// Total predictions made
    pub predictions_made: u64,

    /// Average global prediction error
    pub avg_global_error: f32,
}

impl HierarchicalCfCWorldModel {
    /// Create a new hierarchical world model
    pub fn new(config: WorldModelConfig) -> Self {
        let mut layers = Vec::with_capacity(config.num_levels);

        // Create layers
        // After up-projection, layer i receives level_dims[i] as input
        for i in 0..config.num_levels {
            // Layer i receives input at dimension level_dims[i] (after projection)
            let input_dim = config.level_dims[i];
            let hidden_dim = config.level_dims[i];
            let time_scale = config.time_scales[i];

            layers.push(WorldModelLayer::new(i, input_dim, hidden_dim, time_scale));
        }

        // Create inter-level projections
        let mut up_projections = Vec::new();
        let mut down_projections = Vec::new();

        for i in 0..(config.num_levels - 1) {
            let lower_dim = config.level_dims[i];
            let upper_dim = config.level_dims[i + 1];

            // Upward: lower -> upper
            let scale = (2.0 / (lower_dim + upper_dim) as f32).sqrt();
            let up_proj = Array2::from_shape_fn((upper_dim, lower_dim), |_| {
                (rand::random::<f32>() - 0.5) * 2.0 * scale
            });
            up_projections.push(up_proj);

            // Downward: upper -> lower
            let down_proj = Array2::from_shape_fn((lower_dim, upper_dim), |_| {
                (rand::random::<f32>() - 0.5) * 2.0 * scale
            });
            down_projections.push(down_proj);
        }

        Self {
            config,
            layers,
            up_projections,
            down_projections,
            concepts: HashMap::new(),
            next_concept_id: 1,
            stats: WorldModelStats::default(),
        }
    }

    /// Create a new hierarchical world model with deterministic initialization from a genesis seed.
    ///
    /// Each layer gets domain label `"{label}::layer_{i}"` and projections use
    /// `"{label}::up_{i}"` / `"{label}::down_{i}"`.
    pub fn from_genesis(config: WorldModelConfig, genesis: &GenesisSeed, label: &str) -> Self {
        let mut layers = Vec::with_capacity(config.num_levels);

        for i in 0..config.num_levels {
            let input_dim = config.level_dims[i];
            let hidden_dim = config.level_dims[i];
            let time_scale = config.time_scales[i];
            let layer_label = format!("{label}::layer_{i}");

            layers.push(WorldModelLayer::from_genesis(
                i,
                input_dim,
                hidden_dim,
                time_scale,
                genesis,
                &layer_label,
            ));
        }

        let mut up_projections = Vec::new();
        let mut down_projections = Vec::new();

        for i in 0..(config.num_levels - 1) {
            let lower_dim = config.level_dims[i];
            let upper_dim = config.level_dims[i + 1];
            let scale = (2.0 / (lower_dim + upper_dim) as f32).sqrt();

            let mut up_rng = genesis.domain(&format!("{label}::up_{i}"));
            let up_proj = Array2::from_shape_fn((upper_dim, lower_dim), |_| {
                (up_rng.r#gen::<f32>() - 0.5) * 2.0 * scale
            });
            up_projections.push(up_proj);

            let mut down_rng = genesis.domain(&format!("{label}::down_{i}"));
            let down_proj = Array2::from_shape_fn((lower_dim, upper_dim), |_| {
                (down_rng.r#gen::<f32>() - 0.5) * 2.0 * scale
            });
            down_projections.push(down_proj);
        }

        Self {
            config,
            layers,
            up_projections,
            down_projections,
            concepts: HashMap::new(),
            next_concept_id: 1,
            stats: WorldModelStats::default(),
        }
    }

    /// Update the world model with sensory input
    ///
    /// # Arguments
    /// * `sensory_input` - Raw sensory input at the lowest level
    /// * `dt` - Time step
    ///
    /// # Returns
    /// The highest-level abstract representation
    pub fn update(&mut self, sensory_input: &Array1<f32>, dt: f32) -> Array1<f32> {
        // Bottom-up processing
        let mut current = sensory_input.clone();

        for (i, layer) in self.layers.iter_mut().enumerate() {
            current = layer.update(&current, dt);

            // Project to next level if not at top
            if i < self.up_projections.len() {
                current = self.up_projections[i].dot(&current);
            }
        }

        // Top-down processing if bidirectional
        if self.config.bidirectional {
            let mut top_down = self
                .layers
                .last()
                .expect("layers always has at least one entry")
                .state()
                .clone();

            for i in (0..self.down_projections.len()).rev() {
                top_down = self.down_projections[i].dot(&top_down);

                // Combine with layer state and apply modulation
                let layer_state = self.layers[i].state().clone();
                top_down = &top_down * 0.5 + &layer_state * 0.5;
                self.layers[i].state = top_down.clone();
            }
        }

        self.stats.total_updates += 1;

        // Return top-level representation
        self.layers
            .last()
            .expect("layers always has at least one entry")
            .state()
            .clone()
    }

    /// Make predictions at all levels
    ///
    /// # Arguments
    /// * `dt` - Time step for prediction
    ///
    /// # Returns
    /// Predictions at each level (ordered from lowest to highest)
    pub fn predict(&mut self, dt: f32) -> Vec<Array1<f32>> {
        let predictions: Vec<_> = self
            .layers
            .iter_mut()
            .map(|layer| layer.predict(dt))
            .collect();

        self.stats.predictions_made += 1;
        predictions
    }

    /// Multi-step prediction
    pub fn predict_sequence(&mut self, steps: usize, dt: f32) -> Vec<Vec<Array1<f32>>> {
        let mut sequence = Vec::with_capacity(steps);

        for _ in 0..steps {
            sequence.push(self.predict(dt));
        }

        sequence
    }

    /// Get state at a specific level
    pub fn level_state(&self, level: usize) -> Option<&Array1<f32>> {
        self.layers.get(level).map(|l| l.state())
    }

    /// Get all layer states
    pub fn all_states(&self) -> Vec<&Array1<f32>> {
        self.layers.iter().map(|l| l.state()).collect()
    }

    /// Store a crystallized concept
    pub fn store_concept(&mut self, name: impl Into<String>, embedding: Vec<f32>) -> u64 {
        let id = self.next_concept_id;
        self.next_concept_id += 1;

        let concept = CrystalizedConcept::new(id, name, embedding);
        self.concepts.insert(id, concept);

        self.stats.concepts_stored = self.concepts.len();
        id
    }

    /// Retrieve a concept by ID
    pub fn get_concept(&self, id: u64) -> Option<&CrystalizedConcept> {
        self.concepts.get(&id)
    }

    /// Find similar concepts
    pub fn find_similar_concepts(
        &self,
        embedding: &[f32],
        top_k: usize,
    ) -> Vec<(&CrystalizedConcept, f32)> {
        let query_mag: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if query_mag == 0.0 {
            return Vec::new();
        }

        let mut similarities: Vec<_> = self
            .concepts
            .values()
            .filter_map(|concept| {
                if concept.embedding.len() != embedding.len() {
                    return None;
                }

                let dot: f32 = concept
                    .embedding
                    .iter()
                    .zip(embedding.iter())
                    .map(|(a, b)| a * b)
                    .sum();

                let concept_mag: f32 = concept.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                if concept_mag == 0.0 {
                    return None;
                }

                let similarity = dot / (query_mag * concept_mag);
                Some((concept, similarity))
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        similarities.truncate(top_k);
        similarities
    }

    /// Get statistics
    pub fn stats(&self) -> &WorldModelStats {
        &self.stats
    }

    /// Reset the world model
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        self.stats = WorldModelStats::default();
    }

    /// Get configuration
    pub fn config(&self) -> &WorldModelConfig {
        &self.config
    }

    /// Get number of levels
    pub fn num_levels(&self) -> usize {
        self.config.num_levels
    }

    /// Get combined CfC state from all layers
    /// Returns a flattened vector of all layer states
    pub fn cfc_state(&self) -> Vec<f32> {
        let mut state = Vec::new();
        for layer in &self.layers {
            state.extend(layer.state().iter());
        }
        state
    }

    /// Compute consciousness level from the CfC state
    /// Returns a value between 0.0 and 1.0 based on the integration/complexity of the state
    pub fn consciousness(&self) -> f32 {
        // Simple proxy for consciousness: normalized variance across layers
        // Higher variance = more differentiated states = higher consciousness
        let states: Vec<f32> = self.cfc_state();
        if states.is_empty() {
            return 0.0;
        }

        let mean = states.iter().sum::<f32>() / states.len() as f32;
        let variance = states.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / states.len() as f32;

        // Normalize to 0-1 range using sigmoid-like function
        let normalized = (variance * 10.0).tanh();
        normalized.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_model_creation() {
        let config = WorldModelConfig::default();
        let model = HierarchicalCfCWorldModel::new(config);
        assert_eq!(model.num_levels(), 4);
    }

    #[test]
    fn test_world_model_update() {
        let config = WorldModelConfig {
            num_levels: 3,
            level_dims: vec![32, 64, 32],
            time_scales: vec![1.0, 2.0, 4.0],
            ..Default::default()
        };
        let mut model = HierarchicalCfCWorldModel::new(config);

        let input = Array1::from_vec(vec![0.1; 32]);
        let output = model.update(&input, 0.1);

        assert_eq!(output.len(), 32);
    }

    #[test]
    fn test_concept_storage() {
        let mut model = HierarchicalCfCWorldModel::new(WorldModelConfig::default());

        let id1 = model.store_concept("test_concept", vec![0.1; 64]);
        let id2 = model.store_concept("another_concept", vec![0.2; 64]);

        assert!(model.get_concept(id1).is_some());
        assert!(model.get_concept(id2).is_some());
        assert_eq!(model.stats().concepts_stored, 2);
    }
}
