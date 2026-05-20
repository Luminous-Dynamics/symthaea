// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # HDC-LTC Unified Network Bridge
//!
//! This module provides an adapter/bridge that allows the HdcLtcUnifiedNetwork
//! to be used as a drop-in alternative to the CfC network in the cognitive loop.
//!
//! ## Architecture
//!
//! The bridge converts between:
//! - HDC encodings (compressed f32 vectors) <-> HdcLtcUnifiedNetwork inputs (ContinuousHV)
//! - HdcLtcUnifiedNetwork outputs <-> predictions for the encoder
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::hdc_ltc_bridge::{HdcLtcBridge, HdcLtcBridgeConfig};
//!
//! let config = HdcLtcBridgeConfig::default();
//! let mut bridge = HdcLtcBridge::new(config);
//!
//! // Process input (same interface as CfC)
//! bridge.step(&input_array, dt)?;
//! let state = bridge.read_state()?;
//! ```

use anyhow::Result;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::hdc_ltc_unified::{
    HdcLtcUnifiedNetwork, UnifiedActivation, UnifiedConfig, UnifiedNetworkConfig,
};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the HDC-LTC bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdcLtcBridgeConfig {
    /// Input dimension (from cognitive loop's compressed HDC)
    pub input_dim: usize,

    /// Output dimension (for predictions)
    pub output_dim: usize,

    /// Network layer sizes
    pub layer_sizes: Vec<usize>,

    /// Base time constant for LTC dynamics
    pub tau_base: f32,

    /// State-dependent time constant scaling
    pub backbone_tau: f32,

    /// Learning rate for online adaptation
    pub learning_rate: f32,

    /// Whether to use layer binding in the network
    pub use_layer_binding: bool,

    /// Whether to use skip connections
    pub skip_connections: bool,

    /// Activation function
    pub activation: BridgeActivation,

    /// Random seed for initialization
    pub seed: u64,

    /// HDC dimension for the internal network.
    /// Default is HDC_DIMENSION (16,384). Lower values (e.g. 1024, 2048)
    /// trade accuracy for dramatically faster computation.
    pub hdc_dim: usize,

    /// Optional adaptive dimension scaling. When set, the bridge starts
    /// at `hdc_dim` and scales up/down based on prediction error.
    pub adaptive_dim: Option<AdaptiveDimConfig>,
}

/// Configuration for adaptive HDC dimensionality scaling.
///
/// Start small (fast) and grow only when prediction error demands it.
/// This avoids paying the cost of high-dimensional computation until
/// the task actually requires it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveDimConfig {
    /// Minimum HDC dimension (starting point)
    pub min_dim: usize,
    /// Maximum HDC dimension (ceiling)
    pub max_dim: usize,
    /// If error stays above this, scale up
    pub upscale_error_threshold: f32,
    /// If error drops below this, scale down
    pub downscale_error_threshold: f32,
    /// Dimension change per resize step
    pub scale_step: usize,
    /// Minimum cycles between resizes
    pub cooldown_cycles: usize,
}

impl Default for AdaptiveDimConfig {
    fn default() -> Self {
        Self {
            min_dim: 2048,
            max_dim: 16384,
            upscale_error_threshold: 0.8,
            downscale_error_threshold: 0.3,
            scale_step: 2048,
            cooldown_cycles: 20,
        }
    }
}

/// Activation function selection for the bridge
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum BridgeActivation {
    Tanh,
    Sigmoid,
    SiLU,
    Identity,
}

impl Default for HdcLtcBridgeConfig {
    fn default() -> Self {
        Self {
            input_dim: 256, // Match CfC default
            output_dim: 256,
            layer_sizes: vec![4, 8, 4], // 3-layer network
            tau_base: 0.1,
            backbone_tau: 0.5,
            learning_rate: 0.01,
            use_layer_binding: true,
            skip_connections: false,
            activation: BridgeActivation::Tanh,
            seed: 42,
            hdc_dim: HDC_DIMENSION,
            adaptive_dim: None,
        }
    }
}

impl HdcLtcBridgeConfig {
    /// Create config optimized for fast response
    pub fn fast() -> Self {
        Self {
            tau_base: 0.05,             // Faster time constant
            layer_sizes: vec![2, 4, 2], // Smaller network
            hdc_dim: 2048,              // Reduced dimension for speed
            ..Default::default()
        }
    }

    /// Create config optimized for accuracy
    pub fn accurate() -> Self {
        Self {
            layer_sizes: vec![8, 16, 8], // Larger network
            skip_connections: true,
            ..Default::default()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BRIDGE IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Bridge/adapter for using HdcLtcUnifiedNetwork in the cognitive loop
///
/// This provides the same interface as CfCNetwork, enabling drop-in replacement
/// while leveraging the unified HDC-LTC architecture.
#[derive(Debug, Clone)]
pub struct HdcLtcBridge {
    /// The underlying unified network
    network: HdcLtcUnifiedNetwork,

    /// Configuration
    config: HdcLtcBridgeConfig,

    /// Projection weights: input_dim -> HDC_DIMENSION
    input_projection: Vec<f32>,

    /// Projection weights: HDC_DIMENSION -> output_dim
    output_projection: Vec<f32>,

    /// Current output state (cached for read_state)
    current_output: Vec<f32>,

    /// Total steps processed
    total_steps: u64,

    /// Running state diversity metric
    state_diversity: f32,

    /// Cycles since the last adaptive dimension resize
    cycles_since_resize: usize,

    /// Optional genesis seed for deterministic re-initialization (e.g. during resize)
    genesis: Option<GenesisSeed>,
}

impl HdcLtcBridge {
    /// Create a new bridge with given configuration
    pub fn new(config: HdcLtcBridgeConfig) -> Self {
        let hdc_dim = config.hdc_dim;

        // Create the unified network configuration
        let neuron_config = UnifiedConfig {
            tau_base: config.tau_base,
            backbone_tau: config.backbone_tau,
            dimension: hdc_dim,
            activation: match config.activation {
                BridgeActivation::Tanh => UnifiedActivation::Tanh,
                BridgeActivation::Sigmoid => UnifiedActivation::Sigmoid,
                BridgeActivation::SiLU => UnifiedActivation::SiLU,
                BridgeActivation::Identity => UnifiedActivation::Identity,
            },
            learning_rate: config.learning_rate,
            ..UnifiedConfig::default()
        };

        let network_config = UnifiedNetworkConfig {
            layer_sizes: config.layer_sizes.clone(),
            neuron_config,
            use_layer_binding: config.use_layer_binding,
            skip_connections: config.skip_connections,
        };

        let network = HdcLtcUnifiedNetwork::new(network_config, config.seed);

        // Initialize random projection matrices
        let input_projection =
            Self::init_projection(config.input_dim, hdc_dim, config.seed + 100000);

        let output_projection =
            Self::init_projection(hdc_dim, config.output_dim, config.seed + 200000);

        Self {
            network,
            config: config.clone(),
            input_projection,
            output_projection,
            current_output: vec![0.0; config.output_dim],
            total_steps: 0,
            state_diversity: 0.0,
            cycles_since_resize: 0,
            genesis: None,
        }
    }

    /// Create a bridge with deterministic genesis seeding
    pub fn from_genesis(
        config: HdcLtcBridgeConfig,
        genesis: &symthaea_core::genesis::GenesisSeed,
    ) -> Self {
        let hdc_dim = config.hdc_dim;

        let neuron_config = UnifiedConfig {
            tau_base: config.tau_base,
            backbone_tau: config.backbone_tau,
            dimension: hdc_dim,
            activation: match config.activation {
                BridgeActivation::Tanh => UnifiedActivation::Tanh,
                BridgeActivation::Sigmoid => UnifiedActivation::Sigmoid,
                BridgeActivation::SiLU => UnifiedActivation::SiLU,
                BridgeActivation::Identity => UnifiedActivation::Identity,
            },
            learning_rate: config.learning_rate,
            ..UnifiedConfig::default()
        };

        let network_config = UnifiedNetworkConfig {
            layer_sizes: config.layer_sizes.clone(),
            neuron_config,
            use_layer_binding: config.use_layer_binding,
            skip_connections: config.skip_connections,
        };

        let network = HdcLtcUnifiedNetwork::from_genesis(network_config, genesis);

        let input_projection = Self::init_projection_from_genesis(
            genesis,
            "bridge::input_projection",
            config.input_dim,
            hdc_dim,
        );
        let output_projection = Self::init_projection_from_genesis(
            genesis,
            "bridge::output_projection",
            hdc_dim,
            config.output_dim,
        );

        Self {
            network,
            config: config.clone(),
            input_projection,
            output_projection,
            current_output: vec![0.0; config.output_dim],
            total_steps: 0,
            state_diversity: 0.0,
            cycles_since_resize: 0,
            genesis: Some(genesis.clone()),
        }
    }

    /// Initialize a projection matrix from genesis seed
    fn init_projection_from_genesis(
        genesis: &symthaea_core::genesis::GenesisSeed,
        label: &str,
        input_dim: usize,
        output_dim: usize,
    ) -> Vec<f32> {
        use rand::Rng;
        let mut rng = genesis.domain(label);
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();
        (0..input_dim * output_dim)
            .map(|_| (rng.r#gen::<f32>() * 2.0 - 1.0) * scale)
            .collect()
    }

    /// Initialize a random projection matrix (stored as flattened vec)
    fn init_projection(input_dim: usize, output_dim: usize, seed: u64) -> Vec<f32> {
        let mut projection = Vec::with_capacity(input_dim * output_dim);
        let mut state = seed;

        // Xavier/Glorot initialization scale
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt();

        for _ in 0..(input_dim * output_dim) {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let normalized = (state as f32 / u64::MAX as f32) * 2.0 - 1.0;
            projection.push(normalized * scale);
        }

        projection
    }

    /// Project compressed input to HDC dimension
    ///
    /// Uses row-accumulation pattern for cache-friendly access:
    /// for each input element i, scatter input[i] * projection_row[i] across output.
    /// Layout: input_projection is [input_dim][HDC_DIMENSION] row-major.
    #[inline]
    fn project_to_hdc(&self, input: &Array1<f32>) -> ContinuousHV {
        let input_dim = self.config.input_dim.min(input.len());
        let hdc_dim = self.config.hdc_dim;
        let mut values = vec![0.0f32; hdc_dim];

        // Row-accumulation: iterate rows (input elements), accumulate into output
        // Each row is contiguous in memory → cache-friendly
        for i in 0..input_dim {
            let x = input[i];
            if x.abs() < 1e-10 {
                continue;
            } // Skip near-zero inputs
            let row = &self.input_projection[i * hdc_dim..(i + 1) * hdc_dim];
            for (v, &w) in values.iter_mut().zip(row.iter()) {
                *v += x * w;
            }
        }

        // Apply tanh bounding
        for v in values.iter_mut() {
            *v = v.tanh();
        }

        ContinuousHV::from_values(values)
    }

    /// Project HDC output to compressed dimension
    ///
    /// Uses row-accumulation pattern: for each HDC element j, scatter
    /// hv[j] * projection_row[j] across output.
    /// Layout: output_projection is [HDC_DIMENSION][output_dim] row-major.
    #[inline]
    fn project_from_hdc(&self, hv: &ContinuousHV) -> Vec<f32> {
        let output_dim = self.config.output_dim;
        let hdc_dim = self.config.hdc_dim;
        let mut output = vec![0.0f32; output_dim];

        // Row-accumulation: iterate rows (HDC elements), accumulate into output.
        // NOTE: No near-zero skip — the closed-form CfC evolution produces small
        // but real values in early cycles (e.g., 0.02 * h_inf). Skipping at 1e-10
        // was dropping legitimate signal, causing zero-vector predictions → PE=1.0.
        for j in 0..hdc_dim {
            let x = hv.values[j];
            let row = &self.output_projection[j * output_dim..(j + 1) * output_dim];
            for (o, &w) in output.iter_mut().zip(row.iter()) {
                *o += x * w;
            }
        }

        output
    }

    /// Update state diversity metric
    #[inline]
    fn update_diversity(&mut self) {
        let output = self.network.output();
        let values = &output.values;
        let n = values.len() as f32;

        // Single-pass variance using Welford-like: var = E[x^2] - E[x]^2
        let mut sum = 0.0f32;
        let mut sum_sq = 0.0f32;
        for &v in values.iter() {
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / n;
        let variance = (sum_sq / n - mean * mean).max(0.0);

        // Normalize to 0-1 using sigmoid
        self.state_diversity = 1.0 / (1.0 + (-variance.sqrt() * 10.0).exp());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ADAPTIVE DIMENSIONALITY
    // ═══════════════════════════════════════════════════════════════════════════

    /// Check if the HDC dimension should be resized based on current prediction error.
    ///
    /// If `adaptive_dim` is None, this is a no-op. Otherwise:
    /// - High error (above upscale threshold) -> increase dimension by scale_step
    /// - Low error (below downscale threshold) -> decrease dimension by scale_step
    /// - Cooldown prevents resizing more than once every N cycles
    ///
    /// When resizing occurs, projection matrices are rebuilt at the new dimension.
    pub fn maybe_resize(&mut self, current_error: f32) {
        let adaptive = match &self.config.adaptive_dim {
            Some(a) => a.clone(),
            None => return,
        };

        self.cycles_since_resize += 1;

        if self.cycles_since_resize < adaptive.cooldown_cycles {
            return;
        }

        let current_dim = self.config.hdc_dim;
        let new_dim = if current_error > adaptive.upscale_error_threshold {
            (current_dim + adaptive.scale_step).min(adaptive.max_dim)
        } else if current_error < adaptive.downscale_error_threshold {
            current_dim
                .saturating_sub(adaptive.scale_step)
                .max(adaptive.min_dim)
        } else {
            return; // No change needed
        };

        if new_dim == current_dim {
            return; // Already at boundary
        }

        // Rebuild projection matrices at new dimension
        self.config.hdc_dim = new_dim;
        if let Some(ref genesis) = self.genesis {
            let resize_label = format!("bridge::resize_{}", self.total_steps);
            self.input_projection = Self::init_projection_from_genesis(
                genesis,
                &format!("{resize_label}::input"),
                self.config.input_dim,
                new_dim,
            );
            self.output_projection = Self::init_projection_from_genesis(
                genesis,
                &format!("{resize_label}::output"),
                new_dim,
                self.config.output_dim,
            );
        } else {
            self.input_projection = Self::init_projection(
                self.config.input_dim,
                new_dim,
                self.config.seed + 100000 + self.total_steps,
            );
            self.output_projection = Self::init_projection(
                new_dim,
                self.config.output_dim,
                self.config.seed + 200000 + self.total_steps,
            );
        }

        // Rebuild the network at the new dimension
        let neuron_config = UnifiedConfig {
            tau_base: self.config.tau_base,
            backbone_tau: self.config.backbone_tau,
            dimension: new_dim,
            activation: match self.config.activation {
                BridgeActivation::Tanh => UnifiedActivation::Tanh,
                BridgeActivation::Sigmoid => UnifiedActivation::Sigmoid,
                BridgeActivation::SiLU => UnifiedActivation::SiLU,
                BridgeActivation::Identity => UnifiedActivation::Identity,
            },
            learning_rate: self.config.learning_rate,
            ..UnifiedConfig::default()
        };

        let network_config = UnifiedNetworkConfig {
            layer_sizes: self.config.layer_sizes.clone(),
            neuron_config,
            use_layer_binding: self.config.use_layer_binding,
            skip_connections: self.config.skip_connections,
        };

        if let Some(ref genesis) = self.genesis {
            self.network = HdcLtcUnifiedNetwork::from_genesis(network_config, genesis);
        } else {
            self.network =
                HdcLtcUnifiedNetwork::new(network_config, self.config.seed + self.total_steps);
        }
        self.cycles_since_resize = 0;
    }

    /// Get the current HDC dimension (may differ from initial if adaptive)
    pub fn current_hdc_dim(&self) -> usize {
        self.config.hdc_dim
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // CfC-COMPATIBLE INTERFACE
    // These methods match the CfCNetwork interface for drop-in replacement
    // ═══════════════════════════════════════════════════════════════════════════

    /// Step the network forward (matches CfCNetwork::step)
    pub fn step(&mut self, input: &Array1<f32>, dt: f32) -> Result<()> {
        // Project input to HDC dimension
        let hdc_input = self.project_to_hdc(input);

        // Evolve the unified network using closed-form solution
        self.network.evolve_closed_form(dt, &hdc_input);

        // Project output back
        let hdc_output = self.network.output();
        self.current_output = self.project_from_hdc(&hdc_output);

        // Update metrics
        self.total_steps += 1;
        self.update_diversity();

        Ok(())
    }

    /// Read the current state (matches CfCNetwork::read_state)
    pub fn read_state(&self) -> Result<Array1<f32>> {
        Ok(Array1::from_vec(self.current_output.clone()))
    }

    /// Forward pass and return output (matches CfCNetwork::forward)
    pub fn forward(&mut self, input: &Array1<f32>, dt: f32) -> Array1<f32> {
        let _ = self.step(input, dt);
        Array1::from_vec(self.current_output.clone())
    }

    /// Train step using analytical BPTT gradients (matches CfCNetwork::train_step)
    ///
    /// Replaces Hebbian learning with backpropagation through the closed-form
    /// evolution step, computing exact gradients for weight_hv and input_mask.
    pub fn train_step(
        &mut self,
        input: &Array1<f32>,
        target: &Array1<f32>,
        dt: f32,
        learning_rate: f32,
    ) -> Result<f32> {
        // Project input to HDC
        let hdc_input = self.project_to_hdc(input);

        // Project target to HDC (reuse same projection)
        let hdc_target = self.project_to_hdc(target);

        // Evolve network
        self.network.evolve_closed_form(dt, &hdc_input);

        // Get output and compute error in output space
        let hdc_output = self.network.output();
        let output = self.project_from_hdc(&hdc_output);

        // Compute MSE loss in output space
        let loss: f32 = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| (o - t).powi(2))
            .sum::<f32>()
            / target.len() as f32;

        // Apply BPTT gradients to all layers
        let n_layers = self.network.n_layers();
        for layer_idx in 0..n_layers {
            let layer_in = self.network.layer_input(layer_idx, &hdc_input);
            if let Some(layer) = self.network.layer_mut(layer_idx) {
                for neuron in layer.iter_mut() {
                    let grads = neuron.backward(&layer_in, &hdc_target, dt);
                    neuron.apply_gradients(&grads, learning_rate);
                }
            }
        }

        // Update projection weights using gradient descent
        self.update_projections(&hdc_input, target, &output, learning_rate);

        // Update cached output
        self.current_output = output;
        self.total_steps += 1;
        self.update_diversity();

        Ok(loss)
    }

    /// Update projection weights using simple gradient descent
    fn update_projections(
        &mut self,
        _hdc_input: &ContinuousHV,
        target: &Array1<f32>,
        output: &[f32],
        learning_rate: f32,
    ) {
        let output_dim = self.config.output_dim;

        // Compute output error
        let errors: Vec<f32> = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| o - t)
            .collect();

        // Update output projection (simple gradient descent)
        let hdc_output = self.network.output();
        let hdc_dim = self.config.hdc_dim;
        for i in 0..output_dim {
            for j in 0..hdc_dim {
                let grad = errors[i] * hdc_output.values[j];
                self.output_projection[j * output_dim + i] -= learning_rate * grad;
            }
        }
    }

    /// Predict forward at a specific time horizon (matches CfCNetwork::predict_forward)
    pub fn predict_forward(&mut self, input: &Array1<f32>, horizon: f32) -> Result<Array1<f32>> {
        Ok(self.forward(input, horizon))
    }

    /// Inject state into the network (matches CfCNetwork::inject)
    pub fn inject(&mut self, state: &Array1<f32>) -> Result<()> {
        // Reset the network and set initial state based on input
        self.network.reset();
        self.current_output = state.to_vec();
        Ok(())
    }

    /// Reset the network (matches CfCNetwork::reset)
    pub fn reset(&mut self) {
        self.network.reset();
        self.current_output = vec![0.0; self.config.output_dim];
        self.total_steps = 0;
        self.state_diversity = 0.0;
    }

    /// Get state diversity metric (matches CfCNetwork::state_diversity)
    pub fn state_diversity(&self) -> f32 {
        self.state_diversity
    }

    /// Get all tau values for coherence tracking
    ///
    /// Returns the effective tau from each neuron in the network
    pub fn all_tau(&self) -> Vec<Array1<f32>> {
        let mut taus = Vec::new();

        for layer_idx in 0..self.network.n_layers() {
            if let Some(layer) = self.network.layer(layer_idx) {
                // Create a single tau array for this layer based on neuron configs
                let layer_taus: Vec<f32> = layer
                    .iter()
                    .map(|n| n.config().tau_base * (1.0 + n.config().backbone_tau))
                    .collect();
                taus.push(Array1::from_vec(layer_taus));
            }
        }

        taus
    }

    /// Get flattened tau values
    pub fn flattened_tau(&self) -> Vec<f32> {
        self.all_tau()
            .into_iter()
            .flat_map(|t| t.to_vec())
            .collect()
    }

    /// Get configuration
    pub fn config(&self) -> &HdcLtcBridgeConfig {
        &self.config
    }

    /// Get total steps
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// Get network statistics
    pub fn stats(&self) -> HdcLtcBridgeStats {
        let network_stats = self.network.stats();
        HdcLtcBridgeStats {
            total_steps: self.total_steps,
            state_diversity: self.state_diversity,
            n_layers: network_stats.n_layers,
            n_neurons: network_stats.n_neurons,
            avg_state_norm: network_stats.avg_state_norm,
            avg_weight_norm: network_stats.avg_weight_norm,
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // HDC-DIRECT ACCESS (bypasses temporal CfC processing)
    // ═══════════════════════════════════════════════════════════════════════════

    /// Project input directly to HDC space, bypassing CfC temporal dynamics.
    ///
    /// This returns the pure semantic HDC representation before any temporal
    /// state accumulation occurs. Useful for:
    /// - Semantic similarity comparisons (cosine similarity of HDC vectors)
    /// - Debugging whether semantic structure is preserved
    /// - Comparing HDC-direct clustering vs CfC-output clustering
    ///
    /// The returned vector has `hdc_dim` dimensions (default 16,384).
    pub fn project_to_hdc_vec(&self, input: &[f32]) -> Vec<f32> {
        let input_dim = self.config.input_dim.min(input.len());
        let hdc_dim = self.config.hdc_dim;
        let mut values = vec![0.0f32; hdc_dim];

        // Row-accumulation: iterate rows (input elements), accumulate into output
        // Use 1e-6 threshold instead of 1e-10 to be less aggressive about skipping
        for i in 0..input_dim {
            let x = input[i];
            if x.abs() < 1e-6 {
                continue;
            } // Skip near-zero inputs
            let row_start = i * hdc_dim;
            let row_end = (i + 1) * hdc_dim;

            // Bounds check to avoid panic
            if row_end > self.input_projection.len() {
                break;
            }

            let row = &self.input_projection[row_start..row_end];
            for (v, &w) in values.iter_mut().zip(row.iter()) {
                *v += x * w;
            }
        }

        // Apply tanh bounding (same as internal project_to_hdc)
        for v in values.iter_mut() {
            *v = v.tanh();
        }

        values
    }

    /// Get the HDC dimension used by this bridge
    pub fn hdc_dim(&self) -> usize {
        self.config.hdc_dim
    }
}

/// Statistics from the HDC-LTC bridge
#[derive(Debug, Clone)]
pub struct HdcLtcBridgeStats {
    /// Total steps processed
    pub total_steps: u64,
    /// Current state diversity
    pub state_diversity: f32,
    /// Number of layers
    pub n_layers: usize,
    /// Number of neurons
    pub n_neurons: usize,
    /// Average state norm across neurons
    pub avg_state_norm: f32,
    /// Average weight norm across neurons
    pub avg_weight_norm: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bridge_creation() {
        let config = HdcLtcBridgeConfig::default();
        let bridge = HdcLtcBridge::new(config.clone());

        assert_eq!(bridge.config().input_dim, 256);
        assert_eq!(bridge.config().output_dim, 256);
        assert_eq!(bridge.total_steps(), 0);
    }

    #[test]
    fn test_bridge_step() {
        let config = HdcLtcBridgeConfig::default();
        let mut bridge = HdcLtcBridge::new(config);

        let input = Array1::from_vec(vec![0.5; 256]);
        let result = bridge.step(&input, 0.02);

        assert!(result.is_ok());
        assert_eq!(bridge.total_steps(), 1);
    }

    #[test]
    fn test_bridge_forward() {
        let config = HdcLtcBridgeConfig::default();
        let mut bridge = HdcLtcBridge::new(config);

        let input = Array1::from_vec(vec![0.5; 256]);
        let output = bridge.forward(&input, 0.02);

        assert_eq!(output.len(), 256);
    }

    #[test]
    fn test_bridge_read_state() {
        let config = HdcLtcBridgeConfig::default();
        let mut bridge = HdcLtcBridge::new(config);

        let input = Array1::from_vec(vec![0.5; 256]);
        let _ = bridge.step(&input, 0.02);

        let state = bridge.read_state().unwrap();
        assert_eq!(state.len(), 256);
    }

    #[test]
    fn test_bridge_train_step() {
        let config = HdcLtcBridgeConfig::default();
        let mut bridge = HdcLtcBridge::new(config);

        let input = Array1::from_vec(vec![0.5; 256]);
        let target = Array1::from_vec(vec![0.3; 256]);

        let loss = bridge.train_step(&input, &target, 0.02, 0.01).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_bridge_reset() {
        let config = HdcLtcBridgeConfig::default();
        let mut bridge = HdcLtcBridge::new(config);

        let input = Array1::from_vec(vec![0.5; 256]);
        let _ = bridge.step(&input, 0.02);
        assert_eq!(bridge.total_steps(), 1);

        bridge.reset();
        assert_eq!(bridge.total_steps(), 0);
    }

    #[test]
    fn test_bridge_state_diversity() {
        let config = HdcLtcBridgeConfig::default();
        let mut bridge = HdcLtcBridge::new(config);

        let input = Array1::from_vec(vec![0.5; 256]);
        let _ = bridge.step(&input, 0.02);

        let diversity = bridge.state_diversity();
        assert!((0.0..=1.0).contains(&diversity));
    }

    #[test]
    fn test_bridge_all_tau() {
        let config = HdcLtcBridgeConfig::default();
        let bridge = HdcLtcBridge::new(config);

        let taus = bridge.all_tau();
        assert!(!taus.is_empty());
    }

    #[test]
    fn test_bridge_predict_forward() {
        let config = HdcLtcBridgeConfig::default();
        let mut bridge = HdcLtcBridge::new(config);

        let input = Array1::from_vec(vec![0.5; 256]);

        // Small horizon
        let pred_small = bridge.predict_forward(&input, 0.1).unwrap();
        assert_eq!(pred_small.len(), 256);

        // Large horizon (O(1) closed-form)
        let pred_large = bridge.predict_forward(&input, 10.0).unwrap();
        assert_eq!(pred_large.len(), 256);
    }

    #[test]
    fn test_bridge_inject() {
        let config = HdcLtcBridgeConfig::default();
        let mut bridge = HdcLtcBridge::new(config);

        let state = Array1::from_vec(vec![0.3; 256]);
        let result = bridge.inject(&state);

        assert!(result.is_ok());
    }

    #[test]
    fn test_fast_config() {
        let config = HdcLtcBridgeConfig::fast();
        assert_eq!(config.tau_base, 0.05);
        assert_eq!(config.layer_sizes, vec![2, 4, 2]);
    }

    #[test]
    fn test_accurate_config() {
        let config = HdcLtcBridgeConfig::accurate();
        assert!(config.skip_connections);
        assert_eq!(config.layer_sizes, vec![8, 16, 8]);
    }

    #[test]
    fn test_adaptive_dim_upscale() {
        let config = HdcLtcBridgeConfig {
            hdc_dim: 2048,
            adaptive_dim: Some(AdaptiveDimConfig {
                cooldown_cycles: 20,
                ..AdaptiveDimConfig::default()
            }),
            ..HdcLtcBridgeConfig::default()
        };
        let mut bridge = HdcLtcBridge::new(config);
        assert_eq!(bridge.current_hdc_dim(), 2048);

        // Call maybe_resize with high error 25 times (exceeds cooldown of 20)
        for _ in 0..25 {
            bridge.maybe_resize(0.9);
        }

        assert!(
            bridge.current_hdc_dim() > 2048,
            "dim should have increased, got {}",
            bridge.current_hdc_dim()
        );
    }

    #[test]
    fn test_adaptive_dim_downscale() {
        let config = HdcLtcBridgeConfig {
            hdc_dim: 8192,
            adaptive_dim: Some(AdaptiveDimConfig {
                cooldown_cycles: 20,
                ..AdaptiveDimConfig::default()
            }),
            ..HdcLtcBridgeConfig::default()
        };
        let mut bridge = HdcLtcBridge::new(config);
        assert_eq!(bridge.current_hdc_dim(), 8192);

        // Call maybe_resize with low error 25 times
        for _ in 0..25 {
            bridge.maybe_resize(0.1);
        }

        assert!(
            bridge.current_hdc_dim() < 8192,
            "dim should have decreased, got {}",
            bridge.current_hdc_dim()
        );
    }

    #[test]
    fn test_adaptive_dim_cooldown() {
        let config = HdcLtcBridgeConfig {
            hdc_dim: 4096,
            adaptive_dim: Some(AdaptiveDimConfig {
                cooldown_cycles: 20,
                ..AdaptiveDimConfig::default()
            }),
            ..HdcLtcBridgeConfig::default()
        };
        let mut bridge = HdcLtcBridge::new(config);

        // Call only 10 times (under cooldown of 20) - should NOT resize
        for _ in 0..10 {
            bridge.maybe_resize(0.9);
        }
        assert_eq!(
            bridge.current_hdc_dim(),
            4096,
            "should not resize within cooldown"
        );

        // Call 15 more (total 25, exceeds cooldown) - should resize
        for _ in 0..15 {
            bridge.maybe_resize(0.9);
        }
        assert!(
            bridge.current_hdc_dim() > 4096,
            "should resize after cooldown"
        );
    }
}
