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
    HdcLtcUnifiedNetwork, NetworkStateSnapshot, UnifiedActivation, UnifiedConfig,
    UnifiedNetworkConfig,
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

    /// Reusable snapshot buffer that lets predict_forward evolve-and-restore
    /// without wiping live state (allocation-free after first use)
    predict_scratch: NetworkStateSnapshot,

    /// Reusable snapshot buffer for the training pass's live-state save
    /// (train_step is pure w.r.t. evolution state; see its doc)
    train_scratch: NetworkStateSnapshot,

    /// Reusable snapshot buffer for [`Self::eval_loss_from`]'s live-state save
    /// -- a dedicated buffer (not reusing `train_scratch`/`predict_scratch`)
    /// so this purely-evaluative path can never alias with an in-flight
    /// train/predict call.
    eval_scratch: NetworkStateSnapshot,
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
            predict_scratch: NetworkStateSnapshot::default(),
            train_scratch: NetworkStateSnapshot::default(),
            eval_scratch: NetworkStateSnapshot::default(),
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
            predict_scratch: NetworkStateSnapshot::default(),
            train_scratch: NetworkStateSnapshot::default(),
            eval_scratch: NetworkStateSnapshot::default(),
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

        // SCALE RESTORATION (2026-07-18, probe_signal_scale finding): the bind
        // chain annihilates state magnitude ∝ d^-1.9 — at hdc_dim=16,384 the
        // raw projection lands at ~3e-10 vs unit-norm input, the readout's
        // loss equals mean(target²) (it contributes nothing), and its
        // gradients (~1e-13) are untrainable at any learning rate. Normalize
        // by the state HV's norm so the readout operates at O(1) scale:
        // direction (the meaning-carrier in HDC) is untouched, downstream
        // consumers finally receive signal, and readout gradient flow is
        // restored. Zero-norm guard: a truly empty state projects to zeros
        // (absence, honestly).
        let state_norm = hv.norm();
        if state_norm > 1e-30 {
            for o in output.iter_mut() {
                *o /= state_norm;
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

    /// Train step using analytical BPTT gradients (matches CfCNetwork::train_step).
    ///
    /// PURE with respect to live evolution state (2026-07-17): the training
    /// forward pass runs on a scratch evolution — live neuron state,
    /// `current_output`, `total_steps`, and the diversity metric are all
    /// untouched. Only WEIGHTS (neuron parameters + output projection) change.
    ///
    /// HISTORY: this used to evolve the LIVE network as a side effect and
    /// overwrite `current_output`. With per-cycle training, the live temporal
    /// trajectory was re-evolved every cycle with the PREVIOUS cycle's
    /// encoding after the planning phase had already stepped it with the
    /// current one — shuffling subjective time (enc_t → enc_{t−1} →
    /// enc_{t+1} → …) and making sequence learning impossible by
    /// construction. Keystone Phase 3 measured the consequence: zero order
    /// anticipation in every arm, PE above the uncorrelated baseline (the
    /// one-step-behind signature). docs/KEYSTONE_AB_PROTOCOL_2026-07-17.md.
    pub fn train_step(
        &mut self,
        input: &Array1<f32>,
        target: &Array1<f32>,
        dt: f32,
        learning_rate: f32,
    ) -> Result<f32> {
        self.train_step_impl(None, input, target, dt, learning_rate)
    }

    /// Like [`Self::train_step`], but the training forward pass starts from
    /// the given historical snapshot instead of the current live state — for
    /// callers training on a (state at end of cycle t−2, enc_{t−1} → enc_t)
    /// pair whose correct starting state is no longer the live one.
    pub fn train_step_from(
        &mut self,
        start: &NetworkStateSnapshot,
        input: &Array1<f32>,
        target: &Array1<f32>,
        dt: f32,
        learning_rate: f32,
    ) -> Result<f32> {
        // Stale-snapshot guard: after an adaptive resize the historical
        // snapshot's dimension no longer matches the network — fall back to
        // a live-state start rather than restoring poisoned state. The
        // caller's rolling queue refills with current-dimension snapshots
        // within two cycles.
        let start = if start.dimension() == Some(self.config.hdc_dim) {
            Some(start)
        } else {
            None
        };
        self.train_step_impl(start, input, target, dt, learning_rate)
    }

    /// Predictive Compression Program C2 (docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md §6):
    /// pure loss evaluation from a historical snapshot -- no weights, no evolution state, no
    /// counters change. Lets a caller compute `(pre_loss, post_loss)` around a real
    /// [`Self::train_step_from`] call (evaluate here, train, evaluate again) without touching
    /// `train_step_from`'s existing `Result<f32>` signature or any of its production call
    /// sites (`cognitive_loop::temporal_network`, `cycle_phase_dynamics::training`) -- this is
    /// additive, not a breaking change to the training path.
    ///
    /// Same stale-snapshot guard and scratch-buffer pattern as [`Self::train_step_from`]/
    /// [`Self::predict_forward`], but with its own dedicated `eval_scratch` buffer so this can
    /// never alias an in-flight train/predict call's own scratch state.
    pub fn eval_loss_from(
        &mut self,
        start: &NetworkStateSnapshot,
        input: &Array1<f32>,
        target: &Array1<f32>,
        dt: f32,
    ) -> f32 {
        let start = if start.dimension() == Some(self.config.hdc_dim) {
            Some(start)
        } else {
            None
        };

        let hdc_input = self.project_to_hdc(input);

        // Save live evolution state -- this evaluation must not perturb it, exactly like
        // train_step_impl's own live-state save/restore.
        let mut live = std::mem::take(&mut self.eval_scratch);
        self.network.snapshot_state_into(&mut live);

        if let Some(start) = start {
            self.network.restore_state_from(start);
        }

        self.network.evolve_closed_form(dt, &hdc_input);
        let hdc_output = self.network.output();
        let output = self.project_from_hdc(&hdc_output);

        let loss: f32 = output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| (o - t).powi(2))
            .sum::<f32>()
            / target.len() as f32;

        // Restore live evolution state -- no weights were ever touched in this path, so there
        // is nothing else to persist.
        self.network.restore_state_from(&live);
        self.eval_scratch = live;

        loss
    }

    fn train_step_impl(
        &mut self,
        start: Option<&NetworkStateSnapshot>,
        input: &Array1<f32>,
        target: &Array1<f32>,
        dt: f32,
        learning_rate: f32,
    ) -> Result<f32> {
        // Project input to HDC
        let hdc_input = self.project_to_hdc(input);

        // Project target to HDC (reuse same projection)
        let hdc_target = self.project_to_hdc(target);

        // Save live evolution state — training must not perturb it.
        let mut live = std::mem::take(&mut self.train_scratch);
        self.network.snapshot_state_into(&mut live);

        // Start the training forward pass from the historical state if given
        // (otherwise from the live state, without mutating it observably).
        if let Some(start) = start {
            self.network.restore_state_from(start);
        }

        // Training forward pass (on what is now scratch state)
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

        // Apply BPTT gradients to all layers (weights persist past the restore)
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

        // Update projection weights (reads the training-state output — must
        // run before the live-state restore below)
        self.update_projections(&hdc_input, target, &output, learning_rate);

        // Restore live evolution state; weight updates survive.
        self.network.restore_state_from(&live);
        self.train_scratch = live;

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

        // Update output projection (simple gradient descent).
        // Forward is W·hv/‖hv‖ (scale restoration in project_from_hdc), so the
        // gradient w.r.t. W is error × hv/‖hv‖ — use the same normalization
        // here or the update direction is scaled inconsistently with the
        // forward pass (and collapses with the d^-1.9 annihilation).
        let hdc_output = self.network.output();
        let hdc_dim = self.config.hdc_dim;
        let state_norm = hdc_output.norm();
        if state_norm <= 1e-30 {
            return;
        }
        let inv_norm = 1.0 / state_norm;
        for i in 0..output_dim {
            for j in 0..hdc_dim {
                let grad = errors[i] * hdc_output.values[j] * inv_norm;
                self.output_projection[j * output_dim + i] -= learning_rate * grad;
            }
        }
    }

    /// Predict forward at a specific time horizon (matches CfCNetwork::predict_forward).
    ///
    /// PURE with respect to observable state: evolves the network at the given
    /// horizon, reads the projected output, then restores the pre-call neuron
    /// states and layer outputs from a reusable snapshot buffer. `current_output`,
    /// `total_steps`, and the diversity metric are untouched.
    ///
    /// History: this used to call `forward()` (a real state advance) and rely on
    /// the caller's read_state/inject "save/restore" — but `read_state` only
    /// captures the small projected output and `inject` RESETS the network, so
    /// every prediction wiped the temporal state. See
    /// docs/PHI_SIGNAL_TRACE_2026-07-15.md (PE ≡ 1.0 root cause).
    pub fn predict_forward(&mut self, input: &Array1<f32>, horizon: f32) -> Result<Array1<f32>> {
        let hdc_input = self.project_to_hdc(input);

        let mut scratch = std::mem::take(&mut self.predict_scratch);
        self.network.snapshot_state_into(&mut scratch);

        self.network.evolve_closed_form(horizon, &hdc_input);
        let output = self.project_from_hdc(&self.network.output());

        self.network.restore_state_from(&scratch);
        self.predict_scratch = scratch;

        Ok(Array1::from_vec(output))
    }

    /// Whether `predict_forward` leaves network state untouched (it does — see
    /// its doc). Callers can skip external save/restore when this is true.
    pub fn prediction_is_pure(&self) -> bool {
        true
    }

    /// Capture the network's full evolution state (for callers about to run a
    /// deliberately destructive operation such as consolidation replay).
    pub fn snapshot_evolution_state(&self) -> NetworkStateSnapshot {
        let mut snap = NetworkStateSnapshot::default();
        self.network.snapshot_state_into(&mut snap);
        snap
    }

    /// Restore evolution state captured by [`snapshot_evolution_state`].
    pub fn restore_evolution_state(&mut self, snap: &NetworkStateSnapshot) {
        self.network.restore_state_from(snap);
    }

    /// Inject state into the network (matches CfCNetwork::inject)
    ///
    /// WARNING: unlike CfCNetwork::inject, this cannot restore internal neuron
    /// state — the injected vector lives in the small projected output space.
    /// It RESETS the whole network and seeds only `current_output`. Use it for
    /// deliberate resets (e.g. clean replay), never as the "restore" half of a
    /// save/restore pattern.
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
                // LIVE per-neuron tau: τ ≈ τ₀ × (1 + backbone × ||state||),
                // matching the state-dependent term of the neuron's real
                // τ(||x||, u) (the ±20% input-similarity term is omitted —
                // the bridge doesn't retain the last input HV).
                //
                // HISTORY (2026-07-16): this used to return the CONFIG constant
                // τ₀ × (1 + backbone) per neuron — identical every cycle — so
                // downstream temporal coherence (1/(1+CV)) was frozen and Ψ,
                // whose dominant terms are coherence-derived, had ~0.01 dynamic
                // range (docs/PHI_SIGNAL_TRACE_2026-07-15.md follow-up 4).
                let layer_taus: Vec<f32> = layer
                    .iter()
                    .map(|n| {
                        n.config().tau_base * (1.0 + n.config().backbone_tau * n.state().norm())
                    })
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

    /// The d^-1.9 bind-chain annihilation (probe_signal_scale, 2026-07-18)
    /// must never silently return: after real steps on a unit-norm input, the
    /// bridge output must live at a sane order of magnitude — the readout is
    /// untrainable otherwise (gradients ~1e-13 pre-fix).
    #[test]
    fn output_scale_is_not_annihilated() {
        let config = HdcLtcBridgeConfig {
            input_dim: 8,
            output_dim: 8,
            layer_sizes: vec![2, 2],
            hdc_dim: 2048,
            seed: 11,
            ..Default::default()
        };
        let mut bridge = HdcLtcBridge::new(config);
        let raw: Vec<f32> = (0..8).map(|i| (i as f32 * 0.7).sin()).collect();
        let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        let input = Array1::from_vec(raw.iter().map(|x| x / norm).collect());
        for _ in 0..10 {
            bridge.step(&input, 0.02).unwrap();
        }
        let out = bridge.read_state().unwrap();
        let out_norm = out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            out_norm > 1e-3 && out_norm < 1e3,
            "bridge output norm {out_norm:.3e} is outside sane scale — \
             the bind-chain annihilation (or an explosion) is back"
        );
    }

    /// train_step must not perturb live evolution state (keystone Phase-3
    /// root cause: the old trainer evolved the LIVE network with the previous
    /// cycle's encoding every learning cycle, shuffling subjective time).
    ///
    /// Tests the property train_step actually promises: EVOLUTION STATE
    /// (neuron states, layer outputs, Fourier phase clocks) is bit-preserved
    /// across a training call — even at a real learning rate, where weights
    /// legitimately change. (An earlier twin-trajectory design assumed lr=0
    /// implies unchanged weights; false — apply_gradients applies weight
    /// decay independently of lr, a discovery of the 2026-07-18 scale work.)
    #[test]
    fn train_step_does_not_perturb_live_evolution_state() {
        let config = HdcLtcBridgeConfig {
            input_dim: 8,
            output_dim: 8,
            layer_sizes: vec![2, 2],
            hdc_dim: 256,
            seed: 7,
            ..Default::default()
        };
        let mut bridge = HdcLtcBridge::new(config);

        let warm = Array1::from_vec(vec![0.3; 8]);
        for _ in 0..3 {
            bridge.step(&warm, 0.02).unwrap();
        }

        let before = bridge.snapshot_evolution_state();
        let out_before = bridge.read_state().unwrap();

        let train_in = Array1::from_vec(vec![0.7; 8]);
        let train_target = Array1::from_vec(vec![-0.2; 8]);
        bridge
            .train_step(&train_in, &train_target, 0.02, 0.05)
            .unwrap();

        let after = bridge.snapshot_evolution_state();
        assert!(
            before.approx_eq(&after, 1e-7),
            "train_step perturbed live evolution state (states/outputs/clocks)"
        );
        let out_after = bridge.read_state().unwrap();
        for (x, y) in out_before.iter().zip(out_after.iter()) {
            assert!(
                (x - y).abs() < 1e-7,
                "train_step changed current_output: {x} vs {y}"
            );
        }
    }

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
    fn test_eval_loss_from_is_pure() {
        // Predictive Compression Program C2: eval_loss_from must leave every observable field
        // untouched -- same purity bar as train_step_from/predict_forward.
        let config = HdcLtcBridgeConfig::default();
        let mut bridge = HdcLtcBridge::new(config);

        let seed_input = Array1::from_vec(vec![0.4; 256]);
        let _ = bridge.step(&seed_input, 0.02);
        let start = bridge.snapshot_evolution_state();

        let before_output = bridge.read_state().unwrap();
        let before_steps = bridge.total_steps();
        let before_diversity = bridge.state_diversity();

        let input = Array1::from_vec(vec![0.5; 256]);
        let target = Array1::from_vec(vec![0.3; 256]);
        let _loss = bridge.eval_loss_from(&start, &input, &target, 0.02);

        assert_eq!(bridge.read_state().unwrap(), before_output);
        assert_eq!(bridge.total_steps(), before_steps);
        assert_eq!(bridge.state_diversity(), before_diversity);
    }

    #[test]
    fn test_eval_loss_from_matches_train_step_from_pre_update_loss() {
        // train_step_impl computes its returned loss BEFORE applying any gradient -- so calling
        // eval_loss_from first, then train_step_from with the identical (start, input, target,
        // dt), must return the same loss value train_step_from itself reports. This is the
        // correctness check that the new pure path measures the same thing the real training
        // path already computes internally, not a different formula.
        let config = HdcLtcBridgeConfig::default();
        let mut bridge = HdcLtcBridge::new(config);

        let seed_input = Array1::from_vec(vec![0.4; 256]);
        let _ = bridge.step(&seed_input, 0.02);
        let start = bridge.snapshot_evolution_state();

        let input = Array1::from_vec(vec![0.5; 256]);
        let target = Array1::from_vec(vec![0.3; 256]);

        let pre_loss = bridge.eval_loss_from(&start, &input, &target, 0.02);
        let train_reported_loss = bridge
            .train_step_from(&start, &input, &target, 0.02, 0.01)
            .unwrap();

        assert!(
            (pre_loss - train_reported_loss).abs() < 1e-6,
            "eval_loss_from={pre_loss}, train_step_from's own pre-update loss={train_reported_loss}"
        );
    }

    #[test]
    fn test_train_step_from_reduces_loss_pre_vs_post() {
        // The actual C2 signal: does a single training update on (start, input, target)
        // measurably reduce loss on that same triple? pre_loss and post_loss are both computed
        // via the pure eval_loss_from -- only the weights differ between the two calls.
        let config = HdcLtcBridgeConfig::default();
        let mut bridge = HdcLtcBridge::new(config);

        let seed_input = Array1::from_vec(vec![0.4; 256]);
        let _ = bridge.step(&seed_input, 0.02);
        let start = bridge.snapshot_evolution_state();

        let input = Array1::from_vec(vec![0.5; 256]);
        let target = Array1::from_vec(vec![0.3; 256]);

        let pre_loss = bridge.eval_loss_from(&start, &input, &target, 0.02);
        let _ = bridge
            .train_step_from(&start, &input, &target, 0.02, 0.01)
            .unwrap();
        let post_loss = bridge.eval_loss_from(&start, &input, &target, 0.02);

        assert!(
            post_loss < pre_loss,
            "expected a single training update to reduce loss on the same (start, input, \
             target) triple: pre={pre_loss}, post={post_loss}"
        );
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
