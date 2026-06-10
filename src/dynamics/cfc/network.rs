// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CfC Network: A complete multi-layer Closed-form Continuous-time neural network.
//!
//! Provides stacked CfC cells with output projection, BPTT/SPSA training,
//! online adaptation, Phi-gated attention, and dynamics diagnostics.

use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;

use super::cell::CfCCell;
use super::gradients::{AdamState, OutputAdamState};
use super::phi_gated::{PhiGatedConfig, compute_phi_attention_weights, weighted_array_bundle};
use super::types::{
    CfCConfig, DynamicsDiagnostic, MIN_TAU, NetworkOnlineLearningStats, OnlineLearningConfig,
    mse_loss,
};

/// Configuration for a CfC network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfCNetworkConfig {
    /// Input dimension
    pub input_dim: usize,

    /// Hidden dimension per layer
    pub hidden_dim: usize,

    /// Number of layers
    pub num_layers: usize,

    /// Output dimension
    pub output_dim: usize,

    /// Cell configuration
    pub cell_config: CfCConfig,

    /// Whether to use residual connections
    pub residual: bool,

    /// Whether to use bidirectional processing (doubles computation)
    pub bidirectional: bool,

    /// Whether to enable online learning during inference
    /// When true, the network will adapt weights based on prediction errors
    pub enable_online_learning: bool,

    /// Online learning configuration (used if enable_online_learning is true)
    pub online_learning_config: OnlineLearningConfig,
}

impl Default for CfCNetworkConfig {
    fn default() -> Self {
        let cell_config = CfCConfig {
            input_dim: 64,
            hidden_dim: 128,
            ..Default::default()
        };

        Self {
            input_dim: 64,
            hidden_dim: 128,
            num_layers: 2,
            output_dim: 32,
            cell_config,
            residual: true,
            bidirectional: false,
            enable_online_learning: false,
            online_learning_config: OnlineLearningConfig::default(),
        }
    }
}

impl CfCNetworkConfig {
    /// Create a configuration with online learning enabled
    pub fn with_online_learning(mut self) -> Self {
        self.enable_online_learning = true;
        // Propagate online learning config to cell config
        self.cell_config.online_learning = Some(self.online_learning_config.clone());
        self
    }

    /// Create a configuration with custom online learning settings
    pub fn with_online_learning_config(mut self, config: OnlineLearningConfig) -> Self {
        self.enable_online_learning = true;
        self.online_learning_config = config.clone();
        self.cell_config.online_learning = Some(config);
        self
    }
}

/// A complete CfC neural network
#[derive(Debug, Clone)]
pub struct CfCNetwork {
    /// Network configuration
    pub(crate) config: CfCNetworkConfig,

    /// Stack of CfC cells
    pub(crate) cells: Vec<CfCCell>,

    /// Output projection weights
    pub(crate) output_weights: Array2<f32>,
    pub(crate) output_bias: Array1<f32>,

    /// Statistics
    pub(crate) total_steps: u64,

    /// Adam optimizer states per cell
    pub(crate) adam_states: Vec<AdamState>,
    /// Adam state for output projection
    pub(crate) adam_output: OutputAdamState,

    /// Online learning statistics for the network
    pub(crate) online_stats: NetworkOnlineLearningStats,
}

impl CfCNetwork {
    /// Create a new CfC network
    pub fn new(config: CfCNetworkConfig) -> Self {
        let mut cells = Vec::with_capacity(config.num_layers);

        for i in 0..config.num_layers {
            let cell_config = CfCConfig {
                input_dim: if i == 0 {
                    config.input_dim
                } else {
                    config.hidden_dim
                },
                hidden_dim: config.hidden_dim,
                ..config.cell_config.clone()
            };
            cells.push(CfCCell::new(cell_config));
        }

        let scale = (2.0 / (config.hidden_dim + config.output_dim) as f32).sqrt();
        let output_weights = Array2::from_shape_fn((config.output_dim, config.hidden_dim), |_| {
            (rand::random::<f32>() - 0.5) * 2.0 * scale
        });
        let output_bias = Array1::zeros(config.output_dim);

        let adam_states = cells
            .iter()
            .map(|c| {
                let effective_input_dim = if c.config.use_backbone {
                    c.config.backbone_dim
                } else {
                    c.config.input_dim
                };
                AdamState::new(c.config.hidden_dim, effective_input_dim)
            })
            .collect();

        let adam_output = OutputAdamState::new(config.output_dim, config.hidden_dim);

        Self {
            config,
            cells,
            output_weights,
            output_bias,
            total_steps: 0,
            adam_states,
            adam_output,
            online_stats: NetworkOnlineLearningStats::default(),
        }
    }

    /// Create a new CfC network with deterministic weight initialization from a genesis seed.
    ///
    /// Each cell gets a unique domain label `"{label}::cell_{i}"` and the output
    /// projection uses `"{label}::output"`.
    pub fn from_genesis(config: CfCNetworkConfig, genesis: &GenesisSeed, label: &str) -> Self {
        let mut cells = Vec::with_capacity(config.num_layers);

        for i in 0..config.num_layers {
            let cell_config = CfCConfig {
                input_dim: if i == 0 {
                    config.input_dim
                } else {
                    config.hidden_dim
                },
                hidden_dim: config.hidden_dim,
                ..config.cell_config.clone()
            };
            let cell_label = format!("{label}::cell_{i}");
            cells.push(CfCCell::from_genesis(cell_config, genesis, &cell_label));
        }

        let mut rng = genesis.domain(&format!("{label}::output"));
        let scale = (2.0 / (config.hidden_dim + config.output_dim) as f32).sqrt();
        let output_weights = Array2::from_shape_fn((config.output_dim, config.hidden_dim), |_| {
            (rng.r#gen::<f32>() - 0.5) * 2.0 * scale
        });
        let output_bias = Array1::zeros(config.output_dim);

        let adam_states = cells
            .iter()
            .map(|c| {
                let effective_input_dim = if c.config.use_backbone {
                    c.config.backbone_dim
                } else {
                    c.config.input_dim
                };
                AdamState::new(c.config.hidden_dim, effective_input_dim)
            })
            .collect();

        let adam_output = OutputAdamState::new(config.output_dim, config.hidden_dim);

        Self {
            config,
            cells,
            output_weights,
            output_bias,
            total_steps: 0,
            adam_states,
            adam_output,
            online_stats: NetworkOnlineLearningStats::default(),
        }
    }

    /// Reset all cell states
    pub fn reset(&mut self) {
        for cell in &mut self.cells {
            cell.reset();
        }
        self.total_steps = 0;
    }

    /// Reset online learning statistics (keeps weights but clears tracking)
    pub fn reset_online_stats(&mut self) {
        self.online_stats = NetworkOnlineLearningStats::default();
        for cell in &mut self.cells {
            cell.reset_online_stats();
        }
    }

    /// Get online learning statistics for the network
    pub fn online_stats(&self) -> &NetworkOnlineLearningStats {
        &self.online_stats
    }

    /// Check if online learning is enabled
    pub fn online_learning_enabled(&self) -> bool {
        self.config.enable_online_learning
    }

    /// Forward pass through the network
    pub fn forward(&mut self, input: &Array1<f32>, dt: f32) -> Array1<f32> {
        let mut h = input.clone();

        for (i, cell) in self.cells.iter_mut().enumerate() {
            let prev_h = h.clone();
            h = cell.forward(&h, dt);

            // Add residual connection if enabled and dimensions match
            if self.config.residual && i > 0 && prev_h.len() == h.len() {
                h = &h + &prev_h;
            }
        }

        // Project to output dimension
        let output = self.output_weights.dot(&h) + &self.output_bias;
        self.total_steps += 1;

        output
    }

    /// Process a sequence of inputs.
    ///
    /// If `inputs` and `dts` have different lengths, processes only up to the
    /// shorter length (truncation is safer than panicking in a hot path).
    pub fn forward_sequence(&mut self, inputs: &[Array1<f32>], dts: &[f32]) -> Vec<Array1<f32>> {
        debug_assert_eq!(
            inputs.len(),
            dts.len(),
            "inputs/dts length mismatch in forward_sequence"
        );

        self.reset();
        inputs
            .iter()
            .zip(dts.iter())
            .map(|(input, dt)| self.forward(input, *dt))
            .collect()
    }

    /// Get the current state of all cells
    pub fn state(&self) -> Vec<Array1<f32>> {
        self.cells.iter().map(|c| c.state().clone()).collect()
    }

    /// Set the state of all cells
    pub fn set_state(&mut self, states: Vec<Array1<f32>>) {
        for (cell, state) in self.cells.iter_mut().zip(states.into_iter()) {
            cell.set_state(state);
        }
    }

    /// Get network configuration
    pub fn config(&self) -> &CfCNetworkConfig {
        &self.config
    }

    /// Get number of parameters
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        for cell in &self.cells {
            let cfg = cell.config();
            count += cfg.input_dim * cfg.hidden_dim; // w_in
            count += cfg.hidden_dim * cfg.hidden_dim; // w_h
            count += cfg.hidden_dim; // b_h
            count += cfg.hidden_dim; // tau
        }
        count += self.config.hidden_dim * self.config.output_dim; // output_weights
        count += self.config.output_dim; // output_bias
        count
    }

    // =========================================================================
    // Online Learning During Inference
    // =========================================================================

    /// Adapt network weights online based on prediction error.
    pub fn adapt_online(
        &mut self,
        prediction_error: f32,
        input: &Array1<f32>,
        target: &Array1<f32>,
        dt: f32,
    ) -> bool {
        if !self.config.enable_online_learning {
            return false;
        }

        self.online_stats.total_adaptation_calls += 1;

        // Update network-level EMA error
        let alpha = self.config.online_learning_config.ema_alpha;
        self.online_stats.ema_error =
            self.online_stats.ema_error * (1.0 - alpha) + prediction_error * alpha;

        // Check error threshold at network level
        let threshold = self
            .config
            .online_learning_config
            .error_threshold
            .max(self.online_stats.ema_error * 0.5);

        if prediction_error < threshold {
            self.online_stats.adaptations_skipped += 1;
            return false;
        }

        let lr = self.config.online_learning_config.learning_rate;
        let max_delta = self.config.online_learning_config.max_weight_delta;
        let mut total_delta_sq = 0.0f32;

        // Step 1: Forward pass to get intermediate states
        let mut layer_inputs: Vec<Array1<f32>> = Vec::with_capacity(self.cells.len() + 1);
        layer_inputs.push(input.clone());

        let mut h = input.clone();
        for cell in &mut self.cells {
            h = cell.forward(&h, dt);
            layer_inputs.push(h.clone());
        }

        // Step 2: Compute output and output gradient
        let last_hidden = layer_inputs
            .last()
            .expect("layer_inputs should not be empty after forward pass");
        let output = self.output_weights.dot(last_hidden) + &self.output_bias;
        let n = target.len() as f32;
        let output_error: Array1<f32> = (0..self.config.output_dim)
            .map(|i| 2.0 * (output[i] - target[i]) / n)
            .collect();

        // Step 3: Adapt output layer weights
        for i in 0..self.config.output_dim {
            for j in 0..self.config.hidden_dim {
                let grad = output_error[i] * last_hidden[j];
                let delta = (-lr * grad).clamp(-max_delta, max_delta);
                self.output_weights[[i, j]] += delta;
                total_delta_sq += delta * delta;
            }
            let delta = (-lr * output_error[i]).clamp(-max_delta, max_delta);
            self.output_bias[i] += delta;
            total_delta_sq += delta * delta;
        }

        // Step 4: Backpropagate gradient through output layer to get dL/dh_last
        let dh_last = self.output_weights.t().dot(&output_error);

        // Step 5: Backpropagate through cells using backward_from_grad
        let mut dh = dh_last;
        for i in (0..self.cells.len()).rev() {
            let cell_input = &layer_inputs[i];
            let grads = self.cells[i].backward_from_grad(cell_input, &dh, dt);

            // Apply gradients with online learning constraints
            let (rows_in, cols_in) = self.cells[i].w_in.dim();
            for r in 0..rows_in {
                for c in 0..cols_in {
                    let delta = (-lr * grads.dw_in[[r, c]]).clamp(-max_delta, max_delta);
                    self.cells[i].w_in[[r, c]] += delta;
                    total_delta_sq += delta * delta;
                }
            }

            let hidden_dim = self.cells[i].config.hidden_dim;
            for r in 0..hidden_dim {
                for c in 0..hidden_dim {
                    let delta = (-lr * grads.dw_h[[r, c]]).clamp(-max_delta, max_delta);
                    self.cells[i].w_h[[r, c]] += delta;
                    total_delta_sq += delta * delta;
                }
            }

            for r in 0..hidden_dim {
                let delta = (-lr * grads.db_h[r]).clamp(-max_delta, max_delta);
                self.cells[i].b_h[r] += delta;
                total_delta_sq += delta * delta;
            }

            // Optionally adapt tau
            if self.config.online_learning_config.adapt_tau {
                let tau_lr = lr * self.config.online_learning_config.tau_lr_multiplier;
                for r in 0..hidden_dim {
                    let delta = (-tau_lr * grads.dtau[r]).clamp(-max_delta * 0.1, max_delta * 0.1);
                    self.cells[i].tau[r] = (self.cells[i].tau[r] + delta).clamp(MIN_TAU, 100.0);
                    total_delta_sq += delta * delta;
                }
            }

            // Propagate gradient to previous layer
            dh = self.cells[i].w_h.t().dot(&dh);
        }

        self.online_stats.adaptations_applied += 1;
        self.online_stats.cumulative_weight_change += total_delta_sq.sqrt();

        true
    }

    /// Forward pass with automatic online adaptation.
    pub fn forward_with_adaptation(
        &mut self,
        input: &Array1<f32>,
        dt: f32,
        target: Option<&Array1<f32>>,
    ) -> (Array1<f32>, bool) {
        let output = self.forward(input, dt);

        let adapted = if let Some(tgt) = target {
            let error = mse_loss(&output, tgt);
            self.adapt_online(error, input, tgt, dt)
        } else {
            false
        };

        (output, adapted)
    }

    // =========================================================================
    // Cognitive Loop Compatibility Methods
    // =========================================================================

    /// Step the network forward (alias for forward, returns unit)
    pub fn step(&mut self, input: &Array1<f32>, dt: f32) -> anyhow::Result<()> {
        let _ = self.forward(input, dt);
        Ok(())
    }

    /// Read the current state (returns Result for cognitive_loop compatibility)
    pub fn read_state(&self) -> anyhow::Result<Array1<f32>> {
        if let Some(cell) = self.cells.last() {
            Ok(cell.state().clone())
        } else {
            Ok(Array1::zeros(self.config.hidden_dim))
        }
    }

    /// Train step using BPTT with Adam optimizer (default training method)
    pub fn train_step(
        &mut self,
        input: &Array1<f32>,
        target: &Array1<f32>,
        dt: f32,
        learning_rate: f32,
    ) -> anyhow::Result<f32> {
        self.train_step_bptt_optimized(&[input.clone()], &[target.clone()], &[dt], learning_rate)
    }

    /// Sequence training with BPTT and Adam
    pub fn train_step_bptt(
        &mut self,
        inputs: &[Array1<f32>],
        targets: &[Array1<f32>],
        dts: &[f32],
        learning_rate: f32,
    ) -> anyhow::Result<f32> {
        anyhow::ensure!(
            inputs.len() == targets.len(),
            "inputs/targets length mismatch: {} vs {}",
            inputs.len(),
            targets.len()
        );
        anyhow::ensure!(
            inputs.len() == dts.len(),
            "inputs/dts length mismatch: {} vs {}",
            inputs.len(),
            dts.len()
        );

        let mut total_loss = 0.0f32;

        let clip_val = self.cells[0].config.gradient_clip;
        let clip = |g: f32| g.clamp(-clip_val, clip_val);

        for ((_input, target), dt) in inputs.iter().zip(targets.iter()).zip(dts.iter()) {
            let mut h = _input.clone();
            let mut cell_inputs: Vec<Array1<f32>> = Vec::with_capacity(self.cells.len());
            for cell in self.cells.iter_mut() {
                cell_inputs.push(h.clone());
                h = cell.forward(&h, *dt);
            }

            let output = self.output_weights.dot(&h) + &self.output_bias;
            let loss = mse_loss(&output, target);
            total_loss += loss;

            let n = output.len().min(target.len()) as f32;
            let mut d_output = Array1::zeros(self.config.output_dim);
            for i in 0..output.len().min(target.len()) {
                d_output[i] = 2.0 * (output[i] - target[i]) / n;
            }

            let output_dim = self.config.output_dim;
            let hidden_dim = self.config.hidden_dim;
            self.adam_output.t += 1;
            let t_adam = self.adam_output.t as f32;

            for i in 0..output_dim {
                for j in 0..hidden_dim {
                    let g = clip(d_output[i] * h[j]);
                    self.adam_output.m_w[[i, j]] = self.adam_output.beta1
                        * self.adam_output.m_w[[i, j]]
                        + (1.0 - self.adam_output.beta1) * g;
                    self.adam_output.v_w[[i, j]] = self.adam_output.beta2
                        * self.adam_output.v_w[[i, j]]
                        + (1.0 - self.adam_output.beta2) * g * g;
                    let m_hat =
                        self.adam_output.m_w[[i, j]] / (1.0 - self.adam_output.beta1.powf(t_adam));
                    let v_hat =
                        self.adam_output.v_w[[i, j]] / (1.0 - self.adam_output.beta2.powf(t_adam));
                    self.output_weights[[i, j]] -=
                        learning_rate * m_hat / (v_hat.sqrt() + self.adam_output.eps);
                }
            }

            for i in 0..output_dim {
                let g = clip(d_output[i]);
                self.adam_output.m_b[i] = self.adam_output.beta1 * self.adam_output.m_b[i]
                    + (1.0 - self.adam_output.beta1) * g;
                self.adam_output.v_b[i] = self.adam_output.beta2 * self.adam_output.v_b[i]
                    + (1.0 - self.adam_output.beta2) * g * g;
                let m_hat = self.adam_output.m_b[i] / (1.0 - self.adam_output.beta1.powf(t_adam));
                let v_hat = self.adam_output.v_b[i] / (1.0 - self.adam_output.beta2.powf(t_adam));
                self.output_bias[i] -=
                    learning_rate * m_hat / (v_hat.sqrt() + self.adam_output.eps);
            }

            let dh_last = self.output_weights.t().dot(&d_output);

            let mut dh = dh_last;
            for cell_idx in (0..self.cells.len()).rev() {
                let grads =
                    self.cells[cell_idx].backward_from_grad(&cell_inputs[cell_idx], &dh, *dt);
                self.cells[cell_idx].apply_adam(
                    &grads,
                    &mut self.adam_states[cell_idx],
                    learning_rate,
                );

                if cell_idx > 0 {
                    let decay: Array1<f32> = self.cells[cell_idx]
                        .tau
                        .mapv(|t| (-dt / t.max(MIN_TAU)).exp());
                    let one_minus_decay: Array1<f32> = decay.mapv(|d| 1.0 - d);
                    let _attenuation = one_minus_decay.mean().unwrap_or(0.5);
                    dh *= _attenuation;
                    if dh.len() != self.cells[cell_idx - 1].config.hidden_dim {
                        let prev_dim = self.cells[cell_idx - 1].config.hidden_dim;
                        let mut resized = Array1::zeros(prev_dim);
                        for i in 0..prev_dim.min(dh.len()) {
                            resized[i] = dh[i];
                        }
                        dh = resized;
                    }
                }
            }
        }

        let avg_loss = total_loss / inputs.len() as f32;
        self.clamp_all_weights();

        if !avg_loss.is_finite() {
            return Ok(1.0);
        }

        Ok(avg_loss)
    }

    /// Optimized BPTT training step with cached forward pass and vectorized Adam.
    pub fn train_step_bptt_optimized(
        &mut self,
        inputs: &[Array1<f32>],
        targets: &[Array1<f32>],
        dts: &[f32],
        learning_rate: f32,
    ) -> anyhow::Result<f32> {
        anyhow::ensure!(
            inputs.len() == targets.len(),
            "inputs/targets length mismatch: {} vs {}",
            inputs.len(),
            targets.len()
        );
        anyhow::ensure!(
            inputs.len() == dts.len(),
            "inputs/dts length mismatch: {} vs {}",
            inputs.len(),
            dts.len()
        );

        let mut total_loss = 0.0f32;

        // CONTIGUITY SAFETY: All .expect("... not contiguous") calls in this function
        // are safe. ndarray Array1/Array2 created via ::zeros() or stored as struct
        // fields use default C-contiguous layout. as_slice()/as_slice_mut() only fails
        // on non-contiguous views (transposed 2D, strided subviews). Every array here
        // is either freshly allocated or a contiguous struct field.
        self.adam_output.t += 1;
        let t_adam = self.adam_output.t as f32;
        let bc1_out = 1.0 - self.adam_output.beta1.powf(t_adam);
        let bc2_out = 1.0 - self.adam_output.beta2.powf(t_adam);
        let one_minus_beta1_out = 1.0 - self.adam_output.beta1;
        let one_minus_beta2_out = 1.0 - self.adam_output.beta2;
        let beta1_out = self.adam_output.beta1;
        let beta2_out = self.adam_output.beta2;
        let eps_out = self.adam_output.eps;

        let output_dim = self.config.output_dim;
        let hidden_dim = self.config.hidden_dim;

        for ((_input, target), dt) in inputs.iter().zip(targets.iter()).zip(dts.iter()) {
            let mut h = _input.clone();
            let mut cell_caches = Vec::with_capacity(self.cells.len());

            for cell in self.cells.iter_mut() {
                let (new_h, cache) = cell.forward_with_cache(&h, *dt);
                cell_caches.push(cache);
                h = new_h;
            }

            let output = self.output_weights.dot(&h) + &self.output_bias;
            let loss = mse_loss(&output, target);
            total_loss += loss;

            let n = output.len().min(target.len()) as f32;
            let mut d_output = Array1::zeros(output_dim);
            for i in 0..output.len().min(target.len()) {
                d_output[i] = 2.0 * (output[i] - target[i]) / n;
            }

            {
                let h_slice = h.as_slice().expect("hidden state array not contiguous");
                let d_out_slice = d_output.as_slice().expect("d_output array not contiguous");
                let w_slice = self
                    .output_weights
                    .as_slice_mut()
                    .expect("output_weights array not contiguous");
                let m_slice = self
                    .adam_output
                    .m_w
                    .as_slice_mut()
                    .expect("adam m_w array not contiguous");
                let v_slice = self
                    .adam_output
                    .v_w
                    .as_slice_mut()
                    .expect("adam v_w array not contiguous");

                for i in 0..output_dim {
                    let row_offset = i * hidden_dim;
                    let d_out_i = d_out_slice[i];
                    for j in 0..hidden_dim {
                        let idx = row_offset + j;
                        let g = (d_out_i * h_slice[j]).clamp(-0.5, 0.5);
                        m_slice[idx] = beta1_out * m_slice[idx] + one_minus_beta1_out * g;
                        v_slice[idx] = beta2_out * v_slice[idx] + one_minus_beta2_out * g * g;
                        let m_hat = m_slice[idx] / bc1_out;
                        let v_hat = v_slice[idx] / bc2_out;
                        w_slice[idx] -= learning_rate * m_hat / (v_hat.sqrt() + eps_out);
                    }
                }
            }

            {
                let d_out_slice = d_output.as_slice().expect("d_output array not contiguous");
                let b_slice = self
                    .output_bias
                    .as_slice_mut()
                    .expect("output_bias array not contiguous");
                let m_slice = self
                    .adam_output
                    .m_b
                    .as_slice_mut()
                    .expect("adam m_b array not contiguous");
                let v_slice = self
                    .adam_output
                    .v_b
                    .as_slice_mut()
                    .expect("adam v_b array not contiguous");

                for i in 0..output_dim {
                    let g = d_out_slice[i].clamp(-0.5, 0.5);
                    m_slice[i] = beta1_out * m_slice[i] + one_minus_beta1_out * g;
                    v_slice[i] = beta2_out * v_slice[i] + one_minus_beta2_out * g * g;
                    let m_hat = m_slice[i] / bc1_out;
                    let v_hat = v_slice[i] / bc2_out;
                    b_slice[i] -= learning_rate * m_hat / (v_hat.sqrt() + eps_out);
                }
            }

            let dh_last = self.output_weights.t().dot(&d_output);

            let mut dh = dh_last;
            for cell_idx in (0..self.cells.len()).rev() {
                let grads =
                    self.cells[cell_idx].backward_from_cache(&cell_caches[cell_idx], &dh, *dt);
                self.cells[cell_idx].apply_adam_vectorized(
                    &grads,
                    &mut self.adam_states[cell_idx],
                    learning_rate,
                );

                if cell_idx > 0 {
                    let decay = &cell_caches[cell_idx].decay;
                    let one_minus_decay: Array1<f32> = decay.mapv(|d| 1.0 - d);
                    let attenuation = one_minus_decay.mean().unwrap_or(0.5).max(0.3);
                    dh *= attenuation;

                    if dh.len() != self.cells[cell_idx - 1].config.hidden_dim {
                        let prev_dim = self.cells[cell_idx - 1].config.hidden_dim;
                        let mut resized = Array1::zeros(prev_dim);
                        for i in 0..prev_dim.min(dh.len()) {
                            resized[i] = dh[i];
                        }
                        dh = resized;
                    }
                }
            }
        }

        let avg_loss = total_loss / inputs.len() as f32;
        self.clamp_all_weights();

        if !avg_loss.is_finite() {
            return Ok(1.0);
        }

        Ok(avg_loss)
    }

    /// Train step using perturbation-based gradient estimation (SPSA).
    pub fn train_step_spsa(
        &mut self,
        input: &Array1<f32>,
        target: &Array1<f32>,
        dt: f32,
        learning_rate: f32,
    ) -> anyhow::Result<f32> {
        let baseline_output = self.forward(input, dt);
        let baseline_loss = mse_loss(&baseline_output, target);

        let epsilon = 0.01f32;

        self.update_output_weights(input, target, dt, learning_rate, epsilon, baseline_loss);

        for cell_idx in 0..self.cells.len() {
            self.update_cell_weights(
                cell_idx,
                input,
                target,
                dt,
                learning_rate,
                epsilon,
                baseline_loss,
            );
        }

        self.clamp_all_weights();

        self.reset_states_only();
        let final_output = self.forward(input, dt);
        let final_loss = mse_loss(&final_output, target);

        if !final_loss.is_finite() {
            return Ok(1.0);
        }

        Ok(final_loss)
    }

    /// Update output projection weights via perturbation
    fn update_output_weights(
        &mut self,
        input: &Array1<f32>,
        target: &Array1<f32>,
        dt: f32,
        lr: f32,
        epsilon: f32,
        baseline_loss: f32,
    ) {
        let (rows, cols) = self.output_weights.dim();

        let stride = (rows * cols / 32).max(1);
        for idx in (0..rows * cols).step_by(stride) {
            let r = idx / cols;
            let c = idx % cols;

            self.output_weights[[r, c]] += epsilon;
            self.reset_states_only();
            let output_pos = self.forward(input, dt);
            let loss_pos = mse_loss(&output_pos, target);
            self.output_weights[[r, c]] -= epsilon;

            let grad = (loss_pos - baseline_loss) / epsilon;
            self.output_weights[[r, c]] -= lr * grad;
        }

        for j in 0..self.output_bias.len() {
            self.output_bias[j] += epsilon;
            self.reset_states_only();
            let output_pos = self.forward(input, dt);
            let loss_pos = mse_loss(&output_pos, target);
            self.output_bias[j] -= epsilon;

            let grad = (loss_pos - baseline_loss) / epsilon;
            self.output_bias[j] -= lr * grad;
        }
    }

    /// Update a single CfC cell's weights via perturbation
    fn update_cell_weights(
        &mut self,
        cell_idx: usize,
        input: &Array1<f32>,
        target: &Array1<f32>,
        dt: f32,
        lr: f32,
        epsilon: f32,
        baseline_loss: f32,
    ) {
        let hidden_dim = self.cells[cell_idx].config.hidden_dim;

        // Update tau
        for j in 0..hidden_dim {
            let old_tau = self.cells[cell_idx].tau[j];
            self.cells[cell_idx].tau[j] = old_tau + epsilon;
            self.reset_states_only();
            let output_pos = self.forward(input, dt);
            let loss_pos = mse_loss(&output_pos, target);
            self.cells[cell_idx].tau[j] = old_tau;

            let grad = (loss_pos - baseline_loss) / epsilon;
            let new_tau = (old_tau - lr * grad).max(MIN_TAU);
            self.cells[cell_idx].tau[j] = new_tau;
        }

        // Update bias
        for j in 0..hidden_dim {
            self.cells[cell_idx].b_h[j] += epsilon;
            self.reset_states_only();
            let output_pos = self.forward(input, dt);
            let loss_pos = mse_loss(&output_pos, target);
            self.cells[cell_idx].b_h[j] -= epsilon;

            let grad = (loss_pos - baseline_loss) / epsilon;
            self.cells[cell_idx].b_h[j] -= lr * grad;
        }

        // Update W_h (sparse)
        let stride = (hidden_dim * hidden_dim / 16).max(1);
        for idx in (0..hidden_dim * hidden_dim).step_by(stride) {
            let r = idx / hidden_dim;
            let c = idx % hidden_dim;

            self.cells[cell_idx].w_h[[r, c]] += epsilon;
            self.reset_states_only();
            let output_pos = self.forward(input, dt);
            let loss_pos = mse_loss(&output_pos, target);
            self.cells[cell_idx].w_h[[r, c]] -= epsilon;

            let grad = (loss_pos - baseline_loss) / epsilon;
            self.cells[cell_idx].w_h[[r, c]] -= lr * grad;
        }
    }

    /// Clamp all network weights to [-2, 2] to prevent divergence.
    pub(crate) fn clamp_all_weights(&mut self) {
        let clamp_val = |x: &mut f32| {
            if !x.is_finite() {
                *x = 0.0;
            } else {
                *x = x.clamp(-2.0, 2.0);
            }
        };

        for cell in &mut self.cells {
            cell.w_in.iter_mut().for_each(&clamp_val);
            cell.w_h.iter_mut().for_each(&clamp_val);
            cell.b_h.iter_mut().for_each(&clamp_val);
            cell.tau.mapv_inplace(|t| t.clamp(MIN_TAU, 100.0));
            for w in &mut cell.backbone_weights {
                w.iter_mut().for_each(&clamp_val);
            }
            for b in &mut cell.backbone_biases {
                b.iter_mut().for_each(&clamp_val);
            }
        }

        self.output_weights.iter_mut().for_each(&clamp_val);
        self.output_bias.iter_mut().for_each(&clamp_val);
    }

    /// Reset cell hidden states without resetting step counters
    pub(crate) fn reset_states_only(&mut self) {
        for cell in &mut self.cells {
            cell.state = Array1::zeros(cell.config.hidden_dim);
        }
    }

    /// Compute state diversity across CfC cells.
    pub fn state_diversity(&self) -> f32 {
        let states: Vec<&Array1<f32>> = self.cells.iter().map(|c| c.state()).collect();
        if states.is_empty() {
            return 0.0;
        }

        let mean_activity: f32 = states.iter().flat_map(|s| s.iter()).sum::<f32>()
            / (states.len() * self.config.hidden_dim) as f32;

        let variance: f32 = states
            .iter()
            .flat_map(|s| s.iter())
            .map(|x| (x - mean_activity).powi(2))
            .sum::<f32>()
            / (states.len() * self.config.hidden_dim) as f32;

        1.0 / (1.0 + (-variance.sqrt() * 10.0).exp())
    }

    /// Compute consciousness level using Phi-inspired metric
    pub fn consciousness_level(&self) -> f32 {
        use symthaea_core::hdc::unified_hv::ContinuousHV;
        use symthaea_core::phi_engine::{PhiEngine, PhiMethod};

        let states: Vec<&Array1<f32>> = self.cells.iter().map(|c| c.state()).collect();
        if states.is_empty() {
            return 0.0;
        }

        let mut node_representations = Vec::new();
        for state in &states {
            let step = (state.len() / 8).max(1);
            for i in (0..state.len()).step_by(step).take(8) {
                let mut components = vec![0.0f32; 16];
                for j in 0..16 {
                    let idx = (i + j) % state.len();
                    components[j] = state[idx];
                }
                node_representations.push(ContinuousHV::from_vec(components));
            }
        }

        if node_representations.is_empty() {
            return 0.0;
        }

        node_representations.truncate(16);

        let engine = PhiEngine::new(PhiMethod::Auto);
        let result = engine.compute(&node_representations);
        result.phi as f32
    }

    /// Predict forward at a specific time horizon
    pub fn predict_forward(
        &mut self,
        input: &Array1<f32>,
        horizon: f32,
    ) -> anyhow::Result<Array1<f32>> {
        Ok(self.forward(input, horizon))
    }

    /// Inject state into the network (alias for set_state)
    pub fn inject(&mut self, state: &Array1<f32>) -> anyhow::Result<()> {
        for cell in &mut self.cells {
            cell.set_state(state.clone());
        }
        Ok(())
    }

    /// Create with specific input dimension (for cognitive_loop compatibility)
    pub fn new_with_input(input_dim: usize, hidden_dim: usize) -> Self {
        let config = CfCNetworkConfig {
            input_dim,
            hidden_dim,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Get all tau (time constant) values across all cells
    pub fn all_tau(&self) -> Vec<&Array1<f32>> {
        self.cells.iter().map(|cell| cell.tau()).collect()
    }

    /// Get flattened tau values as a single vector
    pub fn flattened_tau(&self) -> Vec<f32> {
        self.cells
            .iter()
            .flat_map(|cell| cell.tau().iter().cloned())
            .collect()
    }

    /// Scale tau values for a specific layer
    pub fn scale_tau(&mut self, layer_idx: usize, scale: f32) {
        if let Some(cell) = self.cells.get_mut(layer_idx) {
            cell.scale_tau(scale);
        }
    }

    /// Scale tau values for all layers uniformly
    pub fn scale_tau_all(&mut self, scale: f32) {
        for cell in &mut self.cells {
            cell.scale_tau(scale);
        }
    }

    // =========================================================================
    // Weight serialization for async training (background thread weight swap)
    // =========================================================================

    /// Extract all learnable weights into a flat `Vec<f32>`.
    pub fn get_weights(&self) -> Vec<f32> {
        let mut buf = Vec::new();
        for cell in &self.cells {
            buf.extend(cell.w_in.iter());
            buf.extend(cell.w_h.iter());
            buf.extend(cell.b_h.iter());
            buf.extend(cell.tau.iter());
            for bw in &cell.backbone_weights {
                buf.extend(bw.iter());
            }
            for bb in &cell.backbone_biases {
                buf.extend(bb.iter());
            }
        }
        buf.extend(self.output_weights.iter());
        buf.extend(self.output_bias.iter());
        buf
    }

    /// Restore learnable weights from a flat slice produced by [`Self::get_weights`].
    pub fn set_weights(&mut self, weights: &[f32]) {
        let mut pos = 0;
        for cell in &mut self.cells {
            let n = cell.w_in.len();
            cell.w_in
                .as_slice_mut()
                .expect("w_in array not contiguous")
                .copy_from_slice(&weights[pos..pos + n]);
            pos += n;
            let n = cell.w_h.len();
            cell.w_h
                .as_slice_mut()
                .expect("w_h array not contiguous")
                .copy_from_slice(&weights[pos..pos + n]);
            pos += n;
            let n = cell.b_h.len();
            cell.b_h
                .as_slice_mut()
                .expect("b_h array not contiguous")
                .copy_from_slice(&weights[pos..pos + n]);
            pos += n;
            let n = cell.tau.len();
            cell.tau
                .as_slice_mut()
                .expect("tau array not contiguous")
                .copy_from_slice(&weights[pos..pos + n]);
            pos += n;
            for bw in &mut cell.backbone_weights {
                let n = bw.len();
                bw.as_slice_mut()
                    .expect("backbone_weight array not contiguous")
                    .copy_from_slice(&weights[pos..pos + n]);
                pos += n;
            }
            for bb in &mut cell.backbone_biases {
                let n = bb.len();
                bb.as_slice_mut()
                    .expect("backbone_bias array not contiguous")
                    .copy_from_slice(&weights[pos..pos + n]);
                pos += n;
            }
        }
        let n = self.output_weights.len();
        self.output_weights
            .as_slice_mut()
            .expect("output_weights array not contiguous")
            .copy_from_slice(&weights[pos..pos + n]);
        pos += n;
        let n = self.output_bias.len();
        self.output_bias
            .as_slice_mut()
            .expect("output_bias array not contiguous")
            .copy_from_slice(&weights[pos..pos + n]);
        pos += n;
        assert_eq!(pos, weights.len(), "weight count mismatch");
    }

    // =========================================================================
    // Phi-Guided Attention for Multiple Inputs
    // =========================================================================

    /// Forward pass with Phi-guided attention for multiple inputs.
    pub fn forward_phi_gated(
        &mut self,
        inputs: &[Array1<f32>],
        phi_values: &[f64],
        dt: f32,
        phi_config: Option<PhiGatedConfig>,
    ) -> (Array1<f32>, Vec<f32>) {
        if inputs.is_empty() || phi_values.is_empty() {
            return (Array1::zeros(self.config.output_dim), vec![]);
        }

        assert_eq!(
            inputs.len(),
            phi_values.len(),
            "Number of inputs must match number of Phi values"
        );

        let config = phi_config.unwrap_or_default();
        let weights = compute_phi_attention_weights(phi_values, &config);
        let combined_input = weighted_array_bundle(inputs, &weights);
        let output = self.forward(&combined_input, dt);

        (output, weights)
    }

    /// Forward pass with Phi gating enabled by a boolean flag.
    pub fn forward_with_phi_option(
        &mut self,
        inputs: &[Array1<f32>],
        phi_values: &[f64],
        dt: f32,
        phi_gated: bool,
    ) -> (Array1<f32>, Vec<f32>) {
        if phi_gated && inputs.len() > 1 {
            self.forward_phi_gated(inputs, phi_values, dt, None)
        } else if !inputs.is_empty() {
            (self.forward(&inputs[0], dt), vec![1.0])
        } else {
            (Array1::zeros(self.config.output_dim), vec![])
        }
    }

    /// Diagnose the current dynamics of the CfC network.
    pub fn diagnose_dynamics(&mut self, input: &Array1<f32>, dt: f32) -> DynamicsDiagnostic {
        let epsilon = 1e-4f32;
        let dim = self.config.hidden_dim;

        let saved_states: Vec<Array1<f32>> = self.cells.iter().map(|c| c.state().clone()).collect();

        let baseline_output = self.forward(input, dt);
        let baseline: Vec<f32> = baseline_output.to_vec();

        for (cell, state) in self.cells.iter_mut().zip(saved_states.iter()) {
            cell.set_state(state.clone());
        }

        let state_dim = dim.min(baseline.len());
        let output_dim = baseline.len().min(32);
        let perturb_dim = state_dim.min(32);

        let mut jacobian = vec![vec![0.0f64; perturb_dim]; output_dim];

        for j in 0..perturb_dim {
            let mut perturbed_state = saved_states[0].clone();
            perturbed_state[j] += epsilon;

            self.cells[0].set_state(perturbed_state);
            for (idx, state) in saved_states.iter().enumerate().skip(1) {
                self.cells[idx].set_state(state.clone());
            }

            let perturbed_output = self.forward(input, dt);

            for i in 0..output_dim {
                jacobian[i][j] = (perturbed_output[i] - baseline[i]) as f64 / epsilon as f64;
            }

            for (cell, state) in self.cells.iter_mut().zip(saved_states.iter()) {
                cell.set_state(state.clone());
            }
        }

        let n = output_dim.min(perturb_dim);
        let mut max_real = f64::NEG_INFINITY;
        let mut min_abs = f64::INFINITY;
        let mut max_abs = 0.0f64;

        for i in 0..n {
            let diag = if i < jacobian.len() && i < jacobian[i].len() {
                jacobian[i][i]
            } else {
                0.0
            };

            let radius: f64 = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    if i < jacobian.len() && j < jacobian[i].len() {
                        jacobian[i][j].abs()
                    } else {
                        0.0
                    }
                })
                .sum();

            max_real = max_real.max(diag);
            let abs_val = diag.abs();
            if abs_val > 1e-12 {
                min_abs = min_abs.min(abs_val);
            }
            max_abs = max_abs.max(abs_val + radius);
        }

        if min_abs == f64::INFINITY {
            min_abs = 1e-12;
        }

        let condition_number = max_abs / min_abs.max(1e-12);
        let collapsed = max_real < -0.01 && max_abs < 1.0;

        DynamicsDiagnostic {
            max_eigenvalue_real: max_real,
            condition_number,
            collapsed,
            state_norm: saved_states
                .iter()
                .map(|s| s.iter().map(|x| x * x).sum::<f32>().sqrt())
                .sum::<f32>()
                / saved_states.len() as f32,
        }
    }
}

impl symthaea_memory::TrainableNetwork for CfCNetwork {
    fn train_step(
        &mut self,
        input: &ndarray::Array1<f32>,
        target: &ndarray::Array1<f32>,
        dt: f32,
        learning_rate: f32,
    ) -> anyhow::Result<f32> {
        CfCNetwork::train_step(self, input, target, dt, learning_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    /// Helper: build a small network config for fast tests.
    fn small_config() -> CfCNetworkConfig {
        CfCNetworkConfig {
            input_dim: 4,
            hidden_dim: 8,
            num_layers: 2,
            output_dim: 3,
            ..Default::default()
        }
    }

    /// Helper: build a small input vector.
    fn small_input() -> Array1<f32> {
        Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4])
    }

    // ---------------------------------------------------------------
    // 1. Default config produces valid values
    // ---------------------------------------------------------------
    #[test]
    fn test_default_config_is_valid() {
        let config = CfCNetworkConfig::default();
        assert_eq!(config.input_dim, 64);
        assert_eq!(config.hidden_dim, 128);
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.output_dim, 32);
        assert!(config.residual);
        assert!(!config.bidirectional);
        assert!(!config.enable_online_learning);
    }

    // ---------------------------------------------------------------
    // 2. new(config) creates a network successfully
    // ---------------------------------------------------------------
    #[test]
    fn test_new_creates_network() {
        let config = small_config();
        let net = CfCNetwork::new(config);
        assert_eq!(net.cells.len(), 2);
        assert_eq!(net.config.input_dim, 4);
        assert_eq!(net.config.output_dim, 3);
    }

    // ---------------------------------------------------------------
    // 3. new_with_input creates a network with correct dims
    // ---------------------------------------------------------------
    #[test]
    fn test_new_with_input() {
        let net = CfCNetwork::new_with_input(4, 8);
        assert_eq!(net.config.input_dim, 4);
        assert_eq!(net.config.hidden_dim, 8);
        assert_eq!(net.cells.len(), 2); // default num_layers
    }

    // ---------------------------------------------------------------
    // 4. forward() produces output of correct dimension
    // ---------------------------------------------------------------
    #[test]
    fn test_forward_output_dimension() {
        let mut net = CfCNetwork::new(small_config());
        let output = net.forward(&small_input(), 0.1);
        assert_eq!(output.len(), 3);
        for &v in output.iter() {
            assert!(v.is_finite(), "output contains non-finite value: {}", v);
        }
    }

    // ---------------------------------------------------------------
    // 5. forward_sequence() produces correct number of outputs
    // ---------------------------------------------------------------
    #[test]
    fn test_forward_sequence_length() {
        let mut net = CfCNetwork::new(small_config());
        let inputs: Vec<Array1<f32>> = (0..5).map(|_| small_input()).collect();
        let dts = vec![0.1f32; 5];
        let outputs = net.forward_sequence(&inputs, &dts);
        assert_eq!(outputs.len(), 5);
        for out in &outputs {
            assert_eq!(out.len(), 3);
        }
    }

    // ---------------------------------------------------------------
    // 6. reset() completes without panic and zeroes step counter
    // ---------------------------------------------------------------
    #[test]
    fn test_reset_does_not_panic() {
        let mut net = CfCNetwork::new(small_config());
        // Run a few forward passes to accumulate state
        for _ in 0..3 {
            net.forward(&small_input(), 0.1);
        }
        assert!(net.total_steps > 0);
        net.reset();
        assert_eq!(net.total_steps, 0);
    }

    // ---------------------------------------------------------------
    // 7. num_parameters() returns > 0
    // ---------------------------------------------------------------
    #[test]
    fn test_num_parameters_positive() {
        let net = CfCNetwork::new(small_config());
        let n = net.num_parameters();
        assert!(n > 0, "num_parameters should be > 0, got {}", n);

        // Sanity-check: for 2 layers with hidden_dim=8, output_dim=3
        // Layer 0: 4*8 + 8*8 + 8 + 8 = 112
        // Layer 1: 8*8 + 8*8 + 8 + 8 = 144
        // Output: 8*3 + 3 = 27
        // Total: 283
        assert_eq!(n, 283);
    }

    // ---------------------------------------------------------------
    // 8. state() returns correct number of layer states
    // ---------------------------------------------------------------
    #[test]
    fn test_state_returns_correct_count() {
        let net = CfCNetwork::new(small_config());
        let states = net.state();
        assert_eq!(states.len(), 2, "should have one state per layer");
        for s in &states {
            assert_eq!(s.len(), 8, "each state should match hidden_dim");
        }
    }

    // ---------------------------------------------------------------
    // 9. train_step_bptt() returns a finite loss
    // ---------------------------------------------------------------
    #[test]
    fn test_train_step_bptt_finite_loss() {
        let mut net = CfCNetwork::new(small_config());
        let input = small_input();
        let target = Array1::from_vec(vec![0.5, -0.3, 0.1]);
        let dts = vec![0.1f32];
        let loss = net
            .train_step_bptt(&[input], &[target], &dts, 0.001)
            .unwrap();
        assert!(loss.is_finite(), "loss should be finite, got {}", loss);
        assert!(loss >= 0.0, "MSE loss should be non-negative, got {}", loss);
    }

    // ---------------------------------------------------------------
    // 10. adapt_online() works when online learning is enabled
    // ---------------------------------------------------------------
    #[test]
    fn test_adapt_online_enabled() {
        let config = small_config().with_online_learning_config(OnlineLearningConfig {
            learning_rate: 0.01,
            error_threshold: 0.0, // always adapt
            ema_alpha: 0.1,
            max_weight_delta: 0.1,
            adapt_tau: false,
            tau_lr_multiplier: 0.01,
        });
        let mut net = CfCNetwork::new(config);
        assert!(net.online_learning_enabled());

        let input = small_input();
        let target = Array1::from_vec(vec![1.0, -1.0, 0.5]);

        // A large prediction error should trigger adaptation
        let adapted = net.adapt_online(10.0, &input, &target, 0.1);
        assert!(
            adapted,
            "adapt_online should return true when error exceeds threshold"
        );

        let stats = net.online_stats();
        assert_eq!(stats.total_adaptation_calls, 1);
        assert_eq!(stats.adaptations_applied, 1);
        assert!(
            stats.cumulative_weight_change > 0.0,
            "should have accumulated some weight change"
        );
    }

    // ---------------------------------------------------------------
    // 11. adapt_online() returns false when disabled
    // ---------------------------------------------------------------
    #[test]
    fn test_adapt_online_disabled() {
        let mut net = CfCNetwork::new(small_config());
        assert!(!net.online_learning_enabled());

        let input = small_input();
        let target = Array1::from_vec(vec![1.0, -1.0, 0.5]);
        let adapted = net.adapt_online(10.0, &input, &target, 0.1);
        assert!(!adapted, "adapt_online should return false when disabled");
    }

    // ---------------------------------------------------------------
    // 12. state_diversity() returns a finite value
    // ---------------------------------------------------------------
    #[test]
    fn test_state_diversity_finite() {
        let mut net = CfCNetwork::new(small_config());
        // Run forward to populate cell states
        net.forward(&small_input(), 0.1);
        let div = net.state_diversity();
        assert!(
            div.is_finite(),
            "state_diversity should be finite, got {}",
            div
        );
        assert!(
            (0.0..=1.0).contains(&div),
            "diversity is sigmoid-scaled [0,1], got {}",
            div
        );
    }

    // ---------------------------------------------------------------
    // 13. diagnose_dynamics() returns valid diagnostic
    // ---------------------------------------------------------------
    #[test]
    fn test_diagnose_dynamics_valid() {
        let mut net = CfCNetwork::new(small_config());
        let input = small_input();
        // Warm up the network
        net.forward(&input, 0.1);
        let diag = net.diagnose_dynamics(&input, 0.1);
        assert!(
            diag.max_eigenvalue_real.is_finite(),
            "max_eigenvalue_real should be finite"
        );
        assert!(
            diag.condition_number.is_finite(),
            "condition_number should be finite"
        );
        assert!(diag.state_norm.is_finite(), "state_norm should be finite");
    }

    // ---------------------------------------------------------------
    // 14. get_weights() and set_weights() round-trip
    // ---------------------------------------------------------------
    #[test]
    fn test_weights_round_trip() {
        let mut net = CfCNetwork::new(small_config());
        let weights_orig = net.get_weights();
        assert!(!weights_orig.is_empty(), "weights should not be empty");

        // Mutate the network via forward + training
        let input = small_input();
        let target = Array1::from_vec(vec![0.5, -0.3, 0.1]);
        net.train_step_bptt(&[input.clone()], &[target], &[0.1], 0.01)
            .unwrap();
        let weights_after_train = net.get_weights();

        // Weights should have changed
        let any_changed = weights_orig
            .iter()
            .zip(weights_after_train.iter())
            .any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(
            any_changed,
            "training should have modified at least some weights"
        );

        // Restore original weights
        net.set_weights(&weights_orig);
        let weights_restored = net.get_weights();

        // They should match exactly
        assert_eq!(
            weights_orig.len(),
            weights_restored.len(),
            "weight vector lengths must match"
        );
        for (i, (a, b)) in weights_orig.iter().zip(weights_restored.iter()).enumerate() {
            assert!(
                (a - b).abs() < f32::EPSILON,
                "weight mismatch at index {}: {} vs {}",
                i,
                a,
                b
            );
        }
    }
}
