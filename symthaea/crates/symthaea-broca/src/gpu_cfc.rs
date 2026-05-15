// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! GPU-accelerated CfC (Closed-form Continuous-time) network via candle tensors.
//!
//! Packs neuron states and weights into batched [N, D] tensors on GPU (CUDA),
//! replacing the per-neuron CPU loops with batched CUDA kernel launches.
//!
//! All element-wise HDC operations (bind, bundle, normalize) map directly to
//! candle tensor ops. The sequential layer dependency is preserved, but
//! within-layer neuron parallelism is fully exploited by GPU SIMD.
//!
//! # Architecture
//!
//! ```text
//! GpuCfcNetwork
//!   └─ layers: Vec<GpuCfcLayer>      // 3 layers (typical)
//!       └─ states: Tensor [N, D]      // N=8 neurons, D=16384
//!       └─ weight_hv: Tensor [N, D]   // trainable
//!       └─ input_mask: Tensor [N, D]  // trainable
//!       └─ ... (per-neuron weights packed into tensors)
//! ```

use candle_core::{Device, Result, Tensor};

/// Expand a [N] scalar-per-neuron tensor to [N, D] by broadcasting.
/// candle 0.8 doesn't auto-broadcast [N, 1] * [N, D], so we expand explicitly.
fn expand_nd(scalars: &Tensor, n: usize, d: usize) -> Result<Tensor> {
    scalars.unsqueeze(1)?.broadcast_as((n, d))
}

/// GPU-accelerated layer of CfC neurons.
///
/// All N neurons in the layer share the same [N, D] tensor layout,
/// enabling batched element-wise ops via a single CUDA kernel launch.
pub struct GpuCfcLayer {
    /// Current neuron states [N, D]
    pub states: Tensor,
    /// Weight hypervectors [N, D] — trainable
    pub weight_hv: Tensor,
    /// Input mask hypervectors [N, D] — trainable
    pub input_mask: Tensor,
    /// Tau modulator hypervectors [N, D] — trainable (scalar gradient projected)
    pub tau_modulator: Tensor,
    /// Gate weight hypervectors [N, D] — fixed during BPTT
    pub gate_weight: Tensor,
    /// Gate bias hypervectors [N, D] — fixed during BPTT
    pub gate_bias: Tensor,
    /// Weight momentum [N, D] — optimizer state
    pub weight_momentum: Tensor,
    /// Input momentum [N, D] — optimizer state
    pub input_momentum: Tensor,
    /// Number of neurons
    pub n_neurons: usize,
    /// Dimension (16384)
    pub dim: usize,
    /// Tau base (scalar, shared across neurons in layer)
    pub tau_base: f32,
    /// Backbone tau (scalar)
    pub backbone_tau: f32,
    /// Gating steepness (scalar)
    pub gating_steepness: f32,
    /// Momentum coefficient
    pub momentum: f32,
}

impl GpuCfcLayer {
    /// Create from a CPU layer of neurons, packing into GPU tensors.
    pub fn from_cpu_neurons(
        neurons: &[symthaea_core::hdc::hdc_ltc_unified::HdcLtcUnifiedNeuron],
        device: &Device,
    ) -> Result<Self> {
        let n = neurons.len();
        if n == 0 {
            return Err(candle_core::Error::Msg("empty layer".into()));
        }
        let dim = neurons[0].state().values.len();
        let config = neurons[0].config();

        // Pack each neuron's HVs into rows of [N, D] tensors
        let pack = |extract: &dyn Fn(
            &symthaea_core::hdc::hdc_ltc_unified::HdcLtcUnifiedNeuron,
        ) -> &[f32]|
         -> Result<Tensor> {
            let mut flat = Vec::with_capacity(n * dim);
            for neuron in neurons {
                flat.extend_from_slice(extract(neuron));
            }
            Tensor::from_vec(flat, (n, dim), device)
        };

        Ok(Self {
            states: pack(&|n| n.state().as_slice())?,
            weight_hv: pack(&|n| n.weight_hv_ref().as_slice())?,
            input_mask: pack(&|n| n.input_mask_ref().as_slice())?,
            tau_modulator: pack(&|n| n.tau_modulator_ref().as_slice())?,
            gate_weight: pack(&|n| n.gate_weight_ref().as_slice())?,
            gate_bias: pack(&|n| n.gate_bias_ref().as_slice())?,
            weight_momentum: pack(&|n| n.weight_momentum_ref().as_slice())?,
            input_momentum: pack(&|n| n.input_momentum_ref().as_slice())?,
            n_neurons: n,
            dim,
            tau_base: config.tau_base,
            backbone_tau: config.backbone_tau,
            gating_steepness: config.gating_steepness,
            momentum: config.momentum,
        })
    }

    /// Forward pass: batched closed-form evolution for all neurons in this layer.
    ///
    /// `input` is [1, D] (broadcast to all neurons).
    /// Mutates `self.states` in-place.
    pub fn evolve_closed_form(&mut self, dt: f32, input: &Tensor) -> Result<()> {
        // Broadcast input [1, D] → [N, D] for element-wise ops with neuron tensors
        let input_bc = input.broadcast_as(self.states.shape())?;

        // ── 1. Equilibrium: x_inf = tanh((W⊗state + U⊗input) * 0.5) ────
        // bind = element-wise multiply
        let wx = (&self.weight_hv * &self.states)?; // [N, D]
        let uu = (&self.input_mask * &input_bc)?; // [N, D]
        let z = ((&wx + &uu)? * 0.5)?; // [N, D]
        let x_inf = z.tanh()?; // [N, D]

        // ── 2. Tau: per-neuron adaptive time constant ───────────────────
        // state_norms[i] = ||states[i]||
        let state_norms = self.states.sqr()?.sum(1)?.sqrt()?; // [N]

        // input_adjustment[i] = cosine_sim(input, tau_modulator[i])
        let input_dots = (&input_bc * &self.tau_modulator)?.sum(1)?; // [N]
        let input_norm = input_bc.clone().sqr()?.sum(1)?.sqrt()?; // [N]
        let tau_mod_norms = self.tau_modulator.sqr()?.sum(1)?.sqrt()?; // [N]
        let denom = (&input_norm * &tau_mod_norms)?; // [N]
        let input_adj = (&input_dots / &denom.clamp(1e-10, f32::MAX)?)?; // [N]

        // tau = tau_base * (1 + backbone_tau * ||state||) * (1 + 0.2 * input_adj)
        let bt_state = (&state_norms * self.backbone_tau as f64)?;
        let factor1 = (bt_state + 1.0)?;
        let adj_scaled = (&input_adj * 0.2)?;
        let factor2 = (adj_scaled + 1.0)?;
        let tau = ((&factor1 * &factor2)? * self.tau_base as f64)?.clamp(0.01, 10.0)?; // [N]

        // ── 3. Gating: sigma per neuron ─────────────────────────────────
        // bundle_si = (state + input) * 0.5
        let bundle_si = ((&self.states + &input_bc)? * 0.5)?; // [N, D]
                                                              // gate_activation = cosine_sim(bundle_si, gate_weight) + mean(gate_bias)
        let dot_sg = (&bundle_si * &self.gate_weight)?.sum(1)?; // [N]
        let bundle_norms = bundle_si.sqr()?.sum(1)?.sqrt()?; // [N]
        let gw_norms = self.gate_weight.sqr()?.sum(1)?.sqrt()?; // [N]
        let sim_denom = (&bundle_norms * &gw_norms)?.clamp(1e-10, f32::MAX)?; // [N]
        let sim = (&dot_sg / &sim_denom)?; // [N]
        let bias_mean = self.gate_bias.mean(1)?; // [N]
        let gate_activation = (&sim + &bias_mean)?; // [N]

        // sigma_base = sigmoid(gate_activation * gating_steepness)
        let neg_ga = (&gate_activation * (-self.gating_steepness as f64))?;
        let exp_neg_ga = neg_ga.exp()?;
        let one_plus_exp = (exp_neg_ga + 1.0)?;
        let sigma_base = one_plus_exp.recip()?; // sigmoid [N]

        // decay = exp(-dt / tau)
        let tau_recip = tau.recip()?;
        let neg_dt_over_tau = (&tau_recip * (-dt as f64))?;
        let decay = neg_dt_over_tau.exp()?; // [N]

        // sigma = 1 - decay * (1 - sigma_base)
        let one_minus_sb = (1.0 - &sigma_base)?;
        let sigma = (1.0 - (&decay * &one_minus_sb)?)?.clamp(0.0, 1.0)?; // [N]

        // ── 4. Interpolation: state = (1-sigma)*state + sigma*x_inf ─────
        // Reshape sigma to [N, 1] for broadcast with [N, D]
        let sigma_nd = expand_nd(&sigma, self.n_neurons, self.dim)?; // [N, D]
        let one_minus_sigma_nd = (1.0 - &sigma_nd)?;
        self.states = ((&one_minus_sigma_nd * &self.states)? + (&sigma_nd * &x_inf)?)?;

        // ── 5. Norm clip to 5.0 ────────────────────────────────────────
        let norms = self.states.sqr()?.sum(1)?.sqrt()?; // [N]
        let clamped_norms = norms.clamp(5.0, f32::MAX)?;
        let scale = (clamped_norms.recip()? * 5.0)?.clamp(0.0, 1.0)?; // [N]
        let scale_nd = expand_nd(&scale, self.n_neurons, self.dim)?;
        self.states = (&self.states * &scale_nd)?;

        Ok(())
    }

    /// Get layer output: mean of all neuron states → [1, D]
    pub fn output(&self) -> Result<Tensor> {
        self.states.mean(0) // [D] — mean across neurons
    }

    /// Backward pass: compute gradients for all neurons in the layer.
    ///
    /// Returns (dw, du, dtau_scalar, d_input) as tensors.
    /// - dw: [N, D] — gradient w.r.t. weight_hv
    /// - du: [N, D] — gradient w.r.t. input_mask
    /// - dtau: [N] — scalar tau gradient per neuron
    /// - d_input: [N, D] — gradient w.r.t. input (for inter-layer backprop)
    pub fn backward(&self, input: &Tensor, target: &Tensor, dt: f32) -> Result<GpuCfcGradients> {
        // Broadcast input [1, D] → [N, D]
        let input_bc = input.broadcast_as(self.states.shape())?;
        let dim_f = self.dim as f64;

        // ── Forward recomputation ───────────────────────────────────────
        let wx = (&self.weight_hv * &self.states)?;
        let uu = (&self.input_mask * &input_bc)?;
        let z = ((&wx + &uu)? * 0.5)?;
        let x_inf = z.tanh()?;

        // Tau + decay (simplified: using just exp(-dt/tau) without full gating)
        let state_norms = self.states.sqr()?.sum(1)?.sqrt()?;
        let input_dots = (&input_bc * &self.tau_modulator)?.sum(1)?;
        let input_norm = input_bc.clone().sqr()?.sum(1)?.sqrt()?;
        let tau_mod_norms = self.tau_modulator.sqr()?.sum(1)?.sqrt()?;
        let denom = (&input_norm * &tau_mod_norms)?.clamp(1e-10, f32::MAX)?;
        let input_adj = (&input_dots / &denom)?;
        let bt_state = (&state_norms * self.backbone_tau as f64)?;
        let factor1 = (bt_state + 1.0)?;
        let adj_scaled = (&input_adj * 0.2)?;
        let factor2 = (adj_scaled + 1.0)?;
        let tau = ((&factor1 * &factor2)? * self.tau_base as f64)?.clamp(0.01, 10.0)?;

        let tau_recip = tau.recip()?;
        let neg_dt_over_tau = (&tau_recip * (-dt as f64))?;
        let decay = neg_dt_over_tau.exp()?;
        let sigma = ((&decay * -1.0)? + 1.0)?; // 1 - decay [N]

        // new_state = sigma*x_inf + (1-sigma)*state
        let sigma_nd = expand_nd(&sigma, self.n_neurons, self.dim)?;
        let one_minus_sigma_nd = (1.0 - &sigma_nd)?;
        let new_state = ((&sigma_nd * &x_inf)? + (&one_minus_sigma_nd * &self.states)?)?;

        // ── Backward ────────────────────────────────────────────────────
        // dh = 2 * (new_state - target) / D
        let dh = ((&new_state - target)? * (2.0 / dim_f))?; // [N, D]

        // dx_inf = dh * sigma
        let dx_inf = (&dh * &sigma_nd)?; // [N, D]

        // dz = dx_inf * tanh'(z) = dx_inf * (1 - tanh(z)^2)
        let tanh_z_sq = x_inf.sqr()?; // tanh(z)^2
        let dtanh = (1.0 - &tanh_z_sq)?;
        let dz = (&dx_inf * &dtanh)?; // [N, D]

        // dw = dz * state (bind chain rule)
        let dw = (&dz * &self.states)?; // [N, D]

        // du = dz * input (bind chain rule)
        let du = (&dz * &input_bc)?; // [N, D]

        // dtau_scalar = sum(dh * (x_inf - state)) * (-dt/tau^2) * exp(-dt/tau)
        let diff = (&x_inf - &self.states)?;
        let dh_diff = (&dh * &diff)?.sum(1)?; // [N]
        let tau_sq = tau.sqr()?;
        let decay_neg_dt = (&decay * (-dt as f64))?;
        let dtau_factor = (&decay_neg_dt / &tau_sq)?;
        let dtau = (&dh_diff * &dtau_factor)?; // [N]

        // d_input = dz * input_mask (for inter-layer backprop)
        let d_input = (&dz * &self.input_mask)?; // [N, D]

        Ok(GpuCfcGradients {
            dw,
            du,
            dtau,
            d_input,
        })
    }

    /// Apply gradients with momentum (no weight decay for BPTT).
    pub fn apply_gradients(&mut self, grads: &GpuCfcGradients, lr: f32) -> Result<()> {
        let m = self.momentum as f64;
        let neg_lr = -(lr as f64);

        // Weight momentum update: mom = m*mom + (-lr)*dw
        self.weight_momentum = ((&self.weight_momentum * m)? + (&grads.dw * neg_lr)?)?;
        self.weight_hv = (&self.weight_hv + &self.weight_momentum)?;

        // Input momentum update
        self.input_momentum = ((&self.input_momentum * m)? + (&grads.du * neg_lr)?)?;
        self.input_mask = (&self.input_mask + &self.input_momentum)?;

        // Tau modulator update (project scalar gradient)
        let tau_norm = self.tau_modulator.sqr()?.sum(1)?.sqrt()?; // [N]
        let tau_norm_clamped = tau_norm.clamp(1e-10, f32::MAX)?;
        let tau_norm_bc = expand_nd(&tau_norm_clamped, self.n_neurons, self.dim)?;
        let tau_normalized = (&self.tau_modulator / &tau_norm_bc)?;
        let dtau_bc = expand_nd(&grads.dtau, self.n_neurons, self.dim)?; // [N, D]
        let dtau_scaled = (&dtau_bc * neg_lr)?;
        let tau_delta = (&tau_normalized * &dtau_scaled)?;
        self.tau_modulator = (&self.tau_modulator + &tau_delta)?;

        // Norm clip weights to 2.0
        let w_norms = self.weight_hv.sqr()?.sum(1)?.sqrt()?; // [N]
        let w_clamped = w_norms.clamp(2.0, f32::MAX)?;
        let w_scale = (w_clamped.recip()? * 2.0)?.clamp(0.0, 1.0)?;
        let w_scale_bc = expand_nd(&w_scale, self.n_neurons, self.dim)?;
        self.weight_hv = (&self.weight_hv * &w_scale_bc)?;

        let u_norms = self.input_mask.sqr()?.sum(1)?.sqrt()?;
        let u_clamped = u_norms.clamp(2.0, f32::MAX)?;
        let u_scale = (u_clamped.recip()? * 2.0)?.clamp(0.0, 1.0)?;
        let u_scale_bc = expand_nd(&u_scale, self.n_neurons, self.dim)?;
        self.input_mask = (&self.input_mask * &u_scale_bc)?;

        Ok(())
    }

    /// Copy GPU state back to CPU neurons for serialization.
    pub fn sync_to_cpu(
        &self,
        neurons: &mut [symthaea_core::hdc::hdc_ltc_unified::HdcLtcUnifiedNeuron],
    ) -> Result<()> {
        let states: Vec<f32> = self
            .states
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let weights: Vec<f32> = self
            .weight_hv
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let masks: Vec<f32> = self
            .input_mask
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let tau_mods: Vec<f32> = self
            .tau_modulator
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let w_mom: Vec<f32> = self
            .weight_momentum
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let i_mom: Vec<f32> = self
            .input_momentum
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();

        for (i, neuron) in neurons.iter_mut().enumerate() {
            let start = i * self.dim;
            let end = start + self.dim;
            neuron
                .state_mut()
                .values
                .copy_from_slice(&states[start..end]);
            neuron
                .weight_hv_mut()
                .values
                .copy_from_slice(&weights[start..end]);
            neuron
                .input_mask_mut()
                .values
                .copy_from_slice(&masks[start..end]);
            neuron
                .tau_modulator_mut()
                .values
                .copy_from_slice(&tau_mods[start..end]);
            neuron
                .weight_momentum_mut()
                .values
                .copy_from_slice(&w_mom[start..end]);
            neuron
                .input_momentum_mut()
                .values
                .copy_from_slice(&i_mom[start..end]);
        }
        Ok(())
    }
}

/// Gradients from a GPU CfC layer backward pass.
pub struct GpuCfcGradients {
    /// Weight gradient [N, D]
    pub dw: Tensor,
    /// Input mask gradient [N, D]
    pub du: Tensor,
    /// Tau scalar gradient [N]
    pub dtau: Tensor,
    /// Input gradient for inter-layer backprop [N, D]
    pub d_input: Tensor,
}

/// GPU-accelerated CfC network: multiple layers of batched neurons.
pub struct GpuCfcNetwork {
    /// Layers of GPU-accelerated neurons
    pub layers: Vec<GpuCfcLayer>,
    /// Inter-layer binding vectors [D] each, on GPU
    pub layer_bindings: Vec<Tensor>,
    /// Cached layer outputs [D] each
    pub layer_outputs: Vec<Tensor>,
    /// Whether to use layer bindings
    pub use_layer_binding: bool,
    /// Whether to use skip connections
    pub skip_connections: bool,
    /// Device (CUDA or CPU)
    pub device: Device,
    /// Dimension
    pub dim: usize,
}

impl GpuCfcNetwork {
    /// Create from a CPU network, moving all weights to GPU.
    pub fn from_cpu_network(
        network: &symthaea_core::hdc::hdc_ltc_unified::HdcLtcUnifiedNetwork,
        device: &Device,
    ) -> Result<Self> {
        let config = network.config();
        let dim = config.neuron_config.dimension;

        let mut layers = Vec::new();
        for l in 0..network.n_layers() {
            if let Some(cpu_layer) = network.layer(l) {
                layers.push(GpuCfcLayer::from_cpu_neurons(cpu_layer, device)?);
            }
        }

        let mut layer_bindings = Vec::new();
        for l in 0..network.n_layers() {
            let binding = network.layer_binding(l);
            let t = Tensor::from_vec(binding.as_slice().to_vec(), (1, dim), device)?;
            layer_bindings.push(t);
        }

        let layer_outputs = (0..network.n_layers())
            .map(|_| Tensor::zeros((1, dim), candle_core::DType::F32, device))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            layers,
            layer_bindings,
            layer_outputs,
            use_layer_binding: config.use_layer_binding,
            skip_connections: config.skip_connections,
            device: device.clone(),
            dim,
        })
    }

    /// Forward pass through all layers.
    ///
    /// `input` is [1, D].
    pub fn evolve_closed_form(&mut self, dt: f32, input: &Tensor) -> Result<()> {
        // Layer 0: direct input
        self.layers[0].evolve_closed_form(dt, input)?;
        self.layer_outputs[0] = self.layers[0].output()?.unsqueeze(0)?; // [1, D]

        // Subsequent layers
        for l in 1..self.layers.len() {
            let layer_input = self.compute_layer_input(l, input)?;
            self.layers[l].evolve_closed_form(dt, &layer_input)?;
            self.layer_outputs[l] = self.layers[l].output()?.unsqueeze(0)?;
        }

        Ok(())
    }

    /// Compute input for layer l (l >= 1).
    fn compute_layer_input(&self, l: usize, original_input: &Tensor) -> Result<Tensor> {
        let prev_output = &self.layer_outputs[l - 1]; // [1, D]

        let bound = if self.use_layer_binding {
            (&self.layer_bindings[l] * prev_output)? // bind = element-wise mul
        } else {
            prev_output.clone()
        };

        if self.skip_connections {
            let sum = (&bound + original_input)?;
            Ok((sum * 0.5)?)
        } else {
            Ok(bound)
        }
    }

    /// Get network output: last layer output [1, D].
    pub fn output(&self) -> Result<Tensor> {
        self.layer_outputs
            .last()
            .cloned()
            .ok_or_else(|| candle_core::Error::Msg("no layers".into()))
    }

    /// Sync neuron states from CPU network to GPU (after CPU forward pass).
    /// Only syncs states, not weights (weights are updated on GPU and synced back).
    pub fn sync_states_from_cpu(
        &mut self,
        network: &symthaea_core::hdc::hdc_ltc_unified::HdcLtcUnifiedNetwork,
    ) -> Result<()> {
        for (l, gpu_layer) in self.layers.iter_mut().enumerate() {
            if let Some(cpu_layer) = network.layer(l) {
                let n = cpu_layer.len();
                let d = gpu_layer.dim;
                let mut flat = Vec::with_capacity(n * d);
                for neuron in cpu_layer {
                    flat.extend_from_slice(neuron.state().as_slice());
                }
                gpu_layer.states = Tensor::from_vec(flat, (n, d), &self.device)?;
            }
        }
        // Also sync layer outputs
        for (l, gpu_layer) in self.layers.iter().enumerate() {
            self.layer_outputs[l] = gpu_layer.output()?.unsqueeze(0)?;
        }
        Ok(())
    }

    /// Full BPTT: backprop d_output through normalize, then through each layer.
    ///
    /// This replaces `LanguageController::backward_step()` for GPU training.
    /// `d_output` is the gradient of CE loss w.r.t. the normalized output HV [1, D].
    /// `input` is the composed input HV (thought ⊗ token ⊗ pos) [1, D].
    /// `dt` is the time step. `lr` is the network learning rate.
    /// `gradient_attenuation` scales gradients for earlier layers (0.0-1.0).
    pub fn backward_and_update(
        &mut self,
        d_output: &Tensor,
        input: &Tensor,
        dt: f32,
        lr: f32,
        gradient_attenuation: f32,
    ) -> Result<()> {
        let last_idx = self.layers.len() - 1;

        // ── 1. Backprop through normalize ──────────────────────────────
        // output = raw / ||raw||
        // d_raw = (d_output - dot(d_output, hat) * hat) / ||raw||
        let raw_output = self.layer_outputs[last_idx].clone(); // [1, D]
        let raw_norm_sq = raw_output.sqr()?.sum(1)?; // [1]
        let raw_norm = raw_norm_sq.sqrt()?.clamp(1e-8, f32::MAX)?; // [1]
        let raw_norm_bc = raw_norm.unsqueeze(1)?.broadcast_as(raw_output.shape())?; // [1, D]
        let output_hat = (&raw_output / &raw_norm_bc)?; // [1, D]
        let dot_d_hat = (d_output * &output_hat)?.sum(1)?; // [1]
        let dot_d_hat_bc = dot_d_hat.unsqueeze(1)?.broadcast_as(raw_output.shape())?; // [1, D]
        let proj = (&output_hat * &dot_d_hat_bc)?; // [1, D]
        let d_raw = ((d_output - &proj)? / &raw_norm_bc)?; // [1, D]

        // ── 2. Scale by 1/N (output = mean of neuron states) ──────────
        let n_neurons = self.layers[last_idx].n_neurons as f64;
        let d_per_neuron = (&d_raw * (1.0 / n_neurons))?; // [1, D]

        // ── 3. Construct target: target = state - d_per_neuron ─────────
        // This is how the CPU backward works: MSE(new_state, target)
        // where target = state - gradient_direction
        let last_layer = &self.layers[last_idx];
        let d_per_neuron_bc = d_per_neuron.broadcast_as(last_layer.states.shape())?; // [N, D]
        let target = (&last_layer.states - &d_per_neuron_bc)?; // [N, D]

        // ── 4. Layer input for last layer ──────────────────────────────
        let layer_input = if last_idx == 0 {
            input.clone()
        } else {
            self.compute_layer_input(last_idx, input)?
        };

        // ── 5. Backward + gradient update on last layer ────────────────
        let grads = self.layers[last_idx].backward(&layer_input, &target, dt)?;
        self.layers[last_idx].apply_gradients(&grads, lr)?;

        // ── 6. Optionally backprop to earlier layers ───────────────────
        if last_idx > 0 && gradient_attenuation > 0.0 {
            // Average d_input across neurons
            let d_prev = grads.d_input.mean(0)?.unsqueeze(0)?; // [1, D]
            let d_prev_scaled = (&d_prev * gradient_attenuation as f64)?;

            let layer0_input = input.clone(); // layer 0 gets original input
            let n0 = self.layers[0].n_neurons as f64;
            let d_per_neuron_0 = (&d_prev_scaled * (1.0 / n0))?;
            let d_per_neuron_0_bc = d_per_neuron_0.broadcast_as(self.layers[0].states.shape())?;
            let target_0 = (&self.layers[0].states - &d_per_neuron_0_bc)?;
            let grads_0 = self.layers[0].backward(&layer0_input, &target_0, dt)?;
            self.layers[0].apply_gradients(&grads_0, lr * gradient_attenuation)?;
        }

        Ok(())
    }

    /// Copy all GPU state back to CPU network for serialization.
    pub fn sync_to_cpu(
        &self,
        network: &mut symthaea_core::hdc::hdc_ltc_unified::HdcLtcUnifiedNetwork,
    ) -> Result<()> {
        for (l, gpu_layer) in self.layers.iter().enumerate() {
            if let Some(cpu_layer) = network.layer_mut(l) {
                gpu_layer.sync_to_cpu(cpu_layer)?;
            }
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GpuTrainer: Full on-device training loop (forward + backward per token)
// ═══════════════════════════════════════════════════════════════════════════════

/// Detect best available device with automatic CPU fallback.
pub fn detect_device() -> Device {
    match Device::cuda_if_available(0) {
        Ok(dev) if dev.is_cuda() => {
            tracing::info!("GPU training: CUDA device detected");
            dev
        }
        Ok(_) => {
            tracing::info!("GPU training: no CUDA, using CPU tensors");
            Device::Cpu
        }
        Err(e) => {
            tracing::warn!("GPU training: CUDA detection failed ({e}), using CPU");
            Device::Cpu
        }
    }
}

/// Full GPU trainer: keeps all state on device, syncs once per training pair.
///
/// The entire BPTT window (forward + backward for each token position) runs
/// on GPU without CPU↔GPU sync. Only transfers:
/// - Thought HV: once per pair (CPU → GPU, 64KB)
/// - Logits: per token (GPU → CPU, 16KB for loss computation)
/// - Weights: once per pair (GPU → CPU for checkpointing)
pub struct GpuTrainer {
    /// CfC network on GPU
    pub network: GpuCfcNetwork,
    /// Token embeddings [vocab, D] on GPU
    pub embeddings: Tensor,
    /// Pre-computed position permutations [max_pos, D] on GPU
    pub position_cache: Tensor,
    /// Device
    pub device: Device,
    /// HDC dimension
    pub dim: usize,
    /// Logit scale
    pub logit_scale: f32,
    /// Direct thought-logit residual blend weight.
    pub thought_logit_residual_weight: f32,
    /// Gradient attenuation for earlier layers
    pub gradient_attenuation: f32,
    /// dt per token
    pub dt_per_token: f32,
}

impl GpuTrainer {
    /// Create from CPU controller, moving everything to GPU.
    pub fn from_controller(
        controller: &crate::controller::LanguageController,
        device: &Device,
        max_positions: usize,
    ) -> Result<Self> {
        let dim = symthaea_core::hdc::HDC_DIMENSION;

        // Transfer CfC network
        let network = GpuCfcNetwork::from_cpu_network(controller.network(), device)?;

        // Transfer token embeddings [vocab, D] — pre-normalized rows
        let embs = controller.token_embeddings();
        let vocab = embs.len();
        let mut flat_embs = Vec::with_capacity(vocab * dim);
        for emb in embs {
            // Normalize each embedding row (matches GpuEmbeddingCache behavior)
            let norm: f32 = emb
                .as_slice()
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
                .max(1e-10);
            for &v in emb.as_slice() {
                flat_embs.push(v / norm);
            }
        }
        let embeddings = Tensor::from_vec(flat_embs, (vocab, dim), device)?;

        // Pre-compute position permutations (circular shifts)
        let pos_base = controller.position_base_ref();
        let base_vals = pos_base.as_slice();
        let mut pos_cache = Vec::with_capacity(max_positions * dim);
        for pos in 0..max_positions {
            let shift = pos % dim;
            for i in 0..dim {
                pos_cache.push(base_vals[(i + dim - shift) % dim]);
            }
        }
        let position_cache = Tensor::from_vec(pos_cache, (max_positions, dim), device)?;

        tracing::info!(
            "GpuTrainer created: {} layers, {} embeddings, {} positions on {:?}",
            network.layers.len(),
            vocab,
            max_positions,
            device,
        );

        Ok(Self {
            network,
            embeddings,
            position_cache,
            device: device.clone(),
            dim,
            logit_scale: controller.config().logit_scale,
            thought_logit_residual_weight: controller.config().thought_logit_residual_weight,
            gradient_attenuation: controller.config().gradient_attenuation,
            dt_per_token: controller.config().dt_per_token,
        })
    }

    fn renormalize_embeddings(&mut self) -> Result<()> {
        let norms = self
            .embeddings
            .sqr()?
            .sum(1)?
            .sqrt()?
            .clamp(1e-8, f32::MAX)?;
        let norm_bc = norms.unsqueeze(1)?.broadcast_as(self.embeddings.shape())?;
        self.embeddings = (&self.embeddings / &norm_bc)?;
        Ok(())
    }

    /// Run one token step: forward + compute logits. Returns logits on CPU.
    pub fn forward_step(
        &mut self,
        thought_gpu: &Tensor,
        prev_token: u32,
        pos: usize,
    ) -> Result<Vec<f32>> {
        // 1. Compose input: thought ⊗ token_emb ⊗ pos_emb (element-wise mul = HDC bind)
        let token_emb = self.embeddings.get(prev_token as usize)?.unsqueeze(0)?;
        let max_pos = self.position_cache.dim(0)?;
        let pos_emb = self
            .position_cache
            .get(pos.min(max_pos - 1))?
            .unsqueeze(0)?;
        let input = (thought_gpu * &token_emb)?.mul(&pos_emb)?; // [1, D]

        // 2. CfC forward (all on GPU)
        self.network.evolve_closed_form(self.dt_per_token, &input)?;

        // 3. Normalized output
        let raw_output = self.network.output()?; // [1, D]
        let raw_norm = raw_output.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, f32::MAX)?;
        let norm_bc = raw_norm.unsqueeze(1)?.broadcast_as(raw_output.shape())?;
        let output_hat = (&raw_output / &norm_bc)?;

        // 4. Logits: output @ embeddings^T * scale
        let logits_tensor = (output_hat.matmul(&self.embeddings.t()?)? * self.logit_scale as f64)?;
        let logits_tensor = if self.thought_logit_residual_weight > 1e-6 {
            let thought_query = (thought_gpu * &pos_emb)?;
            let thought_norm = thought_query.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, f32::MAX)?;
            let thought_norm_bc = thought_norm
                .unsqueeze(1)?
                .broadcast_as(thought_query.shape())?;
            let thought_hat = (&thought_query / &thought_norm_bc)?;
            let thought_logits =
                (thought_hat.matmul(&self.embeddings.t()?)? * self.logit_scale as f64)?;
            let w = self.thought_logit_residual_weight.clamp(0.0, 1.0) as f64;
            ((&logits_tensor * (1.0 - w))? + (&thought_logits * w)?)?
        } else {
            logits_tensor
        };

        // 5. Transfer logits to CPU (only 16KB for 4096-vocab)
        logits_tensor.squeeze(0)?.to_vec1()
    }

    /// Return the current normalized CfC output as an HDC vector on CPU.
    pub fn current_output_hv(&self) -> Result<symthaea_core::hdc::ContinuousHV> {
        let raw_output = self.network.output()?;
        let raw_norm = raw_output.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, f32::MAX)?;
        let norm_bc = raw_norm.unsqueeze(1)?.broadcast_as(raw_output.shape())?;
        let output_hat = (&raw_output / &norm_bc)?;
        let values: Vec<f32> = output_hat.squeeze(0)?.to_vec1()?;
        Ok(symthaea_core::hdc::ContinuousHV::from_slice(&values))
    }

    /// Compute direct positioned thought logits without evolving the recurrent network.
    ///
    /// Mirrors the auxiliary binding query:
    /// `normalize(thought_hv ⊗ position[pos]) @ normalized_embeddings^T * scale`.
    pub fn thought_position_logits(&self, thought_gpu: &Tensor, pos: usize) -> Result<Vec<f32>> {
        let max_pos = self.position_cache.dim(0)?;
        let pos_emb = self
            .position_cache
            .get(pos.min(max_pos - 1))?
            .unsqueeze(0)?;
        let thought_query = (thought_gpu * &pos_emb)?;
        let thought_norm = thought_query.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, f32::MAX)?;
        let norm_bc = thought_norm
            .unsqueeze(1)?
            .broadcast_as(thought_query.shape())?;
        let thought_hat = (&thought_query / &norm_bc)?;
        let logits_tensor = (thought_hat.matmul(&self.embeddings.t()?)? * self.logit_scale as f64)?;
        logits_tensor.squeeze(0)?.to_vec1()
    }

    /// Backward step: CE gradient → BPTT on GPU.
    pub fn backward_step(
        &mut self,
        logits: &[f32],
        target: usize,
        thought_gpu: &Tensor,
        prev_token: u32,
        pos: usize,
        lr: f32,
    ) -> Result<()> {
        let n = logits.len();
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        if sum_exp < 1e-10 {
            return Ok(());
        }

        let scale = self.logit_scale;

        // error vector
        let errors: Vec<f32> = (0..n)
            .map(|i| {
                let prob = exps[i] / sum_exp;
                scale * (if i == target { prob - 1.0 } else { prob })
            })
            .collect();

        // d_output = error @ E_hat - cos_sum × output_hat (all on GPU)
        let error_t = Tensor::from_vec(errors.clone(), (1, n), &self.device)?;
        let grad_emb = error_t.matmul(&self.embeddings)?; // [1, D]

        let raw_output = self.network.output()?;
        let raw_norm = raw_output.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, f32::MAX)?;
        let norm_bc = raw_norm.unsqueeze(1)?.broadcast_as(raw_output.shape())?;
        let output_hat = (&raw_output / &norm_bc)?;

        let cos_sum: f32 = (0..n)
            .map(|i| {
                let cos_i = if scale.abs() > 1e-6 {
                    logits[i] / scale
                } else {
                    0.0
                };
                errors[i] * cos_i
            })
            .sum();

        let d_output = if cos_sum.abs() > 1e-10 {
            let cs = Tensor::from_vec(vec![cos_sum], (1, 1), &self.device)?;
            let cs_bc = cs.broadcast_as(output_hat.shape())?;
            (&grad_emb - (&output_hat * &cs_bc)?)?
        } else {
            grad_emb
        };

        // Reconstruct input on GPU (same composition as forward_step)
        let token_emb = self.embeddings.get(prev_token as usize)?.unsqueeze(0)?;
        let max_pos = self.position_cache.dim(0)?;
        let pos_emb = self
            .position_cache
            .get(pos.min(max_pos - 1))?
            .unsqueeze(0)?;
        let input = (thought_gpu * &token_emb)?.mul(&pos_emb)?;

        // Full BPTT on GPU (no CPU sync)
        self.network.backward_and_update(
            &d_output,
            &input,
            self.dt_per_token,
            lr,
            self.gradient_attenuation,
        )
    }

    /// Reset CfC states (between training pairs).
    pub fn reset_states(&mut self) -> Result<()> {
        for layer in &mut self.network.layers {
            layer.states =
                Tensor::zeros(layer.states.shape(), candle_core::DType::F32, &self.device)?;
        }
        for out in &mut self.network.layer_outputs {
            *out = Tensor::zeros(out.shape(), candle_core::DType::F32, &self.device)?;
        }
        Ok(())
    }

    /// Seed CfC from thought (first neuron of first layer).
    pub fn seed_from_thought(&mut self, thought_gpu: &Tensor) -> Result<()> {
        if let Some(layer) = self.network.layers.first_mut() {
            let n = layer.n_neurons;
            let d = layer.dim;
            let mut data = vec![0.0f32; n * d];
            let t: Vec<f32> = thought_gpu.squeeze(0)?.to_vec1()?;
            data[..d.min(t.len())].copy_from_slice(&t[..d.min(t.len())]);
            layer.states = Tensor::from_vec(data, (n, d), &self.device)?;
        }
        Ok(())
    }

    /// Sync weights back to CPU for checkpointing.
    pub fn sync_to_cpu(&self, ctrl: &mut crate::controller::LanguageController) -> Result<()> {
        self.network.sync_to_cpu(ctrl.network_mut())
    }

    /// Refresh GPU embeddings from CPU (after CPU embedding updates).
    pub fn refresh_embeddings(
        &mut self,
        ctrl: &crate::controller::LanguageController,
    ) -> Result<()> {
        let embs = ctrl.token_embeddings();
        let vocab = embs.len();
        let mut flat = Vec::with_capacity(vocab * self.dim);
        for emb in embs {
            let norm: f32 = emb
                .as_slice()
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
                .max(1e-10);
            for &v in emb.as_slice() {
                flat.push(v / norm);
            }
        }
        self.embeddings = Tensor::from_vec(flat, (vocab, self.dim), &self.device)?;
        Ok(())
    }

    /// Sync GPU embeddings back to CPU controller (for checkpointing).
    pub fn sync_embeddings_to_cpu(
        &self,
        ctrl: &mut crate::controller::LanguageController,
    ) -> Result<()> {
        let gpu_embs: Vec<Vec<f32>> = self.embeddings.to_vec2()?;
        let cpu_embs = ctrl.token_embeddings_mut();
        for (i, row) in gpu_embs.into_iter().enumerate() {
            if i < cpu_embs.len() {
                cpu_embs[i].values.copy_from_slice(&row);
            }
        }
        // Rebuild GPU embedding cache on controller (for non-GPU paths)
        #[cfg(any(feature = "mamba-cpu", feature = "gpu-logits"))]
        ctrl.rebuild_gpu_cache();
        Ok(())
    }

    /// Apply embedding gradient on GPU via outer product.
    ///
    /// Computes: emb -= lr × (error_scaled ⊗ output_hv) where error_scaled is [V, 1]
    /// and output_hv is [1, D]. The outer product gives the full [V, D] gradient.
    /// This is a single GPU matmul: [V, 1] × [1, D] → [V, D].
    pub fn gpu_embedding_gradient(
        &mut self,
        logits: &[f32],
        target: usize,
        lr: f32,
        grad_clip: f32,
    ) -> Result<()> {
        let n = logits.len();
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        if sum_exp < 1e-10 {
            return Ok(());
        }

        let scale = self.logit_scale;

        // Compute error vector (CPU — cheap, just V floats)
        let errors: Vec<f32> = (0..n)
            .map(|i| {
                let prob = exps[i] / sum_exp;
                let error = if i == target { prob - 1.0 } else { prob };
                (scale * error * lr).clamp(-grad_clip, grad_clip)
            })
            .collect();

        // Get normalized output on GPU
        let raw_output = self.network.output()?;
        let raw_norm = raw_output.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, f32::MAX)?;
        let norm_bc = raw_norm.unsqueeze(1)?.broadcast_as(raw_output.shape())?;
        let output_hat = (&raw_output / &norm_bc)?; // [1, D]

        // Outer product gradient: [V, 1] × [1, D] → [V, D] (single GPU matmul)
        let error_tensor = Tensor::from_vec(errors, (n, 1), &self.device)?;
        let grad_matrix = error_tensor.matmul(&output_hat)?; // [V, D]

        // emb -= grad (in-place on GPU)
        self.embeddings = (&self.embeddings - &grad_matrix)?;
        self.renormalize_embeddings()?;

        Ok(())
    }

    /// Apply an auxiliary embedding update from the thought HV directly.
    ///
    /// This optimizes CE(logits = thought_hat @ embeddings^T * scale, target),
    /// so the encoded thought itself becomes predictive of the next token. It is
    /// intentionally separate from the recurrent output loss: when the decoder
    /// logits are nearly flat, this gives the binding space a direct training
    /// signal instead of waiting for CfC dynamics to discover it indirectly.
    pub fn gpu_thought_logit_aux_gradient(
        &mut self,
        thought_gpu: &Tensor,
        target: usize,
        pos: usize,
        lr: f32,
        grad_clip: f32,
        weight: f32,
    ) -> Result<f32> {
        if weight <= 0.0 {
            return Ok(0.0);
        }

        let logits = self.thought_position_logits(thought_gpu, pos)?;
        let n = logits.len();
        if target >= n {
            return Ok(0.0);
        }

        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        if sum_exp < 1e-10 {
            return Ok(0.0);
        }

        let log_prob = logits[target] - max_logit - sum_exp.ln();
        let loss = -log_prob;
        let scaled_lr = lr * weight;
        let max_pos = self.position_cache.dim(0)?;
        let pos_emb = self
            .position_cache
            .get(pos.min(max_pos - 1))?
            .unsqueeze(0)?;
        let thought_query = (thought_gpu * &pos_emb)?;
        let thought_norm = thought_query.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, f32::MAX)?;
        let norm_bc = thought_norm
            .unsqueeze(1)?
            .broadcast_as(thought_query.shape())?;
        let thought_hat = (&thought_query / &norm_bc)?;
        let errors: Vec<f32> = (0..n)
            .map(|i| {
                let prob = exps[i] / sum_exp;
                let error = if i == target { prob - 1.0 } else { prob };
                (self.logit_scale * error * scaled_lr).clamp(-grad_clip, grad_clip)
            })
            .collect();

        let error_tensor = Tensor::from_vec(errors, (n, 1), &self.device)?;
        let grad_matrix = error_tensor.matmul(&thought_hat)?;
        self.embeddings = (&self.embeddings - &grad_matrix)?;
        self.renormalize_embeddings()?;

        Ok(weight * loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controller::LanguageControllerConfig;
    use crate::generator::BrocaConfig;
    use symthaea_core::genesis::GenesisSeed;

    fn softmax_probability(logits: &[f32], target: usize) -> f32 {
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        if target >= logits.len() || sum_exp <= 0.0 {
            0.0
        } else {
            exps[target] / sum_exp
        }
    }

    fn target_rank(logits: &[f32], target: usize) -> usize {
        if target >= logits.len() {
            return usize::MAX;
        }
        1 + logits
            .iter()
            .filter(|&&logit| logit > logits[target])
            .count()
    }

    fn thought_logits(trainer: &GpuTrainer, thought_gpu: &Tensor, pos: usize) -> Result<Vec<f32>> {
        let max_pos = trainer.position_cache.dim(0)?;
        let pos_emb = trainer
            .position_cache
            .get(pos.min(max_pos - 1))?
            .unsqueeze(0)?;
        let thought_query = (thought_gpu * &pos_emb)?;
        let thought_norm = thought_query.sqr()?.sum(1)?.sqrt()?.clamp(1e-8, f32::MAX)?;
        let norm_bc = thought_norm
            .unsqueeze(1)?
            .broadcast_as(thought_query.shape())?;
        let thought_hat = (&thought_query / &norm_bc)?;
        let logits_tensor =
            (thought_hat.matmul(&trainer.embeddings.t()?)? * trainer.logit_scale as f64)?;
        logits_tensor.squeeze(0)?.to_vec1()
    }

    #[test]
    #[cfg(feature = "gpu")]
    fn gpu_thought_logit_aux_gradient_improves_target_logit() {
        let genesis = GenesisSeed::from_phrase("test-gpu-thought-logit-aux");
        let config = BrocaConfig {
            controller: LanguageControllerConfig {
                network_layers: 1,
                neurons_per_layer: 2,
                vocab_size: 32,
                max_seq_len: 8,
                ..Default::default()
            },
            ..Default::default()
        };
        let generator = crate::generator::BrocaGenerator::new(&genesis, config);
        let device = Device::cuda_if_available(0).expect("device should initialize");
        let mut trainer =
            GpuTrainer::from_controller(generator.controller(), &device, 8).expect("gpu trainer");
        let thought_hv = generator
            .encoder()
            .encode(&crate::encoder::ThoughtChannels::default());
        let thought_gpu = Tensor::from_vec(
            thought_hv.as_slice().to_vec(),
            (1, thought_hv.as_slice().len()),
            &device,
        )
        .expect("thought tensor");
        let target_id = generator.tokenizer().token_id("hello") as usize;

        let before_logits = thought_logits(&trainer, &thought_gpu, 0).expect("before logits");
        let before_rank = target_rank(&before_logits, target_id);
        let before_prob = softmax_probability(&before_logits, target_id);
        let before_logit = before_logits[target_id];

        let loss = trainer
            .gpu_thought_logit_aux_gradient(&thought_gpu, target_id, 0, 0.05, 10.0, 1.0)
            .expect("aux update");

        let after_logits = thought_logits(&trainer, &thought_gpu, 0).expect("after logits");
        let after_rank = target_rank(&after_logits, target_id);
        let after_prob = softmax_probability(&after_logits, target_id);
        let after_logit = after_logits[target_id];

        assert!(loss.is_finite() && loss > 0.0);
        assert!(
            after_logit > before_logit,
            "target logit should increase: before={before_logit} after={after_logit}"
        );
        assert!(
            after_prob > before_prob,
            "target probability should increase: before={before_prob} after={after_prob}"
        );
        assert!(
            after_rank <= before_rank,
            "target rank should not get worse: before={before_rank} after={after_rank}"
        );
    }
}
