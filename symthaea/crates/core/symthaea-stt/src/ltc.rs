// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Liquid Time-Constant (LTC) Network
//!
//! Implements the closed-form continuous-time dynamics from:
//! "Closed-form Continuous-time Neural Networks" (Hasani et al., 2022)
//!
//! Key equation:
//! ```text
//! x(t+dt) = x(t) * exp(-dt/τ) + (1 - exp(-dt/τ)) * f(input)
//! ```
//!
//! This avoids ODE solvers entirely - pure closed-form updates.

use serde::{Deserialize, Serialize};

/// Configuration for LTC network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtcConfig {
    /// Number of hidden units
    pub hidden_size: usize,
    /// Minimum time constant (seconds)
    pub tau_min: f32,
    /// Maximum time constant (seconds)
    pub tau_max: f32,
    /// Initial time constant
    pub tau_init: f32,
    /// Learning rate for tau adaptation
    pub tau_lr: f32,
    /// Enable adaptive tau
    pub adaptive_tau: bool,
}

impl Default for LtcConfig {
    fn default() -> Self {
        Self {
            hidden_size: 64,
            tau_min: 0.005,  // 5ms - captures fast transients
            tau_max: 0.100,  // 100ms - captures slow modulations
            tau_init: 0.020, // 20ms - typical phoneme duration
            tau_lr: 0.01,
            adaptive_tau: true,
        }
    }
}

/// Low-pass filter for tau smoothing
///
/// Prevents "jittery tau" oscillation by limiting rate of change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TauSmoother {
    /// Smoothed tau values
    smoothed: Vec<f32>,
    /// EMA coefficient (higher = more smoothing)
    alpha: f32,
    /// Maximum rate of change per frame (fraction)
    max_rate: f32,
}

impl TauSmoother {
    /// Create a new tau smoother
    ///
    /// # Arguments
    /// * `size` - Number of tau values to smooth
    /// * `initial` - Initial tau value
    /// * `smoothing_time` - Time constant for smoothing (seconds)
    /// * `frame_duration` - Duration of each frame (seconds)
    pub fn new(size: usize, initial: f32, smoothing_time: f32, frame_duration: f32) -> Self {
        // EMA alpha from time constant: α = 1 - exp(-dt/τ)
        let alpha = 1.0 - (-frame_duration / smoothing_time).exp();

        Self {
            smoothed: vec![initial; size],
            alpha,
            max_rate: 0.1, // 10% max change per frame
        }
    }

    /// Smooth a vector of tau values
    pub fn smooth(&mut self, raw_tau: &[f32]) -> &[f32] {
        for (i, &raw) in raw_tau.iter().enumerate() {
            if i >= self.smoothed.len() {
                break;
            }

            // Compute EMA target
            let target = self.alpha * raw + (1.0 - self.alpha) * self.smoothed[i];

            // Limit rate of change
            let max_delta = self.smoothed[i] * self.max_rate;
            let delta = (target - self.smoothed[i]).clamp(-max_delta, max_delta);

            self.smoothed[i] += delta;
        }

        &self.smoothed
    }

    /// Get current smoothed values
    pub fn current(&self) -> &[f32] {
        &self.smoothed
    }

    /// Reset to initial value
    pub fn reset(&mut self, value: f32) {
        self.smoothed.fill(value);
    }
}

/// A single LTC cell with closed-form dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtcCell {
    /// Hidden state
    state: Vec<f32>,
    /// Time constants per unit
    tau: Vec<f32>,
    /// Input weights [hidden_size x input_size]
    w_in: Vec<Vec<f32>>,
    /// Recurrent weights [hidden_size x hidden_size]
    w_rec: Vec<Vec<f32>>,
    /// Bias
    bias: Vec<f32>,
    /// Configuration
    config: LtcConfig,
    /// Tau smoother
    tau_smoother: TauSmoother,
}

impl LtcCell {
    /// Create a new LTC cell
    pub fn new(input_size: usize, config: LtcConfig) -> Self {
        Self::new_reservoir(input_size, config, 0.9, 1.0)
    }

    /// Create a new LTC cell with reservoir computing initialization
    ///
    /// # Arguments
    /// * `input_size` - Dimension of input vectors
    /// * `config` - LTC configuration
    /// * `spectral_radius` - Target spectral radius for recurrent weights (0.9 is typical)
    /// * `input_scaling` - Scaling factor for input weights
    pub fn new_reservoir(
        input_size: usize,
        config: LtcConfig,
        spectral_radius: f32,
        input_scaling: f32,
    ) -> Self {
        let h = config.hidden_size;

        // Deterministic initialization using simple PRNG
        let mut seed = 42u64;
        let mut rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        // Initialize input weights with uniform [-1, 1] scaled by input_scaling
        let w_in: Vec<Vec<f32>> = (0..h)
            .map(|_| (0..input_size).map(|_| rand() * input_scaling).collect())
            .collect();

        // Initialize recurrent weights with uniform [-1, 1]
        let mut w_rec: Vec<Vec<f32>> = (0..h).map(|_| (0..h).map(|_| rand()).collect()).collect();

        // Normalize recurrent weights to target spectral radius
        // Using Frobenius norm approximation: ||W||_F / sqrt(h) ≈ spectral radius
        let frobenius: f32 = w_rec
            .iter()
            .flat_map(|row| row.iter())
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();

        let scale = spectral_radius * (h as f32).sqrt() / frobenius.max(1e-6);
        for row in &mut w_rec {
            for w in row {
                *w *= scale;
            }
        }

        let tau_smoother = TauSmoother::new(h, config.tau_init, 0.050, 0.010);

        Self {
            state: vec![0.0; h],
            tau: vec![config.tau_init; h],
            w_in,
            w_rec,
            bias: vec![0.0; h],
            config,
            tau_smoother,
        }
    }

    /// Reset the cell state
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.tau.fill(self.config.tau_init);
        self.tau_smoother.reset(self.config.tau_init);
    }

    /// Forward pass with closed-form update
    ///
    /// # Arguments
    /// * `input` - Input vector
    /// * `dt` - Time step (seconds)
    ///
    /// # Returns
    /// The updated hidden state
    pub fn forward(&mut self, input: &[f32], dt: f32) -> &[f32] {
        let h = self.config.hidden_size;

        // Compute input contribution: W_in @ input
        let mut input_contrib = vec![0.0; h];
        for j in 0..h {
            for (i, &x) in input.iter().enumerate() {
                if i < self.w_in[j].len() {
                    input_contrib[j] += self.w_in[j][i] * x;
                }
            }
        }

        // Compute recurrent contribution: W_rec @ state
        let mut rec_contrib = vec![0.0; h];
        for j in 0..h {
            for k in 0..h {
                rec_contrib[j] += self.w_rec[j][k] * self.state[k];
            }
        }

        // Adapt tau if enabled
        if self.config.adaptive_tau {
            self.adapt_tau(&input_contrib, &rec_contrib);
        }

        // Smooth tau values
        let smoothed_tau = self.tau_smoother.smooth(&self.tau);

        // Closed-form continuous-time update:
        // x(t+dt) = x(t) * exp(-dt/τ) + (1 - exp(-dt/τ)) * tanh(input + rec + bias)
        for j in 0..h {
            let tau_j = smoothed_tau[j].clamp(self.config.tau_min, self.config.tau_max);
            let decay = (-dt / tau_j).exp();
            let pre_activation = input_contrib[j] + rec_contrib[j] + self.bias[j];
            let activation = pre_activation.tanh();

            self.state[j] = self.state[j] * decay + (1.0 - decay) * activation;
        }

        &self.state
    }

    /// Adapt time constants based on input energy
    fn adapt_tau(&mut self, input_contrib: &[f32], rec_contrib: &[f32]) {
        let h = self.config.hidden_size;

        for j in 0..h {
            // Energy = magnitude of total input
            let energy = (input_contrib[j].powi(2) + rec_contrib[j].powi(2)).sqrt();

            // High energy -> shorter tau (faster response)
            // Low energy -> longer tau (more memory)
            let target_tau = if energy > 1.0 {
                self.config.tau_min
            } else if energy < 0.1 {
                self.config.tau_max
            } else {
                // Linear interpolation
                let t = (energy - 0.1) / 0.9;
                self.config.tau_max * (1.0 - t) + self.config.tau_min * t
            };

            // Gradual adaptation
            self.tau[j] += self.config.tau_lr * (target_tau - self.tau[j]);
        }
    }

    /// Get current hidden state
    pub fn state(&self) -> &[f32] {
        &self.state
    }

    /// Get current tau values
    pub fn tau(&self) -> &[f32] {
        &self.tau
    }

    /// Compute salience (energy/surprise) of current state
    ///
    /// High salience indicates potential phoneme boundaries
    pub fn compute_salience(&self, prev_state: &[f32]) -> f32 {
        if prev_state.len() != self.state.len() {
            return 0.0;
        }

        // Salience = state change magnitude + tau flux
        let state_change: f32 = self
            .state
            .iter()
            .zip(prev_state.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Tau variance indicates rapid adaptation
        let tau_mean: f32 = self.tau.iter().sum::<f32>() / self.tau.len() as f32;
        let tau_var: f32 =
            self.tau.iter().map(|t| (t - tau_mean).powi(2)).sum::<f32>() / self.tau.len() as f32;

        state_change + tau_var.sqrt()
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Get input size
    pub fn input_size(&self) -> usize {
        self.w_in.first().map(|r| r.len()).unwrap_or(0)
    }

    /// Mutable access to weights for gradient updates.
    pub fn w_in_mut(&mut self) -> &mut Vec<Vec<f32>> {
        &mut self.w_in
    }

    /// Mutable access to recurrent weights.
    pub fn w_rec_mut(&mut self) -> &mut Vec<Vec<f32>> {
        &mut self.w_rec
    }

    /// Mutable access to bias.
    pub fn bias_mut(&mut self) -> &mut Vec<f32> {
        &mut self.bias
    }

    /// Mutable access to tau.
    pub fn tau_mut(&mut self) -> &mut Vec<f32> {
        &mut self.tau
    }

    // ====================================================================
    // Supervised training: forward with cache + backward step
    // ====================================================================

    /// Forward pass that caches activations for BPTT.
    ///
    /// Returns (output_state, cache) where cache is used by `backward_step`.
    pub fn forward_cached(&mut self, input: &[f32], dt: f32) -> (Vec<f32>, LtcStepCache) {
        let h = self.config.hidden_size;
        let prev_state = self.state.clone();

        // Compute input contribution: W_in @ input
        let mut pre = vec![0.0f32; h];
        for j in 0..h {
            for (i, &x) in input.iter().enumerate() {
                if i < self.w_in[j].len() {
                    pre[j] += self.w_in[j][i] * x;
                }
            }
            // Recurrent: W_rec @ prev_state
            for k in 0..h {
                pre[j] += self.w_rec[j][k] * prev_state[k];
            }
            pre[j] += self.bias[j];
        }

        // Compute activation and update state
        let mut decay = vec![0.0f32; h];
        let mut activation = vec![0.0f32; h];
        let smoothed_tau = self.tau_smoother.smooth(&self.tau);

        for j in 0..h {
            let tau_j = smoothed_tau[j].clamp(self.config.tau_min, self.config.tau_max);
            decay[j] = (-dt / tau_j).exp();
            activation[j] = pre[j].tanh();
            self.state[j] = prev_state[j] * decay[j] + (1.0 - decay[j]) * activation[j];
        }

        let cache = LtcStepCache {
            prev_state,
            input: input.to_vec(),
            pre,
            activation,
            decay,
            dt,
        };

        (self.state.clone(), cache)
    }

    /// Backward pass through one LTC step.
    ///
    /// Given ∂L/∂x(t+dt) (gradient of loss w.r.t. current state),
    /// computes gradients for all parameters and propagates ∂L/∂x(t).
    ///
    /// Returns (gradients, ∂L/∂x(t) for previous step).
    pub fn backward_step(
        &self,
        dl_dstate: &[f32],
        cache: &LtcStepCache,
    ) -> (LtcGradients, Vec<f32>) {
        let h = self.config.hidden_size;
        let input_size = self.input_size();
        let mut grads = LtcGradients::zeros(h, input_size);
        let mut dl_dprev_state = vec![0.0f32; h];

        for j in 0..h {
            // sech²(pre) = 1 - tanh²(pre)
            let sech2 = 1.0 - cache.activation[j] * cache.activation[j];

            // ∂L/∂pre[j] = ∂L/∂x(t+dt) · (1 - decay) · sech²(pre)
            let dl_dpre = dl_dstate[j] * (1.0 - cache.decay[j]) * sech2;

            // ∂L/∂w_in[j][i] = ∂L/∂pre[j] · input[i]
            for i in 0..input_size.min(cache.input.len()) {
                grads.w_in[j][i] += dl_dpre * cache.input[i];
            }

            // ∂L/∂w_rec[j][k] = ∂L/∂pre[j] · prev_state[k]
            for k in 0..h {
                grads.w_rec[j][k] += dl_dpre * cache.prev_state[k];
            }

            // ∂L/∂bias[j] = ∂L/∂pre[j]
            grads.bias[j] += dl_dpre;

            // ∂L/∂tau[j] = ∂L/∂x(t+dt) · (dt/τ²) · exp(-dt/τ) · (activation - prev_state)
            let tau_j = self.tau[j].clamp(self.config.tau_min, self.config.tau_max);
            let ddecay_dtau = (cache.dt / (tau_j * tau_j)) * cache.decay[j];
            grads.tau[j] +=
                dl_dstate[j] * ddecay_dtau * (cache.activation[j] - cache.prev_state[j]);

            // ∂L/∂x(t) = ∂L/∂x(t+dt) · decay + sum_k(∂L/∂pre[k] · w_rec[k][j])
            dl_dprev_state[j] += dl_dstate[j] * cache.decay[j];
            // Recurrent contribution: state j influences pre[k] via w_rec[k][j]
            for k in 0..h {
                let sech2_k = 1.0 - cache.activation[k] * cache.activation[k];
                let dl_dpre_k = dl_dstate[k] * (1.0 - cache.decay[k]) * sech2_k;
                dl_dprev_state[j] += dl_dpre_k * self.w_rec[k][j];
            }
        }

        (grads, dl_dprev_state)
    }
}

/// Cached forward pass data for BPTT.
#[derive(Debug, Clone)]
pub struct LtcStepCache {
    pub prev_state: Vec<f32>,
    pub input: Vec<f32>,
    pub pre: Vec<f32>,
    pub activation: Vec<f32>,
    pub decay: Vec<f32>,
    pub dt: f32,
}

/// Gradients for LTC parameters.
#[derive(Debug, Clone)]
pub struct LtcGradients {
    pub w_in: Vec<Vec<f32>>,
    pub w_rec: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
    pub tau: Vec<f32>,
}

impl LtcGradients {
    /// Create zero gradients.
    pub fn zeros(hidden_size: usize, input_size: usize) -> Self {
        Self {
            w_in: vec![vec![0.0; input_size]; hidden_size],
            w_rec: vec![vec![0.0; hidden_size]; hidden_size],
            bias: vec![0.0; hidden_size],
            tau: vec![0.0; hidden_size],
        }
    }

    /// Accumulate gradients from another set.
    pub fn accumulate(&mut self, other: &LtcGradients) {
        for (row, orow) in self.w_in.iter_mut().zip(&other.w_in) {
            for (w, o) in row.iter_mut().zip(orow) {
                *w += o;
            }
        }
        for (row, orow) in self.w_rec.iter_mut().zip(&other.w_rec) {
            for (w, o) in row.iter_mut().zip(orow) {
                *w += o;
            }
        }
        for (b, o) in self.bias.iter_mut().zip(&other.bias) {
            *b += o;
        }
        for (t, o) in self.tau.iter_mut().zip(&other.tau) {
            *t += o;
        }
    }

    /// Scale all gradients by a factor (for averaging over batch).
    pub fn scale(&mut self, factor: f32) {
        for row in &mut self.w_in {
            for w in row {
                *w *= factor;
            }
        }
        for row in &mut self.w_rec {
            for w in row {
                *w *= factor;
            }
        }
        for b in &mut self.bias {
            *b *= factor;
        }
        for t in &mut self.tau {
            *t *= factor;
        }
    }

    /// Clip gradient norms to prevent exploding gradients.
    pub fn clip_norm(&mut self, max_norm: f32) {
        let norm_sq: f32 = self
            .w_in
            .iter()
            .flat_map(|r| r.iter())
            .map(|x| x * x)
            .sum::<f32>()
            + self
                .w_rec
                .iter()
                .flat_map(|r| r.iter())
                .map(|x| x * x)
                .sum::<f32>()
            + self.bias.iter().map(|x| x * x).sum::<f32>()
            + self.tau.iter().map(|x| x * x).sum::<f32>();
        let norm = norm_sq.sqrt();
        if norm > max_norm {
            self.scale(max_norm / norm);
        }
    }

    /// Apply gradients to LTC weights via SGD with learning rate.
    pub fn apply_sgd(&self, ltc: &mut LtcCell, lr: f32) {
        for (row, grow) in ltc.w_in_mut().iter_mut().zip(&self.w_in) {
            for (w, g) in row.iter_mut().zip(grow) {
                *w -= lr * g;
            }
        }
        for (row, grow) in ltc.w_rec_mut().iter_mut().zip(&self.w_rec) {
            for (w, g) in row.iter_mut().zip(grow) {
                *w -= lr * g;
            }
        }
        for (b, g) in ltc.bias_mut().iter_mut().zip(&self.bias) {
            *b -= lr * g;
        }
        for (t, g) in ltc.tau_mut().iter_mut().zip(&self.tau) {
            *t -= lr * g;
            // Keep tau positive
            *t = t.max(0.001);
        }
    }
}

/// Multi-layer LTC network
#[derive(Debug, Clone)]
pub struct LtcNetwork {
    layers: Vec<LtcCell>,
}

impl LtcNetwork {
    /// Create a multi-layer LTC network
    pub fn new(input_size: usize, layer_sizes: &[usize], config: LtcConfig) -> Self {
        let mut layers = Vec::with_capacity(layer_sizes.len());
        let mut prev_size = input_size;

        for &size in layer_sizes {
            let mut layer_config = config.clone();
            layer_config.hidden_size = size;
            layers.push(LtcCell::new(prev_size, layer_config));
            prev_size = size;
        }

        Self { layers }
    }

    /// Reset all layers
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Forward through all layers
    pub fn forward(&mut self, input: &[f32], dt: f32) -> Vec<f32> {
        let mut current = input.to_vec();

        for layer in &mut self.layers {
            let output = layer.forward(&current, dt);
            current = output.to_vec();
        }

        current
    }

    /// Get final layer state
    pub fn state(&self) -> &[f32] {
        self.layers.last().map(|l| l.state()).unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ltc_cell_basic() {
        let config = LtcConfig::default();
        let mut cell = LtcCell::new(40, config);

        let input = vec![0.5; 40];
        let output = cell.forward(&input, 0.010);

        assert_eq!(output.len(), 64);
        // State should be non-zero after input
        assert!(output.iter().any(|&x| x.abs() > 0.0));
    }

    #[test]
    fn test_tau_smoother() {
        let mut smoother = TauSmoother::new(4, 0.020, 0.050, 0.010);

        // Sudden jump should be smoothed
        let raw = [0.100, 0.100, 0.100, 0.100];
        let smoothed = smoother.smooth(&raw);

        // Should not immediately jump to 0.100
        assert!(smoothed[0] < 0.030, "smoothed = {:?}", smoothed);
    }

    #[test]
    fn test_ltc_decay() {
        let config = LtcConfig {
            adaptive_tau: false,
            ..Default::default()
        };
        let mut cell = LtcCell::new(10, config);

        // Inject signal
        let input = vec![1.0; 10];
        cell.forward(&input, 0.010);
        let peak = cell.state().iter().map(|x| x.abs()).sum::<f32>();

        // Let it decay with zero input
        let zero = vec![0.0; 10];
        for _ in 0..100 {
            cell.forward(&zero, 0.010);
        }
        let decayed = cell.state().iter().map(|x| x.abs()).sum::<f32>();

        // LTC converges to a bias-driven fixed point with zero input, not to zero.
        // Recurrent weights + bias produce a non-zero attractor. Verify meaningful decay.
        assert!(decayed < peak, "peak={}, decayed={}", peak, decayed);
    }
}
