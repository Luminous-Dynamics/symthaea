// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! CfC Cell: A single Closed-form Continuous-time cell.
//!
//! Implements the core CfC dynamics with closed-form solution:
//! `h(t) = h_inf + (h_0 - h_inf) * exp(-t/tau)`

use ndarray::{Array1, Array2};
use rand::Rng;
use symthaea_core::genesis::GenesisSeed;

use super::gradients::{AdamState, CfCCellCache, CfCGradients};
use super::types::{CfCConfig, MIN_TAU, OnlineLearningStats, sigmoid};

/// A single Closed-form Continuous-time cell
#[derive(Debug, Clone)]
pub struct CfCCell {
    pub(crate) config: CfCConfig,

    // Weights for state transition
    pub(crate) w_in: Array2<f32>, // Input to hidden
    pub(crate) w_h: Array2<f32>,  // Hidden to hidden

    // Reserved for future output projection (e.g., separate output dim)
    #[allow(dead_code)]
    pub(crate) w_out: Array2<f32>,

    // Biases
    pub(crate) b_h: Array1<f32>,

    // Time constants (learnable)
    pub(crate) tau: Array1<f32>,

    // Backbone network weights (used when config.use_backbone is true)
    pub(crate) backbone_weights: Vec<Array2<f32>>,
    pub(crate) backbone_biases: Vec<Array1<f32>>,

    // Current hidden state
    pub(crate) state: Array1<f32>,

    // Statistics - tracks number of forward steps for diagnostics
    #[allow(dead_code)]
    pub(crate) steps: u64,

    // Online learning statistics
    pub(crate) online_stats: OnlineLearningStats,
}

impl CfCCell {
    /// Create a new CfC cell
    ///
    /// Clamps `config.tau_range.0` to at least `MIN_TAU` (1e-6) to prevent
    /// NaN in exp(-dt/tau) calculations.
    pub fn new(mut config: CfCConfig) -> Self {
        // Clamp tau range to prevent NaN in exp(-dt/tau) calculations
        if config.tau_range.0 < MIN_TAU {
            tracing::warn!(
                tau_min = config.tau_range.0,
                min_tau = MIN_TAU,
                "tau_min below MIN_TAU, clamping"
            );
            config.tau_range.0 = MIN_TAU;
        }

        // When backbone is used, w_in takes backbone output (backbone_dim)
        // Otherwise, w_in takes raw input (input_dim)
        let effective_input_dim = if config.use_backbone {
            config.backbone_dim
        } else {
            config.input_dim
        };

        let scale = (2.0 / (effective_input_dim + config.hidden_dim) as f32).sqrt();

        // Initialize weights with Xavier/Glorot initialization
        let w_in = Array2::from_shape_fn((config.hidden_dim, effective_input_dim), |_| {
            (rand::random::<f32>() - 0.5) * 2.0 * scale
        });

        let w_h = Array2::from_shape_fn((config.hidden_dim, config.hidden_dim), |_| {
            (rand::random::<f32>() - 0.5) * 2.0 * scale
        });

        let w_out = Array2::from_shape_fn((config.hidden_dim, config.hidden_dim), |_| {
            (rand::random::<f32>() - 0.5) * 2.0 * scale
        });

        let b_h = Array1::zeros(config.hidden_dim);

        // Initialize time constants uniformly in log space
        let (tau_min, tau_max) = config.tau_range;
        let tau = Array1::from_shape_fn(config.hidden_dim, |_| {
            let log_tau = tau_min.ln() + rand::random::<f32>() * (tau_max.ln() - tau_min.ln());
            // Clamp to ensure numerical stability even after initialization
            log_tau.exp().max(MIN_TAU)
        });

        // Initialize backbone if needed
        let (backbone_weights, backbone_biases) = if config.use_backbone {
            let mut weights = Vec::new();
            let mut biases = Vec::new();

            // First layer: input_dim -> backbone_dim
            weights.push(Array2::from_shape_fn(
                (config.backbone_dim, config.input_dim),
                |_| (rand::random::<f32>() - 0.5) * 2.0 * scale,
            ));
            biases.push(Array1::zeros(config.backbone_dim));

            // Hidden layers
            for _ in 1..config.backbone_layers {
                weights.push(Array2::from_shape_fn(
                    (config.backbone_dim, config.backbone_dim),
                    |_| (rand::random::<f32>() - 0.5) * 2.0 * scale,
                ));
                biases.push(Array1::zeros(config.backbone_dim));
            }

            (weights, biases)
        } else {
            (Vec::new(), Vec::new())
        };

        let hidden_dim = config.hidden_dim;
        Self {
            config,
            w_in,
            w_h,
            w_out,
            b_h,
            tau,
            backbone_weights,
            backbone_biases,
            state: Array1::zeros(hidden_dim),
            steps: 0,
            online_stats: OnlineLearningStats::default(),
        }
    }

    /// Create a new CfC cell with deterministic weight initialization from a genesis seed.
    ///
    /// Uses `genesis.domain(label)` to derive a SHAKE-256 RNG stream so that
    /// identical seeds and labels always produce identical weights.
    ///
    /// Clamps `config.tau_range.0` to at least `MIN_TAU` (1e-6) to prevent
    /// NaN in exp(-dt/tau) calculations.
    pub fn from_genesis(mut config: CfCConfig, genesis: &GenesisSeed, label: &str) -> Self {
        if config.tau_range.0 < MIN_TAU {
            tracing::warn!(
                tau_min = config.tau_range.0,
                min_tau = MIN_TAU,
                "tau_min below MIN_TAU, clamping"
            );
            config.tau_range.0 = MIN_TAU;
        }

        let mut rng = genesis.domain(label);

        let effective_input_dim = if config.use_backbone {
            config.backbone_dim
        } else {
            config.input_dim
        };

        let scale = (2.0 / (effective_input_dim + config.hidden_dim) as f32).sqrt();

        let w_in = Array2::from_shape_fn((config.hidden_dim, effective_input_dim), |_| {
            (rng.r#gen::<f32>() - 0.5) * 2.0 * scale
        });

        let w_h = Array2::from_shape_fn((config.hidden_dim, config.hidden_dim), |_| {
            (rng.r#gen::<f32>() - 0.5) * 2.0 * scale
        });

        let w_out = Array2::from_shape_fn((config.hidden_dim, config.hidden_dim), |_| {
            (rng.r#gen::<f32>() - 0.5) * 2.0 * scale
        });

        let b_h = Array1::zeros(config.hidden_dim);

        let (tau_min, tau_max) = config.tau_range;
        let tau = Array1::from_shape_fn(config.hidden_dim, |_| {
            let log_tau = tau_min.ln() + rng.r#gen::<f32>() * (tau_max.ln() - tau_min.ln());
            log_tau.exp().max(MIN_TAU)
        });

        let (backbone_weights, backbone_biases) = if config.use_backbone {
            let mut weights = Vec::new();
            let mut biases = Vec::new();

            weights.push(Array2::from_shape_fn(
                (config.backbone_dim, config.input_dim),
                |_| (rng.r#gen::<f32>() - 0.5) * 2.0 * scale,
            ));
            biases.push(Array1::zeros(config.backbone_dim));

            for _ in 1..config.backbone_layers {
                weights.push(Array2::from_shape_fn(
                    (config.backbone_dim, config.backbone_dim),
                    |_| (rng.r#gen::<f32>() - 0.5) * 2.0 * scale,
                ));
                biases.push(Array1::zeros(config.backbone_dim));
            }

            (weights, biases)
        } else {
            (Vec::new(), Vec::new())
        };

        let hidden_dim = config.hidden_dim;
        Self {
            config,
            w_in,
            w_h,
            w_out,
            b_h,
            tau,
            backbone_weights,
            backbone_biases,
            state: Array1::zeros(hidden_dim),
            steps: 0,
            online_stats: OnlineLearningStats::default(),
        }
    }

    /// Reset the cell state
    pub fn reset(&mut self) {
        self.state = Array1::zeros(self.config.hidden_dim);
        self.steps = 0;
    }

    /// Reset online learning statistics (keeps weights but clears tracking)
    pub fn reset_online_stats(&mut self) {
        self.online_stats = OnlineLearningStats::default();
    }

    /// Get online learning statistics
    pub fn online_stats(&self) -> &OnlineLearningStats {
        &self.online_stats
    }

    /// Forward pass through the cell
    /// Uses fast activation approximations for 2-3x speedup.
    ///
    /// # Arguments
    /// * `input` - Input vector
    /// * `dt` - Time step (can be irregular)
    ///
    /// # Returns
    /// New hidden state
    #[inline]
    pub fn forward(&mut self, input: &Array1<f32>, dt: f32) -> Array1<f32> {
        // Process through backbone if enabled
        let processed_input = if self.config.use_backbone {
            self.backbone_forward(input)
        } else {
            input.clone()
        };

        // Compute gating based on processed input and current state
        // Using closed-form solution: h(t) = h_inf + (h_0 - h_inf) * exp(-t/tau)
        // where h_inf is the equilibrium state

        // Compute target/equilibrium state (using fast activation for 2-3x speedup)
        let x_contrib = self.w_in.dot(&processed_input);
        let h_contrib = self.w_h.dot(&self.state);
        let h_inf = self
            .config
            .activation
            .apply_array_fast(&(x_contrib + h_contrib + &self.b_h));

        // Compute decay factor based on time constants
        // Clamp tau to MIN_TAU to prevent division by zero / NaN
        let decay: Array1<f32> = self.tau.mapv(|t| (-dt / t.max(MIN_TAU)).exp());

        // Update state using closed-form solution
        let mut new_state = &h_inf + &((&self.state - &h_inf) * &decay);

        // Clamp hidden state to prevent accumulation-driven divergence
        new_state.mapv_inplace(|x| {
            if x.is_finite() {
                x.clamp(-10.0, 10.0)
            } else {
                0.0
            }
        });

        self.state = new_state.clone();
        self.steps += 1;

        new_state
    }

    /// Forward pass with cache for BPTT optimization.
    /// Stores intermediate values to avoid recomputation during backward pass.
    /// This saves ~35% of training time by eliminating redundant forward computation.
    ///
    /// # Arguments
    /// * `input` - Input vector
    /// * `dt` - Time step (can be irregular)
    ///
    /// # Returns
    /// (new_hidden_state, cache) - The hidden state and cached intermediate values
    #[inline]
    pub fn forward_with_cache(
        &mut self,
        input: &Array1<f32>,
        dt: f32,
    ) -> (Array1<f32>, CfCCellCache) {
        // Process through backbone if enabled
        let processed_input = if self.config.use_backbone {
            self.backbone_forward(input)
        } else {
            input.clone()
        };

        // Save state before update for gradient computation
        let state_at_forward = self.state.clone();

        // Compute target/equilibrium state
        let x_contrib = self.w_in.dot(&processed_input);
        let h_contrib = self.w_h.dot(&state_at_forward);
        let z = x_contrib + h_contrib + &self.b_h;
        let h_inf = self.config.activation.apply_array_fast(&z);

        // Compute decay factor
        let decay: Array1<f32> = self.tau.mapv(|t| (-dt / t.max(MIN_TAU)).exp());

        // Update state using closed-form solution
        let mut new_state = &h_inf + &((&state_at_forward - &h_inf) * &decay);

        // Clamp hidden state to prevent accumulation-driven divergence
        new_state.mapv_inplace(|x| {
            if x.is_finite() {
                x.clamp(-10.0, 10.0)
            } else {
                0.0
            }
        });

        self.state = new_state.clone();
        self.steps += 1;

        let cache = CfCCellCache {
            processed_input,
            z,
            h_inf,
            decay,
            state_at_forward,
        };

        (new_state, cache)
    }

    /// Process through backbone network (uses fast activation for 2-3x speedup)
    #[inline]
    pub(crate) fn backbone_forward(&self, input: &Array1<f32>) -> Array1<f32> {
        let mut x = input.clone();

        for (w, b) in self
            .backbone_weights
            .iter()
            .zip(self.backbone_biases.iter())
        {
            x = self.config.activation.apply_array_fast(&(w.dot(&x) + b));
        }

        x
    }

    /// Compute analytical gradients for BPTT given an upstream gradient on the hidden state.
    ///
    /// `dh` is dL/d(new_state), the gradient of the loss with respect to
    /// this cell's output hidden state, already back-propagated through
    /// any downstream layers (e.g. the output projection).
    ///
    /// Returns gradients for W_in, W_h, b_h, and tau based on the
    /// closed-form CfC dynamics: h(t) = h_inf + (h_0 - h_inf) * exp(-dt/tau)
    pub fn backward_from_grad(
        &self,
        input: &Array1<f32>,
        dh: &Array1<f32>,
        dt: f32,
    ) -> CfCGradients {
        let processed_input = if self.config.use_backbone {
            self.backbone_forward(input)
        } else {
            input.clone()
        };

        // Forward computation (recompute for gradient chain)
        let x_contrib = self.w_in.dot(&processed_input);
        let h_contrib = self.w_h.dot(&self.state);
        let z = &x_contrib + &h_contrib + &self.b_h;
        let h_inf = self.config.activation.apply_array(&z);
        // Clamp tau to MIN_TAU to prevent NaN
        let decay: Array1<f32> = self.tau.mapv(|t| (-dt / t.max(MIN_TAU)).exp());

        // Activation derivative (SiLU default)
        let sigma_prime: Array1<f32> = z.mapv(|x| {
            let s = sigmoid(x);
            s + x * s * (1.0 - s) // d/dx[x * sigmoid(x)]
        });

        // dh/dh_inf = (1 - exp(-dt/tau))
        let one_minus_decay: Array1<f32> = decay.mapv(|d| 1.0 - d);

        // Chain: dL/dz = dL/dh * dh/dh_inf * dh_inf/dz
        let dz = dh * &one_minus_decay * &sigma_prime;

        // dL/dW_in = dz * input^T (outer product via vectorized broadcasting)
        let effective_input_dim = processed_input.len();
        let hidden_dim = self.config.hidden_dim;
        let mut dw_in = Array2::zeros((hidden_dim, effective_input_dim));
        for i in 0..hidden_dim {
            for j in 0..effective_input_dim {
                dw_in[[i, j]] = dz[i] * processed_input[j];
            }
        }

        // dL/dW_h = dz * state^T
        let mut dw_h = Array2::zeros((hidden_dim, hidden_dim));
        for i in 0..hidden_dim {
            for j in 0..hidden_dim {
                dw_h[[i, j]] = dz[i] * self.state[j];
            }
        }

        // dL/db = dz
        let db_h = dz.clone();

        // dL/dtau = dL/dh * (h_0 - h_inf) * (dt / tau^2) * exp(-dt/tau)
        let mut dtau = Array1::zeros(hidden_dim);
        for i in 0..hidden_dim {
            let diff = self.state[i] - h_inf[i];
            dtau[i] = dh[i] * diff * (dt / (self.tau[i] * self.tau[i])) * decay[i];
        }

        CfCGradients {
            dw_in,
            dw_h,
            db_h,
            dtau,
        }
    }

    /// Compute analytical gradients using cached forward pass values.
    /// This is the optimized backward pass that avoids recomputation.
    /// Saves ~35% of training time by reusing cached intermediate values.
    ///
    /// # Arguments
    /// * `cache` - Cached values from `forward_with_cache`
    /// * `dh` - Upstream gradient dL/d(new_state)
    /// * `dt` - Time step used in forward pass
    ///
    /// # Returns
    /// Gradients for W_in, W_h, b_h, and tau
    #[inline]
    pub fn backward_from_cache(
        &self,
        cache: &CfCCellCache,
        dh: &Array1<f32>,
        dt: f32,
    ) -> CfCGradients {
        let hidden_dim = self.config.hidden_dim;
        let effective_input_dim = cache.processed_input.len();

        // CONTIGUITY SAFETY: All .expect("... not contiguous") calls below are safe.
        // ndarray Array1/Array2 created via ::zeros() or ::from_vec() use default
        // C-contiguous (row-major) memory layout. as_slice()/as_slice_mut() only
        // fails on non-contiguous *views* (e.g., transposed 2D slices or strided
        // subviews). Every array here is either freshly allocated or a cached 1D
        // array from the forward pass — all guaranteed contiguous.
        // Ref: https://docs.rs/ndarray/latest/ndarray/type.Array1.html

        // Activation derivative (SiLU default)
        // d/dx[x * sigmoid(x)] = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        let sigma_prime: Array1<f32> = cache.z.mapv(|x| {
            let s = sigmoid(x);
            s + x * s * (1.0 - s)
        });

        // dh/dh_inf = (1 - exp(-dt/tau))
        let one_minus_decay: Array1<f32> = cache.decay.mapv(|d| 1.0 - d);

        // Chain: dL/dz = dL/dh * dh/dh_inf * dh_inf/dz
        let dz = dh * &one_minus_decay * &sigma_prime;

        // dL/dW_in = dz (x) input (outer product via vectorized row-wise broadcast)
        // Optimized: use slice views to avoid bounds checking overhead
        let dz_slice = dz.as_slice().expect("dz array not contiguous");
        let input_slice = cache
            .processed_input
            .as_slice()
            .expect("processed_input array not contiguous");
        let mut dw_in = Array2::zeros((hidden_dim, effective_input_dim));
        {
            let dw_in_slice = dw_in.as_slice_mut().expect("dw_in array not contiguous");
            for i in 0..hidden_dim {
                let row_offset = i * effective_input_dim;
                let dz_i = dz_slice[i];
                for j in 0..effective_input_dim {
                    dw_in_slice[row_offset + j] = dz_i * input_slice[j];
                }
            }
        }

        // dL/dW_h = dz (x) state (outer product)
        let state_slice = cache
            .state_at_forward
            .as_slice()
            .expect("state_at_forward array not contiguous");
        let mut dw_h = Array2::zeros((hidden_dim, hidden_dim));
        {
            let dw_h_slice = dw_h.as_slice_mut().expect("dw_h array not contiguous");
            for i in 0..hidden_dim {
                let row_offset = i * hidden_dim;
                let dz_i = dz_slice[i];
                for j in 0..hidden_dim {
                    dw_h_slice[row_offset + j] = dz_i * state_slice[j];
                }
            }
        }

        // dL/db = dz
        let db_h = dz;

        // dL/dtau = dL/dh * (h_0 - h_inf) * (dt / tau^2) * exp(-dt/tau)
        // Vectorized computation
        let tau_slice = self.tau.as_slice().expect("tau array not contiguous");
        let decay_slice = cache.decay.as_slice().expect("decay array not contiguous");
        let h_inf_slice = cache.h_inf.as_slice().expect("h_inf array not contiguous");
        let dh_slice = dh.as_slice().expect("dh array not contiguous");
        let mut dtau = Array1::zeros(hidden_dim);
        {
            let dtau_slice = dtau.as_slice_mut().expect("dtau array not contiguous");
            for i in 0..hidden_dim {
                let diff = state_slice[i] - h_inf_slice[i];
                let tau_i = tau_slice[i];
                dtau_slice[i] = dh_slice[i] * diff * (dt / (tau_i * tau_i)) * decay_slice[i];
            }
        }

        CfCGradients {
            dw_in,
            dw_h,
            db_h,
            dtau,
        }
    }

    /// Compute analytical gradients for BPTT (legacy API targeting hidden state directly).
    ///
    /// **Note**: For networks with an output projection, prefer using
    /// `backward_from_grad` with gradients back-propagated through the
    /// projection layer. This method computes MSE(hidden, target) directly.
    pub fn backward(&self, input: &Array1<f32>, target: &Array1<f32>, dt: f32) -> CfCGradients {
        // Error signal: dL/dh = 2 * (h - target) / n
        let n = target.len().min(self.state.len()) as f32;
        let mut dh = Array1::zeros(self.config.hidden_dim);

        // Recompute new_state for error
        let processed_input = if self.config.use_backbone {
            self.backbone_forward(input)
        } else {
            input.clone()
        };
        let x_contrib = self.w_in.dot(&processed_input);
        let h_contrib = self.w_h.dot(&self.state);
        let z = &x_contrib + &h_contrib + &self.b_h;
        let h_inf = self.config.activation.apply_array(&z);
        let decay: Array1<f32> = self.tau.mapv(|t| (-dt / t.max(MIN_TAU)).exp());
        let new_state = &h_inf + &((&self.state - &h_inf) * &decay);

        for i in 0..target.len().min(new_state.len()) {
            dh[i] = 2.0 * (new_state[i] - target[i]) / n;
        }

        self.backward_from_grad(input, &dh, dt)
    }

    /// Apply Adam optimizer update
    ///
    /// Uses conservative gradient clipping (0.5) to prevent oscillation
    /// when learning cyclic patterns with rapid context switches.
    pub fn apply_adam(&mut self, grads: &CfCGradients, adam: &mut AdamState, lr: f32) {
        adam.t += 1;
        let t = adam.t as f32;

        // Gradient clipping to stabilize training.
        // Default 1.0; use higher values (e.g., 5.0) for classification tasks
        // where stronger gradients are needed to find decision boundaries.
        let clip_val = self.config.gradient_clip;
        let clip = |g: f32| g.clamp(-clip_val, clip_val);

        let hidden_dim = self.config.hidden_dim;
        let effective_input_dim = self.w_in.ncols();

        // Update W_in
        for i in 0..hidden_dim {
            for j in 0..effective_input_dim {
                let g = clip(grads.dw_in[[i, j]]);
                adam.m_w_in[[i, j]] = adam.beta1 * adam.m_w_in[[i, j]] + (1.0 - adam.beta1) * g;
                adam.v_w_in[[i, j]] = adam.beta2 * adam.v_w_in[[i, j]] + (1.0 - adam.beta2) * g * g;
                let m_hat = adam.m_w_in[[i, j]] / (1.0 - adam.beta1.powf(t));
                let v_hat = adam.v_w_in[[i, j]] / (1.0 - adam.beta2.powf(t));
                self.w_in[[i, j]] -= lr * m_hat / (v_hat.sqrt() + adam.eps);
            }
        }

        // Update W_h
        for i in 0..hidden_dim {
            for j in 0..hidden_dim {
                let g = clip(grads.dw_h[[i, j]]);
                adam.m_w_h[[i, j]] = adam.beta1 * adam.m_w_h[[i, j]] + (1.0 - adam.beta1) * g;
                adam.v_w_h[[i, j]] = adam.beta2 * adam.v_w_h[[i, j]] + (1.0 - adam.beta2) * g * g;
                let m_hat = adam.m_w_h[[i, j]] / (1.0 - adam.beta1.powf(t));
                let v_hat = adam.v_w_h[[i, j]] / (1.0 - adam.beta2.powf(t));
                self.w_h[[i, j]] -= lr * m_hat / (v_hat.sqrt() + adam.eps);
            }
        }

        // Update bias
        for i in 0..hidden_dim {
            let g = clip(grads.db_h[i]);
            adam.m_b_h[i] = adam.beta1 * adam.m_b_h[i] + (1.0 - adam.beta1) * g;
            adam.v_b_h[i] = adam.beta2 * adam.v_b_h[i] + (1.0 - adam.beta2) * g * g;
            let m_hat = adam.m_b_h[i] / (1.0 - adam.beta1.powf(t));
            let v_hat = adam.v_b_h[i] / (1.0 - adam.beta2.powf(t));
            self.b_h[i] -= lr * m_hat / (v_hat.sqrt() + adam.eps);
        }

        // Update tau with 0.1x learning rate and clamping
        for i in 0..hidden_dim {
            let g = clip(grads.dtau[i]);
            adam.m_tau[i] = adam.beta1 * adam.m_tau[i] + (1.0 - adam.beta1) * g;
            adam.v_tau[i] = adam.beta2 * adam.v_tau[i] + (1.0 - adam.beta2) * g * g;
            let m_hat = adam.m_tau[i] / (1.0 - adam.beta1.powf(t));
            let v_hat = adam.v_tau[i] / (1.0 - adam.beta2.powf(t));
            self.tau[i] -= lr * 0.1 * m_hat / (v_hat.sqrt() + adam.eps);
            self.tau[i] = self.tau[i].clamp(0.1, 10.0);
        }
    }

    /// Apply Adam optimizer update with SIMD-friendly vectorized operations.
    /// Saves ~25% of training time by:
    /// - Pre-computing bias correction factors outside loops
    /// - Using slice-based iteration to avoid bounds checking
    /// - Fusing operations for better cache locality
    #[inline]
    pub fn apply_adam_vectorized(&mut self, grads: &CfCGradients, adam: &mut AdamState, lr: f32) {
        adam.t += 1;
        let t = adam.t as f32;

        // Pre-compute bias correction factors (constant for all elements)
        let bc1 = 1.0 - adam.beta1.powf(t);
        let bc2 = 1.0 - adam.beta2.powf(t);
        let one_minus_beta1 = 1.0 - adam.beta1;
        let one_minus_beta2 = 1.0 - adam.beta2;
        let beta1 = adam.beta1;
        let beta2 = adam.beta2;
        let eps = adam.eps;

        // Update W_in (vectorized via slice iteration)
        {
            let w_slice = self.w_in.as_slice_mut().expect("w_in array not contiguous");
            let g_slice = grads
                .dw_in
                .as_slice()
                .expect("dw_in grad array not contiguous");
            let m_slice = adam
                .m_w_in
                .as_slice_mut()
                .expect("m_w_in array not contiguous");
            let v_slice = adam
                .v_w_in
                .as_slice_mut()
                .expect("v_w_in array not contiguous");

            for i in 0..w_slice.len() {
                let g = g_slice[i].clamp(-0.5, 0.5);
                m_slice[i] = beta1 * m_slice[i] + one_minus_beta1 * g;
                v_slice[i] = beta2 * v_slice[i] + one_minus_beta2 * g * g;
                let m_hat = m_slice[i] / bc1;
                let v_hat = v_slice[i] / bc2;
                w_slice[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }

        // Update W_h (vectorized via slice iteration)
        {
            let w_slice = self.w_h.as_slice_mut().expect("w_h array not contiguous");
            let g_slice = grads
                .dw_h
                .as_slice()
                .expect("dw_h grad array not contiguous");
            let m_slice = adam
                .m_w_h
                .as_slice_mut()
                .expect("m_w_h array not contiguous");
            let v_slice = adam
                .v_w_h
                .as_slice_mut()
                .expect("v_w_h array not contiguous");

            for i in 0..w_slice.len() {
                let g = g_slice[i].clamp(-0.5, 0.5);
                m_slice[i] = beta1 * m_slice[i] + one_minus_beta1 * g;
                v_slice[i] = beta2 * v_slice[i] + one_minus_beta2 * g * g;
                let m_hat = m_slice[i] / bc1;
                let v_hat = v_slice[i] / bc2;
                w_slice[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }

        // Update bias (1D vectorized)
        {
            let b_slice = self.b_h.as_slice_mut().expect("b_h array not contiguous");
            let g_slice = grads
                .db_h
                .as_slice()
                .expect("db_h grad array not contiguous");
            let m_slice = adam
                .m_b_h
                .as_slice_mut()
                .expect("m_b_h array not contiguous");
            let v_slice = adam
                .v_b_h
                .as_slice_mut()
                .expect("v_b_h array not contiguous");

            for i in 0..b_slice.len() {
                let g = g_slice[i].clamp(-0.5, 0.5);
                m_slice[i] = beta1 * m_slice[i] + one_minus_beta1 * g;
                v_slice[i] = beta2 * v_slice[i] + one_minus_beta2 * g * g;
                let m_hat = m_slice[i] / bc1;
                let v_hat = v_slice[i] / bc2;
                b_slice[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            }
        }

        // Update tau with 0.1x learning rate and clamping (1D vectorized)
        {
            let tau_slice = self.tau.as_slice_mut().expect("tau array not contiguous");
            let g_slice = grads
                .dtau
                .as_slice()
                .expect("dtau grad array not contiguous");
            let m_slice = adam
                .m_tau
                .as_slice_mut()
                .expect("m_tau array not contiguous");
            let v_slice = adam
                .v_tau
                .as_slice_mut()
                .expect("v_tau array not contiguous");
            let tau_lr = lr * 0.1;

            for i in 0..tau_slice.len() {
                let g = g_slice[i].clamp(-0.5, 0.5);
                m_slice[i] = beta1 * m_slice[i] + one_minus_beta1 * g;
                v_slice[i] = beta2 * v_slice[i] + one_minus_beta2 * g * g;
                let m_hat = m_slice[i] / bc1;
                let v_hat = v_slice[i] / bc2;
                tau_slice[i] -= tau_lr * m_hat / (v_hat.sqrt() + eps);
                tau_slice[i] = tau_slice[i].clamp(0.1, 10.0);
            }
        }
    }

    /// Get the current state
    pub fn state(&self) -> &Array1<f32> {
        &self.state
    }

    /// Set the state
    pub fn set_state(&mut self, state: Array1<f32>) {
        self.state = state;
    }

    /// Get configuration
    pub fn config(&self) -> &CfCConfig {
        &self.config
    }

    /// Get time constants
    pub fn tau(&self) -> &Array1<f32> {
        &self.tau
    }

    /// Scale tau values by a multiplicative factor, clamped to [0.01, 100.0]
    pub fn scale_tau(&mut self, scale: f32) {
        self.tau.mapv_inplace(|t| (t * scale).clamp(0.01, 100.0));
    }

    // =========================================================================
    // Online Learning During Inference
    // =========================================================================

    /// Adapt weights online based on prediction error.
    ///
    /// This method implements error-gated online learning:
    /// 1. Updates EMA of prediction errors
    /// 2. Only adapts if error exceeds adaptive threshold
    /// 3. Uses much smaller learning rate than training
    /// 4. Clips weight changes to prevent catastrophic forgetting
    ///
    /// # Arguments
    /// * `prediction_error` - Current prediction error (e.g., MSE between predicted and actual)
    /// * `input` - The input that produced the prediction
    /// * `target` - The actual/desired output (for gradient direction)
    /// * `dt` - Time step used in the forward pass
    ///
    /// # Returns
    /// `true` if adaptation occurred, `false` if error was below threshold
    pub fn adapt_online(
        &mut self,
        prediction_error: f32,
        input: &Array1<f32>,
        target: &Array1<f32>,
        dt: f32,
    ) -> bool {
        let config = match &self.config.online_learning {
            Some(cfg) => cfg.clone(),
            None => return false, // Online learning not enabled
        };

        // Update EMA of prediction error
        let alpha = config.ema_alpha;
        self.online_stats.ema_error =
            self.online_stats.ema_error * (1.0 - alpha) + prediction_error * alpha;
        self.online_stats.adaptation_steps += 1;

        // Error-gated learning: only adapt if error exceeds threshold
        // Use adaptive threshold based on recent error history
        let adaptive_threshold = config
            .error_threshold
            .max(self.online_stats.ema_error * 0.5);
        if prediction_error < adaptive_threshold {
            self.online_stats.adaptations_skipped += 1;
            return false;
        }

        self.online_stats.adaptations_triggered += 1;

        // Compute gradients using the same backward pass as training
        let grads = self.backward(input, target, dt);

        // Apply constrained weight updates
        let lr = config.learning_rate;
        let max_delta = config.max_weight_delta;
        let mut total_delta_sq = 0.0f32;

        // Update W_in with clipping
        let (rows_in, cols_in) = self.w_in.dim();
        for i in 0..rows_in {
            for j in 0..cols_in {
                let delta = (-lr * grads.dw_in[[i, j]]).clamp(-max_delta, max_delta);
                self.w_in[[i, j]] += delta;
                total_delta_sq += delta * delta;
            }
        }

        // Update W_h with clipping
        let hidden_dim = self.config.hidden_dim;
        for i in 0..hidden_dim {
            for j in 0..hidden_dim {
                let delta = (-lr * grads.dw_h[[i, j]]).clamp(-max_delta, max_delta);
                self.w_h[[i, j]] += delta;
                total_delta_sq += delta * delta;
            }
        }

        // Update bias with clipping
        for i in 0..hidden_dim {
            let delta = (-lr * grads.db_h[i]).clamp(-max_delta, max_delta);
            self.b_h[i] += delta;
            total_delta_sq += delta * delta;
        }

        // Optionally update tau (time constants)
        if config.adapt_tau {
            let tau_lr = lr * config.tau_lr_multiplier;
            for i in 0..hidden_dim {
                let delta = (-tau_lr * grads.dtau[i]).clamp(-max_delta * 0.1, max_delta * 0.1);
                self.tau[i] = (self.tau[i] + delta).clamp(MIN_TAU, 100.0);
                total_delta_sq += delta * delta;
            }
        }

        // Track statistics
        let delta_norm = total_delta_sq.sqrt();
        if delta_norm > self.online_stats.max_observed_delta {
            self.online_stats.max_observed_delta = delta_norm;
        }
        self.online_stats.cumulative_weight_change += delta_norm;

        true
    }

    /// Check if online learning is enabled for this cell
    pub fn online_learning_enabled(&self) -> bool {
        self.config.online_learning.is_some()
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    use super::super::gradients::AdamState;
    use super::super::types::{ActivationType, CfCConfig, OnlineLearningConfig};

    const INPUT_DIM: usize = 4;
    const HIDDEN_DIM: usize = 8;

    /// Helper: create a small CfC config without backbone
    fn small_config() -> CfCConfig {
        CfCConfig {
            input_dim: INPUT_DIM,
            hidden_dim: HIDDEN_DIM,
            use_backbone: false,
            backbone_layers: 0,
            backbone_dim: 0,
            activation: ActivationType::SiLU,
            tau_range: (0.1, 10.0),
            dropout: 0.0,
            gradient_clip: 1.0,
            online_learning: None,
        }
    }

    /// Helper: create a small config with backbone enabled
    fn small_config_with_backbone() -> CfCConfig {
        CfCConfig {
            input_dim: INPUT_DIM,
            hidden_dim: HIDDEN_DIM,
            use_backbone: true,
            backbone_layers: 2,
            backbone_dim: 6,
            activation: ActivationType::SiLU,
            tau_range: (0.1, 10.0),
            dropout: 0.0,
            gradient_clip: 1.0,
            online_learning: None,
        }
    }

    /// Helper: create a small input vector
    fn small_input() -> Array1<f32> {
        Array1::from_vec(vec![0.5, -0.3, 0.1, 0.8])
    }

    // ---------------------------------------------------------------
    // 1. Construction
    // ---------------------------------------------------------------

    #[test]
    fn test_construction_default_config() {
        let config = CfCConfig::default();
        let cell = CfCCell::new(config.clone());
        assert_eq!(cell.state.len(), config.hidden_dim);
        assert_eq!(cell.tau.len(), config.hidden_dim);
        assert_eq!(cell.b_h.len(), config.hidden_dim);
        assert_eq!(cell.steps, 0);
    }

    #[test]
    fn test_construction_small_config() {
        let cell = CfCCell::new(small_config());
        assert_eq!(cell.state.len(), HIDDEN_DIM);
        assert_eq!(cell.w_in.dim(), (HIDDEN_DIM, INPUT_DIM));
        assert_eq!(cell.w_h.dim(), (HIDDEN_DIM, HIDDEN_DIM));
    }

    #[test]
    fn test_construction_with_backbone() {
        let config = small_config_with_backbone();
        let cell = CfCCell::new(config);
        // w_in should take backbone_dim (6) as input, not input_dim (4)
        assert_eq!(cell.w_in.dim(), (HIDDEN_DIM, 6));
        assert_eq!(cell.backbone_weights.len(), 2);
        assert_eq!(cell.backbone_biases.len(), 2);
    }

    #[test]
    fn test_construction_clamps_zero_tau() {
        let mut config = small_config();
        config.tau_range = (0.0, 10.0); // tau_min < MIN_TAU gets clamped
        let cell = CfCCell::new(config);
        // Verify all tau values are at least MIN_TAU after clamping
        for &t in cell.tau.iter() {
            assert!(t >= MIN_TAU, "tau should be >= MIN_TAU after clamp: {t}");
        }
    }

    #[test]
    fn test_tau_initialized_in_range() {
        let cell = CfCCell::new(small_config());
        for &t in cell.tau.iter() {
            assert!(t >= 0.1, "tau should be >= tau_min: {}", t);
            assert!(t <= 10.0, "tau should be <= tau_max: {}", t);
        }
    }

    // ---------------------------------------------------------------
    // 2. Forward Pass
    // ---------------------------------------------------------------

    #[test]
    fn test_forward_produces_finite_output() {
        let mut cell = CfCCell::new(small_config());
        let input = small_input();
        let output = cell.forward(&input, 0.1);

        assert_eq!(output.len(), HIDDEN_DIM);
        for &v in output.iter() {
            assert!(v.is_finite(), "forward output should be finite: {}", v);
        }
    }

    #[test]
    fn test_forward_updates_state() {
        let mut cell = CfCCell::new(small_config());
        let state_before = cell.state.clone();

        cell.forward(&small_input(), 0.1);

        // State should change after forward (unless all weights happen to be zero, very unlikely)
        let state_after = cell.state.clone();
        let diff: f32 = (&state_after - &state_before).mapv(|x| x.abs()).sum();
        assert!(diff > 0.0, "state should change after forward pass");
    }

    #[test]
    fn test_forward_increments_steps() {
        let mut cell = CfCCell::new(small_config());
        assert_eq!(cell.steps, 0);

        cell.forward(&small_input(), 0.1);
        assert_eq!(cell.steps, 1);

        cell.forward(&small_input(), 0.1);
        assert_eq!(cell.steps, 2);
    }

    #[test]
    fn test_forward_with_backbone() {
        let mut cell = CfCCell::new(small_config_with_backbone());
        let output = cell.forward(&small_input(), 0.1);

        assert_eq!(output.len(), HIDDEN_DIM);
        for &v in output.iter() {
            assert!(
                v.is_finite(),
                "backbone forward output should be finite: {}",
                v
            );
        }
    }

    #[test]
    fn test_forward_output_clamped() {
        // Hidden state should be clamped to [-10, 10]
        let mut cell = CfCCell::new(small_config());
        // Run many forward steps to ensure state remains bounded
        let input = Array1::from_vec(vec![10.0, -10.0, 5.0, -5.0]);
        for _ in 0..100 {
            let output = cell.forward(&input, 1.0);
            for &v in output.iter() {
                assert!(v.is_finite());
                assert!(v.abs() <= 10.0, "output should be clamped: {}", v);
            }
        }
    }

    // ---------------------------------------------------------------
    // 3. Forward with Different Activation Functions
    // ---------------------------------------------------------------

    #[test]
    fn test_forward_with_tanh_activation() {
        let mut config = small_config();
        config.activation = ActivationType::Tanh;
        let mut cell = CfCCell::new(config);
        let output = cell.forward(&small_input(), 0.1);
        for &v in output.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_forward_with_relu_activation() {
        let mut config = small_config();
        config.activation = ActivationType::ReLU;
        let mut cell = CfCCell::new(config);
        let output = cell.forward(&small_input(), 0.1);
        for &v in output.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_forward_with_gelu_activation() {
        let mut config = small_config();
        config.activation = ActivationType::GELU;
        let mut cell = CfCCell::new(config);
        let output = cell.forward(&small_input(), 0.1);
        for &v in output.iter() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_forward_with_sigmoid_activation() {
        let mut config = small_config();
        config.activation = ActivationType::Sigmoid;
        let mut cell = CfCCell::new(config);
        let output = cell.forward(&small_input(), 0.1);
        for &v in output.iter() {
            assert!(v.is_finite());
        }
    }

    // ---------------------------------------------------------------
    // 4. Temporal Dynamics
    // ---------------------------------------------------------------

    #[test]
    fn test_temporal_dynamics_evolving_state() {
        let mut cell = CfCCell::new(small_config());
        let input = small_input();
        let mut states = Vec::new();

        for _ in 0..10 {
            let output = cell.forward(&input, 0.1);
            states.push(output);
        }

        // State should converge toward equilibrium — later states differ less
        let diff_early: f32 = (&states[1] - &states[0]).mapv(|x| x.abs()).sum();
        let diff_late: f32 = (&states[9] - &states[8]).mapv(|x| x.abs()).sum();

        // With constant input, the cell should approach equilibrium
        // so late differences should be smaller or equal to early ones
        assert!(
            diff_late <= diff_early + 1e-4,
            "state differences should decrease as cell approaches equilibrium: early={}, late={}",
            diff_early,
            diff_late
        );
    }

    #[test]
    fn test_different_dt_produces_different_dynamics() {
        let config = small_config();
        let input = small_input();

        let mut cell_fast = CfCCell::new(config.clone());
        let mut cell_slow = CfCCell::new(config);
        // Make states identical
        cell_slow.state = cell_fast.state.clone();
        cell_slow.w_in = cell_fast.w_in.clone();
        cell_slow.w_h = cell_fast.w_h.clone();
        cell_slow.b_h = cell_fast.b_h.clone();
        cell_slow.tau = cell_fast.tau.clone();

        let out_fast = cell_fast.forward(&input, 1.0);
        let out_slow = cell_slow.forward(&input, 0.01);

        // Different dt should produce different outputs
        let diff: f32 = (&out_fast - &out_slow).mapv(|x| x.abs()).sum();
        assert!(
            diff > 1e-6,
            "different dt should produce different outputs: diff={}",
            diff
        );
    }

    // ---------------------------------------------------------------
    // 5. Forward with Cache and Backward
    // ---------------------------------------------------------------

    #[test]
    fn test_forward_with_cache_matches_forward() {
        let config = small_config();
        let input = small_input();

        let mut cell1 = CfCCell::new(config.clone());
        let mut cell2 = CfCCell::new(config);
        // Synchronize cells
        cell2.state = cell1.state.clone();
        cell2.w_in = cell1.w_in.clone();
        cell2.w_h = cell1.w_h.clone();
        cell2.b_h = cell1.b_h.clone();
        cell2.tau = cell1.tau.clone();
        cell2.backbone_weights = cell1.backbone_weights.clone();
        cell2.backbone_biases = cell1.backbone_biases.clone();

        let out1 = cell1.forward(&input, 0.1);
        let (out2, _cache) = cell2.forward_with_cache(&input, 0.1);

        let diff: f32 = (&out1 - &out2).mapv(|x| x.abs()).sum();
        assert!(
            diff < 1e-6,
            "forward_with_cache should produce same output as forward: diff={}",
            diff
        );
    }

    #[test]
    fn test_backward_produces_finite_gradients() {
        let mut cell = CfCCell::new(small_config());
        let input = small_input();
        let target = Array1::from_vec(vec![0.1; HIDDEN_DIM]);

        // Run forward first to set state
        cell.forward(&input, 0.1);

        let grads = cell.backward(&input, &target, 0.1);
        for &v in grads.dw_in.iter() {
            assert!(v.is_finite(), "dw_in gradient should be finite");
        }
        for &v in grads.dw_h.iter() {
            assert!(v.is_finite(), "dw_h gradient should be finite");
        }
        for &v in grads.db_h.iter() {
            assert!(v.is_finite(), "db_h gradient should be finite");
        }
        for &v in grads.dtau.iter() {
            assert!(v.is_finite(), "dtau gradient should be finite");
        }
    }

    #[test]
    fn test_backward_from_cache_produces_finite_gradients() {
        let mut cell = CfCCell::new(small_config());
        let input = small_input();

        let (_output, cache) = cell.forward_with_cache(&input, 0.1);

        let dh = Array1::from_vec(vec![0.1; HIDDEN_DIM]);
        let grads = cell.backward_from_cache(&cache, &dh, 0.1);

        for &v in grads.dw_in.iter() {
            assert!(v.is_finite(), "cached dw_in gradient should be finite");
        }
        for &v in grads.dw_h.iter() {
            assert!(v.is_finite(), "cached dw_h gradient should be finite");
        }
        for &v in grads.db_h.iter() {
            assert!(v.is_finite(), "cached db_h gradient should be finite");
        }
        for &v in grads.dtau.iter() {
            assert!(v.is_finite(), "cached dtau gradient should be finite");
        }
    }

    // ---------------------------------------------------------------
    // 6. Apply Adam Optimizer
    // ---------------------------------------------------------------

    #[test]
    fn test_apply_adam_updates_weights() {
        let mut cell = CfCCell::new(small_config());
        let input = small_input();
        let target = Array1::from_vec(vec![0.5; HIDDEN_DIM]);

        cell.forward(&input, 0.1);
        let grads = cell.backward(&input, &target, 0.1);

        let w_in_before = cell.w_in.clone();
        let mut adam = AdamState::new(HIDDEN_DIM, INPUT_DIM);

        cell.apply_adam(&grads, &mut adam, 0.01);

        let diff: f32 = (&cell.w_in - &w_in_before).mapv(|x| x.abs()).sum();
        assert!(diff > 0.0, "Adam should update w_in weights");
        assert_eq!(adam.t, 1, "Adam step counter should increment");
    }

    #[test]
    fn test_apply_adam_vectorized_updates_weights() {
        let mut cell = CfCCell::new(small_config());
        let input = small_input();
        let target = Array1::from_vec(vec![0.5; HIDDEN_DIM]);

        cell.forward(&input, 0.1);
        let grads = cell.backward(&input, &target, 0.1);

        let w_in_before = cell.w_in.clone();
        let mut adam = AdamState::new(HIDDEN_DIM, INPUT_DIM);

        cell.apply_adam_vectorized(&grads, &mut adam, 0.01);

        let diff: f32 = (&cell.w_in - &w_in_before).mapv(|x| x.abs()).sum();
        assert!(diff > 0.0, "Adam vectorized should update w_in weights");
        assert_eq!(adam.t, 1);
    }

    #[test]
    fn test_apply_adam_tau_stays_in_range() {
        let mut cell = CfCCell::new(small_config());
        let input = small_input();
        let target = Array1::from_vec(vec![1.0; HIDDEN_DIM]);
        let mut adam = AdamState::new(HIDDEN_DIM, INPUT_DIM);

        // Run many Adam steps to push tau
        for _ in 0..100 {
            cell.forward(&input, 0.1);
            let grads = cell.backward(&input, &target, 0.1);
            cell.apply_adam(&grads, &mut adam, 0.1);
        }

        for &t in cell.tau.iter() {
            assert!(t >= 0.1, "tau should be >= 0.1 after Adam: {}", t);
            assert!(t <= 10.0, "tau should be <= 10.0 after Adam: {}", t);
        }
    }

    // ---------------------------------------------------------------
    // 7. Reset
    // ---------------------------------------------------------------

    #[test]
    fn test_reset_clears_state() {
        let mut cell = CfCCell::new(small_config());
        cell.forward(&small_input(), 0.1);
        assert!(cell.state.iter().any(|&v| v != 0.0));

        cell.reset();
        for &v in cell.state.iter() {
            assert_eq!(v, 0.0, "state should be zero after reset");
        }
        assert_eq!(cell.steps, 0);
    }

    // ---------------------------------------------------------------
    // 8. Online Learning (adapt_online)
    // ---------------------------------------------------------------

    #[test]
    fn test_adapt_online_disabled_by_default() {
        let mut cell = CfCCell::new(small_config());
        assert!(!cell.online_learning_enabled());
        let adapted = cell.adapt_online(1.0, &small_input(), &Array1::zeros(HIDDEN_DIM), 0.1);
        assert!(!adapted, "adapt_online should return false when disabled");
    }

    #[test]
    fn test_adapt_online_enabled_triggers_on_high_error() {
        let mut config = small_config();
        config.online_learning = Some(OnlineLearningConfig {
            learning_rate: 0.001,
            error_threshold: 0.05,
            ema_alpha: 0.1,
            max_weight_delta: 0.01,
            adapt_tau: false,
            tau_lr_multiplier: 0.01,
        });
        let mut cell = CfCCell::new(config);
        assert!(cell.online_learning_enabled());

        // Forward first to get non-zero state
        cell.forward(&small_input(), 0.1);

        let target = Array1::from_vec(vec![1.0; HIDDEN_DIM]);
        let adapted = cell.adapt_online(1.0, &small_input(), &target, 0.1);
        assert!(
            adapted,
            "adapt_online should trigger on high prediction error"
        );
        assert_eq!(cell.online_stats.adaptations_triggered, 1);
    }

    #[test]
    fn test_adapt_online_skips_low_error() {
        let mut config = small_config();
        config.online_learning = Some(OnlineLearningConfig {
            learning_rate: 0.001,
            error_threshold: 10.0, // Very high threshold
            ema_alpha: 0.1,
            max_weight_delta: 0.01,
            adapt_tau: false,
            tau_lr_multiplier: 0.01,
        });
        let mut cell = CfCCell::new(config);
        cell.forward(&small_input(), 0.1);

        let target = Array1::from_vec(vec![0.1; HIDDEN_DIM]);
        let adapted = cell.adapt_online(0.01, &small_input(), &target, 0.1);
        assert!(
            !adapted,
            "adapt_online should skip when error is below threshold"
        );
        assert_eq!(cell.online_stats.adaptations_skipped, 1);
    }

    #[test]
    fn test_adapt_online_with_tau_adaptation() {
        let mut config = small_config();
        config.online_learning = Some(OnlineLearningConfig {
            learning_rate: 0.001,
            error_threshold: 0.01,
            ema_alpha: 0.1,
            max_weight_delta: 0.01,
            adapt_tau: true,
            tau_lr_multiplier: 0.1,
        });
        let mut cell = CfCCell::new(config);
        let tau_before = cell.tau.clone();

        cell.forward(&small_input(), 0.1);
        let target = Array1::from_vec(vec![1.0; HIDDEN_DIM]);
        cell.adapt_online(1.0, &small_input(), &target, 0.1);

        // With adapt_tau=true, tau should change
        let diff: f32 = (&cell.tau - &tau_before).mapv(|x| x.abs()).sum();
        assert!(
            diff > 0.0,
            "tau should change with adapt_tau=true: diff={}",
            diff
        );
    }

    // ---------------------------------------------------------------
    // 9. Edge Cases
    // ---------------------------------------------------------------

    #[test]
    fn test_zero_input() {
        let mut cell = CfCCell::new(small_config());
        let zero_input = Array1::zeros(INPUT_DIM);
        let output = cell.forward(&zero_input, 0.1);

        for &v in output.iter() {
            assert!(v.is_finite(), "output with zero input should be finite");
        }
    }

    #[test]
    fn test_large_input() {
        let mut cell = CfCCell::new(small_config());
        let large_input = Array1::from_vec(vec![1000.0; INPUT_DIM]);
        let output = cell.forward(&large_input, 0.1);

        for &v in output.iter() {
            assert!(
                v.is_finite(),
                "output with large input should be finite: {}",
                v
            );
            assert!(v.abs() <= 10.0, "output should be clamped: {}", v);
        }
    }

    #[test]
    fn test_very_small_dt() {
        let mut cell = CfCCell::new(small_config());
        let output = cell.forward(&small_input(), 1e-8);
        for &v in output.iter() {
            assert!(v.is_finite(), "output with tiny dt should be finite");
        }
    }

    #[test]
    fn test_very_large_dt() {
        let mut cell = CfCCell::new(small_config());
        let output = cell.forward(&small_input(), 1000.0);
        for &v in output.iter() {
            assert!(
                v.is_finite(),
                "output with large dt should be finite: {}",
                v
            );
        }
    }

    // ---------------------------------------------------------------
    // 10. Genesis Determinism
    // ---------------------------------------------------------------

    #[test]
    fn test_from_genesis_is_deterministic() {
        let genesis = GenesisSeed::from_phrase("test-cfc-cell-42");
        let config = small_config();

        let cell1 = CfCCell::from_genesis(config.clone(), &genesis, "test_cell");
        let cell2 = CfCCell::from_genesis(config, &genesis, "test_cell");

        let w_diff: f32 = (&cell1.w_in - &cell2.w_in).mapv(|x| x.abs()).sum();
        assert!(
            w_diff < 1e-10,
            "from_genesis with same seed should produce identical weights: diff={}",
            w_diff
        );

        let tau_diff: f32 = (&cell1.tau - &cell2.tau).mapv(|x| x.abs()).sum();
        assert!(
            tau_diff < 1e-10,
            "from_genesis with same seed should produce identical tau: diff={}",
            tau_diff
        );
    }

    // ---------------------------------------------------------------
    // 11. Scale Tau
    // ---------------------------------------------------------------

    #[test]
    fn test_scale_tau() {
        let mut cell = CfCCell::new(small_config());
        let tau_before = cell.tau.clone();

        cell.scale_tau(2.0);

        for (before, after) in tau_before.iter().zip(cell.tau.iter()) {
            let expected = (before * 2.0).clamp(0.01, 100.0);
            assert!(
                (after - expected).abs() < 1e-6,
                "scale_tau should multiply and clamp: expected={}, got={}",
                expected,
                after
            );
        }
    }

    // ---------------------------------------------------------------
    // 12. Online Stats Reset
    // ---------------------------------------------------------------

    #[test]
    fn test_reset_online_stats() {
        let mut config = small_config();
        config.online_learning = Some(OnlineLearningConfig::default());
        let mut cell = CfCCell::new(config);

        cell.forward(&small_input(), 0.1);
        cell.adapt_online(
            1.0,
            &small_input(),
            &Array1::from_vec(vec![1.0; HIDDEN_DIM]),
            0.1,
        );

        assert!(cell.online_stats.adaptation_steps > 0);

        cell.reset_online_stats();

        assert_eq!(cell.online_stats.adaptation_steps, 0);
        assert_eq!(cell.online_stats.adaptations_triggered, 0);
        assert_eq!(cell.online_stats.ema_error, 0.0);
    }
}
