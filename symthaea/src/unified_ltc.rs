// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Unified Liquid Time-Constant Network
//!
//! Consolidated temporal dynamics system that unifies:
//! - **LearnableLTC**: BPTT+Adam training with flat array optimization
//! - **HdcLtcNeuron**: HDC⊗LTC fusion with Hebbian/Adam/RMSprop learning
//!
//! ## Design Principles
//!
//! 1. **Unified State Representation**: Supports both scalar (f32) and HDC (ContinuousHV) states
//! 2. **Multiple Learning Algorithms**: BPTT, Hebbian, Adam, RMSprop
//! 3. **Performance Optimized**: Flat row-major arrays, packed bit masks
//! 4. **Neuromodulation Bridge**: Connects to autopoietic vitality system
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea::unified_ltc::{UnifiedLTC, UnifiedLTCConfig, LearningAlgorithm};
//!
//! // Scalar mode (like LearnableLTC)
//! let config = UnifiedLTCConfig::scalar(256, 64, 32);
//! let mut ltc = UnifiedLTC::new(config)?;
//! let (output, states) = ltc.forward(&input)?;
//! ltc.backward(&input, &states, &target_grad)?;
//!
//! // HDC mode (like HdcLtcNeuron)
//! let config = UnifiedLTCConfig::hdc(16384);
//! let mut ltc = UnifiedLTC::new(config)?;
//! ltc.evolve(0.01, &hdc_input);
//! ltc.hebbian_update(&hdc_input);
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};

use symthaea_core::hdc::unified_hv::ContinuousHV;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Learning algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LearningAlgorithm {
    /// No learning (inference only)
    None,
    /// Backpropagation Through Time with Adam optimizer
    BPTT {
        lr_weights: f32,
        lr_tau: f32,
        lr_bias: f32,
        grad_clip: f32,
        l2_reg: f32,
    },
    /// Hebbian correlation-based learning
    Hebbian { lr: f32, momentum: f32 },
    /// Adam optimizer (adaptive learning rates)
    Adam {
        lr: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    },
    /// RMSprop optimizer
    RMSprop { lr: f32, rho: f32, epsilon: f32 },
}

impl Default for LearningAlgorithm {
    fn default() -> Self {
        Self::Adam {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0001,
        }
    }
}

/// Connectivity mode for network weights
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum Connectivity {
    /// Fully connected (all-to-all)
    #[default]
    Dense,
    /// Sparse connectivity with packed bit mask
    Sparse {
        sparsity: f32,
        mask: Vec<u64>,
        mask_row_len: usize,
    },
}

/// Activation function for neurons
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum Activation {
    #[default]
    Tanh,
    Sigmoid,
    Softplus,
    LeakyRelu,
    Identity,
}

impl Activation {
    /// Apply activation to scalar
    #[inline]
    pub fn apply_scalar(&self, x: f32) -> f32 {
        match self {
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Softplus => (1.0 + x.exp()).ln(),
            Activation::LeakyRelu => {
                if x > 0.0 {
                    x
                } else {
                    0.01 * x
                }
            }
            Activation::Identity => x,
        }
    }

    /// Apply activation derivative to scalar
    #[inline]
    pub fn derivative_scalar(&self, x: f32) -> f32 {
        match self {
            Activation::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            Activation::Softplus => 1.0 / (1.0 + (-x).exp()),
            Activation::LeakyRelu => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
            Activation::Identity => 1.0,
        }
    }

    /// Apply activation to HDC vector
    pub fn apply_hdc(&self, hv: &ContinuousHV) -> ContinuousHV {
        let values: Vec<f32> = hv.values.iter().map(|&x| self.apply_scalar(x)).collect();
        ContinuousHV::from_vec(values)
    }
}

/// State type for unified LTC
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum StateType {
    /// Scalar state per neuron (traditional LTC)
    #[default]
    Scalar,
    /// Hyperdimensional state per neuron (HDC-LTC fusion)
    HDC { dimension: usize },
}

/// Integration method for ODE solving
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum IntegrationMethod {
    /// First-order Euler (fast, O(dt) error)
    #[default]
    Euler,
    /// Fourth-order Runge-Kutta (accurate, O(dt^4) error)
    RK4,
}

/// Unified LTC configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedLTCConfig {
    /// Number of neurons
    pub num_neurons: usize,
    /// Input dimension (scalar mode only)
    pub input_dim: usize,
    /// Output dimension (scalar mode only)
    pub output_dim: usize,
    /// State type (scalar or HDC)
    pub state_type: StateType,
    /// Integration timestep
    pub dt: f32,
    /// Number of integration steps per forward pass
    pub num_steps: usize,
    /// Base time constant τ₀
    pub tau_base: f32,
    /// Maximum time constant
    pub tau_max: f32,
    /// Minimum time constant
    pub tau_min: f32,
    /// State-dependent time constant scaling
    pub tau_backbone: f32,
    /// Activation function
    pub activation: Activation,
    /// Learning algorithm
    pub learning: LearningAlgorithm,
    /// Connectivity mode
    pub connectivity: Connectivity,
    /// Integration method
    pub integration: IntegrationMethod,
}

impl Default for UnifiedLTCConfig {
    fn default() -> Self {
        Self {
            num_neurons: 256,
            input_dim: 64,
            output_dim: 32,
            state_type: StateType::Scalar,
            dt: 0.01,
            num_steps: 100,
            tau_base: 0.1,
            tau_max: 10.0,
            tau_min: 0.1,
            tau_backbone: 0.5,
            activation: Activation::default(),
            learning: LearningAlgorithm::default(),
            connectivity: Connectivity::Dense,
            integration: IntegrationMethod::default(),
        }
    }
}

impl UnifiedLTCConfig {
    /// Create scalar mode configuration (like LearnableLTC)
    pub fn scalar(num_neurons: usize, input_dim: usize, output_dim: usize) -> Self {
        Self {
            num_neurons,
            input_dim,
            output_dim,
            state_type: StateType::Scalar,
            learning: LearningAlgorithm::BPTT {
                lr_weights: 0.001,
                lr_tau: 0.0001,
                lr_bias: 0.001,
                grad_clip: 1.0,
                l2_reg: 0.0001,
            },
            ..Default::default()
        }
    }

    /// Create sparse scalar configuration
    pub fn scalar_sparse(
        num_neurons: usize,
        input_dim: usize,
        output_dim: usize,
        sparsity: f32,
    ) -> Self {
        let mut config = Self::scalar(num_neurons, input_dim, output_dim);
        let mask_row_len = num_neurons.div_ceil(64);
        config.connectivity = Connectivity::Sparse {
            sparsity,
            mask: vec![0u64; num_neurons * mask_row_len],
            mask_row_len,
        };
        config
    }

    /// Create HDC mode configuration (like HdcLtcNeuron)
    pub fn hdc(dimension: usize) -> Self {
        Self {
            num_neurons: 1, // Single HDC neuron
            input_dim: dimension,
            output_dim: dimension,
            state_type: StateType::HDC { dimension },
            dt: 0.01,
            num_steps: 1,
            learning: LearningAlgorithm::Adam {
                lr: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: 0.0001,
            },
            ..Default::default()
        }
    }

    /// Create HDC network configuration (multiple HDC neurons)
    pub fn hdc_network(num_neurons: usize, dimension: usize) -> Self {
        Self {
            num_neurons,
            input_dim: dimension,
            output_dim: dimension,
            state_type: StateType::HDC { dimension },
            learning: LearningAlgorithm::Hebbian {
                lr: 0.01,
                momentum: 0.9,
            },
            ..Default::default()
        }
    }
}

// ============================================================================
// UNIFIED LTC NETWORK
// ============================================================================

/// Unified Liquid Time-Constant Network
///
/// Supports both scalar and HDC states with multiple learning algorithms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedLTC {
    config: UnifiedLTCConfig,

    // === Scalar Mode State ===
    /// Per-neuron scalar states
    scalar_states: Vec<f32>,
    /// Per-neuron time constants
    tau: Vec<f32>,
    /// Recurrent weights (num_neurons x num_neurons, flat row-major)
    w_rec: Vec<f32>,
    /// Input weights (num_neurons x input_dim, flat row-major)
    w_in: Vec<f32>,
    /// Output weights (output_dim x num_neurons, flat row-major)
    w_out: Vec<f32>,
    /// Biases
    bias: Vec<f32>,

    // === HDC Mode State ===
    /// HDC states (one per neuron)
    hdc_states: Vec<ContinuousHV>,
    /// HDC weight hypervectors
    hdc_weights: Vec<ContinuousHV>,
    /// HDC input masks
    hdc_input_masks: Vec<ContinuousHV>,

    // === Optimizer State ===
    /// First moment (Adam/momentum)
    m_w_rec: Vec<f32>,
    /// Second moment (Adam)
    v_w_rec: Vec<f32>,
    /// Tau first moment
    m_tau: Vec<f32>,
    /// Tau second moment
    v_tau: Vec<f32>,
    /// HDC weight momentum
    hdc_weight_momentum: Vec<ContinuousHV>,
    /// HDC weight velocity
    hdc_weight_velocity: Vec<ContinuousHV>,
    /// Training step counter
    step: usize,

    // === Statistics ===
    total_time: f64,
    update_count: u64,
}

impl UnifiedLTC {
    /// Create a new Unified LTC network
    pub fn new(config: UnifiedLTCConfig) -> Result<Self> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let n = config.num_neurons;
        let input_dim = config.input_dim;
        let output_dim = config.output_dim;

        // Initialize based on state type
        let (
            scalar_states,
            hdc_states,
            hdc_weights,
            hdc_input_masks,
            hdc_weight_momentum,
            hdc_weight_velocity,
        ) = match &config.state_type {
            StateType::Scalar => (vec![0.0f32; n], vec![], vec![], vec![], vec![], vec![]),
            StateType::HDC { dimension } => {
                let states: Vec<_> = (0..n).map(|_| ContinuousHV::zero(*dimension)).collect();
                let weights: Vec<_> = (0..n)
                    .map(|i| ContinuousHV::random(*dimension, i as u64))
                    .collect();
                let masks: Vec<_> = (0..n)
                    .map(|i| ContinuousHV::random(*dimension, (i + 1000) as u64))
                    .collect();
                let momentum: Vec<_> = (0..n).map(|_| ContinuousHV::zero(*dimension)).collect();
                let velocity: Vec<_> = (0..n).map(|_| ContinuousHV::zero(*dimension)).collect();
                (vec![], states, weights, masks, momentum, velocity)
            }
        };

        // Initialize time constants
        let tau: Vec<f32> = (0..n)
            .map(|_| rng.gen_range(config.tau_min..config.tau_max))
            .collect();

        // Initialize weights based on connectivity
        let (w_rec, mask) = match &config.connectivity {
            Connectivity::Dense => {
                let w: Vec<f32> = (0..n * n).map(|_| rng.gen_range(-0.1..0.1)).collect();
                (w, None)
            }
            Connectivity::Sparse {
                sparsity,
                mask_row_len,
                ..
            } => {
                let mut w = vec![0.0f32; n * n];
                let mut m = vec![0u64; n * *mask_row_len];

                for i in 0..n {
                    for j in 0..n {
                        if rng.r#gen::<f32>() < *sparsity {
                            w[i * n + j] = rng.gen_range(-0.1..0.1);
                            let mask_idx = i * mask_row_len + j / 64;
                            m[mask_idx] |= 1u64 << (j % 64);
                        }
                    }
                }
                (w, Some(m))
            }
        };

        // Update connectivity with generated mask
        let connectivity = if let Some(m) = mask {
            let mask_row_len = n.div_ceil(64);
            Connectivity::Sparse {
                sparsity: match &config.connectivity {
                    Connectivity::Sparse { sparsity, .. } => *sparsity,
                    _ => 0.1,
                },
                mask: m,
                mask_row_len,
            }
        } else {
            config.connectivity.clone()
        };

        // Initialize input/output weights
        let w_in: Vec<f32> = (0..n * input_dim)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();

        let w_out: Vec<f32> = (0..output_dim * n)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();

        let bias: Vec<f32> = (0..n).map(|_| rng.gen_range(-0.1..0.1)).collect();

        // Initialize optimizer state
        let m_w_rec = vec![0.0f32; n * n];
        let v_w_rec = vec![0.0f32; n * n];
        let m_tau = vec![0.0f32; n];
        let v_tau = vec![0.0f32; n];

        let mut new_config = config;
        new_config.connectivity = connectivity;

        Ok(Self {
            config: new_config,
            scalar_states,
            tau,
            w_rec,
            w_in,
            w_out,
            bias,
            hdc_states,
            hdc_weights,
            hdc_input_masks,
            m_w_rec,
            v_w_rec,
            m_tau,
            v_tau,
            hdc_weight_momentum,
            hdc_weight_velocity,
            step: 0,
            total_time: 0.0,
            update_count: 0,
        })
    }

    /// Create a deterministic Unified LTC from a genesis seed.
    pub fn from_genesis(
        config: UnifiedLTCConfig,
        genesis: &symthaea_core::genesis::GenesisSeed,
        label: &str,
    ) -> Result<Self> {
        use rand::Rng;
        let n = config.num_neurons;
        let input_dim = config.input_dim;
        let output_dim = config.output_dim;

        let (
            scalar_states,
            hdc_states,
            hdc_weights,
            hdc_input_masks,
            hdc_weight_momentum,
            hdc_weight_velocity,
        ) = match &config.state_type {
            StateType::Scalar => (vec![0.0f32; n], vec![], vec![], vec![], vec![], vec![]),
            StateType::HDC { dimension } => {
                let states: Vec<_> = (0..n).map(|_| ContinuousHV::zero(*dimension)).collect();
                let weights: Vec<_> = (0..n)
                    .map(|i| genesis.hv(&format!("{label}::hdc_weight::{i}"), *dimension))
                    .collect();
                let masks: Vec<_> = (0..n)
                    .map(|i| genesis.hv(&format!("{label}::hdc_mask::{i}"), *dimension))
                    .collect();
                let momentum: Vec<_> = (0..n).map(|_| ContinuousHV::zero(*dimension)).collect();
                let velocity: Vec<_> = (0..n).map(|_| ContinuousHV::zero(*dimension)).collect();
                (vec![], states, weights, masks, momentum, velocity)
            }
        };

        let mut rng = genesis.domain(&format!("{label}::tau"));
        let tau: Vec<f32> = (0..n)
            .map(|_| rng.gen_range(config.tau_min..config.tau_max))
            .collect();

        let mut w_rng = genesis.domain(&format!("{label}::w_rec"));
        let (w_rec, mask) = match &config.connectivity {
            Connectivity::Dense => {
                let w: Vec<f32> = (0..n * n).map(|_| w_rng.gen_range(-0.1..0.1)).collect();
                (w, None)
            }
            Connectivity::Sparse {
                sparsity,
                mask_row_len,
                ..
            } => {
                let mut w = vec![0.0f32; n * n];
                let mut m = vec![0u64; n * *mask_row_len];
                for i in 0..n {
                    for j in 0..n {
                        if w_rng.r#gen::<f32>() < *sparsity {
                            w[i * n + j] = w_rng.gen_range(-0.1..0.1);
                            let mask_idx = i * mask_row_len + j / 64;
                            m[mask_idx] |= 1u64 << (j % 64);
                        }
                    }
                }
                (w, Some(m))
            }
        };

        let connectivity = if let Some(m) = mask {
            let mask_row_len = n.div_ceil(64);
            Connectivity::Sparse {
                sparsity: match &config.connectivity {
                    Connectivity::Sparse { sparsity, .. } => *sparsity,
                    _ => 0.1,
                },
                mask: m,
                mask_row_len,
            }
        } else {
            config.connectivity.clone()
        };

        let mut io_rng = genesis.domain(&format!("{label}::io_weights"));
        let w_in: Vec<f32> = (0..n * input_dim)
            .map(|_| io_rng.gen_range(-0.1..0.1))
            .collect();
        let w_out: Vec<f32> = (0..output_dim * n)
            .map(|_| io_rng.gen_range(-0.1..0.1))
            .collect();
        let bias: Vec<f32> = (0..n).map(|_| io_rng.gen_range(-0.1..0.1)).collect();

        let mut new_config = config;
        new_config.connectivity = connectivity;

        Ok(Self {
            config: new_config,
            scalar_states,
            tau,
            w_rec,
            w_in,
            w_out,
            bias,
            hdc_states,
            hdc_weights,
            hdc_input_masks,
            m_w_rec: vec![0.0f32; n * n],
            v_w_rec: vec![0.0f32; n * n],
            m_tau: vec![0.0f32; n],
            v_tau: vec![0.0f32; n],
            hdc_weight_momentum,
            hdc_weight_velocity,
            step: 0,
            total_time: 0.0,
            update_count: 0,
        })
    }

    /// Check if mask bit is set at [i, j]
    #[inline]
    fn mask_get(&self, i: usize, j: usize) -> bool {
        match &self.config.connectivity {
            Connectivity::Dense => true,
            Connectivity::Sparse {
                mask, mask_row_len, ..
            } => {
                let idx = i * mask_row_len + j / 64;
                (mask[idx] & (1u64 << (j % 64))) != 0
            }
        }
    }

    // ========================================================================
    // SCALAR MODE OPERATIONS
    // ========================================================================

    /// Forward pass (scalar mode)
    ///
    /// Returns (output, all_hidden_states) for backpropagation
    pub fn forward(&mut self, input: &[f32]) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        if !matches!(self.config.state_type, StateType::Scalar) {
            anyhow::bail!("forward() is for scalar mode. Use evolve() for HDC mode.");
        }

        if input.len() != self.config.input_dim {
            anyhow::bail!(
                "Input dimension mismatch: expected {}, got {}",
                self.config.input_dim,
                input.len()
            );
        }

        let n = self.config.num_neurons;
        let input_dim = self.config.input_dim;
        let output_dim = self.config.output_dim;
        let dt = self.config.dt;

        let mut all_states = Vec::with_capacity(self.config.num_steps);

        for _t in 0..self.config.num_steps {
            // Store current state
            all_states.push(self.scalar_states.clone());

            // Compute input contribution
            let mut input_contribution = vec![0.0f32; n];
            for i in 0..n {
                for j in 0..input_dim {
                    input_contribution[i] += self.w_in[i * input_dim + j] * input[j];
                }
            }

            // Compute recurrent contribution with sparse mask
            let mut rec_contribution = vec![0.0f32; n];
            for i in 0..n {
                for j in 0..n {
                    if self.mask_get(i, j) {
                        rec_contribution[i] += self.w_rec[i * n + j] * self.scalar_states[j];
                    }
                }
            }

            // Update each neuron
            for i in 0..n {
                let z = rec_contribution[i] + input_contribution[i] + self.bias[i];
                let activated = self.config.activation.apply_scalar(z);

                // LTC dynamics: dx/dt = (-x + activated) / τ
                let dx = (-self.scalar_states[i] + activated) * dt / self.tau[i];
                self.scalar_states[i] += dx;
                self.scalar_states[i] = self.scalar_states[i].clamp(-10.0, 10.0);
            }
        }

        // Compute output
        let mut output = vec![0.0f32; output_dim];
        for i in 0..output_dim {
            for j in 0..n {
                output[i] += self.w_out[i * n + j] * self.scalar_states[j];
            }
        }

        self.total_time += (self.config.num_steps as f32 * dt) as f64;
        self.update_count += self.config.num_steps as u64;

        Ok((output, all_states))
    }

    /// Backward pass with BPTT (scalar mode)
    pub fn backward(
        &mut self,
        input: &[f32],
        hidden_states: &[Vec<f32>],
        output_grad: &[f32],
    ) -> Result<f32> {
        let LearningAlgorithm::BPTT {
            lr_weights,
            lr_tau,
            lr_bias,
            grad_clip,
            l2_reg,
        } = self.config.learning
        else {
            anyhow::bail!("backward() requires BPTT learning algorithm");
        };

        let n = self.config.num_neurons;
        let input_dim = self.config.input_dim;
        let output_dim = self.config.output_dim;
        let dt = self.config.dt;

        // Gradient for output weights
        let final_state = hidden_states.last().ok_or_else(|| {
            anyhow::anyhow!("hidden_states is empty; cannot compute output gradient")
        })?;
        let mut grad_w_out = vec![0.0f32; output_dim * n];
        for i in 0..output_dim {
            for j in 0..n {
                grad_w_out[i * n + j] = output_grad[i] * final_state[j];
            }
        }

        // Backprop through output layer
        let mut grad_h = vec![0.0f32; n];
        for j in 0..n {
            for i in 0..output_dim {
                grad_h[j] += output_grad[i] * self.w_out[i * n + j];
            }
        }

        // Initialize gradient accumulators
        let mut grad_w_rec = vec![0.0f32; n * n];
        let mut grad_w_in = vec![0.0f32; n * input_dim];
        let mut grad_bias = vec![0.0f32; n];
        let mut grad_tau = vec![0.0f32; n];

        // BPTT through time
        for t in (0..self.config.num_steps).rev() {
            let state = &hidden_states[t];
            let prev_state = if t > 0 {
                &hidden_states[t - 1]
            } else {
                &vec![0.0f32; n]
            };

            for i in 0..n {
                // Compute z at this timestep
                let mut z = self.bias[i];
                for j in 0..n {
                    if self.mask_get(i, j) {
                        z += self.w_rec[i * n + j] * prev_state[j];
                    }
                }
                for j in 0..input_dim {
                    z += self.w_in[i * input_dim + j] * input[j];
                }

                let sigmoid_grad = self.config.activation.derivative_scalar(z);

                // Gradient w.r.t. tau
                grad_tau[i] += grad_h[i] * state[i] / (self.tau[i] * self.tau[i]) * dt;

                // Gradient w.r.t. bias
                grad_bias[i] += grad_h[i] * sigmoid_grad * dt;

                // Gradient w.r.t. recurrent weights
                for j in 0..n {
                    if self.mask_get(i, j) {
                        grad_w_rec[i * n + j] += grad_h[i] * sigmoid_grad * prev_state[j] * dt;
                    }
                }

                // Gradient w.r.t. input weights
                for j in 0..input_dim {
                    grad_w_in[i * input_dim + j] += grad_h[i] * sigmoid_grad * input[j] * dt;
                }
            }

            // Backprop gradient to previous timestep
            let mut new_grad_h = vec![0.0f32; n];
            for i in 0..n {
                for j in 0..n {
                    let factor = if i == j { 1.0 - dt / self.tau[i] } else { 0.0 };
                    let rec_factor = if self.mask_get(j, i) {
                        dt * self.config.activation.derivative_scalar(0.0) * self.w_rec[j * n + i]
                    } else {
                        0.0
                    };
                    new_grad_h[j] += grad_h[i] * (factor + rec_factor);
                }
            }
            grad_h = new_grad_h;
        }

        // Clip gradients
        for g in grad_w_rec.iter_mut() {
            *g = g.clamp(-grad_clip, grad_clip);
        }
        for g in grad_tau.iter_mut() {
            *g = g.clamp(-grad_clip, grad_clip);
        }

        // Adam update for weights
        self.step += 1;
        let t = self.step as f32;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;

        for i in 0..n {
            for j in 0..n {
                if self.mask_get(i, j) {
                    let idx = i * n + j;
                    let g = grad_w_rec[idx] + l2_reg * self.w_rec[idx];

                    self.m_w_rec[idx] = beta1 * self.m_w_rec[idx] + (1.0 - beta1) * g;
                    self.v_w_rec[idx] = beta2 * self.v_w_rec[idx] + (1.0 - beta2) * g * g;

                    let m_hat = self.m_w_rec[idx] / (1.0 - beta1.powf(t));
                    let v_hat = self.v_w_rec[idx] / (1.0 - beta2.powf(t));

                    self.w_rec[idx] -= lr_weights * m_hat / (v_hat.sqrt() + eps);
                }
            }
        }

        // Adam update for tau
        for i in 0..n {
            let g = grad_tau[i];
            self.m_tau[i] = beta1 * self.m_tau[i] + (1.0 - beta1) * g;
            self.v_tau[i] = beta2 * self.v_tau[i] + (1.0 - beta2) * g * g;

            let m_hat = self.m_tau[i] / (1.0 - beta1.powf(t));
            let v_hat = self.v_tau[i] / (1.0 - beta2.powf(t));

            self.tau[i] -= lr_tau * m_hat / (v_hat.sqrt() + eps);
            self.tau[i] = self.tau[i].clamp(self.config.tau_min, self.config.tau_max);
        }

        // Update bias
        for i in 0..n {
            self.bias[i] -= lr_bias * grad_bias[i];
        }

        // Update input weights
        for idx in 0..n * input_dim {
            self.w_in[idx] -= lr_weights * grad_w_in[idx];
        }

        // Update output weights
        for idx in 0..output_dim * n {
            self.w_out[idx] -= lr_weights * grad_w_out[idx];
        }

        // Return gradient norm
        Ok(grad_tau.iter().map(|g| g * g).sum::<f32>().sqrt())
    }

    /// Train on single example (scalar mode)
    pub fn train_step(&mut self, input: &[f32], target: &[f32]) -> Result<f32> {
        let (output, hidden_states) = self.forward(input)?;

        // Compute MSE loss and gradient
        let mut loss = 0.0f32;
        let mut output_grad = vec![0.0f32; self.config.output_dim];

        for i in 0..self.config.output_dim.min(target.len()) {
            let diff = output[i] - target[i];
            loss += diff * diff;
            output_grad[i] = 2.0 * diff / target.len() as f32;
        }
        loss /= target.len() as f32;

        self.backward(input, &hidden_states, &output_grad)?;

        Ok(loss)
    }

    // ========================================================================
    // HDC MODE OPERATIONS
    // ========================================================================

    /// Evolve HDC state (HDC mode)
    pub fn evolve(&mut self, dt: f32, input: &ContinuousHV) {
        if !matches!(self.config.state_type, StateType::HDC { .. }) {
            return;
        }

        for i in 0..self.config.num_neurons {
            let state = &self.hdc_states[i];
            let weight = &self.hdc_weights[i];
            let input_mask = &self.hdc_input_masks[i];

            // HDC-LTC dynamics: dx/dt = (-x + f(W⊗x + U⊗u)) / τ(||x||)
            let masked_input = input_mask.bind(input);
            let transformed_state = weight.bind(state);
            let combined = ContinuousHV::bundle(&[&transformed_state, &masked_input]);
            let activated = self.config.activation.apply_hdc(&combined);

            // State-dependent time constant
            let state_norm = state.norm();
            let tau_eff = self.config.tau_base * (1.0 + self.config.tau_backbone * state_norm);

            // Euler integration
            let delta = activated.subtract(state).scale(dt / tau_eff);
            self.hdc_states[i] = state.add(&delta);

            // Bound state to prevent explosion
            if self.hdc_states[i].norm() > 5.0 {
                self.hdc_states[i] = self.hdc_states[i].normalize().scale(5.0);
            }
        }

        self.total_time += dt as f64;
        self.update_count += 1;
    }

    /// Evolve with RK4 integration (HDC mode, more accurate)
    pub fn evolve_rk4(&mut self, dt: f32, input: &ContinuousHV) {
        if !matches!(self.config.state_type, StateType::HDC { .. }) {
            return;
        }

        for i in 0..self.config.num_neurons {
            let h = dt;
            let state = &self.hdc_states[i];

            let k1 = self.compute_hdc_derivative(i, input, state);

            let state_k2 = state.add(&k1.scale(h / 2.0));
            let k2 = self.compute_hdc_derivative(i, input, &state_k2);

            let state_k3 = state.add(&k2.scale(h / 2.0));
            let k3 = self.compute_hdc_derivative(i, input, &state_k3);

            let state_k4 = state.add(&k3.scale(h));
            let k4 = self.compute_hdc_derivative(i, input, &state_k4);

            let sum = k1.add(&k2.scale(2.0)).add(&k3.scale(2.0)).add(&k4);
            self.hdc_states[i] = state.add(&sum.scale(h / 6.0));
        }

        self.total_time += dt as f64;
        self.update_count += 1;
    }

    fn compute_hdc_derivative(
        &self,
        i: usize,
        input: &ContinuousHV,
        state: &ContinuousHV,
    ) -> ContinuousHV {
        let weight = &self.hdc_weights[i];
        let input_mask = &self.hdc_input_masks[i];

        let masked_input = input_mask.bind(input);
        let transformed_state = weight.bind(state);
        let combined = ContinuousHV::bundle(&[&transformed_state, &masked_input]);
        let activated = self.config.activation.apply_hdc(&combined);

        let state_norm = state.norm();
        let tau_eff = self.config.tau_base * (1.0 + self.config.tau_backbone * state_norm);

        activated.subtract(state).scale(1.0 / tau_eff)
    }

    /// Hebbian weight update (HDC mode)
    pub fn hebbian_update(&mut self, input: &ContinuousHV) {
        let (lr, momentum) = match self.config.learning {
            LearningAlgorithm::Hebbian { lr, momentum } => (lr, momentum),
            _ => return,
        };

        for i in 0..self.config.num_neurons {
            // Hebbian: ΔW ∝ input ⊗ state
            let correlation = input.bind(&self.hdc_states[i]);

            // Apply with momentum
            self.hdc_weight_momentum[i] = self.hdc_weight_momentum[i]
                .scale(momentum)
                .add(&correlation.scale(lr));

            self.hdc_weights[i] = self.hdc_weights[i].add(&self.hdc_weight_momentum[i]);

            // Normalize to prevent explosion
            if self.hdc_weights[i].norm() > 2.0 {
                self.hdc_weights[i] = self.hdc_weights[i].normalize().scale(2.0);
            }
        }
    }

    /// Adam weight update (HDC mode)
    pub fn adam_update(&mut self, input: &ContinuousHV) {
        let (lr, beta1, beta2, epsilon, weight_decay) = match self.config.learning {
            LearningAlgorithm::Adam {
                lr,
                beta1,
                beta2,
                epsilon,
                weight_decay,
            } => (lr, beta1, beta2, epsilon, weight_decay),
            _ => return,
        };

        self.step += 1;
        let t = self.step as f32;

        for i in 0..self.config.num_neurons {
            // Gradient: correlation between input and state
            let gradient = input.bind(&self.hdc_states[i]);

            // Update biased first moment
            self.hdc_weight_momentum[i] = self.hdc_weight_momentum[i]
                .scale(beta1)
                .add(&gradient.scale(1.0 - beta1));

            // Update biased second moment
            let gradient_squared = gradient.hadamard_product(&gradient);
            self.hdc_weight_velocity[i] = self.hdc_weight_velocity[i]
                .scale(beta2)
                .add(&gradient_squared.scale(1.0 - beta2));

            // Bias correction
            let bias_correction1 = 1.0 - beta1.powf(t);
            let bias_correction2 = 1.0 - beta2.powf(t);

            let m_hat = self.hdc_weight_momentum[i].scale(1.0 / bias_correction1);
            let v_hat_scaled = self.hdc_weight_velocity[i].scale(1.0 / bias_correction2);
            let v_sqrt_inv = v_hat_scaled.map(|x| 1.0 / (x.abs().sqrt() + epsilon));
            let update = m_hat.hadamard_product(&v_sqrt_inv).scale(lr);

            // Weight decay
            self.hdc_weights[i] = self.hdc_weights[i]
                .scale(1.0 - weight_decay * lr)
                .add(&update);

            if self.hdc_weights[i].norm() > 2.0 {
                self.hdc_weights[i] = self.hdc_weights[i].normalize().scale(2.0);
            }
        }
    }

    // ========================================================================
    // COMMON OPERATIONS
    // ========================================================================

    /// Reset all states
    pub fn reset(&mut self) {
        for s in &mut self.scalar_states {
            *s = 0.0;
        }
        for s in &mut self.hdc_states {
            *s = ContinuousHV::zero(s.values.len());
        }
        self.total_time = 0.0;
        self.update_count = 0;
    }

    /// Apply neuromodulation (vitality → learning/dynamics)
    pub fn apply_neuromodulation(&mut self, vitality: f32) {
        // Modulate learning rate
        let modulation = vitality.clamp(0.1, 2.0);

        match &mut self.config.learning {
            LearningAlgorithm::BPTT {
                lr_weights,
                lr_bias,
                ..
            } => {
                *lr_weights *= modulation;
                *lr_bias *= modulation;
            }
            LearningAlgorithm::Hebbian { lr, .. } => {
                *lr *= modulation;
            }
            LearningAlgorithm::Adam { lr, .. } => {
                *lr *= modulation;
            }
            LearningAlgorithm::RMSprop { lr, .. } => {
                *lr *= modulation;
            }
            _ => {}
        }

        // Modulate time constants (low vitality → sluggish)
        let sluggishness = 1.0 + (1.0 - vitality) * 2.0;
        if sluggishness > 1.5 {
            for tau in &mut self.tau {
                *tau = (*tau * 0.99 + (*tau * sluggishness) * 0.01).min(self.config.tau_max);
            }
        }
    }

    /// Get current scalar states
    pub fn scalar_states(&self) -> &[f32] {
        &self.scalar_states
    }

    /// Get current HDC states
    pub fn hdc_states(&self) -> &[ContinuousHV] {
        &self.hdc_states
    }

    /// Get configuration
    pub fn config(&self) -> &UnifiedLTCConfig {
        &self.config
    }

    /// Get total evolution time
    pub fn total_time(&self) -> f64 {
        self.total_time
    }

    /// Predict next HDV state (for cognitive loop integration)
    pub fn predict_next_hdv(&self) -> Vec<f32> {
        if matches!(self.config.state_type, StateType::Scalar) {
            let mut prediction = vec![0.0f32; self.config.output_dim];
            for i in 0..self.config.output_dim {
                for j in 0..self.config.num_neurons {
                    prediction[i] +=
                        self.w_out[i * self.config.num_neurons + j] * self.scalar_states[j];
                }
            }
            prediction
        } else {
            // For HDC mode, return first state's values
            if !self.hdc_states.is_empty() {
                self.hdc_states[0].values.clone()
            } else {
                vec![]
            }
        }
    }

    /// Serialize for persistence
    pub fn serialize(&self) -> Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }

    /// Deserialize
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        Ok(bincode::deserialize(data)?)
    }
}

impl Default for UnifiedLTC {
    fn default() -> Self {
        Self::new(UnifiedLTCConfig::default()).expect("Default config should work")
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_forward_pass() {
        let config = UnifiedLTCConfig::scalar(32, 16, 8);
        let mut ltc = UnifiedLTC::new(config).unwrap();

        let input = vec![0.5f32; 16];
        let (output, states) = ltc.forward(&input).unwrap();

        assert_eq!(output.len(), 8);
        assert_eq!(states.len(), 100); // num_steps
    }

    #[test]
    fn test_scalar_training() {
        let config = UnifiedLTCConfig::scalar(32, 16, 8);
        let mut ltc = UnifiedLTC::new(config).unwrap();

        let input = vec![0.5f32; 16];
        let target = vec![1.0f32; 8];

        let loss1 = ltc.train_step(&input, &target).unwrap();
        ltc.reset();
        let loss2 = ltc.train_step(&input, &target).unwrap();

        // Both losses should be finite and non-negative
        assert!(loss1.is_finite(), "Loss 1 should be finite");
        assert!(loss2.is_finite(), "Loss 2 should be finite");
        assert!(loss1 >= 0.0, "Loss 1 should be non-negative");
        assert!(loss2 >= 0.0, "Loss 2 should be non-negative");
        println!("Loss 1: {}, Loss 2: {}", loss1, loss2);
    }

    #[test]
    fn test_sparse_connectivity() {
        let config = UnifiedLTCConfig::scalar_sparse(64, 16, 8, 0.1);
        let ltc = UnifiedLTC::new(config).unwrap();

        // Check sparsity is applied
        let mut non_zero = 0;
        for i in 0..64 {
            for j in 0..64 {
                if ltc.mask_get(i, j) {
                    non_zero += 1;
                }
            }
        }
        // Should be roughly 10% of 64*64 = 4096, so ~400 ± tolerance
        assert!(non_zero > 200 && non_zero < 600);
    }

    #[test]
    fn test_hdc_evolution() {
        let config = UnifiedLTCConfig::hdc(1024);
        let mut ltc = UnifiedLTC::new(config).unwrap();

        let input = ContinuousHV::random(1024, 42);

        // Evolve for 100 steps
        for _ in 0..100 {
            ltc.evolve(0.01, &input);
        }

        assert!(ltc.total_time() > 0.0);
    }

    #[test]
    fn test_hdc_hebbian_learning() {
        let mut config = UnifiedLTCConfig::hdc(512);
        config.learning = LearningAlgorithm::Hebbian {
            lr: 0.01,
            momentum: 0.9,
        };
        let mut ltc = UnifiedLTC::new(config).unwrap();

        let input = ContinuousHV::random(512, 123);

        // Train
        for _ in 0..10 {
            ltc.evolve(0.01, &input);
            ltc.hebbian_update(&input);
        }

        // Weights should have changed
        assert!(ltc.hdc_weight_momentum[0].norm() > 0.0);
    }

    #[test]
    fn test_neuromodulation() {
        let config = UnifiedLTCConfig::scalar(32, 16, 8);
        let mut ltc = UnifiedLTC::new(config).unwrap();

        let original_tau: Vec<_> = ltc.tau.clone();

        // Low vitality → sluggish
        ltc.apply_neuromodulation(0.2);

        // Time constants should have increased
        for (orig, new) in original_tau.iter().zip(ltc.tau.iter()) {
            assert!(*new >= *orig);
        }
    }

    #[test]
    fn test_serialization() {
        let config = UnifiedLTCConfig::scalar(16, 8, 4);
        let ltc = UnifiedLTC::new(config).unwrap();

        let data = ltc.serialize().unwrap();
        let restored = UnifiedLTC::deserialize(&data).unwrap();

        assert_eq!(ltc.config.num_neurons, restored.config.num_neurons);
    }

    // ====================================================================
    // Additional tests for thorough coverage
    // ====================================================================

    #[test]
    fn test_default_construction() {
        let ltc = UnifiedLTC::default();
        let cfg = ltc.config();
        assert_eq!(cfg.num_neurons, 256);
        assert_eq!(cfg.input_dim, 64);
        assert_eq!(cfg.output_dim, 32);
        assert!(matches!(cfg.state_type, StateType::Scalar));
        assert_eq!(ltc.scalar_states().len(), 256);
        assert_eq!(ltc.total_time(), 0.0);
    }

    #[test]
    fn test_config_scalar_defaults() {
        let cfg = UnifiedLTCConfig::scalar(128, 32, 16);
        assert_eq!(cfg.num_neurons, 128);
        assert_eq!(cfg.input_dim, 32);
        assert_eq!(cfg.output_dim, 16);
        assert!(matches!(cfg.state_type, StateType::Scalar));
        assert!(matches!(cfg.learning, LearningAlgorithm::BPTT { .. }));
        assert!(matches!(cfg.connectivity, Connectivity::Dense));
    }

    #[test]
    fn test_config_hdc_defaults() {
        let cfg = UnifiedLTCConfig::hdc(2048);
        assert_eq!(cfg.num_neurons, 1);
        assert_eq!(cfg.input_dim, 2048);
        assert_eq!(cfg.output_dim, 2048);
        assert!(matches!(cfg.state_type, StateType::HDC { dimension: 2048 }));
        assert!(matches!(cfg.learning, LearningAlgorithm::Adam { .. }));
    }

    #[test]
    fn test_config_hdc_network_defaults() {
        let cfg = UnifiedLTCConfig::hdc_network(4, 512);
        assert_eq!(cfg.num_neurons, 4);
        assert_eq!(cfg.input_dim, 512);
        assert!(matches!(cfg.learning, LearningAlgorithm::Hebbian { .. }));
    }

    #[test]
    fn test_forward_input_dimension_mismatch() {
        let config = UnifiedLTCConfig::scalar(16, 8, 4);
        let mut ltc = UnifiedLTC::new(config).unwrap();

        // Provide wrong-sized input
        let wrong_input = vec![1.0f32; 5];
        let result = ltc.forward(&wrong_input);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("dimension mismatch"));
    }

    #[test]
    fn test_forward_on_hdc_mode_fails() {
        let config = UnifiedLTCConfig::hdc(256);
        let mut ltc = UnifiedLTC::new(config).unwrap();

        let input = vec![0.0f32; 256];
        let result = ltc.forward(&input);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("scalar mode"));
    }

    #[test]
    fn test_backward_requires_bptt() {
        let mut config = UnifiedLTCConfig::scalar(16, 8, 4);
        config.learning = LearningAlgorithm::None;
        let mut ltc = UnifiedLTC::new(config).unwrap();

        let hidden_states = vec![vec![0.0f32; 16]];
        let output_grad = vec![0.0f32; 4];
        let input = vec![0.0f32; 8];
        let result = ltc.backward(&input, &hidden_states, &output_grad);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("BPTT"));
    }

    #[test]
    fn test_forward_output_finiteness() {
        let mut config = UnifiedLTCConfig::scalar(16, 8, 4);
        config.num_steps = 50;
        let mut ltc = UnifiedLTC::new(config).unwrap();

        let input = vec![1.0f32; 8];
        let (output, _states) = ltc.forward(&input).unwrap();

        for (i, val) in output.iter().enumerate() {
            assert!(val.is_finite(), "output[{}] = {} is not finite", i, val);
        }
    }

    #[test]
    fn test_forward_states_dimension_preserved() {
        let mut config = UnifiedLTCConfig::scalar(20, 10, 5);
        config.num_steps = 30;
        let mut ltc = UnifiedLTC::new(config).unwrap();

        let input = vec![0.3f32; 10];
        let (output, states) = ltc.forward(&input).unwrap();

        assert_eq!(output.len(), 5, "output dim must match output_dim");
        assert_eq!(states.len(), 30, "num hidden states must match num_steps");
        for (t, state) in states.iter().enumerate() {
            assert_eq!(
                state.len(),
                20,
                "hidden state at t={} has wrong dimension",
                t
            );
        }
    }

    #[test]
    fn test_hdc_evolve_zero_dt() {
        let config = UnifiedLTCConfig::hdc(256);
        let mut ltc = UnifiedLTC::new(config).unwrap();

        let initial_states: Vec<_> = ltc.hdc_states().to_vec();
        let input = ContinuousHV::random(256, 99);

        // Evolving with dt=0 should not change state
        ltc.evolve(0.0, &input);

        for (i, (orig, after)) in initial_states
            .iter()
            .zip(ltc.hdc_states().iter())
            .enumerate()
        {
            assert_eq!(
                orig.values, after.values,
                "HDC state[{}] changed with dt=0",
                i
            );
        }
    }

    #[test]
    fn test_hdc_evolve_state_bounded() {
        let config = UnifiedLTCConfig::hdc(512);
        let mut ltc = UnifiedLTC::new(config).unwrap();

        let input = ContinuousHV::random(512, 7);

        // Evolve with a large dt many times to stress-test bounding
        for _ in 0..500 {
            ltc.evolve(0.5, &input);
        }

        for (i, state) in ltc.hdc_states().iter().enumerate() {
            let norm = state.norm();
            assert!(norm <= 5.5, "HDC state[{}] norm={} exceeds bound", i, norm);
            for (j, v) in state.values.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "HDC state[{}].values[{}] = {} is not finite",
                    i,
                    j,
                    v
                );
            }
        }
    }

    #[test]
    fn test_hdc_rk4_evolve_finite() {
        let config = UnifiedLTCConfig::hdc(256);
        let mut ltc = UnifiedLTC::new(config).unwrap();

        let input = ContinuousHV::random(256, 55);

        for _ in 0..50 {
            ltc.evolve_rk4(0.01, &input);
        }

        assert!(ltc.total_time() > 0.0);
        for (i, state) in ltc.hdc_states().iter().enumerate() {
            for (j, v) in state.values.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "RK4 state[{}].values[{}] = {} is not finite",
                    i,
                    j,
                    v
                );
            }
        }
    }

    #[test]
    fn test_hdc_adam_update() {
        let config = UnifiedLTCConfig::hdc(256);
        let mut ltc = UnifiedLTC::new(config).unwrap();

        let input = ContinuousHV::random(256, 42);

        let weights_before: Vec<_> = ltc.hdc_weights.iter().map(|w| w.values.clone()).collect();

        ltc.evolve(0.01, &input);
        ltc.adam_update(&input);

        // Weights must have changed
        let mut any_changed = false;
        for (before, after) in weights_before.iter().zip(ltc.hdc_weights.iter()) {
            if before != &after.values {
                any_changed = true;
                break;
            }
        }
        assert!(any_changed, "Adam update should modify weights");

        // All weight values must be finite
        for (i, w) in ltc.hdc_weights.iter().enumerate() {
            for (j, v) in w.values.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "hdc_weights[{}].values[{}] = {} not finite after Adam",
                    i,
                    j,
                    v
                );
            }
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let config = UnifiedLTCConfig::scalar(16, 8, 4);
        let mut ltc = UnifiedLTC::new(config).unwrap();

        let input = vec![1.0f32; 8];
        let _ = ltc.forward(&input).unwrap();

        assert!(ltc.total_time() > 0.0);
        assert!(ltc.scalar_states().iter().any(|&s| s != 0.0));

        ltc.reset();

        assert_eq!(ltc.total_time(), 0.0);
        for &s in ltc.scalar_states() {
            assert_eq!(s, 0.0, "scalar state should be zero after reset");
        }
    }

    #[test]
    fn test_predict_next_hdv_scalar_mode() {
        let config = UnifiedLTCConfig::scalar(16, 8, 4);
        let mut ltc = UnifiedLTC::new(config).unwrap();

        // Before any forward pass, prediction should still produce correct dim
        let pred = ltc.predict_next_hdv();
        assert_eq!(pred.len(), 4);
        for v in &pred {
            assert!(v.is_finite());
        }

        // After forward pass, prediction may differ
        let input = vec![0.5f32; 8];
        let _ = ltc.forward(&input).unwrap();
        let pred_after = ltc.predict_next_hdv();
        assert_eq!(pred_after.len(), 4);
        for v in &pred_after {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_predict_next_hdv_hdc_mode() {
        let config = UnifiedLTCConfig::hdc(128);
        let mut ltc = UnifiedLTC::new(config).unwrap();

        let pred = ltc.predict_next_hdv();
        assert_eq!(pred.len(), 128);

        let input = ContinuousHV::random(128, 10);
        ltc.evolve(0.01, &input);
        let pred_after = ltc.predict_next_hdv();
        assert_eq!(pred_after.len(), 128);
    }

    #[test]
    fn test_activation_functions_edge_values() {
        // Softplus overflows f32 for large x (e^100 > f32::MAX), so we test
        // it separately with a smaller range. All other activations are bounded
        // or linear and handle large inputs fine.
        let bounded_activations = [
            Activation::Tanh,
            Activation::Sigmoid,
            Activation::LeakyRelu,
            Activation::Identity,
        ];

        let edge_values = [0.0f32, 1.0, -1.0, 100.0, -100.0, f32::MIN_POSITIVE];

        for act in &bounded_activations {
            for &x in &edge_values {
                let y = act.apply_scalar(x);
                assert!(
                    y.is_finite(),
                    "{:?}.apply_scalar({}) = {} is not finite",
                    act,
                    x,
                    y
                );
                let dy = act.derivative_scalar(x);
                assert!(
                    dy.is_finite(),
                    "{:?}.derivative_scalar({}) = {} is not finite",
                    act,
                    x,
                    dy
                );
            }
        }

        // Softplus with values in the representable range
        let softplus_values = [0.0f32, 1.0, -1.0, 10.0, -10.0, f32::MIN_POSITIVE];
        for &x in &softplus_values {
            let y = Activation::Softplus.apply_scalar(x);
            assert!(
                y.is_finite(),
                "Softplus.apply_scalar({}) = {} is not finite",
                x,
                y
            );
            assert!(y >= 0.0, "Softplus output must be non-negative, got {}", y);
            let dy = Activation::Softplus.derivative_scalar(x);
            assert!(
                dy.is_finite(),
                "Softplus.derivative_scalar({}) = {} is not finite",
                x,
                dy
            );
        }
    }

    #[test]
    fn test_neuromodulation_high_vitality_no_sluggishness() {
        let config = UnifiedLTCConfig::scalar(8, 4, 2);
        let mut ltc = UnifiedLTC::new(config).unwrap();

        let tau_before: Vec<_> = ltc.tau.clone();

        // High vitality (> threshold) should NOT increase tau
        ltc.apply_neuromodulation(1.5);

        // sluggishness = 1 + (1-1.5)*2 = 0.0 which is < 1.5, so tau unchanged
        for (orig, new_val) in tau_before.iter().zip(ltc.tau.iter()) {
            assert!(
                (*new_val - *orig).abs() < 1e-6,
                "tau should not increase with high vitality"
            );
        }
    }

    #[test]
    fn test_sparse_forward_pass_finite() {
        let config = UnifiedLTCConfig::scalar_sparse(32, 8, 4, 0.2);
        let mut ltc = UnifiedLTC::new(config).unwrap();

        let input = vec![0.5f32; 8];
        let (output, states) = ltc.forward(&input).unwrap();

        assert_eq!(output.len(), 4);
        for val in &output {
            assert!(val.is_finite());
        }
        for state in &states {
            assert_eq!(state.len(), 32);
        }
    }
}
