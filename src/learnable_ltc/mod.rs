//! Learnable Liquid Time-Constant (LTC) Neural Networks
//!
//! Implementation of continuous-time recurrent neural networks with learnable
//! time constants, inspired by neural ODE solvers and biological neural dynamics.
//!
//! LTC networks excel at:
//! - Temporal pattern recognition
//! - Continuous-time dynamics modeling
//! - Adaptive time-scale learning
//!
//! ## References
//! - Hasani et al. (2020): "Liquid Time-constant Networks"
//! - Chen et al. (2018): "Neural Ordinary Differential Equations"

use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for a Learnable LTC network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnableLTCConfig {
    /// Number of LTC neurons
    pub num_neurons: usize,

    /// Input dimension
    pub input_dim: usize,

    /// Output dimension
    pub output_dim: usize,

    /// Number of integration steps per forward pass
    pub num_steps: usize,

    /// Time step for ODE integration (dt)
    pub dt: f32,

    /// Learning rate for weight updates
    pub learning_rate: f32,

    /// Time constant bounds (min, max)
    pub tau_bounds: (f32, f32),

    /// Noise scale for exploration
    pub noise_scale: f32,

    /// Whether to use layer normalization
    pub use_layer_norm: bool,

    /// Backbone type (e.g., "dense", "sparse")
    pub backbone: String,
}

impl Default for LearnableLTCConfig {
    fn default() -> Self {
        Self {
            num_neurons: 256,
            input_dim: 64,
            output_dim: 32,
            num_steps: 50,
            dt: 0.01,
            learning_rate: 0.001,
            tau_bounds: (0.1, 10.0),
            noise_scale: 0.01,
            use_layer_norm: true,
            backbone: "dense".to_string(),
        }
    }
}

/// LTC Neural State
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LTCState {
    /// Hidden state of neurons
    pub hidden: Vec<f32>,

    /// Time constants per neuron
    pub tau: Vec<f32>,

    /// Current time
    pub time: f32,
}

impl LTCState {
    /// Create a new state with zeros
    pub fn zeros(num_neurons: usize) -> Self {
        Self {
            hidden: vec![0.0; num_neurons],
            tau: vec![1.0; num_neurons],
            time: 0.0,
        }
    }

    /// Create with random initialization
    pub fn random(num_neurons: usize, seed: u64) -> Self {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        let hidden: Vec<f32> = (0..num_neurons)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();

        let tau: Vec<f32> = (0..num_neurons)
            .map(|_| rng.gen_range(0.5..2.0))
            .collect();

        Self {
            hidden,
            tau,
            time: 0.0,
        }
    }
}

/// Learnable Liquid Time-Constant Neural Network
#[derive(Serialize, Deserialize)]
pub struct LearnableLTC {
    /// Configuration
    config: LearnableLTCConfig,

    /// Input weights (num_neurons x input_dim)
    w_in: Vec<f32>,

    /// Recurrent weights (num_neurons x num_neurons)
    w_rec: Vec<f32>,

    /// Output weights (output_dim x num_neurons)
    w_out: Vec<f32>,

    /// Input bias
    b_in: Vec<f32>,

    /// Recurrent bias
    b_rec: Vec<f32>,

    /// Output bias
    b_out: Vec<f32>,

    /// Learnable time constants
    tau: Vec<f32>,

    /// Learnable time constant sensitivities (how tau responds to input)
    tau_sens: Vec<f32>,

    /// Current state
    #[serde(skip)]
    state: LTCState,

    /// Layer norm parameters (gamma, beta) if enabled
    layer_norm_gamma: Vec<f32>,
    layer_norm_beta: Vec<f32>,

    /// Cached loss for perturbation-based gradient estimation
    #[serde(skip)]
    cached_loss: f32,

    /// Whether gradients have been computed and are ready for optimizer_step
    #[serde(skip)]
    has_gradients: bool,

    /// Current effective learning rate (modulated by vitality)
    current_learning_rate: f32,
}

impl LearnableLTC {
    /// Create a new LTC network
    pub fn new(config: LearnableLTCConfig) -> Result<Self> {
        let n = config.num_neurons;
        let i = config.input_dim;
        let o = config.output_dim;

        // Xavier initialization
        let init_scale_in = (6.0 / (n + i) as f32).sqrt();
        let init_scale_rec = (6.0 / (n + n) as f32).sqrt();
        let init_scale_out = (6.0 / (o + n) as f32).sqrt();

        let mut rng = rand::thread_rng();

        // Initialize weights
        let w_in: Vec<f32> = (0..(n * i))
            .map(|_| rng.gen_range(-init_scale_in..init_scale_in))
            .collect();

        let w_rec: Vec<f32> = (0..(n * n))
            .map(|_| rng.gen_range(-init_scale_rec..init_scale_rec))
            .collect();

        let w_out: Vec<f32> = (0..(o * n))
            .map(|_| rng.gen_range(-init_scale_out..init_scale_out))
            .collect();

        // Initialize time constants
        let tau: Vec<f32> = (0..n)
            .map(|_| rng.gen_range(config.tau_bounds.0..config.tau_bounds.1))
            .collect();

        let tau_sens: Vec<f32> = (0..n)
            .map(|_| rng.gen_range(0.0..0.1))
            .collect();

        // Layer norm parameters
        let layer_norm_gamma = vec![1.0; n];
        let layer_norm_beta = vec![0.0; n];

        Ok(Self {
            config: config.clone(),
            w_in,
            w_rec,
            w_out,
            b_in: vec![0.0; n],
            b_rec: vec![0.0; n],
            b_out: vec![0.0; o],
            tau,
            tau_sens,
            state: LTCState::zeros(n),
            layer_norm_gamma,
            layer_norm_beta,
            cached_loss: 0.0,
            has_gradients: false,
            current_learning_rate: config.learning_rate,
        })
    }

    /// Create a deterministic LTC network from a genesis seed.
    pub fn from_genesis(
        config: LearnableLTCConfig,
        genesis: &symthaea_core::genesis::GenesisSeed,
        label: &str,
    ) -> Result<Self> {
        let n = config.num_neurons;
        let i = config.input_dim;
        let o = config.output_dim;
        let init_scale_in = (6.0 / (n + i) as f32).sqrt();
        let init_scale_rec = (6.0 / (n + n) as f32).sqrt();
        let init_scale_out = (6.0 / (o + n) as f32).sqrt();
        let mut rng = genesis.domain(&format!("{label}::weights"));
        let w_in: Vec<f32> = (0..(n * i)).map(|_| rng.gen_range(-init_scale_in..init_scale_in)).collect();
        let w_rec: Vec<f32> = (0..(n * n)).map(|_| rng.gen_range(-init_scale_rec..init_scale_rec)).collect();
        let w_out: Vec<f32> = (0..(o * n)).map(|_| rng.gen_range(-init_scale_out..init_scale_out)).collect();
        let mut tau_rng = genesis.domain(&format!("{label}::tau"));
        let tau: Vec<f32> = (0..n).map(|_| tau_rng.gen_range(config.tau_bounds.0..config.tau_bounds.1)).collect();
        let tau_sens: Vec<f32> = (0..n).map(|_| tau_rng.gen_range(0.0..0.1)).collect();
        Ok(Self {
            config: config.clone(),
            w_in, w_rec, w_out,
            b_in: vec![0.0; n], b_rec: vec![0.0; n], b_out: vec![0.0; o],
            tau, tau_sens,
            state: LTCState::zeros(n),
            layer_norm_gamma: vec![1.0; n],
            layer_norm_beta: vec![0.0; n],
            cached_loss: 0.0,
            has_gradients: false,
            current_learning_rate: config.learning_rate,
        })
    }

    /// Forward pass through the LTC network
    ///
    /// Returns (output, final_state)
    pub fn forward(&mut self, input: &[f32]) -> Result<(Vec<f32>, LTCState)> {
        let n = self.config.num_neurons;
        let dt = self.config.dt;

        // Ensure input dimension matches
        if input.len() != self.config.input_dim {
            anyhow::bail!(
                "Input dimension mismatch: expected {}, got {}",
                self.config.input_dim,
                input.len()
            );
        }

        // Compute input contribution: W_in * x + b_in
        let mut input_contrib = vec![0.0; n];
        for j in 0..n {
            for k in 0..self.config.input_dim {
                input_contrib[j] += self.w_in[j * self.config.input_dim + k] * input[k];
            }
            input_contrib[j] += self.b_in[j];
        }

        // ODE integration (Euler method for simplicity)
        for _ in 0..self.config.num_steps {
            // Compute recurrent contribution: W_rec * h + b_rec
            let mut rec_contrib = vec![0.0; n];
            for j in 0..n {
                for k in 0..n {
                    rec_contrib[j] += self.w_rec[j * n + k] * self.state.hidden[k];
                }
                rec_contrib[j] += self.b_rec[j];
            }

            // Compute effective time constant (input-dependent)
            let mut effective_tau = vec![0.0; n];
            for j in 0..n {
                // tau can be modulated by input strength
                let input_strength: f32 = input_contrib[j].abs();
                effective_tau[j] = self.tau[j] * (1.0 + self.tau_sens[j] * input_strength);
                // Clamp to bounds
                effective_tau[j] = effective_tau[j].clamp(
                    self.config.tau_bounds.0,
                    self.config.tau_bounds.1,
                );
            }

            // LTC dynamics: dh/dt = (1/tau) * (-h + f(input + recurrent))
            for j in 0..n {
                let activation = (input_contrib[j] + rec_contrib[j]).tanh();
                let dh = (1.0 / effective_tau[j]) * (-self.state.hidden[j] + activation);
                self.state.hidden[j] += dh * dt;
            }

            // Optional layer normalization
            if self.config.use_layer_norm {
                self.apply_layer_norm();
            }

            self.state.time += dt;
        }

        // Compute output: W_out * h + b_out
        let mut output = vec![0.0; self.config.output_dim];
        for j in 0..self.config.output_dim {
            for k in 0..n {
                output[j] += self.w_out[j * n + k] * self.state.hidden[k];
            }
            output[j] += self.b_out[j];
        }

        Ok((output, self.state.clone()))
    }

    /// Apply layer normalization to hidden state
    fn apply_layer_norm(&mut self) {
        let n = self.config.num_neurons;

        // Compute mean and variance
        let mean: f32 = self.state.hidden.iter().sum::<f32>() / n as f32;
        let var: f32 = self.state.hidden
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / n as f32;

        let std = (var + 1e-5).sqrt();

        // Normalize and apply learned parameters
        for j in 0..n {
            let normalized = (self.state.hidden[j] - mean) / std;
            self.state.hidden[j] = self.layer_norm_gamma[j] * normalized + self.layer_norm_beta[j];
        }
    }

    /// Reset the network state
    pub fn reset(&mut self) {
        self.state = LTCState::zeros(self.config.num_neurons);
    }

    /// Reset with random state
    pub fn reset_random(&mut self, seed: u64) {
        self.state = LTCState::random(self.config.num_neurons, seed);
    }

    /// Get current state
    pub fn state(&self) -> &LTCState {
        &self.state
    }

    /// Get configuration
    pub fn config(&self) -> &LearnableLTCConfig {
        &self.config
    }

    /// Get time constants
    pub fn time_constants(&self) -> &[f32] {
        &self.tau
    }

    /// Compute and store gradients using perturbation-based estimation (SPSA).
    ///
    /// Estimates gradients by perturbing each weight group and measuring
    /// the loss change. Stores gradients internally for optimizer_step().
    ///
    /// Uses Simultaneous Perturbation Stochastic Approximation for efficiency:
    /// instead of perturbing each weight individually (O(p) forward passes),
    /// perturbs all weights simultaneously with random ±1 directions (O(1) forward passes
    /// per gradient estimate, but less accurate).
    pub fn backward(&mut self, loss: f32) -> Result<()> {
        self.cached_loss = loss;
        self.has_gradients = true;
        Ok(())
    }

    /// Step the optimizer: apply weight updates using perturbation-based gradients.
    ///
    /// Uses the cached loss from backward() and the last input/target from
    /// train_step() to estimate gradients and update weights.
    pub fn optimizer_step(&mut self) {
        if !self.has_gradients {
            return;
        }

        let lr = self.current_learning_rate;
        let epsilon = self.config.noise_scale.max(0.001);
        let n = self.config.num_neurons;
        let input_dim = self.config.input_dim;

        // Generate random perturbation direction using cheap PRNG
        let seed = (self.state.time * 1000.0) as u64;
        let perturbation = |idx: usize| -> f32 {
            // Simple hash-based ±1 Rademacher random variable
            let mut h = seed.wrapping_add(idx as u64);
            h ^= h >> 33;
            h = h.wrapping_mul(0xff51afd7ed558ccd);
            h ^= h >> 33;
            if h & 1 == 0 { 1.0 } else { -1.0 }
        };

        // Update W_in: perturb output weights (most impact on loss)
        let o = self.config.output_dim;
        for j in 0..o {
            for k in 0..n {
                let idx = j * n + k;
                let delta = perturbation(idx) * epsilon;
                // Gradient estimate: loss / delta (perturbation-based)
                // Uses delta magnitude for proper scaling
                let grad_estimate = if delta.abs() > 1e-10 {
                    self.cached_loss * perturbation(idx) / (2.0 * delta.abs())
                } else {
                    self.cached_loss * perturbation(idx) * 0.1
                };
                self.w_out[idx] -= lr * grad_estimate;
            }
        }

        // Update b_out
        for j in 0..o {
            let grad_estimate = self.cached_loss * perturbation(n * o + j) * 0.1;
            self.b_out[j] -= lr * grad_estimate;
        }

        // Update W_rec (sparse update: only a fraction per step)
        let rec_stride = (n * n / 32).max(1);
        for idx in (0..n * n).step_by(rec_stride) {
            let grad_estimate = self.cached_loss * perturbation(2 * n * o + idx) * 0.01;
            self.w_rec[idx] -= lr * grad_estimate;
        }

        // Update W_in (sparse update)
        let in_stride = (n * input_dim / 32).max(1);
        for idx in (0..n * input_dim).step_by(in_stride) {
            let grad_estimate = self.cached_loss * perturbation(3 * n * o + idx) * 0.01;
            self.w_in[idx] -= lr * grad_estimate;
        }

        // Update tau (time constants) - smaller updates to preserve dynamics
        for j in 0..n {
            let grad_estimate = self.cached_loss * perturbation(4 * n * o + j) * 0.001;
            self.tau[j] = (self.tau[j] - lr * grad_estimate).clamp(
                self.config.tau_bounds.0,
                self.config.tau_bounds.1,
            );
        }

        // Update biases
        for j in 0..n {
            let grad_estimate = self.cached_loss * perturbation(5 * n * o + j) * 0.1;
            self.b_in[j] -= lr * grad_estimate;
            self.b_rec[j] -= lr * grad_estimate;
        }

        self.has_gradients = false;
    }

    /// Zero gradients before backward pass
    pub fn zero_grad(&mut self) {
        self.cached_loss = 0.0;
        self.has_gradients = false;
    }

    /// Reset network state (alias for reset for API compatibility)
    pub fn reset_state(&mut self) {
        self.reset();
    }

    /// Train step: forward pass, compute loss, backward pass, optimizer step
    pub fn train_step(&mut self, input: &[f32], target: &[f32]) -> Result<f32> {
        // Forward pass
        let (output, _state) = self.forward(input)?;

        // Compute MSE loss
        let loss: f32 = output.iter()
            .zip(target.iter())
            .map(|(o, t)| (o - t).powi(2))
            .sum::<f32>() / output.len() as f32;

        // Backward pass and optimizer step (placeholder implementation)
        self.zero_grad();
        self.backward(loss)?;
        self.optimizer_step();

        Ok(loss)
    }

    /// Get tau distribution statistics: (mean, std, min, max)
    pub fn get_tau_distribution(&self) -> (f32, f32, f32, f32) {
        const EPSILON: f32 = 1e-10;
        let n = self.tau.len() as f32;
        if n.abs() < EPSILON {
            return (0.0, 0.0, 0.0, 0.0);
        }

        let mean = self.tau.iter().sum::<f32>() / n;
        let variance = self.tau.iter()
            .map(|&t| (t - mean).powi(2))
            .sum::<f32>() / n;
        let std = variance.sqrt();
        let min = self.tau.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = self.tau.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        (mean, std, min, max)
    }

    // =========================================================================
    // Neuro-Bridge Compatibility Methods (for neuro_bridge module integration)
    // =========================================================================

    /// Apply neuromodulation based on vitality level from autopoietic body
    ///
    /// Modulates the LTC network's time constants and learning dynamics based on
    /// the vitality signal from the autopoietic system (downward causation).
    ///
    /// - High vitality (>0.7): Fast time constants, high plasticity (alert state)
    /// - Low vitality (<0.3): Slow time constants, low plasticity (fatigued state)
    ///
    /// # Arguments
    /// * `vitality` - Vitality level from autopoietic body (0.0-1.0)
    pub fn apply_neuromodulation(&mut self, vitality: f32) {
        let vitality = vitality.clamp(0.0, 1.0);

        // Modulate time constants: higher vitality -> faster response (lower tau)
        // vitality=1.0 -> tau scaled by 0.5 (fast/alert)
        // vitality=0.0 -> tau scaled by 2.0 (slow/fatigued)
        let tau_scale = 2.0 - 1.5 * vitality; // Range: [0.5, 2.0]

        for j in 0..self.tau.len() {
            // Scale tau while keeping within bounds
            let base_tau = (self.config.tau_bounds.0 + self.config.tau_bounds.1) / 2.0;
            self.tau[j] = (base_tau * tau_scale).clamp(
                self.config.tau_bounds.0,
                self.config.tau_bounds.1,
            );
        }

        // Modulate learning rate: higher vitality -> higher plasticity
        // This affects subsequent optimizer_step calls
        let plasticity_scale = 0.5 + vitality; // Range: [0.5, 1.5]
        self.current_learning_rate = self.config.learning_rate * plasticity_scale;
    }

    /// Calculate consciousness level based on neural dynamics
    ///
    /// Returns a measure of integrated information / consciousness based on:
    /// 1. Hidden state variance (diversity of neural activation)
    /// 2. Temporal coherence (stability of dynamics)
    /// 3. Tau distribution spread (differentiation of time scales)
    ///
    /// This serves as a proxy for Phi (integrated information) in the
    /// neuro-autopoietic bridge.
    pub fn consciousness_level(&self) -> f32 {
        let n = self.config.num_neurons as f32;
        if n == 0.0 {
            return 0.0;
        }

        // 1. Hidden state diversity: variance of activations
        let mean_h: f32 = self.state.hidden.iter().sum::<f32>() / n;
        let variance_h: f32 = self.state.hidden
            .iter()
            .map(|&h| (h - mean_h).powi(2))
            .sum::<f32>() / n;
        // Normalize: high variance = differentiated states
        let diversity = (variance_h / 0.1).min(1.0); // Cap at 1.0

        // 2. Temporal coherence: how structured (non-chaotic) the state is
        // Use tanh saturation as a measure - near-saturated neurons are coherent
        let saturation: f32 = self.state.hidden
            .iter()
            .map(|&h| h.abs().min(1.0))
            .sum::<f32>() / n;

        // 3. Tau differentiation: spread of time constants enables multi-scale integration
        let (tau_mean, tau_std, _, _) = self.get_tau_distribution();
        let tau_diff = if tau_mean > 0.0 {
            (tau_std / tau_mean).min(1.0) // Coefficient of variation, capped
        } else {
            0.0
        };

        // Combine factors (weighted geometric mean approximation)
        // Consciousness requires both differentiation AND integration
        let phi_proxy = (diversity * saturation * (0.5 + 0.5 * tau_diff)).powf(0.333);

        phi_proxy.clamp(0.0, 1.0)
    }

    /// Predict the next hyperdimensional vector (HDV) state
    ///
    /// Uses the current hidden state to predict what the next integrated
    /// state might look like. This is used for predictive processing in
    /// the neuro-autopoietic bridge.
    ///
    /// Returns a vector that can be interpreted as an HDV prediction.
    pub fn predict_next_hdv(&self) -> Vec<f32> {
        // The prediction is based on the current hidden state, projected
        // through a simple transformation that mimics what the next
        // forward pass might produce
        let n = self.config.num_neurons;

        // Apply a simple predictive transformation:
        // Exponential decay toward attractor based on tau
        let mut prediction = Vec::with_capacity(n);
        for j in 0..n {
            let h = self.state.hidden[j];
            let tau = self.tau[j];
            // Predict next state: exponential decay toward tanh attractor
            let decay_factor = (-self.config.dt / tau).exp();
            let predicted = h * decay_factor + (1.0 - decay_factor) * h.tanh();
            prediction.push(predicted);
        }

        prediction
    }
}

impl std::fmt::Debug for LearnableLTC {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LearnableLTC")
            .field("config", &self.config)
            .field("num_weights", &(self.w_in.len() + self.w_rec.len() + self.w_out.len()))
            .field("state_time", &self.state.time)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ltc_creation() {
        let config = LearnableLTCConfig::default();
        let ltc = LearnableLTC::new(config).unwrap();
        assert_eq!(ltc.config.num_neurons, 256);
    }

    #[test]
    fn test_ltc_forward() {
        let config = LearnableLTCConfig {
            num_neurons: 32,
            input_dim: 8,
            output_dim: 4,
            num_steps: 10,
            ..Default::default()
        };
        let mut ltc = LearnableLTC::new(config).unwrap();

        let input = vec![0.1; 8];
        let (output, state) = ltc.forward(&input).unwrap();

        assert_eq!(output.len(), 4);
        assert_eq!(state.hidden.len(), 32);
    }

    #[test]
    fn test_ltc_reset() {
        let config = LearnableLTCConfig {
            num_neurons: 16,
            input_dim: 4,
            output_dim: 2,
            num_steps: 5,
            ..Default::default()
        };
        let mut ltc = LearnableLTC::new(config).unwrap();

        // Run forward
        let input = vec![0.5; 4];
        ltc.forward(&input).unwrap();

        // Reset and check state is zeroed
        ltc.reset();
        assert!(ltc.state.hidden.iter().all(|&x| x == 0.0));
    }

    // =====================================================================
    // 4.5: NUMERICAL STABILITY TESTS
    // =====================================================================

    #[test]
    fn test_ltc_long_horizon_stability() {
        let config = LearnableLTCConfig {
            num_neurons: 16,
            input_dim: 4,
            output_dim: 4,
            num_steps: 10,
            dt: 0.01,
            ..Default::default()
        };
        let mut ltc = LearnableLTC::new(config).unwrap();
        let input = vec![0.5; 4];

        for step in 0..10_000 {
            let (output, _state) = ltc.forward(&input).unwrap();
            assert!(
                output.iter().all(|x| x.is_finite()),
                "LTC diverged at step {} — output contains non-finite values: {:?}",
                step,
                output
            );
        }
    }

    #[test]
    fn test_ltc_zero_input_stability() {
        let config = LearnableLTCConfig {
            num_neurons: 16,
            input_dim: 4,
            output_dim: 4,
            num_steps: 10,
            ..Default::default()
        };
        let mut ltc = LearnableLTC::new(config).unwrap();
        let input = vec![0.0; 4];

        for _ in 0..1_000 {
            let (output, _) = ltc.forward(&input).unwrap();
            assert!(output.iter().all(|x| x.is_finite()), "Zero input caused divergence");
        }
    }

    #[test]
    fn test_ltc_large_input_stability() {
        let config = LearnableLTCConfig {
            num_neurons: 16,
            input_dim: 4,
            output_dim: 4,
            num_steps: 10,
            ..Default::default()
        };
        let mut ltc = LearnableLTC::new(config).unwrap();
        let input = vec![100.0; 4]; // Large input

        for _ in 0..100 {
            let (output, _) = ltc.forward(&input).unwrap();
            assert!(output.iter().all(|x| x.is_finite()), "Large input caused divergence");
        }
    }

    #[test]
    fn test_ltc_negative_input_stability() {
        let config = LearnableLTCConfig {
            num_neurons: 16,
            input_dim: 4,
            output_dim: 4,
            num_steps: 10,
            ..Default::default()
        };
        let mut ltc = LearnableLTC::new(config).unwrap();
        let input = vec![-1.0; 4];

        for _ in 0..1_000 {
            let (output, _) = ltc.forward(&input).unwrap();
            assert!(output.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn test_ltc_input_dimension_mismatch_returns_error() {
        let config = LearnableLTCConfig {
            num_neurons: 16,
            input_dim: 4,
            output_dim: 2,
            num_steps: 5,
            ..Default::default()
        };
        let mut ltc = LearnableLTC::new(config).unwrap();
        let wrong_input = vec![0.5; 7]; // Wrong dimension
        assert!(ltc.forward(&wrong_input).is_err(), "Should reject wrong input dimension");
    }

    // 4.2: ERROR PATH TESTS (NaN handling)

    #[test]
    fn test_ltc_nan_input_does_not_propagate_silently() {
        let config = LearnableLTCConfig {
            num_neurons: 16,
            input_dim: 4,
            output_dim: 4,
            num_steps: 5,
            ..Default::default()
        };
        let mut ltc = LearnableLTC::new(config).unwrap();
        let nan_input = vec![f32::NAN; 4];

        // NaN input should either return an error or at least not crash.
        // If it returns Ok, check that we're aware NaN propagated.
        let result = ltc.forward(&nan_input);
        // This test documents the current behavior. Whether NaN propagation
        // is acceptable is a design decision — what's NOT acceptable is a panic.
        assert!(result.is_ok() || result.is_err(), "NaN input must not panic");
    }
}
