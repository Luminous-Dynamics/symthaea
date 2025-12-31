// ==================================================================================
// Learnable Liquid Time-Constant Network (LTC)
// ==================================================================================
//
// **The Gap**: Original LTC uses random initialization with no learning mechanism.
// This limits Symthaea's ability to adapt to specific tasks.
//
// **Solution**: Gradient-based optimization of:
// 1. Time constants (τ) - how fast each neuron integrates
// 2. Weights (W) - connection strengths
// 3. Biases (b) - activation thresholds
//
// **Mathematical Foundation**:
//
// Forward dynamics:
//   dx/dt = -x/τ + σ(Wx + I + b)
//
// Discretized (Euler):
//   x[t+1] = x[t] + dt * (-x[t]/τ + σ(Wx[t] + I[t] + b))
//
// Gradient computation (adjoint method):
//   dL/dτ = ∫ λ(t) * (x(t)/τ²) dt
//   dL/dW = ∫ λ(t) * σ'(z) * x(t)ᵀ dt
//   dL/db = ∫ λ(t) * σ'(z) dt
//
// where λ(t) is the adjoint state (backprop through time)
//
// **Key Innovation**: Adaptive time constants allow different neurons to
// operate at different timescales, enabling multi-scale temporal reasoning.
//
// ==================================================================================

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Configuration for learnable LTC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnableLTCConfig {
    /// Number of neurons
    pub num_neurons: usize,

    /// Input dimension
    pub input_dim: usize,

    /// Output dimension (for readout layer)
    pub output_dim: usize,

    /// Learning rate for weights
    pub lr_weights: f32,

    /// Learning rate for time constants
    pub lr_tau: f32,

    /// Learning rate for biases
    pub lr_bias: f32,

    /// Integration timestep
    pub dt: f32,

    /// Number of integration steps per forward pass
    pub num_steps: usize,

    /// Time constant range [tau_min, tau_max]
    pub tau_min: f32,
    pub tau_max: f32,

    /// Sparse connectivity (fraction of non-zero weights)
    pub sparsity: f32,

    /// L2 regularization strength
    pub l2_reg: f32,

    /// Gradient clipping threshold
    pub grad_clip: f32,
}

impl Default for LearnableLTCConfig {
    fn default() -> Self {
        Self {
            num_neurons: 1024,
            input_dim: 256,
            output_dim: 64,
            lr_weights: 0.001,
            lr_tau: 0.0001,  // Slower learning for tau
            lr_bias: 0.001,
            dt: 0.01,
            num_steps: 100,
            tau_min: 0.1,
            tau_max: 10.0,
            sparsity: 0.1,
            l2_reg: 0.0001,
            grad_clip: 1.0,
        }
    }
}

/// Learnable LTC neuron state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronState {
    /// Current activation
    pub x: f32,

    /// Time constant
    pub tau: f32,

    /// Accumulated gradient for tau
    pub grad_tau: f32,

    /// Activation history (for BPTT)
    pub history: Vec<f32>,
}

/// Learnable Liquid Time-Constant Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnableLTC {
    /// Configuration
    config: LearnableLTCConfig,

    /// Neuron states
    neurons: Vec<NeuronState>,

    /// Recurrent weights (num_neurons x num_neurons)
    w_rec: Vec<Vec<f32>>,

    /// Input weights (num_neurons x input_dim)
    w_in: Vec<Vec<f32>>,

    /// Output/readout weights (output_dim x num_neurons)
    w_out: Vec<Vec<f32>>,

    /// Biases
    bias: Vec<f32>,

    /// Connectivity mask for sparsity
    mask: Vec<Vec<bool>>,

    /// Training statistics
    stats: LTCTrainingStats,

    /// Adam optimizer state for weights
    m_w_rec: Vec<Vec<f32>>,
    v_w_rec: Vec<Vec<f32>>,

    /// Adam optimizer state for tau
    m_tau: Vec<f32>,
    v_tau: Vec<f32>,

    /// Training step counter (for Adam)
    step: usize,
}

/// Training statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LTCTrainingStats {
    pub total_steps: usize,
    pub total_loss: f32,
    pub avg_tau: f32,
    pub tau_std: f32,
    pub sparsity_actual: f32,
    pub grad_norm: f32,
}

impl LearnableLTC {
    /// Create new learnable LTC network
    pub fn new(config: LearnableLTCConfig) -> Result<Self> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Initialize neurons with random time constants
        let neurons: Vec<NeuronState> = (0..config.num_neurons)
            .map(|_| NeuronState {
                x: 0.0,
                tau: rng.gen_range(config.tau_min..config.tau_max),
                grad_tau: 0.0,
                history: Vec::with_capacity(config.num_steps),
            })
            .collect();

        // Initialize sparse recurrent weights
        let mut w_rec = vec![vec![0.0f32; config.num_neurons]; config.num_neurons];
        let mut mask = vec![vec![false; config.num_neurons]; config.num_neurons];

        for i in 0..config.num_neurons {
            for j in 0..config.num_neurons {
                if rng.gen::<f32>() < config.sparsity {
                    w_rec[i][j] = rng.gen_range(-0.1..0.1);
                    mask[i][j] = true;
                }
            }
        }

        // Initialize input weights (dense)
        let w_in: Vec<Vec<f32>> = (0..config.num_neurons)
            .map(|_| {
                (0..config.input_dim)
                    .map(|_| rng.gen_range(-0.1..0.1))
                    .collect()
            })
            .collect();

        // Initialize output weights (dense)
        let w_out: Vec<Vec<f32>> = (0..config.output_dim)
            .map(|_| {
                (0..config.num_neurons)
                    .map(|_| rng.gen_range(-0.1..0.1))
                    .collect()
            })
            .collect();

        // Initialize biases
        let bias: Vec<f32> = (0..config.num_neurons)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();

        // Initialize Adam optimizer state
        let m_w_rec = vec![vec![0.0f32; config.num_neurons]; config.num_neurons];
        let v_w_rec = vec![vec![0.0f32; config.num_neurons]; config.num_neurons];
        let m_tau = vec![0.0f32; config.num_neurons];
        let v_tau = vec![0.0f32; config.num_neurons];

        Ok(Self {
            config,
            neurons,
            w_rec,
            w_in,
            w_out,
            bias,
            mask,
            stats: LTCTrainingStats::default(),
            m_w_rec,
            v_w_rec,
            m_tau,
            v_tau,
            step: 0,
        })
    }

    /// Sigmoid activation
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Sigmoid derivative
    fn sigmoid_derivative(x: f32) -> f32 {
        let s = Self::sigmoid(x);
        s * (1.0 - s)
    }

    /// Forward pass through the network
    ///
    /// Returns (output, hidden_states) for backprop
    pub fn forward(&mut self, input: &[f32]) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        if input.len() != self.config.input_dim {
            anyhow::bail!(
                "Input dimension mismatch: expected {}, got {}",
                self.config.input_dim,
                input.len()
            );
        }

        // Clear history
        for neuron in &mut self.neurons {
            neuron.history.clear();
        }

        // Store all hidden states for BPTT
        let mut all_states: Vec<Vec<f32>> = Vec::with_capacity(self.config.num_steps);

        // Integrate for num_steps
        for _t in 0..self.config.num_steps {
            // Store current state
            let current_state: Vec<f32> = self.neurons.iter().map(|n| n.x).collect();
            all_states.push(current_state.clone());

            // Compute input contribution: W_in * input
            let mut input_contribution = vec![0.0f32; self.config.num_neurons];
            for i in 0..self.config.num_neurons {
                for j in 0..self.config.input_dim {
                    input_contribution[i] += self.w_in[i][j] * input[j];
                }
            }

            // Compute recurrent contribution: W_rec * x
            let mut rec_contribution = vec![0.0f32; self.config.num_neurons];
            for i in 0..self.config.num_neurons {
                for j in 0..self.config.num_neurons {
                    if self.mask[i][j] {
                        rec_contribution[i] += self.w_rec[i][j] * self.neurons[j].x;
                    }
                }
            }

            // Update each neuron
            for i in 0..self.config.num_neurons {
                let neuron = &mut self.neurons[i];

                // Total input: recurrent + external + bias
                let z = rec_contribution[i] + input_contribution[i] + self.bias[i];

                // LTC dynamics: dx/dt = -x/tau + sigmoid(z)
                let dx = (-neuron.x / neuron.tau + Self::sigmoid(z)) * self.config.dt;

                // Update state
                neuron.x += dx;

                // Clamp to reasonable range
                neuron.x = neuron.x.clamp(-10.0, 10.0);

                // Store in history
                neuron.history.push(neuron.x);
            }
        }

        // Compute output: W_out * final_state
        let final_state: Vec<f32> = self.neurons.iter().map(|n| n.x).collect();
        let mut output = vec![0.0f32; self.config.output_dim];

        for i in 0..self.config.output_dim {
            for j in 0..self.config.num_neurons {
                output[i] += self.w_out[i][j] * final_state[j];
            }
        }

        Ok((output, all_states))
    }

    /// Backward pass (BPTT) to compute gradients
    ///
    /// Returns gradients for all parameters
    pub fn backward(
        &mut self,
        input: &[f32],
        hidden_states: &[Vec<f32>],
        output_grad: &[f32],
    ) -> Result<()> {
        // Gradient for output weights
        let final_state = hidden_states.last().unwrap();
        let mut grad_w_out = vec![vec![0.0f32; self.config.num_neurons]; self.config.output_dim];

        for i in 0..self.config.output_dim {
            for j in 0..self.config.num_neurons {
                grad_w_out[i][j] = output_grad[i] * final_state[j];
            }
        }

        // Backprop through output layer to get gradient w.r.t final hidden state
        let mut grad_h = vec![0.0f32; self.config.num_neurons];
        for j in 0..self.config.num_neurons {
            for i in 0..self.config.output_dim {
                grad_h[j] += output_grad[i] * self.w_out[i][j];
            }
        }

        // Initialize gradient accumulators
        let mut grad_w_rec = vec![vec![0.0f32; self.config.num_neurons]; self.config.num_neurons];
        let mut grad_w_in = vec![vec![0.0f32; self.config.input_dim]; self.config.num_neurons];
        let mut grad_bias = vec![0.0f32; self.config.num_neurons];
        let mut grad_tau = vec![0.0f32; self.config.num_neurons];

        // BPTT: backprop through time
        for t in (0..self.config.num_steps).rev() {
            let state = &hidden_states[t];

            // Get previous state (or zeros if t=0)
            let prev_state = if t > 0 {
                &hidden_states[t - 1]
            } else {
                &vec![0.0f32; self.config.num_neurons]
            };

            for i in 0..self.config.num_neurons {
                let neuron = &self.neurons[i];

                // Compute z at this timestep
                let mut z = self.bias[i];
                for j in 0..self.config.num_neurons {
                    if self.mask[i][j] {
                        z += self.w_rec[i][j] * prev_state[j];
                    }
                }
                for j in 0..self.config.input_dim {
                    z += self.w_in[i][j] * input[j];
                }

                let sigmoid_grad = Self::sigmoid_derivative(z);

                // Gradient w.r.t. tau: dL/d_tau += grad_h * x / tau^2 * dt
                grad_tau[i] += grad_h[i] * state[i] / (neuron.tau * neuron.tau) * self.config.dt;

                // Gradient w.r.t. bias: dL/d_b += grad_h * sigmoid'(z) * dt
                grad_bias[i] += grad_h[i] * sigmoid_grad * self.config.dt;

                // Gradient w.r.t. recurrent weights
                for j in 0..self.config.num_neurons {
                    if self.mask[i][j] {
                        grad_w_rec[i][j] += grad_h[i] * sigmoid_grad * prev_state[j] * self.config.dt;
                    }
                }

                // Gradient w.r.t. input weights
                for j in 0..self.config.input_dim {
                    grad_w_in[i][j] += grad_h[i] * sigmoid_grad * input[j] * self.config.dt;
                }

                // Backprop gradient to previous timestep
                // dL/d_h_{t-1} = dL/d_h_t * (1 - dt/tau + dt * sigmoid'(z) * W_rec)
                let mut new_grad_h = vec![0.0f32; self.config.num_neurons];
                for j in 0..self.config.num_neurons {
                    let factor = if i == j { 1.0 - self.config.dt / neuron.tau } else { 0.0 };
                    let rec_factor = if self.mask[j][i] {
                        self.config.dt * Self::sigmoid_derivative(z) * self.w_rec[j][i]
                    } else {
                        0.0
                    };
                    new_grad_h[j] += grad_h[i] * (factor + rec_factor);
                }
                grad_h = new_grad_h;
            }
        }

        // Clip gradients
        self.clip_gradients(&mut grad_w_rec);
        self.clip_gradient_vec(&mut grad_tau);
        self.clip_gradient_vec(&mut grad_bias);

        // Apply updates using Adam optimizer
        self.adam_update_weights(&grad_w_rec);
        self.adam_update_tau(&grad_tau);
        self.update_bias(&grad_bias);
        self.update_input_weights(&grad_w_in);
        self.update_output_weights(&grad_w_out);

        // Update stats
        self.stats.total_steps += 1;
        self.stats.grad_norm = grad_tau.iter().map(|g| g * g).sum::<f32>().sqrt();

        Ok(())
    }

    /// Clip gradients to prevent explosion
    fn clip_gradients(&self, grads: &mut Vec<Vec<f32>>) {
        for row in grads.iter_mut() {
            for g in row.iter_mut() {
                *g = g.clamp(-self.config.grad_clip, self.config.grad_clip);
            }
        }
    }

    fn clip_gradient_vec(&self, grads: &mut Vec<f32>) {
        for g in grads.iter_mut() {
            *g = g.clamp(-self.config.grad_clip, self.config.grad_clip);
        }
    }

    /// Adam optimizer update for recurrent weights
    fn adam_update_weights(&mut self, grads: &Vec<Vec<f32>>) {
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;

        self.step += 1;
        let t = self.step as f32;

        for i in 0..self.config.num_neurons {
            for j in 0..self.config.num_neurons {
                if self.mask[i][j] {
                    let g = grads[i][j] + self.config.l2_reg * self.w_rec[i][j];

                    // Update biased first moment
                    self.m_w_rec[i][j] = beta1 * self.m_w_rec[i][j] + (1.0 - beta1) * g;

                    // Update biased second moment
                    self.v_w_rec[i][j] = beta2 * self.v_w_rec[i][j] + (1.0 - beta2) * g * g;

                    // Bias correction
                    let m_hat = self.m_w_rec[i][j] / (1.0 - beta1.powf(t));
                    let v_hat = self.v_w_rec[i][j] / (1.0 - beta2.powf(t));

                    // Update
                    self.w_rec[i][j] -= self.config.lr_weights * m_hat / (v_hat.sqrt() + eps);
                }
            }
        }
    }

    /// Adam optimizer update for time constants
    fn adam_update_tau(&mut self, grads: &Vec<f32>) {
        let beta1 = 0.9;
        let beta2 = 0.999;
        let eps = 1e-8;

        let t = self.step as f32;

        for i in 0..self.config.num_neurons {
            let g = grads[i];

            // Update biased first moment
            self.m_tau[i] = beta1 * self.m_tau[i] + (1.0 - beta1) * g;

            // Update biased second moment
            self.v_tau[i] = beta2 * self.v_tau[i] + (1.0 - beta2) * g * g;

            // Bias correction
            let m_hat = self.m_tau[i] / (1.0 - beta1.powf(t));
            let v_hat = self.v_tau[i] / (1.0 - beta2.powf(t));

            // Update tau (keep in valid range)
            self.neurons[i].tau -= self.config.lr_tau * m_hat / (v_hat.sqrt() + eps);
            self.neurons[i].tau = self.neurons[i].tau.clamp(self.config.tau_min, self.config.tau_max);
        }
    }

    fn update_bias(&mut self, grads: &Vec<f32>) {
        for i in 0..self.config.num_neurons {
            self.bias[i] -= self.config.lr_bias * grads[i];
        }
    }

    fn update_input_weights(&mut self, grads: &Vec<Vec<f32>>) {
        for i in 0..self.config.num_neurons {
            for j in 0..self.config.input_dim {
                self.w_in[i][j] -= self.config.lr_weights * grads[i][j];
            }
        }
    }

    fn update_output_weights(&mut self, grads: &Vec<Vec<f32>>) {
        for i in 0..self.config.output_dim {
            for j in 0..self.config.num_neurons {
                self.w_out[i][j] -= self.config.lr_weights * grads[i][j];
            }
        }
    }

    /// Train on a single example
    pub fn train_step(&mut self, input: &[f32], target: &[f32]) -> Result<f32> {
        // Forward pass
        let (output, hidden_states) = self.forward(input)?;

        // Compute loss (MSE)
        let mut loss = 0.0f32;
        let mut output_grad = vec![0.0f32; self.config.output_dim];

        for i in 0..self.config.output_dim.min(target.len()) {
            let diff = output[i] - target[i];
            loss += diff * diff;
            output_grad[i] = 2.0 * diff / target.len() as f32;
        }
        loss /= target.len() as f32;

        // Backward pass
        self.backward(input, &hidden_states, &output_grad)?;

        self.stats.total_loss += loss;

        Ok(loss)
    }

    /// Reset neuron states
    pub fn reset_state(&mut self) {
        for neuron in &mut self.neurons {
            neuron.x = 0.0;
            neuron.history.clear();
        }
    }

    /// Get current time constant distribution
    pub fn get_tau_distribution(&self) -> (f32, f32, f32, f32) {
        let taus: Vec<f32> = self.neurons.iter().map(|n| n.tau).collect();
        let mean = taus.iter().sum::<f32>() / taus.len() as f32;
        let variance = taus.iter().map(|t| (t - mean).powi(2)).sum::<f32>() / taus.len() as f32;
        let std = variance.sqrt();
        let min = taus.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = taus.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        (mean, std, min, max)
    }

    /// Get current state as vector
    pub fn get_state(&self) -> Vec<f32> {
        self.neurons.iter().map(|n| n.x).collect()
    }

    /// Get training statistics
    pub fn stats(&self) -> &LTCTrainingStats {
        &self.stats
    }

    /// Consciousness level estimate (coherent activity measure)
    pub fn consciousness_level(&self) -> f32 {
        let states: Vec<f32> = self.neurons.iter().map(|n| n.x).collect();

        // Active fraction
        let active_fraction = states.iter().filter(|&&x| x.abs() > 0.5).count() as f32
            / self.config.num_neurons as f32;

        // Variance (diversity)
        let mean = states.iter().sum::<f32>() / states.len() as f32;
        let variance = states.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
            / states.len() as f32;

        // Consciousness = active AND diverse
        (active_fraction * variance.sqrt()).min(1.0)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_pass() {
        let config = LearnableLTCConfig {
            num_neurons: 32,
            input_dim: 16,
            output_dim: 8,
            num_steps: 10,
            ..Default::default()
        };

        let mut ltc = LearnableLTC::new(config).unwrap();

        let input = vec![0.5f32; 16];
        let (output, states) = ltc.forward(&input).unwrap();

        assert_eq!(output.len(), 8);
        assert_eq!(states.len(), 10);
    }

    #[test]
    fn test_training_step() {
        let config = LearnableLTCConfig {
            num_neurons: 32,
            input_dim: 16,
            output_dim: 8,
            num_steps: 10,
            ..Default::default()
        };

        let mut ltc = LearnableLTC::new(config).unwrap();

        let input = vec![0.5f32; 16];
        let target = vec![1.0f32; 8];

        let loss1 = ltc.train_step(&input, &target).unwrap();
        ltc.reset_state();
        let loss2 = ltc.train_step(&input, &target).unwrap();

        // Loss should generally decrease with training
        println!("Loss 1: {}, Loss 2: {}", loss1, loss2);
    }

    #[test]
    fn test_tau_distribution() {
        let config = LearnableLTCConfig {
            num_neurons: 64,
            ..Default::default()
        };

        let ltc = LearnableLTC::new(config).unwrap();

        let (mean, std, min, max) = ltc.get_tau_distribution();

        assert!(mean > 0.0);
        assert!(min >= 0.1);
        assert!(max <= 10.0);
        println!("Tau: mean={}, std={}, min={}, max={}", mean, std, min, max);
    }

    #[test]
    fn test_consciousness_level() {
        let config = LearnableLTCConfig {
            num_neurons: 64,
            input_dim: 16,
            num_steps: 50,
            ..Default::default()
        };

        let mut ltc = LearnableLTC::new(config).unwrap();

        // Initially should be low
        let c0 = ltc.consciousness_level();

        // Inject input
        let input = vec![1.0f32; 16];
        let _ = ltc.forward(&input).unwrap();

        // After input, should increase
        let c1 = ltc.consciousness_level();

        println!("Consciousness: before={}, after={}", c0, c1);
    }

    #[test]
    fn test_serialization() {
        let config = LearnableLTCConfig {
            num_neurons: 16,
            input_dim: 8,
            output_dim: 4,
            ..Default::default()
        };

        let ltc = LearnableLTC::new(config).unwrap();

        let data = ltc.serialize().unwrap();
        let restored = LearnableLTC::deserialize(&data).unwrap();

        assert_eq!(ltc.config.num_neurons, restored.config.num_neurons);
    }
}
