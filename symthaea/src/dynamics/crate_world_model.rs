// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Hierarchical CfC World Model
============================

The **Learning Heart** of Symthaea - transforms the system from a "zombie"
(moves but doesn't learn) into a genuine AGI substrate.

# Core Innovation

Uses Closed-form Continuous-time (CfC) networks with analytical gradients
to learn world dynamics from experience. NO BPTT required!

**The Loop**:
1. **Predict**: Given (state, action), predict next_state using CfC O(1) jump
2. **Observe**: Receive actual next_state from environment
3. **Surprise**: Compute prediction error (MSE loss)
4. **Learn**: Update weights using analytical gradients
5. **Consciousness**: High surprise = High consciousness (attention needed)

# Mathematical Foundation

The CfC closed-form solution:
$$h(t) = (h_0 - A) \cdot e^{-t/\tau} + A$$

Enables O(1) prediction to ANY future time point, and the analytical
derivative allows gradient computation without backprop-through-time:
$$\frac{\partial h}{\partial A} = 1 - e^{-t/\tau}$$

# Architecture

```text
State (HDC) + Action (HDC)     [Input: 2 x 16,384D]
        |
   JL Projection               [32,768 -> 256]
        |
   CfC Backbone                [256 neurons, continuous-time]
        |
   Output Head                 [256 -> 16,384]
        |
Predicted Next State (HDC)     [Output: 16,384D]
```

Reference: Hasani et al., "Closed-form Continuous-time Neural Networks",
Nature Machine Intelligence (2022).
*/

use anyhow::Result;
use ndarray::{Array1, Array2};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for the World Model
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorldModelConfig {
    /// HDC dimension for input/output states
    pub hdc_dim: usize,
    /// Hidden layer size (CfC backbone)
    pub hidden_dim: usize,
    /// Learning rate for gradient descent
    pub learning_rate: f32,
    /// Momentum for SGD with momentum
    pub momentum: f32,
    /// Exponential moving average decay for consciousness tracking
    pub consciousness_decay: f32,
}

impl Default for WorldModelConfig {
    fn default() -> Self {
        Self {
            hdc_dim: 16_384,           // Standard HDC dimension
            hidden_dim: 256,           // CfC backbone size
            learning_rate: 0.001,      // Conservative learning rate
            momentum: 0.9,             // Standard momentum
            consciousness_decay: 0.99, // Slow decay for surprise tracking
        }
    }
}

/// Hierarchical CfC World Model
///
/// Learns to predict next HDC states from current state + action.
/// Uses closed-form continuous-time dynamics with analytical gradients.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HierarchicalCfCWorldModel {
    /// Configuration
    pub config: WorldModelConfig,

    // === Input Projection (JL-inspired random projection) ===
    /// Projects concatenated (state, action) to hidden dimension
    /// Shape: [hidden_dim, 2 * hdc_dim]
    pub w_input: Array2<f32>,
    pub b_input: Array1<f32>,

    // === CfC Backbone (The "Liquid Brain") ===
    /// Hidden state (persistent across predictions)
    pub cfc_state: Array1<f32>,
    /// Time constants per neuron
    pub cfc_tau: Array1<f32>,
    /// Recurrent weights for backbone
    pub w_backbone: Array2<f32>,
    pub b_backbone: Array1<f32>,
    /// Target state head (the "A" in CfC)
    pub w_target: Array2<f32>,
    pub b_target: Array1<f32>,

    // === Output Projection ===
    /// Projects hidden state back to HDC dimension
    /// Shape: [hdc_dim, hidden_dim]
    pub w_output: Array2<f32>,
    pub b_output: Array1<f32>,

    // === Training State ===
    /// Accumulated surprise (prediction error) - THIS IS CONSCIOUSNESS
    pub accumulated_surprise: f32,
    /// Exponential moving average of surprise
    pub consciousness_level: f32,
    /// Total training steps
    pub training_steps: u64,

    // === Momentum Buffers (for SGD with momentum) ===
    #[serde(skip)]
    pub v_w_input: Option<Array2<f32>>,
    #[serde(skip)]
    pub v_w_output: Option<Array2<f32>>,
    #[serde(skip)]
    pub v_w_target: Option<Array2<f32>>,
}

impl HierarchicalCfCWorldModel {
    /// Xavier/Glorot initialization for weight matrices
    fn xavier_init(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        let scale = (6.0 / (rows + cols) as f32).sqrt();
        Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-scale..scale))
    }

    /// Create a new World Model with given configuration
    pub fn new(config: WorldModelConfig) -> Result<Self> {
        let mut rng = rand::thread_rng();
        let base_seed: u64 = rng.r#gen();

        // Input projection: [hidden_dim, 2 * hdc_dim]
        // Uses sparse random projection (Johnson-Lindenstrauss style)
        let input_dim = 2 * config.hdc_dim;
        let w_input = {
            use rand::SeedableRng;
            let mut sparse_rng = rand_chacha::ChaCha8Rng::seed_from_u64(base_seed);
            let scale = (1.0 / config.hidden_dim as f32).sqrt();
            Array2::from_shape_fn((config.hidden_dim, input_dim), |_| {
                if sparse_rng.r#gen::<f32>() < 0.1 {
                    sparse_rng.gen_range(-scale..scale) * (10.0_f32).sqrt()
                } else {
                    0.0
                }
            })
        };

        // CfC time constants: uniform [0.5, 2.0] (biologically plausible range)
        let cfc_tau = {
            use rand::SeedableRng;
            let mut tau_rng = rand_chacha::ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(1));
            Array1::from_iter((0..config.hidden_dim).map(|_| tau_rng.gen_range(0.5..2.0)))
        };

        Ok(Self {
            config: config.clone(),

            // Input projection
            w_input,
            b_input: Array1::zeros(config.hidden_dim),

            // CfC backbone
            cfc_state: Array1::zeros(config.hidden_dim),
            cfc_tau,
            w_backbone: Self::xavier_init(
                config.hidden_dim,
                config.hidden_dim,
                base_seed.wrapping_add(2),
            ),
            b_backbone: Array1::zeros(config.hidden_dim),
            w_target: Self::xavier_init(
                config.hidden_dim,
                config.hidden_dim,
                base_seed.wrapping_add(3),
            ),
            b_target: Array1::zeros(config.hidden_dim),

            // Output projection
            w_output: Self::xavier_init(
                config.hdc_dim,
                config.hidden_dim,
                base_seed.wrapping_add(4),
            ),
            b_output: Array1::zeros(config.hdc_dim),

            // Training state
            accumulated_surprise: 0.0,
            consciousness_level: 0.0,
            training_steps: 0,

            // Momentum buffers (initialized lazily)
            v_w_input: None,
            v_w_output: None,
            v_w_target: None,
        })
    }

    /// Create with default configuration
    pub fn default_model() -> Result<Self> {
        Self::new(WorldModelConfig::default())
    }

    /// Predict next state given current state and action (HDC-encoded)
    ///
    /// This is the O(1) forward pass using CfC closed-form solution.
    /// No loops over time steps - just one calculation!
    ///
    /// # Arguments
    /// * `state` - Current state as HDC vector (16,384D)
    /// * `action` - Action as HDC vector (16,384D)
    /// * `delta_t` - Time step for prediction (typically 1.0)
    ///
    /// # Returns
    /// Predicted next state as HDC vector (16,384D)
    pub fn predict(&self, state: &[f32], action: &[f32], delta_t: f32) -> Result<Vec<f32>> {
        // 1. Concatenate state and action
        let mut input = Vec::with_capacity(state.len() + action.len());
        input.extend_from_slice(state);
        input.extend_from_slice(action);

        // Handle dimension mismatch gracefully
        let expected_input_dim = 2 * self.config.hdc_dim;
        let input_arr = if input.len() != expected_input_dim {
            // Pad or truncate
            let mut arr = Array1::zeros(expected_input_dim);
            let n = input.len().min(expected_input_dim);
            for i in 0..n {
                arr[i] = input[i];
            }
            arr
        } else {
            Array1::from_vec(input)
        };

        // 2. Project to hidden dimension: z = tanh(W_input · x + b_input)
        let projected = (self.w_input.dot(&input_arr) + &self.b_input).mapv(|v| v.tanh());

        // 3. CfC Backbone (Closed-Form Solution!)
        // Compute target state A: A = W_target · z + b_target
        let backbone_in = self.w_backbone.dot(&self.cfc_state) + &projected;
        let backbone_act = backbone_in.mapv(|v| v.tanh());
        let target_state = self.w_target.dot(&backbone_act) + &self.b_target;

        // Compute decay factor: exp(-dt / tau)
        let decay_factor = (-delta_t / (&self.cfc_tau + 0.01)).mapv(|v| v.exp());

        // CfC Master Equation: h(t) = (h_0 - A) * decay + A
        let next_hidden = (&self.cfc_state - &target_state) * &decay_factor + &target_state;

        // 4. Project back to HDC dimension
        let output = self.w_output.dot(&next_hidden) + &self.b_output;

        Ok(output.to_vec())
    }

    /// Predict and update internal state (stateful prediction)
    pub fn predict_and_update(
        &mut self,
        state: &[f32],
        action: &[f32],
        delta_t: f32,
    ) -> Result<Vec<f32>> {
        let prediction = self.predict(state, action, delta_t)?;

        // Update CfC internal state for next prediction
        // (This creates temporal dependencies in the hidden state)
        let input_arr = {
            let mut input = Vec::with_capacity(state.len() + action.len());
            input.extend_from_slice(state);
            input.extend_from_slice(action);
            let expected_input_dim = 2 * self.config.hdc_dim;
            let mut arr = Array1::zeros(expected_input_dim);
            let n = input.len().min(expected_input_dim);
            for i in 0..n {
                arr[i] = input[i];
            }
            arr
        };

        let projected = (self.w_input.dot(&input_arr) + &self.b_input).mapv(|v| v.tanh());
        let backbone_in = self.w_backbone.dot(&self.cfc_state) + &projected;
        let backbone_act = backbone_in.mapv(|v| v.tanh());
        let target_state = self.w_target.dot(&backbone_act) + &self.b_target;
        let decay_factor = (-delta_t / (&self.cfc_tau + 0.01)).mapv(|v| v.exp());

        self.cfc_state = (&self.cfc_state - &target_state) * &decay_factor + &target_state;

        Ok(prediction)
    }

    /// Train on a single experience tuple (state, action, next_state)
    ///
    /// Uses analytical gradients from CfC closed-form - no BPTT!
    ///
    /// # Returns
    /// The prediction error (MSE loss) - this is the "surprise"
    #[allow(non_snake_case)] // Mathematical notation: A = target state, dL/dA = gradient
    pub fn train_step(
        &mut self,
        state: &[f32],
        action: &[f32],
        next_state: &[f32],
        delta_t: f32,
    ) -> Result<f32> {
        // Forward pass
        let prediction = self.predict(state, action, delta_t)?;

        // Compute loss (MSE)
        let n = prediction.len().min(next_state.len());
        let mut mse = 0.0;
        let mut error = vec![0.0; n];
        for i in 0..n {
            let diff = prediction[i] - next_state[i];
            error[i] = diff;
            mse += diff * diff;
        }
        mse /= n as f32;

        // === Analytical Gradient Computation ===
        //
        // For CfC: h(t) = (h_0 - A) * decay + A
        //          ∂h/∂A = 1 - decay
        //
        // For output: y = W_output · h + b_output
        //          ∂L/∂W_output = ∂L/∂y · h^T
        //
        // We use simplified gradient flow: only update output and target heads
        // (This is sufficient for world model learning and avoids instability)

        let lr = self.config.learning_rate;

        // Recompute forward pass intermediates for gradient
        let mut input = Vec::with_capacity(state.len() + action.len());
        input.extend_from_slice(state);
        input.extend_from_slice(action);
        let expected_input_dim = 2 * self.config.hdc_dim;
        let input_arr = {
            let mut arr = Array1::zeros(expected_input_dim);
            let len = input.len().min(expected_input_dim);
            for i in 0..len {
                arr[i] = input[i];
            }
            arr
        };

        let projected = (self.w_input.dot(&input_arr) + &self.b_input).mapv(|v| v.tanh());
        let backbone_in = self.w_backbone.dot(&self.cfc_state) + &projected;
        let backbone_act = backbone_in.mapv(|v| v.tanh());
        let target_state = self.w_target.dot(&backbone_act) + &self.b_target;
        let decay_factor = (-delta_t / (&self.cfc_tau + 0.01)).mapv(|v| v.exp());
        let next_hidden = (&self.cfc_state - &target_state) * &decay_factor + &target_state;

        // ∂L/∂y = 2 * error / n (MSE gradient)
        let d_loss: Vec<f32> = error.iter().map(|&e| 2.0 * e / n as f32).collect();
        let d_loss_arr = Array1::from_vec(d_loss.clone());

        // ∂L/∂W_output = outer(∂L/∂y, h)
        // Update output weights
        for (i, &dl) in d_loss
            .iter()
            .enumerate()
            .take(self.w_output.nrows().min(d_loss.len()))
        {
            for j in 0..self.w_output.ncols() {
                self.w_output[[i, j]] -= lr * dl * next_hidden[j];
            }
            if i < self.b_output.len() {
                self.b_output[i] -= lr * dl;
            }
        }

        // ∂L/∂h = W_output^T · ∂L/∂y
        let d_hidden = self.w_output.t().dot(&d_loss_arr);

        // ∂h/∂A = 1 - decay_factor
        let d_output_d_A = 1.0 - &decay_factor;
        let d_loss_d_A = &d_hidden * &d_output_d_A;

        // ∂L/∂W_target = outer(∂L/∂A, backbone_act)
        for i in 0..self.w_target.nrows() {
            let grad_Ai = d_loss_d_A[i];
            for j in 0..self.w_target.ncols() {
                self.w_target[[i, j]] -= lr * grad_Ai * backbone_act[j];
            }
            self.b_target[i] -= lr * grad_Ai;
        }

        // Update consciousness metrics
        self.accumulated_surprise += mse;
        self.consciousness_level = self.config.consciousness_decay * self.consciousness_level
            + (1.0 - self.config.consciousness_decay) * mse;
        self.training_steps += 1;

        Ok(mse)
    }

    /// Train on a batch of experiences
    pub fn train_batch(
        &mut self,
        experiences: &[(Vec<f32>, Vec<f32>, Vec<f32>)], // (state, action, next_state)
        delta_t: f32,
    ) -> Result<f32> {
        let mut total_loss = 0.0;
        for (state, action, next_state) in experiences {
            total_loss += self.train_step(state, action, next_state, delta_t)?;
        }
        Ok(total_loss / experiences.len() as f32)
    }

    /// Get current consciousness level (EMA of prediction error)
    ///
    /// High consciousness = High surprise = Model doesn't predict well = Attention needed
    /// Low consciousness = Low surprise = Model predicts well = Habit/automatic
    pub fn consciousness(&self) -> f32 {
        self.consciousness_level
    }

    /// Get accumulated surprise (total prediction error over lifetime)
    pub fn total_surprise(&self) -> f32 {
        self.accumulated_surprise
    }

    /// Reset the CfC hidden state (e.g., at episode boundaries)
    pub fn reset_state(&mut self) {
        self.cfc_state.fill(0.0);
    }

    /// Imagine/dream: Generate a trajectory by recursive prediction
    ///
    /// This enables offline learning and planning by simulating
    /// future states without actual interaction.
    ///
    /// # Arguments
    /// * `initial_state` - Starting state (HDC)
    /// * `actions` - Sequence of actions to imagine taking
    /// * `delta_t` - Time step per action
    ///
    /// # Returns
    /// Trajectory of predicted states
    pub fn imagine(
        &mut self,
        initial_state: &[f32],
        actions: &[Vec<f32>],
        delta_t: f32,
    ) -> Result<Vec<Vec<f32>>> {
        let mut trajectory = Vec::with_capacity(actions.len() + 1);
        trajectory.push(initial_state.to_vec());

        let mut current_state = initial_state.to_vec();
        for action in actions {
            let next_state = self.predict_and_update(&current_state, action, delta_t)?;
            trajectory.push(next_state.clone());
            current_state = next_state;
        }

        Ok(trajectory)
    }

    /// Compute expected surprise for an action (for planning)
    ///
    /// Lower expected surprise = more predictable outcome = safer action
    pub fn expected_surprise(&self, state: &[f32], action: &[f32], delta_t: f32) -> Result<f32> {
        // Predict next state
        let prediction = self.predict(state, action, delta_t)?;

        // Use prediction variance as proxy for uncertainty
        let mean: f32 = prediction.iter().sum::<f32>() / prediction.len() as f32;
        let variance: f32 =
            prediction.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / prediction.len() as f32;

        Ok(variance)
    }

    /// Predict the approximate Φ (integration) of a state
    ///
    /// Uses variance/entropy as a fast, differentiable proxy for Integrated Information.
    /// High variance + High stability = High Φ.
    pub fn estimate_phi(&self, state: &[f32]) -> f32 {
        let mean = state.iter().sum::<f32>() / state.len() as f32;
        let variance = state.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / state.len() as f32;
        // Sigmoid activation of variance to map to [0, 1]
        1.0 / (1.0 + (-5.0 * (variance - 0.2)).exp())
    }

    /// Compute the "Value" of an action: G = -Expected_Surprise + λ * Expected_Phi
    ///
    /// This is the core of Consciousness-Driven Agency.
    /// The agent seeks actions that are predictable (safe) but lead to high-integration states (conscious).
    ///
    /// * `lambda`: The "Curiosity/Consciousness" weight. High λ = seeks consciousness.
    pub fn action_value(
        &self,
        state: &[f32],
        action: &[f32],
        delta_t: f32,
        lambda: f32,
    ) -> Result<f32> {
        let predicted_state = self.predict(state, action, delta_t)?;

        // 1. Expected Surprise (Uncertainty) - approximated by model uncertainty or just prediction error
        // Here we use a heuristic: if the state moves FAR from current, it's surprising.
        // Better: variance of prediction (if we had ensemble).
        // Simple proxy: Distance from equilibrium (A).
        let surprise = 0.1; // Placeholder for variance-based uncertainty

        // 2. Expected Phi (Integration)
        let expected_phi = self.estimate_phi(&predicted_state);

        // Value = Phi - Surprise (Maximize integration, Minimize error)
        Ok(lambda * expected_phi - surprise)
    }

    /// Serialize for persistence
    pub fn serialize(&self) -> Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }

    /// Deserialize from bytes
    pub fn deserialize(data: &[u8]) -> Result<Self> {
        Ok(bincode::deserialize(data)?)
    }
}

/// Experience buffer for online learning
#[derive(Clone, Debug, Default)]
pub struct ExperienceBuffer {
    /// Buffer of (state, action, next_state) tuples
    buffer: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)>,
    /// Maximum buffer size
    max_size: usize,
    /// Current position for circular buffer
    position: usize,
}

impl ExperienceBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(max_size),
            max_size,
            position: 0,
        }
    }

    /// Add experience to buffer
    pub fn push(&mut self, state: Vec<f32>, action: Vec<f32>, next_state: Vec<f32>) {
        if self.buffer.len() < self.max_size {
            self.buffer.push((state, action, next_state));
        } else {
            self.buffer[self.position] = (state, action, next_state);
        }
        self.position = (self.position + 1) % self.max_size;
    }

    /// Sample a random batch for training
    pub fn sample(&self, batch_size: usize) -> Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        let n = batch_size.min(self.buffer.len());
        self.buffer.choose_multiple(&mut rng, n).cloned().collect()
    }

    /// Get all experiences
    pub fn all(&self) -> &[(Vec<f32>, Vec<f32>, Vec<f32>)] {
        &self.buffer
    }

    /// Current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_world_model_predict() {
        let config = WorldModelConfig {
            hdc_dim: 128, // Small for testing
            hidden_dim: 32,
            ..Default::default()
        };
        let model = HierarchicalCfCWorldModel::new(config).unwrap();

        let state = vec![0.5; 128];
        let action = vec![0.1; 128];

        let prediction = model.predict(&state, &action, 1.0).unwrap();
        assert_eq!(prediction.len(), 128);
    }

    #[test]
    fn test_world_model_train_step() {
        let config = WorldModelConfig {
            hdc_dim: 128,
            hidden_dim: 32,
            learning_rate: 0.01,
            ..Default::default()
        };
        let mut model = HierarchicalCfCWorldModel::new(config).unwrap();

        let state = vec![0.5; 128];
        let action = vec![0.1; 128];
        let next_state = vec![0.6; 128];

        let loss1 = model.train_step(&state, &action, &next_state, 1.0).unwrap();

        // Train multiple times - loss should decrease
        for _ in 0..100 {
            model.train_step(&state, &action, &next_state, 1.0).unwrap();
        }
        let loss2 = model.train_step(&state, &action, &next_state, 1.0).unwrap();

        assert!(loss2 < loss1, "Loss should decrease with training");
    }

    #[test]
    fn test_consciousness_increases_with_surprise() {
        let config = WorldModelConfig {
            hdc_dim: 128,
            hidden_dim: 32,
            ..Default::default()
        };
        let mut model = HierarchicalCfCWorldModel::new(config).unwrap();

        let state = vec![0.5; 128];
        let action = vec![0.1; 128];
        let next_state = vec![0.6; 128];

        // Train with predictable data
        for _ in 0..50 {
            model.train_step(&state, &action, &next_state, 1.0).unwrap();
        }
        let consciousness_before = model.consciousness();

        // Introduce surprising data
        let surprising_next_state = vec![-0.9; 128];
        model
            .train_step(&state, &action, &surprising_next_state, 1.0)
            .unwrap();
        let consciousness_after = model.consciousness();

        assert!(
            consciousness_after > consciousness_before,
            "Consciousness should spike with surprising input"
        );
    }

    #[test]
    fn test_imagination_trajectory() {
        let config = WorldModelConfig {
            hdc_dim: 128,
            hidden_dim: 32,
            ..Default::default()
        };
        let mut model = HierarchicalCfCWorldModel::new(config).unwrap();

        let initial_state = vec![0.5; 128];
        let actions = vec![vec![0.1; 128], vec![0.2; 128], vec![0.3; 128]];

        let trajectory = model.imagine(&initial_state, &actions, 1.0).unwrap();

        assert_eq!(trajectory.len(), 4); // initial + 3 predicted
        for state in &trajectory {
            assert_eq!(state.len(), 128);
        }
    }

    #[test]
    fn test_experience_buffer() {
        let mut buffer = ExperienceBuffer::new(10);

        // Fill buffer
        for i in 0..15 {
            buffer.push(vec![i as f32], vec![i as f32], vec![i as f32]);
        }

        assert_eq!(buffer.len(), 10); // Circular buffer

        let batch = buffer.sample(5);
        assert_eq!(batch.len(), 5);
    }
}
