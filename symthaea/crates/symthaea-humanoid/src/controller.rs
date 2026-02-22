//! Humanoid controller: wraps HdcLtcUnifiedNetwork + output projection (16,384D → 21D).
//!
//! Uses the full 16,384D HDC-LTC temporal dynamics engine with a 3-layer × 8-neuron
//! architecture (deeper/wider than the drone's 2×4). All 21 actuator outputs use
//! tanh activation (bipolar [-1, 1] torques — no sigmoid needed).

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{
    ContinuousHV, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig, HDC_DIMENSION,
};

use crate::types::{HumanoidCommand, HumanoidConfig, NUM_ACTUATORS};

/// Humanoid controller wrapping an HdcLtcUnifiedNetwork + linear output head.
///
/// Forward pass:
/// 1. `network.evolve_closed_form(dt, &sensor_hv)` — O(D) temporal evolution
/// 2. `output = network.output()` — bundled final layer (16,384D)
/// 3. `output_weights @ output + output_bias` → 21D raw
/// 4. Activation: tanh for all actuators (bipolar torques)
pub struct HumanoidController {
    /// The temporal dynamics engine — full 16,384D HDC-LTC.
    network: HdcLtcUnifiedNetwork,
    /// Output projection weights: 21 rows × 16,384 columns (flat row-major).
    output_weights: Vec<f32>,
    /// Output bias (21D).
    output_bias: [f32; NUM_ACTUATORS],
    /// Current learning rate (modulated by FEP agent).
    learning_rate: f32,
}

impl HumanoidController {
    /// Create a new controller from a genesis seed and config.
    pub fn new(genesis: &GenesisSeed, config: &HumanoidConfig) -> Self {
        let neuron_config = UnifiedConfig {
            tau_base: 0.025,   // Matches 40Hz physics timestep
            backbone_tau: 0.3, // Moderate state dependency
            dimension: HDC_DIMENSION,
            learning_rate: config.learning_rate,
            ..UnifiedConfig::default()
        };

        let net_config = UnifiedNetworkConfig {
            layer_sizes: vec![config.neurons_per_layer; config.network_layers],
            neuron_config,
            use_layer_binding: true,
            skip_connections: false,
        };

        let network = HdcLtcUnifiedNetwork::from_genesis(net_config, genesis);

        // Initialize output weights from genesis (small values for stability)
        let total_weights = NUM_ACTUATORS * HDC_DIMENSION;
        let weight_hv = genesis.hv("humanoid::output_weights", total_weights);
        let mut output_weights = weight_hv.values;
        for w in &mut output_weights {
            *w *= 0.01;
        }

        // Bias initialized to zero (upright default pose)
        let output_bias = [0.0f32; NUM_ACTUATORS];

        Self {
            network,
            output_weights,
            output_bias,
            learning_rate: config.learning_rate,
        }
    }

    /// Create a controller with a pre-initialized network (e.g., from morphological transfer).
    ///
    /// The output projection is freshly generated from `genesis` (since the source domain
    /// has a different output dimensionality). Only the network backbone is reused.
    pub fn from_network(network: HdcLtcUnifiedNetwork, genesis: &GenesisSeed) -> Self {
        let total_weights = NUM_ACTUATORS * HDC_DIMENSION;
        let weight_hv = genesis.hv("humanoid::output_weights", total_weights);
        let mut output_weights = weight_hv.values;
        for w in &mut output_weights {
            *w *= 0.01;
        }
        let output_bias = [0.0f32; NUM_ACTUATORS];

        Self {
            network,
            output_weights,
            output_bias,
            learning_rate: 0.0005,
        }
    }

    /// Forward pass: evolve the network with sensor input and produce motor command.
    ///
    /// - `sensor_hv`: 16,384D ContinuousHV from the encoder
    /// - `dt`: timestep in seconds (typically 0.025 for 40Hz)
    pub fn forward(&mut self, sensor_hv: &ContinuousHV, dt: f32) -> HumanoidCommand {
        // 1. Evolve network dynamics
        self.network.evolve_closed_form(dt, sensor_hv);

        // 2. Get bundled final-layer output, normalized
        let output_hv = self.network.output().normalize();
        let hv_values = output_hv.as_slice();

        // 3. Linear projection: output_weights @ hv + bias → 21D
        let mut raw = [0.0f32; NUM_ACTUATORS];
        for i in 0..NUM_ACTUATORS {
            let row_offset = i * HDC_DIMENSION;
            let mut sum = self.output_bias[i];
            for j in 0..HDC_DIMENSION {
                sum += self.output_weights[row_offset + j] * hv_values[j];
            }
            raw[i] = sum;
        }

        // 4. Activation: tanh for all actuators (bipolar [-1, 1])
        let mut torques = [0.0f32; NUM_ACTUATORS];
        for i in 0..NUM_ACTUATORS {
            torques[i] = fast_tanh(raw[i]);
        }

        HumanoidCommand { torques }
    }

    /// Train the output projection and network weights via full BPTT.
    ///
    /// Uses `target` as the ground-truth command (from PD baseline).
    pub fn train_step(
        &mut self,
        sensor_hv: &ContinuousHV,
        target: &HumanoidCommand,
        dt: f32,
        lr_override: Option<f32>,
    ) {
        let lr = lr_override.unwrap_or(self.learning_rate);
        let output_lr = lr * (HDC_DIMENSION as f32).sqrt();

        let output_hv = self.network.output().normalize();
        let hv_values = output_hv.as_slice();

        // Compute current raw outputs
        let mut raw = [0.0f32; NUM_ACTUATORS];
        for i in 0..NUM_ACTUATORS {
            let row_offset = i * HDC_DIMENSION;
            let mut sum = self.output_bias[i];
            for j in 0..HDC_DIMENSION {
                sum += self.output_weights[row_offset + j] * hv_values[j];
            }
            raw[i] = sum;
        }

        // Compute activated outputs and errors
        let mut d_raw = [0.0f32; NUM_ACTUATORS];
        for i in 0..NUM_ACTUATORS {
            let pred = fast_tanh(raw[i]);
            let error = pred - target.torques[i];
            // Backprop through tanh: d/dx tanh(x) = 1 - tanh(x)^2
            let tanh_deriv = 1.0 - pred * pred;
            d_raw[i] = error * tanh_deriv;
        }

        // Gradient clipping
        const GRAD_CLIP: f32 = 1.0;
        for g in &mut d_raw {
            *g = g.clamp(-GRAD_CLIP, GRAD_CLIP);
        }

        // Weight decay
        const WEIGHT_DECAY: f32 = 1e-4;
        let decay = 1.0 - WEIGHT_DECAY;
        for w in self.output_weights.iter_mut() {
            *w *= decay;
        }

        // Update output weights: dW[i][j] = -output_lr * d_raw[i] * hv[j]
        for i in 0..NUM_ACTUATORS {
            let row_offset = i * HDC_DIMENSION;
            for j in 0..HDC_DIMENSION {
                self.output_weights[row_offset + j] -= output_lr * d_raw[i] * hv_values[j];
            }
            self.output_bias[i] -= output_lr * d_raw[i];
        }

        // Full BPTT through network layers
        let dim = HDC_DIMENSION;
        let mut grad_hv_values = vec![0.0f32; dim];
        for i in 0..NUM_ACTUATORS {
            let row_offset = i * dim;
            for j in 0..dim {
                grad_hv_values[j] += d_raw[i] * self.output_weights[row_offset + j];
            }
        }

        let n_layers = self.network.n_layers();
        let mut target_hv = output_hv.add(&ContinuousHV::from_vec(grad_hv_values).scale(-1.0));

        for layer_idx in (0..n_layers).rev() {
            let layer_input = self.network.layer_input(layer_idx, sensor_hv);

            let prev_layer_output = if layer_idx > 0 {
                self.network.output_at_layer(layer_idx - 1).cloned()
            } else {
                None
            };

            let mut avg_d_input = ContinuousHV::zero(dim);
            let mut neuron_count = 0usize;

            if let Some(layer) = self.network.layer_mut(layer_idx) {
                for neuron in layer.iter_mut() {
                    let grads = neuron.backward(&layer_input, &target_hv, dt);
                    avg_d_input = avg_d_input.add(&grads.d_input);
                    neuron_count += 1;
                    neuron.apply_gradients(&grads, lr);
                }
            }

            if let Some(prev_output) = prev_layer_output {
                if neuron_count > 0 {
                    let scale = 1.0 / neuron_count as f32;
                    target_hv = prev_output.subtract(&avg_d_input.scale(scale));
                }
            }
        }
    }

    /// Modulate all neuron time constants by a factor.
    pub fn modulate_tau(&mut self, factor: f32) {
        let factor = factor.clamp(0.3, 3.0);
        for layer_idx in 0..self.network.n_layers() {
            if let Some(layer) = self.network.layer_mut(layer_idx) {
                for neuron in layer.iter_mut() {
                    let new_tau = neuron.config().tau_base * factor;
                    neuron.set_tau_base(new_tau);
                }
            }
        }
    }

    /// Normalize all neuron hidden states to unit norm.
    pub fn normalize_states(&mut self) {
        for layer_idx in 0..self.network.n_layers() {
            if let Some(layer) = self.network.layer_mut(layer_idx) {
                for neuron in layer.iter_mut() {
                    let norm = neuron.state().norm();
                    if norm > 1.5 {
                        let normalized = neuron.state().normalize();
                        *neuron.state_mut() = normalized;
                    }
                }
            }
        }
    }

    /// Warmup: pre-train the output projection on static (sensor_hv, target) pairs.
    pub fn warmup(&mut self, samples: &[(ContinuousHV, HumanoidCommand)], n_steps: usize, lr: f32) {
        for step in 0..n_steps {
            let (ref hv, ref target) = samples[step % samples.len()];
            self.forward(hv, 0.025);
            self.train_step(hv, target, 0.025, Some(lr));
        }
        self.network.reset();
    }

    /// Set the learning rate directly.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr.clamp(1e-6, 0.1);
    }

    /// Get current learning rate.
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Reset network state (for new episode).
    pub fn reset(&mut self) {
        self.network.reset();
    }

    /// Get network statistics.
    pub fn stats(&self) -> symthaea_core::hdc::UnifiedNetworkStats {
        self.network.stats()
    }

    /// Get the underlying network (for inspection).
    pub fn network(&self) -> &HdcLtcUnifiedNetwork {
        &self.network
    }

    /// Mask a specific actuator output to zero (for phantom limb perturbation).
    /// Returns the previous torque at that index before masking.
    pub fn mask_actuator(cmd: &mut HumanoidCommand, joint_index: usize) {
        if joint_index < NUM_ACTUATORS {
            cmd.torques[joint_index] = 0.0;
        }
    }
}

/// Fast tanh approximation (Pade).
fn fast_tanh(x: f32) -> f32 {
    if x.abs() > 4.97 {
        x.signum()
    } else {
        let x2 = x * x;
        x * (27.0 + x2) / (27.0 + 9.0 * x2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-humanoid-controller")
    }

    #[test]
    fn test_controller_forward_valid_output() {
        let genesis = test_genesis();
        let config = HumanoidConfig::default();
        let mut ctrl = HumanoidController::new(&genesis, &config);

        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        let cmd = ctrl.forward(&sensor, 0.025);

        for i in 0..NUM_ACTUATORS {
            assert!(
                cmd.torques[i] >= -1.0 && cmd.torques[i] <= 1.0,
                "Torque {i} out of range: {}",
                cmd.torques[i]
            );
        }
    }

    #[test]
    fn test_controller_genesis_determinism() {
        let genesis = test_genesis();
        let config = HumanoidConfig::default();
        let mut ctrl1 = HumanoidController::new(&genesis, &config);
        let mut ctrl2 = HumanoidController::new(&genesis, &config);

        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        let cmd1 = ctrl1.forward(&sensor, 0.025);
        let cmd2 = ctrl2.forward(&sensor, 0.025);

        for i in 0..NUM_ACTUATORS {
            assert!(
                (cmd1.torques[i] - cmd2.torques[i]).abs() < 1e-6,
                "Same genesis -> same output for joint {i}: {} vs {}",
                cmd1.torques[i],
                cmd2.torques[i]
            );
        }
    }

    #[test]
    fn test_controller_training_reduces_error() {
        let genesis = test_genesis();
        let config = HumanoidConfig::default();
        let mut ctrl = HumanoidController::new(&genesis, &config);

        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        // Use a non-trivial target so initial error is meaningful
        let mut target_torques = [0.0f32; NUM_ACTUATORS];
        for i in 0..NUM_ACTUATORS {
            target_torques[i] = ((i as f32 * 0.3).sin()) * 0.5;
        }
        let target = HumanoidCommand {
            torques: target_torques,
        };

        // Initial error
        let initial = ctrl.forward(&sensor, 0.025);
        let initial_err: f32 = initial
            .torques
            .iter()
            .zip(target.torques.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();

        // Train with lower LR for stability (21 outputs vs 4 is harder)
        for _ in 0..100 {
            ctrl.forward(&sensor, 0.025);
            ctrl.train_step(&sensor, &target, 0.025, Some(0.001));
        }

        // Final error
        let final_cmd = ctrl.forward(&sensor, 0.025);
        let final_err: f32 = final_cmd
            .torques
            .iter()
            .zip(target.torques.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();

        assert!(
            final_err < initial_err,
            "Training should reduce error: initial={initial_err:.6}, final={final_err:.6}"
        );
    }

    #[test]
    fn test_controller_reset() {
        let genesis = test_genesis();
        let config = HumanoidConfig::default();
        let mut ctrl = HumanoidController::new(&genesis, &config);

        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        ctrl.forward(&sensor, 0.025);
        ctrl.forward(&sensor, 0.025);

        ctrl.reset();
        let stats = ctrl.stats();
        assert!(stats.avg_state_norm < 1e-6, "Reset should zero state norms");
    }

    #[test]
    fn test_fast_tanh_range() {
        assert!(fast_tanh(0.0).abs() < 1e-6);
        assert!((fast_tanh(10.0) - 1.0).abs() < 0.01);
        assert!((fast_tanh(-10.0) + 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mask_actuator() {
        let mut cmd = HumanoidCommand {
            torques: [0.5; NUM_ACTUATORS],
        };
        HumanoidController::mask_actuator(&mut cmd, 6); // right_knee
        assert!((cmd.torques[6] - 0.0).abs() < 1e-6);
        assert!((cmd.torques[5] - 0.5).abs() < 1e-6); // neighbor unchanged
    }

    #[test]
    fn test_output_dimension() {
        let genesis = test_genesis();
        let config = HumanoidConfig::default();
        let mut ctrl = HumanoidController::new(&genesis, &config);

        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        let cmd = ctrl.forward(&sensor, 0.025);
        assert_eq!(cmd.torques.len(), NUM_ACTUATORS);
        assert_eq!(cmd.to_ctrl().len(), NUM_ACTUATORS);
    }
}
