// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Vehicle controller: wraps HdcLtcUnifiedNetwork + output projection (16,384D → 3D).
//!
//! Uses the full 16,384D HDC-LTC temporal dynamics engine with a 3-layer × 8-neuron
//! architecture. Output activations:
//! - Steering: tanh → [-1, 1] (bipolar)
//! - Throttle: half_tanh → [0, 1] (unipolar)
//! - Brake: half_tanh → [0, 1] (unipolar)

use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{
    ContinuousHV, HDC_DIMENSION, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig,
};

use crate::types::{ACT_STEERING, NUM_ACTUATORS, VehicleCommand, VehicleConfig};

/// Serializable checkpoint for saving/loading trained controllers.
#[derive(Serialize, Deserialize)]
pub struct ControllerCheckpoint {
    /// Output projection weights (3 × 16,384 flat row-major).
    pub output_weights: Vec<f32>,
    /// Output bias (3D).
    pub output_bias: Vec<f32>,
    /// Learned learning rate.
    pub learning_rate: f32,
    /// Genesis phrase for network reconstruction.
    pub genesis_phrase: String,
    /// Network layers config.
    pub network_layers: usize,
    /// Neurons per layer config.
    pub neurons_per_layer: usize,
}

/// Vehicle controller wrapping an HdcLtcUnifiedNetwork + linear output head.
///
/// Forward pass:
/// 1. `network.evolve_closed_form(dt, &sensor_hv)` — O(D) temporal evolution
/// 2. `output = network.output()` — bundled final layer (16,384D)
/// 3. `output_weights @ output + output_bias` → 3D raw
/// 4. Activation: tanh for steering, half_tanh for throttle/brake
pub struct VehicleController {
    /// The temporal dynamics engine — full 16,384D HDC-LTC.
    network: HdcLtcUnifiedNetwork,
    /// Output projection weights: 3 rows × 16,384 columns (flat row-major).
    output_weights: Vec<f32>,
    /// Output bias (3D).
    output_bias: [f32; NUM_ACTUATORS],
    /// Current learning rate (modulated by FEP agent).
    learning_rate: f32,
    /// Temporary LR multiplier.
    lr_scale: f32,
}

impl VehicleController {
    /// Create a new controller from a genesis seed and config.
    pub fn new(genesis: &GenesisSeed, config: &VehicleConfig) -> Self {
        let neuron_config = UnifiedConfig {
            tau_base: 0.005,   // Matches 200Hz physics timestep
            backbone_tau: 0.2, // Faster adaptation than humanoid (driving is reactive)
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
        let weight_hv = genesis.hv("vehicle::output_weights", total_weights);
        let mut output_weights = weight_hv.values;
        for w in &mut output_weights {
            *w *= 0.01;
        }

        let output_bias = [0.0f32; NUM_ACTUATORS];

        Self {
            network,
            output_weights,
            output_bias,
            learning_rate: config.learning_rate,
            lr_scale: 1.0,
        }
    }

    /// Forward pass: evolve the network with sensor input and produce motor command.
    pub fn forward(&mut self, sensor_hv: &ContinuousHV, dt: f32) -> VehicleCommand {
        // 1. Evolve network dynamics
        self.network.evolve_closed_form(dt, sensor_hv);

        // 2. Get bundled final-layer output, normalized
        let output_hv = self.network.output().normalize();
        let hv_values = output_hv.as_slice();

        // 3. Linear projection: output_weights @ hv + bias → 3D
        let mut raw = [0.0f32; NUM_ACTUATORS];
        for i in 0..NUM_ACTUATORS {
            let row_offset = i * HDC_DIMENSION;
            let mut sum = self.output_bias[i];
            for j in 0..HDC_DIMENSION {
                sum += self.output_weights[row_offset + j] * hv_values[j];
            }
            raw[i] = sum;
        }

        // 4. Activation: tanh for steering, half_tanh for throttle/brake
        VehicleCommand {
            steering: fast_tanh(raw[ACT_STEERING]),
            throttle: half_tanh(raw[1]),
            brake: half_tanh(raw[2]),
        }
    }

    /// Train the output projection and network weights via full BPTT.
    pub fn train_step(
        &mut self,
        sensor_hv: &ContinuousHV,
        target: &VehicleCommand,
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

        // Target values for each actuator
        let target_vals = [target.steering, target.throttle, target.brake];

        // Compute activated outputs and errors with per-actuator activation
        let mut d_raw = [0.0f32; NUM_ACTUATORS];
        for i in 0..NUM_ACTUATORS {
            let (pred, deriv) = if i == ACT_STEERING {
                let t = fast_tanh(raw[i]);
                (t, 1.0 - t * t) // tanh derivative
            } else {
                let h = half_tanh(raw[i]);
                let t = fast_tanh(raw[i]);
                (h, (1.0 - t * t) * 0.5) // half_tanh derivative
            };
            let error = pred - target_vals[i];
            d_raw[i] = error * deriv;
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

        // Update output weights
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
    pub fn warmup(&mut self, samples: &[(ContinuousHV, VehicleCommand)], n_steps: usize, lr: f32) {
        for step in 0..n_steps {
            let (ref hv, ref target) = samples[step % samples.len()];
            self.forward(hv, 0.005);
            self.train_step(hv, target, 0.005, Some(lr));
        }
        self.network.reset();
    }

    /// Set the learning rate directly.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr.clamp(1e-6, 0.1);
    }

    /// Get current effective learning rate (base × scale).
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate * self.lr_scale
    }

    /// Set a temporary learning rate multiplier.
    pub fn set_learning_rate_scale(&mut self, scale: f32) {
        self.lr_scale = scale.max(0.0);
    }

    /// Get a copy of the current output projection.
    pub fn output_projection(&self) -> (Vec<f32>, Vec<f32>) {
        (self.output_weights.clone(), self.output_bias.to_vec())
    }

    /// Restore the output projection from a saved copy.
    pub fn set_output_projection(&mut self, weights: &[f32], bias: &[f32]) {
        if weights.len() == self.output_weights.len() {
            self.output_weights.copy_from_slice(weights);
        }
        let n = bias.len().min(NUM_ACTUATORS);
        self.output_bias[..n].copy_from_slice(&bias[..n]);
    }

    /// Reset network state (for new episode).
    pub fn reset(&mut self) {
        self.network.reset();
    }

    /// Get network statistics.
    pub fn stats(&self) -> symthaea_core::hdc::UnifiedNetworkStats {
        self.network.stats()
    }

    /// Save a checkpoint.
    pub fn save_checkpoint(&self, path: &str, config: &VehicleConfig) -> std::io::Result<()> {
        let checkpoint = ControllerCheckpoint {
            output_weights: self.output_weights.clone(),
            output_bias: self.output_bias.to_vec(),
            learning_rate: self.learning_rate,
            genesis_phrase: config.genesis_phrase.clone(),
            network_layers: config.network_layers,
            neurons_per_layer: config.neurons_per_layer,
        };
        let json = serde_json::to_string_pretty(&checkpoint).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// Load a checkpoint and reconstruct the controller.
    pub fn load_checkpoint(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let checkpoint: ControllerCheckpoint =
            serde_json::from_str(&json).map_err(std::io::Error::other)?;

        let genesis = GenesisSeed::from_phrase(&checkpoint.genesis_phrase);
        let config = VehicleConfig {
            network_layers: checkpoint.network_layers,
            neurons_per_layer: checkpoint.neurons_per_layer,
            learning_rate: checkpoint.learning_rate,
            genesis_phrase: checkpoint.genesis_phrase,
            ..VehicleConfig::default()
        };

        let neuron_config = UnifiedConfig {
            tau_base: 0.005,
            backbone_tau: 0.2,
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

        let network = HdcLtcUnifiedNetwork::from_genesis(net_config, &genesis);

        let mut output_bias = [0.0f32; NUM_ACTUATORS];
        for (i, &v) in checkpoint
            .output_bias
            .iter()
            .enumerate()
            .take(NUM_ACTUATORS)
        {
            output_bias[i] = v;
        }

        Ok(Self {
            network,
            output_weights: checkpoint.output_weights,
            output_bias,
            learning_rate: checkpoint.learning_rate,
            lr_scale: 1.0,
        })
    }
}

/// Fast tanh approximation (Pade). Maps R → [-1, 1].
fn fast_tanh(x: f32) -> f32 {
    if x.abs() > 4.97 {
        x.signum()
    } else {
        let x2 = x * x;
        x * (27.0 + x2) / (27.0 + 9.0 * x2)
    }
}

/// Half-tanh: maps R → [0, 1]. Used for throttle and brake.
/// half_tanh(x) = (tanh(x) + 1) / 2
fn half_tanh(x: f32) -> f32 {
    (fast_tanh(x) + 1.0) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-vehicle-controller")
    }

    #[test]
    fn test_controller_forward_valid_output() {
        let genesis = test_genesis();
        let config = VehicleConfig::default();
        let mut ctrl = VehicleController::new(&genesis, &config);

        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        let cmd = ctrl.forward(&sensor, 0.005);

        assert!(
            cmd.steering >= -1.0 && cmd.steering <= 1.0,
            "Steering: {}",
            cmd.steering
        );
        assert!(
            cmd.throttle >= 0.0 && cmd.throttle <= 1.0,
            "Throttle: {}",
            cmd.throttle
        );
        assert!(cmd.brake >= 0.0 && cmd.brake <= 1.0, "Brake: {}", cmd.brake);
    }

    #[test]
    fn test_controller_genesis_determinism() {
        let genesis = test_genesis();
        let config = VehicleConfig::default();
        let mut ctrl1 = VehicleController::new(&genesis, &config);
        let mut ctrl2 = VehicleController::new(&genesis, &config);

        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        let cmd1 = ctrl1.forward(&sensor, 0.005);
        let cmd2 = ctrl2.forward(&sensor, 0.005);

        assert!((cmd1.steering - cmd2.steering).abs() < 1e-6);
        assert!((cmd1.throttle - cmd2.throttle).abs() < 1e-6);
        assert!((cmd1.brake - cmd2.brake).abs() < 1e-6);
    }

    #[test]
    fn test_controller_training_reduces_error() {
        let genesis = test_genesis();
        let config = VehicleConfig::default();
        let mut ctrl = VehicleController::new(&genesis, &config);

        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        let target = VehicleCommand {
            steering: 0.3,
            throttle: 0.6,
            brake: 0.0,
        };

        // Initial error
        let initial = ctrl.forward(&sensor, 0.005);
        let initial_err = (initial.steering - target.steering).powi(2)
            + (initial.throttle - target.throttle).powi(2)
            + (initial.brake - target.brake).powi(2);

        // Train
        for _ in 0..100 {
            ctrl.forward(&sensor, 0.005);
            ctrl.train_step(&sensor, &target, 0.005, Some(0.001));
        }

        // Final error
        let final_cmd = ctrl.forward(&sensor, 0.005);
        let final_err = (final_cmd.steering - target.steering).powi(2)
            + (final_cmd.throttle - target.throttle).powi(2)
            + (final_cmd.brake - target.brake).powi(2);

        assert!(
            final_err < initial_err,
            "Training should reduce error: initial={initial_err:.6}, final={final_err:.6}"
        );
    }

    #[test]
    fn test_controller_reset() {
        let genesis = test_genesis();
        let config = VehicleConfig::default();
        let mut ctrl = VehicleController::new(&genesis, &config);

        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        ctrl.forward(&sensor, 0.005);
        ctrl.forward(&sensor, 0.005);

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
    fn test_half_tanh_range() {
        assert!((half_tanh(0.0) - 0.5).abs() < 1e-6);
        assert!((half_tanh(10.0) - 1.0).abs() < 0.01);
        assert!((half_tanh(-10.0) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_output_dimension() {
        let genesis = test_genesis();
        let config = VehicleConfig::default();
        let mut ctrl = VehicleController::new(&genesis, &config);

        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        let cmd = ctrl.forward(&sensor, 0.005);
        assert_eq!(cmd.to_ctrl().len(), NUM_ACTUATORS);
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let config = VehicleConfig::default();
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let ctrl = VehicleController::new(&genesis, &config);

        let path = "/tmp/symthaea_vehicle_checkpoint_test.json";
        ctrl.save_checkpoint(path, &config).unwrap();

        let mut original = VehicleController::new(&genesis, &config);
        let mut loaded = VehicleController::load_checkpoint(path).unwrap();

        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        let cmd1 = original.forward(&sensor, 0.005);
        let cmd2 = loaded.forward(&sensor, 0.005);

        assert!((cmd1.steering - cmd2.steering).abs() < 1e-6);
        assert!((cmd1.throttle - cmd2.throttle).abs() < 1e-6);
        assert!((cmd1.brake - cmd2.brake).abs() < 1e-6);

        let _ = std::fs::remove_file(path);
    }
}
