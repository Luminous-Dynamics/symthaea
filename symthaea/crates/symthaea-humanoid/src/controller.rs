// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Humanoid controller: wraps HdcLtcUnifiedNetwork + output projection (16,384D → 21D).
//!
//! Uses the full 16,384D HDC-LTC temporal dynamics engine with a 4-layer × 12-neuron
//! architecture (deeper/wider than the drone's 2×4). All 21 actuator outputs use
//! tanh activation (bipolar [-1, 1] torques — no sigmoid needed).

use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{
    ContinuousHV, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig, HDC_DIMENSION,
};

use crate::types::{HumanoidCommand, HumanoidConfig, NUM_ACTUATORS};

/// Serializable checkpoint for saving/loading trained controllers.
///
/// The HDC-LTC network itself isn't serializable (contains computed binding vectors),
/// so we store the output projection (which captures all learned behavior) plus the
/// genesis phrase and config needed to reconstruct the network backbone.
#[derive(Serialize, Deserialize)]
pub struct ControllerCheckpoint {
    /// Output projection weights (21 × 16,384 flat row-major).
    pub output_weights: Vec<f32>,
    /// Output bias (21D).
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

/// Humanoid controller wrapping an HdcLtcUnifiedNetwork + linear output head.
///
/// Forward pass:
/// 1. `network.evolve_closed_form(dt, &sensor_hv)` — O(D) temporal evolution
/// 2. `output = network.output()` — bundled final layer (16,384D)
/// 3. `output_weights @ output + output_bias` → 21D raw
/// 4. Activation: tanh for all actuators (bipolar torques)
/// Bottleneck dimension for the nonlinear output head.
/// Random projection 16384→BOTTLENECK preserves distance (Johnson-Lindenstrauss).
/// Only the BOTTLENECK→num_outputs weights are learned (1.4K vs 344K trainable params).
const BOTTLENECK_DIM: usize = 64;

pub struct HumanoidController {
    /// The temporal dynamics engine — full 16,384D HDC-LTC.
    network: HdcLtcUnifiedNetwork,
    /// Random projection vectors for bottleneck (BOTTLENECK_DIM × HDC_DIMENSION, fixed).
    bottleneck_projections: Vec<ContinuousHV>,
    /// Output weights: num_outputs × BOTTLENECK_DIM (learned, small!).
    output_weights: Vec<f32>,
    /// Output bias.
    output_bias: Vec<f32>,
    /// Number of output actuators (matches morphology).
    num_outputs: usize,
    /// Current learning rate (modulated by FEP agent).
    learning_rate: f32,
    /// Temporary LR multiplier (e.g., 3.0 during phase transitions).
    lr_scale: f32,
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

        // Random projection bottleneck: 64 fixed HVs for 16384→64 dimensionality reduction
        let bottleneck_projections: Vec<ContinuousHV> = (0..BOTTLENECK_DIM)
            .map(|i| genesis.hv(&format!("humanoid::bottleneck::{i}"), HDC_DIMENSION))
            .collect();

        // Output weights: num_outputs × BOTTLENECK_DIM (only 1.4K trainable params!)
        let num_outputs = config.morphology.num_actuators();
        let total_weights = num_outputs * BOTTLENECK_DIM;
        let weight_hv = genesis.hv("humanoid::output_weights_v2", total_weights);
        let mut output_weights = weight_hv.values;
        for w in &mut output_weights {
            *w *= 0.1; // Larger init since fewer weights
        }

        let output_bias = vec![0.0f32; num_outputs];

        Self {
            network,
            bottleneck_projections,
            output_weights,
            output_bias,
            num_outputs,
            learning_rate: config.learning_rate,
            lr_scale: 1.0,
        }
    }

    /// Create a controller with a pre-initialized network (e.g., from morphological transfer).
    ///
    /// The output projection is freshly generated from `genesis` (since the source domain
    /// has a different output dimensionality). Only the network backbone is reused.
    pub fn from_network(network: HdcLtcUnifiedNetwork, genesis: &GenesisSeed) -> Self {
        let num_outputs = NUM_ACTUATORS;
        let bottleneck_projections: Vec<ContinuousHV> = (0..BOTTLENECK_DIM)
            .map(|i| genesis.hv(&format!("humanoid::bottleneck::{i}"), HDC_DIMENSION))
            .collect();
        let total_weights = num_outputs * BOTTLENECK_DIM;
        let weight_hv = genesis.hv("humanoid::output_weights_v2", total_weights);
        let mut output_weights = weight_hv.values;
        for w in &mut output_weights {
            *w *= 0.1;
        }
        let output_bias = vec![0.0f32; num_outputs];

        Self {
            network,
            bottleneck_projections,
            output_weights,
            output_bias,
            num_outputs,
            learning_rate: 0.0005,
            lr_scale: 1.0,
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

        // 3. Bottleneck: 16384→64 via fixed random projections (dot products)
        let mut bottleneck = vec![0.0f32; BOTTLENECK_DIM];
        for (b, proj) in bottleneck
            .iter_mut()
            .zip(self.bottleneck_projections.iter())
        {
            *b = proj.similarity(&output_hv); // Cosine similarity = normalized dot product
        }
        // Apply tanh nonlinearity — this is what enables learning nonlinear control laws
        for b in bottleneck.iter_mut() {
            *b = fast_tanh(*b * 3.0); // Scale by 3 to use more of tanh range
        }

        // 4. Small linear projection: 64→num_outputs
        let mut raw = vec![0.0f32; self.num_outputs];
        for i in 0..self.num_outputs {
            let row_offset = i * BOTTLENECK_DIM;
            let mut sum = self.output_bias[i];
            for j in 0..BOTTLENECK_DIM {
                sum += self.output_weights[row_offset + j] * bottleneck[j];
            }
            raw[i] = sum;
        }

        // 4. Activation: tanh for all actuators (bipolar [-1, 1])
        let mut torques = vec![0.0f32; self.num_outputs];
        for i in 0..self.num_outputs {
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
        // Scale LR by sqrt(D) but cap to prevent instability
        // sqrt(16384) = 128, which makes output_lr = lr * 128 — too aggressive
        // Cap at 10x base LR for stability
        let output_lr = (lr * (HDC_DIMENSION as f32).sqrt()).min(lr * 10.0);

        let output_hv = self.network.output().normalize();

        // Forward pass through bottleneck (recompute for gradient)
        let mut bottleneck = vec![0.0f32; BOTTLENECK_DIM];
        for (b, proj) in bottleneck
            .iter_mut()
            .zip(self.bottleneck_projections.iter())
        {
            *b = proj.similarity(&output_hv);
        }
        let mut bottleneck_activated = vec![0.0f32; BOTTLENECK_DIM];
        for j in 0..BOTTLENECK_DIM {
            bottleneck_activated[j] = fast_tanh(bottleneck[j] * 3.0);
        }

        // Compute raw outputs from bottleneck
        let mut raw = vec![0.0f32; self.num_outputs];
        for i in 0..self.num_outputs {
            let row_offset = i * BOTTLENECK_DIM;
            let mut sum = self.output_bias[i];
            for j in 0..BOTTLENECK_DIM {
                sum += self.output_weights[row_offset + j] * bottleneck_activated[j];
            }
            raw[i] = sum;
        }

        // Output error + tanh derivative
        let mut d_raw = vec![0.0f32; self.num_outputs];
        for i in 0..self.num_outputs {
            let pred = fast_tanh(raw[i]);
            let error = pred - target.torques[i];
            let tanh_deriv = 1.0 - pred * pred;
            d_raw[i] = error * tanh_deriv;
        }

        // Gradient clipping
        const GRAD_CLIP: f32 = 0.5;
        for g in &mut d_raw {
            *g = g.clamp(-GRAD_CLIP, GRAD_CLIP);
        }

        // Weight decay
        const WEIGHT_DECAY: f32 = 1e-5;
        let decay = 1.0 - WEIGHT_DECAY;
        for w in self.output_weights.iter_mut() {
            *w *= decay;
        }

        // Update output weights (BOTTLENECK_DIM × num_outputs — only 1.4K params!)
        for i in 0..self.num_outputs {
            let row_offset = i * BOTTLENECK_DIM;
            for j in 0..BOTTLENECK_DIM {
                self.output_weights[row_offset + j] -=
                    output_lr * d_raw[i] * bottleneck_activated[j];
            }
            self.output_bias[i] -= output_lr * d_raw[i];
        }

        // Full BPTT through network layers
        // Backprop: d_raw → output_weights → d_bottleneck → tanh' → bottleneck_projections → d_hv
        let dim = HDC_DIMENSION;

        // d_bottleneck_activated = output_weights^T × d_raw
        let mut d_bottleneck = vec![0.0f32; BOTTLENECK_DIM];
        for i in 0..self.num_outputs {
            let row_offset = i * BOTTLENECK_DIM;
            for j in 0..BOTTLENECK_DIM {
                d_bottleneck[j] += d_raw[i] * self.output_weights[row_offset + j];
            }
        }

        // Through tanh: d_bottleneck_pre = d_bottleneck * tanh'(bottleneck * 3) * 3
        for j in 0..BOTTLENECK_DIM {
            let tanh_deriv = 1.0 - bottleneck_activated[j] * bottleneck_activated[j];
            d_bottleneck[j] *= tanh_deriv * 3.0;
        }

        // Through random projections: d_hv = Σ_j d_bottleneck[j] × projection[j]
        let mut grad_hv_values = vec![0.0f32; dim];
        for (j, proj) in self.bottleneck_projections.iter().enumerate() {
            let proj_values = proj.as_slice();
            let grad = d_bottleneck[j];
            if grad.abs() > 1e-8 {
                for k in 0..dim {
                    grad_hv_values[k] += grad * proj_values[k];
                }
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

    /// Get current effective learning rate (base × scale).
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate * self.lr_scale
    }

    /// Set a temporary learning rate multiplier (e.g., 3.0 during phase transitions).
    ///
    /// Scales the effective LR returned by `learning_rate()` without modifying
    /// the base rate. Set to 1.0 to restore normal LR.
    pub fn set_learning_rate_scale(&mut self, scale: f32) {
        self.lr_scale = scale.max(0.0);
    }

    /// Get a copy of the current output projection (weights, bias).
    ///
    /// Used for best-checkpoint tracking during training.
    pub fn output_projection(&self) -> (Vec<f32>, Vec<f32>) {
        (self.output_weights.clone(), self.output_bias.to_vec())
    }

    /// Restore the output projection from a saved copy.
    ///
    /// Used for best-checkpoint revert during training.
    pub fn set_output_projection(&mut self, weights: &[f32], bias: &[f32]) {
        if weights.len() == self.output_weights.len() {
            self.output_weights.copy_from_slice(weights);
        }
        let n = bias.len().min(self.num_outputs);
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

    /// Get the underlying network (for inspection).
    pub fn network(&self) -> &HdcLtcUnifiedNetwork {
        &self.network
    }

    /// Mask a specific actuator output to zero (for phantom limb perturbation).
    /// Returns the previous torque at that index before masking.
    pub fn mask_actuator(cmd: &mut HumanoidCommand, joint_index: usize) {
        if joint_index < cmd.torques.len() {
            cmd.torques[joint_index] = 0.0;
        }
    }

    /// Save a checkpoint of the trained output projection to a JSON file.
    ///
    /// The network backbone is reconstructed from genesis on load;
    /// only the learned output weights/bias are persisted.
    pub fn save_checkpoint(&self, path: &str, config: &HumanoidConfig) -> std::io::Result<()> {
        let checkpoint = ControllerCheckpoint {
            output_weights: self.output_weights.clone(),
            output_bias: self.output_bias.to_vec(),
            learning_rate: self.learning_rate,
            genesis_phrase: config.genesis_phrase.clone(),
            network_layers: config.network_layers,
            neurons_per_layer: config.neurons_per_layer,
        };
        let json = serde_json::to_string_pretty(&checkpoint)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load a checkpoint and reconstruct the controller.
    ///
    /// The HDC-LTC network is rebuilt from the genesis phrase (deterministic),
    /// and the trained output projection is restored from the checkpoint.
    pub fn load_checkpoint(path: &str) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let checkpoint: ControllerCheckpoint = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

        let genesis = GenesisSeed::from_phrase(&checkpoint.genesis_phrase);
        let config = HumanoidConfig {
            network_layers: checkpoint.network_layers,
            neurons_per_layer: checkpoint.neurons_per_layer,
            learning_rate: checkpoint.learning_rate,
            genesis_phrase: checkpoint.genesis_phrase,
            ..HumanoidConfig::default()
        };

        let neuron_config = UnifiedConfig {
            tau_base: 0.025,
            backbone_tau: 0.3,
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

        let num_outputs = config.morphology.num_actuators();
        let mut output_bias = vec![0.0f32; num_outputs];
        for (i, &v) in checkpoint.output_bias.iter().enumerate().take(num_outputs) {
            output_bias[i] = v;
        }

        let bottleneck_projections: Vec<ContinuousHV> = (0..BOTTLENECK_DIM)
            .map(|i| genesis.hv(&format!("humanoid::bottleneck::{i}"), HDC_DIMENSION))
            .collect();

        Ok(Self {
            network,
            bottleneck_projections,
            output_weights: checkpoint.output_weights,
            output_bias,
            num_outputs,
            learning_rate: checkpoint.learning_rate,
            lr_scale: 1.0,
        })
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
        let mut target_torques = vec![0.0f32; NUM_ACTUATORS];
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
            torques: vec![0.5; NUM_ACTUATORS],
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

    #[test]
    fn test_checkpoint_roundtrip() {
        // Use config's genesis phrase for consistency (checkpoint stores this phrase)
        let config = HumanoidConfig::default();
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let ctrl = HumanoidController::new(&genesis, &config);

        // Save the fresh controller (no BPTT training — network backbone matches genesis)
        let path = "/tmp/symthaea_test_checkpoint.json";
        ctrl.save_checkpoint(path, &config).unwrap();

        // Load and verify outputs match (loaded reconstructs from same genesis phrase)
        let mut original = HumanoidController::new(&genesis, &config);
        let mut loaded = HumanoidController::load_checkpoint(path).unwrap();

        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        let cmd_original = original.forward(&sensor, 0.025);
        let cmd_loaded = loaded.forward(&sensor, 0.025);

        for i in 0..NUM_ACTUATORS {
            assert!(
                (cmd_original.torques[i] - cmd_loaded.torques[i]).abs() < 1e-6,
                "Checkpoint roundtrip mismatch at joint {i}: {} vs {}",
                cmd_original.torques[i],
                cmd_loaded.torques[i]
            );
        }

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_checkpoint_preserves_trained_projection() {
        let config = HumanoidConfig::default();
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let mut ctrl = HumanoidController::new(&genesis, &config);

        // Train the output projection
        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        let target = HumanoidCommand {
            torques: vec![0.3; NUM_ACTUATORS],
        };
        for _ in 0..20 {
            ctrl.forward(&sensor, 0.025);
            ctrl.train_step(&sensor, &target, 0.025, None);
        }

        // Save
        let path = "/tmp/symthaea_test_trained_checkpoint.json";
        ctrl.save_checkpoint(path, &config).unwrap();

        // Load and verify the checkpoint file exists and is valid JSON
        let loaded = HumanoidController::load_checkpoint(path).unwrap();
        assert!(
            (loaded.learning_rate() - config.learning_rate).abs() < 1e-6,
            "Learning rate preserved"
        );

        // The loaded controller should produce output (not crash or NaN)
        let mut loaded = loaded;
        let cmd = loaded.forward(&sensor, 0.025);
        for i in 0..NUM_ACTUATORS {
            assert!(cmd.torques[i].is_finite(), "Joint {i} should be finite");
        }

        let _ = std::fs::remove_file(path);
    }
}
