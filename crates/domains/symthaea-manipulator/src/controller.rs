// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Manipulator controller: HdcLtcUnifiedNetwork + output projection (16,384D → 8D).

use crate::types::{ManipulatorCommand, ManipulatorConfig, NUM_JOINTS};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig};

const HDC_DIM: usize = symthaea_core::hdc::HDC_DIMENSION;
const NUM_OUTPUTS: usize = NUM_JOINTS + 1; // 7 joints + 1 gripper

pub struct ManipulatorController {
    network: HdcLtcUnifiedNetwork,
    output_weights: Vec<f32>,
    output_bias: [f32; NUM_OUTPUTS],
    learning_rate: f32,
    /// Cached final-layer HV from the last forward() (post-normalize) --
    /// needed by train_step's delta rule.
    last_features: Vec<f32>,
    /// Cached post-activation outputs from the last forward() (tanh for
    /// joints, sigmoid for the gripper).
    last_outputs: [f32; NUM_OUTPUTS],
}

impl ManipulatorController {
    pub fn new(genesis: &GenesisSeed, config: &ManipulatorConfig) -> Self {
        let neuron_config = UnifiedConfig {
            tau_base: 1.0 / config.physics_hz as f32,
            backbone_tau: 0.2, // Fast for precision manipulation
            dimension: HDC_DIM,
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
        let weight_hv = ContinuousHV::from_genesis(
            genesis,
            "manipulator::output_weights",
            NUM_OUTPUTS * HDC_DIM,
        );
        let mut output_weights: Vec<f32> = weight_hv.as_slice().to_vec();
        for w in &mut output_weights {
            *w *= 0.01;
        }
        let output_bias = [0.0; NUM_OUTPUTS]; // Zero torque at rest

        Self {
            network,
            output_weights,
            output_bias,
            learning_rate: config.learning_rate,
            last_features: Vec::new(),
            last_outputs: [0.0; NUM_OUTPUTS],
        }
    }

    pub fn forward(&mut self, sensor_hv: &ContinuousHV, dt: f32) -> ManipulatorCommand {
        self.network.evolve_closed_form(dt, sensor_hv);
        let output_hv = self.network.output().normalize();
        let hv = output_hv.as_slice();
        let mut raw = [0.0f32; NUM_OUTPUTS];
        for i in 0..NUM_OUTPUTS {
            let offset = i * HDC_DIM;
            let mut sum = 0.0f32;
            for j in 0..HDC_DIM {
                sum += self.output_weights[offset + j] * hv[j];
            }
            raw[i] = sum + self.output_bias[i];
        }
        let mut torques = [0.0f32; NUM_JOINTS];
        for i in 0..NUM_JOINTS {
            torques[i] = fast_tanh(raw[i]);
        }
        let gripper = fast_sigmoid(raw[NUM_JOINTS]);
        self.last_features = hv.to_vec();
        self.last_outputs = {
            let mut o = [0.0f32; NUM_OUTPUTS];
            o[..NUM_JOINTS].copy_from_slice(&torques);
            o[NUM_JOINTS] = gripper;
            o
        };
        ManipulatorCommand {
            joint_torques: torques,
            gripper,
        }
        .clamped()
    }

    /// One supervised update of the joint-torque output rows toward
    /// `target` (delta rule through tanh), using the features cached by the
    /// last `forward()`. The gripper output row is left untouched -- the
    /// imitation target (gravity-compensation hold) has no opinion about
    /// grip state. Returns the pre-update mean-squared error over the
    /// joint torques. This is what makes `ManipulatorTrainer` actually
    /// train (Tier 2 of SYMTHAEA_CLASSIC_PLATFORMS_FEP_HONESTY_2026-07-09.md's
    /// real-trainer follow-up).
    pub fn train_step(&mut self, target: &[f32; NUM_JOINTS]) -> f32 {
        if self.last_features.is_empty() {
            return 0.0;
        }
        let mut mse = 0.0f32;
        for i in 0..NUM_JOINTS {
            let out = self.last_outputs[i];
            let err = target[i] - out;
            mse += err * err;
            let delta = self.learning_rate * err * (1.0 - out * out);
            let offset = i * HDC_DIM;
            for (j, f) in self.last_features.iter().enumerate() {
                self.output_weights[offset + j] += delta * f;
            }
            self.output_bias[i] += delta;
        }
        mse / NUM_JOINTS as f32
    }

    pub fn reset(&mut self) {
        self.network.reset();
        self.last_features.clear();
        self.last_outputs = [0.0; NUM_OUTPUTS];
    }

    /// Export the trainable output layer (the only parameters `train_step`
    /// mutates — the LTC network itself is genesis-derived and untouched by
    /// training), so trained weights can be transferred into a shipped
    /// bridge or persisted by the caller.
    pub fn export_weights(&self) -> ControllerWeights {
        ControllerWeights {
            output_weights: self.output_weights.clone(),
            output_bias: self.output_bias.to_vec(),
        }
    }

    /// Install a previously exported output layer. Fails (leaving the
    /// controller unchanged) if the snapshot's dimensions don't match this
    /// controller's output layer.
    pub fn import_weights(&mut self, weights: &ControllerWeights) -> Result<(), String> {
        if weights.output_weights.len() != self.output_weights.len() {
            return Err(format!(
                "output_weights length mismatch: snapshot {} vs controller {}",
                weights.output_weights.len(),
                self.output_weights.len()
            ));
        }
        if weights.output_bias.len() != NUM_OUTPUTS {
            return Err(format!(
                "output_bias length mismatch: snapshot {} vs controller {NUM_OUTPUTS}",
                weights.output_bias.len()
            ));
        }
        self.output_weights.copy_from_slice(&weights.output_weights);
        self.output_bias.copy_from_slice(&weights.output_bias);
        Ok(())
    }
}

/// Portable snapshot of the controller's trainable output layer.
///
/// This is the trainer→bridge transfer contract: `ManipulatorTrainer` (or
/// the intent curriculum in `training.rs`) exports one, and a shipped
/// `ManipulatorEmbodiment` installs it via `install_weights`. Before this
/// existed, trained weights were stranded inside the trainer's private
/// controller and every shipped bridge ran genesis-random weights (see the
/// cognition-ablation example: thought had no task-axis motor authority).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ControllerWeights {
    pub output_weights: Vec<f32>,
    pub output_bias: Vec<f32>,
}

fn fast_tanh(x: f32) -> f32 {
    if x.abs() > 4.97 {
        x.signum()
    } else {
        x * (27.0 + x * x) / (27.0 + 9.0 * x * x)
    }
}
fn fast_sigmoid(x: f32) -> f32 {
    0.5 * (1.0 + fast_tanh(x * 0.5))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_valid() {
        let genesis = GenesisSeed::from_phrase("test-manip-ctrl");
        let mut ctrl = ManipulatorController::new(&genesis, &ManipulatorConfig::default());
        let hv = ContinuousHV::random(HDC_DIM, 42);
        let cmd = ctrl.forward(&hv, 0.002);
        for &t in &cmd.joint_torques {
            assert!(t >= -1.0 && t <= 1.0);
        }
        assert!(cmd.gripper >= 0.0 && cmd.gripper <= 1.0);
    }

    #[test]
    fn test_deterministic() {
        let genesis = GenesisSeed::from_phrase("test-det");
        let config = ManipulatorConfig::default();
        let mut c1 = ManipulatorController::new(&genesis, &config);
        let mut c2 = ManipulatorController::new(&genesis, &config);
        let hv = ContinuousHV::from_genesis(&genesis, "input", HDC_DIM);
        let cmd1 = c1.forward(&hv, 0.002);
        let cmd2 = c2.forward(&hv, 0.002);
        assert_eq!(cmd1.joint_torques, cmd2.joint_torques);
    }

    #[test]
    fn test_weight_export_import_roundtrip() {
        // Train one controller a little, export, import into a fresh
        // controller with a different genesis: outputs must match the
        // trained controller exactly (the output layer is the only
        // trainable state, and it must transfer completely).
        let config = ManipulatorConfig::default();
        let mut trained =
            ManipulatorController::new(&GenesisSeed::from_phrase("test-roundtrip-a"), &config);
        let hv = ContinuousHV::random(HDC_DIM, 7);
        let target = [0.3f32; NUM_JOINTS];
        for _ in 0..5 {
            trained.forward(&hv, 0.002);
            trained.train_step(&target);
        }
        let snapshot = trained.export_weights();

        // Same genesis for the receiving controller so the (untrained,
        // genesis-derived) LTC network matches; only the output layer moves.
        let mut receiver =
            ManipulatorController::new(&GenesisSeed::from_phrase("test-roundtrip-a"), &config);
        receiver.import_weights(&snapshot).unwrap();

        trained.reset();
        receiver.reset();
        let a = trained.forward(&hv, 0.002);
        let b = receiver.forward(&hv, 0.002);
        assert_eq!(a.joint_torques, b.joint_torques);
        assert_eq!(a.gripper, b.gripper);
    }

    #[test]
    fn test_import_weights_rejects_dim_mismatch() {
        let config = ManipulatorConfig::default();
        let mut ctrl = ManipulatorController::new(&GenesisSeed::from_phrase("test-dims"), &config);
        let bad = ControllerWeights {
            output_weights: vec![0.0; 3],
            output_bias: vec![0.0; NUM_OUTPUTS],
        };
        assert!(ctrl.import_weights(&bad).is_err());
    }
}
