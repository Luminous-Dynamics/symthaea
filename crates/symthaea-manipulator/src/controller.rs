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
        ManipulatorCommand {
            joint_torques: torques,
            gripper,
        }
        .clamped()
    }

    pub fn reset(&mut self) {
        self.network.reset();
    }
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
}
