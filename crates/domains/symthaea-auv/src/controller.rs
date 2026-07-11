// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! AUV controller: HdcLtcUnifiedNetwork + output projection (16,384D → 8D).

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig};

use crate::types::{AuvCommand, AuvConfig, NUM_ACTUATORS};

const HDC_DIM: usize = symthaea_core::hdc::HDC_DIMENSION;

/// AUV controller wrapping HdcLtcUnifiedNetwork + linear output head.
pub struct AuvController {
    network: HdcLtcUnifiedNetwork,
    output_weights: Vec<f32>,
    output_bias: [f32; NUM_ACTUATORS],
    learning_rate: f32,
    /// Cached final-layer HV from the last forward() (post-normalize) --
    /// needed by train_step's delta rule.
    last_features: Vec<f32>,
    /// Cached post-tanh outputs from the last forward().
    last_outputs: [f32; NUM_ACTUATORS],
}

impl AuvController {
    pub fn new(genesis: &GenesisSeed, config: &AuvConfig) -> Self {
        let neuron_config = UnifiedConfig {
            tau_base: 1.0 / config.physics_hz as f32,
            backbone_tau: 0.5, // Slower than air platforms (drag-dominated)
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

        let weight_hv =
            ContinuousHV::from_genesis(genesis, "auv::output_weights", NUM_ACTUATORS * HDC_DIM);
        let mut output_weights: Vec<f32> = weight_hv.as_slice().to_vec();
        for w in &mut output_weights {
            *w *= 0.01;
        }

        // Bias: all zero (neutral buoyancy = no thrust needed)
        let output_bias = [0.0f32; NUM_ACTUATORS];

        Self {
            network,
            output_weights,
            output_bias,
            learning_rate: config.learning_rate,
            last_features: Vec::new(),
            last_outputs: [0.0; NUM_ACTUATORS],
        }
    }

    pub fn forward(&mut self, sensor_hv: &ContinuousHV, dt: f32) -> AuvCommand {
        self.network.evolve_closed_form(dt, sensor_hv);
        let output_hv = self.network.output().normalize();
        let hv = output_hv.as_slice();

        let mut raw = [0.0f32; NUM_ACTUATORS];
        for i in 0..NUM_ACTUATORS {
            let offset = i * HDC_DIM;
            let mut sum = 0.0f32;
            for j in 0..HDC_DIM {
                sum += self.output_weights[offset + j] * hv[j];
            }
            raw[i] = sum + self.output_bias[i];
        }

        // tanh activation for all thrusters (bidirectional)
        let mut thrusters = [0.0f32; NUM_ACTUATORS];
        for i in 0..NUM_ACTUATORS {
            thrusters[i] = fast_tanh(raw[i]);
        }
        self.last_features = hv.to_vec();
        self.last_outputs = thrusters;
        AuvCommand { thrusters }.clamped()
    }

    /// One supervised update of the output projection toward `target`
    /// (delta rule through tanh), using the features cached by the last
    /// `forward()`. Returns the pre-update mean-squared error. This is what
    /// makes `AuvTrainer` actually train (real-trainer follow-up to
    /// SYMTHAEA_CLASSIC_PLATFORMS_FEP_HONESTY_2026-07-09.md).
    pub fn train_step(&mut self, target: &[f32; NUM_ACTUATORS]) -> f32 {
        if self.last_features.is_empty() {
            return 0.0;
        }
        let mut mse = 0.0f32;
        for i in 0..NUM_ACTUATORS {
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
        mse / NUM_ACTUATORS as f32
    }

    pub fn reset(&mut self) {
        self.network.reset();
        self.last_features.clear();
        self.last_outputs = [0.0; NUM_ACTUATORS];
    }
}

fn fast_tanh(x: f32) -> f32 {
    if x.abs() > 4.97 {
        x.signum()
    } else {
        x * (27.0 + x * x) / (27.0 + 9.0 * x * x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_valid_command() {
        let genesis = GenesisSeed::from_phrase("test-auv-ctrl");
        let config = AuvConfig::default();
        let mut ctrl = AuvController::new(&genesis, &config);
        let hv = ContinuousHV::random(HDC_DIM, 42);
        let cmd = ctrl.forward(&hv, 0.01);
        for &t in &cmd.thrusters {
            assert!(t >= -1.0 && t <= 1.0);
        }
    }

    #[test]
    fn test_forward_deterministic() {
        let genesis = GenesisSeed::from_phrase("test-auv-det");
        let config = AuvConfig::default();
        let mut c1 = AuvController::new(&genesis, &config);
        let mut c2 = AuvController::new(&genesis, &config);
        let hv = ContinuousHV::from_genesis(&genesis, "test-input", HDC_DIM);
        let cmd1 = c1.forward(&hv, 0.01);
        let cmd2 = c2.forward(&hv, 0.01);
        assert_eq!(cmd1.thrusters, cmd2.thrusters);
    }
}
