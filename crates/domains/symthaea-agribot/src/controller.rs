// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::{AgribotCommand, AgribotConfig, NUM_ACTUATORS};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig};

const HDC_DIM: usize = symthaea_core::hdc::HDC_DIMENSION;

pub struct AgribotController {
    network: HdcLtcUnifiedNetwork,
    weights: Vec<f32>,
    bias: [f32; NUM_ACTUATORS],
    learning_rate: f32,
    /// Cached final-layer HV from the last forward() (post-normalize) --
    /// needed by train_step's delta rule.
    last_features: Vec<f32>,
    /// Cached post-tanh outputs from the last forward().
    last_outputs: [f32; NUM_ACTUATORS],
}

impl AgribotController {
    pub fn new(g: &GenesisSeed, c: &AgribotConfig) -> Self {
        let nc = UnifiedConfig {
            tau_base: 1.0 / c.physics_hz as f32,
            backbone_tau: 0.3,
            dimension: HDC_DIM,
            learning_rate: c.learning_rate,
            ..UnifiedConfig::default()
        };
        let net = UnifiedNetworkConfig {
            layer_sizes: vec![c.neurons_per_layer; c.network_layers],
            neuron_config: nc,
            use_layer_binding: true,
            skip_connections: false,
        };
        let network = HdcLtcUnifiedNetwork::from_genesis(net, g);
        let wh = ContinuousHV::from_genesis(g, "agribot::out_w", NUM_ACTUATORS * HDC_DIM);
        let mut w: Vec<f32> = wh.as_slice().to_vec();
        for v in &mut w {
            *v *= 0.01;
        }
        Self {
            network,
            weights: w,
            bias: [0.0; NUM_ACTUATORS],
            learning_rate: c.learning_rate,
            last_features: Vec::new(),
            last_outputs: [0.0; NUM_ACTUATORS],
        }
    }

    pub fn forward(&mut self, hv: &ContinuousHV, dt: f32) -> AgribotCommand {
        self.network.evolve_closed_form(dt, hv);
        let out = self.network.output().normalize();
        let d = out.as_slice();
        let mut t = [0.0f32; NUM_ACTUATORS];
        for o in 0..NUM_ACTUATORS {
            let off = o * HDC_DIM;
            let mut s = self.bias[o];
            for j in 0..HDC_DIM {
                s += self.weights[off + j] * d[j];
            }
            t[o] = s.tanh();
        }
        self.last_features = d.to_vec();
        self.last_outputs = t;
        AgribotCommand { torques: t }
    }

    /// One supervised update of the output projection toward `target`
    /// (delta rule through the tanh), using the features cached by the last
    /// `forward()`. Returns the pre-update mean-squared error. This is what
    /// makes `AgribotTrainer` actually train (Tier 2 of
    /// SYMTHAEA_UNAUDITED_PLATFORMS_REVIEW_2026-07-07.md).
    pub fn train_step(&mut self, target: &AgribotCommand) -> f32 {
        if self.last_features.is_empty() {
            return 0.0;
        }
        let mut mse = 0.0f32;
        for o in 0..NUM_ACTUATORS {
            let out = self.last_outputs[o];
            let err = target.torques[o] - out;
            mse += err * err;
            // Backprop through tanh: d(out)/d(pre) = 1 - out²
            let delta = self.learning_rate * err * (1.0 - out * out);
            let off = o * HDC_DIM;
            for (j, f) in self.last_features.iter().enumerate() {
                self.weights[off + j] += delta * f;
            }
            self.bias[o] += delta;
        }
        mse / NUM_ACTUATORS as f32
    }

    pub fn reset(&mut self) {
        self.network.reset();
        self.last_features.clear();
        self.last_outputs = [0.0; NUM_ACTUATORS];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_fwd() {
        let mut c =
            AgribotController::new(&GenesisSeed::from_phrase("t"), &AgribotConfig::default());
        let cmd = c.forward(&ContinuousHV::random(HDC_DIM, 42), 0.005);
        assert!(cmd.torques.iter().all(|t| t.is_finite()));
    }
}
