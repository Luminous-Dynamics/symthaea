// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::{ExoskeletonCommand, ExoskeletonConfig, NUM_ACTUATORS};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig};
const HDC_DIM: usize = symthaea_core::hdc::HDC_DIMENSION;
const OUT: usize = NUM_ACTUATORS + 2;

pub struct ExoskeletonController {
    network: HdcLtcUnifiedNetwork,
    weights: Vec<f32>,
    bias: [f32; OUT],
    learning_rate: f32,
    /// Cached final-layer HV from the last forward() (post-normalize) --
    /// needed by train_step's delta rule.
    last_features: Vec<f32>,
    /// Cached post-tanh joint-torque outputs from the last forward().
    last_torque_outputs: [f32; NUM_ACTUATORS],
}

impl ExoskeletonController {
    pub fn new(genesis: &GenesisSeed, config: &ExoskeletonConfig) -> Self {
        let nc = UnifiedConfig {
            tau_base: 1.0 / config.physics_hz as f32,
            backbone_tau: 0.3,
            dimension: HDC_DIM,
            learning_rate: config.learning_rate,
            ..UnifiedConfig::default()
        };
        let net = UnifiedNetworkConfig {
            layer_sizes: vec![config.neurons_per_layer; config.network_layers],
            neuron_config: nc,
            use_layer_binding: true,
            skip_connections: false,
        };
        let network = HdcLtcUnifiedNetwork::from_genesis(net, genesis);
        let wh = ContinuousHV::from_genesis(genesis, "exo::out_w", OUT * HDC_DIM);
        let mut weights: Vec<f32> = wh.as_slice().to_vec();
        for w in &mut weights {
            *w *= 0.01;
        }
        let mut bias = [0.0f32; OUT];
        bias[6] = 0.5;
        bias[7] = 0.3;
        Self {
            network,
            weights,
            bias,
            learning_rate: config.learning_rate,
            last_features: Vec::new(),
            last_torque_outputs: [0.0; NUM_ACTUATORS],
        }
    }
    pub fn forward(&mut self, hv: &ContinuousHV, dt: f32) -> ExoskeletonCommand {
        self.network.evolve_closed_form(dt, hv);
        let out = self.network.output().normalize();
        let d = out.as_slice();
        let mut raw = [0.0f32; OUT];
        for o in 0..OUT {
            let off = o * HDC_DIM;
            let mut s = self.bias[o];
            for j in 0..HDC_DIM {
                s += self.weights[off + j] * d[j];
            }
            raw[o] = s;
        }
        let mut t = [0.0f32; NUM_ACTUATORS];
        for i in 0..NUM_ACTUATORS {
            t[i] = raw[i].tanh();
        }
        fn sig(x: f32) -> f32 {
            1.0 / (1.0 + (-x).exp())
        }
        self.last_features = d.to_vec();
        self.last_torque_outputs = t;
        ExoskeletonCommand {
            joint_torques: t,
            stiffness_gain: sig(raw[6]).clamp(0.0, 1.0),
            damping_gain: sig(raw[7]).clamp(0.0, 1.0),
        }
    }

    /// One supervised update of the joint-torque output rows toward
    /// `target` (delta rule through tanh), using the features cached by the
    /// last `forward()`. The stiffness/damping gain rows are left untouched.
    /// Returns the pre-update mean-squared error. This is what makes
    /// `ExoskeletonTrainer` actually train (real-trainer follow-up to
    /// SYMTHAEA_CLASSIC_PLATFORMS_FEP_HONESTY_2026-07-09.md).
    pub fn train_step(&mut self, target: &[f32; NUM_ACTUATORS]) -> f32 {
        if self.last_features.is_empty() {
            return 0.0;
        }
        let mut mse = 0.0f32;
        for i in 0..NUM_ACTUATORS {
            let out = self.last_torque_outputs[i];
            let err = target[i] - out;
            mse += err * err;
            let delta = self.learning_rate * err * (1.0 - out * out);
            let off = i * HDC_DIM;
            for (j, f) in self.last_features.iter().enumerate() {
                self.weights[off + j] += delta * f;
            }
            self.bias[i] += delta;
        }
        mse / NUM_ACTUATORS as f32
    }

    pub fn reset(&mut self) {
        self.network.reset();
        self.last_features.clear();
        self.last_torque_outputs = [0.0; NUM_ACTUATORS];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_forward() {
        let mut c = ExoskeletonController::new(
            &GenesisSeed::from_phrase("t"),
            &ExoskeletonConfig::default(),
        );
        let cmd = c.forward(&ContinuousHV::random(HDC_DIM, 42), 0.005);
        assert!(
            cmd.joint_torques
                .iter()
                .all(|t| t.is_finite() && t.abs() <= 1.0)
        );
    }
}
