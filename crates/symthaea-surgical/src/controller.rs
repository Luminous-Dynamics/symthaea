// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::types::{NUM_ACTUATORS, SurgicalCommand, SurgicalConfig};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig};
const HDC_DIM: usize = symthaea_core::hdc::HDC_DIMENSION;
pub struct SurgicalController {
    network: HdcLtcUnifiedNetwork,
    weights: Vec<f32>,
    bias: [f32; NUM_ACTUATORS],
    #[allow(dead_code)]
    learning_rate: f32,
}
impl SurgicalController {
    pub fn new(g: &GenesisSeed, c: &SurgicalConfig) -> Self {
        let nc = UnifiedConfig {
            tau_base: 1.0 / c.physics_hz as f32,
            backbone_tau: 0.2,
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
        let wh = ContinuousHV::from_genesis(g, "surg::out_w", NUM_ACTUATORS * HDC_DIM);
        let mut w: Vec<f32> = wh.as_slice().to_vec();
        for v in &mut w {
            *v *= 0.005;
        }
        Self {
            network,
            weights: w,
            bias: [0.0; NUM_ACTUATORS],
            learning_rate: c.learning_rate,
        }
    }
    pub fn forward(&mut self, hv: &ContinuousHV, dt: f32) -> SurgicalCommand {
        self.network.evolve_closed_form(dt, hv);
        let out = self.network.output().normalize();
        let d = out.as_slice();
        let mut raw = [0.0f32; NUM_ACTUATORS];
        for o in 0..NUM_ACTUATORS {
            let off = o * HDC_DIM;
            let mut s = self.bias[o];
            for j in 0..HDC_DIM {
                s += self.weights[off + j] * d[j];
            }
            raw[o] = s;
        }
        let mut t = [0.0f32; 6];
        for i in 0..6 {
            t[i] = raw[i].tanh();
        }
        fn sig(x: f32) -> f32 {
            1.0 / (1.0 + (-x).exp())
        }
        SurgicalCommand {
            joint_torques: t,
            jaw: sig(raw[6]).clamp(0.0, 1.0),
            cautery: sig(raw[7]).clamp(0.0, 1.0),
        }
    }
    pub fn reset(&mut self) {
        self.network.reset();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_fwd() {
        let mut c =
            SurgicalController::new(&GenesisSeed::from_phrase("t"), &SurgicalConfig::default());
        let cmd = c.forward(&ContinuousHV::random(HDC_DIM, 42), 0.001);
        assert!(cmd.joint_torques.iter().all(|t| t.is_finite()));
    }
}
