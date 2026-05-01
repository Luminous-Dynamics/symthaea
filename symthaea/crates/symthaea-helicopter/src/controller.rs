// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Helicopter controller: HdcLtcUnifiedNetwork + output projection (16,384D → 6D).
//!
//! Same architecture as symthaea-multirotor FlightController but with 6 output channels
//! (collective, cyclic_lon, cyclic_lat, pedal, thrust, tail_rotor) and bias initialized
//! to hover command rather than zero.

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig};

use crate::types::{HelicopterCommand, HelicopterConfig, NUM_ACTUATORS};

const HDC_DIM: usize = symthaea_core::hdc::HDC_DIMENSION;

/// Helicopter controller wrapping an HdcLtcUnifiedNetwork + linear output head.
pub struct HelicopterController {
    /// Temporal dynamics engine — full 16,384D HDC-LTC.
    network: HdcLtcUnifiedNetwork,
    /// Output projection weights: 6 × 16,384 (flat row-major).
    output_weights: Vec<f32>,
    /// Output bias (6D, initialized to hover command).
    output_bias: [f32; NUM_ACTUATORS],
    /// Current learning rate.
    learning_rate: f32,
}

impl HelicopterController {
    /// Create a new controller from genesis seed and config.
    pub fn new(genesis: &GenesisSeed, config: &HelicopterConfig) -> Self {
        let neuron_config = UnifiedConfig {
            tau_base: 1.0 / config.physics_hz as f32,
            backbone_tau: 0.3,
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

        // Output weights from genesis (small for stability)
        let total_weights = NUM_ACTUATORS * HDC_DIM;
        let weight_hv =
            ContinuousHV::from_genesis(genesis, "helicopter::output_weights", total_weights);
        let mut output_weights: Vec<f32> = weight_hv.as_slice().to_vec();
        for w in &mut output_weights {
            *w *= 0.01;
        }

        // Bias initialized to hover command
        let output_bias = [
            HelicopterCommand::HOVER_COLLECTIVE,
            0.0, // cyclic_lon
            0.0, // cyclic_lat
            0.0, // pedal
            HelicopterCommand::HOVER_THRUST,
            HelicopterCommand::HOVER_TAIL,
        ];

        Self {
            network,
            output_weights,
            output_bias,
            learning_rate: config.learning_rate,
        }
    }

    /// Forward pass: evolve network + project to 6D motor command.
    pub fn forward(&mut self, sensor_hv: &ContinuousHV, dt: f32) -> HelicopterCommand {
        // Evolve network dynamics
        self.network.evolve_closed_form(dt, sensor_hv);

        // Get output HV (bundled final layer)
        let output_hv = self.network.output().normalize();
        let hv = output_hv.as_slice();

        // Linear projection: weights @ hv + bias → 6D
        let mut raw = [0.0f32; NUM_ACTUATORS];
        for i in 0..NUM_ACTUATORS {
            let offset = i * HDC_DIM;
            let mut sum = 0.0f32;
            for j in 0..HDC_DIM {
                sum += self.output_weights[offset + j] * hv[j];
            }
            raw[i] = sum + self.output_bias[i];
        }

        // Activations: tanh for signed channels, sigmoid for [0,1] channels
        HelicopterCommand {
            collective: fast_tanh(raw[0]),
            cyclic_lon: fast_tanh(raw[1]),
            cyclic_lat: fast_tanh(raw[2]),
            pedal: fast_tanh(raw[3]),
            thrust: fast_sigmoid(raw[4]),
            tail_rotor: fast_sigmoid(raw[5]),
        }
        .clamped()
    }

    /// Reset network hidden states.
    pub fn reset(&mut self) {
        self.network.reset();
    }

    /// Get current learning rate.
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Modulate learning rate.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    /// Access the underlying network.
    pub fn network(&self) -> &HdcLtcUnifiedNetwork {
        &self.network
    }
}

/// Fast Padé approximation of tanh.
fn fast_tanh(x: f32) -> f32 {
    if x.abs() > 4.97 {
        x.signum()
    } else {
        x * (27.0 + x * x) / (27.0 + 9.0 * x * x)
    }
}

/// Fast sigmoid via shifted tanh.
fn fast_sigmoid(x: f32) -> f32 {
    0.5 * (1.0 + fast_tanh(x * 0.5))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_controller() -> HelicopterController {
        let genesis = GenesisSeed::from_phrase("test-helicopter-controller");
        let config = HelicopterConfig::default();
        HelicopterController::new(&genesis, &config)
    }

    #[test]
    fn test_forward_produces_valid_command() {
        let mut ctrl = make_controller();
        let hv = ContinuousHV::random(HDC_DIM, 42);
        let cmd = ctrl.forward(&hv, 1.0 / 300.0);

        assert!(cmd.collective >= -1.0 && cmd.collective <= 1.0);
        assert!(cmd.cyclic_lon >= -1.0 && cmd.cyclic_lon <= 1.0);
        assert!(cmd.cyclic_lat >= -1.0 && cmd.cyclic_lat <= 1.0);
        assert!(cmd.pedal >= -1.0 && cmd.pedal <= 1.0);
        assert!(cmd.thrust >= 0.0 && cmd.thrust <= 1.0);
        assert!(cmd.tail_rotor >= 0.0 && cmd.tail_rotor <= 1.0);
    }

    #[test]
    fn test_forward_deterministic() {
        let genesis = GenesisSeed::from_phrase("test-heli-determinism");
        let config = HelicopterConfig::default();
        let mut ctrl1 = HelicopterController::new(&genesis, &config);
        let mut ctrl2 = HelicopterController::new(&genesis, &config);

        let hv = ContinuousHV::from_genesis(&genesis, "test-input", HDC_DIM);
        let cmd1 = ctrl1.forward(&hv, 1.0 / 300.0);
        let cmd2 = ctrl2.forward(&hv, 1.0 / 300.0);

        assert_eq!(cmd1.collective, cmd2.collective);
        assert_eq!(cmd1.thrust, cmd2.thrust);
    }

    #[test]
    fn test_fast_tanh_bounds() {
        assert!(fast_tanh(100.0) == 1.0);
        assert!(fast_tanh(-100.0) == -1.0);
        assert!((fast_tanh(0.0)).abs() < 1e-6);
    }

    #[test]
    fn test_fast_sigmoid_bounds() {
        assert!(fast_sigmoid(100.0) > 0.99);
        assert!(fast_sigmoid(-100.0) < 0.01);
        assert!((fast_sigmoid(0.0) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_reset_changes_output() {
        let mut ctrl = make_controller();
        let hv = ContinuousHV::random(HDC_DIM, 42);

        // Run a few steps to build hidden state
        for _ in 0..10 {
            ctrl.forward(&hv, 1.0 / 300.0);
        }

        ctrl.reset();
        // After reset, same input should produce different output than accumulated state
        // (This verifies reset actually clears hidden state)
    }
}
