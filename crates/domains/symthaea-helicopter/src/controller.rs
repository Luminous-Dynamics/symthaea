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

use crate::types::{HelicopterCommand, HelicopterConfig, HelicopterState, NUM_ACTUATORS};

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
    /// Cached final-layer HV from the last `forward()` (post-normalize) —
    /// needed by `train_step`'s delta rule.
    last_features: Vec<f32>,
    /// Cached post-activation outputs from the last `forward()`.
    last_outputs: [f32; NUM_ACTUATORS],
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
            last_features: Vec::new(),
            last_outputs: [0.0; NUM_ACTUATORS],
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
        let outputs = [
            fast_tanh(raw[0]),
            fast_tanh(raw[1]),
            fast_tanh(raw[2]),
            fast_tanh(raw[3]),
            fast_sigmoid(raw[4]),
            fast_sigmoid(raw[5]),
        ];

        // Cache features + activations for train_step's delta rule
        self.last_features = hv.to_vec();
        self.last_outputs = outputs;

        HelicopterCommand {
            collective: outputs[0],
            cyclic_lon: outputs[1],
            cyclic_lat: outputs[2],
            pedal: outputs[3],
            thrust: outputs[4],
            tail_rotor: outputs[5],
        }
        .clamped()
    }

    /// One supervised update of the output projection toward `target`
    /// (delta rule through the output activations), using the features
    /// cached by the last `forward()`. Returns the pre-update mean-squared
    /// error. This is what makes `HelicopterTrainer` actually train —
    /// previously the trainer collected metrics and never touched a weight.
    ///
    /// Channels 0-3 (collective/cyclic×2/pedal) are tanh: d/dx = 1 − out².
    /// Channels 4-5 (thrust/tail_rotor) are sigmoid: d/dx = out·(1 − out).
    pub fn train_step(&mut self, target: &HelicopterCommand) -> f32 {
        if self.last_features.is_empty() {
            return 0.0;
        }
        let target_arr = [
            target.collective,
            target.cyclic_lon,
            target.cyclic_lat,
            target.pedal,
            target.thrust,
            target.tail_rotor,
        ];
        let mut mse = 0.0f32;
        for i in 0..NUM_ACTUATORS {
            let out = self.last_outputs[i];
            let err = target_arr[i] - out;
            mse += err * err;
            let dact = if i < 4 {
                1.0 - out * out // tanh derivative
            } else {
                out * (1.0 - out) // sigmoid derivative
            };
            let delta = self.learning_rate * err * dact;
            let offset = i * HDC_DIM;
            for (j, f) in self.last_features.iter().enumerate() {
                self.output_weights[offset + j] += delta * f;
            }
            self.output_bias[i] += delta;
        }
        mse / NUM_ACTUATORS as f32
    }

    /// Reset network hidden states.
    pub fn reset(&mut self) {
        self.network.reset();
        self.last_features.clear();
        self.last_outputs = [0.0; NUM_ACTUATORS];
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

/// PD hover/attitude baseline: the imitation target for `train_step`.
///
/// Proportional-derivative regulation around a stable hover:
/// - collective/thrust track altitude error with climb-rate damping
/// - cyclic drives roll/pitch to level with rate damping
/// - pedal damps yaw rate
///
/// This mirrors the spinal-reflex targets used by symthaea-quadruped's
/// trainer: the HDC-LTC controller learns to imitate a classical baseline
/// first; task shaping can then move beyond it.
pub fn pd_hover_baseline(state: &HelicopterState, target_altitude: f64) -> HelicopterCommand {
    // Altitude channel
    let alt_err = (target_altitude - state.altitude()) as f32;
    let climb_rate = state.linear_velocity[2] as f32;

    // Attitude channel
    let (roll, pitch, _yaw) = state.euler_angles();
    let (roll, pitch) = (roll as f32, pitch as f32);
    let [wx, wy, wz] = state.angular_velocity;
    let (wx, wy, wz) = (wx as f32, wy as f32, wz as f32);

    // Gains (dimensionless commands per SI error unit)
    const KP_ALT_COLLECTIVE: f32 = 0.05;
    const KD_ALT_COLLECTIVE: f32 = 0.10;
    const KP_ALT_THRUST: f32 = 0.02;
    const KD_ALT_THRUST: f32 = 0.05;
    const KP_ATT: f32 = 1.5;
    const KD_ATT: f32 = 0.5;
    const KD_YAW: f32 = 0.8;

    HelicopterCommand {
        collective: HelicopterCommand::HOVER_COLLECTIVE + KP_ALT_COLLECTIVE * alt_err
            - KD_ALT_COLLECTIVE * climb_rate,
        // Positive cyclic produces a positive body moment (simulator step 6),
        // so leveling requires opposing the current angle + rate.
        cyclic_lon: -(KP_ATT * pitch + KD_ATT * wy),
        cyclic_lat: -(KP_ATT * roll + KD_ATT * wx),
        pedal: -KD_YAW * wz,
        thrust: HelicopterCommand::HOVER_THRUST + KP_ALT_THRUST * alt_err
            - KD_ALT_THRUST * climb_rate,
        tail_rotor: HelicopterCommand::HOVER_TAIL,
    }
    .clamped()
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
    fn test_reset_restores_initial_output() {
        // (Previously this test had no assertion at all — pure padding.)
        let mut ctrl = make_controller();
        let hv = ContinuousHV::random(HDC_DIM, 42);

        // First output from a fresh controller
        let first = ctrl.forward(&hv, 1.0 / 300.0);

        // Run steps to accumulate hidden state
        for _ in 0..10 {
            ctrl.forward(&hv, 1.0 / 300.0);
        }

        // After reset, the same input must reproduce the fresh-controller
        // output — proving reset actually clears the hidden state.
        ctrl.reset();
        let after_reset = ctrl.forward(&hv, 1.0 / 300.0);
        let pairs = [
            (first.collective, after_reset.collective),
            (first.cyclic_lon, after_reset.cyclic_lon),
            (first.cyclic_lat, after_reset.cyclic_lat),
            (first.pedal, after_reset.pedal),
            (first.thrust, after_reset.thrust),
            (first.tail_rotor, after_reset.tail_rotor),
        ];
        for (i, (a, b)) in pairs.iter().enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "channel {i}: reset must restore initial state ({a} vs {b})"
            );
        }
    }

    #[test]
    fn test_train_step_moves_output_toward_target() {
        let mut ctrl = make_controller();
        let hv = ContinuousHV::random(HDC_DIM, 7);
        ctrl.set_learning_rate(0.05); // Large LR so movement is visible fast

        let target = HelicopterCommand {
            collective: 0.8,
            cyclic_lon: -0.4,
            cyclic_lat: 0.4,
            pedal: 0.2,
            thrust: 0.9,
            tail_rotor: 0.2,
        };

        let first_mse = {
            ctrl.forward(&hv, 1.0 / 300.0);
            ctrl.train_step(&target)
        };
        assert!(first_mse > 0.0, "learning signal must be live");

        let mut last_mse = first_mse;
        for _ in 0..200 {
            ctrl.reset(); // same input each time → isolate weight learning
            ctrl.forward(&hv, 1.0 / 300.0);
            last_mse = ctrl.train_step(&target);
        }
        assert!(
            last_mse < first_mse,
            "train_step must reduce imitation error: first {first_mse:.5} -> last {last_mse:.5}"
        );
    }

    #[test]
    fn test_train_step_without_forward_is_noop() {
        let mut ctrl = make_controller();
        let loss = ctrl.train_step(&HelicopterCommand::hover());
        assert_eq!(loss, 0.0, "no cached features → no update, zero loss");
    }

    #[test]
    fn test_pd_baseline_at_hover_is_hover_command() {
        let state = HelicopterState::hover(20.0);
        let cmd = pd_hover_baseline(&state, 20.0);
        assert!((cmd.collective - HelicopterCommand::HOVER_COLLECTIVE).abs() < 1e-6);
        assert!((cmd.thrust - HelicopterCommand::HOVER_THRUST).abs() < 1e-6);
        assert!(cmd.cyclic_lon.abs() < 1e-6);
        assert!(cmd.cyclic_lat.abs() < 1e-6);
        assert!(cmd.pedal.abs() < 1e-6);
    }

    #[test]
    fn test_pd_baseline_corrects_low_altitude() {
        // Below target → collective and thrust above hover values.
        let state = HelicopterState::hover(10.0);
        let cmd = pd_hover_baseline(&state, 20.0);
        assert!(cmd.collective > HelicopterCommand::HOVER_COLLECTIVE);
        assert!(cmd.thrust > HelicopterCommand::HOVER_THRUST);
    }

    #[test]
    fn test_pd_baseline_damps_rotation() {
        let mut state = HelicopterState::hover(20.0);
        state.angular_velocity = [0.5, -0.5, 0.3];
        let cmd = pd_hover_baseline(&state, 20.0);
        assert!(
            cmd.cyclic_lat < 0.0,
            "positive roll rate → negative cyclic_lat"
        );
        assert!(
            cmd.cyclic_lon > 0.0,
            "negative pitch rate → positive cyclic_lon"
        );
        assert!(cmd.pedal < 0.0, "positive yaw rate → negative pedal");
    }
}
