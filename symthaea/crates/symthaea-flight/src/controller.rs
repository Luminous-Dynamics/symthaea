//! Flight controller: wraps HdcLtcUnifiedNetwork + output projection (16,384D → 4D).
//!
//! The controller uses the full 16,384D HDC-LTC temporal dynamics engine.
//! Sensor HVs are evolved through the network, then a linear output projection
//! maps the final-layer HV to 4D motor commands (thrust + 3 moments).

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{
    ContinuousHV, HdcLtcUnifiedNetwork, HDC_DIMENSION,
    UnifiedConfig, UnifiedNetworkConfig,
};

use crate::types::{FlightConfig, QuadrotorCommand};

/// Flight controller wrapping an HdcLtcUnifiedNetwork + linear output head.
///
/// Forward pass:
/// 1. `network.evolve_closed_form(dt, &sensor_hv)` — O(D) temporal evolution
/// 2. `output = network.output()` — bundled final layer (16,384D)
/// 3. `output_weights @ output + output_bias` → 4D raw
/// 4. Activations: sigmoid for thrust, tanh for moments
pub struct FlightController {
    /// The temporal dynamics engine — full 16,384D HDC-LTC.
    network: HdcLtcUnifiedNetwork,
    /// Output projection weights: 4 rows × 16,384 columns (flat row-major).
    output_weights: Vec<f32>,
    /// Output bias (4D).
    output_bias: [f32; 4],
    /// Current learning rate (modulated by FEP agent).
    learning_rate: f32,
}

impl FlightController {
    /// Create a new controller from a genesis seed and config.
    pub fn new(genesis: &GenesisSeed, config: &FlightConfig) -> Self {
        let neuron_config = UnifiedConfig {
            tau_base: 0.05,               // 50ms — faster than default for reactive control
            backbone_tau: 0.3,             // Moderate state dependency
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
        let total_weights = 4 * HDC_DIMENSION;
        let weight_hv = genesis.hv("flight::output_weights", total_weights);
        let mut output_weights = weight_hv.values;
        // Scale down: initial weights should be small so output starts near zero
        for w in &mut output_weights {
            *w *= 0.01;
        }

        // Bias initialized to produce hover command
        let output_bias = [
            QuadrotorCommand::HOVER_THRUST, // Default thrust at hover
            0.0,                            // Zero roll moment
            0.0,                            // Zero pitch moment
            0.0,                            // Zero yaw moment
        ];

        Self {
            network,
            output_weights,
            output_bias,
            learning_rate: config.learning_rate,
        }
    }

    /// Forward pass: evolve the network with sensor input and produce motor command.
    ///
    /// - `sensor_hv`: 16,384D ContinuousHV from the encoder
    /// - `dt`: timestep in seconds (typically 0.002 for 500Hz)
    pub fn forward(&mut self, sensor_hv: &ContinuousHV, dt: f32) -> QuadrotorCommand {
        // 1. Evolve network dynamics
        self.network.evolve_closed_form(dt, sensor_hv);

        // 2. Get bundled final-layer output (16,384D)
        let output_hv = self.network.output();
        let hv_values = output_hv.as_slice();

        // 3. Linear projection: output_weights @ hv + bias → 4D
        let mut raw = [0.0f32; 4];
        for i in 0..4 {
            let row_offset = i * HDC_DIMENSION;
            let mut sum = self.output_bias[i];
            for j in 0..HDC_DIMENSION {
                sum += self.output_weights[row_offset + j] * hv_values[j];
            }
            raw[i] = sum;
        }

        // 4. Activations
        let thrust = sigmoid(raw[0]) * QuadrotorCommand::MAX_THRUST;
        let roll = fast_tanh(raw[1]) * QuadrotorCommand::MAX_MOMENT_RP;
        let pitch = fast_tanh(raw[2]) * QuadrotorCommand::MAX_MOMENT_RP;
        let yaw = fast_tanh(raw[3]) * QuadrotorCommand::MAX_MOMENT_YAW;

        QuadrotorCommand {
            thrust,
            roll_moment: roll,
            pitch_moment: pitch,
            yaw_moment: yaw,
        }
    }

    /// Train the output projection and network weights via full BPTT.
    ///
    /// Uses `target` as the ground-truth command (from PD baseline).
    /// Backpropagates through the output projection, then through ALL network layers.
    pub fn train_step(
        &mut self,
        sensor_hv: &ContinuousHV,
        target: &QuadrotorCommand,
        dt: f32,
        lr_override: Option<f32>,
    ) {
        let lr = lr_override.unwrap_or(self.learning_rate);

        // Forward pass to get current output
        let output_hv = self.network.output();
        let hv_values = output_hv.as_slice();

        // Compute current raw outputs
        let mut raw = [0.0f32; 4];
        for i in 0..4 {
            let row_offset = i * HDC_DIMENSION;
            let mut sum = self.output_bias[i];
            for j in 0..HDC_DIMENSION {
                sum += self.output_weights[row_offset + j] * hv_values[j];
            }
            raw[i] = sum;
        }

        // Compute activated outputs
        let pred = [
            sigmoid(raw[0]) * QuadrotorCommand::MAX_THRUST,
            fast_tanh(raw[1]) * QuadrotorCommand::MAX_MOMENT_RP,
            fast_tanh(raw[2]) * QuadrotorCommand::MAX_MOMENT_RP,
            fast_tanh(raw[3]) * QuadrotorCommand::MAX_MOMENT_YAW,
        ];

        let tgt = [target.thrust, target.roll_moment, target.pitch_moment, target.yaw_moment];

        // Compute error (pred - target)
        let errors = [
            pred[0] - tgt[0],
            pred[1] - tgt[1],
            pred[2] - tgt[2],
            pred[3] - tgt[3],
        ];

        // Backprop through activations to get raw gradients
        let s0 = sigmoid(raw[0]);
        let d_raw = [
            errors[0] * s0 * (1.0 - s0) * QuadrotorCommand::MAX_THRUST,
            errors[1] * (1.0 - fast_tanh(raw[1]).powi(2)) * QuadrotorCommand::MAX_MOMENT_RP,
            errors[2] * (1.0 - fast_tanh(raw[2]).powi(2)) * QuadrotorCommand::MAX_MOMENT_RP,
            errors[3] * (1.0 - fast_tanh(raw[3]).powi(2)) * QuadrotorCommand::MAX_MOMENT_YAW,
        ];

        // Update output weights: dW[i][j] = -lr * d_raw[i] * hv[j]
        for i in 0..4 {
            let row_offset = i * HDC_DIMENSION;
            for j in 0..HDC_DIMENSION {
                self.output_weights[row_offset + j] -= lr * d_raw[i] * hv_values[j];
            }
            self.output_bias[i] -= lr * d_raw[i];
        }

        // Backprop through network: compute gradient HV in output space
        // dL/d(hv) = sum_i (d_raw[i] * W[i])
        let dim = HDC_DIMENSION;
        let mut grad_hv_values = vec![0.0f32; dim];
        for i in 0..4 {
            let row_offset = i * dim;
            for j in 0..dim {
                grad_hv_values[j] += d_raw[i] * self.output_weights[row_offset + j];
            }
        }

        // Full BPTT: iterate from last layer backward through all layers
        let n_layers = self.network.n_layers();
        let mut target_hv = output_hv.add(&ContinuousHV::from_vec(grad_hv_values).scale(-1.0));

        for layer_idx in (0..n_layers).rev() {
            let layer_input = self.network.layer_input(layer_idx, sensor_hv);

            // Get the layer output before this layer's update (for propagation)
            let layer_output = if layer_idx > 0 {
                self.network
                    .output_at_layer(layer_idx - 1)
                    .cloned()
            } else {
                None
            };

            if let Some(layer) = self.network.layer_mut(layer_idx) {
                for neuron in layer.iter_mut() {
                    let grads = neuron.backward(&layer_input, &target_hv, dt);
                    neuron.apply_gradients(&grads, lr);
                }
            }

            // Propagate target to previous layer using difference-target approximation
            if let Some(prev_output) = layer_output {
                let current_output_at_layer = self.network
                    .output_at_layer(layer_idx)
                    .cloned()
                    .unwrap_or_else(|| ContinuousHV::zero(dim));
                let diff = target_hv.subtract(&current_output_at_layer);
                target_hv = prev_output.add(&diff.scale(0.5));
            }
        }
    }

    /// Modulate all neuron time constants by a factor.
    ///
    /// - `factor < 1.0`: faster adaptation (high surprise)
    /// - `factor > 1.0`: slower, more stable (low surprise)
    ///
    /// Directly sets `tau_base` on each neuron via `set_tau_base()`.
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

    /// Set the learning rate directly.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr.clamp(1e-6, 0.1);
    }

    /// Get current learning rate.
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Reset network state (for new episode).
    pub fn reset(&mut self) {
        self.network.reset();
    }

    /// Get network statistics.
    pub fn stats(&self) -> symthaea_core::hdc::UnifiedNetworkStats {
        self.network.stats()
    }

    /// Get the underlying network (for advanced inspection).
    pub fn network(&self) -> &HdcLtcUnifiedNetwork {
        &self.network
    }
}

/// Fast sigmoid activation.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Fast tanh approximation (Padé).
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
    use crate::types::FlightConfig;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-flight-controller")
    }

    #[test]
    fn test_controller_forward_valid_output() {
        let genesis = test_genesis();
        let config = FlightConfig::default();
        let mut ctrl = FlightController::new(&genesis, &config);

        // Create a dummy sensor HV
        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        let cmd = ctrl.forward(&sensor, 0.002);

        // Thrust should be in valid range
        assert!(cmd.thrust >= 0.0, "Thrust should be non-negative: {}", cmd.thrust);
        assert!(
            cmd.thrust <= QuadrotorCommand::MAX_THRUST,
            "Thrust should be <= MAX: {}",
            cmd.thrust
        );

        // Moments should be in valid range
        assert!(cmd.roll_moment.abs() <= QuadrotorCommand::MAX_MOMENT_RP);
        assert!(cmd.pitch_moment.abs() <= QuadrotorCommand::MAX_MOMENT_RP);
        assert!(cmd.yaw_moment.abs() <= QuadrotorCommand::MAX_MOMENT_YAW);
    }

    #[test]
    fn test_controller_genesis_determinism() {
        let genesis = test_genesis();
        let config = FlightConfig::default();
        let mut ctrl1 = FlightController::new(&genesis, &config);
        let mut ctrl2 = FlightController::new(&genesis, &config);

        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        let cmd1 = ctrl1.forward(&sensor, 0.002);
        let cmd2 = ctrl2.forward(&sensor, 0.002);

        assert!(
            (cmd1.thrust - cmd2.thrust).abs() < 1e-6,
            "Same genesis → same output: {} vs {}",
            cmd1.thrust,
            cmd2.thrust
        );
        assert!((cmd1.roll_moment - cmd2.roll_moment).abs() < 1e-6);
    }

    #[test]
    fn test_controller_training_reduces_error() {
        let genesis = test_genesis();
        let config = FlightConfig::default();
        let mut ctrl = FlightController::new(&genesis, &config);

        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        let target = QuadrotorCommand::hover();

        // Get initial prediction
        let initial = ctrl.forward(&sensor, 0.002);
        let initial_err = (initial.thrust - target.thrust).powi(2)
            + (initial.roll_moment - target.roll_moment).powi(2)
            + (initial.pitch_moment - target.pitch_moment).powi(2)
            + (initial.yaw_moment - target.yaw_moment).powi(2);

        // Train for several steps
        for _ in 0..50 {
            ctrl.forward(&sensor, 0.002);
            ctrl.train_step(&sensor, &target, 0.002, Some(0.01));
        }

        // Get final prediction
        let final_cmd = ctrl.forward(&sensor, 0.002);
        let final_err = (final_cmd.thrust - target.thrust).powi(2)
            + (final_cmd.roll_moment - target.roll_moment).powi(2)
            + (final_cmd.pitch_moment - target.pitch_moment).powi(2)
            + (final_cmd.yaw_moment - target.yaw_moment).powi(2);

        assert!(
            final_err < initial_err,
            "Training should reduce error: initial={initial_err:.6}, final={final_err:.6}"
        );
    }

    #[test]
    fn test_controller_reset() {
        let genesis = test_genesis();
        let config = FlightConfig::default();
        let mut ctrl = FlightController::new(&genesis, &config);

        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);
        ctrl.forward(&sensor, 0.002);
        ctrl.forward(&sensor, 0.002);

        ctrl.reset();
        // After reset, network state should be back to zero
        let stats = ctrl.stats();
        assert!(stats.avg_state_norm < 1e-6, "Reset should zero state norms");
    }

    #[test]
    fn test_sigmoid_range() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_fast_tanh_range() {
        assert!(fast_tanh(0.0).abs() < 1e-6);
        assert!((fast_tanh(10.0) - 1.0).abs() < 0.01);
        assert!((fast_tanh(-10.0) + 1.0).abs() < 0.01);
    }
}
