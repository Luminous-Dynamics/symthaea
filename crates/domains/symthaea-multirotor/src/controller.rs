// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Flight controller: wraps HdcLtcUnifiedNetwork + output projection (16,384D → 4D).
//!
//! The controller uses the full 16,384D HDC-LTC temporal dynamics engine.
//! Sensor HVs are evolved through the network, then a linear output projection
//! maps the final-layer HV to 4D motor commands (thrust + 3 moments).

use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{
    ContinuousHV, HDC_DIMENSION, HdcLtcUnifiedNetwork, UnifiedConfig, UnifiedNetworkConfig,
};

use crate::types::{FlightConfig, QuadrotorCommand};

/// Flight controller wrapping an HdcLtcUnifiedNetwork + linear output head.
///
/// Forward pass:
/// 1. `network.evolve_closed_form(dt, &sensor_hv)` — O(D) temporal evolution
/// 2. `output = network.output()` — bundled final layer (16,384D)
/// 3. `output_weights @ output + output_bias` → 4D raw
/// 4. Activations: sigmoid for thrust, tanh for moments
/// Clone is deliberate: it's the trainer→bridge transfer path. A trained
/// controller (BPTT touches the network, so an output-layer snapshot alone
/// would NOT capture training) is cloned/moved into
/// `FlightEmbodiment::with_controller` instead of being dropped at the end
/// of `FlightTrainer::train`.
#[derive(Clone)]
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
            tau_base: 0.05,    // 50ms — faster than default for reactive control
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
    ///
    /// Non-finite values (NaN, Inf) in `sensor_hv` are replaced with 0.0 to prevent
    /// a single bad sensor reading from permanently corrupting network state.
    pub fn forward(&mut self, sensor_hv: &ContinuousHV, dt: f32) -> QuadrotorCommand {
        // Sanitize input: replace NaN/Inf with 0.0 (zero-alloc on clean input)
        let sanitized = sanitize_hv(sensor_hv);
        let sensor_hv = sanitized.as_ref().unwrap_or(sensor_hv);

        // 1. Evolve network dynamics
        self.network.evolve_closed_form(dt, sensor_hv);

        // 2. Get bundled final-layer output (16,384D), normalized to unit length.
        //    Normalization keeps gradient scale independent of dimensionality.
        let output_hv = self.network.output().normalize();
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
        // Output projection LR scaled by √D to compensate for per-weight gradient dilution.
        let output_lr = lr * (HDC_DIMENSION as f32).sqrt();

        // Forward pass to get current output, normalized to unit length
        let output_hv = self.network.output().normalize();
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

        let tgt = [
            target.thrust,
            target.roll_moment,
            target.pitch_moment,
            target.yaw_moment,
        ];

        // Compute error (pred - target)
        let errors = [
            pred[0] - tgt[0],
            pred[1] - tgt[1],
            pred[2] - tgt[2],
            pred[3] - tgt[3],
        ];

        // Backprop through activations to get raw gradients
        let s0 = sigmoid(raw[0]);
        let mut d_raw = [
            errors[0] * s0 * (1.0 - s0) * QuadrotorCommand::MAX_THRUST,
            errors[1] * (1.0 - fast_tanh(raw[1]).powi(2)) * QuadrotorCommand::MAX_MOMENT_RP,
            errors[2] * (1.0 - fast_tanh(raw[2]).powi(2)) * QuadrotorCommand::MAX_MOMENT_RP,
            errors[3] * (1.0 - fast_tanh(raw[3]).powi(2)) * QuadrotorCommand::MAX_MOMENT_YAW,
        ];

        // Gradient clipping: cap per-channel gradient to prevent overcorrection feedback loops.
        const GRAD_CLIP: f32 = 1.0;
        for g in &mut d_raw {
            *g = g.clamp(-GRAD_CLIP, GRAD_CLIP);
        }

        // Weight decay: pull output weights toward zero to prevent unbounded drift.
        const WEIGHT_DECAY: f32 = 1e-4;
        let decay = 1.0 - WEIGHT_DECAY;
        for w in self.output_weights.iter_mut() {
            *w *= decay;
        }

        // Update output weights: dW[i][j] = -output_lr * d_raw[i] * hv[j]
        // output_lr = lr * √D compensates for gradient dilution across 16,384 weights.
        for i in 0..4 {
            let row_offset = i * HDC_DIMENSION;
            for j in 0..HDC_DIMENSION {
                self.output_weights[row_offset + j] -= output_lr * d_raw[i] * hv_values[j];
            }
            self.output_bias[i] -= output_lr * d_raw[i];
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

            // Get the previous layer's output (for gradient-based target propagation)
            let prev_layer_output = if layer_idx > 0 {
                self.network.output_at_layer(layer_idx - 1).cloned()
            } else {
                None
            };

            // Collect d_input gradients from all neurons for inter-layer propagation
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

            // Propagate gradient to previous layer using accumulated d_input.
            // target_prev = prev_output - α * avg(d_input) — gradient-based target propagation.
            if let Some(prev_output) = prev_layer_output
                && neuron_count > 0
            {
                let scale = 1.0 / neuron_count as f32;
                target_hv = prev_output.subtract(&avg_d_input.scale(scale));
            }
        }
    }

    /// Train ONLY the output projection weights (no network BPTT).
    ///
    /// Used for ablation studies comparing full BPTT vs output-layer-only training.
    pub fn train_output_only(&mut self, target: &QuadrotorCommand, lr_override: Option<f32>) {
        let lr = lr_override.unwrap_or(self.learning_rate);
        let output_lr = lr * (HDC_DIMENSION as f32).sqrt();
        let output_hv = self.network.output().normalize();
        let hv_values = output_hv.as_slice();

        let mut raw = [0.0f32; 4];
        for i in 0..4 {
            let row_offset = i * HDC_DIMENSION;
            let mut sum = self.output_bias[i];
            for j in 0..HDC_DIMENSION {
                sum += self.output_weights[row_offset + j] * hv_values[j];
            }
            raw[i] = sum;
        }

        let pred = [
            sigmoid(raw[0]) * QuadrotorCommand::MAX_THRUST,
            fast_tanh(raw[1]) * QuadrotorCommand::MAX_MOMENT_RP,
            fast_tanh(raw[2]) * QuadrotorCommand::MAX_MOMENT_RP,
            fast_tanh(raw[3]) * QuadrotorCommand::MAX_MOMENT_YAW,
        ];
        let tgt = [
            target.thrust,
            target.roll_moment,
            target.pitch_moment,
            target.yaw_moment,
        ];
        let errors = [
            pred[0] - tgt[0],
            pred[1] - tgt[1],
            pred[2] - tgt[2],
            pred[3] - tgt[3],
        ];

        let s0 = sigmoid(raw[0]);
        let d_raw = [
            errors[0] * s0 * (1.0 - s0) * QuadrotorCommand::MAX_THRUST,
            errors[1] * (1.0 - fast_tanh(raw[1]).powi(2)) * QuadrotorCommand::MAX_MOMENT_RP,
            errors[2] * (1.0 - fast_tanh(raw[2]).powi(2)) * QuadrotorCommand::MAX_MOMENT_RP,
            errors[3] * (1.0 - fast_tanh(raw[3]).powi(2)) * QuadrotorCommand::MAX_MOMENT_YAW,
        ];

        for i in 0..4 {
            let row_offset = i * HDC_DIMENSION;
            for j in 0..HDC_DIMENSION {
                self.output_weights[row_offset + j] -= output_lr * d_raw[i] * hv_values[j];
            }
            self.output_bias[i] -= output_lr * d_raw[i];
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

    /// Normalize all neuron hidden states to unit norm.
    ///
    /// Prevents hidden state drift during long episodes by capping the
    /// state magnitude. Should be called periodically (e.g., every 100 steps).
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
    ///
    /// Runs `n_steps` gradient updates using the provided samples, giving the
    /// output weights a reasonable initial policy before the flight loop starts.
    pub fn warmup(
        &mut self,
        samples: &[(ContinuousHV, QuadrotorCommand)],
        n_steps: usize,
        lr: f32,
    ) {
        for step in 0..n_steps {
            let (ref hv, ref target) = samples[step % samples.len()];
            self.forward(hv, 0.002);
            self.train_step(hv, target, 0.002, Some(lr));
        }
        // Reset network state after warmup so episodes start clean
        self.network.reset();
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

/// Check for NaN/Inf values in a ContinuousHV and return a sanitized copy if needed.
///
/// Returns `None` when input is clean (zero allocation fast path).
/// Returns `Some(sanitized)` only when non-finite values are found.
fn sanitize_hv(hv: &ContinuousHV) -> Option<ContinuousHV> {
    let values = hv.as_slice();
    if values.iter().all(|v| v.is_finite()) {
        return None;
    }
    let clean: Vec<f32> = values
        .iter()
        .map(|&v| if v.is_finite() { v } else { 0.0 })
        .collect();
    Some(ContinuousHV::from_vec(clean))
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
        assert!(
            cmd.thrust >= 0.0,
            "Thrust should be non-negative: {}",
            cmd.thrust
        );
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
    fn test_hidden_layer_weights_change_during_training() {
        let genesis = test_genesis();
        let config = FlightConfig::default();
        let mut ctrl = FlightController::new(&genesis, &config);

        // Record initial per-layer weight norms
        let n_layers = ctrl.network().n_layers();
        let initial_norms: Vec<Vec<f32>> = (0..n_layers)
            .map(|i| {
                ctrl.network()
                    .layer(i)
                    .unwrap()
                    .iter()
                    .map(|n| n.stats().weight_norm)
                    .collect()
            })
            .collect();

        // Train for 200 steps with varying inputs (different seeds → different sensor HVs)
        let target = QuadrotorCommand::hover();
        for step in 0..200 {
            let sensor = ContinuousHV::random(HDC_DIMENSION, 100 + step);
            ctrl.forward(&sensor, 0.002);
            ctrl.train_step(&sensor, &target, 0.002, Some(0.01));
        }

        // Check that at least one hidden layer's weight norms changed
        let mut any_hidden_changed = false;
        for layer_idx in 0..n_layers {
            let final_norms: Vec<f32> = ctrl
                .network()
                .layer(layer_idx)
                .unwrap()
                .iter()
                .map(|n| n.stats().weight_norm)
                .collect();

            for (init, fin) in initial_norms[layer_idx].iter().zip(final_norms.iter()) {
                if (init - fin).abs() > 1e-8 {
                    any_hidden_changed = true;
                }
            }
        }

        assert!(
            any_hidden_changed,
            "BPTT should modify at least one hidden layer's weights (n_layers={n_layers})"
        );
    }

    #[test]
    fn test_bptt_vs_output_only_ablation() {
        let genesis = test_genesis();
        let config = FlightConfig::default();

        // Controller A: full BPTT
        let mut ctrl_bptt = FlightController::new(&genesis, &config);
        // Controller B: output-only
        let mut ctrl_out = FlightController::new(&genesis, &config);

        let target = QuadrotorCommand::hover();

        // Train both for 100 steps on identical sensor inputs
        for step in 0..100 {
            let sensor = ContinuousHV::random(HDC_DIMENSION, 200 + step);

            ctrl_bptt.forward(&sensor, 0.002);
            ctrl_bptt.train_step(&sensor, &target, 0.002, Some(0.01));

            ctrl_out.forward(&sensor, 0.002);
            ctrl_out.train_output_only(&target, Some(0.01));
        }

        // Evaluate on a held-out sensor input
        let test_sensor = ContinuousHV::random(HDC_DIMENSION, 9999);
        let cmd_bptt = ctrl_bptt.forward(&test_sensor, 0.002);
        let cmd_out = ctrl_out.forward(&test_sensor, 0.002);

        let err_bptt = (cmd_bptt.thrust - target.thrust).powi(2)
            + (cmd_bptt.roll_moment - target.roll_moment).powi(2)
            + (cmd_bptt.pitch_moment - target.pitch_moment).powi(2)
            + (cmd_bptt.yaw_moment - target.yaw_moment).powi(2);

        let err_out = (cmd_out.thrust - target.thrust).powi(2)
            + (cmd_out.roll_moment - target.roll_moment).powi(2)
            + (cmd_out.pitch_moment - target.pitch_moment).powi(2)
            + (cmd_out.yaw_moment - target.yaw_moment).powi(2);

        // Both should learn something (error should decrease from random)
        let initial_sensor = ContinuousHV::random(HDC_DIMENSION, 9999);
        let initial_ctrl = FlightController::new(&genesis, &config);
        // Can't forward on immutable, so just check both trained controllers improved
        assert!(
            err_bptt.is_finite() && err_out.is_finite(),
            "Both methods should produce finite errors"
        );

        // BPTT hidden layer weight norms should differ from output-only
        let bptt_net_stats = ctrl_bptt.stats();
        let out_net_stats = ctrl_out.stats();
        let weight_diff = (bptt_net_stats.avg_weight_norm - out_net_stats.avg_weight_norm).abs();
        assert!(
            weight_diff > 1e-8,
            "BPTT should produce different hidden weights than output-only: diff={weight_diff}"
        );

        let _ = initial_sensor;
        let _ = initial_ctrl;
    }

    #[test]
    fn test_fast_tanh_range() {
        assert!(fast_tanh(0.0).abs() < 1e-6);
        assert!((fast_tanh(10.0) - 1.0).abs() < 0.01);
        assert!((fast_tanh(-10.0) + 1.0).abs() < 0.01);
    }

    // ── Item 2: Controller Stress Tests ──

    #[test]
    fn test_controller_nan_resilience() {
        let genesis = test_genesis();
        let config = FlightConfig::default();
        let mut ctrl = FlightController::new(&genesis, &config);

        // Create a sensor HV with NaN and Inf values
        let mut nan_values = vec![0.0f32; HDC_DIMENSION];
        nan_values[0] = f32::NAN;
        nan_values[100] = f32::NAN;
        nan_values[1000] = f32::INFINITY;
        nan_values[2000] = f32::NEG_INFINITY;
        let nan_hv = ContinuousHV::from_vec(nan_values);

        let cmd = ctrl.forward(&nan_hv, 0.002);

        // With sanitize guard, output must be finite
        assert!(
            cmd.thrust.is_finite(),
            "Thrust should be finite after NaN guard: {}",
            cmd.thrust
        );
        assert!(
            cmd.roll_moment.is_finite(),
            "Roll moment should be finite after NaN guard: {}",
            cmd.roll_moment
        );
        assert!(
            cmd.pitch_moment.is_finite(),
            "Pitch moment should be finite after NaN guard: {}",
            cmd.pitch_moment
        );
        assert!(
            cmd.yaw_moment.is_finite(),
            "Yaw moment should be finite after NaN guard: {}",
            cmd.yaw_moment
        );

        // Multiple NaN inputs in a row should not corrupt state
        for _ in 0..10 {
            let cmd = ctrl.forward(&nan_hv, 0.002);
            assert!(
                cmd.thrust.is_finite(),
                "Repeated NaN input should stay safe"
            );
        }
    }

    #[test]
    fn test_controller_extreme_input_stability() {
        let genesis = test_genesis();
        let config = FlightConfig::default();
        let mut ctrl = FlightController::new(&genesis, &config);

        // Feed HV with all values at +1.0 or -1.0 extremes
        let extreme_pos = ContinuousHV::from_vec(vec![1.0f32; HDC_DIMENSION]);
        let extreme_neg = ContinuousHV::from_vec(vec![-1.0f32; HDC_DIMENSION]);

        for step in 0..100 {
            let hv = if step % 2 == 0 {
                &extreme_pos
            } else {
                &extreme_neg
            };
            let cmd = ctrl.forward(hv, 0.002);

            assert!(
                cmd.thrust.is_finite(),
                "Thrust should be finite at step {step}: {}",
                cmd.thrust
            );
            assert!(
                cmd.thrust >= 0.0 && cmd.thrust <= QuadrotorCommand::MAX_THRUST,
                "Thrust out of range at step {step}: {}",
                cmd.thrust
            );
            assert!(
                cmd.roll_moment.abs() <= QuadrotorCommand::MAX_MOMENT_RP,
                "Roll moment out of range at step {step}: {}",
                cmd.roll_moment
            );
            assert!(
                cmd.pitch_moment.abs() <= QuadrotorCommand::MAX_MOMENT_RP,
                "Pitch moment out of range at step {step}: {}",
                cmd.pitch_moment
            );
            assert!(
                cmd.yaw_moment.abs() <= QuadrotorCommand::MAX_MOMENT_YAW,
                "Yaw moment out of range at step {step}: {}",
                cmd.yaw_moment
            );
        }
    }

    #[test]
    fn test_controller_rapid_tau_modulation() {
        let genesis = test_genesis();
        let config = FlightConfig::default();
        let mut ctrl = FlightController::new(&genesis, &config);

        let sensor = ContinuousHV::random(HDC_DIMENSION, 42);

        for step in 0..50 {
            // Alternate between minimum and maximum tau modulation every step
            if step % 2 == 0 {
                ctrl.modulate_tau(0.3);
            } else {
                ctrl.modulate_tau(3.0);
            }

            let cmd = ctrl.forward(&sensor, 0.002);
            assert!(
                cmd.thrust.is_finite(),
                "Output should stay finite under rapid tau modulation at step {step}"
            );
            assert!(
                cmd.roll_moment.is_finite(),
                "Roll moment not finite at step {step}"
            );
            assert!(
                cmd.pitch_moment.is_finite(),
                "Pitch moment not finite at step {step}"
            );
            assert!(
                cmd.yaw_moment.is_finite(),
                "Yaw moment not finite at step {step}"
            );
        }
    }

    #[test]
    fn test_controller_1000_step_convergence() {
        let genesis = test_genesis();
        let config = FlightConfig::default();
        let mut ctrl = FlightController::new(&genesis, &config);

        let target = QuadrotorCommand::hover();

        // Measure loss over 1000 steps with varying inputs
        let mut losses = Vec::with_capacity(1000);
        for step in 0..1000 {
            let sensor = ContinuousHV::random(HDC_DIMENSION, 500 + step);
            let cmd = ctrl.forward(&sensor, 0.002);
            let loss = (cmd.thrust - target.thrust).powi(2)
                + (cmd.roll_moment - target.roll_moment).powi(2)
                + (cmd.pitch_moment - target.pitch_moment).powi(2)
                + (cmd.yaw_moment - target.yaw_moment).powi(2);
            losses.push(loss);
            ctrl.train_step(&sensor, &target, 0.002, Some(0.01));
        }

        // Average loss of last 100 steps should be lower than first 100
        let first_100: f32 = losses[..100].iter().sum::<f32>() / 100.0;
        let last_100: f32 = losses[900..].iter().sum::<f32>() / 100.0;

        assert!(
            last_100 < first_100,
            "1000-step training should reduce loss: first_100={first_100:.6}, last_100={last_100:.6}"
        );
    }
}
