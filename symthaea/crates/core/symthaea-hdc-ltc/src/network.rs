// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # HDC-LTC Network
//!
//! A multi-layer network of HDC-LTC neurons with optional layer binding,
//! skip connections, and **irregular-time support**.
//!
//! ## Irregular-Time Stepping
//!
//! Classical sequence models require data at fixed, synchronised intervals.
//! Because the LTC/CfC closed-form solution is O(1) in `dt`, this network
//! supports data arriving at any cadence via [`HdcLtcUnifiedNetwork::step_with_timestamp`].
//!
//! ```text
//! ros_node.subscribe(|msg| {
//!     let t = msg.header.stamp.to_secs_f64();
//!     let I_t = probe.project(&encoder.encode(&msg.data));
//!     network.step_with_timestamp(t, &I_t);
//! });
//! ```

use crate::config::NetworkConfig;
use crate::continuous_hv::ContinuousHV;
use crate::neuron::HdcLtcUnifiedNeuron;
use serde::{Deserialize, Serialize};

/// A multi-layer network of unified HDC-LTC neurons.
///
/// Each layer is a collection of neurons that all receive the same input.
/// The output of a layer is the bundled (averaged) states of its neurons.
/// Inter-layer connections optionally use HDC binding for decorrelation.
///
/// Use [`step`][Self::step] for fixed-interval stepping and
/// [`step_with_timestamp`][Self::step_with_timestamp] when sensor data
/// arrives at irregular wall-clock times.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdcLtcUnifiedNetwork {
    /// Layers of neurons.
    layers: Vec<Vec<HdcLtcUnifiedNeuron>>,
    /// Inter-layer binding vectors (one per layer).
    layer_bindings: Vec<ContinuousHV>,
    /// Configuration.
    config: NetworkConfig,
    /// Cached layer outputs (bundled neuron states).
    layer_outputs: Vec<ContinuousHV>,
    /// Last wall-clock time used by `step_with_timestamp` (seconds).
    /// `None` before the first call.
    last_timestamp: Option<f64>,
    /// Step timing configuration for validation.
    timing_config: StepTimingConfig,
    /// Total number of `step` / `step_with_timestamp` calls.
    step_count: u64,
}

/// Timing validation configuration for irregular-time updates.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct StepTimingConfig {
    /// Minimum time step allowed (seconds).
    pub min_dt: f32,
    /// Maximum time step allowed (seconds).
    pub max_dt: f32,
    /// Whether to reject or clamp backward timestamps.
    pub reject_backward_time: bool,
}

impl Default for StepTimingConfig {
    fn default() -> Self {
        Self {
            min_dt: 1e-6,
            max_dt: 10.0,
            reject_backward_time: true,
        }
    }
}

impl HdcLtcUnifiedNetwork {
    /// Create a new network with the given configuration and seed.
    pub fn new(config: NetworkConfig, seed: u64) -> Self {
        let mut layers = Vec::new();
        let mut current_seed = seed;

        for &layer_size in &config.layer_sizes {
            let layer: Vec<HdcLtcUnifiedNeuron> = (0..layer_size)
                .map(|_| {
                    current_seed += 1;
                    HdcLtcUnifiedNeuron::new(config.neuron_config.clone(), current_seed)
                })
                .collect();
            layers.push(layer);
        }

        let dim = config.neuron_config.dim;
        let layer_bindings: Vec<ContinuousHV> = (0..config.layer_sizes.len())
            .map(|i| ContinuousHV::new_random(dim, seed + 10000 + i as u64))
            .collect();

        let layer_outputs = config
            .layer_sizes
            .iter()
            .map(|_| ContinuousHV::new(dim))
            .collect();

        Self {
            layers,
            layer_bindings,
            layer_outputs,
            config,
            last_timestamp: None,
            timing_config: StepTimingConfig::default(),
            step_count: 0,
        }
    }

    /// Set a custom timing configuration for irregular-time stepping.
    pub fn set_timing_config(&mut self, timing_config: StepTimingConfig) {
        self.timing_config = timing_config;
    }

    /// Get the current timing configuration.
    pub fn timing_config(&self) -> StepTimingConfig {
        self.timing_config
    }

    /// **Fixed-interval forward pass**: evolve all layers by `dt` seconds.
    ///
    /// Layer 0 receives the input directly. Subsequent layers receive
    /// the bundled output of the previous layer (optionally bound with
    /// a layer-binding vector and skip-connected with the original input).
    ///
    /// For irregular sensor cadences use [`step_with_timestamp`][Self::step_with_timestamp].
    pub fn step(&mut self, dt: f32, input: &ContinuousHV) {
        // Layer 0: direct input
        for neuron in &mut self.layers[0] {
            neuron.evolve_closed_form(dt, input);
        }
        self.cache_layer_output(0);

        // Subsequent layers
        for layer_idx in 1..self.layers.len() {
            let layer_input = self.compute_layer_input(layer_idx, input);
            for neuron in &mut self.layers[layer_idx] {
                neuron.evolve_closed_form(dt, &layer_input);
            }
            self.cache_layer_output(layer_idx);
        }

        self.step_count += 1;
    }

    /// **Irregular-time forward pass** — computes `dt` from the real elapsed wall-clock.
    ///
    /// Equivalent to calling [`step`][Self::step] with `dt = t_now - t_prev`.
    /// On the first call, `dt` defaults to `tau_base` of the first neuron layer
    /// (a sensible "cold start" interval).
    ///
    /// # Arguments
    /// * `t_now` — current time in seconds (monotonically increasing)
    /// * `input` — sensory hypervector I(t) from the probe stream
    ///
    /// # Example (ROS2 subscriber)
    /// ```rust,ignore
    /// ros_node.subscribe("/camera/embedding", |msg| {
    ///     let t = msg.stamp_secs;
    ///     let I_t = probe.project(&msg.embedding);
    ///     network.step_with_timestamp(t, &I_t);
    /// });
    /// ```
    pub fn step_with_timestamp(&mut self, t_now: f64, input: &ContinuousHV) {
        let dt = match self.last_timestamp {
            Some(t_prev) => {
                let elapsed = (t_now - t_prev) as f32;
                if elapsed < 0.0 {
                    if self.timing_config.reject_backward_time {
                        // Reject backward time, keeping state unchanged (effectively dt = 0)
                        return;
                    } else {
                        // Clamp to min_dt
                        self.timing_config.min_dt
                    }
                } else {
                    elapsed.clamp(self.timing_config.min_dt, self.timing_config.max_dt)
                }
            }
            None => {
                // Cold-start: use tau_base as a reasonable first dt.
                self.config
                    .neuron_config
                    .tau_base
                    .clamp(self.timing_config.min_dt, self.timing_config.max_dt)
            }
        };
        self.last_timestamp = Some(t_now);
        self.step(dt, input);
    }

    /// Last wall-clock timestamp seen by `step_with_timestamp`.
    pub fn last_timestamp(&self) -> Option<f64> {
        self.last_timestamp
    }

    /// Total number of forward steps taken (via either `step` or `step_with_timestamp`).
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Get the network output (bundled states of the last layer).
    pub fn output(&self) -> ContinuousHV {
        self.layer_outputs
            .last()
            .cloned()
            .unwrap_or_else(|| ContinuousHV::new(self.config.neuron_config.dim))
    }

    /// Get the number of layers.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    /// Get the total number of neurons across all layers.
    pub fn neuron_count(&self) -> usize {
        self.layers.iter().map(|l| l.len()).sum()
    }

    /// Reset all neuron states, cached outputs, and timestamp history.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            for neuron in layer {
                neuron.reset();
            }
        }
        for output in &mut self.layer_outputs {
            *output = ContinuousHV::new(self.config.neuron_config.dim);
        }
        self.last_timestamp = None;
        self.step_count = 0;
    }

    /// Get the network configuration.
    pub fn config(&self) -> &NetworkConfig {
        &self.config
    }

    /// Get a reference to a specific layer.
    pub fn layer(&self, idx: usize) -> Option<&[HdcLtcUnifiedNeuron]> {
        self.layers.get(idx).map(|l| l.as_slice())
    }

    /// Get a mutable reference to a specific layer.
    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut Vec<HdcLtcUnifiedNeuron>> {
        self.layers.get_mut(idx)
    }

    // ─────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ─────────────────────────────────────────────────────────────────────

    /// Cache the bundled (averaged) neuron states for a layer.
    #[inline]
    fn cache_layer_output(&mut self, layer_idx: usize) {
        let neurons = &self.layers[layer_idx];
        let output = &mut self.layer_outputs[layer_idx];
        for v in output.values.iter_mut() {
            *v = 0.0;
        }
        if neurons.is_empty() {
            return;
        }
        let inv_n = 1.0 / neurons.len() as f32;
        for neuron in neurons {
            for (o, &s) in output.values.iter_mut().zip(neuron.state().values.iter()) {
                *o += s;
            }
        }
        for v in output.values.iter_mut() {
            *v *= inv_n;
        }
    }

    /// Compute the input for a layer (layer_idx must be >= 1).
    fn compute_layer_input(&self, layer_idx: usize, original_input: &ContinuousHV) -> ContinuousHV {
        debug_assert!(layer_idx > 0);
        let prev_output = &self.layer_outputs[layer_idx - 1];

        let bound_input = if self.config.use_layer_binding {
            self.layer_bindings[layer_idx].bind(prev_output)
        } else {
            prev_output.clone()
        };

        if self.config.skip_connections {
            ContinuousHV::bundle(&[&bound_input, original_input])
        } else {
            bound_input
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{NetworkConfig, NeuronConfig};

    fn small_network_config() -> NetworkConfig {
        NetworkConfig {
            layer_sizes: vec![2, 3, 2],
            neuron_config: NeuronConfig {
                dim: 256,
                ..NeuronConfig::default()
            },
            use_layer_binding: true,
            skip_connections: false,
        }
    }

    #[test]
    fn test_creation() {
        let net = HdcLtcUnifiedNetwork::new(small_network_config(), 42);
        assert_eq!(net.layer_count(), 3);
        assert_eq!(net.neuron_count(), 7); // 2 + 3 + 2
    }

    #[test]
    fn test_step_changes_output() {
        let mut net = HdcLtcUnifiedNetwork::new(small_network_config(), 42);
        let input = ContinuousHV::new_random(256, 100);

        let before = net.output().clone();
        net.step(0.1, &input);
        let after = net.output();

        // Output should have changed from zero
        assert!(after.norm() > 0.0);
        assert_ne!(before.values, after.values);
    }

    #[test]
    fn test_multiple_steps() {
        let mut net = HdcLtcUnifiedNetwork::new(small_network_config(), 42);
        let input = ContinuousHV::new_random(256, 100);

        for _ in 0..10 {
            net.step(0.05, &input);
        }
        assert!(net.output().norm() > 0.0);
    }

    #[test]
    fn test_reset() {
        let mut net = HdcLtcUnifiedNetwork::new(small_network_config(), 42);
        let input = ContinuousHV::new_random(256, 100);
        net.step(0.1, &input);
        assert!(net.output().norm() > 0.0);

        net.reset();
        assert!((net.output().norm()).abs() < 1e-10);
    }

    #[test]
    fn test_layer_access() {
        let net = HdcLtcUnifiedNetwork::new(small_network_config(), 42);
        assert!(net.layer(0).is_some());
        assert_eq!(net.layer(0).unwrap().len(), 2);
        assert!(net.layer(1).is_some());
        assert_eq!(net.layer(1).unwrap().len(), 3);
        assert!(net.layer(3).is_none());
    }

    #[test]
    fn test_skip_connections() {
        let config = NetworkConfig {
            layer_sizes: vec![2, 2],
            neuron_config: NeuronConfig {
                dim: 256,
                ..NeuronConfig::default()
            },
            use_layer_binding: false,
            skip_connections: true,
        };
        let mut net = HdcLtcUnifiedNetwork::new(config, 42);
        let input = ContinuousHV::new_random(256, 100);
        net.step(0.1, &input);
        assert!(net.output().norm() > 0.0);
    }

    // ── Irregular-time stepping ───────────────────────────────────────────────

    #[test]
    fn test_step_with_timestamp_cold_start() {
        let mut net = HdcLtcUnifiedNetwork::new(small_network_config(), 42);
        let input = ContinuousHV::new_random(256, 100);
        assert!(net.last_timestamp().is_none());
        net.step_with_timestamp(1.0, &input);
        assert_eq!(net.last_timestamp(), Some(1.0));
        assert_eq!(net.step_count(), 1);
    }

    #[test]
    fn test_step_with_timestamp_computes_dt() {
        let mut net = HdcLtcUnifiedNetwork::new(small_network_config(), 42);
        let input = ContinuousHV::new_random(256, 100);
        net.step_with_timestamp(0.0, &input);
        net.step_with_timestamp(0.5, &input); // dt = 0.5s
        net.step_with_timestamp(0.6, &input); // dt = 0.1s
        assert_eq!(net.step_count(), 3);
        assert!((net.last_timestamp().unwrap() - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_step_with_timestamp_produces_output() {
        let mut net = HdcLtcUnifiedNetwork::new(small_network_config(), 42);
        let input = ContinuousHV::new_random(256, 100);
        // Irregular timestamps simulating a sporadic sensor
        for (i, &t) in [0.0_f64, 0.017, 0.038, 0.10, 0.102, 0.500]
            .iter()
            .enumerate()
        {
            net.step_with_timestamp(t, &input);
            assert_eq!(net.step_count(), (i + 1) as u64);
        }
        assert!(net.output().norm() > 0.0);
        assert!(net.output().norm().is_finite());
    }

    #[test]
    fn test_step_count_increments() {
        let mut net = HdcLtcUnifiedNetwork::new(small_network_config(), 42);
        let input = ContinuousHV::new_random(256, 100);
        assert_eq!(net.step_count(), 0);
        for i in 1..=5 {
            net.step(0.05, &input);
            assert_eq!(net.step_count(), i);
        }
    }

    #[test]
    fn test_reset_clears_timestamp_and_count() {
        let mut net = HdcLtcUnifiedNetwork::new(small_network_config(), 42);
        let input = ContinuousHV::new_random(256, 100);
        net.step_with_timestamp(1.0, &input);
        net.step_with_timestamp(2.0, &input);
        assert_eq!(net.step_count(), 2);
        assert!(net.last_timestamp().is_some());
        net.reset();
        assert_eq!(net.step_count(), 0);
        assert!(net.last_timestamp().is_none());
    }

    #[test]
    fn test_step_timing_config_validation() {
        let mut net = HdcLtcUnifiedNetwork::new(small_network_config(), 42);
        let input = ContinuousHV::new_random(256, 100);

        net.set_timing_config(StepTimingConfig {
            min_dt: 0.01,
            max_dt: 1.0,
            reject_backward_time: true,
        });

        // 1. Cold start should be clamped if tau_base is out of bounds
        net.step_with_timestamp(1.0, &input);
        // last_timestamp should be set
        assert_eq!(net.last_timestamp(), Some(1.0));
        assert_eq!(net.step_count(), 1);

        // 2. dt below min_dt (0.005s) should clamp to min_dt (0.01s)
        net.step_with_timestamp(1.005, &input);
        assert_eq!(net.last_timestamp(), Some(1.005));
        assert_eq!(net.step_count(), 2);

        // 3. dt above max_dt (5.0s) should clamp to max_dt (1.0s)
        net.step_with_timestamp(6.005, &input);
        assert_eq!(net.last_timestamp(), Some(6.005));
        assert_eq!(net.step_count(), 3);

        // 4. Backward timestamp with reject_backward_time = true should be rejected (step count does not increment, last_timestamp remains unchanged)
        net.step_with_timestamp(5.0, &input);
        assert_eq!(net.last_timestamp(), Some(6.005));
        assert_eq!(net.step_count(), 3);

        // 5. Backward timestamp with reject_backward_time = false should clamp to min_dt (0.01s)
        net.set_timing_config(StepTimingConfig {
            min_dt: 0.01,
            max_dt: 1.0,
            reject_backward_time: false,
        });
        net.step_with_timestamp(5.0, &input);
        assert_eq!(net.last_timestamp(), Some(5.0));
        assert_eq!(net.step_count(), 4);
    }
}

#[cfg(test)]
proptest::proptest! {
    #[test]
    fn test_fuzz_irregular_timestamps(
        timestamps in proptest::collection::vec(0.0..100.0_f64, 1..50),
        min_dt in 0.001..0.1_f32,
        max_dt in 1.0..10.0_f32,
        reject in proptest::bool::ANY,
    ) {
        use crate::network::{HdcLtcUnifiedNetwork, StepTimingConfig};
        use crate::config::NetworkConfig;
        use crate::continuous_hv::ContinuousHV;

        let neuron_config = crate::config::NeuronConfig {
            dim: 128,
            ..crate::config::NeuronConfig::default()
        };
        let net_config = NetworkConfig {
            layer_sizes: vec![2, 2],
            neuron_config,
            use_layer_binding: true,
            skip_connections: false,
        };
        let mut net = HdcLtcUnifiedNetwork::new(net_config, 42);
        net.set_timing_config(StepTimingConfig {
            min_dt,
            max_dt,
            reject_backward_time: reject,
        });

        let input = ContinuousHV::new_random(128, 99);
        for &t in &timestamps {
            net.step_with_timestamp(t, &input);
            // Assert invariants
            assert!(net.output().norm().is_finite());
            assert!(net.output().norm() <= 5.1); // soft boundary
        }
    }
}
