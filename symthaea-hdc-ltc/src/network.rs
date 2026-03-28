// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # HDC-LTC Network
//!
//! A multi-layer network of HDC-LTC neurons with optional layer binding
//! and skip connections.

use crate::config::NetworkConfig;
use crate::continuous_hv::ContinuousHV;
use crate::neuron::HdcLtcUnifiedNeuron;
use serde::{Deserialize, Serialize};

/// A multi-layer network of unified HDC-LTC neurons.
///
/// Each layer is a collection of neurons that all receive the same input.
/// The output of a layer is the bundled (averaged) states of its neurons.
/// Inter-layer connections optionally use HDC binding for decorrelation.
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
        }
    }

    /// Forward pass: evolve all layers with closed-form solution.
    ///
    /// Layer 0 receives the input directly. Subsequent layers receive
    /// the bundled output of the previous layer (optionally bound with
    /// a layer-binding vector and skip-connected with the original input).
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

    /// Reset all neuron states and cached outputs.
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            for neuron in layer {
                neuron.reset();
            }
        }
        for output in &mut self.layer_outputs {
            *output = ContinuousHV::new(self.config.neuron_config.dim);
        }
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
}
