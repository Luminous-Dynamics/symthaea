// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # HDC-LTC Neuron
//!
//! Combines hyperdimensional computing with Liquid Time-Constant dynamics
//! for neurons that process information in continuous time.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::ContinuousHV;

/// Configuration for HDC-LTC neuron
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdcLtcNeuronConfig {
    /// HDC dimension
    pub dimension: usize,
    /// Base time constant
    pub tau_base: f32,
    /// Minimum time constant
    pub tau_min: f32,
    /// Maximum time constant
    pub tau_max: f32,
    /// Learning rate
    pub learning_rate: f32,
}

impl Default for HdcLtcNeuronConfig {
    fn default() -> Self {
        Self {
            dimension: 512,
            tau_base: 1.0,
            tau_min: 0.1,
            tau_max: 10.0,
            learning_rate: 0.01,
        }
    }
}

/// State of an HDC-LTC neuron
#[derive(Debug, Clone)]
pub struct HdcLtcNeuronState {
    /// Hidden state vector
    pub hidden: ContinuousHV,
    /// Time constant
    pub tau: f32,
    /// Activation level
    pub activation: f32,
    /// Accumulated error for learning
    pub error_acc: f32,
}

/// An HDC-LTC neuron
#[derive(Debug, Clone)]
pub struct HdcLtcNeuron {
    config: HdcLtcNeuronConfig,
    /// Input weight matrix (as HDC projection)
    input_weight: ContinuousHV,
    /// Recurrent weight matrix (as HDC binding)
    recurrent_weight: ContinuousHV,
    /// Time constant modulation vector
    tau_modulator: ContinuousHV,
    /// Current state
    state: HdcLtcNeuronState,
    /// Seed counter for deterministic random init
    #[allow(dead_code)]
    seed: u64,
}

impl HdcLtcNeuron {
    /// Create new HDC-LTC neuron
    pub fn new(config: HdcLtcNeuronConfig) -> Self {
        let dim = config.dimension;
        Self {
            input_weight: ContinuousHV::random(dim, 1),
            recurrent_weight: ContinuousHV::random(dim, 2),
            tau_modulator: ContinuousHV::random(dim, 3),
            state: HdcLtcNeuronState {
                hidden: ContinuousHV::zero(dim),
                tau: config.tau_base,
                activation: 0.0,
                error_acc: 0.0,
            },
            config,
            seed: 4,
        }
    }

    /// Process input and advance state
    pub fn step(&mut self, input: &ContinuousHV, dt: f32) -> ContinuousHV {
        // Compute input contribution
        let input_contrib = input.bind(&self.input_weight);

        // Compute recurrent contribution
        let recurrent_contrib = self.state.hidden.bind(&self.recurrent_weight);

        // Compute adaptive time constant based on input
        let tau_adjustment = input.similarity(&self.tau_modulator);
        let adjusted_tau = (self.config.tau_base * (1.0 + tau_adjustment))
            .clamp(self.config.tau_min, self.config.tau_max);

        // LTC dynamics: dh/dt = (-h + input_contrib + recurrent_contrib) / tau
        let combined = ContinuousHV::bundle_owned(&[input_contrib, recurrent_contrib]);
        let neg_hidden = self.state.hidden.scale(-1.0);
        let delta = ContinuousHV::bundle_owned(&[combined, neg_hidden]);

        // Euler integration
        let scale_factor = dt / adjusted_tau;
        let delta_scaled = delta.scale(scale_factor);

        // Update hidden state
        self.state.hidden = ContinuousHV::bundle_owned(&[self.state.hidden.clone(), delta_scaled]);

        // Normalize to prevent explosion
        self.state.hidden = self.state.hidden.normalize();

        // Update tau
        self.state.tau = adjusted_tau;

        // Calculate activation
        let magnitude: f32 = self
            .state
            .hidden
            .values
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        self.state.activation = (magnitude / (self.config.dimension as f32).sqrt()).tanh();

        self.state.hidden.clone()
    }

    /// CfC closed-form prediction: h(t) = A + (h₀ - A) · exp(-t/τ)
    /// Pure function — does not mutate state.
    pub fn predict_forward(&self, input: &ContinuousHV, horizon: f32) -> ContinuousHV {
        let input_contrib = input.bind(&self.input_weight);
        let recurrent_contrib = self.state.hidden.bind(&self.recurrent_weight);
        let attractor = ContinuousHV::bundle_owned(&[input_contrib, recurrent_contrib]).normalize();

        let tau_adjustment = input.similarity(&self.tau_modulator);
        let tau = (self.config.tau_base * (1.0 + tau_adjustment))
            .clamp(self.config.tau_min, self.config.tau_max);

        let decay = (-horizon / tau).exp();

        // h(t) = A + (h₀ - A) · exp(-t/τ)
        let dim = self.config.dimension;
        let mut result = Vec::with_capacity(dim);
        for i in 0..dim {
            let a = attractor.values[i];
            let h0 = self.state.hidden.values[i];
            result.push(a + (h0 - a) * decay);
        }
        ContinuousHV::from_vec(result).normalize()
    }

    /// Get current state
    pub fn state(&self) -> &HdcLtcNeuronState {
        &self.state
    }

    /// Reset state
    pub fn reset(&mut self) {
        self.state.hidden = ContinuousHV::zero(self.config.dimension);
        self.state.tau = self.config.tau_base;
        self.state.activation = 0.0;
        self.state.error_acc = 0.0;
    }

    /// Apply learning update
    pub fn learn(&mut self, error_signal: &ContinuousHV) {
        // Update input weights using HDC correlation
        let correlation = self.state.hidden.bind(error_signal);
        let scaled_corr = correlation.scale(self.config.learning_rate);
        self.input_weight = ContinuousHV::bundle_owned(&[self.input_weight.clone(), scaled_corr]);

        // Update recurrent weights
        let recurrent_correlation = self.state.hidden.bind(&self.state.hidden);
        let recurrent_error = recurrent_correlation.bind(error_signal);
        let scaled_recurrent = recurrent_error.scale(self.config.learning_rate * 0.5);
        self.recurrent_weight =
            ContinuousHV::bundle_owned(&[self.recurrent_weight.clone(), scaled_recurrent]);

        // Accumulate error for monitoring
        self.state.error_acc = 0.9 * self.state.error_acc
            + 0.1 * error_signal.values.iter().map(|x| x.abs()).sum::<f32>()
                / error_signal.dim() as f32;
    }
}

impl Default for HdcLtcNeuron {
    fn default() -> Self {
        Self::new(HdcLtcNeuronConfig::default())
    }
}

/// A layer of HDC-LTC neurons
#[derive(Debug, Clone)]
pub struct HdcLtcLayer {
    /// Neurons in this layer
    neurons: Vec<HdcLtcNeuron>,
    /// Layer output
    output: ContinuousHV,
    /// Layer dimension
    dimension: usize,
}

impl HdcLtcLayer {
    /// Create new layer
    pub fn new(num_neurons: usize, dimension: usize) -> Self {
        let config = HdcLtcNeuronConfig {
            dimension,
            ..Default::default()
        };

        let neurons = (0..num_neurons)
            .map(|_| HdcLtcNeuron::new(config.clone()))
            .collect();

        Self {
            neurons,
            output: ContinuousHV::zero(dimension),
            dimension,
        }
    }

    /// Process input through layer
    pub fn forward(&mut self, input: &ContinuousHV, dt: f32) -> ContinuousHV {
        let outputs: Vec<ContinuousHV> =
            self.neurons.iter_mut().map(|n| n.step(input, dt)).collect();

        // Bundle all neuron outputs
        self.output = if outputs.is_empty() {
            ContinuousHV::zero(self.dimension)
        } else {
            ContinuousHV::bundle_owned(&outputs)
        };

        self.output.clone()
    }

    /// CfC closed-form prediction across all neurons in the layer
    pub fn predict_forward(&self, input: &ContinuousHV, horizon: f32) -> ContinuousHV {
        let predictions: Vec<ContinuousHV> = self
            .neurons
            .iter()
            .map(|n| n.predict_forward(input, horizon))
            .collect();
        if predictions.is_empty() {
            ContinuousHV::zero(self.dimension)
        } else {
            ContinuousHV::bundle_owned(&predictions)
        }
    }

    /// Get neurons
    pub fn neurons(&self) -> &[HdcLtcNeuron] {
        &self.neurons
    }

    /// Get mutable neurons
    pub fn neurons_mut(&mut self) -> &mut [HdcLtcNeuron] {
        &mut self.neurons
    }

    /// Reset all neurons
    pub fn reset(&mut self) {
        for neuron in &mut self.neurons {
            neuron.reset();
        }
        self.output = ContinuousHV::zero(self.dimension);
    }
}

/// A network of HDC-LTC layers
pub struct HdcLtcNetwork {
    /// Layers
    layers: Vec<HdcLtcLayer>,
    /// Final output
    output: ContinuousHV,
}

impl HdcLtcNetwork {
    /// Create new network
    pub fn new(layer_sizes: &[usize], dimension: usize) -> Self {
        let layers = layer_sizes
            .iter()
            .map(|&size| HdcLtcLayer::new(size, dimension))
            .collect();

        Self {
            layers,
            output: ContinuousHV::zero(dimension),
        }
    }

    /// Forward pass
    pub fn forward(&mut self, input: &ContinuousHV, dt: f32) -> ContinuousHV {
        let mut current = input.clone();

        for layer in &mut self.layers {
            current = layer.forward(&current, dt);
        }

        self.output = current.clone();
        current
    }

    /// Reset network
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Get layers
    pub fn layers(&self) -> &[HdcLtcLayer] {
        &self.layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdc_ltc_neuron() {
        let mut neuron = HdcLtcNeuron::default();
        let input = ContinuousHV::random(512, 42);

        let output = neuron.step(&input, 0.1);
        assert_eq!(output.dim(), 512);
    }

    #[test]
    fn test_hdc_ltc_layer() {
        let mut layer = HdcLtcLayer::new(4, 512);
        let input = ContinuousHV::random(512, 42);

        let output = layer.forward(&input, 0.1);
        assert_eq!(output.dim(), 512);
    }

    #[test]
    fn test_predict_forward_zero_horizon() {
        let mut neuron = HdcLtcNeuron::default();
        let input = ContinuousHV::random(512, 42);
        // Step once to populate hidden state
        neuron.step(&input, 0.1);

        // At horizon=0, prediction should equal current hidden state
        let prediction = neuron.predict_forward(&input, 0.0);
        let sim = prediction.similarity(&neuron.state().hidden);
        assert!(
            sim > 0.99,
            "Zero-horizon prediction should match current state, got sim={sim}"
        );
    }

    #[test]
    fn test_predict_forward_convergence() {
        let mut neuron = HdcLtcNeuron::default();
        let input = ContinuousHV::random(512, 42);
        neuron.step(&input, 0.1);

        let near = neuron.predict_forward(&input, 1.0);
        let far = neuron.predict_forward(&input, 100.0);
        let very_far = neuron.predict_forward(&input, 1000.0);

        // Far predictions should converge (be more similar to each other than to near)
        let sim_far_vfar = far.similarity(&very_far);
        let sim_near_vfar = near.similarity(&very_far);
        assert!(
            sim_far_vfar >= sim_near_vfar,
            "Far predictions should converge: far-vfar={sim_far_vfar}, near-vfar={sim_near_vfar}"
        );
    }

    #[test]
    fn test_hdc_ltc_network() {
        let mut network = HdcLtcNetwork::new(&[4, 4, 2], 512);
        let input = ContinuousHV::random(512, 42);

        let output = network.forward(&input, 0.1);
        assert_eq!(output.dim(), 512);
    }
}
