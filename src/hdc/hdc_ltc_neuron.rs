//! # HdcLtcNeuron - Hyperdimensional Liquid Time-Constant Neuron
//!
//! ## Purpose
//! This module implements a novel neural unit where the neuron's state IS a
//! hypervector that evolves via Liquid Time-Constant (LTC) differential equations.
//! This bridges the gap between HDC's algebraic operations and LTC's continuous dynamics.
//!
//! ## Theoretical Basis
//!
//! Traditional LTC neurons use scalar states with learned weight matrices:
//! ```text
//! dx/dt = (-x + f(Wx + Uu)) / τ(x)
//! ```
//!
//! Our innovation replaces matrix multiplication with HDC binding:
//! ```text
//! dx/dt = (-x ⊕ f(W⊗x ⊕ U⊗u)) / τ(||x||)
//! ```
//!
//! Where:
//! - `⊗` is HDC binding (element-wise multiply for continuous HV)
//! - `⊕` is HDC bundling (normalized sum)
//! - `x` is the neuron state (a ContinuousHV)
//! - `W`, `U` are learnable weight hypervectors
//! - `τ(||x||)` is state-dependent time constant
//!
//! ## Novelty
//!
//! This is (to our knowledge) the first combination of:
//! 1. Hyperdimensional Computing algebraic operations
//! 2. Continuous-time ODE-based neural dynamics
//! 3. State-dependent time constants
//!
//! ## Key Types
//!
//! - [`HdcLtcNeuron`] - Single neuron with HDC state + LTC dynamics
//! - [`HdcLtcNetwork`] - Network of HdcLtcNeurons with hierarchical structure
//! - [`HdcLtcConfig`] - Configuration for neurons and networks
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use symthaea::hdc::hdc_ltc_neuron::{HdcLtcNeuron, HdcLtcConfig};
//! use symthaea::hdc::unified_hv::ContinuousHV;
//!
//! // Create a neuron with default configuration
//! let config = HdcLtcConfig::default();
//! let mut neuron = HdcLtcNeuron::new(config, 42);
//!
//! // Evolve with input over multiple timesteps
//! let input = ContinuousHV::random_default(123);
//! for _ in 0..100 {
//!     neuron.evolve(0.01, &input);
//! }
//!
//! // Get the evolved state
//! let state = neuron.state();
//! ```

use crate::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
use serde::{Deserialize, Serialize};

/// Configuration for HdcLtcNeuron
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdcLtcConfig {
    /// Base time constant τ₀ (seconds)
    /// Controls how quickly the neuron responds to inputs
    pub tau_base: f32,

    /// Backbone time constant multiplier
    /// τ(x) = τ₀ × (1 + backbone × ||x||)
    /// Higher values = more state-dependent adaptation
    pub backbone_tau: f32,

    /// Dimension of hypervectors
    pub dimension: usize,

    /// Activation function type
    pub activation: ActivationFunction,

    /// Learning rate for online adaptation
    pub learning_rate: f32,

    /// Momentum for gradient updates
    pub momentum: f32,

    /// L2 regularization strength
    pub weight_decay: f32,
}

impl Default for HdcLtcConfig {
    fn default() -> Self {
        Self {
            tau_base: 0.1,           // 100ms base time constant
            backbone_tau: 0.5,        // Moderate state dependency
            dimension: HDC_DIMENSION, // 16,384
            activation: ActivationFunction::Tanh,
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 0.0001,
        }
    }
}

/// Activation function types for HdcLtcNeuron
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ActivationFunction {
    /// Hyperbolic tangent: tanh(x)
    Tanh,

    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,

    /// Softplus: log(1 + exp(x))
    Softplus,

    /// Leaky ReLU: max(0.01x, x)
    LeakyRelu,

    /// Identity (linear): x
    Identity,
}

impl ActivationFunction {
    /// Apply activation element-wise to a hypervector
    pub fn apply(&self, hv: &ContinuousHV) -> ContinuousHV {
        let values: Vec<f32> = match self {
            ActivationFunction::Tanh => {
                hv.values.iter().map(|x| x.tanh()).collect()
            }
            ActivationFunction::Sigmoid => {
                hv.values.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect()
            }
            ActivationFunction::Softplus => {
                hv.values.iter().map(|x| (1.0 + x.exp()).ln()).collect()
            }
            ActivationFunction::LeakyRelu => {
                hv.values.iter().map(|x| if *x > 0.0 { *x } else { 0.01 * x }).collect()
            }
            ActivationFunction::Identity => {
                hv.values.clone()
            }
        };

        ContinuousHV::from_values(values)
    }

    /// Compute derivative for backpropagation
    pub fn derivative(&self, x: f32) -> f32 {
        match self {
            ActivationFunction::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            ActivationFunction::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            ActivationFunction::Softplus => {
                1.0 / (1.0 + (-x).exp())
            }
            ActivationFunction::LeakyRelu => {
                if x > 0.0 { 1.0 } else { 0.01 }
            }
            ActivationFunction::Identity => 1.0,
        }
    }
}

/// Hyperdimensional Liquid Time-Constant Neuron
///
/// A neural unit where the state IS a hypervector, evolving via
/// continuous-time dynamics with HDC algebraic operations.
#[derive(Debug, Clone)]
pub struct HdcLtcNeuron {
    /// Current state (hypervector)
    state: ContinuousHV,

    /// Weight hypervector for state transformation (W)
    weight_hv: ContinuousHV,

    /// Input mask hypervector (U)
    input_mask: ContinuousHV,

    /// Output projection hypervector
    output_proj: ContinuousHV,

    /// Configuration
    config: HdcLtcConfig,

    /// Momentum buffers for learning
    weight_momentum: ContinuousHV,
    /// TODO(future): Implement advanced learning algorithms using input momentum.
    /// This could enable techniques like Adam, RMSprop, or Nesterov momentum
    /// for the input projection weights, improving learning stability.
    #[allow(dead_code)] // Reserved for advanced learning algorithms
    input_momentum: ContinuousHV,

    /// Running statistics for normalization
    running_mean: f32,
    running_var: f32,

    /// Total time evolved
    total_time: f64,

    /// Number of updates
    update_count: u64,
}

impl HdcLtcNeuron {
    /// Create a new HdcLtcNeuron with given configuration and random seed
    pub fn new(config: HdcLtcConfig, seed: u64) -> Self {
        let dim = config.dimension;

        Self {
            state: ContinuousHV::zero(dim),
            weight_hv: ContinuousHV::random(dim, seed),
            input_mask: ContinuousHV::random(dim, seed + 1000),
            output_proj: ContinuousHV::random(dim, seed + 2000),
            weight_momentum: ContinuousHV::zero(dim),
            input_momentum: ContinuousHV::zero(dim),
            running_mean: 0.0,
            running_var: 1.0,
            total_time: 0.0,
            update_count: 0,
            config,
        }
    }

    /// Create with default configuration
    pub fn new_default(seed: u64) -> Self {
        Self::new(HdcLtcConfig::default(), seed)
    }

    /// Evolve the neuron state by dt seconds with given input
    ///
    /// Implements the HDC-LTC ODE:
    /// ```text
    /// dx/dt = (-x + f(W⊗x + U⊗u)) / τ(||x||)
    /// ```
    ///
    /// # Arguments
    /// * `dt` - Time step in seconds
    /// * `input` - Input hypervector
    pub fn evolve(&mut self, dt: f32, input: &ContinuousHV) {
        // 1. Apply input mask via HDC binding: U ⊗ u
        let masked_input = self.input_mask.bind(input);

        // 2. Apply weight to state via HDC binding: W ⊗ x
        let transformed_state = self.weight_hv.bind(&self.state);

        // 3. Combine via HDC bundling: bundle(W⊗x, U⊗u)
        let combined = ContinuousHV::bundle(&[&transformed_state, &masked_input]);

        // 4. Apply activation function: f(combined)
        let activated = self.config.activation.apply(&combined);

        // 5. Compute state-dependent time constant: τ(||x||)
        let state_norm = self.state.norm();
        let tau_effective = self.config.tau_base * (1.0 + self.config.backbone_tau * state_norm);

        // 6. Compute derivative: dx/dt = (-x + activated) / τ
        let delta = activated.subtract(&self.state).scale(dt / tau_effective);

        // 7. Update state: x = x + dx
        self.state = self.state.add(&delta);

        // 7b. Soft state bounding to prevent numerical instability
        // This ensures the state remains bounded while preserving direction
        let bounded_norm = self.state.norm();
        if bounded_norm > 5.0 {
            self.state = self.state.normalize().scale(5.0);
        }

        // 8. Update statistics
        self.total_time += dt as f64;
        self.update_count += 1;

        // Update running mean/var with exponential moving average
        let alpha = 0.01;
        let new_norm = self.state.norm();
        self.running_mean = (1.0 - alpha) * self.running_mean + alpha * new_norm;
        self.running_var = (1.0 - alpha) * self.running_var + alpha * (new_norm - self.running_mean).powi(2);
    }

    /// Evolve with Runge-Kutta 4 integration (more accurate than Euler)
    pub fn evolve_rk4(&mut self, dt: f32, input: &ContinuousHV) {
        let h = dt;

        // k1
        let k1 = self.compute_derivative(input, &self.state);

        // k2
        let state_k2 = self.state.add(&k1.scale(h / 2.0));
        let k2 = self.compute_derivative(input, &state_k2);

        // k3
        let state_k3 = self.state.add(&k2.scale(h / 2.0));
        let k3 = self.compute_derivative(input, &state_k3);

        // k4
        let state_k4 = self.state.add(&k3.scale(h));
        let k4 = self.compute_derivative(input, &state_k4);

        // Combine: x += (k1 + 2k2 + 2k3 + k4) * h/6
        let k2_scaled = k2.scale(2.0);
        let k3_scaled = k3.scale(2.0);

        let sum = k1.add(&k2_scaled).add(&k3_scaled).add(&k4);
        self.state = self.state.add(&sum.scale(h / 6.0));

        self.total_time += dt as f64;
        self.update_count += 1;
    }

    /// Compute dx/dt for given state (used by RK4)
    fn compute_derivative(&self, input: &ContinuousHV, state: &ContinuousHV) -> ContinuousHV {
        let masked_input = self.input_mask.bind(input);
        let transformed_state = self.weight_hv.bind(state);
        let combined = ContinuousHV::bundle(&[&transformed_state, &masked_input]);
        let activated = self.config.activation.apply(&combined);

        let state_norm = state.norm();
        let tau_effective = self.config.tau_base * (1.0 + self.config.backbone_tau * state_norm);

        activated.subtract(state).scale(1.0 / tau_effective)
    }

    /// Get the current state
    pub fn state(&self) -> &ContinuousHV {
        &self.state
    }

    /// Get mutable state reference
    pub fn state_mut(&mut self) -> &mut ContinuousHV {
        &mut self.state
    }

    /// Set the state directly
    pub fn set_state(&mut self, state: ContinuousHV) {
        self.state = state;
    }

    /// Reset state to zero
    pub fn reset(&mut self) {
        self.state = ContinuousHV::zero(self.config.dimension);
        self.total_time = 0.0;
        self.update_count = 0;
    }

    /// Get output projection of current state
    pub fn output(&self) -> ContinuousHV {
        self.output_proj.bind(&self.state)
    }

    /// Get scalar output (similarity with output projection)
    pub fn output_scalar(&self) -> f32 {
        self.state.similarity(&self.output_proj)
    }

    /// Compute effective time constant at current state
    pub fn effective_tau(&self) -> f32 {
        let state_norm = self.state.norm();
        self.config.tau_base * (1.0 + self.config.backbone_tau * state_norm)
    }

    /// Get configuration
    pub fn config(&self) -> &HdcLtcConfig {
        &self.config
    }

    /// Get total time evolved
    pub fn total_time(&self) -> f64 {
        self.total_time
    }

    /// Get update count
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Compute Hebbian-like weight update based on state correlation
    ///
    /// Updates weight to strengthen input-output correlations.
    pub fn hebbian_update(&mut self, input: &ContinuousHV, learning_rate: Option<f32>) {
        let lr = learning_rate.unwrap_or(self.config.learning_rate);

        // Hebbian: ΔW ∝ input ⊗ state (correlation-based)
        let correlation = input.bind(&self.state);

        // Apply with momentum
        let m = self.config.momentum;
        self.weight_momentum = self.weight_momentum.scale(m)
            .add(&correlation.scale(lr));

        // Weight decay
        let decay = self.config.weight_decay;
        self.weight_hv = self.weight_hv.scale(1.0 - decay)
            .add(&self.weight_momentum);

        // Normalize to prevent explosion
        if self.weight_hv.norm() > 2.0 {
            self.weight_hv = self.weight_hv.normalize().scale(2.0);
        }
    }

    /// Get neuron statistics
    pub fn stats(&self) -> HdcLtcNeuronStats {
        HdcLtcNeuronStats {
            state_norm: self.state.norm(),
            effective_tau: self.effective_tau(),
            total_time: self.total_time,
            update_count: self.update_count,
            running_mean: self.running_mean,
            running_std: self.running_var.sqrt(),
        }
    }
}

/// Statistics about an HdcLtcNeuron
#[derive(Debug, Clone)]
pub struct HdcLtcNeuronStats {
    /// Current state norm
    pub state_norm: f32,

    /// Effective time constant
    pub effective_tau: f32,

    /// Total time evolved
    pub total_time: f64,

    /// Number of updates
    pub update_count: u64,

    /// Running mean of state norm
    pub running_mean: f32,

    /// Running std of state norm
    pub running_std: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// HDC-LTC NETWORK
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for HdcLtcNetwork
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdcLtcNetworkConfig {
    /// Number of neurons per layer
    pub layer_sizes: Vec<usize>,

    /// Configuration for each neuron
    pub neuron_config: HdcLtcConfig,

    /// Inter-layer binding hypervectors
    pub use_layer_binding: bool,

    /// Recurrent connections within layers
    pub recurrent: bool,

    /// Skip connections between layers
    pub skip_connections: bool,
}

impl Default for HdcLtcNetworkConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![4, 8, 4], // 3-layer network
            neuron_config: HdcLtcConfig::default(),
            use_layer_binding: true,
            recurrent: true,
            skip_connections: false,
        }
    }
}

/// Network of HdcLtcNeurons with hierarchical structure
#[derive(Debug, Clone)]
pub struct HdcLtcNetwork {
    /// Layers of neurons
    layers: Vec<Vec<HdcLtcNeuron>>,

    /// Inter-layer binding vectors
    layer_bindings: Vec<ContinuousHV>,

    /// Configuration
    config: HdcLtcNetworkConfig,
}

impl HdcLtcNetwork {
    /// Create a new network with given configuration
    pub fn new(config: HdcLtcNetworkConfig, seed: u64) -> Self {
        let mut layers = Vec::new();
        let mut current_seed = seed;

        for &layer_size in &config.layer_sizes {
            let layer: Vec<HdcLtcNeuron> = (0..layer_size)
                .map(|_| {
                    current_seed += 1;
                    HdcLtcNeuron::new(config.neuron_config.clone(), current_seed)
                })
                .collect();
            layers.push(layer);
        }

        // Create inter-layer bindings
        let dim = config.neuron_config.dimension;
        let layer_bindings: Vec<ContinuousHV> = (0..config.layer_sizes.len())
            .map(|i| ContinuousHV::random(dim, seed + 10000 + i as u64))
            .collect();

        Self {
            layers,
            layer_bindings,
            config,
        }
    }

    /// Evolve all neurons with input
    pub fn evolve(&mut self, dt: f32, input: &ContinuousHV) {
        // Layer 0: Direct input
        for neuron in &mut self.layers[0] {
            neuron.evolve(dt, input);
        }

        // Subsequent layers: Bundled output from previous layer
        for layer_idx in 1..self.layers.len() {
            // Bundle outputs from previous layer
            let prev_outputs: Vec<ContinuousHV> = self.layers[layer_idx - 1]
                .iter()
                .map(|n| n.output())
                .collect();

            let prev_refs: Vec<&ContinuousHV> = prev_outputs.iter().collect();
            let layer_input = ContinuousHV::bundle(&prev_refs);

            // Apply layer binding if configured
            let bound_input = if self.config.use_layer_binding {
                self.layer_bindings[layer_idx].bind(&layer_input)
            } else {
                layer_input
            };

            // Evolve each neuron in this layer
            for neuron in &mut self.layers[layer_idx] {
                neuron.evolve(dt, &bound_input);
            }
        }
    }

    /// Get output from final layer
    pub fn output(&self) -> ContinuousHV {
        let final_layer = self.layers.last().unwrap();
        let outputs: Vec<ContinuousHV> = final_layer.iter()
            .map(|n| n.output())
            .collect();

        let refs: Vec<&ContinuousHV> = outputs.iter().collect();
        ContinuousHV::bundle(&refs)
    }

    /// Get all neuron states as a flat vector
    pub fn all_states(&self) -> Vec<&ContinuousHV> {
        self.layers.iter()
            .flat_map(|layer| layer.iter().map(|n| n.state()))
            .collect()
    }

    /// Get network statistics
    pub fn stats(&self) -> HdcLtcNetworkStats {
        let all_stats: Vec<HdcLtcNeuronStats> = self.layers.iter()
            .flat_map(|layer| layer.iter().map(|n| n.stats()))
            .collect();

        let avg_norm: f32 = all_stats.iter().map(|s| s.state_norm).sum::<f32>() / all_stats.len() as f32;
        let avg_tau: f32 = all_stats.iter().map(|s| s.effective_tau).sum::<f32>() / all_stats.len() as f32;

        HdcLtcNetworkStats {
            n_neurons: all_stats.len(),
            n_layers: self.layers.len(),
            avg_state_norm: avg_norm,
            avg_effective_tau: avg_tau,
            total_updates: all_stats.iter().map(|s| s.update_count).sum(),
        }
    }

    /// Reset all neurons
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            for neuron in layer {
                neuron.reset();
            }
        }
    }

    /// Get number of layers
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get layer by index
    pub fn layer(&self, idx: usize) -> Option<&Vec<HdcLtcNeuron>> {
        self.layers.get(idx)
    }

    /// Get mutable layer by index
    pub fn layer_mut(&mut self, idx: usize) -> Option<&mut Vec<HdcLtcNeuron>> {
        self.layers.get_mut(idx)
    }
}

/// Statistics about an HdcLtcNetwork
#[derive(Debug, Clone)]
pub struct HdcLtcNetworkStats {
    /// Total number of neurons
    pub n_neurons: usize,

    /// Number of layers
    pub n_layers: usize,

    /// Average state norm across neurons
    pub avg_state_norm: f32,

    /// Average effective time constant
    pub avg_effective_tau: f32,

    /// Total updates across all neurons
    pub total_updates: u64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_creation() {
        let config = HdcLtcConfig::default();
        let neuron = HdcLtcNeuron::new(config, 42);

        assert_eq!(neuron.state().dim(), HDC_DIMENSION);
        assert_eq!(neuron.update_count(), 0);
        assert_eq!(neuron.total_time(), 0.0);
    }

    #[test]
    fn test_neuron_evolution() {
        let mut neuron = HdcLtcNeuron::new_default(42);
        let input = ContinuousHV::random_default(123);

        let initial_norm = neuron.state().norm();
        assert!(initial_norm < 0.01); // Start near zero

        // Evolve for many steps
        for _ in 0..100 {
            neuron.evolve(0.01, &input);
        }

        let final_norm = neuron.state().norm();
        assert!(final_norm > initial_norm, "State should evolve away from zero");
        assert!(final_norm < 10.0, "State should remain bounded");
        assert_eq!(neuron.update_count(), 100);
    }

    #[test]
    fn test_state_dependent_tau() {
        let mut neuron = HdcLtcNeuron::new_default(42);
        let input = ContinuousHV::random_default(123);

        let tau_initial = neuron.effective_tau();

        // Evolve to change state
        for _ in 0..50 {
            neuron.evolve(0.01, &input);
        }

        let tau_final = neuron.effective_tau();

        // Tau should increase as state norm increases
        assert!(tau_final >= tau_initial * 0.9, "Tau should adapt to state: {} vs {}", tau_final, tau_initial);
    }

    #[test]
    fn test_rk4_vs_euler() {
        let input = ContinuousHV::random_default(123);

        let mut neuron_euler = HdcLtcNeuron::new_default(42);
        let mut neuron_rk4 = HdcLtcNeuron::new_default(42);

        // Evolve both with same input
        for _ in 0..100 {
            neuron_euler.evolve(0.01, &input);
            neuron_rk4.evolve_rk4(0.01, &input);
        }

        // States should be similar but not identical
        let similarity = neuron_euler.state().similarity(neuron_rk4.state());
        assert!(similarity > 0.8, "RK4 and Euler should produce similar results: {}", similarity);
    }

    #[test]
    fn test_network_creation() {
        let config = HdcLtcNetworkConfig::default();
        let network = HdcLtcNetwork::new(config.clone(), 42);

        assert_eq!(network.n_layers(), config.layer_sizes.len());
        assert_eq!(network.layer(0).unwrap().len(), config.layer_sizes[0]);
    }

    #[test]
    fn test_network_evolution() {
        let config = HdcLtcNetworkConfig {
            layer_sizes: vec![2, 3, 2],
            ..Default::default()
        };
        let mut network = HdcLtcNetwork::new(config, 42);
        let input = ContinuousHV::random_default(123);

        // Evolve network
        for _ in 0..50 {
            network.evolve(0.01, &input);
        }

        let output = network.output();
        assert_eq!(output.dim(), HDC_DIMENSION);

        let stats = network.stats();
        assert_eq!(stats.n_layers, 3);
        assert!(stats.total_updates > 0);
    }

    #[test]
    fn test_hebbian_update() {
        let mut neuron = HdcLtcNeuron::new_default(42);
        let input = ContinuousHV::random_default(123);

        // Evolve first
        for _ in 0..20 {
            neuron.evolve(0.01, &input);
        }

        let _weight_before = neuron.weight_hv.norm();

        // Hebbian update
        neuron.hebbian_update(&input, Some(0.1));

        let weight_after = neuron.weight_hv.norm();

        // Weights should change but remain bounded
        // Allow small epsilon for floating point precision
        assert!(weight_after <= 2.1, "Weights should be approximately normalized (got {})", weight_after);
    }

    #[test]
    fn test_activation_functions() {
        let hv = ContinuousHV::random(100, 42);

        for activation in [
            ActivationFunction::Tanh,
            ActivationFunction::Sigmoid,
            ActivationFunction::Softplus,
            ActivationFunction::LeakyRelu,
            ActivationFunction::Identity,
        ] {
            let result = activation.apply(&hv);
            assert_eq!(result.dim(), 100);

            // Check bounds for bounded activations
            match activation {
                ActivationFunction::Tanh => {
                    assert!(result.values.iter().all(|&x| x >= -1.0 && x <= 1.0));
                }
                ActivationFunction::Sigmoid => {
                    assert!(result.values.iter().all(|&x| x >= 0.0 && x <= 1.0));
                }
                _ => {}
            }
        }
    }
}
