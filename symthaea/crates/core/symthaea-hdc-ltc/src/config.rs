// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Configuration types for HDC-LTC neurons and networks.

use crate::continuous_hv::{ContinuousHV, HDC_DIMENSION};
use serde::{Deserialize, Serialize};

/// Activation function for the neuron equilibrium computation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Activation {
    /// Hyperbolic tangent: tanh(x)
    Tanh,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// SiLU (Swish): x * sigmoid(x)
    SiLU,
    /// Identity (linear)
    Identity,
    /// Bounded tanh with scaling: tanh(x * bound)
    BoundedTanh { bound: f32 },
}

impl Activation {
    /// Apply activation element-wise to a hypervector.
    #[inline]
    pub fn apply(&self, hv: &ContinuousHV) -> ContinuousHV {
        let values: Vec<f32> = match self {
            Activation::Tanh => hv.values.iter().map(|x| fast_tanh(*x)).collect(),
            Activation::Sigmoid => hv.values.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect(),
            Activation::SiLU => hv
                .values
                .iter()
                .map(|x| x * (1.0 / (1.0 + (-x).exp())))
                .collect(),
            Activation::Identity => hv.values.clone(),
            Activation::BoundedTanh { bound } => {
                hv.values.iter().map(|x| fast_tanh(x * bound)).collect()
            }
        };
        ContinuousHV::from_values(values)
    }

    /// Compute the derivative of the activation at a given scalar value.
    #[inline]
    pub fn derivative(&self, x: f32) -> f32 {
        match self {
            Activation::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            Activation::SiLU => {
                let s = 1.0 / (1.0 + (-x).exp());
                s + x * s * (1.0 - s)
            }
            Activation::Identity => 1.0,
            Activation::BoundedTanh { bound } => {
                let t = (x * bound).tanh();
                bound * (1.0 - t * t)
            }
        }
    }
}

/// Fast rational approximation of tanh(x).
///
/// Uses `tanh(x) ~ x * (27 + x^2) / (27 + 9*x^2)` for |x| < 4.97.
/// Max error ~0.004 (0.4%) within the approximation range.
/// For |x| >= 4.97, returns +/-1.0.
#[inline(always)]
pub(crate) fn fast_tanh(x: f32) -> f32 {
    if x.abs() > 4.97 {
        x.signum()
    } else {
        let x2 = x * x;
        let result = x * (27.0 + x2) / (27.0 + 9.0 * x2);
        // Clamp to [-1, 1] since the rational approximation can slightly overshoot
        result.clamp(-1.0, 1.0)
    }
}

/// Configuration for an individual HDC-LTC neuron.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronConfig {
    /// Hypervector dimension (default: 16,384).
    pub dim: usize,
    /// Base time constant tau_0 in seconds (default: 0.1).
    pub tau_base: f32,
    /// State-dependent time constant scaling: tau = tau_0 * (1 + backbone * ||x||).
    pub backbone_tau: f32,
    /// Activation function (default: Tanh).
    pub activation: Activation,
    /// Learning rate for online adaptation (default: 0.01).
    pub learning_rate: f32,
    /// Momentum for gradient updates (default: 0.9).
    pub momentum: f32,
    /// L2 regularization strength (default: 0.0001).
    pub weight_decay: f32,
    /// Gating sigmoid steepness for closed-form solution (default: 1.0).
    pub gating_steepness: f32,
    /// Minimum liquid time-constant (seconds). Default: 0.001.
    pub tau_min: f32,
    /// Maximum liquid time-constant (seconds). Default: 100.0.
    pub tau_max: f32,
    /// Strength of input coupling on the tau field. Default: 0.3.
    pub tau_coupling: f32,
}

impl Default for NeuronConfig {
    fn default() -> Self {
        Self {
            dim: HDC_DIMENSION,
            tau_base: 0.1,
            backbone_tau: 0.5,
            activation: Activation::Tanh,
            learning_rate: 0.01,
            momentum: 0.9,
            weight_decay: 0.0001,
            gating_steepness: 1.0,
            tau_min: 0.001,
            tau_max: 100.0,
            tau_coupling: 0.3,
        }
    }
}

/// Configuration for a multi-layer HDC-LTC network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Number of neurons per layer (e.g., [4, 8, 4]).
    pub layer_sizes: Vec<usize>,
    /// Configuration shared by all neurons.
    pub neuron_config: NeuronConfig,
    /// Use layer-wise binding between layers (default: true).
    pub use_layer_binding: bool,
    /// Use skip connections from input to deeper layers (default: false).
    pub skip_connections: bool,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            layer_sizes: vec![4, 8, 4],
            neuron_config: NeuronConfig::default(),
            use_layer_binding: true,
            skip_connections: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_config_defaults() {
        let cfg = NeuronConfig::default();
        assert_eq!(cfg.dim, HDC_DIMENSION);
        assert!((cfg.tau_base - 0.1).abs() < 1e-6);
        assert_eq!(cfg.activation, Activation::Tanh);
    }

    #[test]
    fn test_network_config_defaults() {
        let cfg = NetworkConfig::default();
        assert_eq!(cfg.layer_sizes, vec![4, 8, 4]);
        assert!(cfg.use_layer_binding);
        assert!(!cfg.skip_connections);
    }

    #[test]
    fn test_fast_tanh_range() {
        for i in -100..=100 {
            let x = i as f32 * 0.1;
            let ft = fast_tanh(x);
            assert!(ft >= -1.0 && ft <= 1.0, "fast_tanh({}) = {}", x, ft);
        }
    }

    #[test]
    fn test_fast_tanh_accuracy() {
        // At zero, the approximation is exact
        assert!((fast_tanh(0.0)).abs() < 1e-6);

        // For the full range, result is bounded in [-1, 1] and monotonic
        let mut prev = -2.0f32;
        for i in -100..=100 {
            let x = i as f32 * 0.1;
            let ft = fast_tanh(x);
            assert!(ft >= -1.0 && ft <= 1.0, "Out of range at x={}: {}", x, ft);
            assert!(ft >= prev, "Not monotonic at x={}: {} < {}", x, ft, prev);
            prev = ft;
        }
    }

    #[test]
    fn test_activation_tanh_apply() {
        let hv = ContinuousHV::from_values(vec![0.0, 1.0, -1.0, 3.0]);
        let result = Activation::Tanh.apply(&hv);
        assert!((result.values[0]).abs() < 1e-5);
        assert!(result.values[1] > 0.7);
        assert!(result.values[2] < -0.7);
    }

    #[test]
    fn test_activation_identity() {
        let hv = ContinuousHV::from_values(vec![1.5, -2.5, 0.0]);
        let result = Activation::Identity.apply(&hv);
        assert_eq!(result.values, hv.values);
    }

    #[test]
    fn test_serde_roundtrip_config() {
        let cfg = NeuronConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let restored: NeuronConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.dim, cfg.dim);
        assert!((restored.tau_base - cfg.tau_base).abs() < 1e-6);
    }

    #[test]
    fn test_serde_roundtrip_network_config() {
        let cfg = NetworkConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let restored: NetworkConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.layer_sizes, cfg.layer_sizes);
    }
}
