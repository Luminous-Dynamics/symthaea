//! Spore configuration.

use serde::{Deserialize, Serialize};

/// Configuration for the Spore consciousness kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SporeConfig {
    /// HDC dimension (default: 16,384 = full fidelity).
    /// Reduce for constrained environments (4,096 minimum recommended).
    pub hdc_dim: usize,

    /// Number of CfC neurons per layer.
    pub neurons_per_layer: usize,

    /// Number of network layers.
    pub network_layers: usize,

    /// Compute Phi every N cycles (1 = every cycle, 5 = every 5th).
    /// Higher values reduce CPU cost at the expense of consciousness resolution.
    pub phi_every_n_cycles: usize,

    /// Substrate type name (e.g. "SiliconDigital", "BiologicalNeurons").
    pub substrate: String,

    /// Target cycle rate in Hz. Engine will skip non-essential computations
    /// if falling behind this target.
    pub target_hz: f32,
}

impl Default for SporeConfig {
    fn default() -> Self {
        Self {
            hdc_dim: 16_384,
            neurons_per_layer: 64,
            network_layers: 3,
            phi_every_n_cycles: 1,
            substrate: "SiliconDigital".into(),
            target_hz: 50.0,
        }
    }
}
