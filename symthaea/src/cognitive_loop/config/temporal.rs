// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal backend configuration: CfC, HdcLtcUnified, training methods.

use serde::{Deserialize, Serialize};

// TEMPORAL BACKEND SELECTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Temporal backend selection for the cognitive loop
///
/// The cognitive loop can use either CfC (Closed-form Continuous-time) or
/// HdcLtcUnified (Unified HDC-LTC) networks for temporal prediction.
///
/// ## CfC (Default)
/// - Traditional approach using ndarray-based weights
/// - Matrix multiplication for state transitions
/// - Well-tested and stable
///
/// ## HdcLtcUnified
/// - Novel approach using hypervector states
/// - HDC binding/bundling instead of matrix multiplication
/// - O(1) temporal jumps via closed-form solution
/// - State IS memory (holographic representation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TemporalBackend {
    /// Original Closed-form Continuous-time network
    CfC,
    /// Unified HDC-LTC network with hypervector states.
    /// ~8.5x faster than CfC: element-wise HDC ops + AVX2 SIMD vs matrix multiply.
    /// O(1) temporal jumps via closed-form solution. State IS memory (holographic).
    #[default]
    HdcLtcUnified,
    /// Hierarchical CfC with multi-scale temporal processing (PP-2)
    HierarchicalCfC,
}

/// Training method selection for the cognitive loop
///
/// Controls how the temporal network is trained each cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TrainingMethod {
    /// Always use BPTT (analytical gradients)
    Bptt,
    /// Always use SPSA (perturbation-based)
    Spsa,
    /// Use BPTT by default, fall back to SPSA when BPTT diverges
    #[default]
    BpttWithSpsaFallback,
}

/// Configuration for CfC in the cognitive loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfCConfig {
    /// Number of CfC neurons
    pub num_neurons: usize,

    /// Input dimension (compressed HDC)
    pub input_dim: usize,

    /// Learning rate for CfC training
    pub learning_rate: f32,

    /// Time step for CfC predictions (seconds)
    pub delta_t: f32,

    /// Future prediction horizons for multi-scale prediction
    pub prediction_horizons: Vec<f32>,
}

impl Default for CfCConfig {
    fn default() -> Self {
        Self {
            num_neurons: 256,
            input_dim: 256, // Must match num_neurons for train_step compatibility
            learning_rate: 0.001,
            delta_t: 0.02, // 50Hz base rate
            // Multi-scale prediction: t+1, t+5, t+10 steps
            prediction_horizons: vec![0.02, 0.1, 0.2],
        }
    }
}

impl CfCConfig {
    /// Validate CfC configuration parameters.
    ///
    /// Checks that all numeric parameters are within valid ranges:
    /// - `num_neurons` must be positive
    /// - `input_dim` must be positive
    /// - `learning_rate` must be in (0.0, 1.0]
    /// - `delta_t` must be positive
    /// - `prediction_horizons` must be non-empty with all positive values
    pub fn validate(&self) -> Result<(), String> {
        if self.num_neurons == 0 {
            return Err("CfCConfig: num_neurons must be > 0".into());
        }
        if self.input_dim == 0 {
            return Err("CfCConfig: input_dim must be > 0".into());
        }
        if self.learning_rate <= 0.0 || self.learning_rate > 1.0 {
            return Err(format!(
                "CfCConfig: learning_rate must be in (0.0, 1.0], got {}",
                self.learning_rate
            ));
        }
        if !self.learning_rate.is_finite() {
            return Err("CfCConfig: learning_rate must be finite".into());
        }
        if self.delta_t <= 0.0 || !self.delta_t.is_finite() {
            return Err(format!(
                "CfCConfig: delta_t must be positive and finite, got {}",
                self.delta_t
            ));
        }
        if self.prediction_horizons.is_empty() {
            return Err("CfCConfig: prediction_horizons must be non-empty".into());
        }
        for (i, &h) in self.prediction_horizons.iter().enumerate() {
            if h <= 0.0 || !h.is_finite() {
                return Err(format!(
                    "CfCConfig: prediction_horizons[{i}] must be positive and finite, got {h}"
                ));
            }
        }
        Ok(())
    }
}
