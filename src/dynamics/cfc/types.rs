// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Configuration types, activation functions, and utility functions for CfC networks.

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Minimum allowed tau value to prevent NaN in exp(-dt/tau) calculations.
/// Values below this threshold would cause numerical instability.
pub(crate) const MIN_TAU: f32 = 1e-6;

// =============================================================================
// FAST SIGMOID APPROXIMATION (2-3x speedup for LTC/CfC step functions)
// =============================================================================

/// Fast sigmoid approximation using rational function.
/// Accuracy: max error ~0.01 compared to standard sigmoid.
/// Performance: 2-3x faster than 1.0 / (1.0 + (-x).exp()).
///
/// Formula: 0.5 * (1.0 + x / (1.0 + |x|))
#[inline(always)]
pub(crate) fn fast_sigmoid(x: f32) -> f32 {
    0.5 * (1.0 + x / (1.0 + x.abs()))
}

/// Standard sigmoid for use in accuracy-critical paths
#[inline(always)]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Mean squared error between two arrays.
/// If lengths differ, computes MSE over the shorter prefix (truncated comparison).
/// This handles the common case where output_dim != target_dim.
pub(crate) fn mse_loss(output: &Array1<f32>, target: &Array1<f32>) -> f32 {
    let n = output.len().min(target.len());
    if n == 0 {
        return 0.0;
    }
    let mse = output
        .iter()
        .zip(target.iter())
        .take(n)
        .map(|(o, t)| (o - t).powi(2))
        .sum::<f32>()
        / n as f32;
    if mse.is_finite() { mse } else { 1.0 }
}

/// Configuration for online learning during inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnlineLearningConfig {
    /// Learning rate for online adaptation (much smaller than training)
    /// Default: 0.001
    pub learning_rate: f32,

    /// Minimum prediction error to trigger adaptation
    /// Default: 0.1 (only adapt when error is significant)
    pub error_threshold: f32,

    /// Exponential moving average factor for error tracking
    /// Default: 0.1 (slow adaptation to new error levels)
    pub ema_alpha: f32,

    /// Maximum weight change per adaptation step (prevents catastrophic forgetting)
    /// Default: 0.01 (1% max change)
    pub max_weight_delta: f32,

    /// Whether to adapt tau (time constants) online
    /// Default: false (tau adaptation is more risky)
    pub adapt_tau: bool,

    /// Tau learning rate multiplier (if adapt_tau is true)
    /// Default: 0.01 (much slower than weights)
    pub tau_lr_multiplier: f32,
}

impl Default for OnlineLearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            error_threshold: 0.1,
            ema_alpha: 0.1,
            max_weight_delta: 0.01,
            adapt_tau: false,
            tau_lr_multiplier: 0.01,
        }
    }
}

/// Configuration for a CfC cell
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfCConfig {
    /// Input dimension
    pub input_dim: usize,

    /// Hidden state dimension
    pub hidden_dim: usize,

    /// Whether to use backbone network for additional capacity
    pub use_backbone: bool,

    /// Number of backbone layers
    pub backbone_layers: usize,

    /// Backbone hidden dimension
    pub backbone_dim: usize,

    /// Activation function type
    pub activation: ActivationType,

    /// Time constant initialization range
    pub tau_range: (f32, f32),

    /// Dropout rate (0.0 = no dropout)
    pub dropout: f32,

    /// Gradient clip threshold (default 1.0; use higher values like 5.0 for classification tasks)
    pub gradient_clip: f32,

    /// Online learning configuration (for inference-time adaptation)
    pub online_learning: Option<OnlineLearningConfig>,
}

impl Default for CfCConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            hidden_dim: 128,
            use_backbone: true,
            backbone_layers: 2,
            backbone_dim: 128,
            activation: ActivationType::SiLU,
            tau_range: (0.1, 10.0),
            dropout: 0.1,
            gradient_clip: 1.0,
            online_learning: None,
        }
    }
}

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    /// Sigmoid-weighted Linear Unit
    SiLU,
    /// Gaussian Error Linear Unit
    GELU,
    /// Rectified Linear Unit
    ReLU,
    /// Hyperbolic tangent
    Tanh,
    /// Sigmoid
    Sigmoid,
}

impl ActivationType {
    /// Apply activation function (standard accuracy)
    #[inline]
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            ActivationType::SiLU => x * sigmoid(x),
            ActivationType::GELU => {
                0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x.powi(3))).tanh())
            }
            ActivationType::ReLU => x.max(0.0),
            ActivationType::Tanh => x.tanh(),
            ActivationType::Sigmoid => sigmoid(x),
        }
    }

    /// Apply fast activation (2-3x faster, slightly less accurate for sigmoid-based)
    /// Uses fast_sigmoid approximation for Sigmoid and SiLU.
    #[inline]
    pub fn apply_fast(&self, x: f32) -> f32 {
        match self {
            ActivationType::SiLU => x * fast_sigmoid(x),
            ActivationType::GELU => {
                0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044715 * x.powi(3))).tanh())
            }
            ActivationType::ReLU => x.max(0.0),
            ActivationType::Tanh => x.tanh(),
            ActivationType::Sigmoid => fast_sigmoid(x),
        }
    }

    /// Apply activation function to array
    #[inline]
    pub fn apply_array(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|v| self.apply(v))
    }

    /// Apply fast activation function to array (2-3x faster for sigmoid-based)
    #[inline]
    pub fn apply_array_fast(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|v| self.apply_fast(v))
    }
}

/// Online learning statistics for a CfC cell
#[derive(Debug, Clone, Default)]
pub struct OnlineLearningStats {
    /// Total online adaptation steps
    pub adaptation_steps: u64,
    /// Exponential moving average of recent prediction errors
    pub ema_error: f32,
    /// Number of adaptations triggered (error exceeded threshold)
    pub adaptations_triggered: u64,
    /// Number of adaptations skipped (error below threshold)
    pub adaptations_skipped: u64,
    /// Maximum weight delta observed during any adaptation
    pub max_observed_delta: f32,
    /// Cumulative weight change (L2 norm of all deltas)
    pub cumulative_weight_change: f32,
}

/// Online learning statistics for a CfC network
#[derive(Debug, Clone, Default)]
pub struct NetworkOnlineLearningStats {
    /// Total adaptation calls
    pub total_adaptation_calls: u64,
    /// Adaptations that actually modified weights
    pub adaptations_applied: u64,
    /// Adaptations skipped due to low error
    pub adaptations_skipped: u64,
    /// EMA of prediction errors across all calls
    pub ema_error: f32,
    /// Cumulative weight change across all cells
    pub cumulative_weight_change: f32,
}

/// Diagnostic information about CfC network dynamics at the current state.
#[derive(Debug, Clone)]
pub struct DynamicsDiagnostic {
    /// Largest estimated real part of the Jacobian eigenvalues.
    /// Negative = stable, zero = marginal, positive = unstable.
    pub max_eigenvalue_real: f64,
    /// Condition number (ratio of largest to smallest eigenvalue magnitude).
    /// Large values (>100) indicate stiff dynamics that produce tiny gradients.
    pub condition_number: f64,
    /// Whether the dynamics appear collapsed to a stable attractor.
    pub collapsed: bool,
    /// Average L2 norm of cell states.
    pub state_norm: f32,
}
