// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! JEPA configuration: latent dimensions, EMA momentum, energy tracking.

use serde::{Deserialize, Serialize};

/// Configuration for the JEPA (Joint Embedding Predictive Architecture) engine.
///
/// JEPA predicts future states in a learned latent space rather than reconstructing
/// raw observations. This reduces the thermodynamic cost of free energy minimization
/// by operating on compressed representations (128D latent vs 16,384D HDC).
///
/// # References
/// - LeCun (2022): "A Path Towards Autonomous Machine Intelligence"
/// - Assran et al. (2023): "Self-Supervised Learning from Images with a Joint-Embedding
///   Predictive Architecture" (I-JEPA)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JepaConfig {
    /// Latent space dimension (default 128).
    /// 128x compression from 16,384D HDC vectors.
    /// Sweet spot: 64 loses too much structure, 256 gains little for 2x cost.
    pub latent_dim: usize,

    /// Input dimension (HDC vector size, default 16,384 = `HDC_DIMENSION`).
    pub input_dim: usize,

    /// Number of motor command types for action encoding (default 8).
    pub num_actions: usize,

    /// EMA momentum for target encoder (default 0.996).
    /// Higher = slower target drift = more stable representations.
    /// Standard BYOL/JEPA range: 0.99-0.999.
    pub ema_momentum: f32,

    /// Learning rate for context encoder and predictor (default 0.001).
    pub learning_rate: f32,

    /// Energy cost per forward pass in joules (Landauer tracking).
    /// Computed from actual FLOP count: each multiply-accumulate erases ~1 bit,
    /// costing kT ln 2 ≈ 2.87e-21 J at 300K (Landauer 1961).
    /// Override with 0.0 to use automatic FLOP-based estimation.
    pub energy_cost_per_forward: f64,

    /// How strongly substrate `tau_factor` modulates JEPA prediction horizons.
    /// 0.0 = no modulation, 1.0 = full substrate coupling.
    pub tau_factor_sensitivity: f32,

    /// Variance regularization floor. If any latent dimension's variance drops
    /// below this, a penalty is added to prevent representation collapse.
    /// Default: 0.01.
    pub variance_floor: f32,
}

/// Landauer limit: kT ln 2 at 300K (joules per bit erasure).
/// Landauer (1961), "Irreversibility and Heat Generation in the Computing Process".
pub const LANDAUER_KT_LN2_300K: f64 = 2.87e-21;

/// Hardware overhead factor: measured energy per FLOP / Landauer floor.
///
/// Derived from actual hardware (Intel i9-8950HK, 14nm Coffee Lake):
///   - Single-core turbo: 4.8 GHz, ~15W per-core TDP
///   - JEPA runs scalar f32 (no AVX vectorization in matmul loops)
///   - Energy/FLOP = 15W / 4.8e9 Hz = 3.12e-9 J
///   - Overhead = 3.12e-9 / 2.87e-21 = 1.09e12
///
/// This is ~10^8× higher than theoretical 7nm limits (Theis & Wong 2017)
/// because: (a) 14nm process, (b) scalar execution, (c) full-core power
/// not just switching energy. The value is honest — it reflects what this
/// specific CPU actually dissipates per floating-point operation.
pub const HARDWARE_OVERHEAD: f64 = 1.09e12;

impl JepaConfig {
    /// Compute energy per forward pass from actual FLOP count.
    ///
    /// FLOPs per forward:
    /// - Context encoder: input_dim × latent_dim (matmul) + latent_dim (SiLU)
    /// - Target encoder: same as context (but no backward, so counted separately)
    /// - Predictor: (latent_dim + num_actions) × latent_dim (layer 1) + latent_dim × latent_dim (layer 2)
    ///
    /// Each FLOP ≈ 1 bit erasure at Landauer floor, multiplied by silicon overhead.
    pub fn compute_energy_per_forward(&self) -> f64 {
        let encoder_flops = self.input_dim * self.latent_dim + self.latent_dim; // matmul + activation
        let predictor_l1 = (self.latent_dim + self.num_actions) * self.latent_dim;
        let predictor_l2 = self.latent_dim * self.latent_dim;
        let total_flops = encoder_flops * 2 + predictor_l1 + predictor_l2; // 2 encoders + predictor

        total_flops as f64 * LANDAUER_KT_LN2_300K * HARDWARE_OVERHEAD
    }

    /// Get the effective energy per forward: uses FLOP-based if configured as 0.0,
    /// otherwise uses the user-specified override.
    pub fn effective_energy_per_forward(&self) -> f64 {
        if self.energy_cost_per_forward == 0.0 {
            self.compute_energy_per_forward()
        } else {
            self.energy_cost_per_forward
        }
    }
}

impl Default for JepaConfig {
    fn default() -> Self {
        Self {
            latent_dim: 128,
            input_dim: symthaea_core::hdc::HDC_DIMENSION,
            num_actions: 8,
            ema_momentum: 0.996,
            learning_rate: 0.001,
            energy_cost_per_forward: 0.0, // Auto-compute from FLOP count
            tau_factor_sensitivity: 0.5,
            variance_floor: 0.01,
        }
    }
}
