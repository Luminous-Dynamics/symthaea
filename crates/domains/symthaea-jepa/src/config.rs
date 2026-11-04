// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! JEPA configuration: latent dimensions, EMA momentum, and a legacy operation-cost estimate.

use serde::{Deserialize, Serialize};

/// Configuration for the JEPA (Joint Embedding Predictive Architecture) engine.
///
/// JEPA predicts future states in a learned latent space rather than reconstructing
/// raw observations. The 128D latent representation can reduce the architecture's
/// operation count relative to 16,384D HDC vectors. Whether this reduces measured
/// hardware energy at matched accuracy must be established by workload measurement;
/// latent compression alone is not a thermodynamic measurement.
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

    /// Legacy model-estimated energy per forward pass, represented in joule-scaled
    /// units. The automatic estimate uses architecture-derived operation counts
    /// and a fixed scenario conversion factor; it is not a hardware energy
    /// measurement and does not count FLOPs as physical bit erasures.
    /// Override with 0.0 to use the legacy operation-count estimate.
    pub energy_cost_per_forward: f64,

    /// How strongly substrate `tau_factor` modulates JEPA prediction horizons.
    /// 0.0 = no modulation, 1.0 = full substrate coupling.
    pub tau_factor_sensitivity: f32,

    /// Variance regularization floor. If any latent dimension's variance drops
    /// below this, a penalty is added to prevent representation collapse.
    /// Default: 0.01.
    pub variance_floor: f32,
}

/// Reference Landauer heat bound at 300 K for one logically irreversible bit
/// erasure under the relevant thermal-bath assumptions.
/// This physical reference is not a per-FLOP energy cost: a floating-point
/// operation is not, by itself, evidence of a one-bit memory-erasure event.
/// Retained under its legacy public name for compatibility.
pub const LANDAUER_KT_LN2_300K: f64 = 2.87e-21;

/// Legacy scenario conversion factor used by the operation-count estimate.
///
/// The historical derivation uses nominal CPU TDP divided by clock frequency
/// (15 W / 4.8 GHz) and compares that scale with the 300 K Landauer erasure
/// reference. TDP/clock is not a measured energy-per-FLOP value: it does not
/// account for instructions per cycle, vector width, utilization, memory and
/// accelerator work, DVFS, idle/package power, or workload-specific energy.
///
/// Retained for output/API compatibility; do not cite it as hardware telemetry
/// or as an empirical measurement of this CPU's energy per floating-point op.
pub const HARDWARE_OVERHEAD: f64 = 1.09e12;

impl JepaConfig {
    /// Compute the legacy scenario estimate from architecture-derived
    /// operation counts.
    ///
    /// Counted operations:
    /// - Context encoder: input_dim × latent_dim (matmul) + latent_dim (SiLU)
    /// - Target encoder: same as context (but no backward, so counted separately)
    /// - Predictor: (latent_dim + num_actions) × latent_dim (layer 1) + latent_dim × latent_dim (layer 2)
    ///
    /// This multiplies the estimated count by a fixed operation-cost scenario.
    /// It does not treat each FLOP as an erased bit in physical memory and is
    /// not a measurement of this workload's joules per inference. The separate
    /// Landauer constant is only a reference for logically irreversible erasure.
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
