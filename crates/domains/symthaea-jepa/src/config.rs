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

    /// Configured energy estimate per forward pass, nominally in joules.
    ///
    /// This value is an implementation-cost override, not a Landauer bound.
    /// The default value of 0.0 selects the legacy operation-count scenario
    /// estimate below. Supply a measured workload value only when its hardware,
    /// measurement boundary, idle baseline, and uncertainty are recorded.
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

/// Legacy scenario estimate per counted JEPA proxy operation, in joules.
///
/// This preserves the historical numerical scale used by the default JEPA
/// energy estimate. Its original derivation used a Landauer reference and a
/// TDP/clock-rate approximation; that derivation is not a device measurement,
/// and one counted proxy operation is not a physical bit erasure.
pub const LEGACY_SCENARIO_ENERGY_PER_COUNTED_OP_J: f64 = 3.1283e-9;

/// Legacy multiplier retained for source compatibility.
///
/// This value was historically described as a measured energy-per-FLOP
/// overhead. It is not a measurement and must not be used to claim hardware
/// efficiency or a physical Landauer cost.
pub const HARDWARE_OVERHEAD: f64 = 1.09e12;

/// Error returned when JEPA's configured operation-count proxy overflows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JepaOperationCountError {
    Overflow,
}

/// Counts of arithmetic work in the configured JEPA forward architecture.
///
/// Matrix multiply-accumulates and activation evaluations are reported
/// separately. A MAC is not automatically an erased bit, and a MAC-count
/// proxy is not a physical energy measurement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JepaOperationCounts {
    pub context_encoder_macs: u128,
    pub target_encoder_macs: u128,
    pub predictor_layer_1_macs: u128,
    pub predictor_layer_2_macs: u128,
    pub activation_evaluations: u128,
    /// Compatibility count matching the historical estimator's convention:
    /// one unit per MAC plus one per activation evaluation.
    pub legacy_operation_proxy_count: u128,
}


impl JepaConfig {
    /// Return checked operation counts for the configured forward architecture.
    ///
    /// The historical architecture count assumes two encoder passes (context
    /// and target), two predictor matrix layers, and one activation evaluation
    /// per output coordinate in each encoder. These are abstract operation
    /// counts, not physical bit erasures.
    pub fn checked_operation_counts(
        &self,
    ) -> Result<JepaOperationCounts, JepaOperationCountError> {
        let input = self.input_dim as u128;
        let latent = self.latent_dim as u128;
        let actions = self.num_actions as u128;

        let encoder_macs = input
            .checked_mul(latent)
            .ok_or(JepaOperationCountError::Overflow)?;
        let predictor_layer_1_macs = latent
            .checked_add(actions)
            .and_then(|width| width.checked_mul(latent))
            .ok_or(JepaOperationCountError::Overflow)?;
        let predictor_layer_2_macs = latent
            .checked_mul(latent)
            .ok_or(JepaOperationCountError::Overflow)?;
        let activation_evaluations = latent
            .checked_mul(2)
            .ok_or(JepaOperationCountError::Overflow)?;
        let total_macs = encoder_macs
            .checked_mul(2)
            .and_then(|count| count.checked_add(predictor_layer_1_macs))
            .and_then(|count| count.checked_add(predictor_layer_2_macs))
            .ok_or(JepaOperationCountError::Overflow)?;
        let legacy_operation_proxy_count = total_macs
            .checked_add(activation_evaluations)
            .ok_or(JepaOperationCountError::Overflow)?;

        Ok(JepaOperationCounts {
            context_encoder_macs: encoder_macs,
            target_encoder_macs: encoder_macs,
            predictor_layer_1_macs,
            predictor_layer_2_macs,
            activation_evaluations,
            legacy_operation_proxy_count,
        })
    }

    /// Checked legacy scenario energy estimate for one forward pass.
    ///
    /// The per-operation coefficient preserves the previous default numerical
    /// estimate, but is not a measured energy/FLOP and does not arise from
    /// counting physical bit erasures. Use hardware energy instrumentation for
    /// measured joules/inference.
    pub fn try_compute_energy_per_forward(
        &self,
    ) -> Result<f64, JepaOperationCountError> {
        let counts = self.checked_operation_counts()?;
        Ok(counts.legacy_operation_proxy_count as f64
            * LEGACY_SCENARIO_ENERGY_PER_COUNTED_OP_J)
    }

    /// Compatibility wrapper for the historical scenario estimate.
    ///
    /// Returns infinity if the operation-count arithmetic overflows; new
    /// callers should use `try_compute_energy_per_forward` for a typed error.
    /// This estimate is not a physical Landauer bound or a measured hardware
    /// energy value.
    pub fn compute_energy_per_forward(&self) -> f64 {
        self.try_compute_energy_per_forward()
            .unwrap_or(f64::INFINITY)
    }

    /// Effective configured estimate per forward.
    ///
    /// A non-zero override is returned unchanged. A zero override selects the
    /// legacy operation-count scenario estimate, not a Landauer-per-FLOP model.
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


#[cfg(test)]
mod operation_count_tests {
    use super::*;

    #[test]
    fn default_operation_counts_are_explicit_and_separate_macs_from_activations() {
        let counts = JepaConfig::default().checked_operation_counts().unwrap();
        assert_eq!(counts.context_encoder_macs, 2_097_152);
        assert_eq!(counts.target_encoder_macs, 2_097_152);
        assert_eq!(counts.predictor_layer_1_macs, 17_408);
        assert_eq!(counts.predictor_layer_2_macs, 16_384);
        assert_eq!(counts.activation_evaluations, 256);
        assert_eq!(counts.legacy_operation_proxy_count, 4_228_352);
    }

    #[test]
    fn zero_dimensions_produce_zero_operation_count() {
        let config = JepaConfig {
            input_dim: 0,
            latent_dim: 0,
            num_actions: 0,
            ..JepaConfig::default()
        };
        let counts = config.checked_operation_counts().unwrap();
        assert_eq!(counts.legacy_operation_proxy_count, 0);
    }

    #[test]
    fn operation_count_overflow_is_typed() {
        let config = JepaConfig {
            input_dim: usize::MAX,
            latent_dim: usize::MAX,
            num_actions: usize::MAX,
            ..JepaConfig::default()
        };
        assert_eq!(
            config.checked_operation_counts(),
            Err(JepaOperationCountError::Overflow)
        );
        assert_eq!(
            config.try_compute_energy_per_forward(),
            Err(JepaOperationCountError::Overflow)
        );
        assert!(config.compute_energy_per_forward().is_infinite());
    }

    #[test]
    fn checked_estimate_preserves_legacy_default_numeric_scale() {
        let config = JepaConfig::default();
        let counts = config.checked_operation_counts().unwrap();
        let expected = counts.legacy_operation_proxy_count as f64
            * LEGACY_SCENARIO_ENERGY_PER_COUNTED_OP_J;
        assert_eq!(config.try_compute_energy_per_forward().unwrap(), expected);
        assert_eq!(config.compute_energy_per_forward(), expected);
    }
}
