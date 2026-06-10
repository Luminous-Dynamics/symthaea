// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Φ_eff Integration: effective_phi = Φ × R^γ
//!
//! Provides the principled confidence operator that gates all tool use.
//!
//! ## Monotonicity Proof (INV-1)
//!
//! `Φ_eff = Φ × R^γ` where γ > 0.
//! `∂Φ_eff/∂R = Φ × γ × R^(γ-1) ≥ 0` for R ∈ [0,1], γ > 0.
//! Therefore Φ_eff is monotone non-decreasing in R.

use super::calibrator::TheoryCalibrator;
use super::types::MultiTheoryMetrics;

/// Compute effective Φ = Φ × R^γ.
///
/// # Invariants
/// - INV-1 (Monotonic Caution): If R decreases, Φ_eff cannot increase.
/// - γ ∈ [1.0, 4.0] (debug-asserted).
///
/// # Arguments
/// - `phi`: Raw integrated information value.
/// - `reliability`: R ∈ [0, 1], consensus among calibrated theories.
/// - `gamma`: Sensitivity exponent ∈ [1.0, 4.0].
pub fn effective_phi(phi: f64, reliability: f64, gamma: f64) -> f64 {
    debug_assert!((1.0..=4.0).contains(&gamma), "γ out of bounds: {}", gamma);
    phi * reliability.powf(gamma)
}

/// Convenience: compute Φ_eff from metrics and a calibrator.
pub fn compute_phi_eff(
    metrics: &MultiTheoryMetrics,
    calibrator: &TheoryCalibrator,
) -> PhiEffResult {
    let reliability = calibrator.reliability(metrics);
    let gamma = calibrator.gamma();
    let phi_eff = effective_phi(metrics.phi, reliability, gamma);

    PhiEffResult {
        phi_raw: metrics.phi,
        reliability,
        gamma,
        phi_eff,
    }
}

/// Result of a Φ_eff computation, including all inputs for telemetry.
#[derive(Debug, Clone, Copy)]
pub struct PhiEffResult {
    /// Raw Φ from IIT.
    pub phi_raw: f64,
    /// Reliability R ∈ [0, 1].
    pub reliability: f64,
    /// Sensitivity γ ∈ [1.0, 4.0].
    pub gamma: f64,
    /// Effective Φ = Φ × R^γ.
    pub phi_eff: f64,
}

/// R thresholds for behavior gating.
pub mod thresholds {
    /// Below this: no simulation at all, epistemic action only.
    pub const R_SIM_MIN: f64 = 0.15;
    /// Below this: epistemic rollouts only (not reward-maximizing).
    pub const R_EPISTEMIC_MIN: f64 = 0.30;
    /// Below this: micro-MCTS only (not full planning).
    pub const R_PLAN_MIN: f64 = 0.50;
    /// EVS threshold for deciding whether to simulate.
    pub const EVS_THRESHOLD: f64 = 0.3;
    /// R below which engine prefers info-seeking over irreversible action (INV-5).
    pub const R_EPISTEMIC_THRESHOLD: f64 = 0.30;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effective_phi_basic() {
        // At R=1.0, Φ_eff = Φ
        assert!((effective_phi(0.8, 1.0, 2.0) - 0.8).abs() < 1e-10);
        // At R=0.0, Φ_eff = 0
        assert!((effective_phi(0.8, 0.0, 2.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_inv1_monotonic_caution() {
        // INV-1: If R decreases, Φ_eff must not increase.
        let phi = 0.8;
        let gamma = 2.0;

        let mut prev_phi_eff = effective_phi(phi, 1.0, gamma);
        // Sweep R from 1.0 down to 0.0
        let mut r = 1.0;
        while r >= 0.0 {
            let phi_eff = effective_phi(phi, r, gamma);
            assert!(
                phi_eff <= prev_phi_eff + 1e-10,
                "INV-1 violated: Φ_eff({}) = {} > Φ_eff_prev = {}",
                r,
                phi_eff,
                prev_phi_eff
            );
            prev_phi_eff = phi_eff;
            r -= 0.01;
        }
    }

    #[test]
    fn test_inv1_monotonic_sweep_all_gammas() {
        // Verify monotonicity for all valid γ values
        let phi = 0.7;
        for gamma_10x in 10..=40 {
            let gamma = gamma_10x as f64 / 10.0;
            let mut prev = effective_phi(phi, 1.0, gamma);
            let mut r = 1.0;
            while r >= 0.0 {
                let current = effective_phi(phi, r, gamma);
                assert!(
                    current <= prev + 1e-10,
                    "INV-1 violated at γ={}, R={}",
                    gamma,
                    r
                );
                prev = current;
                r -= 0.01;
            }
        }
    }

    #[test]
    fn test_gamma_effect_table() {
        // Verify the γ=2.0 behavior table from the RFC
        let phi = 1.0;
        let gamma = 2.0;

        let cases = [
            (0.95, 0.90),
            (0.70, 0.49),
            (0.50, 0.25),
            (0.30, 0.09),
            (0.10, 0.01),
        ];

        for (r, expected_ratio) in cases {
            let phi_eff = effective_phi(phi, r, gamma);
            assert!(
                (phi_eff - expected_ratio).abs() < 0.01,
                "At R={}, expected Φ_eff/Φ ≈ {}, got {}",
                r,
                expected_ratio,
                phi_eff
            );
        }
    }

    #[test]
    fn test_compute_phi_eff_convenience() {
        let metrics = crate::consciousness::epistemic_conflict::MultiTheoryMetrics {
            phi: 0.8,
            gwt: 0.8,
            ast: 0.8,
            pp: 0.8,
            rpt: 0.8,
            embodiment: 0.8,
            unified: 0.8,
        };
        let calibrator = TheoryCalibrator::new();
        let result = compute_phi_eff(&metrics, &calibrator);
        assert!(result.phi_eff > 0.0);
        assert!(result.phi_eff <= result.phi_raw);
        assert!(result.reliability > 0.0);
    }
}
