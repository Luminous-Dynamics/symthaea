// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! 1-loop renormalization group equations and beta functions.
//!
//! Implements the running of gauge couplings and investigates gauge unification.
//!
//! References:
//! - Peskin & Schroeder (1995), Chapter 16.
//! - Langacker, P. (2010). *The Standard Model and Beyond*. CRC Press.
//! - PDG Review: Quantum Chromodynamics (2024).

use crate::constants::*;
use std::f64::consts::PI;

/// Beta function coefficients for a gauge theory.
#[derive(Debug, Clone, Copy)]
pub struct BetaCoefficients {
    /// 1-loop: β₀
    pub b0: f64,
    /// 2-loop: β₁
    pub b1: f64,
}

/// QCD beta function (1-loop + 2-loop).
///
/// β₀ = (33 - 2n_f) / (12π)
/// β₁ = (153 - 19n_f) / (24π²)
pub fn qcd_beta(n_f: u8) -> BetaCoefficients {
    let nf = n_f as f64;
    BetaCoefficients {
        b0: (33.0 - 2.0 * nf) / (12.0 * PI),
        b1: (153.0 - 19.0 * nf) / (24.0 * PI * PI),
    }
}

/// QED beta function (1-loop).
///
/// β₀ = -4/3 × Σ_f N_c Q_f² / (4π)
/// (negative sign = NOT asymptotically free)
pub fn qed_beta() -> BetaCoefficients {
    // Sum over all charged fermions at high energy (3 lepton + 6 quark)
    // Σ N_c Q² = 3×1 + 3×(4/9+1/9+4/9+1/9+4/9+1/9) = 3 + 3×15/9 = 3+5 = 8
    let sum_ncq2 = 3.0 * 1.0 + N_C * (3.0 * 4.0 / 9.0 + 3.0 * 1.0 / 9.0);
    BetaCoefficients {
        b0: -4.0 / 3.0 * sum_ncq2 / (4.0 * PI),
        b1: 0.0, // 2-loop QED not implemented
    }
}

/// Standard Model gauge coupling unification analysis.
///
/// The three SM gauge couplings are:
/// - α₁ = (5/3) α_EM / cos²θ_W  (U(1)_Y, GUT normalized)
/// - α₂ = α_EM / sin²θ_W        (SU(2)_L)
/// - α₃ = α_s                     (SU(3)_C)
///
/// Returns (α₁, α₂, α₃) at scale Q (GeV).
pub fn gauge_couplings_at_scale(q: f64) -> (f64, f64, f64) {
    let q2 = q * q;
    let mz2 = M_Z * M_Z;
    let log_ratio = (q2 / mz2).ln();

    // Values at M_Z
    let alpha1_mz = (5.0 / 3.0) * ALPHA_EM / COS2_THETA_W;
    let alpha2_mz = ALPHA_EM / SIN2_THETA_W;
    let alpha3_mz = ALPHA_S_MZ;

    // 1-loop beta coefficients (SM with 3 generations + 1 Higgs doublet)
    // Standard convention: 1/α_i(μ) = 1/α_i(M_Z) + b_i/(2π) × ln(μ²/M_Z²)
    // b_i = (11C_A - 4T_F n_f)/3 for non-abelian, adjusted for each group
    // PDG convention with GUT normalization:
    let b1 = -41.0 / 10.0; // U(1)_Y: negative → coupling grows
    let b2 = 19.0 / 6.0; // SU(2)_L: positive → coupling decreases
    let b3 = 7.0; // SU(3)_C: positive → asymptotic freedom

    let inv_alpha1 = 1.0 / alpha1_mz + b1 / (4.0 * PI) * log_ratio;
    let inv_alpha2 = 1.0 / alpha2_mz + b2 / (4.0 * PI) * log_ratio;
    let inv_alpha3 = 1.0 / alpha3_mz + b3 / (4.0 * PI) * log_ratio;

    let a1 = if inv_alpha1 > 0.0 { 1.0 / inv_alpha1 } else { 1.0 };
    let a2 = if inv_alpha2 > 0.0 { 1.0 / inv_alpha2 } else { 1.0 };
    let a3 = if inv_alpha3 > 0.0 { 1.0 / inv_alpha3 } else { 1.0 };

    (a1, a2, a3)
}

/// Find approximate gauge unification scale (where α₂ ≈ α₃).
///
/// In the SM, exact unification doesn't occur — the three couplings
/// don't meet at a single point. Returns the scale where α₂ and α₃
/// are closest (approximate GUT scale).
pub fn approximate_unification_scale() -> f64 {
    // Search between M_Z and 10^18 GeV
    let mut best_q = M_Z;
    let mut best_diff = f64::MAX;

    let mut log_q = M_Z.ln();
    let max_log_q = (1e18_f64).ln();
    let step = (max_log_q - log_q) / 1000.0;

    while log_q < max_log_q {
        let q = log_q.exp();
        let (a1, a2, a3) = gauge_couplings_at_scale(q);

        // Measure how close all three are
        let diff = (a1 - a2).abs() + (a2 - a3).abs() + (a1 - a3).abs();
        if diff < best_diff {
            best_diff = diff;
            best_q = q;
        }

        log_q += step;
    }

    best_q
}

/// QCD Lambda parameter (Λ_QCD) in GeV.
///
/// From α_s(Q²) = 1/(β₀ ln(Q²/Λ²)):
/// Λ = Q × exp(-1/(2 β₀ α_s(Q)))
/// Using Q = M_Z, α_s = α_s(M_Z).
pub fn lambda_qcd(n_f: u8) -> f64 {
    let nf = n_f as f64;
    let b0 = (33.0 - 2.0 * nf) / (12.0 * PI);
    M_Z * (-1.0 / (2.0 * b0 * ALPHA_S_MZ)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qcd_asymptotic_freedom() {
        // β₀ > 0 for n_f ≤ 16 → asymptotic freedom
        for nf in 0..=6 {
            let beta = qcd_beta(nf);
            assert!(beta.b0 > 0.0, "QCD should be AF for n_f={}", nf);
        }
    }

    #[test]
    fn test_qed_not_asymptotically_free() {
        let beta = qed_beta();
        assert!(
            beta.b0 < 0.0,
            "QED β₀ = {}, should be negative (not AF)",
            beta.b0
        );
    }

    #[test]
    fn test_couplings_at_mz() {
        let (a1, a2, a3) = gauge_couplings_at_scale(M_Z);
        // At M_Z, should match input values
        let alpha1_expected = (5.0 / 3.0) * ALPHA_EM / COS2_THETA_W;
        let alpha2_expected = ALPHA_EM / SIN2_THETA_W;

        assert!((a1 - alpha1_expected).abs() < 1e-4);
        assert!((a2 - alpha2_expected).abs() < 1e-4);
        assert!((a3 - ALPHA_S_MZ).abs() < 1e-4);
    }

    #[test]
    fn test_coupling_convergence() {
        // At high scale, α₂ and α₃ should be closer than at M_Z
        let (_, a2_low, a3_low) = gauge_couplings_at_scale(M_Z);
        let (_, a2_high, a3_high) = gauge_couplings_at_scale(1e10);

        let diff_low = (a2_low - a3_low).abs();
        let diff_high = (a2_high - a3_high).abs();

        assert!(
            diff_high < diff_low,
            "Couplings should converge: diff_MZ={:.4}, diff_10^10={:.4}",
            diff_low,
            diff_high
        );
    }

    #[test]
    fn test_unification_scale() {
        // SM approximate unification should be ~10^13-10^16 GeV
        let gut_scale = approximate_unification_scale();
        assert!(
            gut_scale > 1e12 && gut_scale < 1e17,
            "GUT scale = {:.2e} GeV, expected ~10^13-10^16",
            gut_scale
        );
    }

    #[test]
    fn test_lambda_qcd() {
        // Λ_QCD at 1-loop (without threshold matching) gives ~80-300 MeV
        // The exact value is scheme-dependent; we verify it's in the right ballpark
        let lambda = lambda_qcd(5);
        assert!(
            lambda > 0.05 && lambda < 0.5,
            "Λ_QCD(5) = {:.3} GeV, expected ~0.08-0.3 (1-loop)",
            lambda
        );
    }
}
