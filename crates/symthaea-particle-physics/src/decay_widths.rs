// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Particle decay width calculations.
//!
//! Computes partial and total decay widths for Standard Model particles.
//! Decay width Γ is related to lifetime τ by τ = ℏ/Γ.
//!
//! References:
//! - PDG Review of Particle Physics (2024).
//! - Griffiths, D. (2008). *Introduction to Elementary Particles*. Chapter 10.

use crate::constants::*;
use std::f64::consts::PI;

/// A decay channel with partial width and branching ratio.
#[derive(Debug, Clone)]
pub struct DecayChannel {
    pub name: &'static str,
    pub partial_width: f64, // GeV
    pub branching_ratio: f64,
}

// ── Muon Decay ──────────────────────────────────────────────────────────────

/// Muon decay width: μ → e ν_μ ν̄_e
///
/// Γ(μ) = G_F² m_μ⁵ / (192π³)
///
/// This gives τ_μ ≈ 2.2 μs.
pub fn muon_decay_width() -> f64 {
    G_FERMI * G_FERMI * M_MUON.powi(5) / (192.0 * PI * PI * PI)
}

/// Muon lifetime in seconds.
pub fn muon_lifetime() -> f64 {
    HBAR_GEV_S / muon_decay_width()
}

// ── Tau Decay ───────────────────────────────────────────────────────────────

/// Tau leptonic decay width: τ → e ν_τ ν̄_e (or τ → μ ν_τ ν̄_μ)
///
/// Same formula as muon decay with m_τ instead of m_μ.
pub fn tau_leptonic_width() -> f64 {
    G_FERMI * G_FERMI * M_TAU.powi(5) / (192.0 * PI * PI * PI)
}

// ── W Boson Decay ───────────────────────────────────────────────────────────

/// W boson partial widths (leading order).
///
/// Γ(W → l ν) = G_F M_W³ / (6π√2) for each lepton generation
/// Γ(W → q q̄') = N_c × G_F M_W³ / (6π√2) × |V_ij|² for each quark pair
pub fn w_boson_channels() -> Vec<DecayChannel> {
    let leptonic_width = G_FERMI * M_W.powi(3) / (6.0 * PI * 2.0_f64.sqrt());
    let hadronic_width = N_C * leptonic_width; // per generation (|V_ud|²≈1)

    let total = 3.0 * leptonic_width + 2.0 * hadronic_width; // 3 lepton + 2 quark generations accessible

    vec![
        DecayChannel {
            name: "W → e νe",
            partial_width: leptonic_width,
            branching_ratio: leptonic_width / total,
        },
        DecayChannel {
            name: "W → μ νμ",
            partial_width: leptonic_width,
            branching_ratio: leptonic_width / total,
        },
        DecayChannel {
            name: "W → τ ντ",
            partial_width: leptonic_width,
            branching_ratio: leptonic_width / total,
        },
        DecayChannel {
            name: "W → ud̄ (+ cs̄)",
            partial_width: 2.0 * hadronic_width,
            branching_ratio: 2.0 * hadronic_width / total,
        },
    ]
}

/// W boson total width (leading order).
pub fn w_total_width() -> f64 {
    let leptonic = G_FERMI * M_W.powi(3) / (6.0 * PI * 2.0_f64.sqrt());
    3.0 * leptonic + 2.0 * N_C * leptonic
}

// ── Z Boson Decay ───────────────────────────────────────────────────────────

/// Z boson partial width to a fermion pair.
///
/// Γ(Z → f f̄) = (G_F M_Z³)/(6π√2) × N_c × (g_V² + g_A²)
///
/// where g_V = T3 - 2Q sin²θ_W, g_A = T3
pub fn z_partial_width(t3: f64, charge: f64, n_colors: f64) -> f64 {
    let gv = t3 - 2.0 * charge * SIN2_THETA_W;
    let ga = t3;
    G_FERMI * M_Z.powi(3) / (6.0 * PI * 2.0_f64.sqrt()) * n_colors * (gv * gv + ga * ga)
}

/// Z boson total width (sum over all accessible fermions).
pub fn z_total_width() -> f64 {
    let mut total = 0.0;

    // Neutrinos (3 generations): T3 = +1/2, Q = 0
    total += 3.0 * z_partial_width(0.5, 0.0, 1.0);

    // Charged leptons (3 generations): T3 = -1/2, Q = -1
    total += 3.0 * z_partial_width(-0.5, -1.0, 1.0);

    // Up-type quarks (u, c): T3 = +1/2, Q = +2/3 (top too heavy)
    total += 2.0 * z_partial_width(0.5, 2.0 / 3.0, N_C);

    // Down-type quarks (d, s, b): T3 = -1/2, Q = -1/3
    total += 3.0 * z_partial_width(-0.5, -1.0 / 3.0, N_C);

    total
}

// ── Pion Decay ──────────────────────────────────────────────────────────────

/// Charged pion decay width: π⁺ → μ⁺ ν_μ
///
/// Γ(π → μν) = (G_F² f_π² m_π)/(8π) × m_μ² × (1 - m_μ²/m_π²)² × |V_ud|²
///
/// f_π ≈ 130.2 MeV (pion decay constant)
pub fn pion_decay_width() -> f64 {
    let f_pi = 0.1302; // GeV
    let v_ud_sq = 0.974_4 * 0.974_4; // |V_ud|²
    let mass_ratio_sq = (M_MUON / M_PION_CHARGED).powi(2);

    G_FERMI * G_FERMI * f_pi * f_pi * M_PION_CHARGED / (8.0 * PI)
        * M_MUON * M_MUON
        * (1.0 - mass_ratio_sq).powi(2)
        * v_ud_sq
}

/// Pion lifetime in seconds.
pub fn pion_lifetime() -> f64 {
    HBAR_GEV_S / pion_decay_width()
}

// ── Top Quark Decay ─────────────────────────────────────────────────────────

/// Top quark decay width: t → W b
///
/// Γ(t → Wb) = (G_F m_t³)/(8π√2) × |V_tb|² × (1 - M_W²/m_t²)² × (1 + 2M_W²/m_t²)
pub fn top_decay_width() -> f64 {
    let v_tb_sq = 0.999_1 * 0.999_1;
    let x = (M_W / M_TOP).powi(2);

    G_FERMI * M_TOP.powi(3) / (8.0 * PI * 2.0_f64.sqrt())
        * v_tb_sq
        * (1.0 - x).powi(2)
        * (1.0 + 2.0 * x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_muon_lifetime() {
        // τ_μ ≈ 2.2 × 10⁻⁶ s
        let tau = muon_lifetime();
        assert!(
            (tau - 2.2e-6).abs() < 0.3e-6,
            "Muon lifetime = {:.3e} s, expected ≈ 2.2e-6 s",
            tau
        );
    }

    #[test]
    fn test_w_total_width() {
        // PDG: Γ_W = 2.085 ± 0.042 GeV
        let gamma = w_total_width();
        assert!(
            (gamma - GAMMA_W).abs() < 0.2,
            "W width = {:.3} GeV, expected ≈ {:.3} GeV",
            gamma,
            GAMMA_W
        );
    }

    #[test]
    fn test_w_leptonic_branching() {
        // BR(W → lν) ≈ 10.86% per generation
        let channels = w_boson_channels();
        let br_e = channels[0].branching_ratio;
        assert!(
            (br_e - 0.1086).abs() < 0.02,
            "BR(W→eν) = {:.4}, expected ≈ 0.1086",
            br_e
        );
    }

    #[test]
    fn test_z_total_width() {
        // PDG: Γ_Z = 2.4952 ± 0.0023 GeV
        let gamma = z_total_width();
        assert!(
            (gamma - GAMMA_Z).abs() < 0.3,
            "Z width = {:.3} GeV, expected ≈ {:.3} GeV",
            gamma,
            GAMMA_Z
        );
    }

    #[test]
    fn test_pion_lifetime() {
        // τ_π ≈ 2.6 × 10⁻⁸ s
        let tau = pion_lifetime();
        assert!(
            tau > 1e-9 && tau < 1e-7,
            "Pion lifetime = {:.3e} s, expected ~2.6e-8 s",
            tau
        );
    }

    #[test]
    fn test_top_decay_width() {
        // PDG: Γ_t ≈ 1.42 GeV
        let gamma = top_decay_width();
        assert!(
            (gamma - GAMMA_TOP).abs() < 0.3,
            "Top width = {:.3} GeV, expected ≈ {:.3} GeV",
            gamma,
            GAMMA_TOP
        );
    }
}
