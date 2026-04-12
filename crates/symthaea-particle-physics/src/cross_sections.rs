// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tree-level cross-section calculations for Standard Model processes.
//!
//! Implements QED and electroweak cross-sections at leading order.
//! All cross-sections are in natural units (GeV⁻²) unless otherwise noted.
//!
//! References:
//! - Peskin & Schroeder (1995). *An Introduction to Quantum Field Theory*.
//! - Halzen & Martin (1984). *Quarks and Leptons*.
//! - PDG Review of Particle Physics (2024).

use crate::constants::*;
use std::f64::consts::PI;

/// Mandelstam variables for 2→2 scattering.
#[derive(Debug, Clone, Copy)]
pub struct Mandelstam {
    /// s = (p1 + p2)² — center-of-mass energy squared
    pub s: f64,
    /// t = (p1 - p3)² — momentum transfer squared
    pub t: f64,
    /// u = (p1 - p4)² — crossed momentum transfer
    pub u: f64,
}

impl Mandelstam {
    /// Create from s and cos(θ) in the CM frame (massless limit).
    /// t = -s/2 * (1 - cosθ), u = -s/2 * (1 + cosθ)
    pub fn from_s_cos_theta(s: f64, cos_theta: f64) -> Self {
        let t = -s / 2.0 * (1.0 - cos_theta);
        let u = -s / 2.0 * (1.0 + cos_theta);
        Self { s, t, u }
    }

    /// Verify s + t + u = Σ mᵢ² (should be ~0 for massless particles).
    pub fn check_sum(&self) -> f64 {
        self.s + self.t + self.u
    }
}

// ── QED Cross-Sections ──────────────────────────────────────────────────────

/// Total cross-section for e⁺e⁻ → μ⁺μ⁻ (QED, tree level).
///
/// σ = 4πα²/(3s) for √s >> m_μ
///
/// This is the "gold standard" QED process.
pub fn sigma_ee_to_mumu(s: f64) -> f64 {
    if s <= 4.0 * M_MUON * M_MUON {
        return 0.0; // Below threshold
    }

    let beta = (1.0 - 4.0 * M_MUON * M_MUON / s).sqrt();
    4.0 * PI * ALPHA_EM * ALPHA_EM / (3.0 * s) * beta * (3.0 - beta * beta) / 2.0
}

/// Total cross-section for e⁺e⁻ → μ⁺μ⁻ including Z-boson resonance.
///
/// Uses the Breit-Wigner propagator for the Z contribution.
/// σ = σ_QED * |1 + χ(s)|² where χ(s) encodes Z interference.
pub fn sigma_ee_to_mumu_with_z(s: f64) -> f64 {
    if s <= 4.0 * M_MUON * M_MUON {
        return 0.0;
    }

    let sigma_qed = sigma_ee_to_mumu(s);

    // Z propagator: χ(s) = s / [(s - M_Z²) + i*M_Z*Γ_Z] × coupling factor
    let mz2 = M_Z * M_Z;
    let denom_re = s - mz2;
    let denom_im = M_Z * GAMMA_Z;
    let denom_sq = denom_re * denom_re + denom_im * denom_im;

    // Coupling factor for Z→e⁺e⁻→μ⁺μ⁻ (vector part only, simplified)
    // g_V^e = -1/2 + 2sin²θ_W, g_A^e = -1/2
    let gv = -0.5 + 2.0 * SIN2_THETA_W;
    let ga = -0.5;

    // Interference term amplitude (simplified)
    let kappa = 1.0 / (4.0 * SIN2_THETA_W * COS2_THETA_W);
    let chi_re = kappa * s * denom_re / denom_sq;
    let chi_im = -kappa * s * denom_im / denom_sq;

    // |1 + χ|² for vector coupling (approximate, ignoring axial-vector)
    let factor = (1.0 + chi_re * (gv * gv + ga * ga)).powi(2) + (chi_im * (gv * gv + ga * ga)).powi(2);

    sigma_qed * factor
}

/// Differential cross-section dσ/dΩ for e⁺e⁻ → μ⁺μ⁻ (QED).
///
/// dσ/dΩ = α²/(4s) * (1 + cos²θ)
pub fn dsigma_domega_ee_mumu(s: f64, cos_theta: f64) -> f64 {
    ALPHA_EM * ALPHA_EM / (4.0 * s) * (1.0 + cos_theta * cos_theta)
}

/// Compton scattering total cross-section (Klein-Nishina).
///
/// σ_KN for photon energy ω in electron rest frame.
/// In the limit ω → 0: σ → σ_Thomson = 8πα²/(3m_e²)
pub fn sigma_compton(omega: f64) -> f64 {
    let x = omega / M_ELECTRON;

    if x < 0.01 {
        // Thomson limit
        return 8.0 * PI * ALPHA_EM * ALPHA_EM / (3.0 * M_ELECTRON * M_ELECTRON);
    }

    // Klein-Nishina formula
    let r_e_sq = ALPHA_EM * ALPHA_EM / (M_ELECTRON * M_ELECTRON);

    let term1 = (1.0 + x) / (x * x * x)
        * (2.0 * x * (1.0 + x) / (1.0 + 2.0 * x) - (1.0 + 2.0 * x).ln());
    let term2 = (1.0 + 2.0 * x).ln() / (2.0 * x);
    let term3 = -(1.0 + 3.0 * x) / (1.0 + 2.0 * x).powi(2);

    2.0 * PI * r_e_sq * (term1 + term2 + term3)
}

// ── R-ratio ─────────────────────────────────────────────────────────────────

/// R-ratio: σ(e⁺e⁻ → hadrons) / σ(e⁺e⁻ → μ⁺μ⁻)
///
/// R = N_c × Σ_q Q_q² for quark flavors kinematically accessible at √s.
/// Includes first-order QCD correction: R → R × (1 + α_s/π).
pub fn r_ratio(sqrt_s: f64) -> f64 {
    let s = sqrt_s * sqrt_s;

    // Quark charges squared × N_c
    let mut r = 0.0;

    // Up: Q = 2/3, threshold ~2*m_u ≈ 0
    if sqrt_s > 2.0 * M_UP {
        r += N_C * (2.0 / 3.0) * (2.0 / 3.0);
    }
    // Down: Q = -1/3
    if sqrt_s > 2.0 * M_DOWN {
        r += N_C * (1.0 / 3.0) * (1.0 / 3.0);
    }
    // Strange: Q = -1/3
    if sqrt_s > 2.0 * M_STRANGE {
        r += N_C * (1.0 / 3.0) * (1.0 / 3.0);
    }
    // Charm: Q = 2/3
    if sqrt_s > 2.0 * M_CHARM {
        r += N_C * (2.0 / 3.0) * (2.0 / 3.0);
    }
    // Bottom: Q = -1/3
    if sqrt_s > 2.0 * M_BOTTOM {
        r += N_C * (1.0 / 3.0) * (1.0 / 3.0);
    }
    // Top: Q = 2/3
    if sqrt_s > 2.0 * M_TOP {
        r += N_C * (2.0 / 3.0) * (2.0 / 3.0);
    }

    // QCD correction (1-loop)
    let alpha_s = alpha_s_running(s);
    r * (1.0 + alpha_s / PI)
}

/// Running strong coupling constant α_s(Q²) at 1-loop.
///
/// α_s(Q²) = α_s(M_Z²) / [1 + (b₀ α_s(M_Z²) / (2π)) ln(Q²/M_Z²)]
///
/// where b₀ = (33 - 2n_f) / (12π) for n_f active flavors.
pub fn alpha_s_running(q2: f64) -> f64 {
    if q2 <= 0.0 {
        return ALPHA_S_MZ;
    }

    // Count active flavors at scale Q
    let q = q2.sqrt();
    let nf = count_active_flavors(q);

    let b0 = (33.0 - 2.0 * nf as f64) / (12.0 * PI);
    let log_ratio = (q2 / (M_Z * M_Z)).ln();

    let denom = 1.0 + b0 * ALPHA_S_MZ * log_ratio;
    if denom <= 0.0 {
        return 1.0; // Landau pole — return maximum (non-perturbative)
    }

    ALPHA_S_MZ / denom
}

/// Running electromagnetic coupling α_EM(Q²) at 1-loop.
///
/// α(Q²) = α(0) / [1 - (α(0)/(3π)) Σ_f N_c Q_f² ln(Q²/m_f²)]
pub fn alpha_em_running(q2: f64) -> f64 {
    if q2 <= M_ELECTRON * M_ELECTRON {
        return ALPHA_EM;
    }

    // Sum over charged fermions lighter than Q
    let q = q2.sqrt();
    let mut delta = 0.0;

    // Leptons
    if q > M_ELECTRON {
        delta += (q2 / (M_ELECTRON * M_ELECTRON)).ln();
    }
    if q > M_MUON {
        delta += (q2 / (M_MUON * M_MUON)).ln();
    }
    if q > M_TAU {
        delta += (q2 / (M_TAU * M_TAU)).ln();
    }

    // Quarks (with N_c and charge factors)
    let quarks: &[(f64, f64)] = &[
        (M_UP, 2.0 / 3.0),
        (M_DOWN, 1.0 / 3.0),
        (M_STRANGE, 1.0 / 3.0),
        (M_CHARM, 2.0 / 3.0),
        (M_BOTTOM, 1.0 / 3.0),
        (M_TOP, 2.0 / 3.0),
    ];

    for &(mq, charge) in quarks {
        if q > mq {
            delta += N_C * charge * charge * (q2 / (mq * mq)).ln();
        }
    }

    let correction = ALPHA_EM * delta / (3.0 * PI);
    ALPHA_EM / (1.0 - correction)
}

/// Count active quark flavors at energy scale Q (GeV).
fn count_active_flavors(q: f64) -> u8 {
    let mut nf = 0u8;
    if q > M_UP { nf += 1; }
    if q > M_DOWN { nf += 1; }
    if q > M_STRANGE { nf += 1; }
    if q > M_CHARM { nf += 1; }
    if q > M_BOTTOM { nf += 1; }
    if q > M_TOP { nf += 1; }
    nf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigma_ee_mumu_high_energy() {
        // At √s = 10 GeV (well above muon threshold):
        // σ = 4πα²/(3s) ≈ 0.87 nb at √s = 10 GeV
        let s = 100.0; // GeV²
        let sigma = sigma_ee_to_mumu(s);
        let sigma_nb = sigma * GEV2_TO_NB;
        assert!(
            (sigma_nb - 0.87).abs() < 0.1,
            "σ(e⁺e⁻→μ⁺μ⁻) at √s=10 GeV = {:.3} nb, expected ≈ 0.87 nb",
            sigma_nb
        );
    }

    #[test]
    fn test_sigma_ee_mumu_threshold() {
        // Below threshold: σ = 0
        let s = 4.0 * M_MUON * M_MUON * 0.99;
        assert_eq!(sigma_ee_to_mumu(s), 0.0);
    }

    #[test]
    fn test_z_resonance_enhancement() {
        // At the Z pole, cross-section should be much larger
        let s_off = 50.0 * 50.0; // √s = 50 GeV (off-pole)
        let s_on = M_Z * M_Z; // √s = M_Z (on-pole)

        let sigma_off = sigma_ee_to_mumu_with_z(s_off);
        let sigma_on = sigma_ee_to_mumu_with_z(s_on);

        assert!(
            sigma_on > 10.0 * sigma_off,
            "Z resonance should enhance: on-pole={:.2e}, off={:.2e}",
            sigma_on,
            sigma_off
        );
    }

    #[test]
    fn test_r_ratio_below_charm() {
        // Below charm threshold: R = N_c × (Q_u² + Q_d² + Q_s²) = 3×(4/9+1/9+1/9) = 2
        let r = r_ratio(1.5); // √s = 1.5 GeV (below charm)
        // R₀ = 2.0 × (1 + α_s/π) ≈ 2.0 × 1.1 = 2.2 with QCD correction
        assert!(
            (r - 2.0).abs() < 0.4,
            "R below charm = {:.3}, expected ≈ 2.0-2.2 (with QCD correction)",
            r
        );
    }

    #[test]
    fn test_r_ratio_above_charm() {
        // Above charm: R adds N_c × Q_c² = 3 × 4/9 = 4/3 → R ≈ 10/3
        let r = r_ratio(5.0); // √s = 5 GeV (above charm, below bottom)
        assert!(
            (r - 10.0 / 3.0).abs() < 0.5,
            "R above charm = {:.3}, expected ≈ 3.33",
            r
        );
    }

    #[test]
    fn test_alpha_s_asymptotic_freedom() {
        // α_s should decrease with increasing Q²
        let alpha_low = alpha_s_running(4.0); // Q = 2 GeV
        let alpha_high = alpha_s_running(10000.0); // Q = 100 GeV

        assert!(
            alpha_low > alpha_high,
            "Asymptotic freedom: α_s(4)={:.4} should > α_s(10000)={:.4}",
            alpha_low,
            alpha_high
        );
    }

    #[test]
    fn test_alpha_s_at_mz() {
        // At M_Z, should return approximately the input value
        let alpha = alpha_s_running(M_Z * M_Z);
        assert!(
            (alpha - ALPHA_S_MZ).abs() < 0.001,
            "α_s(M_Z²) = {:.4}, expected {:.4}",
            alpha,
            ALPHA_S_MZ
        );
    }

    #[test]
    fn test_compton_thomson_limit() {
        // Low energy: Klein-Nishina → Thomson
        let sigma_thomson = 8.0 * PI * ALPHA_EM * ALPHA_EM / (3.0 * M_ELECTRON * M_ELECTRON);
        let sigma_low = sigma_compton(0.001 * M_ELECTRON); // ω << m_e

        assert!(
            (sigma_low / sigma_thomson - 1.0).abs() < 0.05,
            "Thomson limit: σ_KN/σ_T = {:.4}",
            sigma_low / sigma_thomson
        );
    }

    #[test]
    fn test_mandelstam_sum() {
        // s + t + u = 0 for massless particles
        let m = Mandelstam::from_s_cos_theta(100.0, 0.5);
        assert!(m.check_sum().abs() < 1e-10);
    }
}
