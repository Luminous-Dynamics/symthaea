// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Particle physics constants (PDG 2024 values).
//!
//! Natural units: ℏ = c = 1. Energy in GeV, length in GeV⁻¹.

use std::f64::consts::PI;

// ── Coupling Constants ──────────────────────────────────────────────────────

/// Fine structure constant α = e²/(4π) ≈ 1/137.036
pub const ALPHA_EM: f64 = 1.0 / 137.035_999_084;

/// Strong coupling at M_Z: α_s(M_Z) = 0.1179
pub const ALPHA_S_MZ: f64 = 0.1179;

/// Fermi constant G_F in GeV⁻² (muon decay)
pub const G_FERMI: f64 = 1.166_378_7e-5;

/// Weinberg angle: sin²θ_W = 0.23122
pub const SIN2_THETA_W: f64 = 0.231_22;

/// cos²θ_W
pub const COS2_THETA_W: f64 = 1.0 - SIN2_THETA_W;

// ── Particle Masses (GeV/c²) ────────────────────────────────────────────────

/// Electron mass
pub const M_ELECTRON: f64 = 0.000_510_998_950;

/// Muon mass
pub const M_MUON: f64 = 0.105_658_375_5;

/// Tau mass
pub const M_TAU: f64 = 1.776_86;

/// Up quark mass (MS-bar at 2 GeV)
pub const M_UP: f64 = 0.002_16;

/// Down quark mass (MS-bar at 2 GeV)
pub const M_DOWN: f64 = 0.004_67;

/// Strange quark mass (MS-bar at 2 GeV)
pub const M_STRANGE: f64 = 0.093_4;

/// Charm quark mass (MS-bar at m_c)
pub const M_CHARM: f64 = 1.27;

/// Bottom quark mass (MS-bar at m_b)
pub const M_BOTTOM: f64 = 4.18;

/// Top quark mass (pole)
pub const M_TOP: f64 = 172.69;

/// W boson mass
pub const M_W: f64 = 80.377;

/// Z boson mass
pub const M_Z: f64 = 91.1876;

/// Higgs boson mass
pub const M_HIGGS: f64 = 125.25;

/// Proton mass
pub const M_PROTON: f64 = 0.938_272_088_16;

/// Pion (charged) mass
pub const M_PION_CHARGED: f64 = 0.139_570_39;

/// Pion (neutral) mass
pub const M_PION_NEUTRAL: f64 = 0.134_977_0;

// ── Particle Widths (GeV) ───────────────────────────────────────────────────

/// W boson total width
pub const GAMMA_W: f64 = 2.085;

/// Z boson total width
pub const GAMMA_Z: f64 = 2.4952;

/// Top quark width
pub const GAMMA_TOP: f64 = 1.42;

/// Higgs width
pub const GAMMA_HIGGS: f64 = 0.003_2;

// ── Derived Constants ───────────────────────────────────────────────────────

/// Conversion: GeV⁻¹ → seconds (ℏ in GeV·s)
pub const HBAR_GEV_S: f64 = 6.582_119_569e-25;

/// Conversion: GeV⁻² → pb (picobarns)
pub const GEV2_TO_PB: f64 = 3.893_79e8;

/// Conversion: GeV⁻² → nb (nanobarns)
pub const GEV2_TO_NB: f64 = 3.893_79e5;

/// π constant
pub const PI_CONST: f64 = PI;

// ── Color Factors ───────────────────────────────────────────────────────────

/// Number of colors N_c
pub const N_C: f64 = 3.0;

/// Color factor C_F = (N_c² - 1)/(2*N_c) = 4/3
pub const C_F: f64 = 4.0 / 3.0;

/// Color factor C_A = N_c = 3
pub const C_A: f64 = 3.0;

/// Color factor T_F = 1/2
pub const T_F: f64 = 0.5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weinberg_angle_consistency() {
        assert!((SIN2_THETA_W + COS2_THETA_W - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_color_factor_cf() {
        // C_F = (N_c² - 1) / (2*N_c)
        let cf = (N_C * N_C - 1.0) / (2.0 * N_C);
        assert!((cf - C_F).abs() < 1e-10);
    }
}
