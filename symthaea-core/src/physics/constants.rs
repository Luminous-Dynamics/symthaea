// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Physical Constants
//!
//! Standard physical constants used throughout the physics modules.

/// Speed of light in vacuum (m/s)
pub const C: f64 = 299_792_458.0;

/// Gravitational constant (m³/(kg·s²))
pub const G: f64 = 6.674_30e-11;

/// Planck constant (J·s)
pub const H: f64 = 6.626_070_15e-34;

/// Reduced Planck constant (J·s)
pub const HBAR: f64 = 1.054_571_817e-34;

/// Boltzmann constant (J/K)
pub const K_BOLTZMANN: f64 = 1.380_649e-23;

/// Elementary charge (C)
pub const E_CHARGE: f64 = 1.602_176_634e-19;

/// Electron mass (kg)
pub const M_ELECTRON: f64 = 9.109_383_701_5e-31;

/// Proton mass (kg)
pub const M_PROTON: f64 = 1.672_621_923_69e-27;

/// Vacuum permittivity (F/m)
pub const EPSILON_0: f64 = 8.854_187_812_8e-12;

/// Vacuum permeability (H/m)
pub const MU_0: f64 = 1.256_637_062_12e-6;

/// Gas constant (J/(mol·K))
pub const R_GAS: f64 = 8.314_462_618;

/// Avogadro constant (1/mol)
pub const N_AVOGADRO: f64 = 6.022_140_76e23;

/// Room temperature (K) - standard 25°C
pub const T_ROOM: f64 = 298.15;

/// Standard atmosphere (Pa)
pub const P_STANDARD: f64 = 101_325.0;

/// Fine structure constant (dimensionless)
pub const ALPHA: f64 = 7.297_352_569_3e-3;

/// Coulomb constant k_e = 1/(4πε₀) (N·m²/C²)
pub const K_COULOMB: f64 = 8.987_551_792_3e9;

/// Atomic mass unit (kg)
pub const AMU: f64 = 1.660_539_066_60e-27;

/// MeV to Joules conversion factor (E_CHARGE × 1e6)
pub const MEV_TO_J: f64 = 1.602_176_634e-13;

/// Faraday constant (C/mol) — N_A × e
pub const F_FARADAY: f64 = 96_485.332_12;

/// Rydberg energy (eV) — ground-state hydrogen ionization
pub const RYDBERG_EV: f64 = 13.605_693_122_994;

// ── Nuclear & Particle Physics Constants (CODATA 2022 / PDG 2024) ──────────

/// Neutron mass (kg)
pub const M_NEUTRON: f64 = 1.674_927_498_04e-27;

/// Neutron mass (MeV/c²)
pub const M_NEUTRON_MEV: f64 = 939.565_420_52;

/// Proton mass (MeV/c²)
pub const M_PROTON_MEV: f64 = 938.272_088_16;

/// Electron mass (MeV/c²)
pub const M_ELECTRON_MEV: f64 = 0.510_998_950_00;

/// Charged pion mass (MeV/c²) — π±
pub const M_PION_CHARGED: f64 = 139.570_39;

/// Neutral pion mass (MeV/c²) — π⁰
pub const M_PION_NEUTRAL: f64 = 134.976_8;

/// Nuclear magneton (J/T) — eℏ/(2m_p)
pub const MU_NUCLEAR: f64 = 5.050_783_746_1e-27;

/// Bohr magneton (J/T) — eℏ/(2m_e)
pub const MU_BOHR: f64 = 9.274_010_078_3e-24;

/// Bohr radius (m) — ℏ/(m_e·c·α)
pub const A_BOHR: f64 = 5.291_772_109_03e-11;

/// Classical electron radius (m) — α²·a₀
pub const R_ELECTRON: f64 = 2.817_940_326_2e-15;

/// Compton wavelength of electron (m) — ℏ/(m_e·c)
pub const LAMBDA_COMPTON: f64 = 2.426_310_238_67e-12;

/// Stefan-Boltzmann constant (W/(m²·K⁴))
pub const SIGMA_SB: f64 = 5.670_374_419e-8;

/// ℏc product (MeV·fm) — useful for nuclear physics range calculations
pub const HBAR_C_MEV_FM: f64 = 197.326_980_4;

// ── Electroweak & Standard Model Constants (PDG 2024) ──────────────────────

/// W boson mass (MeV/c²)
pub const M_W_BOSON: f64 = 80_369.2;

/// Z boson mass (MeV/c²)
pub const M_Z_BOSON: f64 = 91_187.6;

/// Higgs boson mass (MeV/c²)
pub const M_HIGGS: f64 = 125_250.0;

/// Fermi coupling constant (GeV⁻²)
pub const G_FERMI: f64 = 1.166_378_8e-5;

/// Weak mixing angle sin²(θ_W) (MS-bar, at M_Z)
pub const SIN2_THETA_W: f64 = 0.231_22;

/// Strong coupling constant α_s(M_Z) (MS-bar)
pub const ALPHA_S_MZ: f64 = 0.1179;

/// Inverse fine structure constant 1/α
pub const ALPHA_INV: f64 = 137.035_999_084;

// ── Nuclear Structure Constants ────────────────────────────────────────────

/// Yukawa range for pion exchange (fm) — ℏc/m_π±
pub const YUKAWA_RANGE_FM: f64 = 1.413;

/// Pion-nucleon coupling constant g²_πNN/(4π) (dimensionless)
pub const G_PINN_SQ_OVER_4PI: f64 = 14.0;

/// Pseudovector pion-nucleon coupling f²_πNN/(4π)
pub const F_PINN_SQ_OVER_4PI: f64 = 0.08;

/// Nuclear radius parameter r₀ (fm) — R = r₀·A^(1/3)
pub const R0_NUCLEAR_FM: f64 = 1.25;

// ── Antimatter & Annihilation ──────────────────────────────────────────────

/// Proton-antiproton annihilation energy (MeV) — 2·m_p·c²
pub const PP_ANNIHILATION_ENERGY_MEV: f64 = 1876.544;

/// Average pion multiplicity in p-pbar annihilation
pub const PP_AVG_PION_MULTIPLICITY: f64 = 5.0;

/// Fraction of annihilation energy in charged pions (directable)
pub const CHARGED_PION_ENERGY_FRACTION: f64 = 0.40;

// ── Useful Derived Constants ───────────────────────────────────────────────

/// c² (m²/s²) — mass-energy conversion
pub const C_SQUARED: f64 = C * C;

/// Planck energy (J) — √(ℏc⁵/G)
pub const E_PLANCK: f64 = 1.956_1e9;

/// Solar mass-energy (J) — M_☉·c²
pub const E_SOLAR_MASS: f64 = 1.787e47;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fundamental_constant_consistency() {
        // ℏ = h/(2π)
        let hbar_computed = H / (2.0 * std::f64::consts::PI);
        assert!((HBAR - hbar_computed).abs() / HBAR < 1e-9);

        // k_e = 1/(4πε₀)
        let ke_computed = 1.0 / (4.0 * std::f64::consts::PI * EPSILON_0);
        assert!((K_COULOMB - ke_computed).abs() / K_COULOMB < 1e-6);

        // α = e²/(4πε₀ℏc)
        let alpha_computed =
            E_CHARGE * E_CHARGE / (4.0 * std::f64::consts::PI * EPSILON_0 * HBAR * C);
        assert!((ALPHA - alpha_computed).abs() / ALPHA < 1e-6);
    }

    #[test]
    fn test_nuclear_constants_consistency() {
        // ℏc in MeV·fm should equal ℏ(J·s) * c(m/s) / MEV_TO_J / 1e-15
        let hbar_c_computed = HBAR * C / MEV_TO_J * 1e15;
        assert!(
            (HBAR_C_MEV_FM - hbar_c_computed).abs() / HBAR_C_MEV_FM < 1e-4,
            "ℏc: expected {}, got {}",
            HBAR_C_MEV_FM,
            hbar_c_computed
        );

        // Yukawa range = ℏc / m_π
        let yukawa_computed = HBAR_C_MEV_FM / M_PION_CHARGED;
        assert!(
            (YUKAWA_RANGE_FM - yukawa_computed).abs() / YUKAWA_RANGE_FM < 0.01,
            "Yukawa range: expected {}, got {}",
            YUKAWA_RANGE_FM,
            yukawa_computed
        );
    }

    #[test]
    fn test_mass_ordering() {
        // Masses should satisfy: electron < pion < proton < neutron
        assert!(M_ELECTRON_MEV < M_PION_NEUTRAL);
        assert!(M_PION_NEUTRAL < M_PION_CHARGED);
        assert!(M_PION_CHARGED < M_PROTON_MEV);
        assert!(M_PROTON_MEV < M_NEUTRON_MEV);

        // Boson mass ordering: W < Z < Higgs
        assert!(M_W_BOSON < M_Z_BOSON);
        assert!(M_Z_BOSON < M_HIGGS);
    }

    #[test]
    fn test_annihilation_energy() {
        // p + pbar annihilation = 2 * m_p * c²
        let computed = 2.0 * M_PROTON_MEV;
        assert!(
            (PP_ANNIHILATION_ENERGY_MEV - computed).abs() < 1.0,
            "Annihilation energy: expected {}, got {}",
            PP_ANNIHILATION_ENERGY_MEV,
            computed
        );
    }

    #[test]
    fn test_electroweak_consistency() {
        // sin²θ_W ≈ 1 - (M_W/M_Z)² at tree level
        let sin2_tree = 1.0 - (M_W_BOSON / M_Z_BOSON).powi(2);
        // Tree-level differs from MS-bar by ~2%, but should be in the right ballpark
        assert!(
            (SIN2_THETA_W - sin2_tree).abs() < 0.02,
            "Weinberg angle tree-level check: sin²θ_W={}, tree={}",
            SIN2_THETA_W,
            sin2_tree
        );
    }
}
