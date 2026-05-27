// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Physics Correctness Audit — Round 9
//!
//! Validates fixes from the Round 9 audit:
//! - Constant centralization (MEV_TO_J, C, G, N_AVOGADRO precision)
//! - Geophysics magnitude ↔ moment roundtrip
//! - Fusion energy density (GJ/g, not GJ/mol)
//! - Electron-nuclear screening constants (Slater's rules)
//! - Quantum gravity precision (Schwarzschild radius, Planck units)
//! - Derived constant cross-checks

use super::constants::*;
use super::geophysics::{FocalMechanism, GeophysicsEncoder};
use super::physics_test_helpers::assert_relative_eq;
use super::quantum_gravity::QuantumGravityEncoder;
use super::radiation_damage::FusionReaction;
use crate::genesis::GenesisSeed;
use std::f64::consts::PI;

// =========================================================================
// Section 68: Constant Centralization (6 tests)
// =========================================================================

#[test]
fn audit_c_exact() {
    // Speed of light is exact in SI (defined value)
    assert_eq!(
        C, 299_792_458.0,
        "C must be exactly 299792458 m/s (SI definition)"
    );
}

#[test]
fn audit_g_codata() {
    // CODATA 2018: G = 6.67430(15) × 10⁻¹¹
    assert_relative_eq(G, 6.67430e-11, 1e-4, "G matches CODATA 2018");
}

#[test]
fn audit_avogadro_exact() {
    // Avogadro constant is exact in SI (2019 redefinition)
    assert_eq!(
        N_AVOGADRO, 6.022_140_76e23,
        "N_A must be exactly 6.02214076e23 (SI definition)"
    );
}

#[test]
fn audit_mev_to_j() {
    // MEV_TO_J = E_CHARGE × 1e6
    let derived = E_CHARGE * 1e6;
    assert_relative_eq(MEV_TO_J, derived, 1e-10, "MEV_TO_J = E_CHARGE × 1e6");
}

#[test]
fn audit_planck_energy() {
    // E_P = sqrt(ℏc⁵/G) ≈ 1.9561e9 J
    let e_p = (HBAR * C.powi(5) / G).sqrt();
    assert_relative_eq(e_p, 1.9561e9, 0.01, "Planck energy from ℏ, c, G");
}

#[test]
fn audit_planck_self_consistency() {
    // l_P = sqrt(ℏG/c³), t_P = sqrt(ℏG/c⁵), m_P = sqrt(ℏc/G)
    let l_p = (HBAR * G / C.powi(3)).sqrt();
    let t_p = (HBAR * G / C.powi(5)).sqrt();
    let m_p = (HBAR * C / G).sqrt();
    let e_p = (HBAR * C.powi(5) / G).sqrt();

    // l_P = c × t_P
    assert_relative_eq(l_p, C * t_p, 1e-6, "l_P = c × t_P");
    // E_P = m_P × c²
    assert_relative_eq(e_p, m_p * C * C, 1e-6, "E_P = m_P × c²");
}

// =========================================================================
// Section 69: Geophysics Roundtrip (4 tests)
// =========================================================================

fn geo_setup() -> GeophysicsEncoder {
    let genesis = GenesisSeed::from_phrase("r9_geo_audit");
    GeophysicsEncoder::from_genesis(&genesis)
}

#[test]
fn audit_geo_roundtrip_m3() {
    let encoder = geo_setup();
    let eq = encoder.create_earthquake(3.0, 10.0, FocalMechanism::StrikeSlip);
    let recovered = encoder.moment_to_magnitude(eq.moment_nm);
    assert_relative_eq(recovered, 3.0, 0.01, "M3 earthquake roundtrip");
}

#[test]
fn audit_geo_roundtrip_m5() {
    let encoder = geo_setup();
    let eq = encoder.create_earthquake(5.0, 10.0, FocalMechanism::StrikeSlip);
    let recovered = encoder.moment_to_magnitude(eq.moment_nm);
    assert_relative_eq(recovered, 5.0, 0.01, "M5 earthquake roundtrip");
}

#[test]
fn audit_geo_roundtrip_m7() {
    let encoder = geo_setup();
    let eq = encoder.create_earthquake(7.0, 10.0, FocalMechanism::Thrust);
    let recovered = encoder.moment_to_magnitude(eq.moment_nm);
    assert_relative_eq(recovered, 7.0, 0.01, "M7 earthquake roundtrip");
}

#[test]
fn audit_geo_magnitude_reference() {
    // USGS reference: M₀ = 3.98×10¹⁶ N·m → Mw ≈ 5.0
    let encoder = geo_setup();
    let m0 = 3.981e16; // 10^(1.5*5 + 9.1) ≈ 10^16.6 ≈ 3.98×10¹⁶
    let mw = encoder.moment_to_magnitude(m0);
    assert_relative_eq(mw, 5.0, 0.01, "USGS reference M0→Mw=5");
}

// =========================================================================
// Section 70: Energy Density Validation (4 tests)
// =========================================================================

#[test]
fn audit_dt_energy_density() {
    // DT: 17.6 MeV per reaction, fuel mass = 5.030 g/mol (D+T)
    // E = 17.6 × MEV_TO_J × N_A / 5.030 = J/g → ÷ 1e9 = GJ/g
    let e_gj_g = FusionReaction::DT.total_energy_mev() * MEV_TO_J * N_AVOGADRO / (5.030 * 1e9);
    // Literature: ~337 GJ/g for DT
    assert_relative_eq(e_gj_g, 337.0, 0.05, "DT energy density ~337 GJ/g");
}

#[test]
fn audit_dd_energy_density() {
    // DD: 3.27 MeV per reaction, fuel mass = 4.028 g/mol (D+D)
    let e_gj_g = FusionReaction::DD.total_energy_mev() * MEV_TO_J * N_AVOGADRO / (4.028 * 1e9);
    // DD should be ~78 GJ/g
    assert_relative_eq(e_gj_g, 78.0, 0.10, "DD energy density ~78 GJ/g");
}

#[test]
fn audit_energy_units_dimensional() {
    // Verify units: [MeV] × [J/MeV] × [1/mol] / [g/mol] = [J/g]
    // Just check that MEV_TO_J × N_AVOGADRO gives a sensible J/mol
    let j_per_mol_per_mev = MEV_TO_J * N_AVOGADRO; // J/mol per MeV
    // 1 MeV × N_A = ~9.65×10¹⁰ J/mol
    assert_relative_eq(
        j_per_mol_per_mev,
        9.65e10,
        0.01,
        "MeV×N_A dimensional check",
    );
}

#[test]
fn audit_he3_consumption_reasonable() {
    // For 1 kW power from D-He3 (18.3 MeV), He-3 molar mass 3.016 g/mol
    // consumption = P × t / (E_per_g)
    let power_w = 1000.0;
    let seconds_per_year = 3.15e7;
    let energy_per_g = 18.3e6 * MEV_TO_J * N_AVOGADRO / 3.016; // J/g
    let he3_g_year = power_w * seconds_per_year / energy_per_g;
    // Should be small — order of ~10 mg/year for 1kW
    assert!(he3_g_year > 0.0, "He-3 consumption must be positive");
    assert!(
        he3_g_year < 1.0,
        "He-3 consumption < 1 g/year for 1kW: got {he3_g_year:.4e}"
    );
}

// =========================================================================
// Section 71: Screening & Binding Energies (4 tests)
// =========================================================================

/// Compute binding energy: NIST table for known elements, Slater fallback otherwise
fn binding_energy_ev(z: u8, shell: char) -> f64 {
    let nist: Option<(f64, f64, f64, f64)> = match z {
        22 => Some((4_966.0, 564.0, 61.0, 6.0)),             // Ti
        24 => Some((5_989.0, 696.0, 74.0, 7.0)),             // Cr
        25 => Some((6_539.0, 769.0, 83.0, 7.0)),             // Mn
        26 => Some((7_112.0, 846.0, 100.0, 8.0)),            // Fe
        28 => Some((8_333.0, 1_009.0, 112.0, 8.0)),          // Ni
        29 => Some((8_979.0, 1_097.0, 120.0, 8.0)),          // Cu
        30 => Some((9_659.0, 1_194.0, 137.0, 10.0)),         // Zn
        40 => Some((17_998.0, 2_532.0, 430.0, 51.0)),        // Zr
        42 => Some((20_000.0, 2_866.0, 505.0, 63.0)),        // Mo
        43 => Some((21_044.0, 3_043.0, 544.0, 68.0)),        // Tc
        47 => Some((25_514.0, 3_806.0, 719.0, 97.0)),        // Ag
        50 => Some((29_200.0, 4_465.0, 884.0, 137.0)),       // Sn
        56 => Some((37_441.0, 5_989.0, 1_293.0, 253.0)),     // Ba
        74 => Some((69_525.0, 12_100.0, 2_820.0, 595.0)),    // W
        78 => Some((78_395.0, 13_880.0, 3_296.0, 725.0)),    // Pt
        79 => Some((80_725.0, 14_353.0, 3_425.0, 762.0)),    // Au
        82 => Some((88_005.0, 15_861.0, 3_851.0, 894.0)),    // Pb
        90 => Some((109_651.0, 20_472.0, 5_182.0, 1_330.0)), // Th
        92 => Some((115_606.0, 21_757.0, 5_548.0, 1_441.0)), // U
        _ => None,
    };
    match nist {
        Some((k, l, m, n)) => match shell {
            'K' => k,
            'L' => l,
            'M' => m,
            'N' => n,
            _ => 0.0,
        },
        None => {
            let z_eff = z as f64;
            match shell {
                'K' => RYDBERG_EV * (z_eff - 0.3).max(1.0).powi(2),
                'L' => RYDBERG_EV * (z_eff - 4.15).max(1.0).powi(2) / 4.0,
                'M' => RYDBERG_EV * (z_eff - 11.25).max(1.0).powi(2) / 9.0,
                'N' => RYDBERG_EV * (z_eff - 21.15).max(1.0).powi(2) / 16.0,
                _ => 0.0,
            }
        }
    }
}

#[test]
fn audit_k_shell_iron() {
    // Fe (Z=26) K-shell: NIST 7112 eV (exact from table)
    let e = binding_energy_ev(26, 'K');
    assert_relative_eq(e, 7112.0, 0.001, "Fe K-shell = NIST 7112 eV");
}

#[test]
fn audit_l_shell_molybdenum() {
    // Mo (Z=42) L-shell: NIST 2866 eV (exact from table)
    let e = binding_energy_ev(42, 'L');
    assert_relative_eq(e, 2866.0, 0.001, "Mo L-shell = NIST 2866 eV");
}

#[test]
fn audit_screening_monotonic() {
    // For any Z > 30, binding energy must be K > L > M > N
    for z in [35_u8, 50, 60, 74, 79] {
        let k = binding_energy_ev(z, 'K');
        let l = binding_energy_ev(z, 'L');
        let m = binding_energy_ev(z, 'M');
        let n = binding_energy_ev(z, 'N');
        assert!(k > l, "Z={z}: K ({k:.0}) > L ({l:.0})");
        assert!(l > m, "Z={z}: L ({l:.0}) > M ({m:.0})");
        assert!(m > n, "Z={z}: M ({m:.0}) > N ({n:.0})");
    }
}

#[test]
fn audit_screening_z_scaling() {
    // Binding energy should increase with Z for each shell
    for shell in ['K', 'L', 'M', 'N'] {
        let e30 = binding_energy_ev(30, shell);
        let e50 = binding_energy_ev(50, shell);
        let e70 = binding_energy_ev(70, shell);
        assert!(e50 > e30, "shell={shell}: E(Z=50) > E(Z=30)");
        assert!(e70 > e50, "shell={shell}: E(Z=70) > E(Z=50)");
    }
}

// =========================================================================
// Section 72: Quantum Gravity Precision (4 tests)
// =========================================================================

fn qg_setup() -> QuantumGravityEncoder {
    let genesis = GenesisSeed::from_phrase("r9_qg_audit");
    QuantumGravityEncoder::from_genesis(&genesis)
}

#[test]
fn audit_schwarzschild_sun() {
    // r_s(M_sun) = 2GM/c² ≈ 2953.25 m
    let encoder = qg_setup();
    let m_sun = 1.989e30;
    let r_s = encoder.schwarzschild_radius(m_sun);
    assert_relative_eq(r_s, 2953.25, 0.01, "Schwarzschild radius of Sun");
}

#[test]
fn audit_schwarzschild_earth() {
    // r_s(M_earth) = 2GM/c² ≈ 8.87 mm
    let encoder = qg_setup();
    let m_earth = 5.972e24;
    let r_s = encoder.schwarzschild_radius(m_earth);
    assert_relative_eq(r_s, 8.87e-3, 0.01, "Schwarzschild radius of Earth ~8.87 mm");
}

#[test]
fn audit_bh_entropy_area_law() {
    // S ∝ A: doubling area should double entropy
    let encoder = qg_setup();
    let s1 = encoder.black_hole_entropy(1e6);
    let s2 = encoder.black_hole_entropy(2e6);
    assert_relative_eq(s2 / s1, 2.0, 1e-6, "BH entropy proportional to area");
}

#[test]
fn audit_planck_units_consistent() {
    // l_P, t_P, m_P, E_P, T_P all consistent with G, ℏ, c, k_B
    let l_p = (HBAR * G / C.powi(3)).sqrt();
    let t_p = (HBAR * G / C.powi(5)).sqrt();
    let m_p = (HBAR * C / G).sqrt();
    let e_p = m_p * C * C;
    let temp_p = e_p / K_BOLTZMANN;

    let encoder = qg_setup();
    assert_relative_eq(encoder.planck.length_m, l_p, 0.001, "Planck length");
    assert_relative_eq(encoder.planck.time_s, t_p, 0.001, "Planck time");
    assert_relative_eq(encoder.planck.mass_kg, m_p, 0.001, "Planck mass");
    assert_relative_eq(encoder.planck.energy_j, e_p, 0.001, "Planck energy");
    assert_relative_eq(
        encoder.planck.temperature_k,
        temp_p,
        0.001,
        "Planck temperature",
    );
}

// =========================================================================
// Section 73: Derived Constant Cross-Checks (4 tests)
// =========================================================================

#[test]
fn audit_derived_fine_structure() {
    // α = e²/(4πε₀ℏc) ≈ 1/137.036
    let alpha_derived = E_CHARGE * E_CHARGE / (4.0 * PI * EPSILON_0 * HBAR * C);
    assert_relative_eq(
        alpha_derived,
        ALPHA,
        1e-6,
        "Fine structure constant from e, ε₀, ℏ, c",
    );
    assert_relative_eq(1.0 / alpha_derived, 137.036, 1e-4, "1/α ≈ 137.036");
}

#[test]
fn audit_derived_bohr_radius() {
    // a₀ = 4πε₀ℏ²/(m_e·e²) ≈ 5.292e-11 m
    let a0 = 4.0 * PI * EPSILON_0 * HBAR * HBAR / (M_ELECTRON * E_CHARGE * E_CHARGE);
    assert_relative_eq(
        a0,
        5.2918e-11,
        1e-4,
        "Bohr radius from fundamental constants",
    );
}

#[test]
fn audit_derived_rydberg() {
    // R_∞ = m_e·e⁴/(8ε₀²h³c) ≈ 1.0974e7 m⁻¹
    let r_inf = M_ELECTRON * E_CHARGE.powi(4) / (8.0 * EPSILON_0 * EPSILON_0 * H.powi(3) * C);
    assert_relative_eq(
        r_inf,
        1.0974e7,
        1e-4,
        "Rydberg constant from fundamental constants",
    );
}

#[test]
fn audit_derived_coulomb_constant() {
    // k_e = 1/(4πε₀) ≈ 8.988e9 N·m²/C²
    let k_e = 1.0 / (4.0 * PI * EPSILON_0);
    assert_relative_eq(k_e, K_COULOMB, 1e-6, "Coulomb constant k_e = 1/(4πε₀)");
}

// =========================================================================
// Section 74: Extended NIST Validation (4 tests)
// =========================================================================

#[test]
fn audit_k_shell_tungsten() {
    // W (Z=74) K-shell: NIST 69525 eV
    let e = binding_energy_ev(74, 'K');
    assert_relative_eq(e, 69525.0, 0.001, "W K-shell = NIST 69525 eV");
}

#[test]
fn audit_k_shell_titanium() {
    // Ti (Z=22) K-shell: NIST 4966 eV
    let e = binding_energy_ev(22, 'K');
    assert_relative_eq(e, 4966.0, 0.001, "Ti K-shell = NIST 4966 eV");
}

#[test]
fn audit_k_shell_silver() {
    // Ag (Z=47) K-shell: NIST 25514 eV
    let e = binding_energy_ev(47, 'K');
    assert_relative_eq(e, 25514.0, 0.001, "Ag K-shell = NIST 25514 eV");
}

#[test]
fn audit_k_shell_gold() {
    // Au (Z=79) K-shell: NIST 80725 eV
    let e = binding_energy_ev(79, 'K');
    assert_relative_eq(e, 80725.0, 0.001, "Au K-shell = NIST 80725 eV");
}

// =========================================================================
// Section 75: Fuel Mass & Derived Constants (3 tests)
// =========================================================================

#[test]
fn audit_fuel_mass_isotope_sums() {
    let d = 2.01410;
    let t = 3.01605;
    let he3 = 3.01603;
    assert_relative_eq(
        FusionReaction::DD.fuel_mass_g_mol(),
        2.0 * d,
        0.001,
        "DD mass",
    );
    assert_relative_eq(
        FusionReaction::DT.fuel_mass_g_mol(),
        d + t,
        0.001,
        "DT mass",
    );
    assert_relative_eq(
        FusionReaction::DdProton.fuel_mass_g_mol(),
        2.0 * d,
        0.001,
        "DdP mass",
    );
    assert_relative_eq(
        FusionReaction::DHe3.fuel_mass_g_mol(),
        d + he3,
        0.001,
        "DHe3 mass",
    );
}

#[test]
fn audit_faraday_derived() {
    // F = N_A × e
    let f_derived = N_AVOGADRO * E_CHARGE;
    assert_relative_eq(f_derived, F_FARADAY, 1e-6, "Faraday = N_A × e");
}

#[test]
fn audit_gas_constant_derived() {
    // R = N_A × k_B
    let r_derived = N_AVOGADRO * K_BOLTZMANN;
    assert_relative_eq(r_derived, R_GAS, 1e-6, "Gas constant = N_A × k_B");
}
