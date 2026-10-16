// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Independent reaction-energy fixtures for the FUS-ECON-001 audit.
//!
//! IMPORTANT: the first test verifies ideal stoichiometric arithmetic only.
//! It does not qualify a plant fuel-cycle model.
//!
//! The characterization test records the mismatch currently present in the
//! legacy FuelCosts defaults. It is intentionally an audit witness, not a
//! claim that those defaults are correct. When the production API declares
//! thermal/gross-electric/net-electric basis and the defaults are reconciled,
//! replace this characterization assertion with basis-specific conformance
//! tests in the same change that repairs the model.
//!
//! References:
//! - IAEA, World Fusion Outlook 2023, Table 1 (reaction energies):
//!   https://www-pub.iaea.org/MTCD/Publications/PDF/FusionOutlook2023_web.pdf
//! - NIST, atomic weights and isotopic compositions:
//!   https://pml.nist.gov/cgi-bin/Compositions/stand_alone.pl

use symthaea_core::physics::FuelCosts;

// Exact SI conversion constants.
const JOULES_PER_MWH: f64 = 3.6e9;
const JOULES_PER_EV: f64 = 1.602_176_634e-19;
const KG_PER_U: f64 = 1.660_539_066_60e-27;

// NIST isotopic masses in unified atomic mass units (u).
const DEUTERIUM_U: f64 = 2.014_101_778_12;
const TRITIUM_U: f64 = 3.016_049_281_99;
const HELIUM3_U: f64 = 3.016_029_322_65;

/// Independently derive ideal reactant mass in mg for the supplied amount of
/// fusion reaction energy. nuclei_per_reaction is the stoichiometric count of
/// the requested isotope consumed by each reaction.
///
/// This function intentionally does not call any Symthaea production helper.
fn ideal_reactant_mg(
    energy_mwh_th: f64,
    q_mev_per_reaction: f64,
    nuclei_per_reaction: f64,
    isotope_mass_u: f64,
) -> f64 {
    let energy_j = energy_mwh_th * JOULES_PER_MWH;
    let reaction_energy_j = q_mev_per_reaction * 1.0e6 * JOULES_PER_EV;
    let reaction_count = energy_j / reaction_energy_j;
    let mass_kg = reaction_count * nuclei_per_reaction * isotope_mass_u * KG_PER_U;
    mass_kg * 1.0e6 // kg -> mg
}

fn assert_relative_close(actual: f64, expected: f64, relative_tolerance: f64) {
    let scale = expected.abs().max(f64::MIN_POSITIVE);
    let relative_error = (actual - expected).abs() / scale;
    assert!(
        relative_error <= relative_tolerance,
        "actual={actual:.15e}, expected={expected:.15e}, relative_error={relative_error:.3e}, tolerance={relative_tolerance:.3e}"
    );
}

#[test]
fn independent_ideal_reaction_mass_fixtures_match_frozen_values() {
    // D-T: one D and one T per reaction, Q = 17.60 MeV.
    assert_relative_close(
        ideal_reactant_mg(1.0, 17.60, 1.0, DEUTERIUM_U),
        4.269_823_759_713_879,
        1.0e-10,
    );
    assert_relative_close(
        ideal_reactant_mg(1.0, 17.60, 1.0, TRITIUM_U),
        6.393_916_645_428_638_5,
        1.0e-10,
    );

    // D-3He: one D and one 3He per reaction, Q = 18.35 MeV.
    assert_relative_close(
        ideal_reactant_mg(1.0, 18.35, 1.0, DEUTERIUM_U),
        4.095_307_802_232_384_6,
        1.0e-10,
    );
    assert_relative_close(
        ideal_reactant_mg(1.0, 18.35, 1.0, HELIUM3_U),
        6.132_544_318_758_004,
        1.0e-10,
    );

    // D-D: two D nuclei per reaction. Keep the two branches separate.
    assert_relative_close(
        ideal_reactant_mg(1.0, 3.27, 2.0, DEUTERIUM_U),
        45.962_628_850_742_66,
        1.0e-10,
    );
    assert_relative_close(
        ideal_reactant_mg(1.0, 4.03, 2.0, DEUTERIUM_U),
        37.294_738_546_384_245,
        1.0e-10,
    );

    // Optional declared 50:50 branch-mixture approximation, with mean Q=3.65 MeV.
    // This is separate from, not a replacement for, the branch-resolved model.
    assert_relative_close(
        ideal_reactant_mg(1.0, (3.27 + 4.03) / 2.0, 2.0, DEUTERIUM_U),
        41.177_478_449_843_43,
        1.0e-10,
    );
}

#[test]
fn reaction_mass_scaling_obeys_energy_and_q_value_metamorphisms() {
    let base = ideal_reactant_mg(1.0, 17.60, 1.0, DEUTERIUM_U);
    let doubled_energy = ideal_reactant_mg(2.0, 17.60, 1.0, DEUTERIUM_U);
    let doubled_q = ideal_reactant_mg(1.0, 35.20, 1.0, DEUTERIUM_U);

    assert_relative_close(doubled_energy, 2.0 * base, 1.0e-12);
    assert_relative_close(doubled_q, 0.5 * base, 1.0e-12);

    // Net/gross electric output requires an explicit conversion assumption.
    // For a stated thermal efficiency eta, the ideal mass per gross-electric
    // MWh is m_th / eta, before fuel-cycle and recirculating losses.
    let eta = 0.40;
    let thermal_basis_mg = base;
    let gross_electric_basis_mg = thermal_basis_mg / eta;
    assert_relative_close(gross_electric_basis_mg, 2.5 * thermal_basis_mg, 1.0e-12);
}

#[test]
fn legacy_defaults_reproduce_the_documented_audit_discrepancy() {
    // Characterize, rather than endorse, the legacy assumptions. This test
    // passes when the current values are demonstrably below the ideal thermal
    // stoichiometric floor. Replace it when the API supplies an explicit energy
    // basis and qualified assumptions; do not merely update constants by guess.

    let dt = FuelCosts::dt_fusion();
    let dt_d_mg = dt.deuterium_kg_per_mwh * 1.0e6;
    let dt_t_mg = dt.tritium_g_per_mwh * 1.0e3;
    let dt_d_ideal = ideal_reactant_mg(1.0, 17.60, 1.0, DEUTERIUM_U);
    let dt_t_ideal = ideal_reactant_mg(1.0, 17.60, 1.0, TRITIUM_U);

    assert!(dt_d_mg < 0.60 * dt_d_ideal);
    assert!(dt_t_mg < 0.60 * dt_t_ideal);

    let dhe3 = FuelCosts::dhe3_fusion();
    let dhe3_d_mg = dhe3.deuterium_kg_per_mwh * 1.0e6;
    // Legacy API reuses the tritium-named field to hold He-3 grams.
    let dhe3_he3_mg = dhe3.tritium_g_per_mwh * 1.0e3;
    assert!(dhe3_d_mg < 0.60 * ideal_reactant_mg(1.0, 18.35, 1.0, DEUTERIUM_U));
    assert!(dhe3_he3_mg < 0.60 * ideal_reactant_mg(1.0, 18.35, 1.0, HELIUM3_U));

    let dd = FuelCosts::dd_fusion();
    let dd_d_mg = dd.deuterium_kg_per_mwh * 1.0e6;
    let dd_ideal_mg = ideal_reactant_mg(1.0, 3.27, 2.0, DEUTERIUM_U);
    assert!(
        dd_d_mg < 0.30 * dd_ideal_mg,
        "legacy D-D default {dd_d_mg} mg/MWh is not comparable to the frozen ideal thermal basis {dd_ideal_mg} mg/MWh"
    );
}
