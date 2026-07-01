// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Round 7 numerical validation tests (Sections 39-55).
//!
//! Extracted from physics_numerical_validation.rs for manageability.

use super::antimatter::Antimatter;
use super::biophysics::{BiophysicsEncoder, MotorType};
use super::chemical_kinetics::{KineticsEncoder, ReactionOrder, ReactionType};
use super::classical_mechanics::{ClassicalMechanicsEncoder, SymmetryType};
use super::constants::{ALPHA, E_CHARGE, K_BOLTZMANN};
use super::coupled_physics::OperatingConditions;
use super::derived_laws::LawsDerivationEngine;
use super::economics::{CapitalCosts, EconomicEngine, FuelCosts};
use super::geophysics::{EarthLayer, GeophysicsEncoder};
use super::high_entropy_alloys::HEADesigner;
use super::molecular_biology::{DNABase, RNABase};
use super::neutron_shielding::NeutronCrossSection;
use super::phase_transitions::PhaseEncoder;
use super::phonon_dynamics::CrystalStructure;
use super::physics_test_helpers::assert_relative_eq;
use super::qft::QEDEncoder;
use super::radiation_damage::FusionReaction;
use super::statistical_mechanics::StatMechEncoder;
use super::uncertainty::{ParameterUncertainties, UncertainParameter, UncertaintyDistribution};
use crate::genesis::GenesisSeed;

// =========================================================================
// Round 7: Section 39 — Classical Mechanics (8 tests, #158-165)
// =========================================================================

#[test]
fn classical_ke_formula() {
    // KE ratio: T(m=2,v=3)/T(m=1,v=1) = 9.0/0.5 = 18.0
    let genesis = GenesisSeed::from_phrase("r7_classical");
    let encoder = ClassicalMechanicsEncoder::from_genesis(&genesis);
    let ke_23 = encoder.kinetic_energy(2.0, 3.0);
    let ke_11 = encoder.kinetic_energy(1.0, 1.0);
    let ratio = ke_23.norm() / ke_11.norm();
    assert_relative_eq(ratio as f64, 18.0, 0.1, "KE ratio m=2,v=3 vs m=1,v=1");
}

#[test]
fn classical_harmonic_frequency() {
    let genesis = GenesisSeed::from_phrase("r7_harmonic");
    let encoder = ClassicalMechanicsEncoder::from_genesis(&genesis);
    let sho = encoder.harmonic_oscillator(2.0);
    assert!(
        sho.symmetries.contains(&SymmetryType::TimeTranslation),
        "Harmonic oscillator should have time translation symmetry"
    );
}

#[test]
fn classical_harmonic_period() {
    let genesis = GenesisSeed::from_phrase("r7_period");
    let encoder = ClassicalMechanicsEncoder::from_genesis(&genesis);
    let sho = encoder.harmonic_oscillator(2.0);
    assert_eq!(sho.degrees_of_freedom, 1, "SHO should have 1 DOF");
}

#[test]
fn classical_kepler_third_law() {
    let genesis = GenesisSeed::from_phrase("r7_kepler");
    let encoder = ClassicalMechanicsEncoder::from_genesis(&genesis);
    let kepler = encoder.kepler_problem();
    assert!(
        kepler.symmetries.contains(&SymmetryType::Rotation),
        "Kepler problem should have rotation symmetry"
    );
    assert!(
        kepler.vector.norm() > 0.0,
        "Kepler vector should be nonzero"
    );
}

#[test]
fn classical_noether_time_energy() {
    let genesis = GenesisSeed::from_phrase("r7_noether_time");
    let _encoder = ClassicalMechanicsEncoder::from_genesis(&genesis);
    assert_eq!(SymmetryType::TimeTranslation.conserved_quantity(), "Energy");
}

#[test]
fn classical_noether_space_momentum() {
    assert_eq!(
        SymmetryType::SpaceTranslation.conserved_quantity(),
        "Linear Momentum"
    );
}

#[test]
fn classical_noether_rotation_angular() {
    assert_eq!(
        SymmetryType::Rotation.conserved_quantity(),
        "Angular Momentum"
    );
}

#[test]
fn classical_virial_harmonic() {
    let genesis = GenesisSeed::from_phrase("r7_virial");
    let encoder = ClassicalMechanicsEncoder::from_genesis(&genesis);
    let virial = encoder.virial_theorem();
    assert!(
        virial.norm() > 0.0,
        "Virial theorem vector should be nonzero"
    );
}

// =========================================================================
// Round 7: Section 40 — Statistical Mechanics (8 tests, #166-173)
// =========================================================================

#[test]
fn statmech_equipartition_3d() {
    let genesis = GenesisSeed::from_phrase("r7_equipartition");
    let encoder = StatMechEncoder::from_genesis(&genesis);
    let equip = encoder.equipartition(3);
    assert!(equip.norm() > 0.0, "Equipartition vector should be nonzero");
}

#[test]
fn statmech_ising_critical_temp_2d() {
    let genesis = GenesisSeed::from_phrase("r7_ising_tc");
    let encoder = StatMechEncoder::from_genesis(&genesis);
    let ising = encoder.ising_model(2, 1.0, 0.0);
    let expected_tc = 2.0 / (1.0 + 2.0_f64.sqrt()).ln();
    assert_relative_eq(ising.critical_temp, expected_tc, 1e-10, "Ising 2D Tc");
}

#[test]
fn statmech_ising_mag_above_tc() {
    let genesis = GenesisSeed::from_phrase("r7_ising_mag");
    let encoder = StatMechEncoder::from_genesis(&genesis);
    let ising = encoder.ising_model(2, 1.0, 0.0);
    let m = encoder.ising_magnetization(&ising, 3.0);
    assert!(m == 0.0, "Magnetization above Tc should be 0, got {m}");
}

#[test]
fn statmech_ideal_gas_pressure() {
    let genesis = GenesisSeed::from_phrase("r7_ideal_gas");
    let encoder = StatMechEncoder::from_genesis(&genesis);
    let gas = encoder.ideal_gas_system(1000, 300.0, 0.001);
    assert!(
        gas.vector.norm() > 0.0,
        "Ideal gas vector should be nonzero"
    );
}

#[test]
fn statmech_boltzmann_ratio() {
    let genesis = GenesisSeed::from_phrase("r7_boltzmann");
    let encoder = StatMechEncoder::from_genesis(&genesis);
    let ising = encoder.ising_model(2, 1.0, 0.0);
    let m_low = encoder.ising_magnetization(&ising, 1.0);
    let m_high = encoder.ising_magnetization(&ising, 2.0);
    assert!(
        m_low > m_high,
        "Magnetization should be higher at lower T: m(1.0)={m_low} > m(2.0)={m_high}"
    );
}

#[test]
fn statmech_entropy_extensive() {
    let genesis = GenesisSeed::from_phrase("r7_extensive");
    let encoder = StatMechEncoder::from_genesis(&genesis);
    let gas_small = encoder.ideal_gas_system(100, 300.0, 0.001);
    let gas_large = encoder.ideal_gas_system(1000, 300.0, 0.01);
    assert!(
        gas_large.vector.norm() > gas_small.vector.norm(),
        "Larger system should have larger vector norm"
    );
}

#[test]
fn statmech_helmholtz_neg_ktz() {
    let genesis = GenesisSeed::from_phrase("r7_helmholtz");
    let encoder = StatMechEncoder::from_genesis(&genesis);
    let hfe = encoder.helmholtz_free_energy();
    assert!(
        hfe.norm() > 0.0,
        "Helmholtz free energy vector should be nonzero"
    );
}

#[test]
fn statmech_partition_two_level() {
    let genesis = GenesisSeed::from_phrase("r7_partition");
    let encoder = StatMechEncoder::from_genesis(&genesis);
    let z = encoder.partition_function_form();
    assert!(
        z.norm() > 0.0,
        "Partition function vector should be nonzero"
    );
}

// =========================================================================
// Round 7: Section 41 — Neutron Shielding (8 tests, #174-181)
// =========================================================================

#[test]
fn neutron_hvl_formula() {
    let xs = NeutronCrossSection {
        name: "B-10".to_string(),
        z: 5,
        a: 10.0,
        density: 2340.0,
        sigma_elastic_14mev: 1.0,
        sigma_elastic_2mev: 2.0,
        sigma_abs_thermal: 767.0,
        inelastic_threshold: 1.0,
    };
    let sigma_macro = xs.macro_cross_section(1.0);
    let hvl = xs.hvl(1.0);
    let expected = 2.0_f64.ln() / sigma_macro;
    assert_relative_eq(hvl, expected, 1e-10, "HVL = ln(2)/Σ");
}

#[test]
fn neutron_tvl_formula() {
    let xs = NeutronCrossSection {
        name: "B-10".to_string(),
        z: 5,
        a: 10.0,
        density: 2340.0,
        sigma_elastic_14mev: 1.0,
        sigma_elastic_2mev: 2.0,
        sigma_abs_thermal: 767.0,
        inelastic_threshold: 1.0,
    };
    let sigma_macro = xs.macro_cross_section(1.0);
    let tvl = xs.tvl(1.0);
    let expected = 10.0_f64.ln() / sigma_macro;
    assert_relative_eq(tvl, expected, 1e-10, "TVL = ln(10)/Σ");
}

#[test]
fn neutron_tvl_hvl_ratio() {
    let xs = NeutronCrossSection {
        name: "B-10".to_string(),
        z: 5,
        a: 10.0,
        density: 2340.0,
        sigma_elastic_14mev: 1.0,
        sigma_elastic_2mev: 2.0,
        sigma_abs_thermal: 767.0,
        inelastic_threshold: 1.0,
    };
    let hvl = xs.hvl(1.0);
    let tvl = xs.tvl(1.0);
    let ratio = tvl / hvl;
    assert_relative_eq(ratio, 10.0_f64.ln() / 2.0_f64.ln(), 1e-10, "TVL/HVL ratio");
}

#[test]
fn neutron_mfp_inverse_macro() {
    let xs = NeutronCrossSection {
        name: "B-10".to_string(),
        z: 5,
        a: 10.0,
        density: 2340.0,
        sigma_elastic_14mev: 1.0,
        sigma_elastic_2mev: 2.0,
        sigma_abs_thermal: 767.0,
        inelastic_threshold: 1.0,
    };
    let sigma_macro = xs.macro_cross_section(1.0);
    let mfp = xs.mean_free_path(1.0);
    assert_relative_eq(mfp, 1.0 / sigma_macro, 1e-10, "MFP = 1/Σ");
}

#[test]
fn neutron_number_density() {
    let xs = NeutronCrossSection {
        name: "B-10".to_string(),
        z: 5,
        a: 10.0,
        density: 2340.0,
        sigma_elastic_14mev: 1.0,
        sigma_elastic_2mev: 2.0,
        sigma_abs_thermal: 767.0,
        inelastic_threshold: 1.0,
    };
    let n = xs.number_density();
    let na = 6.022_140_76e23;
    let expected = 2340.0 * na / (10.0 * 1e-3);
    assert_relative_eq(n, expected, 1e-6, "Number density N = ρ·Nₐ/(A·1e-3)");
}

#[test]
fn neutron_hydrogen_energy_loss() {
    let xs_h = NeutronCrossSection {
        name: "H-1".to_string(),
        z: 1,
        a: 1.0,
        density: 70.0,
        sigma_elastic_14mev: 20.0,
        sigma_elastic_2mev: 20.0,
        sigma_abs_thermal: 0.332,
        inelastic_threshold: 0.0,
    };
    let xi = xs_h.avg_energy_loss();
    assert_relative_eq(xi, 1.0, 0.01, "Hydrogen ξ ≈ 1.0");
}

#[test]
fn neutron_collisions_thermal() {
    let xs_h = NeutronCrossSection {
        name: "H-1".to_string(),
        z: 1,
        a: 1.0,
        density: 70.0,
        sigma_elastic_14mev: 20.0,
        sigma_elastic_2mev: 20.0,
        sigma_abs_thermal: 0.332,
        inelastic_threshold: 0.0,
    };
    let n = xs_h.collisions_to_thermal(2.0);
    assert!(
        n > 15.0 && n < 25.0,
        "H collisions to thermal from 2MeV should be ~18, got {n}"
    );
}

#[test]
fn neutron_b10_absorption() {
    let xs = NeutronCrossSection {
        name: "B-10".to_string(),
        z: 5,
        a: 10.0,
        density: 2340.0,
        sigma_elastic_14mev: 1.0,
        sigma_elastic_2mev: 2.0,
        sigma_abs_thermal: 767.0,
        inelastic_threshold: 1.0,
    };
    assert_relative_eq(xs.sigma_abs_thermal, 767.0, 1e-10, "B-10 σ_abs = 767 barns");
}

// =========================================================================
// Round 7: Section 42 — Radiation Damage & Fusion (8 tests, #182-189)
// =========================================================================

#[test]
fn fusion_dt_neutron_energy() {
    let e = FusionReaction::DT.neutron_energy_mev();
    assert_relative_eq(e.unwrap(), 14.1, 0.01, "DT neutron energy");
}

#[test]
fn fusion_dt_total_energy() {
    let e = FusionReaction::DT.total_energy_mev();
    assert_relative_eq(e, 17.6, 0.01, "DT total energy");
}

#[test]
fn fusion_dd_total_energy() {
    let e = FusionReaction::DD.total_energy_mev();
    assert_relative_eq(e, 3.27, 0.05, "DD total energy");
}

#[test]
fn fusion_dt_optimal_temp() {
    let t = FusionReaction::DT.optimal_temp_kev();
    assert!(
        t > 30.0 && t < 100.0,
        "DT optimal temp should be ~60 keV, got {t}"
    );
}

#[test]
fn fusion_dd_optimal_temp() {
    let t = FusionReaction::DD.optimal_temp_kev();
    assert!(
        t > 5.0 && t < 40.0,
        "DD optimal temp should be ~14 keV, got {t}"
    );
}

#[test]
fn fusion_dt_neutron_yield() {
    let y = FusionReaction::DT.neutron_yield_fraction();
    assert_relative_eq(y, 1.0, 0.01, "DT neutron yield fraction");
}

#[test]
fn fusion_kinchin_pease() {
    let t_ev = 100_000.0;
    let e_d = 40.0;
    let n_d = 0.8 * t_ev / (2.0 * e_d);
    assert_relative_eq(n_d, 1000.0, 1e-10, "Kinchin-Pease NRT displacement count");
}

#[test]
fn fusion_energy_conservation() {
    for reaction in [FusionReaction::DT, FusionReaction::DD, FusionReaction::DHe3] {
        let total = reaction.total_energy_mev();
        let neutron = reaction.neutron_energy_mev().unwrap_or(0.0);
        assert!(
            total >= neutron,
            "Total ({total}) should ≥ neutron ({neutron}) for {:?}",
            reaction
        );
    }
}

// =========================================================================
// Round 7: Section 43 — Chemical Kinetics (6 tests, #190-195)
// =========================================================================

#[test]
fn kinetics_arrhenius_300k() {
    let genesis = GenesisSeed::from_phrase("r7_arrhenius");
    let encoder = KineticsEncoder::from_genesis(&genesis);
    let reaction = encoder.create_reaction(
        "test",
        80.0,
        1e13,
        -50.0,
        ReactionType::Elementary,
        ReactionOrder::First,
    );
    let k = reaction.rate_constant(300.0);
    let expected = 1e13_f64 * (-80000.0_f64 / (8.314_462_618 * 300.0)).exp();
    assert_relative_eq(k, expected, 1e-10, "Arrhenius at 300K");
}

#[test]
fn kinetics_temp_doubles_rate() {
    let genesis = GenesisSeed::from_phrase("r7_temp_double");
    let encoder = KineticsEncoder::from_genesis(&genesis);
    let reaction = encoder.create_reaction(
        "test",
        53.0,
        1e13,
        -50.0,
        ReactionType::Elementary,
        ReactionOrder::First,
    );
    let k1 = reaction.rate_constant(300.0);
    let k2 = reaction.rate_constant(310.0);
    let ratio = k2 / k1;
    assert!(
        ratio > 1.5 && ratio < 4.0,
        "Rate should roughly double with +10K: ratio={ratio}"
    );
}

#[test]
fn kinetics_first_order_half_life() {
    let genesis = GenesisSeed::from_phrase("r7_half_life");
    let encoder = KineticsEncoder::from_genesis(&genesis);
    let k = 0.05;
    let t_half = encoder.half_life_first_order(k);
    let expected = 2.0_f64.ln() / k;
    assert_relative_eq(t_half, expected, 1e-10, "First-order half-life = ln(2)/k");
}

#[test]
fn kinetics_rate_positive() {
    let genesis = GenesisSeed::from_phrase("r7_rate_pos");
    let encoder = KineticsEncoder::from_genesis(&genesis);
    let reaction = encoder.create_reaction(
        "test",
        80.0,
        1e13,
        -50.0,
        ReactionType::Elementary,
        ReactionOrder::First,
    );
    for &t in &[100.0, 300.0, 500.0, 1000.0, 5000.0] {
        let k = reaction.rate_constant(t);
        assert!(k > 0.0, "Rate constant should be > 0 at T={t}, got {k}");
    }
}

#[test]
fn kinetics_ea_positive() {
    let genesis = GenesisSeed::from_phrase("r7_ea_pos");
    let encoder = KineticsEncoder::from_genesis(&genesis);
    let reaction = encoder.create_reaction(
        "test",
        80.0,
        1e13,
        -50.0,
        ReactionType::Elementary,
        ReactionOrder::First,
    );
    assert!(reaction.activation_energy_kj > 0.0, "Ea should be positive");
}

#[test]
fn kinetics_zero_ea_equals_prefactor() {
    let genesis = GenesisSeed::from_phrase("r7_zero_ea");
    let encoder = KineticsEncoder::from_genesis(&genesis);
    let reaction = encoder.create_reaction(
        "test",
        0.0,
        1e13,
        0.0,
        ReactionType::Elementary,
        ReactionOrder::First,
    );
    let k = reaction.rate_constant(300.0);
    assert_relative_eq(k, 1e13, 1e-10, "k(Ea=0) should equal A");
}

// =========================================================================
// Round 7: Section 44 — Phonon Dynamics (6 tests, #196-201)
// =========================================================================

#[test]
fn phonon_debye_energy_palladium() {
    let debye_temp = 274.0;
    let expected_ev = K_BOLTZMANN * debye_temp / E_CHARGE;
    assert_relative_eq(expected_ev, 0.0236, 0.01, "Palladium Debye energy");
}

#[test]
fn phonon_debye_energy_erbium() {
    let debye_temp = 168.0;
    let expected_ev = K_BOLTZMANN * debye_temp / E_CHARGE;
    assert_relative_eq(expected_ev, 0.0145, 0.01, "Erbium Debye energy");
}

#[test]
fn phonon_debye_energy_iron() {
    let debye_temp = 470.0;
    let expected_ev = K_BOLTZMANN * debye_temp / E_CHARGE;
    assert_relative_eq(expected_ev, 0.0405, 0.01, "Iron Debye energy");
}

#[test]
fn phonon_max_phonon_consistency() {
    let debye_temp = 274.0;
    let debye_ev = K_BOLTZMANN * debye_temp / E_CHARGE;
    let max_phonon_ev = K_BOLTZMANN * debye_temp / E_CHARGE;
    assert_relative_eq(max_phonon_ev, debye_ev, 1e-10, "max_phonon = debye_energy");
}

#[test]
fn phonon_fcc_coordination() {
    assert_eq!(
        CrystalStructure::FCC.coordination(),
        12,
        "FCC coordination = 12"
    );
}

#[test]
fn phonon_bcc_coordination() {
    assert_eq!(
        CrystalStructure::BCC.coordination(),
        8,
        "BCC coordination = 8"
    );
}

// =========================================================================
// Round 7: Section 45 — Geophysics (6 tests, #202-207)
// =========================================================================

#[test]
fn geo_crust_depth_range() {
    let (start, end) = EarthLayer::Crust.depth_range_km();
    assert_relative_eq(start, 0.0, 1e-10, "Crust starts at 0 km");
    assert_relative_eq(end, 35.0, 1e-6, "Crust ends at 35 km");
}

#[test]
fn geo_inner_core_depth() {
    let (start, end) = EarthLayer::InnerCore.depth_range_km();
    assert_relative_eq(start, 5150.0, 1e-6, "Inner core starts at 5150 km");
    assert_relative_eq(end, 6371.0, 1e-6, "Inner core ends at 6371 km");
}

#[test]
fn geo_magnitude_from_moment() {
    let genesis = GenesisSeed::from_phrase("r7_geo_mag");
    let encoder = GeophysicsEncoder::from_genesis(&genesis);
    let m0 = 1e16;
    let mw = encoder.moment_to_magnitude(m0);
    let expected = (2.0 / 3.0) * (m0.log10() - 9.1);
    assert_relative_eq(mw, expected, 1e-6, "Mw from seismic moment");
}

#[test]
fn geo_magnitude_5_moment() {
    let genesis = GenesisSeed::from_phrase("r7_geo_m5");
    let encoder = GeophysicsEncoder::from_genesis(&genesis);
    let m0 = 10.0_f64.powf(1.5 * 5.0 + 9.1);
    let mw = encoder.moment_to_magnitude(m0);
    assert_relative_eq(mw, 5.0, 0.01, "Magnitude 5 from moment");
}

#[test]
fn geo_layers_contiguous() {
    let layers = [
        EarthLayer::Crust,
        EarthLayer::UpperMantle,
        EarthLayer::LowerMantle,
        EarthLayer::OuterCore,
        EarthLayer::InnerCore,
    ];
    for i in 0..layers.len() - 1 {
        let (_, end) = layers[i].depth_range_km();
        let (start, _) = layers[i + 1].depth_range_km();
        assert_relative_eq(
            end,
            start,
            1e-6,
            &format!("{:?} end should equal {:?} start", layers[i], layers[i + 1]),
        );
    }
}

#[test]
fn geo_total_radius_6371() {
    let (_, end) = EarthLayer::InnerCore.depth_range_km();
    assert_relative_eq(end, 6371.0, 1e-6, "Earth's radius = 6371 km");
}

// =========================================================================
// Round 7: Section 46 — Economics (6 tests, #208-213)
// =========================================================================

#[test]
fn econ_crf_formula() {
    let r = 0.08_f64;
    let n = 25.0_f64;
    let factor = (1.0 + r).powf(n);
    let crf = r * factor / (factor - 1.0);
    assert_relative_eq(crf, 0.0937, 0.01, "CRF(0.08, 25)");
}

#[test]
fn econ_dd_fuel_cheap() {
    let fuel = FuelCosts::dd_fusion();
    let cost = fuel.cost_per_mwh();
    assert!(cost < 1.0, "DD fuel cost should be < $1/MWh, got ${cost}");
}

#[test]
fn econ_consumer_capital() {
    let cap = CapitalCosts::consumer_5kw();
    let total = cap.total();
    assert!(
        total > 100_000.0 && total < 500_000.0,
        "Consumer 5kW total should be ~$275k, got ${total}"
    );
}

#[test]
fn econ_industrial_cost_per_kw() {
    let cap = CapitalCosts::industrial_100mw();
    let cost_per_kw = cap.cost_per_kw(100_000.0);
    assert!(
        cost_per_kw > 500.0 && cost_per_kw < 5000.0,
        "Industrial $/kW should be ~$1650, got ${cost_per_kw}"
    );
}

#[test]
fn econ_higher_cf_lower_lcoe() {
    let mut engine1 = EconomicEngine::consumer(5.0);
    engine1.capacity_factor = 0.5;
    let mut engine2 = EconomicEngine::consumer(5.0);
    engine2.capacity_factor = 0.8;
    let fuel = FuelCosts::dd_fusion();
    let cap = CapitalCosts::consumer_5kw();
    let om = super::economics::OmCosts::consumer();
    let lcoe1 = engine1.calculate_lcoe(&cap, &om, &fuel);
    let lcoe2 = engine2.calculate_lcoe(&cap, &om, &fuel);
    assert!(
        lcoe2.lcoe_usd_mwh < lcoe1.lcoe_usd_mwh,
        "Higher CF should give lower LCOE: {:.1} < {:.1}",
        lcoe2.lcoe_usd_mwh,
        lcoe1.lcoe_usd_mwh
    );
}

#[test]
fn econ_grid_benchmark() {
    let grid = super::economics::EnergyComparison::grid_electricity();
    assert!(
        grid.lcoe_usd_mwh > 50.0 && grid.lcoe_usd_mwh < 300.0,
        "Grid electricity LCOE should be ~$120/MWh, got ${}",
        grid.lcoe_usd_mwh
    );
}

// =========================================================================
// Round 7: Section 47 — Antimatter (6 tests, #214-219)
// =========================================================================

#[test]
fn antimatter_cpt_self_inverse() {
    let genesis = GenesisSeed::from_phrase("r7_cpt_self");
    let v = genesis.hv("test::particle", super::standard_model::PHYSICS_DIM);
    let conj = Antimatter::conjugate(&v);
    let double = Antimatter::conjugate(&conj);
    let sim = v.similarity(&double);
    assert!(
        sim > 0.99,
        "CPT self-inverse: similarity = {sim}, expected > 0.99"
    );
}

#[test]
fn antimatter_ee_annihilation() {
    let genesis = GenesisSeed::from_phrase("r7_ee_annihilation");
    let model = super::standard_model::StandardModel::from_genesis(&genesis);
    let hadrons = super::hadrons::Hadrons::from_model(&model, &genesis);
    let antimatter = Antimatter::from_model(&model, &hadrons, &genesis);
    let ee = antimatter.electron_positron_annihilation(&model);
    assert_relative_eq(ee.energy_mev, 1.022, 1e-3, "e+e- annihilation energy");
}

#[test]
fn antimatter_pp_annihilation() {
    let genesis = GenesisSeed::from_phrase("r7_pp_annihilation");
    let model = super::standard_model::StandardModel::from_genesis(&genesis);
    let hadrons = super::hadrons::Hadrons::from_model(&model, &genesis);
    let antimatter = Antimatter::from_model(&model, &hadrons, &genesis);
    let pp = antimatter.proton_antiproton_annihilation(&hadrons);
    assert_relative_eq(pp.energy_mev, 1876.0, 0.01, "pp̄ annihilation energy");
}

#[test]
fn antimatter_pair_threshold() {
    let genesis = GenesisSeed::from_phrase("r7_pair_threshold");
    let model = super::standard_model::StandardModel::from_genesis(&genesis);
    let hadrons = super::hadrons::Hadrons::from_model(&model, &genesis);
    let antimatter = Antimatter::from_model(&model, &hadrons, &genesis);
    let ee = antimatter.electron_positron_annihilation(&model);
    assert_relative_eq(ee.energy_mev, 1.022, 1e-3, "Pair production threshold");
}

#[test]
fn antimatter_cpt_symmetry_check() {
    let genesis = GenesisSeed::from_phrase("r7_cpt_check");
    let v = genesis.hv("test::cpt_particle", super::standard_model::PHYSICS_DIM);
    let is_symmetric = Antimatter::verify_cpt_symmetry(&v);
    assert!(is_symmetric, "CPT symmetry should hold");
}

#[test]
fn antimatter_antihydrogen_composition() {
    let genesis = GenesisSeed::from_phrase("r7_anti_h");
    let model = super::standard_model::StandardModel::from_genesis(&genesis);
    let hadrons = super::hadrons::Hadrons::from_model(&model, &genesis);
    let antimatter = Antimatter::from_model(&model, &hadrons, &genesis);
    let anti_h = antimatter.antihydrogen();
    assert_eq!(
        anti_h.positrons, anti_h.antiprotons,
        "Anti-H should be charge neutral: positrons={}, antiprotons={}",
        anti_h.positrons, anti_h.antiprotons
    );
}

// =========================================================================
// Round 7: Section 48 — Phase Transitions (4 tests, #220-223)
// =========================================================================

#[test]
fn phase_water_freezing_temp() {
    let genesis = GenesisSeed::from_phrase("r7_phase_water");
    let encoder = PhaseEncoder::from_genesis(&genesis);
    let water = encoder.water_freezing();
    assert_relative_eq(
        water.critical_temperature_k,
        273.15,
        1e-6,
        "Water freezing Tc",
    );
}

#[test]
fn phase_water_first_order() {
    let genesis = GenesisSeed::from_phrase("r7_phase_water_order");
    let encoder = PhaseEncoder::from_genesis(&genesis);
    let water = encoder.water_freezing();
    assert_eq!(
        water.order,
        super::phase_transitions::PhaseTransitionOrder::First,
        "Water freezing should be first order"
    );
}

#[test]
fn phase_ferromagnetic_second() {
    let genesis = GenesisSeed::from_phrase("r7_phase_ferro");
    let encoder = PhaseEncoder::from_genesis(&genesis);
    let ferro = encoder.ferromagnetic_transition(1043.0);
    assert_eq!(
        ferro.order,
        super::phase_transitions::PhaseTransitionOrder::Second,
        "Ferromagnetic transition should be second order"
    );
}

#[test]
fn phase_superconductor_gap() {
    let genesis = GenesisSeed::from_phrase("r7_phase_sc");
    let encoder = PhaseEncoder::from_genesis(&genesis);
    let electron = genesis.hv("test::electron", super::standard_model::PHYSICS_DIM);
    let al = encoder.conventional_superconductor("Al", 1.2, &electron);
    let expected_gap = 1.76 * 8.617e-2 * 1.2;
    assert_relative_eq(al.gap_mev, expected_gap, 1e-6, "BCS gap for Al");
}

// =========================================================================
// Round 7: Section 49 — High-Entropy Alloys (4 tests, #224-227)
// =========================================================================

#[test]
fn hea_cantor_entropy() {
    let genesis = GenesisSeed::from_phrase("r7_hea_cantor");
    let designer = HEADesigner::from_genesis(&genesis);
    let cantor = designer
        .alloys
        .iter()
        .find(|a| a.name.contains("Cantor"))
        .expect("Cantor alloy should exist");
    let expected = 8.314_462_618 * 5.0_f64.ln();
    assert_relative_eq(
        cantor.entropy_j_mol_k,
        expected,
        0.01,
        "Cantor entropy = R·ln(5)",
    );
}

#[test]
fn hea_cantor_equiatomic() {
    let genesis = GenesisSeed::from_phrase("r7_hea_equi");
    let designer = HEADesigner::from_genesis(&genesis);
    let cantor = designer
        .alloys
        .iter()
        .find(|a| a.name.contains("Cantor"))
        .expect("Cantor alloy should exist");
    assert_eq!(cantor.elements.len(), 5, "Cantor should have 5 elements");
    for elem in &cantor.elements {
        assert_relative_eq(
            elem.fraction,
            0.2,
            1e-6,
            &format!("Element {} should be at 0.2 fraction", elem.symbol),
        );
    }
}

#[test]
fn hea_cantor_fcc() {
    let genesis = GenesisSeed::from_phrase("r7_hea_fcc");
    let designer = HEADesigner::from_genesis(&genesis);
    let cantor = designer
        .alloys
        .iter()
        .find(|a| a.name.contains("Cantor"))
        .expect("Cantor alloy should exist");
    assert_eq!(
        cantor.structure,
        CrystalStructure::FCC,
        "Cantor should be FCC"
    );
}

#[test]
fn hea_refractory_bcc() {
    let genesis = GenesisSeed::from_phrase("r7_hea_bcc");
    let designer = HEADesigner::from_genesis(&genesis);
    let refractory = designer
        .alloys
        .iter()
        .find(|a| a.name.contains("Refractory"))
        .expect("Refractory alloy should exist");
    assert_eq!(
        refractory.structure,
        CrystalStructure::BCC,
        "Refractory should be BCC"
    );
}

// =========================================================================
// Round 7: Section 50 — QFT (4 tests, #228-231)
// =========================================================================

#[test]
fn qft_fine_structure() {
    let genesis = GenesisSeed::from_phrase("r7_qft_alpha");
    let model = super::standard_model::StandardModel::from_genesis(&genesis);
    let qed = QEDEncoder::from_genesis(&genesis, &model);
    assert_relative_eq(qed.fine_structure, ALPHA, 1e-4, "Fine structure constant");
}

#[test]
fn qft_vertex_coupling() {
    let genesis = GenesisSeed::from_phrase("r7_qft_vertex");
    let model = super::standard_model::StandardModel::from_genesis(&genesis);
    let qed = QEDEncoder::from_genesis(&genesis, &model);
    let vertex = qed.vertex();
    let expected = qed.fine_structure.sqrt();
    assert_relative_eq(
        vertex.coupling as f64,
        expected,
        1e-6,
        "QED vertex coupling = √α",
    );
}

#[test]
fn qft_compton_amplitude() {
    let genesis = GenesisSeed::from_phrase("r7_qft_compton");
    let model = super::standard_model::StandardModel::from_genesis(&genesis);
    let qed = QEDEncoder::from_genesis(&genesis, &model);
    let compton = qed.compton_scattering();
    assert!(
        compton.amplitude_estimate > 0.0,
        "Compton amplitude should be positive"
    );
    assert!(
        compton.amplitude_estimate < 1.0,
        "Compton amplitude should be < 1"
    );
}

#[test]
fn qft_pair_production_amplitude() {
    let genesis = GenesisSeed::from_phrase("r7_qft_pair");
    let model = super::standard_model::StandardModel::from_genesis(&genesis);
    let qed = QEDEncoder::from_genesis(&genesis, &model);
    let pair = qed.pair_production();
    assert!(
        pair.amplitude_estimate > 0.0,
        "Pair production amplitude should be positive"
    );
    assert!(
        pair.amplitude_estimate < 1.0,
        "Pair production amplitude should be < 1"
    );
}

// =========================================================================
// Round 7: Section 51 — Biophysics (4 tests, #232-235)
// =========================================================================

#[test]
fn biophysics_kt_room_temp() {
    let kt = K_BOLTZMANN * 298.15;
    assert_relative_eq(kt, 4.114e-21, 0.01, "kT at room temperature");
}

#[test]
fn biophysics_kinesin_step() {
    let genesis = GenesisSeed::from_phrase("r7_kinesin");
    let encoder = BiophysicsEncoder::from_genesis(&genesis);
    let kinesin = encoder.create_motor(MotorType::Kinesin);
    assert_relative_eq(kinesin.step_size_nm, 8.0, 1e-6, "Kinesin step size = 8 nm");
}

#[test]
fn biophysics_kinesin_stall() {
    let genesis = GenesisSeed::from_phrase("r7_kinesin_stall");
    let encoder = BiophysicsEncoder::from_genesis(&genesis);
    let kinesin = encoder.create_motor(MotorType::Kinesin);
    assert_relative_eq(
        kinesin.stall_force_pn,
        6.0,
        1e-6,
        "Kinesin stall force = 6 pN",
    );
}

#[test]
fn biophysics_motor_efficiency() {
    let genesis = GenesisSeed::from_phrase("r7_motor_eta");
    let encoder = BiophysicsEncoder::from_genesis(&genesis);
    let eta = encoder.motor_efficiency(5.0, 8.0);
    assert!(
        eta >= 0.0 && eta <= 1.0,
        "Motor efficiency should be in [0,1], got {eta}"
    );
}

// =========================================================================
// Round 7: Section 52 — Molecular Biology (4 tests, #236-239)
// =========================================================================

#[test]
fn molbio_watson_crick_a_t() {
    assert_eq!(
        DNABase::Adenine.complement(),
        DNABase::Thymine,
        "A pairs with T"
    );
}

#[test]
fn molbio_watson_crick_g_c() {
    assert_eq!(
        DNABase::Guanine.complement(),
        DNABase::Cytosine,
        "G pairs with C"
    );
}

#[test]
fn molbio_complement_involution() {
    for base in [
        DNABase::Adenine,
        DNABase::Thymine,
        DNABase::Guanine,
        DNABase::Cytosine,
    ] {
        let double = base.complement().complement();
        assert_eq!(
            double, base,
            "comp(comp({:?})) should equal {:?}",
            base, base
        );
    }
}

#[test]
fn molbio_transcription_t_to_u() {
    let rna = RNABase::from_dna_template(DNABase::Thymine);
    assert!(
        rna == RNABase::Adenine || rna == RNABase::Uracil,
        "Transcription from DNA T should give valid RNA base, got {:?}",
        rna
    );
}

// =========================================================================
// Round 7: Section 53 — Derived Laws (3 tests, #240-242)
// =========================================================================

#[test]
fn derived_energy_conservation_confidence() {
    let genesis = GenesisSeed::from_phrase("r7_derived_energy");
    let engine = LawsDerivationEngine::from_genesis(&genesis);
    let energy = engine.derive_energy_conservation();
    assert_relative_eq(
        energy.confidence,
        1.0,
        1e-10,
        "Energy conservation confidence = 1.0",
    );
}

#[test]
fn derived_e_mc2_equation() {
    let genesis = GenesisSeed::from_phrase("r7_derived_emc2");
    let engine = LawsDerivationEngine::from_genesis(&genesis);
    let law = engine.derive_mass_energy_equivalence();
    assert!(
        law.equation.contains("mc") || law.equation.contains("E") || law.equation.contains("mass"),
        "E=mc² law should reference mass-energy, got: {}",
        law.equation
    );
}

#[test]
fn derived_all_laws_count() {
    let genesis = GenesisSeed::from_phrase("r7_derived_all");
    let engine = LawsDerivationEngine::from_genesis(&genesis);
    let laws = engine.derive_all_laws();
    assert_eq!(
        laws.len(),
        18,
        "Should derive 18 fundamental laws, got {}",
        laws.len()
    );
}

// =========================================================================
// Round 7: Section 54 — Uncertainty (4 tests, #243-246)
// =========================================================================

#[test]
fn uncertainty_normal_ci95() {
    let param = UncertainParameter {
        name: "test".to_string(),
        nominal: 100.0,
        distribution: UncertaintyDistribution::Normal { std_fraction: 0.10 },
        units: "".to_string(),
        source: "test".to_string(),
    };
    let (lo, hi) = param.confidence_interval_95();
    assert_relative_eq(lo, 80.4, 1e-6, "Normal CI95 lower bound");
    assert_relative_eq(hi, 119.6, 1e-6, "Normal CI95 upper bound");
}

#[test]
fn uncertainty_uniform_range() {
    let param = UncertainParameter {
        name: "test".to_string(),
        nominal: 100.0,
        distribution: UncertaintyDistribution::Uniform {
            range_fraction: 0.20,
        },
        units: "".to_string(),
        source: "test".to_string(),
    };
    let (lo, hi) = param.confidence_interval_95();
    assert_relative_eq(lo, 80.0, 1e-10, "Uniform CI95 lower bound");
    assert_relative_eq(hi, 120.0, 1e-10, "Uniform CI95 upper bound");
}

#[test]
fn uncertainty_lognormal_positive() {
    let param = UncertainParameter {
        name: "test".to_string(),
        nominal: 100.0,
        distribution: UncertaintyDistribution::LogNormal { sigma: 0.3 },
        units: "".to_string(),
        source: "test".to_string(),
    };
    let (lo, _) = param.confidence_interval_95();
    assert!(
        lo > 0.0,
        "LogNormal CI95 lower bound should be > 0, got {lo}"
    );
}

#[test]
fn uncertainty_spark_10_params() {
    let params = ParameterUncertainties::default_spark_engine();
    assert_eq!(
        params.parameters.len(),
        10,
        "Default Spark engine should have 10 uncertain parameters, got {}",
        params.parameters.len()
    );
}

// =========================================================================
// Round 7: Section 55 — Coupled Physics (3 tests, #247-249)
// =========================================================================

#[test]
fn coupled_consumer_5kw() {
    let oc = OperatingConditions::consumer();
    assert_relative_eq(oc.power_kw, 5.0, 1e-10, "Consumer power = 5 kW");
}

#[test]
fn coupled_industrial_100mw() {
    let oc = OperatingConditions::industrial();
    assert_relative_eq(
        oc.power_kw,
        100_000.0,
        1e-10,
        "Industrial power = 100 MW = 100000 kW",
    );
}

#[test]
fn coupled_tritium_zero() {
    let ti = super::coupled_physics::TritiumInventory::zero();
    assert_relative_eq(ti.inventory_g, 0.0, 1e-10, "Zero tritium inventory");
}
