// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cross-module physics relationship tests.
//!
//! These tests validate physics *between* modules: conservation laws,
//! mass hierarchies, thermodynamic consistency, and dimensional integrity.

use super::antimatter::Antimatter;
use super::hadrons::{Baryon, Meson};
use super::nuclear::EnergyScale;
use super::physics_test_helpers::{assert_relative_eq, particle_physics_setup};
use super::radiation_damage::FusionReaction;
use super::standard_model::{LeptonFlavor, PHYSICS_DIM, QuarkFlavor, StandardModel};
use crate::genesis::GenesisSeed;

// =========================================================================
// Section 64: Conservation Laws (5 tests)
// =========================================================================

#[test]
fn conservation_beta_decay_charge() {
    // n(0) → p(+1) + e(-1) + ν̄(0): total charge 0 in, 0 out
    let charge_in = Baryon::Neutron.charge(); // 0
    let charge_out = Baryon::Proton.charge() + LeptonFlavor::Electron.charge(); // +1 + (-1) = 0
    assert_eq!(
        charge_in, charge_out,
        "Beta decay charge conservation: {charge_in} = {charge_out}"
    );
}

#[test]
fn conservation_beta_decay_baryon() {
    // Baryon number: B=1 in (neutron), B=1 out (proton); leptons have B=0
    // Neutron and proton are both baryons (B=1)
    // This is structural: the beta_decay method produces a proton from a neutron
    let (_, model, hadrons, _, _, _) = particle_physics_setup();
    let (decay_proton, _, _) = hadrons.beta_decay(&model);
    // The decay proton should be very similar to the proton (baryon number preserved)
    let sim = decay_proton.similarity(&hadrons.proton);
    assert!(
        sim > 0.99,
        "Beta decay preserves baryon identity: similarity = {sim}"
    );
}

#[test]
fn conservation_annihilation_energy() {
    // e⁺e⁻ → 2γ: energy = 2 × 0.511 = 1.022 MeV
    let (_, model, _, _, _, antimatter) = particle_physics_setup();
    let event = antimatter.electron_positron_annihilation(&model);
    let expected = 2.0 * LeptonFlavor::Electron.mass_mev() as f64;
    assert_relative_eq(
        event.energy_mev,
        expected,
        1e-3,
        "e+e- annihilation energy = 2×m_e",
    );
}

#[test]
fn conservation_quark_proton_charge() {
    // Proton = uud: charge = +2/3 + 2/3 + (-1/3) = +1
    let u_thirds = QuarkFlavor::Up.charge_thirds();
    let d_thirds = QuarkFlavor::Down.charge_thirds();
    let proton_charge_thirds = u_thirds + u_thirds + d_thirds;
    assert_eq!(
        proton_charge_thirds, 3,
        "Proton quark charges: uud = 2+2+(-1) = 3 thirds → +1"
    );
    assert_eq!(Baryon::Proton.charge(), 1, "Proton charge = +1");
}

#[test]
fn conservation_quark_neutron_charge() {
    // Neutron = udd: charge = +2/3 + (-1/3) + (-1/3) = 0
    let u_thirds = QuarkFlavor::Up.charge_thirds();
    let d_thirds = QuarkFlavor::Down.charge_thirds();
    let neutron_charge_thirds = u_thirds + d_thirds + d_thirds;
    assert_eq!(
        neutron_charge_thirds, 0,
        "Neutron quark charges: udd = 2+(-1)+(-1) = 0 thirds"
    );
    assert_eq!(Baryon::Neutron.charge(), 0, "Neutron charge = 0");
}

// =========================================================================
// Section 65: Mass Hierarchy Chains (5 tests)
// =========================================================================

#[test]
fn hierarchy_quark_mass_ordering() {
    // up < down < strange < charm < bottom < top
    let masses: Vec<f32> = [
        QuarkFlavor::Up,
        QuarkFlavor::Down,
        QuarkFlavor::Strange,
        QuarkFlavor::Charm,
        QuarkFlavor::Bottom,
        QuarkFlavor::Top,
    ]
    .iter()
    .map(|q| q.mass_mev())
    .collect();

    for i in 0..masses.len() - 1 {
        assert!(
            masses[i] < masses[i + 1],
            "Quark mass ordering violated at index {i}: {} >= {}",
            masses[i],
            masses[i + 1]
        );
    }
}

#[test]
fn hierarchy_lepton_mass_ordering() {
    // electron < muon < tau
    let e = LeptonFlavor::Electron.mass_mev();
    let mu = LeptonFlavor::Muon.mass_mev();
    let tau = LeptonFlavor::Tau.mass_mev();
    assert!(e < mu, "m_e < m_μ: {} < {}", e, mu);
    assert!(mu < tau, "m_μ < m_τ: {} < {}", mu, tau);
}

#[test]
fn hierarchy_nucleon_heavier_than_pion() {
    // 938 > 140 MeV
    let proton_mass = Baryon::Proton.mass_mev();
    let pion_mass = Meson::PionPlus.mass_mev();
    assert!(
        proton_mass > pion_mass,
        "Nucleon should be heavier than pion: {} > {}",
        proton_mass,
        pion_mass
    );
    assert!(
        proton_mass > 900.0,
        "Proton mass ~938 MeV, got {proton_mass}"
    );
    assert!(pion_mass < 200.0, "Pion mass ~140 MeV, got {pion_mass}");
}

#[test]
fn hierarchy_neutron_proton_mass_diff() {
    // m_n - m_p ≈ 1.293 MeV
    let diff = Baryon::Neutron.mass_mev() as f64 - Baryon::Proton.mass_mev() as f64;
    assert_relative_eq(diff, 1.293, 0.01, "Neutron-proton mass difference");
}

#[test]
fn hierarchy_generation_mass_increase() {
    // Gen1 < Gen2 < Gen3 for quarks and leptons
    // Quarks: up < charm < top; down < strange < bottom
    assert!(
        QuarkFlavor::Up.mass_mev() < QuarkFlavor::Charm.mass_mev(),
        "up < charm"
    );
    assert!(
        QuarkFlavor::Charm.mass_mev() < QuarkFlavor::Top.mass_mev(),
        "charm < top"
    );
    assert!(
        QuarkFlavor::Down.mass_mev() < QuarkFlavor::Strange.mass_mev(),
        "down < strange"
    );
    assert!(
        QuarkFlavor::Strange.mass_mev() < QuarkFlavor::Bottom.mass_mev(),
        "strange < bottom"
    );
    // Leptons: electron < muon < tau
    assert!(
        LeptonFlavor::Electron.mass_mev() < LeptonFlavor::Muon.mass_mev(),
        "e < μ"
    );
    assert!(
        LeptonFlavor::Muon.mass_mev() < LeptonFlavor::Tau.mass_mev(),
        "μ < τ"
    );
}

// =========================================================================
// Section 66: Thermodynamic Consistency (5 tests)
// =========================================================================

#[test]
fn thermo_binding_curve_shape() {
    // BE/A increases to iron (~56), decreases after
    let (_, _, _, _, nuclear, _) = particle_physics_setup();
    let be4 = nuclear.binding_energy_per_nucleon(4);
    let be56 = nuclear.binding_energy_per_nucleon(56);
    let be238 = nuclear.binding_energy_per_nucleon(238);
    assert!(be4 < be56, "BE(4) < BE(56): {be4} < {be56}");
    assert!(be56 > be238, "BE(56) > BE(238): {be56} > {be238}");
}

#[test]
fn thermo_energy_scale_self_consistent() {
    // density_ratio = typical_ev / Chemical for all scales
    let chemical = EnergyScale::Chemical.typical_ev();
    let nuclear = EnergyScale::Nuclear.typical_ev();
    let ratio = nuclear / chemical;
    assert_relative_eq(
        ratio,
        EnergyScale::Nuclear.density_ratio(),
        1e-6,
        "Nuclear density ratio should equal Nuclear/Chemical typical_ev ratio",
    );
}

#[test]
fn thermo_stellar_fuel_chain() {
    // H→He→C→Fe: monotonically increasing BE/A
    let (_, _, _, _, nuclear, _) = particle_physics_setup();
    let be_h = nuclear.binding_energy_per_nucleon(2); // Deuterium
    let be_he = nuclear.binding_energy_per_nucleon(4); // Helium-4
    let be_c = nuclear.binding_energy_per_nucleon(12); // Carbon-12
    let be_fe = nuclear.binding_energy_per_nucleon(56); // Iron-56
    assert!(be_h < be_he, "BE(H-2) < BE(He-4): {be_h} < {be_he}");
    assert!(be_he < be_c, "BE(He-4) < BE(C-12): {be_he} < {be_c}");
    assert!(be_c < be_fe, "BE(C-12) < BE(Fe-56): {be_c} < {be_fe}");
}

#[test]
fn thermo_nuclear_vs_chemical() {
    // Nuclear (~MeV) >> Chemical (~eV) by >1e5
    let nuclear = EnergyScale::Nuclear.typical_ev();
    let chemical = EnergyScale::Chemical.typical_ev();
    let ratio = nuclear / chemical;
    assert!(
        ratio > 1e5,
        "Nuclear/Chemical energy ratio should be >1e5, got {ratio}"
    );
}

#[test]
fn thermo_dt_fusion_exceeds_fission() {
    // DT fusion Q/nucleon > U235 fission Q/nucleon
    let (_, _, _, _, nuclear, _) = particle_physics_setup();
    // DT fusion: 2+3→4 nucleons, Q = total energy ~ 17.6 MeV
    let dt_total = FusionReaction::DT.total_energy_mev();
    let dt_q_per_nucleon = dt_total / 5.0; // 5 input nucleons

    // U235 fission: ~200 MeV per fission, 235 nucleons
    let fission_q = nuclear.fission_q_value(235, 140, 95);
    let fission_q_per_nucleon = fission_q as f64 / 235.0;

    assert!(
        dt_q_per_nucleon > fission_q_per_nucleon,
        "DT fusion Q/nucleon ({dt_q_per_nucleon:.3}) > U235 fission Q/nucleon ({fission_q_per_nucleon:.3})"
    );
}

// =========================================================================
// Section 67: Dimensional Consistency (5 tests)
// =========================================================================

#[test]
fn dim_all_sm_particles_physics_dim() {
    // All 19 StandardModel particle vectors should be PHYSICS_DIM
    let genesis = GenesisSeed::from_phrase("cross_dim_sm");
    let model = StandardModel::from_genesis(&genesis);
    let particles = [
        ("up_quark", &model.up_quark),
        ("down_quark", &model.down_quark),
        ("charm_quark", &model.charm_quark),
        ("strange_quark", &model.strange_quark),
        ("top_quark", &model.top_quark),
        ("bottom_quark", &model.bottom_quark),
        ("electron", &model.electron),
        ("electron_neutrino", &model.electron_neutrino),
        ("muon", &model.muon),
        ("muon_neutrino", &model.muon_neutrino),
        ("tau", &model.tau),
        ("tau_neutrino", &model.tau_neutrino),
        ("photon", &model.photon),
        ("gluon", &model.gluon),
        ("w_plus", &model.w_plus),
        ("w_minus", &model.w_minus),
        ("z_boson", &model.z_boson),
    ];
    for (name, vec) in &particles {
        assert_eq!(vec.dim(), PHYSICS_DIM, "{name} dim should be PHYSICS_DIM");
    }
}

#[test]
fn dim_all_hadron_vectors_physics_dim() {
    // All Hadrons fields should be PHYSICS_DIM
    let (_, _, hadrons, _, _, _) = particle_physics_setup();
    let fields = [
        ("proton", &hadrons.proton),
        ("neutron", &hadrons.neutron),
        ("delta_plus_plus", &hadrons.delta_plus_plus),
        ("delta_plus", &hadrons.delta_plus),
        ("delta_zero", &hadrons.delta_zero),
        ("delta_minus", &hadrons.delta_minus),
        ("lambda", &hadrons.lambda),
        ("sigma_plus", &hadrons.sigma_plus),
        ("sigma_zero", &hadrons.sigma_zero),
        ("sigma_minus", &hadrons.sigma_minus),
        ("pion_plus", &hadrons.pion_plus),
        ("pion_zero", &hadrons.pion_zero),
        ("pion_minus", &hadrons.pion_minus),
        ("kaon_plus", &hadrons.kaon_plus),
        ("kaon_zero", &hadrons.kaon_zero),
        ("binding_energy", &hadrons.binding_energy),
        ("nuclear_force", &hadrons.nuclear_force),
    ];
    for (name, vec) in &fields {
        assert_eq!(
            vec.dim(),
            PHYSICS_DIM,
            "Hadron {name} dim should be PHYSICS_DIM"
        );
    }
}

#[test]
fn dim_antimatter_matches_matter() {
    // Antimatter vectors should have same dimension as matter
    let (_, model, _, _, _, antimatter) = particle_physics_setup();
    assert_eq!(
        antimatter.positron.dim(),
        model.electron.dim(),
        "Positron dim should match electron dim"
    );
    assert_eq!(
        antimatter.antiproton.dim(),
        PHYSICS_DIM,
        "Antiproton dim should be PHYSICS_DIM"
    );
    assert_eq!(
        antimatter.antineutron.dim(),
        PHYSICS_DIM,
        "Antineutron dim should be PHYSICS_DIM"
    );
}

#[test]
fn dim_qcd_color_charges() {
    // All 8 color charges should be PHYSICS_DIM
    let genesis = GenesisSeed::from_phrase("cross_dim_qcd");
    let model = StandardModel::from_genesis(&genesis);
    let qcd = super::qft::QCDEncoder::from_genesis(&genesis, &model);
    for (i, charge) in qcd.color_charges.iter().enumerate() {
        assert_eq!(
            charge.dim(),
            PHYSICS_DIM,
            "Color charge {i} dim should be PHYSICS_DIM"
        );
    }
}

#[test]
fn dim_cpt_preserves_dim() {
    // conjugate() should preserve dimension
    let genesis = GenesisSeed::from_phrase("cross_dim_cpt");
    let v = genesis.hv("test::particle_for_cpt", PHYSICS_DIM);
    let conj = Antimatter::conjugate(&v);
    assert_eq!(
        conj.dim(),
        v.dim(),
        "CPT conjugate should preserve dimension"
    );
    assert_eq!(
        conj.dim(),
        PHYSICS_DIM,
        "Conjugated vector dim = PHYSICS_DIM"
    );
}
