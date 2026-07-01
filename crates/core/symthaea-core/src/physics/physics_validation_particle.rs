// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Round 8 particle physics numerical validation tests (Sections 56-63).
//!
//! Extracted from physics_numerical_validation.rs for manageability.

use super::antimatter::{Antimatter, BaryogenesisConcepts};
use super::constants::ALPHA;
use super::hadrons::{Baryon, Meson};
use super::nuclear::EnergyScale;
use super::physics_test_helpers::{assert_relative_eq, particle_physics_setup};
use super::qft::{DivergenceType, ElectroweakEncoder, QCDEncoder, QEDEncoder};
use super::standard_model::{GaugeBoson, LeptonFlavor, PHYSICS_DIM, QuarkFlavor};
use crate::genesis::GenesisSeed;

// =========================================================================
// Round 8: Section 56 — Standard Model Fundamental Constants (16 tests)
// =========================================================================

#[test]
fn sm_up_type_quark_charges() {
    assert_eq!(QuarkFlavor::Up.charge_thirds(), 2, "Up charge_thirds");
    assert_eq!(QuarkFlavor::Charm.charge_thirds(), 2, "Charm charge_thirds");
    assert_eq!(QuarkFlavor::Top.charge_thirds(), 2, "Top charge_thirds");
}

#[test]
fn sm_down_type_quark_charges() {
    assert_eq!(QuarkFlavor::Down.charge_thirds(), -1, "Down charge_thirds");
    assert_eq!(
        QuarkFlavor::Strange.charge_thirds(),
        -1,
        "Strange charge_thirds"
    );
    assert_eq!(
        QuarkFlavor::Bottom.charge_thirds(),
        -1,
        "Bottom charge_thirds"
    );
}

#[test]
fn sm_quark_mass_up() {
    assert_relative_eq(QuarkFlavor::Up.mass_mev() as f64, 2.2, 1e-6, "Up mass");
}

#[test]
fn sm_quark_mass_down() {
    assert_relative_eq(QuarkFlavor::Down.mass_mev() as f64, 4.7, 1e-6, "Down mass");
}

#[test]
fn sm_quark_mass_charm() {
    assert_relative_eq(
        QuarkFlavor::Charm.mass_mev() as f64,
        1280.0,
        1e-6,
        "Charm mass",
    );
}

#[test]
fn sm_quark_mass_strange() {
    assert_relative_eq(
        QuarkFlavor::Strange.mass_mev() as f64,
        96.0,
        1e-6,
        "Strange mass",
    );
}

#[test]
fn sm_quark_mass_top() {
    assert_relative_eq(
        QuarkFlavor::Top.mass_mev() as f64,
        173100.0,
        1e-6,
        "Top mass",
    );
}

#[test]
fn sm_quark_mass_bottom() {
    assert_relative_eq(
        QuarkFlavor::Bottom.mass_mev() as f64,
        4180.0,
        1e-6,
        "Bottom mass",
    );
}

#[test]
fn sm_quark_generations() {
    assert_eq!(QuarkFlavor::Up.generation(), 1, "Up r#gen");
    assert_eq!(QuarkFlavor::Down.generation(), 1, "Down r#gen");
    assert_eq!(QuarkFlavor::Charm.generation(), 2, "Charm r#gen");
    assert_eq!(QuarkFlavor::Strange.generation(), 2, "Strange r#gen");
    assert_eq!(QuarkFlavor::Top.generation(), 3, "Top r#gen");
    assert_eq!(QuarkFlavor::Bottom.generation(), 3, "Bottom r#gen");
}

#[test]
fn sm_charged_lepton_charges() {
    assert_eq!(LeptonFlavor::Electron.charge(), -1, "Electron charge");
    assert_eq!(LeptonFlavor::Muon.charge(), -1, "Muon charge");
    assert_eq!(LeptonFlavor::Tau.charge(), -1, "Tau charge");
}

#[test]
fn sm_neutrino_charges_zero() {
    assert_eq!(LeptonFlavor::ElectronNeutrino.charge(), 0, "νe charge");
    assert_eq!(LeptonFlavor::MuonNeutrino.charge(), 0, "νμ charge");
    assert_eq!(LeptonFlavor::TauNeutrino.charge(), 0, "ντ charge");
}

#[test]
fn sm_lepton_masses() {
    assert_relative_eq(
        LeptonFlavor::Electron.mass_mev() as f64,
        0.511,
        1e-6,
        "Electron mass",
    );
    assert_relative_eq(
        LeptonFlavor::Muon.mass_mev() as f64,
        105.66,
        1e-6,
        "Muon mass",
    );
    assert_relative_eq(
        LeptonFlavor::Tau.mass_mev() as f64,
        1776.86,
        1e-6,
        "Tau mass",
    );
}

#[test]
fn sm_massless_bosons() {
    assert_relative_eq(
        GaugeBoson::Photon.mass_gev() as f64,
        0.0,
        1e-15,
        "Photon mass",
    );
    assert_relative_eq(
        GaugeBoson::Gluon.mass_gev() as f64,
        0.0,
        1e-15,
        "Gluon mass",
    );
    assert_relative_eq(
        GaugeBoson::Graviton.mass_gev() as f64,
        0.0,
        1e-15,
        "Graviton mass",
    );
}

#[test]
fn sm_w_z_boson_masses() {
    assert_relative_eq(GaugeBoson::WPlus.mass_gev() as f64, 80.379, 1e-6, "W+ mass");
    assert_relative_eq(
        GaugeBoson::WMinus.mass_gev() as f64,
        80.379,
        1e-6,
        "W- mass",
    );
    assert_relative_eq(GaugeBoson::Z.mass_gev() as f64, 91.1876, 1e-6, "Z mass");
}

#[test]
fn sm_gauge_boson_spins() {
    assert_eq!(GaugeBoson::Photon.spin(), 1, "Photon spin");
    assert_eq!(GaugeBoson::Gluon.spin(), 1, "Gluon spin");
    assert_eq!(GaugeBoson::WPlus.spin(), 1, "W+ spin");
    assert_eq!(GaugeBoson::WMinus.spin(), 1, "W- spin");
    assert_eq!(GaugeBoson::Z.spin(), 1, "Z spin");
    assert_eq!(GaugeBoson::Graviton.spin(), 2, "Graviton spin");
}

#[test]
fn sm_physics_dim() {
    assert_eq!(PHYSICS_DIM, 16384, "PHYSICS_DIM");
}

// =========================================================================
// Round 8: Section 57 — Hadron Properties (14 tests)
// =========================================================================

#[test]
fn hadron_proton_charge() {
    assert_eq!(Baryon::Proton.charge(), 1, "Proton charge");
}

#[test]
fn hadron_neutron_charge() {
    assert_eq!(Baryon::Neutron.charge(), 0, "Neutron charge");
}

#[test]
fn hadron_delta_pp_charge() {
    assert_eq!(Baryon::DeltaPlusPlus.charge(), 2, "Δ++ charge");
}

#[test]
fn hadron_omega_charge() {
    assert_eq!(Baryon::Omega.charge(), -1, "Ω⁻ charge");
}

#[test]
fn hadron_proton_mass() {
    assert_relative_eq(
        Baryon::Proton.mass_mev() as f64,
        938.272,
        1e-6,
        "Proton mass",
    );
}

#[test]
fn hadron_neutron_mass() {
    assert_relative_eq(
        Baryon::Neutron.mass_mev() as f64,
        939.565,
        1e-6,
        "Neutron mass",
    );
}

#[test]
fn hadron_delta_mass() {
    assert_relative_eq(
        Baryon::DeltaPlusPlus.mass_mev() as f64,
        1232.0,
        1e-6,
        "Δ++ mass",
    );
}

#[test]
fn hadron_lambda_mass() {
    assert_relative_eq(Baryon::Lambda.mass_mev() as f64, 1115.683, 1e-6, "Λ mass");
}

#[test]
fn hadron_nucleon_zero_strangeness() {
    assert_eq!(Baryon::Proton.strangeness(), 0, "Proton strangeness");
    assert_eq!(Baryon::Neutron.strangeness(), 0, "Neutron strangeness");
}

#[test]
fn hadron_omega_strangeness() {
    assert_eq!(Baryon::Omega.strangeness(), -3, "Ω⁻ strangeness");
}

#[test]
fn hadron_xi_strangeness() {
    assert_eq!(Baryon::XiZero.strangeness(), -2, "Ξ⁰ strangeness");
    assert_eq!(Baryon::XiMinus.strangeness(), -2, "Ξ⁻ strangeness");
}

#[test]
fn hadron_pion_charged_mass() {
    assert_relative_eq(Meson::PionPlus.mass_mev() as f64, 139.570, 1e-6, "π+ mass");
    assert_relative_eq(Meson::PionMinus.mass_mev() as f64, 139.570, 1e-6, "π- mass");
}

#[test]
fn hadron_kaon_mass() {
    assert_relative_eq(Meson::KaonPlus.mass_mev() as f64, 493.677, 1e-6, "K+ mass");
    assert_relative_eq(Meson::KaonMinus.mass_mev() as f64, 493.677, 1e-6, "K- mass");
}

#[test]
fn hadron_nucleon_similarity() {
    let (_, _, hadrons, _, _, _) = particle_physics_setup();
    let sim = hadrons.nucleon_similarity();
    assert!(sim > 0.3, "Nucleon similarity should be > 0.3, got {sim}");
}

// =========================================================================
// Round 8: Section 58 — Nuclear Physics (14 tests)
// =========================================================================

#[test]
fn nuclear_h2_binding() {
    let (_, _, _, _, nuclear, _) = particle_physics_setup();
    assert_relative_eq(
        nuclear.binding_energy_per_nucleon(2) as f64,
        1.11,
        1e-3,
        "H-2 binding energy per nucleon",
    );
}

#[test]
fn nuclear_he4_binding() {
    let (_, _, _, _, nuclear, _) = particle_physics_setup();
    assert_relative_eq(
        nuclear.binding_energy_per_nucleon(4) as f64,
        7.07,
        1e-3,
        "He-4 binding energy per nucleon",
    );
}

#[test]
fn nuclear_fe56_binding() {
    let (_, _, _, _, nuclear, _) = particle_physics_setup();
    assert_relative_eq(
        nuclear.binding_energy_per_nucleon(56) as f64,
        8.79,
        1e-3,
        "Fe-56 binding energy per nucleon",
    );
}

#[test]
fn nuclear_u238_binding() {
    let (_, _, _, _, nuclear, _) = particle_physics_setup();
    assert_relative_eq(
        nuclear.binding_energy_per_nucleon(238) as f64,
        7.57,
        1e-3,
        "U-238 binding energy per nucleon",
    );
}

#[test]
fn nuclear_iron_most_stable() {
    let (_, _, _, _, nuclear, _) = particle_physics_setup();
    let most_stable = nuclear.most_stable_mass_number();
    assert!(
        most_stable == 56 || most_stable == 62,
        "Most stable should be 56 or 62, got {most_stable}"
    );
}

#[test]
fn nuclear_dt_fusion_exothermic() {
    let (_, _, _, _, nuclear, _) = particle_physics_setup();
    let q = nuclear.fusion_q_value(2, 3, 4);
    assert!(q > 0.0, "D-T fusion should be exothermic, got Q={q}");
}

#[test]
fn nuclear_u235_fission_exothermic() {
    let (_, _, _, _, nuclear, _) = particle_physics_setup();
    let q = nuclear.fission_q_value(235, 140, 95);
    assert!(q > 0.0, "U-235 fission should be exothermic, got Q={q}");
}

#[test]
fn nuclear_iron_peak_no_fusion() {
    let (_, _, _, _, nuclear, _) = particle_physics_setup();
    let q = nuclear.fusion_q_value(56, 56, 112);
    assert!(
        q <= 0.0,
        "Iron-iron fusion should not be exothermic, got Q={q}"
    );
}

#[test]
fn nuclear_chemical_ev() {
    assert_relative_eq(
        EnergyScale::Chemical.typical_ev(),
        1.0,
        1e-10,
        "Chemical scale",
    );
}

#[test]
fn nuclear_scale_1mev() {
    assert_relative_eq(
        EnergyScale::Nuclear.typical_ev(),
        1_000_000.0,
        1e-10,
        "Nuclear scale",
    );
}

#[test]
fn nuclear_density_ratio() {
    assert_relative_eq(
        EnergyScale::Nuclear.density_ratio(),
        1_000_000.0,
        1e-10,
        "Nuclear density ratio",
    );
}

#[test]
fn nuclear_ta180m_excitation() {
    let (_, _, _, _, nuclear, _) = particle_physics_setup();
    let isomer = nuclear.get_isomer(73, 180).expect("Ta-180m should exist");
    assert_relative_eq(
        isomer.excitation_kev as f64,
        77.0,
        1e-6,
        "Ta-180m excitation",
    );
}

#[test]
fn nuclear_hf178m2_excitation() {
    let (_, _, _, _, nuclear, _) = particle_physics_setup();
    let isomer = nuclear.get_isomer(72, 178).expect("Hf-178m2 should exist");
    assert_relative_eq(
        isomer.excitation_kev as f64,
        2446.0,
        1e-6,
        "Hf-178m2 excitation",
    );
}

#[test]
fn nuclear_tc99m_excitation() {
    let (_, _, _, _, nuclear, _) = particle_physics_setup();
    let isomer = nuclear.get_isomer(43, 99).expect("Tc-99m should exist");
    assert_relative_eq(
        isomer.excitation_kev as f64,
        140.5,
        1e-6,
        "Tc-99m excitation",
    );
}

// =========================================================================
// Round 8: Section 59 — QED Loop Diagrams (8 tests)
// =========================================================================

#[test]
fn qed_bhabha_tree_level() {
    let (genesis, model, _, _, _, _) = particle_physics_setup();
    let qed = QEDEncoder::from_genesis(&genesis, &model);
    let bhabha = qed.bhabha_scattering();
    assert_eq!(bhabha.loop_order, 0, "Bhabha is tree-level");
}

#[test]
fn qed_bhabha_two_vertices() {
    let (genesis, model, _, _, _, _) = particle_physics_setup();
    let qed = QEDEncoder::from_genesis(&genesis, &model);
    let bhabha = qed.bhabha_scattering();
    assert_eq!(bhabha.vertices.len(), 2, "Bhabha has 2 vertices");
}

#[test]
fn qed_bhabha_amplitude_alpha() {
    let (genesis, model, _, _, _, _) = particle_physics_setup();
    let qed = QEDEncoder::from_genesis(&genesis, &model);
    let bhabha = qed.bhabha_scattering();
    assert_relative_eq(
        bhabha.amplitude_estimate,
        ALPHA,
        1e-6,
        "Bhabha amplitude ≈ α",
    );
}

#[test]
fn qed_compton_four_external() {
    let (genesis, model, _, _, _, _) = particle_physics_setup();
    let qed = QEDEncoder::from_genesis(&genesis, &model);
    let compton = qed.compton_scattering();
    assert_eq!(
        compton.external_legs.len(),
        4,
        "Compton has 4 external legs"
    );
}

#[test]
fn qed_electron_self_energy_uv() {
    let (genesis, model, _, _, _, _) = particle_physics_setup();
    let qed = QEDEncoder::from_genesis(&genesis, &model);
    let se = qed.electron_self_energy();
    assert!(
        matches!(se.divergence_type, DivergenceType::UVDivergent),
        "Electron self-energy is UV divergent"
    );
}

#[test]
fn qed_vacuum_polarization_uv() {
    let (genesis, model, _, _, _, _) = particle_physics_setup();
    let qed = QEDEncoder::from_genesis(&genesis, &model);
    let vp = qed.vacuum_polarization();
    assert!(
        matches!(vp.divergence_type, DivergenceType::UVDivergent),
        "Vacuum polarization is UV divergent"
    );
}

#[test]
fn qed_vertex_correction_finite() {
    let (genesis, model, _, _, _, _) = particle_physics_setup();
    let qed = QEDEncoder::from_genesis(&genesis, &model);
    let vc = qed.vertex_correction();
    assert!(
        matches!(vc.divergence_type, DivergenceType::Finite),
        "Vertex correction is finite"
    );
}

#[test]
fn qed_self_energy_has_counterterm() {
    let (genesis, model, _, _, _, _) = particle_physics_setup();
    let qed = QEDEncoder::from_genesis(&genesis, &model);
    let se = qed.electron_self_energy();
    assert!(
        se.counterterm.is_some(),
        "Electron self-energy should have a counterterm"
    );
}

// =========================================================================
// Round 8: Section 60 — QCD (8 tests)
// =========================================================================

#[test]
fn qcd_strong_coupling() {
    let (genesis, model, _, _, _, _) = particle_physics_setup();
    let qcd = QCDEncoder::from_genesis(&genesis, &model);
    assert_relative_eq(qcd.strong_coupling, 0.118, 1e-10, "Strong coupling αs");
}

#[test]
fn qcd_eight_color_charges() {
    let (genesis, model, _, _, _, _) = particle_physics_setup();
    let qcd = QCDEncoder::from_genesis(&genesis, &model);
    assert_eq!(qcd.color_charges.len(), 8, "8 color charges (Gell-Mann)");
}

#[test]
fn qcd_color_charges_distinct() {
    let (genesis, model, _, _, _, _) = particle_physics_setup();
    let qcd = QCDEncoder::from_genesis(&genesis, &model);
    for i in 0..8 {
        for j in (i + 1)..8 {
            let sim = qcd.color_charges[i].similarity(&qcd.color_charges[j]).abs();
            assert!(
                sim < 0.5,
                "Color charges {i} and {j} should be distinct, similarity = {sim}"
            );
        }
    }
}

#[test]
fn qcd_qqg_vertex_coupling() {
    let (genesis, model, _, _, _, _) = particle_physics_setup();
    let qcd = QCDEncoder::from_genesis(&genesis, &model);
    let vertex = qcd.qqg_vertex_with_color(0);
    let expected = 0.118_f64.sqrt();
    assert_relative_eq(vertex.coupling, expected, 1e-6, "qqg coupling = √αs");
}

#[test]
fn qcd_triple_gluon_3legs() {
    let (genesis, model, _, _, _, _) = particle_physics_setup();
    let qcd = QCDEncoder::from_genesis(&genesis, &model);
    let vertex = qcd.triple_gluon_vertex();
    let total_legs = vertex.incoming.len() + vertex.outgoing.len();
    assert_eq!(total_legs, 3, "Triple gluon vertex has 3 legs");
}

#[test]
fn qcd_quartic_gluon_coupling() {
    let (genesis, model, _, _, _, _) = particle_physics_setup();
    let qcd = QCDEncoder::from_genesis(&genesis, &model);
    let vertex = qcd.quartic_gluon_vertex();
    assert_relative_eq(vertex.coupling, 0.118, 1e-10, "Quartic gluon coupling = αs");
}

#[test]
fn qcd_parton_shower_3emissions() {
    let (genesis, model, _, _, _, _) = particle_physics_setup();
    let qcd = QCDEncoder::from_genesis(&genesis, &model);
    let quark = model.up_quark.clone();
    let shower = qcd.parton_shower(&quark, 3);
    assert_eq!(shower.len(), 7, "3 emissions → 1 + 2*3 = 7 partons");
}

#[test]
fn qcd_parton_shower_1emission() {
    let (genesis, model, _, _, _, _) = particle_physics_setup();
    let qcd = QCDEncoder::from_genesis(&genesis, &model);
    let quark = model.up_quark.clone();
    let shower = qcd.parton_shower(&quark, 1);
    assert_eq!(shower.len(), 3, "1 emission → 1 + 2*1 = 3 partons");
}

// =========================================================================
// Round 8: Section 61 — Electroweak (6 tests)
// =========================================================================

#[test]
fn ew_weinberg_angle() {
    let (genesis, model, _, _, _, _) = particle_physics_setup();
    let ew = ElectroweakEncoder::from_genesis(&genesis, &model);
    assert_relative_eq(ew.weinberg_angle, 0.23122, 1e-6, "Weinberg angle sin²θW");
}

#[test]
fn ew_beta_decay_tree_level() {
    let (genesis, model, hadrons, _, _, _) = particle_physics_setup();
    let ew = ElectroweakEncoder::from_genesis(&genesis, &model);
    let diagram = ew.beta_decay(&hadrons, &model);
    assert_eq!(diagram.loop_order, 0, "Beta decay is tree-level");
}

#[test]
fn ew_beta_decay_weak_amplitude() {
    let (genesis, model, hadrons, _, _, _) = particle_physics_setup();
    let ew = ElectroweakEncoder::from_genesis(&genesis, &model);
    let diagram = ew.beta_decay(&hadrons, &model);
    assert!(
        diagram.amplitude_estimate < 1e-4,
        "Beta decay amplitude should be < 1e-4 (weak), got {}",
        diagram.amplitude_estimate
    );
}

#[test]
fn ew_higgs_bb_positive_amplitude() {
    let (genesis, model, _, _, _, _) = particle_physics_setup();
    let ew = ElectroweakEncoder::from_genesis(&genesis, &model);
    let diagram = ew.higgs_to_bb(&model);
    assert!(
        diagram.amplitude_estimate > 0.0,
        "H→bb̄ amplitude should be positive, got {}",
        diagram.amplitude_estimate
    );
}

#[test]
fn ew_beta_decay_name() {
    let (genesis, model, hadrons, _, _, _) = particle_physics_setup();
    let ew = ElectroweakEncoder::from_genesis(&genesis, &model);
    let diagram = ew.beta_decay(&hadrons, &model);
    assert_eq!(diagram.name, "Beta decay", "Beta decay name");
}

#[test]
fn ew_w_pair_production_name() {
    let (genesis, model, _, _, _, _) = particle_physics_setup();
    let ew = ElectroweakEncoder::from_genesis(&genesis, &model);
    let diagram = ew.w_pair_production(&model);
    assert_eq!(diagram.name, "e⁺e⁻ → W⁺W⁻", "W pair production name");
}

// =========================================================================
// Round 8: Section 62 — Antimatter Expanded (8 tests)
// =========================================================================

#[test]
fn am_antideuterium_composition() {
    let (_, _, _, _, _, antimatter) = particle_physics_setup();
    let ad = antimatter.antideuterium();
    assert_eq!(ad.antiprotons, 1, "Antideuterium antiprotons");
    assert_eq!(ad.antineutrons, 1, "Antideuterium antineutrons");
}

#[test]
fn am_antihelium3_composition() {
    let (_, _, _, _, _, antimatter) = particle_physics_setup();
    let ahe3 = antimatter.antihelium3();
    assert_eq!(ahe3.antiprotons, 2, "Antihelium-3 antiprotons");
    assert_eq!(ahe3.antineutrons, 1, "Antihelium-3 antineutrons");
}

#[test]
fn am_antihelium4_composition() {
    let (_, _, _, _, _, antimatter) = particle_physics_setup();
    let ahe4 = antimatter.antihelium4();
    assert_eq!(ahe4.antiprotons, 2, "Antihelium-4 antiprotons");
    assert_eq!(ahe4.antineutrons, 2, "Antihelium-4 antineutrons");
    assert_eq!(ahe4.positrons, 2, "Antihelium-4 positrons");
}

#[test]
fn am_compose_antiatom_positrons() {
    let (_, _, _, _, _, antimatter) = particle_physics_setup();
    let antiatom = antimatter.compose_antiatom(3, 4);
    assert_eq!(
        antiatom.positrons, 3,
        "Antiatom with Z=3 should have 3 positrons"
    );
}

#[test]
fn am_positron_orthogonal_electron() {
    let (_, model, _, _, _, antimatter) = particle_physics_setup();
    let sim = model.electron.similarity(&antimatter.positron).abs();
    assert!(
        sim < 0.3,
        "Positron and electron should be near-orthogonal, got {sim}"
    );
}

#[test]
fn am_pair_production_energy() {
    let (_, model, _, _, _, antimatter) = particle_physics_setup();
    let event = antimatter.pair_production_event(&model);
    assert_relative_eq(event.energy_mev, 1.022, 1e-6, "Pair production threshold");
}

#[test]
fn am_sakharov_conditions_nonzero() {
    let genesis = GenesisSeed::from_phrase("r8_particle_physics");
    let concepts = BaryogenesisConcepts::from_genesis(&genesis);
    let sakharov = concepts.encode_sakharov();
    let norm = sakharov.norm();
    assert!(
        norm > 0.0,
        "Sakharov conditions vector should be nonzero, norm = {norm}"
    );
}

#[test]
fn am_antiproton_dim() {
    let (_, _, _, _, _, antimatter) = particle_physics_setup();
    assert_eq!(
        antimatter.antiproton.dim(),
        PHYSICS_DIM,
        "Antiproton dim = PHYSICS_DIM"
    );
}

// =========================================================================
// Round 8: Section 63 — Cross-Particle Integration (6 tests)
// =========================================================================

#[test]
fn cross_sm_antiparticle_is_conjugate() {
    let (_, model, _, _, _, _) = particle_physics_setup();
    let anti_from_model = model.antiparticle(&model.electron);
    let anti_from_conjugate = Antimatter::conjugate(&model.electron);
    let sim = anti_from_model.similarity(&anti_from_conjugate);
    assert!(
        sim > 0.99,
        "Model antiparticle and conjugate should agree, similarity = {sim}"
    );
}

#[test]
fn cross_beta_decay_proton_match() {
    let (_, model, hadrons, _, _, _) = particle_physics_setup();
    let (decay_proton, _, _) = hadrons.beta_decay(&model);
    let sim = decay_proton.similarity(&hadrons.proton);
    assert!(
        sim > 0.99,
        "Beta decay proton should match hadron proton, similarity = {sim}"
    );
}

#[test]
fn cross_mass_hierarchy_top_electron() {
    let ratio = QuarkFlavor::Top.mass_mev() as f64 / LeptonFlavor::Electron.mass_mev() as f64;
    assert!(
        ratio > 300_000.0,
        "Top/electron mass ratio should be > 300000, got {ratio}"
    );
}

#[test]
fn cross_energy_hierarchy_scales() {
    let chemical = EnergyScale::Chemical.typical_ev();
    let isomeric = EnergyScale::Isomeric.typical_ev();
    let nuclear = EnergyScale::Nuclear.typical_ev();
    assert!(
        nuclear > isomeric,
        "Nuclear > Isomeric: {nuclear} > {isomeric}"
    );
    assert!(
        isomeric > chemical,
        "Isomeric > Chemical: {isomeric} > {chemical}"
    );
}

#[test]
fn cross_binding_curve_monotone_to_iron() {
    let (_, _, _, _, nuclear, _) = particle_physics_setup();
    let be4 = nuclear.binding_energy_per_nucleon(4);
    let be12 = nuclear.binding_energy_per_nucleon(12);
    let be56 = nuclear.binding_energy_per_nucleon(56);
    assert!(be4 < be12, "BE(4) < BE(12): {be4} < {be12}");
    assert!(be12 < be56, "BE(12) < BE(56): {be12} < {be56}");
}

#[test]
fn cross_hadron_vectors_correct_dim() {
    let (_, _, hadrons, _, _, _) = particle_physics_setup();
    assert_eq!(hadrons.proton.dim(), PHYSICS_DIM, "Proton dim");
    assert_eq!(hadrons.neutron.dim(), PHYSICS_DIM, "Neutron dim");
    assert_eq!(hadrons.pion_plus.dim(), PHYSICS_DIM, "Pion dim");
}
