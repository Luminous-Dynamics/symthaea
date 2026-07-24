// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase 0 audit for `symthaea/CHEMICAL_PROCESS_DISCOVERY_PLAN_2026-07-12.md`.
//!
//! Not part of the library API. Prints honest numbers for:
//! 1. The existing HF/STO-3G and HF/6-31G benchmark suite (absolute energy error).
//! 2. HCN <-> HNC isomerization energy vs. the known experimental value (~15 kcal/mol).
//! 3. The water-gas-shift reaction CO + H2O -> CO2 + H2 vs. its known ~-9.8 kcal/mol
//!    reaction enthalpy (a real industrial process: H2 production / ammonia feed prep).
//!
//! Caveat printed inline: these are electronic-energy differences (no ZPE, no thermal
//! correction), compared against experimental enthalpies -- an approximate comparison,
//! but sufficient to tell whether the engine gets sign and rough magnitude right.

use symthaea_quantum_chemistry::basis::BasisSetProvider;
use symthaea_quantum_chemistry::basis::sto3g::Sto3g;
use symthaea_quantum_chemistry::geometry_opt::{GeomOptConfig, optimize_geometry};
use symthaea_quantum_chemistry::integrals::eri::compute_eri_tensor;
use symthaea_quantum_chemistry::molecule::{Atom, Molecule};
use symthaea_quantum_chemistry::post_hf::mp2::mp2_correlation_energy;
use symthaea_quantum_chemistry::scf::rhf::{RhfConfig, restricted_hartree_fock};
use symthaea_quantum_chemistry::validation::{validate_hf_631g, validate_hf_sto3g};

const HARTREE_TO_KCAL: f64 = 627.509;

fn single_point(mol: &Molecule, label: &str) -> f64 {
    let basis = Sto3g::build(mol);
    let result = restricted_hartree_fock(mol, &basis, &RhfConfig::default());
    println!(
        "    {label} (single-point @ literature geometry): E = {:.6} Ha",
        result.total_energy
    );
    result.total_energy
}

/// HF + MP2 correlation, at the same literature geometry `single_point` uses.
fn mp2_energy(mol: &Molecule, label: &str) -> f64 {
    let basis = Sto3g::build(mol);
    let rhf = restricted_hartree_fock(mol, &basis, &RhfConfig::default());
    let (eri, _, _) = compute_eri_tensor(&basis.functions);
    let mp2 = mp2_correlation_energy(&rhf, &eri);
    println!(
        "    {label} (MP2 @ literature geometry): E_HF = {:.6} Ha, E_corr = {:.6} Ha, E_MP2 = {:.6} Ha",
        mp2.hf_energy, mp2.correlation_energy, mp2.total_energy
    );
    mp2.total_energy
}

fn opt_energy(mol: &Molecule, label: &str) -> f64 {
    let result = optimize_geometry(mol, &GeomOptConfig::default());
    println!(
        "    {label}: E = {:.6} Ha, converged={}, steps={}, rms_grad={:.2e}",
        result.energy, result.converged, result.n_steps, result.rms_gradient
    );
    result.energy
}

fn main() {
    println!("=== 1. Existing benchmark suite (absolute energies vs. published refs) ===");
    for r in validate_hf_sto3g() {
        println!(
            "  STO-3G {:<6} computed={:.6} ref={:.6} error={:+.2} kcal/mol (tol={:.2} Ha) pass={}",
            r.molecule, r.computed, r.reference, r.error_kcal, r.tolerance, r.pass
        );
    }
    for r in validate_hf_631g() {
        println!(
            "  6-31G  {:<6} computed={:.6} ref={:.6} error={:+.2} kcal/mol (tol={:.2} Ha) pass={}",
            r.molecule, r.computed, r.reference, r.error_kcal, r.tolerance, r.pass
        );
    }

    println!("\n=== 2. HCN <-> HNC isomerization (experimental: HNC ~15.2 kcal/mol above HCN) ===");
    // Linear H-C#N: H at -1.065 A, C at 0, N at +1.153 A (literature equilibrium bond lengths).
    let hcn = Molecule::new(vec![
        Atom::from_angstrom(1, 0.0, 0.0, -1.065),
        Atom::from_angstrom(6, 0.0, 0.0, 0.0),
        Atom::from_angstrom(7, 0.0, 0.0, 1.153),
    ]);
    // Linear H-N#C: H at -0.994 A, N at 0, C at +1.169 A.
    let hnc = Molecule::new(vec![
        Atom::from_angstrom(1, 0.0, 0.0, -0.994),
        Atom::from_angstrom(7, 0.0, 0.0, 0.0),
        Atom::from_angstrom(6, 0.0, 0.0, 1.169),
    ]);
    // Diagnostic: is the blowup specific to DIIS extrapolation, or does plain Roothaan also diverge?
    for (mol, label) in [(&hcn, "HCN"), (&hnc, "HNC")] {
        let basis = Sto3g::build(mol);
        let no_diis_cfg = RhfConfig {
            use_diis: false,
            ..RhfConfig::default()
        };
        let result = restricted_hartree_fock(mol, &basis, &no_diis_cfg);
        println!(
            "    {label} (single-point, DIIS OFF): E = {:.6} Ha",
            result.total_energy
        );
    }

    let sp_hcn = single_point(&hcn, "HCN");
    let sp_hnc = single_point(&hnc, "HNC");
    let sp_delta_kcal = (sp_hnc - sp_hcn) * HARTREE_TO_KCAL;
    println!(
        "  [single-point] Delta E(HNC - HCN) = {:+.2} kcal/mol  [experimental: +15.2 kcal/mol]  error={:+.2} kcal/mol",
        sp_delta_kcal,
        sp_delta_kcal - 15.2
    );
    let e_hcn = opt_energy(&hcn, "HCN");
    let e_hnc = opt_energy(&hnc, "HNC");
    let delta_kcal = (e_hnc - e_hcn) * HARTREE_TO_KCAL;
    println!(
        "  [geom-opt]     Delta E(HNC - HCN) = {:+.2} kcal/mol  [experimental: +15.2 kcal/mol]  error={:+.2} kcal/mol",
        delta_kcal,
        delta_kcal - 15.2
    );

    println!("\n=== 3. Water-gas shift: CO + H2O -> CO2 + H2 (experimental: -9.8 kcal/mol) ===");
    let co = Molecule::new(vec![
        Atom::from_angstrom(6, 0.0, 0.0, 0.0),
        Atom::from_angstrom(8, 0.0, 0.0, 1.128),
    ]);
    let co2 = Molecule::new(vec![
        Atom::from_angstrom(8, 0.0, 0.0, -1.160),
        Atom::from_angstrom(6, 0.0, 0.0, 0.0),
        Atom::from_angstrom(8, 0.0, 0.0, 1.160),
    ]);
    let h2o = Molecule::water();
    let h2 = Molecule::h2();

    let sp_co = single_point(&co, "CO");
    let sp_h2o = single_point(&h2o, "H2O");
    let sp_co2 = single_point(&co2, "CO2");
    let sp_h2 = single_point(&h2, "H2");
    let sp_rxn_kcal = ((sp_co2 + sp_h2) - (sp_co + sp_h2o)) * HARTREE_TO_KCAL;
    println!(
        "  [single-point] Delta E(rxn) = {:+.2} kcal/mol  [experimental Delta H: -9.8 kcal/mol]  error={:+.2} kcal/mol, same sign={}",
        sp_rxn_kcal,
        sp_rxn_kcal - (-9.8),
        sp_rxn_kcal.signum() == -1.0
    );

    let e_co = opt_energy(&co, "CO");
    let e_h2o = opt_energy(&h2o, "H2O");
    let e_co2 = opt_energy(&co2, "CO2");
    let e_h2 = opt_energy(&h2, "H2");

    let rxn_kcal = ((e_co2 + e_h2) - (e_co + e_h2o)) * HARTREE_TO_KCAL;
    println!(
        "  [geom-opt]     Delta E(rxn) = {:+.2} kcal/mol  [experimental Delta H: -9.8 kcal/mol]  error={:+.2} kcal/mol, same sign={}",
        rxn_kcal,
        rxn_kcal - (-9.8),
        rxn_kcal.signum() == -1.0
    );

    let mp2_co = mp2_energy(&co, "CO");
    let mp2_h2o = mp2_energy(&h2o, "H2O");
    let mp2_co2 = mp2_energy(&co2, "CO2");
    let mp2_h2 = mp2_energy(&h2, "H2");
    let mp2_rxn_kcal = ((mp2_co2 + mp2_h2) - (mp2_co + mp2_h2o)) * HARTREE_TO_KCAL;
    println!(
        "  [MP2]          Delta E(rxn) = {:+.2} kcal/mol  [experimental Delta H: -9.8 kcal/mol]  error={:+.2} kcal/mol, same sign={}",
        mp2_rxn_kcal,
        mp2_rxn_kcal - (-9.8),
        mp2_rxn_kcal.signum() == -1.0
    );

    println!("\n=== Known scope limits this audit surfaced (not measured, just flagging) ===");
    println!("  - No UHF/ROHF: open-shell species (O2 triplet, most radicals/catalytic");
    println!("    intermediates) cannot be correctly described. Combustion/oxidation");
    println!("    chemistry involving O2 is out of scope until open-shell support exists.");
    println!("  - reaction_consciousness.rs has no transition-state search: its 3 scans");
    println!("    (h2_dissociation, h2o_symmetric_stretch, lih_dissociation) are hand-coded");
    println!("    1-D bond-stretch curves, not a general reaction-coordinate or barrier tool.");
    println!("  - Basis sets only cover H, C, N, O, F (both STO-3G and 6-31G).");
}
