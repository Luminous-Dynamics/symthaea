// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase Q6a: energy-component decomposition for the three known HF
//! discrepancies (N2/STO-3G, H2O/6-31G, CH4/6-31G), for direct comparison
//! against pyscf's own component breakdown (computed separately via
//! `nix develop .#qc-verify`, same geometries copied verbatim from
//! `validation::benchmark_molecules`).
//!
//! Reuses only already-tested, already-public crate functions
//! (`overlap_matrix`, `kinetic_matrix`, `nuclear_matrix`, `compute_eri_tensor`,
//! `build_density_matrix`, `build_fock_matrix`) to rebuild the converged
//! density/Fock from `RhfResult`'s public fields -- no new SCF logic.
//!
//! **Diagnostic only.**

use symthaea_quantum_chemistry::basis::BasisSetProvider;
use symthaea_quantum_chemistry::basis::basis_631g::Basis631G;
use symthaea_quantum_chemistry::basis::sto3g::Sto3g;
use symthaea_quantum_chemistry::integrals::eri::compute_eri_tensor;
use symthaea_quantum_chemistry::integrals::kinetic::kinetic_matrix;
use symthaea_quantum_chemistry::integrals::nuclear::nuclear_matrix;
use symthaea_quantum_chemistry::integrals::overlap::overlap_matrix;
use symthaea_quantum_chemistry::molecule::{Atom, Molecule};
use symthaea_quantum_chemistry::scf::density::build_density_matrix;
use symthaea_quantum_chemistry::scf::fock::build_fock_matrix;
use symthaea_quantum_chemistry::{RhfConfig, RhfResult, restricted_hartree_fock};

fn report(name: &str, mol: &Molecule, basis: &symthaea_quantum_chemistry::BasisSet) {
    let rhf: RhfResult = restricted_hartree_fock(mol, basis, &RhfConfig::default());
    let n = rhf.n_basis;

    let s = overlap_matrix(&basis.functions);
    let t = kinetic_matrix(&basis.functions);
    let v = nuclear_matrix(&basis.functions, &mol.atoms);
    let mut h_core = vec![0.0; n * n];
    for i in 0..n * n {
        h_core[i] = t[i] + v[i];
    }
    let (eri, _, _) = compute_eri_tensor(&basis.functions);

    let density = build_density_matrix(
        &rhf.orbital_coefficients,
        n,
        rhf.n_independent,
        rhf.n_occupied,
    );
    let fock = build_fock_matrix(&h_core, &density, &eri, n);

    // one_electron = Tr[P . H_core]; two_electron = 0.5 * Tr[P . (F - H_core)]
    let mut one_electron = 0.0;
    let mut two_electron = 0.0;
    for i in 0..n * n {
        one_electron += density[i] * h_core[i];
        two_electron += density[i] * (fock[i] - h_core[i]);
    }
    two_electron *= 0.5;

    let e_elec = one_electron + two_electron;

    println!("=== {name} ===");
    println!(
        "  nao={} nelec={} converged={} n_iter={}",
        n,
        mol.n_electrons(),
        rhf.converged,
        rhf.n_iterations
    );
    // Sanity: overlap diagonal should be exactly 1.0 for normalized basis functions.
    let s_diag_max_dev = (0..n)
        .map(|i| (s[i * n + i] - 1.0).abs())
        .fold(0.0_f64, f64::max);
    println!("  max|S_ii - 1| = {s_diag_max_dev:.2e}");
    println!("  E_nuc  = {:.10}", rhf.nuclear_repulsion);
    println!("  E_one  = {one_electron:.10}");
    println!("  E_two  = {two_electron:.10}");
    println!(
        "  E_elec = {e_elec:.10}  (RhfResult.electronic_energy = {:.10})",
        rhf.electronic_energy
    );
    println!(
        "  E_tot  = {:.10}  (RhfResult.total_energy = {:.10})",
        e_elec + rhf.nuclear_repulsion,
        rhf.total_energy
    );
    println!();
}

fn main() {
    // N2/STO-3G, R=2.074 Bohr -- exact match to validation::benchmark_molecules
    let n2 = Molecule::new(vec![
        Atom::new(7, 0.0, 0.0, 0.0),
        Atom::new(7, 0.0, 0.0, 2.074),
    ]);
    let basis_n2 = Sto3g::build(&n2);
    report("N2/STO-3G", &n2, &basis_n2);

    // H2O/6-31G, experimental geometry -- exact match to Molecule::water()
    let h2o = Molecule::water();
    let basis_h2o = Basis631G::build(&h2o);
    report("H2O/6-31G", &h2o, &basis_h2o);

    // CH4/6-31G -- exact match to validation::benchmark_molecules
    let ch4 = Molecule::new(vec![
        Atom::new(6, 0.0, 0.0, 0.0),
        Atom::new(1, 1.185, 1.185, 1.185),
        Atom::new(1, -1.185, -1.185, 1.185),
        Atom::new(1, -1.185, 1.185, -1.185),
        Atom::new(1, 1.185, -1.185, -1.185),
    ]);
    let basis_ch4 = Basis631G::build(&ch4);
    report("CH4/6-31G", &ch4, &basis_ch4);
}
