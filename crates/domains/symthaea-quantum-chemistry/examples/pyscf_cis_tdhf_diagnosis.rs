// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase Q6b: CIS/TDHF excitation energies for direct comparison against
//! pyscf's `tdscf.TDA` (CIS) and `tdscf.TDHF` (RPA), computed separately via
//! `nix develop .#qc-verify`, same geometries copied verbatim from
//! `validation::benchmark_molecules`/`Molecule::water()`/`Molecule::h2()`.
//!
//! Reuses only already-tested, already-public crate functions -- no new
//! physics. **Diagnostic only.**

use symthaea_quantum_chemistry::basis::BasisSetProvider;
use symthaea_quantum_chemistry::basis::sto3g::Sto3g;
use symthaea_quantum_chemistry::integrals::eri::compute_eri_tensor;
use symthaea_quantum_chemistry::molecule::{Atom, Molecule};
use symthaea_quantum_chemistry::time_dependent::cis_excitations;
use symthaea_quantum_chemistry::{RhfConfig, restricted_hartree_fock};

fn report(
    name: &str,
    mol: &Molecule,
    basis: &symthaea_quantum_chemistry::BasisSet,
    n_states: usize,
) {
    let rhf = restricted_hartree_fock(mol, basis, &RhfConfig::default());
    let (eri, _, _) = compute_eri_tensor(&basis.functions);
    let states = cis_excitations(
        &rhf,
        &eri,
        n_states.min(rhf.n_independent * rhf.n_independent),
    );
    println!("=== {name} (RHF E_tot={:.8}) ===", rhf.total_energy);
    for (i, s) in states.iter().enumerate() {
        println!(
            "  CIS/TDA state {i}: E={:.8} Ha  dominant {}->{} w={:.4}",
            s.energy, s.dominant_from_orbital, s.dominant_to_orbital, s.dominant_weight
        );
    }
    println!();
}

fn main() {
    let h2o = Molecule::water();
    report("H2O/STO-3G", &h2o, &Sto3g::build(&h2o), 6);

    let h2 = Molecule::h2();
    report("H2/STO-3G", &h2, &Sto3g::build(&h2), 1);

    let nh3 = Molecule::new(vec![
        Atom::new(7, 0.0, 0.0, 0.0),
        Atom::new(1, 1.7715, 0.0, -0.65),
        Atom::new(1, -0.8858, 1.5344, -0.65),
        Atom::new(1, -0.8858, -1.5344, -0.65),
    ]);
    report("NH3/STO-3G", &nh3, &Sto3g::build(&nh3), 6);

    let ch4 = Molecule::new(vec![
        Atom::new(6, 0.0, 0.0, 0.0),
        Atom::new(1, 1.185, 1.185, 1.185),
        Atom::new(1, -1.185, -1.185, 1.185),
        Atom::new(1, -1.185, 1.185, -1.185),
        Atom::new(1, 1.185, -1.185, -1.185),
    ]);
    report("CH4/STO-3G", &ch4, &Sto3g::build(&ch4), 6);

    let lih = Molecule::new(vec![
        Atom::new(3, 0.0, 0.0, 0.0),
        Atom::new(1, 0.0, 0.0, 3.015),
    ]);
    report("LiH/STO-3G", &lih, &Sto3g::build(&lih), 4);

    let n2 = Molecule::new(vec![
        Atom::new(7, 0.0, 0.0, 0.0),
        Atom::new(7, 0.0, 0.0, 2.074),
    ]);
    report("N2/STO-3G", &n2, &Sto3g::build(&n2), 6);
}
