// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-organic-chemistry
//!
//! A self-contained structural organic-chemistry layer for Symthaea: parse
//! SMILES into a molecular graph, then compute molecular formula (Hill
//! notation), molecular weight, and detect functional groups.
//!
//! This fills a confirmed gap — the workspace had ab-initio electronic-structure
//! chemistry (`symthaea-quantum-chemistry`: Hartree-Fock, DFT) but **no
//! structural layer**: no SMILES parser, no molecular graph, no functional-group
//! recognition. Reactions were only abstract HDC transforms.
//!
//! ## Scope
//!
//! - SMILES: organic-subset + bracket atoms, single/double/triple/aromatic
//!   bonds, branches, ring closures, common neutral aromatics.
//! - Derived properties: Hill formula, standard-atomic-weight molecular weight,
//!   implicit-hydrogen counts, elemental mass composition.
//! - Structure: functional-group detection (hydroxyl, carbonyl, carboxyl, ester,
//!   ether, amine, amide, nitrile, halide, aromatic ring), ring perception
//!   (cyclomatic count), degree of unsaturation.
//! - Cheminformatics: H-bond donors/acceptors, Lipinski Rule-of-Five
//!   drug-likeness (logP omitted).
//!
//! Stereochemistry SMILES syntax (`@`/`@@`/`/`/`\`) is tolerated -- parsed
//! and discarded, not modelled (Phase A.6, 2026-07-16; see `smiles.rs`'s
//! module doc for why). Not yet: isotopes, disconnected structures, logP,
//! reaction mechanisms / retrosynthesis (the intended next direction).
//!
//! ## Example
//!
//! ```
//! use symthaea_organic_chemistry::{Molecule, groups::{detect, FunctionalGroup}};
//!
//! let ethanol = Molecule::from_smiles("CCO").unwrap();
//! assert_eq!(ethanol.molecular_formula(), "C2H6O");
//! assert!((ethanol.molecular_weight() - 46.069).abs() < 1e-3);
//! assert_eq!(detect(&ethanol), vec![FunctionalGroup::Hydroxyl]);
//! ```

pub mod analysis;
pub mod element;
pub mod groups;
pub mod properties;
pub mod smiles;

pub use properties::{Lipinski, lipinski};

pub use element::{Element, is_organic_subset, lookup};
pub use groups::{FunctionalGroup, detect};
pub use smiles::{Atom, Bond, BondOrder, Molecule, ParseError};

#[cfg(test)]
mod integration_tests {
    use super::*;

    /// Parse and assert Hill formula + molecular weight against known values.
    fn check(smiles: &str, formula: &str, mw: f64) {
        let m = Molecule::from_smiles(smiles)
            .unwrap_or_else(|e| panic!("failed to parse {smiles}: {e}"));
        assert_eq!(
            m.molecular_formula(),
            formula,
            "formula mismatch for {smiles}"
        );
        assert!(
            (m.molecular_weight() - mw).abs() < 1e-2,
            "MW mismatch for {smiles}: got {:.4}, want {:.4}",
            m.molecular_weight(),
            mw
        );
    }

    #[test]
    fn methane() {
        check("C", "CH4", 16.043);
    }

    #[test]
    fn water() {
        check("O", "H2O", 18.015);
    }

    #[test]
    fn ethanol() {
        check("CCO", "C2H6O", 46.069);
    }

    #[test]
    fn carbon_dioxide() {
        check("O=C=O", "CO2", 44.009);
    }

    #[test]
    fn acetic_acid() {
        check("CC(=O)O", "C2H4O2", 60.052);
    }

    #[test]
    fn acetonitrile() {
        check("CC#N", "C2H3N", 41.053);
    }

    #[test]
    fn benzene_ring_closure_and_aromatic_h() {
        // Exercises ring closure + aromatic implicit-H (each aromatic C → 1 H).
        check("c1ccccc1", "C6H6", 78.114);
    }

    #[test]
    fn chloromethane_two_letter_halogen() {
        check("CCl", "CH3Cl", 50.487);
    }

    #[test]
    fn bracket_atom_explicit_hydrogens() {
        // Ammonium: [NH4+] — bracket H count authoritative, charge respected.
        let m = Molecule::from_smiles("[NH4+]").unwrap();
        assert_eq!(m.molecular_formula(), "H4N");
        assert_eq!(m.atoms[0].charge, 1);
        assert_eq!(m.atoms[0].hydrogens, 4);
    }

    #[test]
    fn total_atom_count_includes_hydrogens() {
        let m = Molecule::from_smiles("CCO").unwrap();
        assert_eq!(m.total_atom_count(), 9); // C2 O1 + H6
    }

    #[test]
    fn unsupported_feature_errors_cleanly() {
        // Phase A.6 (2026-07-16): stereochemistry syntax is now tolerated
        // (parsed, geometric information discarded), not rejected -- see
        // smiles.rs's module doc and its own dedicated tests for why. This
        // test now covers what's still genuinely out of scope.
        assert!(Molecule::from_smiles("F/C=C/F").is_ok());
        // Disconnected structures remain out of scope.
        assert!(Molecule::from_smiles("CC.CC").is_err());
    }

    #[test]
    fn branches_and_isobutane() {
        // CC(C)C — isobutane, C4H10.
        check("CC(C)C", "C4H10", 58.122);
    }
}
