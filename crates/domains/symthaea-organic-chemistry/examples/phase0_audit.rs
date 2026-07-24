// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phase 0 audit for `symthaea/CHEMICAL_PROCESS_DISCOVERY_PLAN_2026-07-12.md`.
//!
//! Not part of the library API. Parses a set of real industrial feedstocks/
//! intermediates by SMILES and checks molecular formula, molecular weight, and
//! functional-group detection against known values -- the representation layer
//! any future process-discovery generator would sit on top of.

use symthaea_organic_chemistry::groups::detect;
use symthaea_organic_chemistry::smiles::Molecule;

struct Case {
    name: &'static str,
    smiles: &'static str,
    expected_formula: &'static str,
    expected_mw: f64,
    expected_groups: &'static [&'static str],
}

fn main() {
    let cases = [
        Case {
            name: "ethylene (world's highest-volume organic feedstock)",
            smiles: "C=C",
            expected_formula: "C2H4",
            expected_mw: 28.05,
            expected_groups: &[],
        },
        Case {
            name: "propylene",
            smiles: "CC=C",
            expected_formula: "C3H6",
            expected_mw: 42.08,
            expected_groups: &[],
        },
        Case {
            name: "benzene",
            smiles: "c1ccccc1",
            expected_formula: "C6H6",
            expected_mw: 78.11,
            expected_groups: &["AromaticRing"],
        },
        Case {
            name: "ethanol (bio-ethanol / fermentation feedstock)",
            smiles: "CCO",
            expected_formula: "C2H6O",
            expected_mw: 46.07,
            expected_groups: &["Hydroxyl"],
        },
        Case {
            name: "acetic acid (vinyl acetate / PTA precursor)",
            smiles: "CC(=O)O",
            expected_formula: "C2H4O2",
            expected_mw: 60.05,
            expected_groups: &["Carboxyl"],
        },
        Case {
            name: "methanol (syngas-derived feedstock)",
            smiles: "CO",
            expected_formula: "CH4O",
            expected_mw: 32.04,
            expected_groups: &["Hydroxyl"],
        },
        Case {
            name: "acrylonitrile (Andrussow/BMA process product -- ties directly to the HCN chemistry from Phase 0's quantum-chemistry audit)",
            smiles: "C=CC#N",
            expected_formula: "C3H3N",
            expected_mw: 53.06,
            expected_groups: &["Nitrile"],
        },
        Case {
            name: "phenol (bisphenol-A / resin precursor)",
            smiles: "c1ccc(cc1)O",
            expected_formula: "C6H6O",
            expected_mw: 94.11,
            expected_groups: &["Hydroxyl", "AromaticRing"],
        },
        Case {
            name: "adipic acid (nylon-6,6 precursor)",
            smiles: "OC(=O)CCCCC(=O)O",
            expected_formula: "C6H10O4",
            expected_mw: 146.14,
            expected_groups: &["Carboxyl"],
        },
        Case {
            name: "caprolactam (nylon-6 precursor)",
            smiles: "O=C1CCCCCN1",
            expected_formula: "C6H11NO",
            expected_mw: 113.16,
            expected_groups: &["Amide"],
        },
    ];

    let mut n_pass = 0;
    let mut n_fail = 0;

    for case in &cases {
        println!("=== {} ({}) ===", case.name, case.smiles);
        match Molecule::from_smiles(case.smiles) {
            Ok(mol) => {
                let formula = mol.molecular_formula();
                let mw = mol.molecular_weight();
                let groups = detect(&mol);
                let group_names: Vec<String> = groups.iter().map(|g| format!("{g:?}")).collect();

                let formula_ok = formula == case.expected_formula;
                let mw_ok = (mw - case.expected_mw).abs() < 0.1;
                let groups_ok = case
                    .expected_groups
                    .iter()
                    .all(|g| group_names.iter().any(|gn| gn == g))
                    && group_names.len() == case.expected_groups.len();

                println!(
                    "  formula: {formula} (expected {})  {}",
                    case.expected_formula,
                    if formula_ok { "OK" } else { "MISMATCH" }
                );
                println!(
                    "  MW: {mw:.2} (expected {:.2})  {}",
                    case.expected_mw,
                    if mw_ok { "OK" } else { "MISMATCH" }
                );
                println!(
                    "  groups: {:?} (expected {:?})  {}",
                    group_names,
                    case.expected_groups,
                    if groups_ok { "OK" } else { "MISMATCH" }
                );

                if formula_ok && mw_ok && groups_ok {
                    n_pass += 1;
                } else {
                    n_fail += 1;
                }
            }
            Err(e) => {
                println!("  PARSE FAILED: {e:?}");
                n_fail += 1;
            }
        }
        println!();
    }

    println!("=== Summary: {n_pass}/{} passed ===", cases.len());
    if n_fail > 0 {
        std::process::exit(1);
    }
}
