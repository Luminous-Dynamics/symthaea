// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Functional-group detection over a parsed [`Molecule`].
//!
//! v0.1 recognizes a conservative, well-defined set of common groups. Detection
//! is substructure pattern-matching on the heavy-atom graph plus derived
//! hydrogen counts. Classes are made mutually exclusive where chemically
//! sensible (a carboxyl `–C(=O)OH` reports `Carboxyl`, not also `Hydroxyl` and
//! `Carbonyl`) so counts read the way a chemist expects.

use crate::smiles::{BondOrder, Molecule};

/// A recognized functional group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FunctionalGroup {
    Hydroxyl,
    Carbonyl,
    Carboxyl,
    Ester,
    Ether,
    Amine,
    Amide,
    Nitrile,
    Halide,
    AromaticRing,
}

/// Does carbon `i` bear a C=O double bond?
fn is_carbonyl_carbon(m: &Molecule, i: usize) -> bool {
    m.atoms[i].element == "C"
        && m.neighbors(i)
            .iter()
            .any(|(j, o)| *o == BondOrder::Double && m.atoms[*j].element == "O")
}

/// A single-bonded oxygen neighbour of `i` bearing at least one hydrogen.
fn hydroxyl_oxygen_of(m: &Molecule, i: usize) -> Option<usize> {
    m.neighbors(i).iter().find_map(|(j, o)| {
        (*o == BondOrder::Single && m.atoms[*j].element == "O" && m.atoms[*j].hydrogens >= 1)
            .then_some(*j)
    })
}

/// A single-bonded oxygen neighbour of `i` that also bonds another carbon
/// (the ester linkage `–C(=O)–O–C`).
fn ester_oxygen_of(m: &Molecule, i: usize) -> Option<usize> {
    m.neighbors(i).iter().find_map(|(j, o)| {
        if *o != BondOrder::Single || m.atoms[*j].element != "O" {
            return None;
        }
        let other_carbon = m
            .neighbors(*j)
            .iter()
            .any(|(k, _)| *k != i && m.atoms[*k].element == "C");
        other_carbon.then_some(*j)
    })
}

fn is_carboxyl_carbon(m: &Molecule, i: usize) -> bool {
    is_carbonyl_carbon(m, i) && hydroxyl_oxygen_of(m, i).is_some()
}

fn is_ester_carbon(m: &Molecule, i: usize) -> bool {
    is_carbonyl_carbon(m, i) && ester_oxygen_of(m, i).is_some()
}

/// Nitrogen `i` single-bonded to a carbonyl carbon → amide nitrogen.
fn is_amide_nitrogen(m: &Molecule, i: usize) -> bool {
    m.atoms[i].element == "N"
        && m.neighbors(i)
            .iter()
            .any(|(j, o)| *o == BondOrder::Single && is_carbonyl_carbon(m, *j))
}

/// Detect all functional groups present, returned sorted and deduplicated.
pub fn detect(m: &Molecule) -> Vec<FunctionalGroup> {
    use FunctionalGroup::*;
    let mut found: Vec<FunctionalGroup> = Vec::new();
    let add = |g: FunctionalGroup, found: &mut Vec<FunctionalGroup>| {
        if !found.contains(&g) {
            found.push(g);
        }
    };

    for i in 0..m.atoms.len() {
        let a = &m.atoms[i];

        if a.aromatic {
            add(AromaticRing, &mut found);
        }

        match a.element {
            "C" => {
                if is_carboxyl_carbon(m, i) {
                    add(Carboxyl, &mut found);
                } else if is_ester_carbon(m, i) {
                    add(Ester, &mut found);
                } else if is_carbonyl_carbon(m, i) {
                    // ketone / aldehyde carbonyl, and not amide (handled at N)
                    let bonded_amide_n = m
                        .neighbors(i)
                        .iter()
                        .any(|(j, o)| *o == BondOrder::Single && m.atoms[*j].element == "N");
                    if !bonded_amide_n {
                        add(Carbonyl, &mut found);
                    }
                }
                // nitrile: C#N
                if m.neighbors(i)
                    .iter()
                    .any(|(j, o)| *o == BondOrder::Triple && m.atoms[*j].element == "N")
                {
                    add(Nitrile, &mut found);
                }
            }
            "O" => {
                // hydroxyl: O–H single-bonded to a non-carboxyl carbon
                if a.hydrogens >= 1 {
                    let on_carbon = m.neighbors(i).iter().any(|(j, o)| {
                        *o == BondOrder::Single
                            && m.atoms[*j].element == "C"
                            && !is_carboxyl_carbon(m, *j)
                    });
                    if on_carbon {
                        add(Hydroxyl, &mut found);
                    }
                } else {
                    // ether: C–O–C, no hydrogens, two single C bonds
                    let carbons = m
                        .neighbors(i)
                        .iter()
                        .filter(|(j, o)| *o == BondOrder::Single && m.atoms[*j].element == "C")
                        .count();
                    let is_ester_o = m
                        .neighbors(i)
                        .iter()
                        .any(|(j, _)| is_carbonyl_carbon(m, *j));
                    if carbons >= 2 && !is_ester_o {
                        add(Ether, &mut found);
                    }
                }
            }
            "N" => {
                if is_amide_nitrogen(m, i) {
                    add(Amide, &mut found);
                } else if a.charge == 0
                    && m.neighbors(i).iter().all(|(_, o)| *o == BondOrder::Single)
                    && m.neighbors(i)
                        .iter()
                        .any(|(j, _)| m.atoms[*j].element == "C")
                {
                    add(Amine, &mut found);
                }
            }
            "F" | "Cl" | "Br" | "I" => {
                if m.neighbors(i)
                    .iter()
                    .any(|(j, _)| m.atoms[*j].element == "C")
                {
                    add(Halide, &mut found);
                }
            }
            _ => {}
        }
    }

    found.sort();
    found
}

#[cfg(test)]
mod tests {
    use super::FunctionalGroup::*;
    use super::*;

    fn groups(smiles: &str) -> Vec<FunctionalGroup> {
        detect(&Molecule::from_smiles(smiles).unwrap())
    }

    #[test]
    fn ethanol_is_hydroxyl() {
        assert_eq!(groups("CCO"), vec![Hydroxyl]);
    }

    #[test]
    fn acetic_acid_is_carboxyl_only() {
        assert_eq!(groups("CC(=O)O"), vec![Carboxyl]);
    }

    #[test]
    fn acetone_is_carbonyl() {
        assert_eq!(groups("CC(=O)C"), vec![Carbonyl]);
    }

    #[test]
    fn dimethyl_ether_is_ether() {
        assert_eq!(groups("COC"), vec![Ether]);
    }

    #[test]
    fn methylamine_is_amine() {
        assert_eq!(groups("CN"), vec![Amine]);
    }

    #[test]
    fn acetamide_is_amide() {
        assert_eq!(groups("CC(=O)N"), vec![Amide]);
    }

    #[test]
    fn acetonitrile_is_nitrile() {
        assert_eq!(groups("CC#N"), vec![Nitrile]);
    }

    #[test]
    fn benzene_is_aromatic_ring() {
        assert_eq!(groups("c1ccccc1"), vec![AromaticRing]);
    }

    #[test]
    fn chloromethane_is_halide() {
        assert_eq!(groups("CCl"), vec![Halide]);
    }
}
