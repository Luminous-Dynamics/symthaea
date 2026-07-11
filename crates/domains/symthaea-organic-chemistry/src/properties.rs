// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cheminformatic properties: hydrogen-bond donors/acceptors, elemental mass
//! composition, and the Lipinski "Rule of Five" drug-likeness screen.

use crate::element;
use crate::smiles::Molecule;

impl Molecule {
    /// Hydrogen-bond donor count: hydrogens attached to N or O (Lipinski's
    /// donor definition — the sum of N–H and O–H).
    pub fn hbond_donors(&self) -> usize {
        self.atoms
            .iter()
            .filter(|a| a.element == "N" || a.element == "O")
            .map(|a| a.hydrogens as usize)
            .sum()
    }

    /// Hydrogen-bond acceptor count: number of N and O atoms (Lipinski's simple
    /// acceptor definition).
    pub fn hbond_acceptors(&self) -> usize {
        self.atoms
            .iter()
            .filter(|a| a.element == "N" || a.element == "O")
            .count()
    }

    /// Elemental mass composition: `(symbol, mass_percent)` in Hill-ish order
    /// (C, H, then others alphabetical), summing to ~100%.
    pub fn mass_composition(&self) -> Vec<(&'static str, f64)> {
        // Tally element counts including hydrogens.
        let mut counts: Vec<(&'static str, usize)> = Vec::new();
        let mut bump = |sym: &'static str, n: usize| {
            if n == 0 {
                return;
            }
            if let Some(e) = counts.iter_mut().find(|(s, _)| *s == sym) {
                e.1 += n;
            } else {
                counts.push((sym, n));
            }
        };
        for a in &self.atoms {
            bump(a.element, 1);
        }
        bump("H", self.atoms.iter().map(|a| a.hydrogens as usize).sum());

        let total: f64 = counts
            .iter()
            .map(|(s, n)| element::lookup(s).map(|e| e.weight).unwrap_or(0.0) * *n as f64)
            .sum();

        let mut out: Vec<(&'static str, f64)> = counts
            .iter()
            .map(|(s, n)| {
                let mass = element::lookup(s).map(|e| e.weight).unwrap_or(0.0) * *n as f64;
                (
                    *s,
                    if total > 0.0 {
                        100.0 * mass / total
                    } else {
                        0.0
                    },
                )
            })
            .collect();

        // Order: C, H, then alphabetical.
        out.sort_by(|a, b| rank(a.0).cmp(&rank(b.0)).then(a.0.cmp(b.0)));
        out
    }
}

fn rank(sym: &str) -> u8 {
    match sym {
        "C" => 0,
        "H" => 1,
        _ => 2,
    }
}

/// Result of a Lipinski "Rule of Five" evaluation.
///
/// **Note:** this checks three of the four classic rules — molecular weight,
/// H-bond donors, and H-bond acceptors. The fourth (logP ≤ 5) needs an
/// atom-contribution model not yet implemented, so it is omitted (documented,
/// not silently dropped). A molecule is flagged drug-like if it violates at most
/// one of the checked rules.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lipinski {
    pub molecular_weight: f64,
    pub hbond_donors: usize,
    pub hbond_acceptors: usize,
    /// Number of the three checked rules violated.
    pub violations: usize,
    /// Drug-like if `violations <= 1`.
    pub drug_like: bool,
}

impl Lipinski {
    /// Evaluate the (logP-omitted) Rule of Five from raw descriptors.
    pub fn evaluate(
        molecular_weight: f64,
        hbond_donors: usize,
        hbond_acceptors: usize,
    ) -> Lipinski {
        let mut violations = 0;
        if molecular_weight > 500.0 {
            violations += 1;
        }
        if hbond_donors > 5 {
            violations += 1;
        }
        if hbond_acceptors > 10 {
            violations += 1;
        }
        Lipinski {
            molecular_weight,
            hbond_donors,
            hbond_acceptors,
            violations,
            drug_like: violations <= 1,
        }
    }
}

/// Screen a molecule with the Lipinski Rule of Five (logP omitted).
pub fn lipinski(m: &Molecule) -> Lipinski {
    Lipinski::evaluate(m.molecular_weight(), m.hbond_donors(), m.hbond_acceptors())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mol(s: &str) -> Molecule {
        Molecule::from_smiles(s).unwrap()
    }

    #[test]
    fn hbond_counts_water_and_ethanol() {
        assert_eq!(mol("O").hbond_donors(), 2); // H2O: 2 O–H
        assert_eq!(mol("O").hbond_acceptors(), 1);
        assert_eq!(mol("CCO").hbond_donors(), 1); // ethanol O–H
        assert_eq!(mol("CCO").hbond_acceptors(), 1);
    }

    #[test]
    fn mass_composition_ethanol() {
        let comp = mol("CCO").mass_composition();
        // C 52.14%, H 13.13%, O 34.73%.
        let get = |s: &str| comp.iter().find(|(x, _)| *x == s).unwrap().1;
        assert!((get("C") - 52.14).abs() < 0.05, "C={}", get("C"));
        assert!((get("H") - 13.13).abs() < 0.05, "H={}", get("H"));
        assert!((get("O") - 34.73).abs() < 0.05, "O={}", get("O"));
        let sum: f64 = comp.iter().map(|(_, p)| p).sum();
        assert!((sum - 100.0).abs() < 1e-6);
    }

    #[test]
    fn evaluate_boundary_logic() {
        // Well within limits → drug-like, 0 violations.
        assert!(Lipinski::evaluate(300.0, 2, 5).drug_like);
        // One violation (MW) → still drug-like.
        let one = Lipinski::evaluate(600.0, 2, 5);
        assert_eq!(one.violations, 1);
        assert!(one.drug_like);
        // Three violations → not drug-like.
        let three = Lipinski::evaluate(700.0, 8, 14);
        assert_eq!(three.violations, 3);
        assert!(!three.drug_like);
    }

    #[test]
    fn aspirin_is_drug_like() {
        let aspirin = mol("CC(=O)Oc1ccccc1C(=O)O"); // C9H8O4
        assert!((aspirin.molecular_weight() - 180.16).abs() < 0.1);
        assert_eq!(aspirin.hbond_donors(), 1); // carboxyl O–H
        assert_eq!(aspirin.hbond_acceptors(), 4); // 4 oxygens
        let l = lipinski(&aspirin);
        assert_eq!(l.violations, 0);
        assert!(l.drug_like);
    }

    #[test]
    fn caffeine_hbonds_and_drug_likeness() {
        let caffeine = mol("Cn1cnc2c1c(=O)n(C)c(=O)n2C"); // C8H10N4O2
        assert_eq!(caffeine.hbond_donors(), 0); // no N–H / O–H
        assert_eq!(caffeine.hbond_acceptors(), 6); // 4 N + 2 O
        assert!(lipinski(&caffeine).drug_like);
    }
}
