// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Derived structural analysis on a parsed [`Molecule`]: ring count and degree
//! of unsaturation.

use crate::element;
use crate::smiles::Molecule;

impl Molecule {
    /// Number of connected components in the heavy-atom graph (usually 1, since
    /// the v0.1 parser rejects disconnected structures).
    pub fn connected_components(&self) -> usize {
        let n = self.atoms.len();
        if n == 0 {
            return 0;
        }
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], mut x: usize) -> usize {
            while parent[x] != x {
                parent[x] = parent[parent[x]]; // path compression
                x = parent[x];
            }
            x
        }
        for b in &self.bonds {
            let ra = find(&mut parent, b.a);
            let rb = find(&mut parent, b.b);
            if ra != rb {
                parent[ra] = rb;
            }
        }
        let mut roots = std::collections::BTreeSet::new();
        for i in 0..n {
            let r = find(&mut parent, i);
            roots.insert(r);
        }
        roots.len()
    }

    /// Number of independent rings (the graph's cyclomatic number / first Betti
    /// number): `rings = bonds − atoms + components`.
    ///
    /// Benzene → 1, naphthalene → 2, cyclohexane → 1, any acyclic molecule → 0.
    pub fn ring_count(&self) -> usize {
        let e = self.bonds.len();
        let v = self.atoms.len();
        let c = self.connected_components();
        // e - v + c is non-negative for any graph.
        (e + c).saturating_sub(v)
    }

    /// Degree of unsaturation (double-bond equivalents): rings + π-bonds.
    ///
    /// `DoU = (2·C + 2 + (N+P) − (H + halogens)) / 2`. Divalent atoms (O, S) do
    /// not affect the count. Benzene → 4 (3 π + 1 ring), CO₂ → 2, ethanol → 0.
    pub fn degree_of_unsaturation(&self) -> u32 {
        let mut n_carbon: i64 = 0;
        let mut n_trivalent: i64 = 0; // N, P
        let mut n_monovalent: i64 = 0; // halogens (+ hydrogens added below)
        for a in &self.atoms {
            match a.element {
                "C" => n_carbon += 1,
                "N" | "P" => n_trivalent += 1,
                "F" | "Cl" | "Br" | "I" => n_monovalent += 1,
                _ => {}
            }
            let _ = element::lookup(a.element); // element must be known
        }
        let h_total: i64 = self.atoms.iter().map(|a| a.hydrogens as i64).sum();
        n_monovalent += h_total;

        let numerator = 2 * n_carbon + 2 + n_trivalent - n_monovalent;
        if numerator <= 0 {
            0
        } else {
            (numerator / 2) as u32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mol(s: &str) -> Molecule {
        Molecule::from_smiles(s).unwrap()
    }

    #[test]
    fn acyclic_has_no_rings() {
        assert_eq!(mol("CCO").ring_count(), 0);
        assert_eq!(mol("CC(=O)O").ring_count(), 0);
    }

    #[test]
    fn benzene_has_one_ring() {
        assert_eq!(mol("c1ccccc1").ring_count(), 1);
    }

    #[test]
    fn cyclohexane_has_one_ring() {
        assert_eq!(mol("C1CCCCC1").ring_count(), 1);
    }

    #[test]
    fn naphthalene_has_two_rings() {
        assert_eq!(mol("c1ccc2ccccc2c1").ring_count(), 2);
    }

    #[test]
    fn degree_of_unsaturation_values() {
        assert_eq!(mol("CCO").degree_of_unsaturation(), 0); // ethanol
        assert_eq!(mol("O=C=O").degree_of_unsaturation(), 2); // CO2: two π
        assert_eq!(mol("CC#N").degree_of_unsaturation(), 2); // nitrile: triple = 2
        assert_eq!(mol("c1ccccc1").degree_of_unsaturation(), 4); // benzene: 3π + 1 ring
        assert_eq!(mol("CC(=O)O").degree_of_unsaturation(), 1); // acetic acid: 1 π
    }

    #[test]
    fn single_atom_is_connected() {
        assert_eq!(mol("C").connected_components(), 1);
        assert_eq!(mol("C").ring_count(), 0);
    }
}
