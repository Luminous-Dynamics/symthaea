// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Quantum Information Theory from molecular physics.
//!
//! References:
//! - Nielsen & Chuang (2000). *Quantum Computation and Quantum Information*.
//! - Rissler, Noack & White (2006). Chem. Phys. 323, 519 (orbital entanglement).
//!
//! ## Status by function (Phase Q0, 2026-07-16)
//!
//! This module's functions are NOT uniformly "computed from molecular orbital
//! density matrices" as the old module doc implied -- reviewed and labeled
//! individually below, since they range from exact generic math to explicit
//! proxies:
//!
//! - `von_neumann_entropy`, `mutual_information`: exact, generic formulas.
//!   Correct given real eigenvalues/entropies as input -- but nothing in
//!   this module computes those inputs from an actual molecular reduced
//!   density matrix; callers must supply them.
//! - `single_orbital_entropy`, `total_orbital_information`: a real formula
//!   for RHF single-orbital entanglement entropy as a function of occupation
//!   number, but RHF is a single Slater determinant with every orbital
//!   either fully occupied (n=2) or fully empty (n=0) -- exactly the two
//!   inputs this formula returns 0 for. Fed only real RHF occupations (as
//!   `total_orbital_information` does), this is **structurally always
//!   zero**, not a bug in the formula itself but a mismatch between what it
//!   needs (fractional/correlated occupations, which nothing in this crate
//!   currently produces) and what it's ever actually called with.
//! - `orbital_mutual_information`: an honestly-labeled proxy (its own doc
//!   comment already says "Simplified: uses orbital energy proximity as a
//!   correlation proxy") -- NOT a real density-matrix-derived mutual
//!   information, just `1/(1+100*Δε²)`. Correctly self-disclosed at the
//!   function level, but the old module-level doc's blanket "computed from
//!   density matrices" claim didn't reflect this.
//! - `bipartition_entanglement`: the real overclaim in this module. Its doc
//!   comment claims to compute "ρ_A = Tr_B|Ψ⟩⟨Ψ|" (a genuine reduced-density-
//!   matrix trace), but the implementation computes something different and
//!   simpler: a Shannon-entropy-like quantity from how each occupied MO's
//!   coefficients are distributed across a basis-function bipartition. This
//!   is a real, well-defined localization measure, but it is not the RDM
//!   trace its doc comment describes and should not be read as genuine
//!   orbital-orbital entanglement entropy.

use crate::scf::rhf::RhfResult;

/// Von Neumann entropy of a density matrix: S = -Tr(ρ ln ρ)
///
/// For a pure state, S = 0. For a maximally mixed state of dimension d, S = ln(d).
pub fn von_neumann_entropy(eigenvalues: &[f64]) -> f64 {
    let mut s = 0.0;
    for &lambda in eigenvalues {
        if lambda > 1e-15 {
            s -= lambda * lambda.ln();
        }
    }
    s
}

/// Single-orbital entropy s(1)_i measuring the entanglement of orbital i
/// with all other orbitals.
///
/// For RHF: the 1-orbital reduced density matrix has eigenvalues
/// n_i/2, n_i/2, (1-n_i/2), (1-n_i/2) for occupation n_i.
///
/// s(1)_i = -n_i ln(n_i/2) - (2-n_i) ln(1-n_i/2) for n_i ∈ (0, 2)
///
/// Occupied orbitals (n_i=2) and empty (n_i=0) have s(1)=0.
/// Partially occupied orbitals have maximum entanglement.
pub fn single_orbital_entropy(occupation: f64) -> f64 {
    if occupation < 1e-12 || (2.0 - occupation).abs() < 1e-12 {
        return 0.0;
    }

    let p1 = occupation / 2.0;
    let p2 = 1.0 - p1;

    let mut s = 0.0;
    if p1 > 1e-15 {
        s -= 2.0 * p1 * p1.ln(); // two spin components
    }
    if p2 > 1e-15 {
        s -= 2.0 * p2 * p2.ln();
    }
    s
}

/// Mutual information between two subsystems A and B.
/// I(A:B) = S(A) + S(B) - S(AB)
pub fn mutual_information(s_a: f64, s_b: f64, s_ab: f64) -> f64 {
    s_a + s_b - s_ab
}

/// Orbital-orbital mutual information I_{ij} measuring quantum correlation
/// between orbitals i and j.
///
/// For a closed-shell RHF state, this is estimated from the orbital
/// pair density: I_{ij} ∝ |K_{ij}|² / (ε_i - ε_j)² where K is the exchange.
///
/// Simplified: uses orbital energy proximity as a correlation proxy.
pub fn orbital_mutual_information(rhf: &RhfResult) -> Vec<Vec<f64>> {
    let n_mo = rhf.n_independent;
    let eps = &rhf.orbital_energies;
    let mut mi = vec![vec![0.0; n_mo]; n_mo];

    for i in 0..n_mo {
        for j in (i + 1)..n_mo {
            let de = (eps[i] - eps[j]).abs();
            // Correlation inversely proportional to energy gap
            // (degenerate orbitals are maximally correlated)
            let corr = if de > 1e-10 {
                1.0 / (1.0 + de * de * 100.0)
            } else {
                1.0
            };
            mi[i][j] = corr;
            mi[j][i] = corr;
        }
    }

    mi
}

/// Total quantum information content of the molecular state.
/// Sum of all single-orbital entropies.
pub fn total_orbital_information(rhf: &RhfResult) -> f64 {
    let n_occ = rhf.n_occupied;
    let n_mo = rhf.n_independent;

    let mut total = 0.0;
    for i in 0..n_mo {
        let occ = if i < n_occ { 2.0 } else { 0.0 };
        total += single_orbital_entropy(occ);
    }
    total
}

/// Basis-function localization entropy of each occupied MO across a
/// bipartition of the AO basis -- NOT a reduced-density-matrix entanglement
/// entropy despite the name (Phase Q0 status note, 2026-07-16; see the
/// module doc's "Status by function" section for the full explanation).
///
/// For each occupied orbital, treats the fraction of its MO-coefficient
/// weight lying in partition A as a binary probability `p` and sums
/// `-p ln p - (1-p) ln(1-p)` across occupied orbitals. This is a genuine,
/// well-defined localization measure, but it is not `Tr_B |Ψ⟩⟨Ψ|` (a real
/// reduced-density-matrix trace over Fock space), which this function does
/// not compute.
pub fn bipartition_entanglement(
    rhf: &RhfResult,
    partition_a: &[usize], // orbital indices in partition A
) -> f64 {
    // For a single Slater determinant, entanglement entropy =
    // -Σ_i [n_i ln(n_i) + (1-n_i) ln(1-n_i)] over the overlap eigenvalues
    // Simplified: count how many occupied orbitals span both partitions

    let n_occ = rhf.n_occupied;
    let n_mo = rhf.n_independent;
    let c = &rhf.orbital_coefficients;

    // Compute overlap of each occupied MO with partition A basis functions
    let mut entropies = 0.0;
    for i in 0..n_occ {
        // Weight of orbital i in partition A
        let mut weight_a = 0.0;
        let mut total_weight = 0.0;

        for &mu in partition_a {
            if mu < rhf.n_basis {
                let c_mi = c[mu * n_mo + i];
                weight_a += c_mi * c_mi;
            }
        }
        for mu in 0..rhf.n_basis {
            let c_mi = c[mu * n_mo + i];
            total_weight += c_mi * c_mi;
        }

        if total_weight > 1e-14 {
            let p = (weight_a / total_weight).clamp(1e-15, 1.0 - 1e-15);
            entropies += -p * p.ln() - (1.0 - p) * (1.0 - p).ln();
        }
    }

    entropies
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisSetProvider;
    use crate::basis::sto3g::Sto3g;
    use crate::molecule::Molecule;
    use crate::scf::rhf::{RhfConfig, restricted_hartree_fock};

    #[test]
    fn test_von_neumann_pure_state() {
        // Pure state: eigenvalues [1, 0, 0] → S = 0
        let s = von_neumann_entropy(&[1.0, 0.0, 0.0]);
        assert!(s.abs() < 1e-14, "Pure state entropy = {}", s);
    }

    #[test]
    fn test_von_neumann_maximally_mixed() {
        // Maximally mixed (d=2): eigenvalues [0.5, 0.5] → S = ln(2)
        let s = von_neumann_entropy(&[0.5, 0.5]);
        assert!(
            (s - 2.0_f64.ln()).abs() < 1e-14,
            "Max mixed entropy = {}, expected ln(2)={}",
            s,
            2.0_f64.ln()
        );
    }

    #[test]
    fn test_single_orbital_entropy_fully_occupied() {
        // n=2 (fully occupied): no entanglement with other orbitals
        let s = single_orbital_entropy(2.0);
        assert!(s.abs() < 1e-12);
    }

    #[test]
    fn test_single_orbital_entropy_half_occupied() {
        // n=1 (half occupied): maximum entanglement
        let s = single_orbital_entropy(1.0);
        assert!(s > 0.0, "Half-occupied should have entropy: {}", s);
        // Should be 2×ln(2) ≈ 1.386 (two spin components each at 50%)
        assert!((s - 2.0 * 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_mutual_information_positive() {
        let i = mutual_information(1.0, 1.5, 2.0);
        assert!(i >= 0.0, "MI should be non-negative: {}", i);
    }

    #[test]
    fn test_orbital_mi_symmetric() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());

        let mi = orbital_mutual_information(&rhf);
        for i in 0..mi.len() {
            for j in 0..mi[i].len() {
                assert!(
                    (mi[i][j] - mi[j][i]).abs() < 1e-14,
                    "MI not symmetric at ({},{})",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_bipartition_entanglement_positive() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());

        // Partition: first 3 basis functions vs rest
        let part_a: Vec<usize> = (0..3).collect();
        let s = bipartition_entanglement(&rhf, &part_a);
        assert!(s >= 0.0, "Entanglement entropy should be ≥ 0: {}", s);
    }
}
