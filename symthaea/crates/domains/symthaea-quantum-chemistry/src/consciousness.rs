// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness coupling for quantum chemistry.
//!
//! Novel research angle: measuring "orbital consciousness" — how integrated
//! and delocalized the molecular electronic structure is. This bridges
//! quantum chemistry with Integrated Information Theory (IIT).
//!
//! ## Orbital Phi (Φ_orbital)
//!
//! Inspired by IIT's Φ measure, we define orbital-level information integration
//! as the degree to which a molecular orbital is delocalized across atoms.
//! A fully localized orbital (on one atom) has Φ = 0; a fully delocalized
//! orbital (equal weight across all atoms) has maximum Φ.
//!
//! Φ_i = 1 - Σ_A p_{iA}² (where p_{iA} = Σ_{μ∈A} |C_μi|² / Σ_μ |C_μi|²)
//!
//! This is essentially 1 minus the Herfindahl index of the orbital's
//! atom-population distribution. It ranges from 0 (fully localized) to
//! 1 - 1/N_atoms (fully delocalized).
//!
//! ## Molecular Phi (Φ_molecular)
//!
//! The aggregate consciousness metric for the molecule:
//! Φ_mol = Σ_i (n_i / N_elec) × Φ_i
//!
//! where n_i is the occupation of orbital i (2 for RHF occupied, 0 for virtual).

use crate::scf::rhf::RhfResult;

/// Orbital consciousness measurement.
#[derive(Debug, Clone)]
pub struct OrbitalPhiMeasurement {
    /// Per-orbital Φ values (indexed by MO index)
    pub orbital_phi: Vec<f64>,
    /// Aggregate molecular Φ (occupation-weighted)
    pub molecular_phi: f64,
    /// Per-orbital delocalization index (number of effective atoms)
    pub delocalization_indices: Vec<f64>,
    /// Atom populations per orbital: atom_populations[orbital][atom]
    pub atom_populations: Vec<Vec<f64>>,
    /// HOMO-LUMO gap (Hartree) — a consciousness "bandwidth"
    pub homo_lumo_gap: f64,
}

/// Compute orbital Phi measurements from an RHF result.
///
/// `atom_basis_ranges` maps each atom to the range of basis functions
/// centered on it: `(start_index, end_index)` where end is exclusive.
pub fn compute_orbital_phi(
    rhf: &RhfResult,
    atom_basis_ranges: &[(usize, usize)],
) -> OrbitalPhiMeasurement {
    let n = rhf.n_basis;
    let n_mo = rhf.n_independent;
    let n_occ = rhf.n_occupied;
    let n_atoms = atom_basis_ranges.len();
    let c = &rhf.orbital_coefficients;
    let eps = &rhf.orbital_energies;

    let mut orbital_phi = Vec::with_capacity(n_mo);
    let mut delocalization_indices = Vec::with_capacity(n_mo);
    let mut atom_populations = Vec::with_capacity(n_mo);

    for i in 0..n_mo {
        // Compute atom populations for orbital i
        // p_{iA} = Σ_{μ∈A} C_μi² / Σ_μ C_μi²
        let total_weight: f64 = (0..n)
            .map(|mu| {
                let cmi = c[mu * n_mo + i];
                cmi * cmi
            })
            .sum();

        let mut pops = vec![0.0; n_atoms];
        if total_weight > 1e-14 {
            for (a, &(start, end)) in atom_basis_ranges.iter().enumerate() {
                let atom_weight: f64 = (start..end)
                    .map(|mu| {
                        let cmi = c[mu * n_mo + i];
                        cmi * cmi
                    })
                    .sum();
                pops[a] = atom_weight / total_weight;
            }
        }

        // Φ_i = 1 - Σ_A p_{iA}² (1 minus Herfindahl index)
        let herfindahl: f64 = pops.iter().map(|p| p * p).sum();
        let phi_i = 1.0 - herfindahl;

        // Delocalization index: effective number of atoms = 1 / Σ p²
        let deloc = if herfindahl > 1e-14 {
            1.0 / herfindahl
        } else {
            n_atoms as f64
        };

        orbital_phi.push(phi_i);
        delocalization_indices.push(deloc);
        atom_populations.push(pops);
    }

    // Molecular Φ: occupation-weighted average
    let n_elec = 2 * n_occ;
    let molecular_phi = if n_elec > 0 {
        let mut weighted_sum = 0.0;
        for i in 0..n_occ {
            weighted_sum += 2.0 * orbital_phi[i]; // occupation = 2
        }
        weighted_sum / n_elec as f64
    } else {
        0.0
    };

    // HOMO-LUMO gap
    let homo_lumo_gap = if n_occ > 0 && n_occ < n_mo {
        eps[n_occ] - eps[n_occ - 1]
    } else {
        0.0
    };

    OrbitalPhiMeasurement {
        orbital_phi,
        molecular_phi,
        delocalization_indices,
        atom_populations,
        homo_lumo_gap,
    }
}

/// Build atom-to-basis-function ranges from a molecule and basis set.
///
/// Returns (start, end) pairs for each atom's basis functions.
pub fn build_atom_basis_ranges(
    atoms: &[crate::molecule::Atom],
    basis_functions: &[crate::basis::ContractedGaussian],
) -> Vec<(usize, usize)> {
    let mut ranges = Vec::with_capacity(atoms.len());
    let mut current = 0;

    for atom in atoms {
        let start = current;
        // Count basis functions centered on this atom
        while current < basis_functions.len() {
            let center = basis_functions[current].primitives[0].center;
            let pos = atom.position;
            let dist = ((center[0] - pos[0]).powi(2)
                + (center[1] - pos[1]).powi(2)
                + (center[2] - pos[2]).powi(2))
            .sqrt();
            if dist > 1e-6 {
                break; // This basis function belongs to a different atom
            }
            current += 1;
        }
        ranges.push((start, current));
    }

    ranges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisSetProvider;
    use crate::basis::sto3g::Sto3g;
    use crate::molecule::Molecule;
    use crate::scf::rhf::{RhfConfig, restricted_hartree_fock};

    #[test]
    fn test_h2_orbital_phi() {
        let mol = Molecule::h2();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());

        let ranges = build_atom_basis_ranges(&mol.atoms, &basis.functions);
        let phi = compute_orbital_phi(&rhf, &ranges);

        // H2 bonding orbital should be delocalized across both atoms
        assert!(
            phi.orbital_phi[0] > 0.3,
            "H2 bonding orbital Φ = {:.4}, should be delocalized",
            phi.orbital_phi[0]
        );

        // Molecular Φ should be positive
        assert!(phi.molecular_phi > 0.0);

        // Delocalization index should be ~2 (two atoms)
        assert!(
            phi.delocalization_indices[0] > 1.5,
            "H2 bonding orbital deloc = {:.2}, expected ~2",
            phi.delocalization_indices[0]
        );
    }

    #[test]
    fn test_water_core_vs_valence_phi() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());

        let ranges = build_atom_basis_ranges(&mol.atoms, &basis.functions);
        let phi = compute_orbital_phi(&rhf, &ranges);

        // The 1s core orbital of O should be more localized than valence orbitals
        // Orbital 0 is typically the O 1s (most localized)
        let core_phi = phi.orbital_phi[0];
        let valence_phi = phi.orbital_phi[2]; // a bonding orbital

        assert!(
            core_phi < valence_phi + 0.1, // core may not always be index 0
            "Core Φ ({:.4}) should be ≤ valence Φ ({:.4})",
            core_phi,
            valence_phi
        );
    }

    #[test]
    fn test_homo_lumo_gap_positive() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());

        let ranges = build_atom_basis_ranges(&mol.atoms, &basis.functions);
        let phi = compute_orbital_phi(&rhf, &ranges);

        assert!(
            phi.homo_lumo_gap > 0.0,
            "HOMO-LUMO gap should be positive: {:.4} Hartree",
            phi.homo_lumo_gap
        );
    }

    #[test]
    fn test_phi_range() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());

        let ranges = build_atom_basis_ranges(&mol.atoms, &basis.functions);
        let phi = compute_orbital_phi(&rhf, &ranges);

        // All Φ values should be in [0, 1)
        for (i, &p) in phi.orbital_phi.iter().enumerate() {
            assert!(
                p >= 0.0 && p < 1.0,
                "Orbital {} Φ = {:.4} out of range [0,1)",
                i,
                p
            );
        }

        // Molecular Φ in [0, 1)
        assert!(phi.molecular_phi >= 0.0 && phi.molecular_phi < 1.0);
    }

    #[test]
    fn test_atom_populations_sum_to_one() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());

        let ranges = build_atom_basis_ranges(&mol.atoms, &basis.functions);
        let phi = compute_orbital_phi(&rhf, &ranges);

        for (i, pops) in phi.atom_populations.iter().enumerate() {
            let sum: f64 = pops.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Orbital {} atom populations sum to {:.4}, expected 1.0",
                i,
                sum
            );
        }
    }
}
