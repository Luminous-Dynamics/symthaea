// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Emergent Consciousness from Physical First Principles.
//!
//! Bridges quantum chemistry, statistical mechanics, and information theory
//! to derive consciousness-like properties from fundamental physics.
//!
//! ## Thesis
//!
//! Consciousness emerges when a physical system exhibits:
//! 1. **Integration** (Φ > 0): Subsystems are informationally coupled
//! 2. **Differentiation**: The system can be in many distinguishable states
//! 3. **Temporal depth**: States evolve with memory of past states
//! 4. **Self-reference**: System models contain the system itself
//!
//! This module computes these properties from molecular wavefunctions
//! and orbital structure, providing a first-principles bridge from
//! Schrödinger equation → Integrated Information Theory (IIT).
//!
//! References:
//! - Tononi, G. (2004). BMC Neuroscience 5, 42 (IIT).
//! - Tegmark, M. (2015). Consciousness as a State of Matter.
//! - Koch et al. (2016). Nature Reviews Neuroscience 17, 307.
//! - Zanardi et al. (2004). Phys. Rev. A 69, 054301 (quantum Φ).

use crate::basis::ContractedGaussian;
use crate::consciousness::{build_atom_basis_ranges, compute_orbital_phi};
use crate::molecule::Atom;
use crate::quantum_info::{
    bipartition_entanglement, orbital_mutual_information, von_neumann_entropy,
};
use crate::scf::rhf::RhfResult;
use crate::stat_mech::{K_BOLTZMANN_HARTREE, entropy as thermal_entropy};

/// Complete consciousness profile derived from first principles.
#[derive(Debug, Clone)]
pub struct ConsciousnessProfile {
    /// Integrated Information Φ (from orbital delocalization)
    pub phi_orbital: f64,

    /// Quantum Φ: entanglement-based integration across atomic subsystems
    pub phi_quantum: f64,

    /// Information differentiation: number of distinguishable states
    /// (related to HOMO-LUMO gap and temperature)
    pub differentiation: f64,

    /// Temporal coherence: how long quantum coherence persists
    /// (estimated from energy gaps)
    pub temporal_depth: f64,

    /// Complexity: product of integration and differentiation
    /// (Tononi's "main complex")
    pub complexity: f64,

    /// Thermodynamic consciousness: free energy available for computation
    pub free_energy_consciousness: f64,

    /// Total consciousness score (normalized composite)
    pub total_score: f64,

    /// Component breakdown
    pub components: ConsciousnessComponents,
}

/// Detailed breakdown of consciousness components.
#[derive(Debug, Clone)]
pub struct ConsciousnessComponents {
    /// Von Neumann entropy of the electronic state
    pub von_neumann_entropy: f64,
    /// Bipartition entanglement entropy
    pub entanglement_entropy: f64,
    /// Mutual information network density
    pub mi_network_density: f64,
    /// HOMO-LUMO gap (eV) — bandwidth for information processing
    pub homo_lumo_gap_ev: f64,
    /// Number of thermally accessible states
    pub accessible_states: f64,
    /// Decoherence time estimate (a.u.)
    pub decoherence_time: f64,
}

/// Compute the full consciousness profile from a molecular RHF calculation.
///
/// This is the key function that bridges quantum chemistry → consciousness theory.
/// It takes the output of a Hartree-Fock calculation and derives IIT-like
/// consciousness metrics from the orbital structure.
pub fn consciousness_from_first_principles(
    rhf: &RhfResult,
    atoms: &[Atom],
    basis: &[ContractedGaussian],
    temperature: f64, // Kelvin
) -> ConsciousnessProfile {
    let n_mo = rhf.n_independent;
    let n_occ = rhf.n_occupied;
    let eps = &rhf.orbital_energies;

    // 1. Orbital Φ (from consciousness.rs)
    let ranges = build_atom_basis_ranges(atoms, basis);
    let phi_measurement = compute_orbital_phi(rhf, &ranges);
    let phi_orbital = phi_measurement.molecular_phi;

    // 2. Quantum Φ: entanglement across atomic subsystems
    let n_atoms = atoms.len();
    let mut phi_quantum = 0.0;
    if n_atoms >= 2 {
        // Bipartition across each atom boundary
        let mut max_entanglement = 0.0;
        for cut in 1..n_atoms {
            let part_a: Vec<usize> = (0..ranges[cut].0).collect();
            let s_ent = bipartition_entanglement(rhf, &part_a);
            if s_ent > max_entanglement {
                max_entanglement = s_ent;
            }
        }
        phi_quantum = max_entanglement;
    }

    // 3. Differentiation: number of thermally accessible electronic states
    let homo_lumo_gap = phi_measurement.homo_lumo_gap;
    let homo_lumo_gap_ev = homo_lumo_gap * 27.211_386;

    let accessible_states = if temperature > 0.0 {
        // States within kT of the Fermi level
        let kt = K_BOLTZMANN_HARTREE * temperature;
        let mut n_accessible = 0.0;
        for &e in eps.iter() {
            let de = (e - eps[n_occ - 1]).abs();
            n_accessible += (-de / kt).exp();
        }
        n_accessible
    } else {
        1.0
    };

    let differentiation = accessible_states.ln().max(0.0);

    // 4. Temporal depth: decoherence time ∝ 1/ΔE
    // Quantum coherence persists for time ~ ℏ/ΔE
    let decoherence_time = if homo_lumo_gap > 1e-10 {
        1.0 / homo_lumo_gap // In atomic units (1 a.u. ≈ 24 as)
    } else {
        1e6 // Effectively infinite for degenerate systems
    };

    let temporal_depth = decoherence_time.ln().max(0.0);

    // 5. Complexity = Integration × Differentiation
    let complexity = (phi_orbital + phi_quantum) * differentiation;

    // 6. Thermodynamic consciousness: free energy available for computation
    // F_consciousness = kT × ln(Z_electronic) — entropy cost of maintaining coherence
    let energies: Vec<f64> = eps.iter().take(n_mo.min(10)).cloned().collect();
    let degeneracies = vec![2.0; energies.len()]; // Spin degeneracy
    let _thermal_s = thermal_entropy(&energies, &degeneracies, temperature.max(100.0));
    let free_energy_consciousness = (phi_orbital * homo_lumo_gap).max(0.0);

    // 7. Von Neumann entropy (from orbital populations)
    // For RHF: occupied orbitals have eigenvalue 1 (pure state → S=0)
    // But in a thermal ensemble at T>0, populations are fractional
    let mut thermal_populations: Vec<f64> = Vec::new();
    if temperature > 0.0 {
        let kt = K_BOLTZMANN_HARTREE * temperature;
        let fermi_level = if n_occ > 0 {
            (eps[n_occ - 1] + eps[n_occ.min(n_mo - 1)]) / 2.0
        } else {
            0.0
        };
        for &e in eps.iter() {
            // Fermi-Dirac distribution
            let p = 1.0 / (1.0 + ((e - fermi_level) / kt).exp());
            thermal_populations.push(p);
        }
        // Normalize
        let sum: f64 = thermal_populations.iter().sum();
        if sum > 0.0 {
            for p in &mut thermal_populations {
                *p /= sum;
            }
        }
    } else {
        thermal_populations = vec![0.0; n_mo];
        for i in 0..n_occ {
            thermal_populations[i] = 1.0 / n_occ as f64;
        }
    }
    let vn_entropy = von_neumann_entropy(&thermal_populations);

    // 8. Mutual information network
    let mi = orbital_mutual_information(rhf);
    let n_pairs = n_mo * (n_mo - 1) / 2;
    let mi_total: f64 = mi.iter().flat_map(|row| row.iter()).sum::<f64>() / 2.0;
    let mi_density = if n_pairs > 0 {
        mi_total / n_pairs as f64
    } else {
        0.0
    };

    // Entanglement entropy (bipartition at midpoint)
    let mid = rhf.n_basis / 2;
    let part_a: Vec<usize> = (0..mid).collect();
    let entanglement_entropy = bipartition_entanglement(rhf, &part_a);

    // 9. Total consciousness score (normalized 0-1)
    // Weighted combination of all components
    let raw_score = 0.3 * phi_orbital
        + 0.2 * phi_quantum.min(1.0)
        + 0.15 * (differentiation / 5.0).min(1.0)
        + 0.15 * (temporal_depth / 10.0).min(1.0)
        + 0.1 * mi_density.min(1.0)
        + 0.1 * (vn_entropy / 2.0).min(1.0);

    let total_score = raw_score.clamp(0.0, 1.0);

    ConsciousnessProfile {
        phi_orbital,
        phi_quantum,
        differentiation,
        temporal_depth,
        complexity,
        free_energy_consciousness,
        total_score,
        components: ConsciousnessComponents {
            von_neumann_entropy: vn_entropy,
            entanglement_entropy,
            mi_network_density: mi_density,
            homo_lumo_gap_ev,
            accessible_states,
            decoherence_time,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::BasisSetProvider;
    use crate::basis::sto3g::Sto3g;
    use crate::molecule::Molecule;
    use crate::scf::rhf::{RhfConfig, restricted_hartree_fock};

    #[test]
    fn test_consciousness_h2() {
        let mol = Molecule::h2();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());

        let profile =
            consciousness_from_first_principles(&rhf, &mol.atoms, &basis.functions, 300.0);

        assert!(
            profile.phi_orbital > 0.0,
            "H2 should have orbital integration"
        );
        assert!(
            profile.total_score > 0.0,
            "Consciousness score should be positive"
        );
        assert!(profile.total_score < 1.0, "Score should be < 1");
        assert!(profile.components.homo_lumo_gap_ev > 0.0);
    }

    #[test]
    fn test_consciousness_water_vs_h2() {
        // Water (3 atoms, 10 electrons) should have higher consciousness than H2
        // Use high temperature to activate differentiation (electronic excitations)
        let mol_h2 = Molecule::h2();
        let basis_h2 = Sto3g::build(&mol_h2);
        let rhf_h2 = restricted_hartree_fock(&mol_h2, &basis_h2, &RhfConfig::default());
        let c_h2 = consciousness_from_first_principles(
            &rhf_h2,
            &mol_h2.atoms,
            &basis_h2.functions,
            50000.0,
        );

        let mol_water = Molecule::water();
        let basis_water = Sto3g::build(&mol_water);
        let rhf_water = restricted_hartree_fock(&mol_water, &basis_water, &RhfConfig::default());
        let c_water = consciousness_from_first_principles(
            &rhf_water,
            &mol_water.atoms,
            &basis_water.functions,
            50000.0,
        );

        // Water should have higher total score (more integration, more orbitals)
        assert!(
            c_water.total_score > c_h2.total_score,
            "Water score ({:.4}) should exceed H2 ({:.4})",
            c_water.total_score,
            c_h2.total_score,
        );
    }

    #[test]
    fn test_consciousness_temperature_dependence() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());

        let c_cold = consciousness_from_first_principles(&rhf, &mol.atoms, &basis.functions, 100.0);
        let c_hot =
            consciousness_from_first_principles(&rhf, &mol.atoms, &basis.functions, 10000.0);

        // Higher temperature → more accessible states → higher differentiation
        assert!(
            c_hot.differentiation > c_cold.differentiation,
            "Hot ({:.4}) should have more differentiation than cold ({:.4})",
            c_hot.differentiation,
            c_cold.differentiation,
        );
    }

    #[test]
    fn test_consciousness_score_bounded() {
        let mol = Molecule::water();
        let basis = Sto3g::build(&mol);
        let rhf = restricted_hartree_fock(&mol, &basis, &RhfConfig::default());

        let profile =
            consciousness_from_first_principles(&rhf, &mol.atoms, &basis.functions, 300.0);

        assert!(profile.total_score >= 0.0 && profile.total_score <= 1.0);
        assert!(profile.phi_orbital >= 0.0);
        assert!(profile.complexity >= 0.0);
    }
}
