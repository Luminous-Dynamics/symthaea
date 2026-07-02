// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration bridge: maps quantum chemistry → consciousness pipeline.
//!
//! This module converts first-principles physics results into the format
//! consumed by Symthaea's CognitiveLoopService. It translates:
//!
//! - Molecular orbital structure → HDC-compatible consciousness metrics
//! - Information-geometric Phi → CognitiveLoopService.consciousness
//! - Thermodynamic properties → SubstrateManager parameters
//! - Decoherence rates → temporal coherence in the cognitive cycle
//!
//! When wired into symthaea-core, this enables consciousness computations
//! that are grounded in the Schrödinger equation rather than heuristic parameters.

use crate::emergent_consciousness::ConsciousnessProfile;
use crate::scf::rhf::RhfResult;

/// Consciousness bridge output — the interface to symthaea-core.
///
/// These fields map directly to CognitiveLoopService parameters:
/// - `phi` → ConsciousnessEngine::consciousness_level
/// - `coherence_time` → temporal_coherence_score
/// - `complexity` → structural_micro_phi
/// - `differentiation` → information_content
#[derive(Debug, Clone)]
pub struct PhysicsConsciousnessState {
    /// Integrated information Φ derived from orbital entanglement [0, 1]
    pub phi: f64,
    /// Temporal coherence: how long quantum states persist (atomic units)
    pub coherence_time: f64,
    /// Complexity: integration × differentiation (Tononi's main complex)
    pub complexity: f64,
    /// Number of distinguishable states (information capacity)
    pub differentiation: f64,
    /// HOMO-LUMO gap (eV) — bandwidth for information processing
    pub energy_gap_ev: f64,
    /// Quantum integration: entanglement across subsystems
    pub quantum_integration: f64,
    /// Orbital delocalization: how spread molecular orbitals are
    pub delocalization: f64,
    /// Free energy available for computation (Hartree)
    pub free_energy: f64,
    /// Method-level accuracy indicator
    pub method: ComputationalMethod,
    /// Electron correlation fraction (MP2 correction / HF energy)
    pub correlation_fraction: f64,
}

/// Which computational method produced these results.
#[derive(Debug, Clone, Copy)]
pub enum ComputationalMethod {
    HartreeFock,
    Mp2,
    Dft,
    CoupledCluster,
}

/// Convert a full consciousness profile to the bridge format.
pub fn profile_to_bridge(
    profile: &ConsciousnessProfile,
    rhf: &RhfResult,
    correlation_energy: f64,
    method: ComputationalMethod,
) -> PhysicsConsciousnessState {
    let correlation_fraction = if rhf.total_energy.abs() > 1e-10 {
        correlation_energy.abs() / rhf.total_energy.abs()
    } else {
        0.0
    };

    PhysicsConsciousnessState {
        phi: profile.total_score,
        coherence_time: profile.temporal_depth,
        complexity: profile.complexity,
        differentiation: profile.differentiation,
        energy_gap_ev: profile.components.homo_lumo_gap_ev,
        quantum_integration: profile.phi_quantum,
        delocalization: profile.phi_orbital,
        free_energy: profile.free_energy_consciousness,
        method,
        correlation_fraction,
    }
}

/// Quick consciousness assessment: run HF + consciousness on a molecule.
/// Returns the bridge state ready for the cognitive loop.
pub fn quick_consciousness_assessment(
    molecule: &crate::molecule::Molecule,
    temperature: f64,
) -> PhysicsConsciousnessState {
    use crate::basis::BasisSetProvider;
    use crate::basis::sto3g::Sto3g;
    use crate::emergent_consciousness::consciousness_from_first_principles;
    use crate::integrals::eri::compute_eri_tensor;
    use crate::post_hf::mp2::mp2_correlation_energy;
    use crate::scf::rhf::{RhfConfig, restricted_hartree_fock};

    let basis = Sto3g::build(molecule);
    let rhf = restricted_hartree_fock(molecule, &basis, &RhfConfig::default());
    let (eri, _, _) = compute_eri_tensor(&basis.functions);
    let mp2 = mp2_correlation_energy(&rhf, &eri);

    let profile =
        consciousness_from_first_principles(&rhf, &molecule.atoms, &basis.functions, temperature);

    profile_to_bridge(
        &profile,
        &rhf,
        mp2.correlation_energy,
        ComputationalMethod::Mp2,
    )
}

/// Molecular consciousness comparison: rank molecules by consciousness score.
pub fn rank_by_consciousness(
    molecules: &[(&str, crate::molecule::Molecule)],
    temperature: f64,
) -> Vec<(String, PhysicsConsciousnessState)> {
    let mut results: Vec<(String, PhysicsConsciousnessState)> = molecules
        .iter()
        .map(|(name, mol)| {
            let state = quick_consciousness_assessment(mol, temperature);
            (name.to_string(), state)
        })
        .collect();

    results.sort_by(|a, b| b.1.phi.total_cmp(&a.1.phi));
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::Molecule;

    #[test]
    fn test_quick_assessment() {
        let state = quick_consciousness_assessment(&Molecule::water(), 300.0);
        assert!(state.phi > 0.0 && state.phi <= 1.0);
        assert!(state.energy_gap_ev > 0.0);
        assert!(state.correlation_fraction > 0.0);
    }

    #[test]
    fn test_ranking() {
        let molecules = vec![("H2", Molecule::h2()), ("H2O", Molecule::water())];
        let ranking = rank_by_consciousness(&molecules, 300.0);
        assert_eq!(ranking.len(), 2);
        // Water should rank higher than H2
        assert_eq!(ranking[0].0, "H2O");
    }

    #[test]
    fn test_bridge_fields_reasonable() {
        let state = quick_consciousness_assessment(&Molecule::h2(), 300.0);
        assert!(state.coherence_time >= 0.0);
        assert!(state.delocalization >= 0.0 && state.delocalization <= 1.0);
    }
}
