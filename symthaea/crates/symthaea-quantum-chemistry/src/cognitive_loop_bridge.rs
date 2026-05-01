// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Bridge to Symthaea's CognitiveLoopService.
//!
//! Maps first-principles physics into the ConsciousnessEquationV2 interface:
//!
//! - `substrate_feasibility` ← multi-theory composite score
//! - `core_values[Integration]` ← IIT bipartition entanglement
//! - `core_values[Binding]` ← orbital delocalization (GWT)
//! - `core_values[Workspace]` ← complexity (MI network density)
//! - `core_values[Attention]` ← HOT excitation gap
//! - `core_values[Recursion]` ← Orch-OR coherence time
//! - `core_values[Efficacy]` ← FEP free energy (thermodynamic order)
//! - `core_values[Knowledge]` ← autopoiesis correlation fraction
//!
//! ## Usage in CognitiveLoopService
//!
//! ```ignore
//! // In cycle_subsystems.rs or cycle_phase_feedback.rs:
//! let physics = cognitive_loop_bridge::physics_to_consciousness_state(
//!     &molecule, 300.0
//! );
//!
//! // Replace the hardcoded 1.0:
//! consciousness_state.substrate_feasibility = physics.substrate_feasibility;
//!
//! // Optionally, ground the 7 core components in physics:
//! for (component, value) in &physics.core_values {
//!     consciousness_state.core_values.insert(component.clone(), *value);
//! }
//! ```

use crate::molecule::Molecule;
use crate::multi_theory::{compute_theory_scores, THEORY_NAMES};

/// Physics-grounded consciousness state for the cognitive loop.
///
/// This struct maps directly to ConsciousnessStateV2 fields:
/// - `substrate_feasibility` → ConsciousnessStateV2.substrate_feasibility
/// - `core_values` → ConsciousnessStateV2.core_values (7 CoreComponent entries)
#[derive(Debug, Clone)]
pub struct PhysicsGroundedState {
    /// Maps to ConsciousnessStateV2.substrate_feasibility [0, 1]
    /// Derived from multi-theory composite score
    pub substrate_feasibility: f64,

    /// Maps to core_values[Integration] — IIT entanglement
    pub integration: f64,
    /// Maps to core_values[Binding] — orbital delocalization (GWT)
    pub binding: f64,
    /// Maps to core_values[Workspace] — MI network density (complexity)
    pub workspace: f64,
    /// Maps to core_values[Attention] — excitation gap accessibility (HOT)
    pub attention: f64,
    /// Maps to core_values[Recursion] — coherence time (Orch-OR)
    pub recursion: f64,
    /// Maps to core_values[Efficacy] — free energy order (FEP)
    pub efficacy: f64,
    /// Maps to core_values[Knowledge] — correlation fraction (autopoiesis)
    pub knowledge: f64,

    /// Additional metrics not in ConsciousnessEquationV2
    pub info_geometry_phi: f64,
    pub dissipative_entropy: f64,
    pub quantum_darwinism: f64,

    /// Diagnostic: which theory is the bottleneck?
    pub limiting_theory: String,
}

/// Compute the physics-grounded consciousness state for the cognitive loop.
///
/// This is the main entry point. Call this on a molecular substrate
/// to get physics-derived values for all 7 ConsciousnessEquationV2 components.
pub fn physics_to_consciousness_state(
    molecule: &Molecule,
    temperature: f64,
) -> PhysicsGroundedState {
    let scores = compute_theory_scores(molecule, "substrate", temperature);

    PhysicsGroundedState {
        substrate_feasibility: scores.composite_score,

        // Map 10 theories → 7 ConsciousnessEquationV2 components
        integration: scores.normalized[0], // IIT
        binding: scores.normalized[1],     // GWT
        workspace: scores.normalized[6],   // Complexity
        attention: scores.normalized[3],   // HOT
        recursion: scores.normalized[4],   // Orch-OR
        efficacy: scores.normalized[2],    // FEP
        knowledge: scores.normalized[5],   // Autopoiesis

        // Extra metrics (for extended_values)
        info_geometry_phi: scores.normalized[7],
        dissipative_entropy: scores.normalized[8],
        quantum_darwinism: scores.normalized[9],

        limiting_theory: THEORY_NAMES[scores.limiting_theory].to_string(),
    }
}

/// Convenience: compute substrate feasibility only (for minimal integration).
/// Returns a single f64 to replace the hardcoded 1.0 in ConsciousnessStateV2.
pub fn substrate_feasibility_from_physics(molecule: &Molecule, temperature: f64) -> f64 {
    let scores = compute_theory_scores(molecule, "substrate", temperature);
    scores.composite_score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_physics_state_bounded() {
        let state = physics_to_consciousness_state(&Molecule::water(), 300.0);
        assert!(state.substrate_feasibility >= 0.0 && state.substrate_feasibility <= 1.0);
        assert!(state.integration >= 0.0 && state.integration <= 1.0);
        assert!(state.binding >= 0.0 && state.binding <= 1.0);
        assert!(state.workspace >= 0.0 && state.workspace <= 1.0);
        assert!(state.attention >= 0.0 && state.attention <= 1.0);
        assert!(state.recursion >= 0.0 && state.recursion <= 1.0);
        assert!(state.efficacy >= 0.0 && state.efficacy <= 1.0);
        assert!(state.knowledge >= 0.0 && state.knowledge <= 1.0);
    }

    #[test]
    fn test_substrate_feasibility_positive() {
        let f = substrate_feasibility_from_physics(&Molecule::water(), 300.0);
        assert!(f > 0.0, "Water should have positive feasibility: {}", f);
    }
}
