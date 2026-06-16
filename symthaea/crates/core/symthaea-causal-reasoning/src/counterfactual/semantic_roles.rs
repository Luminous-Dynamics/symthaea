// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semantic Role Substitution
//!
//! Enables counterfactual reasoning by substituting semantic roles
//! in causal queries. E.g., "What if the AGENT were different?"

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::binary_hv::BinaryHV;

/// Semantic roles for causal reasoning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SemanticRole {
    /// The entity performing the action.
    Agent,
    /// The entity affected by the action.
    Patient,
    /// The tool or means used.
    Instrument,
    /// The context or environment.
    Context,
    /// The purpose or goal.
    Goal,
    /// The source or origin.
    Source,
    /// The destination or result.
    Destination,
}

/// A role substitution: replace one filler with another.
#[derive(Debug, Clone)]
pub struct RoleSubstitution {
    /// Which role to substitute.
    pub role: SemanticRole,
    /// Original role filler (HDC vector).
    pub original: BinaryHV,
    /// New role filler (HDC vector).
    pub replacement: BinaryHV,
}

impl RoleSubstitution {
    pub fn new(role: SemanticRole, original: BinaryHV, replacement: BinaryHV) -> Self {
        Self {
            role,
            original,
            replacement,
        }
    }

    /// Apply the substitution to a causal vector.
    ///
    /// Unbinds the original filler and binds the replacement.
    /// Result: causal_vector ⊗ original ⊗ replacement
    pub fn apply(&self, causal_vector: &BinaryHV) -> BinaryHV {
        // Unbind original, bind replacement
        let unbound = causal_vector.bind(&self.original);
        unbound.bind(&self.replacement)
    }

    /// Verify the substitution preserved structure.
    ///
    /// The substituted vector should be similar to the original
    /// in all roles except the substituted one.
    pub fn verify_preservation(&self, original_vector: &BinaryHV, substituted: &BinaryHV) -> f32 {
        // The vectors should differ (substitution happened)

        // But not be completely different (structure preserved)
        // Good range: 0.3 - 0.7 (different but related)
        original_vector.similarity(substituted)
    }
}

/// Apply multiple role substitutions to a causal vector.
pub fn apply_substitutions(
    causal_vector: &BinaryHV,
    substitutions: &[RoleSubstitution],
) -> BinaryHV {
    let mut result = *causal_vector;
    for sub in substitutions {
        result = sub.apply(&result);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_substitution_changes_vector() {
        let original = BinaryHV::random(1);
        let replacement = BinaryHV::random(2);
        let context = BinaryHV::random(3);

        let sub = RoleSubstitution::new(SemanticRole::Agent, original, replacement);

        // Create a "causal vector" that includes the original agent
        let causal = context.bind(&original);
        let substituted = sub.apply(&causal);

        // The substituted vector should be different from the original
        assert_ne!(causal, substituted);
    }

    #[test]
    fn test_substitution_roundtrip() {
        let original_filler = BinaryHV::random(10);
        let replacement_filler = BinaryHV::random(20);
        let base = BinaryHV::random(30);

        // Apply substitution
        let sub = RoleSubstitution::new(SemanticRole::Agent, original_filler, replacement_filler);
        let causal = base.bind(&original_filler);
        let substituted = sub.apply(&causal);

        // Reverse substitution should recover original
        let reverse =
            RoleSubstitution::new(SemanticRole::Agent, replacement_filler, original_filler);
        let recovered = reverse.apply(&substituted);

        let similarity = recovered.similarity(&causal);
        assert!(
            similarity > 0.99,
            "Reverse substitution should recover original, got {}",
            similarity
        );
    }

    #[test]
    fn test_multiple_substitutions() {
        let agent = BinaryHV::random(1);
        let patient = BinaryHV::random(2);
        let new_agent = BinaryHV::random(3);
        let new_patient = BinaryHV::random(4);

        let base = agent.bind(&patient);

        let subs = vec![
            RoleSubstitution::new(SemanticRole::Agent, agent, new_agent),
            RoleSubstitution::new(SemanticRole::Patient, patient, new_patient),
        ];

        let result = apply_substitutions(&base, &subs);
        // Result should be different from original
        assert_ne!(result, base);
    }
}
