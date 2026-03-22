// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Graph Surgery via HDC Unbinding
//!
//! Implements causal graph surgery in hyperdimensional space:
//! - Remove effects of a variable by unbinding its causal vectors
//! - Compute interventional distributions via do-calculus in HDC
//! - Round-trip verified: unbind(bind(x)) ≈ x

use symthaea_core::hdc::binary_hv::BinaryHV;

/// Graph surgery operations on causal spaces.
pub struct GraphSurgery;

impl GraphSurgery {
    /// Remove the effect of a variable by unbinding its causal vector.
    ///
    /// In Pearl's framework: graph surgery removes arrows INTO the treatment.
    /// In HDC: unbind the treatment's cause vector to "disconnect" it.
    ///
    /// Returns the "intervened" causal vector.
    pub fn remove_incoming(causal_vector: &BinaryHV, cause: &BinaryHV) -> BinaryHV {
        // Unbind: causal_vector ⊗ cause = effect
        // This "removes" the cause from the causal relationship
        causal_vector.bind(cause)
    }

    /// Reconstruct the effect after removing a cause.
    ///
    /// Given C = cause ⊗ effect, unbinding with cause recovers effect:
    /// C ⊗ cause = (cause ⊗ effect) ⊗ cause = effect
    /// (because XOR is self-inverse)
    pub fn recover_effect(causal_vector: &BinaryHV, cause: &BinaryHV) -> BinaryHV {
        causal_vector.bind(cause)
    }

    /// Reconstruct the cause after removing an effect.
    pub fn recover_cause(causal_vector: &BinaryHV, effect: &BinaryHV) -> BinaryHV {
        causal_vector.bind(effect)
    }

    /// Verify round-trip: bind then unbind should recover the original.
    ///
    /// Returns similarity between recovered and original (should be ~1.0).
    pub fn verify_roundtrip(original: &BinaryHV, binding_partner: &BinaryHV) -> f32 {
        let bound = original.bind(binding_partner);
        let recovered = bound.bind(binding_partner);
        recovered.similarity(original)
    }

    /// Remove edges into a node (do-calculus graph surgery).
    ///
    /// Given a set of causal vectors (cause_i ⊗ effect) where effect is the
    /// treatment node, remove them all to simulate do(treatment = value).
    pub fn do_surgery(
        causal_vectors: &[BinaryHV],
        causes: &[BinaryHV],
        treatment: &BinaryHV,
    ) -> Vec<BinaryHV> {
        causal_vectors
            .iter()
            .zip(causes.iter())
            .map(|(cv, cause)| {
                // Check if this causal vector involves the treatment as effect
                let recovered_effect = Self::recover_effect(cv, cause);
                let is_into_treatment = recovered_effect.similarity(treatment) > 0.7;

                if is_into_treatment {
                    // Remove this edge (return zero-like vector)
                    BinaryHV::zero()
                } else {
                    // Keep this edge
                    *cv
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_unbind() {
        let cause = BinaryHV::random(1);
        let effect = BinaryHV::random(2);

        let similarity = GraphSurgery::verify_roundtrip(&cause, &effect);
        assert!(
            similarity > 0.99,
            "Round-trip should recover original with high similarity, got {}",
            similarity
        );
    }

    #[test]
    fn test_recover_effect() {
        let cause = BinaryHV::random(10);
        let effect = BinaryHV::random(20);
        let causal_vector = cause.bind(&effect);

        let recovered = GraphSurgery::recover_effect(&causal_vector, &cause);
        let similarity = recovered.similarity(&effect);
        assert!(
            similarity > 0.99,
            "Should recover effect from causal vector, got similarity {}",
            similarity
        );
    }

    #[test]
    fn test_recover_cause() {
        let cause = BinaryHV::random(30);
        let effect = BinaryHV::random(40);
        let causal_vector = cause.bind(&effect);

        let recovered = GraphSurgery::recover_cause(&causal_vector, &effect);
        let similarity = recovered.similarity(&cause);
        assert!(
            similarity > 0.99,
            "Should recover cause from causal vector, got similarity {}",
            similarity
        );
    }

    #[test]
    fn test_surgery_removes_edge() {
        let cause1 = BinaryHV::random(1);
        let cause2 = BinaryHV::random(2);
        let treatment = BinaryHV::random(3);
        let other = BinaryHV::random(4);

        // Create causal vectors: cause1→treatment, cause2→other
        let cv1 = cause1.bind(&treatment);
        let cv2 = cause2.bind(&other);

        let result = GraphSurgery::do_surgery(&[cv1, cv2], &[cause1, cause2], &treatment);

        // First edge (into treatment) should be removed (zero vector)
        assert_eq!(
            result[0],
            BinaryHV::zero(),
            "Edge into treatment should be removed"
        );
        // Second edge (not into treatment) should be kept
        assert_ne!(
            result[1],
            BinaryHV::zero(),
            "Edge not into treatment should be kept"
        );
    }
}
