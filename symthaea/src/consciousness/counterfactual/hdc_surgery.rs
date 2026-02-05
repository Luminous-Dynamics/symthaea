//! Graph Surgery via HDC Unbinding
//!
//! Implements causal graph surgery in hyperdimensional space:
//! - Remove effects of a variable by unbinding its causal vectors
//! - Compute interventional distributions via do-calculus in HDC
//! - Round-trip verified: unbind(bind(x)) ≈ x

use symthaea_core::hdc::binary_hv::HV16;

/// Graph surgery operations on causal spaces.
pub struct GraphSurgery;

impl GraphSurgery {
    /// Remove the effect of a variable by unbinding its causal vector.
    ///
    /// In Pearl's framework: graph surgery removes arrows INTO the treatment.
    /// In HDC: unbind the treatment's cause vector to "disconnect" it.
    ///
    /// Returns the "intervened" causal vector.
    pub fn remove_incoming(causal_vector: &HV16, cause: &HV16) -> HV16 {
        // Unbind: causal_vector ⊗ cause = effect
        // This "removes" the cause from the causal relationship
        causal_vector.bind(cause)
    }

    /// Reconstruct the effect after removing a cause.
    ///
    /// Given C = cause ⊗ effect, unbinding with cause recovers effect:
    /// C ⊗ cause = (cause ⊗ effect) ⊗ cause = effect
    /// (because XOR is self-inverse)
    pub fn recover_effect(causal_vector: &HV16, cause: &HV16) -> HV16 {
        causal_vector.bind(cause)
    }

    /// Reconstruct the cause after removing an effect.
    pub fn recover_cause(causal_vector: &HV16, effect: &HV16) -> HV16 {
        causal_vector.bind(effect)
    }

    /// Verify round-trip: bind then unbind should recover the original.
    ///
    /// Returns similarity between recovered and original (should be ~1.0).
    pub fn verify_roundtrip(original: &HV16, binding_partner: &HV16) -> f32 {
        let bound = original.bind(binding_partner);
        let recovered = bound.bind(binding_partner);
        recovered.similarity(original)
    }

    /// Remove edges into a node (do-calculus graph surgery).
    ///
    /// Given a set of causal vectors (cause_i ⊗ effect) where effect is the
    /// treatment node, remove them all to simulate do(treatment = value).
    pub fn do_surgery(
        causal_vectors: &[HV16],
        causes: &[HV16],
        treatment: &HV16,
    ) -> Vec<HV16> {
        causal_vectors
            .iter()
            .zip(causes.iter())
            .map(|(cv, cause)| {
                // Check if this causal vector involves the treatment as effect
                let recovered_effect = Self::recover_effect(cv, cause);
                let is_into_treatment = recovered_effect.similarity(treatment) > 0.7;

                if is_into_treatment {
                    // Remove this edge (return zero-like vector)
                    HV16::zero()
                } else {
                    // Keep this edge
                    cv.clone()
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
        let cause = HV16::random(1);
        let effect = HV16::random(2);

        let similarity = GraphSurgery::verify_roundtrip(&cause, &effect);
        assert!(
            similarity > 0.99,
            "Round-trip should recover original with high similarity, got {}",
            similarity
        );
    }

    #[test]
    fn test_recover_effect() {
        let cause = HV16::random(10);
        let effect = HV16::random(20);
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
        let cause = HV16::random(30);
        let effect = HV16::random(40);
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
        let cause1 = HV16::random(1);
        let cause2 = HV16::random(2);
        let treatment = HV16::random(3);
        let other = HV16::random(4);

        // Create causal vectors: cause1→treatment, cause2→other
        let cv1 = cause1.bind(&treatment);
        let cv2 = cause2.bind(&other);

        let result = GraphSurgery::do_surgery(
            &[cv1, cv2],
            &[cause1, cause2],
            &treatment,
        );

        // First edge (into treatment) should be removed (zero vector)
        assert_eq!(result[0], HV16::zero(), "Edge into treatment should be removed");
        // Second edge (not into treatment) should be kept
        assert_ne!(result[1], HV16::zero(), "Edge not into treatment should be kept");
    }
}
