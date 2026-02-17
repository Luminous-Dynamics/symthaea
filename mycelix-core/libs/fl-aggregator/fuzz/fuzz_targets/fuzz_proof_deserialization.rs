#![no_main]

use libfuzzer_sys::fuzz_target;

use fl_aggregator::proofs::{
    RangeProof, MembershipProof, GradientIntegrityProof,
    IdentityAssuranceProof, VoteEligibilityProof,
};

fuzz_target!(|data: &[u8]| {
    // Try to deserialize as each proof type
    // These should never panic, only return errors for invalid data

    // RangeProof
    if let Ok(proof) = RangeProof::from_bytes(data) {
        // If deserialization succeeded, verification should not panic
        let _ = proof.verify();
    }

    // MembershipProof
    if let Ok(proof) = MembershipProof::from_bytes(data) {
        let _ = proof.verify();
    }

    // GradientIntegrityProof
    if let Ok(proof) = GradientIntegrityProof::from_bytes(data) {
        let _ = proof.verify();
    }

    // IdentityAssuranceProof
    if let Ok(proof) = IdentityAssuranceProof::from_bytes(data) {
        let _ = proof.verify();
    }

    // VoteEligibilityProof
    if let Ok(proof) = VoteEligibilityProof::from_bytes(data) {
        let _ = proof.verify();
    }
});
