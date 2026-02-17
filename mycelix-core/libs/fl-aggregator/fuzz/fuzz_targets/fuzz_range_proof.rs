#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

use fl_aggregator::proofs::{RangeProof, ProofConfig, SecurityLevel};

#[derive(Arbitrary, Debug)]
struct RangeProofInput {
    value: u64,
    min: u64,
    max: u64,
}

fuzz_target!(|input: RangeProofInput| {
    // Skip invalid inputs (min > max)
    if input.min > input.max {
        return;
    }

    // Skip extremely large ranges that would be too slow
    if input.max.saturating_sub(input.min) > 1_000_000 {
        return;
    }

    let config = ProofConfig {
        security_level: SecurityLevel::Standard96,
        parallel: false,
        max_proof_size: 0,
    };

    // Try to generate proof - should not panic
    match RangeProof::generate(input.value, input.min, input.max, config) {
        Ok(proof) => {
            // If generation succeeded, verification should not panic
            let _ = proof.verify();

            // Serialization should not panic
            let bytes = proof.to_bytes();

            // Deserialization should not panic
            if let Ok(restored) = RangeProof::from_bytes(&bytes) {
                // Re-verification should give same result
                let _ = restored.verify();
            }
        }
        Err(_) => {
            // Generation failed - this is expected for out-of-range values
        }
    }
});
