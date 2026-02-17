#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;

use fl_aggregator::proofs::{GradientIntegrityProof, ProofConfig, SecurityLevel};

#[derive(Arbitrary, Debug)]
struct GradientProofInput {
    // Use small fixed-size arrays to keep fuzzing fast
    gradients: [f32; 16],
    gradient_count: u8, // Actual count 1-16
    max_norm_raw: u32,  // Will be converted to f32
}

fuzz_target!(|input: GradientProofInput| {
    // Limit gradient count to 1-16
    let count = ((input.gradient_count as usize) % 16) + 1;
    let gradients: Vec<f32> = input.gradients[..count]
        .iter()
        .copied()
        // Filter out NaN and infinite values
        .filter(|&g| g.is_finite())
        // Clamp to reasonable range
        .map(|g| g.clamp(-100.0, 100.0))
        .collect();

    if gradients.is_empty() {
        return;
    }

    // Convert max_norm to a reasonable positive value
    let max_norm = (input.max_norm_raw as f32 / 1000.0).max(0.1).min(1000.0);

    let config = ProofConfig {
        security_level: SecurityLevel::Standard96,
        parallel: false,
        max_proof_size: 0,
    };

    // Try to generate proof - should not panic
    match GradientIntegrityProof::generate(&gradients, max_norm, config) {
        Ok(proof) => {
            // Verification should not panic
            let _ = proof.verify();

            // Serialization should not panic
            let bytes = proof.to_bytes();

            // Deserialization should not panic
            if let Ok(restored) = GradientIntegrityProof::from_bytes(&bytes) {
                let _ = restored.verify();
            }
        }
        Err(_) => {
            // Generation can fail for norm violations
        }
    }
});
