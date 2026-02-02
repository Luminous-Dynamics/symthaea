//! Gradient Quality Verification Guest Program
//!
//! This code runs inside the RISC-0 zkVM and verifies gradient quality
//! without revealing the actual gradient values.
//!
//! # Proof Statement
//!
//! "I possess a gradient G computed from global model M with the following properties:
//! - L2 norm is within [min_norm, max_norm]
//! - All elements have magnitude ≤ max_element_magnitude
//! - Dimension is ≥ min_dimension
//! - All values are finite (no NaN/Inf)"
//!
//! # Privacy Guarantees
//!
//! - The gradient vector NEVER leaves the zkVM
//! - Only the hash commitment is revealed
//! - Verifier learns nothing about actual gradient values

use risc0_zkvm::guest::env;
use zk_gradient_core::{
    GradientProofInput, GradientProofOutput,
    verify_gradient_quality, compute_gradient_hash, compute_commitment,
    hash_to_bytes32,
};

fn main() {
    // Step 1: Read private witness from host
    let input: GradientProofInput = env::read();

    // Step 2: Verify gradient quality inside zkVM
    // This is the core computation - gradient values never leave here
    let (quality_valid, _l2_norm, _max_element) = verify_gradient_quality(
        &input.gradient,
        &input.constraints,
    );

    // Step 3: Compute gradient hash (commitment to gradient values)
    let gradient_hash_u64 = compute_gradient_hash(&input.gradient);
    let gradient_hash = hash_to_bytes32(gradient_hash_u64);

    // Step 4: Compute binding commitment
    let commitment = compute_commitment(
        gradient_hash_u64,
        &input.global_model_hash,
        input.epochs,
        input.round,
        &input.client_id,
    );

    // Step 5: Construct public output (journal)
    let output = GradientProofOutput {
        gradient_hash,
        global_model_hash: input.global_model_hash,
        epochs: input.epochs,
        learning_rate: input.learning_rate,
        quality_valid,
        client_id: input.client_id,
        round: input.round,
        dimension: input.gradient.len(),
        commitment,
    };

    // Step 6: Commit output to the journal (public data)
    env::commit(&output);
}
