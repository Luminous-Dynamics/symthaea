// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gen-7 HYPERION-FL Host: Gradient Validity Proof Generator
//!
//! This host program interfaces with the RISC Zero zkVM to generate and verify
//! proofs of gradient validity for federated learning.
//!
//! ## Features
//!
//! - Generate zkSTARK proofs for gradient validity
//! - Verify proofs locally before submission
//! - Export proofs for on-chain/DHT verification
//! - Support for batch proof generation

use methods::{METHOD_ELF, METHOD_ID};
use risc0_zkvm::{default_prover, ExecutorEnv, Receipt};
use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};
use std::time::Instant;

// =============================================================================
// Type Definitions
// =============================================================================

/// Public inputs for the gradient validity proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientProofInputs {
    /// Node identifier (32-byte hash of public key)
    pub node_id: [u8; 32],

    /// Training round number (for replay protection)
    pub round_number: u64,

    /// Hash of the global model being updated
    pub model_hash: [u8; 32],

    /// Maximum allowed L2 norm squared (0 = use default)
    pub max_norm_squared: i64,
}

/// The gradient data to prove (private witness)
#[derive(Debug, Clone)]
pub struct GradientWitness {
    /// Gradient vector in Q16.16 fixed-point format
    pub gradient: Vec<i32>,
}

/// Outputs extracted from verified proof journal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientProofOutputs {
    /// Node that generated this gradient
    pub node_id: [u8; 32],

    /// Round number (replay protection)
    pub round_number: u64,

    /// Verified gradient hash
    pub gradient_hash: [u8; 32],

    /// Model hash
    pub model_hash: [u8; 32],

    /// Number of gradient elements
    pub gradient_len: u32,

    /// L2 norm squared (Q16.16)
    pub norm_squared: i64,

    /// Mean gradient value (Q16.16)
    pub mean: i32,

    /// Gradient variance (Q16.16)
    pub variance: i32,

    /// Validity flag
    pub is_valid: bool,
}

/// Result of proof generation
#[derive(Debug)]
pub struct ProofResult {
    /// The zkSTARK receipt (proof)
    pub receipt: Receipt,

    /// Extracted public outputs
    pub outputs: GradientProofOutputs,

    /// Proving time in milliseconds
    pub proving_time_ms: u128,

    /// Number of cycles used
    pub total_cycles: u64,

    /// User cycles (excluding syscalls)
    pub user_cycles: u64,
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute SHA256 hash of gradient (Q16.16 fixed-point values)
pub fn hash_gradient(gradient: &[i32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for &g in gradient {
        hasher.update(g.to_le_bytes());
    }
    hasher.finalize().into()
}

/// Convert f32 to Q16.16 fixed-point
pub fn f32_to_fixed(f: f32) -> i32 {
    (f * 65536.0) as i32
}

/// Convert Q16.16 fixed-point to f32
pub fn fixed_to_f32(fixed: i32) -> f32 {
    fixed as f32 / 65536.0
}

/// Convert f32 gradient vector to Q16.16 fixed-point
pub fn gradient_to_fixed(gradient: &[f32]) -> Vec<i32> {
    gradient.iter().map(|&x| f32_to_fixed(x)).collect()
}

/// Convert Q16.16 gradient vector back to f32
pub fn fixed_to_gradient(gradient: &[i32]) -> Vec<f32> {
    gradient.iter().map(|&x| fixed_to_f32(x)).collect()
}

// =============================================================================
// Proof Generation
// =============================================================================

/// Generate a zkSTARK proof of gradient validity
///
/// # Arguments
/// * `inputs` - Public inputs (node ID, round, model hash, norm bound)
/// * `witness` - Private witness (the gradient itself)
///
/// # Returns
/// * `Ok(ProofResult)` - Proof with extracted outputs and statistics
/// * `Err` - If proof generation fails
pub fn prove_gradient_validity(
    inputs: &GradientProofInputs,
    witness: &GradientWitness,
) -> Result<ProofResult, Box<dyn std::error::Error>> {
    // Compute gradient hash for commitment
    let gradient_hash = hash_gradient(&witness.gradient);
    let gradient_len = witness.gradient.len() as u32;

    // Build executor environment
    let mut env_builder = ExecutorEnv::builder();

    // Write public inputs
    env_builder.write(&inputs.node_id)?;
    env_builder.write(&inputs.round_number)?;
    env_builder.write(&gradient_hash)?;
    env_builder.write(&inputs.model_hash)?;
    env_builder.write(&inputs.max_norm_squared)?;
    env_builder.write(&gradient_len)?;

    // Write private witness (gradient values)
    for &g in &witness.gradient {
        env_builder.write(&g)?;
    }

    let env = env_builder.build()?;

    // Generate proof
    let start = Instant::now();
    let prover = default_prover();
    let prove_info = prover.prove(env, METHOD_ELF)?;
    let proving_time_ms = start.elapsed().as_millis();

    // Extract outputs from journal
    let outputs = extract_proof_outputs(&prove_info.receipt)?;

    Ok(ProofResult {
        receipt: prove_info.receipt,
        outputs,
        proving_time_ms,
        total_cycles: prove_info.stats.total_cycles,
        user_cycles: prove_info.stats.user_cycles,
    })
}

/// Extract public outputs from proof journal
fn extract_proof_outputs(receipt: &Receipt) -> Result<GradientProofOutputs, Box<dyn std::error::Error>> {
    let journal = &receipt.journal;
    let mut offset = 0;

    // Helper to read from journal bytes
    fn read_bytes<const N: usize>(data: &[u8], offset: &mut usize) -> [u8; N] {
        let mut arr = [0u8; N];
        arr.copy_from_slice(&data[*offset..*offset + N]);
        *offset += N;
        arr
    }

    fn read_u32(data: &[u8], offset: &mut usize) -> u32 {
        let bytes = read_bytes::<4>(data, offset);
        u32::from_le_bytes(bytes)
    }

    fn read_u64(data: &[u8], offset: &mut usize) -> u64 {
        let bytes = read_bytes::<8>(data, offset);
        u64::from_le_bytes(bytes)
    }

    fn read_i32(data: &[u8], offset: &mut usize) -> i32 {
        let bytes = read_bytes::<4>(data, offset);
        i32::from_le_bytes(bytes)
    }

    fn read_i64(data: &[u8], offset: &mut usize) -> i64 {
        let bytes = read_bytes::<8>(data, offset);
        i64::from_le_bytes(bytes)
    }

    let bytes = journal.bytes.as_slice();

    let node_id = read_bytes::<32>(bytes, &mut offset);
    let round_number = read_u64(bytes, &mut offset);
    let gradient_hash = read_bytes::<32>(bytes, &mut offset);
    let model_hash = read_bytes::<32>(bytes, &mut offset);
    let gradient_len = read_u32(bytes, &mut offset);
    let norm_squared = read_i64(bytes, &mut offset);
    let mean = read_i32(bytes, &mut offset);
    let variance = read_i32(bytes, &mut offset);
    let is_valid = bytes[offset] == 1;

    Ok(GradientProofOutputs {
        node_id,
        round_number,
        gradient_hash,
        model_hash,
        gradient_len,
        norm_squared,
        mean,
        variance,
        is_valid,
    })
}

// =============================================================================
// Proof Verification
// =============================================================================

/// Verify a zkSTARK proof of gradient validity
///
/// # Arguments
/// * `receipt` - The proof receipt to verify
///
/// # Returns
/// * `Ok(GradientProofOutputs)` - Verified outputs if proof is valid
/// * `Err` - If verification fails
pub fn verify_gradient_proof(receipt: &Receipt) -> Result<GradientProofOutputs, Box<dyn std::error::Error>> {
    // Verify the receipt cryptographically
    receipt.verify(METHOD_ID)?;

    // Extract and return outputs
    extract_proof_outputs(receipt)
}

/// Verify a serialized proof (for network/storage)
pub fn verify_serialized_proof(proof_bytes: &[u8]) -> Result<GradientProofOutputs, Box<dyn std::error::Error>> {
    let receipt: Receipt = bincode::deserialize(proof_bytes)?;
    verify_gradient_proof(&receipt)
}

// =============================================================================
// Proof Serialization
// =============================================================================

/// Serialize a receipt for storage/transmission
pub fn serialize_proof(receipt: &Receipt) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    Ok(bincode::serialize(receipt)?)
}

/// Deserialize a receipt from bytes
pub fn deserialize_proof(bytes: &[u8]) -> Result<Receipt, Box<dyn std::error::Error>> {
    Ok(bincode::deserialize(bytes)?)
}

// =============================================================================
// Test Utilities
// =============================================================================

/// Generate a random valid gradient for testing
pub fn generate_test_gradient(size: usize, scale: f32) -> Vec<i32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut gradient = Vec::with_capacity(size);
    let mut hasher = DefaultHasher::new();

    for i in 0..size {
        i.hash(&mut hasher);
        let hash = hasher.finish();
        // Generate value in [-scale, scale]
        let normalized = (hash as f32 / u64::MAX as f32) * 2.0 - 1.0;
        gradient.push(f32_to_fixed(normalized * scale));
    }

    gradient
}

/// Generate a test node ID
pub fn generate_test_node_id(seed: u64) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(seed.to_le_bytes());
    hasher.finalize().into()
}

/// Generate a test model hash
pub fn generate_test_model_hash(seed: u64) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"model_");
    hasher.update(seed.to_le_bytes());
    hasher.finalize().into()
}

// =============================================================================
// Main Entry Point
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for debugging
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::filter::EnvFilter::from_default_env())
        .init();

    println!("============================================================");
    println!("Gen-7 HYPERION-FL: Gradient Validity Proof Generator");
    println!("============================================================\n");

    // =========================================================================
    // Demo 1: Valid Gradient Proof
    // =========================================================================
    println!("[Demo 1] Generating proof for VALID gradient...\n");

    let node_id = generate_test_node_id(12345);
    let model_hash = generate_test_model_hash(1);
    let gradient = generate_test_gradient(1000, 1.0); // 1000 elements, scale=1.0

    let inputs = GradientProofInputs {
        node_id,
        round_number: 42,
        model_hash,
        max_norm_squared: 0, // Use default
    };

    let witness = GradientWitness {
        gradient: gradient.clone(),
    };

    println!("  Node ID: {:?}...", &node_id[..8]);
    println!("  Round: {}", inputs.round_number);
    println!("  Gradient size: {} elements", witness.gradient.len());
    println!("  Gradient hash: {:?}...", &hash_gradient(&witness.gradient)[..8]);
    println!();

    let result = prove_gradient_validity(&inputs, &witness)?;

    println!("  Proof generated successfully!");
    println!("  - Proving time: {} ms", result.proving_time_ms);
    println!("  - Total cycles: {}", result.total_cycles);
    println!("  - User cycles: {}", result.user_cycles);
    println!("  - Proof size: {} bytes", serialize_proof(&result.receipt)?.len());
    println!();

    // Verify the proof
    let verified = verify_gradient_proof(&result.receipt)?;
    println!("  Proof verified successfully!");
    println!("  - Node ID matches: {}", verified.node_id == node_id);
    println!("  - Round matches: {}", verified.round_number == 42);
    println!("  - Is valid: {}", verified.is_valid);
    println!("  - Norm squared: {:.4}", verified.norm_squared as f64 / 65536.0);
    println!("  - Mean: {:.6}", fixed_to_f32(verified.mean));
    println!("  - Variance: {:.6}", fixed_to_f32(verified.variance));
    println!();

    // =========================================================================
    // Demo 2: Invalid Gradient (too large norm) - Will fail in guest
    // =========================================================================
    println!("[Demo 2] Testing gradient with EXCESSIVE NORM...\n");

    // Create gradient with very large values
    let large_gradient: Vec<i32> = (0..100)
        .map(|_| f32_to_fixed(10000.0)) // Very large values
        .collect();

    let inputs2 = GradientProofInputs {
        node_id: generate_test_node_id(99999),
        round_number: 1,
        model_hash: generate_test_model_hash(1),
        max_norm_squared: 1000 * 65536, // Small limit
    };

    let witness2 = GradientWitness {
        gradient: large_gradient,
    };

    println!("  Testing with gradient norm exceeding limit...");
    match prove_gradient_validity(&inputs2, &witness2) {
        Ok(_) => println!("  ERROR: Proof should have failed!"),
        Err(e) => println!("  Expected failure: {}", e),
    }
    println!();

    // =========================================================================
    // Demo 3: Commitment Mismatch Test
    // =========================================================================
    println!("[Demo 3] This would fail if we could modify the gradient after commitment");
    println!("  (The zkVM prevents this by design - commitment is verified inside the proof)\n");

    // =========================================================================
    // Summary
    // =========================================================================
    println!("============================================================");
    println!("Demo Complete!");
    println!("============================================================");
    println!();
    println!("The zkSTARK proof cryptographically guarantees:");
    println!("  1. Gradient matches the pre-committed hash");
    println!("  2. Gradient contains no invalid values");
    println!("  3. Gradient L2 norm is within bounds");
    println!("  4. Gradient is attributed to correct node/round");
    println!();
    println!("Private data (training samples) is NEVER revealed.");

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_gradient_proof() {
        let node_id = generate_test_node_id(1);
        let model_hash = generate_test_model_hash(1);
        let gradient = generate_test_gradient(100, 0.5);

        let inputs = GradientProofInputs {
            node_id,
            round_number: 1,
            model_hash,
            max_norm_squared: 0,
        };

        let witness = GradientWitness { gradient };

        let result = prove_gradient_validity(&inputs, &witness).expect("Proof should succeed");

        assert!(result.outputs.is_valid);
        assert_eq!(result.outputs.node_id, node_id);
        assert_eq!(result.outputs.round_number, 1);
    }

    #[test]
    fn test_gradient_hash_consistency() {
        let gradient = vec![f32_to_fixed(1.0), f32_to_fixed(-0.5), f32_to_fixed(0.25)];
        let hash1 = hash_gradient(&gradient);
        let hash2 = hash_gradient(&gradient);
        assert_eq!(hash1, hash2);

        let gradient2 = vec![f32_to_fixed(1.0), f32_to_fixed(-0.5), f32_to_fixed(0.26)];
        let hash3 = hash_gradient(&gradient2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_fixed_point_conversion() {
        let values = vec![0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0];
        for v in values {
            let fixed = f32_to_fixed(v);
            let back = fixed_to_f32(fixed);
            assert!((v - back).abs() < 0.001, "Conversion failed for {}", v);
        }
    }

    #[test]
    fn test_proof_serialization() {
        let node_id = generate_test_node_id(42);
        let model_hash = generate_test_model_hash(42);
        let gradient = generate_test_gradient(50, 0.1);

        let inputs = GradientProofInputs {
            node_id,
            round_number: 100,
            model_hash,
            max_norm_squared: 0,
        };

        let witness = GradientWitness { gradient };

        let result = prove_gradient_validity(&inputs, &witness).expect("Proof should succeed");

        // Serialize and deserialize
        let bytes = serialize_proof(&result.receipt).expect("Serialization should succeed");
        let deserialized = deserialize_proof(&bytes).expect("Deserialization should succeed");

        // Verify deserialized proof
        let verified = verify_gradient_proof(&deserialized).expect("Verification should succeed");
        assert!(verified.is_valid);
        assert_eq!(verified.node_id, node_id);
    }
}
