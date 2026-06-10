// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! zkSTARK Gradient Proof Module
//!
//! Zero-knowledge proofs for gradient provenance verification.
//! Instead of detecting Byzantine behavior, we cryptographically PROVE honest behavior.
//!
//! # Security Assumptions
//!
//! This module assumes:
//! - **Randomness**: System RNG provides cryptographically secure randomness
//! - **Hash Security**: SHA3-256 is collision-resistant (NIST FIPS 202)
//! - **RISC-0 Soundness**: The zkVM provides 128-bit computational soundness
//! - **Timing**: ZK proof verification should be constant-time
//!
//! ## Threat Model
//!
//! - Adversary can observe proof contents (proofs are public)
//! - Adversary cannot learn private inputs from proofs (zero-knowledge)
//! - Adversary cannot forge valid proofs for false statements (soundness)
//! - Byzantine nodes may submit invalid proofs (detected by verification)
//!
//! ## Limitations
//!
//! - **Simulation mode** provides NO cryptographic guarantees - use RISC-0 for production
//! - Proof generation is computationally expensive (~1-10 seconds)
//! - Private witness data must be protected during proof generation
//! - Memory requirements scale with gradient dimension
//!
//! ## Security Best Practices
//!
//! 1. ALWAYS compile with `risc0` feature for production (simulation is insecure)
//! 2. Zeroize private witnesses after proof generation
//! 3. Verify proofs before accepting any gradient update
//! 4. Use domain-separated commitments for all cryptographic bindings
//!
//! # Proof Statement
//!
//! "I trained on my local data D_i for E epochs with learning rate η,
//! starting from global model W_t, and this gradient is the result."
//!
//! # Properties
//!
//! - **Completeness**: Honest client can always generate valid proof
//! - **Soundness**: Malicious client cannot generate proof for fake gradient
//! - **Zero-knowledge**: Proof reveals nothing about private data D_i
//! - **Succinctness**: Proof size O(log |D_i|), verification O(1)
//!
//! # Architecture
//!
//! Provides two modes:
//! - **Simulation**: Fast testing without real proofs (WARNING: NO SECURITY)
//! - **RISC-0**: Real zkSTARK proofs via RISC Zero zkVM (USE FOR PRODUCTION)
//!
//! # RISC-0 Integration
//!
//! The `zk-gradient-proof/` directory contains a complete RISC-0 project
//! for generating real ZK proofs. See its README for setup instructions.
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_sdk::zkproof::{GradientProver, ProverMode};
//!
//! let prover = GradientProver::for_federated_learning();
//! let gradient: Vec<f32> = vec![0.1; 1000];
//! let model_hash = [0u8; 32];
//!
//! let receipt = prover.prove_gradient_quality(
//!     &gradient,
//!     &model_hash,
//!     5,     // epochs
//!     0.01,  // learning rate
//!     "client-1",
//!     1,     // round
//! ).unwrap();
//!
//! assert!(receipt.is_valid());
//! ```

mod circuit;
mod gradient_proof;
pub mod proof_system;
mod risc0_integration;
#[cfg(any(feature = "simulation", feature = "risc0"))]
mod risc0_prover;
pub mod trust_proof_system;
pub mod trust_risc0;
mod types;

pub use circuit::GradientProofCircuit;
pub use gradient_proof::{
    compute_commitment, hash_gradient, verify_gradient_quality, GradientConstraints,
    GradientProofInput, GradientProofOutput, GradientQualityResult, Risc0ProofConfig,
};
#[cfg(any(feature = "simulation", feature = "risc0"))]
pub use risc0_prover::{BatchGradientProver, GradientProver, ProverError, SimulationProofMarker};

#[cfg(any(feature = "simulation", feature = "risc0"))]
pub use risc0_prover::GradientProofReceipt;
pub use types::{GradientProof, ProofMetadata, PublicInputs};

// Export prover mode only when features are enabled
#[cfg(any(feature = "simulation", feature = "risc0"))]
pub use risc0_prover::ProverMode;

// Export typed provers based on features
#[cfg(feature = "risc0")]
pub use risc0_prover::ProductionProver;

#[cfg(feature = "simulation")]
pub use risc0_prover::SimulationProver;

// RISC-0 integration types (always available)
pub use risc0_integration::{
    hash_gradient as risc0_hash_gradient, verify_gradient_constraints, GradientQualityInput,
    GradientQualityOutput, Risc0ProofReceipt, Risc0ProverConfig, Risc0ProverError,
};

// Real RISC-0 prover (only with feature flag)
#[cfg(feature = "risc0")]
pub use risc0_integration::Risc0GradientProver;

// Generic proof system abstraction
pub use proof_system::{
    is_simulation_proof, BackendConfig, BackendType, GenericReceipt, ProofOutput, ProofReceipt,
    ProofStatement, ProofStats, ProofSystem, ProofSystemBackend, ProofSystemBuilder,
    ProofSystemError, ProofWitness, SimulationBackend,
};

// K-Vector trust proof system (uses generic abstraction)
pub use trust_proof_system::{
    TrustProofSystem, TrustPublicData, TrustReceipt, TrustStatement, TrustWitness,
};

/// Prove gradient computation was done correctly
///
/// Convenience wrapper for common use case.
pub fn prove_gradient(
    client_id: &str,
    global_model: &[f32],
    gradient: &[f32],
    epochs: u32,
    lr: f32,
) -> GradientProof {
    let circuit = GradientProofCircuit::new(client_id);
    circuit.prove_gradient(global_model, gradient, epochs, lr)
}

/// Verify a gradient proof
pub fn verify_proof(proof: &GradientProof) -> bool {
    let circuit = GradientProofCircuit::new(&proof.public_inputs.client_id);
    circuit.verify_proof(proof)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prove_and_verify() {
        let global_model: Vec<f32> = vec![0.1; 1000];
        let gradient: Vec<f32> = vec![0.01; 1000];

        let proof = prove_gradient("node-1", &global_model, &gradient, 5, 0.01);

        assert!(verify_proof(&proof));
        assert_eq!(proof.public_inputs.epochs, 5);
    }

    #[test]
    fn test_invalid_proof_fails() {
        let global_model: Vec<f32> = vec![0.1; 1000];
        let gradient: Vec<f32> = vec![0.01; 1000];

        let mut proof = prove_gradient("node-1", &global_model, &gradient, 5, 0.01);

        // Tamper with proof
        proof.proof_bytes[0] ^= 0xFF;

        assert!(!verify_proof(&proof));
    }
}
