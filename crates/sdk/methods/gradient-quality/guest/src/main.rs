// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gradient Quality ZK Guest Program
//!
//! This code runs INSIDE the RISC-0 zkVM. It:
//! 1. Reads private witness (gradient values, constraints)
//! 2. Verifies the gradient meets quality constraints
//! 3. Commits public output (validity, hash commitment)
//!
//! The proof guarantees: "I know a gradient that meets quality constraints"
//! WITHOUT revealing the actual gradient values.
//!
//! # Constraints Verified
//!
//! - Gradient L2 norm is within [min_norm, max_norm]
//! - Gradient is finite (no NaN/Inf values)
//! - Gradient dimensions are non-empty
//!
//! # Privacy Properties
//!
//! | Information          | Prover Knows | Verifier Sees |
//! |---------------------|--------------|---------------|
//! | Gradient values     | ✅ Yes       | ❌ Never      |
//! | Gradient hash       | ✅ Yes       | ✅ Yes        |
//! | Norm validity       | ✅ Yes       | ✅ Yes        |
//! | Model hash          | ✅ Yes       | ✅ Yes        |
//! | Training parameters | ✅ Yes       | ✅ Yes        |

#![no_main]

use risc0_zkvm::guest::env;

// These types must match the host-side definitions
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GradientQualityInput {
    pub gradients: Vec<f32>,
    pub global_model_hash: [u8; 32],
    pub epochs: u32,
    pub learning_rate: f32,
    pub client_id: String,
    pub round: u32,
    pub min_norm: f32,
    pub max_norm: f32,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct GradientQualityOutput {
    pub gradient_hash: [u8; 32],
    pub global_model_hash: [u8; 32],
    pub epochs: u32,
    pub learning_rate: f32,
    pub norm_valid: bool,
    pub client_id: String,
    pub round: u32,
    pub commitment: [u8; 32],
}

risc0_zkvm::guest::entry!(main);

fn main() {
    // Read the private witness from the host
    let input: GradientQualityInput = env::read();

    // === CONSTRAINT 1: Non-empty gradient ===
    let non_empty = !input.gradients.is_empty();

    // === CONSTRAINT 2: Compute L2 norm ===
    let norm_squared: f32 = input.gradients.iter().map(|g| g * g).sum();
    let norm = norm_squared.sqrt();

    // === CONSTRAINT 3: Norm within bounds ===
    let norm_in_bounds = norm >= input.min_norm && norm <= input.max_norm;

    // === CONSTRAINT 4: All values finite ===
    let all_finite = input.gradients.iter().all(|g| g.is_finite()) && norm.is_finite();

    // === Combined validity ===
    let norm_valid = non_empty && norm_in_bounds && all_finite;

    // === Compute gradient hash (commitment) ===
    // Using a simple hash for zkVM efficiency
    // In practice, use SHA3-256 or similar
    let gradient_hash = compute_gradient_hash(&input.gradients);

    // === Compute overall commitment ===
    let mut output = GradientQualityOutput {
        gradient_hash,
        global_model_hash: input.global_model_hash,
        epochs: input.epochs,
        learning_rate: input.learning_rate,
        norm_valid,
        client_id: input.client_id,
        round: input.round,
        commitment: [0u8; 32],
    };

    // Compute commitment
    output.commitment = compute_commitment(&output);

    // Commit the output to the journal
    // This is what the verifier will see
    env::commit(&output);
}

/// Compute hash of gradient values
/// Using FNV-1a for efficiency in zkVM (real impl would use SHA3)
fn compute_gradient_hash(gradients: &[f32]) -> [u8; 32] {
    let mut hash = [0u8; 32];
    let mut state: u64 = 0xcbf29ce484222325; // FNV offset basis
    let prime: u64 = 0x100000001b3;

    for g in gradients {
        for byte in g.to_le_bytes() {
            state ^= byte as u64;
            state = state.wrapping_mul(prime);
        }
    }

    // Spread the state across the hash
    for i in 0..4 {
        let segment = (state >> (i * 16)) as u64;
        for j in 0..8 {
            hash[i * 8 + j] = ((segment >> (j * 8)) & 0xFF) as u8;
        }
    }

    hash
}

/// Compute commitment from output fields
fn compute_commitment(output: &GradientQualityOutput) -> [u8; 32] {
    let mut hash = [0u8; 32];
    let mut state: u64 = 0xcbf29ce484222325;
    let prime: u64 = 0x100000001b3;

    // Hash all public fields
    for byte in &output.gradient_hash {
        state ^= *byte as u64;
        state = state.wrapping_mul(prime);
    }
    for byte in &output.global_model_hash {
        state ^= *byte as u64;
        state = state.wrapping_mul(prime);
    }
    for byte in output.epochs.to_le_bytes() {
        state ^= byte as u64;
        state = state.wrapping_mul(prime);
    }
    for byte in output.learning_rate.to_le_bytes() {
        state ^= byte as u64;
        state = state.wrapping_mul(prime);
    }
    state ^= output.norm_valid as u64;
    state = state.wrapping_mul(prime);
    for byte in output.client_id.as_bytes() {
        state ^= *byte as u64;
        state = state.wrapping_mul(prime);
    }
    for byte in output.round.to_le_bytes() {
        state ^= byte as u64;
        state = state.wrapping_mul(prime);
    }

    // Spread the state
    for i in 0..4 {
        let segment = (state >> (i * 16)) as u64;
        for j in 0..8 {
            hash[i * 8 + j] = ((segment >> (j * 8)) & 0xFF) as u8;
        }
    }

    hash
}
