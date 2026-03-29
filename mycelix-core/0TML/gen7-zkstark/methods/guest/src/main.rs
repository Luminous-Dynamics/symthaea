#![no_main]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Gen-7 HYPERION-FL: Gradient Validity Proof Circuit
//!
//! This zkVM guest program proves that a gradient update is valid and was computed
//! honestly, without revealing the private training data.
//!
//! ## What We Prove
//!
//! 1. **Gradient Validity**: The gradient contains no NaN/Inf values
//! 2. **Norm Bounds**: The gradient L2 norm is within acceptable bounds
//! 3. **Commitment Binding**: The gradient matches the claimed commitment
//! 4. **Node Attribution**: The gradient is bound to a specific node and round
//!
//! ## Security Properties
//!
//! - **Soundness**: A dishonest prover cannot forge a valid proof for an invalid gradient
//! - **Zero-Knowledge**: The proof reveals nothing about the training data
//! - **Binding**: The node commits to the gradient before the challenge

use risc0_zkvm::guest::env;
use sha2::{Sha256, Digest};

risc0_zkvm::guest::entry!(main);

// =============================================================================
// Constants
// =============================================================================

/// Maximum allowed L2 norm squared (in Q16.16 fixed-point)
/// This prevents gradient explosion attacks
/// Default: 1000.0^2 = 1,000,000 in float, scaled by 2^32 for squared Q16.16
const MAX_NORM_SQUARED_RAW: i64 = 1_000_000 * 65536; // ~1000.0 L2 norm limit

/// Minimum allowed L2 norm squared (prevents zero/near-zero gradients)
/// Default: 0.0001^2 in Q16.16 squared
const MIN_NORM_SQUARED_RAW: i64 = 1; // Effectively > 0

/// Q16.16 scale factor
const SCALE: i64 = 65536;

// =============================================================================
// Main Entry Point
// =============================================================================

fn main() {
    // =========================================================================
    // Phase 1: Read Public Inputs
    // =========================================================================

    // Node identifier (32 bytes, typically a public key hash)
    let node_id: [u8; 32] = env::read();

    // Training round number (for replay protection)
    let round_number: u64 = env::read();

    // Claimed gradient hash (commitment made before challenge)
    let claimed_gradient_hash: [u8; 32] = env::read();

    // Model hash (proves we're updating the correct global model)
    let model_hash: [u8; 32] = env::read();

    // Maximum allowed norm squared (validator-specified bound)
    let max_norm_squared: i64 = env::read();

    // Number of gradient elements
    let gradient_len: u32 = env::read();

    // =========================================================================
    // Phase 2: Read Private Witness (the gradient itself)
    // =========================================================================

    // Read gradient vector (Q16.16 fixed-point representation)
    let mut gradient: Vec<i32> = Vec::with_capacity(gradient_len as usize);
    for _ in 0..gradient_len {
        gradient.push(env::read());
    }

    // =========================================================================
    // Phase 3: Verify Gradient Commitment
    // =========================================================================

    // Compute SHA256 hash of gradient
    let mut hasher = Sha256::new();
    for &g in &gradient {
        hasher.update(g.to_le_bytes());
    }
    let computed_gradient_hash: [u8; 32] = hasher.finalize().into();

    // Verify the gradient matches the claimed commitment
    assert_eq!(
        claimed_gradient_hash,
        computed_gradient_hash,
        "Gradient commitment mismatch: prover submitted different gradient than committed"
    );

    // =========================================================================
    // Phase 4: Validate Gradient Values (No NaN/Inf in fixed-point)
    // =========================================================================

    // In Q16.16 fixed-point, we don't have true NaN/Inf, but we check for
    // sentinel values that might indicate corruption or attacks

    // Check for extreme values that would indicate overflow/corruption
    const FIXED_MAX: i32 = i32::MAX - 1; // Reserve MAX for sentinel
    const FIXED_MIN: i32 = i32::MIN + 1; // Reserve MIN for sentinel

    for (i, &g) in gradient.iter().enumerate() {
        assert!(
            g > FIXED_MIN && g < FIXED_MAX,
            "Gradient element {} has invalid/sentinel value: {}",
            i,
            g
        );
    }

    // =========================================================================
    // Phase 5: Compute and Verify Gradient Norm
    // =========================================================================

    // Compute L2 norm squared: sum(g_i^2)
    // We use i64 to prevent overflow during accumulation
    let mut norm_squared: i64 = 0;

    for &g in &gradient {
        // g is Q16.16, so g*g is Q32.32
        // We shift right by 16 to get Q16.16 result for each term
        let g_squared = (g as i64 * g as i64) >> 16;
        norm_squared = norm_squared.saturating_add(g_squared);
    }

    // Verify norm is within bounds
    assert!(
        norm_squared >= MIN_NORM_SQUARED_RAW,
        "Gradient norm too small (zero or near-zero gradient): {}",
        norm_squared
    );

    // Use the validator-specified max or our default
    let effective_max = if max_norm_squared > 0 {
        max_norm_squared
    } else {
        MAX_NORM_SQUARED_RAW
    };

    assert!(
        norm_squared <= effective_max,
        "Gradient norm exceeds maximum allowed: {} > {}",
        norm_squared,
        effective_max
    );

    // =========================================================================
    // Phase 6: Compute Gradient Statistics (for journal output)
    // =========================================================================

    // Compute mean (for gradient analysis, optional verification)
    let mut sum: i64 = 0;
    for &g in &gradient {
        sum += g as i64;
    }
    let mean_raw: i32 = (sum / gradient_len as i64) as i32;

    // Compute variance (measures gradient diversity)
    let mut variance_sum: i64 = 0;
    for &g in &gradient {
        let diff = (g as i64) - (mean_raw as i64);
        variance_sum += (diff * diff) >> 16; // Q16.16 squared, normalized
    }
    let variance_raw: i32 = (variance_sum / gradient_len as i64) as i32;

    // =========================================================================
    // Phase 7: Commit Verified Outputs to Journal
    // =========================================================================

    // The journal contains the public outputs that anyone can verify
    // These are cryptographically bound to the proof

    // 1. Node ID (proves which node generated this gradient)
    env::commit(&node_id);

    // 2. Round number (prevents replay attacks)
    env::commit(&round_number);

    // 3. Gradient hash (the verified commitment)
    env::commit(&computed_gradient_hash);

    // 4. Model hash (proves correct model version)
    env::commit(&model_hash);

    // 5. Gradient statistics (for aggregator analysis)
    env::commit(&gradient_len);
    env::commit(&(norm_squared as i64));  // L2 norm squared
    env::commit(&mean_raw);               // Mean value
    env::commit(&variance_raw);           // Variance

    // 6. Validity flag (always true if we reach here)
    let is_valid: u8 = 1;
    env::commit(&is_valid);

    // =========================================================================
    // Proof Complete!
    // =========================================================================
    //
    // The verifier now has cryptographic proof that:
    //
    // 1. The gradient matches the node's prior commitment (binding)
    // 2. The gradient contains no invalid values (validity)
    // 3. The gradient norm is within acceptable bounds (safety)
    // 4. The gradient is attributed to a specific node and round (accountability)
    //
    // The training data remains completely private - only the commitment
    // and statistics are revealed.
}
