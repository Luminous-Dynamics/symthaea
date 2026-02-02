//! ZK Tax Bracket Guest Code
//!
//! This runs INSIDE the Risc0 zkVM and verifies:
//! 1. The income falls within the claimed bracket bounds
//! 2. Outputs a commitment to the bracket parameters
//!
//! The proof guarantees: "I know an income in bracket X"
//! WITHOUT revealing what that income actually is.

#![no_main]

use risc0_zkvm::guest::env;
use serde::{Deserialize, Serialize};

risc0_zkvm::guest::entry!(main);

/// Input to the zkVM (private witness)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaxWitnessInput {
    /// The user's actual gross income (PRIVATE)
    pub gross_income: u64,
    /// Lower bound of the claimed bracket
    pub bracket_lower: u64,
    /// Upper bound of the claimed bracket (exclusive)
    pub bracket_upper: u64,
    /// Tax year
    pub tax_year: u32,
    /// Bracket index
    pub bracket_index: u8,
    /// Marginal rate in basis points
    pub rate_bps: u16,
}

/// Output from the zkVM (public)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaxProofOutput {
    /// Tax year proven
    pub tax_year: u32,
    /// Bracket index proven
    pub bracket_index: u8,
    /// Marginal rate
    pub rate_bps: u16,
    /// Whether constraints are satisfied
    pub valid: bool,
    /// Commitment to bracket parameters (SHA3-256 truncated)
    pub commitment: [u8; 32],
    /// Bracket bounds (public for verification)
    pub bracket_lower: u64,
    pub bracket_upper: u64,
}

fn main() {
    // Read the private witness
    let input: TaxWitnessInput = env::read();

    // CRITICAL CONSTRAINT: Verify income is within bracket
    let in_range = input.gross_income >= input.bracket_lower
        && input.gross_income < input.bracket_upper;

    // Compute commitment to bracket parameters
    // Using a simple but deterministic hash
    let commitment = compute_commitment(
        input.bracket_lower,
        input.bracket_upper,
        input.tax_year,
    );

    // Create public output
    let output = TaxProofOutput {
        tax_year: input.tax_year,
        bracket_index: input.bracket_index,
        rate_bps: input.rate_bps,
        valid: in_range,
        commitment,
        bracket_lower: input.bracket_lower,
        bracket_upper: input.bracket_upper,
    };

    // Commit output to journal (this is what verifier sees)
    env::commit(&output);
}

/// Compute deterministic commitment to bracket parameters
fn compute_commitment(lower: u64, upper: u64, tax_year: u32) -> [u8; 32] {
    // FNV-1a hash expanded to 32 bytes
    // Note: In production, use proper SHA3-256
    let mut hash: u64 = 0xcbf29ce484222325;
    let prime: u64 = 0x100000001b3;

    // Hash lower bound
    for byte in lower.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(prime);
    }

    // Hash upper bound
    for byte in upper.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(prime);
    }

    // Hash tax year
    for byte in tax_year.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(prime);
    }

    // Expand to 32 bytes by running additional rounds
    let mut commitment = [0u8; 32];
    let mut h = hash;
    for chunk in commitment.chunks_mut(8) {
        chunk.copy_from_slice(&h.to_le_bytes());
        h = h.wrapping_mul(prime);
    }

    commitment
}
