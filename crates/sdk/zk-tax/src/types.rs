// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core types for ZK tax proofs.

use serde::{Deserialize, Serialize};

/// Tax year type alias for clarity.
pub type TaxYear = u32;

/// A cryptographic commitment to tax bracket parameters.
///
/// This commitment binds the proof to specific bracket bounds
/// without revealing the actual income.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BracketCommitment([u8; 32]);

impl BracketCommitment {
    /// Create a new commitment from raw bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get the raw bytes of this commitment.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Get a copy of the raw bytes.
    pub fn to_bytes(&self) -> [u8; 32] {
        self.0
    }

    /// Encode as hex string.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Decode from hex string.
    pub fn from_hex(s: &str) -> Result<Self, hex::FromHexError> {
        let bytes = hex::decode(s)?;
        if bytes.len() != 32 {
            return Err(hex::FromHexError::InvalidStringLength);
        }
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Ok(Self(arr))
    }
}

impl std::fmt::Display for BracketCommitment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "0x{}", &self.to_hex()[..16])
    }
}

/// Private witness for ZK proof generation.
///
/// This is the input to the zkVM - it contains the private income
/// and the claimed bracket bounds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaxBracketWitness {
    /// The user's actual gross income (PRIVATE - never revealed)
    pub gross_income: u64,
    /// Lower bound of the claimed tax bracket
    pub bracket_lower: u64,
    /// Upper bound of the claimed tax bracket (exclusive)
    pub bracket_upper: u64,
    /// Tax year being proven
    pub tax_year: TaxYear,
    /// Bracket index (for verification)
    pub bracket_index: u8,
    /// Marginal rate in basis points (e.g., 2200 = 22%)
    pub rate_bps: u16,
}

/// Public output from ZK proof.
///
/// This is what the verifier sees - proof that SOME valid income exists
/// in the claimed bracket, without knowing the actual value.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct TaxBracketOutput {
    /// Tax year that was proven
    pub tax_year: TaxYear,
    /// Bracket index that was proven (0-6 for US)
    pub bracket_index: u8,
    /// Marginal rate in basis points
    pub rate_bps: u16,
    /// Whether the proof constraints are satisfied
    pub valid: bool,
    /// Commitment to bracket parameters
    pub commitment: BracketCommitment,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commitment_hex_roundtrip() {
        let bytes = [0x42u8; 32];
        let commitment = BracketCommitment::from_bytes(bytes);
        let hex = commitment.to_hex();
        let decoded = BracketCommitment::from_hex(&hex).unwrap();
        assert_eq!(commitment, decoded);
    }

    #[test]
    fn test_commitment_display() {
        let bytes = [0xAB; 32];
        let commitment = BracketCommitment::from_bytes(bytes);
        let display = format!("{}", commitment);
        assert!(display.starts_with("0x"));
        assert_eq!(display.len(), 18); // "0x" + 16 hex chars
    }
}
