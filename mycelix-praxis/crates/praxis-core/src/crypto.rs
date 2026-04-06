// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cryptographic utilities for EduNet

use blake3::Hash;
use serde::{Deserialize, Serialize};

/// Compute BLAKE3 hash of data
pub fn hash_data(data: &[u8]) -> Hash {
    blake3::hash(data)
}

/// Compute BLAKE3 hash and return as hex string
pub fn hash_to_string(data: &[u8]) -> String {
    hash_data(data).to_hex().to_string()
}

/// Commitment to a value (hash-based)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Commitment {
    pub hash: String,
    pub timestamp: i64,
}

impl Commitment {
    /// Create a new commitment to data with explicit timestamp
    /// For WASM/Holochain, use sys_time(); for native, use chrono::Utc::now().timestamp()
    pub fn new(data: &[u8], timestamp: i64) -> Self {
        Self {
            hash: hash_to_string(data),
            timestamp,
        }
    }

    /// Create a commitment with current timestamp (native builds only)
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_now(data: &[u8]) -> Self {
        Self::new(data, chrono::Utc::now().timestamp())
    }

    /// Verify that data matches this commitment
    pub fn verify(&self, data: &[u8]) -> bool {
        hash_to_string(data) == self.hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_data() {
        let data = b"hello world";
        let hash = hash_data(data);
        assert_eq!(hash.to_hex().len(), 64); // BLAKE3 produces 32 bytes = 64 hex chars
    }

    #[test]
    fn test_commitment() {
        let data = b"secret value";
        let commitment = Commitment::new_now(data);

        // Should verify with correct data
        assert!(commitment.verify(data));

        // Should not verify with wrong data
        assert!(!commitment.verify(b"wrong value"));
    }
}
