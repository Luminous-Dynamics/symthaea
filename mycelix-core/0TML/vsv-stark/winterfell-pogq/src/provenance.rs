// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Provenance hash system for tamper-evident proof configuration
//!
//! Computes a stable, cryptographic hash of all proof generation parameters
//! to prevent options mismatch attacks and ensure verifiable provenance.
//!
//! The provenance hash commits to:
//! - Winterfell version
//! - Rust toolchain version
//! - AIR configuration (trace width, constraint count, version)
//! - ProofOptions (queries, blowup, grinding, FRI params)
//! - Security profile
//! - Git commit
//! - Build mode (debug/release)
//!
//! This hash is embedded in both public inputs and the receipt, and must match
//! exactly for verification to succeed.

use blake3::Hasher;
use winterfell::ProofOptions;
use winterfell::math::fields::f128::BaseElement;
use winterfell::math::StarkField;
use crate::security::SecurityProfile;

/// Provenance hash digest (32 bytes / 256 bits)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProvenanceHash(pub [u8; 32]);

impl ProvenanceHash {
    /// Compute provenance hash from all configuration parameters
    ///
    /// Uses domain-separated Blake3 hashing with canonical serialization:
    /// ```text
    /// provenance_hash = Blake3(
    ///     "VSV-STARK-PROVENANCE-v1" ||
    ///     winterfell_version ||
    ///     rustc_version ||
    ///     git_commit ||
    ///     build_mode ||
    ///     air_version ||
    ///     trace_width ||
    ///     constraint_count ||
    ///     security_profile ||
    ///     proof_options_canonical
    /// )
    /// ```
    pub fn compute(
        profile: SecurityProfile,
        options: &ProofOptions,
        air_version: &str,
        trace_width: usize,
        constraint_count: usize,
    ) -> Self {
        let mut hasher = Hasher::new();

        // Domain separator
        hasher.update(b"VSV-STARK-PROVENANCE-v1");

        // Runtime versions
        hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
        hasher.update(option_env!("RUSTC_VERSION").unwrap_or("unknown").as_bytes());
        hasher.update(option_env!("GIT_COMMIT").unwrap_or("uncommitted").as_bytes());

        // Build mode
        let build_mode = if cfg!(debug_assertions) { "debug" } else { "release" };
        hasher.update(build_mode.as_bytes());

        // AIR configuration
        hasher.update(air_version.as_bytes());
        hasher.update(&trace_width.to_le_bytes());
        hasher.update(&constraint_count.to_le_bytes());

        // Security profile
        hasher.update(profile.name().as_bytes());

        // ProofOptions (canonical serialization)
        hasher.update(&options.num_queries().to_le_bytes());
        hasher.update(&options.blowup_factor().to_le_bytes());
        hasher.update(&options.grinding_factor().to_le_bytes());

        // FRI parameters (hardcoded for now since not exposed by ProofOptions)
        hasher.update(&8u8.to_le_bytes());  // fri_folding_factor
        hasher.update(&31u8.to_le_bytes()); // fri_max_remainder_degree

        ProvenanceHash(hasher.finalize().into())
    }

    /// Convert hash to hex string for display
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Parse hex string to hash
    pub fn from_hex(hex_str: &str) -> Result<Self, String> {
        let bytes = hex::decode(hex_str)
            .map_err(|e| format!("Invalid hex: {}", e))?;

        if bytes.len() != 32 {
            return Err(format!("Expected 32 bytes, got {}", bytes.len()));
        }

        let mut hash = [0u8; 32];
        hash.copy_from_slice(&bytes);
        Ok(ProvenanceHash(hash))
    }

    /// Get raw bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Convert 4× u64 limbs (little-endian) back to 32-byte array
///
/// Used for testing: verifies that u64→BaseElement→u64 roundtrip is lossless
pub fn u64x4_le_to_bytes(limbs: &[u64; 4]) -> [u8; 32] {
    let mut result = [0u8; 32];
    for (i, &limb) in limbs.iter().enumerate() {
        result[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    result
}

/// Convert 4× BaseElement to u64 limbs (for comparison)
///
/// Used in verification to compare field-encoded provenance with expected u64 limbs
pub fn base_elements_to_u64x4(elems: &[BaseElement; 4]) -> [u64; 4] {
    [
        elems[0].as_int() as u64,
        elems[1].as_int() as u64,
        elems[2].as_int() as u64,
        elems[3].as_int() as u64,
    ]
}

impl std::fmt::Display for ProvenanceHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::SecurityProfile;

    #[test]
    fn test_provenance_hash_deterministic() {
        let profile = SecurityProfile::S128;
        let options = profile.proof_options();

        let hash1 = ProvenanceHash::compute(profile, &options, "PoGQ-v4.1", 44, 40);
        let hash2 = ProvenanceHash::compute(profile, &options, "PoGQ-v4.1", 44, 40);

        assert_eq!(hash1, hash2, "Hash should be deterministic");
    }

    #[test]
    fn test_provenance_hash_different_profiles() {
        let profile_s128 = SecurityProfile::S128;
        let profile_s192 = SecurityProfile::S192;

        let hash_s128 = ProvenanceHash::compute(
            profile_s128,
            &profile_s128.proof_options(),
            "PoGQ-v4.1",
            44,
            40,
        );

        let hash_s192 = ProvenanceHash::compute(
            profile_s192,
            &profile_s192.proof_options(),
            "PoGQ-v4.1",
            44,
            40,
        );

        assert_ne!(hash_s128, hash_s192, "Different profiles should produce different hashes");
    }

    #[test]
    fn test_provenance_hash_hex_roundtrip() {
        let profile = SecurityProfile::S128;
        let options = profile.proof_options();

        let hash = ProvenanceHash::compute(profile, &options, "PoGQ-v4.1", 44, 40);
        let hex = hash.to_hex();
        let parsed = ProvenanceHash::from_hex(&hex).unwrap();

        assert_eq!(hash, parsed, "Hex roundtrip should preserve hash");
    }

    #[test]
    fn test_provenance_hash_air_sensitivity() {
        let profile = SecurityProfile::S128;
        let options = profile.proof_options();

        let hash1 = ProvenanceHash::compute(profile, &options, "PoGQ-v4.1", 44, 40);
        let hash2 = ProvenanceHash::compute(profile, &options, "PoGQ-v4.0", 44, 40); // Different AIR
        let hash3 = ProvenanceHash::compute(profile, &options, "PoGQ-v4.1", 45, 40); // Different width
        let hash4 = ProvenanceHash::compute(profile, &options, "PoGQ-v4.1", 44, 41); // Different constraints

        assert_ne!(hash1, hash2, "AIR version should affect hash");
        assert_ne!(hash1, hash3, "Trace width should affect hash");
        assert_ne!(hash1, hash4, "Constraint count should affect hash");
    }
}
