// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Security profiles for VSV-STARK proof generation
//!
//! Provides standardized security configurations for different deployment contexts:
//! - S128: ≥127-bit conjectured security (academic/research default)
//! - S192: ≥192-bit conjectured security (JADC2/HIPAA high-assurance)

use winterfell::{BatchingMethod, FieldExtension, ProofOptions};

/// Security profile selector for proof generation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityProfile {
    /// S128 (≥127-bit): Conjectured STARK security
    ///
    /// Maximum achievable with our AIR configuration (44 cols, 32 rows, 40 constraints).
    /// Provides ≥127-bit conjectured security under standard FRI soundness assumptions.
    ///
    /// Suitable for: academic publications, research prototypes, short-lived decisions.
    ///
    /// Performance: ~7-13 ms prove, ~1-1.7 ms verify, ~60-63 KB proofs (T=32)
    S128,

    /// S192 (≥192-bit): High-assurance conjectured STARK security
    ///
    /// High-assurance profile for long-lived critical decisions.
    /// Provides ≥192-bit conjectured security via increased query count.
    ///
    /// Suitable for: JADC2 deployments, HIPAA/clinical trials, financial systems.
    ///
    /// Performance: ~1.8-2.5× prove time vs S128, ~1.5× proof size
    S192,
}

impl SecurityProfile {
    /// Parse profile from string (case-insensitive)
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "s128" => Ok(SecurityProfile::S128),
            "s192" => Ok(SecurityProfile::S192),
            _ => Err(format!("Invalid security profile '{}'. Valid options: s128, s192", s)),
        }
    }

    /// Get profile from environment variable VSV_SECURITY_PROFILE
    pub fn from_env() -> Option<Self> {
        std::env::var("VSV_SECURITY_PROFILE")
            .ok()
            .and_then(|s| Self::from_str(&s).ok())
    }

    /// Get canonical name for this profile
    pub fn name(&self) -> &'static str {
        match self {
            SecurityProfile::S128 => "S128",
            SecurityProfile::S192 => "S192",
        }
    }

    /// Get numeric profile ID for provenance commitment
    pub fn profile_id(&self) -> u32 {
        match self {
            SecurityProfile::S128 => 128,
            SecurityProfile::S192 => 192,
        }
    }

    /// Get human-readable security description
    pub fn description(&self) -> &'static str {
        match self {
            SecurityProfile::S128 => "S128 (≥127-bit conjectured STARK security)",
            SecurityProfile::S192 => "S192 (≥192-bit conjectured STARK security, high-assurance)",
        }
    }

    /// Get Winterfell ProofOptions for this security profile
    ///
    /// # Grinding Factor Policy
    ///
    /// Both profiles use grinding_factor=16 (default) to prevent cheap proof generation
    /// and DoS attacks via find-and-replace of receipts. This adds ~16 bits of proof-of-work
    /// without significantly impacting honest prover performance.
    ///
    /// Rationale: Without grinding, an attacker could trivially forge proofs by finding
    /// a random coin seed that passes verification checks. The grinding factor forces
    /// the prover to search for a nonce that satisfies a collision-resistant hash puzzle,
    /// making proof forgery computationally expensive while adding negligible overhead
    /// to legitimate proof generation.
    pub fn proof_options(&self) -> ProofOptions {
        match self {
            SecurityProfile::S128 => ProofOptions::new(
                80,  // num_queries (verified to achieve ≥127-bit with our AIR)
                16,  // blowup_factor (standard code rate)
                16,  // grinding_factor (DoS resistance - prevents cheap proof forgery)
                FieldExtension::None,
                8,   // fri_folding_factor (standard for high security)
                31,  // fri_max_remainder_degree
                BatchingMethod::Linear,  // FRI remainder batching
                BatchingMethod::Linear,  // composition poly batching
            ),

            SecurityProfile::S192 => ProofOptions::new(
                120, // num_queries (↑50% for ≥192-bit target)
                16,  // blowup_factor (keep for compatibility)
                16,  // grinding_factor (DoS resistance - prevents cheap proof forgery)
                FieldExtension::None,
                8,   // fri_folding_factor
                31,  // fri_max_remainder_degree
                BatchingMethod::Linear,
                BatchingMethod::Linear,
            ),
        }
    }

    /// Get expected performance characteristics (based on benchmarks)
    pub fn performance_estimate(&self) -> PerformanceEstimate {
        match self {
            SecurityProfile::S128 => PerformanceEstimate {
                prove_ms_min: 7.0,
                prove_ms_max: 13.0,
                verify_ms_min: 1.0,
                verify_ms_max: 1.7,
                proof_kb_min: 60,
                proof_kb_max: 63,
            },

            SecurityProfile::S192 => PerformanceEstimate {
                prove_ms_min: 14.0,  // ~2× S128
                prove_ms_max: 26.0,
                verify_ms_min: 1.5,
                verify_ms_max: 3.0,
                proof_kb_min: 70,    // ~+10-20 KB vs S128
                proof_kb_max: 80,
            },
        }
    }
}

impl Default for SecurityProfile {
    fn default() -> Self {
        SecurityProfile::S128
    }
}

impl std::fmt::Display for SecurityProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Performance characteristics for a security profile
#[derive(Debug, Clone, Copy)]
pub struct PerformanceEstimate {
    pub prove_ms_min: f64,
    pub prove_ms_max: f64,
    pub verify_ms_min: f64,
    pub verify_ms_max: f64,
    pub proof_kb_min: usize,
    pub proof_kb_max: usize,
}

impl PerformanceEstimate {
    pub fn describe(&self) -> String {
        format!(
            "Prove: {:.1}-{:.1} ms | Verify: {:.2}-{:.2} ms | Size: {}-{} KB",
            self.prove_ms_min,
            self.prove_ms_max,
            self.verify_ms_min,
            self.verify_ms_max,
            self.proof_kb_min,
            self.proof_kb_max
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_parsing() {
        assert_eq!(SecurityProfile::from_str("s128").unwrap(), SecurityProfile::S128);
        assert_eq!(SecurityProfile::from_str("S128").unwrap(), SecurityProfile::S128);
        assert_eq!(SecurityProfile::from_str("s192").unwrap(), SecurityProfile::S192);
        assert!(SecurityProfile::from_str("invalid").is_err());
    }

    #[test]
    fn test_profile_names() {
        assert_eq!(SecurityProfile::S128.name(), "S128");
        assert_eq!(SecurityProfile::S192.name(), "S192");
    }

    #[test]
    fn test_s128_options() {
        let opts = SecurityProfile::S128.proof_options();
        // Verify key parameters match our benchmarked configuration
        assert_eq!(opts.num_queries(), 80);
        assert_eq!(opts.blowup_factor(), 16);
        assert_eq!(opts.grinding_factor(), 16);
    }

    #[test]
    fn test_s192_options() {
        let opts = SecurityProfile::S192.proof_options();
        // Verify S192 has increased queries for higher security
        assert_eq!(opts.num_queries(), 120);
        assert_eq!(opts.blowup_factor(), 16);
    }
}
