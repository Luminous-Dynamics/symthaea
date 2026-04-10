// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Centralized configuration for Mycelix ZK proof systems
//!
//! This crate provides a single source of truth for security levels,
//! proof parameters, and configuration across all ZK proof implementations.
//!
//! # Design Philosophy
//!
//! Rather than having security parameters scattered across multiple crates,
//! this crate centralizes them to ensure consistency and make security
//! audits easier.
//!
//! # Example
//!
//! ```
//! use proofs_config::{SecurityLevel, ProofConfig};
//!
//! // Get production-safe configuration
//! let config = ProofConfig::production();
//! assert!(config.security_level.is_production_safe());
//!
//! // Get configuration for specific use case
//! let config = ProofConfig::for_use_case(proofs_config::UseCase::ConstitutionalVote);
//! assert_eq!(config.security_level, SecurityLevel::High);
//! ```

use serde::{Deserialize, Serialize};

/// Security level for ZK proofs
///
/// These levels correspond to estimated security bits based on
/// Winterfell's STARK implementation parameters.
///
/// # Security Estimates (Audited)
///
/// - `Fast`: ~40-bit security - **TESTING ONLY, NOT FOR PRODUCTION**
/// - `Optimized`: ~84-bit security - Internal/low-stakes operations
/// - `Standard`: ~96-bit security - Production default, general use
/// - `High`: ~264-bit security - Critical operations (constitutional votes, large treasury)
///
/// # Production Requirements
///
/// Production systems MUST use `Standard` or `High` security level.
/// The `Fast` level exists only for testing and development.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum SecurityLevel {
    /// ~40-bit security - TESTING ONLY
    ///
    /// Uses minimal parameters for fast iteration during development.
    /// **NEVER use in production - provides inadequate security.**
    Fast = 0,

    /// ~84-bit security - Internal/low-stakes
    ///
    /// Suitable for internal operations where the cost of attack
    /// exceeds the value at stake. Not recommended for user-facing
    /// features or financial operations.
    Optimized = 1,

    /// ~96-bit security - Production default
    ///
    /// The standard security level for production systems.
    /// Provides adequate security for most governance and
    /// identity operations.
    Standard = 2,

    /// ~264-bit security - Critical operations
    ///
    /// Maximum security for critical operations:
    /// - Constitutional amendments
    /// - Large treasury movements
    /// - Identity root key operations
    High = 3,
}

impl Default for SecurityLevel {
    fn default() -> Self {
        SecurityLevel::Standard
    }
}

impl SecurityLevel {
    /// Estimated security bits for this level
    ///
    /// Based on Winterfell STARK parameters:
    /// - Hash function: BLAKE3
    /// - Field: 64-bit prime field
    /// - FRI parameters vary by level
    pub const fn estimated_security_bits(&self) -> u32 {
        match self {
            SecurityLevel::Fast => 40,
            SecurityLevel::Optimized => 84,
            SecurityLevel::Standard => 96,
            SecurityLevel::High => 264,
        }
    }

    /// Check if this level meets a minimum security requirement
    pub const fn meets_minimum(&self, min_bits: u32) -> bool {
        self.estimated_security_bits() >= min_bits
    }

    /// Check if this level is safe for production use
    ///
    /// Returns `true` for `Standard` and `High` levels only.
    pub const fn is_production_safe(&self) -> bool {
        matches!(self, SecurityLevel::Standard | SecurityLevel::High)
    }

    /// Check if this level is testing-only
    pub const fn is_testing_only(&self) -> bool {
        matches!(self, SecurityLevel::Fast)
    }

    /// Get the minimum level that provides at least `bits` security
    pub const fn minimum_level_for_bits(bits: u32) -> Option<Self> {
        if bits <= 40 {
            Some(SecurityLevel::Fast)
        } else if bits <= 84 {
            Some(SecurityLevel::Optimized)
        } else if bits <= 96 {
            Some(SecurityLevel::Standard)
        } else if bits <= 264 {
            Some(SecurityLevel::High)
        } else {
            None
        }
    }

    /// Get the recommended level for production
    pub const fn production_default() -> Self {
        SecurityLevel::Standard
    }

    /// Get the level for maximum security
    pub const fn maximum() -> Self {
        SecurityLevel::High
    }

    /// Human-readable description of the security level
    pub const fn description(&self) -> &'static str {
        match self {
            SecurityLevel::Fast => "Testing only (~40-bit)",
            SecurityLevel::Optimized => "Internal/low-stakes (~84-bit)",
            SecurityLevel::Standard => "Production default (~96-bit)",
            SecurityLevel::High => "Critical operations (~264-bit)",
        }
    }
}

/// Use cases for ZK proofs in the Mycelix ecosystem
///
/// Each use case maps to an appropriate security level based on
/// the sensitivity of the operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UseCase {
    /// Development and testing
    Testing,

    /// K-Vector trust metric proofs (standard operations)
    KVectorProof,

    /// Standard governance voting
    StandardVote,

    /// Treasury-related proposals
    TreasuryVote,

    /// Constitutional amendments
    ConstitutionalVote,

    /// Model governance decisions
    ModelGovernance,

    /// Emergency proposals
    EmergencyProposal,

    /// Identity verification
    IdentityVerification,

    /// Federated learning contribution proofs
    FederatedLearning,

    /// Cross-chain bridge operations
    CrossChainBridge,
}

impl UseCase {
    /// Get the recommended security level for this use case
    pub const fn recommended_security_level(&self) -> SecurityLevel {
        match self {
            UseCase::Testing => SecurityLevel::Fast,
            UseCase::KVectorProof => SecurityLevel::Standard,
            UseCase::StandardVote => SecurityLevel::Standard,
            UseCase::TreasuryVote => SecurityLevel::High,
            UseCase::ConstitutionalVote => SecurityLevel::High,
            UseCase::ModelGovernance => SecurityLevel::Standard,
            UseCase::EmergencyProposal => SecurityLevel::Standard,
            UseCase::IdentityVerification => SecurityLevel::Standard,
            UseCase::FederatedLearning => SecurityLevel::Standard,
            UseCase::CrossChainBridge => SecurityLevel::High,
        }
    }

    /// Get the minimum acceptable security level for this use case
    pub const fn minimum_security_level(&self) -> SecurityLevel {
        match self {
            UseCase::Testing => SecurityLevel::Fast,
            UseCase::KVectorProof => SecurityLevel::Optimized,
            UseCase::StandardVote => SecurityLevel::Standard,
            UseCase::TreasuryVote => SecurityLevel::Standard,
            UseCase::ConstitutionalVote => SecurityLevel::High,
            UseCase::ModelGovernance => SecurityLevel::Optimized,
            UseCase::EmergencyProposal => SecurityLevel::Standard,
            UseCase::IdentityVerification => SecurityLevel::Standard,
            UseCase::FederatedLearning => SecurityLevel::Optimized,
            UseCase::CrossChainBridge => SecurityLevel::Standard,
        }
    }
}

/// Configuration for ZK proof generation and verification
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofConfig {
    /// Security level for the proof
    pub security_level: SecurityLevel,

    /// Whether to enforce strict security checks
    pub strict_mode: bool,

    /// Maximum proof size in bytes (0 = unlimited)
    pub max_proof_size: usize,

    /// Proof validity duration in seconds (0 = no expiry)
    pub validity_duration_secs: u64,
}

impl Default for ProofConfig {
    fn default() -> Self {
        Self::production()
    }
}

impl ProofConfig {
    /// Create a new configuration with the specified security level
    pub const fn new(security_level: SecurityLevel) -> Self {
        Self {
            security_level,
            strict_mode: true,
            max_proof_size: 0,
            validity_duration_secs: 0,
        }
    }

    /// Production-safe configuration with Standard security
    pub const fn production() -> Self {
        Self {
            security_level: SecurityLevel::Standard,
            strict_mode: true,
            max_proof_size: 1024 * 1024,  // 1MB max
            validity_duration_secs: 3600, // 1 hour
        }
    }

    /// High-security configuration for critical operations
    pub const fn high_security() -> Self {
        Self {
            security_level: SecurityLevel::High,
            strict_mode: true,
            max_proof_size: 2 * 1024 * 1024, // 2MB max (larger proofs)
            validity_duration_secs: 1800,    // 30 minutes (shorter validity)
        }
    }

    /// Testing configuration - NOT FOR PRODUCTION
    pub const fn testing() -> Self {
        Self {
            security_level: SecurityLevel::Fast,
            strict_mode: false,
            max_proof_size: 0,
            validity_duration_secs: 0,
        }
    }

    /// Get configuration for a specific use case
    pub const fn for_use_case(use_case: UseCase) -> Self {
        let security_level = use_case.recommended_security_level();
        match use_case {
            UseCase::Testing => Self::testing(),
            UseCase::ConstitutionalVote | UseCase::TreasuryVote | UseCase::CrossChainBridge => {
                Self::high_security()
            }
            _ => Self {
                security_level,
                strict_mode: true,
                max_proof_size: 1024 * 1024,
                validity_duration_secs: 3600,
            },
        }
    }

    /// Check if this configuration is production-safe
    pub const fn is_production_safe(&self) -> bool {
        self.security_level.is_production_safe() && self.strict_mode
    }

    /// Builder method to set security level
    pub const fn with_security_level(mut self, level: SecurityLevel) -> Self {
        self.security_level = level;
        self
    }

    /// Builder method to set strict mode
    pub const fn with_strict_mode(mut self, strict: bool) -> Self {
        self.strict_mode = strict;
        self
    }

    /// Builder method to set max proof size
    pub const fn with_max_proof_size(mut self, size: usize) -> Self {
        self.max_proof_size = size;
        self
    }

    /// Builder method to set validity duration
    pub const fn with_validity_duration(mut self, secs: u64) -> Self {
        self.validity_duration_secs = secs;
        self
    }
}

/// Minimum security bits required for production
pub const PRODUCTION_MIN_SECURITY_BITS: u32 = 96;

/// K-Vector specific constants
pub mod kvector {
    /// Number of K-Vector components
    pub const NUM_COMPONENTS: usize = 8;

    /// Scale factor for fixed-point representation (4 decimal places)
    pub const SCALE_FACTOR: u64 = 10_000;

    /// Maximum scaled value (1.0 * SCALE_FACTOR)
    pub const MAX_SCALED_VALUE: u64 = 10_000;

    /// K-Vector component names
    pub const COMPONENT_NAMES: [&str; NUM_COMPONENTS] = [
        "k_r",    // Reputation
        "k_a",    // Activity
        "k_i",    // Influence
        "k_p",    // Participation
        "k_m",    // Merit
        "k_s",    // Stake
        "k_h",    // History
        "k_topo", // Topology
    ];
}

/// Governance-specific constants
pub mod governance {
    use super::SecurityLevel;

    /// Minimum security level for standard votes
    pub const STANDARD_VOTE_MIN_SECURITY: SecurityLevel = SecurityLevel::Standard;

    /// Minimum security level for constitutional votes
    pub const CONSTITUTIONAL_VOTE_MIN_SECURITY: SecurityLevel = SecurityLevel::High;

    /// Minimum security level for treasury votes
    pub const TREASURY_VOTE_MIN_SECURITY: SecurityLevel = SecurityLevel::Standard;

    /// Default proof validity for votes (1 hour)
    pub const VOTE_PROOF_VALIDITY_SECS: u64 = 3600;

    /// Extended proof validity for constitutional votes (30 minutes - tighter)
    pub const CONSTITUTIONAL_PROOF_VALIDITY_SECS: u64 = 1800;
}

/// Winterfell-specific proof options (feature-gated)
#[cfg(feature = "winterfell")]
pub mod winterfell_options {
    use super::SecurityLevel;
    use winterfell::ProofOptions;

    /// Get Winterfell ProofOptions for the given security level
    pub fn proof_options_for_level(level: SecurityLevel) -> ProofOptions {
        match level {
            SecurityLevel::Fast => ProofOptions::new(
                28, // num_queries
                8,  // blowup_factor
                0,  // grinding_factor
                winterfell::FieldExtension::None,
                4,  // fri_folding_factor
                31, // fri_remainder_max_degree
            ),
            SecurityLevel::Optimized => ProofOptions::new(
                40, // num_queries
                8,  // blowup_factor
                16, // grinding_factor
                winterfell::FieldExtension::None,
                4,  // fri_folding_factor
                31, // fri_remainder_max_degree
            ),
            SecurityLevel::Standard => ProofOptions::new(
                50, // num_queries
                8,  // blowup_factor
                20, // grinding_factor
                winterfell::FieldExtension::None,
                8,   // fri_folding_factor
                127, // fri_remainder_max_degree
            ),
            SecurityLevel::High => ProofOptions::new(
                100, // num_queries
                16,  // blowup_factor
                24,  // grinding_factor
                winterfell::FieldExtension::Quadratic,
                8,   // fri_folding_factor
                255, // fri_remainder_max_degree
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_level_ordering() {
        assert!(SecurityLevel::Fast < SecurityLevel::Optimized);
        assert!(SecurityLevel::Optimized < SecurityLevel::Standard);
        assert!(SecurityLevel::Standard < SecurityLevel::High);
    }

    #[test]
    fn test_security_bits() {
        assert_eq!(SecurityLevel::Fast.estimated_security_bits(), 40);
        assert_eq!(SecurityLevel::Optimized.estimated_security_bits(), 84);
        assert_eq!(SecurityLevel::Standard.estimated_security_bits(), 96);
        assert_eq!(SecurityLevel::High.estimated_security_bits(), 264);
    }

    #[test]
    fn test_production_safety() {
        assert!(!SecurityLevel::Fast.is_production_safe());
        assert!(!SecurityLevel::Optimized.is_production_safe());
        assert!(SecurityLevel::Standard.is_production_safe());
        assert!(SecurityLevel::High.is_production_safe());
    }

    #[test]
    fn test_meets_minimum() {
        assert!(SecurityLevel::Standard.meets_minimum(96));
        assert!(SecurityLevel::Standard.meets_minimum(84));
        assert!(!SecurityLevel::Standard.meets_minimum(100));
        assert!(SecurityLevel::High.meets_minimum(200));
    }

    #[test]
    fn test_minimum_level_for_bits() {
        assert_eq!(
            SecurityLevel::minimum_level_for_bits(40),
            Some(SecurityLevel::Fast)
        );
        assert_eq!(
            SecurityLevel::minimum_level_for_bits(50),
            Some(SecurityLevel::Optimized)
        );
        assert_eq!(
            SecurityLevel::minimum_level_for_bits(96),
            Some(SecurityLevel::Standard)
        );
        assert_eq!(
            SecurityLevel::minimum_level_for_bits(128),
            Some(SecurityLevel::High)
        );
        assert_eq!(SecurityLevel::minimum_level_for_bits(300), None);
    }

    #[test]
    fn test_use_case_security_levels() {
        assert_eq!(
            UseCase::Testing.recommended_security_level(),
            SecurityLevel::Fast
        );
        assert_eq!(
            UseCase::StandardVote.recommended_security_level(),
            SecurityLevel::Standard
        );
        assert_eq!(
            UseCase::ConstitutionalVote.recommended_security_level(),
            SecurityLevel::High
        );
        assert_eq!(
            UseCase::TreasuryVote.recommended_security_level(),
            SecurityLevel::High
        );
    }

    #[test]
    fn test_proof_config_production() {
        let config = ProofConfig::production();
        assert!(config.is_production_safe());
        assert_eq!(config.security_level, SecurityLevel::Standard);
        assert!(config.strict_mode);
    }

    #[test]
    fn test_proof_config_for_use_case() {
        let config = ProofConfig::for_use_case(UseCase::ConstitutionalVote);
        assert_eq!(config.security_level, SecurityLevel::High);
        assert!(config.strict_mode);

        let config = ProofConfig::for_use_case(UseCase::Testing);
        assert_eq!(config.security_level, SecurityLevel::Fast);
        assert!(!config.strict_mode);
    }

    #[test]
    fn test_proof_config_builder() {
        let config = ProofConfig::new(SecurityLevel::Optimized)
            .with_strict_mode(true)
            .with_max_proof_size(512 * 1024)
            .with_validity_duration(7200);

        assert_eq!(config.security_level, SecurityLevel::Optimized);
        assert!(config.strict_mode);
        assert_eq!(config.max_proof_size, 512 * 1024);
        assert_eq!(config.validity_duration_secs, 7200);
    }

    #[test]
    fn test_kvector_constants() {
        assert_eq!(kvector::NUM_COMPONENTS, 8);
        assert_eq!(kvector::SCALE_FACTOR, 10_000);
        assert_eq!(kvector::COMPONENT_NAMES.len(), kvector::NUM_COMPONENTS);
    }

    #[test]
    fn test_serialization() {
        let config = ProofConfig::production();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: ProofConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(config, parsed);
    }

    #[test]
    fn test_security_level_serialization() {
        let level = SecurityLevel::Standard;
        let json = serde_json::to_string(&level).unwrap();
        assert_eq!(json, "\"Standard\"");
        let parsed: SecurityLevel = serde_json::from_str(&json).unwrap();
        assert_eq!(level, parsed);
    }
}
