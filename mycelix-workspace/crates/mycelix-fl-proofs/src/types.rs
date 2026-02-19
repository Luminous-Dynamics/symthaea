// Ported from fl-aggregator/src/proofs/types.rs
//! Common proof types for generation and verification.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Proof circuit types supported by this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProofType {
    /// Range proof — prove value is within bounds.
    Range,
    /// Merkle membership proof — prove element exists in tree.
    Membership,
    /// Gradient integrity proof — prove valid FL contribution.
    GradientIntegrity,
    /// Identity assurance proof — prove assurance level threshold.
    IdentityAssurance,
    /// Vote eligibility proof — prove voter qualification.
    VoteEligibility,
}

impl ProofType {
    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            ProofType::Range => "RangeProof",
            ProofType::Membership => "MembershipProof",
            ProofType::GradientIntegrity => "GradientIntegrityProof",
            ProofType::IdentityAssurance => "IdentityAssuranceProof",
            ProofType::VoteEligibility => "VoteEligibilityProof",
        }
    }
}

/// Security level for proof generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// 96-bit security (faster, smaller proofs).
    Standard96,
    /// 128-bit security (default, recommended).
    #[default]
    Standard128,
    /// 256-bit security (maximum security, larger proofs).
    High256,
}

impl SecurityLevel {
    /// Number of FRI queries.
    pub fn num_queries(&self) -> usize {
        match self {
            SecurityLevel::Standard96 => 27,
            SecurityLevel::Standard128 => 36,
            SecurityLevel::High256 => 72,
        }
    }

    /// Blowup factor for LDE.
    pub fn blowup_factor(&self) -> usize {
        match self {
            SecurityLevel::Standard96 | SecurityLevel::Standard128 => 8,
            SecurityLevel::High256 => 16,
        }
    }

    /// Grinding factor (proof-of-work bits).
    pub fn grinding_factor(&self) -> u32 {
        match self {
            SecurityLevel::Standard96 => 16,
            SecurityLevel::Standard128 => 20,
            SecurityLevel::High256 => 24,
        }
    }
}

/// Configuration for proof generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofConfig {
    /// Security level.
    pub security_level: SecurityLevel,
    /// Enable parallel proof generation.
    pub parallel: bool,
    /// Maximum proof size in bytes (0 = unlimited).
    pub max_proof_size: usize,
}

impl Default for ProofConfig {
    fn default() -> Self {
        Self {
            security_level: SecurityLevel::default(),
            parallel: true,
            max_proof_size: 0,
        }
    }
}

/// Result of proof verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofVerificationResult {
    /// Whether the proof is valid.
    pub valid: bool,
    /// Proof type that was verified.
    pub proof_type: ProofType,
    /// Verification time.
    pub verification_time: Duration,
    /// Additional verification details.
    pub details: Option<String>,
}

impl ProofVerificationResult {
    /// Create a successful verification result.
    pub fn success(proof_type: ProofType, time: Duration) -> Self {
        Self {
            valid: true,
            proof_type,
            verification_time: time,
            details: None,
        }
    }

    /// Create a failed verification result.
    pub fn failure(proof_type: ProofType, time: Duration, reason: impl Into<String>) -> Self {
        Self {
            valid: false,
            proof_type,
            verification_time: time,
            details: Some(reason.into()),
        }
    }
}

/// Blake3 commitment to a value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Commitment {
    /// The commitment value (256-bit hash).
    pub value: [u8; 32],
}

impl Commitment {
    /// Create a new commitment from bytes.
    pub fn new(value: [u8; 32]) -> Self {
        Self { value }
    }

    /// Create from a slice (must be 32 bytes).
    pub fn from_slice(bytes: &[u8]) -> Option<Self> {
        if bytes.len() == 32 {
            let mut value = [0u8; 32];
            value.copy_from_slice(bytes);
            Some(Self { value })
        } else {
            None
        }
    }

    /// Get as bytes.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.value
    }
}

/// Statistics about proof generation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProofStats {
    /// Number of proofs generated.
    pub proofs_generated: u64,
    /// Number of proofs verified.
    pub proofs_verified: u64,
    /// Number of verification failures.
    pub verification_failures: u64,
    /// Total proof generation time (milliseconds).
    pub total_generation_time_ms: u64,
    /// Total verification time (milliseconds).
    pub total_verification_time_ms: u64,
    /// Average proof size in bytes.
    pub average_proof_size: usize,
}

impl ProofStats {
    /// Record a proof generation.
    pub fn record_generation(&mut self, duration: Duration, size: usize) {
        self.proofs_generated += 1;
        self.total_generation_time_ms += duration.as_millis() as u64;
        if self.proofs_generated > 0 {
            let total_size =
                self.average_proof_size as u64 * (self.proofs_generated - 1) + size as u64;
            self.average_proof_size = (total_size / self.proofs_generated) as usize;
        }
    }

    /// Record a proof verification.
    pub fn record_verification(&mut self, duration: Duration, valid: bool) {
        self.proofs_verified += 1;
        self.total_verification_time_ms += duration.as_millis() as u64;
        if !valid {
            self.verification_failures += 1;
        }
    }

    /// Average generation time in milliseconds.
    pub fn avg_generation_time_ms(&self) -> u64 {
        if self.proofs_generated > 0 {
            self.total_generation_time_ms / self.proofs_generated
        } else {
            0
        }
    }

    /// Average verification time in milliseconds.
    pub fn avg_verification_time_ms(&self) -> u64 {
        if self.proofs_verified > 0 {
            self.total_verification_time_ms / self.proofs_verified
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_type_names() {
        assert_eq!(ProofType::Range.name(), "RangeProof");
        assert_eq!(ProofType::GradientIntegrity.name(), "GradientIntegrityProof");
    }

    #[test]
    fn test_security_levels() {
        assert!(SecurityLevel::Standard128.num_queries() > SecurityLevel::Standard96.num_queries());
        assert!(SecurityLevel::High256.num_queries() > SecurityLevel::Standard128.num_queries());
    }

    #[test]
    fn test_verification_result() {
        let success =
            ProofVerificationResult::success(ProofType::Range, Duration::from_millis(10));
        assert!(success.valid);

        let failure = ProofVerificationResult::failure(
            ProofType::Range,
            Duration::from_millis(5),
            "constraint failed",
        );
        assert!(!failure.valid);
        assert!(failure.details.is_some());
    }

    #[test]
    fn test_proof_stats() {
        let mut stats = ProofStats::default();
        stats.record_generation(Duration::from_millis(100), 1000);
        stats.record_generation(Duration::from_millis(200), 2000);

        assert_eq!(stats.proofs_generated, 2);
        assert_eq!(stats.avg_generation_time_ms(), 150);
        assert_eq!(stats.average_proof_size, 1500);
    }

    #[test]
    fn test_commitment() {
        let bytes = [42u8; 32];
        let commitment = Commitment::new(bytes);
        assert_eq!(commitment.as_bytes(), &bytes);

        let from_slice = Commitment::from_slice(&bytes).unwrap();
        assert_eq!(commitment, from_slice);
        assert!(Commitment::from_slice(&[0u8; 16]).is_none());
    }

    #[test]
    fn test_default_config() {
        let config = ProofConfig::default();
        assert_eq!(config.security_level, SecurityLevel::Standard128);
        assert!(config.parallel);
    }
}
