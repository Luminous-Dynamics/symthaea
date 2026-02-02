//! # Compute Infrastructure PoG
//!
//! Processing capacity verification for Proof of Grounding.

use super::{PogScore, VerificationLevel};
use serde::{Deserialize, Serialize};

/// Trusted Execution Environment type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TeeType {
    /// Intel SGX
    IntelSgx,
    /// ARM TrustZone
    ArmTrustZone,
    /// AMD SEV
    AmdSev,
    /// RISC-V Keystone
    RiscVKeystone,
    /// No TEE (software only)
    None,
}

impl TeeType {
    /// Get PoG multiplier for TEE type
    pub fn multiplier(&self) -> f64 {
        match self {
            TeeType::IntelSgx => 1.4,
            TeeType::ArmTrustZone => 1.3,
            TeeType::AmdSev => 1.4,
            TeeType::RiscVKeystone => 1.2,
            TeeType::None => 1.0,
        }
    }
}

/// Compute attestation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeAttestation {
    /// Provider DID
    pub provider_did: String,
    /// TEE type
    pub tee_type: TeeType,
    /// Remote attestation data
    pub remote_attestation: Option<RemoteAttestation>,
    /// Benchmark results
    pub benchmark_results: BenchmarkResults,
    /// Availability hours per week
    pub availability_hours_per_week: u32,
    /// Verification level
    pub verification_level: VerificationLevel,
}

/// Remote attestation from TEE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteAttestation {
    /// Quote/report from TEE
    pub quote: Vec<u8>,
    /// Signature over quote
    pub signature: Vec<u8>,
    /// Attestation timestamp
    pub timestamp: u64,
    /// Platform configuration
    pub platform_config: Vec<u8>,
}

/// Benchmark results for compute capacity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// FLOPS available (floating point operations per second)
    pub flops_available: u64,
    /// Memory in GB
    pub memory_gb: u32,
    /// Benchmark timestamp
    pub measured_at: u64,
    /// Verifier signature
    pub verifier_signature: Option<Vec<u8>>,
}

impl ComputeAttestation {
    /// Calculate PoG score
    pub fn calculate_pog(&self) -> PogScore {
        const MIN_TFLOPS: f64 = 1.0;

        let tflops = self.benchmark_results.flops_available as f64 / 1e12;
        if tflops < MIN_TFLOPS {
            return PogScore::zero();
        }

        // Logarithmic scaling
        let compute_factor = tflops.ln() + 1.0;

        // Availability factor (40 hrs/week = 1.0)
        let availability = (self.availability_hours_per_week as f64 / 40.0).min(1.0);

        // TEE multiplier
        let tee_mult = self.tee_type.multiplier();

        // Verification weight
        let verification = self.verification_level.weight();

        let score = (compute_factor * availability * tee_mult * verification * 0.15).min(1.0);

        PogScore::new(score)
    }

    /// Check if remote attestation is valid
    pub fn has_valid_attestation(&self) -> bool {
        matches!(self.tee_type, TeeType::None) || self.remote_attestation.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_pog() {
        let attestation = ComputeAttestation {
            provider_did: "did:test:1".to_string(),
            tee_type: TeeType::IntelSgx,
            remote_attestation: Some(RemoteAttestation {
                quote: vec![1, 2, 3],
                signature: vec![4, 5, 6],
                timestamp: 1000,
                platform_config: vec![],
            }),
            benchmark_results: BenchmarkResults {
                flops_available: 5_000_000_000_000, // 5 TFLOPS
                memory_gb: 32,
                measured_at: 1000,
                verifier_signature: None,
            },
            availability_hours_per_week: 40,
            verification_level: VerificationLevel::HardwareAttested,
        };

        let pog = attestation.calculate_pog();
        assert!(pog.value() > 0.0);
    }

    #[test]
    fn test_minimum_compute() {
        let attestation = ComputeAttestation {
            provider_did: "did:test:1".to_string(),
            tee_type: TeeType::None,
            remote_attestation: None,
            benchmark_results: BenchmarkResults {
                flops_available: 500_000_000_000, // 0.5 TFLOPS (below minimum)
                memory_gb: 8,
                measured_at: 1000,
                verifier_signature: None,
            },
            availability_hours_per_week: 20,
            verification_level: VerificationLevel::SelfReported,
        };

        let pog = attestation.calculate_pog();
        assert!((pog.value() - 0.0).abs() < 0.001);
    }
}
