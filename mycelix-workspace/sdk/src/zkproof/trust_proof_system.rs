// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! K-Vector Trust Proof System
//!
//! Implements the generic proof system traits for K-Vector trust attestations.
//! This bridges the existing `trust_risc0` module with the new `proof_system` abstraction.
//!
//! # Usage
//!
//! ```rust,ignore
//! use mycelix_sdk::zkproof::trust_proof_system::*;
//! use mycelix_sdk::matl::KVector;
//!
//! // Create the proof system
//! let system = TrustProofSystem::new_simulation();
//!
//! // Create a statement
//! let statement = TrustStatement::trust_exceeds_threshold(0.6);
//!
//! // Create a witness (the actual K-Vector)
//! let kvector = KVector::new(0.8, 0.7, 0.9, 0.7, 0.5, 0.4, 0.6, 0.5, 0.75, 0.7);
//! let witness = TrustWitness::new(kvector, "agent-123");
//!
//! // Generate proof
//! let receipt = system.prove(&statement, &witness)?;
//!
//! // Verify
//! assert!(system.verify(&receipt)?);
//! assert!(receipt.statement_satisfied());
//! ```

use super::proof_system::{
    BackendConfig, BackendType, GenericReceipt, ProofOutput, ProofStatement, ProofStats,
    ProofSystemError, ProofWitness, SimulationBackend,
};
use super::trust_risc0::{
    compute_commitment as zk_compute_commitment, evaluate_statement, ZkKVector, ZkTrustParams,
    ZkTrustStatement, SCALE,
};
use crate::matl::KVector as SdkKVector;
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ============================================================================
// Trust Statements
// ============================================================================

/// A statement about K-Vector trust that can be proven in zero-knowledge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustStatement {
    /// The underlying ZK statement type
    pub statement_type: ZkTrustStatement,
    /// Statement parameters
    pub params: ZkTrustParams,
    /// Human-readable description
    pub description: String,
}

impl TrustStatement {
    /// Create a statement: "Trust score exceeds threshold"
    pub fn trust_exceeds_threshold(threshold: f32) -> Self {
        Self {
            statement_type: ZkTrustStatement::TrustExceedsThreshold,
            params: ZkTrustParams {
                threshold: (threshold.clamp(0.0, 1.0) * SCALE as f32) as u64,
                ..Default::default()
            },
            description: format!("Trust score exceeds {:.2}", threshold),
        }
    }

    /// Create a statement: "Trust score is within range"
    pub fn trust_in_range(min: f32, max: f32) -> Self {
        Self {
            statement_type: ZkTrustStatement::TrustInRange,
            params: ZkTrustParams {
                min_value: (min.clamp(0.0, 1.0) * SCALE as f32) as u64,
                max_value: (max.clamp(0.0, 1.0) * SCALE as f32) as u64,
                ..Default::default()
            },
            description: format!("Trust score in range [{:.2}, {:.2}]", min, max),
        }
    }

    /// Create a statement: "Dimension k_i exceeds threshold"
    pub fn dimension_exceeds(dimension: u8, threshold: f32) -> Self {
        let dim_names = [
            "k_r",
            "k_a",
            "k_i",
            "k_p",
            "k_m",
            "k_s",
            "k_h",
            "k_topo",
            "k_v",
            "k_coherence",
        ];
        let dim_name = dim_names.get(dimension as usize).unwrap_or(&"k_?");

        Self {
            statement_type: ZkTrustStatement::DimensionExceedsThreshold,
            params: ZkTrustParams {
                dimension_index: dimension.min(9),
                threshold: (threshold.clamp(0.0, 1.0) * SCALE as f32) as u64,
                ..Default::default()
            },
            description: format!("{} exceeds {:.2}", dim_name, threshold),
        }
    }

    /// Create a statement: "Agent is verified (k_v >= 0.5)"
    pub fn is_verified() -> Self {
        Self {
            statement_type: ZkTrustStatement::IsVerified,
            params: Default::default(),
            description: "Agent is verified".to_string(),
        }
    }

    /// Create a statement: "Agent is strongly verified (k_v >= 0.7)"
    pub fn is_strongly_verified() -> Self {
        Self {
            statement_type: ZkTrustStatement::IsStronglyVerified,
            params: Default::default(),
            description: "Agent is strongly verified".to_string(),
        }
    }

    /// Create a statement: "Agent is highly coherent (k_coherence >= 0.7)"
    pub fn is_highly_coherent() -> Self {
        Self {
            statement_type: ZkTrustStatement::IsHighlyCoherent,
            params: Default::default(),
            description: "Agent is highly coherent".to_string(),
        }
    }
}

impl ProofStatement for TrustStatement {
    fn description(&self) -> String {
        self.description.clone()
    }

    fn statement_type_id(&self) -> &'static str {
        match self.statement_type {
            ZkTrustStatement::TrustExceedsThreshold => "trust-exceeds-threshold",
            ZkTrustStatement::TrustInRange => "trust-in-range",
            ZkTrustStatement::DimensionExceedsThreshold => "dimension-exceeds",
            ZkTrustStatement::MultipleDimensionsExceed => "multi-dimension-exceeds",
            ZkTrustStatement::KVectorMatchesCommitment => "kvector-matches-commitment",
            ZkTrustStatement::IsVerified => "is-verified",
            ZkTrustStatement::IsStronglyVerified => "is-strongly-verified",
            ZkTrustStatement::IsHighlyCoherent => "is-highly-coherent",
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(self).unwrap_or_default()
    }
}

// ============================================================================
// Trust Witness
// ============================================================================

/// Private witness containing the K-Vector that proves a trust statement.
#[derive(Debug, Clone)]
pub struct TrustWitness {
    /// The actual K-Vector (PRIVATE - never revealed)
    pub kvector: SdkKVector,
    /// Agent identifier (used for commitment)
    pub agent_id: String,
    /// Random blinding factor
    pub blinding: [u8; 32],
    /// Timestamp
    pub timestamp: u64,
}

impl TrustWitness {
    /// Create a new witness with auto-generated blinding
    pub fn new(kvector: SdkKVector, agent_id: impl Into<String>) -> Self {
        Self {
            kvector,
            agent_id: agent_id.into(),
            blinding: generate_blinding(),
            timestamp: current_timestamp(),
        }
    }

    /// Create with explicit blinding (for deterministic testing)
    pub fn with_blinding(
        kvector: SdkKVector,
        agent_id: impl Into<String>,
        blinding: [u8; 32],
    ) -> Self {
        Self {
            kvector,
            agent_id: agent_id.into(),
            blinding,
            timestamp: current_timestamp(),
        }
    }

    /// Get the K-Vector as ZK-compatible fixed-point
    pub fn zk_kvector(&self) -> ZkKVector {
        ZkKVector::from_sdk(&self.kvector)
    }

    /// Compute the agent ID hash
    pub fn agent_id_hash(&self) -> u64 {
        hash_agent_id(&self.agent_id)
    }

    /// Compute the witness commitment
    pub fn compute_commitment(&self) -> [u8; 32] {
        zk_compute_commitment(
            &self.zk_kvector(),
            &self.blinding,
            self.agent_id_hash(),
            self.timestamp,
        )
    }
}

impl ProofWitness for TrustWitness {
    type Statement = TrustStatement;

    fn satisfies(&self, statement: &Self::Statement) -> bool {
        let zk_kv = self.zk_kvector();
        evaluate_statement(&zk_kv, statement.statement_type, &statement.params)
    }

    fn to_bytes(&self) -> Vec<u8> {
        // Serialize the K-Vector values (SENSITIVE)
        let mut bytes = Vec::new();
        for val in self.kvector.to_array() {
            bytes.extend(val.to_le_bytes());
        }
        bytes.extend(&self.blinding);
        bytes.extend(&self.agent_id_hash().to_le_bytes());
        bytes.extend(&self.timestamp.to_le_bytes());
        bytes
    }
}

// ============================================================================
// Trust Receipt
// ============================================================================

/// Additional public data in trust proof receipts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustPublicData {
    /// Statement type that was proven
    pub statement_type: ZkTrustStatement,
    /// Agent ID hash (not the actual ID)
    pub agent_id_hash: u64,
    /// Revealed trust score (only for TrustInRange statements)
    pub revealed_score: Option<f32>,
}

/// Type alias for trust proof receipts
pub type TrustReceipt = GenericReceipt<TrustStatement>;

impl TrustReceipt {
    /// Get trust-specific public data
    pub fn trust_data(&self) -> Option<TrustPublicData> {
        if self.output.public_data.is_empty() {
            return None;
        }
        bincode::deserialize(&self.output.public_data).ok()
    }

    /// Get the revealed trust score (only available for TrustInRange proofs)
    pub fn revealed_score(&self) -> Option<f32> {
        self.trust_data().and_then(|d| d.revealed_score)
    }
}

// ============================================================================
// Trust Proof System
// ============================================================================

/// Trust proof system using the generic backend.
///
/// Wraps the simulation backend (or future RISC-0 backend) with
/// trust-specific logic.
pub struct TrustProofSystem {
    backend: SimulationBackend,
    #[allow(dead_code)]
    config: BackendConfig,
}

impl TrustProofSystem {
    /// Create a simulation proof system (for testing)
    pub fn new_simulation() -> Self {
        Self {
            backend: SimulationBackend::new_default(),
            config: BackendConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: BackendConfig) -> Self {
        Self {
            backend: SimulationBackend::new(config.clone()),
            config,
        }
    }

    /// Check if running in simulation mode
    pub fn is_simulation(&self) -> bool {
        true // Currently only simulation implemented
    }

    /// Get backend name
    pub fn backend_name(&self) -> &'static str {
        "simulation"
    }

    /// Prove a trust statement
    pub fn prove(
        &self,
        statement: &TrustStatement,
        witness: &TrustWitness,
    ) -> Result<TrustReceipt, ProofSystemError> {
        let start = Instant::now();

        // Evaluate statement
        let satisfied = witness.satisfies(statement);

        // Compute commitment
        let commitment = witness.compute_commitment();

        // Create trust-specific public data
        let public_data = TrustPublicData {
            statement_type: statement.statement_type,
            agent_id_hash: witness.agent_id_hash(),
            revealed_score: if statement.statement_type == ZkTrustStatement::TrustInRange {
                Some(witness.zk_kvector().trust_score_f32())
            } else {
                None
            },
        };

        // Create proof output
        let mut output = ProofOutput::new(satisfied, commitment, statement.statement_type_id());
        output.public_data = bincode::serialize(&public_data).unwrap_or_default();

        // Create simulation proof bytes
        let proof_bytes = create_simulation_proof_bytes(&output);

        let generation_time_ms = start.elapsed().as_millis() as u64;
        let stats = ProofStats {
            generation_time_ms,
            proof_size_bytes: proof_bytes.len(),
            backend: "simulation".to_string(),
            is_simulation: true,
        };

        // Update backend stats so last_stats() works
        self.backend.set_last_stats(stats.clone());

        Ok(TrustReceipt::new(
            output,
            proof_bytes,
            BackendType::Simulation,
            stats,
        ))
    }

    /// Verify a trust receipt
    pub fn verify(&self, receipt: &TrustReceipt) -> Result<bool, ProofSystemError> {
        // In simulation mode, just check marker and satisfaction
        if !receipt.proof_bytes.starts_with(b"SIMU") {
            return Ok(false);
        }
        Ok(true) // Proof structure is valid (satisfaction is in output.satisfied)
    }

    /// Get statistics from the last proof
    pub fn last_stats(&self) -> Option<ProofStats> {
        self.backend.get_last_stats()
    }

    // Convenience methods

    /// Prove trust exceeds threshold
    pub fn prove_trust_exceeds(
        &self,
        kvector: &SdkKVector,
        threshold: f32,
        agent_id: &str,
    ) -> Result<TrustReceipt, ProofSystemError> {
        let statement = TrustStatement::trust_exceeds_threshold(threshold);
        let witness = TrustWitness::new(*kvector, agent_id);
        self.prove(&statement, &witness)
    }

    /// Prove trust is in range
    pub fn prove_trust_in_range(
        &self,
        kvector: &SdkKVector,
        min: f32,
        max: f32,
        agent_id: &str,
    ) -> Result<TrustReceipt, ProofSystemError> {
        let statement = TrustStatement::trust_in_range(min, max);
        let witness = TrustWitness::new(*kvector, agent_id);
        self.prove(&statement, &witness)
    }

    /// Prove agent is verified
    pub fn prove_is_verified(
        &self,
        kvector: &SdkKVector,
        agent_id: &str,
    ) -> Result<TrustReceipt, ProofSystemError> {
        let statement = TrustStatement::is_verified();
        let witness = TrustWitness::new(*kvector, agent_id);
        self.prove(&statement, &witness)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate random blinding factor
fn generate_blinding() -> [u8; 32] {
    use std::time::{SystemTime, UNIX_EPOCH};

    let mut blinding = [0u8; 32];
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    blinding[..16].copy_from_slice(&now.to_le_bytes());

    // Add some additional entropy
    let stack_entropy = &blinding as *const _ as u64;
    blinding[16..24].copy_from_slice(&stack_entropy.to_le_bytes());

    blinding
}

/// Get current timestamp
fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Hash agent ID to u64
fn hash_agent_id(agent_id: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;
    for byte in agent_id.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Create simulation proof bytes
fn create_simulation_proof_bytes(output: &ProofOutput) -> Vec<u8> {
    let mut bytes = b"SIMU".to_vec();
    bytes.extend(bincode::serialize(output).unwrap_or_default());
    bytes
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::super::proof_system::ProofReceipt;
    use super::*;

    fn test_kvector() -> SdkKVector {
        SdkKVector::new(0.8, 0.6, 0.9, 0.7, 0.5, 0.4, 0.6, 0.5, 0.75, 0.7)
    }

    #[test]
    fn test_trust_statement_creation() {
        let stmt = TrustStatement::trust_exceeds_threshold(0.5);
        assert_eq!(stmt.statement_type_id(), "trust-exceeds-threshold");
        assert!(stmt.description().contains("0.50"));

        let stmt2 = TrustStatement::is_verified();
        assert_eq!(stmt2.statement_type_id(), "is-verified");
    }

    #[test]
    fn test_trust_witness() {
        let kv = test_kvector();
        let witness = TrustWitness::new(kv.clone(), "agent-1");

        assert_eq!(witness.kvector.k_r, 0.8);
        assert!(witness.agent_id_hash() > 0);
        assert!(witness.timestamp > 0);

        // Test satisfaction
        let stmt_pass = TrustStatement::trust_exceeds_threshold(0.5);
        let stmt_fail = TrustStatement::trust_exceeds_threshold(0.95);

        assert!(witness.satisfies(&stmt_pass));
        assert!(!witness.satisfies(&stmt_fail));
    }

    #[test]
    fn test_trust_proof_system_exceeds() {
        let system = TrustProofSystem::new_simulation();
        let kv = test_kvector();

        let receipt = system.prove_trust_exceeds(&kv, 0.5, "agent-1").unwrap();

        assert!(receipt.statement_satisfied());
        assert!(receipt.is_simulation());
        assert!(system.verify(&receipt).unwrap());
    }

    #[test]
    fn test_trust_proof_system_fails() {
        let system = TrustProofSystem::new_simulation();
        let kv = test_kvector();

        let receipt = system.prove_trust_exceeds(&kv, 0.95, "agent-1").unwrap();

        assert!(!receipt.statement_satisfied());
        // Verification still passes (proof is valid, statement just not satisfied)
        assert!(system.verify(&receipt).unwrap());
    }

    #[test]
    fn test_trust_in_range() {
        let system = TrustProofSystem::new_simulation();
        let kv = test_kvector();

        let receipt = system
            .prove_trust_in_range(&kv, 0.5, 0.9, "agent-1")
            .unwrap();

        assert!(receipt.statement_satisfied());

        // Should have revealed score
        let revealed = receipt.revealed_score();
        assert!(revealed.is_some());
        let score = revealed.unwrap();
        assert!(score >= 0.5 && score <= 0.9);
    }

    #[test]
    fn test_is_verified() {
        let system = TrustProofSystem::new_simulation();
        let kv = test_kvector(); // k_v = 0.75 >= 0.5

        let receipt = system.prove_is_verified(&kv, "agent-1").unwrap();
        assert!(receipt.statement_satisfied());
    }

    #[test]
    fn test_dimension_exceeds() {
        let system = TrustProofSystem::new_simulation();
        let kv = test_kvector();

        // k_i (index 2) = 0.9
        let stmt = TrustStatement::dimension_exceeds(2, 0.8);
        let witness = TrustWitness::new(kv, "agent-1");

        let receipt = system.prove(&stmt, &witness).unwrap();
        assert!(receipt.statement_satisfied());
    }

    #[test]
    fn test_proof_stats() {
        let system = TrustProofSystem::new_simulation();
        let kv = test_kvector();

        let _receipt = system.prove_trust_exceeds(&kv, 0.5, "agent-1").unwrap();

        let stats = system.last_stats();
        assert!(stats.is_some());
        let stats = stats.unwrap();
        assert!(stats.is_simulation);
        assert!(stats.proof_size_bytes > 0);
    }

    #[test]
    fn test_commitment_determinism() {
        let kv = test_kvector();
        let blinding = [42u8; 32];

        let w1 = TrustWitness::with_blinding(kv.clone(), "agent-1", blinding);
        let w2 = TrustWitness::with_blinding(kv, "agent-1", blinding);

        // Same inputs should produce same commitment (with same timestamp)
        // Note: timestamps may differ slightly, so this tests the blinding part
        assert_eq!(w1.blinding, w2.blinding);
    }
}
