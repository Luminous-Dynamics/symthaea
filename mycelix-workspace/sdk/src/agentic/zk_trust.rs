//! # Zero-Knowledge Trust Proofs
//!
//! Cryptographic proofs of K-Vector properties without revealing values.
//!
//! ## Philosophy
//!
//! Trust should be verifiable without being exposable. An agent can prove:
//! - "My trust score exceeds 0.7" without revealing it's exactly 0.82
//! - "My integrity is high" without exposing reputation or activity
//! - "I improved over time" without showing the trajectory
//!
//! ## Architecture
//!
//! - **ProofStatement**: What property is being proven
//! - **KVectorCommitment**: Cryptographic commitment to a K-Vector
//! - **TrustProof**: A verifiable proof of a statement
//! - **ProofVerifier**: Verification without K-Vector access
//!
//! ## Security Model
//!
//! In production (risc0 feature): Real ZK proofs using RISC-0 zkVM
//! In simulation: Deterministic proofs for testing (NOT cryptographically secure)
//!
//! WARNING: Never use simulation mode in production!

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;

use super::provenance::{DerivationType, ProvenanceChain};
use crate::matl::{GovernanceTier, KVector, KVectorDimension};

#[cfg(feature = "ts-export")]
use ts_rs::TS;

// ============================================================================
// Proof Statements
// ============================================================================

/// What property of a K-Vector is being proven
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum ProofStatement {
    /// Trust score exceeds threshold: T >= threshold
    TrustExceedsThreshold {
        /// Minimum trust threshold
        threshold: f32,
    },

    /// Trust score is within range: lower <= T <= upper
    TrustInRange {
        /// Lower bound (inclusive)
        lower: f32,
        /// Upper bound (inclusive)
        upper: f32,
    },

    /// Single dimension exceeds threshold: k_x >= threshold
    DimensionExceedsThreshold {
        /// K-Vector dimension to check
        dimension: KVectorDimension,
        /// Minimum dimension value
        threshold: f32,
    },

    /// Multiple dimensions all exceed thresholds
    MultipleDimensionsExceed {
        /// List of (dimension, threshold) requirements
        requirements: Vec<(KVectorDimension, f32)>,
    },

    /// Meets requirements for a governance tier
    MeetsGovernanceTier {
        /// Governance tier requirements
        tier: GovernanceTier,
    },

    /// K-Vector is well-formed (all values in \[0,1\])
    WellFormed,

    /// Trust improved: T(now) > T(previous)
    TrustImproved {
        /// Commitment to previous K-Vector
        previous_commitment: [u8; 32],
    },

    /// Dimension improved: k_x(now) > k_x(previous)
    DimensionImproved {
        /// Dimension to check for improvement
        dimension: KVectorDimension,
        /// Commitment to previous K-Vector
        previous_commitment: [u8; 32],
    },

    /// Agent is verified (k_v >= 0.5)
    IsVerified,

    /// Agent is strongly verified (k_v >= 0.7)
    IsStronglyVerified,

    /// Compound: All of the given statements are true
    And(
        /// List of statements that must all be true
        Vec<ProofStatement>,
    ),

    /// Compound: At least one of the given statements is true
    Or(
        /// List of statements, at least one must be true
        Vec<ProofStatement>,
    ),

    // ========================================================================
    // Provenance-Related Proofs (Option C)
    // ========================================================================
    /// Prove output was derived from sources at minimum epistemic level
    /// "I derived this from E3+ sources" without revealing which ones
    ProvenanceEpistemicFloor {
        /// Minimum epistemic level of all sources
        min_level: u8,
    },

    /// Prove derivation chain depth is within limits
    /// "My knowledge chain is at most N hops from original sources"
    ProvenanceChainDepth {
        /// Maximum allowed chain depth
        max_depth: u32,
    },

    /// Prove minimum confidence through derivation chain
    /// "My derived confidence is at least X"
    ProvenanceConfidence {
        /// Minimum derived confidence
        min_confidence: f64,
    },

    /// Prove derivation uses only specific types
    /// "I only used citation and verification, not inference"
    ProvenanceDerivationTypes {
        /// Allowed derivation types (by ordinal)
        allowed_types: Vec<u8>,
    },

    /// Prove chain has verified root sources
    /// "All my root sources are original (not derived)"
    ProvenanceHasVerifiedRoots,

    /// Prove chain integrity (all commitments verify)
    ProvenanceChainValid,

    /// Combined provenance quality proof
    ProvenanceQuality {
        /// Minimum epistemic floor
        min_epistemic_level: u8,
        /// Maximum chain depth
        max_chain_depth: u32,
        /// Minimum derived confidence
        min_confidence: f64,
    },
}

impl ProofStatement {
    /// Create a threshold proof for trust score
    pub fn trust_above(threshold: f32) -> Self {
        Self::TrustExceedsThreshold { threshold }
    }

    /// Create a dimension threshold proof
    pub fn dimension_above(dimension: KVectorDimension, threshold: f32) -> Self {
        Self::DimensionExceedsThreshold {
            dimension,
            threshold,
        }
    }

    /// Create a governance tier proof
    pub fn meets_tier(tier: GovernanceTier) -> Self {
        Self::MeetsGovernanceTier { tier }
    }

    /// Create compound AND proof
    pub fn all(statements: Vec<ProofStatement>) -> Self {
        Self::And(statements)
    }

    /// Create compound OR proof
    pub fn any(statements: Vec<ProofStatement>) -> Self {
        Self::Or(statements)
    }

    // Provenance proof constructors

    /// Prove epistemic floor of provenance chain
    pub fn provenance_epistemic_floor(min_level: u8) -> Self {
        Self::ProvenanceEpistemicFloor { min_level }
    }

    /// Prove chain depth limit
    pub fn provenance_depth(max_depth: u32) -> Self {
        Self::ProvenanceChainDepth { max_depth }
    }

    /// Prove minimum confidence through chain
    pub fn provenance_confidence(min_confidence: f64) -> Self {
        Self::ProvenanceConfidence { min_confidence }
    }

    /// Prove combined provenance quality
    pub fn provenance_quality(
        min_epistemic_level: u8,
        max_chain_depth: u32,
        min_confidence: f64,
    ) -> Self {
        Self::ProvenanceQuality {
            min_epistemic_level,
            max_chain_depth,
            min_confidence,
        }
    }
}

// ============================================================================
// Commitments
// ============================================================================

/// Cryptographic commitment to a K-Vector
///
/// Binds the agent to a specific K-Vector state without revealing it.
/// Can be used later to prove properties or show improvement.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct KVectorCommitment {
    /// SHA3-256 commitment hash
    pub commitment: [u8; 32],

    /// Agent who made the commitment
    pub agent_id: String,

    /// Timestamp of commitment
    pub timestamp: u64,

    /// Optional epoch/round number
    pub epoch: Option<u64>,
}

impl KVectorCommitment {
    /// Create a new commitment from a K-Vector
    pub fn create(kvector: &KVector, agent_id: &str, timestamp: u64, blinding: &[u8; 32]) -> Self {
        let commitment = compute_commitment(kvector, agent_id, timestamp, blinding);
        Self {
            commitment,
            agent_id: agent_id.to_string(),
            timestamp,
            epoch: None,
        }
    }

    /// Create with epoch
    pub fn create_with_epoch(
        kvector: &KVector,
        agent_id: &str,
        timestamp: u64,
        epoch: u64,
        blinding: &[u8; 32],
    ) -> Self {
        let mut c = Self::create(kvector, agent_id, timestamp, blinding);
        c.epoch = Some(epoch);
        c
    }

    /// Verify the commitment opens to the given K-Vector
    pub fn verify_opening(&self, kvector: &KVector, blinding: &[u8; 32]) -> bool {
        let computed = compute_commitment(kvector, &self.agent_id, self.timestamp, blinding);
        constant_time_eq(&self.commitment, &computed)
    }
}

/// Compute Pedersen-style commitment (simplified for simulation)
fn compute_commitment(
    kvector: &KVector,
    agent_id: &str,
    timestamp: u64,
    blinding: &[u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha3_256::new();

    // Domain separation
    hasher.update(b"mycelix-kvector-commitment-v1");

    // Include all K-Vector values
    for val in kvector.to_array() {
        hasher.update(val.to_le_bytes());
    }

    // Include agent and timestamp
    hasher.update(agent_id.as_bytes());
    hasher.update(timestamp.to_le_bytes());

    // Include blinding factor
    hasher.update(blinding);

    hasher.finalize().into()
}

/// Constant-time comparison
fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    result == 0
}

// ============================================================================
// Proofs
// ============================================================================

/// A zero-knowledge proof of a K-Vector property
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct TrustProof {
    /// What is being proven
    pub statement: ProofStatement,

    /// Commitment to the K-Vector
    pub commitment: KVectorCommitment,

    /// The proof data (format depends on backend)
    pub proof_data: ProofData,

    /// Proof generation timestamp
    pub created_at: u64,

    /// Optional: Previous commitment (for improvement proofs)
    pub previous_commitment: Option<KVectorCommitment>,
}

/// Proof data (simulation or real ZK)
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum ProofData {
    /// Simulation proof (NOT cryptographically secure)
    /// Contains a hash that can be recomputed for verification
    Simulation {
        /// Hash of (statement, commitment, result)
        verification_hash: [u8; 32],
        /// The result of the statement (true/false)
        result: bool,
    },

    /// RISC-0 zkVM proof (cryptographically secure)
    #[cfg(feature = "risc0")]
    Risc0 {
        /// Serialized RISC-0 receipt
        receipt: Vec<u8>,
        /// Image ID of the prover program
        image_id: [u8; 32],
    },

    /// Placeholder for when risc0 feature is not enabled
    #[cfg(not(feature = "risc0"))]
    Risc0Placeholder,
}

impl TrustProof {
    /// Check if this is a simulation proof (not secure for production)
    pub fn is_simulation(&self) -> bool {
        matches!(self.proof_data, ProofData::Simulation { .. })
    }

    /// Get the proof result (from simulation or by verifying ZK proof)
    pub fn get_result(&self) -> Option<bool> {
        match &self.proof_data {
            ProofData::Simulation { result, .. } => Some(*result),
            #[cfg(feature = "risc0")]
            ProofData::Risc0 { .. } => None, // Must verify
            #[cfg(not(feature = "risc0"))]
            ProofData::Risc0Placeholder => None,
        }
    }
}

// ============================================================================
// Prover
// ============================================================================

/// Configuration for proof generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProverConfig {
    /// Use simulation mode (WARNING: not secure for production!)
    pub simulation_mode: bool,

    /// Agent ID for commitments
    pub agent_id: String,

    /// Current timestamp function (for tests)
    pub timestamp: u64,
}

impl Default for ProverConfig {
    fn default() -> Self {
        Self {
            #[cfg(feature = "simulation")]
            simulation_mode: true,
            #[cfg(not(feature = "simulation"))]
            simulation_mode: false,
            agent_id: String::new(),
            timestamp: 0,
        }
    }
}

/// Generates zero-knowledge proofs for K-Vector properties
pub struct TrustProver {
    config: ProverConfig,
}

impl TrustProver {
    /// Create a new prover
    pub fn new(config: ProverConfig) -> Self {
        Self { config }
    }

    /// Create a commitment to a K-Vector
    pub fn commit(&self, kvector: &KVector, blinding: &[u8; 32]) -> KVectorCommitment {
        KVectorCommitment::create(
            kvector,
            &self.config.agent_id,
            self.config.timestamp,
            blinding,
        )
    }

    /// Generate a proof for a statement
    pub fn prove(
        &self,
        kvector: &KVector,
        statement: &ProofStatement,
        blinding: &[u8; 32],
    ) -> Result<TrustProof, ProofError> {
        // Evaluate the statement
        let result = evaluate_statement(kvector, statement, None)?;

        // Create commitment
        let commitment = self.commit(kvector, blinding);

        // Generate proof data
        let proof_data = if self.config.simulation_mode {
            self.generate_simulation_proof(&commitment, statement, result)
        } else {
            #[cfg(feature = "risc0")]
            {
                self.generate_risc0_proof(kvector, statement)?
            }
            #[cfg(not(feature = "risc0"))]
            {
                return Err(ProofError::Risc0NotEnabled);
            }
        };

        Ok(TrustProof {
            statement: statement.clone(),
            commitment,
            proof_data,
            created_at: self.config.timestamp,
            previous_commitment: None,
        })
    }

    /// Generate proof of improvement between two K-Vectors
    pub fn prove_improvement(
        &self,
        previous: &KVector,
        current: &KVector,
        dimension: Option<KVectorDimension>,
        previous_blinding: &[u8; 32],
        current_blinding: &[u8; 32],
    ) -> Result<TrustProof, ProofError> {
        // Create previous commitment
        let prev_commitment = KVectorCommitment::create(
            previous,
            &self.config.agent_id,
            self.config.timestamp.saturating_sub(1), // Previous timestamp
            previous_blinding,
        );

        // Check if improvement occurred
        let result = match dimension {
            Some(dim) => {
                let prev_val = get_dimension(previous, dim);
                let curr_val = get_dimension(current, dim);
                curr_val > prev_val
            }
            None => current.trust_score() > previous.trust_score(),
        };

        // Create statement
        let statement = match dimension {
            Some(dim) => ProofStatement::DimensionImproved {
                dimension: dim,
                previous_commitment: prev_commitment.commitment,
            },
            None => ProofStatement::TrustImproved {
                previous_commitment: prev_commitment.commitment,
            },
        };

        // Create current commitment
        let commitment = self.commit(current, current_blinding);

        // Generate proof
        let proof_data = if self.config.simulation_mode {
            self.generate_simulation_proof(&commitment, &statement, result)
        } else {
            #[cfg(feature = "risc0")]
            {
                return Err(ProofError::Risc0NotImplemented);
            }
            #[cfg(not(feature = "risc0"))]
            {
                return Err(ProofError::Risc0NotEnabled);
            }
        };

        Ok(TrustProof {
            statement,
            commitment,
            proof_data,
            created_at: self.config.timestamp,
            previous_commitment: Some(prev_commitment),
        })
    }

    /// Generate proof about a provenance chain
    ///
    /// Proves properties of the provenance chain without revealing
    /// the actual knowledge sources or derivation details.
    pub fn prove_provenance(
        &self,
        chain: &ProvenanceChain,
        kvector: &KVector,
        statement: &ProofStatement,
        blinding: &[u8; 32],
    ) -> Result<TrustProof, ProofError> {
        // Evaluate the provenance statement
        let result = evaluate_provenance_statement(chain, statement)?;

        // Create commitment (to K-Vector + chain head hash)
        let commitment = self.commit(kvector, blinding);

        // Generate proof data
        let proof_data = if self.config.simulation_mode {
            self.generate_simulation_proof(&commitment, statement, result)
        } else {
            #[cfg(feature = "risc0")]
            {
                return Err(ProofError::Risc0NotImplemented);
            }
            #[cfg(not(feature = "risc0"))]
            {
                return Err(ProofError::Risc0NotEnabled);
            }
        };

        Ok(TrustProof {
            statement: statement.clone(),
            commitment,
            proof_data,
            created_at: self.config.timestamp,
            previous_commitment: None,
        })
    }

    /// Generate combined proof for K-Vector and provenance
    ///
    /// Proves both trust properties AND provenance quality in a single proof.
    pub fn prove_with_provenance(
        &self,
        kvector: &KVector,
        chain: &ProvenanceChain,
        trust_statement: &ProofStatement,
        provenance_statement: &ProofStatement,
        blinding: &[u8; 32],
    ) -> Result<TrustProof, ProofError> {
        // Evaluate trust statement
        let trust_result = evaluate_statement(kvector, trust_statement, None)?;

        // Evaluate provenance statement
        let prov_result = evaluate_provenance_statement(chain, provenance_statement)?;

        // Combined result
        let result = trust_result && prov_result;

        // Create combined statement
        let combined_statement =
            ProofStatement::And(vec![trust_statement.clone(), provenance_statement.clone()]);

        // Create commitment
        let commitment = self.commit(kvector, blinding);

        // Generate proof data
        let proof_data = if self.config.simulation_mode {
            self.generate_simulation_proof(&commitment, &combined_statement, result)
        } else {
            #[cfg(feature = "risc0")]
            {
                return Err(ProofError::Risc0NotImplemented);
            }
            #[cfg(not(feature = "risc0"))]
            {
                return Err(ProofError::Risc0NotEnabled);
            }
        };

        Ok(TrustProof {
            statement: combined_statement,
            commitment,
            proof_data,
            created_at: self.config.timestamp,
            previous_commitment: None,
        })
    }

    /// Generate simulation proof (NOT secure!)
    fn generate_simulation_proof(
        &self,
        commitment: &KVectorCommitment,
        statement: &ProofStatement,
        result: bool,
    ) -> ProofData {
        #[cfg(feature = "std")]
        {
            eprintln!("WARNING: Using simulation proof - NOT cryptographically secure!");
        }

        // Create verification hash
        let mut hasher = Sha3_256::new();
        hasher.update(b"mycelix-simulation-proof-v1");
        hasher.update(commitment.commitment);
        hasher.update(bincode::serialize(statement).unwrap_or_default());
        hasher.update([result as u8]);
        let verification_hash = hasher.finalize().into();

        ProofData::Simulation {
            verification_hash,
            result,
        }
    }

    /// Generate RISC-0 proof (placeholder - real implementation would use zkVM)
    #[cfg(feature = "risc0")]
    fn generate_risc0_proof(
        &self,
        _kvector: &KVector,
        _statement: &ProofStatement,
    ) -> Result<ProofData, ProofError> {
        // In a real implementation, this would:
        // 1. Serialize the K-Vector and statement
        // 2. Run the RISC-0 zkVM prover
        // 3. Return the receipt
        Err(ProofError::Risc0NotImplemented)
    }
}

// ============================================================================
// Verifier
// ============================================================================

/// Verifies zero-knowledge proofs without K-Vector access
pub struct TrustVerifier {
    /// Known commitments for cross-referencing
    known_commitments: HashMap<[u8; 32], KVectorCommitment>,
}

impl TrustVerifier {
    /// Create a new verifier
    pub fn new() -> Self {
        Self {
            known_commitments: HashMap::new(),
        }
    }

    /// Register a known commitment for future verification
    pub fn register_commitment(&mut self, commitment: KVectorCommitment) {
        self.known_commitments
            .insert(commitment.commitment, commitment);
    }

    /// Verify a proof
    pub fn verify(&self, proof: &TrustProof) -> VerificationResult {
        match &proof.proof_data {
            ProofData::Simulation {
                verification_hash,
                result,
            } => {
                // Recompute verification hash
                let mut hasher = Sha3_256::new();
                hasher.update(b"mycelix-simulation-proof-v1");
                hasher.update(proof.commitment.commitment);
                hasher.update(bincode::serialize(&proof.statement).unwrap_or_default());
                hasher.update([*result as u8]);
                let expected_hash: [u8; 32] = hasher.finalize().into();

                if !constant_time_eq(verification_hash, &expected_hash) {
                    return VerificationResult::Invalid(VerificationError::HashMismatch);
                }

                // Check improvement proofs reference known commitments
                if let Some(prev) = &proof.previous_commitment {
                    if !self.known_commitments.contains_key(&prev.commitment) {
                        return VerificationResult::Valid {
                            statement_result: *result,
                            warning: Some("Previous commitment not registered".to_string()),
                        };
                    }
                }

                VerificationResult::Valid {
                    statement_result: *result,
                    warning: Some("Simulation proof - not cryptographically secure".to_string()),
                }
            }

            #[cfg(feature = "risc0")]
            ProofData::Risc0 { receipt, image_id } => {
                // In a real implementation, this would verify the RISC-0 receipt
                VerificationResult::Invalid(VerificationError::Risc0NotImplemented)
            }

            #[cfg(not(feature = "risc0"))]
            ProofData::Risc0Placeholder => {
                VerificationResult::Invalid(VerificationError::Risc0NotEnabled)
            }
        }
    }

    /// Verify and extract result
    pub fn verify_and_get_result(&self, proof: &TrustProof) -> Option<bool> {
        match self.verify(proof) {
            VerificationResult::Valid {
                statement_result, ..
            } => Some(statement_result),
            _ => None,
        }
    }
}

impl Default for TrustVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of proof verification
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VerificationResult {
    /// Proof is valid
    Valid {
        /// Result of the proven statement
        statement_result: bool,
        /// Optional warning (e.g., simulation mode)
        warning: Option<String>,
    },
    /// Proof is invalid
    Invalid(VerificationError),
}

impl VerificationResult {
    /// Check if verification succeeded and statement was true
    pub fn is_valid_and_true(&self) -> bool {
        matches!(
            self,
            Self::Valid {
                statement_result: true,
                ..
            }
        )
    }

    /// Check if verification succeeded (regardless of statement result)
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::Valid { .. })
    }
}

/// Verification errors
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum VerificationError {
    /// Hash doesn't match
    HashMismatch,
    /// RISC-0 not implemented
    Risc0NotImplemented,
    /// RISC-0 feature not enabled
    Risc0NotEnabled,
    /// Invalid proof format
    InvalidFormat,
    /// Previous commitment not found
    PreviousCommitmentNotFound,
}

// ============================================================================
// Proof Errors
// ============================================================================

/// Errors during proof generation
#[derive(Clone, Debug)]
pub enum ProofError {
    /// Statement cannot be evaluated
    InvalidStatement(String),
    /// RISC-0 feature not enabled
    Risc0NotEnabled,
    /// RISC-0 proof generation not implemented
    Risc0NotImplemented,
    /// Internal error
    Internal(String),
}

// ============================================================================
// Statement Evaluation
// ============================================================================

/// Evaluate a proof statement against a K-Vector
fn evaluate_statement(
    kvector: &KVector,
    statement: &ProofStatement,
    previous: Option<&KVector>,
) -> Result<bool, ProofError> {
    match statement {
        ProofStatement::TrustExceedsThreshold { threshold } => {
            Ok(kvector.trust_score() >= *threshold)
        }

        ProofStatement::TrustInRange { lower, upper } => {
            let score = kvector.trust_score();
            Ok(score >= *lower && score <= *upper)
        }

        ProofStatement::DimensionExceedsThreshold { dimension, threshold } => {
            let val = get_dimension(kvector, *dimension);
            Ok(val >= *threshold)
        }

        ProofStatement::MultipleDimensionsExceed { requirements } => {
            for (dim, threshold) in requirements {
                if get_dimension(kvector, *dim) < *threshold {
                    return Ok(false);
                }
            }
            Ok(true)
        }

        ProofStatement::MeetsGovernanceTier { tier } => {
            Ok(kvector.meets_governance_threshold(*tier))
        }

        ProofStatement::WellFormed => {
            let arr = kvector.to_array();
            Ok(arr.iter().all(|v| *v >= 0.0 && *v <= 1.0))
        }

        ProofStatement::TrustImproved { .. } => {
            match previous {
                Some(prev) => Ok(kvector.trust_score() > prev.trust_score()),
                None => Err(ProofError::InvalidStatement(
                    "Improvement proof requires previous K-Vector".to_string(),
                )),
            }
        }

        ProofStatement::DimensionImproved { dimension, .. } => {
            match previous {
                Some(prev) => {
                    let curr = get_dimension(kvector, *dimension);
                    let prev_val = get_dimension(prev, *dimension);
                    Ok(curr > prev_val)
                }
                None => Err(ProofError::InvalidStatement(
                    "Improvement proof requires previous K-Vector".to_string(),
                )),
            }
        }

        ProofStatement::IsVerified => {
            Ok(kvector.is_verified())
        }

        ProofStatement::IsStronglyVerified => {
            Ok(kvector.is_strongly_verified())
        }

        ProofStatement::And(statements) => {
            for s in statements {
                if !evaluate_statement(kvector, s, previous)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }

        ProofStatement::Or(statements) => {
            for s in statements {
                if evaluate_statement(kvector, s, previous)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }

        // Provenance statements require ProvenanceChain context
        // Use evaluate_provenance_statement instead
        ProofStatement::ProvenanceEpistemicFloor { .. }
        | ProofStatement::ProvenanceChainDepth { .. }
        | ProofStatement::ProvenanceConfidence { .. }
        | ProofStatement::ProvenanceDerivationTypes { .. }
        | ProofStatement::ProvenanceHasVerifiedRoots
        | ProofStatement::ProvenanceChainValid
        | ProofStatement::ProvenanceQuality { .. } => {
            Err(ProofError::InvalidStatement(
                "Provenance statements require ProvenanceChain context. Use evaluate_provenance_statement.".to_string(),
            ))
        }
    }
}

/// Evaluate a provenance-related proof statement against a ProvenanceChain
pub fn evaluate_provenance_statement(
    chain: &ProvenanceChain,
    statement: &ProofStatement,
) -> Result<bool, ProofError> {
    match statement {
        ProofStatement::ProvenanceEpistemicFloor { min_level } => {
            // Check that the epistemic floor meets minimum level
            let floor_level = chain.head.epistemic_floor as u8;
            Ok(floor_level >= *min_level)
        }

        ProofStatement::ProvenanceChainDepth { max_depth } => {
            // Check chain depth doesn't exceed maximum
            Ok(chain.max_depth <= *max_depth)
        }

        ProofStatement::ProvenanceConfidence { min_confidence } => {
            // Check derived confidence meets minimum
            Ok(chain.head.derived_confidence >= *min_confidence)
        }

        ProofStatement::ProvenanceDerivationTypes { allowed_types } => {
            // Check all derivation types in chain are allowed
            for node in chain.nodes.values() {
                let derivation_type = node.derivation_type as u8;
                if !allowed_types.contains(&derivation_type) {
                    return Ok(false);
                }
            }
            Ok(true)
        }

        ProofStatement::ProvenanceHasVerifiedRoots => {
            // Check all root nodes have Original derivation type
            for root_id in &chain.roots {
                if let Some(node) = chain.nodes.get(root_id) {
                    if node.derivation_type != DerivationType::Original {
                        return Ok(false);
                    }
                } else {
                    return Ok(false); // Missing root
                }
            }
            Ok(!chain.roots.is_empty())
        }

        ProofStatement::ProvenanceChainValid => {
            // Check chain integrity
            Ok(chain.verified)
        }

        ProofStatement::ProvenanceQuality {
            min_epistemic_level,
            max_chain_depth,
            min_confidence,
        } => {
            // Combined quality check
            let floor_ok = chain.head.epistemic_floor as u8 >= *min_epistemic_level;
            let depth_ok = chain.max_depth <= *max_chain_depth;
            let conf_ok = chain.head.derived_confidence >= *min_confidence;
            Ok(floor_ok && depth_ok && conf_ok)
        }

        // Non-provenance statements should use evaluate_statement
        _ => Err(ProofError::InvalidStatement(
            "Non-provenance statement passed to evaluate_provenance_statement".to_string(),
        )),
    }
}

/// Get a dimension value from K-Vector
fn get_dimension(kvector: &KVector, dimension: KVectorDimension) -> f32 {
    match dimension {
        KVectorDimension::Reputation => kvector.k_r,
        KVectorDimension::Activity => kvector.k_a,
        KVectorDimension::Integrity => kvector.k_i,
        KVectorDimension::Performance => kvector.k_p,
        KVectorDimension::Membership => kvector.k_m,
        KVectorDimension::Stake => kvector.k_s,
        KVectorDimension::Historical => kvector.k_h,
        KVectorDimension::Topology => kvector.k_topo,
        KVectorDimension::Verification => kvector.k_v,
        KVectorDimension::Coherence => kvector.k_coherence,
    }
}

// ============================================================================
// Proof Aggregation
// ============================================================================

/// Aggregated proof from multiple agents
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AggregatedTrustProof {
    /// Individual proofs
    pub proofs: Vec<TrustProof>,

    /// Aggregated statement (what was collectively proven)
    pub aggregate_statement: AggregateStatement,

    /// Number of agents who proved the statement true
    pub true_count: usize,

    /// Total number of agents
    pub total_count: usize,

    /// Weighted result (using trust scores as weights, if available)
    pub weighted_result: Option<f64>,
}

/// Aggregate statements over multiple agents
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AggregateStatement {
    /// All agents prove the same statement
    AllProve(
        /// Statement that all agents must prove
        ProofStatement,
    ),

    /// Threshold of agents must prove true
    ThresholdProve {
        /// Statement to prove
        statement: ProofStatement,
        /// Minimum fraction that must prove true
        threshold: f64,
    },

    /// Weighted threshold (by trust)
    WeightedThreshold {
        /// Statement to prove with weight
        statement: ProofStatement,
        /// Minimum weighted fraction
        threshold: f64,
    },
}

impl AggregatedTrustProof {
    /// Check if the aggregate statement is satisfied
    pub fn is_satisfied(&self) -> bool {
        match &self.aggregate_statement {
            AggregateStatement::AllProve(_) => self.true_count == self.total_count,
            AggregateStatement::ThresholdProve { threshold, .. } => {
                let fraction = self.true_count as f64 / self.total_count as f64;
                fraction >= *threshold
            }
            AggregateStatement::WeightedThreshold { threshold, .. } => self
                .weighted_result
                .map(|w| w >= *threshold)
                .unwrap_or(false),
        }
    }
}

/// Aggregate proofs from multiple agents
pub fn aggregate_proofs(
    proofs: Vec<TrustProof>,
    statement: AggregateStatement,
    weights: Option<&[f64]>,
) -> AggregatedTrustProof {
    let total_count = proofs.len();
    let mut true_count = 0;
    let mut weighted_sum = 0.0;
    let mut weight_total = 0.0;

    for (i, proof) in proofs.iter().enumerate() {
        let result = proof.get_result().unwrap_or(false);
        if result {
            true_count += 1;
        }

        if let Some(w) = weights {
            let weight = w.get(i).copied().unwrap_or(1.0);
            weight_total += weight;
            if result {
                weighted_sum += weight;
            }
        }
    }

    let weighted_result = if weight_total > 0.0 {
        Some(weighted_sum / weight_total)
    } else {
        None
    };

    AggregatedTrustProof {
        proofs,
        aggregate_statement: statement,
        true_count,
        total_count,
        weighted_result,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epistemic::EmpiricalLevel;

    fn test_kvector() -> KVector {
        KVector::new(0.8, 0.6, 0.9, 0.7, 0.5, 0.4, 0.6, 0.5, 0.75, 0.7)
    }

    fn test_blinding() -> [u8; 32] {
        let mut b = [0u8; 32];
        b[0] = 42;
        b[31] = 7;
        b
    }

    #[test]
    fn test_commitment_creation() {
        let kv = test_kvector();
        let blinding = test_blinding();

        let commitment = KVectorCommitment::create(&kv, "agent-1", 1000, &blinding);

        assert_eq!(commitment.agent_id, "agent-1");
        assert_eq!(commitment.timestamp, 1000);
        assert!(commitment.verify_opening(&kv, &blinding));
    }

    #[test]
    fn test_commitment_tamper_detection() {
        let kv = test_kvector();
        let blinding = test_blinding();

        let commitment = KVectorCommitment::create(&kv, "agent-1", 1000, &blinding);

        // Try to verify with different K-Vector
        let tampered = KVector::new(0.9, 0.6, 0.9, 0.7, 0.5, 0.4, 0.6, 0.5, 0.75, 0.7);
        assert!(!commitment.verify_opening(&tampered, &blinding));

        // Try to verify with different blinding
        let wrong_blinding = [1u8; 32];
        assert!(!commitment.verify_opening(&kv, &wrong_blinding));
    }

    #[test]
    fn test_simple_proof() {
        let kv = test_kvector();
        let blinding = test_blinding();

        let config = ProverConfig {
            simulation_mode: true,
            agent_id: "agent-1".to_string(),
            timestamp: 1000,
        };

        let prover = TrustProver::new(config);
        let statement = ProofStatement::trust_above(0.5);

        let proof = prover.prove(&kv, &statement, &blinding).unwrap();

        // Verify
        let verifier = TrustVerifier::new();
        let result = verifier.verify(&proof);

        assert!(result.is_valid_and_true());
    }

    #[test]
    fn test_failing_proof() {
        let kv = test_kvector(); // Trust score ~0.68
        let blinding = test_blinding();

        let config = ProverConfig {
            simulation_mode: true,
            agent_id: "agent-1".to_string(),
            timestamp: 1000,
        };

        let prover = TrustProver::new(config);
        let statement = ProofStatement::trust_above(0.9); // Higher than actual

        let proof = prover.prove(&kv, &statement, &blinding).unwrap();

        // Verify - should be valid proof of false statement
        let verifier = TrustVerifier::new();
        let result = verifier.verify(&proof);

        assert!(result.is_valid());
        assert!(!result.is_valid_and_true()); // Statement is false
    }

    #[test]
    fn test_dimension_proof() {
        let kv = test_kvector(); // k_i = 0.9
        let blinding = test_blinding();

        let config = ProverConfig {
            simulation_mode: true,
            agent_id: "agent-1".to_string(),
            timestamp: 1000,
        };

        let prover = TrustProver::new(config);
        let statement = ProofStatement::dimension_above(KVectorDimension::Integrity, 0.8);

        let proof = prover.prove(&kv, &statement, &blinding).unwrap();

        let verifier = TrustVerifier::new();
        assert!(verifier.verify(&proof).is_valid_and_true());
    }

    #[test]
    fn test_governance_tier_proof() {
        let kv = test_kvector(); // Trust ~0.68
        let blinding = test_blinding();

        let config = ProverConfig {
            simulation_mode: true,
            agent_id: "agent-1".to_string(),
            timestamp: 1000,
        };

        let prover = TrustProver::new(config);

        // Should meet Major tier (0.4)
        let proof = prover
            .prove(
                &kv,
                &ProofStatement::meets_tier(GovernanceTier::Major),
                &blinding,
            )
            .unwrap();
        assert!(TrustVerifier::new().verify(&proof).is_valid_and_true());

        // Should NOT meet Constitutional tier (0.6)
        let proof = prover
            .prove(
                &kv,
                &ProofStatement::meets_tier(GovernanceTier::Constitutional),
                &blinding,
            )
            .unwrap();
        // Actually ~0.68 should meet 0.6 threshold
        assert!(TrustVerifier::new().verify(&proof).is_valid_and_true());
    }

    #[test]
    fn test_compound_proof() {
        let kv = test_kvector();
        let blinding = test_blinding();

        let config = ProverConfig {
            simulation_mode: true,
            agent_id: "agent-1".to_string(),
            timestamp: 1000,
        };

        let prover = TrustProver::new(config);

        // AND: trust > 0.5 AND integrity > 0.8
        let statement = ProofStatement::all(vec![
            ProofStatement::trust_above(0.5),
            ProofStatement::dimension_above(KVectorDimension::Integrity, 0.8),
        ]);

        let proof = prover.prove(&kv, &statement, &blinding).unwrap();
        assert!(TrustVerifier::new().verify(&proof).is_valid_and_true());

        // AND: trust > 0.5 AND integrity > 0.95 (should fail)
        let statement = ProofStatement::all(vec![
            ProofStatement::trust_above(0.5),
            ProofStatement::dimension_above(KVectorDimension::Integrity, 0.95),
        ]);

        let proof = prover.prove(&kv, &statement, &blinding).unwrap();
        assert!(!TrustVerifier::new().verify(&proof).is_valid_and_true());
    }

    #[test]
    fn test_or_proof() {
        let kv = test_kvector();
        let blinding = test_blinding();

        let config = ProverConfig {
            simulation_mode: true,
            agent_id: "agent-1".to_string(),
            timestamp: 1000,
        };

        let prover = TrustProver::new(config);

        // OR: trust > 0.9 OR integrity > 0.8 (second is true)
        let statement = ProofStatement::any(vec![
            ProofStatement::trust_above(0.9),
            ProofStatement::dimension_above(KVectorDimension::Integrity, 0.8),
        ]);

        let proof = prover.prove(&kv, &statement, &blinding).unwrap();
        assert!(TrustVerifier::new().verify(&proof).is_valid_and_true());
    }

    #[test]
    fn test_improvement_proof() {
        let prev_kv = KVector::new(0.5, 0.4, 0.6, 0.5, 0.3, 0.2, 0.4, 0.3, 0.5, 0.5);
        let curr_kv = KVector::new(0.8, 0.6, 0.9, 0.7, 0.5, 0.4, 0.6, 0.5, 0.75, 0.7);
        let prev_blinding = [1u8; 32];
        let curr_blinding = [2u8; 32];

        let config = ProverConfig {
            simulation_mode: true,
            agent_id: "agent-1".to_string(),
            timestamp: 2000,
        };

        let prover = TrustProver::new(config);

        let proof = prover
            .prove_improvement(
                &prev_kv,
                &curr_kv,
                None, // Overall trust
                &prev_blinding,
                &curr_blinding,
            )
            .unwrap();

        let verifier = TrustVerifier::new();
        assert!(verifier.verify(&proof).is_valid_and_true());
    }

    #[test]
    fn test_verification_proof() {
        let verified_kv = KVector::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.5);
        let unverified_kv = KVector::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.5);
        let blinding = test_blinding();

        let config = ProverConfig {
            simulation_mode: true,
            agent_id: "agent-1".to_string(),
            timestamp: 1000,
        };

        let prover = TrustProver::new(config);

        // Verified agent
        let proof = prover
            .prove(&verified_kv, &ProofStatement::IsVerified, &blinding)
            .unwrap();
        assert!(TrustVerifier::new().verify(&proof).is_valid_and_true());

        // Unverified agent
        let proof = prover
            .prove(&unverified_kv, &ProofStatement::IsVerified, &blinding)
            .unwrap();
        assert!(!TrustVerifier::new().verify(&proof).is_valid_and_true());
    }

    #[test]
    fn test_aggregate_proofs() {
        let config = ProverConfig {
            simulation_mode: true,
            agent_id: "agent-1".to_string(),
            timestamp: 1000,
        };
        let prover = TrustProver::new(config);

        let statement = ProofStatement::trust_above(0.5);
        let blinding = test_blinding();

        // Create 3 proofs: 2 passing, 1 failing
        let kv_high = KVector::new(0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8);
        let kv_low = KVector::new(0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2);

        let proofs = vec![
            prover.prove(&kv_high, &statement, &blinding).unwrap(),
            prover.prove(&kv_high, &statement, &blinding).unwrap(),
            prover.prove(&kv_low, &statement, &blinding).unwrap(),
        ];

        // Threshold: 50% must pass
        let aggregate = aggregate_proofs(
            proofs,
            AggregateStatement::ThresholdProve {
                statement: statement.clone(),
                threshold: 0.5,
            },
            None,
        );

        assert_eq!(aggregate.true_count, 2);
        assert_eq!(aggregate.total_count, 3);
        assert!(aggregate.is_satisfied()); // 2/3 > 0.5
    }

    #[test]
    fn test_well_formed_proof() {
        let kv = test_kvector();
        let blinding = test_blinding();

        let config = ProverConfig {
            simulation_mode: true,
            agent_id: "agent-1".to_string(),
            timestamp: 1000,
        };

        let prover = TrustProver::new(config);
        let proof = prover
            .prove(&kv, &ProofStatement::WellFormed, &blinding)
            .unwrap();

        assert!(TrustVerifier::new().verify(&proof).is_valid_and_true());
    }

    // =========================================================================
    // Provenance Proof Tests (Option C)
    // =========================================================================

    use super::super::provenance::{ChainBuilder, ProvenanceBuilder};

    fn create_test_chain() -> ProvenanceChain {
        // Create a simple chain: root -> derived
        let root = ProvenanceBuilder::original(b"Original source", "agent-1")
            .confidence(0.95)
            .epistemic(EmpiricalLevel::E3Cryptographic)
            .build(1000);

        let derived =
            ProvenanceBuilder::derived(b"Derived insight", "agent-2", DerivationType::Citation)
                .parent(root.clone())
                .confidence(0.9)
                .epistemic(EmpiricalLevel::E3Cryptographic)
                .build(2000);

        ChainBuilder::new().add_node(root).build(derived)
    }

    #[test]
    fn test_provenance_epistemic_floor_proof() {
        let chain = create_test_chain();
        let statement = ProofStatement::provenance_epistemic_floor(2); // E2 or higher

        let result = evaluate_provenance_statement(&chain, &statement).unwrap();
        assert!(result); // E3 >= E2
    }

    #[test]
    fn test_provenance_chain_depth_proof() {
        let chain = create_test_chain();

        // Depth 1 should pass (max_depth = 1)
        let statement = ProofStatement::provenance_depth(2);
        let result = evaluate_provenance_statement(&chain, &statement).unwrap();
        assert!(result);

        // Depth 0 should fail
        let statement = ProofStatement::provenance_depth(0);
        let result = evaluate_provenance_statement(&chain, &statement).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_provenance_confidence_proof() {
        let chain = create_test_chain();

        // High confidence requirement
        let statement = ProofStatement::provenance_confidence(0.8);
        let result = evaluate_provenance_statement(&chain, &statement).unwrap();
        assert!(result);

        // Too high requirement
        let statement = ProofStatement::provenance_confidence(0.99);
        let result = evaluate_provenance_statement(&chain, &statement).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_provenance_derivation_types_proof() {
        let chain = create_test_chain();

        // Allow Original (0) and Citation (1)
        let statement = ProofStatement::ProvenanceDerivationTypes {
            allowed_types: vec![0, 1], // Original, Citation
        };
        let result = evaluate_provenance_statement(&chain, &statement).unwrap();
        assert!(result);

        // Only allow Original - should fail (has Citation)
        let statement = ProofStatement::ProvenanceDerivationTypes {
            allowed_types: vec![0], // Only Original
        };
        let result = evaluate_provenance_statement(&chain, &statement).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_provenance_has_verified_roots() {
        let chain = create_test_chain();

        let statement = ProofStatement::ProvenanceHasVerifiedRoots;
        let result = evaluate_provenance_statement(&chain, &statement).unwrap();
        assert!(result);
    }

    #[test]
    fn test_provenance_chain_valid() {
        let chain = create_test_chain();

        let statement = ProofStatement::ProvenanceChainValid;
        let result = evaluate_provenance_statement(&chain, &statement).unwrap();
        assert!(result);
    }

    #[test]
    fn test_provenance_quality_combined() {
        let chain = create_test_chain();

        // Should pass all quality checks
        let statement = ProofStatement::provenance_quality(2, 5, 0.7);
        let result = evaluate_provenance_statement(&chain, &statement).unwrap();
        assert!(result);

        // Should fail due to too strict epistemic requirement
        let statement = ProofStatement::provenance_quality(4, 5, 0.7);
        let result = evaluate_provenance_statement(&chain, &statement).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_prove_provenance() {
        let chain = create_test_chain();
        let kv = test_kvector();
        let blinding = test_blinding();

        let config = ProverConfig {
            simulation_mode: true,
            agent_id: "agent-1".to_string(),
            timestamp: 1000,
        };
        let prover = TrustProver::new(config);

        let statement = ProofStatement::provenance_epistemic_floor(2);
        let proof = prover
            .prove_provenance(&chain, &kv, &statement, &blinding)
            .unwrap();

        let verifier = TrustVerifier::new();
        assert!(verifier.verify(&proof).is_valid_and_true());
    }

    #[test]
    fn test_prove_with_provenance_combined() {
        let chain = create_test_chain();
        let kv = test_kvector();
        let blinding = test_blinding();

        let config = ProverConfig {
            simulation_mode: true,
            agent_id: "agent-1".to_string(),
            timestamp: 1000,
        };
        let prover = TrustProver::new(config);

        let trust_statement = ProofStatement::trust_above(0.5);
        let prov_statement = ProofStatement::provenance_quality(2, 5, 0.7);

        let proof = prover
            .prove_with_provenance(&kv, &chain, &trust_statement, &prov_statement, &blinding)
            .unwrap();

        let verifier = TrustVerifier::new();
        assert!(verifier.verify(&proof).is_valid_and_true());
    }
}
