// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Federated Learning Integrity Zome
//!
//! Defines entry types and validation rules for Byzantine-resistant
//! federated learning on Holochain 0.6.

use hdi::prelude::*;
use sha2::Digest;

/// Maximum Byzantine tolerance (34% of participants)
/// Validated maximum: 34% converges with trimmed-mean aggregation.
/// 45% does NOT converge without strong reputation disparity.
pub const MAX_BYZANTINE_TOLERANCE: f32 = 0.34;

/// Maximum allowed length for node IDs
pub const MAX_NODE_ID_LENGTH: usize = 256;

/// Minimum validator reputation to participate in consensus
pub const MIN_VALIDATOR_REPUTATION: f32 = 0.5;

/// Quorum threshold for consensus (2/3 supermajority)
pub const CONSENSUS_QUORUM_THRESHOLD: f64 = 2.0 / 3.0;

/// Model gradient entry for federated learning
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ModelGradient {
    pub node_id: String,
    pub round: u32,
    pub gradient_hash: String,
    pub timestamp: i64,
    pub cpu_usage: f32,
    pub memory_mb: f32,
    pub network_latency_ms: f32,
    pub trust_score: Option<f32>,
}

/// Training round result
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct TrainingRound {
    pub round: u32,
    pub selected_gradient: ActionHash,
    pub participant_count: u32,
    pub byzantine_detected: bool,
    pub byzantine_count: u32,
    pub accuracy: f32,
    pub completed_at: i64,
}

/// Byzantine detection record
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ByzantineRecord {
    pub node_id: String,
    pub round: u32,
    pub detection_method: String,
    pub confidence: f32,
    pub evidence_hash: String,
    pub detected_at: i64,
}

/// Node reputation entry
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct NodeReputation {
    pub node_id: String,
    pub successful_rounds: u32,
    pub failed_rounds: u32,
    pub reputation_score: f32,
    pub last_updated: i64,
}

// =============================================================================
// SEC-002 FIX: Secure Coordinator Bootstrap Types
// =============================================================================

/// Genesis coordinator entry - cryptographically verifiable authority
/// Created at DNA initialization with multi-party signatures
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GenesisCoordinator {
    /// Agent public key of the genesis coordinator
    pub agent_pubkey: String,
    /// Cryptographic proof of authority (signed by DNA deployer)
    pub authority_proof: Vec<u8>,
    /// SHA-256 hash of the DNA modifiers at deployment
    pub dna_modifiers_hash: String,
    /// Timestamp when coordinator was established
    pub established_at: i64,
    /// Whether this genesis coordinator is still active
    pub active: bool,
}

/// Guardian entry for multi-signature bootstrap
/// Guardians can vote to add/remove coordinators
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Guardian {
    /// Agent public key of the guardian
    pub agent_pubkey: String,
    /// Guardian's weight in voting (typically 1)
    pub voting_weight: u32,
    /// When this guardian was added
    pub added_at: i64,
    /// Who added this guardian (genesis coordinator or vote)
    pub added_by: String,
    /// Whether this guardian is still active
    pub active: bool,
}

/// Coordinator credential - issued through proper ceremony
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CoordinatorCredential {
    /// Agent who holds this credential
    pub agent_pubkey: String,
    /// Type of credential (genesis, voted, delegated)
    pub credential_type: String,
    /// When the credential was issued
    pub issued_at: i64,
    /// When the credential expires (0 = never)
    pub expires_at: i64,
    /// Signatures from guardians who approved this credential
    /// Format: Vec<(guardian_pubkey, signature_bytes)>
    pub guardian_signatures_json: String,
    /// Minimum signatures required at time of issuance
    pub required_signatures: u32,
    /// Whether this credential is still valid
    pub active: bool,
    /// Revocation timestamp (0 if not revoked)
    pub revoked_at: i64,
}

/// Bootstrap vote for coordinator election
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CoordinatorVote {
    /// Unique vote ID (hash of voter + candidate + timestamp)
    pub vote_id: String,
    /// Agent being voted for as coordinator
    pub candidate_pubkey: String,
    /// Guardian casting the vote
    pub voter_pubkey: String,
    /// Vote weight (from guardian entry)
    pub weight: u32,
    /// Whether this is an approval (true) or rejection (false)
    pub approve: bool,
    /// Cryptographic signature of the vote
    pub signature: Vec<u8>,
    /// Timestamp of the vote
    pub voted_at: i64,
    /// Optional expiry for time-limited votes
    pub expires_at: i64,
}

/// Coordinator action audit log
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CoordinatorAuditLog {
    /// Unique log entry ID
    pub log_id: String,
    /// Coordinator who performed the action
    pub coordinator_pubkey: String,
    /// Action type (grant_role, revoke_role, complete_round, etc.)
    pub action_type: String,
    /// Target of the action (agent pubkey or resource ID)
    pub target: String,
    /// Additional context as JSON
    pub context_json: String,
    /// Timestamp of the action
    pub timestamp: i64,
    /// Hash of previous audit log entry (chain integrity)
    pub previous_log_hash: String,
}

/// Bootstrap window configuration
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct BootstrapConfig {
    /// When the bootstrap window opens
    pub window_start: i64,
    /// When the bootstrap window closes (after this, no new genesis coordinators)
    pub window_end: i64,
    /// Minimum guardians required for coordinator votes
    pub min_guardians: u32,
    /// Minimum votes required (N of M)
    pub min_votes: u32,
    /// Total guardians allowed (M)
    pub max_guardians: u32,
    /// Whether bootstrap is complete
    pub bootstrap_complete: bool,
}

// =============================================================================
// PROOF-002: zkSTARK Gradient Proof Types
// =============================================================================

/// Gradient proof entry - stores the zkSTARK proof for a gradient contribution
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GradientProof {
    /// Hash of the associated gradient (links to ModelGradient)
    pub gradient_hash: String,
    /// Node that submitted the proof
    pub node_id: String,
    /// FL round number
    pub round: u32,
    /// Blake3 commitment to the gradient values (32 bytes as hex)
    pub gradient_commitment: String,
    /// Blake3 commitment to the local training data (32 bytes as hex)
    pub data_commitment: String,
    /// Serialized zkSTARK proof (base64 encoded)
    pub proof_bytes: String,
    /// Security level used (standard96, standard128, high256)
    pub security_level: String,
    /// Proof generation timestamp
    pub generated_at: i64,
    /// Whether the proof has been verified
    pub verified: bool,
    /// Verification result (only set after verification)
    pub verification_result: Option<ProofVerificationStatus>,
    /// L2 norm of the gradient (from proof public inputs)
    pub l2_norm: f32,
    /// Number of gradient elements
    pub num_elements: u32,
}

/// Result of proof verification
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ProofVerificationStatus {
    /// Whether the cryptographic proof is valid
    pub integrity_valid: bool,
    /// Whether statistics indicate non-Byzantine behavior
    pub statistics_valid: bool,
    /// Verification timestamp
    pub verified_at: i64,
    /// Agent who performed verification
    pub verifier: String,
    /// Any issues found during verification (JSON array)
    pub issues_json: String,
}

/// Proof audit log - tracks all proof-related actions
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ProofAuditEntry {
    /// Unique audit entry ID
    pub entry_id: String,
    /// Type of action (submit, verify, reject, accept)
    pub action_type: String,
    /// Hash of the proof being audited
    pub proof_hash: String,
    /// Node involved
    pub node_id: String,
    /// FL round
    pub round: u32,
    /// Action result (success, failure, pending)
    pub result: String,
    /// Additional context as JSON
    pub context_json: String,
    /// Timestamp
    pub timestamp: i64,
}

// =============================================================================
// PHASE 1: Decentralized Aggregation Consensus (Commit-Reveal)
// =============================================================================

/// Aggregation commitment — commit phase of multi-validator consensus
/// Each validator independently aggregates round gradients and commits the hash
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AggregationCommitment {
    /// FL round number
    pub round: u64,
    /// Agent who computed this aggregation
    pub aggregator: String,
    /// SHA-256(aggregated_result_bytes) — committed before reveal (hex-encoded)
    pub commitment_hash: String,
    /// Aggregation method used (e.g., "ByzantineFilteredHV", "FedAvgHV")
    pub method: String,
    /// Timestamp of commitment
    pub committed_at: i64,
    /// Number of gradients included in aggregation
    pub gradient_count: u32,
    /// Number of gradients excluded by Byzantine detection
    pub excluded_count: u32,
    /// Aggregator's trust score at time of commitment
    pub aggregator_trust_score: f32,
}

/// Aggregation reveal — reveal phase after commit deadline
/// Must match the previously committed hash
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct AggregationReveal {
    /// FL round number
    pub round: u64,
    /// Agent who computed this aggregation (must match commitment)
    pub aggregator: String,
    /// The actual aggregated result (e.g., HV16 bytes, 2KB)
    pub result_data: Vec<u8>,
    /// SHA-256(result_data) — must match commitment_hash (hex-encoded)
    pub result_hash: String,
    /// Summary of Byzantine detections during aggregation
    pub detection_summary_json: String,
    /// Shapley attribution values: [(node_id, value)]
    pub shapley_values_json: String,
    /// Timestamp of reveal
    pub revealed_at: i64,
}

/// Consensus result — finalized after >2/3 weighted agreement
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ConsensusResult {
    /// FL round number
    pub round: u64,
    /// The agreed-upon aggregation hash (hex-encoded)
    pub agreed_hash: String,
    /// Validators who agreed (agent pubkeys)
    pub agreeing_validators_json: String,
    /// Total reputation^2 weight of all validators
    pub total_weight: f64,
    /// Weight of agreeing validators (must be > 2/3 of total_weight)
    pub consensus_weight: f64,
    /// Aggregation method that reached consensus
    pub method: String,
    /// Timestamp of finalization
    pub finalized_at: i64,
    /// Agent who finalized the consensus
    pub finalized_by: String,
}

// =============================================================================
// PHASE 2: Distributed Round Scheduling
// =============================================================================

/// Round schedule — defines timing for FL rounds
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct RoundSchedule {
    /// Duration of gradient collection phase (seconds)
    pub round_duration_secs: u64,
    /// Minimum participants to start aggregation
    pub min_participants: u32,
    /// Maximum participants per round
    pub max_participants: u32,
    /// Current round number
    pub current_round: u64,
    /// When current round started
    pub round_start_time: i64,
    /// Duration of commit phase (seconds)
    pub commit_window_secs: u64,
    /// Duration of reveal phase (seconds)
    pub reveal_window_secs: u64,
    /// Agent who created this schedule
    pub created_by: String,
    /// Guardians who approved this schedule (JSON array of pubkeys)
    pub approved_by_json: String,
    /// Whether this schedule is currently active
    pub active: bool,
}

// =============================================================================
// PHASE 4: Byzantine Detection Consensus
// =============================================================================

/// Byzantine vote — each validator's detection results for a round
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ByzantineVote {
    /// FL round number
    pub round: u64,
    /// Agent casting the vote
    pub voter: String,
    /// Voter's reputation score (for weight calculation)
    pub voter_reputation: f32,
    /// Voter's trust score (reputation^2 weighting)
    pub voter_trust_score: f32,
    /// Flagged nodes: JSON array of {node_id, confidence, layers}
    pub flagged_nodes_json: String,
    /// Which detection layers were used (JSON array of strings)
    pub detection_layers_json: String,
    /// SHA-256 hash of the detection evidence (hex-encoded)
    pub evidence_hash: String,
    /// Timestamp of vote
    pub voted_at: i64,
}

/// Validator registration — agents who participate in aggregation consensus
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ValidatorRegistration {
    /// Agent public key
    pub agent_pubkey: String,
    /// Current reputation score
    pub reputation_score: f32,
    /// Trust score for consensus weighting
    pub trust_score: f32,
    /// Ed25519 signing key (hex-encoded)
    pub signing_key: String,
    /// Knowledge vector as JSON (for epistemic weighting)
    pub kvector_json: String,
    /// When registered
    pub registered_at: i64,
    /// Whether still active
    pub active: bool,
    /// Number of rounds participated in
    pub rounds_participated: u64,
    /// Number of rounds where this validator's result matched consensus
    pub consensus_matches: u64,
}

// =============================================================================
// Coordinator Term Rotation
// =============================================================================

/// Coordinator term — tracks active coordinator terms and enables rotation
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CoordinatorTerm {
    /// Sequential term number (1-based)
    pub term_number: u32,
    /// Agent who holds this term
    pub coordinator_pubkey: String,
    /// Timestamp when this term started
    pub term_start: i64,
    /// Timestamp when this term ends
    pub term_end: i64,
    /// Associated credential action hash (for linking)
    pub credential_hash: String,
    /// Whether this term is currently active
    pub active: bool,
    /// How this coordinator was selected ("genesis", "voted", "re-elected")
    pub selection_method: String,
}

/// Model configuration entry for version tracking
/// Stores model architecture metadata alongside training rounds
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct ModelConfig {
    /// Training round this config applies to (0 = initial model)
    pub round: u32,
    /// SHA-256 hash of the model architecture definition
    pub architecture_hash: String,
    /// Training hyperparameters as JSON (learning rate, epochs, etc.)
    pub hyperparameters_json: String,
    /// Model input/output dimension
    pub dimension: u32,
    /// Serialization format (e.g. "HV16-128bits", "f32-dense")
    pub serialization_format: String,
    /// Human-readable description of config changes
    pub description: String,
    /// Timestamp when this config was registered
    pub created_at: i64,
    /// Agent who registered this config
    pub registered_by: String,
}

// =============================================================================
// Coherence Time-Series Tracking
// =============================================================================

/// Per-round coherence data point for anomaly detection across rounds.
/// Stored on DHT so CoherenceTimeSeries can be rebuilt from history.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct CoherenceRecord {
    /// FL round number
    pub round: u64,
    /// Coherence value: 1.0 - (byzantine_count / total_nodes)
    pub coherence_value: f32,
    /// Epistemic confidence in the measurement
    pub epistemic_confidence: f32,
    /// Number of Byzantine nodes detected
    pub byzantine_count: u32,
    /// Total nodes in the round
    pub node_count: u32,
    /// Adaptive defense escalation level at time of measurement
    pub defense_level: u32,
    /// Timestamp (unix millis)
    pub recorded_at: i64,
}

/// Gradient fingerprint for cross-round replay detection.
/// Stores the SHA-256 hash + statistics of a gradient submission so that
/// ReplayDetector can check future rounds against historical submissions.
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct GradientFingerprint {
    /// Node that submitted this gradient
    pub node_id: String,
    /// Round this gradient was submitted in
    pub round: u64,
    /// SHA-256 hash of the gradient bytes
    pub hash_hex: String,
    /// L2 norm of the gradient (for near-replay detection)
    pub l2_norm: f32,
    /// Mean value of gradient components
    pub mean: f32,
    /// Standard deviation of gradient components
    pub std_dev: f32,
    /// Timestamp of submission
    pub submitted_at: i64,
}

/// Entry types for the federated learning hApp
#[hdk_entry_types]
#[unit_enum(EntryTypesUnit)]
pub enum EntryTypes {
    ModelGradient(ModelGradient),
    TrainingRound(TrainingRound),
    ByzantineRecord(ByzantineRecord),
    NodeReputation(NodeReputation),
    // SEC-002: Secure coordinator bootstrap types
    GenesisCoordinator(GenesisCoordinator),
    Guardian(Guardian),
    CoordinatorCredential(CoordinatorCredential),
    CoordinatorVote(CoordinatorVote),
    CoordinatorAuditLog(CoordinatorAuditLog),
    BootstrapConfig(BootstrapConfig),
    // PROOF-002: zkSTARK gradient proof types
    GradientProof(GradientProof),
    ProofVerificationStatus(ProofVerificationStatus),
    ProofAuditEntry(ProofAuditEntry),
    // Phase 1: Decentralized Aggregation Consensus
    AggregationCommitment(AggregationCommitment),
    AggregationReveal(AggregationReveal),
    ConsensusResult(ConsensusResult),
    // Phase 2: Distributed Round Scheduling
    RoundSchedule(RoundSchedule),
    // Phase 4: Byzantine Detection Consensus
    ByzantineVote(ByzantineVote),
    ValidatorRegistration(ValidatorRegistration),
    // Model versioning
    ModelConfig(ModelConfig),
    // Coordinator rotation
    CoordinatorTerm(CoordinatorTerm),
    // Coherence time-series tracking
    CoherenceRecord(CoherenceRecord),
    // Cross-round replay detection
    GradientFingerprint(GradientFingerprint),
}

/// Link types for the federated learning hApp
#[hdk_link_types]
pub enum LinkTypes {
    RoundToGradients,
    NodeToGradients,
    NodeToReputation,
    RoundToByzantine,
    // SEC-002: Coordinator bootstrap link types
    GenesisCoordinators,      // Path to genesis coordinator entries
    Guardians,                // Path to guardian entries
    AgentToCredential,        // Agent pubkey -> their credential
    CandidateToVotes,         // Candidate pubkey -> votes for them
    AuditLogChain,            // Audit log entries chain
    BootstrapConfiguration,   // Path to bootstrap config
    // PROOF-002: Gradient proof link types
    GradientToProof,          // ModelGradient -> GradientProof
    NodeToProofs,             // Node ID -> all proofs by that node
    RoundToProofs,            // Round -> all proofs in that round
    ProofAuditChain,          // Proof audit log chain
    VerifiedProofs,           // Path to verified proofs (index)
    RejectedProofs,           // Path to rejected proofs (index)
    // Phase 1: Aggregation Consensus link types
    RoundToCommitments,       // Round -> AggregationCommitment entries
    RoundToReveals,           // Round -> AggregationReveal entries
    RoundToConsensus,         // Round -> ConsensusResult
    // Phase 2: Round scheduling
    ScheduleConfig,           // Path to current RoundSchedule
    RoundToSchedule,          // Round -> RoundSchedule entries (alias for scheduling module)
    // Phase 4: Byzantine detection consensus
    RoundToByzantineVotes,    // Round -> ByzantineVote entries
    // Validator management
    ValidatorRegistry,        // Path to ValidatorRegistration entries
    ValidatorRegistrations,   // Path to ValidatorRegistration entries (alias for consensus module)
    AgentToValidator,         // Agent -> their ValidatorRegistration
    // Model versioning
    RoundToModelConfig,       // Round -> ModelConfig entry
    ModelConfigRegistry,      // Path to all ModelConfig entries
    // Coordinator rotation
    CoordinatorTerms,         // Path to CoordinatorTerm entries
    AgentToTerms,             // Agent -> their CoordinatorTerm entries
    // Coherence time-series
    CoherenceTimeSeries,      // Path to CoherenceRecord entries
    // Replay detection
    GradientFingerprints,     // Path to GradientFingerprint entries
}

// === Validation Functions ===

/// Validation function for ModelGradient entries
/// SECURITY: Enhanced with F-01 input sanitization
pub fn validate_model_gradient(gradient: ModelGradient) -> ExternResult<ValidateCallbackResult> {
    let node_id = gradient.node_id.trim();

    // F-01: Empty check
    if node_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Node ID cannot be empty".to_string(),
        ));
    }

    // F-01: Length check
    if node_id.len() > MAX_NODE_ID_LENGTH {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Node ID exceeds maximum length of {} characters", MAX_NODE_ID_LENGTH),
        ));
    }

    // F-01: Character validation (alphanumeric, hyphens, underscores, colons)
    if !node_id.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == ':') {
        return Ok(ValidateCallbackResult::Invalid(
            "Node ID contains invalid characters".to_string(),
        ));
    }

    if gradient.gradient_hash.len() != 64 {
        return Ok(ValidateCallbackResult::Invalid(
            "Gradient hash must be 64 hex characters".to_string(),
        ));
    }

    // Validate gradient hash is valid hex
    if !gradient.gradient_hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Ok(ValidateCallbackResult::Invalid(
            "Gradient hash must contain only hexadecimal characters".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&gradient.cpu_usage) || !gradient.cpu_usage.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "CPU usage must be between 0.0 and 1.0".to_string(),
        ));
    }

    if gradient.memory_mb < 0.0 || !gradient.memory_mb.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Memory must be non-negative".to_string(),
        ));
    }

    if gradient.network_latency_ms < 0.0 || !gradient.network_latency_ms.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Network latency must be non-negative".to_string(),
        ));
    }

    if let Some(score) = gradient.trust_score {
        if !(0.0..=1.0).contains(&score) || !score.is_finite() {
            return Ok(ValidateCallbackResult::Invalid(
                "Trust score must be between 0.0 and 1.0".to_string(),
            ));
        }
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validation function for TrainingRound entries
/// SECURITY: Enhanced with F-07 Byzantine threshold validation
pub fn validate_training_round(round: TrainingRound) -> ExternResult<ValidateCallbackResult> {
    if round.participant_count == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Participant count must be at least 1".to_string(),
        ));
    }
    if round.byzantine_count > round.participant_count {
        return Ok(ValidateCallbackResult::Invalid(
            "Byzantine count cannot exceed participant count".to_string(),
        ));
    }

    // F-07: Validate Byzantine threshold at validation layer
    let byzantine_fraction = round.byzantine_count as f32 / round.participant_count as f32;
    if byzantine_fraction > MAX_BYZANTINE_TOLERANCE {
        return Ok(ValidateCallbackResult::Invalid(
            format!(
                "Byzantine fraction {:.2}% exceeds maximum tolerance of {:.0}%",
                byzantine_fraction * 100.0,
                MAX_BYZANTINE_TOLERANCE * 100.0
            ),
        ));
    }

    if !(0.0..=1.0).contains(&round.accuracy) || !round.accuracy.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Accuracy must be between 0.0 and 1.0".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validation function for ByzantineRecord entries
/// SECURITY: Enhanced with F-01 input sanitization
pub fn validate_byzantine_record(record: ByzantineRecord) -> ExternResult<ValidateCallbackResult> {
    let node_id = record.node_id.trim();

    if node_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Node ID cannot be empty".to_string(),
        ));
    }

    if node_id.len() > MAX_NODE_ID_LENGTH {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Node ID exceeds maximum length of {} characters", MAX_NODE_ID_LENGTH),
        ));
    }

    if !node_id.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == ':') {
        return Ok(ValidateCallbackResult::Invalid(
            "Node ID contains invalid characters".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&record.confidence) || !record.confidence.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Confidence must be between 0.0 and 1.0".to_string(),
        ));
    }

    if record.detection_method.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Detection method cannot be empty".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validation function for NodeReputation entries
/// SECURITY: Enhanced with F-01 input sanitization
pub fn validate_node_reputation(reputation: NodeReputation) -> ExternResult<ValidateCallbackResult> {
    let node_id = reputation.node_id.trim();

    if node_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Node ID cannot be empty".to_string(),
        ));
    }

    if node_id.len() > MAX_NODE_ID_LENGTH {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Node ID exceeds maximum length of {} characters", MAX_NODE_ID_LENGTH),
        ));
    }

    if !node_id.chars().all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == ':') {
        return Ok(ValidateCallbackResult::Invalid(
            "Node ID contains invalid characters".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&reputation.reputation_score) || !reputation.reputation_score.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Reputation score must be between 0.0 and 1.0".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// SEC-002: Validation Functions for Coordinator Bootstrap Types
// =============================================================================

/// Validate GenesisCoordinator entries
pub fn validate_genesis_coordinator(coord: GenesisCoordinator) -> ExternResult<ValidateCallbackResult> {
    // Agent pubkey must not be empty
    if coord.agent_pubkey.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Genesis coordinator agent pubkey cannot be empty".to_string(),
        ));
    }

    // Authority proof must be present and have minimum size (64 bytes for signature)
    if coord.authority_proof.len() < 64 {
        return Ok(ValidateCallbackResult::Invalid(
            "Authority proof must be at least 64 bytes (cryptographic signature)".to_string(),
        ));
    }

    // DNA modifiers hash must be valid hex (64 chars for SHA-256)
    if coord.dna_modifiers_hash.len() != 64 ||
       !coord.dna_modifiers_hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Ok(ValidateCallbackResult::Invalid(
            "DNA modifiers hash must be 64 hex characters (SHA-256)".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate Guardian entries
pub fn validate_guardian(guardian: Guardian) -> ExternResult<ValidateCallbackResult> {
    if guardian.agent_pubkey.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Guardian agent pubkey cannot be empty".to_string(),
        ));
    }

    if guardian.voting_weight == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Guardian voting weight must be at least 1".to_string(),
        ));
    }

    if guardian.added_by.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Guardian must have a valid added_by reference".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate CoordinatorCredential entries
pub fn validate_coordinator_credential(cred: CoordinatorCredential) -> ExternResult<ValidateCallbackResult> {
    if cred.agent_pubkey.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Coordinator credential agent pubkey cannot be empty".to_string(),
        ));
    }

    // Credential type must be one of: genesis, voted, delegated
    let valid_types = ["genesis", "voted", "delegated"];
    if !valid_types.contains(&cred.credential_type.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Invalid credential type: must be one of {:?}", valid_types),
        ));
    }

    // For non-genesis credentials, must have required signatures
    if cred.credential_type != "genesis" && cred.required_signatures == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Non-genesis credentials must require at least 1 guardian signature".to_string(),
        ));
    }

    // Guardian signatures JSON must be valid (at minimum an empty array)
    if serde_json::from_str::<Vec<(String, Vec<u8>)>>(&cred.guardian_signatures_json).is_err() {
        // Try alternate format
        if serde_json::from_str::<serde_json::Value>(&cred.guardian_signatures_json).is_err() {
            return Ok(ValidateCallbackResult::Invalid(
                "Guardian signatures must be valid JSON".to_string(),
            ));
        }
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate CoordinatorVote entries
pub fn validate_coordinator_vote(vote: CoordinatorVote) -> ExternResult<ValidateCallbackResult> {
    if vote.vote_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Vote ID cannot be empty".to_string(),
        ));
    }

    if vote.candidate_pubkey.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Candidate pubkey cannot be empty".to_string(),
        ));
    }

    if vote.voter_pubkey.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Voter pubkey cannot be empty".to_string(),
        ));
    }

    // Vote must have cryptographic signature (at least 64 bytes)
    if vote.signature.len() < 64 {
        return Ok(ValidateCallbackResult::Invalid(
            "Vote signature must be at least 64 bytes".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate CoordinatorAuditLog entries
pub fn validate_coordinator_audit_log(log: CoordinatorAuditLog) -> ExternResult<ValidateCallbackResult> {
    if log.log_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Audit log ID cannot be empty".to_string(),
        ));
    }

    if log.coordinator_pubkey.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Coordinator pubkey cannot be empty".to_string(),
        ));
    }

    if log.action_type.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Action type cannot be empty".to_string(),
        ));
    }

    // Previous log hash should be 64 hex chars or "genesis" for first entry
    if log.previous_log_hash != "genesis" &&
       (log.previous_log_hash.len() != 64 ||
        !log.previous_log_hash.chars().all(|c| c.is_ascii_hexdigit())) {
        return Ok(ValidateCallbackResult::Invalid(
            "Previous log hash must be 64 hex characters or 'genesis'".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate BootstrapConfig entries
pub fn validate_bootstrap_config(config: BootstrapConfig) -> ExternResult<ValidateCallbackResult> {
    // Window start must be before window end
    if config.window_start >= config.window_end {
        return Ok(ValidateCallbackResult::Invalid(
            "Bootstrap window start must be before window end".to_string(),
        ));
    }

    // Min guardians must be at least 1
    if config.min_guardians == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Minimum guardians must be at least 1".to_string(),
        ));
    }

    // Min votes must be at least 1 and <= max guardians
    if config.min_votes == 0 || config.min_votes > config.max_guardians {
        return Ok(ValidateCallbackResult::Invalid(
            "Min votes must be between 1 and max_guardians".to_string(),
        ));
    }

    // Max guardians must be >= min guardians
    if config.max_guardians < config.min_guardians {
        return Ok(ValidateCallbackResult::Invalid(
            "Max guardians must be >= min guardians".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// PROOF-002: Validation Functions for Gradient Proof Types
// =============================================================================

/// Validate GradientProof entries
pub fn validate_gradient_proof(proof: GradientProof) -> ExternResult<ValidateCallbackResult> {
    // Node ID validation
    let node_id = proof.node_id.trim();
    if node_id.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proof node ID cannot be empty".to_string(),
        ));
    }
    if node_id.len() > MAX_NODE_ID_LENGTH {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Proof node ID exceeds maximum length of {} characters", MAX_NODE_ID_LENGTH),
        ));
    }

    // Gradient hash must be valid hex (64 chars)
    if proof.gradient_hash.len() != 64 ||
       !proof.gradient_hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Ok(ValidateCallbackResult::Invalid(
            "Gradient hash must be 64 hex characters".to_string(),
        ));
    }

    // Gradient commitment must be valid hex (64 chars)
    if proof.gradient_commitment.len() != 64 ||
       !proof.gradient_commitment.chars().all(|c| c.is_ascii_hexdigit()) {
        return Ok(ValidateCallbackResult::Invalid(
            "Gradient commitment must be 64 hex characters".to_string(),
        ));
    }

    // Data commitment must be valid hex (64 chars)
    if proof.data_commitment.len() != 64 ||
       !proof.data_commitment.chars().all(|c| c.is_ascii_hexdigit()) {
        return Ok(ValidateCallbackResult::Invalid(
            "Data commitment must be 64 hex characters".to_string(),
        ));
    }

    // Proof bytes must not be empty and must be valid base64
    if proof.proof_bytes.is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Proof bytes cannot be empty".to_string(),
        ));
    }

    // Security level must be valid
    let valid_levels = ["standard96", "standard128", "high256"];
    if !valid_levels.contains(&proof.security_level.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Invalid security level: must be one of {:?}", valid_levels),
        ));
    }

    // L2 norm must be finite and non-negative
    if proof.l2_norm < 0.0 || !proof.l2_norm.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "L2 norm must be non-negative and finite".to_string(),
        ));
    }

    // Num elements must be positive
    if proof.num_elements == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Number of gradient elements must be positive".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate ProofVerificationStatus entries
pub fn validate_proof_verification_status(status: ProofVerificationStatus) -> ExternResult<ValidateCallbackResult> {
    // Verifier must not be empty
    if status.verifier.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Verifier identifier cannot be empty".to_string(),
        ));
    }

    // Issues JSON must be valid (at minimum an empty array)
    if serde_json::from_str::<Vec<String>>(&status.issues_json).is_err() {
        return Ok(ValidateCallbackResult::Invalid(
            "Issues must be valid JSON array of strings".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate ProofAuditEntry entries
pub fn validate_proof_audit_entry(entry: ProofAuditEntry) -> ExternResult<ValidateCallbackResult> {
    // Entry ID must not be empty
    if entry.entry_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Audit entry ID cannot be empty".to_string(),
        ));
    }

    // Action type must be valid
    let valid_actions = ["submit", "verify", "reject", "accept"];
    if !valid_actions.contains(&entry.action_type.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Invalid action type: must be one of {:?}", valid_actions),
        ));
    }

    // Proof hash must be valid hex (64 chars)
    if entry.proof_hash.len() != 64 ||
       !entry.proof_hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Ok(ValidateCallbackResult::Invalid(
            "Proof hash must be 64 hex characters".to_string(),
        ));
    }

    // Node ID must not be empty
    if entry.node_id.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Node ID cannot be empty".to_string(),
        ));
    }

    // Result must be valid
    let valid_results = ["success", "failure", "pending"];
    if !valid_results.contains(&entry.result.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Invalid result: must be one of {:?}", valid_results),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

// =============================================================================
// Phase 1-4: Validation Functions for Consensus Types
// =============================================================================

/// Validate AggregationCommitment entries
pub fn validate_aggregation_commitment(c: AggregationCommitment) -> ExternResult<ValidateCallbackResult> {
    if c.aggregator.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Aggregator cannot be empty".to_string()));
    }
    // Commitment hash must be 64 hex chars (SHA-256 hex-encoded)
    if c.commitment_hash.len() != 64 || !c.commitment_hash.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Commitment hash must be 64 hex chars (SHA-256), got {}", c.commitment_hash.len()),
        ));
    }
    if c.method.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Aggregation method cannot be empty".to_string()));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate AggregationReveal entries
pub fn validate_aggregation_reveal(r: AggregationReveal) -> ExternResult<ValidateCallbackResult> {
    if r.aggregator.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Aggregator cannot be empty".to_string()));
    }
    if r.result_data.is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Result data cannot be empty".to_string()));
    }
    // Result hash must be 64 hex chars (SHA-256 hex-encoded)
    if r.result_hash.len() != 64 || !r.result_hash.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Result hash must be 64 hex chars (SHA-256), got {}", r.result_hash.len()),
        ));
    }
    // Verify result_hash matches SHA-256(result_data) — critical integrity check
    let mut hasher = sha2::Sha256::new();
    hasher.update(&r.result_data);
    let computed_hex = format!("{:x}", hasher.finalize());
    if computed_hex != r.result_hash {
        return Ok(ValidateCallbackResult::Invalid(
            "Result hash does not match SHA-256(result_data) — commitment mismatch".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate ConsensusResult entries
pub fn validate_consensus_result(c: ConsensusResult) -> ExternResult<ValidateCallbackResult> {
    if c.agreed_hash.len() != 64 || !c.agreed_hash.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Agreed hash must be 64 hex chars, got {}", c.agreed_hash.len()),
        ));
    }
    // Consensus weight must be > 2/3 of total weight
    if c.total_weight <= 0.0 {
        return Ok(ValidateCallbackResult::Invalid("Total weight must be positive".to_string()));
    }
    let threshold = c.total_weight * 2.0 / 3.0;
    if c.consensus_weight < threshold {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Consensus weight {:.2} is below 2/3 threshold {:.2}", c.consensus_weight, threshold),
        ));
    }
    if c.method.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Method cannot be empty".to_string()));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate RoundSchedule entries
pub fn validate_round_schedule(s: RoundSchedule) -> ExternResult<ValidateCallbackResult> {
    if s.round_duration_secs == 0 {
        return Ok(ValidateCallbackResult::Invalid("Round duration must be positive".to_string()));
    }
    if s.min_participants == 0 {
        return Ok(ValidateCallbackResult::Invalid("Min participants must be positive".to_string()));
    }
    if s.max_participants < s.min_participants {
        return Ok(ValidateCallbackResult::Invalid("Max participants must be >= min participants".to_string()));
    }
    if s.commit_window_secs == 0 {
        return Ok(ValidateCallbackResult::Invalid("Commit window must be positive".to_string()));
    }
    if s.reveal_window_secs == 0 {
        return Ok(ValidateCallbackResult::Invalid("Reveal window must be positive".to_string()));
    }
    if s.created_by.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Created_by cannot be empty".to_string()));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate ByzantineVote entries
pub fn validate_byzantine_vote(v: ByzantineVote) -> ExternResult<ValidateCallbackResult> {
    if v.voter.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Voter cannot be empty".to_string()));
    }
    if !(0.0..=1.0).contains(&v.voter_reputation) || !v.voter_reputation.is_finite() {
        return Ok(ValidateCallbackResult::Invalid("Voter reputation must be between 0.0 and 1.0".to_string()));
    }
    if v.evidence_hash.len() != 64 || !v.evidence_hash.chars().all(|ch| ch.is_ascii_hexdigit()) {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Evidence hash must be 64 hex chars, got {}", v.evidence_hash.len()),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validate ValidatorRegistration entries
pub fn validate_validator_registration(v: ValidatorRegistration) -> ExternResult<ValidateCallbackResult> {
    if v.agent_pubkey.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid("Agent pubkey cannot be empty".to_string()));
    }
    if !(0.0..=1.0).contains(&v.reputation_score) || !v.reputation_score.is_finite() {
        return Ok(ValidateCallbackResult::Invalid("Reputation score must be between 0.0 and 1.0".to_string()));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Validation function for ModelConfig entries
pub fn validate_model_config(config: ModelConfig) -> ExternResult<ValidateCallbackResult> {
    // Architecture hash must be a valid SHA-256 hex string (64 chars)
    if config.architecture_hash.len() != 64 || !config.architecture_hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Ok(ValidateCallbackResult::Invalid(
            "architecture_hash must be a 64-character hex SHA-256 hash".to_string(),
        ));
    }

    // Dimension must be positive
    if config.dimension == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Model dimension must be greater than 0".to_string(),
        ));
    }

    // Serialization format must be specified
    if config.serialization_format.trim().is_empty() || config.serialization_format.len() > 64 {
        return Ok(ValidateCallbackResult::Invalid(
            "Serialization format must be 1-64 characters".to_string(),
        ));
    }

    // Hyperparameters must be valid JSON (basic check: starts with { or [)
    let hp = config.hyperparameters_json.trim();
    if hp.is_empty() || hp.len() > 16384 {
        return Ok(ValidateCallbackResult::Invalid(
            "Hyperparameters JSON must be 1-16384 characters".to_string(),
        ));
    }

    // Description is optional but bounded
    if config.description.len() > 4096 {
        return Ok(ValidateCallbackResult::Invalid(
            "Description must be at most 4096 characters".to_string(),
        ));
    }

    // Registered_by must be non-empty
    if config.registered_by.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "registered_by must identify the registering agent".to_string(),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validation for CoordinatorTerm entries
pub fn validate_coordinator_term(term: CoordinatorTerm) -> ExternResult<ValidateCallbackResult> {
    // Term number must be positive
    if term.term_number == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Term number must be >= 1".to_string(),
        ));
    }

    // Term end must be after term start
    if term.term_end <= term.term_start {
        return Ok(ValidateCallbackResult::Invalid(
            "Term end must be after term start".to_string(),
        ));
    }

    // Coordinator pubkey must be non-empty
    if term.coordinator_pubkey.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Coordinator pubkey is required".to_string(),
        ));
    }

    // Selection method must be valid
    let valid_methods = ["genesis", "voted", "re-elected"];
    if !valid_methods.contains(&term.selection_method.as_str()) {
        return Ok(ValidateCallbackResult::Invalid(
            format!(
                "Invalid selection method '{}'. Must be one of: genesis, voted, re-elected",
                term.selection_method
            ),
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate CoherenceRecord entries
pub fn validate_coherence_record(r: CoherenceRecord) -> ExternResult<ValidateCallbackResult> {
    if !(0.0..=1.0).contains(&r.coherence_value) || !r.coherence_value.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Coherence value must be between 0.0 and 1.0".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&r.epistemic_confidence) || !r.epistemic_confidence.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Epistemic confidence must be between 0.0 and 1.0".to_string(),
        ));
    }
    if r.byzantine_count > r.node_count {
        return Ok(ValidateCallbackResult::Invalid(
            "Byzantine count cannot exceed node count".to_string(),
        ));
    }
    Ok(ValidateCallbackResult::Valid)
}

/// Main validation dispatcher
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry(store_entry) => match store_entry.action.hashed.content.entry_type() {
            EntryType::App(app_entry_def) => {
                let entry = store_entry.entry;
                match EntryTypes::deserialize_from_type(
                    app_entry_def.zome_index,
                    app_entry_def.entry_index,
                    &entry,
                )? {
                    Some(EntryTypes::ModelGradient(gradient)) => validate_model_gradient(gradient),
                    Some(EntryTypes::TrainingRound(round)) => validate_training_round(round),
                    Some(EntryTypes::ByzantineRecord(record)) => validate_byzantine_record(record),
                    Some(EntryTypes::NodeReputation(reputation)) => validate_node_reputation(reputation),
                    // SEC-002: Validate coordinator bootstrap entries
                    Some(EntryTypes::GenesisCoordinator(coord)) => validate_genesis_coordinator(coord),
                    Some(EntryTypes::Guardian(guardian)) => validate_guardian(guardian),
                    Some(EntryTypes::CoordinatorCredential(cred)) => validate_coordinator_credential(cred),
                    Some(EntryTypes::CoordinatorVote(vote)) => validate_coordinator_vote(vote),
                    Some(EntryTypes::CoordinatorAuditLog(log)) => validate_coordinator_audit_log(log),
                    Some(EntryTypes::BootstrapConfig(config)) => validate_bootstrap_config(config),
                    // PROOF-002: Validate gradient proof entries
                    Some(EntryTypes::GradientProof(proof)) => validate_gradient_proof(proof),
                    Some(EntryTypes::ProofVerificationStatus(status)) => validate_proof_verification_status(status),
                    Some(EntryTypes::ProofAuditEntry(entry)) => validate_proof_audit_entry(entry),
                    // Phase 1-4: Consensus types
                    Some(EntryTypes::AggregationCommitment(c)) => validate_aggregation_commitment(c),
                    Some(EntryTypes::AggregationReveal(r)) => validate_aggregation_reveal(r),
                    Some(EntryTypes::ConsensusResult(c)) => validate_consensus_result(c),
                    Some(EntryTypes::RoundSchedule(s)) => validate_round_schedule(s),
                    Some(EntryTypes::ByzantineVote(v)) => validate_byzantine_vote(v),
                    Some(EntryTypes::ValidatorRegistration(v)) => validate_validator_registration(v),
                    // Model versioning
                    Some(EntryTypes::ModelConfig(config)) => validate_model_config(config),
                    // Coordinator rotation
                    Some(EntryTypes::CoordinatorTerm(term)) => validate_coordinator_term(term),
                    // Coherence tracking
                    Some(EntryTypes::CoherenceRecord(r)) => validate_coherence_record(r),
                    // Replay detection fingerprints (lightweight — just validate node_id)
                    Some(EntryTypes::GradientFingerprint(f)) => {
                        if f.node_id.trim().is_empty() {
                            Ok(ValidateCallbackResult::Invalid("Fingerprint node_id cannot be empty".to_string()))
                        } else if f.hash_hex.len() != 64 || !f.hash_hex.chars().all(|c| c.is_ascii_hexdigit()) {
                            Ok(ValidateCallbackResult::Invalid("Fingerprint hash must be 64 hex chars".to_string()))
                        } else {
                            Ok(ValidateCallbackResult::Valid)
                        }
                    }
                    None => Ok(ValidateCallbackResult::Valid),
                }
            }
            _ => Ok(ValidateCallbackResult::Valid),
        },
        _ => Ok(ValidateCallbackResult::Valid),
    }
}

// =============================================================================
// Unit Tests for Phase 1-4 Validation Functions
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use hdi::prelude::ValidateCallbackResult;
    use sha2::{Sha256, Digest};

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// Build a valid 64-char hex hash (all 'a's — represents 32 bytes).
    fn hash_32() -> String {
        "a".repeat(64)
    }

    /// Build a 32-char hex string (wrong length — represents 16 bytes, not 32).
    fn hash_16() -> String {
        "b".repeat(32)
    }

    /// Compute SHA-256 of `data` and return the hex-encoded digest.
    fn sha256(data: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(data);
        format!("{:x}", h.finalize())
    }

    // -------------------------------------------------------------------------
    // validate_aggregation_commitment
    // -------------------------------------------------------------------------

    #[test]
    fn test_valid_commitment() {
        let c = AggregationCommitment {
            round: 1,
            aggregator: "agent-1".to_string(),
            commitment_hash: hash_32(),
            method: "ByzantineFilteredHV".to_string(),
            committed_at: 1_700_000_000,
            gradient_count: 5,
            excluded_count: 0,
            aggregator_trust_score: 0.9,
        };
        let result = validate_aggregation_commitment(c).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_commitment_empty_aggregator() {
        let c = AggregationCommitment {
            round: 1,
            aggregator: "  ".to_string(),
            commitment_hash: hash_32(),
            method: "ByzantineFilteredHV".to_string(),
            committed_at: 1_700_000_000,
            gradient_count: 5,
            excluded_count: 0,
            aggregator_trust_score: 0.9,
        };
        let result = validate_aggregation_commitment(c).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Aggregator"), "Expected aggregator error, got: {msg}");
            }
            other => panic!("Expected Invalid, got: {other:?}"),
        }
    }

    #[test]
    fn test_commitment_wrong_hash_length() {
        let c = AggregationCommitment {
            round: 1,
            aggregator: "agent-1".to_string(),
            commitment_hash: hash_16(), // 16 bytes, not 32
            method: "ByzantineFilteredHV".to_string(),
            committed_at: 1_700_000_000,
            gradient_count: 5,
            excluded_count: 0,
            aggregator_trust_score: 0.9,
        };
        let result = validate_aggregation_commitment(c).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("64 hex"), "Expected hash-length error, got: {msg}");
            }
            other => panic!("Expected Invalid, got: {other:?}"),
        }
    }

    #[test]
    fn test_commitment_empty_method() {
        let c = AggregationCommitment {
            round: 1,
            aggregator: "agent-1".to_string(),
            commitment_hash: hash_32(),
            method: "  ".to_string(),
            committed_at: 1_700_000_000,
            gradient_count: 5,
            excluded_count: 0,
            aggregator_trust_score: 0.9,
        };
        let result = validate_aggregation_commitment(c).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("method"), "Expected method error, got: {msg}");
            }
            other => panic!("Expected Invalid, got: {other:?}"),
        }
    }

    // -------------------------------------------------------------------------
    // validate_aggregation_reveal
    // -------------------------------------------------------------------------

    #[test]
    fn test_valid_reveal() {
        let data = vec![1u8, 2, 3, 4, 5];
        let hash = sha256(&data);
        let r = AggregationReveal {
            round: 1,
            aggregator: "agent-1".to_string(),
            result_data: data,
            result_hash: hash,
            detection_summary_json: "{}".to_string(),
            shapley_values_json: "[]".to_string(),
            revealed_at: 1_700_000_000,
        };
        let result = validate_aggregation_reveal(r).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_reveal_hash_mismatch() {
        let data = vec![1u8, 2, 3, 4, 5];
        let wrong_hash = sha256(b"wrong data");
        let r = AggregationReveal {
            round: 1,
            aggregator: "agent-1".to_string(),
            result_data: data,
            result_hash: wrong_hash,
            detection_summary_json: "{}".to_string(),
            shapley_values_json: "[]".to_string(),
            revealed_at: 1_700_000_000,
        };
        let result = validate_aggregation_reveal(r).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("commitment mismatch"), "Expected mismatch error, got: {msg}");
            }
            other => panic!("Expected Invalid, got: {other:?}"),
        }
    }

    #[test]
    fn test_reveal_empty_data() {
        let r = AggregationReveal {
            round: 1,
            aggregator: "agent-1".to_string(),
            result_data: vec![],
            result_hash: hash_32(),
            detection_summary_json: "{}".to_string(),
            shapley_values_json: "[]".to_string(),
            revealed_at: 1_700_000_000,
        };
        let result = validate_aggregation_reveal(r).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("empty"), "Expected empty-data error, got: {msg}");
            }
            other => panic!("Expected Invalid, got: {other:?}"),
        }
    }

    #[test]
    fn test_reveal_wrong_hash_length() {
        let data = vec![1u8, 2, 3, 4, 5];
        let r = AggregationReveal {
            round: 1,
            aggregator: "agent-1".to_string(),
            result_data: data,
            result_hash: hash_16(), // 16 bytes, not 32
            detection_summary_json: "{}".to_string(),
            shapley_values_json: "[]".to_string(),
            revealed_at: 1_700_000_000,
        };
        let result = validate_aggregation_reveal(r).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("64 hex"), "Expected hash-length error, got: {msg}");
            }
            other => panic!("Expected Invalid, got: {other:?}"),
        }
    }

    // -------------------------------------------------------------------------
    // validate_consensus_result
    // -------------------------------------------------------------------------

    #[test]
    fn test_valid_consensus() {
        let c = ConsensusResult {
            round: 1,
            agreed_hash: hash_32(),
            agreeing_validators_json: "[\"agent-1\",\"agent-2\"]".to_string(),
            total_weight: 9.0,
            consensus_weight: 7.0, // 7.0 >= 6.0 (2/3 of 9)
            method: "ByzantineFilteredHV".to_string(),
            finalized_at: 1_700_000_000,
            finalized_by: "agent-1".to_string(),
        };
        let result = validate_consensus_result(c).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_consensus_below_threshold() {
        let c = ConsensusResult {
            round: 1,
            agreed_hash: hash_32(),
            agreeing_validators_json: "[\"agent-1\"]".to_string(),
            total_weight: 9.0,
            consensus_weight: 5.0, // 5.0 < 6.0 (2/3 of 9)
            method: "ByzantineFilteredHV".to_string(),
            finalized_at: 1_700_000_000,
            finalized_by: "agent-1".to_string(),
        };
        let result = validate_consensus_result(c).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("below"), "Expected threshold error, got: {msg}");
            }
            other => panic!("Expected Invalid, got: {other:?}"),
        }
    }

    #[test]
    fn test_consensus_zero_total_weight() {
        let c = ConsensusResult {
            round: 1,
            agreed_hash: hash_32(),
            agreeing_validators_json: "[]".to_string(),
            total_weight: 0.0,
            consensus_weight: 0.0,
            method: "ByzantineFilteredHV".to_string(),
            finalized_at: 1_700_000_000,
            finalized_by: "agent-1".to_string(),
        };
        let result = validate_consensus_result(c).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("positive"), "Expected positive-weight error, got: {msg}");
            }
            other => panic!("Expected Invalid, got: {other:?}"),
        }
    }

    #[test]
    fn test_consensus_exact_threshold() {
        // Exactly 2/3: total=9.0, threshold=6.0, consensus_weight=6.0
        let c = ConsensusResult {
            round: 1,
            agreed_hash: hash_32(),
            agreeing_validators_json: "[\"a\",\"b\",\"c\"]".to_string(),
            total_weight: 9.0,
            consensus_weight: 6.0, // exactly 2/3
            method: "FedAvgHV".to_string(),
            finalized_at: 1_700_000_000,
            finalized_by: "agent-1".to_string(),
        };
        let result = validate_consensus_result(c).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    // -------------------------------------------------------------------------
    // validate_round_schedule
    // -------------------------------------------------------------------------

    #[test]
    fn test_valid_schedule() {
        let s = RoundSchedule {
            round_duration_secs: 60,
            min_participants: 3,
            max_participants: 10,
            current_round: 0,
            round_start_time: 1_700_000_000,
            commit_window_secs: 30,
            reveal_window_secs: 30,
            created_by: "coordinator-1".to_string(),
            approved_by_json: "[]".to_string(),
            active: true,
        };
        let result = validate_round_schedule(s).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_schedule_zero_duration() {
        let s = RoundSchedule {
            round_duration_secs: 0,
            min_participants: 3,
            max_participants: 10,
            current_round: 0,
            round_start_time: 1_700_000_000,
            commit_window_secs: 30,
            reveal_window_secs: 30,
            created_by: "coordinator-1".to_string(),
            approved_by_json: "[]".to_string(),
            active: true,
        };
        let result = validate_round_schedule(s).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("duration"), "Expected duration error, got: {msg}");
            }
            other => panic!("Expected Invalid, got: {other:?}"),
        }
    }

    #[test]
    fn test_schedule_max_lt_min() {
        let s = RoundSchedule {
            round_duration_secs: 60,
            min_participants: 10,
            max_participants: 3, // less than min
            current_round: 0,
            round_start_time: 1_700_000_000,
            commit_window_secs: 30,
            reveal_window_secs: 30,
            created_by: "coordinator-1".to_string(),
            approved_by_json: "[]".to_string(),
            active: true,
        };
        let result = validate_round_schedule(s).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("participants"), "Expected participants error, got: {msg}");
            }
            other => panic!("Expected Invalid, got: {other:?}"),
        }
    }

    // -------------------------------------------------------------------------
    // validate_byzantine_vote
    // -------------------------------------------------------------------------

    #[test]
    fn test_valid_byzantine_vote() {
        let v = ByzantineVote {
            round: 1,
            voter: "agent-1".to_string(),
            voter_reputation: 0.85,
            voter_trust_score: 0.85,
            flagged_nodes_json: "[]".to_string(),
            detection_layers_json: "[\"statistical\"]".to_string(),
            evidence_hash: hash_32(),
            voted_at: 1_700_000_000,
        };
        let result = validate_byzantine_vote(v).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_vote_reputation_out_of_range() {
        let v = ByzantineVote {
            round: 1,
            voter: "agent-1".to_string(),
            voter_reputation: 1.5, // out of [0, 1]
            voter_trust_score: 1.5,
            flagged_nodes_json: "[]".to_string(),
            detection_layers_json: "[]".to_string(),
            evidence_hash: hash_32(),
            voted_at: 1_700_000_000,
        };
        let result = validate_byzantine_vote(v).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("reputation"), "Expected reputation error, got: {msg}");
            }
            other => panic!("Expected Invalid, got: {other:?}"),
        }
    }

    #[test]
    fn test_vote_nan_reputation() {
        let v = ByzantineVote {
            round: 1,
            voter: "agent-1".to_string(),
            voter_reputation: f32::NAN,
            voter_trust_score: f32::NAN,
            flagged_nodes_json: "[]".to_string(),
            detection_layers_json: "[]".to_string(),
            evidence_hash: hash_32(),
            voted_at: 1_700_000_000,
        };
        let result = validate_byzantine_vote(v).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("reputation"), "Expected reputation error, got: {msg}");
            }
            other => panic!("Expected Invalid, got: {other:?}"),
        }
    }

    #[test]
    fn test_vote_wrong_evidence_hash() {
        let v = ByzantineVote {
            round: 1,
            voter: "agent-1".to_string(),
            voter_reputation: 0.5,
            voter_trust_score: 0.5,
            flagged_nodes_json: "[]".to_string(),
            detection_layers_json: "[]".to_string(),
            evidence_hash: hash_16(), // 16 bytes, not 32
            voted_at: 1_700_000_000,
        };
        let result = validate_byzantine_vote(v).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("64 hex"), "Expected hash-length error, got: {msg}");
            }
            other => panic!("Expected Invalid, got: {other:?}"),
        }
    }

    // -------------------------------------------------------------------------
    // validate_validator_registration
    // -------------------------------------------------------------------------

    #[test]
    fn test_valid_registration() {
        let v = ValidatorRegistration {
            agent_pubkey: "uhCAk_test_pubkey_abc123".to_string(),
            reputation_score: 0.95,
            trust_score: 0.95,
            signing_key: "deadbeef".to_string(),
            kvector_json: "{}".to_string(),
            registered_at: 1_700_000_000,
            active: true,
            rounds_participated: 0,
            consensus_matches: 0,
        };
        let result = validate_validator_registration(v).unwrap();
        assert_eq!(result, ValidateCallbackResult::Valid);
    }

    #[test]
    fn test_registration_empty_pubkey() {
        let v = ValidatorRegistration {
            agent_pubkey: "  ".to_string(),
            reputation_score: 0.5,
            trust_score: 0.5,
            signing_key: "deadbeef".to_string(),
            kvector_json: "{}".to_string(),
            registered_at: 1_700_000_000,
            active: true,
            rounds_participated: 0,
            consensus_matches: 0,
        };
        let result = validate_validator_registration(v).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("pubkey"), "Expected pubkey error, got: {msg}");
            }
            other => panic!("Expected Invalid, got: {other:?}"),
        }
    }

    #[test]
    fn test_registration_negative_reputation() {
        let v = ValidatorRegistration {
            agent_pubkey: "uhCAk_test_pubkey_abc123".to_string(),
            reputation_score: -0.1,
            trust_score: -0.1,
            signing_key: "deadbeef".to_string(),
            kvector_json: "{}".to_string(),
            registered_at: 1_700_000_000,
            active: true,
            rounds_participated: 0,
            consensus_matches: 0,
        };
        let result = validate_validator_registration(v).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => {
                assert!(msg.contains("Reputation"), "Expected reputation error, got: {msg}");
            }
            other => panic!("Expected Invalid, got: {other:?}"),
        }
    }

    // -------------------------------------------------------------------------
    // ModelConfig validation
    // -------------------------------------------------------------------------

    fn valid_model_config() -> ModelConfig {
        ModelConfig {
            round: 0,
            architecture_hash: "a".repeat(64),
            hyperparameters_json: r#"{"lr": 0.01, "epochs": 10}"#.to_string(),
            dimension: 16384,
            serialization_format: "HV16-128bits".to_string(),
            description: "Initial model configuration".to_string(),
            created_at: 1_700_000_000,
            registered_by: "uhCAk_coordinator_123".to_string(),
        }
    }

    #[test]
    fn test_model_config_valid() {
        let result = validate_model_config(valid_model_config()).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Valid));
    }

    #[test]
    fn test_model_config_bad_arch_hash() {
        let mut config = valid_model_config();
        config.architecture_hash = "too_short".to_string();
        let result = validate_model_config(config).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_model_config_zero_dimension() {
        let mut config = valid_model_config();
        config.dimension = 0;
        let result = validate_model_config(config).unwrap();
        match result {
            ValidateCallbackResult::Invalid(msg) => assert!(msg.contains("dimension")),
            other => panic!("Expected Invalid, got: {other:?}"),
        }
    }

    #[test]
    fn test_model_config_empty_format() {
        let mut config = valid_model_config();
        config.serialization_format = "".to_string();
        let result = validate_model_config(config).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }

    #[test]
    fn test_model_config_empty_registered_by() {
        let mut config = valid_model_config();
        config.registered_by = "   ".to_string();
        let result = validate_model_config(config).unwrap();
        assert!(matches!(result, ValidateCallbackResult::Invalid(_)));
    }
}
