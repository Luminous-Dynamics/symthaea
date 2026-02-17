//! Zome Bridge for Vote Eligibility Proofs
//!
//! Provides the integration layer between the fl-aggregator proof system and
//! Holochain governance zomes. Since Holochain WASM cannot run full STARK
//! verification (Winterfell is too heavy), this module provides:
//!
//! 1. **Proof Generation**: Off-chain proof generation using full Winterfell
//! 2. **Serialization**: Proof serialization compatible with zome storage
//! 3. **Verification Service**: External verification that produces attestations
//! 4. **Attestation Format**: Signed attestations for on-chain storage
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     Client Application                          │
//! │  ┌──────────────┐    ┌──────────────┐    ┌───────────────────┐ │
//! │  │ VoterProfile │───▶│ ProofClient  │───▶│ Serialized Proof  │ │
//! │  └──────────────┘    └──────────────┘    └─────────┬─────────┘ │
//! └────────────────────────────────────────────────────┼───────────┘
//!                                                      │
//!                                          store_eligibility_proof()
//!                                                      │
//!                                                      ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Holochain DHT (Voting Zome)                  │
//! │  ┌──────────────────┐                  ┌─────────────────────┐  │
//! │  │ EligibilityProof │◀────────────────│   VerifiedVote      │  │
//! │  │ (stored, not     │                  │ (requires verified  │  │
//! │  │  verified)       │                  │  proof reference)   │  │
//! │  └────────┬─────────┘                  └─────────────────────┘  │
//! └───────────┼─────────────────────────────────────────────────────┘
//!             │
//!             │ (external verifier fetches)
//!             ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                Verification Oracle / Service                    │
//! │  ┌──────────────────┐    ┌──────────────┐    ┌───────────────┐ │
//! │  │ VerifierService  │───▶│ STARK Verify │───▶│ Attestation   │ │
//! │  └──────────────────┘    └──────────────┘    └───────┬───────┘ │
//! └──────────────────────────────────────────────────────┼─────────┘
//!                                                        │
//!                                           store_attestation()
//!                                                        │
//!                                                        ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Holochain DHT (Voting Zome)                  │
//! │                    ProofAttestation stored                      │
//! │          (vote can now be cast with verified proof)             │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Client Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::integration::zome_bridge::{
//!     ZomeBridgeClient, VoterProfileBuilder, SerializableProof,
//! };
//!
//! // Build voter profile
//! let profile = VoterProfileBuilder::new("did:mycelix:voter123")
//!     .assurance_level(2)
//!     .matl_score(0.7)
//!     .stake(500.0)
//!     .account_age_days(60)
//!     .participation(0.4)
//!     .has_humanity_proof(true)
//!     .build();
//!
//! // Generate proof for Constitutional vote
//! let client = ZomeBridgeClient::new();
//! let proof = client.generate_proof(&profile, ProofProposalType::Constitutional)?;
//!
//! // Serialize for zome storage
//! let serialized = proof.to_zome_format()?;
//! // serialized can now be passed to store_eligibility_proof zome call
//! ```
//!
//! ## Verifier Service Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::integration::zome_bridge::VerifierService;
//!
//! let service = VerifierService::new(signing_key);
//!
//! // Verify a proof from the DHT
//! let attestation = service.verify_and_attest(&proof_bytes)?;
//!
//! // Attestation can be stored on-chain
//! ```

use crate::proofs::{
    VoteEligibilityProof, ProofProposalType, ProofVoterProfile, ProofEligibilityRequirements,
    ProofConfig, ProofResult, ProofError, SecurityLevel, compute_voter_commitment,
};
use blake3::Hasher;
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ============================================================================
// Voter Profile Builder
// ============================================================================

/// Builder for voter profiles
///
/// Provides a fluent API for constructing voter profiles with sensible defaults.
#[derive(Debug, Clone)]
pub struct VoterProfileBuilder {
    did: String,
    assurance_level: u8,
    matl_score: f32,
    stake: f32,
    account_age_days: u32,
    participation_rate: f32,
    has_humanity_proof: bool,
    fl_contributions: u32,
}

impl VoterProfileBuilder {
    /// Create a new builder with a DID
    pub fn new(did: impl Into<String>) -> Self {
        Self {
            did: did.into(),
            assurance_level: 0,
            matl_score: 0.0,
            stake: 0.0,
            account_age_days: 0,
            participation_rate: 0.0,
            has_humanity_proof: false,
            fl_contributions: 0,
        }
    }

    /// Set assurance level (0-4)
    pub fn assurance_level(mut self, level: u8) -> Self {
        self.assurance_level = level.min(4);
        self
    }

    /// Set MATL score (0.0-1.0)
    pub fn matl_score(mut self, score: f32) -> Self {
        self.matl_score = score.clamp(0.0, 1.0);
        self
    }

    /// Set stake amount
    pub fn stake(mut self, amount: f32) -> Self {
        self.stake = amount.max(0.0);
        self
    }

    /// Set account age in days
    pub fn account_age_days(mut self, days: u32) -> Self {
        self.account_age_days = days;
        self
    }

    /// Set participation rate (0.0-1.0)
    pub fn participation(mut self, rate: f32) -> Self {
        self.participation_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set humanity proof status
    pub fn has_humanity_proof(mut self, has_proof: bool) -> Self {
        self.has_humanity_proof = has_proof;
        self
    }

    /// Set FL contributions count
    pub fn fl_contributions(mut self, count: u32) -> Self {
        self.fl_contributions = count;
        self
    }

    /// Build the voter profile
    pub fn build(self) -> ProofVoterProfile {
        ProofVoterProfile {
            did: self.did,
            assurance_level: self.assurance_level,
            matl_score: self.matl_score,
            stake: self.stake,
            account_age_days: self.account_age_days,
            participation_rate: self.participation_rate,
            has_humanity_proof: self.has_humanity_proof,
            fl_contributions: self.fl_contributions,
        }
    }
}

// ============================================================================
// Serializable Proof Format
// ============================================================================

/// Serializable proof format for zome storage
///
/// This format is designed to be compatible with the Holochain voting zome's
/// `EligibilityProof` entry type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableProof {
    /// Voter's DID
    pub voter_did: String,
    /// Blake3 commitment to voter profile (32 bytes)
    pub voter_commitment: Vec<u8>,
    /// Proposal type (0-5)
    pub proposal_type: u8,
    /// Whether voter is eligible
    pub eligible: bool,
    /// Number of requirements met
    pub requirements_met: u8,
    /// Total active requirements
    pub active_requirements: u8,
    /// Serialized STARK proof bytes
    pub proof_bytes: Vec<u8>,
    /// Proof generation timestamp (Unix millis)
    pub generated_at_ms: u64,
    /// Proof size in bytes
    pub proof_size: usize,
}

impl SerializableProof {
    /// Create from a generated proof
    pub fn from_proof(
        voter: &ProofVoterProfile,
        proof: &VoteEligibilityProof,
    ) -> Self {
        let public_inputs = proof.public_inputs();
        let proof_bytes = proof.to_bytes();
        let proof_size = proof_bytes.len();

        Self {
            voter_did: voter.did.clone(),
            voter_commitment: public_inputs.voter_commitment.to_vec(),
            proposal_type: public_inputs.proposal_type,
            eligible: public_inputs.eligible,
            requirements_met: public_inputs.requirements_met,
            active_requirements: public_inputs.active_requirements,
            proof_bytes,
            generated_at_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            proof_size,
        }
    }

    /// Serialize to JSON for zome calls
    pub fn to_json(&self) -> ProofResult<String> {
        serde_json::to_string(self).map_err(|e| {
            ProofError::SerializationFailed(format!("JSON serialization failed: {}", e))
        })
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> ProofResult<Self> {
        serde_json::from_str(json).map_err(|e| {
            ProofError::InvalidProofFormat(format!("JSON deserialization failed: {}", e))
        })
    }

    /// Serialize to MessagePack for compact storage
    pub fn to_msgpack(&self) -> ProofResult<Vec<u8>> {
        rmp_serde::to_vec(self).map_err(|e| {
            ProofError::SerializationFailed(format!("MessagePack serialization failed: {}", e))
        })
    }

    /// Deserialize from MessagePack
    pub fn from_msgpack(bytes: &[u8]) -> ProofResult<Self> {
        rmp_serde::from_slice(bytes).map_err(|e| {
            ProofError::InvalidProofFormat(format!("MessagePack deserialization failed: {}", e))
        })
    }

    /// Reconstruct the VoteEligibilityProof
    pub fn to_vote_proof(&self) -> ProofResult<VoteEligibilityProof> {
        VoteEligibilityProof::from_bytes(&self.proof_bytes)
    }

    /// Calculate proof hash for deduplication
    pub fn proof_hash(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(b"proof:");
        hasher.update(&self.voter_commitment);
        hasher.update(&[self.proposal_type]);
        hasher.update(&self.proof_bytes);
        *hasher.finalize().as_bytes()
    }
}

// ============================================================================
// Zome Bridge Client
// ============================================================================

/// Client for generating proofs for zome storage
pub struct ZomeBridgeClient {
    config: ProofConfig,
}

impl ZomeBridgeClient {
    /// Create a new client with default configuration
    pub fn new() -> Self {
        Self {
            config: ProofConfig {
                security_level: SecurityLevel::Standard96,
                parallel: true,
                max_proof_size: 0,
            },
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ProofConfig) -> Self {
        Self { config }
    }

    /// Set security level
    pub fn security_level(mut self, level: SecurityLevel) -> Self {
        self.config.security_level = level;
        self
    }

    /// Enable/disable parallel proof generation
    pub fn parallel(mut self, enabled: bool) -> Self {
        self.config.parallel = enabled;
        self
    }

    /// Generate an eligibility proof
    pub fn generate_proof(
        &self,
        voter: &ProofVoterProfile,
        proposal_type: ProofProposalType,
    ) -> ProofResult<SerializableProof> {
        let proof = VoteEligibilityProof::generate(voter, proposal_type, self.config.clone())?;
        Ok(SerializableProof::from_proof(voter, &proof))
    }

    /// Generate proofs for all proposal types (for caching)
    pub fn generate_all_proofs(
        &self,
        voter: &ProofVoterProfile,
    ) -> Vec<(ProofProposalType, ProofResult<SerializableProof>)> {
        let types = [
            ProofProposalType::Standard,
            ProofProposalType::Constitutional,
            ProofProposalType::ModelGovernance,
            ProofProposalType::Emergency,
            ProofProposalType::Treasury,
            ProofProposalType::Membership,
        ];

        types
            .into_iter()
            .map(|pt| (pt, self.generate_proof(voter, pt)))
            .collect()
    }

    /// Check eligibility without generating proof (fast check)
    pub fn check_eligibility(
        &self,
        voter: &ProofVoterProfile,
        proposal_type: ProofProposalType,
    ) -> EligibilityCheck {
        let requirements = proposal_type.requirements();
        let checks = vec![
            (
                "assurance_level",
                voter.assurance_level >= requirements.min_assurance_level,
                true,
            ),
            (
                "matl_score",
                voter.matl_score >= requirements.min_matl_score,
                true,
            ),
            (
                "stake",
                voter.stake >= requirements.min_stake,
                requirements.min_stake > 0.0,
            ),
            (
                "account_age",
                voter.account_age_days >= requirements.min_account_age_days,
                requirements.min_account_age_days > 0,
            ),
            (
                "participation",
                voter.participation_rate >= requirements.min_participation,
                requirements.min_participation > 0.0,
            ),
            (
                "humanity_proof",
                voter.has_humanity_proof,
                requirements.humanity_proof_required,
            ),
            (
                "fl_contributions",
                voter.fl_contributions >= requirements.min_fl_contributions,
                requirements.fl_participation_required,
            ),
        ];

        let mut passed = Vec::new();
        let mut failed = Vec::new();
        let mut not_required = Vec::new();

        for (name, satisfied, active) in checks {
            if !active {
                not_required.push(name.to_string());
            } else if satisfied {
                passed.push(name.to_string());
            } else {
                failed.push(name.to_string());
            }
        }

        let active_count = passed.len() + failed.len();
        let eligible = failed.is_empty();

        EligibilityCheck {
            proposal_type,
            eligible,
            requirements_met: passed.len() as u8,
            active_requirements: active_count as u8,
            passed_checks: passed,
            failed_checks: failed,
            not_required_checks: not_required,
        }
    }

    /// Compute voter commitment
    pub fn compute_commitment(&self, voter: &ProofVoterProfile) -> [u8; 32] {
        compute_voter_commitment(voter)
    }
}

impl Default for ZomeBridgeClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of eligibility check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EligibilityCheck {
    /// Proposal type checked
    pub proposal_type: ProofProposalType,
    /// Whether voter is eligible
    pub eligible: bool,
    /// Number of requirements met
    pub requirements_met: u8,
    /// Total active requirements
    pub active_requirements: u8,
    /// Names of passed checks
    pub passed_checks: Vec<String>,
    /// Names of failed checks
    pub failed_checks: Vec<String>,
    /// Names of checks not required for this proposal type
    pub not_required_checks: Vec<String>,
}

// ============================================================================
// Proof Attestation
// ============================================================================

/// Attestation that a proof has been verified
///
/// This is produced by a verification oracle and stored on-chain to indicate
/// that the STARK proof has been verified externally.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofAttestation {
    /// Hash of the proof being attested
    pub proof_hash: [u8; 32],
    /// Voter commitment from the proof
    pub voter_commitment: [u8; 32],
    /// Proposal type
    pub proposal_type: u8,
    /// Whether the proof verified successfully
    pub verified: bool,
    /// Verification timestamp (Unix millis)
    pub verified_at_ms: u64,
    /// Attestation expiry (Unix millis)
    pub expires_at_ms: u64,
    /// Verifier's public key (32 bytes, Ed25519)
    pub verifier_pubkey: [u8; 32],
    /// Signature over attestation data (64 bytes, Ed25519)
    pub signature: [u8; 64],
    /// Security level used in verification
    pub security_level: String,
    /// Verification time in milliseconds
    pub verification_time_ms: u64,
}

impl ProofAttestation {
    /// Default attestation validity period (24 hours)
    pub const DEFAULT_VALIDITY_HOURS: u64 = 24;

    /// Data to sign for attestation
    pub fn signing_data(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(128);
        data.extend_from_slice(b"mycelix:proof_attestation:v1:");
        data.extend_from_slice(&self.proof_hash);
        data.extend_from_slice(&self.voter_commitment);
        data.push(self.proposal_type);
        data.push(if self.verified { 1 } else { 0 });
        data.extend_from_slice(&self.verified_at_ms.to_le_bytes());
        data.extend_from_slice(&self.expires_at_ms.to_le_bytes());
        data
    }

    /// Check if attestation has expired
    pub fn is_expired(&self) -> bool {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        now_ms > self.expires_at_ms
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> ProofResult<String> {
        serde_json::to_string(self).map_err(|e| {
            ProofError::SerializationFailed(format!("Attestation JSON serialization failed: {}", e))
        })
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> ProofResult<Self> {
        serde_json::from_str(json).map_err(|e| {
            ProofError::InvalidProofFormat(format!("Attestation JSON deserialization failed: {}", e))
        })
    }
}

// ============================================================================
// Verifier Service
// ============================================================================

/// Service for verifying proofs and producing attestations
///
/// This should run in a trusted environment (not in Holochain WASM).
pub struct VerifierService {
    /// Ed25519 signing key pair (secret + public)
    signing_keypair: Option<ed25519_dalek::SigningKey>,
    /// Configuration
    config: VerifierConfig,
}

/// Verifier configuration
#[derive(Debug, Clone)]
pub struct VerifierConfig {
    /// Attestation validity period in hours
    pub validity_hours: u64,
    /// Security level to use for verification
    pub security_level: SecurityLevel,
    /// Whether to verify commitment matches
    pub verify_commitment: bool,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            validity_hours: ProofAttestation::DEFAULT_VALIDITY_HOURS,
            security_level: SecurityLevel::Standard96,
            verify_commitment: true,
        }
    }
}

impl VerifierService {
    /// Create a new verifier service without signing capability
    ///
    /// Useful for verification-only scenarios
    pub fn new() -> Self {
        Self {
            signing_keypair: None,
            config: VerifierConfig::default(),
        }
    }

    /// Create with signing key
    pub fn with_signing_key(key_bytes: &[u8; 32]) -> ProofResult<Self> {
        let signing_key = ed25519_dalek::SigningKey::from_bytes(key_bytes);
        Ok(Self {
            signing_keypair: Some(signing_key),
            config: VerifierConfig::default(),
        })
    }

    /// Set configuration
    pub fn with_config(mut self, config: VerifierConfig) -> Self {
        self.config = config;
        self
    }

    /// Verify a serialized proof
    pub fn verify(&self, proof: &SerializableProof) -> ProofResult<VerificationResult> {
        use std::time::Instant;
        let start = Instant::now();

        // Deserialize and verify the STARK proof
        let vote_proof = proof.to_vote_proof()?;
        let stark_result = vote_proof.verify()?;

        let verification_time = start.elapsed();

        Ok(VerificationResult {
            valid: stark_result.valid,
            eligible: proof.eligible,
            proposal_type: proof.proposal_type,
            requirements_met: proof.requirements_met,
            active_requirements: proof.active_requirements,
            verification_time,
            error: if stark_result.valid {
                None
            } else {
                Some(stark_result.details.unwrap_or_default())
            },
        })
    }

    /// Verify and produce a signed attestation
    pub fn verify_and_attest(&self, proof: &SerializableProof) -> ProofResult<ProofAttestation> {
        use ed25519_dalek::Signer;

        let signing_keypair = self.signing_keypair.as_ref().ok_or_else(|| {
            ProofError::InvalidConfiguration("No signing key configured".to_string())
        })?;

        let result = self.verify(proof)?;
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let expires_at_ms = now_ms + (self.config.validity_hours * 60 * 60 * 1000);

        let mut attestation = ProofAttestation {
            proof_hash: proof.proof_hash(),
            voter_commitment: proof.voter_commitment.clone().try_into().map_err(|_| {
                ProofError::InvalidProofFormat("Invalid voter commitment length".to_string())
            })?,
            proposal_type: proof.proposal_type,
            verified: result.valid && proof.eligible,
            verified_at_ms: now_ms,
            expires_at_ms,
            verifier_pubkey: signing_keypair.verifying_key().to_bytes(),
            signature: [0u8; 64],
            security_level: format!("{:?}", self.config.security_level),
            verification_time_ms: result.verification_time.as_millis() as u64,
        };

        // Sign the attestation
        let signing_data = attestation.signing_data();
        let signature = signing_keypair.sign(&signing_data);
        attestation.signature = signature.to_bytes();

        Ok(attestation)
    }

    /// Verify an attestation signature
    pub fn verify_attestation(&self, attestation: &ProofAttestation) -> ProofResult<bool> {
        use ed25519_dalek::{Signature, VerifyingKey};

        let verifying_key = VerifyingKey::from_bytes(&attestation.verifier_pubkey)
            .map_err(|e| ProofError::VerificationFailed(format!("Invalid verifier key: {}", e)))?;

        let signature = Signature::from_bytes(&attestation.signature);
        let signing_data = attestation.signing_data();

        verifying_key
            .verify_strict(&signing_data, &signature)
            .map(|_| true)
            .map_err(|e| ProofError::VerificationFailed(format!("Signature verification failed: {}", e)))
    }
}

impl Default for VerifierService {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of proof verification
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether the STARK proof is valid
    pub valid: bool,
    /// Whether the voter is eligible
    pub eligible: bool,
    /// Proposal type
    pub proposal_type: u8,
    /// Requirements met
    pub requirements_met: u8,
    /// Active requirements
    pub active_requirements: u8,
    /// Time taken to verify
    pub verification_time: Duration,
    /// Error message if verification failed
    pub error: Option<String>,
}

// ============================================================================
// Zome Input/Output Types
// ============================================================================

/// Input type for store_eligibility_proof zome call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreProofInput {
    pub voter_did: String,
    pub voter_commitment: Vec<u8>,
    pub proposal_type: u8,
    pub eligible: bool,
    pub requirements_met: u8,
    pub active_requirements: u8,
    pub proof_bytes: Vec<u8>,
    pub validity_hours: Option<i64>,
}

impl From<SerializableProof> for StoreProofInput {
    fn from(proof: SerializableProof) -> Self {
        Self {
            voter_did: proof.voter_did,
            voter_commitment: proof.voter_commitment,
            proposal_type: proof.proposal_type,
            eligible: proof.eligible,
            requirements_met: proof.requirements_met,
            active_requirements: proof.active_requirements,
            proof_bytes: proof.proof_bytes,
            validity_hours: Some(24), // Default 24 hours
        }
    }
}

impl StoreProofInput {
    /// Set custom validity period
    pub fn with_validity_hours(mut self, hours: i64) -> Self {
        self.validity_hours = Some(hours);
        self
    }
}

/// Input type for cast_verified_vote zome call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CastVerifiedVoteInput {
    pub proposal_id: String,
    pub voter_did: String,
    pub tier: String, // "Basic", "Major", or "Constitutional"
    pub choice: String, // "For", "Against", or "Abstain"
    pub eligibility_proof_hash: Vec<u8>, // ActionHash as bytes
    pub voter_commitment: Vec<u8>,
    pub reason: Option<String>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_voter() -> ProofVoterProfile {
        VoterProfileBuilder::new("did:mycelix:test123")
            .assurance_level(3)
            .matl_score(0.8)
            .stake(600.0)
            .account_age_days(100)
            .participation(0.5)
            .has_humanity_proof(true)
            .fl_contributions(25)
            .build()
    }

    #[test]
    fn test_voter_profile_builder() {
        let profile = VoterProfileBuilder::new("did:test")
            .assurance_level(2)
            .matl_score(0.7)
            .stake(100.0)
            .build();

        assert_eq!(profile.did, "did:test");
        assert_eq!(profile.assurance_level, 2);
        assert_eq!(profile.matl_score, 0.7);
        assert_eq!(profile.stake, 100.0);
    }

    #[test]
    fn test_voter_profile_builder_clamping() {
        let profile = VoterProfileBuilder::new("did:test")
            .assurance_level(10) // Should clamp to 4
            .matl_score(2.0)     // Should clamp to 1.0
            .participation(-0.5) // Should clamp to 0.0
            .stake(-100.0)       // Should become 0.0
            .build();

        assert_eq!(profile.assurance_level, 4);
        assert_eq!(profile.matl_score, 1.0);
        assert_eq!(profile.participation_rate, 0.0);
        assert_eq!(profile.stake, 0.0);
    }

    #[test]
    fn test_eligibility_check() {
        let voter = test_voter();
        let client = ZomeBridgeClient::new();

        // Should be eligible for Standard
        let check = client.check_eligibility(&voter, ProofProposalType::Standard);
        assert!(check.eligible);
        assert!(check.failed_checks.is_empty());

        // Should be eligible for Constitutional (has humanity proof)
        let check = client.check_eligibility(&voter, ProofProposalType::Constitutional);
        assert!(check.eligible);

        // Should be eligible for ModelGovernance (has FL contributions)
        let check = client.check_eligibility(&voter, ProofProposalType::ModelGovernance);
        assert!(check.eligible);
    }

    #[test]
    fn test_eligibility_check_ineligible() {
        let voter = VoterProfileBuilder::new("did:test")
            .assurance_level(0) // Too low
            .build();

        let client = ZomeBridgeClient::new();
        let check = client.check_eligibility(&voter, ProofProposalType::Standard);

        assert!(!check.eligible);
        assert!(check.failed_checks.contains(&"assurance_level".to_string()));
    }

    #[test]
    fn test_proof_generation() {
        let voter = test_voter();
        let client = ZomeBridgeClient::new();

        let proof = client.generate_proof(&voter, ProofProposalType::Standard).unwrap();

        assert!(proof.eligible);
        assert_eq!(proof.voter_did, "did:mycelix:test123");
        assert!(!proof.proof_bytes.is_empty());
    }

    #[test]
    fn test_serializable_proof_json() {
        let voter = test_voter();
        let client = ZomeBridgeClient::new();

        let proof = client.generate_proof(&voter, ProofProposalType::Standard).unwrap();
        let json = proof.to_json().unwrap();

        let restored = SerializableProof::from_json(&json).unwrap();
        assert_eq!(restored.voter_did, proof.voter_did);
        assert_eq!(restored.eligible, proof.eligible);
    }

    #[test]
    fn test_verification_service() {
        let voter = test_voter();
        let client = ZomeBridgeClient::new();
        let proof = client.generate_proof(&voter, ProofProposalType::Standard).unwrap();

        let service = VerifierService::new();
        let result = service.verify(&proof).unwrap();

        assert!(result.valid);
        assert!(result.eligible);
    }

    #[test]
    fn test_verification_with_attestation() {
        let voter = test_voter();
        let client = ZomeBridgeClient::new();
        let proof = client.generate_proof(&voter, ProofProposalType::Standard).unwrap();

        // Generate a signing key
        let secret_key = [1u8; 32]; // In production, use secure random
        let service = VerifierService::with_signing_key(&secret_key).unwrap();

        let attestation = service.verify_and_attest(&proof).unwrap();

        assert!(attestation.verified);
        assert!(!attestation.is_expired());

        // Verify the attestation signature
        assert!(service.verify_attestation(&attestation).unwrap());
    }

    #[test]
    fn test_store_proof_input_conversion() {
        let voter = test_voter();
        let client = ZomeBridgeClient::new();
        let proof = client.generate_proof(&voter, ProofProposalType::Standard).unwrap();

        let input: StoreProofInput = proof.clone().into();

        assert_eq!(input.voter_did, proof.voter_did);
        assert_eq!(input.eligible, proof.eligible);
        assert_eq!(input.validity_hours, Some(24));
    }

    #[test]
    fn test_commitment_computation() {
        let voter = test_voter();
        let client = ZomeBridgeClient::new();

        let commitment1 = client.compute_commitment(&voter);
        let commitment2 = client.compute_commitment(&voter);

        assert_eq!(commitment1, commitment2);

        // Different voter -> different commitment
        let voter2 = VoterProfileBuilder::new("did:different").build();
        let commitment3 = client.compute_commitment(&voter2);
        assert_ne!(commitment1, commitment3);
    }
}
