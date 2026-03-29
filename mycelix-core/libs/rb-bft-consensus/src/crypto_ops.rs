// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integrated Cryptographic Operations for Consensus
//!
//! This module provides high-level cryptographic operations that integrate
//! VRF-based leader election, BLS aggregate signatures, and batch verification
//! into the consensus workflow.
//!
//! ## Features
//!
//! - **VRF Leader Election**: Unpredictable but verifiable leader selection
//! - **BLS Vote Aggregation**: Compress multiple vote signatures into one
//! - **Batch Verification**: Fast verification of multiple signatures
//!
//! ## Usage Flow
//!
//! ```text
//! Round Start
//!     │
//!     ▼
//! ┌──────────────────────┐
//! │  VRF Leader Election │  ← All validators compute VRF
//! │  (select_leader_vrf) │    Lowest output wins
//! └──────────┬───────────┘
//!            │
//!            ▼
//! ┌──────────────────────┐
//! │   Leader Proposes    │  ← Signed with ed25519
//! └──────────┬───────────┘
//!            │
//!            ▼
//! ┌──────────────────────┐
//! │   Validators Vote    │  ← Each vote signed
//! │  (batch_verify_votes)│    Batch verified
//! └──────────┬───────────┘
//!            │
//!            ▼
//! ┌──────────────────────┐
//! │  Aggregate Votes     │  ← BLS aggregate signatures
//! │  (aggregate_votes)   │    96 bytes total
//! └──────────┬───────────┘
//!            │
//!            ▼
//!        Commit
//! ```

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::error::{ConsensusError, ConsensusResult};
use crate::crypto::{ConsensusSignature, ValidatorKeypair};
use crate::vote::Vote;
use crate::vrf::{VrfKeypair, VrfOutput, VrfPublicKey};
use crate::batch::{BatchVerifier, BatchVerificationResult};

#[cfg(feature = "bls")]
use crate::bls::{BlsKeypair, BlsPublicKey, BlsSignature, BlsAggregateSignature, DST_VOTE};

/// Validator cryptographic identity combining all key types
#[derive(Debug)]
pub struct ValidatorCryptoIdentity {
    /// Ed25519 keypair for regular signatures
    pub ed25519: ValidatorKeypair,
    /// VRF keypair for leader election
    pub vrf: VrfKeypair,
    /// BLS keypair for aggregate signatures (optional)
    #[cfg(feature = "bls")]
    pub bls: Option<BlsKeypair>,
}

impl ValidatorCryptoIdentity {
    /// Generate a new validator identity with all key types
    pub fn generate() -> Self {
        Self {
            ed25519: ValidatorKeypair::generate(),
            vrf: VrfKeypair::generate(),
            #[cfg(feature = "bls")]
            bls: Some(BlsKeypair::generate()),
        }
    }

    /// Generate without BLS (for minimal builds)
    pub fn generate_minimal() -> Self {
        Self {
            ed25519: ValidatorKeypair::generate(),
            vrf: VrfKeypair::generate(),
            #[cfg(feature = "bls")]
            bls: None,
        }
    }

    /// Get the validator's public key identifier (ed25519 pubkey hex)
    pub fn id(&self) -> String {
        self.ed25519.public_key_hex()
    }

    /// Get the VRF public key
    pub fn vrf_public_key(&self) -> VrfPublicKey {
        self.vrf.public_key()
    }
}

/// VRF-based leader election for a round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderElectionResult {
    /// Round number
    pub round: u64,
    /// Elected leader's validator ID (ed25519 pubkey hex)
    pub leader_id: String,
    /// Leader's VRF public key
    pub leader_vrf_pubkey: VrfPublicKey,
    /// Leader's VRF output for this round
    pub vrf_output: VrfOutput,
    /// All VRF outputs from participating validators (for verification)
    pub all_outputs: Vec<(String, VrfPublicKey, VrfOutput)>,
    /// Reputation weights for weighted election (None = unweighted)
    pub reputation_weights: Option<HashMap<String, f64>>,
}

impl LeaderElectionResult {
    /// Verify the leader election was correct
    pub fn verify(&self) -> ConsensusResult<()> {
        // Verify the leader's VRF output
        self.vrf_output.verify_round(self.round)?;

        // Verify all outputs for the round
        for (_, _, output) in &self.all_outputs {
            output.verify_round(self.round)?;
        }

        // Check election criteria based on whether weights were used
        match &self.reputation_weights {
            Some(weights) => self.verify_weighted(weights),
            None => self.verify_unweighted(),
        }
    }

    /// Verify unweighted election (lowest raw VRF output wins)
    fn verify_unweighted(&self) -> ConsensusResult<()> {
        let leader_value = self.vrf_output.output_as_u128();

        for (id, _pk, output) in &self.all_outputs {
            if output.output_as_u128() < leader_value && id != &self.leader_id {
                return Err(ConsensusError::InvalidSignature {
                    reason: format!(
                        "Validator {} has lower VRF output than claimed leader {}",
                        id, self.leader_id
                    ),
                });
            }
        }

        Ok(())
    }

    /// Verify weighted election (lowest VRF/reputation² wins)
    fn verify_weighted(&self, weights: &HashMap<String, f64>) -> ConsensusResult<()> {
        let leader_rep = weights.get(&self.leader_id)
            .ok_or_else(|| ConsensusError::InvalidSignature {
                reason: format!("Missing reputation weight for leader {}", self.leader_id),
            })?;

        if *leader_rep <= 0.0 {
            return Err(ConsensusError::InvalidSignature {
                reason: "Leader has zero or negative reputation".to_string(),
            });
        }

        let leader_priority = self.vrf_output.output_as_f64() / (leader_rep * leader_rep);

        for (id, _pk, output) in &self.all_outputs {
            if id == &self.leader_id {
                continue;
            }

            if let Some(rep) = weights.get(id) {
                if *rep > 0.0 {
                    let priority = output.output_as_f64() / (rep * rep);
                    if priority < leader_priority {
                        return Err(ConsensusError::InvalidSignature {
                            reason: format!(
                                "Validator {} has lower weighted priority than claimed leader {}",
                                id, self.leader_id
                            ),
                        });
                    }
                }
            }
        }

        Ok(())
    }
}

/// Conduct VRF-based leader election
///
/// Each validator computes their VRF output for the round.
/// The validator with the lowest output becomes leader.
pub fn elect_leader_vrf(
    round: u64,
    validators: &[(String, &VrfKeypair)], // (validator_id, vrf_keypair)
) -> ConsensusResult<LeaderElectionResult> {
    if validators.is_empty() {
        return Err(ConsensusError::InsufficientValidators { have: 0, need: 1 });
    }

    let all_outputs: Vec<(String, VrfPublicKey, VrfOutput)> = validators
        .iter()
        .map(|(id, keypair)| {
            let output = keypair.evaluate_round(round);
            (id.clone(), keypair.public_key(), output)
        })
        .collect();

    // Find validator with lowest VRF output
    let (leader_id, leader_vrf_pubkey, vrf_output) = all_outputs
        .iter()
        .min_by_key(|(_, _, output)| output.output_as_u128())
        .cloned()
        .expect("validators is non-empty");

    Ok(LeaderElectionResult {
        round,
        leader_id,
        leader_vrf_pubkey,
        vrf_output,
        all_outputs,
        reputation_weights: None,
    })
}

/// Conduct weighted VRF leader election using reputation
///
/// Validators with higher reputation have proportionally better chances.
/// Effective priority = VRF_output / reputation²
pub fn elect_leader_vrf_weighted(
    round: u64,
    validators: &[(String, &VrfKeypair, f64)], // (validator_id, vrf_keypair, reputation)
) -> ConsensusResult<LeaderElectionResult> {
    if validators.is_empty() {
        return Err(ConsensusError::InsufficientValidators { have: 0, need: 1 });
    }

    let all_outputs: Vec<(String, VrfPublicKey, VrfOutput)> = validators
        .iter()
        .map(|(id, keypair, _rep)| {
            let output = keypair.evaluate_round(round);
            (id.clone(), keypair.public_key(), output)
        })
        .collect();

    // Build reputation weights map for verification
    let reputation_weights: HashMap<String, f64> = validators
        .iter()
        .map(|(id, _, rep)| (id.clone(), *rep))
        .collect();

    // Find validator with lowest weighted priority
    // priority = vrf_output / reputation²
    let (leader_id, leader_vrf_pubkey, vrf_output) = validators
        .iter()
        .zip(all_outputs.iter())
        .filter(|((_, _, rep), _)| *rep > 0.0)
        .min_by(|((_, _, rep_a), (_, _, out_a)), ((_, _, rep_b), (_, _, out_b))| {
            let priority_a = out_a.output_as_f64() / (rep_a * rep_a);
            let priority_b = out_b.output_as_f64() / (rep_b * rep_b);
            priority_a.partial_cmp(&priority_b).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(_, (id, pk, out))| (id.clone(), pk.clone(), out.clone()))
        .ok_or(ConsensusError::InsufficientValidators { have: 0, need: 1 })?;

    Ok(LeaderElectionResult {
        round,
        leader_id,
        leader_vrf_pubkey,
        vrf_output,
        all_outputs,
        reputation_weights: Some(reputation_weights),
    })
}

/// Batch verify multiple votes
///
/// Returns detailed results about which votes are valid/invalid.
pub fn batch_verify_vote_signatures(
    votes: &[Vote],
    messages: &[Vec<u8>], // Pre-computed signable data for each vote
) -> ConsensusResult<BatchVerificationResult> {
    if votes.len() != messages.len() {
        return Err(ConsensusError::Internal(
            "Vote and message count mismatch".to_string(),
        ));
    }

    let mut batch = BatchVerifier::with_capacity(votes.len());

    for (i, (vote, msg)) in votes.iter().zip(messages.iter()).enumerate() {
        // Get signature and public key from vote
        if vote.signature.signature.is_empty() {
            return Err(ConsensusError::InvalidVote {
                validator: vote.voter.clone(),
                reason: "Vote is unsigned".to_string(),
            });
        }

        let pubkey: [u8; 32] = vote.signature.signer_pubkey.as_slice()
            .try_into()
            .map_err(|_| ConsensusError::InvalidVote {
                validator: vote.voter.clone(),
                reason: "Invalid public key size".to_string(),
            })?;

        let sig: [u8; 64] = vote.signature.signature.as_slice()
            .try_into()
            .map_err(|_| ConsensusError::InvalidVote {
                validator: vote.voter.clone(),
                reason: "Invalid signature size".to_string(),
            })?;

        batch.add_signature_labeled(&pubkey, &sig, msg, format!("vote_{}", i));
    }

    batch.verify()
}

/// Aggregated vote result with BLS signature
#[cfg(feature = "bls")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedVotes {
    /// Round number
    pub round: u64,
    /// Proposal hash being voted on
    pub proposal_hash: String,
    /// Aggregated BLS signature (96 bytes)
    pub aggregate_signature: BlsAggregateSignature,
    /// Public keys of signers (for verification)
    pub signer_pubkeys: Vec<BlsPublicKey>,
    /// Validator IDs who signed
    pub signer_ids: Vec<String>,
    /// Total weighted vote
    pub total_weight: f32,
    /// Number of approvals
    pub approval_count: usize,
    /// Number of rejections
    pub rejection_count: usize,
}

#[cfg(feature = "bls")]
impl AggregatedVotes {
    /// Verify the aggregate signature
    pub fn verify(&self, message: &[u8]) -> ConsensusResult<()> {
        crate::bls::verify_aggregate_same_message(
            &self.aggregate_signature,
            message,
            DST_VOTE,
            &self.signer_pubkeys,
        )
    }

    /// Check if consensus threshold is met
    pub fn meets_threshold(&self, threshold: f32) -> bool {
        self.total_weight >= threshold
    }

    /// Get the number of signers
    pub fn signer_count(&self) -> usize {
        self.signer_ids.len()
    }
}

/// BLS signature for a vote (from a validator with BLS keypair)
#[cfg(feature = "bls")]
#[derive(Debug, Clone)]
pub struct BlsVoteSignature {
    /// Validator ID
    pub validator_id: String,
    /// BLS public key
    pub bls_pubkey: BlsPublicKey,
    /// BLS signature
    pub signature: BlsSignature,
    /// Whether approving
    pub approve: bool,
    /// Vote weight
    pub weight: f32,
}

/// Aggregate BLS vote signatures
///
/// Combines multiple BLS signatures into a single 96-byte aggregate.
#[cfg(feature = "bls")]
pub fn aggregate_vote_signatures(
    round: u64,
    proposal_hash: &str,
    vote_signatures: &[BlsVoteSignature],
) -> ConsensusResult<AggregatedVotes> {
    if vote_signatures.is_empty() {
        return Err(ConsensusError::InvalidVote {
            validator: "none".to_string(),
            reason: "No votes to aggregate".to_string(),
        });
    }

    let signatures: Vec<BlsSignature> = vote_signatures.iter()
        .map(|vs| vs.signature.clone())
        .collect();

    let aggregate_signature = crate::bls::aggregate_signatures(&signatures)?;

    let signer_pubkeys: Vec<BlsPublicKey> = vote_signatures.iter()
        .map(|vs| vs.bls_pubkey.clone())
        .collect();

    let signer_ids: Vec<String> = vote_signatures.iter()
        .map(|vs| vs.validator_id.clone())
        .collect();

    let total_weight: f32 = vote_signatures.iter()
        .filter(|vs| vs.approve)
        .map(|vs| vs.weight)
        .sum();

    let approval_count = vote_signatures.iter().filter(|vs| vs.approve).count();
    let rejection_count = vote_signatures.iter().filter(|vs| !vs.approve).count();

    Ok(AggregatedVotes {
        round,
        proposal_hash: proposal_hash.to_string(),
        aggregate_signature,
        signer_pubkeys,
        signer_ids,
        total_weight,
        approval_count,
        rejection_count,
    })
}

/// Combined commit proof containing both ed25519 and optional BLS signatures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitProof {
    /// Round number
    pub round: u64,
    /// Committed proposal hash
    pub proposal_hash: String,
    /// Individual ed25519 signatures from voters
    pub signatures: Vec<(String, ConsensusSignature)>, // (validator_id, signature)
    /// Aggregated BLS signature (if BLS enabled)
    #[cfg(feature = "bls")]
    pub bls_aggregate: Option<AggregatedVotes>,
    /// Total vote weight
    pub total_weight: f32,
    /// Number of signers
    pub signer_count: usize,
}

impl CommitProof {
    /// Create from votes with ed25519 signatures only
    pub fn from_votes(round: u64, proposal_hash: String, votes: &[Vote]) -> Self {
        let signatures: Vec<(String, ConsensusSignature)> = votes
            .iter()
            .map(|v| (v.voter.clone(), v.signature.clone()))
            .collect();

        let total_weight: f32 = votes.iter()
            .filter(|v| v.decision.is_approval())
            .map(|v| v.reputation.powi(2))
            .sum();

        Self {
            round,
            proposal_hash,
            signer_count: signatures.len(),
            signatures,
            #[cfg(feature = "bls")]
            bls_aggregate: None,
            total_weight,
        }
    }

    /// Add BLS aggregate signature
    #[cfg(feature = "bls")]
    pub fn with_bls_aggregate(mut self, aggregate: AggregatedVotes) -> Self {
        self.bls_aggregate = Some(aggregate);
        self
    }

    /// Verify the commit proof
    pub fn verify(&self, vote_messages: &HashMap<String, Vec<u8>>) -> ConsensusResult<()> {
        // Verify individual signatures
        let mut batch = BatchVerifier::with_capacity(self.signatures.len());

        for (validator_id, sig) in &self.signatures {
            let message = vote_messages.get(validator_id)
                .ok_or_else(|| ConsensusError::InvalidSignature {
                    reason: format!("Missing message for validator {}", validator_id),
                })?;

            let pubkey: [u8; 32] = sig.signer_pubkey.as_slice()
                .try_into()
                .map_err(|_| ConsensusError::InvalidSignature {
                    reason: "Invalid public key size".to_string(),
                })?;

            let signature: [u8; 64] = sig.signature.as_slice()
                .try_into()
                .map_err(|_| ConsensusError::InvalidSignature {
                    reason: "Invalid signature size".to_string(),
                })?;

            batch.add_signature_labeled(&pubkey, &signature, message, validator_id.clone());
        }

        let result = batch.verify()?;
        if !result.all_valid {
            return Err(ConsensusError::InvalidSignature {
                reason: format!(
                    "Commit proof verification failed: {} invalid signatures",
                    result.invalid_count
                ),
            });
        }

        Ok(())
    }

    /// Get the size in bytes (for bandwidth estimation)
    pub fn size_bytes(&self) -> usize {
        // Each ed25519 signature: 64 bytes + 32 bytes pubkey = 96 bytes
        let ed25519_size = self.signatures.len() * 96;

        #[cfg(feature = "bls")]
        let bls_size = if self.bls_aggregate.is_some() {
            96 // Single aggregate signature
        } else {
            0
        };

        #[cfg(not(feature = "bls"))]
        let bls_size = 0;

        ed25519_size + bls_size + 8 + 64 // + round + proposal_hash
    }

    /// Verify BLS aggregate signature if present
    #[cfg(feature = "bls")]
    pub fn verify_bls(&self, message: &[u8]) -> ConsensusResult<()> {
        if let Some(ref aggregate) = self.bls_aggregate {
            aggregate.verify(message)?;
        }
        Ok(())
    }
}

/// Create a BLS vote signature from a vote (for validators with BLS keys)
///
/// This allows validators to provide both ed25519 and BLS signatures for their votes.
#[cfg(feature = "bls")]
pub fn sign_vote_bls(
    vote: &Vote,
    bls_keypair: &BlsKeypair,
) -> BlsVoteSignature {
    // Create the message to sign (same as ed25519 but for BLS domain)
    let message = format!(
        "vote:{}:{}:{}:{}",
        vote.round,
        vote.proposal_id,
        vote.voter,
        if vote.decision.is_approval() { "approve" } else { "reject" }
    );

    BlsVoteSignature {
        validator_id: vote.voter.clone(),
        bls_pubkey: bls_keypair.public_key(),
        signature: bls_keypair.sign(message.as_bytes(), DST_VOTE),
        approve: vote.decision.is_approval(),
        weight: vote.weight,
    }
}

/// Build a commit proof from votes, optionally with BLS aggregation
///
/// If BLS keypairs are provided for the validators, this will create an aggregate
/// BLS signature in addition to the ed25519 signatures.
#[cfg(feature = "bls")]
pub fn build_commit_proof_with_bls(
    round: u64,
    proposal_hash: String,
    votes: &[Vote],
    bls_keypairs: Option<&HashMap<String, BlsKeypair>>,
) -> ConsensusResult<CommitProof> {
    let mut proof = CommitProof::from_votes(round, proposal_hash.clone(), votes);

    if let Some(keypairs) = bls_keypairs {
        // Create BLS signatures for votes where we have keypairs
        let mut bls_sigs: Vec<BlsVoteSignature> = Vec::new();

        for vote in votes {
            if let Some(kp) = keypairs.get(&vote.voter) {
                bls_sigs.push(sign_vote_bls(vote, kp));
            }
        }

        // Only aggregate if we have signatures
        if !bls_sigs.is_empty() {
            let aggregate = aggregate_vote_signatures(round, &proposal_hash, &bls_sigs)?;
            proof = proof.with_bls_aggregate(aggregate);
        }
    }

    Ok(proof)
}

/// Build a commit proof from votes without BLS (for non-BLS builds)
#[cfg(not(feature = "bls"))]
pub fn build_commit_proof_with_bls(
    round: u64,
    proposal_hash: String,
    votes: &[Vote],
    _bls_keypairs: Option<&HashMap<String, ()>>,
) -> ConsensusResult<CommitProof> {
    Ok(CommitProof::from_votes(round, proposal_hash, votes))
}

/// Consensus round commit data with cryptographic proofs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusCommit {
    /// Round number
    pub round: u64,
    /// Proposal that was committed
    pub proposal_hash: String,
    /// Result hash of the committed content
    pub result_hash: String,
    /// Cryptographic proof of consensus
    pub proof: CommitProof,
    /// Leader who proposed
    pub leader_id: String,
    /// Total validators in round
    pub validator_count: usize,
    /// Timestamp of commit
    pub committed_at: u64,
}

impl ConsensusCommit {
    /// Create a new consensus commit
    pub fn new(
        round: u64,
        proposal_hash: String,
        result_hash: String,
        proof: CommitProof,
        leader_id: String,
        validator_count: usize,
    ) -> Self {
        Self {
            round,
            proposal_hash,
            result_hash,
            proof,
            leader_id,
            validator_count,
            committed_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }

    /// Get vote count from proof
    pub fn vote_count(&self) -> usize {
        self.proof.signer_count
    }

    /// Get total weight from proof
    pub fn total_weight(&self) -> f32 {
        self.proof.total_weight
    }

    /// Check if BLS aggregate is present
    #[cfg(feature = "bls")]
    pub fn has_bls_aggregate(&self) -> bool {
        self.proof.bls_aggregate.is_some()
    }

    /// Verify the commit
    pub fn verify(&self, vote_messages: &HashMap<String, Vec<u8>>) -> ConsensusResult<()> {
        self.proof.verify(vote_messages)
    }
}

/// Statistics for cryptographic operations
#[derive(Debug, Clone, Default)]
pub struct CryptoOpsStats {
    /// VRF evaluations performed
    pub vrf_evaluations: u64,
    /// Batch verifications performed
    pub batch_verifications: u64,
    /// Total signatures batch verified
    pub signatures_verified: u64,
    /// BLS aggregations performed
    #[cfg(feature = "bls")]
    pub bls_aggregations: u64,
    /// Total bytes saved by BLS aggregation
    #[cfg(feature = "bls")]
    pub bytes_saved_by_bls: u64,
}

impl CryptoOpsStats {
    /// Record a VRF evaluation
    pub fn record_vrf(&mut self) {
        self.vrf_evaluations += 1;
    }

    /// Record a batch verification
    pub fn record_batch_verify(&mut self, signature_count: usize) {
        self.batch_verifications += 1;
        self.signatures_verified += signature_count as u64;
    }

    /// Record BLS aggregation
    #[cfg(feature = "bls")]
    pub fn record_bls_aggregation(&mut self, signature_count: usize) {
        self.bls_aggregations += 1;
        // Each individual sig is 64 bytes, aggregate is 96 bytes
        // Savings = n * 64 - 96 bytes
        if signature_count > 1 {
            self.bytes_saved_by_bls += (signature_count as u64 * 64).saturating_sub(96);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_crypto_identity() {
        let identity = ValidatorCryptoIdentity::generate();

        // Should have all keys
        assert!(!identity.id().is_empty());
        assert_eq!(identity.vrf_public_key().as_bytes().len(), 32);
    }

    #[test]
    fn test_vrf_leader_election() {
        let validators: Vec<(String, VrfKeypair)> = (0..5)
            .map(|i| (format!("validator_{}", i), VrfKeypair::generate()))
            .collect();

        let validator_refs: Vec<(String, &VrfKeypair)> = validators
            .iter()
            .map(|(id, kp)| (id.clone(), kp))
            .collect();

        let result = elect_leader_vrf(42, &validator_refs).unwrap();

        // Verify the result
        assert!(result.verify().is_ok());
        assert_eq!(result.round, 42);
        assert!(!result.leader_id.is_empty());
        assert_eq!(result.all_outputs.len(), 5);
    }

    #[test]
    fn test_vrf_leader_election_weighted() {
        let validators: Vec<(String, VrfKeypair, f64)> = (0..5)
            .map(|i| {
                let rep = 0.5 + (i as f64) * 0.1; // 0.5, 0.6, 0.7, 0.8, 0.9
                (format!("validator_{}", i), VrfKeypair::generate(), rep)
            })
            .collect();

        let validator_refs: Vec<(String, &VrfKeypair, f64)> = validators
            .iter()
            .map(|(id, kp, rep)| (id.clone(), kp, *rep))
            .collect();

        let result = elect_leader_vrf_weighted(100, &validator_refs).unwrap();

        assert!(result.verify().is_ok());
        assert_eq!(result.round, 100);
    }

    #[test]
    fn test_vrf_leader_deterministic() {
        // Same validator set should elect same leader for same round
        let keypairs: Vec<VrfKeypair> = (0..3).map(|_| VrfKeypair::generate()).collect();

        let validators: Vec<(String, &VrfKeypair)> = keypairs
            .iter()
            .enumerate()
            .map(|(i, kp)| (format!("v{}", i), kp))
            .collect();

        let result1 = elect_leader_vrf(42, &validators).unwrap();
        let result2 = elect_leader_vrf(42, &validators).unwrap();

        assert_eq!(result1.leader_id, result2.leader_id);
        assert_eq!(result1.vrf_output.output, result2.vrf_output.output);
    }

    #[test]
    fn test_commit_proof_from_votes() {
        let keypair = ValidatorKeypair::generate();
        let mut vote = Vote::approve(
            "proposal-123".to_string(),
            1,
            keypair.public_key_hex(),
            0.8,
        );
        vote.sign(&keypair).unwrap();

        let proof = CommitProof::from_votes(1, "proposal-hash".to_string(), &[vote]);

        assert_eq!(proof.round, 1);
        assert_eq!(proof.signer_count, 1);
        assert!(proof.total_weight > 0.0);
    }

    #[test]
    fn test_crypto_ops_stats() {
        let mut stats = CryptoOpsStats::default();

        stats.record_vrf();
        stats.record_vrf();
        stats.record_batch_verify(10);
        stats.record_batch_verify(20);

        assert_eq!(stats.vrf_evaluations, 2);
        assert_eq!(stats.batch_verifications, 2);
        assert_eq!(stats.signatures_verified, 30);
    }

    #[cfg(feature = "bls")]
    #[test]
    fn test_bls_aggregation() {
        let keypairs: Vec<BlsKeypair> = (0..5).map(|_| BlsKeypair::generate()).collect();

        let message = b"proposal:abc123:approve";

        let vote_sigs: Vec<BlsVoteSignature> = keypairs
            .iter()
            .enumerate()
            .map(|(i, kp)| {
                BlsVoteSignature {
                    validator_id: format!("validator_{}", i),
                    bls_pubkey: kp.public_key(),
                    signature: kp.sign(message, DST_VOTE),
                    approve: true,
                    weight: 0.64, // 0.8²
                }
            })
            .collect();

        let aggregated = aggregate_vote_signatures(1, "abc123", &vote_sigs).unwrap();

        assert_eq!(aggregated.signer_count(), 5);
        assert!(aggregated.verify(message).is_ok());
        assert!((aggregated.total_weight - 3.2).abs() < 0.01); // 5 * 0.64

        // Test size savings
        let individual_size = 5 * 64; // 5 signatures * 64 bytes
        let aggregate_size = 96; // BLS aggregate
        assert!(aggregate_size < individual_size);
    }

    #[cfg(feature = "bls")]
    #[test]
    fn test_sign_vote_bls() {
        let ed_keypair = ValidatorKeypair::generate();
        let bls_keypair = BlsKeypair::generate();

        let mut vote = Vote::approve(
            "proposal-123".to_string(),
            1,
            ed_keypair.public_key_hex(),
            0.8,
        );
        vote.sign(&ed_keypair).unwrap();

        let bls_sig = sign_vote_bls(&vote, &bls_keypair);

        assert_eq!(bls_sig.validator_id, vote.voter);
        assert!(bls_sig.approve);
        assert!((bls_sig.weight - 0.64).abs() < 0.01); // 0.8²
    }

    #[cfg(feature = "bls")]
    #[test]
    fn test_build_commit_proof_with_bls() {
        // Create validators and collect their IDs first
        let ed_keypairs: Vec<ValidatorKeypair> = (0..5)
            .map(|_| ValidatorKeypair::generate())
            .collect();

        let ids: Vec<String> = ed_keypairs.iter()
            .map(|kp| kp.public_key_hex())
            .collect();

        // Create BLS keypairs separately (can't clone them for security)
        let bls_keypairs_vec: Vec<BlsKeypair> = (0..5)
            .map(|_| BlsKeypair::generate())
            .collect();

        // Build the hashmap by moving keypairs
        let mut bls_keypairs: HashMap<String, BlsKeypair> = HashMap::new();
        for (id, bls_kp) in ids.iter().zip(bls_keypairs_vec.into_iter()) {
            bls_keypairs.insert(id.clone(), bls_kp);
        }

        // Create and sign votes
        let mut votes: Vec<Vote> = Vec::new();
        for ed_kp in &ed_keypairs {
            let mut vote = Vote::approve(
                "proposal-123".to_string(),
                1,
                ed_kp.public_key_hex(),
                0.8,
            );
            vote.sign(ed_kp).unwrap();
            votes.push(vote);
        }

        // Build commit proof with BLS
        let proof = build_commit_proof_with_bls(
            1,
            "proposal-123".to_string(),
            &votes,
            Some(&bls_keypairs),
        ).unwrap();

        assert_eq!(proof.round, 1);
        assert_eq!(proof.signer_count, 5);
        assert!(proof.bls_aggregate.is_some());

        let aggregate = proof.bls_aggregate.as_ref().unwrap();
        assert_eq!(aggregate.signer_count(), 5);
        assert_eq!(aggregate.approval_count, 5);
    }

    #[cfg(feature = "bls")]
    #[test]
    fn test_consensus_commit() {
        let keypair = ValidatorKeypair::generate();
        let mut vote = Vote::approve(
            "proposal-123".to_string(),
            1,
            keypair.public_key_hex(),
            0.8,
        );
        vote.sign(&keypair).unwrap();

        let proof = CommitProof::from_votes(1, "proposal-123".to_string(), &[vote]);

        let commit = ConsensusCommit::new(
            1,
            "proposal-123".to_string(),
            "result-hash-abc".to_string(),
            proof,
            keypair.public_key_hex(),
            5,
        );

        assert_eq!(commit.round, 1);
        assert_eq!(commit.vote_count(), 1);
        assert!(commit.total_weight() > 0.0);
        assert!(!commit.has_bls_aggregate());
    }

    #[test]
    fn test_commit_proof_size_estimation() {
        let keypair = ValidatorKeypair::generate();
        let mut vote = Vote::approve(
            "proposal-123".to_string(),
            1,
            keypair.public_key_hex(),
            0.8,
        );
        vote.sign(&keypair).unwrap();

        let proof = CommitProof::from_votes(1, "proposal-123".to_string(), &[vote]);

        // 1 signature * 96 bytes + 8 (round) + 64 (hash) = 168
        let expected_min = 96 + 8 + 64;
        assert!(proof.size_bytes() >= expected_min);
    }
}
