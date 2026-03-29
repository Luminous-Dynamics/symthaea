// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! FROST Threshold Signatures for Consensus
//!
//! This module integrates FROST (Flexible Round-Optimized Schnorr Threshold)
//! signatures into the RB-BFT consensus protocol, enabling distributed
//! signing where t-of-n validators must cooperate.
//!
//! ## Use Cases in Consensus
//!
//! 1. **Distributed Block Finalization**: Multiple validators sign block commits
//! 2. **Threshold Cryptography**: No single validator controls signing
//! 3. **Byzantine Resilience**: Can tolerate f faulty validators in t-of-n scheme
//!
//! ## Protocol Flow
//!
//! 1. **Setup**: DKG or dealer generates key shares for n validators
//! 2. **Round 1**: Validators generate and share nonce commitments
//! 3. **Round 2**: Validators generate signature shares
//! 4. **Aggregation**: Coordinator combines t shares into final signature
//! 5. **Verification**: Standard Ed25519 verification with group public key

use frost_ed25519 as frost;
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::error::{ConsensusError, ConsensusResult};

/// Threshold configuration for consensus
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThresholdConfig {
    /// Minimum signers required (t)
    pub threshold: u16,
    /// Total number of participants (n)
    pub max_signers: u16,
}

impl ThresholdConfig {
    /// Create a new threshold configuration
    pub fn new(threshold: u16, max_signers: u16) -> ConsensusResult<Self> {
        if threshold < 2 {
            return Err(ConsensusError::InvalidConfiguration {
                reason: "Threshold must be at least 2".to_string(),
            });
        }
        if threshold > max_signers {
            return Err(ConsensusError::InvalidConfiguration {
                reason: "Threshold cannot exceed max signers".to_string(),
            });
        }
        Ok(Self { threshold, max_signers })
    }

    /// 2-of-3 threshold (minimal)
    pub fn two_of_three() -> Self {
        Self { threshold: 2, max_signers: 3 }
    }

    /// 3-of-5 threshold (common)
    pub fn three_of_five() -> Self {
        Self { threshold: 3, max_signers: 5 }
    }

    /// 5-of-7 threshold (high availability)
    pub fn five_of_seven() -> Self {
        Self { threshold: 5, max_signers: 7 }
    }

    /// 11-of-21 threshold (large committee)
    pub fn eleven_of_twenty_one() -> Self {
        Self { threshold: 11, max_signers: 21 }
    }

    /// Calculate Byzantine tolerance for this configuration
    /// Returns the fraction of validators that can be Byzantine
    pub fn byzantine_tolerance(&self) -> f64 {
        // With t-of-n threshold, we can tolerate n-t Byzantine nodes
        // (as long as t honest nodes remain)
        let faulty = self.max_signers - self.threshold;
        faulty as f64 / self.max_signers as f64
    }
}

/// A validator's threshold key share
#[derive(Clone)]
pub struct ValidatorKeyShare {
    /// Unique identifier for this validator
    pub identifier: frost::Identifier,
    /// Secret signing share
    signing_share: frost::keys::SigningShare,
    /// Public verifying share
    pub verifying_share: frost::keys::VerifyingShare,
    /// Group's public key
    group_public_key: frost::VerifyingKey,
    /// Threshold for signing
    min_signers: u16,
}

impl ValidatorKeyShare {
    /// Get the key package for signing
    pub fn key_package(&self) -> frost::keys::KeyPackage {
        frost::keys::KeyPackage::new(
            self.identifier,
            self.signing_share.clone(),
            self.verifying_share.clone(),
            self.group_public_key.clone(),
            self.min_signers,
        )
    }
}

impl std::fmt::Debug for ValidatorKeyShare {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidatorKeyShare")
            .field("identifier", &format!("{:?}", self.identifier))
            .field("verifying_share", &"[...]")
            .field("signing_share", &"[REDACTED]")
            .finish()
    }
}

/// Group public key for threshold verification
#[derive(Clone, Debug)]
pub struct GroupPublicKey {
    /// The group's verifying key
    verifying_key: frost::VerifyingKey,
}

impl GroupPublicKey {
    /// Create from verifying key
    pub fn new(verifying_key: frost::VerifyingKey) -> Self {
        Self { verifying_key }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> ConsensusResult<Vec<u8>> {
        self.verifying_key.serialize()
            .map_err(|e| ConsensusError::InvalidSignature {
                reason: format!("Failed to serialize group public key: {:?}", e),
            })
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> ConsensusResult<Self> {
        let verifying_key = frost::VerifyingKey::deserialize(bytes)
            .map_err(|e| ConsensusError::InvalidSignature {
                reason: format!("Invalid group public key: {:?}", e),
            })?;
        Ok(Self { verifying_key })
    }

    /// Verify a threshold signature
    pub fn verify(&self, message: &[u8], signature: &ThresholdSignature) -> ConsensusResult<()> {
        let frost_sig = frost::Signature::deserialize(&signature.signature)
            .map_err(|e| ConsensusError::InvalidSignature {
                reason: format!("Invalid signature format: {:?}", e),
            })?;

        self.verifying_key.verify(message, &frost_sig)
            .map_err(|e| ConsensusError::InvalidSignature {
                reason: format!("Threshold signature verification failed: {:?}", e),
            })
    }
}

/// Threshold signature result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThresholdSignature {
    /// The aggregated signature (64 bytes, Ed25519 compatible)
    pub signature: Vec<u8>,
    /// Number of signers who contributed
    pub signer_count: usize,
    /// Identifiers of participating signers
    pub signer_ids: Vec<u16>,
}

impl ThresholdSignature {
    /// Check if this signature meets a threshold requirement
    pub fn meets_threshold(&self, threshold: u16) -> bool {
        self.signer_count >= threshold as usize
    }
}

/// Generate key shares using trusted dealer
///
/// For testing and bootstrap only. Production should use DKG.
pub fn generate_shares_with_dealer(
    config: &ThresholdConfig,
) -> ConsensusResult<(Vec<ValidatorKeyShare>, GroupPublicKey, frost::keys::PublicKeyPackage)> {
    let (shares_map, pubkey_package) = frost::keys::generate_with_dealer(
        config.max_signers,
        config.threshold,
        frost::keys::IdentifierList::Default,
        &mut OsRng,
    ).map_err(|e| ConsensusError::InvalidConfiguration {
        reason: format!("Key generation failed: {:?}", e),
    })?;

    let group_public = GroupPublicKey::new(pubkey_package.verifying_key().clone());

    let shares: Vec<ValidatorKeyShare> = shares_map
        .into_iter()
        .map(|(id, secret_share)| {
            ValidatorKeyShare {
                identifier: id,
                signing_share: secret_share.signing_share().clone(),
                verifying_share: pubkey_package.verifying_shares().get(&id)
                    .expect("verifying share must exist for every generated share id").clone(),
                group_public_key: pubkey_package.verifying_key().clone(),
                min_signers: config.threshold,
            }
        })
        .collect();

    Ok((shares, group_public, pubkey_package))
}

/// A participant in threshold signing for consensus
pub struct ThresholdSigner {
    /// Key package
    key_package: frost::keys::KeyPackage,
    /// Current signing nonces
    signing_nonces: Option<frost::round1::SigningNonces>,
}

impl ThresholdSigner {
    /// Create from key share
    pub fn new(key_share: &ValidatorKeyShare) -> Self {
        Self {
            key_package: key_share.key_package(),
            signing_nonces: None,
        }
    }

    /// Get identifier
    pub fn identifier(&self) -> frost::Identifier {
        *self.key_package.identifier()
    }

    /// Round 1: Generate nonce commitment
    pub fn generate_commitment(&mut self) -> frost::round1::SigningCommitments {
        let (nonces, commitments) = frost::round1::commit(
            self.key_package.signing_share(),
            &mut OsRng,
        );
        self.signing_nonces = Some(nonces);
        commitments
    }

    /// Round 2: Generate signature share
    pub fn sign(
        &self,
        signing_package: &frost::SigningPackage,
    ) -> ConsensusResult<frost::round2::SignatureShare> {
        let nonces = self.signing_nonces.as_ref()
            .ok_or_else(|| ConsensusError::InvalidSignature {
                reason: "Must call generate_commitment first".to_string(),
            })?;

        frost::round2::sign(signing_package, nonces, &self.key_package)
            .map_err(|e| ConsensusError::InvalidSignature {
                reason: format!("Signing failed: {:?}", e),
            })
    }
}

/// Coordinator for threshold signing sessions
pub struct ThresholdCoordinator {
    /// Message being signed
    message: Vec<u8>,
    /// Collected nonce commitments
    commitments: BTreeMap<frost::Identifier, frost::round1::SigningCommitments>,
    /// Collected signature shares
    shares: BTreeMap<frost::Identifier, frost::round2::SignatureShare>,
    /// Public key package
    pubkey_package: frost::keys::PublicKeyPackage,
}

impl ThresholdCoordinator {
    /// Start a new signing session
    pub fn new(message: &[u8], pubkey_package: frost::keys::PublicKeyPackage) -> Self {
        Self {
            message: message.to_vec(),
            commitments: BTreeMap::new(),
            shares: BTreeMap::new(),
            pubkey_package,
        }
    }

    /// Add a nonce commitment from a participant
    pub fn add_commitment(
        &mut self,
        identifier: frost::Identifier,
        commitment: frost::round1::SigningCommitments,
    ) {
        self.commitments.insert(identifier, commitment);
    }

    /// Check if we have enough commitments
    pub fn has_enough_commitments(&self, threshold: u16) -> bool {
        self.commitments.len() >= threshold as usize
    }

    /// Create the signing package for round 2
    pub fn create_signing_package(&self) -> ConsensusResult<frost::SigningPackage> {
        Ok(frost::SigningPackage::new(self.commitments.clone(), &self.message))
    }

    /// Add a signature share from a participant
    pub fn add_signature_share(
        &mut self,
        identifier: frost::Identifier,
        share: frost::round2::SignatureShare,
    ) {
        self.shares.insert(identifier, share);
    }

    /// Check if we have enough shares
    pub fn has_enough_shares(&self, threshold: u16) -> bool {
        self.shares.len() >= threshold as usize
    }

    /// Aggregate signature shares into final signature
    pub fn aggregate(
        &self,
        signing_package: &frost::SigningPackage,
    ) -> ConsensusResult<ThresholdSignature> {
        let signature = frost::aggregate(
            signing_package,
            &self.shares,
            &self.pubkey_package,
        ).map_err(|e| ConsensusError::InvalidSignature {
            reason: format!("Signature aggregation failed: {:?}", e),
        })?;

        // Extract signer IDs
        let signer_ids: Vec<u16> = self.shares.keys()
            .map(|id| {
                // Convert identifier to u16 (FROST identifiers are 1-indexed)
                let bytes = id.serialize();
                u16::from_le_bytes([bytes[0], bytes[1]])
            })
            .collect();

        let sig_bytes = signature.serialize()
            .map_err(|e| ConsensusError::InvalidSignature {
                reason: format!("Failed to serialize signature: {:?}", e),
            })?;

        Ok(ThresholdSignature {
            signature: sig_bytes,
            signer_count: self.shares.len(),
            signer_ids,
        })
    }

    /// Get the number of collected commitments
    pub fn commitment_count(&self) -> usize {
        self.commitments.len()
    }

    /// Get the number of collected shares
    pub fn share_count(&self) -> usize {
        self.shares.len()
    }
}

/// State of a threshold signing session
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThresholdSigningState {
    /// Collecting nonce commitments (Round 1)
    CollectingCommitments { received: usize, required: usize },
    /// Collecting signature shares (Round 2)
    CollectingShares { received: usize, required: usize },
    /// Signing complete
    Complete,
    /// Signing failed
    Failed { reason: String },
}

/// Message types for threshold signing protocol
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ThresholdMessage {
    /// Round 1: Nonce commitment
    Commitment {
        /// Sender's identifier
        sender_id: u16,
        /// Serialized commitment
        commitment: Vec<u8>,
    },
    /// Round 2: Signature share
    SignatureShare {
        /// Sender's identifier
        sender_id: u16,
        /// Serialized signature share
        share: Vec<u8>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_config() {
        let config = ThresholdConfig::new(3, 5).unwrap();
        assert_eq!(config.threshold, 3);
        assert_eq!(config.max_signers, 5);

        // 5-3=2 faulty out of 5 = 40% Byzantine tolerance
        let tolerance = config.byzantine_tolerance();
        assert!((tolerance - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_invalid_config() {
        assert!(ThresholdConfig::new(1, 5).is_err()); // threshold < 2
        assert!(ThresholdConfig::new(6, 5).is_err()); // threshold > max
    }

    #[test]
    fn test_key_generation() {
        let config = ThresholdConfig::two_of_three();
        let (shares, group_public, _) = generate_shares_with_dealer(&config).unwrap();

        assert_eq!(shares.len(), 3);
        assert_eq!(group_public.to_bytes().unwrap().len(), 32);
    }

    #[test]
    fn test_full_threshold_signing() {
        // Setup: 2-of-3 threshold
        let config = ThresholdConfig::two_of_three();
        let (shares, group_public, pubkey_package) = generate_shares_with_dealer(&config).unwrap();

        // Message to sign (e.g., block hash)
        let message = b"block_hash:abc123def456";

        // Create signers (using first 2 of 3)
        let mut signers: Vec<ThresholdSigner> = shares[..2]
            .iter()
            .map(|share| ThresholdSigner::new(share))
            .collect();

        // Round 1: Collect commitments
        let mut coordinator = ThresholdCoordinator::new(message, pubkey_package);
        for signer in &mut signers {
            let commitment = signer.generate_commitment();
            coordinator.add_commitment(signer.identifier(), commitment);
        }

        assert!(coordinator.has_enough_commitments(config.threshold));

        // Create signing package
        let signing_package = coordinator.create_signing_package().unwrap();

        // Round 2: Collect signature shares
        for signer in &signers {
            let sig_share = signer.sign(&signing_package).unwrap();
            coordinator.add_signature_share(signer.identifier(), sig_share);
        }

        assert!(coordinator.has_enough_shares(config.threshold));

        // Aggregate
        let signature = coordinator.aggregate(&signing_package).unwrap();
        assert_eq!(signature.signer_count, 2);
        assert!(signature.meets_threshold(2));

        // Verify
        assert!(group_public.verify(message, &signature).is_ok());
    }

    #[test]
    fn test_wrong_message_fails() {
        let config = ThresholdConfig::two_of_three();
        let (shares, group_public, pubkey_package) = generate_shares_with_dealer(&config).unwrap();

        let message = b"original message";
        let wrong_message = b"different message";

        let mut signers: Vec<ThresholdSigner> = shares[..2]
            .iter()
            .map(|share| ThresholdSigner::new(share))
            .collect();

        let mut coordinator = ThresholdCoordinator::new(message, pubkey_package);
        for signer in &mut signers {
            let commitment = signer.generate_commitment();
            coordinator.add_commitment(signer.identifier(), commitment);
        }

        let signing_package = coordinator.create_signing_package().unwrap();

        for signer in &signers {
            let sig_share = signer.sign(&signing_package).unwrap();
            coordinator.add_signature_share(signer.identifier(), sig_share);
        }

        let signature = coordinator.aggregate(&signing_package).unwrap();

        // Should fail with wrong message
        assert!(group_public.verify(wrong_message, &signature).is_err());
    }

    #[test]
    fn test_insufficient_signers_fails() {
        let config = ThresholdConfig::three_of_five();
        let (shares, _group_public, pubkey_package) = generate_shares_with_dealer(&config).unwrap();

        let message = b"test message";

        // Only use 2 signers (need 3)
        let mut signers: Vec<ThresholdSigner> = shares[..2]
            .iter()
            .map(|share| ThresholdSigner::new(share))
            .collect();

        let mut coordinator = ThresholdCoordinator::new(message, pubkey_package);
        for signer in &mut signers {
            let commitment = signer.generate_commitment();
            coordinator.add_commitment(signer.identifier(), commitment);
        }

        // Should not have enough commitments
        assert!(!coordinator.has_enough_commitments(config.threshold));
    }

    #[test]
    fn test_group_public_key_serialization() {
        let config = ThresholdConfig::two_of_three();
        let (_, group_public, _) = generate_shares_with_dealer(&config).unwrap();

        let bytes = group_public.to_bytes().unwrap();
        let restored = GroupPublicKey::from_bytes(&bytes).unwrap();

        assert_eq!(
            group_public.to_bytes().unwrap(),
            restored.to_bytes().unwrap()
        );
    }
}
