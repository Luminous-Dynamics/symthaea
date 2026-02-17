//! Threshold Signatures using FROST
//!
//! Distributed threshold signatures for proof attestation using
//! FROST (Flexible Round-Optimized Schnorr Threshold) over Ed25519.
//!
//! ## Features
//!
//! - t-of-n threshold signing (e.g., 3-of-5)
//! - Distributed key generation (DKG)
//! - No single point of trust
//! - Compatible with standard Ed25519 verification
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fl_aggregator::proofs::integration::threshold::{
//!     ThresholdConfig, generate_shares, sign_with_threshold, verify_threshold_signature
//! };
//!
//! // Setup: 3-of-5 threshold
//! let config = ThresholdConfig::new(3, 5);
//! let (shares, group_public) = generate_shares(&config)?;
//!
//! // Each participant signs
//! let partial_sigs: Vec<_> = participants.iter()
//!     .map(|p| p.sign(message, &shares[p.id]))
//!     .collect();
//!
//! // Aggregate t signatures
//! let signature = aggregate_signatures(&partial_sigs, &config)?;
//!
//! // Verify using standard Ed25519
//! assert!(verify_threshold_signature(&signature, message, &group_public));
//! ```

#[cfg(feature = "proofs-threshold")]
use frost_ed25519 as frost;

use crate::proofs::{ProofError, ProofResult};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Threshold configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThresholdConfig {
    /// Minimum signers required (t)
    pub threshold: u16,
    /// Total number of participants (n)
    pub max_signers: u16,
}

impl ThresholdConfig {
    /// Create a new threshold configuration
    ///
    /// # Arguments
    /// * `threshold` - Minimum number of signers required (t)
    /// * `max_signers` - Total number of participants (n)
    ///
    /// # Requirements
    /// * threshold >= 2
    /// * threshold <= max_signers
    pub fn new(threshold: u16, max_signers: u16) -> ProofResult<Self> {
        if threshold < 2 {
            return Err(ProofError::InvalidPublicInputs(
                "Threshold must be at least 2".to_string()
            ));
        }
        if threshold > max_signers {
            return Err(ProofError::InvalidPublicInputs(
                "Threshold cannot exceed max signers".to_string()
            ));
        }

        Ok(Self { threshold, max_signers })
    }

    /// Common configurations
    pub fn two_of_three() -> Self {
        Self { threshold: 2, max_signers: 3 }
    }

    pub fn three_of_five() -> Self {
        Self { threshold: 3, max_signers: 5 }
    }

    pub fn five_of_seven() -> Self {
        Self { threshold: 5, max_signers: 7 }
    }
}

/// Participant's key share
#[cfg(feature = "proofs-threshold")]
#[derive(Clone)]
pub struct KeyShare {
    /// Participant identifier
    pub identifier: frost::Identifier,
    /// Secret key share
    pub signing_share: frost::keys::SigningShare,
    /// Public key share for verification
    pub verifying_share: frost::keys::VerifyingShare,
}

/// Group public key (used for verification)
#[cfg(feature = "proofs-threshold")]
#[derive(Clone)]
pub struct GroupPublicKey {
    /// The group's verifying key
    pub verifying_key: frost::VerifyingKey,
}

#[cfg(feature = "proofs-threshold")]
impl GroupPublicKey {
    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        self.verifying_key.serialize().as_ref().to_vec()
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> ProofResult<Self> {
        let bytes_array: [u8; 32] = bytes.try_into()
            .map_err(|_| ProofError::InvalidPublicInputs("Invalid group public key length".to_string()))?;

        let verifying_key = frost::VerifyingKey::deserialize(bytes_array)
            .map_err(|e| ProofError::InvalidPublicInputs(format!("Invalid group public key: {:?}", e)))?;

        Ok(Self { verifying_key })
    }
}

/// Generate key shares using trusted dealer (for testing/setup)
///
/// In production, use distributed key generation (DKG) instead.
#[cfg(feature = "proofs-threshold")]
pub fn generate_shares(
    config: &ThresholdConfig,
) -> ProofResult<(Vec<KeyShare>, GroupPublicKey)> {
    use rand::rngs::OsRng;

    let (shares_map, pubkey_package) = frost::keys::generate_with_dealer(
        config.max_signers,
        config.threshold,
        frost::keys::IdentifierList::Default,
        &mut OsRng,
    ).map_err(|e| ProofError::GenerationFailed(format!("Key generation failed: {:?}", e)))?;

    let shares: Vec<KeyShare> = shares_map
        .into_iter()
        .map(|(id, secret_share)| {
            KeyShare {
                identifier: id,
                signing_share: secret_share.signing_share().clone(),
                verifying_share: pubkey_package.verifying_shares().get(&id).unwrap().clone(),
            }
        })
        .collect();

    let group_public = GroupPublicKey {
        verifying_key: pubkey_package.verifying_key().clone(),
    };

    Ok((shares, group_public))
}

/// Threshold signature (compatible with Ed25519)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThresholdSignature {
    /// The aggregated signature bytes
    pub signature: Vec<u8>,
    /// Number of signers who contributed
    pub signer_count: usize,
}

/// Signing round state (coordinator)
#[cfg(feature = "proofs-threshold")]
pub struct SigningCoordinator {
    /// Message being signed
    message: Vec<u8>,
    /// Collected nonce commitments
    nonce_commitments: BTreeMap<frost::Identifier, frost::round1::SigningCommitments>,
    /// Collected signature shares
    signature_shares: BTreeMap<frost::Identifier, frost::round2::SignatureShare>,
    /// Public key package
    pubkey_package: frost::keys::PublicKeyPackage,
}

#[cfg(feature = "proofs-threshold")]
impl SigningCoordinator {
    /// Start a new signing session
    pub fn new(message: &[u8], pubkey_package: frost::keys::PublicKeyPackage) -> Self {
        Self {
            message: message.to_vec(),
            nonce_commitments: BTreeMap::new(),
            signature_shares: BTreeMap::new(),
            pubkey_package,
        }
    }

    /// Add a nonce commitment from a participant
    pub fn add_commitment(
        &mut self,
        identifier: frost::Identifier,
        commitment: frost::round1::SigningCommitments,
    ) {
        self.nonce_commitments.insert(identifier, commitment);
    }

    /// Add a signature share from a participant
    pub fn add_signature_share(
        &mut self,
        identifier: frost::Identifier,
        share: frost::round2::SignatureShare,
    ) {
        self.signature_shares.insert(identifier, share);
    }

    /// Create the signing package for round 2
    pub fn create_signing_package(&self) -> ProofResult<frost::SigningPackage> {
        frost::SigningPackage::new(self.nonce_commitments.clone(), &self.message)
            .map_err(|e| ProofError::GenerationFailed(format!("Failed to create signing package: {:?}", e)))
    }

    /// Aggregate signature shares into final signature
    pub fn aggregate(&self, signing_package: &frost::SigningPackage) -> ProofResult<ThresholdSignature> {
        let signature = frost::aggregate(
            signing_package,
            &self.signature_shares,
            &self.pubkey_package,
        ).map_err(|e| ProofError::GenerationFailed(format!("Signature aggregation failed: {:?}", e)))?;

        Ok(ThresholdSignature {
            signature: signature.serialize().to_vec(),
            signer_count: self.signature_shares.len(),
        })
    }
}

/// Participant in a threshold signing session
#[cfg(feature = "proofs-threshold")]
pub struct ThresholdParticipant {
    /// Key share
    key_share: frost::keys::KeyPackage,
    /// Nonce for current signing session
    signing_nonces: Option<frost::round1::SigningNonces>,
}

#[cfg(feature = "proofs-threshold")]
impl ThresholdParticipant {
    /// Create a participant from key share
    pub fn new(
        identifier: frost::Identifier,
        signing_share: frost::keys::SigningShare,
        verifying_share: frost::keys::VerifyingShare,
        verifying_key: frost::VerifyingKey,
        min_signers: u16,
    ) -> ProofResult<Self> {
        let key_package = frost::keys::KeyPackage::new(
            identifier,
            signing_share,
            verifying_share,
            verifying_key,
            min_signers,
        );

        Ok(Self {
            key_share: key_package,
            signing_nonces: None,
        })
    }

    /// Generate nonce commitment for round 1
    pub fn round1_commit(&mut self) -> frost::round1::SigningCommitments {
        use rand::rngs::OsRng;
        let (nonces, commitments) = frost::round1::commit(
            self.key_share.signing_share(),
            &mut OsRng,
        );
        self.signing_nonces = Some(nonces);
        commitments
    }

    /// Generate signature share for round 2
    pub fn round2_sign(
        &self,
        signing_package: &frost::SigningPackage,
    ) -> ProofResult<frost::round2::SignatureShare> {
        let nonces = self.signing_nonces.as_ref()
            .ok_or_else(|| ProofError::GenerationFailed("Must call round1_commit first".to_string()))?;

        frost::round2::sign(signing_package, nonces, &self.key_share)
            .map_err(|e| ProofError::GenerationFailed(format!("Signing failed: {:?}", e)))
    }

    /// Get participant identifier
    pub fn identifier(&self) -> frost::Identifier {
        *self.key_share.identifier()
    }
}

/// Verify a threshold signature
#[cfg(feature = "proofs-threshold")]
pub fn verify_threshold_signature(
    signature: &ThresholdSignature,
    message: &[u8],
    group_public: &GroupPublicKey,
) -> ProofResult<bool> {
    let sig_bytes: [u8; 64] = signature.signature.as_slice().try_into()
        .map_err(|_| ProofError::VerificationFailed("Invalid signature length".to_string()))?;

    let frost_sig = frost::Signature::deserialize(sig_bytes)
        .map_err(|e| ProofError::VerificationFailed(format!("Invalid signature format: {:?}", e)))?;

    match group_public.verifying_key.verify(message, &frost_sig) {
        Ok(_) => Ok(true),
        Err(_) => Ok(false),
    }
}

/// Attestation using threshold signature
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThresholdAttestation {
    /// The threshold signature
    pub signature: ThresholdSignature,
    /// Group public key bytes
    pub group_public_key: Vec<u8>,
    /// Timestamp
    pub timestamp: u64,
    /// Threshold configuration
    pub config: ThresholdConfig,
    /// Optional context
    pub context: Option<String>,
}

#[cfg(all(test, feature = "proofs-threshold"))]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_config() {
        let config = ThresholdConfig::new(3, 5).unwrap();
        assert_eq!(config.threshold, 3);
        assert_eq!(config.max_signers, 5);
    }

    #[test]
    fn test_invalid_threshold() {
        assert!(ThresholdConfig::new(1, 5).is_err()); // threshold < 2
        assert!(ThresholdConfig::new(6, 5).is_err()); // threshold > max
    }

    #[test]
    fn test_key_generation() {
        let config = ThresholdConfig::two_of_three();
        let (shares, group_public) = generate_shares(&config).unwrap();

        assert_eq!(shares.len(), 3);
        assert_eq!(group_public.to_bytes().len(), 32);
    }

    #[test]
    fn test_full_signing_flow() {
        // Setup: 2-of-3 threshold
        let config = ThresholdConfig::two_of_three();
        let (shares, group_public) = generate_shares(&config).unwrap();

        // Create public key package for coordinator
        let mut verifying_shares = BTreeMap::new();
        for share in &shares {
            verifying_shares.insert(share.identifier, share.verifying_share.clone());
        }
        let pubkey_package = frost::keys::PublicKeyPackage::new(
            verifying_shares,
            group_public.verifying_key.clone(),
        );

        // Message to sign
        let message = b"Proof composition merkle root attestation";

        // Create participants (using first 2 of 3)
        let mut participants: Vec<ThresholdParticipant> = shares[..2]
            .iter()
            .map(|share| {
                ThresholdParticipant::new(
                    share.identifier,
                    share.signing_share.clone(),
                    share.verifying_share.clone(),
                    group_public.verifying_key.clone(),
                    config.threshold,
                ).unwrap()
            })
            .collect();

        // Round 1: Collect commitments
        let mut coordinator = SigningCoordinator::new(message, pubkey_package);
        for participant in &mut participants {
            let commitment = participant.round1_commit();
            coordinator.add_commitment(participant.identifier(), commitment);
        }

        // Create signing package
        let signing_package = coordinator.create_signing_package().unwrap();

        // Round 2: Collect signature shares
        for participant in &participants {
            let sig_share = participant.round2_sign(&signing_package).unwrap();
            coordinator.add_signature_share(participant.identifier(), sig_share);
        }

        // Aggregate
        let signature = coordinator.aggregate(&signing_package).unwrap();
        assert_eq!(signature.signer_count, 2);

        // Verify
        let valid = verify_threshold_signature(&signature, message, &group_public).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_signature_verification_fails_with_wrong_message() {
        let config = ThresholdConfig::two_of_three();
        let (shares, group_public) = generate_shares(&config).unwrap();

        let mut verifying_shares = BTreeMap::new();
        for share in &shares {
            verifying_shares.insert(share.identifier, share.verifying_share.clone());
        }
        let pubkey_package = frost::keys::PublicKeyPackage::new(
            verifying_shares,
            group_public.verifying_key.clone(),
        );

        let message = b"Original message";
        let wrong_message = b"Different message";

        let mut participants: Vec<ThresholdParticipant> = shares[..2]
            .iter()
            .map(|share| {
                ThresholdParticipant::new(
                    share.identifier,
                    share.signing_share.clone(),
                    share.verifying_share.clone(),
                    group_public.verifying_key.clone(),
                    config.threshold,
                ).unwrap()
            })
            .collect();

        let mut coordinator = SigningCoordinator::new(message, pubkey_package);
        for participant in &mut participants {
            let commitment = participant.round1_commit();
            coordinator.add_commitment(participant.identifier(), commitment);
        }

        let signing_package = coordinator.create_signing_package().unwrap();

        for participant in &participants {
            let sig_share = participant.round2_sign(&signing_package).unwrap();
            coordinator.add_signature_share(participant.identifier(), sig_share);
        }

        let signature = coordinator.aggregate(&signing_package).unwrap();

        // Should fail with wrong message
        let valid = verify_threshold_signature(&signature, wrong_message, &group_public).unwrap();
        assert!(!valid);
    }
}
