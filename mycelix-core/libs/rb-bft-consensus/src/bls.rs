// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! BLS12-381 Aggregate Signatures for RB-BFT Consensus
//!
//! This module provides BLS (Boneh-Lynn-Shacham) signatures over the BLS12-381 curve.
//! The key advantage of BLS signatures is that multiple signatures can be aggregated
//! into a single signature, dramatically reducing bandwidth in consensus protocols.
//!
//! ## Features
//!
//! - **Signature Aggregation**: Compress n signatures into 1 (48 bytes)
//! - **Batch Verification**: Verify aggregate signature against multiple messages
//! - **Proof of Possession**: Prevent rogue key attacks
//! - **Domain Separation**: Prevent cross-protocol attacks
//!
//! ## Usage
//!
//! ```ignore
//! use rb_bft_consensus::bls::{BlsKeypair, aggregate_signatures, verify_aggregate};
//!
//! // Generate keypairs for validators
//! let kp1 = BlsKeypair::generate();
//! let kp2 = BlsKeypair::generate();
//!
//! // Sign the same message
//! let msg = b"consensus proposal for round 42";
//! let sig1 = kp1.sign(msg);
//! let sig2 = kp2.sign(msg);
//!
//! // Aggregate into single signature
//! let agg_sig = aggregate_signatures(&[sig1, sig2]).unwrap();
//!
//! // Verify aggregate (same message, multiple keys)
//! let pubkeys = vec![kp1.public_key(), kp2.public_key()];
//! assert!(verify_aggregate(&agg_sig, msg, &pubkeys).is_ok());
//! ```

#[cfg(feature = "bls")]
use blst::{
    min_pk::{AggregateSignature, PublicKey, SecretKey, Signature},
    BLST_ERROR,
};
use serde::{Deserialize, Serialize};
use zeroize::ZeroizeOnDrop;

use crate::error::{ConsensusError, ConsensusResult};

/// Domain separation tag for consensus votes
pub const DST_VOTE: &[u8] = b"MYCELIX-RBBFT-VOTE-BLS12381-V1";
/// Domain separation tag for proposals
pub const DST_PROPOSAL: &[u8] = b"MYCELIX-RBBFT-PROPOSAL-BLS12381-V1";
/// Domain separation tag for proof of possession
pub const DST_POP: &[u8] = b"MYCELIX-RBBFT-POP-BLS12381-V1";

/// BLS keypair for a validator
#[cfg(feature = "bls")]
#[derive(ZeroizeOnDrop)]
pub struct BlsKeypair {
    /// Secret key (zeroized on drop)
    #[zeroize(skip)] // blst handles its own memory
    secret_key: SecretKey,
    /// Public key
    #[zeroize(skip)]
    public_key: PublicKey,
}

#[cfg(feature = "bls")]
impl BlsKeypair {
    /// Generate a new random BLS keypair
    pub fn generate() -> Self {
        let mut ikm = [0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::rngs::OsRng, &mut ikm);
        let secret_key = SecretKey::key_gen(&ikm, &[]).expect("valid ikm");
        let public_key = secret_key.sk_to_pk();
        Self { secret_key, public_key }
    }

    /// Create keypair from seed bytes
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        let secret_key = SecretKey::key_gen(seed, &[]).expect("valid seed");
        let public_key = secret_key.sk_to_pk();
        Self { secret_key, public_key }
    }

    /// Get the public key
    pub fn public_key(&self) -> BlsPublicKey {
        BlsPublicKey(self.public_key.compress())
    }

    /// Get the public key bytes (48 bytes compressed)
    pub fn public_key_bytes(&self) -> [u8; 48] {
        self.public_key.compress()
    }

    /// Sign a message with domain separation
    pub fn sign(&self, message: &[u8], dst: &[u8]) -> BlsSignature {
        let sig = self.secret_key.sign(message, dst, &[]);
        BlsSignature(sig.compress())
    }

    /// Sign a vote message
    pub fn sign_vote(&self, message: &[u8]) -> BlsSignature {
        self.sign(message, DST_VOTE)
    }

    /// Sign a proposal message
    pub fn sign_proposal(&self, message: &[u8]) -> BlsSignature {
        self.sign(message, DST_PROPOSAL)
    }

    /// Generate proof of possession (prevents rogue key attacks)
    pub fn proof_of_possession(&self) -> BlsSignature {
        let pk_bytes = self.public_key.compress();
        self.sign(&pk_bytes, DST_POP)
    }
}

#[cfg(feature = "bls")]
impl std::fmt::Debug for BlsKeypair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlsKeypair")
            .field("public_key", &hex::encode(self.public_key.compress()))
            .field("secret_key", &"[REDACTED]")
            .finish()
    }
}

/// A BLS public key (48 bytes compressed G1 point)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlsPublicKey(#[serde(with = "hex_bytes")] pub [u8; 48]);

impl BlsPublicKey {
    /// Create from compressed bytes
    pub fn from_bytes(bytes: [u8; 48]) -> Self {
        Self(bytes)
    }

    /// Get the compressed bytes
    pub fn as_bytes(&self) -> &[u8; 48] {
        &self.0
    }

    /// Convert to hex string
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Verify proof of possession
    #[cfg(feature = "bls")]
    pub fn verify_pop(&self, pop: &BlsSignature) -> ConsensusResult<()> {
        let pk = PublicKey::uncompress(&self.0)
            .map_err(|e| ConsensusError::InvalidSignature {
                reason: format!("invalid public key: {:?}", e),
            })?;
        let sig = Signature::uncompress(&pop.0)
            .map_err(|e| ConsensusError::InvalidSignature {
                reason: format!("invalid signature: {:?}", e),
            })?;

        let result = sig.verify(true, &self.0, DST_POP, &[], &pk, true);
        if result == BLST_ERROR::BLST_SUCCESS {
            Ok(())
        } else {
            Err(ConsensusError::InvalidSignature {
                reason: format!("proof of possession verification failed: {:?}", result),
            })
        }
    }
}

/// A BLS signature (96 bytes compressed G2 point)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlsSignature(#[serde(with = "hex_bytes")] pub [u8; 96]);

impl BlsSignature {
    /// Create from compressed bytes
    pub fn from_bytes(bytes: [u8; 96]) -> Self {
        Self(bytes)
    }

    /// Get the compressed bytes
    pub fn as_bytes(&self) -> &[u8; 96] {
        &self.0
    }

    /// Convert to hex string
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }

    /// Verify this signature against a message and public key
    #[cfg(feature = "bls")]
    pub fn verify(&self, message: &[u8], dst: &[u8], pubkey: &BlsPublicKey) -> ConsensusResult<()> {
        let pk = PublicKey::uncompress(&pubkey.0)
            .map_err(|e| ConsensusError::InvalidSignature {
                reason: format!("invalid public key: {:?}", e),
            })?;
        let sig = Signature::uncompress(&self.0)
            .map_err(|e| ConsensusError::InvalidSignature {
                reason: format!("invalid signature: {:?}", e),
            })?;

        let result = sig.verify(true, message, dst, &[], &pk, true);
        if result == BLST_ERROR::BLST_SUCCESS {
            Ok(())
        } else {
            Err(ConsensusError::InvalidSignature {
                reason: format!("BLS signature verification failed: {:?}", result),
            })
        }
    }

    /// Verify as a vote signature
    #[cfg(feature = "bls")]
    pub fn verify_vote(&self, message: &[u8], pubkey: &BlsPublicKey) -> ConsensusResult<()> {
        self.verify(message, DST_VOTE, pubkey)
    }

    /// Verify as a proposal signature
    #[cfg(feature = "bls")]
    pub fn verify_proposal(&self, message: &[u8], pubkey: &BlsPublicKey) -> ConsensusResult<()> {
        self.verify(message, DST_PROPOSAL, pubkey)
    }
}

/// An aggregated BLS signature (96 bytes, same size as single signature!)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlsAggregateSignature(#[serde(with = "hex_bytes")] pub [u8; 96]);

impl BlsAggregateSignature {
    /// Get the compressed bytes
    pub fn as_bytes(&self) -> &[u8; 96] {
        &self.0
    }

    /// Convert to hex string
    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }
}

/// Aggregate multiple BLS signatures into one
///
/// This is the key feature of BLS: n signatures → 1 signature (96 bytes)
/// The aggregate can be verified against all original messages and public keys.
#[cfg(feature = "bls")]
pub fn aggregate_signatures(signatures: &[BlsSignature]) -> ConsensusResult<BlsAggregateSignature> {
    if signatures.is_empty() {
        return Err(ConsensusError::InvalidSignature {
            reason: "cannot aggregate empty signature list".to_string(),
        });
    }

    let sigs: Vec<Signature> = signatures
        .iter()
        .map(|s| Signature::uncompress(&s.0))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ConsensusError::InvalidSignature {
            reason: format!("invalid signature in aggregation: {:?}", e),
        })?;

    let sig_refs: Vec<&Signature> = sigs.iter().collect();
    let agg = AggregateSignature::aggregate(&sig_refs, true)
        .map_err(|e| ConsensusError::InvalidSignature {
            reason: format!("aggregation failed: {:?}", e),
        })?;

    Ok(BlsAggregateSignature(agg.to_signature().compress()))
}

/// Verify an aggregate signature where all signers signed the SAME message
///
/// This is the common case in consensus: all validators sign the same proposal.
#[cfg(feature = "bls")]
pub fn verify_aggregate_same_message(
    aggregate: &BlsAggregateSignature,
    message: &[u8],
    dst: &[u8],
    pubkeys: &[BlsPublicKey],
) -> ConsensusResult<()> {
    if pubkeys.is_empty() {
        return Err(ConsensusError::InvalidSignature {
            reason: "cannot verify with empty public key list".to_string(),
        });
    }

    let pks: Vec<PublicKey> = pubkeys
        .iter()
        .map(|pk| PublicKey::uncompress(&pk.0))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| ConsensusError::InvalidSignature {
            reason: format!("invalid public key: {:?}", e),
        })?;

    let pk_refs: Vec<&PublicKey> = pks.iter().collect();

    let sig = Signature::uncompress(&aggregate.0)
        .map_err(|e| ConsensusError::InvalidSignature {
            reason: format!("invalid aggregate signature: {:?}", e),
        })?;

    let result = sig.fast_aggregate_verify(true, message, dst, &pk_refs);
    if result == BLST_ERROR::BLST_SUCCESS {
        Ok(())
    } else {
        Err(ConsensusError::InvalidSignature {
            reason: format!("aggregate verification failed: {:?}", result),
        })
    }
}

/// Verify an aggregate vote signature
#[cfg(feature = "bls")]
pub fn verify_aggregate_vote(
    aggregate: &BlsAggregateSignature,
    message: &[u8],
    pubkeys: &[BlsPublicKey],
) -> ConsensusResult<()> {
    verify_aggregate_same_message(aggregate, message, DST_VOTE, pubkeys)
}

/// Serde helper for fixed-size byte arrays as hex
mod hex_bytes {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S, const N: usize>(bytes: &[u8; N], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&hex::encode(bytes))
    }

    pub fn deserialize<'de, D, const N: usize>(deserializer: D) -> Result<[u8; N], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let bytes = hex::decode(&s).map_err(serde::de::Error::custom)?;
        let len = bytes.len();
        bytes.try_into().map_err(|_| {
            serde::de::Error::custom(format!("expected {} bytes, got {}", N, len))
        })
    }
}

/// Statistics about BLS aggregation
#[derive(Debug, Clone)]
pub struct AggregationStats {
    /// Number of signatures aggregated
    pub signature_count: usize,
    /// Original size (96 bytes * n)
    pub original_size_bytes: usize,
    /// Aggregated size (always 96 bytes)
    pub aggregated_size_bytes: usize,
    /// Compression ratio
    pub compression_ratio: f64,
}

impl AggregationStats {
    /// Calculate stats for a given number of signatures
    pub fn for_count(n: usize) -> Self {
        let original = n * 96;
        let aggregated = 96;
        Self {
            signature_count: n,
            original_size_bytes: original,
            aggregated_size_bytes: aggregated,
            compression_ratio: if n > 0 { original as f64 / aggregated as f64 } else { 0.0 },
        }
    }
}

#[cfg(all(test, feature = "bls"))]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let kp = BlsKeypair::generate();
        let pk = kp.public_key();
        assert_eq!(pk.as_bytes().len(), 48);
    }

    #[test]
    fn test_sign_and_verify() {
        let kp = BlsKeypair::generate();
        let message = b"test message for BLS signing";

        let sig = kp.sign_vote(message);
        assert!(sig.verify_vote(message, &kp.public_key()).is_ok());
    }

    #[test]
    fn test_proof_of_possession() {
        let kp = BlsKeypair::generate();
        let pop = kp.proof_of_possession();
        assert!(kp.public_key().verify_pop(&pop).is_ok());
    }

    #[test]
    fn test_signature_aggregation() {
        let kp1 = BlsKeypair::generate();
        let kp2 = BlsKeypair::generate();
        let kp3 = BlsKeypair::generate();

        let message = b"consensus proposal for round 42";

        let sig1 = kp1.sign_vote(message);
        let sig2 = kp2.sign_vote(message);
        let sig3 = kp3.sign_vote(message);

        // Aggregate 3 signatures into 1
        let agg = aggregate_signatures(&[sig1, sig2, sig3]).unwrap();

        // Verify aggregate
        let pubkeys = vec![kp1.public_key(), kp2.public_key(), kp3.public_key()];
        assert!(verify_aggregate_vote(&agg, message, &pubkeys).is_ok());

        // Check compression stats
        let stats = AggregationStats::for_count(3);
        assert_eq!(stats.original_size_bytes, 288); // 3 * 96
        assert_eq!(stats.aggregated_size_bytes, 96); // always 96
        assert!((stats.compression_ratio - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_aggregate_fails_with_wrong_pubkey() {
        let kp1 = BlsKeypair::generate();
        let kp2 = BlsKeypair::generate();
        let kp_wrong = BlsKeypair::generate();

        let message = b"test message";
        let sig1 = kp1.sign_vote(message);
        let sig2 = kp2.sign_vote(message);

        let agg = aggregate_signatures(&[sig1, sig2]).unwrap();

        // Use wrong public key
        let pubkeys = vec![kp1.public_key(), kp_wrong.public_key()];
        assert!(verify_aggregate_vote(&agg, message, &pubkeys).is_err());
    }

    #[test]
    fn test_aggregate_fails_with_wrong_message() {
        let kp1 = BlsKeypair::generate();
        let kp2 = BlsKeypair::generate();

        let message = b"original message";
        let sig1 = kp1.sign_vote(message);
        let sig2 = kp2.sign_vote(message);

        let agg = aggregate_signatures(&[sig1, sig2]).unwrap();

        let pubkeys = vec![kp1.public_key(), kp2.public_key()];
        let wrong_message = b"tampered message";
        assert!(verify_aggregate_vote(&agg, wrong_message, &pubkeys).is_err());
    }

    #[test]
    fn test_serialization() {
        let kp = BlsKeypair::generate();
        let message = b"test";
        let sig = kp.sign_vote(message);

        // Serialize and deserialize
        let json = serde_json::to_string(&sig).unwrap();
        let restored: BlsSignature = serde_json::from_str(&json).unwrap();
        assert_eq!(sig, restored);
    }

    #[test]
    fn test_large_aggregation() {
        // Simulate 100 validators signing
        let keypairs: Vec<BlsKeypair> = (0..100).map(|_| BlsKeypair::generate()).collect();
        let message = b"proposal hash for round 1000";

        let signatures: Vec<BlsSignature> = keypairs.iter()
            .map(|kp| kp.sign_vote(message))
            .collect();

        let agg = aggregate_signatures(&signatures).unwrap();

        let pubkeys: Vec<BlsPublicKey> = keypairs.iter()
            .map(|kp| kp.public_key())
            .collect();

        assert!(verify_aggregate_vote(&agg, message, &pubkeys).is_ok());

        // 100 signatures compressed to 1
        let stats = AggregationStats::for_count(100);
        assert_eq!(stats.compression_ratio, 100.0);
        println!("Compressed {} signatures from {} bytes to {} bytes ({}x reduction)",
            stats.signature_count,
            stats.original_size_bytes,
            stats.aggregated_size_bytes,
            stats.compression_ratio
        );
    }
}
