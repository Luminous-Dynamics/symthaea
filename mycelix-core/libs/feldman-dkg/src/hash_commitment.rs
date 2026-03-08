//! Hash-based commitments for post-quantum security
//!
//! Feldman's original commitments (C_i = g^{a_i}) rely on the discrete log
//! assumption, which is broken by Shor's algorithm on a quantum computer.
//! This module provides hash-based commitments using SHA3-256 that are
//! quantum-resistant.
//!
//! ## Commit-Reveal Protocol
//!
//! 1. **Commit phase**: Each dealer computes `H(share_i || dealer_id || salt)`
//!    for each share and broadcasts these hash commitments.
//! 2. **Reveal phase**: Each dealer reveals the actual shares and salts.
//! 3. **Verify phase**: Recipients verify `H(share_i || dealer_id || salt) == commitment`.
//!
//! ## Tradeoff
//!
//! Hash-based commitments are quantum-resistant but lose the homomorphic
//! verification property of Feldman commitments (can't verify g^{s_i} = Π C_j^{i^j}).
//! Instead, verification requires revealing the share to the recipient, who checks
//! the hash. This is acceptable because each share is only revealed to its recipient.

use sha2::{Sha256, Digest};
use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

use crate::dealer::Deal;
use crate::error::{DkgError, DkgResult};
use crate::participant::ParticipantId;
use crate::scalar::Scalar;

/// A hash-based commitment to a share value
///
/// `commitment = SHA-256(share_bytes || dealer_id_le || salt)`
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HashCommitment {
    /// The hash value (32 bytes)
    pub hash: [u8; 32],
    /// The recipient participant index
    pub recipient: u32,
}

/// Salt used in hash commitments (32 bytes of randomness)
#[derive(Clone, Debug, Zeroize, Serialize, Deserialize)]
#[zeroize(drop)]
pub struct CommitmentSalt(pub [u8; 32]);

impl CommitmentSalt {
    /// Generate a random salt
    pub fn random(rng: &mut impl rand_core::CryptoRngCore) -> Self {
        let mut bytes = [0u8; 32];
        rng.fill_bytes(&mut bytes);
        Self(bytes)
    }

    /// Create from bytes
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Get the salt bytes
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// A set of hash commitments from a single dealer
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashCommitmentSet {
    /// The dealer who created these commitments
    pub dealer: ParticipantId,
    /// Hash commitments, one per recipient
    pub commitments: Vec<HashCommitment>,
}

/// A reveal message containing the salt for verification
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashReveal {
    /// The dealer
    pub dealer: ParticipantId,
    /// Salt used for each recipient (indexed by recipient participant index)
    pub salts: Vec<(u32, CommitmentSalt)>,
}

impl HashCommitment {
    /// Compute a hash commitment for a share
    pub fn compute(share_value: &Scalar, dealer_id: u32, salt: &CommitmentSalt) -> Self {
        let share_bytes = share_value.to_bytes();
        let hash = Self::hash_share(&share_bytes, dealer_id, salt);
        Self {
            hash,
            recipient: 0, // Set by caller
        }
    }

    /// Compute the hash for a share with dealer ID and salt
    fn hash_share(share_bytes: &[u8], dealer_id: u32, salt: &CommitmentSalt) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(share_bytes);
        hasher.update(dealer_id.to_le_bytes());
        hasher.update(salt.as_bytes());
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Verify a share against this commitment
    pub fn verify(&self, share_value: &Scalar, dealer_id: u32, salt: &CommitmentSalt) -> bool {
        let share_bytes = share_value.to_bytes();
        let computed = Self::hash_share(&share_bytes, dealer_id, salt);
        // Constant-time comparison to prevent timing attacks
        constant_time_eq(&self.hash, &computed)
    }
}

/// Constant-time byte comparison
fn constant_time_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    let mut diff = 0u8;
    for i in 0..32 {
        diff |= a[i] ^ b[i];
    }
    diff == 0
}

impl HashCommitmentSet {
    /// Create hash commitments for all shares in a deal
    pub fn from_deal(deal: &Deal, rng: &mut impl rand_core::CryptoRngCore) -> (Self, HashReveal) {
        let mut commitments = Vec::with_capacity(deal.shares.len());
        let mut salts = Vec::with_capacity(deal.shares.len());

        for share in &deal.shares {
            let salt = CommitmentSalt::random(rng);
            let mut commitment = HashCommitment::compute(&share.value, deal.dealer.0, &salt);
            commitment.recipient = share.index;
            commitments.push(commitment);
            salts.push((share.index, salt));
        }

        let commitment_set = Self {
            dealer: deal.dealer,
            commitments,
        };

        let reveal = HashReveal {
            dealer: deal.dealer,
            salts,
        };

        (commitment_set, reveal)
    }

    /// Verify a revealed share against its commitment
    pub fn verify_share(
        &self,
        recipient: u32,
        share_value: &Scalar,
        salt: &CommitmentSalt,
    ) -> DkgResult<bool> {
        let commitment = self
            .commitments
            .iter()
            .find(|c| c.recipient == recipient)
            .ok_or(DkgError::ParticipantNotFound(recipient))?;

        Ok(commitment.verify(share_value, self.dealer.0, salt))
    }

    /// Get the commitment for a specific recipient
    pub fn get_commitment(&self, recipient: u32) -> Option<&HashCommitment> {
        self.commitments.iter().find(|c| c.recipient == recipient)
    }

    /// Get the number of commitments
    pub fn len(&self) -> usize {
        self.commitments.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.commitments.is_empty()
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> DkgResult<Vec<u8>> {
        serde_json::to_vec(self)
            .map_err(|e| DkgError::SerializationError(e.to_string()))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> DkgResult<Self> {
        serde_json::from_slice(bytes)
            .map_err(|e| DkgError::SerializationError(e.to_string()))
    }
}

/// Commitment scheme selector
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitmentScheme {
    /// Classical Feldman commitments (C_i = g^{a_i})
    /// Fast, homomorphic verification, but quantum-vulnerable
    Feldman,
    /// Hash-based commitments (SHA-256)
    /// Quantum-resistant, but requires reveal phase
    HashBased,
    /// Both: Feldman for efficient verification + hash for quantum safety
    #[default]
    Hybrid,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dealer::Dealer;
    use rand::rngs::OsRng;

    #[test]
    fn test_hash_commitment_basic() {
        let value = Scalar::from_u64(42);
        let salt = CommitmentSalt::random(&mut OsRng);
        let commitment = HashCommitment::compute(&value, 1, &salt);

        assert!(commitment.verify(&value, 1, &salt));
    }

    #[test]
    fn test_hash_commitment_wrong_value() {
        let value = Scalar::from_u64(42);
        let wrong_value = Scalar::from_u64(43);
        let salt = CommitmentSalt::random(&mut OsRng);
        let commitment = HashCommitment::compute(&value, 1, &salt);

        assert!(!commitment.verify(&wrong_value, 1, &salt));
    }

    #[test]
    fn test_hash_commitment_wrong_dealer() {
        let value = Scalar::from_u64(42);
        let salt = CommitmentSalt::random(&mut OsRng);
        let commitment = HashCommitment::compute(&value, 1, &salt);

        assert!(!commitment.verify(&value, 2, &salt));
    }

    #[test]
    fn test_hash_commitment_wrong_salt() {
        let value = Scalar::from_u64(42);
        let salt1 = CommitmentSalt::random(&mut OsRng);
        let salt2 = CommitmentSalt::random(&mut OsRng);
        let commitment = HashCommitment::compute(&value, 1, &salt1);

        assert!(!commitment.verify(&value, 1, &salt2));
    }

    #[test]
    fn test_hash_commitment_deterministic() {
        let value = Scalar::from_u64(42);
        let salt = CommitmentSalt::from_bytes([1u8; 32]);
        let c1 = HashCommitment::compute(&value, 1, &salt);
        let c2 = HashCommitment::compute(&value, 1, &salt);

        assert_eq!(c1.hash, c2.hash);
    }

    #[test]
    fn test_hash_commitment_set_from_deal() {
        let dealer = Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();

        let (commitment_set, reveal) = HashCommitmentSet::from_deal(&deal, &mut OsRng);

        assert_eq!(commitment_set.len(), 3);
        assert_eq!(reveal.salts.len(), 3);

        // Verify each share against its commitment
        for share in &deal.shares {
            let salt = reveal.salts.iter().find(|(r, _)| *r == share.index).unwrap();
            let valid = commitment_set.verify_share(share.index, &share.value, &salt.1).unwrap();
            assert!(valid, "Share {} should verify", share.index);
        }
    }

    #[test]
    fn test_hash_commitment_set_tampered_share() {
        let dealer = Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();

        let (commitment_set, reveal) = HashCommitmentSet::from_deal(&deal, &mut OsRng);

        // Try to verify a wrong share value
        let wrong_value = Scalar::from_u64(999999);
        let salt = &reveal.salts[0].1;
        let valid = commitment_set.verify_share(1, &wrong_value, salt).unwrap();
        assert!(!valid, "Tampered share should not verify");
    }

    #[test]
    fn test_hash_commitment_set_missing_recipient() {
        let dealer = Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();

        let (commitment_set, _) = HashCommitmentSet::from_deal(&deal, &mut OsRng);

        let result = commitment_set.verify_share(99, &Scalar::from_u64(0), &CommitmentSalt::from_bytes([0u8; 32]));
        assert!(matches!(result, Err(DkgError::ParticipantNotFound(99))));
    }

    #[test]
    fn test_hash_commitment_set_serialization() {
        let dealer = Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();

        let (commitment_set, _) = HashCommitmentSet::from_deal(&deal, &mut OsRng);

        let bytes = commitment_set.to_bytes().unwrap();
        let recovered = HashCommitmentSet::from_bytes(&bytes).unwrap();

        assert_eq!(recovered.dealer, commitment_set.dealer);
        assert_eq!(recovered.len(), commitment_set.len());
        for (orig, recov) in commitment_set.commitments.iter().zip(recovered.commitments.iter()) {
            assert_eq!(orig.hash, recov.hash);
            assert_eq!(orig.recipient, recov.recipient);
        }
    }

    #[test]
    fn test_constant_time_eq() {
        let a = [1u8; 32];
        let b = [1u8; 32];
        let c = [2u8; 32];

        assert!(constant_time_eq(&a, &b));
        assert!(!constant_time_eq(&a, &c));
    }

    #[test]
    fn test_commitment_scheme_default() {
        assert_eq!(CommitmentScheme::default(), CommitmentScheme::Hybrid);
    }

    #[test]
    fn test_salt_zeroize() {
        // Ensure salt can be created and dropped without issues
        let salt = CommitmentSalt::random(&mut OsRng);
        let bytes = *salt.as_bytes();
        drop(salt);
        // Can't verify zeroize happened (UB to read after drop), but this tests no panic
        let _ = bytes;
    }

    #[test]
    fn test_hash_commitment_set_get_commitment() {
        let dealer = Dealer::new(ParticipantId(1), 2, 3, &mut OsRng).unwrap();
        let deal = dealer.generate_deal();

        let (commitment_set, _) = HashCommitmentSet::from_deal(&deal, &mut OsRng);

        assert!(commitment_set.get_commitment(1).is_some());
        assert!(commitment_set.get_commitment(2).is_some());
        assert!(commitment_set.get_commitment(3).is_some());
        assert!(commitment_set.get_commitment(99).is_none());
    }

    #[test]
    fn test_full_commit_reveal_protocol() {
        // Simulate a complete commit-reveal ceremony with 3 dealers
        let dealers: Vec<_> = (1..=3u32)
            .map(|i| Dealer::new(ParticipantId(i), 2, 3, &mut OsRng).unwrap())
            .collect();
        let deals: Vec<_> = dealers.iter().map(|d| d.generate_deal()).collect();

        // Phase 1: All dealers commit
        let commit_data: Vec<(HashCommitmentSet, HashReveal)> = deals
            .iter()
            .map(|deal| HashCommitmentSet::from_deal(deal, &mut OsRng))
            .collect();

        // Phase 2: All dealers reveal (after all commitments are collected)
        // Phase 3: Each recipient verifies their shares
        for recipient in 1..=3u32 {
            for (deal_idx, deal) in deals.iter().enumerate() {
                let share = deal.get_share(recipient).unwrap();
                let (ref commitment_set, ref reveal) = commit_data[deal_idx];
                let salt = reveal.salts.iter().find(|(r, _)| *r == recipient).unwrap();
                let valid = commitment_set.verify_share(recipient, &share.value, &salt.1).unwrap();
                assert!(valid, "Recipient {} share from dealer {} should verify", recipient, deal_idx + 1);
            }
        }
    }
}
