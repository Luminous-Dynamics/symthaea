// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # HDC Homomorphic Computation -- Privacy-Preserving Collective Intelligence
//!
//! Binary hypervectors support natural homomorphic operations:
//!
//! ```text
//! bind(enc(A), enc(B)) = enc(bind(A, B))       [XOR distributes over XOR]
//! bundle(enc(A), enc(B)) ~ enc(bundle(A, B))   [majority vote preserves under shared mask]
//! ```
//!
//! ## Security Model
//!
//! [`EncryptedHV`] uses a one-time pad (XOR with random mask):
//! - **Perfect secrecy** (Shannon 1949): ciphertext reveals zero information
//!   about the plaintext **only when the mask comes from a true entropy
//!   source and is used exactly once**. Generate masks with
//!   [`crate::binary_hv::BinaryHV::new_secure_random`] -- a mask derived from
//!   [`crate::binary_hv::BinaryHV::new_random`] (a deterministic function of
//!   a `u64` seed) provides no real secrecy, since the seed can be guessed
//!   or brute-forced.
//! - **Key size**: D = 16,384 bits (mask is same size as message).
//! - **Homomorphic property**: XOR binding distributes over XOR encryption.
//!
//! ## Limitations
//!
//! - Masks must come from a genuine entropy source and must never be reused (OTP constraint).
//! - Majority-vote bundling on encrypted vectors is approximate, not exact.
//! - This is NOT a general-purpose FHE scheme (supports only HDC algebra).
//!
//! ## References
//!
//! - Shannon, C. (1949). Communication theory of secrecy systems.
//! - Imani et al. (2019). A framework for collaborative learning in secure HDC.
//! - Kanerva, P. (2009). Hyperdimensional computing: An introduction.

use crate::binary_hv::BinaryHV;
use crate::crypto::HdcThresholdSharing;

// =========================================================================
// ENCRYPTED HYPERVECTOR
// =========================================================================

/// An encrypted BinaryHV using a one-time pad (XOR mask).
///
/// # Security
///
/// Perfect secrecy (Shannon 1949): when the mask is uniformly random over
/// {0,1}^D and used only once, the ciphertext is statistically independent
/// of the plaintext. **This requires the mask to be generated with
/// [`crate::binary_hv::BinaryHV::new_secure_random`]** (or another CSPRNG) --
/// a mask from [`crate::binary_hv::BinaryHV::new_random(seed)`] is
/// deterministic and does not provide this guarantee.
///
/// # Homomorphic Properties
///
/// ```text
/// enc(A, Ma) ^ enc(B, Mb) = enc(A ^ B, Ma ^ Mb)
/// ```
#[derive(Clone, PartialEq, Eq)]
pub struct EncryptedHV {
    /// Ciphertext: plaintext XOR mask.
    pub ciphertext: BinaryHV,
}

impl EncryptedHV {
    /// Encrypt a BinaryHV with a random mask (one-time pad).
    ///
    /// The mask must be retained by the encryptor for later decryption.
    /// **Never reuse a mask** -- OTP security requires fresh masks. **The
    /// mask must come from [`crate::binary_hv::BinaryHV::new_secure_random`]**
    /// (or another CSPRNG) -- passing a mask derived from
    /// `BinaryHV::new_random(seed)` reduces security to the guessability of
    /// `seed`, not information-theoretic secrecy.
    #[inline]
    pub fn encrypt(plaintext: &BinaryHV, mask: &BinaryHV) -> Self {
        Self {
            ciphertext: plaintext.bind(mask),
        }
    }

    /// Decrypt with the original mask (XOR is self-inverse).
    #[inline]
    pub fn decrypt(&self, mask: &BinaryHV) -> BinaryHV {
        self.ciphertext.bind(mask)
    }

    /// Homomorphic bind: bind two encrypted HVs.
    ///
    /// The result decrypts to `bind(A, B)` using `bind(mask_a, mask_b)`.
    #[inline]
    pub fn hom_bind(&self, other: &EncryptedHV) -> EncryptedHV {
        EncryptedHV {
            ciphertext: self.ciphertext.bind(&other.ciphertext),
        }
    }

    /// Similarity between two encrypted vectors.
    ///
    /// When both are encrypted with the **same mask**, returns the true similarity.
    /// When masks differ, the result is noisy (~0.5).
    #[inline]
    pub fn encrypted_similarity(&self, other: &EncryptedHV) -> f32 {
        self.ciphertext.similarity(&other.ciphertext)
    }
}

impl std::fmt::Debug for EncryptedHV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncryptedHV")
            .field("ciphertext_density", &self.ciphertext.density())
            .finish()
    }
}

// =========================================================================
// COLLECTIVE WISDOM POOL
// =========================================================================

/// Default maximum pool size (256 contributions = 512 KB memory).
pub const DEFAULT_MAX_POOL_SIZE: usize = 256;

/// Privacy-preserving collective wisdom aggregation.
///
/// Each peer encrypts their wisdom vector with a per-session mask and
/// contributes the encrypted version. The pool bundles contributions
/// via majority vote and the aggregate can only be decrypted when
/// sufficient mask shares are reconstructed.
///
/// # Protocol
///
/// ```text
/// 1. Coordinator generates collective mask, splits into (k,n) shares
/// 2. Each peer receives one mask share
/// 3. Each peer encrypts their wisdom: enc(wisdom_i, collective_mask)
/// 4. Pool bundles encrypted contributions
/// 5. k peers reconstruct the mask -> decrypt the aggregate
/// ```
pub struct CollectiveWisdomPool {
    contributions: Vec<EncryptedHV>,
    contributor_ids: Vec<String>,
    max_size: usize,
}

impl CollectiveWisdomPool {
    /// Create a new empty pool with the default size limit (256).
    pub fn new() -> Self {
        Self {
            contributions: Vec::new(),
            contributor_ids: Vec::new(),
            max_size: DEFAULT_MAX_POOL_SIZE,
        }
    }

    /// Create a pool with a custom size limit.
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            contributions: Vec::with_capacity(max_size.min(DEFAULT_MAX_POOL_SIZE)),
            contributor_ids: Vec::new(),
            max_size,
        }
    }

    /// Add an encrypted contribution from a peer.
    ///
    /// Returns `false` if the pool is full.
    pub fn contribute(&mut self, peer_id: &str, encrypted: EncryptedHV) -> bool {
        if self.contributions.len() >= self.max_size {
            return false;
        }
        self.contributions.push(encrypted);
        self.contributor_ids.push(peer_id.to_string());
        true
    }

    /// Number of contributions in the pool.
    pub fn contribution_count(&self) -> usize {
        self.contributions.len()
    }

    /// Whether the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.contributions.is_empty()
    }

    /// Compute encrypted collective wisdom via majority-vote bundling.
    ///
    /// All contributions must be encrypted with the **same mask** for the
    /// bundling to preserve semantic content.
    ///
    /// Returns `None` if the pool is empty.
    pub fn aggregate(&self) -> Option<EncryptedHV> {
        if self.contributions.is_empty() {
            return None;
        }

        let ciphertexts: Vec<BinaryHV> = self.contributions.iter().map(|e| e.ciphertext).collect();

        Some(EncryptedHV {
            ciphertext: BinaryHV::bundle(&ciphertexts),
        })
    }

    /// Clear the pool (for the next round).
    pub fn clear(&mut self) {
        self.contributions.clear();
        self.contributor_ids.clear();
    }

    /// Get the list of contributor peer IDs.
    pub fn contributors(&self) -> &[String] {
        &self.contributor_ids
    }
}

impl Default for CollectiveWisdomPool {
    fn default() -> Self {
        Self::new()
    }
}

// =========================================================================
// SESSION KEY DISTRIBUTION
// =========================================================================

/// Generate a random collective mask and split it into threshold shares.
///
/// Returns `(mask, shares)` where `mask` is the full mask (held only transiently)
/// and `shares` are the (k,n) threshold shares to distribute to peers.
///
/// The full mask should be zeroized after share distribution.
pub fn generate_collective_mask(
    k: usize,
    n: usize,
    seed: u64,
) -> (BinaryHV, Vec<crate::crypto::HdcShare>) {
    let mask = BinaryHV::new_random(seed);
    let shares = HdcThresholdSharing::split(&mask, k, n, seed.wrapping_add(1));
    (mask, shares)
}

// =========================================================================
// TESTS
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_otp_encrypt_decrypt_roundtrip() {
        let plaintext = BinaryHV::new_random(42);
        let mask = BinaryHV::new_random(99);
        let encrypted = EncryptedHV::encrypt(&plaintext, &mask);
        let decrypted = encrypted.decrypt(&mask);
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_otp_ciphertext_hides_plaintext() {
        let plaintext = BinaryHV::new_random(42);
        let mask = BinaryHV::new_random(99);
        let encrypted = EncryptedHV::encrypt(&plaintext, &mask);
        let sim = encrypted.ciphertext.similarity(&plaintext);
        assert!(
            (sim - 0.5).abs() < 0.05,
            "Ciphertext should be ~0.5 similar to plaintext (OTP hiding), got {sim}"
        );
    }

    #[test]
    fn test_homomorphic_bind() {
        let a = BinaryHV::new_random(1);
        let b = BinaryHV::new_random(2);
        let mask_a = BinaryHV::new_random(10);
        let mask_b = BinaryHV::new_random(20);

        let enc_a = EncryptedHV::encrypt(&a, &mask_a);
        let enc_b = EncryptedHV::encrypt(&b, &mask_b);

        let enc_ab = enc_a.hom_bind(&enc_b);

        // Decrypt with combined mask
        let combined_mask = mask_a.bind(&mask_b);
        let decrypted = enc_ab.decrypt(&combined_mask);

        let expected = a.bind(&b);
        assert_eq!(
            decrypted, expected,
            "Homomorphic bind should produce bind(a,b)"
        );
    }

    #[test]
    fn test_same_mask_preserves_similarity() {
        let a = BinaryHV::new_random(1);
        let b = BinaryHV::new_random(2);
        let mask = BinaryHV::new_random(99);

        let true_sim = a.similarity(&b);

        let enc_a = EncryptedHV::encrypt(&a, &mask);
        let enc_b = EncryptedHV::encrypt(&b, &mask);
        let encrypted_sim = enc_a.encrypted_similarity(&enc_b);

        assert!(
            (encrypted_sim - true_sim).abs() < 0.001,
            "Same-mask similarity should be preserved: true={true_sim}, encrypted={encrypted_sim}"
        );
    }

    #[test]
    fn test_different_masks_destroy_similarity() {
        let a = BinaryHV::new_random(1);
        let b = BinaryHV::new_random(1); // Same vector!
        let mask_a = BinaryHV::new_random(10);
        let mask_b = BinaryHV::new_random(20);

        let enc_a = EncryptedHV::encrypt(&a, &mask_a);
        let enc_b = EncryptedHV::encrypt(&b, &mask_b);
        let sim = enc_a.encrypted_similarity(&enc_b);

        assert!(
            (sim - 0.5).abs() < 0.05,
            "Different-mask similarity should be ~0.5 (random), got {sim}"
        );
    }

    #[test]
    fn test_collective_pool_basic() {
        let mut pool = CollectiveWisdomPool::new();
        assert!(pool.is_empty());
        assert_eq!(pool.contribution_count(), 0);

        let mask = BinaryHV::new_random(99);
        let wisdom = BinaryHV::new_random(1);
        let encrypted = EncryptedHV::encrypt(&wisdom, &mask);

        assert!(pool.contribute("peer-1", encrypted));
        assert_eq!(pool.contribution_count(), 1);
        assert!(!pool.is_empty());
    }

    #[test]
    fn test_collective_pool_aggregate() {
        let mask = BinaryHV::new_random(99);
        let mut pool = CollectiveWisdomPool::new();

        for i in 0..5 {
            let wisdom = BinaryHV::new_random(i as u64);
            let encrypted = EncryptedHV::encrypt(&wisdom, &mask);
            pool.contribute(&format!("peer-{i}"), encrypted);
        }

        let aggregate = pool.aggregate().expect("should aggregate");
        let decrypted_aggregate = aggregate.decrypt(&mask);

        let plaintexts: Vec<BinaryHV> = (0..5).map(|i| BinaryHV::new_random(i as u64)).collect();
        let expected_bundle = BinaryHV::bundle(&plaintexts);

        let sim = decrypted_aggregate.similarity(&expected_bundle);
        assert!(
            sim > 0.85,
            "Decrypted aggregate should be close to plaintext bundle, got sim={sim}"
        );
    }

    #[test]
    fn test_collective_pool_with_threshold_mask() {
        let (mask, shares) = generate_collective_mask(3, 5, 42);

        let mut pool = CollectiveWisdomPool::new();
        for i in 0..5 {
            let wisdom = BinaryHV::new_random(100 + i as u64);
            let encrypted = EncryptedHV::encrypt(&wisdom, &mask);
            pool.contribute(&format!("peer-{i}"), encrypted);
        }

        let aggregate = pool.aggregate().expect("should aggregate");

        let recovered_mask = HdcThresholdSharing::recover(&shares[..3]);
        assert_eq!(recovered_mask, mask, "3-of-5 should recover mask exactly");

        let decrypted = aggregate.decrypt(&recovered_mask);
        let first_wisdom = BinaryHV::new_random(100);
        let sim = decrypted.similarity(&first_wisdom);
        assert!(
            sim > 0.5,
            "Decrypted aggregate should be closer to contributors than random, got {sim}"
        );
    }

    #[test]
    fn test_collective_pool_capacity_limit() {
        let mut pool = CollectiveWisdomPool::with_capacity(3);
        let mask = BinaryHV::new_random(99);

        for i in 0..3 {
            let enc = EncryptedHV::encrypt(&BinaryHV::new_random(i as u64), &mask);
            assert!(pool.contribute(&format!("peer-{i}"), enc));
        }

        let enc = EncryptedHV::encrypt(&BinaryHV::new_random(99), &mask);
        assert!(!pool.contribute("peer-overflow", enc));
        assert_eq!(pool.contribution_count(), 3);
    }

    #[test]
    fn test_collective_pool_clear() {
        let mask = BinaryHV::new_random(99);
        let mut pool = CollectiveWisdomPool::new();
        pool.contribute(
            "peer-1",
            EncryptedHV::encrypt(&BinaryHV::new_random(1), &mask),
        );
        assert_eq!(pool.contribution_count(), 1);

        pool.clear();
        assert!(pool.is_empty());
        assert_eq!(pool.contribution_count(), 0);
    }

    #[test]
    fn test_aggregate_empty_pool_returns_none() {
        let pool = CollectiveWisdomPool::new();
        assert!(pool.aggregate().is_none());
    }

    #[test]
    fn test_contributor_ids_tracked() {
        let mask = BinaryHV::new_random(99);
        let mut pool = CollectiveWisdomPool::new();
        pool.contribute(
            "alice",
            EncryptedHV::encrypt(&BinaryHV::new_random(1), &mask),
        );
        pool.contribute("bob", EncryptedHV::encrypt(&BinaryHV::new_random(2), &mask));
        assert_eq!(pool.contributors(), &["alice", "bob"]);
    }
}
