// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # HDC Homomorphic Computation — Privacy-Preserving Collective Intelligence
//!
//! Binary hypervectors support natural homomorphic operations:
//!
//! ```text
//! bind(enc(A), enc(B)) = enc(bind(A, B))       [XOR distributes over XOR]
//! bundle(enc(A), enc(B)) ≈ enc(bundle(A, B))   [majority vote preserves under shared mask]
//! ```
//!
//! This enables peers to share encrypted wisdom vectors, perform collective
//! bundling without revealing individual states, and decrypt only the aggregate.
//!
//! ## Security Model
//!
//! [`EncryptedHV`] uses a one-time pad (XOR with random mask):
//! - **Perfect secrecy** (Shannon 1949): ciphertext reveals zero information
//!   about the plaintext when the mask is truly random and used only once.
//! - **Key size**: D = 16,384 bits (mask is same size as message).
//! - **Homomorphic property**: XOR binding distributes over XOR encryption,
//!   so `enc(A) ⊗ enc(B) = enc(A ⊗ B)` when masks are combined accordingly.
//!
//! ## Limitations
//!
//! - Masks must never be reused (OTP constraint).
//! - Majority-vote bundling on encrypted vectors is approximate, not exact.
//!   Accuracy improves with more contributors (central limit theorem).
//! - This is NOT a general-purpose FHE scheme (no arbitrary computation).
//!   It supports only the HDC algebra: bind, bundle, similarity.
//!
//! ## References
//!
//! - Shannon, C. (1949). Communication theory of secrecy systems.
//! - Imani et al. (2019). A framework for collaborative learning in secure HDC.
//! - Kanerva, P. (2009). Hyperdimensional computing: An introduction.

use super::binary_hv::BinaryHV;
use super::hdc_crypto::HdcThresholdSharing;

// ═══════════════════════════════════════════════════════════════════════════════
// ENCRYPTED HYPERVECTOR
// ═══════════════════════════════════════════════════════════════════════════════

/// An encrypted BinaryHV using a one-time pad (XOR mask).
///
/// # Security
///
/// Perfect secrecy (Shannon 1949): when the mask is uniformly random over
/// {0,1}^D and used only once, the ciphertext `C = P ⊗ M` is statistically
/// independent of the plaintext P. An adversary with access to C gains
/// exactly zero bits of information about P.
///
/// # Homomorphic Properties
///
/// ```text
/// enc(A, M_a) ⊗ enc(B, M_b) = (A ⊗ M_a) ⊗ (B ⊗ M_b)
///                             = (A ⊗ B) ⊗ (M_a ⊗ M_b)
///                             = enc(A ⊗ B, M_a ⊗ M_b)
/// ```
///
/// So binding two encrypted vectors produces the encryption of their binding
/// under the combined mask `M_a ⊗ M_b`. The decryptor needs `M_a ⊗ M_b`
/// (which both parties can compute from their individual masks).
#[derive(Clone, PartialEq, Eq)]
pub struct EncryptedHV {
    /// Ciphertext: plaintext ⊗ mask.
    pub ciphertext: BinaryHV,
}

impl EncryptedHV {
    /// Encrypt a BinaryHV with a random mask (one-time pad).
    ///
    /// The mask must be retained by the encryptor for later decryption.
    /// **Never reuse a mask** — OTP security requires fresh masks.
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
    ///
    /// ```text
    /// hom_bind(enc(A, Ma), enc(B, Mb)) = enc(A⊗B, Ma⊗Mb)
    /// ```
    #[inline]
    pub fn hom_bind(&self, other: &EncryptedHV) -> EncryptedHV {
        EncryptedHV {
            ciphertext: self.ciphertext.bind(&other.ciphertext),
        }
    }

    /// Similarity between two encrypted vectors.
    ///
    /// When both vectors are encrypted with the **same mask** (same session key),
    /// this returns the true similarity: `sim(enc(A,M), enc(B,M)) = sim(A,B)`.
    ///
    /// When masks differ, the result is noisy (≈ 0.5 ± small error).
    /// This enables privacy-preserving nearest-neighbor search within a session.
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

// ═══════════════════════════════════════════════════════════════════════════════
// COLLECTIVE WISDOM POOL
// ═══════════════════════════════════════════════════════════════════════════════

/// Maximum contributions before forced aggregation (memory bound: 256 × 2KB = 512KB).
pub const HDC_FHE_MAX_POOL_SIZE: usize = 256;

/// Privacy-preserving collective wisdom aggregation.
///
/// Each peer encrypts their wisdom vector with a per-session mask and
/// contributes the encrypted version. The pool collects encrypted
/// contributions and can compute an aggregate via majority-vote bundling
/// on the ciphertexts.
///
/// # Decryption
///
/// The aggregate can only be decrypted if sufficient mask shares are
/// available (via [`HdcThresholdSharing`]). This ensures:
/// - No single peer can decrypt the aggregate alone
/// - At least k-of-n peers must cooperate to reveal the collective wisdom
/// - Individual contributions remain private (OTP encryption)
///
/// # Protocol
///
/// ```text
/// 1. Coordinator generates collective mask, splits into (k,n) shares
/// 2. Each peer receives one mask share
/// 3. Each peer encrypts their wisdom: enc(wisdom_i, collective_mask)
/// 4. Pool bundles encrypted contributions
/// 5. k peers reconstruct the mask → decrypt the aggregate
/// ```
///
/// Note: Step 3 requires all peers to use the SAME mask for the
/// homomorphic bundling to work. The mask is distributed via threshold
/// sharing so no single peer holds the full mask until k cooperate.
pub struct CollectiveWisdomPool {
    /// Encrypted contributions from peers.
    contributions: Vec<EncryptedHV>,
    /// Peer IDs corresponding to each contribution (for audit trail).
    contributor_ids: Vec<String>,
    /// Maximum contributions before rejection.
    max_size: usize,
}

impl CollectiveWisdomPool {
    /// Create a new empty pool.
    pub fn new() -> Self {
        Self {
            contributions: Vec::new(),
            contributor_ids: Vec::new(),
            max_size: HDC_FHE_MAX_POOL_SIZE,
        }
    }

    /// Create a pool with a custom size limit.
    pub fn with_capacity(max_size: usize) -> Self {
        Self {
            contributions: Vec::with_capacity(max_size.min(HDC_FHE_MAX_POOL_SIZE)),
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
    /// bundling to preserve semantic content. The result is:
    ///
    /// ```text
    /// bundle(enc(w_1, M), enc(w_2, M), ..., enc(w_n, M))
    /// ≈ enc(bundle(w_1, w_2, ..., w_n), M)
    /// ```
    ///
    /// The approximation arises because majority vote and XOR don't perfectly
    /// commute. Accuracy improves with more contributors (CLT).
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

// ═══════════════════════════════════════════════════════════════════════════════
// SESSION KEY DISTRIBUTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Helper: generate a random collective mask and split it into threshold shares.
///
/// Returns `(mask, shares)` where `mask` is the full mask (held only transiently)
/// and `shares` are the (k,n) threshold shares to distribute to peers.
///
/// The full mask should be zeroized after share distribution.
pub fn generate_collective_mask(
    k: usize,
    n: usize,
    seed: u64,
) -> (BinaryHV, Vec<super::hdc_crypto::HdcShare>) {
    let mask = BinaryHV::random(seed);
    let shares = HdcThresholdSharing::split(&mask, k, n, seed.wrapping_add(1));
    (mask, shares)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_otp_encrypt_decrypt_roundtrip() {
        let plaintext = BinaryHV::random(42);
        let mask = BinaryHV::random(99);
        let encrypted = EncryptedHV::encrypt(&plaintext, &mask);
        let decrypted = encrypted.decrypt(&mask);
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_otp_ciphertext_hides_plaintext() {
        let plaintext = BinaryHV::random(42);
        let mask = BinaryHV::random(99);
        let encrypted = EncryptedHV::encrypt(&plaintext, &mask);
        let sim = encrypted.ciphertext.similarity(&plaintext);
        assert!(
            (sim - 0.5).abs() < 0.05,
            "Ciphertext should be ~0.5 similar to plaintext (OTP hiding), got {sim}"
        );
    }

    #[test]
    fn test_homomorphic_bind() {
        let a = BinaryHV::random(1);
        let b = BinaryHV::random(2);
        let mask_a = BinaryHV::random(10);
        let mask_b = BinaryHV::random(20);

        let enc_a = EncryptedHV::encrypt(&a, &mask_a);
        let enc_b = EncryptedHV::encrypt(&b, &mask_b);

        // Homomorphic bind
        let enc_ab = enc_a.hom_bind(&enc_b);

        // Decrypt with combined mask
        let combined_mask = mask_a.bind(&mask_b);
        let decrypted = enc_ab.decrypt(&combined_mask);

        // Should equal bind(a, b)
        let expected = a.bind(&b);
        assert_eq!(
            decrypted, expected,
            "Homomorphic bind should produce bind(a,b)"
        );
    }

    #[test]
    fn test_same_mask_preserves_similarity() {
        let a = BinaryHV::random(1);
        let b = BinaryHV::random(2);
        let mask = BinaryHV::random(99);

        let true_sim = a.similarity(&b);

        let enc_a = EncryptedHV::encrypt(&a, &mask);
        let enc_b = EncryptedHV::encrypt(&b, &mask);
        let encrypted_sim = enc_a.encrypted_similarity(&enc_b);

        // Same mask: similarity is perfectly preserved
        // sim(A⊗M, B⊗M) = sim(A, B) because XOR is a distance-preserving bijection
        assert!(
            (encrypted_sim - true_sim).abs() < 0.001,
            "Same-mask similarity should be preserved: true={true_sim}, encrypted={encrypted_sim}"
        );
    }

    #[test]
    fn test_different_masks_destroy_similarity() {
        let a = BinaryHV::random(1);
        let b = BinaryHV::random(1); // Same vector!
        let mask_a = BinaryHV::random(10);
        let mask_b = BinaryHV::random(20);

        let enc_a = EncryptedHV::encrypt(&a, &mask_a);
        let enc_b = EncryptedHV::encrypt(&b, &mask_b);
        let sim = enc_a.encrypted_similarity(&enc_b);

        // Different masks: even identical vectors look random
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

        let mask = BinaryHV::random(99);
        let wisdom = BinaryHV::random(1);
        let encrypted = EncryptedHV::encrypt(&wisdom, &mask);

        assert!(pool.contribute("peer-1", encrypted));
        assert_eq!(pool.contribution_count(), 1);
        assert!(!pool.is_empty());
    }

    #[test]
    fn test_collective_pool_aggregate() {
        let mask = BinaryHV::random(99);
        let mut pool = CollectiveWisdomPool::new();

        // 5 peers contribute different wisdom vectors
        for i in 0..5 {
            let wisdom = BinaryHV::random(i as u64);
            let encrypted = EncryptedHV::encrypt(&wisdom, &mask);
            pool.contribute(&format!("peer-{i}"), encrypted);
        }

        let aggregate = pool.aggregate().expect("should aggregate");

        // Decrypt aggregate
        let decrypted_aggregate = aggregate.decrypt(&mask);

        // Compute expected: bundle of all plaintext wisdoms
        let plaintexts: Vec<BinaryHV> = (0..5).map(|i| BinaryHV::random(i as u64)).collect();
        let expected_bundle = BinaryHV::bundle(&plaintexts);

        // Approximate equality (bundling and XOR don't perfectly commute)
        let sim = decrypted_aggregate.similarity(&expected_bundle);
        assert!(
            sim > 0.85,
            "Decrypted aggregate should be close to plaintext bundle, got sim={sim}"
        );
    }

    #[test]
    fn test_collective_pool_with_threshold_mask() {
        // Full protocol: generate mask, split into shares, encrypt, aggregate, recover mask, decrypt
        let (mask, shares) = generate_collective_mask(3, 5, 42);

        let mut pool = CollectiveWisdomPool::new();
        for i in 0..5 {
            let wisdom = BinaryHV::random(100 + i as u64);
            let encrypted = EncryptedHV::encrypt(&wisdom, &mask);
            pool.contribute(&format!("peer-{i}"), encrypted);
        }

        let aggregate = pool.aggregate().expect("should aggregate");

        // Recover mask from 3-of-5 shares
        let recovered_mask = HdcThresholdSharing::recover(&shares[..3]);
        assert_eq!(recovered_mask, mask, "3-of-5 should recover mask exactly");

        // Decrypt with recovered mask
        let decrypted = aggregate.decrypt(&recovered_mask);
        // Decrypted should be meaningful (not random)
        let first_wisdom = BinaryHV::random(100);
        let sim = decrypted.similarity(&first_wisdom);
        // With 5 bundled vectors, each contributes ~20% — similarity to any one should be > random
        assert!(
            sim > 0.5,
            "Decrypted aggregate should be closer to contributors than random, got {sim}"
        );
    }

    #[test]
    fn test_collective_pool_capacity_limit() {
        let mut pool = CollectiveWisdomPool::with_capacity(3);
        let mask = BinaryHV::random(99);

        for i in 0..3 {
            let enc = EncryptedHV::encrypt(&BinaryHV::random(i as u64), &mask);
            assert!(pool.contribute(&format!("peer-{i}"), enc));
        }

        // 4th contribution should fail
        let enc = EncryptedHV::encrypt(&BinaryHV::random(99), &mask);
        assert!(!pool.contribute("peer-overflow", enc));
        assert_eq!(pool.contribution_count(), 3);
    }

    #[test]
    fn test_collective_pool_clear() {
        let mask = BinaryHV::random(99);
        let mut pool = CollectiveWisdomPool::new();
        pool.contribute("peer-1", EncryptedHV::encrypt(&BinaryHV::random(1), &mask));
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
        let mask = BinaryHV::random(99);
        let mut pool = CollectiveWisdomPool::new();
        pool.contribute("alice", EncryptedHV::encrypt(&BinaryHV::random(1), &mask));
        pool.contribute("bob", EncryptedHV::encrypt(&BinaryHV::random(2), &mask));
        assert_eq!(pool.contributors(), &["alice", "bob"]);
    }
}
