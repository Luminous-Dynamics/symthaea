// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # Shared-Mask HDC Algebra Demonstration
//!
//! **QUARANTINED:** this module demonstrates XOR and majority-vote algebra. It
//! is not FHE. Shared-mask operations reveal pairwise XOR/Hamming relationships
//! and must not be treated as privacy preserving.
//!
//! Binary hypervectors support natural homomorphic operations:
//!
//! ```text
//! bind(enc(A), enc(B)) = enc(bind(A, B))       [XOR distributes over XOR]
//! bundle(enc(A), enc(B)) ~ enc(bundle(A, B))   [majority vote preserves under shared mask]
//! ```
//!
//! ## Single-use XOR fact
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
//! - XOR binding distributes algebraically when masks are combined accordingly.
//!
//! ## Limitations
//!
//! - Masks must come from a genuine entropy source and must never be reused (OTP constraint).
//! - Majority-vote bundling under a shared mask is **exact for an odd number of
//!   contributors** (ties are structurally impossible) and only **approximate for an
//!   even number** (a tied local tally resolves to 0 both in plaintext and ciphertext,
//!   but if the shared mask bit is 1, decrypting the ciphertext tally flips that 0 to 1,
//!   disagreeing with the plaintext bundle). See
//!   `legacy_attack_even_count_bundle_disagrees_with_plaintext` for a concrete
//!   counterexample. The earlier blanket "approximate, not exact" claim understated
//!   this: it is exact for odd counts and has this specific failure mode for even ones.
//! - This is NOT a general-purpose FHE scheme (supports only HDC algebra).
//! - The pool below deliberately reuses one mask, exposing pairwise plaintext
//!   XORs and exact Hamming distances.
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
///
/// # Quarantine note
///
/// This is a bare XOR transform with no authentication, no integrity check, and
/// no protection against mask reuse by the caller -- the single-use discipline
/// the "perfect secrecy" claim depends on is entirely the caller's
/// responsibility and nothing here enforces or even detects a violation.
/// [`CollectiveWisdomPool`] in this same module *is* such a violation (it
/// reuses one mask across every contribution). Do not treat this type as a
/// general-purpose encryption primitive.
#[derive(Clone, PartialEq, Eq)]
#[deprecated(
    note = "correct only as a single-use OTP XOR transform with a secure-random mask you track yourself; provides no authentication, integrity, or misuse resistance"
)]
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

/// Shared-mask collective aggregation demonstration. This is not
/// privacy-preserving aggregation.
///
/// Each peer transforms a wisdom vector with the same mask and contributes the
/// result. Reusing that mask leaks pairwise plaintext relationships, and the
/// compatibility `HdcThresholdSharing` records each reveal the mask alone.
///
/// # Protocol
///
/// ```text
/// 1. Coordinator generates collective mask, splits into (k,n) shares
/// 2. Each peer receives one mask share
/// 3. Each peer encrypts their wisdom: enc(wisdom_i, collective_mask)
/// 4. Pool bundles encrypted contributions
/// 5. any one broken compatibility share can reconstruct the mask
/// ```
///
/// # Known failed invariants (attack-study subjects, not bugs to silently patch)
///
/// - No check that all contributions actually share the same mask or round;
///   the aggregate is meaningless algebra if they don't.
/// - Duplicate `contributor_ids` are accepted, so one participant can submit
///   multiple contributions and disproportionately weight the majority vote.
/// - No authentication or provenance on any contribution.
/// - `with_capacity(n)` sets `max_size` to the caller-supplied `n` with no
///   ceiling; only the *internal `Vec`'s pre-allocation* is capped at
///   [`DEFAULT_MAX_POOL_SIZE`]. The "256 default safety bound" describes the
///   default constructor, not an enforced ceiling on every pool.
/// - Aggregation is exact only for an odd `contribution_count()` (see
///   [`Self::aggregate`]).
#[deprecated(
    note = "shared-mask aggregation toy model; leaks pairwise relationships and is not FHE"
)]
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

    /// Compute a shared-mask algebraic aggregate via majority-vote bundling.
    ///
    /// All contributions must be encrypted with the **same mask** for the
    /// bundling to preserve semantic content. This is exact when
    /// `contribution_count()` is odd and only approximate when it is even
    /// (see the module-level "Limitations" section for why).
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

/// Generate a mask and broken compatibility share records.
///
/// Returns `(mask, shares)` where `mask` is the full mask (held only transiently)
/// Each returned record independently reveals the mask; do not distribute it
/// as if it were a threshold share.
///
/// The full mask should be zeroized after share distribution.
#[deprecated(note = "uses the broken compatibility threshold-sharing construction")]
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

    /// Renamed from `test_otp_ciphertext_hides_plaintext`: a low-similarity
    /// sample under one deterministic-seed mask does not establish hiding
    /// (the whole point of this crate is that it does not). This is a
    /// decorrelation observation on one sample, nothing more.
    #[test]
    fn deterministic_xor_mask_decorrelates_one_sample() {
        let plaintext = BinaryHV::new_random(42);
        let mask = BinaryHV::new_random(99);
        let encrypted = EncryptedHV::encrypt(&plaintext, &mask);
        let sim = encrypted.ciphertext.similarity(&plaintext);
        assert!(
            (sim - 0.5).abs() < 0.05,
            "Ciphertext should be ~0.5 similar to plaintext for this one sample, got {sim}"
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

    /// CI-004: mask cancellation exposes the exact pairwise XOR relation and
    /// therefore the exact Hamming similarity of both plaintexts.
    #[test]
    fn legacy_attack_reused_mask_leaks_pairwise_relationships() {
        let a = BinaryHV::new_random(1);
        let b = BinaryHV::new_random(2);
        let shared_mask = BinaryHV::new_random(99);
        let enc_a = EncryptedHV::encrypt(&a, &shared_mask);
        let enc_b = EncryptedHV::encrypt(&b, &shared_mask);

        assert_eq!(
            enc_a.ciphertext.bind(&enc_b.ciphertext),
            a.bind(&b),
            "the shared mask cancels exactly"
        );
        assert_eq!(enc_a.encrypted_similarity(&enc_b), a.similarity(&b));
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
    fn test_collective_pool_aggregate_odd_count_is_exact() {
        let mask = BinaryHV::new_random(99);
        let mut pool = CollectiveWisdomPool::new();

        // 5 contributors: an odd count, so majority-vote ties are structurally
        // impossible and mask cancellation is exact -- see the module docs.
        for i in 0..5 {
            let wisdom = BinaryHV::new_random(i as u64);
            let encrypted = EncryptedHV::encrypt(&wisdom, &mask);
            pool.contribute(&format!("peer-{i}"), encrypted);
        }

        let aggregate = pool.aggregate().expect("should aggregate");
        let decrypted_aggregate = aggregate.decrypt(&mask);

        let plaintexts: Vec<BinaryHV> = (0..5).map(|i| BinaryHV::new_random(i as u64)).collect();
        let expected_bundle = BinaryHV::bundle(&plaintexts);

        assert_eq!(
            decrypted_aggregate, expected_bundle,
            "odd contributor count must decrypt to the exact plaintext bundle"
        );
    }

    /// CI-004 (extended): for an even contributor count, a locally-tied bit
    /// resolves to 0 in both the plaintext and ciphertext tallies, but if the
    /// shared mask bit there is 1, decrypting the ciphertext tally flips that
    /// 0 to 1 -- disagreeing with the plaintext bundle. This falsifies the
    /// blanket "bundle(enc(A),enc(B)) ~ enc(bundle(A,B))" claim for even
    /// counts; it is only exact for odd counts.
    #[test]
    fn legacy_attack_even_count_bundle_disagrees_with_plaintext() {
        // Construct two 1-bit-wide vectors (only bit 0 matters here) with a
        // mask whose bit 0 is 1, and plaintexts that disagree in bit 0 --
        // i.e. a tied 2-contributor local tally.
        let mut p0_bytes = [0u8; BinaryHV::BYTES];
        let mut p1_bytes = [0u8; BinaryHV::BYTES];
        let mut mask_bytes = [0u8; BinaryHV::BYTES];
        p0_bytes[0] = 0b0000_0000; // bit 0 = 0
        p1_bytes[0] = 0b0000_0001; // bit 0 = 1
        mask_bytes[0] = 0b0000_0001; // mask bit 0 = 1

        let p0 = BinaryHV(p0_bytes);
        let p1 = BinaryHV(p1_bytes);
        let mask = BinaryHV(mask_bytes);

        let plaintext_bundle = BinaryHV::bundle(&[p0, p1]);
        assert_eq!(
            plaintext_bundle.0[0] & 1,
            0,
            "tied plaintext bit must resolve to 0 (count=1 is not > threshold=1)"
        );

        let enc0 = EncryptedHV::encrypt(&p0, &mask);
        let enc1 = EncryptedHV::encrypt(&p1, &mask);
        let ciphertext_bundle = BinaryHV::bundle(&[enc0.ciphertext, enc1.ciphertext]);
        let decrypted = ciphertext_bundle.bind(&mask);

        assert_eq!(
            decrypted.0[0] & 1,
            1,
            "decrypting the tied ciphertext tally flips the bit versus the plaintext bundle"
        );
        assert_ne!(
            decrypted, plaintext_bundle,
            "even-count aggregation must disagree with the plaintext bundle here"
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
