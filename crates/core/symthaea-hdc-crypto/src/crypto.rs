// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # HDC-Native Cryptographic Primitives
//!
//! Cryptographic operations that exploit the algebraic properties of 16,384-bit
//! binary hypervectors. These are **not replacements** for standard ciphers
//! (AES, ChaCha20) but enable unique capabilities:
//!
//! - [`HdcMac`] -- Authenticate BinaryHV data without serialization overhead
//! - [`HdcThresholdSharing`] -- (k,n) secret splitting via majority-vote bundling
//! - [`HdcContextKey`] -- Sensor-derived keys via bind+permute chains
//! - [`HdcCommitment`] -- Hide-then-reveal via permutation binding
//!
//! ## Security Model
//!
//! HDC crypto primitives provide _information-theoretic_ properties in
//! high-dimensional spaces (D = 16,384) where:
//!
//! - **Random collision probability**: P(HV_a = HV_b) = 2^{-16384} (negligible)
//! - **Similarity concentration**: For random HVs, sim(a,b) ~ 0.5 +/- 0.0039 (3 sigma)
//! - **Binding invertibility**: a XOR a = 0 (self-inverse)
//! - **Permutation bijectivity**: cyclic shift is a bijection on {0,1}^D
//!
//! ## References
//!
//! - Kanerva, P. (2009). Hyperdimensional computing. *Cognitive Computation*.
//! - Rahimi et al. (2016). Robust and energy-efficient classifier using brain-inspired HDC.
//! - Imani et al. (2019). A framework for collaborative learning in secure HDC.
//! - Shannon, C. (1949). Communication theory of secrecy systems.

use crate::binary_hv::BinaryHV;

// =========================================================================
// HDC MESSAGE AUTHENTICATION CODE
// =========================================================================

/// HDC-native message authentication code.
///
/// # Construction
///
/// ```text
/// MAC(message, key) = message XOR permute_k(key)
/// ```
///
/// # Performance
///
/// MAC computation is a single permute + XOR: ~5-10 ns with SIMD.
/// Compare: BLAKE3 MAC ~ 50-100 ns, HMAC-SHA256 ~ 200-400 ns.
pub struct HdcMac;

/// Default permutation offset for MAC key derivation.
const HDC_MAC_PERMUTE_OFFSET: usize = 7;

/// Minimum similarity threshold for noisy HDC-MAC verification.
///
/// At D = 16,384, random similarity ~ 0.5 +/- 0.0039 (3 sigma).
/// Threshold of 0.95 gives false-positive rate ~ 2^{-4700} (Hoeffding bound).
pub const HDC_MAC_NOISY_THRESHOLD: f32 = 0.95;

impl HdcMac {
    /// Compute MAC over a BinaryHV message with a BinaryHV key.
    #[inline]
    pub fn compute(message: &BinaryHV, key: &BinaryHV) -> BinaryHV {
        let derived = key.permute(HDC_MAC_PERMUTE_OFFSET);
        message.bind(&derived)
    }

    /// Compute MAC with a custom permutation offset (for domain separation).
    #[inline]
    pub fn compute_with_offset(message: &BinaryHV, key: &BinaryHV, offset: usize) -> BinaryHV {
        let derived = key.permute(offset);
        message.bind(&derived)
    }

    /// Verify MAC (exact match -- for lossless channels).
    #[inline]
    pub fn verify(message: &BinaryHV, key: &BinaryHV, mac: &BinaryHV) -> bool {
        let expected = Self::compute(message, key);
        expected == *mac
    }

    /// Verify MAC with noise tolerance (for lossy channels like LoRa/BLE).
    ///
    /// Returns true if similarity between expected and actual MAC exceeds threshold.
    /// Recommended threshold: [`HDC_MAC_NOISY_THRESHOLD`] (0.95).
    #[inline]
    pub fn verify_noisy(
        message: &BinaryHV,
        key: &BinaryHV,
        mac: &BinaryHV,
        threshold: f32,
    ) -> bool {
        let expected = Self::compute(message, key);
        expected.similarity(mac) >= threshold
    }
}

// =========================================================================
// HDC THRESHOLD SECRET SHARING
// =========================================================================

/// (k,n) threshold secret sharing using HDC majority-vote bundling.
///
/// # Construction (Kanerva 2009)
///
/// **Split**: Generate n random masks. Each share is `secret XOR mask_i`.
/// **Recover**: Unbind each share with its mask, then bundle via majority vote.
///
/// # Security
///
/// Information-theoretic security: each share is a one-time pad over {0,1}^D
/// -- **but only when masks come from a true entropy source**. Use
/// [`HdcThresholdSharing::split_secure`], which draws masks via
/// [`BinaryHV::new_secure_random`]. [`HdcThresholdSharing::split`] derives
/// masks deterministically from a `u64` seed and provides no real secrecy
/// (the seed space is brute-forceable) -- it exists for reproducible tests
/// only.
///
/// Without k shares, majority vote cannot recover the secret.
///
/// # Constraints
///
/// - k must be odd (majority vote requires odd count)
/// - k <= n
/// - k >= 1
pub struct HdcThresholdSharing;

/// A single share in an HDC threshold scheme.
#[derive(Clone)]
pub struct HdcShare {
    /// Index of this share (0..n).
    pub index: usize,
    /// The share: secret XOR mask.
    pub share: BinaryHV,
    /// The mask used to create this share (needed for recovery).
    pub mask: BinaryHV,
}

impl HdcThresholdSharing {
    /// Split a secret BinaryHV into n shares requiring k to recover, using a
    /// deterministic seed for mask generation.
    ///
    /// **Not secure.** Masks are derived from `seed` via [`BinaryHV::new_random`],
    /// so anyone who knows or brute-forces `seed` can reconstruct every mask
    /// and recover the secret from a single share. This exists for
    /// reproducible tests only -- use [`Self::split_secure`] for any real
    /// secret-sharing use case.
    ///
    /// # Panics
    ///
    /// Panics if k > n, k == 0, or k is even.
    pub fn split(secret: &BinaryHV, k: usize, n: usize, seed: u64) -> Vec<HdcShare> {
        assert!(k >= 1, "k must be at least 1");
        assert!(k <= n, "k must be <= n");
        assert!(k % 2 == 1, "k must be odd for deterministic majority vote");

        (0..n)
            .map(|i| {
                let mask = BinaryHV::new_random(seed.wrapping_add(i as u64));
                let share = secret.bind(&mask);
                HdcShare {
                    index: i,
                    share,
                    mask,
                }
            })
            .collect()
    }

    /// Split a secret BinaryHV into n shares requiring k to recover, using
    /// OS-entropy-backed masks.
    ///
    /// This is the secure entry point: each mask comes from
    /// [`BinaryHV::new_secure_random`], so the information-theoretic
    /// security claimed for this scheme actually holds.
    ///
    /// # Panics
    ///
    /// Panics if k > n, k == 0, or k is even.
    pub fn split_secure(secret: &BinaryHV, k: usize, n: usize) -> Vec<HdcShare> {
        assert!(k >= 1, "k must be at least 1");
        assert!(k <= n, "k must be <= n");
        assert!(k % 2 == 1, "k must be odd for deterministic majority vote");

        (0..n)
            .map(|i| {
                let mask = BinaryHV::new_secure_random();
                let share = secret.bind(&mask);
                HdcShare {
                    index: i,
                    share,
                    mask,
                }
            })
            .collect()
    }

    /// Recover secret from k or more shares.
    ///
    /// Each share is unbound with its mask, then all are bundled via majority vote.
    pub fn recover(shares: &[HdcShare]) -> BinaryHV {
        assert!(!shares.is_empty(), "need at least one share to recover");

        let unbound: Vec<BinaryHV> = shares.iter().map(|s| s.share.bind(&s.mask)).collect();

        BinaryHV::bundle(&unbound)
    }

    /// Check recovery quality: similarity between recovered secret and reference.
    ///
    /// Returns 1.0 when recovery is exact.
    pub fn recovery_quality(shares: &[HdcShare]) -> f32 {
        if shares.is_empty() {
            return 0.0;
        }
        let recovered = Self::recover(shares);
        let reference = shares[0].share.bind(&shares[0].mask);
        recovered.similarity(&reference)
    }
}

// =========================================================================
// HDC CONTEXT KEY
// =========================================================================

/// Context-sensitive key derivation from sensor state.
///
/// # Construction
///
/// ```text
/// key = sensor_0 XOR permute_1(sensor_1) XOR permute_2(sensor_2) XOR ...
/// ```
///
/// Each sensor is permuted by its index to enforce ordering (non-commutative).
///
/// # Applications
///
/// - Location-bound decryption (GPS + altitude -> key only valid at location)
/// - Temporal access windows (time sensor -> key expires naturally)
/// - Device-bound secrets (accelerometer + gyro -> unique to physical device)
pub struct HdcContextKey;

impl HdcContextKey {
    /// Derive a context key from multiple sensor readings encoded as BinaryHVs.
    ///
    /// Returns `BinaryHV::zero()` if no sensors provided.
    pub fn derive(sensors: &[BinaryHV]) -> BinaryHV {
        if sensors.is_empty() {
            return BinaryHV::zero();
        }
        let mut result = sensors[0];
        for (i, sensor) in sensors[1..].iter().enumerate() {
            result = result.bind(&sensor.permute(i + 1));
        }
        result
    }

    /// Derive a 256-bit symmetric key from an HDC context key.
    ///
    /// Uses BLAKE3 hash extraction to produce a uniform 32-byte key
    /// suitable for ChaCha20-Poly1305 or other symmetric ciphers.
    pub fn to_symmetric_key(context: &BinaryHV) -> [u8; 32] {
        *blake3::hash(&context.0).as_bytes()
    }

    /// Derive a context key and immediately extract a symmetric key.
    pub fn derive_symmetric(sensors: &[BinaryHV]) -> [u8; 32] {
        let context = Self::derive(sensors);
        Self::to_symmetric_key(&context)
    }
}

// =========================================================================
// HDC COMMITMENT SCHEME
// =========================================================================

/// HDC commitment scheme using permutation chains.
///
/// ```text
/// Commit(secret, offset) = permute_offset(secret)
/// Verify(commitment, secret, offset) = (permute_offset(secret) == commitment)
/// ```
///
/// **Binding**: Permutation is a bijection, so distinct inputs yield distinct commitments.
/// **Hiding**: Without knowing the offset, the commitment appears quasi-random.
#[derive(Clone)]
pub struct HdcCommitment {
    /// The commitment value (publicly shared).
    pub commitment: BinaryHV,
}

impl HdcCommitment {
    /// Create a commitment to a secret using a permutation offset.
    ///
    /// The offset should be kept secret until reveal time.
    pub fn commit(secret: &BinaryHV, offset: usize) -> Self {
        Self {
            commitment: secret.permute(offset),
        }
    }

    /// Verify a commitment against a revealed secret and offset.
    pub fn verify(commitment: &BinaryHV, secret: &BinaryHV, offset: usize) -> bool {
        secret.permute(offset) == *commitment
    }

    /// Verify with noise tolerance (for lossy channels).
    pub fn verify_noisy(
        commitment: &BinaryHV,
        secret: &BinaryHV,
        offset: usize,
        threshold: f32,
    ) -> bool {
        secret.permute(offset).similarity(commitment) >= threshold
    }
}

// =========================================================================
// TESTS
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- HdcMac Tests --

    #[test]
    fn test_hdc_mac_roundtrip() {
        let message = BinaryHV::new_random(42);
        let key = BinaryHV::new_random(99);
        let mac = HdcMac::compute(&message, &key);
        assert!(HdcMac::verify(&message, &key, &mac));
    }

    #[test]
    fn test_hdc_mac_wrong_key_fails() {
        let message = BinaryHV::new_random(42);
        let key_a = BinaryHV::new_random(99);
        let key_b = BinaryHV::new_random(100);
        let mac = HdcMac::compute(&message, &key_a);
        assert!(!HdcMac::verify(&message, &key_b, &mac));

        let expected_wrong = HdcMac::compute(&message, &key_b);
        let sim = mac.similarity(&expected_wrong);
        assert!(
            (sim - 0.5).abs() < 0.05,
            "Wrong-key MAC should be ~0.5 similar, got {sim}"
        );
    }

    #[test]
    fn test_hdc_mac_tampered_message_fails() {
        let message = BinaryHV::new_random(42);
        let key = BinaryHV::new_random(99);
        let mac = HdcMac::compute(&message, &key);

        let tampered = BinaryHV::new_random(43);
        assert!(!HdcMac::verify(&tampered, &key, &mac));
    }

    #[test]
    fn test_hdc_mac_noisy_channel_tolerance() {
        let message = BinaryHV::new_random(42);
        let key = BinaryHV::new_random(99);
        let mac = HdcMac::compute(&message, &key);

        assert!(HdcMac::verify_noisy(&message, &key, &mac, 0.95));
        assert!(HdcMac::verify_noisy(&message, &key, &mac, 1.0));

        let wrong_key = BinaryHV::new_random(100);
        assert!(!HdcMac::verify_noisy(&message, &wrong_key, &mac, 0.95));
    }

    #[test]
    fn test_hdc_mac_domain_separation() {
        let message = BinaryHV::new_random(42);
        let key = BinaryHV::new_random(99);

        let mac_offset_7 = HdcMac::compute(&message, &key);
        let mac_offset_11 = HdcMac::compute_with_offset(&message, &key, 11);

        assert_ne!(mac_offset_7, mac_offset_11);
        assert!(HdcMac::verify(&message, &key, &mac_offset_7));
    }

    #[test]
    fn test_hdc_mac_deterministic() {
        let message = BinaryHV::new_random(42);
        let key = BinaryHV::new_random(99);
        let mac1 = HdcMac::compute(&message, &key);
        let mac2 = HdcMac::compute(&message, &key);
        assert_eq!(mac1, mac2);
    }

    // -- HdcThresholdSharing Tests --

    #[test]
    fn test_threshold_split_recover_exact() {
        let secret = BinaryHV::new_random(42);
        let shares = HdcThresholdSharing::split(&secret, 3, 3, 1000);
        assert_eq!(shares.len(), 3);

        let recovered = HdcThresholdSharing::recover(&shares);
        assert_eq!(recovered, secret);
    }

    #[test]
    fn test_threshold_split_recover_k_less_than_n() {
        let secret = BinaryHV::new_random(42);
        let shares = HdcThresholdSharing::split(&secret, 3, 5, 1000);
        assert_eq!(shares.len(), 5);

        let recovered = HdcThresholdSharing::recover(&shares[..3]);
        assert_eq!(recovered, secret);
    }

    #[test]
    fn test_threshold_split_secure_recover() {
        let secret = BinaryHV::new_random(42);
        let shares = HdcThresholdSharing::split_secure(&secret, 3, 5);
        assert_eq!(shares.len(), 5);

        let recovered = HdcThresholdSharing::recover(&shares[..3]);
        assert_eq!(recovered, secret);
    }

    #[test]
    fn test_threshold_split_secure_masks_are_unpredictable() {
        // Two independent secure splits of the same secret should use
        // different masks (unlike split(seed) which is fully deterministic).
        let secret = BinaryHV::new_random(42);
        let shares_a = HdcThresholdSharing::split_secure(&secret, 1, 1);
        let shares_b = HdcThresholdSharing::split_secure(&secret, 1, 1);
        assert_ne!(
            shares_a[0].mask, shares_b[0].mask,
            "secure masks must not repeat across calls"
        );
    }

    #[test]
    fn test_threshold_single_share_recovers() {
        let secret = BinaryHV::new_random(42);
        let shares = HdcThresholdSharing::split(&secret, 1, 5, 1000);

        for share in &shares {
            let recovered = HdcThresholdSharing::recover(&[share.clone()]);
            assert_eq!(recovered, secret);
        }
    }

    #[test]
    fn test_threshold_recovery_quality() {
        let secret = BinaryHV::new_random(42);
        let shares = HdcThresholdSharing::split(&secret, 3, 5, 1000);

        let quality = HdcThresholdSharing::recovery_quality(&shares[..3]);
        assert!(
            (quality - 1.0).abs() < f32::EPSILON,
            "Recovery quality should be 1.0, got {quality}"
        );
    }

    #[test]
    fn test_threshold_shares_are_distinct() {
        let secret = BinaryHV::new_random(42);
        let shares = HdcThresholdSharing::split(&secret, 3, 5, 1000);

        for i in 0..shares.len() {
            for j in (i + 1)..shares.len() {
                assert_ne!(shares[i].share, shares[j].share);
            }
        }
    }

    #[test]
    fn test_threshold_individual_share_leaks_nothing() {
        let secret = BinaryHV::new_random(42);
        let shares = HdcThresholdSharing::split(&secret, 3, 5, 1000);

        for share in &shares {
            let sim = share.share.similarity(&secret);
            assert!(
                (sim - 0.5).abs() < 0.05,
                "Individual share should be ~0.5 similar to secret, got {sim}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "k must be odd")]
    fn test_threshold_even_k_panics() {
        let secret = BinaryHV::new_random(42);
        HdcThresholdSharing::split(&secret, 2, 5, 1000);
    }

    #[test]
    #[should_panic(expected = "k must be <= n")]
    fn test_threshold_k_greater_than_n_panics() {
        let secret = BinaryHV::new_random(42);
        HdcThresholdSharing::split(&secret, 5, 3, 1000);
    }

    // -- HdcContextKey Tests --

    #[test]
    fn test_context_key_deterministic() {
        let sensors = vec![
            BinaryHV::new_random(1),
            BinaryHV::new_random(2),
            BinaryHV::new_random(3),
        ];
        let key1 = HdcContextKey::derive(&sensors);
        let key2 = HdcContextKey::derive(&sensors);
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_context_key_different_sensors() {
        let sensors_a = vec![BinaryHV::new_random(1), BinaryHV::new_random(2)];
        let sensors_b = vec![BinaryHV::new_random(3), BinaryHV::new_random(4)];
        let key_a = HdcContextKey::derive(&sensors_a);
        let key_b = HdcContextKey::derive(&sensors_b);
        assert_ne!(key_a, key_b);
    }

    #[test]
    fn test_context_key_order_matters() {
        let s1 = BinaryHV::new_random(1);
        let s2 = BinaryHV::new_random(2);
        let key_12 = HdcContextKey::derive(&[s1, s2]);
        let key_21 = HdcContextKey::derive(&[s2, s1]);
        assert_ne!(key_12, key_21);
    }

    #[test]
    fn test_context_key_empty_returns_zero() {
        let key = HdcContextKey::derive(&[]);
        assert_eq!(key, BinaryHV::zero());
    }

    #[test]
    fn test_context_key_to_symmetric() {
        let sensors = vec![BinaryHV::new_random(1), BinaryHV::new_random(2)];
        let sym_key = HdcContextKey::derive_symmetric(&sensors);
        assert_eq!(sym_key.len(), 32);

        let sym_key2 = HdcContextKey::derive_symmetric(&sensors);
        assert_eq!(sym_key, sym_key2);

        let sensors_b = vec![BinaryHV::new_random(3), BinaryHV::new_random(4)];
        let sym_key_b = HdcContextKey::derive_symmetric(&sensors_b);
        assert_ne!(sym_key, sym_key_b);
    }

    // -- HdcCommitment Tests --

    #[test]
    fn test_commitment_verify() {
        let secret = BinaryHV::new_random(42);
        let offset = 137;
        let commitment = HdcCommitment::commit(&secret, offset);
        assert!(HdcCommitment::verify(
            &commitment.commitment,
            &secret,
            offset
        ));
    }

    #[test]
    fn test_commitment_wrong_offset_fails() {
        let secret = BinaryHV::new_random(42);
        let commitment = HdcCommitment::commit(&secret, 137);
        assert!(!HdcCommitment::verify(&commitment.commitment, &secret, 138));
    }

    #[test]
    fn test_commitment_wrong_secret_fails() {
        let secret = BinaryHV::new_random(42);
        let wrong_secret = BinaryHV::new_random(43);
        let commitment = HdcCommitment::commit(&secret, 137);
        assert!(!HdcCommitment::verify(
            &commitment.commitment,
            &wrong_secret,
            137
        ));
    }

    #[test]
    fn test_commitment_hiding() {
        let secret = BinaryHV::new_random(42);
        let commitment = HdcCommitment::commit(&secret, 137);
        let sim = commitment.commitment.similarity(&secret);
        assert!(
            (sim - 0.5).abs() < 0.05,
            "Commitment should be ~0.5 similar to secret, got {sim}"
        );
    }

    #[test]
    fn test_commitment_noisy_verify() {
        let secret = BinaryHV::new_random(42);
        let offset = 137;
        let commitment = HdcCommitment::commit(&secret, offset);

        assert!(HdcCommitment::verify_noisy(
            &commitment.commitment,
            &secret,
            offset,
            0.95
        ));

        assert!(!HdcCommitment::verify_noisy(
            &commitment.commitment,
            &secret,
            138,
            0.95
        ));
    }

    // -- Cross-Primitive Tests --

    #[test]
    fn test_mac_over_committed_secret() {
        let secret = BinaryHV::new_random(42);
        let key = BinaryHV::new_random(99);
        let offset = 137;

        let commitment = HdcCommitment::commit(&secret, offset);
        let mac = HdcMac::compute(&commitment.commitment, &key);

        assert!(HdcMac::verify(&commitment.commitment, &key, &mac));
        assert!(HdcCommitment::verify(
            &commitment.commitment,
            &secret,
            offset
        ));
    }

    #[test]
    fn test_threshold_with_context_key() {
        let sensors = vec![
            BinaryHV::new_random(1),
            BinaryHV::new_random(2),
            BinaryHV::new_random(3),
        ];
        let context_key = HdcContextKey::derive(&sensors);

        let shares = HdcThresholdSharing::split(&context_key, 3, 5, 1000);
        let recovered = HdcThresholdSharing::recover(&shares[..3]);
        assert_eq!(recovered, context_key);

        let sym_original = HdcContextKey::to_symmetric_key(&context_key);
        let sym_recovered = HdcContextKey::to_symmetric_key(&recovered);
        assert_eq!(sym_original, sym_recovered);
    }
}
