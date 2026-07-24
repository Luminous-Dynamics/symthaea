// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! # Quarantined HDC Security-Claim Demonstrations
//!
//! **QUARANTINED:** the security-named types in this module are retained only
//! for compatibility and executable attack demonstrations. They do not satisfy
//! the cryptographic notions their historical names imply.
//!
//! Historical operations that exploit the algebraic properties of 16,384-bit
//! binary hypervectors. The names below are compatibility names, not validated
//! security claims:
//!
//! - [`HdcMac`] -- a forgeable linear message transform
//! - [`HdcThresholdSharing`] -- copies recoverable secret material into every share
//! - [`HdcContextKey`] -- a deterministic sensor-context fingerprint
//! - [`HdcCommitment`] -- a reversible cyclic rotation
//!
//! ## Algebraic facts (not security properties)
//!
//! HDC crypto primitives provide _information-theoretic_ properties in
//! high-dimensional spaces (D = 16,384) where:
//!
//! - **Random collision probability**: P(HV_a = HV_b) = 2^{-16384} (negligible)
//! - **Similarity concentration**: For random HVs, sim(a,b) ~ 0.5 +/- 0.0039 (1 sigma;
//!   sigma = 1/(2*sqrt(D)) at D=16,384. 3 sigma is ~0.0117, not 0.0039 -- an earlier
//!   version of this comment mislabeled 1 sigma as 3 sigma.)
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

/// Historical `HdcMac` compatibility transform. This is not a MAC.
///
/// # Construction
///
/// ```text
/// MAC(message, key) = message XOR permute_k(key)
/// ```
///
/// One known message/tag pair reveals `permute_k(key)`, allowing tags for every
/// chosen message. Its speed is therefore irrelevant to authentication.
#[deprecated(note = "forgeable compatibility transform; use a standard audited MAC")]
pub struct HdcMac;

/// Default permutation offset for MAC key derivation.
const HDC_MAC_PERMUTE_OFFSET: usize = 7;

/// Minimum similarity threshold for noisy HDC-MAC verification.
///
/// At D = 16,384, random similarity ~ 0.5 +/- 0.0039 (1 sigma; sigma = 1/(2*sqrt(D))).
/// By Hoeffding's inequality, P(similarity >= tau | random) <= exp(-2*D*(tau-0.5)^2).
/// At tau = 0.95: exponent = 2*16384*0.45^2 ~= 6635.5 nats ~= 9573 bits, so the
/// false-positive rate is ~ 2^-9573, not the previously stated 2^-4700 (that number
/// did not correctly convert the exp(.) bound to a base-2 exponent).
///
/// This bound describes only the *noise-tolerance* threshold; it says nothing about
/// [`HdcMac`]'s forgeability, which is unconditional (see [`HdcMac`] docs).
pub const HDC_MAC_NOISY_THRESHOLD: f32 = 0.95;

impl HdcMac {
    /// Compute MAC over a BinaryHV message with a BinaryHV key.
    #[inline]
    pub fn compute(message: &BinaryHV, key: &BinaryHV) -> BinaryHV {
        let derived = key.permute(HDC_MAC_PERMUTE_OFFSET);
        message.bind(&derived)
    }

    /// Compute the transform with a custom permutation offset.
    ///
    /// This does **not** provide domain separation. Given one known
    /// `(message, tag)` pair at any offset, an attacker recovers
    /// `permute(key, offset)` and can derive `permute(key, offset2)` for any
    /// other offset by a further rotation (`permute` is a cyclic shift, so
    /// `permute(permute(key, o1), o2 - o1) == permute(key, o2)`). One known
    /// pair in one "domain" therefore forges tags in every other "domain".
    /// See `legacy_attack_cross_offset_forgery`.
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

/// Historical share-container compatibility transform. This is not threshold
/// secret sharing.
///
/// # Construction (Kanerva 2009)
///
/// **Split**: Generate n random masks. Each share is `secret XOR mask_i`.
/// **Recover**: Unbind each share with its mask, then bundle via majority vote.
///
/// Every returned [`HdcShare`] contains both `secret XOR mask` and `mask`, so
/// any one share recovers `secret`. The `k` argument does not enforce access.
/// OS randomness in [`HdcThresholdSharing::split_secure`] does not repair this
/// construction-level failure.
///
/// # Constraints
///
/// - k must be odd (majority vote requires odd count)
/// - k <= n
/// - k >= 1
#[deprecated(
    note = "not threshold sharing: every HdcShare includes enough data to recover the secret"
)]
pub struct HdcThresholdSharing;

/// A compatibility record that reveals the original input by XORing its fields.
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
    /// Produce `n` insecure compatibility records using deterministic masks.
    ///
    /// **Not secure.** Masks are derived from `seed` via [`BinaryHV::new_random`],
    /// so anyone who knows or brute-forces `seed` can reconstruct every mask
    /// and recover the secret from a single share. Even without the seed, each
    /// record already contains its mask. This exists for reproducible tests only.
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

    /// Produce `n` insecure compatibility records using OS-entropy-backed masks.
    ///
    /// This remains insecure: every record includes its mask and independently
    /// reveals the secret. The method name is retained for API compatibility.
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

    /// Recover the input from one or more records; `k` is not checked or used.
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

/// Deterministic context fingerprint derived from sensor state. This is not a KDF.
///
/// # Construction
///
/// ```text
/// key = sensor_0 XOR permute_1(sensor_1) XOR permute_2(sensor_2) XOR ...
/// ```
///
/// Each sensor is permuted by its index to enforce ordering (non-commutative).
///
/// Hashing the output does not add entropy. Use it only as public context to a
/// standard KDF that also receives independent secret key material.
///
/// This is a second, independent failure beyond low entropy: the pre-hash
/// derivation is also linearly malleable and highly non-injective. Because
/// XOR/permute is linear, replacing `sensors[0]` with `sensors[0] XOR delta`
/// and `sensors[1]` with `sensors[1] XOR permute(delta, -1)` cancels exactly,
/// so two different sensor tuples derive an identical fingerprint (see
/// `legacy_attack_context_key_collision`). Hashing the output cannot recover
/// a distinction the pre-hash derivation already destroyed.
#[deprecated(
    note = "does not add entropy; combine context with an independent secret using a standard KDF"
)]
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

    /// Hash a context fingerprint to 32 bytes. The result is not secret unless
    /// the input already contains independently established secret entropy.
    ///
    /// Uses BLAKE3 hash extraction to produce a uniform 32-byte key
    /// This output must not be used directly as a cipher key for enumerable
    /// sensor inputs.
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

/// Historical rotation/opening compatibility transform. This is not a
/// commitment scheme.
///
/// ```text
/// Commit(secret, offset) = permute_offset(secret)
/// Verify(commitment, secret, offset) = (permute_offset(secret) == commitment)
/// ```
///
/// Rotation exposes structure and permits alternative openings: for another
/// offset, rotate the message correspondingly to obtain the same public value.
#[derive(Clone)]
#[deprecated(note = "cyclic rotation is neither a hiding nor binding commitment scheme")]
pub struct HdcCommitment {
    /// The commitment value (publicly shared).
    pub commitment: BinaryHV,
}

impl HdcCommitment {
    /// Rotate an input by an offset; this does not create a secure commitment.
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

    /// `compute_with_offset` is not domain separation: one known (message, tag)
    /// pair at offset 7 lets an attacker forge a tag at a completely different
    /// offset (11) for a chosen message, without the key. This replaces the
    /// prior `test_hdc_mac_domain_separation`, which only checked that two
    /// offsets produce different tags -- true, but irrelevant to whether an
    /// attacker who saw one tag can forge another.
    #[test]
    fn legacy_attack_cross_offset_forgery() {
        let known_offset = 7;
        let target_offset = 11;
        let known_message = BinaryHV::new_random(42);
        let target_message = BinaryHV::new_random(43);
        let key = BinaryHV::new_random(99);

        let known_tag = HdcMac::compute_with_offset(&known_message, &key, known_offset);

        // Recover permute(key, known_offset), then rotate it the extra distance
        // to permute(key, target_offset), all without the key.
        let permuted_key_at_known_offset = known_message.bind(&known_tag);
        let shift_between_offsets = target_offset - known_offset;
        let permuted_key_at_target_offset =
            permuted_key_at_known_offset.permute(shift_between_offsets);
        let forged_tag = target_message.bind(&permuted_key_at_target_offset);

        assert_eq!(
            forged_tag,
            HdcMac::compute_with_offset(&target_message, &key, target_offset),
            "cross-offset forgery should reproduce the real tag exactly"
        );
    }

    #[test]
    fn test_hdc_mac_deterministic() {
        let message = BinaryHV::new_random(42);
        let key = BinaryHV::new_random(99);
        let mac1 = HdcMac::compute(&message, &key);
        let mac2 = HdcMac::compute(&message, &key);
        assert_eq!(mac1, mac2);
    }

    /// CI-001: one known message/tag pair reveals the reusable XOR pad, so an
    /// attacker can produce a valid tag for any target message without the key.
    #[test]
    fn legacy_attack_known_pair_forges_hdc_mac() {
        let known_message = BinaryHV::new_random(42);
        let target_message = BinaryHV::new_random(43);
        let key = BinaryHV::new_random(99);
        let known_tag = HdcMac::compute(&known_message, &key);

        let recovered_pad = known_message.bind(&known_tag);
        let forged_tag = target_message.bind(&recovered_pad);

        assert!(HdcMac::verify(&target_message, &key, &forged_tag));
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
    fn legacy_attack_single_threshold_share_recovers_secret() {
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
    fn legacy_share_payload_looks_random_but_included_mask_recovers_secret() {
        let secret = BinaryHV::new_random(42);
        let shares = HdcThresholdSharing::split(&secret, 3, 5, 1000);

        for share in &shares {
            let sim = share.share.similarity(&secret);
            assert!(
                (sim - 0.5).abs() < 0.05,
                "Individual share should be ~0.5 similar to secret, got {sim}"
            );
            assert_eq!(share.share.bind(&share.mask), secret);
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

    /// CI-005 (extended): the derivation is linearly malleable, so two
    /// distinct 2-sensor tuples derive the identical context fingerprint.
    /// `result = s0 XOR permute(s1, 1)`. Replacing `s0` with `s0 XOR delta`
    /// and `s1` with `s1 XOR permute(delta, -1)` cancels exactly:
    /// `permute(s1 XOR permute(delta,-1), 1) = permute(s1,1) XOR delta`,
    /// so the `delta` terms cancel and the derived key is unchanged.
    #[test]
    fn legacy_attack_context_key_collision() {
        let s0 = BinaryHV::new_random(1);
        let s1 = BinaryHV::new_random(2);
        let delta = BinaryHV::new_random(999);

        let original = HdcContextKey::derive(&[s0, s1]);

        let inverse_permuted_delta = delta.permute(BinaryHV::DIM - 1);
        let s0_alt = s0.bind(&delta);
        let s1_alt = s1.bind(&inverse_permuted_delta);
        let collided = HdcContextKey::derive(&[s0_alt, s1_alt]);

        assert_ne!(
            (s0, s1),
            (s0_alt, s1_alt),
            "sensor tuples must actually differ"
        );
        assert_eq!(
            original, collided,
            "two different sensor tuples derived the same context fingerprint"
        );
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
    fn test_rotated_value_has_low_similarity_but_is_not_hiding() {
        let secret = BinaryHV::new_random(42);
        let commitment = HdcCommitment::commit(&secret, 137);
        let sim = commitment.commitment.similarity(&secret);
        assert!(
            (sim - 0.5).abs() < 0.05,
            "Commitment should be ~0.5 similar to secret, got {sim}"
        );
    }

    /// CI-003: for every chosen offset there is a corresponding preimage, so
    /// the same public value can be opened as many different messages.
    #[test]
    fn legacy_attack_commitment_has_multiple_valid_openings() {
        let secret = BinaryHV::new_random(42);
        let first_offset = 137;
        let second_offset = 911;
        let commitment = HdcCommitment::commit(&secret, first_offset);
        let alternate_secret = commitment
            .commitment
            .permute(BinaryHV::DIM - (second_offset % BinaryHV::DIM));

        assert_ne!(alternate_secret, secret);
        assert!(HdcCommitment::verify(
            &commitment.commitment,
            &secret,
            first_offset
        ));
        assert!(HdcCommitment::verify(
            &commitment.commitment,
            &alternate_secret,
            second_offset
        ));
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
