// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![allow(deprecated)]
//! # Quarantined HDC Security-Claim Demonstrations
//!
//! **QUARANTINED INSECURE RESEARCH CODE.** These security-named types are
//! retained for compatibility and attack demonstrations only. They do not
//! provide a secure MAC, threshold scheme, KDF, or commitment.
//!
//! Historical operations over 16,384-bit binary hypervectors ([`BinaryHV`]).
//! Their names are retained for compatibility, not as security claims:
//!
//! - [`HdcMac`] — a forgeable linear message transform
//! - [`HdcThresholdSharing`] — records that each reveal the input
//! - [`HdcContextKey`] — a deterministic sensor-context fingerprint
//! - [`HdcCommitment`] — a reversible cyclic rotation
//!
//! ## Algebraic facts (not security properties)
//!
//! - **Random collision probability**: P(HV_a = HV_b) = 2^{-16384} (negligible)
//! - **Similarity concentration**: For random HVs, sim(a,b) ≈ 0.5 ± 0.0039 (1σ; σ =
//!   1/(2√D) at D=16,384. 3σ is ≈0.0117, not 0.0039 — an earlier version of this
//!   comment mislabeled 1σ as 3σ.)
//! - **Binding invertibility**: a ⊗ a = 0 (XOR self-inverse)
//! - **Permutation bijectivity**: π_k is a bijection on {0,1}^D for all k
//!
//! These facts do not establish authentication, threshold privacy, key entropy,
//! hiding, or binding. Do not use this module at a security boundary.
//!
//! ## References
//!
//! - Kanerva, P. (2009). Hyperdimensional computing. *Cognitive Computation*.
//! - Rahimi et al. (2016). Robust and energy-efficient classifier using brain-inspired HDC.
//! - Imani et al. (2019). A framework for collaborative learning in secure HDC.
//! - Shannon, C. (1949). Communication theory of secrecy systems. *Bell System Technical Journal*.

use super::binary_hv::BinaryHV;

// ═══════════════════════════════════════════════════════════════════════════════
// HDC MESSAGE AUTHENTICATION CODE
// ═══════════════════════════════════════════════════════════════════════════════

/// Historical `HdcMac` compatibility transform. This is not a MAC.
///
/// # Construction
///
/// ```text
/// MAC(message, key) = message ⊗ π_k(key)
/// ```
///
/// where ⊗ is XOR binding and π_k is cyclic permutation by offset k.
///
/// A known message/tag pair directly reveals `π_k(key)`, after which an
/// attacker can tag any chosen message. Key entropy does not repair this.
#[deprecated(note = "forgeable compatibility transform; use a standard audited MAC")]
pub struct HdcMac;

/// Default permutation offset for MAC key derivation.
/// Using a small prime avoids alignment artifacts in the permutation.
const HDC_MAC_PERMUTE_OFFSET: usize = 7;

/// Minimum similarity threshold for noisy HDC-MAC verification.
///
/// At D = 16,384, random similarity ≈ 0.5 ± 0.0039 (1σ; σ = 1/(2√D)).
/// By Hoeffding's inequality, P(similarity ≥ τ | random) ≤ exp(-2·D·(τ-0.5)²).
/// At τ = 0.95: exponent = 2·16384·0.45² ≈ 6635.5 nats ≈ 9573 bits, so the
/// false-positive rate is ≈ 2^-9573, not the previously stated 2^-4700 (that
/// figure did not correctly convert the exp(·) bound to a base-2 exponent).
///
/// This bound describes only the *noise-tolerance* threshold; it says nothing
/// about [`HdcMac`]'s forgeability, which is unconditional (see [`HdcMac`] docs).
pub const HDC_MAC_NOISY_THRESHOLD: f32 = 0.95;

impl HdcMac {
    /// Compute MAC over a BinaryHV message with a BinaryHV key.
    ///
    /// MAC = message ⊗ π_7(key)
    #[inline]
    pub fn compute(message: &BinaryHV, key: &BinaryHV) -> BinaryHV {
        let derived = key.permute(HDC_MAC_PERMUTE_OFFSET);
        message.bind(&derived)
    }

    /// Compute the transform with a custom permutation offset.
    ///
    /// This does **not** provide domain separation. Given one known
    /// (message, tag) pair at any offset, an attacker recovers
    /// `π_offset(key)` and can derive `π_offset2(key)` for any other offset
    /// by a further rotation (`permute` is a cyclic shift, so
    /// `π_{o2-o1}(π_{o1}(key)) == π_{o2}(key)`). One known pair in one
    /// "domain" therefore forges tags in every other "domain". See
    /// `legacy_attack_cross_offset_forgery`.
    #[inline]
    pub fn compute_with_offset(message: &BinaryHV, key: &BinaryHV, offset: usize) -> BinaryHV {
        let derived = key.permute(offset);
        message.bind(&derived)
    }

    /// Verify MAC (exact match — for lossless channels).
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

// ═══════════════════════════════════════════════════════════════════════════════
// HDC THRESHOLD SECRET SHARING
// ═══════════════════════════════════════════════════════════════════════════════

/// Historical share-container compatibility transform. This is not threshold
/// secret sharing.
///
/// # Construction (Kanerva 2009)
///
/// **Split**: Generate n random masks. Each share is `secret ⊗ mask_i`.
/// **Recover**: Unbind each share with its mask → approximation of secret.
/// Bundle k approximations via majority vote → recovered secret.
///
/// Every returned [`HdcShare`] includes both `secret XOR mask` and `mask`, so
/// one record recovers the secret. The `k` argument does not enforce access,
/// and OS-random masks do not repair the construction.
///
/// # Constraints
///
/// - k must be odd (majority vote requires odd count for deterministic tiebreaking)
/// - k ≤ n (cannot require more shares than exist)
/// - k ≥ 1 (at least one share needed)
#[deprecated(
    note = "not threshold sharing: every HdcShare includes enough data to recover the secret"
)]
pub struct HdcThresholdSharing;

/// A compatibility record that reveals the input by XORing its fields.
#[derive(Clone)]
pub struct HdcShare {
    /// Index of this share (0..n)
    pub index: usize,
    /// The share: secret ⊗ mask
    pub share: BinaryHV,
    /// The mask used to create this share (needed for recovery)
    pub mask: BinaryHV,
}

impl HdcThresholdSharing {
    /// Produce `n` insecure compatibility records using deterministic masks.
    ///
    /// **Not secure.** Masks are derived from `seed` via [`BinaryHV::random`],
    /// so anyone who knows or brute-forces `seed` can reconstruct every mask
    /// and recover the secret from a single share. Even without the seed, each
    /// record contains its own mask. This exists for reproducible tests only.
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
                // Deterministic mask from seed + index (reproducible shares)
                let mask = BinaryHV::random(seed.wrapping_add(i as u64));
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
    /// This remains insecure because every record includes its mask. The name
    /// is retained for compatibility only.
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
                let mask = BinaryHV::secure_random();
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
    /// With k correct shares (where k is the threshold), recovery is exact.
    pub fn recover(shares: &[HdcShare]) -> BinaryHV {
        assert!(!shares.is_empty(), "need at least one share to recover");

        let unbound: Vec<BinaryHV> = shares
            .iter()
            .map(|s| s.share.bind(&s.mask)) // XOR is self-inverse: (secret ⊗ mask) ⊗ mask = secret
            .collect();

        BinaryHV::bundle(&unbound)
    }

    /// Check if recovery with the given shares would be exact.
    ///
    /// Returns the Hamming similarity between the recovered secret and each
    /// individual unbound share. If all are 1.0, recovery is exact.
    pub fn recovery_quality(shares: &[HdcShare]) -> f32 {
        if shares.is_empty() {
            return 0.0;
        }
        let recovered = Self::recover(shares);
        // Check against first share's unbound value
        let reference = shares[0].share.bind(&shares[0].mask);
        recovered.similarity(&reference)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HDC CONTEXT KEY
// ═══════════════════════════════════════════════════════════════════════════════

/// Deterministic context fingerprint derived from sensor state. This is not a KDF.
///
/// # Construction
///
/// ```text
/// key = sensor_0 ⊗ π_1(sensor_1) ⊗ π_2(sensor_2) ⊗ ... ⊗ π_{n-1}(sensor_{n-1})
/// ```
///
/// Each sensor reading is permuted by its index to prevent commutativity
/// (sensor order matters: GPS ⊗ π_1(accel) ≠ accel ⊗ π_1(GPS)).
///
/// # Security warning
///
/// This deterministic transform cannot create entropy. Hashing the result does
/// not make enumerable sensor states secret. It may be used as public context
/// input to a standard KDF only when combined with an independent secret.
///
/// Use the result only as public context to a standard KDF that also receives
/// independently established secret input.
///
/// This is a second, independent failure beyond low entropy: the pre-hash
/// derivation is also linearly malleable and highly non-injective. Because
/// XOR/permute is linear, replacing `sensors[0]` with `sensors[0] ⊗ delta`
/// and `sensors[1]` with `sensors[1] ⊗ π^{-1}_1(delta)` cancels exactly, so
/// two different sensor tuples derive an identical fingerprint (see
/// `legacy_attack_context_key_collision`). Hashing the output cannot recover
/// a distinction the pre-hash derivation already destroyed.
#[deprecated(
    note = "does not add entropy; combine context with an independent secret using a standard KDF"
)]
pub struct HdcContextKey;

impl HdcContextKey {
    /// Derive a context key from multiple sensor readings encoded as BinaryHVs.
    ///
    /// Each sensor is permuted by its position index to enforce ordering.
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

    /// Hash a context fingerprint to 32 bytes; hashing does not add secrecy.
    ///
    /// Uses BLAKE3 hash extraction to produce a uniform 32-byte key
    /// Do not use the result directly as a cipher key for enumerable inputs.
    pub fn to_symmetric_key(context: &BinaryHV) -> [u8; 32] {
        *blake3::hash(&context.0).as_bytes()
    }

    /// Hash a derived context fingerprint to 32 bytes. This is not a KDF.
    ///
    /// Convenience method combining `derive()` + `to_symmetric_key()`.
    pub fn derive_symmetric(sensors: &[BinaryHV]) -> [u8; 32] {
        let context = Self::derive(sensors);
        Self::to_symmetric_key(&context)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HDC COMMITMENT SCHEME
// ═══════════════════════════════════════════════════════════════════════════════

/// Historical rotation/opening compatibility transform. This is not a
/// commitment scheme.
///
/// # Construction
///
/// ```text
/// Commit(secret, offset) = π_offset(secret)
/// Verify(commitment, secret, offset) = (π_offset(secret) == commitment)
/// ```
///
/// # Security warning
///
/// Cyclic rotation is reversible and has only `D` possible offsets. For every
/// alternative offset there is a corresponding message that opens the same
/// value. This is neither a hiding nor a binding commitment scheme.
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

    /// Compare a rotated input with tolerance; this has no commitment security.
    pub fn verify_noisy(
        commitment: &BinaryHV,
        secret: &BinaryHV,
        offset: usize,
        threshold: f32,
    ) -> bool {
        secret.permute(offset).similarity(commitment) >= threshold
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── HdcMac Tests ─────────────────────────────────────────────────────

    #[test]
    fn test_hdc_mac_roundtrip() {
        let message = BinaryHV::random(42);
        let key = BinaryHV::random(99);
        let mac = HdcMac::compute(&message, &key);
        assert!(HdcMac::verify(&message, &key, &mac));
    }

    #[test]
    fn test_hdc_mac_wrong_key_fails() {
        let message = BinaryHV::random(42);
        let key_a = BinaryHV::random(99);
        let key_b = BinaryHV::random(100);
        let mac = HdcMac::compute(&message, &key_a);
        assert!(!HdcMac::verify(&message, &key_b, &mac));
        // Wrong key should produce near-random similarity
        let expected_wrong = HdcMac::compute(&message, &key_b);
        let sim = mac.similarity(&expected_wrong);
        assert!(
            (sim - 0.5).abs() < 0.05,
            "Wrong-key MAC should be ~0.5 similar, got {sim}"
        );
    }

    #[test]
    fn test_hdc_mac_tampered_message_fails() {
        let message = BinaryHV::random(42);
        let key = BinaryHV::random(99);
        let mac = HdcMac::compute(&message, &key);

        let tampered = BinaryHV::random(43); // different message
        assert!(!HdcMac::verify(&tampered, &key, &mac));
    }

    #[test]
    fn test_hdc_mac_noisy_channel_tolerance() {
        let message = BinaryHV::random(42);
        let key = BinaryHV::random(99);
        let mac = HdcMac::compute(&message, &key);

        // Exact MAC should pass noisy verification
        assert!(HdcMac::verify_noisy(&message, &key, &mac, 0.95));
        assert!(HdcMac::verify_noisy(&message, &key, &mac, 1.0));

        // Wrong key should fail noisy verification
        let wrong_key = BinaryHV::random(100);
        assert!(!HdcMac::verify_noisy(&message, &wrong_key, &mac, 0.95));
    }

    /// `compute_with_offset` is not domain separation: one known (message,
    /// tag) pair at offset 7 lets an attacker forge a tag at a completely
    /// different offset (11) for a chosen message, without the key. Replaces
    /// the prior `test_hdc_mac_domain_separation`, which only checked that
    /// two offsets produce different tags — true, but irrelevant to whether
    /// an attacker who saw one tag can forge another.
    #[test]
    fn legacy_attack_cross_offset_forgery() {
        let known_offset = 7;
        let target_offset = 11;
        let known_message = BinaryHV::random(42);
        let target_message = BinaryHV::random(43);
        let key = BinaryHV::random(99);

        let known_tag = HdcMac::compute_with_offset(&known_message, &key, known_offset);

        // Recover π_known_offset(key), then rotate it the extra distance to
        // π_target_offset(key), all without the key.
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
        let message = BinaryHV::random(42);
        let key = BinaryHV::random(99);
        let mac1 = HdcMac::compute(&message, &key);
        let mac2 = HdcMac::compute(&message, &key);
        assert_eq!(mac1, mac2);
    }

    #[test]
    fn legacy_attack_known_pair_forges_hdc_mac() {
        let known_message = BinaryHV::random(42);
        let target_message = BinaryHV::random(43);
        let key = BinaryHV::random(99);
        let known_tag = HdcMac::compute(&known_message, &key);
        let recovered_pad = known_message.bind(&known_tag);
        let forged_tag = target_message.bind(&recovered_pad);

        assert!(HdcMac::verify(&target_message, &key, &forged_tag));
    }

    // ── HdcThresholdSharing Tests ────────────────────────────────────────

    #[test]
    fn test_threshold_split_recover_exact_k_equals_n() {
        let secret = BinaryHV::random(42);
        let shares = HdcThresholdSharing::split(&secret, 3, 3, 1000);
        assert_eq!(shares.len(), 3);

        let recovered = HdcThresholdSharing::recover(&shares);
        assert_eq!(
            recovered, secret,
            "k=n recovery should be exact (all shares used)"
        );
    }

    #[test]
    fn test_threshold_split_recover_k_less_than_n() {
        let secret = BinaryHV::random(42);
        let shares = HdcThresholdSharing::split(&secret, 3, 5, 1000);
        assert_eq!(shares.len(), 5);

        // Use first 3 shares (k=3)
        let recovered = HdcThresholdSharing::recover(&shares[..3]);
        assert_eq!(
            recovered, secret,
            "k < n recovery should be exact when all k shares are valid"
        );
    }

    #[test]
    fn test_threshold_split_secure_recover() {
        let secret = BinaryHV::random(42);
        let shares = HdcThresholdSharing::split_secure(&secret, 3, 5);
        assert_eq!(shares.len(), 5);

        let recovered = HdcThresholdSharing::recover(&shares[..3]);
        assert_eq!(
            recovered, secret,
            "secure split/recover should round-trip exactly"
        );
    }

    #[test]
    fn test_threshold_split_secure_masks_are_unpredictable() {
        let secret = BinaryHV::random(42);
        let shares_a = HdcThresholdSharing::split_secure(&secret, 1, 1);
        let shares_b = HdcThresholdSharing::split_secure(&secret, 1, 1);
        assert_ne!(
            shares_a[0].mask, shares_b[0].mask,
            "secure masks must not repeat across calls"
        );
    }

    #[test]
    fn test_binary_hv_secure_random_is_not_deterministic() {
        let a = BinaryHV::secure_random();
        let b = BinaryHV::secure_random();
        assert_ne!(a, b, "two secure-random draws should not collide");
    }

    #[test]
    fn legacy_attack_single_share_recovers_despite_k_three() {
        let secret = BinaryHV::random(42);
        let shares = HdcThresholdSharing::split(&secret, 3, 5, 1000);

        // Every serialized share carries its own mask, so k is irrelevant.
        for share in &shares {
            let recovered = HdcThresholdSharing::recover(&[share.clone()]);
            assert_eq!(
                recovered, secret,
                "one share recovers even though the API claimed k=3"
            );
        }
    }

    #[test]
    fn test_threshold_recovery_quality() {
        let secret = BinaryHV::random(42);
        let shares = HdcThresholdSharing::split(&secret, 3, 5, 1000);

        let quality = HdcThresholdSharing::recovery_quality(&shares[..3]);
        assert!(
            (quality - 1.0).abs() < f32::EPSILON,
            "Recovery quality with k valid shares should be 1.0, got {quality}"
        );
    }

    #[test]
    fn test_threshold_shares_are_distinct() {
        let secret = BinaryHV::random(42);
        let shares = HdcThresholdSharing::split(&secret, 3, 5, 1000);

        // All shares should be distinct (different random masks)
        for i in 0..shares.len() {
            for j in (i + 1)..shares.len() {
                assert_ne!(
                    shares[i].share, shares[j].share,
                    "Shares {i} and {j} should be distinct"
                );
            }
        }
    }

    #[test]
    fn legacy_share_value_looks_random_but_included_mask_reveals_secret() {
        let secret = BinaryHV::random(42);
        let shares = HdcThresholdSharing::split(&secret, 3, 5, 1000);

        // Each share should be ~0.5 similar to the secret (random, information-theoretic)
        for share in &shares {
            let sim = share.share.similarity(&secret);
            assert!(
                (sim - 0.5).abs() < 0.05,
                "Individual share should be ~0.5 similar to secret (OTP), got {sim}"
            );
            assert_eq!(share.share.bind(&share.mask), secret);
        }
    }

    #[test]
    #[should_panic(expected = "k must be odd")]
    fn test_threshold_even_k_panics() {
        let secret = BinaryHV::random(42);
        HdcThresholdSharing::split(&secret, 2, 5, 1000);
    }

    #[test]
    #[should_panic(expected = "k must be <= n")]
    fn test_threshold_k_greater_than_n_panics() {
        let secret = BinaryHV::random(42);
        HdcThresholdSharing::split(&secret, 5, 3, 1000);
    }

    // ── HdcContextKey Tests ──────────────────────────────────────────────

    #[test]
    fn test_context_key_deterministic() {
        let sensors = vec![
            BinaryHV::random(1),
            BinaryHV::random(2),
            BinaryHV::random(3),
        ];
        let key1 = HdcContextKey::derive(&sensors);
        let key2 = HdcContextKey::derive(&sensors);
        assert_eq!(key1, key2, "Same sensors should produce same key");
    }

    #[test]
    fn test_context_key_different_sensors_different_key() {
        let sensors_a = vec![BinaryHV::random(1), BinaryHV::random(2)];
        let sensors_b = vec![BinaryHV::random(3), BinaryHV::random(4)];
        let key_a = HdcContextKey::derive(&sensors_a);
        let key_b = HdcContextKey::derive(&sensors_b);
        assert_ne!(key_a, key_b);
    }

    #[test]
    fn test_context_key_order_matters() {
        let s1 = BinaryHV::random(1);
        let s2 = BinaryHV::random(2);
        let key_12 = HdcContextKey::derive(&[s1, s2]);
        let key_21 = HdcContextKey::derive(&[s2, s1]);
        assert_ne!(
            key_12, key_21,
            "Sensor order should matter (permutation breaks commutativity)"
        );
    }

    #[test]
    fn test_context_key_empty_returns_zero() {
        let key = HdcContextKey::derive(&[]);
        assert_eq!(key, BinaryHV::zero());
    }

    /// CI-005 (extended): the derivation is linearly malleable, so two
    /// distinct 2-sensor tuples derive the identical context fingerprint.
    /// `result = s0 ⊗ π_1(s1)`. Replacing `s0` with `s0 ⊗ delta` and `s1`
    /// with `s1 ⊗ π_{-1}(delta)` cancels exactly: `π_1(s1 ⊗ π_{-1}(delta)) =
    /// π_1(s1) ⊗ delta`, so the `delta` terms cancel and the derived key is
    /// unchanged.
    #[test]
    fn legacy_attack_context_key_collision() {
        let s0 = BinaryHV::random(1);
        let s1 = BinaryHV::random(2);
        let delta = BinaryHV::random(999);

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
        let sensors = vec![BinaryHV::random(1), BinaryHV::random(2)];
        let sym_key = HdcContextKey::derive_symmetric(&sensors);
        assert_eq!(sym_key.len(), 32);

        // Same sensors → same symmetric key
        let sym_key2 = HdcContextKey::derive_symmetric(&sensors);
        assert_eq!(sym_key, sym_key2);

        // Different sensors → different symmetric key
        let sensors_b = vec![BinaryHV::random(3), BinaryHV::random(4)];
        let sym_key_b = HdcContextKey::derive_symmetric(&sensors_b);
        assert_ne!(sym_key, sym_key_b);
    }

    // ── HdcCommitment Tests ──────────────────────────────────────────────

    #[test]
    fn test_commitment_verify() {
        let secret = BinaryHV::random(42);
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
        let secret = BinaryHV::random(42);
        let commitment = HdcCommitment::commit(&secret, 137);
        assert!(!HdcCommitment::verify(&commitment.commitment, &secret, 138));
    }

    #[test]
    fn test_commitment_wrong_secret_fails() {
        let secret = BinaryHV::random(42);
        let wrong_secret = BinaryHV::random(43);
        let commitment = HdcCommitment::commit(&secret, 137);
        assert!(!HdcCommitment::verify(
            &commitment.commitment,
            &wrong_secret,
            137
        ));
    }

    #[test]
    fn test_rotated_value_has_low_similarity_but_is_not_hiding() {
        let secret = BinaryHV::random(42);
        let commitment = HdcCommitment::commit(&secret, 137);
        let sim = commitment.commitment.similarity(&secret);
        assert!(
            (sim - 0.5).abs() < 0.05,
            "Commitment should be ~0.5 similar to secret (hiding), got {sim}"
        );
    }

    #[test]
    fn legacy_attack_commitment_has_multiple_valid_openings() {
        let secret = BinaryHV::random(42);
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
        let secret = BinaryHV::random(42);
        let offset = 137;
        let commitment = HdcCommitment::commit(&secret, offset);

        // Exact commitment should pass noisy verification
        assert!(HdcCommitment::verify_noisy(
            &commitment.commitment,
            &secret,
            offset,
            0.95
        ));

        // Wrong offset should fail
        assert!(!HdcCommitment::verify_noisy(
            &commitment.commitment,
            &secret,
            138,
            0.95
        ));
    }

    // ── Cross-Primitive Tests ────────────────────────────────────────────

    #[test]
    fn test_mac_over_committed_secret() {
        // Scenario: commit to a secret, then MAC the commitment
        let secret = BinaryHV::random(42);
        let key = BinaryHV::random(99);
        let offset = 137;

        let commitment = HdcCommitment::commit(&secret, offset);
        let mac = HdcMac::compute(&commitment.commitment, &key);

        // Verify both the MAC and the commitment
        assert!(HdcMac::verify(&commitment.commitment, &key, &mac));
        assert!(HdcCommitment::verify(
            &commitment.commitment,
            &secret,
            offset
        ));
    }

    #[test]
    fn test_threshold_with_context_key() {
        // Compatibility scenario only: deterministic context plus broken records.
        let sensors = vec![
            BinaryHV::random(1),
            BinaryHV::random(2),
            BinaryHV::random(3),
        ];
        let context_key = HdcContextKey::derive(&sensors);

        // Produce five records (each one actually suffices to recover).
        let shares = HdcThresholdSharing::split(&context_key, 3, 5, 1000);

        // Exercise the historical three-record call path.
        let recovered = HdcThresholdSharing::recover(&shares[..3]);
        assert_eq!(recovered, context_key);

        // Extract symmetric key from recovered HDC key
        let sym_original = HdcContextKey::to_symmetric_key(&context_key);
        let sym_recovered = HdcContextKey::to_symmetric_key(&recovered);
        assert_eq!(sym_original, sym_recovered);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// BENCHMARKS (run with `cargo bench -p symthaea-core`)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    /// Quick benchmark of the forgeable compatibility transform's latency.
    #[test]
    fn bench_hdc_mac_compute() {
        let message = BinaryHV::random(42);
        let key = BinaryHV::random(99);

        // Warmup
        for _ in 0..1000 {
            let _ = HdcMac::compute(&message, &key);
        }

        let iters = 100_000;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(HdcMac::compute(&message, &key));
        }
        let elapsed = start.elapsed();
        let ns_per_op = elapsed.as_nanos() as f64 / iters as f64;

        eprintln!(
            "forgeable HDC tag transform: {:.1} ns/op ({} iterations, {:.1} ms total)",
            ns_per_op,
            iters,
            elapsed.as_secs_f64() * 1000.0
        );
        // Sanity: should be under 100 microseconds even in debug mode.
        // Release mode target: ~5-10 ns (SIMD), ~80 ns (scalar).
        assert!(
            ns_per_op < 100_000.0,
            "HDC-MAC should be < 100µs, got {ns_per_op:.0} ns"
        );
    }

    /// Quick benchmark: HDC-MAC verify latency.
    #[test]
    fn bench_hdc_mac_verify() {
        let message = BinaryHV::random(42);
        let key = BinaryHV::random(99);
        let mac = HdcMac::compute(&message, &key);

        let iters = 100_000;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(HdcMac::verify(&message, &key, &mac));
        }
        let elapsed = start.elapsed();
        let ns_per_op = elapsed.as_nanos() as f64 / iters as f64;

        eprintln!(
            "HDC-MAC verify: {:.1} ns/op ({} iterations, {:.1} ms total)",
            ns_per_op,
            iters,
            elapsed.as_secs_f64() * 1000.0
        );
        // Debug mode: ~2-20µs. Release mode target: ~10-20 ns.
        assert!(
            ns_per_op < 100_000.0,
            "HDC-MAC verify should be < 100µs, got {ns_per_op:.0} ns"
        );
    }

    /// Quick benchmark: threshold sharing (3-of-5 split + recover).
    #[test]
    fn bench_threshold_3_of_5() {
        let secret = BinaryHV::random(42);

        let iters = 10_000;
        let start = Instant::now();
        for i in 0..iters {
            let shares = HdcThresholdSharing::split(&secret, 3, 5, i as u64);
            let _ = std::hint::black_box(HdcThresholdSharing::recover(&shares[..3]));
        }
        let elapsed = start.elapsed();
        let us_per_op = elapsed.as_micros() as f64 / iters as f64;

        eprintln!(
            "Threshold 3-of-5 (split+recover): {:.1} µs/op ({} iterations)",
            us_per_op, iters
        );
        // Debug mode: ~50-200µs. Release mode target: ~5-10µs.
        assert!(
            us_per_op < 10_000.0,
            "Threshold 3-of-5 should be < 10ms, got {us_per_op:.0} µs"
        );
    }

    /// Quick benchmark: context key derivation (3 sensors → symmetric key).
    #[test]
    fn bench_context_key_derive() {
        let sensors = vec![
            BinaryHV::random(1),
            BinaryHV::random(2),
            BinaryHV::random(3),
        ];

        let iters = 100_000;
        let start = Instant::now();
        for _ in 0..iters {
            let _ = std::hint::black_box(HdcContextKey::derive_symmetric(&sensors));
        }
        let elapsed = start.elapsed();
        let ns_per_op = elapsed.as_nanos() as f64 / iters as f64;

        eprintln!(
            "Context key derive (3 sensors): {:.1} ns/op ({} iterations)",
            ns_per_op, iters
        );
        // Debug mode: ~10-15µs. Release mode target: ~200-500 ns.
        assert!(
            ns_per_op < 100_000.0,
            "Context key should be < 100µs, got {ns_per_op:.0} ns"
        );
    }
}
