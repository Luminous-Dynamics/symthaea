// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property tests for HDC-native cryptographic primitives.
//!
//! These tests empirically validate the formal security claims made in
//! the doc comments of `hdc_crypto.rs`:
//!
//! - HdcMac: correct key always verifies, wrong key never does
//! - HdcThresholdSharing: k shares always recover, individual shares leak nothing
//! - HdcContextKey: different sensor orderings always produce different keys
//! - HdcCommitment: binding property (no second preimage)

use proptest::prelude::*;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::hdc_crypto::{
    HDC_MAC_NOISY_THRESHOLD, HdcCommitment, HdcContextKey, HdcMac, HdcThresholdSharing,
};

// ═══════════════════════════════════════════════════════════════════════════════
// HDC-MAC PROPERTIES
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    /// **Correctness**: MAC computed with a key always verifies with the same key.
    /// This must hold for ALL message/key pairs — a single failure would indicate
    /// a broken implementation.
    #[test]
    fn prop_mac_correct_key_always_verifies(
        msg_seed in 0u64..1_000_000,
        key_seed in 0u64..1_000_000,
    ) {
        let message = BinaryHV::random(msg_seed);
        let key = BinaryHV::random(key_seed);
        let mac = HdcMac::compute(&message, &key);
        prop_assert!(HdcMac::verify(&message, &key, &mac),
            "MAC should always verify with correct key (msg_seed={}, key_seed={})",
            msg_seed, key_seed);
    }

    /// **Unforgeability**: MAC computed with key_a never verifies with key_b.
    /// At D=16,384, P(collision) = 2^{-16384}. We test 256 pairs — if any verify,
    /// the implementation is catastrophically broken.
    #[test]
    fn prop_mac_wrong_key_never_verifies(
        msg_seed in 0u64..1_000_000,
        key_a_seed in 0u64..500_000,
        key_b_seed in 500_001u64..1_000_000,
    ) {
        let message = BinaryHV::random(msg_seed);
        let key_a = BinaryHV::random(key_a_seed);
        let key_b = BinaryHV::random(key_b_seed);
        let mac = HdcMac::compute(&message, &key_a);

        // Exact verification must fail
        prop_assert!(!HdcMac::verify(&message, &key_b, &mac),
            "MAC must never verify with wrong key");

        // Noisy verification must also fail at threshold 0.95
        prop_assert!(!HdcMac::verify_noisy(&message, &key_b, &mac, HDC_MAC_NOISY_THRESHOLD),
            "Noisy MAC must not verify with wrong key at threshold {}",
            HDC_MAC_NOISY_THRESHOLD);
    }

    /// **Determinism**: Same inputs always produce the same MAC.
    #[test]
    fn prop_mac_deterministic(
        msg_seed in 0u64..1_000_000,
        key_seed in 0u64..1_000_000,
    ) {
        let message = BinaryHV::random(msg_seed);
        let key = BinaryHV::random(key_seed);
        let mac1 = HdcMac::compute(&message, &key);
        let mac2 = HdcMac::compute(&message, &key);
        prop_assert_eq!(mac1, mac2);
    }

    /// **Wrong-key similarity**: MAC with wrong key should produce ~0.5 similarity
    /// to the correct MAC (indistinguishable from random).
    #[test]
    fn prop_mac_wrong_key_similarity_near_random(
        msg_seed in 0u64..100_000,
        key_a_seed in 0u64..50_000,
        key_b_seed in 50_001u64..100_000,
    ) {
        let message = BinaryHV::random(msg_seed);
        let key_a = BinaryHV::random(key_a_seed);
        let key_b = BinaryHV::random(key_b_seed);
        let mac_a = HdcMac::compute(&message, &key_a);
        let mac_b = HdcMac::compute(&message, &key_b);
        let sim = mac_a.similarity(&mac_b);
        // At D=16,384, random similarity ≈ 0.5 ± 0.0039 (3σ).
        // Allow wider tolerance (±0.05) for proptest robustness.
        prop_assert!((sim - 0.5).abs() < 0.05,
            "Wrong-key MAC similarity should be ~0.5, got {}", sim);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// THRESHOLD SHARING PROPERTIES
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    /// **Exact recovery**: k shares always recover the secret exactly.
    #[test]
    fn prop_threshold_k_shares_always_recover(
        secret_seed in 0u64..100_000,
        split_seed in 0u64..100_000,
    ) {
        let secret = BinaryHV::random(secret_seed);
        // Use k=3, n=5 (fixed for proptest speed)
        let shares = HdcThresholdSharing::split(&secret, 3, 5, split_seed);
        let recovered = HdcThresholdSharing::recover(&shares[..3]);
        prop_assert_eq!(recovered, secret,
            "k=3 shares should always recover the secret exactly");
    }

    /// **Information-theoretic security**: Individual shares have ~0.5 similarity
    /// to the secret (one-time pad — no information leakage).
    #[test]
    fn prop_threshold_individual_share_leaks_nothing(
        secret_seed in 0u64..100_000,
        split_seed in 0u64..100_000,
        share_idx in 0usize..5,
    ) {
        let secret = BinaryHV::random(secret_seed);
        let shares = HdcThresholdSharing::split(&secret, 3, 5, split_seed);
        let sim = shares[share_idx].share.similarity(&secret);
        // OTP: share should be indistinguishable from random relative to secret
        prop_assert!((sim - 0.5).abs() < 0.05,
            "Individual share should be ~0.5 similar to secret (OTP), got {}", sim);
    }

    /// **All-shares recovery**: Using all n shares always recovers exactly.
    #[test]
    fn prop_threshold_all_shares_recover(
        secret_seed in 0u64..100_000,
        split_seed in 0u64..100_000,
        n in (3usize..=9).prop_filter("must be odd", |n| n % 2 == 1),
    ) {
        let secret = BinaryHV::random(secret_seed);
        let shares = HdcThresholdSharing::split(&secret, n, n, split_seed);
        let recovered = HdcThresholdSharing::recover(&shares);
        prop_assert_eq!(recovered, secret,
            "All n={} shares should recover exactly", n);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONTEXT KEY PROPERTIES
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    /// **Non-commutativity**: Different sensor orderings always produce different keys.
    /// This is critical — sensor_bridge encodes physical context, and
    /// (GPS, accel) must differ from (accel, GPS).
    #[test]
    fn prop_context_key_order_matters(
        seed_a in 0u64..100_000,
        seed_b in 100_001u64..200_000,
    ) {
        let s1 = BinaryHV::random(seed_a);
        let s2 = BinaryHV::random(seed_b);
        let key_12 = HdcContextKey::derive(&[s1, s2]);
        let key_21 = HdcContextKey::derive(&[s2, s1]);
        prop_assert_ne!(key_12, key_21,
            "Different sensor orderings must produce different context keys");
    }

    /// **Determinism**: Same sensors always produce the same symmetric key.
    #[test]
    fn prop_context_key_deterministic(
        seed_a in 0u64..100_000,
        seed_b in 100_001u64..200_000,
        seed_c in 200_001u64..300_000,
    ) {
        let sensors = vec![
            BinaryHV::random(seed_a),
            BinaryHV::random(seed_b),
            BinaryHV::random(seed_c),
        ];
        let key1 = HdcContextKey::derive_symmetric(&sensors);
        let key2 = HdcContextKey::derive_symmetric(&sensors);
        prop_assert_eq!(key1, key2);
    }

    /// **Sensitivity**: Changing any single sensor changes the key.
    #[test]
    fn prop_context_key_sensitive_to_each_sensor(
        seed_a in 0u64..100_000,
        seed_b in 100_001u64..200_000,
        seed_c in 200_001u64..300_000,
        seed_alt in 300_001u64..400_000,
    ) {
        let original = vec![
            BinaryHV::random(seed_a),
            BinaryHV::random(seed_b),
            BinaryHV::random(seed_c),
        ];
        let key_original = HdcContextKey::derive_symmetric(&original);

        // Change sensor 0
        let mut modified = original.clone();
        modified[0] = BinaryHV::random(seed_alt);
        let key_mod0 = HdcContextKey::derive_symmetric(&modified);
        prop_assert_ne!(key_original, key_mod0,
            "Changing sensor 0 must change the key");

        // Change sensor 1
        let mut modified = original.clone();
        modified[1] = BinaryHV::random(seed_alt);
        let key_mod1 = HdcContextKey::derive_symmetric(&modified);
        prop_assert_ne!(key_original, key_mod1,
            "Changing sensor 1 must change the key");

        // Change sensor 2
        let mut modified = original;
        modified[2] = BinaryHV::random(seed_alt);
        let key_mod2 = HdcContextKey::derive_symmetric(&modified);
        prop_assert_ne!(key_original, key_mod2,
            "Changing sensor 2 must change the key");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COMMITMENT PROPERTIES
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    /// **Correctness**: Commitment always verifies with the correct secret and offset.
    #[test]
    fn prop_commitment_correct_verifies(
        secret_seed in 0u64..100_000,
        offset in 1usize..16384,
    ) {
        let secret = BinaryHV::random(secret_seed);
        let commitment = HdcCommitment::commit(&secret, offset);
        prop_assert!(HdcCommitment::verify(&commitment.commitment, &secret, offset),
            "Commitment should verify with correct (secret, offset)");
    }

    /// **Binding**: Wrong offset never verifies (no second preimage).
    #[test]
    fn prop_commitment_wrong_offset_fails(
        secret_seed in 0u64..100_000,
        offset in 1usize..8192,
        wrong_offset in 8193usize..16384,
    ) {
        let secret = BinaryHV::random(secret_seed);
        let commitment = HdcCommitment::commit(&secret, offset);
        prop_assert!(!HdcCommitment::verify(&commitment.commitment, &secret, wrong_offset),
            "Commitment must not verify with wrong offset");
    }

    /// **Hiding**: Commitment has ~0.5 similarity to the secret
    /// (quasi-independent in 16,384D).
    #[test]
    fn prop_commitment_hides_secret(
        secret_seed in 0u64..100_000,
        offset in 1usize..16384,
    ) {
        let secret = BinaryHV::random(secret_seed);
        let commitment = HdcCommitment::commit(&secret, offset);
        let sim = commitment.commitment.similarity(&secret);
        prop_assert!((sim - 0.5).abs() < 0.05,
            "Commitment should be ~0.5 similar to secret (hiding), got {}", sim);
    }
}
