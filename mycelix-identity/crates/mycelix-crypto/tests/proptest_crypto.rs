//! Property-based tests for mycelix-crypto sign/verify round-trips.
//!
//! Uses proptest to verify that sign→verify holds for all algorithms
//! with arbitrary payloads, and that wrong-message detection works.

#![cfg(feature = "native")]

use mycelix_crypto::algorithm::AlgorithmId;
use mycelix_crypto::envelope::{TaggedPublicKey, TaggedSignature};
use mycelix_crypto::pqc::dilithium::{
    MlDsa65Signer, MlDsa65Verifier, MlDsa87Signer, MlDsa87Verifier,
};
use mycelix_crypto::pqc::ed25519_native::{Ed25519Signer, Ed25519Verifier};
use mycelix_crypto::pqc::encryption::XChaCha20Encryptor;
use mycelix_crypto::pqc::hybrid::{HybridSigner, HybridVerifier};
use mycelix_crypto::pqc::ml_kem::MlKem768KeyPair;
use mycelix_crypto::pqc::sphincs::{SlhDsaSha2128sSigner, SlhDsaSha2128sVerifier};
use mycelix_crypto::traits::{Encryptor, KeyEncapsulator, Signer, Verifier};
use proptest::prelude::*;

// ============================================================================
// Signature round-trip properties
// ============================================================================

// Ed25519: sign→verify succeeds for any message
proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn ed25519_sign_verify_roundtrip(msg in prop::collection::vec(any::<u8>(), 0..4096)) {
        let signer = Ed25519Signer::generate();
        let pk = signer.public_key();
        let sig = signer.sign(&msg).unwrap();
        let ok = Ed25519Verifier.verify(&pk, &msg, &sig).unwrap();
        prop_assert!(ok, "Ed25519 verify failed for msg len {}", msg.len());
    }

    #[test]
    fn ed25519_wrong_message_rejected(
        msg in prop::collection::vec(any::<u8>(), 1..1024),
        flip_idx in 0usize..1024,
    ) {
        let signer = Ed25519Signer::generate();
        let pk = signer.public_key();
        let sig = signer.sign(&msg).unwrap();
        let mut wrong_msg = msg.clone();
        let idx = flip_idx % wrong_msg.len();
        wrong_msg[idx] ^= 0xFF;
        let ok = Ed25519Verifier.verify(&pk, &wrong_msg, &sig).unwrap();
        prop_assert!(!ok, "Ed25519 should reject tampered message");
    }
}

// ML-DSA-65: sign→verify succeeds for any message
// Reduced case count due to PQC signing cost (~20ms/op)
proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn mldsa65_sign_verify_roundtrip(msg in prop::collection::vec(any::<u8>(), 0..2048)) {
        let signer = MlDsa65Signer::generate();
        let pk = signer.public_key();
        let sig = signer.sign(&msg).unwrap();
        let ok = MlDsa65Verifier.verify(&pk, &msg, &sig).unwrap();
        prop_assert!(ok, "ML-DSA-65 verify failed for msg len {}", msg.len());
    }

    #[test]
    fn mldsa65_wrong_message_rejected(
        msg in prop::collection::vec(any::<u8>(), 1..512),
        flip_idx in 0usize..512,
    ) {
        let signer = MlDsa65Signer::generate();
        let pk = signer.public_key();
        let sig = signer.sign(&msg).unwrap();
        let mut wrong_msg = msg.clone();
        let idx = flip_idx % wrong_msg.len();
        wrong_msg[idx] ^= 0xFF;
        let ok = MlDsa65Verifier.verify(&pk, &wrong_msg, &sig).unwrap();
        prop_assert!(!ok, "ML-DSA-65 should reject tampered message");
    }
}

// ML-DSA-87: sign→verify succeeds + tamper detection
proptest! {
    #![proptest_config(ProptestConfig::with_cases(4))]

    #[test]
    fn mldsa87_sign_verify_roundtrip(msg in prop::collection::vec(any::<u8>(), 0..1024)) {
        let signer = MlDsa87Signer::generate();
        let pk = signer.public_key();
        let sig = signer.sign(&msg).unwrap();
        let ok = MlDsa87Verifier.verify(&pk, &msg, &sig).unwrap();
        prop_assert!(ok, "ML-DSA-87 verify failed for msg len {}", msg.len());
    }

    #[test]
    fn mldsa87_wrong_message_rejected(
        msg in prop::collection::vec(any::<u8>(), 1..512),
        flip_idx in 0usize..512,
    ) {
        let signer = MlDsa87Signer::generate();
        let pk = signer.public_key();
        let sig = signer.sign(&msg).unwrap();
        let mut wrong_msg = msg.clone();
        let idx = flip_idx % wrong_msg.len();
        wrong_msg[idx] ^= 0xFF;
        let ok = MlDsa87Verifier.verify(&pk, &wrong_msg, &sig).unwrap();
        prop_assert!(!ok, "ML-DSA-87 should reject tampered message");
    }
}

// SLH-DSA-SHA2-128s: sign→verify succeeds + tamper detection
// Slow (~1s/sign) so limited cases
proptest! {
    #![proptest_config(ProptestConfig::with_cases(4))]

    #[test]
    fn slhdsa_sign_verify_roundtrip(msg in prop::collection::vec(any::<u8>(), 0..512)) {
        let signer = SlhDsaSha2128sSigner::generate();
        let pk = signer.public_key();
        let sig = signer.sign(&msg).unwrap();
        let ok = SlhDsaSha2128sVerifier.verify(&pk, &msg, &sig).unwrap();
        prop_assert!(ok, "SLH-DSA verify failed for msg len {}", msg.len());
    }

    #[test]
    fn slhdsa_wrong_message_rejected(
        msg in prop::collection::vec(any::<u8>(), 1..256),
        flip_idx in 0usize..256,
    ) {
        let signer = SlhDsaSha2128sSigner::generate();
        let pk = signer.public_key();
        let sig = signer.sign(&msg).unwrap();
        let mut wrong_msg = msg.clone();
        let idx = flip_idx % wrong_msg.len();
        wrong_msg[idx] ^= 0xFF;
        let ok = SlhDsaSha2128sVerifier.verify(&pk, &wrong_msg, &sig).unwrap();
        prop_assert!(!ok, "SLH-DSA should reject tampered message");
    }
}

// Hybrid Ed25519+ML-DSA-65: sign→verify succeeds
proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    #[test]
    fn hybrid_sign_verify_roundtrip(msg in prop::collection::vec(any::<u8>(), 0..2048)) {
        let signer = HybridSigner::generate();
        let pk = signer.public_key();
        let sig = signer.sign(&msg).unwrap();
        let ok = HybridVerifier.verify(&pk, &msg, &sig).unwrap();
        prop_assert!(ok, "Hybrid verify failed for msg len {}", msg.len());
    }

    #[test]
    fn hybrid_wrong_message_rejected(
        msg in prop::collection::vec(any::<u8>(), 1..512),
        flip_idx in 0usize..512,
    ) {
        let signer = HybridSigner::generate();
        let pk = signer.public_key();
        let sig = signer.sign(&msg).unwrap();
        let mut wrong_msg = msg.clone();
        let idx = flip_idx % wrong_msg.len();
        wrong_msg[idx] ^= 0xFF;
        let ok = HybridVerifier.verify(&pk, &wrong_msg, &sig).unwrap();
        prop_assert!(!ok, "Hybrid should reject tampered message");
    }
}

// ============================================================================
// Key/signature size invariants
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn ed25519_key_size_invariant(msg in prop::collection::vec(any::<u8>(), 0..256)) {
        let signer = Ed25519Signer::generate();
        let pk = signer.public_key();
        let sig = signer.sign(&msg).unwrap();
        prop_assert_eq!(pk.key_bytes.len(), 32, "Ed25519 pubkey must be 32 bytes");
        prop_assert_eq!(sig.signature_bytes.len(), 64, "Ed25519 sig must be 64 bytes");
        prop_assert_eq!(pk.algorithm, AlgorithmId::Ed25519);
        prop_assert_eq!(sig.algorithm, AlgorithmId::Ed25519);
    }
}

#[test]
fn mldsa65_key_size_invariant() {
    let signer = MlDsa65Signer::generate();
    let pk = signer.public_key();
    let sig = signer.sign(b"test").unwrap();
    assert_eq!(
        pk.key_bytes.len(),
        1952,
        "ML-DSA-65 pubkey must be 1952 bytes"
    );
    assert_eq!(
        sig.signature_bytes.len(),
        3309,
        "ML-DSA-65 sig must be 3309 bytes"
    );
}

#[test]
fn mldsa87_key_size_invariant() {
    let signer = MlDsa87Signer::generate();
    let pk = signer.public_key();
    let sig = signer.sign(b"test").unwrap();
    assert_eq!(pk.key_bytes.len(), 2592);
    assert_eq!(sig.signature_bytes.len(), 4627);
}

#[test]
fn slhdsa_key_size_invariant() {
    let signer = SlhDsaSha2128sSigner::generate();
    let pk = signer.public_key();
    let sig = signer.sign(b"test").unwrap();
    assert_eq!(pk.key_bytes.len(), 32);
    assert_eq!(sig.signature_bytes.len(), 7856);
}

#[test]
fn hybrid_key_size_invariant() {
    let signer = HybridSigner::generate();
    let pk = signer.public_key();
    let sig = signer.sign(b"test").unwrap();
    assert_eq!(
        pk.key_bytes.len(),
        32 + 1952,
        "Hybrid pubkey = Ed25519 + ML-DSA-65"
    );
    assert_eq!(
        sig.signature_bytes.len(),
        64 + 3309,
        "Hybrid sig = Ed25519 + ML-DSA-65"
    );
}

// ============================================================================
// Multibase encoding round-trips
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn ed25519_multibase_roundtrip(_seed in 0u64..1000) {
        let signer = Ed25519Signer::generate();
        let pk = signer.public_key();
        let multibase = pk.to_multibase();
        let restored = TaggedPublicKey::from_multibase(&multibase).unwrap();
        prop_assert_eq!(restored.algorithm, AlgorithmId::Ed25519);
        prop_assert_eq!(restored.key_bytes, pk.key_bytes);
    }

    #[test]
    fn ed25519_signature_multibase_roundtrip(msg in prop::collection::vec(any::<u8>(), 0..256)) {
        let signer = Ed25519Signer::generate();
        let sig = signer.sign(&msg).unwrap();
        let multibase = sig.to_multibase();
        let restored = TaggedSignature::from_multibase(&multibase).unwrap();
        prop_assert_eq!(restored.algorithm, AlgorithmId::Ed25519);
        prop_assert_eq!(restored.signature_bytes, sig.signature_bytes);
    }
}

#[test]
fn mldsa65_multibase_roundtrip() {
    let signer = MlDsa65Signer::generate();
    let pk = signer.public_key();
    let multibase = pk.to_multibase();
    let restored = TaggedPublicKey::from_multibase(&multibase).unwrap();
    assert_eq!(restored.algorithm, AlgorithmId::MlDsa65);
    assert_eq!(restored.key_bytes, pk.key_bytes);
}

#[test]
fn hybrid_multibase_roundtrip() {
    let signer = HybridSigner::generate();
    let pk = signer.public_key();
    let multibase = pk.to_multibase();
    let restored = TaggedPublicKey::from_multibase(&multibase).unwrap();
    assert_eq!(restored.algorithm, AlgorithmId::HybridEd25519MlDsa65);
    assert_eq!(restored.key_bytes, pk.key_bytes);
}

// ============================================================================
// KEM encapsulate/decapsulate round-trip
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4))]

    #[test]
    fn kem_roundtrip(_seed in 0u64..100) {
        let recipient = MlKem768KeyPair::generate();
        let sender = MlKem768KeyPair::generate();
        let recipient_pk = recipient.public_key();
        let (ciphertext, sender_secret) = sender.encapsulate(&recipient_pk).unwrap();
        let recipient_secret = recipient.decapsulate(&ciphertext).unwrap();
        prop_assert!(!sender_secret.is_empty());
        prop_assert_eq!(sender_secret, recipient_secret,
            "KEM shared secrets must match");
    }
}

// ============================================================================
// KEM: tampered ciphertext and wrong recipient detection
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4))]

    #[test]
    fn kem_tampered_ciphertext_rejected(flip_idx in 0usize..1088) {
        let recipient = MlKem768KeyPair::generate();
        let sender = MlKem768KeyPair::generate();
        let recipient_pk = recipient.public_key();
        let (mut ciphertext, sender_secret) = sender.encapsulate(&recipient_pk).unwrap();
        // Flip a bit in the ciphertext
        let idx = flip_idx % ciphertext.len();
        ciphertext[idx] ^= 0x01;
        // ML-KEM implicit rejection: always returns a secret, but it must differ
        match recipient.decapsulate(&ciphertext) {
            Ok(tampered_secret) => prop_assert_ne!(tampered_secret, sender_secret,
                "Tampered KEM ciphertext must produce different shared secret"),
            Err(_) => {} // Error is also acceptable
        }
    }

    #[test]
    fn kem_wrong_recipient_rejected(_seed in 0u64..100) {
        let recipient = MlKem768KeyPair::generate();
        let wrong_recipient = MlKem768KeyPair::generate();
        let sender = MlKem768KeyPair::generate();
        let recipient_pk = recipient.public_key();
        let (ciphertext, sender_secret) = sender.encapsulate(&recipient_pk).unwrap();
        // Decapsulate with wrong keypair
        match wrong_recipient.decapsulate(&ciphertext) {
            Ok(wrong_secret) => prop_assert_ne!(wrong_secret, sender_secret,
                "Wrong recipient must derive different shared secret"),
            Err(_) => {} // Error is also acceptable
        }
    }
}

// ============================================================================
// AEAD encrypt/decrypt round-trip
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn aead_roundtrip(plaintext in prop::collection::vec(any::<u8>(), 0..4096)) {
        let key = [42u8; 32];
        let encryptor = XChaCha20Encryptor::from_raw_key(&key);
        let envelope = encryptor.encrypt(&plaintext, b"", "test-key").unwrap();
        let decrypted = encryptor.decrypt(&envelope, b"").unwrap();
        prop_assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn aead_different_nonces(plaintext in prop::collection::vec(any::<u8>(), 1..256)) {
        let key = [42u8; 32];
        let encryptor = XChaCha20Encryptor::from_raw_key(&key);
        let env1 = encryptor.encrypt(&plaintext, b"", "k").unwrap();
        let env2 = encryptor.encrypt(&plaintext, b"", "k").unwrap();
        // Same plaintext, different nonces → different ciphertexts
        prop_assert_ne!(env1.ciphertext, env2.ciphertext,
            "Same plaintext should produce different ciphertexts (random nonce)");
    }
}

// ============================================================================
// AEAD tamper detection
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn aead_tampered_ciphertext_rejected(
        plaintext in prop::collection::vec(any::<u8>(), 1..1024),
        flip_idx in 0usize..1024,
    ) {
        let key = [42u8; 32];
        let encryptor = XChaCha20Encryptor::from_raw_key(&key);
        let mut envelope = encryptor.encrypt(&plaintext, b"", "test-key").unwrap();
        let idx = flip_idx % envelope.ciphertext.len();
        envelope.ciphertext[idx] ^= 0x01;
        let result = encryptor.decrypt(&envelope, b"");
        prop_assert!(result.is_err(), "Tampered AEAD ciphertext must be rejected");
    }

    #[test]
    fn aead_wrong_key_rejected(plaintext in prop::collection::vec(any::<u8>(), 1..256)) {
        let key1 = [42u8; 32];
        let key2 = [99u8; 32];
        let encryptor1 = XChaCha20Encryptor::from_raw_key(&key1);
        let encryptor2 = XChaCha20Encryptor::from_raw_key(&key2);
        let envelope = encryptor1.encrypt(&plaintext, b"", "k").unwrap();
        let result = encryptor2.decrypt(&envelope, b"");
        prop_assert!(result.is_err(), "Wrong key must reject AEAD ciphertext");
    }
}

// ============================================================================
// Cross-algorithm: wrong verifier rejects
// ============================================================================

#[test]
fn ed25519_sig_rejected_by_mldsa_verifier() {
    let signer = Ed25519Signer::generate();
    let pk = signer.public_key();
    let sig = signer.sign(b"hello").unwrap();
    // Create a ML-DSA-65 verifier and try to verify an Ed25519 signature
    let result = MlDsa65Verifier.verify(&pk, b"hello", &sig);
    // Should either error or return false
    match result {
        Ok(valid) => assert!(!valid, "ML-DSA verifier should not accept Ed25519 sig"),
        Err(_) => {} // Error is also acceptable (algorithm mismatch)
    }
}

#[test]
fn empty_message_sign_verify() {
    let signer = Ed25519Signer::generate();
    let pk = signer.public_key();
    let sig = signer.sign(b"").unwrap();
    let ok = Ed25519Verifier.verify(&pk, b"", &sig).unwrap();
    assert!(ok, "Empty message should sign and verify");
}

// ============================================================================
// Fuzz-style: from_multibase with arbitrary bytes (parser robustness)
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn pubkey_from_multibase_never_panics(raw in prop::collection::vec(any::<u8>(), 0..512)) {
        // Encode random bytes as base58btc with 'z' prefix
        let encoded = format!("z{}", bs58::encode(&raw).into_string());
        // Must not panic — errors are fine
        let _ = TaggedPublicKey::from_multibase(&encoded);
    }

    #[test]
    fn sig_from_multibase_never_panics(raw in prop::collection::vec(any::<u8>(), 0..512)) {
        let encoded = format!("z{}", bs58::encode(&raw).into_string());
        let _ = TaggedSignature::from_multibase(&encoded);
    }

    #[test]
    fn pubkey_from_multibase_rejects_bad_prefix(s in "[^z].{0,100}") {
        let result = TaggedPublicKey::from_multibase(&s);
        prop_assert!(result.is_err(), "Non-'z' prefix should be rejected");
    }

    #[test]
    fn pubkey_from_multibase_rejects_empty(_seed in 0u64..1) {
        assert!(TaggedPublicKey::from_multibase("").is_err());
        assert!(TaggedPublicKey::from_multibase("z").is_err());
    }

    #[test]
    fn sig_from_multibase_rejects_empty(_seed in 0u64..1) {
        assert!(TaggedSignature::from_multibase("").is_err());
        assert!(TaggedSignature::from_multibase("z").is_err());
    }

    #[test]
    fn pubkey_from_multibase_rejects_invalid_base58(
        bad_char in prop::sample::select(vec!['0', 'O', 'I', 'l']),
    ) {
        // These characters are not in base58btc alphabet
        let s = format!("z{}", bad_char);
        let result = TaggedPublicKey::from_multibase(&s);
        prop_assert!(result.is_err(), "Invalid base58 char should be rejected");
    }
}

// Truncation: valid multibase keys with bytes chopped off should fail or return wrong result
#[test]
fn pubkey_truncated_multibase_rejected() {
    let signer = Ed25519Signer::generate();
    let pk = signer.public_key();
    let multibase = pk.to_multibase();
    // Chop off last 5 chars
    let truncated = &multibase[..multibase.len().saturating_sub(5)];
    let result = TaggedPublicKey::from_multibase(truncated);
    // Should either error or produce a different key
    match result {
        Err(_) => {} // Expected
        Ok(restored) => assert_ne!(
            restored.key_bytes, pk.key_bytes,
            "Truncated multibase should not produce original key"
        ),
    }
}

#[test]
fn sig_truncated_multibase_rejected() {
    let signer = Ed25519Signer::generate();
    let sig = signer.sign(b"test").unwrap();
    let multibase = sig.to_multibase();
    let truncated = &multibase[..multibase.len().saturating_sub(5)];
    let result = TaggedSignature::from_multibase(truncated);
    match result {
        Err(_) => {}
        Ok(restored) => assert_ne!(
            restored.signature_bytes, sig.signature_bytes,
            "Truncated multibase should not produce original sig"
        ),
    }
}
