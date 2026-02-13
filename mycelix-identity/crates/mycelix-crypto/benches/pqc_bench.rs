//! Criterion benchmarks for PQC cryptographic operations.
//!
//! Run with: cargo bench --features native
//!
//! Covers keygen, sign, and verify for all 5 signature algorithms,
//! plus ML-KEM-768 encapsulate/decapsulate and XChaCha20-Poly1305 AEAD.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

use mycelix_crypto::pqc::ed25519_native::{Ed25519Signer, Ed25519Verifier};
use mycelix_crypto::pqc::dilithium::{MlDsa65Signer, MlDsa65Verifier, MlDsa87Signer, MlDsa87Verifier};
use mycelix_crypto::pqc::sphincs::{SlhDsaSha2128sSigner, SlhDsaSha2128sVerifier};
use mycelix_crypto::pqc::hybrid::{HybridSigner, HybridVerifier};
use mycelix_crypto::pqc::ml_kem::MlKem768KeyPair;
use mycelix_crypto::pqc::encryption::XChaCha20Encryptor;
use mycelix_crypto::traits::{Signer, Verifier, KeyEncapsulator, Encryptor};

// ============================================================================
// Key generation benchmarks
// ============================================================================

fn bench_keygen(c: &mut Criterion) {
    let mut group = c.benchmark_group("keygen");

    group.bench_function("Ed25519", |b| {
        b.iter(|| black_box(Ed25519Signer::generate()))
    });

    group.bench_function("ML-DSA-65", |b| {
        b.iter(|| black_box(MlDsa65Signer::generate()))
    });

    group.bench_function("ML-DSA-87", |b| {
        b.iter(|| black_box(MlDsa87Signer::generate()))
    });

    group.bench_function("SLH-DSA-SHA2-128s", |b| {
        b.iter(|| black_box(SlhDsaSha2128sSigner::generate()))
    });

    group.bench_function("Hybrid-Ed25519-ML-DSA-65", |b| {
        b.iter(|| black_box(HybridSigner::generate()))
    });

    group.bench_function("ML-KEM-768", |b| {
        b.iter(|| black_box(MlKem768KeyPair::generate()))
    });

    group.finish();
}

// ============================================================================
// Signing benchmarks (fixed 256-byte message)
// ============================================================================

fn bench_sign(c: &mut Criterion) {
    let msg = vec![0x42u8; 256];
    let mut group = c.benchmark_group("sign");

    let ed_signer = Ed25519Signer::generate();
    group.bench_function("Ed25519", |b| {
        b.iter(|| ed_signer.sign(black_box(&msg)).unwrap())
    });

    let ml65_signer = MlDsa65Signer::generate();
    group.bench_function("ML-DSA-65", |b| {
        b.iter(|| ml65_signer.sign(black_box(&msg)).unwrap())
    });

    let ml87_signer = MlDsa87Signer::generate();
    group.bench_function("ML-DSA-87", |b| {
        b.iter(|| ml87_signer.sign(black_box(&msg)).unwrap())
    });

    let slh_signer = SlhDsaSha2128sSigner::generate();
    group.sample_size(10); // SPHINCS+ is very slow (~1s/sign)
    group.bench_function("SLH-DSA-SHA2-128s", |b| {
        b.iter(|| slh_signer.sign(black_box(&msg)).unwrap())
    });
    group.sample_size(100); // Reset for remaining

    let hybrid_signer = HybridSigner::generate();
    group.bench_function("Hybrid-Ed25519-ML-DSA-65", |b| {
        b.iter(|| hybrid_signer.sign(black_box(&msg)).unwrap())
    });

    group.finish();
}

// ============================================================================
// Verification benchmarks (fixed 256-byte message)
// ============================================================================

fn bench_verify(c: &mut Criterion) {
    let msg = vec![0x42u8; 256];
    let mut group = c.benchmark_group("verify");

    let ed_signer = Ed25519Signer::generate();
    let ed_pk = ed_signer.public_key();
    let ed_sig = ed_signer.sign(&msg).unwrap();
    group.bench_function("Ed25519", |b| {
        b.iter(|| Ed25519Verifier.verify(black_box(&ed_pk), black_box(&msg), black_box(&ed_sig)).unwrap())
    });

    let ml65_signer = MlDsa65Signer::generate();
    let ml65_pk = ml65_signer.public_key();
    let ml65_sig = ml65_signer.sign(&msg).unwrap();
    group.bench_function("ML-DSA-65", |b| {
        b.iter(|| MlDsa65Verifier.verify(black_box(&ml65_pk), black_box(&msg), black_box(&ml65_sig)).unwrap())
    });

    let ml87_signer = MlDsa87Signer::generate();
    let ml87_pk = ml87_signer.public_key();
    let ml87_sig = ml87_signer.sign(&msg).unwrap();
    group.bench_function("ML-DSA-87", |b| {
        b.iter(|| MlDsa87Verifier.verify(black_box(&ml87_pk), black_box(&msg), black_box(&ml87_sig)).unwrap())
    });

    let slh_signer = SlhDsaSha2128sSigner::generate();
    let slh_pk = slh_signer.public_key();
    let slh_sig = slh_signer.sign(&msg).unwrap();
    group.sample_size(10);
    group.bench_function("SLH-DSA-SHA2-128s", |b| {
        b.iter(|| SlhDsaSha2128sVerifier.verify(black_box(&slh_pk), black_box(&msg), black_box(&slh_sig)).unwrap())
    });
    group.sample_size(100);

    let hybrid_signer = HybridSigner::generate();
    let hybrid_pk = hybrid_signer.public_key();
    let hybrid_sig = hybrid_signer.sign(&msg).unwrap();
    group.bench_function("Hybrid-Ed25519-ML-DSA-65", |b| {
        b.iter(|| HybridVerifier.verify(black_box(&hybrid_pk), black_box(&msg), black_box(&hybrid_sig)).unwrap())
    });

    group.finish();
}

// ============================================================================
// Signing across message sizes
// ============================================================================

fn bench_sign_message_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("sign_by_size");
    let sizes = [64, 256, 1024, 4096];

    let ed_signer = Ed25519Signer::generate();
    for &size in &sizes {
        let msg = vec![0x42u8; size];
        group.bench_with_input(BenchmarkId::new("Ed25519", size), &msg, |b, m| {
            b.iter(|| ed_signer.sign(black_box(m)).unwrap())
        });
    }

    let ml65_signer = MlDsa65Signer::generate();
    for &size in &sizes {
        let msg = vec![0x42u8; size];
        group.bench_with_input(BenchmarkId::new("ML-DSA-65", size), &msg, |b, m| {
            b.iter(|| ml65_signer.sign(black_box(m)).unwrap())
        });
    }

    let hybrid_signer = HybridSigner::generate();
    for &size in &sizes {
        let msg = vec![0x42u8; size];
        group.bench_with_input(BenchmarkId::new("Hybrid", size), &msg, |b, m| {
            b.iter(|| hybrid_signer.sign(black_box(m)).unwrap())
        });
    }

    group.finish();
}

// ============================================================================
// KEM encapsulate/decapsulate
// ============================================================================

fn bench_kem(c: &mut Criterion) {
    let mut group = c.benchmark_group("kem");

    group.bench_function("ML-KEM-768/keygen", |b| {
        b.iter(|| black_box(MlKem768KeyPair::generate()))
    });

    let recipient = MlKem768KeyPair::generate();
    let sender = MlKem768KeyPair::generate();
    let recipient_pk = recipient.public_key();
    group.bench_function("ML-KEM-768/encapsulate", |b| {
        b.iter(|| sender.encapsulate(black_box(&recipient_pk)).unwrap())
    });

    let (ciphertext, _) = sender.encapsulate(&recipient_pk).unwrap();
    group.bench_function("ML-KEM-768/decapsulate", |b| {
        b.iter(|| recipient.decapsulate(black_box(&ciphertext)).unwrap())
    });

    group.finish();
}

// ============================================================================
// AEAD encrypt/decrypt
// ============================================================================

fn bench_aead(c: &mut Criterion) {
    let key = [42u8; 32];
    let encryptor = XChaCha20Encryptor::from_raw_key(&key);
    let mut group = c.benchmark_group("aead");
    let sizes = [64, 256, 1024, 4096, 16384];

    for &size in &sizes {
        let plaintext = vec![0x42u8; size];
        group.bench_with_input(BenchmarkId::new("encrypt", size), &plaintext, |b, pt| {
            b.iter(|| encryptor.encrypt(black_box(pt), b"", "k").unwrap())
        });
    }

    for &size in &sizes {
        let plaintext = vec![0x42u8; size];
        let envelope = encryptor.encrypt(&plaintext, b"", "k").unwrap();
        group.bench_with_input(BenchmarkId::new("decrypt", size), &envelope, |b, env| {
            b.iter(|| encryptor.decrypt(black_box(env), b"").unwrap())
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_keygen,
    bench_sign,
    bench_verify,
    bench_sign_message_sizes,
    bench_kem,
    bench_aead,
);
criterion_main!(benches);
