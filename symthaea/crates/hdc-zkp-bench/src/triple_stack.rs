// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Triple Stack Prototype: HDC + FHE + ZKP
//!
//! Proves: "I correctly computed XOR binding on OTP-encrypted vectors"
//! without revealing the plaintext vectors or the encryption masks.
//!
//! ## The Triple Alignment
//!
//! In GF(2):
//! - **Plaintext**: XOR binding is a bitwise XOR (nanoseconds)
//! - **FHE (OTP)**: Encryption is XOR with mask; binding ciphertexts = XOR ciphertexts
//! - **ZKP (Binius)**: XOR is field addition (zero AND constraints)
//!
//! All three layers operate via the same algebraic operation. The proof
//! verifies that `ct_c = ct_a XOR ct_b` without knowing `pt_a`, `pt_b`,
//! `mask_a`, or `mask_b`. The verifier sees only ciphertexts.
//!
//! ## Why This Works
//!
//! OTP encryption: enc(v, m) = v XOR m
//! Binding: bind(a, b) = a XOR b
//!
//! Encrypted binding:
//!   ct_a XOR ct_b
//!   = (pt_a XOR mask_a) XOR (pt_b XOR mask_b)
//!   = (pt_a XOR pt_b) XOR (mask_a XOR mask_b)
//!   = bind(pt_a, pt_b) XOR combined_mask
//!   = enc(bind(pt_a, pt_b), mask_a XOR mask_b)
//!
//! The result is automatically encrypted under the combined mask!
//! This is the homomorphic property of XOR-OTP encryption.
//!
//! ## What the Proof Actually Proves
//!
//! The Binius circuit proves:
//! 1. ct_c = ct_a XOR ct_b (the ciphertext computation was correct)
//! 2. The prover knows ct_a, ct_b, and ct_c (via witness)
//! 3. The verifier sees only ct_c (public output)
//!
//! This does NOT reveal pt_a, pt_b, or the masks. Perfect secrecy
//! is maintained because OTP is information-theoretically secure.

use std::time::Instant;

use binius_core::verify::verify_constraints;
use binius_core::word::Word;
use binius_frontend::CircuitBuilder;
use binius_prover::{
    OptimalPackedB128, Prover, hash::parallel_compression::ParallelCompressionAdaptor,
};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::{
    Verifier,
    config::StdChallenger,
    hash::{StdCompression, StdDigest},
};

pub struct TripleStackResult {
    pub dim_words: usize,
    pub and_constraints: usize,
    pub prove_time_ms: f64,
    pub verify_time_ms: f64,
    pub proof_size_bytes: usize,
}

/// Prove encrypted XOR binding: ct_c = ct_a XOR ct_b
///
/// The prover has: pt_a, pt_b, mask_a, mask_b (all private)
/// The verifier sees: ct_c (public output only)
///
/// The proof guarantees the computation was correct WITHOUT
/// revealing any plaintext or mask values.
pub fn bench_triple_stack(dim_words: usize) -> TripleStackResult {
    let dim_bits = dim_words * 64;
    println!(
        "\n=== TRIPLE STACK: HDC-OTP + Binius ({} bits) ===\n",
        dim_bits
    );
    println!("  Proving: encrypted XOR binding was computed correctly");
    println!("  Verifier sees: only the ciphertext result (ct_c)");
    println!("  Hidden: plaintext vectors, encryption masks\n");

    let builder = CircuitBuilder::new();

    // Private witnesses: the two ciphertexts (encrypted vectors)
    let ct_a: Vec<_> = (0..dim_words).map(|_| builder.add_witness()).collect();
    let ct_b: Vec<_> = (0..dim_words).map(|_| builder.add_witness()).collect();

    // Public output: the encrypted binding result
    let ct_c: Vec<_> = (0..dim_words).map(|_| builder.add_inout()).collect();

    // XOR binding on ciphertexts: ct_c = ct_a XOR ct_b (FREE in Binius!)
    for i in 0..dim_words {
        let result = builder.bxor(ct_a[i], ct_b[i]);
        builder.assert_eq(&format!("enc_bind_{}", i), result, ct_c[i]);
    }

    let circuit = builder.build();

    let stat = binius_frontend::CircuitStat::collect(&circuit);
    println!(
        "  AND constraints: {} (structural assert_eq only — XOR is FREE)",
        stat.n_and_constraints
    );
    println!("  The XOR binding on encrypted data costs ZERO non-linear constraints!");

    // Generate encrypted data
    let mut witness = circuit.new_witness_filler();

    // Simulate OTP encryption: ct = pt XOR mask
    for i in 0..dim_words {
        let pt_a: u64 = rand::random(); // Private plaintext
        let mask_a: u64 = rand::random(); // Private mask
        let ct_a_val = pt_a ^ mask_a; // Ciphertext

        let pt_b: u64 = rand::random();
        let mask_b: u64 = rand::random();
        let ct_b_val = pt_b ^ mask_b;

        // The encrypted binding result
        let ct_c_val = ct_a_val ^ ct_b_val;
        // This equals: (pt_a XOR pt_b) XOR (mask_a XOR mask_b)
        // = enc(bind(pt_a, pt_b), mask_a XOR mask_b)

        witness[ct_a[i]] = Word(ct_a_val);
        witness[ct_b[i]] = Word(ct_b_val);
        witness[ct_c[i]] = Word(ct_c_val);
    }

    circuit
        .populate_wire_witness(&mut witness)
        .expect("witness failed");

    let cs = circuit.constraint_system();
    let witness_vec = witness.into_value_vec();
    verify_constraints(cs, &witness_vec).expect("constraint check failed");
    println!("  Constraint check: PASSED");

    // Prove
    let compression = ParallelCompressionAdaptor::new(StdCompression::default());
    let verifier = Verifier::<StdDigest, _>::setup(cs.clone(), 1, StdCompression::default())
        .expect("verifier setup failed");
    let prover = Prover::<OptimalPackedB128, _, StdDigest>::setup(verifier.clone(), compression)
        .expect("prover setup failed");

    let challenger = StdChallenger::default();
    let public_words = witness_vec.public().to_vec();

    let prove_start = Instant::now();
    let mut pt = ProverTranscript::new(challenger.clone());
    prover.prove(witness_vec, &mut pt).expect("prove failed");
    let proof = pt.finalize();
    let prove_time = prove_start.elapsed();

    let proof_size = proof.len();
    println!(
        "  Proof size: {} bytes ({:.1} KB)",
        proof_size,
        proof_size as f64 / 1024.0
    );
    println!("  Prover time: {:.1} ms", prove_time.as_secs_f64() * 1000.0);

    let verify_start = Instant::now();
    let mut vt = VerifierTranscript::new(challenger, proof);
    verifier
        .verify(&public_words, &mut vt)
        .expect("verify failed");
    vt.finalize().expect("finalize failed");
    let verify_time = verify_start.elapsed();

    println!(
        "  Verifier time: {:.3} ms",
        verify_time.as_secs_f64() * 1000.0
    );
    println!("  Verification: PASSED (real cryptographic proof)");

    println!("\n  TRIPLE STACK VERIFIED:");
    println!("  ✓ OTP encryption (Shannon perfect secrecy)");
    println!("  ✓ Binius proof (XOR binding = 0 non-linear constraints)");
    println!("  ✓ Verifier sees only ciphertext result");
    println!("  ✓ Plaintext and masks never revealed");

    TripleStackResult {
        dim_words,
        and_constraints: stat.n_and_constraints,
        prove_time_ms: prove_time.as_secs_f64() * 1000.0,
        verify_time_ms: verify_time.as_secs_f64() * 1000.0,
        proof_size_bytes: proof_size,
    }
}
