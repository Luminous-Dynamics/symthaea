// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Legacy module path for a packed-word XOR integrity-proof demonstration.
//!
//! The only constrained relation is `word_c = word_a XOR word_b`. The host
//! generates opaque random words; no encryption, mask generation, plaintext
//! relation, or participant identity is part of the circuit.
//!
//! The vendored `vendor/binius64/ARCHITECTURE.md` explicitly marks the standard
//! Binius64 prover/verifier backend as not zero-knowledge. A proof from this
//! backend therefore carries no input-confidentiality guarantee.

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

pub struct OpaqueXorBenchResult {
    pub dim_words: usize,
    pub and_constraints: usize,
    pub prove_time_ms: f64,
    pub verify_time_ms: f64,
    pub proof_size_bytes: usize,
}

/// Prove the packed-word relation `word_c = word_a XOR word_b`.
///
/// This demonstrates computational integrity only. It does not prove that the
/// words are ciphertexts, that masks were generated or applied correctly, or
/// that witness values remain confidential under the selected Binius64 backend.
pub fn bench_opaque_xor_relation(dim_words: usize) -> OpaqueXorBenchResult {
    let dim_bits = dim_words.checked_mul(64).expect("dimension overflow");
    println!(
        "\n=== OPAQUE XOR RELATION: Binius64 ({} bits) ===\n",
        dim_bits
    );
    println!("  Statement: public word_c = witness word_a XOR witness word_b");
    println!("  Scope: computational integrity only; witness confidentiality is not guaranteed.\n");

    let builder = CircuitBuilder::new();

    // Witness words: the circuit does not establish encryption or confidentiality.
    let word_a: Vec<_> = (0..dim_words).map(|_| builder.add_witness()).collect();
    let word_b: Vec<_> = (0..dim_words).map(|_| builder.add_witness()).collect();

    // Public output: the XOR result
    let word_c: Vec<_> = (0..dim_words).map(|_| builder.add_inout()).collect();

    // Constrain the opaque-word XOR relation: word_c = word_a XOR word_b.
    for i in 0..dim_words {
        let result = builder.bxor(word_a[i], word_b[i]);
        builder.assert_eq(&format!("opaque_xor_{}", i), result, word_c[i]);
    }

    let circuit = builder.build();

    let stat = binius_frontend::CircuitStat::collect(&circuit);
    println!(
        "  AND constraints: {} (structural assert_eq only — XOR is FREE)",
        stat.n_and_constraints
    );
    println!("  XOR itself is linear; structural equality constraints are still counted.");

    // Generate opaque witness words
    let mut witness = circuit.new_witness_filler();

    // Use opaque deterministic fixture words. This example intentionally does not model OTP.
    let a_values = crate::bench_inputs::deterministic_words(
        crate::bench_inputs::SEED_TRIPLE_XOR_A,
        dim_words,
    );
    let b_values = crate::bench_inputs::deterministic_words(
        crate::bench_inputs::SEED_TRIPLE_XOR_B,
        dim_words,
    );
    for i in 0..dim_words {
        let a = a_values[i];
        let b = b_values[i];
        let c = a ^ b;

        witness[word_a[i]] = Word(a);
        witness[word_b[i]] = Word(b);
        witness[word_c[i]] = Word(c);
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
    println!("  Verification: PASSED (integrity proof; no Binius64 ZK privacy guarantee)");

    println!("\n  XOR RELATION VERIFIED (COMPUTATIONAL INTEGRITY ONLY):");
    println!("  ✓ Public word_c equals word_a XOR word_b");
    println!("  ✗ No OTP encryption or mask-generation relation is constrained");
    println!("  ✗ This Binius64 backend provides no zero-knowledge input-privacy guarantee");

    OpaqueXorBenchResult {
        dim_words,
        and_constraints: stat.n_and_constraints,
        prove_time_ms: prove_time.as_secs_f64() * 1000.0,
        verify_time_ms: verify_time.as_secs_f64() * 1000.0,
        proof_size_bytes: proof_size,
    }
}
