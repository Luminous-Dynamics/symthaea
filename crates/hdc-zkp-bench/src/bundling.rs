// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Binius benchmark: HDC majority-vote bundling.
//!
//! This proves constraint integrity only; the vendored Binius64 backend does not
//! guarantee zero-knowledge input privacy.
//!
//! Bundling N vectors: for each bit position, output 1 if majority of
//! N input bits are 1. Requires counting + threshold comparison.
//!
//! In Binius:
//! - Uses bit-sliced binary counters and a strict-majority threshold circuit.
//! - Ties for an even number of inputs resolve to zero.
//! - Gate counts and timings must be regenerated after semantic changes.
//!
//! In prime-field STARKs:
//! - Bit decomposition of each input: O(D × N) constraints
//! - Addition for counting: O(D × N) multiplications
//! - Threshold comparison: O(D × log N) constraints
//! - Total: O(D × N) constraints

use std::time::Instant;

use binius_core::verify::verify_constraints;
use binius_core::word::Word;
use binius_frontend::CircuitBuilder;

use crate::boolean_gadgets::{bitwise_majority, bitwise_majority_reference};
use binius_prover::{
    OptimalPackedB128, Prover, hash::parallel_compression::ParallelCompressionAdaptor,
};
use binius_transcript::{ProverTranscript, VerifierTranscript};
use binius_verifier::{
    Verifier,
    config::StdChallenger,
    hash::{StdCompression, StdDigest},
};

pub struct BundlingBenchResult {
    pub num_vectors: usize,
    pub dim_words: usize,
    pub and_constraints: usize,
    pub mul_constraints: usize,
    pub prove_time_ms: f64,
    pub verify_time_ms: f64,
    pub proof_size_bytes: usize,
}

/// Benchmark a proof of strict per-bit majority for N packed vectors.
///
/// Each output bit is set iff strictly more than half the corresponding input
/// bits are one; ties resolve to zero. This general circuit works for N >= 2.
pub fn bench_bundling(num_vectors: usize, dim_words: usize) -> BundlingBenchResult {
    println!(
        "\n=== BINIUS: HDC Bundling ({} vectors × {} words = {} bits) ===\n",
        num_vectors,
        dim_words,
        dim_words * 64
    );

    let builder = CircuitBuilder::new();

    // Input vectors (witness wires; no input-confidentiality guarantee)
    let inputs: Vec<Vec<_>> = (0..num_vectors)
        .map(|_| {
            (0..dim_words)
                .map(|_| builder.add_witness())
                .collect::<Vec<_>>()
        })
        .collect();

    // Output vector (public)
    let output: Vec<_> = (0..dim_words).map(|_| builder.add_inout()).collect();

    assert!(num_vectors >= 2, "bundling requires at least two input vectors");
    assert!(dim_words > 0, "bundling requires at least one word per vector");

    for w in 0..dim_words {
        let word_inputs: Vec<_> = inputs.iter().map(|v| v[w]).collect();
        let result = bitwise_majority(&builder, &word_inputs);
        builder.assert_eq(&format!("bundle_{}", w), result, output[w]);
    }

    let circuit = builder.build();

    let stat = binius_frontend::CircuitStat::collect(&circuit);
    println!("  AND constraints: {}", stat.n_and_constraints);
    println!("  MUL constraints: {}", stat.n_mul_constraints);

    println!("  Aggregation: strict bitwise majority; ties resolve to 0.");

    // Fill witness
    let mut witness = circuit.new_witness_filler();

    // Deterministic synthetic vectors; each participant gets a domain-separated seed.
    let mut input_vals: Vec<Vec<u64>> = Vec::new();
    for v in 0..num_vectors {
        let vals = crate::bench_inputs::deterministic_words(
            crate::bench_inputs::SEED_BUNDLING_BASE.wrapping_add(v as u64),
            dim_words,
        );
        for w in 0..dim_words {
            witness[inputs[v][w]] = Word(vals[w]);
        }
        input_vals.push(vals);
    }

    for w in 0..dim_words {
        let words_at_position: Vec<u64> = input_vals.iter().map(|v| v[w]).collect();
        let result = bitwise_majority_reference(&words_at_position);
        witness[output[w]] = Word(result);
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

    // Verify
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

    BundlingBenchResult {
        num_vectors: num_vectors,
        dim_words: dim_words,
        and_constraints: stat.n_and_constraints,
        mul_constraints: stat.n_mul_constraints,
        prove_time_ms: prove_time.as_secs_f64() * 1000.0,
        verify_time_ms: verify_time.as_secs_f64() * 1000.0,
        proof_size_bytes: proof_size,
    }
}
