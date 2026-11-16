// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Legacy-path demonstration of a bitwise-majority proof over opaque words.
//!
//! The circuit constrains only that a public packed-word output is strict per-bit
//! majority of witness input words. Binius64 gives no ZK input-privacy guarantee.
//! It does NOT prove one-time-pad
//! encryption, participant identity, numeric gradient aggregation, or an
//! end-to-end federated-learning protocol.

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

use crate::boolean_gadgets::{bitwise_majority, bitwise_majority_reference};

pub struct BitwiseMajorityBenchResult {
    pub participants: usize,
    pub dim_words: usize,
    pub and_constraints: usize,
    pub prove_time_ms: f64,
    pub verify_time_ms: f64,
    pub proof_size_bytes: usize,
}

/// Prove strict bitwise majority over N witness inputs; input confidentiality is not guaranteed.
pub fn bench_bitwise_majority(
    participants: usize,
    dim_words: usize,
) -> BitwiseMajorityBenchResult {
    assert!(participants >= 2, "majority requires at least two inputs");
    assert!(dim_words > 0, "majority input must contain at least one word");
    let dim_bits = dim_words.checked_mul(64).expect("dimension overflow");

    println!(
        "\n=== BINIUS: Bitwise Majority ({} inputs × {} bits) ===\n",
        participants, dim_bits
    );

    let builder = CircuitBuilder::new();
    let inputs: Vec<Vec<_>> = (0..participants)
        .map(|_| (0..dim_words).map(|_| builder.add_witness()).collect())
        .collect();
    let output: Vec<_> = (0..dim_words).map(|_| builder.add_inout()).collect();

    for w in 0..dim_words {
        let word_inputs: Vec<_> = inputs.iter().map(|v| v[w]).collect();
        let result = bitwise_majority(&builder, &word_inputs);
        builder.assert_eq(&format!("majority_{}", w), result, output[w]);
    }

    let circuit = builder.build();
    let stat = binius_frontend::CircuitStat::collect(&circuit);
    println!("  AND constraints: {}", stat.n_and_constraints);
    println!("  MUL constraints: {}", stat.n_mul_constraints);
    println!("  Tie policy: output bit is 1 iff more than half the inputs have it set.");

    let mut witness = circuit.new_witness_filler();
    let mut input_values: Vec<Vec<u64>> = Vec::with_capacity(participants);
    for p in 0..participants {
        let values = crate::bench_inputs::deterministic_words(
            crate::bench_inputs::SEED_FL_TRIPLE_BASE.wrapping_add(p as u64),
            dim_words,
        );
        for w in 0..dim_words {
            witness[inputs[p][w]] = Word(values[w]);
        }
        input_values.push(values);
    }

    for w in 0..dim_words {
        let words_at_position: Vec<u64> = input_values.iter().map(|v| v[w]).collect();
        witness[output[w]] = Word(bitwise_majority_reference(&words_at_position));
    }

    circuit
        .populate_wire_witness(&mut witness)
        .expect("witness population failed");
    let cs = circuit.constraint_system();
    let witness_vec = witness.into_value_vec();
    verify_constraints(cs, &witness_vec).expect("majority constraints failed");
    println!("  Constraint check: PASSED (opaque-word majority relation)");

    let compression = ParallelCompressionAdaptor::new(StdCompression::default());
    let verifier = Verifier::<StdDigest, _>::setup(cs.clone(), 1, StdCompression::default())
        .expect("verifier setup failed");
    let prover = Prover::<OptimalPackedB128, _, StdDigest>::setup(verifier.clone(), compression)
        .expect("prover setup failed");

    let challenger = StdChallenger::default();
    let public_words = witness_vec.public().to_vec();

    let prove_start = Instant::now();
    let mut pt = ProverTranscript::new(challenger.clone());
    prover.prove(witness_vec, &mut pt).expect("proving failed");
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
    verifier.verify(&public_words, &mut vt).expect("verification failed");
    vt.finalize().expect("transcript finalization failed");
    let verify_time = verify_start.elapsed();

    println!(
        "  Verifier time: {:.3} ms",
        verify_time.as_secs_f64() * 1000.0
    );
    println!("  Verification: PASSED (integrity relation; no Binius64 ZK privacy guarantee)");

    BitwiseMajorityBenchResult {
        participants,
        dim_words,
        and_constraints: stat.n_and_constraints,
        prove_time_ms: prove_time.as_secs_f64() * 1000.0,
        verify_time_ms: verify_time.as_secs_f64() * 1000.0,
        proof_size_bytes: proof_size,
    }
}
