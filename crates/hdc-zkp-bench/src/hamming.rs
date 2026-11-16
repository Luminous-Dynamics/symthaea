// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Binius proof benchmark for Hamming distance of packed binary hypervectors.
//!
//! The circuit proves distance(a,b) = sum(popcount(a[i] XOR b[i])).
//! XOR computes each word difference; the constrained SWAR popcount counts its
//! set bits; a constrained integer sum produces the public distance output.

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

use crate::boolean_gadgets::popcount_u64;

pub struct HammingBenchResult {
    pub dim_words: usize,
    pub and_constraints: usize,
    pub prove_time_ms: f64,
    pub verify_time_ms: f64,
    pub proof_size_bytes: usize,
}

/// Benchmark an integrity proof of Hamming distance; witness privacy is not guaranteed.
///
/// Witness inputs are two vectors of dim_words 64-bit words. The Binius64 backend does not
/// provide zero-knowledge input privacy. The public output
/// is the exact number of differing bits. No normalized similarity score is
/// claimed by this circuit.
pub fn bench_hamming(dim_words: usize) -> HammingBenchResult {
    assert!(dim_words > 0, "Hamming vectors must contain at least one word");
    let dim_bits = dim_words
        .checked_mul(64)
        .expect("Hamming vector dimension overflow");
    assert!(
        dim_bits <= u32::MAX as usize,
        "Hamming distance must fit in the 32-bit accumulator"
    );

    println!(
        "\n=== BINIUS: Hamming Distance Proof ({} bits = {} words) ===\n",
        dim_bits, dim_words
    );

    let builder = CircuitBuilder::new();
    let vec_a: Vec<_> = (0..dim_words).map(|_| builder.add_witness()).collect();
    let vec_b: Vec<_> = (0..dim_words).map(|_| builder.add_witness()).collect();
    let distance_out = builder.add_inout();

    let xor_words: Vec<_> = (0..dim_words)
        .map(|i| builder.bxor(vec_a[i], vec_b[i]))
        .collect();
    let popcounts: Vec<_> = xor_words
        .iter()
        .map(|&word| popcount_u64(&builder, word))
        .collect();

    let mut total_distance = popcounts[0];
    for &count in popcounts.iter().skip(1) {
        // The total distance is bounded by dim_words * 64 <= u32::MAX.
        total_distance = builder.iadd_32(total_distance, count);
    }
    builder.assert_eq("hamming_distance", total_distance, distance_out);

    let circuit = builder.build();
    let stat = binius_frontend::CircuitStat::collect(&circuit);
    println!("  AND constraints: {}", stat.n_and_constraints);
    println!("  MUL constraints: {}", stat.n_mul_constraints);
    println!("  Word XOR operations: {} (linear gates)", dim_words);

    let mut witness = circuit.new_witness_filler();
    let a_values = crate::bench_inputs::deterministic_words(
        crate::bench_inputs::SEED_HAMMING_A,
        dim_words,
    );
    let b_values = crate::bench_inputs::deterministic_words(
        crate::bench_inputs::SEED_HAMMING_B,
        dim_words,
    );
    for i in 0..dim_words {
        witness[vec_a[i]] = Word(a_values[i]);
        witness[vec_b[i]] = Word(b_values[i]);
    }
    let expected_distance: u64 = a_values
        .iter()
        .zip(&b_values)
        .map(|(a, b)| (a ^ b).count_ones() as u64)
        .sum();
    witness[distance_out] = Word(expected_distance);

    circuit
        .populate_wire_witness(&mut witness)
        .expect("witness population failed");
    let cs = circuit.constraint_system();
    let witness_vec = witness.into_value_vec();
    verify_constraints(cs, &witness_vec).expect("Hamming-distance constraints failed");
    println!("  Constraint check: PASSED (reference distance {expected_distance})");

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
    println!("  Verification: PASSED (integrity proof; no Binius64 ZK privacy guarantee)");

    HammingBenchResult {
        dim_words,
        and_constraints: stat.n_and_constraints,
        prove_time_ms: prove_time.as_secs_f64() * 1000.0,
        verify_time_ms: verify_time.as_secs_f64() * 1000.0,
        proof_size_bytes: proof_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_distance_sums_popcounts_instead_of_word_values() {
        let builder = CircuitBuilder::new();
        let a0 = builder.add_witness();
        let a1 = builder.add_witness();
        let b0 = builder.add_witness();
        let b1 = builder.add_witness();
        let output = builder.add_inout();

        let xor0 = builder.bxor(a0, b0);
        let xor1 = builder.bxor(a1, b1);
        let count0 = popcount_u64(&builder, xor0);
        let count1 = popcount_u64(&builder, xor1);
        let total = builder.iadd_32(count0, count1);
        builder.assert_eq("hamming_distance", total, output);

        let circuit = builder.build();
        let mut witness = circuit.new_witness_filler();
        witness[a0] = Word(0);
        witness[a1] = Word(u64::MAX);
        witness[b0] = Word(1);
        witness[b1] = Word(0);
        witness[output] = Word(65);

        circuit
            .populate_wire_witness(&mut witness)
            .expect("valid Hamming-distance witness");
        assert_eq!(witness[total], Word(65));
        let cs = circuit.constraint_system();
        verify_constraints(cs, &witness.into_value_vec())
            .expect("Hamming-distance reference relation must verify");
    }
}
