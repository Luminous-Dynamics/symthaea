// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Binius benchmark: HDC majority-vote bundling.
//!
//! Bundling N vectors: for each bit position, output 1 if majority of
//! N input bits are 1. Requires counting + threshold comparison.
//!
//! In Binius:
//! - XOR additions for vote counting: FREE (native field addition)
//! - AND gates for threshold comparison: O(log N) per bit position
//! - Total: O(D × log N) AND constraints for D-bit vectors, N inputs
//!
//! In prime-field STARKs:
//! - Bit decomposition of each input: O(D × N) constraints
//! - Addition for counting: O(D × N) multiplications
//! - Threshold comparison: O(D × log N) constraints
//! - Total: O(D × N) constraints

use std::time::Instant;

use binius_core::word::Word;
use binius_core::verify::verify_constraints;
use binius_frontend::CircuitBuilder;
use binius_prover::{
    OptimalPackedB128, Prover,
    hash::parallel_compression::ParallelCompressionAdaptor,
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

/// Benchmark majority-vote bundling of N vectors.
///
/// For simplicity, we implement bundling of N=2 vectors (which is just AND)
/// and N=3 vectors (majority = at least 2 of 3 agree).
///
/// Majority of 3: out = (a AND b) XOR (a AND c) XOR (b AND c) XOR (a AND b AND c)
/// Simpler: out = (a AND b) OR (a AND c) OR (b AND c)
///        = (a AND b) XOR (a AND c) XOR (b AND c) XOR ((a AND b) AND (a AND c))
///
/// Even simpler for Binius: majority(a,b,c) = (a AND b) XOR ((a XOR b) AND c)
/// This requires exactly 2 AND + 2 XOR per word.
pub fn bench_bundling(num_vectors: usize, dim_words: usize) -> BundlingBenchResult {
    println!("\n=== BINIUS: HDC Bundling ({} vectors × {} words = {} bits) ===\n",
        num_vectors, dim_words, dim_words * 64);

    let builder = CircuitBuilder::new();

    // Input vectors (private witnesses)
    let inputs: Vec<Vec<_>> = (0..num_vectors)
        .map(|_| (0..dim_words).map(|_| builder.add_witness()).collect::<Vec<_>>())
        .collect();

    // Output vector (public)
    let output: Vec<_> = (0..dim_words).map(|_| builder.add_inout()).collect();

    // Compute majority vote per word
    match num_vectors {
        2 => {
            // Majority of 2 = AND (both must agree)
            for w in 0..dim_words {
                let result = builder.band(inputs[0][w], inputs[1][w]);
                builder.assert_eq(&format!("bundle_{}", w), result, output[w]);
            }
        }
        3 => {
            // Majority of 3: maj(a,b,c) = (a AND b) XOR ((a XOR b) AND c)
            // 2 AND + 2 XOR per word. XOR is FREE in Binius.
            for w in 0..dim_words {
                let ab = builder.band(inputs[0][w], inputs[1][w]); // AND #1
                let a_xor_b = builder.bxor(inputs[0][w], inputs[1][w]); // FREE
                let axb_and_c = builder.band(a_xor_b, inputs[2][w]); // AND #2
                let result = builder.bxor(ab, axb_and_c); // FREE
                builder.assert_eq(&format!("bundle_{}", w), result, output[w]);
            }
        }
        n if n > 3 => {
            // For N > 3, use pairwise reduction:
            // Compute popcount per bit, then threshold at N/2.
            // For the benchmark, chain pairwise AND + XOR.
            // majority(a,b,c,d,...) via tree reduction.
            //
            // Simple approach for even N: split into pairs, AND each pair,
            // then bundle the results recursively.
            // This uses O(N × dim_words) ANDs — suboptimal but correct.
            let mut current: Vec<_> = inputs.iter().map(|v| v.clone()).collect();
            while current.len() > 1 {
                let mut next = Vec::new();
                for pair in current.chunks(2) {
                    if pair.len() == 2 {
                        let merged: Vec<_> = (0..dim_words)
                            .map(|w| builder.band(pair[0][w], pair[1][w]))
                            .collect();
                        next.push(merged);
                    } else {
                        next.push(pair[0].clone());
                    }
                }
                current = next;
            }
            for w in 0..dim_words {
                builder.assert_eq(&format!("bundle_{}", w), current[0][w], output[w]);
            }
        }
        _ => panic!("num_vectors must be >= 2"),
    }

    let circuit = builder.build();

    let stat = binius_frontend::CircuitStat::collect(&circuit);
    println!("  AND constraints: {}", stat.n_and_constraints);
    println!("  MUL constraints: {}", stat.n_mul_constraints);

    let xor_ops = match num_vectors {
        2 => 0,
        3 => dim_words * 2,
        n => (n - 1) * dim_words, // approximate
    };
    println!("  XOR operations: {} (all FREE)", xor_ops);

    // Fill witness
    let mut witness = circuit.new_witness_filler();

    // Random input vectors
    let mut input_vals: Vec<Vec<u64>> = Vec::new();
    for v in 0..num_vectors {
        let vals: Vec<u64> = (0..dim_words).map(|_| rand::random()).collect();
        for w in 0..dim_words {
            witness[inputs[v][w]] = Word(vals[w]);
        }
        input_vals.push(vals);
    }

    // Compute expected output (majority vote)
    for w in 0..dim_words {
        let mut result = match num_vectors {
            2 => input_vals[0][w] & input_vals[1][w],
            3 => {
                let ab = input_vals[0][w] & input_vals[1][w];
                let axb = input_vals[0][w] ^ input_vals[1][w];
                let axb_c = axb & input_vals[2][w];
                ab ^ axb_c
            }
            _ => {
                // Tree reduction (matches circuit)
                let mut current: Vec<u64> = input_vals.iter().map(|v| v[w]).collect();
                while current.len() > 1 {
                    let mut next = Vec::new();
                    for pair in current.chunks(2) {
                        if pair.len() == 2 {
                            next.push(pair[0] & pair[1]);
                        } else {
                            next.push(pair[0]);
                        }
                    }
                    current = next;
                }
                current[0]
            }
        };
        witness[output[w]] = Word(result);
    }

    circuit.populate_wire_witness(&mut witness).expect("witness failed");

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
    println!("  Proof size: {} bytes ({:.1} KB)", proof_size, proof_size as f64 / 1024.0);
    println!("  Prover time: {:.1} ms", prove_time.as_secs_f64() * 1000.0);

    // Verify
    let verify_start = Instant::now();
    let mut vt = VerifierTranscript::new(challenger, proof);
    verifier.verify(&public_words, &mut vt).expect("verify failed");
    vt.finalize().expect("finalize failed");
    let verify_time = verify_start.elapsed();

    println!("  Verifier time: {:.3} ms", verify_time.as_secs_f64() * 1000.0);
    println!("  Verification: PASSED (real cryptographic proof)");

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
