// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Binius demonstration: gated bitwise state recurrence over packed u64 words.
//!
//! The circuit proves only the recurrence `h_next = h XOR (gate AND (target XOR h))`
//! independently for each bit lane. Each gate is an unconstrained witness
//! word; it is not derived from a sigmoid or a numerical time constant.
//!
//! This is not a proof of real-valued or fixed-point CfC dynamics and establishes
//! no performance advantage over LTC/RK4. Historical function/type names remain
//! for harness compatibility; labels describe the actual Boolean relation.

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

/// Measurements for the packed-bit gated-recurrence demonstration.
pub struct CfcBenchResult {
    pub neurons: usize,
    pub timesteps: usize,
    pub and_constraints: usize,
    pub mul_constraints: usize,
    pub prove_time_ms: f64,
    pub verify_time_ms: f64,
    pub proof_size_bytes: usize,
}

/// Prove a sequence of bitwise gate-controlled updates over packed 64-bit words.
///
/// The private gate word is unconstrained apart from its use in the recurrence.
/// The historical function name is retained, but it does not encode numeric CfC.
pub fn bench_cfc_temporal(neurons: usize, timesteps: usize) -> CfcBenchResult {
    println!(
        "\n=== BINIUS: Packed-bit Gated Recurrence ({} words × {} timesteps) ===\n",
        neurons, timesteps
    );

    let builder = CircuitBuilder::new();

    // Per packed word: initial state h(0), target state x_inf
    let mut h_wires = Vec::with_capacity(neurons);
    let mut x_inf_wires = Vec::with_capacity(neurons);
    let mut final_h_wires = Vec::with_capacity(neurons);
    let mut gate_wires = Vec::new(); // Track all gate witnesses

    for _ in 0..neurons {
        h_wires.push(builder.add_witness()); // h(0) — private
        x_inf_wires.push(builder.add_witness()); // x_inf — private
        final_h_wires.push(builder.add_inout()); // h(T) — public
    }

    // Build the gated bitwise recurrence for each timestep and packed word
    let mut current_h: Vec<_> = h_wires.clone();

    for _t in 0..timesteps {
        let mut new_h = Vec::with_capacity(neurons);
        for n in 0..neurons {
            let gate = builder.add_witness(); // Unconstrained private gate word
            gate_wires.push(gate);

            // delta = x_inf XOR h_old (FREE in Binius)
            let delta = builder.bxor(x_inf_wires[n], current_h[n]);
            // product = gate AND delta (1 AND constraint)
            let product = builder.band(gate, delta);
            // h_new = h_old XOR product (FREE in Binius)
            let h_new = builder.bxor(current_h[n], product);

            new_h.push(h_new);
        }
        current_h = new_h;
    }

    for n in 0..neurons {
        builder.assert_eq(&format!("final_h_{}", n), current_h[n], final_h_wires[n]);
    }

    let circuit = builder.build();

    let stat = binius_frontend::CircuitStat::collect(&circuit);
    println!(
        "  AND constraints: {} (1 per packed word per timestep)",
        stat.n_and_constraints
    );
    println!("  MUL constraints: {}", stat.n_mul_constraints);
    println!(
        "  Expected: {} AND ({} words × {} timesteps)",
        neurons * timesteps,
        neurons,
        timesteps
    );
    println!("  XOR operations: {} (all FREE)", neurons * timesteps * 2);

    // Fill witness: simulate bitwise recurrence evolution matching circuit construction order
    // Circuit builds: timestep-major, with one private gate word per packed word
    // Witness must fill gate_wires in the SAME order
    let mut witness = circuit.new_witness_filler();

    // Generate deterministic synthetic initial values, independent by role.
    let h0_vals = crate::bench_inputs::deterministic_words(
        crate::bench_inputs::SEED_CFC_INITIAL_STATE,
        neurons,
    );
    let x_inf_vals =
        crate::bench_inputs::deterministic_words(crate::bench_inputs::SEED_CFC_INPUT, neurons);

    for n in 0..neurons {
        witness[h_wires[n]] = Word(h0_vals[n]);
        witness[x_inf_wires[n]] = Word(x_inf_vals[n]);
    }

    // Pre-generate deterministic gate values in timestep-major order (matching circuit).
    let gate_vals = crate::bench_inputs::deterministic_words(
        crate::bench_inputs::SEED_CFC_GATE,
        timesteps * neurons,
    );
    for (i, &gate) in gate_vals.iter().enumerate() {
        witness[gate_wires[i]] = Word(gate);
    }

    // Simulate gate-controlled word recurrence to compute final states
    let mut h_state = h0_vals.clone();
    for t in 0..timesteps {
        for n in 0..neurons {
            let gate = gate_vals[t * neurons + n]; // timestep-major indexing
            let delta = x_inf_vals[n] ^ h_state[n];
            let product = gate & delta;
            h_state[n] ^= product;
        }
    }

    for n in 0..neurons {
        witness[final_h_wires[n]] = Word(h_state[n]);
    }

    circuit
        .populate_wire_witness(&mut witness)
        .expect("witness population failed");

    let cs = circuit.constraint_system();
    let witness_vec = witness.into_value_vec();
    verify_constraints(cs, &witness_vec).expect("constraint check failed");
    println!("  Constraint check: PASSED");

    // Generate proof
    let compression = ParallelCompressionAdaptor::new(StdCompression::default());
    let verifier = Verifier::<StdDigest, _>::setup(cs.clone(), 1, StdCompression::default())
        .expect("verifier setup failed");
    let prover = Prover::<OptimalPackedB128, _, StdDigest>::setup(verifier.clone(), compression)
        .expect("prover setup failed");

    let challenger = StdChallenger::default();
    let public_words = witness_vec.public().to_vec();

    let prove_start = Instant::now();
    let mut prover_transcript = ProverTranscript::new(challenger.clone());
    prover
        .prove(witness_vec, &mut prover_transcript)
        .expect("proving failed");
    let proof = prover_transcript.finalize();
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
    let mut verifier_transcript = VerifierTranscript::new(challenger, proof);
    verifier
        .verify(&public_words, &mut verifier_transcript)
        .expect("verification failed");
    verifier_transcript.finalize().expect("finalize failed");
    let verify_time = verify_start.elapsed();

    println!(
        "  Verifier time: {:.3} ms",
        verify_time.as_secs_f64() * 1000.0
    );
    println!("  Verification: PASSED (integrity proof; no Binius64 ZK privacy guarantee)");

    CfcBenchResult {
        neurons,
        timesteps,
        and_constraints: stat.n_and_constraints,
        mul_constraints: stat.n_mul_constraints,
        prove_time_ms: prove_time.as_secs_f64() * 1000.0,
        verify_time_ms: verify_time.as_secs_f64() * 1000.0,
        proof_size_bytes: proof_size,
    }
}
