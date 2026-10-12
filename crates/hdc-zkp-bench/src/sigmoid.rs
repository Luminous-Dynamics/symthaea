// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Legacy module path for a packed-bit recurrence with a low-byte gate mask.
//!
//! Despite the historical `sigmoid` name, this code implements no sigmoid,
//! fixed-point activation, clamp, comparison, or range check. It masks an
//! unconstrained private u64 witness with 0xFF and uses that result as a
//! bitwise gate word in `h_next = h XOR (gate AND (target XOR h))`.
//! Only the lowest eight bit lanes can switch; the other lanes retain their
//! prior state. This is not numerical CfC dynamics.

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

pub struct MaskedGateBenchResult {
    pub neurons: usize,
    pub and_constraints: usize,
    pub prove_time_ms: f64,
    pub verify_time_ms: f64,
    pub proof_size_bytes: usize,
}

/// Demonstrate a low-byte-masked gate in a packed-bit recurrence.
///
/// The mask source remains an unconstrained witness word; no numerical
/// activation is computed or proven. The historical name is retained in the
/// file path for compatibility with the benchmark harness.
pub fn bench_masked_gate_recurrence(neurons: usize, timesteps: usize) -> MaskedGateBenchResult {
    println!(
        "\n=== BINIUS: Packed-bit Recurrence with Low-Byte Mask ({} words × {} timesteps) ===\n",
        neurons, timesteps
    );

    let builder = CircuitBuilder::new();

    let mut h_wires = Vec::with_capacity(neurons);
    let mut x_inf_wires = Vec::with_capacity(neurons);
    let mut final_h_wires = Vec::with_capacity(neurons);

    for _ in 0..neurons {
        h_wires.push(builder.add_witness());
        x_inf_wires.push(builder.add_witness());
        final_h_wires.push(builder.add_inout());
    }

    // Each timestep uses an unconstrained witness word as a gate source.
    // Masking with 0xFF leaves only the lowest eight bit lanes enabled.
    // The recurrence is Boolean and does not model numerical activation dynamics.
    let mut current_h: Vec<_> = h_wires.clone();
    let mut gate_input_wires = Vec::new();

    // Low-byte mask; it is not an activation function.
    let gate_mask = builder.add_constant_64(0xFF); // Enables only the lowest eight bit lanes

    for _t in 0..timesteps {
        let mut new_h = Vec::with_capacity(neurons);
        for n in 0..neurons {
            let gate_input = builder.add_witness(); // unconstrained witness word
            gate_input_wires.push(gate_input);

            // Gate mask: retain the lowest eight bits of the witness word (1 AND).
            let gate = builder.band(gate_input, gate_mask);

            // Bitwise gated update: h_new = h XOR (gate AND (x_inf XOR h)).
            let delta = builder.bxor(x_inf_wires[n], current_h[n]); // FREE
            let product = builder.band(gate, delta); // 1 AND
            let h_new = builder.bxor(current_h[n], product); // FREE

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
        "  AND constraints: {} (gate masking + recurrence update)",
        stat.n_and_constraints
    );
    println!("  MUL constraints: {}", stat.n_mul_constraints);
    println!(
        "  Expected: ~{} AND (2 per word per step: mask + update)",
        2 * neurons * timesteps
    );
    println!("  XOR operations: {} (all FREE)", neurons * timesteps * 2);

    // Fill witness with deterministic synthetic fixtures.
    let mut witness = circuit.new_witness_filler();
    let h0_vals = crate::bench_inputs::deterministic_words(
        crate::bench_inputs::SEED_MASK_INITIAL_STATE,
        neurons,
    );
    let x_inf_vals =
        crate::bench_inputs::deterministic_words(crate::bench_inputs::SEED_MASK_INPUT, neurons);

    for n in 0..neurons {
        witness[h_wires[n]] = Word(h0_vals[n]);
        witness[x_inf_wires[n]] = Word(x_inf_vals[n]);
    }

    // Pre-generate synthetic private gate inputs in circuit order.
    let gate_input_values = crate::bench_inputs::deterministic_words(
        crate::bench_inputs::SEED_MASK_GATE,
        timesteps * neurons,
    );
    for (i, &value) in gate_input_values.iter().enumerate() {
        witness[gate_input_wires[i]] = Word(value);
    }

    // Simulate the same low-byte-masked gate recurrence as the circuit.
    let mut h_state = h0_vals.clone();
    for t in 0..timesteps {
        for n in 0..neurons {
            let gate_input = gate_input_values[t * neurons + n];
            let gate = gate_input & 0xFF; // Low-byte mask, not a sigmoid
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

    println!("\n  Masked recurrence: two AND gates per word per timestep (mask + update).");
    println!("  The mask source remains private; no sigmoid or numeric CfC is encoded.");

    MaskedGateBenchResult {
        neurons,
        and_constraints: stat.n_and_constraints,
        prove_time_ms: prove_time.as_secs_f64() * 1000.0,
        verify_time_ms: verify_time.as_secs_f64() * 1000.0,
        proof_size_bytes: proof_size,
    }
}
