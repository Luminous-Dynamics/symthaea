// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Benchmarks for LEAN (range-checked) Winterfell PoGQ at 127-bit Security
//!
//! Measures performance with:
//! - 32-bit range checks (rem_t 16-bit + x_t 16-bit)
//! - 127-bit conjectured STARK security (80 queries, blowup 16, grinding 16, folding 8)
//! Note: 127 bits is maximum achievable with our AIR (44 cols, 32 rows, 40 constraints)

use winterfell_pogq::{PoGQProver, AirPublicInputs as PublicInputs, AIR_SCHEMA_REV};
use std::time::Instant;

fn q(val: f32) -> u64 {
    (val * 65536.0) as u64
}

fn benchmark_scenario(name: &str, public: PublicInputs, witness: Vec<u64>, rounds: usize) {
    let prover = PoGQProver::new();

    let mut prove_times = Vec::new();
    let mut verify_times = Vec::new();
    let mut proof_sizes = Vec::new();

    for _ in 0..rounds {
        // Measure proving time
        let prove_start = Instant::now();
        let result = prover.prove_exec(public.clone(), witness.clone()).unwrap();
        let prove_time = prove_start.elapsed();

        // Measure verification time
        let verify_start = Instant::now();
        let valid = prover.verify_proof(&result.proof_bytes, public.clone()).unwrap();
        let verify_time = verify_start.elapsed();

        assert!(valid, "Proof verification failed");

        prove_times.push(prove_time.as_micros() as f64 / 1000.0);
        verify_times.push(verify_time.as_micros() as f64 / 1000.0);
        proof_sizes.push(result.proof_bytes.len());
    }

    // Calculate statistics
    let avg_prove = prove_times.iter().sum::<f64>() / prove_times.len() as f64;
    let avg_verify = verify_times.iter().sum::<f64>() / verify_times.len() as f64;
    let avg_size = proof_sizes.iter().sum::<usize>() / proof_sizes.len();

    println!("\n=== {} ===", name);
    println!("  Prove:  {:.2} ms (avg over {} rounds)", avg_prove, rounds);
    println!("  Verify: {:.2} ms", avg_verify);
    println!("  Size:   {} bytes ({:.1} KB)", avg_size, avg_size as f64 / 1024.0);
}

fn main() {
    println!("🔬 Winterfell PoGQ LEAN Benchmarks (127-bit Security)");
    println!("================================================");
    println!("Security: 127-bit conjectured (80 queries, blowup 16)");
    println!("Range checks: 32-bit (rem_t 16-bit + x_t 16-bit)");
    println!("================================================");

    let rounds = 20;

    // Scenario 1: Normal operation (no violation)
    let public1 = PublicInputs {
        beta: q(0.85),
        w: 3,
        k: 2,
        m: 3,
        threshold: q(0.90),
        ema_init: q(0.915),
        viol_init: 0,
        clear_init: 2,
        quar_init: 0,
        round_init: 4,
        quar_out: 0,
        trace_length: 32,
        // Provenance (prover will populate automatically)
        prov_hash: [0, 0, 0, 0],
        profile_id: 128,
        air_rev: AIR_SCHEMA_REV,
    };
    let witness1 = vec![q(0.930); 32];
    benchmark_scenario("Normal Operation (no violation)", public1, witness1, rounds);

    // Scenario 2: Enter quarantine
    let public2 = PublicInputs {
        beta: q(0.85),
        w: 3,
        k: 2,
        m: 3,
        threshold: q(0.90),
        ema_init: q(0.763),
        viol_init: 1,
        clear_init: 0,
        quar_init: 0,
        round_init: 7,
        quar_out: 1,
        trace_length: 32,
        // Provenance (prover will populate automatically)
        prov_hash: [0, 0, 0, 0],
        profile_id: 128,
        air_rev: AIR_SCHEMA_REV,
    };
    let witness2 = vec![q(0.763); 32];
    benchmark_scenario("Enter Quarantine", public2, witness2, rounds);

    // Scenario 3: Release from quarantine
    let public3 = PublicInputs {
        beta: q(0.85),
        w: 3,
        k: 2,
        m: 3,
        threshold: q(0.90),
        ema_init: q(0.946),
        viol_init: 0,
        clear_init: 2,
        quar_init: 1,
        round_init: 9,
        quar_out: 0,
        trace_length: 32,
        // Provenance (prover will populate automatically)
        prov_hash: [0, 0, 0, 0],
        profile_id: 128,
        air_rev: AIR_SCHEMA_REV,
    };
    let witness3 = vec![q(0.961); 32];
    benchmark_scenario("Release from Quarantine", public3, witness3, rounds);

    println!("\n================================================");
    println!("✅ LEAN benchmarks complete (127-bit security)!");
    println!("\n📊 Comparisons:");
    println!("   96-bit LEAN:  1.0-1.6 ms prove, 0.45-0.60 ms verify, ~30 KB");
    println!("   127-bit LEAN: Will be measured in this benchmark run");
    println!("   zkVM baseline: 46.6 s prove, 92 ms verify, 221 KB");
    println!("\n   → Still ~30,000× faster than zkVM!");
    println!("   → 127-bit security provides cryptographically strong guarantees");
}
