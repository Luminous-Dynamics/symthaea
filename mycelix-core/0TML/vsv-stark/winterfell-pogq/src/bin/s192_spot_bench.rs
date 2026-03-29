// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Quick S192 spot benchmark - measure 192-bit security overhead
//!
//! Runs a single scenario with S192 profile to measure actual performance vs S128

use winterfell_pogq::{PoGQProver, AirPublicInputs as PublicInputs, SecurityProfile, AIR_SCHEMA_REV};
use std::time::Instant;

fn q(val: f32) -> u64 {
    (val * 65536.0) as u64
}

fn main() {
    println!("🔬 VSV-STARK S192 Spot Benchmark");
    println!("================================================");
    println!("Profile: S192 (120 queries, ≥192-bit security)");
    println!("Scenario: Normal operation (T=32)");
    println!("Rounds: 10");
    println!("================================================\n");

    let public = PublicInputs {
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
        profile_id: 192,  // S192
        air_rev: AIR_SCHEMA_REV,
    };
    let witness = vec![q(0.930); 32];

    // Create prover with S192 profile
    let prover = PoGQProver::with_profile(SecurityProfile::S192);

    let mut prove_times = Vec::new();
    let mut verify_times = Vec::new();
    let mut proof_sizes = Vec::new();

    for i in 0..10 {
        print!("Round {}/10... ", i + 1);

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

        println!("✓ {:.2}ms prove, {:.2}ms verify", prove_times.last().unwrap(), verify_times.last().unwrap());
    }

    // Calculate statistics
    let avg_prove = prove_times.iter().sum::<f64>() / prove_times.len() as f64;
    let avg_verify = verify_times.iter().sum::<f64>() / verify_times.len() as f64;
    let avg_size = proof_sizes.iter().sum::<usize>() / proof_sizes.len();

    println!("\n================================================");
    println!("📊 S192 Results:");
    println!("  Prove:  {:.2} ms (avg)", avg_prove);
    println!("  Verify: {:.2} ms (avg)", avg_verify);
    println!("  Size:   {} bytes ({:.1} KB)", avg_size, avg_size as f64 / 1024.0);

    println!("\n📊 Comparison with S128 (from LEAN_BENCHMARKS_127BIT_FINAL.txt):");
    println!("  S128 Prove:  12.52 ms → S192: {:.2} ms ({:.1}× overhead)", avg_prove, avg_prove / 12.52);
    println!("  S128 Verify: 1.72 ms  → S192: {:.2} ms ({:.1}× overhead)", avg_verify, avg_verify / 1.72);
    println!("  S128 Size:   59.8 KB  → S192: {:.1} KB ({:.1}× overhead)", avg_size as f64 / 1024.0, (avg_size as f64 / 1024.0) / 59.8);
    println!("================================================");
}
