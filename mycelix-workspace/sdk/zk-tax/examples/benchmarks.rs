// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Benchmarks: Performance Measurement for All Proof Types
//!
//! Measures proof generation and verification times for:
//! - Tax bracket proofs
//! - Income range proofs
//! - Multi-year batch proofs
//! - Effective tax rate proofs
//! - Cross-jurisdiction proofs
//!
//! Run with: cargo run --example benchmarks --release

use std::time::{Duration, Instant};
use mycelix_zk_tax::{FilingStatus, Jurisdiction, TaxBracketProver};
use mycelix_zk_tax::proof::{
    IncomeRangeProof, RangeProofBuilder, BatchProofBuilder,
    EffectiveTaxRateProof, CrossJurisdictionProofBuilder,
};

fn main() {
    println!("=== ZK Tax Proof Performance Benchmarks ===");
    println!("(Running in {} mode)\n", if cfg!(debug_assertions) { "DEBUG" } else { "RELEASE" });

    let iterations = 1000;

    // =========================================================================
    // Benchmark 1: Tax Bracket Proof (Dev Mode)
    // =========================================================================
    println!("--- 1. Tax Bracket Proof ---");

    let prover = TaxBracketProver::dev_mode();
    let (avg, min, max) = benchmark(iterations, || {
        prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, 2024)
            .expect("proof failed")
    });

    println!("  Dev mode (commitment only):");
    println!("    Iterations: {}", iterations);
    println!("    Average: {:?}", avg);
    println!("    Min: {:?} | Max: {:?}", min, max);
    println!("    Throughput: {:.0} proofs/sec", 1_000_000.0 / avg.as_micros() as f64);

    // =========================================================================
    // Benchmark 2: Income Range Proof
    // =========================================================================
    println!("\n--- 2. Income Range Proof ---");

    let (avg, min, max) = benchmark(iterations, || {
        IncomeRangeProof::prove_dev(85_000, 50_000, 150_000, 2024)
            .expect("proof failed")
    });

    println!("  Range proof (prove income within bounds):");
    println!("    Average: {:?}", avg);
    println!("    Min: {:?} | Max: {:?}", min, max);
    println!("    Throughput: {:.0} proofs/sec", 1_000_000.0 / avg.as_micros() as f64);

    // Builder pattern
    let (avg_builder, _, _) = benchmark(iterations, || {
        RangeProofBuilder::new(85_000, 2024)
            .prove_above(50_000)
            .expect("proof failed")
    });
    println!("    Builder pattern: {:?}", avg_builder);

    // =========================================================================
    // Benchmark 3: Batch Proof (Multi-Year)
    // =========================================================================
    println!("\n--- 3. Batch Proof (Multi-Year) ---");

    // 3-year batch
    let (avg_3yr, _, _) = benchmark(iterations / 10, || {
        BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)
            .add_year(2022, 75_000)
            .add_year(2023, 80_000)
            .add_year(2024, 85_000)
            .build_dev()
            .expect("proof failed")
    });
    println!("  3-year batch:");
    println!("    Average: {:?}", avg_3yr);

    // 6-year batch (all supported years)
    let (avg_6yr, _, _) = benchmark(iterations / 10, || {
        BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)
            .add_year_range(2020, 2025, 85_000)
            .build_dev()
            .expect("proof failed")
    });
    println!("  6-year batch:");
    println!("    Average: {:?}", avg_6yr);
    println!("    Cost per year: {:?}", avg_6yr / 6);

    // =========================================================================
    // Benchmark 4: Effective Tax Rate Proof
    // =========================================================================
    println!("\n--- 4. Effective Tax Rate Proof ---");

    let (avg, min, max) = benchmark(iterations, || {
        EffectiveTaxRateProof::prove_dev(
            85_000,
            Jurisdiction::US,
            FilingStatus::Single,
            2024,
        ).expect("proof failed")
    });

    println!("  US effective rate calculation:");
    println!("    Average: {:?}", avg);
    println!("    Min: {:?} | Max: {:?}", min, max);
    println!("    Throughput: {:.0} proofs/sec", 1_000_000.0 / avg.as_micros() as f64);

    // High income (more brackets to calculate)
    let (avg_high, _, _) = benchmark(iterations, || {
        EffectiveTaxRateProof::prove_dev(
            500_000,
            Jurisdiction::US,
            FilingStatus::Single,
            2024,
        ).expect("proof failed")
    });
    println!("  High income ($500K):");
    println!("    Average: {:?}", avg_high);

    // =========================================================================
    // Benchmark 5: Cross-Jurisdiction Proof
    // =========================================================================
    println!("\n--- 5. Cross-Jurisdiction Proof ---");

    // 2 jurisdictions
    let (avg_2j, _, _) = benchmark(iterations / 10, || {
        CrossJurisdictionProofBuilder::new(100_000, 2024)
            .add_jurisdiction(Jurisdiction::US, FilingStatus::Single)
            .add_jurisdiction(Jurisdiction::UK, FilingStatus::Single)
            .build_dev()
            .expect("proof failed")
    });
    println!("  2 jurisdictions (US, UK):");
    println!("    Average: {:?}", avg_2j);

    // 7 jurisdictions (OECD common)
    let (avg_7j, _, _) = benchmark(iterations / 10, || {
        CrossJurisdictionProofBuilder::new(100_000, 2024)
            .add_oecd_common(FilingStatus::Single)
            .build_dev()
            .expect("proof failed")
    });
    println!("  7 jurisdictions (OECD common):");
    println!("    Average: {:?}", avg_7j);
    println!("    Cost per jurisdiction: {:?}", avg_7j / 7);

    // All G20 (19 jurisdictions)
    let (avg_all, _, _) = benchmark(iterations / 10, || {
        CrossJurisdictionProofBuilder::new(100_000, 2024)
            .add_jurisdictions(&[
                Jurisdiction::US, Jurisdiction::CA, Jurisdiction::MX,
                Jurisdiction::BR, Jurisdiction::AR, Jurisdiction::UK,
                Jurisdiction::DE, Jurisdiction::FR, Jurisdiction::IT,
                Jurisdiction::RU, Jurisdiction::TR, Jurisdiction::JP,
                Jurisdiction::CN, Jurisdiction::IN, Jurisdiction::KR,
                Jurisdiction::ID, Jurisdiction::AU, Jurisdiction::SA,
                Jurisdiction::ZA,
            ], FilingStatus::Single)
            .build_dev()
            .expect("proof failed")
    });
    println!("  All 19 G20 jurisdictions:");
    println!("    Average: {:?}", avg_all);
    println!("    Cost per jurisdiction: {:?}", avg_all / 19);

    // =========================================================================
    // Benchmark 6: Verification Times
    // =========================================================================
    println!("\n--- 6. Verification Times ---");

    // Create proofs once
    let bracket_proof = prover.prove(85_000, Jurisdiction::US, FilingStatus::Single, 2024)
        .expect("proof failed");
    let range_proof = IncomeRangeProof::prove_dev(85_000, 50_000, 150_000, 2024)
        .expect("proof failed");
    let batch_proof = BatchProofBuilder::new(Jurisdiction::US, FilingStatus::Single)
        .add_year_range(2022, 2024, 85_000)
        .build_dev()
        .expect("proof failed");
    let effective_proof = EffectiveTaxRateProof::prove_dev(85_000, Jurisdiction::US, FilingStatus::Single, 2024)
        .expect("proof failed");
    let cross_proof = CrossJurisdictionProofBuilder::new(100_000, 2024)
        .add_oecd_common(FilingStatus::Single)
        .build_dev()
        .expect("proof failed");

    let (v_bracket, _, _) = benchmark(iterations, || bracket_proof.verify());
    let (v_range, _, _) = benchmark(iterations, || range_proof.verify());
    let (v_batch, _, _) = benchmark(iterations, || batch_proof.verify());
    let (v_effective, _, _) = benchmark(iterations, || effective_proof.verify());
    let (v_cross, _, _) = benchmark(iterations, || cross_proof.verify());

    println!("  Bracket proof verify: {:?}", v_bracket);
    println!("  Range proof verify: {:?}", v_range);
    println!("  Batch proof verify (3yr): {:?}", v_batch);
    println!("  Effective rate verify: {:?}", v_effective);
    println!("  Cross-jurisdiction verify (7): {:?}", v_cross);

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n=== Summary ===");
    println!("All benchmarks use DEV MODE (commitment-only, no Risc0 zkVM)");
    println!("Production ZK proofs (~40 seconds) would be much slower");
    println!();
    println!("Dev mode is suitable for:");
    println!("  - Testing and development");
    println!("  - Non-adversarial use cases");
    println!("  - Integration testing");
    println!();
    println!("For production (adversarial scenarios), enable:");
    println!("  cargo run --example benchmarks --release --features prover");

    println!("\n Done!");
}

/// Run a benchmark and return (average, min, max) durations
fn benchmark<T, F: FnMut() -> T>(iterations: usize, mut f: F) -> (Duration, Duration, Duration) {
    let mut times = Vec::with_capacity(iterations);

    // Warmup
    for _ in 0..10 {
        let _ = f();
    }

    // Actual benchmark
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = f();
        times.push(start.elapsed());
    }

    let min = *times.iter().min().unwrap();
    let max = *times.iter().max().unwrap();
    let total: Duration = times.iter().sum();
    let avg = total / iterations as u32;

    (avg, min, max)
}
