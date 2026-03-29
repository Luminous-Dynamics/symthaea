// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Benchmarks for K-Vector ZK provers
//!
//! Run with: cargo bench -p kvector-zkp
//! Run specific benchmark: cargo bench -p kvector-zkp -- fast_proof

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use kvector_zkp::{
    optimized_prover::{OptimizedKVectorProver, SecurityLevel},
    prover::{KVectorProver, KVectorTrace, KVectorWitness},
};
use winterfell::Prover;

fn sample_witness() -> KVectorWitness {
    KVectorWitness {
        k_r: 0.8,
        k_a: 0.7,
        k_i: 0.9,
        k_p: 0.6,
        k_m: 0.5,
        k_s: 0.4,
        k_h: 0.85,
        k_topo: 0.75,
    }
}

fn random_witness() -> KVectorWitness {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Use range 0.01..1.0 to avoid all-zeros (degenerate case)
    KVectorWitness {
        k_r: rng.gen_range(0.01..1.0),
        k_a: rng.gen_range(0.01..1.0),
        k_i: rng.gen_range(0.01..1.0),
        k_p: rng.gen_range(0.01..1.0),
        k_m: rng.gen_range(0.01..1.0),
        k_s: rng.gen_range(0.01..1.0),
        k_h: rng.gen_range(0.01..1.0),
        k_topo: rng.gen_range(0.01..1.0),
    }
}

fn benchmark_standard_prover(c: &mut Criterion) {
    let witness = sample_witness();

    c.bench_function("standard_prover", |b| {
        b.iter(|| {
            let trace = KVectorTrace::new(black_box(&witness)).unwrap();
            let prover = KVectorProver::new();
            prover.prove(trace).unwrap()
        })
    });
}

fn benchmark_security_levels(c: &mut Criterion) {
    let witness = sample_witness();
    let mut group = c.benchmark_group("security_levels");

    // Configure for longer benchmarks since proofs take seconds
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(60));

    for level in [SecurityLevel::Fast, SecurityLevel::Standard, SecurityLevel::High] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", level)),
            &level,
            |b, &level| {
                let prover = OptimizedKVectorProver::with_security_level(level);
                prover.clear_cache(); // Ensure no caching affects benchmark

                b.iter(|| {
                    prover.prove(black_box(&witness)).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn benchmark_caching(c: &mut Criterion) {
    let witness = sample_witness();
    let prover = OptimizedKVectorProver::new();

    let mut group = c.benchmark_group("caching");

    group.bench_function("cold_cache", |b| {
        b.iter(|| {
            prover.clear_cache();
            prover.prove(black_box(&witness)).unwrap()
        })
    });

    // Warm up cache
    let _ = prover.prove(&witness).unwrap();

    group.bench_function("warm_cache", |b| {
        b.iter(|| {
            prover.prove(black_box(&witness)).unwrap()
        })
    });

    group.finish();
}

fn benchmark_trace_creation(c: &mut Criterion) {
    let witness = sample_witness();

    c.bench_function("trace_creation", |b| {
        b.iter(|| {
            KVectorTrace::new(black_box(&witness)).unwrap()
        })
    });
}

fn benchmark_commitment_hash(c: &mut Criterion) {
    let witness = sample_witness();

    c.bench_function("commitment_hash", |b| {
        b.iter(|| {
            black_box(&witness).commitment()
        })
    });
}

#[cfg(feature = "parallel")]
fn benchmark_batch_proving(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_proving");
    group.sample_size(10);

    for batch_size in [2, 4, 8] {
        let witnesses: Vec<_> = (0..batch_size).map(|_| random_witness()).collect();
        let prover = OptimizedKVectorProver::with_security_level(SecurityLevel::Fast)
            .without_caching();

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &witnesses,
            |b, witnesses| {
                b.iter(|| {
                    prover.prove_batch(black_box(witnesses))
                })
            },
        );
    }

    group.finish();
}

fn benchmark_verification(c: &mut Criterion) {
    use kvector_zkp::proof::KVectorRangeProof;

    let witness = sample_witness();

    // Generate a proof first
    let proof = KVectorRangeProof::prove(&witness).expect("proof generation");

    c.bench_function("verification", |b| {
        b.iter(|| {
            black_box(&proof).verify().unwrap()
        })
    });
}

fn benchmark_end_to_end(c: &mut Criterion) {
    use kvector_zkp::proof::KVectorRangeProof;

    let witness = sample_witness();

    let mut group = c.benchmark_group("end_to_end");
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));

    group.bench_function("prove_and_verify", |b| {
        b.iter(|| {
            let proof = KVectorRangeProof::prove(black_box(&witness)).unwrap();
            proof.verify().unwrap();
            proof
        })
    });

    group.finish();
}

fn benchmark_proof_size(c: &mut Criterion) {
    use kvector_zkp::proof::KVectorRangeProof;

    let witness = sample_witness();
    let proof = KVectorRangeProof::prove(&witness).expect("proof generation");

    println!("\n=== Proof Size ===");
    println!("Binary size: {} bytes ({:.2} KB)", proof.size(), proof.size() as f64 / 1024.0);

    c.bench_function("serialization", |b| {
        b.iter(|| {
            black_box(&proof).to_bytes()
        })
    });
}

// Fast benchmarks for quick iteration
criterion_group!(
    name = fast_benches;
    config = Criterion::default();
    targets = benchmark_trace_creation,
              benchmark_commitment_hash,
              benchmark_verification,
              benchmark_proof_size,
);

// Full prover benchmarks (slower, more thorough)
criterion_group!(
    name = prover_benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(std::time::Duration::from_secs(60));
    targets = benchmark_standard_prover,
              benchmark_security_levels,
              benchmark_caching,
              benchmark_end_to_end,
);

criterion_main!(fast_benches, prover_benches);
