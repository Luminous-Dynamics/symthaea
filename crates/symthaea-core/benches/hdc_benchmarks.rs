// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Core HDC Operations Benchmark Suite
//!
//! Benchmarks for fundamental BinaryHV (16,384-bit) operations:
//! 1. BinaryHV::random()           - Deterministic vector generation (BLAKE3)
//! 2. BinaryHV::bind()             - XOR binding (SIMD-accelerated)
//! 3. BinaryHV::bundle()           - Majority-vote bundling (2, 5, 10 vectors)
//! 4. BinaryHV::similarity()       - Hamming similarity (SIMD popcount)
//! 5. BinaryHV::permute()          - Word-level cyclic rotation
//!
//! Run with: cargo bench --bench hdc_benchmarks

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use symthaea_core::hdc::binary_hv::BinaryHV;

// ============================================================================
// 1. RANDOM VECTOR GENERATION
// ============================================================================

fn bench_random(c: &mut Criterion) {
    let mut group = c.benchmark_group("BinaryHV_random");

    group.bench_function("single", |b| {
        let mut seed = 0u64;
        b.iter(|| {
            seed += 1;
            black_box(BinaryHV::random(black_box(seed)))
        })
    });

    group.bench_function("batch_100", |b| {
        b.iter(|| {
            let vecs: Vec<BinaryHV> = (0..100).map(|i| BinaryHV::random(i)).collect();
            black_box(vecs)
        })
    });

    group.finish();
}

// ============================================================================
// 2. BIND (XOR) OPERATION
// ============================================================================

fn bench_bind(c: &mut Criterion) {
    let mut group = c.benchmark_group("BinaryHV_bind");

    let a = BinaryHV::random(1);
    let b = BinaryHV::random(2);

    group.bench_function("simd", |bench| {
        bench.iter(|| black_box(a.bind(black_box(&b))))
    });

    group.bench_function("scalar", |bench| {
        bench.iter(|| black_box(a.bind_scalar(black_box(&b))))
    });

    // Chained binding: A ^ B ^ C ^ D (common in role-filler encoding)
    let c_vec = BinaryHV::random(3);
    let d = BinaryHV::random(4);

    group.bench_function("chained_4", |bench| {
        bench.iter(|| {
            let ab = a.bind(&b);
            let abc = ab.bind(&c_vec);
            black_box(abc.bind(&d))
        })
    });

    group.finish();
}

// ============================================================================
// 3. BUNDLE (MAJORITY VOTE) OPERATION
// ============================================================================

fn bench_bundle(c: &mut Criterion) {
    let mut group = c.benchmark_group("BinaryHV_bundle");

    for n in [2, 5, 10, 50, 100] {
        let vectors: Vec<BinaryHV> = (0..n).map(|i| BinaryHV::random(i)).collect();

        group.bench_with_input(BenchmarkId::new("standard", n), &vectors, |b, vecs| {
            b.iter(|| black_box(BinaryHV::bundle(black_box(vecs))))
        });

        group.bench_with_input(BenchmarkId::new("safe", n), &vectors, |b, vecs| {
            b.iter(|| black_box(BinaryHV::bundle_safe(black_box(vecs))))
        });
    }

    group.finish();
}

// ============================================================================
// 4. SIMILARITY (HAMMING) OPERATION
// ============================================================================

fn bench_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("BinaryHV_similarity");

    let a = BinaryHV::random(1);
    let b = BinaryHV::random(2);

    group.bench_function("simd", |bench| {
        bench.iter(|| black_box(a.similarity(black_box(&b))))
    });

    group.bench_function("scalar", |bench| {
        bench.iter(|| black_box(a.similarity_scalar(black_box(&b))))
    });

    group.bench_function("cosine_bipolar", |bench| {
        bench.iter(|| black_box(a.cosine_similarity(black_box(&b))))
    });

    // Batch similarity: search through N candidates (common in associative memory)
    for n in [10, 100, 1000] {
        let query = BinaryHV::random(0);
        let candidates: Vec<BinaryHV> = (1..=n).map(|i| BinaryHV::random(i as u64)).collect();

        group.bench_with_input(
            BenchmarkId::new("batch_search", n),
            &(&query, &candidates),
            |b, (q, cands)| {
                b.iter(|| {
                    let best = cands.iter().map(|c| q.similarity(c)).fold(0.0f32, f32::max);
                    black_box(best)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// 5. PERMUTE (CYCLIC ROTATION) OPERATION
// ============================================================================

fn bench_permute(c: &mut Criterion) {
    let mut group = c.benchmark_group("BinaryHV_permute");

    let v = BinaryHV::random(42);

    // Test various shift amounts: aligned (word boundary) and unaligned
    for shift in [1, 8, 64, 128, 1000, 8192] {
        group.bench_with_input(BenchmarkId::new("fast", shift), &shift, |b, &s| {
            b.iter(|| black_box(v.permute(black_box(s))))
        });
    }

    // Compare fast vs legacy for representative shift
    group.bench_function("legacy_shift_1", |b| {
        b.iter(|| black_box(v.permute_legacy(black_box(1))))
    });

    group.bench_function("fast_shift_1", |b| {
        b.iter(|| black_box(v.permute(black_box(1))))
    });

    group.finish();
}

// ============================================================================
// COMPOSITE OPERATIONS (real-world patterns)
// ============================================================================

fn bench_composite(c: &mut Criterion) {
    let mut group = c.benchmark_group("BinaryHV_composite");

    // Sequence encoding: sum_i permute(bind(role_i, filler_i), i)
    // This is the core operation for representing structured data in HDC
    group.bench_function("sequence_encode_5", |bench| {
        let roles: Vec<BinaryHV> = (0..5).map(|i| BinaryHV::random(100 + i)).collect();
        let fillers: Vec<BinaryHV> = (0..5).map(|i| BinaryHV::random(200 + i)).collect();

        bench.iter(|| {
            let bound: Vec<BinaryHV> = roles
                .iter()
                .zip(fillers.iter())
                .enumerate()
                .map(|(i, (r, f))| r.bind(f).permute(i))
                .collect();
            black_box(BinaryHV::bundle(&bound))
        })
    });

    // Record-like structure: bind(role, filler) for each field, then bundle
    group.bench_function("record_encode_10_fields", |bench| {
        let roles: Vec<BinaryHV> = (0..10).map(|i| BinaryHV::random(300 + i)).collect();
        let fillers: Vec<BinaryHV> = (0..10).map(|i| BinaryHV::random(400 + i)).collect();

        bench.iter(|| {
            let bindings: Vec<BinaryHV> = roles
                .iter()
                .zip(fillers.iter())
                .map(|(r, f)| r.bind(f))
                .collect();
            black_box(BinaryHV::bundle(&bindings))
        })
    });

    // Similarity-based retrieval from a small codebook
    group.bench_function("codebook_lookup_256", |bench| {
        let codebook: Vec<BinaryHV> = (0..256).map(|i| BinaryHV::random(i)).collect();
        let query = BinaryHV::random(999);

        bench.iter(|| {
            let (best_idx, best_sim) = codebook
                .iter()
                .enumerate()
                .map(|(i, v)| (i, query.similarity(v)))
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            black_box((best_idx, best_sim))
        })
    });

    group.finish();
}

// ============================================================================
// CRITERION MAIN
// ============================================================================

criterion_group!(
    benches,
    bench_random,
    bench_bind,
    bench_bundle,
    bench_similarity,
    bench_permute,
    bench_composite,
);

criterion_main!(benches);
