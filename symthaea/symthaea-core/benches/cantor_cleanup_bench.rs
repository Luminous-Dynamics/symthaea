// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Benchmarks: Cantor Recursive HV cleanup vs flat BinaryHV codebook cleanup
//!
//! Measures whether fractal layer-preserving cleanup (CRHV) yields better
//! memory fidelity than standard flat snap-to-nearest-codebook cleanup,
//! and at what throughput cost.
//!
//! Run: `cargo bench -p symthaea-core --bench cantor_cleanup_bench`

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::cantor_recursive_hv::CantorRecursiveHV;
use symthaea_core::hdc::cantor_resonator_cleanup::{
    BinaryCodebook, CantorCleanupConfig, CantorCleanupEngine,
};

/// Build a codebook of N random "concept" vectors
fn build_codebook(n: usize) -> BinaryCodebook {
    let mut cb = BinaryCodebook::new(n + 10);
    for i in 0..n {
        cb.add(&format!("concept_{i}"), BinaryHV::random(i as u64));
    }
    cb
}

/// Add noise to a BinaryHV by flipping a fraction of bits
fn add_noise(hv: &BinaryHV, noise_fraction: f32, seed: u64) -> BinaryHV {
    let noise = BinaryHV::random(seed);
    let flip_threshold = (BinaryHV::DIM as f32 * noise_fraction) as usize;
    let mut result = *hv;
    let mut flipped = 0;
    for byte_idx in 0..2048 {
        for bit in 0..8 {
            if flipped >= flip_threshold {
                return result;
            }
            if noise.0[byte_idx] & (1 << bit) != 0 {
                result.0[byte_idx] ^= 1 << bit;
                flipped += 1;
            }
        }
    }
    result
}

// =============================================================================
// THROUGHPUT BENCHMARKS
// =============================================================================

fn bench_crhv_cleanup_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("cantor_cleanup_throughput");

    for codebook_size in [16, 64, 256] {
        for depth in [3, 5, 7] {
            let label = format!("cb{codebook_size}_d{depth}");
            let mut engine = CantorCleanupEngine::new(CantorCleanupConfig::default());
            for i in 0..codebook_size {
                engine
                    .codebook
                    .add(&format!("concept_{i}"), BinaryHV::random(i as u64));
            }

            let crhv = CantorRecursiveHV::from_base_with_depth(BinaryHV::random(42), depth);
            engine.learn("target", &crhv);

            group.bench_function(BenchmarkId::new("crhv_cleanup", &label), |b| {
                b.iter(|| {
                    let result = engine.cleanup(black_box(&crhv));
                    black_box(result.quality)
                })
            });
        }
    }

    group.finish();
}

fn bench_flat_cleanup_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("flat_cleanup_throughput");

    for codebook_size in [16, 64, 256] {
        let label = format!("cb{codebook_size}");
        let codebook = build_codebook(codebook_size);
        let query = BinaryHV::random(42);

        group.bench_function(BenchmarkId::new("flat_cleanup", &label), |b| {
            b.iter(|| {
                let result = codebook.cleanup(black_box(&query));
                black_box(result.map(|(_, sim, _)| sim))
            })
        });
    }

    group.finish();
}

// =============================================================================
// FIDELITY BENCHMARKS — Does CRHV preserve more information after noise+cleanup?
// =============================================================================

fn bench_memory_fidelity(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_fidelity");
    group.sample_size(50);

    for noise_level in [0.05, 0.10, 0.20, 0.30] {
        let label = format!("noise_{:.0}pct", noise_level * 100.0);

        let mut engine = CantorCleanupEngine::new(CantorCleanupConfig::default());
        let concepts: Vec<CantorRecursiveHV> = (0..32)
            .map(|i| {
                let crhv = CantorRecursiveHV::from_base_with_depth(BinaryHV::random(i), 5);
                engine.learn(&format!("c{i}"), &crhv);
                crhv
            })
            .collect();

        group.bench_function(BenchmarkId::new("crhv_fidelity", &label), |b| {
            b.iter(|| {
                let mut total_sim = 0.0f32;
                for (i, original) in concepts.iter().enumerate() {
                    let noisy_base = add_noise(&original.base, noise_level, (i * 1000) as u64);
                    let noisy_crhv =
                        CantorRecursiveHV::from_base_with_depth(noisy_base, original.depth);
                    let result = engine.cleanup(&noisy_crhv);
                    total_sim += result.cleaned.vector.similarity(&original.vector);
                }
                black_box(total_sim / concepts.len() as f32)
            })
        });

        let codebook = {
            let mut cb = BinaryCodebook::new(64);
            for (i, c) in concepts.iter().enumerate() {
                cb.add(&format!("c{i}"), c.base);
            }
            cb
        };

        group.bench_function(BenchmarkId::new("flat_fidelity", &label), |b| {
            b.iter(|| {
                let mut total_sim = 0.0f32;
                for (i, original) in concepts.iter().enumerate() {
                    let noisy = add_noise(&original.base, noise_level, (i * 1000) as u64);
                    if let Some((_, _, cleaned)) = codebook.cleanup(&noisy) {
                        total_sim += cleaned.similarity(&original.base);
                    }
                }
                black_box(total_sim / concepts.len() as f32)
            })
        });
    }

    group.finish();
}

// =============================================================================
// SELF-SIMILARITY PRESERVATION — Does cleanup maintain fractal depth?
// =============================================================================

fn bench_self_similarity_preservation(c: &mut Criterion) {
    let mut group = c.benchmark_group("self_similarity_preservation");
    group.sample_size(50);

    let mut engine = CantorCleanupEngine::new(CantorCleanupConfig::default());
    let concepts: Vec<CantorRecursiveHV> = (0..16)
        .map(|i| {
            let crhv = CantorRecursiveHV::from_base_with_depth(BinaryHV::random(i), 7);
            engine.learn(&format!("deep_{i}"), &crhv);
            crhv
        })
        .collect();

    for noise_level in [0.05, 0.15, 0.25] {
        let label = format!("noise_{:.0}pct", noise_level * 100.0);

        group.bench_function(BenchmarkId::new("self_sim_after_cleanup", &label), |b| {
            b.iter(|| {
                let mut total_delta = 0.0f32;
                for (i, original) in concepts.iter().enumerate() {
                    let original_ss = original.self_similarity();
                    let noisy_base = add_noise(&original.base, noise_level, (i * 1000) as u64);
                    let noisy_crhv =
                        CantorRecursiveHV::from_base_with_depth(noisy_base, original.depth);
                    let result = engine.cleanup(&noisy_crhv);
                    let cleaned_ss = result.cleaned.self_similarity();
                    total_delta += (cleaned_ss - original_ss).abs();
                }
                black_box(total_delta / concepts.len() as f32)
            })
        });
    }

    group.finish();
}

// =============================================================================
// BATCH CLEANUP — Dream consolidation throughput
// =============================================================================

fn bench_batch_cleanup(c: &mut Criterion) {
    let mut group = c.benchmark_group("dream_consolidation_batch");
    group.sample_size(20);

    let mut engine = CantorCleanupEngine::new(CantorCleanupConfig::default());
    for i in 0..64 {
        engine
            .codebook
            .add(&format!("concept_{i}"), BinaryHV::random(i));
    }

    for batch_size in [4, 8, 16, 32] {
        let crhvs: Vec<CantorRecursiveHV> = (0..batch_size)
            .map(|i| CantorRecursiveHV::from_base_with_depth(BinaryHV::random(i as u64 + 100), 4))
            .collect();

        group.bench_function(BenchmarkId::new("batch", batch_size), |b| {
            b.iter(|| {
                let results = engine.cleanup_batch(black_box(&crhvs));
                black_box(results.len())
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_crhv_cleanup_throughput,
    bench_flat_cleanup_throughput,
    bench_memory_fidelity,
    bench_self_similarity_preservation,
    bench_batch_cleanup,
);
criterion_main!(benches);
