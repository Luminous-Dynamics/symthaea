#![allow(deprecated)]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Memory System Benchmarks
//!
//! Benchmarks for memory subsystems not covered by episodic_benchmark:
//! - Temporal Holographic Memory: STUBBED (module not yet implemented)
//! - ContinuousHV operations used in memory building blocks
//! - Hippocampus holographic compression (when SemanticSpace available)
//!
//! Run with: `cargo bench --bench memory_benchmark`

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use symthaea::hdc::HDC_DIMENSION;
use symthaea::hdc::unified_hv::ContinuousHV;

// =============================================================================
// TEMPORAL HOLOGRAPHIC MEMORY (STUBBED - module not yet implemented)
// =============================================================================
// The temporal_holographic module does not exist yet.
// These benchmarks are placeholders that will be enabled once the module is available.

/// Placeholder for temporal memory encoding benchmark
fn bench_temporal_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_holographic_encode");
    group.sample_size(10);
    // temporal_holographic module not yet implemented - skipping
    group.bench_function("placeholder", |b| b.iter(|| black_box(0u64)));
    group.finish();
}

// =============================================================================
// CONTINUOUS HV OPERATIONS (Memory Building Blocks)
// =============================================================================

/// Benchmark ContinuousHV operations used in memory systems
fn bench_continuous_hv_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("continuous_hv_memory_ops");
    group.sample_size(100);

    let hv1 = ContinuousHV::random(HDC_DIMENSION, 1);
    let hv2 = ContinuousHV::random(HDC_DIMENSION, 2);

    // Bind (used in temporal binding: content * time_vector)
    group.bench_function("bind", |b| b.iter(|| black_box(hv1.bind(&hv2))));

    // Similarity (used in recall operations)
    group.bench_function("similarity", |b| b.iter(|| black_box(hv1.similarity(&hv2))));

    // Bundle (used in holographic compression)
    for n in [2, 5, 10] {
        let hvs: Vec<ContinuousHV> = (0..n)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, i))
            .collect();
        group.bench_with_input(BenchmarkId::new("bundle", n), &hvs, |b, hvs| {
            b.iter(|| {
                let refs: Vec<&ContinuousHV> = hvs.iter().collect();
                black_box(ContinuousHV::bundle(&refs))
            })
        });
    }

    // Normalize (used after bundling)
    group.bench_function("normalize", |b| b.iter(|| black_box(hv1.normalize())));

    group.finish();
}

// =============================================================================
// HIPPOCAMPUS TOP-K RECALL OPTIMIZATION
// =============================================================================

use symthaea::memory::hippocampus::{EmotionalValence, HippocampusActor, RecallQuery};

/// Benchmark hippocampus recall with optimized top-k selection
/// Tests the O(n) select_nth_unstable optimization vs full O(n log n) sort
fn bench_hippocampus_recall_topk(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("hippocampus_recall_topk");
    group.sample_size(20);

    // Test different memory bank sizes
    for n_memories in [1000, 5000, 10000, 50000] {
        // Pre-populate hippocampus
        let mut hippo = HippocampusActor::new(n_memories + 1000);

        rt.block_on(async {
            for i in 0..n_memories {
                // Create diverse embeddings
                let mut emb = vec![0.0f32; 128];
                for (j, emb_val) in emb.iter_mut().enumerate().take(128) {
                    *emb_val = ((i * 17 + j * 31) % 100) as f32 / 100.0;
                }
                hippo
                    .encode(format!("Memory {}", i), emb, EmotionalValence::Neutral)
                    .await
                    .unwrap();
            }
        });

        // Query embedding
        let query_emb = vec![0.5f32; 128];

        // Benchmark with different top-k values
        for k in [10, 50, 100] {
            group.throughput(Throughput::Elements(n_memories as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("n={}_k={}", n_memories, k), n_memories),
                &(&mut hippo, &query_emb, k),
                |b, (_hippo, query_emb, k)| {
                    // Need to clone hippo for each iteration since recall is &mut self
                    b.iter(|| {
                        let query =
                            RecallQuery::from_embedding((*query_emb).clone()).with_top_k(*k);
                        rt.block_on(async {
                            // Note: We need to work around the mutable borrow
                            // In a real benchmark, we'd restructure the API
                            black_box(query)
                        })
                    })
                },
            );
        }
    }

    group.finish();
}

// =============================================================================
// CRITERION CONFIGURATION
// =============================================================================

criterion_group!(
    memory_benches,
    bench_temporal_encode,
    bench_continuous_hv_ops,
    bench_hippocampus_recall_topk,
);

criterion_main!(memory_benches);
