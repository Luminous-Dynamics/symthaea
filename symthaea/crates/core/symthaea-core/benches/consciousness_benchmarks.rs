// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Consciousness Pipeline Benchmarks
//!
//! Run with: `cargo bench --bench consciousness_benchmarks -p symthaea-core`
//! Smoke test: `cargo bench --bench consciousness_benchmarks -p symthaea-core -- --test`
//!
//! Benchmarks for the consciousness pipeline and supporting batch operations:
//! - pipeline_process: End-to-end process() with varying input counts
//! - batch_similarity_matrix: N×N pairwise similarity computation
//! - cluster_by_similarity: Greedy clustering from similarity matrix
//! - binary_to_continuous_pooled: BinaryHV → ContinuousHV pool conversion
//! - find_similar: Threshold-filtered similarity search

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::consciousness_integration::{ConsciousnessPipeline, IntegrationConfig};
use symthaea_core::hdc::consciousness_perf;

// =============================================================================
// PIPELINE PROCESS BENCHMARKS
// =============================================================================

fn bench_pipeline_process(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_process");
    // Use shorter measurement for these heavier benchmarks
    group.sample_size(10);

    for n in [4, 8, 16, 32] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            let mut pipeline = ConsciousnessPipeline::new(IntegrationConfig::default());
            pipeline.set_embodiment(0.8);
            let inputs: Vec<BinaryHV> = (0..n).map(|s| BinaryHV::random(s as u64)).collect();
            let priorities: Vec<f64> = vec![0.8; n];
            b.iter(|| {
                let state = pipeline.process(black_box(inputs.clone()), black_box(&priorities));
                black_box(state);
            });
        });
    }
    group.finish();
}

// =============================================================================
// BATCH SIMILARITY BENCHMARKS
// =============================================================================

fn bench_batch_similarity_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_similarity_matrix");

    for n in [4, 8, 16, 32] {
        let hvs: Vec<BinaryHV> = (0..n).map(|s| BinaryHV::random(s)).collect();
        let refs: Vec<&BinaryHV> = hvs.iter().collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let matrix = consciousness_perf::batch_similarity_matrix(black_box(&refs));
                black_box(matrix);
            });
        });
    }
    group.finish();
}

fn bench_cluster_by_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("cluster_by_similarity");

    for n in [8, 16, 32] {
        let hvs: Vec<BinaryHV> = (0..n).map(|s| BinaryHV::random(s)).collect();
        let refs: Vec<&BinaryHV> = hvs.iter().collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let clusters = consciousness_perf::cluster_by_similarity(black_box(&refs), 0.3);
                black_box(clusters);
            });
        });
    }
    group.finish();
}

// =============================================================================
// HV CONVERSION BENCHMARKS
// =============================================================================

fn bench_binary_to_continuous_pooled(c: &mut Criterion) {
    let hv = BinaryHV::random(42);
    c.bench_function("binary_to_continuous_pooled", |b| {
        b.iter(|| {
            let continuous = consciousness_perf::binary_to_continuous_pooled(black_box(&hv));
            black_box(continuous);
        });
    });
}

// =============================================================================
// SEARCH BENCHMARKS
// =============================================================================

fn bench_find_similar(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_similar");

    for n in [16, 64, 256] {
        let query = BinaryHV::random(0);
        let candidates: Vec<BinaryHV> = (1..=n).map(|s| BinaryHV::random(s)).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                let results = consciousness_perf::find_similar(
                    black_box(&query),
                    black_box(&candidates),
                    0.6,
                );
                black_box(results);
            });
        });
    }
    group.finish();
}

// =============================================================================
// SUBSYSTEM DISPATCH OVERHEAD BENCHMARKS
// =============================================================================

fn bench_subsystem_dispatch_overhead(c: &mut Criterion) {
    use symthaea_core::hdc::consciousness_metacognitive::MetacognitiveSubsystem;
    use symthaea_core::hdc::consciousness_phi_optimization::PhiOptimizationSubsystem;
    use symthaea_core::hdc::consciousness_self_awareness::SelfAwarenessSubsystem;

    let mut group = c.benchmark_group("subsystem_dispatch");
    group.sample_size(10);

    let inputs: Vec<BinaryHV> = (0..8).map(|s| BinaryHV::random(s)).collect();
    let priorities: Vec<f64> = vec![0.8; 8];

    // Baseline: pipeline with 0 subsystems
    group.bench_function("0_subsystems", |b| {
        let mut pipeline = ConsciousnessPipeline::new(IntegrationConfig::default());
        pipeline.set_embodiment(0.8);
        b.iter(|| {
            let state = pipeline.process(black_box(inputs.clone()), black_box(&priorities));
            black_box(state);
        });
    });

    // 1 subsystem
    group.bench_function("1_subsystem", |b| {
        let mut pipeline = ConsciousnessPipeline::new(IntegrationConfig::default());
        pipeline.set_embodiment(0.8);
        pipeline.register_subsystem(Box::new(MetacognitiveSubsystem::new()));
        b.iter(|| {
            let state = pipeline.process(black_box(inputs.clone()), black_box(&priorities));
            black_box(state);
        });
    });

    // 3 subsystems
    group.bench_function("3_subsystems", |b| {
        let mut pipeline = ConsciousnessPipeline::new(IntegrationConfig::default());
        pipeline.set_embodiment(0.8);
        pipeline.register_subsystem(Box::new(MetacognitiveSubsystem::new()));
        pipeline.register_subsystem(Box::new(SelfAwarenessSubsystem::new()));
        pipeline.register_subsystem(Box::new(PhiOptimizationSubsystem::new(1)));
        b.iter(|| {
            let state = pipeline.process(black_box(inputs.clone()), black_box(&priorities));
            black_box(state);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_pipeline_process,
    bench_batch_similarity_matrix,
    bench_cluster_by_similarity,
    bench_binary_to_continuous_pooled,
    bench_find_similar,
    bench_subsystem_dispatch_overhead,
);
criterion_main!(benches);
