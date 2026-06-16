// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Stress Test Benchmarks
//!
//! Tests system behavior under heavy load:
//! - Memory usage under sustained operations
//! - Large topology handling
//! - Concurrent load testing
//!
//! ## Usage
//!
//! ```bash
//! cargo bench --bench stress
//! cargo bench --bench stress -- memory
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use symthaea::hdc::{
    HDC_DIMENSION, binary_hv::BinaryHV, consciousness_topology_generators::ConsciousnessTopology,
    spectral_connectivity::ConnectivityCalculator, unified_hv::ContinuousHV,
};

// =============================================================================
// MEMORY STRESS TEST
// =============================================================================

fn bench_memory_stress(c: &mut Criterion) {
    let mut group = c.benchmark_group("stress_memory");
    group.sample_size(10);

    for n_ops in [1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(*n_ops as u64));

        group.bench_with_input(
            BenchmarkId::new("sustained_operations", n_ops),
            n_ops,
            |b, &n| {
                let calc = ConnectivityCalculator::new();

                b.iter(|| {
                    // Create and discard many topologies
                    let results: Vec<_> = (0..n)
                        .map(|i| {
                            let topo = ConsciousnessTopology::random(8, HDC_DIMENSION, i as u64);
                            calc.algebraic_connectivity(&topo.node_representations)
                        })
                        .collect();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// LARGE TOPOLOGY STRESS
// =============================================================================

fn bench_large_topology(c: &mut Criterion) {
    let mut group = c.benchmark_group("stress_large_topology");
    group.sample_size(10);

    let calc = ConnectivityCalculator::new();

    for n_nodes in [16, 32, 64, 128].iter() {
        group.bench_with_input(
            BenchmarkId::new("topology_nodes", n_nodes),
            n_nodes,
            |b, &n| {
                b.iter(|| {
                    let topo = ConsciousnessTopology::ring(n, HDC_DIMENSION, 42);
                    let result = calc.algebraic_connectivity(&topo.node_representations);
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// HDC VECTOR STRESS
// =============================================================================

fn bench_hdc_vector_stress(c: &mut Criterion) {
    let mut group = c.benchmark_group("stress_hdc_vectors");
    group.sample_size(10);

    for n_vectors in [100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*n_vectors as u64));

        group.bench_with_input(
            BenchmarkId::new("create_vectors", n_vectors),
            n_vectors,
            |b, &n| {
                b.iter(|| {
                    let vectors: Vec<_> = (0..n)
                        .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
                        .collect();
                    black_box(vectors)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bind_chain", n_vectors),
            n_vectors,
            |b, &n| {
                let vectors: Vec<_> = (0..n).map(|i| BinaryHV::random(i as u64)).collect();

                b.iter(|| {
                    let mut result = vectors[0];
                    for hv in vectors.iter().skip(1) {
                        result = result.bind(hv);
                    }
                    black_box(result)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// PARALLEL STRESS TEST
// =============================================================================

fn bench_parallel_stress(c: &mut Criterion) {
    use rayon::prelude::*;

    let mut group = c.benchmark_group("stress_parallel");
    group.sample_size(10);

    let calc = ConnectivityCalculator::new();

    for n_parallel in [10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*n_parallel as u64));

        group.bench_with_input(
            BenchmarkId::new("parallel_phi", n_parallel),
            n_parallel,
            |b, &n| {
                let topologies: Vec<_> = (0..n)
                    .map(|i| ConsciousnessTopology::ring(8, HDC_DIMENSION, i as u64))
                    .collect();

                b.iter(|| {
                    let results: Vec<_> = topologies
                        .par_iter()
                        .map(|topo| calc.algebraic_connectivity(&topo.node_representations))
                        .collect();
                    black_box(results)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// DIMENSIONAL SCALING STRESS
// =============================================================================

fn bench_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("stress_dimension_scaling");
    group.sample_size(10);

    let calc = ConnectivityCalculator::new();

    // Test hypercube dimensions 1-7
    for dim in 1..=7 {
        let n_nodes = 1 << dim; // 2^dim nodes

        if n_nodes <= 128 {
            // Limit for reasonable benchmark time
            group.bench_with_input(BenchmarkId::new("hypercube_dim", dim), &dim, |b, &d| {
                b.iter(|| {
                    let topo = ConsciousnessTopology::hypercube(d, HDC_DIMENSION, 42);
                    let result = calc.algebraic_connectivity(&topo.node_representations);
                    black_box(result)
                })
            });
        }
    }

    group.finish();
}

// =============================================================================
// TOPOLOGY VARIETY STRESS
// =============================================================================

fn bench_all_topologies(c: &mut Criterion) {
    let mut group = c.benchmark_group("stress_all_topologies");
    group.sample_size(10);

    let calc = ConnectivityCalculator::new();
    let n_nodes = 8;

    // Test multiple topology types
    let topologies = vec![
        (
            "random",
            ConsciousnessTopology::random(n_nodes, HDC_DIMENSION, 42),
        ),
        (
            "star",
            ConsciousnessTopology::star(n_nodes, HDC_DIMENSION, 42),
        ),
        (
            "ring",
            ConsciousnessTopology::ring(n_nodes, HDC_DIMENSION, 42),
        ),
        (
            "line",
            ConsciousnessTopology::line(n_nodes, HDC_DIMENSION, 42),
        ),
        (
            "dense_network",
            ConsciousnessTopology::dense_network(n_nodes, HDC_DIMENSION, None, 42),
        ),
        (
            "torus",
            ConsciousnessTopology::torus(3, 3, HDC_DIMENSION, 42),
        ),
        (
            "hypercube_3d",
            ConsciousnessTopology::hypercube(3, HDC_DIMENSION, 42),
        ),
        (
            "hypercube_4d",
            ConsciousnessTopology::hypercube(4, HDC_DIMENSION, 42),
        ),
    ];

    for (name, topo) in topologies {
        group.bench_with_input(BenchmarkId::new("topology", name), &topo, |b, t| {
            b.iter(|| {
                let result = calc.algebraic_connectivity(&t.node_representations);
                black_box(result)
            })
        });
    }

    group.finish();
}

// =============================================================================
// CRITERION CONFIGURATION
// =============================================================================

criterion_group!(
    stress_benches,
    bench_memory_stress,
    bench_large_topology,
    bench_hdc_vector_stress,
    bench_parallel_stress,
    bench_dimension_scaling,
    bench_all_topologies,
);

criterion_main!(stress_benches);
