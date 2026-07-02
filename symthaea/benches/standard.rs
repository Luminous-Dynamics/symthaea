// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Standard Benchmark Suite
//!
//! Main benchmark suite (~30 min) covering core capabilities.
//! Reference: BENCHMARKING_STRATEGY.md Section 36
//!
//! ## Included Benchmarks
//!
//! - HDC operations (comprehensive)
//! - Φ on 8 standard topologies
//! - Temporal reasoning
//! - Scalability tests
//!
//! ## Usage
//!
//! ```bash
//! cargo bench --bench standard
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use symthaea::hdc::{
    HDC_DIMENSION, binary_hv::BinaryHV, consciousness_topology_generators::ConsciousnessTopology,
    phi_resonant::ResonantPhiCalculator, spectral_connectivity::ConnectivityCalculator,
};
// Note: temporal_primitives module is not yet implemented - benchmark removed

// =============================================================================
// HDC COMPREHENSIVE
// =============================================================================

fn bench_hdc_comprehensive(c: &mut Criterion) {
    let mut group = c.benchmark_group("std_hdc");
    group.sample_size(100);

    let hv1 = BinaryHV::random(42);
    let hv2 = BinaryHV::random(43);

    // All core operations
    group.bench_function("bind", |b| b.iter(|| black_box(hv1.bind(&hv2))));

    group.bench_function("similarity", |b| b.iter(|| black_box(hv1.similarity(&hv2))));

    group.bench_function("hamming_distance", |b| {
        b.iter(|| black_box(hv1.hamming_distance(&hv2)))
    });

    group.bench_function("popcount", |b| b.iter(|| black_box(hv1.popcount())));

    // Bundle scaling
    for size in [2, 5, 10, 20] {
        let hvs: Vec<BinaryHV> = (0..size).map(BinaryHV::random).collect();
        group.bench_with_input(BenchmarkId::new("bundle", size), &hvs, |b, hvs| {
            b.iter(|| black_box(BinaryHV::bundle(hvs)))
        });
    }

    // Permute scaling
    for steps in [1, 10, 100] {
        group.bench_with_input(BenchmarkId::new("permute", steps), &steps, |b, &steps| {
            b.iter(|| black_box(hv1.permute(steps)))
        });
    }

    group.finish();
}

// =============================================================================
// STANDARD TOPOLOGIES (8)
// =============================================================================

fn bench_standard_topologies(c: &mut Criterion) {
    let mut group = c.benchmark_group("std_topologies");
    group.sample_size(50);

    let calc = ConnectivityCalculator::new();

    let topologies = vec![
        ("Ring", ConsciousnessTopology::ring(8, HDC_DIMENSION, 42)),
        (
            "Torus",
            ConsciousnessTopology::torus(3, 3, HDC_DIMENSION, 42),
        ),
        (
            "Dense",
            ConsciousnessTopology::dense_network(8, HDC_DIMENSION, None, 42),
        ),
        (
            "Lattice",
            ConsciousnessTopology::lattice(8, HDC_DIMENSION, 42),
        ),
        (
            "Modular",
            ConsciousnessTopology::modular(8, HDC_DIMENSION, 2, 42),
        ),
        ("Line", ConsciousnessTopology::line(8, HDC_DIMENSION, 42)),
        ("Star", ConsciousnessTopology::star(8, HDC_DIMENSION, 42)),
        (
            "Random",
            ConsciousnessTopology::random(8, HDC_DIMENSION, 42),
        ),
    ];

    for (name, topo) in topologies {
        group.bench_function(name, |b| {
            b.iter(|| black_box(calc.algebraic_connectivity(&topo.node_representations)))
        });
    }

    group.finish();
}

// =============================================================================
// TEMPORAL REASONING (disabled - module not yet implemented)
// =============================================================================
// TODO: Re-enable when temporal_primitives module is implemented
// fn bench_temporal_reasoning(c: &mut Criterion) { ... }

// =============================================================================
// SCALABILITY
// =============================================================================

fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("std_scalability");
    group.sample_size(20);

    // BinaryHV search scalability
    let query = BinaryHV::random(42);

    for size in [100, 1000, 5000] {
        let corpus: Vec<BinaryHV> = (0..size).map(|i| BinaryHV::random(i as u64)).collect();
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("linear_search", size),
            &corpus,
            |b, corpus| {
                b.iter(|| {
                    let max_sim = corpus
                        .iter()
                        .map(|hv| query.similarity(hv))
                        .max_by(|a, b| a.total_cmp(b));
                    black_box(max_sim)
                })
            },
        );
    }

    // Φ scaling
    let calc = ResonantPhiCalculator::fast();
    for n_nodes in [8, 16, 32] {
        group.bench_with_input(BenchmarkId::new("phi_ring", n_nodes), &n_nodes, |b, &n| {
            let topo = ConsciousnessTopology::ring(n, HDC_DIMENSION, 42);
            b.iter(|| black_box(calc.compute(&topo.node_representations)))
        });
    }

    group.finish();
}

// =============================================================================
// HYPERCUBE DIMENSION SCALING
// =============================================================================

fn bench_hypercube_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("std_hypercube");
    group.sample_size(30);

    let calc = ConnectivityCalculator::new();

    for dim in 3..=5 {
        let n_nodes = 1 << dim;
        group.throughput(Throughput::Elements(n_nodes as u64));

        group.bench_with_input(
            BenchmarkId::new("dimension", format!("{}D", dim)),
            &dim,
            |b, &d| {
                let topo = ConsciousnessTopology::hypercube(d, HDC_DIMENSION, 42);
                b.iter(|| black_box(calc.algebraic_connectivity(&topo.node_representations)))
            },
        );
    }

    group.finish();
}

// =============================================================================
// CRITERION CONFIGURATION
// =============================================================================

criterion_group!(
    standard_benches,
    bench_hdc_comprehensive,
    bench_standard_topologies,
    // bench_temporal_reasoning,  // Disabled - temporal_primitives not yet implemented
    bench_scalability,
    bench_hypercube_scaling,
);

criterion_main!(standard_benches);
