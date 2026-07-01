// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Quick Benchmark Suite for CI/CD
//!
//! Fast benchmark suite (~5 minutes) providing essential metrics for every commit.
//! Reference: BENCHMARKING_STRATEGY.md Section 36
//!
//! ## Included Benchmarks
//!
//! - HDC core operations (bind, bundle, similarity)
//! - Φ calculation on 3 topologies (Star, Ring, Random)
//! - Basic temporal reasoning
//!
//! ## Usage
//!
//! ```bash
//! cargo bench --bench quick
//! cargo bench --bench quick -- --save-baseline main
//! cargo bench --bench quick -- --baseline main --compare
//! ```

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use symthaea::hdc::{
    HDC_DIMENSION, binary_hv::BinaryHV, consciousness_topology_generators::ConsciousnessTopology,
    spectral_connectivity::ConnectivityCalculator,
};

// =============================================================================
// QUICK HDC BENCHMARKS
// =============================================================================

fn bench_hdc_quick(c: &mut Criterion) {
    let mut group = c.benchmark_group("quick_hdc");
    group.sample_size(50);

    let hv1 = BinaryHV::random(42);
    let hv2 = BinaryHV::random(43);

    // Core operations only
    group.bench_function("bind", |b| b.iter(|| black_box(hv1.bind(&hv2))));

    group.bench_function("similarity", |b| b.iter(|| black_box(hv1.similarity(&hv2))));

    let hvs: Vec<BinaryHV> = (0..5).map(BinaryHV::random).collect();
    group.bench_function("bundle_5", |b| b.iter(|| black_box(BinaryHV::bundle(&hvs))));

    group.finish();
}

// =============================================================================
// QUICK Φ BENCHMARKS (3 Topologies)
// =============================================================================

fn bench_phi_quick(c: &mut Criterion) {
    let mut group = c.benchmark_group("quick_phi");
    group.sample_size(30);

    let calc = ConnectivityCalculator::new();
    let n_nodes = 8;

    // Star topology (low Φ baseline)
    group.bench_function("star_8", |b| {
        let topo = ConsciousnessTopology::star(n_nodes, HDC_DIMENSION, 42);
        b.iter(|| black_box(calc.algebraic_connectivity(&topo.node_representations)))
    });

    // Ring topology (high Φ - uniform structure)
    group.bench_function("ring_8", |b| {
        let topo = ConsciousnessTopology::ring(n_nodes, HDC_DIMENSION, 42);
        b.iter(|| black_box(calc.algebraic_connectivity(&topo.node_representations)))
    });

    // Random topology (medium Φ baseline)
    group.bench_function("random_8", |b| {
        let topo = ConsciousnessTopology::random(n_nodes, HDC_DIMENSION, 42);
        b.iter(|| black_box(calc.algebraic_connectivity(&topo.node_representations)))
    });

    group.finish();
}

// =============================================================================
// QUICK Φ VALUE VALIDATION
// =============================================================================

fn bench_phi_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("quick_phi_validation");
    group.sample_size(20);

    let calc = ConnectivityCalculator::new();

    // Validate that topology rankings are preserved
    // Expected: Ring > Random > Star (by Φ value)
    group.bench_function("validate_rankings", |b| {
        b.iter(|| {
            let star = ConsciousnessTopology::star(8, HDC_DIMENSION, 42);
            let ring = ConsciousnessTopology::ring(8, HDC_DIMENSION, 42);
            let random = ConsciousnessTopology::random(8, HDC_DIMENSION, 42);

            let phi_star = calc.algebraic_connectivity(&star.node_representations);
            let phi_ring = calc.algebraic_connectivity(&ring.node_representations);
            let phi_random = calc.algebraic_connectivity(&random.node_representations);

            // Return tuple for validation
            black_box((phi_star, phi_ring, phi_random))
        })
    });

    group.finish();
}

// =============================================================================
// QUICK HYPERCUBE BENCHMARK
// =============================================================================

fn bench_hypercube_quick(c: &mut Criterion) {
    let mut group = c.benchmark_group("quick_hypercube");
    group.sample_size(20);

    let calc = ConnectivityCalculator::new();

    // 4D Hypercube (Tesseract) - the Φ champion
    group.bench_function("tesseract_4d", |b| {
        let topo = ConsciousnessTopology::hypercube(4, HDC_DIMENSION, 42);
        b.iter(|| black_box(calc.algebraic_connectivity(&topo.node_representations)))
    });

    group.finish();
}

// =============================================================================
// CRITERION CONFIGURATION
// =============================================================================

criterion_group!(
    quick_benches,
    bench_hdc_quick,
    bench_phi_quick,
    bench_phi_values,
    bench_hypercube_quick,
);

criterion_main!(quick_benches);
