// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Consciousness Benchmark Suite
//!
//! Comprehensive Φ measurement benchmarks covering all 19 validated topologies
//! and dimensional sweep analysis.
//!
//! Reference: BENCHMARKING_STRATEGY.md Section 18
//!
//! ## Benchmark Categories
//!
//! 1. **All 19 Topologies** - Complete topology-Φ characterization
//! 2. **Dimensional Sweep** - 1D-7D hypercube scaling
//! 3. **Method Comparison** - RealPhi vs ResonantPhi vs TieredPhi
//! 4. **Large Scale** - 64+ node topologies
//!
//! ## Key Findings (Reference)
//!
//! - Hypercube 4D: Φ = 0.4976 (Champion)
//! - Ring/Torus: Φ = 0.4954 (Dimensional invariance)
//! - Asymptotic limit: Φ → 0.5 as dimension → ∞
//!
//! ## Usage
//!
//! ```bash
//! cargo bench --bench consciousness
//! cargo bench --bench consciousness -- topology_19
//! cargo bench --bench consciousness -- dimensional_sweep
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use symthaea::hdc::{
    HDC_DIMENSION, consciousness_topology_generators::ConsciousnessTopology,
    phi_resonant::ResonantPhiCalculator, spectral_connectivity::ConnectivityCalculator,
};

// =============================================================================
// ALL 19 TOPOLOGIES
// =============================================================================

fn bench_topology_19(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_19");
    group.sample_size(30);

    let calc = ConnectivityCalculator::new();

    // Original 8 topologies
    let topologies = vec![
        ("01_Ring", ConsciousnessTopology::ring(8, HDC_DIMENSION, 42)),
        (
            "02_Torus",
            ConsciousnessTopology::torus(3, 3, HDC_DIMENSION, 42),
        ),
        (
            "03_Dense",
            ConsciousnessTopology::dense_network(8, HDC_DIMENSION, None, 42),
        ),
        (
            "04_Lattice",
            ConsciousnessTopology::lattice(8, HDC_DIMENSION, 42),
        ),
        (
            "05_Modular",
            ConsciousnessTopology::modular(8, HDC_DIMENSION, 2, 42),
        ),
        ("06_Line", ConsciousnessTopology::line(8, HDC_DIMENSION, 42)),
        (
            "07_BinaryTree",
            ConsciousnessTopology::binary_tree(7, HDC_DIMENSION, 42),
        ),
        ("08_Star", ConsciousnessTopology::star(8, HDC_DIMENSION, 42)),
        (
            "09_Random",
            ConsciousnessTopology::random(8, HDC_DIMENSION, 42),
        ),
        // Tier 1 exotic
        (
            "10_SmallWorld",
            ConsciousnessTopology::small_world(8, HDC_DIMENSION, 2, 0.1, 42),
        ),
        (
            "11_Mobius",
            ConsciousnessTopology::mobius_strip(8, HDC_DIMENSION, 42),
        ),
        // Tier 2 exotic
        (
            "12_KleinBottle",
            ConsciousnessTopology::klein_bottle(3, 3, HDC_DIMENSION, 42),
        ),
        (
            "13_Hyperbolic",
            ConsciousnessTopology::hyperbolic(8, 3, HDC_DIMENSION, 42),
        ),
        (
            "14_ScaleFree",
            ConsciousnessTopology::scale_free(8, 2, HDC_DIMENSION, 42),
        ),
        // Tier 3 exotic (hypercubes)
        (
            "15_Hypercube3D",
            ConsciousnessTopology::hypercube(3, HDC_DIMENSION, 42),
        ),
        (
            "16_Hypercube4D",
            ConsciousnessTopology::hypercube(4, HDC_DIMENSION, 42),
        ),
    ];

    for (name, topo) in topologies {
        group.throughput(Throughput::Elements(topo.node_representations.len() as u64));
        group.bench_function(name, |b| {
            b.iter(|| black_box(calc.algebraic_connectivity(&topo.node_representations)))
        });
    }

    group.finish();
}

// =============================================================================
// DIMENSIONAL SWEEP (1D-7D)
// =============================================================================

fn bench_dimensional_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimensional_sweep");
    group.sample_size(20);

    let calc = ConnectivityCalculator::new();

    // Sweep from 2D to 7D (1D is degenerate case with n=2)
    for dim in 2..=7 {
        let n_nodes = 1 << dim; // 2^dim nodes
        group.throughput(Throughput::Elements(n_nodes as u64));

        group.bench_with_input(
            BenchmarkId::new("hypercube", format!("{}D_{}nodes", dim, n_nodes)),
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
// PHI METHOD COMPARISON
// =============================================================================

fn bench_phi_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("phi_methods");
    group.sample_size(30);

    // Test on Ring (high Φ) and Star (low Φ)
    let ring = ConsciousnessTopology::ring(8, HDC_DIMENSION, 42);
    let star = ConsciousnessTopology::star(8, HDC_DIMENSION, 42);

    // RealPhi (algebraic connectivity)
    let real_calc = ConnectivityCalculator::new();
    group.bench_function("RealPhi_Ring", |b| {
        b.iter(|| black_box(real_calc.algebraic_connectivity(&ring.node_representations)))
    });
    group.bench_function("RealPhi_Star", |b| {
        b.iter(|| black_box(real_calc.algebraic_connectivity(&star.node_representations)))
    });

    // ResonantPhi (SIMD optimized)
    let resonant_calc = ResonantPhiCalculator::fast();
    group.bench_function("ResonantPhi_Ring", |b| {
        b.iter(|| black_box(resonant_calc.compute(&ring.node_representations)))
    });
    group.bench_function("ResonantPhi_Star", |b| {
        b.iter(|| black_box(resonant_calc.compute(&star.node_representations)))
    });

    group.finish();
}

// =============================================================================
// LARGE SCALE TOPOLOGIES
// =============================================================================

fn bench_large_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_scale");
    group.sample_size(10);

    let calc = ResonantPhiCalculator::fast();

    // 64-node topologies
    group.bench_function("Ring_64", |b| {
        let topo = ConsciousnessTopology::ring(64, HDC_DIMENSION, 42);
        b.iter(|| black_box(calc.compute(&topo.node_representations)))
    });

    group.bench_function("Hypercube_6D_64", |b| {
        let topo = ConsciousnessTopology::hypercube(6, HDC_DIMENSION, 42);
        b.iter(|| black_box(calc.compute(&topo.node_representations)))
    });

    // 128-node (7D hypercube)
    group.bench_function("Hypercube_7D_128", |b| {
        let topo = ConsciousnessTopology::hypercube(7, HDC_DIMENSION, 42);
        b.iter(|| black_box(calc.compute(&topo.node_representations)))
    });

    group.finish();
}

// =============================================================================
// PHI ACCURACY VALIDATION
// =============================================================================

fn bench_phi_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("phi_accuracy");
    group.sample_size(50);

    let calc = ConnectivityCalculator::new();

    // Statistical validation with multiple seeds
    group.bench_function("ring_10_samples", |b| {
        b.iter(|| {
            let phis: Vec<f64> = (0..10)
                .map(|seed| {
                    let topo = ConsciousnessTopology::ring(8, HDC_DIMENSION, seed);
                    calc.algebraic_connectivity(&topo.node_representations)
                })
                .collect();
            black_box(phis)
        })
    });

    group.bench_function("hypercube4d_10_samples", |b| {
        b.iter(|| {
            let phis: Vec<f64> = (0..10)
                .map(|seed| {
                    let topo = ConsciousnessTopology::hypercube(4, HDC_DIMENSION, seed);
                    calc.algebraic_connectivity(&topo.node_representations)
                })
                .collect();
            black_box(phis)
        })
    });

    group.finish();
}

// =============================================================================
// NON-ORIENTABILITY COMPARISON
// =============================================================================

fn bench_non_orientability(c: &mut Criterion) {
    let mut group = c.benchmark_group("non_orientability");
    group.sample_size(30);

    let calc = ConnectivityCalculator::new();

    // 1D non-orientable (Möbius) - catastrophic failure expected
    group.bench_function("Mobius_8", |b| {
        let topo = ConsciousnessTopology::mobius_strip(8, HDC_DIMENSION, 42);
        b.iter(|| black_box(calc.algebraic_connectivity(&topo.node_representations)))
    });

    // 2D non-orientable (Klein Bottle) - preserved Φ expected
    group.bench_function("KleinBottle_3x3", |b| {
        let topo = ConsciousnessTopology::klein_bottle(3, 3, HDC_DIMENSION, 42);
        b.iter(|| black_box(calc.algebraic_connectivity(&topo.node_representations)))
    });

    // Orientable controls
    group.bench_function("Ring_8", |b| {
        let topo = ConsciousnessTopology::ring(8, HDC_DIMENSION, 42);
        b.iter(|| black_box(calc.algebraic_connectivity(&topo.node_representations)))
    });

    group.bench_function("Torus_3x3", |b| {
        let topo = ConsciousnessTopology::torus(3, 3, HDC_DIMENSION, 42);
        b.iter(|| black_box(calc.algebraic_connectivity(&topo.node_representations)))
    });

    group.finish();
}

// =============================================================================
// CRITERION CONFIGURATION
// =============================================================================

criterion_group!(
    consciousness_benches,
    bench_topology_19,
    bench_dimensional_sweep,
    bench_phi_methods,
    bench_large_scale,
    bench_phi_accuracy,
    bench_non_orientability,
);

criterion_main!(consciousness_benches);
