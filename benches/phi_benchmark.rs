//! Φ (Phi) Calculation Benchmarks
//!
//! Comprehensive benchmarks for consciousness measurement algorithms:
//! - RealPhiCalculator: Algebraic connectivity method
//! - ResonantPhiCalculator: SIMD-optimized resonator dynamics
//! - Topology scaling: 4, 8, 16, 32, 64 nodes
//! - Topology types: Star, Ring, Random, Dense, Hypercube
//!
//! Run with: `cargo bench --bench phi_benchmark`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use symthaea::hdc::{
    consciousness_topology_generators::ConsciousnessTopology,
    phi_real::RealPhiCalculator,
    phi_resonant::{ResonantPhiCalculator, ResonantConfig},
    HDC_DIMENSION,
};

// =============================================================================
// TOPOLOGY GENERATORS
// =============================================================================

fn create_star(n_nodes: usize, seed: u64) -> ConsciousnessTopology {
    ConsciousnessTopology::star(n_nodes, HDC_DIMENSION, seed)
}

fn create_ring(n_nodes: usize, seed: u64) -> ConsciousnessTopology {
    ConsciousnessTopology::ring(n_nodes, HDC_DIMENSION, seed)
}

fn create_random(n_nodes: usize, seed: u64) -> ConsciousnessTopology {
    ConsciousnessTopology::random(n_nodes, HDC_DIMENSION, seed)
}

fn create_dense(n_nodes: usize, seed: u64) -> ConsciousnessTopology {
    ConsciousnessTopology::dense_network(n_nodes, HDC_DIMENSION, None, seed)
}

fn create_hypercube_3d(seed: u64) -> ConsciousnessTopology {
    ConsciousnessTopology::hypercube(3, HDC_DIMENSION, seed)  // 8 nodes
}

fn create_hypercube_4d(seed: u64) -> ConsciousnessTopology {
    ConsciousnessTopology::hypercube(4, HDC_DIMENSION, seed)  // 16 nodes
}

fn create_hypercube_5d(seed: u64) -> ConsciousnessTopology {
    ConsciousnessTopology::hypercube(5, HDC_DIMENSION, seed)  // 32 nodes
}

// =============================================================================
// REAL PHI CALCULATOR BENCHMARKS (Algebraic Connectivity)
// =============================================================================

fn bench_real_phi_topology_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("RealPhi_Scaling");
    group.sample_size(30);

    let calc = RealPhiCalculator::new();

    for n_nodes in [4, 8, 16, 32].iter() {
        group.throughput(Throughput::Elements(*n_nodes as u64));

        group.bench_with_input(
            BenchmarkId::new("Star", n_nodes),
            n_nodes,
            |b, &n| {
                let topo = create_star(n, 42);
                b.iter(|| {
                    black_box(calc.compute(&topo.node_representations))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Ring", n_nodes),
            n_nodes,
            |b, &n| {
                let topo = create_ring(n, 42);
                b.iter(|| {
                    black_box(calc.compute(&topo.node_representations))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Random", n_nodes),
            n_nodes,
            |b, &n| {
                let topo = create_random(n, 42);
                b.iter(|| {
                    black_box(calc.compute(&topo.node_representations))
                });
            },
        );
    }

    group.finish();
}

fn bench_real_phi_topology_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("RealPhi_Types");
    group.sample_size(50);

    let calc = RealPhiCalculator::new();
    let n_nodes = 8;

    let topologies = vec![
        ("Star", create_star(n_nodes, 42)),
        ("Ring", create_ring(n_nodes, 42)),
        ("Random", create_random(n_nodes, 42)),
        ("Dense", create_dense(n_nodes, 42)),
        ("Hypercube3D", create_hypercube_3d(42)),
    ];

    for (name, topo) in topologies {
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(calc.compute(&topo.node_representations))
            });
        });
    }

    group.finish();
}

// =============================================================================
// RESONANT PHI CALCULATOR BENCHMARKS (SIMD-Optimized)
// =============================================================================

fn bench_resonant_phi_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("ResonantPhi_Scaling");
    group.sample_size(30);

    // Fast config for benchmarking
    let calc = ResonantPhiCalculator::fast();

    for n_nodes in [4, 8, 16, 32].iter() {
        group.throughput(Throughput::Elements(*n_nodes as u64));

        group.bench_with_input(
            BenchmarkId::new("Star", n_nodes),
            n_nodes,
            |b, &n| {
                let topo = create_star(n, 42);
                b.iter(|| {
                    black_box(calc.compute(&topo.node_representations))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Ring", n_nodes),
            n_nodes,
            |b, &n| {
                let topo = create_ring(n, 42);
                b.iter(|| {
                    black_box(calc.compute(&topo.node_representations))
                });
            },
        );
    }

    group.finish();
}

fn bench_resonant_phi_configs(c: &mut Criterion) {
    let mut group = c.benchmark_group("ResonantPhi_Configs");
    group.sample_size(30);

    let topo = create_ring(8, 42);

    // Test different configurations
    group.bench_function("Fast", |b| {
        let calc = ResonantPhiCalculator::fast();
        b.iter(|| {
            black_box(calc.compute(&topo.node_representations))
        });
    });

    group.bench_function("Default", |b| {
        let calc = ResonantPhiCalculator::new();
        b.iter(|| {
            black_box(calc.compute(&topo.node_representations))
        });
    });

    group.bench_function("Accurate", |b| {
        let calc = ResonantPhiCalculator::accurate();
        b.iter(|| {
            black_box(calc.compute(&topo.node_representations))
        });
    });

    group.bench_function("Sequential", |b| {
        let calc = ResonantPhiCalculator::with_config(ResonantConfig::sequential());
        b.iter(|| {
            black_box(calc.compute(&topo.node_representations))
        });
    });

    group.finish();
}

// =============================================================================
// COMPARISON: REAL PHI vs RESONANT PHI
// =============================================================================

fn bench_phi_method_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Phi_Methods");
    group.sample_size(50);

    let topo = create_ring(8, 42);

    // RealPhi (algebraic connectivity - O(n³) eigenvalue)
    group.bench_function("RealPhi_8nodes", |b| {
        let calc = RealPhiCalculator::new();
        b.iter(|| {
            black_box(calc.compute(&topo.node_representations))
        });
    });

    // ResonantPhi (SIMD-optimized resonator - O(n log N))
    group.bench_function("ResonantPhi_8nodes", |b| {
        let calc = ResonantPhiCalculator::fast();
        b.iter(|| {
            black_box(calc.compute(&topo.node_representations))
        });
    });

    group.finish();
}

// =============================================================================
// HYPERCUBE DIMENSION SCALING
// =============================================================================

fn bench_hypercube_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hypercube_Scaling");
    group.sample_size(20);

    let calc = RealPhiCalculator::new();

    // 3D Hypercube (8 nodes)
    group.bench_function("3D_8nodes", |b| {
        let topo = create_hypercube_3d(42);
        b.iter(|| {
            black_box(calc.compute(&topo.node_representations))
        });
    });

    // 4D Hypercube (16 nodes)
    group.bench_function("4D_16nodes", |b| {
        let topo = create_hypercube_4d(42);
        b.iter(|| {
            black_box(calc.compute(&topo.node_representations))
        });
    });

    // 5D Hypercube (32 nodes)
    group.bench_function("5D_32nodes", |b| {
        let topo = create_hypercube_5d(42);
        b.iter(|| {
            black_box(calc.compute(&topo.node_representations))
        });
    });

    group.finish();
}

// =============================================================================
// LARGE TOPOLOGY BENCHMARK
// =============================================================================

fn bench_large_topologies(c: &mut Criterion) {
    let mut group = c.benchmark_group("Large_Topologies");
    group.sample_size(10);  // Fewer samples for large topologies

    let calc_fast = ResonantPhiCalculator::fast();

    // 64 nodes (6D hypercube)
    group.bench_function("Hypercube_6D_64nodes", |b| {
        let topo = ConsciousnessTopology::hypercube(6, HDC_DIMENSION, 42);
        b.iter(|| {
            black_box(calc_fast.compute(&topo.node_representations))
        });
    });

    // 64 node ring
    group.bench_function("Ring_64nodes", |b| {
        let topo = create_ring(64, 42);
        b.iter(|| {
            black_box(calc_fast.compute(&topo.node_representations))
        });
    });

    group.finish();
}

// =============================================================================
// CRITERION CONFIGURATION
// =============================================================================

criterion_group!(
    benches,
    bench_real_phi_topology_scaling,
    bench_real_phi_topology_types,
    bench_resonant_phi_scaling,
    bench_resonant_phi_configs,
    bench_phi_method_comparison,
    bench_hypercube_dimensions,
    bench_large_topologies,
);

criterion_main!(benches);
