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

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use symthaea::hdc::{
    consciousness_topology_generators::ConsciousnessTopology,
    spectral_connectivity::RealPhiCalculator,
    real_hv::RealHV,
    HV16, HDC_DIMENSION,
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
                let calc = RealPhiCalculator::new();

                b.iter(|| {
                    // Create and discard many topologies
                    let results: Vec<_> = (0..n)
                        .map(|i| {
                            let topo = ConsciousnessTopology::random(8, HDC_DIMENSION, i as u64);
                            calc.compute(&topo.node_representations)
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

    let calc = RealPhiCalculator::new();

    for n_nodes in [16, 32, 64, 128].iter() {
        group.bench_with_input(
            BenchmarkId::new("topology_nodes", n_nodes),
            n_nodes,
            |b, &n| {
                b.iter(|| {
                    let topo = ConsciousnessTopology::ring(n, HDC_DIMENSION, 42);
                    let result = calc.compute(&topo.node_representations);
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
                        .map(|i| RealHV::random(HDC_DIMENSION, i as u64))
                        .collect();
                    black_box(vectors)
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("bind_chain", n_vectors),
            n_vectors,
            |b, &n| {
                let vectors: Vec<_> = (0..n)
                    .map(|i| HV16::random(i as u64))
                    .collect();

                b.iter(|| {
                    let mut result = vectors[0].clone();
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

    let calc = RealPhiCalculator::new();

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
                        .map(|topo| calc.compute(&topo.node_representations))
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

    let calc = RealPhiCalculator::new();

    // Test hypercube dimensions 1-7
    for dim in 1..=7 {
        let n_nodes = 1 << dim; // 2^dim nodes

        if n_nodes <= 128 {
            // Limit for reasonable benchmark time
            group.bench_with_input(
                BenchmarkId::new("hypercube_dim", dim),
                &dim,
                |b, &d| {
                    b.iter(|| {
                        let topo = ConsciousnessTopology::hypercube(d, HDC_DIMENSION, 42);
                        let result = calc.compute(&topo.node_representations);
                        black_box(result)
                    })
                },
            );
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

    let calc = RealPhiCalculator::new();
    let n_nodes = 8;

    // Test multiple topology types
    let topologies = vec![
        ("random", ConsciousnessTopology::random(n_nodes, HDC_DIMENSION, 42)),
        ("star", ConsciousnessTopology::star(n_nodes, HDC_DIMENSION, 42)),
        ("ring", ConsciousnessTopology::ring(n_nodes, HDC_DIMENSION, 42)),
        ("line", ConsciousnessTopology::line(n_nodes, HDC_DIMENSION, 42)),
        ("dense_network", ConsciousnessTopology::dense_network(n_nodes, HDC_DIMENSION, None, 42)),
        ("torus", ConsciousnessTopology::torus(3, 3, HDC_DIMENSION, 42)),
        ("hypercube_3d", ConsciousnessTopology::hypercube(3, HDC_DIMENSION, 42)),
        ("hypercube_4d", ConsciousnessTopology::hypercube(4, HDC_DIMENSION, 42)),
    ];

    for (name, topo) in topologies {
        group.bench_with_input(BenchmarkId::new("topology", name), &topo, |b, t| {
            b.iter(|| {
                let result = calc.compute(&t.node_representations);
                black_box(result)
            })
        });
    }

    group.finish();
}

// =============================================================================
// CONCURRENT CFC INFERENCE BENCHMARK
// =============================================================================

fn bench_concurrent_cfc_inference(c: &mut Criterion) {
    use std::thread;
    use symthaea::dynamics::cfc::{CfCNetwork, CfCNetworkConfig, CfCConfig};

    let mut group = c.benchmark_group("stress_concurrent_cfc");
    group.sample_size(10);

    for n_threads in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("threads", n_threads),
            n_threads,
            |b, &n| {
                b.iter(|| {
                    let handles: Vec<_> = (0..n)
                        .map(|_| {
                            thread::spawn(|| {
                                let cell_config = CfCConfig {
                                    input_dim: 64,
                                    hidden_dim: 32,
                                    ..Default::default()
                                };
                                let config = CfCNetworkConfig {
                                    input_dim: 64,
                                    hidden_dim: 32,
                                    num_layers: 2,
                                    output_dim: 16,
                                    cell_config,
                                    ..Default::default()
                                };
                                let mut network = CfCNetwork::new(config);
                                let input =
                                    ndarray::Array1::from_shape_fn(64, |i| (i as f32).sin());
                                for _ in 0..100 {
                                    black_box(network.forward(&input, 0.02));
                                }
                            })
                        })
                        .collect();
                    for h in handles {
                        h.join().unwrap();
                    }
                })
            },
        );
    }
    group.finish();
}

// =============================================================================
// CONCURRENT STREAMING BENCHMARK
// =============================================================================

fn bench_concurrent_streaming(c: &mut Criterion) {
    use std::thread;
    use symthaea::dynamics::cfc::{CfCNetwork, CfCNetworkConfig, CfCConfig};
    use symthaea::inference::{StreamingInference, StreamingConfig};

    let mut group = c.benchmark_group("stress_concurrent_streaming");
    group.sample_size(10);

    for n_pipelines in [1, 2, 4].iter() {
        group.bench_with_input(
            BenchmarkId::new("pipelines", n_pipelines),
            n_pipelines,
            |b, &n| {
                b.iter(|| {
                    let handles: Vec<_> = (0..n)
                        .map(|_| {
                            thread::spawn(|| {
                                let cell_config = CfCConfig {
                                    input_dim: 32,
                                    hidden_dim: 16,
                                    ..Default::default()
                                };
                                let network = CfCNetwork::new(CfCNetworkConfig {
                                    input_dim: 32,
                                    hidden_dim: 16,
                                    num_layers: 1,
                                    output_dim: 8,
                                    cell_config,
                                    ..Default::default()
                                });
                                let config = StreamingConfig {
                                    batch_accumulation: 2,
                                    warmup_samples: 4,
                                    ..Default::default()
                                };
                                let streamer = StreamingInference::new(network, config);

                                for i in 0..50 {
                                    let input = ndarray::Array1::from_shape_fn(32, |j| {
                                        ((j + i) as f32).cos()
                                    });
                                    streamer.push(input);
                                    while let Some(out) = streamer.poll() {
                                        black_box(out);
                                    }
                                }
                                while let Some(out) = streamer.flush() {
                                    black_box(out);
                                }
                            })
                        })
                        .collect();
                    for h in handles {
                        h.join().unwrap();
                    }
                })
            },
        );
    }
    group.finish();
}

// =============================================================================
// FEDERATED AGGREGATION SCALING BENCHMARK
// =============================================================================

fn bench_federated_aggregation_scaling(c: &mut Criterion) {
    use symthaea::swarm::{FederatedAggregator, GradientMessage};

    let mut group = c.benchmark_group("stress_federated_aggregation");
    group.sample_size(10);

    for n_nodes in [3, 5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("nodes", n_nodes),
            n_nodes,
            |b, &n| {
                b.iter(|| {
                    let initial_weights = vec![0.0f32; 200];
                    let mut aggregator =
                        FederatedAggregator::new(initial_weights).with_byzantine_tolerance(0.2);

                    // Submit gradients from n nodes
                    for i in 0..n {
                        let gradients: Vec<f32> =
                            (0..200).map(|j| (j as f32 * 0.01) + (i as f32 * 0.001)).collect();
                        let mut source_id = [0u8; 32];
                        source_id[0] = i as u8;
                        source_id[1] = (i >> 8) as u8;
                        let msg = GradientMessage::new(source_id, gradients, 0.8);
                        aggregator.receive_gradient(msg);
                    }

                    // Aggregate
                    black_box(aggregator.aggregate())
                })
            },
        );
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
    bench_concurrent_cfc_inference,
    bench_concurrent_streaming,
    bench_federated_aggregation_scaling,
);

criterion_main!(stress_benches);
