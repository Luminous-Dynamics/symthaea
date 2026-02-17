//! Benchmarks for aggregation algorithms.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fl_aggregator::{
    aggregator::{Aggregator, AggregatorConfig},
    byzantine::{ByzantineAggregator, Defense, DefenseConfig},
    compression::{CompressionConfig, GradientCompressor},
    Gradient,
};
use ndarray::Array1;
use rand::Rng;

fn generate_gradients(num_nodes: usize, dim: usize) -> Vec<Gradient> {
    let mut rng = rand::thread_rng();
    (0..num_nodes)
        .map(|_| Array1::from_iter((0..dim).map(|_| rng.gen_range(-1.0..1.0))))
        .collect()
}

fn bench_byzantine_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("byzantine_defense");

    for num_nodes in [10, 50, 100] {
        let gradients = generate_gradients(num_nodes, 10000);

        // FedAvg (baseline)
        group.bench_with_input(
            BenchmarkId::new("fedavg", num_nodes),
            &gradients,
            |b, grads| {
                let aggregator =
                    ByzantineAggregator::new(DefenseConfig::with_defense(Defense::FedAvg));
                b.iter(|| aggregator.aggregate(black_box(grads)))
            },
        );

        // Krum
        if num_nodes >= 5 {
            group.bench_with_input(
                BenchmarkId::new("krum", num_nodes),
                &gradients,
                |b, grads| {
                    let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(
                        Defense::Krum { f: 1 },
                    ));
                    b.iter(|| aggregator.aggregate(black_box(grads)))
                },
            );
        }

        // Median
        group.bench_with_input(
            BenchmarkId::new("median", num_nodes),
            &gradients,
            |b, grads| {
                let aggregator =
                    ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Median));
                b.iter(|| aggregator.aggregate(black_box(grads)))
            },
        );

        // TrimmedMean
        group.bench_with_input(
            BenchmarkId::new("trimmed_mean", num_nodes),
            &gradients,
            |b, grads| {
                let aggregator = ByzantineAggregator::new(DefenseConfig::with_defense(
                    Defense::TrimmedMean { beta: 0.1 },
                ));
                b.iter(|| aggregator.aggregate(black_box(grads)))
            },
        );
    }

    group.finish();
}

fn bench_gradient_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_dimension");

    for dim in [1000, 10000, 100000, 1000000] {
        group.throughput(Throughput::Elements(dim as u64));

        let gradients = generate_gradients(10, dim);

        group.bench_with_input(BenchmarkId::new("krum", dim), &gradients, |b, grads| {
            let aggregator =
                ByzantineAggregator::new(DefenseConfig::with_defense(Defense::Krum { f: 1 }));
            b.iter(|| aggregator.aggregate(black_box(grads)))
        });
    }

    group.finish();
}

fn bench_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression");

    for compression_ratio in [2.0, 10.0, 40.0] {
        let gradient = Array1::from_iter((0..100000).map(|i| (i as f32).sin()));

        group.bench_with_input(
            BenchmarkId::new("compress", format!("{}x", compression_ratio as u32)),
            &gradient,
            |b, grad| {
                let config = CompressionConfig {
                    compression_ratio,
                    quantization_bits: 8,
                    error_feedback: false,
                };
                let mut compressor = GradientCompressor::new(config);
                b.iter(|| compressor.compress(black_box(grad)))
            },
        );
    }

    group.finish();
}

fn bench_full_aggregation_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_cycle");

    for num_nodes in [10, 50, 100] {
        let gradients = generate_gradients(num_nodes, 10000);

        group.bench_with_input(
            BenchmarkId::new("aggregation_cycle", num_nodes),
            &gradients,
            |b, grads| {
                b.iter(|| {
                    let config = AggregatorConfig::default()
                        .with_expected_nodes(grads.len())
                        .with_defense(Defense::Krum { f: 1 });

                    let mut aggregator = Aggregator::new(config);

                    for (i, grad) in grads.iter().enumerate() {
                        aggregator
                            .submit(format!("node{}", i), grad.clone())
                            .unwrap();
                    }

                    aggregator.finalize_round().unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    group.sample_size(10);

    // Large scale test: 1000 nodes, 1M parameters
    let num_nodes = 1000;
    let dim = 1_000_000;

    group.bench_function("1000_nodes_1M_params", |b| {
        b.iter(|| {
            let config = AggregatorConfig::default()
                .with_expected_nodes(num_nodes)
                .with_defense(Defense::FedAvg); // Use FedAvg for memory test

            let mut aggregator = Aggregator::new(config);
            let mut rng = rand::thread_rng();

            for i in 0..num_nodes {
                let gradient = Array1::from_iter((0..dim).map(|_| rng.gen_range(-1.0f32..1.0)));
                aggregator
                    .submit(format!("node{}", i), gradient)
                    .unwrap();
            }

            aggregator.finalize_round().unwrap()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_byzantine_algorithms,
    bench_gradient_dimensions,
    bench_compression,
    bench_full_aggregation_cycle,
);

// Memory benchmark is separate due to long runtime
criterion_group! {
    name = memory_benches;
    config = Criterion::default().sample_size(10);
    targets = bench_memory_efficiency
}

criterion_main!(benches);
