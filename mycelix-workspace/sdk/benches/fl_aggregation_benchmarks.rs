//! Federated Learning Aggregation Benchmarks
//!
//! Performance benchmarks for FL aggregation algorithms with varying participant counts.
//! Measures throughput, latency, and scaling characteristics.
//!
//! Run with: `cargo bench --bench fl_aggregation_benchmarks --features simulation`

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use mycelix_sdk::fl::{
    coordinate_median, fedavg, fedavg_optimized, krum, trimmed_mean, AggregationMethod, FLConfig,
    FLCoordinator, GradientAccumulator, GradientUpdate,
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Create test gradient updates with specified count and dimension
fn create_gradient_updates(participant_count: usize, gradient_dim: usize) -> Vec<GradientUpdate> {
    (0..participant_count)
        .map(|i| {
            let gradients: Vec<f64> = (0..gradient_dim)
                .map(|j| 0.1 * ((i + j) as f64 / gradient_dim as f64))
                .collect();
            GradientUpdate::new(
                format!("participant-{}", i),
                1, // round
                gradients,
                100 + i,                 // batch size
                0.5 - (i as f64 * 0.01), // loss
            )
        })
        .collect()
}

/// Create gradient updates with some Byzantine participants
fn create_updates_with_byzantine(
    honest_count: usize,
    byzantine_count: usize,
    gradient_dim: usize,
) -> Vec<GradientUpdate> {
    let mut updates = Vec::with_capacity(honest_count + byzantine_count);

    // Honest updates
    for i in 0..honest_count {
        let gradients: Vec<f64> = (0..gradient_dim)
            .map(|j| 0.1 * (j as f64 / gradient_dim as f64))
            .collect();
        updates.push(GradientUpdate::new(
            format!("honest-{}", i),
            1,
            gradients,
            100,
            0.5,
        ));
    }

    // Byzantine updates (outliers)
    for i in 0..byzantine_count {
        let gradients: Vec<f64> = (0..gradient_dim)
            .map(|j| 10.0 + (j as f64)) // Very large values
            .collect();
        updates.push(GradientUpdate::new(
            format!("byzantine-{}", i),
            1,
            gradients,
            100,
            0.5,
        ));
    }

    updates
}

// =============================================================================
// BASIC AGGREGATION BENCHMARKS
// =============================================================================

/// Benchmark FedAvg with different participant counts
fn benchmark_fedavg_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("fl_fedavg_scaling");
    let gradient_dim = 1000;

    for participant_count in [10, 50, 100, 500, 1000].iter() {
        let updates = create_gradient_updates(*participant_count, gradient_dim);

        group.throughput(Throughput::Elements(*participant_count as u64));
        group.bench_with_input(
            BenchmarkId::new("participants", participant_count),
            &updates,
            |b, updates| b.iter(|| fedavg(black_box(updates))),
        );
    }

    group.finish();
}

/// Benchmark optimized FedAvg vs standard
fn benchmark_fedavg_optimized(c: &mut Criterion) {
    let mut group = c.benchmark_group("fl_fedavg_comparison");
    let gradient_dim = 1000;
    let participant_count = 100;

    let updates = create_gradient_updates(participant_count, gradient_dim);

    group.bench_function("fedavg_standard", |b| {
        b.iter(|| fedavg(black_box(&updates)))
    });

    group.bench_function("fedavg_optimized", |b| {
        b.iter(|| fedavg_optimized(black_box(&updates)))
    });

    group.finish();
}

/// Benchmark gradient dimension scaling
fn benchmark_gradient_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("fl_gradient_dimension");
    let participant_count = 100;

    for gradient_dim in [100, 1000, 10000, 100000].iter() {
        let updates = create_gradient_updates(participant_count, *gradient_dim);

        group.throughput(Throughput::Elements(*gradient_dim as u64));
        group.bench_with_input(
            BenchmarkId::new("dimension", gradient_dim),
            &updates,
            |b, updates| b.iter(|| fedavg(black_box(updates))),
        );
    }

    group.finish();
}

// =============================================================================
// BYZANTINE-RESISTANT AGGREGATION BENCHMARKS
// =============================================================================

/// Benchmark trimmed mean with different trim ratios
fn benchmark_trimmed_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("fl_trimmed_mean");
    let gradient_dim = 1000;

    let updates = create_updates_with_byzantine(80, 20, gradient_dim);

    for trim_ratio in [0.1, 0.2, 0.3].iter() {
        group.bench_with_input(
            BenchmarkId::new("trim_ratio", format!("{:.1}", trim_ratio)),
            &(updates.clone(), *trim_ratio),
            |b, (updates, ratio)| b.iter(|| trimmed_mean(black_box(updates), black_box(*ratio))),
        );
    }

    group.finish();
}

/// Benchmark coordinate median
fn benchmark_coordinate_median(c: &mut Criterion) {
    let mut group = c.benchmark_group("fl_coordinate_median");
    let gradient_dim = 1000;

    for participant_count in [10, 50, 100, 500].iter() {
        let updates = create_updates_with_byzantine(
            (*participant_count as f64 * 0.8) as usize,
            (*participant_count as f64 * 0.2) as usize,
            gradient_dim,
        );

        group.throughput(Throughput::Elements(*participant_count as u64));
        group.bench_with_input(
            BenchmarkId::new("participants", participant_count),
            &updates,
            |b, updates| b.iter(|| coordinate_median(black_box(updates))),
        );
    }

    group.finish();
}

/// Benchmark Krum algorithm
fn benchmark_krum(c: &mut Criterion) {
    let mut group = c.benchmark_group("fl_krum");
    let gradient_dim = 1000;

    // Krum is O(n^2) in participants, so test smaller counts
    for participant_count in [10, 20, 50, 100].iter() {
        let byzantine_count = (*participant_count as f64 * 0.2) as usize;
        let updates = create_updates_with_byzantine(
            *participant_count - byzantine_count,
            byzantine_count,
            gradient_dim,
        );

        group.throughput(Throughput::Elements(*participant_count as u64));
        group.bench_with_input(
            BenchmarkId::new("participants", participant_count),
            &(updates.clone(), byzantine_count),
            |b, (updates, byz_count)| b.iter(|| krum(black_box(updates), black_box(*byz_count))),
        );
    }

    group.finish();
}

// =============================================================================
// STREAMING AGGREGATION BENCHMARKS
// =============================================================================

/// Benchmark gradient accumulator (streaming aggregation)
fn benchmark_gradient_accumulator(c: &mut Criterion) {
    let mut group = c.benchmark_group("fl_gradient_accumulator");
    let gradient_dim = 1000;

    for participant_count in [10, 50, 100, 500].iter() {
        let updates = create_gradient_updates(*participant_count, gradient_dim);

        group.throughput(Throughput::Elements(*participant_count as u64));
        group.bench_with_input(
            BenchmarkId::new("participants", participant_count),
            &updates,
            |b, updates| {
                b.iter(|| {
                    let mut accumulator = GradientAccumulator::new(gradient_dim);
                    for update in black_box(updates) {
                        accumulator.accumulate(black_box(update), 1.0);
                    }
                    accumulator.finalize()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// FL COORDINATOR BENCHMARKS
// =============================================================================

/// Benchmark full FL round execution
fn benchmark_fl_round(c: &mut Criterion) {
    let mut group = c.benchmark_group("fl_coordinator_round");
    let gradient_dim = 1000;

    for participant_count in [10, 50, 100].iter() {
        group.throughput(Throughput::Elements(*participant_count as u64));
        group.bench_with_input(
            BenchmarkId::new("participants", participant_count),
            participant_count,
            |b, &count| {
                b.iter_batched(
                    || {
                        // Setup: create coordinator and register participants
                        let config = FLConfig {
                            min_participants: count as usize,
                            max_participants: count as usize + 10,
                            round_timeout_ms: 60_000,
                            byzantine_tolerance: 0.33,
                            aggregation_method: AggregationMethod::FedAvg,
                            trust_threshold: 0.5,
                        };
                        let mut coordinator = FLCoordinator::new(config);

                        for i in 0..count {
                            coordinator.register_participant(format!("p{}", i));
                        }
                        coordinator.start_round().unwrap();

                        let updates = create_gradient_updates(count as usize, gradient_dim);
                        (coordinator, updates)
                    },
                    |(mut coordinator, updates)| {
                        // Benchmark: submit updates and aggregate
                        for update in black_box(updates) {
                            coordinator.submit_update(black_box(update));
                        }
                        coordinator.aggregate_round()
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}

/// Benchmark different aggregation methods in coordinator
fn benchmark_aggregation_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("fl_aggregation_methods");
    let gradient_dim = 1000;
    let participant_count = 100;

    let methods = [
        ("FedAvg", AggregationMethod::FedAvg),
        ("TrimmedMean", AggregationMethod::TrimmedMean),
        ("Median", AggregationMethod::Median),
        ("Krum", AggregationMethod::Krum),
        ("TrustWeighted", AggregationMethod::TrustWeighted),
    ];

    for (name, method) in methods.iter() {
        group.bench_with_input(BenchmarkId::new("method", name), method, |b, method| {
            b.iter_batched(
                || {
                    let config = FLConfig {
                        min_participants: participant_count,
                        max_participants: participant_count + 10,
                        round_timeout_ms: 60_000,
                        byzantine_tolerance: 0.33,
                        aggregation_method: method.clone(),
                        trust_threshold: 0.5,
                    };
                    let mut coordinator = FLCoordinator::new(config);

                    for i in 0..participant_count {
                        coordinator.register_participant(format!("p{}", i));
                    }
                    coordinator.start_round().unwrap();

                    let updates = create_gradient_updates(participant_count, gradient_dim);
                    (coordinator, updates)
                },
                |(mut coordinator, updates)| {
                    for update in black_box(updates) {
                        coordinator.submit_update(black_box(update));
                    }
                    coordinator.aggregate_round()
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

// =============================================================================
// CRITERION GROUPS
// =============================================================================

criterion_group!(
    basic_aggregation,
    benchmark_fedavg_scaling,
    benchmark_fedavg_optimized,
    benchmark_gradient_dimension_scaling,
);

criterion_group!(
    byzantine_resistant,
    benchmark_trimmed_mean,
    benchmark_coordinate_median,
    benchmark_krum,
);

criterion_group!(streaming, benchmark_gradient_accumulator,);

criterion_group!(
    coordinator,
    benchmark_fl_round,
    benchmark_aggregation_methods,
);

criterion_main!(
    basic_aggregation,
    byzantine_resistant,
    streaming,
    coordinator
);
