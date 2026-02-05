//! Streaming Inference Benchmarks
//!
//! Comprehensive benchmarks for real-time streaming inference:
//! - Throughput (samples/second)
//! - Latency (p50, p95, p99)
//! - Memory usage over time
//! - Backpressure handling
//!
//! ## Usage
//!
//! ```bash
//! CARGO_TARGET_DIR=/tmp/symthaea-streaming cargo bench --bench streaming
//! CARGO_TARGET_DIR=/tmp/symthaea-streaming cargo bench --bench streaming -- throughput
//! CARGO_TARGET_DIR=/tmp/symthaea-streaming cargo bench --bench streaming -- latency
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array1;
use std::time::{Duration, Instant};

use symthaea::dynamics::cfc::{CfCCell, CfCConfig, CfCNetwork, CfCNetworkConfig};
use symthaea::inference::{
    benchmark_throughput, default_streaming, high_capacity_streaming, low_latency_streaming,
    BenchmarkResults, StreamingConfig, StreamingInference, StreamingOutput,
};

// =============================================================================
// TEST UTILITIES
// =============================================================================

fn make_input(dim: usize, seed: u64) -> Array1<f32> {
    Array1::from_shape_fn(dim, |i| ((i as u64 + seed) as f32 * 0.1).sin())
}

fn make_network(input_dim: usize, hidden_dim: usize, output_dim: usize, layers: usize) -> CfCNetwork {
    let config = CfCNetworkConfig {
        input_dim,
        hidden_dim,
        num_layers: layers,
        output_dim,
        cell_config: CfCConfig {
            input_dim,
            hidden_dim,
            use_backbone: false,
            ..Default::default()
        },
        ..Default::default()
    };
    CfCNetwork::new(config)
}

// =============================================================================
// THROUGHPUT BENCHMARKS
// =============================================================================

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_throughput");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(10));

    // Benchmark different configurations
    for (name, batch_accum) in [
        ("single", 1),
        ("batch_4", 4),
        ("batch_8", 8),
        ("batch_16", 16),
    ] {
        let config = StreamingConfig {
            batch_accumulation: batch_accum,
            warmup_samples: batch_accum * 2,
            max_latency_ms: 1000, // Disable timeout forcing
            ..Default::default()
        };

        let network = make_network(64, 128, 16, 2);
        let streamer = StreamingInference::new(network, config);

        group.throughput(Throughput::Elements(1000));
        group.bench_with_input(
            BenchmarkId::new("config", name),
            &1000usize,
            |b, &num_samples| {
                b.iter(|| {
                    streamer.reset();
                    for i in 0..num_samples {
                        streamer.push(make_input(64, i as u64));
                    }
                    // Drain all outputs
                    while let Some(_) = streamer.poll() {}
                    black_box(streamer.stats().total_outputs)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// LATENCY BENCHMARKS
// =============================================================================

fn bench_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_latency");
    group.sample_size(100);

    // Low latency configuration
    let low_lat_config = StreamingConfig::low_latency();
    let network = make_network(64, 64, 16, 1);
    let low_lat_streamer = StreamingInference::new(network, low_lat_config);

    group.bench_function("low_latency/push_poll", |b| {
        // Warmup
        for i in 0..10 {
            low_lat_streamer.push(make_input(64, i));
        }
        while let Some(_) = low_lat_streamer.poll() {}

        b.iter(|| {
            let start = Instant::now();
            low_lat_streamer.push(make_input(64, 0));
            let output = low_lat_streamer.poll();
            let elapsed = start.elapsed();
            black_box((output, elapsed))
        });
    });

    // Balanced configuration
    let balanced_config = StreamingConfig::balanced();
    let network = make_network(64, 128, 16, 2);
    let balanced_streamer = StreamingInference::new(network, balanced_config);

    group.bench_function("balanced/batch_8", |b| {
        // Warmup
        for i in 0..16 {
            balanced_streamer.push(make_input(64, i));
        }
        while let Some(_) = balanced_streamer.poll() {}

        b.iter(|| {
            let start = Instant::now();
            for i in 0..8 {
                balanced_streamer.push(make_input(64, i));
            }
            let output = balanced_streamer.poll();
            let elapsed = start.elapsed();
            black_box((output, elapsed))
        });
    });

    group.finish();
}

// =============================================================================
// LATENCY PERCENTILE BENCHMARKS
// =============================================================================

fn bench_latency_percentiles(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_latency_percentiles");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    for (name, config_fn) in [
        ("low_latency", StreamingConfig::low_latency as fn() -> StreamingConfig),
        ("balanced", StreamingConfig::balanced as fn() -> StreamingConfig),
        ("high_throughput", StreamingConfig::high_throughput as fn() -> StreamingConfig),
    ] {
        let config = config_fn();
        let network = make_network(64, 128, 16, 2);
        let streamer = StreamingInference::new(network, config);

        group.bench_function(BenchmarkId::new("percentiles", name), |b| {
            b.iter(|| {
                streamer.reset();
                let results = benchmark_throughput(&streamer, 64, 1000);
                black_box((
                    results.latency_p50_us,
                    results.latency_p95_us,
                    results.latency_p99_us,
                ))
            });
        });
    }

    group.finish();
}

// =============================================================================
// NETWORK SIZE SCALING BENCHMARKS
// =============================================================================

fn bench_network_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_network_scaling");
    group.sample_size(20);

    // Scale hidden dimension
    for hidden_dim in [32, 64, 128, 256] {
        let config = StreamingConfig {
            batch_accumulation: 1,
            warmup_samples: 0,
            ..Default::default()
        };
        let network = make_network(64, hidden_dim, 16, 2);
        let streamer = StreamingInference::new(network, config);

        group.throughput(Throughput::Elements(100));
        group.bench_with_input(
            BenchmarkId::new("hidden_dim", hidden_dim),
            &100usize,
            |b, &n| {
                b.iter(|| {
                    streamer.reset();
                    for i in 0..n {
                        streamer.push(make_input(64, i as u64));
                    }
                    while let Some(_) = streamer.poll() {}
                    black_box(streamer.stats().total_outputs)
                });
            },
        );
    }

    // Scale number of layers
    for num_layers in [1, 2, 4, 6] {
        let config = StreamingConfig {
            batch_accumulation: 1,
            warmup_samples: 0,
            ..Default::default()
        };
        let network = make_network(64, 128, 16, num_layers);
        let streamer = StreamingInference::new(network, config);

        group.throughput(Throughput::Elements(100));
        group.bench_with_input(
            BenchmarkId::new("num_layers", num_layers),
            &100usize,
            |b, &n| {
                b.iter(|| {
                    streamer.reset();
                    for i in 0..n {
                        streamer.push(make_input(64, i as u64));
                    }
                    while let Some(_) = streamer.poll() {}
                    black_box(streamer.stats().total_outputs)
                });
            },
        );
    }

    group.finish();
}

// =============================================================================
// BACKPRESSURE BENCHMARKS
// =============================================================================

fn bench_backpressure(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_backpressure");
    group.sample_size(20);

    // Test with dropping enabled
    let drop_config = StreamingConfig {
        batch_accumulation: 1,
        warmup_samples: 0,
        max_output_queue: 10,
        drop_on_backpressure: true,
        ..Default::default()
    };
    let network = make_network(64, 64, 16, 1);
    let drop_streamer = StreamingInference::new(network, drop_config);

    group.bench_function("drop_mode/1000_samples", |b| {
        b.iter(|| {
            drop_streamer.reset();
            for i in 0..1000 {
                drop_streamer.push(make_input(64, i));
            }
            let stats = drop_streamer.stats();
            black_box((stats.outputs_dropped, stats.total_outputs))
        });
    });

    // Test without dropping
    let queue_config = StreamingConfig {
        batch_accumulation: 1,
        warmup_samples: 0,
        max_output_queue: 10,
        drop_on_backpressure: false,
        ..Default::default()
    };
    let network = make_network(64, 64, 16, 1);
    let queue_streamer = StreamingInference::new(network, queue_config);

    group.bench_function("queue_mode/1000_samples", |b| {
        b.iter(|| {
            queue_streamer.reset();
            for i in 0..1000 {
                queue_streamer.push(make_input(64, i));
            }
            // Drain queue
            while let Some(_) = queue_streamer.poll() {}
            black_box(queue_streamer.stats().total_outputs)
        });
    });

    group.finish();
}

// =============================================================================
// MEMORY USAGE BENCHMARKS
// =============================================================================

fn bench_memory_stability(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_memory");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));

    // Long-running stability test
    let config = StreamingConfig {
        buffer_size: 1024,
        batch_accumulation: 8,
        warmup_samples: 8,
        ..Default::default()
    };
    let network = make_network(64, 128, 16, 2);
    let streamer = StreamingInference::new(network, config);

    group.bench_function("long_run/10k_samples", |b| {
        b.iter(|| {
            streamer.reset();
            for i in 0..10_000 {
                streamer.push(make_input(64, i));
                // Periodically drain to simulate realistic usage
                if i % 100 == 0 {
                    while let Some(_) = streamer.poll() {}
                }
            }
            while let Some(_) = streamer.poll() {}
            black_box(streamer.stats())
        });
    });

    group.finish();
}

// =============================================================================
// BATCH VS SINGLE COMPARISON
// =============================================================================

fn bench_batch_vs_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_batch_comparison");
    group.sample_size(50);

    let num_inputs = 100usize;
    let inputs: Vec<_> = (0..num_inputs).map(|i| make_input(64, i as u64)).collect();

    // Single push
    let single_config = StreamingConfig {
        batch_accumulation: 1,
        warmup_samples: 0,
        ..Default::default()
    };
    let network = make_network(64, 128, 16, 2);
    let single_streamer = StreamingInference::new(network, single_config);

    group.throughput(Throughput::Elements(num_inputs as u64));
    group.bench_function("single_push", |b| {
        b.iter(|| {
            single_streamer.reset();
            for input in &inputs {
                single_streamer.push(input.clone());
            }
            while let Some(_) = single_streamer.poll() {}
            black_box(single_streamer.stats().total_outputs)
        });
    });

    // Batch push
    let batch_config = StreamingConfig {
        batch_accumulation: 8,
        warmup_samples: 0,
        ..Default::default()
    };
    let network = make_network(64, 128, 16, 2);
    let batch_streamer = StreamingInference::new(network, batch_config);

    group.bench_function("batch_push", |b| {
        b.iter(|| {
            batch_streamer.reset();
            batch_streamer.push_batch(inputs.clone());
            while let Some(_) = batch_streamer.poll() {}
            black_box(batch_streamer.stats().total_outputs)
        });
    });

    group.finish();
}

// =============================================================================
// CHECKPOINT OVERHEAD BENCHMARKS
// =============================================================================

fn bench_checkpointing(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_checkpointing");
    group.sample_size(20);

    // Without checkpointing
    let no_checkpoint_config = StreamingConfig {
        batch_accumulation: 4,
        warmup_samples: 0,
        enable_checkpoints: false,
        ..Default::default()
    };
    let network = make_network(64, 128, 16, 2);
    let no_cp_streamer = StreamingInference::new(network, no_checkpoint_config);

    group.throughput(Throughput::Elements(1000));
    group.bench_function("no_checkpoints", |b| {
        b.iter(|| {
            no_cp_streamer.reset();
            for i in 0..1000 {
                no_cp_streamer.push(make_input(64, i));
            }
            while let Some(_) = no_cp_streamer.poll() {}
            black_box(no_cp_streamer.stats().total_outputs)
        });
    });

    // With checkpointing (every 100 samples)
    let checkpoint_config = StreamingConfig {
        batch_accumulation: 4,
        warmup_samples: 0,
        enable_checkpoints: true,
        checkpoint_interval: 100,
        ..Default::default()
    };
    let network = make_network(64, 128, 16, 2);
    let cp_streamer = StreamingInference::new(network, checkpoint_config);

    group.bench_function("checkpoints_100", |b| {
        b.iter(|| {
            cp_streamer.reset();
            for i in 0..1000 {
                cp_streamer.push(make_input(64, i));
            }
            while let Some(_) = cp_streamer.poll() {}
            black_box((cp_streamer.stats().total_outputs, cp_streamer.checkpoint_sequences().len()))
        });
    });

    group.finish();
}

// =============================================================================
// UTILITY FUNCTIONS BENCHMARKS
// =============================================================================

fn bench_utility_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_utilities");
    group.sample_size(50);

    // Default streaming
    group.bench_function("default_streaming/create", |b| {
        b.iter(|| black_box(default_streaming(64, 16)))
    });

    // Low latency streaming
    group.bench_function("low_latency_streaming/create", |b| {
        b.iter(|| black_box(low_latency_streaming(64, 16)))
    });

    // High capacity streaming
    group.bench_function("high_capacity_streaming/create", |b| {
        b.iter(|| black_box(high_capacity_streaming(64, 16)))
    });

    group.finish();
}

// =============================================================================
// CRITERION GROUPS
// =============================================================================

criterion_group!(
    throughput_benches,
    bench_throughput,
    bench_network_scaling,
);

criterion_group!(
    latency_benches,
    bench_latency,
    bench_latency_percentiles,
);

criterion_group!(
    stress_benches,
    bench_backpressure,
    bench_memory_stability,
    bench_batch_vs_single,
);

criterion_group!(
    overhead_benches,
    bench_checkpointing,
    bench_utility_functions,
);

criterion_main!(
    throughput_benches,
    latency_benches,
    stress_benches,
    overhead_benches,
);
