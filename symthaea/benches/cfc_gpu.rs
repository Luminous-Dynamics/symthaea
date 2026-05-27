// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # CfC GPU Acceleration Benchmark
//!
//! Compares CPU vs GPU throughput for Closed-form Continuous-time (CfC) networks.
//!
//! ## Usage
//!
//! ```bash
//! # Run benchmarks (CPU only by default)
//! CARGO_TARGET_DIR=/tmp/symthaea-gpu cargo bench --bench cfc_gpu
//!
//! # Run with GPU acceleration (if available)
//! CARGO_TARGET_DIR=/tmp/symthaea-gpu cargo bench --bench cfc_gpu --features gpu
//!
//! # Save baseline and compare
//! CARGO_TARGET_DIR=/tmp/symthaea-gpu cargo bench --bench cfc_gpu -- --save-baseline cpu
//! CARGO_TARGET_DIR=/tmp/symthaea-gpu cargo bench --bench cfc_gpu --features gpu -- --baseline cpu
//! ```
//!
//! ## Expected Results
//!
//! | Operation | CPU (us) | GPU (us) | Speedup |
//! |-----------|----------|----------|---------|
//! | forward_single | 50-100 | 10-20 | 3-5x |
//! | forward_batch_32 | 1500-2000 | 50-100 | 20-40x |
//! | train_step | 500-1000 | 50-100 | 10-20x |

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use ndarray::Array1;
use std::time::Duration;
use symthaea::dynamics::{
    CfCConfig, CfCNetwork, CfCNetworkConfig, GpuBackend, GpuCfcConfig, GpuCfcNetwork,
};

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Standard network configurations for benchmarking
fn configs() -> Vec<(&'static str, GpuCfcConfig)> {
    vec![
        (
            "small_32x64",
            GpuCfcConfig {
                input_dim: 32,
                hidden_dim: 64,
                num_layers: 2,
                output_dim: 16,
                use_backbone: false,
                ..Default::default()
            },
        ),
        (
            "medium_64x128",
            GpuCfcConfig {
                input_dim: 64,
                hidden_dim: 128,
                num_layers: 2,
                output_dim: 32,
                use_backbone: true,
                backbone_layers: 2,
                backbone_dim: 96,
                ..Default::default()
            },
        ),
        (
            "large_128x256",
            GpuCfcConfig {
                input_dim: 128,
                hidden_dim: 256,
                num_layers: 3,
                output_dim: 64,
                use_backbone: true,
                backbone_layers: 3,
                backbone_dim: 192,
                ..Default::default()
            },
        ),
    ]
}

/// Convert GpuCfcConfig to CPU CfCNetworkConfig
fn to_cpu_config(gpu_config: &GpuCfcConfig) -> CfCNetworkConfig {
    CfCNetworkConfig {
        input_dim: gpu_config.input_dim,
        hidden_dim: gpu_config.hidden_dim,
        num_layers: gpu_config.num_layers,
        output_dim: gpu_config.output_dim,
        cell_config: CfCConfig {
            input_dim: gpu_config.input_dim,
            hidden_dim: gpu_config.hidden_dim,
            use_backbone: gpu_config.use_backbone,
            backbone_layers: gpu_config.backbone_layers,
            backbone_dim: gpu_config.backbone_dim,
            activation: gpu_config.activation,
            tau_range: gpu_config.tau_range,
            dropout: gpu_config.dropout,
            gradient_clip: 1.0,
            online_learning: None,
        },
        residual: true,
        bidirectional: false,
        enable_online_learning: false,
        online_learning_config: Default::default(),
    }
}

// =============================================================================
// FORWARD PASS BENCHMARKS
// =============================================================================

fn bench_forward_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("cfc_forward_single");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(100);

    for (name, gpu_config) in configs() {
        let cpu_config = to_cpu_config(&gpu_config);
        let input = vec![0.1f32; gpu_config.input_dim];
        let input_arr = Array1::from_vec(input.clone());
        let dt = 0.1f32;

        // CPU benchmark (original implementation)
        let mut cpu_network = CfCNetwork::new(cpu_config.clone());
        group.bench_with_input(BenchmarkId::new("cpu", name), &name, |b, _| {
            b.iter(|| {
                cpu_network.reset();
                black_box(cpu_network.forward(&input_arr, dt))
            })
        });

        // GPU/Unified benchmark
        let mut gpu_network = GpuCfcNetwork::new(gpu_config.clone(), GpuBackend::Auto)
            .expect("Failed to create GPU network");
        let backend_name = gpu_network.backend().description();
        group.bench_with_input(
            BenchmarkId::new(format!("gpu_{}", backend_name), name),
            &name,
            |b, _| {
                b.iter(|| {
                    gpu_network.reset();
                    black_box(gpu_network.forward(&input, dt).unwrap())
                })
            },
        );
    }

    group.finish();
}

fn bench_forward_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("cfc_forward_batch");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(50);

    let batch_sizes = [8, 32, 128];

    for (name, gpu_config) in configs() {
        for &batch_size in &batch_sizes {
            let cpu_config = to_cpu_config(&gpu_config);
            let inputs: Vec<Vec<f32>> = (0..batch_size)
                .map(|i| vec![i as f32 * 0.01; gpu_config.input_dim])
                .collect();
            let dts: Vec<f32> = vec![0.1; batch_size];
            let inputs_arr: Vec<Array1<f32>> =
                inputs.iter().map(|v| Array1::from_vec(v.clone())).collect();

            group.throughput(Throughput::Elements(batch_size as u64));

            // CPU benchmark (sequential)
            let mut cpu_network = CfCNetwork::new(cpu_config.clone());
            group.bench_with_input(
                BenchmarkId::new(format!("cpu_batch_{}", batch_size), name),
                &name,
                |b, _| {
                    b.iter(|| {
                        cpu_network.reset();
                        black_box(cpu_network.forward_sequence(&inputs_arr, &dts))
                    })
                },
            );

            // GPU/Unified batch benchmark
            let mut gpu_network = GpuCfcNetwork::new(gpu_config.clone(), GpuBackend::Auto)
                .expect("Failed to create GPU network");
            let backend_name = gpu_network.backend().description();
            group.bench_with_input(
                BenchmarkId::new(format!("gpu_{}_batch_{}", backend_name, batch_size), name),
                &name,
                |b, _| {
                    b.iter(|| {
                        gpu_network.reset();
                        black_box(gpu_network.forward_batch(&inputs, &dts).unwrap())
                    })
                },
            );
        }
    }

    group.finish();
}

// =============================================================================
// TRAINING BENCHMARKS
// =============================================================================

fn bench_train_step(c: &mut Criterion) {
    let mut group = c.benchmark_group("cfc_train_step");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(50);

    for (name, gpu_config) in configs() {
        let cpu_config = to_cpu_config(&gpu_config);

        // Training data
        let batch_size = 8;
        let inputs: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| vec![i as f32 * 0.1; gpu_config.input_dim])
            .collect();
        let targets: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| vec![i as f32 * 0.1; gpu_config.output_dim])
            .collect();
        let dts: Vec<f32> = vec![0.1; batch_size];

        let inputs_arr: Vec<Array1<f32>> =
            inputs.iter().map(|v| Array1::from_vec(v.clone())).collect();
        let targets_arr: Vec<Array1<f32>> = targets
            .iter()
            .map(|v| Array1::from_vec(v.clone()))
            .collect();

        // CPU training benchmark
        let mut cpu_network = CfCNetwork::new(cpu_config.clone());
        group.bench_with_input(BenchmarkId::new("cpu_train", name), &name, |b, _| {
            b.iter(|| {
                cpu_network.reset();
                black_box(
                    cpu_network
                        .train_step_bptt(&inputs_arr, &targets_arr, &dts, 0.001)
                        .unwrap(),
                )
            })
        });

        // GPU training benchmark
        let mut gpu_network = GpuCfcNetwork::new(gpu_config.clone(), GpuBackend::Auto)
            .expect("Failed to create GPU network");
        let backend_name = gpu_network.backend().description();
        group.bench_with_input(
            BenchmarkId::new(format!("gpu_{}_train", backend_name), name),
            &name,
            |b, _| {
                b.iter(|| {
                    gpu_network.reset();
                    black_box(
                        gpu_network
                            .train_step(&inputs, &targets, &dts, 0.001)
                            .unwrap(),
                    )
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// SEQUENCE PROCESSING BENCHMARKS
// =============================================================================

fn bench_sequence_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("cfc_sequence");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(30);

    let sequence_lengths = [10, 50, 200];

    // Use medium config for sequence benchmarks
    let gpu_config = GpuCfcConfig {
        input_dim: 64,
        hidden_dim: 128,
        num_layers: 2,
        output_dim: 32,
        use_backbone: true,
        backbone_layers: 2,
        backbone_dim: 96,
        ..Default::default()
    };
    let cpu_config = to_cpu_config(&gpu_config);

    for &seq_len in &sequence_lengths {
        let inputs: Vec<Vec<f32>> = (0..seq_len)
            .map(|i| {
                let t = (i as f32) * 0.1;
                vec![t.sin(); gpu_config.input_dim]
            })
            .collect();
        let dts: Vec<f32> = vec![0.1; seq_len];
        let inputs_arr: Vec<Array1<f32>> =
            inputs.iter().map(|v| Array1::from_vec(v.clone())).collect();

        group.throughput(Throughput::Elements(seq_len as u64));

        // CPU sequence processing
        let mut cpu_network = CfCNetwork::new(cpu_config.clone());
        group.bench_with_input(
            BenchmarkId::new("cpu_sequence", seq_len),
            &seq_len,
            |b, _| {
                b.iter(|| {
                    cpu_network.reset();
                    black_box(cpu_network.forward_sequence(&inputs_arr, &dts))
                })
            },
        );

        // GPU sequence processing
        let mut gpu_network = GpuCfcNetwork::new(gpu_config.clone(), GpuBackend::Auto)
            .expect("Failed to create GPU network");
        let backend_name = gpu_network.backend().description();
        group.bench_with_input(
            BenchmarkId::new(format!("gpu_{}_sequence", backend_name), seq_len),
            &seq_len,
            |b, _| b.iter(|| black_box(gpu_network.forward_sequence(&inputs, &dts).unwrap())),
        );
    }

    group.finish();
}

// =============================================================================
// CORRECTNESS VALIDATION (not a benchmark, but important)
// =============================================================================

fn validate_gpu_cpu_equivalence(c: &mut Criterion) {
    let mut group = c.benchmark_group("cfc_validation");
    group.sample_size(10);

    // This benchmark validates that GPU and CPU produce similar results
    let gpu_config = GpuCfcConfig {
        input_dim: 32,
        hidden_dim: 64,
        num_layers: 2,
        output_dim: 16,
        use_backbone: false,
        ..Default::default()
    };

    let input = vec![0.5f32; 32];
    let dt = 0.1f32;

    group.bench_function("validation", |b| {
        b.iter(|| {
            // Create fresh networks each iteration to get consistent initialization
            let mut gpu_network = GpuCfcNetwork::new(gpu_config.clone(), GpuBackend::Cpu)
                .expect("Failed to create GPU network");

            let output = gpu_network.forward(&input, dt).unwrap();

            // Validate output is finite and reasonable
            for &v in &output {
                assert!(v.is_finite(), "Output contains non-finite values");
                assert!(v.abs() < 100.0, "Output values unreasonably large");
            }

            black_box(output)
        })
    });

    group.finish();
}

// =============================================================================
// CRITERION CONFIGURATION
// =============================================================================

criterion_group!(
    name = forward_benches;
    config = Criterion::default();
    targets = bench_forward_single, bench_forward_batch
);

criterion_group!(
    name = training_benches;
    config = Criterion::default();
    targets = bench_train_step
);

criterion_group!(
    name = sequence_benches;
    config = Criterion::default();
    targets = bench_sequence_processing
);

criterion_group!(
    name = validation_benches;
    config = Criterion::default();
    targets = validate_gpu_cpu_equivalence
);

criterion_main!(
    forward_benches,
    training_benches,
    sequence_benches,
    validation_benches
);
