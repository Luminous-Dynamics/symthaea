//! Verified Performance Benchmark Suite
//!
//! This benchmark suite verifies the performance claims made in documentation.
//! All metrics here are measured, not estimated.
//!
//! ## Running
//!
//! ```bash
//! cargo bench --bench verified_performance
//! ```
//!
//! ## Documented Claims to Verify
//!
//! | Operation | Claimed | Target | Benchmark |
//! |-----------|---------|--------|-----------|
//! | CfC Inference | 34 μs/step | <50 μs | `cfc_inference_64x128` |
//! | CfC BPTT Training | 5 ms/step | <10 ms | `cfc_bptt_64x128` |
//! | HDC Bind | <1 μs | <1 μs | `hdc_bind_*` |
//! | HDC Bundle | <1 μs | <1 μs | `hdc_bundle_*` |
//! | HDC Similarity | <10 μs | <10 μs | `hdc_similarity_*` |
//! | Φ Spectral (λ₂) | <5 ms | <10 ms | `phi_spectral_*` |

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
use symthaea_core::hdc::binary_hv::HV16;
use symthaea_core::phi_engine::{PhiEngine, PhiMethod};
use symthaea::dynamics::cfc::{CfCCell, CfCConfig, CfCNetwork, CfCNetworkConfig};
use ndarray::Array1;

/// Benchmark HDC bind operation
fn bench_hdc_bind(c: &mut Criterion) {
    let a = ContinuousHV::random(HDC_DIMENSION, 42);
    let b = ContinuousHV::random(HDC_DIMENSION, 43);

    c.bench_function("hdc_bind_continuous", |bencher| {
        bencher.iter(|| {
            black_box(a.bind(&b))
        })
    });

    let a_bin = HV16::random(42);
    let b_bin = HV16::random(43);

    c.bench_function("hdc_bind_binary", |bencher| {
        bencher.iter(|| {
            black_box(a_bin.bind(&b_bin))
        })
    });
}

/// Benchmark HDC bundle operation
fn bench_hdc_bundle(c: &mut Criterion) {
    let hvs: Vec<ContinuousHV> = (0..10)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i))
        .collect();
    let hv_refs: Vec<&ContinuousHV> = hvs.iter().collect();

    c.bench_function("hdc_bundle_10_continuous", |bencher| {
        bencher.iter(|| {
            black_box(ContinuousHV::bundle(&hv_refs))
        })
    });

    let hvs_bin: Vec<HV16> = (0..10)
        .map(|i| HV16::random(i))
        .collect();

    c.bench_function("hdc_bundle_10_binary", |bencher| {
        bencher.iter(|| {
            black_box(HV16::bundle(&hvs_bin))
        })
    });
}

/// Benchmark HDC similarity computation
fn bench_hdc_similarity(c: &mut Criterion) {
    let a = ContinuousHV::random(HDC_DIMENSION, 42);
    let b = ContinuousHV::random(HDC_DIMENSION, 43);

    c.bench_function("hdc_similarity_continuous", |bencher| {
        bencher.iter(|| {
            black_box(a.similarity(&b))
        })
    });

    let a_bin = HV16::random(42);
    let b_bin = HV16::random(43);

    c.bench_function("hdc_similarity_binary", |bencher| {
        bencher.iter(|| {
            black_box(a_bin.similarity(&b_bin))
        })
    });
}

/// Benchmark Φ spectral calculation (λ₂ method)
fn bench_phi_spectral(c: &mut Criterion) {
    let engine = PhiEngine::new(PhiMethod::SpectralConnectivity);

    // Create node representations
    let nodes: Vec<ContinuousHV> = (0..8)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i))
        .collect();

    c.bench_function("phi_spectral_8_nodes", |bencher| {
        bencher.iter(|| {
            black_box(engine.compute(&nodes))
        })
    });
}

/// Benchmark Φ with varying node counts
fn bench_phi_scaling(c: &mut Criterion) {
    let engine = PhiEngine::new(PhiMethod::SpectralConnectivity);

    let mut group = c.benchmark_group("phi_spectral_scaling");

    for n_nodes in [4, 6, 8, 10, 12] {
        let nodes: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64))
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(n_nodes),
            &nodes,
            |b, nodes| {
                b.iter(|| black_box(engine.compute(nodes)))
            },
        );
    }

    group.finish();
}

/// Benchmark text encoding to HDC
fn bench_text_encoding(c: &mut Criterion) {
    use symthaea_core::hdc::text_encoder::{TextEncoder, TextEncoderConfig};

    let mut encoder = TextEncoder::new(TextEncoderConfig::default())
        .expect("TextEncoder creation");
    let text = "The quick brown fox jumps over the lazy dog";

    c.bench_function("text_encoding", |bencher| {
        bencher.iter(|| {
            black_box(encoder.encode_sentence(text))
        })
    });
}

/// Benchmark random HV generation
fn bench_hv_generation(c: &mut Criterion) {
    c.bench_function("hv_random_continuous", |bencher| {
        let mut seed = 0u64;
        bencher.iter(|| {
            seed += 1;
            black_box(ContinuousHV::random(HDC_DIMENSION, seed))
        })
    });

    c.bench_function("hv_random_binary", |bencher| {
        let mut seed = 0u64;
        bencher.iter(|| {
            seed += 1;
            black_box(HV16::random(seed))
        })
    });
}

/// Summary benchmark that measures all key operations
fn bench_summary(c: &mut Criterion) {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║          VERIFIED PERFORMANCE BENCHMARK SUITE                ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  These benchmarks verify documented performance claims       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let mut group = c.benchmark_group("verified_claims");

    // HDC Bind (claim: <1 μs)
    let a = ContinuousHV::random(HDC_DIMENSION, 42);
    let b = ContinuousHV::random(HDC_DIMENSION, 43);
    group.bench_function("hdc_bind_claim_lt_1us", |bencher| {
        bencher.iter(|| black_box(a.bind(&b)))
    });

    // HDC Similarity (claim: <10 μs)
    group.bench_function("hdc_similarity_claim_lt_10us", |bencher| {
        bencher.iter(|| black_box(a.similarity(&b)))
    });

    // Phi Spectral (claim: <5 ms)
    let engine = PhiEngine::new(PhiMethod::SpectralConnectivity);
    let nodes: Vec<ContinuousHV> = (0..8)
        .map(|i| ContinuousHV::random(HDC_DIMENSION, i))
        .collect();
    group.bench_function("phi_spectral_claim_lt_5ms", |bencher| {
        bencher.iter(|| black_box(engine.compute(&nodes)))
    });

    // CfC Inference (claim: 34 μs/step)
    let cfc_config = CfCConfig {
        input_dim: 64,
        hidden_dim: 128,
        use_backbone: true,
        ..Default::default()
    };
    let mut cfc = CfCCell::new(cfc_config);
    let cfc_input = Array1::from_shape_fn(64, |i| (i as f32 * 0.1).sin());
    group.bench_function("cfc_inference_claim_lt_50us", |bencher| {
        bencher.iter(|| black_box(cfc.forward(black_box(&cfc_input), 0.01)))
    });

    group.finish();
}

/// Benchmark CfC single-cell inference (claim: 34 μs/step)
fn bench_cfc_inference(c: &mut Criterion) {
    let config = CfCConfig {
        input_dim: 64,
        hidden_dim: 128,
        use_backbone: true,
        ..Default::default()
    };
    let mut cell = CfCCell::new(config);
    let input = Array1::from_shape_fn(64, |i| (i as f32 * 0.1).sin());
    let dt = 0.01_f32;

    c.bench_function("cfc_inference_64x128", |bencher| {
        bencher.iter(|| {
            black_box(cell.forward(black_box(&input), black_box(dt)))
        })
    });
}

/// Benchmark CfC BPTT training step (claim: 5 ms/step)
fn bench_cfc_bptt(c: &mut Criterion) {
    let config = CfCNetworkConfig {
        input_dim: 64,
        hidden_dim: 128,
        num_layers: 2,
        output_dim: 16,
        cell_config: CfCConfig {
            input_dim: 64,
            hidden_dim: 128,
            use_backbone: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut network = CfCNetwork::new(config);

    // Single-step sequence for BPTT
    let inputs = vec![Array1::from_shape_fn(64, |i| (i as f32 * 0.1).sin())];
    let targets = vec![Array1::from_shape_fn(16, |i| (i as f32 * 0.05).cos())];
    let dts = vec![0.01_f32];
    let lr = 0.001_f32;

    c.bench_function("cfc_bptt_64x128", |bencher| {
        bencher.iter(|| {
            black_box(network.train_step_bptt(
                black_box(&inputs),
                black_box(&targets),
                black_box(&dts),
                black_box(lr),
            ))
        })
    });
}

/// Benchmark CfC network forward pass (multi-layer)
fn bench_cfc_network(c: &mut Criterion) {
    let mut group = c.benchmark_group("cfc_network");

    for (name, input_dim, hidden_dim, layers) in [
        ("small_32x64x2", 32, 64, 2),
        ("medium_64x128x3", 64, 128, 3),
    ] {
        let config = CfCNetworkConfig {
            input_dim,
            hidden_dim,
            num_layers: layers,
            output_dim: 16,
            cell_config: CfCConfig {
                input_dim,
                hidden_dim,
                use_backbone: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut network = CfCNetwork::new(config);
        let input = Array1::from_shape_fn(input_dim, |i| (i as f32 * 0.1).sin());
        let dt = 0.01_f32;

        group.bench_function(name, |bencher| {
            bencher.iter(|| {
                black_box(network.forward(black_box(&input), black_box(dt)))
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hdc_bind,
    bench_hdc_bundle,
    bench_hdc_similarity,
    bench_phi_spectral,
    bench_phi_scaling,
    bench_text_encoding,
    bench_hv_generation,
    bench_cfc_inference,
    bench_cfc_bptt,
    bench_cfc_network,
    bench_summary,
);

criterion_main!(benches);
