// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mycelix_fl::compression::HyperFeelCompressor;
use mycelix_fl::fl_core::GradientMetadata;
use mycelix_fl::pipeline::{DecentralizedPipeline, PipelineConfig};
use mycelix_fl::types::CompressedGradient;
use std::collections::HashMap;

fn generate_gradient(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|j| ((seed * 7 + j * 13) % 1000) as f32 / 1000.0 - 0.5)
        .collect()
}

fn bench_hyperfeel_encode_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyperfeel");
    let compressor = HyperFeelCompressor::default_seed();

    for &dim in &[1_000, 10_000, 100_000] {
        let gradient = generate_gradient(dim, 42);

        group.bench_with_input(BenchmarkId::new("encode", dim), &gradient, |b, gradient| {
            b.iter(|| compressor.compress("node-0", 1, black_box(gradient), 0.9))
        });

        // Encode once for decode benchmark
        let compressed = compressor.compress("node-0", 1, &gradient, 0.9).unwrap();
        group.bench_with_input(
            BenchmarkId::new("decode", dim),
            &compressed,
            |b, compressed| {
                b.iter(|| compressor.decompress(black_box(&compressed.hv_data), dim))
            },
        );
    }
    group.finish();
}

fn bench_decentralized_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("decentralized_pipeline");
    let compressor = HyperFeelCompressor::default_seed();

    for &n in &[10, 50] {
        let compressed: Vec<CompressedGradient> = (0..n)
            .map(|i| {
                let gradient = generate_gradient(1000, i);
                compressor
                    .compress(&format!("node-{}", i), 1, &gradient, 0.9)
                    .unwrap()
            })
            .collect();
        let reputations: HashMap<String, f32> =
            (0..n).map(|i| (format!("node-{}", i), 0.8)).collect();

        group.bench_with_input(
            BenchmarkId::new("aggregate_compressed", n),
            &(compressed, reputations),
            |b, (compressed, reps)| {
                b.iter(|| {
                    let pipeline = DecentralizedPipeline::new(PipelineConfig::default());
                    pipeline.aggregate_compressed(black_box(compressed), black_box(reps))
                })
            },
        );
    }
    group.finish();
}

#[cfg(feature = "pogq")]
fn bench_pogq_scoring(c: &mut Criterion) {
    use mycelix_fl::pogq::{PoGQLiteConfig, PoGQLiteDetector};

    let mut group = c.benchmark_group("pogq");
    let dim = 1_000;

    for &n_clients in &[10, 100] {
        let gradients: Vec<Vec<f32>> = (0..n_clients)
            .map(|i| generate_gradient(dim, i))
            .collect();
        let reference = generate_gradient(dim, 999);

        group.bench_with_input(
            BenchmarkId::new("score_gradient", n_clients),
            &(&gradients, &reference),
            |b, (grads, ref_grad)| {
                b.iter(|| {
                    let mut det = PoGQLiteDetector::new(PoGQLiteConfig::default());
                    for (i, g) in grads.iter().enumerate() {
                        det.score_gradient(
                            black_box(g),
                            &format!("c{}", i),
                            black_box(ref_grad),
                            0,
                        );
                    }
                })
            },
        );
    }
    group.finish();
}

#[cfg(feature = "pogq")]
criterion_group!(
    benches,
    bench_hyperfeel_encode_decode,
    bench_decentralized_pipeline,
    bench_pogq_scoring
);

#[cfg(not(feature = "pogq"))]
criterion_group!(benches, bench_hyperfeel_encode_decode, bench_decentralized_pipeline);

criterion_main!(benches);
