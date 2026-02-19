use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mycelix_fl::compression::HyperFeelCompressor;
use mycelix_fl::fl_core::GradientMetadata;
use mycelix_fl::pipeline::{DecentralizedPipeline, PipelineConfig};
use mycelix_fl::types::{CompressedGradient, HV16_BYTES};
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

criterion_group!(benches, bench_hyperfeel_encode_decode, bench_decentralized_pipeline);
criterion_main!(benches);
