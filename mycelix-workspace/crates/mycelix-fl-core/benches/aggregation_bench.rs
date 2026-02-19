use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mycelix_fl_core::*;
use std::collections::HashMap;

fn generate_updates(n: usize, dim: usize) -> Vec<GradientUpdate> {
    // Generate n GradientUpdate with `dim` dimensions, random f32 values using simple deterministic formula
    (0..n)
        .map(|i| {
            let gradients: Vec<f32> = (0..dim)
                .map(|j| ((i * 7 + j * 13) % 1000) as f32 / 1000.0 - 0.5)
                .collect();
            GradientUpdate::new(format!("node-{}", i), 1, gradients, 100, 0.5)
        })
        .collect()
}

fn generate_reputations(n: usize) -> HashMap<String, f32> {
    (0..n).map(|i| (format!("node-{}", i), 0.8)).collect()
}

fn bench_aggregation_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("aggregation");

    for &n in &[10, 50, 100] {
        let dim = 10_000;
        let updates = generate_updates(n, dim);

        group.bench_with_input(BenchmarkId::new("fedavg", n), &updates, |b, updates| {
            b.iter(|| fedavg(black_box(updates)))
        });

        group.bench_with_input(
            BenchmarkId::new("coordinate_median", n),
            &updates,
            |b, updates| b.iter(|| coordinate_median(black_box(updates))),
        );

        group.bench_with_input(
            BenchmarkId::new("trimmed_mean", n),
            &updates,
            |b, updates| b.iter(|| trimmed_mean(black_box(updates), 0.1)),
        );

        if n <= 50 {
            // Krum is O(n^2 * d), expensive for 100
            group.bench_with_input(BenchmarkId::new("krum", n), &updates, |b, updates| {
                b.iter(|| krum(black_box(updates), 1))
            });
        }
    }
    group.finish();
}

fn bench_gradient_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("dimensions");
    let n = 10;

    for &dim in &[1_000, 10_000, 100_000] {
        let updates = generate_updates(n, dim);

        group.bench_with_input(BenchmarkId::new("fedavg", dim), &updates, |b, updates| {
            b.iter(|| fedavg(black_box(updates)))
        });
    }
    group.finish();
}

fn bench_unified_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline");

    for &n in &[10, 50] {
        let dim = 10_000;
        let updates = generate_updates(n, dim);
        let reputations = generate_reputations(n);

        group.bench_with_input(
            BenchmarkId::new("full_pipeline", n),
            &(updates, reputations),
            |b, (updates, reps)| {
                b.iter(|| {
                    let mut pipeline = UnifiedPipeline::new(PipelineConfig::default());
                    pipeline.aggregate(black_box(updates), black_box(reps))
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_aggregation_algorithms,
    bench_gradient_dimensions,
    bench_unified_pipeline
);
criterion_main!(benches);
