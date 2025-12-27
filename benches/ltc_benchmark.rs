//! LTC (Liquid Time-Constant) Network Performance Benchmarks
//!
//! Validates the performance claims from README.md:
//! - LTC Step: ~0.02ms (50,000 steps/sec)
//! - Consciousness Check: ~0.01ms (100,000 ops/sec)
//! - Full Query: ~0.50ms (2,000 queries/sec)

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use symthaea::ltc::LiquidNetwork;
use symthaea::hdc::SemanticSpace;

/// Benchmark LTC network creation
fn bench_ltc_new(c: &mut Criterion) {
    let mut group = c.benchmark_group("LiquidNetwork::new");

    for neurons in [100, 500, 1000, 2000].iter() {
        group.bench_with_input(
            BenchmarkId::new("neurons", neurons),
            neurons,
            |b, &neurons| {
                b.iter(|| black_box(LiquidNetwork::new(neurons).unwrap()))
            }
        );
    }
    group.finish();
}

/// Benchmark LTC step operation
/// Claim: ~0.02ms (20μs)
fn bench_ltc_step(c: &mut Criterion) {
    let mut network = LiquidNetwork::new(1000).unwrap();

    c.bench_function("LiquidNetwork::step (1000 neurons)", |b| {
        b.iter(|| black_box(network.step().unwrap()))
    });
}

/// Benchmark LTC injection
fn bench_ltc_inject(c: &mut Criterion) {
    let mut network = LiquidNetwork::new(1000).unwrap();
    let mut space = SemanticSpace::new(10000).unwrap();
    let input = space.encode("install nginx").unwrap();

    c.bench_function("LiquidNetwork::inject", |b| {
        b.iter(|| black_box(network.inject(black_box(&input)).unwrap()))
    });
}

/// Benchmark consciousness level check
/// Claim: ~0.01ms (10μs)
fn bench_consciousness_level(c: &mut Criterion) {
    let mut network = LiquidNetwork::new(1000).unwrap();
    // Run a few steps to populate state
    for _ in 0..10 {
        let _ = network.step();
    }

    c.bench_function("LiquidNetwork::consciousness_level", |b| {
        b.iter(|| black_box(network.consciousness_level()))
    });
}

/// Benchmark full LTC processing cycle (inject → step → read)
fn bench_ltc_full_cycle(c: &mut Criterion) {
    let mut network = LiquidNetwork::new(1000).unwrap();
    let mut space = SemanticSpace::new(10000).unwrap();
    let input = space.encode("search package manager").unwrap();

    c.bench_function("LTC full cycle (inject → 10 steps → read)", |b| {
        b.iter(|| {
            network.inject(black_box(&input)).unwrap();
            for _ in 0..10 {
                network.step().unwrap();
            }
            black_box(network.read_state().unwrap())
        })
    });
}

/// Benchmark state reading
fn bench_ltc_read_state(c: &mut Criterion) {
    let mut network = LiquidNetwork::new(1000).unwrap();
    for _ in 0..10 {
        let _ = network.step();
    }

    c.bench_function("LiquidNetwork::read_state", |b| {
        b.iter(|| black_box(network.read_state().unwrap()))
    });
}

/// Benchmark varying neuron counts
fn bench_ltc_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("LTC Scaling (step time)");

    for neurons in [100, 250, 500, 1000, 2000].iter() {
        let mut network = LiquidNetwork::new(*neurons).unwrap();

        group.bench_with_input(
            BenchmarkId::new("neurons", neurons),
            neurons,
            |b, _| {
                b.iter(|| black_box(network.step().unwrap()))
            }
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_ltc_new,
    bench_ltc_step,
    bench_ltc_inject,
    bench_consciousness_level,
    bench_ltc_full_cycle,
    bench_ltc_read_state,
    bench_ltc_scaling,
);

criterion_main!(benches);
