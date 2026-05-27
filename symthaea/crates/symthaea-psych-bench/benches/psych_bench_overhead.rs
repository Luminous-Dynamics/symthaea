// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Criterion benchmark: wall-clock per trial for each benchmark suite.

use criterion::{Criterion, criterion_group, criterion_main};
use symthaea_psych_bench::benchmarks::cogbench::ProbabilisticReasoningBenchmark;
use symthaea_psych_bench::benchmarks::memory_agent::AccurateRetrievalBenchmark;
use symthaea_psych_bench::benchmarks::tombench::FalseBeliefBenchmark;
use symthaea_psych_bench::benchmarks::worm::NBackBenchmark;
use symthaea_psych_bench::harness::{BenchmarkConfig, PsychBenchmark};

fn bench_config() -> BenchmarkConfig {
    BenchmarkConfig {
        dimension: 256,
        trials_per_condition: 10,
        ..Default::default()
    }
}

fn bench_nback(c: &mut Criterion) {
    let config = bench_config();
    c.bench_function("worm::nback", |b| {
        b.iter(|| NBackBenchmark.run(&config));
    });
}

fn bench_probabilistic(c: &mut Criterion) {
    let config = bench_config();
    c.bench_function("cogbench::probabilistic", |b| {
        b.iter(|| ProbabilisticReasoningBenchmark.run(&config));
    });
}

fn bench_false_belief(c: &mut Criterion) {
    let config = bench_config();
    c.bench_function("tombench::false_belief", |b| {
        b.iter(|| FalseBeliefBenchmark.run(&config));
    });
}

fn bench_accurate_retrieval(c: &mut Criterion) {
    let config = bench_config();
    c.bench_function("memory_agent::accurate_retrieval", |b| {
        b.iter(|| AccurateRetrievalBenchmark.run(&config));
    });
}

criterion_group!(
    benches,
    bench_nback,
    bench_probabilistic,
    bench_false_belief,
    bench_accurate_retrieval
);
criterion_main!(benches);
