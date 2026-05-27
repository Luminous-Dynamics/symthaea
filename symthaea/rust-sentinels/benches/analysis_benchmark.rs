// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Performance benchmarks for Sentinels library

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use sentinels::{AnalysisMode, analyze_consciousness};

fn generate_signal(n_samples: usize, sample_rate: f32) -> Vec<f32> {
    (0..n_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            let pi2 = 2.0 * std::f32::consts::PI;
            0.5 * (pi2 * 10.0 * t).sin()
                + 0.3 * (pi2 * 6.0 * t).sin()
                + 0.1 * (pi2 * 20.0 * t).sin()
        })
        .collect()
}

fn benchmark_full_analysis(c: &mut Criterion) {
    let sample_rate = 256.0;

    let mut group = c.benchmark_group("full_analysis");

    for duration in [5.0, 10.0, 30.0] {
        let n_samples = (sample_rate * duration) as usize;
        let signal = generate_signal(n_samples, sample_rate);

        group.bench_with_input(
            BenchmarkId::new("auto", format!("{}s", duration as i32)),
            &signal,
            |b, data| {
                b.iter(|| analyze_consciousness(black_box(data), sample_rate, AnalysisMode::Auto));
            },
        );
    }

    group.finish();
}

fn benchmark_individual_sentinels(c: &mut Criterion) {
    let sample_rate = 256.0;
    let duration = 30.0;
    let n_samples = (sample_rate * duration) as usize;
    let signal = generate_signal(n_samples, sample_rate);

    let mut group = c.benchmark_group("individual_sentinels");

    group.bench_function("emotion_only", |b| {
        b.iter(|| analyze_consciousness(black_box(&signal), sample_rate, AnalysisMode::Emotion));
    });

    group.bench_function("sleep_only", |b| {
        b.iter(|| analyze_consciousness(black_box(&signal), sample_rate, AnalysisMode::Sleep));
    });

    group.bench_function("meditation_only", |b| {
        b.iter(|| analyze_consciousness(black_box(&signal), sample_rate, AnalysisMode::Meditation));
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_full_analysis,
    benchmark_individual_sentinels
);
criterion_main!(benches);