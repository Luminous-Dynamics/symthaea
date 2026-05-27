// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal Signature Encoder Benchmark Suite
//!
//! Measures HDC-based temporal pattern encoding, feature extraction,
//! similarity computation, and state classification for consciousness
//! pattern detection.
//!
//! Run with: cargo bench --bench dynamics_temporal

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use symthaea::dynamics::{
    ConsciousnessPattern, SignatureConfig, TemporalSignatureEncoder, TrajectoryFeatures,
};

// ---------------------------------------------------------------------------
// Helper: generate deterministic tau trajectories
// ---------------------------------------------------------------------------

/// Generate a "focused" trajectory: nearly constant tau with small jitter.
fn focused_trajectory(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let jitter = ((i as f64 * 0.7).sin() * 0.01) as f32;
            0.5 + jitter
        })
        .collect()
}

/// Generate an "excited" trajectory: oscillating tau with high variance.
fn excited_trajectory(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let phase = i as f32 * 0.5;
            0.5 + 0.3 * phase.sin()
        })
        .collect()
}

/// Generate an "exploratory" trajectory: diverse tau values.
fn exploratory_trajectory(len: usize) -> Vec<f32> {
    let mut state = 42u64 ^ 0x9E3779B97F4A7C15;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            0.1 + 0.9 * (state as f32 / u64::MAX as f32)
        })
        .collect()
}

/// Build a pre-warmed encoder with N recorded tau values.
fn warmed_encoder(trajectory: &[f32]) -> TemporalSignatureEncoder {
    let config = SignatureConfig {
        history_length: trajectory.len().max(100),
        min_history: 20,
        ..Default::default()
    };
    let mut encoder = TemporalSignatureEncoder::new(config);
    for &tau in trajectory {
        encoder.record(tau);
    }
    encoder
}

// ---------------------------------------------------------------------------
// Benchmark: TrajectoryFeatures::from_history
// ---------------------------------------------------------------------------

fn bench_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_feature_extraction");

    for &len in &[50, 100, 200, 500] {
        let trajectory = focused_trajectory(len);

        group.bench_with_input(BenchmarkId::new("len", len), &len, |b, _| {
            b.iter(|| {
                black_box(TrajectoryFeatures::from_history(black_box(&trajectory)));
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Encoder record (single tau append + re-encode)
// ---------------------------------------------------------------------------

fn bench_record_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_record_single");

    for &history_len in &[50, 100, 200] {
        let trajectory = focused_trajectory(history_len);
        let mut encoder = warmed_encoder(&trajectory);

        group.bench_with_input(
            BenchmarkId::new("history", history_len),
            &history_len,
            |b, _| {
                b.iter(|| {
                    encoder.record(black_box(0.55));
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Encoder record_batch
// ---------------------------------------------------------------------------

fn bench_record_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_record_batch");

    for &batch_size in &[16, 64, 128, 256] {
        let trajectory = focused_trajectory(50);
        let mut encoder = warmed_encoder(&trajectory);
        let batch: Vec<f32> = (0..batch_size)
            .map(|i| 0.3 + 0.4 * (i as f32 / batch_size as f32))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    encoder.record_batch(black_box(&batch));
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Similarity computation (HDC cosine distance)
// ---------------------------------------------------------------------------

fn bench_similarity_to(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_similarity_to");

    let trajectory = focused_trajectory(100);
    let encoder = warmed_encoder(&trajectory);

    for pattern in ConsciousnessPattern::all() {
        group.bench_with_input(
            BenchmarkId::new("pattern", format!("{:?}", pattern)),
            pattern,
            |b, pattern| {
                b.iter(|| {
                    black_box(encoder.similarity_to(black_box(*pattern)));
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: All similarities (7 patterns)
// ---------------------------------------------------------------------------

fn bench_all_similarities(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_all_similarities");

    let trajectories: Vec<(&str, Vec<f32>)> = vec![
        ("focused", focused_trajectory(100)),
        ("excited", excited_trajectory(100)),
        ("exploratory", exploratory_trajectory(100)),
    ];

    for (name, trajectory) in &trajectories {
        let encoder = warmed_encoder(trajectory);

        group.bench_with_input(BenchmarkId::new("state", *name), name, |b, _| {
            b.iter(|| {
                black_box(encoder.all_similarities());
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: classify_state (similarity + argmax)
// ---------------------------------------------------------------------------

fn bench_classify_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_classify_state");

    let trajectories: Vec<(&str, Vec<f32>)> = vec![
        ("focused", focused_trajectory(100)),
        ("excited", excited_trajectory(100)),
        ("exploratory", exploratory_trajectory(100)),
    ];

    for (name, trajectory) in &trajectories {
        let encoder = warmed_encoder(trajectory);

        group.bench_with_input(BenchmarkId::new("state", *name), name, |b, _| {
            b.iter(|| {
                black_box(encoder.classify_state());
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Encoder initialization (base vector + pattern creation)
// ---------------------------------------------------------------------------

fn bench_encoder_init(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_encoder_init");

    for &num_bins in &[8, 16, 32] {
        let config = SignatureConfig {
            num_bins,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::new("bins", num_bins), &num_bins, |b, _| {
            b.iter(|| {
                black_box(TemporalSignatureEncoder::new(config.clone()));
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Full pipeline (record N values then classify)
// ---------------------------------------------------------------------------

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_full_pipeline");
    group.sample_size(10);

    for &n_values in &[50, 100, 200] {
        let trajectory = excited_trajectory(n_values);
        let config = SignatureConfig {
            history_length: n_values,
            min_history: 20,
            ..Default::default()
        };

        group.bench_with_input(BenchmarkId::new("values", n_values), &n_values, |b, _| {
            b.iter(|| {
                let mut encoder = TemporalSignatureEncoder::new(config.clone());
                for &tau in &trajectory {
                    encoder.record(black_box(tau));
                }
                black_box(encoder.classify_state());
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_feature_extraction,
    bench_record_single,
    bench_record_batch,
    bench_similarity_to,
    bench_all_similarities,
    bench_classify_state,
    bench_encoder_init,
    bench_full_pipeline,
);
criterion_main!(benches);
