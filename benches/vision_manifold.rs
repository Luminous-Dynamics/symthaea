// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Vision Manifold Benchmark Suite
//!
//! Measures per-frame latency for the vision pipeline at various resolutions
//! and feature configurations. Key question: does observe_frame fit the
//! 20Hz (50ms) cognitive loop budget?
//!
//! ## Usage
//!
//! ```bash
//! cargo bench --bench vision_manifold --features vision-manifold
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use symthaea_vision_manifold::{VisionBridge, VisionConfig, VisionManifold};

fn make_frame(width: u32, height: u32, channels: usize, seed: u8) -> Vec<u8> {
    (0..(width as usize * height as usize * channels))
        .map(|i| ((i as u8).wrapping_mul(seed).wrapping_add(37)) as u8)
        .collect()
}

fn bench_observe_frame(c: &mut Criterion) {
    let mut group = c.benchmark_group("observe_frame");
    group.sample_size(20);

    for &(w, h, label) in &[(64, 64, "64x64"), (320, 240, "320x240"), (640, 480, "640x480")] {
        let cfg = VisionConfig::default();
        let mut m = VisionManifold::new(cfg, w, h);
        let frame = make_frame(w, h, 3, 42);
        // Warm up
        for _ in 0..5 {
            m.observe_frame(&frame, w, h, 3, 0.033);
        }

        group.bench_with_input(BenchmarkId::new("rgb", label), &(), |b, _| {
            b.iter(|| {
                black_box(m.observe_frame(black_box(&frame), w, h, 3, 0.033));
            });
        });
    }
    group.finish();
}

fn bench_observe_frame_all_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("observe_frame_all_features");
    group.sample_size(20);

    for &(w, h, label) in &[(64, 64, "64x64"), (320, 240, "320x240")] {
        let mut cfg = VisionConfig::default();
        cfg.enable_depth = true;
        cfg.enable_object_binding = true;
        cfg.enable_temporal_binding = true;
        let mut m = VisionManifold::new(cfg, w, h);
        m.enable_object_memory(32);
        m.enable_working_memory(4);
        m.enable_scene_graph();
        m.enable_scene_memory(16);

        let frame = make_frame(w, h, 3, 42);
        for _ in 0..5 {
            m.observe_frame(&frame, w, h, 3, 0.033);
        }

        group.bench_with_input(BenchmarkId::new("all_features", label), &(), |b, _| {
            b.iter(|| {
                black_box(m.observe_frame(black_box(&frame), w, h, 3, 0.033));
            });
        });
    }
    group.finish();
}

fn bench_bridge_process_frame(c: &mut Criterion) {
    let mut group = c.benchmark_group("bridge_process_frame");
    group.sample_size(20);

    for &(w, h, label) in &[(64, 64, "64x64"), (320, 240, "320x240")] {
        let mut bridge = VisionBridge::new(VisionConfig::default(), w, h);
        let frame = make_frame(w, h, 3, 42);
        for _ in 0..5 {
            bridge.process_frame(&frame, w, h, 3, 0.033);
        }

        group.bench_with_input(BenchmarkId::new("with_attention", label), &(), |b, _| {
            b.iter(|| {
                black_box(bridge.process_frame(black_box(&frame), w, h, 3, 0.033));
            });
        });
    }
    group.finish();
}

fn bench_dream_ahead(c: &mut Criterion) {
    let mut group = c.benchmark_group("dream_ahead");

    let mut m = VisionManifold::new(VisionConfig::default(), 64, 64);
    let frame = make_frame(64, 64, 3, 42);
    m.observe_frame(&frame, 64, 64, 3, 0.033);

    for &steps in &[1, 5, 20] {
        group.bench_with_input(BenchmarkId::new("steps", steps), &steps, |b, &n| {
            b.iter(|| {
                black_box(m.dream_ahead(n, 0.033));
            });
        });
    }
    group.finish();
}

fn bench_stereo_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("stereo_depth");
    group.sample_size(20);

    let cfg = VisionConfig::default();
    let encoder = symthaea_vision_manifold::PatchHdcEncoder::new(&cfg, 320, 240);
    let left = make_frame(320, 240, 1, 42);
    let right = make_frame(320, 240, 1, 43); // slightly different

    group.bench_function("320x240_disp16", |b| {
        b.iter(|| {
            black_box(encoder.compute_stereo_depth(
                black_box(&left),
                black_box(&right),
                320,
                240,
                16,
            ));
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_observe_frame,
    bench_observe_frame_all_features,
    bench_bridge_process_frame,
    bench_dream_ahead,
    bench_stereo_depth,
);
criterion_main!(benches);
