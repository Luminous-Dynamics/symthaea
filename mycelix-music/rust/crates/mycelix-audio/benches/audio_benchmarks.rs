// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Benchmarks for Mycelix Audio Processing

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::f32::consts::PI;

fn generate_sine_wave(samples: usize, frequency: f32, sample_rate: u32) -> Vec<f32> {
    (0..samples)
        .map(|i| (i as f32 / sample_rate as f32 * frequency * 2.0 * PI).sin())
        .collect()
}

fn bench_audio_buffer_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio_buffer");

    for size in [1024, 4096, 16384, 65536].iter() {
        let audio_data = generate_sine_wave(*size, 440.0, 44100);

        group.bench_with_input(BenchmarkId::new("peak_detection", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    audio_data.iter()
                        .map(|s| s.abs())
                        .fold(0.0f32, f32::max)
                )
            })
        });

        group.bench_with_input(BenchmarkId::new("rms_calculation", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    (audio_data.iter()
                        .map(|s| s * s)
                        .sum::<f32>() / audio_data.len() as f32)
                        .sqrt()
                )
            })
        });

        group.bench_with_input(BenchmarkId::new("normalize", size), size, |b, _| {
            let mut data = audio_data.clone();
            b.iter(|| {
                let peak = data.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                if peak > 0.0 {
                    let gain = 1.0 / peak;
                    for sample in data.iter_mut() {
                        *sample *= gain;
                    }
                }
                black_box(&data)
            })
        });
    }

    group.finish();
}

fn bench_stereo_to_mono(c: &mut Criterion) {
    let mut group = c.benchmark_group("stereo_to_mono");

    for size in [2048, 8192, 32768].iter() {
        let stereo_data: Vec<f32> = generate_sine_wave(*size * 2, 440.0, 44100);

        group.bench_with_input(BenchmarkId::new("chunks", size), size, |b, _| {
            b.iter(|| {
                black_box(
                    stereo_data
                        .chunks(2)
                        .map(|chunk| (chunk[0] + chunk.get(1).copied().unwrap_or(0.0)) / 2.0)
                        .collect::<Vec<f32>>()
                )
            })
        });
    }

    group.finish();
}

fn bench_zero_crossing_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_crossing");

    for size in [4096, 16384, 65536].iter() {
        let audio_data = generate_sine_wave(*size, 440.0, 44100);

        group.bench_with_input(BenchmarkId::new("zcr", size), size, |b, _| {
            b.iter(|| {
                let mut crossings = 0;
                for window in audio_data.windows(2) {
                    if (window[0] >= 0.0) != (window[1] >= 0.0) {
                        crossings += 1;
                    }
                }
                black_box(crossings)
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_audio_buffer_operations,
    bench_stereo_to_mono,
    bench_zero_crossing_rate,
);

criterion_main!(benches);
