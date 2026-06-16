// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Criterion benchmarks for Symthaea STT core operations

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use std::f32::consts::PI;
use symthaea_stt::{
    AudioConfig, AudioFrontend, AudioProjector, HV16, LtcCell, LtcConfig, PhonemeResonator, bundle,
    weighted_bundle,
};

/// Generate synthetic audio for benchmarks
fn generate_test_audio(duration_sec: f32, sample_rate: u32) -> Vec<f32> {
    let num_samples = (duration_sec * sample_rate as f32) as usize;
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            // Mix of frequencies to simulate speech
            (2.0 * PI * 200.0 * t).sin() * 0.3
                + (2.0 * PI * 400.0 * t).sin() * 0.2
                + (2.0 * PI * 800.0 * t).sin() * 0.1
        })
        .collect()
}

// ============================================================================
// HDC BENCHMARKS
// ============================================================================

fn bench_hdc_similarity(c: &mut Criterion) {
    let hv1 = HV16::random("bench_hv1");
    let hv2 = HV16::random("bench_hv2");

    c.bench_function("hdc/similarity", |b| {
        b.iter(|| black_box(hv1.similarity(&hv2)))
    });
}

fn bench_hdc_bind(c: &mut Criterion) {
    let hv1 = HV16::random("bench_bind1");
    let hv2 = HV16::random("bench_bind2");

    c.bench_function("hdc/bind", |b| b.iter(|| black_box(hv1.bind(&hv2))));
}

fn bench_hdc_bundle(c: &mut Criterion) {
    let mut group = c.benchmark_group("hdc/bundle");

    for size in [5, 10, 50, 100, 500].iter() {
        let hvs: Vec<HV16> = (0..*size)
            .map(|i| HV16::random(&format!("bundle_{}", i)))
            .collect();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| black_box(bundle(&hvs)))
        });
    }

    group.finish();
}

fn bench_hdc_weighted_bundle(c: &mut Criterion) {
    let weighted_hvs: Vec<(HV16, f32)> = (0..100)
        .map(|i| {
            (
                HV16::random(&format!("wbundle_{}", i)),
                (i as f32 + 1.0) / 100.0,
            )
        })
        .collect();

    c.bench_function("hdc/weighted_bundle_100", |b| {
        b.iter(|| black_box(weighted_bundle(&weighted_hvs)))
    });
}

fn bench_hdc_random(c: &mut Criterion) {
    c.bench_function("hdc/random", |b| {
        let mut i = 0u64;
        b.iter(|| {
            i += 1;
            black_box(HV16::random(&format!("rand_{}", i)))
        })
    });
}

// ============================================================================
// LTC BENCHMARKS
// ============================================================================

fn bench_ltc_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("ltc/forward");

    for hidden_size in [32, 64, 128, 256].iter() {
        let config = LtcConfig {
            hidden_size: *hidden_size,
            ..Default::default()
        };
        let mut ltc = LtcCell::new(40, config);
        let input = vec![0.5; 40];

        group.bench_with_input(
            BenchmarkId::from_parameter(hidden_size),
            hidden_size,
            |b, _| {
                b.iter(|| {
                    let state = ltc.forward(&input, 0.01);
                    black_box(state.len())
                })
            },
        );
    }

    group.finish();
}

fn bench_ltc_sequence(c: &mut Criterion) {
    let config = LtcConfig {
        hidden_size: 64,
        ..Default::default()
    };
    let mut ltc = LtcCell::new(40, config);
    let frames: Vec<Vec<f32>> = (0..100)
        .map(|i| (0..40).map(|j| ((i + j) as f32).sin() * 0.5).collect())
        .collect();

    c.bench_function("ltc/sequence_100_frames", |b| {
        b.iter(|| {
            ltc.reset();
            let mut total = 0;
            for frame in &frames {
                let state = ltc.forward(frame, 0.01);
                total += state.len();
            }
            black_box(total)
        })
    });
}

// ============================================================================
// AUDIO BENCHMARKS
// ============================================================================

fn bench_mel_spectrogram(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio/mel_spectrogram");

    for duration in [0.1, 0.5, 1.0, 5.0].iter() {
        let audio = generate_test_audio(*duration, 16000);
        let mut frontend = AudioFrontend::new(AudioConfig::default());

        group.throughput(Throughput::Bytes((audio.len() * 4) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}s", duration)),
            &audio,
            |b, audio| b.iter(|| black_box(frontend.extract_features(audio))),
        );
    }

    group.finish();
}

fn bench_audio_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("audio/projection");

    for duration in [0.5, 1.0, 2.0].iter() {
        let audio = generate_test_audio(*duration, 16000);
        let mut projector = AudioProjector::default_config();

        group.throughput(Throughput::Bytes((audio.len() * 4) as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}s", duration)),
            &audio,
            |b, audio| {
                b.iter(|| {
                    projector.reset();
                    black_box(projector.project(audio))
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// PHONEME RESONATOR BENCHMARKS
// ============================================================================

fn bench_resonator_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("phoneme/resonator_query");

    for num_prototypes in [10, 39, 100, 500].iter() {
        let mut resonator = PhonemeResonator::new();

        // Add prototypes
        for i in 0..*num_prototypes {
            resonator.store(&format!("P{}", i), HV16::random(&format!("proto_{}", i)));
        }

        let query = HV16::random("query");

        group.bench_with_input(
            BenchmarkId::from_parameter(num_prototypes),
            num_prototypes,
            |b, _| b.iter(|| black_box(resonator.query(&query, 5))),
        );
    }

    group.finish();
}

fn bench_resonator_store(c: &mut Criterion) {
    c.bench_function("phoneme/resonator_store", |b| {
        let mut resonator = PhonemeResonator::new();
        let mut i = 0;

        b.iter(|| {
            i += 1;
            resonator.store(&format!("P{}", i), HV16::random(&format!("proto_{}", i)));
        })
    });
}

// ============================================================================
// FULL PIPELINE BENCHMARKS
// ============================================================================

fn bench_full_pipeline(c: &mut Criterion) {
    // Setup
    let audio = generate_test_audio(1.0, 16000);
    let mut projector = AudioProjector::default_config();
    let mut resonator = PhonemeResonator::new();

    // Add some prototypes
    for phoneme in [
        "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH",
        "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH",
        "UH", "UW", "V", "W", "Y", "Z", "ZH",
    ] {
        resonator.store(phoneme, HV16::random(phoneme));
    }

    c.bench_function("pipeline/full_1s", |b| {
        b.iter(|| {
            projector.reset();
            let hvs = projector.project(&audio);

            let mut phonemes = Vec::new();
            for hv in &hvs {
                let results = resonator.query(hv, 1);
                if let Some((phoneme, _)) = results.into_iter().next() {
                    phonemes.push(phoneme);
                }
            }
            black_box(phonemes)
        })
    });
}

// ============================================================================
// CRITERION GROUPS
// ============================================================================

criterion_group!(
    hdc_benches,
    bench_hdc_similarity,
    bench_hdc_bind,
    bench_hdc_bundle,
    bench_hdc_weighted_bundle,
    bench_hdc_random,
);

criterion_group!(ltc_benches, bench_ltc_forward, bench_ltc_sequence,);

criterion_group!(audio_benches, bench_mel_spectrogram, bench_audio_projection,);

criterion_group!(
    phoneme_benches,
    bench_resonator_query,
    bench_resonator_store,
);

criterion_group!(pipeline_benches, bench_full_pipeline,);

criterion_main!(
    hdc_benches,
    ltc_benches,
    audio_benches,
    phoneme_benches,
    pipeline_benches,
);
