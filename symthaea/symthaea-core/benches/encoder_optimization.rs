// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Encoder Optimization Benchmark Suite
//!
//! Benchmarks for text encoding pipeline components in symthaea-core:
//! 1. Bipolar conversion: BinaryHV::to_bipolar() (f32) vs to_bipolar_i8() (i8)
//! 2. Word encoding: TextEncoder::encode_word() cold vs warm (cached) paths
//! 3. Sentence encoding: TextEncoder::encode_sentence() short (3 words) vs medium (10 words)
//! 4. Full cycle: PredictiveHdcEncoder::encode() end-to-end
//!
//! Run with: cargo bench --bench encoder_optimization -p symthaea-core

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::predictive_encoder::{PredictiveEncoderConfig, PredictiveHdcEncoder};
use symthaea_core::hdc::text_encoder::{TextEncoder, TextEncoderConfig};

// ============================================================================
// 1. BIPOLAR CONVERSION
// ============================================================================
//
// BinaryHV stores 16,384 bits packed into 2KB. Unpacking to bipolar form is
// needed for continuous operations. Compare f32 output (64KB) vs i8 output
// (16KB) to quantify the memory/throughput benefit of the compact path.

fn bench_bipolar_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("bipolar_conversion");

    let hv = BinaryHV::random(42);

    group.bench_function("to_bipolar_f32", |b| {
        b.iter(|| black_box(black_box(&hv).to_bipolar()))
    });

    group.bench_function("to_bipolar_i8", |b| {
        b.iter(|| black_box(black_box(&hv).to_bipolar_i8()))
    });

    group.finish();
}

// ============================================================================
// 2. WORD ENCODING
// ============================================================================
//
// TextEncoder maintains a HashMap cache of word embeddings. First encode of a
// word computes character n-grams and hashes; subsequent calls return an
// Arc-clone (O(1)). Measuring cold vs warm quantifies cache effectiveness.

fn bench_word_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("word_encoding");

    // Cold path: each iteration uses a fresh encoder so every word is a cache miss.
    group.bench_function("cold_new_word", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for i in 0..iters {
                let mut encoder =
                    TextEncoder::new(TextEncoderConfig::default()).expect("encoder init");
                // Use a unique word per iteration to guarantee cache miss
                let word = format!("benchword{}", i);
                let start = std::time::Instant::now();
                let _ = black_box(encoder.encode_word(black_box(&word)));
                total += start.elapsed();
            }
            total
        })
    });

    // Warm path: single encoder, same word repeated (cache hit after first call).
    group.bench_function("warm_cached_word", |b| {
        let mut encoder = TextEncoder::new(TextEncoderConfig::default()).expect("encoder init");
        // Prime the cache
        let _ = encoder.encode_word("consciousness");
        b.iter(|| black_box(encoder.encode_word(black_box("consciousness"))))
    });

    group.finish();
}

// ============================================================================
// 3. SENTENCE ENCODING
// ============================================================================
//
// Sentence encoding splits text into words, encodes each, applies positional
// permutation, and bundles. Cost scales roughly linearly with word count.
// Benchmarking short vs medium sentences reveals per-word overhead.

fn bench_sentence_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("sentence_encoding");

    let short_sentence = "the red ball";
    let medium_sentence = "the quick brown fox jumps over the lazy sleeping dog";

    // Short sentence (3 words) - warm encoder (pre-encode to fill caches)
    group.bench_function(BenchmarkId::new("words", 3), |b| {
        let mut encoder = TextEncoder::new(TextEncoderConfig::default()).expect("encoder init");
        // Warm up: encode once so word caches are primed
        let _ = encoder.encode_sentence(short_sentence);
        b.iter(|| black_box(encoder.encode_sentence(black_box(short_sentence))))
    });

    // Medium sentence (10 words) - warm encoder
    group.bench_function(BenchmarkId::new("words", 10), |b| {
        let mut encoder = TextEncoder::new(TextEncoderConfig::default()).expect("encoder init");
        let _ = encoder.encode_sentence(medium_sentence);
        b.iter(|| black_box(encoder.encode_sentence(black_box(medium_sentence))))
    });

    // Cold sentence encoding: fresh encoder each time (worst case)
    group.bench_function(BenchmarkId::new("cold_words", 10), |b| {
        b.iter(|| {
            let mut encoder = TextEncoder::new(TextEncoderConfig::default()).expect("encoder init");
            black_box(encoder.encode_sentence(black_box(medium_sentence)))
        })
    });

    group.finish();
}

// ============================================================================
// 4. FULL CYCLE (PredictiveHdcEncoder)
// ============================================================================
//
// PredictiveHdcEncoder wraps TextEncoder with attention modulation and
// prediction error tracking. This is the actual hot path in the cognitive
// loop. Benchmarking reveals the overhead of the prediction layer on top of
// raw text encoding.

fn bench_full_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_cycle");

    let short_input = "hello world";
    let medium_input = "the agent perceives a bright red object moving quickly to the left";

    // Short text input
    group.bench_function(BenchmarkId::new("predictive_encode", "short"), |b| {
        let mut encoder =
            PredictiveHdcEncoder::new(PredictiveEncoderConfig::default()).expect("encoder init");
        // Warm up
        let _ = encoder.encode(short_input);
        b.iter(|| black_box(encoder.encode(black_box(short_input))))
    });

    // Medium text input (12 words)
    group.bench_function(BenchmarkId::new("predictive_encode", "medium"), |b| {
        let mut encoder =
            PredictiveHdcEncoder::new(PredictiveEncoderConfig::default()).expect("encoder init");
        let _ = encoder.encode(medium_input);
        b.iter(|| black_box(encoder.encode(black_box(medium_input))))
    });

    // Measure prediction convergence: repeated identical input should reduce error over time.
    // This benchmarks steady-state performance after the predictor has adapted.
    group.bench_function("predictive_encode_converged", |b| {
        let mut encoder =
            PredictiveHdcEncoder::new(PredictiveEncoderConfig::default()).expect("encoder init");
        // Run 50 cycles to let the predictor converge
        for _ in 0..50 {
            let _ = encoder.encode("repeated stimulus for convergence");
        }
        b.iter(|| black_box(encoder.encode(black_box("repeated stimulus for convergence"))))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_bipolar_conversion,
    bench_word_encoding,
    bench_sentence_encoding,
    bench_full_cycle,
);
criterion_main!(benches);
