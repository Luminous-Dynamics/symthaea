// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semantic Encoder Benchmarks
//!
//! Benchmarks for semantic encoding subsystems:
//! - CharNgramEncoder (baseline, fast, no semantics)
//! - CachedSemanticEncoder (with cache hits/misses)
//! - RandomProjection (dimensionality expansion)
//! - Semantic similarity computation
//! - Batch encoding operations
//!
//! Run with: `cargo bench --bench semantic_benchmark`

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use symthaea::hdc::HDC_DIMENSION;
use symthaea::hdc::semantic_encoder::{
    CachedSemanticEncoder, CharNgramEncoder, RandomProjection, SemanticEncoder,
};

// =============================================================================
// TEST DATA
// =============================================================================

const SHORT_TEXTS: &[&str] = &[
    "Hello world",
    "NixOS is great",
    "Compile error",
    "Git commit",
];

const MEDIUM_TEXTS: &[&str] = &[
    "The quick brown fox jumps over the lazy dog",
    "A configuration management system based on functional programming",
    "Memory is not storage, memory is reconstruction",
    "Holographic hypervector encoding preserves semantic similarity",
];

const LONG_TEXTS: &[&str] = &[
    "In the field of hyperdimensional computing, vectors with thousands of dimensions \
     are used to represent complex concepts. These high-dimensional vectors exhibit \
     unique mathematical properties that make them ideal for cognitive computing.",
    "The NixOS operating system provides reproducible builds through a purely functional \
     package manager. By treating packages as immutable values, it ensures that builds \
     are deterministic and can be reproduced exactly on any machine.",
    "Consciousness emerges from the integration of information across multiple brain \
     regions. The Global Workspace Theory suggests that conscious awareness arises when \
     information becomes globally available through a neural broadcasting mechanism.",
    "Episodic memory encodes personal experiences with temporal context, allowing us to \
     mentally travel through time. The hippocampus plays a crucial role in consolidating \
     these memories from short-term to long-term storage during sleep.",
];

// =============================================================================
// CHAR N-GRAM ENCODER (BASELINE)
// =============================================================================

fn bench_char_ngram_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("char_ngram_encoder");
    group.sample_size(100);

    let encoder = CharNgramEncoder::new(42);

    // Different text lengths
    group.bench_function("short_text", |b| {
        b.iter(|| black_box(encoder.encode(SHORT_TEXTS[0])))
    });

    group.bench_function("medium_text", |b| {
        b.iter(|| black_box(encoder.encode(MEDIUM_TEXTS[0])))
    });

    group.bench_function("long_text", |b| {
        b.iter(|| black_box(encoder.encode(LONG_TEXTS[0])))
    });

    // Batch encoding
    for batch_size in [4, 10, 50] {
        let texts: Vec<&str> = MEDIUM_TEXTS
            .iter()
            .cycle()
            .take(batch_size)
            .copied()
            .collect();

        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_encode", batch_size),
            &texts,
            |b, texts| b.iter(|| black_box(encoder.encode_batch(texts))),
        );
    }

    group.finish();
}

/// Benchmark semantic similarity computation
fn bench_char_ngram_similarity(c: &mut Criterion) {
    let mut group = c.benchmark_group("char_ngram_similarity");
    group.sample_size(100);

    let encoder = CharNgramEncoder::new(42);

    // Pre-encode for similarity benchmarks
    let hv1 = encoder.encode("The quick brown fox");
    let hv2 = encoder.encode("A fast brown dog");

    group.bench_function("similarity_pre_encoded", |b| {
        b.iter(|| black_box(hv1.similarity(&hv2)))
    });

    group.bench_function("similarity_from_text", |b| {
        b.iter(|| black_box(encoder.similarity("The quick brown fox", "A fast brown dog")))
    });

    // Multiple similarity comparisons
    for n_comparisons in [10, 50, 100] {
        let pairs: Vec<(&str, &str)> = MEDIUM_TEXTS
            .iter()
            .cycle()
            .zip(MEDIUM_TEXTS.iter().cycle().skip(1))
            .take(n_comparisons)
            .map(|(&a, &b)| (a, b))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batch_similarity", n_comparisons),
            &pairs,
            |b, pairs| {
                b.iter(|| {
                    pairs
                        .iter()
                        .map(|(a, b)| encoder.similarity(a, b))
                        .sum::<f64>()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// CACHED SEMANTIC ENCODER
// =============================================================================

fn bench_cached_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("cached_semantic_encoder");
    group.sample_size(50);

    let encoder = CachedSemanticEncoder::new(384); // Standard embedding dim

    // Add some cached entries
    for text in MEDIUM_TEXTS {
        // Simulate cached embedding (384-dim vector)
        let fake_embedding: Vec<f32> = (0..384).map(|i| (i as f32 * 0.001).sin()).collect();
        encoder.add_embedding(text, fake_embedding);
    }

    // Cache hit (should be fast)
    group.bench_function("cache_hit", |b| {
        b.iter(|| black_box(encoder.encode(MEDIUM_TEXTS[0])))
    });

    // Cache miss (falls back to CharNgram)
    group.bench_function("cache_miss", |b| {
        b.iter(|| black_box(encoder.encode("This text is definitely not in the cache at all")))
    });

    // Mixed workload (50% hit rate)
    let mixed_texts = vec![
        MEDIUM_TEXTS[0],
        "Uncached text one",
        MEDIUM_TEXTS[1],
        "Uncached text two",
        MEDIUM_TEXTS[2],
        "Uncached text three",
    ];

    group.bench_function("mixed_hit_miss", |b| {
        b.iter(|| {
            for text in &mixed_texts {
                black_box(encoder.encode(text));
            }
        })
    });

    group.finish();
}

// =============================================================================
// RANDOM PROJECTION
// =============================================================================

fn bench_random_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_projection");
    group.sample_size(50);

    // Different input dimensions
    for input_dim in [128, 256, 384, 768] {
        let projection = RandomProjection::new(input_dim, HDC_DIMENSION, 42);
        let input: Vec<f32> = (0..input_dim).map(|i| (i as f32 * 0.01).sin()).collect();

        group.throughput(Throughput::Elements(input_dim as u64));
        group.bench_with_input(
            BenchmarkId::new("project", input_dim),
            &(&projection, &input),
            |b, (proj, input)| b.iter(|| black_box(proj.project(input))),
        );
    }

    // Projection initialization
    for input_dim in [384, 768] {
        group.bench_with_input(
            BenchmarkId::new("init", input_dim),
            &input_dim,
            |b, &dim| b.iter(|| black_box(RandomProjection::new(dim, HDC_DIMENSION, 42))),
        );
    }

    group.finish();
}

// =============================================================================
// TEXT LENGTH SCALING
// =============================================================================

fn bench_encoding_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoding_scaling");
    group.sample_size(50);

    let encoder = CharNgramEncoder::new(42);

    // Generate texts of different lengths
    let base = "hello world ";
    for n_repeats in [1, 5, 10, 50, 100] {
        let text = base.repeat(n_repeats);
        let text_len = text.len();

        group.throughput(Throughput::Bytes(text_len as u64));
        group.bench_with_input(
            BenchmarkId::new("text_length", text_len),
            &text,
            |b, text| b.iter(|| black_box(encoder.encode(text))),
        );
    }

    group.finish();
}

// =============================================================================
// SEMANTIC SPACE OPERATIONS
// =============================================================================

fn bench_semantic_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("semantic_operations");
    group.sample_size(50);

    let encoder = CharNgramEncoder::new(42);

    // Encode corpus for search operations
    let corpus_size = 100;
    let corpus: Vec<_> = (0..corpus_size)
        .map(|i| encoder.encode(&format!("Document number {} about topic {}", i, i % 10)))
        .collect();

    let query = encoder.encode("Document about topic 5");

    // Linear search for most similar
    group.bench_function("linear_search_100", |b| {
        b.iter(|| {
            corpus
                .iter()
                .map(|hv| query.similarity(hv))
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        })
    });

    // Top-k search
    for k in [3, 5, 10] {
        group.bench_with_input(BenchmarkId::new("top_k_search", k), &k, |b, &k| {
            b.iter(|| {
                let mut similarities: Vec<_> = corpus
                    .iter()
                    .map(|hv| query.similarity(hv))
                    .enumerate()
                    .collect();
                similarities.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
                similarities.truncate(k);
                black_box(similarities)
            })
        });
    }

    // Threshold-based retrieval
    for threshold in [0.3, 0.5, 0.7] {
        group.bench_with_input(
            BenchmarkId::new("threshold_retrieval", format!("{:.1}", threshold)),
            &threshold,
            |b, &threshold| {
                b.iter(|| {
                    corpus
                        .iter()
                        .filter(|hv| query.similarity(hv) > threshold)
                        .count()
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// COMPREHENSIVE ENCODE-STORE-SEARCH CYCLE
// =============================================================================

fn bench_full_semantic_cycle(c: &mut Criterion) {
    let mut group = c.benchmark_group("semantic_full_cycle");
    group.sample_size(20);

    let encoder = CharNgramEncoder::new(42);

    for corpus_size in [50, 100, 200] {
        group.throughput(Throughput::Elements(corpus_size as u64));

        group.bench_with_input(
            BenchmarkId::new("encode_and_search", corpus_size),
            &corpus_size,
            |b, &size| {
                b.iter(|| {
                    // Encode corpus
                    let corpus: Vec<_> = (0..size)
                        .map(|i| encoder.encode(&format!("Document {} topic {}", i, i % 10)))
                        .collect();

                    // Perform 5 searches
                    let mut total_matches = 0;
                    for q in 0..5 {
                        let query = encoder.encode(&format!("Search query {}", q));
                        total_matches += corpus
                            .iter()
                            .filter(|hv| query.similarity(hv) > 0.3)
                            .count();
                    }

                    black_box(total_matches)
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// CRITERION CONFIGURATION
// =============================================================================

criterion_group!(
    semantic_benches,
    bench_char_ngram_encode,
    bench_char_ngram_similarity,
    bench_cached_encoder,
    bench_random_projection,
    bench_encoding_scaling,
    bench_semantic_operations,
    bench_full_semantic_cycle,
);

criterion_main!(semantic_benches);
