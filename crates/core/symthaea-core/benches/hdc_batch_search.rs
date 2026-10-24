// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Batched exact BinaryHV search: query-major vs corpus-major traversal.
//!
//! The benchmark holds the corpus fixed and changes batch size and corpus size.
//! It compares two exact loop orders:
//! - query-major: finish one top-1 scan before starting the next query;
//! - corpus-major: visit each candidate and update the incumbent for every query.
//!
//! This measures host software wall time only. It does not claim physical memory
//! traffic or joules, and does not model DMA/accelerator launch costs.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use symthaea_core::hdc::binary_hv::BinaryHV;

const BATCH_SIZES: [usize; 4] = [1, 4, 8, 16];
const CORPUS_SIZES: [usize; 4] = [256, 1024, 4096, 16384];
const QUERY_SEED: u64 = 0xBA7C_16384;
const CORPUS_SEED: u64 = 0xC0A5_16K;

/// Deterministic binary corpus with canonical BinaryHV encodings.
fn seeded_corpus(count: usize) -> Vec<BinaryHV> {
    (0..count)
        .map(|i| BinaryHV::random(CORPUS_SEED.wrapping_add(i as u64)))
        .collect()
}

/// Independent query streams establish a non-locality baseline.
fn independent_queries(count: usize) -> Vec<BinaryHV> {
    (0..count)
        .map(|i| BinaryHV::random(QUERY_SEED.wrapping_add(i as u64)))
        .collect()
}

/// Correlated query batch: each query is a small deterministic perturbation
/// of the same anchor. This is a workload fixture, not a claim that all real
/// query streams have this correlation structure.
fn correlated_queries(count: usize) -> Vec<BinaryHV> {
    let anchor = BinaryHV::random(QUERY_SEED);
    (0..count)
        .map(|query_index| {
            let mut bytes = anchor.0;
            let flips = query_index * 64;
            for bit_index in 0..flips {
                let bit = (bit_index * 8191 + query_index * 127) % BinaryHV::DIM;
                bytes[bit / 8] ^= 1u8 << (bit % 8);
            }
            BinaryHV(bytes)
        })
        .collect()
}

/// Exact top-1 for one query. Strict comparison preserves the first candidate
/// on equal scores because candidates are traversed in increasing ID order.
fn query_major_one(query: &BinaryHV, corpus: &[BinaryHV]) -> (usize, f32) {
    assert!(!corpus.is_empty());
    let mut best = (0usize, query.similarity(&corpus[0]));
    for (index, candidate) in corpus.iter().enumerate().skip(1) {
        let score = query.similarity(candidate);
        if score > best.1 {
            best = (index, score);
        }
    }
    best
}

/// Independent-query reference: scan the full corpus for each query.
fn query_major_batch(
    queries: &[BinaryHV],
    corpus: &[BinaryHV],
) -> Vec<(usize, f32)> {
    queries
        .iter()
        .map(|query| query_major_one(query, corpus))
        .collect()
}

/// Corpus-major traversal: process each candidate once in the outer loop,
/// updating every query's exact incumbent while candidate bytes are hot.
fn corpus_major_batch(
    queries: &[BinaryHV],
    corpus: &[BinaryHV],
) -> Vec<(usize, f32)> {
    assert!(!queries.is_empty());
    assert!(!corpus.is_empty());

    let mut best: Vec<(usize, f32)> = queries
        .iter()
        .map(|query| (0usize, query.similarity(&corpus[0])))
        .collect();

    for (candidate_index, candidate) in corpus.iter().enumerate().skip(1) {
        for (query_index, query) in queries.iter().enumerate() {
            let score = query.similarity(candidate);
            if score > best[query_index].1 {
                best[query_index] = (candidate_index, score);
            }
        }
    }
    best
}

fn assert_same_answers(
    queries: &[BinaryHV],
    corpus: &[BinaryHV],
    candidate: &[(usize, f32)],
) {
    let reference = query_major_batch(queries, corpus);
    assert_eq!(reference.len(), candidate.len());
    for (query_index, (expected, actual)) in reference.iter().zip(candidate).enumerate() {
        assert_eq!(
            expected.0, actual.0,
            "exact winner mismatch for query {query_index}"
        );
        assert!(
            (expected.1 - actual.1).abs() <= 1.0e-6,
            "exact score mismatch for query {query_index}"
        );
    }
}

fn correctness_contract() {
    for batch_size in BATCH_SIZES {
        let corpus = seeded_corpus(256);
        let independent = independent_queries(batch_size);
        let correlated = correlated_queries(batch_size);
        assert_same_answers(
            &independent,
            &corpus,
            &corpus_major_batch(&independent, &corpus),
        );
        assert_same_answers(
            &correlated,
            &corpus,
            &corpus_major_batch(&correlated, &corpus),
        );
    }

    // Explicit tie rule: duplicate candidates with equal exact match must
    // always return the lowest corpus index for every query in the batch.
    let query = BinaryHV::random(0x71E_001);
    let corpus = vec![
        BinaryHV::ones(),
        query,
        BinaryHV::zero(),
        query,
        BinaryHV::ones(),
    ];
    let queries = vec![query, BinaryHV::zero(), query, BinaryHV::ones()];
    let reference = query_major_batch(&queries, &corpus);
    let candidate_major = corpus_major_batch(&queries, &corpus);
    assert_eq!(reference, candidate_major);
    assert_eq!(reference[0].0, 1);
    assert_eq!(reference[2].0, 1);
}

fn bench_batched_search(c: &mut Criterion) {
    correctness_contract();

    let mut group = c.benchmark_group("hdc_batched_exact_search_16k");
    group.sample_size(15);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(1));

    for corpus_size in CORPUS_SIZES {
        let corpus = seeded_corpus(corpus_size);
        for batch_size in BATCH_SIZES {
            let independent = independent_queries(batch_size);
            let correlated = correlated_queries(batch_size);

            // Exact equality is checked outside the timed section for each
            // query profile; score vectors returned from timed code are black-boxed.
            assert_same_answers(
                &independent,
                &corpus,
                &corpus_major_batch(&independent, &corpus),
            );
            assert_same_answers(
                &correlated,
                &corpus,
                &corpus_major_batch(&correlated, &corpus),
            );

            group.throughput(Throughput::Elements(batch_size as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    format!("independent_query_major_corpus_{corpus_size}"),
                    batch_size,
                ),
                &batch_size,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(query_major_batch(
                            black_box(&independent),
                            black_box(&corpus),
                        ))
                    })
                },
            );
            group.throughput(Throughput::Elements(batch_size as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    format!("independent_corpus_major_corpus_{corpus_size}"),
                    batch_size,
                ),
                &batch_size,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(corpus_major_batch(
                            black_box(&independent),
                            black_box(&corpus),
                        ))
                    })
                },
            );
            group.throughput(Throughput::Elements(batch_size as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    format!("correlated_query_major_corpus_{corpus_size}"),
                    batch_size,
                ),
                &batch_size,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(query_major_batch(
                            black_box(&correlated),
                            black_box(&corpus),
                        ))
                    })
                },
            );
            group.throughput(Throughput::Elements(batch_size as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    format!("correlated_corpus_major_corpus_{corpus_size}"),
                    batch_size,
                ),
                &batch_size,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(corpus_major_batch(
                            black_box(&correlated),
                            black_box(&corpus),
                        ))
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_batched_search);
criterion_main!(benches);
