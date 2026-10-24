// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Cache-tiled batched exact BinaryHV search.
//!
//! The experiment compares query-major and corpus-major baselines with a
//! two-dimensional tiled traversal. Candidate and query tile sizes are
//! explicit hypotheses; this benchmark does not assume that a particular
//! cache size or traversal is optimal on every target.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use symthaea_core::hdc::binary_hv::BinaryHV;

const BATCH_SIZES: [usize; 2] = [4, 16];
const CORPUS_SIZES: [usize; 3] = [256, 4096, 16384];
const QUERY_TILE_SIZES: [usize; 2] = [1, 4];
const CANDIDATE_TILE_SIZES: [usize; 3] = [1, 4, 16];
const QUERY_SEED: u64 = 0x71ED_16384;
const CORPUS_SEED: u64 = 0x71E5_C0A5;

/// Deterministic corpus using the canonical BinaryHV representation.
fn seeded_corpus(count: usize) -> Vec<BinaryHV> {
    (0..count)
        .map(|i| BinaryHV::random(CORPUS_SEED.wrapping_add(i as u64)))
        .collect()
}

/// Independent queries are the no-temporal-locality control.
fn independent_queries(count: usize) -> Vec<BinaryHV> {
    (0..count)
        .map(|i| BinaryHV::random(QUERY_SEED.wrapping_add(i as u64)))
        .collect()
}

/// Related queries perturb a common anchor. This is one explicit workload,
/// not a claim that production query streams always have this structure.
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

/// Exact top-1 for one query. Strict comparison retains the lowest candidate
/// index when multiple candidates have the same score.
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

fn query_major_batch(queries: &[BinaryHV], corpus: &[BinaryHV]) -> Vec<(usize, f32)> {
    queries
        .iter()
        .map(|query| query_major_one(query, corpus))
        .collect()
}

/// Candidate outer-loop baseline. Each candidate is compared with every query
/// before advancing to the next candidate.
fn corpus_major_batch(queries: &[BinaryHV], corpus: &[BinaryHV]) -> Vec<(usize, f32)> {
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

/// Two-dimensional blocked traversal.
///
/// The candidate block is the outer tile so its vectors can potentially be
/// reused across several query tiles. The query tile bounds the number of
/// active query vectors in the inner working set. These are locality
/// hypotheses only: wall-clock timings are needed to determine whether a tile
/// helps on a specific CPU and compiler.
fn tiled_corpus_major_batch(
    queries: &[BinaryHV],
    corpus: &[BinaryHV],
    query_tile_size: usize,
    candidate_tile_size: usize,
) -> Vec<(usize, f32)> {
    assert!(!queries.is_empty());
    assert!(!corpus.is_empty());
    assert!(query_tile_size > 0);
    assert!(candidate_tile_size > 0);

    let mut best: Vec<(usize, f32)> = queries
        .iter()
        .map(|query| (0usize, query.similarity(&corpus[0])))
        .collect();

    // Candidate ranges are monotonically increasing. Combined with strict
    // score comparison, this preserves the canonical lowest-index tie rule.
    for candidate_start in (1..corpus.len()).step_by(candidate_tile_size) {
        let candidate_end = (candidate_start + candidate_tile_size).min(corpus.len());

        for query_start in (0..queries.len()).step_by(query_tile_size) {
            let query_end = (query_start + query_tile_size).min(queries.len());

            for candidate_index in candidate_start..candidate_end {
                let candidate = &corpus[candidate_index];
                for query_index in query_start..query_end {
                    let score = queries[query_index].similarity(candidate);
                    if score > best[query_index].1 {
                        best[query_index] = (candidate_index, score);
                    }
                }
            }
        }
    }
    best
}

fn assert_same_answers(expected: &[(usize, f32)], actual: &[(usize, f32)]) {
    assert_eq!(expected.len(), actual.len());
    for (query_index, (expected, actual)) in expected.iter().zip(actual).enumerate() {
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
    // Non-power-of-two corpus count catches assumptions hidden by aligned
    // benchmark sizes. Duplicate best matches test deterministic tie behavior.
    let query = BinaryHV::random(0x71E_001);
    let tie_corpus = vec![
        BinaryHV::ones(),
        query,
        BinaryHV::zero(),
        query,
        BinaryHV::ones(),
    ];
    let tie_queries = vec![query, BinaryHV::zero(), query, BinaryHV::ones()];
    let reference = query_major_batch(&tie_queries, &tie_corpus);
    assert_eq!(reference, corpus_major_batch(&tie_queries, &tie_corpus));

    for query_tile in QUERY_TILE_SIZES {
        for candidate_tile in CANDIDATE_TILE_SIZES {
            let actual = tiled_corpus_major_batch(
                &tie_queries,
                &tie_corpus,
                query_tile,
                candidate_tile,
            );
            assert_eq!(reference, actual);
        }
    }
    assert_eq!(reference[0].0, 1);
    assert_eq!(reference[2].0, 1);

    // Check every benchmark batch size and both query profiles against the
    // canonical query-major answer before any timed iteration.
    let corpus = seeded_corpus(257);
    for batch_size in BATCH_SIZES {
        let independent = independent_queries(batch_size);
        let correlated = correlated_queries(batch_size);
        for queries in [&independent[..], &correlated[..]] {
            let reference = query_major_batch(queries, &corpus);
            assert_same_answers(&reference, &corpus_major_batch(queries, &corpus));
            for query_tile in QUERY_TILE_SIZES {
                for candidate_tile in CANDIDATE_TILE_SIZES {
                    assert_same_answers(
                        &reference,
                        &tiled_corpus_major_batch(
                            queries,
                            &corpus,
                            query_tile,
                            candidate_tile,
                        ),
                    );
                }
            }
        }
    }
}

fn bench_tiled_batched_search(c: &mut Criterion) {
    correctness_contract();

    let mut group = c.benchmark_group("hdc_tiled_batched_exact_search_16k");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(500));

    for corpus_size in CORPUS_SIZES {
        let corpus = seeded_corpus(corpus_size);

        for batch_size in BATCH_SIZES {
            let independent = independent_queries(batch_size);
            let correlated = correlated_queries(batch_size);

            for (profile, queries) in [
                ("independent", independent.as_slice()),
                ("correlated", correlated.as_slice()),
            ] {
                let reference = query_major_batch(queries, &corpus);
                let corpus_major = corpus_major_batch(queries, &corpus);
                assert_same_answers(&reference, &corpus_major);

                group.throughput(Throughput::Elements(batch_size as u64));
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("{profile}_query_major_n{corpus_size}"),
                        batch_size,
                    ),
                    &batch_size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(query_major_batch(
                                black_box(queries),
                                black_box(&corpus),
                            ))
                        })
                    },
                );

                group.throughput(Throughput::Elements(batch_size as u64));
                group.bench_with_input(
                    BenchmarkId::new(
                        format!("{profile}_corpus_major_n{corpus_size}"),
                        batch_size,
                    ),
                    &batch_size,
                    |bencher, _| {
                        bencher.iter(|| {
                            black_box(corpus_major_batch(
                                black_box(queries),
                                black_box(&corpus),
                            ))
                        })
                    },
                );

                for query_tile in QUERY_TILE_SIZES {
                    for candidate_tile in CANDIDATE_TILE_SIZES {
                        let tiled = tiled_corpus_major_batch(
                            queries,
                            &corpus,
                            query_tile,
                            candidate_tile,
                        );
                        assert_same_answers(&reference, &tiled);

                        group.throughput(Throughput::Elements(batch_size as u64));
                        group.bench_with_input(
                            BenchmarkId::new(
                                format!(
                                    "{profile}_tile_q{query_tile}_c{candidate_tile}_n{corpus_size}"
                                ),
                                batch_size,
                            ),
                            &batch_size,
                            |bencher, _| {
                                bencher.iter(|| {
                                    black_box(tiled_corpus_major_batch(
                                        black_box(queries),
                                        black_box(&corpus),
                                        query_tile,
                                        candidate_tile,
                                    ))
                                })
                            },
                        );
                    }
                }
            }
        }
    }
    group.finish();
}

criterion_group!(benches, bench_tiled_batched_search);
criterion_main!(benches);
