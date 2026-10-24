// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Data-dependent chunk ordering for exact Hamming top-1 retrieval.
//!
//! The algorithm is exact: chunks may be visited in any order because
//! Hamming mismatches are non-negative and commutative. A corpus-level
//! variability score ranks informative chunks first, hoping to reject worse
//! candidates earlier without per-candidate metadata. The index-build and
//! query costs are measured separately.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use symthaea_core::hdc::binary_hv::BinaryHV;

const CHUNK_BYTES: usize = 64;
const CHUNK_COUNT: usize = BinaryHV::BYTES / CHUNK_BYTES;
const CORPUS_SIZES: [usize; 4] = [256, 1024, 4096, 16384];
const QUERY_SEED: u64 = 0xC0DE_B10C_9876;
const CORPUS_SEED: u64 = 0x71A1_64BA_5EED;
const LOCALIZED_TAIL_BYTES: usize = 512;

#[derive(Clone, Debug)]
struct ChunkOrder {
    order: Vec<usize>,
    variability: Vec<u64>,
}

impl ChunkOrder {
    /// Build a corpus-only block ranking. For each bit, min(ones, zeros)
    /// measures how many candidates disagree with the majority value. Summing
    /// this quantity ranks chunks by candidate diversity without looking at
    /// query outcomes or candidate winner IDs.
    fn from_corpus(corpus: &[BinaryHV]) -> Self {
        assert!(!corpus.is_empty());
        let mut ones = vec![[0u32; CHUNK_BYTES * 8]; CHUNK_COUNT];

        for candidate in corpus {
            for chunk in 0..CHUNK_COUNT {
                let byte_start = chunk * CHUNK_BYTES;
                for byte_offset in 0..CHUNK_BYTES {
                    let byte = candidate.0[byte_start + byte_offset];
                    for bit in 0..8 {
                        if ((byte >> bit) & 1) != 0 {
                            ones[chunk][byte_offset * 8 + bit] += 1;
                        }
                    }
                }
            }
        }

        let n = corpus.len() as u32;
        let variability: Vec<u64> = ones
            .iter()
            .map(|chunk_ones| {
                chunk_ones
                    .iter()
                    .map(|&count| u64::from(count.min(n - count)))
                    .sum()
            })
            .collect();

        let mut order: Vec<usize> = (0..CHUNK_COUNT).collect();
        order.sort_by(|a, b| {
            variability[*b]
                .cmp(&variability[*a])
                .then_with(|| a.cmp(b))
        });

        Self { order, variability }
    }

    fn natural() -> Self {
        Self {
            order: (0..CHUNK_COUNT).collect(),
            variability: vec![0; CHUNK_COUNT],
        }
    }
}

fn seeded_corpus(count: usize) -> Vec<BinaryHV> {
    (0..count)
        .map(|index| BinaryHV::random(CORPUS_SEED.wrapping_add(index as u64)))
        .collect()
}

/// Query/corpus share an anchor; all candidate variation is restricted to the
/// last eight 64-byte chunks. Natural order examines 24 invariant chunks first,
/// while a diversity-ranked order can inspect varying chunks first.
fn localized_tail_corpus(count: usize) -> Vec<BinaryHV> {
    assert!(count > 0);
    let anchor = BinaryHV::random(CORPUS_SEED.wrapping_add(0x1000));
    let tail_start = BinaryHV::BYTES - LOCALIZED_TAIL_BYTES;
    let mut corpus = Vec::with_capacity(count);
    corpus.push(anchor);

    for index in 1..count {
        let mut bytes = anchor.0;
        let flips = 16 + (index * 53) % 2048;
        for bit_index in 0..flips {
            let bit = (bit_index * 8191 + index * 131) % (LOCALIZED_TAIL_BYTES * 8);
            let absolute_bit = tail_start * 8 + bit;
            bytes[absolute_bit / 8] ^= 1u8 << (absolute_bit % 8);
        }
        corpus.push(BinaryHV(bytes));
    }
    corpus
}

fn localized_tail_query() -> BinaryHV {
    let anchor = BinaryHV::random(CORPUS_SEED.wrapping_add(0x1000));
    let mut bytes = anchor.0;
    let tail_start = BinaryHV::BYTES - LOCALIZED_TAIL_BYTES;
    for bit_index in 0..24 {
        let bit = (bit_index * 8191 + 773) % (LOCALIZED_TAIL_BYTES * 8);
        let absolute_bit = tail_start * 8 + bit;
        bytes[absolute_bit / 8] ^= 1u8 << (absolute_bit % 8);
    }
    BinaryHV(bytes)
}

fn exact_argmin(query: &BinaryHV, corpus: &[BinaryHV]) -> (usize, u32) {
    assert!(!corpus.is_empty());
    let mut best = (0usize, query.hamming_distance(&corpus[0]));
    for (candidate_id, candidate) in corpus.iter().enumerate().skip(1) {
        let distance = query.hamming_distance(candidate);
        if distance < best.1 {
            best = (candidate_id, distance);
        }
    }
    best
}

/// Exact partial-distance search using a chosen block order. Because candidate
/// IDs are visited in increasing order, a partial distance >= incumbent is a
/// safe prune: it cannot produce a lower distance, and a tie cannot beat the
/// lower-ID incumbent. TRACK_BYTES is used only in diagnostics.
fn ordered_early_abandon<const TRACK_BYTES: bool>(
    query: &BinaryHV,
    corpus: &[BinaryHV],
    order: &[usize],
) -> (usize, u32, usize) {
    assert!(!corpus.is_empty());
    assert_eq!(order.len(), CHUNK_COUNT);

    let mut best_id = 0usize;
    let mut best_distance = query.hamming_distance(&corpus[0]);
    let mut bytes_scanned = BinaryHV::BYTES;

    for (candidate_id, candidate) in corpus.iter().enumerate().skip(1) {
        let mut partial_distance = 0u32;
        let mut pruned = false;

        for &chunk in order {
            let start = chunk * CHUNK_BYTES;
            let end = start + CHUNK_BYTES;
            for byte_index in start..end {
                partial_distance +=
                    (query.0[byte_index] ^ candidate.0[byte_index]).count_ones();
            }
            if TRACK_BYTES {
                bytes_scanned += CHUNK_BYTES;
            }

            if partial_distance >= best_distance {
                pruned = true;
                break;
            }
        }

        if !pruned && partial_distance < best_distance {
            best_id = candidate_id;
            best_distance = partial_distance;
        }
    }

    (
        best_id,
        best_distance,
        if TRACK_BYTES { bytes_scanned } else { 0 },
    )
}

fn assert_same(expected: (usize, u32), actual: (usize, u32), context: &str) {
    assert_eq!(expected.0, actual.0, "winner mismatch for {context}");
    assert_eq!(expected.1, actual.1, "distance mismatch for {context}");
}

fn correctness_contract() {
    let random_query = BinaryHV::random(QUERY_SEED);
    let random_corpus = seeded_corpus(257);
    let natural = ChunkOrder::natural();
    let ranked = ChunkOrder::from_corpus(&random_corpus);
    let expected = exact_argmin(&random_query, &random_corpus);
    assert_same(
        expected,
        {
            let got = ordered_early_abandon::<true>(&random_query, &random_corpus, &natural.order);
            (got.0, got.1)
        },
        "random natural order",
    );
    assert_same(
        expected,
        {
            let got = ordered_early_abandon::<true>(&random_query, &random_corpus, &ranked.order);
            (got.0, got.1)
        },
        "random ranked order",
    );

    let query = localized_tail_query();
    let corpus = localized_tail_corpus(257);
    let ranked = ChunkOrder::from_corpus(&corpus);
    let expected = exact_argmin(&query, &corpus);
    assert_same(
        expected,
        {
            let got = ordered_early_abandon::<true>(&query, &corpus, &natural.order);
            (got.0, got.1)
        },
        "localized natural order",
    );
    assert_same(
        expected,
        {
            let got = ordered_early_abandon::<true>(&query, &corpus, &ranked.order);
            (got.0, got.1)
        },
        "localized ranked order",
    );

    // Duplicate exact winners at distinct IDs must preserve the first result.
    let q = BinaryHV::random(0x71E_0D3E);
    let duplicate_corpus = vec![
        BinaryHV::ones(),
        q,
        BinaryHV::zero(),
        q,
        BinaryHV::ones(),
    ];
    let ranked = ChunkOrder::from_corpus(&duplicate_corpus);
    assert_eq!(exact_argmin(&q, &duplicate_corpus), (1, 0));
    for order in [&natural.order[..], &ranked.order[..]] {
        let got = ordered_early_abandon::<true>(&q, &duplicate_corpus, order);
        assert_eq!((got.0, got.1), (1, 0));
    }
}

fn bench_ordered_early_abandon(c: &mut Criterion) {
    correctness_contract();

    let mut group = c.benchmark_group("hdc_ordered_early_abandon_16k");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(500));

    for corpus_size in CORPUS_SIZES {
        for profile in ["random", "localized_tail"] {
            let corpus = match profile {
                "random" => seeded_corpus(corpus_size),
                "localized_tail" => localized_tail_corpus(corpus_size),
                _ => unreachable!(),
            };
            let query = match profile {
                "random" => BinaryHV::random(QUERY_SEED),
                "localized_tail" => localized_tail_query(),
                _ => unreachable!(),
            };
            if profile == "random" {
                assert!(
                    corpus.iter().all(|candidate| candidate != &query),
                    "random-control query unexpectedly duplicates a corpus vector"
                );
            }

            let expected = exact_argmin(&query, &corpus);
            let natural = ChunkOrder::natural();
            let ranked = ChunkOrder::from_corpus(&corpus);
            assert_same(
                expected,
                {
                    let got = ordered_early_abandon::<true>(&query, &corpus, &natural.order);
                    (got.0, got.1)
                },
                "timed natural path",
            );
            assert_same(
                expected,
                {
                    let got = ordered_early_abandon::<true>(&query, &corpus, &ranked.order);
                    (got.0, got.1)
                },
                "timed ranked path",
            );

            for (strategy, order) in [
                ("flat_exact", None),
                ("early_natural", Some(&natural.order)),
                ("early_variability", Some(&ranked.order)),
            ] {
                group.throughput(Throughput::Elements(corpus_size as u64));
                group.bench_with_input(
                    BenchmarkId::new(format!("{profile}_{strategy}_n{corpus_size}"), corpus_size),
                    &corpus_size,
                    |bencher, _| {
                        bencher.iter(|| match order {
                            None => black_box(exact_argmin(black_box(&query), black_box(&corpus))),
                            Some(order) => {
                                let got = ordered_early_abandon::<false>(
                                    black_box(&query),
                                    black_box(&corpus),
                                    black_box(order),
                                );
                                black_box((got.0, got.1))
                            }
                        })
                    },
                );
            }

            // Untimed diagnostics separate algorithmic bytes visited from
            // actual runtime and physical memory traffic.
            let natural_diag =
                ordered_early_abandon::<true>(&query, &corpus, &natural.order);
            let ranked_diag =
                ordered_early_abandon::<true>(&query, &corpus, &ranked.order);
            eprintln!(
                "chunk-order-diagnostic profile={profile} n={corpus_size} \
                 natural_bytes={}/{} ranked_bytes={}/{} \
                 first_ranked_chunk={} first_ranked_variability={}",
                natural_diag.2,
                corpus_size * BinaryHV::BYTES,
                ranked_diag.2,
                corpus_size * BinaryHV::BYTES,
                ranked.order[0],
                ranked.variability[ranked.order[0]],
            );
        }
    }
    group.finish();

    let mut build_group = c.benchmark_group("hdc_chunk_order_build_16k");
    build_group.sample_size(10);
    build_group.warm_up_time(Duration::from_millis(250));
    build_group.measurement_time(Duration::from_millis(500));

    for corpus_size in CORPUS_SIZES {
        for profile in ["random", "localized_tail"] {
            let corpus = match profile {
                "random" => seeded_corpus(corpus_size),
                "localized_tail" => localized_tail_corpus(corpus_size),
                _ => unreachable!(),
            };
            build_group.throughput(Throughput::Elements(corpus_size as u64));
            build_group.bench_with_input(
                BenchmarkId::new(format!("{profile}_n{corpus_size}"), corpus_size),
                &corpus_size,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(ChunkOrder::from_corpus(black_box(&corpus)))
                    })
                },
            );
        }
    }
    build_group.finish();
}

criterion_group!(benches, bench_ordered_early_abandon);
criterion_main!(benches);
