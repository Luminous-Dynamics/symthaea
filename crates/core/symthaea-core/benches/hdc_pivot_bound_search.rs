// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pivot-bound exact BinaryHV top-1 search.
//!
//! Triangle inequality gives a lower bound on query-to-candidate Hamming
//! distance from distances to a small set of pivots. The bound is used only
//! to reject candidates that cannot beat the exact incumbent; the winner is
//! still verified with canonical BinaryHV Hamming distance.
//!
//! This benchmark contrasts random, high-dimensional data with a clustered
//! synthetic fixture. Pivot bounds are expected to be workload-dependent.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use symthaea_core::hdc::binary_hv::BinaryHV;

const CORPUS_SIZES: [usize; 4] = [256, 1024, 4096, 16384];
const PIVOT_COUNTS: [usize; 3] = [2, 4, 8];
const CLUSTER_COUNT: usize = 8;
const QUERY_SEED: u64 = 0xDEAD_BEEF_9876;
const CORPUS_SEED: u64 = 0xA11C_C0A5;

#[derive(Clone)]
struct PivotIndex {
    pivots: Vec<BinaryHV>,
    // Candidate-major, contiguous u16 distances. Hamming distance is <= 16384.
    signatures: Vec<u16>,
    pivot_count: usize,
    candidate_count: usize,
}

impl PivotIndex {
    fn build(corpus: &[BinaryHV], pivot_ids: &[usize]) -> Self {
        assert!(!corpus.is_empty());
        assert!(!pivot_ids.is_empty());
        assert!(pivot_ids.iter().all(|&id| id < corpus.len()));

        let pivots: Vec<BinaryHV> = pivot_ids.iter().map(|&id| corpus[id]).collect();
        let pivot_count = pivots.len();
        let mut signatures = Vec::with_capacity(corpus.len() * pivot_count);

        for candidate in corpus {
            for pivot in &pivots {
                signatures.push(candidate.hamming_distance(pivot) as u16);
            }
        }

        Self {
            pivots,
            signatures,
            pivot_count,
            candidate_count: corpus.len(),
        }
    }

    #[inline]
    fn candidate_signature(&self, candidate_index: usize) -> &[u16] {
        let start = candidate_index * self.pivot_count;
        &self.signatures[start..start + self.pivot_count]
    }
}

/// Build an exact lower bound:
/// d(q, x) >= |d(q, p) - d(x, p)| for every pivot p.
#[inline]
fn pivot_lower_bound(query_signature: &[u16], candidate_signature: &[u16]) -> u16 {
    query_signature
        .iter()
        .zip(candidate_signature)
        .map(|(&query_distance, &candidate_distance)| {
            query_distance.abs_diff(candidate_distance)
        })
        .max()
        .unwrap_or(0)
}

/// Returns (winner ID, exact Hamming distance, full-vector candidate evaluations).
///
/// Candidate signatures are scanned first to derive lower bounds. The candidate
/// with the smallest bound is exactly evaluated as the initial incumbent, then
/// every other candidate is either fully checked or safely pruned. The bound
/// never certifies a winner by itself: it only rejects candidates that cannot
/// beat the verified incumbent. Lowest-index tie semantics are preserved even
/// when the best-bound seed is not the lowest candidate ID.
///
/// TRACK_COUNTERS=false compiles the diagnostic counter path out of timed runs.
fn pivot_exact_argmin<const TRACK_COUNTERS: bool>(
    query: &BinaryHV,
    corpus: &[BinaryHV],
    index: &PivotIndex,
) -> (usize, u32, usize) {
    assert_eq!(corpus.len(), index.candidate_count);
    assert!(!corpus.is_empty());

    let query_signature: Vec<u16> = index
        .pivots
        .iter()
        .map(|pivot| query.hamming_distance(pivot) as u16)
        .collect();

    let mut lower_bounds = Vec::with_capacity(corpus.len());
    let mut seed_index = 0usize;
    let mut smallest_bound = u16::MAX;

    for candidate_index in 0..corpus.len() {
        let bound = pivot_lower_bound(
            &query_signature,
            index.candidate_signature(candidate_index),
        );
        lower_bounds.push(bound);

        // Strict comparison leaves the first (lowest-ID) candidate as the seed
        // when multiple candidates share the minimum lower bound.
        if bound < smallest_bound {
            smallest_bound = bound;
            seed_index = candidate_index;
        }
    }

    let mut best_index = seed_index;
    let mut best_distance = query.hamming_distance(&corpus[seed_index]);
    let mut full_evaluations = 1usize;

    for candidate_index in 0..corpus.len() {
        if candidate_index == seed_index {
            continue;
        }

        let lower_bound = lower_bounds[candidate_index] as u32;
        // Exact pruning with lowest-index tie semantics. The bound guarantees
        // actual distance >= lower_bound. On equality, only a later candidate
        // can be pruned without risking a lower-ID tied winner.
        let cannot_win = lower_bound > best_distance
            || (lower_bound == best_distance && candidate_index > best_index);
        if cannot_win {
            continue;
        }

        let distance = query.hamming_distance(&corpus[candidate_index]);
        if TRACK_COUNTERS {
            full_evaluations += 1;
        }
        if distance < best_distance || (distance == best_distance && candidate_index < best_index)
        {
            best_index = candidate_index;
            best_distance = distance;
        }
    }

    (
        best_index,
        best_distance,
        if TRACK_COUNTERS { full_evaluations } else { 0 },
    )
}

fn exact_argmin(query: &BinaryHV, corpus: &[BinaryHV]) -> (usize, u32) {
    assert!(!corpus.is_empty());
    let mut best = (0usize, query.hamming_distance(&corpus[0]));
    for (candidate_index, candidate) in corpus.iter().enumerate().skip(1) {
        let distance = query.hamming_distance(candidate);
        if distance < best.1 {
            best = (candidate_index, distance);
        }
    }
    best
}

fn perturb(anchor: &BinaryHV, flips: usize, offset: usize) -> BinaryHV {
    assert!(flips <= BinaryHV::DIM);
    let mut bytes = anchor.0;
    for bit_index in 0..flips {
        let bit = (bit_index * 8191 + offset) % BinaryHV::DIM;
        bytes[bit / 8] ^= 1u8 << (bit % 8);
    }
    BinaryHV(bytes)
}

fn random_corpus(count: usize) -> Vec<BinaryHV> {
    (0..count)
        .map(|index| BinaryHV::random(CORPUS_SEED.wrapping_add(index as u64)))
        .collect()
}

/// Eight coherent clusters, interleaved by member rank. This prevents a
/// corpus-order shortcut from selecting one representative per cluster.
fn clustered_corpus(count: usize) -> Vec<BinaryHV> {
    assert_eq!(count % CLUSTER_COUNT, 0);
    let per_cluster = count / CLUSTER_COUNT;
    let anchors: Vec<BinaryHV> = (0..CLUSTER_COUNT)
        .map(|cluster| BinaryHV::random(CORPUS_SEED.wrapping_add(10_000 + cluster as u64)))
        .collect();

    let mut members_by_cluster: Vec<Vec<BinaryHV>> = Vec::with_capacity(CLUSTER_COUNT);
    for (cluster, anchor) in anchors.iter().enumerate() {
        let mut members = Vec::with_capacity(per_cluster);
        members.push(*anchor);
        for local_index in 1..per_cluster {
            let flips = 32 + (local_index * 37) % 2048;
            let offset = cluster * 2039 + local_index * 131;
            members.push(perturb(anchor, flips, offset));
        }
        members_by_cluster.push(members);
    }

    let mut corpus = Vec::with_capacity(count);
    for local_index in 0..per_cluster {
        for members in &members_by_cluster {
            corpus.push(members[local_index]);
        }
    }
    corpus
}

/// Cheap order-sensitive control. On an interleaved corpus, these positions may
/// select multiple pivots from the same cluster.
fn evenly_spaced_pivot_ids(corpus_len: usize, pivot_count: usize) -> Vec<usize> {
    assert!(pivot_count > 0 && pivot_count <= CLUSTER_COUNT);
    assert_eq!(corpus_len % CLUSTER_COUNT, 0);
    let per_cluster = corpus_len / CLUSTER_COUNT;
    (0..pivot_count)
        .map(|pivot| (pivot * CLUSTER_COUNT / pivot_count) * per_cluster)
        .collect()
}

/// Deterministic farthest-first (maximin) selection. Each new pivot maximizes
/// distance to its nearest selected pivot. This can cover separated clusters,
/// but its extra exact distance work is included in index-build timing.
fn farthest_first_pivot_ids(corpus: &[BinaryHV], pivot_count: usize) -> Vec<usize> {
    assert!(!corpus.is_empty());
    assert!(pivot_count > 0 && pivot_count <= CLUSTER_COUNT.min(corpus.len()));

    let mut selected = vec![0usize];
    let mut nearest_distance: Vec<u32> = corpus
        .iter()
        .map(|candidate| candidate.hamming_distance(&corpus[0]))
        .collect();

    while selected.len() < pivot_count {
        let mut next_id = None;
        let mut farthest_distance = 0u32;
        for (candidate_id, &distance) in nearest_distance.iter().enumerate() {
            if selected.contains(&candidate_id) {
                continue;
            }
            if next_id.is_none() || distance > farthest_distance {
                next_id = Some(candidate_id);
                farthest_distance = distance;
            }
        }
        let next_id = next_id.expect("pivot count is bounded by corpus length");
        selected.push(next_id);
        let pivot = &corpus[next_id];

        for (candidate_id, candidate) in corpus.iter().enumerate() {
            let distance = candidate.hamming_distance(pivot);
            nearest_distance[candidate_id] = nearest_distance[candidate_id].min(distance);
        }
    }
    selected
}
fn verify_correctness() {
    let query = BinaryHV::random(QUERY_SEED);
    let corpus = random_corpus(257);
    for pivot_count in PIVOT_COUNTS {
        let selections = [
            evenly_spaced_pivot_ids(256, pivot_count),
            farthest_first_pivot_ids(&corpus, pivot_count),
        ];
        for ids in selections {
            // Use a non-power-of-two query corpus and choose valid pivot IDs.
            let ids: Vec<usize> = ids.into_iter().filter(|&id| id < corpus.len()).collect();
            let index = PivotIndex::build(&corpus, &ids);
            assert_eq!(
                exact_argmin(&query, &corpus),
                {
                    let result = pivot_exact_argmin::<true>(&query, &corpus, &index);
                    (result.0, result.1)
                },
                "pivot search differs from exhaustive search"
            );
        }
    }

    // Duplicate exact matches at non-adjacent IDs; lower-ID tie wins regardless
    // of which candidate the minimum-bound seed chooses.
    let q = BinaryHV::random(0x71E_71E);
    let mut ties = vec![
        BinaryHV::random(1),
        BinaryHV::random(2),
        BinaryHV::random(3),
        q,
        BinaryHV::random(4),
        q,
        BinaryHV::random(5),
    ];
    // Ensure pivot IDs are valid and include candidates around both duplicates.
    let tie_index = PivotIndex::build(&ties, &[0, 3, 6]);
    assert_eq!(exact_argmin(&q, &ties), (3, 0));
    let actual = pivot_exact_argmin::<true>(&q, &ties, &tie_index);
    assert_eq!((actual.0, actual.1), (3, 0));
    ties[0] = q;
    let tie_index = PivotIndex::build(&ties, &[0, 3, 6]);
    assert_eq!(exact_argmin(&q, &ties), (0, 0));
    let actual = pivot_exact_argmin::<true>(&q, &ties, &tie_index);
    assert_eq!((actual.0, actual.1), (0, 0));

    // A non-identical equal-distance tie that forces the best-bound seed to be
    // candidate 2. Candidate 1 has LB == best distance, but its lower ID must
    // prevent pruning so it can replace candidate 2 under the public tie rule.
    let zero = BinaryHV::zero();
    let mut pivot = BinaryHV::zero();
    for bit in 0..5 {
        pivot.0[bit / 8] |= 1u8 << (bit % 8);
    }
    let mut candidate_one = BinaryHV::zero();
    candidate_one.0[0] |= 1u8 << 5;
    candidate_one.0[0] |= 1u8 << 6;
    let mut candidate_two = BinaryHV::zero();
    candidate_two.0[0] |= 1u8;
    candidate_two.0[0] |= 1u8 << 5;
    let tie_candidates = vec![pivot, candidate_one, candidate_two];
    let tie_index = PivotIndex::build(&tie_candidates, &[0]);
    assert_eq!(exact_argmin(&zero, &tie_candidates), (1, 2));
    let actual = pivot_exact_argmin::<true>(&zero, &tie_candidates, &tie_index);
    assert_eq!((actual.0, actual.1), (1, 2));
}

fn bench_pivot_exact_search(c: &mut Criterion) {
    verify_correctness();

    let mut group = c.benchmark_group("hdc_pivot_bound_exact_search_16k");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(500));

    for corpus_size in CORPUS_SIZES {
        for profile in ["random", "clustered"] {
            let corpus = match profile {
                "random" => random_corpus(corpus_size),
                "clustered" => clustered_corpus(corpus_size),
                _ => unreachable!(),
            };
            let query = match profile {
                "random" => BinaryHV::random(QUERY_SEED),
                "clustered" => perturb(&corpus[5], 32, 491),
                _ => unreachable!(),
            };
            let expected = exact_argmin(&query, &corpus);

            group.throughput(Throughput::Elements(corpus_size as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("{profile}_flat_n{corpus_size}"), corpus_size),
                &corpus_size,
                |bencher, _| {
                    bencher.iter(|| black_box(exact_argmin(black_box(&query), black_box(&corpus))))
                },
            );

            for pivot_count in PIVOT_COUNTS {
                let selections = [
                    ("evenly_spaced", evenly_spaced_pivot_ids(corpus_size, pivot_count)),
                    ("farthest_first", farthest_first_pivot_ids(&corpus, pivot_count)),
                ];

                for (selector, ids) in selections {
                    let index = PivotIndex::build(&corpus, &ids);
                    let actual = pivot_exact_argmin::<true>(&query, &corpus, &index);
                    assert_eq!((actual.0, actual.1), expected);

                    // Untimed evidence counters: exact full-vector evaluations and
                    // compact signature size. Not physical DRAM-read measurements.
                    let (_, _, full_evaluations) =
                        pivot_exact_argmin::<true>(&query, &corpus, &index);
                    let metadata_bytes = index.signatures.len() * std::mem::size_of::<u16>();
                    eprintln!(
                        "pivot-diagnostic profile={profile} selector={selector} n={corpus_size} pivots={pivot_count} exact_vectors={full_evaluations}/{} signature_bytes={} result_id={} distance={}",
                        corpus_size, metadata_bytes, actual.0, actual.1,
                    );

                    group.throughput(Throughput::Elements(corpus_size as u64));
                    group.bench_with_input(
                        BenchmarkId::new(
                            format!("{profile}_{selector}_pivot{pivot_count}_n{corpus_size}"),
                            corpus_size,
                        ),
                        &corpus_size,
                        |bencher, _| {
                            bencher.iter(|| {
                                black_box(pivot_exact_argmin::<false>(
                                    black_box(&query),
                                    black_box(&corpus),
                                    black_box(&index),
                                ))
                            })
                        },
                    );
                }
            }
        }
    }
    group.finish();

    // Build cost is separate from repeated query latency, making the amortization
    // boundary explicit rather than pretending index construction is free.
    let mut build_group = c.benchmark_group("hdc_pivot_index_build_16k");
    build_group.sample_size(10);
    build_group.warm_up_time(Duration::from_millis(250));
    build_group.measurement_time(Duration::from_millis(500));
    for corpus_size in CORPUS_SIZES {
        let corpus = random_corpus(corpus_size);
        for pivot_count in PIVOT_COUNTS {
            for selector in ["evenly_spaced", "farthest_first"] {
                build_group.throughput(Throughput::Elements(corpus_size as u64));
                build_group.bench_with_input(
                    BenchmarkId::new(
                        format!("n{corpus_size}_p{pivot_count}_{selector}"),
                        corpus_size,
                    ),
                    &corpus_size,
                    |bencher, _| {
                        bencher.iter(|| {
                            let ids = match selector {
                                "evenly_spaced" => {
                                    evenly_spaced_pivot_ids(corpus.len(), pivot_count)
                                }
                                "farthest_first" => {
                                    farthest_first_pivot_ids(&corpus, pivot_count)
                                }
                                _ => unreachable!(),
                            };
                            black_box(PivotIndex::build(black_box(&corpus), black_box(&ids)))
                        })
                    },
                );
            }
        }
    }
    build_group.finish();
}

criterion_group!(benches, bench_pivot_exact_search);
criterion_main!(benches);
