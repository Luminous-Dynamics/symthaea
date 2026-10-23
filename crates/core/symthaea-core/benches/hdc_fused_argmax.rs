// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Exact HDC similarity-to-argmax benchmark.
//!
//! Complements hdc_simd_compare by returning and cross-checking the winning
//! candidate ID, pinning lowest-index tie handling, including small batches,
//! and separating packed-representation preparation from repeated search.
//!
//! Run:
//!   cargo bench -p symthaea-core --bench hdc_fused_argmax
//!
//! This is a CPU/software benchmark. It does not measure power, accelerator
//! launch overhead, RTL performance, or physical memory traffic.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::native_similarity::PackedBipolar;

const CANDIDATE_COUNTS: [usize; 5] = [1, 16, 64, 256, 1024];
const STREAMING_CANDIDATE_COUNTS: [usize; 3] = [4096, 16384, 65536];
const SEED_BASE: u64 = 0x5EED_0000;
const QUERY_SEED: u64 = 0xABCD_16384;

fn seeded_vectors(count: usize, seed_base: u64) -> Vec<BinaryHV> {
    (0..count)
        .map(|index| BinaryHV::random(seed_base.wrapping_add(index as u64)))
        .collect()
}


/// Create a deterministic near-match with approximately 10% of its bits flipped.
/// Multiplication by an odd stride permutes the 2^14 bit positions, distributing
/// flips across the full vector rather than clustering them at the prefix.
fn distributed_near_match(query: &BinaryHV) -> BinaryHV {
    let mut bytes = query.0;
    for index in 0..(BinaryHV::DIM / 10) {
        let bit = (index * 8191) % BinaryHV::DIM;
        bytes[bit / 8] ^= 1u8 << (bit % 8);
    }
    BinaryHV(bytes)
}

/// Early-abandon exact top-1. Partial Hamming distance is a monotone lower bound
/// on the final distance. Once it reaches the best complete distance, this later
/// candidate cannot win under lowest-index tie-breaking. TRACK_BYTES is only for
/// untimed diagnostics; false lets the compiler remove the counter path.
fn early_abandon_hamming_argmin<const CHUNK_BYTES: usize, const TRACK_BYTES: bool>(
    query: &BinaryHV,
    candidates: &[BinaryHV],
) -> (usize, u32, usize) {
    assert!(!candidates.is_empty(), "top-1 search needs at least one candidate");
    assert!(CHUNK_BYTES > 0 && BinaryHV::BYTES % CHUNK_BYTES == 0);

    let mut best_index = 0usize;
    let mut best_distance = query.hamming_distance(&candidates[0]);
    let mut bytes_scanned = BinaryHV::BYTES;

    for (index, candidate) in candidates.iter().enumerate().skip(1) {
        let mut partial_distance = 0u32;
        let mut pruned = false;
        for start in (0..BinaryHV::BYTES).step_by(CHUNK_BYTES) {
            let end = start + CHUNK_BYTES;
            for byte_index in start..end {
                partial_distance += (query.0[byte_index] ^ candidate.0[byte_index]).count_ones();
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
            best_index = index;
            best_distance = partial_distance;
        }
    }

    (best_index, best_distance, if TRACK_BYTES { bytes_scanned } else { 0 })
}

/// Exact top-1 search seeded by a previously known candidate ID.
///
/// The hint initializes only the incumbent upper bound; all other candidates
/// are still evaluated fully or pruned by a monotone lower bound. For a
/// candidate with a lower ID than the current incumbent, equality cannot prune
/// because it could win the required lowest-index tie. For higher IDs, equality
/// is safe to prune because it cannot improve the winner.
fn hinted_early_abandon_hamming_argmin<
    const CHUNK_BYTES: usize,
    const TRACK_BYTES: bool,
>(
    query: &BinaryHV,
    candidates: &[BinaryHV],
    hint_index: usize,
) -> (usize, u32, usize) {
    assert!(!candidates.is_empty(), "top-1 search needs at least one candidate");
    assert!(hint_index < candidates.len(), "hint index must identify a candidate");
    assert!(CHUNK_BYTES > 0 && BinaryHV::BYTES % CHUNK_BYTES == 0);

    let mut best_index = hint_index;
    let mut best_distance = query.hamming_distance(&candidates[hint_index]);
    let mut bytes_scanned = BinaryHV::BYTES;

    for (index, candidate) in candidates.iter().enumerate() {
        if index == hint_index {
            continue;
        }

        let mut partial_distance = 0u32;
        let mut pruned = false;
        for start in (0..BinaryHV::BYTES).step_by(CHUNK_BYTES) {
            let end = start + CHUNK_BYTES;
            for byte_index in start..end {
                partial_distance += (query.0[byte_index] ^ candidate.0[byte_index]).count_ones();
            }
            if TRACK_BYTES {
                bytes_scanned += CHUNK_BYTES;
            }

            let cannot_win = partial_distance > best_distance
                || (partial_distance == best_distance && index > best_index);
            if cannot_win {
                pruned = true;
                break;
            }
        }

        if !pruned
            && (partial_distance < best_distance
                || (partial_distance == best_distance && index < best_index))
        {
            best_index = index;
            best_distance = partial_distance;
        }
    }

    (best_index, best_distance, if TRACK_BYTES { bytes_scanned } else { 0 })
}

fn hv_to_bipolar(hv: &BinaryHV) -> Vec<i8> {
    let mut bipolar = Vec::with_capacity(BinaryHV::DIM);
    for byte in hv.0 {
        for bit_idx in 0..8 {
            bipolar.push(if (byte >> bit_idx) & 1 == 1 { 1 } else { -1 });
        }
    }
    bipolar
}

fn pack_vector(hv: &BinaryHV) -> PackedBipolar {
    PackedBipolar::from_bipolar(&hv_to_bipolar(hv))
}

fn pack_vectors(vectors: &[BinaryHV]) -> Vec<PackedBipolar> {
    vectors.iter().map(pack_vector).collect()
}

fn pack_vector_direct(hv: &BinaryHV) -> PackedBipolar {
    PackedBipolar::from_binary_hv(hv)
}

fn pack_vectors_direct(vectors: &[BinaryHV]) -> Vec<PackedBipolar> {
    vectors.iter().map(pack_vector_direct).collect()
}

/// Single-pass score reduction. Strict greater-than preserves the lowest ID on ties.
fn fused_simd_argmax(query: &BinaryHV, candidates: &[BinaryHV]) -> (usize, f32) {
    assert!(!candidates.is_empty(), "top-1 search needs at least one candidate");
    let mut best = (0usize, query.similarity(&candidates[0]));
    for (index, candidate) in candidates.iter().enumerate().skip(1) {
        let score = query.similarity(candidate);
        if score > best.1 {
            best = (index, score);
        }
    }
    best
}

/// Deliberately materializes all scores before reducing them.
fn materialized_simd_argmax(query: &BinaryHV, candidates: &[BinaryHV]) -> (usize, f32) {
    assert!(!candidates.is_empty(), "top-1 search needs at least one candidate");
    let scores: Vec<f32> = candidates
        .iter()
        .map(|candidate| query.similarity(candidate))
        .collect();

    let mut best = (0usize, scores[0]);
    for (index, &score) in scores.iter().enumerate().skip(1) {
        if score > best.1 {
            best = (index, score);
        }
    }
    best
}

/// Exact integer-distance reduction also preserves the lowest ID on ties.
fn hamming_argmin(query: &BinaryHV, candidates: &[BinaryHV]) -> (usize, u32) {
    assert!(!candidates.is_empty(), "top-1 search needs at least one candidate");
    let mut best = (0usize, query.hamming_distance(&candidates[0]));
    for (index, candidate) in candidates.iter().enumerate().skip(1) {
        let distance = query.hamming_distance(candidate);
        if distance < best.1 {
            best = (index, distance);
        }
    }
    best
}

fn packed_argmax(query: &PackedBipolar, candidates: &[PackedBipolar]) -> (usize, f32) {
    assert!(!candidates.is_empty(), "top-1 search needs at least one candidate");
    let mut best = (0usize, query.xor_similarity(&candidates[0]));
    for (index, candidate) in candidates.iter().enumerate().skip(1) {
        let score = query.xor_similarity(candidate);
        if score > best.1 {
            best = (index, score);
        }
    }
    best
}

/// Fail before timing if a candidate implementation changes exact top-1 semantics.
fn verify_fixture(query: &BinaryHV, candidates: &[BinaryHV], expected_index: Option<usize>) {
    assert!(!candidates.is_empty());
    let simd = fused_simd_argmax(query, candidates);
    let materialized = materialized_simd_argmax(query, candidates);
    let hamming = hamming_argmin(query, candidates);
    let packed_query = pack_vector(query);
    let packed_candidates = pack_vectors(candidates);
    let packed = packed_argmax(&packed_query, &packed_candidates);
    let direct_query = pack_vector_direct(query);
    let direct_candidates = pack_vectors_direct(candidates);
    let direct = packed_argmax(&direct_query, &direct_candidates);

    assert_eq!(simd.0, materialized.0, "materialized-score winner differs");
    assert_eq!(simd.0, hamming.0, "similarity and Hamming winners differ");
    assert_eq!(simd.0, packed.0, "BinaryHV and packed winners differ");
    assert_eq!(simd.0, direct.0, "BinaryHV and direct-packed winners differ");
    assert!(
        (packed.1 - direct.1).abs() <= 1.0e-6,
        "legacy and direct-packed scores differ"
    );
    assert_eq!(
        simd.0,
        early_abandon_hamming_argmin::<16, false>(query, candidates).0,
        "16-byte early-abandon winner differs"
    );
    assert_eq!(
        simd.0,
        early_abandon_hamming_argmin::<64, false>(query, candidates).0,
        "64-byte early-abandon winner differs"
    );
    let hint_index = candidates.len() / 2;
    let hinted_16 =
        hinted_early_abandon_hamming_argmin::<16, false>(query, candidates, hint_index);
    let hinted_64 =
        hinted_early_abandon_hamming_argmin::<64, false>(query, candidates, hint_index);
    assert_eq!(simd.0, hinted_16.0, "16-byte hinted early-abandon winner differs");
    assert_eq!(simd.0, hinted_64.0, "64-byte hinted early-abandon winner differs");
    assert_eq!(hamming.1, hinted_16.1, "16-byte hinted distance differs");
    assert_eq!(hamming.1, hinted_64.1, "64-byte hinted distance differs");
    let hamming_score = 1.0 - (hamming.1 as f32 / BinaryHV::DIM as f32);
    assert!(
        (simd.1 - hamming_score).abs() <= 1.0e-6,
        "Hamming distance and canonical similarity score disagree"
    );
    assert!(
        (simd.1 - packed.1).abs() <= 1.0e-6,
        "packed score differs from canonical BinaryHV score"
    );
    if let Some(index) = expected_index {
        assert_eq!(simd.0, index, "fixture violated expected lowest-index winner");
    }
}

fn verify_correctness_contract() {
    for count in CANDIDATE_COUNTS {
        let query = BinaryHV::random(QUERY_SEED);
        let candidates = seeded_vectors(count, SEED_BASE);
        verify_fixture(&query, &candidates, None);
    }

    // Two identical best candidates must select the first candidate ID.
    let query = BinaryHV::random(0x71E_001);
    let tie_fixture = vec![
        BinaryHV::zero(),
        query,
        BinaryHV::ones(),
        query,
    ];
    verify_fixture(&query, &tie_fixture, Some(1));

    // A zero query and duplicate exact matches exercise tie behavior again.
    let zero_query = BinaryHV::zero();
    let zero_tie_fixture = vec![BinaryHV::ones(), zero_query, zero_query, BinaryHV::ones()];
    verify_fixture(&zero_query, &zero_tie_fixture, Some(1));
}

fn bench_streaming_exact_search(c: &mut Criterion) {
    use std::time::Duration;

    let query = BinaryHV::random(QUERY_SEED);
    let near_match = distributed_near_match(&query);
    let mut group = c.benchmark_group("hdc_streaming_argmax_16k");
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    for count in STREAMING_CANDIDATE_COUNTS {
        let mut candidates = seeded_vectors(count, SEED_BASE);
        let target_index = count / 2;
        let replaced = candidates[target_index];

        // Scenario 1: no planted near-match; expose the weak-pruning case.
        let full_random = hamming_argmin(&query, &candidates);
        assert_eq!(
            full_random.0,
            early_abandon_hamming_argmin::<16, false>(&query, &candidates).0
        );
        assert_eq!(
            full_random.0,
            early_abandon_hamming_argmin::<64, false>(&query, &candidates).0
        );

        let (_, _, random_bytes_16) =
            early_abandon_hamming_argmin::<16, true>(&query, &candidates);
        let (_, _, random_bytes_64) =
            early_abandon_hamming_argmin::<64, true>(&query, &candidates);
        eprintln!(
            "HDC-DSE random corpus candidates={count} logical_bytes={} scanned_16B={} scanned_64B={}",
            count * BinaryHV::BYTES,
            random_bytes_16,
            random_bytes_64
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("random_full_scan", count),
            &count,
            |bench, _| {
                bench.iter(|| black_box(hamming_argmin(
                    black_box(&query),
                    black_box(&candidates),
                )))
            },
        );
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("random_early_abandon_16B", count),
            &count,
            |bench, _| {
                bench.iter(|| black_box(early_abandon_hamming_argmin::<16, false>(
                    black_box(&query),
                    black_box(&candidates),
                )))
            },
        );
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("random_early_abandon_64B", count),
            &count,
            |bench, _| {
                bench.iter(|| black_box(early_abandon_hamming_argmin::<64, false>(
                    black_box(&query),
                    black_box(&candidates),
                )))
            },
        );

        // Scenario 2 models temporal locality: the prior query is stored at
        // target_index, and the current query is a deterministic ~10%-bit
        // perturbation of that prior query. The prior winner initializes the
        // bound, but all candidates remain subject to exact checking/pruning.
        let previous_query = query;
        let current_query = near_match;
        candidates[target_index] = previous_query;
        let correlated_full = hamming_argmin(&current_query, &candidates);
        assert_eq!(correlated_full.0, target_index);
        assert_eq!(
            correlated_full.0,
            early_abandon_hamming_argmin::<16, false>(&current_query, &candidates).0
        );
        assert_eq!(
            correlated_full.0,
            early_abandon_hamming_argmin::<64, false>(&current_query, &candidates).0
        );
        assert_eq!(
            correlated_full.0,
            hinted_early_abandon_hamming_argmin::<16, false>(
                &current_query,
                &candidates,
                target_index,
            )
            .0
        );
        assert_eq!(
            correlated_full.0,
            hinted_early_abandon_hamming_argmin::<64, false>(
                &current_query,
                &candidates,
                target_index,
            )
            .0
        );

        let (_, _, cold_bytes_16) =
            early_abandon_hamming_argmin::<16, true>(&current_query, &candidates);
        let (_, _, cold_bytes_64) =
            early_abandon_hamming_argmin::<64, true>(&current_query, &candidates);
        let (_, _, warm_bytes_16) = hinted_early_abandon_hamming_argmin::<16, true>(
            &current_query,
            &candidates,
            target_index,
        );
        let (_, _, warm_bytes_64) = hinted_early_abandon_hamming_argmin::<64, true>(
            &current_query,
            &candidates,
            target_index,
        );
        eprintln!(
            "HDC-DSE correlated-query candidates={count} logical_bytes={} prior_winner_index={target_index} cold_scanned_16B={} cold_scanned_64B={} hinted_scanned_16B={} hinted_scanned_64B={}",
            count * BinaryHV::BYTES,
            cold_bytes_16,
            cold_bytes_64,
            warm_bytes_16,
            warm_bytes_64
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("correlated_full_scan", count),
            &count,
            |bench, _| {
                bench.iter(|| black_box(hamming_argmin(
                    black_box(&current_query),
                    black_box(&candidates),
                )))
            },
        );
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("correlated_cold_early_abandon_16B", count),
            &count,
            |bench, _| {
                bench.iter(|| black_box(early_abandon_hamming_argmin::<16, false>(
                    black_box(&current_query),
                    black_box(&candidates),
                )))
            },
        );
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("correlated_hinted_early_abandon_16B", count),
            &count,
            |bench, _| {
                bench.iter(|| black_box(hinted_early_abandon_hamming_argmin::<16, false>(
                    black_box(&current_query),
                    black_box(&candidates),
                    target_index,
                )))
            },
        );
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("correlated_cold_early_abandon_64B", count),
            &count,
            |bench, _| {
                bench.iter(|| black_box(early_abandon_hamming_argmin::<64, false>(
                    black_box(&current_query),
                    black_box(&candidates),
                )))
            },
        );
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("correlated_hinted_early_abandon_64B", count),
            &count,
            |bench, _| {
                bench.iter(|| black_box(hinted_early_abandon_hamming_argmin::<64, false>(
                    black_box(&current_query),
                    black_box(&candidates),
                    target_index,
                )))
            },
        );
        candidates[target_index] = replaced;
    }
    group.finish();
}

fn bench_exact_search(c: &mut Criterion) {
    verify_correctness_contract();

    let mut group = c.benchmark_group("hdc_fused_argmax_16k");
    for count in CANDIDATE_COUNTS {
        let query = BinaryHV::random(QUERY_SEED);
        let candidates = seeded_vectors(count, SEED_BASE);
        let packed_query = pack_vector(&query);
        let packed_candidates = pack_vectors(&candidates);
        let direct_packed_query = pack_vector_direct(&query);
        let direct_packed_candidates = pack_vectors_direct(&candidates);
        let expected = fused_simd_argmax(&query, &candidates);
        assert_eq!(expected.0, packed_argmax(&packed_query, &packed_candidates).0);
        assert_eq!(
            expected.0,
            packed_argmax(&direct_packed_query, &direct_packed_candidates).0
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("simd_fused_top1", count),
            &count,
            |bench, _| {
                bench.iter(|| {
                    black_box(fused_simd_argmax(
                        black_box(&query),
                        black_box(&candidates),
                    ))
                })
            },
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("simd_materialized_scores_top1", count),
            &count,
            |bench, _| {
                bench.iter(|| {
                    black_box(materialized_simd_argmax(
                        black_box(&query),
                        black_box(&candidates),
                    ))
                })
            },
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("hamming_distance_argmin", count),
            &count,
            |bench, _| {
                bench.iter(|| {
                    black_box(hamming_argmin(
                        black_box(&query),
                        black_box(&candidates),
                    ))
                })
            },
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("hamming_early_abandon_16B", count),
            &count,
            |bench, _| {
                bench.iter(|| {
                    black_box(early_abandon_hamming_argmin::<16, false>(
                        black_box(&query),
                        black_box(&candidates),
                    ))
                })
            },
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("hamming_early_abandon_64B", count),
            &count,
            |bench, _| {
                bench.iter(|| {
                    black_box(early_abandon_hamming_argmin::<64, false>(
                        black_box(&query),
                        black_box(&candidates),
                    ))
                })
            },
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("packed_persisted_argmax", count),
            &count,
            |bench, _| {
                bench.iter(|| {
                    black_box(packed_argmax(
                        black_box(&packed_query),
                        black_box(&packed_candidates),
                    ))
                })
            },
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("direct_packed_persisted_argmax", count),
            &count,
            |bench, _| {
                bench.iter(|| {
                    black_box(packed_argmax(
                        black_box(&direct_packed_query),
                        black_box(&direct_packed_candidates),
                    ))
                })
            },
        );

        // Intentionally setup-inclusive: the packed representation is not free
        // when the caller does not persist it between searches.
        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("packed_prepare_plus_argmax", count),
            &count,
            |bench, _| {
                bench.iter(|| {
                    let packed_query = pack_vector(black_box(&query));
                    let packed_candidates = pack_vectors(black_box(&candidates));
                    black_box(packed_argmax(&packed_query, &packed_candidates))
                })
            },
        );

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(
            BenchmarkId::new("direct_packed_prepare_plus_argmax", count),
            &count,
            |bench, _| {
                bench.iter(|| {
                    let packed_query = pack_vector_direct(black_box(&query));
                    let packed_candidates = pack_vectors_direct(black_box(&candidates));
                    black_box(packed_argmax(&packed_query, &packed_candidates))
                })
            },
        );
    }
    group.finish();

    let query = BinaryHV::random(QUERY_SEED);
    let candidates = seeded_vectors(256, SEED_BASE);
    let mut prep_group = c.benchmark_group("hdc_packed_representation_prepare_16k");
    prep_group.throughput(Throughput::Elements(candidates.len() as u64));
    prep_group.bench_function("candidate_corpus_256", |bench| {
        bench.iter(|| black_box(pack_vectors(black_box(&candidates))))
    });
    prep_group.bench_function("single_query", |bench| {
        bench.iter(|| black_box(pack_vector(black_box(&query))))
    });
    prep_group.bench_function("candidate_corpus_direct_256", |bench| {
        bench.iter(|| black_box(pack_vectors_direct(black_box(&candidates))))
    });
    prep_group.bench_function("single_query_direct", |bench| {
        bench.iter(|| black_box(pack_vector_direct(black_box(&query))))
    });
    prep_group.finish();
}

criterion_group!(benches, bench_exact_search, bench_streaming_exact_search);
criterion_main!(benches);
