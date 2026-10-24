// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compositional HDC retrieval workload: role-bound facts bundled into records.
//!
//! Unlike a uniformly random corpus, each record is a majority-bundle of
//! role-bound atoms. The query is one atom known to have contributed to a
//! target record. This separates exact nearest-neighbour correctness from
//! semantic retrieval quality: an exhaustive scan can be implemented exactly
//! while an overloaded representation may still rank the wrong record first.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use symthaea_core::hdc::binary_hv::BinaryHV;

const BUNDLE_ARITIES: [usize; 3] = [3, 7, 15];
const CORPUS_SIZES: [usize; 4] = [256, 1024, 4096, 16384];
const BUILD_BENCH_SIZES: [usize; 3] = [256, 1024, 4096];
const METRIC_QUERY_COUNT: usize = 8;
const ROLE_SEED: u64 = 0xA11C_E000_0000;
const VALUE_SEED: u64 = 0xC0DE_0000_0000;

struct CompositionalWorkload {
    corpus: Vec<BinaryHV>,
    query_atoms: Vec<BinaryHV>,
}

/// Construct each record from distinct role/value bindings, then bundle those
/// facts into one BinaryHV. The first role-bound fact is retained as a query
/// cue for that record. Every benchmark workload is deterministic.
fn build_workload(corpus_size: usize, arity: usize) -> CompositionalWorkload {
    assert!(arity > 0 && arity % 2 == 1, "majority bundle arity must be odd");
    let roles: Vec<BinaryHV> = (0..arity)
        .map(|slot| BinaryHV::random(ROLE_SEED.wrapping_add(slot as u64 * 0x10001)))
        .collect();

    let mut corpus = Vec::with_capacity(corpus_size);
    let mut query_atoms = Vec::with_capacity(corpus_size);
    for candidate_id in 0..corpus_size {
        let mut facts = Vec::with_capacity(arity);
        for slot in 0..arity {
            let value_seed = VALUE_SEED
                .wrapping_add((candidate_id as u64).wrapping_mul(64))
                .wrapping_add(slot as u64);
            let value = BinaryHV::random(value_seed);
            facts.push(roles[slot].bind(&value));
        }
        query_atoms.push(facts[0]);
        corpus.push(BinaryHV::bundle(&facts));
    }

    CompositionalWorkload { corpus, query_atoms }
}

/// Canonical optimized exact top-1; strict score improvement preserves the
/// lowest candidate ID on ties because the corpus is traversed in ID order.
fn exact_top1(query: &BinaryHV, corpus: &[BinaryHV]) -> (usize, f32) {
    assert!(!corpus.is_empty());
    let mut best = (0usize, query.similarity(&corpus[0]));
    for (candidate_id, candidate) in corpus.iter().enumerate().skip(1) {
        let score = query.similarity(candidate);
        if score > best.1 {
            best = (candidate_id, score);
        }
    }
    best
}

/// Independent byte-popcount oracle. It intentionally does not call the
/// optimized BinaryHV similarity path.
fn scalar_hamming(left: &BinaryHV, right: &BinaryHV) -> u32 {
    left.0
        .iter()
        .zip(right.0.iter())
        .map(|(a, b)| (a ^ b).count_ones())
        .sum()
}

fn scalar_top1_id(query: &BinaryHV, corpus: &[BinaryHV]) -> usize {
    let mut best_id = 0usize;
    let mut best_distance = scalar_hamming(query, &corpus[0]);
    for (candidate_id, candidate) in corpus.iter().enumerate().skip(1) {
        let distance = scalar_hamming(query, candidate);
        if distance < best_distance {
            best_id = candidate_id;
            best_distance = distance;
        }
    }
    best_id
}

struct RetrievalSummary {
    hit_at_one: usize,
    sampled: usize,
    median_target_rank: usize,
    minimum_target_margin: f32,
}

/// Evaluate a fixed, spread-out sample of query/target pairs. Rank is based on
/// the optimized exact similarity score and the same lowest-ID tie policy.
fn summarize_retrieval(workload: &CompositionalWorkload) -> RetrievalSummary {
    let n = workload.corpus.len();
    let sampled = METRIC_QUERY_COUNT.min(n);
    let mut ranks = Vec::with_capacity(sampled);
    let mut hit_at_one = 0usize;
    let mut minimum_target_margin = f32::INFINITY;

    for sample_id in 0..sampled {
        // 9973 is coprime with the configured power-of-two corpus sizes; this
        // rotates the sample across the corpus instead of testing only its head.
        let target_id = (sample_id * 9973 + 17) % n;
        let query = &workload.query_atoms[target_id];
        let (winner_id, _) = exact_top1(query, &workload.corpus);
        if winner_id == target_id {
            hit_at_one += 1;
        }

        let target_score = query.similarity(&workload.corpus[target_id]);
        let mut rank = 1usize;
        let mut strongest_impostor = f32::NEG_INFINITY;
        for (candidate_id, candidate) in workload.corpus.iter().enumerate() {
            if candidate_id == target_id {
                continue;
            }
            let score = query.similarity(candidate);
            if score > strongest_impostor {
                strongest_impostor = score;
            }
            if score > target_score || (score == target_score && candidate_id < target_id) {
                rank += 1;
            }
        }
        ranks.push(rank);
        minimum_target_margin = minimum_target_margin.min(target_score - strongest_impostor);
    }

    ranks.sort_unstable();
    RetrievalSummary {
        hit_at_one,
        sampled,
        median_target_rank: ranks[ranks.len() / 2],
        minimum_target_margin,
    }
}

fn correctness_contract() {
    // Check the optimized score ordering against an independently written
    // byte-popcount oracle on small non-power-of-two and ordinary corpora.
    for corpus_size in [17usize, 257usize] {
        let workload = build_workload(corpus_size, 3);
        for target_id in [0usize, corpus_size / 2, corpus_size - 1] {
            let query = &workload.query_atoms[target_id];
            assert_eq!(
                exact_top1(query, &workload.corpus).0,
                scalar_top1_id(query, &workload.corpus),
                "optimized and scalar exact top-1 differ for n={corpus_size}, target={target_id}"
            );
        }
    }

    // Explicit lowest-ID tie test, including repeated exact matches.
    let query = BinaryHV::random(0x71E_001);
    let corpus = vec![
        BinaryHV::ones(),
        query,
        query,
        BinaryHV::zero(),
        query,
    ];
    assert_eq!(exact_top1(&query, &corpus).0, 1);
    assert_eq!(scalar_top1_id(&query, &corpus), 1);
}

fn bench_compositional_retrieval(c: &mut Criterion) {
    correctness_contract();

    let mut group = c.benchmark_group("hdc_compositional_exact_retrieval_16k");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_millis(750));

    for arity in BUNDLE_ARITIES {
        for corpus_size in CORPUS_SIZES {
            let workload = build_workload(corpus_size, arity);
            let summary = summarize_retrieval(&workload);
            eprintln!(
                "compositional-retrieval arity={arity} n={corpus_size} \
                 semantic_hit_at_1={}/{} median_target_rank={} min_target_margin={:.6}",
                summary.hit_at_one,
                summary.sampled,
                summary.median_target_rank,
                summary.minimum_target_margin,
            );

            let query = workload.query_atoms[corpus_size / 2];
            group.throughput(Throughput::Elements(corpus_size as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("bundle_arity_{arity}"), corpus_size),
                &workload,
                |bencher, input| {
                    bencher.iter(|| {
                        black_box(exact_top1(
                            black_box(&query),
                            black_box(&input.corpus),
                        ))
                    })
                },
            );
        }
    }
    group.finish();

    // Build/encoding cost is a distinct phase from querying a resident corpus.
    // Keep this group smaller so the benchmark remains practical on laptops.
    let mut build_group = c.benchmark_group("hdc_compositional_record_build_16k");
    build_group.sample_size(10);
    build_group.warm_up_time(Duration::from_millis(200));
    build_group.measurement_time(Duration::from_millis(500));
    for arity in BUNDLE_ARITIES {
        for corpus_size in BUILD_BENCH_SIZES {
            build_group.throughput(Throughput::Elements(corpus_size as u64));
            build_group.bench_with_input(
                BenchmarkId::new(format!("bundle_arity_{arity}"), corpus_size),
                &(corpus_size, arity),
                |bencher, &(n, k)| {
                    bencher.iter(|| black_box(build_workload(black_box(n), black_box(k))))
                },
            );
        }
    }
    build_group.finish();
}

criterion_group!(benches, bench_compositional_retrieval);
criterion_main!(benches);
