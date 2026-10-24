// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Representation-level retrieval trade-off: bundled prototypes vs component facts.
//!
//! This benchmark deliberately changes what is stored, not just how the same
//! corpus is traversed. A single bundled vector is compact but can erase a
//! constituent cue. Retaining role-bound facts costs K vectors per record but
//! permits exact-cue lookup and a fuzzy component-scan fallback.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use symthaea_core::hdc::binary_hv::BinaryHV;

const ARITIES: [usize; 3] = [3, 7, 15];
const CORPUS_SIZES: [usize; 3] = [256, 1024, 4096];
const NOISE_BPS: [usize; 4] = [0, 100, 500, 1000];
const TIMED_NOISE_BPS: [usize; 3] = [0, 500, 1000];
const METRIC_QUERIES: usize = 8;
const ROLE_SEED: u64 = 0xA11C_E000_0000;
const VALUE_SEED: u64 = 0xC0DE_1000_0000;
const VECTOR_BYTES: usize = BinaryHV::BYTES;

struct Workload {
    bundles: Vec<BinaryHV>,
    /// Flattened record-major storage: fact index / arity gives its record ID.
    facts: Vec<BinaryHV>,
    query_atoms: Vec<BinaryHV>,
    arity: usize,
}

fn build_workload(corpus_size: usize, arity: usize) -> Workload {
    assert!(arity > 0 && arity % 2 == 1);
    let roles: Vec<BinaryHV> = (0..arity)
        .map(|slot| BinaryHV::random(ROLE_SEED.wrapping_add(slot as u64 * 0x10001)))
        .collect();

    let mut bundles = Vec::with_capacity(corpus_size);
    let mut facts = Vec::with_capacity(corpus_size * arity);
    let mut query_atoms = Vec::with_capacity(corpus_size);

    for record_id in 0..corpus_size {
        let mut record_facts = Vec::with_capacity(arity);
        for slot in 0..arity {
            let seed = VALUE_SEED
                .wrapping_add((record_id as u64).wrapping_mul(64))
                .wrapping_add(slot as u64);
            record_facts.push(roles[slot].bind(&BinaryHV::random(seed)));
        }
        query_atoms.push(record_facts[0]);
        bundles.push(BinaryHV::bundle(&record_facts));
        facts.extend(record_facts);
    }

    Workload {
        bundles,
        facts,
        query_atoms,
        arity,
    }
}

fn corrupt_query(query: &BinaryHV, noise_bps: usize, salt: usize) -> BinaryHV {
    assert!(noise_bps <= 10_000);
    let mut out = *query;
    let flip_count = BinaryHV::DIM * noise_bps / 10_000;
    for i in 0..flip_count {
        let bit = (i * 8191 + salt * 127) % BinaryHV::DIM;
        out.0[bit / 8] ^= 1u8 << (bit % 8);
    }
    out
}

fn bundled_top1(query: &BinaryHV, bundles: &[BinaryHV]) -> usize {
    assert!(!bundles.is_empty());
    let mut best_id = 0usize;
    let mut best_score = query.similarity(&bundles[0]);
    for (record_id, bundle) in bundles.iter().enumerate().skip(1) {
        let score = query.similarity(bundle);
        if score > best_score {
            best_score = score;
            best_id = record_id;
        }
    }
    best_id
}

/// Rank records by the highest similarity to any retained role-bound fact.
/// Strict update plus record-major traversal makes ties prefer lower record ID.
fn component_scan_top1(query: &BinaryHV, facts: &[BinaryHV], arity: usize) -> usize {
    assert!(arity > 0 && facts.len() >= arity);
    let record_count = facts.len() / arity;
    let mut best_record = 0usize;
    let mut best_score = f32::NEG_INFINITY;

    for record_id in 0..record_count {
        let start = record_id * arity;
        let mut record_score = query.similarity(&facts[start]);
        for fact in &facts[start + 1..start + arity] {
            let score = query.similarity(fact);
            if score > record_score {
                record_score = score;
            }
        }
        if record_score > best_score {
            best_score = record_score;
            best_record = record_id;
        }
    }
    best_record
}

/// Independent Hamming-distance oracle for the bundled representation.
fn scalar_hamming(left: &BinaryHV, right: &BinaryHV) -> u32 {
    left.0.iter().zip(right.0.iter())
        .map(|(a, b)| (*a ^ *b).count_ones())
        .sum()
}

fn scalar_bundled_top1(query: &BinaryHV, bundles: &[BinaryHV]) -> usize {
    let mut best_id = 0usize;
    let mut best_distance = scalar_hamming(query, &bundles[0]);
    for (record_id, bundle) in bundles.iter().enumerate().skip(1) {
        let distance = scalar_hamming(query, bundle);
        if distance < best_distance {
            best_distance = distance;
            best_id = record_id;
        }
    }
    best_id
}

fn scalar_component_top1(query: &BinaryHV, facts: &[BinaryHV], arity: usize) -> usize {
    let record_count = facts.len() / arity;
    let mut best_record = 0usize;
    let mut best_record_distance = u32::MAX;
    for record_id in 0..record_count {
        let start = record_id * arity;
        let mut record_distance = scalar_hamming(query, &facts[start]);
        for fact in &facts[start + 1..start + arity] {
            record_distance = record_distance.min(scalar_hamming(query, fact));
        }
        if record_distance < best_record_distance {
            best_record_distance = record_distance;
            best_record = record_id;
        }
    }
    best_record
}

type ExactFactIndex = HashMap<u64, Vec<usize>>;

fn fingerprint(vector: &BinaryHV) -> u64 {
    let mut hasher = DefaultHasher::new();
    vector.hash(&mut hasher);
    hasher.finish()
}

/// Build a fingerprint-to-flat-fact-index map. The fingerprint only routes to
/// candidate facts; every hit is verified against all 2,048 bytes before use.
fn build_exact_fact_index(facts: &[BinaryHV]) -> ExactFactIndex {
    let mut index: ExactFactIndex = HashMap::with_capacity(facts.len());
    for (flat_id, fact) in facts.iter().enumerate() {
        index.entry(fingerprint(fact)).or_default().push(flat_id);
    }
    index
}

fn resolve_exact_candidates(
    query: &BinaryHV,
    candidates: &[usize],
    facts: &[BinaryHV],
    arity: usize,
) -> Option<usize> {
    if arity == 0 {
        return None;
    }
    candidates.iter()
        .copied()
        .filter(|&flat_id| facts.get(flat_id).is_some_and(|fact| fact == query))
        .map(|flat_id| flat_id / arity)
        .min()
}

fn exact_fact_lookup(
    query: &BinaryHV,
    facts: &[BinaryHV],
    index: &ExactFactIndex,
    arity: usize,
) -> Option<usize> {
    let candidates = index.get(&fingerprint(query))?;
    resolve_exact_candidates(query, candidates, facts, arity)
}

fn hybrid_top1(
    query: &BinaryHV,
    workload: &Workload,
    index: &ExactFactIndex,
) -> (usize, bool) {
    match exact_fact_lookup(query, &workload.facts, index, workload.arity) {
        Some(record_id) => (record_id, true),
        None => (
            component_scan_top1(query, &workload.facts, workload.arity),
            false,
        ),
    }
}

fn correctness_contract() {
    for n in [17usize, 65usize] {
        let workload = build_workload(n, 3);
        let index = build_exact_fact_index(&workload.facts);
        for target_id in [0, n / 2, n - 1] {
            let query = workload.query_atoms[target_id];
            for noise in NOISE_BPS {
                let noisy = corrupt_query(&query, noise, target_id);
                assert_eq!(
                    bundled_top1(&noisy, &workload.bundles),
                    scalar_bundled_top1(&noisy, &workload.bundles),
                    "bundled exact top-1 oracle mismatch: n={n}, target={target_id}, noise={noise}"
                );
                assert_eq!(
                    component_scan_top1(&noisy, &workload.facts, workload.arity),
                    scalar_component_top1(&noisy, &workload.facts, workload.arity),
                    "component exact top-1 oracle mismatch: n={n}, target={target_id}, noise={noise}"
                );
                assert_eq!(
                    hybrid_top1(&noisy, &workload, &index).0,
                    component_scan_top1(&noisy, &workload.facts, workload.arity),
                    "hybrid and component-scan retrieval differ: n={n}, target={target_id}, noise={noise}"
                );
                if noise == 0 {
                    assert_eq!(exact_fact_lookup(&query, &workload.facts, &index, workload.arity), Some(target_id));
                    assert_eq!(hybrid_top1(&query, &workload, &index), (target_id, true));
                } else {
                    assert_eq!(hybrid_top1(&noisy, &workload, &index).1, false);
                }
            }
        }
    }

    // Force two unrelated vectors into one artificial fingerprint bucket.
    // Exact equality verification must reject a collision rather than returning
    // the first bucket entry as a false identity match.
    let facts = vec![BinaryHV::random(11), BinaryHV::random(12)];
    let query = BinaryHV::random(13);
    assert_eq!(resolve_exact_candidates(&query, &[0, 1], &facts, 1), None);
    assert_eq!(resolve_exact_candidates(&facts[1], &[0, 1], &facts, 1), Some(1));

    // Deliberate duplicate facts prefer the lowest record ID.
    let duplicate = BinaryHV::random(14);
    let facts = vec![duplicate, BinaryHV::random(15), duplicate, BinaryHV::random(16)];
    assert_eq!(resolve_exact_candidates(&duplicate, &[0, 2], &facts, 2), Some(0));
}

fn emit_quality_diagnostics(workload: &Workload, index: &ExactFactIndex) {
    let n = workload.bundles.len();
    for noise in NOISE_BPS {
        let mut bundled_hits = 0usize;
        let mut component_hits = 0usize;
        let mut hybrid_hits = 0usize;
        let mut exact_index_hits = 0usize;
        for sample in 0..METRIC_QUERIES.min(n) {
            let target_id = (sample * 9973 + 17) % n;
            let query = corrupt_query(&workload.query_atoms[target_id], noise, target_id);
            bundled_hits += usize::from(bundled_top1(&query, &workload.bundles) == target_id);
            component_hits += usize::from(component_scan_top1(&query, &workload.facts, workload.arity) == target_id);
            let (hybrid_id, used_exact_index) = hybrid_top1(&query, workload, index);
            hybrid_hits += usize::from(hybrid_id == target_id);
            exact_index_hits += usize::from(used_exact_index);
        }
        eprintln!(
            "representation-quality n={} arity={} noise_bps={} \
             bundled_hit_at_1={}/{} component_hit_at_1={}/{} hybrid_hit_at_1={}/{} exact_index_coverage={}/{}",
            n, workload.arity, noise,
            bundled_hits, METRIC_QUERIES.min(n),
            component_hits, METRIC_QUERIES.min(n),
            hybrid_hits, METRIC_QUERIES.min(n),
            exact_index_hits, METRIC_QUERIES.min(n),
        );
    }
}

fn bench_representation_tradeoff(c: &mut Criterion) {
    correctness_contract();

    let mut group = c.benchmark_group("hdc_representation_query_16k");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(150));
    group.measurement_time(Duration::from_millis(300));

    for arity in ARITIES {
        for corpus_size in CORPUS_SIZES {
            let workload = build_workload(corpus_size, arity);
            let index = build_exact_fact_index(&workload.facts);
            emit_quality_diagnostics(&workload, &index);

            let clean = workload.query_atoms[corpus_size / 2];
            for noise in TIMED_NOISE_BPS {
                let query = corrupt_query(&clean, noise, corpus_size / 2);
                for (strategy, _) in [
                    ("bundled_prototype", 0u8),
                    ("component_scan", 1u8),
                    ("hybrid_exact_then_fuzzy", 2u8),
                ] {
                    group.throughput(Throughput::Elements(corpus_size as u64));
                    group.bench_with_input(
                        BenchmarkId::new(
                            format!("k{arity}_{strategy}_noise{noise}"),
                            corpus_size,
                        ),
                        &(query, &workload, &index),
                        |bencher, (query, workload, index)| {
                            bencher.iter(|| {
                                let result = match strategy {
                                    "bundled_prototype" => {
                                        (bundled_top1(black_box(query), &workload.bundles), false)
                                    }
                                    "component_scan" => {
                                        (component_scan_top1(black_box(query), &workload.facts, workload.arity), false)
                                    }
                                    "hybrid_exact_then_fuzzy" => hybrid_top1(black_box(query), workload, index),
                                    _ => unreachable!(),
                                };
                                black_box(result)
                            })
                        },
                    );
                }
            }
        }
    }
    group.finish();

    let mut index_group = c.benchmark_group("hdc_exact_fact_index_build_16k");
    index_group.sample_size(10);
    index_group.warm_up_time(Duration::from_millis(100));
    index_group.measurement_time(Duration::from_millis(250));
    for arity in ARITIES {
        for corpus_size in CORPUS_SIZES {
            let workload = build_workload(corpus_size, arity);
            index_group.throughput(Throughput::Elements((corpus_size * arity) as u64));
            index_group.bench_with_input(
                BenchmarkId::new(format!("k{arity}_n{corpus_size}"), corpus_size),
                &workload.facts,
                |bencher, facts| bencher.iter(|| black_box(build_exact_fact_index(black_box(facts)))),
            );
        }
    }
    index_group.finish();

    let mut build_group = c.benchmark_group("hdc_representation_build_16k");
    build_group.sample_size(10);
    build_group.warm_up_time(Duration::from_millis(100));
    build_group.measurement_time(Duration::from_millis(250));
    for arity in ARITIES {
        for corpus_size in CORPUS_SIZES {
            build_group.throughput(Throughput::Elements(corpus_size as u64));
            build_group.bench_with_input(
                BenchmarkId::new(format!("k{arity}_n{corpus_size}"), corpus_size),
                &(corpus_size, arity),
                |bencher, &(n, k)| bencher.iter(|| black_box(build_workload(black_box(n), black_box(k)))),
            );
        }
    }
    build_group.finish();
}

criterion_group!(benches, bench_representation_tradeoff);
criterion_main!(benches);
