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
const OPEN_SET_CALIBRATION_COUNT: usize = 8;
const OPEN_SET_TEST_COUNT: usize = 8;
const NEAR_MISS_QUERY_COUNT: usize = 4;
const NEAR_MISS_NOISE_BPS: [usize; 4] = [1000, 2000, 3000, 4000];
const QUERY_NOISE_BPS: [usize; 5] = [0, 100, 200, 500, 1000];
const ROLE_SEED: u64 = 0xA11C_E000_0000;
const VALUE_SEED: u64 = 0xC0DE_0000_0000;

struct CompositionalWorkload {
    corpus: Vec<BinaryHV>,
    query_atoms: Vec<BinaryHV>,
    role_zero: BinaryHV,
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

    CompositionalWorkload {
        corpus,
        query_atoms,
        role_zero: roles[0],
    }
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
        .map(|(a, b)| (*a ^ *b).count_ones())
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

/// Flip an exact, deterministic number of query bits. The odd stride 8191 is
/// coprime with DIM, so the selected bit positions do not repeat for this sweep.
fn corrupt_query(query: &BinaryHV, noise_bps: usize, salt: usize) -> BinaryHV {
    assert!(noise_bps <= 10_000);
    let mut corrupted = *query;
    let flip_count = BinaryHV::DIM * noise_bps / 10_000;
    for i in 0..flip_count {
        let bit = (i * 8191 + salt * 127) % BinaryHV::DIM;
        corrupted.0[bit / 8] ^= 1u8 << (bit % 8);
    }
    corrupted
}

/// An open-set query has the same role-bound cue form as a stored record but
/// its value seed is outside the corpus seed range.
fn unknown_query(workload: &CompositionalWorkload, unknown_index: usize) -> BinaryHV {
    let seed_index = workload
        .corpus
        .len()
        .wrapping_add(100_000)
        .wrapping_add(unknown_index) as u64;
    let value = BinaryHV::random(
        VALUE_SEED.wrapping_add(seed_index.wrapping_mul(64)),
    );
    workload.role_zero.bind(&value)
}

/// Construct an absent-value cue deliberately close to a stored cue. Since
/// binding is XOR and self-inverse, recover the source value, flip a fixed
/// fraction of its bits, and bind it to the same role. This is a hard-negative
/// identity test, not a claim that bitwise proximity determines human meaning.
fn near_miss_unknown_query(
    workload: &CompositionalWorkload,
    source_record_id: usize,
    noise_bps: usize,
) -> BinaryHV {
    let source_value = workload.query_atoms[source_record_id].bind(&workload.role_zero);
    let unseen_value = corrupt_query(&source_value, noise_bps, source_record_id);
    workload.role_zero.bind(&unseen_value)
}

#[derive(Clone, Copy)]
struct AcceptanceThreshold {
    score: f32,
    calibration_false_reject_rate: f32,
    calibration_false_accept_rate: f32,
}

/// Choose a similarity threshold on calibration-only examples. Accept iff the
/// nearest score is >= threshold. Minimize false-reject + false-accept rate,
/// then prefer the more balanced error rate, then the higher threshold.
fn calibrate_acceptance_threshold(
    known_scores: &[f32],
    unknown_scores: &[f32],
) -> AcceptanceThreshold {
    assert!(!known_scores.is_empty());
    assert!(!unknown_scores.is_empty());

    let mut candidates = vec![0.0f32, 1.0f32];
    candidates.extend(known_scores.iter().copied());
    candidates.extend(unknown_scores.iter().copied());
    candidates.sort_by(f32::total_cmp);
    candidates.dedup_by(|a, b| a.total_cmp(b).is_eq());
    let adjacent: Vec<f32> = candidates
        .windows(2)
        .map(|pair| (pair[0] + pair[1]) * 0.5)
        .collect();
    candidates.extend(adjacent);

    let mut best = AcceptanceThreshold {
        score: 0.0,
        calibration_false_reject_rate: 1.0,
        calibration_false_accept_rate: 1.0,
    };
    let mut best_error = f32::INFINITY;
    let mut best_imbalance = f32::INFINITY;

    for threshold in candidates {
        let false_rejects = known_scores.iter().filter(|&&s| s < threshold).count();
        let false_accepts = unknown_scores.iter().filter(|&&s| s >= threshold).count();
        let fnr = false_rejects as f32 / known_scores.len() as f32;
        let fpr = false_accepts as f32 / unknown_scores.len() as f32;
        let error = fnr + fpr;
        let imbalance = (fnr - fpr).abs();

        if error < best_error
            || (error == best_error && imbalance < best_imbalance)
            || (error == best_error
                && imbalance == best_imbalance
                && threshold > best.score)
        {
            best_error = error;
            best_imbalance = imbalance;
            best = AcceptanceThreshold {
                score: threshold,
                calibration_false_reject_rate: fnr,
                calibration_false_accept_rate: fpr,
            };
        }
    }

    best
}

/// Calibrate on clean cues and unknown values, then evaluate only on distinct
/// held-out record IDs and unknown-value seeds. Noise is applied after the
/// clean threshold is frozen so the robustness curve cannot retune itself.
fn open_set_diagnostics(workload: &CompositionalWorkload, arity: usize) {
    let n = workload.corpus.len();
    let known_calibration: Vec<f32> = (0..OPEN_SET_CALIBRATION_COUNT)
        .map(|i| {
            let target_id = (i * 9973 + 17) % n;
            exact_top1(&workload.query_atoms[target_id], &workload.corpus).1
        })
        .collect();
    let unknown_calibration: Vec<f32> = (0..OPEN_SET_CALIBRATION_COUNT)
        .map(|i| exact_top1(&unknown_query(workload, i), &workload.corpus).1)
        .collect();
    let threshold = calibrate_acceptance_threshold(&known_calibration, &unknown_calibration);

    let known_test_ids: Vec<usize> = (0..OPEN_SET_TEST_COUNT)
        .map(|i| ((i + 64) * 9973 + 17) % n)
        .collect();
    let unknown_test_scores: Vec<f32> = (OPEN_SET_CALIBRATION_COUNT
        ..OPEN_SET_CALIBRATION_COUNT + OPEN_SET_TEST_COUNT)
        .map(|i| exact_top1(&unknown_query(workload, i), &workload.corpus).1)
        .collect();
    let false_accepts = unknown_test_scores
        .iter()
        .filter(|&&score| score >= threshold.score)
        .count();

    // These hard negatives encode absent values intentionally close to stored
    // values. They are evaluated against the same frozen threshold so the
    // diagnostic exposes false accepts missed by independent random negatives.
    for noise_bps in NEAR_MISS_NOISE_BPS {
        let near_miss_scores: Vec<f32> = (0..NEAR_MISS_QUERY_COUNT)
            .map(|i| {
                let source_id = ((i + 128) * 9973 + 17) % n;
                let query = near_miss_unknown_query(workload, source_id, noise_bps);
                exact_top1(&query, &workload.corpus).1
            })
            .collect();
        let near_miss_false_accepts = near_miss_scores
            .iter()
            .filter(|&&score| score >= threshold.score)
            .count();
        eprintln!(
            "open-set-near-miss arity={} n={} near_miss_noise_bps={} \
             threshold_similarity={:.6} false_accepts={}/{} false_accept_rate={:.3}",
            arity,
            n,
            noise_bps,
            threshold.score,
            near_miss_false_accepts,
            NEAR_MISS_QUERY_COUNT,
            near_miss_false_accepts as f32 / NEAR_MISS_QUERY_COUNT as f32,
        );
    }

    for noise_bps in QUERY_NOISE_BPS {
        let false_rejects = known_test_ids
            .iter()
            .map(|&target_id| corrupt_query(&workload.query_atoms[target_id], noise_bps, target_id))
            .map(|query| exact_top1(&query, &workload.corpus).1)
            .filter(|&score| score < threshold.score)
            .count();

        eprintln!(
            "open-set-acceptance arity={} n={} noise_bps={} threshold_similarity={:.6} \
             cal_fnr={:.3} cal_fpr={:.3} test_fnr={:.3} test_fpr={:.3} \
             test_known={}/{} test_unknown_false_accepts={}/{}",
            arity,
            n,
            noise_bps,
            threshold.score,
            threshold.calibration_false_reject_rate,
            threshold.calibration_false_accept_rate,
            false_rejects as f32 / OPEN_SET_TEST_COUNT as f32,
            false_accepts as f32 / OPEN_SET_TEST_COUNT as f32,
            OPEN_SET_TEST_COUNT - false_rejects,
            OPEN_SET_TEST_COUNT,
            false_accepts,
            OPEN_SET_TEST_COUNT,
        );
    }
}

struct RetrievalSummary {
    hit_at_one: usize,
    sampled: usize,
    median_target_rank: usize,
    minimum_target_margin: f32,
}

/// Evaluate a fixed, spread-out sample of query/target pairs. Rank is based on
/// the optimized exact similarity score and the same lowest-ID tie policy.
fn summarize_retrieval(workload: &CompositionalWorkload, noise_bps: usize) -> RetrievalSummary {
    let n = workload.corpus.len();
    let sampled = METRIC_QUERY_COUNT.min(n);
    let mut ranks = Vec::with_capacity(sampled);
    let mut hit_at_one = 0usize;
    let mut minimum_target_margin = f32::INFINITY;

    for sample_id in 0..sampled {
        // 9973 is coprime with the configured power-of-two corpus sizes; this
        // rotates the sample across the corpus instead of testing only its head.
        let target_id = (sample_id * 9973 + 17) % n;
        let clean_query = &workload.query_atoms[target_id];
        let query = corrupt_query(clean_query, noise_bps, target_id);
        let (winner_id, _) = exact_top1(&query, &workload.corpus);
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
            for noise_bps in QUERY_NOISE_BPS {
                let noisy = corrupt_query(query, noise_bps, target_id);
                let expected_flips = (BinaryHV::DIM * noise_bps / 10_000) as u32;
                assert_eq!(
                    scalar_hamming(query, &noisy),
                    expected_flips,
                    "query-noise fixture flipped an unexpected number of bits"
                );
                assert_eq!(
                    exact_top1(&noisy, &workload.corpus).0,
                    scalar_top1_id(&noisy, &workload.corpus),
                    "optimized and scalar exact top-1 differ for n={corpus_size}, target={target_id}, noise_bps={noise_bps}"
                );
            }
        }
    }

    // Open-set threshold sanity checks: a cleanly separated calibration set
    // must yield a threshold between the classes with zero calibration errors.
    let threshold = calibrate_acceptance_threshold(
        &[0.80, 0.82, 0.84, 0.86],
        &[0.50, 0.52, 0.54, 0.56],
    );
    assert!(threshold.score > 0.56 && threshold.score < 0.80);
    assert_eq!(threshold.calibration_false_reject_rate, 0.0);
    assert_eq!(threshold.calibration_false_accept_rate, 0.0);

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
            open_set_diagnostics(&workload, arity);
            for noise_bps in QUERY_NOISE_BPS {
                let summary = summarize_retrieval(&workload, noise_bps);
                eprintln!(
                    "compositional-retrieval arity={arity} n={corpus_size} noise_bps={noise_bps} \
                     semantic_hit_at_1={}/{} median_target_rank={} min_target_margin={:.6}",
                    summary.hit_at_one,
                    summary.sampled,
                    summary.median_target_rank,
                    summary.minimum_target_margin,
                );
            }

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
