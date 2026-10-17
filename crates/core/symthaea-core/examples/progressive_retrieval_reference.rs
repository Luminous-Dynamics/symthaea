// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Progressive-width HDC retrieval experiment.
//!
//! Compare a cheap, prefix-width shortlist with exhaustive retrieval at 16,384 dimensions.
//! Records encode eight categorical properties as bipolar role/filler bindings bundled
//! into one ContinuousHV. Queries reveal only the first 1, 2, 4 or 6 properties, creating
//! controlled ambiguity through shared facts rather than only independent random vectors.
//!
//! Run:
//!   cargo run --release --example progressive_retrieval_reference -p symthaea-core -- /tmp/progressive-retrieval.csv 12345
//!
//! This experiment measures candidate preservation and coordinate-evaluation counts. It
//! does NOT learn a threshold, claim an extreme-tail false-positive guarantee, or show
//! persistent-memory savings. All full-width records stay resident so shortlist and
//! exhaustive-reference scores use the same underlying records. The high-width top-K is
//! the oracle reference for candidate-set preservation, not an absolute semantic truth.
//!
//! Each row is one query × coarse width × shortlist K. Output preserves raw query IDs,
//! relevant-candidate recall, overlap with the exhaustive high-width top-K, reference
//! top-1 preservation and arithmetic-coordinate counts. Results must be evaluated on a
//! separate held-out corpus/query-seed split before any routing gate can be claimed.

use std::fmt::Write as _;
use symthaea_core::hdc::unified_hv::ContinuousHV;

const MAX_DIM: usize = 16_384;
const COARSE_DIMS: &[usize] = &[1_024, 2_048, 4_096, 8_192];
const SHORTLIST_SIZES: &[usize] = &[1, 4, 8, 16, 32];
const RECORDS: usize = 512;
const QUERIES: usize = 64;
const PROPERTIES: usize = 8;
const VALUES_PER_PROPERTY: usize = 32;
const QUERY_PROPERTY_COUNTS: &[usize] = &[1, 2, 4, 6];
const DEFAULT_CORPUS_SEED: u64 = 0xC0FF_EE77_2201_0001;

struct Record {
    values: [usize; PROPERTIES],
    vector: ContinuousHV,
}

fn next_u64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn item_seed(root: u64, property: usize, value: usize, domain: u64) -> u64 {
    root.wrapping_add((property as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add((value as u64).wrapping_mul(0xD1B5_4A32_D192_ED03))
        .wrapping_add(domain)
}

fn bipolar_continuous(dim: usize, seed: u64) -> ContinuousHV {
    let raw = ContinuousHV::random(dim, seed);
    ContinuousHV::from_values(
        raw.values
            .into_iter()
            .map(|value| if value >= 0.0 { 1.0_f32 } else { -1.0_f32 })
            .collect(),
    )
}

/// Ensure the current deterministic generator makes a same-seed vector prefix-stable.
fn assert_prefix_stability() {
    let seed = 0xA55A_1234_9876_FEDCu64;
    let short = bipolar_continuous(1_024, seed);
    let medium = bipolar_continuous(4_096, seed);
    let wide = bipolar_continuous(MAX_DIM, seed);
    assert_eq!(&short.values[..], &medium.values[..1_024]);
    assert_eq!(&medium.values[..], &wide.values[..4_096]);
}

fn similarity_prefix(a: &ContinuousHV, b: &ContinuousHV, dim: usize) -> f64 {
    assert!(dim <= a.dim() && dim <= b.dim());
    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    for (&x, &y) in a.values[..dim].iter().zip(&b.values[..dim]) {
        let x = x as f64;
        let y = y as f64;
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    if norm_a <= 0.0 || norm_b <= 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

fn build_codebook(corpus_seed: u64) -> (Vec<ContinuousHV>, Vec<Vec<ContinuousHV>>) {
    let roles = (0..PROPERTIES)
        .map(|property| {
            bipolar_continuous(
                MAX_DIM,
                item_seed(corpus_seed, property, 0, 0x524F_4C45_0000_0001),
            )
        })
        .collect::<Vec<_>>();
    let values = (0..PROPERTIES)
        .map(|property| {
            (0..VALUES_PER_PROPERTY)
                .map(|value| {
                    bipolar_continuous(
                        MAX_DIM,
                        item_seed(corpus_seed, property, value, 0x5641_4C55_4500_0002),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    (roles, values)
}

fn build_corpus(
    corpus_seed: u64,
    roles: &[ContinuousHV],
    values: &[Vec<ContinuousHV>],
) -> Vec<Record> {
    let mut state = corpus_seed ^ 0xA076_1D64_78BD_642F;
    let mut records = Vec::with_capacity(RECORDS);
    for _record_id in 0..RECORDS {
        let property_values = std::array::from_fn(|_| {
            (next_u64(&mut state) % VALUES_PER_PROPERTY as u64) as usize
        });
        let facts = (0..PROPERTIES)
            .map(|property| roles[property].bind(&values[property][property_values[property]]))
            .collect::<Vec<_>>();
        let fact_refs = facts.iter().collect::<Vec<_>>();
        records.push(Record {
            values: property_values,
            vector: ContinuousHV::bundle(&fact_refs),
        });
    }
    records
}

fn build_partial_query(
    record: &Record,
    query_properties: usize,
    roles: &[ContinuousHV],
    values: &[Vec<ContinuousHV>],
) -> ContinuousHV {
    let facts = (0..query_properties)
        .map(|property| {
            roles[property].bind(&values[property][record.values[property]])
        })
        .collect::<Vec<_>>();
    let refs = facts.iter().collect::<Vec<_>>();
    ContinuousHV::bundle(&refs)
}

fn rank_records(
    records: &[Record],
    query: &ContinuousHV,
    dim: usize,
) -> Vec<(usize, f64)> {
    let mut scores = records
        .iter()
        .enumerate()
        .map(|(record_id, record)| {
            (record_id, similarity_prefix(&record.vector, query, dim))
        })
        .collect::<Vec<_>>();
    scores.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    scores
}

fn main() {
    assert_prefix_stability();
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let output_path = args.first().cloned();
    let corpus_seed = args
        .get(1)
        .map(|value| value.parse::<u64>().expect("corpus seed must be an unsigned integer"))
        .unwrap_or(DEFAULT_CORPUS_SEED);
    let (roles, values) = build_codebook(corpus_seed);
    let records = build_corpus(corpus_seed, &roles, &values);
    let index_bytes = RECORDS * MAX_DIM * std::mem::size_of::<f32>();

    let mut csv = String::from(
        "corpus_seed,query_id,target_record_id,observed_properties,corpus_size,coarse_dimension,shortlist_k,reference_top1_id,coarse_top1_id,reference_top1_preserved,reference_topk_overlap_count,reference_topk_overlap_fraction,exact_matching_records,exact_matching_records_in_shortlist,exact_match_shortlist_recall,false_shortlist_candidates,target_in_shortlist,reference_top1_margin,coordinate_evaluations_staged,coordinate_evaluations_exhaustive,coordinate_reduction_ratio,full_index_vector_bytes\n",
    );

    for query_id in 0..QUERIES {
        // Multiplication by an odd number permutes the selected IDs modulo 2^9.
        let target_id = (query_id * 37 + 11) % RECORDS;
        let query_properties =
            QUERY_PROPERTY_COUNTS[query_id % QUERY_PROPERTY_COUNTS.len()];
        let target_record = &records[target_id];
        let query = build_partial_query(target_record, query_properties, &roles, &values);
        let reference = rank_records(&records, &query, MAX_DIM);
        let reference_top1_id = reference[0].0;
        let reference_margin = if reference.len() > 1 {
            reference[0].1 - reference[1].1
        } else {
            1.0
        };

        let relevant_ids = records
            .iter()
            .enumerate()
            .filter(|(_, candidate)| {
                (0..query_properties)
                    .all(|property| candidate.values[property] == target_record.values[property])
            })
            .map(|(record_id, _)| record_id)
            .collect::<Vec<_>>();
        assert!(
            relevant_ids.contains(&target_id),
            "the source record must satisfy its own partial query"
        );
        let exact_match_count = relevant_ids.len();

        for &dim in COARSE_DIMS {
            let coarse_ranked = rank_records(&records, &query, dim);
            for &shortlist_k in SHORTLIST_SIZES {
                let k = shortlist_k.min(RECORDS);
                let shortlist = coarse_ranked
                    .iter()
                    .take(k)
                    .map(|entry| entry.0)
                    .collect::<Vec<_>>();
                let reference_topk = reference
                    .iter()
                    .take(k)
                    .map(|entry| entry.0)
                    .collect::<Vec<_>>();
                let reference_overlap = reference_topk
                    .iter()
                    .filter(|record_id| shortlist.contains(record_id))
                    .count();
                let relevant_in_shortlist = shortlist
                    .iter()
                    .filter(|record_id| relevant_ids.contains(record_id))
                    .count();
                let target_in_shortlist = shortlist.contains(&target_id);
                let reference_top1_preserved = shortlist.contains(&reference_top1_id);
                let coordinate_staged = RECORDS * dim + k * MAX_DIM;
                let coordinate_exhaustive = RECORDS * MAX_DIM;
                let coordinate_reduction =
                    coordinate_exhaustive as f64 / coordinate_staged as f64;
                let overlap_fraction = reference_overlap as f64 / k as f64;
                let relevant_recall = relevant_in_shortlist as f64 / exact_match_count as f64;
                let false_shortlist_candidates = k.saturating_sub(relevant_in_shortlist);

                writeln!(
                    csv,
                    "{corpus_seed},{query_id},{target_id},{query_properties},{RECORDS},{dim},{k},{reference_top1_id},{},{},{reference_overlap},{overlap_fraction:.8},{exact_match_count},{relevant_in_shortlist},{relevant_recall:.8},{false_shortlist_candidates},{target_in_shortlist},{reference_margin:.8},{coordinate_staged},{coordinate_exhaustive},{coordinate_reduction:.8},{index_bytes}",
                    coarse_ranked[0].0,
                    reference_top1_preserved,
                )
                .unwrap();
            }
        }
    }

    match output_path {
        Some(path) => {
            std::fs::write(&path, csv).unwrap_or_else(|error| {
                panic!("failed to write progressive retrieval CSV to {path}: {error}");
            });
            eprintln!("Progressive retrieval rows written to {path}");
        }
        None => print!("{csv}"),
    }
}
