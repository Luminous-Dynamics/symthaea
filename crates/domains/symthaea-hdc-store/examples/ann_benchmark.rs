// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Source-bound, deterministic benchmark for exact vs LSH HDC retrieval.
//!
//! Run with:
//! cargo run -p symthaea-hdc-store --example ann_benchmark --release -- \
//!   --source-sha=<40-hex-commit> --size=10000 --queries=200 --rounds=5 --top-k=10

use std::collections::HashSet;
use std::env;
use std::error::Error;
use std::io;
use std::process::Command;
use std::time::Instant;

use symthaea_core::hdc::BinaryHV;
use symthaea_hdc_store::{ApproximateSearchOptions, HdcStore, StoreConfig};
use tempfile::tempdir;

const PROTOCOL: &str = "BROCA-ANN-RETRIEVAL-V1";
const DEFAULT_SIZE: usize = 10_000;
const DEFAULT_QUERIES: usize = 200;
const DEFAULT_ROUNDS: usize = 5;
const DEFAULT_TOP_K: usize = 10;
const DEFAULT_NOISE_BITS: usize = 128;
const DEFAULT_CLUSTERS: usize = 64;
const DEFAULT_CLUSTER_NOISE_BITS: usize = 1_024;
const DEFAULT_SEED: u64 = 42;

#[derive(Debug)]
struct Config {
    source_sha: String,
    size: usize,
    clusters: usize,
    cluster_noise_bits: usize,
    queries: usize,
    rounds: usize,
    top_k: usize,
    noise_bits: usize,
    seed: u64,
    candidate_multiplier: usize,
    fallback_on_empty: bool,
    lsh_bands: u32,
    lsh_rows: u32,
}

#[derive(Clone, Copy)]
struct Query {
    vector: BinaryHV,
    kind: &'static str,
}

struct Sample {
    query_index: usize,
    round: usize,
    kind: &'static str,
    exact_ns: u128,
    approximate_ns: u128,
    recall_at_k: f64,
    examined: usize,
    total_live: usize,
    candidate_fraction: f64,
    approximate_exact: bool,
    fell_back_to_exact: bool,
    exact_ids: Vec<u64>,
    approximate_ids: Vec<u64>,
}

struct Quality {
    kind: &'static str,
    recall_at_k: f64,
    examined: usize,
    candidate_fraction: f64,
    approximate_exact: bool,
    fell_back_to_exact: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let config = parse_args().map_err(|message| {
        io::Error::new(io::ErrorKind::InvalidInput, message)
    })?;
    verify_source_binding(&config.source_sha)?;
    let dir = tempdir()?;
    let path = dir.path().join("ann-benchmark.hdc");

    let mut store_config = StoreConfig::default();
    store_config.initial_capacity = u64::try_from(config.size)?;
    store_config.lsh_bands = config.lsh_bands;
    store_config.lsh_rows = config.lsh_rows;
    let mut store = HdcStore::create(&path, store_config)?;

    for id in 0..config.size {
        store.append(id as u64, &corpus_vector(&config, id))?;
    }

    let queries = build_queries(&config, &store);
    let options = ApproximateSearchOptions {
        candidate_multiplier: config.candidate_multiplier,
        fallback_on_empty: config.fallback_on_empty,
    };

    // Warm both paths before collecting samples. The measured intervals contain
    // only the retrieval calls, not query generation, ID comparison or CSV output.
    for query in &queries {
        let _ = store.scan_similar(&query.vector, config.top_k);
        let _ = store.scan_similar_approx(&query.vector, config.top_k, options);
    }

    let mut samples = Vec::with_capacity(config.queries.saturating_mul(config.rounds));
    let mut first_round_quality = Vec::with_capacity(config.queries);
    for round in 0..config.rounds {
        for (query_index, query) in queries.iter().enumerate() {
            // Alternate call order to reduce systematic cache/order bias.
            let (exact, exact_ns, approximate, approximate_ns) =
                if (round + query_index) % 2 == 0 {
                    let (result, elapsed) = measure_exact(&store, &query.vector, config.top_k);
                    let (ann, ann_elapsed) =
                        measure_approximate(&store, &query.vector, config.top_k, options);
                    (result, elapsed, ann, ann_elapsed)
                } else {
                    let (ann, ann_elapsed) =
                        measure_approximate(&store, &query.vector, config.top_k, options);
                    let (result, elapsed) = measure_exact(&store, &query.vector, config.top_k);
                    (result, elapsed, ann, ann_elapsed)
                };

            let exact_ids: Vec<u64> = exact.iter().map(|(id, _)| *id).collect();
            let approximate_ids: Vec<u64> =
                approximate.neighbors.iter().map(|(id, _)| *id).collect();
            let recall_at_k = recall_at_k(&exact_ids, &approximate_ids);
            let total_live = approximate.total_live;
            let candidate_fraction = fraction(approximate.examined, total_live);

            if round == 0 {
                first_round_quality.push(Quality {
                    kind: query.kind,
                    recall_at_k,
                    examined: approximate.examined,
                    candidate_fraction,
                    approximate_exact: approximate.exact,
                    fell_back_to_exact: approximate.fell_back_to_exact,
                });
            }

            samples.push(Sample {
                query_index,
                round,
                kind: query.kind,
                exact_ns,
                approximate_ns,
                recall_at_k,
                examined: approximate.examined,
                total_live,
                candidate_fraction,
                approximate_exact: approximate.exact,
                fell_back_to_exact: approximate.fell_back_to_exact,
                exact_ids,
                approximate_ids,
            });
        }
    }

    emit_report(&config, &samples, &first_round_quality);
    Ok(())
}

fn verify_source_binding(expected_sha: &str) -> Result<(), Box<dyn Error>> {
    let head = Command::new("git").args(["rev-parse", "HEAD"]).output()?;
    if !head.status.success() {
        return Err(io::Error::other("git rev-parse HEAD failed").into());
    }
    let actual_sha = String::from_utf8(head.stdout)?.trim().to_owned();
    if actual_sha != expected_sha {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("source SHA mismatch: argument {expected_sha}, checked-out HEAD {actual_sha}"),
        )
        .into());
    }

    let status = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()?;
    if !status.status.success() {
        return Err(io::Error::other("git status failed").into());
    }
    let dirty = String::from_utf8(status.stdout)?;
    if !dirty.trim().is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "tracked working tree is dirty; commit or revert tracked changes before benchmarking",
        )
        .into());
    }
    Ok(())
}

fn parse_args() -> Result<Config, String> {
    let allowed = [
        "source-sha",
        "size",
        "clusters",
        "cluster-noise-bits",
        "queries",
        "rounds",
        "top-k",
        "noise-bits",
        "seed",
        "candidate-multiplier",
        "fallback-on-empty",
        "lsh-bands",
        "lsh-rows",
    ];
    let mut values = std::collections::HashMap::<String, String>::new();
    for arg in env::args().skip(1) {
        let Some(rest) = arg.strip_prefix("--") else {
            return Err(format!("unexpected argument {arg:?}; expected --name=value"));
        };
        let Some((key, value)) = rest.split_once('=') else {
            return Err(format!("argument {arg:?} must use --name=value form"));
        };
        if !allowed.contains(&key) {
            return Err(format!("unknown argument --{key}"));
        }
        if values.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(format!("duplicate argument --{key}"));
        }
    }

    let source_sha = values
        .get("source-sha")
        .ok_or_else(|| "--source-sha=<40-hex-commit> is required".to_owned())?
        .clone();
    if source_sha.len() != 40 || !source_sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("--source-sha must be exactly 40 hexadecimal characters".into());
    }

    let size = parse_value(&values, "size", DEFAULT_SIZE)?;
    let clusters = parse_value(&values, "clusters", DEFAULT_CLUSTERS.min(size))?;
    let cluster_noise_bits =
        parse_value(&values, "cluster-noise-bits", DEFAULT_CLUSTER_NOISE_BITS)?;
    let queries = parse_value(&values, "queries", DEFAULT_QUERIES)?;
    let rounds = parse_value(&values, "rounds", DEFAULT_ROUNDS)?;
    let top_k = parse_value(&values, "top-k", DEFAULT_TOP_K)?;
    let noise_bits = parse_value(&values, "noise-bits", DEFAULT_NOISE_BITS)?;
    let seed = parse_value(&values, "seed", DEFAULT_SEED)?;
    let candidate_multiplier = parse_value(&values, "candidate-multiplier", 4usize)?;
    let default_store_config = StoreConfig::default();
    let lsh_bands = parse_value(&values, "lsh-bands", default_store_config.lsh_bands)?;
    let lsh_rows = parse_value(&values, "lsh-rows", default_store_config.lsh_rows)?;
    let fallback_on_empty = match values.get("fallback-on-empty").map(String::as_str) {
        None | Some("true") => true,
        Some("false") => false,
        Some(other) => {
            return Err(format!(
                "--fallback-on-empty must be true or false, got {other:?}"
            ));
        }
    };

    if size == 0 || queries == 0 || rounds == 0 || top_k == 0 {
        return Err("--size, --queries, --rounds and --top-k must all be positive".into());
    }
    if clusters == 0 || clusters > size {
        return Err("--clusters must be positive and no greater than --size".into());
    }
    if noise_bits > BinaryHV::DIM || cluster_noise_bits > BinaryHV::DIM {
        return Err(format!(
            "--noise-bits and --cluster-noise-bits must be <= {}",
            BinaryHV::DIM
        ));
    }
    if u64::try_from(size).is_err() {
        return Err("--size exceeds the store's u64 ID range".into());
    }

    Ok(Config {
        source_sha,
        size,
        clusters,
        cluster_noise_bits,
        queries,
        rounds,
        top_k,
        noise_bits,
        seed,
        candidate_multiplier,
        fallback_on_empty,
        lsh_bands,
        lsh_rows,
    })
}

fn parse_value<T>(
    values: &std::collections::HashMap<String, String>,
    key: &str,
    default: T,
) -> Result<T, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    values
        .get(key)
        .map(|value| {
            value
                .parse::<T>()
                .map_err(|error| format!("invalid --{key} value {value:?}: {error}"))
        })
        .unwrap_or(Ok(default))
}

fn build_queries(config: &Config, store: &HdcStore) -> Vec<Query> {
    let mut queries = Vec::with_capacity(config.queries);
    for query_index in 0..config.queries {
        // One in five queries deliberately has no stored source vector. The
        // remaining queries are bounded-noise probes of known stored memories.
        if query_index % 5 == 0 {
            let novel_input = config.seed
                ^ (1u64 << 63)
                ^ (query_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            queries.push(Query {
                vector: BinaryHV::random(mix64(novel_input)),
                kind: "novel",
            });
        } else {
            let anchor = query_index
                .wrapping_mul(997)
                .wrapping_add((config.seed % config.size as u64) as usize)
                % config.size;
            let mut vector = *store
                .get(anchor as u64)
                .expect("generated corpus must contain the selected anchor ID");
            let start_bit = mix64(config.seed ^ query_index as u64) as usize % BinaryHV::DIM;
            flip_distinct_bits(&mut vector, config.noise_bits, start_bit);
            queries.push(Query {
                vector,
                kind: "near_neighbor",
            });
        }
    }
    queries
}

fn corpus_vector(config: &Config, id: usize) -> BinaryHV {
    let cluster_id = id % config.clusters;
    let mut vector = BinaryHV::random(cluster_seed(config.seed, cluster_id as u64));
    let noise_seed = config.seed ^ (id as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
    flip_seeded_bits(&mut vector, config.cluster_noise_bits, noise_seed);
    vector
}

fn cluster_seed(seed: u64, cluster_id: u64) -> u64 {
    mix64(seed ^ 0x434C_5553_5445_5200 ^ cluster_id.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn flip_seeded_bits(vector: &mut BinaryHV, count: usize, seed: u64) {
    let start_bit = mix64(seed) as usize % BinaryHV::DIM;
    let stride = (mix64(seed ^ 0xA5A5_5A5A_DEAD_BEEF) as usize % BinaryHV::DIM) | 1;
    for offset in 0..count {
        let bit_index = (start_bit + offset * stride) % BinaryHV::DIM;
        vector.0[bit_index / 8] ^= 1u8 << (bit_index % 8);
    }
}

fn flip_distinct_bits(vector: &mut BinaryHV, count: usize, start_bit: usize) {
    // 127 is coprime to 2^14, so the walk visits distinct bit positions
    // until it wraps after all BinaryHV::DIM bits.
    for offset in 0..count {
        let bit_index = (start_bit + offset * 127) % BinaryHV::DIM;
        vector.0[bit_index / 8] ^= 1u8 << (bit_index % 8);
    }
}

fn measure_exact(store: &HdcStore, query: &BinaryHV, top_k: usize) -> (Vec<(u64, f32)>, u128) {
    let start = Instant::now();
    let neighbors = std::hint::black_box(store.scan_similar(std::hint::black_box(query), top_k));
    (neighbors, start.elapsed().as_nanos())
}

fn measure_approximate(
    store: &HdcStore,
    query: &BinaryHV,
    top_k: usize,
    options: ApproximateSearchOptions,
) -> (symthaea_hdc_store::SearchOutcome, u128) {
    let start = Instant::now();
    let outcome = std::hint::black_box(store.scan_similar_approx(
        std::hint::black_box(query),
        top_k,
        options,
    ));
    (outcome, start.elapsed().as_nanos())
}

fn recall_at_k(exact_ids: &[u64], approximate_ids: &[u64]) -> f64 {
    if exact_ids.is_empty() {
        return 1.0;
    }
    let approximate: HashSet<u64> = approximate_ids.iter().copied().collect();
    exact_ids.iter().filter(|id| approximate.contains(id)).count() as f64 / exact_ids.len() as f64
}

fn fraction(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn percentile(values: &[u128], quantile: f64) -> u128 {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let rank = ((quantile.clamp(0.0, 1.0) * sorted.len() as f64).ceil() as usize)
        .max(1)
        .min(sorted.len());
    sorted[rank - 1]
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let collected: Vec<f64> = values.collect();
    if collected.is_empty() {
        0.0
    } else {
        collected.iter().sum::<f64>() / collected.len() as f64
    }
}

fn emit_report(config: &Config, samples: &[Sample], quality: &[Quality]) {
    println!("# protocol={PROTOCOL}");
    println!("# source_sha={}", config.source_sha);
    println!(
        "# workload=clustered BinaryHV memories; 80% known-memory probes; 20% random OOD probes"
    );
    println!("# store_size={}", config.size);
    println!("# clusters={}", config.clusters);
    println!("# cluster_noise_bits={}", config.cluster_noise_bits);
    println!("# unique_queries={}", config.queries);
    println!("# rounds={}", config.rounds);
    println!("# top_k={}", config.top_k);
    println!("# noise_bits={}", config.noise_bits);
    println!("# seed={}", config.seed);
    println!("# candidate_multiplier={}", config.candidate_multiplier);
    println!("# fallback_on_empty={}", config.fallback_on_empty);
    println!("# lsh_bands={}", config.lsh_bands);
    println!("# lsh_rows={}", config.lsh_rows);
    println!("# target_os={}", env::consts::OS);
    println!("# target_arch={}", env::consts::ARCH);
    let available_parallelism = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    println!("# available_parallelism={available_parallelism}");
    println!("# quality metrics below are computed over unique queries in round 0");
    println!(
        "query_index,round,kind,exact_ns,approximate_ns,recall_at_k,examined,total_live,candidate_fraction,approximate_exact,fell_back_to_exact,exact_ids,approximate_ids"
    );
    for sample in samples {
        println!(
            "{},{},{},{},{},{:.9},{},{},{:.9},{},{},\"{}\",\"{}\"",
            sample.query_index,
            sample.round,
            sample.kind,
            sample.exact_ns,
            sample.approximate_ns,
            sample.recall_at_k,
            sample.examined,
            sample.total_live,
            sample.candidate_fraction,
            sample.approximate_exact,
            sample.fell_back_to_exact,
            join_ids(&sample.exact_ids),
            join_ids(&sample.approximate_ids),
        );
    }

    let exact_ns: Vec<u128> = samples.iter().map(|sample| sample.exact_ns).collect();
    let approximate_ns: Vec<u128> = samples.iter().map(|sample| sample.approximate_ns).collect();
    let exact_p50 = percentile(&exact_ns, 0.50);
    let exact_p95 = percentile(&exact_ns, 0.95);
    let approximate_p50 = percentile(&approximate_ns, 0.50);
    let approximate_p95 = percentile(&approximate_ns, 0.95);
    let p95_ratio = if approximate_p95 == 0 {
        0.0
    } else {
        exact_p95 as f64 / approximate_p95 as f64
    };

    println!("# summary_all_unique_queries={}", quality.len());
    println!("# summary_latency_samples={}", samples.len());
    println!("# summary_all_mean_recall_at_k={:.9}", mean(quality.iter().map(|q| q.recall_at_k)));
    println!(
        "# summary_all_worst_recall_at_k={:.9}",
        quality.iter().map(|q| q.recall_at_k).fold(1.0, f64::min)
    );
    println!(
        "# summary_all_mean_candidate_fraction={:.9}",
        mean(quality.iter().map(|q| q.candidate_fraction))
    );
    println!(
        "# summary_all_mean_examined_vectors={:.3}",
        mean(quality.iter().map(|q| q.examined as f64))
    );
    println!(
        "# summary_all_exhaustive_query_fraction={:.9}",
        fraction(quality.iter().filter(|q| q.approximate_exact).count(), quality.len())
    );
    println!(
        "# summary_all_fallback_query_fraction={:.9}",
        fraction(quality.iter().filter(|q| q.fell_back_to_exact).count(), quality.len())
    );
    println!("# summary_all_exact_p50_ns={exact_p50}");
    println!("# summary_all_exact_p95_ns={exact_p95}");
    println!("# summary_all_approximate_p50_ns={approximate_p50}");
    println!("# summary_all_approximate_p95_ns={approximate_p95}");
    println!("# summary_all_ratio_exact_p95_over_approximate_p95={p95_ratio:.6}");
    emit_quality_summary("near_neighbor", quality);
    emit_quality_summary("novel", quality);
    emit_latency_summary("near_neighbor", samples);
    emit_latency_summary("novel", samples);
    println!(
        "# interpretation={}",
        concat!(
            "store-level search only; not end-to-end MemoryBridge latency ",
            "or downstream task quality"
        )
    );
}

fn emit_quality_summary(kind: &str, quality: &[Quality]) {
    let selected: Vec<&Quality> = quality.iter().filter(|query| query.kind == kind).collect();
    println!("# {kind}_query_count={}", selected.len());
    println!(
        "# {kind}_mean_recall_at_k={:.9}",
        mean(selected.iter().map(|query| query.recall_at_k))
    );
    println!(
        "# {kind}_worst_recall_at_k={:.9}",
        selected.iter().map(|query| query.recall_at_k).fold(1.0, f64::min)
    );
    println!(
        "# {kind}_mean_candidate_fraction={:.9}",
        mean(selected.iter().map(|query| query.candidate_fraction))
    );
    println!(
        "# {kind}_mean_examined_vectors={:.3}",
        mean(selected.iter().map(|query| query.examined as f64))
    );
    println!(
        "# {kind}_exhaustive_query_fraction={:.9}",
        fraction(
            selected.iter().filter(|query| query.approximate_exact).count(),
            selected.len()
        )
    );
    println!(
        "# {kind}_fallback_query_fraction={:.9}",
        fraction(
            selected.iter().filter(|query| query.fell_back_to_exact).count(),
            selected.len()
        )
    );
}

fn emit_latency_summary(kind: &str, samples: &[Sample]) {
    let exact_ns: Vec<u128> = samples
        .iter()
        .filter(|sample| sample.kind == kind)
        .map(|sample| sample.exact_ns)
        .collect();
    let approximate_ns: Vec<u128> = samples
        .iter()
        .filter(|sample| sample.kind == kind)
        .map(|sample| sample.approximate_ns)
        .collect();
    let exact_p50 = percentile(&exact_ns, 0.50);
    let exact_p95 = percentile(&exact_ns, 0.95);
    let approximate_p50 = percentile(&approximate_ns, 0.50);
    let approximate_p95 = percentile(&approximate_ns, 0.95);
    let ratio = if approximate_p95 == 0 {
        0.0
    } else {
        exact_p95 as f64 / approximate_p95 as f64
    };
    println!("# {kind}_exact_p50_ns={exact_p50}");
    println!("# {kind}_exact_p95_ns={exact_p95}");
    println!("# {kind}_approximate_p50_ns={approximate_p50}");
    println!("# {kind}_approximate_p95_ns={approximate_p95}");
    println!("# {kind}_ratio_exact_p95_over_approximate_p95={ratio:.6}");
}

fn join_ids(ids: &[u64]) -> String {
    ids.iter().map(u64::to_string).collect::<Vec<_>>().join(";")
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_noise_flips_exactly_the_requested_number_of_distinct_bits() {
        let original = BinaryHV::random(123);
        let mut mutated = original;
        flip_distinct_bits(&mut mutated, 128, 16_000);
        let changed_bits: usize = original
            .0
            .iter()
            .zip(mutated.0.iter())
            .map(|(left, right)| (left ^ right).count_ones() as usize)
            .sum();
        assert_eq!(changed_bits, 128);
    }

    #[test]
    fn full_dimension_noise_flips_each_bit_once() {
        let original = BinaryHV::random(456);
        let mut mutated = original;
        flip_distinct_bits(&mut mutated, BinaryHV::DIM, 7_777);
        let changed_bits: usize = original
            .0
            .iter()
            .zip(mutated.0.iter())
            .map(|(left, right)| (left ^ right).count_ones() as usize)
            .sum();
        assert_eq!(changed_bits, BinaryHV::DIM);
    }

    #[test]
    fn recall_uses_exact_result_count_as_denominator() {
        assert_eq!(recall_at_k(&[], &[7]), 1.0);
        assert!((recall_at_k(&[1, 2, 3], &[2, 3, 9]) - (2.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn nearest_rank_percentiles_are_deterministic() {
        assert_eq!(percentile(&[40, 10, 30, 20], 0.50), 20);
        assert_eq!(percentile(&[40, 10, 30, 20], 0.95), 40);
        assert_eq!(percentile(&[], 0.95), 0);
    }

    #[test]
    fn clustered_corpus_separates_within_cluster_from_between_cluster_similarity() {
        let config = Config {
            source_sha: "0123456789abcdef0123456789abcdef01234567".into(),
            size: 32,
            clusters: 8,
            cluster_noise_bits: 1_024,
            queries: 20,
            rounds: 1,
            top_k: 5,
            noise_bits: 128,
            seed: 42,
            candidate_multiplier: 4,
            fallback_on_empty: true,
            lsh_bands: 32,
            lsh_rows: 8,
        };
        let anchor = corpus_vector(&config, 0);
        let same_cluster = corpus_vector(&config, 8);
        let different_cluster = corpus_vector(&config, 1);
        assert!(anchor.similarity(&same_cluster) > 0.80);
        assert!(anchor.similarity(&different_cluster) < 0.65);
    }

    #[test]
    fn synthetic_query_workload_replays_for_the_same_seed() {
        let config = Config {
            source_sha: "0123456789abcdef0123456789abcdef01234567".into(),
            size: 128,
            clusters: 16,
            cluster_noise_bits: 1_024,
            queries: 20,
            rounds: 1,
            top_k: 5,
            noise_bits: 128,
            seed: 42,
            candidate_multiplier: 4,
            fallback_on_empty: true,
            lsh_bands: 32,
            lsh_rows: 8,
        };
        let dir = tempdir().unwrap();
        let mut store_config = StoreConfig::default();
        store_config.initial_capacity = config.size as u64;
        let mut store = HdcStore::create(dir.path().join("query-fixture.hdc"), store_config).unwrap();
        for id in 0..config.size {
            store.append(id as u64, &corpus_vector(&config, id)).unwrap();
        }
        let left = build_queries(&config, &store);
        let right = build_queries(&config, &store);
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right.iter()) {
            assert_eq!(left.kind, right.kind);
            assert_eq!(left.vector, right.vector);
        }
        assert_eq!(left.iter().filter(|query| query.kind == "novel").count(), 4);
        assert_eq!(left.iter().filter(|query| query.kind == "near_neighbor").count(), 16);
    }
}
