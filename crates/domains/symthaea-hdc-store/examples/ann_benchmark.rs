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
const DEFAULT_SEED: u64 = 42;

#[derive(Debug)]
struct Config {
    source_sha: String,
    size: usize,
    queries: usize,
    rounds: usize,
    top_k: usize,
    noise_bits: usize,
    seed: u64,
    candidate_multiplier: usize,
    fallback_on_empty: bool,
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
    let dir = tempdir()?;
    let path = dir.path().join("ann-benchmark.hdc");

    let mut store_config = StoreConfig::default();
    store_config.initial_capacity = u64::try_from(config.size)?;
    let mut store = HdcStore::create(&path, store_config)?;

    for id in 0..config.size {
        store.append(id as u64, &BinaryHV::random(memory_seed(config.seed, id as u64)))?;
    }

    let queries = build_queries(&config);
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

fn parse_args() -> Result<Config, String> {
    let allowed = [
        "source-sha",
        "size",
        "queries",
        "rounds",
        "top-k",
        "noise-bits",
        "seed",
        "candidate-multiplier",
        "fallback-on-empty",
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
    let queries = parse_value(&values, "queries", DEFAULT_QUERIES)?;
    let rounds = parse_value(&values, "rounds", DEFAULT_ROUNDS)?;
    let top_k = parse_value(&values, "top-k", DEFAULT_TOP_K)?;
    let noise_bits = parse_value(&values, "noise-bits", DEFAULT_NOISE_BITS)?;
    let seed = parse_value(&values, "seed", DEFAULT_SEED)?;
    let candidate_multiplier = parse_value(&values, "candidate-multiplier", 4usize)?;
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
    if noise_bits > BinaryHV::DIM {
        return Err(format!("--noise-bits must be <= {}", BinaryHV::DIM));
    }
    if u64::try_from(size).is_err() {
        return Err("--size exceeds the store's u64 ID range".into());
    }

    Ok(Config {
        source_sha,
        size,
        queries,
        rounds,
        top_k,
        noise_bits,
        seed,
        candidate_multiplier,
        fallback_on_empty,
    })
}

fn parse_value<T>(values: &std::collections::HashMap<String, String>, key: &str, default: T) -> Result<T, String>
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

fn build_queries(config: &Config) -> Vec<Query> {
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
            let mut vector = BinaryHV::random(memory_seed(config.seed, anchor as u64));
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

fn memory_seed(seed: u64, id: u64) -> u64 {
    mix64(seed ^ id.wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
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
    println!("# workload=80% bounded-noise stored-memory probes; 20% novel probes");
    println!("# store_size={}", config.size);
    println!("# unique_queries={}", config.queries);
    println!("# rounds={}", config.rounds);
    println!("# top_k={}", config.top_k);
    println!("# noise_bits={}", config.noise_bits);
    println!("# seed={}", config.seed);
    println!("# candidate_multiplier={}", config.candidate_multiplier);
    println!("# fallback_on_empty={}", config.fallback_on_empty);
    println!("# target_os={}", env::consts::OS);
    println!("# target_arch={}", env::consts::ARCH);
    println!("# available_parallelism={}", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    println!("# quality metrics below are computed over unique queries in round 0");
    println!("query_index,round,kind,exact_ns,approximate_ns,recall_at_k,examined,total_live,candidate_fraction,approximate_exact,fell_back_to_exact,exact_ids,approximate_ids");
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

    println!("# summary_unique_queries={}", quality.len());
    println!("# summary_latency_samples={}", samples.len());
    println!("# summary_mean_recall_at_k={:.9}", mean(quality.iter().map(|q| q.recall_at_k)));
    println!("# summary_worst_recall_at_k={:.9}", quality.iter().map(|q| q.recall_at_k).fold(1.0, f64::min));
    println!("# summary_mean_candidate_fraction={:.9}", mean(quality.iter().map(|q| q.candidate_fraction)));
    println!("# summary_mean_examined_vectors={:.3}", mean(quality.iter().map(|q| q.examined as f64)));
    println!("# summary_exhaustive_query_fraction={:.9}", fraction(quality.iter().filter(|q| q.approximate_exact).count(), quality.len()));
    println!("# summary_fallback_query_fraction={:.9}", fraction(quality.iter().filter(|q| q.fell_back_to_exact).count(), quality.len()));
    println!("# summary_exact_p50_ns={exact_p50}");
    println!("# summary_exact_p95_ns={exact_p95}");
    println!("# summary_approximate_p50_ns={approximate_p50}");
    println!("# summary_approximate_p95_ns={approximate_p95}");
    println!("# summary_ratio_exact_p95_over_approximate_p95={p95_ratio:.6}");
    println!("# interpretation=store-level search only; not end-to-end MemoryBridge latency or downstream task quality");
}

fn join_ids(ids: &[u64]) -> String {
    ids.iter().map(u64::to_string).collect::<Vec<_>>().join(";")
}
