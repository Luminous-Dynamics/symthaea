// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Source-bound, deterministic benchmark for exact vs LSH HDC retrieval.
//!
//! Run with:
//! cargo run -p symthaea-hdc-store --example ann_benchmark --release -- \
//!   --source-sha=<40-hex-commit> --size=10000 --queries=200 --rounds=5 --top-k=10

use std::collections::{BTreeMap, HashSet};
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
const DEFAULT_HARD_NOISE_BITS: usize = BinaryHV::DIM / 4;
const DEFAULT_CLUSTERS: usize = 64;
const DEFAULT_CLUSTER_NOISE_BITS: usize = 1_024;
const DEFAULT_LSH_CONFIGS: &str = "32x8,16x16,64x4";
const BOOTSTRAP_RESAMPLES: usize = 2_000;
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
    hard_noise_bits: usize,
    seed: u64,
    candidate_multiplier: usize,
    fallback_on_empty: bool,
    lsh_configs: Vec<LshConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LshConfig {
    bands: u32,
    rows: u32,
}

impl LshConfig {
    fn label(self) -> String {
        format!("{}x{}", self.bands, self.rows)
    }
}

#[derive(Clone, Copy)]
struct Query {
    vector: BinaryHV,
    kind: &'static str,
    /// Statistical resampling block: one source cluster per known-memory lineage;
    /// novel/OOD probes get their own block.
    group_id: u64,
}

struct Sample {
    query_index: usize,
    round: usize,
    kind: &'static str,
    group_id: u64,
    lsh_bands: u32,
    lsh_rows: u32,
    exact_ns: u128,
    approximate_ns: u128,
    recall_at_k: f64,
    tie_aware_recall_at_k: f64,
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
    lsh_bands: u32,
    lsh_rows: u32,
    recall_at_k: f64,
    tie_aware_recall_at_k: f64,
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
    let mut corpus_config = StoreConfig::default();
    corpus_config.initial_capacity = u64::try_from(config.size)?;
    let corpus_path = dir.path().join("ann-corpus.hdc");
    let mut corpus_store = HdcStore::create(&corpus_path, corpus_config)?;
    for id in 0..config.size {
        corpus_store.append(id as u64, &corpus_vector(&config, id))?;
    }
    let queries = build_queries(&config, &corpus_store);
    drop(corpus_store);

    let options = ApproximateSearchOptions {
        candidate_multiplier: config.candidate_multiplier,
        fallback_on_empty: config.fallback_on_empty,
    };
    let mut samples = Vec::with_capacity(
        config.queries.saturating_mul(config.rounds).saturating_mul(config.lsh_configs.len()),
    );
    let mut first_round_quality = Vec::with_capacity(config.queries.saturating_mul(config.lsh_configs.len()));

    for lsh_config in config.lsh_configs.iter().copied() {
        // Every arm rebuilds the same deterministic corpus with only LSH band/row
        // parameters changed; queries remain identical across arms.
        let store_path = dir.path().join(format!("ann-{}.hdc", lsh_config.label()));
        let mut store_config = StoreConfig::default();
        store_config.initial_capacity = u64::try_from(config.size)?;
        store_config.lsh_bands = lsh_config.bands;
        store_config.lsh_rows = lsh_config.rows;
        let mut store = HdcStore::create(&store_path, store_config)?;
        for id in 0..config.size {
            store.append(id as u64, &corpus_vector(&config, id))?;
        }

        // Warm both paths before collecting samples. Timed intervals exclude corpus
        // construction, query generation, ID comparison and CSV output.
        for query in &queries {
            let _ = store.scan_similar(&query.vector, config.top_k);
            let _ = store.scan_similar_approx(&query.vector, config.top_k, options);
        }

        for round in 0..config.rounds {
            for (query_index, query) in queries.iter().enumerate() {
                // Alternate call order to reduce systematic cache/order bias.
                let (exact, exact_ns, approximate, approximate_ns) =
                    if (round + query_index) % 2 == 0 {
                        let (result, elapsed) =
                            measure_exact(&store, &query.vector, config.top_k);
                        let (ann, ann_elapsed) =
                            measure_approximate(&store, &query.vector, config.top_k, options);
                        (result, elapsed, ann, ann_elapsed)
                    } else {
                        let (ann, ann_elapsed) =
                            measure_approximate(&store, &query.vector, config.top_k, options);
                        let (result, elapsed) =
                            measure_exact(&store, &query.vector, config.top_k);
                        (result, elapsed, ann, ann_elapsed)
                    };

                let exact_ids: Vec<u64> = exact.iter().map(|(id, _)| *id).collect();
                let approximate_ids: Vec<u64> =
                    approximate.neighbors.iter().map(|(id, _)| *id).collect();
                let recall_at_k = recall_at_k(&exact_ids, &approximate_ids);
                let tie_aware_recall_at_k =
                    tie_aware_recall_at_k(&exact, &approximate.neighbors);
                let total_live = approximate.total_live;
                let candidate_fraction = fraction(approximate.examined, total_live);

                if round == 0 {
                    first_round_quality.push(Quality {
                        kind: query.kind,
                        lsh_bands: lsh_config.bands,
                        lsh_rows: lsh_config.rows,
                        recall_at_k,
                        tie_aware_recall_at_k,
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
                    group_id: query.group_id,
                    lsh_bands: lsh_config.bands,
                    lsh_rows: lsh_config.rows,
                    exact_ns,
                    approximate_ns,
                    recall_at_k,
                    tie_aware_recall_at_k,
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

    let root = Command::new("git")
        .args(["rev-parse", "--show-toplevel"])
        .output()?;
    if !root.status.success() {
        return Err(io::Error::other("git rev-parse --show-toplevel failed").into());
    }
    let root = String::from_utf8(root.stdout)?.trim().to_owned();
    let status = Command::new("git")
        .args([
            "-C",
            root.as_str(),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ])
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
        "hard-noise-bits",
        "seed",
        "candidate-multiplier",
        "fallback-on-empty",
        "lsh-configs",
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
    let hard_noise_bits = parse_value(&values, "hard-noise-bits", DEFAULT_HARD_NOISE_BITS)?;
    let seed = parse_value(&values, "seed", DEFAULT_SEED)?;
    let candidate_multiplier = parse_value(&values, "candidate-multiplier", 4usize)?;
    let lsh_configs = parse_lsh_configs(
        values.get("lsh-configs").map(String::as_str).unwrap_or(DEFAULT_LSH_CONFIGS),
    )?;
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
    if noise_bits > BinaryHV::DIM || hard_noise_bits > BinaryHV::DIM || cluster_noise_bits > BinaryHV::DIM {
        return Err(format!(
            "--noise-bits, --hard-noise-bits and --cluster-noise-bits must be <= {}",
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
        hard_noise_bits,
        seed,
        candidate_multiplier,
        fallback_on_empty,
        lsh_configs,
    })
}

fn parse_lsh_configs(input: &str) -> Result<Vec<LshConfig>, String> {
    let mut configs = Vec::new();
    for token in input.split(',') {
        let token = token.trim();
        if token.is_empty() {
            return Err("--lsh-configs contains an empty entry".into());
        }
        let Some((bands_text, rows_text)) = token.split_once('x') else {
            return Err(format!(
                "invalid LSH arm {token:?}; expected BANDSxROWS, e.g. 32x8"
            ));
        };
        let bands = bands_text
            .parse::<u32>()
            .map_err(|error| format!("invalid band count in {token:?}: {error}"))?;
        let rows = rows_text
            .parse::<u32>()
            .map_err(|error| format!("invalid row count in {token:?}: {error}"))?;
        if bands == 0 || bands > u32::from(u16::MAX) || !(1..=32).contains(&rows) {
            return Err(format!(
                "LSH arm {token:?} must have 1..={} bands and 1..=32 rows",
                u16::MAX
            ));
        }
        let config = LshConfig { bands, rows };
        if configs.contains(&config) {
            return Err(format!("duplicate LSH arm {}", config.label()));
        }
        configs.push(config);
        if configs.len() > 16 {
            return Err("--lsh-configs supports at most 16 arms per run".into());
        }
    }
    if configs.is_empty() {
        return Err("--lsh-configs must contain at least one BANDSxROWS arm".into());
    }
    Ok(configs)
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
        // The deterministic schedule is 60% low-noise known probes, 20% hard
        // known probes (large but bounded perturbation), and 20% novel/OOD probes.
        // Only the known-memory classes have a source anchor.
        if query_index % 5 == 0 {
            let novel_input = config.seed
                ^ (1u64 << 63)
                ^ (query_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            queries.push(Query {
                vector: BinaryHV::random(mix64(novel_input)),
                kind: "novel",
                // Keep the OOD group ID disjoint from all source-cluster IDs.
                group_id: config.clusters as u64 + query_index as u64,
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
            let (kind, noise_bits) = if query_index % 5 == 1 {
                ("hard_neighbor", config.hard_noise_bits)
            } else {
                ("near_neighbor", config.noise_bits)
            };
            flip_distinct_bits(&mut vector, noise_bits, start_bit);
            queries.push(Query {
                vector,
                kind,
                // Probes from a shared source cluster are not assumed independent.
                group_id: (anchor % config.clusters) as u64,
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

fn tie_aware_recall_at_k(exact: &[(u64, f32)], approximate: &[(u64, f32)]) -> f64 {
    let Some((_, cutoff_similarity)) = exact.last() else {
        return 1.0;
    };
    let recovered = approximate
        .iter()
        .filter(|neighbor| neighbor.1 >= *cutoff_similarity)
        .count()
        .min(exact.len());
    recovered as f64 / exact.len() as f64
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
    println!("# workload=clustered BinaryHV memories; 60% low-noise known probes; 20% hard-neighbor probes; 20% random OOD probes");
    println!("# store_size={}", config.size);
    println!("# clusters={}", config.clusters);
    println!("# cluster_noise_bits={}", config.cluster_noise_bits);
    println!("# unique_queries={}", config.queries);
    println!("# rounds={}", config.rounds);
    println!("# top_k={}", config.top_k);
    println!("# noise_bits={}", config.noise_bits);
    println!("# hard_noise_bits={}", config.hard_noise_bits);
    println!("# seed={}", config.seed);
    println!("# candidate_multiplier={}", config.candidate_multiplier);
    println!("# fallback_on_empty={}", config.fallback_on_empty);
    println!("# lsh_configs={}", config.lsh_configs.iter().map(|arm| arm.label()).collect::<Vec<_>>().join(";"));
    println!("# target_os={}", env::consts::OS);
    println!("# target_arch={}", env::consts::ARCH);
    let available_parallelism = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    println!("# available_parallelism={available_parallelism}");
    println!("# quality metrics are computed from unique queries (round 0); latencies pool repeated rounds");
    println!("# bootstrap_unit=source_cluster_for_near_neighbor; singleton_query_for_novel");
    println!("# bootstrap_resampling=sample query medians by group with replacement, retaining all queries in each sampled group");
    println!(
        "query_index,round,kind,group_id,lsh_bands,lsh_rows,exact_ns,approximate_ns,recall_at_k,tie_aware_recall_at_k,examined,total_live,candidate_fraction,approximate_exact,fell_back_to_exact,exact_ids,approximate_ids"
    );
    for sample in samples {
        println!(
            "{},{},{},{},{},{},{},{},{:.9},{:.9},{},{},{:.9},{},{},{},{}",
            sample.query_index,
            sample.round,
            sample.kind,
            sample.group_id,
            sample.lsh_bands,
            sample.lsh_rows,
            sample.exact_ns,
            sample.approximate_ns,
            sample.recall_at_k,
            sample.tie_aware_recall_at_k,
            sample.examined,
            sample.total_live,
            sample.candidate_fraction,
            sample.approximate_exact,
            sample.fell_back_to_exact,
            join_ids(&sample.exact_ids),
            join_ids(&sample.approximate_ids),
        );
    }

    for arm in config.lsh_configs.iter().copied() {
        let label = arm.label();
        let arm_samples: Vec<&Sample> = samples
            .iter()
            .filter(|sample| sample.lsh_bands == arm.bands && sample.lsh_rows == arm.rows)
            .collect();
        let arm_quality: Vec<&Quality> = quality
            .iter()
            .filter(|query| query.lsh_bands == arm.bands && query.lsh_rows == arm.rows)
            .collect();
        let exact_ns: Vec<u128> = arm_samples.iter().map(|sample| sample.exact_ns).collect();
        let approximate_ns: Vec<u128> =
            arm_samples.iter().map(|sample| sample.approximate_ns).collect();
        let exact_p50 = percentile(&exact_ns, 0.50);
        let exact_p95 = percentile(&exact_ns, 0.95);
        let approximate_p50 = percentile(&approximate_ns, 0.50);
        let approximate_p95 = percentile(&approximate_ns, 0.95);
        let p95_ratio = if approximate_p95 == 0 {
            0.0
        } else {
            exact_p95 as f64 / approximate_p95 as f64
        };
        println!("# arm_{label}_unique_queries={}", arm_quality.len());
        println!("# arm_{label}_latency_samples={}", arm_samples.len());
        println!(
            "# arm_{label}_all_mean_recall_at_k={:.9}",
            mean(arm_quality.iter().map(|query| query.recall_at_k))
        );
        println!(
            "# arm_{label}_all_mean_tie_aware_recall_at_k={:.9}",
            mean(arm_quality.iter().map(|query| query.tie_aware_recall_at_k))
        );
        println!(
            "# arm_{label}_all_worst_recall_at_k={:.9}",
            arm_quality.iter().map(|query| query.recall_at_k).fold(1.0, f64::min)
        );
        println!(
            "# arm_{label}_all_mean_candidate_fraction={:.9}",
            mean(arm_quality.iter().map(|query| query.candidate_fraction))
        );
        println!(
            "# arm_{label}_all_mean_examined_vectors={:.3}",
            mean(arm_quality.iter().map(|query| query.examined as f64))
        );
        println!(
            "# arm_{label}_all_exhaustive_query_fraction={:.9}",
            fraction(
                arm_quality.iter().filter(|query| query.approximate_exact).count(),
                arm_quality.len()
            )
        );
        println!(
            "# arm_{label}_all_fallback_query_fraction={:.9}",
            fraction(
                arm_quality.iter().filter(|query| query.fell_back_to_exact).count(),
                arm_quality.len()
            )
        );
        println!("# arm_{label}_exact_p50_ns={exact_p50}");
        println!("# arm_{label}_exact_p95_ns={exact_p95}");
        println!("# arm_{label}_approximate_p50_ns={approximate_p50}");
        println!("# arm_{label}_approximate_p95_ns={approximate_p95}");
        println!("# arm_{label}_ratio_exact_p95_over_approximate_p95={p95_ratio:.6}");
        for kind in ["near_neighbor", "hard_neighbor", "novel"] {
            emit_quality_summary(kind, arm, quality);
            emit_latency_summary(kind, arm, samples);
            let grouped_speedups = paired_query_speedups(kind, arm, samples);
            let speedups: Vec<f64> =
                grouped_speedups.iter().map(|(_, speedup)| *speedup).collect();
            let seed = config.seed
                ^ (u64::from(arm.bands) << 32)
                ^ u64::from(arm.rows)
                ^ kind_seed(kind);
            let (lower, upper) = bootstrap_grouped_median_ci(&grouped_speedups, seed);
            println!(
                "# arm_{label}_{kind}_median_per_query_speedup={:.6}",
                median_f64(&speedups)
            );
            println!("# arm_{label}_{kind}_median_speedup_group_bootstrap_95ci_low={lower:.6}");
            println!("# arm_{label}_{kind}_median_speedup_group_bootstrap_95ci_high={upper:.6}");
            println!("# arm_{label}_{kind}_bootstrap_query_count={}", speedups.len());
            println!(
                "# arm_{label}_{kind}_bootstrap_group_count={}",
                grouped_speedups.iter().map(|(group, _)| *group).collect::<HashSet<_>>().len()
            );
            let group_count = grouped_speedups
                .iter()
                .map(|(group, _)| *group)
                .collect::<HashSet<_>>()
                .len();
            println!(
                "# arm_{label}_{kind}_bootstrap_resamples={}",
                if group_count < 2 { 0 } else { BOOTSTRAP_RESAMPLES }
            );
        }
    }
    println!(
        "# interpretation={}",
        concat!(
            "store-level search only; not end-to-end MemoryBridge latency ",
            "or downstream task quality"
        )
    );
}

fn emit_quality_summary(kind: &str, arm: LshConfig, quality: &[Quality]) {
    let selected: Vec<&Quality> = quality
        .iter()
        .filter(|query| {
            query.kind == kind && query.lsh_bands == arm.bands && query.lsh_rows == arm.rows
        })
        .collect();
    println!("# arm_{}_{}_query_count={}", arm.label(), kind, selected.len());
    println!(
        "# arm_{}_{}_mean_recall_at_k={:.9}",
        arm.label(),
        kind,
        mean(selected.iter().map(|query| query.recall_at_k))
    );
    println!(
        "# arm_{}_{}_mean_tie_aware_recall_at_k={:.9}",
        arm.label(),
        kind,
        mean(selected.iter().map(|query| query.tie_aware_recall_at_k))
    );
    println!(
        "# arm_{}_{}_worst_recall_at_k={:.9}",
        arm.label(),
        kind,
        selected.iter().map(|query| query.recall_at_k).fold(1.0, f64::min)
    );
    println!(
        "# arm_{}_{}_mean_candidate_fraction={:.9}",
        arm.label(),
        kind,
        mean(selected.iter().map(|query| query.candidate_fraction))
    );
    println!(
        "# arm_{}_{}_mean_examined_vectors={:.3}",
        arm.label(),
        kind,
        mean(selected.iter().map(|query| query.examined as f64))
    );
    println!(
        "# arm_{}_{}_exhaustive_query_fraction={:.9}",
        arm.label(),
        kind,
        fraction(
            selected.iter().filter(|query| query.approximate_exact).count(),
            selected.len()
        )
    );
    println!(
        "# arm_{}_{}_fallback_query_fraction={:.9}",
        arm.label(),
        kind,
        fraction(
            selected.iter().filter(|query| query.fell_back_to_exact).count(),
            selected.len()
        )
    );
}

fn emit_latency_summary(kind: &str, arm: LshConfig, samples: &[Sample]) {
    let exact_ns: Vec<u128> = samples
        .iter()
        .filter(|sample| {
            sample.kind == kind && sample.lsh_bands == arm.bands && sample.lsh_rows == arm.rows
        })
        .map(|sample| sample.exact_ns)
        .collect();
    let approximate_ns: Vec<u128> = samples
        .iter()
        .filter(|sample| {
            sample.kind == kind && sample.lsh_bands == arm.bands && sample.lsh_rows == arm.rows
        })
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
    println!("# arm_{}_{}_exact_p50_ns={exact_p50}", arm.label(), kind);
    println!("# arm_{}_{}_exact_p95_ns={exact_p95}", arm.label(), kind);
    println!("# arm_{}_{}_approximate_p50_ns={approximate_p50}", arm.label(), kind);
    println!("# arm_{}_{}_approximate_p95_ns={approximate_p95}", arm.label(), kind);
    println!(
        "# arm_{}_{}_ratio_exact_p95_over_approximate_p95={ratio:.6}",
        arm.label(),
        kind
    );
}

fn paired_query_speedups(kind: &str, arm: LshConfig, samples: &[Sample]) -> Vec<(u64, f64)> {
    let mut speedups = Vec::new();
    for first_sample in samples.iter().filter(|sample| {
        sample.round == 0
            && sample.kind == kind
            && sample.lsh_bands == arm.bands
            && sample.lsh_rows == arm.rows
    }) {
        let ratios: Vec<f64> = samples
            .iter()
            .filter(|sample| {
                sample.query_index == first_sample.query_index
                    && sample.kind == kind
                    && sample.lsh_bands == arm.bands
                    && sample.lsh_rows == arm.rows
            })
            .filter_map(|sample| {
                (sample.approximate_ns > 0)
                    .then_some(sample.exact_ns as f64 / sample.approximate_ns as f64)
            })
            .collect();
        if !ratios.is_empty() {
            speedups.push((first_sample.group_id, median_f64(&ratios)));
        }
    }
    speedups
}

fn median_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

fn percentile_f64_sorted(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((quantile.clamp(0.0, 1.0) * sorted.len() as f64).ceil() as usize)
        .max(1)
        .min(sorted.len());
    sorted[rank - 1]
}

fn bootstrap_grouped_median_ci(values: &[(u64, f64)], seed: u64) -> (f64, f64) {
    let mut groups = BTreeMap::<u64, Vec<f64>>::new();
    for &(group_id, value) in values {
        groups.entry(group_id).or_default().push(value);
    }
    if groups.len() < 2 {
        let point = median_f64(&values.iter().map(|(_, value)| *value).collect::<Vec<_>>());
        return (point, point);
    }

    // Cluster bootstrap: resample independent groups, not individual queries.
    // All per-query observations in a selected group travel together.
    let blocks: Vec<&[f64]> = groups.values().map(Vec::as_slice).collect();
    let mut estimates = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    for resample in 0..BOOTSTRAP_RESAMPLES {
        let mut sample = Vec::with_capacity(values.len());
        let mut state = mix64(seed ^ (resample as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        for _ in 0..blocks.len() {
            state = mix64(state.wrapping_add(0xD1B5_4A32_D192_ED03));
            let index = (state % blocks.len() as u64) as usize;
            sample.extend_from_slice(blocks[index]);
        }
        estimates.push(median_f64(&sample));
    }
    estimates.sort_by(f64::total_cmp);
    (
        percentile_f64_sorted(&estimates, 0.025),
        percentile_f64_sorted(&estimates, 0.975),
    )
}

fn kind_seed(kind: &str) -> u64 {
    kind.bytes()
        .fold(0u64, |state, byte| state.wrapping_mul(131).wrapping_add(u64::from(byte)))
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
    fn tie_aware_recall_accepts_equivalent_items_at_exact_cutoff() {
        let exact = vec![(1, 0.9), (2, 0.8)];
        let approximate = vec![(1, 0.9), (7, 0.8)];
        assert_eq!(recall_at_k(&[1, 2], &[1, 7]), 0.5);
        assert_eq!(tie_aware_recall_at_k(&exact, &approximate), 1.0);
    }

    #[test]
    fn nearest_rank_percentiles_are_deterministic() {
        assert_eq!(percentile(&[40, 10, 30, 20], 0.50), 20);
        assert_eq!(percentile(&[40, 10, 30, 20], 0.95), 40);
        assert_eq!(percentile(&[], 0.95), 0);
    }

    #[test]
    fn lsh_config_parser_preserves_declared_sweep_order() {
        assert_eq!(
            parse_lsh_configs("32x8, 16x16,64x4").unwrap(),
            vec![
                LshConfig { bands: 32, rows: 8 },
                LshConfig { bands: 16, rows: 16 },
                LshConfig { bands: 64, rows: 4 },
            ]
        );
    }

    #[test]
    fn lsh_config_parser_rejects_duplicates_and_invalid_arms() {
        assert!(parse_lsh_configs("").is_err());
        assert!(parse_lsh_configs("32x8,").is_err());
        assert!(parse_lsh_configs("32x8,32x8").is_err());
        assert!(parse_lsh_configs("0x8").is_err());
        assert!(parse_lsh_configs("8x0").is_err());
        assert!(parse_lsh_configs("8x33").is_err());
        assert!(parse_lsh_configs("65536x8").is_err());
        assert!(parse_lsh_configs("not-an-arm").is_err());
    }

    #[test]
    fn grouped_bootstrap_is_reproducible_and_collapses_for_constant_data() {
        let values: Vec<(u64, f64)> = (0..8).map(|group| (group, 2.0)).collect();
        let first = bootstrap_grouped_median_ci(&values, 12345);
        let second = bootstrap_grouped_median_ci(&values, 12345);
        assert_eq!(first, second);
        assert_eq!(first, (2.0, 2.0));
        assert_eq!(median_f64(&values.iter().map(|(_, value)| *value).collect::<Vec<_>>()), 2.0);
    }

    #[test]
    fn grouped_bootstrap_keeps_queries_from_the_same_cluster_together() {
        let values = vec![
            (10, 1.0),
            (10, 1.2),
            (10, 1.1),
            (20, 4.0),
            (20, 4.2),
            (30, 8.0),
        ];
        let first = bootstrap_grouped_median_ci(&values, 77);
        let second = bootstrap_grouped_median_ci(&values, 77);
        assert_eq!(first, second);
        assert!(first.0.is_finite());
        assert!(first.1.is_finite());
        assert!(first.0 <= first.1);
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
            hard_noise_bits: DEFAULT_HARD_NOISE_BITS,
            seed: 42,
            candidate_multiplier: 4,
            fallback_on_empty: true,
            lsh_configs: vec![LshConfig { bands: 32, rows: 8 }],
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
            hard_noise_bits: DEFAULT_HARD_NOISE_BITS,
            seed: 42,
            candidate_multiplier: 4,
            fallback_on_empty: true,
            lsh_configs: vec![LshConfig { bands: 32, rows: 8 }],
        };
        let dir = tempdir().unwrap();
        let mut store_config = StoreConfig::default();
        store_config.initial_capacity = config.size as u64;
        let mut store = HdcStore::create(dir.path().join("query-fixture.hdc"), store_config).unwrap();
        for id in 0..config.size {
            store.append(id as u64, &corpus_vector(&config, id)).unwrap();
        }
        let left_queries = build_queries(&config, &store);
        let right_queries = build_queries(&config, &store);
        assert_eq!(left_queries.len(), right_queries.len());
        for (left, right) in left_queries.iter().zip(right_queries.iter()) {
            assert_eq!(left.kind, right.kind);
            assert_eq!(left.vector, right.vector);
            assert_eq!(left.group_id, right.group_id);
        }
        assert_eq!(
            left_queries.iter().filter(|query| query.kind == "novel").count(),
            4
        );
        assert_eq!(
            left_queries
                .iter()
                .filter(|query| query.kind == "near_neighbor")
                .count(),
            12
        );
        assert_eq!(
            left_queries
                .iter()
                .filter(|query| query.kind == "hard_neighbor")
                .count(),
            4
        );
        let hard_anchor = 997usize
            .wrapping_add((config.seed % config.size as u64) as usize)
            % config.size;
        let original_anchor = store.get(hard_anchor as u64).unwrap();
        let hard_distance: usize = original_anchor
            .0
            .iter()
            .zip(left_queries[1].vector.0.iter())
            .map(|(left, right)| (left ^ right).count_ones() as usize)
            .sum();
        assert_eq!(left_queries[1].kind, "hard_neighbor");
        assert_eq!(hard_distance, config.hard_noise_bits);
        assert!(left_queries
            .iter()
            .filter(|query| query.kind == "near_neighbor")
            .all(|query| query.group_id < config.clusters as u64));
        assert!(left_queries
            .iter()
            .filter(|query| query.kind == "novel")
            .all(|query| query.group_id >= config.clusters as u64));
    }
}
