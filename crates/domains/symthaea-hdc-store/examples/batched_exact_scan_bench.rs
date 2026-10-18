// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Exact search benchmark: repeated single-query scans vs corpus-major query batches.
//!
//! Run from the repository root:
//! cargo run -p symthaea-hdc-store --release --example batched_exact_scan_bench -- \
//!   --vectors 16384 --queries 32 --top-k 10 --batch-sizes 1,2,4,8,16 \
//!   --rounds 30 --seed 7 --csv /tmp/hdc-batched-exact.csv
//!
//! This is an exact software-layout experiment, not an approximation benchmark.
//! Both paths call the same BinaryHV similarity kernel and use identical,
//! deterministic ranking (score descending, then ID ascending). Cache state is
//! not controlled; this measures the named host's mapped-store behavior.

use std::error::Error;
use std::env;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use symthaea_core::hdc::BinaryHV;
use symthaea_hdc_store::{HdcStore, HdcStoreReader, StoreConfig, WriteBatch};

const MAX_VECTORS: usize = 131_072;
const MAX_QUERIES: usize = 4_096;
const MAX_BATCH_SIZE: usize = 32;
const MAX_SCORE_TUPLES: usize = 4_194_304;

#[derive(Debug)]
struct Config {
    vectors: usize,
    queries: usize,
    top_k: usize,
    batch_sizes: Vec<usize>,
    rounds: usize,
    write_batch_size: usize,
    seed: u64,
    csv_path: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vectors: 4_096,
            queries: 32,
            top_k: 10,
            batch_sizes: vec![1, 2, 4, 8, 16],
            rounds: 5,
            write_batch_size: 128,
            seed: 7,
            csv_path: None,
        }
    }
}

#[derive(Debug)]
struct RawSample {
    algorithm: &'static str,
    batch_size: usize,
    round: usize,
    batch_index: usize,
    query_count: usize,
    elapsed_ns: u128,
    vectors: usize,
    top_k: usize,
    checksum: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("batched_exact_scan_bench: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    validate_config(&config)?;

    if let Some(path) = config.csv_path.as_ref() {
        let meta = path.with_extension("meta.txt");
        if path.exists() || meta.exists() {
            return Err(format!(
                "benchmark output already exists; refusing to overwrite CSV or sidecar: {}",
                path.display()
            ).into());
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.is_dir() {
                return Err(format!("CSV parent directory does not exist: {}", parent.display()).into());
            }
        }
    }

    let tempdir = tempfile::tempdir()?;
    let store_path = tempdir.path().join("batched-exact-workload.hdc");
    let store_config = StoreConfig {
        initial_capacity: config.vectors as u64,
        ..StoreConfig::default()
    };

    let create_start = Instant::now();
    let mut store = HdcStore::create(&store_path, store_config)?;
    let create_ns = create_start.elapsed().as_nanos();

    let generation_start = Instant::now();
    let mut corpus = Vec::with_capacity(config.vectors);
    for index in 0..config.vectors {
        corpus.push(BinaryHV::random(config.seed.wrapping_add(index as u64 + 1)));
    }
    let generation_ns = generation_start.elapsed().as_nanos();

    for (batch_index, vectors) in corpus.chunks(config.write_batch_size).enumerate() {
        let offset = batch_index * config.write_batch_size;
        let mut batch = WriteBatch::new();
        for (local_index, hv) in vectors.iter().enumerate() {
            batch.push_append((offset + local_index) as u64 + 1, *hv);
        }
        let report = store.apply_batch(batch)?;
        if report.appended != vectors.len() as u64 || report.deleted != 0 {
            return Err(format!("unexpected corpus write report: {report:?}").into());
        }
    }
    store.sync_all()?;
    let content_checksum = store.content_checksum();
    if store.live_count() != config.vectors as u64 {
        return Err(format!(
            "live-count mismatch: expected {}, got {}",
            config.vectors,
            store.live_count()
        ).into());
    }
    drop(store);
    drop(corpus);

    let reader_open_start = Instant::now();
    let reader = HdcStoreReader::open(&store_path)?;
    let reader_open_ns = reader_open_start.elapsed().as_nanos();

    // Deterministic query vectors are not derived from corpus contents, avoiding
    // a benchmark that only measures exact self-matches.
    let query_seed_base = config.seed ^ 0xD1B5_4A32_D192_ED03;
    let queries: Vec<BinaryHV> = (0..config.queries)
        .map(|index| BinaryHV::random(query_seed_base.wrapping_add(index as u64)))
        .collect();

    // Compute the reference before timed runs. This is deliberately the public
    // exact API; each candidate batch must match these complete ordered top-k
    // lists, including scores and ID tie-breaking.
    let reference_start = Instant::now();
    let reference: Vec<Vec<(u64, f32)>> = queries
        .iter()
        .map(|query| reader.scan_similar(query, config.top_k))
        .collect();
    let reference_ns = reference_start.elapsed().as_nanos();

    println!("benchmark=batched_exact_scan_bench");
    println!("host_os={},host_arch={}", env::consts::OS, env::consts::ARCH);
    println!("source_sha={}", env::var("SYMTHAEA_SOURCE_SHA").unwrap_or_else(|_| "not-injected".into()));
    println!("rustc_version={}", env::var("RUSTC_VERSION").unwrap_or_else(|_| "not-injected".into()));
    println!("host_profile={}", env::var("HOST_PROFILE").unwrap_or_else(|_| "not-injected".into()));
    println!("storage_profile={}", env::var("STORAGE_PROFILE").unwrap_or_else(|_| "not-injected".into()));
    println!(
        "vectors={},queries={},top_k={},batch_sizes={:?},rounds={},seed={}",
        config.vectors, config.queries, config.top_k, config.batch_sizes, config.rounds, config.seed
    );
    println!("store_content_checksum={content_checksum}");
    println!("setup_store_create_ns={create_ns},corpus_generate_ns={generation_ns},reader_open_ns={reader_open_ns},reference_ns={reference_ns}");
    println!("cache_state=not_controlled; measurements include warm mapped-store behavior after reference generation");
    println!("exact_contract=every timed batch must match canonical per-query IDs, scores, tie ordering");
    println!("score_memory_budget_tuples={MAX_SCORE_TUPLES}; approximate search is not used");

    let mut samples = Vec::new();
    for &batch_size in &config.batch_sizes {
        let groups: Vec<(usize, &[BinaryHV])> = queries
            .chunks(batch_size)
            .enumerate()
            .map(|(index, batch)| (index, batch))
            .collect();

        for round in 0..config.rounds {
            for (batch_index, batch_queries) in &groups {
                // Alternate algorithm order across rounds/groups to limit
                // systematic order bias. Cache is not forcibly flushed.
                let baseline_first = (round + batch_index) % 2 == 0;
                if baseline_first {
                    let row = time_baseline(
                        &reader, batch_queries, *batch_index, round, batch_size,
                        config.vectors, config.top_k, &reference,
                        (batch_index * batch_size)..((batch_index * batch_size) + batch_queries.len()),
                    )?;
                    samples.push(row);
                    let row = time_fused(
                        &reader, batch_queries, *batch_index, round, batch_size,
                        config.vectors, config.top_k, &reference,
                        (batch_index * batch_size)..((batch_index * batch_size) + batch_queries.len()),
                    )?;
                    samples.push(row);
                } else {
                    let row = time_fused(
                        &reader, batch_queries, *batch_index, round, batch_size,
                        config.vectors, config.top_k, &reference,
                        (batch_index * batch_size)..((batch_index * batch_size) + batch_queries.len()),
                    )?;
                    samples.push(row);
                    let row = time_baseline(
                        &reader, batch_queries, *batch_index, round, batch_size,
                        config.vectors, config.top_k, &reference,
                        (batch_index * batch_size)..((batch_index * batch_size) + batch_queries.len()),
                    )?;
                    samples.push(row);
                }
            }
        }
    }

    print_summary(&samples);
    let csv = render_csv(&samples);
    if let Some(path) = config.csv_path.as_ref() {
        fs::write(path, csv.as_bytes())?;
        let metadata_path = path.with_extension("meta.txt");
        let metadata = format!(
            "benchmark=batched_exact_scan_bench\nsource_sha={}\nrustc_version={}\nhost_os={}\nhost_arch={}\nhost_profile={}\nstorage_profile={}\nvectors={}\nqueries={}\ntop_k={}\nbatch_sizes={:?}\nrounds={}\nwrite_batch_size={}\nseed={}\nstore_content_checksum={}\ncache_state=not_controlled\nphysical_device_io=not_measured\nenergy=not_measured\napproximate_search=not_used\n",
            env::var("SYMTHAEA_SOURCE_SHA").unwrap_or_else(|_| "not-injected".into()),
            env::var("RUSTC_VERSION").unwrap_or_else(|_| "not-injected".into()),
            env::consts::OS,
            env::consts::ARCH,
            env::var("HOST_PROFILE").unwrap_or_else(|_| "not-injected".into()),
            env::var("STORAGE_PROFILE").unwrap_or_else(|_| "not-injected".into()),
            config.vectors,
            config.queries,
            config.top_k,
            config.batch_sizes,
            config.rounds,
            config.write_batch_size,
            config.seed,
            content_checksum,
        );
        fs::write(&metadata_path, metadata)?;
        eprintln!("raw_samples_csv={}", path.display());
        eprintln!("run_metadata={}", metadata_path.display());
    }
    println!("RAW_SAMPLE_CSV_BEGIN");
    print!("{csv}");
    println!("RAW_SAMPLE_CSV_END");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn time_baseline(
    reader: &HdcStoreReader,
    batch_queries: &[BinaryHV],
    batch_index: usize,
    round: usize,
    batch_size: usize,
    vectors: usize,
    top_k: usize,
    reference: &[Vec<(u64, f32)>],
    reference_range: std::ops::Range<usize>,
) -> Result<RawSample, Box<dyn Error>> {
    let start = Instant::now();
    let result: Vec<Vec<(u64, f32)>> = batch_queries
        .iter()
        .map(|query| reader.scan_similar(query, top_k))
        .collect();
    let elapsed_ns = start.elapsed().as_nanos();
    validate_results(&result, &reference[reference_range])?;
    let checksum = result_checksum(&result);
    black_box(checksum);
    Ok(RawSample {
        algorithm: "repeated_single_query",
        batch_size,
        round,
        batch_index,
        query_count: batch_queries.len(),
        elapsed_ns,
        vectors,
        top_k,
        checksum,
    })
}

#[allow(clippy::too_many_arguments)]
fn time_fused(
    reader: &HdcStoreReader,
    batch_queries: &[BinaryHV],
    batch_index: usize,
    round: usize,
    batch_size: usize,
    vectors: usize,
    top_k: usize,
    reference: &[Vec<(u64, f32)>],
    reference_range: std::ops::Range<usize>,
) -> Result<RawSample, Box<dyn Error>> {
    let start = Instant::now();
    let result = fused_batch_scan(reader, batch_queries, top_k)?;
    let elapsed_ns = start.elapsed().as_nanos();
    validate_results(&result, &reference[reference_range])?;
    let checksum = result_checksum(&result);
    black_box(checksum);
    Ok(RawSample {
        algorithm: "corpus_major_batch",
        batch_size,
        round,
        batch_index,
        query_count: batch_queries.len(),
        elapsed_ns,
        vectors,
        top_k,
        checksum,
    })
}

fn fused_batch_scan(
    reader: &HdcStoreReader,
    queries: &[BinaryHV],
    top_k: usize,
) -> Result<Vec<Vec<(u64, f32)>>, Box<dyn Error>> {
    if top_k == 0 {
        return Ok(vec![Vec::new(); queries.len()]);
    }
    let live = usize::try_from(reader.live_count())?;
    let required_tuples = live.checked_mul(queries.len()).ok_or("batch score count overflow")?;
    if required_tuples > MAX_SCORE_TUPLES {
        return Err(format!(
            "batch score budget exceeded: {required_tuples} tuples > {MAX_SCORE_TUPLES}; reduce --vectors or --batch-sizes"
        ).into());
    }
    let mut scores: Vec<Vec<(u64, f32)>> = (0..queries.len())
        .map(|_| Vec::with_capacity(live))
        .collect();

    // Corpus-major order holds each mapped BinaryHV as the inner batch loop
    // consumes the current vector against several queries before advancing.
    for (id, hv) in reader.iter() {
        for (query, result) in queries.iter().zip(scores.iter_mut()) {
            result.push((id, query.similarity(hv)));
        }
    }
    for result in &mut scores {
        sort_exact(result);
        result.truncate(top_k);
    }
    Ok(scores)
}

fn validate_results(
    observed: &[Vec<(u64, f32)>],
    expected: &[Vec<(u64, f32)>],
) -> Result<(), Box<dyn Error>> {
    if observed.len() != expected.len() {
        return Err(format!(
            "result batch length mismatch: observed {}, expected {}",
            observed.len(), expected.len()
        ).into());
    }
    for (query_index, (got, want)) in observed.iter().zip(expected).enumerate() {
        if got.len() != want.len() {
            return Err(format!(
                "top-k length mismatch for batch query {query_index}: {} vs {}",
                got.len(), want.len()
            ).into());
        }
        for (rank, (got_entry, want_entry)) in got.iter().zip(want).enumerate() {
            if got_entry.0 != want_entry.0 || got_entry.1.to_bits() != want_entry.1.to_bits() {
                return Err(format!(
                    "exact top-k mismatch at query {query_index}, rank {rank}: observed {:?}, expected {:?}",
                    got_entry, want_entry
                ).into());
            }
        }
    }
    Ok(())
}

fn sort_exact(results: &mut [(u64, f32)]) {
    results.sort_unstable_by(|left, right| {
        right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0))
    });
}

fn result_checksum(results: &[Vec<(u64, f32)>]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for result in results {
        for (id, score) in result {
            for byte in id.to_le_bytes().into_iter().chain(score.to_bits().to_le_bytes()) {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    hash
}

fn print_summary(samples: &[RawSample]) {
    let mut configurations = Vec::new();
    for sample in samples {
        let key = (sample.algorithm, sample.batch_size);
        if !configurations.contains(&key) {
            configurations.push(key);
        }
    }
    println!("algorithm,batch_size,samples,queries,total_ns,queries_per_s,p50_batch_ns,p95_batch_ns,p99_batch_ns");
    for (algorithm, batch_size) in configurations {
        let rows: Vec<&RawSample> = samples
            .iter()
            .filter(|row| row.algorithm == algorithm && row.batch_size == batch_size)
            .collect();
        let mut times: Vec<u128> = rows.iter().map(|row| row.elapsed_ns).collect();
        times.sort_unstable();
        let total_ns: u128 = rows.iter().map(|row| row.elapsed_ns).sum();
        let total_queries: usize = rows.iter().map(|row| row.query_count).sum();
        let rate = if total_ns == 0 {
            0.0
        } else {
            total_queries as f64 * 1_000_000_000.0 / total_ns as f64
        };
        println!(
            "{algorithm},{batch_size},{},{total_queries},{total_ns},{rate:.3},{},{},{}",
            rows.len(),
            percentile(&times, 50),
            percentile(&times, 95),
            percentile(&times, 99),
        );
    }
}

fn render_csv(samples: &[RawSample]) -> String {
    let mut output = String::from(
        "algorithm,batch_size,round,batch_index,query_count,elapsed_ns,vectors,top_k,topk_checksum_fnv1a64\n",
    );
    for row in samples {
        output.push_str(&format!(
            "{},{},{},{},{},{},{},{},{:016x}\n",
            row.algorithm,
            row.batch_size,
            row.round,
            row.batch_index,
            row.query_count,
            row.elapsed_ns,
            row.vectors,
            row.top_k,
            row.checksum,
        ));
    }
    output
}

fn percentile(sorted: &[u128], percentile: usize) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (sorted.len() * percentile).div_ceil(100).max(1);
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn validate_config(config: &Config) -> Result<(), Box<dyn Error>> {
    if config.vectors == 0 || config.vectors > MAX_VECTORS {
        return Err(format!("--vectors must be in 1..={MAX_VECTORS}").into());
    }
    if config.queries == 0 || config.queries > MAX_QUERIES {
        return Err(format!("--queries must be in 1..={MAX_QUERIES}").into());
    }
    if config.top_k > config.vectors {
        return Err("--top-k cannot exceed --vectors".into());
    }
    if config.rounds == 0 || config.rounds > 10_000 {
        return Err("--rounds must be in 1..=10000".into());
    }
    if config.write_batch_size == 0 || config.write_batch_size > config.vectors {
        return Err("--write-batch-size must be in 1..=vectors".into());
    }
    if config.batch_sizes.is_empty() {
        return Err("--batch-sizes must include at least one size".into());
    }
    for &size in &config.batch_sizes {
        if size == 0 || size > MAX_BATCH_SIZE {
            return Err(format!("batch size must be in 1..={MAX_BATCH_SIZE}, found {size}").into());
        }
        let tuples = config.vectors.checked_mul(size).ok_or("score tuple count overflow")?;
        if tuples > MAX_SCORE_TUPLES {
            return Err(format!(
                "vectors * batch size = {tuples} exceeds score budget {MAX_SCORE_TUPLES}; reduce --vectors or --batch-sizes"
            ).into());
        }
    }
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn Error>> {
    let mut config = Config::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--vectors" => config.vectors = parse_value("--vectors", args.next())?,
            "--queries" => config.queries = parse_value("--queries", args.next())?,
            "--top-k" => config.top_k = parse_value("--top-k", args.next())?,
            "--rounds" => config.rounds = parse_value("--rounds", args.next())?,
            "--write-batch-size" => config.write_batch_size = parse_value("--write-batch-size", args.next())?,
            "--seed" => config.seed = parse_value("--seed", args.next())?,
            "--batch-sizes" => {
                let raw = args.next().ok_or("--batch-sizes requires a comma-separated list")?;
                let mut sizes = Vec::new();
                for value in raw.split(',') {
                    sizes.push(value.parse::<usize>()?);
                }
                sizes.sort_unstable();
                sizes.dedup();
                config.batch_sizes = sizes;
            }
            "--csv" => config.csv_path = Some(PathBuf::from(args.next().ok_or("--csv requires a path")?)),
            "--help" | "-h" => {
                println!(
                    "Usage: batched_exact_scan_bench [--vectors N] [--queries N] [--top-k K] [--batch-sizes 1,2,4,8] [--rounds N] [--write-batch-size N] [--seed N] [--csv PATH]\nDefaults: vectors=4096, queries=32, top-k=10, batch-sizes=1,2,4,8,16, rounds=5, seed=7"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(config)
}

fn parse_value<T>(flag: &str, value: Option<String>) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    Ok(value.ok_or_else(|| format!("{flag} requires a value"))?.parse()?)
}
