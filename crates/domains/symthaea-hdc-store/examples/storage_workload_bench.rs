// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Reproducible, synthetic workload trace for symthaea-hdc-store.
//!
//! Example:
//! cargo run -p symthaea-hdc-store --release --example storage_workload_bench -- \
//!   --vectors 1024 --queries 32 --batch-size 128 --seed 7 --csv /tmp/hdc-store.csv
//!
//! This is a workload benchmark, not a device benchmark: timings include OS,
//! filesystem, mmap, index and runtime behavior. The program cannot flush the OS
//! page cache or infer physical-media bytes/energy. Run on a named host/storage
//! profile and preserve the exact executable/source identity with results.

use std::env;
use std::error::Error;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use symthaea_core::hdc::BinaryHV;
use symthaea_hdc_store::{
    ApproximateSearchOptions, HdcStore, HdcStoreReader, StoreConfig, WriteBatch,
};

const BYTES_PER_BINARY_HV: u64 = 2_048;
const MAX_VECTORS: usize = 1_048_576;

#[derive(Debug)]
struct Config {
    vectors: usize,
    queries: usize,
    batch_size: usize,
    seed: u64,
    csv_path: Option<PathBuf>,
    store_path: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vectors: 1_024,
            queries: 32,
            batch_size: 128,
            seed: 7,
            csv_path: None,
            store_path: None,
        }
    }
}

#[derive(Debug)]
struct RawSample {
    profile: &'static str,
    sample: usize,
    elapsed_ns: u128,
    operations: u64,
    logical_bytes_estimate: u64,
    successes: u64,
    recall_hits: u64,
    recall_total: u64,
    exact_fallbacks: u64,
    candidates_examined: u64,
    checksum_sink: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("storage_workload_bench: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    if config.vectors == 0 || config.vectors > MAX_VECTORS {
        return Err(format!("--vectors must be in 1..={MAX_VECTORS}").into());
    }
    if config.queries == 0 {
        return Err("--queries must be at least 1".into());
    }
    if config.batch_size == 0 {
        return Err("--batch-size must be at least 1".into());
    }

    let mut samples = Vec::new();
    let generation_start = Instant::now();
    let mut corpus = Vec::with_capacity(config.vectors);
    for index in 0..config.vectors {
        let id_seed = config.seed.wrapping_add(index as u64);
        corpus.push(BinaryHV::random(id_seed));
    }
    let generation_elapsed = generation_start.elapsed();
    let corpus_fingerprint = fingerprint_corpus(&corpus);
    let corpus_bytes = u64::try_from(config.vectors)?
        .checked_mul(BYTES_PER_BINARY_HV)
        .ok_or("corpus byte count overflow")?;
    samples.push(sample(
        "corpus_generate",
        0,
        generation_elapsed.as_nanos(),
        config.vectors as u64,
        0,
        config.vectors as u64,
        0,
        0,
        0,
        0,
        corpus_fingerprint,
    ));

    // Queries are drawn from the fixed corpus, so each exact query has a known
    // in-store exact match. The seed and chosen IDs are deterministic.
    let mut rng = config.seed ^ 0x9E37_79B9_7F4A_7C15;
    let query_plan: Vec<(u64, BinaryHV)> = (0..config.queries)
        .map(|_| {
            let index = (next_rng(&mut rng) % config.vectors as u64) as usize;
            ((index as u64) + 1, corpus[index])
        })
        .collect();

    let tempdir = if config.store_path.is_none() {
        Some(tempfile::tempdir()?)
    } else {
        None
    };
    let store_path = config.store_path.clone().unwrap_or_else(|| {
        tempdir
            .as_ref()
            .expect("temporary directory exists for the default store path")
            .path()
            .join("workload.hdc")
    });
    if store_path.exists() {
        return Err(format!(
            "store path already exists (the benchmark never overwrites data): {}",
            store_path.display()
        ).into());
    }
    if let Some(parent) = store_path.parent() {
        if parent != std::path::Path::new("") && !parent.exists() {
            return Err(format!("store parent directory does not exist: {}", parent.display()).into());
        }
    }

    if config.csv_path.as_ref().is_some_and(|csv| {
        csv == &store_path || csv.with_extension("meta.txt") == store_path
    }) {
        return Err("--csv and --store must name different paths".into());
    }
    if let Some(csv) = config.csv_path.as_ref() {
        let meta_path = csv.with_extension("meta.txt");
        if csv.exists() || meta_path.exists() {
            return Err(format!(
                "benchmark output already exists; choose new --csv and metadata paths: {}",
                csv.display()
            ).into());
        }
    }

    let store_config = StoreConfig {
        initial_capacity: config.vectors as u64,
        ..StoreConfig::default()
    };
    let create_start = Instant::now();
    let mut store = HdcStore::create(&store_path, store_config)?;
    let create_elapsed = create_start.elapsed();
    samples.push(sample(
        "store_create",
        0,
        create_elapsed.as_nanos(),
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
    ));

    // Each WriteBatch publishes one generation. Time each commit separately
    // rather than hiding durable-append cost inside corpus generation.
    for (batch_index, chunk) in corpus.chunks(config.batch_size).enumerate() {
        let first_index = batch_index * config.batch_size;
        let mut batch = WriteBatch::new();
        for (offset, hv) in chunk.iter().enumerate() {
            let stable_id = (first_index + offset) as u64 + 1;
            batch.push_append(stable_id, *hv);
        }
        let op_count = batch.len() as u64;
        let batch_start = Instant::now();
        let commit = store.apply_batch(batch)?;
        let elapsed = batch_start.elapsed();
        if commit.appended != op_count || commit.deleted != 0 {
            return Err(format!("unexpected batch commit report: {commit:?}").into());
        }
        samples.push(sample(
            "batch_append_commit",
            batch_index,
            elapsed.as_nanos(),
            op_count,
            op_count.saturating_mul(BYTES_PER_BINARY_HV),
            op_count,
            0,
            0,
            0,
            0,
            commit.generation_after,
        ));
    }

    let sync_start = Instant::now();
    store.sync_all()?;
    let sync_elapsed = sync_start.elapsed();
    samples.push(sample(
        "sync_all",
        0,
        sync_elapsed.as_nanos(),
        1,
        corpus_bytes,
        1,
        0,
        0,
        0,
        0,
        store.header_generation(),
    ));

    let content_checksum = store.content_checksum();
    if store.live_count() != config.vectors as u64 {
        return Err(format!(
            "live-count mismatch: expected {}, got {}",
            config.vectors,
            store.live_count()
        )
        .into());
    }

    let mutable_open_start = Instant::now();
    drop(store);
    let mut store = HdcStore::open(&store_path)?;
    let mutable_open_elapsed = mutable_open_start.elapsed();
    samples.push(sample(
        "mutable_reopen",
        0,
        mutable_open_elapsed.as_nanos(),
        1,
        corpus_bytes,
        1,
        0,
        0,
        0,
        0,
        store.header_generation(),
    ));

    // Compare exact ground truth and the explicitly approximate LSH API over
    // identical queries; never infer ANN quality from candidate count alone.
    let mut exact_top_ids = Vec::with_capacity(config.queries);
    for (sample_index, (query_id, query)) in query_plan.iter().enumerate() {
        let start = Instant::now();
        let neighbors = store.scan_similar(query, 1);
        let elapsed = start.elapsed();
        let Some((top_id, score)) = neighbors.first().copied() else {
            return Err(format!("exact scan returned no neighbor for query ID {query_id}").into());
        };
        if score != 1.0 {
            return Err(format!(
                "known in-store query {query_id} did not produce exact similarity 1.0 (found {score})"
            ).into());
        }
        exact_top_ids.push(top_id);
        samples.push(sample(
            "exact_scan_top1",
            sample_index,
            elapsed.as_nanos(),
            1,
            corpus_bytes,
            1,
            0,
            0,
            0,
            0,
            top_id,
        ));
    }

    for (sample_index, ((query_id, query), exact_id)) in
        query_plan.iter().zip(exact_top_ids.iter()).enumerate()
    {
        let start = Instant::now();
        let outcome = store.scan_similar_approx(
            query,
            1,
            ApproximateSearchOptions::default(),
        );
        let elapsed = start.elapsed();
        let approximate_id = outcome.neighbors.first().map(|entry| entry.0);
        let recall_hit = if approximate_id == Some(*exact_id) { 1_u64 } else { 0 };
        let success = if approximate_id.is_some() { 1_u64 } else { 0 };
        if outcome.total_live != config.vectors {
            return Err(format!(
                "approximate search reported {} live vectors for query {query_id}, expected {}",
                outcome.total_live, config.vectors
            ).into());
        }
        samples.push(sample(
            "approx_scan_top1",
            sample_index,
            elapsed.as_nanos(),
            1,
            (outcome.examined as u64).saturating_mul(BYTES_PER_BINARY_HV),
            success,
            recall_hit,
            1,
            if outcome.fell_back_to_exact { 1 } else { 0 },
            outcome.examined as u64,
            approximate_id.unwrap_or(0),
        ));
    }

    // Compare the mutable in-memory ID hash map with the read-only reader's
    // sorted-ID mapping below. This separates lookup policy from vector I/O.
    let mut mutable_point_rng = config.seed ^ 0xA076_1D64_78BD_642F;
    let mut mutable_point_checksum = 0_u64;
    for sample_index in 0..config.queries {
        let index = (next_rng(&mut mutable_point_rng) % config.vectors as u64) as usize;
        let id = index as u64 + 1;
        let start = Instant::now();
        let hv = store.get(id).ok_or_else(|| format!("missing mutable point-read ID {id}"))?;
        let observed_byte = black_box(hv.0[index % BYTES_PER_BINARY_HV as usize]);
        let elapsed = start.elapsed();
        mutable_point_checksum = mutable_point_checksum
            .rotate_left(7)
            .wrapping_add(id)
            .wrapping_add(observed_byte as u64);
        samples.push(sample(
            "mutable_random_point_read",
            sample_index,
            elapsed.as_nanos(),
            1,
            BYTES_PER_BINARY_HV,
            1,
            0,
            0,
            0,
            0,
            mutable_point_checksum,
        ));
    }

    let store_file_bytes = fs::metadata(&store_path)?.len();
    let generation = store.header_generation();
    let live_count = store.live_count();
    let tombstone_count = store.tombstone_count();
    drop(store);

    let reader_open_start = Instant::now();
    let reader = HdcStoreReader::open(&store_path)?;
    let reader_open_elapsed = reader_open_start.elapsed();
    samples.push(sample(
        "readonly_reader_open",
        0,
        reader_open_elapsed.as_nanos(),
        1,
        corpus_bytes,
        1,
        0,
        0,
        0,
        0,
        reader.generation(),
    ));

    // Pinned random ID reads. The logical-byte estimate is requested vector
    // payload only, not physical device bytes or pages fetched by the OS.
    let mut point_rng = config.seed ^ 0xD1B5_4A32_D192_ED03;
    let mut point_checksum = 0_u64;
    for sample_index in 0..config.queries {
        let index = (next_rng(&mut point_rng) % config.vectors as u64) as usize;
        let id = index as u64 + 1;
        let start = Instant::now();
        let hv = reader.get(id).ok_or_else(|| format!("missing point-read ID {id}"))?;
        let observed_byte = black_box(hv.0[index % BYTES_PER_BINARY_HV as usize]);
        let elapsed = start.elapsed();
        point_checksum = point_checksum
            .rotate_left(5)
            .wrapping_add(id)
            .wrapping_add(observed_byte as u64);
        samples.push(sample(
            "random_point_read",
            sample_index,
            elapsed.as_nanos(),
            1,
            BYTES_PER_BINARY_HV,
            1,
            0,
            0,
            0,
            0,
            point_checksum,
        ));
    }

    // Sequential whole-store iteration catches a distinct I/O shape from
    // random lookup. Cap repeats to keep large corpora practical.
    let sequential_repeats = config.queries.min(10).max(1);
    let mut sequential_checksum = 0_u64;
    for sample_index in 0..sequential_repeats {
        let start = Instant::now();
        let mut seen = 0_u64;
        for (id, hv) in reader.iter() {
            sequential_checksum = sequential_checksum
                .rotate_left(3)
                .wrapping_add(id)
                .wrapping_add(hv.0[(id as usize) % BYTES_PER_BINARY_HV] as u64);
            seen += 1;
        }
        let elapsed = start.elapsed();
        if seen != config.vectors as u64 {
            return Err(format!(
                "sequential scan found {seen} entries, expected {}",
                config.vectors
            )
            .into());
        }
        samples.push(sample(
            "sequential_full_scan",
            sample_index,
            elapsed.as_nanos(),
            seen,
            corpus_bytes,
            seen,
            0,
            0,
            0,
            0,
            sequential_checksum,
        ));
    }
    drop(reader);

    let recovery_open_start = Instant::now();
    let (recovered, recovery_report) = HdcStore::open_recovering(&store_path)?;
    let recovery_elapsed = recovery_open_start.elapsed();
    samples.push(sample(
        "clean_recovery_open",
        0,
        recovery_elapsed.as_nanos(),
        1,
        corpus_bytes,
        if !recovery_report.changed_store() { 1 } else { 0 },
        0,
        0,
        0,
        0,
        recovery_report.final_generation,
    ));
    if recovered.live_count() != config.vectors as u64 {
        return Err(format!(
            "recovery-open live count mismatch: expected {}, got {}",
            config.vectors,
            recovered.live_count()
        )
        .into());
    }
    drop(recovered);

    let approximate_samples: Vec<&RawSample> = samples
        .iter()
        .filter(|row| row.profile == "approx_scan_top1")
        .collect();
    let approx_hits: u64 = approximate_samples.iter().map(|row| row.recall_hits).sum();
    let approx_total: u64 = approximate_samples.iter().map(|row| row.recall_total).sum();
    let approx_fallbacks: u64 = approximate_samples.iter().map(|row| row.exact_fallbacks).sum();

    println!("HDC store synthetic workload benchmark (no hardware/energy claim)");
    println!("vectors={}, queries={}, batch_size={}, seed={}", config.vectors, config.queries, config.batch_size, config.seed);
    println!("host_os={}, host_arch={}", env::consts::OS, env::consts::ARCH);
    println!("source_sha={}", env::var("SYMTHAEA_SOURCE_SHA").unwrap_or_else(|_| "not-injected".to_string()));
    println!("rustc_version={}", env::var("RUSTC_VERSION").unwrap_or_else(|_| "not-injected".to_string()));
    println!("host_profile={}", env::var("HOST_PROFILE").unwrap_or_else(|_| "not-injected".to_string()));
    println!("storage_profile={}", env::var("STORAGE_PROFILE").unwrap_or_else(|_| "not-injected".to_string()));
    println!("store_path={}", store_path.display());
    println!("lsh_bands={}, lsh_rows={}", store_config.lsh_bands, store_config.lsh_rows);
    println!("corpus_fingerprint_fnv1a64_noncryptographic={corpus_fingerprint:016x}");
    println!("logical_vector_payload_bytes={corpus_bytes}");
    println!("store_file_bytes_including_preallocation_and_metadata={store_file_bytes}");
    println!("content_checksum={content_checksum:?}");
    println!("generation={generation}, live_count={live_count}, tombstones={tombstone_count}");
    println!(
        "approx_top1_recall={}/{} ({:.4}), exact_fallbacks={}/{}",
        approx_hits,
        approx_total,
        if approx_total == 0 { 0.0 } else { approx_hits as f64 / approx_total as f64 },
        approx_fallbacks,
        approx_samples.len()
    );
    println!("cold OS cache, physical media bytes, write amplification, and energy are NOT controlled or measured.");
    print_summaries(&samples);

    let csv = render_csv(&samples);
    if let Some(path) = config.csv_path {
        fs::write(&path, csv.as_bytes())?;
        let meta_path = path.with_extension("meta.txt");
        let metadata = format!(
            "benchmark=storage_workload_bench\nsource_sha={}\nrustc_version={}\nhost_os={}\nhost_arch={}\nhost_profile={}\nstorage_profile={}\nstore_path={}\nvectors={}\nqueries={}\nbatch_size={}\nseed={}\nlsh_bands={}\nlsh_rows={}\ncorpus_fingerprint_fnv1a64_noncryptographic={:016x}\nlogical_vector_payload_bytes={}\nstore_file_bytes_including_preallocation_and_metadata={}\ncontent_checksum={:?}\ngeneration={}\nlive_count={}\ntombstones={}\napprox_top1_recall={}/{}\nexact_fallbacks={}/{}\nphysical_media_bytes=not_measured\nenergy=not_measured\nos_page_cache_coldness=not_controlled\n",
            env::var("SYMTHAEA_SOURCE_SHA").unwrap_or_else(|_| "not-injected".to_string()),
            env::var("RUSTC_VERSION").unwrap_or_else(|_| "not-injected".to_string()),
            env::consts::OS,
            env::consts::ARCH,
            env::var("HOST_PROFILE").unwrap_or_else(|_| "not-injected".to_string()),
            env::var("STORAGE_PROFILE").unwrap_or_else(|_| "not-injected".to_string()),
            store_path.display(),
            config.vectors,
            config.queries,
            config.batch_size,
            config.seed,
            store_config.lsh_bands,
            store_config.lsh_rows,
            corpus_fingerprint,
            corpus_bytes,
            store_file_bytes,
            content_checksum,
            generation,
            live_count,
            tombstone_count,
            approx_hits,
            approx_total,
            approx_fallbacks,
            approximate_samples.len(),
        );
        fs::write(&meta_path, metadata)?;
        eprintln!("raw_samples_csv={}", path.display());
        eprintln!("run_metadata={}", meta_path.display());
    }
    println!("RAW_SAMPLE_CSV_BEGIN");
    print!("{csv}");
    println!("RAW_SAMPLE_CSV_END");
    Ok(())
}

fn parse_args() -> Result<Config, Box<dyn Error>> {
    let mut config = Config::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--vectors" => config.vectors = parse_value("--vectors", args.next())?,
            "--queries" => config.queries = parse_value("--queries", args.next())?,
            "--batch-size" => config.batch_size = parse_value("--batch-size", args.next())?,
            "--seed" => config.seed = parse_value("--seed", args.next())?,
            "--csv" => {
                config.csv_path = Some(PathBuf::from(
                    args.next().ok_or("--csv requires a path")?,
                ))
            }
            "--store" => {
                config.store_path = Some(PathBuf::from(
                    args.next().ok_or("--store requires a path")?,
                ))
            }
            "--help" | "-h" => {
                println!(
                    "Usage: storage_workload_bench [--vectors N] [--queries N] [--batch-size N] [--seed N] [--csv PATH] [--store PATH]\nDefaults: vectors=1024, queries=32, batch-size=128, seed=7"
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

fn sample(
    profile: &'static str,
    sample: usize,
    elapsed_ns: u128,
    operations: u64,
    logical_bytes_estimate: u64,
    successes: u64,
    recall_hits: u64,
    recall_total: u64,
    exact_fallbacks: u64,
    candidates_examined: u64,
    checksum_sink: u64,
) -> RawSample {
    RawSample {
        profile,
        sample,
        elapsed_ns,
        operations,
        logical_bytes_estimate,
        successes,
        recall_hits,
        recall_total,
        exact_fallbacks,
        candidates_examined,
        checksum_sink,
    }
}

fn next_rng(state: &mut u64) -> u64 {
    // xorshift64*: fixed deterministic sequence; not cryptographic.
    let mut value = *state;
    value ^= value << 13;
    value ^= value >> 7;
    value ^= value << 17;
    if value == 0 {
        value = 0xA076_1D64_78BD_642F;
    }
    *state = value;
    value.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

fn fingerprint_corpus(corpus: &[BinaryHV]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for hv in corpus {
        for byte in hv.0 {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn print_summaries(samples: &[RawSample]) {
    let mut profiles: Vec<&str> = Vec::new();
    for row in samples {
        if !profiles.contains(&row.profile) {
            profiles.push(row.profile);
        }
    }
    println!();
    println!("profile,samples,ops,total_ns,p50_ns,p95_ns,p99_ns,ops_per_s,logical_bytes_estimate,recall,exact_fallbacks,candidates_examined");
    for profile in profiles {
        let rows: Vec<&RawSample> = samples.iter().filter(|row| row.profile == profile).collect();
        let mut times: Vec<u128> = rows.iter().map(|row| row.elapsed_ns).collect();
        times.sort_unstable();
        let total_ns: u128 = rows.iter().map(|row| row.elapsed_ns).sum();
        let operations: u64 = rows.iter().map(|row| row.operations).sum();
        let logical_bytes: u64 = rows.iter().map(|row| row.logical_bytes_estimate).sum();
        let recall_hits: u64 = rows.iter().map(|row| row.recall_hits).sum();
        let recall_total: u64 = rows.iter().map(|row| row.recall_total).sum();
        let fallbacks: u64 = rows.iter().map(|row| row.exact_fallbacks).sum();
        let examined: u64 = rows.iter().map(|row| row.candidates_examined).sum();
        let throughput = if total_ns == 0 {
            0.0
        } else {
            operations as f64 * 1_000_000_000.0 / total_ns as f64
        };
        println!(
            "{profile},{},{operations},{total_ns},{},{},{},{throughput:.3},{logical_bytes},{}/{},{fallbacks},{examined}",
            rows.len(),
            percentile(&times, 50),
            percentile(&times, 95),
            percentile(&times, 99),
            recall_hits,
            recall_total,
        );
    }
}

fn percentile(sorted: &[u128], percentile: usize) -> u128 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = (sorted.len() * percentile).div_ceil(100).max(1);
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn render_csv(samples: &[RawSample]) -> String {
    let mut output = String::from(
        "profile,sample,elapsed_ns,operations,logical_bytes_estimate_not_physical_io,successes,top1_recall_hits,top1_recall_total,exact_fallbacks,candidates_examined,checksum_sink\n",
    );
    for row in samples {
        output.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{}\n",
            row.profile,
            row.sample,
            row.elapsed_ns,
            row.operations,
            row.logical_bytes_estimate,
            row.successes,
            row.recall_hits,
            row.recall_total,
            row.exact_fallbacks,
            row.candidates_examined,
            row.checksum_sink,
        ));
    }
    output
}
