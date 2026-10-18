// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Exact top-k benchmark: canonical full sort, bounded heap, and chunked
//! Hamming-distance early abandonment. All candidates are checked against the
//! public canonical search API before their timings are retained.
//!
//! Example:
//! cargo run -p symthaea-hdc-store --release --example exact_topk_pruning_bench -- \
//!   --vectors 16384 --queries 16 --top-ks 1,10,100 --corpora random,clustered,mixed \
//!   --chunk-sizes 32,64,128,256 --rounds 30 --seed 7 --csv /tmp/exact-topk.csv
//!
//! A byte-work counter counts corpus payload bytes processed by the algorithm;
//! it is not a physical-device-I/O counter.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::error::Error;
use std::env;
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use symthaea_core::hdc::BinaryHV;
use symthaea_hdc_store::{HdcStore, HdcStoreReader, StoreConfig, WriteBatch};

const HV_BYTES: usize = 2_048;
const MAX_VECTORS: usize = 131_072;
const MAX_QUERIES: usize = 256;
const MAX_ROUNDS: usize = 10_000;
const MAX_CHUNKS: usize = 8;
const MAX_TOP_K: usize = 4_096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CorpusKind {
    Random,
    Clustered,
    Mixed,
}
impl CorpusKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Clustered => "clustered",
            Self::Mixed => "mixed",
        }
    }
}
#[derive(Clone, Copy, Debug)]
enum Algorithm {
    FullSort,
    BoundedHeap,
    EarlyAbandon { chunk_bytes: usize },
}
impl Algorithm {
    fn as_str(self) -> String {
        match self {
            Self::FullSort => "canonical_full_sort".to_string(),
            Self::BoundedHeap => "bounded_heap_full_distance".to_string(),
            Self::EarlyAbandon { chunk_bytes } => format!("early_abandon_{chunk_bytes}B"),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Candidate {
    distance: u32,
    id: u64,
}
// BinaryHeap is a max-heap; the lexicographically largest (distance, ID) is
// the worst retained candidate and is therefore at the root.
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.cmp(&other.distance).then_with(|| self.id.cmp(&other.id))
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
#[derive(Debug)]
struct Config {
    vectors: usize,
    queries: usize,
    top_ks: Vec<usize>,
    corpora: Vec<CorpusKind>,
    chunk_sizes: Vec<usize>,
    rounds: usize,
    write_batch_size: usize,
    seed: u64,
    csv_path: Option<PathBuf>,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            vectors: 4_096,
            queries: 8,
            top_ks: vec![1, 10, 100],
            corpora: vec![CorpusKind::Random, CorpusKind::Clustered, CorpusKind::Mixed],
            chunk_sizes: vec![32, 64, 128, 256],
            rounds: 5,
            write_batch_size: 128,
            seed: 7,
            csv_path: None,
        }
    }
}
#[derive(Debug)]
struct RawSample {
    corpus: &'static str,
    algorithm: String,
    top_k: usize,
    query_index: usize,
    round: usize,
    elapsed_ns: u128,
    vectors: usize,
    payload_bytes_processed: u64,
    vectors_pruned: u64,
    result_checksum: u64,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("exact_topk_pruning_bench: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let config = parse_args()?;
    validate_config(&config)?;
    if let Some(path) = config.csv_path.as_ref() {
        let metadata_path = path.with_extension("meta.txt");
        if path.exists() || metadata_path.exists() {
            return Err(format!("refusing to overwrite benchmark output or sidecar: {}", path.display()).into());
        }
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() && !parent.is_dir() {
                return Err(format!("CSV parent directory does not exist: {}", parent.display()).into());
            }
        }
    }

    let tempdir = tempfile::tempdir()?;
    let mut rows = Vec::new();
    for &corpus_kind in &config.corpora {
        let store_path = tempdir.path().join(format!("{}.hdc", corpus_kind.as_str()));
        let store_config = StoreConfig {
            initial_capacity: config.vectors as u64,
            ..StoreConfig::default()
        };
        let mut store = HdcStore::create(&store_path, store_config)?;
        let centers: [BinaryHV; 4] = std::array::from_fn(|i| {
            BinaryHV::random(config.seed.wrapping_add(0xC001_0000 + i as u64))
        });
        let build_start = Instant::now();
        for offset in (0..config.vectors).step_by(config.write_batch_size) {
            let count = config.write_batch_size.min(config.vectors - offset);
            let mut batch = WriteBatch::new();
            for local_index in 0..count {
                let zero_based = offset + local_index;
                let id = zero_based as u64 + 1;
                let vector = make_corpus_vector(corpus_kind, id, config.seed, &centers);
                batch.push_append(id, vector);
            }
            let commit = store.apply_batch(batch)?;
            if commit.appended != count as u64 || commit.deleted != 0 {
                return Err(format!("unexpected corpus commit report: {commit:?}").into());
            }
        }
        store.sync_all()?;
        let generation = store.header_generation();
        let content_checksum = store.content_checksum();
        if store.live_count() != config.vectors as u64 {
            return Err("corpus build live-count mismatch".into());
        }
        drop(store);
        let build_ns = build_start.elapsed().as_nanos();

        let open_start = Instant::now();
        let reader = HdcStoreReader::open(&store_path)?;
        let reader_open_ns = open_start.elapsed().as_nanos();
        let queries: Vec<BinaryHV> = (0..config.queries)
            .map(|index| make_query(corpus_kind, index, config.seed, &centers))
            .collect();

        println!(
            "corpus={},store_generation={},store_checksum={},build_ns={},reader_open_ns={},vectors={},queries={}",
            corpus_kind.as_str(), generation, content_checksum, build_ns, reader_open_ns,
            config.vectors, config.queries
        );

        for &top_k in &config.top_ks {
            // Canonical exact reference results are calculated before timed runs.
            let reference: Vec<Vec<(u64, f32)>> = queries
                .iter()
                .map(|query| reader.scan_similar(query, top_k))
                .collect();

            let mut algorithms = vec![Algorithm::FullSort, Algorithm::BoundedHeap];
            algorithms.extend(config.chunk_sizes.iter().copied().map(|chunk_bytes| {
                Algorithm::EarlyAbandon { chunk_bytes }
            }));

            for round in 0..config.rounds {
                for query_index in 0..queries.len() {
                    let query = &queries[query_index];
                    let expected = &reference[query_index];
                    // Alternate algorithm order to reduce systematic position bias.
                    let reverse = (round + query_index + top_k) % 2 == 1;
                    if reverse {
                        for &algorithm in algorithms.iter().rev() {
                            let sample = time_algorithm(
                                &reader, query, expected, corpus_kind, algorithm,
                                top_k, query_index, round, config.vectors
                            )?;
                            rows.push(sample);
                        }
                    } else {
                        for &algorithm in &algorithms {
                            let sample = time_algorithm(
                                &reader, query, expected, corpus_kind, algorithm,
                                top_k, query_index, round, config.vectors
                            )?;
                            rows.push(sample);
                        }
                    }
                }
            }
        }
    }

    println!("benchmark=exact_topk_pruning_bench");
    println!("host_os={},host_arch={}", env::consts::OS, env::consts::ARCH);
    println!("source_sha={}", env::var("SYMTHAEA_SOURCE_SHA").unwrap_or_else(|_| "not-injected".into()));
    println!("rustc_version={}", env::var("RUSTC_VERSION").unwrap_or_else(|_| "not-injected".into()));
    println!("host_profile={}", env::var("HOST_PROFILE").unwrap_or_else(|_| "not-injected".into()));
    println!("storage_profile={}", env::var("STORAGE_PROFILE").unwrap_or_else(|_| "not-injected".into()));
    println!("cache_state=not_controlled; canonical reference queries warm the mapped corpus before timed variants");
    println!("ranking_contract=exact integer Hamming distance, then ascending ID; output scores checked by exact f32 bits");
    println!("payload_bytes_processed=algorithmic corpus-payload byte-work estimate; not physical I/O");
    print_summary(&rows);
    let csv = render_csv(&rows);
    if let Some(path) = config.csv_path.as_ref() {
        fs::write(path, csv.as_bytes())?;
        let metadata_path = path.with_extension("meta.txt");
        let metadata = format!(
            "benchmark=exact_topk_pruning_bench\nsource_sha={}\nrustc_version={}\nhost_os={}\nhost_arch={}\nhost_profile={}\nstorage_profile={}\nvectors={}\nqueries={}\ntop_ks={:?}\ncorpora={:?}\nchunk_sizes={:?}\nrounds={}\nwrite_batch_size={}\nseed={}\ncache_state=not_controlled\npayload_bytes_processed=algorithmic_estimate_not_physical_io\nphysical_device_io=not_measured\nenergy=not_measured\n",
            env::var("SYMTHAEA_SOURCE_SHA").unwrap_or_else(|_| "not-injected".into()),
            env::var("RUSTC_VERSION").unwrap_or_else(|_| "not-injected".into()),
            env::consts::OS,
            env::consts::ARCH,
            env::var("HOST_PROFILE").unwrap_or_else(|_| "not-injected".into()),
            env::var("STORAGE_PROFILE").unwrap_or_else(|_| "not-injected".into()),
            config.vectors,
            config.queries,
            config.top_ks,
            config.corpora.iter().map(|c| c.as_str()).collect::<Vec<_>>(),
            config.chunk_sizes,
            config.rounds,
            config.write_batch_size,
            config.seed,
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

fn time_algorithm(
    reader: &HdcStoreReader,
    query: &BinaryHV,
    expected: &[(u64, f32)],
    corpus: CorpusKind,
    algorithm: Algorithm,
    top_k: usize,
    query_index: usize,
    round: usize,
    vector_count: usize,
) -> Result<RawSample, Box<dyn Error>> {
    let start = Instant::now();
    let (observed, payload_bytes_processed, vectors_pruned) = match algorithm {
        Algorithm::FullSort => (
            reader.scan_similar(query, top_k),
            if top_k == 0 {
                0
            } else {
                (vector_count as u64).saturating_mul(HV_BYTES as u64)
            },
            0,
        ),
        Algorithm::BoundedHeap => bounded_heap_search(reader, query, top_k)?,
        Algorithm::EarlyAbandon { chunk_bytes } => {
            early_abandon_search(reader, query, top_k, chunk_bytes)?
        }
    };
    let elapsed_ns = start.elapsed().as_nanos();
    validate_top_k(&observed, expected)?;
    let result_checksum = checksum_results(&observed);
    black_box(result_checksum);
    Ok(RawSample {
        corpus: corpus.as_str(),
        algorithm: algorithm.as_str(),
        top_k,
        query_index,
        round,
        elapsed_ns,
        vectors: vector_count,
        payload_bytes_processed,
        vectors_pruned,
        result_checksum,
    })
}

fn bounded_heap_search(
    reader: &HdcStoreReader,
    query: &BinaryHV,
    top_k: usize,
) -> Result<(Vec<(u64, f32)>, u64, u64), Box<dyn Error>> {
    if top_k == 0 {
        return Ok((Vec::new(), 0, 0));
    }
    let mut heap = BinaryHeap::with_capacity(top_k);
    let mut bytes_processed = 0_u64;
    for (id, hv) in reader.iter() {
        let distance = query.hamming_distance(hv);
        bytes_processed = bytes_processed.saturating_add(HV_BYTES as u64);
        retain_candidate(&mut heap, Candidate { distance, id }, top_k);
    }
    Ok((finish_heap(heap), bytes_processed, 0))
}

fn early_abandon_search(
    reader: &HdcStoreReader,
    query: &BinaryHV,
    top_k: usize,
    chunk_bytes: usize,
) -> Result<(Vec<(u64, f32)>, u64, u64), Box<dyn Error>> {
    if top_k == 0 {
        return Ok((Vec::new(), 0, 0));
    }
    if chunk_bytes == 0 || HV_BYTES % chunk_bytes != 0 {
        return Err(format!("chunk size must be a nonzero divisor of {HV_BYTES}, got {chunk_bytes}").into());
    }

    let mut heap = BinaryHeap::with_capacity(top_k);
    let mut bytes_processed = 0_u64;
    let mut vectors_pruned = 0_u64;
    let mut previous_id = None;

    for (id, hv) in reader.iter() {
        if let Some(previous) = previous_id {
            if id <= previous {
                return Err("early-abandon proof requires strictly ascending IDs from reader.iter()".into());
            }
        }
        previous_id = Some(id);
        let mut partial_distance = 0_u32;
        let mut abandoned = false;

        for offset in (0..HV_BYTES).step_by(chunk_bytes) {
            let end = offset + chunk_bytes;
            for index in offset..end {
                partial_distance += (query.0[index] ^ hv.0[index]).count_ones();
            }
            bytes_processed = bytes_processed.saturating_add(chunk_bytes as u64);

            if heap.len() == top_k {
                let worst = heap.peek().expect("heap has top_k candidates");
                // Partial Hamming distance can only increase. If already worse,
                // prune. On an equal-distance lower bound, prune only when this
                // ID cannot win the ascending-ID tie-break.
                if partial_distance > worst.distance
                    || (partial_distance == worst.distance && id >= worst.id)
                {
                    vectors_pruned = vectors_pruned.saturating_add(1);
                    abandoned = true;
                    break;
                }
            }
        }
        if !abandoned {
            retain_candidate(
                &mut heap,
                Candidate { distance: partial_distance, id },
                top_k,
            );
        }
    }
    Ok((finish_heap(heap), bytes_processed, vectors_pruned))
}

fn retain_candidate(heap: &mut BinaryHeap<Candidate>, candidate: Candidate, top_k: usize) {
    if heap.len() < top_k {
        heap.push(candidate);
    } else if heap.peek().is_some_and(|worst| candidate < *worst) {
        let _ = heap.pop();
        heap.push(candidate);
    }
}

fn finish_heap(mut heap: BinaryHeap<Candidate>) -> Vec<(u64, f32)> {
    let mut candidates = heap.drain().collect::<Vec<_>>();
    candidates.sort_unstable_by(|left, right| {
        left.distance.cmp(&right.distance).then_with(|| left.id.cmp(&right.id))
    });
    candidates
        .into_iter()
        .map(|candidate| {
            let matching_bits = BinaryHV::DIM as u32 - candidate.distance;
            let score = matching_bits as f32 / BinaryHV::DIM as f32;
            (candidate.id, score)
        })
        .collect()
}

fn validate_top_k(
    observed: &[(u64, f32)],
    expected: &[(u64, f32)],
) -> Result<(), Box<dyn Error>> {
    if observed.len() != expected.len() {
        return Err(format!("top-k length mismatch: {} vs {}", observed.len(), expected.len()).into());
    }
    for (rank, (got, want)) in observed.iter().zip(expected).enumerate() {
        if got.0 != want.0 || got.1.to_bits() != want.1.to_bits() {
            return Err(format!(
                "exact top-k mismatch at rank {rank}: observed {got:?}, expected {want:?}"
            ).into());
        }
    }
    Ok(())
}

fn make_corpus_vector(
    kind: CorpusKind,
    id: u64,
    seed: u64,
    centers: &[BinaryHV; 4],
) -> BinaryHV {
    match kind {
        CorpusKind::Random => BinaryHV::random(seed.wrapping_add(id)),
        CorpusKind::Clustered => {
            let cluster = (id.wrapping_sub(1) % 4) as usize;
            let radius = 16 + ((id as usize * 13) % 65);
            perturb(&centers[cluster], seed.wrapping_add(id), radius)
        }
        CorpusKind::Mixed => {
            if id % 4 == 0 {
                // The every-fourth-vector stream must rotate across all four
                // centers; (id - 1) % 4 would always select center 3 here.
                let cluster = (id / 4).wrapping_sub(1) as usize % centers.len();
                let radius = 16 + ((id as usize * 13) % 65);
                perturb(&centers[cluster], seed.wrapping_add(id), radius)
            } else {
                BinaryHV::random(seed.wrapping_add(id))
            }
        }
    }
}

fn make_query(kind: CorpusKind, index: usize, seed: u64, centers: &[BinaryHV; 4]) -> BinaryHV {
    match kind {
        CorpusKind::Random => BinaryHV::random(seed ^ 0xD1B5_4A32_D192_ED03_u64.wrapping_add(index as u64)),
        CorpusKind::Clustered | CorpusKind::Mixed => centers[index % centers.len()],
    }
}

fn perturb(center: &BinaryHV, seed: u64, flips: usize) -> BinaryHV {
    let mut result = *center;
    let dimension = BinaryHV::DIM as u64;
    // DIM=16,384; the odd stride traverses unique bit positions for the radii
    // used here, yielding reproducible, controlled-distance cluster members.
    for index in 0..flips {
        let bit = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((index as u64).wrapping_mul(73))
            % dimension;
        result.0[(bit / 8) as usize] ^= 1_u8 << (bit % 8);
    }
    result
}

fn checksum_results(results: &[(u64, f32)]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for (id, score) in results {
        for byte in id.to_le_bytes().into_iter().chain(score.to_bits().to_le_bytes()) {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

fn print_summary(rows: &[RawSample]) {
    let mut keys: Vec<(&str, usize, &str)> = Vec::new();
    for row in rows {
        let key = (row.corpus, row.top_k, row.algorithm.as_str());
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    println!("corpus,algorithm,top_k,samples,query_ops,total_ns,queries_per_s,p50_ns,p95_ns,p99_ns,payload_bytes_processed,vectors_pruned");
    for (corpus, top_k, algorithm) in keys {
        let selected: Vec<&RawSample> = rows.iter()
            .filter(|row| row.corpus == corpus && row.top_k == top_k && row.algorithm == algorithm)
            .collect();
        let mut times: Vec<u128> = selected.iter().map(|row| row.elapsed_ns).collect();
        times.sort_unstable();
        let total_ns: u128 = selected.iter().map(|row| row.elapsed_ns).sum();
        let n = selected.len() as u64;
        let rate = if total_ns == 0 { 0.0 } else { n as f64 * 1_000_000_000.0 / total_ns as f64 };
        let bytes: u64 = selected.iter().map(|row| row.payload_bytes_processed).sum();
        let pruned: u64 = selected.iter().map(|row| row.vectors_pruned).sum();
        println!(
            "{corpus},{algorithm},{top_k},{n},{n},{total_ns},{rate:.3},{},{},{},{bytes},{pruned}",
            percentile(&times, 50), percentile(&times, 95), percentile(&times, 99)
        );
    }
}

fn render_csv(rows: &[RawSample]) -> String {
    let mut output = String::from(
        "corpus,algorithm,top_k,query_index,round,elapsed_ns,vectors,payload_bytes_processed_not_physical_io,vectors_pruned,result_checksum_fnv1a64\n",
    );
    for row in rows {
        output.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{:016x}\n",
            row.corpus,
            row.algorithm,
            row.top_k,
            row.query_index,
            row.round,
            row.elapsed_ns,
            row.vectors,
            row.payload_bytes_processed,
            row.vectors_pruned,
            row.result_checksum,
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
    if config.rounds == 0 || config.rounds > MAX_ROUNDS {
        return Err(format!("--rounds must be in 1..={MAX_ROUNDS}").into());
    }
    if config.write_batch_size == 0 || config.write_batch_size > config.vectors {
        return Err("--write-batch-size must be in 1..=vectors".into());
    }
    if config.top_ks.is_empty() {
        return Err("--top-ks must have at least one entry".into());
    }
    for &k in &config.top_ks {
        if k > config.vectors || k > MAX_TOP_K {
            return Err(format!("top-k must be <= min(vectors, {MAX_TOP_K}), found {k}").into());
        }
    }
    if config.corpora.is_empty() {
        return Err("--corpora must have at least one entry".into());
    }
    if config.chunk_sizes.is_empty() || config.chunk_sizes.len() > MAX_CHUNKS {
        return Err(format!("--chunk-sizes must include 1..={MAX_CHUNKS} entries").into());
    }
    for &chunk in &config.chunk_sizes {
        if chunk == 0 || chunk > HV_BYTES || HV_BYTES % chunk != 0 {
            return Err(format!("each chunk size must be a nonzero divisor of {HV_BYTES}, found {chunk}").into());
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
            "--rounds" => config.rounds = parse_value("--rounds", args.next())?,
            "--write-batch-size" => config.write_batch_size = parse_value("--write-batch-size", args.next())?,
            "--seed" => config.seed = parse_value("--seed", args.next())?,
            "--top-ks" => {
                config.top_ks = parse_list("--top-ks", args.next())?;
                config.top_ks.sort_unstable();
                config.top_ks.dedup();
            }
            "--chunk-sizes" => {
                config.chunk_sizes = parse_list("--chunk-sizes", args.next())?;
                config.chunk_sizes.sort_unstable();
                config.chunk_sizes.dedup();
            }
            "--corpora" => {
                let raw = args.next().ok_or("--corpora requires a comma-separated list")?;
                config.corpora.clear();
                for part in raw.split(',') {
                    let kind = match part {
                        "random" => CorpusKind::Random,
                        "clustered" => CorpusKind::Clustered,
                        "mixed" => CorpusKind::Mixed,
                        other => return Err(format!("unknown corpus kind: {other}").into()),
                    };
                    if !config.corpora.contains(&kind) { config.corpora.push(kind); }
                }
            }
            "--csv" => config.csv_path = Some(PathBuf::from(args.next().ok_or("--csv requires a path")?)),
            "--help" | "-h" => {
                println!(
                    "Usage: exact_topk_pruning_bench [--vectors N] [--queries N] [--top-ks 1,10,100] [--corpora random,clustered,mixed] [--chunk-sizes 32,64,128,256] [--rounds N] [--write-batch-size N] [--seed N] [--csv PATH]\nDefaults: vectors=4096, queries=8, top-ks=1,10,100, corpora=random,clustered,mixed, chunk-sizes=32,64,128,256, rounds=5, seed=7"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}").into()),
        }
    }
    Ok(config)
}

fn parse_list<T>(flag: &str, value: Option<String>) -> Result<Vec<T>, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    let raw = value.ok_or_else(|| format!("{flag} requires a comma-separated list"))?;
    raw.split(',').map(|part| part.parse::<T>().map_err(|e| Box::new(e) as Box<dyn Error>)).collect()
}

fn parse_value<T>(flag: &str, value: Option<String>) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: Error + 'static,
{
    Ok(value.ok_or_else(|| format!("{flag} requires a value"))?.parse()?)
}
