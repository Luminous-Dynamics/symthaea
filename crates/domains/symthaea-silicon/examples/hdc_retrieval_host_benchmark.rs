// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic host-only retrieval baseline for NEURO-SILICON-003.
//!
//! Run from the repository root:
//!   cargo run --locked -p symthaea-silicon --example hdc_retrieval_host_benchmark
//!
//! Optional flags:
//!   --store-size 64|256|1024|4096
//!   --batch-size 1|4|16|64
//!   --repetitions N     (measurement batches; default 30)
//!   --warmup N           (warm-up batches; default 5)
//!   --out DIR            (default target/neuro-silicon-003)
//!
//! This is deliberately a software-only retrieval micro-workload. It uses the
//! canonical BinaryHV::random and BinaryHV::hamming_distance APIs. Store/query
//! generation and serialization are outside the latency boundary. Thus a clean
//! run emits status=incomplete, not pass, for the complete end-to-end protocol.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use symthaea_core::hdc::binary_hv::BinaryHV;

const DIMENSION_BITS: usize = BinaryHV::DIM;
const VECTOR_BYTES: usize = BinaryHV::BYTES;
const STORE_SIZES: [usize; 4] = [64, 256, 1024, 4096];
const BATCH_SIZES: [usize; 4] = [1, 4, 16, 64];
const TARGET_NOISE_BITS: usize = 64;
const NOISE_STRIDE: usize = 7919;

#[derive(Debug)]
struct Config {
    store_size: usize,
    batch_size: usize,
    repetitions: usize,
    warmup: usize,
    output_dir: PathBuf,
}

struct QueryCase {
    index: usize,
    target_id: usize,
    vector: BinaryHV,
}

#[derive(Debug)]
struct Observation {
    repetition: usize,
    query_index: usize,
    expected_id: usize,
    actual_id: usize,
    distance: u32,
    latency_ns: u128,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("NEURO-SILICON-003 host benchmark blocked: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let config = parse_args(&args)?;
    let repo_root = PathBuf::from(git_text(None, &["rev-parse", "--show-toplevel"])?);
    let repo_root = fs::canonicalize(&repo_root)
        .map_err(|error| format!("cannot canonicalize repository root: {error}"))?;

    let head_sha = git_text(Some(&repo_root), &["rev-parse", "HEAD"])?;
    let tree_sha = git_text(Some(&repo_root), &["rev-parse", "HEAD^{tree}"])?;
    let dirty = git_text(Some(&repo_root), &["status", "--porcelain", "--untracked-files=normal"])?;
    if !dirty.is_empty() {
        return Err(format!(
            "refusing to bind a benchmark receipt to a dirty source tree; git status reports: {}",
            dirty.lines().take(8).collect::<Vec<_>>().join(" | ")
        ));
    }

    let output_dir = if config.output_dir.is_absolute() {
        config.output_dir.clone()
    } else {
        repo_root.join(&config.output_dir)
    };
    fs::create_dir_all(&output_dir)
        .map_err(|error| format!("cannot create output directory {}: {error}", output_dir.display()))?;

    let setup_started = Instant::now();
    let store: Vec<BinaryHV> = (0..config.store_size)
        .map(|index| BinaryHV::random(0x5349_4c49_434f_4e00_u64.wrapping_add(index as u64)))
        .collect();
    let queries: Vec<QueryCase> = (0..config.batch_size)
        .map(|index| {
            let target_id = (index * 17 + 3) % config.store_size;
            let vector = perturb_query(
                store[target_id],
                index,
                target_id,
                TARGET_NOISE_BITS,
            );
            QueryCase { index, target_id, vector }
        })
        .collect();
    let setup_ns = setup_started.elapsed().as_nanos();

    let store_hash = hash_vectors(store.iter().map(|hv| &hv.0));
    let query_hash = hash_vectors(queries.iter().map(|query| &query.vector.0));
    let manifest = json!({
        "manifest_version": "neuro-silicon-003-input-v0.1",
        "generation": {
            "store_operation": "BinaryHV::random(seed)",
            "store_seed_base_u64": 0x5349_4c49_434f_4e00_u64,
            "query_operation": "copy declared store target and flip deterministic unique bit positions",
            "query_target_index_rule": "(query_index * 17 + 3) % store_vector_count",
            "query_noise_bits": TARGET_NOISE_BITS,
            "query_noise_bit_rule": "(flip_index * 7919 + target_id * 1013 + query_index * 97) % 16384",
            "query_noise_stride": NOISE_STRIDE,
            "oracle_source_api": "symthaea_core::hdc::binary_hv::BinaryHV"
        },
        "dimension_bits": DIMENSION_BITS,
        "vector_bytes": VECTOR_BYTES,
        "store_vector_count": config.store_size,
        "query_batch_size": config.batch_size,
        "store_vectors_sha256": store_hash,
        "query_vectors_sha256": query_hash,
        "queries": queries.iter().map(|query| json!({
            "query_index": query.index,
            "expected_target_id": query.target_id
        })).collect::<Vec<_>>()
    });
    let expected = json!({
        "manifest_version": "neuro-silicon-003-expected-v0.1",
        "expected_top1_ids": queries.iter().map(|query| query.target_id).collect::<Vec<_>>(),
        "tie_break_policy": "lowest stable vector-store index wins equal Hamming distance",
        "expected_condition": format!("each query is its declared target with {TARGET_NOISE_BITS} deterministic bit flips; target must remain top-1")
    });
    let manifest_bytes = json_bytes(&manifest)?;
    let expected_bytes = json_bytes(&expected)?;
    let manifest_sha = sha256(&manifest_bytes);
    let expected_sha = sha256(&expected_bytes);
    let manifest_artifact = write_artifact(&output_dir, "input_manifest.json", &manifest_bytes)?;
    let expected_artifact = write_artifact(&output_dir, "expected_results.json", &expected_bytes)?;

    // Warm-up runs are intentionally not part of the measured population.
    for _ in 0..config.warmup {
        for query in &queries {
            let _ = retrieve_top1(&store, &query.vector);
        }
    }

    let mut observations = Vec::with_capacity(config.repetitions * config.batch_size);
    let mut batch_elapsed_ns = Vec::with_capacity(config.repetitions);
    let wall_started = Instant::now();
    for repetition in 0..config.repetitions {
        let batch_started = Instant::now();
        for query in &queries {
            let query_started = Instant::now();
            let (actual_id, distance) = retrieve_top1(&store, &query.vector);
            let latency_ns = query_started.elapsed().as_nanos();
            observations.push(Observation {
                repetition,
                query_index: query.index,
                expected_id: query.target_id,
                actual_id,
                distance,
                latency_ns,
            });
        }
        batch_elapsed_ns.push(batch_started.elapsed().as_nanos());
    }
    let wall_elapsed_ns = wall_started.elapsed().as_nanos().max(1);

    let measured_queries = observations.len();
    let mismatches = observations
        .iter()
        .filter(|observation| observation.actual_id != observation.expected_id)
        .count();
    let mut latency_samples: Vec<u128> = observations.iter().map(|observation| observation.latency_ns).collect();
    latency_samples.sort_unstable();
    let throughput_qps = measured_queries as f64 / (wall_elapsed_ns as f64 / 1_000_000_000.0);
    let raw = json!({
        "raw_observation_version": "neuro-silicon-003-observations-v0.1",
        "measurement_boundary": "prebuilt query scan + canonical Hamming distance + deterministic top-1 selection",
        "boundary_exclusions": ["vector-store construction", "query construction/encoding", "artifact serialization", "external host/device transfers (CPU-only profile)"],
        "wall_elapsed_ns": wall_elapsed_ns,
        "batch_elapsed_ns": batch_elapsed_ns,
        "observations": observations.iter().map(|observation| json!({
            "repetition": observation.repetition,
            "query_index": observation.query_index,
            "expected_id": observation.expected_id,
            "actual_id": observation.actual_id,
            "hamming_distance": observation.distance,
            "latency_ns": observation.latency_ns
        })).collect::<Vec<_>>()
    });
    let raw_bytes = json_bytes(&raw)?;
    let raw_artifact = write_artifact(&output_dir, "raw_observations.json", &raw_bytes)?;

    let hardware_identity = env::var("SYTHAEA_BENCH_HARDWARE")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(detect_cpu_identity);
    let working_set_bytes = config.store_size * VECTOR_BYTES + config.batch_size * VECTOR_BYTES;
    let latency_scope = "single query: full scan of prebuilt host-resident store, exact canonical Hamming scoring, deterministic top-1 selection; excludes store/query construction and serialization";
    let metrics = vec![
        metric("latency_p50", percentile(&latency_samples, 0.50) as f64, "ns", "measured", latency_scope, measured_queries, "nearest-rank percentile over individually timed queries"),
        metric("latency_p95", percentile(&latency_samples, 0.95) as f64, "ns", "measured", latency_scope, measured_queries, "nearest-rank percentile over individually timed queries"),
        metric("latency_p99", percentile(&latency_samples, 0.99) as f64, "ns", "measured", latency_scope, measured_queries, "nearest-rank percentile over individually timed queries"),
        metric("throughput", throughput_qps, "queries/s", "measured", "aggregate retrieval loop wall time, including per-query timing instrumentation; excludes warm-up and setup", measured_queries, "measured query count divided by aggregate loop wall time"),
        metric("working_set_bytes", working_set_bytes as f64, "bytes", "derived", "logical vector-store plus one batch of query hypervectors; excludes allocator metadata and runtime overhead", 1, "derived from canonical 2048-byte vectors"),
        metric("other", setup_ns as f64, "ns", "measured", "setup duration for deterministic store and query generation; outside retrieval latency", 1, "monotonic wall clock around generation")
    ];
    let mut limitations = vec![
        "Synthetic nearest-neighbour retrieval only; not a production semantic retrieval quality benchmark.".to_owned(),
        "The query vector store and queries are generated before timing; query encoding and store construction are not part of retrieval latency.".to_owned(),
        "CPU-only profile: no host/device transfer, GPU, FPGA, power, or energy result is measured.".to_owned(),
        "The receipt is incomplete for NEURO-SILICON-003's full end-to-end promotion gate because boundary_complete is false.".to_owned(),
        "No malformed hardware frames are applicable to this host-only retrieval run; malformed_frames_accepted is null, not a tested zero.".to_owned(),
        "Command stdout/stderr hashes are not captured by this in-process example; preserve external runner logs with any qualification package.".to_owned(),
    ];
    let status = if mismatches > 0 {
        limitations.push("At least one known-target query did not return its declared target as top-1.".to_owned());
        "fail"
    } else {
        "incomplete"
    };
    let receipt = json!({
        "schema_version": "0.1",
        "status": status,
        "claim_ceiling": "software_only",
        "source": {
            "repository": "Luminous-Dynamics/symthaea",
            "commit_sha": head_sha,
            "tree_sha": tree_sha
        },
        "workload": {
            "workload_id": "synthetic-hdc-retrieval-v0.1",
            "input_manifest_sha256": manifest_sha,
            "expected_results_sha256": expected_sha,
            "vector_dimension_bits": DIMENSION_BITS,
            "store_vector_count": config.store_size,
            "query_batch_size": config.batch_size,
            "query_count": measured_queries,
            "operations": ["hamming_distance", "retrieve"],
            "tie_break_policy": "lowest stable vector-store index wins equal Hamming distance",
            "task_quality_metric": "synthetic target top-1 accuracy",
            "task_quality_value": (measured_queries.saturating_sub(mismatches) as f64) / measured_queries as f64,
            "task_quality_floor": "1.0"
        },
        "profile": {
            "profile_id": "canonical-rust-binaryhv-host-scan",
            "backend": "cpu_dispatch",
            "host_stages": ["scan store", "BinaryHV::hamming_distance", "deterministic top-1 selection"],
            "device_stages": [],
            "hardware_identity": hardware_identity,
            "memory_profile": format!("host-resident vector store; logical working set {working_set_bytes} bytes; actual cache/DRAM residency not measured"),
            "clock_power_profile": env::var("SYTHAEA_BENCH_CLOCK_POWER_PROFILE").unwrap_or_else(|_| "not captured".to_owned())
        },
        "execution": {
            "measurement_boundary": latency_scope,
            "boundary_complete": false,
            "warmup_batches": config.warmup,
            "measured_batches": config.repetitions,
            "invocation": args.join(" "),
            "commands": []
        },
        "correctness": {
            "oracle_comparison": if mismatches == 0 { "exact_match" } else { "mismatch" },
            "oracle_identity": "frozen synthetic target IDs in expected_results.json; this is not an independent hardware differential oracle",
            "fixtures_attempted": measured_queries,
            "fixtures_passed": measured_queries.saturating_sub(mismatches),
            "mismatches": mismatches,
            "malformed_frames_accepted": Value::Null
        },
        "metrics": metrics,
        "artifacts": [manifest_artifact, expected_artifact, raw_artifact],
        "limitations": limitations
    });
    let receipt_bytes = json_bytes(&receipt)?;
    fs::write(output_dir.join("receipt.json"), &receipt_bytes)
        .map_err(|error| format!("cannot write receipt.json: {error}"))?;

    println!("{}", String::from_utf8_lossy(&receipt_bytes));
    eprintln!(
        "output={} status={} batches={} queries={} mismatches={}",
        output_dir.display(), status, config.repetitions, measured_queries, mismatches
    );
    Ok(())
}

fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut config = Config {
        store_size: 64,
        batch_size: 1,
        repetitions: 30,
        warmup: 5,
        output_dir: PathBuf::from("target/neuro-silicon-003"),
    };
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        if flag == "--help" || flag == "-h" {
            println!("hdc_retrieval_host_benchmark [--store-size 64|256|1024|4096] [--batch-size 1|4|16|64] [--repetitions N] [--warmup N] [--out DIR]");
            std::process::exit(0);
        }
        let value = args.get(index + 1).ok_or_else(|| format!("missing value for {flag}"))?;
        match flag {
            "--store-size" => config.store_size = parse_usize(flag, value)?,
            "--batch-size" => config.batch_size = parse_usize(flag, value)?,
            "--repetitions" => config.repetitions = parse_usize(flag, value)?,
            "--warmup" => config.warmup = parse_usize(flag, value)?,
            "--out" => config.output_dir = PathBuf::from(value),
            _ => return Err(format!("unknown argument: {flag}; use --help")),
        }
        index += 2;
    }
    if !STORE_SIZES.contains(&config.store_size) {
        return Err(format!("unsupported store size {}; choose one of {:?}", config.store_size, STORE_SIZES));
    }
    if !BATCH_SIZES.contains(&config.batch_size) || config.batch_size > config.store_size {
        return Err(format!("unsupported batch size {}; choose one of {:?}", config.batch_size, BATCH_SIZES));
    }
    if config.repetitions == 0 {
        return Err("--repetitions must be greater than zero".to_owned());
    }
    Ok(config)
}

fn parse_usize(flag: &str, value: &str) -> Result<usize, String> {
    value.parse::<usize>().map_err(|error| format!("invalid integer for {flag}: {error}"))
}

fn perturb_query(mut vector: BinaryHV, query_index: usize, target_id: usize, flips: usize) -> BinaryHV {
    for flip_index in 0..flips {
        let bit_index =
            (flip_index * NOISE_STRIDE + target_id * 1013 + query_index * 97) % DIMENSION_BITS;
        vector.0[bit_index / 8] ^= 1 << (bit_index % 8);
    }
    vector
}

fn retrieve_top1(store: &[BinaryHV], query: &BinaryHV) -> (usize, u32) {
    let mut best_id = 0usize;
    let mut best_distance = u32::MAX;
    for (item_id, item) in store.iter().enumerate() {
        let distance = query.hamming_distance(item);
        if distance < best_distance || (distance == best_distance && item_id < best_id) {
            best_id = item_id;
            best_distance = distance;
        }
    }
    (best_id, best_distance)
}

fn percentile(sorted_samples: &[u128], percentile: f64) -> u128 {
    let rank = ((sorted_samples.len() as f64) * percentile).ceil() as usize;
    sorted_samples[rank.saturating_sub(1).min(sorted_samples.len() - 1)]
}

fn metric(
    metric_id: &str,
    value: f64,
    unit: &str,
    kind: &str,
    scope: &str,
    sample_count: usize,
    method: &str,
) -> Value {
    json!({
        "metric_id": metric_id,
        "value": value,
        "unit": unit,
        "kind": kind,
        "scope": scope,
        "sample_count": sample_count,
        "method": method
    })
}

fn hash_vectors<'a>(vectors: impl Iterator<Item = &'a [u8; VECTOR_BYTES]>) -> String {
    let mut hasher = Sha256::new();
    for vector in vectors {
        hasher.update(vector);
    }
    format!("{:x}", hasher.finalize())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_bytes(value: &Value) -> Result<Vec<u8>, String> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("cannot serialize JSON artifact: {error}"))?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn write_artifact(output_dir: &Path, name: &str, bytes: &[u8]) -> Result<Value, String> {
    fs::write(output_dir.join(name), bytes)
        .map_err(|error| format!("cannot write {name}: {error}"))?;
    Ok(json!({
        "path": name,
        "sha256": sha256(bytes),
        "media_type": "application/json",
        "size_bytes": bytes.len()
    }))
}

fn git_text(cwd: Option<&Path>, args: &[&str]) -> Result<String, String> {
    let mut command = Command::new("git");
    command.args(args);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    let output = command.output().map_err(|error| format!("cannot execute git {:?}: {error}", args))?;
    if !output.status.success() {
        return Err(format!(
            "git {:?} failed: {}",
            args,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

fn detect_cpu_identity() -> String {
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        if let Some(line) = cpuinfo.lines().find(|line| line.starts_with("model name\t") || line.starts_with("Hardware\t")) {
            if let Some((_, value)) = line.split_once(':') {
                return format!("{}; {} {}", value.trim(), env::consts::ARCH, env::consts::OS);
            }
        }
    }
    format!("unidentified CPU; {} {}", env::consts::ARCH, env::consts::OS)
}
