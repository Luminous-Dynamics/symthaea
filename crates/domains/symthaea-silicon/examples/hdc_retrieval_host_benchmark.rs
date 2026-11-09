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
//!   --validate-receipt PATH/receipt.json
//!
//! This is deliberately a software-only retrieval micro-workload. It uses the
//! canonical BinaryHV::random and BinaryHV::hamming_distance APIs. Store/query
//! generation and serialization are outside the latency boundary. Thus a clean
//! run emits status=incomplete, not pass, for the complete end-to-end protocol.

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Component, Path, PathBuf};
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

#[derive(Debug)]
struct ScanOrderObservation {
    repetition: usize,
    arm: &'static str,
    elapsed_ns: u128,
    parity_mismatches: usize,
    oracle_mismatches: usize,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("NEURO-SILICON-003 host benchmark blocked: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--validate-receipt") {
        if args.len() != 2 {
            return Err("usage: hdc_retrieval_host_benchmark --validate-receipt PATH/receipt.json".to_owned());
        }
        let receipt_path = PathBuf::from(&args[1]);
        let bytes = fs::read(&receipt_path)
            .map_err(|error| format!("cannot read receipt {}: {error}", receipt_path.display()))?;
        let receipt: Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("receipt is not valid JSON: {error}"))?;
        let base_dir = receipt_path.parent().filter(|path| !path.as_os_str().is_empty()).unwrap_or(Path::new("."));
        validate_receipt(&receipt, base_dir)?;
        println!(
            "receipt preflight valid: status={} source_commit={}",
            receipt["status"].as_str().unwrap_or("unknown"),
            receipt["source"]["commit_sha"].as_str().unwrap_or("unknown")
        );
        return Ok(());
    }
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
            std::hint::black_box(retrieve_top1(&store, &query.vector));
        }
        std::hint::black_box(retrieve_batch_store_major(&store, &queries)?);
        std::hint::black_box(retrieve_batch_store_major_tiled(&store, &queries, 8)?);
    }

    // A single canonical query-major output vector is the exact parity oracle
    // for the alternative traversal orders. Each timed arm uses the same store,
    // query batch and canonical BinaryHV::hamming_distance implementation.
    let query_major_reference = retrieve_batch_query_major(&store, &queries)?;
    let mut scan_order_observations = Vec::with_capacity(config.repetitions * 3);
    let mut scan_order_elapsed_ns: [Vec<u128>; 3] =
        std::array::from_fn(|_| Vec::with_capacity(config.repetitions));
    let mut scan_order_parity_mismatches = [0usize; 3];
    let mut scan_order_oracle_mismatches = [0usize; 3];

    // Rotate the three arms each repetition to reduce fixed order bias.
    // Per-arm samples are batch latencies; we do not fabricate per-query
    // timing measurements for store-major traversal.
    for repetition in 0..config.repetitions {
        for offset in 0..3 {
            let arm = (repetition + offset) % 3;
            let started = Instant::now();
            let results = match arm {
                0 => retrieve_batch_query_major(&store, &queries)?,
                1 => retrieve_batch_store_major(&store, &queries)?,
                2 => retrieve_batch_store_major_tiled(&store, &queries, 8)?,
                _ => unreachable!("three scan-order arms"),
            };
            let elapsed_ns = started.elapsed().as_nanos().max(1);
            let parity_mismatches = results
                .iter()
                .zip(query_major_reference.iter())
                .filter(|(actual, expected)| actual != expected)
                .count()
                + results.len().abs_diff(query_major_reference.len());
            let oracle_mismatches = results
                .iter()
                .zip(queries.iter())
                .filter(|((actual_id, _), query)| *actual_id != query.target_id)
                .count()
                + results.len().abs_diff(queries.len());
            scan_order_parity_mismatches[arm] += parity_mismatches;
            scan_order_oracle_mismatches[arm] += oracle_mismatches;
            scan_order_elapsed_ns[arm].push(elapsed_ns);
            scan_order_observations.push(ScanOrderObservation {
                repetition,
                arm: match arm {
                    0 => "query_major",
                    1 => "store_major",
                    2 => "store_major_tiled_q8",
                    _ => unreachable!("three scan-order arms"),
                },
                elapsed_ns,
                parity_mismatches,
                oracle_mismatches,
            });
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
        })).collect::<Vec<_>>(),
        "scan_order_comparison": {
            "boundary": "prebuilt resident store and prebuilt query batch; complete exact scan and top-1 selection; batch timed as one unit",
            "canonical_distance": "BinaryHV::hamming_distance",
            "arms": ["query_major", "store_major", "store_major_tiled_q8"],
            "interleaving": "arm order rotated once per repetition",
            "observations": scan_order_observations.iter().map(|observation| json!({
                "repetition": observation.repetition,
                "arm": observation.arm,
                "batch_elapsed_ns": observation.elapsed_ns,
                "parity_mismatches_vs_query_major": observation.parity_mismatches,
                "oracle_top1_mismatches": observation.oracle_mismatches
            })).collect::<Vec<_>>(),
            "aggregate_parity_mismatches": {
                "query_major": scan_order_parity_mismatches[0],
                "store_major": scan_order_parity_mismatches[1],
                "store_major_tiled_q8": scan_order_parity_mismatches[2]
            },
            "aggregate_oracle_mismatches": {
                "query_major": scan_order_oracle_mismatches[0],
                "store_major": scan_order_oracle_mismatches[1],
                "store_major_tiled_q8": scan_order_oracle_mismatches[2]
            }
        }
    });
    let raw_bytes = json_bytes(&raw)?;
    let raw_artifact = write_artifact(&output_dir, "raw_observations.json", &raw_bytes)?;

    let hardware_identity = env::var("SYMTHAEA_BENCH_HARDWARE")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(detect_cpu_identity);
    let working_set_bytes = config.store_size * VECTOR_BYTES + config.batch_size * VECTOR_BYTES;
    let latency_scope = "single query: full scan of prebuilt host-resident store, exact canonical Hamming scoring, deterministic top-1 selection; excludes store/query construction and serialization";
    let mut metrics = vec![
        metric("latency_p50", percentile(&latency_samples, 0.50) as f64, "ns", "measured", latency_scope, measured_queries, "nearest-rank percentile over individually timed queries"),
        metric("latency_p95", percentile(&latency_samples, 0.95) as f64, "ns", "measured", latency_scope, measured_queries, "nearest-rank percentile over individually timed queries"),
        metric("throughput", throughput_qps, "queries/s", "measured", "aggregate query-major retrieval loop wall time, including per-query timing instrumentation; excludes warm-up and setup", measured_queries, "measured query count divided by aggregate loop wall time"),
        metric("working_set_bytes", working_set_bytes as f64, "bytes", "derived", "logical vector-store plus one batch of query hypervectors; excludes allocator metadata and runtime overhead", 1, "derived from canonical 2048-byte vectors"),
        metric("other", setup_ns as f64, "ns", "measured", "setup duration for deterministic store and query generation; outside retrieval latency", 1, "monotonic wall clock around generation")
    ];
    if latency_samples.len() >= 1_000 {
        metrics.push(metric("latency_p99", percentile(&latency_samples, 0.99) as f64, "ns", "measured", latency_scope, measured_queries, "nearest-rank percentile; emitted only with at least 1,000 individual query observations"));
    }

    for (arm_index, arm_name) in ["query_major", "store_major", "store_major_tiled_q8"].iter().enumerate() {
        let mut samples = scan_order_elapsed_ns[arm_index].clone();
        samples.sort_unstable();
        let arm_scope = format!("complete prebuilt query-batch scan using {arm_name}; canonical BinaryHV::hamming_distance; excludes query/store construction and serialization");
        metrics.push(metric(
            "latency_p50",
            percentile(&samples, 0.50) as f64,
            "ns",
            "measured",
            &arm_scope,
            samples.len(),
            &format!("whole-batch nearest-rank p50 for {arm_name}"),
        ));
        metrics.push(metric(
            "latency_p95",
            percentile(&samples, 0.95) as f64,
            "ns",
            "measured",
            &arm_scope,
            samples.len(),
            &format!("whole-batch nearest-rank p95 for {arm_name}; exploratory at low sample counts"),
        ));
        if samples.len() >= 1_000 {
            metrics.push(metric(
                "latency_p99",
                percentile(&samples, 0.99) as f64,
                "ns",
                "measured",
                &arm_scope,
                samples.len(),
                "nearest-rank percentile; emitted only with at least 1,000 whole-batch observations",
            ));
        }
        let total_ns: u128 = samples.iter().copied().sum::<u128>().max(1);
        let qps = (config.batch_size * samples.len()) as f64
            / (total_ns as f64 / 1_000_000_000.0);
        metrics.push(metric(
            "throughput",
            qps,
            "queries/s",
            "measured",
            &arm_scope,
            samples.len(),
            "query count divided by summed batch elapsed time; excludes warm-up",
        ));
        metrics.push(metric(
            "other",
            scan_order_parity_mismatches[arm_index] as f64,
            "count",
            "derived",
            &format!("individual result parity mismatches versus query-major for {arm_name}"),
            samples.len(),
            "exact tuple equality on (top1 ID, Hamming distance)",
        ));
        metrics.push(metric(
            "other",
            scan_order_oracle_mismatches[arm_index] as f64,
            "count",
            "derived",
            &format!("top-1 oracle mismatches for {arm_name}"),
            samples.len(),
            "exact ID equality against expected_results.json",
        ));
    }
    let mut limitations = vec![
        "Synthetic nearest-neighbour retrieval only; not a production semantic retrieval quality benchmark.".to_owned(),
        "The query vector store and queries are generated before timing; query encoding and store construction are not part of retrieval latency.".to_owned(),
        "CPU-only profile: no host/device transfer, GPU, FPGA, power, or energy result is measured.".to_owned(),
        "The receipt is incomplete for NEURO-SILICON-003's full end-to-end promotion gate because boundary_complete is false.".to_owned(),
        "The scan-order comparison is an exact host-only traversal experiment; it does not measure remote placement, GPU/FPGA execution, hardware cache counters or energy.".to_owned(),
        "Scan-order p99 batch latency is omitted unless each arm has at least 1,000 batch observations; p95 from the default 30 repetitions is exploratory.".to_owned(),
        "No malformed hardware frames are applicable to this host-only retrieval run; malformed_frames_accepted is null, not a tested zero.".to_owned(),
        "Command stdout/stderr hashes are not captured by this in-process example; preserve external runner logs with any qualification package.".to_owned(),
    ];
    let scan_order_failed = scan_order_parity_mismatches.iter().any(|count| *count > 0)
        || scan_order_oracle_mismatches.iter().any(|count| *count > 0);
    let status = if mismatches > 0 || scan_order_failed {
        if mismatches > 0 {
            limitations.push("At least one query-major known-target query did not return its declared target as top-1.".to_owned());
        }
        if scan_order_failed {
            limitations.push("At least one scan-order arm disagreed with query-major or the frozen target oracle.".to_owned());
        }
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
            "query_count": config.batch_size,
            "operations": ["hamming_distance", "retrieve"],
            "tie_break_policy": "lowest stable vector-store index wins equal Hamming distance",
            "task_quality_metric": "synthetic target top-1 accuracy",
            "task_quality_value": (measured_queries.saturating_sub(mismatches) as f64) / measured_queries as f64,
            "task_quality_floor": 1.0
        },
        "profile": {
            "profile_id": "canonical-rust-binaryhv-host-scan",
            "backend": "cpu_dispatch",
            "host_stages": ["scan store", "BinaryHV::hamming_distance", "deterministic top-1 selection"],
            "device_stages": [],
            "hardware_identity": hardware_identity,
            "memory_profile": format!("host-resident vector store; logical working set {working_set_bytes} bytes; actual cache/DRAM residency not measured"),
            "clock_power_profile": env::var("SYMTHAEA_BENCH_CLOCK_POWER_PROFILE").unwrap_or_else(|_| "not captured".to_owned())
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
    validate_receipt(&receipt, &output_dir)?;
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

/// Check the receipt's high-impact invariants and all referenced artifact hashes.
/// This is a semantic preflight, not a replacement for validating against the
/// versioned JSON Schema or independently reproducing the measurements.
fn validate_receipt(receipt: &Value, base_dir: &Path) -> Result<(), String> {
    let status = receipt.get("status").and_then(Value::as_str)
        .ok_or_else(|| "receipt is missing string status".to_owned())?;
    if !["pass", "fail", "incomplete", "not_comparable"].contains(&status) {
        return Err(format!("receipt has unsupported status {status:?}"));
    }
    if receipt.get("schema_version").and_then(Value::as_str) != Some("0.1") {
        return Err("receipt schema_version must be 0.1".to_owned());
    }

    let source = receipt.get("source").ok_or_else(|| "receipt is missing source".to_owned())?;
    for key in ["commit_sha", "tree_sha"] {
        let sha = source.get(key).and_then(Value::as_str)
            .ok_or_else(|| format!("receipt source is missing {key}"))?;
        if sha.len() != 40 || !sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!("receipt source {key} is not a 40-character hexadecimal identifier"));
        }
    }

    let canonical_base = fs::canonicalize(base_dir)
        .map_err(|error| format!("cannot resolve receipt artifact directory {}: {error}", base_dir.display()))?;
    let mut artifact_hashes = std::collections::BTreeMap::<String, String>::new();
    let artifacts = receipt.get("artifacts").and_then(Value::as_array)
        .ok_or_else(|| "receipt artifacts must be an array".to_owned())?;
    if artifacts.is_empty() && status == "pass" {
        return Err("pass receipt must reference at least one evidence artifact".to_owned());
    }
    for artifact in artifacts {
        let path_text = artifact.get("path").and_then(Value::as_str)
            .ok_or_else(|| "artifact is missing path".to_owned())?;
        let relative = Path::new(path_text);
        if relative.is_absolute() || relative.components().any(|component| {
            matches!(component, Component::ParentDir | Component::RootDir | Component::Prefix(_))
        }) {
            return Err(format!("artifact path must be relative and cannot escape receipt directory: {path_text}"));
        }
        let candidate_path = canonical_base.join(relative);
        let canonical_artifact = fs::canonicalize(&candidate_path)
            .map_err(|error| format!("cannot resolve artifact {path_text}: {error}"))?;
        if !canonical_artifact.starts_with(&canonical_base) {
            return Err(format!("artifact resolves outside the receipt directory: {path_text}"));
        }
        let bytes = fs::read(&canonical_artifact)
            .map_err(|error| format!("cannot read artifact {path_text}: {error}"))?;
        let expected_sha = artifact.get("sha256").and_then(Value::as_str)
            .ok_or_else(|| format!("artifact {path_text} is missing sha256"))?;
        if sha256(&bytes) != expected_sha {
            return Err(format!("artifact hash mismatch: {path_text}"));
        }
        let expected_size = artifact.get("size_bytes").and_then(Value::as_u64)
            .ok_or_else(|| format!("artifact {path_text} is missing integer size_bytes"))?;
        if expected_size != bytes.len() as u64 {
            return Err(format!("artifact size mismatch: {path_text}"));
        }
        if artifact_hashes.insert(path_text.to_owned(), expected_sha.to_owned()).is_some() {
            return Err(format!("duplicate artifact path: {path_text}"));
        }
    }

    let workload = receipt.get("workload").ok_or_else(|| "receipt is missing workload".to_owned())?;
    for (field, artifact_path) in [
        ("input_manifest_sha256", "input_manifest.json"),
        ("expected_results_sha256", "expected_results.json"),
    ] {
        if status == "pass" || workload.get(field).is_some() {
            let declared = workload.get(field).and_then(Value::as_str)
                .ok_or_else(|| format!("workload is missing {field}"))?;
            let artifact = artifact_hashes.get(artifact_path)
                .ok_or_else(|| format!("required artifact {artifact_path} is not referenced"))?;
            if declared != artifact {
                return Err(format!("{field} does not match artifact {artifact_path}"));
            }
        }
    }

    let execution = receipt.get("execution").ok_or_else(|| "receipt is missing execution".to_owned())?;
    let boundary_complete = execution.get("boundary_complete").and_then(Value::as_bool)
        .ok_or_else(|| "execution.boundary_complete must be boolean".to_owned())?;
    let measured_batches = execution.get("measured_batches").and_then(Value::as_u64)
        .ok_or_else(|| "execution.measured_batches must be an integer".to_owned())?;
    let correctness = receipt.get("correctness").ok_or_else(|| "receipt is missing correctness".to_owned())?;
    let mismatch_count = correctness.get("mismatches").and_then(Value::as_u64)
        .ok_or_else(|| "correctness.mismatches must be an integer".to_owned())?;
    let attempted = correctness.get("fixtures_attempted").and_then(Value::as_u64)
        .ok_or_else(|| "correctness.fixtures_attempted must be an integer".to_owned())?;
    let passed = correctness.get("fixtures_passed").and_then(Value::as_u64)
        .ok_or_else(|| "correctness.fixtures_passed must be an integer".to_owned())?;
    if passed.saturating_add(mismatch_count) != attempted {
        return Err("fixtures_passed + mismatches must equal fixtures_attempted".to_owned());
    }
    if !boundary_complete && status == "pass" {
        return Err("a workload with an incomplete measurement boundary cannot pass".to_owned());
    }
    if mismatch_count > 0 && status != "fail" {
        return Err("a receipt with correctness mismatches must have status=fail".to_owned());
    }
    if status == "pass" {
        if measured_batches < 30 {
            return Err("pass receipt requires at least 30 measured batches".to_owned());
        }
        if correctness.get("oracle_comparison").and_then(Value::as_str) != Some("exact_match") {
            return Err("pass receipt requires exact_match oracle comparison".to_owned());
        }
        if correctness.get("malformed_frames_accepted").and_then(Value::as_u64) != Some(0) {
            return Err("pass receipt requires an explicit zero malformed-frame acceptance count".to_owned());
        }
        let metrics = receipt.get("metrics").and_then(Value::as_array)
            .ok_or_else(|| "receipt metrics must be an array".to_owned())?;
        let has_measured_tail = metrics.iter().any(|metric| {
            matches!(
                metric.get("metric_id").and_then(Value::as_str),
                Some("latency_p95" | "throughput" | "energy_per_query")
            ) && metric.get("kind").and_then(Value::as_str) == Some("measured")
                && metric.get("sample_count").and_then(Value::as_u64).unwrap_or(0) >= 30
        });
        if !has_measured_tail {
            return Err("pass receipt requires at least one measured p95/throughput/energy metric with 30 or more samples".to_owned());
        }
        if artifacts.len() == 0 {
            return Err("pass receipt requires evidence artifacts".to_owned());
        }
    }
    if matches!(status, "incomplete" | "not_comparable")
        && receipt.get("limitations").and_then(Value::as_array).map_or(true, Vec::is_empty)
    {
        return Err(format!("status={status} requires at least one explicit limitation"));
    }
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

/// Query-major batch scan: the current single-query implementation is reused
/// unchanged, and each output is (stable top-1 ID, exact Hamming distance).
fn retrieve_batch_query_major(
    store: &[BinaryHV],
    queries: &[QueryCase],
) -> Result<Vec<(usize, u32)>, String> {
    if store.is_empty() && !queries.is_empty() {
        return Err("cannot retrieve from an empty store".to_owned());
    }
    Ok(queries
        .iter()
        .map(|query| retrieve_top1(store, &query.vector))
        .collect())
}

/// Store-major exact batch scan. Each resident store vector is traversed once
/// per batch and compared with every query using the canonical Hamming kernel.
fn retrieve_batch_store_major(
    store: &[BinaryHV],
    queries: &[QueryCase],
) -> Result<Vec<(usize, u32)>, String> {
    if store.is_empty() && !queries.is_empty() {
        return Err("cannot retrieve from an empty store".to_owned());
    }
    let mut best_ids = vec![usize::MAX; queries.len()];
    let mut best_distances = vec![u32::MAX; queries.len()];

    for (item_id, item) in store.iter().enumerate() {
        for (query_index, query) in queries.iter().enumerate() {
            let distance = query.vector.hamming_distance(item);
            if distance < best_distances[query_index]
                || (distance == best_distances[query_index] && item_id < best_ids[query_index])
            {
                best_ids[query_index] = item_id;
                best_distances[query_index] = distance;
            }
        }
    }

    Ok(best_ids.into_iter().zip(best_distances).collect())
}

/// Tiled store-major scan. The query tile bounds the active query working set;
/// the inner operation still calls the canonical Hamming-distance kernel.
fn retrieve_batch_store_major_tiled(
    store: &[BinaryHV],
    queries: &[QueryCase],
    query_tile_size: usize,
) -> Result<Vec<(usize, u32)>, String> {
    if query_tile_size == 0 {
        return Err("query tile size must be greater than zero".to_owned());
    }
    if store.is_empty() && !queries.is_empty() {
        return Err("cannot retrieve from an empty store".to_owned());
    }

    let mut outputs = Vec::with_capacity(queries.len());
    for query_tile in queries.chunks(query_tile_size) {
        let mut best_ids = vec![usize::MAX; query_tile.len()];
        let mut best_distances = vec![u32::MAX; query_tile.len()];

        for (item_id, item) in store.iter().enumerate() {
            for (tile_index, query) in query_tile.iter().enumerate() {
                let distance = query.vector.hamming_distance(item);
                if distance < best_distances[tile_index]
                    || (distance == best_distances[tile_index] && item_id < best_ids[tile_index])
                {
                    best_ids[tile_index] = item_id;
                    best_distances[tile_index] = distance;
                }
            }
        }

        outputs.extend(best_ids.into_iter().zip(best_distances));
    }
    Ok(outputs)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perturbation_is_deterministic_and_has_declared_hamming_distance() {
        let base = BinaryHV::random(42);
        let first = perturb_query(base, 7, 3, TARGET_NOISE_BITS);
        let second = perturb_query(base, 7, 3, TARGET_NOISE_BITS);
        assert_eq!(first, second);
        assert_eq!(first.hamming_distance(&base), TARGET_NOISE_BITS as u32);
    }

    #[test]
    fn deterministic_query_retrieves_declared_target() {
        let store: Vec<BinaryHV> = (0..64)
            .map(|index| BinaryHV::random(0x5349_4c49_434f_4e00_u64.wrapping_add(index)))
            .collect();
        let target_id = 3;
        let query = perturb_query(store[target_id], 0, target_id, TARGET_NOISE_BITS);
        let (actual_id, distance) = retrieve_top1(&store, &query);
        assert_eq!(actual_id, target_id);
        assert_eq!(distance, TARGET_NOISE_BITS as u32);
    }

    #[test]
    fn equal_distance_ties_choose_lowest_stable_index() {
        let vector = BinaryHV::random(99);
        let store = vec![vector, vector];
        assert_eq!(retrieve_top1(&store, &vector), (0, 0));
    }

    #[test]
    fn store_major_exactly_matches_query_major_for_mixed_batch() {
        let store: Vec<BinaryHV> = (0..256)
            .map(|index| BinaryHV::random(0x4844_435f_5354_4f52_u64.wrapping_add(index)))
            .collect();
        let queries: Vec<QueryCase> = (0..16)
            .map(|index| {
                let target_id = (index * 17 + 3) % store.len();
                QueryCase {
                    index,
                    target_id,
                    vector: perturb_query(store[target_id], index, target_id, TARGET_NOISE_BITS),
                }
            })
            .collect();

        let reference = retrieve_batch_query_major(&store, &queries).unwrap();
        let store_major = retrieve_batch_store_major(&store, &queries).unwrap();
        let tiled = retrieve_batch_store_major_tiled(&store, &queries, 8).unwrap();
        assert_eq!(store_major, reference);
        assert_eq!(tiled, reference);
        assert!(reference.iter().zip(&queries).all(|((id, _), query)| *id == query.target_id));
    }

    #[test]
    fn tiled_store_major_matches_for_tile_sizes_and_partial_final_tile() {
        let store: Vec<BinaryHV> = (0..64).map(|i| BinaryHV::random(i as u64)).collect();
        let queries: Vec<QueryCase> = (0..7)
            .map(|index| {
                let target_id = (index * 9 + 1) % store.len();
                QueryCase {
                    index,
                    target_id,
                    vector: perturb_query(store[target_id], index, target_id, TARGET_NOISE_BITS),
                }
            })
            .collect();
        let expected = retrieve_batch_query_major(&store, &queries).unwrap();
        for tile_size in [1, 2, 3, 8, 64] {
            assert_eq!(
                retrieve_batch_store_major_tiled(&store, &queries, tile_size).unwrap(),
                expected,
                "tile size {tile_size}"
            );
        }
    }

    #[test]
    fn all_scan_orders_preserve_lowest_id_tie_break() {
        let vector = BinaryHV::random(199);
        let store = vec![vector, vector, vector];
        let queries = vec![QueryCase { index: 0, target_id: 0, vector }];
        let expected = vec![(0, 0)];
        assert_eq!(retrieve_batch_query_major(&store, &queries).unwrap(), expected);
        assert_eq!(retrieve_batch_store_major(&store, &queries).unwrap(), expected);
        assert_eq!(retrieve_batch_store_major_tiled(&store, &queries, 1).unwrap(), expected);
    }

    #[test]
    fn empty_query_batch_is_supported_but_empty_store_is_not() {
        assert!(retrieve_batch_query_major(&[], &[]).unwrap().is_empty());
        assert!(retrieve_batch_store_major(&[], &[]).unwrap().is_empty());
        assert!(retrieve_batch_store_major_tiled(&[], &[], 1).unwrap().is_empty());

        let query = QueryCase { index: 0, target_id: 0, vector: BinaryHV::random(7) };
        assert!(retrieve_batch_query_major(&[], &[query]).is_err());
    }

    #[test]
    fn tiled_store_major_rejects_zero_tile_size() {
        let store = vec![BinaryHV::random(1)];
        let queries = vec![QueryCase { index: 0, target_id: 0, vector: BinaryHV::random(1) }];
        assert!(retrieve_batch_store_major_tiled(&store, &queries, 0).is_err());
    }

    #[test]
    fn percentile_uses_nearest_rank_for_small_samples() {
        let samples = [10, 20, 30, 40, 50];
        assert_eq!(percentile(&samples, 0.50), 30);
        assert_eq!(percentile(&samples, 0.95), 50);
        assert_eq!(percentile(&samples, 0.99), 50);
    }

    #[test]
    fn unsupported_sweep_sizes_are_rejected() {
        let result = parse_args(&[
            "--store-size".to_owned(), "100".to_owned(),
        ]);
        assert!(result.is_err());
        let result = parse_args(&[
            "--batch-size".to_owned(), "3".to_owned(),
        ]);
        assert!(result.is_err());
    }
}
