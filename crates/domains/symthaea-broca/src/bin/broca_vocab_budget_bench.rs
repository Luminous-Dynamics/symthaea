// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Measures Broca's current full-vocabulary cosine scoring cost.
//!
//! This is a performance/footprint baseline, not a vocabulary-quality benchmark.
//! It calls the controller's real compute_logits method with deterministic queries,
//! including whichever scoring backend the build enables. Run in release mode.
//!
//! Usage:
//!   cargo run --release -p symthaea-broca --bin broca-vocab-budget-bench
//!   cargo run --release -p symthaea-broca --bin broca-vocab-budget-bench -- --json-out budget.json

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::error::Error;
use std::hint::black_box;
use std::time::Instant;
use symthaea_broca::controller::{LanguageController, LanguageControllerConfig};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::{ContinuousHV, HDC_DIMENSION};

const VOCAB_SIZES: [usize; 5] = [64, 256, 512, 1024, 2048];
const QUERY_COUNT: usize = 4;
const WARMUP_ROUNDS: usize = 2;
const MEASURED_ROUNDS: usize = 9;
const GENESIS_PHRASE: &str = "broca-vocab-budget-bench-v1";

#[derive(Debug, Serialize)]
struct BudgetPoint {
    vocab_size: usize,
    raw_embedding_bytes_lower_bound: usize,
    raw_embedding_mib_lower_bound: f64,
    measured_query_count: usize,
    measured_rounds: usize,
    p50_ms_per_query: f64,
    p95_ms_per_query: f64,
    min_ms_per_query: f64,
    max_ms_per_query: f64,
    tokens_scored_per_second_at_p50: f64,
    embedding_coordinates_scored_per_second_at_p50: f64,
    samples_ms_per_query: Vec<f64>,
}

#[derive(Debug, Serialize)]
struct Receipt {
    schema_version: u32,
    benchmark_id: &'static str,
    evidence_level: &'static str,
    source_revision: Option<&'static str>,
    genesis_phrase: &'static str,
    seed_scheme: &'static str,
    embedding_dimension: usize,
    bytes_per_coordinate: usize,
    query_count_per_round: usize,
    warmup_rounds: usize,
    measured_rounds: usize,
    build_mode: &'static str,
    operating_system: &'static str,
    architecture: &'static str,
    logical_cpu_count: Option<usize>,
    scoring_backend_mode: &'static str,
    rayon_parallel_feature_enabled: bool,
    candle_logits_feature_enabled: bool,
    fixture_spec_sha256: String,
    results: Vec<BudgetPoint>,
    claim_boundary: &'static str,
}

fn controller_for_vocab(genesis: &GenesisSeed, vocab_size: usize) -> LanguageController {
    let config = LanguageControllerConfig {
        network_layers: 1,
        neurons_per_layer: 2,
        vocab_size,
        max_seq_len: 8,
        logit_scale: 20.0,
        ..Default::default()
    };
    LanguageController::new(genesis, &config)
}

fn percentile_nearest_rank(sorted: &[f64], percentile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (percentile * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

fn benchmark_point(
    genesis: &GenesisSeed,
    queries: &[ContinuousHV],
    vocab_size: usize,
) -> BudgetPoint {
    let controller = controller_for_vocab(genesis, vocab_size);

    for _ in 0..WARMUP_ROUNDS {
        for query in queries {
            black_box(controller.compute_logits(black_box(query)));
        }
    }

    let mut samples_ms_per_query = Vec::with_capacity(MEASURED_ROUNDS);
    for _ in 0..MEASURED_ROUNDS {
        let start = Instant::now();
        for query in queries {
            black_box(controller.compute_logits(black_box(query)));
        }
        let elapsed = start.elapsed().as_secs_f64() * 1_000.0;
        samples_ms_per_query.push(elapsed / queries.len().max(1) as f64);
    }

    let mut sorted = samples_ms_per_query.clone();
    sorted.sort_by(f64::total_cmp);
    let p50_ms_per_query = percentile_nearest_rank(&sorted, 0.50);
    let p95_ms_per_query = percentile_nearest_rank(&sorted, 0.95);
    let raw_embedding_bytes_lower_bound =
        vocab_size * HDC_DIMENSION * std::mem::size_of::<f32>();
    let mib = raw_embedding_bytes_lower_bound as f64 / (1024.0 * 1024.0);
    let seconds = (p50_ms_per_query / 1_000.0).max(f64::MIN_POSITIVE);

    BudgetPoint {
        vocab_size,
        raw_embedding_bytes_lower_bound,
        raw_embedding_mib_lower_bound: mib,
        measured_query_count: queries.len(),
        measured_rounds: MEASURED_ROUNDS,
        p50_ms_per_query,
        p95_ms_per_query,
        min_ms_per_query: sorted.first().copied().unwrap_or(0.0),
        max_ms_per_query: sorted.last().copied().unwrap_or(0.0),
        tokens_scored_per_second_at_p50: vocab_size as f64 / seconds,
        embedding_coordinates_scored_per_second_at_p50:
            (vocab_size * HDC_DIMENSION) as f64 / seconds,
        samples_ms_per_query,
    }
}

fn build_mode() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

fn scoring_backend_mode() -> &'static str {
    if cfg!(any(feature = "mamba-cpu", feature = "gpu-logits")) {
        "candle-enabled; active device/fallback is not exposed by this receipt"
    } else if cfg!(feature = "parallel") {
        "rayon-cpu-above-parallel-threshold-otherwise-scalar"
    } else {
        "scalar-cpu"
    }
}

fn fixture_digest() -> String {
    let fixture = [
        format!("schema=1;genesis={GENESIS_PHRASE}"),
        format!("dimension={HDC_DIMENSION};vocab_sizes={VOCAB_SIZES:?}"),
        format!("queries={QUERY_COUNT};warmup={WARMUP_ROUNDS}"),
        format!("rounds={MEASURED_ROUNDS};build_mode={}", build_mode()),
        format!("backend={}", scoring_backend_mode()),
    ]
    .join(";");
    format!("{:x}", Sha256::digest(fixture.as_bytes()))
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    let json_out = match args.iter().position(|arg| arg == "--json-out") {
        Some(index) => Some(
            args.get(index + 1)
                .filter(|path| !path.trim().is_empty())
                .ok_or("--json-out requires a non-empty path")?
                .clone(),
        ),
        None => None,
    };

    let genesis = GenesisSeed::from_phrase(GENESIS_PHRASE);
    let queries: Vec<ContinuousHV> = (0..QUERY_COUNT)
        .map(|index| {
            ContinuousHV::from_genesis(
                &genesis,
                &format!("broca-vocab-budget-query::{index}"),
                HDC_DIMENSION,
            )
        })
        .collect();
    let results: Vec<BudgetPoint> = VOCAB_SIZES
        .iter()
        .map(|vocab_size| benchmark_point(&genesis, &queries, *vocab_size))
        .collect();

    let receipt = Receipt {
        schema_version: 1,
        benchmark_id: "broca-vocab-budget-scoring-v1",
        evidence_level: "measured_local_runtime",
        source_revision: option_env!("GIT_COMMIT_SHA"),
        genesis_phrase: GENESIS_PHRASE,
        seed_scheme: "GenesisSeed::from_phrase with named deterministic query domains",
        embedding_dimension: HDC_DIMENSION,
        bytes_per_coordinate: std::mem::size_of::<f32>(),
        query_count_per_round: QUERY_COUNT,
        warmup_rounds: WARMUP_ROUNDS,
        measured_rounds: MEASURED_ROUNDS,
        build_mode: build_mode(),
        operating_system: std::env::consts::OS,
        architecture: std::env::consts::ARCH,
        logical_cpu_count: std::thread::available_parallelism().ok().map(usize::from),
        scoring_backend_mode: scoring_backend_mode(),
        rayon_parallel_feature_enabled: cfg!(feature = "parallel"),
        candle_logits_feature_enabled: cfg!(any(feature = "mamba-cpu", feature = "gpu-logits")),
        fixture_spec_sha256: fixture_digest(),
        results,
        claim_boundary: "Measures current local compute_logits cost; no semantic-quality or cross-hardware SOTA claim.",
    };

    let json = serde_json::to_string_pretty(&receipt)?;
    if let Some(path) = json_out {
        std::fs::write(&path, &json)?;
        eprintln!("wrote receipt to {path}");
    }

    println!("{json}");
    Ok(())
}
