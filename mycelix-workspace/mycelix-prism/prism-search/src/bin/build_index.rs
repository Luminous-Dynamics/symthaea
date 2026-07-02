// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pre-compute search indices and serialize to disk.
//!
//! Produces two files:
//! - prism-index-core.bin: curated + mycelix claims (~200 claims, <500 KB)
//! - prism-index.bin: full index including Wikidata (~10K claims, ~22 MB)
//!
//! Run: cargo run -p prism-search --bin build_index --release

use prism_search::SearchEngine;

fn main() {
    let base = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "prism-search".to_string());

    // Core index (curated + mycelix only — instant mobile loading)
    let start = std::time::Instant::now();
    let core = SearchEngine::with_core_claims();
    let core_time = start.elapsed();
    let core_bytes = bincode::serialize(&core).expect("Failed to serialize core index");
    let core_path = format!("{}/prism-index-core.bin", base);
    std::fs::write(&core_path, &core_bytes).expect("Failed to write core index");
    println!(
        "Core index: {} claims in {:.2}s — {} bytes ({:.0} KB) → {}",
        core.claim_count(),
        core_time.as_secs_f64(),
        core_bytes.len(),
        core_bytes.len() as f64 / 1024.0,
        core_path
    );

    // Full index (all sources)
    let start = std::time::Instant::now();
    let full = SearchEngine::with_seed_claims();
    let full_time = start.elapsed();
    let full_bytes = bincode::serialize(&full).expect("Failed to serialize full index");
    let full_path = format!("{}/prism-index.bin", base);
    std::fs::write(&full_path, &full_bytes).expect("Failed to write full index");
    println!(
        "Full index: {} claims in {:.2}s — {} bytes ({:.1} MB) → {}",
        full.claim_count(),
        full_time.as_secs_f64(),
        full_bytes.len(),
        full_bytes.len() as f64 / 1_048_576.0,
        full_path
    );
}
