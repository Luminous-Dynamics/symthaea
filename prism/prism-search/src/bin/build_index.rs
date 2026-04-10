// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pre-compute the search index and serialize to disk.
//!
//! Run: cargo run -p prism-search --bin build_index
//! Output: prism-search/prism-index.bin

use prism_search::SearchEngine;

fn main() {
    let start = std::time::Instant::now();
    let engine = SearchEngine::with_seed_claims();
    let build_time = start.elapsed();

    let bytes = bincode::serialize(&engine).expect("Failed to serialize index");

    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "prism-search/prism-index.bin".to_string());

    std::fs::write(&out_path, &bytes).expect("Failed to write index file");

    println!(
        "Index built in {:.2}s — {} bytes ({:.1} MB) written to {}",
        build_time.as_secs_f64(),
        bytes.len(),
        bytes.len() as f64 / 1_048_576.0,
        out_path
    );
}
