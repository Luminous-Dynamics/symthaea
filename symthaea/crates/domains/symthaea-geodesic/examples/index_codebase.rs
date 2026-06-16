// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Index Symthaea's own codebase into the ProgramManifold.
//!
//! Demonstrates real-world manifold population and clustering analysis.
//!
//! Run: `cargo run -p symthaea-geodesic --example index_codebase`
//! Or with a specific path:
//!   `cargo run -p symthaea-geodesic --example index_codebase -- /path/to/rust/project`

use std::path::PathBuf;
use symthaea_geodesic::codebase_bridge::index_directory;
use symthaea_geodesic::manifold::ProgramManifold;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        // Default: index symthaea-geodesic's own source
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src")
    };

    let max_files: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(500);

    println!("=== Codebase Indexing into ProgramManifold ===");
    println!("Root: {}", root.display());
    println!("Max files: {}\n", max_files);

    let mut manifold = ProgramManifold::new();
    let result = index_directory(&mut manifold, &root, max_files);

    println!("=== Indexing Results ===");
    println!("Files scanned:     {}", result.files_scanned);
    println!("Functions found:   {}", result.functions_found);
    println!("Functions indexed: {}", result.functions_indexed);
    println!("Fibers created:    {}", result.fibers_created);
    println!("Total fibers:      {}", manifold.fiber_count());
    println!("Total points:      {}", manifold.total_points());
    if !result.errors.is_empty() {
        println!("Errors:            {}", result.errors.len());
        for e in result.errors.iter().take(5) {
            println!("  - {}", e);
        }
    }

    // Analyze fiber distribution
    println!("\n=== Fiber Analysis ===");
    println!(
        "Clustering ratio: {:.1}% (lower = better clustering, 100% = no clustering)",
        if manifold.total_points() > 0 {
            manifold.fiber_count() as f64 / manifold.total_points() as f64 * 100.0
        } else {
            0.0
        }
    );

    // Query: what does the manifold think "sort" looks like?
    println!("\n=== Query Tests ===");
    let test_queries = ["sort", "search", "parse", "encode", "validate"];

    for query in &test_queries {
        let query_hv = symthaea_core::hdc::binary_hv::BinaryHV::random(
            query
                .bytes()
                .fold(0u64, |a, b| a.wrapping_mul(31).wrapping_add(b as u64)),
        );
        match manifold.nearest_fiber(&query_hv) {
            Some(fiber) => {
                let names: Vec<&str> = fiber
                    .points
                    .iter()
                    .take(3)
                    .map(|p| p.name.as_str())
                    .collect();
                let has_source = fiber.points.iter().filter(|p| p.source.is_some()).count();
                println!(
                    "  '{}' → fiber with {} points ({} with source): {:?}",
                    query,
                    fiber.points.len(),
                    has_source,
                    names,
                );
            }
            None => {
                println!("  '{}' → no match (empty manifold)", query);
            }
        }
    }

    println!("\n=== Done ===");
}
