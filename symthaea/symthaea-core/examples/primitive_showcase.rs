// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Primitive System Showcase
//!
//! Demonstrates the full primitive system pipeline for symthaea-core:
//! system summary, lookup, composition, similarity search, algebra,
//! graphs, caching, and persistence.
//!
//! Run with:
//!   cargo run --example primitive_showcase

use symthaea_core::hdc::primitive_system::{
    CompositionAlgebra, CompositionCache, HistoryEntry, PrimitiveGraph, PrimitivePersistence,
    PrimitiveSystem, PrimitiveTier,
};

fn main() {
    // ========================================================================
    // 1. PRIMITIVE SYSTEM SUMMARY
    // ========================================================================
    println!("================================================================");
    println!("  SECTION 1: Primitive System Summary");
    println!("================================================================\n");

    let system = PrimitiveSystem::global();

    println!("Total primitives: {}", system.count());
    println!();

    let tiers = [
        PrimitiveTier::NSM,
        PrimitiveTier::Mathematical,
        PrimitiveTier::Physical,
        PrimitiveTier::Geometric,
        PrimitiveTier::Strategic,
        PrimitiveTier::MetaCognitive,
        PrimitiveTier::Temporal,
        PrimitiveTier::Compositional,
        PrimitiveTier::Consciousness,
    ];

    println!("Primitives by tier:");
    for tier in &tiers {
        let count = system.count_tier(*tier);
        if count > 0 {
            println!("  {:?}: {} primitives", tier, count);
        }
    }
    println!();

    // ========================================================================
    // 2. LOOK UP INDIVIDUAL PRIMITIVES
    // ========================================================================
    println!("================================================================");
    println!("  SECTION 2: Primitive Lookup");
    println!("================================================================\n");

    let names_to_inspect = ["CAUSE", "ENERGY", "POINT", "IDENTITY", "QUALE"];

    for name in &names_to_inspect {
        if let Some(prim) = system.get(name) {
            println!("Primitive: {}", prim.name);
            println!("  Tier:       {:?}", prim.tier);
            println!("  Domain:     {}", prim.domain);
            println!("  Definition: {}", prim.definition);
            println!("  Base:       {}", prim.is_base);
            println!();
        } else {
            println!("Primitive '{}' not found.\n", name);
        }
    }

    // ========================================================================
    // 3. COMPOSE PRIMITIVES
    // ========================================================================
    println!("================================================================");
    println!("  SECTION 3: Primitive Composition");
    println!("================================================================\n");

    // 3a. Bind: CAUSE ^ EFFECT
    println!("--- Bind: CAUSE ^ EFFECT ---");
    match system.bind_primitives("CAUSE", "EFFECT") {
        Ok(result) => {
            println!("  Operation: {}", result.operation);
            println!("  Sources:   {:?}", result.source_primitives);
            let similar = result.find_similar(system, 3);
            println!("  Most similar primitives:");
            for (name, sim) in &similar {
                println!("    {}: {:.4}", name, sim);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // 3b. Bundle: MASS + ENERGY + FORCE
    println!("--- Bundle: MASS + ENERGY + FORCE ---");
    match system.bundle_primitives(&["MASS", "ENERGY", "FORCE"]) {
        Ok(result) => {
            println!("  Operation: {}", result.operation);
            println!("  Sources:   {:?}", result.source_primitives);
            let similar = result.find_similar(system, 3);
            println!("  Most similar primitives:");
            for (name, sim) in &similar {
                println!("    {}: {:.4}", name, sim);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // 3c. Sequence: BEFORE > DURING > AFTER
    println!("--- Sequence: BEFORE > DURING > AFTER ---");
    match system.encode_sequence(&["BEFORE", "DURING", "AFTER"]) {
        Ok(result) => {
            println!("  Operation: {}", result.operation);
            println!("  Sources:   {:?}", result.source_primitives);
            let similar = result.find_similar(system, 3);
            println!("  Most similar primitives:");
            for (name, sim) in &similar {
                println!("    {}: {:.4}", name, sim);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // 3d. Weighted bundle: CARE-dominant ethical blend
    println!("--- Weighted Bundle: CARE:0.6 + FAIRNESS:0.3 + OBLIGATION:0.1 ---");
    match system.bundle_weighted(&[("CARE", 0.6), ("FAIRNESS", 0.3), ("OBLIGATION", 0.1)]) {
        Ok(result) => {
            println!("  Operation: {}", result.operation);
            println!("  Sources:   {:?}", result.source_primitives);
            let similar = result.find_similar(system, 3);
            println!("  Most similar primitives:");
            for (name, sim) in &similar {
                println!("    {}: {:.4}", name, sim);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // 3e. Analogy: CAUSE:EFFECT :: BEFORE:?
    println!("--- Analogy: CAUSE is to EFFECT as BEFORE is to ? ---");
    match system.analogy("CAUSE", "EFFECT", "BEFORE") {
        Ok(result) => {
            println!("  Operation: {}", result.operation);
            let similar = result.find_similar(system, 5);
            println!("  Best analogical completions:");
            for (name, sim) in &similar {
                println!("    {}: {:.4}", name, sim);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // 3f. Permute
    println!("--- Permute: ENERGY shifted by 3 steps ---");
    match system.permute_primitive("ENERGY", 3) {
        Ok(result) => {
            println!("  Operation: {}", result.operation);
            let similar = result.find_similar(system, 3);
            println!("  Most similar after permutation:");
            for (name, sim) in &similar {
                println!("    {}: {:.4}", name, sim);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // ========================================================================
    // 4. SIMILARITY SEARCH (BRUTE FORCE AND LSH)
    // ========================================================================
    println!("================================================================");
    println!("  SECTION 4: Similarity Search");
    println!("================================================================\n");

    // 4a. Brute-force similarity
    println!("--- Brute-force: top 5 similar to CAUSE ---");
    let similar = system.find_similar("CAUSE", 5);
    for (name, sim) in &similar {
        println!("  {}: {:.4}", name, sim);
    }
    println!();

    // 4b. Build LSH index and search
    println!("--- LSH index: 10 bands x 64 bits ---");
    let lsh = system.build_lsh_index(10, 64);
    let lsh_stats = lsh.stats();
    println!(
        "  LSH index: {} bands, {} bits/band, {} total buckets",
        lsh_stats.num_bands, lsh_stats.bits_per_band, lsh_stats.total_buckets
    );

    if let Some(cause_prim) = system.get("CAUSE") {
        let lsh_results = system.find_similar_lsh(&cause_prim.encoding, 5, &lsh);
        println!("  Top 5 similar to CAUSE (via LSH):");
        for (name, sim) in &lsh_results {
            println!("    {}: {:.4}", name, sim);
        }
    }
    println!();

    // 4c. Batch similarity
    println!("--- Batch similarity: query MASS and POINT simultaneously ---");
    let queries: Vec<_> = ["MASS", "POINT"]
        .iter()
        .filter_map(|n| system.get(n).map(|p| p.encoding.clone()))
        .collect();
    let batch_results = system.batch_find_similar(&queries, 3);
    for (i, name) in ["MASS", "POINT"].iter().enumerate() {
        if i < batch_results.len() {
            println!("  Top 3 similar to {}:", name);
            for (n, sim) in &batch_results[i] {
                println!("    {}: {:.4}", n, sim);
            }
        }
    }
    println!();

    // 4d. Query: decode an arbitrary encoding back to a primitive
    println!("--- Query: decode a composed encoding ---");
    if let Ok(bound) = system.bind_primitives("MASS", "VELOCITY") {
        let (best_name, best_sim) = system.query(&bound.encoding);
        println!(
            "  bind(MASS, VELOCITY) closest match: {} (similarity: {:.4})",
            best_name, best_sim
        );
    }
    println!();

    // ========================================================================
    // 5. COMPOSITION ALGEBRA
    // ========================================================================
    println!("================================================================");
    println!("  SECTION 5: Composition Algebra");
    println!("================================================================\n");

    let mut algebra = CompositionAlgebra::new();

    // Define named compositions using the expression language
    let definitions = [
        ("CAUSATION", "CAUSE ^ EFFECT"),
        ("TEMPORAL_FLOW", "BEFORE > DURING > AFTER"),
        ("PHYSICS_CORE", "MASS + ENERGY + FORCE"),
        ("ETHICAL_BLEND", "CARE:0.6 + FAIRNESS:0.3 + OBLIGATION:0.1"),
    ];

    for (name, expr) in &definitions {
        match algebra.define(name, expr, system) {
            Ok(()) => println!("  Defined {} = {}", name, expr),
            Err(e) => println!("  Error defining {}: {}", name, e),
        }
    }
    println!();

    // Compose previously defined compositions
    match algebra.define("CAUSAL_FLOW", "CAUSATION ^ TEMPORAL_FLOW", system) {
        Ok(()) => println!("  Defined CAUSAL_FLOW = CAUSATION ^ TEMPORAL_FLOW (higher-order!)"),
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // List all defined compositions
    println!("All defined compositions:");
    let mut compositions: Vec<_> = algebra.list();
    compositions.sort_by_key(|c| c.name.clone());
    for comp in &compositions {
        println!(
            "  {} = {} (sources: {:?})",
            comp.name, comp.expression, comp.sources
        );
    }
    println!();

    // Find what each composition is closest to
    println!("Nearest primitives for each composition:");
    for comp in &compositions {
        let similar = system.find_similar_to_encoding(&comp.encoding, 3);
        print!("  {} -> ", comp.name);
        let matches: Vec<String> = similar
            .iter()
            .map(|(n, s)| format!("{}({:.3})", n, s))
            .collect();
        println!("{}", matches.join(", "));
    }
    println!();

    // ========================================================================
    // 6. SIMILARITY MATRIX
    // ========================================================================
    println!("================================================================");
    println!("  SECTION 6: Similarity Matrix");
    println!("================================================================\n");

    let matrix_names = ["CAUSE", "EFFECT", "BEFORE", "AFTER", "ENERGY"];
    let matrix = system.similarity_matrix(&matrix_names);

    // Print header
    print!("{:>12}", "");
    for name in &matrix_names {
        print!("{:>10}", name);
    }
    println!();

    // Print rows
    for (i, row) in matrix.iter().enumerate() {
        print!("{:>12}", matrix_names[i]);
        for val in row {
            print!("{:>10.4}", val);
        }
        println!();
    }
    println!();

    // ========================================================================
    // 7. GRAPH VISUALIZATION
    // ========================================================================
    println!("================================================================");
    println!("  SECTION 7: Graph Visualization");
    println!("================================================================\n");

    // Build a graph from the Physical tier
    let graph = PrimitiveGraph::from_tier(system, PrimitiveTier::Physical, 0.55);
    let stats = graph.stats();
    println!(
        "Physical tier graph: {} nodes, {} edges, density={:.3}, max_sim={:.4}",
        stats.node_count, stats.edge_count, stats.density, stats.max_similarity
    );
    println!();

    // Print the DOT format (first 30 lines to keep output manageable)
    let dot = graph.to_dot();
    println!("DOT output (first 30 lines):");
    for (i, line) in dot.lines().enumerate() {
        if i >= 30 {
            println!("  ... ({} more lines)", dot.lines().count() - 30);
            break;
        }
        println!("  {}", line);
    }
    println!();

    // ASCII representation
    let ascii = graph.to_ascii();
    println!("ASCII graph:");
    for line in ascii.lines().take(25) {
        println!("  {}", line);
    }
    println!();

    // Also build a small custom graph
    let custom_graph = PrimitiveGraph::from_primitives(
        system,
        &["CAUSE", "EFFECT", "BEFORE", "AFTER", "ENERGY", "FORCE"],
        0.52,
    );
    let custom_stats = custom_graph.stats();
    println!(
        "Custom graph (causal+temporal+physics): {} nodes, {} edges",
        custom_stats.node_count, custom_stats.edge_count
    );
    println!();

    // ========================================================================
    // 8. COMPOSITION CACHE
    // ========================================================================
    println!("================================================================");
    println!("  SECTION 8: Composition Cache");
    println!("================================================================\n");

    let mut cache = CompositionCache::new(100);

    // Perform operations through the cache
    println!("Performing cached operations...");

    // First calls: cache misses
    let _ = cache.bind_cached(system, "CAUSE", "EFFECT");
    let _ = cache.bundle_cached(system, &["MASS", "ENERGY", "FORCE"]);
    let _ = cache.sequence_cached(system, &["BEFORE", "DURING", "AFTER"]);
    let _ = cache.analogy_cached(system, "CAUSE", "EFFECT", "BEFORE");
    let _ = cache.bundle_weighted_cached(system, &[("CARE", 0.7), ("HARM", 0.3)]);

    let stats_after_misses = cache.stats();
    println!(
        "  After first pass: size={}, hits={}, misses={}, hit_rate={:.1}%",
        stats_after_misses.size,
        stats_after_misses.hits,
        stats_after_misses.misses,
        stats_after_misses.hit_rate * 100.0
    );

    // Second calls: cache hits
    let _ = cache.bind_cached(system, "CAUSE", "EFFECT");
    let _ = cache.bundle_cached(system, &["MASS", "ENERGY", "FORCE"]);
    let _ = cache.sequence_cached(system, &["BEFORE", "DURING", "AFTER"]);
    let _ = cache.analogy_cached(system, "CAUSE", "EFFECT", "BEFORE");
    let _ = cache.bundle_weighted_cached(system, &[("CARE", 0.7), ("HARM", 0.3)]);

    let stats_after_hits = cache.stats();
    println!(
        "  After second pass: size={}, hits={}, misses={}, hit_rate={:.1}%",
        stats_after_hits.size,
        stats_after_hits.hits,
        stats_after_hits.misses,
        stats_after_hits.hit_rate * 100.0
    );

    // Third round: mix of hits and new operations
    let _ = cache.bind_cached(system, "CAUSE", "EFFECT"); // hit
    let _ = cache.bind_cached(system, "MASS", "FORCE"); // miss (new pair)
    let _ = cache.bind_cached(system, "CAUSE", "EFFECT"); // hit

    let final_stats = cache.stats();
    println!(
        "  Final stats: size={}, hits={}, misses={}, hit_rate={:.1}%",
        final_stats.size,
        final_stats.hits,
        final_stats.misses,
        final_stats.hit_rate * 100.0
    );
    println!();

    // ========================================================================
    // 9. PERSISTENCE (SAVE/LOAD SESSION)
    // ========================================================================
    println!("================================================================");
    println!("  SECTION 9: Persistence (Save/Load Session)");
    println!("================================================================\n");

    let persistence = PrimitivePersistence::new();

    // Create some history entries
    let history = vec![
        HistoryEntry {
            operation: "bind(CAUSE, EFFECT)".to_string(),
            result_match: "CAUSE".to_string(),
            similarity: 0.51,
        },
        HistoryEntry {
            operation: "bundle(MASS, ENERGY, FORCE)".to_string(),
            result_match: "ENERGY".to_string(),
            similarity: 0.62,
        },
        HistoryEntry {
            operation: "analogy(CAUSE, EFFECT, BEFORE)".to_string(),
            result_match: "AFTER".to_string(),
            similarity: 0.48,
        },
    ];

    // Save to a temp file
    let temp_path = std::env::temp_dir().join("symthaea_showcase_session.json");
    let temp_str = temp_path.to_string_lossy().to_string();

    println!("Saving session to: {}", temp_str);
    match persistence.save_session(&temp_str, &algebra, &history, Some("Showcase demo session")) {
        Ok(()) => println!("  Session saved successfully."),
        Err(e) => println!("  Save error: {}", e),
    }

    // Load it back
    println!("Loading session from: {}", temp_str);
    match persistence.load_session(&temp_str, system) {
        Ok((loaded_algebra, loaded_history)) => {
            let loaded_comps = loaded_algebra.list();
            println!("  Loaded {} compositions:", loaded_comps.len());
            for comp in &loaded_comps {
                println!("    {} = {}", comp.name, comp.expression);
            }
            println!("  Loaded {} history entries:", loaded_history.len());
            for entry in &loaded_history {
                println!(
                    "    {} -> {} ({:.2})",
                    entry.operation, entry.result_match, entry.similarity
                );
            }
        }
        Err(e) => println!("  Load error: {}", e),
    }

    // Clean up temp file
    let _ = std::fs::remove_file(&temp_path);
    println!();

    // ========================================================================
    // DONE
    // ========================================================================
    println!("================================================================");
    println!("  Showcase complete.");
    println!("================================================================");
}
