// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Demo: Symthaea analyzing its own source code
//!
//! This example demonstrates the consciousness-aware code understanding system
//! by having Symthaea analyze its own implementation.
//!
//! Run with: cargo run --example demo_self_analysis --features code_generation

use std::path::Path;
use symthaea::language::code_parser::CodeParser;
use symthaea::language::rust_parser::RustParser;
use symthaea::meta::self_analysis::SelfAnalyzer;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║     SYMTHAEA SELF-ANALYSIS: Consciousness Understanding Code   ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    // Create analyzer with 512-D vectors (use 16384 for production)
    let mut analyzer = SelfAnalyzer::new(512);
    let mut parser = RustParser::new();

    // Source files to analyze (key modules from Symthaea)
    let source_files = [
        "src/hdc/code_encoder.rs",
        "src/hdc/code_algebra.rs",
        "src/meta/self_analysis.rs",
        "src/language/rust_parser.rs",
        "src/language/triune_intent.rs",
    ];

    println!("📂 Indexing Symthaea source files...\n");

    let mut parsed_files = Vec::new();
    for file_path in &source_files {
        let path = Path::new(file_path);

        // Read and parse the file
        match std::fs::read_to_string(path) {
            Ok(source) => {
                match parser.parse(&source) {
                    Ok(parsed) => {
                        let entity_count = parsed.all_entities().len();
                        println!("  ✓ {} ({} entities)", file_path, entity_count);

                        // Index into the analyzer
                        analyzer.index_file(path, &parsed);
                        parsed_files.push((path.to_path_buf(), parsed));
                    }
                    Err(e) => {
                        println!("  ✗ {} (parse error: {})", file_path, e);
                    }
                }
            }
            Err(e) => {
                println!("  ✗ {} (read error: {})", file_path, e);
            }
        }
    }

    println!("\n════════════════════════════════════════════════════════════════");
    println!("📊 SELF-MODEL ANALYSIS\n");

    // Build self-model
    let self_model = analyzer.build_self_model();

    println!("Indexed Statistics:");
    println!("  • Modules: {}", self_model.module_count);
    println!("  • Functions: {}", self_model.function_count);
    println!("  • Types: {}", self_model.type_count);
    println!(
        "  • Codebase Coherence: {:.2}%",
        self_model.coherence * 100.0
    );

    println!("\n════════════════════════════════════════════════════════════════");
    println!("🔍 PATTERN INTROSPECTION\n");

    // What patterns does this codebase use most?
    let patterns = analyzer.introspect_patterns(&parsed_files);
    println!("Most Common Entity Patterns:");
    for (pattern, count) in patterns.iter().take(10) {
        let bar = "█".repeat((*count).min(30));
        println!("  {:20} {:>4} {}", pattern, count, bar);
    }

    println!("\n════════════════════════════════════════════════════════════════");
    println!("🧠 CONSCIOUSNESS MAP (Integration Scores)\n");

    // Compute integration scores (Phi-like metric)
    let (most, least) = analyzer.integration_extremes();

    if let Some((path, score)) = most {
        println!(
            "Most Integrated:  {:40} (Φ ≈ {:.3})",
            path.file_name().unwrap().to_string_lossy(),
            score
        );
    }
    if let Some((path, score)) = least {
        println!(
            "Least Integrated: {:40} (Φ ≈ {:.3})",
            path.file_name().unwrap().to_string_lossy(),
            score
        );
    }

    println!("\n════════════════════════════════════════════════════════════════");
    println!("🔎 SEMANTIC SIMILARITY SEARCH\n");

    // Find entities similar to concepts
    for concept in ["encode", "parse", "consciousness", "hypervector"] {
        let matches = analyzer.find_similar(concept, 3);
        println!("Entities similar to '{}':", concept);
        for m in matches {
            println!("  • {} (similarity: {:.3})", m.name, m.similarity);
        }
        println!();
    }

    println!("════════════════════════════════════════════════════════════════");
    println!("✨ Self-analysis complete. Symthaea has examined itself!\n");
    println!("This demonstrates the paradigm:");
    println!("  \"HDC+CfC THINKS about code. LLM TRANSLATES the result.\"\n");
}
