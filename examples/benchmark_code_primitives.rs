// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Benchmark: Code Tier Primitives Performance
//!
//! Tests the consciousness-aware code understanding system by:
//! 1. Primitive selection speed
//! 2. Primitive composition quality (Phi scores)
//! 3. Code generator with primitive integration
//! 4. Self-analysis on Symthaea's own source
//!
//! Run: cargo run --features code_generation --example benchmark_code_primitives

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use symthaea::cognitive_loop::routing::CodeTaskDetector;
use symthaea::consciousness::code_primitives::{
    CodeOperation, CodePrimitiveExecutor, CodePrimitiveRouter,
};
use symthaea::hdc::code_encoder::CodeHDEncoder;
use symthaea::language::code_generator::CodeGenerator;
use symthaea::language::code_intent::{CodeIntent, CodeSpec, CodeTarget};
use symthaea::language::code_parser::{CodeEntity, CodeParser, EntityKind, ParsedCode, Span};
use symthaea::language::rust_parser::RustParser;
use symthaea::meta::code_health::CodeHealthScanner;
use symthaea::meta::self_analysis::SelfAnalyzer;
use symthaea::mind::structured_thought::EpistemicStatus;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║       Symthaea Code Tier Primitives Benchmark v2              ║");
    println!("║       HDC+CfC thinks about code. Primitives provide vocab.    ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    // Benchmark 1: Primitive Router Performance
    benchmark_primitive_router();

    // Benchmark 2: Primitive Executor Phi Scores
    benchmark_phi_scores();

    // Benchmark 3: Code Task Detection (improved)
    benchmark_task_detection();

    // Benchmark 4: Code Generator with Primitives
    benchmark_code_generator();

    // Benchmark 5: Self-Analysis
    benchmark_self_analysis();

    // Benchmark 6: Cross-Tier Composition
    benchmark_cross_tier();

    // Benchmark 7: Code Health Scanner
    benchmark_code_health();

    // Benchmark 8: Real Rust Parsing with Primitives
    benchmark_real_parsing();

    println!("\n✅ All benchmarks complete!");
}

fn benchmark_primitive_router() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Benchmark 1: Primitive Router Performance");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let operations = [
        CodeOperation::Parse,
        CodeOperation::Encode,
        CodeOperation::Generate,
        CodeOperation::Modify,
        CodeOperation::Explain,
        CodeOperation::FindSimilar,
        CodeOperation::Refactor,
        CodeOperation::Debug,
        CodeOperation::Verify,
    ];

    let mut router = CodePrimitiveRouter::new(16384);

    // Warm up: cache primitives
    let start = Instant::now();
    router.cache_primitives();
    let cache_time = start.elapsed();
    println!("Primitive cache time: {:?}", cache_time);

    // Benchmark each operation
    let mut total_time = std::time::Duration::ZERO;
    let iterations = 1000;

    for op in &operations {
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = router.select_primitives(*op);
        }
        let elapsed = start.elapsed();
        total_time += elapsed;
        let per_op = elapsed.as_nanos() / iterations;
        println!(
            "  {:?}: {} ns/op ({} primitives)",
            op,
            per_op,
            router.select_primitives(*op).len()
        );
    }

    let avg_per_op = total_time.as_nanos() / (iterations * operations.len() as u128);
    println!("\n📊 Average primitive selection: {} ns/op", avg_per_op);
    println!(
        "   Total operations tested: {}",
        operations.len() * iterations as usize
    );
}

fn benchmark_phi_scores() {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Benchmark 2: Primitive Executor Phi Scores");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let executor = CodePrimitiveExecutor::new(16384);

    let operations = [
        ("Parse", CodeOperation::Parse),
        ("Encode", CodeOperation::Encode),
        ("Generate", CodeOperation::Generate),
        ("Modify", CodeOperation::Modify),
        ("Explain", CodeOperation::Explain),
        ("FindSimilar", CodeOperation::FindSimilar),
        ("Refactor", CodeOperation::Refactor),
        ("Debug", CodeOperation::Debug),
        ("Verify", CodeOperation::Verify),
    ];

    let mut phi_scores = HashMap::new();
    let mut total_phi = 0.0f32;

    for (name, op) in &operations {
        let result = executor.execute(*op);
        phi_scores.insert(*name, result.phi);
        total_phi += result.phi;

        let bar_len = (result.phi * 40.0) as usize;
        let bar = "█".repeat(bar_len);
        let empty = "░".repeat(40 - bar_len);

        println!(
            "  {:12} Φ={:.4} |{}{}| {} primitives",
            name,
            result.phi,
            bar,
            empty,
            result.primitives.len()
        );
    }

    let avg_phi = total_phi / operations.len() as f32;
    println!("\n📊 Average Φ across operations: {:.4}", avg_phi);

    // Find highest and lowest
    let highest = phi_scores
        .iter()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap());
    let lowest = phi_scores
        .iter()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap());

    if let (Some((h_name, h_val)), Some((l_name, l_val))) = (highest, lowest) {
        println!("   Highest Φ: {} ({:.4})", h_name, h_val);
        println!("   Lowest Φ:  {} ({:.4})", l_name, l_val);
    }
}

fn benchmark_task_detection() {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Benchmark 3: Code Task Detection");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let detector = CodeTaskDetector::new();

    let test_cases = [
        ("Write a function to sort numbers", true),
        ("What's the weather like today?", false),
        ("Debug this Rust code:\n```rust\nfn main() {}\n```", true),
        ("Explain how struct works in Rust", true),
        ("Tell me about philosophy", false),
        (
            "Refactor the parse_input function to be more efficient",
            true,
        ),
        ("What is 2 + 2?", false),
        ("Create a Python class for database connections", true),
        ("Fix the bug in main.rs", true),
        ("How does photosynthesis work?", false),
    ];

    let mut correct = 0;

    for (input, expected) in &test_cases {
        let (detected, confidence) = detector.detect(input);
        let status = if detected == *expected { "✓" } else { "✗" };
        if detected == *expected {
            correct += 1;
        }

        println!(
            "  {} [{:.2}] \"{}...\" → {:?}",
            status,
            confidence,
            &input[..input.len().min(35)],
            detector.detect_task_type(input)
        );
    }

    let accuracy = correct as f32 / test_cases.len() as f32 * 100.0;
    println!(
        "\n📊 Detection accuracy: {:.1}% ({}/{})",
        accuracy,
        correct,
        test_cases.len()
    );
}

fn benchmark_code_generator() {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Benchmark 4: Code Generator with Primitives");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let generator = CodeGenerator::new(CodeHDEncoder::new(512));

    let intents = [
        (
            "Rust function",
            CodeIntent::Create {
                target: CodeTarget::new("binary_search", EntityKind::Function),
                spec: CodeSpec::new("rust", "binary_search", "Binary search in a sorted array")
                    .with_signature(
                        "fn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize>",
                    )
                    .with_epistemic(EpistemicStatus::Certain),
            },
        ),
        (
            "Python class",
            CodeIntent::Create {
                target: CodeTarget::new("DataLoader", EntityKind::Class),
                spec: CodeSpec::new("python", "DataLoader", "Load and preprocess data")
                    .with_epistemic(EpistemicStatus::Probable),
            },
        ),
        (
            "Debug request",
            CodeIntent::Debug {
                target: CodeTarget::new("parse_config", EntityKind::Function)
                    .with_path("src/config.rs"),
                symptoms: vec![
                    "Panic on empty input".to_string(),
                    "Wrong default values".to_string(),
                ],
            },
        ),
    ];

    let context = symthaea::language::code_generator::CodeContext::default();

    for (name, intent) in &intents {
        let start = Instant::now();
        let result = generator.generate(intent, &context);
        let elapsed = start.elapsed();

        println!("  {} (Φ={:.4}, {:?}):", name, result.phi_score, elapsed);
        println!("    Primitives: {:?}", result.primitives_used);
        println!(
            "    Source preview: {}...",
            result
                .source
                .lines()
                .next()
                .unwrap_or("[empty]")
                .chars()
                .take(50)
                .collect::<String>()
        );
        println!();
    }
}

fn benchmark_self_analysis() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Benchmark 5: Self-Analysis (Symthaea reads itself)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut analyzer = SelfAnalyzer::new(512);

    // Create some mock parsed files representing Symthaea modules
    let modules = [
        (
            "src/consciousness/code_primitives.rs",
            vec!["CodePrimitiveRouter", "CodePrimitiveExecutor", "execute"],
            vec!["CodeExecutionResult"],
        ),
        (
            "src/hdc/code_encoder.rs",
            vec!["encode_entity", "encode_function", "encode_module"],
            vec!["CodeHDEncoder"],
        ),
        (
            "src/language/code_generator.rs",
            vec!["generate", "generate_create", "generate_modify"],
            vec!["CodeGenerator", "GeneratedCode"],
        ),
        (
            "src/meta/self_analysis.rs",
            vec![
                "introspect_module",
                "consciousness_map",
                "operation_phi_scores",
            ],
            vec!["SelfAnalyzer", "ModuleInsight"],
        ),
        (
            "src/cognitive_loop/routing.rs",
            vec!["route", "detect"],
            vec!["ThalamicRouter", "CodeTaskDetector"],
        ),
    ];

    let start = Instant::now();

    for (path, funcs, types) in &modules {
        let parsed = make_mock_parsed(funcs, types);
        analyzer.index_file(Path::new(path), &parsed);
    }

    let index_time = start.elapsed();
    let model = analyzer.build_self_model();

    println!(
        "  Indexed {} modules in {:?}",
        model.module_count, index_time
    );
    println!(
        "  Functions: {}, Types: {}",
        model.function_count, model.type_count
    );
    println!("  Codebase coherence: {:.4}", model.coherence);

    // Operation Phi scores
    println!("\n  Operation Phi scores:");
    let phi_scores = analyzer.operation_phi_scores();
    for (op, phi) in &phi_scores {
        println!("    {}: {:.4}", op, phi);
    }

    // Consciousness map
    println!("\n  Consciousness map (integration per module):");
    let map = analyzer.consciousness_map();
    for (path, integration) in &map {
        println!("    {}: {:.4}", path.display(), integration);
    }

    // Find least integrated
    println!("\n  Introspecting modules...");
    let insights = analyzer.find_least_integrated();
    if let Some(least) = insights.first() {
        println!(
            "    Least integrated: {} (Φ={:.4})",
            least.path.display(),
            least.integration
        );
    }
    if let Some(most) = insights.last() {
        println!(
            "    Most integrated:  {} (Φ={:.4})",
            most.path.display(),
            most.integration
        );
    }
}

fn make_mock_parsed(funcs: &[&str], types: &[&str]) -> ParsedCode {
    let span = Span {
        start_byte: 0,
        end_byte: 100,
        start_line: 0,
        start_col: 0,
        end_line: 10,
        end_col: 0,
    };

    let mut parsed = ParsedCode::new("", "rust");
    for f in funcs {
        parsed
            .entities
            .push(CodeEntity::new(EntityKind::Function, *f, span.clone()));
    }
    for t in types {
        parsed
            .entities
            .push(CodeEntity::new(EntityKind::Struct, *t, span.clone()));
    }
    parsed
}

fn benchmark_cross_tier() {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Benchmark 6: Cross-Tier Composition");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let executor = CodePrimitiveExecutor::new(16384);

    let operations = [
        ("Parse", CodeOperation::Parse),
        ("Generate", CodeOperation::Generate),
        ("Debug", CodeOperation::Debug),
        ("Refactor", CodeOperation::Refactor),
    ];

    println!("  Code + Consciousness tier composition:");
    for (name, op) in &operations {
        let start = Instant::now();
        let result = executor.execute_with_consciousness(*op);
        let elapsed = start.elapsed();

        println!(
            "    {:12} Code Φ={:.4}, Cross Φ={:.4}, Combined Φ={:.4} ({:?})",
            name, result.code_phi, result.cross_phi, result.combined_phi, elapsed
        );
    }

    println!("\n  Code + Metacognitive tier composition:");
    for (name, op) in &operations {
        let start = Instant::now();
        let result = executor.execute_with_metacognitive(*op);
        let elapsed = start.elapsed();

        println!(
            "    {:12} Code Φ={:.4}, Cross Φ={:.4}, Combined Φ={:.4} ({:?})",
            name, result.code_phi, result.cross_phi, result.combined_phi, elapsed
        );
    }

    // Average combined Phi across all compositions
    let consciousness_phis: Vec<f32> = operations
        .iter()
        .map(|(_, op)| executor.execute_with_consciousness(*op).combined_phi)
        .collect();
    let metacog_phis: Vec<f32> = operations
        .iter()
        .map(|(_, op)| executor.execute_with_metacognitive(*op).combined_phi)
        .collect();

    let avg_consciousness =
        consciousness_phis.iter().sum::<f32>() / consciousness_phis.len() as f32;
    let avg_metacog = metacog_phis.iter().sum::<f32>() / metacog_phis.len() as f32;

    println!("\n📊 Average combined Φ:");
    println!("   Code + Consciousness: {:.4}", avg_consciousness);
    println!("   Code + Metacognitive: {:.4}", avg_metacog);
}

fn benchmark_code_health() {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Benchmark 7: Code Health Scanner (Phi as one factor)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let scanner = CodeHealthScanner::new(16384);

    // Create test modules with varying health
    let test_cases = [
        (
            "Well-structured module",
            vec![
                ("parse_config", "Config"),
                ("validate_config", "Config"),
                ("load_config", "Config"),
                ("save_config", "Config"),
            ],
        ),
        (
            "Mixed cohesion module",
            vec![
                ("parse_json", "JsonData"),
                ("send_email", "EmailClient"),
                ("calculate_tax", "TaxResult"),
                ("render_html", "HtmlOutput"),
            ],
        ),
        ("Simple module", vec![("main", "Result")]),
        (
            "Complex module",
            vec![
                ("init", "State"),
                ("process", "State"),
                ("validate", "State"),
                ("transform", "Output"),
                ("serialize", "Bytes"),
                ("compress", "Bytes"),
                ("encrypt", "Encrypted"),
                ("send", "Response"),
                ("log", "LogEntry"),
                ("cleanup", "Result"),
            ],
        ),
    ];

    for (name, entities) in &test_cases {
        let parsed = make_test_parsed(entities);
        let start = Instant::now();
        let health = scanner.scan_module(Path::new("test.rs"), &parsed);
        let elapsed = start.elapsed();

        println!("  {} ({:?}):", name, elapsed);
        println!("    Overall Score: {:.2}/100", health.overall_score * 100.0);
        println!("    Factors:");
        println!("      Φ Integration: {:.4}", health.factors.phi);
        println!("      Complexity:    {:.4}", health.factors.complexity);
        println!("      Cohesion:      {:.4}", health.factors.cohesion);
        println!("      Density:       {:.4}", health.factors.density);
        println!("      Structure:     {:.4}", health.factors.structure);
        println!("      Primitive Φ:   {:.4}", health.factors.primitive_psi);

        if !health.suggestions.is_empty() {
            println!(
                "    Suggestions: {:?}",
                &health.suggestions[..health.suggestions.len().min(2)]
            );
        }
        println!();
    }

    // Quick scan performance
    let iterations = 1000;
    let parsed = make_test_parsed(&test_cases[0].1);
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = scanner.quick_scan(&parsed);
    }
    let elapsed = start.elapsed();
    let per_scan = elapsed.as_nanos() / iterations;

    println!("📊 Quick scan performance: {} ns/scan", per_scan);
}

fn make_test_parsed(entities: &[(&str, &str)]) -> ParsedCode {
    let span = Span {
        start_byte: 0,
        end_byte: 100,
        start_line: 0,
        start_col: 0,
        end_line: 10,
        end_col: 0,
    };

    let mut parsed = ParsedCode::new("", "rust");
    for (func_name, type_name) in entities {
        parsed.entities.push(CodeEntity::new(
            EntityKind::Function,
            *func_name,
            span.clone(),
        ));
        parsed.entities.push(CodeEntity::new(
            EntityKind::Struct,
            *type_name,
            span.clone(),
        ));
    }
    parsed
}

fn benchmark_real_parsing() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Benchmark 8: Real Rust Parsing with Primitives");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Parse a real Symthaea source file
    let test_files = [
        "src/hdc/real_hv.rs",
        "src/consciousness/integrated_information.rs",
        "src/dynamics/cfc.rs",
        "src/language/code_parser.rs",
    ];

    let mut parser = RustParser::new();
    let encoder = CodeHDEncoder::new(512);

    for file_path in &test_files {
        let full_path = Path::new(file_path);

        // Try to read the file
        match std::fs::read_to_string(full_path) {
            Ok(source) => {
                let start = Instant::now();

                // Parse with tree-sitter
                match parser.parse(&source) {
                    Ok(parsed) => {
                        let parse_time = start.elapsed();

                        // Encode to HDC
                        let encode_start = Instant::now();
                        let module_hv = encoder.encode_module(&parsed);
                        let encode_time = encode_start.elapsed();

                        let loc = source.lines().count();
                        let entities = parsed.entities.len();
                        let funcs = parsed
                            .entities
                            .iter()
                            .filter(|e| matches!(e.kind, EntityKind::Function))
                            .count();
                        let types = parsed
                            .entities
                            .iter()
                            .filter(|e| {
                                matches!(
                                    e.kind,
                                    EntityKind::Struct | EntityKind::Enum | EntityKind::Trait
                                )
                            })
                            .count();

                        println!("  {}:", file_path);
                        println!(
                            "    {} LOC, {} entities ({} funcs, {} types)",
                            loc, entities, funcs, types
                        );
                        println!("    Parse: {:?}, Encode: {:?}", parse_time, encode_time);
                        println!(
                            "    HDC dim: {}, norm: {:.4}",
                            module_hv.dim(),
                            module_hv.norm()
                        );
                        println!();
                    }
                    Err(e) => {
                        println!("  {}: Parse error - {}", file_path, e);
                    }
                }
            }
            Err(_) => {
                println!(
                    "  {}: File not found (running from wrong directory?)",
                    file_path
                );
            }
        }
    }

    // Test primitive-aware parsing on inline source
    println!("  Primitive-aware parsing test:");
    let test_source = r#"
        pub fn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {
            let mut lo = 0;
            let mut hi = arr.len();
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                match arr[mid].cmp(target) {
                    std::cmp::Ordering::Less => lo = mid + 1,
                    std::cmp::Ordering::Greater => hi = mid,
                    std::cmp::Ordering::Equal => return Some(mid),
                }
            }
            None
        }
    "#;

    let start = Instant::now();
    match parser.parse_with_primitives(test_source, 16384) {
        Ok(result) => {
            let elapsed = start.elapsed();
            println!("    Inline source parsed in {:?}", elapsed);
            println!("    Entities: {}", result.parsed.entities.len());
            println!("    Primitive Φ: {:.4}", result.primitive_result.phi);
            println!(
                "    Primitives used: {:?}",
                result
                    .primitive_result
                    .primitives
                    .iter()
                    .map(|p| p.primitive.name.as_str())
                    .collect::<Vec<_>>()
            );
        }
        Err(e) => {
            println!("    Parse error: {}", e);
        }
    }
}
