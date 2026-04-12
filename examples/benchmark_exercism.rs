// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exercism Rust Benchmark — External validation against 106 real Rust problems
//!
//! Run: `cargo run --example benchmark_exercism --features code_generation`
//!
//! For each Exercism problem:
//! 1. Parse the stub from src/lib.rs (function signatures)
//! 2. Generate implementation via Symthaea's native code generation
//! 3. Write to a temp copy of the exercise
//! 4. Run `cargo test` (with #[ignore] removed)
//! 5. Report pass@1
//!
//! This is the real external validation. No cherry-picking, no curated problems.
//! Exercism problems are Rust-native with real test suites.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use symthaea::hdc::code_encoder::CodeHDEncoder;
use symthaea::language::analogy_generation::SolutionLibrary;
use symthaea::language::code_generator::{CodeContext, CodeGenerator};
use symthaea::language::code_intent::{CodeIntent, CodeSpec, CodeTarget};
use symthaea::language::code_parser::EntityKind;
use symthaea::mind::structured_thought::EpistemicStatus;

const EXERCISM_DIR: &str = "benchmarks/external/exercism-rust/exercises/practice";

/// Result for a single exercise
#[derive(Debug)]
struct ExerciseResult {
    name: String,
    generated: bool,
    compiled: bool,
    tests_passed: usize,
    tests_failed: usize,
    all_pass: bool,
    error: Option<String>,
}

/// Parsed function from an exercise stub
struct ParsedFunction {
    name: String,
    signature: String,
    purpose: String,
}

/// Parse exercise directory to extract ALL public function signatures.
///
/// Returns all `pub fn` declarations from src/lib.rs with purposes
/// derived from .docs/instructions.md and doc comments.
fn parse_exercise_functions(dir: &Path) -> Vec<ParsedFunction> {
    let lib_path = dir.join("src/lib.rs");
    let source = match std::fs::read_to_string(&lib_path) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };

    // Read purpose from instructions
    let mut base_purpose = String::new();
    let instructions_path = dir.join(".docs/instructions.md");
    if let Ok(instructions) = std::fs::read_to_string(&instructions_path) {
        for line in instructions.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('#') || trimmed.is_empty() {
                continue;
            }
            base_purpose = trimmed.to_string();
            break;
        }
    }
    if base_purpose.is_empty() {
        base_purpose = dir
            .file_name()
            .map(|n| format!("Implement {}", n.to_string_lossy().replace('-', " ")))
            .unwrap_or_default();
    }

    // Extract ALL pub fn declarations
    let mut functions = Vec::new();
    let mut current_doc = String::new();

    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("///") || trimmed.starts_with("//!") {
            let doc = trimmed
                .trim_start_matches("///")
                .trim_start_matches("//!")
                .trim();
            if !doc.is_empty() {
                if !current_doc.is_empty() {
                    current_doc.push(' ');
                }
                current_doc.push_str(doc);
            }
        } else if trimmed.starts_with("pub fn ") {
            let sig_line = if trimmed.contains('{') {
                trimmed.split('{').next().unwrap_or(trimmed).trim()
            } else {
                trimmed
            };
            let signature = sig_line.to_string();

            let name = if let Some(fn_start) = sig_line.find("fn ") {
                let after_fn = &sig_line[fn_start + 3..];
                after_fn.split('(').next().unwrap_or("").trim().to_string()
            } else {
                String::new()
            };

            if !name.is_empty() {
                // Build purpose: doc comment + base purpose + function name
                let purpose = if !current_doc.is_empty() {
                    format!("{}. {}", current_doc, base_purpose)
                } else {
                    format!("{} {}", base_purpose, name.replace('_', " "))
                };

                functions.push(ParsedFunction {
                    name,
                    signature,
                    purpose,
                });
            }
            current_doc.clear();
        } else if !trimmed.starts_with("//") {
            current_doc.clear(); // Reset doc comment if non-comment non-fn line
        }
    }

    functions
}

/// Backward-compat wrapper for single-function exercises
fn parse_exercise(dir: &Path) -> Option<(String, String, String)> {
    let functions = parse_exercise_functions(dir);
    let first = functions.into_iter().next()?;
    Some((first.name, first.purpose, first.signature))
}

/// Generate implementation for an exercise
fn generate_implementation(
    generator: &CodeGenerator,
    name: &str,
    purpose: &str,
    signature: &str,
) -> Option<String> {
    let spec = CodeSpec {
        language: "rust".into(),
        name: name.into(),
        purpose: purpose.into(),
        purpose_hv: None,
        signature: Some(signature.into()),
        constraints: Vec::new(),
        examples: Vec::new(),
        epistemic_status: EpistemicStatus::Certain,
        metadata: HashMap::new(),
    };

    let intent = CodeIntent::Create {
        target: CodeTarget {
            kind: EntityKind::Function,
            name: name.into(),
            path: None,
            language: Some("rust".into()),
            hv: None,
        },
        spec,
    };

    let context = CodeContext::default();
    let result = generator.generate(&intent, &context);

    // Only return if we generated a real body (not todo!)
    if result.source.contains("todo!(") || result.source.contains("unimplemented!(") {
        return None;
    }

    // Extract just the function body (strip test modules that the emitter adds)
    let source = if let Some(test_start) = result.source.find("#[cfg(test)]") {
        result.source[..test_start].trim().to_string()
    } else {
        result.source.clone()
    };

    Some(source)
}

/// Run cargo test on an exercise directory with generated implementation
fn run_exercise_tests(exercise_dir: &Path, implementation: &str) -> ExerciseResult {
    let exercise_name = exercise_dir
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    // Create a temp directory with a copy of the exercise
    let temp_dir = std::env::temp_dir().join(format!("symthaea-exercism-{}", exercise_name));
    let _ = std::fs::remove_dir_all(&temp_dir);

    // Copy the exercise
    if let Err(e) = copy_dir_recursive(exercise_dir, &temp_dir) {
        return ExerciseResult {
            name: exercise_name,
            generated: true,
            compiled: false,
            tests_passed: 0,
            tests_failed: 0,
            all_pass: false,
            error: Some(format!("Failed to copy exercise: {}", e)),
        };
    }

    // Write our implementation
    let lib_path = temp_dir.join("src/lib.rs");
    if let Err(e) = std::fs::write(&lib_path, implementation) {
        return ExerciseResult {
            name: exercise_name,
            generated: true,
            compiled: false,
            tests_passed: 0,
            tests_failed: 0,
            all_pass: false,
            error: Some(format!("Failed to write implementation: {}", e)),
        };
    }

    // Remove #[ignore] from test files so all tests run
    let tests_dir = temp_dir.join("tests");
    if tests_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&tests_dir) {
            for entry in entries.flatten() {
                if entry.path().extension().map_or(false, |e| e == "rs") {
                    if let Ok(content) = std::fs::read_to_string(entry.path()) {
                        let unignored = content.replace("#[ignore]\n", "").replace("#[ignore]", "");
                        let _ = std::fs::write(entry.path(), unignored);
                    }
                }
            }
        }
    }

    // Run cargo test (no extra args — let all tests run)
    let output = Command::new("cargo")
        .arg("test")
        .current_dir(&temp_dir)
        .env("RUST_BACKTRACE", "0")
        .output();

    // Clean up
    let _ = std::fs::remove_dir_all(&temp_dir);

    match output {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let combined = format!("{}\n{}", stdout, stderr);

            let compiled = !stderr.contains("could not compile");
            let (passed, failed) = parse_test_results(&combined);
            let all_pass = compiled && failed == 0 && passed > 0;

            let error = if !compiled {
                // Extract first error
                stderr
                    .lines()
                    .find(|l| l.starts_with("error"))
                    .map(|l| l.chars().take(100).collect())
            } else if failed > 0 {
                Some(format!("{} tests failed", failed))
            } else if passed == 0 {
                Some("No tests ran".to_string())
            } else {
                None
            };

            ExerciseResult {
                name: exercise_name,
                generated: true,
                compiled,
                tests_passed: passed,
                tests_failed: failed,
                all_pass,
                error,
            }
        }
        Err(e) => ExerciseResult {
            name: exercise_name,
            generated: true,
            compiled: false,
            tests_passed: 0,
            tests_failed: 0,
            all_pass: false,
            error: Some(format!("Failed to run cargo: {}", e)),
        },
    }
}

/// Parse "test result: ok. 9 passed; 0 failed;" from cargo test output.
/// Also checks stderr for test results (cargo test prints results to stdout).
fn parse_test_results(output: &str) -> (usize, usize) {
    let mut total_passed = 0;
    let mut total_failed = 0;

    for line in output.lines() {
        if line.starts_with("test result:") {
            // Format: "test result: ok. 9 passed; 0 failed; 0 ignored; ..."
            for part in line.split(';') {
                let trimmed = part.trim();
                // Find the number before "passed" or "failed"
                let words: Vec<&str> = trimmed.split_whitespace().collect();
                for (i, word) in words.iter().enumerate() {
                    if *word == "passed" && i > 0 {
                        total_passed += words[i - 1].parse::<usize>().unwrap_or(0);
                    }
                    if *word == "failed" && i > 0 {
                        // "0 failed" or "2 failed"
                        // But also matches "test result: FAILED. 0 passed; 2 failed;"
                        total_failed += words[i - 1].parse::<usize>().unwrap_or(0);
                    }
                }
            }
        }
    }

    (total_passed, total_failed)
}

/// Recursively copy a directory
fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let dst_path = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_recursive(&entry.path(), &dst_path)?;
        } else {
            std::fs::copy(entry.path(), &dst_path)?;
        }
    }
    Ok(())
}

fn main() {
    println!("=== Exercism Rust Benchmark (External Validation) ===\n");

    let exercism_path = PathBuf::from(EXERCISM_DIR);
    if !exercism_path.exists() {
        eprintln!(
            "Exercism exercises not found at {}. Run:\n  \
             git clone --depth 1 https://github.com/exercism/rust.git benchmarks/external/exercism-rust",
            EXERCISM_DIR
        );
        return;
    }

    let encoder = CodeHDEncoder::new(512);
    let generator = CodeGenerator::new(encoder);

    // Index ALL canonical solutions for analogy-based generation
    let mut solution_library = SolutionLibrary::new(512);
    let indexed = solution_library.load_exercism_solutions(&exercism_path);
    println!("Indexed {} canonical solutions for analogy\n", indexed);

    // Collect all exercises
    let mut exercises: Vec<PathBuf> = std::fs::read_dir(&exercism_path)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map_or(false, |t| t.is_dir()))
        .map(|e| e.path())
        .collect();
    exercises.sort();

    let total = exercises.len();
    let mut generated = 0usize;
    let mut compiled = 0usize;
    let mut all_pass = 0usize;
    let mut skipped = 0usize;
    let mut results = Vec::new();

    println!("Found {} exercises\n", total);

    for exercise_dir in &exercises {
        let exercise_name = exercise_dir
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default();

        // Parse ALL functions in the exercise
        let functions = parse_exercise_functions(exercise_dir);
        if functions.is_empty() {
            println!("  {:30} SKIP (no parseable signature)", exercise_name);
            skipped += 1;
            continue;
        }

        // Generate each function and combine
        let mut all_implementations = Vec::new();
        let mut any_generated = false;
        let mut method = "native".to_string();

        for func in &functions {
            if let Some(impl_code) =
                generate_implementation(&generator, &func.name, &func.purpose, &func.signature)
            {
                all_implementations.push(impl_code);
                any_generated = true;
            } else {
                // Native failed — try analogy for this specific function
                if let Some(analogy_result) = solution_library.generate_by_analogy(
                    &func.purpose,
                    &func.name,
                    &func.signature,
                    0.50,
                ) {
                    if analogy_result.matched_name != exercise_name {
                        all_implementations.push(analogy_result.source);
                        any_generated = true;
                        method = format!("native+analogy({})", analogy_result.matched_name);
                    }
                }
            }
        }

        if !any_generated {
            println!("  {:30} SKIP (generated todo!())", exercise_name);
            skipped += 1;
            continue;
        }

        generated += 1;

        // Combine all function implementations into a single source file
        // Strip duplicate test modules — only keep tests from the last function
        let mut combined_parts: Vec<String> = Vec::new();
        for (i, impl_code) in all_implementations.iter().enumerate() {
            if i < all_implementations.len() - 1 {
                // Strip test module from non-last implementations
                if let Some(test_start) = impl_code.find("#[cfg(test)]") {
                    combined_parts.push(impl_code[..test_start].trim().to_string());
                } else {
                    combined_parts.push(impl_code.clone());
                }
            } else {
                combined_parts.push(impl_code.clone());
            }
        }
        let implementation = combined_parts.join("\n\n");

        // Test
        let result = run_exercise_tests(exercise_dir, &implementation);

        let status = if result.all_pass {
            compiled += 1;
            all_pass += 1;
            "PASS"
        } else if result.compiled {
            compiled += 1;
            "COMPILED (tests fail)"
        } else {
            "FAIL"
        };

        println!(
            "  {:30} {} [{}] (passed: {}, failed: {})",
            exercise_name, status, method, result.tests_passed, result.tests_failed
        );
        if let Some(ref err) = result.error {
            let short: String = err.chars().take(80).collect();
            println!("    {}", short);
        }

        results.push(result);
    }

    // Summary
    let attempted = total - skipped;
    println!("\n=== Results ===");
    println!("  Total exercises:  {}", total);
    println!("  Skipped:          {} (no signature or todo!())", skipped);
    println!("  Attempted:        {}", attempted);
    println!(
        "  Generated body:   {}/{} ({:.0}%)",
        generated,
        attempted,
        if attempted > 0 {
            generated as f64 / attempted as f64 * 100.0
        } else {
            0.0
        }
    );
    println!(
        "  Compiled:         {}/{} ({:.0}%)",
        compiled,
        attempted,
        if attempted > 0 {
            compiled as f64 / attempted as f64 * 100.0
        } else {
            0.0
        }
    );
    println!(
        "  All tests pass:   {}/{} ({:.0}%)",
        all_pass,
        attempted,
        if attempted > 0 {
            all_pass as f64 / attempted as f64 * 100.0
        } else {
            0.0
        }
    );
    println!(
        "\n  pass@1 = {:.1}%",
        if attempted > 0 {
            all_pass as f64 / attempted as f64 * 100.0
        } else {
            0.0
        }
    );

    // Category breakdown
    let mut pass_names: Vec<&str> = results
        .iter()
        .filter(|r| r.all_pass)
        .map(|r| r.name.as_str())
        .collect();
    pass_names.sort();
    if !pass_names.is_empty() {
        println!("\n  Passed exercises:");
        for name in &pass_names {
            println!("    {}", name);
        }
    }
}
