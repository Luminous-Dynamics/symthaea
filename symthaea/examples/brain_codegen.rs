// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Brain Codegen CLI Demo
//!
//! Demonstrates the full neurosymbolic code generation pipeline:
//! natural language → HDC encoding → System 1 (classify) → System 2
//! (assemble + self-repair) → compilable Rust.
//!
//! Usage:
//!   cargo run --example brain_codegen --features code_generation -- \
//!       "reverse a string" \
//!       "pub fn reverse(input: &str) -> String"
//!
//! Without arguments, runs a built-in demo of 5 example problems.

use symthaea::language::algorithm_training::{
    build_training_pairs, generate_with_repair_hybrid, strong_keyword_class,
    train_linear_classifier,
};

fn print_separator(title: &str) {
    println!("\n┌─────────────────────────────────────────────────────────");
    println!("│ {title}");
    println!("└─────────────────────────────────────────────────────────");
}

fn run_one(
    purpose: &str,
    signature: &str,
    classifier: &symthaea::language::algorithm_training::AlgorithmClassifier,
    pairs: &[symthaea::language::algorithm_encoder::AlgorithmTrainingPair],
) {
    print_separator(&format!("PROBLEM: {purpose}"));
    println!("Signature: {signature}");

    // Show System 1 prediction
    let kw = strong_keyword_class(purpose);
    if let Some(class) = kw {
        println!("System 1 (keyword prior): {:?}", class);
    } else {
        println!("System 1: no strong keyword → using linear classifier");
    }

    let result = generate_with_repair_hybrid(purpose, signature, pairs, classifier, 3);

    println!("\nDetected class: {:?}", result.class);
    println!("Iterations:     {}", result.iterations);
    if !result.error_history.is_empty() {
        println!(
            "Repair attempts: {} errors fed back",
            result.error_history.len()
        );
    }
    println!(
        "Compiles:       {}",
        if result.compiles { "✓ YES" } else { "✗ NO" }
    );

    println!("\n--- Generated Rust ---");
    println!("{}", result.final_code);

    if !result.compiles {
        println!("\n--- Last error ---");
        if let Some(e) = result.error_history.last() {
            // Show first 3 lines of the error
            for line in e.lines().take(3) {
                println!("  {line}");
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    println!("┌─────────────────────────────────────────────────────────");
    println!("│ Symthaea Brain Codegen — Neurosymbolic Pipeline Demo");
    println!("└─────────────────────────────────────────────────────────");
    println!("\nLoading training corpus + training System 1 classifier...");
    let (classifier, train_acc, eval_acc, _) = train_linear_classifier(100, 0.01);
    let pairs = build_training_pairs();
    println!(
        "  → corpus: {} solutions across 8 algorithm classes",
        pairs.len()
    );
    println!(
        "  → classifier: train={:.0}% eval={:.0}%",
        train_acc * 100.0,
        eval_acc * 100.0
    );

    if args.len() == 2 {
        // Custom problem from CLI
        run_one(&args[0], &args[1], &classifier, &pairs);
    } else {
        // Built-in demo
        let demos = [
            (
                "reverse the characters in a string",
                "pub fn reverse(input: &str) -> String",
            ),
            (
                "check if a number is prime",
                "pub fn is_prime(n: u64) -> bool",
            ),
            (
                "sort a list of integers ascending",
                "pub fn sort_nums(nums: Vec<i32>) -> Vec<i32>",
            ),
            (
                "compute the nth fibonacci number",
                "pub fn fib(n: u64) -> u64",
            ),
            (
                "count how many words are in a sentence",
                "pub fn word_count(s: &str) -> usize",
            ),
        ];

        for (purpose, sig) in &demos {
            run_one(purpose, sig, &classifier, &pairs);
        }

        // Final summary
        let mut compiles = 0;
        for (purpose, sig) in &demos {
            let r = generate_with_repair_hybrid(purpose, sig, &pairs, &classifier, 3);
            if r.compiles {
                compiles += 1;
            }
        }

        println!("\n╔═════════════════════════════════════════════════════════");
        println!(
            "║ FINAL: {}/{} compile from natural language description",
            compiles,
            demos.len()
        );
        println!("║ Pipeline: text → HDC → classify → assemble → compile");
        println!("║ No LLM. Pure neurosymbolic generation.");
        println!("╚═════════════════════════════════════════════════════════");
    }
}
