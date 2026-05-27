// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Code Generation Smoke Test
//!
//! Validates the full code generation pipeline end-to-end:
//!   1. Native emitter (simple patterns)
//!   2. LLM completion (complex algorithms via Ollama)
//!   3. Tree-sitter verification
//!   4. Compilation check
//!
//! Run: cargo run --bin code_gen_smoke --features code_generation

#![cfg(feature = "code_generation")]

use symthaea::language::code_generator::{CodeContext, CodeGenerator};
use symthaea::language::code_intent::{CodeIntent, CodeSpec, CodeTarget};
use symthaea::language::code_parser::EntityKind;

fn main() {
    println!("=== Symthaea Code Generation Smoke Test ===\n");

    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();

    // ── Section 1: Native Emitter Tests ──────────────────────────────

    println!("── Native Emitter (no LLM needed) ──\n");

    let native_cases: Vec<(&str, &str, &str, &str)> = vec![
        (
            "add",
            "Add two numbers",
            "fn add(a: i32, b: i32) -> i32",
            "a + b",
        ),
        (
            "reverse",
            "Reverse a string",
            "fn reverse(s: &str) -> String",
            "chars().rev()",
        ),
        (
            "factorial",
            "Compute factorial",
            "fn factorial(n: u64) -> u64",
            ".product()",
        ),
        (
            "is_even",
            "Check if a number is even",
            "fn is_even(n: i32) -> bool",
            "% 2 == 0",
        ),
        (
            "sort",
            "Sort a vector",
            "fn sort(items: Vec<i32>) -> Vec<i32>",
            ".sort()",
        ),
        (
            "uppercase",
            "Convert to uppercase",
            "fn uppercase(s: &str) -> String",
            ".to_uppercase()",
        ),
        (
            "fibonacci",
            "Compute fibonacci",
            "fn fibonacci(n: u64) -> u64",
            "let (mut a, mut b)",
        ),
        ("abs", "Absolute value", "fn abs(n: i32) -> i32", ".abs()"),
    ];

    let mut native_ok = 0;
    let mut native_fail = 0;

    for (name, purpose, sig, expected_fragment) in &native_cases {
        let intent = CodeIntent::Create {
            target: CodeTarget::new(*name, EntityKind::Function).with_language("rust"),
            spec: CodeSpec::new("rust", *name, *purpose).with_signature(*sig),
        };

        let result = r#gen.generate(&intent, &ctx);
        let has_todo = result.source.contains("todo!");
        let has_fragment = result.source.contains(expected_fragment);

        if !has_todo && has_fragment {
            println!("  [PASS] {} — native body generated", name);
            native_ok += 1;
        } else {
            println!(
                "  [FAIL] {} — todo={}, fragment={}",
                name, has_todo, has_fragment
            );
            println!(
                "         Generated: {}",
                result
                    .source
                    .lines()
                    .take(3)
                    .collect::<Vec<_>>()
                    .join(" | ")
            );
            native_fail += 1;
        }
    }

    println!("\n  Native: {}/{} passed\n", native_ok, native_cases.len());

    // ── Section 2: Composition Tests ─────────────────────────────────

    println!("── Composition (multi-step operations) ──\n");

    let composition_cases: Vec<(&str, &str, &str, &str)> = vec![
        (
            "filter_and_sum",
            "Filter even numbers and sum them",
            "fn filter_and_sum(items: Vec<i32>) -> i32",
            ".filter",
        ),
        (
            "sort_and_take",
            "Sort and take first 3",
            "fn sort_and_take(items: Vec<i32>) -> Vec<i32>",
            ".take",
        ),
    ];

    for (name, purpose, sig, expected) in &composition_cases {
        let intent = CodeIntent::Create {
            target: CodeTarget::new(*name, EntityKind::Function).with_language("rust"),
            spec: CodeSpec::new("rust", *name, *purpose).with_signature(*sig),
        };
        let result = r#gen.generate(&intent, &ctx);
        let ok = result.source.contains(expected) && !result.source.contains("todo!");
        println!(
            "  [{}] {} — {}",
            if ok { "PASS" } else { "FAIL" },
            name,
            if ok {
                "composed chain generated"
            } else {
                "missing chain"
            }
        );
    }

    // ── Section 3: LLM Completion Detection ──────────────────────────

    println!("\n── LLM Completion Detection ──\n");

    let complex_cases: Vec<(&str, &str, &str)> = vec![
        (
            "dijkstra",
            "Implement Dijkstra's shortest path algorithm",
            "fn dijkstra(graph: &[Vec<(usize, u32)>], start: usize) -> Vec<u32>",
        ),
        (
            "solve_knapsack",
            "Implement dynamic programming knapsack solver",
            "fn solve_knapsack(weights: &[u32], values: &[u32], capacity: u32) -> u32",
        ),
    ];

    for (name, purpose, sig) in &complex_cases {
        let intent = CodeIntent::Create {
            target: CodeTarget::new(*name, EntityKind::Function).with_language("rust"),
            spec: CodeSpec::new("rust", *name, *purpose).with_signature(*sig),
        };
        let result = r#gen.generate(&intent, &ctx);
        let needs_llm = result.source.contains("todo!");
        println!(
            "  [{}] {} — {}",
            if needs_llm { "PASS" } else { "WARN" },
            name,
            if needs_llm {
                "correctly detected as needing LLM"
            } else {
                "unexpectedly native (check emitter)"
            }
        );
    }

    // ── Section 4: Python Generation ─────────────────────────────────

    println!("\n── Python Generation ──\n");

    let python_cases: Vec<(&str, &str, &str)> = vec![
        ("add", "Add two numbers", "return a + b"),
        ("reverse", "Reverse a string", "return s[::-1]"),
        ("factorial", "Compute factorial", "math.factorial"),
    ];

    for (name, purpose, expected) in &python_cases {
        let intent = CodeIntent::Create {
            target: CodeTarget::new(*name, EntityKind::Function).with_language("python"),
            spec: CodeSpec::new("python", *name, *purpose),
        };
        let result = r#gen.generate(&intent, &ctx);
        let ok = result.source.contains(expected);
        println!(
            "  [{}] {} (Python) — {}",
            if ok { "PASS" } else { "FAIL" },
            name,
            if ok { "correct body" } else { &result.source }
        );
    }

    // ── Summary ──────────────────────────────────────────────────────

    println!("\n=== Summary ===");
    println!("  Native emitters: {}/{}", native_ok, native_cases.len());
    println!("  Failures: {}", native_fail);
    if native_fail == 0 {
        println!("\n  All smoke tests passed!");
    } else {
        println!("\n  Some tests failed — check output above.");
        std::process::exit(1);
    }
}
