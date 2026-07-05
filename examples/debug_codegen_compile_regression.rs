// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Throwaway debug aid: prints the raw source CodeGenerator emits for the 7
//! cases that fail benchmark_compile_verification, to root-cause without
//! reading code_generator.rs blind. Delete once the regression is fixed.

use symthaea::language::code_generator::{CodeContext, CodeGenerator};
use symthaea::language::code_intent::{CodeIntent, CodeSpec, CodeTarget};
use symthaea::language::code_parser::EntityKind;

fn show(name: &str, purpose: &str, sig: Option<&str>) {
    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();
    let intent = CodeIntent::Create {
        target: CodeTarget::new(name, EntityKind::Function).with_language("rust"),
        spec: {
            let mut s = CodeSpec::new("rust", name, purpose);
            if let Some(sig) = sig {
                s = s.with_signature(sig);
            }
            s
        },
    };
    let result = r#gen.generate(&intent, &ctx);
    println!("=== {} ===\n{}\n", name, result.source);
}

fn main() {
    show(
        "reverse",
        "Reverse a string",
        Some("fn reverse(s: &str) -> String"),
    );
    show(
        "max_vec",
        "Find the max element in a vector",
        Some("fn max_vec(items: Vec<i32>) -> Option<i32>"),
    );
    show(
        "min_vec",
        "Find the min element in a vector",
        Some("fn min_vec(items: Vec<i32>) -> Option<i32>"),
    );
    show(
        "unique",
        "Remove duplicate elements from a vector (unique/deduplicate)",
        Some("fn unique(items: Vec<i32>) -> Vec<i32>"),
    );
    show(
        "sort",
        "Sort a vector of integers",
        Some("fn sort(items: Vec<i32>) -> Vec<i32>"),
    );
    show(
        "parse_integer",
        "Parse a string to integer with error handling",
        Some("fn parse_integer(s: &str) -> Result<i32, String>"),
    );
    show(
        "find_first_even",
        "Find first even number in a vector",
        Some("fn find_first_even(items: Vec<i32>) -> Option<i32>"),
    );
}
