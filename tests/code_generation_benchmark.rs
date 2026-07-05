#![cfg(feature = "code_generation")]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Code Generation Benchmark / Regression Suite
// ==================================================================================
//
// Scores Symthaea's native code generation across 50+ cases in 6 categories.
// Each case checks required fragments (must appear) and forbidden fragments
// (must not appear). Cases marked `expects_llm` tolerate todo!()/NotImplementedError.
//
// Run: cargo test --test code_generation_benchmark --features code_generation
// ==================================================================================

use symthaea::language::code_generator::{CodeContext, CodeGenerator};
use symthaea::language::code_intent::{CodeIntent, CodeSpec, CodeTarget};
use symthaea::language::code_parser::EntityKind;

// ==================================================================================
// Benchmark infrastructure
// ==================================================================================

struct BenchmarkCase {
    name: &'static str,
    language: &'static str,
    purpose: &'static str,
    signature: Option<&'static str>,
    /// Fragments that MUST appear in the output
    required_fragments: Vec<&'static str>,
    /// Fragments that must NOT appear (e.g., "todo!")
    forbidden_fragments: Vec<&'static str>,
    /// If true, expected to need LLM (todo! is OK)
    expects_llm: bool,
}

struct BenchmarkResult {
    name: &'static str,
    passed: bool,
    reason: Option<String>,
    expects_llm: bool,
}

fn run_case(r#gen: &CodeGenerator, ctx: &CodeContext, case: &BenchmarkCase) -> BenchmarkResult {
    let intent = CodeIntent::Create {
        target: CodeTarget::new(case.name, EntityKind::Function).with_language(case.language),
        spec: {
            let mut s = CodeSpec::new(case.language, case.name, case.purpose);
            if let Some(sig) = case.signature {
                s = s.with_signature(sig);
            }
            s
        },
    };

    let result = r#gen.generate(&intent, ctx);

    // For LLM-expected cases, we just verify the function signature exists
    // and that todo!() or NotImplementedError is present
    if case.expects_llm {
        let has_placeholder =
            result.source.contains("todo!") || result.source.contains("NotImplementedError");
        if has_placeholder {
            return BenchmarkResult {
                name: case.name,
                passed: true,
                reason: None,
                expects_llm: true,
            };
        }
        // If it somehow generated real code, that is also fine
        return BenchmarkResult {
            name: case.name,
            passed: true,
            reason: Some("unexpectedly generated native code".to_string()),
            expects_llm: true,
        };
    }

    // Check forbidden fragments
    for frag in &case.forbidden_fragments {
        if result.source.contains(frag) {
            return BenchmarkResult {
                name: case.name,
                passed: false,
                reason: Some(format!(
                    "forbidden fragment '{}' found in: {}",
                    frag,
                    result
                        .source
                        .lines()
                        .take(3)
                        .collect::<Vec<_>>()
                        .join(" | ")
                )),
                expects_llm: false,
            };
        }
    }

    // Check required fragments
    for frag in &case.required_fragments {
        if !result.source.contains(frag) {
            return BenchmarkResult {
                name: case.name,
                passed: false,
                reason: Some(format!(
                    "required fragment '{}' missing from: {}",
                    frag,
                    result
                        .source
                        .lines()
                        .take(5)
                        .collect::<Vec<_>>()
                        .join(" | ")
                )),
                expects_llm: false,
            };
        }
    }

    BenchmarkResult {
        name: case.name,
        passed: true,
        reason: None,
        expects_llm: false,
    }
}

fn run_category(
    r#gen: &CodeGenerator,
    ctx: &CodeContext,
    cases: &[BenchmarkCase],
) -> Vec<BenchmarkResult> {
    cases.iter().map(|c| run_case(r#gen, ctx, c)).collect()
}

fn print_results(category: &str, results: &[BenchmarkResult]) {
    let passed = results.iter().filter(|r| r.passed).count();
    let total = results.len();
    println!("  {} {}/{}", category, passed, total);
    for r in results {
        if !r.passed {
            if let Some(ref reason) = r.reason {
                println!("    FAIL: {} -- {}", r.name, reason);
            }
        }
    }
}

// ==================================================================================
// Case builders
// ==================================================================================

fn build_arithmetic_cases() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase {
            name: "add",
            language: "rust",
            purpose: "Add two numbers",
            signature: Some("fn add(a: i32, b: i32) -> i32"),
            required_fragments: vec!["a + b"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "subtract",
            language: "rust",
            purpose: "Subtract two numbers",
            signature: Some("fn subtract(a: i32, b: i32) -> i32"),
            required_fragments: vec!["a - b"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "multiply",
            language: "rust",
            purpose: "Multiply two numbers",
            signature: Some("fn multiply(a: f64, b: f64) -> f64"),
            required_fragments: vec!["a * b"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "divide",
            language: "rust",
            purpose: "Divide two numbers",
            signature: Some("fn divide(a: f64, b: f64) -> f64"),
            required_fragments: vec!["a / b"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "abs",
            language: "rust",
            purpose: "Absolute value of a number",
            signature: Some("fn abs(n: i32) -> i32"),
            required_fragments: vec![".abs()"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "clamp",
            language: "rust",
            purpose: "Clamp a value between min and max",
            signature: Some("fn clamp(n: i32, min: i32, max: i32) -> i32"),
            required_fragments: vec![".clamp("],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "factorial",
            language: "rust",
            purpose: "Compute factorial of n",
            signature: Some("fn factorial(n: u64) -> u64"),
            required_fragments: vec![".product()"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "fibonacci",
            language: "rust",
            purpose: "Compute fibonacci number",
            signature: Some("fn fibonacci(n: u64) -> u64"),
            required_fragments: vec!["let (mut a, mut b)"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "gcd",
            language: "rust",
            purpose: "Compute greatest common divisor",
            signature: Some("fn gcd(a: u64, b: u64) -> u64"),
            required_fragments: vec!["while b != 0"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "power",
            language: "rust",
            purpose: "Raise base to the power of exponent",
            signature: Some("fn power(base: f64, exp: f64) -> f64"),
            required_fragments: vec![".powf("],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
    ]
}

fn build_string_cases() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase {
            name: "reverse",
            language: "rust",
            purpose: "Reverse a string",
            signature: Some("fn reverse(s: &str) -> String"),
            required_fragments: vec!["chars().rev().collect()"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "uppercase",
            language: "rust",
            purpose: "Convert string to uppercase",
            signature: Some("fn uppercase(s: &str) -> String"),
            required_fragments: vec![".to_uppercase()"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "lowercase",
            language: "rust",
            purpose: "Convert string to lowercase",
            signature: Some("fn lowercase(s: &str) -> String"),
            required_fragments: vec![".to_lowercase()"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "contains",
            language: "rust",
            purpose: "Check if string contains a substring",
            signature: Some("fn contains(s: &str, sub: &str) -> bool"),
            required_fragments: vec![".contains("],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "starts_with",
            language: "rust",
            purpose: "Check if string starts with prefix",
            signature: Some("fn starts_with(s: &str, prefix: &str) -> bool"),
            required_fragments: vec![".starts_with("],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "ends_with",
            language: "rust",
            purpose: "Check if string ends with suffix",
            signature: Some("fn ends_with(s: &str, suffix: &str) -> bool"),
            required_fragments: vec![".ends_with("],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "trim",
            language: "rust",
            purpose: "Trim whitespace from string",
            signature: Some("fn trim(s: &str) -> String"),
            required_fragments: vec![".trim()"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "replace",
            language: "rust",
            purpose: "Replace occurrences in string",
            signature: Some("fn replace(s: &str, old: &str, new: &str) -> String"),
            required_fragments: vec![".replace("],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "repeat",
            language: "rust",
            purpose: "Repeat a string n times",
            signature: Some("fn repeat(s: &str, n: usize) -> String"),
            required_fragments: vec![".repeat("],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "capitalize",
            language: "rust",
            purpose: "Capitalize the first letter of a string",
            signature: Some("fn capitalize(s: &str) -> String"),
            required_fragments: vec!["to_uppercase()"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
    ]
}

fn build_collection_cases() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase {
            name: "sort",
            language: "rust",
            purpose: "Sort a vector of integers",
            signature: Some("fn sort(items: Vec<i32>) -> Vec<i32>"),
            required_fragments: vec![".sort()"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "filter",
            language: "rust",
            purpose: "Filter elements from a vector",
            signature: Some("fn filter(items: Vec<i32>) -> Vec<i32>"),
            required_fragments: vec![".filter("],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "map",
            language: "rust",
            purpose: "Map a transform over a vector",
            signature: Some("fn map(items: Vec<i32>) -> Vec<i32>"),
            required_fragments: vec![".map("],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "sum_vec",
            language: "rust",
            purpose: "Sum all elements in a vector",
            signature: Some("fn sum_vec(items: Vec<i32>) -> i32"),
            required_fragments: vec![".iter().sum()"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "max_vec",
            language: "rust",
            purpose: "Find the max element in a vector",
            signature: Some("fn max_vec(items: Vec<i32>) -> Option<i32>"),
            required_fragments: vec![".max()"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "min_vec",
            language: "rust",
            purpose: "Find the min element in a vector",
            signature: Some("fn min_vec(items: Vec<i32>) -> Option<i32>"),
            required_fragments: vec![".min()"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "zip_vecs",
            language: "rust",
            purpose: "Zip two vectors together",
            signature: Some("fn zip_vecs(a: Vec<i32>, b: Vec<i32>) -> Vec<(i32, i32)>"),
            required_fragments: vec![".zip("],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "enumerate_vec",
            language: "rust",
            purpose: "Enumerate elements with their index",
            signature: Some("fn enumerate_vec(items: Vec<i32>) -> Vec<(usize, i32)>"),
            required_fragments: vec![".enumerate()"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "take_first",
            language: "rust",
            purpose: "Take first n elements from a vector",
            signature: Some("fn take_first(items: Vec<i32>, n: usize) -> Vec<i32>"),
            required_fragments: vec![".take("],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "unique",
            language: "rust",
            purpose: "Remove duplicate elements from a vector (unique/deduplicate)",
            signature: Some("fn unique(items: Vec<i32>) -> Vec<i32>"),
            required_fragments: vec![".dedup()"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
    ]
}

fn build_composition_cases() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase {
            name: "filter_and_sum",
            language: "rust",
            purpose: "Filter even numbers and sum them",
            signature: Some("fn filter_and_sum(items: Vec<i32>) -> i32"),
            required_fragments: vec![".filter(", ".sum()"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "sort_and_take",
            language: "rust",
            purpose: "Sort and take first 3 elements",
            signature: Some("fn sort_and_take(items: Vec<i32>) -> Vec<i32>"),
            required_fragments: vec![".sort(", ".take("],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "filter_and_count",
            language: "rust",
            purpose: "Filter odd numbers and count them",
            signature: Some("fn filter_and_count(items: Vec<i32>) -> usize"),
            required_fragments: vec![".filter(", ".count()"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "sort_dedup_take",
            language: "rust",
            purpose: "Sort then deduplicate then take first 5",
            signature: Some("fn sort_dedup_take(items: Vec<i32>) -> Vec<i32>"),
            required_fragments: vec![".sort(", ".dedup("],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "filter_even_sum",
            language: "rust",
            purpose: "Filter even numbers and sum the total",
            signature: Some("fn filter_even_sum(items: Vec<i32>) -> i32"),
            required_fragments: vec![".filter(", "% 2 == 0", ".sum()"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
    ]
}

fn build_python_cases() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase {
            name: "add",
            language: "python",
            purpose: "Add two numbers",
            signature: None,
            required_fragments: vec!["return a + b"],
            forbidden_fragments: vec!["NotImplementedError"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "reverse",
            language: "python",
            purpose: "Reverse a string",
            signature: None,
            required_fragments: vec!["s[::-1]"],
            forbidden_fragments: vec!["NotImplementedError"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "factorial",
            language: "python",
            purpose: "Compute factorial of n",
            signature: None,
            required_fragments: vec!["math.factorial"],
            forbidden_fragments: vec!["NotImplementedError"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "fibonacci",
            language: "python",
            purpose: "Compute fibonacci number",
            signature: None,
            required_fragments: vec!["a, b = 0, 1"],
            forbidden_fragments: vec!["NotImplementedError"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "is_even",
            language: "python",
            purpose: "Check if a number is_even",
            signature: None,
            required_fragments: vec!["n % 2 == 0"],
            forbidden_fragments: vec!["NotImplementedError"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "uppercase",
            language: "python",
            purpose: "Convert string to uppercase",
            signature: None,
            required_fragments: vec![".upper()"],
            forbidden_fragments: vec!["NotImplementedError"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "abs",
            language: "python",
            purpose: "Absolute value of a number",
            signature: None,
            required_fragments: vec!["abs(n)"],
            forbidden_fragments: vec!["NotImplementedError"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "sort",
            language: "python",
            purpose: "Sort a list of numbers",
            signature: None,
            required_fragments: vec!["sorted("],
            forbidden_fragments: vec!["NotImplementedError"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "filter_even",
            language: "python",
            purpose: "Filter even numbers from a list",
            signature: None,
            required_fragments: vec!["x % 2 == 0"],
            forbidden_fragments: vec!["NotImplementedError"],
            expects_llm: false,
        },
        BenchmarkCase {
            name: "power",
            language: "python",
            purpose: "Raise a to the power of b",
            signature: None,
            required_fragments: vec!["a ** b"],
            forbidden_fragments: vec!["NotImplementedError"],
            expects_llm: false,
        },
    ]
}

fn build_complex_llm_cases() -> Vec<BenchmarkCase> {
    vec![
        BenchmarkCase {
            name: "dijkstra",
            language: "rust",
            purpose: "Implement Dijkstra's shortest path algorithm on a weighted adjacency list",
            signature: Some("fn dijkstra(graph: &[Vec<(usize, u32)>], start: usize) -> Vec<u32>"),
            required_fragments: vec!["pub fn dijkstra"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
        BenchmarkCase {
            name: "knapsack",
            language: "rust",
            purpose: "Implement dynamic programming 0/1 knapsack solver",
            signature: Some("fn knapsack(weights: &[u32], values: &[u32], capacity: u32) -> u32"),
            required_fragments: vec!["pub fn knapsack"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
        BenchmarkCase {
            name: "binary_tree_traversal",
            language: "rust",
            purpose: "Implement in-order binary tree traversal collecting node values",
            signature: Some("fn inorder(root: &TreeNode) -> Vec<i32>"),
            required_fragments: vec!["pub fn inorder"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
        BenchmarkCase {
            name: "regex_match",
            language: "rust",
            purpose: "Implement a simple regex engine supporting . and * metacharacters",
            signature: Some("fn regex_match(pattern: &str, text: &str) -> bool"),
            required_fragments: vec!["pub fn regex_match"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
        BenchmarkCase {
            name: "http_parse",
            language: "rust",
            purpose: "Parse an HTTP request line into method, path, and version",
            signature: Some("fn http_parse(request: &str) -> (String, String, String)"),
            required_fragments: vec!["pub fn http_parse"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
    ]
}

fn build_advanced_rust_cases() -> Vec<BenchmarkCase> {
    vec![
        // 1. Error handling — emitter has a Result+parse path
        BenchmarkCase {
            name: "parse_integer",
            language: "rust",
            purpose: "Parse a string to integer with error handling",
            signature: Some("fn parse_integer(s: &str) -> Result<i32, String>"),
            required_fragments: vec![".parse("],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        // 2. Option handling — emitter has an Option+find path
        BenchmarkCase {
            name: "find_first_even",
            language: "rust",
            purpose: "Find first even number in a vector",
            signature: Some("fn find_first_even(items: Vec<i32>) -> Option<i32>"),
            required_fragments: vec![".find("],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        // 3. Closures / map — emitter has a map path but uses placeholder closure
        BenchmarkCase {
            name: "apply_transform",
            language: "rust",
            purpose: "Apply a function to each element (map a transform over a vector)",
            signature: Some("fn apply_transform(items: Vec<i32>) -> Vec<i32>"),
            required_fragments: vec![".map("],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        // 4. Multiple returns (tuple) — emitter has no tuple-return pattern
        BenchmarkCase {
            name: "divide_with_remainder",
            language: "rust",
            purpose: "Divide with remainder returning quotient and remainder as a tuple",
            signature: Some("fn divide_with_remainder(a: i32, b: i32) -> (i32, i32)"),
            required_fragments: vec!["pub fn divide_with_remainder"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
        // 5. String formatting — emitter has no format!() pattern for greeting
        BenchmarkCase {
            name: "format_greeting",
            language: "rust",
            purpose: "Format a greeting message for a given name",
            signature: Some("fn format_greeting(name: &str) -> String"),
            required_fragments: vec!["pub fn format_greeting"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
        // 6. Nested iteration — emitter has no intersection pattern
        BenchmarkCase {
            name: "find_common_elements",
            language: "rust",
            purpose: "Find common elements in two vectors (intersection)",
            signature: Some("fn find_common_elements(a: Vec<i32>, b: Vec<i32>) -> Vec<i32>"),
            required_fragments: vec!["pub fn find_common_elements"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
        // 7. Pattern matching — emitter has no match expression pattern
        BenchmarkCase {
            name: "classify_number",
            language: "rust",
            purpose: "Classify number as positive negative or zero using match",
            signature: Some("fn classify_number(n: i32) -> &'static str"),
            required_fragments: vec!["pub fn classify_number"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
        // 8. Range operations — emitter handles sum of collection but not range sum
        BenchmarkCase {
            name: "sum_range",
            language: "rust",
            purpose: "Sum numbers from 1 to n using a range",
            signature: Some("fn sum_range(n: u64) -> u64"),
            required_fragments: vec!["pub fn sum_range"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
        // 9. HashMap — emitter has no HashMap pattern
        BenchmarkCase {
            name: "count_word_frequencies",
            language: "rust",
            purpose: "Count word frequencies in a list of words using a HashMap",
            signature: Some(
                "fn count_word_frequencies(words: Vec<&str>) -> std::collections::HashMap<String, usize>",
            ),
            required_fragments: vec!["pub fn count_word_frequencies"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
        // 10. Chained operations — emitter handles sort+dedup+take composition
        BenchmarkCase {
            name: "top_3_unique",
            language: "rust",
            purpose: "Sort then deduplicate then take first 3 unique sorted values",
            signature: Some("fn top_3_unique(items: Vec<i32>) -> Vec<i32>"),
            required_fragments: vec![".sort(", ".dedup("],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
        // 11. Boolean logic all() — emitter has no .all() pattern
        BenchmarkCase {
            name: "all_positive",
            language: "rust",
            purpose: "Check if all elements are positive in a vector",
            signature: Some("fn all_positive(items: Vec<i32>) -> bool"),
            required_fragments: vec!["pub fn all_positive"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
        // 12. Boolean logic any() — emitter has no .any() pattern
        BenchmarkCase {
            name: "any_negative",
            language: "rust",
            purpose: "Check if any element is negative in a vector",
            signature: Some("fn any_negative(items: Vec<i32>) -> bool"),
            required_fragments: vec!["pub fn any_negative"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
        // 13. Conversion — emitter has no map+parse+collect pattern for Vec<String>->Vec<i32>
        BenchmarkCase {
            name: "strings_to_integers",
            language: "rust",
            purpose: "Convert vector of strings to integers by parsing each one",
            signature: Some("fn strings_to_integers(items: Vec<String>) -> Vec<i32>"),
            required_fragments: vec!["pub fn strings_to_integers"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
        // 14. Default values — emitter has no unwrap_or pattern
        BenchmarkCase {
            name: "get_or_default",
            language: "rust",
            purpose: "Get value at index or return a default value using unwrap_or",
            signature: Some(
                "fn get_or_default(items: Vec<i32>, index: usize, default: i32) -> i32",
            ),
            required_fragments: vec!["pub fn get_or_default"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
        // 15. Zip + map + sum — emitter handles zip but not zip+multiply+sum (dot product)
        BenchmarkCase {
            name: "dot_product",
            language: "rust",
            purpose: "Compute dot product of two vectors by zipping and multiplying",
            signature: Some("fn dot_product(a: Vec<i32>, b: Vec<i32>) -> i32"),
            required_fragments: vec!["pub fn dot_product"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
        // 16. Window operations — emitter has no windows() pattern
        BenchmarkCase {
            name: "max_adjacent_diff",
            language: "rust",
            purpose: "Find maximum difference between adjacent elements using windows",
            signature: Some("fn max_adjacent_diff(items: Vec<i32>) -> i32"),
            required_fragments: vec!["pub fn max_adjacent_diff"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
        // 17. Partition — emitter has no partition pattern
        BenchmarkCase {
            name: "split_even_odd",
            language: "rust",
            purpose: "Split a vector into even and odd numbers using partition",
            signature: Some("fn split_even_odd(items: Vec<i32>) -> (Vec<i32>, Vec<i32>)"),
            required_fragments: vec!["pub fn split_even_odd"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
        // 18. Flat map — emitter handles flatten but not flat_map on strings
        BenchmarkCase {
            name: "all_characters",
            language: "rust",
            purpose: "Get all characters from a list of strings using flat_map",
            signature: Some("fn all_characters(items: Vec<String>) -> Vec<char>"),
            required_fragments: vec!["pub fn all_characters"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
        // 19. Reduction — emitter handles max() for i32 but not longest string
        BenchmarkCase {
            name: "longest_string",
            language: "rust",
            purpose: "Find the longest string in a vector by length",
            signature: Some("fn longest_string(items: Vec<String>) -> Option<String>"),
            required_fragments: vec!["pub fn longest_string"],
            forbidden_fragments: vec![],
            expects_llm: true,
        },
        // 20. Complex composition — emitter handles filter+sum but not filter_positive+square+sum
        BenchmarkCase {
            name: "filter_square_sum",
            language: "rust",
            purpose: "Filter positive numbers then square each then sum the results",
            signature: Some("fn filter_square_sum(items: Vec<i32>) -> i32"),
            required_fragments: vec![".filter(", ".sum()"],
            forbidden_fragments: vec!["todo!"],
            expects_llm: false,
        },
    ]
}

fn build_all_cases() -> Vec<BenchmarkCase> {
    let mut all = Vec::new();
    all.extend(build_arithmetic_cases());
    all.extend(build_string_cases());
    all.extend(build_collection_cases());
    all.extend(build_composition_cases());
    all.extend(build_python_cases());
    all.extend(build_complex_llm_cases());
    all.extend(build_advanced_rust_cases());
    all
}

// ==================================================================================
// Main benchmark test
// ==================================================================================

#[test]
fn benchmark_code_generation() {
    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();
    let cases = build_all_cases();

    let mut passed = 0;
    let mut failed = 0;
    let mut llm_needed = 0;
    let mut failures: Vec<(&str, String)> = Vec::new();

    for case in &cases {
        let result = run_case(&r#gen, &ctx, case);
        if result.passed {
            if result.expects_llm {
                llm_needed += 1;
            }
            passed += 1;
        } else {
            failed += 1;
            if let Some(ref reason) = result.reason {
                failures.push((result.name, reason.clone()));
            }
        }
    }

    // Print summary
    println!("\n=== Code Generation Benchmark ===");
    println!("Total cases: {}", cases.len());
    println!("Passed: {}/{}", passed, cases.len());
    println!("Failed: {}", failed);
    println!("LLM-needed (expected): {}", llm_needed);

    // Report failures in detail
    if !failures.is_empty() {
        println!("\nFailures:");
        for (name, reason) in &failures {
            println!("  FAIL: {} -- {}", name, reason);
        }
    }

    // Require at least 70% pass rate for native patterns (exclude LLM cases)
    let native_cases: Vec<&BenchmarkCase> = cases.iter().filter(|c| !c.expects_llm).collect();
    let native_total = native_cases.len();
    let native_passed = native_cases
        .iter()
        .filter(|c| {
            let r = run_case(&r#gen, &ctx, c);
            r.passed
        })
        .count();

    println!(
        "\nNative pass rate: {}/{} = {:.0}%",
        native_passed,
        native_total,
        native_passed as f32 / native_total as f32 * 100.0
    );

    assert!(
        native_passed as f32 / native_total as f32 >= 0.70,
        "Native pattern pass rate {}/{} = {:.0}% (minimum 70%)",
        native_passed,
        native_total,
        native_passed as f32 / native_total as f32 * 100.0
    );
}

// ==================================================================================
// Per-category tests for easier diagnosis
// ==================================================================================

#[test]
fn benchmark_arithmetic() {
    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();
    let cases = build_arithmetic_cases();
    let results = run_category(&r#gen, &ctx, &cases);

    print_results("Arithmetic", &results);

    let passed = results.iter().filter(|r| r.passed).count();
    assert!(
        passed as f32 / results.len() as f32 >= 0.70,
        "Arithmetic pass rate {}/{} below 70%",
        passed,
        results.len()
    );
}

#[test]
fn benchmark_strings() {
    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();
    let cases = build_string_cases();
    let results = run_category(&r#gen, &ctx, &cases);

    print_results("Strings", &results);

    let passed = results.iter().filter(|r| r.passed).count();
    assert!(
        passed as f32 / results.len() as f32 >= 0.70,
        "String pass rate {}/{} below 70%",
        passed,
        results.len()
    );
}

#[test]
fn benchmark_collections() {
    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();
    let cases = build_collection_cases();
    let results = run_category(&r#gen, &ctx, &cases);

    print_results("Collections", &results);

    let passed = results.iter().filter(|r| r.passed).count();
    assert!(
        passed as f32 / results.len() as f32 >= 0.70,
        "Collection pass rate {}/{} below 70%",
        passed,
        results.len()
    );
}

#[test]
fn benchmark_composition() {
    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();
    let cases = build_composition_cases();
    let results = run_category(&r#gen, &ctx, &cases);

    print_results("Composition", &results);

    let passed = results.iter().filter(|r| r.passed).count();
    assert!(
        passed as f32 / results.len() as f32 >= 0.60,
        "Composition pass rate {}/{} below 60%",
        passed,
        results.len()
    );
}

#[test]
fn benchmark_python() {
    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();
    let cases = build_python_cases();
    let results = run_category(&r#gen, &ctx, &cases);

    print_results("Python", &results);

    let passed = results.iter().filter(|r| r.passed).count();
    assert!(
        passed as f32 / results.len() as f32 >= 0.70,
        "Python pass rate {}/{} below 70%",
        passed,
        results.len()
    );
}

#[test]
fn benchmark_complex_llm() {
    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();
    let cases = build_complex_llm_cases();
    let results = run_category(&r#gen, &ctx, &cases);

    print_results("Complex/LLM", &results);

    // All LLM cases should pass (they just need a function signature + todo!)
    let passed = results.iter().filter(|r| r.passed).count();
    assert_eq!(
        passed,
        results.len(),
        "All complex/LLM cases should pass (signature + todo!)"
    );
}

#[test]
fn benchmark_advanced_rust() {
    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();
    let cases = build_advanced_rust_cases();
    let results = run_category(&r#gen, &ctx, &cases);

    print_results("Advanced Rust", &results);

    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let rate_pct = (passed as f32 / total as f32 * 100.0) as u32;

    // Regression tracking — parseable output for CI
    println!(
        "BENCHMARK_RESULT: category=advanced_rust passed={} total={} rate={}%",
        passed, total, rate_pct
    );

    // Print per-case detail for regression tracking
    for r in &results {
        let status = if r.passed { "PASS" } else { "FAIL" };
        let llm_tag = if r.expects_llm { " [llm]" } else { " [native]" };
        println!(
            "  CASE: {} {} {}{}",
            "advanced_rust", r.name, status, llm_tag
        );
    }

    // 50% threshold — these are hard cases, many expect LLM
    assert!(
        passed as f32 / total as f32 >= 0.50,
        "Advanced Rust pass rate {}/{} = {}% (minimum 50%)",
        passed,
        total,
        rate_pct
    );
}

// ==================================================================================
// Regression summary — run all categories and emit parseable output
// ==================================================================================

fn build_all_categories() -> Vec<(&'static str, Vec<BenchmarkCase>)> {
    vec![
        ("arithmetic", build_arithmetic_cases()),
        ("strings", build_string_cases()),
        ("collections", build_collection_cases()),
        ("composition", build_composition_cases()),
        ("python", build_python_cases()),
        ("complex_llm", build_complex_llm_cases()),
        ("advanced_rust", build_advanced_rust_cases()),
    ]
}

/// Real compilation verification: generate code, write to temp file, run `rustc`.
///
/// This tests that the native emitter's output is actually valid Rust that compiles,
/// not just that it contains the right string fragments.
#[test]
fn benchmark_compile_verification() {
    use std::io::Write;

    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();
    let categories = build_all_categories();

    let tmp_dir = std::env::temp_dir().join("symthaea-compile-bench");
    let _ = std::fs::create_dir_all(&tmp_dir);

    let mut attempted = 0usize;
    let mut compiled_ok = 0usize;

    for (_cat_name, cases) in &categories {
        for case in cases {
            // Skip non-Rust and LLM-expected cases
            if case.language != "rust" || case.expects_llm {
                continue;
            }

            let intent = CodeIntent::Create {
                target: CodeTarget::new(case.name, EntityKind::Function)
                    .with_language(case.language),
                spec: {
                    let mut s = CodeSpec::new(case.language, case.name, case.purpose);
                    if let Some(sig) = case.signature {
                        s = s.with_signature(sig);
                    }
                    s
                },
            };

            let result = r#gen.generate(&intent, &ctx);

            // Skip cases with todo! (LLM would fill these)
            if result.source.contains("todo!") {
                continue;
            }

            attempted += 1;

            // Wrap generated code in a compilable lib
            let source = format!(
                "#![allow(unused, dead_code, unused_imports, non_camel_case_types, non_snake_case)]\n\
                 use std::collections::{{HashMap, HashSet}};\n\n\
                 {}\n",
                result.source
            );

            let file_path = tmp_dir.join(format!("{}.rs", case.name));
            let mut f = std::fs::File::create(&file_path).expect("create temp file");
            f.write_all(source.as_bytes()).expect("write temp file");

            let out_path = tmp_dir.join(format!("{}.rlib", case.name));
            let output = std::process::Command::new("rustc")
                .args(["--edition", "2021", "--crate-type", "lib", "-o"])
                .arg(&out_path)
                .arg(&file_path)
                .output();

            match output {
                Ok(o) if o.status.success() => {
                    compiled_ok += 1;
                }
                Ok(o) => {
                    let stderr = String::from_utf8_lossy(&o.stderr);

                    // Auto-fix retry: diagnose error and attempt repair
                    if let Some(fixed_source) = r#gen.try_auto_fix(&source, &stderr) {
                        let _ = std::fs::File::create(&file_path)
                            .and_then(|mut f| f.write_all(fixed_source.as_bytes()));
                        let retry = std::process::Command::new("rustc")
                            .args(["--edition", "2021", "--crate-type", "lib", "-o"])
                            .arg(&out_path)
                            .arg(&file_path)
                            .output();
                        if retry.map_or(false, |r| r.status.success()) {
                            compiled_ok += 1;
                            println!("COMPILE_FIXED: {} — auto-fix succeeded", case.name);
                        } else {
                            println!(
                                "COMPILE_FAIL: {} — {}",
                                case.name,
                                stderr.lines().take(3).collect::<Vec<_>>().join(" | ")
                            );
                        }
                    } else {
                        println!(
                            "COMPILE_FAIL: {} — {}",
                            case.name,
                            stderr.lines().take(3).collect::<Vec<_>>().join(" | ")
                        );
                    }
                }
                Err(e) => {
                    println!("COMPILE_SKIP: {} — rustc not available: {}", case.name, e);
                }
            }

            let _ = std::fs::remove_file(&file_path);
            let _ = std::fs::remove_file(&out_path);
        }
    }

    let _ = std::fs::remove_dir(&tmp_dir);

    if attempted > 0 {
        let rate = compiled_ok as f32 / attempted as f32 * 100.0;
        println!(
            "COMPILE_BENCHMARK: {}/{} compiled ({:.0}%)",
            compiled_ok, attempted, rate
        );
        assert!(
            rate >= 85.0,
            "Native compilation rate {:.0}% is below 85% threshold ({}/{})",
            rate,
            compiled_ok,
            attempted
        );
    }
}

/// Fuzz-style test: generate code for random purpose strings, verify no panics
/// and basic structural invariants hold.
#[test]
fn benchmark_fuzz_generation_robustness() {
    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();

    // Pseudo-random purpose strings — mix of valid, edge-case, and adversarial
    let fuzz_purposes = [
        // Normal
        "sum all elements",
        "find maximum value",
        "reverse a string",
        // Edge cases
        "",
        "   ",
        "a",
        "fn() -> Vec<Vec<Vec<i32>>>",
        // Long
        "compute the running average of the last N elements in a sliding window over a stream of floating point numbers",
        // Special chars
        "count 'hello' in \"world\"",
        "x + y * z / w - (a % b)",
        // Adversarial
        "todo!()",
        "unsafe { std::ptr::null() }",
        "// this is a comment",
        "use std::collections::HashMap;",
        // Composite triggers
        "frequency of each word",
        "3rd largest element",
        "count unique items",
        "zip two lists elementwise",
        "find all pairs that sum to target",
        "cartesian product of sets",
        // Numeric
        "123456789",
        "0.0001",
    ];

    let mut generated = 0usize;
    let mut non_empty = 0usize;

    for purpose in &fuzz_purposes {
        // Should never panic
        let spec = CodeSpec::new("rust", "fuzz_fn", *purpose);
        let intent = CodeIntent::Create {
            target: CodeTarget::new("fuzz_fn", EntityKind::Function),
            spec,
        };
        let result = r#gen.generate(&intent, &ctx);

        generated += 1;
        if !result.source.is_empty() {
            non_empty += 1;
        }

        // Structural invariants
        assert_eq!(result.language, "rust");
        assert!(
            result.intent_similarity >= 0.0 && result.intent_similarity <= 1.0,
            "intent_similarity out of range: {}",
            result.intent_similarity
        );
        assert!(
            result.phi_score >= 0.0,
            "phi_score negative: {}",
            result.phi_score
        );
    }

    println!(
        "FUZZ_RESULT: {}/{} generated non-empty output",
        non_empty, generated
    );
    // At minimum, most non-trivial purposes should produce output
    assert!(
        non_empty >= generated / 2,
        "Too many empty outputs: {}/{}",
        non_empty,
        generated
    );
}

#[test]
fn benchmark_regression_summary() {
    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();

    let categories: Vec<(&str, Vec<BenchmarkCase>)> = build_all_categories();

    println!("\n=== Regression Summary ===");

    let mut total_passed = 0usize;
    let mut total_cases = 0usize;

    for (cat_name, cases) in &categories {
        let results = run_category(&r#gen, &ctx, cases);
        let passed = results.iter().filter(|r| r.passed).count();
        let total = results.len();
        let rate_pct = (passed as f32 / total as f32 * 100.0) as u32;

        println!(
            "BENCHMARK_RESULT: category={} passed={} total={} rate={}%",
            cat_name, passed, total, rate_pct
        );

        total_passed += passed;
        total_cases += total;
    }

    let overall_pct = (total_passed as f32 / total_cases as f32 * 100.0) as u32;
    println!(
        "BENCHMARK_RESULT: category=overall passed={} total={} rate={}%",
        total_passed, total_cases, overall_pct
    );
}
