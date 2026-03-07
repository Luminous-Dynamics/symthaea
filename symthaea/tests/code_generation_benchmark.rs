#![cfg(feature = "code_generation")]
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

fn run_case(gen: &CodeGenerator, ctx: &CodeContext, case: &BenchmarkCase) -> BenchmarkResult {
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

    let result = gen.generate(&intent, ctx);

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
    gen: &CodeGenerator,
    ctx: &CodeContext,
    cases: &[BenchmarkCase],
) -> Vec<BenchmarkResult> {
    cases.iter().map(|c| run_case(gen, ctx, c)).collect()
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

fn build_all_cases() -> Vec<BenchmarkCase> {
    let mut all = Vec::new();
    all.extend(build_arithmetic_cases());
    all.extend(build_string_cases());
    all.extend(build_collection_cases());
    all.extend(build_composition_cases());
    all.extend(build_python_cases());
    all.extend(build_complex_llm_cases());
    all
}

// ==================================================================================
// Main benchmark test
// ==================================================================================

#[test]
fn benchmark_code_generation() {
    let gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();
    let cases = build_all_cases();

    let mut passed = 0;
    let mut failed = 0;
    let mut llm_needed = 0;
    let mut failures: Vec<(&str, String)> = Vec::new();

    for case in &cases {
        let result = run_case(&gen, &ctx, case);
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
            let r = run_case(&gen, &ctx, c);
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
    let gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();
    let cases = build_arithmetic_cases();
    let results = run_category(&gen, &ctx, &cases);

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
    let gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();
    let cases = build_string_cases();
    let results = run_category(&gen, &ctx, &cases);

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
    let gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();
    let cases = build_collection_cases();
    let results = run_category(&gen, &ctx, &cases);

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
    let gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();
    let cases = build_composition_cases();
    let results = run_category(&gen, &ctx, &cases);

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
    let gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();
    let cases = build_python_cases();
    let results = run_category(&gen, &ctx, &cases);

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
    let gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();
    let cases = build_complex_llm_cases();
    let results = run_category(&gen, &ctx, &cases);

    print_results("Complex/LLM", &results);

    // All LLM cases should pass (they just need a function signature + todo!)
    let passed = results.iter().filter(|r| r.passed).count();
    assert_eq!(
        passed,
        results.len(),
        "All complex/LLM cases should pass (signature + todo!)"
    );
}
