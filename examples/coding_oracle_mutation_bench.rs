// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mutation controls for Symthaea's Rust coding benchmark.
//!
//! The harness must demonstrate both directions:
//! - each reference implementation compiles and passes its complete oracle;
//! - each behaviorally incorrect, type-correct mutant compiles and fails that oracle.
//!
//! A compile failure, simulated execution, or incomplete test run is not a killed
//! behavioral mutant. This binary reports those cases as qualification failures.

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

use symthaea::language::code_executor::CodeExecutor;

struct Case {
    id: &'static str,
    reference: &'static str,
    mutant: &'static str,
    tests: &'static str,
}

struct VariantResult {
    status: &'static str,
    compiled: bool,
    simulated: bool,
    tests_expected: usize,
    tests_executed: usize,
    tests_passed: usize,
    tests_failed: usize,
    elapsed_ms: u128,
    compile_errors: Vec<String>,
    runtime_error: Option<String>,
}

impl VariantResult {
    fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "status": self.status,
            "compiled": self.compiled,
            "simulated": self.simulated,
            "tests_expected": self.tests_expected,
            "tests_executed": self.tests_executed,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "elapsed_ms": self.elapsed_ms,
            "compile_errors": self.compile_errors.clone(),
            "runtime_error": self.runtime_error.clone(),
        })
    }
}

const CASES: &[Case] = &[
    Case {
        id: "add_signed_integers",
        reference: "fn add(a: i32, b: i32) -> i32 { a + b }",
        mutant: "fn add(a: i32, b: i32) -> i32 { a - b }",
        tests: r#"
#[test]
fn positive_values() { assert_eq!(add(2, 3), 5); }
#[test]
fn cancellation() { assert_eq!(add(-4, 4), 0); }
#[test]
fn zero_identity() { assert_eq!(add(0, 7), 7); }
"#,
    },
    Case {
        id: "absolute_value",
        reference: "fn abs_value(value: i32) -> i32 { if value < 0 { -value } else { value } }",
        mutant: "fn abs_value(value: i32) -> i32 { value }",
        tests: r#"
#[test]
fn negative_input() { assert_eq!(abs_value(-9), 9); }
#[test]
fn positive_input() { assert_eq!(abs_value(4), 4); }
#[test]
fn zero_input() { assert_eq!(abs_value(0), 0); }
"#,
    },
    Case {
        id: "greatest_common_divisor",
        reference: "fn gcd(mut a: u64, mut b: u64) -> u64 { while b != 0 { let remainder = a % b; a = b; b = remainder; } a }",
        mutant: "fn gcd(a: u64, b: u64) -> u64 { a.min(b) }",
        tests: r#"
#[test]
fn shared_factor() { assert_eq!(gcd(12, 8), 4); }
#[test]
fn coprime_values() { assert_eq!(gcd(7, 13), 1); }
#[test]
fn zero_left_operand() { assert_eq!(gcd(0, 9), 9); }
"#,
    },
    Case {
        id: "clamp_both_bounds",
        reference: "fn clamp_value(value: i32, lower: i32, upper: i32) -> i32 { value.max(lower).min(upper) }",
        mutant: "fn clamp_value(value: i32, lower: i32, _upper: i32) -> i32 { value.max(lower) }",
        tests: r#"
#[test]
fn below_lower_bound() { assert_eq!(clamp_value(-5, 0, 10), 0); }
#[test]
fn above_upper_bound() { assert_eq!(clamp_value(20, 0, 10), 10); }
#[test]
fn inside_bounds() { assert_eq!(clamp_value(5, 0, 10), 5); }
"#,
    },
    Case {
        id: "evenness_for_negative_values",
        reference: "fn is_even(value: i32) -> bool { value % 2 == 0 }",
        mutant: "fn is_even(value: i32) -> bool { value % 2 == 1 }",
        tests: r#"
#[test]
fn positive_even() { assert!(is_even(8)); }
#[test]
fn negative_even() { assert!(is_even(-8)); }
#[test]
fn negative_odd() { assert!(!is_even(-3)); }
"#,
    },
    Case {
        id: "case_insensitive_vowel_count",
        reference: "fn count_vowels(input: &str) -> usize { input.chars().filter(|ch| \"aeiouAEIOU\".contains(*ch)).count() }",
        mutant: "fn count_vowels(input: &str) -> usize { input.chars().filter(|ch| \"aeiou\".contains(*ch)).count() }",
        tests: r#"
#[test]
fn lowercase_vowels() { assert_eq!(count_vowels("symthaea"), 3); }
#[test]
fn uppercase_vowels() { assert_eq!(count_vowels("AEIOU"), 5); }
#[test]
fn consonants_only() { assert_eq!(count_vowels("rhythm"), 0); }
"#,
    },
    Case {
        id: "reverse_character_order",
        reference: "fn reverse_text(input: &str) -> String { input.chars().rev().collect::<String>() }",
        mutant: "fn reverse_text(input: &str) -> String { input.to_string() }",
        tests: r#"
#[test]
fn reverse_word() { assert_eq!(reverse_text("rust"), "tsur"); }
#[test]
fn reverse_empty() { assert_eq!(reverse_text(""), ""); }
#[test]
fn reverse_repeated_characters() { assert_eq!(reverse_text("abca"), "acba"); }
"#,
    },
    Case {
        id: "palindrome_detection",
        reference: "fn is_palindrome(input: &str) -> bool { input.chars().eq(input.chars().rev()) }",
        mutant: "fn is_palindrome(input: &str) -> bool { input.len() <= 1 }",
        tests: r#"
#[test]
fn odd_length_palindrome() { assert!(is_palindrome("racecar")); }
#[test]
fn even_length_palindrome() { assert!(is_palindrome("abba")); }
#[test]
fn non_palindrome() { assert!(!is_palindrome("rust")); }
"#,
    },
    Case {
        id: "sort_and_deduplicate",
        reference: "fn dedup_sorted(values: &mut Vec<i32>) { values.sort_unstable(); values.dedup(); }",
        mutant: "fn dedup_sorted(values: &mut Vec<i32>) { values.sort_unstable(); }",
        tests: r#"
#[test]
fn duplicate_values_removed() { let mut values = vec![3, 1, 3, 2, 2]; dedup_sorted(&mut values); assert_eq!(values, vec![1, 2, 3]); }
#[test]
fn empty_input() { let mut values: Vec<i32> = Vec::new(); dedup_sorted(&mut values); assert!(values.is_empty()); }
"#,
    },
    Case {
        id: "first_negative_index",
        reference: "fn first_negative_index(values: &[i32]) -> Option<usize> { values.iter().position(|value| *value < 0) }",
        mutant: "fn first_negative_index(values: &[i32]) -> Option<usize> { values.iter().position(|value| *value == 0) }",
        tests: r#"
#[test]
fn negative_after_zero() { assert_eq!(first_negative_index(&[2, 0, -1]), Some(2)); }
#[test]
fn no_negative_value() { assert_eq!(first_negative_index(&[1, 2, 3]), None); }
#[test]
fn first_value_is_negative() { assert_eq!(first_negative_index(&[-3, 0, -1]), Some(0)); }
"#,
    },
    Case {
        id: "second_largest_distinct",
        reference: "fn second_largest(values: &[i32]) -> Option<i32> { let mut sorted = values.to_vec(); sorted.sort_unstable(); sorted.dedup(); if sorted.len() < 2 { None } else { Some(sorted[sorted.len() - 2]) } }",
        mutant: "fn second_largest(values: &[i32]) -> Option<i32> { values.iter().max().copied() }",
        tests: r#"
#[test]
fn duplicate_maximum() { assert_eq!(second_largest(&[3, 8, 4, 8]), Some(4)); }
#[test]
fn all_values_equal() { assert_eq!(second_largest(&[7, 7]), None); }
#[test]
fn empty_input() { assert_eq!(second_largest(&[]), None); }
"#,
    },
    Case {
        id: "zero_has_one_digit",
        reference: "fn digit_count(mut value: u64) -> usize { let mut count = 1; while value >= 10 { count += 1; value /= 10; } count }",
        mutant: "fn digit_count(mut value: u64) -> usize { let mut count = 0; while value > 0 { count += 1; value /= 10; } count }",
        tests: r#"
#[test]
fn zero_digit_count() { assert_eq!(digit_count(0), 1); }
#[test]
fn one_digit_count() { assert_eq!(digit_count(7), 1); }
#[test]
fn multi_digit_count() { assert_eq!(digit_count(1000), 4); }
"#,
    },
    Case {
        id: "nondecreasing_sorted_check",
        reference: "fn is_sorted(values: &[i32]) -> bool { values.windows(2).all(|pair| pair[0] <= pair[1]) }",
        mutant: "fn is_sorted(values: &[i32]) -> bool { values.windows(2).all(|pair| pair[0] < pair[1]) }",
        tests: r#"
#[test]
fn empty_is_sorted() { assert!(is_sorted(&[])); }
#[test]
fn duplicates_are_allowed() { assert!(is_sorted(&[1, 2, 2, 3])); }
#[test]
fn descending_pair_is_not_sorted() { assert!(!is_sorted(&[2, 1])); }
"#,
    },
    Case {
        id: "upper_bound_with_duplicates",
        reference: "fn upper_bound(values: &[i32], target: i32) -> usize { let mut low = 0; let mut high = values.len(); while low < high { let middle = low + (high - low) / 2; if values[middle] <= target { low = middle + 1; } else { high = middle; } } low }",
        mutant: "fn upper_bound(values: &[i32], target: i32) -> usize { let mut low = 0; let mut high = values.len(); while low < high { let middle = low + (high - low) / 2; if values[middle] < target { low = middle + 1; } else { high = middle; } } low }",
        tests: r#"
#[test]
fn skips_all_equal_values() { assert_eq!(upper_bound(&[1, 2, 2, 2, 5], 2), 4); }
#[test]
fn target_above_maximum() { assert_eq!(upper_bound(&[1, 2, 5], 9), 3); }
#[test]
fn target_below_minimum() { assert_eq!(upper_bound(&[1, 2, 5], 0), 0); }
"#,
    },
    Case {
        id: "merge_sorted_slices",
        reference: "fn merge_sorted(a: &[i32], b: &[i32]) -> Vec<i32> { let mut output = Vec::with_capacity(a.len() + b.len()); let (mut i, mut j) = (0, 0); while i < a.len() && j < b.len() { if a[i] <= b[j] { output.push(a[i]); i += 1; } else { output.push(b[j]); j += 1; } } output.extend_from_slice(&a[i..]); output.extend_from_slice(&b[j..]); output }",
        mutant: "fn merge_sorted(a: &[i32], _b: &[i32]) -> Vec<i32> { let mut output = a.to_vec(); output.sort_unstable(); output }",
        tests: r#"
#[test]
fn interleaved_inputs() { assert_eq!(merge_sorted(&[1, 3, 5], &[2, 4, 6]), vec![1, 2, 3, 4, 5, 6]); }
#[test]
fn empty_left_side() { assert_eq!(merge_sorted(&[], &[1, 2]), vec![1, 2]); }
#[test]
fn empty_right_side() { assert_eq!(merge_sorted(&[1, 2], &[]), vec![1, 2]); }
"#,
    },
    Case {
        id: "checked_factorial_overflow",
        reference: "fn factorial(value: u32) -> Option<u64> { let mut result = 1u64; for factor in 2..=value { result = result.checked_mul(factor as u64)?; } Some(result) }",
        mutant: "fn factorial(value: u32) -> Option<u64> { let mut result = 1u64; for factor in 2..=value { result *= factor as u64; } Some(result) }",
        tests: r#"
#[test]
fn factorial_zero() { assert_eq!(factorial(0), Some(1)); }
#[test]
fn factorial_five() { assert_eq!(factorial(5), Some(120)); }
#[test]
fn factorial_overflow_returns_none() { assert_eq!(factorial(21), None); }
"#,
    },
];

fn count_expected_tests(test_source: &str) -> usize {
    test_source
        .lines()
        .filter(|line| line.trim_start().starts_with("#[test]"))
        .count()
}

fn digest(content: &str) -> String {
    blake3::hash(content.as_bytes()).to_hex().to_string()
}

fn run_variant(
    executor: &mut CodeExecutor,
    source: &str,
    test_source: &str,
    tests_expected: usize,
) -> VariantResult {
    let result = executor.execute_rust(source, Some(test_source));
    let tests_executed = result.tests_passed + result.tests_failed;
    let status = if result.simulated {
        "simulated_unverified"
    } else if !result.compiled {
        "compile_failed"
    } else if tests_expected == 0 {
        "missing_oracle"
    } else if tests_executed == 0 {
        "no_tests_executed"
    } else if tests_executed != tests_expected {
        "test_count_mismatch"
    } else if result.tests_failed > 0 {
        "failed"
    } else if result.tests_passed == tests_expected {
        "passed"
    } else {
        "unverified"
    };

    VariantResult {
        status,
        compiled: result.compiled,
        simulated: result.simulated,
        tests_expected,
        tests_executed,
        tests_passed: result.tests_passed,
        tests_failed: result.tests_failed,
        elapsed_ms: result.elapsed.as_millis(),
        compile_errors: result.compile_errors,
        runtime_error: result.runtime_error,
    }
}

fn rustc_version() -> Option<String> {
    let output = Command::new("rustc").arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn receipt_path() -> PathBuf {
    let mut args = std::env::args().skip(1);
    let mut path = None;
    while let Some(argument) = args.next() {
        if argument == "--json-out" {
            match args.next() {
                Some(value) => path = Some(PathBuf::from(value)),
                None => {
                    eprintln!("--json-out requires a path");
                    std::process::exit(2);
                }
            }
        } else {
            eprintln!("unknown argument: {argument}");
            std::process::exit(2);
        }
    }
    path.unwrap_or_else(|| std::env::temp_dir().join("symthaea-coding-oracle-mutation-v1.json"))
}

fn main() {
    let started = Instant::now();
    let mut executor = CodeExecutor::with_real_execution();
    let mut records = Vec::with_capacity(CASES.len());
    let mut reference_passes = 0usize;
    let mut mutants_killed = 0usize;
    let mut qualification_errors = Vec::new();
    let mut aggregate = String::new();

    for case in CASES {
        let expected = count_expected_tests(case.tests);
        if expected == 0 {
            qualification_errors.push(format!("{}: fixture has no tests", case.id));
        }

        let reference = run_variant(&mut executor, case.reference, case.tests, expected);
        let mutant = run_variant(&mut executor, case.mutant, case.tests, expected);
        let reference_passed = reference.status == "passed";
        let mutation_killed = reference_passed && mutant.status == "failed";

        if reference_passed {
            reference_passes += 1;
        } else {
            qualification_errors.push(format!(
                "{}: reference control status was {} (expected passed)",
                case.id, reference.status
            ));
        }

        if mutation_killed {
            mutants_killed += 1;
        } else {
            qualification_errors.push(format!(
                "{}: mutant status was {} (expected a complete behavioral failure)",
                case.id, mutant.status
            ));
        }

        let test_digest = digest(case.tests);
        let reference_digest = digest(case.reference);
        let mutant_digest = digest(case.mutant);
        aggregate.push_str(case.id);
        aggregate.push('\n');
        aggregate.push_str(&reference_digest);
        aggregate.push('\n');
        aggregate.push_str(&mutant_digest);
        aggregate.push('\n');
        aggregate.push_str(&test_digest);
        aggregate.push('\n');

        eprintln!(
            "[{}] {} — reference: {}, mutant: {} ({} tests)",
            if reference_passed && mutation_killed { "PASS" } else { "FAIL" },
            case.id,
            reference.status,
            mutant.status,
            expected
        );

        records.push(serde_json::json!({
            "task_id": case.id,
            "fixture_digest_blake3": test_digest,
            "reference_source_digest_blake3": reference_digest,
            "mutant_source_digest_blake3": mutant_digest,
            "tests_expected": expected,
            "reference": reference.to_json(),
            "mutant": mutant.to_json(),
            "reference_control_passed": reference_passed,
            "mutation_killed": mutation_killed,
        }));
    }

    let qualified = qualification_errors.is_empty()
        && reference_passes == CASES.len()
        && mutants_killed == CASES.len();
    let total_tests: usize = CASES.iter().map(|case| count_expected_tests(case.tests)).sum();
    let source_revision = std::env::var("GIT_COMMIT_SHA")
        .ok()
        .or_else(|| option_env!("GIT_COMMIT_SHA").map(str::to_string));

    let receipt = serde_json::json!({
        "schema_version": 1,
        "benchmark": "symthaea_coding_oracle_mutation",
        "source_revision": source_revision,
        "fixture_digest_blake3": digest(&aggregate),
        "runtime": {
            "rustc_version": rustc_version(),
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "debug_assertions": cfg!(debug_assertions),
            "code_generation_feature": cfg!(feature = "code_generation"),
            "executor": "CodeExecutor::with_real_execution",
        },
        "summary": {
            "task_count": CASES.len(),
            "expected_assertion_tests": total_tests,
            "reference_controls_passed": reference_passes,
            "mutants_killed": mutants_killed,
            "mutation_kill_rate": mutants_killed as f64 / CASES.len() as f64,
            "qualified": qualified,
            "elapsed_ms": started.elapsed().as_millis(),
            "qualification_errors": qualification_errors,
        },
        "tasks": records,
    });

    let path = receipt_path();
    if let Some(parent) = path.parent() {
        if let Err(error) = std::fs::create_dir_all(parent) {
            eprintln!("failed to create receipt directory {}: {error}", parent.display());
            std::process::exit(2);
        }
    }
    let encoded = match serde_json::to_vec_pretty(&receipt) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("failed to serialize receipt: {error}");
            std::process::exit(2);
        }
    };
    if let Err(error) = std::fs::write(&path, encoded) {
        eprintln!("failed to write receipt {}: {error}", path.display());
        std::process::exit(2);
    }

    eprintln!(
        "Oracle mutation qualification: {}/{} references passed; {}/{} mutants killed; receipt: {}",
        reference_passes,
        CASES.len(),
        mutants_killed,
        CASES.len(),
        path.display()
    );

    if !qualified {
        std::process::exit(1);
    }
}
