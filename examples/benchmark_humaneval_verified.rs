// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! HumanEval Verified Benchmark — Generate, Compile, Test, Report pass@1
//!
//! Run: `cargo run --example benchmark_humaneval_verified --features code_generation`
//!
//! This is the external validation benchmark. It generates Rust code for
//! HumanEval-style problems, compiles each one, runs the tests, and reports
//! pass@1 (fraction that compile AND pass all tests on first try).
//!
//! Unlike the discriminative HumanEval-Mini in psych-bench (which picks from
//! 4 candidates via HDC similarity), this benchmark actually GENERATES code
//! and VERIFIES it compiles and runs correctly.

use symthaea::hdc::code_encoder::CodeHDEncoder;
use symthaea::language::code_executor::CodeExecutor;
use symthaea::language::code_generator::CodeGenerator;
use symthaea::language::verified_generation::generate_verified_function;

/// A HumanEval-style problem with Rust signature and test cases
struct Problem {
    name: &'static str,
    purpose: &'static str,
    signature: &'static str,
    test_cases: Vec<(&'static str, &'static str)>,
}

fn humaneval_problems() -> Vec<Problem> {
    vec![
        Problem {
            name: "add",
            purpose: "Add two integers",
            signature: "fn add(a: i32, b: i32) -> i32",
            test_cases: vec![("add(2, 3)", "5"), ("add(0, 0)", "0"), ("add(-1, 1)", "0")],
        },
        Problem {
            name: "subtract",
            purpose: "Subtract b from a",
            signature: "fn subtract(a: i32, b: i32) -> i32",
            test_cases: vec![("subtract(5, 3)", "2"), ("subtract(0, 0)", "0")],
        },
        Problem {
            name: "multiply",
            purpose: "Multiply two numbers",
            signature: "fn multiply(a: i32, b: i32) -> i32",
            test_cases: vec![("multiply(3, 4)", "12"), ("multiply(0, 5)", "0")],
        },
        Problem {
            name: "absolute_value",
            purpose: "Return absolute value",
            signature: "fn absolute_value(x: i32) -> i32",
            test_cases: vec![("absolute_value(-5)", "5"), ("absolute_value(3)", "3"), ("absolute_value(0)", "0")],
        },
        Problem {
            name: "is_even",
            purpose: "Check if number is even",
            signature: "fn is_even(n: i32) -> bool",
            test_cases: vec![("is_even(4)", "true"), ("is_even(7)", "false"), ("is_even(0)", "true")],
        },
        Problem {
            name: "factorial",
            purpose: "Compute factorial of n",
            signature: "fn factorial(n: u64) -> u64",
            test_cases: vec![("factorial(0)", "1"), ("factorial(1)", "1"), ("factorial(5)", "120")],
        },
        Problem {
            name: "fibonacci",
            purpose: "Compute nth fibonacci number",
            signature: "fn fibonacci(n: u64) -> u64",
            test_cases: vec![("fibonacci(0)", "0"), ("fibonacci(1)", "1"), ("fibonacci(10)", "55")],
        },
        Problem {
            name: "reverse_string",
            purpose: "Reverse a string",
            signature: "fn reverse_string(s: &str) -> String",
            test_cases: vec![
                ("reverse_string(\"hello\")", "\"olleh\".to_string()"),
                ("reverse_string(\"\")", "\"\".to_string()"),
            ],
        },
        Problem {
            name: "to_uppercase",
            purpose: "Convert string to uppercase",
            signature: "fn to_uppercase(s: &str) -> String",
            test_cases: vec![("to_uppercase(\"hello\")", "\"HELLO\".to_string()")],
        },
        Problem {
            name: "string_length",
            purpose: "Get length of a string",
            signature: "fn string_length(s: &str) -> usize",
            test_cases: vec![("string_length(\"hello\")", "5"), ("string_length(\"\")", "0")],
        },
        Problem {
            name: "max_of_two",
            purpose: "Return the maximum of two integers",
            signature: "fn max_of_two(a: i32, b: i32) -> i32",
            test_cases: vec![("max_of_two(3, 7)", "7"), ("max_of_two(10, 2)", "10")],
        },
        Problem {
            name: "gcd",
            purpose: "Compute greatest common divisor",
            signature: "fn gcd(a: u64, b: u64) -> u64",
            test_cases: vec![("gcd(12, 8)", "4"), ("gcd(7, 3)", "1"), ("gcd(0, 5)", "5")],
        },
        Problem {
            name: "sum_list",
            purpose: "Sum all elements in the list",
            signature: "fn sum_list(nums: &[i32]) -> i32",
            test_cases: vec![("sum_list(&[1, 2, 3])", "6"), ("sum_list(&[])", "0")],
        },
        Problem {
            name: "filter_even",
            purpose: "Filter even numbers",
            signature: "fn filter_even(v: Vec<i32>) -> Vec<i32>",
            test_cases: vec![("filter_even(vec![1, 2, 3, 4, 5])", "vec![2, 4]")],
        },
        Problem {
            name: "sort_vec",
            purpose: "Sort a vector of integers",
            signature: "fn sort_vec(v: &[i32]) -> Vec<i32>",
            test_cases: vec![("sort_vec(&[3, 1, 2])", "vec![1, 2, 3]")],
        },
        Problem {
            name: "contains_substr",
            purpose: "Check if string contains a substring",
            signature: "fn contains_substr(s: &str, sub: &str) -> bool",
            test_cases: vec![("contains_substr(\"hello world\", \"world\")", "true")],
        },
        Problem {
            name: "is_palindrome",
            purpose: "Check if string is a palindrome",
            signature: "fn is_palindrome(s: &str) -> bool",
            test_cases: vec![
                ("is_palindrome(\"racecar\")", "true"),
                ("is_palindrome(\"hello\")", "false"),
            ],
        },
        Problem {
            name: "celsius_to_fahrenheit",
            purpose: "Convert celsius to fahrenheit",
            signature: "fn celsius_to_fahrenheit(c: f64) -> f64",
            test_cases: vec![("celsius_to_fahrenheit(0.0)", "32.0"), ("celsius_to_fahrenheit(100.0)", "212.0")],
        },
        Problem {
            name: "safe_divide",
            purpose: "Divide with zero-check returning Result",
            signature: "fn safe_divide(a: f64, b: f64) -> Result<f64, String>",
            test_cases: vec![("safe_divide(10.0, 2.0)", "Ok(5.0)")],
        },
        Problem {
            name: "clamp_value",
            purpose: "Clamp value between min and max",
            signature: "fn clamp_value(x: i32, min: i32, max: i32) -> i32",
            test_cases: vec![
                ("clamp_value(5, 0, 10)", "5"),
                ("clamp_value(-5, 0, 10)", "0"),
                ("clamp_value(15, 0, 10)", "10"),
            ],
        },
    ]
}

fn main() {
    println!("=== HumanEval Verified Benchmark (20 problems) ===\n");

    let encoder = CodeHDEncoder::new(512);
    let generator = CodeGenerator::new(encoder);
    let mut executor = CodeExecutor::with_real_execution();

    let problems = humaneval_problems();
    let mut compiled = 0usize;
    let mut tests_passed = 0usize;
    let mut verified = 0usize;
    let total = problems.len();

    for problem in &problems {
        let result = generate_verified_function(
            &generator,
            &mut executor,
            problem.name,
            problem.purpose,
            problem.signature,
            &problem.test_cases,
        );

        let status = if result.is_guaranteed() {
            verified += 1;
            compiled += 1;
            tests_passed += 1;
            "PASS"
        } else if result.compiled {
            compiled += 1;
            if result.tests_passed {
                tests_passed += 1;
                "PASS (no auto-tests)"
            } else {
                "COMPILED (tests fail)"
            }
        } else {
            "FAIL"
        };

        println!(
            "  {:30} {} (retries: {}, confidence: {:.0}%)",
            problem.name,
            status,
            result.compile_retries,
            result.confidence.confidence * 100.0
        );

        if !result.compile_errors.is_empty() {
            for err in &result.compile_errors {
                let short: String = err.chars().take(80).collect();
                println!("    ERROR: {}", short);
            }
        }
    }

    println!("\n=== Results ===");
    println!("  Compiled:     {}/{} ({:.0}%)", compiled, total, compiled as f64 / total as f64 * 100.0);
    println!("  Tests passed: {}/{} ({:.0}%)", tests_passed, total, tests_passed as f64 / total as f64 * 100.0);
    println!("  Verified:     {}/{} ({:.0}%)", verified, total, verified as f64 / total as f64 * 100.0);
    println!("\n  pass@1 = {:.1}%", verified as f64 / total as f64 * 100.0);
}
