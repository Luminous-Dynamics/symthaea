// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rust-native CodeOrchestrator comparison benchmark.
//!
//! The existing `tests/code_generation_benchmark.rs` tests the raw
//! `CodeGenerator` directly and only checks weak string fragments for its
//! `expects_llm` cases (e.g. "pub fn dijkstra" exists, nothing about
//! correctness). This harness instead runs 3 of those same harder cases
//! (reusing their purpose/signature) through the full `CodingAgent` pipeline
//! — the actual consumer of `CodeOrchestrator`/`MagiCodeBridge` — with real
//! `rustc`+test verification, comparing `use_orchestrator: false` (raw
//! IntelligentDispatcher) against `use_orchestrator: true` (native/analogy/LLM
//! cascade, each compiler-verified). This is the first fair, Rust-native test
//! of whether the orchestrator stack adds real value — everything measured
//! on Python HumanEval up to now was a no-op for the orchestrator, since its
//! only verification path (`execute_rust_with_inline_tests`) can't accept
//! Python candidates at all (see docs/CODE_ABILITY_IMPROVEMENT_PLAN.md,
//! 2026-07-04).
//!
//! Run:
//!   cargo run --example rust_orchestrator_benchmark --features code_generation

use std::path::PathBuf;
use std::time::Instant;
use symthaea::coding_agent::{CodingAgent, CodingAgentConfig};
use symthaea::language::code_executor::CodeExecutor;

struct Case {
    name: &'static str,
    purpose: &'static str,
    signature: &'static str,
    /// Extra source (e.g. struct defs) the test harness needs alongside the
    /// generated function.
    preamble: &'static str,
    /// Rust test assertions appended after the generated function.
    test_body: &'static str,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "knapsack",
            purpose: "Implement dynamic programming 0/1 knapsack solver",
            signature: "fn knapsack(weights: &[u32], values: &[u32], capacity: u32) -> u32",
            preamble: "",
            test_body: "
                assert_eq!(knapsack(&[10, 20, 30], &[60, 100, 120], 50), 220);
                assert_eq!(knapsack(&[], &[], 10), 0);
                assert_eq!(knapsack(&[5], &[10], 4), 0);
                assert_eq!(knapsack(&[5], &[10], 5), 10);
            ",
        },
        Case {
            name: "http_parse",
            purpose: "Parse an HTTP request line into method, path, and version",
            signature: "fn http_parse(request: &str) -> (String, String, String)",
            preamble: "",
            test_body: "
                assert_eq!(
                    http_parse(\"GET /index.html HTTP/1.1\"),
                    (\"GET\".to_string(), \"/index.html\".to_string(), \"HTTP/1.1\".to_string())
                );
                assert_eq!(
                    http_parse(\"POST /api/users HTTP/2\"),
                    (\"POST\".to_string(), \"/api/users\".to_string(), \"HTTP/2\".to_string())
                );
            ",
        },
        Case {
            name: "binary_tree_traversal",
            purpose: "Implement in-order binary tree traversal collecting node values",
            signature: "fn inorder(root: &TreeNode) -> Vec<i32>",
            preamble: "
                pub struct TreeNode {
                    pub val: i32,
                    pub left: Option<Box<TreeNode>>,
                    pub right: Option<Box<TreeNode>>,
                }
                impl TreeNode {
                    fn leaf(val: i32) -> Self {
                        TreeNode { val, left: None, right: None }
                    }
                }
            ",
            test_body: "
                // Tree:      2
                //           / \\
                //          1   3
                let tree = TreeNode {
                    val: 2,
                    left: Some(Box::new(TreeNode::leaf(1))),
                    right: Some(Box::new(TreeNode::leaf(3))),
                };
                assert_eq!(inorder(&tree), vec![1, 2, 3]);
            ",
        },
        Case {
            name: "fizzbuzz",
            purpose: "Return a Vec<String> for the numbers 1..=n where multiples of 3 become \"Fizz\", multiples of 5 become \"Buzz\", multiples of both become \"FizzBuzz\", and all others are the number as a decimal string",
            signature: "fn fizzbuzz(n: u32) -> Vec<String>",
            preamble: "",
            test_body: "
                assert_eq!(fizzbuzz(5), vec![\"1\", \"2\", \"Fizz\", \"4\", \"Buzz\"]);
                assert_eq!(fizzbuzz(15)[14], \"FizzBuzz\".to_string());
                assert_eq!(fizzbuzz(3), vec![\"1\", \"2\", \"Fizz\"]);
            ",
        },
        Case {
            name: "is_palindrome",
            purpose: "Return true if the string reads the same forwards and backwards character-for-character (no normalization); the empty string and single characters are palindromes",
            signature: "fn is_palindrome(s: &str) -> bool",
            preamble: "",
            test_body: "
                assert!(is_palindrome(\"racecar\"));
                assert!(!is_palindrome(\"hello\"));
                assert!(is_palindrome(\"\"));
                assert!(is_palindrome(\"a\"));
            ",
        },
        Case {
            name: "fibonacci",
            purpose: "Return the nth Fibonacci number where fib(0) = 0, fib(1) = 1, and fib(n) = fib(n-1) + fib(n-2)",
            signature: "fn fibonacci(n: u32) -> u64",
            preamble: "",
            test_body: "
                assert_eq!(fibonacci(0), 0);
                assert_eq!(fibonacci(1), 1);
                assert_eq!(fibonacci(10), 55);
                assert_eq!(fibonacci(20), 6765);
            ",
        },
        Case {
            name: "gcd",
            purpose: "Return the greatest common divisor of two non-negative integers (gcd(x, 0) = x)",
            signature: "fn gcd(a: u64, b: u64) -> u64",
            preamble: "",
            test_body: "
                assert_eq!(gcd(48, 36), 12);
                assert_eq!(gcd(17, 5), 1);
                assert_eq!(gcd(0, 5), 5);
                assert_eq!(gcd(100, 10), 10);
            ",
        },
        Case {
            name: "reverse_words",
            purpose: "Split the input on whitespace, reverse the order of the words, and join them with a single space",
            signature: "fn reverse_words(s: &str) -> String",
            preamble: "",
            test_body: "
                assert_eq!(reverse_words(\"the quick brown fox\"), \"fox brown quick the\");
                assert_eq!(reverse_words(\"hello\"), \"hello\");
                assert_eq!(reverse_words(\"a b\"), \"b a\");
            ",
        },
        Case {
            name: "two_sum",
            purpose: "Return Some((i, j)) with i < j for the unique pair of indices whose values sum to target, or None if no such pair exists",
            signature: "fn two_sum(nums: &[i32], target: i32) -> Option<(usize, usize)>",
            preamble: "",
            test_body: "
                assert_eq!(two_sum(&[2, 7, 11, 15], 9), Some((0, 1)));
                assert_eq!(two_sum(&[3, 2, 4], 6), Some((1, 2)));
                assert_eq!(two_sum(&[1, 2, 3], 7), None);
            ",
        },
        Case {
            name: "count_vowels",
            purpose: "Count the vowels (a, e, i, o, u, case-insensitive) in the string",
            signature: "fn count_vowels(s: &str) -> usize",
            preamble: "",
            test_body: "
                assert_eq!(count_vowels(\"hello\"), 2);
                assert_eq!(count_vowels(\"AEIOU\"), 5);
                assert_eq!(count_vowels(\"xyz\"), 0);
            ",
        },
        Case {
            name: "roman_to_int",
            purpose: "Convert an uppercase Roman numeral string to its integer value, honoring subtractive pairs (IV=4, IX=9, XL=40, XC=90, CD=400, CM=900)",
            signature: "fn roman_to_int(s: &str) -> i32",
            preamble: "",
            test_body: "
                assert_eq!(roman_to_int(\"III\"), 3);
                assert_eq!(roman_to_int(\"IX\"), 9);
                assert_eq!(roman_to_int(\"LVIII\"), 58);
                assert_eq!(roman_to_int(\"MCMXCIV\"), 1994);
            ",
        },
        Case {
            name: "merge_sorted",
            purpose: "Merge two already-sorted ascending slices into a single sorted ascending Vec",
            signature: "fn merge_sorted(a: &[i32], b: &[i32]) -> Vec<i32>",
            preamble: "",
            test_body: "
                assert_eq!(merge_sorted(&[1, 3, 5], &[2, 4, 6]), vec![1, 2, 3, 4, 5, 6]);
                assert_eq!(merge_sorted(&[], &[1]), vec![1]);
                assert_eq!(merge_sorted(&[1, 2], &[]), vec![1, 2]);
            ",
        },
        Case {
            name: "max_subarray",
            purpose: "Return the maximum sum of any contiguous non-empty subarray (Kadane's algorithm)",
            signature: "fn max_subarray(nums: &[i32]) -> i32",
            preamble: "",
            test_body: "
                assert_eq!(max_subarray(&[-2, 1, -3, 4, -1, 2, 1, -5, 4]), 6);
                assert_eq!(max_subarray(&[1]), 1);
                assert_eq!(max_subarray(&[-1, -2, -3]), -1);
                assert_eq!(max_subarray(&[5, 4, -1, 7, 8]), 23);
            ",
        },
        Case {
            name: "valid_parens",
            purpose: "Return true if the brackets in the string are balanced and correctly nested; the string contains only the characters ()[]{}",
            signature: "fn valid_parens(s: &str) -> bool",
            preamble: "",
            test_body: "
                assert!(valid_parens(\"()\"));
                assert!(valid_parens(\"()[]{}\"));
                assert!(!valid_parens(\"(]\"));
                assert!(!valid_parens(\"([)]\"));
                assert!(valid_parens(\"{[]}\"));
            ",
        },
    ]
}

fn run_agent(case: &Case, use_orchestrator: bool) -> (bool, bool, Option<String>, u128) {
    let start = Instant::now();
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let config = CodingAgentConfig {
        max_iterations: 1, // single-attempt pass@1 (truer metric, ~5x faster than 5-round self-repair)
        working_dir: temp_dir.path().to_path_buf(),
        target_file: Some(PathBuf::from("solution.rs")),
        use_local_llm: true,
        use_orchestrator,
        enable_self_modification: use_orchestrator,
        ..Default::default()
    };

    let task = format!(
        "{}\n{}\n// Complete the function above.\n",
        case.preamble, case.signature
    );

    let mut agent = CodingAgent::new(config).expect("agent");
    let _ = agent.run(&format!("{}\n\n{}", case.purpose, task));

    let target = temp_dir.path().join("solution.rs");
    let generated = std::fs::read_to_string(&target).unwrap_or_default();
    let elapsed = start.elapsed().as_millis();

    if generated.trim().is_empty() {
        return (false, false, Some("no code generated".to_string()), elapsed);
    }

    let full_source = format!(
        "#![allow(unused, dead_code)]\n{}\n{}\n\n#[cfg(test)]\nmod tests {{\n    use super::*;\n    #[test]\n    fn t() {{\n{}\n    }}\n}}\n",
        case.preamble, generated, case.test_body
    );

    let mut executor = CodeExecutor::with_real_execution();
    let result = executor.execute_rust_with_inline_tests(&full_source);

    let passed = result.compiled && result.tests_passed > 0 && result.tests_failed == 0;
    let error = if !result.compiled {
        result.compile_errors.first().cloned()
    } else if result.tests_failed > 0 {
        Some(result.test_output.clone())
    } else {
        None
    };

    (passed, result.compiled, error, elapsed)
}

fn main() {
    use std::collections::HashSet;
    use std::fs::OpenOptions;
    use std::io::{BufRead, Write};

    let all = cases();
    let total = all.len();

    // Resumable + incremental: each case's result is appended to a TSV as soon
    // as it completes, and a restart skips already-recorded cases. This makes the
    // ~30-min run survive the contended cargo-gate / session teardowns that killed
    // ~6 non-incremental attempts. Row: name\tpass_off\tcomp_off\tpass_on\tcomp_on.
    let results_path = std::env::var("ORCH_AB_RESULTS")
        .unwrap_or_else(|_| "/tmp/rust_orch_ab_results.tsv".to_string());

    let mut done: HashSet<String> = HashSet::new();
    if let Ok(f) = std::fs::File::open(&results_path) {
        for line in std::io::BufReader::new(f).lines().map_while(Result::ok) {
            if let Some(name) = line.split('\t').next() {
                done.insert(name.to_string());
            }
        }
    }
    println!(
        "Rust-native CodeOrchestrator A/B — {} cases (max_iterations=1, pass@1), resumable at {} ({} already recorded).\n",
        total,
        results_path,
        done.len()
    );

    for case in &all {
        if done.contains(case.name) {
            println!("=== {} (skip: already recorded) ===", case.name);
            continue;
        }
        println!("=== {} ===", case.name);
        let (passed_off, compiled_off, err_off, ms_off) = run_agent(case, false);
        println!(
            "  orchestrator=false: passed={} compiled={} ({}ms){}",
            passed_off,
            compiled_off,
            ms_off,
            err_off
                .as_ref()
                .map(|e| format!(" — {}", e.chars().take(120).collect::<String>()))
                .unwrap_or_default()
        );
        let (passed_on, compiled_on, err_on, ms_on) = run_agent(case, true);
        println!(
            "  orchestrator=true:  passed={} compiled={} ({}ms){}",
            passed_on,
            compiled_on,
            ms_on,
            err_on
                .as_ref()
                .map(|e| format!(" — {}", e.chars().take(120).collect::<String>()))
                .unwrap_or_default()
        );
        println!();
        if let Ok(mut f) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&results_path)
        {
            let _ = writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}",
                case.name, passed_off as u8, compiled_off as u8, passed_on as u8, compiled_on as u8
            );
        }
    }

    // Summary from the full accumulated file (survives resumes).
    let (mut pass_off, mut comp_off, mut pass_on, mut comp_on, mut n) = (0, 0, 0, 0, 0usize);
    if let Ok(f) = std::fs::File::open(&results_path) {
        for line in std::io::BufReader::new(f).lines().map_while(Result::ok) {
            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() == 5 {
                pass_off += (cols[1] == "1") as usize;
                comp_off += (cols[2] == "1") as usize;
                pass_on += (cols[3] == "1") as usize;
                comp_on += (cols[4] == "1") as usize;
                n += 1;
            }
        }
    }
    let denom = n.max(1);
    let pct = |x: usize| 100.0 * x as f64 / denom as f64;
    println!(
        "========== SUMMARY (pass@1, N={} recorded of {}) ==========",
        n, total
    );
    println!(
        "  orchestrator=false: {}/{} passed ({:.1}%), {}/{} compiled ({:.1}%)",
        pass_off,
        n,
        pct(pass_off),
        comp_off,
        n,
        pct(comp_off)
    );
    println!(
        "  orchestrator=true:  {}/{} passed ({:.1}%), {}/{} compiled ({:.1}%)",
        pass_on,
        n,
        pct(pass_on),
        comp_on,
        n,
        pct(comp_on)
    );
    let delta = pass_on as i64 - pass_off as i64;
    println!(
        "  delta (on - off): {:+} passed ({:+.1} pp) — orchestrator {}",
        delta,
        pct(pass_on) - pct(pass_off),
        if delta > 0 {
            "HELPS"
        } else if delta < 0 {
            "HURTS"
        } else {
            "no measured effect on this set"
        }
    );
}
