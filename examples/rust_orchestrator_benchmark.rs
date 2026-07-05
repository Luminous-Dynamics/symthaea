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
    ]
}

fn run_agent(case: &Case, use_orchestrator: bool) -> (bool, bool, Option<String>, u128) {
    let start = Instant::now();
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let config = CodingAgentConfig {
        max_iterations: 5,
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
    println!("Rust-native CodeOrchestrator comparison — 3 cases, requires LLM escalation\n");

    for case in cases() {
        println!("=== {} ===", case.name);
        let (passed_off, compiled_off, err_off, ms_off) = run_agent(&case, false);
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

        let (passed_on, compiled_on, err_on, ms_on) = run_agent(&case, true);
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
    }
}
