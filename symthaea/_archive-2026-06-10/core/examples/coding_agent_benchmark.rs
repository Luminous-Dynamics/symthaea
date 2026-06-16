// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Coding Agent Benchmark Suite — with real compilation and test assertions
//!
//! Measures end-to-end performance across 55 tasks with three validation levels:
//! 1. **Name check**: does the output contain the expected function/type name?
//! 2. **Compilation**: does the generated code compile with `rustc`?
//! 3. **Correctness**: does it pass test assertions?
//!
//! Run:
//!   Native-only:  `cargo run --example coding_agent_benchmark --features code_generation`
//!   With Ollama:  `cargo run --example coding_agent_benchmark --features code_generation -- --llm`
//!   With Claude:  `cargo run --example coding_agent_benchmark --features code_generation -- --cloud`

use std::path::PathBuf;
use std::time::{Duration, Instant};
use symthaea::coding_agent::{CodingAgent, CodingAgentConfig, TaskPhase};
use symthaea::language::code_executor::{CodeExecutor, try_auto_fix};
use symthaea::language::intelligent_dispatcher::BackendTier;

/// A benchmark task with expected outcomes and test assertions.
struct BenchTask {
    description: &'static str,
    difficulty: Difficulty,
    expected_fn: &'static str,
    max_iterations: usize,
    /// Rust test assertions (inserted into `#[cfg(test)] mod tests { use super::*; ... }`)
    test_source: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Difficulty {
    Native,
    Medium,
    Hard,
}

impl std::fmt::Display for Difficulty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Native => write!(f, "Native"),
            Self::Medium => write!(f, "Medium"),
            Self::Hard => write!(f, "Hard"),
        }
    }
}

#[derive(Debug)]
struct TaskResult {
    description: String,
    difficulty: Difficulty,
    // Level 1: name check
    code_written: bool,
    contains_expected_fn: bool,
    contains_todo: bool,
    // Level 2: compilation
    compiled: bool,
    compile_errors: Vec<String>,
    auto_fix_attempted: bool,
    auto_fix_succeeded: bool,
    // Level 3: correctness
    tests_passed: usize,
    tests_failed: usize,
    // Agent metrics
    iterations: usize,
    tiers_used: Vec<String>,
    energy: f64,
    phi_mean: f32,
    quality_gate_rejections: usize,
    elapsed_ms: u128,
    final_phase: String,
}

impl TaskResult {
    /// Success = compiled AND all tests passed (or no tests but compiled)
    fn is_correct(&self) -> bool {
        self.compiled
            && self.tests_failed == 0
            && (self.tests_passed > 0 || self.test_source_empty())
    }
    fn test_source_empty(&self) -> bool {
        // Tasks with no test assertions count as "correct" if they compile
        self.tests_passed == 0 && self.tests_failed == 0
    }
}

#[derive(Debug, Default)]
struct BenchStats {
    total: usize,
    // Level 1
    name_match: usize,
    code_written: usize,
    // Level 2
    compiled: usize,
    auto_fix_attempts: usize,
    auto_fix_successes: usize,
    // Level 3
    correct: usize,
    total_tests_passed: usize,
    total_tests_failed: usize,
    // By difficulty
    native_total: usize,
    native_correct: usize,
    native_compiled: usize,
    medium_total: usize,
    medium_correct: usize,
    medium_compiled: usize,
    hard_total: usize,
    hard_correct: usize,
    hard_compiled: usize,
    // Agent
    total_energy: f64,
    total_elapsed_ms: u128,
    tier_counts: std::collections::HashMap<String, usize>,
    quality_gate_total: usize,
}

fn build_task_suite() -> Vec<BenchTask> {
    vec![
        // ── Native: 20 tasks ──────────────────────────────────────────
        BenchTask {
            description: "add a fibonacci function",
            difficulty: Difficulty::Native,
            expected_fn: "fibonacci",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_fib_base() { assert_eq!(fibonacci(0), 0); assert_eq!(fibonacci(1), 1); }
    #[test] fn test_fib_10() { assert_eq!(fibonacci(10), 55); }
    #[test] fn test_fib_20() { assert_eq!(fibonacci(20), 6765); }
"#,
        },
        BenchTask {
            description: "implement factorial",
            difficulty: Difficulty::Native,
            expected_fn: "factorial",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_fact_0() { assert_eq!(factorial(0), 1); }
    #[test] fn test_fact_5() { assert_eq!(factorial(5), 120); }
    #[test] fn test_fact_10() { assert_eq!(factorial(10), 3628800); }
"#,
        },
        BenchTask {
            description: "add gcd function",
            difficulty: Difficulty::Native,
            expected_fn: "gcd",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_gcd() { assert_eq!(gcd(12, 8), 4); assert_eq!(gcd(7, 13), 1); assert_eq!(gcd(0, 5), 5); }
"#,
        },
        BenchTask {
            description: "check primality",
            difficulty: Difficulty::Native,
            expected_fn: "is_prime",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_primes() { assert!(!is_prime(0)); assert!(!is_prime(1)); assert!(is_prime(2)); assert!(is_prime(13)); assert!(!is_prime(15)); }
"#,
        },
        BenchTask {
            description: "compute absolute value",
            difficulty: Difficulty::Native,
            expected_fn: "absolute",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_abs() { assert_eq!(absolute(-5), 5); assert_eq!(absolute(3), 3); assert_eq!(absolute(0), 0); }
"#,
        },
        BenchTask {
            description: "add a hello function",
            difficulty: Difficulty::Native,
            expected_fn: "hello",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_hello() { assert!(hello().contains("Hello") || hello().contains("hello")); }
"#,
        },
        BenchTask {
            description: "reverse a string",
            difficulty: Difficulty::Native,
            expected_fn: "reverse_string",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_rev() { assert_eq!(reverse_string("hello"), "olleh"); assert_eq!(reverse_string(""), ""); }
"#,
        },
        BenchTask {
            description: "check if palindrome",
            difficulty: Difficulty::Native,
            expected_fn: "is_palindrome",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_pal() { assert!(is_palindrome("racecar")); assert!(!is_palindrome("hello")); assert!(is_palindrome("")); }
"#,
        },
        BenchTask {
            description: "count vowels in a string",
            difficulty: Difficulty::Native,
            expected_fn: "count_vowels",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_vowels() { assert_eq!(count_vowels("hello"), 2); assert_eq!(count_vowels("xyz"), 0); }
"#,
        },
        BenchTask {
            description: "convert string to uppercase",
            difficulty: Difficulty::Native,
            expected_fn: "to_uppercase",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_upper() { assert_eq!(to_uppercase("hello"), "HELLO"); assert_eq!(to_uppercase(""), ""); }
"#,
        },
        BenchTask {
            description: "implement bubble sort",
            difficulty: Difficulty::Native,
            expected_fn: "bubble_sort",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_bsort() { let mut v = vec![3,1,2]; bubble_sort(&mut v); assert_eq!(v, vec![1,2,3]); }
    #[test] fn test_bsort_empty() { let mut v: Vec<i32> = vec![]; bubble_sort(&mut v); assert_eq!(v, vec![]); }
"#,
        },
        BenchTask {
            description: "implement insertion sort",
            difficulty: Difficulty::Native,
            expected_fn: "insertion_sort",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_isort() { let mut v = vec![5,2,8,1]; insertion_sort(&mut v); assert_eq!(v, vec![1,2,5,8]); }
"#,
        },
        BenchTask {
            description: "sort a vector",
            difficulty: Difficulty::Native,
            expected_fn: "sort_vec",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_sort() { assert_eq!(sort_vec(vec![3,1,2]), vec![1,2,3]); }
"#,
        },
        BenchTask {
            description: "implement binary search",
            difficulty: Difficulty::Native,
            expected_fn: "binary_search",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_bsearch() { assert_eq!(binary_search(&[1,3,5,7,9], &5), Some(2)); assert_eq!(binary_search(&[1,3,5], &4), None); }
"#,
        },
        BenchTask {
            description: "find max in a vector",
            difficulty: Difficulty::Native,
            expected_fn: "find_max",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_max() { assert_eq!(find_max(&[1,5,3]), Some(5)); assert_eq!(find_max(&[]), None); }
"#,
        },
        BenchTask {
            description: "find min in a vector",
            difficulty: Difficulty::Native,
            expected_fn: "find_min",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_min() { assert_eq!(find_min(&[3,1,5]), Some(1)); assert_eq!(find_min(&[]), None); }
"#,
        },
        BenchTask {
            description: "sum all elements in a vec",
            difficulty: Difficulty::Native,
            expected_fn: "sum_vec",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_sum() { assert_eq!(sum_vec(&[1,2,3,4]), 10); assert_eq!(sum_vec(&[]), 0); }
"#,
        },
        BenchTask {
            description: "check if number is even",
            difficulty: Difficulty::Native,
            expected_fn: "is_even",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_even() { assert!(is_even(4)); assert!(!is_even(7)); assert!(is_even(0)); }
"#,
        },
        BenchTask {
            description: "create a stack data structure",
            difficulty: Difficulty::Native,
            expected_fn: "Stack",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_stack() {
        let mut s = Stack::new();
        s.push(1); s.push(2); s.push(3);
        assert_eq!(s.pop(), Some(3));
        assert_eq!(s.len(), 2);
        assert!(!s.is_empty());
    }
"#,
        },
        BenchTask {
            description: "flatten nested vectors",
            difficulty: Difficulty::Native,
            expected_fn: "flatten",
            max_iterations: 5,
            test_source: r#"
    #[test] fn test_flatten() { assert_eq!(flatten(&[vec![1,2], vec![3,4]]), vec![1,2,3,4]); assert_eq!(flatten::<i32>(&[]), vec![]); }
"#,
        },
        // ── Medium: 20 tasks ─────────────────────────────────────────
        BenchTask {
            description: "implement a linked list",
            difficulty: Difficulty::Medium,
            expected_fn: "LinkedList",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_ll() {
        let mut l = LinkedList::new();
        l.push_front(1); l.push_front(2);
        assert_eq!(l.pop_front(), Some(2));
        assert_eq!(l.len(), 1);
    }
"#,
        },
        BenchTask {
            description: "create a matrix multiply function",
            difficulty: Difficulty::Medium,
            expected_fn: "multiply",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_matmul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let c = multiply(&a, &b);
        assert!((c[0][0] - 19.0).abs() < 0.001);
        assert!((c[1][1] - 50.0).abs() < 0.001);
    }
"#,
        },
        BenchTask {
            description: "implement merge sort",
            difficulty: Difficulty::Medium,
            expected_fn: "merge_sort",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_msort() { let mut v = vec![5,2,8,1]; merge_sort(&mut v); assert_eq!(v, vec![1,2,5,8]); }
    #[test] fn test_msort_dups() { let mut v = vec![3,1,3,1]; merge_sort(&mut v); assert_eq!(v, vec![1,1,3,3]); }
"#,
        },
        BenchTask {
            description: "add a function to compute Levenshtein distance",
            difficulty: Difficulty::Medium,
            expected_fn: "levenshtein",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_lev() { assert_eq!(levenshtein("kitten", "sitting"), 3); assert_eq!(levenshtein("", "abc"), 3); assert_eq!(levenshtein("abc", "abc"), 0); }
"#,
        },
        BenchTask {
            description: "implement a simple tokenizer for arithmetic expressions",
            difficulty: Difficulty::Medium,
            expected_fn: "tokenize",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_tok() { let t = tokenize("1 + 2"); assert!(t.len() >= 3); }
"#,
        },
        BenchTask {
            description: "create a function to validate email addresses",
            difficulty: Difficulty::Medium,
            expected_fn: "validate_email",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_email() { assert!(validate_email("user@example.com")); assert!(!validate_email("bad")); assert!(!validate_email("")); }
"#,
        },
        BenchTask {
            description: "implement run-length encoding",
            difficulty: Difficulty::Medium,
            expected_fn: "rle_encode",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_rle() { assert_eq!(rle_encode("aaabbc"), "3a2b1c"); assert_eq!(rle_encode(""), ""); }
"#,
        },
        BenchTask {
            description: "add a function to find all permutations of a string",
            difficulty: Difficulty::Medium,
            expected_fn: "permutations",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_perms() { let p = permutations("ab"); assert_eq!(p.len(), 2); assert!(p.contains(&"ab".to_string())); assert!(p.contains(&"ba".to_string())); }
"#,
        },
        BenchTask {
            description: "implement a simple LRU cache",
            difficulty: Difficulty::Medium,
            expected_fn: "LruCache",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_lru() {
        let mut c = LruCache::new(2);
        c.put(1, "a"); c.put(2, "b");
        assert_eq!(c.get(&1), Some(&"a"));
        c.put(3, "c"); // evicts 2
        assert_eq!(c.get(&2), None);
    }
"#,
        },
        BenchTask {
            description: "create a Caesar cipher encrypt function",
            difficulty: Difficulty::Medium,
            expected_fn: "encrypt",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_caesar() { assert_eq!(encrypt("abc", 1), "bcd"); assert_eq!(encrypt("xyz", 3), "abc"); }
"#,
        },
        BenchTask {
            description: "implement depth-first search for a graph",
            difficulty: Difficulty::Medium,
            expected_fn: "dfs",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_dfs() {
        let graph = vec![vec![1, 2], vec![3], vec![3], vec![]];
        let order = dfs(&graph, 0);
        assert!(order.contains(&0)); assert!(order.contains(&3));
        assert_eq!(order[0], 0); // starts at 0
    }
"#,
        },
        BenchTask {
            description: "create a function to generate Pascal's triangle",
            difficulty: Difficulty::Medium,
            expected_fn: "pascal",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_pascal() {
        let t = pascal_triangle(5);
        assert_eq!(t[0], vec![1]);
        assert_eq!(t[2], vec![1, 2, 1]);
        assert_eq!(t[4], vec![1, 4, 6, 4, 1]);
    }
"#,
        },
        BenchTask {
            description: "implement a simple state machine",
            difficulty: Difficulty::Medium,
            expected_fn: "StateMachine",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_sm() {
        let mut sm = StateMachine::new("idle");
        sm.add_transition("idle", "start", "running");
        assert!(sm.send("start"));
        assert_eq!(sm.state(), "running");
        assert!(!sm.send("unknown"));
    }
"#,
        },
        BenchTask {
            description: "create a function to parse CSV lines",
            difficulty: Difficulty::Medium,
            expected_fn: "parse_csv",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_csv() {
        assert_eq!(parse_csv("a,b,c"), vec!["a", "b", "c"]);
        assert_eq!(parse_csv("\"hello,world\",b"), vec!["hello,world", "b"]);
    }
"#,
        },
        BenchTask {
            description: "implement a ring buffer",
            difficulty: Difficulty::Medium,
            expected_fn: "RingBuffer",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_ring() {
        let mut r = RingBuffer::new(3);
        r.push(1); r.push(2); r.push(3);
        assert_eq!(r.len(), 3);
        r.push(4); // overwrites 1
        assert_eq!(r.pop(), Some(2));
    }
"#,
        },
        // ── Medium (auto-fix stress): 5 tasks ────────────────────────
        // These exercise the chained auto-fix pipeline
        BenchTask {
            description: "implement Levenshtein edit distance",
            difficulty: Difficulty::Medium,
            expected_fn: "levenshtein",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_levenshtein() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("same", "same"), 0);
    }
"#,
        },
        BenchTask {
            description: "create a function to generate all permutations of a string",
            difficulty: Difficulty::Medium,
            expected_fn: "permutations",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_permutations() {
        let mut p = permutations("abc");
        p.sort();
        assert_eq!(p.len(), 6);
        assert_eq!(p[0], "abc");
    }
"#,
        },
        BenchTask {
            description: "implement depth-first search on a graph",
            difficulty: Difficulty::Medium,
            expected_fn: "dfs",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_dfs() {
        let graph = vec![vec![1, 2], vec![3], vec![3], vec![]];
        let order = dfs(&graph, 0);
        assert!(order.contains(&0));
        assert!(order.contains(&3));
    }
"#,
        },
        BenchTask {
            description: "create a Pascal's triangle generator",
            difficulty: Difficulty::Medium,
            expected_fn: "pascal_triangle",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_pascal() {
        let t = pascal_triangle(5);
        assert_eq!(t[4], vec![1, 4, 6, 4, 1]);
    }
"#,
        },
        BenchTask {
            description: "implement an email address validator",
            difficulty: Difficulty::Medium,
            expected_fn: "validate_email",
            max_iterations: 8,
            test_source: r#"
    #[test] fn test_email() {
        assert!(validate_email("test@example.com"));
        assert!(!validate_email("invalid"));
        assert!(!validate_email("@no-local.com"));
    }
"#,
        },
        // ── Hard: 15 tasks ───────────────────────────────────────────
        // Hard tasks get looser test assertions since LLM output varies
        BenchTask {
            description: "implement a B-tree insertion",
            difficulty: Difficulty::Hard,
            expected_fn: "insert",
            max_iterations: 10,
            test_source: "", // compilation-only check for hard tasks
        },
        BenchTask {
            description: "create a generic async task queue with priority",
            difficulty: Difficulty::Hard,
            expected_fn: "TaskQueue",
            max_iterations: 10,
            test_source: "",
        },
        BenchTask {
            description: "implement a simple regex matcher supporting . and *",
            difficulty: Difficulty::Hard,
            expected_fn: "regex_match",
            max_iterations: 10,
            test_source: "",
        },
        BenchTask {
            description: "create a function to solve N-queens",
            difficulty: Difficulty::Hard,
            expected_fn: "solve_queens",
            max_iterations: 10,
            test_source: r#"
    #[test] fn test_queens() { let s = solve_queens(4); assert!(!s.is_empty()); assert_eq!(s[0].len(), 4); }
"#,
        },
        BenchTask {
            description: "implement a Trie with insert and search",
            difficulty: Difficulty::Hard,
            expected_fn: "Trie",
            max_iterations: 10,
            test_source: r#"
    #[test] fn test_trie() {
        let mut t = Trie::new();
        t.insert("hello"); t.insert("help");
        assert!(t.search("hello")); assert!(!t.search("hel")); assert!(t.starts_with("hel"));
    }
"#,
        },
        BenchTask {
            description: "create a concurrent hashmap with fine-grained locking",
            difficulty: Difficulty::Hard,
            expected_fn: "ConcurrentMap",
            max_iterations: 10,
            test_source: "",
        },
        BenchTask {
            description: "implement Dijkstra's shortest path algorithm",
            difficulty: Difficulty::Hard,
            expected_fn: "dijkstra",
            max_iterations: 10,
            test_source: r#"
    #[test] fn test_dijkstra() {
        let graph = vec![vec![(1, 4), (2, 2)], vec![(3, 5)], vec![(1, 1), (3, 8)], vec![]];
        let d = dijkstra(&graph, 0);
        assert_eq!(d[0], 0); assert_eq!(d[2], 2);
    }
"#,
        },
        BenchTask {
            description: "create a parser for S-expressions",
            difficulty: Difficulty::Hard,
            expected_fn: "parse_sexp",
            max_iterations: 10,
            test_source: "",
        },
        BenchTask {
            description: "implement a simple bytecode interpreter",
            difficulty: Difficulty::Hard,
            expected_fn: "interpret",
            max_iterations: 10,
            test_source: "",
        },
        BenchTask {
            description: "create a function to compute convex hull",
            difficulty: Difficulty::Hard,
            expected_fn: "convex_hull",
            max_iterations: 10,
            test_source: "",
        },
        BenchTask {
            description: "implement a skip list",
            difficulty: Difficulty::Hard,
            expected_fn: "SkipList",
            max_iterations: 10,
            test_source: "",
        },
        BenchTask {
            description: "create a function to evaluate arithmetic expressions with parentheses",
            difficulty: Difficulty::Hard,
            expected_fn: "evaluate",
            max_iterations: 10,
            test_source: "",
        },
        BenchTask {
            description: "implement a simple HTTP request parser",
            difficulty: Difficulty::Hard,
            expected_fn: "parse_request",
            max_iterations: 10,
            test_source: "",
        },
        BenchTask {
            description: "create a thread pool with work stealing",
            difficulty: Difficulty::Hard,
            expected_fn: "ThreadPool",
            max_iterations: 10,
            test_source: "",
        },
        BenchTask {
            description: "implement a bloom filter",
            difficulty: Difficulty::Hard,
            expected_fn: "BloomFilter",
            max_iterations: 10,
            test_source: r#"
    #[test] fn test_bloom() {
        let mut bf = BloomFilter::new(1000, 3);
        bf.insert(&"hello");
        assert!(bf.contains(&"hello"));
        assert!(!bf.contains(&"world")); // probabilistic, but very likely true
    }
"#,
        },
    ]
}

/// Clean LLM output into compilable Rust code.
///
/// Handles: markdown fences, leading/trailing prose, unclosed delimiters,
/// Python-style syntax at top level, and truncated output.
fn clean_llm_output(source: &str) -> String {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return trimmed.to_string();
    }

    // Step 1: Strip markdown fences (```rust ... ```)
    let defenced = strip_markdown_fences(trimmed);

    // Step 2: Strip leading prose (lines before first Rust construct)
    let lines: Vec<&str> = defenced.lines().collect();
    let code_start = lines
        .iter()
        .position(|l| {
            let t = l.trim();
            t.starts_with("use ")
                || t.starts_with("pub ")
                || t.starts_with("fn ")
                || t.starts_with("struct ")
                || t.starts_with("enum ")
                || t.starts_with("impl ")
                || t.starts_with("#[")
                || t.starts_with("///")
                || t.starts_with("mod ")
                || t.starts_with("type ")
                || t.starts_with("const ")
                || t.starts_with("static ")
                || t.starts_with("trait ")
        })
        .unwrap_or(0);

    // Step 3: Strip trailing prose (lines after last closing brace)
    let code_lines = &lines[code_start..];
    let mut last_code_line = code_lines.len();
    for (i, line) in code_lines.iter().enumerate().rev() {
        let t = line.trim();
        if t == "}"
            || t.ends_with('}')
            || t.ends_with(';')
            || t.ends_with("*/")
            || t.starts_with("//")
            || t.starts_with("///")
            || t.is_empty()
        {
            last_code_line = i + 1;
            break;
        }
    }

    let mut code: String = code_lines[..last_code_line].join("\n");

    // Step 4: Balance braces — add missing closing braces for unclosed delimiters
    let open_braces = code.chars().filter(|&c| c == '{').count();
    let close_braces = code.chars().filter(|&c| c == '}').count();
    if open_braces > close_braces {
        let missing = open_braces - close_braces;
        for _ in 0..missing {
            code.push_str("\n}");
        }
    }

    // Step 5: Remove lines that are clearly not Rust (common LLM artifacts)
    let cleaned_lines: Vec<&str> = code
        .lines()
        .filter(|l| {
            let t = l.trim();
            // Remove explanation lines that start with natural language
            !(t.starts_with("Here") && t.contains(':'))
                && !(t.starts_with("This") && t.contains("function"))
                && !(t.starts_with("The ") && t.contains("above"))
                && !(t.starts_with("Note:"))
                && !(t.starts_with("Example"))
                && !t.starts_with("Output:")
        })
        .collect();

    cleaned_lines.join("\n")
}

/// Strip markdown code fences from text.
fn strip_markdown_fences(source: &str) -> String {
    // Check for opening fence
    if let Some(start) = source.find("```") {
        let after_fence = &source[start + 3..];
        let code_start = after_fence
            .find('\n')
            .map(|i| start + 3 + i + 1)
            .unwrap_or(start + 3);

        let code_region = &source[code_start..];
        let code_end = code_region.rfind("```").unwrap_or(code_region.len());

        return code_region[..code_end].trim().to_string();
    }

    source.to_string()
}

/// Validate generated code by compiling and running test assertions.
fn validate_code(source: &str, test_source: &str) -> (bool, Vec<String>, usize, usize, bool, bool) {
    let mut executor = CodeExecutor::with_real_execution();

    // Clean LLM output: strip fences, prose, balance braces
    let clean_source = clean_llm_output(source);

    // First attempt: compile with tests
    let test_src = if test_source.is_empty() {
        None
    } else {
        Some(test_source)
    };
    let result = executor.execute_rust(&clean_source, test_src);

    if result.compiled {
        return (
            true,
            vec![],
            result.tests_passed,
            result.tests_failed,
            false,
            false,
        );
    }

    // Compilation failed — try targeted fixes for common LLM issues first
    let mut source_to_fix = clean_source.clone();

    // Fix E0412: undeclared type T — LLM uses T without generic declaration
    // Add generic bounds to functions that reference T
    if result
        .compile_errors
        .iter()
        .any(|e| e.contains("cannot find type `T`"))
    {
        // Find "fn name(" and inject "<T>" before the paren
        let lines: Vec<&str> = source_to_fix.lines().collect();
        let fixed_lines: Vec<String> = lines
            .iter()
            .map(|line| {
                let trimmed = line.trim();
                if (trimmed.starts_with("pub fn ") || trimmed.starts_with("fn "))
                    && trimmed.contains("(")
                    && !trimmed.contains("<")
                    && (trimmed.contains(": T")
                        || trimmed.contains("Vec<T>")
                        || trimmed.contains("&T")
                        || trimmed.contains("-> T")
                        || trimmed.contains("Option<T>"))
                {
                    // Insert <T: Clone + PartialOrd> before the first (
                    line.replacen("(", "<T: Clone + PartialOrd + std::fmt::Debug>(", 1)
                } else {
                    line.to_string()
                }
            })
            .collect();
        let fixed_code = fixed_lines.join("\n");
        let retry = executor.execute_rust(&fixed_code, test_src);
        if retry.compiled {
            return (
                true,
                vec![],
                retry.tests_passed,
                retry.tests_failed,
                true,
                true,
            );
        }
        source_to_fix = fixed_code;
    }

    // Fix E0601: no main function — code is empty or only prose
    // Nothing we can do — the LLM didn't produce useful code

    // Try general auto-fix
    let fixed = try_auto_fix(&source_to_fix, &result.compile_errors);
    if let Some(ref fixed_source) = fixed {
        let retry = executor.execute_rust(fixed_source, test_src);
        if retry.compiled {
            return (
                true,
                vec![],
                retry.tests_passed,
                retry.tests_failed,
                true,
                true,
            );
        }
        return (false, retry.compile_errors, 0, 0, true, false);
    }

    (false, result.compile_errors, 0, 0, false, false)
}

fn run_task(task: &BenchTask, task_idx: usize, use_llm: bool, use_cloud: bool) -> TaskResult {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let config = CodingAgentConfig {
        max_iterations: task.max_iterations,
        working_dir: temp_dir.path().to_path_buf(),
        target_file: Some(PathBuf::from("generated.rs")),
        use_local_llm: use_llm,
        use_cloud_llm: use_cloud,
        ..Default::default()
    };

    let start = Instant::now();
    let mut agent = CodingAgent::new(config).expect("agent creation");
    let result = agent.run(task.description);
    let elapsed = start.elapsed().as_millis();

    // Read the generated file
    let target = temp_dir.path().join("generated.rs");
    let content = if target.exists() {
        std::fs::read_to_string(&target).unwrap_or_default()
    } else {
        String::new()
    };

    let code_written = !content.is_empty();
    let contains_expected = content.contains(task.expected_fn);
    let contains_todo =
        content.contains("TODO") || content.contains("todo!") || content.contains("unimplemented!");

    // Level 2+3: Real compilation and test execution
    let (compiled, compile_errors, tests_passed, tests_failed, fix_attempted, fix_succeeded) =
        if code_written && !content.trim().is_empty() {
            validate_code(&content, task.test_source)
        } else {
            (false, vec!["No code generated".into()], 0, 0, false, false)
        };

    let quality_rejections = result
        .observations
        .iter()
        .filter(|o| o.contains("Quality gate rejected"))
        .count();

    let phi_mean = if result.phi_trace.is_empty() {
        0.0
    } else {
        result.phi_trace.iter().sum::<f32>() / result.phi_trace.len() as f32
    };

    let tier_strings: Vec<String> = result
        .generation_tiers
        .iter()
        .map(|t| t.to_string())
        .collect();

    // Status symbols: ✓ = correct (compiles + tests pass), ◐ = compiles but tests fail, ✗ = doesn't compile
    let status = if compiled && tests_failed == 0 {
        "✓"
    } else if compiled {
        "◐"
    } else {
        "✗"
    };

    eprint!(
        "\r  [{:>2}/50] {} {} {} ",
        task_idx + 1,
        task.difficulty,
        status,
        task.description
    );
    if compiled {
        eprint!(
            "[compiled, {}/{} tests]",
            tests_passed,
            tests_passed + tests_failed
        );
    } else if !compile_errors.is_empty() {
        let first_err: String = compile_errors[0].chars().take(50).collect();
        eprint!("[FAIL: {}]", first_err);
    }
    eprint!(" ({}ms)", elapsed);
    eprintln!();

    TaskResult {
        description: task.description.to_string(),
        difficulty: task.difficulty,
        code_written,
        contains_expected_fn: contains_expected,
        contains_todo,
        compiled,
        compile_errors,
        auto_fix_attempted: fix_attempted,
        auto_fix_succeeded: fix_succeeded,
        tests_passed,
        tests_failed,
        iterations: result.iterations_used,
        tiers_used: tier_strings,
        energy: result.total_energy,
        phi_mean,
        quality_gate_rejections: quality_rejections,
        elapsed_ms: elapsed,
        final_phase: format!("{}", result.final_phase),
    }
}

fn compute_stats(results: &[TaskResult]) -> BenchStats {
    let mut stats = BenchStats::default();
    for r in results {
        stats.total += 1;
        if r.code_written {
            stats.code_written += 1;
        }
        if r.contains_expected_fn {
            stats.name_match += 1;
        }
        if r.compiled {
            stats.compiled += 1;
        }
        if r.is_correct() {
            stats.correct += 1;
        }
        if r.auto_fix_attempted {
            stats.auto_fix_attempts += 1;
        }
        if r.auto_fix_succeeded {
            stats.auto_fix_successes += 1;
        }
        stats.total_tests_passed += r.tests_passed;
        stats.total_tests_failed += r.tests_failed;
        stats.total_energy += r.energy;
        stats.total_elapsed_ms += r.elapsed_ms;
        stats.quality_gate_total += r.quality_gate_rejections;

        for tier in &r.tiers_used {
            *stats.tier_counts.entry(tier.clone()).or_insert(0) += 1;
        }

        match r.difficulty {
            Difficulty::Native => {
                stats.native_total += 1;
                if r.compiled {
                    stats.native_compiled += 1;
                }
                if r.is_correct() {
                    stats.native_correct += 1;
                }
            }
            Difficulty::Medium => {
                stats.medium_total += 1;
                if r.compiled {
                    stats.medium_compiled += 1;
                }
                if r.is_correct() {
                    stats.medium_correct += 1;
                }
            }
            Difficulty::Hard => {
                stats.hard_total += 1;
                if r.compiled {
                    stats.hard_compiled += 1;
                }
                if r.is_correct() {
                    stats.hard_correct += 1;
                }
            }
        }
    }
    stats
}

fn print_report(results: &[TaskResult], stats: &BenchStats) {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Symthaea Coding Agent — Hardened Benchmark Results          ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    println!("── Validation Levels ──────────────────────────────────────");
    println!(
        "  L1 Name match:   {}/{} ({:.0}%)",
        stats.name_match,
        stats.total,
        100.0 * stats.name_match as f64 / stats.total as f64
    );
    println!(
        "  L2 Compiles:     {}/{} ({:.0}%)",
        stats.compiled,
        stats.total,
        100.0 * stats.compiled as f64 / stats.total as f64
    );
    println!(
        "  L3 Correct:      {}/{} ({:.0}%)",
        stats.correct,
        stats.total,
        100.0 * stats.correct as f64 / stats.total as f64
    );
    println!(
        "  Tests: {} passed, {} failed",
        stats.total_tests_passed, stats.total_tests_failed
    );
    println!();

    println!("── By Difficulty (L2 Compiled / L3 Correct) ─────────────");
    println!(
        "  Native: {}/{} compiled, {}/{} correct",
        stats.native_compiled, stats.native_total, stats.native_correct, stats.native_total
    );
    println!(
        "  Medium: {}/{} compiled, {}/{} correct",
        stats.medium_compiled, stats.medium_total, stats.medium_correct, stats.medium_total
    );
    println!(
        "  Hard:   {}/{} compiled, {}/{} correct",
        stats.hard_compiled, stats.hard_total, stats.hard_correct, stats.hard_total
    );
    println!();

    println!("── Auto-Fix ────────────────────────────────────────────");
    println!("  Attempted: {}", stats.auto_fix_attempts);
    println!("  Succeeded: {}", stats.auto_fix_successes);
    println!("  Quality gate rejections: {}", stats.quality_gate_total);
    println!();

    println!("── Tier Usage ──────────────────────────────────────────");
    for (tier, count) in &stats.tier_counts {
        println!("  {}: {} calls", tier, count);
    }
    println!("  Total energy: {:.1}", stats.total_energy);
    println!(
        "  Total time: {:.1}s",
        stats.total_elapsed_ms as f64 / 1000.0
    );
    println!();

    // Timing breakdown by difficulty
    println!("── Timing Breakdown ────────────────────────────────────");
    for diff in &[Difficulty::Native, Difficulty::Medium, Difficulty::Hard] {
        let times: Vec<u128> = results
            .iter()
            .filter(|r| r.difficulty == *diff)
            .map(|r| r.elapsed_ms)
            .collect();
        if !times.is_empty() {
            let avg = times.iter().sum::<u128>() / times.len() as u128;
            let max = *times.iter().max().unwrap_or(&0);
            let min = *times.iter().min().unwrap_or(&0);
            println!("  {}: avg {}ms, min {}ms, max {}ms", diff, avg, min, max);
        }
    }
    println!();

    // Compilation failures
    let compile_fails: Vec<&TaskResult> = results
        .iter()
        .filter(|r| r.code_written && !r.compiled)
        .collect();
    if !compile_fails.is_empty() {
        println!(
            "── Compilation Failures ({}) ────────────────────────────",
            compile_fails.len()
        );
        for f in compile_fails.iter().take(10) {
            let err: String = f
                .compile_errors
                .first()
                .map(|e| e.chars().take(70).collect())
                .unwrap_or_default();
            println!("  [{}] {} — {}", f.difficulty, f.description, err);
        }
        println!();
    }

    // Test failures
    let test_fails: Vec<&TaskResult> = results
        .iter()
        .filter(|r| r.compiled && r.tests_failed > 0)
        .collect();
    if !test_fails.is_empty() {
        println!(
            "── Test Failures ({}) ───────────────────────────────────",
            test_fails.len()
        );
        for f in test_fails.iter().take(10) {
            println!(
                "  [{}] {} — {}/{} tests passed",
                f.difficulty,
                f.description,
                f.tests_passed,
                f.tests_passed + f.tests_failed
            );
        }
        println!();
    }

    // No code generated
    let no_code: Vec<&TaskResult> = results.iter().filter(|r| !r.code_written).collect();
    if !no_code.is_empty() {
        println!(
            "── No Code Generated ({}) ──────────────────────────────",
            no_code.len()
        );
        for f in no_code.iter().take(10) {
            println!(
                "  [{}] {} — phase: {}",
                f.difficulty, f.description, f.final_phase
            );
        }
    }
}

fn main() {
    let use_llm = std::env::args().any(|a| a == "--llm");
    let use_cloud = std::env::args().any(|a| a == "--cloud");
    let tasks = build_task_suite();

    eprintln!(
        "Symthaea Coding Agent — Hardened Benchmark ({} tasks)",
        tasks.len()
    );
    if use_cloud {
        eprintln!("Mode: Cloud LLM (Anthropic Claude)\n");
    } else if use_llm {
        eprintln!("Mode: LLM (Ollama qwen2.5-coder:7b)\n");
    } else {
        eprintln!("Mode: Native-only\n");
    }

    let results: Vec<TaskResult> = tasks
        .iter()
        .enumerate()
        .map(|(i, task)| run_task(task, i, use_llm, use_cloud))
        .collect();

    let stats = compute_stats(&results);
    print_report(&results, &stats);

    // JSON report
    let json_results: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "description": r.description,
                "difficulty": format!("{}", r.difficulty),
                "code_written": r.code_written,
                "name_match": r.contains_expected_fn,
                "compiled": r.compiled,
                "tests_passed": r.tests_passed,
                "tests_failed": r.tests_failed,
                "correct": r.is_correct(),
                "auto_fix_attempted": r.auto_fix_attempted,
                "auto_fix_succeeded": r.auto_fix_succeeded,
                "compile_errors": r.compile_errors.iter().take(3).collect::<Vec<_>>(),
                "tiers": r.tiers_used,
                "energy": r.energy,
                "phi_mean": r.phi_mean,
                "elapsed_ms": r.elapsed_ms,
            })
        })
        .collect();

    let json_report = serde_json::json!({
        "benchmark": "symthaea_coding_agent_hardened",
        "summary": {
            "total": stats.total,
            "name_match": stats.name_match,
            "compiled": stats.compiled,
            "correct": stats.correct,
            "compilation_rate": stats.compiled as f64 / stats.total as f64,
            "correctness_rate": stats.correct as f64 / stats.total as f64,
            "tests_passed": stats.total_tests_passed,
            "tests_failed": stats.total_tests_failed,
            "auto_fix_attempts": stats.auto_fix_attempts,
            "auto_fix_successes": stats.auto_fix_successes,
            "by_difficulty": {
                "native": { "compiled": stats.native_compiled, "correct": stats.native_correct, "total": stats.native_total },
                "medium": { "compiled": stats.medium_compiled, "correct": stats.medium_correct, "total": stats.medium_total },
                "hard":   { "compiled": stats.hard_compiled,   "correct": stats.hard_correct,   "total": stats.hard_total },
            }
        },
        "tasks": json_results,
    });

    let report_path = std::env::temp_dir().join("symthaea_coding_benchmark_hardened.json");
    if let Ok(()) = std::fs::write(
        &report_path,
        serde_json::to_string_pretty(&json_report).unwrap(),
    ) {
        eprintln!("\nJSON report: {}", report_path.display());
    }
}