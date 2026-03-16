//! Coding Agent Benchmark Suite
//!
//! Measures end-to-end performance of the coding agent across 50 tasks
//! of varying difficulty. Reports success rates, tier usage, auto-fix
//! effectiveness, energy consumption, and quality gate behavior.
//!
//! Run: `cargo run --example coding_agent_benchmark --features code_generation`
//!
//! Output: JSON report to stdout + human-readable summary.

use std::path::PathBuf;
use std::time::Instant;
use symthaea::coding_agent::{AgentResult, CodingAgent, CodingAgentConfig, TaskPhase};
use symthaea::language::intelligent_dispatcher::BackendTier;

/// A benchmark task with expected outcomes.
struct BenchTask {
    /// Task description given to the agent.
    description: &'static str,
    /// Difficulty tier: Native (should succeed without LLM), Medium (may need LLM), Hard.
    difficulty: Difficulty,
    /// Expected function name in output (for validation).
    expected_fn: &'static str,
    /// Maximum acceptable iterations.
    max_iterations: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Difficulty {
    /// Should be solvable by native pattern matching alone.
    Native,
    /// May require LLM assistance.
    Medium,
    /// Likely requires LLM; tests agent's escalation and error recovery.
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

/// Result of a single benchmark task.
#[derive(Debug)]
struct TaskResult {
    description: String,
    difficulty: Difficulty,
    success: bool,
    code_written: bool,
    contains_expected_fn: bool,
    contains_todo: bool,
    iterations: usize,
    tiers_used: Vec<String>,
    energy: f64,
    phi_mean: f32,
    phi_min: f32,
    quality_gate_rejections: usize,
    auto_fix_applied: bool,
    elapsed_ms: u128,
    final_phase: String,
}

/// Aggregate statistics for the benchmark run.
#[derive(Debug, Default)]
struct BenchStats {
    total: usize,
    succeeded: usize,
    code_written: usize,
    todo_free: usize,
    native_tasks: usize,
    native_succeeded: usize,
    medium_tasks: usize,
    medium_succeeded: usize,
    hard_tasks: usize,
    hard_succeeded: usize,
    total_iterations: usize,
    total_energy: f64,
    total_elapsed_ms: u128,
    tier_counts: std::collections::HashMap<String, usize>,
    quality_gate_total: usize,
    auto_fix_count: usize,
    native_escalations: usize,
}

fn build_task_suite() -> Vec<BenchTask> {
    vec![
        // ── Native difficulty: 20 tasks the pattern library should handle ──
        BenchTask { description: "add a fibonacci function", difficulty: Difficulty::Native, expected_fn: "fibonacci", max_iterations: 5 },
        BenchTask { description: "implement factorial", difficulty: Difficulty::Native, expected_fn: "factorial", max_iterations: 5 },
        BenchTask { description: "add gcd function", difficulty: Difficulty::Native, expected_fn: "gcd", max_iterations: 5 },
        BenchTask { description: "check primality", difficulty: Difficulty::Native, expected_fn: "is_prime", max_iterations: 5 },
        BenchTask { description: "compute absolute value", difficulty: Difficulty::Native, expected_fn: "absolute", max_iterations: 5 },
        BenchTask { description: "add a hello function", difficulty: Difficulty::Native, expected_fn: "hello", max_iterations: 5 },
        BenchTask { description: "reverse a string", difficulty: Difficulty::Native, expected_fn: "reverse_string", max_iterations: 5 },
        BenchTask { description: "check if palindrome", difficulty: Difficulty::Native, expected_fn: "is_palindrome", max_iterations: 5 },
        BenchTask { description: "count vowels in a string", difficulty: Difficulty::Native, expected_fn: "count_vowels", max_iterations: 5 },
        BenchTask { description: "convert string to uppercase", difficulty: Difficulty::Native, expected_fn: "to_uppercase", max_iterations: 5 },
        BenchTask { description: "implement bubble sort", difficulty: Difficulty::Native, expected_fn: "bubble_sort", max_iterations: 5 },
        BenchTask { description: "implement insertion sort", difficulty: Difficulty::Native, expected_fn: "insertion_sort", max_iterations: 5 },
        BenchTask { description: "sort a vector", difficulty: Difficulty::Native, expected_fn: "sort_vec", max_iterations: 5 },
        BenchTask { description: "implement binary search", difficulty: Difficulty::Native, expected_fn: "binary_search", max_iterations: 5 },
        BenchTask { description: "find max in a vector", difficulty: Difficulty::Native, expected_fn: "find_max", max_iterations: 5 },
        BenchTask { description: "find min in a vector", difficulty: Difficulty::Native, expected_fn: "find_min", max_iterations: 5 },
        BenchTask { description: "sum all elements in a vec", difficulty: Difficulty::Native, expected_fn: "sum_vec", max_iterations: 5 },
        BenchTask { description: "check if number is even", difficulty: Difficulty::Native, expected_fn: "is_even", max_iterations: 5 },
        BenchTask { description: "create a stack data structure", difficulty: Difficulty::Native, expected_fn: "Stack", max_iterations: 5 },
        BenchTask { description: "flatten nested vectors", difficulty: Difficulty::Native, expected_fn: "flatten", max_iterations: 5 },

        // ── Medium difficulty: 15 tasks that may need LLM ──────────────
        BenchTask { description: "implement a linked list", difficulty: Difficulty::Medium, expected_fn: "LinkedList", max_iterations: 8 },
        BenchTask { description: "create a matrix multiply function", difficulty: Difficulty::Medium, expected_fn: "multiply", max_iterations: 8 },
        BenchTask { description: "implement merge sort", difficulty: Difficulty::Medium, expected_fn: "merge_sort", max_iterations: 8 },
        BenchTask { description: "add a function to compute Levenshtein distance", difficulty: Difficulty::Medium, expected_fn: "levenshtein", max_iterations: 8 },
        BenchTask { description: "implement a simple tokenizer for arithmetic expressions", difficulty: Difficulty::Medium, expected_fn: "tokenize", max_iterations: 8 },
        BenchTask { description: "create a function to validate email addresses", difficulty: Difficulty::Medium, expected_fn: "validate_email", max_iterations: 8 },
        BenchTask { description: "implement run-length encoding", difficulty: Difficulty::Medium, expected_fn: "rle_encode", max_iterations: 8 },
        BenchTask { description: "add a function to find all permutations of a string", difficulty: Difficulty::Medium, expected_fn: "permutations", max_iterations: 8 },
        BenchTask { description: "implement a simple LRU cache", difficulty: Difficulty::Medium, expected_fn: "LruCache", max_iterations: 8 },
        BenchTask { description: "create a Caesar cipher encrypt function", difficulty: Difficulty::Medium, expected_fn: "encrypt", max_iterations: 8 },
        BenchTask { description: "implement depth-first search for a graph", difficulty: Difficulty::Medium, expected_fn: "dfs", max_iterations: 8 },
        BenchTask { description: "create a function to generate Pascal's triangle", difficulty: Difficulty::Medium, expected_fn: "pascal", max_iterations: 8 },
        BenchTask { description: "implement a simple state machine", difficulty: Difficulty::Medium, expected_fn: "StateMachine", max_iterations: 8 },
        BenchTask { description: "create a function to parse CSV lines", difficulty: Difficulty::Medium, expected_fn: "parse_csv", max_iterations: 8 },
        BenchTask { description: "implement a ring buffer", difficulty: Difficulty::Medium, expected_fn: "RingBuffer", max_iterations: 8 },

        // ── Hard difficulty: 15 tasks testing escalation and recovery ───
        BenchTask { description: "implement a B-tree insertion", difficulty: Difficulty::Hard, expected_fn: "insert", max_iterations: 10 },
        BenchTask { description: "create a generic async task queue with priority", difficulty: Difficulty::Hard, expected_fn: "TaskQueue", max_iterations: 10 },
        BenchTask { description: "implement a simple regex matcher supporting . and *", difficulty: Difficulty::Hard, expected_fn: "regex_match", max_iterations: 10 },
        BenchTask { description: "create a function to solve N-queens", difficulty: Difficulty::Hard, expected_fn: "solve_queens", max_iterations: 10 },
        BenchTask { description: "implement a Trie with insert and search", difficulty: Difficulty::Hard, expected_fn: "Trie", max_iterations: 10 },
        BenchTask { description: "create a concurrent hashmap with fine-grained locking", difficulty: Difficulty::Hard, expected_fn: "ConcurrentMap", max_iterations: 10 },
        BenchTask { description: "implement Dijkstra's shortest path algorithm", difficulty: Difficulty::Hard, expected_fn: "dijkstra", max_iterations: 10 },
        BenchTask { description: "create a parser for S-expressions", difficulty: Difficulty::Hard, expected_fn: "parse_sexp", max_iterations: 10 },
        BenchTask { description: "implement a simple bytecode interpreter", difficulty: Difficulty::Hard, expected_fn: "interpret", max_iterations: 10 },
        BenchTask { description: "create a function to compute convex hull", difficulty: Difficulty::Hard, expected_fn: "convex_hull", max_iterations: 10 },
        BenchTask { description: "implement a skip list", difficulty: Difficulty::Hard, expected_fn: "SkipList", max_iterations: 10 },
        BenchTask { description: "create a function to evaluate arithmetic expressions with parentheses", difficulty: Difficulty::Hard, expected_fn: "evaluate", max_iterations: 10 },
        BenchTask { description: "implement a simple HTTP request parser", difficulty: Difficulty::Hard, expected_fn: "parse_request", max_iterations: 10 },
        BenchTask { description: "create a thread pool with work stealing", difficulty: Difficulty::Hard, expected_fn: "ThreadPool", max_iterations: 10 },
        BenchTask { description: "implement a bloom filter", difficulty: Difficulty::Hard, expected_fn: "BloomFilter", max_iterations: 10 },
    ]
}

fn run_task(task: &BenchTask, task_idx: usize, use_llm: bool) -> TaskResult {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let config = CodingAgentConfig {
        max_iterations: task.max_iterations,
        working_dir: temp_dir.path().to_path_buf(),
        target_file: Some(PathBuf::from("generated.rs")),
        use_local_llm: use_llm,
        ..Default::default()
    };

    let start = Instant::now();
    let mut agent = CodingAgent::new(config).expect("agent creation");
    let result = agent.run(task.description);
    let elapsed = start.elapsed().as_millis();

    // Read the generated file to check content
    let target = temp_dir.path().join("generated.rs");
    let (code_written, contains_expected, contains_todo) = if target.exists() {
        let content = std::fs::read_to_string(&target).unwrap_or_default();
        (
            !content.is_empty(),
            content.contains(task.expected_fn),
            content.contains("TODO") || content.contains("todo!") || content.contains("unimplemented!"),
        )
    } else {
        (false, false, false)
    };

    // Count quality gate rejections from observations
    let quality_rejections = result.observations.iter()
        .filter(|o| o.contains("Quality gate rejected"))
        .count();

    let auto_fix = result.observations.iter()
        .any(|o| o.contains("auto-fix applied"));

    // Determine success: code was written, contains expected fn, no TODOs
    let success = code_written && contains_expected && !contains_todo;

    let phi_mean = if result.phi_trace.is_empty() {
        0.0
    } else {
        result.phi_trace.iter().sum::<f32>() / result.phi_trace.len() as f32
    };
    let phi_min = result.phi_trace.iter().copied().fold(f32::MAX, f32::min);

    let tier_strings: Vec<String> = result.generation_tiers.iter().map(|t| t.to_string()).collect();

    eprint!("\r  [{:>2}/50] {} — {} {} ({} ms)",
        task_idx + 1,
        task.difficulty,
        if success { "✓" } else { "✗" },
        task.description,
        elapsed
    );
    eprintln!();

    TaskResult {
        description: task.description.to_string(),
        difficulty: task.difficulty,
        success,
        code_written,
        contains_expected_fn: contains_expected,
        contains_todo,
        iterations: result.iterations_used,
        tiers_used: tier_strings,
        energy: result.total_energy,
        phi_mean,
        phi_min,
        quality_gate_rejections: quality_rejections,
        auto_fix_applied: auto_fix,
        elapsed_ms: elapsed,
        final_phase: format!("{}", result.final_phase),
    }
}

fn compute_stats(results: &[TaskResult]) -> BenchStats {
    let mut stats = BenchStats::default();

    for r in results {
        stats.total += 1;
        if r.success { stats.succeeded += 1; }
        if r.code_written { stats.code_written += 1; }
        if r.code_written && !r.contains_todo { stats.todo_free += 1; }

        match r.difficulty {
            Difficulty::Native => {
                stats.native_tasks += 1;
                if r.success { stats.native_succeeded += 1; }
            }
            Difficulty::Medium => {
                stats.medium_tasks += 1;
                if r.success { stats.medium_succeeded += 1; }
            }
            Difficulty::Hard => {
                stats.hard_tasks += 1;
                if r.success { stats.hard_succeeded += 1; }
            }
        }

        stats.total_iterations += r.iterations;
        stats.total_energy += r.energy;
        stats.total_elapsed_ms += r.elapsed_ms;
        stats.quality_gate_total += r.quality_gate_rejections;
        if r.auto_fix_applied { stats.auto_fix_count += 1; }

        for tier in &r.tiers_used {
            *stats.tier_counts.entry(tier.clone()).or_insert(0) += 1;
        }

        // Count native escalations: native was tried but task needed LLM
        if r.difficulty == Difficulty::Native && r.tiers_used.iter().any(|t| t != "Native") {
            stats.native_escalations += 1;
        }
    }

    stats
}

fn print_report(results: &[TaskResult], stats: &BenchStats) {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║     Symthaea Coding Agent — Benchmark Results           ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // Overall
    println!("── Overall ─────────────────────────────────────────────");
    println!("  Tasks:        {}", stats.total);
    println!("  Succeeded:    {} ({:.0}%)", stats.succeeded, 100.0 * stats.succeeded as f64 / stats.total as f64);
    println!("  Code written: {} ({:.0}%)", stats.code_written, 100.0 * stats.code_written as f64 / stats.total as f64);
    println!("  TODO-free:    {} ({:.0}%)", stats.todo_free, 100.0 * stats.todo_free as f64 / stats.total as f64);
    println!("  Total time:   {:.1}s", stats.total_elapsed_ms as f64 / 1000.0);
    println!("  Avg time:     {:.0}ms/task", stats.total_elapsed_ms as f64 / stats.total as f64);
    println!();

    // By difficulty
    println!("── By Difficulty ───────────────────────────────────────");
    println!("  Native: {}/{} ({:.0}%)", stats.native_succeeded, stats.native_tasks,
        100.0 * stats.native_succeeded as f64 / stats.native_tasks.max(1) as f64);
    println!("  Medium: {}/{} ({:.0}%)", stats.medium_succeeded, stats.medium_tasks,
        100.0 * stats.medium_succeeded as f64 / stats.medium_tasks.max(1) as f64);
    println!("  Hard:   {}/{} ({:.0}%)", stats.hard_succeeded, stats.hard_tasks,
        100.0 * stats.hard_succeeded as f64 / stats.hard_tasks.max(1) as f64);
    println!();

    // Tier usage
    println!("── Tier Usage ──────────────────────────────────────────");
    for (tier, count) in &stats.tier_counts {
        println!("  {}: {} calls", tier, count);
    }
    println!("  Native escalations: {}", stats.native_escalations);
    println!();

    // Consciousness
    let phi_means: Vec<f32> = results.iter().map(|r| r.phi_mean).collect();
    let avg_phi = phi_means.iter().sum::<f32>() / phi_means.len() as f32;
    println!("── Consciousness ───────────────────────────────────────");
    println!("  Avg Phi:      {:.4}", avg_phi);
    println!("  Total energy: {:.1}", stats.total_energy);
    println!("  Avg energy:   {:.2}/task", stats.total_energy / stats.total as f64);
    println!();

    // Auto-fix & quality gate
    println!("── Auto-Fix & Quality Gate ─────────────────────────────");
    println!("  Auto-fix applied: {} tasks", stats.auto_fix_count);
    println!("  Quality gate rejections: {}", stats.quality_gate_total);
    println!("  Avg iterations: {:.1}", stats.total_iterations as f64 / stats.total as f64);
    println!();

    // Failed tasks detail
    let failures: Vec<&TaskResult> = results.iter().filter(|r| !r.success).collect();
    if !failures.is_empty() {
        println!("── Failed Tasks ({}) ───────────────────────────────────", failures.len());
        for f in failures.iter().take(15) {
            println!("  [{}] {} — phase: {}, iters: {}, code: {}, fn: {}, todo: {}",
                f.difficulty, f.description, f.final_phase,
                f.iterations, f.code_written, f.contains_expected_fn, f.contains_todo);
        }
        if failures.len() > 15 {
            println!("  ... and {} more", failures.len() - 15);
        }
    }
}

fn main() {
    let use_llm = std::env::args().any(|a| a == "--llm");
    let tasks = build_task_suite();

    eprintln!("Symthaea Coding Agent Benchmark — {} tasks", tasks.len());
    if use_llm {
        eprintln!("(LLM mode — using Ollama qwen2.5-coder:7b for escalation)\n");
    } else {
        eprintln!("(native-only mode — no LLM backend)\n");
    }

    let results: Vec<TaskResult> = tasks.iter().enumerate()
        .map(|(i, task)| run_task(task, i, use_llm))
        .collect();

    let stats = compute_stats(&results);
    print_report(&results, &stats);

    // JSON output for programmatic consumption
    let json_results: Vec<serde_json::Value> = results.iter().map(|r| {
        serde_json::json!({
            "description": r.description,
            "difficulty": format!("{}", r.difficulty),
            "success": r.success,
            "code_written": r.code_written,
            "expected_fn_found": r.contains_expected_fn,
            "contains_todo": r.contains_todo,
            "iterations": r.iterations,
            "tiers": r.tiers_used,
            "energy": r.energy,
            "phi_mean": r.phi_mean,
            "quality_gate_rejections": r.quality_gate_rejections,
            "auto_fix": r.auto_fix_applied,
            "elapsed_ms": r.elapsed_ms,
            "final_phase": r.final_phase,
        })
    }).collect();

    let json_report = serde_json::json!({
        "benchmark": "symthaea_coding_agent",
        "tasks": json_results,
        "summary": {
            "total": stats.total,
            "succeeded": stats.succeeded,
            "success_rate": stats.succeeded as f64 / stats.total as f64,
            "native_success_rate": stats.native_succeeded as f64 / stats.native_tasks.max(1) as f64,
            "medium_success_rate": stats.medium_succeeded as f64 / stats.medium_tasks.max(1) as f64,
            "hard_success_rate": stats.hard_succeeded as f64 / stats.hard_tasks.max(1) as f64,
            "total_energy": stats.total_energy,
            "total_elapsed_ms": stats.total_elapsed_ms,
            "tier_counts": stats.tier_counts,
            "auto_fix_count": stats.auto_fix_count,
            "quality_gate_rejections": stats.quality_gate_total,
        }
    });

    // Write JSON report to file
    let report_path = std::env::temp_dir().join("symthaea_coding_benchmark.json");
    if let Ok(()) = std::fs::write(&report_path, serde_json::to_string_pretty(&json_report).unwrap()) {
        eprintln!("\nJSON report: {}", report_path.display());
    }
}
