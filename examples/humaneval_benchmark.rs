// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HumanEval Benchmark — 164 Python coding problems
//!
//! Measures pass@1 on OpenAI's HumanEval dataset.
//!
//! Modes:
//!   --direct     Direct LLM call (bypasses cognitive loop — best for benchmarks)
//!   --llm        Full agent pipeline with LLM escalation
//!   (default)    Native-only (no LLM)
//!
//! Run:
//!   Direct:  `cargo run --example humaneval_benchmark --features code_generation -- --direct --limit 20`
//!   Agent:   `cargo run --example humaneval_benchmark --features code_generation -- --llm --limit 20`
//!   Verify:  `... -- --direct --limit 20 --verify-canonical`
//!
//! Requires: HumanEval.jsonl.gz at benchmarks/ai_benchmarks/data/humaneval/human-eval-master/data/

use std::path::PathBuf;
use std::time::Instant;
use symthaea::coding_agent::{CodingAgent, CodingAgentConfig};
use symthaea::language::code_executor::CodeExecutor;
use symthaea::language::llm_backend::{GenerationParams, LLMBackend, OllamaBackend};

/// A HumanEval problem loaded from JSONL.
struct HumanEvalProblem {
    task_id: String,
    prompt: String,
    entry_point: String,
    canonical_solution: String,
    test: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum BenchMode {
    /// Native patterns only (no LLM)
    Native,
    /// Full consciousness-gated agent pipeline + LLM escalation
    Agent,
    /// Direct LLM call — bypasses cognitive loop (best for benchmarks)
    Direct,
}

/// Result of evaluating one problem.
#[derive(Debug)]
struct EvalResult {
    task_id: String,
    entry_point: String,
    passed: bool,
    compiled: bool,
    error: Option<String>,
    mode: String,
    elapsed_ms: u128,
    code_len: usize,
}

fn load_humaneval(path: &std::path::Path) -> Vec<HumanEvalProblem> {
    let output = std::process::Command::new("zcat")
        .arg(path)
        .output()
        .expect("Failed to run zcat — is gzip installed?");

    let content = String::from_utf8_lossy(&output.stdout);
    let mut problems = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line)
            .unwrap_or_else(|e| panic!("Failed to parse JSONL line: {e}"));
        problems.push(HumanEvalProblem {
            task_id: v["task_id"].as_str().unwrap_or("").to_string(),
            prompt: v["prompt"].as_str().unwrap_or("").to_string(),
            entry_point: v["entry_point"].as_str().unwrap_or("").to_string(),
            canonical_solution: v["canonical_solution"].as_str().unwrap_or("").to_string(),
            test: v["test"].as_str().unwrap_or("").to_string(),
        });
    }
    problems
}

// ── Direct LLM Mode ──────────────────────────────────────────────────────

/// Generate a solution by calling Ollama directly — no cognitive loop overhead.
///
/// This is the "fair comparison" mode: same model, same prompt format as
/// standard HumanEval benchmarks. The only Symthaea-specific piece is the
/// output cleaning.
fn generate_direct(problem: &HumanEvalProblem) -> String {
    let backend = OllamaBackend::new();

    let params = GenerationParams {
        temperature: 0.0, // deterministic for reproducibility
        max_tokens: 1024,
        system_prompt: Some(
            "Complete the Python function. Output ONLY the function code (signature + body). \
             No markdown, no explanation, no tests."
                .into(),
        ),
        consciousness_context: None,
    };

    // Send the HumanEval prompt directly — it already has signature + docstring
    let prompt = problem.prompt.clone();

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    match rt.block_on(backend.generate(&prompt, &params)) {
        Ok(output) => clean_python_output(&output),
        Err(e) => {
            eprintln!("    LLM error: {e}");
            format!("{}\n    pass\n", problem.prompt)
        }
    }
}

// ── Agent Mode ───────────────────────────────────────────────────────────

/// Generate a solution using the full coding agent pipeline.
fn generate_agent(problem: &HumanEvalProblem, use_llm: bool) -> String {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let config = CodingAgentConfig {
        max_iterations: 5,
        working_dir: temp_dir.path().to_path_buf(),
        target_file: Some(PathBuf::from("solution.py")),
        use_local_llm: use_llm,
        ..Default::default()
    };

    // Minimal prompt — HumanEval prompt already has everything needed
    let task = format!("{}\n# Complete the function above.\n", problem.prompt);

    let mut agent = CodingAgent::new(config).expect("agent");
    let _result = agent.run(&task);

    let target = temp_dir.path().join("solution.py");
    let code = std::fs::read_to_string(&target).unwrap_or_default();
    clean_python_output(&code)
}

// ── Output Cleaning ──────────────────────────────────────────────────────

/// Clean LLM-generated Python output for HumanEval.
///
/// Handles: markdown fences, bare language tags, leading prose.
/// Preserves: blank lines within functions, indented code, decorators.
fn clean_python_output(source: &str) -> String {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return trimmed.to_string();
    }

    // Step 1: Strip markdown fences (only if explicitly present)
    let defenced = if trimmed.starts_with("```") {
        // Find end of opening fence line
        let after_fence = &trimmed[3..];
        let code_start = after_fence.find('\n').map(|i| 3 + i + 1).unwrap_or(3);
        let code_region = &trimmed[code_start..];
        // Find closing fence
        let code_end = code_region.rfind("```").unwrap_or(code_region.len());
        code_region[..code_end].trim().to_string()
    } else if let Some(fence_pos) = trimmed.find("```python") {
        // Fence appears mid-text (after prose)
        let after_fence = &trimmed[fence_pos + 9..];
        let code_start = after_fence.find('\n').map(|i| i + 1).unwrap_or(0);
        let code_region = &after_fence[code_start..];
        let code_end = code_region.rfind("```").unwrap_or(code_region.len());
        code_region[..code_end].trim().to_string()
    } else {
        trimmed.to_string()
    };

    // Step 2: Find the first line that looks like Python code
    // (skip prose, bare language tags like "python", "Here's the solution:")
    let lines: Vec<&str> = defenced.lines().collect();
    let code_start = lines
        .iter()
        .position(|l| {
            let t = l.trim();
            // Must be a non-empty Python construct (NOT blank lines or bare words)
            t.starts_with("def ")
                || t.starts_with("class ")
                || t.starts_with("import ")
                || t.starts_with("from ")
                || t.starts_with("@")
                || t.starts_with("# ")
                || (t.starts_with("    ") && t.len() > 4) // indented code body
        })
        .unwrap_or(0);

    // Step 3: Take everything from code_start to end (including blank lines in functions)
    let code_lines = &lines[code_start..];

    // Step 4: Trim trailing prose (lines after the last line that looks like code)
    let last_code = code_lines
        .iter()
        .rposition(|l| {
            let t = l.trim();
            t.is_empty()  // blank lines are OK inside functions
            || t.starts_with("def ")
            || t.starts_with("class ")
            || t.starts_with("return ")
            || t.starts_with("    ")
            || t.starts_with("#")
            || t.starts_with("@")
            || t.starts_with("import ")
            || t.starts_with("from ")
            || t.ends_with(':')
            || t.ends_with(')')
            || t.ends_with(']')
        })
        .map(|i| i + 1)
        .unwrap_or(code_lines.len());

    code_lines[..last_code].join("\n")
}

// ── Test Execution ───────────────────────────────────────────────────────

/// Run a HumanEval problem's test suite against generated code.
fn run_test(problem: &HumanEvalProblem, solution: &str) -> (bool, bool, Option<String>) {
    let mut executor = CodeExecutor::with_real_execution();

    // If the solution doesn't contain the entry point, prepend the prompt
    // (which has the function signature) so at least the function exists
    let full_solution = if solution.contains(&problem.entry_point) {
        solution.to_string()
    } else {
        format!("{}\n    pass\n\n{}", problem.prompt, solution)
    };

    // Build test file: solution + HumanEval test harness + check() call
    let test_code = format!(
        "{}\n\n{}\n\ncheck({})\n",
        full_solution, problem.test, problem.entry_point
    );

    let result = executor.execute_python(&test_code);

    if result.compiled && result.runtime_error.is_none() && result.tests_passed > 0 {
        (true, true, None)
    } else if result.compiled {
        let err = result.runtime_error.or_else(|| {
            if !result.test_output.is_empty() {
                // Extract the last line of the traceback (most informative)
                let last_meaningful = result
                    .test_output
                    .lines()
                    .filter(|l| !l.trim().is_empty())
                    .last()
                    .unwrap_or("Test failed");
                Some(last_meaningful.chars().take(200).collect())
            } else {
                Some("Test failed".into())
            }
        });
        (false, true, err)
    } else {
        (
            false,
            false,
            Some(result.compile_errors.first().cloned().unwrap_or_default()),
        )
    }
}

/// Verify canonical solution passes our test harness.
fn verify_canonical(problem: &HumanEvalProblem) -> bool {
    let mut executor = CodeExecutor::with_real_execution();
    let code = format!(
        "{}{}\n\n{}\n\ncheck({})\n",
        problem.prompt, problem.canonical_solution, problem.test, problem.entry_point
    );
    let result = executor.execute_python(&code);
    result.compiled && result.runtime_error.is_none()
}

// ── Main ─────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = if args.iter().any(|a| a == "--direct") {
        BenchMode::Direct
    } else if args.iter().any(|a| a == "--llm") {
        BenchMode::Agent
    } else {
        BenchMode::Native
    };
    let limit = args
        .iter()
        .position(|a| a == "--limit")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(164);
    let verify = args.iter().any(|a| a == "--verify-canonical");

    let data_path = PathBuf::from(
        "benchmarks/ai_benchmarks/data/humaneval/human-eval-master/data/HumanEval.jsonl.gz",
    );
    if !data_path.exists() {
        eprintln!("HumanEval dataset not found at: {}", data_path.display());
        eprintln!("Run: python3 benchmarks/ai_benchmarks/scripts/download_benchmarks.py");
        std::process::exit(1);
    }

    let problems = load_humaneval(&data_path);
    let problems: Vec<&HumanEvalProblem> = problems.iter().take(limit).collect();

    let mode_str = match mode {
        BenchMode::Direct => "Direct LLM (qwen2.5-coder:7b, no cognitive loop)",
        BenchMode::Agent => "Agent + LLM (full consciousness pipeline)",
        BenchMode::Native => "Native-only (no LLM)",
    };
    eprintln!("HumanEval Benchmark — {} problems", problems.len());
    eprintln!("Mode: {}\n", mode_str);

    if verify {
        eprintln!("Verifying canonical solutions...");
        let mut canonical_pass = 0;
        for p in &problems {
            if verify_canonical(p) {
                canonical_pass += 1;
            } else {
                eprintln!("  WARN: canonical fails for {}", p.task_id);
            }
        }
        eprintln!("  Canonical: {}/{} pass\n", canonical_pass, problems.len());
    }

    let mut results = Vec::new();
    for (i, problem) in problems.iter().enumerate() {
        let start = Instant::now();

        let solution = match mode {
            BenchMode::Direct => generate_direct(problem),
            BenchMode::Agent => generate_agent(problem, true),
            BenchMode::Native => generate_agent(problem, false),
        };

        let (passed, compiled, error) = run_test(problem, &solution);
        let elapsed = start.elapsed().as_millis();

        let status = if passed {
            "✓"
        } else if compiled {
            "◐"
        } else {
            "✗"
        };
        eprint!(
            "\r  [{:>3}/{}] {} {} ",
            i + 1,
            problems.len(),
            status,
            problem.task_id
        );
        if let Some(ref e) = error {
            let short: String = e.chars().take(50).collect();
            eprint!("— {}", short);
        }
        eprint!(" ({}ms)", elapsed);
        eprintln!();

        results.push(EvalResult {
            task_id: problem.task_id.clone(),
            entry_point: problem.entry_point.clone(),
            passed,
            compiled,
            error,
            mode: format!("{:?}", mode),
            elapsed_ms: elapsed,
            code_len: solution.len(),
        });
    }

    // Summary
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let compiled = results.iter().filter(|r| r.compiled).count();
    let total_time: u128 = results.iter().map(|r| r.elapsed_ms).sum();

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║     Symthaea × HumanEval — Results                      ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!("── Mode: {} ──", mode_str);
    println!(
        "  Pass@1:   {}/{} ({:.1}%)",
        passed,
        total,
        100.0 * passed as f64 / total as f64
    );
    println!(
        "  Compiled: {}/{} ({:.1}%)",
        compiled,
        total,
        100.0 * compiled as f64 / total as f64
    );
    println!(
        "  Time:     {:.1}s ({:.0}ms/problem)",
        total_time as f64 / 1000.0,
        total_time as f64 / total as f64
    );
    println!();

    let failures: Vec<&EvalResult> = results.iter().filter(|r| !r.passed).collect();
    if !failures.is_empty() && failures.len() <= 30 {
        println!(
            "── Failures ({}) ───────────────────────────────────────",
            failures.len()
        );
        for f in failures.iter().take(20) {
            let err: String = f
                .error
                .as_ref()
                .map(|e| e.chars().take(60).collect())
                .unwrap_or_default();
            println!("  {} ({}) — {}", f.task_id, f.entry_point, err);
        }
    }

    // JSON report
    let json_results: Vec<serde_json::Value> = results
        .iter()
        .map(|r| {
            serde_json::json!({
                "task_id": r.task_id, "entry_point": r.entry_point,
                "passed": r.passed, "compiled": r.compiled,
                "error": r.error, "mode": r.mode,
                "elapsed_ms": r.elapsed_ms, "code_len": r.code_len,
            })
        })
        .collect();

    let json_report = serde_json::json!({
        "benchmark": "humaneval",
        "mode": format!("{:?}", mode),
        "model": "qwen2.5-coder:7b",
        "total": total, "passed": passed,
        "pass_at_1": passed as f64 / total as f64,
        "compiled": compiled,
        "results": json_results,
    });

    let report_path = std::env::temp_dir().join("symthaea_humaneval.json");
    let _ = std::fs::write(
        &report_path,
        serde_json::to_string_pretty(&json_report).unwrap(),
    );
    eprintln!("\nJSON report: {}", report_path.display());
}
