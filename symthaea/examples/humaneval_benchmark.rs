//! HumanEval Benchmark — 164 Python coding problems
//!
//! Measures pass@1 on OpenAI's HumanEval dataset using Symthaea's
//! consciousness-gated coding agent with LLM escalation.
//!
//! Run:
//!   Native:  `cargo run --example humaneval_benchmark --features code_generation`
//!   With LLM: `cargo run --example humaneval_benchmark --features code_generation -- --llm`
//!   Subset:  `... -- --llm --limit 20`
//!
//! Requires: HumanEval.jsonl.gz at benchmarks/ai_benchmarks/data/humaneval/human-eval-master/data/

use std::path::PathBuf;
use std::time::Instant;
use symthaea::coding_agent::{CodingAgent, CodingAgentConfig};
use symthaea::language::code_executor::CodeExecutor;

/// A HumanEval problem loaded from JSONL.
struct HumanEvalProblem {
    task_id: String,
    prompt: String,
    entry_point: String,
    canonical_solution: String,
    test: String,
}

/// Result of evaluating one problem.
#[derive(Debug)]
struct EvalResult {
    task_id: String,
    entry_point: String,
    passed: bool,
    compiled: bool,
    error: Option<String>,
    used_native: bool,
    used_llm: bool,
    elapsed_ms: u128,
    code_len: usize,
}

fn load_humaneval(path: &std::path::Path) -> Vec<HumanEvalProblem> {
    // Use system gzip to decompress — avoids adding flate2 dependency
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

/// Generate a Python solution for a HumanEval problem using the coding agent.
fn generate_solution(problem: &HumanEvalProblem, use_llm: bool) -> (String, bool, bool) {
    let temp_dir = tempfile::tempdir().expect("tempdir");
    let config = CodingAgentConfig {
        max_iterations: 5,
        working_dir: temp_dir.path().to_path_buf(),
        target_file: Some(PathBuf::from("solution.py")),
        use_local_llm: use_llm,
        ..Default::default()
    };

    let task = format!(
        "Complete this Python function. Output ONLY the function body, no explanation:\n\n{}",
        problem.prompt
    );

    let mut agent = CodingAgent::new(config).expect("agent");
    let result = agent.run(&task);

    // Read generated file
    let target = temp_dir.path().join("solution.py");
    let code = std::fs::read_to_string(&target).unwrap_or_default();

    let used_native = result
        .generation_tiers
        .iter()
        .any(|t| t.to_string() == "Native");
    let used_llm = result
        .generation_tiers
        .iter()
        .any(|t| t.to_string() == "LocalLLM");

    // Clean LLM output: strip markdown fences, language tags, prose
    let clean = clean_python_output(&code);

    // If the agent generated code, use it. Otherwise fall back to the prompt + pass.
    let solution = if clean.contains(&problem.entry_point) {
        clean
    } else if clean.contains("def ") {
        // Code has a function but maybe wrong name — try it anyway
        clean
    } else {
        // No useful code generated — use prompt + pass (will fail tests)
        format!("{}\n    pass\n", problem.prompt)
    };

    (solution, used_native, used_llm)
}

/// Clean LLM-generated Python output for HumanEval.
///
/// Strips markdown fences, bare language tags, leading prose, and ensures
/// the output starts with valid Python (import, def, class, etc.).
fn clean_python_output(source: &str) -> String {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return trimmed.to_string();
    }

    // Strip markdown fences
    let mut code = if let Some(start) = trimmed.find("```") {
        let after_fence = &trimmed[start + 3..];
        let code_start = after_fence.find('\n').map(|i| start + 3 + i + 1).unwrap_or(start + 3);
        let code_region = &trimmed[code_start..];
        let code_end = code_region.rfind("```").unwrap_or(code_region.len());
        code_region[..code_end].trim().to_string()
    } else {
        trimmed.to_string()
    };

    // Strip leading lines that aren't Python code
    let lines: Vec<&str> = code.lines().collect();
    let code_start = lines.iter().position(|l| {
        let t = l.trim();
        t.starts_with("def ")
            || t.starts_with("class ")
            || t.starts_with("import ")
            || t.starts_with("from ")
            || t.starts_with("#")
            || t.starts_with("@")
            || t.starts_with("\"\"\"")
            || t.is_empty()
            || t.starts_with("    ") // indented code (function body)
    }).unwrap_or(0);

    if code_start > 0 {
        code = lines[code_start..].join("\n");
    }

    code
}

/// Run a HumanEval problem's test suite against generated code.
fn run_test(problem: &HumanEvalProblem, solution: &str) -> (bool, bool, Option<String>) {
    let mut executor = CodeExecutor::with_real_execution();

    // Build the full test file:
    // 1. Solution code (includes the function)
    // 2. Test harness from HumanEval
    // 3. Call to check() with the entry point
    let test_code = format!(
        "{}\n\n{}\n\ncheck({})\n",
        solution, problem.test, problem.entry_point
    );

    let result = executor.execute_python(&test_code);

    if result.compiled && result.runtime_error.is_none() && result.tests_passed > 0 {
        (true, true, None)
    } else if result.compiled {
        let err = result.runtime_error.or_else(|| {
            if result.test_output.contains("AssertionError")
                || result.test_output.contains("assert")
            {
                Some("Assertion failed".into())
            } else if !result.test_output.is_empty() {
                Some(result.test_output.chars().take(200).collect())
            } else {
                Some("Unknown test failure".into())
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

/// Also test the canonical solution to verify our test harness works.
fn verify_canonical(problem: &HumanEvalProblem) -> bool {
    let mut executor = CodeExecutor::with_real_execution();
    let code = format!(
        "{}{}\n\n{}\n\ncheck({})\n",
        problem.prompt, problem.canonical_solution, problem.test, problem.entry_point
    );
    let result = executor.execute_python(&code);
    result.compiled && result.runtime_error.is_none()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let use_llm = args.iter().any(|a| a == "--llm");
    let limit = args
        .iter()
        .position(|a| a == "--limit")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(164);
    let verify = args.iter().any(|a| a == "--verify-canonical");

    // Find the dataset
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

    eprintln!("HumanEval Benchmark — {} problems", problems.len());
    if use_llm {
        eprintln!("Mode: LLM (Ollama qwen2.5-coder:7b)\n");
    } else {
        eprintln!("Mode: Native-only (will attempt Python pattern matching)\n");
    }

    // Optionally verify canonical solutions work with our test harness
    if verify {
        eprintln!("Verifying canonical solutions...");
        let mut canonical_pass = 0;
        for p in &problems {
            if verify_canonical(p) {
                canonical_pass += 1;
            } else {
                eprintln!("  WARN: canonical solution FAILS for {}", p.task_id);
            }
        }
        eprintln!("  Canonical: {}/{} pass\n", canonical_pass, problems.len());
    }

    // Run benchmark
    let mut results = Vec::new();
    for (i, problem) in problems.iter().enumerate() {
        let start = Instant::now();

        let (solution, used_native, used_llm_flag) = generate_solution(problem, use_llm);
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
            used_native: used_native,
            used_llm: used_llm_flag,
            elapsed_ms: elapsed,
            code_len: solution.len(),
        });
    }

    // Summary
    let total = results.len();
    let passed = results.iter().filter(|r| r.passed).count();
    let compiled = results.iter().filter(|r| r.compiled).count();
    let used_llm_count = results.iter().filter(|r| r.used_llm).count();
    let used_native_count = results.iter().filter(|r| r.used_native).count();
    let total_time: u128 = results.iter().map(|r| r.elapsed_ms).sum();

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║     Symthaea × HumanEval — Results                      ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!("── Pass@1 ─────────────────────────────────────────────");
    println!(
        "  Passed:   {}/{} ({:.1}%)",
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
    println!();

    println!("── Tier Usage ──────────────────────────────────────────");
    println!("  Native generation: {}", used_native_count);
    println!("  LLM generation:    {}", used_llm_count);
    println!("  Total time: {:.1}s", total_time as f64 / 1000.0);
    println!(
        "  Avg time:   {:.0}ms/problem",
        total_time as f64 / total as f64
    );
    println!();

    // Show failures
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
                "task_id": r.task_id,
                "entry_point": r.entry_point,
                "passed": r.passed,
                "compiled": r.compiled,
                "error": r.error,
                "used_native": r.used_native,
                "used_llm": r.used_llm,
                "elapsed_ms": r.elapsed_ms,
                "code_len": r.code_len,
            })
        })
        .collect();

    let json_report = serde_json::json!({
        "benchmark": "humaneval",
        "model": if use_llm { "symthaea + qwen2.5-coder:7b" } else { "symthaea native-only" },
        "total": total,
        "passed": passed,
        "pass_at_1": passed as f64 / total as f64,
        "compiled": compiled,
        "compilation_rate": compiled as f64 / total as f64,
        "results": json_results,
    });

    let report_path = std::env::temp_dir().join("symthaea_humaneval.json");
    if let Ok(()) = std::fs::write(
        &report_path,
        serde_json::to_string_pretty(&json_report).unwrap(),
    ) {
        eprintln!("\nJSON report: {}", report_path.display());
    }
}
