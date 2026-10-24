// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Paired runtime ablation for GCS topology advisories.
//!
//! The same fixed tasks are evaluated with GCS advisories disabled and enabled.
//! Every task has a hand-authored reference implementation and external tests.
//! Reference-pass + empty-patch-fail controls are required before agent trials.
//!
//! Run with: cargo run --release --example coding_gcs_ablation_benchmark --features 'code_generation geodesic_synthesis' -- --repeats 3

use std::path::PathBuf;
use std::time::Instant;
use symthaea::coding_agent::{CodingAgent, CodingAgentConfig};
use symthaea::language::code_executor::CodeExecutor;

const BENCHMARK_ID: &str = "symthaea_gcs_runtime_ablation";
const SCHEMA_VERSION: u32 = 3;

struct TaskCase {
    id: &'static str,
    // Declared metadata is part of the frozen task contract; do not infer it by scanning Rust text.
    expected_test_functions: usize,
    description: &'static str,
    expected_symbol: &'static str,
    reference_source: &'static str,
    test_source: &'static str,
    mutant_sources: &'static [&'static str],
}

#[derive(Debug, Clone)]
struct OracleRun {
    compiled: bool,
    simulated: bool,
    tests_passed: usize,
    tests_failed: usize,
    runtime_error: Option<String>,
    compile_errors: Vec<String>,
    elapsed_ms: u128,
}

#[derive(Debug, Clone)]
struct EvalResult {
    task_id: String,
    repeat: usize,
    gcs_advisories_enabled: bool,
    execution_order: String,
    code_written: bool,
    expected_symbol_found: bool,
    generated_source_blake3: String,
    compiled: bool,
    simulated: bool,
    expected_test_functions: usize,
    tests_passed: usize,
    tests_failed: usize,
    oracle_complete: bool,
    behavioral_success: bool,
    runtime_error: Option<String>,
    compile_errors: Vec<String>,
    agent_elapsed_ms: u128,
    verification_elapsed_ms: u128,
    iterations: usize,
    energy: f64,
    final_phase: String,
    generation_tiers: Vec<String>,
}

fn task_suite() -> Vec<TaskCase> {
    vec![
        TaskCase {
            id: "add_numbers",
            expected_test_functions: 3,
            description: "Implement `pub fn add_numbers(a: i32, b: i32) -> i32` that adds two integers.",
            expected_symbol: "add_numbers",
            reference_source: "pub fn add_numbers(a: i32, b: i32) -> i32 { a + b }",
            test_source: r#"
#[test] fn add_positive() { assert_eq!(add_numbers(2, 3), 5); }
#[test] fn add_negative() { assert_eq!(add_numbers(-4, -7), -11); }
#[test] fn add_zero() { assert_eq!(add_numbers(9, 0), 9); }
"#,
            mutant_sources: &[
                "pub fn add_numbers(a: i32, b: i32) -> i32 { a - b }",
                "pub fn add_numbers(a: i32, b: i32) -> i32 { a + b + 1 }",
            ],
        },
        TaskCase {
            id: "factorial",
            expected_test_functions: 3,
            description: "Implement `pub fn factorial(n: u64) -> u64` with factorial(0) equal to 1.",
            expected_symbol: "factorial",
            reference_source: "pub fn factorial(n: u64) -> u64 { (1..=n).product() }",
            test_source: r#"
#[test] fn factorial_zero() { assert_eq!(factorial(0), 1); }
#[test] fn factorial_five() { assert_eq!(factorial(5), 120); }
#[test] fn factorial_eight() { assert_eq!(factorial(8), 40320); }
"#,
            mutant_sources: &[
                "pub fn factorial(n: u64) -> u64 { if n == 0 { 0 } else { (1..=n).product() } }",
                "pub fn factorial(n: u64) -> u64 { (1..n).product() }",
            ],
        },
        TaskCase {
            id: "gcd",
            expected_test_functions: 4,
            description: "Implement `pub fn gcd(a: u64, b: u64) -> u64` using the Euclidean algorithm.",
            expected_symbol: "gcd",
            reference_source: "pub fn gcd(mut a: u64, mut b: u64) -> u64 { while b != 0 { let r = a % b; a = b; b = r; } a }",
            test_source: r#"
#[test] fn gcd_common_factor() { assert_eq!(gcd(12, 8), 4); }
#[test] fn gcd_coprime() { assert_eq!(gcd(7, 13), 1); }
#[test] fn gcd_zero_left() { assert_eq!(gcd(0, 5), 5); }
#[test] fn gcd_zero_right() { assert_eq!(gcd(9, 0), 9); }
"#,
            mutant_sources: &["pub fn gcd(a: u64, b: u64) -> u64 { a.min(b) }"],
        },
        TaskCase {
            id: "is_even",
            expected_test_functions: 4,
            description: "Implement `pub fn is_even(n: i64) -> bool` for positive, zero, and negative integers.",
            expected_symbol: "is_even",
            reference_source: "pub fn is_even(n: i64) -> bool { n % 2 == 0 }",
            test_source: r#"
#[test] fn even_positive() { assert!(is_even(4)); }
#[test] fn odd_positive() { assert!(!is_even(7)); }
#[test] fn even_negative() { assert!(is_even(-8)); }
#[test] fn even_zero() { assert!(is_even(0)); }
"#,
            mutant_sources: &["pub fn is_even(n: i64) -> bool { n % 2 != 0 }"],
        },
        TaskCase {
            id: "reverse_string",
            expected_test_functions: 3,
            description: "Implement `pub fn reverse_string(s: &str) -> String` by reversing Unicode scalar values.",
            expected_symbol: "reverse_string",
            reference_source: "pub fn reverse_string(s: &str) -> String { s.chars().rev().collect() }",
            test_source: r#"
#[test] fn reverse_word() { assert_eq!(reverse_string("hello"), "olleh"); }
#[test] fn reverse_empty() { assert_eq!(reverse_string(""), ""); }
#[test] fn reverse_unicode() { assert_eq!(reverse_string("hé🙂"), "🙂éh"); }
"#,
            mutant_sources: &["pub fn reverse_string(s: &str) -> String { s.to_string() }"],
        },
        TaskCase {
            id: "sum_slice",
            expected_test_functions: 3,
            description: "Implement `pub fn sum_slice(values: &[i32]) -> i32` that sums all elements.",
            expected_symbol: "sum_slice",
            reference_source: "pub fn sum_slice(values: &[i32]) -> i32 { values.iter().copied().sum() }",
            test_source: r#"
#[test] fn sum_values() { assert_eq!(sum_slice(&[1, 2, 3, 4]), 10); }
#[test] fn sum_empty() { assert_eq!(sum_slice(&[]), 0); }
#[test] fn sum_negative() { assert_eq!(sum_slice(&[-2, 5, -3]), 0); }
"#,
            mutant_sources: &[
                "pub fn sum_slice(values: &[i32]) -> i32 { values.iter().copied().max().unwrap_or(0) }",
            ],
        },
        TaskCase {
            id: "binary_search",
            expected_test_functions: 4,
            description: "Implement `pub fn binary_search(values: &[i32], needle: i32) -> Option<usize>` on sorted slices.",
            expected_symbol: "binary_search",
            reference_source: "pub fn binary_search(values: &[i32], needle: i32) -> Option<usize> { values.binary_search(&needle).ok() }",
            test_source: r#"
#[test] fn search_found_middle() { assert_eq!(binary_search(&[1, 3, 5, 7], 5), Some(2)); }
#[test] fn search_found_first() { assert_eq!(binary_search(&[1, 3, 5, 7], 1), Some(0)); }
#[test] fn search_missing() { assert_eq!(binary_search(&[1, 3, 5, 7], 4), None); }
#[test] fn search_empty() { assert_eq!(binary_search(&[], 1), None); }
"#,
            mutant_sources: &[
                "pub fn binary_search(values: &[i32], needle: i32) -> Option<usize> { values.binary_search(&needle).ok().map(|index| index + 1) }",
            ],
        },
        TaskCase {
            id: "is_palindrome",
            expected_test_functions: 5,
            description: "Implement `pub fn is_palindrome(s: &str) -> bool` with exact, case-sensitive character comparison.",
            expected_symbol: "is_palindrome",
            reference_source: "pub fn is_palindrome(s: &str) -> bool { s.chars().eq(s.chars().rev()) }",
            test_source: r#"
#[test] fn palindrome_racecar() { assert!(is_palindrome("racecar")); }
#[test] fn palindrome_even_length() { assert!(is_palindrome("abba")); }
#[test] fn not_palindrome() { assert!(!is_palindrome("hello")); }
#[test] fn not_palindrome_same_ends() { assert!(!is_palindrome("abca")); }
#[test] fn palindrome_empty() { assert!(is_palindrome("")); }
"#,
            mutant_sources: &[
                "pub fn is_palindrome(s: &str) -> bool { s.chars().next() == s.chars().last() }",
            ],
        },
    ]
}

fn expected_test_functions(task: &TaskCase) -> usize {
    task.expected_test_functions
}

fn source_digest(source: &str) -> String {
    blake3::hash(source.as_bytes()).to_hex().to_string()
}

fn task_digest(task: &TaskCase) -> String {
    // Length-prefix each field so embedded newlines cannot create ambiguous digests.
    let mut hasher = blake3::Hasher::new();
    let fields = [
        task.id,
        task.description,
        task.expected_symbol,
        task.reference_source,
        task.test_source,
    ];
    for field in fields {
        hasher.update(&(field.len() as u64).to_be_bytes());
        hasher.update(field.as_bytes());
    }
    hasher.update(&(task.expected_test_functions as u64).to_be_bytes());
    hasher.update(&(task.mutant_sources.len() as u64).to_be_bytes());
    for mutant in task.mutant_sources {
        hasher.update(&(mutant.len() as u64).to_be_bytes());
        hasher.update(mutant.as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn manifest_digest(tasks: &[TaskCase]) -> String {
    // The manifest is an ordered list of length-prefixed (task ID, task digest)
    // pairs, so separators/newlines inside a future task ID cannot alias a row.
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(tasks.len() as u64).to_be_bytes());
    for task in tasks {
        let id = task.id.as_bytes();
        let digest = task_digest(task);
        hasher.update(&(id.len() as u64).to_be_bytes());
        hasher.update(id);
        hasher.update(&(digest.len() as u64).to_be_bytes());
        hasher.update(digest.as_bytes());
    }
    hasher.finalize().to_hex().to_string()
}

fn evaluate_source(source: &str, test_source: &str) -> OracleRun {
    // Even an empty patch is sent to the real compiler with the same external oracle.
    // This is essential to the reference-pass / empty-patch-fail control; do not treat
    // an empty source string as a synthetic failure without exercising the test harness.
    let start = Instant::now();
    let mut executor = CodeExecutor::with_real_execution();
    let result = executor.execute_rust(source, Some(test_source));
    OracleRun {
        compiled: result.compiled,
        simulated: result.simulated,
        tests_passed: result.tests_passed,
        tests_failed: result.tests_failed,
        runtime_error: result.runtime_error,
        compile_errors: result.compile_errors.into_iter().take(5).collect(),
        elapsed_ms: start.elapsed().as_millis(),
    }
}

fn is_oracle_complete(run: &OracleRun, expected: usize) -> bool {
    expected > 0
        && !run.simulated
        && run.compiled
        && run.tests_passed + run.tests_failed == expected
}

fn is_behaviorally_successful(run: &OracleRun, expected: usize) -> bool {
    is_oracle_complete(run, expected)
        && run.runtime_error.is_none()
        && run.tests_passed == expected
        && run.tests_failed == 0
}

fn run_candidate(
    task: &TaskCase,
    repeat: usize,
    gcs_advisories_enabled: bool,
    execution_order: &str,
    use_llm: bool,
    use_cloud: bool,
) -> EvalResult {
    let temp_dir = tempfile::tempdir().expect("temporary work directory");
    let config = CodingAgentConfig {
        max_iterations: 8,
        working_dir: temp_dir.path().to_path_buf(),
        target_file: Some(PathBuf::from("generated.rs")),
        use_local_llm: use_llm,
        use_cloud_llm: use_cloud,
        enable_real_exec: false,
        enable_gcs_advisories: gcs_advisories_enabled,
        ..Default::default()
    };
    let start = Instant::now();
    let mut agent = CodingAgent::new(config).expect("coding agent creation");
    let result = agent.run(task.description);
    let agent_elapsed_ms = start.elapsed().as_millis();
    let output_path = temp_dir.path().join("generated.rs");
    let source = std::fs::read_to_string(&output_path).unwrap_or_default();
    let code_written = !source.trim().is_empty();
    let expected_symbol_found = source.contains(task.expected_symbol);
    let expected = expected_test_functions(task);
    let oracle = evaluate_source(&source, task.test_source);
    let verification_elapsed_ms = oracle.elapsed_ms;
    // CodeExecutor stores a nonzero test-process exit in runtime_error even when
    // ordinary assertions fail. Exact test counts distinguish a completed failing
    // oracle from an incomplete/crashed test run; success still requires no error.
    let oracle_complete = is_oracle_complete(&oracle, expected);
    let behavioral_success = oracle_complete && is_behaviorally_successful(&oracle, expected);
    EvalResult {
        task_id: task.id.to_string(),
        repeat,
        gcs_advisories_enabled,
        execution_order: execution_order.to_string(),
        code_written,
        expected_symbol_found,
        generated_source_blake3: source_digest(&source),
        compiled: oracle.compiled,
        simulated: oracle.simulated,
        expected_test_functions: expected,
        tests_passed: oracle.tests_passed,
        tests_failed: oracle.tests_failed,
        oracle_complete,
        behavioral_success,
        runtime_error: oracle.runtime_error,
        compile_errors: oracle.compile_errors,
        agent_elapsed_ms,
        verification_elapsed_ms,
        iterations: result.iterations_used,
        energy: result.total_energy,
        final_phase: format!("{}", result.final_phase),
        generation_tiers: result
            .generation_tiers
            .iter()
            .map(ToString::to_string)
            .collect(),
    }
}

fn eval_json(result: &EvalResult) -> serde_json::Value {
    serde_json::json!({
        "task_id": result.task_id,
        "repeat": result.repeat,
        "gcs_advisories_enabled": result.gcs_advisories_enabled,
        "execution_order": result.execution_order,
        "code_written": result.code_written,
        "expected_symbol_found": result.expected_symbol_found,
        "generated_source_blake3": result.generated_source_blake3,
        "compiled": result.compiled,
        "simulated": result.simulated,
        "expected_test_functions": result.expected_test_functions,
        "tests_passed": result.tests_passed,
        "tests_failed": result.tests_failed,
        "oracle_complete": result.oracle_complete,
        "behavioral_success": result.behavioral_success,
        "runtime_error": result.runtime_error,
        "compile_errors": result.compile_errors,
        "agent_elapsed_ms": result.agent_elapsed_ms,
        "verification_elapsed_ms": result.verification_elapsed_ms,
        "iterations": result.iterations,
        "energy": result.energy,
        "final_phase": result.final_phase,
        "generation_tiers": result.generation_tiers
    })
}

fn mode_summary(results: &[EvalResult], enabled: bool) -> serde_json::Value {
    let mode: Vec<&EvalResult> = results
        .iter()
        .filter(|r| r.gcs_advisories_enabled == enabled)
        .collect();
    let total = mode.len();
    let passed = mode.iter().filter(|r| r.behavioral_success).count();
    let compiled = mode.iter().filter(|r| r.compiled && !r.simulated).count();
    let complete = mode.iter().filter(|r| r.oracle_complete).count();
    let agent_ms: u128 = mode.iter().map(|r| r.agent_elapsed_ms).sum();
    let energy: f64 = mode.iter().map(|r| r.energy).sum();
    serde_json::json!({
        "gcs_advisories_enabled": enabled,
        "runs": total,
        "behavioral_successes": passed,
        "pass_rate": if total == 0 { 0.0 } else { passed as f64 / total as f64 },
        "real_compiles": compiled,
        "complete_oracles": complete,
        "complete_oracle_rate": if total == 0 { 0.0 } else { complete as f64 / total as f64 },
        "agent_elapsed_ms_total": agent_ms,
        "agent_elapsed_ms_mean": if total == 0 { 0.0 } else { agent_ms as f64 / total as f64 },
        "energy_total": energy,
        "energy_mean": if total == 0 { 0.0 } else { energy / total as f64 }
    })
}

fn paired_summary(pairs: &[(EvalResult, EvalResult)]) -> serde_json::Value {
    let total_pairs = pairs.len();
    // Incomplete execution is unknown, not a loss or a tie. Restrict paired
    // treatment-effect counts to pairs where both arms had complete oracles.
    let comparable: Vec<&(EvalResult, EvalResult)> = pairs
        .iter()
        .filter(|(off, on)| off.oracle_complete && on.oracle_complete)
        .collect();
    let comparable_pairs = comparable.len();
    let unresolved_pairs = total_pairs.saturating_sub(comparable_pairs);
    let on_wins = comparable
        .iter()
        .filter(|(off, on)| !off.behavioral_success && on.behavioral_success)
        .count();
    let off_wins = comparable
        .iter()
        .filter(|(off, on)| off.behavioral_success && !on.behavioral_success)
        .count();
    let ties = comparable_pairs.saturating_sub(on_wins + off_wins);
    serde_json::json!({
        "paired_task_trials": total_pairs,
        "comparable_pairs": comparable_pairs,
        "unresolved_pairs": unresolved_pairs,
        "gcs_on_wins": on_wins,
        "gcs_off_wins": off_wins,
        "ties": ties,
        "net_pass_delta_percentage_points": if comparable_pairs == 0 { 0.0 } else { (on_wins as f64 - off_wins as f64) / comparable_pairs as f64 * 100.0 },
        "interpretation": "Descriptive pilot only; win/loss/tie and pass delta include only pairs with complete behavioral oracles. Unresolved pairs are excluded, not treated as ties. Do not claim statistical significance from this small fixed task suite."
    })
}

fn resolve_model_identity(
    use_llm: bool,
    use_cloud: bool,
    local_model: Option<&str>,
    cloud_model: Option<&str>,
    cloud_api_key_present: bool,
) -> Result<(String, String), String> {
    if use_llm && use_cloud {
        return Err("choose at most one backend: --llm or --cloud".to_string());
    }
    if use_cloud {
        if !cloud_api_key_present {
            return Err("--cloud requires a non-empty ANTHROPIC_API_KEY; refusing to silently benchmark a missing/simulated cloud backend".to_string());
        }
        let model = configured_model("ANTHROPIC_MODEL", cloud_model, "claude-sonnet-4-20250514")?;
        return Ok(("cloud".to_string(), model));
    }
    if use_llm {
        // Keep this default aligned with OllamaBackend::new() in llm_backend.rs.
        let model = configured_model("SYMTHAEA_LLM_MODEL", local_model, "gemma4:e2b")?;
        return Ok(("ollama".to_string(), model));
    }
    Ok((
        "native".to_string(),
        "native dispatcher / no external LLM requested".to_string(),
    ))
}

fn configured_model(env_name: &str, value: Option<&str>, default: &str) -> Result<String, String> {
    match value {
        Some(value) if value.trim().is_empty() => Err(format!(
            "{env_name} is set but empty; refusing an ambiguous model identity"
        )),
        Some(value) => Ok(value.trim().to_string()),
        None => Ok(default.to_string()),
    }
}

fn validate_source_revision(supplied: &str, actual_head: &str) -> Result<(), &'static str> {
    let valid_sha = |sha: &str| sha.len() == 40 && sha.bytes().all(|byte| byte.is_ascii_hexdigit());
    if !valid_sha(supplied) {
        return Err("GIT_COMMIT_SHA must be a full 40-character hexadecimal commit SHA");
    }
    if !valid_sha(actual_head) {
        return Err("git rev-parse HEAD did not return a full commit SHA");
    }
    if !supplied.eq_ignore_ascii_case(actual_head) {
        return Err("GIT_COMMIT_SHA does not match git rev-parse HEAD");
    }
    Ok(())
}

fn validate_clean_worktree(porcelain_status: &str) -> Result<(), &'static str> {
    if porcelain_status.is_empty() {
        Ok(())
    } else {
        Err("working tree is dirty; use a clean checkout for an auditable benchmark")
    }
}

fn require_source_revision() -> String {
    let supplied = match std::env::var("GIT_COMMIT_SHA") {
        Ok(value) => value,
        Err(_) => {
            eprintln!("GIT_COMMIT_SHA is required; set it to `git rev-parse HEAD`");
            std::process::exit(2);
        }
    };
    let root = env!("CARGO_MANIFEST_DIR");
    let head_output = match std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(root)
        .output()
    {
        Ok(output) if output.status.success() => output,
        _ => {
            eprintln!("Unable to resolve the source revision using git rev-parse HEAD");
            std::process::exit(2);
        }
    };
    let actual_head = String::from_utf8_lossy(&head_output.stdout)
        .trim()
        .to_string();
    if let Err(error) = validate_source_revision(&supplied, &actual_head) {
        eprintln!("{error}");
        std::process::exit(2);
    }

    let status_output = match std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=all"])
        .current_dir(root)
        .output()
    {
        Ok(output) if output.status.success() => output,
        _ => {
            eprintln!("Unable to verify git working-tree cleanliness");
            std::process::exit(2);
        }
    };
    let porcelain_status = String::from_utf8_lossy(&status_output.stdout);
    if let Err(error) = validate_clean_worktree(porcelain_status.trim()) {
        eprintln!("{error}");
        std::process::exit(2);
    }

    actual_head
}

struct CliOptions {
    use_llm: bool,
    use_cloud: bool,
    repeats: usize,
    output_path: PathBuf,
    print_manifest: bool,
}

fn parse_cli(args: &[String]) -> Result<Option<CliOptions>, String> {
    if args.len() == 1 && args[0] == "--print-manifest" {
        return Ok(Some(CliOptions {
            use_llm: false,
            use_cloud: false,
            repeats: 3,
            output_path: std::env::temp_dir().join("symthaea_gcs_ablation_v3.json"),
            print_manifest: true,
        }));
    }
    if args.iter().any(|arg| arg == "--print-manifest") {
        return Err("--print-manifest must be used by itself".to_string());
    }
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        if args.len() == 1 {
            return Ok(None);
        }
        return Err("--help/-h must be used by itself".to_string());
    }

    let mut use_llm = false;
    let mut use_cloud = false;
    let mut repeats = 3_usize;
    let mut repeats_seen = false;
    let mut output_path = std::env::temp_dir().join("symthaea_gcs_ablation_v3.json");
    let mut output_seen = false;
    let mut index = 0;

    while index < args.len() {
        match args[index].as_str() {
            "--llm" if !use_llm => use_llm = true,
            "--llm" => return Err("--llm may only be specified once".to_string()),
            "--cloud" if !use_cloud => use_cloud = true,
            "--cloud" => return Err("--cloud may only be specified once".to_string()),
            "--repeats" => {
                if repeats_seen {
                    return Err("--repeats may only be specified once".to_string());
                }
                let value = args.get(index + 1).ok_or("--repeats requires an integer")?;
                repeats = value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid --repeats value: {value:?}"))?;
                if !(1..=20).contains(&repeats) {
                    return Err("--repeats must be in 1..=20".to_string());
                }
                repeats_seen = true;
                index += 1;
            }
            "--json-out" => {
                if output_seen {
                    return Err("--json-out may only be specified once".to_string());
                }
                let value = args.get(index + 1).ok_or("--json-out requires a path")?;
                if value.trim().is_empty() || value.starts_with("--") {
                    return Err("--json-out requires a non-empty path".to_string());
                }
                output_path = PathBuf::from(value);
                output_seen = true;
                index += 1;
            }
            option => return Err(format!("unknown or duplicate option: {option}")),
        }
        index += 1;
    }

    if use_llm && use_cloud {
        return Err("choose at most one backend: --llm or --cloud".to_string());
    }

    Ok(Some(CliOptions {
        use_llm,
        use_cloud,
        repeats,
        output_path,
        print_manifest: false,
    }))
}

fn print_usage() {
    eprintln!(
        "Usage: coding_gcs_ablation_benchmark [--print-manifest | [--llm | --cloud] [--repeats 1..20] [--json-out PATH]]\n\
         The working tree must be clean and GIT_COMMIT_SHA must match git rev-parse HEAD.\n\
         --print-manifest prints the source SHA and task-manifest digest without executing controls or trials."
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let options = match parse_cli(&args) {
        Ok(Some(options)) => options,
        Ok(None) => {
            print_usage();
            return;
        }
        Err(error) => {
            eprintln!("{error}");
            print_usage();
            std::process::exit(2);
        }
    };
    // Refuse to produce a benchmark receipt that cannot be tied to an exact,
    // clean source checkout. Do this before controls or any agent trials.
    let source_revision = require_source_revision();
    let CliOptions {
        use_llm,
        use_cloud,
        repeats,
        output_path,
        print_manifest,
    } = options;
    let tasks = task_suite();
    if tasks
        .iter()
        .any(|task| expected_test_functions(task) == 0 || task.mutant_sources.is_empty())
    {
        eprintln!(
            "Invalid task suite: every task must have declared tests and at least one semantic mutant"
        );
        std::process::exit(2);
    }
    let manifest = manifest_digest(&tasks);
    if print_manifest {
        println!("source_revision={source_revision}");
        println!("task_count={}", tasks.len());
        println!("task_manifest_blake3={manifest}");
        println!("This command did not execute oracle controls or coding-agent trials.");
        return;
    }

    // Resolve model identity before any trials. The cloud backend is optional in
    // CodingAgent::new, so --cloud without credentials can otherwise silently run
    // without a real cloud backend. Never label such trials as a cloud evaluation.
    let (model_mode, model_label) = match resolve_model_identity(
        use_llm,
        use_cloud,
        std::env::var("SYMTHAEA_LLM_MODEL").ok().as_deref(),
        std::env::var("ANTHROPIC_MODEL").ok().as_deref(),
        std::env::var("ANTHROPIC_API_KEY").is_ok_and(|key| !key.trim().is_empty()),
    ) {
        Ok(identity) => identity,
        Err(error) => {
            eprintln!("Invalid benchmark backend configuration: {error}");
            std::process::exit(2);
        }
    };

    // Two-sided oracle controls run before any agent trials.
    let mut reference_controls = Vec::new();
    let mut empty_patch_controls = Vec::new();
    let mut mutation_controls = Vec::new();
    for task in &tasks {
        let expected = expected_test_functions(task);
        let reference = evaluate_source(task.reference_source, task.test_source);
        let reference_passes = is_behaviorally_successful(&reference, expected);
        let reference_oracle_complete = is_oracle_complete(&reference, expected);
        let empty = evaluate_source("", task.test_source);
        // This benchmark's empty patch should fail specifically as a real
        // rustc compilation rejection, not as a sandbox/runner failure.
        let empty_patch_fails = !empty.simulated
            && !empty.compiled
            && empty.runtime_error.is_none()
            && empty
                .compile_errors
                .iter()
                .any(|error| error.trim_start().starts_with("error"));
        reference_controls.push(serde_json::json!({
            "task_id": task.id,
            "reference_blake3": source_digest(task.reference_source),
            "reference_passes": reference_passes,
            "compiled": reference.compiled,
            "simulated": reference.simulated,
            "runtime_error": reference.runtime_error,
            "tests_expected": expected,
            "tests_passed": reference.tests_passed,
            "tests_failed": reference.tests_failed,
            "oracle_complete": reference_oracle_complete
        }));
        empty_patch_controls.push(serde_json::json!({
            "task_id": task.id,
            "empty_patch_blake3": source_digest(""),
            "empty_patch_fails": empty_patch_fails,
            "compiled": empty.compiled,
            "simulated": empty.simulated,
            "runtime_error": empty.runtime_error,
            "compile_errors": empty.compile_errors.iter().take(2).collect::<Vec<_>>()
        }));
        if !reference_passes || !empty_patch_fails {
            eprintln!(
                "Oracle control failed for task {}: reference_passes={}, empty_patch_fails={}",
                task.id, reference_passes, empty_patch_fails
            );
            std::process::exit(3);
        }
        if task.mutant_sources.is_empty() {
            eprintln!("Oracle mutation controls missing for task {}", task.id);
            std::process::exit(3);
        }
        for (mutation_index, mutant_source) in task.mutant_sources.iter().enumerate() {
            let mutant = evaluate_source(mutant_source, task.test_source);
            // A failed assertion makes rustc's test binary exit nonzero. Count
            // it as a killed mutant only when the full test catalogue was observed;
            // runtime/infrastructure crashes with missing counts stay unresolved.
            let mutant_oracle_complete = is_oracle_complete(&mutant, expected);
            let mutant_killed = mutant_oracle_complete && mutant.tests_failed > 0;
            mutation_controls.push(serde_json::json!({
                "task_id": task.id,
                "mutation_index": mutation_index,
                "mutation_source_blake3": source_digest(mutant_source),
                "compiled": mutant.compiled,
                "simulated": mutant.simulated,
                "tests_expected": expected,
                "tests_passed": mutant.tests_passed,
                "tests_failed": mutant.tests_failed,
                "runtime_error": mutant.runtime_error,
                "oracle_complete": mutant_oracle_complete,
                "mutant_killed": mutant_killed,
                "compile_errors": mutant.compile_errors.iter().take(2).collect::<Vec<_>>()
            }));
            if !mutant_killed {
                eprintln!(
                    "Oracle mutation control failed for task {} mutant {}: compiled={}, complete={}, failed_tests={}",
                    task.id,
                    mutation_index,
                    mutant.compiled,
                    mutant_oracle_complete,
                    mutant.tests_failed
                );
                std::process::exit(3);
            }
        }
    }

    let mut all_results = Vec::new();
    let mut pairs = Vec::new();
    for (task_idx, task) in tasks.iter().enumerate() {
        for repeat_idx in 0..repeats {
            let order = if (task_idx + repeat_idx) % 2 == 0 {
                "off_then_on"
            } else {
                "on_then_off"
            };
            let (off, on) = if order == "off_then_on" {
                let off = run_candidate(task, repeat_idx + 1, false, order, use_llm, use_cloud);
                let on = run_candidate(task, repeat_idx + 1, true, order, use_llm, use_cloud);
                (off, on)
            } else {
                let on = run_candidate(task, repeat_idx + 1, true, order, use_llm, use_cloud);
                let off = run_candidate(task, repeat_idx + 1, false, order, use_llm, use_cloud);
                (off, on)
            };
            eprintln!(
                "[{}/{}] task={} repeat={} off={} on={} order={}",
                task_idx + 1,
                tasks.len(),
                task.id,
                repeat_idx + 1,
                off.behavioral_success,
                on.behavioral_success,
                order
            );
            all_results.push(off.clone());
            all_results.push(on.clone());
            pairs.push((off, on));
        }
    }

    let report = serde_json::json!({
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_ID,
        "source_revision": source_revision,
        "model_mode": model_mode,
        "model_label": model_label,
        "runtime_configuration": {
            "compiled_with_geodesic_synthesis": cfg!(feature = "geodesic_synthesis"),
            "gcs_toggle": "CodingAgentConfig.enable_gcs_advisories",
            "agent_internal_real_exec": false,
            "external_oracle_real_exec": true,
            "requested_backend": model_mode,
            "requested_model": model_label
        },
        "task_count": tasks.len(),
        "repeats": repeats,
        "task_manifest_blake3": manifest,
        "task_manifest": tasks.iter().map(|task| serde_json::json!({
            "task_id": task.id,
            "description": task.description,
            "expected_symbol": task.expected_symbol,
            "reference_source": task.reference_source,
            "test_source": task.test_source,
            "mutant_sources": task.mutant_sources,
            "task_digest_blake3": task_digest(task),
            "expected_test_functions": expected_test_functions(task),
            "reference_source_blake3": source_digest(task.reference_source),
            "test_source_blake3": source_digest(task.test_source)
        })).collect::<Vec<_>>(),
        "oracle_controls": {
            "reference_controls": reference_controls,
            "empty_patch_controls": empty_patch_controls,
            "mutation_controls": mutation_controls,
            "all_references_pass": true,
            "all_empty_patches_fail": true,
            "all_mutants_killed": true
        },
        "summary": {
            "gcs_off": mode_summary(&all_results, false),
            "gcs_on": mode_summary(&all_results, true),
            "paired": paired_summary(&pairs)
        },
        "runs": all_results.iter().map(eval_json).collect::<Vec<_>>()
    });
    if let Some(parent) = output_path.parent() {
        if let Err(error) = std::fs::create_dir_all(parent) {
            eprintln!("Failed to create JSON output directory: {error}");
            std::process::exit(4);
        }
    }
    match serde_json::to_string_pretty(&report) {
        Ok(json) => match std::fs::write(&output_path, json) {
            Ok(()) => eprintln!("GCS ablation receipt: {}", output_path.display()),
            Err(error) => {
                eprintln!("Failed to write receipt: {error}");
                std::process::exit(4);
            }
        },
        Err(error) => {
            eprintln!("Failed to serialize receipt: {error}");
            std::process::exit(4);
        }
    }
}

#[cfg(test)]
mod provenance_tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn manifest_only_mode_is_explicit_and_cannot_be_combined() {
        let options = parse_cli(&args(&["--print-manifest"])).unwrap().unwrap();
        assert!(options.print_manifest);
        assert!(!options.use_llm && !options.use_cloud);
        assert!(parse_cli(&args(&["--print-manifest", "--cloud"])).is_err());
        assert!(parse_cli(&args(&["--print-manifest", "--repeats", "3"])).is_err());
    }

    #[test]
    fn cli_defaults_are_explicit_and_reproducible() {
        let parsed = parse_cli(&args(&[])).unwrap().unwrap();
        assert!(!parsed.use_llm);
        assert!(!parsed.use_cloud);
        assert_eq!(parsed.repeats, 3);
        assert!(
            parsed
                .output_path
                .ends_with("symthaea_gcs_ablation_v3.json")
        );
    }

    #[test]
    fn cli_accepts_each_single_backend() {
        let local = parse_cli(&args(&["--llm"])).unwrap().unwrap();
        assert!(local.use_llm && !local.use_cloud);
        let cloud = parse_cli(&args(&["--cloud"])).unwrap().unwrap();
        assert!(cloud.use_cloud && !cloud.use_llm);
    }

    #[test]
    fn cli_rejects_conflicting_backends_and_bad_repeats() {
        assert!(parse_cli(&args(&["--llm", "--cloud"])).is_err());
        assert!(parse_cli(&args(&["--repeats", "nonsense"])).is_err());
        assert!(parse_cli(&args(&["--repeats", "0"])).is_err());
        assert!(parse_cli(&args(&["--repeats", "21"])).is_err());
        assert!(parse_cli(&args(&["--repeats"])).is_err());
    }

    #[test]
    fn cli_rejects_missing_or_duplicate_output_options() {
        assert!(parse_cli(&args(&["--json-out"])).is_err());
        assert!(parse_cli(&args(&["--json-out", "--cloud"])).is_err());
        assert!(parse_cli(&args(&["--json-out", "a.json", "--json-out", "b.json"])).is_err());
        assert!(parse_cli(&args(&["--unknown"])).is_err());
        assert!(parse_cli(&args(&["--help", "--cloud"])).is_err());
        assert!(parse_cli(&args(&["--help"])).unwrap().is_none());
    }

    #[test]
    fn model_identity_matches_actual_local_backend_default() {
        assert_eq!(
            resolve_model_identity(true, false, None, None, false).unwrap(),
            ("ollama".to_string(), "gemma4:e2b".to_string())
        );
        assert_eq!(
            resolve_model_identity(true, false, Some("qwen3-coder:30b"), None, false).unwrap(),
            ("ollama".to_string(), "qwen3-coder:30b".to_string())
        );
    }

    #[test]
    fn cloud_backend_requires_real_credentials_and_resolves_actual_default() {
        assert!(resolve_model_identity(false, true, None, None, false).is_err());
        assert!(resolve_model_identity(false, true, None, Some("claude-test"), true).is_ok());
        assert_eq!(
            resolve_model_identity(false, true, None, None, true).unwrap(),
            ("cloud".to_string(), "claude-sonnet-4-20250514".to_string())
        );
    }

    #[test]
    fn backend_identity_rejects_empty_or_conflicting_configuration() {
        assert!(resolve_model_identity(true, true, None, None, true).is_err());
        assert!(resolve_model_identity(true, false, Some("  "), None, false).is_err());
        assert!(resolve_model_identity(false, true, None, Some(""), true).is_err());
        assert_eq!(
            resolve_model_identity(false, false, None, None, false)
                .unwrap()
                .0,
            "native"
        );
    }

    #[test]
    fn assertion_failure_with_full_counts_is_complete_but_not_success() {
        let run = OracleRun {
            compiled: true,
            simulated: false,
            tests_passed: 2,
            tests_failed: 1,
            runtime_error: Some("test process exited nonzero after assertion failure".to_string()),
            compile_errors: Vec::new(),
            elapsed_ms: 10,
        };
        assert!(is_oracle_complete(&run, 3));
        assert!(!is_behaviorally_successful(&run, 3));
    }

    #[test]
    fn missing_test_results_remain_unresolved() {
        let run = OracleRun {
            compiled: true,
            simulated: false,
            tests_passed: 2,
            tests_failed: 0,
            runtime_error: Some("test process crashed before all cases completed".to_string()),
            compile_errors: Vec::new(),
            elapsed_ms: 10,
        };
        assert!(!is_oracle_complete(&run, 3));
        assert!(!is_behaviorally_successful(&run, 3));
    }

    #[test]
    fn declared_oracle_count_is_bound_into_task_digest() {
        let mut task = task_suite().remove(0);
        let original = task_digest(&task);
        task.expected_test_functions += 1;
        assert_ne!(original, task_digest(&task));
    }

    #[test]
    fn accepts_exact_clean_revision_metadata() {
        let sha = "a".repeat(40);
        assert!(validate_source_revision(&sha, &sha).is_ok());
        assert!(validate_clean_worktree("").is_ok());
    }

    #[test]
    fn rejects_missing_malformed_or_mismatched_revision() {
        let sha = "a".repeat(40);
        assert!(validate_source_revision("", &sha).is_err());
        assert!(validate_source_revision(&"g".repeat(40), &sha).is_err());
        assert!(validate_source_revision(&"b".repeat(40), &sha).is_err());
        assert!(validate_source_revision(&sha, "short").is_err());
    }

    #[test]
    fn rejects_dirty_worktree_status() {
        assert!(validate_clean_worktree(" M src/coding_agent/mod.rs\\n").is_err());
        assert!(validate_clean_worktree("?? untracked.rs\\n").is_err());
    }
}
