// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Verified coding testbench for HumanEval-style Rust tasks.
//!
//! This is an executable validation harness, not a discriminator benchmark.
//! It asks Symthaea to generate code, verifies it through the real executor,
//! and reports pass@1-style results.
//!
//! Run:
//!   cargo run --example benchmark_humaneval_verified --features code_generation
//!   cargo run --example benchmark_humaneval_verified --features code_generation -- --input data/benchmarks/humaneval-rust.jsonl
//!
//! JSONL schema:
//! {"id":"add","name":"add","purpose":"Add two integers","signature":"fn add(a: i32, b: i32) -> i32","examples":[{"input":"add(2, 3)","output":"5"}]}

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use symthaea::hdc::code_encoder::CodeHDEncoder;
use symthaea::language::code_executor::CodeExecutor;
use symthaea::language::code_generator::CodeGenerator;
use symthaea::language::verified_generation::{generate_verified_function, VerifiedCode};

#[derive(Debug, Clone, Deserialize)]
struct ExternalTask {
    id: String,
    name: String,
    purpose: String,
    signature: String,
    #[serde(default)]
    examples: Vec<IoExample>,
}

#[derive(Debug, Clone, Deserialize)]
struct IoExample {
    input: String,
    output: String,
}

#[derive(Debug, Serialize)]
struct TaskReport {
    id: String,
    name: String,
    compiled: bool,
    tests_passed: bool,
    guaranteed: bool,
    test_count_passed: usize,
    test_count_failed: usize,
    compile_retries: usize,
    test_retries: usize,
    elapsed_ms: u128,
    first_error: Option<String>,
}

#[derive(Debug, Serialize)]
struct BenchReport {
    benchmark: String,
    task_count: usize,
    guaranteed_count: usize,
    compiled_count: usize,
    pass_at_1: f64,
    compile_rate: f64,
    elapsed_ms: u128,
    source: String,
    tasks: Vec<TaskReport>,
}

#[derive(Debug, Default)]
struct Args {
    input: Option<PathBuf>,
    limit: Option<usize>,
    json: bool,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let source = args
        .input
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "built-in-smoke".to_string());
    let mut tasks = match args.input.as_deref() {
        Some(path) => load_jsonl_tasks(path)?,
        None => built_in_tasks(),
    };

    if let Some(limit) = args.limit {
        tasks.truncate(limit);
    }
    if tasks.is_empty() {
        bail!("no benchmark tasks loaded");
    }

    let start = Instant::now();
    let generator = CodeGenerator::new(CodeHDEncoder::new(512));
    let mut executor = CodeExecutor::with_real_execution();
    if !executor.supports_real_execution() {
        bail!("benchmark_humaneval_verified requires real execution");
    }

    let mut task_reports = Vec::with_capacity(tasks.len());
    for task in &tasks {
        let task_start = Instant::now();
        let example_refs: Vec<(&str, &str)> = task
            .examples
            .iter()
            .map(|example| (example.input.as_str(), example.output.as_str()))
            .collect();

        let result = generate_verified_function(
            &generator,
            &mut executor,
            &task.name,
            &task.purpose,
            &task.signature,
            &example_refs,
        );

        task_reports.push(report_task(task, &result, task_start.elapsed().as_millis()));
    }

    let guaranteed_count = task_reports.iter().filter(|task| task.guaranteed).count();
    let compiled_count = task_reports.iter().filter(|task| task.compiled).count();
    let report = BenchReport {
        benchmark: "benchmark_humaneval_verified".to_string(),
        task_count: task_reports.len(),
        guaranteed_count,
        compiled_count,
        pass_at_1: guaranteed_count as f64 / task_reports.len() as f64,
        compile_rate: compiled_count as f64 / task_reports.len() as f64,
        elapsed_ms: start.elapsed().as_millis(),
        source,
        tasks: task_reports,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_text_report(&report);
    }

    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut args = Args::default();
    let mut iter = std::env::args().skip(1);

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--input" => {
                let path = iter.next().context("--input requires a path")?;
                args.input = Some(PathBuf::from(path));
            }
            "--limit" => {
                let limit = iter
                    .next()
                    .context("--limit requires a number")?
                    .parse::<usize>()
                    .context("--limit must be a positive integer")?;
                args.limit = Some(limit);
            }
            "--json" => args.json = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(args)
}

fn load_jsonl_tasks(path: &Path) -> Result<Vec<ExternalTask>> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut tasks = Vec::new();

    for (idx, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("failed to read line {}", idx + 1))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let task: ExternalTask = serde_json::from_str(trimmed)
            .with_context(|| format!("failed to parse JSONL line {}", idx + 1))?;
        validate_task(&task).with_context(|| format!("invalid task on line {}", idx + 1))?;
        tasks.push(task);
    }

    Ok(tasks)
}

fn validate_task(task: &ExternalTask) -> Result<()> {
    if task.id.trim().is_empty() {
        bail!("id is required");
    }
    if task.name.trim().is_empty() {
        bail!("name is required");
    }
    if task.purpose.trim().is_empty() {
        bail!("purpose is required");
    }
    if !task.signature.trim_start().starts_with("fn ") {
        bail!("signature must be a Rust function signature beginning with `fn `");
    }
    if task.examples.is_empty() {
        bail!("at least one example is required for verification");
    }
    Ok(())
}

fn built_in_tasks() -> Vec<ExternalTask> {
    vec![
        ExternalTask {
            id: "rust/add".to_string(),
            name: "add".to_string(),
            purpose: "Add two integers".to_string(),
            signature: "fn add(a: i32, b: i32) -> i32".to_string(),
            examples: vec![
                IoExample {
                    input: "add(2, 3)".to_string(),
                    output: "5".to_string(),
                },
                IoExample {
                    input: "add(-4, 9)".to_string(),
                    output: "5".to_string(),
                },
            ],
        },
        ExternalTask {
            id: "rust/is_palindrome".to_string(),
            name: "is_palindrome".to_string(),
            purpose: "Return whether the input string reads the same forward and backward"
                .to_string(),
            signature: "fn is_palindrome(text: &str) -> bool".to_string(),
            examples: vec![
                IoExample {
                    input: "is_palindrome(\"racecar\")".to_string(),
                    output: "true".to_string(),
                },
                IoExample {
                    input: "is_palindrome(\"rust\")".to_string(),
                    output: "false".to_string(),
                },
            ],
        },
        ExternalTask {
            id: "rust/remove_vowels".to_string(),
            name: "remove_vowels".to_string(),
            purpose: "Remove all ASCII vowels from the input string".to_string(),
            signature: "fn remove_vowels(text: &str) -> String".to_string(),
            examples: vec![
                IoExample {
                    input: "remove_vowels(\"hello\")".to_string(),
                    output: "\"hll\".to_string()".to_string(),
                },
                IoExample {
                    input: "remove_vowels(\"SYmthaea\")".to_string(),
                    output: "\"SYmth\".to_string()".to_string(),
                },
            ],
        },
    ]
}

fn report_task(task: &ExternalTask, result: &VerifiedCode, elapsed_ms: u128) -> TaskReport {
    TaskReport {
        id: task.id.clone(),
        name: task.name.clone(),
        compiled: result.compiled,
        tests_passed: result.tests_passed,
        guaranteed: result.is_guaranteed(),
        test_count_passed: result.test_count_passed,
        test_count_failed: result.test_count_failed,
        compile_retries: result.compile_retries,
        test_retries: result.test_retries,
        elapsed_ms,
        first_error: result
            .compile_errors
            .first()
            .or_else(|| result.test_failures.first())
            .map(|err| err.lines().next().unwrap_or(err).to_string()),
    }
}

fn print_text_report(report: &BenchReport) {
    println!("benchmark: {}", report.benchmark);
    println!("source: {}", report.source);
    println!("tasks: {}", report.task_count);
    println!(
        "pass@1: {:.3} ({}/{})",
        report.pass_at_1, report.guaranteed_count, report.task_count
    );
    println!(
        "compile_rate: {:.3} ({}/{})",
        report.compile_rate, report.compiled_count, report.task_count
    );
    println!("elapsed_ms: {}", report.elapsed_ms);
    println!();

    for task in &report.tasks {
        let status = if task.guaranteed {
            "PASS"
        } else if task.compiled {
            "COMPILE_ONLY"
        } else {
            "FAIL"
        };
        println!(
            "{} {} compiled={} tests={}/{} elapsed_ms={}",
            status,
            task.id,
            task.compiled,
            task.test_count_passed,
            task.test_count_passed + task.test_count_failed,
            task.elapsed_ms
        );
        if let Some(error) = &task.first_error {
            println!("  first_error: {error}");
        }
    }
}

fn print_help() {
    println!("benchmark_humaneval_verified");
    println!();
    println!("Options:");
    println!("  --input <path>   Rust-adapted HumanEval-style JSONL task file");
    println!("  --limit <n>      Run only the first n tasks");
    println!("  --json           Emit JSON report");
}
