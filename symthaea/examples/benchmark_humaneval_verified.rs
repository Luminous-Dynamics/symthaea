// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Verified coding testbench for HumanEval-style Rust tasks.
//!
//! Uses CodeOrchestrator to perform verified generation and automatically
//! capture distillation data for successful solutions.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use symthaea::language::code_orchestrator::CodeOrchestrator;
use symthaea_core::synthesis_trait::SynthesisRequest;

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
    verified: bool,
    similarity: f32,
    compile_retries: usize,
    elapsed_ms: u128,
    narrative: Option<String>,
}

#[derive(Debug, Serialize)]
struct BenchReport {
    benchmark: String,
    task_count: usize,
    verified_count: usize,
    pass_at_1: f64,
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

#[tokio::main]
async fn main() -> Result<()> {
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
    let mut orchestrator = CodeOrchestrator::new();

    // Skip heavy repo indexing for simple HumanEval tasks to avoid hangs.
    // Full repo indexing is reserved for SWE-bench style 'Solve' intents.
    /*
    if Path::new("Cargo.toml").exists() {
        orchestrator.index_project(".")?;
    }
    */

    let mut task_reports = Vec::with_capacity(tasks.len());
    for (idx, task) in tasks.iter().enumerate() {
        let task_start = Instant::now();
        println!("[{:3}/{}] Processing: {}", idx + 1, tasks.len(), task.id);

        // Convert ExternalTask to SynthesisRequest
        let mut request = SynthesisRequest::new("rust", &task.name, &task.purpose);
        if !task.signature.is_empty() {
            request = request.with_signature(&task.signature);
        }
        for example in &task.examples {
            request = request.with_example(&example.input, &example.output);
        }

        // Use the Orchestrator (which handles retries and distillation)
        let response = orchestrator.synthesize(&request);

        // Map response back to TaskReport
        let history = orchestrator.attempt_history();
        let retries = history.len().saturating_sub(1);

        task_reports.push(TaskReport {
            id: task.id.clone(),
            name: task.name.clone(),
            verified: response.accepted,
            similarity: response.confidence,
            compile_retries: retries,
            elapsed_ms: task_start.elapsed().as_millis(),
            narrative: response.narrative,
        });
    }

    let verified_count = task_reports.iter().filter(|task| task.verified).count();
    let report = BenchReport {
        benchmark: "benchmark_humaneval_verified".to_string(),
        task_count: task_reports.len(),
        verified_count,
        pass_at_1: verified_count as f64 / task_reports.len() as f64,
        elapsed_ms: start.elapsed().as_millis(),
        source,
        tasks: task_reports,
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_text_report(&report);
    }

    // EXPORT DISTILLATION DATA
    let distillation_path = Path::new("data/benchmarks/humaneval/distillation_baseline.jsonl");
    if let Some(parent) = distillation_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    orchestrator.save_distillation(distillation_path)?;
    println!(
        "\n✅ Distillation data ({} entries) saved to: {}",
        orchestrator.distillation_buffer().len(),
        distillation_path.display()
    );

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
    Ok(())
}

fn built_in_tasks() -> Vec<ExternalTask> {
    vec![ExternalTask {
        id: "rust/add".to_string(),
        name: "add".to_string(),
        purpose: "Add two integers".to_string(),
        signature: "fn add(a: i32, b: i32) -> i32".to_string(),
        examples: vec![IoExample {
            input: "add(2, 3)".to_string(),
            output: "5".to_string(),
        }],
    }]
}

fn print_text_report(report: &BenchReport) {
    println!("benchmark: {}", report.benchmark);
    println!("source: {}", report.source);
    println!("tasks: {}", report.task_count);
    println!(
        "pass@1: {:.3} ({}/{})",
        report.pass_at_1, report.verified_count, report.task_count
    );
    println!("elapsed_ms: {}", report.elapsed_ms);
    println!();

    for task in &report.tasks {
        let status = if task.verified { "PASS" } else { "FAIL" };
        println!(
            "{} {} sim={:.3} retries={} elapsed_ms={}",
            status, task.id, task.similarity, task.compile_retries, task.elapsed_ms
        );
        if let Some(narrative) = &task.narrative {
            println!("  {}", narrative);
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
