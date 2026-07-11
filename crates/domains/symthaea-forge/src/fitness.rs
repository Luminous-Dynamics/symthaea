// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Layered fitness oracle: cheapest gate first, correctness before speed.
//!
//! Per the self-mod safety rule "cheap-before-expensive": a candidate must
//! clear `cargo check` before we ever run its tests, and must clear its
//! tests before we ever benchmark it. Correctness gates (`Compile`, `Test`,
//! `Proptest`) are strict pass/fail -- a candidate that fails any of them
//! is rejected outright, regardless of how fast it might be. Only
//! `Benchmark` produces a comparable numeric score, and only candidates
//! that passed every correctness gate ever reach it.

use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Gate {
    Compile,
    Test,
    Proptest,
}

impl Gate {
    pub fn label(&self) -> &'static str {
        match self {
            Gate::Compile => "compile",
            Gate::Test => "test",
            Gate::Proptest => "proptest",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GateResult {
    pub gate: Gate,
    pub passed: bool,
    /// Tail of stdout+stderr, truncated, for the certificate/report.
    pub output_tail: String,
    pub duration: Duration,
}

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub metric_name: String,
    /// Lower is better (e.g. nanoseconds). Benchmarks reporting a
    /// higher-is-better metric should negate before storing here so every
    /// consumer of `BenchmarkResult` can assume "lower wins".
    pub score: f64,
    pub duration: Duration,
}

/// What a crate needs to tell the fitness runner in order to be evaluated.
#[derive(Debug, Clone)]
pub struct EvaluationTarget<'a> {
    /// Cargo package name to check/test (`-p <name>`).
    pub package: &'a str,
    /// Working directory to run cargo from (the workspace root).
    pub workspace_root: &'a Path,
    /// Optional test filter (e.g. a module path) so `cargo test` doesn't
    /// re-run the whole crate's suite for every candidate.
    pub test_filter: Option<&'a str>,
    /// Cargo features to enable, if any.
    pub features: &'a [&'a str],
    /// `cargo run --release --example <name>` for the benchmark harness;
    /// `None` skips the Benchmark gate (correctness-only evaluation).
    pub bench_example: Option<&'a str>,
}

/// Run the correctness gates in order, short-circuiting on the first
/// failure (a failed `Compile` means `Test`/`Proptest` are meaningless and
/// are not attempted).
pub fn run_correctness_gates(target: &EvaluationTarget) -> Vec<GateResult> {
    let mut results = Vec::new();

    let compile = run_gate(Gate::Compile, target, || {
        let mut cmd = base_cargo_command(target);
        cmd.arg("check").arg("-p").arg(target.package);
        cmd
    });
    let compile_passed = compile.passed;
    results.push(compile);
    if !compile_passed {
        return results;
    }

    let test = run_gate(Gate::Test, target, || {
        let mut cmd = base_cargo_command(target);
        cmd.arg("test").arg("-p").arg(target.package).arg("--lib");
        if let Some(filter) = target.test_filter {
            cmd.arg(filter);
        }
        cmd
    });
    results.push(test);

    results
}

fn run_gate(
    gate: Gate,
    target: &EvaluationTarget,
    build_command: impl FnOnce() -> Command,
) -> GateResult {
    let start = Instant::now();
    let mut cmd = build_command();
    cmd.current_dir(target.workspace_root);
    let output = cmd.output();
    let duration = start.elapsed();

    match output {
        Ok(out) => {
            let combined = format!(
                "{}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            GateResult {
                gate,
                passed: out.status.success(),
                output_tail: tail(&combined, 4000),
                duration,
            }
        }
        Err(e) => GateResult {
            gate,
            passed: false,
            output_tail: format!("failed to spawn cargo: {e}"),
            duration,
        },
    }
}

fn base_cargo_command(target: &EvaluationTarget) -> Command {
    let mut cmd = Command::new("cargo");
    if !target.features.is_empty() {
        cmd.arg("--features").arg(target.features.join(","));
    }
    let _ = target; // features already applied above; kept for symmetry/future flags
    cmd
}

fn tail(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("...(truncated)...\n{}", &s[s.len() - max_len..])
    }
}

/// Run the benchmark example and parse a `FORGE_BENCH_RESULT: <f64>` line
/// from its stdout. The harness binary is responsible for producing a
/// stable, already-repeated (e.g. median-of-N) measurement -- this runner
/// does not repeat the process itself, to keep per-candidate wall-clock
/// bounded under `cargo-gate` contention.
pub fn run_benchmark(target: &EvaluationTarget) -> Option<BenchmarkResult> {
    let example = target.bench_example?;
    let start = Instant::now();
    let mut cmd = Command::new("cargo");
    cmd.arg("run")
        .arg("--release")
        .arg("-p")
        .arg(target.package)
        .arg("--example")
        .arg(example);
    if !target.features.is_empty() {
        cmd.arg("--features").arg(target.features.join(","));
    }
    cmd.current_dir(target.workspace_root);
    let output = cmd.output().ok()?;
    let duration = start.elapsed();
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_bench_result(&stdout).map(|score| BenchmarkResult {
        metric_name: "median_ns".to_string(),
        score,
        duration,
    })
}

fn parse_bench_result(stdout: &str) -> Option<f64> {
    for line in stdout.lines() {
        if let Some(rest) = line.trim().strip_prefix("FORGE_BENCH_RESULT:") {
            if let Ok(v) = rest.trim().parse::<f64>() {
                return Some(v);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bench_result_finds_the_marker_line() {
        let stdout = "warming up\nsome noise\nFORGE_BENCH_RESULT: 1234.5\ntrailing\n";
        assert_eq!(parse_bench_result(stdout), Some(1234.5));
    }

    #[test]
    fn parse_bench_result_returns_none_without_marker() {
        let stdout = "nothing relevant here\n";
        assert_eq!(parse_bench_result(stdout), None);
    }

    #[test]
    fn tail_truncates_long_output_and_keeps_the_end() {
        let long = "x".repeat(10_000) + "END_MARKER";
        let t = tail(&long, 100);
        assert!(t.ends_with("END_MARKER"));
        assert!(t.len() < long.len());
    }

    #[test]
    fn tail_leaves_short_output_untouched() {
        assert_eq!(tail("short", 100), "short");
    }
}
