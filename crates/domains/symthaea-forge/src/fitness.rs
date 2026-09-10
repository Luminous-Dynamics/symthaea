// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Layered Forge fitness oracle: cheap gates first, correctness before performance.
//!
//! Candidate failures and apparatus failures are distinct. A Cargo process that executes and exits
//! unsuccessfully is candidate evidence; inability to spawn Cargo is an apparatus failure and must
//! abort the search rather than being mislabeled as candidate rejection. Configured benchmarks are
//! likewise fail-closed and preserve spawn failures separately from candidate/evaluator failures.

use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};
use thiserror::Error;

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

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum GateExecutionError {
    #[error("failed to spawn Cargo for {gate} gate: {detail}")]
    Spawn { gate: &'static str, detail: String },
}

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub metric_name: String,
    /// Lower is better. The value is guaranteed finite by [`run_benchmark`].
    pub score: f64,
    pub duration: Duration,
}

#[derive(Debug, Error, Clone, PartialEq)]
pub enum BenchmarkError {
    #[error("failed to spawn configured benchmark: {0}")]
    Spawn(String),
    #[error("configured benchmark exited unsuccessfully: {output_tail}")]
    Failed { output_tail: String },
    #[error("configured benchmark emitted no FORGE_BENCH_RESULT marker")]
    MissingResult,
    #[error("configured benchmark emitted more than one FORGE_BENCH_RESULT marker")]
    AmbiguousResult,
    #[error("configured benchmark emitted an invalid score: {0}")]
    InvalidResult(String),
    #[error("configured benchmark emitted a non-finite score: {0}")]
    NonFiniteResult(f64),
}

impl BenchmarkError {
    /// Only inability to start the benchmark process is an apparatus failure. Once the configured
    /// process executes, non-zero exit or malformed output is evidence about this evaluation path.
    pub fn is_apparatus_failure(&self) -> bool {
        matches!(self, Self::Spawn(_))
    }
}

/// What a crate needs to tell the fitness runner in order to be evaluated.
#[derive(Debug, Clone)]
pub struct EvaluationTarget<'a> {
    pub package: &'a str,
    pub workspace_root: &'a Path,
    pub test_filter: Option<&'a str>,
    pub features: &'a [&'a str],
    pub bench_example: Option<&'a str>,
}

/// Run correctness gates in order, short-circuiting after the first candidate failure.
///
/// Process-spawn failure is returned separately so search orchestration can record `SearchAborted`
/// rather than teaching the discovery ledger that the candidate failed compilation/correctness.
pub fn run_correctness_gates(
    target: &EvaluationTarget,
) -> Result<Vec<GateResult>, GateExecutionError> {
    let mut results = Vec::new();

    let compile = run_gate(Gate::Compile, target, || {
        let mut cmd = Command::new("cargo");
        cmd.arg("check").arg("-p").arg(target.package);
        apply_features(&mut cmd, target.features);
        cmd
    })?;
    let compile_passed = compile.passed;
    results.push(compile);
    if !compile_passed {
        return Ok(results);
    }

    let test = run_gate(Gate::Test, target, || {
        let mut cmd = Command::new("cargo");
        cmd.arg("test").arg("-p").arg(target.package).arg("--lib");
        apply_features(&mut cmd, target.features);
        if let Some(filter) = target.test_filter {
            cmd.arg(filter);
        }
        cmd
    })?;
    results.push(test);

    Ok(results)
}

fn apply_features(cmd: &mut Command, features: &[&str]) {
    if !features.is_empty() {
        cmd.arg("--features").arg(features.join(","));
    }
}

fn run_gate(
    gate: Gate,
    target: &EvaluationTarget,
    build_command: impl FnOnce() -> Command,
) -> Result<GateResult, GateExecutionError> {
    let start = Instant::now();
    let mut cmd = build_command();
    cmd.current_dir(target.workspace_root);
    let output = cmd.output().map_err(|error| GateExecutionError::Spawn {
        gate: gate.label(),
        detail: error.to_string(),
    })?;
    let duration = start.elapsed();
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(GateResult {
        gate,
        passed: output.status.success(),
        output_tail: tail(&combined, 4000),
        duration,
    })
}

/// Run the configured benchmark harness.
///
/// `Ok(None)` means and only means that no benchmark was configured. A configured benchmark that
/// cannot produce one exact finite marker is `Err`; callers can inspect [`BenchmarkError::Spawn`]
/// to distinguish apparatus failure from an executed-but-invalid evaluation.
pub fn run_benchmark(
    target: &EvaluationTarget,
) -> Result<Option<BenchmarkResult>, BenchmarkError> {
    let Some(example) = target.bench_example else {
        return Ok(None);
    };

    let start = Instant::now();
    let mut cmd = Command::new("cargo");
    cmd.arg("run")
        .arg("--release")
        .arg("-p")
        .arg(target.package)
        .arg("--example")
        .arg(example);
    apply_features(&mut cmd, target.features);
    cmd.current_dir(target.workspace_root);
    let output = cmd
        .output()
        .map_err(|error| BenchmarkError::Spawn(error.to_string()))?;
    let duration = start.elapsed();

    if !output.status.success() {
        let combined = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return Err(BenchmarkError::Failed {
            output_tail: tail(&combined, 4000),
        });
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let score = parse_bench_result(&stdout)?;
    Ok(Some(BenchmarkResult {
        metric_name: "median_ns".to_string(),
        score,
        duration,
    }))
}

fn parse_bench_result(stdout: &str) -> Result<f64, BenchmarkError> {
    let mut markers = stdout.lines().filter_map(|line| {
        line.trim()
            .strip_prefix("FORGE_BENCH_RESULT:")
            .map(str::trim)
    });
    let raw = markers.next().ok_or(BenchmarkError::MissingResult)?;
    if markers.next().is_some() {
        return Err(BenchmarkError::AmbiguousResult);
    }
    let value = raw
        .parse::<f64>()
        .map_err(|_| BenchmarkError::InvalidResult(raw.to_string()))?;
    if !value.is_finite() {
        return Err(BenchmarkError::NonFiniteResult(value));
    }
    Ok(if value == 0.0 { 0.0 } else { value })
}

fn tail(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        return s.to_string();
    }
    let mut start = s.len().saturating_sub(max_len);
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    format!("...(truncated)...\n{}", &s[start..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bench_result_accepts_one_finite_marker() {
        let stdout = "warming up\nsome noise\nFORGE_BENCH_RESULT: 1234.5\ntrailing\n";
        assert_eq!(parse_bench_result(stdout).unwrap(), 1234.5);
    }

    #[test]
    fn parse_bench_result_rejects_missing_marker() {
        assert_eq!(
            parse_bench_result("nothing relevant here\n").unwrap_err(),
            BenchmarkError::MissingResult
        );
    }

    #[test]
    fn parse_bench_result_rejects_ambiguous_markers() {
        let output = "FORGE_BENCH_RESULT: 1\nFORGE_BENCH_RESULT: 2\n";
        assert_eq!(
            parse_bench_result(output).unwrap_err(),
            BenchmarkError::AmbiguousResult
        );
    }

    #[test]
    fn parse_bench_result_rejects_non_finite_values() {
        assert!(matches!(
            parse_bench_result("FORGE_BENCH_RESULT: NaN\n"),
            Err(BenchmarkError::NonFiniteResult(value)) if value.is_nan()
        ));
        assert_eq!(
            parse_bench_result("FORGE_BENCH_RESULT: inf\n").unwrap_err(),
            BenchmarkError::NonFiniteResult(f64::INFINITY)
        );
    }

    #[test]
    fn benchmark_error_classifies_only_spawn_as_apparatus_failure() {
        assert!(BenchmarkError::Spawn("missing cargo".into()).is_apparatus_failure());
        assert!(!BenchmarkError::MissingResult.is_apparatus_failure());
    }

    #[test]
    fn tail_truncates_utf8_without_panicking() {
        let long = format!("{}END_MARKER", "λ".repeat(10_000));
        let truncated = tail(&long, 101);
        assert!(truncated.ends_with("END_MARKER"));
        assert!(truncated.len() < long.len());
    }

    #[test]
    fn tail_leaves_short_output_untouched() {
        assert_eq!(tail("short", 100), "short");
    }
}
