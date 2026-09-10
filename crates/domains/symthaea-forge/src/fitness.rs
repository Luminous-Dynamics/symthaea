// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Layered Forge fitness oracle: cheap gates first, correctness before performance.
//!
//! Correctness gates are strict pass/fail. A configured benchmark is also fail-closed: inability
//! to spawn it, non-zero exit, missing/ambiguous result markers, or a non-finite score are
//! evaluation failures, never aliases for "no benchmark configured".

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

/// What a crate needs to tell the fitness runner in order to be evaluated.
#[derive(Debug, Clone)]
pub struct EvaluationTarget<'a> {
    /// Cargo package name to check/test (`-p <name>`).
    pub package: &'a str,
    /// Working directory to run Cargo from (the workspace root).
    pub workspace_root: &'a Path,
    /// Optional test filter so every candidate need not run an entire crate suite.
    pub test_filter: Option<&'a str>,
    /// Cargo features to enable, if any.
    pub features: &'a [&'a str],
    /// `cargo run --release --example <name>` benchmark harness; `None` is explicitly
    /// correctness-only search mode.
    pub bench_example: Option<&'a str>,
}

/// Run correctness gates in order, short-circuiting after the first failure.
pub fn run_correctness_gates(target: &EvaluationTarget) -> Vec<GateResult> {
    let mut results = Vec::new();

    let compile = run_gate(Gate::Compile, target, || {
        let mut cmd = Command::new("cargo");
        cmd.arg("check").arg("-p").arg(target.package);
        apply_features(&mut cmd, target.features);
        cmd
    });
    let compile_passed = compile.passed;
    results.push(compile);
    if !compile_passed {
        return results;
    }

    let test = run_gate(Gate::Test, target, || {
        let mut cmd = Command::new("cargo");
        cmd.arg("test").arg("-p").arg(target.package).arg("--lib");
        apply_features(&mut cmd, target.features);
        if let Some(filter) = target.test_filter {
            cmd.arg(filter);
        }
        cmd
    });
    results.push(test);

    results
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
        Err(error) => GateResult {
            gate,
            passed: false,
            output_tail: format!("failed to spawn cargo: {error}"),
            duration,
        },
    }
}

/// Run the configured benchmark harness.
///
/// `Ok(None)` means and only means that no benchmark was configured. A configured benchmark that
/// cannot produce one exact finite marker is `Err`, so search logic cannot accidentally treat a
/// benchmark failure as correctness-only mode.
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

    // Avoid slicing in the middle of a UTF-8 code point while retaining the requested tail size
    // approximately. Gate output is evidence/report text and must never panic because it contains
    // non-ASCII diagnostics.
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