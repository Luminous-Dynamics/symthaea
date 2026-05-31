// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! broca-checkpoint-compare: compare two Broca measurement artifact directories.
//!
//! This binary treats benchmark JSON as evidence. Missing metrics are reported
//! as unavailable rather than silently interpreted as success.

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
struct Options {
    baseline_dir: PathBuf,
    candidate_dir: PathBuf,
    json_out: Option<PathBuf>,
    fail_on_regression: bool,
    max_drift_regression: f64,
    max_hallucination_regression: f64,
    max_compile_rate_regression: f64,
    max_test_rate_regression: f64,
    max_structured_confidence_regression: f64,
    max_structured_validity_regression: f64,
    max_structured_required_role_rate_regression: f64,
    require_all_metrics: bool,
}

#[derive(Debug, Serialize)]
struct CompareReport {
    schema_version: u32,
    evidence_level: &'static str,
    baseline_dir: String,
    candidate_dir: String,
    promotion: PromotionDecision,
    metrics: Vec<MetricComparison>,
    missing_metrics: Vec<String>,
}

#[derive(Debug, Serialize)]
struct PromotionDecision {
    passed: bool,
    failures: Vec<PromotionFailure>,
    missing_required_metrics: Vec<String>,
}

#[derive(Debug, Serialize)]
struct PromotionFailure {
    metric: String,
    baseline: f64,
    candidate: f64,
    delta: f64,
    allowed_regression: f64,
}

#[derive(Debug, Serialize)]
struct MetricComparison {
    metric: String,
    direction: MetricDirection,
    baseline: f64,
    candidate: f64,
    delta: f64,
    allowed_regression: f64,
    passed: bool,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum MetricDirection {
    LowerIsBetter,
    HigherIsBetter,
}

fn main() -> Result<()> {
    let opts = parse_args()?;
    let report = compare(&opts)?;
    let passed = report.promotion.passed;
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = &opts.json_out {
        std::fs::write(path, json)?;
    } else {
        println!("{json}");
    }
    if opts.fail_on_regression && !passed {
        anyhow::bail!("candidate measurement artifacts failed promotion gate");
    }
    Ok(())
}

fn compare(opts: &Options) -> Result<CompareReport> {
    let baseline_decoder = read_optional_json(&opts.baseline_dir.join("decoder-ab.json"))?;
    let candidate_decoder = read_optional_json(&opts.candidate_dir.join("decoder-ab.json"))?;
    let baseline_exercism = read_optional_json(&opts.baseline_dir.join("exercism-bench.json"))?;
    let candidate_exercism = read_optional_json(&opts.candidate_dir.join("exercism-bench.json"))?;

    let mut metrics = Vec::new();
    let mut missing_metrics = Vec::new();

    compare_metric(
        &mut metrics,
        &mut missing_metrics,
        "decoder.avg_direct_semantic_drift",
        MetricDirection::LowerIsBetter,
        value_at(
            baseline_decoder.as_ref(),
            &["aggregate", "avg_direct_semantic_drift"],
        ),
        value_at(
            candidate_decoder.as_ref(),
            &["aggregate", "avg_direct_semantic_drift"],
        ),
        opts.max_drift_regression,
    );
    compare_metric(
        &mut metrics,
        &mut missing_metrics,
        "decoder.avg_mamba_semantic_drift",
        MetricDirection::LowerIsBetter,
        value_at(
            baseline_decoder.as_ref(),
            &["aggregate", "avg_mamba_semantic_drift"],
        ),
        value_at(
            candidate_decoder.as_ref(),
            &["aggregate", "avg_mamba_semantic_drift"],
        ),
        opts.max_drift_regression,
    );
    compare_metric(
        &mut metrics,
        &mut missing_metrics,
        "decoder.direct_hallucination_rate",
        MetricDirection::LowerIsBetter,
        value_at(
            baseline_decoder.as_ref(),
            &["aggregate", "direct_hallucination_rate"],
        ),
        value_at(
            candidate_decoder.as_ref(),
            &["aggregate", "direct_hallucination_rate"],
        ),
        opts.max_hallucination_regression,
    );
    compare_metric(
        &mut metrics,
        &mut missing_metrics,
        "decoder.mamba_hallucination_rate",
        MetricDirection::LowerIsBetter,
        value_at(
            baseline_decoder.as_ref(),
            &["aggregate", "mamba_hallucination_rate"],
        ),
        value_at(
            candidate_decoder.as_ref(),
            &["aggregate", "mamba_hallucination_rate"],
        ),
        opts.max_hallucination_regression,
    );
    compare_metric(
        &mut metrics,
        &mut missing_metrics,
        "decoder.avg_structured_confidence",
        MetricDirection::HigherIsBetter,
        value_at(
            baseline_decoder.as_ref(),
            &["aggregate", "avg_structured_confidence"],
        ),
        value_at(
            candidate_decoder.as_ref(),
            &["aggregate", "avg_structured_confidence"],
        ),
        opts.max_structured_confidence_regression,
    );
    compare_metric(
        &mut metrics,
        &mut missing_metrics,
        "decoder.avg_structured_validity",
        MetricDirection::HigherIsBetter,
        value_at(
            baseline_decoder.as_ref(),
            &["aggregate", "avg_structured_validity"],
        ),
        value_at(
            candidate_decoder.as_ref(),
            &["aggregate", "avg_structured_validity"],
        ),
        opts.max_structured_validity_regression,
    );
    compare_metric(
        &mut metrics,
        &mut missing_metrics,
        "decoder.structured_required_role_rate",
        MetricDirection::HigherIsBetter,
        value_at(
            baseline_decoder.as_ref(),
            &["aggregate", "structured_required_role_rate"],
        ),
        value_at(
            candidate_decoder.as_ref(),
            &["aggregate", "structured_required_role_rate"],
        ),
        opts.max_structured_required_role_rate_regression,
    );
    compare_metric(
        &mut metrics,
        &mut missing_metrics,
        "exercism.compile_success_rate",
        MetricDirection::HigherIsBetter,
        success_rate(
            value_at(baseline_exercism.as_ref(), &["compile_successes"]),
            value_at(baseline_exercism.as_ref(), &["total_exercises"]),
        ),
        success_rate(
            value_at(candidate_exercism.as_ref(), &["compile_successes"]),
            value_at(candidate_exercism.as_ref(), &["total_exercises"]),
        ),
        opts.max_compile_rate_regression,
    );
    compare_metric(
        &mut metrics,
        &mut missing_metrics,
        "exercism.test_success_rate",
        MetricDirection::HigherIsBetter,
        success_rate(
            value_at(baseline_exercism.as_ref(), &["test_successes"]),
            value_at(baseline_exercism.as_ref(), &["total_exercises"]),
        ),
        success_rate(
            value_at(candidate_exercism.as_ref(), &["test_successes"]),
            value_at(candidate_exercism.as_ref(), &["total_exercises"]),
        ),
        opts.max_test_rate_regression,
    );

    let failures = metrics
        .iter()
        .filter(|metric| !metric.passed)
        .map(|metric| PromotionFailure {
            metric: metric.metric.clone(),
            baseline: metric.baseline,
            candidate: metric.candidate,
            delta: metric.delta,
            allowed_regression: metric.allowed_regression,
        })
        .collect::<Vec<_>>();

    let missing_required_metrics = if opts.require_all_metrics {
        missing_metrics.clone()
    } else {
        Vec::new()
    };

    Ok(CompareReport {
        schema_version: 1,
        evidence_level: "measured",
        baseline_dir: opts.baseline_dir.display().to_string(),
        candidate_dir: opts.candidate_dir.display().to_string(),
        promotion: PromotionDecision {
            passed: failures.is_empty() && missing_required_metrics.is_empty(),
            failures,
            missing_required_metrics,
        },
        metrics,
        missing_metrics,
    })
}

fn compare_metric(
    metrics: &mut Vec<MetricComparison>,
    missing_metrics: &mut Vec<String>,
    name: &str,
    direction: MetricDirection,
    baseline: Option<f64>,
    candidate: Option<f64>,
    allowed_regression: f64,
) {
    let (Some(baseline), Some(candidate)) = (baseline, candidate) else {
        missing_metrics.push(name.to_string());
        return;
    };
    let delta = candidate - baseline;
    let passed = match direction {
        MetricDirection::LowerIsBetter => delta <= allowed_regression,
        MetricDirection::HigherIsBetter => delta >= -allowed_regression,
    };
    metrics.push(MetricComparison {
        metric: name.to_string(),
        direction,
        baseline,
        candidate,
        delta,
        allowed_regression,
        passed,
    });
}

fn read_optional_json(path: &Path) -> Result<Option<Value>> {
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading artifact {}", path.display()))?;
    serde_json::from_str(&text)
        .with_context(|| format!("parsing artifact {}", path.display()))
        .map(Some)
}

fn value_at(root: Option<&Value>, path: &[&str]) -> Option<f64> {
    let mut current = root?;
    for part in path {
        current = current.get(*part)?;
    }
    current.as_f64()
}

fn success_rate(successes: Option<f64>, total: Option<f64>) -> Option<f64> {
    let total = total?;
    if total <= 0.0 {
        return None;
    }
    Some(successes? / total)
}

fn parse_args() -> Result<Options> {
    let mut opts = Options {
        baseline_dir: PathBuf::new(),
        candidate_dir: PathBuf::new(),
        json_out: None,
        fail_on_regression: false,
        max_drift_regression: 0.05,
        max_hallucination_regression: 0.05,
        max_compile_rate_regression: 0.0,
        max_test_rate_regression: 0.0,
        max_structured_confidence_regression: 0.05,
        max_structured_validity_regression: 0.05,
        max_structured_required_role_rate_regression: 0.0,
        require_all_metrics: false,
    };

    let args = std::env::args().collect::<Vec<_>>();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--baseline-dir" => {
                i += 1;
                opts.baseline_dir = PathBuf::from(value(&args, i, "--baseline-dir")?);
            }
            "--candidate-dir" => {
                i += 1;
                opts.candidate_dir = PathBuf::from(value(&args, i, "--candidate-dir")?);
            }
            "--json-out" => {
                i += 1;
                opts.json_out = Some(PathBuf::from(value(&args, i, "--json-out")?));
            }
            "--fail-on-regression" => opts.fail_on_regression = true,
            "--require-all-metrics" => opts.require_all_metrics = true,
            "--max-drift-regression" => {
                i += 1;
                opts.max_drift_regression = value(&args, i, "--max-drift-regression")?.parse()?;
            }
            "--max-hallucination-regression" => {
                i += 1;
                opts.max_hallucination_regression =
                    value(&args, i, "--max-hallucination-regression")?.parse()?;
            }
            "--max-compile-rate-regression" => {
                i += 1;
                opts.max_compile_rate_regression =
                    value(&args, i, "--max-compile-rate-regression")?.parse()?;
            }
            "--max-test-rate-regression" => {
                i += 1;
                opts.max_test_rate_regression =
                    value(&args, i, "--max-test-rate-regression")?.parse()?;
            }
            "--max-structured-confidence-regression" => {
                i += 1;
                opts.max_structured_confidence_regression =
                    value(&args, i, "--max-structured-confidence-regression")?.parse()?;
            }
            "--max-structured-validity-regression" => {
                i += 1;
                opts.max_structured_validity_regression =
                    value(&args, i, "--max-structured-validity-regression")?.parse()?;
            }
            "--max-structured-required-role-rate-regression" => {
                i += 1;
                opts.max_structured_required_role_rate_regression =
                    value(&args, i, "--max-structured-required-role-rate-regression")?.parse()?;
            }
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument {other}"),
        }
        i += 1;
    }

    if opts.baseline_dir.as_os_str().is_empty() || opts.candidate_dir.as_os_str().is_empty() {
        anyhow::bail!("--baseline-dir and --candidate-dir are required");
    }
    Ok(opts)
}

fn value<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str> {
    args.get(index)
        .map(String::as_str)
        .with_context(|| format!("{flag} requires a value"))
}

fn print_usage() {
    eprintln!(
        "Usage: broca-checkpoint-compare --baseline-dir DIR --candidate-dir DIR [--json-out PATH] [--fail-on-regression] [--require-all-metrics] [--max-drift-regression F] [--max-hallucination-regression F] [--max-compile-rate-regression F] [--max-test-rate-regression F] [--max-structured-confidence-regression F] [--max-structured-validity-regression F] [--max-structured-required-role-rate-regression F]"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn opts(baseline_dir: PathBuf, candidate_dir: PathBuf) -> Options {
        Options {
            baseline_dir,
            candidate_dir,
            json_out: None,
            fail_on_regression: false,
            max_drift_regression: 0.05,
            max_hallucination_regression: 0.05,
            max_compile_rate_regression: 0.0,
            max_test_rate_regression: 0.0,
            max_structured_confidence_regression: 0.05,
            max_structured_validity_regression: 0.05,
            max_structured_required_role_rate_regression: 0.0,
            require_all_metrics: false,
        }
    }

    fn write_artifacts(
        dir: &Path,
        drift: f64,
        hallucination: f64,
        confidence: f64,
        validity: f64,
        required_role_rate: f64,
        compile_successes: u64,
        test_successes: u64,
    ) {
        std::fs::create_dir_all(dir).unwrap();
        let decoder = json!({
            "aggregate": {
                "avg_direct_semantic_drift": drift,
                "direct_hallucination_rate": hallucination,
                "avg_structured_confidence": confidence,
                "avg_structured_validity": validity,
                "structured_required_role_rate": required_role_rate
            }
        });
        let exercism = json!({
            "total_exercises": 4,
            "compile_successes": compile_successes,
            "test_successes": test_successes
        });
        std::fs::write(
            dir.join("decoder-ab.json"),
            serde_json::to_string_pretty(&decoder).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("exercism-bench.json"),
            serde_json::to_string_pretty(&exercism).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn fixture_comparison_passes_within_thresholds() {
        let baseline = PathBuf::from("tests/fixtures/measurement-baseline-v1")
            .canonicalize()
            .unwrap();
        let candidate = PathBuf::from("tests/fixtures/measurement-candidate-v1")
            .canonicalize()
            .unwrap();
        let report = compare(&opts(baseline, candidate)).unwrap();
        assert!(report.promotion.passed);
        assert!(report.promotion.failures.is_empty());
    }

    #[test]
    fn drift_regression_fails_beyond_threshold() {
        let temp = tempfile::tempdir().unwrap();
        let baseline = temp.path().join("baseline");
        let candidate = temp.path().join("candidate");
        write_artifacts(&baseline, 0.4, 0.2, 0.8, 0.9, 1.0, 2, 1);
        write_artifacts(&candidate, 0.6, 0.2, 0.8, 0.9, 1.0, 2, 1);

        let report = compare(&opts(baseline, candidate)).unwrap();
        assert!(!report.promotion.passed);
        assert!(
            report
                .promotion
                .failures
                .iter()
                .any(|failure| failure.metric == "decoder.avg_direct_semantic_drift")
        );
    }

    #[test]
    fn compile_rate_regression_fails() {
        let temp = tempfile::tempdir().unwrap();
        let baseline = temp.path().join("baseline");
        let candidate = temp.path().join("candidate");
        write_artifacts(&baseline, 0.4, 0.2, 0.8, 0.9, 1.0, 3, 1);
        write_artifacts(&candidate, 0.4, 0.2, 0.8, 0.9, 1.0, 2, 1);

        let report = compare(&opts(baseline, candidate)).unwrap();
        assert!(!report.promotion.passed);
        assert!(
            report
                .promotion
                .failures
                .iter()
                .any(|failure| failure.metric == "exercism.compile_success_rate")
        );
    }

    #[test]
    fn structured_validity_regression_fails() {
        let temp = tempfile::tempdir().unwrap();
        let baseline = temp.path().join("baseline");
        let candidate = temp.path().join("candidate");
        write_artifacts(&baseline, 0.4, 0.2, 0.8, 0.9, 1.0, 2, 1);
        write_artifacts(&candidate, 0.4, 0.2, 0.8, 0.8, 1.0, 2, 1);

        let report = compare(&opts(baseline, candidate)).unwrap();
        assert!(!report.promotion.passed);
        assert!(
            report
                .promotion
                .failures
                .iter()
                .any(|failure| failure.metric == "decoder.avg_structured_validity")
        );
    }

    #[test]
    fn missing_metrics_fail_only_when_required() {
        let temp = tempfile::tempdir().unwrap();
        let baseline = temp.path().join("baseline");
        let candidate = temp.path().join("candidate");
        write_artifacts(&baseline, 0.4, 0.2, 0.8, 0.9, 1.0, 2, 1);
        write_artifacts(&candidate, 0.4, 0.2, 0.8, 0.9, 1.0, 2, 1);

        let report = compare(&opts(baseline.clone(), candidate.clone())).unwrap();
        assert!(report.promotion.passed);
        assert!(!report.missing_metrics.is_empty());

        let mut strict = opts(baseline, candidate);
        strict.require_all_metrics = true;
        let report = compare(&strict).unwrap();
        assert!(!report.promotion.passed);
        assert!(!report.promotion.missing_required_metrics.is_empty());
    }
}
