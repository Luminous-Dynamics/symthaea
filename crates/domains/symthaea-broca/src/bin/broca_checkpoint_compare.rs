// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! broca-checkpoint-compare: compare two Broca measurement artifact directories.
//!
//! This binary treats benchmark JSON as evidence. Missing metrics are reported
//! as unavailable rather than silently interpreted as success.
//!
//! ## Two gates, not one
//!
//! `promotion.regression_passed` proves the candidate didn't get *worse* on
//! drift/hallucination/compile/test/structured metrics — that's the whole of
//! what this binary used to check. It structurally cannot tell you whether
//! the candidate got *better*: a checkpoint that memorized nothing new still
//! clears every regression bound, because those bounds only look at drift
//! from the *previous* behavior, not at performance on genuinely new
//! material (see `fuzzy-beaming-brook.md` incident notes and
//! `DISCOVERY_AND_SELF_IMPROVEMENT_PLAN_2026-07-06.md` Tier 1.1).
//!
//! `promotion.improvement_passed` closes that gap: when
//! `--baseline-topic-coverage`/`--candidate-topic-coverage` (both produced
//! by `broca-topic-coverage` against the SAME held-out-objective batch — one
//! run against the production checkpoint, one against the candidate) are
//! supplied, the candidate must beat production by a configured positive
//! margin on those held-out, never-trained-on topics. `promotion.passed` is
//! `regression_passed && improvement_passed` — a candidate that merely
//! didn't regress is not enough to promote.

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
    max_structured_translation_validity_regression: f64,
    max_structured_translation_grounding_rate_regression: f64,
    max_structured_translation_drift_regression: f64,
    require_all_metrics: bool,
    // ── Presence-of-improvement gate (Tier 1.1) ────────────────────────────
    /// `broca-topic-coverage` report generated against the PRODUCTION
    /// checkpoint, evaluated on this cycle's held-out objectives.
    baseline_topic_coverage: Option<PathBuf>,
    /// `broca-topic-coverage` report generated against the CANDIDATE
    /// checkpoint, evaluated on the same held-out objectives.
    candidate_topic_coverage: Option<PathBuf>,
    /// Candidate's `avg_keyword_overlap` must exceed baseline's by at least
    /// this much on held-out objectives.
    min_keyword_overlap_improvement: f64,
    /// Candidate's `avg_generated_coherence` must exceed baseline's by at
    /// least this much on held-out objectives (metric skipped, not failed,
    /// when coherence wasn't measured on either side).
    min_coherence_improvement: f64,
    /// Fail closed if topic-coverage evidence was expected (this flag set)
    /// but couldn't be loaded/parsed on either side. Without this flag, a
    /// missing/unreadable topic-coverage report makes the improvement gate
    /// vacuously pass — the correct default for callers that never intended
    /// to evaluate it (e.g. cycles with no new held-out material), but wrong
    /// for callers that did.
    require_improvement_evaluated: bool,
    /// Fail the improvement gate if any individual improvement metric (e.g.
    /// coherence) couldn't be computed, instead of silently skipping it.
    require_all_improvement_metrics: bool,
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
    improvement_metrics: Vec<ImprovementMetricComparison>,
    missing_improvement_metrics: Vec<String>,
}

#[derive(Debug, Serialize)]
struct PromotionDecision {
    /// Combined gate: `regression_passed && improvement_passed`. This is
    /// what callers should check to decide promotion eligibility.
    passed: bool,
    /// Absence-of-regression verdict only (the historical `passed` field's
    /// exact semantics, kept as its own field for callers that only care
    /// about drift/hallucination/compile/test bounds).
    regression_passed: bool,
    /// Presence-of-improvement verdict on held-out objectives. `true`
    /// vacuously when the improvement gate wasn't evaluated (no
    /// topic-coverage evidence supplied) AND it wasn't required to be.
    improvement_passed: bool,
    /// Whether the improvement gate actually had evidence to evaluate
    /// (i.e. both `--baseline-topic-coverage` and `--candidate-topic-coverage`
    /// were supplied and parsed successfully).
    improvement_evaluated: bool,
    failures: Vec<PromotionFailure>,
    missing_required_metrics: Vec<String>,
    improvement_failures: Vec<ImprovementFailure>,
    missing_required_improvement_metrics: Vec<String>,
}

#[derive(Debug, Serialize)]
struct PromotionFailure {
    metric: String,
    baseline: f64,
    candidate: f64,
    delta: f64,
    allowed_regression: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ImprovementMetricComparison {
    metric: String,
    baseline: f64,
    candidate: f64,
    delta: f64,
    required_improvement: f64,
    passed: bool,
}

#[derive(Debug, Serialize)]
struct ImprovementFailure {
    metric: String,
    baseline: f64,
    candidate: f64,
    delta: f64,
    required_improvement: f64,
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
        "decoder.avg_structured_translation_validity",
        MetricDirection::HigherIsBetter,
        value_at(
            baseline_decoder.as_ref(),
            &["aggregate", "avg_structured_translation_validity"],
        ),
        value_at(
            candidate_decoder.as_ref(),
            &["aggregate", "avg_structured_translation_validity"],
        ),
        opts.max_structured_translation_validity_regression,
    );
    compare_metric(
        &mut metrics,
        &mut missing_metrics,
        "decoder.structured_translation_grounding_rate",
        MetricDirection::HigherIsBetter,
        value_at(
            baseline_decoder.as_ref(),
            &["aggregate", "structured_translation_grounding_rate"],
        ),
        value_at(
            candidate_decoder.as_ref(),
            &["aggregate", "structured_translation_grounding_rate"],
        ),
        opts.max_structured_translation_grounding_rate_regression,
    );
    compare_metric(
        &mut metrics,
        &mut missing_metrics,
        "decoder.avg_structured_translation_semantic_drift",
        MetricDirection::LowerIsBetter,
        value_at(
            baseline_decoder.as_ref(),
            &["aggregate", "avg_structured_translation_semantic_drift"],
        ),
        value_at(
            candidate_decoder.as_ref(),
            &["aggregate", "avg_structured_translation_semantic_drift"],
        ),
        opts.max_structured_translation_drift_regression,
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
    let regression_passed = failures.is_empty() && missing_required_metrics.is_empty();

    // ── Presence-of-improvement gate (Tier 1.1) ─────────────────────────────
    // Distinct from the regression comparisons above: those compare the SAME
    // artifact kind (decoder-ab / exercism-bench) between baseline_dir and
    // candidate_dir. This compares `broca-topic-coverage` reports — evidence
    // of held-out, never-trained-on objective performance — which live at
    // caller-supplied paths, not necessarily inside baseline_dir/candidate_dir.
    let baseline_topic = opts
        .baseline_topic_coverage
        .as_deref()
        .map(read_optional_json)
        .transpose()?
        .flatten();
    let candidate_topic = opts
        .candidate_topic_coverage
        .as_deref()
        .map(read_optional_json)
        .transpose()?
        .flatten();
    let improvement_evaluated = baseline_topic.is_some() && candidate_topic.is_some();

    let mut improvement_metrics = Vec::new();
    let mut missing_improvement_metrics = Vec::new();
    if improvement_evaluated {
        compare_improvement_metric(
            &mut improvement_metrics,
            &mut missing_improvement_metrics,
            "topic_coverage.avg_keyword_overlap",
            value_at(baseline_topic.as_ref(), &["avg_keyword_overlap"]),
            value_at(candidate_topic.as_ref(), &["avg_keyword_overlap"]),
            opts.min_keyword_overlap_improvement,
        );
        compare_improvement_metric(
            &mut improvement_metrics,
            &mut missing_improvement_metrics,
            "topic_coverage.avg_generated_coherence",
            value_at(baseline_topic.as_ref(), &["avg_generated_coherence"]),
            value_at(candidate_topic.as_ref(), &["avg_generated_coherence"]),
            opts.min_coherence_improvement,
        );
    }

    let improvement_failures = improvement_metrics
        .iter()
        .filter(|metric| !metric.passed)
        .map(|metric| ImprovementFailure {
            metric: metric.metric.clone(),
            baseline: metric.baseline,
            candidate: metric.candidate,
            delta: metric.delta,
            required_improvement: metric.required_improvement,
        })
        .collect::<Vec<_>>();

    let missing_required_improvement_metrics =
        if improvement_evaluated && opts.require_all_improvement_metrics {
            missing_improvement_metrics.clone()
        } else {
            Vec::new()
        };

    let improvement_passed = if improvement_evaluated {
        // Evaluated but produced zero comparable metrics (e.g. both reports
        // parsed but neither had a usable field) is itself a failure, not a
        // vacuous pass — this gate exists to require positive evidence.
        !improvement_metrics.is_empty()
            && improvement_failures.is_empty()
            && missing_required_improvement_metrics.is_empty()
    } else {
        !opts.require_improvement_evaluated
    };

    Ok(CompareReport {
        schema_version: 2,
        evidence_level: "measured",
        baseline_dir: opts.baseline_dir.display().to_string(),
        candidate_dir: opts.candidate_dir.display().to_string(),
        promotion: PromotionDecision {
            passed: regression_passed && improvement_passed,
            regression_passed,
            improvement_passed,
            improvement_evaluated,
            failures,
            missing_required_metrics,
            improvement_failures,
            missing_required_improvement_metrics,
        },
        metrics,
        missing_metrics,
        improvement_metrics,
        missing_improvement_metrics,
    })
}

fn compare_improvement_metric(
    metrics: &mut Vec<ImprovementMetricComparison>,
    missing_metrics: &mut Vec<String>,
    name: &str,
    baseline: Option<f64>,
    candidate: Option<f64>,
    required_improvement: f64,
) {
    let (Some(baseline), Some(candidate)) = (baseline, candidate) else {
        missing_metrics.push(name.to_string());
        return;
    };
    let delta = candidate - baseline;
    let passed = delta >= required_improvement;
    metrics.push(ImprovementMetricComparison {
        metric: name.to_string(),
        baseline,
        candidate,
        delta,
        required_improvement,
        passed,
    });
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
        max_structured_translation_validity_regression: 0.02,
        max_structured_translation_grounding_rate_regression: 0.0,
        max_structured_translation_drift_regression: 0.02,
        require_all_metrics: false,
        baseline_topic_coverage: None,
        candidate_topic_coverage: None,
        min_keyword_overlap_improvement: 0.05,
        min_coherence_improvement: 0.02,
        require_improvement_evaluated: false,
        require_all_improvement_metrics: false,
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
            "--max-structured-translation-validity-regression" => {
                i += 1;
                opts.max_structured_translation_validity_regression =
                    value(&args, i, "--max-structured-translation-validity-regression")?.parse()?;
            }
            "--max-structured-translation-grounding-rate-regression" => {
                i += 1;
                opts.max_structured_translation_grounding_rate_regression = value(
                    &args,
                    i,
                    "--max-structured-translation-grounding-rate-regression",
                )?
                .parse()?;
            }
            "--max-structured-translation-drift-regression" => {
                i += 1;
                opts.max_structured_translation_drift_regression =
                    value(&args, i, "--max-structured-translation-drift-regression")?.parse()?;
            }
            "--baseline-topic-coverage" => {
                i += 1;
                opts.baseline_topic_coverage =
                    Some(PathBuf::from(value(&args, i, "--baseline-topic-coverage")?));
            }
            "--candidate-topic-coverage" => {
                i += 1;
                opts.candidate_topic_coverage = Some(PathBuf::from(value(
                    &args,
                    i,
                    "--candidate-topic-coverage",
                )?));
            }
            "--min-keyword-overlap-improvement" => {
                i += 1;
                opts.min_keyword_overlap_improvement =
                    value(&args, i, "--min-keyword-overlap-improvement")?.parse()?;
            }
            "--min-coherence-improvement" => {
                i += 1;
                opts.min_coherence_improvement =
                    value(&args, i, "--min-coherence-improvement")?.parse()?;
            }
            "--require-improvement-evaluated" => opts.require_improvement_evaluated = true,
            "--require-all-improvement-metrics" => opts.require_all_improvement_metrics = true,
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
        "Usage: broca-checkpoint-compare --baseline-dir DIR --candidate-dir DIR [--json-out PATH] [--fail-on-regression] [--require-all-metrics] [--max-drift-regression F] [--max-hallucination-regression F] [--max-compile-rate-regression F] [--max-test-rate-regression F] [--max-structured-confidence-regression F] [--max-structured-validity-regression F] [--max-structured-required-role-rate-regression F] [--max-structured-translation-validity-regression F] [--max-structured-translation-grounding-rate-regression F] [--max-structured-translation-drift-regression F] [--baseline-topic-coverage PATH] [--candidate-topic-coverage PATH] [--min-keyword-overlap-improvement F] [--min-coherence-improvement F] [--require-improvement-evaluated] [--require-all-improvement-metrics]\n\n\
         The improvement gate (--baseline-topic-coverage/--candidate-topic-coverage) is a\n\
         PRESENCE-of-improvement check, distinct from the regression bounds above which only\n\
         prove ABSENCE of regression. Both --baseline-topic-coverage and --candidate-topic-coverage\n\
         must point at `broca-topic-coverage` reports generated against the SAME held-out\n\
         objective batch (one run against the production checkpoint, one against the candidate).\n\
         promotion.passed requires both regression_passed and improvement_passed."
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
            max_structured_translation_validity_regression: 0.02,
            max_structured_translation_grounding_rate_regression: 0.0,
            max_structured_translation_drift_regression: 0.02,
            require_all_metrics: false,
            baseline_topic_coverage: None,
            candidate_topic_coverage: None,
            min_keyword_overlap_improvement: 0.05,
            min_coherence_improvement: 0.02,
            require_improvement_evaluated: false,
            require_all_improvement_metrics: false,
        }
    }

    fn write_topic_coverage(path: &Path, avg_keyword_overlap: f64, avg_generated_coherence: f64) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let report = json!({
            "schema_version": 2,
            "checkpoint_path": "irrelevant-for-test.bin",
            "num_objectives": 5,
            "avg_keyword_overlap": avg_keyword_overlap,
            "avg_generated_coherence": avg_generated_coherence,
            "coherence_measured_count": 5,
            "veto_count": 0,
            "cases": []
        });
        std::fs::write(path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
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
                "structured_required_role_rate": required_role_rate,
                "avg_structured_translation_validity": validity,
                "structured_translation_grounding_rate": required_role_rate,
                "avg_structured_translation_semantic_drift": 1.0 - validity
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
    fn structured_translation_regression_fails() {
        let temp = tempfile::tempdir().unwrap();
        let baseline = temp.path().join("baseline");
        let candidate = temp.path().join("candidate");
        write_artifacts(&baseline, 0.4, 0.2, 0.8, 0.9, 1.0, 2, 1);
        write_artifacts(&candidate, 0.4, 0.2, 0.8, 0.82, 1.0, 2, 1);

        let report = compare(&opts(baseline, candidate)).unwrap();
        assert!(!report.promotion.passed);
        assert!(
            report
                .promotion
                .failures
                .iter()
                .any(|failure| { failure.metric == "decoder.avg_structured_translation_validity" })
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

    // ── Presence-of-improvement gate (Tier 1.1) ─────────────────────────────
    // These four tests are the acceptance criterion for
    // DISCOVERY_AND_SELF_IMPROVEMENT_PLAN_2026-07-06.md Tier 1.1: "a Broca
    // candidate has been rejected by the improvement gate at least once,
    // proving it can fail" — plus the accept path, and the two gate-shape
    // tests around what happens when topic-coverage evidence is absent.

    #[test]
    fn improvement_gate_vacuously_passes_without_topic_coverage() {
        // No --baseline-topic-coverage/--candidate-topic-coverage supplied,
        // and not required: the improvement gate must not silently block
        // every cycle that has no new held-out material (the common case),
        // matching the historical `passed` semantics for callers that never
        // opt into the improvement check.
        let temp = tempfile::tempdir().unwrap();
        let baseline = temp.path().join("baseline");
        let candidate = temp.path().join("candidate");
        write_artifacts(&baseline, 0.4, 0.2, 0.8, 0.9, 1.0, 2, 1);
        write_artifacts(&candidate, 0.4, 0.2, 0.8, 0.9, 1.0, 2, 1);

        let report = compare(&opts(baseline, candidate)).unwrap();
        assert!(report.promotion.regression_passed);
        assert!(!report.promotion.improvement_evaluated);
        assert!(report.promotion.improvement_passed);
        assert!(report.promotion.passed);
    }

    #[test]
    fn improvement_gate_required_but_missing_fails_closed() {
        // Same as above, but the caller explicitly required the improvement
        // gate to be evaluated (e.g. the cycle script knows it generated new
        // held-out material this cycle) and forgot/failed to supply reports.
        // Must fail closed, not silently pass.
        let temp = tempfile::tempdir().unwrap();
        let baseline = temp.path().join("baseline");
        let candidate = temp.path().join("candidate");
        write_artifacts(&baseline, 0.4, 0.2, 0.8, 0.9, 1.0, 2, 1);
        write_artifacts(&candidate, 0.4, 0.2, 0.8, 0.9, 1.0, 2, 1);

        let mut strict = opts(baseline, candidate);
        strict.require_improvement_evaluated = true;
        let report = compare(&strict).unwrap();
        assert!(report.promotion.regression_passed);
        assert!(!report.promotion.improvement_evaluated);
        assert!(!report.promotion.improvement_passed);
        assert!(!report.promotion.passed);
    }

    #[test]
    fn improvement_gate_rejects_flat_candidate_despite_clean_regression() {
        // The exact failure mode Tier 1.1 exists to catch: a candidate that
        // clears every regression bound (didn't get worse) but shows no real
        // signal of having learned the new held-out material (flat/noise-level
        // delta vs. production on the same held-out objectives). Regression
        // gate alone would promote this; the improvement gate must reject it.
        let temp = tempfile::tempdir().unwrap();
        let baseline_dir = temp.path().join("baseline");
        let candidate_dir = temp.path().join("candidate");
        write_artifacts(&baseline_dir, 0.4, 0.2, 0.8, 0.9, 1.0, 2, 1);
        write_artifacts(&candidate_dir, 0.4, 0.2, 0.8, 0.9, 1.0, 2, 1);

        let baseline_topic = temp.path().join("topic-coverage-baseline.json");
        let candidate_topic = temp.path().join("topic-coverage-candidate.json");
        // Candidate barely nudges keyword overlap (+0.02, below the 0.05
        // default margin) and coherence actually dips slightly — this is
        // what "didn't learn anything new" looks like in this report shape.
        write_topic_coverage(&baseline_topic, 0.10, 0.55);
        write_topic_coverage(&candidate_topic, 0.12, 0.54);

        let mut opts = opts(baseline_dir, candidate_dir);
        opts.baseline_topic_coverage = Some(baseline_topic);
        opts.candidate_topic_coverage = Some(candidate_topic);
        let report = compare(&opts).unwrap();

        assert!(
            report.promotion.regression_passed,
            "regression gate should pass — this candidate did not get worse"
        );
        assert!(report.promotion.improvement_evaluated);
        assert!(
            !report.promotion.improvement_passed,
            "improvement gate should reject a flat/no-improvement candidate"
        );
        assert!(!report.promotion.passed);
        assert!(
            report
                .promotion
                .improvement_failures
                .iter()
                .any(|f| f.metric == "topic_coverage.avg_keyword_overlap")
        );
    }

    #[test]
    fn improvement_gate_accepts_genuine_improvement() {
        // Mirror image: candidate clears regression AND shows a real,
        // above-margin delta over production on the same held-out
        // objectives. Both gates pass, so the combined verdict passes.
        let temp = tempfile::tempdir().unwrap();
        let baseline_dir = temp.path().join("baseline");
        let candidate_dir = temp.path().join("candidate");
        write_artifacts(&baseline_dir, 0.4, 0.2, 0.8, 0.9, 1.0, 2, 1);
        write_artifacts(&candidate_dir, 0.4, 0.2, 0.8, 0.9, 1.0, 2, 1);

        let baseline_topic = temp.path().join("topic-coverage-baseline.json");
        let candidate_topic = temp.path().join("topic-coverage-candidate.json");
        // Production has never seen these objectives (low overlap/coherence);
        // the fine-tuned candidate clearly landed the new material.
        write_topic_coverage(&baseline_topic, 0.08, 0.40);
        write_topic_coverage(&candidate_topic, 0.35, 0.60);

        let mut opts = opts(baseline_dir, candidate_dir);
        opts.baseline_topic_coverage = Some(baseline_topic);
        opts.candidate_topic_coverage = Some(candidate_topic);
        let report = compare(&opts).unwrap();

        assert!(report.promotion.regression_passed);
        assert!(report.promotion.improvement_evaluated);
        assert!(
            report.promotion.improvement_passed,
            "improvement gate should accept a candidate with a real above-margin delta"
        );
        assert!(report.promotion.improvement_failures.is_empty());
        assert!(report.promotion.passed);
    }
}
