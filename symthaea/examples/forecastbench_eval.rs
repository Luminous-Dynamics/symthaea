// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Local ForecastBench-compatible epistemic calibration lane.
//!
//! This is intentionally separate from coding pass-rate benchmarks. It measures
//! whether Symthaea's orchestration layer is calibrated about uncertain future
//! outcomes, such as "will this repair pass the next check?".
//!
//! Input format: JSONL, one `ForecastQuestion` per line. Official ForecastBench
//! exports can be converted into this shape; unresolved questions are reported
//! but excluded from proper scoring until their outcomes are known.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

const EPSILON: f64 = 1e-6;

#[derive(Debug, Clone, Deserialize)]
struct ForecastQuestion {
    id: String,
    #[serde(default)]
    category: String,
    question: String,
    #[serde(default)]
    resolution: Option<bool>,
    #[serde(default)]
    probability: Option<f64>,
    #[serde(default)]
    baseline_probability: Option<f64>,
    #[serde(default)]
    evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ForecastPrediction {
    id: String,
    category: String,
    question: String,
    probability: f64,
    baseline_probability: f64,
    resolution: Option<bool>,
    brier: Option<f64>,
    log_score: Option<f64>,
    surprise: Option<f64>,
    evidence_count: usize,
    rationale: String,
}

#[derive(Debug, Serialize)]
struct ForecastCategoryReport {
    task_count: usize,
    resolved_count: usize,
    brier_score: Option<f64>,
    accuracy_at_50: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ForecastReport {
    benchmark: String,
    source: String,
    task_count: usize,
    resolved_count: usize,
    unresolved_count: usize,
    resolved_coverage: f64,
    brier_score: Option<f64>,
    baseline_brier_score: Option<f64>,
    brier_improvement: Option<f64>,
    log_score: Option<f64>,
    accuracy_at_50: Option<f64>,
    mean_confidence: Option<f64>,
    expected_calibration_error: Option<f64>,
    overconfidence: Option<f64>,
    fep_prediction_error: Option<f64>,
    fep_free_energy: Option<f64>,
    calibration_signal: String,
    router_trust_multiplier: f64,
    quality_gate_passed: bool,
    category_reports: BTreeMap<String, ForecastCategoryReport>,
    predictions: Vec<ForecastPrediction>,
}

#[derive(Debug)]
struct Args {
    input: PathBuf,
    json: bool,
    source: String,
    max_brier: f64,
    max_ece: f64,
    min_resolved: usize,
}

fn main() {
    let args = Args::parse();
    let questions = match load_questions(&args.input) {
        Ok(questions) => questions,
        Err(err) => {
            eprintln!("forecastbench_eval: {err}");
            std::process::exit(2);
        }
    };

    let report = evaluate(&questions, &args);
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("serialize forecast report")
        );
    } else {
        print_human_report(&report);
    }

    if !report.quality_gate_passed {
        std::process::exit(1);
    }
}

impl Args {
    fn parse() -> Self {
        let mut input = PathBuf::from("tests/fixtures/forecastbench_local.jsonl");
        let mut json = false;
        let mut source = "local_coding_forecastbench".to_string();
        let mut max_brier = 0.25;
        let mut max_ece = 0.20;
        let mut min_resolved = 10usize;

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--input" => {
                    input = PathBuf::from(args.next().expect("--input requires a path"));
                }
                "--json" => json = true,
                "--source" => {
                    source = args.next().expect("--source requires a value");
                }
                "--max-brier" => {
                    max_brier = args
                        .next()
                        .expect("--max-brier requires a value")
                        .parse()
                        .expect("--max-brier must be numeric");
                }
                "--max-ece" => {
                    max_ece = args
                        .next()
                        .expect("--max-ece requires a value")
                        .parse()
                        .expect("--max-ece must be numeric");
                }
                "--min-resolved" => {
                    min_resolved = args
                        .next()
                        .expect("--min-resolved requires a value")
                        .parse()
                        .expect("--min-resolved must be an integer");
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    eprintln!("unknown argument: {other}");
                    print_help();
                    std::process::exit(2);
                }
            }
        }

        Self {
            input,
            json,
            source,
            max_brier,
            max_ece,
            min_resolved,
        }
    }
}

fn print_help() {
    eprintln!(
        "Usage: cargo run --example forecastbench_eval -- [--input path.jsonl] [--json] [--source name] [--max-brier 0.25] [--max-ece 0.20] [--min-resolved 10]"
    );
}

fn load_questions(path: &PathBuf) -> Result<Vec<ForecastQuestion>, String> {
    let text = fs::read_to_string(path).map_err(|err| format!("failed to read {path:?}: {err}"))?;
    let mut questions = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let question: ForecastQuestion = serde_json::from_str(trimmed)
            .map_err(|err| format!("{}:{} invalid JSONL: {err}", path.display(), idx + 1))?;
        questions.push(question);
    }
    if questions.is_empty() {
        return Err(format!(
            "{} contained no forecast questions",
            path.display()
        ));
    }
    Ok(questions)
}

fn evaluate(questions: &[ForecastQuestion], args: &Args) -> ForecastReport {
    let predictions: Vec<_> = questions.iter().map(predict).collect();
    let resolved: Vec<_> = predictions
        .iter()
        .filter(|prediction| prediction.resolution.is_some())
        .collect();

    let brier_score = mean_option(resolved.iter().filter_map(|prediction| prediction.brier));
    let baseline_brier_score = mean_option(resolved.iter().filter_map(|prediction| {
        prediction
            .resolution
            .map(|resolution| brier(prediction.baseline_probability, resolution))
    }));
    let brier_improvement = match (brier_score, baseline_brier_score) {
        (Some(score), Some(baseline)) => Some(baseline - score),
        _ => None,
    };
    let log_score = mean_option(
        resolved
            .iter()
            .filter_map(|prediction| prediction.log_score),
    );
    let accuracy_at_50 = accuracy_at_50(&resolved);
    let mean_confidence = mean_option(resolved.iter().map(|prediction| {
        if prediction.probability >= 0.5 {
            prediction.probability
        } else {
            1.0 - prediction.probability
        }
    }));
    let expected_calibration_error = ece(&resolved, 10);
    let overconfidence = match (mean_confidence, accuracy_at_50) {
        (Some(confidence), Some(accuracy)) => Some((confidence - accuracy).max(0.0)),
        _ => None,
    };
    let (calibration_signal, router_trust_multiplier) =
        calibration_signal(brier_score, expected_calibration_error, overconfidence);
    let quality_gate_passed = resolved.len() >= args.min_resolved
        && brier_score
            .map(|score| score <= args.max_brier)
            .unwrap_or(false)
        && expected_calibration_error
            .map(|score| score <= args.max_ece)
            .unwrap_or(false);

    ForecastReport {
        benchmark: "forecastbench_epistemic_calibration".to_string(),
        source: args.source.clone(),
        task_count: predictions.len(),
        resolved_count: resolved.len(),
        unresolved_count: predictions.len().saturating_sub(resolved.len()),
        resolved_coverage: resolved.len() as f64 / predictions.len().max(1) as f64,
        brier_score,
        baseline_brier_score,
        brier_improvement,
        log_score,
        accuracy_at_50,
        mean_confidence,
        expected_calibration_error,
        overconfidence,
        fep_prediction_error: brier_score,
        fep_free_energy: log_score,
        calibration_signal: calibration_signal.to_string(),
        router_trust_multiplier,
        quality_gate_passed,
        category_reports: category_reports(&predictions),
        predictions,
    }
}

fn predict(question: &ForecastQuestion) -> ForecastPrediction {
    let probability = question
        .probability
        .unwrap_or_else(|| heuristic_probability(question))
        .clamp(EPSILON, 1.0 - EPSILON);
    let baseline_probability = question
        .baseline_probability
        .unwrap_or(0.5)
        .clamp(EPSILON, 1.0 - EPSILON);
    let brier = question
        .resolution
        .map(|resolution| brier(probability, resolution));
    let log_score = question
        .resolution
        .map(|resolution| log_loss(probability, resolution));
    ForecastPrediction {
        id: question.id.clone(),
        category: normalize_category(&question.category),
        question: question.question.clone(),
        probability,
        baseline_probability,
        resolution: question.resolution,
        brier,
        log_score,
        surprise: brier,
        evidence_count: question.evidence.len(),
        rationale: rationale(question, probability),
    }
}

fn heuristic_probability(question: &ForecastQuestion) -> f64 {
    let text = format!(
        "{} {} {}",
        question.category,
        question.question,
        question.evidence.join(" ")
    )
    .to_lowercase();

    let mut probability: f64 = 0.5;
    for positive in [
        "passed",
        "green",
        "exact",
        "cached",
        "deterministic",
        "registered",
        "baseline",
        "proof",
        "small diff",
    ] {
        if text.contains(positive) {
            probability += 0.07;
        }
    }
    for negative in [
        "parse failure",
        "compile error",
        "refuted",
        "unsafe",
        "missing dependency",
        "flaky",
        "large diff",
        "external",
        "network",
    ] {
        if text.contains(negative) {
            probability -= 0.08;
        }
    }
    probability.clamp(0.05, 0.95)
}

fn rationale(question: &ForecastQuestion, probability: f64) -> String {
    let source = if question.probability.is_some() {
        "provided probability"
    } else {
        "local deterministic heuristic"
    };
    format!(
        "{source}; p={probability:.3}; evidence_items={}",
        question.evidence.len()
    )
}

fn normalize_category(category: &str) -> String {
    if category.trim().is_empty() {
        "uncategorized".to_string()
    } else {
        category.trim().to_string()
    }
}

fn brier(probability: f64, resolution: bool) -> f64 {
    let y = if resolution { 1.0 } else { 0.0 };
    (probability - y).powi(2)
}

fn log_loss(probability: f64, resolution: bool) -> f64 {
    if resolution {
        -probability.ln()
    } else {
        -(1.0 - probability).ln()
    }
}

fn mean_option(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut count = 0usize;
    let mut sum = 0.0;
    for value in values {
        count += 1;
        sum += value;
    }
    (count > 0).then_some(sum / count as f64)
}

fn accuracy_at_50(predictions: &[&ForecastPrediction]) -> Option<f64> {
    let mut correct = 0usize;
    let mut count = 0usize;
    for prediction in predictions {
        let Some(resolution) = prediction.resolution else {
            continue;
        };
        count += 1;
        if (prediction.probability >= 0.5) == resolution {
            correct += 1;
        }
    }
    (count > 0).then_some(correct as f64 / count as f64)
}

fn ece(predictions: &[&ForecastPrediction], bins: usize) -> Option<f64> {
    if predictions.is_empty() || bins == 0 {
        return None;
    }

    let mut total = 0.0;
    for bin in 0..bins {
        let lo = bin as f64 / bins as f64;
        let hi = (bin + 1) as f64 / bins as f64;
        let bucket: Vec<_> = predictions
            .iter()
            .copied()
            .filter(|prediction| {
                let confidence = if prediction.probability >= 0.5 {
                    prediction.probability
                } else {
                    1.0 - prediction.probability
                };
                if bin + 1 == bins {
                    confidence >= lo && confidence <= hi
                } else {
                    confidence >= lo && confidence < hi
                }
            })
            .collect();
        if bucket.is_empty() {
            continue;
        }
        let accuracy = accuracy_at_50(&bucket).unwrap_or(0.0);
        let confidence = mean_option(bucket.iter().map(|prediction| {
            if prediction.probability >= 0.5 {
                prediction.probability
            } else {
                1.0 - prediction.probability
            }
        }))
        .unwrap_or(0.0);
        total += (bucket.len() as f64 / predictions.len() as f64) * (confidence - accuracy).abs();
    }
    Some(total)
}

fn calibration_signal(
    brier_score: Option<f64>,
    expected_calibration_error: Option<f64>,
    overconfidence: Option<f64>,
) -> (&'static str, f64) {
    let brier_score = brier_score.unwrap_or(1.0);
    let ece = expected_calibration_error.unwrap_or(1.0);
    let overconfidence = overconfidence.unwrap_or(1.0);
    if brier_score <= 0.08 && ece <= 0.12 && overconfidence <= 0.10 {
        ("promote", 1.15)
    } else if brier_score <= 0.18 && ece <= 0.22 && overconfidence <= 0.20 {
        ("hold", 1.0)
    } else if brier_score <= 0.30 && ece <= 0.35 {
        ("caution", 0.75)
    } else {
        ("demote", 0.50)
    }
}

fn category_reports(
    predictions: &[ForecastPrediction],
) -> BTreeMap<String, ForecastCategoryReport> {
    let mut grouped: BTreeMap<String, Vec<&ForecastPrediction>> = BTreeMap::new();
    for prediction in predictions {
        grouped
            .entry(prediction.category.clone())
            .or_default()
            .push(prediction);
    }

    grouped
        .into_iter()
        .map(|(category, items)| {
            let resolved: Vec<_> = items
                .iter()
                .copied()
                .filter(|prediction| prediction.resolution.is_some())
                .collect();
            (
                category,
                ForecastCategoryReport {
                    task_count: items.len(),
                    resolved_count: resolved.len(),
                    brier_score: mean_option(resolved.iter().filter_map(|p| p.brier)),
                    accuracy_at_50: accuracy_at_50(&resolved),
                },
            )
        })
        .collect()
}

fn print_human_report(report: &ForecastReport) {
    println!("ForecastBench epistemic calibration");
    println!("  source: {}", report.source);
    println!(
        "  tasks: {} resolved: {}",
        report.task_count, report.resolved_count
    );
    println!("  brier: {:.4}", report.brier_score.unwrap_or(f64::NAN));
    println!(
        "  ece: {:.4}",
        report.expected_calibration_error.unwrap_or(f64::NAN)
    );
    println!(
        "  calibration_signal: {} router_trust_multiplier: {:.2}",
        report.calibration_signal, report.router_trust_multiplier
    );
    println!("  quality_gate_passed: {}", report.quality_gate_passed);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_args() -> Args {
        Args {
            input: PathBuf::from("unused.jsonl"),
            json: true,
            source: "test".to_string(),
            max_brier: 0.25,
            max_ece: 0.20,
            min_resolved: 2,
        }
    }

    #[test]
    fn brier_and_log_loss_are_directional() {
        assert!(brier(0.9, true) < brier(0.1, true));
        assert!(brier(0.1, false) < brier(0.9, false));
        assert!(log_loss(0.9, true) < log_loss(0.1, true));
    }

    #[test]
    fn unresolved_questions_are_excluded_from_scores() {
        let questions = vec![
            ForecastQuestion {
                id: "a".into(),
                category: "gate".into(),
                question: "will pass".into(),
                resolution: Some(true),
                probability: Some(0.9),
                baseline_probability: Some(0.5),
                evidence: vec![],
            },
            ForecastQuestion {
                id: "b".into(),
                category: "gate".into(),
                question: "will resolve later".into(),
                resolution: None,
                probability: Some(0.1),
                baseline_probability: Some(0.5),
                evidence: vec![],
            },
        ];
        let report = evaluate(&questions, &test_args());
        assert_eq!(report.task_count, 2);
        assert_eq!(report.resolved_count, 1);
        assert_eq!(report.unresolved_count, 1);
        assert_eq!(report.brier_score, Some(0.009999999999999995));
    }

    #[test]
    fn calibration_signal_promotes_only_strong_calibration() {
        assert_eq!(
            calibration_signal(Some(0.04), Some(0.05), Some(0.02)),
            ("promote", 1.15)
        );
        assert_eq!(
            calibration_signal(Some(0.40), Some(0.40), Some(0.10)),
            ("demote", 0.50)
        );
    }
}
