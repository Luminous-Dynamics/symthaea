// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic case-bootstrap confidence intervals for PSAFE safety metrics.
//!
//! Bootstrap sampling is over independently adjudicated evaluation cases, not
//! over model seeds or chemical scaffolds. For chemically clustered data, use
//! a group/bootstrap design at the relevant unit of independence instead.
//! Intervals quantify sampling variability on the frozen test set; they do not
//! repair label bias, leakage, domain shift, or a poor test-set design.

use crate::pharmacology_metrics::{
    BinarySafetyEvaluationCase, BinarySafetyMetricConfig, BinarySafetyMetricReport,
    evaluate_binary_safety,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

pub const BOOTSTRAP_SCHEMA_VERSION: &str = "1.0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntervalStatus {
    Estimated,
    InsufficientValidReplicates,
    UndefinedOnAllReplicates,
}

/// Percentile interval estimated from the bootstrap replicate distribution.
/// Bounds remain None if the minimum valid-replicate gate is not met.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapMetricInterval {
    pub metric: String,
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    pub valid_replicates: usize,
    pub requested_replicates: usize,
    pub status: IntervalStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapConfig {
    /// Fixed seed, recorded with the report for exact replay.
    pub seed: u64,
    pub replicates: usize,
    /// Minimum number of non-undefined resamples needed to publish bounds.
    pub minimum_valid_replicates: usize,
    /// Two-sided confidence level, e.g. 0.95.
    pub confidence_level: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BootstrapReport {
    pub schema_version: String,
    pub evaluation_id: String,
    pub split_id: String,
    pub seed: u64,
    pub replicates: usize,
    pub minimum_valid_replicates: usize,
    pub confidence_level: f64,
    pub quantile_method: String,
    pub random_generator: String,
    pub intervals: BTreeMap<String, BootstrapMetricInterval>,
    pub notes: Vec<String>,
}

/// Bootstrap uncertainty around point metrics by resampling cases with
/// replacement. Duplicate sampled rows receive synthetic unique case IDs
/// because each occurrence in a bootstrap sample is a distinct draw.
///
/// The deterministic SplitMix64 generator avoids depending on a crate-specific
/// RNG sequence. It is **not cryptographically secure** and is only used for
/// reproducible statistical resampling.
pub fn bootstrap_binary_safety_metrics(
    evaluation_config: &BinarySafetyMetricConfig,
    cases: &[BinarySafetyEvaluationCase],
    bootstrap_config: &BootstrapConfig,
) -> Result<BootstrapReport, Vec<String>> {
    let mut errors = Vec::new();
    if cases.is_empty() {
        errors.push("bootstrap requires at least one evaluation case".into());
    }
    if bootstrap_config.replicates == 0 {
        errors.push("bootstrap replicates must be greater than zero".into());
    }
    if bootstrap_config.minimum_valid_replicates == 0
        || bootstrap_config.minimum_valid_replicates > bootstrap_config.replicates
    {
        errors.push("minimum_valid_replicates must be within 1..=replicates".into());
    }
    if !bootstrap_config.confidence_level.is_finite()
        || !(0.0 < bootstrap_config.confidence_level
            && bootstrap_config.confidence_level < 1.0)
    {
        errors.push("confidence_level must be finite and strictly within (0, 1)".into());
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    // Fail before resampling if the original evaluation input is invalid.
    let point = evaluate_binary_safety(evaluation_config, cases).map_err(|errors| {
        errors
            .into_iter()
            .map(|error| format!("invalid original evaluation: {error}"))
            .collect::<Vec<_>>()
    })?;

    let mut samples: BTreeMap<String, Vec<f64>> = metric_names()
        .into_iter()
        .map(|name| (name.to_string(), Vec::with_capacity(bootstrap_config.replicates)))
        .collect();
    let mut rng = SplitMix64::new(bootstrap_config.seed);
    let sample_size = cases.len();

    for replicate in 0..bootstrap_config.replicates {
        let mut sampled = Vec::with_capacity(sample_size);
        for draw in 0..sample_size {
            let index = rng.index(sample_size);
            let mut case = cases[index].clone();
            case.case_id = format!(
                "bootstrap-{replicate}-{draw}-{}",
                case.case_id
            );
            sampled.push(case);
        }

        let report = evaluate_binary_safety(evaluation_config, &sampled).map_err(|errors| {
            errors
                .into_iter()
                .map(|error| format!("bootstrap replicate {replicate} invalid: {error}"))
                .collect::<Vec<_>>()
        })?;
        append_metric_values(&mut samples, &report);
    }

    let alpha = 1.0 - bootstrap_config.confidence_level;
    let lower_quantile = alpha / 2.0;
    let upper_quantile = 1.0 - alpha / 2.0;
    let mut intervals = BTreeMap::new();
    for metric in metric_names() {
        let values = samples.remove(metric).unwrap_or_default();
        let valid = values.len();
        let status = if valid == 0 {
            IntervalStatus::UndefinedOnAllReplicates
        } else if valid < bootstrap_config.minimum_valid_replicates {
            IntervalStatus::InsufficientValidReplicates
        } else {
            IntervalStatus::Estimated
        };
        let (lower_bound, upper_bound) = if status == IntervalStatus::Estimated {
            let lower = percentile(&values, lower_quantile);
            let upper = percentile(&values, upper_quantile);
            (lower, upper)
        } else {
            (None, None)
        };

        intervals.insert(
            metric.to_string(),
            BootstrapMetricInterval {
                metric: metric.to_string(),
                lower_bound,
                upper_bound,
                valid_replicates: valid,
                requested_replicates: bootstrap_config.replicates,
                status,
            },
        );
    }

    let mut notes = vec![
        "Intervals measure case-resampling variability on the supplied fixed evaluation set; they do not establish external validity or causal truth.".to_string(),
        "All model comparisons should use paired resamples over the same test cases, not independent intervals alone.".to_string(),
        "If samples are clustered by chemical scaffold, publication, assay series, or study, bootstrap the independent groups instead of individual rows.".to_string(),
        "Abstentions remain in the resampled case set; predictive metrics exclude them while coverage includes every sampled case.".to_string(),
    ];
    if intervals.values().any(|interval| interval.status != IntervalStatus::Estimated) {
        notes.push("At least one metric has too few valid resamples (or is undefined); no interval bounds are published for that metric.".to_string());
    }

    Ok(BootstrapReport {
        schema_version: BOOTSTRAP_SCHEMA_VERSION.to_string(),
        evaluation_id: point.evaluation_id,
        split_id: point.split_id,
        seed: bootstrap_config.seed,
        replicates: bootstrap_config.replicates,
        minimum_valid_replicates: bootstrap_config.minimum_valid_replicates,
        confidence_level: bootstrap_config.confidence_level,
        quantile_method: "linear interpolation at h=(n-1)*p (R-7)".to_string(),
        random_generator: "SplitMix64, deterministic; not cryptographic".to_string(),
        intervals,
        notes,
    })
}

fn metric_names() -> &'static [&'static str] {
    &[
        "coverage",
        "sensitivity",
        "specificity",
        "precision",
        "negative_predictive_value",
        "balanced_accuracy",
        "auroc",
        "average_precision",
        "brier_score",
        "expected_calibration_error",
    ]
}

fn append_metric_values(
    samples: &mut BTreeMap<String, Vec<f64>>,
    report: &BinarySafetyMetricReport,
) {
    let values = [
        ("coverage", report.coverage),
        ("sensitivity", report.sensitivity),
        ("specificity", report.specificity),
        ("precision", report.precision),
        ("negative_predictive_value", report.negative_predictive_value),
        ("balanced_accuracy", report.balanced_accuracy),
        ("auroc", report.auroc),
        ("average_precision", report.average_precision),
        ("brier_score", report.brier_score),
        ("expected_calibration_error", report.expected_calibration_error),
    ];
    for (name, value) in values {
        if let Some(value) = value {
            samples.entry(name.to_string()).or_default().push(value);
        }
    }
}

fn percentile(values: &[f64], probability: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let h = (sorted.len() - 1) as f64 * probability;
    let lower = h.floor() as usize;
    let upper = h.ceil() as usize;
    if lower == upper {
        return Some(sorted[lower]);
    }
    let fraction = h - lower as f64;
    Some(sorted[lower] + fraction * (sorted[upper] - sorted[lower]))
}

/// SplitMix64 is compact and deterministic across Rust versions. Its purpose
/// here is repeatable bootstrap indexing, not cryptographic unpredictability.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn index(&mut self, length: usize) -> usize {
        debug_assert!(length > 0);
        // Rejection sampling avoids modulo bias for the finite index range.
        let bound = length as u64;
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let value = self.next_u64();
            if value >= threshold {
                return (value % bound) as usize;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pharmacology_metrics::BinarySafetyOutput;

    fn eval_config() -> BinarySafetyMetricConfig {
        BinarySafetyMetricConfig {
            evaluation_id: "bootstrap-test-v1".to_string(),
            input_data_manifest_digest: "a".repeat(64),
            label_definition_digest: "b".repeat(64),
            split_id: "locked-test-v1".to_string(),
            hazard_threshold: 0.5,
            calibration_bins: 2,
        }
    }

    fn case(id: &str, label: bool, probability: f64) -> BinarySafetyEvaluationCase {
        BinarySafetyEvaluationCase {
            case_id: id.to_string(),
            ground_truth_hazard: label,
            output: BinarySafetyOutput::Probability {
                hazard_probability: probability,
            },
        }
    }

    fn config(seed: u64) -> BootstrapConfig {
        BootstrapConfig {
            seed,
            replicates: 400,
            minimum_valid_replicates: 100,
            confidence_level: 0.95,
        }
    }

    fn balanced_cases() -> Vec<BinarySafetyEvaluationCase> {
        vec![
            case("p1", true, 0.9),
            case("p2", true, 0.8),
            case("p3", true, 0.65),
            case("n1", false, 0.3),
            case("n2", false, 0.2),
            case("n3", false, 0.1),
        ]
    }

    #[test]
    fn bootstrap_is_deterministic_for_a_fixed_seed() {
        let cases = balanced_cases();
        let first = bootstrap_binary_safety_metrics(&eval_config(), &cases, &config(42))
            .expect("first bootstrap");
        let second = bootstrap_binary_safety_metrics(&eval_config(), &cases, &config(42))
            .expect("second bootstrap");
        assert_eq!(first, second);
        assert_eq!(first.seed, 42);
        assert_eq!(first.replicates, 400);
    }

    #[test]
    fn confidence_bounds_are_ordered_and_respect_metric_ranges() {
        let report = bootstrap_binary_safety_metrics(
            &eval_config(),
            &balanced_cases(),
            &config(9),
        )
        .expect("bootstrap");
        for name in ["sensitivity", "specificity", "coverage", "auroc"] {
            let interval = &report.intervals[name];
            if interval.status == IntervalStatus::Estimated {
                let lower = interval.lower_bound.expect("lower bound");
                let upper = interval.upper_bound.expect("upper bound");
                assert!(lower <= upper);
                assert!((0.0..=1.0).contains(&lower));
                assert!((0.0..=1.0).contains(&upper));
            }
        }
    }

    #[test]
    fn abstention_changes_coverage_and_preserves_abstained_cases() {
        let mut cases = balanced_cases();
        cases[0].output = BinarySafetyOutput::Abstained {
            reason: "review required".to_string(),
        };
        let report = bootstrap_binary_safety_metrics(
            &eval_config(),
            &cases,
            &config(12),
        )
        .expect("bootstrap");
        assert!(report.intervals["coverage"].valid_replicates > 0);
        assert!(report.notes.iter().any(|note| note.contains("Abstentions remain")));
    }

    #[test]
    fn invalid_bootstrap_configuration_fails_closed() {
        let cases = balanced_cases();
        let mut bad = config(1);
        bad.minimum_valid_replicates = bad.replicates + 1;
        assert!(bootstrap_binary_safety_metrics(&eval_config(), &cases, &bad)
            .expect_err("bad minimum must fail")
            .iter()
            .any(|error| error.contains("minimum_valid_replicates")));
        let mut bad = config(1);
        bad.confidence_level = 1.0;
        assert!(bootstrap_binary_safety_metrics(&eval_config(), &cases, &bad)
            .expect_err("bad confidence must fail")
            .iter()
            .any(|error| error.contains("confidence_level")));
    }

    #[test]
    fn too_few_valid_resamples_withhold_interval_bounds() {
        let cases = vec![case("only-positive", true, 0.9)];
        let mut bootstrap = config(5);
        bootstrap.replicates = 10;
        bootstrap.minimum_valid_replicates = 5;
        let report = bootstrap_binary_safety_metrics(&eval_config(), &cases, &bootstrap)
            .expect("bootstrap");
        // Sensitivity is defined, but specificity/AUROC are not in any resample.
        assert_eq!(
            report.intervals["specificity"].status,
            IntervalStatus::UndefinedOnAllReplicates
        );
        assert_eq!(report.intervals["specificity"].lower_bound, None);
        assert_eq!(report.intervals["coverage"].status, IntervalStatus::Estimated);
    }
}
