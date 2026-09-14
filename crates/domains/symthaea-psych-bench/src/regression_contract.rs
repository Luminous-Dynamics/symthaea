// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Direction-aware, completeness-preserving regression comparison.
//!
//! This module is an additive replacement path for the historical
//! `harness::snapshot::RegressionReport::compare` semantics. It deliberately
//! leaves that comparator unchanged until this contract is independently
//! qualified and downstream snapshot/report consumers migrate.
//!
//! Core invariants:
//!
//! - numeric decrease is not inherently regression;
//! - numeric increase is not inherently improvement;
//! - every baseline-required metric produces exactly one disposition;
//! - missing, invalid, incomparable, and policy-missing evidence remain visible;
//! - near-zero baselines never auto-pass percentage comparisons;
//! - metric policy id/version are preserved in every comparison result;
//! - snapshot schema compatibility is checked before numeric comparison.

use crate::harness::report::MetricValue;
use crate::harness::snapshot::{RegressionSnapshot, SNAPSHOT_SCHEMA_VERSION};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

const NEAR_ZERO: f64 = 1e-10;

/// Stable identity of one benchmark metric.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MetricKey {
    pub benchmark: String,
    pub metric: String,
}

impl MetricKey {
    pub fn new(benchmark: impl Into<String>, metric: impl Into<String>) -> Self {
        Self {
            benchmark: benchmark.into(),
            metric: metric.into(),
        }
    }
}

/// Explicit interpretation of movement in one metric.
///
/// Direction is never inferred from the metric name.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MetricObjective {
    HigherIsBetter,
    LowerIsBetter,
    TargetRange { min: f64, max: f64 },
    TwoSidedStable {
        warning_abs: f64,
        critical_abs: f64,
    },
}

/// Versioned comparison semantics for one metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricComparisonPolicy {
    pub policy_id: String,
    pub version: u32,
    pub objective: MetricObjective,
}

impl MetricComparisonPolicy {
    pub fn higher(policy_id: impl Into<String>, version: u32) -> Self {
        Self {
            policy_id: policy_id.into(),
            version,
            objective: MetricObjective::HigherIsBetter,
        }
    }

    pub fn lower(policy_id: impl Into<String>, version: u32) -> Self {
        Self {
            policy_id: policy_id.into(),
            version,
            objective: MetricObjective::LowerIsBetter,
        }
    }

    pub fn target_range(
        policy_id: impl Into<String>,
        version: u32,
        min: f64,
        max: f64,
    ) -> Self {
        Self {
            policy_id: policy_id.into(),
            version,
            objective: MetricObjective::TargetRange { min, max },
        }
    }

    pub fn two_sided(
        policy_id: impl Into<String>,
        version: u32,
        warning_abs: f64,
        critical_abs: f64,
    ) -> Self {
        Self {
            policy_id: policy_id.into(),
            version,
            objective: MetricObjective::TwoSidedStable {
                warning_abs,
                critical_abs,
            },
        }
    }

    fn validate(&self) -> Result<(), String> {
        if self.policy_id.trim().is_empty() {
            return Err("policy_id must be non-empty".to_string());
        }
        if self.version == 0 {
            return Err("policy version must be >= 1".to_string());
        }

        match &self.objective {
            MetricObjective::HigherIsBetter | MetricObjective::LowerIsBetter => Ok(()),
            MetricObjective::TargetRange { min, max } => {
                if !min.is_finite() || !max.is_finite() {
                    Err("target range bounds must be finite".to_string())
                } else if min > max {
                    Err("target range min must be <= max".to_string())
                } else {
                    Ok(())
                }
            }
            MetricObjective::TwoSidedStable {
                warning_abs,
                critical_abs,
            } => validate_nonnegative_ordered(
                *warning_abs,
                *critical_abs,
                "two-sided thresholds",
            ),
        }
    }
}

/// Explicit metric-policy registry used by the comparator.
pub type MetricPolicyRegistry = BTreeMap<MetricKey, MetricComparisonPolicy>;

/// Historical percentage thresholds, now validated before use.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RegressionThresholds {
    pub warning_fraction: f64,
    pub critical_fraction: f64,
}

impl RegressionThresholds {
    pub fn new(warning_fraction: f64, critical_fraction: f64) -> Result<Self, String> {
        validate_nonnegative_ordered(
            warning_fraction,
            critical_fraction,
            "regression thresholds",
        )?;
        Ok(Self {
            warning_fraction,
            critical_fraction,
        })
    }
}

fn validate_nonnegative_ordered(warning: f64, critical: f64, what: &str) -> Result<(), String> {
    if !warning.is_finite() || !critical.is_finite() {
        return Err(format!("{what} must be finite"));
    }
    if warning < 0.0 || critical < 0.0 {
        return Err(format!("{what} must be non-negative"));
    }
    if critical < warning {
        return Err(format!("{what} require critical >= warning"));
    }
    Ok(())
}

/// One accounted disposition for every baseline-required metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonDisposition {
    Pass,
    Warning,
    Critical,
    MissingCurrentBenchmark,
    MissingCurrentMetric,
    InvalidBaseline,
    InvalidCurrent,
    Incomparable,
    PolicyMismatch,
    InvalidPolicy,
}

impl ComparisonDisposition {
    /// Blocking integrity failure for an authority-bearing comparison lane.
    ///
    /// Warning remains visible but non-blocking here; callers may adopt a stricter
    /// release policy without changing the scientific meaning of the disposition.
    pub const fn is_blocking_integrity_failure(self) -> bool {
        matches!(
            self,
            Self::Critical
                | Self::MissingCurrentBenchmark
                | Self::MissingCurrentMetric
                | Self::InvalidBaseline
                | Self::InvalidCurrent
                | Self::Incomparable
                | Self::PolicyMismatch
                | Self::InvalidPolicy
        )
    }
}

/// Policy-aware result for one baseline-required metric.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyAwareRegressionResult {
    pub key: MetricKey,
    pub baseline_mean: Option<f64>,
    pub baseline_ci_lower: Option<f64>,
    pub baseline_ci_upper: Option<f64>,
    pub current_mean: Option<f64>,
    /// Signed current-minus-baseline delta when both values are valid.
    pub delta: Option<f64>,
    /// Signed relative delta when percentage comparison is meaningful.
    pub delta_fraction: Option<f64>,
    pub disposition: ComparisonDisposition,
    /// Exact policy that produced this disposition. Missing policy stays `None`.
    pub policy: Option<MetricComparisonPolicy>,
    /// Diagnostic explanation for nontrivial/non-comparable states.
    pub note: Option<String>,
}

/// Aggregate counts. `total_required` is always the number of baseline metrics,
/// not merely the subset that happened to exist in the current result.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyAwareRegressionSummary {
    pub total_required: usize,
    pub pass: usize,
    pub warning: usize,
    pub critical: usize,
    pub missing: usize,
    pub invalid: usize,
    pub incomparable: usize,
    pub policy_failure: usize,
    pub extra_current: usize,
}

/// Complete comparison report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyAwareRegressionReport {
    pub baseline_name: String,
    pub current_name: String,
    pub baseline_schema_version: String,
    pub current_schema_version: String,
    pub thresholds: RegressionThresholds,
    pub results: Vec<PolicyAwareRegressionResult>,
    /// Metrics present only in the current snapshot. Recorded, not failed by default.
    pub extra_current_metrics: Vec<MetricKey>,
    pub summary: PolicyAwareRegressionSummary,
}

impl PolicyAwareRegressionReport {
    pub fn has_blocking_integrity_failure(&self) -> bool {
        self.results
            .iter()
            .any(|result| result.disposition.is_blocking_integrity_failure())
    }
}

/// Setup failures that make a numerical comparison invalid as a whole.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComparisonSetupError {
    InvalidThresholdPolicy(String),
    SnapshotSchemaMismatch {
        baseline: String,
        current: String,
        required: String,
    },
}

/// Compare two snapshots using explicit, versioned metric semantics.
pub fn compare_snapshots(
    baseline: &RegressionSnapshot,
    current: &RegressionSnapshot,
    policies: &MetricPolicyRegistry,
    thresholds: RegressionThresholds,
) -> Result<PolicyAwareRegressionReport, ComparisonSetupError> {
    RegressionThresholds::new(thresholds.warning_fraction, thresholds.critical_fraction)
        .map_err(ComparisonSetupError::InvalidThresholdPolicy)?;

    if baseline.schema_version != SNAPSHOT_SCHEMA_VERSION
        || current.schema_version != SNAPSHOT_SCHEMA_VERSION
        || baseline.schema_version != current.schema_version
    {
        return Err(ComparisonSetupError::SnapshotSchemaMismatch {
            baseline: baseline.schema_version.clone(),
            current: current.schema_version.clone(),
            required: SNAPSHOT_SCHEMA_VERSION.to_string(),
        });
    }

    let mut results = Vec::new();
    let mut baseline_keys = BTreeSet::new();

    for (benchmark, baseline_metrics) in &baseline.metrics {
        let current_metrics = current.metrics.get(benchmark);
        for (metric, baseline_value) in baseline_metrics {
            let key = MetricKey::new(benchmark, metric);
            baseline_keys.insert(key.clone());
            let policy = policies.get(&key).cloned();

            let result = match current_metrics {
                None => missing_result(
                    key,
                    baseline_value,
                    policy,
                    ComparisonDisposition::MissingCurrentBenchmark,
                    "baseline-required benchmark is absent from current snapshot",
                ),
                Some(metrics) => match metrics.get(metric) {
                    None => missing_result(
                        key,
                        baseline_value,
                        policy,
                        ComparisonDisposition::MissingCurrentMetric,
                        "baseline-required metric is absent from current snapshot",
                    ),
                    Some(current_value) => compare_metric(
                        key,
                        baseline_value,
                        current_value,
                        policy,
                        thresholds,
                    ),
                },
            };
            results.push(result);
        }
    }

    let mut extra_current_metrics = Vec::new();
    for (benchmark, current_metrics) in &current.metrics {
        for metric in current_metrics.keys() {
            let key = MetricKey::new(benchmark, metric);
            if !baseline_keys.contains(&key) {
                extra_current_metrics.push(key);
            }
        }
    }
    extra_current_metrics.sort();

    let summary = summarize(&results, extra_current_metrics.len());

    Ok(PolicyAwareRegressionReport {
        baseline_name: baseline.name.clone(),
        current_name: current.name.clone(),
        baseline_schema_version: baseline.schema_version.clone(),
        current_schema_version: current.schema_version.clone(),
        thresholds,
        results,
        extra_current_metrics,
        summary,
    })
}

fn missing_result(
    key: MetricKey,
    baseline: &MetricValue,
    policy: Option<MetricComparisonPolicy>,
    disposition: ComparisonDisposition,
    note: &str,
) -> PolicyAwareRegressionResult {
    PolicyAwareRegressionResult {
        key,
        baseline_mean: Some(baseline.mean),
        baseline_ci_lower: Some(baseline.ci_lower),
        baseline_ci_upper: Some(baseline.ci_upper),
        current_mean: None,
        delta: None,
        delta_fraction: None,
        disposition,
        policy,
        note: Some(note.to_string()),
    }
}

fn compare_metric(
    key: MetricKey,
    baseline: &MetricValue,
    current: &MetricValue,
    policy: Option<MetricComparisonPolicy>,
    thresholds: RegressionThresholds,
) -> PolicyAwareRegressionResult {
    let Some(policy) = policy else {
        return PolicyAwareRegressionResult {
            key,
            baseline_mean: Some(baseline.mean),
            baseline_ci_lower: Some(baseline.ci_lower),
            baseline_ci_upper: Some(baseline.ci_upper),
            current_mean: Some(current.mean),
            delta: None,
            delta_fraction: None,
            disposition: ComparisonDisposition::PolicyMismatch,
            policy: None,
            note: Some("baseline-required metric has no explicit comparison policy".to_string()),
        };
    };

    if let Err(reason) = policy.validate() {
        return result_with_policy(
            key,
            baseline,
            current,
            policy,
            ComparisonDisposition::InvalidPolicy,
            None,
            None,
            Some(reason),
        );
    }

    if !metric_value_is_valid(baseline) {
        return result_with_policy(
            key,
            baseline,
            current,
            policy,
            ComparisonDisposition::InvalidBaseline,
            None,
            None,
            Some("baseline metric contains non-finite/inconsistent statistics".to_string()),
        );
    }
    if !metric_value_is_valid(current) {
        return result_with_policy(
            key,
            baseline,
            current,
            policy,
            ComparisonDisposition::InvalidCurrent,
            None,
            None,
            Some("current metric contains non-finite/inconsistent statistics".to_string()),
        );
    }

    let delta = current.mean - baseline.mean;
    match policy.objective.clone() {
        MetricObjective::HigherIsBetter => {
            if baseline.mean.abs() < NEAR_ZERO {
                return result_with_policy(
                    key,
                    baseline,
                    current,
                    policy,
                    ComparisonDisposition::Incomparable,
                    Some(delta),
                    None,
                    Some(
                        "near-zero baseline cannot support relative higher-is-better comparison"
                            .to_string(),
                    ),
                );
            }
            let delta_fraction = delta / baseline.mean.abs();
            let degradation = (-delta_fraction).max(0.0);
            let disposition = if current.mean < baseline.ci_lower {
                threshold_disposition(degradation, thresholds)
            } else {
                ComparisonDisposition::Pass
            };
            result_with_policy(
                key,
                baseline,
                current,
                policy,
                disposition,
                Some(delta),
                Some(delta_fraction),
                None,
            )
        }
        MetricObjective::LowerIsBetter => {
            if baseline.mean.abs() < NEAR_ZERO {
                return result_with_policy(
                    key,
                    baseline,
                    current,
                    policy,
                    ComparisonDisposition::Incomparable,
                    Some(delta),
                    None,
                    Some(
                        "near-zero baseline cannot support relative lower-is-better comparison"
                            .to_string(),
                    ),
                );
            }
            let delta_fraction = delta / baseline.mean.abs();
            let degradation = delta_fraction.max(0.0);
            let disposition = if current.mean > baseline.ci_upper {
                threshold_disposition(degradation, thresholds)
            } else {
                ComparisonDisposition::Pass
            };
            result_with_policy(
                key,
                baseline,
                current,
                policy,
                disposition,
                Some(delta),
                Some(delta_fraction),
                None,
            )
        }
        MetricObjective::TargetRange { min, max } => {
            let disposition = if (min..=max).contains(&current.mean) {
                ComparisonDisposition::Pass
            } else {
                ComparisonDisposition::Critical
            };
            result_with_policy(
                key,
                baseline,
                current,
                policy,
                disposition,
                Some(delta),
                None,
                None,
            )
        }
        MetricObjective::TwoSidedStable {
            warning_abs,
            critical_abs,
        } => {
            let abs_delta = delta.abs();
            let disposition = if abs_delta >= critical_abs {
                ComparisonDisposition::Critical
            } else if abs_delta >= warning_abs {
                ComparisonDisposition::Warning
            } else {
                ComparisonDisposition::Pass
            };
            result_with_policy(
                key,
                baseline,
                current,
                policy,
                disposition,
                Some(delta),
                None,
                None,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn result_with_policy(
    key: MetricKey,
    baseline: &MetricValue,
    current: &MetricValue,
    policy: MetricComparisonPolicy,
    disposition: ComparisonDisposition,
    delta: Option<f64>,
    delta_fraction: Option<f64>,
    note: Option<String>,
) -> PolicyAwareRegressionResult {
    PolicyAwareRegressionResult {
        key,
        baseline_mean: Some(baseline.mean),
        baseline_ci_lower: Some(baseline.ci_lower),
        baseline_ci_upper: Some(baseline.ci_upper),
        current_mean: Some(current.mean),
        delta,
        delta_fraction,
        disposition,
        policy: Some(policy),
        note,
    }
}

fn metric_value_is_valid(value: &MetricValue) -> bool {
    value.mean.is_finite()
        && value.std_dev.is_finite()
        && value.ci_lower.is_finite()
        && value.ci_upper.is_finite()
        && value.std_dev >= 0.0
        && value.ci_lower <= value.ci_upper
        && value.n > 0
}

fn threshold_disposition(
    degradation_fraction: f64,
    thresholds: RegressionThresholds,
) -> ComparisonDisposition {
    if degradation_fraction >= thresholds.critical_fraction {
        ComparisonDisposition::Critical
    } else if degradation_fraction >= thresholds.warning_fraction {
        ComparisonDisposition::Warning
    } else {
        ComparisonDisposition::Pass
    }
}

fn summarize(
    results: &[PolicyAwareRegressionResult],
    extra_current: usize,
) -> PolicyAwareRegressionSummary {
    let mut summary = PolicyAwareRegressionSummary {
        total_required: results.len(),
        extra_current,
        ..Default::default()
    };

    for result in results {
        match result.disposition {
            ComparisonDisposition::Pass => summary.pass += 1,
            ComparisonDisposition::Warning => summary.warning += 1,
            ComparisonDisposition::Critical => summary.critical += 1,
            ComparisonDisposition::MissingCurrentBenchmark
            | ComparisonDisposition::MissingCurrentMetric => summary.missing += 1,
            ComparisonDisposition::InvalidBaseline | ComparisonDisposition::InvalidCurrent => {
                summary.invalid += 1
            }
            ComparisonDisposition::Incomparable => summary.incomparable += 1,
            ComparisonDisposition::PolicyMismatch | ComparisonDisposition::InvalidPolicy => {
                summary.policy_failure += 1
            }
        }
    }

    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metric(mean: f64, ci_lower: f64, ci_upper: f64) -> MetricValue {
        MetricValue {
            mean,
            std_dev: 0.05,
            n: 20,
            ci_lower,
            ci_upper,
        }
    }

    fn snapshot(name: &str, entries: &[(&str, &str, MetricValue)]) -> RegressionSnapshot {
        let mut metrics: BTreeMap<String, BTreeMap<String, MetricValue>> = BTreeMap::new();
        for (benchmark, metric_name, value) in entries {
            metrics
                .entry((*benchmark).to_string())
                .or_default()
                .insert((*metric_name).to_string(), value.clone());
        }
        RegressionSnapshot {
            name: name.to_string(),
            timestamp: "2026-09-14T00:00:00Z".to_string(),
            git_hash: Some("test-subject".to_string()),
            config_summary: "typed-regression-test".to_string(),
            schema_version: SNAPSHOT_SCHEMA_VERSION.to_string(),
            metrics,
        }
    }

    fn thresholds() -> RegressionThresholds {
        RegressionThresholds::new(0.05, 0.10).expect("valid thresholds")
    }

    #[test]
    fn lower_is_better_improvement_below_baseline_ci_passes() {
        let baseline = snapshot("baseline", &[("Latency", "ms", metric(100.0, 95.0, 105.0))]);
        let current = snapshot("current", &[("Latency", "ms", metric(80.0, 78.0, 82.0))]);
        let mut policies = MetricPolicyRegistry::new();
        policies.insert(
            MetricKey::new("Latency", "ms"),
            MetricComparisonPolicy::lower("latency-v1", 1),
        );

        let report = compare_snapshots(&baseline, &current, &policies, thresholds()).unwrap();
        assert_eq!(report.results[0].disposition, ComparisonDisposition::Pass);
        assert!(!report.has_blocking_integrity_failure());
    }

    #[test]
    fn higher_is_better_drop_below_ci_is_critical() {
        let baseline = snapshot("baseline", &[("Accuracy", "score", metric(0.90, 0.88, 0.92))]);
        let current = snapshot("current", &[("Accuracy", "score", metric(0.75, 0.73, 0.77))]);
        let mut policies = MetricPolicyRegistry::new();
        policies.insert(
            MetricKey::new("Accuracy", "score"),
            MetricComparisonPolicy::higher("accuracy-v1", 1),
        );

        let report = compare_snapshots(&baseline, &current, &policies, thresholds()).unwrap();
        assert_eq!(report.results[0].disposition, ComparisonDisposition::Critical);
        assert!(report.has_blocking_integrity_failure());
    }

    #[test]
    fn two_sided_stability_flags_movement_in_either_direction() {
        for current_mean in [0.70, 1.30] {
            let baseline = snapshot("baseline", &[("Stable", "value", metric(1.0, 0.95, 1.05))]);
            let current = snapshot(
                "current",
                &[("Stable", "value", metric(current_mean, current_mean - 0.01, current_mean + 0.01))],
            );
            let mut policies = MetricPolicyRegistry::new();
            policies.insert(
                MetricKey::new("Stable", "value"),
                MetricComparisonPolicy::two_sided("stable-v1", 1, 0.10, 0.20),
            );

            let report = compare_snapshots(&baseline, &current, &policies, thresholds()).unwrap();
            assert_eq!(report.results[0].disposition, ComparisonDisposition::Critical);
        }
    }

    #[test]
    fn missing_current_benchmark_and_metric_are_both_accounted() {
        let baseline = snapshot(
            "baseline",
            &[
                ("A", "x", metric(1.0, 0.9, 1.1)),
                ("B", "y", metric(1.0, 0.9, 1.1)),
            ],
        );
        let current = snapshot("current", &[("B", "other", metric(1.0, 0.9, 1.1))]);
        let policies = MetricPolicyRegistry::new();

        let report = compare_snapshots(&baseline, &current, &policies, thresholds()).unwrap();
        assert_eq!(report.summary.total_required, 2);
        assert_eq!(report.summary.missing, 2);
        assert!(report.results.iter().any(|r| {
            r.key.benchmark == "A"
                && r.disposition == ComparisonDisposition::MissingCurrentBenchmark
        }));
        assert!(report.results.iter().any(|r| {
            r.key.benchmark == "B" && r.disposition == ComparisonDisposition::MissingCurrentMetric
        }));
    }

    #[test]
    fn extra_current_metric_is_recorded_without_becoming_failure() {
        let baseline = snapshot("baseline", &[("A", "x", metric(1.0, 0.9, 1.1))]);
        let current = snapshot(
            "current",
            &[
                ("A", "x", metric(1.0, 0.9, 1.1)),
                ("A", "new", metric(5.0, 4.9, 5.1)),
            ],
        );
        let mut policies = MetricPolicyRegistry::new();
        policies.insert(
            MetricKey::new("A", "x"),
            MetricComparisonPolicy::higher("a-x-v1", 1),
        );

        let report = compare_snapshots(&baseline, &current, &policies, thresholds()).unwrap();
        assert_eq!(report.summary.extra_current, 1);
        assert_eq!(report.extra_current_metrics, vec![MetricKey::new("A", "new")]);
        assert!(!report.has_blocking_integrity_failure());
    }

    #[test]
    fn near_zero_relative_baseline_is_incomparable_not_pass() {
        for policy in [
            MetricComparisonPolicy::higher("zero-higher", 1),
            MetricComparisonPolicy::lower("zero-lower", 1),
        ] {
            let baseline = snapshot("baseline", &[("A", "x", metric(0.0, -0.01, 0.01))]);
            let current = snapshot("current", &[("A", "x", metric(0.1, 0.09, 0.11))]);
            let mut policies = MetricPolicyRegistry::new();
            policies.insert(MetricKey::new("A", "x"), policy);

            let report = compare_snapshots(&baseline, &current, &policies, thresholds()).unwrap();
            assert_eq!(
                report.results[0].disposition,
                ComparisonDisposition::Incomparable
            );
            assert!(report.has_blocking_integrity_failure());
        }
    }

    #[test]
    fn invalid_global_thresholds_are_rejected() {
        for (warning, critical) in [
            (-0.01, 0.10),
            (0.10, 0.05),
            (f64::NAN, 0.10),
            (0.05, f64::INFINITY),
        ] {
            assert!(RegressionThresholds::new(warning, critical).is_err());
        }
    }

    #[test]
    fn snapshot_schema_mismatch_is_rejected_before_numeric_comparison() {
        let baseline = snapshot("baseline", &[("A", "x", metric(1.0, 0.9, 1.1))]);
        let mut current = snapshot("current", &[("A", "x", metric(1.0, 0.9, 1.1))]);
        current.schema_version = "future-schema".to_string();

        let err = compare_snapshots(
            &baseline,
            &current,
            &MetricPolicyRegistry::new(),
            thresholds(),
        )
        .expect_err("schema mismatch must fail before comparison");
        assert!(matches!(err, ComparisonSetupError::SnapshotSchemaMismatch { .. }));
    }

    #[test]
    fn output_preserves_exact_policy_identity_and_version() {
        let baseline = snapshot("baseline", &[("A", "x", metric(1.0, 0.9, 1.1))]);
        let current = snapshot("current", &[("A", "x", metric(1.0, 0.9, 1.1))]);
        let policy = MetricComparisonPolicy::higher("policy-A/x", 7);
        let mut policies = MetricPolicyRegistry::new();
        policies.insert(MetricKey::new("A", "x"), policy.clone());

        let report = compare_snapshots(&baseline, &current, &policies, thresholds()).unwrap();
        assert_eq!(report.results[0].policy.as_ref(), Some(&policy));
    }

    #[test]
    fn target_range_is_explicit_and_direction_free() {
        let baseline = snapshot("baseline", &[("Calibration", "ece", metric(0.10, 0.08, 0.12))]);
        let policy = MetricComparisonPolicy::target_range("ece-range-v1", 1, 0.05, 0.15);
        let mut policies = MetricPolicyRegistry::new();
        policies.insert(MetricKey::new("Calibration", "ece"), policy);

        let inside = snapshot("inside", &[("Calibration", "ece", metric(0.12, 0.10, 0.14))]);
        let outside = snapshot("outside", &[("Calibration", "ece", metric(0.25, 0.23, 0.27))]);

        let inside_report = compare_snapshots(&baseline, &inside, &policies, thresholds()).unwrap();
        let outside_report = compare_snapshots(&baseline, &outside, &policies, thresholds()).unwrap();
        assert_eq!(inside_report.results[0].disposition, ComparisonDisposition::Pass);
        assert_eq!(
            outside_report.results[0].disposition,
            ComparisonDisposition::Critical
        );
    }

    #[test]
    fn missing_policy_is_visible_and_fail_closed() {
        let baseline = snapshot("baseline", &[("A", "x", metric(1.0, 0.9, 1.1))]);
        let current = snapshot("current", &[("A", "x", metric(1.0, 0.9, 1.1))]);

        let report = compare_snapshots(
            &baseline,
            &current,
            &MetricPolicyRegistry::new(),
            thresholds(),
        )
        .unwrap();
        assert_eq!(
            report.results[0].disposition,
            ComparisonDisposition::PolicyMismatch
        );
        assert_eq!(report.summary.policy_failure, 1);
        assert!(report.has_blocking_integrity_failure());
    }

    #[test]
    fn invalid_current_statistics_remain_visible() {
        let baseline = snapshot("baseline", &[("A", "x", metric(1.0, 0.9, 1.1))]);
        let current = snapshot("current", &[("A", "x", metric(f64::NAN, 0.9, 1.1))]);
        let mut policies = MetricPolicyRegistry::new();
        policies.insert(
            MetricKey::new("A", "x"),
            MetricComparisonPolicy::higher("a-x-v1", 1),
        );

        let report = compare_snapshots(&baseline, &current, &policies, thresholds()).unwrap();
        assert_eq!(
            report.results[0].disposition,
            ComparisonDisposition::InvalidCurrent
        );
        assert_eq!(report.summary.invalid, 1);
    }
}
