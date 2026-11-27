// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned, adjudication-only selective-prediction evaluation.
//!
//! Risk is measured only against independently adjudicated truth. Pending cases
//! and adjudicated cases with invalid confidence values are counted but excluded.
//! Equal confidence values enter the coverage set atomically: no ordering inside
//! a tie can change risk, coverage, or AURC.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

pub const RISK_COVERAGE_METRIC_VERSION: &str = "symthaea-risk-coverage-v1";
const WILSON_Z_95: f64 = 1.959_963_984_540_054;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectivePredictionRecord {
    /// Stable, unique ID for this prediction in the frozen evaluation cohort.
    pub id: String,
    /// Binary label predicted by the candidate.
    pub predicted_label: bool,
    /// Confidence in the predicted label, not confidence that the record is correct.
    pub predicted_confidence: f64,
    /// Independently adjudicated ground truth. None is pending/censored.
    pub adjudicated_truth: Option<bool>,
    /// Whether the candidate actually abstained on this case.
    pub model_abstained: bool,
    /// Candidate-reported reason; used only when model_abstained is true.
    pub abstention_reason: Option<String>,
    pub domain: Option<String>,
    pub source_family: Option<String>,
    /// None remains an explicit unknown stratum.
    pub delayed_or_censored: Option<bool>,
    pub adjudicator: Option<String>,
}

impl SelectivePredictionRecord {
    /// Bridge for claim-correctness records such as VerificationOutcome.
    ///
    /// Existing claim records assert a positive proposition, so correctness is
    /// equivalent to the proposition's adjudicated truth. General binary tasks
    /// should construct the record with their actual predicted and true labels.
    pub fn from_claim_correctness(
        id: impl Into<String>,
        predicted_confidence: f64,
        was_correct: Option<bool>,
    ) -> Self {
        Self {
            id: id.into(),
            predicted_label: true,
            predicted_confidence,
            adjudicated_truth: was_correct,
            model_abstained: false,
            abstention_reason: None,
            domain: None,
            source_family: None,
            delayed_or_censored: None,
            adjudicator: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DecisionCosts {
    /// Cost of asserting a proposition that is false.
    pub false_positive: f64,
    /// Cost of rejecting a proposition that is true.
    pub false_negative: f64,
    /// Cost of withholding an answer.
    pub abstention: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskCoverageConfig {
    /// Scores are withheld when the eligible cohort or selected subset is smaller.
    pub minimum_sample_size: usize,
    /// Requested coverage levels. Tied confidence values may make achieved coverage higher.
    pub target_coverages: Vec<f64>,
    /// Inclusive confidence thresholds used as simple policy baselines.
    pub confidence_thresholds: Vec<f64>,
    /// None disables decision-cost reporting rather than inventing a utility scale.
    pub decision_costs: Option<DecisionCosts>,
    pub include_stratified_reports: bool,
}

impl Default for RiskCoverageConfig {
    fn default() -> Self {
        Self {
            minimum_sample_size: 20,
            target_coverages: vec![0.50, 0.80, 0.90],
            confidence_thresholds: vec![0.50, 0.70, 0.90],
            decision_costs: None,
            include_stratified_reports: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RiskCoverageError {
    EmptyRecordId,
    DuplicateRecordId(String),
    ZeroMinimumSampleSize,
    InvalidCoverageTarget(f64),
    DuplicateCoverageTarget(f64),
    InvalidConfidenceThreshold(f64),
    DuplicateConfidenceThreshold(f64),
    InvalidDecisionCost(&'static str, f64),
}

impl std::fmt::Display for RiskCoverageError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRecordId => write!(formatter, "record IDs must not be empty"),
            Self::DuplicateRecordId(id) => write!(formatter, "duplicate record ID: {id}"),
            Self::ZeroMinimumSampleSize => write!(formatter, "minimum sample size must be positive"),
            Self::InvalidCoverageTarget(value) => {
                write!(formatter, "coverage target must be finite and in (0, 1]: {value}")
            }
            Self::DuplicateCoverageTarget(value) => {
                write!(formatter, "duplicate coverage target: {value}")
            }
            Self::InvalidConfidenceThreshold(value) => {
                write!(formatter, "confidence threshold must be finite and in [0, 1]: {value}")
            }
            Self::DuplicateConfidenceThreshold(value) => {
                write!(formatter, "duplicate confidence threshold: {value}")
            }
            Self::InvalidDecisionCost(name, value) => {
                write!(formatter, "{name} decision cost must be finite and nonnegative: {value}")
            }
        }
    }
}

impl std::error::Error for RiskCoverageError {}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WilsonInterval {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskCoveragePoint {
    /// Inclusive confidence cutoff for this complete tie group.
    pub confidence_cutoff: f64,
    pub accepted_count: usize,
    pub error_count: usize,
    pub coverage: f64,
    /// Withheld when this selected set is below minimum_sample_size.
    pub risk: Option<f64>,
    pub risk_ci95: Option<WilsonInterval>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicySummary {
    pub accepted_count: usize,
    pub abstained_count: usize,
    pub coverage: f64,
    pub error_count: usize,
    pub risk: Option<f64>,
    pub risk_ci95: Option<WilsonInterval>,
    /// Mean per-original-case cost; present only when costs were supplied.
    pub mean_decision_cost: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageOperatingPoint {
    pub target_coverage: f64,
    /// Achieved coverage can exceed target because ties are never split.
    pub achieved_coverage: f64,
    pub accepted_count: usize,
    pub error_count: usize,
    pub risk: Option<f64>,
    pub risk_ci95: Option<WilsonInterval>,
    /// Expected risk for a random subset at the requested target coverage.
    pub random_abstention_expected_risk: Option<f64>,
    pub random_abstention_risk_ci95: Option<WilsonInterval>,
    pub mean_decision_cost: Option<f64>,
    pub random_abstention_expected_cost: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceThresholdPoint {
    pub threshold: f64,
    pub policy: PolicySummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstentionReasonCount {
    pub reason: String,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservedPolicySummary {
    pub policy: PolicySummary,
    pub abstention_reasons: Vec<AbstentionReasonCount>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreRiskCoverageReport {
    pub total_records: usize,
    pub pending_records: usize,
    pub invalid_confidence_records: usize,
    pub evaluated_records: usize,
    pub sample_adequate: bool,
    /// Right-step AURC over complete confidence-tie groups. None when n is inadequate.
    pub aurc: Option<f64>,
    pub curve: Vec<RiskCoveragePoint>,
    pub operating_points: Vec<CoverageOperatingPoint>,
    pub confidence_threshold_baselines: Vec<ConfidenceThresholdPoint>,
    pub always_answer_baseline: Option<PolicySummary>,
    pub observed_model_policy: Option<ObservedPolicySummary>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StratificationDimension {
    Domain,
    SourceFamily,
    DelayedOrCensored,
    Adjudicator,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratifiedRiskCoverageReport {
    pub dimension: StratificationDimension,
    pub value: String,
    pub report: CoreRiskCoverageReport,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskCoverageReport {
    pub metric_version: String,
    pub confidence_interval_method: String,
    pub total_records: usize,
    pub report: CoreRiskCoverageReport,
    pub stratified: Vec<StratifiedRiskCoverageReport>,
}

/// Evaluate a frozen cohort. Only valid-probability, independently adjudicated
/// outcomes contribute to risk, AURC, operating points, or confidence intervals.
pub fn evaluate_risk_coverage(
    records: &[SelectivePredictionRecord],
    config: &RiskCoverageConfig,
) -> Result<RiskCoverageReport, RiskCoverageError> {
    validate_config(config)?;
    let mut ids = HashSet::with_capacity(records.len());
    for record in records {
        if record.id.trim().is_empty() {
            return Err(RiskCoverageError::EmptyRecordId);
        }
        if !ids.insert(record.id.as_str()) {
            return Err(RiskCoverageError::DuplicateRecordId(record.id.clone()));
        }
    }

    let all: Vec<&SelectivePredictionRecord> = records.iter().collect();
    let report = summarize(&all, config);
    let stratified = if config.include_stratified_reports {
        stratify(records, config)
    } else {
        Vec::new()
    };

    Ok(RiskCoverageReport {
        metric_version: RISK_COVERAGE_METRIC_VERSION.to_string(),
        confidence_interval_method: "Wilson score interval, two-sided 95%".to_string(),
        total_records: records.len(),
        report,
        stratified,
    })
}

fn validate_config(config: &RiskCoverageConfig) -> Result<(), RiskCoverageError> {
    if config.minimum_sample_size == 0 {
        return Err(RiskCoverageError::ZeroMinimumSampleSize);
    }
    let mut targets = HashSet::new();
    for &target in &config.target_coverages {
        if !target.is_finite() || target <= 0.0 || target > 1.0 {
            return Err(RiskCoverageError::InvalidCoverageTarget(target));
        }
        if !targets.insert(target.to_bits()) {
            return Err(RiskCoverageError::DuplicateCoverageTarget(target));
        }
    }
    let mut thresholds = HashSet::new();
    for &threshold in &config.confidence_thresholds {
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err(RiskCoverageError::InvalidConfidenceThreshold(threshold));
        }
        if !thresholds.insert(threshold.to_bits()) {
            return Err(RiskCoverageError::DuplicateConfidenceThreshold(threshold));
        }
    }
    if let Some(costs) = config.decision_costs {
        for (name, value) in [
            ("false_positive", costs.false_positive),
            ("false_negative", costs.false_negative),
            ("abstention", costs.abstention),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(RiskCoverageError::InvalidDecisionCost(name, value));
            }
        }
    }
    Ok(())
}

fn summarize(
    records: &[&SelectivePredictionRecord],
    config: &RiskCoverageConfig,
) -> CoreRiskCoverageReport {
    let mut pending_records = 0;
    let mut invalid_confidence_records = 0;
    let mut eligible = Vec::new();

    for record in records {
        if record.adjudicated_truth.is_none() {
            pending_records += 1;
            continue;
        }
        if !record.predicted_confidence.is_finite()
            || !(0.0..=1.0).contains(&record.predicted_confidence)
        {
            invalid_confidence_records += 1;
            continue;
        }
        eligible.push(*record);
    }

    let evaluated_records = eligible.len();
    let sample_adequate = evaluated_records >= config.minimum_sample_size;
    let mut report = CoreRiskCoverageReport {
        total_records: records.len(),
        pending_records,
        invalid_confidence_records,
        evaluated_records,
        sample_adequate,
        aurc: None,
        curve: Vec::new(),
        operating_points: Vec::new(),
        confidence_threshold_baselines: Vec::new(),
        always_answer_baseline: None,
        observed_model_policy: None,
    };

    if !sample_adequate {
        return report;
    }

    eligible.sort_by(|left, right| {
        right
            .predicted_confidence
            .total_cmp(&left.predicted_confidence)
            .then_with(|| left.id.cmp(&right.id))
    });

    let total = eligible.len();
    let overall_errors = eligible
        .iter()
        .filter(|record| record.predicted_label != record.adjudicated_truth.unwrap())
        .count();
    let overall_risk = overall_errors as f64 / total as f64;
    let overall_ci = wilson_interval(overall_errors, total);

    report.always_answer_baseline = Some(PolicySummary {
        accepted_count: total,
        abstained_count: 0,
        coverage: 1.0,
        error_count: overall_errors,
        risk: Some(overall_risk),
        risk_ci95: overall_ci,
        mean_decision_cost: decision_cost(&eligible, &eligible, config.decision_costs),
    });

    let observed_accepted: Vec<&SelectivePredictionRecord> = eligible
        .iter()
        .copied()
        .filter(|record| !record.model_abstained)
        .collect();
    let observed_abstained = total - observed_accepted.len();
    let observed_errors = count_errors(&observed_accepted);
    let observed = policy_summary(
        &eligible,
        &observed_accepted,
        observed_errors,
        config,
    );
    let mut reason_counts = BTreeMap::<String, usize>::new();
    for record in &eligible {
        if record.model_abstained {
            let reason = record
                .abstention_reason
                .as_deref()
                .filter(|reason| !reason.trim().is_empty())
                .unwrap_or("unspecified");
            *reason_counts.entry(reason.to_string()).or_default() += 1;
        }
    }
    report.observed_model_policy = Some(ObservedPolicySummary {
        policy: PolicySummary {
            abstained_count: observed_abstained,
            ..observed
        },
        abstention_reasons: reason_counts
            .into_iter()
            .map(|(reason, count)| AbstentionReasonCount { reason, count })
            .collect(),
    });

    // Build one point per distinct confidence. A tie group is admitted in full.
    // AURC uses a right-step integral: each newly admitted coverage interval is
    // charged the risk of the resulting accepted set.
    let mut index = 0;
    let mut cumulative_errors = 0;
    let mut area = 0.0_f64;
    while index < total {
        let cutoff = eligible[index].predicted_confidence;
        let mut end = index;
        let mut group_errors = 0;
        while end < total && eligible[end].predicted_confidence == cutoff {
            let record = eligible[end];
            if record.predicted_label != record.adjudicated_truth.unwrap() {
                group_errors += 1;
            }
            end += 1;
        }
        cumulative_errors += group_errors;
        let accepted_count = end;
        let coverage = accepted_count as f64 / total as f64;
        let risk = cumulative_errors as f64 / accepted_count as f64;
        let newly_added_coverage = (end - index) as f64 / total as f64;
        area += newly_added_coverage * risk;
        report.curve.push(RiskCoveragePoint {
            confidence_cutoff: cutoff,
            accepted_count,
            error_count: cumulative_errors,
            coverage,
            risk: (accepted_count >= config.minimum_sample_size).then_some(risk),
            risk_ci95: (accepted_count >= config.minimum_sample_size)
                .then(|| wilson_interval(cumulative_errors, accepted_count))
                .flatten(),
        });
        index = end;
    }
    report.aurc = Some(area);

    let mut targets = config.target_coverages.clone();
    targets.sort_by(f64::total_cmp);
    for target in targets {
        // Curve points are tie-group boundaries; choosing the first one at or
        // above target avoids arbitrary ordering among equal-confidence cases.
        let Some(point) = report.curve.iter().find(|point| point.coverage >= target) else {
            continue;
        };
        let accepted: Vec<&SelectivePredictionRecord> =
            eligible.iter().take(point.accepted_count).copied().collect();
        let error_count = count_errors(&accepted);
        let decision_cost = decision_cost(&eligible, &accepted, config.decision_costs);
        let random_cost = random_abstention_cost(
            &eligible,
            target,
            config.decision_costs,
        );
        report.operating_points.push(CoverageOperatingPoint {
            target_coverage: target,
            achieved_coverage: point.coverage,
            accepted_count: point.accepted_count,
            error_count,
            risk: (point.accepted_count >= config.minimum_sample_size)
                .then_some(error_count as f64 / point.accepted_count as f64),
            risk_ci95: (point.accepted_count >= config.minimum_sample_size)
                .then(|| wilson_interval(error_count, point.accepted_count))
                .flatten(),
            random_abstention_expected_risk: Some(overall_risk),
            random_abstention_risk_ci95: overall_ci,
            mean_decision_cost: decision_cost,
            random_abstention_expected_cost: random_cost,
        });
    }

    let mut thresholds = config.confidence_thresholds.clone();
    thresholds.sort_by(f64::total_cmp);
    for threshold in thresholds {
        let accepted: Vec<&SelectivePredictionRecord> = eligible
            .iter()
            .copied()
            .filter(|record| record.predicted_confidence >= threshold)
            .collect();
        let errors = count_errors(&accepted);
        report.confidence_threshold_baselines.push(ConfidenceThresholdPoint {
            threshold,
            policy: policy_summary(&eligible, &accepted, errors, config),
        });
    }

    report
}

fn policy_summary(
    all: &[&SelectivePredictionRecord],
    accepted: &[&SelectivePredictionRecord],
    errors: usize,
    config: &RiskCoverageConfig,
) -> PolicySummary {
    let accepted_count = accepted.len();
    let enough = accepted_count >= config.minimum_sample_size;
    PolicySummary {
        accepted_count,
        abstained_count: all.len() - accepted_count,
        coverage: if all.is_empty() {
            0.0
        } else {
            accepted_count as f64 / all.len() as f64
        },
        error_count: errors,
        risk: enough.then_some(errors as f64 / accepted_count as f64),
        risk_ci95: enough.then(|| wilson_interval(errors, accepted_count)).flatten(),
        mean_decision_cost: decision_cost(all, accepted, config.decision_costs),
    }
}

fn count_errors(records: &[&SelectivePredictionRecord]) -> usize {
    records
        .iter()
        .filter(|record| record.predicted_label != record.adjudicated_truth.unwrap())
        .count()
}

fn decision_cost(
    all: &[&SelectivePredictionRecord],
    accepted: &[&SelectivePredictionRecord],
    costs: Option<DecisionCosts>,
) -> Option<f64> {
    let costs = costs?;
    if all.is_empty() {
        return None;
    }
    let mut false_positives = 0;
    let mut false_negatives = 0;
    for record in accepted {
        match (record.predicted_label, record.adjudicated_truth.unwrap()) {
            (true, false) => false_positives += 1,
            (false, true) => false_negatives += 1,
            _ => {}
        }
    }
    let abstained = all.len() - accepted.len();
    let total_cost = false_positives as f64 * costs.false_positive
        + false_negatives as f64 * costs.false_negative
        + abstained as f64 * costs.abstention;
    Some(total_cost / all.len() as f64)
}

fn random_abstention_cost(
    all: &[&SelectivePredictionRecord],
    target_coverage: f64,
    costs: Option<DecisionCosts>,
) -> Option<f64> {
    let costs = costs?;
    if all.is_empty() {
        return None;
    }
    let mut false_positives = 0;
    let mut false_negatives = 0;
    for record in all {
        match (record.predicted_label, record.adjudicated_truth.unwrap()) {
            (true, false) => false_positives += 1,
            (false, true) => false_negatives += 1,
            _ => {}
        }
    }
    let mean_answer_error_cost = (false_positives as f64 * costs.false_positive
        + false_negatives as f64 * costs.false_negative)
        / all.len() as f64;
    Some(
        target_coverage * mean_answer_error_cost
            + (1.0 - target_coverage) * costs.abstention,
    )
}

fn wilson_interval(errors: usize, sample_size: usize) -> Option<WilsonInterval> {
    if sample_size == 0 {
        return None;
    }
    let n = sample_size as f64;
    let proportion = errors as f64 / n;
    let z_squared = WILSON_Z_95 * WILSON_Z_95;
    let denominator = 1.0 + z_squared / n;
    let center = (proportion + z_squared / (2.0 * n)) / denominator;
    let half_width = WILSON_Z_95
        * ((proportion * (1.0 - proportion) / n
            + z_squared / (4.0 * n * n))
            .sqrt())
        / denominator;
    Some(WilsonInterval {
        lower: (center - half_width).max(0.0),
        upper: (center + half_width).min(1.0),
    })
}

fn stratify(
    records: &[SelectivePredictionRecord],
    config: &RiskCoverageConfig,
) -> Vec<StratifiedRiskCoverageReport> {
    let dimensions = [
        StratificationDimension::Domain,
        StratificationDimension::SourceFamily,
        StratificationDimension::DelayedOrCensored,
        StratificationDimension::Adjudicator,
    ];
    let mut reports = Vec::new();
    for dimension in dimensions {
        let mut groups = BTreeMap::<String, Vec<&SelectivePredictionRecord>>::new();
        for record in records {
            let value = match dimension {
                StratificationDimension::Domain => record.domain.as_deref().unwrap_or("unknown").to_string(),
                StratificationDimension::SourceFamily => record.source_family.as_deref().unwrap_or("unknown").to_string(),
                StratificationDimension::DelayedOrCensored => match record.delayed_or_censored {
                    Some(true) => "delayed_or_censored".to_string(),
                    Some(false) => "not_delayed_or_censored".to_string(),
                    None => "unknown".to_string(),
                },
                StratificationDimension::Adjudicator => record.adjudicator.as_deref().unwrap_or("unknown").to_string(),
            };
            groups.entry(value).or_default().push(record);
        }
        for (value, group) in groups {
            reports.push(StratifiedRiskCoverageReport {
                dimension,
                value,
                report: summarize(&group, config),
            });
        }
    }
    reports
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> RiskCoverageConfig {
        RiskCoverageConfig {
            minimum_sample_size: 1,
            target_coverages: vec![0.5, 0.8],
            confidence_thresholds: vec![0.5, 0.9],
            decision_costs: None,
            include_stratified_reports: true,
        }
    }

    fn record(id: &str, confidence: f64, predicted: bool, truth: Option<bool>) -> SelectivePredictionRecord {
        SelectivePredictionRecord {
            id: id.to_string(),
            predicted_label: predicted,
            predicted_confidence: confidence,
            adjudicated_truth: truth,
            model_abstained: false,
            abstention_reason: None,
            domain: Some("science".to_string()),
            source_family: Some("journal-a".to_string()),
            delayed_or_censored: Some(false),
            adjudicator: Some("expert-1".to_string()),
        }
    }

    #[test]
    fn pending_and_invalid_confidence_records_never_enter_metrics() {
        let mut invalid = record("invalid", 1.2, true, Some(true));
        let pending = record("pending", 0.99, true, None);
        let valid = record("valid", 0.8, true, Some(false));
        invalid.predicted_confidence = f64::NAN;
        let report = evaluate_risk_coverage(&[invalid, pending, valid], &config()).unwrap();
        assert_eq!(report.report.pending_records, 1);
        assert_eq!(report.report.invalid_confidence_records, 1);
        assert_eq!(report.report.evaluated_records, 1);
        assert_eq!(report.report.aurc, Some(1.0));
    }

    #[test]
    fn equal_confidence_values_are_admitted_as_one_atomic_group() {
        let records = vec![
            record("a", 0.9, true, Some(true)),
            record("b", 0.9, true, Some(false)),
            record("c", 0.1, true, Some(true)),
        ];
        let report = evaluate_risk_coverage(&records, &config()).unwrap();
        let point = report.report.operating_points[0].clone();
        assert_eq!(point.accepted_count, 2);
        assert!((point.achieved_coverage - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(point.risk, Some(0.5));
        assert!((report.report.aurc.unwrap() - 4.0 / 9.0).abs() < 1e-12);
    }

    #[test]
    fn all_correct_cohort_has_zero_risk_and_zero_aurc() {
        let records = vec![
            record("a", 0.2, true, Some(true)),
            record("b", 0.9, false, Some(false)),
            record("c", 0.5, true, Some(true)),
        ];
        let report = evaluate_risk_coverage(&records, &config()).unwrap();
        assert_eq!(report.report.aurc, Some(0.0));
        assert!(report.report.curve.iter().all(|point| point.risk == Some(0.0)));
    }

    #[test]
    fn all_wrong_cohort_has_unit_risk_and_unit_aurc() {
        let records = vec![
            record("a", 0.2, true, Some(false)),
            record("b", 0.9, false, Some(true)),
            record("c", 0.5, true, Some(false)),
        ];
        let report = evaluate_risk_coverage(&records, &config()).unwrap();
        assert_eq!(report.report.aurc, Some(1.0));
        assert!(report.report.curve.iter().all(|point| point.risk == Some(1.0)));
    }

    #[test]
    fn empty_cohort_reports_counts_but_withholds_scores() {
        let report = evaluate_risk_coverage(&[], &config()).unwrap();
        assert_eq!(report.report.evaluated_records, 0);
        assert!(!report.report.sample_adequate);
        assert_eq!(report.report.aurc, None);
        assert!(report.report.curve.is_empty());
        assert!(report.report.always_answer_baseline.is_none());
    }

    #[test]
    fn single_sample_cohort_is_evaluated_when_minimum_is_one() {
        let records = vec![record("one", 0.7, true, Some(false))];
        let report = evaluate_risk_coverage(&records, &config()).unwrap();
        assert_eq!(report.report.aurc, Some(1.0));
        let baseline = report.report.always_answer_baseline.as_ref().unwrap();
        assert_eq!(baseline.risk, Some(1.0));
        assert!(baseline.risk_ci95.unwrap().lower <= 1.0);
        assert!(baseline.risk_ci95.unwrap().upper >= 1.0);
    }

    #[test]
    fn minimum_sample_gate_withholds_all_scores_below_threshold() {
        let mut cfg = config();
        cfg.minimum_sample_size = 3;
        let records = vec![
            record("a", 0.9, true, Some(true)),
            record("b", 0.8, true, Some(false)),
        ];
        let report = evaluate_risk_coverage(&records, &cfg).unwrap();
        assert!(!report.report.sample_adequate);
        assert_eq!(report.report.aurc, None);
        assert!(report.report.operating_points.is_empty());
    }

    #[test]
    fn asymmetric_false_positive_false_negative_and_abstention_costs_are_respected() {
        let mut cfg = config();
        cfg.target_coverages = vec![0.3];
        cfg.decision_costs = Some(DecisionCosts {
            false_positive: 5.0,
            false_negative: 2.0,
            abstention: 0.5,
        });
        let records = vec![
            record("fp", 0.9, true, Some(false)),
            record("fn", 0.8, false, Some(true)),
            record("ok", 0.7, true, Some(true)),
        ];
        let report = evaluate_risk_coverage(&records, &cfg).unwrap();
        let point = &report.report.operating_points[0];
        assert!((point.mean_decision_cost.unwrap() - 2.0).abs() < 1e-12);
        assert!((point.random_abstention_expected_cost.unwrap() - 1.05).abs() < 1e-12);
    }

    #[test]
    fn observed_abstention_reasons_are_counted_without_treating_pending_as_errors() {
        let mut abstained = record("a", 0.4, true, Some(true));
        abstained.model_abstained = true;
        abstained.abstention_reason = Some("insufficient_evidence".to_string());
        let mut unspecified = record("b", 0.3, true, Some(false));
        unspecified.model_abstained = true;
        let report = evaluate_risk_coverage(&[abstained, unspecified], &config()).unwrap();
        let observed = report.report.observed_model_policy.as_ref().unwrap();
        assert_eq!(observed.policy.accepted_count, 0);
        assert_eq!(observed.policy.abstained_count, 2);
        assert_eq!(observed.abstention_reasons.len(), 2);
        assert!(observed.policy.risk.is_none());
    }

    #[test]
    fn threshold_baseline_is_inclusive_and_keeps_ties() {
        let records = vec![
            record("a", 0.8, true, Some(true)),
            record("b", 0.8, true, Some(false)),
            record("c", 0.2, true, Some(true)),
        ];
        let report = evaluate_risk_coverage(&records, &config()).unwrap();
        let threshold = report
            .report
            .confidence_threshold_baselines
            .iter()
            .find(|point| point.threshold == 0.8)
            .unwrap();
        assert_eq!(threshold.policy.accepted_count, 2);
        assert_eq!(threshold.policy.risk, Some(0.5));
    }

    #[test]
    fn stratification_reports_unknown_values_and_each_required_dimension() {
        let mut first = record("a", 0.8, true, Some(true));
        first.domain = Some("physics".to_string());
        first.source_family = None;
        first.delayed_or_censored = None;
        first.adjudicator = Some("panel-a".to_string());
        let report = evaluate_risk_coverage(&[first], &config()).unwrap();
        assert_eq!(report.stratified.len(), 4);
        assert!(report.stratified.iter().any(|group| group.value == "physics"));
        assert!(report.stratified.iter().any(|group| group.value == "unknown"));
    }

    #[test]
    fn duplicate_ids_are_rejected_before_scoring() {
        let records = vec![
            record("same", 0.8, true, Some(true)),
            record("same", 0.7, true, Some(false)),
        ];
        assert_eq!(
            evaluate_risk_coverage(&records, &config()).unwrap_err(),
            RiskCoverageError::DuplicateRecordId("same".to_string())
        );
    }

    #[test]
    fn invalid_configurations_fail_closed() {
        let mut cfg = config();
        cfg.minimum_sample_size = 0;
        assert_eq!(
            evaluate_risk_coverage(&[], &cfg).unwrap_err(),
            RiskCoverageError::ZeroMinimumSampleSize
        );
        cfg = config();
        cfg.target_coverages = vec![1.1];
        assert!(matches!(
            evaluate_risk_coverage(&[], &cfg),
            Err(RiskCoverageError::InvalidCoverageTarget(_))
        ));
        cfg = config();
        cfg.decision_costs = Some(DecisionCosts {
            false_positive: f64::NAN,
            false_negative: 1.0,
            abstention: 0.0,
        });
        assert!(matches!(
            evaluate_risk_coverage(&[], &cfg),
            Err(RiskCoverageError::InvalidDecisionCost(_, _))
        ));
    }

    #[test]
    fn claim_correctness_bridge_preserves_pending_state() {
        let pending = SelectivePredictionRecord::from_claim_correctness("pending", 0.95, None);
        assert_eq!(pending.adjudicated_truth, None);
        assert!(pending.predicted_label);
        let adjudicated = SelectivePredictionRecord::from_claim_correctness("done", 0.95, Some(false));
        assert_eq!(adjudicated.adjudicated_truth, Some(false));
    }

    #[test]
    fn disabling_stratification_does_not_change_overall_metrics() {
        let records = vec![
            record("a", 0.9, true, Some(true)),
            record("b", 0.1, true, Some(false)),
        ];
        let mut cfg = config();
        let with_groups = evaluate_risk_coverage(&records, &cfg).unwrap();
        cfg.include_stratified_reports = false;
        let without_groups = evaluate_risk_coverage(&records, &cfg).unwrap();
        assert_eq!(with_groups.report.aurc, without_groups.report.aurc);
        assert!(without_groups.stratified.is_empty());
    }
}
