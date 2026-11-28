// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Design-based risk estimation for custodian-randomized stratified audits.
//!
//! This module composes a frozen candidate-prediction cohort with the
//! first-/second-order inclusion design from stratified_audit.rs. For fixed
//! acceptance sets it reports Horvitz-Thompson error totals and the standard
//! unbiased SRSWOR variance estimator. Normal intervals are explicitly
//! asymptotic pointwise intervals, not finite-sample or simultaneous guarantees.

use super::selective_prediction::{RiskCoverageConfig, SelectivePredictionRecord};
use super::stratified_audit::{STRATIFIED_AUDIT_DESIGN_VERSION, StratifiedAuditPlan};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

const NORMAL_95_Z: f64 = 1.959_963_984_540_054;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApproximateRiskInterval {
    pub lower: f64,
    pub upper: f64,
    pub confidence_level: f64,
    pub method: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StratumRiskEstimate {
    pub stratum_id: String,
    pub population_size: usize,
    pub audit_sample_size: usize,
    pub accepted_population_count: usize,
    pub accepted_audit_count: usize,
    pub observed_error_count: usize,
    pub estimated_error_total: f64,
    pub estimated_error_total_variance: Option<f64>,
    pub sample_error_variance: Option<f64>,
    pub variance_estimable: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignBasedRiskEstimate {
    pub acceptance_rule: String,
    pub accepted_population_count: usize,
    pub accepted_coverage: f64,
    pub accepted_audit_count: usize,
    pub estimated_error_total: f64,
    /// HT estimated error total divided by the exact accepted count in the
    /// frozen prediction cohort. Never clipped to [0, 1].
    pub estimated_risk: Option<f64>,
    pub estimated_risk_outside_unit_interval: bool,
    pub estimated_error_total_variance: Option<f64>,
    pub estimated_risk_variance: Option<f64>,
    pub standard_error: Option<f64>,
    /// Pointwise normal approximation only. Not finite-sample, anytime-valid
    /// or simultaneous across operating points/strata.
    pub approximate_normal_95_interval: Option<ApproximateRiskInterval>,
    pub variance_estimable: bool,
    pub variance_unavailable_strata: Vec<String>,
    pub uncertainty_unavailable_reason: Option<String>,
    pub strata: Vec<StratumRiskEstimate>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignBasedOperatingPoint {
    pub target_coverage: f64,
    pub achieved_coverage: f64,
    pub confidence_cutoff: f64,
    pub estimate: DesignBasedRiskEstimate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignBasedThresholdPoint {
    pub confidence_threshold: f64,
    pub estimate: DesignBasedRiskEstimate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignBasedCurvePoint {
    pub confidence_cutoff: f64,
    pub estimate: DesignBasedRiskEstimate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StratifiedRiskReport {
    pub metric_version: String,
    pub design_id: String,
    pub population_size: usize,
    pub audit_sample_size: usize,
    pub audit_fraction: f64,
    pub always_answer: DesignBasedRiskEstimate,
    pub observed_model_policy: DesignBasedRiskEstimate,
    pub operating_points: Vec<DesignBasedOperatingPoint>,
    pub confidence_threshold_baselines: Vec<DesignBasedThresholdPoint>,
    pub risk_coverage_curve: Vec<DesignBasedCurvePoint>,
    /// Tie-aware item-wise point estimate. No AURC uncertainty is reported
    /// because covariance across nested coverage prefixes is not yet included.
    pub aurc: Option<f64>,
    pub aurc_uncertainty_status: String,
    pub interval_scope: String,
    pub assumptions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StratifiedRiskError {
    UnsupportedDesignVersion,
    EmptyDesignIdentity,
    InvalidPlan(String),
    PredictionCountMismatch,
    EmptyPredictionId,
    DuplicatePredictionId(String),
    UnknownPredictionId(String),
    MissingPredictionId(String),
    InvalidPredictionConfidence(String),
    SelectedRecordMissingAdjudication(String),
    NonSelectedRecordHasAdjudication(String),
    NoAuditSample,
    InvalidCoverageTarget,
    InvalidConfidenceThreshold,
    ZeroMinimumSampleSize,
    DecisionCostsNotSupported,
}

impl std::fmt::Display for StratifiedRiskError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedDesignVersion => {
                write!(formatter, "unsupported stratified audit design version")
            }
            Self::EmptyDesignIdentity => write!(
                formatter,
                "audit design identity or randomization manifest reference is empty"
            ),
            Self::InvalidPlan(message) => {
                write!(formatter, "invalid stratified audit plan: {message}")
            }
            Self::PredictionCountMismatch => {
                write!(
                    formatter,
                    "prediction cohort size does not match the audit plan"
                )
            }
            Self::EmptyPredictionId => write!(formatter, "prediction record ID must not be empty"),
            Self::DuplicatePredictionId(id) => {
                write!(formatter, "duplicate prediction record ID: {id}")
            }
            Self::UnknownPredictionId(id) => {
                write!(formatter, "prediction record is not in audit plan: {id}")
            }
            Self::MissingPredictionId(id) => {
                write!(
                    formatter,
                    "audit-plan record has no candidate prediction: {id}"
                )
            }
            Self::InvalidPredictionConfidence(id) => {
                write!(
                    formatter,
                    "prediction confidence is invalid for record: {id}"
                )
            }
            Self::SelectedRecordMissingAdjudication(id) => {
                write!(
                    formatter,
                    "selected audit record has no independent adjudication: {id}"
                )
            }
            Self::NonSelectedRecordHasAdjudication(id) => {
                write!(formatter, "non-selected record has an audit label: {id}")
            }
            Self::NoAuditSample => write!(formatter, "audit plan contains no selected records"),
            Self::InvalidCoverageTarget => {
                write!(formatter, "coverage targets must be finite and in (0, 1]")
            }
            Self::InvalidConfidenceThreshold => {
                write!(
                    formatter,
                    "confidence thresholds must be finite and in [0, 1]"
                )
            }
            Self::ZeroMinimumSampleSize => {
                write!(formatter, "minimum audit sample size must be positive")
            }
            Self::DecisionCostsNotSupported => write!(
                formatter,
                "decision costs are not implemented in this design-based risk report"
            ),
        }
    }
}

impl std::error::Error for StratifiedRiskError {}

fn close_probability(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-12
}

fn validate_plan(plan: &StratifiedAuditPlan) -> Result<(), StratifiedRiskError> {
    if plan.design_version != STRATIFIED_AUDIT_DESIGN_VERSION {
        return Err(StratifiedRiskError::UnsupportedDesignVersion);
    }
    if plan.design_id.trim().is_empty() || plan.randomization_manifest_ref.trim().is_empty() {
        return Err(StratifiedRiskError::EmptyDesignIdentity);
    }
    if plan.population_size == 0 || plan.selections.len() != plan.population_size {
        return Err(StratifiedRiskError::InvalidPlan(
            "selection list must contain every unit in the nonempty cohort".to_owned(),
        ));
    }

    let mut stratum_by_id = HashMap::new();
    let mut population_sum = 0usize;
    let mut sample_sum = 0usize;
    for stratum in &plan.strata {
        if stratum.stratum_id.trim().is_empty()
            || stratum.population_size == 0
            || stratum.sample_size == 0
            || stratum.sample_size > stratum.population_size
            || stratum_by_id
                .insert(stratum.stratum_id.as_str(), stratum)
                .is_some()
        {
            return Err(StratifiedRiskError::InvalidPlan(
                "stratum IDs, population sizes, quotas or uniqueness are invalid".to_owned(),
            ));
        }
        let pi = stratum.sample_size as f64 / stratum.population_size as f64;
        let pi_pair = if stratum.sample_size < 2 || stratum.population_size < 2 {
            0.0
        } else {
            stratum.sample_size as f64 * (stratum.sample_size - 1) as f64
                / (stratum.population_size as f64 * (stratum.population_size - 1) as f64)
        };
        let variance_estimable = stratum.sample_size == stratum.population_size
            || stratum.sample_size >= 2
            || stratum.population_size == 1;
        if !close_probability(pi, stratum.first_order_inclusion_probability)
            || !close_probability(pi, stratum.sampling_fraction)
            || !close_probability(pi_pair, stratum.distinct_pair_joint_inclusion_probability)
            || variance_estimable != stratum.variance_estimable
        {
            return Err(StratifiedRiskError::InvalidPlan(format!(
                "inclusion probabilities or variance support disagree for stratum {}",
                stratum.stratum_id
            )));
        }
        population_sum += stratum.population_size;
        sample_sum += stratum.sample_size;
    }

    if population_sum != plan.population_size || sample_sum != plan.audit_sample_size {
        return Err(StratifiedRiskError::InvalidPlan(
            "stratum populations/quotas do not sum to plan totals".to_owned(),
        ));
    }

    let mut unit_ids = HashSet::with_capacity(plan.selections.len());
    let mut selected_by_stratum = BTreeMap::<String, usize>::new();
    let mut population_by_stratum = BTreeMap::<String, usize>::new();
    for selection in &plan.selections {
        let Some(stratum) = stratum_by_id.get(selection.stratum_id.as_str()) else {
            return Err(StratifiedRiskError::InvalidPlan(
                "selection references an unknown stratum".to_owned(),
            ));
        };
        if selection.record_id.trim().is_empty() || !unit_ids.insert(selection.record_id.as_str()) {
            return Err(StratifiedRiskError::InvalidPlan(
                "selection IDs must be nonempty and unique".to_owned(),
            ));
        }
        if !close_probability(
            selection.inclusion_probability,
            stratum.first_order_inclusion_probability,
        ) {
            return Err(StratifiedRiskError::InvalidPlan(format!(
                "unit inclusion probability disagrees for {}",
                selection.record_id
            )));
        }
        *population_by_stratum
            .entry(selection.stratum_id.clone())
            .or_default() += 1;
        if selection.selected_for_adjudication {
            *selected_by_stratum
                .entry(selection.stratum_id.clone())
                .or_default() += 1;
        }
    }
    for (stratum_id, stratum) in stratum_by_id {
        if population_by_stratum.get(stratum_id).copied().unwrap_or(0) != stratum.population_size
            || selected_by_stratum.get(stratum_id).copied().unwrap_or(0) != stratum.sample_size
        {
            return Err(StratifiedRiskError::InvalidPlan(format!(
                "selection counts disagree for stratum {stratum_id}"
            )));
        }
    }
    if plan.audit_sample_size == 0 {
        return Err(StratifiedRiskError::NoAuditSample);
    }
    Ok(())
}

/// Estimate accepted-prediction risk on the complete frozen cohort using the
/// custodian's stratified SRSWOR audit. The candidate prediction cohort must be
/// frozen before sampling; only selected audit records may carry audit labels.
///
/// For each fixed acceptance set, x_i is 1 if the candidate answers and is
/// wrong, otherwise 0. Within each stratum the HT total is N_h * mean_s(x).
/// The usual SRSWOR variance estimate is used when n_h >= 2 or the stratum is
/// a census. A one-unit non-census sample still yields a point estimate, but
/// the overall variance/interval is withheld.
pub fn evaluate_stratified_audit_risk(
    predictions: &[SelectivePredictionRecord],
    plan: &StratifiedAuditPlan,
    config: &RiskCoverageConfig,
) -> Result<StratifiedRiskReport, StratifiedRiskError> {
    validate_plan(plan)?;
    if config.minimum_sample_size == 0 {
        return Err(StratifiedRiskError::ZeroMinimumSampleSize);
    }
    if config.decision_costs.is_some() {
        return Err(StratifiedRiskError::DecisionCostsNotSupported);
    }
    if config
        .target_coverages
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0 || *value > 1.0)
    {
        return Err(StratifiedRiskError::InvalidCoverageTarget);
    }
    if config
        .confidence_thresholds
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(StratifiedRiskError::InvalidConfidenceThreshold);
    }
    if predictions.len() != plan.population_size {
        return Err(StratifiedRiskError::PredictionCountMismatch);
    }

    let mut prediction_by_id = HashMap::with_capacity(predictions.len());
    for prediction in predictions {
        if prediction.id.trim().is_empty() {
            return Err(StratifiedRiskError::EmptyPredictionId);
        }
        if prediction_by_id
            .insert(prediction.id.as_str(), prediction)
            .is_some()
        {
            return Err(StratifiedRiskError::DuplicatePredictionId(
                prediction.id.clone(),
            ));
        }
        if !prediction.predicted_confidence.is_finite()
            || !(0.0..=1.0).contains(&prediction.predicted_confidence)
        {
            return Err(StratifiedRiskError::InvalidPredictionConfidence(
                prediction.id.clone(),
            ));
        }
    }

    let plan_ids: HashSet<&str> = plan
        .selections
        .iter()
        .map(|selection| selection.record_id.as_str())
        .collect();
    for prediction in predictions {
        if !plan_ids.contains(prediction.id.as_str()) {
            return Err(StratifiedRiskError::UnknownPredictionId(
                prediction.id.clone(),
            ));
        }
    }
    for selection in &plan.selections {
        let Some(prediction) = prediction_by_id.get(selection.record_id.as_str()) else {
            return Err(StratifiedRiskError::MissingPredictionId(
                selection.record_id.clone(),
            ));
        };
        if selection.selected_for_adjudication && prediction.adjudicated_truth.is_none() {
            return Err(StratifiedRiskError::SelectedRecordMissingAdjudication(
                selection.record_id.clone(),
            ));
        }
        if !selection.selected_for_adjudication && prediction.adjudicated_truth.is_some() {
            return Err(StratifiedRiskError::NonSelectedRecordHasAdjudication(
                selection.record_id.clone(),
            ));
        }
    }

    let mut sorted: Vec<&SelectivePredictionRecord> = predictions.iter().collect();
    sorted.sort_by(|left, right| {
        right
            .predicted_confidence
            .total_cmp(&left.predicted_confidence)
            .then_with(|| left.id.cmp(&right.id))
    });

    let all_ids: HashSet<&str> = sorted.iter().map(|row| row.id.as_str()).collect();
    let always_answer = estimate_from_stratum_counts(
        "always_answer",
        &counts_for_acceptance(&all_ids, predictions, plan),
        plan,
        config.minimum_sample_size,
    );
    let observed_ids: HashSet<&str> = sorted
        .iter()
        .filter(|row| !row.model_abstained)
        .map(|row| row.id.as_str())
        .collect();
    let observed_model_policy = estimate_from_stratum_counts(
        "observed_model_policy",
        &counts_for_acceptance(&observed_ids, predictions, plan),
        plan,
        config.minimum_sample_size,
    );

    let stratum_index_by_id: HashMap<&str, usize> = plan
        .strata
        .iter()
        .enumerate()
        .map(|(index, stratum)| (stratum.stratum_id.as_str(), index))
        .collect();
    let selection_by_id: HashMap<&str, (usize, bool)> = plan
        .selections
        .iter()
        .map(|selection| {
            (
                selection.record_id.as_str(),
                (
                    stratum_index_by_id[selection.stratum_id.as_str()],
                    selection.selected_for_adjudication,
                ),
            )
        })
        .collect();

    let mut curve = Vec::new();
    let mut cumulative_counts = vec![StratumCounts::default(); plan.strata.len()];
    let mut index = 0usize;
    let mut aurc_area = 0.0_f64;
    let mut aurc_supported = true;
    let total = sorted.len();
    while index < total {
        let cutoff = sorted[index].predicted_confidence;
        let mut end = index;
        while end < total && sorted[end].predicted_confidence == cutoff {
            end += 1;
        }

        // Update counts once per record; each confidence prefix is then scored
        // in O(number_of_strata), not by rescanning the entire cohort.
        for prediction in &sorted[index..end] {
            let (stratum_index, selected) = selection_by_id[prediction.id.as_str()];
            let counts = &mut cumulative_counts[stratum_index];
            counts.accepted_population_count += 1;
            if selected {
                counts.accepted_audit_count += 1;
                if prediction.predicted_label != prediction.adjudicated_truth.unwrap() {
                    counts.observed_error_count += 1;
                }
            }
        }

        let estimate = estimate_from_stratum_counts(
            &format!("confidence_at_least_{cutoff:.17}"),
            &cumulative_counts,
            plan,
            config.minimum_sample_size,
        );
        if estimate.estimated_risk.is_some() {
            let previous_error_total = curve
                .last()
                .map_or(0.0, |previous: &DesignBasedCurvePoint| {
                    previous.estimate.estimated_error_total
                });
            let group_error_total = estimate.estimated_error_total - previous_error_total;
            let group_size = end - index;
            for rank in 1..=group_size {
                let expected_error_total =
                    previous_error_total + rank as f64 * group_error_total / group_size as f64;
                let accepted_at_rank = index + rank;
                aurc_area += (expected_error_total / accepted_at_rank as f64) / total as f64;
            }
        } else {
            aurc_supported = false;
        }
        curve.push(DesignBasedCurvePoint {
            confidence_cutoff: cutoff,
            estimate,
        });
        index = end;
    }
    let aurc = (aurc_supported && total > 0).then_some(aurc_area);

    let mut operating_points = Vec::new();
    let mut targets = config.target_coverages.clone();
    targets.sort_by(f64::total_cmp);
    for target in targets {
        if !target.is_finite() || target <= 0.0 || target > 1.0 {
            continue;
        }
        let required = (target * total as f64).ceil() as usize;
        if let Some(point) = curve
            .iter()
            .find(|point| point.estimate.accepted_population_count >= required)
        {
            operating_points.push(DesignBasedOperatingPoint {
                target_coverage: target,
                achieved_coverage: point.estimate.accepted_coverage,
                confidence_cutoff: point.confidence_cutoff,
                estimate: point.estimate.clone(),
            });
        }
    }

    let mut thresholds = config.confidence_thresholds.clone();
    thresholds.sort_by(f64::total_cmp);
    let confidence_threshold_baselines = thresholds
        .into_iter()
        .filter(|threshold| threshold.is_finite() && (0.0..=1.0).contains(threshold))
        .map(|threshold| {
            let accepted_ids: HashSet<&str> = sorted
                .iter()
                .filter(|row| row.predicted_confidence >= threshold)
                .map(|row| row.id.as_str())
                .collect();
            DesignBasedThresholdPoint {
                confidence_threshold: threshold,
                estimate: estimate_from_stratum_counts(
                    &format!("confidence_threshold_{threshold:.17}"),
                    &counts_for_acceptance(&accepted_ids, predictions, plan),
                    plan,
                    config.minimum_sample_size,
                ),
            }
        })
        .collect();

    Ok(StratifiedRiskReport {
        metric_version: "symthaea-stratified-design-risk-v1".to_owned(),
        design_id: plan.design_id.clone(),
        population_size: plan.population_size,
        audit_sample_size: plan.audit_sample_size,
        audit_fraction: plan.audit_sample_size as f64 / plan.population_size as f64,
        always_answer,
        observed_model_policy,
        operating_points,
        confidence_threshold_baselines,
        risk_coverage_curve: curve,
        aurc,
        aurc_uncertainty_status: concat!(
            "Not estimated: nested confidence-prefix covariance and simultaneous ",
            "uncertainty are not implemented"
        )
        .to_owned(),
        interval_scope: concat!(
            "Pointwise asymptotic normal approximation only; not finite-sample, ",
            "anytime-valid or simultaneous"
        )
        .to_owned(),
        assumptions: vec![
            concat!(
                "The candidate prediction, confidence and abstention state for every ",
                "cohort row is frozen before audit selection."
            )
            .to_owned(),
            concat!(
                "The audit plan is a valid stratified simple-random sample without ",
                "replacement within each pre-label stratum."
            )
            .to_owned(),
            concat!(
                "Random priorities were generated independently and uniformly by an ",
                "external custodian; the plan does not authenticate that process."
            )
            .to_owned(),
            concat!(
                "Every selected unit received an independent adjudication and ",
                "nonselected units did not receive labels in this audit lane."
            )
            .to_owned(),
            concat!(
                "The strata and quotas were frozen before labels and are not selected ",
                "using adjudication outcomes."
            )
            .to_owned(),
            concat!(
                "Normal intervals are approximate and pointwise; no multiple-comparison ",
                "or finite-sample risk guarantee is asserted."
            )
            .to_owned(),
        ],
    })
}

#[derive(Debug, Clone, Copy, Default)]
struct StratumCounts {
    accepted_population_count: usize,
    accepted_audit_count: usize,
    observed_error_count: usize,
}

fn counts_for_acceptance(
    accepted_ids: &HashSet<&str>,
    predictions: &[SelectivePredictionRecord],
    plan: &StratifiedAuditPlan,
) -> Vec<StratumCounts> {
    let stratum_index_by_id: HashMap<&str, usize> = plan
        .strata
        .iter()
        .enumerate()
        .map(|(index, stratum)| (stratum.stratum_id.as_str(), index))
        .collect();
    let selection_by_id: HashMap<&str, (usize, bool)> = plan
        .selections
        .iter()
        .map(|selection| {
            (
                selection.record_id.as_str(),
                (
                    stratum_index_by_id[selection.stratum_id.as_str()],
                    selection.selected_for_adjudication,
                ),
            )
        })
        .collect();

    let mut counts = vec![StratumCounts::default(); plan.strata.len()];
    for prediction in predictions {
        if !accepted_ids.contains(prediction.id.as_str()) {
            continue;
        }
        let (stratum_index, selected) = selection_by_id[prediction.id.as_str()];
        let stratum_counts = &mut counts[stratum_index];
        stratum_counts.accepted_population_count += 1;
        if selected {
            stratum_counts.accepted_audit_count += 1;
            if prediction.predicted_label != prediction.adjudicated_truth.unwrap() {
                stratum_counts.observed_error_count += 1;
            }
        }
    }
    counts
}

fn estimate_from_stratum_counts(
    acceptance_rule: &str,
    counts: &[StratumCounts],
    plan: &StratifiedAuditPlan,
    minimum_sample_size: usize,
) -> DesignBasedRiskEstimate {
    let mut accepted_population_count = 0usize;
    let mut accepted_audit_count = 0usize;
    let mut estimated_error_total = 0.0_f64;
    let mut estimated_variance_total = 0.0_f64;
    let mut variance_estimable = true;
    let mut variance_unavailable_strata = Vec::new();
    let mut strata = Vec::with_capacity(plan.strata.len());

    for (index, stratum) in plan.strata.iter().enumerate() {
        let counts = counts[index];
        let n = stratum.sample_size;
        let population_size = stratum.population_size;
        let sample_sum = counts.observed_error_count as f64;
        let stratum_error_total = population_size as f64 * sample_sum / n as f64;
        estimated_error_total += stratum_error_total;
        accepted_population_count += counts.accepted_population_count;
        accepted_audit_count += counts.accepted_audit_count;

        let sample_error_variance = if counts.accepted_population_count == 0 || population_size == n
        {
            Some(0.0)
        } else if n >= 2 {
            // For a binary sample with k ones, s^2 = k(n-k)/(n(n-1)).
            Some(sample_sum * (n as f64 - sample_sum) / (n as f64 * (n - 1) as f64))
        } else {
            None
        };
        let stratum_variance = sample_error_variance.map(|sample_variance| {
            (population_size as f64).powi(2)
                * (1.0 - n as f64 / population_size as f64)
                * sample_variance
                / n as f64
        });
        if let Some(value) = stratum_variance {
            estimated_variance_total += value;
        } else {
            variance_estimable = false;
            variance_unavailable_strata.push(stratum.stratum_id.clone());
        }
        strata.push(StratumRiskEstimate {
            stratum_id: stratum.stratum_id.clone(),
            population_size,
            audit_sample_size: n,
            accepted_population_count: counts.accepted_population_count,
            accepted_audit_count: counts.accepted_audit_count,
            observed_error_count: counts.observed_error_count,
            estimated_error_total: stratum_error_total,
            estimated_error_total_variance: stratum_variance,
            sample_error_variance,
            variance_estimable: stratum_variance.is_some(),
        });
    }

    let estimated_risk = (accepted_population_count > 0)
        .then_some(estimated_error_total / accepted_population_count as f64);
    let estimated_error_total_variance = variance_estimable.then_some(estimated_variance_total);
    let estimated_risk_variance = estimated_error_total_variance.and_then(|variance| {
        (accepted_population_count > 0)
            .then_some(variance / (accepted_population_count as f64).powi(2))
    });
    let enough_audit_labels = accepted_audit_count >= minimum_sample_size;
    let standard_error = if enough_audit_labels {
        estimated_risk_variance.map(f64::sqrt)
    } else {
        None
    };
    let approximate_normal_95_interval =
        estimated_risk
            .zip(standard_error)
            .map(|(risk, se)| ApproximateRiskInterval {
                lower: risk - NORMAL_95_Z * se,
                upper: risk + NORMAL_95_Z * se,
                confidence_level: 0.95,
                method: concat!(
                    "pointwise normal approximation using unbiased ",
                    "stratified-SRSWOR variance"
                )
                .to_owned(),
            });

    DesignBasedRiskEstimate {
        acceptance_rule: acceptance_rule.to_owned(),
        accepted_population_count,
        accepted_coverage: accepted_population_count as f64 / plan.population_size as f64,
        accepted_audit_count,
        estimated_error_total,
        estimated_risk,
        estimated_risk_outside_unit_interval: estimated_risk
            .is_some_and(|risk| !(0.0..=1.0).contains(&risk)),
        estimated_error_total_variance,
        estimated_risk_variance,
        standard_error,
        approximate_normal_95_interval,
        variance_estimable,
        variance_unavailable_strata: variance_unavailable_strata.clone(),
        uncertainty_unavailable_reason: if !variance_estimable {
            Some(
                concat!(
                    "at least one non-census stratum has fewer than two ",
                    "selected audit labels"
                )
                .to_owned(),
            )
        } else if !enough_audit_labels {
            Some(format!(
                concat!(
                    "accepted audit label count {} is below ",
                    "the configured minimum {}"
                ),
                accepted_audit_count,
                minimum_sample_size
            ))
        } else {
            None
        },
        strata,
    }
}

#[cfg(test)]
mod tests {
    use super::super::stratified_audit::{
        AuditCandidate, AuditStratumQuota, build_stratified_random_audit_plan,
    };
    use super::*;

    fn four_unit_plan() -> StratifiedAuditPlan {
        let candidates = vec![
            AuditCandidate {
                record_id: "a".to_owned(),
                stratum_id: "all".to_owned(),
                random_priority: 10,
            },
            AuditCandidate {
                record_id: "b".to_owned(),
                stratum_id: "all".to_owned(),
                random_priority: 20,
            },
            AuditCandidate {
                record_id: "c".to_owned(),
                stratum_id: "all".to_owned(),
                random_priority: 30,
            },
            AuditCandidate {
                record_id: "d".to_owned(),
                stratum_id: "all".to_owned(),
                random_priority: 40,
            },
        ];
        let quotas = vec![AuditStratumQuota {
            stratum_id: "all".to_owned(),
            sample_size: 2,
        }];
        build_stratified_random_audit_plan(
            &candidates,
            &quotas,
            "design-v1",
            "custody://randomization/four",
        )
        .unwrap()
    }

    fn prediction(
        id: &str,
        confidence: f64,
        label: bool,
        truth: Option<bool>,
    ) -> SelectivePredictionRecord {
        SelectivePredictionRecord {
            id: id.to_owned(),
            predicted_label: label,
            predicted_confidence: confidence,
            adjudicated_truth: truth,
            model_abstained: false,
            abstention_reason: None,
            domain: Some("science".to_owned()),
            source_family: Some("source-a".to_owned()),
            delayed_or_censored: Some(false),
            adjudicator: truth.map(|_| "independent-panel".to_owned()),
        }
    }

    fn config() -> RiskCoverageConfig {
        RiskCoverageConfig {
            minimum_sample_size: 1,
            target_coverages: vec![0.5, 0.8],
            confidence_thresholds: vec![0.5, 0.9],
            decision_costs: None,
            include_stratified_reports: false,
        }
    }

    #[test]
    fn ht_risk_and_srswor_variance_match_known_hand_calculation() {
        let plan = four_unit_plan();
        let rows = vec![
            prediction("a", 0.9, true, Some(false)),
            prediction("b", 0.8, true, Some(true)),
            prediction("c", 0.4, true, None),
            prediction("d", 0.2, true, None),
        ];
        let report = evaluate_stratified_audit_risk(&rows, &plan, &config()).unwrap();
        let estimate = report.always_answer;
        assert_eq!(estimate.accepted_population_count, 4);
        assert_eq!(estimate.accepted_audit_count, 2);
        assert_eq!(estimate.estimated_error_total, 2.0);
        assert_eq!(estimate.estimated_risk, Some(0.5));
        assert_eq!(estimate.estimated_error_total_variance, Some(2.0));
        assert_eq!(estimate.estimated_risk_variance, Some(0.125));
        assert!(estimate.approximate_normal_95_interval.is_some());
    }

    #[test]
    fn independent_stratum_variances_add_for_total_risk() {
        let candidates = vec![
            AuditCandidate {
                record_id: "a1".to_owned(),
                stratum_id: "census".to_owned(),
                random_priority: 1,
            },
            AuditCandidate {
                record_id: "a2".to_owned(),
                stratum_id: "census".to_owned(),
                random_priority: 2,
            },
            AuditCandidate {
                record_id: "b1".to_owned(),
                stratum_id: "sampled".to_owned(),
                random_priority: 10,
            },
            AuditCandidate {
                record_id: "b2".to_owned(),
                stratum_id: "sampled".to_owned(),
                random_priority: 20,
            },
            AuditCandidate {
                record_id: "b3".to_owned(),
                stratum_id: "sampled".to_owned(),
                random_priority: 30,
            },
            AuditCandidate {
                record_id: "b4".to_owned(),
                stratum_id: "sampled".to_owned(),
                random_priority: 40,
            },
        ];
        let quotas = vec![
            AuditStratumQuota {
                stratum_id: "census".to_owned(),
                sample_size: 2,
            },
            AuditStratumQuota {
                stratum_id: "sampled".to_owned(),
                sample_size: 2,
            },
        ];
        let plan = build_stratified_random_audit_plan(
            &candidates,
            &quotas,
            "multi-stratum-v1",
            "custody://randomization/multi-stratum",
        )
        .unwrap();
        let rows = vec![
            prediction("a1", 0.9, true, Some(false)),
            prediction("a2", 0.8, true, Some(true)),
            prediction("b1", 0.7, true, Some(false)),
            prediction("b2", 0.6, true, Some(true)),
            prediction("b3", 0.5, true, None),
            prediction("b4", 0.4, true, None),
        ];
        let report = evaluate_stratified_audit_risk(&rows, &plan, &config()).unwrap();
        assert_eq!(report.always_answer.estimated_error_total, 3.0);
        assert_eq!(report.always_answer.estimated_risk, Some(0.5));
        assert_eq!(
            report.always_answer.estimated_error_total_variance,
            Some(2.0)
        );
        assert!((report.always_answer.estimated_risk_variance.unwrap() - 2.0 / 36.0).abs() < 1e-12);
    }

    #[test]
    fn tampered_inclusion_probability_is_rejected_before_scoring() {
        let mut plan = four_unit_plan();
        plan.selections[0].inclusion_probability = 0.75;
        let rows = vec![
            prediction("a", 0.9, true, Some(false)),
            prediction("b", 0.8, true, Some(true)),
            prediction("c", 0.4, true, None),
            prediction("d", 0.2, true, None),
        ];
        assert!(matches!(
            evaluate_stratified_audit_risk(&rows, &plan, &config()),
            Err(StratifiedRiskError::InvalidPlan(_))
        ));
    }

    #[test]
    fn census_has_zero_design_variance() {
        let candidates = vec![
            AuditCandidate {
                record_id: "a".to_owned(),
                stratum_id: "all".to_owned(),
                random_priority: 10,
            },
            AuditCandidate {
                record_id: "b".to_owned(),
                stratum_id: "all".to_owned(),
                random_priority: 20,
            },
        ];
        let quotas = vec![AuditStratumQuota {
            stratum_id: "all".to_owned(),
            sample_size: 2,
        }];
        let plan = build_stratified_random_audit_plan(
            &candidates,
            &quotas,
            "census-v1",
            "custody://randomization/census",
        )
        .unwrap();
        let rows = vec![
            prediction("a", 0.9, true, Some(false)),
            prediction("b", 0.8, true, Some(true)),
        ];
        let report = evaluate_stratified_audit_risk(&rows, &plan, &config()).unwrap();
        assert_eq!(report.always_answer.estimated_risk, Some(0.5));
        assert_eq!(report.always_answer.estimated_risk_variance, Some(0.0));
        assert_eq!(report.always_answer.standard_error, Some(0.0));
    }

    #[test]
    fn singleton_non_census_stratum_withholds_variance_not_point_estimate() {
        let candidates = vec![
            AuditCandidate {
                record_id: "a".to_owned(),
                stratum_id: "single".to_owned(),
                random_priority: 10,
            },
            AuditCandidate {
                record_id: "b".to_owned(),
                stratum_id: "single".to_owned(),
                random_priority: 20,
            },
        ];
        let quotas = vec![AuditStratumQuota {
            stratum_id: "single".to_owned(),
            sample_size: 1,
        }];
        let plan = build_stratified_random_audit_plan(
            &candidates,
            &quotas,
            "singleton-v1",
            "custody://randomization/singleton",
        )
        .unwrap();
        let rows = vec![
            prediction("a", 0.9, true, Some(false)),
            prediction("b", 0.8, true, None),
        ];
        let report = evaluate_stratified_audit_risk(&rows, &plan, &config()).unwrap();
        assert_eq!(report.always_answer.estimated_risk, Some(1.0));
        assert!(!report.always_answer.variance_estimable);
        assert_eq!(report.always_answer.estimated_risk_variance, None);
        assert_eq!(
            report.always_answer.variance_unavailable_strata,
            vec!["single".to_owned()]
        );
    }

    #[test]
    fn selected_audit_rows_must_be_adjudicated_and_nonselected_rows_pending() {
        let plan = four_unit_plan();
        let missing = vec![
            prediction("a", 0.9, true, None),
            prediction("b", 0.8, true, Some(true)),
            prediction("c", 0.4, true, None),
            prediction("d", 0.2, true, None),
        ];
        assert!(matches!(
            evaluate_stratified_audit_risk(&missing, &plan, &config()),
            Err(StratifiedRiskError::SelectedRecordMissingAdjudication(_))
        ));

        let leaked = vec![
            prediction("a", 0.9, true, Some(false)),
            prediction("b", 0.8, true, Some(true)),
            prediction("c", 0.4, true, Some(true)),
            prediction("d", 0.2, true, None),
        ];
        assert!(matches!(
            evaluate_stratified_audit_risk(&leaked, &plan, &config()),
            Err(StratifiedRiskError::NonSelectedRecordHasAdjudication(_))
        ));
    }

    #[test]
    fn confidence_ties_are_atomic_and_aurc_is_order_invariant() {
        let plan = four_unit_plan();
        let rows = vec![
            prediction("a", 0.9, true, Some(false)),
            prediction("b", 0.9, true, Some(true)),
            prediction("c", 0.4, true, None),
            prediction("d", 0.2, true, None),
        ];
        let mut reversed = rows.clone();
        reversed.reverse();
        let first = evaluate_stratified_audit_risk(&rows, &plan, &config()).unwrap();
        let second = evaluate_stratified_audit_risk(&reversed, &plan, &config()).unwrap();
        assert_eq!(first.aurc, second.aurc);
        assert_eq!(first.operating_points[0].achieved_coverage, 0.5);
        assert_eq!(
            first.operating_points[0].estimate.accepted_population_count,
            2
        );
    }

    #[test]
    fn unsupported_decision_costs_fail_closed_instead_of_being_ignored() {
        let plan = four_unit_plan();
        let rows = vec![
            prediction("a", 0.9, true, Some(false)),
            prediction("b", 0.8, true, Some(true)),
            prediction("c", 0.4, true, None),
            prediction("d", 0.2, true, None),
        ];
        let mut configured = config();
        configured.decision_costs = Some(super::super::selective_prediction::DecisionCosts {
            false_positive: 2.0,
            false_negative: 3.0,
            abstention: 0.5,
        });
        assert_eq!(
            evaluate_stratified_audit_risk(&rows, &plan, &configured).unwrap_err(),
            StratifiedRiskError::DecisionCostsNotSupported
        );
    }

    #[test]
    fn invalid_operating_point_configuration_fails_closed() {
        let plan = four_unit_plan();
        let rows = vec![
            prediction("a", 0.9, true, Some(false)),
            prediction("b", 0.8, true, Some(true)),
            prediction("c", 0.4, true, None),
            prediction("d", 0.2, true, None),
        ];
        let mut invalid = config();
        invalid.target_coverages = vec![1.1];
        assert_eq!(
            evaluate_stratified_audit_risk(&rows, &plan, &invalid).unwrap_err(),
            StratifiedRiskError::InvalidCoverageTarget
        );
        let mut invalid = config();
        invalid.confidence_thresholds = vec![f64::NAN];
        assert_eq!(
            evaluate_stratified_audit_risk(&rows, &plan, &invalid).unwrap_err(),
            StratifiedRiskError::InvalidConfidenceThreshold
        );
    }

    #[test]
    fn invalid_prediction_cohort_size_and_duplicate_ids_fail_closed() {
        let plan = four_unit_plan();
        let rows = vec![
            prediction("a", 0.9, true, Some(false)),
            prediction("b", 0.8, true, Some(true)),
            prediction("c", 0.4, true, None),
        ];
        assert_eq!(
            evaluate_stratified_audit_risk(&rows, &plan, &config()).unwrap_err(),
            StratifiedRiskError::PredictionCountMismatch
        );

        let rows = vec![
            prediction("a", 0.9, true, Some(false)),
            prediction("a", 0.8, true, Some(true)),
            prediction("c", 0.4, true, None),
            prediction("d", 0.2, true, None),
        ];
        assert!(matches!(
            evaluate_stratified_audit_risk(&rows, &plan, &config()),
            Err(StratifiedRiskError::DuplicatePredictionId(_))
        ));
    }

    #[test]
    fn threshold_baselines_and_observed_policy_are_reported() {
        let plan = four_unit_plan();
        let mut rows = vec![
            prediction("a", 0.9, true, Some(false)),
            prediction("b", 0.8, true, Some(true)),
            prediction("c", 0.4, true, None),
            prediction("d", 0.2, true, None),
        ];
        rows[1].model_abstained = true;
        let report = evaluate_stratified_audit_risk(&rows, &plan, &config()).unwrap();
        assert_eq!(report.observed_model_policy.accepted_population_count, 3);
        assert_eq!(report.confidence_threshold_baselines.len(), 2);
        assert_eq!(
            report.confidence_threshold_baselines[0]
                .estimate
                .accepted_population_count,
            2
        );
        assert_eq!(
            report.confidence_threshold_baselines[1]
                .estimate
                .accepted_population_count,
            1
        );
    }
}
