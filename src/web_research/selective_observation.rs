// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Population-risk diagnostics for selectively adjudicated scientific claims.
//!
//! The ordinary selective-prediction evaluator describes the adjudicated
//! cohort. This module optionally estimates risk for the full frozen prediction
//! cohort when each record has a known, strictly positive pre-label inclusion
//! probability and the sampling design is declared. It uses Horvitz-Thompson
//! estimates of error totals; finite-sample estimates can exceed [0, 1].
//!
//! This module does not authenticate the sampling design, prove conditional
//! ignorability, or provide confidence intervals. Those require independent
//! design custody and a variance procedure appropriate to the joint sampling
//! design (including clustering where applicable).

use super::selective_prediction::{
    RiskCoverageConfig, SelectivePredictionRecord, StratificationDimension,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

const AURC_CONVENTION: &str =
    "Item-wise empirical AURC; expected over uniform within-tie permutations";
const UNCERTAINTY_LIMITATION: &str = concat!(
    "No confidence interval: requires joint inclusion probabilities or a ",
    "design-valid clustered variance estimator"
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdjudicationSamplingMechanism {
    /// A randomized audit assigns each case a known, positive chance of
    /// independent adjudication, including cases the candidate would answer
    /// confidently or abstain on.
    RandomizedAudit,
    /// Inclusion probabilities are known conditional on recorded variables
    /// available before the adjudication outcome is observed.
    KnownPropensityConditionalOnPrelabelHistory,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdjudicationSamplingDesign {
    /// Stable identifier for the frozen sampling/audit design.
    pub design_id: String,
    pub mechanism: AdjudicationSamplingMechanism,
    /// Variables used to assign inclusion probabilities, all measured before
    /// the label is observed. Randomized audits may leave this empty.
    pub prelabel_conditioning_variables: Vec<String>,
    /// Human-readable reference to the independently frozen design manifest.
    pub design_manifest_ref: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectiveObservationRecord {
    pub prediction: SelectivePredictionRecord,
    /// Probability that this record would receive an independent label under
    /// the declared design, conditional only on pre-label information.
    /// Must be present and in (0, 1] for every record in the target cohort.
    pub adjudication_inclusion_probability: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropensitySupportDiagnostics {
    pub population_records: usize,
    pub adjudicated_records: usize,
    pub pending_records: usize,
    pub missing_probability_records: usize,
    pub zero_probability_records: usize,
    pub invalid_probability_records: usize,
    pub invalid_prediction_confidence_records: usize,
    /// Pending records with inclusion probability 1.0 contradict a design that
    /// defines inclusion as "label observed by this frozen evaluation cutoff".
    pub pending_certain_inclusion_records: usize,
    pub positive_support_for_entire_cohort: bool,
    pub maximum_inverse_probability_weight: Option<f64>,
    pub kish_effective_sample_size: Option<f64>,
    /// Explicitly descriptive: the evaluator cannot verify that the design was
    /// randomized or that all confounders were measured before adjudication.
    pub design_assumptions_require_external_audit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropensityWeightedPoint {
    pub accepted_count_in_full_cohort: usize,
    pub coverage_in_full_cohort: f64,
    pub independently_labelled_accepted_count: usize,
    pub kish_effective_sample_size: f64,
    /// Horvitz-Thompson estimate of the number of errors in the full accepted
    /// cohort, using observed error_i / inclusion_probability_i.
    pub estimated_error_total: f64,
    /// Estimated error total divided by the known accepted population count.
    /// This is not clipped; finite-sample estimates can exceed 1.
    pub estimated_risk: Option<f64>,
    pub estimate_outside_unit_interval: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropensityWeightedCurvePoint {
    pub confidence_cutoff: f64,
    pub point: PropensityWeightedPoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropensityWeightedOperatingPoint {
    pub target_coverage: f64,
    /// Equal-confidence groups are admitted atomically, so achieved coverage
    /// may exceed the target.
    pub achieved_coverage: f64,
    pub point: PropensityWeightedPoint,
    /// Expected risk under random selection at the achieved coverage. Under
    /// random selection independent of correctness this equals full-cohort risk.
    pub random_abstention_expected_risk: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropensityWeightedThresholdPoint {
    pub threshold: f64,
    pub point: PropensityWeightedPoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropensityWeightedReport {
    pub metric_version: String,
    pub design_id: String,
    pub sampling_mechanism: AdjudicationSamplingMechanism,
    pub population_records: usize,
    pub adjudicated_records: usize,
    pub kish_effective_sample_size: f64,
    pub maximum_inverse_probability_weight: f64,
    /// None when the effective sample is too small for the configured gate.
    pub aurc: Option<f64>,
    pub aurc_convention: String,
    pub curve: Vec<PropensityWeightedCurvePoint>,
    pub operating_points: Vec<PropensityWeightedOperatingPoint>,
    pub confidence_threshold_baselines: Vec<PropensityWeightedThresholdPoint>,
    pub always_answer: PropensityWeightedPoint,
    pub observed_model_policy: PropensityWeightedPoint,
    pub uncertainty_limitation: String,
    pub assumptions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropensityStratifiedReport {
    pub dimension: StratificationDimension,
    pub value: String,
    pub report: Option<PropensityWeightedReport>,
    pub unavailable_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectiveObservationEvaluation {
    pub design: AdjudicationSamplingDesign,
    pub diagnostics: PropensitySupportDiagnostics,
    pub population_report: Option<PropensityWeightedReport>,
    pub unavailable_reason: Option<String>,
    pub stratified: Vec<PropensityStratifiedReport>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectiveObservationError {
    EmptyDesignId,
    EmptyDesignManifestReference,
    EmptyConditioningVariable,
    EmptyRecordId,
    DuplicateRecordId(String),
    InvalidMinimumSampleSize,
    InvalidCoverageTarget,
    InvalidConfidenceThreshold,
}

impl std::fmt::Display for SelectiveObservationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDesignId => write!(formatter, "sampling design ID must not be empty"),
            Self::EmptyDesignManifestReference => {
                write!(formatter, "sampling design manifest reference must not be empty")
            }
            Self::EmptyConditioningVariable => {
                write!(formatter, "pre-label conditioning variable names must not be empty")
            }
            Self::EmptyRecordId => write!(formatter, "record IDs must not be empty"),
            Self::DuplicateRecordId(id) => write!(formatter, "duplicate record ID: {id}"),
            Self::InvalidMinimumSampleSize => {
                write!(formatter, "minimum sample size must be positive")
            }
            Self::InvalidCoverageTarget => {
                write!(formatter, "coverage targets must be finite and in (0, 1]")
            }
            Self::InvalidConfidenceThreshold => {
                write!(formatter, "confidence thresholds must be finite and in [0, 1]")
            }
        }
    }
}

impl std::error::Error for SelectiveObservationError {}

/// Evaluate a frozen prediction cohort under an explicit adjudication sampling
/// design. If any record has zero, missing or invalid inclusion probability,
/// the population estimate is withheld rather than extrapolated across an
/// unobserved region.
pub fn evaluate_selective_observation(
    records: &[SelectiveObservationRecord],
    design: &AdjudicationSamplingDesign,
    config: &RiskCoverageConfig,
) -> Result<SelectiveObservationEvaluation, SelectiveObservationError> {
    if config.minimum_sample_size == 0 {
        return Err(SelectiveObservationError::InvalidMinimumSampleSize);
    }
    if config
        .target_coverages
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0 || *value > 1.0)
    {
        return Err(SelectiveObservationError::InvalidCoverageTarget);
    }
    if config
        .confidence_thresholds
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(SelectiveObservationError::InvalidConfidenceThreshold);
    }
    if design.design_id.trim().is_empty() {
        return Err(SelectiveObservationError::EmptyDesignId);
    }
    if design.design_manifest_ref.trim().is_empty() {
        return Err(SelectiveObservationError::EmptyDesignManifestReference);
    }
    if design.mechanism
        == AdjudicationSamplingMechanism::KnownPropensityConditionalOnPrelabelHistory
        && design
            .prelabel_conditioning_variables
            .iter()
            .any(|value| value.trim().is_empty())
    {
        return Err(SelectiveObservationError::EmptyConditioningVariable);
    }

    let mut ids = HashSet::with_capacity(records.len());
    for record in records {
        let id = record.prediction.id.trim();
        if id.is_empty() {
            return Err(SelectiveObservationError::EmptyRecordId);
        }
        if !ids.insert(id) {
            return Err(SelectiveObservationError::DuplicateRecordId(id.to_owned()));
        }
    }

    let diagnostics = support_diagnostics(records);
    let (population_report, unavailable_reason) =
        if diagnostics.pending_certain_inclusion_records > 0 {
            (
                None,
                Some(
                    concat!(
                        "pending labels have inclusion probability 1.0 under a design ",
                        "defined at this evaluation cutoff"
                    )
                        .to_owned(),
                ),
            )
        } else if !diagnostics.positive_support_for_entire_cohort {
            (
                None,
                Some(
                    concat!(
                        "population risk is not identifiable: every record needs a known, ",
                        "strictly positive adjudication inclusion probability"
                    )
                    .to_owned(),
                ),
            )
        } else if diagnostics.invalid_prediction_confidence_records > 0 {
            (
                None,
                Some(
                    concat!(
                        "population estimate withheld because the candidate emitted invalid ",
                        "confidence values"
                    )
                    .to_owned(),
                ),
            )
        } else if diagnostics.kish_effective_sample_size.unwrap_or(0.0)
            < config.minimum_sample_size as f64
        {
            (
                None,
                Some(format!(
                    "effective adjudication sample size is below the configured minimum of {}",
                    config.minimum_sample_size
                )),
            )
        } else {
            (
                Some(compute_population_report(records, design, config)),
                None,
            )
        };

    let stratified = if config.include_stratified_reports {
        stratify_population(records, design, config)
    } else {
        Vec::new()
    };

    Ok(SelectiveObservationEvaluation {
        design: design.clone(),
        diagnostics,
        population_report,
        unavailable_reason,
        stratified,
    })
}

fn support_diagnostics(records: &[SelectiveObservationRecord]) -> PropensitySupportDiagnostics {
    let mut adjudicated_records = 0;
    let mut pending_records = 0;
    let mut missing_probability_records = 0;
    let mut zero_probability_records = 0;
    let mut invalid_probability_records = 0;
    let mut invalid_prediction_confidence_records = 0;
    let mut pending_certain_inclusion_records = 0;
    let mut weights = Vec::new();

    for record in records {
        if !record.prediction.predicted_confidence.is_finite()
            || !(0.0..=1.0).contains(&record.prediction.predicted_confidence)
        {
            invalid_prediction_confidence_records += 1;
        }
        if record.prediction.adjudicated_truth.is_some() {
            adjudicated_records += 1;
        } else {
            pending_records += 1;
        }
        match record.adjudication_inclusion_probability {
            None => missing_probability_records += 1,
            Some(probability) if probability == 0.0 => zero_probability_records += 1,
            Some(probability)
                if !probability.is_finite()
                    || probability < 0.0
                    || probability > 1.0
                    || (probability > 0.0 && !(1.0 / probability).is_finite()) =>
            {
                invalid_probability_records += 1
            }
            Some(probability) if record.prediction.adjudicated_truth.is_some() => {
                weights.push(1.0 / probability);
            }
            Some(probability) if probability == 1.0 => {
                pending_certain_inclusion_records += 1;
            }
            Some(_) => {}
        }
    }

    let positive_support_for_entire_cohort = !records.is_empty()
        && missing_probability_records == 0
        && zero_probability_records == 0
        && invalid_probability_records == 0
        && pending_certain_inclusion_records == 0;
    let maximum_weight = weights.iter().copied().reduce(f64::max);
    let scaled_weights: Vec<f64> = maximum_weight
        .filter(|maximum| *maximum > 0.0)
        .map(|maximum| weights.iter().map(|weight| weight / maximum).collect())
        .unwrap_or_default();
    let scaled_sum = scaled_weights.iter().sum::<f64>();
    let scaled_sum_squared = scaled_weights
        .iter()
        .map(|weight| weight * weight)
        .sum::<f64>();
    let kish_effective_sample_size = (scaled_sum_squared > 0.0)
        .then_some(scaled_sum * scaled_sum / scaled_sum_squared);

    PropensitySupportDiagnostics {
        population_records: records.len(),
        adjudicated_records,
        pending_records,
        missing_probability_records,
        zero_probability_records,
        invalid_probability_records,
        invalid_prediction_confidence_records,
        pending_certain_inclusion_records,
        positive_support_for_entire_cohort,
        maximum_inverse_probability_weight: maximum_weight,
        kish_effective_sample_size,
        design_assumptions_require_external_audit: true,
    }
}

fn compute_population_report(
    records: &[SelectiveObservationRecord],
    design: &AdjudicationSamplingDesign,
    config: &RiskCoverageConfig,
) -> PropensityWeightedReport {
    let mut sorted: Vec<&SelectiveObservationRecord> = records.iter().collect();
    sorted.sort_by(|left, right| {
        right
            .prediction
            .predicted_confidence
            .total_cmp(&left.prediction.predicted_confidence)
            .then_with(|| left.prediction.id.cmp(&right.prediction.id))
    });

    let total = sorted.len();
    let all_indices: Vec<usize> = (0..total).collect();
    let always_answer = weighted_point(records, &all_indices, config.minimum_sample_size);
    let observed_indices: Vec<usize> = sorted
        .iter()
        .enumerate()
        .filter_map(|(index, record)| (!record.prediction.model_abstained).then_some(index))
        .collect();
    let observed_model_policy =
        weighted_point(&sorted, &observed_indices, config.minimum_sample_size);

    let mut curve = Vec::new();
    let mut index = 0;
    let mut accepted_label_weights: Vec<f64> = Vec::new();
    let mut cumulative_error_weight = 0.0_f64;
    let mut area = 0.0_f64;
    let mut aurc_supported = true;

    while index < total {
        let cutoff = sorted[index].prediction.predicted_confidence;
        let mut end = index;
        let mut group_error_weight = 0.0_f64;
        let mut group_label_weights = Vec::new();
        while end < total && sorted[end].prediction.predicted_confidence == cutoff {
            let record = sorted[end];
            if let Some(truth) = record.prediction.adjudicated_truth {
                let probability = record.adjudication_inclusion_probability.unwrap();
                let weight = 1.0 / probability;
                group_label_weights.push(weight);
                if record.prediction.predicted_label != truth {
                    group_error_weight += weight;
                }
            }
            end += 1;
        }

        accepted_label_weights.extend(group_label_weights);
        cumulative_error_weight += group_error_weight;
        let accepted_count = end;
        let effective_n = kish_ess(&accepted_label_weights);
        let point = PropensityWeightedPoint {
            accepted_count_in_full_cohort: accepted_count,
            coverage_in_full_cohort: accepted_count as f64 / total as f64,
            independently_labelled_accepted_count: accepted_label_weights.len(),
            kish_effective_sample_size: effective_n,
            estimated_error_total: cumulative_error_weight,
            // AURC is exploratory at each rank; no pointwise confidence
            // guarantee is asserted. The overall effective sample-size gate
            // is applied before this report is constructed.
            estimated_risk: (effective_n > 0.0)
                .then_some(cumulative_error_weight / accepted_count as f64),
            estimate_outside_unit_interval: effective_n > 0.0
                && cumulative_error_weight / accepted_count as f64 > 1.0,
        };
        if point.estimated_risk.is_none() {
            aurc_supported = false;
        }
        curve.push(PropensityWeightedCurvePoint {
            confidence_cutoff: cutoff,
            point,
        });

        // Within a confidence tie, average item-wise risk over all permutations.
        // Horvitz-Thompson error contributions are fractional after expectation;
        // every record in the tie group has equal probability of appearing in
        // each of the group's rank positions.
        let group_size = end - index;
        let previous_accepted = index;
        let previous_error_weight = cumulative_error_weight - group_error_weight;
        for rank_in_group in 1..=group_size {
            let expected_error_total = previous_error_weight
                + rank_in_group as f64 * group_error_weight / group_size as f64;
            let accepted_at_rank = previous_accepted + rank_in_group;
            area += (expected_error_total / accepted_at_rank as f64) / total as f64;
        }
        index = end;
    }

    let aurc = (aurc_supported && total > 0).then_some(area);
    let full_risk = always_answer.estimated_risk;
    let mut operating_points = Vec::new();
    let mut targets = config.target_coverages.clone();
    targets.sort_by(f64::total_cmp);
    for target in targets {
        if let Some(point) = curve
            .iter()
            .find(|point| point.point.coverage_in_full_cohort >= target)
        {
            operating_points.push(PropensityWeightedOperatingPoint {
                target_coverage: target,
                achieved_coverage: point.point.coverage_in_full_cohort,
                point: point.point.clone(),
                random_abstention_expected_risk: full_risk,
            });
        }
    }

    let mut thresholds = config.confidence_thresholds.clone();
    thresholds.sort_by(f64::total_cmp);
    let confidence_threshold_baselines = thresholds
        .into_iter()
        .map(|threshold| {
            let indices: Vec<usize> = sorted
                .iter()
                .enumerate()
                .filter_map(|(index, record)| {
                    (record.prediction.predicted_confidence >= threshold).then_some(index)
                })
                .collect();
            PropensityWeightedThresholdPoint {
                threshold,
                point: weighted_point(&sorted, &indices, config.minimum_sample_size),
            }
        })
        .collect();

    PropensityWeightedReport {
        metric_version: "symthaea-ipw-risk-coverage-v1".to_owned(),
        design_id: design.design_id.clone(),
        sampling_mechanism: design.mechanism,
        population_records: total,
        adjudicated_records: sorted
            .iter()
            .filter(|record| record.prediction.adjudicated_truth.is_some())
            .count(),
        kish_effective_sample_size: support_diagnostics(records)
            .kish_effective_sample_size
            .unwrap_or(0.0),
        maximum_inverse_probability_weight: support_diagnostics(records)
            .maximum_inverse_probability_weight
            .unwrap_or(0.0),
        aurc,
        aurc_convention: AURC_CONVENTION.to_owned(),
        curve,
        operating_points,
        confidence_threshold_baselines,
        always_answer,
        observed_model_policy,
        uncertainty_limitation: UNCERTAINTY_LIMITATION.to_owned(),
        assumptions: vec![
            concat!(
                "The target population is the complete frozen prediction cohort; ",
                "all candidate predictions are available for every record."
            )
            .to_owned(),
            "Every record has a known, strictly positive adjudication inclusion probability."
                .to_owned(),
            "The inclusion probability is assigned using only pre-label information.".to_owned(),
            concat!(
                "Conditional ignorability and the declared sampling design are external ",
                "assumptions; this evaluator cannot verify them."
            )
            .to_owned(),
            "Horvitz-Thompson estimates are not clipped and may leave [0, 1] in finite samples."
                .to_owned(),
            concat!(
                "Wilson intervals from the adjudicated-only lane are not valid for ",
                "these weighted estimates."
            )
            .to_owned(),
        ],
    }
}

fn weighted_point(
    records: &[&SelectiveObservationRecord],
    accepted_indices: &[usize],
    minimum_sample_size: usize,
) -> PropensityWeightedPoint {
    let mut weights = Vec::new();
    let mut estimated_error_total = 0.0_f64;
    for &index in accepted_indices {
        let record = records[index];
        if let Some(truth) = record.prediction.adjudicated_truth {
            let weight = 1.0 / record.adjudication_inclusion_probability.unwrap();
            weights.push(weight);
            if record.prediction.predicted_label != truth {
                estimated_error_total += weight;
            }
        }
    }
    let effective_n = kish_ess(&weights);
    let accepted_count = accepted_indices.len();
    let estimated_risk = if accepted_count > 0 && effective_n >= minimum_sample_size as f64 {
        Some(estimated_error_total / accepted_count as f64)
    } else {
        None
    };
    PropensityWeightedPoint {
        accepted_count_in_full_cohort: accepted_count,
        coverage_in_full_cohort: if records.is_empty() {
            0.0
        } else {
            accepted_count as f64 / records.len() as f64
        },
        independently_labelled_accepted_count: weights.len(),
        kish_effective_sample_size: effective_n,
        estimated_error_total,
        estimate_outside_unit_interval: estimated_risk.is_some_and(|risk| risk > 1.0),
        estimated_risk,
    }
}

fn kish_ess(weights: &[f64]) -> f64 {
    let sum = weights.iter().sum::<f64>();
    let sum_squared = weights.iter().map(|weight| weight * weight).sum::<f64>();
    if sum_squared == 0.0 {
        0.0
    } else {
        sum * sum / sum_squared
    }
}

fn stratify_population(
    records: &[SelectiveObservationRecord],
    design: &AdjudicationSamplingDesign,
    config: &RiskCoverageConfig,
) -> Vec<PropensityStratifiedReport> {
    let dimensions = [
        StratificationDimension::Domain,
        StratificationDimension::SourceFamily,
        StratificationDimension::DelayedOrCensored,
        StratificationDimension::Adjudicator,
    ];
    let mut reports = Vec::new();

    for dimension in dimensions {
        let mut groups = BTreeMap::<String, Vec<SelectiveObservationRecord>>::new();
        for record in records {
            let value = match dimension {
                StratificationDimension::Domain => record
                    .prediction
                    .domain
                    .as_deref()
                    .unwrap_or("unknown")
                    .to_owned(),
                StratificationDimension::SourceFamily => record
                    .prediction
                    .source_family
                    .as_deref()
                    .unwrap_or("unknown")
                    .to_owned(),
                StratificationDimension::DelayedOrCensored => {
                    match record.prediction.delayed_or_censored {
                        Some(true) => "delayed_or_censored".to_owned(),
                        Some(false) => "not_delayed_or_censored".to_owned(),
                        None => "unknown".to_owned(),
                    }
                }
                StratificationDimension::Adjudicator => record
                    .prediction
                    .adjudicator
                    .as_deref()
                    .unwrap_or("unknown")
                    .to_owned(),
            };
            groups.entry(value).or_default().push(record.clone());
        }

        for (value, group) in groups {
            let diagnostics = support_diagnostics(&group);
            let (report, reason) = if diagnostics.pending_certain_inclusion_records > 0 {
                (
                    None,
                    Some("stratum has pending records with certain inclusion".to_owned()),
                )
            } else if !diagnostics.positive_support_for_entire_cohort {
                (
                    None,
                    Some("stratum has zero, missing or invalid adjudication support".to_owned()),
                )
            } else if diagnostics.invalid_prediction_confidence_records > 0 {
                (
                    None,
                    Some("stratum contains invalid candidate confidence values".to_owned()),
                )
            } else if diagnostics.kish_effective_sample_size.unwrap_or(0.0)
                < config.minimum_sample_size as f64
            {
                (
                    None,
                    Some(
                        "stratum effective sample size is below the configured minimum"
                            .to_owned(),
                    ),
                )
            } else {
                (Some(compute_population_report(&group, design, config)), None)
            };
            reports.push(PropensityStratifiedReport {
                dimension,
                value,
                report,
                unavailable_reason: reason,
            });
        }
    }
    reports
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prediction(
        id: &str,
        confidence: f64,
        predicted: bool,
        truth: Option<bool>,
    ) -> SelectivePredictionRecord {
        SelectivePredictionRecord {
            id: id.to_owned(),
            predicted_label: predicted,
            predicted_confidence: confidence,
            adjudicated_truth: truth,
            model_abstained: false,
            abstention_reason: None,
            domain: Some("science".to_owned()),
            source_family: Some("family-a".to_owned()),
            delayed_or_censored: Some(false),
            adjudicator: Some("panel-1".to_owned()),
        }
    }

    fn design() -> AdjudicationSamplingDesign {
        AdjudicationSamplingDesign {
            design_id: "audit-v1".to_owned(),
            mechanism: AdjudicationSamplingMechanism::RandomizedAudit,
            prelabel_conditioning_variables: Vec::new(),
            design_manifest_ref: "artifact://frozen-audit-design-v1".to_owned(),
        }
    }

    fn config() -> RiskCoverageConfig {
        RiskCoverageConfig {
            minimum_sample_size: 1,
            target_coverages: vec![0.5, 0.8],
            confidence_thresholds: vec![0.5, 0.9],
            decision_costs: None,
            include_stratified_reports: true,
        }
    }

    fn observed(
        prediction: SelectivePredictionRecord,
        inclusion_probability: Option<f64>,
    ) -> SelectiveObservationRecord {
        SelectiveObservationRecord {
            prediction,
            adjudication_inclusion_probability: inclusion_probability,
        }
    }

    #[test]
    fn zero_or_missing_inclusion_probability_blocks_population_estimation() {
        let records = vec![
            observed(prediction("audited", 0.9, true, Some(true)), Some(0.5)),
            observed(prediction("never-audited", 0.2, true, None), Some(0.0)),
        ];
        let report = evaluate_selective_observation(&records, &design(), &config()).unwrap();
        assert!(!report.diagnostics.positive_support_for_entire_cohort);
        assert_eq!(report.population_report, None);
        assert!(report.unavailable_reason.unwrap().contains("not identifiable"));
    }

    #[test]
    fn inverse_probability_weight_recovers_known_finite_population_error_total() {
        let records = vec![
            observed(prediction("a", 0.9, true, Some(false)), Some(0.5)),
            observed(prediction("b", 0.8, true, Some(true)), Some(1.0)),
            observed(prediction("c", 0.2, true, None), Some(0.5)),
            observed(prediction("d", 0.1, true, Some(true)), Some(1.0)),
        ];
        let report = evaluate_selective_observation(&records, &design(), &config()).unwrap();
        let population = report.population_report.unwrap();
        // One observed error with inclusion probability 0.5 represents two
        // errors in the four-record target population under this fixture.
        assert_eq!(population.always_answer.estimated_error_total, 2.0);
        assert_eq!(population.always_answer.estimated_risk, Some(0.5));
        assert_eq!(population.always_answer.accepted_count_in_full_cohort, 4);
    }

    #[test]
    fn weighted_risk_can_exceed_one_and_is_never_clipped() {
        let records = vec![
            observed(prediction("a", 0.9, true, Some(false)), Some(0.1)),
            observed(prediction("b", 0.2, true, None), Some(1.0)),
        ];
        let report = evaluate_selective_observation(&records, &design(), &config()).unwrap();
        let point = report.population_report.unwrap().always_answer;
        assert_eq!(point.estimated_error_total, 10.0);
        assert_eq!(point.estimated_risk, Some(5.0));
        assert!(point.estimate_outside_unit_interval);
    }

    #[test]
    fn pending_records_with_positive_audit_probability_remain_in_population_denominator() {
        let records = vec![
            observed(prediction("a", 0.9, true, Some(true)), Some(0.5)),
            observed(prediction("b", 0.2, true, None), Some(0.1)),
        ];
        let report = evaluate_selective_observation(&records, &design(), &config()).unwrap();
        let point = report.population_report.unwrap().always_answer;
        assert_eq!(point.accepted_count_in_full_cohort, 2);
        assert_eq!(point.independently_labelled_accepted_count, 1);
        assert_eq!(point.coverage_in_full_cohort, 1.0);
    }

    #[test]
    fn incomplete_or_invalid_propensities_fail_closed() {
        for bad in [None, Some(-0.1), Some(0.0), Some(1.1), Some(f64::NAN)] {
            let records = vec![
                observed(prediction("labelled", 0.8, true, Some(true)), Some(0.5)),
                observed(prediction("bad", 0.2, true, None), bad),
            ];
            let report =
                evaluate_selective_observation(&records, &design(), &config()).unwrap();
            assert!(report.population_report.is_none());
        }
    }

    #[test]
    fn empty_sampling_design_id_is_rejected() {
        let design = AdjudicationSamplingDesign {
            design_id: String::new(),
            ..design()
        };
        assert_eq!(
            evaluate_selective_observation(&[], &design, &config()).unwrap_err(),
            SelectiveObservationError::EmptyDesignId
        );
    }

    #[test]
    fn known_propensity_design_requires_prelabel_conditioning_variables() {
        let design = AdjudicationSamplingDesign {
            mechanism: AdjudicationSamplingMechanism::KnownPropensityConditionalOnPrelabelHistory,
            prelabel_conditioning_variables: vec!["".to_owned()],
            ..design()
        };
        assert_eq!(
            evaluate_selective_observation(&[], &design, &config()).unwrap_err(),
            SelectiveObservationError::EmptyConditioningVariable
        );
    }

    #[test]
    fn invalid_prediction_confidence_blocks_population_estimation() {
        let invalid = prediction("invalid", f64::NAN, true, Some(true));
        let records = vec![
            observed(invalid, Some(1.0)),
            observed(prediction("valid", 0.5, true, Some(true)), Some(1.0)),
        ];
        let report = evaluate_selective_observation(&records, &design(), &config()).unwrap();
        assert_eq!(report.diagnostics.invalid_prediction_confidence_records, 1);
        assert!(report.population_report.is_none());
    }

    #[test]
    fn certain_inclusion_cannot_remain_pending_at_the_frozen_cutoff() {
        let records = vec![
            observed(prediction("labelled", 0.9, true, Some(true)), Some(1.0)),
            observed(prediction("pending", 0.2, true, None), Some(1.0)),
        ];
        let report = evaluate_selective_observation(&records, &design(), &config()).unwrap();
        assert_eq!(report.diagnostics.pending_certain_inclusion_records, 1);
        assert!(report.population_report.is_none());
    }

    #[test]
    fn duplicate_cohort_ids_are_rejected() {
        let records = vec![
            observed(prediction("same", 0.9, true, Some(true)), Some(1.0)),
            observed(prediction("same", 0.8, true, Some(false)), Some(1.0)),
        ];
        assert_eq!(
            evaluate_selective_observation(&records, &design(), &config()).unwrap_err(),
            SelectiveObservationError::DuplicateRecordId("same".to_owned())
        );
    }

    #[test]
    fn propensity_report_is_invariant_to_row_order_within_confidence_ties() {
        let records = vec![
            observed(prediction("a", 0.9, true, Some(false)), Some(0.5)),
            observed(prediction("b", 0.9, true, Some(true)), Some(1.0)),
            observed(prediction("c", 0.2, true, Some(true)), Some(0.5)),
        ];
        let mut reversed = records.clone();
        reversed.reverse();
        let forward = evaluate_selective_observation(&records, &design(), &config()).unwrap();
        let backward = evaluate_selective_observation(&reversed, &design(), &config()).unwrap();
        assert_eq!(
            forward.population_report.unwrap().aurc,
            backward.population_report.unwrap().aurc
        );
    }

    #[test]
    fn stratified_reports_preserve_unknown_groups_and_do_not_hide_zero_support() {
        let mut zero_support = prediction("b", 0.2, true, None);
        zero_support.domain = None;
        let records = vec![
            observed(prediction("a", 0.9, true, Some(true)), Some(1.0)),
            observed(zero_support, Some(0.0)),
        ];
        let report = evaluate_selective_observation(&records, &design(), &config()).unwrap();
        assert!(report.stratified.iter().any(|group| group.value == "unknown"));
        assert!(report.stratified.iter().any(|group| group.report.is_none()));
    }
}
