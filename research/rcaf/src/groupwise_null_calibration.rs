//! Stratum-conditional calibration and a worst-group guard for RCAF.
//!
//! A pooled null threshold can hide subgroup miscalibration. This module calibrates one threshold
//! per predeclared acquisition stratum and also computes a single conservative guard threshold as
//! the maximum stratum threshold. An unseen or under-resolved stratum has no threshold: it abstains.

use crate::held_out_null_calibration::{
    HeldOutEvaluationObservation, NullCalibrationObservation, ProportionInterval, ESTIMATOR_IDS,
};
use std::collections::{BTreeMap, HashSet};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct StratifiedNullCalibrationObservation {
    pub stratum_id: String,
    pub observation: NullCalibrationObservation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StratifiedHeldOutEvaluationObservation {
    pub stratum_id: String,
    pub observation: HeldOutEvaluationObservation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GroupwiseCalibrationConfig {
    pub calibration_split_id: String,
    pub evaluation_split_id: String,
    /// Predeclared acquisition strata, e.g. site × echo × motion-burden classes.
    pub expected_strata: Vec<String>,
    pub alpha: f64,
    pub minimum_direction_cosine: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GroupwiseRateSet {
    pub held_out_false_positive_rate: ProportionInterval,
    pub signal_detection_rate: ProportionInterval,
    pub directionally_correct_signal_rate: ProportionInterval,
    pub wrong_direction_acceptance_rate: ProportionInterval,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GroupwiseStratumEstimatorSummary {
    pub estimator_id: &'static str,
    pub calibration_replicates: usize,
    pub evaluation_replicates: usize,
    pub minimum_resolvable_alpha: f64,
    /// Group-conditional threshold; None means this stratum is under-resolved and must abstain.
    pub stratum_critical_value: Option<f64>,
    /// The shared max-over-strata guard threshold; None if any expected stratum is under-resolved.
    pub worst_group_guard_critical_value: Option<f64>,
    /// Metrics using the stratum-specific threshold.
    pub stratum_threshold_rates: Option<GroupwiseRateSet>,
    /// Metrics using the single conservative max-over-strata threshold.
    pub worst_group_guard_rates: Option<GroupwiseRateSet>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GroupwiseStratumSummary {
    pub stratum_id: String,
    pub estimators: Vec<GroupwiseStratumEstimatorSummary>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorstStratumRate {
    pub stratum_id: String,
    pub rate: ProportionInterval,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GroupwiseEstimatorGuardSummary {
    pub estimator_id: &'static str,
    pub all_strata_resolvable: bool,
    pub all_strata_evaluated: bool,
    pub worst_observed_adaptive_fpr: Option<WorstStratumRate>,
    pub worst_observed_guard_fpr: Option<WorstStratumRate>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnseenStratumSummary {
    pub stratum_id: String,
    pub evaluation_replicates: usize,
    pub disposition: &'static str,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GroupwiseNullCalibrationReport {
    pub schema_version: &'static str,
    pub calibration_split_id: String,
    pub evaluation_split_id: String,
    pub alpha: f64,
    pub minimum_direction_cosine: f64,
    pub calibration_replicates: usize,
    pub evaluation_replicates: usize,
    pub expected_strata: Vec<String>,
    pub unseen_evaluation_strata: Vec<UnseenStratumSummary>,
    pub strata: Vec<GroupwiseStratumSummary>,
    pub estimators: Vec<GroupwiseEstimatorGuardSummary>,
}

impl GroupwiseNullCalibrationReport {
    pub fn to_json_pretty(&self) -> String {
        let strata = self
            .strata
            .iter()
            .map(render_stratum)
            .collect::<Vec<_>>()
            .join(",\n");
        let unseen = self
            .unseen_evaluation_strata
            .iter()
            .map(|item| {
                format!(
                    "    {{\"stratum_id\": \"{}\", \"evaluation_replicates\": {}, \"disposition\": \"{}\"}}",
                    item.stratum_id, item.evaluation_replicates, item.disposition
                )
            })
            .collect::<Vec<_>>()
            .join(",\n");
        let estimators = self
            .estimators
            .iter()
            .map(|item| {
                format!(
                    concat!(
                        "    {{\n",
                        "      \"estimator_id\": \"{}\",\n",
                        "      \"all_strata_resolvable\": {},\n",
                        "      \"all_strata_evaluated\": {},\n",
                        "      \"worst_observed_adaptive_fpr\": {},\n",
                        "      \"worst_observed_guard_fpr\": {}\n",
                        "    }}"
                    ),
                    item.estimator_id,
                    item.all_strata_resolvable,
                    item.all_strata_evaluated,
                    worst_rate_json(item.worst_observed_adaptive_fpr.as_ref()),
                    worst_rate_json(item.worst_observed_guard_fpr.as_ref()),
                )
            })
            .collect::<Vec<_>>()
            .join(",\n");
        format!(
            concat!(
                "{{\n",
                "  \"schema_version\": \"{}\",\n",
                "  \"source_status\": \"synthetic_groupwise_null_calibration\",\n",
                "  \"calibration_split_id\": \"{}\",\n",
                "  \"evaluation_split_id\": \"{}\",\n",
                "  \"alpha\": {:.12},\n",
                "  \"minimum_direction_cosine\": {:.12},\n",
                "  \"calibration_replicates\": {},\n",
                "  \"evaluation_replicates\": {},\n",
                "  \"expected_strata\": [{}],\n",
                "  \"unseen_evaluation_strata\": [\n{}\n  ],\n",
                "  \"strata\": [\n{}\n  ],\n",
                "  \"estimators\": [\n{}\n  ],\n",
                "  \"claim_limits\": [\n",
                "    \"each predeclared stratum receives its own calibration threshold\",\n",
                "    \"the shared guard threshold is the maximum stratum threshold and is undefined if any stratum is under-resolved\",\n",
                "    \"unseen strata abstain and are excluded from calibrated-rate estimates\",\n",
                "    \"validity is conditional on within-stratum exchangeability; this does not solve arbitrary continuous distribution shift\",\n",
                "    \"synthetic groupwise calibration only; no empirical participant/site qualification\"\n",
                "  ]\n",
                "}}\n"
            ),
            self.schema_version,
            self.calibration_split_id,
            self.evaluation_split_id,
            self.alpha,
            self.minimum_direction_cosine,
            self.calibration_replicates,
            self.evaluation_replicates,
            self.expected_strata
                .iter()
                .map(|value| format!("\"{value}\""))
                .collect::<Vec<_>>()
                .join(", "),
            unseen,
            strata,
            estimators,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroupwiseCalibrationError {
    InvalidAlpha,
    InvalidDirectionThreshold,
    EmptyExpectedStrata,
    DuplicateExpectedStratum,
    InvalidIdentifier(&'static str),
    InvalidStratumIdentifier { stratum_id: String },
    UnexpectedCalibrationStratum { stratum_id: String },
    DuplicateReplicateId(&'static str),
    CalibrationEvaluationOverlap,
    InvalidStatistic { replicate_id: String },
    InvalidDirectionCosine { replicate_id: String },
}

impl fmt::Display for GroupwiseCalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAlpha => write!(f, "alpha must be finite and in (0, 0.5]"),
            Self::InvalidDirectionThreshold => {
                write!(f, "minimum direction cosine must be finite and in [-1, 1]")
            }
            Self::EmptyExpectedStrata => write!(f, "at least one expected stratum is required"),
            Self::DuplicateExpectedStratum => write!(f, "expected stratum identifiers must be unique"),
            Self::InvalidIdentifier(field) => {
                write!(f, "{field} must be a non-empty stable ASCII identifier")
            }
            Self::InvalidStratumIdentifier { stratum_id } => {
                write!(f, "invalid stratum identifier: {stratum_id}")
            }
            Self::UnexpectedCalibrationStratum { stratum_id } => {
                write!(f, "calibration observation belongs to undeclared stratum {stratum_id}")
            }
            Self::DuplicateReplicateId(split) => {
                write!(f, "replicate identifiers must be unique within {split}")
            }
            Self::CalibrationEvaluationOverlap => {
                write!(f, "calibration and evaluation replicate identifiers must be disjoint")
            }
            Self::InvalidStatistic { replicate_id } => {
                write!(f, "replicate {replicate_id} contains a non-finite or negative statistic")
            }
            Self::InvalidDirectionCosine { replicate_id } => {
                write!(f, "replicate {replicate_id} has a direction cosine outside [-1, 1]")
            }
        }
    }
}

impl std::error::Error for GroupwiseCalibrationError {}

/// Calibrate per-stratum thresholds and a common worst-group threshold.
///
/// The common guard is max(q_g) across the predeclared strata for each estimator. A common
/// threshold is emitted only when every expected stratum has enough calibration replicates to
/// resolve alpha. Evaluation observations in unknown strata are retained as explicit abstentions;
/// they never inherit a threshold from another stratum.
pub fn calibrate_groupwise_null(
    calibration: &[StratifiedNullCalibrationObservation],
    evaluation: &[StratifiedHeldOutEvaluationObservation],
    config: &GroupwiseCalibrationConfig,
) -> Result<GroupwiseNullCalibrationReport, GroupwiseCalibrationError> {
    validate_config(config)?;
    let mut expected = config.expected_strata.clone();
    expected.sort();

    let expected_set = expected.iter().map(String::as_str).collect::<HashSet<_>>();
    let mut calibration_ids = HashSet::with_capacity(calibration.len());
    let mut calibration_by_stratum = BTreeMap::<String, Vec<&NullCalibrationObservation>>::new();
    for item in calibration {
        validate_id(&item.observation.replicate_id, "calibration")?;
        validate_id(&item.stratum_id, "stratum")
            .map_err(|_| GroupwiseCalibrationError::InvalidStratumIdentifier {
                stratum_id: item.stratum_id.clone(),
            })?;
        if !expected_set.contains(item.stratum_id.as_str()) {
            return Err(GroupwiseCalibrationError::UnexpectedCalibrationStratum {
                stratum_id: item.stratum_id.clone(),
            });
        }
        if !calibration_ids.insert(item.observation.replicate_id.as_str()) {
            return Err(GroupwiseCalibrationError::DuplicateReplicateId("calibration"));
        }
        if item
            .observation
            .statistics
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(GroupwiseCalibrationError::InvalidStatistic {
                replicate_id: item.observation.replicate_id.clone(),
            });
        }
        calibration_by_stratum
            .entry(item.stratum_id.clone())
            .or_default()
            .push(&item.observation);
    }

    let mut evaluation_ids = HashSet::with_capacity(evaluation.len());
    let mut evaluation_by_stratum =
        BTreeMap::<String, Vec<&HeldOutEvaluationObservation>>::new();
    for item in evaluation {
        validate_id(&item.observation.replicate_id, "evaluation")?;
        validate_id(&item.stratum_id, "stratum")
            .map_err(|_| GroupwiseCalibrationError::InvalidStratumIdentifier {
                stratum_id: item.stratum_id.clone(),
            })?;
        if !evaluation_ids.insert(item.observation.replicate_id.as_str()) {
            return Err(GroupwiseCalibrationError::DuplicateReplicateId("evaluation"));
        }
        if calibration_ids.contains(item.observation.replicate_id.as_str()) {
            return Err(GroupwiseCalibrationError::CalibrationEvaluationOverlap);
        }
        if item
            .observation
            .null_statistics
            .iter()
            .chain(item.observation.signal_statistics.iter())
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(GroupwiseCalibrationError::InvalidStatistic {
                replicate_id: item.observation.replicate_id.clone(),
            });
        }
        if !item.observation.signal_directional_cosine.is_finite()
            || !(-1.0..=1.0).contains(&item.observation.signal_directional_cosine)
        {
            return Err(GroupwiseCalibrationError::InvalidDirectionCosine {
                replicate_id: item.observation.replicate_id.clone(),
            });
        }
        evaluation_by_stratum
            .entry(item.stratum_id.clone())
            .or_default()
            .push(&item.observation);
    }

    let unseen_evaluation_strata = evaluation_by_stratum
        .iter()
        .filter(|(stratum, _)| !expected_set.contains(stratum.as_str()))
        .map(|(stratum, observations)| UnseenStratumSummary {
            stratum_id: stratum.clone(),
            evaluation_replicates: observations.len(),
            disposition: "abstain_unseen_stratum",
        })
        .collect::<Vec<_>>();

    let mut thresholds_by_estimator = [[None; 1]; 3];
    let mut thresholds = vec![[None; 3]; expected.len()];
    let mut minimum_resolvable = vec![1.0; expected.len()];
    for (stratum_index, stratum) in expected.iter().enumerate() {
        let observations = calibration_by_stratum.get(stratum).map_or(&[][..], Vec::as_slice);
        minimum_resolvable[stratum_index] = 1.0 / (observations.len() as f64 + 1.0);
        for estimator_index in 0..ESTIMATOR_IDS.len() {
            thresholds[stratum_index][estimator_index] = upper_tail_threshold(
                observations
                    .iter()
                    .map(|item| item.statistics[estimator_index])
                    .collect(),
                config.alpha,
            );
        }
    }
    for estimator_index in 0..ESTIMATOR_IDS.len() {
        let all_resolvable = thresholds
            .iter()
            .all(|values| values[estimator_index].is_some());
        thresholds_by_estimator[estimator_index][0] = all_resolvable.then(|| {
            thresholds
                .iter()
                .filter_map(|values| values[estimator_index])
                .fold(f64::NEG_INFINITY, f64::max)
        });
    }

    let mut strata = Vec::with_capacity(expected.len());
    for (stratum_index, stratum) in expected.iter().enumerate() {
        let eval = evaluation_by_stratum
            .get(stratum)
            .map_or(&[][..], Vec::as_slice);
        let estimators = ESTIMATOR_IDS
            .iter()
            .enumerate()
            .map(|(estimator_index, estimator_id)| {
                let critical = thresholds[stratum_index][estimator_index];
                let guard_critical = thresholds_by_estimator[estimator_index][0];
                GroupwiseStratumEstimatorSummary {
                    estimator_id,
                    calibration_replicates: calibration_by_stratum
                        .get(stratum)
                        .map_or(0, Vec::len),
                    evaluation_replicates: eval.len(),
                    minimum_resolvable_alpha: minimum_resolvable[stratum_index],
                    stratum_critical_value: critical,
                    worst_group_guard_critical_value: guard_critical,
                    stratum_threshold_rates: rates_if_resolved(
                        eval,
                        estimator_index,
                        critical,
                        config.minimum_direction_cosine,
                    ),
                    worst_group_guard_rates: rates_if_resolved(
                        eval,
                        estimator_index,
                        guard_critical,
                        config.minimum_direction_cosine,
                    ),
                }
            })
            .collect();
        strata.push(GroupwiseStratumSummary {
            stratum_id: stratum.clone(),
            estimators,
        });
    }

    let estimators = ESTIMATOR_IDS
        .iter()
        .enumerate()
        .map(|(estimator_index, estimator_id)| {
            let all_strata_resolvable = thresholds
                .iter()
                .all(|values| values[estimator_index].is_some());
            let all_strata_evaluated = expected.iter().all(|stratum| {
                evaluation_by_stratum
                    .get(stratum)
                    .is_some_and(|items| !items.is_empty())
            });
            GroupwiseEstimatorGuardSummary {
                estimator_id,
                all_strata_resolvable,
                all_strata_evaluated,
                worst_observed_adaptive_fpr: worst_observed_rate(
                    &strata,
                    estimator_index,
                    false,
                ),
                worst_observed_guard_fpr: worst_observed_rate(
                    &strata,
                    estimator_index,
                    true,
                ),
            }
        })
        .collect();

    Ok(GroupwiseNullCalibrationReport {
        schema_version: "rcaf.groupwise-null-calibration/0.1.0",
        calibration_split_id: config.calibration_split_id.clone(),
        evaluation_split_id: config.evaluation_split_id.clone(),
        alpha: config.alpha,
        minimum_direction_cosine: config.minimum_direction_cosine,
        calibration_replicates: calibration.len(),
        evaluation_replicates: evaluation.len(),
        expected_strata: expected,
        unseen_evaluation_strata,
        strata,
        estimators,
    })
}

fn validate_config(config: &GroupwiseCalibrationConfig) -> Result<(), GroupwiseCalibrationError> {
    validate_id(&config.calibration_split_id, "calibration_split_id")?;
    validate_id(&config.evaluation_split_id, "evaluation_split_id")?;
    if config.calibration_split_id == config.evaluation_split_id {
        return Err(GroupwiseCalibrationError::CalibrationEvaluationOverlap);
    }
    if !config.alpha.is_finite() || !(0.0..=0.5).contains(&config.alpha) || config.alpha == 0.0 {
        return Err(GroupwiseCalibrationError::InvalidAlpha);
    }
    if !config.minimum_direction_cosine.is_finite()
        || !(-1.0..=1.0).contains(&config.minimum_direction_cosine)
    {
        return Err(GroupwiseCalibrationError::InvalidDirectionThreshold);
    }
    if config.expected_strata.is_empty() {
        return Err(GroupwiseCalibrationError::EmptyExpectedStrata);
    }
    let mut seen = HashSet::with_capacity(config.expected_strata.len());
    for stratum in &config.expected_strata {
        validate_id(stratum, "expected_strata")
            .map_err(|_| GroupwiseCalibrationError::InvalidStratumIdentifier {
                stratum_id: stratum.clone(),
            })?;
        if !seen.insert(stratum.as_str()) {
            return Err(GroupwiseCalibrationError::DuplicateExpectedStratum);
        }
    }
    Ok(())
}

fn validate_id(value: &str, field: &'static str) -> Result<(), GroupwiseCalibrationError> {
    if value.trim().is_empty()
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-')
        })
    {
        return Err(GroupwiseCalibrationError::InvalidIdentifier(field));
    }
    Ok(())
}

fn upper_tail_threshold(mut values: Vec<f64>, alpha: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let k = (alpha * (values.len() as f64 + 1.0)).floor() as usize;
    (k > 0).then(|| values[values.len() - k])
}

fn is_rejected(statistic: f64, threshold: Option<f64>) -> bool {
    threshold.is_some_and(|critical| statistic > critical)
}

fn rates_if_resolved(
    evaluation: &[&HeldOutEvaluationObservation],
    estimator_index: usize,
    threshold: Option<f64>,
    minimum_direction_cosine: f64,
) -> Option<GroupwiseRateSet> {
    if evaluation.is_empty() || threshold.is_none() {
        return None;
    }
    let threshold = threshold.unwrap();
    let false_positives = evaluation
        .iter()
        .filter(|item| is_rejected(item.null_statistics[estimator_index], Some(threshold)))
        .count();
    let detections = evaluation
        .iter()
        .filter(|item| is_rejected(item.signal_statistics[estimator_index], Some(threshold)))
        .count();
    let directionally_correct = evaluation
        .iter()
        .filter(|item| {
            is_rejected(item.signal_statistics[estimator_index], Some(threshold))
                && item.signal_directional_cosine >= minimum_direction_cosine
        })
        .count();
    let wrong_direction = evaluation
        .iter()
        .filter(|item| {
            is_rejected(item.signal_statistics[estimator_index], Some(threshold))
                && item.signal_directional_cosine < minimum_direction_cosine
        })
        .count();
    Some(GroupwiseRateSet {
        held_out_false_positive_rate: wilson_interval(false_positives, evaluation.len()),
        signal_detection_rate: wilson_interval(detections, evaluation.len()),
        directionally_correct_signal_rate: wilson_interval(
            directionally_correct,
            evaluation.len(),
        ),
        wrong_direction_acceptance_rate: wilson_interval(wrong_direction, evaluation.len()),
    })
}

fn wilson_interval(successes: usize, trials: usize) -> ProportionInterval {
    let n = trials as f64;
    let p = successes as f64 / n;
    let z = 1.959_963_984_540_054;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denominator;
    ProportionInterval {
        successes,
        trials,
        estimate: p,
        wilson_95_lower: (center - half_width).clamp(0.0, 1.0),
        wilson_95_upper: (center + half_width).clamp(0.0, 1.0),
    }
}

fn worst_observed_rate(
    strata: &[GroupwiseStratumSummary],
    estimator_index: usize,
    guard: bool,
) -> Option<WorstStratumRate> {
    strata
        .iter()
        .filter_map(|stratum| {
            let summary = &stratum.estimators[estimator_index];
            let rates = if guard {
                summary.worst_group_guard_rates.as_ref()
            } else {
                summary.stratum_threshold_rates.as_ref()
            }?;
            Some(WorstStratumRate {
                stratum_id: stratum.stratum_id.clone(),
                rate: rates.held_out_false_positive_rate.clone(),
            })
        })
        .max_by(|left, right| left.rate.estimate.total_cmp(&right.rate.estimate))
}

fn proportion_json(value: &ProportionInterval) -> String {
    format!(
        "{{\"successes\": {}, \"trials\": {}, \"estimate\": {:.12}, \"wilson_95_lower\": {:.12}, \"wilson_95_upper\": {:.12}}}",
        value.successes,
        value.trials,
        value.estimate,
        value.wilson_95_lower,
        value.wilson_95_upper
    )
}

fn optional_rate_json(value: Option<&ProportionInterval>) -> String {
    value.map(proportion_json).unwrap_or_else(|| "null".to_string())
}

fn worst_rate_json(value: Option<&WorstStratumRate>) -> String {
    value
        .map(|item| {
            format!(
                "{{\"stratum_id\": \"{}\", \"rate\": {}}}",
                item.stratum_id,
                proportion_json(&item.rate)
            )
        })
        .unwrap_or_else(|| "null".to_string())
}

fn render_stratum(stratum: &GroupwiseStratumSummary) -> String {
    let estimators = stratum
        .estimators
        .iter()
        .map(|item| {
            format!(
                concat!(
                    "      {{\n",
                    "        \"estimator_id\": \"{}\",\n",
                    "        \"calibration_replicates\": {},\n",
                    "        \"evaluation_replicates\": {},\n",
                    "        \"minimum_resolvable_alpha\": {:.12},\n",
                    "        \"stratum_critical_value\": {},\n",
                    "        \"worst_group_guard_critical_value\": {},\n",
                    "        \"stratum_threshold_rates\": {},\n",
                    "        \"worst_group_guard_rates\": {}\n",
                    "      }}"
                ),
                item.estimator_id,
                item.calibration_replicates,
                item.evaluation_replicates,
                item.minimum_resolvable_alpha,
                option_number(item.stratum_critical_value),
                option_number(item.worst_group_guard_critical_value),
                item.stratum_threshold_rates
                    .as_ref()
                    .map(|rates| render_rates(rates))
                    .unwrap_or_else(|| "null".to_string()),
                item.worst_group_guard_rates
                    .as_ref()
                    .map(|rates| render_rates(rates))
                    .unwrap_or_else(|| "null".to_string()),
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    format!(
        concat!(
            "    {{\n",
            "      \"stratum_id\": \"{}\",\n",
            "      \"estimators\": [\n{}\n      ]\n",
            "    }}"
        ),
        stratum.stratum_id,
        estimators
    )
}

fn render_rates(rates: &GroupwiseRateSet) -> String {
    format!(
        concat!(
            "{{\n",
            "  \"held_out_false_positive_rate\": {},\n",
            "  \"signal_detection_rate\": {},\n",
            "  \"directionally_correct_signal_rate\": {},\n",
            "  \"wrong_direction_acceptance_rate\": {}\n",
            "}}"
        ),
        proportion_json(&rates.held_out_false_positive_rate),
        proportion_json(&rates.signal_detection_rate),
        proportion_json(&rates.directionally_correct_signal_rate),
        proportion_json(&rates.wrong_direction_acceptance_rate),
    )
}

fn option_number(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.12}"))
        .unwrap_or_else(|| "null".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> GroupwiseCalibrationConfig {
        GroupwiseCalibrationConfig {
            calibration_split_id: "cal-sites-v1".to_string(),
            evaluation_split_id: "eval-sites-v1".to_string(),
            expected_strata: vec![
                "low_motion_single_echo".to_string(),
                "high_motion_single_echo".to_string(),
                "multi_echo".to_string(),
            ],
            alpha: 0.05,
            minimum_direction_cosine: 0.0,
        }
    }

    fn calibration() -> Vec<StratifiedNullCalibrationObservation> {
        let groups = [
            ("low_motion_single_echo", 1.0),
            ("high_motion_single_echo", 2.0),
            ("multi_echo", 3.0),
        ];
        groups
            .iter()
            .flat_map(|(stratum, scale)| {
                (0..99).map(move |index| {
                    let value = (index + 1) as f64 / 100.0 * scale;
                    StratifiedNullCalibrationObservation {
                        stratum_id: (*stratum).to_string(),
                        observation: NullCalibrationObservation {
                            replicate_id: format!("cal-{stratum}-{index:03}"),
                            statistics: [value, value * 0.8, value * 0.6],
                        },
                    }
                })
            })
            .collect()
    }

    fn evaluation() -> Vec<StratifiedHeldOutEvaluationObservation> {
        let groups = [
            ("low_motion_single_echo", 1.0),
            ("high_motion_single_echo", 2.0),
            ("multi_echo", 3.0),
        ];
        groups
            .iter()
            .flat_map(|(stratum, scale)| {
                (0..100).map(move |index| {
                    let value = (index + 1) as f64 / 100.0 * scale;
                    let signal = (0.90 + index as f64 / 1000.0) * scale;
                    StratifiedHeldOutEvaluationObservation {
                        stratum_id: (*stratum).to_string(),
                        observation: HeldOutEvaluationObservation {
                            replicate_id: format!("eval-{stratum}-{index:03}"),
                            null_statistics: [value, value * 0.8, value * 0.6],
                            signal_statistics: [signal, signal * 0.95, signal * 0.9],
                            signal_directional_cosine: if index % 10 == 0 { -0.2 } else { 0.8 },
                        },
                    }
                })
            })
            .collect()
    }

    #[test]
    fn calibrates_each_stratum_and_builds_a_conservative_common_guard() {
        let report = calibrate_groupwise_null(&calibration(), &evaluation(), &config()).unwrap();
        assert_eq!(report.strata.len(), 3);
        assert_eq!(report.estimators.len(), 3);
        for estimator_index in 0..3 {
            let first = report.strata[0].estimators[estimator_index].stratum_critical_value.unwrap();
            let second = report.strata[1].estimators[estimator_index].stratum_critical_value.unwrap();
            let third = report.strata[2].estimators[estimator_index].stratum_critical_value.unwrap();
            assert!(first < second && second < third);
            assert_eq!(
                report.estimators[estimator_index]
                    .worst_observed_guard_fpr
                    .as_ref()
                    .unwrap()
                    .rate
                    .successes,
                5
            );
            assert!(report.estimators[estimator_index].all_strata_resolvable);
            assert!(report.estimators[estimator_index].all_strata_evaluated);
        }
    }

    #[test]
    fn each_stratum_controls_its_own_null_rate_and_direction_errors() {
        let report = calibrate_groupwise_null(&calibration(), &evaluation(), &config()).unwrap();
        for stratum in &report.strata {
            for estimator in &stratum.estimators {
                assert_eq!(
                    estimator
                        .stratum_threshold_rates
                        .as_ref()
                        .unwrap()
                        .held_out_false_positive_rate
                        .successes,
                    5
                );
                assert_eq!(
                    estimator
                        .stratum_threshold_rates
                        .as_ref()
                        .unwrap()
                        .wrong_direction_acceptance_rate
                        .successes,
                    10
                );
            }
        }
    }

    #[test]
    fn unseen_strata_are_reported_as_abstentions_not_borrowed_thresholds() {
        let mut eval = evaluation();
        eval.extend((0..10).map(|index| StratifiedHeldOutEvaluationObservation {
            stratum_id: "new_scanner".to_string(),
            observation: HeldOutEvaluationObservation {
                replicate_id: format!("eval-new-{index:03}"),
                null_statistics: [100.0; 3],
                signal_statistics: [100.0; 3],
                signal_directional_cosine: 1.0,
            },
        }));
        let report = calibrate_groupwise_null(&calibration(), &eval, &config()).unwrap();
        assert_eq!(report.unseen_evaluation_strata.len(), 1);
        assert_eq!(report.unseen_evaluation_strata[0].evaluation_replicates, 10);
        assert_eq!(
            report.unseen_evaluation_strata[0].disposition,
            "abstain_unseen_stratum"
        );
    }

    #[test]
    fn under_resolved_stratum_disables_the_common_guard_and_its_local_rate() {
        let mut calibration = calibration();
        calibration.retain(|item| {
            item.stratum_id != "high_motion_single_echo"
                || item.observation.replicate_id.ends_with("000")
                || item.observation.replicate_id.ends_with("001")
                || item.observation.replicate_id.ends_with("002")
                || item.observation.replicate_id.ends_with("003")
        });
        let report = calibrate_groupwise_null(&calibration, &evaluation(), &config()).unwrap();
        let estimator = &report.estimators[0];
        assert!(!estimator.all_strata_resolvable);
        assert!(estimator.worst_observed_guard_fpr.is_none());
        let high_motion = report
            .strata
            .iter()
            .find(|item| item.stratum_id == "high_motion_single_echo")
            .unwrap();
        assert!(high_motion.estimators[0].stratum_critical_value.is_none());
        assert!(high_motion.estimators[0].stratum_threshold_rates.is_none());
        assert!(high_motion.estimators[0].worst_group_guard_rates.is_none());
    }

    #[test]
    fn missing_evaluation_stratum_is_marked_incomplete() {
        let eval = evaluation()
            .into_iter()
            .filter(|item| item.stratum_id != "multi_echo")
            .collect::<Vec<_>>();
        let report = calibrate_groupwise_null(&calibration(), &eval, &config()).unwrap();
        assert!(!report.estimators[0].all_strata_evaluated);
        assert!(report.strata[2].estimators[0].stratum_threshold_rates.is_none());
    }

    #[test]
    fn duplicate_and_overlapping_ids_fail_closed_across_strata() {
        let mut duplicate = calibration();
        duplicate[1].observation.replicate_id = duplicate[0].observation.replicate_id.clone();
        assert!(matches!(
            calibrate_groupwise_null(&duplicate, &evaluation(), &config()),
            Err(GroupwiseCalibrationError::DuplicateReplicateId("calibration"))
        ));
        let mut overlap = evaluation();
        overlap[0].observation.replicate_id = calibration()[0].observation.replicate_id.clone();
        assert!(matches!(
            calibrate_groupwise_null(&calibration(), &overlap, &config()),
            Err(GroupwiseCalibrationError::CalibrationEvaluationOverlap)
        ));
    }

    #[test]
    fn invalid_strata_and_unexpected_calibration_groups_fail_closed() {
        let mut config = config();
        config.expected_strata.push("low_motion_single_echo".to_string());
        assert_eq!(
            calibrate_groupwise_null(&calibration(), &evaluation(), &config),
            Err(GroupwiseCalibrationError::DuplicateExpectedStratum)
        );
        let mut unexpected = calibration();
        unexpected[0].stratum_id = "undeclared_site".to_string();
        assert!(matches!(
            calibrate_groupwise_null(&unexpected, &evaluation(), &config()),
            Err(GroupwiseCalibrationError::DuplicateExpectedStratum)
        ));
    }

    #[test]
    fn invalid_alpha_and_direction_threshold_fail_closed() {
        let mut config = config();
        config.alpha = 0.0;
        assert_eq!(
            calibrate_groupwise_null(&calibration(), &evaluation(), &config),
            Err(GroupwiseCalibrationError::InvalidAlpha)
        );
        let mut config = config();
        config.minimum_direction_cosine = 1.2;
        assert_eq!(
            calibrate_groupwise_null(&calibration(), &evaluation(), &config),
            Err(GroupwiseCalibrationError::InvalidDirectionThreshold)
        );
    }

    #[test]
    fn report_json_exposes_worst_group_guard_and_abstentions() {
        let mut eval = evaluation();
        eval.extend((0..2).map(|index| StratifiedHeldOutEvaluationObservation {
            stratum_id: "unseen_site".to_string(),
            observation: HeldOutEvaluationObservation {
                replicate_id: format!("eval-unseen-{index:03}"),
                null_statistics: [0.8; 3],
                signal_statistics: [0.9; 3],
                signal_directional_cosine: 0.9,
            },
        }));
        let report = calibrate_groupwise_null(&calibration(), &eval, &config()).unwrap();
        let json = report.to_json_pretty();
        assert!(json.contains("\"abstain_unseen_stratum\""));
        assert!(json.contains("worst_group_guard_critical_value"));
        assert!(json.contains("worst_observed_guard_fpr"));
    }
}
