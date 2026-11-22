//! Independent held-out null calibration for estimator response statistics.
//!
//! Calibration and evaluation replicate identifiers must be disjoint. The critical value uses
//! a conservative finite-sample upper-tail conformal rank rule; ties do not reject. Signal power,
//! held-out null false-positive rate and directionally qualified outcomes are reported separately.

use crate::estimator_response::FactorialEstimatorResponseReport;
use std::collections::HashSet;
use std::fmt;

pub const ESTIMATOR_IDS: [&str; 3] = [
    "absolute_weighted_jaccard",
    "signed_shift_weighted_jaccard",
    "normalized_mean_absolute_edge_error",
];

#[derive(Debug, Clone, PartialEq)]
pub struct NullCalibrationObservation {
    pub replicate_id: String,
    /// Pre-registered non-negative distance statistic, one per estimator in ESTIMATOR_IDS order.
    pub statistics: [f64; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct HeldOutEvaluationObservation {
    pub replicate_id: String,
    /// Nuisance-only statistic. Used only for held-out false-positive measurement.
    pub null_statistics: [f64; 3],
    /// Signal-plus-nuisance statistic. Compared with the frozen null threshold.
    pub signal_statistics: [f64; 3],
    /// Connectivity-space target-direction cosine, common to the three estimators.
    pub signal_directional_cosine: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProportionInterval {
    pub successes: usize,
    pub trials: usize,
    pub estimate: f64,
    pub wilson_95_lower: f64,
    pub wilson_95_upper: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EstimatorNullCalibrationSummary {
    pub estimator_id: &'static str,
    pub alpha: f64,
    /// None means the requested alpha is below the finite-sample calibration resolution.
    pub critical_value: Option<f64>,
    pub minimum_resolvable_alpha: f64,
    pub calibration_replicates: usize,
    pub evaluation_replicates: usize,
    pub held_out_false_positive_rate: ProportionInterval,
    pub signal_detection_rate: ProportionInterval,
    /// Signal statistic exceeds threshold and directional cosine >= minimum_direction_cosine.
    pub directionally_correct_signal_rate: ProportionInterval,
    /// Signal statistic exceeds threshold and directional cosine < minimum_direction_cosine.
    pub wrong_direction_acceptance_rate: ProportionInterval,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HeldOutNullCalibrationReport {
    pub schema_version: &'static str,
    pub calibration_split_id: String,
    pub evaluation_split_id: String,
    pub alpha: f64,
    pub minimum_direction_cosine: f64,
    pub calibration_replicates: usize,
    pub evaluation_replicates: usize,
    /// FNV-1a diagnostics over the ordered replicate identifiers; receipt supplies crypto digests.
    pub calibration_replicate_ids_fingerprint: String,
    pub evaluation_replicate_ids_fingerprint: String,
    pub estimators: Vec<EstimatorNullCalibrationSummary>,
}

impl HeldOutNullCalibrationReport {
    pub fn to_json_pretty(&self) -> String {
        let estimators = self
            .estimators
            .iter()
            .map(|item| {
                let threshold = item
                    .critical_value
                    .map(|value| format!("{value:.12}"))
                    .unwrap_or_else(|| "null".to_string());
                format!(
                    concat!(
                        "    {{\n",
                        "      \"estimator_id\": \"{}\",\n",
                        "      \"alpha\": {:.12},\n",
                        "      \"critical_value\": {},\n",
                        "      \"minimum_resolvable_alpha\": {:.12},\n",
                        "      \"calibration_replicates\": {},\n",
                        "      \"evaluation_replicates\": {},\n",
                        "      \"held_out_false_positive_rate\": {},\n",
                        "      \"signal_detection_rate\": {},\n",
                        "      \"directionally_correct_signal_rate\": {},\n",
                        "      \"wrong_direction_acceptance_rate\": {}\n",
                        "    }}"
                    ),
                    item.estimator_id,
                    item.alpha,
                    threshold,
                    item.minimum_resolvable_alpha,
                    item.calibration_replicates,
                    item.evaluation_replicates,
                    proportion_json(&item.held_out_false_positive_rate),
                    proportion_json(&item.signal_detection_rate),
                    proportion_json(&item.directionally_correct_signal_rate),
                    proportion_json(&item.wrong_direction_acceptance_rate),
                )
            })
            .collect::<Vec<_>>()
            .join(",\n");
        format!(
            concat!(
                "{{\n",
                "  \"schema_version\": \"{}\",\n",
                "  \"source_status\": \"synthetic_held_out_null_calibration\",\n",
                "  \"calibration_split_id\": \"{}\",\n",
                "  \"evaluation_split_id\": \"{}\",\n",
                "  \"alpha\": {:.12},\n",
                "  \"minimum_direction_cosine\": {:.12},\n",
                "  \"calibration_replicates\": {},\n",
                "  \"evaluation_replicates\": {},\n",
                "  \"calibration_replicate_ids_fingerprint\": \"{}\",\n",
                "  \"evaluation_replicate_ids_fingerprint\": \"{}\",\n",
                "  \"estimators\": [\n{}\n  ],\n",
                "  \"claim_limits\": [\n",
                "    \"calibration and evaluation replicate identifiers are disjoint\",\n",
                "    \"critical values use a conservative finite-sample upper-tail rank rule\",\n",
                "    \"Wilson intervals quantify binomial Monte Carlo uncertainty only\",\n",
                "    \"Monte Carlo replicate count does not replace independent participant or site count\",\n",
                "    \"synthetic held-out calibration only; no empirical or clinical qualification\"\n",
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
            self.calibration_replicate_ids_fingerprint,
            self.evaluation_replicate_ids_fingerprint,
            estimators,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeldOutCalibrationError {
    InvalidAlpha,
    InvalidDirectionThreshold,
    EmptyCalibrationSet,
    EmptyEvaluationSet,
    InvalidSplitIdentifier(&'static str),
    DuplicateReplicateId(&'static str),
    CalibrationEvaluationOverlap,
    InvalidStatistic { replicate_id: String },
    InvalidDirectionCosine { replicate_id: String },
    UndefinedSignalDirection { replicate_id: String },
}

impl fmt::Display for HeldOutCalibrationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAlpha => write!(f, "alpha must be finite and in (0, 0.5]"),
            Self::InvalidDirectionThreshold => {
                write!(f, "minimum direction cosine must be finite and in [-1, 1]")
            }
            Self::EmptyCalibrationSet => {
                write!(f, "at least one calibration null replicate is required")
            }
            Self::EmptyEvaluationSet => {
                write!(f, "at least one held-out evaluation replicate is required")
            }
            Self::InvalidSplitIdentifier(field) => {
                write!(f, "{field} must be a non-empty stable ASCII identifier")
            }
            Self::DuplicateReplicateId(split) => {
                write!(f, "replicate identifiers must be unique within {split}")
            }
            Self::CalibrationEvaluationOverlap => {
                write!(
                    f,
                    "calibration and evaluation replicate identifiers must be disjoint"
                )
            }
            Self::InvalidStatistic { replicate_id } => {
                write!(
                    f,
                    "replicate {replicate_id} contains a non-finite or negative statistic"
                )
            }
            Self::InvalidDirectionCosine { replicate_id } => {
                write!(
                    f,
                    "replicate {replicate_id} has a direction cosine outside [-1, 1]"
                )
            }
            Self::UndefinedSignalDirection { replicate_id } => {
                write!(
                    f,
                    "replicate {replicate_id} has no defined signal direction"
                )
            }
        }
    }
}

impl std::error::Error for HeldOutCalibrationError {}

/// Convert an RCAF nuisance-only estimator-response report into calibration statistics.
///
/// The three non-negative nuisance-only displacements are used in ESTIMATOR_IDS order. The
/// response must contain all three expected estimators; an incomplete report fails closed.
pub fn null_observation_from_estimator_response(
    replicate_id: &str,
    response: &FactorialEstimatorResponseReport,
) -> Result<NullCalibrationObservation, HeldOutCalibrationError> {
    if response.estimators.len() != ESTIMATOR_IDS.len()
        || response
            .estimators
            .iter()
            .zip(ESTIMATOR_IDS.iter())
            .any(|(item, expected)| item.estimator_id != *expected)
    {
        return Err(HeldOutCalibrationError::InvalidStatistic {
            replicate_id: replicate_id.to_string(),
        });
    }
    let mut statistics = [0.0; 3];
    for (index, item) in response.estimators.iter().enumerate() {
        statistics[index] = item.nuisance_only_distance;
    }
    Ok(NullCalibrationObservation {
        replicate_id: replicate_id.to_string(),
        statistics,
    })
}

/// Convert a signal-plus-nuisance RCAF response into a held-out evaluation observation.
///
/// Null statistics come from the matched nuisance-only cell. Signal statistics are the distance
/// between clean and signal-plus-nuisance connectivity. The connectivity-space directional cosine
/// must be defined; zero-signal reports belong in null-only controls, not this signal evaluation set.
pub fn evaluation_observation_from_estimator_response(
    replicate_id: &str,
    response: &FactorialEstimatorResponseReport,
) -> Result<HeldOutEvaluationObservation, HeldOutCalibrationError> {
    if response.estimators.len() != ESTIMATOR_IDS.len()
        || response
            .estimators
            .iter()
            .zip(ESTIMATOR_IDS.iter())
            .any(|(item, expected)| item.estimator_id != *expected)
    {
        return Err(HeldOutCalibrationError::InvalidStatistic {
            replicate_id: replicate_id.to_string(),
        });
    }
    let signal_directional_cosine = response
        .connectivity_direction
        .signal_directional_cosine
        .ok_or_else(|| HeldOutCalibrationError::UndefinedSignalDirection {
            replicate_id: replicate_id.to_string(),
        })?;
    let mut null_statistics = [0.0; 3];
    let mut signal_statistics = [0.0; 3];
    for (index, item) in response.estimators.iter().enumerate() {
        null_statistics[index] = item.nuisance_only_distance;
        signal_statistics[index] = 1.0 - item.clean_to_combined_similarity;
    }
    Ok(HeldOutEvaluationObservation {
        replicate_id: replicate_id.to_string(),
        null_statistics,
        signal_statistics,
        signal_directional_cosine,
    })
}

/// Calibrate three estimator-specific null thresholds, then evaluate on disjoint held-out data.
///
/// For n calibration null statistics and requested alpha, let k=floor(alpha*(n+1)).
/// If k=0, the sample cannot resolve the requested alpha and no threshold is returned. Otherwise
/// the critical value is the k-th largest calibration statistic. Evaluation rejects only when
/// score > critical value; equality is a non-rejection, making ties conservative.
///
/// This is a finite-sample conformal-style upper-tail rule under exchangeability. It is not
/// automatically valid under participant/site dependence, non-exchangeable acquisition shifts,
/// or data-dependent estimator/threshold selection.
pub fn calibrate_and_evaluate_held_out_null(
    calibration: &[NullCalibrationObservation],
    evaluation: &[HeldOutEvaluationObservation],
    calibration_split_id: &str,
    evaluation_split_id: &str,
    alpha: f64,
    minimum_direction_cosine: f64,
) -> Result<HeldOutNullCalibrationReport, HeldOutCalibrationError> {
    if !alpha.is_finite() || !(0.0..=0.5).contains(&alpha) || alpha == 0.0 {
        return Err(HeldOutCalibrationError::InvalidAlpha);
    }
    if !minimum_direction_cosine.is_finite() || !(-1.0..=1.0).contains(&minimum_direction_cosine) {
        return Err(HeldOutCalibrationError::InvalidDirectionThreshold);
    }
    validate_split_id(calibration_split_id, "calibration_split_id")?;
    validate_split_id(evaluation_split_id, "evaluation_split_id")?;
    if calibration.is_empty() {
        return Err(HeldOutCalibrationError::EmptyCalibrationSet);
    }
    if evaluation.is_empty() {
        return Err(HeldOutCalibrationError::EmptyEvaluationSet);
    }

    let mut calibration_ids = HashSet::with_capacity(calibration.len());
    for replicate in calibration {
        validate_id(&replicate.replicate_id, "calibration")?;
        if !calibration_ids.insert(replicate.replicate_id.as_str()) {
            return Err(HeldOutCalibrationError::DuplicateReplicateId("calibration"));
        }
        if replicate
            .statistics
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(HeldOutCalibrationError::InvalidStatistic {
                replicate_id: replicate.replicate_id.clone(),
            });
        }
    }

    let mut evaluation_ids = HashSet::with_capacity(evaluation.len());
    for replicate in evaluation {
        validate_id(&replicate.replicate_id, "evaluation")?;
        if !evaluation_ids.insert(replicate.replicate_id.as_str()) {
            return Err(HeldOutCalibrationError::DuplicateReplicateId("evaluation"));
        }
        if calibration_ids.contains(replicate.replicate_id.as_str()) {
            return Err(HeldOutCalibrationError::CalibrationEvaluationOverlap);
        }
        if replicate
            .null_statistics
            .iter()
            .chain(replicate.signal_statistics.iter())
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(HeldOutCalibrationError::InvalidStatistic {
                replicate_id: replicate.replicate_id.clone(),
            });
        }
        if !replicate.signal_directional_cosine.is_finite()
            || !(-1.0..=1.0).contains(&replicate.signal_directional_cosine)
        {
            return Err(HeldOutCalibrationError::InvalidDirectionCosine {
                replicate_id: replicate.replicate_id.clone(),
            });
        }
    }

    let minimum_resolvable_alpha = 1.0 / (calibration.len() as f64 + 1.0);
    let mut estimators = Vec::with_capacity(3);
    for estimator_index in 0..3 {
        let critical_value = conservative_upper_threshold(
            calibration
                .iter()
                .map(|replicate| replicate.statistics[estimator_index])
                .collect(),
            alpha,
        );
        let held_out_false_positives = evaluation
            .iter()
            .filter(|replicate| {
                is_rejected(replicate.null_statistics[estimator_index], critical_value)
            })
            .count();
        let signal_detections = evaluation
            .iter()
            .filter(|replicate| {
                is_rejected(replicate.signal_statistics[estimator_index], critical_value)
            })
            .count();
        let directionally_correct = evaluation
            .iter()
            .filter(|replicate| {
                is_rejected(replicate.signal_statistics[estimator_index], critical_value)
                    && replicate.signal_directional_cosine >= minimum_direction_cosine
            })
            .count();
        let wrong_direction = evaluation
            .iter()
            .filter(|replicate| {
                is_rejected(replicate.signal_statistics[estimator_index], critical_value)
                    && replicate.signal_directional_cosine < minimum_direction_cosine
            })
            .count();

        estimators.push(EstimatorNullCalibrationSummary {
            estimator_id: ESTIMATOR_IDS[estimator_index],
            alpha,
            critical_value,
            minimum_resolvable_alpha,
            calibration_replicates: calibration.len(),
            evaluation_replicates: evaluation.len(),
            held_out_false_positive_rate: wilson_interval(
                held_out_false_positives,
                evaluation.len(),
            ),
            signal_detection_rate: wilson_interval(signal_detections, evaluation.len()),
            directionally_correct_signal_rate: wilson_interval(
                directionally_correct,
                evaluation.len(),
            ),
            wrong_direction_acceptance_rate: wilson_interval(wrong_direction, evaluation.len()),
        });
    }

    Ok(HeldOutNullCalibrationReport {
        schema_version: "rcaf.held-out-null-calibration/0.1.0",
        calibration_split_id: calibration_split_id.to_string(),
        evaluation_split_id: evaluation_split_id.to_string(),
        alpha,
        minimum_direction_cosine,
        calibration_replicates: calibration.len(),
        evaluation_replicates: evaluation.len(),
        calibration_replicate_ids_fingerprint: identifier_fingerprint(
            calibration.iter().map(|item| item.replicate_id.as_str()),
        ),
        evaluation_replicate_ids_fingerprint: identifier_fingerprint(
            evaluation.iter().map(|item| item.replicate_id.as_str()),
        ),
        estimators,
    })
}

fn identifier_fingerprint<'a>(identifiers: impl Iterator<Item = &'a str>) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for identifier in identifiers {
        for byte in identifier.as_bytes().iter().copied().chain([0xff]) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    format!("{hash:016x}")
}

fn validate_split_id(value: &str, field: &'static str) -> Result<(), HeldOutCalibrationError> {
    validate_id(value, field)
}

fn validate_id(value: &str, split: &'static str) -> Result<(), HeldOutCalibrationError> {
    if value.trim().is_empty()
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-'))
    {
        return Err(HeldOutCalibrationError::InvalidSplitIdentifier(split));
    }
    Ok(())
}

fn conservative_upper_threshold(mut values: Vec<f64>, alpha: f64) -> Option<f64> {
    values.sort_by(f64::total_cmp);
    let allowed_exceedances = (alpha * (values.len() as f64 + 1.0)).floor() as usize;
    if allowed_exceedances == 0 {
        None
    } else {
        Some(values[values.len() - allowed_exceedances])
    }
}

fn is_rejected(statistic: f64, threshold: Option<f64>) -> bool {
    threshold.is_some_and(|value| statistic > value)
}

/// Two-sided 95% Wilson score interval for a binomial proportion.
fn wilson_interval(successes: usize, trials: usize) -> ProportionInterval {
    let n = trials as f64;
    let p = successes as f64 / n;
    let z = 1.959_963_984_540_054;
    let z2 = z * z;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half_width = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    ProportionInterval {
        successes,
        trials,
        estimate: p,
        wilson_95_lower: (center - half_width).clamp(0.0, 1.0),
        wilson_95_upper: (center + half_width).clamp(0.0, 1.0),
    }
}

fn proportion_json(value: &ProportionInterval) -> String {
    format!(
        concat!(
            "{{\"successes\": {}, \"trials\": {}, \"estimate\": {:.12}, ",
            "\"wilson_95_lower\": {:.12}, \"wilson_95_upper\": {:.12}}}"
        ),
        value.successes, value.trials, value.estimate, value.wilson_95_lower, value.wilson_95_upper,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration() -> Vec<NullCalibrationObservation> {
        (0..99)
            .map(|index| NullCalibrationObservation {
                replicate_id: format!("cal-{index:03}"),
                statistics: [
                    (index + 1) as f64 / 100.0,
                    (index + 1) as f64 / 120.0,
                    (index + 1) as f64 / 150.0,
                ],
            })
            .collect()
    }

    fn evaluation() -> Vec<HeldOutEvaluationObservation> {
        (0..100)
            .map(|index| {
                let null = (index + 1) as f64 / 100.0;
                let signal = 0.90 + index as f64 / 1000.0;
                HeldOutEvaluationObservation {
                    replicate_id: format!("eval-{index:03}"),
                    null_statistics: [null, null * 0.8, null * 0.6],
                    signal_statistics: [signal, signal * 0.95, signal * 0.9],
                    signal_directional_cosine: if index % 10 == 0 { -0.2 } else { 0.8 },
                }
            })
            .collect()
    }

    #[test]
    fn calibration_is_disjoint_and_reports_held_out_false_positive_and_power() {
        let report = calibrate_and_evaluate_held_out_null(
            &calibration(),
            &evaluation(),
            "split-cal-01",
            "split-eval-01",
            0.05,
            0.0,
        )
        .unwrap();
        assert_eq!(report.estimators.len(), 3);
        let expected_signal_detections = [49, 100, 100];
        let expected_directionally_correct = [45, 90, 90];
        let expected_wrong_direction = [4, 10, 10];
        for (index, estimator) in report.estimators.iter().enumerate() {
            assert_eq!(estimator.calibration_replicates, 99);
            assert_eq!(estimator.evaluation_replicates, 100);
            assert!(estimator.critical_value.is_some());
            assert_eq!(estimator.held_out_false_positive_rate.successes, 5);
            assert_eq!(
                estimator.signal_detection_rate.successes,
                expected_signal_detections[index]
            );
            assert_eq!(
                estimator.directionally_correct_signal_rate.successes,
                expected_directionally_correct[index]
            );
            assert_eq!(
                estimator.wrong_direction_acceptance_rate.successes,
                expected_wrong_direction[index]
            );
        }
    }

    #[test]
    fn calibration_threshold_is_conservative_and_ties_do_not_reject() {
        let report = calibrate_and_evaluate_held_out_null(
            &calibration(),
            &evaluation(),
            "cal",
            "eval",
            0.05,
            0.0,
        )
        .unwrap();
        let threshold = report.estimators[0].critical_value.unwrap();
        assert!((threshold - 0.95).abs() < 1e-12);
        let eval = vec![HeldOutEvaluationObservation {
            replicate_id: "tie".to_string(),
            null_statistics: [threshold; 3],
            signal_statistics: [threshold; 3],
            signal_directional_cosine: 1.0,
        }];
        let single =
            calibrate_and_evaluate_held_out_null(&calibration(), &eval, "cal", "one", 0.05, 0.0)
                .unwrap();
        assert_eq!(
            single.estimators[0].held_out_false_positive_rate.successes,
            0
        );
        assert_eq!(single.estimators[0].signal_detection_rate.successes, 0);
    }

    #[test]
    fn alpha_below_resolution_returns_no_threshold_and_no_rejections() {
        let small_calibration = calibration().into_iter().take(4).collect::<Vec<_>>();
        let evaluation = evaluation().into_iter().take(5).collect::<Vec<_>>();
        let report = calibrate_and_evaluate_held_out_null(
            &small_calibration,
            &evaluation,
            "cal-small",
            "eval-small",
            0.05,
            0.0,
        )
        .unwrap();
        assert!(report
            .estimators
            .iter()
            .all(|item| item.critical_value.is_none()));
        assert!(report.estimators.iter().all(|item| {
            item.held_out_false_positive_rate.successes == 0
                && item.signal_detection_rate.successes == 0
        }));
    }

    #[test]
    fn wrong_direction_acceptance_is_separate_from_detection() {
        let calibration = calibration();
        let mut evaluation = evaluation();
        for item in &mut evaluation {
            item.signal_statistics = [1.0; 3];
            item.signal_directional_cosine = -0.5;
        }
        let report = calibrate_and_evaluate_held_out_null(
            &calibration,
            &evaluation,
            "cal",
            "eval-wrong",
            0.05,
            0.0,
        )
        .unwrap();
        for estimator in &report.estimators {
            assert_eq!(estimator.signal_detection_rate.successes, 100);
            assert_eq!(estimator.directionally_correct_signal_rate.successes, 0);
            assert_eq!(estimator.wrong_direction_acceptance_rate.successes, 100);
        }
    }

    #[test]
    fn wilson_interval_is_valid_at_zero_and_one_success_extremes() {
        let zero = wilson_interval(0, 100);
        let one = wilson_interval(100, 100);
        assert_eq!(zero.estimate, 0.0);
        assert_eq!(one.estimate, 1.0);
        assert!(zero.wilson_95_lower == 0.0 && zero.wilson_95_upper > 0.0);
        assert!(one.wilson_95_upper == 1.0 && one.wilson_95_lower < 1.0);
    }

    #[test]
    fn duplicate_and_overlapping_replicate_ids_fail_closed() {
        let mut duplicate = calibration();
        duplicate[1].replicate_id = duplicate[0].replicate_id.clone();
        assert!(matches!(
            calibrate_and_evaluate_held_out_null(
                &duplicate,
                &evaluation(),
                "cal",
                "eval",
                0.05,
                0.0
            ),
            Err(HeldOutCalibrationError::DuplicateReplicateId("calibration"))
        ));

        let mut overlapping = evaluation();
        overlapping[0].replicate_id = calibration()[0].replicate_id.clone();
        assert!(matches!(
            calibrate_and_evaluate_held_out_null(
                &calibration(),
                &overlapping,
                "cal",
                "eval",
                0.05,
                0.0
            ),
            Err(HeldOutCalibrationError::CalibrationEvaluationOverlap)
        ));
    }

    fn estimator_response(direction_cosine: Option<f64>) -> FactorialEstimatorResponseReport {
        use crate::estimator_response::{ConnectivityDirectionReport, EstimatorResponseSummary};
        FactorialEstimatorResponseReport {
            schema_version: "rcaf.factorial-estimator-response/0.1.0",
            experiment_id: "adapter-fixture".into(),
            signal_id: "target".into(),
            nuisance_family_id: "transient".into(),
            nuisance_instance_id: "instance-01".into(),
            time_points: 64,
            roi_count: 6,
            connectivity_estimator: "pearson_sample_correlation_v1",
            estimators: ESTIMATOR_IDS
                .iter()
                .enumerate()
                .map(|(index, estimator_id)| EstimatorResponseSummary {
                    estimator_id,
                    clean_to_signal_similarity: 0.9,
                    clean_to_nuisance_similarity: 0.8,
                    signal_to_combined_similarity: 0.85,
                    clean_to_combined_similarity: 0.75,
                    clean_signal_distance: 0.1,
                    nuisance_only_distance: 0.2 + index as f64 * 0.01,
                    combined_signal_residual_distance: 0.15,
                    nuisance_to_signal_distance_ratio: Some(2.0),
                    signal_retention_fraction: Some(0.5),
                })
                .collect(),
            connectivity_direction: ConnectivityDirectionReport {
                signal_projection_gain: direction_cosine.map(|_| 0.8),
                signal_directional_cosine: direction_cosine,
                signal_change_residual_fraction: direction_cosine.map(|_| 0.4),
                clean_signal_change_norm: 0.2,
                combined_signal_change_norm: 0.18,
            },
        }
    }

    #[test]
    fn estimator_response_adapter_uses_nuisance_and_combined_distances() {
        let response = estimator_response(Some(0.9));
        let null = null_observation_from_estimator_response("cal-001", &response).unwrap();
        let evaluation =
            evaluation_observation_from_estimator_response("eval-001", &response).unwrap();
        assert_eq!(null.statistics, [0.2, 0.21, 0.22]);
        assert_eq!(evaluation.null_statistics, [0.2, 0.21, 0.22]);
        assert_eq!(evaluation.signal_statistics, [0.25, 0.25, 0.25]);
        assert_eq!(evaluation.signal_directional_cosine, 0.9);
    }

    #[test]
    fn estimator_response_adapter_rejects_undefined_signal_direction() {
        assert!(matches!(
            evaluation_observation_from_estimator_response(
                "zero-signal",
                &estimator_response(None)
            ),
            Err(HeldOutCalibrationError::UndefinedSignalDirection { .. })
        ));
    }

    #[test]
    fn estimator_response_adapter_rejects_missing_or_reordered_estimators() {
        let mut response = estimator_response(Some(0.5));
        response.estimators.swap(0, 1);
        assert!(matches!(
            null_observation_from_estimator_response("bad-order", &response),
            Err(HeldOutCalibrationError::InvalidStatistic { .. })
        ));
        response.estimators.swap(0, 1);
        response.estimators.pop();
        assert!(matches!(
            evaluation_observation_from_estimator_response("missing", &response),
            Err(HeldOutCalibrationError::InvalidStatistic { .. })
        ));
    }

    #[test]
    fn invalid_statistics_directions_and_thresholds_fail_closed() {
        let mut invalid_calibration = calibration();
        invalid_calibration[0].statistics[0] = f64::NAN;
        assert!(matches!(
            calibrate_and_evaluate_held_out_null(
                &invalid_calibration,
                &evaluation(),
                "cal",
                "eval",
                0.05,
                0.0
            ),
            Err(HeldOutCalibrationError::InvalidStatistic { .. })
        ));

        let mut invalid_evaluation = evaluation();
        invalid_evaluation[0].signal_directional_cosine = 1.5;
        assert!(matches!(
            calibrate_and_evaluate_held_out_null(
                &calibration(),
                &invalid_evaluation,
                "cal",
                "eval",
                0.05,
                0.0
            ),
            Err(HeldOutCalibrationError::InvalidDirectionCosine { .. })
        ));

        assert!(matches!(
            calibrate_and_evaluate_held_out_null(
                &calibration(),
                &evaluation(),
                "cal",
                "eval",
                0.0,
                0.0
            ),
            Err(HeldOutCalibrationError::InvalidAlpha)
        ));
    }

    #[test]
    fn report_json_preserves_null_threshold_and_three_estimator_order() {
        let report = calibrate_and_evaluate_held_out_null(
            &calibration().into_iter().take(4).collect::<Vec<_>>(),
            &evaluation(),
            "cal",
            "eval",
            0.05,
            0.0,
        )
        .unwrap();
        let json = report.to_json_pretty();
        assert!(json.contains("\"critical_value\": null"));
        assert!(json.contains("absolute_weighted_jaccard"));
        assert!(json.contains("wilson_95_lower"));
    }
}
