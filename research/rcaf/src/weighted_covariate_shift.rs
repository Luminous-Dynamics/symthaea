//! Importance-weighted upper-tail calibration diagnostics for continuous covariate shift.
//!
//! This module consumes externally supplied density-ratio weights. It does not fit a density-ratio
//! model. It computes the test-point-specific weighted conformal upper quantile, calibration ESS,
//! chi-square divergence proxy and test-point mass. Estimated ratios are explicitly diagnostic:
//! they do not inherit the exact finite-sample guarantee of known likelihood ratios.

use crate::held_out_null_calibration::{ProportionInterval, ESTIMATOR_IDS};
use std::collections::HashSet;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub struct WeightedNullCalibrationObservation {
    pub replicate_id: String,
    pub statistics: [f64; 3],
    /// Source-to-target covariate density ratio dQ_X/dP_X at this calibration covariate.
    pub importance_weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WeightedHeldOutEvaluationObservation {
    pub replicate_id: String,
    pub null_statistics: [f64; 3],
    pub signal_statistics: [f64; 3],
    pub signal_directional_cosine: f64,
    /// Target-to-source covariate density ratio at this particular test covariate.
    pub importance_weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RatioProvenance {
    /// Caller attests that weights are the known likelihood ratio dQ_X/dP_X.
    KnownDensityRatio,
    /// Ratio model is estimated on a separate, explicitly identified covariate split.
    EstimatedDensityRatio {
        estimator_id: String,
        estimation_split_id: String,
        cross_fitted: bool,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct WeightedCovariateShiftConfig {
    pub calibration_split_id: String,
    pub evaluation_split_id: String,
    pub ratio_provenance: RatioProvenance,
    pub alpha: f64,
    pub minimum_direction_cosine: f64,
    /// Operational abstention floor for the calibration-weight ESS.
    pub minimum_effective_sample_size: f64,
    /// Caller attestation of Q_X << P_X on the target support.
    pub source_support_attested: bool,
    /// Caller attestation that P(Y|X), hence the score conditional, is invariant.
    pub conditional_score_invariance_attested: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportanceWeightDiagnostics {
    pub calibration_weight_sum: f64,
    pub calibration_weight_square_sum: f64,
    pub effective_sample_size: f64,
    pub effective_sample_fraction: f64,
    /// Empirical normalized second-moment proxy: n*sum(w^2)/sum(w)^2 - 1.
    pub chi_square_divergence_proxy: f64,
    pub maximum_normalized_calibration_weight: f64,
    pub maximum_to_mean_weight_ratio: f64,
    pub minimum_test_weight_share: f64,
    pub maximum_test_weight_share: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WeightedEvaluationAbstention {
    pub replicate_id: String,
    pub reason: &'static str,
    pub test_weight_share: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WeightedEstimatorSummary {
    pub estimator_id: &'static str,
    pub finite_threshold_count: usize,
    pub abstained_evaluation_count: usize,
    pub minimum_finite_threshold: Option<f64>,
    pub maximum_finite_threshold: Option<f64>,
    pub held_out_false_positive_rate: Option<ProportionInterval>,
    pub signal_detection_rate: Option<ProportionInterval>,
    pub directionally_correct_signal_rate: Option<ProportionInterval>,
    pub wrong_direction_acceptance_rate: Option<ProportionInterval>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WeightedCovariateShiftReport {
    pub schema_version: &'static str,
    pub calibration_split_id: String,
    pub evaluation_split_id: String,
    pub ratio_provenance: &'static str,
    pub ratio_estimator_id: Option<String>,
    pub ratio_estimation_split_id: Option<String>,
    pub cross_fitted_ratio_estimator: Option<bool>,
    pub finite_sample_guarantee_applicable: bool,
    pub validity_status: &'static str,
    pub alpha: f64,
    pub minimum_direction_cosine: f64,
    pub minimum_effective_sample_size: f64,
    pub source_support_attested: bool,
    pub conditional_score_invariance_attested: bool,
    pub calibration_replicates: usize,
    pub evaluation_replicates: usize,
    pub calibration_replicate_ids_fingerprint: String,
    pub evaluation_replicate_ids_fingerprint: String,
    pub diagnostics: ImportanceWeightDiagnostics,
    pub estimators: Vec<WeightedEstimatorSummary>,
    pub abstentions: Vec<WeightedEvaluationAbstention>,
}

impl WeightedCovariateShiftReport {
    pub fn to_json_pretty(&self) -> String {
        let estimators = self
            .estimators
            .iter()
            .map(render_estimator)
            .collect::<Vec<_>>()
            .join(",\n");
        let abstentions = self
            .abstentions
            .iter()
            .map(|item| {
                format!(
                    "    {{\"replicate_id\": \"{}\", \"reason\": \"{}\", \"test_weight_share\": {:.12}}}",
                    item.replicate_id, item.reason, item.test_weight_share
                )
            })
            .collect::<Vec<_>>()
            .join(",\n");
        format!(
            concat!(
                "{{\n",
                "  \"schema_version\": \"{}\",\n",
                "  \"source_status\": \"synthetic_weighted_covariate_shift_diagnostics\",\n",
                "  \"calibration_split_id\": \"{}\",\n",
                "  \"evaluation_split_id\": \"{}\",\n",
                "  \"ratio_provenance\": \"{}\",\n",
                "  \"ratio_estimator_id\": {},\n",
                "  \"ratio_estimation_split_id\": {},\n",
                "  \"cross_fitted_ratio_estimator\": {},\n",
                "  \"finite_sample_guarantee_applicable\": {},\n",
                "  \"validity_status\": \"{}\",\n",
                "  \"alpha\": {:.12},\n",
                "  \"minimum_direction_cosine\": {:.12},\n",
                "  \"minimum_effective_sample_size\": {:.12},\n",
                "  \"source_support_attested\": {},\n",
                "  \"conditional_score_invariance_attested\": {},\n",
                "  \"calibration_replicates\": {},\n",
                "  \"evaluation_replicates\": {},\n",
                "  \"calibration_replicate_ids_fingerprint\": \"{}\",\n",
                "  \"evaluation_replicate_ids_fingerprint\": \"{}\",\n",
                "  \"diagnostics\": {{\n",
                "    \"calibration_weight_sum\": {:.12},\n",
                "    \"calibration_weight_square_sum\": {:.12},\n",
                "    \"effective_sample_size\": {:.12},\n",
                "    \"effective_sample_fraction\": {:.12},\n",
                "    \"chi_square_divergence_proxy\": {:.12},\n",
                "    \"maximum_normalized_calibration_weight\": {:.12},\n",
                "    \"maximum_to_mean_weight_ratio\": {:.12},\n",
                "    \"minimum_test_weight_share\": {:.12},\n",
                "    \"maximum_test_weight_share\": {:.12}\n",
                "  }},\n",
                "  \"estimators\": [\n{}\n  ],\n",
                "  \"abstentions\": [\n{}\n  ],\n",
                "  \"claim_limits\": [\n",
                "    \"weights are supplied externally; this module does not estimate a density ratio\",\n",
                "    \"exact finite-sample covariate-shift validity applies only to caller-attested known ratios and the stated assumptions\",\n",
                "    \"estimated ratios are diagnostic and do not receive an exact finite-sample guarantee\",\n",
                "    \"ESS and chi-square quantities are empirical diagnostics, not proof of support overlap\",\n",
                "    \"the covariate-shift assumption requires the score conditional law to remain invariant\"\n",
                "  ]\n",
                "}}\n"
            ),
            self.schema_version,
            self.calibration_split_id,
            self.evaluation_split_id,
            self.ratio_provenance,
            optional_string_json(self.ratio_estimator_id.as_deref()),
            optional_string_json(self.ratio_estimation_split_id.as_deref()),
            self.cross_fitted_ratio_estimator
                .map(|value| value.to_string())
                .unwrap_or_else(|| "null".to_string()),
            self.finite_sample_guarantee_applicable,
            self.validity_status,
            self.alpha,
            self.minimum_direction_cosine,
            self.minimum_effective_sample_size,
            self.source_support_attested,
            self.conditional_score_invariance_attested,
            self.calibration_replicates,
            self.evaluation_replicates,
            self.calibration_replicate_ids_fingerprint,
            self.evaluation_replicate_ids_fingerprint,
            self.diagnostics.calibration_weight_sum,
            self.diagnostics.calibration_weight_square_sum,
            self.diagnostics.effective_sample_size,
            self.diagnostics.effective_sample_fraction,
            self.diagnostics.chi_square_divergence_proxy,
            self.diagnostics.maximum_normalized_calibration_weight,
            self.diagnostics.maximum_to_mean_weight_ratio,
            self.diagnostics.minimum_test_weight_share,
            self.diagnostics.maximum_test_weight_share,
            estimators,
            abstentions,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightedCovariateShiftError {
    InvalidAlpha,
    InvalidDirectionThreshold,
    InvalidMinimumEffectiveSampleSize,
    EmptyCalibrationSet,
    EmptyEvaluationSet,
    InvalidIdentifier(&'static str),
    DuplicateReplicateId(&'static str),
    CalibrationEvaluationOverlap,
    RatioEstimationSplitOverlap,
    InvalidRatioEstimatorId,
    InvalidImportanceWeight { replicate_id: String },
    NonPositiveEvaluationWeight { replicate_id: String },
    InvalidStatistic { replicate_id: String },
    InvalidDirectionCosine { replicate_id: String },
    NonFiniteWeightMoments,
}

impl fmt::Display for WeightedCovariateShiftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAlpha => write!(f, "alpha must be finite and in (0, 0.5]"),
            Self::InvalidDirectionThreshold => {
                write!(f, "minimum direction cosine must be finite and in [-1, 1]")
            }
            Self::InvalidMinimumEffectiveSampleSize => {
                write!(f, "minimum effective sample size must be finite and positive")
            }
            Self::EmptyCalibrationSet => write!(f, "at least one weighted calibration replicate is required"),
            Self::EmptyEvaluationSet => write!(f, "at least one weighted evaluation replicate is required"),
            Self::InvalidIdentifier(field) => {
                write!(f, "{field} must be a non-empty stable ASCII identifier")
            }
            Self::DuplicateReplicateId(split) => {
                write!(f, "replicate identifiers must be unique within {split}")
            }
            Self::CalibrationEvaluationOverlap => {
                write!(f, "calibration and evaluation replicate identifiers must be disjoint")
            }
            Self::RatioEstimationSplitOverlap => {
                write!(f, "ratio-estimation split must be disjoint from calibration and evaluation splits")
            }
            Self::InvalidRatioEstimatorId => write!(f, "estimated density ratio requires a non-empty estimator id"),
            Self::InvalidImportanceWeight { replicate_id } => {
                write!(f, "replicate {replicate_id} has a non-finite or negative importance weight")
            }
            Self::NonPositiveEvaluationWeight { replicate_id } => {
                write!(f, "evaluation replicate {replicate_id} must have a strictly positive target density-ratio weight")
            }
            Self::InvalidStatistic { replicate_id } => {
                write!(f, "replicate {replicate_id} contains a non-finite or negative statistic")
            }
            Self::InvalidDirectionCosine { replicate_id } => {
                write!(f, "replicate {replicate_id} has a direction cosine outside [-1, 1]")
            }
            Self::NonFiniteWeightMoments => write!(f, "importance-weight moments are non-finite or degenerate"),
        }
    }
}

impl std::error::Error for WeightedCovariateShiftError {}

/// Compute test-point-specific weighted upper-tail thresholds and evaluate held-out outcomes.
///
/// The weighted empirical CDF places the test point's importance weight at +infinity.
/// For calibration weights w_i and test weight w*, the target mass is
/// (1-alpha) * (sum_i w_i + w*). If this exceeds the calibration weight mass, the
/// threshold is infinite and the evaluation point abstains.
///
/// This is the known-ratio weighted-conformal construction under covariate shift. Estimated
/// ratios produce a diagnostic threshold only; no exact finite-sample guarantee is asserted.
pub fn evaluate_weighted_covariate_shift(
    calibration: &[WeightedNullCalibrationObservation],
    evaluation: &[WeightedHeldOutEvaluationObservation],
    config: &WeightedCovariateShiftConfig,
) -> Result<WeightedCovariateShiftReport, WeightedCovariateShiftError> {
    validate_config(config)?;
    if calibration.is_empty() {
        return Err(WeightedCovariateShiftError::EmptyCalibrationSet);
    }
    if evaluation.is_empty() {
        return Err(WeightedCovariateShiftError::EmptyEvaluationSet);
    }

    let mut calibration_ids = HashSet::with_capacity(calibration.len());
    for item in calibration {
        validate_id(&item.replicate_id, "calibration_split_id")?;
        if !calibration_ids.insert(item.replicate_id.as_str()) {
            return Err(WeightedCovariateShiftError::DuplicateReplicateId("calibration"));
        }
        validate_statistics(&item.replicate_id, &item.statistics)?;
        validate_weight(&item.replicate_id, item.importance_weight)?;
    }

    let mut evaluation_ids = HashSet::with_capacity(evaluation.len());
    for item in evaluation {
        validate_id(&item.replicate_id, "evaluation_split_id")?;
        if !evaluation_ids.insert(item.replicate_id.as_str()) {
            return Err(WeightedCovariateShiftError::DuplicateReplicateId("evaluation"));
        }
        if calibration_ids.contains(item.replicate_id.as_str()) {
            return Err(WeightedCovariateShiftError::CalibrationEvaluationOverlap);
        }
        validate_statistics(&item.replicate_id, &item.null_statistics)?;
        validate_statistics(&item.replicate_id, &item.signal_statistics)?;
        validate_weight(&item.replicate_id, item.importance_weight)?;
        if item.importance_weight <= 0.0 {
            return Err(WeightedCovariateShiftError::NonPositiveEvaluationWeight {
                replicate_id: item.replicate_id.clone(),
            });
        }
        if !item.signal_directional_cosine.is_finite()
            || !(-1.0..=1.0).contains(&item.signal_directional_cosine)
        {
            return Err(WeightedCovariateShiftError::InvalidDirectionCosine {
                replicate_id: item.replicate_id.clone(),
            });
        }
    }

    let weights = calibration
        .iter()
        .map(|item| item.importance_weight)
        .collect::<Vec<_>>();
    let diagnostics = calculate_weight_diagnostics(&weights, evaluation)?;
    let (ratio_provenance, ratio_estimator_id, ratio_estimation_split_id, cross_fitted, exact_ratio) =
        match &config.ratio_provenance {
            RatioProvenance::KnownDensityRatio => (
                "known_density_ratio",
                None,
                None,
                None,
                true,
            ),
            RatioProvenance::EstimatedDensityRatio {
                estimator_id,
                estimation_split_id,
                cross_fitted,
            } => (
                "estimated_density_ratio",
                Some(estimator_id.clone()),
                Some(estimation_split_id.clone()),
                Some(*cross_fitted),
                false,
            ),
        };
    let finite_sample_guarantee_applicable =
        exact_ratio && config.source_support_attested && config.conditional_score_invariance_attested;
    let validity_status = if finite_sample_guarantee_applicable {
        "known_ratio_under_attested_covariate_shift"
    } else if exact_ratio {
        "abstain_assumptions_unverified"
    } else {
        "estimated_ratio_diagnostic_only"
    };

    let mut summaries = Vec::with_capacity(ESTIMATOR_IDS.len());
    let mut abstentions = Vec::new();
    for (estimator_index, estimator_id) in ESTIMATOR_IDS.iter().enumerate() {
        let mut thresholds = Vec::new();
        let mut null_rejections = 0usize;
        let mut signal_detections = 0usize;
        let mut directionally_correct = 0usize;
        let mut wrong_direction = 0usize;

        for item in evaluation {
            let test_weight_share = item.importance_weight
                / (diagnostics.calibration_weight_sum + item.importance_weight);
            let abstention_reason = if !config.source_support_attested {
                Some("source_support_unverified")
            } else if !config.conditional_score_invariance_attested {
                Some("conditional_score_invariance_unverified")
            } else if diagnostics.effective_sample_size < config.minimum_effective_sample_size {
                Some("effective_sample_size_below_floor")
            } else {
                None
            };
            if let Some(reason) = abstention_reason {
                abstentions.push(WeightedEvaluationAbstention {
                    replicate_id: item.replicate_id.clone(),
                    reason,
                    test_weight_share,
                });
                continue;
            }

            let scores = calibration
                .iter()
                .map(|observation| {
                    (
                        observation.statistics[estimator_index],
                        observation.importance_weight,
                    )
                })
                .collect::<Vec<_>>();
            let critical_value = weighted_upper_threshold(
                &scores,
                item.importance_weight,
                config.alpha,
            );
            let Some(critical_value) = critical_value else {
                abstentions.push(WeightedEvaluationAbstention {
                    replicate_id: item.replicate_id.clone(),
                    reason: "test_point_mass_exceeds_alpha",
                    test_weight_share,
                });
                continue;
            };
            thresholds.push(critical_value);
            if item.null_statistics[estimator_index] > critical_value {
                null_rejections += 1;
            }
            if item.signal_statistics[estimator_index] > critical_value {
                signal_detections += 1;
                if item.signal_directional_cosine >= config.minimum_direction_cosine {
                    directionally_correct += 1;
                } else {
                    wrong_direction += 1;
                }
            }
        }

        let finite_threshold_count = thresholds.len();
        let abstained_evaluation_count = evaluation.len() - finite_threshold_count;
        summaries.push(WeightedEstimatorSummary {
            estimator_id,
            finite_threshold_count,
            abstained_evaluation_count,
            minimum_finite_threshold: thresholds.iter().copied().min_by(f64::total_cmp),
            maximum_finite_threshold: thresholds.iter().copied().max_by(f64::total_cmp),
            held_out_false_positive_rate: (finite_threshold_count > 0)
                .then(|| wilson_interval(null_rejections, finite_threshold_count)),
            signal_detection_rate: (finite_threshold_count > 0)
                .then(|| wilson_interval(signal_detections, finite_threshold_count)),
            directionally_correct_signal_rate: (finite_threshold_count > 0)
                .then(|| wilson_interval(directionally_correct, finite_threshold_count)),
            wrong_direction_acceptance_rate: (finite_threshold_count > 0)
                .then(|| wilson_interval(wrong_direction, finite_threshold_count)),
        });
    }

    Ok(WeightedCovariateShiftReport {
        schema_version: "rcaf.weighted-covariate-shift/0.1.0",
        calibration_split_id: config.calibration_split_id.clone(),
        evaluation_split_id: config.evaluation_split_id.clone(),
        ratio_provenance,
        ratio_estimator_id,
        ratio_estimation_split_id,
        cross_fitted_ratio_estimator: cross_fitted,
        finite_sample_guarantee_applicable,
        validity_status,
        alpha: config.alpha,
        minimum_direction_cosine: config.minimum_direction_cosine,
        minimum_effective_sample_size: config.minimum_effective_sample_size,
        source_support_attested: config.source_support_attested,
        conditional_score_invariance_attested: config.conditional_score_invariance_attested,
        calibration_replicates: calibration.len(),
        evaluation_replicates: evaluation.len(),
        calibration_replicate_ids_fingerprint: identifier_fingerprint(
            calibration.iter().map(|item| item.replicate_id.as_str()),
        ),
        evaluation_replicate_ids_fingerprint: identifier_fingerprint(
            evaluation.iter().map(|item| item.replicate_id.as_str()),
        ),
        diagnostics,
        estimators: summaries,
        abstentions,
    })
}

fn validate_config(config: &WeightedCovariateShiftConfig) -> Result<(), WeightedCovariateShiftError> {
    validate_id(&config.calibration_split_id, "calibration_split_id")?;
    validate_id(&config.evaluation_split_id, "evaluation_split_id")?;
    if config.calibration_split_id == config.evaluation_split_id {
        return Err(WeightedCovariateShiftError::CalibrationEvaluationOverlap);
    }
    if !config.alpha.is_finite() || !(0.0..=0.5).contains(&config.alpha) || config.alpha == 0.0 {
        return Err(WeightedCovariateShiftError::InvalidAlpha);
    }
    if !config.minimum_direction_cosine.is_finite()
        || !(-1.0..=1.0).contains(&config.minimum_direction_cosine)
    {
        return Err(WeightedCovariateShiftError::InvalidDirectionThreshold);
    }
    if !config.minimum_effective_sample_size.is_finite()
        || config.minimum_effective_sample_size <= 0.0
    {
        return Err(WeightedCovariateShiftError::InvalidMinimumEffectiveSampleSize);
    }
    if let RatioProvenance::EstimatedDensityRatio {
        estimator_id,
        estimation_split_id,
        ..
    } = &config.ratio_provenance
    {
        validate_id(estimator_id, "ratio_estimator_id")?;
        validate_id(estimation_split_id, "ratio_estimation_split_id")?;
        if estimation_split_id == &config.calibration_split_id
            || estimation_split_id == &config.evaluation_split_id
        {
            return Err(WeightedCovariateShiftError::RatioEstimationSplitOverlap);
        }
    }
    Ok(())
}

fn validate_statistics(
    replicate_id: &str,
    values: &[f64; 3],
) -> Result<(), WeightedCovariateShiftError> {
    if values.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(WeightedCovariateShiftError::InvalidStatistic {
            replicate_id: replicate_id.to_string(),
        });
    }
    Ok(())
}

fn validate_weight(
    replicate_id: &str,
    weight: f64,
) -> Result<(), WeightedCovariateShiftError> {
    if !weight.is_finite() || weight < 0.0 {
        return Err(WeightedCovariateShiftError::InvalidImportanceWeight {
            replicate_id: replicate_id.to_string(),
        });
    }
    Ok(())
}

fn calculate_weight_diagnostics(
    calibration_weights: &[f64],
    evaluation: &[WeightedHeldOutEvaluationObservation],
) -> Result<ImportanceWeightDiagnostics, WeightedCovariateShiftError> {
    let weight_sum = calibration_weights.iter().sum::<f64>();
    let squared_sum = calibration_weights
        .iter()
        .map(|weight| weight * weight)
        .sum::<f64>();
    if !weight_sum.is_finite()
        || !squared_sum.is_finite()
        || weight_sum <= 0.0
        || squared_sum <= 0.0
    {
        return Err(WeightedCovariateShiftError::NonFiniteWeightMoments);
    }
    let effective_sample_size = weight_sum * weight_sum / squared_sum;
    let sample_count = calibration_weights.len() as f64;
    let maximum_weight = calibration_weights.iter().copied().fold(0.0, f64::max);
    let maximum_normalized_weight = maximum_weight / weight_sum;
    let maximum_to_mean_weight_ratio = maximum_weight / (weight_sum / sample_count);
    let chi_square_divergence_proxy =
        (sample_count * squared_sum / (weight_sum * weight_sum) - 1.0).max(0.0);
    let shares = evaluation
        .iter()
        .map(|item| {
            item.importance_weight / (weight_sum + item.importance_weight)
        })
        .collect::<Vec<_>>();
    if [
        effective_sample_size,
        maximum_normalized_weight,
        maximum_to_mean_weight_ratio,
        chi_square_divergence_proxy,
    ]
    .iter()
    .any(|value| !value.is_finite())
        || shares.iter().any(|value| !value.is_finite())
    {
        return Err(WeightedCovariateShiftError::NonFiniteWeightMoments);
    }
    Ok(ImportanceWeightDiagnostics {
        calibration_weight_sum: weight_sum,
        calibration_weight_square_sum: squared_sum,
        effective_sample_size,
        effective_sample_fraction: effective_sample_size / sample_count,
        chi_square_divergence_proxy,
        maximum_normalized_calibration_weight: maximum_normalized_weight,
        maximum_to_mean_weight_ratio,
        minimum_test_weight_share: shares.iter().copied().fold(1.0, f64::min),
        maximum_test_weight_share: shares.iter().copied().fold(0.0, f64::max),
    })
}

/// Upper quantile of calibration scores with the test-point mass placed at +infinity.
///
/// If (1-alpha)*(sum(w_cal)+w_test) exceeds sum(w_cal), no finite threshold exists.
pub fn weighted_upper_threshold(
    calibration_scores_and_weights: &[(f64, f64)],
    test_weight: f64,
    alpha: f64,
) -> Option<f64> {
    if calibration_scores_and_weights.is_empty()
        || !test_weight.is_finite()
        || test_weight <= 0.0
        || !alpha.is_finite()
        || !(0.0..1.0).contains(&alpha)
        || calibration_scores_and_weights.iter().any(|(score, weight)| {
            !score.is_finite() || !weight.is_finite() || *weight < 0.0
        })
    {
        return None;
    }
    let calibration_weight_sum = calibration_scores_and_weights
        .iter()
        .map(|(_, weight)| *weight)
        .sum::<f64>();
    if !calibration_weight_sum.is_finite() || calibration_weight_sum <= 0.0 {
        return None;
    }
    let target_mass = (1.0 - alpha) * (calibration_weight_sum + test_weight);
    if !target_mass.is_finite() || target_mass > calibration_weight_sum {
        return None;
    }

    let mut ordered = calibration_scores_and_weights.to_vec();
    ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut cumulative_weight = 0.0;
    for (score, weight) in ordered {
        cumulative_weight += weight;
        if cumulative_weight >= target_mass {
            return Some(score);
        }
    }
    None
}

fn wilson_interval(successes: usize, trials: usize) -> ProportionInterval {
    let n = trials as f64;
    let p = successes as f64 / n;
    let z = 1.959_963_984_540_054;
    let z2 = z * z;
    let (lower, upper) = if successes == 0 {
        (0.0, z2 / (n + z2))
    } else if successes == trials {
        (n / (n + z2), 1.0)
    } else {
        let denominator = 1.0 + z2 / n;
        let center = (p + z2 / (2.0 * n)) / denominator;
        let half_width = z * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt() / denominator;
        (
            (center - half_width).clamp(0.0, 1.0),
            (center + half_width).clamp(0.0, 1.0),
        )
    };
    ProportionInterval {
        successes,
        trials,
        estimate: p,
        wilson_95_lower: lower,
        wilson_95_upper: upper,
    }
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

fn optional_string_json(value: Option<&str>) -> String {
    value
        .map(|value| format!("\"{value}\""))
        .unwrap_or_else(|| "null".to_string())
}

fn optional_number_json(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.12}"))
        .unwrap_or_else(|| "null".to_string())
}

fn optional_interval_json(value: Option<&ProportionInterval>) -> String {
    value
        .map(|interval| {
            format!(
                concat!(
                    "{{\"successes\": {}, \"trials\": {}, \"estimate\": {:.12}, ",
                    "\"wilson_95_lower\": {:.12}, \"wilson_95_upper\": {:.12}}}"
                ),
                interval.successes,
                interval.trials,
                interval.estimate,
                interval.wilson_95_lower,
                interval.wilson_95_upper
            )
        })
        .unwrap_or_else(|| "null".to_string())
}

fn render_estimator(item: &WeightedEstimatorSummary) -> String {
    format!(
        concat!(
            "    {{\n",
            "      \"estimator_id\": \"{}\",\n",
            "      \"finite_threshold_count\": {},\n",
            "      \"abstained_evaluation_count\": {},\n",
            "      \"minimum_finite_threshold\": {},\n",
            "      \"maximum_finite_threshold\": {},\n",
            "      \"held_out_false_positive_rate\": {},\n",
            "      \"signal_detection_rate\": {},\n",
            "      \"directionally_correct_signal_rate\": {},\n",
            "      \"wrong_direction_acceptance_rate\": {}\n",
            "    }}"
        ),
        item.estimator_id,
        item.finite_threshold_count,
        item.abstained_evaluation_count,
        optional_number_json(item.minimum_finite_threshold),
        optional_number_json(item.maximum_finite_threshold),
        optional_interval_json(item.held_out_false_positive_rate.as_ref()),
        optional_interval_json(item.signal_detection_rate.as_ref()),
        optional_interval_json(item.directionally_correct_signal_rate.as_ref()),
        optional_interval_json(item.wrong_direction_acceptance_rate.as_ref()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calibration(scores: &[f64], weights: &[f64]) -> Vec<WeightedNullCalibrationObservation> {
        scores
            .iter()
            .zip(weights.iter())
            .enumerate()
            .map(|(index, (score, weight))| WeightedNullCalibrationObservation {
                replicate_id: format!("cal-{index:03}"),
                statistics: [*score, *score * 0.8, *score * 0.6],
                importance_weight: *weight,
            })
            .collect()
    }

    fn evaluation(weights: &[f64]) -> Vec<WeightedHeldOutEvaluationObservation> {
        weights
            .iter()
            .enumerate()
            .map(|(index, weight)| WeightedHeldOutEvaluationObservation {
                replicate_id: format!("eval-{index:03}"),
                null_statistics: [0.92, 0.92 * 0.8, 0.92 * 0.6],
                signal_statistics: [0.99, 0.99 * 0.8, 0.99 * 0.6],
                signal_directional_cosine: if index % 5 == 0 { -0.2 } else { 0.8 },
                importance_weight: *weight,
            })
            .collect()
    }

    fn known_config() -> WeightedCovariateShiftConfig {
        WeightedCovariateShiftConfig {
            calibration_split_id: "cal-source-v1".to_string(),
            evaluation_split_id: "eval-target-v1".to_string(),
            ratio_provenance: RatioProvenance::KnownDensityRatio,
            alpha: 0.05,
            minimum_direction_cosine: 0.0,
            minimum_effective_sample_size: 20.0,
            source_support_attested: true,
            conditional_score_invariance_attested: true,
        }
    }

    #[test]
    fn equal_weights_recover_the_unweighted_finite_sample_threshold() {
        let scores = (1..=99).map(|value| value as f64 / 100.0).collect::<Vec<_>>();
        let weights = vec![1.0; 99];
        let report = evaluate_weighted_covariate_shift(
            &calibration(&scores, &weights),
            &evaluation(&[1.0]),
            &known_config(),
        )
        .unwrap();
        assert!(report.finite_sample_guarantee_applicable);
        assert_eq!(report.diagnostics.effective_sample_size, 99.0);
        assert!(report.diagnostics.chi_square_divergence_proxy.abs() < 1e-12);
        for estimator in &report.estimators {
            assert_eq!(estimator.finite_threshold_count, 1);
            assert!((estimator.minimum_finite_threshold.unwrap() - 0.95).abs() < 1e-12);
        }
    }

    #[test]
    fn high_test_weight_places_too_much_mass_at_infinity_and_abstains() {
        let scores = (1..=99).map(|value| value as f64 / 100.0).collect::<Vec<_>>();
        let report = evaluate_weighted_covariate_shift(
            &calibration(&scores, &vec![1.0; 99]),
            &evaluation(&[20.0]),
            &known_config(),
        )
        .unwrap();
        assert_eq!(report.abstentions.len(), 1);
        assert_eq!(
            report.abstentions[0].reason,
            "test_point_mass_exceeds_alpha"
        );
        assert!(report.estimators.iter().all(|item| item.finite_threshold_count == 0));
    }

    #[test]
    fn estimated_density_ratios_are_explicitly_not_exact_finite_sample_guarantees() {
        let scores = (1..=99).map(|value| value as f64 / 100.0).collect::<Vec<_>>();
        let mut config = known_config();
        config.ratio_provenance = RatioProvenance::EstimatedDensityRatio {
            estimator_id: "logistic-ratio-v1".to_string(),
            estimation_split_id: "ratio-fit-v1".to_string(),
            cross_fitted: true,
        };
        let report = evaluate_weighted_covariate_shift(
            &calibration(&scores, &vec![1.0; 99]),
            &evaluation(&[1.0]),
            &config,
        )
        .unwrap();
        assert!(!report.finite_sample_guarantee_applicable);
        assert_eq!(report.validity_status, "estimated_ratio_diagnostic_only");
        assert_eq!(report.ratio_estimation_split_id.as_deref(), Some("ratio-fit-v1"));
    }

    #[test]
    fn weight_diagnostics_expose_concentration_and_ess_collapse() {
        let scores = (1..=20).map(|value| value as f64).collect::<Vec<_>>();
        let mut weights = vec![1.0; 19];
        weights.push(100.0);
        let report = evaluate_weighted_covariate_shift(
            &calibration(&scores, &weights),
            &evaluation(&[1.0]),
            &WeightedCovariateShiftConfig {
                minimum_effective_sample_size: 1.0,
                ..known_config()
            },
        )
        .unwrap();
        assert!(report.diagnostics.maximum_normalized_calibration_weight > 0.8);
        assert!(report.diagnostics.effective_sample_size < 2.0);
        assert!(report.diagnostics.chi_square_divergence_proxy > 0.0);
    }

    #[test]
    fn low_effective_sample_size_abstains_without_fabricating_a_threshold() {
        let scores = (1..=20).map(|value| value as f64).collect::<Vec<_>>();
        let mut weights = vec![1.0; 19];
        weights.push(100.0);
        let report = evaluate_weighted_covariate_shift(
            &calibration(&scores, &weights),
            &evaluation(&[1.0]),
            &known_config(),
        )
        .unwrap();
        assert_eq!(report.abstentions.len(), 1);
        assert_eq!(
            report.abstentions[0].reason,
            "effective_sample_size_below_floor"
        );
    }

    #[test]
    fn support_and_conditional_invariance_must_be_attested() {
        let scores = (1..=99).map(|value| value as f64 / 100.0).collect::<Vec<_>>();
        for config in [
            WeightedCovariateShiftConfig {
                source_support_attested: false,
                ..known_config()
            },
            WeightedCovariateShiftConfig {
                conditional_score_invariance_attested: false,
                ..known_config()
            },
        ] {
            let report = evaluate_weighted_covariate_shift(
                &calibration(&scores, &vec![1.0; 99]),
                &evaluation(&[1.0]),
                &config,
            )
            .unwrap();
            assert!(!report.finite_sample_guarantee_applicable);
            assert_eq!(report.estimators[0].finite_threshold_count, 0);
            assert_eq!(report.abstentions.len(), 1);
        }
    }

    #[test]
    fn distinct_test_weights_produce_test_point_specific_thresholds() {
        let scores = (1..=99).map(|value| value as f64 / 100.0).collect::<Vec<_>>();
        let report = evaluate_weighted_covariate_shift(
            &calibration(&scores, &vec![1.0; 99]),
            &evaluation(&[1.0, 2.0]),
            &known_config(),
        )
        .unwrap();
        let estimator = &report.estimators[0];
        assert_eq!(estimator.finite_threshold_count, 2);
        assert!(
            estimator.minimum_finite_threshold.unwrap()
                < estimator.maximum_finite_threshold.unwrap()
        );
        assert!(report.estimators.iter().all(|item| item.finite_threshold_count == 2));
    }

    #[test]
    fn weighted_upper_tail_quantile_uses_test_mass_at_infinity() {
        let scores = (1..=9).map(|value| (value as f64, 1.0)).collect::<Vec<_>>();
        assert_eq!(weighted_upper_threshold(&scores, 1.0, 0.1), Some(9.0));
        assert_eq!(weighted_upper_threshold(&scores, 20.0, 0.1), None);
    }

    #[test]
    fn calibration_and_ratio_estimation_splits_must_be_disjoint() {
        let mut config = known_config();
        config.ratio_provenance = RatioProvenance::EstimatedDensityRatio {
            estimator_id: "ratio-model-v1".to_string(),
            estimation_split_id: config.calibration_split_id.clone(),
            cross_fitted: false,
        };
        let scores = (1..=99).map(|value| value as f64 / 100.0).collect::<Vec<_>>();
        assert!(matches!(
            evaluate_weighted_covariate_shift(
                &calibration(&scores, &vec![1.0; 99]),
                &evaluation(&[1.0]),
                &config
            ),
            Err(WeightedCovariateShiftError::RatioEstimationSplitOverlap)
        ));
    }

    #[test]
    fn invalid_weights_statistics_and_ids_fail_closed() {
        let scores = (1..=99).map(|value| value as f64 / 100.0).collect::<Vec<_>>();
        let mut bad_weight = calibration(&scores, &vec![1.0; 99]);
        bad_weight[0].importance_weight = f64::NAN;
        assert!(matches!(
            evaluate_weighted_covariate_shift(&bad_weight, &evaluation(&[1.0]), &known_config()),
            Err(WeightedCovariateShiftError::InvalidImportanceWeight { .. })
        ));
        let mut bad_eval = evaluation(&[1.0]);
        bad_eval[0].signal_statistics[0] = f64::INFINITY;
        assert!(matches!(
            evaluate_weighted_covariate_shift(
                &calibration(&scores, &vec![1.0; 99]),
                &bad_eval,
                &known_config()
            ),
            Err(WeightedCovariateShiftError::InvalidStatistic { .. })
        ));
    }

    #[test]
    fn zero_weight_evaluation_point_is_rejected_as_outside_target_support() {
        let scores = (1..=99).map(|value| value as f64 / 100.0).collect::<Vec<_>>();
        assert!(matches!(
            evaluate_weighted_covariate_shift(
                &calibration(&scores, &vec![1.0; 99]),
                &evaluation(&[0.0]),
                &known_config(),
            ),
            Err(WeightedCovariateShiftError::NonPositiveEvaluationWeight { .. })
        ));
    }

    #[test]
    fn zero_weight_source_rows_are_allowed_but_all_zero_weights_fail() {
        let scores = (1..=99).map(|value| value as f64 / 100.0).collect::<Vec<_>>();
        let mut weights = vec![1.0; 99];
        weights[0] = 0.0;
        let report = evaluate_weighted_covariate_shift(
            &calibration(&scores, &weights),
            &evaluation(&[1.0]),
            &known_config(),
        )
        .unwrap();
        assert!(report.diagnostics.effective_sample_size < 99.0);

        assert!(matches!(
            evaluate_weighted_covariate_shift(
                &calibration(&scores, &vec![0.0; 99]),
                &evaluation(&[1.0]),
                &known_config()
            ),
            Err(WeightedCovariateShiftError::NonFiniteWeightMoments)
        ));
    }
}
