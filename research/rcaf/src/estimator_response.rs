//! Estimator-level response of the matched factorial counterfactual cells.
//!
//! Raw time x ROI observations are converted into Pearson ROI-connectivity matrices first.
//! The same three RCAF matrix-similarity estimators are then applied to clean, signal-only,
//! nuisance-only, and signal-plus-nuisance connectivity matrices. This exposes nonlinear
//! signal × nuisance interactions induced by correlation estimation.

use crate::factorial_counterfactual::FactorialCounterfactualExperiment;
use crate::simulation::{correlation_similarity, SimilarityEstimator};
use crate::MetricError;
use std::fmt;

/// Connectivity-space change in the signal intervention before and after nuisance.
#[derive(Debug, Clone, PartialEq)]
pub struct ConnectivityDirectionReport {
    /// Least-squares projection gain of the combined signal change onto the clean signal change.
    pub signal_projection_gain: Option<f64>,
    /// Cosine alignment between the two upper-triangle connectivity-change vectors.
    pub signal_directional_cosine: Option<f64>,
    /// ||Delta_combined - Delta_signal|| / ||Delta_signal||.
    pub signal_change_residual_fraction: Option<f64>,
    /// Frobenius norm of the clean signal-induced upper-triangle change.
    pub clean_signal_change_norm: f64,
    /// Frobenius norm of the signal-induced change in the nuisance background.
    pub combined_signal_change_norm: f64,
}

/// Per-estimator similarities and normalized displacements for one factorial experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct EstimatorResponseSummary {
    pub estimator_id: &'static str,
    /// Similarity between the clean connectivity and signal-only connectivity.
    pub clean_to_signal_similarity: f64,
    /// Similarity between clean and nuisance-only connectivity.
    pub clean_to_nuisance_similarity: f64,
    /// Similarity between signal-only and signal-plus-nuisance connectivity.
    pub signal_to_combined_similarity: f64,
    /// Similarity between clean and signal-plus-nuisance connectivity.
    pub clean_to_combined_similarity: f64,
    /// Signal effect distance: 1 - clean_to_signal_similarity.
    pub clean_signal_distance: f64,
    /// Nuisance-only distance: 1 - clean_to_nuisance_similarity.
    pub nuisance_only_distance: f64,
    /// Residual combined distance relative to the signal-only connectivity.
    pub combined_signal_residual_distance: f64,
    /// nuisance_only_distance / clean_signal_distance; undefined for zero signal.
    pub nuisance_to_signal_distance_ratio: Option<f64>,
    /// 1 - combined_signal_residual_distance / clean_signal_distance.
    /// This may be negative or exceed one; values are not clipped.
    pub signal_retention_fraction: Option<f64>,
}

/// Full estimator response for a single four-cell raw observation experiment.
#[derive(Debug, Clone, PartialEq)]
pub struct FactorialEstimatorResponseReport {
    pub schema_version: &'static str,
    pub experiment_id: String,
    pub signal_id: String,
    pub nuisance_family_id: String,
    pub nuisance_instance_id: String,
    pub time_points: usize,
    pub roi_count: usize,
    pub connectivity_estimator: &'static str,
    pub estimators: Vec<EstimatorResponseSummary>,
    pub connectivity_direction: ConnectivityDirectionReport,
}

impl FactorialEstimatorResponseReport {
    /// Render deterministic JSON. Estimator IDs are fixed source-code constants.
    pub fn to_json_pretty(&self) -> String {
        let summaries = self
            .estimators
            .iter()
            .map(|item| {
                let nuisance_ratio = option_number_json(item.nuisance_to_signal_distance_ratio);
                let retention = option_number_json(item.signal_retention_fraction);
                format!(
                    concat!(
                        "    {{\n",
                        "      \"estimator_id\": \"{}\",\n",
                        "      \"clean_to_signal_similarity\": {:.12},\n",
                        "      \"clean_to_nuisance_similarity\": {:.12},\n",
                        "      \"signal_to_combined_similarity\": {:.12},\n",
                        "      \"clean_to_combined_similarity\": {:.12},\n",
                        "      \"clean_signal_distance\": {:.12},\n",
                        "      \"nuisance_only_distance\": {:.12},\n",
                        "      \"combined_signal_residual_distance\": {:.12},\n",
                        "      \"nuisance_to_signal_distance_ratio\": {},\n",
                        "      \"signal_retention_fraction\": {}\n",
                        "    }}"
                    ),
                    item.estimator_id,
                    item.clean_to_signal_similarity,
                    item.clean_to_nuisance_similarity,
                    item.signal_to_combined_similarity,
                    item.clean_to_combined_similarity,
                    item.clean_signal_distance,
                    item.nuisance_only_distance,
                    item.combined_signal_residual_distance,
                    nuisance_ratio,
                    retention,
                )
            })
            .collect::<Vec<_>>()
            .join(",\n");
        format!(
            concat!(
                "{{\n",
                "  \"schema_version\": \"{}\",\n",
                "  \"experiment_id\": \"{}\",\n",
                "  \"signal_id\": \"{}\",\n",
                "  \"nuisance_family_id\": \"{}\",\n",
                "  \"nuisance_instance_id\": \"{}\",\n",
                "  \"time_points\": {},\n",
                "  \"roi_count\": {},\n",
                "  \"connectivity_estimator\": \"{}\",\n",
                "  \"estimator_count\": {},\n",
                "  \"estimators\": [\n{}\n  ],\n",
                "  \"connectivity_direction\": {{\n",
                "    \"signal_projection_gain\": {},\n",
                "    \"signal_directional_cosine\": {},\n",
                "    \"signal_change_residual_fraction\": {},\n",
                "    \"clean_signal_change_norm\": {:.12},\n",
                "    \"combined_signal_change_norm\": {:.12}\n",
                "  }},\n",
                "  \"claim_limits\": [\n",
                "    \"synthetic matched time-series cells only\",\n",
                "    \"Pearson connectivity is estimated separately in every factorial cell\",\n",
                "    \"similarity response scores are not participant-level detection power or confidence coverage\",\n",
                "    \"the directional matrix diagnostic measures connectivity-change geometry, not causal direction\",\n",
                "    \"no empirical fMRI or biological endpoint is qualified\"\n",
                "  ]\n",
                "}}\n"
            ),
            self.schema_version,
            self.experiment_id,
            self.signal_id,
            self.nuisance_family_id,
            self.nuisance_instance_id,
            self.time_points,
            self.roi_count,
            self.connectivity_estimator,
            self.estimators.len(),
            summaries,
            option_number_json(self.connectivity_direction.signal_projection_gain),
            option_number_json(self.connectivity_direction.signal_directional_cosine),
            option_number_json(self.connectivity_direction.signal_change_residual_fraction),
            self.connectivity_direction.clean_signal_change_norm,
            self.connectivity_direction.combined_signal_change_norm,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EstimatorResponseError {
    EmptyObservationMatrix,
    RaggedObservationMatrix,
    InsufficientObservationShape,
    NonFiniteObservation,
    ConstantRoi { roi_index: usize },
    InvalidConnectivityEstimate,
    Metric(MetricError),
}

impl fmt::Display for EstimatorResponseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyObservationMatrix => write!(f, "factorial observations must be non-empty"),
            Self::RaggedObservationMatrix => write!(f, "factorial observation matrix must be rectangular"),
            Self::InsufficientObservationShape => write!(
                f,
                "Pearson connectivity requires at least three time points and two ROIs"
            ),
            Self::NonFiniteObservation => write!(f, "factorial observations contain non-finite values"),
            Self::ConstantRoi { roi_index } => {
                write!(f, "ROI {roi_index} has zero or near-zero temporal variance")
            }
            Self::InvalidConnectivityEstimate => {
                write!(f, "connectivity estimation produced a non-finite value")
            }
            Self::Metric(error) => write!(f, "RCAF estimator metric failed: {error}"),
        }
    }
}

impl std::error::Error for EstimatorResponseError {}

impl From<MetricError> for EstimatorResponseError {
    fn from(value: MetricError) -> Self {
        Self::Metric(value)
    }
}

/// Evaluate the three existing RCAF similarity estimators on the four factorial cells.
///
/// Each time × ROI cell is separately transformed into a Pearson ROI-connectivity matrix.
/// For estimator E, the following scores are calculated:
///
/// - baseline target distance: 1 - E(C(X), C(X + S));
/// - nuisance-only displacement: 1 - E(C(X), C(X + A));
/// - residual signal-plus-nuisance displacement: 1 - E(C(X + S), C(X + S + A)).
///
/// The normalized retention fraction is 1 - residual_signal_displacement / baseline_target_distance.
/// It is undefined for a zero-signal point, and is intentionally not clipped to [0, 1].
pub fn evaluate_factorial_estimator_response(
    experiment: &FactorialCounterfactualExperiment,
) -> Result<FactorialEstimatorResponseReport, EstimatorResponseError> {
    let clean = pearson_connectivity_matrix(&experiment.clean_null)?;
    let signal = pearson_connectivity_matrix(&experiment.signal_only)?;
    let nuisance = pearson_connectivity_matrix(&experiment.nuisance_only)?;
    let combined = pearson_connectivity_matrix(&experiment.signal_plus_nuisance)?;

    let estimators = [
        SimilarityEstimator::AbsoluteWeightedJaccard,
        SimilarityEstimator::SignedShiftWeightedJaccard,
        SimilarityEstimator::NormalizedMeanAbsoluteEdgeError,
    ];
    let mut summaries = Vec::with_capacity(estimators.len());
    for estimator in estimators {
        let clean_to_signal_similarity = correlation_similarity(&clean, &signal, estimator)?;
        let clean_to_nuisance_similarity = correlation_similarity(&clean, &nuisance, estimator)?;
        let signal_to_combined_similarity = correlation_similarity(&signal, &combined, estimator)?;
        let clean_to_combined_similarity = correlation_similarity(&clean, &combined, estimator)?;

        let clean_signal_distance = 1.0 - clean_to_signal_similarity;
        let nuisance_only_distance = 1.0 - clean_to_nuisance_similarity;
        let combined_signal_residual_distance = 1.0 - signal_to_combined_similarity;
        let nuisance_to_signal_distance_ratio = (clean_signal_distance > 1e-12)
            .then_some(nuisance_only_distance / clean_signal_distance);
        let signal_retention_fraction = (clean_signal_distance > 1e-12)
            .then_some(1.0 - combined_signal_residual_distance / clean_signal_distance);

        let finite = [
            clean_to_signal_similarity,
            clean_to_nuisance_similarity,
            signal_to_combined_similarity,
            clean_to_combined_similarity,
            clean_signal_distance,
            nuisance_only_distance,
            combined_signal_residual_distance,
        ]
        .iter()
        .all(|value| value.is_finite())
            && nuisance_to_signal_distance_ratio.is_none_or(f64::is_finite)
            && signal_retention_fraction.is_none_or(f64::is_finite);
        if !finite {
            return Err(EstimatorResponseError::InvalidConnectivityEstimate);
        }

        summaries.push(EstimatorResponseSummary {
            estimator_id: estimator.as_str(),
            clean_to_signal_similarity,
            clean_to_nuisance_similarity,
            signal_to_combined_similarity,
            clean_to_combined_similarity,
            clean_signal_distance,
            nuisance_only_distance,
            combined_signal_residual_distance,
            nuisance_to_signal_distance_ratio,
            signal_retention_fraction,
        });
    }

    let clean_signal_delta = upper_triangle_delta(&signal, &clean);
    let combined_signal_delta = upper_triangle_delta(&combined, &nuisance);
    let direction = summarize_directional_recovery(&clean_signal_delta, &combined_signal_delta)?;

    Ok(FactorialEstimatorResponseReport {
        schema_version: "rcaf.factorial-estimator-response/0.1.0",
        experiment_id: experiment.report.experiment_id.clone(),
        signal_id: experiment.report.signal_id.clone(),
        nuisance_family_id: experiment.report.nuisance_family_id.clone(),
        nuisance_instance_id: experiment.report.nuisance_instance_id.clone(),
        time_points: experiment.report.time_points,
        roi_count: experiment.report.roi_count,
        connectivity_estimator: "pearson_sample_correlation_v1",
        estimators: summaries,
        connectivity_direction: direction,
    })
}

/// Estimate one Pearson correlation matrix from time × ROI observations.
pub fn pearson_connectivity_matrix(
    observations: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, EstimatorResponseError> {
    if observations.is_empty() || observations.first().is_none_or(Vec::is_empty) {
        return Err(EstimatorResponseError::EmptyObservationMatrix);
    }
    let time_points = observations.len();
    let roi_count = observations[0].len();
    if observations.iter().any(|row| row.len() != roi_count) {
        return Err(EstimatorResponseError::RaggedObservationMatrix);
    }
    if time_points < 3 || roi_count < 2 {
        return Err(EstimatorResponseError::InsufficientObservationShape);
    }
    if observations.iter().flatten().any(|value| !value.is_finite()) {
        return Err(EstimatorResponseError::NonFiniteObservation);
    }

    let means = (0..roi_count)
        .map(|roi| observations.iter().map(|row| row[roi]).sum::<f64>() / time_points as f64)
        .collect::<Vec<_>>();
    let mut sums_squares = vec![0.0; roi_count];
    for row in observations {
        for (roi, value) in row.iter().enumerate() {
            sums_squares[roi] += (value - means[roi]).powi(2);
        }
    }
    for (roi, variance_sum) in sums_squares.iter().enumerate() {
        if !variance_sum.is_finite() || *variance_sum <= 1e-20 {
            return Err(EstimatorResponseError::ConstantRoi { roi_index: roi });
        }
    }

    let mut matrix = vec![vec![0.0; roi_count]; roi_count];
    for i in 0..roi_count {
        matrix[i][i] = 1.0;
        for j in (i + 1)..roi_count {
            let covariance_sum = observations
                .iter()
                .map(|row| (row[i] - means[i]) * (row[j] - means[j]))
                .sum::<f64>();
            let correlation = covariance_sum / (sums_squares[i] * sums_squares[j]).sqrt();
            if !correlation.is_finite() {
                return Err(EstimatorResponseError::InvalidConnectivityEstimate);
            }
            let correlation = correlation.clamp(-1.0, 1.0);
            matrix[i][j] = correlation;
            matrix[j][i] = correlation;
        }
    }
    Ok(matrix)
}

fn upper_triangle_delta(
    after: &[Vec<f64>],
    before: &[Vec<f64>],
) -> Vec<f64> {
    after
        .iter()
        .enumerate()
        .flat_map(|(row_index, row)| {
            row.iter()
                .enumerate()
                .skip(row_index + 1)
                .map(move |(column_index, value)| value - before[row_index][column_index])
        })
        .collect()
}

fn summarize_directional_recovery(
    signal_delta: &[f64],
    combined_delta: &[f64],
) -> Result<ConnectivityDirectionReport, EstimatorResponseError> {
    if signal_delta.len() != combined_delta.len() || signal_delta.is_empty() {
        return Err(EstimatorResponseError::InvalidConnectivityEstimate);
    }
    let signal_norm_squared = signal_delta.iter().map(|value| value * value).sum::<f64>();
    let combined_norm_squared = combined_delta.iter().map(|value| value * value).sum::<f64>();
    let clean_signal_change_norm = signal_norm_squared.sqrt();
    let combined_signal_change_norm = combined_norm_squared.sqrt();
    if !clean_signal_change_norm.is_finite() || !combined_signal_change_norm.is_finite() {
        return Err(EstimatorResponseError::InvalidConnectivityEstimate);
    }
    let dot = signal_delta
        .iter()
        .zip(combined_delta.iter())
        .map(|(left, right)| left * right)
        .sum::<f64>();
    if !dot.is_finite() {
        return Err(EstimatorResponseError::InvalidConnectivityEstimate);
    }
    if signal_norm_squared <= 1e-24 || combined_norm_squared <= 1e-24 {
        return Ok(ConnectivityDirectionReport {
            signal_projection_gain: None,
            signal_directional_cosine: None,
            signal_change_residual_fraction: None,
            clean_signal_change_norm,
            combined_signal_change_norm,
        });
    }

    let projection_gain = dot / signal_norm_squared;
    let cosine = dot / (signal_norm_squared * combined_norm_squared).sqrt();
    let residual_squared = signal_delta
        .iter()
        .zip(combined_delta.iter())
        .map(|(signal, combined)| (combined - signal).powi(2))
        .sum::<f64>();
    let residual_fraction = (residual_squared.max(0.0).sqrt()) / clean_signal_change_norm;
    if [projection_gain, cosine, residual_fraction]
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(EstimatorResponseError::InvalidConnectivityEstimate);
    }
    Ok(ConnectivityDirectionReport {
        signal_projection_gain: Some(projection_gain),
        signal_directional_cosine: Some(cosine.clamp(-1.0, 1.0)),
        signal_change_residual_fraction: Some(residual_fraction),
        clean_signal_change_norm,
        combined_signal_change_norm,
    })
}

fn option_number_json(value: Option<f64>) -> String {
    value
        .map(|value| format!("{value:.12}"))
        .unwrap_or_else(|| "null".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::factorial_counterfactual::{
        run_factorial_counterfactual_experiment, FactorialExperimentMetadata,
    };

    fn correlation_base() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 0.28, 0.16, -0.10],
            vec![0.28, 1.0, 0.22, 0.08],
            vec![0.16, 0.22, 1.0, 0.30],
            vec![-0.10, 0.08, 0.30, 1.0],
        ]
    }

    fn time_series() -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let clean = (0..96)
            .map(|time| {
                (0..4)
                    .map(|roi| {
                        ((time as f64 + 1.0) * (roi as f64 + 1.0) * 0.071).sin()
                            + 0.10 * ((time as f64 + roi as f64) * 0.13).cos()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let signal = (0..96)
            .map(|time| {
                (0..4)
                    .map(|roi| 0.03 * ((time as f64 + 2.0) * (roi as f64 + 1.0) * 0.047).sin())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let nuisance = (0..96)
            .map(|time| {
                (0..4)
                    .map(|roi| 0.02 * ((time as f64 + 3.0) * (roi as f64 + 1.0) * 0.039).cos())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        (clean, signal, nuisance)
    }

    fn factorial() -> FactorialCounterfactualExperiment {
        let (clean, signal, nuisance) = time_series();
        run_factorial_counterfactual_experiment(
            &clean,
            &signal,
            &nuisance,
            &FactorialExperimentMetadata {
                experiment_id: "estimator-response-01".into(),
                signal_id: "signal-01".into(),
                nuisance_family_id: "motion_transient".into(),
                nuisance_instance_id: "transient-01".into(),
                target_component_variance_fraction: None,
            },
        )
        .unwrap()
    }

    #[test]
    fn reports_all_three_existing_estimators() {
        let response = evaluate_factorial_estimator_response(&factorial()).unwrap();
        assert_eq!(response.estimators.len(), 3);
        assert_eq!(
            response.estimators.iter().map(|item| item.estimator_id).collect::<Vec<_>>(),
            vec![
                "absolute_weighted_jaccard",
                "signed_shift_weighted_jaccard",
                "normalized_mean_absolute_edge_error",
            ]
        );
        assert!(response.estimators.iter().all(|item| {
            (0.0..=1.0).contains(&item.clean_to_signal_similarity)
                && (0.0..=1.0).contains(&item.clean_to_nuisance_similarity)
                && (0.0..=1.0).contains(&item.signal_to_combined_similarity)
        }));
    }

    #[test]
    fn zero_signal_has_undefined_retention_not_a_fabricated_zero_or_one() {
        let (clean, _, nuisance) = time_series();
        let zero = vec![vec![0.0; 4]; 96];
        let experiment = run_factorial_counterfactual_experiment(
            &clean,
            &zero,
            &nuisance,
            &FactorialExperimentMetadata {
                experiment_id: "zero-signal".into(),
                signal_id: "zero".into(),
                nuisance_family_id: "fixture".into(),
                nuisance_instance_id: "a".into(),
                target_component_variance_fraction: None,
            },
        )
        .unwrap();
        let response = evaluate_factorial_estimator_response(&experiment).unwrap();
        assert!(response
            .estimators
            .iter()
            .all(|item| item.signal_retention_fraction.is_none()));
        assert!(response
            .estimators
            .iter()
            .all(|item| item.nuisance_to_signal_distance_ratio.is_none()));
    }

    #[test]
    fn directional_diagnostic_matches_exact_additive_signal_control() {
        let response = evaluate_factorial_estimator_response(&factorial()).unwrap();
        assert!((response.connectivity_direction.signal_projection_gain.unwrap() - 1.0).abs() < 1e-9);
        assert!((response.connectivity_direction.signal_directional_cosine.unwrap() - 1.0).abs() < 1e-9);
        assert!(response.connectivity_direction.signal_change_residual_fraction.unwrap() < 1e-9);
    }

    #[test]
    fn correlation_estimator_rejects_constant_roi_and_bad_dimensions() {
        let constant = vec![vec![0.0, 1.0], vec![0.0, 2.0], vec![0.0, 3.0]];
        assert!(matches!(
            pearson_connectivity_matrix(&constant),
            Err(EstimatorResponseError::ConstantRoi { roi_index: 0 })
        ));
        assert_eq!(
            pearson_connectivity_matrix(&[vec![0.0, 1.0], vec![1.0]]),
            Err(EstimatorResponseError::RaggedObservationMatrix)
        );
    }

    #[test]
    fn estimator_response_is_deterministic_and_json_is_valid_shape() {
        let experiment = factorial();
        let first = evaluate_factorial_estimator_response(&experiment).unwrap();
        let second = evaluate_factorial_estimator_response(&experiment).unwrap();
        assert_eq!(first, second);
        let json = first.to_json_pretty();
        assert!(json.contains("absolute_weighted_jaccard"));
        assert!(json.contains("signal_retention_fraction"));
        assert!(json.contains("signal_directional_cosine"));
    }
}
