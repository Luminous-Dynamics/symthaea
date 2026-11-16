// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Versioned, provenance-bearing observation events for time-aware inference.
//!
//! This module deliberately does not adapt events to the legacy dense
//! `Observation` type or call `ActiveInferenceAgent::perceive`. The legacy
//! path currently has different timestamp and precision semantics. A silent
//! conversion would discard precisely the information this type is intended
//! to preserve.
//!
//! Missing dimensions are represented by `value: None`, so a hidden raw value
//! is not present in the inference-facing event at all. The Gaussian
//! likelihood oracle excludes those dimensions rather than imputing a number
//! and scoring that imputation as if it were measured.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::f64::consts::TAU;

/// Why a dimension was not observed, when the source can say.
/// This is provenance about the measurement process, not evidence of a
/// physical cause in the environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MissingnessReason {
    /// The source's sampling policy did not schedule this variable.
    NotSampled,
    /// The sensor or upstream source was unavailable.
    SensorUnavailable,
    /// A quality or validity gate rejected the measurement.
    QualityRejected,
    /// The value was intentionally withheld or redacted.
    Redacted,
    /// A transport or delivery failure prevented receipt.
    TransportLoss,
    /// The source reported another reason not represented above.
    Other,
}

/// Coarse health declaration from the source at capture/delivery time.
/// This is source telemetry, not a claim that the physical environment is healthy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SourceHealthV1 {
    /// Source reports its nominal operating state.
    Healthy,
    /// Source reports degraded operation or reduced confidence.
    Degraded,
    /// Source reports that it was unavailable for this event.
    Unavailable,
    /// No source-health declaration was supplied.
    #[default]
    Unknown,
}

/// One variable in an observation event.
///
/// `value == None` means that no measurement is available to inference.
/// There is deliberately no second "hidden raw value" field. The offline
/// evaluation harness may retain ground truth separately, outside this type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationDimensionV1 {
    /// Stable variable identity in the generative model's observation space.
    pub variable_id: String,
    /// Physical or domain unit; callers must not silently mix units.
    pub unit: String,
    /// The measured value, or `None` when not observed.
    pub value: Option<f64>,
    /// Per-dimension inverse variance when known and calibrated.
    /// If absent for a visible value, the evaluator's explicit fallback is used.
    pub precision: Option<f64>,
    /// Optional source-declared reason for absence.
    pub missingness: Option<MissingnessReason>,
}

/// One model prediction bound to the variable coordinate it represents.
///
/// The unit must match the event dimension exactly. This type prevents an
/// equal-length but reordered vector from being scored against the wrong
/// variables.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationPredictionDimensionV1 {
    /// Stable variable identity in the generative model's observation space.
    pub variable_id: String,
    /// Physical or domain unit for this prediction.
    pub unit: String,
    /// Predicted value, absent when no prediction is available. It is required
    /// only for a dimension that has an observed measurement.
    pub value: Option<f64>,
}

/// Coordinate-bound prediction for an observation event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationPredictionV1 {
    /// Predictions in the exact ordered coordinate system declared by the event.
    pub dimensions: Vec<ObservationPredictionDimensionV1>,
}

/// Declared dependence structure for measurement noise.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ObservationNoiseModelV1 {
    /// No independence/covariance assumption has been declared.
    #[default]
    Unspecified,
    /// The caller explicitly declares conditionally independent dimensions.
    /// The optional identity points to the calibrated diagonal noise model.
    IndependentDiagonal { model_id: Option<String> },
    /// Cross-dimension covariance is represented by a separate calibrated model.
    /// Do not score this event with a diagonal-independent likelihood.
    JointCovariance { model_id: String },
}

/// An observation as an event in the world and in a delivery system.
///
/// Capture and arrival timestamps are kept separate, with explicit clock
/// domains. They must not be compared unless they share a clock domain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservationEventV1 {
    /// Stable event identity, unique within the source's declared namespace.
    pub event_id: String,
    /// Identity of the sensor or upstream source.
    pub source_id: String,
    /// Semantic observation modality (for example, visual, thermal, or audio).
    pub modality: String,
    /// Measurement-noise dependence contract; defaults to unspecified.
    #[serde(default)]
    pub noise_model: ObservationNoiseModelV1,
    /// Source-reported health; this does not imply physical-world health.
    #[serde(default)]
    pub source_health: SourceHealthV1,
    /// When the underlying measurement was captured, in nanoseconds.
    pub capture_time_ns: u64,
    /// Clock domain that gives meaning to `capture_time_ns`.
    pub capture_clock_domain: String,
    /// When this system received the event, in nanoseconds.
    pub arrival_time_ns: u64,
    /// Clock domain that gives meaning to `arrival_time_ns`.
    pub arrival_clock_domain: String,
    /// Optional identity of the sensor sampling policy.
    pub sampling_policy_id: Option<String>,
    /// Optional identity of the calibration used to derive precision.
    pub calibration_id: Option<String>,
    /// Ordered variables in the generative model's observation space.
    pub dimensions: Vec<ObservationDimensionV1>,
}

/// Structural or numerical contract violations for an observation event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservationEventError {
    EmptyDimensions,
    DimensionCountMismatch {
        expected: usize,
        actual: usize,
    },
    EmptyIdentifier {
        field: &'static str,
    },
    DuplicateVariableId {
        index: usize,
    },
    NonFiniteObservedValue {
        index: usize,
    },
    InvalidPrecision {
        index: usize,
    },
    PrecisionForMissingDimension {
        index: usize,
    },
    MissingnessForObservedDimension {
        index: usize,
    },
    ArrivalBeforeCapture,
    InvalidFallbackPrecision,
    NonFinitePrediction {
        index: usize,
    },
    MissingPredictionForObservedDimension {
        index: usize,
    },
    PredictionCoordinateMismatch {
        index: usize,
    },
    NonFiniteLikelihood,
    NoiseModelNotDiagonalIndependent,
    EmptyLikelihoodMatrix,
    LikelihoodMatrixDimensionMismatch {
        row: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteLikelihoodCoefficient {
        row: usize,
        column: usize,
    },
    NonFiniteWeightedResidual {
        index: usize,
    },
    NonFiniteBeliefGradient {
        row: usize,
    },
}

impl std::fmt::Display for ObservationEventError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ObservationEventError {}

/// Result of evaluating only the observed dimensions under an explicitly
/// declared independent diagonal Gaussian likelihood.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationLikelihoodV1 {
    /// Number of dimensions that contributed a likelihood term.
    pub observed_dimensions: usize,
    /// Total dimensions in the declared observation space.
    pub total_dimensions: usize,
    /// Sum of normalized Gaussian log-likelihood terms, in nats.
    pub log_likelihood: f64,
    /// Sum of precision-weighted squared residuals.
    pub weighted_squared_error: f64,
    /// Mean log likelihood, absent when no dimensions were observed.
    pub mean_log_likelihood: Option<f64>,
}

/// Log-likelihood ascent gradient with respect to hidden-state coordinates.
///
/// For a linear observation model y_hat[j] = sum_i A[i][j] * s[i], this
/// stores d log p(y_observed | s) / d s[i] = sum_j A[i][j] *
/// precision[j] * (value[j] - prediction[j]). Absent dimensions contribute
/// exactly zero and their predictions are never read.
#[derive(Debug, Clone, PartialEq)]
pub struct ObservationBeliefGradientV1 {
    /// Number of observation dimensions that contributed to the gradient.
    pub observed_dimensions: usize,
    /// Total dimensions in the event's declared observation space.
    pub total_dimensions: usize,
    /// Gradient in hidden-state coordinate order (one value per matrix row).
    pub values: Vec<f64>,
}

impl ObservationEventV1 {
    fn validate_prediction_coordinates(
        &self,
        predicted: &ObservationPredictionV1,
    ) -> Result<(), ObservationEventError> {
        if predicted.dimensions.len() != self.dimensions.len() {
            return Err(ObservationEventError::DimensionCountMismatch {
                expected: self.dimensions.len(),
                actual: predicted.dimensions.len(),
            });
        }
        for (index, (observed, prediction)) in self
            .dimensions
            .iter()
            .zip(predicted.dimensions.iter())
            .enumerate()
        {
            if observed.variable_id != prediction.variable_id || observed.unit != prediction.unit {
                return Err(ObservationEventError::PredictionCoordinateMismatch { index });
            }
        }
        Ok(())
    }

    /// Delivery delay when capture and arrival timestamps share a clock domain.
    /// Returns `None` when clocks differ because subtracting incomparable
    /// timestamps would invent an age measurement.
    pub fn delivery_delay_ns(&self) -> Option<u64> {
        if self.capture_clock_domain == self.arrival_clock_domain {
            self.arrival_time_ns.checked_sub(self.capture_time_ns)
        } else {
            None
        }
    }

    /// Validate event structure against the fixed observation dimension.
    ///
    /// When capture and arrival clocks differ, their raw timestamps are
    /// intentionally not ordered here: a clock conversion contract is needed
    /// before such a comparison is meaningful.
    pub fn validate(&self, expected_dimensions: usize) -> Result<(), ObservationEventError> {
        if expected_dimensions == 0 {
            return Err(ObservationEventError::EmptyDimensions);
        }
        if self.dimensions.len() != expected_dimensions {
            return Err(ObservationEventError::DimensionCountMismatch {
                expected: expected_dimensions,
                actual: self.dimensions.len(),
            });
        }

        for (field, value) in [
            ("event_id", self.event_id.as_str()),
            ("source_id", self.source_id.as_str()),
            ("modality", self.modality.as_str()),
            ("capture_clock_domain", self.capture_clock_domain.as_str()),
            ("arrival_clock_domain", self.arrival_clock_domain.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(ObservationEventError::EmptyIdentifier { field });
            }
        }
        let noise_model_id = match &self.noise_model {
            ObservationNoiseModelV1::Unspecified => None,
            ObservationNoiseModelV1::IndependentDiagonal { model_id } => model_id.as_deref(),
            ObservationNoiseModelV1::JointCovariance { model_id } => Some(model_id.as_str()),
        };
        if noise_model_id.is_some_and(|value| value.trim().is_empty()) {
            return Err(ObservationEventError::EmptyIdentifier {
                field: "noise_model_id",
            });
        }

        for (field, value) in [
            ("sampling_policy_id", self.sampling_policy_id.as_deref()),
            ("calibration_id", self.calibration_id.as_deref()),
        ] {
            if value.is_some_and(|value| value.trim().is_empty()) {
                return Err(ObservationEventError::EmptyIdentifier { field });
            }
        }

        if self.capture_clock_domain == self.arrival_clock_domain
            && self.arrival_time_ns < self.capture_time_ns
        {
            return Err(ObservationEventError::ArrivalBeforeCapture);
        }

        let mut variable_ids = HashSet::with_capacity(self.dimensions.len());
        for (index, dimension) in self.dimensions.iter().enumerate() {
            if dimension.variable_id.trim().is_empty() {
                return Err(ObservationEventError::EmptyIdentifier {
                    field: "variable_id",
                });
            }
            if dimension.unit.trim().is_empty() {
                return Err(ObservationEventError::EmptyIdentifier { field: "unit" });
            }
            if !variable_ids.insert(dimension.variable_id.as_str()) {
                return Err(ObservationEventError::DuplicateVariableId { index });
            }

            match (dimension.value, dimension.precision, dimension.missingness) {
                (Some(value), precision, None) => {
                    if !value.is_finite() {
                        return Err(ObservationEventError::NonFiniteObservedValue { index });
                    }
                    if precision.is_some_and(|p| !p.is_finite() || p <= 0.0) {
                        return Err(ObservationEventError::InvalidPrecision { index });
                    }
                }
                (Some(_), _, Some(_)) => {
                    return Err(ObservationEventError::MissingnessForObservedDimension { index });
                }
                (None, Some(_), _) => {
                    return Err(ObservationEventError::PrecisionForMissingDimension { index });
                }
                (None, None, _) => {}
            }
        }

        Ok(())
    }

    /// Return residuals for observed dimensions and `None` for absent ones.
    ///
    /// Predictions for absent dimensions are not read; they may be arbitrary
    /// without changing this result.
    pub fn masked_residuals(
        &self,
        predicted: &ObservationPredictionV1,
    ) -> Result<Vec<Option<f64>>, ObservationEventError> {
        self.validate(predicted.dimensions.len())?;
        self.validate_prediction_coordinates(predicted)?;
        self.dimensions
            .iter()
            .enumerate()
            .map(|(index, dimension)| match dimension.value {
                Some(value) => {
                    let prediction = predicted.dimensions[index].value.ok_or(
                        ObservationEventError::MissingPredictionForObservedDimension { index },
                    )?;
                    if !prediction.is_finite() {
                        return Err(ObservationEventError::NonFinitePrediction { index });
                    }
                    Ok(Some(prediction - value))
                }
                None => Ok(None),
            })
            .collect()
    }

    /// Compute a mask-aware hidden-state likelihood gradient for a linear
    /// observation mapping. The matrix uses
    /// [hidden_state_dimension][observation_dimension] order.
    ///
    /// For the explicitly declared IndependentDiagonal noise model, each
    /// observed coordinate contributes A[i][j] * precision[j] *
    /// (value[j] - prediction[j]). An absent coordinate contributes exactly
    /// zero; even a non-finite placeholder prediction at that absent coordinate
    /// is never read. The matrix is validated before a gradient is returned.
    ///
    /// An all-missing event returns an all-zero gradient even when its noise
    /// dependence is unspecified, because no observation likelihood term is
    /// evaluated. Otherwise unspecified or joint-covariance noise fails closed;
    /// use the separately owned covariance-aware path for correlated noise.
    ///
    /// This is a pure O1 reference primitive. It does not mutate an agent,
    /// generative model, TD learner, eligibility trace or transition history.
    pub fn masked_belief_gradient(
        &self,
        predicted: &ObservationPredictionV1,
        likelihood_matrix: &[Vec<f64>],
        fallback_precision: f64,
    ) -> Result<ObservationBeliefGradientV1, ObservationEventError> {
        self.validate(predicted.dimensions.len())?;
        self.validate_prediction_coordinates(predicted)?;

        if likelihood_matrix.is_empty() {
            return Err(ObservationEventError::EmptyLikelihoodMatrix);
        }

        for (row_index, row) in likelihood_matrix.iter().enumerate() {
            if row.len() != self.dimensions.len() {
                return Err(ObservationEventError::LikelihoodMatrixDimensionMismatch {
                    row: row_index,
                    expected: self.dimensions.len(),
                    actual: row.len(),
                });
            }
            for (column_index, coefficient) in row.iter().enumerate() {
                if !coefficient.is_finite() {
                    return Err(ObservationEventError::NonFiniteLikelihoodCoefficient {
                        row: row_index,
                        column: column_index,
                    });
                }
            }
        }

        let observed_dimensions = self.dimensions.iter().filter(|d| d.value.is_some()).count();
        if observed_dimensions == 0 {
            return Ok(ObservationBeliefGradientV1 {
                observed_dimensions: 0,
                total_dimensions: self.dimensions.len(),
                values: vec![0.0; likelihood_matrix.len()],
            });
        }

        if !matches!(
            &self.noise_model,
            ObservationNoiseModelV1::IndependentDiagonal { .. }
        ) {
            return Err(ObservationEventError::NoiseModelNotDiagonalIndependent);
        }
        if !fallback_precision.is_finite() || fallback_precision <= 0.0 {
            return Err(ObservationEventError::InvalidFallbackPrecision);
        }

        let mut weighted_residuals = vec![None; self.dimensions.len()];
        for (index, dimension) in self.dimensions.iter().enumerate() {
            let Some(value) = dimension.value else {
                continue;
            };
            let prediction = predicted.dimensions[index]
                .value
                .ok_or(ObservationEventError::MissingPredictionForObservedDimension { index })?;
            if !prediction.is_finite() {
                return Err(ObservationEventError::NonFinitePrediction { index });
            }

            let precision = dimension.precision.unwrap_or(fallback_precision);
            let weighted_residual = precision * (value - prediction);
            if !weighted_residual.is_finite() {
                return Err(ObservationEventError::NonFiniteWeightedResidual { index });
            }
            weighted_residuals[index] = Some(weighted_residual);
        }

        let mut values = Vec::with_capacity(likelihood_matrix.len());
        for (row_index, row) in likelihood_matrix.iter().enumerate() {
            let mut gradient = 0.0;
            for (column_index, weighted_residual) in weighted_residuals.iter().enumerate() {
                if let Some(weighted_residual) = weighted_residual {
                    gradient += row[column_index] * weighted_residual;
                }
            }
            if !gradient.is_finite() {
                return Err(ObservationEventError::NonFiniteBeliefGradient { row: row_index });
            }
            values.push(gradient);
        }

        Ok(ObservationBeliefGradientV1 {
            observed_dimensions,
            total_dimensions: self.dimensions.len(),
            values,
        })
    }

    /// Evaluate a normalized diagonal-Gaussian log likelihood over observed
    /// dimensions only. The event must explicitly declare
    /// `IndependentDiagonal`; unspecified or joint covariance models fail closed.
    ///
    /// A visible dimension uses its calibrated per-dimension inverse variance
    /// when present; otherwise it uses `fallback_precision`. A missing
    /// dimension contributes exactly zero and its prediction is not read.
    ///
    /// This reference does not model cross-dimension covariance. Use a
    /// separately qualified joint-covariance likelihood when channels are
    /// correlated; do not diagonalize a joint model silently.
    ///
    /// The Gaussian normalization term is retained:
    /// `0.5 * (ln(precision) - ln(2π) - precision * residual²)`.
    /// This matters when comparing different precision values; the
    /// unnormalized squared-error term alone is not a proper likelihood.
    pub fn evaluate_diagonal_gaussian_likelihood(
        &self,
        predicted: &ObservationPredictionV1,
        fallback_precision: f64,
    ) -> Result<ObservationLikelihoodV1, ObservationEventError> {
        self.validate(predicted.dimensions.len())?;
        self.validate_prediction_coordinates(predicted)?;
        if !matches!(
            &self.noise_model,
            ObservationNoiseModelV1::IndependentDiagonal { .. }
        ) {
            return Err(ObservationEventError::NoiseModelNotDiagonalIndependent);
        }
        if !fallback_precision.is_finite() || fallback_precision <= 0.0 {
            return Err(ObservationEventError::InvalidFallbackPrecision);
        }

        let mut observed_dimensions = 0usize;
        let mut log_likelihood = 0.0;
        let mut weighted_squared_error = 0.0;

        for (index, dimension) in self.dimensions.iter().enumerate() {
            let Some(value) = dimension.value else {
                continue;
            };
            let prediction = predicted.dimensions[index]
                .value
                .ok_or(ObservationEventError::MissingPredictionForObservedDimension { index })?;
            if !prediction.is_finite() {
                return Err(ObservationEventError::NonFinitePrediction { index });
            }

            let precision = dimension.precision.unwrap_or(fallback_precision);
            let residual = prediction - value;
            let weighted_error = precision * residual * residual;
            let term = 0.5 * (precision.ln() - TAU.ln() - weighted_error);

            weighted_squared_error += weighted_error;
            log_likelihood += term;
            observed_dimensions += 1;
        }

        if !weighted_squared_error.is_finite() || !log_likelihood.is_finite() {
            return Err(ObservationEventError::NonFiniteLikelihood);
        }

        Ok(ObservationLikelihoodV1 {
            observed_dimensions,
            total_dimensions: self.dimensions.len(),
            log_likelihood,
            weighted_squared_error,
            mean_log_likelihood: (observed_dimensions > 0)
                .then_some(log_likelihood / observed_dimensions as f64),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dimension(
        variable_id: &str,
        value: Option<f64>,
        precision: Option<f64>,
        missingness: Option<MissingnessReason>,
    ) -> ObservationDimensionV1 {
        ObservationDimensionV1 {
            variable_id: variable_id.to_string(),
            unit: "unit".to_string(),
            value,
            precision,
            missingness,
        }
    }

    fn prediction(values: &[(&str, Option<f64>)]) -> ObservationPredictionV1 {
        ObservationPredictionV1 {
            dimensions: values
                .iter()
                .map(|(variable_id, value)| ObservationPredictionDimensionV1 {
                    variable_id: (*variable_id).to_string(),
                    unit: "unit".to_string(),
                    value: *value,
                })
                .collect(),
        }
    }

    fn event(dimensions: Vec<ObservationDimensionV1>) -> ObservationEventV1 {
        ObservationEventV1 {
            event_id: "event-1".to_string(),
            source_id: "sensor-1".to_string(),
            modality: "thermal".to_string(),
            noise_model: ObservationNoiseModelV1::IndependentDiagonal {
                model_id: Some("independent-diagonal-v1".to_string()),
            },
            source_health: SourceHealthV1::Healthy,
            capture_time_ns: 1_000,
            capture_clock_domain: "sensor-clock".to_string(),
            arrival_time_ns: 1_200,
            arrival_clock_domain: "sensor-clock".to_string(),
            sampling_policy_id: Some("policy-regular-1hz".to_string()),
            calibration_id: Some("calibration-2026-01".to_string()),
            dimensions,
        }
    }

    #[test]
    fn normalized_gaussian_likelihood_matches_closed_form() {
        let observation = event(vec![dimension("temperature", Some(0.5), Some(4.0), None)]);
        let likelihood = observation
            .evaluate_diagonal_gaussian_likelihood(&prediction(&[("temperature", Some(0.25))]), 1.0)
            .expect("valid observation");
        let expected = 0.5 * (4.0_f64.ln() - TAU.ln() - 4.0 * 0.25_f64.powi(2));

        assert_eq!(likelihood.observed_dimensions, 1);
        assert_eq!(likelihood.total_dimensions, 1);
        assert!((likelihood.log_likelihood - expected).abs() < 1e-12);
        assert!((likelihood.weighted_squared_error - 0.25).abs() < 1e-12);
    }

    #[test]
    fn missing_dimension_is_excluded_not_imputed_or_scored() {
        let observation = event(vec![
            dimension("temperature", Some(0.5), Some(4.0), None),
            dimension(
                "pressure",
                None,
                None,
                Some(MissingnessReason::SensorUnavailable),
            ),
        ]);

        // A non-finite prediction in the absent dimension is intentionally
        // ignored: it cannot contribute a residual or likelihood term.
        let residuals = observation
            .masked_residuals(&prediction(&[
                ("temperature", Some(0.25)),
                ("pressure", None),
            ]))
            .expect("missing dimension is not read");
        assert_eq!(residuals[0], Some(-0.25));
        assert_eq!(residuals[1], None);

        let likelihood = observation
            .evaluate_diagonal_gaussian_likelihood(
                &prediction(&[("temperature", Some(0.25)), ("pressure", None)]),
                1.0,
            )
            .expect("missing dimension is not scored");
        assert_eq!(likelihood.observed_dimensions, 1);
        assert_eq!(likelihood.total_dimensions, 2);
        assert!((likelihood.weighted_squared_error - 0.25).abs() < 1e-12);
    }

    #[test]
    fn all_missing_event_has_no_observation_likelihood() {
        let observation = event(vec![
            dimension(
                "temperature",
                None,
                None,
                Some(MissingnessReason::NotSampled),
            ),
            dimension("pressure", None, None, None),
        ]);
        let likelihood = observation
            .evaluate_diagonal_gaussian_likelihood(
                &prediction(&[("temperature", Some(f64::NAN)), ("pressure", None)]),
                1.0,
            )
            .expect("no missing prediction is read");

        assert_eq!(likelihood.observed_dimensions, 0);
        assert_eq!(likelihood.log_likelihood, 0.0);
        assert_eq!(likelihood.weighted_squared_error, 0.0);
        assert_eq!(likelihood.mean_log_likelihood, None);
    }

    #[test]
    fn per_dimension_precision_uses_normalized_likelihood() {
        let low = event(vec![dimension("temperature", Some(0.0), Some(1.0), None)]);
        let high = event(vec![dimension("temperature", Some(0.0), Some(4.0), None)]);
        let low_score = low
            .evaluate_diagonal_gaussian_likelihood(&prediction(&[("temperature", Some(0.5))]), 1.0)
            .expect("valid low precision");
        let high_score = high
            .evaluate_diagonal_gaussian_likelihood(&prediction(&[("temperature", Some(0.5))]), 1.0)
            .expect("valid high precision");

        assert!(
            (high_score.weighted_squared_error - 4.0 * low_score.weighted_squared_error).abs()
                < 1e-12
        );
        // The normalization term makes scores comparable across precision
        // values; weighted squared error alone is not a Gaussian likelihood.
        assert!(high_score.log_likelihood.is_finite());
        assert!(low_score.log_likelihood.is_finite());
    }

    #[test]
    fn diagonal_likelihood_refuses_unspecified_or_joint_noise_models() {
        let mut unspecified = event(vec![dimension("temperature", Some(0.5), None, None)]);
        unspecified.noise_model = ObservationNoiseModelV1::Unspecified;
        assert_eq!(
            unspecified.evaluate_diagonal_gaussian_likelihood(
                &prediction(&[("temperature", Some(0.25))]),
                1.0,
            ),
            Err(ObservationEventError::NoiseModelNotDiagonalIndependent)
        );

        let mut joint = event(vec![dimension("temperature", Some(0.5), None, None)]);
        joint.noise_model = ObservationNoiseModelV1::JointCovariance {
            model_id: "calibrated-joint-covariance-1".to_string(),
        };
        assert_eq!(
            joint.evaluate_diagonal_gaussian_likelihood(
                &prediction(&[("temperature", Some(0.25))]),
                1.0,
            ),
            Err(ObservationEventError::NoiseModelNotDiagonalIndependent)
        );
    }

    #[test]
    fn absent_per_dimension_precision_uses_explicit_fallback() {
        let observation = event(vec![dimension("temperature", Some(0.5), None, None)]);
        let actual = observation
            .evaluate_diagonal_gaussian_likelihood(&prediction(&[("temperature", Some(0.25))]), 4.0)
            .expect("valid fallback");
        let explicit = event(vec![dimension("temperature", Some(0.5), Some(4.0), None)])
            .evaluate_diagonal_gaussian_likelihood(&prediction(&[("temperature", Some(0.25))]), 1.0)
            .expect("valid explicit precision");

        assert!((actual.log_likelihood - explicit.log_likelihood).abs() < 1e-12);
    }

    #[test]
    fn malformed_dimensions_are_rejected() {
        let bad_precision = event(vec![dimension("temperature", Some(0.5), Some(0.0), None)]);
        assert_eq!(
            bad_precision.validate(1),
            Err(ObservationEventError::InvalidPrecision { index: 0 })
        );

        let precision_for_missing = event(vec![dimension("temperature", None, Some(1.0), None)]);
        assert_eq!(
            precision_for_missing.validate(1),
            Err(ObservationEventError::PrecisionForMissingDimension { index: 0 })
        );

        let mismatched = event(vec![dimension("temperature", Some(0.5), None, None)]);
        assert_eq!(
            mismatched.validate(2),
            Err(ObservationEventError::DimensionCountMismatch {
                expected: 2,
                actual: 1
            })
        );

        let empty_noise_model_id = ObservationEventV1 {
            noise_model: ObservationNoiseModelV1::JointCovariance {
                model_id: "   ".to_string(),
            },
            ..event(vec![dimension("temperature", Some(0.5), None, None)])
        };
        assert_eq!(
            empty_noise_model_id.validate(1),
            Err(ObservationEventError::EmptyIdentifier {
                field: "noise_model_id"
            })
        );
    }

    #[test]
    fn duplicate_variables_and_non_finite_observations_are_rejected() {
        let duplicate = event(vec![
            dimension("temperature", Some(0.5), None, None),
            dimension("temperature", Some(0.6), None, None),
        ]);
        assert_eq!(
            duplicate.validate(2),
            Err(ObservationEventError::DuplicateVariableId { index: 1 })
        );

        let non_finite = event(vec![dimension("temperature", Some(f64::NAN), None, None)]);
        assert_eq!(
            non_finite.validate(1),
            Err(ObservationEventError::NonFiniteObservedValue { index: 0 })
        );
    }

    #[test]
    fn delivery_age_is_only_derived_within_a_shared_clock_domain() {
        let mut same_clock = event(vec![dimension("temperature", Some(0.5), None, None)]);
        assert_eq!(same_clock.delivery_delay_ns(), Some(200));

        same_clock.arrival_time_ns = 999;
        assert_eq!(same_clock.delivery_delay_ns(), None);
        same_clock.arrival_time_ns = 1_200;

        same_clock.arrival_clock_domain = "host-monotonic-clock".to_string();
        assert_eq!(same_clock.delivery_delay_ns(), None);
    }

    #[test]
    fn capture_and_arrival_order_is_checked_only_within_one_clock_domain() {
        let mut same_clock = event(vec![dimension("temperature", Some(0.5), None, None)]);
        same_clock.arrival_time_ns = 999;
        assert_eq!(
            same_clock.validate(1),
            Err(ObservationEventError::ArrivalBeforeCapture)
        );

        let mut different_clocks = same_clock;
        different_clocks.arrival_clock_domain = "host-monotonic-clock".to_string();
        assert!(different_clocks.validate(1).is_ok());
    }

    #[test]
    fn observed_dimensions_cannot_also_claim_a_missingness_reason() {
        let malformed = event(vec![dimension(
            "temperature",
            Some(0.5),
            None,
            Some(MissingnessReason::QualityRejected),
        )]);
        assert_eq!(
            malformed.validate(1),
            Err(ObservationEventError::MissingnessForObservedDimension { index: 0 })
        );
    }

    #[test]
    fn reordered_prediction_variables_are_rejected_before_scoring() {
        let observation = event(vec![
            dimension("temperature", Some(0.5), Some(4.0), None),
            dimension("pressure", Some(0.8), Some(2.0), None),
        ]);
        let reordered = prediction(&[("pressure", Some(0.25)), ("temperature", Some(0.75))]);

        assert_eq!(
            observation.evaluate_diagonal_gaussian_likelihood(&reordered, 1.0),
            Err(ObservationEventError::PredictionCoordinateMismatch { index: 0 })
        );
        assert_eq!(
            observation.masked_residuals(&reordered),
            Err(ObservationEventError::PredictionCoordinateMismatch { index: 0 })
        );
    }

    #[test]
    fn observed_dimension_requires_a_prediction() {
        let observation = event(vec![dimension("temperature", Some(0.5), Some(4.0), None)]);
        let missing_prediction = prediction(&[("temperature", None)]);

        assert_eq!(
            observation.evaluate_diagonal_gaussian_likelihood(&missing_prediction, 1.0),
            Err(ObservationEventError::MissingPredictionForObservedDimension { index: 0 })
        );
        assert_eq!(
            observation.masked_residuals(&missing_prediction),
            Err(ObservationEventError::MissingPredictionForObservedDimension { index: 0 })
        );
    }

    #[test]
    fn prediction_unit_mismatch_is_rejected_before_scoring() {
        let observation = event(vec![dimension("temperature", Some(0.5), Some(4.0), None)]);
        let prediction = ObservationPredictionV1 {
            dimensions: vec![ObservationPredictionDimensionV1 {
                variable_id: "temperature".to_string(),
                unit: "fahrenheit".to_string(),
                value: 0.25,
            }],
        };

        assert_eq!(
            observation.evaluate_diagonal_gaussian_likelihood(&prediction, 1.0),
            Err(ObservationEventError::PredictionCoordinateMismatch { index: 0 })
        );
    }

    #[test]
    fn masked_belief_gradient_matches_linear_gaussian_closed_form() {
        let observation = event(vec![dimension("temperature", Some(0.5), Some(4.0), None)]);
        let predicted = prediction(&[("temperature", Some(0.25))]);
        let gradient = observation
            .masked_belief_gradient(&predicted, &[vec![2.0], vec![3.0]], 1.0)
            .expect("valid independent observation model");

        assert_eq!(gradient.observed_dimensions, 1);
        assert_eq!(gradient.total_dimensions, 1);
        assert_eq!(gradient.values, vec![2.0, 3.0]);
    }

    #[test]
    fn fully_observed_gradient_matches_dense_precision_weighted_projection() {
        let observation = event(vec![
            dimension("temperature", Some(0.5), Some(4.0), None),
            dimension("pressure", Some(0.8), Some(2.0), None),
        ]);
        let predicted = prediction(&[
            ("temperature", Some(0.25)),
            ("pressure", Some(0.3)),
        ]);
        let matrix = vec![vec![2.0, 4.0], vec![3.0, 5.0]];
        let gradient = observation
            .masked_belief_gradient(&predicted, &matrix, 1.0)
            .expect("fully observed diagonal-Gaussian update");

        let dense_weighted_error = [4.0 * (0.5 - 0.25), 2.0 * (0.8 - 0.3)];
        let dense_reference = vec![
            matrix[0][0] * dense_weighted_error[0]
                + matrix[0][1] * dense_weighted_error[1],
            matrix[1][0] * dense_weighted_error[0]
                + matrix[1][1] * dense_weighted_error[1],
        ];

        assert_eq!(gradient.observed_dimensions, 2);
        assert_eq!(gradient.total_dimensions, 2);
        assert_eq!(gradient.values, dense_reference);
        assert_eq!(gradient.values, vec![6.0, 8.0]);
    }

    #[test]
    fn masked_belief_gradient_never_reads_absent_dimension_predictions() {
        let observation = event(vec![
            dimension("temperature", Some(0.5), Some(4.0), None),
            dimension(
                "pressure",
                None,
                None,
                Some(MissingnessReason::SensorUnavailable),
            ),
        ]);
        let first = observation
            .masked_belief_gradient(
                &prediction(&[("temperature", Some(0.25)), ("pressure", Some(f64::NAN))]),
                &[vec![2.0, 100.0], vec![3.0, -500.0]],
                1.0,
            )
            .expect("absent prediction must not be read");
        let second = observation
            .masked_belief_gradient(
                &prediction(&[("temperature", Some(0.25)), ("pressure", None)]),
                &[vec![2.0, 100.0], vec![3.0, -500.0]],
                1.0,
            )
            .expect("missing absent prediction is equivalent");

        assert_eq!(first, second);
        assert_eq!(first.observed_dimensions, 1);
        assert_eq!(first.values, vec![2.0, 3.0]);
    }

    #[test]
    fn all_missing_gradient_is_zero_without_a_noise_independence_claim() {
        let mut observation = event(vec![
            dimension(
                "temperature",
                None,
                None,
                Some(MissingnessReason::NotSampled),
            ),
            dimension("pressure", None, None, None),
        ]);
        observation.noise_model = ObservationNoiseModelV1::Unspecified;
        let gradient = observation
            .masked_belief_gradient(
                &prediction(&[("temperature", Some(f64::NAN)), ("pressure", None)]),
                &[vec![2.0, 3.0], vec![4.0, 5.0]],
                1.0,
            )
            .expect("no observation likelihood is evaluated");

        assert_eq!(gradient.observed_dimensions, 0);
        assert_eq!(gradient.total_dimensions, 2);
        assert_eq!(gradient.values, vec![0.0, 0.0]);
    }

    #[test]
    fn masked_belief_gradient_rejects_joint_noise_and_malformed_matrices() {
        let observation = event(vec![dimension("temperature", Some(0.5), Some(4.0), None)]);
        let predicted = prediction(&[("temperature", Some(0.25))]);
        let mut joint = observation.clone();
        joint.noise_model = ObservationNoiseModelV1::JointCovariance {
            model_id: "joint-covariance-v1".to_string(),
        };

        assert_eq!(
            joint.masked_belief_gradient(&predicted, &[vec![1.0]], 1.0),
            Err(ObservationEventError::NoiseModelNotDiagonalIndependent)
        );
        assert_eq!(
            observation.masked_belief_gradient(&predicted, &[vec![1.0, 2.0]], 1.0),
            Err(ObservationEventError::LikelihoodMatrixDimensionMismatch {
                row: 0,
                expected: 1,
                actual: 2,
            })
        );
        assert_eq!(
            observation.masked_belief_gradient(&predicted, &[vec![f64::NAN]], 1.0),
            Err(ObservationEventError::NonFiniteLikelihoodCoefficient { row: 0, column: 0 })
        );
    }

    #[test]
    fn masked_belief_gradient_rejects_empty_matrix_and_non_finite_terms() {
        let observation = event(vec![dimension("temperature", Some(0.5), Some(4.0), None)]);
        let predicted = prediction(&[("temperature", Some(0.25))]);

        assert_eq!(
            observation.masked_belief_gradient(&predicted, &[], 1.0),
            Err(ObservationEventError::EmptyLikelihoodMatrix)
        );

        let extreme_precision = event(vec![dimension(
            "temperature",
            Some(10.5),
            Some(1.0e308),
            None,
        )]);
        assert_eq!(
            extreme_precision.masked_belief_gradient(&predicted, &[vec![1.0]], 1.0),
            Err(ObservationEventError::NonFiniteWeightedResidual { index: 0 })
        );

        let extreme_coefficient = event(vec![dimension(
            "temperature",
            Some(0.5),
            Some(1.0e10),
            None,
        )]);
        assert_eq!(
            extreme_coefficient.masked_belief_gradient(&predicted, &[vec![1.0e308]], 1.0,),
            Err(ObservationEventError::NonFiniteBeliefGradient { row: 0 })
        );
    }

    #[test]
    fn invalid_prediction_and_fallback_precision_are_rejected_for_observed_dims() {
        let observation = event(vec![dimension("temperature", Some(0.5), None, None)]);
        assert_eq!(
            observation.evaluate_diagonal_gaussian_likelihood(
                &prediction(&[("temperature", Some(f64::NAN))]),
                1.0,
            ),
            Err(ObservationEventError::NonFinitePrediction { index: 0 })
        );
        assert_eq!(
            observation.evaluate_diagonal_gaussian_likelihood(
                &prediction(&[("temperature", Some(0.5))]),
                0.0,
            ),
            Err(ObservationEventError::InvalidFallbackPrecision)
        );
    }
}
