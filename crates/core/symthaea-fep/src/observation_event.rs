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
    DimensionCountMismatch { expected: usize, actual: usize },
    EmptyIdentifier { field: &'static str },
    DuplicateVariableId { index: usize },
    NonFiniteObservedValue { index: usize },
    InvalidPrecision { index: usize },
    PrecisionForMissingDimension { index: usize },
    MissingnessForObservedDimension { index: usize },
    ArrivalBeforeCapture,
    InvalidFallbackPrecision,
    NonFinitePrediction { index: usize },
    NonFiniteLikelihood,
}

/// Result of evaluating only the observed dimensions under an independent
/// diagonal Gaussian likelihood.
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

impl ObservationEventV1 {
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
        predicted: &[f64],
    ) -> Result<Vec<Option<f64>>, ObservationEventError> {
        self.validate(predicted.len())?;
        self.dimensions
            .iter()
            .enumerate()
            .map(|(index, dimension)| match dimension.value {
                Some(value) => {
                    let prediction = predicted[index];
                    if !prediction.is_finite() {
                        return Err(ObservationEventError::NonFinitePrediction { index });
                    }
                    Ok(Some(prediction - value))
                }
                None => Ok(None),
            })
            .collect()
    }

    /// Evaluate a normalized diagonal-Gaussian log likelihood over observed
    /// dimensions only.
    ///
    /// A visible dimension uses its calibrated per-dimension inverse variance
    /// when present; otherwise it uses `fallback_precision`. A missing
    /// dimension contributes exactly zero and its prediction is not read.
    ///
    /// The Gaussian normalization term is retained:
    /// `0.5 * (ln(precision) - ln(2π) - precision * residual²)`.
    /// This matters when comparing different precision values; the
    /// unnormalized squared-error term alone is not a proper likelihood.
    pub fn evaluate_gaussian_likelihood(
        &self,
        predicted: &[f64],
        fallback_precision: f64,
    ) -> Result<ObservationLikelihoodV1, ObservationEventError> {
        self.validate(predicted.len())?;
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
            let prediction = predicted[index];
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

    fn event(dimensions: Vec<ObservationDimensionV1>) -> ObservationEventV1 {
        ObservationEventV1 {
            event_id: "event-1".to_string(),
            source_id: "sensor-1".to_string(),
            modality: "thermal".to_string(),
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
            .evaluate_gaussian_likelihood(&[0.25], 1.0)
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
            dimension("pressure", None, None, Some(MissingnessReason::SensorUnavailable)),
        ]);

        // A non-finite prediction in the absent dimension is intentionally
        // ignored: it cannot contribute a residual or likelihood term.
        let residuals = observation
            .masked_residuals(&[0.25, f64::NAN])
            .expect("missing dimension is not read");
        assert_eq!(residuals[0], Some(-0.25));
        assert_eq!(residuals[1], None);

        let likelihood = observation
            .evaluate_gaussian_likelihood(&[0.25, f64::NAN], 1.0)
            .expect("missing dimension is not scored");
        assert_eq!(likelihood.observed_dimensions, 1);
        assert_eq!(likelihood.total_dimensions, 2);
        assert!((likelihood.weighted_squared_error - 0.25).abs() < 1e-12);
    }

    #[test]
    fn all_missing_event_has_no_observation_likelihood() {
        let observation = event(vec![
            dimension("temperature", None, None, Some(MissingnessReason::NotSampled)),
            dimension("pressure", None, None, None),
        ]);
        let likelihood = observation
            .evaluate_gaussian_likelihood(&[f64::NAN, f64::INFINITY], 1.0)
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
            .evaluate_gaussian_likelihood(&[0.5], 1.0)
            .expect("valid low precision");
        let high_score = high
            .evaluate_gaussian_likelihood(&[0.5], 1.0)
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
    fn absent_per_dimension_precision_uses_explicit_fallback() {
        let observation = event(vec![dimension("temperature", Some(0.5), None, None)]);
        let actual = observation
            .evaluate_gaussian_likelihood(&[0.25], 4.0)
            .expect("valid fallback");
        let explicit = event(vec![dimension("temperature", Some(0.5), Some(4.0), None)])
            .evaluate_gaussian_likelihood(&[0.25], 1.0)
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

        let precision_for_missing =
            event(vec![dimension("temperature", None, Some(1.0), None)]);
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
    fn invalid_prediction_and_fallback_precision_are_rejected_for_observed_dims() {
        let observation = event(vec![dimension("temperature", Some(0.5), None, None)]);
        assert_eq!(
            observation.evaluate_gaussian_likelihood(&[f64::NAN], 1.0),
            Err(ObservationEventError::NonFinitePrediction { index: 0 })
        );
        assert_eq!(
            observation.evaluate_gaussian_likelihood(&[0.5], 0.0),
            Err(ObservationEventError::InvalidFallbackPrecision)
        );
    }
}
