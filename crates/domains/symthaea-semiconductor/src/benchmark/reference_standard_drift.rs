// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Time-dependent reference-standard correction prediction for SEMI-BENCH-001.
//!
//! This module evaluates a pre-fitted, source-qualified linear drift packet.
//! It does not fit a drift model, authenticate a reference standard, replace a
//! certificate, or mutate the static calibration gate. It reports a correction
//! and predictive uncertainty at the measurement time, with explicit unknown
//! and rejected states outside the declared model support.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::calibration_requalification::MeasurementTimeBoundsV1;
use super::metrology_qualification_gate::{
    CalibrationCertificateV1, MeasurementEvidenceV1,
};

pub const REFERENCE_DRIFT_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.reference-standard-drift-input.v1";
pub const REFERENCE_DRIFT_MODEL_SCHEMA_V1: &str =
    "luminous.semi-bench.reference-standard-drift-model.v1";
pub const REFERENCE_DRIFT_POLICY_SCHEMA_V1: &str =
    "luminous.semi-bench.reference-standard-drift-policy.v1";
pub const REFERENCE_DRIFT_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.reference-standard-drift-report.v1";
const COVARIANCE_TOLERANCE_V1: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceDriftModelQualificationV1 {
    Qualified,
    Unknown,
    Rejected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceDriftPredictionStatusV1 {
    Qualified,
    Unknown,
    Rejected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceDriftReasonV1 {
    ModelNotQualified,
    ModelQualificationUnknown,
    ModelIdentityMismatch,
    ReferenceCertificateMismatch,
    MeasurementScopeMismatch,
    InvalidModelChronology,
    PredictionOutsideValidityHorizon,
    ExtrapolationLimitExceeded,
    InsufficientHistoryEpochs,
    MissingHoldoutValidationManifest,
    InvalidModelDigest,
    InvalidHistoryManifestDigest,
    InvalidHoldoutManifestDigest,
    InvalidModelParameters,
    InvalidParameterCovariance,
    CorrectionAnchorMismatch,
    PredictionUncertaintyExceedsPolicy,
    MeasurementTimeBoundsInvalid,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReferenceStandardDriftModelV1 {
    pub schema: String,
    pub model_id: String,
    pub reference_standard_id: String,
    pub reference_certificate_sha256: String,
    pub measurand_id: String,
    pub unit: String,
    /// Time origin at which beta_0 is the additive certificate correction.
    pub anchor_time_unix_seconds: i64,
    /// Calibration history used for model fitting.
    pub fit_start_unix_seconds: i64,
    pub fit_end_unix_seconds: i64,
    /// The declared prediction domain, distinct from the fit interval.
    pub prediction_valid_from_unix_seconds: i64,
    pub prediction_valid_until_unix_seconds: i64,
    /// When the model packet became available to this consumer.
    pub model_available_at_unix_seconds: i64,
    pub fit_observation_count: u64,
    pub history_manifest_sha256: String,
    pub holdout_validation_manifest_sha256: Option<String>,
    pub qualification: ReferenceDriftModelQualificationV1,
    /// Additive raw-value correction at the anchor time.
    pub beta_0_correction: f64,
    /// Change in additive correction per day relative to anchor time.
    pub beta_1_correction_per_day: f64,
    /// Covariance of [beta_0, beta_1] in their declared units.
    pub parameter_covariance: [[f64; 2]; 2],
    pub residual_standard_uncertainty: f64,
    pub model_standard_uncertainty: f64,
    /// Digest over every structured model field other than this digest.
    pub model_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReferenceDriftPredictionPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    pub minimum_history_epochs: u64,
    pub maximum_extrapolation_days: f64,
    pub maximum_standard_uncertainty: f64,
    pub required_coverage_factor: f64,
    /// Maximum permitted difference between beta_0 and the certificate correction.
    pub correction_anchor_tolerance: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReferenceDriftPredictionInputV1 {
    pub schema: String,
    pub input_id: String,
    pub measurement: MeasurementEvidenceV1,
    pub calibration_certificate: CalibrationCertificateV1,
    pub measurement_time_bounds: MeasurementTimeBoundsV1,
    /// A knowledge frontier; models not yet available at this time are excluded.
    pub as_known_at_unix_seconds: i64,
    pub model: ReferenceStandardDriftModelV1,
    pub policy: ReferenceDriftPredictionPolicyV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReferenceDriftPredictionReportV1 {
    pub schema: String,
    pub input_id: String,
    pub status: ReferenceDriftPredictionStatusV1,
    pub reasons: Vec<ReferenceDriftReasonV1>,
    pub measurement_event_id: String,
    pub reference_standard_id: String,
    pub model_id: String,
    pub model_sha256: String,
    pub history_manifest_sha256: String,
    pub holdout_validation_manifest_sha256: Option<String>,
    pub as_known_at_unix_seconds: i64,
    pub measurement_time_earliest_unix_seconds: i64,
    pub measurement_time_latest_unix_seconds: i64,
    pub predicted_at_midpoint_unix_seconds: i64,
    pub predicted_additive_correction: Option<f64>,
    pub predicted_corrected_measurement: Option<f64>,
    /// Conservative standard-uncertainty upper bound at the measurement time.
    pub predictive_standard_uncertainty_upper_bound: Option<f64>,
    pub expanded_uncertainty_upper_bound: Option<f64>,
    pub extrapolation_days_upper_bound: Option<f64>,
    /// The model is a declared input; this module does not authenticate its source.
    pub source_authenticity_proven: bool,
    pub calibration_certificate_mutated: bool,
    pub equipment_control_authority: bool,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct ReferenceDriftErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), ReferenceDriftErrorV1> {
    if value.trim().is_empty() {
        return Err(ReferenceDriftErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, ReferenceDriftErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        ReferenceDriftErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn model_digest(model: &ReferenceStandardDriftModelV1) -> Result<String, ReferenceDriftErrorV1> {
    let mut canonical = model.clone();
    canonical.model_sha256.clear();
    json_sha256(&canonical)
}

fn report_digest(report: &ReferenceDriftPredictionReportV1) -> Result<String, ReferenceDriftErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn validate_covariance(covariance: [[f64; 2]; 2]) -> bool {
    let [[a, b], [c, d]] = covariance;
    [a, b, c, d].iter().all(|value| value.is_finite())
        && (b - c).abs() <= COVARIANCE_TOLERANCE_V1
        && a >= -COVARIANCE_TOLERANCE_V1
        && d >= -COVARIANCE_TOLERANCE_V1
        && a * d - b * c >= -COVARIANCE_TOLERANCE_V1
}

fn validate_policy(policy: &ReferenceDriftPredictionPolicyV1) -> Result<(), ReferenceDriftErrorV1> {
    if policy.schema != REFERENCE_DRIFT_POLICY_SCHEMA_V1 {
        return Err(ReferenceDriftErrorV1("policy schema is unsupported".into()));
    }
    require_text(&policy.policy_id, "policy.policy_id")?;
    if policy.minimum_history_epochs < 2
        || !policy.maximum_extrapolation_days.is_finite()
        || policy.maximum_extrapolation_days < 0.0
        || !policy.maximum_standard_uncertainty.is_finite()
        || policy.maximum_standard_uncertainty <= 0.0
        || !policy.required_coverage_factor.is_finite()
        || policy.required_coverage_factor <= 0.0
        || !policy.correction_anchor_tolerance.is_finite()
        || policy.correction_anchor_tolerance < 0.0
    {
        return Err(ReferenceDriftErrorV1(
            "policy contains invalid range, budget or uncertainty parameters".into(),
        ));
    }
    Ok(())
}

fn validate_model_shape(model: &ReferenceStandardDriftModelV1) -> Result<(), ReferenceDriftErrorV1> {
    if model.schema != REFERENCE_DRIFT_MODEL_SCHEMA_V1 {
        return Err(ReferenceDriftErrorV1("model schema is unsupported".into()));
    }
    require_text(&model.model_id, "model.model_id")?;
    require_text(&model.reference_standard_id, "model.reference_standard_id")?;
    require_text(&model.measurand_id, "model.measurand_id")?;
    require_text(&model.unit, "model.unit")?;
    if !valid_sha256(&model.reference_certificate_sha256)
        || !valid_sha256(&model.history_manifest_sha256)
        || model
            .holdout_validation_manifest_sha256
            .as_deref()
            .is_some_and(|digest| !valid_sha256(digest))
    {
        return Err(ReferenceDriftErrorV1(
            "model contains a malformed lowercase SHA-256 digest".into(),
        ));
    }
    if model.fit_start_unix_seconds > model.anchor_time_unix_seconds
        || model.anchor_time_unix_seconds > model.fit_end_unix_seconds
        || model.fit_start_unix_seconds > model.fit_end_unix_seconds
        || model.prediction_valid_from_unix_seconds > model.prediction_valid_until_unix_seconds
        || model.anchor_time_unix_seconds < model.prediction_valid_from_unix_seconds
        || model.anchor_time_unix_seconds > model.prediction_valid_until_unix_seconds
        || model.model_available_at_unix_seconds < model.fit_end_unix_seconds
    {
        return Err(ReferenceDriftErrorV1(
            "model chronology or prediction horizon is inconsistent".into(),
        ));
    }
    if !model.beta_0_correction.is_finite()
        || !model.beta_1_correction_per_day.is_finite()
        || !model.residual_standard_uncertainty.is_finite()
        || model.residual_standard_uncertainty < 0.0
        || !model.model_standard_uncertainty.is_finite()
        || model.model_standard_uncertainty < 0.0
    {
        return Err(ReferenceDriftErrorV1(
            "model parameters and uncertainty terms must be finite and non-negative where required"
                .into(),
        ));
    }
    if !validate_covariance(model.parameter_covariance) {
        return Err(ReferenceDriftErrorV1(
            "model parameter covariance must be symmetric positive semidefinite".into(),
        ));
    }
    if model_digest(model)? != model.model_sha256 {
        return Err(ReferenceDriftErrorV1(
            "model digest does not match the structured model packet".into(),
        ));
    }
    Ok(())
}

fn parameter_variance(model: &ReferenceStandardDriftModelV1, delta_days: f64) -> f64 {
    let [[c00, c01], [c10, c11]] = model.parameter_covariance;
    c00 + delta_days * (c01 + c10) + delta_days * delta_days * c11
}

fn blank_report(
    input: &ReferenceDriftPredictionInputV1,
    status: ReferenceDriftPredictionStatusV1,
    reasons: Vec<ReferenceDriftReasonV1>,
) -> ReferenceDriftPredictionReportV1 {
    ReferenceDriftPredictionReportV1 {
        schema: REFERENCE_DRIFT_REPORT_SCHEMA_V1.into(),
        input_id: input.input_id.clone(),
        status,
        reasons,
        measurement_event_id: input.measurement.measurement_event_id.clone(),
        reference_standard_id: input.model.reference_standard_id.clone(),
        model_id: input.model.model_id.clone(),
        model_sha256: input.model.model_sha256.clone(),
        history_manifest_sha256: input.model.history_manifest_sha256.clone(),
        holdout_validation_manifest_sha256: input.model.holdout_validation_manifest_sha256.clone(),
        as_known_at_unix_seconds: input.as_known_at_unix_seconds,
        measurement_time_earliest_unix_seconds: input.measurement_time_bounds.earliest_measured_at_unix_seconds,
        measurement_time_latest_unix_seconds: input.measurement_time_bounds.latest_measured_at_unix_seconds,
        predicted_at_midpoint_unix_seconds: 0,
        predicted_additive_correction: None,
        predicted_corrected_measurement: None,
        predictive_standard_uncertainty_upper_bound: None,
        expanded_uncertainty_upper_bound: None,
        extrapolation_days_upper_bound: None,
        source_authenticity_proven: false,
        calibration_certificate_mutated: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    }
}

/// Evaluates a declared linear drift model at a measurement-time interval.
///
/// The parameter covariance contributes a time-dependent prediction variance.
/// Residual, model-discrepancy and time-interval bounds are then added linearly
/// as a conservative no-independence upper bound. This function does not fit
/// the model, authenticate its history, or replace the static certificate gate.
pub fn predict_reference_standard_drift_v1(
    input: &ReferenceDriftPredictionInputV1,
) -> Result<ReferenceDriftPredictionReportV1, ReferenceDriftErrorV1> {
    if input.schema != REFERENCE_DRIFT_INPUT_SCHEMA_V1 {
        return Err(ReferenceDriftErrorV1("input schema is unsupported".into()));
    }
    require_text(&input.input_id, "input.input_id")?;
    validate_policy(&input.policy)?;
    validate_model_shape(&input.model)?;

    let lower = input.measurement_time_bounds.earliest_measured_at_unix_seconds;
    let upper = input.measurement_time_bounds.latest_measured_at_unix_seconds;
    if lower > upper
        || input.measurement.measured_at_unix_seconds < lower
        || input.measurement.measured_at_unix_seconds > upper
    {
        return Err(ReferenceDriftErrorV1(
            "measurement time bounds must be ordered and contain measured_at".into(),
        ));
    }

    let mut reasons = Vec::new();
    let mut status = ReferenceDriftPredictionStatusV1::Qualified;

    if input.model.model_available_at_unix_seconds > input.as_known_at_unix_seconds {
        reasons.push(ReferenceDriftReasonV1::ModelQualificationUnknown);
        status = ReferenceDriftPredictionStatusV1::Unknown;
    }
    if input.model.reference_standard_id != input.calibration_certificate.reference_standard_id
        || input
            .calibration_certificate
            .reference_certificate_sha256
            .as_ref()
            != Some(&input.model.reference_certificate_sha256)
    {
        reasons.push(ReferenceDriftReasonV1::ReferenceCertificateMismatch);
        status = ReferenceDriftPredictionStatusV1::Rejected;
    }
    if input.model.measurand_id != input.measurement.measurand_id
        || input.model.unit != input.measurement.unit
    {
        reasons.push(ReferenceDriftReasonV1::MeasurementScopeMismatch);
        status = ReferenceDriftPredictionStatusV1::Rejected;
    }
    if (input.model.beta_0_correction - input.calibration_certificate.correction).abs()
        > input.policy.correction_anchor_tolerance
    {
        reasons.push(ReferenceDriftReasonV1::CorrectionAnchorMismatch);
        status = ReferenceDriftPredictionStatusV1::Rejected;
    }
    if input.model.fit_observation_count < input.policy.minimum_history_epochs {
        reasons.push(ReferenceDriftReasonV1::InsufficientHistoryEpochs);
        if status != ReferenceDriftPredictionStatusV1::Rejected {
            status = ReferenceDriftPredictionStatusV1::Unknown;
        }
    }
    if input.model.holdout_validation_manifest_sha256.is_none() {
        reasons.push(ReferenceDriftReasonV1::MissingHoldoutValidationManifest);
        if status != ReferenceDriftPredictionStatusV1::Rejected {
            status = ReferenceDriftPredictionStatusV1::Unknown;
        }
    }
    match input.model.qualification {
        ReferenceDriftModelQualificationV1::Qualified => {}
        ReferenceDriftModelQualificationV1::Unknown => {
            reasons.push(ReferenceDriftReasonV1::ModelQualificationUnknown);
            if status != ReferenceDriftPredictionStatusV1::Rejected {
                status = ReferenceDriftPredictionStatusV1::Unknown;
            }
        }
        ReferenceDriftModelQualificationV1::Rejected => {
            reasons.push(ReferenceDriftReasonV1::ModelNotQualified);
            status = ReferenceDriftPredictionStatusV1::Rejected;
        }
    }

    if lower < input.model.prediction_valid_from_unix_seconds
        || upper > input.model.prediction_valid_until_unix_seconds
    {
        reasons.push(ReferenceDriftReasonV1::PredictionOutsideValidityHorizon);
        if status != ReferenceDriftPredictionStatusV1::Rejected {
            status = ReferenceDriftPredictionStatusV1::Unknown;
        }
    }

    let midpoint = lower + (upper - lower) / 2;
    let midpoint_days =
        (midpoint - input.model.anchor_time_unix_seconds) as f64 / 86_400.0;
    let lower_days =
        (lower - input.model.anchor_time_unix_seconds) as f64 / 86_400.0;
    let upper_days =
        (upper - input.model.anchor_time_unix_seconds) as f64 / 86_400.0;
    let extrapolation_days = [
        0.0_f64,
        (input.model.fit_start_unix_seconds - lower).max(0) as f64 / 86_400.0,
        (upper - input.model.fit_end_unix_seconds).max(0) as f64 / 86_400.0,
    ]
    .into_iter()
    .fold(0.0_f64, f64::max);

    if extrapolation_days > input.policy.maximum_extrapolation_days {
        reasons.push(ReferenceDriftReasonV1::ExtrapolationLimitExceeded);
        if status != ReferenceDriftPredictionStatusV1::Rejected {
            status = ReferenceDriftPredictionStatusV1::Unknown;
        }
    }

    let predicted_correction =
        input.model.beta_0_correction + input.model.beta_1_correction_per_day * midpoint_days;
    let variance_lower = parameter_variance(&input.model, lower_days);
    let variance_upper = parameter_variance(&input.model, upper_days);
    let max_parameter_variance = variance_lower.max(variance_upper);

    if !predicted_correction.is_finite()
        || !variance_lower.is_finite()
        || !variance_upper.is_finite()
        || max_parameter_variance < -COVARIANCE_TOLERANCE_V1
    {
        reasons.push(ReferenceDriftReasonV1::InvalidModelParameters);
        status = ReferenceDriftPredictionStatusV1::Rejected;
    }

    let parameter_standard_uncertainty =
        max_parameter_variance.max(0.0).sqrt();
    let half_interval_days = (upper - lower) as f64 / (2.0 * 86_400.0);
    let time_interval_uncertainty =
        input.model.beta_1_correction_per_day.abs() * half_interval_days;
    let predictive_uncertainty = parameter_standard_uncertainty
        + input.model.residual_standard_uncertainty
        + input.model.model_standard_uncertainty
        + time_interval_uncertainty;
    let expanded_uncertainty =
        predictive_uncertainty * input.policy.required_coverage_factor;

    if !predictive_uncertainty.is_finite()
        || predictive_uncertainty > input.policy.maximum_standard_uncertainty
    {
        reasons.push(ReferenceDriftReasonV1::PredictionUncertaintyExceedsPolicy);
        if status != ReferenceDriftPredictionStatusV1::Rejected {
            status = ReferenceDriftPredictionStatusV1::Rejected;
        }
    }

    reasons.sort();
    reasons.dedup();

    let mut report = blank_report(input, status, reasons);
    report.predicted_at_midpoint_unix_seconds = midpoint;
    report.predicted_additive_correction = Some(predicted_correction);
    report.predicted_corrected_measurement =
        Some(input.measurement.raw_value + predicted_correction);
    report.predictive_standard_uncertainty_upper_bound = Some(predictive_uncertainty);
    report.expanded_uncertainty_upper_bound = Some(expanded_uncertainty);
    report.extrapolation_days_upper_bound = Some(extrapolation_days);
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

