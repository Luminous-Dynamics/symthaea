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
    CalibrationCertificateV1, MeasurementEvidenceV1, MeasurementQualificationPolicyV1,
    MeasurementQualificationReportV1, MeasurementQualificationStatusV1,
    qualify_measurement_evidence_v1,
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
    BaseMeasurementQualificationUnknown,
    BaseMeasurementQualificationRejected,
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
    pub measurement_qualification_policy: MeasurementQualificationPolicyV1,
    pub measurement_qualification_report: MeasurementQualificationReportV1,
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

/// Seals the exact structured model packet. The digest is content identity,
/// not proof that the model issuer or reference-standard history is authentic.
pub fn seal_reference_standard_drift_model_v1(
    model: &ReferenceStandardDriftModelV1,
) -> Result<String, ReferenceDriftErrorV1> {
    model_digest(model)
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

    let reproduced_qualification = qualify_measurement_evidence_v1(
        &input.measurement,
        Some(&input.calibration_certificate),
        &input.measurement_qualification_policy,
    )
    .map_err(|error| ReferenceDriftErrorV1(error.to_string()))?;
    if reproduced_qualification != input.measurement_qualification_report {
        return Err(ReferenceDriftErrorV1(
            "base calibration qualification receipt does not replay from frozen inputs".into(),
        ));
    }
    if reproduced_qualification.status != MeasurementQualificationStatusV1::Qualified {
        let (status, reason) = match reproduced_qualification.status {
            MeasurementQualificationStatusV1::Unknown => (
                ReferenceDriftPredictionStatusV1::Unknown,
                ReferenceDriftReasonV1::BaseMeasurementQualificationUnknown,
            ),
            MeasurementQualificationStatusV1::Rejected => (
                ReferenceDriftPredictionStatusV1::Rejected,
                ReferenceDriftReasonV1::BaseMeasurementQualificationRejected,
            ),
            MeasurementQualificationStatusV1::Qualified => unreachable!(),
        };
        let mut report = blank_report(input, status, vec![reason]);
        report.report_sha256 = report_digest(&report)?;
        return Ok(report);
    }

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

    let time_span = upper.checked_sub(lower).ok_or_else(|| {
        ReferenceDriftErrorV1("measurement time interval overflows signed seconds".into())
    })?;
    let midpoint = lower.checked_add(time_span / 2).ok_or_else(|| {
        ReferenceDriftErrorV1("measurement time midpoint overflows signed seconds".into())
    })?;
    let midpoint_days = midpoint
        .checked_sub(input.model.anchor_time_unix_seconds)
        .ok_or_else(|| ReferenceDriftErrorV1("midpoint/anchor time delta overflows".into()))?
        as f64
        / 86_400.0;
    let lower_days = lower
        .checked_sub(input.model.anchor_time_unix_seconds)
        .ok_or_else(|| ReferenceDriftErrorV1("lower/anchor time delta overflows".into()))?
        as f64
        / 86_400.0;
    let upper_days = upper
        .checked_sub(input.model.anchor_time_unix_seconds)
        .ok_or_else(|| ReferenceDriftErrorV1("upper/anchor time delta overflows".into()))?
        as f64
        / 86_400.0;
    let days_before_fit = input
        .model
        .fit_start_unix_seconds
        .checked_sub(lower)
        .ok_or_else(|| ReferenceDriftErrorV1("fit-start time delta overflows".into()))?
        .max(0);
    let days_after_fit = upper
        .checked_sub(input.model.fit_end_unix_seconds)
        .ok_or_else(|| ReferenceDriftErrorV1("fit-end time delta overflows".into()))?
        .max(0);
    let extrapolation_days =
        days_before_fit.max(days_after_fit) as f64 / 86_400.0;

    if extrapolation_days > input.policy.maximum_extrapolation_days {
        reasons.push(ReferenceDriftReasonV1::ExtrapolationLimitExceeded);
        if status != ReferenceDriftPredictionStatusV1::Rejected {
            status = ReferenceDriftPredictionStatusV1::Unknown;
        }
    }

    // Never expose a usable correction from a model that is unknown, not yet
    // available at the requested frontier, or outside its qualified support.
    if status != ReferenceDriftPredictionStatusV1::Qualified {
        reasons.sort();
        reasons.dedup();
        let mut report = blank_report(input, status, reasons);
        report.extrapolation_days_upper_bound = Some(extrapolation_days);
        report.report_sha256 = report_digest(&report)?;
        return Ok(report);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::calibration_requalification::MeasurementTimeBoundsV1;
    use crate::benchmark::metrology_qualification_gate::{
        CALIBRATION_CERTIFICATE_SCHEMA_V1, MEASUREMENT_EVIDENCE_SCHEMA_V1,
        QUALIFICATION_POLICY_SCHEMA_V1, MeasurementUncertaintyComponentV1,
        MeasurementQualificationPolicyV1, TraceabilityDeclarationV1,
        UncertaintyComponentKindV1, qualify_measurement_evidence_v1,
    };

    fn measurement() -> MeasurementEvidenceV1 {
        MeasurementEvidenceV1 {
            schema: MEASUREMENT_EVIDENCE_SCHEMA_V1.into(),
            measurement_event_id: "measurement-drift-001".into(),
            source_record_sha256: "c".repeat(64),
            instrument_id: "ellipsometer-07".into(),
            measurand_id: "film-thickness".into(),
            unit: "nm".into(),
            method_revision: "film-method-3".into(),
            measured_at_unix_seconds: 1_850_000_000,
            raw_value: 12.50,
            correction_applied: -0.05,
            corrected_value: 12.45,
            uncertainty_components: vec![MeasurementUncertaintyComponentV1 {
                component_id: "repeatability".into(),
                source_ref: "msa-study-2026-11".into(),
                source_sha256: "d".repeat(64),
                kind: UncertaintyComponentKindV1::TypeA,
                standard_uncertainty: 0.02,
            }],
        }
    }

    fn certificate() -> CalibrationCertificateV1 {
        CalibrationCertificateV1 {
            schema: CALIBRATION_CERTIFICATE_SCHEMA_V1.into(),
            certificate_id: "cal-cert-drift-001".into(),
            certificate_sha256: "a".repeat(64),
            issuing_authority_ref: "metrology-lab-01".into(),
            instrument_id: "ellipsometer-07".into(),
            measurand_id: "film-thickness".into(),
            unit: "nm".into(),
            method_revision: "film-method-3".into(),
            reference_standard_id: "step-height-standard-02".into(),
            reference_certificate_sha256: Some("b".repeat(64)),
            traceability_declaration: TraceabilityDeclarationV1::ClaimedTraceable,
            calibration_performed_at_unix_seconds: 1_800_000_000,
            issued_at_unix_seconds: 1_800_000_100,
            valid_from_unix_seconds: 1_800_000_100,
            valid_until_unix_seconds: 1_900_000_000,
            revoked_at_unix_seconds: None,
            correction: -0.05,
            calibration_standard_uncertainty: 0.01,
            coverage_factor: 2.0,
        }
    }

    fn qualification_policy() -> MeasurementQualificationPolicyV1 {
        MeasurementQualificationPolicyV1 {
            schema: QUALIFICATION_POLICY_SCHEMA_V1.into(),
            policy_id: "strict-static-calibration.v1".into(),
            revision: 1,
            maximum_standard_uncertainty: 0.20,
            required_coverage_factor: 2.0,
            require_claimed_traceability: true,
            require_reference_certificate_digest: true,
        }
    }

    fn model() -> ReferenceStandardDriftModelV1 {
        let mut model = ReferenceStandardDriftModelV1 {
            schema: REFERENCE_DRIFT_MODEL_SCHEMA_V1.into(),
            model_id: "reference-linear-drift-001".into(),
            reference_standard_id: "step-height-standard-02".into(),
            reference_certificate_sha256: "b".repeat(64),
            measurand_id: "film-thickness".into(),
            unit: "nm".into(),
            anchor_time_unix_seconds: 1_840_000_000,
            fit_start_unix_seconds: 1_830_000_000,
            fit_end_unix_seconds: 1_845_000_000,
            prediction_valid_from_unix_seconds: 1_830_000_000,
            prediction_valid_until_unix_seconds: 1_900_000_000,
            model_available_at_unix_seconds: 1_846_000_000,
            fit_observation_count: 8,
            history_manifest_sha256: "d".repeat(64),
            holdout_validation_manifest_sha256: Some("e".repeat(64)),
            qualification: ReferenceDriftModelQualificationV1::Qualified,
            beta_0_correction: -0.05,
            beta_1_correction_per_day: 0.0001,
            parameter_covariance: [[0.0001, 0.0], [0.0, 0.000001]],
            residual_standard_uncertainty: 0.002,
            model_standard_uncertainty: 0.003,
            model_sha256: String::new(),
        };
        model.model_sha256 = seal_reference_standard_drift_model_v1(&model).unwrap();
        model
    }

    fn policy() -> ReferenceDriftPredictionPolicyV1 {
        ReferenceDriftPredictionPolicyV1 {
            schema: REFERENCE_DRIFT_POLICY_SCHEMA_V1.into(),
            policy_id: "reference-time-drift-policy.v1".into(),
            revision: 1,
            minimum_history_epochs: 3,
            maximum_extrapolation_days: 20.0,
            maximum_standard_uncertainty: 0.10,
            required_coverage_factor: 2.0,
            correction_anchor_tolerance: 1.0e-9,
        }
    }

    fn input() -> ReferenceDriftPredictionInputV1 {
        let measured = measurement();
        let cert = certificate();
        let static_policy = qualification_policy();
        let qualification =
            qualify_measurement_evidence_v1(&measured, Some(&cert), &static_policy).unwrap();
        ReferenceDriftPredictionInputV1 {
            schema: REFERENCE_DRIFT_INPUT_SCHEMA_V1.into(),
            input_id: "reference-drift-prediction-input-001".into(),
            measurement: measured,
            calibration_certificate: cert,
            measurement_qualification_policy: static_policy,
            measurement_qualification_report: qualification,
            measurement_time_bounds: MeasurementTimeBoundsV1 {
                earliest_measured_at_unix_seconds: 1_850_000_000,
                latest_measured_at_unix_seconds: 1_850_000_000,
            },
            as_known_at_unix_seconds: 1_860_000_000,
            model: model(),
            policy: policy(),
        }
    }

    fn reseal(input: &mut ReferenceDriftPredictionInputV1) {
        input.model.model_sha256 =
            seal_reference_standard_drift_model_v1(&input.model).unwrap();
    }

    #[test]
    fn linear_drift_updates_correction_at_measurement_time() {
        let report = predict_reference_standard_drift_v1(&input()).unwrap();
        assert_eq!(report.status, ReferenceDriftPredictionStatusV1::Qualified);
        assert!((report.predicted_additive_correction.unwrap() - (-0.049)).abs() < 1.0e-12);
        assert!((report.predicted_corrected_measurement.unwrap() - 12.451).abs() < 1.0e-12);
        assert!(report.expanded_uncertainty_upper_bound.unwrap() > 0.0);
        assert!(!report.source_authenticity_proven);
        assert!(!report.calibration_certificate_mutated);
        assert!(!report.equipment_control_authority);
    }

    #[test]
    fn prediction_uncertainty_includes_covariance_residual_and_model_terms() {
        let report = predict_reference_standard_drift_v1(&input()).unwrap();
        let parameter = (0.0001_f64 + 100.0 * 0.000001).sqrt();
        let expected = parameter + 0.002 + 0.003;
        assert!(
            (report.predictive_standard_uncertainty_upper_bound.unwrap() - expected).abs()
                < 1.0e-12
        );
        assert!(
            (report.expanded_uncertainty_upper_bound.unwrap() - expected * 2.0).abs()
                < 1.0e-12
        );
    }

    #[test]
    fn uncertain_measurement_time_adds_a_conservative_slope_bound() {
        let mut value = input();
        value.measurement_time_bounds = MeasurementTimeBoundsV1 {
            earliest_measured_at_unix_seconds: 1_849_956_800,
            latest_measured_at_unix_seconds: 1_850_043_200,
        };
        let point = predict_reference_standard_drift_v1(&input()).unwrap();
        let interval = predict_reference_standard_drift_v1(&value).unwrap();
        assert!(interval.predictive_standard_uncertainty_upper_bound
            > point.predictive_standard_uncertainty_upper_bound);
    }

    #[test]
    fn model_not_available_at_selected_knowledge_frontier_is_unknown() {
        let mut value = input();
        value.as_known_at_unix_seconds = 1_845_000_000;
        let report = predict_reference_standard_drift_v1(&value).unwrap();
        assert_eq!(report.status, ReferenceDriftPredictionStatusV1::Unknown);
        assert!(
            report
                .reasons
                .contains(&ReferenceDriftReasonV1::ModelQualificationUnknown)
        );
        assert!(report.predicted_additive_correction.is_some());
    }

    #[test]
    fn prediction_outside_declared_horizon_is_unknown() {
        let mut value = input();
        value.model.prediction_valid_until_unix_seconds = 1_849_999_999;
        reseal(&mut value);
        let report = predict_reference_standard_drift_v1(&value).unwrap();
        assert_eq!(report.status, ReferenceDriftPredictionStatusV1::Unknown);
        assert!(
            report
                .reasons
                .contains(&ReferenceDriftReasonV1::PredictionOutsideValidityHorizon)
        );
    }

    #[test]
    fn excessive_extrapolation_is_unknown_even_inside_model_horizon() {
        let mut value = input();
        value.policy.maximum_extrapolation_days = 2.0;
        let report = predict_reference_standard_drift_v1(&value).unwrap();
        assert_eq!(report.status, ReferenceDriftPredictionStatusV1::Unknown);
        assert!(
            report
                .reasons
                .contains(&ReferenceDriftReasonV1::ExtrapolationLimitExceeded)
        );
    }

    #[test]
    fn insufficient_history_and_missing_holdout_leave_model_unknown() {
        let mut too_few = input();
        too_few.model.fit_observation_count = 2;
        reseal(&mut too_few);
        let few_report = predict_reference_standard_drift_v1(&too_few).unwrap();
        assert_eq!(few_report.status, ReferenceDriftPredictionStatusV1::Unknown);
        assert!(
            few_report
                .reasons
                .contains(&ReferenceDriftReasonV1::InsufficientHistoryEpochs)
        );

        let mut no_holdout = input();
        no_holdout.model.holdout_validation_manifest_sha256 = None;
        reseal(&mut no_holdout);
        let no_holdout_report = predict_reference_standard_drift_v1(&no_holdout).unwrap();
        assert_eq!(no_holdout_report.status, ReferenceDriftPredictionStatusV1::Unknown);
        assert!(
            no_holdout_report
                .reasons
                .contains(&ReferenceDriftReasonV1::MissingHoldoutValidationManifest)
        );
    }

    #[test]
    fn certificate_anchor_mismatch_is_rejected() {
        let mut value = input();
        value.model.beta_0_correction = -0.04;
        reseal(&mut value);
        let report = predict_reference_standard_drift_v1(&value).unwrap();
        assert_eq!(report.status, ReferenceDriftPredictionStatusV1::Rejected);
        assert!(
            report
                .reasons
                .contains(&ReferenceDriftReasonV1::CorrectionAnchorMismatch)
        );
    }

    #[test]
    fn reference_standard_or_measurand_identity_mismatch_is_rejected() {
        let mut value = input();
        value.model.reference_standard_id = "different-standard".into();
        reseal(&mut value);
        let report = predict_reference_standard_drift_v1(&value).unwrap();
        assert_eq!(report.status, ReferenceDriftPredictionStatusV1::Rejected);
        assert!(
            report
                .reasons
                .contains(&ReferenceDriftReasonV1::ReferenceCertificateMismatch)
        );
    }

    #[test]
    fn malformed_covariance_is_rejected_before_prediction() {
        let mut value = input();
        value.model.parameter_covariance[0][1] = 0.01;
        reseal(&mut value);
        assert!(predict_reference_standard_drift_v1(&value).is_err());
    }

    #[test]
    fn static_gate_rejection_blocks_dynamic_prediction() {
        let mut value = input();
        value.measurement_qualification_policy.maximum_standard_uncertainty = 0.001;
        value.measurement_qualification_report = qualify_measurement_evidence_v1(
            &value.measurement,
            Some(&value.calibration_certificate),
            &value.measurement_qualification_policy,
        )
        .unwrap();

        let report = predict_reference_standard_drift_v1(&value).unwrap();
        assert_eq!(report.status, ReferenceDriftPredictionStatusV1::Rejected);
        assert!(
            report
                .reasons
                .contains(&ReferenceDriftReasonV1::BaseMeasurementQualificationRejected)
        );
    }

    #[test]
    fn same_frozen_inputs_produce_identical_prediction_receipts() {
        let value = input();
        let a = predict_reference_standard_drift_v1(&value).unwrap();
        let b = predict_reference_standard_drift_v1(&value).unwrap();
        assert_eq!(a.report_sha256, b.report_sha256);
    }
}
