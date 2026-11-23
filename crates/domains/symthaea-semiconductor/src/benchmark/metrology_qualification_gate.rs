// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed calibration, traceability-declaration and uncertainty eligibility.
//!
//! A QUALIFIED result means only that the supplied evidence passes the encoded
//! policy. It does not authenticate source certificates or establish physical
//! traceability/accreditation. Uncertainty components are summed as a
//! fully-correlated upper bound when no validated covariance model is supplied.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

pub const CALIBRATION_CERTIFICATE_SCHEMA_V1: &str =
    "luminous.semi-bench.calibration-certificate.v1";
pub const MEASUREMENT_EVIDENCE_SCHEMA_V1: &str =
    "luminous.semi-bench.measurement-evidence.v1";
pub const QUALIFICATION_POLICY_SCHEMA_V1: &str =
    "luminous.semi-bench.measurement-qualification-policy.v1";
pub const QUALIFICATION_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.measurement-qualification-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceabilityDeclarationV1 {
    ClaimedTraceable,
    Limited,
    Unverified,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UncertaintyComponentKindV1 {
    TypeA,
    TypeB,
    Mixed,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationCertificateV1 {
    pub schema: String,
    pub certificate_id: String,
    /// Digest of the externally issued certificate bytes.
    pub certificate_sha256: String,
    pub issuing_authority_ref: String,
    pub instrument_id: String,
    pub measurand_id: String,
    pub unit: String,
    pub method_revision: String,
    pub reference_standard_id: String,
    /// Digest of the certificate for the reference standard/artifact.
    pub reference_certificate_sha256: Option<String>,
    /// A source declaration, not independent proof of a traceability chain.
    pub traceability_declaration: TraceabilityDeclarationV1,
    pub calibration_performed_at_unix_seconds: i64,
    pub issued_at_unix_seconds: i64,
    pub valid_from_unix_seconds: i64,
    pub valid_until_unix_seconds: i64,
    /// Effective revocation time. A later revocation does not automatically
    /// invalidate measurements made before this instant.
    pub revoked_at_unix_seconds: Option<i64>,
    /// Additive correction: corrected_value = raw_value + correction.
    pub correction: f64,
    /// Standard uncertainty associated with calibration and its reference.
    pub calibration_standard_uncertainty: f64,
    pub coverage_factor: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeasurementUncertaintyComponentV1 {
    pub component_id: String,
    pub source_ref: String,
    pub source_sha256: String,
    pub kind: UncertaintyComponentKindV1,
    /// Standard uncertainty expressed in the measurement's declared unit.
    pub standard_uncertainty: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeasurementEvidenceV1 {
    pub schema: String,
    pub measurement_event_id: String,
    pub source_record_sha256: String,
    pub instrument_id: String,
    pub measurand_id: String,
    pub unit: String,
    pub method_revision: String,
    pub measured_at_unix_seconds: i64,
    pub raw_value: f64,
    pub correction_applied: f64,
    pub corrected_value: f64,
    /// Components beyond the calibration standard uncertainty.
    pub uncertainty_components: Vec<MeasurementUncertaintyComponentV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeasurementQualificationPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    /// Maximum permitted worst-case standard-uncertainty upper bound.
    pub maximum_standard_uncertainty: f64,
    /// Coverage factor used to report expanded uncertainty U = k × u.
    pub required_coverage_factor: f64,
    pub require_claimed_traceability: bool,
    pub require_reference_certificate_digest: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementQualificationStatusV1 {
    Qualified,
    Unknown,
    Rejected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MeasurementQualificationReasonV1 {
    QualifiedAgainstDeclaredPolicy,
    CalibrationCertificateMissing,
    TraceabilityNotEstablished,
    ReferenceCertificateMissing,
    MeasurementIdentityMismatch,
    CalibrationIdentityMismatch,
    CalibrationNotYetPerformed,
    CalibrationNotYetIssued,
    CalibrationNotYetValid,
    CalibrationExpired,
    CalibrationRevokedAtMeasurement,
    InvalidCalibrationChronology,
    InvalidCalibrationCorrection,
    MissingUncertaintyComponents,
    InvalidUncertaintyComponent,
    UncertaintyExceedsPolicy,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeasurementQualificationReportV1 {
    pub schema: String,
    pub status: MeasurementQualificationStatusV1,
    pub reasons: Vec<MeasurementQualificationReasonV1>,
    pub measurement_event_id: String,
    pub measurement_sha256: String,
    pub calibration_certificate_sha256: Option<String>,
    /// Digest of the structured certificate fields used by the gate.
    pub calibration_record_sha256: Option<String>,
    pub policy_sha256: String,
    pub corrected_value: f64,
    pub conservative_standard_uncertainty_upper_bound: Option<f64>,
    pub expanded_uncertainty_upper_bound: Option<f64>,
    /// Explicitly identifies the no-independence assumption used for the bound.
    pub uncertainty_combination_method: String,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct MeasurementQualificationErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), MeasurementQualificationErrorV1> {
    if value.trim().is_empty() {
        return Err(MeasurementQualificationErrorV1(format!(
            "{field}: expected non-empty string"
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

fn json_sha256<T: Serialize + ?Sized>(
    value: &T,
) -> Result<String, MeasurementQualificationErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        MeasurementQualificationErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn measurement_digest(
    measurement: &MeasurementEvidenceV1,
) -> Result<String, MeasurementQualificationErrorV1> {
    let mut material = measurement.clone();
    material.uncertainty_components
        .sort_by(|left, right| left.component_id.cmp(&right.component_id));
    json_sha256(&material)
}

fn validate_policy(
    policy: &MeasurementQualificationPolicyV1,
) -> Result<(), MeasurementQualificationErrorV1> {
    if policy.schema != QUALIFICATION_POLICY_SCHEMA_V1 {
        return Err(MeasurementQualificationErrorV1(
            "policy.schema: unsupported".into(),
        ));
    }
    require_text(&policy.policy_id, "policy.policy_id")?;
    if policy.revision == 0
        || !policy.maximum_standard_uncertainty.is_finite()
        || policy.maximum_standard_uncertainty <= 0.0
        || !policy.required_coverage_factor.is_finite()
        || policy.required_coverage_factor <= 0.0
    {
        return Err(MeasurementQualificationErrorV1(
            "policy requires positive revision, uncertainty limit and coverage factor".into(),
        ));
    }
    Ok(())
}

fn validate_measurement_shape(
    measurement: &MeasurementEvidenceV1,
) -> Result<(), MeasurementQualificationErrorV1> {
    if measurement.schema != MEASUREMENT_EVIDENCE_SCHEMA_V1 {
        return Err(MeasurementQualificationErrorV1(
            "measurement.schema: unsupported".into(),
        ));
    }
    for (field, value) in [
        ("measurement_event_id", measurement.measurement_event_id.as_str()),
        ("instrument_id", measurement.instrument_id.as_str()),
        ("measurand_id", measurement.measurand_id.as_str()),
        ("unit", measurement.unit.as_str()),
        ("method_revision", measurement.method_revision.as_str()),
    ] {
        require_text(value, field)?;
    }
    if !valid_sha256(&measurement.source_record_sha256) {
        return Err(MeasurementQualificationErrorV1(
            "measurement.source_record_sha256: expected lowercase SHA-256".into(),
        ));
    }
    for (field, value) in [
        ("raw_value", measurement.raw_value),
        ("correction_applied", measurement.correction_applied),
        ("corrected_value", measurement.corrected_value),
    ] {
        if !value.is_finite() {
            return Err(MeasurementQualificationErrorV1(format!(
                "measurement.{field}: expected finite value"
            )));
        }
    }
    let expected_corrected = measurement.raw_value + measurement.correction_applied;
    let tolerance = 1.0e-12 * expected_corrected.abs().max(1.0);
    if (measurement.corrected_value - expected_corrected).abs() > tolerance {
        return Err(MeasurementQualificationErrorV1(
            "measurement corrected_value does not equal raw_value + correction_applied".into(),
        ));
    }
    let mut component_ids = BTreeSet::new();
    for (index, component) in measurement.uncertainty_components.iter().enumerate() {
        require_text(&component.component_id, &format!("uncertainty_components[{index}].component_id"))?;
        require_text(&component.source_ref, &format!("uncertainty_components[{index}].source_ref"))?;
        if !valid_sha256(&component.source_sha256)
            || !component.standard_uncertainty.is_finite()
            || component.standard_uncertainty < 0.0
            || !component_ids.insert(component.component_id.as_str())
        {
            return Err(MeasurementQualificationErrorV1(format!(
                "uncertainty_components[{index}]: invalid digest, uncertainty or duplicate ID"
            )));
        }
    }
    Ok(())
}

fn validate_certificate_shape(
    certificate: &CalibrationCertificateV1,
) -> Result<(), MeasurementQualificationErrorV1> {
    if certificate.schema != CALIBRATION_CERTIFICATE_SCHEMA_V1 {
        return Err(MeasurementQualificationErrorV1(
            "calibration.schema: unsupported".into(),
        ));
    }
    for (field, value) in [
        ("certificate_id", certificate.certificate_id.as_str()),
        ("issuing_authority_ref", certificate.issuing_authority_ref.as_str()),
        ("instrument_id", certificate.instrument_id.as_str()),
        ("measurand_id", certificate.measurand_id.as_str()),
        ("unit", certificate.unit.as_str()),
        ("method_revision", certificate.method_revision.as_str()),
        ("reference_standard_id", certificate.reference_standard_id.as_str()),
    ] {
        require_text(value, &format!("calibration.{field}"))?;
    }
    if !valid_sha256(&certificate.certificate_sha256)
        || certificate
            .reference_certificate_sha256
            .as_ref()
            .is_some_and(|digest| !valid_sha256(digest))
    {
        return Err(MeasurementQualificationErrorV1(
            "calibration certificate and any reference certificate digest must be lowercase SHA-256"
                .into(),
        ));
    }
    if !certificate.correction.is_finite()
        || !certificate.calibration_standard_uncertainty.is_finite()
        || certificate.calibration_standard_uncertainty < 0.0
        || !certificate.coverage_factor.is_finite()
        || certificate.coverage_factor <= 0.0
    {
        return Err(MeasurementQualificationErrorV1(
            "calibration correction and uncertainty values are invalid".into(),
        ));
    }
    Ok(())
}

fn report_digest(
    report: &MeasurementQualificationReportV1,
) -> Result<String, MeasurementQualificationErrorV1> {
    let mut material = report.clone();
    material.report_sha256.clear();
    json_sha256(&material)
}

/// Assesses whether a measurement is eligible for a downstream evidence update.
/// Missing or unverified evidence yields UNKNOWN; contradictions or expired
/// or revoked certificates yield REJECTED. Neither status grants control authority.
pub fn qualify_measurement_evidence_v1(
    measurement: &MeasurementEvidenceV1,
    calibration: Option<&CalibrationCertificateV1>,
    policy: &MeasurementQualificationPolicyV1,
) -> Result<MeasurementQualificationReportV1, MeasurementQualificationErrorV1> {
    validate_policy(policy)?;
    validate_measurement_shape(measurement)?;
    let measurement_sha256 = measurement_digest(measurement)?;
    let policy_sha256 = json_sha256(policy)?;
    let mut reasons = Vec::new();
    let mut status = MeasurementQualificationStatusV1::Qualified;

    let Some(certificate) = calibration else {
        reasons.push(MeasurementQualificationReasonV1::CalibrationCertificateMissing);
        status = MeasurementQualificationStatusV1::Unknown;
        return finish_report(
            measurement,
            None,
            None,
            policy_sha256,
            measurement_sha256,
            status,
            reasons,
            None,
            policy.required_coverage_factor,
        );
    };

    validate_certificate_shape(certificate)?;
    let certificate_digest = certificate.certificate_sha256.clone();
    let certificate_record_digest = json_sha256(certificate)?;

    if certificate.instrument_id != measurement.instrument_id
        || certificate.measurand_id != measurement.measurand_id
        || certificate.unit != measurement.unit
        || certificate.method_revision != measurement.method_revision
    {
        reasons.push(MeasurementQualificationReasonV1::CalibrationIdentityMismatch);
        status = MeasurementQualificationStatusV1::Rejected;
    }

    if certificate.calibration_performed_at_unix_seconds > certificate.issued_at_unix_seconds
        || certificate.issued_at_unix_seconds > certificate.valid_from_unix_seconds
        || certificate.valid_from_unix_seconds >= certificate.valid_until_unix_seconds
        || certificate.calibration_performed_at_unix_seconds
            > certificate.valid_until_unix_seconds
    {
        reasons.push(MeasurementQualificationReasonV1::InvalidCalibrationChronology);
        status = MeasurementQualificationStatusV1::Rejected;
    }

    if certificate.calibration_performed_at_unix_seconds
        > measurement.measured_at_unix_seconds
    {
        reasons.push(MeasurementQualificationReasonV1::CalibrationNotYetPerformed);
        status = MeasurementQualificationStatusV1::Rejected;
    }
    if certificate.issued_at_unix_seconds > measurement.measured_at_unix_seconds {
        reasons.push(MeasurementQualificationReasonV1::CalibrationNotYetIssued);
        status = MeasurementQualificationStatusV1::Rejected;
    }
    if measurement.measured_at_unix_seconds < certificate.valid_from_unix_seconds {
        reasons.push(MeasurementQualificationReasonV1::CalibrationNotYetValid);
        status = MeasurementQualificationStatusV1::Rejected;
    }
    if measurement.measured_at_unix_seconds > certificate.valid_until_unix_seconds {
        reasons.push(MeasurementQualificationReasonV1::CalibrationExpired);
        status = MeasurementQualificationStatusV1::Rejected;
    }
    if certificate
        .revoked_at_unix_seconds
        .is_some_and(|revoked_at| revoked_at <= measurement.measured_at_unix_seconds)
    {
        reasons.push(MeasurementQualificationReasonV1::CalibrationRevokedAtMeasurement);
        status = MeasurementQualificationStatusV1::Rejected;
    }

    if measurement.correction_applied != certificate.correction {
        reasons.push(MeasurementQualificationReasonV1::InvalidCalibrationCorrection);
        status = MeasurementQualificationStatusV1::Rejected;
    }

    if policy.require_claimed_traceability
        && certificate.traceability_declaration != TraceabilityDeclarationV1::ClaimedTraceable
    {
        reasons.push(MeasurementQualificationReasonV1::TraceabilityNotEstablished);
        if status == MeasurementQualificationStatusV1::Qualified {
            status = MeasurementQualificationStatusV1::Unknown;
        }
    }
    if policy.require_reference_certificate_digest
        && certificate
            .reference_certificate_sha256
            .as_ref()
            .is_none_or(|digest| !valid_sha256(digest))
    {
        reasons.push(MeasurementQualificationReasonV1::ReferenceCertificateMissing);
        if status == MeasurementQualificationStatusV1::Qualified {
            status = MeasurementQualificationStatusV1::Unknown;
        }
    }

    if measurement.uncertainty_components.is_empty() {
        reasons.push(MeasurementQualificationReasonV1::MissingUncertaintyComponents);
        if status == MeasurementQualificationStatusV1::Qualified {
            status = MeasurementQualificationStatusV1::Unknown;
        }
        return finish_report(
            measurement,
            Some(certificate_digest),
            Some(certificate_record_digest),
            policy_sha256,
            measurement_sha256,
            status,
            reasons,
            None,
            policy.required_coverage_factor,
        );
    }

    // Without a validated covariance model, the sum of standard uncertainties
    // is a conservative upper bound valid even under perfect positive correlation.
    let uncertainty_upper_bound = certificate.calibration_standard_uncertainty
        + measurement
            .uncertainty_components
            .iter()
            .map(|component| component.standard_uncertainty)
            .sum::<f64>();
    if !uncertainty_upper_bound.is_finite() {
        reasons.push(MeasurementQualificationReasonV1::InvalidUncertaintyComponent);
        status = MeasurementQualificationStatusV1::Rejected;
    } else if uncertainty_upper_bound > policy.maximum_standard_uncertainty {
        reasons.push(MeasurementQualificationReasonV1::UncertaintyExceedsPolicy);
        status = MeasurementQualificationStatusV1::Rejected;
    }

    finish_report(
        measurement,
        Some(certificate_digest),
        Some(certificate_record_digest),
        policy_sha256,
        measurement_sha256,
        status,
        reasons,
        Some(uncertainty_upper_bound),
        policy.required_coverage_factor,
    )
}

fn finish_report(
    measurement: &MeasurementEvidenceV1,
    calibration_certificate_sha256: Option<String>,
    calibration_record_sha256: Option<String>,
    policy_sha256: String,
    measurement_sha256: String,
    status: MeasurementQualificationStatusV1,
    mut reasons: Vec<MeasurementQualificationReasonV1>,
    uncertainty_upper_bound: Option<f64>,
    coverage_factor: f64,
) -> Result<MeasurementQualificationReportV1, MeasurementQualificationErrorV1> {
    if reasons.is_empty() {
        reasons.push(MeasurementQualificationReasonV1::QualifiedAgainstDeclaredPolicy);
    }
    reasons.sort_by_key(|reason| *reason as u8);
    let mut report = MeasurementQualificationReportV1 {
        schema: QUALIFICATION_REPORT_SCHEMA_V1.into(),
        status,
        reasons,
        measurement_event_id: measurement.measurement_event_id.clone(),
        measurement_sha256,
        calibration_certificate_sha256,
        calibration_record_sha256,
        policy_sha256,
        corrected_value: measurement.corrected_value,
        conservative_standard_uncertainty_upper_bound: uncertainty_upper_bound,
        expanded_uncertainty_upper_bound: uncertainty_upper_bound.map(|value| value * coverage_factor),
        uncertainty_combination_method:
            "fully_correlated_upper_bound_sum; no independence assumption".into(),
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> MeasurementQualificationPolicyV1 {
        MeasurementQualificationPolicyV1 {
            schema: QUALIFICATION_POLICY_SCHEMA_V1.into(),
            policy_id: "strict-metrology-evidence.v1".into(),
            revision: 1,
            maximum_standard_uncertainty: 0.20,
            required_coverage_factor: 2.0,
            require_claimed_traceability: true,
            require_reference_certificate_digest: true,
        }
    }

    fn certificate() -> CalibrationCertificateV1 {
        CalibrationCertificateV1 {
            schema: CALIBRATION_CERTIFICATE_SCHEMA_V1.into(),
            certificate_id: "cal-cert-001".into(),
            certificate_sha256: "a".repeat(64),
            issuing_authority_ref: "metrology-lab-01".into(),
            instrument_id: "ellipsometer-07".into(),
            measurand_id: "film-thickness".into(),
            unit: "nm".into(),
            method_revision: "film-method-3".into(),
            reference_standard_id: "traceable-step-height-std-02".into(),
            reference_certificate_sha256: Some("b".repeat(64)),
            traceability_declaration: TraceabilityDeclarationV1::ClaimedTraceable,
            calibration_performed_at_unix_seconds: 1_800_000_000,
            issued_at_unix_seconds: 1_800_000_100,
            valid_from_unix_seconds: 1_800_000_100,
            valid_until_unix_seconds: 1_900_000_000,
            revoked_at_unix_seconds: None,
            correction: -0.05,
            calibration_standard_uncertainty: 0.04,
            coverage_factor: 2.0,
        }
    }

    fn measurement() -> MeasurementEvidenceV1 {
        MeasurementEvidenceV1 {
            schema: MEASUREMENT_EVIDENCE_SCHEMA_V1.into(),
            measurement_event_id: "measurement-event-001".into(),
            source_record_sha256: "c".repeat(64),
            instrument_id: "ellipsometer-07".into(),
            measurand_id: "film-thickness".into(),
            unit: "nm".into(),
            method_revision: "film-method-3".into(),
            measured_at_unix_seconds: 1_850_000_000,
            raw_value: 12.50,
            correction_applied: -0.05,
            corrected_value: 12.45,
            uncertainty_components: vec![
                MeasurementUncertaintyComponentV1 {
                    component_id: "repeatability".into(),
                    source_ref: "msa-study-2026-11".into(),
                    source_sha256: "d".repeat(64),
                    kind: UncertaintyComponentKindV1::TypeA,
                    standard_uncertainty: 0.03,
                },
                MeasurementUncertaintyComponentV1 {
                    component_id: "resolution".into(),
                    source_ref: "instrument-spec-rev-7".into(),
                    source_sha256: "e".repeat(64),
                    kind: UncertaintyComponentKindV1::TypeB,
                    standard_uncertainty: 0.02,
                },
            ],
        }
    }

    #[test]
    fn valid_measurement_qualifies_with_conservative_uncertainty_bound() {
        let report =
            qualify_measurement_evidence_v1(&measurement(), Some(&certificate()), &policy())
                .unwrap();
        assert_eq!(report.status, MeasurementQualificationStatusV1::Qualified);
        assert_eq!(
            report.conservative_standard_uncertainty_upper_bound,
            Some(0.09)
        );
        assert_eq!(report.expanded_uncertainty_upper_bound, Some(0.18));
        assert_eq!(
            report.uncertainty_combination_method,
            "fully_correlated_upper_bound_sum; no independence assumption"
        );
    }

    #[test]
    fn missing_calibration_returns_unknown_not_qualified() {
        let report =
            qualify_measurement_evidence_v1(&measurement(), None, &policy()).unwrap();
        assert_eq!(report.status, MeasurementQualificationStatusV1::Unknown);
        assert!(
            report
                .reasons
                .contains(&MeasurementQualificationReasonV1::CalibrationCertificateMissing)
        );
    }

    #[test]
    fn instrument_mismatch_is_rejected_even_when_certificate_digest_is_valid() {
        let mut measured = measurement();
        measured.instrument_id = "different-instrument".into();
        let report =
            qualify_measurement_evidence_v1(&measured, Some(&certificate()), &policy()).unwrap();
        assert_eq!(report.status, MeasurementQualificationStatusV1::Rejected);
        assert!(
            report
                .reasons
                .contains(&MeasurementQualificationReasonV1::CalibrationIdentityMismatch)
        );
    }

    #[test]
    fn expired_or_revoked_calibration_is_rejected() {
        let mut measured = measurement();
        measured.measured_at_unix_seconds = 1_910_000_000;
        let expired =
            qualify_measurement_evidence_v1(&measured, Some(&certificate()), &policy()).unwrap();
        assert_eq!(expired.status, MeasurementQualificationStatusV1::Rejected);
        assert!(
            expired
                .reasons
                .contains(&MeasurementQualificationReasonV1::CalibrationExpired)
        );

        let mut revoked_certificate = certificate();
        revoked_certificate.revoked_at_unix_seconds = Some(1_840_000_000);
        let revoked =
            qualify_measurement_evidence_v1(&measurement(), Some(&revoked_certificate), &policy())
                .unwrap();
        assert_eq!(revoked.status, MeasurementQualificationStatusV1::Rejected);
        assert!(
            revoked
                .reasons
                .contains(&MeasurementQualificationReasonV1::CalibrationRevokedAtMeasurement)
        );
    }

    #[test]
    fn unknown_traceability_remains_unknown_and_blocks_promotion() {
        let mut unverified = certificate();
        unverified.traceability_declaration = TraceabilityDeclarationV1::Unverified;
        let report =
            qualify_measurement_evidence_v1(&measurement(), Some(&unverified), &policy()).unwrap();
        assert_eq!(report.status, MeasurementQualificationStatusV1::Unknown);
        assert!(
            report
                .reasons
                .contains(&MeasurementQualificationReasonV1::TraceabilityNotEstablished)
        );
    }

    #[test]
    fn conservative_uncertainty_bound_detects_unmodeled_correlation_risk() {
        let mut measured = measurement();
        measured.uncertainty_components[0].standard_uncertainty = 0.09;
        measured.uncertainty_components[1].standard_uncertainty = 0.09;
        let mut strict_policy = policy();
        strict_policy.maximum_standard_uncertainty = 0.20;
        let report =
            qualify_measurement_evidence_v1(&measured, Some(&certificate()), &strict_policy)
                .unwrap();
        // RSS would be below the limit, but the no-independence upper bound is
        // 0.04 + 0.09 + 0.09 = 0.22 and must fail closed.
        assert_eq!(report.status, MeasurementQualificationStatusV1::Rejected);
        assert_eq!(
            report.conservative_standard_uncertainty_upper_bound,
            Some(0.22)
        );
        assert!(
            report
                .reasons
                .contains(&MeasurementQualificationReasonV1::UncertaintyExceedsPolicy)
        );
    }

    #[test]
    fn reordered_uncertainty_components_produce_identical_evidence_digest() {
        let mut first = measurement();
        let mut second = measurement();
        second.uncertainty_components.reverse();
        let first_report =
            qualify_measurement_evidence_v1(&first, Some(&certificate()), &policy()).unwrap();
        let second_report =
            qualify_measurement_evidence_v1(&second, Some(&certificate()), &policy()).unwrap();
        assert_eq!(first_report.measurement_sha256, second_report.measurement_sha256);
        assert_eq!(first_report.report_sha256, second_report.report_sha256);
        // Retain mutability to ensure the test is about canonical ordering only.
        first.uncertainty_components.swap(0, 1);
    }

    #[test]
    fn mutated_correction_cannot_be_smuggled_into_a_measurement() {
        let mut measured = measurement();
        measured.correction_applied = 0.2;
        measured.corrected_value = 12.7;
        let report =
            qualify_measurement_evidence_v1(&measured, Some(&certificate()), &policy()).unwrap();
        assert_eq!(report.status, MeasurementQualificationStatusV1::Rejected);
        assert!(
            report
                .reasons
                .contains(&MeasurementQualificationReasonV1::InvalidCalibrationCorrection)
        );
    }

    #[test]
    fn missing_reference_certificate_preserves_unknown_status() {
        let mut cert = certificate();
        cert.reference_certificate_sha256 = None;
        let report =
            qualify_measurement_evidence_v1(&measurement(), Some(&cert), &policy()).unwrap();
        assert_eq!(report.status, MeasurementQualificationStatusV1::Unknown);
        assert!(
            report
                .reasons
                .contains(&MeasurementQualificationReasonV1::ReferenceCertificateMissing)
        );
        assert!(report.calibration_record_sha256.is_some());
    }

    #[test]
    fn structured_certificate_mutation_changes_record_digest() {
        let measured = measurement();
        let mut first_certificate = certificate();
        let mut second_certificate = certificate();
        second_certificate.calibration_standard_uncertainty = 0.05;

        let first = qualify_measurement_evidence_v1(
            &measured,
            Some(&first_certificate),
            &policy(),
        )
        .unwrap();
        let second = qualify_measurement_evidence_v1(
            &measured,
            Some(&second_certificate),
            &policy(),
        )
        .unwrap();

        assert_eq!(
            first.calibration_certificate_sha256,
            second.calibration_certificate_sha256
        );
        assert_ne!(
            first.calibration_record_sha256,
            second.calibration_record_sha256
        );
        assert_ne!(first.report_sha256, second.report_sha256);
    }

    #[test]
    fn missing_uncertainty_components_is_unknown() {
        let mut measured = measurement();
        measured.uncertainty_components.clear();
        let report =
            qualify_measurement_evidence_v1(&measured, Some(&certificate()), &policy()).unwrap();
        assert_eq!(report.status, MeasurementQualificationStatusV1::Unknown);
        assert!(
            report
                .reasons
                .contains(&MeasurementQualificationReasonV1::MissingUncertaintyComponents)
        );
    }
}
