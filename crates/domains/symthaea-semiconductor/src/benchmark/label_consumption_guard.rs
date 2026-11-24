// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only eligibility gate for consuming validated metrology labels.
//!
//! Composes process-event labels, measurement qualification and bitemporal
//! calibration requalification. It does not promote labels, mutate training
//! snapshots, authenticate issuers, retrain a VM, or authorize equipment effects.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use super::calibration_requalification::{
    CalibrationRequalificationInputV1, requalify_calibration_measurement_v1,
};
use super::metrology_qualification_gate::{
    MeasurementQualificationStatusV1, qualify_measurement_evidence_v1,
};
use super::process_event_graph::{MetrologyLabelStatusV1, MetrologyLabelV1};

pub const LABEL_CONSUMPTION_GUARD_SCHEMA_V1: &str =
    "luminous.semi-bench.label-consumption-guard.v1";
pub const LABEL_CONSUMPTION_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.label-consumption-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelEvidenceFrontierV1 {
    AsKnownAtK0,
    CurrentAtK1,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelConsumptionEligibilityV1 {
    EligibleUnderDeclaredPolicy,
    Unknown,
    Blocked,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LabelConsumptionReasonV1 {
    LabelNotValidated,
    LabelInvalid,
    LabelStatusUnknown,
    LabelInstrumentMissing,
    LabelInstrumentMismatch,
    LabelMeasurandMismatch,
    LabelUnitMismatch,
    LabelValueMissing,
    LabelValueMismatch,
    LabelUncertaintyMissing,
    LabelUncertaintyNonFinite,
    LabelUncertaintyUnderstatesQualifiedBound,
    QualifiedUncertaintyBoundMissing,
    LabelCalibrationReferenceMissing,
    LabelCalibrationReferenceMismatch,
    MeasurementEventMismatch,
    OriginalQualificationNotQualified,
    CalibrationEligibilityUnknown,
    CalibrationRejected,
    EventInventoryOrOrderingUnknown,
    RelevantCalibrationEventNotMaterialized,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LabelConsumptionEligibilityReportV1 {
    pub schema: String,
    pub label_id: String,
    pub label_revision: u64,
    pub measurement_event_id: String,
    pub calibration_certificate_id: String,
    pub frontier: LabelEvidenceFrontierV1,
    pub eligibility: LabelConsumptionEligibilityV1,
    /// Eligibility only; this is not authority to mutate a model or training set.
    pub downstream_consumption_allowed: bool,
    pub reasons: Vec<LabelConsumptionReasonV1>,
    pub label_sha256: String,
    pub measurement_sha256: String,
    pub qualified_corrected_value: f64,
    pub qualified_standard_uncertainty_upper_bound: Option<f64>,
    pub qualification_report_sha256: String,
    pub requalification_report_sha256: String,
    pub requalification_frontier_sha256: String,
    /// Content digests and declared references do not prove source authenticity.
    pub source_authenticity_proven: bool,
    pub training_snapshot_mutated: bool,
    pub equipment_control_authority: bool,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct LabelConsumptionGuardErrorV1(pub String);

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, LabelConsumptionGuardErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        LabelConsumptionGuardErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn report_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, LabelConsumptionGuardErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn map_unknown_calibration_reasons(
    reasons: &[super::calibration_requalification::CalibrationRequalificationReasonV1],
) -> Vec<LabelConsumptionReasonV1> {
    use super::calibration_requalification::CalibrationRequalificationReasonV1 as Source;
    let mut mapped = Vec::new();
    for reason in reasons {
        match reason {
            Source::EventInventoryIncomplete
            | Source::EventOrderingUnknown
            | Source::UnknownEventScope
            | Source::MeasurementTimeBoundaryUncertain
            | Source::EventSourceNotQualified => {
                mapped.push(LabelConsumptionReasonV1::EventInventoryOrOrderingUnknown);
            }
            Source::RelevantEventNotMaterialized => {
                mapped.push(LabelConsumptionReasonV1::RelevantCalibrationEventNotMaterialized);
            }
            Source::NoApplicableChange
            | Source::OriginalQualificationNotQualified
            | Source::CalibrationRevoked
            | Source::CalibrationSuperseded
            | Source::CalibrationCorrected
            | Source::CalibrationScopeChanged
            | Source::CalibrationReinstatedRequiresNewEvidence => {}
        }
    }
    mapped.sort();
    mapped.dedup();
    mapped
}

/// Assess whether one process-event metrology label may be consumed at K0 or K1.
///
/// The guard replays both the original measurement qualification and the
/// bitemporal calibration assessment from their frozen inputs. It then binds
/// the process-event label to the exact measurement event, instrument,
/// measurand, unit and certificate ID. A positive result means only that this
/// declared software policy permits consumption; it is not an issuer-authenticity
/// proof, model update, physical disposition or control authorization.
pub fn assess_label_consumption_eligibility_v1(
    label: &MetrologyLabelV1,
    input: &CalibrationRequalificationInputV1,
    frontier: LabelEvidenceFrontierV1,
) -> Result<LabelConsumptionEligibilityReportV1, LabelConsumptionGuardErrorV1> {
    let measurement = &input.measurement;
    let certificate = &input.calibration_at_k0;

    let measurement_sha256 = json_sha256(measurement)?;
    if label.value.is_some_and(|value| !value.is_finite())
        || label
            .standard_uncertainty
            .is_some_and(|value| !value.is_finite() || value < 0.0)
        || label
            .censoring_lower_bound
            .is_some_and(|value| !value.is_finite())
        || label
            .censoring_upper_bound
            .is_some_and(|value| !value.is_finite())
    {
        return Err(LabelConsumptionGuardErrorV1(
            "label contains non-finite numeric value or uncertainty metadata".into(),
        ));
    }
    let label_sha256 = json_sha256(label)?;
    let qualification = qualify_measurement_evidence_v1(
        measurement,
        Some(certificate),
        &input.qualification_policy,
    )
    .map_err(|error| LabelConsumptionGuardErrorV1(error.to_string()))?;
    if qualification != input.original_qualification_at_k0 {
        return Err(LabelConsumptionGuardErrorV1(
            "original qualification receipt does not replay from frozen measurement/certificate/policy"
                .into(),
        ));
    }

    let requalification = requalify_calibration_measurement_v1(input)
        .map_err(|error| LabelConsumptionGuardErrorV1(error.to_string()))?;

    let mut reasons = Vec::new();
    let mut eligibility = LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy;

    if label.measurement_event_id != measurement.measurement_event_id
        || label.measurement_event_id != requalification.measurement_event_id
    {
        reasons.push(LabelConsumptionReasonV1::MeasurementEventMismatch);
        eligibility = LabelConsumptionEligibilityV1::Blocked;
    }

    match label.instrument_id.as_deref() {
        Some(value) if value == measurement.instrument_id => {}
        Some(_) => {
            reasons.push(LabelConsumptionReasonV1::LabelInstrumentMismatch);
            eligibility = LabelConsumptionEligibilityV1::Blocked;
        }
        None => {
            reasons.push(LabelConsumptionReasonV1::LabelInstrumentMissing);
            if eligibility != LabelConsumptionEligibilityV1::Blocked {
                eligibility = LabelConsumptionEligibilityV1::Unknown;
            }
        }
    }

    if label.measurand_id != measurement.measurand_id {
        reasons.push(LabelConsumptionReasonV1::LabelMeasurandMismatch);
        eligibility = LabelConsumptionEligibilityV1::Blocked;
    }
    if label.unit != measurement.unit {
        reasons.push(LabelConsumptionReasonV1::LabelUnitMismatch);
        eligibility = LabelConsumptionEligibilityV1::Blocked;
    }

    if label.status == MetrologyLabelStatusV1::Validated {
        match label.value {
            Some(value) if value == qualification.corrected_value => {}
            Some(_) => {
                reasons.push(LabelConsumptionReasonV1::LabelValueMismatch);
                eligibility = LabelConsumptionEligibilityV1::Blocked;
            }
            None => {
                reasons.push(LabelConsumptionReasonV1::LabelValueMissing);
                eligibility = LabelConsumptionEligibilityV1::Blocked;
            }
        }

        match (
            label.standard_uncertainty,
            qualification.conservative_standard_uncertainty_upper_bound,
        ) {
            (Some(label_uncertainty), Some(qualified_bound))
                if label_uncertainty >= qualified_bound => {}
            (Some(_), Some(_)) => {
                reasons.push(LabelConsumptionReasonV1::LabelUncertaintyUnderstatesQualifiedBound);
                eligibility = LabelConsumptionEligibilityV1::Blocked;
            }
            (None, _) => {
                reasons.push(LabelConsumptionReasonV1::LabelUncertaintyMissing);
                if eligibility != LabelConsumptionEligibilityV1::Blocked {
                    eligibility = LabelConsumptionEligibilityV1::Unknown;
                }
            }
            (Some(_), None) => {
                reasons.push(LabelConsumptionReasonV1::QualifiedUncertaintyBoundMissing);
                if eligibility != LabelConsumptionEligibilityV1::Blocked {
                    eligibility = LabelConsumptionEligibilityV1::Unknown;
                }
            }
        }
    }

    match label.calibration_ref.as_deref() {
        Some(value) if value == certificate.certificate_id => {}
        Some(_) => {
            reasons.push(LabelConsumptionReasonV1::LabelCalibrationReferenceMismatch);
            eligibility = LabelConsumptionEligibilityV1::Blocked;
        }
        None => {
            reasons.push(LabelConsumptionReasonV1::LabelCalibrationReferenceMissing);
            if eligibility != LabelConsumptionEligibilityV1::Blocked {
                eligibility = LabelConsumptionEligibilityV1::Unknown;
            }
        }
    }

    match label.status {
        MetrologyLabelStatusV1::Validated => {}
        MetrologyLabelStatusV1::Provisional | MetrologyLabelStatusV1::Censored => {
            reasons.push(LabelConsumptionReasonV1::LabelNotValidated);
            if eligibility != LabelConsumptionEligibilityV1::Blocked {
                eligibility = LabelConsumptionEligibilityV1::Unknown;
            }
        }
        MetrologyLabelStatusV1::Invalid => {
            reasons.push(LabelConsumptionReasonV1::LabelInvalid);
            eligibility = LabelConsumptionEligibilityV1::Blocked;
        }
        MetrologyLabelStatusV1::Unknown => {
            reasons.push(LabelConsumptionReasonV1::LabelStatusUnknown);
            if eligibility != LabelConsumptionEligibilityV1::Blocked {
                eligibility = LabelConsumptionEligibilityV1::Unknown;
            }
        }
    }

    if input.original_qualification_at_k0.status != MeasurementQualificationStatusV1::Qualified {
        reasons.push(LabelConsumptionReasonV1::OriginalQualificationNotQualified);
        eligibility = match input.original_qualification_at_k0.status {
            MeasurementQualificationStatusV1::Qualified => eligibility,
            MeasurementQualificationStatusV1::Unknown => LabelConsumptionEligibilityV1::Unknown,
            MeasurementQualificationStatusV1::Rejected => LabelConsumptionEligibilityV1::Blocked,
        };
    }

    let qualified_frontier = match frontier {
        LabelEvidenceFrontierV1::AsKnownAtK0 => &requalification.as_known_at_k0,
        LabelEvidenceFrontierV1::CurrentAtK1 => &requalification.current_at_k1,
    };

    if qualified_frontier.status != MeasurementQualificationStatusV1::Qualified
        || !qualified_frontier.downstream_consumption_allowed_at_frontier
    {
        match qualified_frontier.status {
            MeasurementQualificationStatusV1::Rejected => {
                reasons.push(LabelConsumptionReasonV1::CalibrationRejected);
                eligibility = LabelConsumptionEligibilityV1::Blocked;
            }
            MeasurementQualificationStatusV1::Unknown => {
                reasons.extend(map_unknown_calibration_reasons(
                    &qualified_frontier.reason_codes,
                ));
                reasons.push(LabelConsumptionReasonV1::CalibrationEligibilityUnknown);
                if eligibility != LabelConsumptionEligibilityV1::Blocked {
                    eligibility = LabelConsumptionEligibilityV1::Unknown;
                }
            }
            MeasurementQualificationStatusV1::Qualified => {
                reasons.push(LabelConsumptionReasonV1::CalibrationEligibilityUnknown);
                if eligibility != LabelConsumptionEligibilityV1::Blocked {
                    eligibility = LabelConsumptionEligibilityV1::Unknown;
                }
            }
        }
    }

    reasons.sort();
    reasons.dedup();

    let mut report = LabelConsumptionEligibilityReportV1 {
        schema: LABEL_CONSUMPTION_REPORT_SCHEMA_V1.into(),
        label_id: label.label_id.clone(),
        measurement_event_id: measurement.measurement_event_id.clone(),
        calibration_certificate_id: certificate.certificate_id.clone(),
        frontier,
        eligibility,
        downstream_consumption_allowed: eligibility
            == LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        reasons,
        label_sha256,
        measurement_sha256,
        qualified_corrected_value: qualification.corrected_value,
        qualified_standard_uncertainty_upper_bound: qualification
            .conservative_standard_uncertainty_upper_bound,
        qualification_report_sha256: qualification.report_sha256,
        requalification_report_sha256: requalification.report_sha256,
        requalification_frontier_sha256: qualified_frontier.frontier_input_sha256.clone(),
        source_authenticity_proven: requalification.source_authenticity_proven,
        training_snapshot_mutated: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::calibration_requalification::{
        CALIBRATION_CHANGE_EVENT_SCHEMA_V1, CALIBRATION_REQUALIFICATION_INPUT_SCHEMA_V1,
        CalibrationChangeEventV1, CalibrationChangeKindV1, CalibrationChangeScopeV1,
        CalibrationChangeTargetV1, CalibrationEventFrontierV1, CalibrationEventInventoryStateV1,
        CalibrationEventOrderingStateV1, CalibrationEventQualificationV1,
        CalibrationOperatingRangeV1, CalibrationRequalificationInputV1, MeasurementTimeBoundsV1,
        seal_calibration_event_inventory_v1,
    };
    use crate::benchmark::metrology_qualification_gate::{
        CALIBRATION_CERTIFICATE_SCHEMA_V1, CalibrationCertificateV1,
        MEASUREMENT_EVIDENCE_SCHEMA_V1, MeasurementEvidenceV1, MeasurementQualificationPolicyV1,
        MeasurementUncertaintyComponentV1, QUALIFICATION_POLICY_SCHEMA_V1,
        TraceabilityDeclarationV1, UncertaintyComponentKindV1, qualify_measurement_evidence_v1,
    };
    use crate::benchmark::process_event_graph::{ClockBasisV1, MetrologyLabelStatusV1};

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

    fn input() -> CalibrationRequalificationInputV1 {
        let measured = measurement();
        let calibrated = certificate();
        let qualification_policy = policy();
        let qualification =
            qualify_measurement_evidence_v1(&measured, Some(&calibrated), &qualification_policy)
                .unwrap();
        CalibrationRequalificationInputV1 {
            schema: CALIBRATION_REQUALIFICATION_INPUT_SCHEMA_V1.into(),
            input_id: "label-consumption-input-001".into(),
            measurement: measured,
            calibration_at_k0: calibrated,
            qualification_policy,
            original_qualification_at_k0: qualification,
            measurement_time_bounds: MeasurementTimeBoundsV1 {
                earliest_measured_at_unix_seconds: 1_850_000_000,
                latest_measured_at_unix_seconds: 1_850_000_000,
            },
            k0_unix_seconds: 1_860_000_000,
            k1_unix_seconds: 1_880_000_000,
            owner_event_inventory_sha256: seal_calibration_event_inventory_v1(&[]).unwrap(),
            owner_events: Vec::new(),
            k0: CalibrationEventFrontierV1 {
                inventory_state: CalibrationEventInventoryStateV1::CompleteFrozen,
                ordering_state: CalibrationEventOrderingStateV1::Total,
                materialized_event_ids: Vec::new(),
            },
            k1: CalibrationEventFrontierV1 {
                inventory_state: CalibrationEventInventoryStateV1::CompleteFrozen,
                ordering_state: CalibrationEventOrderingStateV1::Total,
                materialized_event_ids: Vec::new(),
            },
        }
    }

    fn label() -> MetrologyLabelV1 {
        MetrologyLabelV1 {
            label_id: "label-001".into(),
            measurement_event_id: "measurement-event-001".into(),
            subject_id: "wafer-001".into(),
            measurand_id: "film-thickness".into(),
            value: Some(12.45),
            unit: "nm".into(),
            standard_uncertainty: Some(0.09),
            censoring_lower_bound: None,
            censoring_upper_bound: None,
            censoring_basis: None,
            status: MetrologyLabelStatusV1::Validated,
            measurement_time: "2026-11-20T10:00:00Z".into(),
            measurement_clock_domain_id: "fab-clock".into(),
            measurement_clock_basis: ClockBasisV1::OffsetAware,
            available_at: "2026-11-20T10:05:00Z".into(),
            availability_clock_domain_id: "fab-clock".into(),
            availability_clock_basis: ClockBasisV1::OffsetAware,
            instrument_id: Some("ellipsometer-07".into()),
            calibration_ref: Some("cal-cert-001".into()),
            revision: 1,
            supersedes_label_id: None,
            source_artifact_sha256: "f".repeat(64),
            authority_ref: "metrology-owner-01".into(),
        }
    }

    fn revocation_event(event_id: &str) -> CalibrationChangeEventV1 {
        CalibrationChangeEventV1 {
            schema: CALIBRATION_CHANGE_EVENT_SCHEMA_V1.into(),
            event_id: event_id.into(),
            event_kind: CalibrationChangeKindV1::Revoked,
            source_id: "metrology-lab-01".into(),
            source_payload_sha256: "9".repeat(64),
            authority_ref: "calibration-owner-01".into(),
            source_sequence: Some(1),
            effective_at_unix_seconds: 1_840_000_000,
            observed_at_unix_seconds: 1_870_000_000,
            qualification: CalibrationEventQualificationV1::Qualified,
            scope: Some(CalibrationChangeScopeV1 {
                target: CalibrationChangeTargetV1::Certificate {
                    certificate_id: "cal-cert-001".into(),
                    certificate_sha256: "a".repeat(64),
                },
                instrument_id: "ellipsometer-07".into(),
                measurand_id: "film-thickness".into(),
                unit: "nm".into(),
                method_revision: "film-method-3".into(),
                operating_range: CalibrationOperatingRangeV1::EntireDeclaredCertificateRange,
            }),
        }
    }

    fn assert_no_authority(report: &LabelConsumptionEligibilityReportV1) {
        assert!(!report.source_authenticity_proven);
        assert!(!report.training_snapshot_mutated);
        assert!(!report.equipment_control_authority);
    }

    #[test]
    fn validated_label_is_consumable_at_both_frontiers_under_declared_policy() {
        let input = input();
        let label = label();
        let k0 = assess_label_consumption_eligibility_v1(
            &label,
            &input,
            LabelEvidenceFrontierV1::AsKnownAtK0,
        )
        .unwrap();
        let k1 = assess_label_consumption_eligibility_v1(
            &label,
            &input,
            LabelEvidenceFrontierV1::CurrentAtK1,
        )
        .unwrap();

        assert_eq!(
            k0.eligibility,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
        );
        assert_eq!(
            k1.eligibility,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
        );
        assert!(k0.downstream_consumption_allowed);
        assert!(k1.downstream_consumption_allowed);
        assert_no_authority(&k0);
        assert_no_authority(&k1);
    }

    #[test]
    fn eligibility_receipt_attests_exact_label_revision() {
        let input = input();
        let mut candidate = label();
        candidate.revision = 7;
        let report = assess_label_consumption_eligibility_v1(
            &candidate,
            &input,
            LabelEvidenceFrontierV1::CurrentAtK1,
        )
        .unwrap();
        assert_eq!(report.label_revision, 7);
    }

    #[test]
    fn validated_label_point_value_must_match_qualified_measurement() {
        let input = input();
        let mut candidate = label();
        candidate.value = Some(12.46);
        let report = assess_label_consumption_eligibility_v1(
            &candidate,
            &input,
            LabelEvidenceFrontierV1::CurrentAtK1,
        )
        .unwrap();
        assert_eq!(report.eligibility, LabelConsumptionEligibilityV1::Blocked);
        assert!(
            report
                .reasons
                .contains(&LabelConsumptionReasonV1::LabelValueMismatch)
        );
        assert!(!report.downstream_consumption_allowed);
    }

    #[test]
    fn validated_label_requires_a_point_value_and_qualified_uncertainty() {
        let input = input();
        let mut missing_value = label();
        missing_value.value = None;
        let value_report = assess_label_consumption_eligibility_v1(
            &missing_value,
            &input,
            LabelEvidenceFrontierV1::CurrentAtK1,
        )
        .unwrap();
        assert_eq!(
            value_report.eligibility,
            LabelConsumptionEligibilityV1::Blocked
        );
        assert!(
            value_report
                .reasons
                .contains(&LabelConsumptionReasonV1::LabelValueMissing)
        );

        let mut missing_uncertainty = label();
        missing_uncertainty.standard_uncertainty = None;
        let uncertainty_report = assess_label_consumption_eligibility_v1(
            &missing_uncertainty,
            &input,
            LabelEvidenceFrontierV1::CurrentAtK1,
        )
        .unwrap();
        assert_eq!(
            uncertainty_report.eligibility,
            LabelConsumptionEligibilityV1::Unknown
        );
        assert!(
            uncertainty_report
                .reasons
                .contains(&LabelConsumptionReasonV1::LabelUncertaintyMissing)
        );
    }

    #[test]
    fn label_uncertainty_may_not_understate_the_qualified_uncertainty_bound() {
        let input = input();
        let mut candidate = label();
        candidate.standard_uncertainty = Some(0.001);
        let report = assess_label_consumption_eligibility_v1(
            &candidate,
            &input,
            LabelEvidenceFrontierV1::CurrentAtK1,
        )
        .unwrap();
        assert_eq!(report.eligibility, LabelConsumptionEligibilityV1::Blocked);
        assert!(
            report
                .reasons
                .contains(&LabelConsumptionReasonV1::LabelUncertaintyUnderstatesQualifiedBound)
        );
    }

    #[test]
    fn numeric_label_edits_change_the_label_and_eligibility_receipt_digests() {
        let input = input();
        let original = label();
        let original_report = assess_label_consumption_eligibility_v1(
            &original,
            &input,
            LabelEvidenceFrontierV1::CurrentAtK1,
        )
        .unwrap();
        let mut changed = original;
        changed.value = Some(12.46);
        let changed_report = assess_label_consumption_eligibility_v1(
            &changed,
            &input,
            LabelEvidenceFrontierV1::CurrentAtK1,
        )
        .unwrap();
        assert_ne!(original_report.label_sha256, changed_report.label_sha256);
        assert_ne!(original_report.report_sha256, changed_report.report_sha256);

        let mut uncertainty_changed = label();
        uncertainty_changed.standard_uncertainty = Some(0.10);
        let uncertainty_report = assess_label_consumption_eligibility_v1(
            &uncertainty_changed,
            &input,
            LabelEvidenceFrontierV1::CurrentAtK1,
        )
        .unwrap();
        assert_ne!(
            original_report.label_sha256,
            uncertainty_report.label_sha256
        );
        assert_ne!(
            original_report.report_sha256,
            uncertainty_report.report_sha256
        );
    }

    #[test]
    fn non_finite_label_numeric_metadata_is_rejected_before_receipt() {
        let input = input();
        let mut value = label();
        value.value = Some(f64::NAN);
        assert!(
            assess_label_consumption_eligibility_v1(
                &value,
                &input,
                LabelEvidenceFrontierV1::CurrentAtK1
            )
            .is_err()
        );

        let mut uncertainty = label();
        uncertainty.standard_uncertainty = Some(f64::INFINITY);
        assert!(
            assess_label_consumption_eligibility_v1(
                &uncertainty,
                &input,
                LabelEvidenceFrontierV1::CurrentAtK1
            )
            .is_err()
        );
    }

    #[test]
    fn provisional_and_censored_labels_are_not_promoted_as_point_labels() {
        let input = input();
        for status in [
            MetrologyLabelStatusV1::Provisional,
            MetrologyLabelStatusV1::Censored,
        ] {
            let mut candidate = label();
            candidate.status = status;
            candidate.value = None;
            let report = assess_label_consumption_eligibility_v1(
                &candidate,
                &input,
                LabelEvidenceFrontierV1::CurrentAtK1,
            )
            .unwrap();
            assert_eq!(report.eligibility, LabelConsumptionEligibilityV1::Unknown);
            assert!(!report.downstream_consumption_allowed);
        }
    }

    #[test]
    fn missing_instrument_or_calibration_reference_is_unknown() {
        let input = input();
        let mut candidate = label();
        candidate.instrument_id = None;
        candidate.calibration_ref = None;
        let report = assess_label_consumption_eligibility_v1(
            &candidate,
            &input,
            LabelEvidenceFrontierV1::CurrentAtK1,
        )
        .unwrap();

        assert_eq!(report.eligibility, LabelConsumptionEligibilityV1::Unknown);
        assert!(
            report
                .reasons
                .contains(&LabelConsumptionReasonV1::LabelInstrumentMissing)
        );
        assert!(
            report
                .reasons
                .contains(&LabelConsumptionReasonV1::LabelCalibrationReferenceMissing)
        );
    }

    #[test]
    fn mismatched_measurement_or_certificate_reference_is_blocked() {
        let input = input();
        let mut candidate = label();
        candidate.measurement_event_id = "different-measurement".into();
        candidate.calibration_ref = Some("other-calibration".into());
        let report = assess_label_consumption_eligibility_v1(
            &candidate,
            &input,
            LabelEvidenceFrontierV1::CurrentAtK1,
        )
        .unwrap();

        assert_eq!(report.eligibility, LabelConsumptionEligibilityV1::Blocked);
        assert!(
            report
                .reasons
                .contains(&LabelConsumptionReasonV1::MeasurementEventMismatch)
        );
        assert!(
            report
                .reasons
                .contains(&LabelConsumptionReasonV1::LabelCalibrationReferenceMismatch)
        );
    }

    #[test]
    fn instrument_measurand_and_unit_must_match_physical_measurement() {
        let input = input();
        let mut candidate = label();
        candidate.instrument_id = Some("other-gauge".into());
        candidate.measurand_id = "etch-depth".into();
        candidate.unit = "um".into();
        let report = assess_label_consumption_eligibility_v1(
            &candidate,
            &input,
            LabelEvidenceFrontierV1::CurrentAtK1,
        )
        .unwrap();

        assert_eq!(report.eligibility, LabelConsumptionEligibilityV1::Blocked);
        assert!(
            report
                .reasons
                .contains(&LabelConsumptionReasonV1::LabelInstrumentMismatch)
        );
        assert!(
            report
                .reasons
                .contains(&LabelConsumptionReasonV1::LabelMeasurandMismatch)
        );
        assert!(
            report
                .reasons
                .contains(&LabelConsumptionReasonV1::LabelUnitMismatch)
        );
    }

    #[test]
    fn invalid_label_is_blocked_and_unknown_label_stays_unknown() {
        let input = input();
        let mut invalid = label();
        invalid.status = MetrologyLabelStatusV1::Invalid;
        let invalid_report = assess_label_consumption_eligibility_v1(
            &invalid,
            &input,
            LabelEvidenceFrontierV1::CurrentAtK1,
        )
        .unwrap();
        assert_eq!(
            invalid_report.eligibility,
            LabelConsumptionEligibilityV1::Blocked
        );

        let mut unknown = label();
        unknown.status = MetrologyLabelStatusV1::Unknown;
        let unknown_report = assess_label_consumption_eligibility_v1(
            &unknown,
            &input,
            LabelEvidenceFrontierV1::CurrentAtK1,
        )
        .unwrap();
        assert_eq!(
            unknown_report.eligibility,
            LabelConsumptionEligibilityV1::Unknown
        );
    }

    #[test]
    fn retroactive_revocation_preserves_k0_but_blocks_current_k1_consumption() {
        let mut input = input();
        let event = revocation_event("cal-revoke-001");
        input.owner_event_inventory_sha256 =
            seal_calibration_event_inventory_v1(std::slice::from_ref(&event)).unwrap();
        input.owner_events = vec![event];
        input.k1.materialized_event_ids = vec!["cal-revoke-001".into()];

        let label = label();
        let k0 = assess_label_consumption_eligibility_v1(
            &label,
            &input,
            LabelEvidenceFrontierV1::AsKnownAtK0,
        )
        .unwrap();
        let k1 = assess_label_consumption_eligibility_v1(
            &label,
            &input,
            LabelEvidenceFrontierV1::CurrentAtK1,
        )
        .unwrap();

        assert_eq!(
            k0.eligibility,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
        );
        assert_eq!(k1.eligibility, LabelConsumptionEligibilityV1::Blocked);
        assert!(!k1.downstream_consumption_allowed);
    }

    #[test]
    fn relevant_but_unmaterialized_revocation_blocks_current_use_as_unknown() {
        let mut input = input();
        let event = revocation_event("cal-revoke-unmaterialized");
        input.owner_event_inventory_sha256 =
            seal_calibration_event_inventory_v1(std::slice::from_ref(&event)).unwrap();
        input.owner_events = vec![event];

        let report = assess_label_consumption_eligibility_v1(
            &label(),
            &input,
            LabelEvidenceFrontierV1::CurrentAtK1,
        )
        .unwrap();

        assert_eq!(report.eligibility, LabelConsumptionEligibilityV1::Unknown);
        assert!(
            report
                .reasons
                .contains(&LabelConsumptionReasonV1::RelevantCalibrationEventNotMaterialized)
        );
        assert!(!report.downstream_consumption_allowed);
    }

    #[test]
    fn stale_original_qualification_receipt_is_rejected() {
        let mut input = input();
        input.original_qualification_at_k0.report_sha256 = "0".repeat(64);
        assert!(
            assess_label_consumption_eligibility_v1(
                &label(),
                &input,
                LabelEvidenceFrontierV1::CurrentAtK1
            )
            .is_err()
        );
    }

    #[test]
    fn identical_inputs_produce_identical_eligibility_receipts() {
        let input = input();
        let label = label();
        let first = assess_label_consumption_eligibility_v1(
            &label,
            &input,
            LabelEvidenceFrontierV1::CurrentAtK1,
        )
        .unwrap();
        let second = assess_label_consumption_eligibility_v1(
            &label,
            &input,
            LabelEvidenceFrontierV1::CurrentAtK1,
        )
        .unwrap();
        assert_eq!(first.report_sha256, second.report_sha256);
    }
}
